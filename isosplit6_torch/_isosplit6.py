# Copyright 2016-2017 Flatiron Institute, Simons Foundation
# PyTorch port Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core Isosplit6 clustering algorithm — pure PyTorch.

Faithful port of isosplit6.cpp + supporting helpers from isosplit5.cpp.
All bulk data (X, labels, centroids, covariance matrices) stays on the
input device.  Only tiny scalars (dip scores, cluster counts) are synced
to CPU for control-flow decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from ._isocut6 import isocut6
from ._parcelate import parcelate2


@dataclass
class Isosplit6Opts:
    isocut_threshold: float = 2.0
    min_cluster_size: int = 10
    K_init: int = 200
    max_iterations_per_pass: int = 500


def isosplit6_run(
    X: torch.Tensor,
    *,
    initial_labels: Optional[torch.Tensor] = None,
    opts: Optional[Isosplit6Opts] = None,
) -> torch.Tensor:
    """Run isosplit6 clustering.

    Parameters
    ----------
    X : torch.Tensor
        Data matrix of shape ``(N, M)`` on any device.
    initial_labels : torch.Tensor | None
        Optional pre-computed labels (1-based int32).  If ``None``,
        ``parcelate2`` is used for initialisation.
    opts : Isosplit6Opts | None
        Algorithm parameters.

    Returns
    -------
    labels : torch.Tensor
        Cluster assignments in ``[1 .. K]``, dtype ``int32``, same device
        as *X*.
    """
    if opts is None:
        opts = Isosplit6Opts()

    N, M = X.shape
    device = X.device

    if N == 0:
        return torch.zeros(0, dtype=torch.int32, device=device)

    # ------------------------------------------------------------------
    # 1. Initialise labels
    # ------------------------------------------------------------------
    if initial_labels is not None:
        labels = initial_labels.clone().to(dtype=torch.int32, device=device)
    else:
        labels = parcelate2(X, target_parcel_size=opts.min_cluster_size,
                            target_num_parcels=opts.K_init)

    Kmax = int(labels.max().item())
    if Kmax == 0:
        return labels

    # ------------------------------------------------------------------
    # 2. Compute initial centroids and covariance matrices
    # ------------------------------------------------------------------
    compute_mask = torch.ones(Kmax, dtype=torch.bool, device=device)
    centroids = _compute_centroids(X, labels, Kmax, compute_mask)
    covmats = _compute_covmats(X, labels, Kmax, centroids, compute_mask)

    # Active labels and comparison tracking
    active_labels_vec = torch.zeros(Kmax, dtype=torch.bool, device=device)
    for k in range(1, Kmax + 1):
        active_labels_vec[k - 1] = True
    active_labels = list(range(1, Kmax + 1))

    comparisons_made = torch.zeros(Kmax, Kmax, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    # 3. Main loop: passes
    # ------------------------------------------------------------------
    final_pass = False
    while True:
        something_merged = False
        clusters_changed_in_pass = torch.zeros(Kmax, dtype=torch.bool, device=device)
        iteration_number = 0

        # --- inner loop: iterations within a pass ---
        while True:
            clusters_changed_in_iter = torch.zeros(Kmax, dtype=torch.bool, device=device)
            iteration_number += 1
            if iteration_number > opts.max_iterations_per_pass:
                break

            if len(active_labels) == 0:
                break

            # Get pairs to compare (mutual nearest neighbours)
            K_active = len(active_labels)
            active_centroids = centroids[torch.tensor([k - 1 for k in active_labels],
                                                       dtype=torch.long, device=device)]
            active_cm = comparisons_made[
                torch.tensor([k - 1 for k in active_labels], dtype=torch.long, device=device)
            ][:, torch.tensor([k - 1 for k in active_labels], dtype=torch.long, device=device)]

            pairs = _get_pairs_to_compare(active_centroids, active_cm)
            if len(pairs) == 0:
                break

            # Remap to original label space
            pairs_orig = [(active_labels[a], active_labels[b]) for a, b in pairs]

            # Compare each pair
            for k1, k2 in pairs_orig:
                inds1 = (labels == k1).nonzero(as_tuple=True)[0]
                inds2 = (labels == k2).nonzero(as_tuple=True)[0]
                if inds1.shape[0] == 0 or inds2.shape[0] == 0:
                    comparisons_made[k1 - 1, k2 - 1] = True
                    comparisons_made[k2 - 1, k1 - 1] = True
                    continue

                do_merge: bool
                if inds1.shape[0] < opts.min_cluster_size or inds2.shape[0] < opts.min_cluster_size:
                    do_merge = True
                    L12 = torch.ones(inds1.shape[0] + inds2.shape[0],
                                     dtype=torch.int32, device=device)
                else:
                    do_merge, L12 = _merge_test(
                        X, inds1, inds2,
                        centroids[k1 - 1], centroids[k2 - 1],
                        covmats[k1 - 1], covmats[k2 - 1],
                        opts.isocut_threshold,
                    )

                comparisons_made[k1 - 1, k2 - 1] = True
                comparisons_made[k2 - 1, k1 - 1] = True

                if do_merge:
                    labels[inds2] = k1
                    clusters_changed_in_iter[k1 - 1] = True
                    clusters_changed_in_iter[k2 - 1] = True
                else:
                    # Redistribute
                    N1 = inds1.shape[0]
                    L12_1 = L12[:N1]
                    L12_2 = L12[N1:]
                    reassigned = False
                    move_to_k2 = inds1[L12_1 == 2]
                    if move_to_k2.shape[0] > 0:
                        labels[move_to_k2] = k2
                        reassigned = True
                    move_to_k1 = inds2[L12_2 == 1]
                    if move_to_k1.shape[0] > 0:
                        labels[move_to_k1] = k1
                        reassigned = True
                    if reassigned:
                        clusters_changed_in_iter[k1 - 1] = True
                        clusters_changed_in_iter[k2 - 1] = True

            # Update centroids / covmats for changed clusters
            clusters_changed_in_pass |= clusters_changed_in_iter
            if clusters_changed_in_iter.any():
                centroids = _compute_centroids(X, labels, Kmax, clusters_changed_in_iter, centroids)
                covmats = _compute_covmats(X, labels, Kmax, centroids, clusters_changed_in_iter, covmats)

            # Refresh active labels
            new_active_vec = torch.zeros(Kmax, dtype=torch.bool, device=device)
            new_active_vec.scatter_(0, (labels - 1).long(), True)
            new_active_labels = (new_active_vec.nonzero(as_tuple=True)[0] + 1).tolist()
            if len(new_active_labels) < len(active_labels):
                something_merged = True
            active_labels = new_active_labels

        # Reset comparisons for changed clusters
        for k0 in range(Kmax):
            if clusters_changed_in_pass[k0]:
                comparisons_made[k0, :] = False
                comparisons_made[:, k0] = False

        if something_merged:
            final_pass = False
        if final_pass:
            break
        if not something_merged:
            final_pass = True

    # ------------------------------------------------------------------
    # 4. Remap labels to contiguous 1..K
    # ------------------------------------------------------------------
    labels = _remap_labels(labels, active_labels)
    return labels


# ======================================================================
# Helpers
# ======================================================================

def _compute_centroids(
    X: torch.Tensor,
    labels: torch.Tensor,
    Kmax: int,
    compute_mask: torch.Tensor,
    existing: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute cluster centroids using scatter_add."""
    N, M = X.shape
    device = X.device
    if existing is not None:
        centroids = existing.clone()
    else:
        centroids = torch.zeros(Kmax, M, dtype=X.dtype, device=device)

    # Which clusters to recompute
    ks = compute_mask.nonzero(as_tuple=True)[0]  # 0-based
    if ks.shape[0] == 0:
        return centroids

    # For each cluster in ks, accumulate via scatter
    lab = (labels - 1).long()  # 0-based
    sums = torch.zeros(Kmax, M, dtype=X.dtype, device=device)
    counts = torch.zeros(Kmax, dtype=X.dtype, device=device)
    sums.scatter_add_(0, lab.unsqueeze(1).expand_as(X), X)
    counts.scatter_add_(0, lab, torch.ones(N, dtype=X.dtype, device=device))

    for k in ks.tolist():
        if counts[k] > 0:
            centroids[k] = sums[k] / counts[k]
        else:
            centroids[k] = 0

    return centroids


def _compute_covmats(
    X: torch.Tensor,
    labels: torch.Tensor,
    Kmax: int,
    centroids: torch.Tensor,
    compute_mask: torch.Tensor,
    existing: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-cluster covariance matrices."""
    N, M = X.shape
    device = X.device
    if existing is not None:
        covmats = existing.clone()
    else:
        covmats = torch.zeros(Kmax, M, M, dtype=X.dtype, device=device)

    ks = compute_mask.nonzero(as_tuple=True)[0]
    if ks.shape[0] == 0:
        return covmats

    lab = (labels - 1).long()

    for k in ks.tolist():
        mask = lab == k
        nk = mask.sum().item()
        if nk == 0:
            covmats[k] = 0
            continue
        Xk = X[mask]  # (nk, M)
        diff = Xk - centroids[k].unsqueeze(0)
        covmats[k] = (diff.T @ diff) / nk

    return covmats


def _get_pairs_to_compare(
    centroids: torch.Tensor,
    comparisons_made: torch.Tensor,
) -> List[Tuple[int, int]]:
    """Find mutual-nearest-neighbour pairs that haven't been compared yet.

    Parameters
    ----------
    centroids : (K, M)
    comparisons_made : (K, K) bool

    Returns
    -------
    List of (i, j) pairs — 0-based indices into the active centroid array.
    """
    K = centroids.shape[0]
    if K < 2:
        return []

    # Pairwise distance matrix
    dists = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)

    # Mask out already-compared pairs and self-distances
    mask_out = comparisons_made | torch.eye(K, dtype=torch.bool, device=centroids.device)
    dists[mask_out] = float('inf')

    # For each cluster, find its nearest uncompared neighbour
    best_inds = torch.argmin(dists, dim=1)  # (K,)

    # Collect mutual nearest pairs
    pairs: List[Tuple[int, int]] = []
    used = set()
    best_list = best_inds.tolist()
    for j in range(K):
        bj = best_list[j]
        if dists[j, bj].item() == float('inf'):
            continue
        if bj > j and best_list[bj] == j:  # mutual
            if j not in used and bj not in used:
                pairs.append((j, bj))
                used.add(j)
                used.add(bj)

    return pairs


def _merge_test(
    X: torch.Tensor,
    inds1: torch.Tensor,
    inds2: torch.Tensor,
    centroid1: torch.Tensor,
    centroid2: torch.Tensor,
    covmat1: torch.Tensor,
    covmat2: torch.Tensor,
    isocut_threshold: float,
) -> Tuple[bool, torch.Tensor]:
    """Test whether two clusters should be merged.

    Returns (do_merge, L12) where L12 is a label vector (1 or 2) for
    the concatenated points [inds1; inds2].
    """
    M = X.shape[1]
    device = X.device
    N1 = inds1.shape[0]
    N2 = inds2.shape[0]
    N12 = N1 + N2

    L12 = torch.ones(N12, dtype=torch.int32, device=device)

    # Direction vector: centroid2 - centroid1
    V = centroid2 - centroid1

    # Mahalanobis direction: inv(avg_covmat) @ V
    avg_cov = (covmat1 + covmat2) / 2.0
    try:
        inv_avg_cov = torch.linalg.inv(avg_cov)
    except torch.linalg.LinAlgError:
        # Singular matrix — add regularisation
        inv_avg_cov = torch.linalg.inv(
            avg_cov + 1e-6 * torch.eye(M, dtype=avg_cov.dtype, device=device)
        )

    V = inv_avg_cov @ V

    # Normalise
    norm_sq = (V * V).sum()
    if norm_sq.item() > 0:
        V = V / torch.sqrt(norm_sq)

    # Project both clusters onto V
    X1 = X[inds1]  # (N1, M)
    X2 = X[inds2]  # (N2, M)
    proj1 = X1 @ V  # (N1,)
    proj2 = X2 @ V  # (N2,)
    proj12 = torch.cat([proj1, proj2])  # (N12,)

    # Run isocut6 on projected data
    dipscore, cutpoint = isocut6(proj12)

    do_merge = dipscore < isocut_threshold

    # Assign labels based on cutpoint regardless of merge decision
    L12 = torch.where(proj12 < cutpoint,
                       torch.tensor(1, dtype=torch.int32, device=device),
                       torch.tensor(2, dtype=torch.int32, device=device))

    return do_merge, L12


def _remap_labels(labels: torch.Tensor, active_labels: List[int]) -> torch.Tensor:
    """Remap active labels to contiguous 1..K."""
    device = labels.device
    if len(active_labels) == 0:
        return labels.clone()

    Kmax = int(labels.max().item())
    label_map = torch.zeros(Kmax + 1, dtype=torch.int32, device=device)
    for i, k in enumerate(active_labels):
        label_map[k] = i + 1

    return label_map[labels.long()]
