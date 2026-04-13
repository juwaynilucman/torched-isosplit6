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

"""Isosplit6Clustering — drop-in nn.Module for the MountainSort5 pipeline.

Replaces the CPU-bound wrapper that called the C++ isosplit6 library,
sklearn PCA, and scipy hierarchical clustering.  Everything stays on the
input device.

Reads:  batch.features
Writes: batch.labels
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ._isosplit6 import isosplit6_run, Isosplit6Opts
from ._pca import compute_pca_features


class Isosplit6Clustering(nn.Module):
    """Cluster PCA features using the isosplit6 subdivision method.

    This is a **pure PyTorch** drop-in replacement for the original
    CPU-bound implementation.  No C++ extensions, no numpy, no scipy.

    Parameters
    ----------
    npca_per_subdivision : int
        Number of PCA components used for the local PCA inside each
        subdivision step.
    isocut_threshold : float
        Dip-score threshold for the merge test (default 2.0).
    min_cluster_size : int
        Clusters smaller than this are force-merged (default 10).
    K_init : int
        Target number of initial parcels (default 200).
    """

    def __init__(
        self,
        npca_per_subdivision: int = 10,
        isocut_threshold: float = 2.0,
        min_cluster_size: int = 10,
        K_init: int = 200,
    ):
        super().__init__()
        self.npca_per_subdivision = npca_per_subdivision
        self.opts = Isosplit6Opts(
            isocut_threshold=isocut_threshold,
            min_cluster_size=min_cluster_size,
            K_init=K_init,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run subdivision clustering on *features*.

        Parameters
        ----------
        features : torch.Tensor
            ``(N, D)`` feature matrix.

        Returns
        -------
        labels : torch.Tensor
            ``(N,)`` int32 cluster labels starting at 1.
        """
        labels = _isosplit6_subdivision_method(
            features,
            npca_per_subdivision=self.npca_per_subdivision,
            opts=self.opts,
        )
        return labels


# ======================================================================
# Subdivision method (replaces scipy linkage + recursive isosplit6)
# ======================================================================

def _isosplit6_subdivision_method(
    X: torch.Tensor,
    *,
    npca_per_subdivision: int,
    opts: Isosplit6Opts,
    inds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Recursive isosplit6 subdivision — fully on-device.

    Runs isosplit6 on a local PCA of the data, then splits the resulting
    clusters into two groups via single-linkage (MST cut) and recurses.
    """
    device = X.device

    if inds is not None:
        X_sub = X[inds]
    else:
        X_sub = X

    L = X_sub.shape[0]
    if L == 0:
        return torch.zeros(0, dtype=torch.int32, device=device)

    # Local PCA
    features = compute_pca_features(X_sub, npca=npca_per_subdivision)

    # Core isosplit6 on the reduced features
    labels = isosplit6_run(features, opts=opts)

    K = int(labels.max().item()) if labels.numel() > 0 else 0
    if K <= 1:
        return labels

    # --- Hierarchical split into two groups (replaces scipy linkage) ---
    centroids = torch.zeros(K, X.shape[1], dtype=X.dtype, device=device)
    for k in range(1, K + 1):
        mask = labels == k
        subset = X_sub[mask]
        if subset.is_cuda and torch.are_deterministic_algorithms_enabled():
            centroids[k - 1] = subset.cpu().median(dim=0).values.to(device)
        else:
            centroids[k - 1] = subset.median(dim=0).values

    cluster_inds_1, cluster_inds_2 = _single_linkage_split(centroids)

    # Map back to point indices
    mask1 = torch.zeros(L, dtype=torch.bool, device=device)
    mask2 = torch.zeros(L, dtype=torch.bool, device=device)
    for k in cluster_inds_1:
        mask1 |= labels == (k + 1)  # cluster_inds are 0-based, labels are 1-based
    for k in cluster_inds_2:
        mask2 |= labels == (k + 1)

    point_inds1 = mask1.nonzero(as_tuple=True)[0]
    point_inds2 = mask2.nonzero(as_tuple=True)[0]

    if inds is not None:
        global_inds1 = inds[point_inds1]
        global_inds2 = inds[point_inds2]
    else:
        global_inds1 = point_inds1
        global_inds2 = point_inds2

    labels1 = _isosplit6_subdivision_method(
        X, npca_per_subdivision=npca_per_subdivision, opts=opts, inds=global_inds1
    )
    labels2 = _isosplit6_subdivision_method(
        X, npca_per_subdivision=npca_per_subdivision, opts=opts, inds=global_inds2
    )

    K1 = int(labels1.max().item()) if labels1.numel() > 0 else 0
    ret_labels = torch.zeros(L, dtype=torch.int32, device=device)
    ret_labels[point_inds1] = labels1
    ret_labels[point_inds2] = labels2 + K1

    return ret_labels


# ======================================================================
# Single-linkage split (replaces scipy linkage + cut_tree)
# ======================================================================

def _single_linkage_split(
    centroids: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Split K centroids into 2 groups via MST cut (= single-linkage at k=2).

    Uses Prim's algorithm to build the MST, then removes the longest edge
    to produce exactly two connected components.
    """
    K = centroids.shape[0]
    device = centroids.device

    if K <= 1:
        return list(range(K)), []
    if K == 2:
        return [0], [1]

    # Full pairwise distances (K is small, typically < 50)
    dists = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)

    # Prim's MST
    in_tree = torch.zeros(K, dtype=torch.bool, device=device)
    in_tree[0] = True
    min_edge = dists[0].clone()
    min_edge[0] = float('inf')
    parent = torch.zeros(K, dtype=torch.long, device=device)

    mst_u: list[int] = []
    mst_v: list[int] = []
    mst_w: list[float] = []

    for _ in range(K - 1):
        # Find cheapest edge from tree to non-tree
        temp = min_edge.clone()
        temp[in_tree] = float('inf')
        v = int(temp.argmin().item())
        u = int(parent[v].item())
        mst_u.append(u)
        mst_v.append(v)
        mst_w.append(min_edge[v].item())
        in_tree[v] = True

        # Update min_edge with distances from new node
        new_dists = dists[v]
        better = new_dists < min_edge
        min_edge = torch.where(better, new_dists, min_edge)
        parent = torch.where(better, torch.tensor(v, device=device), parent)

    # Cut the longest MST edge
    longest_idx = max(range(len(mst_w)), key=lambda i: mst_w[i])

    # BFS from node 0 in MST without the cut edge to find component 1
    adj: dict[int, list[int]] = {i: [] for i in range(K)}
    for i in range(len(mst_u)):
        if i == longest_idx:
            continue
        adj[mst_u[i]].append(mst_v[i])
        adj[mst_v[i]].append(mst_u[i])

    visited = set()
    stack = [mst_u[longest_idx]]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                stack.append(nb)

    group1 = sorted(visited)
    group2 = sorted(set(range(K)) - visited)

    return group1, group2
