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

"""Parcelate2 — hierarchical spatial partitioning for isosplit6 init.

Faithful port of the parcelate2() function from isosplit5.cpp.
All heavy ops (distance computation, nearest-seed assignment) use
PyTorch tensor operations on-device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class _Parcel:
    indices: torch.Tensor          # 1-D LongTensor of point indices
    centroid: torch.Tensor         # (M,) float
    radius: float = 0.0


def parcelate2(
    X: torch.Tensor,
    target_parcel_size: int = 10,
    target_num_parcels: int = 200,
) -> torch.Tensor:
    """Create initial cluster parcels by hierarchical splitting.

    Parameters
    ----------
    X : torch.Tensor
        Data matrix, shape ``(N, M)`` — *row-major* (PyTorch convention).
    target_parcel_size, target_num_parcels : int
        Stopping criteria matching the C++ defaults.

    Returns
    -------
    labels : torch.Tensor
        ``int32`` labels in ``[1 .. K]``, shape ``(N,)``.
    """
    N, M = X.shape
    device = X.device

    labels = torch.ones(N, dtype=torch.int32, device=device)

    # Bootstrap: single parcel containing everything
    all_inds = torch.arange(N, dtype=torch.long, device=device)
    centroid = X.mean(dim=0)
    radius = _max_distance(centroid, X, all_inds)
    parcels: List[_Parcel] = [_Parcel(indices=all_inds, centroid=centroid, radius=radius)]

    split_factor = 3  # deterministic: first 3 points as seeds (matches C++)

    something_changed = True
    while len(parcels) < target_num_parcels and something_changed:
        something_changed = False

        # Find target_radius (0.95 × max radius among oversized parcels)
        candidate_found = False
        target_radius = 0.0
        for p in parcels:
            if p.indices.shape[0] > target_parcel_size and p.radius > 0:
                candidate_found = True
                r = p.radius * 0.95
                if r > target_radius:
                    target_radius = r
        if not candidate_found or target_radius == 0:
            break

        p_index = 0
        while p_index < len(parcels):
            p = parcels[p_index]
            inds = p.indices
            sz = inds.shape[0]
            rad = p.radius

            if sz > target_parcel_size and rad >= target_radius:
                # Pick first split_factor points as seeds (deterministic)
                n_seeds = min(split_factor, sz)
                seed_inds = inds[:n_seeds]  # indices into X
                seeds = X[seed_inds]        # (n_seeds, M)

                # Assign every point in this parcel to nearest seed
                pts = X[inds]               # (sz, M)
                # dists: (sz, n_seeds)
                dists = torch.cdist(pts.unsqueeze(0), seeds.unsqueeze(0)).squeeze(0)
                assignments = torch.argmin(dists, dim=1)  # (sz,) values in [0, n_seeds)

                # Rebuild parcel 0 (seed 0) in-place
                mask0 = assignments == 0
                new_inds0 = inds[mask0]
                if new_inds0.shape[0] == sz:
                    # Nothing split — advance to avoid infinite loop
                    p_index += 1
                    continue
                something_changed = True
                c0 = X[new_inds0].mean(dim=0)
                r0 = _max_distance(c0, X, new_inds0)
                parcels[p_index] = _Parcel(indices=new_inds0, centroid=c0, radius=r0)
                labels[new_inds0] = p_index + 1  # 1-based

                # Create new parcels for seeds 1..n_seeds-1
                for jj in range(1, n_seeds):
                    mask_j = assignments == jj
                    new_inds_j = inds[mask_j]
                    if new_inds_j.shape[0] == 0:
                        continue
                    cj = X[new_inds_j].mean(dim=0)
                    rj = _max_distance(cj, X, new_inds_j)
                    parcels.append(_Parcel(indices=new_inds_j, centroid=cj, radius=rj))
                    labels[new_inds_j] = len(parcels)  # 1-based
                # Don't advance p_index — re-examine the replaced parcel
            else:
                p_index += 1

    return labels


def _max_distance(centroid: torch.Tensor, X: torch.Tensor, inds: torch.Tensor) -> float:
    """Max Euclidean distance from *centroid* to points ``X[inds]``."""
    if inds.shape[0] == 0:
        return 0.0
    diff = X[inds] - centroid.unsqueeze(0)
    dists = torch.sqrt((diff * diff).sum(dim=1))
    return dists.max().item()
