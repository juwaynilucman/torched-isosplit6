# Copyright 2026
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

"""Lightweight PCA for the isosplit6 subdivision step.

This is a self-contained helper so that isosplit6_torch has no dependency
on the MountainSort5 compute_pca module.
"""

from __future__ import annotations

import torch


def compute_pca_features(
    X: torch.Tensor,
    *,
    npca: int,
    random_state: int = 0,
) -> torch.Tensor:
    """Compute PCA scores, staying entirely on-device.

    Parameters
    ----------
    X : torch.Tensor
        ``(L, D)`` data matrix.
    npca : int
        Number of components.

    Returns
    -------
    scores : torch.Tensor
        ``(L, k)`` float32 PCA scores where ``k = min(npca, L, D)``.
    """
    L, D = X.shape
    k = min(npca, L, D)
    dtype = torch.float32 if X.dtype not in (torch.float32, torch.float64) else X.dtype

    if L == 0 or D == 0 or k == 0:
        return torch.zeros((L, k), dtype=torch.float32, device=X.device)

    Xw = X.to(dtype)
    mean = Xw.mean(dim=0, keepdim=True)
    Xc = Xw - mean

    # Choose solver
    if D <= 1000 and L >= 10 * D:
        # Covariance + eigh — fast when D is small relative to L
        C = (Xc.T @ Xc) / max(L - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(C)
        Vt = torch.flip(eigvecs, dims=[1]).T[:k]
        scores = Xc @ Vt.T
    elif max(L, D) > 500 and k < int(0.8 * min(L, D)):
        # Randomised SVD via torch.pca_lowrank
        q = min(k + 10, min(L, D))
        if X.is_cuda:
            torch.cuda.manual_seed_all(random_state)
        torch.manual_seed(random_state)
        U, S, V = torch.pca_lowrank(Xc, q=q, center=False, niter=4)
        U = U[:, :k]
        S = S[:k]
        # Sign-flip for determinism
        idx = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[idx, torch.arange(k, device=U.device)])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        U = U * signs
        scores = U * S
    else:
        # Full SVD
        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        # Sign-flip
        idx = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[idx, torch.arange(U.shape[1], device=U.device)])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        U = U * signs
        scores = U[:, :k] * S[:k]

    return scores.to(torch.float32)
