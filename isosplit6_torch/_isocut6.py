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

"""Isocut6 — Hartigan dip-statistic test for unimodality.

Faithful port of isocut6.cpp.  Heavy vectorised work (sort, spacings,
exp/log, cumsum, KS statistic) stays on-device via PyTorch ops.  Only
the PAVA step drops to CPU lists (see _jisotonic5.py).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from ._jisotonic5 import jisotonic5_updown, jisotonic5_downup


def isocut6(samples: torch.Tensor) -> Tuple[float, float]:
    """Compute the dip score and optimal cut-point for 1-D *samples*.

    Parameters
    ----------
    samples : torch.Tensor
        1-D tensor of shape ``(N,)``.

    Returns
    -------
    dipscore : float
        Hartigan dip statistic.  Values < ``isocut_threshold`` (default 2.0)
        indicate the distribution is consistent with unimodality.
    cutpoint : float
        Optimal split point between the two modes (meaningful only when
        *dipscore* exceeds the threshold).
    """
    N = samples.shape[0]
    if N < 2:
        return 0.0, 0.0

    X = torch.sort(samples).values  # GPU sort

    # --- spacings, multiplicities, log-densities (vectorised) ---
    spacings = X[1:] - X[:-1]  # (N-1,)
    multiplicities_val = 1.0    # isocut6 uses constant multiplicity = 1

    # Guard against zero spacings
    safe_spacings = torch.where(
        spacings > 0, spacings, torch.tensor(1e-9, dtype=spacings.dtype, device=spacings.device)
    )
    log_densities = torch.log(torch.tensor(multiplicities_val, dtype=X.dtype, device=X.device) / safe_spacings)

    # --- unimodal fit via PAVA (CPU round-trip on small array) ---
    ld_list = log_densities.tolist()
    mult_list = [1.0] * (N - 1)
    log_densities_unimodal_fit_list = jisotonic5_updown(ld_list, mult_list)
    log_densities_unimodal_fit = torch.tensor(
        log_densities_unimodal_fit_list, dtype=X.dtype, device=X.device
    )

    # densities_unimodal_fit * spacings
    dens_fit_times_spacings = torch.exp(log_densities_unimodal_fit) * spacings

    # --- KS5 statistic with critical range ---
    peak_index = int(torch.argmax(log_densities_unimodal_fit).item())
    dipscore, cr_min, cr_max = _compute_ks5(
        mult_list, dens_fit_times_spacings, peak_index
    )

    # --- residuals on the critical range ---
    log_densities_resid = log_densities - log_densities_unimodal_fit
    cr_len = cr_max - cr_min + 1
    resid_cr = log_densities_resid[cr_min: cr_max + 1].tolist()
    weights_cr = [1.0] * cr_len

    resid_fit_cr = jisotonic5_downup(resid_cr, weights_cr)

    # cut-point: location of minimum of the down-up fit
    min_idx = _argmin_list(resid_fit_cr)
    cutpoint_idx = cr_min + min_idx
    cutpoint = (X[cutpoint_idx].item() + X[cutpoint_idx + 1].item()) / 2.0

    return dipscore, cutpoint


# ---------------------------------------------------------------------------
# KS5 statistic  (port of ns_isocut5::compute_ks5)
# ---------------------------------------------------------------------------

def _compute_ks4(counts1: list, counts2: torch.Tensor) -> float:
    """KS-like statistic between two discrete distributions."""
    N = len(counts1)
    if N == 0:
        return 0.0

    c2 = counts2.tolist()

    sum1 = sum(counts1)
    sum2 = sum(c2)
    if sum1 == 0 or sum2 == 0:
        return 0.0

    cumsum1 = 0.0
    cumsum2 = 0.0
    max_diff = 0.0
    for i in range(N):
        cumsum1 += counts1[i]
        cumsum2 += c2[i]
        diff = abs(cumsum1 / sum1 - cumsum2 / sum2)
        if diff > max_diff:
            max_diff = diff

    return max_diff * math.sqrt((sum1 + sum2) / 2.0)


def _compute_ks5(
    counts1: list,
    counts2: torch.Tensor,
    peak_index: int,
) -> Tuple[float, int, int]:
    """Compute KS5 with critical range detection around the peak."""
    N = len(counts1)
    cr_min = 0
    cr_max = N - 1
    ks_best = -1.0

    c2_list = counts2.tolist()

    # From the left
    length = peak_index + 1
    while length >= 4 or length == peak_index + 1:
        c1_sub = counts1[:length]
        c2_sub_t = counts2[:length]
        ks0 = _compute_ks4(c1_sub, c2_sub_t)
        if ks0 > ks_best:
            cr_min = 0
            cr_max = length - 1
            ks_best = ks0
        length = length // 2

    # From the right
    length = N - peak_index
    c1_rev = counts1[::-1]
    c2_rev = c2_list[::-1]
    while length >= 4 or length == N - peak_index:
        c1_sub = c1_rev[:length]
        c2_sub_t = counts2.flip(0)[:length]
        ks0 = _compute_ks4(c1_sub, c2_sub_t)
        if ks0 > ks_best:
            cr_min = N - length
            cr_max = N - 1
            ks_best = ks0
        length = length // 2

    return ks_best, cr_min, cr_max


def _argmin_list(x: list) -> int:
    best = 0
    for i in range(1, len(x)):
        if x[i] < x[best]:
            best = i
    return best
