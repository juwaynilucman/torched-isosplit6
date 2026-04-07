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

"""Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.

Faithful port of jisotonic5.cpp.  The PAVA is inherently sequential so we
run it in pure Python on small arrays (typically < 5 000 elements from
isocut6 spacing vectors).  Data is transferred to/from the GPU only at
the isocut6 boundary — see _isocut6.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Pure-Python PAVA (Tier 1 — always available)
# ---------------------------------------------------------------------------

def jisotonic5(
    AA: List[float],
    WW: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    """Non-decreasing isotonic regression via PAVA.

    Parameters
    ----------
    AA : list[float]
        Input values (length *N*).
    WW : list[float] | None
        Weights.  ``None`` => unit weights.

    Returns
    -------
    BB : list[float]
        Fitted (isotonic) values, same length as *AA*.
    MSE : list[float]
        Cumulative weighted MSE after processing each element.
    """
    N = len(AA)
    if N == 0:
        return [], []

    # Each "block" tracks: unweighted_count, count(=sum of weights),
    # weighted_sum, weighted_sum_of_squares
    uwc: List[int] = []
    cnt: List[float] = []
    s: List[float] = []
    ssq: List[float] = []
    MSE: List[float] = [0.0] * N

    w0 = WW[0] if WW is not None else 1.0
    uwc.append(1)
    cnt.append(w0)
    s.append(AA[0] * w0)
    ssq.append(AA[0] * AA[0] * w0)

    for j in range(1, N):
        w0 = WW[j] if WW is not None else 1.0
        uwc.append(1)
        cnt.append(w0)
        s.append(AA[j] * w0)
        ssq.append(AA[j] * AA[j] * w0)
        MSE[j] = MSE[j - 1]

        li = len(cnt) - 1  # last_index
        while li > 0:
            if s[li - 1] / cnt[li - 1] < s[li] / cnt[li]:
                break
            # Merge block li into li-1
            prev_mse = (
                ssq[li - 1] - s[li - 1] * s[li - 1] / cnt[li - 1]
                + ssq[li] - s[li] * s[li] / cnt[li]
            )
            uwc[li - 1] += uwc[li]
            cnt[li - 1] += cnt[li]
            s[li - 1] += s[li]
            ssq[li - 1] += ssq[li]
            new_mse = ssq[li - 1] - s[li - 1] * s[li - 1] / cnt[li - 1]
            MSE[j] += new_mse - prev_mse

            uwc.pop()
            cnt.pop()
            s.pop()
            ssq.pop()
            li -= 1

    # Expand blocks back to per-element fitted values
    BB: List[float] = [0.0] * N
    ii = 0
    for k in range(len(cnt)):
        val = s[k] / cnt[k]
        for _ in range(uwc[k]):
            BB[ii] = val
            ii += 1

    return BB, MSE


def jisotonic5_updown(
    in_vals: List[float],
    weights: Optional[List[float]] = None,
) -> List[float]:
    """Unimodal (non-decreasing then non-increasing) isotonic fit.

    Finds the optimal peak position that minimises combined MSE of the
    left-increasing and right-decreasing fits.
    """
    N = len(in_vals)
    if N == 0:
        return []

    # Forward (non-decreasing) fit on the full array
    B1, MSE1 = jisotonic5(in_vals, weights)

    # Backward (non-decreasing on reversed = non-increasing on original)
    in_rev = in_vals[::-1]
    w_rev = weights[::-1] if weights is not None else None
    B2, MSE2 = jisotonic5(in_rev, w_rev)

    # Combined MSE: forward up to j + backward from j
    combined = [MSE1[j] + MSE2[N - 1 - j] for j in range(N)]

    # Best split point (peak location)
    best_ind = 0
    best_val = combined[0]
    for j in range(1, N):
        if combined[j] < best_val:
            best_val = combined[j]
            best_ind = j

    # Re-fit on the two halves
    B1, _ = jisotonic5(in_vals[: best_ind + 1], weights[: best_ind + 1] if weights else None)
    B2, _ = jisotonic5(in_rev[: N - best_ind], w_rev[: N - best_ind] if w_rev else None)

    out = [0.0] * N
    for j in range(best_ind + 1):
        out[j] = B1[j]
    for j in range(N - best_ind - 1):
        out[N - 1 - j] = B2[j]

    return out


def jisotonic5_downup(
    in_vals: List[float],
    weights: Optional[List[float]] = None,
) -> List[float]:
    """Valley (non-increasing then non-decreasing) isotonic fit."""
    neg = [-v for v in in_vals]
    out = jisotonic5_updown(neg, weights)
    return [-v for v in out]


# ---------------------------------------------------------------------------
# Tensor convenience wrappers (transfer boundary)
# ---------------------------------------------------------------------------

def jisotonic5_updown_t(
    x: torch.Tensor,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wrapper: accepts and returns tensors, runs PAVA on CPU lists."""
    device = x.device
    out = jisotonic5_updown(x.tolist(), w.tolist() if w is not None else None)
    return torch.tensor(out, dtype=x.dtype, device=device)


def jisotonic5_downup_t(
    x: torch.Tensor,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wrapper: accepts and returns tensors, runs PAVA on CPU lists."""
    device = x.device
    out = jisotonic5_downup(x.tolist(), w.tolist() if w is not None else None)
    return torch.tensor(out, dtype=x.dtype, device=device)
