"""Tests for isosplit6_torch — correctness and parity with C++ isosplit6."""

from __future__ import annotations

import math

import pytest
import torch

from isosplit6_torch._jisotonic5 import jisotonic5, jisotonic5_updown, jisotonic5_downup
from isosplit6_torch._isocut6 import isocut6
from isosplit6_torch._parcelate import parcelate2
from isosplit6_torch._isosplit6 import isosplit6_run, Isosplit6Opts
from isosplit6_torch.clustering import Isosplit6Clustering


# ======================================================================
# Helpers
# ======================================================================

def _make_blobs(
    centers: list[list[float]],
    n_per_cluster: int = 500,
    std: float = 0.3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create well-separated Gaussian blobs."""
    torch.manual_seed(seed)
    parts = []
    labels = []
    for i, c in enumerate(centers):
        c_t = torch.tensor(c, dtype=torch.float32, device=device)
        pts = c_t + std * torch.randn(n_per_cluster, len(c), device=device)
        parts.append(pts)
        labels.append(torch.full((n_per_cluster,), i + 1, dtype=torch.int32, device=device))
    return torch.cat(parts), torch.cat(labels)


def _label_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Best-case label overlap (accounting for permutation)."""
    a_np = a.cpu()
    b_np = b.cpu()
    ka = int(a_np.max().item())
    kb = int(b_np.max().item())
    if ka == 0 or kb == 0:
        return 0.0

    # Build confusion matrix and find best alignment via greedy matching
    n = a_np.shape[0]
    best = 0
    from itertools import permutations
    # For small K, brute force. For large K, use Hungarian (not needed here).
    if kb <= 8:
        for perm in permutations(range(1, kb + 1)):
            mapped = torch.zeros_like(b_np)
            for old, new in enumerate(perm, 1):
                mapped[b_np == old] = new
            overlap = (a_np == mapped).sum().item()
            if overlap > best:
                best = overlap
    else:
        # Greedy matching
        remaining_b = set(range(1, kb + 1))
        used_a = set()
        total = 0
        for ka_i in range(1, ka + 1):
            mask_a = a_np == ka_i
            best_match = -1
            best_count = -1
            for kb_j in remaining_b:
                count = (mask_a & (b_np == kb_j)).sum().item()
                if count > best_count:
                    best_count = count
                    best_match = kb_j
            if best_match >= 0:
                total += best_count
                remaining_b.discard(best_match)
        best = total

    return best / n


# ======================================================================
# PAVA tests
# ======================================================================

class TestJisotonic5:
    def test_already_sorted(self):
        BB, MSE = jisotonic5([1.0, 2.0, 3.0, 4.0])
        assert BB == [1.0, 2.0, 3.0, 4.0]

    def test_single_violation(self):
        BB, MSE = jisotonic5([3.0, 1.0, 2.0, 4.0])
        assert BB[0] == BB[1] == pytest.approx(2.0)

    def test_constant(self):
        BB, MSE = jisotonic5([5.0, 5.0, 5.0])
        assert BB == [5.0, 5.0, 5.0]

    def test_reverse(self):
        BB, MSE = jisotonic5([4.0, 3.0, 2.0, 1.0])
        expected = 2.5
        for b in BB:
            assert b == pytest.approx(expected)

    def test_weighted(self):
        BB, MSE = jisotonic5([3.0, 1.0], [1.0, 2.0])
        # Weighted mean: (3*1 + 1*2)/(1+2) = 5/3
        expected = 5.0 / 3.0
        assert BB[0] == pytest.approx(expected)
        assert BB[1] == pytest.approx(expected)

    def test_empty(self):
        BB, MSE = jisotonic5([])
        assert BB == []
        assert MSE == []


class TestUpdown:
    def test_unimodal_peak(self):
        data = [1.0, 3.0, 5.0, 4.0, 2.0]
        out = jisotonic5_updown(data)
        # Should be non-decreasing then non-increasing
        peak_idx = out.index(max(out))
        for i in range(peak_idx):
            assert out[i] <= out[i + 1] + 1e-10
        for i in range(peak_idx, len(out) - 1):
            assert out[i] >= out[i + 1] - 1e-10


class TestDownup:
    def test_valley(self):
        data = [5.0, 3.0, 1.0, 2.0, 4.0]
        out = jisotonic5_downup(data)
        valley_idx = out.index(min(out))
        for i in range(valley_idx):
            assert out[i] >= out[i + 1] - 1e-10
        for i in range(valley_idx, len(out) - 1):
            assert out[i] <= out[i + 1] + 1e-10


# ======================================================================
# Isocut6 tests
# ======================================================================

class TestIsocut6:
    def test_unimodal(self):
        torch.manual_seed(0)
        samples = torch.randn(5000)
        dipscore, _ = isocut6(samples)
        assert dipscore < 2.0, f"Unimodal data should have low dip score, got {dipscore}"

    def test_bimodal(self):
        torch.manual_seed(0)
        s1 = torch.randn(2500) - 5.0
        s2 = torch.randn(2500) + 5.0
        samples = torch.cat([s1, s2])
        dipscore, cutpoint = isocut6(samples)
        assert dipscore >= 2.0, f"Bimodal data should have high dip score, got {dipscore}"
        assert -2.0 < cutpoint < 2.0, f"Cutpoint should be near 0, got {cutpoint}"

    def test_small_input(self):
        dipscore, cutpoint = isocut6(torch.tensor([1.0, 2.0]))
        # Should not crash, score should be small
        assert isinstance(dipscore, float)


# ======================================================================
# Parcelate tests
# ======================================================================

class TestParcelate:
    def test_basic(self):
        torch.manual_seed(42)
        X = torch.randn(1000, 5)
        labels = parcelate2(X, target_parcel_size=10, target_num_parcels=50)
        assert labels.shape == (1000,)
        assert labels.min().item() >= 1
        K = labels.max().item()
        assert K >= 2, f"Should create multiple parcels, got {K}"

    def test_small_input(self):
        X = torch.randn(5, 3)
        labels = parcelate2(X, target_parcel_size=10, target_num_parcels=200)
        assert labels.shape == (5,)
        assert labels.min().item() >= 1


# ======================================================================
# Core isosplit6 tests
# ======================================================================

class TestIsosplit6:
    def test_single_cluster(self):
        torch.manual_seed(42)
        X = torch.randn(200, 3) * 0.1
        labels = isosplit6_run(X)
        K = labels.max().item()
        assert K == 1, f"Single Gaussian should yield 1 cluster, got {K}"

    def test_two_clusters(self):
        X, true_labels = _make_blobs([[0, 0], [10, 10]], n_per_cluster=500)
        labels = isosplit6_run(X)
        K = labels.max().item()
        assert K == 2, f"Expected 2 clusters, got {K}"
        overlap = _label_overlap(true_labels, labels)
        assert overlap > 0.95, f"Label overlap {overlap:.2%} below 95%"

    def test_three_clusters(self):
        X, true_labels = _make_blobs([[0, 0], [10, 0], [5, 10]], n_per_cluster=400)
        labels = isosplit6_run(X)
        K = labels.max().item()
        assert K >= 2, f"Expected >= 2 clusters, got {K}"
        overlap = _label_overlap(true_labels, labels)
        assert overlap > 0.90, f"Label overlap {overlap:.2%} below 90%"

    def test_empty(self):
        X = torch.zeros(0, 5)
        labels = isosplit6_run(X)
        assert labels.shape == (0,)


# ======================================================================
# Clustering module (nn.Module) tests
# ======================================================================

class TestIsosplit6Clustering:
    def test_forward(self):
        X, true_labels = _make_blobs([[0, 0, 0], [10, 10, 10]], n_per_cluster=500)
        module = Isosplit6Clustering(npca_per_subdivision=3)
        labels = module(X)
        assert labels.shape == (1000,)
        assert labels.dtype == torch.int32
        K = labels.max().item()
        assert K >= 1

    def test_device_preservation(self):
        """Labels must reside on the same device as input."""
        X = torch.randn(100, 4)
        module = Isosplit6Clustering(npca_per_subdivision=3)
        labels = module(X)
        assert labels.device == X.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        X, true_labels = _make_blobs(
            [[0, 0], [10, 10]], n_per_cluster=500, device="cuda"
        )
        module = Isosplit6Clustering(npca_per_subdivision=2)
        labels = module(X)
        assert labels.device.type == "cuda"
        K = labels.max().item()
        assert K == 2


# ======================================================================
# Parity with C++ isosplit6 (optional — requires `isosplit6` package)
# ======================================================================

class TestParityWithCpp:
    @pytest.fixture(autouse=True)
    def _skip_if_no_cpp(self):
        pytest.importorskip("isosplit6")

    def test_parity_two_blobs(self):
        import numpy as np
        from isosplit6 import isosplit6 as isosplit6_cpp

        X, true_labels = _make_blobs([[0, 0], [10, 10]], n_per_cluster=500, seed=99)
        X_np = X.numpy()

        labels_cpp = isosplit6_cpp(X_np)
        labels_torch = isosplit6_run(X).numpy()

        overlap = _label_overlap(
            torch.as_tensor(labels_cpp, dtype=torch.int32),
            torch.as_tensor(labels_torch, dtype=torch.int32),
        )
        assert overlap > 0.90, (
            f"Parity overlap {overlap:.2%} below 90% — "
            "some divergence is expected due to floating-point differences"
        )

    def test_parity_three_blobs(self):
        import numpy as np
        from isosplit6 import isosplit6 as isosplit6_cpp

        X, true_labels = _make_blobs(
            [[0, 0, 0], [8, 0, 0], [4, 8, 0]],
            n_per_cluster=400, seed=123,
        )
        X_np = X.numpy()

        labels_cpp = isosplit6_cpp(X_np)
        labels_torch = isosplit6_run(X).numpy()

        overlap = _label_overlap(
            torch.as_tensor(labels_cpp, dtype=torch.int32),
            torch.as_tensor(labels_torch, dtype=torch.int32),
        )
        assert overlap > 0.85, f"Parity overlap {overlap:.2%} below 85%"
