"""isosplit6_torch — Pure PyTorch implementation of the Isosplit6 clustering algorithm.

Usage::

    from isosplit6_torch import isosplit6, isocut6
    from isosplit6_torch.clustering import Isosplit6Clustering

    # Standalone usage
    labels = isosplit6(X)            # X: (N, M) torch.Tensor → labels: (N,) int32

    # As nn.Module (MountainSort5 drop-in)
    clustering = Isosplit6Clustering(npca_per_subdivision=10)
    labels = clustering(features)    # features: (N, D) → labels: (N,) int32
"""

from ._isosplit6 import isosplit6_run as isosplit6, Isosplit6Opts
from ._isocut6 import isocut6
from .clustering import Isosplit6Clustering

__all__ = [
    "isosplit6",
    "isocut6",
    "Isosplit6Opts",
    "Isosplit6Clustering",
]

__version__ = "0.1.0"
