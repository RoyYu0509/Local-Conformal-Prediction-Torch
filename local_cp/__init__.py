"""
Local CP - Conformal Prediction for PyTorch Regression Models
================================================================

A PyTorch package for uncertainty quantification using:
  • Scaled Conformal Prediction (CP)
  • Locally Adaptive Conformal Prediction (Local CP / Adaptive CP)

Supports CUDA, MPS (Apple Silicon), and CPU.

Example usage:
    >>> from local_cp import CP, AdaptiveCP
    >>> from local_cp.metrics import coverage, sharpness
    >>> 
    >>> # Wrap your trained PyTorch model
    >>> cp_model = CP(model, device='cuda')
    >>> lower, upper = cp_model.predict(
    ...     alpha=0.05,
    ...     X_test=X_test,
    ...     X_train=X_train,
    ...     Y_train=Y_train,
    ...     X_cal=X_cal,
    ...     Y_cal=Y_cal,
    ...     heuristic='feature',
    ...     k=20
    ... )
    >>> 
    >>> # Compute metrics
    >>> cov = coverage(Y_test, lower, upper)
    >>> sharp = sharpness(lower, upper)
"""

__version__ = "0.1.0"
__author__ = "Yifan Yu, Yangshuai Wang, Cheuk Hin Ho"

from .cp import CP
from .adaptive_cp import AdaptiveCP
from .metrics import coverage, sharpness, interval_score
from .batch_eval import cp_test_uncertainties, adaptive_cp_test_uncertainties_grid

__all__ = [
    "CP",
    "AdaptiveCP",
    "coverage",
    "sharpness",
    "interval_score",
    "cp_test_uncertainties",
    "adaptive_cp_test_uncertainties_grid",
]
