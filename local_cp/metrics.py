"""
Evaluation metrics for conformal prediction intervals.

Provides standard metrics for assessing prediction interval quality:
  • Coverage: Empirical coverage probability
  • Sharpness: Average interval width
  • Interval Score: Proper scoring rule combining coverage and sharpness
"""

import numpy as np
import torch
from typing import Union


def _to_numpy_metric(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy for metric computation."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def coverage(y_true: Union[torch.Tensor, np.ndarray],
            lower: Union[torch.Tensor, np.ndarray],
            upper: Union[torch.Tensor, np.ndarray],
            reduction: str = "mean") -> Union[float, np.ndarray]:
    """
    Compute empirical coverage of prediction intervals.
    
    Coverage = P(Y_true ∈ [lower, upper])
    
    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        True target values.
    lower : torch.Tensor or np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Lower bounds of prediction intervals.
    upper : torch.Tensor or np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Upper bounds of prediction intervals.
    reduction : {'mean', 'none'}, default='mean'
        How to reduce over samples:
          • 'mean' - Return average coverage (scalar)
          • 'none' - Return per-sample coverage (array)
    
    Returns
    -------
    coverage : float or np.ndarray
        Empirical coverage probability.
        - If reduction='mean': scalar in [0, 1]
        - If reduction='none': boolean array of shape (n_samples,) or (n_samples, n_outputs)
    
    Examples
    --------
    >>> y_true = torch.randn(100, 1)
    >>> lower = y_true - 1.0
    >>> upper = y_true + 1.0
    >>> cov = coverage(y_true, lower, upper)
    >>> print(f"Coverage: {cov:.2%}")
    Coverage: 100.00%
    
    >>> # Per-sample coverage
    >>> cov_per_sample = coverage(y_true, lower, upper, reduction='none')
    >>> print(cov_per_sample.shape)  # (100, 1)
    """
    y_true = _to_numpy_metric(y_true)
    lower = _to_numpy_metric(lower)
    upper = _to_numpy_metric(upper)
    
    # Check if y_true is within [lower, upper]
    covered = (y_true >= lower) & (y_true <= upper)
    
    if reduction == "mean":
        return float(covered.mean())
    elif reduction == "none":
        return covered
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Choose 'mean' or 'none'.")


def sharpness(lower: Union[torch.Tensor, np.ndarray],
             upper: Union[torch.Tensor, np.ndarray],
             reduction: str = "mean") -> Union[float, np.ndarray]:
    """
    Compute sharpness (average interval width) of prediction intervals.
    
    Sharpness = E[upper - lower]
    
    Lower sharpness (narrower intervals) is better, given adequate coverage.
    
    Parameters
    ----------
    lower : torch.Tensor or np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Lower bounds of prediction intervals.
    upper : torch.Tensor or np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Upper bounds of prediction intervals.
    reduction : {'mean', 'none'}, default='mean'
        How to reduce over samples:
          • 'mean' - Return average width (scalar)
          • 'none' - Return per-sample widths (array)
    
    Returns
    -------
    sharpness : float or np.ndarray
        Average interval width.
        - If reduction='mean': scalar
        - If reduction='none': array of shape (n_samples,) or (n_samples, n_outputs)
    
    Examples
    --------
    >>> lower = torch.zeros(100, 1)
    >>> upper = torch.ones(100, 1)
    >>> sharp = sharpness(lower, upper)
    >>> print(f"Average width: {sharp:.3f}")
    Average width: 1.000
    """
    lower = _to_numpy_metric(lower)
    upper = _to_numpy_metric(upper)
    
    widths = upper - lower
    
    if reduction == "mean":
        return float(widths.mean())
    elif reduction == "none":
        return widths
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Choose 'mean' or 'none'.")


def interval_score(y_true: Union[torch.Tensor, np.ndarray],
                   lower: Union[torch.Tensor, np.ndarray],
                   upper: Union[torch.Tensor, np.ndarray],
                   alpha: float = 0.05,
                   reduction: str = "mean") -> Union[float, np.ndarray]:
    """
    Compute interval score (proper scoring rule for prediction intervals).
    
    The interval score balances sharpness (interval width) with coverage penalties:
    
        IS = (upper - lower) + 
             (2/α) * (lower - y_true) * I(y_true < lower) +
             (2/α) * (y_true - upper) * I(y_true > upper)
    
    Lower scores are better. This is a strictly proper scoring rule that
    encourages both narrow intervals and adequate coverage.
    
    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        True target values.
    lower : torch.Tensor or np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Lower bounds of prediction intervals.
    upper : torch.Tensor or np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Upper bounds of prediction intervals.
    alpha : float, default=0.05
        Miscoverage level (e.g., 0.05 for 95% intervals).
    reduction : {'mean', 'none'}, default='mean'
        How to reduce over samples:
          • 'mean' - Return average score (scalar)
          • 'none' - Return per-sample scores (array)
    
    Returns
    -------
    score : float or np.ndarray
        Interval score.
        - If reduction='mean': scalar
        - If reduction='none': array of shape (n_samples,) or (n_samples, n_outputs)
    
    Examples
    --------
    >>> y_true = torch.randn(100, 1)
    >>> lower = y_true - 1.0
    >>> upper = y_true + 1.0
    >>> score = interval_score(y_true, lower, upper, alpha=0.1)
    >>> print(f"Interval Score: {score:.3f}")
    
    References
    ----------
    .. [1] Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
           prediction, and estimation. Journal of the American Statistical Association,
           102(477), 359-378.
    """
    y_true = _to_numpy_metric(y_true)
    lower = _to_numpy_metric(lower)
    upper = _to_numpy_metric(upper)
    
    # Interval width
    width = upper - lower
    
    # Penalties for under-coverage
    lower_penalty = (2.0 / alpha) * np.maximum(lower - y_true, 0)
    upper_penalty = (2.0 / alpha) * np.maximum(y_true - upper, 0)
    
    # Total score
    scores = width + lower_penalty + upper_penalty
    
    if reduction == "mean":
        return float(scores.mean())
    elif reduction == "none":
        return scores
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Choose 'mean' or 'none'.")


def calibration_error(y_true: Union[torch.Tensor, np.ndarray],
                      lower: Union[torch.Tensor, np.ndarray],
                      upper: Union[torch.Tensor, np.ndarray],
                      alpha: float) -> float:
    """
    Compute absolute calibration error.
    
    Calibration Error = |empirical_coverage - (1 - alpha)|
    
    Measures how close the empirical coverage is to the target coverage.
    Lower is better (0 = perfect calibration).
    
    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        True target values.
    lower : torch.Tensor or np.ndarray
        Lower bounds of prediction intervals.
    upper : torch.Tensor or np.ndarray
        Upper bounds of prediction intervals.
    alpha : float
        Target miscoverage level.
    
    Returns
    -------
    error : float
        Absolute calibration error.
    
    Examples
    --------
    >>> y_true = torch.randn(100, 1)
    >>> lower, upper = y_true - 1.0, y_true + 1.0
    >>> cal_err = calibration_error(y_true, lower, upper, alpha=0.1)
    >>> print(f"Calibration Error: {cal_err:.4f}")
    """
    empirical_cov = coverage(y_true, lower, upper, reduction="mean")
    target_cov = 1.0 - alpha
    return abs(empirical_cov - target_cov)


def stratified_coverage(y_true: Union[torch.Tensor, np.ndarray],
                       lower: Union[torch.Tensor, np.ndarray],
                       upper: Union[torch.Tensor, np.ndarray],
                       strata: Union[torch.Tensor, np.ndarray]) -> dict:
    """
    Compute coverage per stratum (e.g., per class, per region).
    
    Useful for diagnosing conditional coverage properties.
    
    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        True target values.
    lower : torch.Tensor or np.ndarray
        Lower bounds of prediction intervals.
    upper : torch.Tensor or np.ndarray
        Upper bounds of prediction intervals.
    strata : torch.Tensor or np.ndarray, shape (n_samples,)
        Stratum labels (e.g., integer class IDs).
    
    Returns
    -------
    coverage_dict : dict
        Dictionary mapping stratum label to (coverage, count).
    
    Examples
    --------
    >>> y_true = torch.randn(100, 1)
    >>> lower, upper = y_true - 1.0, y_true + 1.0
    >>> regions = torch.randint(0, 3, (100,))  # 3 regions
    >>> cov_by_region = stratified_coverage(y_true, lower, upper, regions)
    >>> for region, (cov, n) in cov_by_region.items():
    ...     print(f"Region {region}: coverage={cov:.2%}, n={n}")
    """
    y_true = _to_numpy_metric(y_true)
    lower = _to_numpy_metric(lower)
    upper = _to_numpy_metric(upper)
    strata = _to_numpy_metric(strata).flatten()
    
    covered = (y_true >= lower) & (y_true <= upper)
    if covered.ndim > 1:
        covered = covered.all(axis=1)  # All outputs must be covered
    
    unique_strata = np.unique(strata)
    result = {}
    
    for stratum in unique_strata:
        mask = strata == stratum
        stratum_coverage = covered[mask].mean()
        stratum_count = mask.sum()
        result[stratum] = (float(stratum_coverage), int(stratum_count))
    
    return result


def conditional_coverage_diagnostic(y_true: Union[torch.Tensor, np.ndarray],
                                   lower: Union[torch.Tensor, np.ndarray],
                                   upper: Union[torch.Tensor, np.ndarray],
                                   X: Union[torch.Tensor, np.ndarray],
                                   n_bins: int = 10) -> dict:
    """
    Diagnose conditional coverage by binning along each feature dimension.
    
    Useful for detecting regions where coverage is inadequate.
    
    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        True target values.
    lower : torch.Tensor or np.ndarray
        Lower bounds.
    upper : torch.Tensor or np.ndarray
        Upper bounds.
    X : torch.Tensor or np.ndarray, shape (n_samples, n_features)
        Input features.
    n_bins : int, default=10
        Number of bins per feature dimension.
    
    Returns
    -------
    diagnostics : dict
        Dictionary with keys 'feature_0', 'feature_1', ..., each containing
        arrays of (bin_edges, coverage_per_bin).
    
    Examples
    --------
    >>> X = torch.randn(100, 2)
    >>> y_true = X.sum(dim=1, keepdim=True)
    >>> lower, upper = y_true - 1.0, y_true + 1.0
    >>> diag = conditional_coverage_diagnostic(y_true, lower, upper, X, n_bins=5)
    >>> for feat, (edges, covs) in diag.items():
    ...     print(f"{feat}: coverage per bin = {covs}")
    """
    y_true = _to_numpy_metric(y_true)
    lower = _to_numpy_metric(lower)
    upper = _to_numpy_metric(upper)
    X = _to_numpy_metric(X)
    
    covered = (y_true >= lower) & (y_true <= upper)
    if covered.ndim > 1:
        covered = covered.all(axis=1)
    
    n_features = X.shape[1]
    diagnostics = {}
    
    for feat_idx in range(n_features):
        x_feat = X[:, feat_idx]
        
        # Create bins
        bin_edges = np.percentile(x_feat, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(x_feat, bin_edges[1:-1])  # 0 to n_bins-1
        
        # Compute coverage per bin
        coverage_per_bin = []
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                bin_coverage = covered[mask].mean()
            else:
                bin_coverage = np.nan
            coverage_per_bin.append(bin_coverage)
        
        diagnostics[f"feature_{feat_idx}"] = (bin_edges, np.array(coverage_per_bin))
    
    return diagnostics
