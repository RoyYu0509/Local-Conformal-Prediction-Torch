"""
Scaled Conformal Prediction (CP) for PyTorch regression models.

This module implements split conformal prediction with distance-based scaling
heuristics for uncertainty quantification.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, Literal
from .utils import get_device, to_numpy, to_tensor, knn_distance, validate_tensors, ensure_2d


class CP:
    """
    Scaled Conformal Prediction for PyTorch regression models.
    
    Provides prediction intervals with finite-sample coverage guarantees using
    three scaling heuristics:
      • 'feature' - k-NN distance in input (feature) space
      • 'latent'  - k-NN distance in hidden layer space
      • 'raw_std' - predictive interval width from the model itself
    
    The model must be pre-trained before using CP.
    
    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained PyTorch regression model. For 'latent' heuristic, the model's
        forward method should accept `return_hidden=True` and return (output, hidden).
    device : str, torch.device, or None, default=None
        Device for computations ('cuda', 'mps', 'cpu'). If None, auto-detects.
    
    Attributes
    ----------
    model : torch.nn.Module
        The wrapped regression model (set to eval mode).
    device : torch.device
        Computation device.
    
    Examples
    --------
    >>> # Train your model first
    >>> model = MyNeuralNetwork()
    >>> # ... training code ...
    >>> 
    >>> # Wrap with CP
    >>> cp = CP(model, device='cuda')
    >>> 
    >>> # Get prediction intervals
    >>> lower, upper = cp.predict(
    ...     alpha=0.1,  # 90% coverage
    ...     X_test=X_test,
    ...     X_train=X_train,
    ...     Y_train=Y_train,
    ...     X_cal=X_cal,
    ...     Y_cal=Y_cal,
    ...     heuristic='feature',
    ...     k=20
    ... )
    
    Notes
    -----
    Based on "A Conformal Prediction Framework for Uncertainty Quantification
    in Physics-Informed Neural Networks" (Yu et al., 2025).
    
    References
    ----------
    .. [1] Yu, Y., Ho, C. H., & Wang, Y. (2025). A Conformal Prediction Framework
           for Uncertainty Quantification in Physics-Informed Neural Networks.
           arXiv:2509.13717
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: Union[str, torch.device, None] = None):
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device)
        self.model.eval()
    
    # ═══════════════════════════════════════════════════════════════
    # Private: Heuristic computation helpers
    # ═══════════════════════════════════════════════════════════════
    
    def _feature_distance(self, 
                         X_query: torch.Tensor, 
                         X_train: torch.Tensor, 
                         k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute k-NN distance in feature space and model predictions.
        
        Returns
        -------
        distances : np.ndarray, shape (n_query,)
            Mean k-NN distances.
        predictions : np.ndarray, shape (n_query, n_outputs)
            Model predictions on X_query.
        """
        with torch.no_grad():
            y_pred = self.model(X_query.to(self.device)).cpu().numpy()
        
        distances = knn_distance(
            to_numpy(X_query), 
            to_numpy(X_train), 
            k=k, 
            exclude_self=False
        )
        
        return distances, y_pred
    
    def _latent_distance(self, 
                        X_query: torch.Tensor, 
                        X_train: torch.Tensor, 
                        k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute k-NN distance in latent (hidden) space and model predictions.
        
        Requires model.forward(x, return_hidden=True) to return (output, hidden).
        
        Returns
        -------
        distances : np.ndarray, shape (n_query,)
            Mean k-NN distances in latent space.
        predictions : np.ndarray, shape (n_query, n_outputs)
            Model predictions on X_query.
        """
        with torch.no_grad():
            # Get hidden representations
            try:
                out_query, H_query = self.model(X_query.to(self.device), return_hidden=True)
                out_train, H_train = self.model(X_train.to(self.device), return_hidden=True)
            except TypeError:
                raise ValueError(
                    "Model must support forward(x, return_hidden=True) for 'latent' heuristic. "
                    "Expected return: (output, hidden_representation)."
                )
            
            y_pred = out_query.cpu().numpy()
        
        distances = knn_distance(
            to_numpy(H_query), 
            to_numpy(H_train), 
            k=k, 
            exclude_self=False
        )
        
        return distances, y_pred
    
    def _rawstd_width(self, 
                     alpha: float, 
                     X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictive interval width from model's own predict method.
        
        Requires model.predict(alpha, X) to return (lower, upper).
        
        Returns
        -------
        widths : np.ndarray, shape (n_samples,)
            Interval widths (upper - lower).
        predictions : np.ndarray, shape (n_samples, n_outputs)
            Model mean predictions.
        """
        with torch.no_grad():
            try:
                lower, upper = self.model.predict(alpha, X.to(self.device))
            except AttributeError:
                raise ValueError(
                    "Model must have .predict(alpha, X) method for 'raw_std' heuristic. "
                    "Expected return: (lower, upper) tensors."
                )
            
            lower_np = to_numpy(lower)
            upper_np = to_numpy(upper)
            y_pred = (lower_np + upper_np) / 2
            widths = (upper_np - lower_np).squeeze(-1)
        
        return widths, y_pred
    
    # ═══════════════════════════════════════════════════════════════
    # Private: Conformity score computation
    # ═══════════════════════════════════════════════════════════════
    
    def _compute_conformity_scores(self,
                                   X_cal: torch.Tensor,
                                   Y_cal: torch.Tensor,
                                   X_train: torch.Tensor,
                                   heuristic: str,
                                   k: int,
                                   alpha: float,
                                   eps: float = 1e-8) -> np.ndarray:
        """
        Compute scaled conformity scores on calibration set.
        
        Score = |Y_cal - Ŷ_cal| / scale
        
        where scale comes from the chosen heuristic.
        
        Returns
        -------
        scores : np.ndarray, shape (n_cal, n_outputs)
            Conformity scores for calibration set.
        """
        Y_cal_np = to_numpy(Y_cal)
        
        if heuristic == "feature":
            scale, y_pred = self._feature_distance(X_cal, X_train, k)
        elif heuristic == "latent":
            scale, y_pred = self._latent_distance(X_cal, X_train, k)
        elif heuristic == "raw_std":
            scale, y_pred = self._rawstd_width(alpha, X_cal)
        else:
            raise ValueError(
                f"Unknown heuristic '{heuristic}'. "
                "Choose from: 'feature', 'latent', 'raw_std'."
            )
        
        # Compute residuals
        residuals = np.abs(Y_cal_np - y_pred)
        
        # Avoid division by zero
        scale = np.maximum(scale, eps)
        
        # Broadcast scale for multi-output case
        if residuals.ndim > 1:
            scale = scale[:, None]
        
        return residuals / scale
    
    # ═══════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════
    
    def predict(self,
                alpha: float,
                X_test: Union[torch.Tensor, np.ndarray],
                X_train: Union[torch.Tensor, np.ndarray],
                Y_train: Union[torch.Tensor, np.ndarray],
                X_cal: Union[torch.Tensor, np.ndarray],
                Y_cal: Union[torch.Tensor, np.ndarray],
                heuristic: Literal["feature", "latent", "raw_std"] = "feature",
                k: int = 10,
                eps: float = 1e-8,
                deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conformal prediction intervals.
        
        Uses split conformal prediction: calibrates on (X_cal, Y_cal) and
        predicts on X_test. Training data (X_train, Y_train) is used for
        computing distance-based scales.
        
        Parameters
        ----------
        alpha : float
            Miscoverage level (e.g., 0.1 for 90% coverage).
        X_test : torch.Tensor or np.ndarray, shape (n_test, n_features)
            Test features.
        X_train : torch.Tensor or np.ndarray, shape (n_train, n_features)
            Training features (for distance computation).
        Y_train : torch.Tensor or np.ndarray, shape (n_train,) or (n_train, n_outputs)
            Training targets (not used in standard CP, but kept for API consistency).
        X_cal : torch.Tensor or np.ndarray, shape (n_cal, n_features)
            Calibration features.
        Y_cal : torch.Tensor or np.ndarray, shape (n_cal,) or (n_cal, n_outputs)
            Calibration targets.
        heuristic : {'feature', 'latent', 'raw_std'}, default='feature'
            Scaling heuristic:
              • 'feature' - k-NN distance in input space
              • 'latent'  - k-NN distance in hidden space
              • 'raw_std' - predictive interval width from model
        k : int, default=10
            Number of nearest neighbors for distance computation.
        eps : float, default=1e-8
            Small constant to avoid division by zero.
        deterministic : bool, default=True
            If True, sets random seeds for reproducible results.
        
        Returns
        -------
        lower : torch.Tensor, shape (n_test, n_outputs)
            Lower bounds of prediction intervals.
        upper : torch.Tensor, shape (n_test, n_outputs)
            Upper bounds of prediction intervals.
        
        Examples
        --------
        >>> cp = CP(model, device='cuda')
        >>> lower, upper = cp.predict(
        ...     alpha=0.1,
        ...     X_test=X_test,
        ...     X_train=X_train,
        ...     Y_train=Y_train,
        ...     X_cal=X_cal,
        ...     Y_cal=Y_cal,
        ...     heuristic='feature',
        ...     k=20
        ... )
        >>> print(f"Coverage: {((Y_test >= lower) & (Y_test <= upper)).float().mean():.2%}")
        
        Notes
        -----
        The conformal quantile level is computed as:
            q = ceil((n_cal + 1) * (1 - alpha)) / n_cal
        
        This guarantees at least (1 - alpha) marginal coverage in expectation.
        """
        # Set deterministic mode if requested
        if deterministic:
            from .utils import set_deterministic
            set_deterministic(seed=0)
        
        # Convert inputs to tensors
        X_test = to_tensor(X_test, self.device)
        X_train = to_tensor(X_train, self.device)
        Y_train = to_tensor(Y_train, self.device)
        X_cal = to_tensor(X_cal, self.device)
        Y_cal = to_tensor(Y_cal, self.device)
        
        # Ensure Y tensors are 2D
        Y_train = ensure_2d(Y_train)
        Y_cal = ensure_2d(Y_cal)
        
        # Validate inputs
        validate_tensors(X_train, Y_train, "X_train", "Y_train")
        validate_tensors(X_cal, Y_cal, "X_cal", "Y_cal")
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # ─── Step 1: Compute conformity scores on calibration set ───
        cal_scores = self._compute_conformity_scores(
            X_cal, Y_cal, X_train, heuristic, k, alpha, eps
        )
        
        n_cal = cal_scores.shape[0]
        
        # ─── Step 2: Compute conformal quantile ───
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = np.clip(q_level, 0.0, 1.0 - 1e-12)
        
        q_hat = np.quantile(
            cal_scores,
            q_level,
            axis=0,
            method="higher"
        )  # shape: (n_outputs,) or scalar
        
        # ─── Step 3: Get predictions and scales on test set ───
        if heuristic == "feature":
            test_scale, y_pred = self._feature_distance(X_test, X_train, k)
        elif heuristic == "latent":
            test_scale, y_pred = self._latent_distance(X_test, X_train, k)
        elif heuristic == "raw_std":
            test_scale, y_pred = self._rawstd_width(alpha, X_test)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")
        
        test_scale = np.maximum(test_scale, eps)
        
        # ─── Step 4: Build prediction intervals ───
        # Broadcast: test_scale is (n_test,), q_hat is (n_outputs,)
        if y_pred.ndim > 1:
            epsilon = q_hat * test_scale[:, None]
        else:
            epsilon = q_hat * test_scale
        
        lower = torch.tensor(y_pred - epsilon, dtype=torch.float32, device=self.device)
        upper = torch.tensor(y_pred + epsilon, dtype=torch.float32, device=self.device)
        
        return lower, upper
