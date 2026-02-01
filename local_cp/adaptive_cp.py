"""
Locally Adaptive Conformal Prediction (Local CP / Adaptive CP).

Learns a quantile regression network to predict local conformity scores,
enabling spatially varying prediction interval widths.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, Literal, List
from .utils import (get_device, to_numpy, to_tensor, knn_distance, 
                    validate_tensors, ensure_2d, set_deterministic)


class AdaptiveCP:
    """
    Locally Adaptive Conformal Prediction for PyTorch regression models.
    
    Extends standard conformal prediction by learning a quantile regression network
    that predicts local conformity scores, allowing prediction interval widths to
    adapt to local complexity and uncertainty.
    
    The model must be pre-trained before using AdaptiveCP.
    
    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained PyTorch regression model.
    alpha : float, default=0.05
        Target miscoverage level (e.g., 0.05 for 95% coverage).
    device : str, torch.device, or None, default=None
        Device for computations. If None, auto-detects.
    heuristic : {'feature', 'latent', 'raw_std'}, default='feature'
        Scaling heuristic for conformity scores.
    hidden_layers : tuple of int, default=(64, 64, 64)
        Hidden layer sizes for the quantile regression network.
    learning_rate : float, default=5e-4
        Learning rate for training the quantile network.
    epochs : int, default=20000
        Training epochs for the quantile network.
    step_size : int, default=5000
        Learning rate scheduler step size.
    gamma : float, default=0.5
        Learning rate decay factor.
    quant_seed : int or None, default=12345
        Random seed for deterministic quantile network initialization.
    
    Attributes
    ----------
    model : torch.nn.Module
        The wrapped regression model.
    device : torch.device
        Computation device.
    quantile_model : torch.nn.Module or None
        Learned quantile regression network (initialized on first predict call).
    quantile_model_trained : bool
        Whether the quantile network has been trained.
    
    Examples
    --------
    >>> # Train your model first
    >>> model = MyNeuralNetwork()
    >>> # ... training code ...
    >>> 
    >>> # Wrap with AdaptiveCP
    >>> acp = AdaptiveCP(
    ...     model, 
    ...     alpha=0.1,  # 90% coverage
    ...     device='cuda',
    ...     heuristic='feature',
    ...     hidden_layers=(128, 128, 128),
    ...     epochs=15000
    ... )
    >>> 
    >>> # Get adaptive prediction intervals
    >>> lower, upper = acp.predict(
    ...     alpha=0.1,
    ...     X_test=X_test,
    ...     X_train=X_train,
    ...     Y_train=Y_train,
    ...     X_cal=X_cal,
    ...     Y_cal=Y_cal,
    ...     k=20
    ... )
    
    Notes
    -----
    Based on the Local CP algorithm from "A Conformal Prediction Framework for
    Uncertainty Quantification in Physics-Informed Neural Networks" (Yu et al., 2025).
    
    The quantile network is trained once on the training set using pinball loss,
    then calibrated using the calibration set to ensure valid coverage.
    
    References
    ----------
    .. [1] Yu, Y., Ho, C. H., & Wang, Y. (2025). A Conformal Prediction Framework
           for Uncertainty Quantification in Physics-Informed Neural Networks.
           arXiv:2509.13717
    """
    
    def __init__(self,
                 model: nn.Module,
                 alpha: float = 0.05,
                 device: Union[str, torch.device, None] = None,
                 heuristic: Literal["feature", "latent", "raw_std"] = "feature",
                 hidden_layers: Tuple[int, ...] = (64, 64, 64),
                 learning_rate: float = 5e-4,
                 epochs: int = 20000,
                 step_size: int = 5000,
                 gamma: float = 0.5,
                 quant_seed: Optional[int] = 12345):
        
        self.model = model
        self.alpha = alpha
        self.device = get_device(device)
        self.heuristic = heuristic
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.step_size = step_size
        self.gamma = gamma
        self.quant_seed = quant_seed
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Quantile network (initialized lazily on first predict call)
        self.quantile_model: Optional[nn.Module] = None
        self.quantile_model_trained = False
    
    # ═══════════════════════════════════════════════════════════════
    # Private: Distance and width computation
    # ═══════════════════════════════════════════════════════════════
    
    def _feature_distance(self, X_query: torch.Tensor, X_train: torch.Tensor, k: int) -> np.ndarray:
        """Compute mean k-NN distance in feature space."""
        return knn_distance(to_numpy(X_query), to_numpy(X_train), k, exclude_self=False)
    
    def _latent_distance(self, X_query: torch.Tensor, X_train: torch.Tensor, k: int) -> np.ndarray:
        """Compute mean k-NN distance in latent (hidden) space."""
        with torch.no_grad():
            try:
                _, H_query = self.model(X_query.to(self.device), return_hidden=True)
                _, H_train = self.model(X_train.to(self.device), return_hidden=True)
            except TypeError:
                raise ValueError(
                    "Model must support forward(x, return_hidden=True) for 'latent' heuristic."
                )
        
        return knn_distance(to_numpy(H_query), to_numpy(H_train), k, exclude_self=False)
    
    def _rawstd_width(self, alpha: float, X: torch.Tensor) -> np.ndarray:
        """Get interval width from model's predict method."""
        with torch.no_grad():
            try:
                lower, upper = self.model.predict(alpha, X.to(self.device))
            except AttributeError:
                raise ValueError(
                    "Model must have .predict(alpha, X) method for 'raw_std' heuristic."
                )
        
        width = (to_numpy(upper) - to_numpy(lower)).squeeze(-1)
        return width
    
    # ═══════════════════════════════════════════════════════════════
    # Private: Conformity scores
    # ═══════════════════════════════════════════════════════════════
    
    def _compute_conformity_scores(self,
                                   X: torch.Tensor,
                                   Y: torch.Tensor,
                                   X_train: torch.Tensor,
                                   k: int,
                                   eps: float = 1e-8,
                                   exclude_self: bool = False) -> np.ndarray:
        """
        Compute scaled conformity scores: |Y - Ŷ| / scale.
        
        Parameters
        ----------
        exclude_self : bool
            If True, excludes self-neighbor (useful when X == X_train).
        
        Returns
        -------
        scores : np.ndarray, shape (n, n_outputs) or (n,)
            Conformity scores.
        """
        # Get predictions
        with torch.no_grad():
            Y_pred = self.model(X.to(self.device)).cpu().numpy()
        
        # Compute residuals
        residuals = np.abs(to_numpy(Y) - Y_pred)
        
        # Get scale based on heuristic
        if self.heuristic == "feature":
            scale = knn_distance(to_numpy(X), to_numpy(X_train), 
                               k + (1 if exclude_self else 0), exclude_self=exclude_self)
        elif self.heuristic == "latent":
            scale = self._latent_distance(X, X_train, k)
        elif self.heuristic == "raw_std":
            scale = self._rawstd_width(self.alpha, X)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")
        
        # Avoid division by zero
        scale = np.maximum(scale, eps)
        
        # Broadcast scale for multi-output
        if residuals.ndim > 1:
            scale = scale[:, None]
        
        return residuals / scale
    
    # ═══════════════════════════════════════════════════════════════
    # Private: Quantile network training
    # ═══════════════════════════════════════════════════════════════
    
    def _build_quantile_network(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build quantile regression network."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers).to(self.device)
    
    def _train_quantile_network(self,
                                X_features: torch.Tensor,
                                scores: torch.Tensor,
                                verbose: bool = True):
        """
        Train quantile regression network using pinball loss.
        
        Parameters
        ----------
        X_features : torch.Tensor, shape (n, n_features)
            Input features (could be raw features, latent representations, etc.).
        scores : torch.Tensor, shape (n, n_outputs)
            Target conformity scores.
        verbose : bool
            Whether to print training progress.
        """
        # Set random seed for reproducible initialization
        if self.quant_seed is not None:
            torch.manual_seed(self.quant_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.quant_seed)
        
        # Build network
        self.quantile_model = self._build_quantile_network(
            input_dim=X_features.shape[1],
            output_dim=scores.shape[1] if scores.ndim > 1 else 1
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(self.quantile_model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        
        # Quantile level for pinball loss
        tau = 1.0 - self.alpha
        
        def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Pinball (quantile) loss."""
            diff = target - pred
            return torch.where(diff >= 0, tau * diff, (tau - 1) * diff).mean()
        
        # Move data to device
        X_features = X_features.to(self.device)
        scores = scores.to(self.device)
        
        # Training loop
        self.quantile_model.train()
        
        if verbose:
            print(f"\n[Adaptive CP] Training quantile network: {scores.shape[0]} samples, "
                  f"{self.epochs} epochs, tau={tau:.3f}")
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            predictions = self.quantile_model(X_features)
            loss = pinball_loss(predictions, scores)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if verbose and (epoch + 1) % max(self.epochs // 10, 1) == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4e}, "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        self.quantile_model.eval()
        self.quantile_model_trained = True
        
        if verbose:
            print("[Adaptive CP] Quantile network training complete.\n")
    
    # ═══════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════
    
    def predict(self,
                alpha: float,
                X_test: Union[torch.Tensor, np.ndarray],
                X_train: Union[torch.Tensor, np.ndarray],
                Y_train: Union[torch.Tensor, np.ndarray],
                X_cal: Optional[Union[torch.Tensor, np.ndarray]] = None,
                Y_cal: Optional[Union[torch.Tensor, np.ndarray]] = None,
                heuristic: Optional[Literal["feature", "latent", "raw_std"]] = None,
                k: int = 10,
                eps: float = 1e-8,
                deterministic: bool = False,
                verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute locally adaptive conformal prediction intervals.
        
        Trains a quantile network on the training set (if not already trained),
        optionally calibrates using the calibration set, then predicts adaptive
        intervals on the test set.
        
        Parameters
        ----------
        alpha : float
            Miscoverage level (must match the alpha used in __init__).
        X_test : torch.Tensor or np.ndarray, shape (n_test, n_features)
            Test features.
        X_train : torch.Tensor or np.ndarray, shape (n_train, n_features)
            Training features (used for distance and quantile network training).
        Y_train : torch.Tensor or np.ndarray, shape (n_train,) or (n_train, n_outputs)
            Training targets (used for quantile network training).
        X_cal : torch.Tensor or np.ndarray or None, shape (n_cal, n_features)
            Calibration features (optional, for conformal calibration).
        Y_cal : torch.Tensor or np.ndarray or None, shape (n_cal,) or (n_cal, n_outputs)
            Calibration targets (optional, for conformal calibration).
        heuristic : {'feature', 'latent', 'raw_std'} or None
            Override the heuristic set in __init__. If None, uses self.heuristic.
        k : int, default=10
            Number of nearest neighbors for distance computation.
        eps : float, default=1e-8
            Small constant to avoid division by zero.
        deterministic : bool, default=False
            If True, sets random seeds for reproducibility.
        verbose : bool, default=True
            Whether to print training progress.
        
        Returns
        -------
        lower : torch.Tensor, shape (n_test, n_outputs)
            Lower bounds of adaptive prediction intervals.
        upper : torch.Tensor, shape (n_test, n_outputs)
            Upper bounds of adaptive prediction intervals.
        
        Examples
        --------
        >>> acp = AdaptiveCP(model, alpha=0.1, device='cuda')
        >>> 
        >>> # First call trains the quantile network
        >>> lower, upper = acp.predict(
        ...     alpha=0.1,
        ...     X_test=X_test,
        ...     X_train=X_train,
        ...     Y_train=Y_train,
        ...     X_cal=X_cal,
        ...     Y_cal=Y_cal,
        ...     k=20,
        ...     verbose=True
        ... )
        >>> 
        >>> # Subsequent calls reuse the trained network
        >>> lower2, upper2 = acp.predict(0.1, X_test2, X_train, Y_train, X_cal, Y_cal)
        
        Notes
        -----
        The quantile network is trained to predict the (1-alpha)-quantile of the
        scaled residuals, allowing interval widths to adapt to local data density
        and model uncertainty.
        
        If calibration data is provided, an additional conformal calibration step
        ensures finite-sample coverage guarantees.
        """
        # Validate alpha
        if alpha != self.alpha:
            raise ValueError(
                f"Alpha mismatch: initialized with {self.alpha}, called with {alpha}. "
                "AdaptiveCP requires consistent alpha for quantile network training."
            )
        
        # Use provided heuristic or default
        heuristic = heuristic or self.heuristic
        
        # Set deterministic mode if requested
        if deterministic:
            set_deterministic(seed=0)
        
        # Convert inputs to tensors
        X_test = to_tensor(X_test, self.device)
        X_train = to_tensor(X_train, self.device)
        Y_train = to_tensor(Y_train, self.device)
        
        if X_cal is not None:
            X_cal = to_tensor(X_cal, self.device)
        if Y_cal is not None:
            Y_cal = to_tensor(Y_cal, self.device)
        
        # Ensure Y tensors are 2D
        Y_train = ensure_2d(Y_train)
        if Y_cal is not None:
            Y_cal = ensure_2d(Y_cal)
        
        # Validate inputs
        validate_tensors(X_train, Y_train, "X_train", "Y_train")
        if X_cal is not None and Y_cal is not None:
            validate_tensors(X_cal, Y_cal, "X_cal", "Y_cal")
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # ─── Step 1: Train quantile network (if not already trained) ───
        if not self.quantile_model_trained:
            if verbose:
                print(f"[Adaptive CP] First prediction call - training quantile network...")
            
            # Build features for quantile network
            if heuristic == "feature":
                X_features = X_train
            elif heuristic == "latent":
                with torch.no_grad():
                    _, X_features = self.model(X_train, return_hidden=True)
            elif heuristic == "raw_std":
                width = torch.tensor(
                    self._rawstd_width(self.alpha, X_train),
                    dtype=torch.float32,
                    device=self.device
                )
                X_features = torch.cat([X_train, width.unsqueeze(-1)], dim=1)
            else:
                raise ValueError(f"Unknown heuristic: {heuristic}")
            
            # Compute conformity scores on training set (exclude self-neighbor)
            scores_train = self._compute_conformity_scores(
                X_train, Y_train, X_train, k, eps, exclude_self=True
            )
            scores_train = torch.tensor(scores_train, dtype=torch.float32)
            
            # Ensure scores are 2D
            if scores_train.ndim == 1:
                scores_train = scores_train.unsqueeze(-1)
            
            # Train the quantile network
            self._train_quantile_network(X_features, scores_train, verbose=verbose)
        
        # ─── Step 2: Get quantile predictions and base predictions on test set ───
        
        # Build test features
        if heuristic == "feature":
            X_features_test = X_test
        elif heuristic == "latent":
            with torch.no_grad():
                _, X_features_test = self.model(X_test, return_hidden=True)
        elif heuristic == "raw_std":
            width_test = torch.tensor(
                self._rawstd_width(self.alpha, X_test),
                dtype=torch.float32,
                device=self.device
            )
            X_features_test = torch.cat([X_test, width_test.unsqueeze(-1)], dim=1)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")
        
        # Get quantile predictions
        with torch.no_grad():
            q_hat_test = self.quantile_model(X_features_test).cpu().numpy()
            y_pred_test = self.model(X_test).cpu().numpy()
        
        q_hat_test = np.maximum(q_hat_test, 0.0)  # Ensure non-negative
        
        # Get local scales on test set
        if heuristic == "feature":
            scale_test = self._feature_distance(X_test, X_train, k)
        elif heuristic == "latent":
            scale_test = self._latent_distance(X_test, X_train, k)
        elif heuristic == "raw_std":
            scale_test = self._rawstd_width(self.alpha, X_test)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")
        
        scale_test = np.maximum(scale_test, eps)
        
        # ─── Step 3: Conformal calibration (if calibration set provided) ───
        c = 1.0  # Calibration multiplier
        
        if X_cal is not None and Y_cal is not None:
            if verbose:
                print(f"[Adaptive CP] Performing conformal calibration with {X_cal.shape[0]} samples...")
            
            # Build calibration features
            if heuristic == "feature":
                X_features_cal = X_cal
            elif heuristic == "latent":
                with torch.no_grad():
                    _, X_features_cal = self.model(X_cal, return_hidden=True)
            elif heuristic == "raw_std":
                width_cal = torch.tensor(
                    self._rawstd_width(self.alpha, X_cal),
                    dtype=torch.float32,
                    device=self.device
                )
                X_features_cal = torch.cat([X_cal, width_cal.unsqueeze(-1)], dim=1)
            
            # Get quantile and model predictions on calibration set
            with torch.no_grad():
                q_hat_cal = self.quantile_model(X_features_cal).cpu().numpy()
                y_pred_cal = self.model(X_cal).cpu().numpy()
            
            # Get scales on calibration set
            if heuristic == "feature":
                scale_cal = self._feature_distance(X_cal, X_train, k)
            elif heuristic == "latent":
                scale_cal = self._latent_distance(X_cal, X_train, k)
            elif heuristic == "raw_std":
                scale_cal = self._rawstd_width(self.alpha, X_cal)
            
            scale_cal = np.maximum(scale_cal, eps)
            
            # Compute normalized residuals
            residuals_cal = np.abs(to_numpy(Y_cal) - y_pred_cal)
            
            if residuals_cal.ndim > 1 and q_hat_cal.ndim > 1:
                denominators = q_hat_cal * scale_cal[:, None]
            else:
                denominators = q_hat_cal.squeeze() * scale_cal
            
            R = residuals_cal / np.maximum(denominators, eps)
            
            # Take maximum over output dimensions for conservative coverage
            if R.ndim > 1:
                R = R.max(axis=1)
            
            # Compute conformal quantile
            m = R.shape[0]
            k_idx = int(np.ceil((m + 1) * (1 - self.alpha))) - 1
            k_idx = max(0, min(k_idx, m - 1))
            
            c = float(np.partition(R, k_idx)[k_idx])
            
            if verbose:
                print(f"[Adaptive CP] Calibration multiplier: c = {c:.4f}")
        
        # ─── Step 4: Assemble final intervals ───
        # epsilon = c * q_hat * scale
        if y_pred_test.ndim > 1 and q_hat_test.ndim > 1:
            epsilon = c * q_hat_test * scale_test[:, None]
        else:
            epsilon = c * q_hat_test.squeeze() * scale_test
        
        lower = torch.tensor(y_pred_test - epsilon, dtype=torch.float32, device=self.device)
        upper = torch.tensor(y_pred_test + epsilon, dtype=torch.float32, device=self.device)
        
        return lower, upper
