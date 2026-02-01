"""
Batch evaluation functions for conformal prediction with parallel alpha processing.

Provides utilities for testing CP methods across multiple miscoverage levels
with efficient parallel computation.
"""

import numpy as np
import pandas as pd
import torch
from typing import Union, List, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

from .cp import CP
from .adaptive_cp import AdaptiveCP
from .metrics import coverage, sharpness, interval_score


def _evaluate_single_alpha(
    alpha: float,
    cp_model,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_cal: torch.Tensor,
    Y_cal: torch.Tensor,
    heuristic: str,
    k: int,
    idx_subset: Optional[torch.Tensor] = None
) -> dict:
    """Evaluate CP for a single alpha value."""
    # Get predictions
    lower, upper = cp_model.predict(
        alpha=alpha,
        X_test=X_test,
        X_train=X_train,
        Y_train=Y_train,
        X_cal=X_cal,
        Y_cal=Y_cal,
        heuristic=heuristic,
        k=k,
        deterministic=True
    )
    
    # Apply subset mask if provided
    if idx_subset is not None:
        # Ensure idx_subset is the correct type (long tensor)
        if isinstance(idx_subset, torch.Tensor):
            idx_subset = idx_subset.long()
        Y_test_eval = Y_test[idx_subset]
        lower_eval = lower[idx_subset]
        upper_eval = upper[idx_subset]
        n_points = len(idx_subset)
    else:
        Y_test_eval = Y_test
        lower_eval = lower
        upper_eval = upper
        n_points = len(Y_test)
    
    # Compute metrics
    cov = coverage(Y_test_eval, lower_eval, upper_eval)
    sharp = sharpness(lower_eval, upper_eval)
    int_sc = interval_score(Y_test_eval, lower_eval, upper_eval, alpha)
    
    return {
        'alpha': alpha,
        'coverage': cov,
        'sharpness': sharp,
        'interval_score': int_sc,
        'n_points': n_points
    }


def cp_test_uncertainties(
    cp_model: CP,
    alphas: Union[List[float], np.ndarray],
    X_train: Union[torch.Tensor, np.ndarray],
    Y_train: Union[torch.Tensor, np.ndarray],
    X_cal: Union[torch.Tensor, np.ndarray],
    Y_cal: Union[torch.Tensor, np.ndarray],
    X_test: Union[torch.Tensor, np.ndarray],
    Y_test: Union[torch.Tensor, np.ndarray],
    heuristic: str = 'feature',
    k: int = 10,
    idx_subset: Optional[Union[torch.Tensor, np.ndarray]] = None,
    parallel: bool = True,
    n_workers: Optional[int] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Evaluate CP across multiple alpha levels with parallel computation.
    
    Parameters
    ----------
    cp_model : CP
        Fitted CP model instance.
    alphas : list or np.ndarray
        List of miscoverage levels to evaluate.
    X_train, Y_train : torch.Tensor or np.ndarray
        Training data.
    X_cal, Y_cal : torch.Tensor or np.ndarray
        Calibration data.
    X_test, Y_test : torch.Tensor or np.ndarray
        Test data.
    heuristic : str, default='feature'
        Heuristic to use ('feature', 'latent', 'raw_std').
    k : int, default=10
        Number of nearest neighbors.
    idx_subset : torch.Tensor or np.ndarray, optional
        Subset of test indices to evaluate on.
    parallel : bool, default=True
        Whether to parallelize over alpha values.
    n_workers : int, optional
        Number of parallel workers. If None, uses CPU count.
    show_progress : bool, default=True
        Whether to show progress bar.
    
    Returns
    -------
    pd.DataFrame
        Results with columns: alpha, coverage, sharpness, interval_score, n_points
    
    Examples
    --------
    >>> cp = CP(model, device='cuda')
    >>> results = cp_test_uncertainties(
    ...     cp, alphas=[0.01, 0.05, 0.1, 0.2],
    ...     X_train=X_train, Y_train=Y_train,
    ...     X_cal=X_cal, Y_cal=Y_cal,
    ...     X_test=X_test, Y_test=Y_test,
    ...     parallel=True
    ... )
    >>> print(results)
    """
    # Convert to tensors
    from .utils import to_tensor, get_device
    
    device = cp_model.device
    X_train = to_tensor(X_train, device)
    Y_train = to_tensor(Y_train, device)
    X_cal = to_tensor(X_cal, device)
    Y_cal = to_tensor(Y_cal, device)
    X_test = to_tensor(X_test, device)
    Y_test = to_tensor(Y_test, device)
    
    if idx_subset is not None:
        idx_subset = to_tensor(idx_subset, device)
    
    results = []
    
    if parallel and len(alphas) > 1:
        # Parallel evaluation using ThreadPoolExecutor (better for GPU)
        n_workers = n_workers or min(len(alphas), 4)
        
        eval_fn = partial(
            _evaluate_single_alpha,
            cp_model=cp_model,
            X_test=X_test,
            Y_test=Y_test,
            X_train=X_train,
            Y_train=Y_train,
            X_cal=X_cal,
            Y_cal=Y_cal,
            heuristic=heuristic,
            k=k,
            idx_subset=idx_subset
        )
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(eval_fn, alpha) for alpha in alphas]
            
            iterator = futures
            if show_progress:
                iterator = tqdm(futures, desc="Evaluating CP", total=len(alphas))
            
            for future in iterator:
                results.append(future.result())
    else:
        # Sequential evaluation
        iterator = alphas
        if show_progress:
            iterator = tqdm(alphas, desc="Evaluating CP")
        
        for alpha in iterator:
            result = _evaluate_single_alpha(
                alpha, cp_model, X_test, Y_test,
                X_train, Y_train, X_cal, Y_cal,
                heuristic, k, idx_subset
            )
            results.append(result)
    
    return pd.DataFrame(results)


def _evaluate_single_alpha_adaptive(
    alpha: float,
    base_model,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_cal: torch.Tensor,
    Y_cal: torch.Tensor,
    heuristic: str,
    k: int,
    hidden_layers: tuple,
    learning_rate: float,
    epochs: int,
    step_size: int,
    gamma: float,
    device: torch.device,
    idx_subset: Optional[torch.Tensor] = None
) -> dict:
    """Evaluate Adaptive CP for a single alpha value."""
    # Create new AdaptiveCP instance for this alpha
    acp = AdaptiveCP(
        base_model,
        alpha=alpha,
        device=device,
        heuristic=heuristic,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        step_size=step_size,
        gamma=gamma,
        quant_seed=42  # Fixed seed for reproducibility
    )
    
    # Get predictions
    lower, upper = acp.predict(
        alpha=alpha,
        X_test=X_test,
        X_train=X_train,
        Y_train=Y_train,
        X_cal=X_cal,
        Y_cal=Y_cal,
        k=k,
        verbose=False,
        deterministic=True
    )
    
    # Apply subset mask if provided
    if idx_subset is not None:
        # Ensure idx_subset is the correct type (long tensor)
        if isinstance(idx_subset, torch.Tensor):
            idx_subset = idx_subset.long()
        Y_test_eval = Y_test[idx_subset]
        lower_eval = lower[idx_subset]
        upper_eval = upper[idx_subset]
        n_points = len(idx_subset)
    else:
        Y_test_eval = Y_test
        lower_eval = lower
        upper_eval = upper
        n_points = len(Y_test)
    
    # Compute metrics
    cov = coverage(Y_test_eval, lower_eval, upper_eval)
    sharp = sharpness(lower_eval, upper_eval)
    int_sc = interval_score(Y_test_eval, lower_eval, upper_eval, alpha)
    
    return {
        'alpha': alpha,
        'coverage': cov,
        'sharpness': sharp,
        'interval_score': int_sc,
        'n_points': n_points
    }


def adaptive_cp_test_uncertainties_grid(
    base_model: torch.nn.Module,
    alphas: Union[List[float], np.ndarray],
    X_train: Union[torch.Tensor, np.ndarray],
    Y_train: Union[torch.Tensor, np.ndarray],
    X_cal: Union[torch.Tensor, np.ndarray],
    Y_cal: Union[torch.Tensor, np.ndarray],
    X_test: Union[torch.Tensor, np.ndarray],
    Y_test: Union[torch.Tensor, np.ndarray],
    heuristic: str = 'feature',
    k: int = 10,
    hidden_layers: tuple = (64, 64, 64),
    learning_rate: float = 5e-4,
    epochs: int = 20000,
    step_size: int = 5000,
    gamma: float = 0.5,
    device: Optional[Union[str, torch.device]] = None,
    idx_subset: Optional[Union[torch.Tensor, np.ndarray]] = None,
    parallel: bool = True,
    n_workers: Optional[int] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Evaluate Adaptive CP across multiple alpha levels with parallel computation.
    
    Each alpha value trains its own quantile network, so parallelization provides
    significant speedup.
    
    Parameters
    ----------
    base_model : torch.nn.Module
        Pre-trained PyTorch model.
    alphas : list or np.ndarray
        List of miscoverage levels to evaluate.
    X_train, Y_train : torch.Tensor or np.ndarray
        Training data.
    X_cal, Y_cal : torch.Tensor or np.ndarray
        Calibration data.
    X_test, Y_test : torch.Tensor or np.ndarray
        Test data.
    heuristic : str, default='feature'
        Heuristic to use ('feature', 'latent', 'raw_std').
    k : int, default=10
        Number of nearest neighbors.
    hidden_layers : tuple, default=(64, 64, 64)
        Quantile network architecture.
    learning_rate : float, default=5e-4
        Learning rate for quantile network.
    epochs : int, default=20000
        Training epochs for quantile network.
    step_size : int, default=5000
        LR scheduler step size.
    gamma : float, default=0.5
        LR decay factor.
    device : str or torch.device, optional
        Computation device.
    idx_subset : torch.Tensor or np.ndarray, optional
        Subset of test indices to evaluate on.
    parallel : bool, default=True
        Whether to parallelize over alpha values.
    n_workers : int, optional
        Number of parallel workers. If None, uses CPU count.
    show_progress : bool, default=True
        Whether to show progress bar.
    
    Returns
    -------
    pd.DataFrame
        Results with columns: alpha, coverage, sharpness, interval_score, n_points
    
    Examples
    --------
    >>> results = adaptive_cp_test_uncertainties_grid(
    ...     model, alphas=[0.01, 0.05, 0.1, 0.2],
    ...     X_train=X_train, Y_train=Y_train,
    ...     X_cal=X_cal, Y_cal=Y_cal,
    ...     X_test=X_test, Y_test=Y_test,
    ...     parallel=True, n_workers=4
    ... )
    """
    # Convert to tensors
    from .utils import to_tensor, get_device
    
    device = get_device(device)
    X_train = to_tensor(X_train, device)
    Y_train = to_tensor(Y_train, device)
    X_cal = to_tensor(X_cal, device)
    Y_cal = to_tensor(Y_cal, device)
    X_test = to_tensor(X_test, device)
    Y_test = to_tensor(Y_test, device)
    
    if idx_subset is not None:
        idx_subset = to_tensor(idx_subset, device)
    
    results = []
    
    if parallel and len(alphas) > 1:
        # Parallel evaluation - each alpha trains independently
        # Use ProcessPoolExecutor for better CPU utilization during training
        import multiprocessing as mp
        n_workers = n_workers or min(len(alphas), mp.cpu_count())
        
        eval_fn = partial(
            _evaluate_single_alpha_adaptive,
            base_model=base_model,
            X_test=X_test,
            Y_test=Y_test,
            X_train=X_train,
            Y_train=Y_train,
            X_cal=X_cal,
            Y_cal=Y_cal,
            heuristic=heuristic,
            k=k,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            epochs=epochs,
            step_size=step_size,
            gamma=gamma,
            device=device,
            idx_subset=idx_subset
        )
        
        # Use ThreadPoolExecutor for GPU compatibility
        # (ProcessPoolExecutor would require pickling CUDA tensors)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(eval_fn, alpha) for alpha in alphas]
            
            iterator = futures
            if show_progress:
                iterator = tqdm(futures, desc="Evaluating Adaptive CP", total=len(alphas))
            
            for future in iterator:
                results.append(future.result())
    else:
        # Sequential evaluation
        iterator = alphas
        if show_progress:
            iterator = tqdm(alphas, desc="Evaluating Adaptive CP")
        
        for alpha in iterator:
            result = _evaluate_single_alpha_adaptive(
                alpha, base_model, X_test, Y_test,
                X_train, Y_train, X_cal, Y_cal,
                heuristic, k, hidden_layers,
                learning_rate, epochs, step_size, gamma,
                device, idx_subset
            )
            results.append(result)
    
    return pd.DataFrame(results)
