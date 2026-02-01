"""
Utility functions for device handling, tensor conversions, and k-NN operations.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from typing import Union, Tuple


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Parameters
    ----------
    device : str, torch.device, or None
        Requested device. If None, automatically detects best available device.
        Supported: 'cuda', 'mps', 'cpu', or torch.device objects.
    
    Returns
    -------
    torch.device
        The device to use for computations.
    
    Examples
    --------
    >>> device = get_device()  # Auto-detect
    >>> device = get_device('cuda')  # Force CUDA
    >>> device = get_device('mps')  # Force Apple Silicon GPU
    """
    if device is not None:
        if isinstance(device, str):
            return torch.device(device)
        return device
    
    # Auto-detect best device
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert a PyTorch Tensor or NumPy array to NumPy array.
    
    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input data.
    
    Returns
    -------
    np.ndarray
        NumPy array on CPU.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x: Union[torch.Tensor, np.ndarray], 
              device: torch.device, 
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert NumPy array or PyTorch Tensor to PyTorch Tensor on specified device.
    
    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input data.
    device : torch.device
        Target device.
    dtype : torch.dtype, default=torch.float32
        Data type for the tensor.
    
    Returns
    -------
    torch.Tensor
        Tensor on the specified device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def knn_distance(X_query: np.ndarray, 
                 X_ref: np.ndarray, 
                 k: int, 
                 exclude_self: bool = False) -> np.ndarray:
    """
    Compute mean k-nearest neighbor distance.
    
    Parameters
    ----------
    X_query : np.ndarray, shape (n_query, n_features)
        Query points.
    X_ref : np.ndarray, shape (n_ref, n_features)
        Reference points for k-NN.
    k : int
        Number of nearest neighbors.
    exclude_self : bool, default=False
        If True, excludes the first neighbor (useful when X_query == X_ref).
    
    Returns
    -------
    np.ndarray, shape (n_query,)
        Mean distance to k nearest neighbors for each query point.
    
    Examples
    --------
    >>> X_train = np.random.randn(100, 5)
    >>> X_test = np.random.randn(20, 5)
    >>> distances = knn_distance(X_test, X_train, k=10)
    >>> print(distances.shape)  # (20,)
    """
    # Guard against k being too large
    n_ref = X_ref.shape[0]
    k_actual = min(k + (1 if exclude_self else 0), n_ref)
    
    if k_actual < 1:
        raise ValueError(f"Not enough reference points (n_ref={n_ref}) for k={k}")
    
    nbrs = NearestNeighbors(n_neighbors=k_actual).fit(X_ref)
    distances, _ = nbrs.kneighbors(X_query)
    
    # Exclude self-neighbor if requested
    if exclude_self and distances.shape[1] > 1:
        distances = distances[:, 1:]
    
    return distances.mean(axis=1)


def set_deterministic(seed: int = 0):
    """
    Set random seeds for reproducibility across NumPy, PyTorch, and Python.
    
    Parameters
    ----------
    seed : int, default=0
        Random seed value.
    
    Notes
    -----
    This enables deterministic algorithms in PyTorch which may impact performance.
    """
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Enable deterministic algorithms (may reduce performance)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)


def validate_tensors(X: torch.Tensor, 
                     Y: torch.Tensor, 
                     name_X: str = "X", 
                     name_Y: str = "Y") -> None:
    """
    Validate that input tensors have compatible shapes.
    
    Parameters
    ----------
    X : torch.Tensor
        Feature tensor, shape (n_samples, n_features).
    Y : torch.Tensor
        Target tensor, shape (n_samples,) or (n_samples, n_outputs).
    name_X : str, default="X"
        Name of X tensor for error messages.
    name_Y : str, default="Y"
        Name of Y tensor for error messages.
    
    Raises
    ------
    ValueError
        If tensors have incompatible shapes.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"{name_X} and {name_Y} must have same number of samples. "
            f"Got {name_X}.shape={X.shape}, {name_Y}.shape={Y.shape}"
        )
    
    if X.ndim < 2:
        raise ValueError(f"{name_X} must be 2D, got shape {X.shape}")
    
    if Y.ndim not in [1, 2]:
        raise ValueError(f"{name_Y} must be 1D or 2D, got shape {Y.shape}")


def ensure_2d(Y: torch.Tensor) -> torch.Tensor:
    """
    Ensure target tensor is 2D (n_samples, n_outputs).
    
    Parameters
    ----------
    Y : torch.Tensor
        Target tensor, shape (n_samples,) or (n_samples, n_outputs).
    
    Returns
    -------
    torch.Tensor
        2D tensor of shape (n_samples, n_outputs).
    """
    if Y.ndim == 1:
        return Y.unsqueeze(-1)
    return Y
