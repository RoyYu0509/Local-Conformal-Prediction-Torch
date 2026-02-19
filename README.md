# Local CP: Conformal Prediction for PyTorch

[![arXiv](https://img.shields.io/badge/arXiv-2509.13717-b31b1b.svg)](https://arxiv.org/abs/2509.13717)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch package for **uncertainty quantification** in regression models using conformal prediction. Provides distribution-free, finite-sample coverage guarantees with support for **CUDA**, **MPS** (Apple Silicon), and **CPU**.

<img width="758" height="660" alt="image" src="https://github.com/user-attachments/assets/76cd28da-9884-4dbb-badf-64b687e9995d" />



## Features

- 🎯 **Scaled Conformal Prediction (CP)**: Distance-based scaling for adaptive intervals
- 🌍 **Local CP (Adaptive CP)**: Learned quantile networks for spatially varying uncertainty
- 🚀 **Device Support**: CUDA, MPS (Apple M1/M2/M3), and CPU
- 📊 **Comprehensive Metrics**: Coverage, sharpness, interval score, and more
- 🔧 **Model-Agnostic**: Works with any PyTorch regression model
- 📖 **Well-Documented**: Extensive docstrings and examples

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [Heuristics](#heuristics)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Method Overview](#method-overview)
- [Advanced Usage](#advanced-usage)
- [Performance & Troubleshooting](#performance--troubleshooting)
- [Citation](#citation)
- [License](#license)

## Installation

### Using pip

```bash
cd local_cp
pip install -e .
```

### Using uv (Recommended)

Install uv package dependency management library:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install all dependencies:
```bash
uv sync
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy ≥ 1.20
- scikit-learn ≥ 1.0

### Verify Installation

Run the test suite:
```bash
uv run python test_package.py
# or with pip
python test_package.py
```

## Quick Start

### 1. Scaled Conformal Prediction (CP)

```python
import torch
from local_cp import CP
from local_cp.metrics import coverage, sharpness

# Your pre-trained PyTorch model
model = YourNeuralNetwork()
# ... train your model ...

# Wrap with CP
cp = CP(model, device='cuda')

# Get prediction intervals
lower, upper = cp.predict(
    alpha=0.1,          # 90% coverage target
    X_test=X_test,
    X_train=X_train,
    Y_train=Y_train,
    X_cal=X_cal,
    Y_cal=Y_cal,
    heuristic='feature',  # or 'latent', 'raw_std'
    k=20                  # number of nearest neighbors
)

# Evaluate
print(f"Coverage: {coverage(Y_test, lower, upper):.2%}")
print(f"Average width: {sharpness(lower, upper):.3f}")
```

### 2. Locally Adaptive CP (Local CP)

```python
from local_cp import AdaptiveCP

# Create adaptive CP with learned quantile network
acp = AdaptiveCP(
    model,
    alpha=0.1,
    device='cuda',
    heuristic='feature',
    hidden_layers=(128, 128, 128),
    epochs=15000
)

# First call trains the quantile network
lower, upper = acp.predict(
    alpha=0.1,
    X_test=X_test,
    X_train=X_train,
    Y_train=Y_train,
    X_cal=X_cal,
    Y_cal=Y_cal,
    k=20
)

# Subsequent calls reuse the trained network
lower2, upper2 = acp.predict(0.1, X_test2, X_train, Y_train, X_cal, Y_cal)
```

## Heuristics

Three scaling heuristics are available:

| Heuristic | Description | When to Use |
|-----------|-------------|-------------|
| **`feature`** | k-NN distance in input (feature) space | Default choice, works well for most problems |
| **`latent`** | k-NN distance in hidden layer space | When learned representations are informative |
| **`raw_std`** | Model's own predictive uncertainty | For ensemble or Bayesian models with `predict()` method |

### Using `feature` Heuristic
No special requirements. Works with any PyTorch model.

### Using `latent` Heuristic

Your model's `forward()` method should support `return_hidden=True`:

```python
class MyModel(nn.Module):
    def forward(self, x, return_hidden=False):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        output = self.output_layer(h2)
        
        if return_hidden:
            return output, h2  # Return (output, hidden_representation)
        return output
```

### Using `raw_std` Heuristic

Your model should have a `predict()` method returning intervals:

```python
class MyEnsembleModel(nn.Module):
    def predict(self, alpha, X):
        # Return lower and upper bounds from ensemble
        predictions = self.get_ensemble_predictions(X)
        lower = predictions.quantile(alpha / 2, dim=0)
        upper = predictions.quantile(1 - alpha / 2, dim=0)
        return lower, upper
```

## API Reference

### `CP`

**Scaled Conformal Prediction**

```python
CP(model, device=None)
```

**Parameters:**
- `model` (nn.Module): Pre-trained PyTorch regression model
- `device` (str, torch.device, None): Computation device ('cuda', 'mps', 'cpu')

**Methods:**
- `predict(alpha, X_test, X_train, Y_train, X_cal, Y_cal, heuristic='feature', k=10, **kwargs)`: Returns `(lower, upper)` prediction intervals

---

### `AdaptiveCP`

**Locally Adaptive Conformal Prediction with quantile network**

```python
AdaptiveCP(
    model,
    alpha=0.05,
    device=None,
    heuristic='feature',
    hidden_layers=(64, 64, 64),
    learning_rate=5e-4,
    epochs=20000,
    step_size=5000,
    gamma=0.5,
    quant_seed=12345
)
```

**Parameters:**
- `model` (nn.Module): Pre-trained PyTorch regression model
- `alpha` (float): Target miscoverage level
- `device` (str, torch.device, None): Computation device
- `heuristic` (str): Scaling heuristic ('feature', 'latent', 'raw_std')
- `hidden_layers` (tuple): Quantile network architecture
- `learning_rate` (float): Learning rate for quantile network
- `epochs` (int): Training epochs for quantile network
- `step_size` (int): LR scheduler step size
- `gamma` (float): LR decay factor
- `quant_seed` (int): Random seed for reproducible quantile network initialization

**Methods:**
- `predict(alpha, X_test, X_train, Y_train, X_cal=None, Y_cal=None, **kwargs)`: Returns `(lower, upper)` adaptive prediction intervals


---
### Metrics

```python
from local_cp.metrics import (
    coverage,
    sharpness,
    interval_score,
    calibration_error,
    stratified_coverage,
    conditional_coverage_diagnostic
)
```

**Functions:**
- `coverage(y_true, lower, upper, reduction='mean')`: Empirical coverage probability
- `sharpness(lower, upper, reduction='mean')`: Average interval width
- `interval_score(y_true, lower, upper, alpha, reduction='mean')`: Proper scoring rule
- `calibration_error(y_true, lower, upper, alpha)`: Absolute calibration error

## Package Structure
```
local_cp/
├── local_cp/              # Main package
│   ├── __init__.py       # Package initialization
│   ├── cp.py             # Scaled Conformal Prediction
│   ├── adaptive_cp.py    # Local CP (Adaptive CP)
│   ├── metrics.py        # Evaluation metrics
│   ├── batch_eval.py     # Batch evaluation utilities
│   └── utils.py          # Utility functions
├── examples/             # Usage examples
│   ├── basic_usage.py
│   ├── adaptive_cp_demo.py
│   ├── metrics_demo.py
│   ├── device_comparison.py
│   └── tutorial_conformal_prediction.ipynb
├── setup.py              # Package installation
├── requirements.txt      # Dependencies
├── README.md            # This file
├── LICENSE              # MIT License
└── test_package.py      # Test suite
```

## Examples

See the [`examples/`](examples/) directory for complete working examples:

- [`basic_usage.py`](examples/basic_usage.py): Simple regression with CP
- [`adaptive_cp_demo.py`](examples/adaptive_cp_demo.py): Local CP on synthetic data
- [`device_comparison.py`](examples/device_comparison.py): CUDA vs MPS vs CPU benchmarks
- [`metrics_demo.py`](examples/metrics_demo.py): Using evaluation metrics
- [`tutorial_conformal_prediction.ipynb`](examples/tutorial_conformal_prediction.ipynb): Comprehensive Jupyter notebook tutorial

## Method Overview

### Scaled Conformal Prediction (CP)

Split conformal prediction with distance-based scaling:

1. **Calibration**: Compute scaled residuals on calibration set
   ```
   Score_i = |Y_cal_i - Ŷ_cal_i| / scale(X_cal_i)
   ```

2. **Quantile**: Find (1-α)-quantile of calibration scores
   ```
   q̂ = Quantile_{(1-α)}(Scores)
   ```

3. **Prediction**: Build intervals on test set
   ```
   [Ŷ_test - q̂·scale(X_test), Ŷ_test + q̂·scale(X_test)]
   ```

### Local CP (Adaptive CP)

Learns spatially varying quantiles via a neural network:

1. **Train Quantile Network**: Learn q̂(x) on training set using pinball loss
2. **Calibrate**: Compute conformal multiplier c on calibration set
3. **Predict**: Adaptive intervals with learned local quantiles
   ```
   [Ŷ_test - c·q̂(X_test)·scale(X_test), Ŷ_test + c·q̂(X_test)·scale(X_test)]
   ```

***Note: You can replace the` quantile network kernel` with your own PyTorch quantile model by subclassing `AdaptiveCP` and overriding the `build_quantile_network()` method to fit the local quantile $\hat q$.***

**Paper**: https://arxiv.org/abs/2509.13717

## Performance & Troubleshooting

### Coverage Guarantees
- Split conformal prediction provides marginal coverage: E[Coverage] ≥ 1 - α
- Finite-sample guarantees (not asymptotic)
- Distribution-free (no assumptions on data distribution)

### Computational Complexity

Let n_train, n_cal, n_test be the sizes of training, calibration, and test sets respectively. Let k be the number of neighbors in k-NN.

#### Standard CP
- Training Cost: O(1) (no training)
- Inference Cost: O(n_cal + n_test × k) for k-NN search and interval construction

#### Local CP
- Training Cost: O(n_train × epochs) for quantile network training
- Inference Cost: O(n_cal + n_test × k) same as CP

### Common Issues

#### Issue: "Model must support return_hidden=True"
**Solution**: Add `return_hidden` parameter to your model's `forward()` method or use `'feature'` heuristic instead.

#### Issue: Low coverage
**Possible causes**:
- Calibration set too small (increase to at least 100 samples)
- Model poorly calibrated (retrain with better regularization)
- Data drift between calibration and test sets

#### Issue: Intervals too wide
**Possible causes**:
- k value too small (try increasing k)
- Using wrong heuristic (experiment with different heuristics)
- Try Adaptive CP for spatially varying uncertainty

#### Issue: CUDA out of memory
**Solutions**:
- Reduce batch size
- Use CPU or MPS device
- Process test set in smaller chunks
- Reduce quantile network size for Adaptive CP

## Citation

If you use this package in your research, please cite:

```bibtex
@article{yu2025conformal,
  title={A Conformal Prediction Framework for Uncertainty Quantification in Physics-Informed Neural Networks},
  author={Yu, Yifan and Ho, Cheuk Hin and Wang, Yangshuai},
  journal={arXiv preprint arXiv:2509.13717},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.


## Authors
- **Yifan Yu**

**Version**: 0.1.0  
**Last Updated**: February 2025


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This package implements methods described in our paper on conformal prediction for physics-informed neural networks. The algorithms are designed to be general-purpose and work with any PyTorch regression model.
