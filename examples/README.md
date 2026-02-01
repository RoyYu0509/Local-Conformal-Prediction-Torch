# Local CP Examples

This directory contains comprehensive examples demonstrating the Local CP package.

## Examples

### 1. Basic Usage (`basic_usage.py`)

Simple introduction to scaled conformal prediction:
- Generate synthetic regression data
- Train a neural network
- Apply CP with different heuristics
- Evaluate coverage and sharpness

**Run:**
```bash
python basic_usage.py
```

### 2. Adaptive CP Demo (`adaptive_cp_demo.py`)

Demonstrates locally adaptive conformal prediction:
- Heteroscedastic data with spatially varying noise
- Quantile network training
- Conditional coverage analysis
- Comparison of interval widths in different regions

**Run:**
```bash
python adaptive_cp_demo.py
```

### 3. Metrics Demo (`metrics_demo.py`)

Showcases all evaluation metrics:
- Coverage, sharpness, interval score
- Calibration error
- Stratified coverage
- Conditional coverage diagnostics
- Trade-off between coverage and sharpness

**Run:**
```bash
python metrics_demo.py
```

### 4. Device Comparison (`device_comparison.py`)

Benchmarks performance across devices:
- CUDA vs MPS vs CPU comparison
- Timing for CP and Adaptive CP
- Memory usage analysis

**Run:**
```bash
python device_comparison.py
```

## Requirements

All examples require:
- PyTorch
- NumPy
- scikit-learn
- local_cp package (install with `pip install -e ..` from package root)

## Tips

- Start with `basic_usage.py` to understand the core concepts
- Use `adaptive_cp_demo.py` to see the benefits of Local CP
- Check `device_comparison.py` to optimize for your hardware
- Refer to `metrics_demo.py` for comprehensive evaluation
