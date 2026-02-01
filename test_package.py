"""
Quick test to verify the local_cp package installation and basic functionality.
"""

import torch
import torch.nn as nn
import numpy as np


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from local_cp import CP, AdaptiveCP
        from local_cp.metrics import coverage, sharpness, interval_score
        from local_cp.utils import get_device, to_numpy, to_tensor
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_device_detection():
    """Test device detection."""
    print("\nTesting device detection...")
    
    from local_cp.utils import get_device
    
    device = get_device()
    print(f"  Auto-detected device: {device}")
    
    # Test manual device specification
    cpu_device = get_device('cpu')
    print(f"  CPU device: {cpu_device}")
    
    if torch.cuda.is_available():
        cuda_device = get_device('cuda')
        print(f"  CUDA device: {cuda_device}")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        mps_device = get_device('mps')
        print(f"  MPS device: {mps_device}")
    
    print("✓ Device detection working")
    return True


def test_cp_basic():
    """Test basic CP functionality."""
    print("\nTesting basic CP...")
    
    from local_cp import CP
    from local_cp.metrics import coverage
    
    # Simple model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Generate tiny dataset
    torch.manual_seed(42)
    X = torch.randn(300, 3)
    Y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(300, 1)
    
    X_train, Y_train = X[:180], Y[:180]
    X_cal, Y_cal = X[180:240], Y[180:240]
    X_test, Y_test = X[240:], Y[240:]
    
    # Train model
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    for _ in range(100):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(X_train), Y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    
    # Test CP
    cp = CP(model, device='cpu')
    lower, upper = cp.predict(
        alpha=0.1,
        X_test=X_test,
        X_train=X_train,
        Y_train=Y_train,
        X_cal=X_cal,
        Y_cal=Y_cal,
        heuristic='feature',
        k=10
    )
    
    cov = coverage(Y_test, lower, upper)
    
    print(f"  Coverage: {cov:.2%} (target: 90%)")
    print(f"  Interval shape: {lower.shape}")
    
    if cov >= 0.80:  # Allow some slack for small test set
        print("✓ CP basic test passed")
        return True
    else:
        print("✗ CP coverage too low")
        return False


def test_adaptive_cp():
    """Test Adaptive CP functionality."""
    print("\nTesting Adaptive CP...")
    
    from local_cp import AdaptiveCP
    from local_cp.metrics import coverage
    
    # Simple model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Generate dataset
    torch.manual_seed(42)
    X = torch.randn(300, 3)
    Y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(300, 1)
    
    X_train, Y_train = X[:180], Y[:180]
    X_cal, Y_cal = X[180:240], Y[180:240]
    X_test, Y_test = X[240:], Y[240:]
    
    # Train model
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    for _ in range(100):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(X_train), Y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    
    # Test Adaptive CP
    acp = AdaptiveCP(
        model,
        alpha=0.1,
        device='cpu',
        hidden_layers=(16, 16),
        epochs=500,  # Reduced for testing
        step_size=200
    )
    
    lower, upper = acp.predict(
        alpha=0.1,
        X_test=X_test,
        X_train=X_train,
        Y_train=Y_train,
        X_cal=X_cal,
        Y_cal=Y_cal,
        k=10,
        verbose=False
    )
    
    cov = coverage(Y_test, lower, upper)
    
    print(f"  Coverage: {cov:.2%} (target: 90%)")
    print(f"  Interval shape: {lower.shape}")
    print(f"  Quantile network trained: {acp.quantile_model_trained}")
    
    if cov >= 0.80 and acp.quantile_model_trained:
        print("✓ Adaptive CP test passed")
        return True
    else:
        print("✗ Adaptive CP test failed")
        return False


def test_metrics():
    """Test metrics module."""
    print("\nTesting metrics...")
    
    from local_cp.metrics import (
        coverage, sharpness, interval_score,
        calibration_error, stratified_coverage
    )
    
    # Dummy data
    y_true = torch.randn(100, 1)
    lower = y_true - 1.0
    upper = y_true + 1.0
    
    # Test all metrics
    cov = coverage(y_true, lower, upper)
    sharp = sharpness(lower, upper)
    int_sc = interval_score(y_true, lower, upper, alpha=0.1)
    cal_err = calibration_error(y_true, lower, upper, alpha=0.1)
    
    strata = torch.randint(0, 3, (100,))
    strat_cov = stratified_coverage(y_true, lower, upper, strata)
    
    print(f"  Coverage: {cov:.2%}")
    print(f"  Sharpness: {sharp:.3f}")
    print(f"  Interval Score: {int_sc:.3f}")
    print(f"  Calibration Error: {cal_err:.4f}")
    print(f"  Stratified coverage keys: {list(strat_cov.keys())}")
    
    if 0.95 <= cov <= 1.0 and sharp > 0:
        print("✓ Metrics test passed")
        return True
    else:
        print("✗ Metrics test failed")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("LOCAL CP PACKAGE TEST SUITE")
    print("="*70)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Device Detection", test_device_detection()))
    results.append(("CP Basic", test_cp_basic()))
    results.append(("Adaptive CP", test_adaptive_cp()))
    results.append(("Metrics", test_metrics()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + ("="*70))
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
