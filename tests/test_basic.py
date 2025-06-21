"""Basic tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src import feature_engineering as fe
from src.model import GPRModel


def test_feature_extraction() -> None:
    """Test feature extraction."""
    # Create dummy data
    thermal_data = np.random.rand(100, 100).astype(np.float32) * 2
    mask = np.ones((100, 100), dtype=bool)

    features = fe.extract_thermal_features(thermal_data, mask, prefix="test")

    assert "test_mean" in features
    assert "test_p95" in features
    assert features["test_mean"] > 0


def test_gpr_model() -> None:
    """Test GPR model."""
    # Create dummy data
    X = np.random.randn(20, 5)
    y = X[:, 0] * 2 + np.random.randn(20) * 0.5

    # Create and train model
    model = GPRModel()
    model.fit(X, y)

    # Predict
    y_pred, y_std = model.predict(X)

    assert len(y_pred) == len(y)
    assert len(y_std) == len(y)
    assert all(y_std > 0)

    # Test evaluate
    metrics = model.evaluate(X, y)
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "coverage_95" in metrics


if __name__ == "__main__":
    test_feature_extraction()
    test_gpr_model()
    print("All tests passed!")
