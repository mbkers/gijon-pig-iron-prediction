"""Feature engineering functions."""

from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd


def extract_thermal_features(
    thermal_data: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    cloud_mask: Optional[npt.NDArray[np.int32]] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Extract thermal statistics with optional cloud awareness."""
    # Apply mask
    valid_data = thermal_data[mask]

    features: Dict[str, float] = {}

    if cloud_mask is not None:
        # Cloud-aware extraction
        cloud_masked = cloud_mask[mask]
        clear_data = valid_data[cloud_masked == 0]

        if len(clear_data) > 50:  # Minimum clear pixels
            features[f"{prefix}_clear_mean"] = float(np.mean(clear_data))
            features[f"{prefix}_clear_max"] = float(np.max(clear_data))
            features[f"{prefix}_clear_p95"] = float(np.percentile(clear_data, 95))
            features[f"{prefix}_clear_p99"] = float(np.percentile(clear_data, 99))
            features[f"{prefix}_clear_pixels"] = float(len(clear_data))
            features[f"{prefix}_clear_ratio"] = float(len(clear_data) / len(valid_data))

    # Standard features (always computed)
    if len(valid_data) > 0:
        features[f"{prefix}_mean"] = float(np.mean(valid_data))
        features[f"{prefix}_std"] = float(np.std(valid_data))
        features[f"{prefix}_max"] = float(np.max(valid_data))
        features[f"{prefix}_p95"] = float(np.percentile(valid_data, 95))
        features[f"{prefix}_p99"] = float(np.percentile(valid_data, 99))

        # Count pixels above thresholds
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            features[f"{prefix}_above_{threshold}"] = float(
                np.sum(valid_data > threshold)
            )

    return features


def create_monthly_features(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily features to monthly."""
    # Group by month
    monthly = daily_features.groupby(pd.Grouper(freq="MS")).agg(
        {
            col: ["mean", "max", "std"]  # if "mean" in col or "max" in col else "mean"
            for col in daily_features.columns
        }
    )

    # Flatten column names
    monthly.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in monthly.columns
    ]

    # Add observation count
    monthly["obs_count"] = daily_features.groupby(pd.Grouper(freq="MS")).size()

    # Filter months with too few observations
    return monthly[monthly["obs_count"] >= 5]


def select_top_features(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    feature_names: List[str],
    n_features: int = 10,
) -> List[str]:
    """Select top features by correlation."""
    # Remove features with zero variance to avoid division by zero
    variances = np.var(X, axis=0)
    non_constant_mask = variances > 1e-10

    if not np.any(non_constant_mask):
        # If all features are constant, return empty list
        return []

    # Filter features and names
    X_filtered = X[:, non_constant_mask]
    feature_names_filtered = [
        name for name, keep in zip(feature_names, non_constant_mask) if keep
    ]

    # Calculate correlations only for non-constant features
    correlations = (
        pd.DataFrame(X_filtered, columns=feature_names_filtered)
        .corrwith(pd.Series(y))
        .abs()
    )

    # Drop any NaN correlations (shouldn't happen after filtering, but safe practice)
    correlations = correlations.dropna()

    # Select top features
    n_features = min(n_features, len(correlations))
    top_indices = correlations.nlargest(n_features).index
    return top_indices.tolist()
