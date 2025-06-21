"""Utility functions for visualisation and saving results."""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure


def plot_predictions(
    dates: pd.DatetimeIndex,
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    y_std: Optional[npt.NDArray[np.float64]] = None,
    title: str = "GPR Predictions with Uncertainty",
    save_path: Optional[str] = None,
    prediction_type: str = "test",  # "test" or "loo"
) -> Figure:
    """Plot time series predictions with uncertainty."""
    # Set color scheme based on prediction type
    color = "blue" if prediction_type == "test" else "red"
    label = "Predicted" if prediction_type == "test" else "LOO-CV Predicted"

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual values
    ax.plot(
        dates, y_true, "ko-", label="Actual", markersize=7, linewidth=1.5, alpha=0.8
    )

    # Plot predictions
    ax.plot(
        dates,
        y_pred,
        color=color,
        marker="o",
        linestyle="-",
        label=label,
        linewidth=2,
        markersize=5,
        alpha=0.9,
    )

    # Add uncertainty bands if provided
    if y_std is not None:
        ax.fill_between(
            dates,
            y_pred - 1.96 * y_std,
            y_pred + 1.96 * y_std,
            alpha=0.2,
            color=color,
            label="95% PI",
        )

    # Styling
    # ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Production (1000 tons)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Format x-axis
    plt.xticks(rotation=45)

    # Add subtle background
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_actual_vs_predicted(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    y_std: Optional[npt.NDArray[np.float64]] = None,
    save_path: Optional[str] = None,
    prediction_type: str = "test",  # "test" or "loo"
) -> Figure:
    """Create comprehensive actual vs predicted plots with residuals."""
    from sklearn.metrics import r2_score

    # Set color scheme
    color = "blue" if prediction_type == "test" else "red"
    title_prefix = "Test Set" if prediction_type == "test" else "LOO-CV"

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: Actual vs Predicted
    ax1 = axes[0]

    # Main scatter plot
    ax1.scatter(
        y_true, y_pred, alpha=0.6, s=60, color=color, edgecolors="black", linewidth=0.5
    )

    # Add error bars if std provided
    if y_std is not None:
        ax1.errorbar(
            y_true,
            y_pred,
            yerr=1.96 * y_std,
            fmt="none",
            alpha=0.3,
            ecolor="gray",
            capsize=3,
            capthick=1,
        )

    # Diagonal reference line
    min_val = min(y_true.min(), y_pred.min()) * 0.95
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=2)

    # Styling
    ax1.set_xlabel("Actual Production (1000 tons)", fontsize=12)
    ax1.set_ylabel(f"{title_prefix} Predicted Production (1000 tons)", fontsize=12)
    ax1.set_title(f"{title_prefix} Predictions", fontsize=14, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_facecolor("#f8f9fa")

    # Add R² annotation
    ax1.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
        ),
        fontsize=12,
        fontweight="bold",
    )

    # Right panel: Residual Plot
    ax2 = axes[1]

    # Residual scatter
    ax2.scatter(
        y_pred,
        residuals,
        alpha=0.6,
        s=60,
        color=color,
        edgecolors="black",
        linewidth=0.5,
    )
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=2)

    # Add ±1 std bands if available
    if y_std is not None:
        ax2.fill_between(
            sorted(y_pred),
            -1.96 * np.mean(y_std),
            1.96 * np.mean(y_std),
            alpha=0.1,
            color=color,
            label="±1.96σ",
        )

    # Styling
    ax2.set_xlabel(f"{title_prefix} Predicted Production (1000 tons)", fontsize=12)
    ax2.set_ylabel("Residuals (1000 tons)", fontsize=12)
    ax2.set_title("Residual Analysis", fontsize=14, fontweight="bold", pad=15)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_facecolor("#f8f9fa")

    # Add residual statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    ax2.text(
        0.95,
        0.95,
        f"RMSE = {rmse:.1f}\nMAE = {mae:.1f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
        ),
        fontsize=10,
    )

    # Overall styling
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_results(
    results_dir: str,
    test_predictions_df: pd.DataFrame,
    test_metrics: Dict[str, float],
    feature_importance: Optional[pd.DataFrame] = None,
    loo_predictions_df: Optional[pd.DataFrame] = None,
    loo_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save model results to files."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Save test predictions
    test_predictions_df.to_csv(
        results_path / "csv" / "test_predictions.csv", index=False
    )

    # Save test metrics
    pd.DataFrame([test_metrics]).to_csv(
        results_path / "csv" / "test_metrics.csv", index=False
    )

    # Save LOO-CV predictions and metrics if provided
    if loo_predictions_df is not None:
        loo_predictions_df.to_csv(
            results_path / "csv" / "loo_predictions.csv", index=False
        )

    if loo_metrics is not None:
        pd.DataFrame([loo_metrics]).to_csv(
            results_path / "csv" / "loo_metrics.csv", index=False
        )

    # Save feature importance if provided
    if feature_importance is not None:
        feature_importance.to_csv(
            results_path / "csv" / "feature_importance.csv", index=False
        )

    # Create summary
    with open(results_path / "summary.txt", "w") as f:
        f.write("GPR Model Performance Summary\n")
        f.write("=" * 30 + "\n\n")

        f.write("Test Set Performance:\n")
        f.write("-" * 20 + "\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric}: {value:.3f}\n")

        if loo_metrics is not None:
            f.write("\nLOO-CV Performance:\n")
            f.write("-" * 20 + "\n")
            for metric, value in loo_metrics.items():
                f.write(f"{metric}: {value:.3f}\n")
