"""Main script for pig iron production prediction."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from src import data_loader as dl
from src import feature_engineering as fe
from src import utils
from src.model import GPRModel


def extract_daily_features(
    base_path: str, indices: List[str] = ["TAI", "NHI_SWIR", "NHI_SWNIR"]
) -> pd.DataFrame:
    """Extract daily features from thermal indices."""
    # Load perimeter mask
    perimeter_mask = dl.load_perimeter_mask(base_path)

    all_features = []

    for index_type in indices:
        print(f"Processing {index_type}...")
        dates = dl.get_available_dates(base_path, index_type)

        for date in dates:
            # Load thermal data
            thermal_data = dl.load_thermal_index(base_path, index_type, date)
            if thermal_data is None:
                continue

            cloud_mask = dl.load_cloud_mask(base_path, date)

            # Extract features
            features = fe.extract_thermal_features(
                thermal_data, perimeter_mask, cloud_mask, index_type
            )

            if features:
                features["date"] = pd.to_datetime(date, format="%Y%m%d")
                all_features.append(features)

    # Create DataFrame
    df = pd.DataFrame(all_features)
    return df.set_index("date").sort_index()


def main(data_path: str, results_dir: str = "./results") -> None:
    """Run the pig iron production prediction pipeline."""
    print("Starting pig iron production prediction...")

    # Create results directories
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{results_dir}/csv").mkdir(exist_ok=True)
    Path(f"{results_dir}/figures").mkdir(exist_ok=True)

    # Step 1: Extract daily features
    print("\nStep 1: Extracting daily features from satellite data...")
    daily_features = extract_daily_features(data_path)
    daily_features.to_csv(f"{results_dir}/csv/daily_features.csv")
    print(f"Done: Extracted features for {daily_features.index.nunique()} days")

    # Step 2: Aggregate to monthly level
    print("\nStep 2: Aggregating daily features to monthly...")
    monthly_features = fe.create_monthly_features(daily_features)
    print(f"Done: Created monthly features for {len(monthly_features)} months")

    # Step 3: Load and merge production data
    print("\nStep 3: Loading and aligning production data...")
    production = dl.load_production_data(data_path)

    # Align dates
    monthly_features.index = pd.to_datetime(monthly_features.index.strftime("%Y-%m-01"))
    merged_data = monthly_features.join(production, how="inner")
    merged_data.to_csv(f"{results_dir}/csv/modeling_data.csv")
    print(f"Done: Merged data contains {len(merged_data)} months")

    # Step 4: Prepare feature matrix
    print("\nStep 4: Preparing feature matrix...")
    feature_cols = [
        col for col in merged_data.columns if col not in ["Value", "obs_count"]
    ]

    X = merged_data[feature_cols].values
    y = merged_data["Value"].values
    dates = merged_data.index

    print(f"Done: Feature matrix shape: {X.shape}")

    # Step 5: Feature selection
    print("\nStep 5: Selecting most informative features...")
    n_features = min(15, len(feature_cols))
    top_features = fe.select_top_features(X, y, feature_cols, n_features)
    print(f"Done: Selected {len(top_features)} features")

    X_selected = merged_data[top_features].values

    # Step 6: Train-test split
    print(f"\nStep 6: Splitting data...")
    split_idx = int(len(X_selected) * 0.8)
    X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Step 7: Train model
    print("\nStep 7: Training Gaussian Process Regression model...")
    model = GPRModel()
    model.fit(X_train, y_train)
    print("Model training complete")

    # Step 8: Test set evaluation
    print("\nStep 8: Evaluating on test set...")
    y_pred, y_std = model.predict(X_test)
    test_metrics = model.evaluate(X_test, y_test)

    print("Test Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Step 9: Leave-one-out cross-validation
    print("\nStep 9: Running leave-one-out cross-validation...")
    loo_pred, loo_std, loo_metrics = model.leave_one_out_cv(X_selected, y)

    print("LOO-CV Performance:")
    for metric, value in loo_metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Step 10: Save results
    print("\nStep 10: Saving results...")

    # Create prediction DataFrames
    test_predictions_df = pd.DataFrame(
        {
            "date": dates_test,
            "actual": y_test,
            "predicted": y_pred,
            "std": y_std,
            "lower_95": y_pred - 1.96 * y_std,
            "upper_95": y_pred + 1.96 * y_std,
        }
    )

    loo_predictions_df = pd.DataFrame(
        {
            "date": dates,
            "actual": y,
            "predicted": loo_pred,
            "std": loo_std,
            "lower_95": loo_pred - 1.96 * loo_std,
            "upper_95": loo_pred + 1.96 * loo_std,
        }
    )

    # Save feature importance
    feature_importance = pd.DataFrame(
        {"feature": top_features, "index": list(range(len(top_features)))}
    )

    # Save all CSV files
    utils.save_results(
        results_dir,
        test_predictions_df,
        test_metrics,
        feature_importance,
        loo_predictions_df,
        loo_metrics,
    )
    print("Done: Results saved to CSV files")

    # Step 11: Create visualisations
    print("\nStep 11: Creating visualisations...")

    # Test predictions time series
    utils.plot_predictions(
        dates_test,
        y_test,
        y_pred,
        y_std,
        title="Test Set Predictions",
        save_path=f"{results_dir}/figures/test_predictions.png",
        prediction_type="test",
    )

    # LOO-CV predictions time series
    utils.plot_predictions(
        dates,
        y,
        loo_pred,
        loo_std,
        title="LOO-CV Predictions",
        save_path=f"{results_dir}/figures/loo_predictions.png",
        prediction_type="loo",
    )

    # Test actual vs predicted
    utils.plot_actual_vs_predicted(
        y_test,
        y_pred,
        y_std,
        save_path=f"{results_dir}/figures/test_actual_vs_predicted.png",
        prediction_type="test",
    )

    # LOO-CV actual vs predicted
    utils.plot_actual_vs_predicted(
        y,
        loo_pred,
        loo_std,
        save_path=f"{results_dir}/figures/loo_actual_vs_predicted.png",
        prediction_type="loo",
    )

    print("Done: Visualisations saved")

    print(f"\nPipeline complete! Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pig Iron Production Prediction")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to Gijon data  directory"
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results", help="Results directory"
    )

    args = parser.parse_args()
    main(args.data_path, args.results_dir)
