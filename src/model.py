"""Gaussian Process Regression model for production prediction."""

from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


class GPRModel:
    """Gaussian Process Regression model with uncertainty quantification."""

    def __init__(self) -> None:
        """Initialise GPR model with Matern kernel."""
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            1.0, (1e-3, 1e3), nu=1.5
        ) + WhiteKernel(1e-2, (1e-10, 1))

        self.model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=False
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        """Fit the model with scaling."""
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Fit GPR
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True

    def predict(
        self, X: npt.NDArray[np.float64], return_std: bool = True
    ) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
        """Make predictions with optional uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler_X.transform(X)

        if return_std:
            y_pred_scaled, y_std_scaled = self.model.predict(X_scaled, return_std=True)

            # Transform back to original scale
            y_pred = self.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()

            # Scale standard deviation
            y_std = y_std_scaled * self.scaler_y.scale_[0]
            return y_pred, y_std
        else:
            y_pred_scaled = self.model.predict(X_scaled, return_std=False)
            y_pred = self.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            return y_pred, None

    def evaluate(
        self, X: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred, y_std = self.predict(X, return_std=True)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
        }

        if y_std is not None:
            # Add uncertainty metrics
            lower_95 = y_pred - 1.96 * y_std
            upper_95 = y_pred + 1.96 * y_std
            coverage = np.mean((y_true >= lower_95) & (y_true <= upper_95))
            metrics["coverage_95"] = float(coverage)

        return metrics

    def leave_one_out_cv(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Dict[str, float]]:
        """Perform leave-one-out cross-validation."""
        loo = LeaveOneOut()
        predictions = np.zeros(len(y))
        uncertainties = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create new instance for CV
            cv_model = GPRModel()
            cv_model.fit(X_train, y_train)

            # Predict
            y_pred, y_std = cv_model.predict(X_test, return_std=True)
            predictions[test_idx] = y_pred
            uncertainties[test_idx] = y_std

        # Calculate metrics
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "r2": float(r2_score(y, predictions)),
            "mae": float(np.mean(np.abs(y - predictions))),
            "coverage_95": float(
                np.mean(
                    (y >= predictions - 1.96 * uncertainties)
                    & (y <= predictions + 1.96 * uncertainties)
                )
            ),
        }

        return predictions, uncertainties, metrics
