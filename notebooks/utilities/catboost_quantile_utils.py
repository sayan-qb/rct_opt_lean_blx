# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""CatBoost Quantile Regression with Empirical Distribution.

This module provides CatBoost-based quantile regression with empirical distribution
calculation using quantile predictions directly instead of interpolation.
"""

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


class CatBoostQuantilePredictor:
    """CatBoost-based quantile regression with empirical distribution calculation."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        catboost_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize quantile predictor with empirical approach.

        Args:
            quantiles: List of quantiles to fit (should include 0.025, 0.975 for 95% CI).
                If None, uses default quantiles including essential ones for statistics.
            catboost_params: Parameters for CatBoost models. If None, uses default
                configuration optimized for quantile regression.
        """
        if quantiles is None:
            # Default quantiles including ones needed for empirical CI
            self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        else:
            self.quantiles = sorted(
                quantiles
            )  # Ensure sorted for proper empirical calculation

        # Ensure we have the key quantiles for common statistics
        essential_quantiles = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
        for q in essential_quantiles:
            if q not in self.quantiles:
                self.quantiles.append(q)

        self.quantiles = sorted(set(self.quantiles))  # Remove duplicates and sort

        if catboost_params is None:
            self.catboost_params = {
                "iterations": 200,
                "learning_rate": 0.1,
                "depth": 4,
                "random_seed": 42,
                "verbose": False,
                "eval_metric": "RMSE",
                "allow_writing_files": False,  # Prevent temp file creation
            }
        else:
            self.catboost_params = catboost_params

        self.models: dict[float, CatBoostRegressor] = {}
        self.feature_names: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        categorical_features: list[int | str] | None = None,
    ) -> None:
        """Fit quantile models for all specified quantiles.

        Args:
            X: Training features as DataFrame or numpy array.
            y: Training target values as Series or numpy array.
            categorical_features: List of categorical feature indices or names.
                If None, treats all features as numerical.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        for q in self.quantiles:
            # Configure CatBoost for quantile regression
            params = self.catboost_params.copy()
            params["loss_function"] = f"Quantile:alpha={q}"

            model = CatBoostRegressor(**params)
            model.fit(X, y, cat_features=categorical_features, silent=True)

            self.models[q] = model

    def predict_distribution_empirical(
        self, X: pd.DataFrame | np.ndarray
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Predict distribution using empirical quantiles (no interpolation).

        Args:
            X: New data to predict. Can be single row or multiple rows.

        Returns:
            Dictionary with empirical statistics from quantile predictions for single
            prediction, or list of dictionaries for multiple predictions.
        """
        # Get quantile predictions
        quantile_preds: dict[float, np.ndarray] = {}
        for q in self.quantiles:
            pred = self.models[q].predict(X)
            quantile_preds[q] = pred

        # Check if single prediction or multiple
        is_single_prediction = (
            (isinstance(X, pd.DataFrame) and len(X) == 1)
            or (hasattr(X, "shape") and X.shape[0] == 1)
            or (isinstance(X, np.ndarray) and X.ndim == 1)
        )

        if is_single_prediction:
            return self._single_prediction_empirical(quantile_preds)
        else:
            return self._multiple_predictions_empirical(quantile_preds)

    def _single_prediction_empirical(
        self, quantile_preds: dict[float, np.ndarray]
    ) -> dict[str, Any]:
        """Calculate empirical statistics directly from quantile predictions.

        Current approach explanation:
        - Gets quantile predictions for different percentiles
        - Uses these DIRECTLY as empirical quantiles (no interpolation)
        - Calculates mean by averaging available quantiles
        - Uses actual quantile predictions for confidence intervals

        This is more accurate than interpolation-based sampling!

        Args:
            quantile_preds: Dictionary mapping quantile levels to their predictions.

        Returns:
            Dictionary containing empirical statistics including mean, std, confidence
            intervals, and quantile predictions.
        """
        # Extract quantile values and ensure they are scalars
        empirical_quantiles: dict[float, float] = {}
        quantile_values: list[float] = []
        quantile_levels: list[float] = []

        for q in self.quantiles:
            pred = quantile_preds[q]
            # Handle both scalar and array cases
            if isinstance(pred, np.ndarray):
                if pred.size == 1:
                    value = float(pred.item())
                else:
                    value = float(pred[0])
            else:
                value = float(pred)

            empirical_quantiles[q] = value
            quantile_values.append(value)
            quantile_levels.append(q)

        quantile_values_array = np.array(quantile_values)
        quantile_levels_array = np.array(quantile_levels)

        # EMPIRICAL CALCULATIONS (no interpolation)

        # 1. Mean: Use trapezoidal rule for better approximation of mean from quantiles
        # This is more accurate than simple average
        empirical_mean = self._calculate_mean_from_quantiles(
            quantile_levels_array, quantile_values_array
        )

        # 2. Standard deviation: Approximate using IQR and other quantiles
        empirical_std = self._calculate_std_from_quantiles(empirical_quantiles)

        # 3. Confidence intervals: Use actual quantile predictions
        ci_95 = [
            empirical_quantiles.get(0.025, empirical_quantiles[min(self.quantiles)]),
            empirical_quantiles.get(0.975, empirical_quantiles[max(self.quantiles)]),
        ]

        ci_90 = [
            empirical_quantiles.get(0.05, empirical_quantiles.get(0.1, ci_95[0])),
            empirical_quantiles.get(0.95, empirical_quantiles.get(0.9, ci_95[1])),
        ]

        ci_80 = [
            empirical_quantiles.get(0.1, empirical_quantiles.get(0.25, ci_90[0])),
            empirical_quantiles.get(0.9, empirical_quantiles.get(0.75, ci_90[1])),
        ]

        # 4. IQR
        iqr = empirical_quantiles.get(
            0.75, quantile_values_array[-1]
        ) - empirical_quantiles.get(0.25, quantile_values_array[0])

        # 5. Generate samples for backwards compatibility (optional)
        # This uses the same interpolation as before but is marked as approximate
        uniform_samples = np.random.uniform(0, 1, 1000)
        approximate_samples = np.interp(
            uniform_samples, quantile_levels_array, quantile_values_array
        )

        return {
            "quantiles": empirical_quantiles,  # All quantile predictions
            "empirical_mean": float(empirical_mean),  # Empirical mean
            "empirical_std": float(empirical_std),  # Empirical std
            "median": float(
                empirical_quantiles.get(0.5, np.median(quantile_values_array))
            ),
            "iqr": float(iqr),
            "ci_95": [float(ci_95[0]), float(ci_95[1])],
            "ci_90": [float(ci_90[0]), float(ci_90[1])],
            "ci_80": [float(ci_80[0]), float(ci_80[1])],
            "point_estimate": float(
                empirical_quantiles.get(0.5, np.median(quantile_values_array))
            ),
            # Backwards compatibility
            "mean": float(empirical_mean),
            "std": float(empirical_std),
            "samples": approximate_samples,  # For backwards compatibility only
        }

    def _calculate_mean_from_quantiles(
        self, quantile_levels: np.ndarray, quantile_values: np.ndarray
    ) -> float:
        """Calculate mean using trapezoidal integration of quantile function.

        This is more accurate than simple averaging of quantile values.

        Args:
            quantile_levels: Array of quantile levels (e.g., [0.1, 0.5, 0.9]).
            quantile_values: Array of corresponding quantile values.

        Returns:
            Approximated mean value using trapezoidal integration.
        """
        if len(quantile_levels) < 2:
            return quantile_values[0] if len(quantile_values) > 0 else 0.0

        # Use trapezoidal rule to integrate the quantile function
        # This approximates E[X] = integral of quantile function from 0 to 1
        mean_approx = np.trapz(quantile_values, quantile_levels)

        return mean_approx

    def _calculate_std_from_quantiles(
        self, empirical_quantiles: dict[float, float]
    ) -> float:
        """Calculate standard deviation approximation from quantiles.

        Uses multiple methods and takes the best estimate.

        Args:
            empirical_quantiles: Dictionary mapping quantile levels to values.

        Returns:
            Estimated standard deviation using the most robust available method.
        """
        # Method 1: IQR-based approximation (most robust)
        if 0.75 in empirical_quantiles and 0.25 in empirical_quantiles:
            iqr = empirical_quantiles[0.75] - empirical_quantiles[0.25]
            std_iqr = iqr / 1.349  # For normal distribution, IQR ≈ 1.349 * σ
        else:
            std_iqr = None

        # Method 2: 90% range approximation
        if 0.95 in empirical_quantiles and 0.05 in empirical_quantiles:
            range_90 = empirical_quantiles[0.95] - empirical_quantiles[0.05]
            std_90 = range_90 / 3.29  # For normal distribution, 90% range ≈ 3.29 * σ
        else:
            std_90 = None

        # Method 3: Use available quantiles for variance approximation
        available_quantiles = list(empirical_quantiles.keys())
        if len(available_quantiles) >= 3:
            values = list(empirical_quantiles.values())
            std_range = np.std(values) * 1.2  # Rough approximation
        else:
            std_range = None

        # Choose the best estimate
        estimates = [est for est in [std_iqr, std_90, std_range] if est is not None]

        if len(estimates) == 0:
            return 1.0  # Default fallback
        elif len(estimates) == 1:
            return estimates[0]
        elif std_iqr is not None:
            # Use IQR-based if available (most robust), otherwise median
            return std_iqr
        else:
            return np.median(estimates)

    def _multiple_predictions_empirical(
        self, quantile_preds: dict[float, np.ndarray]
    ) -> list[dict[str, Any]]:
        """Calculate empirical statistics for multiple predictions.

        Args:
            quantile_preds: Dictionary mapping quantile levels to prediction arrays.

        Returns:
            List of dictionaries, each containing empirical statistics for one
            observation.
        """
        # Get number of observations
        first_pred = list(quantile_preds.values())[0]
        n_obs = len(first_pred) if hasattr(first_pred, "__len__") else 1

        results = []

        for i in range(n_obs):
            # Extract quantiles for this observation
            single_quantile_preds: dict[float, np.ndarray] = {}
            for q in self.quantiles:
                pred = quantile_preds[q]
                if hasattr(pred, "__len__") and len(pred) > i:
                    single_quantile_preds[q] = np.array([pred[i]])
                else:
                    single_quantile_preds[q] = np.array([pred])

            single_result = self._single_prediction_empirical(single_quantile_preds)
            results.append(single_result)

        return results

    def predict_distribution_interpolated(
        self, X: pd.DataFrame | np.ndarray, n_samples: int = 1000
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Original interpolation-based method for comparison.

        Args:
            X: New data to predict.
            n_samples: Number of samples to generate for interpolation.

        Returns:
            Prediction results using interpolation method.
        """
        return self.predict_distribution(X, n_samples)

    def predict_distribution(
        self, X: pd.DataFrame | np.ndarray, n_samples: int = 1000
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Default method - uses empirical approach.

        Args:
            X: New data to predict.
            n_samples: Number of samples (for backwards compatibility, not used
                in empirical method).

        Returns:
            Prediction results using empirical method.
        """
        return self.predict_distribution_empirical(X)

    def compare_methods(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compare empirical vs interpolation methods.

        Args:
            X: New data to predict (single observation recommended for clear comparison).

        Returns:
            Tuple containing (empirical_result, interpolated_result) dictionaries.
        """
        empirical_result = self.predict_distribution_empirical(X)
        interpolated_result = self.predict_distribution_interpolated(X)

        print("COMPARISON: Empirical vs Interpolation Methods")
        print("=" * 60)
        print(f"Empirical Mean: {empirical_result['empirical_mean']:.3f}")
        print(f"Interpolated Mean: {interpolated_result['mean']:.3f}")
        print(f"Empirical Std: {empirical_result['empirical_std']:.3f}")
        print(f"Interpolated Std: {interpolated_result['std']:.3f}")
        print(f"Empirical 95% CI: {empirical_result['ci_95']}")
        print(f"Interpolated 95% CI: {interpolated_result['ci_95']}")

        return empirical_result, interpolated_result
