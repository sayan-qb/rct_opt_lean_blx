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
"""RCT Lean Bayesian Neural Network Helper functions."""

from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


def predict_bayesian_with_transforms(
    trace: az.InferenceData,
    new_data: dict[str, float],
    feature_transforms: dict[str, dict[str, float]] | None = None,
    n_samples: int = 1000,
    intervention_outcome_col: str = "intervention_outcome",
    bayesian_features: list[str] | None = None,
) -> np.ndarray:
    """Sample from the posterior to simulate a predictive distribution.

    Handles standardization if feature_transforms are provided.

    Args:
        trace: Bayesian model trace from PyMC sampling.
        new_data: Dictionary with keys: intervention_outcome and bayesian_features.
        feature_transforms: Dictionary with standardization parameters (mean, std)
            for each feature. If None, no standardization is applied.
        n_samples: Number of samples to draw from posterior predictive distribution.
        intervention_outcome_col: Column name for intervention outcome variable.
        bayesian_features: List of feature names used in the Bayesian model.

    Returns:
        Array of prediction samples from the posterior predictive distribution.
    """
    if bayesian_features is None:
        bayesian_features = []

    # Apply standardization if transforms are provided
    if feature_transforms is not None:
        standardized_data = new_data.copy()
        for feature, transform in feature_transforms.items():
            if feature in standardized_data:
                mean_val = transform["mean"]
                std_val = transform["std"]
                standardized_data[feature] = (
                    standardized_data[feature] - mean_val
                ) / std_val
        prediction_data = standardized_data
    else:
        prediction_data = new_data

    posterior = trace.posterior
    n_chains = posterior.dims["chain"]
    n_draws = posterior.dims["draw"]
    predictions = []

    for _i in range(n_samples):
        chain_idx = np.random.randint(0, n_chains)
        draw_idx = np.random.randint(0, n_draws)

        # Base linear predictor: intercept + beta_intervention * intervention_outcome
        intercept_val = posterior["intercept"][chain_idx, draw_idx].values
        beta_int_val = posterior["beta_intervention"][chain_idx, draw_idx].values
        pred = intercept_val + beta_int_val * prediction_data[intervention_outcome_col]

        # Add contribution from each feature
        for f in bayesian_features:
            beta_f = posterior[f"beta_{f}"][chain_idx, draw_idx].values
            pred += beta_f * prediction_data[f]

        # Add noise
        sigma_val = posterior["sigma"][chain_idx, draw_idx].values
        pred_sample = np.random.normal(pred, sigma_val)
        predictions.append(pred_sample)

    return np.array(predictions)


def compare_linear_vs_bnn_predictions(
    linear_trace: az.InferenceData,
    bnn_trace: az.InferenceData,
    bnn_transforms: dict[str, dict[str, float]],
    new_data: dict[str, float],
    features: list[str],
    intervention_outcome_col: str,
    linear_transforms: dict[str, dict[str, float]] | None = None,
    hidden_units: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Compare predictions from linear model vs BNN for the same input data.

    Args:
        linear_trace: MCMC trace from linear Bayesian model.
        bnn_trace: MCMC trace from Bayesian Neural Network.
        bnn_transforms: Feature standardization parameters for BNN.
        new_data: Input data dictionary for prediction.
        features: List of feature names used in models.
        intervention_outcome_col: Column name for intervention outcome.
        linear_transforms: Feature standardization parameters for linear model.
            If None, no standardization is applied.
        hidden_units: Architecture of BNN hidden layers.

    Returns:
        Dictionary with prediction arrays for both models containing keys:
        'linear' and 'bnn' with corresponding prediction arrays.
    """
    if hidden_units is None:
        hidden_units = [10, 5]

    # Linear model predictions
    if linear_transforms is not None:
        linear_preds = predict_bayesian_with_transforms(
            linear_trace, new_data, linear_transforms, n_samples=1000
        )
    else:
        linear_preds = predict_bayesian_with_transforms(
            linear_trace, new_data, n_samples=1000
        )

    # BNN predictions
    bnn_preds = predict_bnn_with_transforms(
        bnn_trace,
        new_data,
        bnn_transforms,
        features,
        intervention_outcome_col,
        hidden_units,
        n_samples=1000,
    )

    return {"linear": linear_preds, "bnn": bnn_preds}


def build_bayesian_neural_network(
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    intervention_outcome_col: str,
    hidden_units: list[int] | None = None,
    standardize: bool = True,
    use_informed_priors: bool = True,
) -> tuple[pm.Model, az.InferenceData, dict[str, dict[str, float]]]:
    """Build Bayesian Neural Network with PyMC for control arm prediction.

    Args:
        data: DataFrame with training data.
        features: List of feature column names to use as inputs.
        target_col: Target variable column name.
        intervention_outcome_col: Intervention outcome column name.
        hidden_units: List of hidden layer sizes [layer1_size, layer2_size, ...].
        standardize: Whether to standardize features and target variable.
        use_informed_priors: Whether to use informed priors based on data statistics.

    Returns:
        Tuple containing:
        - PyMC model object
        - MCMC trace (InferenceData)
        - Feature transforms dictionary (standardization parameters if standardize=True)
    """
    if hidden_units is None:
        hidden_units = [10, 5]

    return _build_bnn_core(
        data,
        features,
        target_col,
        intervention_outcome_col,
        hidden_units,
        standardize,
        use_informed_priors,
    )


def _build_bnn_core(
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    intervention_outcome_col: str,
    hidden_units: list[int],
    standardize: bool,
    use_informed_priors: bool,
) -> tuple[pm.Model, az.InferenceData, dict[str, dict[str, float]]]:
    """Core BNN building logic extracted to reduce statement count."""
    print(
        f"Building BNN with architecture: {len(features) + 1} -> {' -> '.join(map(str, hidden_units))} -> 1"
    )

    # Prepare data
    model_data = data.copy()
    feature_transforms = {}

    if standardize:
        feature_transforms = _standardize_features(
            data, model_data, features, intervention_outcome_col, target_col
        )

    # Prepare input features
    X_features = model_data[features].values
    X_intervention = model_data[intervention_outcome_col].values.reshape(-1, 1)
    X_combined = np.concatenate([X_features, X_intervention], axis=1)
    y = model_data[target_col].values

    n_samples, n_input = X_combined.shape

    # Calculate informed priors
    weight_sigma, bias_sigma, noise_sigma = _calculate_priors(
        use_informed_priors, n_input, standardize, data, target_col
    )

    with pm.Model() as bnn_model:
        X_tensor = pt.as_tensor_variable(X_combined)
        layer_sizes = [n_input] + hidden_units + [1]
        current_input = X_tensor

        for i, (_in_size, out_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
        ):
            W = pm.Normal(
                f"W_{i}", mu=0, sigma=weight_sigma, shape=(_in_size, out_size)
            )
            b = pm.Normal(f"b_{i}", mu=0, sigma=bias_sigma, shape=out_size)

            linear_out = pt.dot(current_input, W) + b

            if i < len(layer_sizes) - 2:
                current_input = pt.maximum(linear_out, 0)
            else:
                network_output = linear_out.flatten()

        sigma = pm.HalfNormal("sigma", sigma=noise_sigma)
        _y_obs = pm.Normal("y_obs", mu=network_output, sigma=sigma, observed=y)

        trace = _sample_bnn_posterior(bnn_model)

    return bnn_model, trace, feature_transforms


def _standardize_features(
    data: pd.DataFrame,
    model_data: pd.DataFrame,
    features: list[str],
    intervention_outcome_col: str,
    target_col: str,
) -> dict[str, dict[str, float]]:
    """Standardize features and target variable."""
    print("Standardizing features...")
    feature_transforms = {}

    # Standardize input features
    for feature in features:
        if feature in data.columns:
            mean_val = data[feature].mean()
            std_val = data[feature].std()
            if std_val > 0:
                model_data[feature] = (data[feature] - mean_val) / std_val
            else:
                model_data[feature] = data[feature] - mean_val
            feature_transforms[feature] = {
                "mean": mean_val,
                "std": std_val if std_val > 0 else 1.0,
            }

    # Standardize intervention outcome
    int_mean = data[intervention_outcome_col].mean()
    int_std = data[intervention_outcome_col].std()
    if int_std > 0:
        model_data[intervention_outcome_col] = (
            data[intervention_outcome_col] - int_mean
        ) / int_std
    else:
        model_data[intervention_outcome_col] = data[intervention_outcome_col] - int_mean
    feature_transforms[intervention_outcome_col] = {
        "mean": int_mean,
        "std": int_std if int_std > 0 else 1.0,
    }

    # Standardize target variable
    target_mean = data[target_col].mean()
    target_std = data[target_col].std()
    if target_std > 0:
        model_data[target_col] = (data[target_col] - target_mean) / target_std
    else:
        model_data[target_col] = data[target_col] - target_mean
    feature_transforms[target_col] = {
        "mean": target_mean,
        "std": target_std if target_std > 0 else 1.0,
    }

    print(f"Target variable stats - Mean: {target_mean:.2f}, Std: {target_std:.2f}")
    return feature_transforms


def _calculate_priors(
    use_informed_priors: bool,
    n_input: int,
    standardize: bool,
    data: pd.DataFrame,
    target_col: str,
) -> tuple[float, float, float]:
    """Calculate prior parameters for BNN."""
    if use_informed_priors:
        weight_sigma = 0.5 / np.sqrt(n_input)
        bias_sigma = 0.1
        noise_sigma = 0.5 if standardize else data[target_col].std() * 0.2
    else:
        weight_sigma = 1.0
        bias_sigma = 1.0
        noise_sigma = 1.0

    print(
        f"Using weight_sigma={weight_sigma:.3f}, bias_sigma={bias_sigma:.3f}, noise_sigma={noise_sigma:.3f}"
    )
    return weight_sigma, bias_sigma, noise_sigma


def _sample_bnn_posterior(bnn_model: pm.Model) -> az.InferenceData:
    """Sample from BNN posterior and compute model assessment."""
    print("Sampling from BNN posterior...")
    trace = pm.sample(
        draws=2000,
        tune=1000,
        target_accept=0.90,
        random_seed=42,
        return_inferencedata=True,
        progressbar=True,
        chains=4,
    )

    print("\nComputing model assessment metrics...")
    trace = pm.compute_log_likelihood(trace)
    waic_data = az.waic(trace)
    loo_data = az.loo(trace)

    print("\nBNN Model Assessment:")
    print(f"WAIC: {waic_data.elpd_waic:.2f} ± {waic_data.se:.2f}")
    print(f"LOO: {loo_data.elpd_loo:.2f} ± {loo_data.se:.2f}")

    return trace


def build_bnn_with_log_target(
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    intervention_outcome_col: str,
    hidden_units: list[int] | None = None,
    standardize_features: bool = True,
) -> tuple[pm.Model, az.InferenceData, dict[str, dict[str, float] | dict[str, Any]]]:
    """Build BNN with log-transformed target to ensure positive predictions.

    Args:
        data: DataFrame with training data.
        features: List of feature column names.
        target_col: Target variable column name (will be log-transformed).
        intervention_outcome_col: Intervention outcome column name.
        hidden_units: List of hidden layer sizes.
        standardize_features: Whether to standardize input features only.

    Returns:
        Tuple containing:
        - PyMC model object
        - MCMC trace (InferenceData)
        - Feature transforms dictionary including log transform info
    """
    if hidden_units is None:
        hidden_units = [8, 4]

    print("Building BNN with log-transformed target...")

    model_data = data.copy()
    feature_transforms = {}

    # Standardize features only (not target)
    if standardize_features:
        for feature in features + [intervention_outcome_col]:
            mean_val = data[feature].mean()
            std_val = data[feature].std()
            if std_val > 0:
                model_data[feature] = (data[feature] - mean_val) / std_val
                feature_transforms[feature] = {"mean": mean_val, "std": std_val}

    # Log-transform target to handle positive constraint
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    y_original = data[target_col].values
    y_log = np.log(y_original + epsilon)

    # Store log transform info
    feature_transforms["target_log"] = {
        "type": "log",
        "epsilon": epsilon,
        "original_mean": y_original.mean(),
        "original_std": y_original.std(),
    }

    # Prepare inputs
    X_features = model_data[features].values
    X_intervention = model_data[intervention_outcome_col].values.reshape(-1, 1)
    X_combined = np.concatenate([X_features, X_intervention], axis=1)

    n_samples, n_input = X_combined.shape

    with pm.Model() as model:
        X_tensor = pt.as_tensor_variable(X_combined)

        # Network layers
        layer_sizes = [n_input] + hidden_units + [1]
        current_input = X_tensor

        # Informed priors based on log scale
        weight_sigma = 1.0 / np.sqrt(n_input)

        for i, (_in_size, out_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
        ):
            W = pm.Normal(
                f"W_{i}", mu=0, sigma=weight_sigma, shape=(_in_size, out_size)
            )
            b = pm.Normal(f"b_{i}", mu=0, sigma=0.1, shape=out_size)

            linear_out = pt.dot(current_input, W) + b

            if i < len(layer_sizes) - 2:  # Hidden layers
                current_input = pt.tanh(
                    linear_out
                )  # Can use tanh since we're in log space
            else:  # Output layer
                log_network_output = linear_out.flatten()

        # Prior for noise in log space
        sigma_log = pm.HalfNormal("sigma_log", sigma=0.5)

        # Likelihood in log space
        _y_log_obs = pm.Normal(
            "y_log_obs", mu=log_network_output, sigma=sigma_log, observed=y_log
        )

        # Sample
        trace = pm.sample(
            draws=2000,
            tune=1000,
            target_accept=0.90,
            random_seed=42,
            return_inferencedata=True,
        )

    return model, trace, feature_transforms


def predict_log_transform_bnn(
    trace: az.InferenceData,
    new_data: dict[str, float],
    feature_transforms: dict[str, dict[str, float] | dict[str, Any]],
    features: list[str],
    intervention_outcome_col: str,
    hidden_units: list[int],
    n_samples: int = 1000,
) -> np.ndarray:
    """Make predictions for log-transformed target BNN.

    Args:
        trace: MCMC trace from log-target BNN.
        new_data: Input data dictionary for prediction.
        feature_transforms: Feature standardization and log transform parameters.
        features: List of feature names.
        intervention_outcome_col: Intervention outcome column name.
        hidden_units: Network architecture (must match training).
        n_samples: Number of posterior samples to use.

    Returns:
        Array of prediction samples in original (non-log) scale.
    """
    # Standardize inputs
    standardized_data = new_data.copy()
    for feature in features + [intervention_outcome_col]:
        if feature in feature_transforms:
            transform = feature_transforms[feature]
            standardized_data[feature] = (
                new_data[feature] - transform["mean"]
            ) / transform["std"]

    # Prepare input
    X_features = np.array([standardized_data[f] for f in features])
    X_intervention = np.array([standardized_data[intervention_outcome_col]])
    X_combined = np.concatenate([X_features, X_intervention])

    posterior = trace.posterior
    predictions_log = []

    # Network forward pass (similar to before but simpler)
    layer_sizes = [len(features) + 1] + hidden_units + [1]

    for _ in range(n_samples):
        chain_idx = np.random.randint(0, posterior.dims["chain"])
        draw_idx = np.random.randint(0, posterior.dims["draw"])

        current_input = X_combined

        for layer_idx, (_in_size, _out_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
        ):
            W = posterior[f"W_{layer_idx}"][chain_idx, draw_idx].values
            b = posterior[f"b_{layer_idx}"][chain_idx, draw_idx].values

            linear_out = np.dot(current_input, W) + b

            if layer_idx < len(layer_sizes) - 2:
                current_input = np.tanh(linear_out)
            else:
                log_output = linear_out[0]

        # Add noise in log space
        sigma_log = posterior["sigma_log"][chain_idx, draw_idx].values
        log_pred = np.random.normal(log_output, sigma_log)

        # Transform back to original scale
        pred = np.exp(log_pred) - feature_transforms["target_log"]["epsilon"]
        predictions_log.append(pred)

    return np.array(predictions_log)


def predict_bnn_with_transforms(
    trace: az.InferenceData,
    new_data: dict[str, float],
    feature_transforms: dict[str, dict[str, float]] | None,
    features: list[str],
    intervention_outcome_col: str,
    hidden_units: list[int] | None = None,
    n_samples: int = 1000,
    return_standardized: bool = False,
) -> np.ndarray:
    """Make predictions using trained BNN with proper feature transformation.

    Args:
        trace: BNN posterior trace from PyMC sampling.
        new_data: Dictionary with feature values for prediction.
        feature_transforms: Standardization parameters for features and target.
            If None, no standardization is applied.
        features: List of feature names used in the model.
        intervention_outcome_col: Intervention outcome column name.
        hidden_units: Network architecture (must match training architecture).
        n_samples: Number of posterior samples to use for prediction.
        return_standardized: If True, return predictions in standardized scale.

    Returns:
        Array of prediction samples (in original scale by default).
    """
    if hidden_units is None:
        hidden_units = [10, 5]

    return _predict_bnn_core(
        trace,
        new_data,
        feature_transforms,
        features,
        intervention_outcome_col,
        hidden_units,
        n_samples,
        return_standardized,
    )


def _predict_bnn_core(
    trace: az.InferenceData,
    new_data: dict[str, float],
    feature_transforms: dict[str, dict[str, float]] | None,
    features: list[str],
    intervention_outcome_col: str,
    hidden_units: list[int],
    n_samples: int,
    return_standardized: bool,
) -> np.ndarray:
    """Core BNN prediction logic."""
    # Apply standardization if transforms are provided
    if feature_transforms is not None:
        standardized_data = new_data.copy()
        for feature in features + [intervention_outcome_col]:
            if feature in standardized_data and feature in feature_transforms:
                transform = feature_transforms[feature]
                mean_val = transform["mean"]
                std_val = transform["std"]
                standardized_data[feature] = (
                    standardized_data[feature] - mean_val
                ) / std_val
        prediction_data = standardized_data
    else:
        prediction_data = new_data

    # Prepare input vector
    X_features = np.array([prediction_data[f] for f in features])
    X_intervention = np.array([prediction_data[intervention_outcome_col]])
    X_combined = np.concatenate([X_features, X_intervention])

    predictions = _forward_pass_bnn(
        trace, X_combined, hidden_units, features, n_samples
    )

    # Transform predictions back to original scale if needed
    if not return_standardized and feature_transforms is not None:
        predictions = _transform_predictions_to_original_scale(
            predictions, feature_transforms, features, intervention_outcome_col
        )

    return predictions


def _forward_pass_bnn(
    trace: az.InferenceData,
    X_combined: np.ndarray,
    hidden_units: list[int],
    features: list[str],
    n_samples: int,
) -> np.ndarray:
    """Perform forward pass through BNN."""
    posterior = trace.posterior
    n_chains = posterior.dims["chain"]
    n_draws = posterior.dims["draw"]
    predictions = []

    layer_sizes = [len(features) + 1] + hidden_units + [1]
    sample_indices = np.random.choice(n_chains * n_draws, size=n_samples, replace=True)

    for idx in sample_indices:
        chain_idx = idx // n_draws
        draw_idx = idx % n_draws

        current_input = X_combined

        for layer_idx, (_in_size, _out_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
        ):
            W = posterior[f"W_{layer_idx}"][chain_idx, draw_idx].values
            b = posterior[f"b_{layer_idx}"][chain_idx, draw_idx].values

            linear_out = np.dot(current_input, W) + b

            if layer_idx < len(layer_sizes) - 2:
                current_input = np.maximum(linear_out, 0)
            else:
                network_output = linear_out[0]

        sigma_val = posterior["sigma"][chain_idx, draw_idx].values
        pred_sample = np.random.normal(network_output, sigma_val)
        predictions.append(pred_sample)

    return np.array(predictions)


def _transform_predictions_to_original_scale(
    predictions: np.ndarray,
    feature_transforms: dict[str, dict[str, float]],
    features: list[str],
    intervention_outcome_col: str,
) -> np.ndarray:
    """Transform predictions back to original scale."""
    # Get target transformation parameters
    if "PFS_months" in feature_transforms:
        target_transform = feature_transforms["PFS_months"]
    elif list(feature_transforms.keys())[-1] not in features + [
        intervention_outcome_col
    ]:
        target_transform = feature_transforms[list(feature_transforms.keys())[-1]]
    else:
        for key in feature_transforms:
            if key not in features and key != intervention_outcome_col:
                target_transform = feature_transforms[key]
                break
        else:
            print(
                "Warning: Could not find target transformation. Returning standardized predictions."
            )
            return predictions

    target_mean = target_transform["mean"]
    target_std = target_transform["std"]
    predictions = predictions * target_std + target_mean

    print(
        f"Transformed predictions back to original scale (mean={target_mean:.2f}, std={target_std:.2f})"
    )

    return predictions


def plot_model_comparison(
    predictions_dict: dict[str, np.ndarray],
    title: str = "Model Comparison",
    save_path: str | None = None,
) -> None:
    """Plot comparison between different model predictions.

    Args:
        predictions_dict: Dictionary mapping model names to prediction arrays.
        title: Title for the plot.
        save_path: Path to save the plot. If None, plot is not saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot distributions
    for model_name, preds in predictions_dict.items():
        axes[0].hist(
            preds,
            bins=50,
            alpha=0.6,
            density=True,
            label=f"{model_name.upper()}: μ={np.mean(preds):.2f}",
        )

        # Add mean line
        axes[0].axvline(
            np.mean(preds),
            linestyle="--",
            alpha=0.8,
            label=f"{model_name.upper()} Mean",
        )

    axes[0].set_xlabel("Predicted Outcome")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{title} - Posterior Distributions")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot comparison
    data_for_box = list(predictions_dict.values())
    labels_for_box = [name.upper() for name in predictions_dict.keys()]

    box_plot = axes[1].boxplot(data_for_box, labels=labels_for_box, patch_artist=True)

    # Color the boxes
    colors = ["lightblue", "lightcoral", "lightgreen", "lightyellow"]
    for patch, color in zip(
        box_plot["boxes"], colors[: len(box_plot["boxes"])], strict=False
    ):
        patch.set_facecolor(color)

    axes[1].set_ylabel("Predicted Outcome")
    axes[1].set_title(f"{title} - Box Plot Comparison")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    # Print summary statistics
    print(f"\n{title} - Summary Statistics:")
    print("=" * 50)
    for model_name, preds in predictions_dict.items():
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
        ci_low, ci_high = np.percentile(preds, [2.5, 97.5])
        print(f"{model_name.upper()}:")
        print(f"  Mean: {mean_pred:.3f}")
        print(f"  Std:  {std_pred:.3f}")
        print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        print()
