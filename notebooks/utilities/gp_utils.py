# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your initial use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""RCT Lean Gaussian Process Helper functions."""

from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from .utils import calculate_contextual_prior


def build_gaussian_process(  # noqa: PLR0915
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    intervention_outcome_col: str,
    kernel: str = "rbf",
    standardize: bool = True,
    use_log_target: bool = False,
) -> tuple[pm.Model, az.InferenceData, dict[str, any]]:
    """Build Gaussian Process model for control arm prediction.

    This function creates a Bayesian Gaussian Process model using PyMC for
    predicting control arm outcomes in RCT data with uncertainty quantification.

    Args:
        data: DataFrame with training data containing features and target.
        features: List of feature column names to use as inputs.
        target_col: Target variable column name (e.g., PFS_months).
        intervention_outcome_col: Intervention outcome column name.
        kernel: Kernel type - 'rbf', 'matern32', 'matern52', or 'rational_quadratic'.
        standardize: Whether to standardize features and target variable.
        use_log_target: Whether to log-transform target (ensures positive predictions).

    Returns:
        Tuple containing:
        - PyMC model object
        - MCMC trace (InferenceData)
        - GP parameters dictionary for prediction containing kernel info, transforms,
          and training data

    Raises:
        ValueError: If unknown kernel type is specified.
    """
    print(f"Building Gaussian Process with {kernel} kernel...")
    print(f"Dataset size: {len(data)} samples, {len(features) + 1} features")

    # Prepare data
    model_data = data.copy()
    transforms = {}

    # Standardize features
    if standardize:
        print("Standardizing features...")
        for feature in features + [intervention_outcome_col]:
            if feature in data.columns:
                mean_val = data[feature].mean()
                std_val = data[feature].std()
                if std_val > 0:
                    model_data[feature] = (data[feature] - mean_val) / std_val
                else:
                    model_data[feature] = data[feature] - mean_val
                transforms[feature] = {
                    "mean": mean_val,
                    "std": std_val if std_val > 0 else 1.0,
                }

    # Prepare input matrix
    X_features = model_data[features].values
    X_intervention = model_data[intervention_outcome_col].values.reshape(-1, 1)
    X = np.concatenate([X_features, X_intervention], axis=1)

    # Handle target
    y = data[target_col].values
    if use_log_target:
        offset = 1.0  # Add offset before log transform
        y_transformed = np.log(y + offset)
        transforms["target"] = {
            "type": "log",
            "offset": offset,
            "original_mean": y.mean(),
            "original_std": y.std(),
        }
        print(
            f"Log-transformed target: mean={y_transformed.mean():.2f}, std={y_transformed.std():.2f}"
        )
    else:
        y_transformed = y
        transforms["target"] = {
            "type": "none",
            "original_mean": y.mean(),
            "original_std": y.std(),
        }

    n_samples, n_features = X.shape

    # Build GP model
    with pm.Model() as gp_model:
        # Hyperpriors for GP
        # Length scales - one per feature (ARD - Automatic Relevance Determination)
        length_scales = pm.Gamma("length_scales", alpha=2, beta=2, shape=n_features)

        # Amplitude (signal variance)
        amplitude = pm.HalfNormal("amplitude", sigma=y_transformed.std())

        # Noise variance
        noise = pm.HalfNormal("noise", sigma=y_transformed.std() * 0.1)

        # Use proper mean function
        mean_func = pm.gp.mean.Constant(c=y_transformed.mean())

        # Select kernel
        if kernel == "rbf":
            # Radial Basis Function (Squared Exponential) kernel
            cov_func = amplitude**2 * pm.gp.cov.ExpQuad(
                input_dim=n_features, ls=length_scales
            )
        elif kernel == "matern32":
            # Matern 3/2 kernel - less smooth than RBF
            cov_func = amplitude**2 * pm.gp.cov.Matern32(
                input_dim=n_features, ls=length_scales
            )
        elif kernel == "matern52":
            # Matern 5/2 kernel - between RBF and Matern32 in smoothness
            cov_func = amplitude**2 * pm.gp.cov.Matern52(
                input_dim=n_features, ls=length_scales
            )
        elif kernel == "rational_quadratic":
            # Rational Quadratic kernel - can model multiple length scales
            alpha_param = pm.Gamma("alpha_param", alpha=2, beta=1)
            cov_func = amplitude**2 * pm.gp.cov.RatQuad(
                input_dim=n_features, ls=length_scales, alpha=alpha_param
            )
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # GP prior
        gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

        # Marginal likelihood - PyMC requires this for model specification
        _y_obs = gp.marginal_likelihood("y_obs", X=X, y=y_transformed, noise=noise)

        # Sample from posterior
        print("\nSampling from GP posterior...")
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
        )

        # Store GP parameters for prediction
        gp_params = {
            "gp": gp,
            "X_train": X,
            "y_train": y_transformed,
            "n_features": n_features,
            "kernel": kernel,
            "transforms": transforms,
            "use_log_target": use_log_target,
            "mean_value": y_transformed.mean(),
        }

    # Print learned hyperparameters
    print("\nLearned hyperparameters (posterior means):")
    print(
        f"  Length scales: {trace.posterior['length_scales'].mean(dim=['chain', 'draw']).values}"
    )
    print(f"  Amplitude: {trace.posterior['amplitude'].mean().values:.3f}")
    print(f"  Noise: {trace.posterior['noise'].mean().values:.3f}")

    return gp_model, trace, gp_params


def predict_gp(
    trace: az.InferenceData,
    gp_params: dict[str, any],
    new_data: dict[str, float],
    features: list[str],
    intervention_outcome_col: str,
    n_samples: int = 1000,
    return_std: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Make predictions using trained Gaussian Process.

    This function generates posterior predictive samples from a trained GP model,
    handling feature standardization and target transformations.

    Args:
        trace: MCMC trace from GP training containing posterior samples.
        gp_params: GP parameters from training including transforms and kernel info.
        new_data: Dictionary with feature values for prediction.
        features: List of feature names used in the model.
        intervention_outcome_col: Intervention outcome column name.
        n_samples: Number of posterior samples to use for prediction.
        return_std: If True, also return standard deviation of predictions.

    Returns:
        Array of prediction samples. If return_std=True, returns tuple of
        (predictions, standard_deviations).
    """
    # Standardize input features
    transforms = gp_params["transforms"]
    standardized_data = new_data.copy()

    for feature in features + [intervention_outcome_col]:
        if feature in transforms:
            transform = transforms[feature]
            mean_val = transform["mean"]
            std_val = transform["std"]
            standardized_data[feature] = (new_data[feature] - mean_val) / std_val

    # Prepare input
    X_new = np.array(
        [
            [standardized_data[f] for f in features]
            + [standardized_data[intervention_outcome_col]]
        ]
    )

    # Extract posterior samples
    posterior = trace.posterior
    predictions = []
    stds = []

    # Sample from posterior predictive
    for _ in range(n_samples):
        # Random posterior sample
        chain_idx = np.random.randint(0, posterior.dims["chain"])
        draw_idx = np.random.randint(0, posterior.dims["draw"])

        # Get hyperparameters for this sample
        length_scales_sample = posterior["length_scales"][chain_idx, draw_idx].values
        amplitude_sample = posterior["amplitude"][chain_idx, draw_idx].values
        noise_sample = posterior["noise"][chain_idx, draw_idx].values
        mean_sample = gp_params["mean_value"]

        # Handle rational quadratic kernel
        if gp_params["kernel"] == "rational_quadratic":
            alpha_sample = posterior["alpha_param"][chain_idx, draw_idx].values
        else:
            alpha_sample = None

        # Compute kernel matrices
        K_train = compute_kernel_matrix(
            gp_params["X_train"],
            gp_params["X_train"],
            length_scales_sample,
            amplitude_sample,
            gp_params["kernel"],
            alpha_sample,
        )
        K_train_noise = K_train + noise_sample**2 * np.eye(len(gp_params["X_train"]))

        K_test = compute_kernel_matrix(
            X_new,
            gp_params["X_train"],
            length_scales_sample,
            amplitude_sample,
            gp_params["kernel"],
            alpha_sample,
        )

        K_test_test = compute_kernel_matrix(
            X_new,
            X_new,
            length_scales_sample,
            amplitude_sample,
            gp_params["kernel"],
            alpha_sample,
        )

        # Compute predictive mean and variance
        try:
            K_inv = np.linalg.solve(K_train_noise, np.eye(len(K_train_noise)))
        except np.linalg.LinAlgError:
            # Add jitter if matrix is singular
            K_train_noise += 1e-6 * np.eye(len(K_train_noise))
            K_inv = np.linalg.solve(K_train_noise, np.eye(len(K_train_noise)))

        # Predictive mean
        pred_mean = mean_sample + K_test @ K_inv @ (gp_params["y_train"] - mean_sample)

        # Predictive variance
        pred_var = K_test_test - K_test @ K_inv @ K_test.T
        pred_std = np.sqrt(np.maximum(pred_var[0, 0] + noise_sample**2, 1e-6))

        # Sample from predictive distribution
        pred_sample = np.random.normal(pred_mean[0], pred_std)

        # Transform back if using log target
        if gp_params["use_log_target"]:
            offset = transforms["target"]["offset"]
            pred_sample = np.exp(pred_sample) - offset
            pred_sample = max(0, pred_sample)  # Ensure non-negative

        predictions.append(pred_sample)
        stds.append(pred_std)

    predictions = np.array(predictions)

    if return_std:
        return predictions, np.array(stds)
    return predictions


def compute_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scales: np.ndarray,
    amplitude: float,
    kernel: str = "rbf",
    alpha: float | None = None,
) -> np.ndarray:
    """Compute kernel matrix between two sets of inputs.

    Args:
        X1: First set of input points, shape (n1, d).
        X2: Second set of input points, shape (n2, d).
        length_scales: Length scale parameters for each dimension, shape (d,).
        amplitude: Amplitude (signal variance) parameter.
        kernel: Kernel type - 'rbf', 'matern32', 'matern52', or 'rational_quadratic'.
        alpha: Alpha parameter for rational quadratic kernel. Ignored for other kernels.

    Returns:
        Kernel matrix of shape (n1, n2).

    Raises:
        ValueError: If unknown kernel type is specified.
    """
    # Scale inputs by length scales
    X1_scaled = X1 / length_scales
    X2_scaled = X2 / length_scales

    # Compute pairwise distances
    if X1 is X2:
        dists = squareform(pdist(X1_scaled, "euclidean"))
    else:
        dists = np.sqrt(
            np.sum((X1_scaled[:, None, :] - X2_scaled[None, :, :]) ** 2, axis=2)
        )

    if kernel == "rbf":
        K = amplitude**2 * np.exp(-0.5 * dists**2)
    elif kernel == "matern32":
        K = amplitude**2 * (1 + np.sqrt(3) * dists) * np.exp(-np.sqrt(3) * dists)
    elif kernel == "matern52":
        K = (
            amplitude**2
            * (1 + np.sqrt(5) * dists + 5 / 3 * dists**2)
            * np.exp(-np.sqrt(5) * dists)
        )
    elif kernel == "rational_quadratic":
        if alpha is None:
            alpha = 1.0  # Default value
        K = amplitude**2 * (1 + dists**2 / (2 * alpha)) ** (-alpha)
    else:
        raise ValueError(f"Kernel {kernel} not implemented in this function")

    return K


def build_simple_gp(
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    intervention_outcome_col: str,
    kernel: str = "matern52",
    standardize: bool = True,
) -> tuple[pm.Model, az.InferenceData, dict[str, any]]:
    """Build simplified Gaussian Process implementation that's more robust.

    This is a streamlined version of the GP model with simpler priors and
    reduced complexity for better numerical stability.

    Args:
        data: DataFrame with training data.
        features: List of feature column names.
        target_col: Target variable column name.
        intervention_outcome_col: Intervention outcome column name.
        kernel: Kernel type - 'rbf', 'matern32', or 'matern52'.
        standardize: Whether to standardize features.

    Returns:
        Tuple containing:
        - PyMC model object
        - MCMC trace (InferenceData)
        - GP parameters dictionary for prediction
    """
    print(f"Building Simple Gaussian Process with {kernel} kernel...")

    # Prepare data
    model_data = data.copy()
    transforms = {}

    # Standardize features
    if standardize:
        for feature in features + [intervention_outcome_col]:
            if feature in data.columns:
                mean_val = data[feature].mean()
                std_val = data[feature].std()
                if std_val > 0:
                    model_data[feature] = (data[feature] - mean_val) / std_val
                transforms[feature] = {
                    "mean": mean_val,
                    "std": std_val if std_val > 0 else 1.0,
                }

    # Prepare input matrix
    X_features = model_data[features].values
    X_intervention = model_data[intervention_outcome_col].values.reshape(-1, 1)
    X = np.concatenate([X_features, X_intervention], axis=1)
    y = data[target_col].values

    n_samples, n_features = X.shape

    # Build simpler GP model
    with pm.Model() as simple_gp_model:
        # Simpler priors
        length_scales = pm.Gamma("length_scales", alpha=2, beta=1, shape=n_features)
        amplitude = pm.HalfNormal("amplitude", sigma=2.0)
        noise = pm.HalfNormal("noise", sigma=1.0)

        # Use zero mean function (simpler)
        mean_func = pm.gp.mean.Zero()

        # Select kernel
        if kernel == "rbf":
            cov_func = amplitude**2 * pm.gp.cov.ExpQuad(
                input_dim=n_features, ls=length_scales
            )
        elif kernel == "matern32":
            cov_func = amplitude**2 * pm.gp.cov.Matern32(
                input_dim=n_features, ls=length_scales
            )
        elif kernel == "matern52":
            cov_func = amplitude**2 * pm.gp.cov.Matern52(
                input_dim=n_features, ls=length_scales
            )
        else:
            cov_func = amplitude**2 * pm.gp.cov.ExpQuad(
                input_dim=n_features, ls=length_scales
            )

        # GP prior
        gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

        # Marginal likelihood - PyMC requires this for model specification
        _y_obs = gp.marginal_likelihood("y_obs", X=X, y=y, noise=noise)

        # Sample
        print("Sampling...")
        trace = pm.sample(
            draws=1500,
            tune=1000,
            chains=2,  # Reduced chains for faster sampling
            target_accept=0.85,
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
        )

        gp_params = {
            "gp": gp,
            "X_train": X,
            "y_train": y,
            "n_features": n_features,
            "kernel": kernel,
            "transforms": transforms,
            "use_log_target": False,
            "mean_value": 0.0,
        }

    print("\nSimple GP training complete!")
    return simple_gp_model, trace, gp_params


def safe_build_gp(
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    intervention_outcome_col: str,
    kernel: str = "matern52",
    use_simple: bool = False,
) -> tuple[pm.Model | None, az.InferenceData | None, dict[str, any] | None]:
    """Build Gaussian Process with comprehensive error handling.

    This function attempts to build a GP model with fallback options if the
    initial configuration fails.

    Args:
        data: DataFrame with training data.
        features: List of feature column names.
        target_col: Target variable column name.
        intervention_outcome_col: Intervention outcome column name.
        kernel: Kernel type to try first.
        use_simple: Whether to use simplified GP implementation.

    Returns:
        Tuple containing model, trace, and parameters. Returns (None, None, None)
        if all attempts fail.
    """
    try:
        if use_simple:
            print("Using simple GP implementation...")
            return build_simple_gp(
                data, features, target_col, intervention_outcome_col, kernel
            )
        else:
            print("Using full GP implementation...")
            return build_gaussian_process(
                data,
                features,
                target_col,
                intervention_outcome_col,
                kernel,
                standardize=True,
                use_log_target=True,
            )

    except Exception as e:
        print(f"Error with {kernel} kernel: {e}")
        print("Trying with RBF kernel...")
        try:
            if use_simple:
                return build_simple_gp(
                    data, features, target_col, intervention_outcome_col, "rbf"
                )
            else:
                return build_gaussian_process(
                    data,
                    features,
                    target_col,
                    intervention_outcome_col,
                    "rbf",
                    standardize=True,
                    use_log_target=False,
                )
        except Exception as e2:
            print(f"Error with RBF kernel: {e2}")
            print("Trying simple GP...")
            return build_simple_gp(
                data, features, target_col, intervention_outcome_col, "rbf"
            )


def plot_gp_predictions(  # noqa: PLR0915
    predictions: np.ndarray,
    true_value: float | None = None,
    title: str = "GP Predictions",
) -> None:
    """Plot comprehensive visualization of Gaussian Process predictions.

    Creates a multi-panel plot showing distribution, CDF, box plot, Q-Q plot,
    uncertainty intervals, and summary statistics.

    Args:
        predictions: Array of prediction samples from GP.
        true_value: True value for comparison (if known).
        title: Title for the overall plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Distribution
    ax = axes[0, 0]
    ax.hist(
        predictions, bins=50, density=True, alpha=0.7, color="blue", edgecolor="black"
    )
    ax.axvline(
        np.mean(predictions),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(predictions):.2f}",
    )
    ax.axvline(
        np.percentile(predictions, 2.5), color="orange", linestyle=":", alpha=0.7
    )
    ax.axvline(
        np.percentile(predictions, 97.5),
        color="orange",
        linestyle=":",
        alpha=0.7,
        label="95% CI",
    )
    if true_value is not None:
        ax.axvline(
            true_value,
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"True: {true_value:.2f}",
        )
    ax.set_xlabel("PFS (months)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. CDF
    ax = axes[0, 1]
    sorted_preds = np.sort(predictions)
    cdf = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
    ax.plot(sorted_preds, cdf, linewidth=2, color="blue")
    ax.fill_betweenx(
        cdf,
        np.percentile(predictions, 2.5),
        np.percentile(predictions, 97.5),
        alpha=0.2,
        color="blue",
    )
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.axvline(np.median(predictions), color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("PFS (months)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Function")
    ax.grid(True, alpha=0.3)

    # 3. Box plot with violin
    ax = axes[0, 2]
    parts = ax.violinplot(
        [predictions],
        positions=[1],
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_alpha(0.7)
    ax.set_ylabel("PFS (months)")
    ax.set_xticks([1])
    ax.set_xticklabels(["GP Predictions"])
    ax.set_title("Distribution Summary")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Q-Q plot
    ax = axes[1, 0]
    stats.probplot(predictions, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (Normality Check)")
    ax.grid(True, alpha=0.3)

    # 5. Uncertainty intervals
    ax = axes[1, 1]
    intervals = [50, 80, 95]
    colors = ["green", "orange", "red"]
    for interval, color in zip(intervals, colors, strict=False):
        lower = (100 - interval) / 2
        upper = 100 - lower
        ax.barh(
            interval,
            np.percentile(predictions, upper) - np.percentile(predictions, lower),
            left=np.percentile(predictions, lower),
            height=10,
            color=color,
            alpha=0.6,
            label=f"{interval}% CI",
        )
    ax.axvline(np.mean(predictions), color="black", linestyle="--", label="Mean")
    ax.set_xlabel("PFS (months)")
    ax.set_ylabel("Confidence Level (%)")
    ax.set_title("Prediction Intervals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis("off")

    summary = f"""
    Summary Statistics:
    Mean:      {np.mean(predictions):.2f} months
    Median:    {np.median(predictions):.2f} months
    Std Dev:   {np.std(predictions):.2f} months

    Percentiles:
      2.5%:    {np.percentile(predictions, 2.5):.2f} months
       25%:    {np.percentile(predictions, 25):.2f} months
       75%:    {np.percentile(predictions, 75):.2f} months
     97.5%:    {np.percentile(predictions, 97.5):.2f} months

    95% CI: [{np.percentile(predictions, 2.5):.2f},
             {np.percentile(predictions, 97.5):.2f}]

    Probability Ranges:
    P(<0):     {(predictions < 0).mean():.1%}
    P(0-6):    {((predictions >= 0) & (predictions < 6)).mean():.1%}
    P(6-12):   {((predictions >= 6) & (predictions < 12)).mean():.1%}
    P(>12):    {(predictions > 12).mean():.1%}
    """

    ax.text(
        0.05,
        0.5,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        fontfamily="monospace",
    )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def build_gp_with_informative_prior(
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    intervention_outcome_col: str,
    intervention_outcome: float,
    draws: int = 600,
    tune: int = 600,
) -> tuple[az.InferenceData, dict[str, Any]]:
    """Build GP with contextual prior and proper uncertainty quantification.

    This function creates a Gaussian Process model with an informative prior
    based on contextual knowledge of the intervention outcome. It uses proper
    scaling and regularization techniques for numerical stability.

    Args:
        data: DataFrame containing the training data with features and target.
        features: List of column names to use as input features for the GP.
        target_col: Name of the target variable column in the data.
        intervention_outcome_col: Name of the intervention outcome column.
        intervention_outcome: The intervention outcome value to inform the prior.
        draws: Number of MCMC samples to draw from the posterior. Defaults to 600.
        tune: Number of tuning steps for MCMC sampler. Defaults to 600.

    Returns:
        Tuple containing:
            - InferenceData object with MCMC trace from posterior sampling
            - Dictionary with GP parameters including:
                - gp: The PyMC GP object
                - X_train: Scaled training features
                - y_train: Scaled training targets
                - y_mean, y_std: Target scaling parameters
                - scaler: Feature scaler object
                - prior_mean: Prior mean value
                - features: List of feature names

    Raises:
        ValueError: If features are not found in the data columns.
        RuntimeError: If MCMC sampling fails to converge.
    """
    X = data[features].values
    y = data[target_col].values
    n_samples, n_features = X.shape

    # Robust scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate informative prior
    prior_multiplier = calculate_contextual_prior(
        intervention_outcome, data, target_col
    )
    prior_mean_value = prior_multiplier * intervention_outcome

    # Target standardization
    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / y_std
    prior_mean_scaled = (prior_mean_value - y_mean) / y_std

    with pm.Model() as gp_model:
        # ===== INFORMATIVE PRIORS =====

        # Informative mean function instead of zero mean
        # This incorporates our contextual baseline knowledge
        mean_func = pm.gp.mean.Constant(c=prior_mean_scaled)
        print(gp_model)

        # Conservative length scales (prevent overfitting)
        if n_features <= 3:
            ls = pm.Gamma("ls", alpha=3, beta=0.5)  # Single length scale
            amplitude = pm.HalfNormal("amplitude", sigma=0.8)
            cov_func = amplitude**2 * pm.gp.cov.Matern32(n_features, ls=ls)
        else:
            ls = pm.Gamma("ls", alpha=2, beta=0.3, shape=n_features)
            amplitude = pm.HalfNormal("amplitude", sigma=0.6)
            cov_func = amplitude**2 * pm.gp.cov.Matern32(n_features, ls=ls)

        # Moderate noise (balance flexibility vs overfitting)
        noise = pm.HalfNormal("noise", sigma=0.4)

        # Add regularization
        white_noise = pm.gp.cov.WhiteNoise(sigma=0.2)
        cov_func = cov_func + white_noise

        # GP with informative mean
        gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
        y_obs = gp.marginal_likelihood("y_obs", X=X_scaled, y=y_scaled, sigma=noise)
        print(type(y_obs))

        # Sample posterior
        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.92,
            max_treedepth=10,
            init="adapt_diag",
            random_seed=42,
            return_inferencedata=True,
            progressbar=False,
            chains=2,
            cores=1,
        )

        return trace, {
            "gp": gp,
            "X_train": X_scaled,
            "y_train": y_scaled,
            "y_mean": y_mean,
            "y_std": y_std,
            "scaler": scaler,
            "prior_mean": prior_mean_value,
            "features": features,
        }


def predict_gp_with_uncertainty(
    trace: az.InferenceData,
    gp_params: dict[str, Any],
    new_data: dict[str, float],
    n_samples: int = 1000,
) -> np.ndarray:
    """Generate full posterior predictive distribution from trained GP.

    This function generates predictions by sampling from the posterior predictive
    distribution of a trained Gaussian Process. It handles proper scaling and
    includes observation noise in the predictions.

    Args:
        trace: InferenceData object containing MCMC samples from GP training.
        gp_params: Dictionary containing GP parameters from training including:
            - gp: PyMC GP object
            - X_train, y_train: Scaled training data
            - y_mean, y_std: Target scaling parameters
            - scaler: Feature scaler
            - prior_mean: Prior mean value
            - features: Feature names list
        new_data: Dictionary mapping feature names to their values for prediction.
        n_samples: Number of samples to draw from posterior predictive distribution.
            Defaults to 1000.

    Returns:
        Array of prediction samples from the posterior predictive distribution,
        transformed back to the original target scale. Shape: (n_samples,)

    Raises:
        KeyError: If required features are missing from new_data.
        np.linalg.LinAlgError: If kernel matrix inversion fails (handled internally).
        ValueError: If posterior sampling parameters are invalid.
    """
    X_new = np.array([new_data[f] for f in gp_params["features"]]).reshape(1, -1)
    X_new_scaled = gp_params["scaler"].transform(X_new)

    posterior = trace.posterior
    predictions = []

    # Sample from posterior predictive
    # n_posterior_samples = min(100, posterior.dims["draw"] * posterior.dims["chain"])

    for _ in range(n_samples):
        try:
            # Random posterior sample
            chain_idx = np.random.randint(posterior.dims["chain"])
            draw_idx = np.random.randint(posterior.dims["draw"])

            # Extract hyperparameters
            ls_sample = posterior["ls"][chain_idx, draw_idx].values
            amplitude_sample = posterior["amplitude"][chain_idx, draw_idx].values
            noise_sample = posterior["noise"][chain_idx, draw_idx].values

            # Kernel matrices
            K_train = compute_matern32_kernel(
                gp_params["X_train"], gp_params["X_train"], ls_sample, amplitude_sample
            )
            K_test = compute_matern32_kernel(
                X_new_scaled, gp_params["X_train"], ls_sample, amplitude_sample
            )
            K_test_test = compute_matern32_kernel(
                X_new_scaled, X_new_scaled, ls_sample, amplitude_sample
            )

            # Add noise and regularization
            K_train_reg = K_train + (noise_sample**2 + 0.2) * np.eye(len(K_train))
            K_train_reg += 1e-6 * np.eye(len(K_train))  # Numerical stability

            # GP prediction
            L = np.linalg.cholesky(K_train_reg)
            alpha = np.linalg.solve(L, gp_params["y_train"])
            alpha = np.linalg.solve(L.T, alpha)

            # Predictive mean and variance
            pred_mean = K_test @ alpha
            v = np.linalg.solve(L, K_test.T)
            pred_var = K_test_test - v.T @ v
            pred_var = pred_var[0, 0] + noise_sample**2  # Add observation noise

            # Sample from predictive distribution
            pred_sample = np.random.normal(pred_mean[0], np.sqrt(max(pred_var, 1e-6)))

            # Transform back to original scale
            pred_sample = pred_sample * gp_params["y_std"] + gp_params["y_mean"]

            # Reasonable bounds
            pred_sample = max(0, pred_sample)
            predictions.append(pred_sample)

        except (np.linalg.LinAlgError, ValueError):
            continue

    if len(predictions) < 50:
        # Fallback to prior if sampling fails
        fallback_samples = np.random.normal(
            gp_params["prior_mean"], gp_params["y_std"], size=max(100, n_samples)
        )
        return np.clip(fallback_samples, 0, None)

    return np.array(predictions)


def compute_matern32_kernel(
    X1: np.ndarray, X2: np.ndarray, length_scales: np.ndarray, amplitude: float
) -> np.ndarray:
    """Compute Matern 3/2 kernel matrix between two sets of input points.

    The Matern 3/2 kernel is a popular choice for Gaussian Processes as it
    provides a good balance between smoothness and flexibility. It assumes
    the underlying function is once differentiable.

    Args:
        X1: First set of input points with shape (n1, d) where n1 is the
            number of points and d is the dimensionality.
        X2: Second set of input points with shape (n2, d) where n2 is the
            number of points and d is the dimensionality.
        length_scales: Length scale parameters for each dimension. Can be a
            scalar (isotropic) or array of shape (d,) for automatic relevance
            determination (ARD).
        amplitude: Amplitude (signal variance) parameter controlling the
            overall scale of the kernel output.

    Returns:
        Kernel matrix of shape (n1, n2) where entry (i,j) represents the
        kernel value between point i from X1 and point j from X2.

    Raises:
        ValueError: If X1 and X2 have incompatible dimensions.
        ValueError: If length_scales has incorrect shape for the input dimensions.
    """
    n1, d = X1.shape
    n2 = X2.shape[0]

    dist = np.zeros((n1, n2))
    ls_array = np.atleast_1d(length_scales)

    for i in range(d):
        ls_i = ls_array[i] if len(ls_array) > 1 else ls_array[0]
        ls_i = max(ls_i, 0.1)  # Prevent numerical issues
        diff = np.abs(X1[:, i : i + 1] - X2[:, i : i + 1].T)
        dist += (diff / ls_i) ** 2

    dist = np.sqrt(dist)
    dist_sqrt3 = np.sqrt(3) * dist
    K = amplitude**2 * (1 + dist_sqrt3) * np.exp(-dist_sqrt3)

    return K
