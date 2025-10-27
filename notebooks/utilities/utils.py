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
"""RCT Lean Helper functions."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


def prepare_data(raw_df: pd.DataFrame, list_ftr: list[str]) -> pd.DataFrame:
    """Prepare data for modeling by handling null values and converting to numeric.

    Args:
        raw_df (pd.DataFrame): Input dataframe containing raw data.
        list_ftr (List[str]): List of feature column names to use for modeling.

    Returns:
        pd.DataFrame: Prepared dataframe with selected features, problematic values
            replaced with None, and all values converted to float type.
    """
    # Select only the columns we need
    train_x = raw_df[list_ftr].copy()

    # Replace problematic values
    train_x = train_x.replace("need help", None)
    train_x = train_x.replace("?", None)
    train_x = train_x.replace("??", None)

    # Convert to float
    train_x = train_x.astype(float)

    return train_x


def run_fillna(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in a dataframe.

    Args:
        raw_df (pd.DataFrame): Input dataframe with NaN values to be filled.

    Returns:
        pd.DataFrame: Dataframe with NaN values filled. EGFR_wild column is filled
            with 0, and all other NaN values are filled with column means.
    """
    # Create a copy to avoid modifying the original
    clean_df = raw_df.copy()

    # Special handling for EGFR_wild column
    if "EGFR_wild" in raw_df.columns:
        clean_df["EGFR_wild"] = clean_df["EGFR_wild"].fillna(0)

    # Fill remaining NaN with column means
    clean_df = clean_df.fillna(clean_df.mean())

    return clean_df


def calculate_perf(input_df: pd.DataFrame, name: str) -> dict:
    """Calculate performance metrics for ATE (Average Treatment Effect) predictions.

    Args:
        input_df (pd.DataFrame): Input dataframe containing actual and predicted values
            with columns 'real_ate', 'pred_ate', 'outcome_control', 'predicted_outcome'.
        name (str): Name/identifier for the approach being evaluated.

    Returns:
        dict: Dictionary containing performance metrics including R² scores, RMSE values,
            directional accuracy, and Spearman correlation for ATE predictions.
    """
    r2_ate = round(r2_score(input_df["real_ate"], input_df["pred_ate"]), 2)
    r2_outcome = round(
        r2_score(input_df["outcome_control"], input_df["predicted_outcome"]), 2
    )

    input_df["helper"] = input_df["real_ate"] * input_df["pred_ate"]
    input_df["is_ate_right"] = np.where(input_df["helper"] > 0, 1, 0)
    is_right = round(input_df["is_ate_right"].mean(), 2)

    res = stats.spearmanr(input_df["pred_ate"], input_df["real_ate"])
    spearman_ate, _ = round(res.statistic, 2), round(res.pvalue, 2)

    rmse_ate = round(
        np.sqrt(mean_squared_error(input_df["real_ate"], input_df["pred_ate"])), 2
    )
    rmse_outcome = round(
        np.sqrt(
            mean_squared_error(
                input_df["outcome_control"], input_df["predicted_outcome"]
            )
        ),
        2,
    )

    return {
        "Approach": name,
        "ATE direction true": is_right,
        "r2_ate": r2_ate,
        "spearman_ate": spearman_ate,
        "rmse_ate": rmse_ate,
        "r2_outcome": r2_outcome,
        "rmse_outcome": rmse_outcome,
    }


def calculate_interval_iou(
    pred_interval: list[float] | tuple[float, ...] | np.ndarray,
    true_interval: list[float] | tuple[float, ...] | np.ndarray,
) -> float:
    """Calculate Intersection over Union (IoU) for confidence intervals.

    Computes the IoU metric between predicted and true confidence intervals,
    which measures the overlap between two intervals as a proportion of their union.
    This is useful for evaluating the quality of uncertainty quantification in
    machine learning models.

    Args:
        pred_interval: Predicted confidence interval as [lower_bound, upper_bound].
            Can be a list, tuple, or numpy array with exactly 2 elements.
        true_interval: True confidence interval as [lower_bound, upper_bound].
            Can be a list, tuple, or numpy array with exactly 2 elements.

    Returns:
        IoU score between 0 and 1, where:
        - 0 indicates no overlap between intervals
        - 1 indicates perfect overlap (identical intervals)
        - Values rounded to 3 decimal places

    Examples:
        >>> calculate_interval_iou([1.0, 3.0], [2.0, 4.0])
        0.5
        >>> calculate_interval_iou([1.0, 2.0], [3.0, 4.0])
        0.0
        >>> calculate_interval_iou([1.0, 3.0], [1.0, 3.0])
        1.0
    """
    if isinstance(pred_interval, (list, tuple, np.ndarray)) and len(pred_interval) == 2:
        pred_lower, pred_upper = pred_interval[0], pred_interval[1]
    else:
        return 0.0

    if isinstance(true_interval, (list, tuple, np.ndarray)) and len(true_interval) == 2:
        true_lower, true_upper = true_interval[0], true_interval[1]
    else:
        return 0.0

    # Calculate intersection
    intersection_lower = max(pred_lower, true_lower)
    intersection_upper = min(pred_upper, true_upper)

    # If no intersection, return 0
    if intersection_lower >= intersection_upper:
        return 0.0

    intersection_length = intersection_upper - intersection_lower

    # Calculate union
    union_lower = min(pred_lower, true_lower)
    union_upper = max(pred_upper, true_upper)
    union_length = union_upper - union_lower

    # Avoid division by zero
    if union_length == 0:
        return 1.0 if intersection_length == 0 else 0.0

    iou = intersection_length / union_length
    return round(iou, 3)


def calculate_coverage(
    true_values: np.ndarray | pd.Series,
    ci_lower: np.ndarray | pd.Series,
    ci_upper: np.ndarray | pd.Series,
) -> float:
    """Calculate coverage rate for confidence intervals.

    Computes the proportion of true values that fall within their corresponding
    confidence intervals. This is a key metric for evaluating the quality of
    uncertainty quantification.

    Args:
        true_values: Array of true values to check coverage for.
        ci_lower: Array of lower bounds for confidence intervals.
        ci_upper: Array of upper bounds for confidence intervals.

    Returns:
        Coverage rate between 0 and 1, rounded to 3 decimal places.
        A value of 0.95 indicates 95% coverage for 95% confidence intervals.

    Examples:
        >>> true_vals = np.array([1.0, 2.0, 3.0])
        >>> lower = np.array([0.5, 1.5, 2.5])
        >>> upper = np.array([1.5, 2.5, 3.5])
        >>> calculate_coverage(true_vals, lower, upper)
        1.0
    """
    within_ci = (true_values >= ci_lower) & (true_values <= ci_upper)
    return round(np.mean(within_ci), 3)


def extract_ci_bounds(
    ci_column: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract lower and upper bounds from confidence interval column.

    Handles various formats of confidence interval storage including lists,
    tuples, numpy arrays, or separate values.

    Args:
        ci_column: Pandas Series containing confidence intervals. Each element
            should be a list, tuple, or array with [lower, upper] bounds.

    Returns:
        A tuple containing:
        - ci_lower: Array of lower bounds
        - ci_upper: Array of upper bounds

    Examples:
        >>> ci_data = pd.Series([[1.0, 2.0], [2.5, 3.5], [0.8, 1.2]])
        >>> lower, upper = extract_ci_bounds(ci_data)
        >>> print(lower)
        [1.0 2.5 0.8]
        >>> print(upper)
        [2.0 3.5 1.2]
    """
    if isinstance(ci_column.iloc[0], (list, tuple, np.ndarray)):
        ci_lower = np.array([ci[0] for ci in ci_column])
        ci_upper = np.array([ci[1] for ci in ci_column])
    else:
        # If stored as separate columns or other format
        ci_lower = ci_column.apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) else x
        )
        ci_upper = ci_column.apply(
            lambda x: x[1] if isinstance(x, (list, tuple)) else x
        )
    return ci_lower, ci_upper


def calculate_coverage_from_samples(
    true_vals: np.ndarray | pd.Series,
    samples_list: list[np.ndarray] | pd.Series,
) -> float | None:
    """Calculate coverage using full sample distributions.

    Computes coverage by calculating empirical confidence intervals from
    sample distributions and checking if true values fall within them.

    Args:
        true_vals: Array of true values to check coverage for.
        samples_list: List of sample arrays or Series containing sample
            distributions for each observation.

    Returns:
        Coverage rate between 0 and 1, rounded to 3 decimal places.
        Returns None if no valid samples are found.

    Examples:
        >>> true_vals = np.array([2.0, 3.0])
        >>> samples = [np.array([1.5, 2.0, 2.5]), np.array([2.8, 3.0, 3.2])]
        >>> coverage = calculate_coverage_from_samples(true_vals, samples)
        >>> print(coverage)
        1.0
    """
    coverage_list = []
    for true_val, samples in zip(true_vals, samples_list, strict=False):
        if isinstance(samples, str):
            continue  # Skip if samples are stored as string
        samples_array = np.array(samples)
        ci_lower_sample = np.percentile(samples_array, 2.5)
        ci_upper_sample = np.percentile(samples_array, 97.5)
        is_covered = (true_val >= ci_lower_sample) & (true_val <= ci_upper_sample)
        coverage_list.append(is_covered)
    return round(np.mean(coverage_list), 3) if coverage_list else None


def calculate_iou_metrics(
    input_df: pd.DataFrame,
    actual_ci_col: str,
    pred_ci_col: str,
) -> dict[str, float | list[float] | None]:
    """Calculate IoU metrics for confidence interval comparison.

    Computes various IoU (Intersection over Union) statistics comparing
    predicted and actual confidence intervals.

    Args:
        input_df: DataFrame containing the confidence interval data.
        actual_ci_col: Column name for actual confidence intervals.
        pred_ci_col: Column name for predicted confidence intervals.

    Returns:
        Dictionary containing IoU metrics:
        - iou_outcome_scores: List of individual IoU scores
        - avg_iou_outcome: Mean IoU score
        - median_iou_outcome: Median IoU score
        - min_iou_outcome: Minimum IoU score
        - max_iou_outcome: Maximum IoU score
        - iou_std_outcome: Standard deviation of IoU scores
        - high_iou_percentage: Percentage with IoU > 0.5
        - very_high_iou_percentage: Percentage with IoU > 0.7

    Examples:
        >>> df = pd.DataFrame({
        ...     'actual_ci': [[1.0, 2.0], [2.0, 3.0]],
        ...     'pred_ci': [[1.1, 2.1], [1.9, 2.9]]
        ... })
        >>> metrics = calculate_iou_metrics(df, 'actual_ci', 'pred_ci')
        >>> print(f"Average IoU: {metrics['avg_iou_outcome']}")
        Average IoU: 0.826
    """
    iou_outcome_scores = []

    if actual_ci_col in input_df.columns and pred_ci_col in input_df.columns:
        print(
            f"Calculating IoU using actual CI column: {actual_ci_col} "
            f"and predicted CI column: {pred_ci_col}"
        )

        for _idx, row in input_df.iterrows():
            actual_ci = row[actual_ci_col]
            pred_ci = row[pred_ci_col]
            iou_score = calculate_interval_iou(pred_ci, actual_ci)
            iou_outcome_scores.append(iou_score)

    # Calculate aggregate IoU metrics
    if iou_outcome_scores:
        avg_iou_outcome = round(np.mean(iou_outcome_scores), 3)
        median_iou_outcome = round(np.median(iou_outcome_scores), 3)
        min_iou_outcome = round(np.min(iou_outcome_scores), 3)
        max_iou_outcome = round(np.max(iou_outcome_scores), 3)
        iou_std_outcome = round(np.std(iou_outcome_scores), 3)
        high_iou_pct = round(np.mean(np.array(iou_outcome_scores) > 0.5), 3)
        very_high_iou_pct = round(np.mean(np.array(iou_outcome_scores) > 0.7), 3)

        print(f"IoU Statistics - Mean: {avg_iou_outcome}, Median: {median_iou_outcome}")
        print(
            f"IoU Range: [{min_iou_outcome}, {max_iou_outcome}], Std: {iou_std_outcome}"
        )
        print(
            f"High IoU (>0.5): {high_iou_pct*100:.1f}%, "
            f"Very High IoU (>0.7): {very_high_iou_pct*100:.1f}%"
        )

        return {
            "iou_outcome_scores": iou_outcome_scores,
            "avg_iou_outcome": avg_iou_outcome,
            "median_iou_outcome": median_iou_outcome,
            "min_iou_outcome": min_iou_outcome,
            "max_iou_outcome": max_iou_outcome,
            "iou_std_outcome": iou_std_outcome,
            "high_iou_percentage": high_iou_pct,
            "very_high_iou_percentage": very_high_iou_pct,
        }
    else:
        print(
            f"Warning: Could not calculate IoU. Check if columns {actual_ci_col} "
            f"and {pred_ci_col} exist and contain valid intervals."
        )
        return {
            "iou_outcome_scores": [],
            "avg_iou_outcome": None,
            "median_iou_outcome": None,
            "min_iou_outcome": None,
            "max_iou_outcome": None,
            "iou_std_outcome": None,
            "high_iou_percentage": None,
            "very_high_iou_percentage": None,
        }


def calculate_basic_metrics(
    input_df: pd.DataFrame,
) -> dict[str, float]:
    """Calculate basic performance metrics for ATE and outcome predictions.

    Computes fundamental regression metrics including R-squared, RMSE,
    Spearman correlation, and directional accuracy.

    Args:
        input_df: DataFrame containing actual and predicted values with columns:
            - 'real_ate': True ATE values
            - 'pred_ate': Point estimates of ATE
            - 'outcome_control': True control outcomes
            - 'predicted_outcome': Predicted control outcomes

    Returns:
        Dictionary containing basic performance metrics:
        - r2_ate: R-squared for ATE predictions
        - r2_outcome: R-squared for outcome predictions
        - rmse_ate: Root Mean Square Error for ATE
        - rmse_outcome: Root Mean Square Error for outcomes
        - spearman_ate: Spearman correlation for ATE
        - is_right: Proportion of correct directional predictions
        - abs_bias_ate: Mean absolute bias for ATE predictions

    Raises:
        KeyError: If required columns are missing from input_df.
        ValueError: If arrays have mismatched lengths.
    """
    # Basic regression metrics
    r2_ate = round(r2_score(input_df["real_ate"], input_df["pred_ate"]), 2)
    r2_outcome = round(
        r2_score(input_df["outcome_control"], input_df["predicted_outcome"]), 2
    )

    # RMSE calculations
    rmse_ate = round(
        np.sqrt(mean_squared_error(input_df["real_ate"], input_df["pred_ate"])), 2
    )
    rmse_outcome = round(
        np.sqrt(
            mean_squared_error(
                input_df["outcome_control"], input_df["predicted_outcome"]
            )
        ),
        2,
    )

    # Directional accuracy
    df_copy = input_df.copy()
    df_copy["helper"] = df_copy["real_ate"] * df_copy["pred_ate"]
    df_copy["is_ate_right"] = np.where(df_copy["helper"] > 0, 1, 0)
    is_right = round(df_copy["is_ate_right"].mean(), 2)

    # Spearman correlation
    res = stats.spearmanr(input_df["pred_ate"], input_df["real_ate"])
    spearman_ate = round(res.statistic, 2)

    # Absolute bias
    abs_bias_ate = round(
        np.mean(np.abs(input_df["pred_ate"] - input_df["real_ate"])), 3
    )

    return {
        "r2_ate": r2_ate,
        "r2_outcome": r2_outcome,
        "rmse_ate": rmse_ate,
        "rmse_outcome": rmse_outcome,
        "spearman_ate": spearman_ate,
        "ATE direction true": is_right,
        "abs_bias_ate": abs_bias_ate,
    }


def calculate_uncertainty_metrics(
    input_df: pd.DataFrame,
) -> dict[str, float | None]:
    """Calculate uncertainty quantification metrics.

    Computes coverage rates and confidence interval widths to evaluate
    the quality of uncertainty estimates.

    Args:
        input_df: DataFrame containing confidence interval data with columns:
            - 'real_ate': True ATE values
            - 'ate_ci_95': 95% confidence intervals for ATE (optional)
            - 'ate_samples': Full distribution samples (optional)

    Returns:
        Dictionary containing uncertainty metrics:
        - coverage_95_ate: Coverage rate for 95% CI
        - coverage_95_ate_from_samples: Coverage from sample distributions
        - avg_ci_width_ate: Average width of confidence intervals

    Note:
        Returns None values for metrics that cannot be computed due to
        missing required columns.
    """
    results = {}

    # 95% CI Coverage for ATE
    if "ate_ci_95" in input_df.columns:
        ci_95_lower, ci_95_upper = extract_ci_bounds(input_df["ate_ci_95"])
        coverage_95_ate = calculate_coverage(
            input_df["real_ate"], ci_95_lower, ci_95_upper
        )
        results["coverage_95_ate"] = coverage_95_ate

        # Average CI width for ATE
        ci_widths = ci_95_upper - ci_95_lower
        avg_ci_width_ate = round(np.mean(ci_widths), 3)
        results["avg_ci_width_ate"] = avg_ci_width_ate
    else:
        results["coverage_95_ate"] = None
        results["avg_ci_width_ate"] = None

    # Alternative coverage using samples if available
    if "ate_samples" in input_df.columns:
        coverage_95_ate_samples = calculate_coverage_from_samples(
            input_df["real_ate"], input_df["ate_samples"]
        )
        results["coverage_95_ate_from_samples"] = coverage_95_ate_samples
    else:
        results["coverage_95_ate_from_samples"] = None

    return results


def calculate_perf_enhanced_with_iou(
    input_df: pd.DataFrame,
    name: str,
    actual_ci_col: str = "PFS_median_CI",
    pred_ci_col: str = "pred_ci_95",
) -> dict[str, float | list[float] | str | None]:
    """Calculate comprehensive performance metrics for ATE predictions with uncertainty quantification.

    This function computes a wide range of performance metrics including basic
    regression metrics, uncertainty quantification metrics, and IoU metrics for
    confidence interval overlap evaluation.

    Args:
        input_df: Input dataframe containing actual and predicted values with columns:
            - 'real_ate': True ATE values
            - 'pred_ate': Point estimates of ATE (typically median/mean)
            - 'ate_mean': Mean of ATE distribution
            - 'ate_std': Standard deviation of ATE distribution
            - 'ate_ci_95': 95% confidence intervals for ATE [lower, upper]
            - 'ate_samples': Full distribution samples (optional, for more precise coverage)
            - 'outcome_control': True control outcomes
            - 'predicted_outcome': Predicted control outcomes
            - actual_ci_col: Column containing actual CI from RCT data
            - pred_ci_col: Column containing predicted CI
        name: Name/identifier for the approach being evaluated.
        actual_ci_col: Column name for actual confidence intervals of outcome.
        pred_ci_col: Column name for predicted confidence intervals of outcome.

    Returns:
        Comprehensive dictionary of performance metrics including:
        - Basic metrics: R², RMSE, Spearman correlation, directional accuracy
        - Bias metrics: Absolute bias
        - Uncertainty metrics: Coverage rates, CI widths
        - IoU metrics: Intersection over Union for confidence intervals

    Raises:
        KeyError: If required columns are missing from input_df.
        ValueError: If input data has invalid format or mismatched lengths.

    Examples:
        >>> df = pd.DataFrame({
        ...     'real_ate': [0.1, 0.2, -0.1],
        ...     'pred_ate': [0.12, 0.18, -0.08],
        ...     'outcome_control': [1.0, 1.5, 0.8],
        ...     'predicted_outcome': [1.02, 1.48, 0.82],
        ...     'PFS_median_CI': [[0.8, 1.2], [1.3, 1.7], [0.6, 1.0]],
        ...     'pred_ci_95': [[0.82, 1.22], [1.28, 1.68], [0.58, 1.02]]
        ... })
        >>> results = calculate_perf_enhanced_with_iou(df, "Test Model")
        >>> print(f"R² ATE: {results['r2_ate']}")
        R² ATE: 0.99
    """
    # Calculate basic performance metrics
    basic_metrics = calculate_basic_metrics(input_df)

    # Calculate uncertainty metrics
    uncertainty_metrics = calculate_uncertainty_metrics(input_df)

    # Calculate IoU metrics
    iou_metrics = calculate_iou_metrics(input_df, actual_ci_col, pred_ci_col)

    # Combine all results
    results = {
        "Approach": name,
        **basic_metrics,
        **uncertainty_metrics,
        **{
            k: v
            for k, v in iou_metrics.items()
            if k in ["iou_outcome_scores", "avg_iou_outcome", "median_iou_outcome"]
        },
    }

    # Remove None values for cleaner output
    results = {k: v for k, v in results.items() if v is not None}

    return results


def calculate_perf_comparison(
    results_list: list[pd.DataFrame],
    approach_names: list[str],
) -> pd.DataFrame:
    """Compare performance metrics across multiple approaches.

    Creates a comparison table of performance metrics from multiple model
    evaluation results for easy side-by-side comparison.

    Args:
        results_list: List of DataFrames containing performance results
            from different approaches.
        approach_names: List of names corresponding to each approach
            for labeling in the comparison table.

    Returns:
        DataFrame with approaches as rows and metrics as columns,
        facilitating easy comparison of model performance.

    Raises:
        ValueError: If results_list and approach_names have different lengths.

    Examples:
        >>> results1 = pd.DataFrame({'r2_ate': [0.85], 'rmse_ate': [0.12]})
        >>> results2 = pd.DataFrame({'r2_ate': [0.78], 'rmse_ate': [0.15]})
        >>> comparison = calculate_perf_comparison(
        ...     [results1, results2],
        ...     ["Model A", "Model B"]
        ... )
        >>> print(comparison)
                 r2_ate  rmse_ate
        Model A    0.85      0.12
        Model B    0.78      0.15
    """
    if len(results_list) != len(approach_names):
        raise ValueError("results_list and approach_names must have the same length")

    comparison_data = []
    for results_df, name in zip(results_list, approach_names, strict=True):
        if isinstance(results_df, dict):
            # If results is already a dictionary, use it directly
            row_data = {"Approach": name, **results_df}
        else:
            # If results is a DataFrame, take the first row
            row_data = {"Approach": name, **results_df.iloc[0].to_dict()}
        comparison_data.append(row_data)

    return pd.DataFrame(comparison_data).set_index("Approach")


def calculate_contextual_prior(
    intervention_outcome: float, training_data: pd.DataFrame, target_col: str
) -> float:
    """Calculate informative prior mean based on contextual similarity.

    Determines an appropriate prior multiplier by analyzing similar trials in the
    training data. Finds trials with similar intervention outcomes and computes
    a weighted average of their target-to-intervention ratios.

    Args:
        intervention_outcome: The expected intervention outcome value to find
            similar trials for.
        training_data: DataFrame containing historical trial data with columns
            for intervention outcomes and targets.
        target_col: Name of the target column in the training data.

    Returns:
        Prior multiplier value clipped between 0.4 and 1.2. Returns base
        multiplier of 0.712 if insufficient similar trials are found.

    Example:
        >>> data = pd.DataFrame({
        ...     'intervention_outcome': [10.0, 12.0, 8.0],
        ...     'target': [7.1, 8.5, 5.8]
        ... })
        >>> calculate_contextual_prior(11.0, data, 'target')
        0.712
    """
    base_multiplier = 0.712
    intervention_outcome_col = "intervention_outcome"

    # Find similar trials
    intervention_diff = abs(
        training_data[intervention_outcome_col] - intervention_outcome
    )
    similar_mask = intervention_diff <= 2.0

    if similar_mask.sum() >= 3:
        similar_data = training_data[similar_mask]
        similar_multipliers = (
            similar_data[target_col] / similar_data[intervention_outcome_col]
        )
        # Filter reasonable multipliers
        valid_mult = similar_multipliers[
            (similar_multipliers > 0.3) & (similar_multipliers < 1.5)
        ]
        if len(valid_mult) >= 2:
            context_multiplier = valid_mult.mean()
            # Blend with base multiplier
            prior_multiplier = 0.7 * base_multiplier + 0.3 * context_multiplier
            return np.clip(prior_multiplier, 0.4, 1.2)

    return base_multiplier


def select_features_with_correlation_analysis(
    data: pd.DataFrame, all_features: list[str], target_col: str, max_features: int = 5
) -> list[str]:
    """Select features using multiple correlation measures.

    Evaluates features using both Pearson and Spearman correlation coefficients
    with the target variable. Combines both measures to create a robust ranking
    and selects the top features. Ensures intervention_outcome is always included
    if available.

    Args:
        data: DataFrame containing the feature and target data.
        all_features: List of all candidate feature column names to consider.
        target_col: Name of the target column to correlate features against.
        max_features: Maximum number of features to select. Defaults to 5.

    Returns:
        List of selected feature names, ranked by correlation strength.
        Always includes 'intervention_outcome' if present in data.

    Example:
        >>> data = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4],
        ...     'feature2': [4, 3, 2, 1],
        ...     'target': [2, 4, 6, 8],
        ...     'intervention_outcome': [1.5, 3.5, 5.5, 7.5]
        ... })
        >>> select_features_with_correlation_analysis(
        ...     data, ['feature1', 'feature2'], 'target', max_features=2
        ... )
        ['feature1', 'intervention_outcome']
    """
    feature_scores = []
    intervention_outcome_col = "intervention_outcome"

    for feature in all_features:
        if feature in data.columns and not data[feature].isna().all():
            try:
                # Multiple correlation measures
                pearson_corr, _ = pearsonr(data[feature], data[target_col])
                spearman_corr, _ = spearmanr(data[feature], data[target_col])

                # Combined robust score
                pearson_score = abs(pearson_corr) if not np.isnan(pearson_corr) else 0
                spearman_score = (
                    abs(spearman_corr) if not np.isnan(spearman_corr) else 0
                )

                # Average of both measures
                combined_score = (pearson_score + spearman_score) / 2
                feature_scores.append((feature, combined_score))

            except (ValueError, TypeError):
                continue

    # Sort by score and select top features
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [f for f, score in feature_scores if score > 0.1][:max_features]

    # Ensure intervention outcome is included
    if (
        intervention_outcome_col not in selected
        and intervention_outcome_col in data.columns
    ):
        if len(selected) == max_features:
            selected[-1] = intervention_outcome_col  # Replace lowest scoring
        else:
            selected.append(intervention_outcome_col)

    return selected


# Example usage:
"""
# For your CatBoost quantile results:
results_df = pd.DataFrame(results_catboost_quantile)
enhanced_metrics = calculate_perf_enhanced(results_df, "CatBoost Quantile")

# Print detailed summary
print_uncertainty_summary(results_df, "CatBoost Quantile")

# Compare multiple approaches
comparison_df = calculate_perf_comparison(
    [results_df_1, results_df_2],
    ["CatBoost Quantile", "Other Method"]
)
print(comparison_df)
"""
