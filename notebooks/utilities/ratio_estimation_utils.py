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
"""RCT Lean Ratio Estimation Helper functions."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def estimate_optimal_ratio_simple(
    training_df: pd.DataFrame,
    trial_id_col: str,
    control_arm_col: str,
    target_col: str,
    exclude_trial: str | None = None,
) -> float:
    """Estimate optimal ratio by analyzing control/treatment ratios in training data.

    This function calculates the median ratio of control to treatment outcomes
    across multiple trials, excluding the target trial to prevent data leakage.

    Args:
        training_df: DataFrame containing trial data with treatment and control arms.
        trial_id_col: Column name containing trial identifiers.
        control_arm_col: Column name indicating control arm (1 for control, 0 for treatment).
        target_col: Column name containing outcome values.
        exclude_trial: Trial ID to exclude from ratio estimation (prevents data leakage).

    Returns:
        Estimated optimal ratio (control/treatment). Returns 0.8 as fallback if no
        valid ratios can be calculated.
    """
    ratios = []

    for trial_id in training_df[trial_id_col].unique():
        if exclude_trial and trial_id == exclude_trial:
            continue

        trial_data = training_df[training_df[trial_id_col] == trial_id]

        # Get control and treatment outcomes
        control_outcomes = trial_data[trial_data[control_arm_col] == 1][target_col]
        treatment_outcomes = trial_data[trial_data[control_arm_col] != 1][target_col]

        if not control_outcomes.empty and not treatment_outcomes.empty:
            control_mean = control_outcomes.mean()
            treatment_mean = treatment_outcomes.mean()

            if treatment_mean > 0:  # Avoid division by zero
                ratio = control_mean / treatment_mean
                ratios.append(ratio)

    if ratios:
        return np.median(ratios)  # Use median for robustness
    else:
        return 0.8  # Fallback to default


def estimate_optimal_ratio_regression(
    training_df: pd.DataFrame,
    trial_id_col: str,
    control_arm_col: str,
    target_col: str,
    exclude_trial: str | None = None,
) -> float:
    """Use linear regression to estimate the relationship between treatment and control outcomes.

    Fits a linear regression model through the origin to estimate the proportional
    relationship between treatment and control outcomes across trials.

    Args:
        training_df: DataFrame containing trial data.
        trial_id_col: Column name containing trial identifiers.
        control_arm_col: Column name indicating control arm (1 for control, 0 for treatment).
        target_col: Column name containing outcome values.
        exclude_trial: Trial ID to exclude from estimation.

    Returns:
        Regression coefficient representing the optimal ratio, bounded between 0.1 and 2.0.
        Returns 0.8 as fallback if insufficient data for regression.
    """
    X_pairs = []
    y_pairs = []

    for trial_id in training_df[trial_id_col].unique():
        if exclude_trial and trial_id == exclude_trial:
            continue

        trial_data = training_df[training_df[trial_id_col] == trial_id]

        control_outcomes = trial_data[trial_data[control_arm_col] == 1][target_col]
        treatment_outcomes = trial_data[trial_data[control_arm_col] != 1][target_col]

        if not control_outcomes.empty and not treatment_outcomes.empty:
            treatment_mean = treatment_outcomes.mean()
            control_mean = control_outcomes.mean()

            X_pairs.append([treatment_mean])
            y_pairs.append(control_mean)

    if len(X_pairs) >= 2:  # Need at least 2 points for regression
        reg = LinearRegression(fit_intercept=False)  # Force through origin for ratio
        reg.fit(X_pairs, y_pairs)
        return max(0.1, min(2.0, reg.coef_[0]))  # Bound between 0.1 and 2.0
    else:
        return 0.8  # Fallback


def estimate_optimal_ratio_cv(
    training_df: pd.DataFrame, trial_id_col: str, control_arm_col: str, target_col: str
) -> float:
    """Use cross-validation approach to find optimal ratio that minimizes prediction error.

    Tests multiple candidate ratios and selects the one that minimizes mean squared
    error when predicting control outcomes from treatment outcomes.

    Args:
        training_df: DataFrame containing trial data.
        trial_id_col: Column name containing trial identifiers.
        control_arm_col: Column name indicating control arm (1 for control, 0 for treatment).
        target_col: Column name containing outcome values.

    Returns:
        Optimal ratio that minimizes prediction error across candidate values from
        0.3 to 1.5. Returns 0.8 as fallback if no valid errors can be calculated.
    """
    candidate_ratios = np.arange(0.3, 1.5, 0.05)  # Test ratios from 0.3 to 1.5
    ratio_errors = []

    for ratio in candidate_ratios:
        errors = []

        for trial_id in training_df[trial_id_col].unique():
            trial_data = training_df[training_df[trial_id_col] == trial_id]

            control_outcomes = trial_data[trial_data[control_arm_col] == 1][target_col]
            treatment_outcomes = trial_data[trial_data[control_arm_col] != 1][
                target_col
            ]

            if not control_outcomes.empty and not treatment_outcomes.empty:
                treatment_mean = treatment_outcomes.mean()
                control_mean = control_outcomes.mean()

                # Predict control using current ratio
                predicted_control = ratio * treatment_mean
                error = (predicted_control - control_mean) ** 2
                errors.append(error)

        if errors:
            ratio_errors.append(np.mean(errors))
        else:
            ratio_errors.append(float("inf"))

    if ratio_errors:
        best_idx = np.argmin(ratio_errors)
        return candidate_ratios[best_idx]
    else:
        return 0.8


def dynamic_ratio_method(
    training_df: pd.DataFrame,
    trial_id_col: str,
    control_arm_col: str,
    target_col: str,
    method: str = "cv",
) -> list[dict]:
    """Dynamic ratio method that estimates optimal ratio from training data.

    For each control arm observation, estimates the optimal control/treatment ratio
    using other trials in the dataset, then predicts the control outcome and
    calculates the Average Treatment Effect (ATE).

    Args:
        training_df: DataFrame containing trial data with treatment and control arms.
        trial_id_col: Column name containing trial identifiers.
        control_arm_col: Column name indicating control arm (1 for control, 0 for treatment).
        target_col: Column name containing outcome values.
        method: Ratio estimation method - 'simple', 'regression', or 'cv'.

    Returns:
        List of dictionaries containing ATE predictions and metadata for each
        control arm observation, including:
        - real_ate: True average treatment effect
        - pred_ate: Predicted average treatment effect
        - outcome_control: True control outcome
        - predicted_outcome: Predicted control outcome
        - estimated_ratio: Ratio used for prediction
        - rct_name: Trial identifier
        - intervention: Intervention type
        - Arm: Arm identifier
    """
    results_dynamic = []

    for _index, row in training_df.iterrows():
        rct_name = row[trial_id_col]
        is_arm_control = row[control_arm_col]

        if is_arm_control == 1:
            # Ground truth control outcome
            outcome_control = round(row[target_col], 2)

            # Get the treatment outcome of the RCT targeted
            trt_arm = training_df.loc[training_df[trial_id_col] == rct_name, :]
            trt_outcome = trt_arm.loc[trt_arm[control_arm_col] != 1, target_col]

            if trt_outcome.empty or pd.isna(trt_outcome.mean()):
                continue

            trt_outcome = round(trt_outcome.mean(), 2)

            # Estimate optimal ratio (excluding current trial to prevent leakage)
            if method == "simple":
                optimal_ratio = estimate_optimal_ratio_simple(
                    training_df,
                    trial_id_col,
                    control_arm_col,
                    target_col,
                    exclude_trial=rct_name,
                )
            elif method == "regression":
                optimal_ratio = estimate_optimal_ratio_regression(
                    training_df,
                    trial_id_col,
                    control_arm_col,
                    target_col,
                    exclude_trial=rct_name,
                )
            elif method == "cv":
                # For CV, we use all data except current trial
                training_subset = training_df[training_df[trial_id_col] != rct_name]
                optimal_ratio = estimate_optimal_ratio_cv(
                    training_subset, trial_id_col, control_arm_col, target_col
                )

            # Predict control arm using estimated ratio
            predicted_outcome = round(optimal_ratio * trt_outcome, 2)

            # Calculate ATEs
            real_ate = round(trt_outcome - outcome_control, 2)
            pred_ate = round(trt_outcome - predicted_outcome, 2)

            # Store results
            results_dynamic.append(
                {
                    "real_ate": real_ate,
                    "pred_ate": pred_ate,
                    "outcome_control": outcome_control,
                    "predicted_outcome": predicted_outcome,
                    "rct_name": rct_name,
                    "intervention": row["intervention"],
                    "Arm": row["Arm"],
                    "estimated_ratio": round(optimal_ratio, 3),
                }
            )

            print(
                f"{rct_name} - arm: {row['Arm']} - Dynamic Ratio ({optimal_ratio:.3f}) - intervention: {trt_outcome} - real_outcome: {outcome_control}, pred_outcome: {predicted_outcome}, real ATE: {real_ate} vs pred ATE: {pred_ate}"
            )

    return results_dynamic


def estimate_optimal_ratio_ols(
    training_df: pd.DataFrame,
    trial_id_col: str,
    control_arm_col: str,
    target_col: str,
    exclude_trial: str | None = None,
) -> dict[str, any]:
    """Use OLS regression to estimate the relationship between treatment and control outcomes.

    Fits an OLS model without intercept: Control_Outcome = β * Treatment_Outcome + ε
    The coefficient β represents the optimal ratio with statistical inference.

    Args:
        training_df: DataFrame containing trial data.
        trial_id_col: Column name containing trial identifiers.
        control_arm_col: Column name indicating control arm (1 for control, 0 for treatment).
        target_col: Column name containing outcome values.
        exclude_trial: Trial ID to exclude from estimation.

    Returns:
        Dictionary containing:
        - ratio: Estimated ratio coefficient
        - p_value: Statistical significance of the ratio
        - r_squared: Model fit quality
        - conf_int_lower: Lower bound of 95% confidence interval
        - conf_int_upper: Upper bound of 95% confidence interval
        - n_trials: Number of trials used in estimation
        - model: Fitted statsmodels OLS object (or None if fallback used)
    """
    treatment_outcomes = []
    control_outcomes = []

    for trial_id in training_df[trial_id_col].unique():
        if exclude_trial and trial_id == exclude_trial:
            continue

        trial_data = training_df[training_df[trial_id_col] == trial_id]

        # Get control and treatment outcomes for this trial
        control_data = trial_data[trial_data[control_arm_col] == 1][target_col]
        treatment_data = trial_data[trial_data[control_arm_col] != 1][target_col]

        if not control_data.empty and not treatment_data.empty:
            control_mean = control_data.mean()
            treatment_mean = treatment_data.mean()

            # Only include if treatment outcome is positive to avoid division issues
            if treatment_mean > 0:
                treatment_outcomes.append(treatment_mean)
                control_outcomes.append(control_mean)

    if len(treatment_outcomes) >= 2:  # Need at least 2 points for regression
        # Fit OLS model: Control = β * Treatment + ε
        # No intercept to force proportional relationship
        X = np.array(treatment_outcomes).reshape(-1, 1)
        y = np.array(control_outcomes)

        # Fit OLS model without intercept
        model = sm.OLS(y, X).fit()

        # Extract coefficient (ratio) and statistics
        ratio = model.params[0]
        p_value = model.pvalues[0]
        r_squared = model.rsquared
        conf_int = model.conf_int(alpha=0.05)  # 95% confidence interval

        # Bound the ratio to reasonable range
        ratio = max(0.1, min(2.5, ratio))

        return {
            "ratio": ratio,
            "p_value": p_value,
            "r_squared": r_squared,
            "conf_int_lower": conf_int[0][0],
            "conf_int_upper": conf_int[0][1],
            "n_trials": len(treatment_outcomes),
            "model": model,
        }
    else:
        # Fallback to simple median if insufficient data
        ratios = [
            c / t
            for c, t in zip(control_outcomes, treatment_outcomes, strict=False)
            if t > 0
        ]
        fallback_ratio = np.median(ratios) if ratios else 0.8

        return {
            "ratio": fallback_ratio,
            "p_value": None,
            "r_squared": None,
            "conf_int_lower": None,
            "conf_int_upper": None,
            "n_trials": len(treatment_outcomes),
            "model": None,
        }


def estimate_optimal_ratio_ols_with_intercept(
    training_df: pd.DataFrame,
    trial_id_col: str,
    control_arm_col: str,
    target_col: str,
    exclude_trial: str | None = None,
) -> dict[str, any]:
    """OLS regression with intercept to capture baseline effects.

    Fits model: Control_Outcome = α + β * Treatment_Outcome + ε
    This allows for non-proportional relationships and baseline effects.

    Args:
        training_df: DataFrame containing trial data.
        trial_id_col: Column name containing trial identifiers.
        control_arm_col: Column name indicating control arm (1 for control, 0 for treatment).
        target_col: Column name containing outcome values.
        exclude_trial: Trial ID to exclude from estimation.

    Returns:
        Dictionary containing intercept and slope parameters with statistical inference.
        Falls back to no-intercept model if insufficient data (< 3 trials).
    """
    treatment_outcomes = []
    control_outcomes = []

    for trial_id in training_df[trial_id_col].unique():
        if exclude_trial and trial_id == exclude_trial:
            continue

        trial_data = training_df[training_df[trial_id_col] == trial_id]

        control_data = trial_data[trial_data[control_arm_col] == 1][target_col]
        treatment_data = trial_data[trial_data[control_arm_col] != 1][target_col]

        if not control_data.empty and not treatment_data.empty:
            control_mean = control_data.mean()
            treatment_mean = treatment_data.mean()

            treatment_outcomes.append(treatment_mean)
            control_outcomes.append(control_mean)

    if len(treatment_outcomes) >= 3:  # Need more data for intercept model
        X = np.array(treatment_outcomes).reshape(-1, 1)
        X = sm.add_constant(X)  # Add intercept
        y = np.array(control_outcomes)

        model = sm.OLS(y, X).fit()

        intercept = model.params[0]
        slope = model.params[1]

        # For prediction, we use: Control = intercept + slope * Treatment
        # But we'll return both components
        return {
            "intercept": intercept,
            "slope": slope,
            "p_value_intercept": model.pvalues[0],
            "p_value_slope": model.pvalues[1],
            "r_squared": model.rsquared,
            "n_trials": len(treatment_outcomes),
            "model": model,
        }
    else:
        # Fallback to no-intercept model
        return estimate_optimal_ratio_ols(
            training_df, trial_id_col, control_arm_col, target_col, exclude_trial
        )


def ols_ratio_method(
    training_df: pd.DataFrame,
    trial_id_col: str,
    control_arm_col: str,
    target_col: str,
    use_intercept: bool = False,
) -> list[dict]:
    """Dynamic ratio method using OLS regression to estimate optimal ratio from training data.

    Uses ordinary least squares regression to estimate the relationship between
    treatment and control outcomes, with optional intercept term for baseline effects.

    Args:
        training_df: DataFrame containing trial data.
        trial_id_col: Column name containing trial identifiers.
        control_arm_col: Column name indicating control arm (1 for control, 0 for treatment).
        target_col: Column name containing outcome values.
        use_intercept: If True, fits model with intercept (Control = α + β * Treatment).
            If False, fits proportional model (Control = β * Treatment).

    Returns:
        List of dictionaries containing detailed ATE predictions and OLS statistics
        for each control arm observation, including confidence intervals, p-values,
        and model fit metrics.
    """
    results_ols = []

    for _index, row in training_df.iterrows():
        rct_name = row[trial_id_col]
        is_arm_control = row[control_arm_col]

        if is_arm_control == 1:
            # Ground truth control outcome
            outcome_control = round(row[target_col], 2)

            # Get the treatment outcome of the RCT targeted
            trt_arm = training_df.loc[training_df[trial_id_col] == rct_name, :]
            trt_outcome = trt_arm.loc[trt_arm[control_arm_col] != 1, target_col]

            if trt_outcome.empty or pd.isna(trt_outcome.mean()):
                continue

            trt_outcome = round(trt_outcome.mean(), 2)

            # Estimate optimal ratio using OLS (excluding current trial to prevent leakage)
            if use_intercept:
                ols_results = estimate_optimal_ratio_ols_with_intercept(
                    training_df,
                    trial_id_col,
                    control_arm_col,
                    target_col,
                    exclude_trial=rct_name,
                )

                # Predict using intercept model if available
                if "intercept" in ols_results and "slope" in ols_results:
                    predicted_outcome = (
                        ols_results["intercept"] + ols_results["slope"] * trt_outcome
                    )
                    estimated_ratio = ols_results["slope"]
                    model_type = "OLS_with_intercept"
                else:
                    # Fallback to proportional model
                    predicted_outcome = ols_results["ratio"] * trt_outcome
                    estimated_ratio = ols_results["ratio"]
                    model_type = "OLS_proportional_fallback"
            else:
                ols_results = estimate_optimal_ratio_ols(
                    training_df,
                    trial_id_col,
                    control_arm_col,
                    target_col,
                    exclude_trial=rct_name,
                )
                predicted_outcome = ols_results["ratio"] * trt_outcome
                estimated_ratio = ols_results["ratio"]
                model_type = "OLS_proportional"

            predicted_outcome = round(predicted_outcome, 2)

            # Calculate ATEs
            real_ate = round(trt_outcome - outcome_control, 2)
            pred_ate = round(trt_outcome - predicted_outcome, 2)

            # Store results with OLS statistics
            result_dict = {
                "real_ate": real_ate,
                "pred_ate": pred_ate,
                "outcome_control": outcome_control,
                "predicted_outcome": predicted_outcome,
                "rct_name": rct_name,
                "intervention": row["intervention"],
                "Arm": row["Arm"],
                "estimated_ratio": round(estimated_ratio, 4),
                "model_type": model_type,
                "n_trials_used": ols_results["n_trials"],
            }

            # Add OLS-specific statistics
            if ols_results.get("p_value") is not None:
                result_dict.update(
                    {
                        "p_value": round(ols_results["p_value"], 4),
                        "r_squared": round(ols_results["r_squared"], 4),
                        "conf_int_lower": round(ols_results["conf_int_lower"], 4),
                        "conf_int_upper": round(ols_results["conf_int_upper"], 4),
                    }
                )

            if use_intercept and "intercept" in ols_results:
                result_dict.update(
                    {
                        "intercept": round(ols_results["intercept"], 4),
                        "p_value_intercept": round(ols_results["p_value_intercept"], 4),
                        "p_value_slope": round(ols_results["p_value_slope"], 4),
                    }
                )

            results_ols.append(result_dict)

            # Print detailed results
            stats_str = ""
            if ols_results.get("r_squared") is not None:
                stats_str = (
                    f" (R²={ols_results['r_squared']:.3f}, n={ols_results['n_trials']})"
                )

            print(
                f"{rct_name} - {model_type} - Ratio: {estimated_ratio:.4f}{stats_str}"
            )
            print(
                f"  Treatment: {trt_outcome}, Real Control: {outcome_control}, Predicted Control: {predicted_outcome}"
            )
            print(f"  Real ATE: {real_ate}, Predicted ATE: {pred_ate}")
            print("-" * 80)

    return results_ols


def analyze_ols_results(results_ols: list[dict]) -> pd.DataFrame | None:
    """Analyze the OLS results and provide summary statistics.

    Computes comprehensive summary statistics for OLS ratio estimation results,
    including prediction accuracy metrics and model quality assessments.

    Args:
        results_ols: List of dictionaries containing OLS ratio estimation results
            from ols_ratio_method function.

    Returns:
        DataFrame containing the results for further analysis, or None if no
        results to analyze. Prints detailed summary statistics to console.
    """
    if not results_ols:
        print("No results to analyze.")
        return None

    df_results = pd.DataFrame(results_ols)

    print("=== OLS Ratio Estimation Analysis ===")
    print(f"Total predictions: {len(df_results)}")
    print(f"Average estimated ratio: {df_results['estimated_ratio'].mean():.4f}")
    print(f"Ratio std deviation: {df_results['estimated_ratio'].std():.4f}")
    print(
        f"Ratio range: [{df_results['estimated_ratio'].min():.4f}, {df_results['estimated_ratio'].max():.4f}]"
    )

    # Calculate prediction accuracy metrics
    mae = np.mean(np.abs(df_results["real_ate"] - df_results["pred_ate"]))
    mse = np.mean((df_results["real_ate"] - df_results["pred_ate"]) ** 2)
    rmse = np.sqrt(mse)

    print("\nPrediction Accuracy:")
    print(f"Mean Absolute Error (ATE): {mae:.4f}")
    print(f"Root Mean Square Error (ATE): {rmse:.4f}")

    # Analyze model quality where available
    valid_r2 = df_results["r_squared"].dropna()
    if not valid_r2.empty:
        print("\nOLS Model Quality:")
        print(f"Average R-squared: {valid_r2.mean():.4f}")
        print(f"R-squared range: [{valid_r2.min():.4f}, {valid_r2.max():.4f}]")

    valid_pvals = df_results["p_value"].dropna()
    if not valid_pvals.empty:
        significant = (valid_pvals < 0.05).sum()
        print(
            f"Statistically significant ratios (p<0.05): {significant}/{len(valid_pvals)}"
        )

    return df_results
