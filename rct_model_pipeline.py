#!/usr/bin/env python3
"""
RCT Lean Modeling Pipeline

The module ingests clinical trial data from Excel files,
runs a machine learning model for predicting control arm outcomes,
and outputs the results in JSON format.

This module currently implements CatBoost Quantile Regression approach, which showed
the best performance in terms of uncertainty quantification and directional accuracy
for Average Treatment Effect (ATE) predictions.

Usage:
    python rct_model_pipeline.py --excel_path "path/to/trials.xlsx" --output_path "results.json"
    
    Or programmatically:
    pipeline = RCTModelPipeline()
    results = pipeline.run_full_pipeline("trials.xlsx", "results.json")
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ast
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy import stats
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add the notebooks directory to the path to import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'notebooks'))

try:
    from utilities.utils import prepare_data, run_fillna, calculate_perf_enhanced_with_iou
    from utilities.catboost_quantile_utils import CatBoostQuantilePredictor
    from utilities.ratio_estimation_utils import dynamic_ratio_method, ols_ratio_method
except ImportError as e:
    print(f"Warning: Could not import utility functions: {e}")
    print("Some advanced features may not be available.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class RCTModelPipeline:
    """
    Complete pipeline for RCT Lean modeling with CatBoost Quantile Regression.
    
    This class handles the entire workflow from data ingestion to model training
    and prediction generation, implementing the best-performing approach from the
    research notebooks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RCT modeling pipeline.
        
        Args:
            config_path: Path to feature configuration JSON file. If None, uses default config.
        """
        self.feature_config = self._load_feature_config(config_path)
        self.model = None
        self.processed_data = None
        self.target_col = self.feature_config.get('target', 'PFS_median_months')
        self.trial_id_col = self.feature_config.get('trial_id_column', 'NCT_ID')
        self.control_arm_col = self.feature_config.get('control_arm_column', 'is_arm_control')
        self.intervention_outcome_col = self.feature_config.get('intervention_outcome_column', 'intervention_outcome')
        
        logger.info("RCT Model Pipeline initialized")
        
    def _load_feature_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load feature configuration from JSON file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration if no file provided
        return {
            "target": "PFS_median_months",
            "target_ci": "PFS_median_CI",
            "trial_id_column": "NCT_ID",
            "control_arm_column": "is_arm_control",
            "intervention_outcome_column": "intervention_outcome",
            "feature_groups_to_use": ["patient_demographics", "disease_characteristics", "drug_class", "treatment_features"],
            "features": {
                "patient_demographics": [
                    "gender_male_percent", "age_median", "no_smoker_percent", "ecog_1", "smoker_percent"
                ],
                "disease_characteristics": [
                    "brain_metastase_yes", "disease_stage_recurrent", "disease_stage_III", 
                    "disease_stage_IV", "EGFR_wild", "EGFR_positive_mutation"
                ],
                "drug_class": [
                    "EGFR_TKI", "Platinum_Chemotherapy", "Anti_VEGF", "PD1_PDL1_Inhibitor",
                    "Antimetabolite", "Taxane", "Antibody", "Placebo_Supportive-Care",
                    "Chemotherapy", "Targeted_Therapy", "Immunotherapy", "Anti-angiogenic_Other"
                ],
                "treatment_features": [
                    "combo_therapy", "treatment_complexity", "RCT_with_control_inter",
                    "is_arm_control", "Population", "First-in-Class", "Next-Generation"
                ]
            },
            "interaction_terms": [
                {"name": "int_outcome_x_egfr", "feature1": "intervention_outcome", "feature2": "EGFR_TKI"},
                {"name": "int_outcome_x_immuno", "feature1": "intervention_outcome", "feature2": "PD1_PDL1_Inhibitor"}
            ],
            "similarity_features": {
                "drug_class": [
                    "EGFR_TKI", "Platinum_Chemotherapy", "Anti_VEGF", "PD1_PDL1_Inhibitor",
                    "Antimetabolite", "Taxane", "Antibody", "Placebo_Supportive-Care"
                ]
            }
        }
    
    def load_and_preprocess_data(self, excel_path: str) -> pd.DataFrame:
        """
        Load and preprocess data from Excel file.
        
        Args:
            excel_path: Path to the Excel file containing trial data
            
        Returns:
            Preprocessed DataFrame ready for modeling
        """
        logger.info(f"Loading data from {excel_path}")
        
        try:
            # Load trial data from two sheets (as in the original notebook)
            # TODO: This needs to be updated based on the input provided from the UI --> probably would be an user input on UI
            df_1 = pd.read_excel(excel_path, sheet_name="trials by arm", skiprows=2)
            
            # Fill NA values with 0 for specific columns
            na_fill_cols = ['brain_metastase_yes', 'disease_stage_recurrent', 'disease_stage_III', 
                            'disease_stage_IV', 'EGFR_wild', 'no_smoker_percent']
            df_1[na_fill_cols] = df_1[na_fill_cols].fillna(0)
            
            # Load and process second data sheet
            # TODO: This needs to be updated based on the input provided from the UI --> probably will be received from agent through s3
            df_2 = pd.read_excel(excel_path, sheet_name="250529_NSCLC", skiprows=2)
            df_2 = df_2[df_2['to keep'] == 1.0]
            df_2 = df_2.rename(columns={'arm_n': 'Population'})
            df_2['NCT_ID'] = df_2['NCT_ID'].ffill()
            
            # Combine the two datasets
            df = pd.concat([df_1, df_2], ignore_index=True, axis=0)
            df['age_median'] = df['age_median'].fillna(df['age_clean'])
            
            logger.info(f"Combined dataset shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
        
        # Select relevant columns for analysis
        cols = self._get_required_columns()
        training_df = df[cols].copy()
        
        # Add control arm indicator
        training_df["is_arm_control"] = (training_df["Arm"] == 'Control').astype(int)
        
        # Remove rows that need to be dropped
        training_df = training_df.loc[training_df.get("need_to_be_dropped", 0) != 1, :]
        
        # Clean problematic values
        training_df = training_df.replace(["need help", "?", "??"], None)
        
        # Special handling for EGFR_wild column
        if "EGFR_wild" in training_df.columns:
            training_df["EGFR_wild"] = training_df["EGFR_wild"].fillna(0)
        
        # Add engineered features
        training_df = self._add_features(training_df)
        
        # Handle PFS_median_CI column if it exists and is stored as strings
        if 'PFS_median_CI' in training_df.columns:
            training_df['PFS_median_CI'] = training_df['PFS_median_CI'].apply(self._safe_literal_eval)
        
        logger.info(f"Preprocessed dataset shape: {training_df.shape}")
        logger.info(f"Number of unique trials: {training_df[self.trial_id_col].nunique()}")
        
        self.processed_data = training_df
        return training_df
    
    # TODO: currently hardcoding to test module --> should be parameterized later
    def _get_required_columns(self) -> List[str]:
        """Get the list of required columns for analysis."""
        return [
            "NCT_ID", "Arm", "Population", "intervention", "RCT_with_control_inter",
            "gender_male_percent", "age_median", "no_smoker_percent", "ecog_1",
            "brain_metastase_yes", "disease_stage_recurrent", "disease_stage_III", 
            "disease_stage_IV", "EGFR_wild", "EGFR_positive_mutation",
            "PFS_median_months", "PFS_median_CI", "CI", "need_to_be_dropped",
            "First-in-Class", "Next-Generation",
            "EGFR_TKI", "Platinum_Chemotherapy", "Anti_VEGF", "PD1_PDL1_Inhibitor",
            "Antimetabolite", "Taxane", "Antibody", "Placebo_Supportive-Care",
            "Chemotherapy", "Targeted_Therapy", "Immunotherapy", "Anti-angiogenic_Other",
            "subgroup (Y/N)", "Subgroup characteristics"
        ]
    
    def _safe_literal_eval(self, val):
        """Safely convert string to list."""
        if pd.isna(val):
            return np.nan
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    
    # TODO: currently using same code as in notebooks --> should be dynamic and change based on indication and features
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the dataframe."""
        df = df.copy()
        
        # Treatment combinations
        df['combo_therapy'] = ((df.get('Chemotherapy', 0) + df.get('Targeted_Therapy', 0) + 
                              df.get('Immunotherapy', 0) + df.get('Anti-angiogenic_Other', 0)) > 1).astype(int)
        
        # EGFR status interaction with treatments
        df['egfr_targeted'] = df.get('EGFR_positive_mutation', 0) * df.get('Targeted_Therapy', 0)
        df['egfr_tki_use'] = df.get('EGFR_positive_mutation', 0) * df.get('EGFR_TKI', 0)
        
        # Patient risk profile
        df['high_risk_profile'] = ((df.get('brain_metastase_yes', 0) > 0) | 
                                  (df.get('disease_stage_IV', 0) > 0)).astype(int)
        
        # Treatment novelty score
        df['novelty_score'] = df.get('First-in-Class', 0) + df.get('Next-Generation', 0)
        
        # Patient demographics composite
        df['elderly_male'] = ((df.get('gender_male_percent', 0) > 60) & 
                             (df.get('age_median', 0) > 60)).astype(int)
        
        # Trial size category
        if 'Population' in df.columns:
            df['large_trial'] = (df['Population'] > df['Population'].median()).astype(int)
        
        # Treatment complexity
        treatment_cols = ['EGFR_TKI', 'Anti_VEGF', 'PD1_PDL1_Inhibitor', 
                          'Antimetabolite', 'Taxane', 'Antibody']
        available_treatment_cols = [col for col in treatment_cols if col in df.columns]
        df['treatment_complexity'] = df[available_treatment_cols].sum(axis=1)
        
        # Calculate percentage of smokers
        if 'no_smoker_percent' in df.columns:
            df['smoker_percent'] = 100 - df['no_smoker_percent']
        
        return df
    
    def get_feature_list(self) -> List[str]:
        """Get the complete list of features to use for modeling."""
        feature_groups_to_use = self.feature_config.get('feature_groups_to_use', 
                                                        list(self.feature_config['features'].keys()))
        
        ftr = []
        for group in feature_groups_to_use:
            ftr.extend(self.feature_config['features'].get(group, []))
        
        return ftr
    
    def run_catboost_quantile_model(self, training_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run the CatBoost Quantile Regression model using Leave-One-Out protocol.
        
        Args:
            training_df: Preprocessed training data
            
        Returns:
            List of prediction results with uncertainty quantification
        """
        logger.info("Running CatBoost Quantile Regression model")
        
        ftr = self.get_feature_list()
        interaction_terms = self.feature_config.get('interaction_terms', [])
        results = []
        
        for index, row in training_df.iterrows():
            rct_name = row[self.trial_id_col]
            is_arm_control = row[self.control_arm_col]
            
            if is_arm_control == 1:
                # Ground truth
                outcome_control = round(row[self.target_col], 2)
                
                # Get the treatment outcome of the RCT targeted
                trt_arm = training_df.loc[training_df[self.trial_id_col] == rct_name, :]
                trt_outcome = trt_arm.loc[trt_arm[self.control_arm_col] != 1, self.target_col]
                
                if trt_outcome.empty or pd.isna(trt_outcome.mean()):
                    continue
                    
                trt_outcome = round(trt_outcome.mean(), 2)
                
                # Prepare data for CatBoost
                training_j = training_df.copy()
                
                # Flag inference row and target RCT
                training_j["is_to_predict"] = np.where(training_j.index == index, 1, 0)
                training_j["is_targeted_rct"] = np.where(training_j[self.trial_id_col] == rct_name, 1, 0)

                # Add intervention outcome as a feature for all control arms
                for idx, control_row in training_j[training_j[self.control_arm_col] == 1].iterrows():
                    ctrl_rct = control_row[self.trial_id_col]
                    ctrl_trt_outcome = training_j.loc[(training_j[self.trial_id_col] == ctrl_rct) & 
                                                     (training_j[self.control_arm_col] != 1), self.target_col]
                    
                    if not ctrl_trt_outcome.empty and not pd.isna(ctrl_trt_outcome.mean()):
                        training_j.loc[idx, self.intervention_outcome_col] = round(ctrl_trt_outcome.mean(), 2)
                    else:
                        training_j.loc[idx, self.intervention_outcome_col] = np.nan
                
                training_j = training_j.dropna(subset=[self.intervention_outcome_col])
                
                # Create interaction terms
                for f in interaction_terms:
                    if f['feature1'] in training_j.columns and f['feature2'] in training_j.columns:
                        training_j[f['name']] = training_j[f['feature1']] * training_j[f['feature2']]
                
                # Only consider control arms for training
                training_j = training_j.loc[training_j[self.control_arm_col] == 1, :]
                
                # Prepare features
                feature_list = ftr + ["is_to_predict", "is_targeted_rct", self.intervention_outcome_col]
                if interaction_terms:
                    feature_list += [f['name'] for f in interaction_terms if f['name'] in training_j.columns]
                feature_list += [self.target_col]
                
                training_j = prepare_data(training_j, feature_list)
                training_j = run_fillna(training_j)

                # Make sure the target row has the intervention outcome
                training_j.loc[training_j["is_to_predict"] == 1, self.intervention_outcome_col] = trt_outcome
                
                # Separate inference and training data
                inference_df = training_j.loc[training_j["is_to_predict"] == 1, :]
                if inference_df.empty:
                    logger.warning(f"Skipping {rct_name} as no inference data available.")
                    continue
                    
                # Remove target trial from training
                training_subset = training_j.loc[training_j["is_to_predict"] != 1, :]
                training_subset = training_subset.loc[training_subset["is_targeted_rct"] != 1, :]
                
                if len(training_subset) < 5:
                    logger.warning(f"Skipping {rct_name} - insufficient training data ({len(training_subset)} samples)")
                    continue
                
                # Features for modeling
                model_features = ftr + [self.intervention_outcome_col]
                if interaction_terms:
                    model_features += [f['name'] for f in interaction_terms if f['name'] in training_subset.columns]
                
                try:
                    # Fit CatBoost quantile models
                    quantile_predictor = CatBoostQuantilePredictor()
                    quantile_predictor.fit(
                        training_subset[model_features], 
                        training_subset[self.target_col]
                    )
                    
                    # Get distributional prediction
                    pred_distribution = quantile_predictor.predict_distribution(
                        inference_df[model_features]
                    )
                    
                    # Use median as point estimate
                    predicted_outcome = round(pred_distribution['point_estimate'], 2)
                    
                    # Calculate ATEs
                    real_ate = round(trt_outcome - outcome_control, 2)
                    pred_ate = round(trt_outcome - predicted_outcome, 2)

                    # Calculate ATE distribution
                    ate_samples = trt_outcome - pred_distribution['samples']
                    ate_mean = round(np.mean(ate_samples), 2)
                    ate_std = round(np.std(ate_samples), 2)
                    ate_ci_95 = [
                        round(np.percentile(ate_samples, 2.5), 2),
                        round(np.percentile(ate_samples, 97.5), 2)
                    ]
                    
                    # Probability of positive ATE
                    prob_positive_ate = round(np.mean(ate_samples > 0), 3)
                    
                    # Get actual CI for IoU calculation if available
                    actual_outcome_ci = None
                    if "PFS_median_CI" in training_df.columns:
                        actual_ci_row = training_df.loc[training_df.index == index, "PFS_median_CI"]
                        if not actual_ci_row.empty:
                            actual_outcome_ci = actual_ci_row.iloc[0]
                    
                    result = {
                        "rct_name": rct_name,
                        "intervention": row["intervention"],
                        "arm": row["Arm"],
                        "real_ate": real_ate,
                        "pred_ate": pred_ate,
                        "outcome_control": outcome_control,
                        "predicted_outcome": predicted_outcome,
                        "intervention_outcome": trt_outcome,
                        "ate_mean": ate_mean,
                        "ate_std": ate_std,
                        "ate_ci_95": ate_ci_95,
                        "prob_positive_ate": prob_positive_ate,
                        "pred_ci_95": pred_distribution['ci_95'],
                        "pred_std": round(pred_distribution['std'], 2),
                        "pred_median": round(pred_distribution['median'], 2),
                        "n_training_samples": len(training_subset),
                        "actual_outcome_ci": actual_outcome_ci,
                        "quantile_predictions": pred_distribution['quantiles'],
                        "prediction_method": "catboost_quantile"
                    }
                    
                    results.append(result)
                    
                    logger.info(f"Processed {rct_name}: Real ATE={real_ate}, Pred ATE={pred_ate}, P(ATE>0)={prob_positive_ate}")
                    
                except Exception as e:
                    logger.error(f"Error processing {rct_name}: {str(e)}")
                    continue
        
        logger.info(f"Completed modeling for {len(results)} trials")
        return results
    
    def calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for the model results."""
        if not results:
            return {"error": "No results to evaluate"}
        
        results_df = pd.DataFrame(results)
        
        # Basic metrics
        r2_ate = round(r2_score(results_df["real_ate"], results_df["pred_ate"]), 3)
        r2_outcome = round(r2_score(results_df["outcome_control"], results_df["predicted_outcome"]), 3)
        
        # RMSE calculations
        rmse_ate = round(np.sqrt(mean_squared_error(results_df["real_ate"], results_df["pred_ate"])), 3)
        rmse_outcome = round(np.sqrt(mean_squared_error(results_df["outcome_control"], results_df["predicted_outcome"])), 3)
        
        # Directional accuracy
        helper = results_df["real_ate"] * results_df["pred_ate"]
        directional_accuracy = round((helper > 0).mean(), 3)
        
        # Spearman correlation
        spearman_corr, _ = stats.spearmanr(results_df["pred_ate"], results_df["real_ate"])
        spearman_corr = round(spearman_corr, 3)
        
        # Coverage metrics if available
        coverage_metrics = {}
        if 'ate_ci_95' in results_df.columns:
            # Check if real ATE falls within predicted CI
            within_ci = []
            for _, row in results_df.iterrows():
                ci = row['ate_ci_95']
                real_ate = row['real_ate']
                if isinstance(ci, list) and len(ci) == 2:
                    within_ci.append(ci[0] <= real_ate <= ci[1])
            
            if within_ci:
                coverage_metrics['ate_coverage_95'] = round(np.mean(within_ci), 3)
        
        # Average confidence interval width
        if 'pred_ci_95' in results_df.columns:
            ci_widths = []
            for _, row in results_df.iterrows():
                ci = row['pred_ci_95']
                if isinstance(ci, list) and len(ci) == 2:
                    ci_widths.append(ci[1] - ci[0])
            
            if ci_widths:
                coverage_metrics['avg_ci_width'] = round(np.mean(ci_widths), 3)
        
        # Probability metrics
        prob_metrics = {}
        if 'prob_positive_ate' in results_df.columns:
            avg_prob_positive = round(results_df['prob_positive_ate'].mean(), 3)
            prob_metrics['avg_prob_positive_ate'] = avg_prob_positive
            
            # Accuracy of probability predictions
            actual_positive = (results_df['real_ate'] > 0).astype(int)
            prob_accuracy = []
            for _, row in results_df.iterrows():
                prob = row['prob_positive_ate']
                actual = 1 if row['real_ate'] > 0 else 0
                # Use probability as confidence - closer to 0.5 means less confident
                prob_accuracy.append(1 - abs(prob - actual))
            
            prob_metrics['prob_calibration'] = round(np.mean(prob_accuracy), 3)
        
        return {
            "model_performance": {
                "r2_ate": r2_ate,
                "r2_outcome": r2_outcome,
                "rmse_ate": rmse_ate,
                "rmse_outcome": rmse_outcome,
                "directional_accuracy": directional_accuracy,
                "spearman_correlation": spearman_corr,
                **coverage_metrics,
                **prob_metrics
            },
            "summary_statistics": {
                "n_trials": len(results),
                "avg_real_ate": round(results_df["real_ate"].mean(), 3),
                "avg_pred_ate": round(results_df["pred_ate"].mean(), 3),
                "std_real_ate": round(results_df["real_ate"].std(), 3),
                "std_pred_ate": round(results_df["pred_ate"].std(), 3),
                "positive_ate_trials": int((results_df["real_ate"] > 0).sum()),
                "predicted_positive_ate_trials": int((results_df["pred_ate"] > 0).sum())
            }
        }
    
    def run_comparative_models(self, training_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run multiple models for comparison purposes.
        
        Args:
            training_df: Preprocessed training data
            
        Returns:
            Dictionary containing results from different modeling approaches
        """
        logger.info("Running comparative models")
        
        models = {
            "catboost_quantile": self.run_catboost_quantile_model(training_df),
            "simple_average": self._run_simple_average_model(training_df),
            "ratio_method": self._run_ratio_method(training_df),
            "random_forest": self._run_random_forest_model(training_df),
            "lasso_regression": self._run_lasso_model(training_df)
        }
        
        return models
    
    def _run_simple_average_model(self, training_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run the simple average baseline model."""
        results = []
        
        for index, row in training_df.iterrows():
            rct_name = row[self.trial_id_col]
            is_arm_control = row[self.control_arm_col]
            
            if is_arm_control == 1:
                outcome_control = round(row[self.target_col], 2)
                
                # Get treatment outcome
                trt_arm = training_df.loc[training_df[self.trial_id_col] == rct_name, :]
                trt_outcome = trt_arm.loc[trt_arm[self.control_arm_col] != 1, self.target_col]
                
                if trt_outcome.empty or pd.isna(trt_outcome.mean()):
                    continue
                    
                trt_outcome = round(trt_outcome.mean(), 2)
                
                # Use average of other control arms
                other_controls = training_df[(training_df[self.control_arm_col] == 1) & 
                                           (training_df[self.trial_id_col] != rct_name)]
                predicted_outcome = round(other_controls[self.target_col].mean(), 2)
                
                real_ate = round(trt_outcome - outcome_control, 2)
                pred_ate = round(trt_outcome - predicted_outcome, 2)
                
                results.append({
                    "rct_name": rct_name,
                    "intervention": row["intervention"],
                    "arm": row["Arm"],
                    "real_ate": real_ate,
                    "pred_ate": pred_ate,
                    "outcome_control": outcome_control,
                    "predicted_outcome": predicted_outcome,
                    "intervention_outcome": trt_outcome,
                    "prediction_method": "simple_average"
                })
        
        return results
    
    def _run_ratio_method(self, training_df: pd.DataFrame, ratio: float = 0.7) -> List[Dict[str, Any]]:
        """Run the ratio-based baseline model."""
        results = []
        
        for index, row in training_df.iterrows():
            rct_name = row[self.trial_id_col]
            is_arm_control = row[self.control_arm_col]
            
            if is_arm_control == 1:
                outcome_control = round(row[self.target_col], 2)
                
                # Get treatment outcome
                trt_arm = training_df.loc[training_df[self.trial_id_col] == rct_name, :]
                trt_outcome = trt_arm.loc[trt_arm[self.control_arm_col] != 1, self.target_col]
                
                if trt_outcome.empty or pd.isna(trt_outcome.mean()):
                    continue
                    
                trt_outcome = round(trt_outcome.mean(), 2)
                
                # Predict control as ratio of treatment
                predicted_outcome = round(ratio * trt_outcome, 2)
                
                real_ate = round(trt_outcome - outcome_control, 2)
                pred_ate = round(trt_outcome - predicted_outcome, 2)
                
                results.append({
                    "rct_name": rct_name,
                    "intervention": row["intervention"],
                    "arm": row["Arm"],
                    "real_ate": real_ate,
                    "pred_ate": pred_ate,
                    "outcome_control": outcome_control,
                    "predicted_outcome": predicted_outcome,
                    "intervention_outcome": trt_outcome,
                    "ratio_used": ratio,
                    "prediction_method": "ratio_method"
                })
        
        return results
    
    def _run_random_forest_model(self, training_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run Random Forest model with intervention outcome conditioning."""
        results = []
        ftr = self.get_feature_list()
        
        for index, row in training_df.iterrows():
            rct_name = row[self.trial_id_col]
            is_arm_control = row[self.control_arm_col]
            
            if is_arm_control == 1:
                outcome_control = round(row[self.target_col], 2)
                
                # Get treatment outcome
                trt_arm = training_df.loc[training_df[self.trial_id_col] == rct_name, :]
                trt_outcome = trt_arm.loc[trt_arm[self.control_arm_col] != 1, self.target_col]
                
                if trt_outcome.empty or pd.isna(trt_outcome.mean()):
                    continue
                    
                trt_outcome = round(trt_outcome.mean(), 2)
                
                # Prepare data similar to CatBoost approach
                training_j = training_df.copy()
                training_j["is_to_predict"] = np.where(training_j.index == index, 1, 0)
                training_j["is_targeted_rct"] = np.where(training_j[self.trial_id_col] == rct_name, 1, 0)

                # Add intervention outcomes
                for idx, control_row in training_j[training_j[self.control_arm_col] == 1].iterrows():
                    ctrl_rct = control_row[self.trial_id_col]
                    ctrl_trt_outcome = training_j.loc[(training_j[self.trial_id_col] == ctrl_rct) & 
                                                     (training_j[self.control_arm_col] != 1), self.target_col]
                    
                    if not ctrl_trt_outcome.empty and not pd.isna(ctrl_trt_outcome.mean()):
                        training_j.loc[idx, self.intervention_outcome_col] = round(ctrl_trt_outcome.mean(), 2)
                    else:
                        training_j.loc[idx, self.intervention_outcome_col] = np.nan
                
                training_j = training_j.dropna(subset=[self.intervention_outcome_col])
                training_j = training_j.loc[training_j[self.control_arm_col] == 1, :]
                training_j = prepare_data(training_j, ftr + ["is_to_predict", "is_targeted_rct", 
                                                           self.intervention_outcome_col, self.target_col])
                training_j = run_fillna(training_j)
                training_j.loc[training_j["is_to_predict"] == 1, self.intervention_outcome_col] = trt_outcome
                
                inference_df = training_j.loc[training_j["is_to_predict"] == 1, :]
                if inference_df.empty:
                    continue
                    
                training_subset = training_j.loc[training_j["is_to_predict"] != 1, :]
                training_subset = training_subset.loc[training_subset["is_targeted_rct"] != 1, :]
                
                if len(training_subset) < 5:
                    continue
                
                try:
                    # Train Random Forest
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, 
                                                   max_depth=10, min_samples_split=10, min_samples_leaf=5)
                    model_features = ftr + [self.intervention_outcome_col]
                    rf_model.fit(training_subset[model_features], training_subset[self.target_col])
                    predicted_outcome = round(rf_model.predict(inference_df[model_features])[0], 2)
                    
                    real_ate = round(trt_outcome - outcome_control, 2)
                    pred_ate = round(trt_outcome - predicted_outcome, 2)
                    
                    results.append({
                        "rct_name": rct_name,
                        "intervention": row["intervention"],
                        "arm": row["Arm"],
                        "real_ate": real_ate,
                        "pred_ate": pred_ate,
                        "outcome_control": outcome_control,
                        "predicted_outcome": predicted_outcome,
                        "intervention_outcome": trt_outcome,
                        "n_training_samples": len(training_subset),
                        "prediction_method": "random_forest"
                    })
                    
                except Exception:
                    continue
        
        return results
    
    def _run_lasso_model(self, training_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run Lasso regression model with intervention outcome conditioning."""
        results = []
        ftr = self.get_feature_list()
        interaction_terms = self.feature_config.get('interaction_terms', [])
        
        for index, row in training_df.iterrows():
            rct_name = row[self.trial_id_col]
            is_arm_control = row[self.control_arm_col]
            
            if is_arm_control == 1:
                outcome_control = round(row[self.target_col], 2)
                
                # Get treatment outcome
                trt_arm = training_df.loc[training_df[self.trial_id_col] == rct_name, :]
                trt_outcome = trt_arm.loc[trt_arm[self.control_arm_col] != 1, self.target_col]
                
                if trt_outcome.empty or pd.isna(trt_outcome.mean()):
                    continue
                    
                trt_outcome = round(trt_outcome.mean(), 2)
                
                # Prepare data
                training_j = training_df.copy()
                training_j["is_to_predict"] = np.where(training_j.index == index, 1, 0)
                training_j["is_targeted_rct"] = np.where(training_j[self.trial_id_col] == rct_name, 1, 0)

                # Add intervention outcomes
                for idx, control_row in training_j[training_j[self.control_arm_col] == 1].iterrows():
                    ctrl_rct = control_row[self.trial_id_col]
                    ctrl_trt_outcome = training_j.loc[(training_j[self.trial_id_col] == ctrl_rct) & 
                                                     (training_j[self.control_arm_col] != 1), self.target_col]
                    
                    if not ctrl_trt_outcome.empty and not pd.isna(ctrl_trt_outcome.mean()):
                        training_j.loc[idx, self.intervention_outcome_col] = round(ctrl_trt_outcome.mean(), 2)
                    else:
                        training_j.loc[idx, self.intervention_outcome_col] = np.nan
                
                training_j = training_j.dropna(subset=[self.intervention_outcome_col])
                
                # Create interaction terms
                for f in interaction_terms:
                    if f['feature1'] in training_j.columns and f['feature2'] in training_j.columns:
                        training_j[f['name']] = training_j[f['feature1']] * training_j[f['feature2']]
                
                training_j = training_j.loc[training_j[self.control_arm_col] == 1, :]
                
                feature_list = ftr + ["is_to_predict", "is_targeted_rct", self.intervention_outcome_col]
                if interaction_terms:
                    feature_list += [f['name'] for f in interaction_terms if f['name'] in training_j.columns]
                feature_list += [self.target_col]
                
                training_j = prepare_data(training_j, feature_list)
                training_j = run_fillna(training_j)
                training_j.loc[training_j["is_to_predict"] == 1, self.intervention_outcome_col] = trt_outcome
                
                inference_df = training_j.loc[training_j["is_to_predict"] == 1, :]
                if inference_df.empty:
                    continue
                    
                training_subset = training_j.loc[training_j["is_to_predict"] != 1, :]
                training_subset = training_subset.loc[training_subset["is_targeted_rct"] != 1, :]
                
                if len(training_subset) < 5:
                    continue
                
                try:
                    # Train Lasso
                    lasso_model = linear_model.Lasso(alpha=0.4)
                    model_features = ftr + [self.intervention_outcome_col]
                    if interaction_terms:
                        model_features += [f['name'] for f in interaction_terms 
                                         if f['name'] in training_subset.columns]
                    
                    lasso_model.fit(training_subset[model_features], training_subset[self.target_col])
                    predicted_outcome = round(lasso_model.predict(inference_df[model_features])[0], 2)
                    
                    real_ate = round(trt_outcome - outcome_control, 2)
                    pred_ate = round(trt_outcome - predicted_outcome, 2)
                    
                    results.append({
                        "rct_name": rct_name,
                        "intervention": row["intervention"],
                        "arm": row["Arm"],
                        "real_ate": real_ate,
                        "pred_ate": pred_ate,
                        "outcome_control": outcome_control,
                        "predicted_outcome": predicted_outcome,
                        "intervention_outcome": trt_outcome,
                        "prediction_method": "lasso_regression"
                    })
                    
                except Exception:
                    continue
        
        return results
    
    def run_full_pipeline(self, excel_path: str, output_path: str, 
                         include_comparisons: bool = False) -> Dict[str, Any]:
        """
        Run the complete modeling pipeline.
        
        Args:
            excel_path: Path to the Excel file containing trial data
            output_path: Path where to save the JSON results
            include_comparisons: Whether to include results from comparative models
            
        Returns:
            Dictionary containing all results and performance metrics
        """
        logger.info("Starting full RCT modeling pipeline")
        
        # Load and preprocess data
        training_df = self.load_and_preprocess_data(excel_path)
        
        # Run primary model (CatBoost Quantile)
        primary_results = self.run_catboost_quantile_model(training_df)
        primary_metrics = self.calculate_performance_metrics(primary_results)
        
        final_results = {
            "metadata": {
                "pipeline_version": "1.0.0",
                "model_type": "catboost_quantile_regression",
                "excel_source": excel_path,
                "n_trials_processed": len(primary_results),
                "feature_config": {
                    "target_variable": self.target_col,
                    "n_features": len(self.get_feature_list()),
                    "feature_groups": list(self.feature_config['features'].keys())
                }
            },
            "primary_model_results": primary_results,
            "primary_model_performance": primary_metrics,
        }
        
        # Run comparative models if requested
        if include_comparisons:
            logger.info("Running comparative models")
            comparative_results = self.run_comparative_models(training_df)
            comparative_metrics = {}
            
            for model_name, results in comparative_results.items():
                if results:  # Only calculate metrics if we have results
                    comparative_metrics[model_name] = self.calculate_performance_metrics(results)
            
            final_results["comparative_models"] = {
                "results": comparative_results,
                "performance_metrics": comparative_metrics
            }
        
        # Save results to JSON
        try:
            with open(output_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
        
        logger.info("Pipeline completed successfully")
        return final_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='RCT Lean Modeling Pipeline')
    parser.add_argument('--excel_path', required=True, help='Path to Excel file with trial data')
    parser.add_argument('--output_path', required=True, help='Path for JSON output file')
    parser.add_argument('--config_path', help='Path to feature configuration JSON file')
    parser.add_argument('--include_comparisons', action='store_true', 
                       help='Include comparative model results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run pipeline
    pipeline = RCTModelPipeline(config_path=args.config_path)
    results = pipeline.run_full_pipeline(
        excel_path=args.excel_path,
        output_path=args.output_path,
        include_comparisons=args.include_comparisons
    )
    
    # Print summary
    primary_perf = results['primary_model_performance']['model_performance']
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Trials processed: {results['metadata']['n_trials_processed']}")
    print(f"Primary model R² (ATE): {primary_perf['r2_ate']}")
    print(f"Directional accuracy: {primary_perf['directional_accuracy']}")
    print(f"Spearman correlation: {primary_perf['spearman_correlation']}")
    print(f"Results saved to: {args.output_path}")
    print("="*60)


if __name__ == "__main__":
    main()