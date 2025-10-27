#!/usr/bin/env python3
"""
RCT Model Pipeline - Target vs Similar Trials Prediction

This module implements a simplified approach for predicting treatment effects:
- Takes one target trial and similar trials from a CSV file
- Uses CatBoost Quantile Regression for uncertainty quantification
- Returns ATE predictions with confidence intervals

Architecture:
- User selects target trial via config
- GenAI provides similar trials data
- ML pipeline predicts ATE with uncertainty

Usage:
    python rct_model_pipeline.py data.csv --config config.json
"""

import argparse
import json
import logging
import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class RCTModelPipeline:
    """
    Simplified RCT Model Pipeline for Target vs Similar Trials Prediction.
    
    This class implements a streamlined approach:
    1. Load target trial and similar trials from CSV
    2. Use CatBoost for uncertainty quantification
    3. Return ATE predictions with confidence intervals
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the RCT modeling pipeline.
        
        Args:
            config_path: Path to configuration JSON file (required)
        """
        self.config = self._load_config(config_path)
        self.target_trial_id = self.config['target_trial']['rct_id']
        self.data_config = self.config['data_config']
        self.features = self.config['features'] 
        self.interaction_terms = self.config.get('interaction_terms', [])
        self.model_config = self.config['model_config']
        
        logger.info(f"RCT Model Pipeline initialized for target trial: {self.target_trial_id}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not config_path:
            raise ValueError("Config file path is required. Please provide a config.json file.")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                
                # Validate required config sections
                required_sections = ['target_trial', 'data_config', 'features', 'model_config']
                for section in required_sections:
                    if section not in config:
                        raise ValueError(f"Missing required config section: {section}")
                
                # Validate target_trial section
                if 'rct_id' not in config['target_trial']:
                    raise ValueError("Missing 'rct_id' in target_trial config")
                
                # Validate data_config section
                required_data_config = ['data_path', 'target_column', 'trial_id_column', 'control_arm_column', 'intervention_outcome_column']
                for key in required_data_config:
                    if key not in config['data_config']:
                        raise ValueError(f"Missing '{key}' in data_config")
                
                return config
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config file: {e}")
    
    def predict_target_trial(self) -> Dict[str, Any]:
        """
        Predict target trial outcome using similar trials.
        
        Returns:
            Dictionary with prediction results
        """
        logger.info("Starting target trial prediction")
        
        # Get data path from config
        data_path = self.data_config['data_path']
        
        # Load and preprocess data
        df = self._load_data(data_path)
        processed_df = self._preprocess_data(df)
        
        # Check if target trial exists in original data (may not be in control arms)
        target_exists = self.target_trial_id in df[self.data_config['trial_id_column']].values
        if not target_exists:
            raise ValueError(f"Target trial {self.target_trial_id} not found in data")
        
        # Get target trial intervention outcome from original data
        target_intervention = df[
            (df[self.data_config['trial_id_column']] == self.target_trial_id) & 
            (df[self.data_config['control_arm_column']] == 0)
        ]
        
        if target_intervention.empty:
            raise ValueError(f"Target trial {self.target_trial_id} intervention arm not found")
        
        target_outcome = target_intervention[self.data_config['target_column']].iloc[0]
        logger.info(f"Target trial intervention outcome: {target_outcome}")
        
        # Use all available control arms as similar trials (excluding NaN trial_ids for training)
        similar_trials = processed_df.dropna(subset=[self.data_config['trial_id_column']])
        
        if len(similar_trials) < 3:
            raise ValueError(f"Insufficient similar trials for training: {len(similar_trials)}")
        
        # Create a synthetic target trial row for prediction using control arm features but target outcome
        target_trial = self._create_target_trial_for_prediction(df, similar_trials)
        
        logger.info(f"Target trial: {self.target_trial_id}")
        logger.info(f"Similar trials: {len(similar_trials)}")
        
        # Make prediction
        result = self._predict_single_trial(target_trial, similar_trials)
        
        # Save results
        output_file = data_path.replace('.csv', '_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
        
        logger.info("Target trial prediction completed successfully")
        return result
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded dataset shape: {df.shape}")
            
            # Validate required columns exist
            required_cols = [
                self.data_config['trial_id_column'],
                self.data_config['control_arm_column'], 
                self.data_config['target_column']
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in data: {missing_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data file: {e}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for modeling."""        
        # Filter to control arms only (for prediction)
        control_df = df[df[self.data_config['control_arm_column']] == 1].copy()
        logger.info(f"Filtered to control arms only: {control_df.shape}")
        
        # Add intervention outcomes for each trial
        for trial_id in control_df[self.data_config['trial_id_column']].unique():
            if pd.isna(trial_id):
                continue
                
            # Get intervention outcome for this trial
            intervention_rows = df[
                (df[self.data_config['trial_id_column']] == trial_id) & 
                (df[self.data_config['control_arm_column']] == 0)
            ]
            
            if not intervention_rows.empty:
                intervention_outcome = intervention_rows[self.data_config['target_column']].mean()
                control_df.loc[
                    control_df[self.data_config['trial_id_column']] == trial_id, 
                    self.data_config['intervention_outcome_column']
                ] = intervention_outcome
        
        # Add engineered features
        control_df = self._add_features(control_df)
        
        # Add interaction terms
        for term in self.interaction_terms:
            if term['feature1'] in control_df.columns and term['feature2'] in control_df.columns:
                control_df[term['name']] = (
                    control_df[term['feature1']] * control_df[term['feature2']]
                )
        
        # Clean data
        control_df = control_df.dropna(subset=[
            self.data_config['target_column'], 
            self.data_config['intervention_outcome_column']
        ])
        
        logger.info(f"Preprocessed dataset shape: {control_df.shape}")
        return control_df
    
    def _create_target_trial_for_prediction(self, df: pd.DataFrame, similar_trials: pd.DataFrame) -> pd.DataFrame:
        """Create a synthetic target trial row for prediction using median values from similar trials."""
        # Get target trial intervention outcome
        target_intervention = df[
            (df[self.data_config['trial_id_column']] == self.target_trial_id) & 
            (df[self.data_config['control_arm_column']] == 0)
        ]
        
        target_outcome = target_intervention[self.data_config['target_column']].iloc[0]
        
        # Create synthetic target trial using median/mode values from similar trials
        synthetic_target = similar_trials.iloc[0:1].copy()  # Use first row as template
        
        # Set the target trial ID and outcome
        synthetic_target[self.data_config['trial_id_column']] = self.target_trial_id
        synthetic_target[self.data_config['intervention_outcome_column']] = target_outcome
        
        # Use median values for numeric features
        numeric_cols = similar_trials.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.data_config['trial_id_column'] and col != self.data_config['intervention_outcome_column']:
                synthetic_target[col] = similar_trials[col].median()
        
        logger.info(f"Created synthetic target trial for prediction with intervention outcome: {target_outcome}")
        return synthetic_target
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features."""
        df = df.copy()
        
        # Treatment combinations
        treatment_cols = ['Chemotherapy', 'Targeted_Therapy', 'Immunotherapy', 'Anti-angiogenic_Other']
        available_treatment_cols = [col for col in treatment_cols if col in df.columns]
        if available_treatment_cols:
            df['combo_therapy'] = (df[available_treatment_cols].sum(axis=1) > 1).astype(int)
        
        # Treatment complexity
        complexity_cols = ['EGFR_TKI', 'Anti_VEGF', 'PD1_PDL1_Inhibitor', 'Antimetabolite', 'Taxane', 'Antibody']
        available_complexity_cols = [col for col in complexity_cols if col in df.columns]
        if available_complexity_cols:
            df['treatment_complexity'] = df[available_complexity_cols].sum(axis=1)
        
        # Calculate percentage of smokers
        if 'no_smoker_percent' in df.columns:
            df['smoker_percent'] = 100 - df['no_smoker_percent']
        
        return df
    
    def _predict_single_trial(self, target_trial: pd.DataFrame, similar_trials: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction for a single target trial."""
        logger.info(f"Predicting target trial using {len(similar_trials)} similar trials")
        
        # Get available features
        available_features = [f for f in self.features if f in similar_trials.columns]
        
        # Add intervention outcome and interaction terms
        model_features = available_features + [self.data_config['intervention_outcome_column']]
        for term in self.interaction_terms:
            if term['name'] in similar_trials.columns:
                model_features.append(term['name'])
        
        # Prepare training data
        X_train = similar_trials[model_features].fillna(0)
        y_train = similar_trials[self.data_config['target_column']]
        
        # Prepare target data
        X_target = target_trial[model_features].fillna(0)
        
        # Train CatBoost quantile models
        quantiles = self.model_config['quantiles']
        quantile_predictions = {}
        
        for q in quantiles:
            model = CatBoostRegressor(
                loss_function=f'Quantile:alpha={q}',
                **self.model_config['catboost_params']
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_target)[0]
            quantile_predictions[q] = pred
        
        # Calculate key statistics
        median_pred = quantile_predictions[0.5]
        ci_95 = [quantile_predictions[0.025], quantile_predictions[0.975]]
        std_pred = (quantile_predictions[0.75] - quantile_predictions[0.25]) / 1.35  # Approximate std
        
        # Get actual values
        actual_control = target_trial[self.data_config['target_column']].iloc[0]
        intervention_outcome = target_trial[self.data_config['intervention_outcome_column']].iloc[0]
        
        # Calculate ATEs
        predicted_ate = intervention_outcome - median_pred
        actual_ate = intervention_outcome - actual_control
        
        # Calculate ATE confidence interval
        ate_ci_95 = [
            intervention_outcome - ci_95[1],  # Lower bound
            intervention_outcome - ci_95[0]   # Upper bound
        ]
        
        # Probability of positive ATE (approximate)
        prob_positive_ate = np.mean([
            intervention_outcome - quantile_predictions[q] > 0 
            for q in quantiles
        ])
        
        # Create result
        result = {
            "metadata": {
                "pipeline_version": "2.0.0",
                "model_type": self.model_config['model_type'],
                "target_trial_id": self.target_trial_id,
                "n_similar_trials": len(similar_trials),
                "n_features": len(available_features),
                "prediction_date": datetime.now().isoformat()
            },
            "prediction_result": {
                "target_trial_id": self.target_trial_id,
                "predicted_control_outcome": round(median_pred, 2),
                "actual_control_outcome": round(actual_control, 2),
                "intervention_outcome": round(intervention_outcome, 2),
                "predicted_ate": round(predicted_ate, 2),
                "actual_ate": round(actual_ate, 2),
                "ate_ci_95": [round(ate_ci_95[0], 2), round(ate_ci_95[1], 2)],
                "prob_positive_ate": round(prob_positive_ate, 2),
                "pred_ci_95": [round(ci_95[0], 2), round(ci_95[1], 2)],
                "pred_std": round(std_pred, 2),
                "n_training_trials": len(similar_trials),
                "features_used": available_features,
                "quantile_predictions": {str(q): round(v, 2) for q, v in quantile_predictions.items()}
            },
            "config_used": self.config
        }
        
        logger.info(f"Prediction completed: ATE = {predicted_ate:.2f}, P(ATE>0) = {prob_positive_ate:.2f}")
        return result


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='RCT Model Pipeline - Target vs Similar Trials')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    parser.add_argument('--output', help='Path for output results file')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RCTModelPipeline(config_path=args.config)
    
    # Make prediction
    results = pipeline.predict_target_trial()
    
    # Save to specified output file if provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")
    
    # Print summary
    pred_result = results['prediction_result']
    print("\n" + "="*60)
    print("PREDICTION COMPLETED")
    print("="*60)
    print(f"Target Trial: {pred_result['target_trial_id']}")
    print(f"Predicted ATE: {pred_result['predicted_ate']} months")
    print(f"ATE 95% CI: {pred_result['ate_ci_95']}")
    print(f"P(ATE > 0): {pred_result['prob_positive_ate']}")
    print("="*60)


if __name__ == "__main__":
    main()