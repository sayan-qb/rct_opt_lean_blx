#!/usr/bin/env python3
"""
RCT Model Pipeline - LightGBM Implementation

This module implements a streamlined RCT prediction pipeline:
- Takes config file (JSON) and Excel file as inputs
- Uses LightGBM as the primary algorithm
- Predicts target RCT outcome based on similar RCTs

Config structure:
- ftr_list: list of features to use
- target_variable: outcome variable name
- rct_id_col_name: column name for RCT IDs
- arm_type_col_name: column with "Intervention" or "Control" values
- target_rct_id: specific RCT ID to use as target
- data_path: path to the data file

Usage:
    python rct_model_pipeline.py --config config.json --output results.json
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
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class RCTLightGBMPipeline:
    """
    RCT Model Pipeline using LightGBM for target vs similar trials prediction.
    
    This class implements:
    1. Load target RCT and similar RCTs from Excel/CSV
    2. Use LightGBM for prediction
    3. Return predictions with confidence intervals
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the RCT modeling pipeline.
        
        Args:
            config_path: Path to configuration JSON file (required)
        """
        self.config = self._load_config(config_path)
        self.features = self.config['ftr_list']
        self.target_variable = self.config['target_variable']
        self.rct_id_col = self.config['rct_id_col_name']
        self.arm_type_col = self.config['arm_type_col_name']
        self.target_rct_id = self.config['target_rct_id']
        self.data_path = self.config['data_path']
        
        logger.info(f"RCT LightGBM Pipeline initialized")
        logger.info(f"Features: {len(self.features)}")
        logger.info(f"Target variable: {self.target_variable}")
        
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
                
                # Validate required config keys
                required_keys = [
                    'ftr_list', 'target_variable', 'rct_id_col_name', 
                    'arm_type_col_name', 'target_rct_id', 'data_path'
                ]
                for key in required_keys:
                    if key not in config:
                        raise ValueError(f"Missing required config key: {key}")
                
                return config
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config file: {e}")
    
    def predict_target_rct(self) -> Dict[str, Any]:
        """
        Predict target RCT outcome using similar RCTs.
        
        Returns:
            Dictionary with prediction results
        """
        logger.info("Starting target RCT prediction")
        
        # Load and preprocess data
        df = self._load_data(self.data_path)
        target_rct_data, similar_rcts_data = self._preprocess_data(df)
        
        # Make prediction
        result = self._predict_with_lightgbm(target_rct_data, similar_rcts_data)
        
        logger.info("Target RCT prediction completed successfully")
        return result
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        try:
            # Determine file type and load accordingly
            if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path)
            else:
                df = pd.read_csv(data_path)
                
            logger.info(f"Loaded dataset shape: {df.shape}")
            
            # Validate required columns exist
            required_cols = [
                self.rct_id_col,
                self.arm_type_col,
                self.target_variable
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in data: {missing_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data file: {e}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> tuple:
        """
        Preprocess the data for modeling.
        
        Returns:
            target_rct_data: Data for the target RCT
            similar_rcts_data: Data for similar RCTs (training data)
        """
        logger.info("Preprocessing data...")
        
        # Get target RCT by ID
        target_rct_data = df[df[self.rct_id_col] == self.target_rct_id].copy()
        
        if target_rct_data.empty:
            raise ValueError(f"Target RCT '{self.target_rct_id}' not found in data")
        
        logger.info(f"Target RCT ID: {self.target_rct_id}")
        
        # Get similar RCTs (all RCTs except target)
        similar_rcts_data = df[df[self.rct_id_col] != self.target_rct_id].copy()
        
        if similar_rcts_data.empty:
            raise ValueError("No similar RCTs found for training")
        
        logger.info(f"Target RCT data shape: {target_rct_data.shape}")
        logger.info(f"Similar RCTs data shape: {similar_rcts_data.shape}")
        
        # Check arm types
        logger.info(f"Target RCT arm types: {target_rct_data[self.arm_type_col].value_counts().to_dict()}")
        logger.info(f"Similar RCTs arm types: {similar_rcts_data[self.arm_type_col].value_counts().to_dict()}")
        
        return target_rct_data, similar_rcts_data
    
    def _add_intervention_outcome_conditioning(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add intervention outcome conditioning features based on the methodology from the modeling notebook.
        
        For each trial with both intervention and control arms:
        - When predicting control outcome, use the intervention outcome as a feature
        - This helps the model understand the trial-specific context
        
        Returns:
            Enhanced dataset with intervention outcome conditioning features
        """
        logger.info("Adding intervention outcome conditioning features...")
        
        enhanced_data = data.copy()
        
        # Find trials that have both intervention and control arms
        trial_arm_counts = data.groupby(self.rct_id_col)[self.arm_type_col].nunique()
        paired_trials = trial_arm_counts[trial_arm_counts == 2].index
        
        logger.info(f"Found {len(paired_trials)} trials with both intervention and control arms")
        
        # For each trial with paired arms, add intervention outcome as feature for control arm
        enhanced_data['intervention_outcome_feature'] = np.nan
        
        for trial_id in paired_trials:
            trial_data = data[data[self.rct_id_col] == trial_id]
            
            # Get intervention and control outcomes for this trial
            intervention_row = trial_data[trial_data[self.arm_type_col] == 'Intervention']
            control_row = trial_data[trial_data[self.arm_type_col] == 'Control']
            
            if not intervention_row.empty and not control_row.empty:
                intervention_outcome = intervention_row[self.target_variable].iloc[0]
                
                # Add intervention outcome as feature for control arm
                control_idx = enhanced_data[
                    (enhanced_data[self.rct_id_col] == trial_id) & 
                    (enhanced_data[self.arm_type_col] == 'Control')
                ].index
                
                enhanced_data.loc[control_idx, 'intervention_outcome_feature'] = intervention_outcome
                
                logger.debug(f"Trial {trial_id}: Added intervention outcome {intervention_outcome:.2f} as feature for control arm")
        
        # For unpaired trials (single-arm), we can't use this feature - fill with 0 or mean
        unpaired_mask = enhanced_data['intervention_outcome_feature'].isna()
        n_unpaired = unpaired_mask.sum()
        
        if n_unpaired > 0:
            logger.info(f"Found {n_unpaired} unpaired arms - filling with mean intervention outcome")
            # Use mean intervention outcome from all intervention arms
            mean_intervention = data[data[self.arm_type_col] == 'Intervention'][self.target_variable].mean()
            enhanced_data.loc[unpaired_mask, 'intervention_outcome_feature'] = mean_intervention
        
        # Add interaction terms with drug classes (if available)
        if 'EGFR_TKI' in enhanced_data.columns:
            enhanced_data['int_outcome_x_egfr'] = enhanced_data['intervention_outcome_feature'] * enhanced_data['EGFR_TKI']
            logger.info("Created interaction: intervention_outcome × EGFR_TKI")
        
        if 'PD1_PDL1_Inhibitor' in enhanced_data.columns:
            enhanced_data['int_outcome_x_immuno'] = enhanced_data['intervention_outcome_feature'] * enhanced_data['PD1_PDL1_Inhibitor']
            logger.info("Created interaction: intervention_outcome × PD1_PDL1_Inhibitor")
        
        logger.info(f"Enhanced dataset shape: {enhanced_data.shape}")
        logger.info(f"Intervention outcome conditioning added for control arms")
        
        return enhanced_data
    
    def _predict_with_lightgbm(self, target_rct_data: pd.DataFrame, similar_rcts_data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction using LightGBM with intervention outcome conditioning."""
        logger.info("Training LightGBM model with intervention outcome conditioning...")
        
        # Combine all data and add intervention outcome conditioning
        all_data = pd.concat([target_rct_data, similar_rcts_data], ignore_index=True)
        enhanced_data = self._add_intervention_outcome_conditioning(all_data)
        
        # Filter to control arms only for training
        control_arms = enhanced_data[enhanced_data[self.arm_type_col] == 'Control'].copy()
        
        if control_arms.empty:
            raise ValueError("No control arms found for training")
        
        # Get available features (including intervention outcome and interaction terms)
        base_features = [f for f in self.features if f in control_arms.columns]
        intervention_features = ['intervention_outcome_feature']
        interaction_features = []
        
        # Add interaction features if they exist
        if 'int_outcome_x_egfr' in control_arms.columns:
            interaction_features.append('int_outcome_x_egfr')
        if 'int_outcome_x_immuno' in control_arms.columns:
            interaction_features.append('int_outcome_x_immuno')
        
        available_features = base_features + intervention_features + interaction_features
        available_features = [f for f in available_features if f in control_arms.columns]
        
        missing_features = [f for f in self.features if f not in control_arms.columns]
        if missing_features:
            logger.warning(f"Missing base features in data: {missing_features}")
        
        logger.info(f"Using {len(available_features)} features for training")
        logger.info(f"Intervention conditioning features: {[f for f in available_features if 'intervention' in f.lower() or 'int_outcome' in f]}")
        
        # Prepare training data (exclude target RCT from training)
        training_control_arms = control_arms[control_arms[self.rct_id_col] != self.target_rct_id].copy()
        
        if training_control_arms.empty:
            raise ValueError("No control arms from similar RCTs available for training")
        
        X_train = training_control_arms[available_features].fillna(0)
        y_train = training_control_arms[self.target_variable]
        
        # Remove rows with missing target values
        valid_mask = ~y_train.isna()
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        if len(X_train) < 3:
            raise ValueError(f"Insufficient training data: {len(X_train)} samples")
        
        logger.info(f"Training data shape: {X_train.shape}")
        
        # Train LightGBM model with parameters suitable for small datasets
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': min(31, max(6, len(X_train) // 3)),
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': max(1, len(X_train) // 10),
            'verbose': -1,
            'random_state': 42
        }
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Train model
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Cross-validation for model validation
        cv_scores = cross_val_score(
            lgb.LGBMRegressor(**lgb_params, n_estimators=100),
            X_train, y_train, cv=min(5, len(X_train)), 
            scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(-cv_scores).std()
        
        logger.info(f"Cross-validation RMSE: {cv_rmse:.3f} ± {cv_std:.3f}")
        
        # Make predictions for target RCT
        target_predictions = self._predict_target_arms(model, enhanced_data, available_features)
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        
        # Create result with required output format
        result = {
            "predicted_ate": target_predictions.get("predicted_ate"),
            "predicted_outcome_control_arm": target_predictions.get("predicted_outcome_control_arm"),
            "metadata": {
                "pipeline_version": "4.0.0",
                "model_type": "lightgbm_with_intervention_conditioning",
                "prediction_date": datetime.now().isoformat(),
                "n_training_samples": len(X_train),
                "n_features": len(available_features),
                "cv_rmse": round(cv_rmse, 3),
                "cv_std": round(cv_std, 3),
                "train_rmse": round(train_rmse, 3),
                "train_mae": round(train_mae, 3)
            },
            "prediction_details": target_predictions.get("prediction_details"),
            "model_performance": {
                "cross_validation_rmse": round(cv_rmse, 3),
                "cross_validation_std": round(cv_std, 3),
                "training_rmse": round(train_rmse, 3),
                "training_mae": round(train_mae, 3)
            },
            "features_used": available_features,
            "config_used": self.config
        }
        
        logger.info("Prediction completed successfully")
        return result
    
    def _predict_target_arms(self, model, enhanced_data: pd.DataFrame, available_features: List[str]) -> Dict[str, Any]:
        """Make predictions for target RCT using intervention outcome conditioning."""
        
        # Get target RCT data
        target_rct = enhanced_data[enhanced_data[self.rct_id_col] == self.target_rct_id]
        
        if target_rct.empty:
            raise ValueError(f"Target RCT {self.target_rct_id} not found in enhanced dataset")
        
        # Check if target RCT has both intervention and control arms
        target_arms = target_rct[self.arm_type_col].unique()
        logger.info(f"Target RCT arms: {target_arms}")
        
        # Get intervention and control arms
        intervention_arm = target_rct[target_rct[self.arm_type_col] == 'Intervention']
        control_arm = target_rct[target_rct[self.arm_type_col] == 'Control']
        
        if not intervention_arm.empty and not control_arm.empty:
            # Both arms available - predict control using intervention outcome
            actual_intervention_outcome = intervention_arm[self.target_variable].iloc[0]
            actual_control_outcome = control_arm[self.target_variable].iloc[0]
            
            # Use control arm features with intervention outcome conditioning for prediction
            control_features = control_arm[available_features].fillna(0)
            predicted_control_outcome = model.predict(control_features)[0]
            
            # Calculate ATE using actual intervention outcome
            predicted_ate = actual_intervention_outcome - predicted_control_outcome
            
            logger.info(f"Target RCT has both arms - using intervention conditioning")
            logger.info(f"Actual control: {actual_control_outcome:.2f}, Predicted control: {predicted_control_outcome:.2f}")
            
            prediction_details = {
                "target_rct_id": self.target_rct_id,
                "intervention_actual": round(actual_intervention_outcome, 2),
                "control_actual": round(actual_control_outcome, 2),
                "control_predicted": round(predicted_control_outcome, 2),
                "intervention_outcome_conditioning": round(control_arm['intervention_outcome_feature'].iloc[0], 2),
                "ate_actual": round(actual_intervention_outcome - actual_control_outcome, 2),
                "ate_predicted": round(predicted_ate, 2),
                "prediction_method": "intervention_outcome_conditioning",
                "data_structure": "paired_arms"
            }
            
        elif not intervention_arm.empty:
            # Only intervention arm - predict what control would have been
            actual_intervention_outcome = intervention_arm[self.target_variable].iloc[0]
            
            # Create virtual control features using intervention arm features but with intervention outcome
            virtual_control_features = intervention_arm[available_features].copy()
            virtual_control_features['intervention_outcome_feature'] = actual_intervention_outcome
            virtual_control_features = virtual_control_features.fillna(0)
            
            predicted_control_outcome = model.predict(virtual_control_features)[0]
            predicted_ate = actual_intervention_outcome - predicted_control_outcome
            
            logger.info(f"Target RCT intervention-only - predicting counterfactual control")
            
            prediction_details = {
                "target_rct_id": self.target_rct_id,
                "intervention_actual": round(actual_intervention_outcome, 2),
                "control_predicted": round(predicted_control_outcome, 2),
                "intervention_outcome_conditioning": round(actual_intervention_outcome, 2),
                "ate_predicted": round(predicted_ate, 2),
                "prediction_method": "counterfactual_control_prediction",
                "data_structure": "intervention_only"
            }
            
        else:
            raise ValueError(f"Target RCT {self.target_rct_id} has no intervention arm")
        
        # Create the required output format
        predictions = {
            "predicted_outcome_control_arm": round(predicted_control_outcome, 2),
            "predicted_ate": round(predicted_ate, 2),
            "prediction_details": prediction_details
        }
        
        return predictions


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='RCT Model Pipeline - LightGBM Implementation')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    parser.add_argument('--output', help='Path for output results file')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RCTLightGBMPipeline(config_path=args.config)
    
    # Make prediction
    results = pipeline.predict_target_rct()
    
    # Save to specified output file if provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")
    
    # Print summary
    metadata = results['metadata']
    
    print("\n" + "="*60)
    print("RCT PREDICTION COMPLETED - LIGHTGBM")
    print("="*60)
    print(f"Model Performance (CV RMSE): {metadata['cv_rmse']} ± {metadata['cv_std']}")
    print(f"Training Samples: {metadata['n_training_samples']}")
    print(f"Features Used: {metadata['n_features']}")
    print("\nPrediction Results:")
    print(f"  Predicted Control Arm Outcome: {results['predicted_outcome_control_arm']} months")
    print(f"  Predicted ATE: {results['predicted_ate']} months")
    
    if 'prediction_details' in results and results['prediction_details']:
        details = results['prediction_details']
        if details.get('intervention_actual') is not None:
            print(f"  Actual Intervention Outcome: {details['intervention_actual']} months")
        if details.get('ate_calculation'):
            print(f"  ATE Calculation: {details['ate_calculation']}")
        elif details.get('ate_predicted') is not None:
            print(f"  ATE (Predicted): {details['ate_predicted']} months")
    
    print("="*60)


if __name__ == "__main__":
    main()