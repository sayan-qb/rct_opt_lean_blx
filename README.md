# RCT Lean Pipeline - LightGBM Implementation

Approach for predicting treatment effects in randomized controlled trials (RCTs) using LightGBM for a selected trial

## Overview

The pipeline predicts treatment effects for clinical trials by:
1. Training on control arms from similar RCTs
2. Predicting what the control arm outcome would be for the target RCT  
3. Calculating Average Treatment Effect (ATE) using actual intervention outcome

## Project Structure

```
├── data_alignment.ipynb    # Data preprocessing notebook (notebook to preprocess and align the raw data and the creating the config directly from it for now)
├── rct_pipeline.py          # Main LightGBM prediction pipeline
├── config.json              # Pipeline configuration
├── data/                    # Processed data directory
├── results/                 # Output results directory
└── 20250521_Trials for dev.xlsx  # Raw data file
```

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```
```bash
source .venv/bin/activate
```

### 2. Prepare Data
- Loads raw Excel data
- Create required columns (`rct_id`, `Arm`, `is_target_trial`)
- Generate processed CSV file
- Create pipeline configuration

### 3. Run Prediction Pipeline
```bash
python rct_pipeline.py --config path_to_config.json --output path_to_result.json
```

## Configuration

Edit `config.json` to specify:

```json
{
  "ftr_list": ["age_median", "gender_male_percent", ...],
  "target_variable": "PFS_median_months",
  "rct_id_col_name": "rct_id", 
  "arm_type_col_name": "Arm",
  "is_target_rct": "is_target_trial",
  "target_rct_id": "NCT02578680",
  "data_path": "data/trial_data.csv"
}
```

## Output Format

The pipeline outputs JSON with the requested format:

```json
{
  "predicted_ate": 3.06,
  "predicted_outcome_control_arm": 5.74,
  "metadata": {
    "model_type": "lightgbm",
    "n_training_samples": 31,
    "cv_rmse": 1.371
  },
  "prediction_details": {
    "target_rct_id": "NCT02578680",
    "intervention_actual": 8.8,
    "control_predicted": 5.74,
    "ate_calculation": "8.8 - 5.74 = 3.06"
  }
}
```

## Data Requirements

Your Excel file should contain:
- **NCT_ID**: RCT identifier (renamed to `rct_id`)
- **Arm**: "Control" or "Intervention" 
- **PFS_median_months**: Target outcome variable
- **Feature columns**: As specified in `ftr_list` config
- **Target RCT**: Specified by `target_rct_id` in config (e.g., "NCT02578680")

## Example Output

```
============================================================
RCT PREDICTION COMPLETED - LIGHTGBM
============================================================
Model Performance (CV RMSE): 1.371 ± 0.556
Training Samples: 31
Features Used: 11

Prediction Results:
  Predicted Control Arm Outcome: 5.74 months
  Predicted ATE: 3.06 months
  Actual Intervention Outcome: 8.8 months
  ATE Calculation: 8.8 - 5.74 = 3.06
============================================================
```

## Implementation Changes from Research Notebooks

This pipeline adapts the methodology from `02_modeling.ipynb` with key modifications:

- **Validation Approach**: Changed from Leave-One-Out cross-validation to single target trial prediction
- **Feature Engineering**: Simplified approach using features directly from config instead of automatic interaction terms
- **Intervention Conditioning**: Direct use of target trial's intervention outcome rather than feature-based conditioning
- **Algorithm Focus**: LightGBM only (notebooks tested multiple approaches)
- **Output Format**: Structured JSON with `predicted_ate` and `predicted_outcome_control_arm` keys

**Rationale**: Since we'll be aligning with LLM for feature engineering, we kept the feature selection simple and config-driven for now.

## Algorithm Details

- **Model**: LightGBM with 5-fold cross-validation
- **Training Data**: Control arms from similar RCTs
- **Features**: Config-specified clinical and treatment features (no automatic interaction terms)
- **Prediction**: Target RCT control arm outcome
- **ATE Calculation**: Intervention_actual - Control_predicted

## Files Created by Pipeline

1. **data/trial_data.csv**: Processed trial data ready for modeling
2. **config_generated.json**: Auto-generated configuration file  
3. **results/*.json**: Prediction results with ATE and control arm predictions

---

*Prepared for BlackRock Demo - October 2025*