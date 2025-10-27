# RCT Lean Modeling Pipeline

A production-ready machine learning pipeline for predicting control arm outcomes in randomized controlled trials (RCTs) and calculating Average Treatment Effects (ATE) with uncertainty quantification.

## Overview

This module implements the best-performing approach from the research notebooks: **CatBoost Quantile Regression** with Leave-One-Out validation protocol. The pipeline can:

- Ingest clinical trial data from Excel files
- Preprocess and engineer features automatically
- Train sophisticated ML models with uncertainty quantification
- Predict control arm outcomes for new trials
- Calculate ATE with confidence intervals and probability estimates
- Output comprehensive results in JSON format

## Key Features

### 🎯 Best-in-Class Model Performance
- **CatBoost Quantile Regression**: State-of-the-art gradient boosting with full uncertainty quantification
- **Leave-One-Out Protocol**: Rigorous validation methodology ensuring no data leakage
- **Intervention Conditioning**: Uses intervention arm outcomes to improve control arm predictions

### 📊 Comprehensive Uncertainty Quantification
- **Full Prediction Distributions**: Not just point estimates, but complete probability distributions
- **Confidence Intervals**: 80%, 90%, and 95% confidence intervals for all predictions
- **ATE Probability**: P(ATE > 0) for treatment effect significance assessment
- **Coverage Analysis**: Model calibration and reliability metrics

### 🔬 Research-Backed Methodology
- **Comparative Analysis**: Includes 5 different modeling approaches for benchmarking
- **Feature Engineering**: Automated creation of interaction terms and composite features
- **Clinical Validation**: Metrics designed specifically for clinical trial decision-making

### 🚀 Production Ready
- **Command Line Interface**: Easy integration into existing workflows
- **Comprehensive Logging**: Detailed execution tracking and error handling
- **JSON Output**: Structured results for downstream analysis and reporting

## Installation

1. Clone or download this repository
2. Install required packages:

```bash
pip install -r pipeline_requirements.txt
```

Required packages:
- numpy>=1.21.0
- pandas>=1.3.0  
- scikit-learn>=1.0.0
- scipy>=1.7.0
- catboost>=1.0.0
- lightgbm>=3.0.0
- openpyxl>=3.0.0

## Quick Start

### Command Line Usage

```bash
# Basic usage with CatBoost model only
python rct_model_pipeline.py --excel_path "path/to/trials.xlsx" --output_path "results.json"

# Include comparative models for benchmarking
python rct_model_pipeline.py --excel_path "path/to/trials.xlsx" --output_path "results.json" --include_comparisons

# Use custom feature configuration
python rct_model_pipeline.py --excel_path "path/to/trials.xlsx" --output_path "results.json" --config_path "custom_config.json"
```

### Python API Usage

```python
from rct_model_pipeline import RCTModelPipeline

# Initialize pipeline
pipeline = RCTModelPipeline()

# Run complete analysis
results = pipeline.run_full_pipeline(
    excel_path="trials.xlsx",
    output_path="results.json",
    include_comparisons=True
)

# Access results
performance = results['primary_model_performance']
predictions = results['primary_model_results']
```

### Example Script

See `example_usage.py` for detailed examples including:
- Simple pipeline execution
- Results analysis and visualization
- Trial-specific predictions
- Performance comparison across models

## Input Data Format

The pipeline expects an Excel file with two sheets:

### Sheet 1: "trials by arm"
Contains trial arm-level data with columns including:
- `NCT_ID`: Trial identifier
- `Arm`: Treatment arm designation
- `Population`: Number of patients
- `intervention`: Treatment description
- `PFS_median_months`: Progression-free survival (target variable)
- Patient demographics: `gender_male_percent`, `age_median`, etc.
- Disease characteristics: `brain_metastase_yes`, `disease_stage_IV`, etc.
- Drug classifications: `EGFR_TKI`, `PD1_PDL1_Inhibitor`, etc.

### Sheet 2: "250529_NSCLC" 
Additional trial data in similar format with a `to keep` column for filtering.

## Output Format

The pipeline generates a comprehensive JSON file containing:

### Metadata
```json
{
  "metadata": {
    "pipeline_version": "1.0.0",
    "model_type": "catboost_quantile_regression", 
    "n_trials_processed": 45,
    "feature_config": {
      "target_variable": "PFS_median_months",
      "n_features": 31,
      "feature_groups": ["patient_demographics", "disease_characteristics", ...]
    }
  }
}
```

### Primary Model Results
For each trial, detailed predictions including:
```json
{
  "rct_name": "NCT12345678",
  "intervention": "Pembrolizumab + Chemotherapy",
  "real_ate": 2.3,
  "pred_ate": 2.1,
  "outcome_control": 11.2,
  "predicted_outcome": 11.4,
  "intervention_outcome": 13.5,
  "ate_ci_95": [0.8, 3.4],
  "prob_positive_ate": 0.892,
  "pred_ci_95": [9.8, 13.1],
  "quantile_predictions": {
    "0.025": 9.2,
    "0.5": 11.4,
    "0.975": 13.6
  },
  "prediction_method": "catboost_quantile"
}
```

### Performance Metrics
```json
{
  "model_performance": {
    "r2_ate": 0.73,
    "directional_accuracy": 0.84,
    "spearman_correlation": 0.81,
    "rmse_ate": 1.23,
    "ate_coverage_95": 0.91,
    "avg_prob_positive_ate": 0.76
  },
  "summary_statistics": {
    "n_trials": 45,
    "avg_real_ate": 1.85,
    "avg_pred_ate": 1.79,
    "positive_ate_trials": 38
  }
}
```

## Model Approaches

### Primary Model: CatBoost Quantile Regression
- **Method**: Gradient boosting with quantile loss functions
- **Uncertainty**: Full distributional predictions with empirical quantiles
- **Features**: Intervention conditioning + interaction terms
- **Validation**: Leave-one-out cross-validation

### Comparative Models (Optional)
1. **Simple Average**: Baseline using mean of other control arms
2. **Ratio Method**: Fixed proportion of intervention outcome (70% default)
3. **Random Forest**: Ensemble method with intervention conditioning  
4. **Lasso Regression**: L1-regularized linear model with interactions

## Key Performance Metrics

### Clinical Relevance
- **Directional Accuracy**: % of trials where ATE sign is predicted correctly (most important)
- **R² for ATE**: Goodness of fit for treatment effect predictions
- **Spearman Correlation**: Rank-order correlation (robust to outliers)

### Prediction Quality
- **RMSE**: Root mean squared error for absolute accuracy
- **Coverage**: % of true values falling within predicted confidence intervals
- **Calibration**: How well predicted probabilities match actual outcomes

### Uncertainty Quantification
- **CI Width**: Average width of confidence intervals (precision)
- **P(ATE > 0)**: Probability of positive treatment effect
- **IoU**: Intersection over Union with actual confidence intervals (when available)

## Feature Engineering

The pipeline automatically creates engineered features:

### Treatment Features
- `combo_therapy`: Multi-drug treatment indicator
- `treatment_complexity`: Number of different drug classes
- `novelty_score`: First-in-class + next-generation indicator

### Patient Risk Features  
- `high_risk_profile`: Brain metastases or Stage IV disease
- `elderly_male`: Age > 60 and male > 60%
- `smoker_percent`: Derived from non-smoker percentage

### Interaction Terms
- `int_outcome_x_egfr`: Intervention outcome × EGFR TKI treatment
- `int_outcome_x_immuno`: Intervention outcome × Immunotherapy

## Configuration

Customize the pipeline behavior using a JSON configuration file:

```json
{
  "target": "PFS_median_months",
  "feature_groups_to_use": ["patient_demographics", "disease_characteristics", "drug_class"],
  "features": {
    "patient_demographics": ["gender_male_percent", "age_median", "no_smoker_percent"],
    "disease_characteristics": ["brain_metastase_yes", "disease_stage_IV"],
    "drug_class": ["EGFR_TKI", "PD1_PDL1_Inhibitor", "Chemotherapy"]
  },
  "interaction_terms": [
    {"name": "int_outcome_x_egfr", "feature1": "intervention_outcome", "feature2": "EGFR_TKI"}
  ]
}
```

## Research Background

This pipeline implements the methodology developed in the research notebooks, specifically:

### Leave-One-Out Protocol
1. **Target Selection**: Choose one RCT to predict (hold-out)
2. **Training Set**: Use control arms from all other RCTs
3. **Prediction**: Predict held-out trial's control arm outcome
4. **ATE Calculation**: Compare predicted control vs actual intervention outcome
5. **Evaluation**: Compare predicted ATE with actual ATE

### Intervention Conditioning
A key innovation is using the intervention arm outcome as a conditional feature:
- Adds intervention outcome as a feature for all control arms
- Creates interaction terms between intervention outcome and drug classes
- Enables the model to adjust predictions based on intervention effectiveness

### Uncertainty Quantification
Rather than point estimates, the model provides:
- Full probability distributions via quantile regression
- Confidence intervals at multiple levels (80%, 90%, 95%)
- Probability of positive treatment effect
- Coverage analysis for model calibration

## Clinical Applications

### Trial Design
- **Go/No-Go Decisions**: Use directional accuracy for treatment promising assessment
- **Sample Size Planning**: Use control arm predictions for power calculations
- **Endpoint Selection**: Leverage uncertainty estimates for realistic planning

### Portfolio Management
- **Trial Prioritization**: Rank trials by probability of positive ATE
- **Resource Allocation**: Focus on trials with highest expected treatment effects
- **Risk Assessment**: Use confidence intervals for investment decisions

### Regulatory Strategy
- **Benchmark Setting**: Compare new trials against predicted control performance
- **Success Probability**: Quantify likelihood of meeting primary endpoints
- **Evidence Planning**: Use uncertainty estimates for evidence generation strategies

## Limitations and Considerations

### Data Requirements
- Requires historical RCT data with consistent endpoints
- Performance depends on similarity between historical and target trials
- Limited to progression-free survival as primary endpoint

### Model Assumptions
- Assumes control arm outcomes are predictable from trial characteristics
- Leave-one-out validation may be optimistic for very different trial populations
- Feature engineering is tailored to lung cancer trials

### Clinical Context
- Predictions are estimates and should inform, not replace, clinical judgment
- External validity depends on representativeness of training data
- Regulatory acceptance may vary by indication and geography

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all packages are installed with correct versions
```bash
pip install -r pipeline_requirements.txt
```

**Excel File Format**: Check that sheets "trials by arm" and "250529_NSCLC" exist
- Verify column names match expected format
- Ensure `NCT_ID`, `Arm`, `PFS_median_months` columns are present

**Missing Data**: Pipeline handles missing values but requires core columns
- `PFS_median_months`: Target variable (required)
- `NCT_ID`: Trial identifier (required) 
- `Arm`: Treatment arm designation (required)

**Memory Issues**: For large datasets, consider:
- Running without comparative models (`include_comparisons=False`)
- Reducing the number of quantiles in CatBoost configuration
- Processing subsets of trials separately

### Performance Optimization

**Speed Up Training**:
- Set `include_comparisons=False` to run only CatBoost model
- Reduce CatBoost iterations in configuration
- Use fewer quantiles for uncertainty quantification

**Improve Accuracy**:
- Ensure high-quality feature engineering
- Validate that intervention outcomes are available for all trials
- Check for data quality issues in key features

## Support and Development

This pipeline was developed based on the research methodology in the accompanying Jupyter notebooks. For questions about:

- **Methodology**: Refer to `01_preprocessing.ipynb` and `02_modeling.ipynb`
- **Implementation**: Check `example_usage.py` for detailed examples
- **Configuration**: See `notebooks/config/feature_config.json` for feature definitions
- **Utilities**: Review `notebooks/utilities/` for underlying functions

## License

This software framework contains confidential and proprietary information of QuantumBlack (a McKinsey company). Use is governed by the agreement between your organization and QuantumBlack.

---

*Last updated: October 2025*
