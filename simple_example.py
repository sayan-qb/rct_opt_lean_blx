#!/usr/bin/env python3
"""
Simple example showing how to use the RCT Model Pipeline
with the config-driven approach for target vs similar trials prediction.
"""

import json
import numpy as np
import pandas as pd
from rct_model_pipeline import RCTModelPipeline


def create_sample_data():
    """Create sample trial data for demonstration."""
    np.random.seed(42)
    
    # Sample trial data
    trials = ['NCT02578680', 'NCT02142738', 'NCT02041533', 'NCT02657434', 'NCT03950674']
    
    data = []
    for trial_id in trials:
        # Control arm
        control_row = {
            'rct_id': trial_id,
            'is_arm_control': 1,
            'Arm': 'Control',
            'PFS_median_months': np.random.normal(8, 2),
            'gender_male_percent': np.random.uniform(50, 80),
            'age_median': np.random.uniform(60, 70),
            'no_smoker_percent': np.random.uniform(10, 30),
            'ecog_1': np.random.uniform(60, 80),
            'brain_metastase_yes': np.random.uniform(5, 15),
            'disease_stage_IV': 100,
            'EGFR_wild': np.random.uniform(70, 90),
            'EGFR_positive_mutation': np.random.uniform(10, 30),
            'EGFR_TKI': np.random.choice([0, 1]),
            'PD1_PDL1_Inhibitor': np.random.choice([0, 1]),
            'Chemotherapy': 1,
            'Targeted_Therapy': np.random.choice([0, 1]),
            'Population': np.random.randint(100, 500)
        }
        data.append(control_row)
        
        # Intervention arm
        intervention_row = control_row.copy()
        intervention_row.update({
            'is_arm_control': 0,
            'Arm': 'Intervention',
            'PFS_median_months': control_row['PFS_median_months'] + np.random.normal(2, 1)
        })
        data.append(intervention_row)
    
    df = pd.DataFrame(data)
    
    # Set target trial outcomes (NCT02578680)
    df.loc[(df['rct_id'] == 'NCT02578680') & (df['Arm'] == 'Control'), 'PFS_median_months'] = 11.2
    df.loc[(df['rct_id'] == 'NCT02578680') & (df['Arm'] == 'Intervention'), 'PFS_median_months'] = 13.5
    
    # Fill missing columns with defaults
    for col in ['smoker_percent', 'disease_stage_recurrent', 'disease_stage_III', 
                'combo_therapy', 'treatment_complexity', 'RCT_with_control_inter',
                'First-in-Class', 'Next-Generation']:
        df[col] = 0
    
    df.to_csv('sample_trials.csv', index=False)
    print("Sample data created: sample_trials.csv")
    return df


def run_simple_example():
    """Run the simple example."""
    print("RCT Model Pipeline - Simple Example")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize pipeline with config
    print("Initializing pipeline with config: config.json")
    pipeline = RCTModelPipeline('config.json')
    
    # Update config to point to sample data
    import json
    with open('config.json', 'r') as f:
        config = json.load(f)
    config['data_config']['data_path'] = 'sample_trials.csv'
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Reinitialize pipeline with updated config
    pipeline = RCTModelPipeline('config.json')
    
    # Run prediction
    print("Running prediction on data: sample_trials.csv")
    results = pipeline.predict_target_trial()
    
    # Display results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    
    pred_result = results['prediction_result']
    metadata = results['metadata']
    
    print(f"Target Trial: {pred_result['target_trial_id']}")
    print(f"Model Type: {metadata['model_type']}")
    print(f"Training Trials: {pred_result['n_training_trials']}")
    print(f"Features Used: {len(pred_result['features_used'])}")
    
    print(f"\nPREDICTED OUTCOMES:")
    print(f"  Control Outcome: {pred_result['predicted_control_outcome']} months")
    print(f"  Intervention Outcome: {pred_result['intervention_outcome']} months")
    print(f"  Predicted ATE: {pred_result['predicted_ate']} months")
    
    print(f"\nACTUAL OUTCOMES:")
    print(f"  Actual Control: {pred_result['actual_control_outcome']} months")
    print(f"  Actual ATE: {pred_result['actual_ate']} months")
    print(f"  ATE Prediction Error: {abs(pred_result['predicted_ate'] - pred_result['actual_ate']):.2f} months")
    
    print(f"\nUNCERTAINTY QUANTIFICATION:")
    print(f"  Control 95% CI: {pred_result['pred_ci_95']} months")
    print(f"  ATE 95% CI: {pred_result['ate_ci_95']} months")
    print(f"  P(ATE > 0): {pred_result['prob_positive_ate']}")
    
    print(f"\nCLINICAL INTERPRETATION:")
    prob_positive = pred_result['prob_positive_ate']
    ate_ci_width = pred_result['ate_ci_95'][1] - pred_result['ate_ci_95'][0]
    
    if prob_positive > 0.8:
        confidence = "🟢 HIGH CONFIDENCE: Strong evidence for positive treatment effect"
    elif prob_positive > 0.6:
        confidence = "🟡 MODERATE CONFIDENCE: Some evidence for positive treatment effect"
    else:
        confidence = "🔴 LOW CONFIDENCE: Uncertain treatment effect"
    
    if ate_ci_width < 3:
        precision = "🎯 HIGH PRECISION: Narrow confidence interval"
    elif ate_ci_width < 6:
        precision = "📊 MODERATE PRECISION: Reasonable confidence interval"
    else:
        precision = "📈 LOW PRECISION: Wide confidence interval"
    
    print(f"  {confidence}")
    print(f"  {precision}")
    
    print(f"\nResults saved to: sample_trials_results.json")
    print("=" * 50)
    
    print(f"\n✅ Example completed successfully!")
    print(f"💡 To change target trial, edit 'target_trial.rct_id' in config.json")
    print(f"📊 Check sample_trials_results.json for detailed results")


if __name__ == "__main__":
    run_simple_example()