"""
Example usage script for the RCT Model Pipeline

This script demonstrates how to use the RCTModelPipeline class to:
1. Ingest Excel data containing clinical trial information
2. Run the best-performing CatBoost Quantile Regression model
3. Output comprehensive results in JSON format

The pipeline implements the methodology from the research notebooks,
specifically the Leave-One-Out protocol for predicting control arm
outcomes and calculating Average Treatment Effects (ATE).
"""

import json
import os
from rct_model_pipeline import RCTModelPipeline

def run_pipeline_example():
    """Example of running the full RCT modeling pipeline."""
    
    # Initialize the pipeline
    print("Initializing RCT Model Pipeline...")
    pipeline = RCTModelPipeline()
    
    # Example file paths (adjust these to your actual file locations)
    excel_path = "~/Downloads/20250521_Trials for dev.xlsx"  # Path to your Excel file
    output_path = "rct_model_results.json"  # Where to save results
    
    # Check if Excel file exists
    if not os.path.exists(os.path.expanduser(excel_path)):
        print(f"Excel file not found at {excel_path}")
        print("Please update the excel_path variable with the correct path to your Excel file.")
        return
    
    try:
        # Run the complete pipeline
        print(f"Running pipeline on {excel_path}...")
        results = pipeline.run_full_pipeline(
            excel_path=os.path.expanduser(excel_path),
            output_path=output_path,
            include_comparisons=True  # Set to False for faster execution
        )
        
        # Print summary of results
        print_results_summary(results)
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise

def print_results_summary(results):
    """Print a summary of the pipeline results."""
    
    print("\n" + "="*80)
    print("RCT MODEL PIPELINE RESULTS SUMMARY")
    print("="*80)
    
    # Metadata
    metadata = results['metadata']
    print(f"Model Type: {metadata['model_type']}")
    print(f"Trials Processed: {metadata['n_trials_processed']}")
    print(f"Features Used: {metadata['feature_config']['n_features']}")
    
    # Primary model performance
    primary_perf = results['primary_model_performance']['model_performance']
    print(f"\nPRIMARY MODEL PERFORMANCE (CatBoost Quantile):")
    print(f"  R² for ATE: {primary_perf['r2_ate']}")
    print(f"  Directional Accuracy: {primary_perf['directional_accuracy']} ({primary_perf['directional_accuracy']*100:.1f}%)")
    print(f"  Spearman Correlation: {primary_perf['spearman_correlation']}")
    print(f"  RMSE for ATE: {primary_perf['rmse_ate']}")
    print(f"  RMSE for Control Outcome: {primary_perf['rmse_outcome']}")
    
    if 'ate_coverage_95' in primary_perf:
        print(f"  95% CI Coverage: {primary_perf['ate_coverage_95']} ({primary_perf['ate_coverage_95']*100:.1f}%)")
    
    if 'avg_prob_positive_ate' in primary_perf:
        print(f"  Avg P(ATE > 0): {primary_perf['avg_prob_positive_ate']}")
    
    # Summary statistics
    summary_stats = results['primary_model_performance']['summary_statistics']
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Average Real ATE: {summary_stats['avg_real_ate']:.3f} months")
    print(f"  Average Predicted ATE: {summary_stats['avg_pred_ate']:.3f} months")
    print(f"  Trials with Positive Real ATE: {summary_stats['positive_ate_trials']}/{metadata['n_trials_processed']}")
    print(f"  Trials with Predicted Positive ATE: {summary_stats['predicted_positive_ate_trials']}/{metadata['n_trials_processed']}")
    
    # Comparative models (if available)
    if 'comparative_models' in results:
        print(f"\nCOMPARATIVE MODELS PERFORMANCE:")
        comp_metrics = results['comparative_models']['performance_metrics']
        
        print(f"{'Model':<20} {'R² ATE':<10} {'Dir. Acc.':<10} {'Spearman':<10}")
        print("-" * 50)
        
        for model_name, metrics in comp_metrics.items():
            if 'model_performance' in metrics:
                perf = metrics['model_performance']
                print(f"{model_name:<20} {perf['r2_ate']:<10} {perf['directional_accuracy']:<10} {perf['spearman_correlation']:<10}")
    
    print("\n" + "="*80)
    print("Results saved to: rct_model_results.json")
    print("="*80)

def analyze_specific_trial_results(results, trial_name=None):
    """Analyze results for a specific trial or show top performers."""
    
    primary_results = results['primary_model_results']
    
    if trial_name:
        # Find specific trial
        trial_result = next((r for r in primary_results if r['rct_name'] == trial_name), None)
        if trial_result:
            print(f"\nDETAILED RESULTS FOR TRIAL: {trial_name}")
            print("-" * 50)
            print(f"Intervention: {trial_result['intervention']}")
            print(f"Real Control Outcome: {trial_result['outcome_control']} months")
            print(f"Predicted Control Outcome: {trial_result['predicted_outcome']} months")
            print(f"Intervention Outcome: {trial_result['intervention_outcome']} months")
            print(f"Real ATE: {trial_result['real_ate']} months")
            print(f"Predicted ATE: {trial_result['pred_ate']} months")
            print(f"ATE 95% CI: [{trial_result['ate_ci_95'][0]}, {trial_result['ate_ci_95'][1]}]")
            print(f"P(ATE > 0): {trial_result['prob_positive_ate']}")
        else:
            print(f"Trial {trial_name} not found in results.")
    else:
        # Show trials with highest positive ATE predictions
        sorted_results = sorted(primary_results, key=lambda x: x['prob_positive_ate'], reverse=True)
        
        print(f"\nTOP 5 TRIALS BY PROBABILITY OF POSITIVE ATE:")
        print("-" * 80)
        print(f"{'Trial':<15} {'Real ATE':<10} {'Pred ATE':<10} {'P(ATE>0)':<10} {'Intervention':<30}")
        print("-" * 80)
        
        for result in sorted_results[:5]:
            print(f"{result['rct_name']:<15} {result['real_ate']:<10} {result['pred_ate']:<10} "
                  f"{result['prob_positive_ate']:<10} {result['intervention'][:28]:<30}")

def run_simplified_example():
    """Simplified example focusing on the primary CatBoost model only."""
    
    print("Running simplified RCT modeling example...")
    
    # Initialize pipeline
    pipeline = RCTModelPipeline()
    
    # Example paths (adjust as needed)
    excel_path = "~/Downloads/20250521_Trials for dev.xlsx"
    output_path = "rct_results_simplified.json"
    
    if not os.path.exists(os.path.expanduser(excel_path)):
        print("Excel file not found. Please update the path.")
        return
    
    try:
        # Run only the primary model (faster)
        results = pipeline.run_full_pipeline(
            excel_path=os.path.expanduser(excel_path),
            output_path=output_path,
            include_comparisons=False  # Only run CatBoost model
        )
        
        # Quick summary
        n_trials = results['metadata']['n_trials_processed']
        directional_acc = results['primary_model_performance']['model_performance']['directional_accuracy']
        
        print(f"\nSUCCESS! Processed {n_trials} trials.")
        print(f"Directional accuracy: {directional_acc*100:.1f}%")
        print(f"Results saved to: {output_path}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # You can run either the full example or the simplified version
    
    print("RCT Model Pipeline Example")
    print("Choose an option:")
    print("1. Full pipeline with comparative models (slower)")
    print("2. Simplified pipeline with CatBoost only (faster)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        run_pipeline_example()
    elif choice == "2":
        results = run_simplified_example()
        if results:
            # Analyze top trials
            analyze_specific_trial_results(results)
    else:
        print("Invalid choice. Running simplified example...")
        run_simplified_example()