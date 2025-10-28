# RCT Pipeline Data Validation

## Overview
The RCT Pipeline now includes comprehensive data validation designed for frontend deployment with clear, user-friendly error messages.

## Validation Checks

### 1. Configuration Validation
- **Required parameters**: Ensures all required config keys are present
- **Feature list validation**: Checks feature list is non-empty and properly formatted
- **Parameter types**: Validates string and list parameters are correct types
- **JSON format**: Clear error messages for malformed JSON

### 2. Data File Validation
- **File existence**: Checks if data file exists at specified path
- **File format**: Supports CSV, Excel (.xlsx, .xls) with proper error messages
- **File corruption**: Handles corrupted or unreadable files
- **Empty files**: Detects and reports empty datasets
- **Minimum data**: Ensures sufficient rows for modeling (minimum 3)

### 3. Data Structure Validation  
- **Required columns**: Validates presence of RCT ID, arm type, and target columns
- **Feature columns**: Checks all specified features exist in data
- **Target RCT**: Ensures target RCT ID exists in dataset with available alternatives
- **Arm types**: Validates arm types are 'Intervention' and 'Control' only

### 4. Data Quality Validation
- **Numerical features**: Ensures features are numeric with examples of non-numeric data
- **Missing data**: Flags features with excessive missing values (>80%)
- **Constant features**: Detects features with no variation
- **Target variable**: Validates target is numeric, not excessively missing, and positive

### 5. Training Data Validation
- **Sufficient samples**: Ensures enough control arms for training (minimum 3)
- **Feature-to-sample ratio**: Warns when features exceed training samples
- **Both arm types**: Ensures dataset contains both intervention and control arms

## Error Message Format

All error messages follow a consistent, user-friendly format:

```
RCT PIPELINE VALIDATION ERROR
============================================================

[Category] Error: [Specific issue description]. 
[Available options or suggestions].
[Actionable guidance for resolution].

============================================================
```

## Example Error Messages

### Empty Feature List
```
Configuration Error: Feature list is empty. 
Please specify at least one feature for the model to use.
```

### Invalid Target RCT
```
Target RCT Error: RCT 'INVALID_ID' not found in data. 
Available RCTs: NCT01364012, NCT01469000, NCT02099058, ...
Please specify a valid RCT ID from your dataset.
```

### Missing Features
```
Feature Error: Missing feature columns: invalid_feature_name. 
Available columns: age_median, gender_male_percent, ...
Please update your feature list or add missing columns to your data.
```

### File Not Found
```
Data File Error: File 'nonexistent.csv' not found. 
Please ensure the data file exists and the path is correct.
```

## Usage

The validation runs automatically when initializing the pipeline:

```python
try:
    pipeline = RCTLightGBMPipeline(config_path='config.json')
    results = pipeline.predict_target_rct()
except RCTDataValidationError as e:
    print(f"Validation Error: {e}")
    # Handle error appropriately for frontend
```

## Frontend Integration

The `RCTDataValidationError` exception provides:
- Clear error categories (Configuration, Data File, Feature, etc.)
- Specific problem descriptions
- Available options when applicable
- Actionable resolution guidance

This makes it easy for frontend applications to:
1. Catch validation errors
2. Display user-friendly messages
3. Provide specific guidance for fixes
4. Show available options (e.g., valid RCT IDs, column names)