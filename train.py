import pandas as pd
import numpy as np
import os
from model import MentalHealthModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

print("Loading mental health dataset...")
# Replace with your dataset path
data_path = 'students_mental_health_survey.csv'  # Update this with your actual path

# Load the data
data = pd.read_csv(data_path)

# Basic data exploration
print(f"Dataset shape: {data.shape}")
print(f"Column names: {data.columns.tolist()}")

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Drop rows with missing values
data = data.dropna()
print(f"Dataset shape after dropping missing values: {data.shape}")

# Handle categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"Unique values in {col}: {data[col].unique()}")

# Encode categorical variables
for col in categorical_cols:
    # Use label encoding for binary/ordinal variables, one-hot for nominal
    if data[col].nunique() == 2:
        # Binary variables - label encode
        data[col] = data[col].map({data[col].unique()[0]: 0, data[col].unique()[1]: 1})
    elif data[col].nunique() <= 5:
        # Ordinal variables with few levels - label encode
        mapping = {val: i for i, val in enumerate(sorted(data[col].unique()))}
        data[col] = data[col].map(mapping)
    else:
        # Nominal variables with many levels - one-hot encode
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)

# Define target variable (replace 'target_column' with your actual target column)
# Example: 'Stress_Level' or 'Depression_Score' could be your target
target_column = 'Stress_Level'  # Update this with your actual target column

if target_column in data.columns:
    # For binary classification, convert target to binary if needed
    if data[target_column].nunique() > 2:
        print(f"Converting {target_column} to binary target...")
        # Example: values above median are considered high stress (1), below are low stress (0)
        median = data[target_column].median()
        data['binary_target'] = (data[target_column] > median).astype(int)
        y = data['binary_target']
        print(f"Target distribution: {y.value_counts()}")
    else:
        y = data[target_column]
        print(f"Target distribution: {y.value_counts()}")
    
    # Get features (all columns except target)
    X = data.drop([target_column, 'binary_target'] if 'binary_target' in data.columns else [target_column], axis=1)
else:
    print(f"Target column '{target_column}' not found in dataset. Please update the target column.")
    exit(1)

# Print information about features
print(f"\nFeature matrix shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# Initialize the model
model = MentalHealthModel()

# ===== NEW: Perform t-tests on features =====
print("\n===== Statistical Feature Analysis with T-Tests =====")
feature_t_test_results = model.apply_feature_t_tests(X, y)
print(f"Total features: {X.shape[1]}")
print(f"Statistically significant features: {feature_t_test_results['significant'].sum()}")
print(f"Significant after Bonferroni correction: {feature_t_test_results['bonferroni_significant'].sum()}")

# Top 5 most significant features
top_features = feature_t_test_results.sort_values('p_value').head(5)
print("\nTop 5 most significant features:")
for _, row in top_features.iterrows():
    print(f"- {row['feature']}: p-value = {row['p_value']:.6f}, effect size = {row['effect_size']:.4f}")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check for class imbalance
print(f"\nClass distribution in training set: {y_train.value_counts()}")
imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

# ===== NEW: Statistical comparison of balancing methods =====
print("\n===== Statistical Comparison of Balancing Methods (T-Tests) =====")
balancing_t_test_results = model.compare_balancing_methods_t_test(X, y)

# Get the best balancing method from t-test results
significant_improvements = balancing_t_test_results[balancing_t_test_results['significant'] & 
                                                 (balancing_t_test_results['mean_diff'] > 0)]
if len(significant_improvements) > 0:
    # Get the method with highest mean difference
    best_balancing_method = significant_improvements.sort_values('mean_diff', ascending=False)['comparison'].iloc[0].split('_vs_')[1]
    print(f"\nStatistically significant best balancing method: {best_balancing_method}")
else:
    # If no significant improvements, use original data
    best_balancing_method = 'original'
    print("\nNo statistically significant improvements from balancing methods. Using original data.")

# Balance the dataset using three different approaches
print("\nBalancing dataset using different techniques...")

# 1. Random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print(f"Random undersampling - training set shape: {X_train_rus.shape}")
print(f"Class distribution after undersampling: {pd.Series(y_train_rus).value_counts()}")

# 2. SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE - training set shape: {X_train_smote.shape}")
print(f"Class distribution after SMOTE: {pd.Series(y_train_smote).value_counts()}")

# 3. Combined approach (undersampling majority + oversampling minority)
undersampling_ratio = 0.5  # Adjust as needed
oversampler = SMOTE(sampling_strategy=0.7, random_state=42)  # Adjust sampling_strategy as needed
undersampler = RandomUnderSampler(sampling_strategy=undersampling_ratio, random_state=42)

# Apply SMOTE first, then undersample
X_temp, y_temp = oversampler.fit_resample(X_train, y_train)
X_train_combined, y_train_combined = undersampler.fit_resample(X_temp, y_temp)
print(f"Combined approach - training set shape: {X_train_combined.shape}")
print(f"Class distribution after combined approach: {pd.Series(y_train_combined).value_counts()}")

# Compare different balancing methods using train_dataset function
print("\n=== Comparing different balancing methods ===")
results = []

# Train on original (imbalanced) data
results.append(model.train_dataset(X_train, y_train, X_test, y_test, 'original'))

# Train on undersampled data
results.append(model.train_dataset(X_train_rus, y_train_rus, X_test, y_test, 'undersampled'))

# Train on SMOTE-balanced data
results.append(model.train_dataset(X_train_smote, y_train_smote, X_test, y_test, 'smote'))

# Train on combined approach data
results.append(model.train_dataset(X_train_combined, y_train_combined, X_test, y_test, 'combined'))

# Compare results
model.compare_results(results)

# Use the best balancing method (from t-test results) for further training
if best_balancing_method == 'original':
    X_train_balanced, y_train_balanced = X_train, y_train
elif best_balancing_method == 'undersampled':
    X_train_balanced, y_train_balanced = X_train_rus, y_train_rus
elif best_balancing_method == 'smote':
    X_train_balanced, y_train_balanced = X_train_smote, y_train_smote
else:  # combined
    X_train_balanced, y_train_balanced = X_train_combined, y_train_combined

print(f"\nUsing {best_balancing_method} method for further model training")

print("\n=== Feature engineering and dimensionality reduction ===")

# First, try feature selection
X_selected, selected_features = model.select_features(X_train_balanced, y_train_balanced)
print(f"Selected {len(selected_features)} important features")

# Compare dimensionality reduction techniques (PCA, ICA, and no reduction)
print("\nComparing dimensionality reduction techniques...")
reduction_results = model.compare_dimensionality_reduction(X_train_balanced, y_train_balanced)

# Perform statistical significance testing for dimensionality reduction techniques
print("\n=== Statistical significance testing for dimensionality reduction ===")
significance_results = model.compare_dimensionality_reduction_significance(X_train_balanced, y_train_balanced)

# Now train the full model using the best balancing method
print("\n=== Training final model with the best preprocessing approach ===")

# Check if feature selection improved accuracy
feature_selection_accuracy = model.train_dataset(
    X_selected, y_train_balanced, 
    X_test[selected_features], y_test, 
    'feature_selection'
)['accuracy']

# Find best dimensionality reduction technique
baseline_acc = reduction_results['baseline']['accuracy']
pca_acc = reduction_results['pca']['accuracy']
ica_acc = reduction_results['ica']['accuracy']

print(f"Accuracy comparison:")
print(f"- No dimensionality reduction: {baseline_acc:.4f}")
print(f"- PCA: {pca_acc:.4f}")
print(f"- ICA: {ica_acc:.4f}")
print(f"- Feature selection: {feature_selection_accuracy:.4f}")

# Create list of different approaches
models_to_compare = [
    ('baseline', baseline_acc),
    ('pca', pca_acc),
    ('ica', ica_acc),
    ('feature_selection', feature_selection_accuracy)
]

# Sort by accuracy
models_to_compare.sort(key=lambda x: x[1], reverse=True)
best_approach, best_acc = models_to_compare[0]
second_best_approach, second_best_acc = models_to_compare[1]

# Compare the best and second-best approaches with t-test
print(f"\n=== Statistical comparison of {best_approach} vs {second_best_approach} ===")
if best_approach == 'baseline':
    model1 = RandomForestClassifier(n_estimators=200, random_state=42)
    X1 = X_train_balanced
elif best_approach == 'pca':
    model1 = RandomForestClassifier(n_estimators=200, random_state=42)
    X1 = reduction_results['pca']['X_test'] # Using test set from reduction_results
elif best_approach == 'ica':
    model1 = RandomForestClassifier(n_estimators=200, random_state=42)
    X1 = reduction_results['ica']['X_test']
else:  # feature_selection
    model1 = RandomForestClassifier(n_estimators=200, random_state=42)
    X1 = X_selected

if second_best_approach == 'baseline':
    model2 = RandomForestClassifier(n_estimators=200, random_state=42)
    X2 = X_train_balanced
elif second_best_approach == 'pca':
    model2 = RandomForestClassifier(n_estimators=200, random_state=42)
    X2 = reduction_results['pca']['X_test']
elif second_best_approach == 'ica':
    model2 = RandomForestClassifier(n_estimators=200, random_state=42)
    X2 = reduction_results['ica']['X_test']
else:  # feature_selection
    model2 = RandomForestClassifier(n_estimators=200, random_state=42)
    X2 = X_selected

# Perform t-test comparison
model.compare_models_t_test(model1, model2, X_train_balanced, y_train_balanced)

# Set flags based on best approach
use_pca = False
use_ica = False
use_feature_selection = False

if best_approach == 'pca':
    use_pca = True
    print("\nUsing PCA for final model (best approach)")
elif best_approach == 'ica':
    use_ica = True
    print("\nUsing ICA for final model (best approach)")
elif best_approach == 'feature_selection':
    use_feature_selection = True
    print("\nUsing Feature Selection for final model (best approach)")
else:
    print("\nUsing original features (no dimensionality reduction) for final model (best approach)")

# Train final model with hyperparameter tuning
print("\nPerforming hyperparameter tuning...")
if use_feature_selection:
    tuned_model = model.hyperparameter_tuning(X_selected, y_train_balanced)
    X_final_train = X_selected
    X_final_test = X_test[selected_features]
else:
    tuned_model = model.hyperparameter_tuning(X_train_balanced, y_train_balanced)
    X_final_train = X_train_balanced
    X_final_test = X_test

# Final training using the best approach
print("\nTraining final model...")
final_results = model.train(
    X_final_train, y_train_balanced, 
    X_final_test, y_test,
    use_pca=use_pca,
    use_ica=use_ica
)

# Evaluate the final model
print("\nEvaluating final model...")
model.evaluate(final_results)

# ===== NEW: Calculate confidence intervals with bootstrap =====
print("\n===== Bootstrap Confidence Intervals and Statistical Testing =====")
bootstrap_results = model.predict_with_confidence_intervals(X_final_test, y_test, n_bootstrap=1000)

# Display confidence interval information
print(f"Final model accuracy: {bootstrap_results['accuracy']:.4f}")
print(f"95% Confidence Interval: [{bootstrap_results['accuracy_ci_lower']:.4f}, {bootstrap_results['accuracy_ci_upper']:.4f}]")

# Compare with baseline model
if bootstrap_results['p_value_vs_baseline'] < 0.05:
    print("Model performance is statistically significantly better than baseline")
else:
    print("Model performance is not statistically significantly better than baseline")

# Show feature importance if applicable
if not (use_pca or use_ica):
    if use_feature_selection:
        model.feature_importance(selected_features)
    else:
        model.feature_importance(X.columns)

# Cross-validation
print("\nPerforming cross-validation on final model...")
if use_feature_selection:
    model.cross_validate(X_selected, y_train_balanced)
else:
    model.cross_validate(X_train_balanced, y_train_balanced)

print("\nModel training and evaluation complete!")
print("Check the 'output' directory for visualizations and results.")

# Final summary
print("\n=== Final Performance Summary ===")
if use_pca:
    dim_reduction = "PCA"
    n_components = reduction_results['pca']['n_components']
elif use_ica:
    dim_reduction = "ICA"
    n_components = reduction_results['ica']['n_components']
elif use_feature_selection:
    dim_reduction = "Feature Selection"
    n_components = len(selected_features)
else:
    dim_reduction = "None"
    n_components = X.shape[1]

print(f"Best balancing method: {best_balancing_method}")
print(f"Best dimensionality reduction: {best_approach}")
print(f"Dimensionality reduction: {dim_reduction} (features: {n_components})")
print(f"Final accuracy: {accuracy_score(final_results['y_test'], final_results['y_pred']):.4f}")
print(f"95% CI: [{bootstrap_results['accuracy_ci_lower']:.4f}, {bootstrap_results['accuracy_ci_upper']:.4f}]")

# Summary of statistical significance findings
print("\n=== Statistical Significance Summary ===")
print("A p-value < 0.05 indicates that the difference between models is statistically significant.")
print("Statistical significance test results:")
for comparison, result in significance_results.items():
    p_value = result['p_value']
    t_stat = result['t_statistic']
    significance = "Significant" if p_value < 0.05 else "Not significant"
    print(f"{comparison}: p={p_value:.4f}, t={t_stat:.4f} ({significance})")

# Print top significant features from t-tests
print("\n=== Top Significant Features from T-Tests ===")
top_n = min(10, len(feature_t_test_results))
print(f"Top {top_n} features with significant differences between classes:")
for i, (_, row) in enumerate(feature_t_test_results.sort_values('p_value').head(top_n).iterrows()):
    print(f"{i+1}. {row['feature']}: p-value = {row['p_value']:.6f}, effect size = {row['effect_size']:.4f}")

print("\n===== Analysis Complete =====")
print("The model has been evaluated with extensive statistical testing to ensure reliable results.") 