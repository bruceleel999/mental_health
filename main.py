import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from model import MentalHealthModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import warnings
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    # Create output directory for visualizations
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Initialize preprocessor and model
    preprocessor = DataPreprocessor()
    model = MentalHealthModel()
    
    # Load and preprocess data
    print("Loading data...")
    df = preprocessor.load_data('students_mental_health_survey.csv')
    
    print("\nData Shape:", df.shape)
    print("\nSample of the data:")
    print(df.head())
    
    # Add more robust data cleaning and preprocessing
    # Check for missing values
    missing_vals = df.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_vals[missing_vals > 0])
    
    # Clean the data
    print("\nCleaning data...")
    df_cleaned = preprocessor.clean_data(df)
    
    # Check data distribution
    print("\nNumeric columns summary statistics:")
    print(df_cleaned.describe())
    
    # Identify categorical columns
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
    print("\nCategorical columns:", categorical_columns.tolist())
    
    # Encode categorical features
    print("\nEncoding categorical features...")
    df_encoded = preprocessor.encode_categorical_features(df_cleaned, categorical_columns)
    
    # Visualize the data
    print("\nCreating visualizations...")
    fig = preprocessor.visualize_data(df_encoded, 'Stress_Level')
    plt.savefig('output/data_visualization.png')
    plt.close()
    
    # Check for class imbalance
    target_column = 'Stress_Level'
    
    # Apply binary transformation to Stress_Level: 0-2 = low (0), 3-5 = high (1)
    print("\nBinarizing Stress Level scores...")
    df_encoded['Stress_Level'] = df_encoded['Stress_Level'].replace({0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1})
    
    class_counts = df_encoded[target_column].value_counts()
    print("\nClass distribution after binarization:")
    print(class_counts)
    
    # Step 1: Split data into training/holdout sets (80/20)
    print("\nStep 1: Splitting dataset into training and holdout sets (80/20)...")
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    
    X_base, X_holdout, y_base, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Base dataset: {X_base.shape[0]} samples")
    print(f"Holdout dataset: {X_holdout.shape[0]} samples")
    
    # Step 2: Check class distribution in base dataset
    print("\nStep 2: Analyzing class distribution in base dataset...")
    print(f"Class distribution in base dataset: {Counter(y_base)}")
    
    imbalance_ratio = Counter(y_base)[0] / Counter(y_base)[1] if Counter(y_base)[1] > 0 else float('inf')
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Step 3: Apply balancing techniques to create three balanced datasets
    print("\nStep 3: Creating three balanced datasets...")
    
    # Naive random undersampling for majority class
    print("Creating naively balanced dataset...")
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_naive, y_naive = rus.fit_resample(X_base, y_base)
    print(f"Naive balanced dataset: {X_naive.shape[0]} samples, distribution: {Counter(y_naive)}")
    
    # SMOTE-NC for mixed categorical and numerical data
    print("Creating SMOTE-NC balanced dataset...")
    # Find categorical columns
    cat_indices = [i for i, col in enumerate(X_base.columns) if X_base[col].nunique() < 10]
    smote_nc = SMOTE(random_state=42, k_neighbors=5)
    X_smote, y_smote = smote_nc.fit_resample(X_base, y_base)
    print(f"SMOTE-NC balanced dataset: {X_smote.shape[0]} samples, distribution: {Counter(y_smote)}")
    
    # Hybrid approach (SMOTE + Tomek Links)
    print("Creating hybrid balanced dataset...")
    smote_tomek = SMOTETomek(random_state=42)
    X_hybrid, y_hybrid = smote_tomek.fit_resample(X_base, y_base)
    print(f"Hybrid balanced dataset: {X_hybrid.shape[0]} samples, distribution: {Counter(y_hybrid)}")
    
    # Step 4: Train and evaluate models using each balanced dataset
    print("\nStep 4: Training models on each balanced dataset...")
    
    # Train on naive balanced dataset
    print("\nTraining on naive balanced dataset...")
    naive_results = model.train_dataset(X_naive, y_naive, X_holdout, y_holdout, 'naive')
    
    # Train on SMOTE-NC balanced dataset
    print("\nTraining on SMOTE-NC balanced dataset...")
    smote_results = model.train_dataset(X_smote, y_smote, X_holdout, y_holdout, 'smote-nc')
    
    # Train on hybrid balanced dataset
    print("\nTraining on hybrid balanced dataset...")
    hybrid_results = model.train_dataset(X_hybrid, y_hybrid, X_holdout, y_holdout, 'hybrid')
    
    # Compare results
    print("\nStep 5: Comparing model performance across different preprocessing techniques...")
    model.compare_results([naive_results, smote_results, hybrid_results])
    
    # Train final model using best preprocessing
    best_X, best_y = X_hybrid, y_hybrid  # Default to hybrid, will be updated below
    
    # Determine best preprocessing based on results
    if naive_results['accuracy'] > smote_results['accuracy'] and naive_results['accuracy'] > hybrid_results['accuracy']:
        best_X, best_y = X_naive, y_naive
        best_method = "naive"
    elif smote_results['accuracy'] > naive_results['accuracy'] and smote_results['accuracy'] > hybrid_results['accuracy']:
        best_X, best_y = X_smote, y_smote
        best_method = "SMOTE-NC"
    else:
        best_method = "hybrid"
    
    print(f"\nBest preprocessing method: {best_method}")
    
    # Feature selection on best dataset
    print("\nPerforming feature selection on best dataset...")
    X_selected, selected_features = model.select_features(best_X, best_y)
    print("Selected features:", selected_features.tolist())
    
    # Perform hyperparameter tuning
    print("\nPerforming hyperparameter tuning on best dataset...")
    optimized_model = model.hyperparameter_tuning(X_selected, best_y)
    
    # Train the optimized model
    print("\nTraining the optimized model...")
    results = model.train(X_selected, best_y, X_holdout[selected_features], y_holdout)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    model.evaluate(results)
    
    # Show feature importance
    print("\nAnalyzing feature importance...")
    model.feature_importance(selected_features)
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    model.cross_validate(X_selected, best_y)
    
    print("\nComplete! Results saved to output directory.")

if __name__ == "__main__":
    main() 