#!/usr/bin/env python3
"""
Script to train and save the mental health model for use in the web app.
This processes the dataset, trains the best model configuration, and saves it for the Flask app to use.
"""
import os
import pandas as pd
import numpy as np
import pickle
from model import MentalHealthModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """Load and preprocess the mental health survey data."""
    print("Loading dataset...")
    
    # Check if dataset exists
    if os.path.exists('students_mental_health_survey.csv'):
        df = pd.read_csv('students_mental_health_survey.csv')
    else:
        raise FileNotFoundError("Dataset file not found. Please ensure 'students_mental_health_survey.csv' is in the current directory.")
    
    print(f"Dataset loaded with {df.shape[0]} samples and {df.shape[1]} features.")
    
    # Basic preprocessing
    print("Preprocessing data...")
    
    # Handle any missing values (if needed)
    if df.isnull().sum().sum() > 0:
        print(f"Found {df.isnull().sum().sum()} missing values. Filling with appropriate values...")
        # Fill numerical values with mean
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        # Fill categorical values with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Set target variable - checking if it already exists
    target_col = 'mental_health_condition'  # Replace with actual target column name
    
    # Check for existing stress levels or similar target columns
    if 'Stress_Level' in df.columns:
        print("Using existing 'Stress_Level' as target variable")
        # Binarize the stress level (0-2: low, 3-5: high)
        if df['Stress_Level'].max() > 1:  # If not already binary
            df[target_col] = (df['Stress_Level'] >= 3).astype(int)
        else:
            df[target_col] = df['Stress_Level']
    else:
        # For demo purposes, if target column doesn't exist, create a binary target
        print("Note: Creating a simulated target variable for demonstration purposes")
        # Get numeric columns only for creating synthetic target
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        
        # If we have at least a few numeric columns
        if numeric_df.shape[1] >= 2:
            # Use first few numeric columns to create a target
            feature_cols = numeric_df.columns[:min(4, numeric_df.shape[1])]
            df[target_col] = (numeric_df[feature_cols].mean(axis=1) > 
                             numeric_df[feature_cols].mean(axis=1).median()).astype(int)
        else:
            # If not enough numeric columns, just use a random binary variable
            print("Not enough numeric columns, creating random target")
            np.random.seed(42)
            df[target_col] = np.random.randint(0, 2, size=df.shape[0])
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical features
    cat_cols = X.select_dtypes(include=['object']).columns
    print(f"Encoding {len(cat_cols)} categorical features...")
    for col in cat_cols:
        # Create dummy variables
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    
    return X, y

def train_model(X, y):
    """Train the model with best practices from the experiments."""
    print("Training the mental health model...")
    
    # Create an 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize the model
    model = MentalHealthModel()
    
    # Train the model with best practices
    # Using PCA since it performed best in experiments
    results = model.train(X_train, y_train, X_test, y_test, use_pca=True)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(results['y_test'], results['y_pred']):.4f}")
    print("\nClassification Report:")
    print(classification_report(results['y_test'], results['y_pred']))
    
    return model

def save_model(model, output_dir='models'):
    """Save the trained model for use in the web app."""
    print(f"Saving model to {output_dir}/mental_health_model.pkl...")
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    with open(os.path.join(output_dir, 'mental_health_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully!")

def main():
    """Main function to run the training process."""
    print("Starting the model training process...")
    
    # Load and preprocess data
    X, y = load_data()
    
    # Train the model
    model = train_model(X, y)
    
    # Save the model
    save_model(model)
    
    print("\nTraining and saving process complete! The model is ready for use in the web app.")
    print("Run 'python app.py' to start the web application.")

if __name__ == "__main__":
    main() 