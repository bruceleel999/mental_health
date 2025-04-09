import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.robust_scaler = RobustScaler()
        
    def load_data(self, file_path):
        """Load the dataset from CSV file."""
        df = pd.read_csv(file_path)
        # Convert column names to be more Python-friendly
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        return df
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and outliers."""
        # Create a copy to avoid modifying the original
        df_cleaned = df.copy()
        
        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates()
        
        # Handle missing values for numerical columns more intelligently
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        
        # For numerical columns with less than 10% missing values, use KNN imputation
        for col in numerical_cols:
            missing_pct = df_cleaned[col].isnull().mean()
            if missing_pct > 0 and missing_pct < 0.1:
                print(f"Using KNN imputation for {col} with {missing_pct:.2%} missing values")
                # Create a subset of the DataFrame for imputation
                imputer = KNNImputer(n_neighbors=5)
                df_cleaned[col] = imputer.fit_transform(df_cleaned[[col]])
            elif missing_pct >= 0.1:
                print(f"Using median imputation for {col} with {missing_pct:.2%} missing values")
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                
        # For categorical columns, use mode imputation
        for col in categorical_cols:
            missing_pct = df_cleaned[col].isnull().mean()
            if missing_pct > 0:
                print(f"Using mode imputation for {col} with {missing_pct:.2%} missing values")
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        
        # Handle outliers in numerical columns
        for col in numerical_cols:
            # Calculate IQR
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"Found {outliers} outliers in {col}")
                
                # Cap outliers instead of removing them
                df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
        
        return df_cleaned
    
    def encode_categorical_features(self, df, categorical_columns):
        """Encode categorical features using Label Encoding."""
        df_encoded = df.copy()
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                
            # Check if there's only one category
            if df_encoded[column].nunique() == 1:
                print(f"Warning: Column {column} has only one unique value. This might not be useful for modeling.")
                
            # Fit and transform
            df_encoded[column] = self.label_encoders[column].fit_transform(df_encoded[column])
        return df_encoded
    
    def create_interaction_features(self, df, numerical_cols):
        """Create interaction features between numerical columns."""
        df_with_interactions = df.copy()
        
        # Create interactions between pairs of numerical columns
        for i, col1 in enumerate(numerical_cols[:-1]):
            for col2 in numerical_cols[i+1:]:
                # Multiply features
                interaction_name = f"{col1}_x_{col2}"
                df_with_interactions[interaction_name] = df[col1] * df[col2]
                
                # Ratio features (avoiding division by zero)
                if (df[col2] != 0).all():
                    ratio_name = f"{col1}_div_{col2}"
                    df_with_interactions[ratio_name] = df[col1] / df[col2]
                    
        return df_with_interactions
    
    def visualize_data(self, df, target_column):
        """Create visualizations for data analysis."""
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Distribution of target variable
        ax = axes[0, 0]
        target_counts = df[target_column].value_counts().sort_index()
        bars = ax.bar(target_counts.index.astype(str), target_counts.values)
        ax.set_title('Distribution of Stress Levels')
        ax.set_xlabel('Stress Level')
        ax.set_ylabel('Count')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        # Correlation matrix
        corr_matrix = df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   vmin=-1, vmax=1, square=True, ax=axes[0, 1])
        axes[0, 1].set_title('Correlation Matrix')
        
        # Box plots for numerical features vs. target
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_column]
        
        # Select top correlated features for visualization
        target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)
        top_features = target_corr[1:5].index  # Skip the target itself
        
        # Plot boxplots for top correlated features
        for i, feature in enumerate(top_features[:4]):
            row, col = (i // 2) + 1, i % 2
            sns.boxplot(x=target_column, y=feature, data=df, ax=axes[row, col])
            axes[row, col].set_title(f'{feature} vs {target_column}')
            axes[row, col].set_xlabel('Stress Level')
        
        plt.tight_layout()
        return fig
    
    def apply_pca(self, X, n_components=0.95):
        """Apply PCA for dimensionality reduction."""
        # Scale the data before PCA
        X_scaled = self.robust_scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Print variance explained
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        print(f"PCA reduced dimensions from {X.shape[1]} to {pca.n_components_}")
        print(f"Total variance explained: {cumulative_variance[-1]:.4f}")
        
        # Create a DataFrame with the PCA components
        pca_df = pd.DataFrame(
            X_pca, 
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=X.index
        )
        
        return pca_df, pca
    
    def prepare_features(self, df, target_column):
        """Prepare features for model training."""
        # Identify numerical and categorical columns
        categorical_cols = [col for col in df.columns if col != target_column and df[col].dtype == 'object']
        numerical_cols = [col for col in df.columns if col != target_column and df[col].dtype != 'object']
        
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # Apply logarithmic transformation to skewed numerical features
        for col in numerical_cols:
            # Check if the column has all positive values
            if (df_features[col] > 0).all():
                # Check if the column is skewed
                skewness = df_features[col].skew()
                if abs(skewness) > 1:
                    print(f"Applying log transform to {col} (skewness: {skewness:.2f})")
                    df_features[col] = np.log1p(df_features[col])
        
        # Create interaction features
        if len(numerical_cols) >= 2:
            df_features = self.create_interaction_features(df_features, numerical_cols)
            
        # Separate features and target
        X = df_features.drop(columns=[target_column])
        y = df_features[target_column]
        
        return X, y 