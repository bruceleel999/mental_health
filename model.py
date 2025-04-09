import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

class MentalHealthModel:
    def __init__(self):
        # Define base models for the ensemble
        self.rf = RandomForestClassifier(n_estimators=200, random_state=42)
        self.gb = GradientBoostingClassifier(random_state=42)
        self.svm = SVC(probability=True, random_state=42)
        self.nn = MLPClassifier(random_state=42, max_iter=1000)
        self.xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        
        # Create a voting classifier
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('gb', self.gb),
                ('svm', self.svm),
                ('nn', self.nn),
                ('xgb', self.xgb)
            ],
            voting='soft'
        )
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.pca = None
        self.ica = None
        
    def analyze_feature_correlations(self, X, y):
        """Analyze correlations between features and target"""
        print("Analyzing feature correlations...")
        
        # Create a DataFrame with features and target
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Calculate correlations
        correlations = []
        for col in df.columns[:-1]:  # Exclude target column
            if df[col].dtype in ['int64', 'float64']:
                corr, p_value = stats.pointbiserialr(df['target'], df[col])
                correlations.append((col, abs(corr), p_value))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Print top correlations
        print("\nTop feature correlations with target:")
        for col, corr, p_value in correlations[:10]:
            print(f"Feature {col}: correlation={corr:.4f}, p-value={p_value:.4f}")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('output/correlation_matrix.png')
        plt.close()
        
        return correlations
    
    def train_dataset(self, X_train, y_train, X_test, y_test, method_name):
        """Train and evaluate a model on a specific dataset."""
        print(f"Training model on {method_name} balanced dataset...")
        
        # For comparison, use RandomForest with more estimators
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
        
        # Scale the features - try both scalers
        X_train_scaled = self.min_max_scaler.fit_transform(X_train)
        X_test_scaled = self.min_max_scaler.transform(X_test)
        
        # Train the model
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
        except:
            roc_auc = 0
            pr_auc = 0
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{method_name.upper()} Model Confusion Matrix (Accuracy: {accuracy:.4f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'output/{method_name}_confusion_matrix.png')
        plt.close()
        
        print(f"{method_name} model - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        
        return {
            'method': method_name,
            'model': rf_model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm
        }
    
    def compare_results(self, results_list):
        """Compare results from different preprocessing methods."""
        # Create comparison bar chart
        methods = [r['method'] for r in results_list]
        accuracies = [r['accuracy'] for r in results_list]
        roc_aucs = [r['roc_auc'] for r in results_list]
        pr_aucs = [r['pr_auc'] for r in results_list]
        
        metrics_df = pd.DataFrame({
            'Method': methods * 3,
            'Metric': ['Accuracy'] * len(methods) + ['ROC-AUC'] * len(methods) + ['PR-AUC'] * len(methods),
            'Value': accuracies + roc_aucs + pr_aucs
        })
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Method', y='Value', hue='Metric', data=metrics_df)
        plt.title('Performance Comparison of Different Balancing Methods')
        plt.ylim(0, 1.0)
        plt.savefig('output/balancing_methods_comparison.png')
        plt.close()
        
        # Print comparison table
        print("\nPerformance Comparison Table:")
        comparison_df = pd.DataFrame({
            'Method': methods,
            'Accuracy': accuracies,
            'ROC-AUC': roc_aucs,
            'PR-AUC': pr_aucs
        })
        print(comparison_df)
        
        # Save comparison to CSV
        comparison_df.to_csv('output/balancing_methods_comparison.csv', index=False)
        
    def select_features(self, X, y):
        """Select important features using RandomForest feature importance."""
        print("Performing feature selection...")
        # Train a random forest for feature selection
        selector = RandomForestClassifier(n_estimators=200, random_state=42)
        selector.fit(X, y)
        
        # Select features using the model
        sfm = SelectFromModel(selector, threshold='median')
        sfm.fit(X, y)
        
        # Get selected feature indices
        feature_indices = sfm.get_support()
        selected_features = X.columns[feature_indices]
        print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
        
        return X[selected_features], selected_features
        
    def hyperparameter_tuning(self, X, y):
        """Tune hyperparameters for each model in the ensemble."""
        X_scaled = self.min_max_scaler.fit_transform(X)
        
        # Define hyperparameter grids for each model
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        gb_param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        
        svm_param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        
        nn_param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }
        
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        
        # Dictionary to store best models
        best_models = {}
        
        # Tune RF
        print("Tuning Random Forest...")
        rf_grid = GridSearchCV(self.rf, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_scaled, y)
        best_models['rf'] = rf_grid.best_estimator_
        print(f"Best RF params: {rf_grid.best_params_}, Score: {rf_grid.best_score_:.4f}")
        
        # Tune Gradient Boosting
        print("Tuning Gradient Boosting...")
        gb_grid = GridSearchCV(self.gb, gb_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        gb_grid.fit(X_scaled, y)
        best_models['gb'] = gb_grid.best_estimator_
        print(f"Best GB params: {gb_grid.best_params_}, Score: {gb_grid.best_score_:.4f}")
        
        # Tune SVM
        print("Tuning SVM...")
        svm_grid = GridSearchCV(self.svm, svm_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        svm_grid.fit(X_scaled, y)
        best_models['svm'] = svm_grid.best_estimator_
        print(f"Best SVM params: {svm_grid.best_params_}, Score: {svm_grid.best_score_:.4f}")
        
        # Tune Neural Network
        print("Tuning Neural Network...")
        nn_grid = GridSearchCV(self.nn, nn_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        nn_grid.fit(X_scaled, y)
        best_models['nn'] = nn_grid.best_estimator_
        print(f"Best NN params: {nn_grid.best_params_}, Score: {nn_grid.best_score_:.4f}")
        
        # Tune XGBoost
        print("Tuning XGBoost...")
        xgb_grid = GridSearchCV(self.xgb, xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        xgb_grid.fit(X_scaled, y)
        best_models['xgb'] = xgb_grid.best_estimator_
        print(f"Best XGB params: {xgb_grid.best_params_}, Score: {xgb_grid.best_score_:.4f}")
        
        # Create optimized ensemble
        self.model = VotingClassifier(
            estimators=[
                ('rf', best_models['rf']),
                ('gb', best_models['gb']),
                ('svm', best_models['svm']),
                ('nn', best_models['nn']),
                ('xgb', best_models['xgb'])
            ],
            voting='soft'
        )
        
        # Find best individual model
        best_score = 0
        for name, model in best_models.items():
            if name == 'rf':
                score = rf_grid.best_score_
            elif name == 'gb':
                score = gb_grid.best_score_
            elif name == 'svm':
                score = svm_grid.best_score_
            elif name == 'nn':
                score = nn_grid.best_score_
            else:  # xgb
                score = xgb_grid.best_score_
                
            if score > best_score:
                best_score = score
                self.best_model = model
        
        print(f"Best individual model score: {best_score:.4f}")
        return self.model
        
    def apply_pca(self, X, n_components=None, variance_threshold=0.95):
        """Apply Principal Component Analysis to reduce dimensionality.
        
        Args:
            X: Feature matrix
            n_components: Number of components to keep (if None, use variance_threshold)
            variance_threshold: Keep enough components to explain this much variance
            
        Returns:
            Transformed feature matrix, PCA model
        """
        print("Applying PCA dimensionality reduction...")
        
        # Scale the data first
        X_scaled = self.min_max_scaler.fit_transform(X)
        
        # Determine number of components if not provided
        if n_components is None:
            # Start with min(n_samples, n_features)
            n_components = min(X_scaled.shape[0], X_scaled.shape[1])
            
        # Initialize and fit PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # If using variance threshold, determine how many components to keep
        if n_components is None:
            explained_variance_ratio = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            n_components_to_keep = np.argmax(cumulative_variance >= variance_threshold) + 1
            print(f"Keeping {n_components_to_keep} components to explain {variance_threshold*100:.1f}% of variance")
            
            # Re-initialize PCA with the determined number of components
            self.pca = PCA(n_components=n_components_to_keep)
            X_pca = self.pca.fit_transform(X_scaled)
        
        # Plot explained variance
        self._plot_explained_variance(self.pca, 'PCA')
        
        print(f"Reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")
        return X_pca
    
    def apply_ica(self, X, n_components=None):
        """Apply Independent Component Analysis for dimensionality reduction.
        
        Args:
            X: Feature matrix
            n_components: Number of components (default: min(n_samples, n_features))
            
        Returns:
            Transformed feature matrix, ICA model
        """
        print("Applying ICA dimensionality reduction...")
        
        # Scale the data first
        X_scaled = self.min_max_scaler.fit_transform(X)
        
        # Determine number of components if not provided
        if n_components is None:
            n_components = min(X_scaled.shape[0], X_scaled.shape[1])
            n_components = min(n_components, 10)  # ICA works better with fewer components
        
        # Initialize and fit ICA
        self.ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        X_ica = self.ica.fit_transform(X_scaled)
        
        # Plot kurtosis of components
        self._plot_ica_kurtosis(X_ica)
        
        print(f"Reduced dimensions from {X.shape[1]} to {X_ica.shape[1]}")
        return X_ica
    
    def _plot_explained_variance(self, pca_model, method_name='PCA'):
        """Plot explained variance from a PCA model."""
        plt.figure(figsize=(10, 6))
        
        # Plot individual explained variance
        plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), 
                pca_model.explained_variance_ratio_, alpha=0.5, label='Individual explained variance')
        
        # Plot cumulative explained variance
        plt.step(range(1, len(pca_model.explained_variance_ratio_) + 1), 
                 np.cumsum(pca_model.explained_variance_ratio_), where='mid', label='Cumulative explained variance')
        
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title(f'{method_name} Explained Variance')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'output/{method_name.lower()}_explained_variance.png')
        plt.close()
    
    def _plot_ica_kurtosis(self, X_ica):
        """Plot kurtosis of ICA components."""
        # Calculate kurtosis for each component
        kurtosis = []
        for i in range(X_ica.shape[1]):
            k = stats.kurtosis(X_ica[:, i])
            kurtosis.append(k)
        
        # Plot kurtosis values
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(kurtosis) + 1), kurtosis)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Independent Component')
        plt.ylabel('Kurtosis')
        plt.title('Kurtosis of Independent Components')
        plt.tight_layout()
        plt.savefig('output/ica_kurtosis.png')
        plt.close()
    
    def compare_dimensionality_reduction(self, X, y, test_size=0.2):
        """Compare different dimensionality reduction techniques.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Test set proportion
            
        Returns:
            Dictionary of results
        """
        print("Comparing dimensionality reduction techniques...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create results dictionary
        results = {}
        
        # Baseline (no dimensionality reduction)
        print("\nTraining baseline model (no dimensionality reduction)...")
        base_model = RandomForestClassifier(n_estimators=200, random_state=42)
        X_train_scaled = self.min_max_scaler.fit_transform(X_train)
        X_test_scaled = self.min_max_scaler.transform(X_test)
        
        base_model.fit(X_train_scaled, y_train)
        y_pred_base = base_model.predict(X_test_scaled)
        accuracy_base = accuracy_score(y_test, y_pred_base)
        print(f"Baseline accuracy: {accuracy_base:.4f}")
        
        results['baseline'] = {
            'accuracy': accuracy_base,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred_base
        }
        
        # PCA
        print("\nTraining model with PCA...")
        # Apply PCA to training data
        pca = PCA(n_components=0.95)  # Preserve 95% of variance
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        pca_model = RandomForestClassifier(n_estimators=200, random_state=42)
        pca_model.fit(X_train_pca, y_train)
        y_pred_pca = pca_model.predict(X_test_pca)
        accuracy_pca = accuracy_score(y_test, y_pred_pca)
        print(f"PCA accuracy: {accuracy_pca:.4f}")
        
        results['pca'] = {
            'accuracy': accuracy_pca,
            'X_test': X_test_pca,
            'y_test': y_test,
            'y_pred': y_pred_pca,
            'n_components': X_train_pca.shape[1]
        }
        
        # ICA
        print("\nTraining model with ICA...")
        # Apply ICA to training data
        n_components_ica = min(10, X_train_scaled.shape[1])  # ICA works better with fewer components
        ica = FastICA(n_components=n_components_ica, random_state=42, max_iter=1000)
        X_train_ica = ica.fit_transform(X_train_scaled)
        X_test_ica = ica.transform(X_test_scaled)
        
        ica_model = RandomForestClassifier(n_estimators=200, random_state=42)
        ica_model.fit(X_train_ica, y_train)
        y_pred_ica = ica_model.predict(X_test_ica)
        accuracy_ica = accuracy_score(y_test, y_pred_ica)
        print(f"ICA accuracy: {accuracy_ica:.4f}")
        
        results['ica'] = {
            'accuracy': accuracy_ica,
            'X_test': X_test_ica,
            'y_test': y_test,
            'y_pred': y_pred_ica,
            'n_components': n_components_ica
        }
        
        # Plot comparison
        methods = ['Baseline', 'PCA', 'ICA']
        accuracies = [accuracy_base, accuracy_pca, accuracy_ica]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, accuracies, color=['blue', 'green', 'orange'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.0)
        plt.ylabel('Accuracy')
        plt.title('Comparison of Dimensionality Reduction Techniques')
        plt.tight_layout()
        plt.savefig('output/dimensionality_reduction_comparison.png')
        plt.close()
        
        # Find best method
        best_method = methods[np.argmax(accuracies)]
        print(f"\nBest dimensionality reduction method: {best_method}")
        
        return results
        
    def train(self, X, y, X_test=None, y_test=None, use_pca=False, use_ica=False, compare_reduction=True):
        """Train the model with the given data."""
        # First analyze feature correlations
        self.analyze_feature_correlations(X, y)
        
        # Optionally compare dimensionality reduction techniques
        if compare_reduction:
            reduction_results = self.compare_dimensionality_reduction(X, y)
            
            # Get the best method based on accuracy
            accuracies = [reduction_results[m]['accuracy'] for m in ['baseline', 'pca', 'ica']]
            best_method_idx = np.argmax(accuracies)
            methods = ['baseline', 'pca', 'ica']
            best_method = methods[best_method_idx]
            
            print(f"Based on comparison, {best_method} performs best with accuracy: {accuracies[best_method_idx]:.4f}")
            
            # Use the best method automatically
            use_pca = (best_method == 'pca')
            use_ica = (best_method == 'ica')
        
        # Split the data if test set not provided
        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
        
        # Apply dimensionality reduction if requested
        if use_pca:
            print("Using PCA for training...")
            X_train_scaled = self.min_max_scaler.fit_transform(X_train)
            X_test_scaled = self.min_max_scaler.transform(X_test)
            
            self.pca = PCA(n_components=0.95)  # Preserve 95% of variance
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            
            print(f"Reduced dimensions to {X_train_scaled.shape[1]} with PCA")
            
        elif use_ica:
            print("Using ICA for training...")
            X_train_scaled = self.min_max_scaler.fit_transform(X_train)
            X_test_scaled = self.min_max_scaler.transform(X_test)
            
            n_components_ica = min(10, X_train_scaled.shape[1])
            self.ica = FastICA(n_components=n_components_ica, random_state=42, max_iter=1000)
            X_train_scaled = self.ica.fit_transform(X_train_scaled)
            X_test_scaled = self.ica.transform(X_test_scaled)
            
            print(f"Reduced dimensions to {X_train_scaled.shape[1]} with ICA")
            
        else:
            # Try both scaling approaches without dimensionality reduction
            # Scale with StandardScaler first
            X_train_std = self.scaler.fit_transform(X_train)
            X_test_std = self.scaler.transform(X_test)
            
            # Also scale with MinMaxScaler
            X_train_minmax = self.min_max_scaler.fit_transform(X_train)
            X_test_minmax = self.min_max_scaler.transform(X_test)
            
            # Train models with both scaling methods to compare
            print("Training with StandardScaler...")
            self.model.fit(X_train_std, y_train)
            y_pred_std = self.model.predict(X_test_std)
            acc_std = accuracy_score(y_test, y_pred_std)
            print(f"StandardScaler accuracy: {acc_std:.4f}")
            
            # Reset model
            self.model = VotingClassifier(
                estimators=[
                    ('rf', self.rf),
                    ('gb', self.gb),
                    ('svm', self.svm),
                    ('nn', self.nn),
                    ('xgb', self.xgb)
                ],
                voting='soft'
            )
            
            print("Training with MinMaxScaler...")
            self.model.fit(X_train_minmax, y_train)
            y_pred_minmax = self.model.predict(X_test_minmax)
            acc_minmax = accuracy_score(y_test, y_pred_minmax)
            print(f"MinMaxScaler accuracy: {acc_minmax:.4f}")
            
            # Use the better scaling method
            if acc_minmax > acc_std:
                print("MinMaxScaler performs better. Using MinMaxScaler.")
                X_train_scaled = X_train_minmax
                X_test_scaled = X_test_minmax
                y_pred = y_pred_minmax
            else:
                print("StandardScaler performs better. Using StandardScaler.")
                X_train_scaled = X_train_std
                X_test_scaled = X_test_std
                y_pred = y_pred_std
        
        # Train the model if using dimensionality reduction
        if use_pca or use_ica:
            print("Training model with reduced dimensions...")
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy with reduced dimensions: {accuracy:.4f}")
        
        # If best model was chosen during hyperparameter tuning
        if self.best_model:
            self.best_model.fit(X_train_scaled, y_train)
            best_y_pred = self.best_model.predict(X_test_scaled)
            best_accuracy = accuracy_score(y_test, best_y_pred)
            ensemble_accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
            print(f"Best individual model accuracy: {best_accuracy:.4f}")
            
            # Use the better performing model
            if best_accuracy > ensemble_accuracy:
                print("Using best individual model for predictions")
                y_pred = best_y_pred
        
        return {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_train': X_train_scaled,
            'y_train': y_train
        }
    
    def evaluate(self, results):
        """Evaluate the model performance."""
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(results['y_test'], results['y_pred']))
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title(f'Confusion Matrix (Accuracy: {accuracy_score(results["y_test"], results["y_pred"]):.4f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('output/confusion_matrix.png')
        plt.close()
    
    def feature_importance(self, feature_names):
        """Plot feature importance."""
        # Check if the model or best model has feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif self.best_model and hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.model, 'estimators_') and hasattr(self.model.estimators_[0], 'feature_importances_'):
            # For voting classifier, use the first model that has feature_importances_
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importance = estimator.feature_importances_
                    break
        else:
            print("Feature importance not available for this model type")
            return
            
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('output/feature_importance.png')
        plt.close()
    
    def cross_validate(self, X, y):
        """Perform cross-validation."""
        X_scaled = self.min_max_scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=5, n_jobs=-1)
        print(f"\nCross-validation scores: {scores}")
        print(f"Average CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    def compare_balancing_methods_t_test(self, X, y, test_size=0.2, n_repeats=10, cv=5):
        """
        Compare different data balancing methods using paired t-tests.
        
        Args:
            X: Original feature matrix
            y: Target vector
            test_size: Proportion for test split
            n_repeats: Number of times to repeat CV for robust comparison
            cv: Number of cross-validation folds
            
        Returns:
            DataFrame with p-values and significance results
        """
        print("Performing statistical significance testing for data balancing methods...")
        
        # Split data once for consistent testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Apply different balancing methods
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        
        # Original (imbalanced)
        X_orig, y_orig = X_train, y_train
        
        # Random undersampling
        rus = RandomUnderSampler(random_state=42)
        X_rus, y_rus = rus.fit_resample(X_train, y_train)
        
        # SMOTE oversampling
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        
        # Combined approach
        oversampler = SMOTE(sampling_strategy=0.7, random_state=42)
        undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_temp, y_temp = oversampler.fit_resample(X_train, y_train)
        X_combined, y_combined = undersampler.fit_resample(X_temp, y_temp)
        
        # Create base classifier for testing
        base_clf = RandomForestClassifier(n_estimators=200, random_state=42)
        
        # Dictionary to store scores for each method
        methods = {
            'original': (X_orig, y_orig),
            'undersampled': (X_rus, y_rus),
            'smote': (X_smote, y_smote),
            'combined': (X_combined, y_combined)
        }
        
        # Store all scores
        all_scores = {method: [] for method in methods}
        
        # Perform repeated cross-validation
        for i in range(n_repeats):
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=i)
            
            # For each method
            for method_name, (X_bal, y_bal) in methods.items():
                X_bal_scaled = self.min_max_scaler.fit_transform(X_bal)
                
                # Collect scores across folds
                for train_idx, val_idx in skf.split(X_bal_scaled, y_bal):
                    X_fold_train, X_fold_val = X_bal_scaled[train_idx], X_bal_scaled[val_idx]
                    y_fold_train, y_fold_val = y_bal[train_idx], y_bal[val_idx]
                    
                    # Train and evaluate
                    base_clf.fit(X_fold_train, y_fold_train)
                    score = accuracy_score(y_fold_val, base_clf.predict(X_fold_val))
                    all_scores[method_name].append(score)
        
        # Perform t-tests comparing original to each balancing method
        t_test_results = []
        
        for method in ['undersampled', 'smote', 'combined']:
            t_stat, p_val = stats.ttest_rel(all_scores['original'], all_scores[method])
            
            # Mean scores
            original_mean = np.mean(all_scores['original'])
            method_mean = np.mean(all_scores[method])
            
            # Store results
            t_test_results.append({
                'comparison': f'original_vs_{method}',
                'original_mean': original_mean,
                'method_mean': method_mean,
                'mean_diff': method_mean - original_mean,
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'better_method': method if method_mean > original_mean and p_val < 0.05 else
                                'original' if original_mean > method_mean and p_val < 0.05 else
                                'no significant difference'
            })
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(t_test_results)
        
        # Print results
        print("\nBalancing Methods T-Test Results:")
        print(results_df[['comparison', 'original_mean', 'method_mean', 'p_value', 'significant', 'better_method']])
        
        # Save results to CSV
        results_df.to_csv('output/balancing_methods_t_test.csv', index=False)
        
        # Visualize the results
        plt.figure(figsize=(12, 8))
        
        # Plot mean scores for each method
        mean_scores = {method: np.mean(scores) for method, scores in all_scores.items()}
        methods_list = list(mean_scores.keys())
        scores_list = [mean_scores[m] for m in methods_list]
        
        # Create bar chart with error bars
        std_errors = {method: np.std(scores) / np.sqrt(len(scores)) for method, scores in all_scores.items()}
        errors = [std_errors[m] for m in methods_list]
        
        bars = plt.bar(methods_list, scores_list, yerr=errors, capsize=10, alpha=0.7)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{scores_list[i]:.4f}', ha='center', va='bottom')
        
        # Add significance indicators
        for result in t_test_results:
            comp = result['comparison'].split('_vs_')[1]
            idx = methods_list.index(comp)
            sig = '*' if result['significant'] else 'ns'
            plt.text(idx, scores_list[idx] - 0.03, sig, ha='center', fontsize=14, 
                    color='red' if result['significant'] else 'black')
        
        plt.ylim(0.5, 1.0)
        plt.ylabel('Mean Accuracy Score')
        plt.title('Comparison of Data Balancing Methods with Statistical Significance')
        plt.legend(['* = significant difference (p<0.05)', 'ns = not significant'])
        plt.tight_layout()
        plt.savefig('output/balancing_methods_significance.png')
        plt.close()
        
        return results_df
    
    def apply_feature_t_tests(self, X, y):
        """
        Perform t-tests to identify features that differ significantly between classes.
        
        Args:
            X: Feature matrix
            y: Target vector (binary)
            
        Returns:
            DataFrame with t-test results for each feature
        """
        print("Performing t-tests for feature significance between classes...")
        
        # Convert to DataFrame for easier handling
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Separate samples by class
        class_0_samples = X[y == 0]
        class_1_samples = X[y == 1]
        
        # Store results
        results = []
        
        # Perform t-test for each feature
        for col in X.columns:
            t_stat, p_val = stats.ttest_ind(
                class_0_samples[col].values,
                class_1_samples[col].values,
                equal_var=False  # Welch's t-test (does not assume equal variance)
            )
            
            # Calculate effect size (Cohen's d)
            mean_diff = class_1_samples[col].mean() - class_0_samples[col].mean()
            pooled_std = np.sqrt((class_0_samples[col].std()**2 + class_1_samples[col].std()**2) / 2)
            effect_size = mean_diff / pooled_std if pooled_std != 0 else 0
            
            results.append({
                'feature': col,
                't_statistic': t_stat,
                'p_value': p_val,
                'mean_class_0': class_0_samples[col].mean(),
                'mean_class_1': class_1_samples[col].mean(),
                'mean_difference': mean_diff,
                'effect_size': effect_size,
                'significant': p_val < 0.05
            })
        
        # Create DataFrame and sort by p-value
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p_value')
        
        # Apply Bonferroni correction for multiple testing
        results_df['bonferroni_significant'] = results_df['p_value'] < (0.05 / len(X.columns))
        
        # Count significant features
        n_significant = results_df['significant'].sum()
        n_bonferroni_significant = results_df['bonferroni_significant'].sum()
        
        print(f"\nFound {n_significant} features with significant differences between classes (p < 0.05)")
        print(f"After Bonferroni correction: {n_bonferroni_significant} significant features")
        
        # Print top significant features
        print("\nTop significant features:")
        top_features = results_df[results_df['significant']].head(10)
        for _, row in top_features.iterrows():
            print(f"Feature {row['feature']}: p={row['p_value']:.4f}, effect size={row['effect_size']:.4f}")
        
        # Save results to CSV
        results_df.to_csv('output/feature_t_tests.csv', index=False)
        
        # Visualize significant features
        plt.figure(figsize=(12, 8))
        
        # Plot -log10(p-value) for easier visualization (higher values = more significant)
        results_df['neg_log_p'] = -np.log10(results_df['p_value'])
        
        # Sort for better visualization
        plot_df = results_df.sort_values('neg_log_p', ascending=False).head(20)
        
        bars = plt.bar(plot_df['feature'], plot_df['neg_log_p'], color=[
            'red' if sig else 'gray' for sig in plot_df['bonferroni_significant']
        ])
        
        # Add horizontal line for significance threshold (p=0.05)
        plt.axhline(y=-np.log10(0.05), color='blue', linestyle='--', label='p=0.05')
        
        # Add horizontal line for Bonferroni threshold
        plt.axhline(y=-np.log10(0.05/len(X.columns)), color='green', linestyle='--', 
                   label=f'Bonferroni (p={0.05/len(X.columns):.6f})')
        
        plt.xticks(rotation=90)
        plt.ylabel('-log10(p-value)')
        plt.title('Feature Significance (t-test) Between Classes')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/feature_t_test_significance.png')
        plt.close()
        
        return results_df
    
    def predict_with_confidence_intervals(self, X_test, y_test, n_bootstrap=1000, alpha=0.05):
        """
        Make predictions with bootstrap confidence intervals.
        
        Args:
            X_test: Test feature matrix
            y_test: True test labels
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level (default 0.05 for 95% confidence interval)
            
        Returns:
            Dictionary with predictions, confidence intervals, and metrics
        """
        print(f"Computing bootstrap {(1-alpha)*100:.0f}% confidence intervals for model performance...")
        
        # Scale the test data
        X_test_scaled = self.min_max_scaler.transform(X_test)
        
        # Original prediction
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        
        # Original performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Bootstrap predictions
        bootstrap_accuracies = []
        bootstrap_predictions = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Perform bootstrap
        for i in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(X_test_scaled), len(X_test_scaled), replace=True)
            X_bootstrap = X_test_scaled[indices]
            y_bootstrap = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]
            
            # Predict on bootstrap sample
            y_bootstrap_pred = self.model.predict(X_bootstrap)
            bootstrap_predictions.append(y_bootstrap_pred)
            
            # Calculate accuracy
            bootstrap_accuracies.append(accuracy_score(y_bootstrap, y_bootstrap_pred))
        
        # Calculate confidence intervals for overall accuracy
        lower_bound = np.percentile(bootstrap_accuracies, alpha/2 * 100)
        upper_bound = np.percentile(bootstrap_accuracies, (1 - alpha/2) * 100)
        
        print(f"Original accuracy: {accuracy:.4f}")
        print(f"Bootstrap {(1-alpha)*100:.0f}% CI: [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        # Create t-score distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(bootstrap_accuracies, bins=30, alpha=0.7, color='blue')
        plt.axvline(x=accuracy, color='red', linestyle='--', label=f'Original accuracy: {accuracy:.4f}')
        plt.axvline(x=lower_bound, color='green', linestyle='--', 
                   label=f'Lower bound ({alpha/2*100:.1f}%): {lower_bound:.4f}')
        plt.axvline(x=upper_bound, color='green', linestyle='--', 
                   label=f'Upper bound ({(1-alpha/2)*100:.1f}%): {upper_bound:.4f}')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.title(f'Bootstrap Distribution of Accuracy ({n_bootstrap} resamples)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/bootstrap_accuracy_distribution.png')
        plt.close()
        
        # Perform t-test to check if accuracy is significantly better than baseline
        # Baseline is predicting the majority class
        majority_class = np.argmax(np.bincount(y_test))
        baseline_accuracy = sum(y_test == majority_class) / len(y_test)
        
        t_stat, p_value = stats.ttest_1samp(bootstrap_accuracies, baseline_accuracy)
        
        print(f"Baseline accuracy (majority class): {baseline_accuracy:.4f}")
        print(f"T-test against baseline: t={t_stat:.4f}, p={p_value:.6f}")
        
        if p_value < 0.05 and accuracy > baseline_accuracy:
            print("The model performs significantly better than the baseline (p < 0.05)")
        else:
            print("The model does not perform significantly better than the baseline (p >= 0.05)")
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'accuracy_ci_lower': lower_bound,
            'accuracy_ci_upper': upper_bound,
            'bootstrap_accuracies': bootstrap_accuracies,
            'p_value_vs_baseline': p_value,
            't_statistic_vs_baseline': t_stat
        } 