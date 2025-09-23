import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class PowerSystemFaultClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def load_data(self, features_file='power_system_features.csv'):
        """Load the features dataset."""
        print("Loading dataset...")
        self.df = pd.read_csv(features_file)
        print(f"Dataset loaded: {self.df.shape[0]} samples, {self.df.shape[1]-1} features")
        
        # Separate features and labels
        self.X = self.df.drop('label', axis=1)
        self.y = self.df['label']
        self.feature_names = self.X.columns.tolist()
        
        # Print label distribution
        print("\nLabel distribution:")
        print(self.y.value_counts())
        
        return self.X, self.y
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess the data: encode labels, split, and scale."""
        print("\nPreprocessing data...")
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.label_classes = self.label_encoder.classes_
        print(f"Label classes: {self.label_classes}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=test_size, random_state=random_state, 
            stratify=self.y_encoded
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def train_models(self):
        """Train multiple models and compare performance."""
        print("\nTraining multiple models...")
        
        # Define models
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', random_state=42, probability=True
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
            )
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for SVM and Neural Network
            if name in ['SVM', 'Neural Network']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Predictions
            y_pred = model.predict(X_test_use)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation score
            if name in ['SVM', 'Neural Network']:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            self.models[name] = model
            
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model_name = best_model_name
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best CV score: {results[best_model_name]['cv_mean']:.4f}")
        
        return results
    
    def optimize_best_model(self):
        """Perform hyperparameter optimization on the best model."""
        print(f"\nOptimizing {self.best_model_name}...")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            X_use = self.X_train
            
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            X_use = self.X_train
            
        elif self.best_model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            }
            X_use = self.X_train_scaled
            
        else:  # Neural Network
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
            X_use = self.X_train_scaled
        
        # Grid search
        grid_search = GridSearchCV(
            self.best_model, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_use, self.y_train)
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        self.models[f"{self.best_model_name}_Optimized"] = self.best_model
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
    def evaluate_final_model(self):
        """Evaluate the final optimized model."""
        print(f"\nFinal evaluation of {self.best_model_name}...")
        
        # Use appropriate data
        if self.best_model_name in ['SVM', 'Neural Network']:
            X_test_use = self.X_test_scaled
        else:
            X_test_use = self.X_test
        
        # Predictions
        y_pred = self.best_model.predict(X_test_use)
        y_pred_proba = self.best_model.predict_proba(X_test_use)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Final Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            self.y_test, y_pred, 
            target_names=self.label_classes
        ))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_classes,
                    yticklabels=self.label_classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Feature importance (if available)
        self.plot_feature_importance()
        
        return accuracy, y_pred, y_pred_proba
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        if hasattr(self.best_model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.tail(10))
    
    def save_model(self, filename_prefix='power_system_fault_model'):
        """Save the trained model and preprocessors."""
        print(f"\nSaving model components...")
        
        # Save model
        joblib.dump(self.best_model, f'{filename_prefix}.pkl')
        joblib.dump(self.scaler, f'{filename_prefix}_scaler.pkl')
        joblib.dump(self.label_encoder, f'{filename_prefix}_label_encoder.pkl')
        
        # Save feature names and other metadata
        metadata = {
            'feature_names': self.feature_names,
            'label_classes': self.label_classes.tolist(),
            'model_name': self.best_model_name,
            'scaler_needed': self.best_model_name in ['SVM', 'Neural Network']
        }
        
        import json
        with open(f'{filename_prefix}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved as: {filename_prefix}.pkl")
        print(f"Scaler saved as: {filename_prefix}_scaler.pkl")
        print(f"Label encoder saved as: {filename_prefix}_label_encoder.pkl")
        print(f"Metadata saved as: {filename_prefix}_metadata.json")
    
    def predict_sample(self, features):
        """Predict a single sample."""
        if self.best_model_name in ['SVM', 'Neural Network']:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.best_model.predict(features_scaled)[0]
            probability = self.best_model.predict_proba(features_scaled)[0]
        else:
            prediction = self.best_model.predict(features.reshape(1, -1))[0]
            probability = self.best_model.predict_proba(features.reshape(1, -1))[0]
        
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        prob_dict = {}
        for i, prob in enumerate(probability):
            label = self.label_encoder.inverse_transform([i])[0]
            prob_dict[label] = prob
        
        return predicted_label, prob_dict

def plot_model_comparison(results):
    """Plot comparison of different models."""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    cv_means = [results[model]['cv_mean'] for model in models]
    cv_stds = [results[model]['cv_std'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.7)
    bars2 = ax.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.7, yerr=cv_stds, capsize=5)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# --- Main Training Pipeline ---

def main():
    """Main training pipeline."""
    print("=== Power System Fault Detection Model Training ===")
    
    # Initialize classifier
    classifier = PowerSystemFaultClassifier()
    
    # Load data
    try:
        X, y = classifier.load_data()
    except FileNotFoundError:
        print("Error: power_system_features.csv not found!")
        print("Please run the dataset generator first.")
        return
    
    # Preprocess data
    classifier.preprocess_data()
    
    # Train multiple models
    results = classifier.train_models()
    
    # Plot model comparison
    plot_model_comparison(results)
    
    # Optimize best model
    classifier.optimize_best_model()
    
    # Final evaluation
    accuracy, predictions, probabilities = classifier.evaluate_final_model()
    
    # Save model
    classifier.save_model()
    
    print("\n=== Training Completed Successfully! ===")
    print(f"Final Model: {classifier.best_model_name}")
    print(f"Final Accuracy: {accuracy:.4f}")
    
    # Demonstrate prediction on a sample
    print("\n=== Sample Prediction Demonstration ===")
    sample_idx = 0
    sample_features = classifier.X_test.iloc[sample_idx].values
    true_label = classifier.label_encoder.inverse_transform([classifier.y_test[sample_idx]])[0]
    
    pred_label, pred_probs = classifier.predict_sample(sample_features)
    
    print(f"Sample {sample_idx}:")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {pred_label}")
    print("Prediction Probabilities:")
    for label, prob in pred_probs.items():
        print(f"  {label}: {prob:.4f}")

if __name__ == "__main__":
    main()