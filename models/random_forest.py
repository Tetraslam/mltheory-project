"""
Random Forest classifier.
From syllabus: Week 8 (Decision Trees, Ensemble Methods).
"""

from pathlib import Path

import joblib
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


class RandomForest:
    """
    Random Forest - an ensemble of decision trees using bagging.
    Each tree is trained on a bootstrap sample with random feature subsets.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        tune_hyperparams: bool = True
    ):
        """Train the Random Forest model with optional hyperparameter tuning."""
        if tune_hyperparams:
            param_dist = {
                'n_estimators': randint(100, 500),
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', 0.3, 0.5]
            }
            
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )
            
            print("Tuning Random Forest hyperparameters...")
            random_search = RandomizedSearchCV(
                base_model,
                param_dist,
                n_iter=15,  # Reduced from 30
                cv=3,
                scoring='roc_auc',
                n_jobs=8,  # Explicit instead of -1
                verbose=1,
                random_state=self.random_state
            )
            random_search.fit(X_train, y_train)
            
            self.model = random_search.best_estimator_
            self.best_params = random_search.best_params_
            print(f"Best params: {self.best_params}")
            print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names: list) -> dict:
        """Get feature importance from the forest."""
        importance = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path: str = "outputs/models/random_forest.joblib"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params
        }, path)
        
    def load(self, path: str = "outputs/models/random_forest.joblib"):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']
        return self

