"""
Logistic Regression with ElasticNet regularization (L1 + L2).
From syllabus: Week 2-4 (Linear Methods + Regularization).
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class ElasticNetLogisticRegression:
    """
    Logistic Regression with ElasticNet regularization.
    Combines L1 (Lasso) and L2 (Ridge) penalties for both
    feature selection and regularization.
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
        """
        Train the logistic regression model.
        Uses ElasticNet penalty with hyperparameter tuning.
        """
        if tune_hyperparams:
            # ElasticNet uses 'l1_ratio' to balance L1 and L2
            # l1_ratio=0 is pure L2 (Ridge), l1_ratio=1 is pure L1 (Lasso)
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
            }
            
            base_model = LogisticRegression(
                penalty='elasticnet',
                solver='saga',  # Required for elasticnet
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Use validation set if provided, otherwise 3-fold CV
            if X_val is not None and y_val is not None:
                # Combine for CV but we'll use validation performance
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.hstack([y_train, y_val])
                cv = 3
            else:
                X_combined = X_train
                y_combined = y_train
                cv = 3
            
            print("Tuning ElasticNet Logistic Regression hyperparameters...")
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=cv, 
                scoring='roc_auc',
                n_jobs=8,  # Explicit instead of -1
                verbose=1
            )
            grid_search.fit(X_combined, y_combined)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best params: {self.best_params}")
            print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        else:
            # Default reasonable params
            self.model = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                C=1.0,
                l1_ratio=0.5,
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
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
        """Get feature coefficients (importance for linear models)."""
        coeffs = self.model.coef_[0]
        importance = dict(zip(feature_names, np.abs(coeffs)))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path: str = "outputs/models/logistic_regression.joblib"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params
        }, path)
        
    def load(self, path: str = "outputs/models/logistic_regression.joblib"):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']
        return self

