"""
Support Vector Machine with RBF kernel.
From syllabus: Week 7 (Kernel Methods, SVM).
"""

from pathlib import Path

import joblib
import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC


class RBFSupportVectorMachine:
    """
    SVM with Radial Basis Function (RBF) kernel.
    Uses the kernel trick to handle non-linear decision boundaries.
    
    Note: SVMs don't scale well to very large datasets, so we use
    a subset for hyperparameter tuning.
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
        tune_hyperparams: bool = True,
        subsample_for_tuning: int = 50000
    ):
        """
        Train the SVM model.
        For large datasets, subsamples for hyperparameter tuning.
        """
        if tune_hyperparams:
            # Subsample for tuning (SVM is O(n^2) or worse)
            if len(X_train) > subsample_for_tuning:
                print(f"Subsampling to {subsample_for_tuning:,} for SVM hyperparameter tuning...")
                np.random.seed(self.random_state)
                indices = np.random.choice(len(X_train), subsample_for_tuning, replace=False)
                X_tune = X_train[indices]
                y_tune = y_train[indices]
            else:
                X_tune = X_train
                y_tune = y_train
            
            # Randomized search is more efficient for SVM
            param_dist = {
                'C': loguniform(0.01, 100),
                'gamma': loguniform(0.0001, 1)
            }
            
            base_model = SVC(
                kernel='rbf',
                probability=True,  # Enable probability estimates
                random_state=self.random_state,
                cache_size=1000  # Increase cache for speed
            )
            
            print("Tuning RBF SVM hyperparameters...")
            random_search = RandomizedSearchCV(
                base_model,
                param_dist,
                n_iter=20,  # Limited iterations due to SVM cost
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
            random_search.fit(X_tune, y_tune)
            
            self.best_params = random_search.best_params_
            print(f"Best params: {self.best_params}")
            print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
            
            # Retrain on full training data with best params
            print("Retraining on full dataset with best params...")
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                cache_size=1000,
                **self.best_params
            )
            self.model.fit(X_train, y_train)
        else:
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state,
                cache_size=1000
            )
            self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path: str = "outputs/models/svm.joblib"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params
        }, path)
        
    def load(self, path: str = "outputs/models/svm.joblib"):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']
        return self

