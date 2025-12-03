"""
LightGBM (Light Gradient Boosting Machine).
NOT from syllabus - state-of-the-art for tabular data.
Uses gradient boosting with histogram-based learning for speed.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV


class LightGBMClassifier:
    """
    LightGBM - Microsoft's gradient boosting framework.
    Key advantages over traditional GBM:
    - Histogram-based algorithm (faster, less memory)
    - Leaf-wise tree growth (better accuracy)
    - Native categorical feature support
    - Efficient handling of large datasets
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
        """Train LightGBM with optional hyperparameter tuning."""
        if tune_hyperparams:
            param_dist = {
                'n_estimators': randint(100, 500),
                'max_depth': randint(3, 15),
                'num_leaves': randint(20, 150),
                'learning_rate': uniform(0.01, 0.2),
                'min_child_samples': randint(10, 100),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1)
            }
            
            base_model = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
                class_weight='balanced'
            )
            
            print("Tuning LightGBM hyperparameters...")
            random_search = RandomizedSearchCV(
                base_model,
                param_dist,
                n_iter=20,  # Reduced from 40
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
            
            # Optionally retrain with early stopping using validation set
            if X_val is not None and y_val is not None:
                print("Retraining with early stopping...")
                self.model = lgb.LGBMClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=-1,
                    class_weight='balanced',
                    **self.best_params
                )
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
        else:
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                num_leaves=50,
                learning_rate=0.05,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
                class_weight='balanced'
            )
            
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names: list) -> dict:
        """Get feature importance (gain-based)."""
        importance = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path: str = "outputs/models/lightgbm.joblib"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params
        }, path)
        
    def load(self, path: str = "outputs/models/lightgbm.joblib"):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']
        return self

