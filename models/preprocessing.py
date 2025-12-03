"""
Data preprocessing pipeline for diabetes prediction.
Handles loading, cleaning, encoding, and scaling of features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessor:
    """Handles all data preprocessing for the diabetes dataset."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_cols = [
            'gender', 'ethnicity', 'education_level', 
            'income_level', 'smoking_status', 'employment_status'
        ]
        self.binary_cols = [
            'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'
        ]
        self.numerical_cols = None  # Set during fitting
        
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test datasets."""
        train_df = pd.read_csv(self.data_dir / "train.csv")
        test_df = pd.read_csv(self.data_dir / "test.csv")
        return train_df, test_df
    
    def _get_numerical_cols(self, df: pd.DataFrame) -> list[str]:
        """Identify numerical columns (excluding id and target)."""
        exclude = ['id', 'diagnosed_diabetes'] + self.categorical_cols + self.binary_cols
        return [col for col in df.columns if col not in exclude]
    
    def fit_transform(self, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessing on training data and transform it.
        Returns X_train, y_train as numpy arrays.
        """
        # Identify numerical columns
        self.numerical_cols = self._get_numerical_cols(train_df)
        
        # Extract target
        y = train_df['diagnosed_diabetes'].values
        
        # Process features
        X_processed = self._process_features(train_df, fit=True)
        
        return X_processed, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessors.
        For test data (no target column).
        """
        return self._process_features(df, fit=False)
    
    def _process_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Process all features: encode categoricals, scale numericals."""
        processed_parts = []
        feature_names = []
        
        # Process numerical columns
        numerical_data = df[self.numerical_cols].values
        if fit:
            numerical_scaled = self.scaler.fit_transform(numerical_data)
        else:
            numerical_scaled = self.scaler.transform(numerical_data)
        processed_parts.append(numerical_scaled)
        feature_names.extend(self.numerical_cols)
        
        # Process categorical columns (label encoding)
        for col in self.categorical_cols:
            if fit:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                encoded = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                ).values
            processed_parts.append(encoded.reshape(-1, 1))
            feature_names.append(col)
        
        # Process binary columns (already 0/1)
        for col in self.binary_cols:
            processed_parts.append(df[col].values.reshape(-1, 1))
            feature_names.append(col)
        
        if fit:
            self.feature_names = feature_names
        
        return np.hstack(processed_parts)
    
    def get_train_val_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        val_size: float = 0.15,
        random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split training data into train and validation sets."""
        return train_test_split(
            X, y, 
            test_size=val_size, 
            random_state=random_state, 
            stratify=y
        )
    
    def get_test_ids(self, test_df: pd.DataFrame) -> np.ndarray:
        """Get test IDs for submission."""
        return test_df['id'].values


def load_and_preprocess(val_size: float = 0.15, random_state: int = 42):
    """
    Convenience function to load and preprocess all data.
    Returns: X_train, X_val, y_train, y_val, X_test, test_ids, processor
    """
    processor = DataProcessor()
    train_df, test_df = processor.load_data()
    
    # Fit and transform training data
    X, y = processor.fit_transform(train_df)
    
    # Split into train/val
    X_train, X_val, y_train, y_val = processor.get_train_val_split(
        X, y, val_size=val_size, random_state=random_state
    )
    
    # Transform test data
    X_test = processor.transform(test_df)
    test_ids = processor.get_test_ids(test_df)
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution (train): {np.bincount(y_train.astype(int))}")
    
    return X_train, X_val, y_train, y_val, X_test, test_ids, processor

