"""
Neural Network (Multi-Layer Perceptron) for tabular data.
From syllabus: Week 9 (Neural Networks).
Uses PyTorch for implementation.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TabularMLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular data.
    Architecture: Input -> [Hidden + BatchNorm + ReLU + Dropout] x N -> Output
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (single unit for binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class NeuralNetworkClassifier:
    """
    PyTorch-based neural network for diabetes prediction.
    Includes training loop with early stopping and proper GPU support.
    """
    
    def __init__(
        self, 
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        
        # Set seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> DataLoader:
        """Convert numpy arrays to PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=(y is not None)
        )
        return loader
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        tune_hyperparams: bool = True  # Not used but kept for API consistency
    ):
        """Train the neural network with early stopping."""
        print(f"Training on device: {self.device}")
        
        input_dim = X_train.shape[1]
        self.model = TabularMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Class weights for imbalanced data
        pos_weight = torch.FloatTensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        train_loader = self._prepare_data(X_train, y_train)
        val_loader = self._prepare_data(X_val, y_val) if X_val is not None else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        print(f"Training neural network ({sum(p.numel() for p in self.model.parameters()):,} parameters)...")
        
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                all_probs = []
                all_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        all_probs.extend(probs)
                        all_labels.extend(batch_y.cpu().numpy())
                
                val_loss /= len(val_loader)
                self.training_history['val_loss'].append(val_loss)
                
                # Calculate AUC
                from sklearn.metrics import roc_auc_score
                val_auc = roc_auc_score(all_labels, all_probs)
                self.training_history['val_auc'].append(val_auc)
                
                scheduler.step(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.max_epochs} - "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}")
        
        # Load best model state
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        all_probs = []
        with torch.no_grad():
            # Process in batches to avoid memory issues
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                outputs = self.model(batch).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs if len(probs.shape) > 0 else [probs.item()])
        
        return np.array(all_probs)
    
    def save(self, path: str = "outputs/models/neural_network.pt"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'training_history': self.training_history
        }, path)
        
    def load(self, path: str = "outputs/models/neural_network.pt", input_dim: int = None):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout = checkpoint['dropout']
        self.training_history = checkpoint['training_history']
        
        if input_dim is not None:
            self.model = TabularMLP(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self

