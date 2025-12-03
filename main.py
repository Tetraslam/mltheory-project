"""
Main training and evaluation script for diabetes prediction.
Trains 5 models, evaluates them, and generates comparison reports.

Models:
1. Logistic Regression with ElasticNet (from syllabus)
2. SVM with RBF Kernel (from syllabus)
3. Random Forest (from syllabus)
4. LightGBM (NOT from syllabus - state-of-the-art for tabular)
5. Neural Network MLP (from syllabus)
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

from models.evaluation import ModelEvaluator
from models.lightgbm_model import LightGBMClassifier
from models.logistic_regression import ElasticNetLogisticRegression
from models.neural_network import NeuralNetworkClassifier
from models.preprocessing import DataProcessor, load_and_preprocess
from models.random_forest import RandomForest
from models.svm import RBFSupportVectorMachine


def plot_feature_importance(importances: dict, title: str, save_path: str, top_n: int = 15):
    """Plot feature importance bar chart."""
    top_features = dict(list(importances.items())[:top_n])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(list(top_features.keys())[::-1], list(top_features.values())[::-1])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(history: dict, save_path: str):
    """Plot neural network training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC curve
    if 'val_auc' in history and len(history['val_auc']) > 0:
        axes[1].plot(history['val_auc'], label='Val AUC', linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_data_distribution(processor: DataProcessor, train_df: pd.DataFrame, save_dir: Path):
    """Generate EDA plots for the dataset."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Target distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    train_df['diagnosed_diabetes'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'coral'])
    ax.set_xlabel('Diagnosed Diabetes')
    ax.set_ylabel('Count')
    ax.set_title('Target Variable Distribution')
    ax.set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'], rotation=0)
    for i, v in enumerate(train_df['diagnosed_diabetes'].value_counts().values):
        ax.text(i, v + 1000, f'{v:,}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir / 'target_distribution.png', dpi=150)
    plt.close()
    
    # Numerical feature distributions
    numerical_cols = processor._get_numerical_cols(train_df)
    n_cols = 4
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        train_df[col].hist(ax=axes[i], bins=50, alpha=0.7, color='steelblue')
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(labelsize=8)
    
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Numerical Feature Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'numerical_distributions.png', dpi=150)
    plt.close()
    
    # Correlation heatmap for numerical features
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = train_df[numerical_cols + ['diagnosed_diabetes']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'correlation_heatmap.png', dpi=150)
    plt.close()


def train_and_evaluate_all(
    tune_hyperparams: bool = True,
    skip_svm: bool = False,
    skip_nn: bool = False
):
    """
    Main training pipeline.
    
    Args:
        tune_hyperparams: Whether to tune hyperparameters (slower but better)
        skip_svm: Skip SVM (it's slow on large datasets)
        skip_nn: Skip neural network
    """
    print("="*70)
    print(" Diabetes Prediction - Model Training & Evaluation")
    print("="*70)
    
    # Load and preprocess data
    print("\n[1/7] Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, X_test, test_ids, processor = load_and_preprocess()
    
    # Load raw data for EDA
    train_df, _ = processor.load_data()
    
    # Generate EDA plots
    print("\n[2/7] Generating EDA visualizations...")
    plot_data_distribution(processor, train_df, Path("outputs/figures"))
    
    evaluator = ModelEvaluator()
    results = {}
    
    # Model 1: Logistic Regression with ElasticNet
    print("\n[3/7] Training Logistic Regression with ElasticNet...")
    start = time.time()
    lr_model = ElasticNetLogisticRegression()
    lr_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=tune_hyperparams)
    
    y_pred = lr_model.predict(X_val)
    y_prob = lr_model.predict_proba(X_val)
    
    evaluator.evaluate("Logistic Regression (ElasticNet)", y_val, y_pred, y_prob)
    evaluator.print_report("Logistic Regression (ElasticNet)")
    
    # Feature importance
    importance = lr_model.get_feature_importance(processor.feature_names)
    plot_feature_importance(importance, "Logistic Regression - Feature Importance",
                           "outputs/figures/lr_feature_importance.png")
    
    lr_model.save()
    results['Logistic Regression'] = {'time': time.time() - start, 'model': lr_model}
    
    # Model 2: SVM with RBF Kernel
    if not skip_svm:
        print("\n[4/7] Training SVM with RBF Kernel...")
        start = time.time()
        svm_model = RBFSupportVectorMachine()
        svm_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=tune_hyperparams)
        
        y_pred = svm_model.predict(X_val)
        y_prob = svm_model.predict_proba(X_val)
        
        evaluator.evaluate("SVM (RBF Kernel)", y_val, y_pred, y_prob)
        evaluator.print_report("SVM (RBF Kernel)")
        
        svm_model.save()
        results['SVM'] = {'time': time.time() - start, 'model': svm_model}
    else:
        print("\n[4/7] Skipping SVM (--skip-svm flag)")
    
    # Model 3: Random Forest
    print("\n[5/7] Training Random Forest...")
    start = time.time()
    rf_model = RandomForest()
    rf_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=tune_hyperparams)
    
    y_pred = rf_model.predict(X_val)
    y_prob = rf_model.predict_proba(X_val)
    
    evaluator.evaluate("Random Forest", y_val, y_pred, y_prob)
    evaluator.print_report("Random Forest")
    
    importance = rf_model.get_feature_importance(processor.feature_names)
    plot_feature_importance(importance, "Random Forest - Feature Importance",
                           "outputs/figures/rf_feature_importance.png")
    
    rf_model.save()
    results['Random Forest'] = {'time': time.time() - start, 'model': rf_model}
    
    # Model 4: LightGBM
    print("\n[6/7] Training LightGBM...")
    start = time.time()
    lgb_model = LightGBMClassifier()
    lgb_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=tune_hyperparams)
    
    y_pred = lgb_model.predict(X_val)
    y_prob = lgb_model.predict_proba(X_val)
    
    evaluator.evaluate("LightGBM", y_val, y_pred, y_prob)
    evaluator.print_report("LightGBM")
    
    importance = lgb_model.get_feature_importance(processor.feature_names)
    plot_feature_importance(importance, "LightGBM - Feature Importance",
                           "outputs/figures/lgb_feature_importance.png")
    
    lgb_model.save()
    results['LightGBM'] = {'time': time.time() - start, 'model': lgb_model}
    
    # Model 5: Neural Network
    if not skip_nn:
        print("\n[7/7] Training Neural Network...")
        start = time.time()
        nn_model = NeuralNetworkClassifier(
            hidden_dims=[256, 128, 64],
            dropout=0.3,
            learning_rate=0.001,
            batch_size=2048,
            max_epochs=100,
            patience=15
        )
        nn_model.train(X_train, y_train, X_val, y_val)
        
        y_pred = nn_model.predict(X_val)
        y_prob = nn_model.predict_proba(X_val)
        
        evaluator.evaluate("Neural Network (MLP)", y_val, y_pred, y_prob)
        evaluator.print_report("Neural Network (MLP)")
        
        plot_training_curves(nn_model.training_history, "outputs/figures/nn_training_curves.png")
        
        nn_model.save()
        results['Neural Network'] = {'time': time.time() - start, 'model': nn_model}
    else:
        print("\n[7/7] Skipping Neural Network (--skip-nn flag)")
    
    # Generate comparison plots and save results
    print("\n" + "="*70)
    print(" Generating Comparison Reports")
    print("="*70)
    
    evaluator.generate_all_plots()
    evaluator.save_results()
    
    # Print summary table
    summary = evaluator.get_summary_table()
    print("\n" + "="*70)
    print(" Model Comparison Summary")
    print("="*70)
    print(summary.to_string())
    
    # Print training times
    print("\n" + "-"*40)
    print(" Training Times")
    print("-"*40)
    for name, data in results.items():
        print(f"{name}: {data['time']:.1f}s")
    
    # Find best model
    if 'roc_auc' in summary.columns:
        best_model = summary['roc_auc'].idxmax()
        best_auc = summary.loc[best_model, 'roc_auc']
        print(f"\nBest model by ROC-AUC: {best_model} ({best_auc:.4f})")
    
    # Generate predictions on test set using best model
    print("\n" + "="*70)
    print(" Generating Test Predictions")
    print("="*70)
    
    # Use LightGBM for final predictions (typically best for tabular data)
    best = results.get('LightGBM', results.get('Random Forest'))
    if best:
        test_probs = best['model'].predict_proba(X_test)
        test_preds = (test_probs >= 0.5).astype(int)
        
        submission = pd.DataFrame({
            'id': test_ids,
            'diagnosed_diabetes': test_preds
        })
        submission.to_csv('outputs/submission.csv', index=False)
        print("Submission saved to outputs/submission.csv")
        print(f"Predicted positive rate: {test_preds.mean() * 100:.2f}%")
    
    print("\n" + "="*70)
    print(" Training Complete!")
    print("="*70)
    
    return evaluator, results


def generate_kaggle_submission():
    """Generate submission with probabilities for Kaggle ROC-AUC scoring."""
    import joblib
    
    print("Loading saved LightGBM model...")
    data = joblib.load('outputs/models/lightgbm.joblib')
    model = data['model']  # Extract actual model from saved dict
    
    print("Loading and preprocessing test data...")
    _, _, _, _, X_test, test_ids, _ = load_and_preprocess()
    
    print("Generating probability predictions...")
    probs = model.predict_proba(X_test)[:, 1]
    
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': probs
    })
    submission.to_csv('outputs/submission_kaggle.csv', index=False)
    
    print("Saved to outputs/submission_kaggle.csv")
    print(f"Mean predicted probability: {probs.mean():.4f}")
    print(f"First 5 rows:\n{submission.head()}")


def main():
    parser = argparse.ArgumentParser(description='Train diabetes prediction models')
    parser.add_argument('--no-tune', action='store_true', 
                       help='Skip hyperparameter tuning (faster but worse)')
    parser.add_argument('--skip-svm', action='store_true',
                       help='Skip SVM training (slow on large datasets)')
    parser.add_argument('--skip-nn', action='store_true',
                       help='Skip neural network training')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: skip tuning, SVM, and use fewer epochs')
    parser.add_argument('--kaggle', action='store_true',
                       help='Generate Kaggle submission with probabilities (requires trained model)')
    
    args = parser.parse_args()
    
    if args.kaggle:
        generate_kaggle_submission()
        return
    
    tune = not args.no_tune and not args.quick
    skip_svm = args.skip_svm or args.quick
    skip_nn = args.skip_nn
    
    train_and_evaluate_all(
        tune_hyperparams=tune,
        skip_svm=skip_svm,
        skip_nn=skip_nn
    )


if __name__ == "__main__":
    main()
