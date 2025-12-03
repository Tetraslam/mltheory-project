"""
Evaluation utilities for model comparison.
Handles metrics computation, visualization, and reporting.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def evaluate(
        self, 
        model_name: str,
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> dict:
        """
        Compute all metrics for a model.
        y_prob should be probabilities for the positive class.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        return metrics
    
    def print_report(self, model_name: str):
        """Print detailed classification report for a model."""
        if model_name not in self.results:
            print(f"No results found for {model_name}")
            return
            
        result = self.results[model_name]
        metrics = result['metrics']
        
        print(f"\n{'='*60}")
        print(f" {model_name} - Evaluation Report")
        print(f"{'='*60}")
        print(f"\nAccuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1 Score:     {metrics['f1']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC:      {metrics['roc_auc']:.4f}")
            print(f"Avg Precision:{metrics['avg_precision']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negatives']:,}  FP: {metrics['false_positives']:,}")
        print(f"  FN: {metrics['false_negatives']:,}  TP: {metrics['true_positives']:,}")
        print(f"{'='*60}\n")
    
    def plot_confusion_matrix(self, model_name: str, save: bool = True):
        """Plot confusion matrix for a model."""
        if model_name not in self.results:
            return
            
        result = self.results[model_name]
        cm = np.array(result['metrics']['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            ax=ax
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / f"{model_name.lower().replace(' ', '_')}_confusion.png", dpi=150)
        plt.close()
    
    def plot_roc_curves(self, save: bool = True):
        """Plot ROC curves for all models with probabilities."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            if result['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(result['y_true'], result['y_prob'])
                auc = result['metrics']['roc_auc']
                ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})", linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "roc_curves_comparison.png", dpi=150)
        plt.close()
    
    def plot_precision_recall_curves(self, save: bool = True):
        """Plot Precision-Recall curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            if result['y_prob'] is not None:
                precision, recall, _ = precision_recall_curve(result['y_true'], result['y_prob'])
                ap = result['metrics']['avg_precision']
                ax.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})", linewidth=2)
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "pr_curves_comparison.png", dpi=150)
        plt.close()
    
    def plot_metrics_comparison(self, save: bool = True):
        """Bar chart comparing key metrics across all models."""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        data = []
        for model_name, result in self.results.items():
            for metric in metrics_to_plot:
                if metric in result['metrics']:
                    data.append({
                        'Model': model_name,
                        'Metric': metric.upper().replace('_', ' '),
                        'Value': result['metrics'][metric]
                    })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create grouped bar chart
        models = list(self.results.keys())
        x = np.arange(len(metrics_to_plot))
        width = 0.15
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            values = [model_data[model_data['Metric'] == m.upper().replace('_', ' ')]['Value'].values[0] 
                     if len(model_data[model_data['Metric'] == m.upper().replace('_', ' ')]) > 0 else 0 
                     for m in metrics_to_plot]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics_to_plot])
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "metrics_comparison.png", dpi=150)
        plt.close()
    
    def get_summary_table(self) -> pd.DataFrame:
        """Return a summary DataFrame of all model metrics."""
        rows = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update({k: v for k, v in result['metrics'].items() 
                       if not isinstance(v, (list, np.ndarray))})
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.set_index('Model')
            # Reorder columns for better readability
            cols_order = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 
                         'roc_auc', 'avg_precision', 'true_positives', 'true_negatives',
                         'false_positives', 'false_negatives']
            cols_order = [c for c in cols_order if c in df.columns]
            df = df[cols_order]
        
        return df
    
    def save_results(self):
        """Save all results to files."""
        # Save summary table
        summary = self.get_summary_table()
        summary.to_csv(self.output_dir / "model_comparison.csv")
        
        # Save detailed metrics as JSON (excluding numpy arrays)
        json_results = {}
        for model_name, result in self.results.items():
            json_results[model_name] = {
                k: v for k, v in result['metrics'].items()
            }
        
        with open(self.output_dir / "detailed_metrics.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}/")
    
    def generate_all_plots(self):
        """Generate all comparison plots."""
        # Individual confusion matrices
        for model_name in self.results:
            self.plot_confusion_matrix(model_name)
        
        # Comparison plots
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_metrics_comparison()
        
        print(f"Plots saved to {self.figures_dir}/")

