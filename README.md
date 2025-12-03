# Math 7243: ML Theory 1 - Diabetes Prediction

Binary classification project predicting diabetes from health features.

## Authors

- [Shresht Bhowmick](https://github.com/tetraslam)
- [Xiaole Su](https://github.com/suxls)
- [Mouad Tiahi](https://github.com/muuseotia)
- [Colin Johnson](https://github.com/Colont)

## Results

**Winner: LightGBM**.
- Validation ROC-AUC: **0.724**
- Kaggle Public Leaderboard: **0.696** (20% test set)

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Logistic Regression (ElasticNet) | 66.5% | 0.763 | 0.695 |
| Random Forest | 65.1% | 0.710 | 0.701 |
| **LightGBM** | 65.3% | 0.691 | **0.724** |
| Neural Network (MLP) | 62.4% | 0.656 | 0.696 |

## Models

| Model | Source | Description |
|-------|--------|-------------|
| Logistic Regression (ElasticNet) | Syllabus (Week 2-4) | Linear model with L1+L2 regularization |
| Random Forest | Syllabus (Week 8) | Ensemble of decision trees with bagging |
| LightGBM | **Not in syllabus** | Gradient boosting, SOTA for tabular data |
| Neural Network (MLP) | Syllabus (Week 9) | Deep learning with batch norm and dropout |

SVM excluded due to O(n^2) complexity on 700k samples (though we did try it out).

## Quick Start

```bash
# Install dependencies
uv sync

# Quick mode (recommended for iteration)
uv run python main.py --quick

# Full training with hyperparameter tuning
uv run python main.py

# Generate Kaggle submission (uses saved LightGBM model)
uv run python main.py --kaggle

# Train on Modal (cloud)
modal run train_modal.py
```

## Structure

```
mltheory-project/
├── data/                 # train.csv (700k), test.csv (300k)
├── models/               # Model implementations
├── outputs/
│   ├── figures/          # ROC curves, confusion matrices, etc.
│   ├── models/           # Trained model files (.joblib, .pt)
│   ├── model_comparison.csv
│   └── submission_kaggle.csv
├── main.py               # Local training script
├── train_modal.py        # Cloud training script using Modal
└── info.local.md         # Detailed analysis and insights
```

## Key Findings

1. **Gradient boosting wins** - LightGBM beats all syllabus models on ROC-AUC
2. **Deep learning underperforms** - MLP struggles on tabular data (no spatial/sequential structure)
3. **~65% accuracy ceiling** - Feature set limits achievable accuracy
4. **Class imbalance** - 62% diabetic, handled via class weights
