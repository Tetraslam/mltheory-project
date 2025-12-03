"""
Modal training script for diabetes prediction models.
Runs training on cloud infrastructure to save your laptop.

Usage:
    modal run train_modal.py
"""

import modal

# Define the Modal app
app = modal.App("diabetes-prediction")

# Volume to persist outputs
output_volume = modal.Volume.from_name("diabetes-outputs", create_if_missing=True)

# Create image with all dependencies and local files
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "scikit-learn>=1.5.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "lightgbm>=4.0.0",
        "torch>=2.0.0",
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
    )
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("models", remote_path="/root/models")
)


@app.function(
    image=image,
    volumes={"/root/outputs": output_volume},
    timeout=36000,  # 10 hours max
    cpu=16,
    memory=32768,
    gpu="T4",
)
def train_all_models(skip_svm: bool = True, skip_nn: bool = False, tune: bool = False):
    """Train all models on Modal infrastructure."""
    import sys
    sys.path.insert(0, "/root")
    
    import time
    import warnings
    from pathlib import Path

    import matplotlib
    import pandas as pd
    matplotlib.use('Agg')  # Non-interactive backend
    
    warnings.filterwarnings('ignore')
    
    # Import our modules
    from models.evaluation import ModelEvaluator
    from models.lightgbm_model import LightGBMClassifier
    from models.logistic_regression import ElasticNetLogisticRegression
    from models.neural_network import NeuralNetworkClassifier
    from models.preprocessing import DataProcessor
    from models.random_forest import RandomForest
    
    if not skip_svm:
        from models.svm import RBFSupportVectorMachine
    
    print("=" * 70)
    print(" Diabetes Prediction - Modal Cloud Training")
    print("=" * 70)
    
    # Setup paths
    data_dir = Path("/root/data")
    output_dir = Path("/root/outputs")
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print("\n[1/7] Loading and preprocessing data...")
    processor = DataProcessor(data_dir=str(data_dir))
    train_df, test_df = processor.load_data()
    
    X, y = processor.fit_transform(train_df)
    X_train, X_val, y_train, y_val = processor.get_train_val_split(X, y)
    X_test = processor.transform(test_df)
    test_ids = processor.get_test_ids(test_df)
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    print(f"Features: {X_train.shape[1]}")
    
    evaluator = ModelEvaluator(output_dir=str(output_dir))
    results = {}
    
    # Model 1: Logistic Regression
    print("\n[2/7] Training Logistic Regression with ElasticNet...")
    start = time.time()
    lr_model = ElasticNetLogisticRegression()
    lr_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=tune)
    
    y_pred = lr_model.predict(X_val)
    y_prob = lr_model.predict_proba(X_val)
    evaluator.evaluate("Logistic Regression (ElasticNet)", y_val, y_pred, y_prob)
    evaluator.print_report("Logistic Regression (ElasticNet)")
    lr_model.save(str(models_dir / "logistic_regression.joblib"))
    results['Logistic Regression'] = {'time': time.time() - start}
    
    # Model 2: SVM (optional)
    if not skip_svm:
        print("\n[3/7] Training SVM with RBF Kernel...")
        start = time.time()
        svm_model = RBFSupportVectorMachine()
        svm_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=tune)
        
        y_pred = svm_model.predict(X_val)
        y_prob = svm_model.predict_proba(X_val)
        evaluator.evaluate("SVM (RBF Kernel)", y_val, y_pred, y_prob)
        evaluator.print_report("SVM (RBF Kernel)")
        svm_model.save(str(models_dir / "svm.joblib"))
        results['SVM'] = {'time': time.time() - start}
    else:
        print("\n[3/7] Skipping SVM")
    
    # Model 3: Random Forest
    print("\n[4/7] Training Random Forest...")
    start = time.time()
    rf_model = RandomForest()
    rf_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=tune)
    
    y_pred = rf_model.predict(X_val)
    y_prob = rf_model.predict_proba(X_val)
    evaluator.evaluate("Random Forest", y_val, y_pred, y_prob)
    evaluator.print_report("Random Forest")
    rf_model.save(str(models_dir / "random_forest.joblib"))
    results['Random Forest'] = {'time': time.time() - start}
    
    # Model 4: LightGBM
    print("\n[5/7] Training LightGBM...")
    start = time.time()
    lgb_model = LightGBMClassifier()
    lgb_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=tune)
    
    y_pred = lgb_model.predict(X_val)
    y_prob = lgb_model.predict_proba(X_val)
    evaluator.evaluate("LightGBM", y_val, y_pred, y_prob)
    evaluator.print_report("LightGBM")
    lgb_model.save(str(models_dir / "lightgbm.joblib"))
    results['LightGBM'] = {'time': time.time() - start, 'model': lgb_model}
    
    # Model 5: Neural Network
    if not skip_nn:
        print("\n[6/7] Training Neural Network...")
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
        nn_model.save(str(models_dir / "neural_network.pt"))
        results['Neural Network'] = {'time': time.time() - start}
    else:
        print("\n[6/7] Skipping Neural Network")
    
    # Generate plots and save results
    print("\n[7/7] Generating reports...")
    evaluator.generate_all_plots()
    evaluator.save_results()
    
    # Summary
    summary = evaluator.get_summary_table()
    print("\n" + "=" * 70)
    print(" Model Comparison Summary")
    print("=" * 70)
    print(summary.to_string())
    
    print("\n" + "-" * 40)
    print(" Training Times")
    print("-" * 40)
    for name, data in results.items():
        print(f"{name}: {data['time']:.1f}s")
    
    # Generate test predictions with LightGBM
    print("\nGenerating test predictions with LightGBM...")
    test_probs = lgb_model.predict_proba(X_test)
    test_preds = (test_probs >= 0.5).astype(int)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': test_preds
    })
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"Submission saved. Predicted positive rate: {test_preds.mean() * 100:.2f}%")
    
    # Commit the volume to persist outputs
    output_volume.commit()
    
    print("\n" + "=" * 70)
    print(" Training Complete! Outputs saved to Modal volume.")
    print("=" * 70)
    
    return summary.to_dict()


@app.function(
    image=image,
    volumes={"/root/outputs": output_volume},
)
def download_results():
    """Download results from Modal volume."""
    from pathlib import Path
    
    output_dir = Path("/root/outputs")
    files = []
    
    for f in output_dir.rglob("*"):
        if f.is_file():
            files.append(str(f.relative_to(output_dir)))
    
    return files


@app.local_entrypoint()
def main():
    """Run training on Modal."""
    print("Starting training on Modal cloud...")
    print("This will take 10-20 minutes.\n")
    
    # Run training (skip tuning - Modal sandbox doesn't handle sklearn multiprocessing well)
    results = train_all_models.remote(skip_svm=True, skip_nn=False, tune=False)
    
    print("\n\nTraining complete! Results:")
    print(results)
    
    # List output files
    print("\nOutput files generated:")
    files = download_results.remote()
    for f in files:
        print(f"  - {f}")
    
    print("\nTo download results, run:")
    print("  modal volume get diabetes-outputs outputs/")
