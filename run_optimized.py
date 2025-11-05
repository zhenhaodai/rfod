"""
Optimized RFOD training and inference for 240-core CPU
"""
from train_and_infer import train_and_infer

if __name__ == "__main__":
    print("RFOD Training and Inference")
    print("CPU cores: 240")
    print("Optimization: Multi-threading + Large batch processing\n")

    results = train_and_infer(
        train_csv="data/processes_train.csv",
        valid_csv=None,
        test_csv="data/processes_test.csv",
        output_path="result/submission.csv",

        batch_size=100000,
        alpha=0.005,
        beta=0.7,
        n_estimators=80,
        max_depth=20,
        random_state=42,
        n_jobs=240,

        process_args=False,
        drop_labelled_anomalies=False,

        use_threshold=False,
        normalize_method="minmax",
        out_dir="model",
        verbose=True
    )

    print(f"\nModel saved: {results.get('model_path')}")
    if 'output_path' in results:
        print(f"Predictions saved: {results.get('output_path')}")
        print(f"Test samples: {results.get('n_test')}")

    if results.get('metrics'):
        print("\nValidation metrics:")
        metrics = results['metrics']
        for key in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            if key in metrics and metrics[key] is not None:
                print(f"  {key}: {metrics[key]:.4f}")
