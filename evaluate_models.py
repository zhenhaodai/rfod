"""
Model Evaluation Script
Evaluate all models in model/ folder on validation set
Metrics: Average Precision (primary), ROC-AUC (secondary)
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Dict, List, Tuple
import glob
import tempfile

# Import cleaning functions and RFOD class from rfod_complete
sys.path.insert(0, os.path.dirname(__file__))

# Import all necessary components for pickle deserialization
try:
    from rfod_complete import (
        clean_csv,
        _select_and_align_features,
        RFOD,
        filter_high_cardinality_features,
        convert_dtypes_for_training,
        extract_temporal_features
    )

    # Register RFOD in the global namespace for pickle compatibility
    # This helps pickle find the class when loading old models
    import rfod_complete
    sys.modules['__main__'].RFOD = RFOD

    # Also ensure rfod_complete is accessible
    sys.modules['rfod_complete'] = rfod_complete

except ImportError as e:
    print(f"❌ Failed to import from rfod_complete: {e}")
    print(f"   Make sure rfod_complete.py is in the same directory")
    sys.exit(1)


def load_and_clean_validation_data(val_csv: str, num_temporal_features: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and clean validation data with labels

    Args:
        val_csv: Path to validation CSV file
        num_temporal_features: Number of temporal features (0-3) - must match training

    Returns:
        df_cleaned: Cleaned DataFrame with features
        y: Target labels (0=normal, 1=anomaly)
    """
    # First, read raw data to extract labels
    df_raw = pd.read_csv(val_csv)

    if 'target' not in df_raw.columns:
        raise ValueError("Validation set must contain 'target' column with labels")

    # Save labels before cleaning
    y = df_raw['target'].copy()

    # Clean validation data using the same pipeline as training
    tmp_out = os.path.join(tempfile.gettempdir(), f"cleaned_validation.csv")
    df_cleaned = clean_csv(
        val_csv,
        tmp_out,
        num_temporal_features=num_temporal_features,
        save=False
    )

    # Re-attach target column (it might have been processed)
    if 'target' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['target'])

    return df_cleaned, y


class RenameUnpickler(pickle.Unpickler):
    """
    Custom unpickler to handle module path changes
    Redirects old module paths to current rfod_complete module
    """
    def find_class(self, module, name):
        # Redirect various possible old module names to rfod_complete
        if module == '__main__' or module.startswith('rfod'):
            try:
                return getattr(rfod_complete, name)
            except AttributeError:
                pass
        return super().find_class(module, name)


def load_model_safe(model_path: str):
    """
    Safely load a pickled model with module path handling

    Args:
        model_path: Path to model pickle file

    Returns:
        model_data dictionary or raises exception
    """
    with open(model_path, 'rb') as f:
        return RenameUnpickler(f).load()


def evaluate_model(model_path: str, val_csv: str, batch_size: int = 50000) -> Dict:
    """
    Evaluate a single model on validation set

    Args:
        model_path: Path to model pickle file
        val_csv: Path to validation CSV file
        batch_size: Batch size for prediction

    Returns:
        Dictionary with model info and scores
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {os.path.basename(model_path)}")
    print(f"{'='*70}")

    # Load model using custom unpickler for compatibility
    try:
        model_data = load_model_safe(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

    rfod = model_data.get('model')
    feature_cols = model_data.get('feature_cols')
    num_temporal_features = model_data.get('num_temporal_features', 0)

    if rfod is None or feature_cols is None:
        print(f"❌ Invalid model file (missing model or feature_cols)")
        return None

    print(f"Model info:")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Temporal features: {num_temporal_features}")
    if model_data.get('params'):
        params = model_data['params']
        print(f"  - n_estimators: {params.get('n_estimators', 'N/A')}")
        print(f"  - max_depth: {params.get('max_depth', 'N/A')}")

    # Clean validation data using the same temporal features as training
    print(f"\nCleaning validation data (temporal_features={num_temporal_features})...")
    try:
        df_val_cleaned, y_val = load_and_clean_validation_data(val_csv, num_temporal_features)
        print(f"✅ Validation data cleaned: {len(df_val_cleaned)} samples")
    except Exception as e:
        print(f"❌ Failed to clean validation data: {e}")
        return None

    # Align features with model's expected features
    print(f"Aligning features...")
    X_val = _select_and_align_features(df_val_cleaned, feature_cols)

    print(f"  - Validation samples: {len(X_val)}")
    print(f"  - Features aligned: {len(feature_cols)}")

    # Predict anomaly scores
    print("Predicting...")
    try:
        scores = rfod.predict(X_test, clip_scores=False, batch_size=batch_size)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

    # Normalize scores to [0, 1]
    score_min = scores.min()
    score_max = scores.max()
    if score_max - score_min > 1e-10:
        scores_norm = (scores - score_min) / (score_max - score_min)
    else:
        scores_norm = np.zeros_like(scores)

    print(f"  - Score range: [{score_min:.6f}, {score_max:.6f}]")

    # Calculate metrics
    try:
        avg_precision = average_precision_score(y_val, scores_norm)
        roc_auc = roc_auc_score(y_val, scores_norm)
    except Exception as e:
        print(f"❌ Metric calculation failed: {e}")
        return None

    print(f"\n📊 Results:")
    print(f"  ⭐ Average Precision: {avg_precision:.6f}")
    print(f"  📈 ROC-AUC:          {roc_auc:.6f}")

    # Additional statistics
    n_anomalies = int(y_val.sum())
    n_normal = len(y_val) - n_anomalies
    anomaly_rate = n_anomalies / len(y_val) * 100

    print(f"\n📋 Dataset info:")
    print(f"  - Total samples:  {len(y_val)}")
    print(f"  - Anomalies:      {n_anomalies} ({anomaly_rate:.2f}%)")
    print(f"  - Normal:         {n_normal} ({100-anomaly_rate:.2f}%)")

    return {
        'model_path': model_path,
        'model_name': os.path.basename(model_path),
        'avg_precision': avg_precision,
        'roc_auc': roc_auc,
        'n_features': len(feature_cols),
        'num_temporal_features': num_temporal_features,
        'params': model_data.get('params', {}),
        'saved_at': model_data.get('saved_at', 'Unknown')
    }


def evaluate_all_models(model_dir: str, val_csv: str, batch_size: int = 50000):
    """
    Evaluate all models in model directory
    """
    print("="*70)
    print("MODEL EVALUATION ON VALIDATION SET")
    print("="*70)
    print(f"Validation data: {val_csv}")
    print(f"Model directory: {model_dir}")
    print(f"Metrics: Average Precision (primary), ROC-AUC (secondary)")
    print("="*70)

    # Find all model files
    model_files = glob.glob(os.path.join(model_dir, "rfod_*.pkl"))

    if not model_files:
        print(f"\n❌ No model files found in {model_dir}")
        return

    print(f"\nFound {len(model_files)} model(s) to evaluate")

    # Check validation file exists
    if not os.path.exists(val_csv):
        print(f"❌ Validation file not found: {val_csv}")
        return

    # Evaluate each model (each will clean validation data independently)
    results = []
    for model_path in model_files:
        result = evaluate_model(model_path, val_csv, batch_size)
        if result is not None:
            results.append(result)

    if not results:
        print(f"\n❌ No models were successfully evaluated")
        return

    # Sort results by Average Precision (primary), ROC-AUC (secondary)
    results.sort(key=lambda x: (-x['avg_precision'], -x['roc_auc']))

    # Print summary table
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Rank':<6} {'Model':<35} {'Avg Prec':<12} {'ROC-AUC':<12} {'Features':<10}")
    print("-"*70)

    for i, result in enumerate(results, 1):
        model_name = result['model_name']
        if len(model_name) > 33:
            model_name = model_name[:30] + "..."

        print(f"{i:<6} {model_name:<35} "
              f"{result['avg_precision']:<12.6f} "
              f"{result['roc_auc']:<12.6f} "
              f"{result['n_features']:<10}")

    print("="*70)

    # Best model details
    best = results[0]
    print(f"\n🏆 BEST MODEL:")
    print(f"   Model: {best['model_name']}")
    print(f"   ⭐ Average Precision: {best['avg_precision']:.6f}")
    print(f"   📈 ROC-AUC:          {best['roc_auc']:.6f}")
    print(f"   Features:           {best['n_features']}")
    print(f"   Temporal features:  {best['num_temporal_features']}")
    print(f"   Trained at:         {best['saved_at']}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(model_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n💾 Results saved to: {results_csv}")

    return results


if __name__ == "__main__":
    # Configuration
    MODEL_DIR = "model"
    VAL_CSV = "data/processes_valid.csv"
    BATCH_SIZE = 50000

    # Check if files exist
    if not os.path.exists(VAL_CSV):
        print(f"❌ Validation file not found: {VAL_CSV}")
        print(f"   Please ensure the file exists with 'target' column for labels")
        exit(1)

    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory not found: {MODEL_DIR}")
        exit(1)

    # Run evaluation
    results = evaluate_all_models(MODEL_DIR, VAL_CSV, BATCH_SIZE)

    print("\n✅ Evaluation complete!")
