"""
RFOD training and inference pipeline
"""
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict
from tqdm import tqdm

from rfod import (
    RFOD,
    _safe_clean_csv,
    _select_and_align_features,
    REQ_FEATURES,
    _compute_binary_metrics,
    _scan_thresholds_from_scores,
    _stable_param_signature
)
from sklearn.metrics import f1_score, confusion_matrix


def train_and_infer(
    train_csv: str,
    valid_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    output_path: str = "result/prediction.csv",
    batch_size: int = 50000,
    alpha: float = 0.02,
    beta: float = 0.7,
    n_estimators: int = 30,
    max_depth: int = 6,
    random_state: int = 42,
    n_jobs: int = -1,
    process_args: bool = False,
    drop_labelled_anomalies: bool = False,
    threshold: Optional[float] = None,
    threshold_percentile: int = 95,
    threshold_strategy: str = "train_quantile",
    use_threshold: bool = False,
    normalize_method: str = "minmax",
    out_dir: str = "model",
    verbose: bool = True
) -> Dict:

    results = {}

    if verbose:
        print("\n" + "="*60)
        print("STEP 1: Training Model")
        print("="*60)

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train file not found: {train_csv}")

    df_train = _safe_clean_csv(train_csv, process_args=process_args)
    if df_train.empty:
        raise ValueError(f"Failed to load train data: {train_csv}")

    if drop_labelled_anomalies and "target" in df_train.columns:
        before = len(df_train)
        df_train = df_train[df_train["target"].astype(str) != "1"]
        if verbose:
            print(f"Removed {before - len(df_train)} labeled anomalies from training set")

    X_train = _select_and_align_features(df_train, REQ_FEATURES)
    if X_train.empty:
        raise ValueError("Training data is empty")

    if verbose:
        print(f"Train samples: {len(X_train)}, Features: {len(REQ_FEATURES)}")

    params = {
        'alpha': alpha,
        'beta': beta,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'random_state': random_state,
        'n_jobs': n_jobs,
        'verbose': verbose
    }

    rfod = RFOD(**params)
    rfod.fit(X_train)

    train_scores = rfod.predict(X_train, batch_size=batch_size)

    if verbose:
        print("\n" + "="*60)
        print("STEP 2: Determining Threshold")
        print("="*60)

    thr = None
    thr_src = None
    metrics = {}

    if threshold is not None:
        thr = float(threshold)
        thr_src = "user_provided"
        if verbose:
            print(f"Using user-specified threshold: {thr:.6f}")

    elif valid_csv is not None:
        if not os.path.exists(valid_csv):
            raise FileNotFoundError(f"Valid file not found: {valid_csv}")

        df_valid = _safe_clean_csv(valid_csv, process_args=process_args)
        if df_valid.empty:
            raise ValueError(f"Failed to load valid data: {valid_csv}")

        if "target" not in df_valid.columns:
            if verbose:
                print("Warning: no target column in validation set, using train quantile")
            thr = float(np.percentile(train_scores, threshold_percentile))
            thr_src = f"train_quantile_{threshold_percentile}"
        else:
            X_valid = _select_and_align_features(df_valid, REQ_FEATURES)
            y_true = df_valid["target"].astype(int).values

            if verbose:
                print(f"Valid samples: {len(X_valid)}, Anomalies: {y_true.sum()} ({y_true.sum()/len(y_true)*100:.2f}%)")

            valid_scores = rfod.predict(X_valid, batch_size=batch_size)

            if threshold_strategy == "train_quantile":
                thr = float(np.percentile(train_scores, threshold_percentile))
                thr_src = f"train_quantile_{threshold_percentile}"

            elif threshold_strategy == "valid_best_f1":
                candidates = _scan_thresholds_from_scores(valid_scores, n_points=256)
                best_f1, thr = -1.0, candidates[0]
                for t in tqdm(candidates, desc="Searching best F1 threshold", disable=not verbose):
                    yp = (valid_scores > t).astype(int)
                    f1v = f1_score(y_true, yp, zero_division=0)
                    if f1v > best_f1:
                        best_f1, thr = f1v, float(t)
                thr_src = "valid_best_f1"

            elif threshold_strategy == "valid_best_j":
                candidates = _scan_thresholds_from_scores(valid_scores, n_points=256)
                best_j, thr = -1.0, candidates[0]
                for t in tqdm(candidates, desc="Searching best Youden's J threshold", disable=not verbose):
                    yp = (valid_scores > t).astype(int)
                    cm = confusion_matrix(y_true, yp, labels=[0, 1])
                    if cm.size == 4:
                        tn, fp, fn, tp = cm.ravel()
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        j = tpr - fpr
                    else:
                        j = 0.0
                    if j > best_j:
                        best_j, thr = j, float(t)
                thr_src = "valid_best_j"

            else:
                raise ValueError(f"Unknown threshold_strategy: {threshold_strategy}")

            y_pred = (valid_scores > thr).astype(int)
            metrics = _compute_binary_metrics(y_true, valid_scores, y_pred)

            if verbose:
                print(f"\nThreshold: {thr:.6f} ({thr_src})")
                print(f"Validation metrics:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}, Balanced: {metrics['balanced_accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
                if metrics.get('roc_auc'):
                    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")

    else:
        thr = float(np.percentile(train_scores, threshold_percentile))
        thr_src = f"train_quantile_{threshold_percentile}"
        if verbose:
            print(f"Threshold: {thr:.6f} ({thr_src})")

    if verbose:
        print("\n" + "="*60)
        print("STEP 3: Saving Model")
        print("="*60)

    os.makedirs(out_dir, exist_ok=True)

    model_params = {
        'alpha': alpha,
        'beta': beta,
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }
    sig, ord_params, readable = _stable_param_signature(model_params)

    model_path = os.path.join(out_dir, f"rfod_{sig}.pkl")

    model_data = {
        'model': rfod,
        'threshold': thr,
        'threshold_source': thr_src,
        'params': params,
        'metrics': metrics,
        'saved_at': datetime.now().isoformat(timespec="seconds"),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    if verbose:
        print(f"Model saved: {model_path}")

    results['model_path'] = model_path
    results['threshold'] = thr
    results['threshold_source'] = thr_src
    results['params'] = params
    results['metrics'] = metrics

    if test_csv is not None:
        if verbose:
            print("\n" + "="*60)
            print("STEP 4: Testing Inference")
            print("="*60)

        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test file not found: {test_csv}")

        df_test = _safe_clean_csv(test_csv, process_args=process_args)

        if 'Id' not in df_test.columns:
            raise ValueError("Test set must contain 'Id' column")

        if verbose:
            print(f"Test samples: {len(df_test)}")

        X_test = _select_and_align_features(df_test, REQ_FEATURES)

        if X_test.empty:
            raise ValueError("Test data is empty")

        test_scores = rfod.predict(X_test, clip_scores=False, batch_size=batch_size)

        original_min = test_scores.min()
        original_max = test_scores.max()

        if verbose:
            print(f"Raw score range: [{original_min:.6f}, {original_max:.6f}]")

        if normalize_method == "minmax":
            score_range = original_max - original_min
            if score_range > 1e-10:
                normalized_scores = (test_scores - original_min) / score_range
            else:
                normalized_scores = np.zeros_like(test_scores)

        elif normalize_method == "robust":
            q25, q50, q75 = np.percentile(test_scores, [25, 50, 75])
            iqr = q75 - q25
            if iqr > 1e-10:
                normalized_scores = (test_scores - q50) / iqr
                normalized_scores = (normalized_scores - normalized_scores.min()) / \
                                  (normalized_scores.max() - normalized_scores.min())
            else:
                normalized_scores = np.zeros_like(test_scores)

        elif normalize_method == "clip":
            normalized_scores = np.clip(test_scores, 0.0, 1.0)

        elif normalize_method == "none":
            normalized_scores = test_scores

        else:
            raise ValueError(f"Unknown normalize_method: {normalize_method}")

        if use_threshold and thr is not None:
            predictions = (normalized_scores > thr).astype(int)
            if verbose:
                print(f"Using threshold {thr:.6f}, predicted anomalies: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")

            out_df = pd.DataFrame({
                'Id': df_test['Id'],
                'target': predictions
            })
            output_type = "binary"
        else:
            out_df = pd.DataFrame({
                'Id': df_test['Id'],
                'target': normalized_scores
            })
            output_type = f"scores ({normalize_method})"

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        out_df.to_csv(output_path, index=False)

        if verbose:
            print(f"Predictions saved: {output_path} ({output_type})")
            print(f"Output preview:\n{out_df.head(10)}")

        results['output_path'] = output_path
        results['output_type'] = output_type
        results['n_test'] = len(df_test)
        results['predictions'] = out_df

    if verbose:
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)

    return results
