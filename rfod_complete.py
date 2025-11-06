"""
RFOD Complete: All-in-one implementation
Includes: Data processing, RFOD model, Training & Inference pipeline
"""
import os
import ast
import json
import tempfile
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Tuple, Optional, Union, Any, Set
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.metrics import (
    roc_auc_score, r2_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, fbeta_score,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import hashlib
from datetime import datetime
from collections import OrderedDict


# ============================================================================
# DATA PROCESSING
# ============================================================================

def parse_list_field(value: Any) -> List:
    """Parse list-like string fields safely"""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return []


def extract_arg_features(df: pd.DataFrame, args_col: str = "args") -> pd.DataFrame:
    """Flatten args field (old method - causes dimension explosion)"""
    if args_col not in df.columns:
        print(f"Warning: '{args_col}' column not found, skipping args extraction")
        return df

    all_feature_names: Set[str] = set()
    feature_types: Dict[str, Set[str]] = {}

    for args_str in df[args_col]:
        for arg in parse_list_field(args_str):
            if isinstance(arg, dict) and "name" in arg:
                name = arg["name"]
                all_feature_names.add(name)
                t = arg.get("type", "unknown")
                feature_types.setdefault(name, set()).add(t)

    all_feature_names = sorted(list(all_feature_names))

    flattened_features = []
    for args_str in df[args_col]:
        feature_map = {name: None for name in all_feature_names}
        for arg in parse_list_field(args_str):
            if isinstance(arg, dict) and "name" in arg and "value" in arg:
                feature_map[arg["name"]] = arg["value"]
        flattened_features.append(feature_map)

    args_df = pd.DataFrame(flattened_features)
    df = pd.concat([df.reset_index(drop=True), args_df.reset_index(drop=True)], axis=1)

    print(f"Args flattened: {len(all_feature_names)} new features")
    return df


def extract_args_topk_stats(df: pd.DataFrame, args_col: str = "args", top_k: int = 5) -> pd.DataFrame:
    """
    Extract Top-K most frequent args + statistics (recommended by research)
    Based on: "On Improving Deep Learning Trace Analysis with System Call Arguments" (IEEE 2021)

    Strategy:
    1. Find Top-K most frequent argument names
    2. Extract their values as features
    3. Add statistical features (counts by type)
    """
    if args_col not in df.columns:
        print(f"Warning: '{args_col}' column not found, skipping args extraction")
        return df

    # Step 1: Count argument name frequencies
    arg_freq = {}
    for args_str in df[args_col]:
        for arg in parse_list_field(args_str):
            if isinstance(arg, dict) and "name" in arg:
                name = arg.get("name")
                arg_freq[name] = arg_freq.get(name, 0) + 1

    # Step 2: Select Top-K
    top_args = sorted(arg_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_names = [name for name, count in top_args]

    if top_names:
        print(f"Top-{top_k} args: {top_names}")

    # Step 3: Extract features for each row
    features_list = []
    for args_str in df[args_col]:
        parsed = parse_list_field(args_str)

        # Top-K argument values
        topk_features = {f'arg_{name}': None for name in top_names}
        for arg in parsed:
            if isinstance(arg, dict) and arg.get("name") in top_names:
                topk_features[f'arg_{arg["name"]}'] = arg.get("value")

        # Statistical features (based on research categories)
        stats_features = {
            'args_count': len(parsed),
            'args_int_count': sum(1 for a in parsed if isinstance(a, dict) and a.get('type') == 'int'),
            'args_str_count': sum(1 for a in parsed if isinstance(a, dict) and a.get('type') == 'string'),
            'args_ptr_count': sum(1 for a in parsed if isinstance(a, dict) and ('ptr' in str(a.get('type', '')) or '*' in str(a.get('type', '')))),
            'args_unique_names': len(set(a.get('name') for a in parsed if isinstance(a, dict) and 'name' in a))
        }

        features_list.append({**topk_features, **stats_features})

    args_df = pd.DataFrame(features_list)
    df = pd.concat([df.reset_index(drop=True), args_df.reset_index(drop=True)], axis=1)

    print(f"Args processed: {len(top_names)} Top-K features + 5 statistics = {len(args_df.columns)} new features")
    return df


def convert_dtypes_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dtypes:
    - Numeric: timestamp, argsNum, stack_depth, args_*count features
    - Categorical: all others (except 'args' column if still present)
    """
    # Base numeric columns
    numeric_cols = ["timestamp", "argsNum", "stack_depth"]

    # Add args statistics features (these are all numeric)
    args_stat_cols = [col for col in df.columns if col.startswith('args_') and
                      col in ['args_count', 'args_int_count', 'args_str_count',
                              'args_ptr_count', 'args_unique_names']]
    numeric_cols.extend(args_stat_cols)

    # Convert types
    for col in df.columns:
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col != "args":
            # arg_* features (Top-K values) and other features are categorical
            df[col] = df[col].astype(str)

    print(f"Type conversion: {len(numeric_cols)} numeric features, "
          f"{len([c for c in df.columns if c not in numeric_cols and c != 'args'])} categorical features")
    return df


def clean_csv(input_path: str, output_path: str, process_args: Union[bool, str] = "topk", save: bool = True):
    """
    Main data cleaning function

    Args:
        process_args: How to process args column
            - False: Drop args completely (no processing)
            - True or "full": Full expansion (causes 100+ features - NOT RECOMMENDED)
            - "topk" or "smart": Top-K + statistics (RECOMMENDED, 10-15 features)
    """
    print(f"Processing: {input_path}")
    df = pd.read_csv(input_path)

    if "stackAddresses" in df.columns:
        df["stackAddresses"] = df["stackAddresses"].apply(parse_list_field)
        df["stack_depth"] = df["stackAddresses"].apply(len)
        print("stack_depth feature computed")

    drop_cols = ["threadId", "eventId", "stackAddresses"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    print(f"Dropped columns: {drop_cols}")

    args_col_present = "args" in df.columns

    if not args_col_present:
        if process_args:
            print("Warning: 'args' column not found")
    elif process_args == False:
        df = df.drop(columns=["args"], errors="ignore")
        print("Args dropped (process_args=False)")
    elif process_args in [True, "full"]:
        print("WARNING: Full args expansion may create 100+ sparse features!")
        df = extract_arg_features(df, args_col="args")
        df = df.drop(columns=["args"], errors="ignore")
    elif process_args in ["topk", "smart"]:
        df = extract_args_topk_stats(df, args_col="args", top_k=5)
        df = df.drop(columns=["args"], errors="ignore")
    else:
        raise ValueError(f"Unknown process_args value: {process_args}. Use False, True, 'full', 'topk', or 'smart'")

    if "processId" in df.columns and "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["timestamp"] = df.groupby("processId")["timestamp"].transform(lambda x: x - x.min())
        print("Timestamp normalized by processId")
    else:
        print("Warning: Missing processId or timestamp, skipping normalization")

    df = convert_dtypes_for_training(df)

    if save:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    return df


# ============================================================================
# RFOD MODEL
# ============================================================================

class RFOD:
    def __init__(
        self,
        alpha: float = 0.02,
        beta: float = 0.7,
        n_estimators: int = 30,
        max_depth: int = 6,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        self.alpha = alpha
        self.beta = beta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.forests_ = {}
        self.feature_types_ = {}
        self.quantiles_ = {}
        self.feature_names_ = []
        self.n_features_ = 0
        self.encoders_: Dict[str, LabelEncoder] = {}

    def _identify_feature_types(self, X: pd.DataFrame) -> Dict[int, str]:
        feature_types = {}
        for idx, col in enumerate(X.columns):
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_types[idx] = 'numeric'
            else:
                feature_types[idx] = 'categorical'
        return feature_types

    def _compute_quantiles(self, X: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
        quantiles = {}
        for idx, col in enumerate(X.columns):
            if self.feature_types_[idx] == 'numeric':
                q_low = X[col].quantile(self.alpha)
                q_high = X[col].quantile(1 - self.alpha)
                if q_high - q_low < 1e-10:
                    q_high = q_low + 1.0
                quantiles[idx] = (q_low, q_high)
        return quantiles

    def _fit_encoders(self, X: pd.DataFrame):
        self.encoders_ = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                series = X[col].astype(str).fillna("NaN_TOKEN")
                le.fit(series)
                self.encoders_[col] = le

    def _transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        for col, le in self.encoders_.items():
            if col in X_transformed.columns:
                series = X_transformed[col].astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~series.isin(le.classes_)
                series.loc[unseen_mask] = le.classes_[0]
                transformed_series = le.transform(series)
                transformed_series[unseen_mask] = -1
                X_transformed[col] = transformed_series
        return X_transformed

    def _train_feature_forest(self, X: pd.DataFrame, feature_idx: int):
        X_train_df = X.drop(X.columns[feature_idx], axis=1)
        y_train = X.iloc[:, feature_idx]
        X_train_encoded = self._transform_data(X_train_df)
        target_col_name = X.columns[feature_idx]

        if self.feature_types_[feature_idx] == 'categorical':
            if target_col_name in self.encoders_:
                y_train_series = y_train.astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~y_train_series.isin(self.encoders_[target_col_name].classes_)
                y_train_series.loc[unseen_mask] = self.encoders_[target_col_name].classes_[0]
                y_train_encoded = self.encoders_[target_col_name].transform(y_train_series)
                y_train_encoded[unseen_mask] = -1
                y_train = y_train_encoded

            forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                oob_score=True,
                bootstrap=True
            )
        else:
            y_train = y_train.fillna(y_train.mean())
            forest = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                oob_score=True,
                bootstrap=True
            )

        forest.fit(X_train_encoded, y_train)
        return forest

    def _prune_forest(self, forest, X: pd.DataFrame, feature_idx: int):
        X_train_df = X.drop(X.columns[feature_idx], axis=1)
        y_train = X.iloc[:, feature_idx]
        X_train_encoded = self._transform_data(X_train_df)
        target_col_name = X.columns[feature_idx]
        is_classifier = isinstance(forest, RandomForestClassifier)

        if is_classifier:
            if target_col_name in self.encoders_:
                y_train_series = y_train.astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~y_train_series.isin(self.encoders_[target_col_name].classes_)
                y_train_series.loc[unseen_mask] = self.encoders_[target_col_name].classes_[0]
                y_train_encoded = self.encoders_[target_col_name].transform(y_train_series)
                y_train_encoded[unseen_mask] = -1
                y_train = pd.Series(y_train_encoded, index=X_train_df.index)
        else:
            y_train = y_train.fillna(y_train.mean())

        n_samples = X_train_encoded.shape[0]
        if n_samples == 0:
            return forest

        if forest.max_samples is None:
            n_samples_bootstrap = n_samples
        elif isinstance(forest.max_samples, int):
            n_samples_bootstrap = forest.max_samples
        else:
            n_samples_bootstrap = int(forest.max_samples * n_samples)

        tree_scores = []
        for tree in forest.estimators_:
            try:
                oob_indices = _generate_unsampled_indices(tree.random_state, n_samples, n_samples_bootstrap)

                if len(oob_indices) == 0:
                    tree_scores.append(0.0)
                    continue

                X_oob = X_train_encoded.iloc[oob_indices]
                y_oob = y_train.iloc[oob_indices]

                if len(y_oob) == 0:
                    tree_scores.append(0.0)
                    continue

                if is_classifier:
                    if len(np.unique(y_oob)) <= 1:
                        tree_scores.append(0.0)
                        continue
                    y_pred_proba = tree.predict_proba(X_oob)
                    score = roc_auc_score(y_oob, y_pred_proba, multi_class='ovr', average='macro', labels=forest.classes_)
                else:
                    y_pred = tree.predict(X_oob)
                    score = r2_score(y_oob, y_pred)
                    score = max(0, score)
            except Exception:
                score = 0.0
            tree_scores.append(score)

        n_trees_keep = max(1, int(self.beta * len(forest.estimators_)))
        top_indices = np.argsort(tree_scores)[-n_trees_keep:]

        if is_classifier:
            pruned = RandomForestClassifier(n_estimators=n_trees_keep, random_state=self.random_state)
        else:
            pruned = RandomForestRegressor(n_estimators=n_trees_keep, random_state=self.random_state)

        pruned.estimators_ = [forest.estimators_[i] for i in top_indices]
        pruned.n_estimators = n_trees_keep
        for attr in ["classes_", "n_classes_", "n_features_in_", "feature_names_in_"]:
            if hasattr(forest, attr):
                setattr(pruned, attr, getattr(forest, attr))

        if hasattr(pruned, 'oob_score_'):
            pruned.oob_score_ = None

        return pruned

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'RFOD':
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        self.feature_names_ = list(X.columns)
        self.n_features_ = len(self.feature_names_)

        if self.verbose:
            print(f"Training RFOD: {len(X)} samples, {self.n_features_} features")

        self.feature_types_ = self._identify_feature_types(X)
        self._fit_encoders(X)
        self.quantiles_ = self._compute_quantiles(X)

        iterator = tqdm(range(self.n_features_), desc="Training forests", disable=not self.verbose)
        for feature_idx in iterator:
            forest = self._train_feature_forest(X, feature_idx)
            if self.beta < 1.0:
                forest = self._prune_forest(forest, X, feature_idx)
            self.forests_[feature_idx] = forest

        if self.verbose:
            print("Training complete")
        return self

    def _predict_feature(self, X: pd.DataFrame, feature_idx: int, batch_size: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
        forest = self.forests_[feature_idx]
        X_input_df = X.drop(X.columns[feature_idx], axis=1)
        X_input_encoded = self._transform_data(X_input_df)
        n_samples = X_input_encoded.shape[0]

        if isinstance(forest, RandomForestClassifier):
            n_classes = len(forest.classes_)
            sum_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
            sum_sq_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_input_encoded.iloc[start:end]
                all_tree_probs = np.array([tree.predict_proba(X_batch) for tree in forest.estimators_])
                mean_probs_batch = all_tree_probs.mean(axis=0)
                std_probs_batch = all_tree_probs.std(axis=0)
                sum_probs[start:end] = mean_probs_batch
                sum_sq_probs[start:end] = std_probs_batch

            uncertainties = sum_sq_probs.max(axis=1)
            return sum_probs, uncertainties

        else:
            predictions = np.zeros(n_samples, dtype=np.float64)
            std_devs = np.zeros(n_samples, dtype=np.float64)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_input_encoded.iloc[start:end]
                preds = np.array([tree.predict(X_batch) for tree in forest.estimators_])
                mean_batch = preds.mean(axis=0)
                std_batch = preds.std(axis=0)
                predictions[start:end] = mean_batch
                std_devs[start:end] = std_batch

            return predictions, std_devs

    def _compute_cell_scores(self, X: pd.DataFrame, predictions: Dict[int, np.ndarray]) -> np.ndarray:
        n_samples = len(X)
        cell_scores = np.zeros((n_samples, self.n_features_))

        for feature_idx in range(self.n_features_):
            true_values_series = X.iloc[:, feature_idx]
            pred_values = predictions[feature_idx]

            if self.feature_types_[feature_idx] == 'numeric':
                q_low, q_high = self.quantiles_.get(feature_idx, (0.0, 1.0))
                denom = (q_high - q_low) if (q_high - q_low) != 0 else 1.0
                true_values_filled = true_values_series.fillna(np.mean(pred_values)).values.astype(float)
                pred_values_filled = np.nan_to_num(pred_values, nan=np.mean(pred_values)).astype(float)
                diff = np.abs(true_values_filled - pred_values_filled)
                cell_scores[:, feature_idx] = diff / denom

            else:
                forest = self.forests_[feature_idx]
                classes = getattr(forest, "classes_", None)
                if classes is None:
                    continue

                target_col_name = self.feature_names_[feature_idx]
                le = self.encoders_.get(target_col_name)
                if le is None:
                    continue

                true_values_str = true_values_series.astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~true_values_str.isin(le.classes_)
                true_values_str.loc[unseen_mask] = le.classes_[0]
                true_values_encoded = le.transform(true_values_str)
                true_values_encoded[unseen_mask] = -1

                class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
                probs = np.zeros(n_samples, dtype=np.float64)
                for i in range(n_samples):
                    true_class = true_values_encoded[i]
                    if true_class in class_to_idx:
                        probs[i] = pred_values[i, class_to_idx[true_class]]
                cell_scores[:, feature_idx] = 1.0 - probs

        return cell_scores

    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_cell_scores: bool = False,
                clip_scores: bool = False, clip_min: float = 0.0, clip_max: float = 1.0,
                batch_size: int = 50000) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)

        n_samples = len(X)
        if self.verbose:
            print(f"Predicting {n_samples} samples (batch_size={batch_size})")

        predictions = {}
        uncertainties = {}

        iterator = tqdm(range(self.n_features_), desc="Predicting features", disable=not self.verbose)
        for feature_idx in iterator:
            pred, uncert = self._predict_feature(X, feature_idx, batch_size=batch_size)
            predictions[feature_idx] = pred
            uncertainties[feature_idx] = uncert

        cell_scores = self._compute_cell_scores(X, predictions)
        uncertainty_matrix = np.column_stack([uncertainties[i] for i in range(self.n_features_)])
        row_sums = uncertainty_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10
        uncertainty_norm = uncertainty_matrix / row_sums
        weights = 1.0 - uncertainty_norm
        weighted_scores = weights * cell_scores
        row_scores = weighted_scores.mean(axis=1)

        if clip_scores:
            row_scores = np.clip(row_scores, clip_min, clip_max)

        if self.verbose:
            print(f"Prediction complete: score range [{row_scores.min():.6f}, {row_scores.max():.6f}]")

        if return_cell_scores:
            return row_scores, cell_scores
        else:
            return row_scores


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

BASE_FEATURES = [
    "timestamp", "processId", "parentProcessId", "userId", "mountNamespace",
    "processName", "hostName", "eventName", "argsNum", "returnValue", "stack_depth"
]


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Dynamically extract feature columns from cleaned data
    Includes: BASE_FEATURES + args features (arg_*, args_*count, args_unique_names)
    """
    feature_cols = []

    # Add base features that exist in df
    for feat in BASE_FEATURES:
        if feat in df.columns:
            feature_cols.append(feat)

    # Add args-related features (generated by extract_args_topk_stats)
    for col in df.columns:
        if col.startswith('arg_') or col.startswith('args_'):
            if col not in feature_cols:
                feature_cols.append(col)

    return feature_cols


def _safe_clean_csv(input_path: str, process_args: Union[bool, str] = "topk") -> pd.DataFrame:
    tmp_out = os.path.join(tempfile.gettempdir(), f"cleaned_{os.path.basename(input_path)}")
    df = clean_csv(input_path, tmp_out, process_args=process_args, save=False)
    return df


def _select_and_align_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    for f in feature_names:
        if f in df.columns:
            out[f] = df[f]
        else:
            out[f] = np.nan
            print(f"Warning: missing feature '{f}', filled with NaN")
    return out


def _stable_param_signature(params: Dict) -> Tuple[str, OrderedDict, str]:
    ordered = OrderedDict(sorted(params.items(), key=lambda x: x[0]))
    readable = "_".join([f"{k}={ordered[k]}" for k in ordered])
    safe = readable.replace(" ", "").replace(".", "p")
    md5 = hashlib.md5(readable.encode()).hexdigest()[:8]
    return f"{safe}__{md5}", ordered, readable


# ============================================================================
# TRAINING & INFERENCE PIPELINE
# ============================================================================

def train_and_infer(
    train_csv: str,
    test_csv: Optional[str] = None,
    output_path: str = "result/prediction.csv",
    batch_size: int = 50000,
    alpha: float = 0.02,
    beta: float = 0.7,
    n_estimators: int = 30,
    max_depth: int = 6,
    random_state: int = 42,
    n_jobs: int = -1,
    process_args: Union[bool, str] = "topk",
    drop_labelled_anomalies: bool = False,
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

    # Dynamically get feature columns (includes BASE_FEATURES + args features)
    feature_cols = _get_feature_columns(df_train)
    if verbose:
        base_count = len([f for f in feature_cols if f in BASE_FEATURES])
        args_count = len([f for f in feature_cols if f.startswith('arg_') or f.startswith('args_')])
        print(f"Feature extraction: {base_count} base features + {args_count} args features = {len(feature_cols)} total")

    X_train = _select_and_align_features(df_train, feature_cols)
    if X_train.empty:
        raise ValueError("Training data is empty")

    if verbose:
        print(f"Train samples: {len(X_train)}, Features: {len(feature_cols)}")

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

    if verbose:
        print("\n" + "="*60)
        print("STEP 2: Saving Model")
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
        'params': params,
        'feature_cols': feature_cols,  # Save feature list for inference consistency
        'saved_at': datetime.now().isoformat(timespec="seconds"),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    if verbose:
        print(f"Model saved: {model_path}")

    results['model_path'] = model_path
    results['params'] = params

    if test_csv is not None:
        if verbose:
            print("\n" + "="*60)
            print("STEP 3: Testing Inference")
            print("="*60)

        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test file not found: {test_csv}")

        df_test = _safe_clean_csv(test_csv, process_args=process_args)

        if 'Id' not in df_test.columns:
            raise ValueError("Test set must contain 'Id' column")

        if verbose:
            print(f"Test samples: {len(df_test)}")

        # Use the same feature columns as training
        X_test = _select_and_align_features(df_test, feature_cols)

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

        out_df = pd.DataFrame({
            'Id': df_test['Id'],
            'target': normalized_scores
        })

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        out_df.to_csv(output_path, index=False)

        if verbose:
            print(f"Predictions saved: {output_path}")
            print(f"Output preview:\n{out_df.head(10)}")

        results['output_path'] = output_path
        results['n_test'] = len(df_test)
        results['predictions'] = out_df

    if verbose:
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("RFOD Training and Inference")
    print("Local configuration: 8GB RAM")
    print("Args processing: Top-K + Statistics (research-based)\n")

    results = train_and_infer(
        train_csv="data/processes_train.csv",
        test_csv="data/processes_test.csv",
        output_path="result/submission.csv",

        batch_size=10000,
        alpha=0.005,
        beta=0.7,
        n_estimators=80,
        max_depth=20,
        random_state=42,
        n_jobs=4,

        process_args="topk",  # Options: False, "topk", "full"
        drop_labelled_anomalies=False,

        normalize_method="minmax",
        out_dir="model",
        verbose=True
    )

    print(f"\nModel saved: {results.get('model_path')}")
    if 'output_path' in results:
        print(f"Predictions saved: {results.get('output_path')}")
        print(f"Test samples: {results.get('n_test')}")
