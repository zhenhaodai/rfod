"""
RFOD (Random Forest-based Outlier Detection)
"""
import os
import tempfile
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.metrics import (
    roc_auc_score, r2_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, fbeta_score,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
from data_process import clean_csv
from tqdm import tqdm
import hashlib
from datetime import datetime
from collections import OrderedDict


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

    def fit_predict(self, X_train: Union[pd.DataFrame, np.ndarray], X_test: Union[pd.DataFrame, np.ndarray], return_cell_scores: bool = False):
        self.fit(X_train)
        return self.predict(X_test, return_cell_scores=return_cell_scores)


def _stable_param_signature(params: Dict) -> Tuple[str, OrderedDict, str]:
    ordered = OrderedDict(sorted(params.items(), key=lambda x: x[0]))
    readable = "_".join([f"{k}={ordered[k]}" for k in ordered])
    safe = readable.replace(" ", "").replace(".", "p")
    md5 = hashlib.md5(readable.encode()).hexdigest()[:8]
    return f"{safe}__{md5}", ordered, readable


def _scan_thresholds_from_scores(scores: np.ndarray, n_points: int = 256) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, num=n_points)
    thrs = np.quantile(scores, qs)
    return np.unique(thrs)


def _compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[float, int, None]]:
    out: Dict[str, Union[float, int, None]] = {}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape == (1, 1) and y_true[0] == 0 else 0
        fp = fn = tp = 0
    out.update(dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn)))

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["f0p5"] = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
    out["f2"] = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if (tp+tn+fp+fn) > 0 else 0.0

    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = 1.0 - tnr
    out["specificity"] = float(tnr)
    out["fpr"] = float(fpr)
    out["youden_j"] = float(out["recall"] - fpr)

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = None

    return out


REQ_FEATURES = [
    "timestamp", "processId", "parentProcessId", "userId", "mountNamespace",
    "processName", "hostName", "eventName", "argsNum", "returnValue", "stack_depth"
]


def _safe_clean_csv(input_path: str, process_args: bool = True) -> pd.DataFrame:
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
