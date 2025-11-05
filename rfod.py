"""
RFOD (Random Forest-based Outlier Detection) 
"""
import os
import tempfile
import json
import itertools
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from data_process import clean_csv
import hashlib
from datetime import datetime
from collections import OrderedDict

from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score, balanced_accuracy_score,
    matthews_corrcoef, average_precision_score, precision_recall_curve
)

# 可选GPU依赖（RAPIDS cuML）
try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRFClassifier
    from cuml.ensemble import RandomForestRegressor as cuRFRegressor
    HAS_CUML = True
except ImportError:
    HAS_CUML = False
    cudf = None
    cp = None
    cuRFClassifier = None
    cuRFRegressor = None



class RFOD:
    def __init__(
        self,
        alpha: float = 0.02,
        beta: float = 0.7,
        n_estimators: int = 30,
        max_depth: int = 6,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True,
        backend: str = "sklearn",  # "sklearn" or "cuml"
        n_streams: int = 4  # cuML专用：GPU并行流数
    ):
        self.alpha = alpha
        self.beta = beta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend
        self.n_streams = n_streams

        # 检查cuML可用性
        if self.backend == "cuml":
            if not HAS_CUML:
                raise RuntimeError(
                    "backend='cuml' 但未检测到 cuML。\n"
                    "请在 WSL2/Linux 环境中安装 RAPIDS:\n"
                    "  conda create -n rapids -c rapidsai -c conda-forge -c nvidia rapids python=3.10 cudatoolkit"
                )
            if self.verbose:
                print("[RFOD] 使用 GPU 加速 (cuML backend)")
        elif self.verbose and self.backend == "sklearn":
            print("[RFOD] 使用 CPU 计算 (sklearn backend)")

        self.forests_ = {}
        self.feature_types_ = {}
        self.quantiles_ = {}
        self.feature_names_ = []
        self.n_features_ = 0
        self.encoders_: Dict[str, LabelEncoder] = {}

    def _identify_feature_types(self, X: pd.DataFrame) -> Dict[int, str]:
        """识别数值型和类别型特征"""
        feature_types = {}
        for idx, col in enumerate(X.columns):
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_types[idx] = 'numeric'
            else:
                feature_types[idx] = 'categorical'
        return feature_types

    def _compute_quantiles(self, X: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
        """计算数值型特征的 alpha 和 1-alpha 分位数"""
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
        """拟合类别特征的 LabelEncoders"""
        self.encoders_ = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                series = X[col].astype(str).fillna("NaN_TOKEN")
                le.fit(series)
                self.encoders_[col] = le

    def _transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用存储的 LabelEncoders 转换数据"""
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
        """训练单个特征的预测森林"""
        X_train_df = X.drop(X.columns[feature_idx], axis=1)
        y_train = X.iloc[:, feature_idx]

        X_train_encoded = self._transform_data(X_train_df)

        target_col_name = X.columns[feature_idx]

        # cuML 后端
        if self.backend == "cuml":
            X_cu = cudf.DataFrame.from_pandas(X_train_encoded)

            if self.feature_types_[feature_idx] == 'categorical':
                if target_col_name in self.encoders_:
                    y_train_series = y_train.astype(str).fillna("NaN_TOKEN")
                    unseen_mask = ~y_train_series.isin(self.encoders_[target_col_name].classes_)
                    y_train_series.loc[unseen_mask] = self.encoders_[target_col_name].classes_[0]
                    y_train_encoded = self.encoders_[target_col_name].transform(y_train_series)
                    y_train_encoded[unseen_mask] = -1
                    y_train = y_train_encoded
                y_cu = cudf.Series(pd.Series(y_train).astype('int32'))
                forest = cuRFClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_streams=self.n_streams
                )
            else:
                y_train = y_train.fillna(y_train.mean())
                y_cu = cudf.Series(pd.Series(y_train).astype('float32'))
                forest = cuRFRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_streams=self.n_streams
                )
            forest.fit(X_cu, y_cu)
            return forest

        # sklearn 后端（原逻辑）
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
        """使用真实的 OOB 样本修剪森林"""
        # cuML 不支持逐树修剪
        if self.backend == "cuml":
            if self.verbose and self.beta < 1.0:
                print(f"    [注意] cuML 不支持森林修剪，beta={self.beta} 将被忽略")
            return forest

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
        
        # 修复: 直接计算 n_samples_bootstrap
        if forest.max_samples is None:
            n_samples_bootstrap = n_samples
        elif isinstance(forest.max_samples, int):
            n_samples_bootstrap = forest.max_samples
        else:  # float
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
            except Exception as e:
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
            print(f"[RFOD] 开始训练, 样本数: {len(X)}, 特征数: {self.n_features_}")
            
        self.feature_types_ = self._identify_feature_types(X)
        self._fit_encoders(X)
        
        if self.verbose:
            n_numeric = sum(1 for t in self.feature_types_.values() if t == 'numeric')
            n_categorical = self.n_features_ - n_numeric
            print(f"[RFOD] 特征类型: {n_numeric} 个数值型, {n_categorical} 个类别型")
            
        self.quantiles_ = self._compute_quantiles(X)
        
        if self.verbose:
            print("[RFOD] 训练特征专属随机森林...")
            
        for feature_idx in range(self.n_features_):
            if self.verbose:
                print(f"  训练特征 {feature_idx+1}/{self.n_features_}: {self.feature_names_[feature_idx]} ({self.feature_types_[feature_idx]})")
            
            forest = self._train_feature_forest(X, feature_idx)
            
            if self.beta < 1.0:
                forest = self._prune_forest(forest, X, feature_idx)
                
            self.forests_[feature_idx] = forest
            
        if self.verbose:
            print("[RFOD] 训练完成。")
        return self

    def _predict_feature(self, X: pd.DataFrame, feature_idx: int, batch_size: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
        """预测单个特征的值及其不确定性"""
        forest = self.forests_[feature_idx]
        X_input_df = X.drop(X.columns[feature_idx], axis=1)
        X_input_encoded = self._transform_data(X_input_df)

        n_samples = X_input_encoded.shape[0]

        # cuML 后端
        if self.backend == "cuml":
            X_cu = cudf.DataFrame.from_pandas(X_input_encoded)

            if isinstance(forest, (cuRFClassifier,)) or hasattr(forest, 'predict_proba'):
                proba = forest.predict_proba(X_cu)
                # 转换回 numpy
                if hasattr(proba, 'values'):
                    proba_np = proba.values.get() if hasattr(proba.values, 'get') else proba.values
                elif hasattr(proba, 'get'):
                    proba_np = proba.get()
                else:
                    proba_np = cp.asnumpy(proba) if isinstance(proba, cp.ndarray) else np.array(proba)
                # 不确定度用 1 - max_prob 近似
                uncertainties = 1.0 - proba_np.max(axis=1)
                return proba_np, uncertainties
            else:
                # 回归
                preds = forest.predict(X_cu)
                if hasattr(preds, 'values'):
                    preds_np = preds.values.get() if hasattr(preds.values, 'get') else preds.values
                elif hasattr(preds, 'get'):
                    preds_np = preds.get()
                else:
                    preds_np = cp.asnumpy(preds) if isinstance(preds, cp.ndarray) else np.array(preds)
                # cuML 无法逐树方差，给常数 0（权重=1）
                std = np.zeros_like(preds_np, dtype=np.float64)
                return preds_np, std

        # sklearn 后端
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
        """计算 AGD 作为单元格异常分数"""
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
                # 分类特征：计算 1 - P(true_class)
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

                # 向量化：避免逐行循环
                # 构建类别到索引的映射
                class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

                # 批量查找真实类别对应的概率
                probs = np.zeros(n_samples, dtype=np.float64)
                for i in range(n_samples):
                    true_class = true_values_encoded[i]
                    if true_class in class_to_idx:
                        probs[i] = pred_values[i, class_to_idx[true_class]]
                    # else: probs[i] = 0.0 (已初始化)

                cell_scores[:, feature_idx] = 1.0 - probs
                    
        return cell_scores

    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_cell_scores: bool = False,
                clip_scores: bool = False, clip_min: float = 0.0, clip_max: float = 1.0,
                batch_size: int = 50000) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测异常分数

        参数:
            X: 输入数据
            return_cell_scores: 是否返回单元格分数
            clip_scores: 是否裁剪分数到指定范围（默认True）
            clip_min: 裁剪下限（默认0.0）
            clip_max: 裁剪上限（默认1.0）
            batch_size: 批处理大小，越大使用内存越多但速度可能更快（默认50000）

        返回:
            row_scores: 行级异常分数
            cell_scores: 单元格分数（如果 return_cell_scores=True）
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)

        n_samples = len(X)
        if self.verbose:
            print(f"[RFOD] 开始预测 {n_samples} 个样本... (batch_size={batch_size})")

        predictions = {}
        uncertainties = {}

        for feature_idx in range(self.n_features_):
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

        # 裁剪分数到指定范围
        if clip_scores:
            original_min, original_max = row_scores.min(), row_scores.max()
            row_scores = np.clip(row_scores, clip_min, clip_max)
            if self.verbose and (original_min < clip_min or original_max > clip_max):
                print(f"[RFOD] 分数已裁剪: [{original_min:.6f}, {original_max:.6f}] -> [{row_scores.min():.6f}, {row_scores.max():.6f}]")

        if self.verbose:
            print(f"[RFOD] 预测完成，row_scores 范围: [{row_scores.min():.6f}, {row_scores.max():.6f}]")

        if return_cell_scores:
            return row_scores, cell_scores
        else:
            return row_scores

    def fit_predict(self, X_train: Union[pd.DataFrame, np.ndarray], X_test: Union[pd.DataFrame, np.ndarray], return_cell_scores: bool = False):
        self.fit(X_train)
        return self.predict(X_test, return_cell_scores=return_cell_scores)

def _stable_param_signature(params: Dict) -> Tuple[str, OrderedDict, str]:
    """
    依据超参数字典生成稳定的文件名签名。
    返回: (signature, ordered_params, readable_str)
    """
    ordered = OrderedDict(sorted(params.items(), key=lambda x: x[0]))
    readable = "_".join([f"{k}={ordered[k]}" for k in ordered])
    # 文件名友好（小数点 -> p，空格去掉）
    safe = readable.replace(" ", "").replace(".", "p")
    md5 = hashlib.md5(readable.encode()).hexdigest()[:8]
    return f"{safe}__{md5}", ordered, readable


def _scan_thresholds_from_scores(scores: np.ndarray, n_points: int = 256) -> np.ndarray:
    """
    从分数分布中抽取候选阈值（基于分位数，去重），用于在验证集上选最优阈值。
    """
    qs = np.linspace(0.0, 1.0, num=n_points)
    thrs = np.quantile(scores, qs)
    return np.unique(thrs)


def _compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[float, int, None]]:
    """
    针对异常检测（二分类：1=异常），给出全面指标。
    y_score: 连续分数（越大越“异常”）
    y_pred: 经过阈值后的 0/1 预测
    """
    out: Dict[str, Union[float, int, None]] = {}

    # 混淆矩阵元素
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # 单类退化情形
        tn = cm[0, 0] if cm.shape == (1, 1) and y_true[0] == 0 else 0
        fp = fn = tp = 0
    out.update(dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn)))

    # 基础指标
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))  # = TPR / sensitivity
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["f0p5"] = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
    out["f2"] = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if (tp+tn+fp+fn) > 0 else 0.0

    # 特异度/假阳性率
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = 1.0 - tnr
    out["specificity"] = float(tnr)
    out["fpr"] = float(fpr)
    out["youden_j"] = float(out["recall"] - fpr)

    # 曲线类指标（对单类退化要兜底）
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = None

    return out

# Pipeline 相关代码
REQ_FEATURES = [
    "timestamp", "processId", "parentProcessId", "userId", "mountNamespace",
    "processName", "hostName", "eventName", "argsNum", "returnValue", "stack_depth"
]

def _safe_clean_csv(input_path: str, process_args: bool = True) -> pd.DataFrame:
    """调用 clean_csv 返回 cleaned DataFrame"""
    tmp_out = os.path.join(tempfile.gettempdir(), f"cleaned_{os.path.basename(input_path)}")
    df = clean_csv(input_path, tmp_out, process_args=process_args, save=False)
    return df

def _select_and_align_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """选取特征，缺失则补 NaN"""
    out = pd.DataFrame()
    for f in feature_names:
        if f in df.columns:
            out[f] = df[f]
        else:
            out[f] = np.nan
            print(f"⚠️ 特征缺失，已用 NaN 填充: {f}")
    return out

def train_validate_pipeline(
    train_csv: str,
    valid_csv: str,
    process_args: bool = True,
    threshold: Optional[float] = None,
    threshold_percentile: int = 95,
    verbose: bool = True,
    drop_labelled_anomalies: bool = False,
    param_grid: Optional[Dict[str, List]] = None,
    # === 新增：比较方式与阈值策略 ===
    selection_metric: str = "f1",          # 可选: 'f1' | 'pr_auc' | 'roc_auc' | 'balanced_accuracy' | 'youden_j' | 'accuracy' | 'mcc'
    threshold_strategy: str = "train_quantile",   # 可选: 'train_quantile' | 'valid_best_f1' | 'valid_best_j'
    # === 新增：保存控制 ===
    out_dir: str = "model",
    save_all_models: bool = True,
    max_thr_scan_points: int = 256,
) -> Dict[str, Union[float, int, str]]:
    """
    训练+验证流程。关键增强：
      - 每个超参组合的模型都会各自保存（含阈值与评测）；
      - 保存整表 metrics 到 CSV/JSONL；
      - 最佳模型按 selection_metric 选择。
    """

    print(f"--> 清洗训练集: {train_csv}")
    df_train = _safe_clean_csv(train_csv, process_args=process_args)
    if df_train.empty:
        print(f"错误: 无法加载或清洗 {train_csv}")
        return {}

    if drop_labelled_anomalies and "target" in df_train.columns:
        before = len(df_train)
        df_train = df_train[df_train["target"].astype(str) != "1"]
        print(f"  已移除 {before - len(df_train)} 个标注为异常的训练样本")

    X_train = _select_and_align_features(df_train, REQ_FEATURES)
    if X_train.empty:
        print("错误: 训练数据为空或所有特征缺失。")
        return {}

    print(f"--> 清洗验证集: {valid_csv}")
    df_valid = _safe_clean_csv(valid_csv, process_args=process_args)
    if df_valid.empty:
        print(f"错误: 无法加载或清洗 {valid_csv}")
        return {}

    if "target" not in df_valid.columns:
        print("警告: 验证集必须包含 'target' 列，已用 0 填充。")
        df_valid["target"] = 0

    X_valid = _select_and_align_features(df_valid, REQ_FEATURES)
    y_true = df_valid["target"].astype(int).values
    if X_valid.empty:
        print("错误: 验证数据为空。")
        return {}

    # === 输出目录 ===
    os.makedirs(out_dir, exist_ok=True)
    all_dir = os.path.join(out_dir, "all")
    os.makedirs(all_dir, exist_ok=True)

    metrics_rows: List[Dict[str, Union[str, int, float, None]]] = []

    def _select_best(rows: List[Dict]) -> Dict:
        """
        按 selection_metric 选择最优（越大越好）。若 metric 缺失则按顺序回退。
        """
        order = [selection_metric, "pr_auc", "roc_auc", "f1", "balanced_accuracy", "accuracy", "youden_j", "mcc"]
        def key_fn(r):
            for m in order:
                v = r.get(m, None)
                if v is not None:
                    return v
            return -1e9
        return max(rows, key=key_fn) if rows else {}

    # === 单模型流程（未给 param_grid） ===
    if param_grid is None:
        print("--> 使用默认参数训练 RFOD")
        params = dict(alpha=0.02, beta=0.7, n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)  # 与 __init__ 保持一致
        rfod = RFOD(verbose=verbose, **params)
        rfod.fit(X_train)

        train_scores = rfod.predict(X_train)
        if threshold is not None:
            thr = float(threshold)
            thr_src = "user_provided"
        elif threshold_strategy == "train_quantile":
            thr = float(np.percentile(train_scores, threshold_percentile))
            thr_src = f"train_quantile_{threshold_percentile}"
        else:
            # 需要验证集分数来选阈值
            valid_scores = rfod.predict(X_valid)
            if threshold_strategy == "valid_best_f1":
                candidates = _scan_thresholds_from_scores(valid_scores, n_points=max_thr_scan_points)
                best_f1, thr = -1.0, candidates[0]
                for t in candidates:
                    yp = (valid_scores > t).astype(int)
                    f1v = f1_score(y_true, yp, zero_division=0)
                    if f1v > best_f1:
                        best_f1, thr = f1v, float(t)
                thr_src = "valid_best_f1"
            elif threshold_strategy == "valid_best_j":
                candidates = _scan_thresholds_from_scores(valid_scores, n_points=max_thr_scan_points)
                best_j, thr = -1.0, candidates[0]
                for t in candidates:
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

        valid_scores = rfod.predict(X_valid)
        y_pred = (valid_scores > thr).astype(int)
        metrics = _compute_binary_metrics(y_true, valid_scores, y_pred)

        # 保存当前模型
        sig, ord_params, readable = _stable_param_signature(params)
        model_path = os.path.join(all_dir, f"rfod_{sig}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": rfod,
                    "threshold": thr,
                    "threshold_source": thr_src,
                    "params": dict(ord_params),
                    "metrics": metrics,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                },
                f,
            )
        print(f"--> 已保存模型: {model_path}")

        # 记录指标
        row = dict(signature=sig, threshold=thr, threshold_source=thr_src, n_valid=len(y_true))
        row.update(dict(ord_params))
        row.update(metrics)
        metrics_rows.append(row)

        best_row = row  # 只有一个

    # === 网格搜索流程 ===
    else:
        print("--> 开始网格搜索...")
        keys, values = zip(*param_grid.items())
        combos = list(itertools.product(*values))
        print(f"  总参数组合数: {len(combos)}")

        for idx, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            print(f"  [{idx}/{len(combos)}] 测试参数: {params}")

            rfod = RFOD(verbose=verbose, **params)
            rfod.fit(X_train)

            train_scores = rfod.predict(X_train)
            # 先拿验证分数，以便可能的阈值搜索
            valid_scores = rfod.predict(X_valid)

            if threshold is not None:
                thr = float(threshold)
                thr_src = "user_provided"
            elif threshold_strategy == "train_quantile":
                thr = float(np.percentile(train_scores, threshold_percentile))
                thr_src = f"train_quantile_{threshold_percentile}"
            elif threshold_strategy == "valid_best_f1":
                candidates = _scan_thresholds_from_scores(valid_scores, n_points=max_thr_scan_points)
                best_f1, thr = -1.0, candidates[0]
                for t in candidates:
                    yp = (valid_scores > t).astype(int)
                    f1v = f1_score(y_true, yp, zero_division=0)
                    if f1v > best_f1:
                        best_f1, thr = f1v, float(t)
                thr_src = "valid_best_f1"
            elif threshold_strategy == "valid_best_j":
                candidates = _scan_thresholds_from_scores(valid_scores, n_points=max_thr_scan_points)
                best_j, thr = -1.0, candidates[0]
                for t in candidates:
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

            # 保存“每个模型”
            sig, ord_params, readable = _stable_param_signature(params)
            model_path = os.path.join(all_dir, f"rfod_{sig}.pkl")
            if save_all_models:
                with open(model_path, "wb") as f:
                    pickle.dump(
                        {
                            "model": rfod,
                            "threshold": thr,
                            "threshold_source": thr_src,
                            "params": dict(ord_params),
                            "metrics": metrics,
                            "saved_at": datetime.now().isoformat(timespec="seconds"),
                        },
                        f,
                    )
                print(f"    -> 已保存: {model_path}")

            row = dict(signature=sig, threshold=thr, threshold_source=thr_src, n_valid=len(y_true))
            row.update(dict(ord_params))
            row.update(metrics)
            metrics_rows.append(row)

        # 选择最优
        best_row = _select_best(metrics_rows)
        print(f"--> 最佳组合（按 {selection_metric} 选）: {best_row.get('signature')}")
        if selection_metric in best_row:
            print(f"    {selection_metric} = {best_row[selection_metric]:.6f}")

        # 将最优模型另存为 best_model.pkl（从 all 中取回）
        best_sig = best_row["signature"]
        best_model_pkl = os.path.join(all_dir, f"rfod_{best_sig}.pkl")
        with open(best_model_pkl, "rb") as f:
            payload = pickle.load(f)
        # 单独再放一份
        with open(os.path.join(out_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(payload, f)
        print(f"--> 已保存最佳模型副本到: {os.path.join(out_dir, 'best_model.pkl')}")

    # 汇总表保存
    if metrics_rows:
        dfm = pd.DataFrame(metrics_rows)
        csv_path = os.path.join(out_dir, "metrics.csv")
        jsonl_path = os.path.join(out_dir, "metrics.jsonl")
        dfm.to_csv(csv_path, index=False)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in metrics_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"--> 已保存评测汇总: {csv_path}")
        print(f"--> 已保存评测明细: {jsonl_path}")

        # 友好地打印 Top-5 排行
        key = selection_metric if selection_metric in dfm.columns else "f1"
        topn = dfm.sort_values(by=key, ascending=False).head(5)
        print("\n=== TOP-5 (按 {} 降序) ===".format(key))
        print(topn[["signature", key, "threshold", "threshold_source", "accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "precision", "recall", "f1", "mcc"]].fillna("-"))

    # 返回简要总结
    ret = dict(
        best_signature=best_row.get("signature", ""),
        selection_metric=selection_metric,
        best_metric_value=float(best_row.get(selection_metric, -1)) if best_row.get(selection_metric) is not None else None,
        best_threshold=float(best_row.get("threshold", 0.0)) if "threshold" in best_row else None,
        n_valid=len(y_true),
        out_dir=out_dir,
    )
    return ret


if __name__ == "__main__":
    
    train_csv = "data/processes_train.csv"
    valid_csv = "data/processes_valid.csv"

    param_grid = {
        # alpha: 控制分位数范围，更小的值可能捕获更多异常模式
        "alpha": [0.005],

        # beta: 森林修剪比例，更高的值保留更多树，可能提高稳定性
        "beta": [0.7],

        # 增加树的数量，提高模型稳定性
        "n_estimators": [80],

        # 调整树的深度，更深可能捕获更复杂的模式
        "max_depth": [20]  # None表示不限制深度
    }
    
    res = train_validate_pipeline(
        train_csv=train_csv,
        valid_csv=valid_csv,
        process_args=False,
        threshold=None,
        threshold_percentile=99.7,

        verbose=True,
        drop_labelled_anomalies=False,
        param_grid=param_grid
    )
    print("\nDone. Summary:", res)