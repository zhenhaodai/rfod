"""
RFOD 训练和推理合并脚本
功能：在训练集上训练模型，保存后直接在测试集上推理
"""
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List

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
    # 训练相关参数
    train_csv: str,
    valid_csv: Optional[str] = None,
    # 推理相关参数
    test_csv: Optional[str] = None,
    output_path: str = "result/prediction.csv",
    # 模型参数
    alpha: float = 0.02,
    beta: float = 0.7,
    n_estimators: int = 30,
    max_depth: int = 6,
    random_state: int = 42,
    n_jobs: int = -1,
    backend: str = "sklearn",
    n_streams: int = 4,
    # 数据处理参数
    process_args: bool = False,
    drop_labelled_anomalies: bool = False,
    # 阈值相关参数
    threshold: Optional[float] = None,
    threshold_percentile: int = 95,
    threshold_strategy: str = "train_quantile",  # 'train_quantile' | 'valid_best_f1' | 'valid_best_j'
    # 推理输出参数
    use_threshold: bool = False,
    normalize_method: str = "minmax",  # 'minmax' | 'robust' | 'clip' | 'none'
    # 保存参数
    out_dir: str = "model",
    verbose: bool = True
) -> Dict:
    """
    完整的训练和推理流程

    参数:
        train_csv: 训练集CSV路径
        valid_csv: 验证集CSV路径（用于评估和阈值选择，可选）
        test_csv: 测试集CSV路径（用于最终推理，可选）
        output_path: 推理结果输出路径

        # 模型超参数
        alpha: RFOD的alpha参数
        beta: RFOD的beta参数（森林修剪比例）
        n_estimators: 随机森林树的数量
        max_depth: 树的最大深度
        random_state: 随机种子
        n_jobs: 并行任务数
        backend: 'sklearn' 或 'cuml'
        n_streams: cuML GPU流数

        # 数据处理
        process_args: 是否处理args列
        drop_labelled_anomalies: 是否从训练集中移除标注的异常样本

        # 阈值设置
        threshold: 手动指定阈值（如果指定，将忽略其他阈值策略）
        threshold_percentile: train_quantile策略使用的百分位数
        threshold_strategy: 阈值选择策略

        # 推理输出
        use_threshold: 是否使用阈值输出0/1（False则输出分数）
        normalize_method: 分数归一化方法

        # 其他
        out_dir: 模型保存目录
        verbose: 是否显示详细信息

    返回:
        包含训练和推理信息的字典
    """

    results = {}

    # ============ 第一步：训练模型 ============
    if verbose:
        print("=" * 60)
        print("第一步：训练RFOD模型")
        print("=" * 60)

    # 清洗训练集
    if verbose:
        print(f"--> 清洗训练集: {train_csv}")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"训练集文件不存在: {train_csv}")

    df_train = _safe_clean_csv(train_csv, process_args=process_args)
    if df_train.empty:
        raise ValueError(f"无法加载或清洗训练集: {train_csv}")

    # 移除标注的异常样本（如果需要）
    if drop_labelled_anomalies and "target" in df_train.columns:
        before = len(df_train)
        df_train = df_train[df_train["target"].astype(str) != "1"]
        if verbose:
            print(f"  已移除 {before - len(df_train)} 个标注为异常的训练样本")

    # 选择特征
    X_train = _select_and_align_features(df_train, REQ_FEATURES)
    if X_train.empty:
        raise ValueError("训练数据为空或所有特征缺失")

    if verbose:
        print(f"  训练集样本数: {len(X_train)}")
        print(f"  特征数: {len(REQ_FEATURES)}")

    # 构建模型参数
    params = {
        'alpha': alpha,
        'beta': beta,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'random_state': random_state,
        'n_jobs': n_jobs,
        'backend': backend,
        'n_streams': n_streams,
        'verbose': verbose
    }

    # 训练模型
    if verbose:
        print(f"\n--> 使用参数训练模型: {params}")

    rfod = RFOD(**params)
    rfod.fit(X_train)

    # 在训练集上预测（用于阈值计算）
    train_scores = rfod.predict(X_train)

    # ============ 第二步：确定阈值 ============
    if verbose:
        print("\n" + "=" * 60)
        print("第二步：确定异常检测阈值")
        print("=" * 60)

    thr = None
    thr_src = None
    metrics = {}

    # 如果手动指定阈值
    if threshold is not None:
        thr = float(threshold)
        thr_src = "user_provided"
        if verbose:
            print(f"--> 使用用户指定阈值: {thr:.6f}")

    # 如果有验证集，可以用于评估和阈值选择
    elif valid_csv is not None:
        if verbose:
            print(f"--> 清洗验证集: {valid_csv}")

        if not os.path.exists(valid_csv):
            raise FileNotFoundError(f"验证集文件不存在: {valid_csv}")

        df_valid = _safe_clean_csv(valid_csv, process_args=process_args)
        if df_valid.empty:
            raise ValueError(f"无法加载或清洗验证集: {valid_csv}")

        # 检查是否有标签
        if "target" not in df_valid.columns:
            if verbose:
                print("  警告: 验证集没有target列，将使用训练集分位数策略")
            thr = float(np.percentile(train_scores, threshold_percentile))
            thr_src = f"train_quantile_{threshold_percentile}"
        else:
            X_valid = _select_and_align_features(df_valid, REQ_FEATURES)
            y_true = df_valid["target"].astype(int).values

            if verbose:
                print(f"  验证集样本数: {len(X_valid)}")
                print(f"  验证集异常样本数: {y_true.sum()} ({y_true.sum()/len(y_true)*100:.2f}%)")

            # 在验证集上预测
            valid_scores = rfod.predict(X_valid)

            # 根据策略选择阈值
            if threshold_strategy == "train_quantile":
                thr = float(np.percentile(train_scores, threshold_percentile))
                thr_src = f"train_quantile_{threshold_percentile}"
                if verbose:
                    print(f"--> 使用训练集第{threshold_percentile}百分位数作为阈值: {thr:.6f}")

            elif threshold_strategy == "valid_best_f1":
                if verbose:
                    print(f"--> 在验证集上搜索最优F1阈值...")
                candidates = _scan_thresholds_from_scores(valid_scores, n_points=256)
                best_f1, thr = -1.0, candidates[0]
                for t in candidates:
                    yp = (valid_scores > t).astype(int)
                    f1v = f1_score(y_true, yp, zero_division=0)
                    if f1v > best_f1:
                        best_f1, thr = f1v, float(t)
                thr_src = "valid_best_f1"
                if verbose:
                    print(f"  最优阈值: {thr:.6f} (F1={best_f1:.4f})")

            elif threshold_strategy == "valid_best_j":
                if verbose:
                    print(f"--> 在验证集上搜索最优Youden's J阈值...")
                candidates = _scan_thresholds_from_scores(valid_scores, n_points=256)
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
                if verbose:
                    print(f"  最优阈值: {thr:.6f} (Youden's J={best_j:.4f})")

            else:
                raise ValueError(f"未知的阈值策略: {threshold_strategy}")

            # 计算验证集指标
            y_pred = (valid_scores > thr).astype(int)
            metrics = _compute_binary_metrics(y_true, valid_scores, y_pred)

            if verbose:
                print(f"\n验证集性能:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")
                if metrics['roc_auc'] is not None:
                    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                if metrics['pr_auc'] is not None:
                    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")

    # 如果既没有手动阈值也没有验证集，使用训练集分位数
    else:
        thr = float(np.percentile(train_scores, threshold_percentile))
        thr_src = f"train_quantile_{threshold_percentile}"
        if verbose:
            print(f"--> 使用训练集第{threshold_percentile}百分位数作为阈值: {thr:.6f}")

    # ============ 第三步：保存模型 ============
    if verbose:
        print("\n" + "=" * 60)
        print("第三步：保存模型")
        print("=" * 60)

    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)

    # 生成模型签名
    model_params = {
        'alpha': alpha,
        'beta': beta,
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }
    sig, ord_params, readable = _stable_param_signature(model_params)

    # 保存模型
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
        print(f"--> 模型已保存到: {model_path}")

    results['model_path'] = model_path
    results['threshold'] = thr
    results['threshold_source'] = thr_src
    results['params'] = params
    results['metrics'] = metrics

    # ============ 第四步：测试集推理 ============
    if test_csv is not None:
        if verbose:
            print("\n" + "=" * 60)
            print("第四步：测试集推理")
            print("=" * 60)

        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"测试集文件不存在: {test_csv}")

        # 清洗测试集
        if verbose:
            print(f"--> 清洗测试集: {test_csv}")

        df_test = _safe_clean_csv(test_csv, process_args=process_args)

        if 'Id' not in df_test.columns:
            raise ValueError("测试集必须包含 'Id' 列")

        if verbose:
            print(f"  测试集样本数: {len(df_test)}")

        X_test = _select_and_align_features(df_test, REQ_FEATURES)

        if X_test.empty:
            raise ValueError("测试数据为空或所有特征缺失")

        # 预测
        if verbose:
            print("--> 开始预测...")

        test_scores = rfod.predict(X_test, clip_scores=False)

        original_min = test_scores.min()
        original_max = test_scores.max()

        if verbose:
            print(f"  原始分数范围: [{original_min:.6f}, {original_max:.6f}]")
            print(f"  原始分数均值: {test_scores.mean():.6f}")
            print(f"  原始分数标准差: {test_scores.std():.6f}")

        # 归一化处理
        if normalize_method == "minmax":
            score_range = original_max - original_min
            if score_range > 1e-10:
                normalized_scores = (test_scores - original_min) / score_range
            else:
                normalized_scores = np.zeros_like(test_scores)
            if verbose:
                print(f"--> Min-Max归一化到[0,1]")

        elif normalize_method == "robust":
            q25, q50, q75 = np.percentile(test_scores, [25, 50, 75])
            iqr = q75 - q25
            if iqr > 1e-10:
                normalized_scores = (test_scores - q50) / iqr
                normalized_scores = (normalized_scores - normalized_scores.min()) / \
                                  (normalized_scores.max() - normalized_scores.min())
            else:
                normalized_scores = np.zeros_like(test_scores)
            if verbose:
                print(f"--> 鲁棒归一化 (IQR={iqr:.4f})")

        elif normalize_method == "clip":
            normalized_scores = np.clip(test_scores, 0.0, 1.0)
            if verbose:
                print(f"--> 裁剪到[0,1]")

        elif normalize_method == "none":
            normalized_scores = test_scores
            if verbose:
                print(f"--> 不归一化，保留原始分数")

        else:
            raise ValueError(f"未知的归一化方法: {normalize_method}")

        # 根据参数决定输出格式
        if use_threshold and thr is not None:
            predictions = (normalized_scores > thr).astype(int)
            if verbose:
                print(f"\n--> 使用阈值 {thr:.6f} 进行二分类")
                print(f"  预测为异常的样本数: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")

            out_df = pd.DataFrame({
                'Id': df_test['Id'],
                'target': predictions
            })
            output_type = "二分类 (0/1)"
        else:
            if verbose:
                print(f"\n--> 输出归一化后的异常分数")

            out_df = pd.DataFrame({
                'Id': df_test['Id'],
                'target': normalized_scores
            })
            output_type = f"异常概率 (归一化: {normalize_method})"

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 保存结果
        out_df.to_csv(output_path, index=False)

        if verbose:
            print(f"--> 预测结果已保存到: {output_path}")
            print(f"  输出格式: {output_type}")
            print("\n预测结果预览:")
            print(out_df.head(10))

        results['output_path'] = output_path
        results['output_type'] = output_type
        results['n_test'] = len(df_test)
        results['predictions'] = out_df

    if verbose:
        print("\n" + "=" * 60)
        print("完成！")
        print("=" * 60)

    return results


if __name__ == "__main__":
    # 示例用法
    results = train_and_infer(
        # 数据路径
        train_csv="data/processes_train.csv",
        valid_csv="data/processes_valid.csv",  # 可选，用于评估
        test_csv="data/processes_test.csv",    # 可选，用于最终推理
        output_path="result/submission.csv",

        # 模型参数
        alpha=0.005,
        beta=0.7,
        n_estimators=80,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        backend="sklearn",  # 或 "cuml" 如果有GPU

        # 数据处理
        process_args=False,
        drop_labelled_anomalies=False,

        # 阈值策略
        threshold=None,  # 或手动指定如 0.5
        threshold_percentile=99.7,
        threshold_strategy="train_quantile",  # 'train_quantile' | 'valid_best_f1' | 'valid_best_j'

        # 推理输出
        use_threshold=False,  # False=输出分数, True=输出0/1
        normalize_method="minmax",  # 'minmax' | 'robust' | 'clip' | 'none'

        # 其他
        out_dir="model",
        verbose=True
    )

    print("\n训练和推理完成！")
    print(f"模型保存路径: {results.get('model_path')}")
    if 'output_path' in results:
        print(f"预测结果保存路径: {results.get('output_path')}")
