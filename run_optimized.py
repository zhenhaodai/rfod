"""
高性能配置脚本 - 针对240核心CPU优化
仅推理测试集，不在训练集和验证集上推理
"""
from train_and_infer import train_and_infer

if __name__ == "__main__":
    print("=" * 80)
    print("高性能RFOD训练和推理")
    print("CPU核心数: 240")
    print("优化策略: 多线程 + 大批处理")
    print("=" * 80)

    results = train_and_infer(
        # ============ 数据路径 ============
        train_csv="data/processes_train.csv",
        valid_csv="data/processes_valid.csv",  # 用于阈值选择和评估
        test_csv="data/processes_test.csv",    # 仅在此推理
        output_path="result/submission.csv",

        # ============ 性能优化参数 ============
        batch_size=100000,   # 批处理大小：增大到100K以充分利用内存
                             # 如果内存够大，可以设置为200000甚至更大

        # ============ 模型参数 ============
        alpha=0.005,         # RFOD的alpha参数
        beta=0.7,            # 森林修剪比例
        n_estimators=80,     # 随机森林树的数量
        max_depth=20,        # 树的最大深度
        random_state=42,     # 随机种子
        n_jobs=240,          # 使用全部240个核心
                             # 注意：也可以用-1自动检测，但明确指定240更好
        backend="sklearn",   # CPU使用sklearn，GPU使用"cuml"
        n_streams=4,         # cuML GPU流数（仅GPU模式有效）

        # ============ 数据处理 ============
        process_args=False,              # 是否处理args列
        drop_labelled_anomalies=False,   # 是否移除训练集中的已标注异常

        # ============ 阈值策略 ============
        threshold=None,                  # 手动阈值（None则自动计算）
        threshold_percentile=99.7,       # train_quantile策略使用的百分位
        threshold_strategy="train_quantile",
        # 可选策略:
        #   - "train_quantile": 使用训练集分位数（最快）
        #   - "valid_best_f1": 在验证集上搜索最优F1（需要标签）
        #   - "valid_best_j": 在验证集上搜索最优Youden's J（需要标签）

        # ============ 推理输出 ============
        use_threshold=False,         # False=输出分数, True=输出0/1
        normalize_method="minmax",   # 分数归一化方法
        # 可选方法:
        #   - "minmax": Min-Max归一化到[0,1]（推荐）
        #   - "robust": 鲁棒归一化（对极端值不敏感）
        #   - "clip": 简单裁剪到[0,1]
        #   - "none": 不归一化

        # ============ 其他 ============
        out_dir="model",     # 模型保存目录
        verbose=True         # 显示详细信息
    )

    print("\n" + "=" * 80)
    print("训练和推理完成！")
    print("=" * 80)
    print(f"模型保存路径: {results.get('model_path')}")
    if 'output_path' in results:
        print(f"预测结果保存路径: {results.get('output_path')}")
        print(f"测试集样本数: {results.get('n_test')}")

    # 打印性能指标（如果有）
    if results.get('metrics'):
        print("\n验证集性能指标:")
        metrics = results['metrics']
        for key in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            if key in metrics and metrics[key] is not None:
                print(f"  {key}: {metrics[key]:.4f}")
