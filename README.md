# PGA Prediction for Engineering Education

## 项目概述

本项目是一个用于工程教育中学生成绩预测的机器学习基线系统。通过分析学生的测验成绩和相关特征，预测学生的总成绩表现。

## 功能特性

- **多任务支持**: 支持分类和回归两种预测任务
- **多模型对比**: 集成了 Random Forest、XGBoost、LightGBM 等主流机器学习模型
- **自动化特征工程**: 包含特征选择、缺失值处理、编码等预处理步骤
- **超参数优化**: 使用 Optuna 进行自动化超参数调优
- **模型解释**: 提供 SHAP 值分析和特征重要性可视化
- **交叉验证**: 5折交叉验证确保模型稳定性

## 项目结构

```
baseline/
├── README.md              # 详细使用说明
├── data/                  # 数据文件目录
│   ├── 常规班成绩汇总.CSV
│   ├── 常规班测验.CSV
│   ├── 挑战班成绩汇总.CSV
│   └── 挑战班测验.CSV
├── outputs/               # 模型输出结果
├── train_baselines.py     # 主训练脚本
├── models.py             # 模型定义和优化
├── features.py           # 特征工程
├── evaluation.py         # 模型评估
├── explain.py            # 模型解释
└── data_loader.py        # 数据加载
```

## 快速开始

### 环境配置

1. 克隆项目：
```bash
git clone https://github.com/leecyang/PGA-prediction-engineering-education.git
cd PGA-prediction-engineering-education
```

2. 创建虚拟环境：
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

### 运行示例

#### 分类任务
```bash
python -m baseline.train_baselines --task classification --target 总成绩 --models rf xgb lgbm --n_trials 10
```

#### 回归任务
```bash
python -m baseline.train_baselines --task regression --target 总成绩 --models rf xgb lgbm --n_trials 10
```

### 参数说明

- `--task`: 任务类型 (`classification` 或 `regression`)
- `--target`: 目标变量列名
- `--models`: 模型列表 (`rf`, `xgb`, `lgbm`, `ridge`, `lasso`, `svr`)
- `--n_trials`: Optuna 优化试验次数
- `--output_suffix`: 输出目录后缀

## 输出结果

运行完成后，结果保存在 `baseline/outputs/` 目录下，包含：

- `results.json`: 模型性能指标和最优参数
- `importance_{model}.png`: 特征重要性图表
- `shap_{model}.png`: SHAP 值分析图表（树模型）

## 性能指标

### 分类任务
- Accuracy: 准确率
- F1_macro: 宏平均F1分数
- AUC: ROC曲线下面积
- Brier Score: 布里尔分数

### 回归任务
- RMSE: 均方根误差
- MAE: 平均绝对误差
- R²: 决定系数

## 注意事项

1. **数据泄漏检查**: 系统会自动排除目标变量及其衍生特征，避免数据泄漏
2. **依赖管理**: XGBoost 和 LightGBM 为可选依赖，缺失时会自动降级到其他模型
3. **交叉验证**: 默认使用5折交叉验证，分类任务使用分层抽样

## 故障排除

### 常见问题

1. **完美分数问题**: 如果出现 AUC=1.0 等完美分数，可能是特征与目标高度相关，建议检查特征选择
2. **依赖缺失**: 如果 XGBoost/LightGBM 安装失败，系统会自动使用 Random Forest 等替代模型
3. **内存不足**: 对于大数据集，可以减少 `n_trials` 参数或使用更简单的模型

### 联系方式

- 邮箱: yangyangli0426@gmail.com
- 协作者: lyyzka

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。