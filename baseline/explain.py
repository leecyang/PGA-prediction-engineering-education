from __future__ import annotations

import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in ["xgb", "lgbm", "randomforest", "extraTrees", "gbm"]) or hasattr(model, "feature_importances_")


def shap_summary(model, X, feature_names, out_path: str):
    if not is_tree_model(model):
        return False
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 8))
    try:
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    except Exception:
        shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def rf_feature_importance(model, feature_names, out_path: str):
    if not hasattr(model, "feature_importances_"):
        return False
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    
    # 只显示前20个最重要的特征，避免标签过于密集
    top_n = min(20, len(feature_names))
    top_indices = order[:top_n]
    top_names = [feature_names[i] for i in top_indices]
    top_vals = importances[top_indices]
    
    # 创建更大的图形以容纳标签
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    
    # 水平条形图，从上到下排列（最重要的在顶部）
    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_vals[::-1])
    
    # 设置y轴标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
    
    # 在条形图上添加数值标签
    for i, (bar, val) in enumerate(zip(bars, top_vals[::-1])):
        ax.text(bar.get_width() + max(top_vals) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True