from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from .utils import ensure_dir, save_json
from .download_data import download_kaggle_dataset
from .data_loader import load_concat_csvs, detect_target_column, detect_id_column, detect_time_column
from .features import build_static_features
from .models import optimize_model
from .evaluation import regression_metrics, classification_metrics, calibration_plot
from .explain import shap_summary, rf_feature_importance


def build_X_y(features: pd.DataFrame, raw: pd.DataFrame, target: str, task: str):
    # 特征仅保留数值列并处理缺失值
    X_all = features.select_dtypes(include=[np.number])
    # 更强的缺失值处理：先用0填充，再用均值填充剩余的NaN
    X_all = X_all.fillna(0.0)
    X_all = X_all.fillna(X_all.mean())
    # 如果仍有NaN（例如全为NaN的列），用0填充
    X_all = X_all.fillna(0.0)

    # 明确移除目标及其派生特征，防止泄露
    leak_cols = {
        target,
        f"recent__{target}",
        f"{target}__mean",
        f"{target}__std",
        f"{target}__min",
        f"{target}__max",
        f"{target}__median",
        "entity_id",
    }
    X_all = X_all.drop(columns=[c for c in leak_cols if c in X_all.columns], errors="ignore")

    if target in features.columns:
        # 目标已在特征表中，直接对齐
        y = features[target]
        X = X_all
    elif target in raw.columns:
        # 将原始数据的目标按实体ID聚合到特征的 entity_id 上
        id_raw = detect_id_column(raw)
        if "entity_id" in features.columns and id_raw and id_raw in raw.columns:
            # 按ID对目标取均值（或可改为最近值）
            y_df = raw[[id_raw, target]].groupby(id_raw, as_index=False)[target].mean()
            y_df = y_df.rename(columns={id_raw: "entity_id"})
            merged = features[["entity_id"]].merge(y_df, on="entity_id", how="inner")
            y = merged[target]
            # 对齐X到有标签的行
            X = features.loc[merged.index, X_all.columns]
        else:
            # 无法按ID对齐时，退化为长度截断对齐（不推荐但保证可运行）
            y = raw[target].reset_index(drop=True)
            X = X_all.iloc[: len(y)].copy()
    else:
        raise ValueError(f"未找到目标列: {target}")

    # 再次确保X中没有NaN值
    X = X.fillna(0.0)
    
    if task == "classification" and y.dtype.kind in {"f", "i"}:
        bins = np.quantile(y, [0.0, 0.5, 1.0])
        y = pd.cut(y, bins=np.unique(bins), include_lowest=True, labels=[0, 1]).astype(int)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--features_file", type=str, default="")
    parser.add_argument("--task", type=str, choices=["regression", "classification"], default="regression")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--models", type=str, nargs="*", default=["ridge", "lasso", "rf", "xgb", "lgbm", "svr"])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output files to avoid overwriting")
    args = parser.parse_args()

    # 默认输出与数据目录定位到 baseline 子目录
    pkg_dir = Path(__file__).parent
    if not args.data_dir or args.data_dir == "data":
        args.data_dir = str(pkg_dir / "data")

    if args.download:
        p = download_kaggle_dataset("lyyzka/grades-and-tests")
        data_dir = Path(p)
    else:
        data_dir = ensure_dir(Path(args.data_dir))

    raw = load_concat_csvs(data_dir)
    # 先检测目标列，再构建特征以便排除目标
    target = args.target or detect_target_column(raw)
    if not target:
        raise ValueError("无法自动检测目标列，请通过 --target 指定")
    if args.features_file:
        features = pd.read_csv(args.features_file)
    else:
        # 构建静态特征时排除目标列，避免生成目标的统计/最近值特征
        features = build_static_features(raw, exclude_cols=[target])
    
    # 生成带时间戳的输出目录，避免不同运行的文件混合
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    run_dir_name = f"{args.task}_{target}_{timestamp}{suffix}"
    out_dir = ensure_dir(pkg_dir / "outputs" / run_dir_name)

    X, y = build_X_y(features, raw, target, args.task)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state,
        stratify=y if args.task == "classification" else None,
    )

    # 生成带时间戳的输出文件名，避免覆盖
    result_filename = f"results.json"

    res_all = {}
    for name in args.models:
        model, best_params, best_score = optimize_model(args.task, name, X_train, y_train, n_trials=args.n_trials, random_state=args.random_state)
        if args.task == "classification":
            y_pred = model.predict(X_test)
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                    if y_prob is not None and y_prob.ndim == 2:
                        y_prob = y_prob[:, 1]
                except Exception:
                    y_prob = None
            metrics = classification_metrics(y_test, y_pred, y_prob)
            if args.calibrate and y_prob is not None:
                try:
                    calibrated = CalibratedClassifierCV(model, cv=5, method="isotonic")
                    calibrated.fit(X_train, y_train)
                    y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
                    brier = calibration_plot(y_test, y_prob_cal, str(out_dir / f"calibration_{name}.png"))
                    metrics["Brier"] = brier
                except Exception:
                    pass
        else:
            y_pred = model.predict(X_test)
            metrics = regression_metrics(y_test, y_pred)

        res_all[name] = {
            "best_params": best_params,
            "cv_best": best_score,
            "metrics": metrics,
        }

        fnames = X.columns.tolist()
        shap_summary(model, X_test, fnames, str(out_dir / f"shap_{name}.png"))
        rf_feature_importance(model, fnames, str(out_dir / f"importance_{name}.png"))

    # 保存结果到独立目录的results.json文件
    save_json(res_all, out_dir / result_filename)
    print(f"Results saved to: {out_dir / result_filename}")
    print(f"All outputs saved to directory: {out_dir}")


if __name__ == "__main__":
    main()