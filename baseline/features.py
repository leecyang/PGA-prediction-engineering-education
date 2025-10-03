from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from .data_loader import detect_id_column, detect_time_column


def to_datetime_safe(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=s.index)


def compute_stat_features(df: pd.DataFrame, group_col: str | None, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    exclude_set = set(exclude_cols or [])
    if group_col:
        exclude_set.add(group_col)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in exclude_set]
    if not num_cols:
        return pd.DataFrame(index=df.index)
    if group_col and group_col in df.columns:
        g = df.groupby(group_col)[num_cols]
        agg = g.agg(["mean", "std", "min", "max", "median"])
        agg.columns = ["__".join(c) for c in agg.columns]
        agg = agg.reset_index()
        return agg
    else:
        # 全局统计：为每个数值列生成一行包含多种统计量的平铺特征
        flat = {}
        for c in num_cols:
            s = df[c].agg(["mean", "std", "min", "max", "median"])
            for k, v in s.items():
                flat[f"{c}__{k}"] = v
        out = pd.DataFrame([flat])
        out.insert(0, "group", "global")
        return out


def compute_recent_features(df: pd.DataFrame, group_col: str | None, time_col: str | None, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    exclude_set = set(exclude_cols or [])
    if group_col:
        exclude_set.add(group_col)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in exclude_set]
    if not num_cols:
        return pd.DataFrame(index=df.index)
    if time_col and time_col in df.columns:
        df_sorted = df.copy()
        dt = to_datetime_safe(df_sorted[time_col])
        df_sorted["__dt__"] = dt
        df_sorted = df_sorted.sort_values("__dt__")
        if group_col and group_col in df_sorted.columns:
            last = df_sorted.groupby(group_col).tail(1)
            out = last[[group_col] + num_cols].copy()
            out.columns = [out.columns[0]] + [f"recent__{c}" for c in num_cols]
            return out
        else:
            out = df_sorted[num_cols].tail(1).copy()
            out.columns = [f"recent__{c}" for c in num_cols]
            out.insert(0, "group", "global")
            return out
    else:
        if group_col and group_col in df.columns:
            last = df.groupby(group_col).tail(1)
            out = last[[group_col] + num_cols].copy()
            out.columns = [out.columns[0]] + [f"recent__{c}" for c in num_cols]
            return out
        out = df[num_cols].tail(1).copy()
        out.columns = [f"recent__{c}" for c in num_cols]
        out.insert(0, "group", "global")
        return out


def compute_time_features(df: pd.DataFrame, group_col: str | None, time_col: str | None) -> pd.DataFrame:
    if not (time_col and time_col in df.columns):
        return pd.DataFrame(index=df.index)
    s = to_datetime_safe(df[time_col])
    f = pd.DataFrame({
        "dayofweek": s.dt.dayofweek,
        "hour": s.dt.hour,
        "month": s.dt.month,
    })
    if group_col and group_col in df.columns:
        df2 = pd.concat([df[[group_col]].reset_index(drop=True), f.reset_index(drop=True)], axis=1)
        g = df2.groupby(group_col)
        agg = g.agg(["mean", "median", "nunique"])
        agg.columns = ["time__" + "__".join(c) for c in agg.columns]
        agg = agg.reset_index()
        return agg
    else:
        agg = f.agg(["mean", "median", "nunique"]).T
        agg.index = ["global"]
        agg.reset_index(inplace=True)
        agg.rename(columns={"index": "group"}, inplace=True)
        return agg


def build_static_features(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    id_col = detect_id_column(df)
    time_col = detect_time_column(df)
    stat_f = compute_stat_features(df, id_col, exclude_cols)
    recent_f = compute_recent_features(df, id_col, time_col, exclude_cols)
    time_f = compute_time_features(df, id_col, time_col)
    # 统一使用单键进行合并：优先使用识别到的实体ID，否则使用全局"group"
    key_col = id_col if id_col is not None else "group"
    base = None
    for part in [stat_f, recent_f, time_f]:
        if part is None or part.empty:
            continue
        if base is None:
            base = part
        else:
            base = base.merge(part, on=key_col, how="outer")
    if base is None:
        base = pd.DataFrame()
    if id_col and id_col in base.columns:
        base = base.rename(columns={id_col: "entity_id"})
    elif "group" in base.columns:
        base = base.rename(columns={"group": "entity_id"})
    else:
        base["entity_id"] = 0
    return base