from pathlib import Path
import pandas as pd
import numpy as np


def read_csv_fallback(file: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "cp936"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV读取失败: {file} ; 错误: {last_err}")


def list_csv_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*.csv")] + [p for p in root.rglob("*.CSV")]
    return sorted(files)


def load_concat_csvs(root: Path) -> pd.DataFrame:
    files = list_csv_files(root)
    if not files:
        raise FileNotFoundError(f"未在{root}下找到CSV文件")
    dfs = []
    for f in files:
        df = read_csv_fallback(f)
        df["source_file"] = f.name
        dfs.append(df)
    if not dfs:
        raise RuntimeError("CSV读取失败或为空")
    return pd.concat(dfs, axis=0, ignore_index=True)


def detect_id_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "student_id",
        "sid",
        "user_id",
        "uid",
        "id",
        "序号",
        "学号",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_time_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "timestamp",
        "time",
        "date",
        "datetime",
        "event_time",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_target_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "final_grade",
        "grade",
        "score",
        "label",
        "target",
        "总成绩",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        return num_cols[-1]
    return None