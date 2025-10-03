from pathlib import Path


def download_kaggle_dataset(dataset: str = "lyyzka/grades-and-tests") -> Path:
    try:
        import kagglehub
    except Exception as e:
        raise RuntimeError(
            "kagglehub 未安装或不可用，请先安装：pip install kagglehub"
        ) from e
    p = kagglehub.dataset_download(dataset)
    return Path(p)