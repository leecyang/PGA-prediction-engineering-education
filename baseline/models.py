from __future__ import annotations

import numpy as np
import optuna
from typing import Literal

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception:
    XGBRegressor = None
    XGBClassifier = None

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    LGBMRegressor = None
    LGBMClassifier = None


def _cv(task: Literal["regression", "classification"], y):
    if task == "classification":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return KFold(n_splits=5, shuffle=True, random_state=42)


def _scorer(task: Literal["regression", "classification"]):
    if task == "classification":
        return make_scorer(accuracy_score)
    return make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))


def optimize_model(task: Literal["regression", "classification"], model_name: str, X, y, n_trials: int = 30, random_state: int = 42):
    cv = _cv(task, y)
    scorer = _scorer(task)

    def objective(trial: optuna.Trial):
        if model_name == "ridge":
            alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
            model = Ridge(alpha=alpha, random_state=random_state)
        elif model_name == "lasso":
            alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
            model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
        elif model_name == "svr":
            c = trial.suggest_float("C", 1e-2, 1e3, log=True)
            eps = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
            model = SVR(C=c, epsilon=eps, kernel=kernel)
        elif model_name == "rf":
            if task == "classification":
                n = trial.suggest_int("n_estimators", 50, 600)
                md = trial.suggest_int("max_depth", 2, 20)
                mf = trial.suggest_float("max_features", 0.2, 1.0)
                model = RandomForestClassifier(n_estimators=n, max_depth=md, max_features=mf, random_state=random_state, n_jobs=-1)
            else:
                n = trial.suggest_int("n_estimators", 50, 600)
                md = trial.suggest_int("max_depth", 2, 20)
                mf = trial.suggest_float("max_features", 0.2, 1.0)
                model = RandomForestRegressor(n_estimators=n, max_depth=md, max_features=mf, random_state=random_state, n_jobs=-1)
        elif model_name == "xgb":
            if task == "classification":
                if XGBClassifier is None:
                    raise optuna.TrialPruned()
                n = trial.suggest_int("n_estimators", 50, 600)
                md = trial.suggest_int("max_depth", 2, 12)
                lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
                subsample = trial.suggest_float("subsample", 0.5, 1.0)
                colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                model = XGBClassifier(n_estimators=n, max_depth=md, learning_rate=lr, subsample=subsample, colsample_bytree=colsample, objective="binary:logistic", n_jobs=-1, random_state=random_state)
            else:
                if XGBRegressor is None:
                    raise optuna.TrialPruned()
                n = trial.suggest_int("n_estimators", 50, 600)
                md = trial.suggest_int("max_depth", 2, 12)
                lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
                subsample = trial.suggest_float("subsample", 0.5, 1.0)
                colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                model = XGBRegressor(n_estimators=n, max_depth=md, learning_rate=lr, subsample=subsample, colsample_bytree=colsample, objective="reg:squarederror", n_jobs=-1, random_state=random_state)
        elif model_name == "lgbm":
            if task == "classification":
                if LGBMClassifier is None:
                    raise optuna.TrialPruned()
                n = trial.suggest_int("n_estimators", 50, 600)
                md = trial.suggest_int("max_depth", -1, 12)
                lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
                num_leaves = trial.suggest_int("num_leaves", 8, 256)
                model = LGBMClassifier(n_estimators=n, max_depth=md, learning_rate=lr, num_leaves=num_leaves, random_state=random_state, verbose=-1)
            else:
                if LGBMRegressor is None:
                    raise optuna.TrialPruned()
                n = trial.suggest_int("n_estimators", 50, 600)
                md = trial.suggest_int("max_depth", -1, 12)
                lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
                num_leaves = trial.suggest_int("num_leaves", 8, 256)
                model = LGBMRegressor(n_estimators=n, max_depth=md, learning_rate=lr, num_leaves=num_leaves, random_state=random_state, verbose=-1)
        else:
            raise ValueError(f"未知模型: {model_name}")

        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    # 若所有trial均被裁剪（例如缺少XGB/LGBM依赖），优雅回退到默认模型
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        # 构造一个合理的默认模型以继续流程，避免抛错
        if model_name == "ridge":
            model = Ridge()
            best_params = {}
        elif model_name == "lasso":
            model = Lasso(max_iter=10000)
            best_params = {}
        elif model_name == "svr":
            model = SVR()
            best_params = {}
        elif model_name == "rf":
            model = RandomForestClassifier(random_state=random_state, n_jobs=-1) if task == "classification" else RandomForestRegressor(random_state=random_state, n_jobs=-1)
            best_params = {}
        elif model_name == "xgb":
            # xgb不可用时回退为随机森林
            model = RandomForestClassifier(random_state=random_state, n_jobs=-1) if task == "classification" else RandomForestRegressor(random_state=random_state, n_jobs=-1)
            best_params = {"fallback": "rf"}
        elif model_name == "lgbm":
            # lgbm不可用时回退为随机森林
            model = RandomForestClassifier(random_state=random_state, n_jobs=-1) if task == "classification" else RandomForestRegressor(random_state=random_state, n_jobs=-1)
            best_params = {"fallback": "rf"}
        else:
            raise ValueError(f"未知模型: {model_name}")

        # 估计一个基准cv分数用于记录
        try:
            base_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
            best_value = float(np.mean(base_scores))
        except Exception:
            best_value = float("nan")
        model.fit(X, y)
        return model, best_params, best_value
    
    best_params = study.best_params

    if model_name == "ridge":
        model = Ridge(**best_params)
    elif model_name == "lasso":
        model = Lasso(**best_params, max_iter=10000)
    elif model_name == "svr":
        model = SVR(**best_params)
    elif model_name == "rf":
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1, **best_params) if task == "classification" else RandomForestRegressor(random_state=random_state, n_jobs=-1, **best_params)
    elif model_name == "xgb":
        if task == "classification":
            model = XGBClassifier(random_state=random_state, n_jobs=-1, **best_params)
        else:
            model = XGBRegressor(random_state=random_state, n_jobs=-1, **best_params)
    elif model_name == "lgbm":
        model = LGBMClassifier(random_state=random_state, verbose=-1, **best_params) if task == "classification" else LGBMRegressor(random_state=random_state, verbose=-1, **best_params)
    else:
        raise ValueError(f"未知模型: {model_name}")

    model.fit(X, y)
    return model, best_params, study.best_value