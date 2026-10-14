"""
Hyperparameter tuning for XGBoost and LightGBM on the xwOBA project.

Uses the SAME train/test split as run_pipeline.py (test_size=0.2, random_state=42
on model_data.parquet) so results are directly comparable to the earlier run, and
the SAME feature sets that stepwise selection already chose for each model:
  - xgboost:  launch_speed, launch_angle, spray_angle, stand_L
  - lightgbm: launch_speed, launch_angle, spray_angle, stand_L, swing_path_tilt

RandomizedSearchCV (5-fold CV, 60 iterations, scoring = neg RMSE) over each
model's native hyperparameters; refit=True means the winning config is
automatically refit on the full training set, then scored once on the held-out
test set for a fair before/after comparison.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform, loguniform
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

RNG = 42
DATA = "data/model_data.parquet"
LOG_FILE = "results/logs/tuning_log.txt"
N_ITER = 60
CV_FOLDS = 5

TARGET = "woba_value"
BENCHMARK = "estimated_woba_using_speedangle"

# Feature sets chosen by the earlier stepwise-selection run (run_pipeline.py)
XGB_FEATURES = ["launch_speed", "launch_angle", "spray_angle", "stand_L"]
LGB_FEATURES = ["launch_speed", "launch_angle", "spray_angle", "stand_L", "swing_path_tilt"]

# Baselines from the earlier (untuned) run, for comparison
PRIOR = {
    "xgboost":  {"test_rmse": 0.3792300533900804, "test_mae": 0.266720574621206, "test_r2": 0.5616807875899338, "test_corr": 0.7496588968790765},
    "lightgbm": {"test_rmse": 0.372115866449456,  "test_mae": 0.255315141058689, "test_r2": 0.5779718804070857, "test_corr": 0.7605248852217769},
}
BENCH_PRIOR = {"test_rmse": 0.4301347648318181, "test_mae": 0.3015317799718632, "test_r2": 0.43611037679298814, "test_corr": 0.6604815198516275}
BEST_PRIOR_OVERALL = {"name": "random_forest", "test_rmse": 0.37210070471714896}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    corr = float(pearsonr(y_test, preds)[0])
    return {"test_rmse": rmse, "test_mae": mae, "test_r2": r2, "test_corr": corr}


def main():
    open(LOG_FILE, "w").close()
    log("Loading modeling dataset...")
    df = pd.read_parquet(DATA)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RNG)
    log(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    y_train = train_df[TARGET].reset_index(drop=True)
    y_test = test_df[TARGET].reset_index(drop=True)

    results = {}

    # ------------------------------------------------------------------ XGBoost
    log("=== XGBoost: RandomizedSearchCV ===")
    X_train_xgb = train_df[XGB_FEATURES].reset_index(drop=True).values
    X_test_xgb = test_df[XGB_FEATURES].reset_index(drop=True).values

    xgb_param_dist = {
        "n_estimators": randint(200, 1200),
        "max_depth": randint(3, 10),
        "learning_rate": loguniform(0.005, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 20),
        "gamma": uniform(0, 5),
        "reg_alpha": loguniform(1e-3, 10),
        "reg_lambda": loguniform(1e-3, 10),
    }

    xgb_base = xgb.XGBRegressor(random_state=RNG, n_jobs=1, tree_method="hist", verbosity=0)
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_param_dist, n_iter=N_ITER, cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error", random_state=RNG,
        n_jobs=-1, refit=True, verbose=1,
    )
    t0 = time.time()
    xgb_search.fit(X_train_xgb, y_train.values)
    log(f"XGBoost search done in {time.time()-t0:.1f}s")
    log(f"XGBoost best CV RMSE: {-xgb_search.best_score_:.5f}")
    log(f"XGBoost best params: {xgb_search.best_params_}")

    xgb_metrics = evaluate(xgb_search.best_estimator_, X_test_xgb, y_test)
    log(f"XGBoost TUNED test metrics: {xgb_metrics}")
    log(f"XGBoost prior (untuned) test metrics: {PRIOR['xgboost']}")
    results["xgboost_tuned"] = {
        "features": XGB_FEATURES, "best_params": xgb_search.best_params_,
        "cv_rmse": float(-xgb_search.best_score_), **xgb_metrics,
    }
    with open("results/tuned_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ----------------------------------------------------------------- LightGBM
    log("=== LightGBM: RandomizedSearchCV ===")
    X_train_lgb = train_df[LGB_FEATURES].reset_index(drop=True).values
    X_test_lgb = test_df[LGB_FEATURES].reset_index(drop=True).values

    lgb_param_dist = {
        "n_estimators": randint(200, 1200),
        "num_leaves": randint(15, 255),
        "max_depth": [-1, 4, 5, 6, 7, 8, 10, 12],
        "learning_rate": loguniform(0.005, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_samples": randint(5, 100),
        "reg_alpha": loguniform(1e-3, 10),
        "reg_lambda": loguniform(1e-3, 10),
    }

    lgb_base = lgb.LGBMRegressor(random_state=RNG, n_jobs=1, verbosity=-1)
    lgb_search = RandomizedSearchCV(
        lgb_base, lgb_param_dist, n_iter=N_ITER, cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error", random_state=RNG,
        n_jobs=-1, refit=True, verbose=1,
    )
    t0 = time.time()
    lgb_search.fit(X_train_lgb, y_train.values)
    log(f"LightGBM search done in {time.time()-t0:.1f}s")
    log(f"LightGBM best CV RMSE: {-lgb_search.best_score_:.5f}")
    log(f"LightGBM best params: {lgb_search.best_params_}")

    lgb_metrics = evaluate(lgb_search.best_estimator_, X_test_lgb, y_test)
    log(f"LightGBM TUNED test metrics: {lgb_metrics}")
    log(f"LightGBM prior (untuned) test metrics: {PRIOR['lightgbm']}")
    results["lightgbm_tuned"] = {
        "features": LGB_FEATURES, "best_params": lgb_search.best_params_,
        "cv_rmse": float(-lgb_search.best_score_), **lgb_metrics,
    }
    with open("results/tuned_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ------------------------------------------------------------------ Summary
    log("=== SUMMARY (test RMSE, lower is better) ===")
    log(f"  statcast xwOBA benchmark  : {BENCH_PRIOR['test_rmse']:.5f}")
    log(f"  random_forest (prior best): {BEST_PRIOR_OVERALL['test_rmse']:.5f}")
    log(f"  xgboost  untuned          : {PRIOR['xgboost']['test_rmse']:.5f}")
    log(f"  xgboost  TUNED            : {xgb_metrics['test_rmse']:.5f}")
    log(f"  lightgbm untuned          : {PRIOR['lightgbm']['test_rmse']:.5f}")
    log(f"  lightgbm TUNED            : {lgb_metrics['test_rmse']:.5f}")
    log("DONE.")


if __name__ == "__main__":
    main()
