"""
Hyperparameter tuning for Random Forest on the xwOBA project.

Same train/test split as run_pipeline.py (test_size=0.2, random_state=42 on
model_data.parquet) and the same stepwise-selected feature set:
  launch_speed, launch_angle, spray_angle, stand_L, swing_path_tilt

RandomizedSearchCV: 5-fold CV, 40 iterations, scoring = neg RMSE. bootstrap is
fixed True (so max_samples is always a valid param -- avoids the conditional-
parameter trap of sampling max_samples when bootstrap=False). refit=True means
the winning config is automatically refit on the full training set.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

RNG = 42
DATA = "data/model_data.parquet"
LOG_FILE = "results/logs/tuning_rf_log.txt"
N_ITER = 40
CV_FOLDS = 5

TARGET = "woba_value"
RF_FEATURES = ["launch_speed", "launch_angle", "spray_angle", "stand_L", "swing_path_tilt"]

PRIOR_RF = {"test_rmse": 0.37210070471714896, "test_mae": 0.2358701630034039, "test_r2": 0.5780062704860707, "test_corr": 0.7605615001895862}
BENCH_PRIOR_RMSE = 0.4301347648318181
BEST_TUNED_SO_FAR = {"name": "lightgbm_tuned", "test_rmse": 0.36611073499838986}


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

    X_train = train_df[RF_FEATURES].reset_index(drop=True).values
    y_train = train_df[TARGET].reset_index(drop=True).values
    X_test = test_df[RF_FEATURES].reset_index(drop=True).values
    y_test = test_df[TARGET].reset_index(drop=True)

    param_dist = {
        "n_estimators": randint(200, 700),
        "max_depth": [None, 6, 8, 10, 12, 15, 20, 25],
        "min_samples_split": randint(2, 40),
        "min_samples_leaf": randint(1, 40),
        "max_features": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, "sqrt", "log2"],
        "max_samples": uniform(0.5, 0.5),  # 0.5 - 1.0, only used since bootstrap=True
    }

    base = RandomForestRegressor(bootstrap=True, random_state=RNG, n_jobs=1)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=N_ITER, cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error", random_state=RNG,
        n_jobs=-1, refit=True, verbose=1,
    )

    log(f"=== Random Forest: RandomizedSearchCV ({N_ITER} iters x {CV_FOLDS}-fold) ===")
    t0 = time.time()
    search.fit(X_train, y_train)
    log(f"Random Forest search done in {time.time()-t0:.1f}s")
    log(f"Best CV RMSE: {-search.best_score_:.5f}")
    log(f"Best params: {search.best_params_}")

    metrics = evaluate(search.best_estimator_, X_test, y_test)
    log(f"Random Forest TUNED test metrics: {metrics}")
    log(f"Random Forest prior (untuned) test metrics: {PRIOR_RF}")

    result = {
        "random_forest_tuned": {
            "features": RF_FEATURES, "best_params": search.best_params_,
            "cv_rmse": float(-search.best_score_), **metrics,
        }
    }
    with open("results/tuned_rf_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    log("=== SUMMARY (test RMSE, lower is better) ===")
    log(f"  statcast xwOBA benchmark      : {BENCH_PRIOR_RMSE:.5f}")
    log(f"  lightgbm tuned (current best) : {BEST_TUNED_SO_FAR['test_rmse']:.5f}")
    log(f"  random_forest untuned         : {PRIOR_RF['test_rmse']:.5f}")
    log(f"  random_forest TUNED           : {metrics['test_rmse']:.5f}")
    log("DONE.")


if __name__ == "__main__":
    main()
