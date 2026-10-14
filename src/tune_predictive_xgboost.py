"""
Tune XGBoost for the PREDICTIVE xwOBAcon model (Part 2: season-N features ->
season-N+1 actual wOBAcon), fixed to the user-specified feature set:
  bat_speed_mean, attack_direction_mean, launch_speed_mean, launch_angle_mean

Same temporal split as run_predictive_pipeline.py: train on the 2023->2024 +
2024->2025 transitions (511 rows), test on the held-out, later 2025->2026
transition (288 rows). RandomizedSearchCV with 10-fold CV (dataset is tiny,
so fits are cheap -- more folds means a more stable estimate) and refit=True
so the winning config is automatically refit on the full 511-row training set.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform, loguniform, pearsonr
import xgboost as xgb

warnings.filterwarnings("ignore")

RNG = 42
DATA = "data/predictive_dataset.csv"
LOG_FILE = "results/logs/tuning_predictive_xgboost_log.txt"
N_ITER = 3000
CV_FOLDS = 10

FEATURES = ["bat_speed_mean", "attack_direction_mean", "launch_speed_mean", "launch_angle_mean"]
TARGET = "target_next_actual_wobacon"

# Prior results, for comparison
PRIOR_XGB_STEPWISE = {"test_rmse": 0.042143058755971116, "test_mae": 0.03358713123765257, "test_r2": 0.3941940991156726, "test_corr": 0.635196698079777}
BENCH_STATCAST_NAIVE = {"test_rmse": 0.047670069080200744, "test_r2": 0.22487284003690822, "test_corr": 0.595476770755478}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "test_rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "test_mae": float(mean_absolute_error(y, preds)),
        "test_r2": float(r2_score(y, preds)),
        "test_corr": float(pearsonr(y, preds)[0]),
    }


def main():
    open(LOG_FILE, "w").close()
    df = pd.read_csv(DATA)

    train_df = df[df["pair"].isin(["2023->2024", "2024->2025"])].reset_index(drop=True)
    test_df = df[df["pair"] == "2025->2026"].reset_index(drop=True)
    log(f"Train: {len(train_df)}  Test (held-out, later): {len(test_df)}")
    log(f"Features (fixed, as specified): {FEATURES}")

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    param_dist = {
        "n_estimators": randint(50, 1000),
        "max_depth": randint(2, 8),
        "learning_rate": loguniform(0.002, 0.3),
        "subsample": uniform(0.4, 0.6),
        "colsample_bytree": uniform(0.4, 0.6),
        "min_child_weight": randint(1, 50),
        "gamma": uniform(0, 8),
        "reg_alpha": loguniform(1e-4, 30),
        "reg_lambda": loguniform(1e-4, 30),
    }

    base = xgb.XGBRegressor(random_state=RNG, n_jobs=1, tree_method="hist", verbosity=0)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=N_ITER, cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error", random_state=RNG,
        n_jobs=-1, refit=True, verbose=1,
    )

    log(f"=== RandomizedSearchCV: {N_ITER} iters x {CV_FOLDS}-fold CV ===")
    t0 = time.time()
    search.fit(X_train, y_train)
    log(f"Search done in {time.time()-t0:.1f}s")
    log(f"Best CV RMSE: {-search.best_score_:.5f}")
    log(f"Best params: {search.best_params_}")

    metrics = evaluate(search.best_estimator_, X_test, y_test)
    log(f"TUNED test metrics: {metrics}")
    log(f"Prior stepwise-selected XGBoost (different features, untuned): {PRIOR_XGB_STEPWISE}")
    log(f"Statcast naive benchmark: {BENCH_STATCAST_NAIVE}")

    result = {
        "predictive_xgboost_tuned_fixed_features": {
            "features": FEATURES, "best_params": search.best_params_,
            "cv_rmse": float(-search.best_score_), **metrics,
        }
    }
    with open("results/tuned_predictive_xgboost_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    log("=== SUMMARY (test RMSE predicting next season's actual wOBAcon) ===")
    log(f"  statcast naive benchmark              : {BENCH_STATCAST_NAIVE['test_rmse']:.5f}")
    log(f"  xgboost stepwise-selected (untuned)    : {PRIOR_XGB_STEPWISE['test_rmse']:.5f}")
    log(f"  xgboost fixed-features TUNED           : {metrics['test_rmse']:.5f}")
    log("DONE.")


if __name__ == "__main__":
    main()
