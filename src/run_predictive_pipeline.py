"""
PREDICTIVE xwOBAcon: forward + backward stepwise feature selection across the
same 9 model families, but now at the batter-SEASON level. Features are a
batter's season-N mean of each of the 9 candidates; the target is their
ACTUAL wOBA-on-contact in season N+1. This directly optimizes for projecting
a hitter forward, unlike the earlier explanatory (same-season) model.

Temporal train/test split (not random) since this is inherently a forecasting
task: train on the 2023->2024 and 2024->2025 transitions (511 rows), test on
the held-out, chronologically later 2025->2026 transition (288 rows).

Hyperparameters are deliberately conservative/shallow throughout -- ~400 rows
per CV fold can't support the same tree depth / estimator counts as the
396k-row explanatory model without badly overfitting.

Benchmarks (scored on the same 288 test rows):
  - Statcast xwOBAcon (season N) predicting actual wOBAcon (season N+1) directly.
  - Our earlier EXPLANATORY custom xwOBAcon (season N) doing the same, naively.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")

RNG = 42
DATA = "data/predictive_dataset.csv"
LOG_FILE = "results/logs/predictive_pipeline_log.txt"
CV_FOLDS = 5

RAW_FEATURES = [
    "launch_speed", "launch_angle", "spray_angle", "bat_speed", "stand_L",
    "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
]
FEATURES = [f"{c}_mean" for c in RAW_FEATURES]
TARGET = "target_next_actual_wobacon"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def make_pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def get_linear_alphas(X, y):
    Xs = StandardScaler().fit_transform(X)
    lasso_cv = LassoCV(cv=5, random_state=RNG, max_iter=20000).fit(Xs, y)
    ridge_cv = RidgeCV(cv=5).fit(Xs, y)
    enet_cv = ElasticNetCV(cv=5, random_state=RNG, max_iter=20000,
                            l1_ratio=[.1, .3, .5, .7, .9, .95, .99]).fit(Xs, y)
    return {
        "lasso_alpha": lasso_cv.alpha_, "ridge_alpha": ridge_cv.alpha_,
        "enet_alpha": enet_cv.alpha_, "enet_l1_ratio": enet_cv.l1_ratio_,
    }


def build_model_factories(alphas):
    f = {}
    f["linear_regression"] = lambda: make_pipeline(LinearRegression())
    f["lasso"] = lambda: make_pipeline(Lasso(alpha=alphas["lasso_alpha"], max_iter=20000))
    f["ridge"] = lambda: make_pipeline(Ridge(alpha=alphas["ridge_alpha"]))
    f["elasticnet"] = lambda: make_pipeline(ElasticNet(alpha=alphas["enet_alpha"], l1_ratio=alphas["enet_l1_ratio"], max_iter=20000))
    # Shallow / heavily regularized -- ~400 rows/fold, not 300k
    f["xgboost"] = lambda: make_pipeline(xgb.XGBRegressor(
        n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=5,
        random_state=RNG, n_jobs=1, tree_method="hist", verbosity=0))
    f["lightgbm"] = lambda: make_pipeline(lgb.LGBMRegressor(
        n_estimators=120, num_leaves=7, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
        reg_lambda=2.0, random_state=RNG, n_jobs=1, verbosity=-1))
    f["catboost"] = lambda: make_pipeline(cb.CatBoostRegressor(
        iterations=120, depth=3, learning_rate=0.05, l2_leaf_reg=6,
        random_seed=RNG, verbose=False, thread_count=1))
    f["random_forest"] = lambda: make_pipeline(RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=8,
        random_state=RNG, n_jobs=1))
    f["neural_net"] = lambda: make_pipeline(MLPRegressor(
        hidden_layer_sizes=(8,), alpha=0.02, max_iter=500, early_stopping=True,
        n_iter_no_change=15, random_state=RNG))
    return f


def cv_rmse(model_factory, X, y, feature_subset, cv_folds=CV_FOLDS):
    if len(feature_subset) == 0:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RNG)
        errs = []
        for tr_idx, va_idx in kf.split(X):
            pred = np.full(len(va_idx), y.iloc[tr_idx].mean())
            errs.append(mean_squared_error(y.iloc[va_idx], pred))
        return float(np.sqrt(np.mean(errs)))
    Xs = X[feature_subset].values
    yv = y.values
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RNG)
    errs = []
    for tr_idx, va_idx in kf.split(Xs):
        m = model_factory()
        m.fit(Xs[tr_idx], yv[tr_idx])
        pred = m.predict(Xs[va_idx])
        errs.append(mean_squared_error(yv[va_idx], pred))
    return float(np.sqrt(np.mean(errs)))


def forward_path(model_factory, X, y, all_features):
    selected, remaining = [], list(all_features)
    path = [(tuple(), cv_rmse(model_factory, X, y, []))]
    while remaining:
        scores = [(cv_rmse(model_factory, X, y, selected + [f]), f) for f in remaining]
        scores.sort(key=lambda t: t[0])
        best_rmse, best_f = scores[0]
        selected.append(best_f)
        remaining.remove(best_f)
        path.append((tuple(selected), best_rmse))
    return path


def backward_path(model_factory, X, y, all_features):
    selected = list(all_features)
    path = [(tuple(selected), cv_rmse(model_factory, X, y, selected))]
    while len(selected) > 1:
        scores = [(cv_rmse(model_factory, X, y, [x for x in selected if x != f]), f) for f in selected]
        scores.sort(key=lambda t: t[0])
        best_rmse, drop_f = scores[0]
        selected = [x for x in selected if x != drop_f]
        path.append((tuple(selected), best_rmse))
    return path


def best_subset_from_path(path):
    best = min(path, key=lambda t: t[1])
    return list(best[0]), best[1]


def evaluate(preds, y_true):
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))
    corr = float(pearsonr(y_true, preds)[0])
    return {"test_rmse": rmse, "test_mae": mae, "test_r2": r2, "test_corr": corr}


def main():
    open(LOG_FILE, "w").close()
    df = pd.read_csv(DATA)
    log(f"Total batter-transitions: {len(df)}")

    train_df = df[df["pair"].isin(["2023->2024", "2024->2025"])].reset_index(drop=True)
    test_df = df[df["pair"] == "2025->2026"].reset_index(drop=True)
    log(f"Train (2023->24 + 2024->25): {len(train_df)}  Test (2025->26, held out): {len(test_df)}")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # --- Benchmarks on the held-out test rows ---
    bench_statcast = evaluate(test_df["statcast_xwobacon"].values, y_test)
    bench_custom_explanatory = evaluate(test_df["custom_xwobacon"].values, y_test)
    log(f"BENCHMARK Statcast xwOBAcon (season N) -> actual (N+1): {bench_statcast}")
    log(f"BENCHMARK Custom EXPLANATORY xwOBAcon (season N) -> actual (N+1): {bench_custom_explanatory}")

    log("Tuning linear regularization strengths once on full feature set...")
    alphas = get_linear_alphas(X_train, y_train)
    log(f"Alphas: {alphas}")
    factories = build_model_factories(alphas)

    all_results = {
        "statcast_xwobacon_naive": {"selected_features": ["statcast_xwobacon"], "n_features": 1, **bench_statcast},
        "custom_explanatory_naive": {"selected_features": ["custom_xwobacon"], "n_features": 1, **bench_custom_explanatory},
    }
    selection_paths = {}

    for name, factory in factories.items():
        t0 = time.time()
        log(f"=== {name}: forward selection ===")
        fwd = forward_path(factory, X_train, y_train, FEATURES)
        for feats, rmse in fwd:
            log(f"  forward  n={len(feats)} rmse={rmse:.5f}  {feats}")

        log(f"=== {name}: backward elimination ===")
        bwd = backward_path(factory, X_train, y_train, FEATURES)
        for feats, rmse in bwd:
            log(f"  backward n={len(feats)} rmse={rmse:.5f}  {feats}")

        fwd_feats, fwd_rmse = best_subset_from_path(fwd)
        bwd_feats, bwd_rmse = best_subset_from_path(bwd)
        if fwd_rmse <= bwd_rmse:
            chosen_feats, chosen_rmse, direction = fwd_feats, fwd_rmse, "forward"
        else:
            chosen_feats, chosen_rmse, direction = bwd_feats, bwd_rmse, "backward"

        log(f"{name}: SELECTED ({direction}) {chosen_feats} cv_rmse={chosen_rmse:.5f} "
            f"[{time.time()-t0:.1f}s]")

        selection_paths[name] = {
            "forward": [{"features": list(f), "cv_rmse": r} for f, r in fwd],
            "backward": [{"features": list(f), "cv_rmse": r} for f, r in bwd],
            "selected_features": chosen_feats, "selected_direction": direction,
            "selected_cv_rmse": chosen_rmse,
        }

        final_model = factory()
        final_model.fit(X_train[chosen_feats].values, y_train.values)
        preds = final_model.predict(X_test[chosen_feats].values)
        metrics = evaluate(preds, y_test)
        all_results[name] = {
            "selected_features": chosen_feats, "selected_direction": direction,
            "n_features": len(chosen_feats), **metrics,
        }
        log(f"{name}: TEST {metrics}")

        with open("results/predictive_selection_paths.json", "w") as fp:
            json.dump(selection_paths, fp, indent=2)
        with open("results/predictive_model_results.json", "w") as fp:
            json.dump(all_results, fp, indent=2)

    log("=== FINAL COMPARISON (test RMSE predicting NEXT season's actual wOBAcon, lower is better) ===")
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["test_rmse"])
    for name, r in ranked:
        log(f"  {name:28s} rmse={r['test_rmse']:.5f} mae={r['test_mae']:.5f} "
            f"r2={r['test_r2']:.5f} corr={r['test_corr']:.5f} n_feat={r['n_features']}")
    log("DONE.")


if __name__ == "__main__":
    main()
