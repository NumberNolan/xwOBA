"""
Custom xwOBA project: forward + backward stepwise feature selection across
9 model families, then a final held-out comparison against Statcast's own
estimated_woba_using_speedangle.

Candidate features (9): launch_speed, launch_angle, spray_angle, bat_speed,
stand_L, swing_length, attack_angle, attack_direction, swing_path_tilt.
Target: woba_value (actual outcome value of the batted ball).

Design choices (for speed, on ~396k rows x 9 features):
  - Stepwise search runs on a 50k-row random subsample of the TRAINING split,
    scored with 3-fold CV, using cheap/fast hyperparameters ("search" params).
  - Whichever subset (forward-path vs backward-path) achieves the lower CV
    RMSE is selected as that model's final feature set.
  - Each model is then refit on the FULL training split with more thorough
    ("final") hyperparameters, using its selected features, and scored once
    on the held-out test split -- compared row-for-row against Statcast's
    own estimated_woba_using_speedangle.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
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

RNG = 42
DATA = "data/model_data.parquet"
LOG_FILE = "results/logs/pipeline_log.txt"

FEATURES = [
    "launch_speed", "launch_angle", "spray_angle", "bat_speed", "stand_L",
    "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
]
TARGET = "woba_value"
BENCHMARK = "estimated_woba_using_speedangle"

SEARCH_SUBSAMPLE_N = 50_000
CV_FOLDS = 3


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Model factories: (search_params_factory, final_params_factory)
# Each factory returns a fresh sklearn-compatible estimator (wrapped with
# StandardScaler in a Pipeline so linear/MLP models get scaled inputs; scaling
# is a no-op in effect for the tree models' splits).
# ---------------------------------------------------------------------------

def make_pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def get_linear_alphas(X, y):
    """Tune regularization strength once on the full feature set so the
    stepwise search doesn't have to re-tune alpha at every candidate step."""
    Xs = StandardScaler().fit_transform(X)
    lasso_cv = LassoCV(cv=3, random_state=RNG, max_iter=20000, n_jobs=-1).fit(Xs, y)
    ridge_cv = RidgeCV(cv=3).fit(Xs, y)
    enet_cv = ElasticNetCV(cv=3, random_state=RNG, max_iter=20000, n_jobs=-1,
                            l1_ratio=[.1, .3, .5, .7, .9, .95, .99]).fit(Xs, y)
    return {
        "lasso_alpha": lasso_cv.alpha_,
        "ridge_alpha": ridge_cv.alpha_,
        "enet_alpha": enet_cv.alpha_,
        "enet_l1_ratio": enet_cv.l1_ratio_,
    }


def build_model_factories(alphas):
    factories = {}

    factories["linear_regression"] = {
        "search": lambda: make_pipeline(LinearRegression()),
        "final": lambda: make_pipeline(LinearRegression()),
    }
    factories["lasso"] = {
        "search": lambda: make_pipeline(Lasso(alpha=alphas["lasso_alpha"], max_iter=20000)),
        "final": lambda: make_pipeline(Lasso(alpha=alphas["lasso_alpha"], max_iter=50000)),
    }
    factories["ridge"] = {
        "search": lambda: make_pipeline(Ridge(alpha=alphas["ridge_alpha"])),
        "final": lambda: make_pipeline(Ridge(alpha=alphas["ridge_alpha"])),
    }
    factories["elasticnet"] = {
        "search": lambda: make_pipeline(ElasticNet(alpha=alphas["enet_alpha"], l1_ratio=alphas["enet_l1_ratio"], max_iter=20000)),
        "final": lambda: make_pipeline(ElasticNet(alpha=alphas["enet_alpha"], l1_ratio=alphas["enet_l1_ratio"], max_iter=50000)),
    }
    factories["xgboost"] = {
        "search": lambda: make_pipeline(xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=RNG, n_jobs=-1, tree_method="hist", verbosity=0)),
        "final": lambda: make_pipeline(xgb.XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=RNG, n_jobs=-1, tree_method="hist", verbosity=0)),
    }
    factories["lightgbm"] = {
        "search": lambda: make_pipeline(lgb.LGBMRegressor(
            n_estimators=100, num_leaves=31, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=RNG, n_jobs=-1, verbosity=-1)),
        "final": lambda: make_pipeline(lgb.LGBMRegressor(
            n_estimators=500, num_leaves=31, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=RNG, n_jobs=-1, verbosity=-1)),
    }
    factories["catboost"] = {
        "search": lambda: make_pipeline(cb.CatBoostRegressor(
            iterations=100, depth=6, learning_rate=0.1, random_seed=RNG, verbose=False, thread_count=-1)),
        "final": lambda: make_pipeline(cb.CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05, random_seed=RNG, verbose=False, thread_count=-1)),
    }
    factories["random_forest"] = {
        "search": lambda: make_pipeline(RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RNG, n_jobs=-1)),
        "final": lambda: make_pipeline(RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=5, random_state=RNG, n_jobs=-1)),
    }
    factories["neural_net"] = {
        "search": lambda: make_pipeline(MLPRegressor(
            hidden_layer_sizes=(16,), max_iter=300, early_stopping=True,
            n_iter_no_change=10, random_state=RNG)),
        "final": lambda: make_pipeline(MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=800, early_stopping=True,
            n_iter_no_change=15, random_state=RNG)),
    }
    return factories


# ---------------------------------------------------------------------------
# CV scoring helper
# ---------------------------------------------------------------------------

def cv_rmse(model_factory, X, y, feature_subset, cv_folds=CV_FOLDS):
    if len(feature_subset) == 0:
        # baseline: predict the mean
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


# ---------------------------------------------------------------------------
# Forward / backward stepwise search
# ---------------------------------------------------------------------------

def forward_path(model_factory, X, y, all_features):
    selected = []
    remaining = list(all_features)
    path = []  # list of (features_tuple, rmse)
    base_rmse = cv_rmse(model_factory, X, y, [])
    path.append((tuple(), base_rmse))
    while remaining:
        scores = []
        for f in remaining:
            trial = selected + [f]
            rmse = cv_rmse(model_factory, X, y, trial)
            scores.append((rmse, f))
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
        scores = []
        for f in selected:
            trial = [x for x in selected if x != f]
            rmse = cv_rmse(model_factory, X, y, trial)
            scores.append((rmse, f))
        scores.sort(key=lambda t: t[0])
        best_rmse, drop_f = scores[0]
        selected = [x for x in selected if x != drop_f]
        path.append((tuple(selected), best_rmse))
    return path


def best_subset_from_path(path):
    best = min(path, key=lambda t: t[1])
    return list(best[0]), best[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    open(LOG_FILE, "w").close()
    log("Loading modeling dataset...")
    df = pd.read_parquet(DATA)
    log(f"Total rows: {len(df)}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RNG)
    log(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    X_train_full = train_df[FEATURES].reset_index(drop=True)
    y_train_full = train_df[TARGET].reset_index(drop=True)
    X_test = test_df[FEATURES].reset_index(drop=True)
    y_test = test_df[TARGET].reset_index(drop=True)
    bench_test = test_df[BENCHMARK].reset_index(drop=True)

    search_df = train_df.sample(n=min(SEARCH_SUBSAMPLE_N, len(train_df)), random_state=RNG)
    X_search = search_df[FEATURES].reset_index(drop=True)
    y_search = search_df[TARGET].reset_index(drop=True)
    log(f"Search subsample rows: {len(X_search)}")

    log("Tuning linear-model regularization strengths (once, on full feature set)...")
    alphas = get_linear_alphas(X_search[FEATURES], y_search)
    log(f"Alphas: {alphas}")

    factories = build_model_factories(alphas)

    all_results = {}
    selection_paths = {}

    for name, fac in factories.items():
        t0 = time.time()
        log(f"=== {name}: forward selection ===")
        fwd = forward_path(fac["search"], X_search, y_search, FEATURES)
        for feats, rmse in fwd:
            log(f"  forward  n={len(feats):d} rmse={rmse:.5f}  {feats}")

        log(f"=== {name}: backward elimination ===")
        bwd = backward_path(fac["search"], X_search, y_search, FEATURES)
        for feats, rmse in bwd:
            log(f"  backward n={len(feats):d} rmse={rmse:.5f}  {feats}")

        fwd_best_feats, fwd_best_rmse = best_subset_from_path(fwd)
        bwd_best_feats, bwd_best_rmse = best_subset_from_path(bwd)

        if fwd_best_rmse <= bwd_best_rmse:
            chosen_feats, chosen_rmse, chosen_dir = fwd_best_feats, fwd_best_rmse, "forward"
        else:
            chosen_feats, chosen_rmse, chosen_dir = bwd_best_feats, bwd_best_rmse, "backward"

        log(f"{name}: SELECTED ({chosen_dir}) features={chosen_feats} search_cv_rmse={chosen_rmse:.5f}  "
            f"[search phase took {time.time()-t0:.1f}s]")

        selection_paths[name] = {
            "forward": [{"features": list(f), "cv_rmse": r} for f, r in fwd],
            "backward": [{"features": list(f), "cv_rmse": r} for f, r in bwd],
            "selected_features": chosen_feats,
            "selected_direction": chosen_dir,
            "selected_search_cv_rmse": chosen_rmse,
        }

        # Final fit on FULL training data with selected features, more thorough hyperparams
        log(f"{name}: refitting on full training set ({len(X_train_full)} rows) with final hyperparams...")
        t1 = time.time()
        final_model = fac["final"]()
        final_model.fit(X_train_full[chosen_feats].values, y_train_full.values)
        preds = final_model.predict(X_test[chosen_feats].values)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        corr = float(pearsonr(y_test, preds)[0])

        all_results[name] = {
            "selected_features": chosen_feats,
            "selected_direction": chosen_dir,
            "n_features": len(chosen_feats),
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_corr": corr,
        }
        log(f"{name}: TEST rmse={rmse:.5f} mae={mae:.5f} r2={r2:.5f} corr={corr:.5f} "
            f"[final fit took {time.time()-t1:.1f}s]")

        # Save incrementally so partial progress is never lost
        with open("results/selection_paths.json", "w") as f:
            json.dump(selection_paths, f, indent=2)
        with open("results/model_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Benchmark: Statcast's own xwOBA on the same test rows
    bench_rmse = float(np.sqrt(mean_squared_error(y_test, bench_test)))
    bench_mae = float(mean_absolute_error(y_test, bench_test))
    bench_r2 = float(r2_score(y_test, bench_test))
    bench_corr = float(pearsonr(y_test, bench_test)[0])
    all_results["statcast_xwoba_benchmark"] = {
        "selected_features": ["launch_speed", "launch_angle"],
        "selected_direction": "n/a (Statcast's own model)",
        "n_features": 2,
        "test_rmse": bench_rmse,
        "test_mae": bench_mae,
        "test_r2": bench_r2,
        "test_corr": bench_corr,
    }
    log(f"BENCHMARK statcast xwOBA: rmse={bench_rmse:.5f} mae={bench_mae:.5f} r2={bench_r2:.5f} corr={bench_corr:.5f}")

    with open("results/model_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log("=== FINAL COMPARISON TABLE (sorted by test RMSE, lower is better) ===")
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["test_rmse"])
    for name, r in ranked:
        log(f"  {name:28s} rmse={r['test_rmse']:.5f} mae={r['test_mae']:.5f} "
            f"r2={r['test_r2']:.5f} corr={r['test_corr']:.5f} n_feat={r['n_features']}")

    log("DONE.")


if __name__ == "__main__":
    main()
