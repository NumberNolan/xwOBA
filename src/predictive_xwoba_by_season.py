"""
Predictive xwOBAcon per batter per season: for each batter-season, forecast
that season's xwOBAcon using ONLY the prior season's batted-ball profile
(the tuned XGBoost model from tune_predictive_xgboost.py), then line it up
against what Statcast's own xwOBAcon says and what actually happened that
season.

season here = the PREDICTED season (season N+1 in the transition data) --
i.e. the forecast for a batter's 2025 was made using their 2024 profile,
before any 2025 data existed.

Refits the tuned XGBoost on ALL 799 transitions (not just the held-out
2025->2026 test split used for benchmarking) since this is a production
output, not a held-out evaluation.
"""
import pandas as pd
import xgboost as xgb

TRANSITIONS = "data/predictive_dataset.csv"
SEASON_STATS = "results/batter_xwoba_by_season.csv"
OUT = "results/predictive_xwoba_by_season.csv"

FEATURES = ["bat_speed_mean", "attack_direction_mean", "launch_speed_mean", "launch_angle_mean"]

# Best hyperparameters found by tune_predictive_xgboost.py (3,000-config RandomizedSearchCV)
TUNED_PARAMS = dict(
    n_estimators=578, max_depth=4, learning_rate=0.03221961227560633,
    subsample=0.5967332311822868, colsample_bytree=0.6041664773013142,
    min_child_weight=2, gamma=0.016090114763485985,
    reg_alpha=0.009342607478572521, reg_lambda=4.356262418008069,
    random_state=42, n_jobs=-1, tree_method="hist", verbosity=0,
)

print("Loading transitions dataset...")
trans = pd.read_csv(TRANSITIONS)
print(f"Rows (batter-transitions): {len(trans)}")

print("Refitting tuned XGBoost on the FULL transitions dataset...")
model = xgb.XGBRegressor(**TUNED_PARAMS)
model.fit(trans[FEATURES].values, trans["target_next_actual_wobacon"].values)

trans["predictive_xwobacon"] = model.predict(trans[FEATURES].values)

# Pull in the PREDICTED season's own Statcast xwOBAcon (the transitions file only
# carries the SOURCE season's statcast_xwobacon, needed as a model input elsewhere --
# here we want the target season's, for a fair side-by-side).
season_stats = pd.read_csv(SEASON_STATS)[["batter", "season", "n_bbe", "statcast_xwobacon", "actual_wobacon"]]
season_stats = season_stats.rename(columns={"season": "season_n_plus_1", "n_bbe": "n_bbe_predicted_season"})

# trans already carries statcast_xwobacon/actual_wobacon for the SOURCE season (season_n);
# suffix those so the predicted-season versions we're merging in keep the plain names.
out = trans.rename(columns={
    "statcast_xwobacon": "statcast_xwobacon_source_season",
    "actual_wobacon": "actual_wobacon_source_season",
    "n_bbe": "n_bbe_source_season",
}).merge(season_stats, on=["batter", "season_n_plus_1"])

out = out.rename(columns={
    "season_n_plus_1": "season",
    "n_bbe_predicted_season": "n_bbe",
})

final = out[["batter", "player_name", "season", "n_bbe", "predictive_xwobacon", "statcast_xwobacon", "actual_wobacon"]].copy()
final["predictive_xwobacon"] = final["predictive_xwobacon"].round(4)
final = final.sort_values(["season", "predictive_xwobacon"], ascending=[True, False])

final.to_csv(OUT, index=False)
print(f"Saved {len(final)} player-season predictions to {OUT}")
print("\nRows per (predicted) season:")
print(final.groupby("season").size())

print("\n=== Top 10 predictive xwOBAcon per season (min 100 BBE in predicted season) ===")
qualified = final[final["n_bbe"] >= 100]
for season, grp in qualified.groupby("season"):
    print(f"\n-- {season} --")
    print(grp.sort_values("predictive_xwobacon", ascending=False).head(10)[
        ["player_name", "n_bbe", "predictive_xwobacon", "statcast_xwobacon", "actual_wobacon"]
    ].to_string(index=False))
