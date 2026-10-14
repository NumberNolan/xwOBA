"""
Build the dataset for a PREDICTIVE xwOBAcon model: given a batter's season-N
batted-ball profile (mean of the 9 candidate features across their batted
balls that season), predict their ACTUAL wOBA-on-contact in season N+1.

This is a different unit of analysis than the explanatory model (run_pipeline.py):
one row per batter per season-TRANSITION, not one row per pitch. Same 100-BBE
qualifying threshold (both seasons) as the earlier stability check, so the
season-pair sample sizes match exactly (215 / 296 / 288).
"""
import pandas as pd
import numpy as np

EVENTS = "data/model_data.parquet"
SEASON_STATS = "results/batter_xwoba_by_season.csv"
OUT = "data/predictive_dataset.csv"
MIN_BBE = 100

FEATURES = [
    "launch_speed", "launch_angle", "spray_angle", "bat_speed", "stand_L",
    "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
]

print("Loading event-level modeling dataset...")
events = pd.read_parquet(EVENTS)

print("Aggregating to one row per batter per season (mean of each feature)...")
agg = events.groupby(["batter", "game_year"], as_index=False)[FEATURES].mean()
agg = agg.rename(columns={"game_year": "season"})
agg.columns = ["batter", "season"] + [f"{c}_mean" for c in FEATURES]

season_stats = pd.read_csv(SEASON_STATS)  # batter, player_name, season, n_bbe, custom_xwobacon, actual_wobacon, statcast_xwobacon
full = season_stats.merge(agg, on=["batter", "season"])
print(f"Batter-seasons with aggregated features: {len(full)}")

seasons = sorted(full["season"].unique())
pairs = list(zip(seasons[:-1], seasons[1:]))
print(f"Season pairs: {pairs}")

feature_cols = [f"{c}_mean" for c in FEATURES]
rows = []
for y1, y2 in pairs:
    left = full[(full["season"] == y1) & (full["n_bbe"] >= MIN_BBE)]
    right = full[(full["season"] == y2) & (full["n_bbe"] >= MIN_BBE)][
        ["batter", "actual_wobacon"]
    ].rename(columns={"actual_wobacon": "target_next_actual_wobacon"})

    merged = left.merge(right, on="batter")
    merged["pair"] = f"{y1}->{y2}"
    merged["season_n"] = y1
    merged["season_n_plus_1"] = y2
    rows.append(merged)
    print(f"  {y1} -> {y2}: {len(merged)} matched batters")

transitions = pd.concat(rows, ignore_index=True)

keep_cols = (
    ["batter", "player_name", "pair", "season_n", "season_n_plus_1", "n_bbe"]
    + feature_cols
    + ["statcast_xwobacon", "custom_xwobacon", "actual_wobacon", "target_next_actual_wobacon"]
)
transitions = transitions[keep_cols]
transitions.to_csv(OUT, index=False)
print(f"\nSaved {len(transitions)} batter-transitions to {OUT}")
print(transitions["pair"].value_counts())
