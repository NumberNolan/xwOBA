"""
Produce custom xwOBA-on-contact (xwOBAcon) per batter per season using the
tuned LightGBM model.

NOTE ON SCOPE: model_data.parquet only contains SWINGS (bat-tracking data only
exists when the batter swings), so it has no called balls/strikes and no
walks/HBP -- meaning a true full-season xwOBA (which blends in actual BB/K/HBP
values) can't be computed from this dataset. What follows is xwOBA **on
contact** (xwOBAcon): the average expected value of a batter's batted balls
only, which is exactly the same scope as Statcast's own published xwOBAcon
metric -- so it's compared against estimated_woba_using_speedangle, not
full xwOBA.

Refits the tuned LightGBM (from tune_boosted_models.py's best params) on the
FULL 395,774-row dataset (not just the 80% training split used for the
model-comparison benchmark) since this is now meant as a production model,
not a held-out evaluation.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb

DATA = "data/model_data.parquet"
OUT = "results/batter_xwoba_by_season.csv"

FEATURES = ["launch_speed", "launch_angle", "spray_angle", "stand_L", "swing_path_tilt"]
TARGET = "woba_value"
BENCHMARK = "estimated_woba_using_speedangle"

# Best hyperparameters found by tune_boosted_models.py (RandomizedSearchCV)
TUNED_PARAMS = dict(
    n_estimators=861, num_leaves=67, max_depth=12, learning_rate=0.05859268690985102,
    subsample=0.6849356442713105, colsample_bytree=0.9330880728874675,
    min_child_samples=7, reg_alpha=7.579479953348009, reg_lambda=2.1368329072358767,
    random_state=42, n_jobs=-1, verbosity=-1,
)

print("Loading modeling dataset...")
df = pd.read_parquet(DATA)
print(f"Rows: {len(df)}")

print("Refitting tuned LightGBM on the FULL dataset...")
model = lgb.LGBMRegressor(**TUNED_PARAMS)
model.fit(df[FEATURES].values, df[TARGET].values)

df["custom_xwobacon"] = model.predict(df[FEATURES].values)

print("Aggregating by batter x season...")
agg = (
    df.groupby(["batter", "player_name", "game_year"], as_index=False)
      .agg(
          n_bbe=("custom_xwobacon", "size"),
          custom_xwobacon=("custom_xwobacon", "mean"),
          actual_wobacon=(TARGET, "mean"),
          statcast_xwobacon=(BENCHMARK, "mean"),
      )
)
agg["custom_minus_statcast"] = agg["custom_xwobacon"] - agg["statcast_xwobacon"]
agg = agg.rename(columns={"game_year": "season"})
agg = agg.sort_values(["season", "custom_xwobacon"], ascending=[True, False])

for col in ["custom_xwobacon", "actual_wobacon", "statcast_xwobacon", "custom_minus_statcast"]:
    agg[col] = agg[col].round(4)

agg.to_csv(OUT, index=False)
print(f"Saved {len(agg)} batter-season rows to {OUT}")

print("\nRows per season:")
print(agg.groupby("season").size())

MIN_BBE = 100
print(f"\n=== Top 15 by custom xwOBAcon per season (min {MIN_BBE} batted balls) ===")
qualified = agg[agg["n_bbe"] >= MIN_BBE]
for season, grp in qualified.groupby("season"):
    print(f"\n-- {season} --")
    top = grp.sort_values("custom_xwobacon", ascending=False).head(15)
    print(top[["player_name", "n_bbe", "custom_xwobacon", "statcast_xwobacon", "actual_wobacon"]].to_string(index=False))
