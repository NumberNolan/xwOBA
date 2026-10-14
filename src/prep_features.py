"""
Build the modeling dataset for the custom xwOBA project.

- Restrict to batted-ball-in-play events (type == 'X') that are wOBA-eligible
  (woba_denom == 1, which excludes sac bunts etc. -- matches how Statcast's
  own xwOBA / wOBA are scoped).
- Derive spray_angle from hit coordinates (hc_x, hc_y).
- Encode `stand` as a binary feature (stand_L).
- Drop rows missing any of the 9 candidate features or the target.
- Save a clean parquet file with: the 9 candidate features, target (woba_value),
  benchmark (estimated_woba_using_speedangle), and identifying columns.
"""
import pandas as pd
import numpy as np

RAW = "data/statcast_bat_tracking_data.csv"
OUT = "data/model_data.parquet"

cols_needed = [
    "game_date", "game_year", "game_pk", "at_bat_number", "pitch_number",
    "type", "events", "description", "stand", "p_throws",
    "launch_speed", "launch_angle", "hc_x", "hc_y",
    "bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    "batter", "player_name",
]

print("Reading raw CSV (only needed columns)...")
df = pd.read_csv(RAW, usecols=cols_needed, low_memory=False)
print("Raw rows:", len(df))

# 1) Restrict to wOBA-eligible batted-ball-in-play events
df = df[(df["type"] == "X") & (df["woba_denom"] == 1)].copy()
print("After type=='X' & woba_denom==1:", len(df))

# 2) Derive spray angle from hit coordinates.
#    Standard Statcast-field-coordinate formula (home plate origin at
#    hc_x=125.42, hc_y=198.27); positive = pull side for a RHH is handled by
#    leaving sign raw and letting `stand` capture batter handedness -- the
#    model can learn the stand x spray_angle interaction itself via the
#    tree-based / NN models, and linear models get `stand` as a separate term.
df["spray_angle"] = np.degrees(
    np.arctan((df["hc_x"] - 125.42) / (198.27 - df["hc_y"]))
) * 0.75

# 3) Encode stand as binary (1 = left-handed batter, 0 = right-handed)
df["stand_L"] = (df["stand"] == "L").astype(float)

FEATURES = [
    "launch_speed", "launch_angle", "spray_angle", "bat_speed", "stand_L",
    "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
]
TARGET = "woba_value"
BENCHMARK = "estimated_woba_using_speedangle"

keep_cols = FEATURES + [TARGET, BENCHMARK, "game_date", "game_year", "game_pk",
                          "at_bat_number", "pitch_number", "batter", "player_name",
                          "events", "stand"]

before = len(df)
df_model = df.dropna(subset=FEATURES + [TARGET, BENCHMARK]).copy()
print(f"Dropped {before - len(df_model)} rows with missing feature/target/benchmark values")
print("Final modeling rows:", len(df_model))

df_model = df_model[keep_cols]

print("\nFeature summary stats:")
print(df_model[FEATURES].describe().T)

print("\nTarget (woba_value) summary:")
print(df_model[TARGET].describe())

print("\nRows per season:")
print(df_model["game_year"].value_counts().sort_index())

df_model.to_parquet(OUT, index=False)
print(f"\nSaved modeling dataset to {OUT}")
