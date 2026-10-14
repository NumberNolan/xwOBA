import pandas as pd
import numpy as np

path = "data/statcast_bat_tracking_data.csv"

cols_needed = [
    "game_date", "game_year", "game_pk", "at_bat_number", "pitch_number",
    "type", "events", "description", "stand", "p_throws",
    "launch_speed", "launch_angle", "hc_x", "hc_y",
    "bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    "batter", "player_name",
]

# Read just the header first to confirm which columns actually exist
header = pd.read_csv(path, nrows=0).columns.tolist()
missing = [c for c in cols_needed if c not in header]
print("Columns in dataset:", len(header))
print("Requested columns missing from file:", missing)

use_cols = [c for c in cols_needed if c in header]

# Read only needed columns to keep memory reasonable
df = pd.read_csv(path, usecols=use_cols, low_memory=False)
print("\nTotal rows (all bat-tracking swings):", len(df))

print("\n--- Missingness of key columns (overall) ---")
for c in ["launch_speed","launch_angle","hc_x","hc_y","bat_speed","swing_length",
          "attack_angle","attack_direction","swing_path_tilt","woba_value","woba_denom",
          "estimated_woba_using_speedangle"]:
    if c in df.columns:
        n_na = df[c].isna().sum()
        print(f"{c:35s} missing={n_na:9d} ({n_na/len(df)*100:5.1f}%)")
    else:
        print(f"{c:35s} NOT IN FILE")

print("\n--- type value counts ---")
print(df["type"].value_counts(dropna=False))

# Batted ball subset
bip = df[df["type"] == "X"].copy()
print("\nBatted-ball-in-play rows (type=='X'):", len(bip))

print("\n--- Missingness within batted-ball subset ---")
for c in ["launch_speed","launch_angle","hc_x","hc_y","bat_speed","swing_length",
          "attack_angle","attack_direction","swing_path_tilt","woba_value","woba_denom",
          "estimated_woba_using_speedangle"]:
    if c in bip.columns:
        n_na = bip[c].isna().sum()
        print(f"{c:35s} missing={n_na:9d} ({n_na/len(bip)*100:5.1f}%)")

print("\n--- woba_denom value counts (batted balls) ---")
print(bip["woba_denom"].value_counts(dropna=False))

# Check when attack_angle/direction/tilt actually start appearing
if "attack_angle" in bip.columns:
    have_attack = bip.dropna(subset=["attack_angle"])
    print("\nEarliest game_date with non-null attack_angle:", have_attack["game_date"].min())
    print("Rows with non-null attack_angle:", len(have_attack), "of", len(bip))

print("\nEarliest/Latest game_date overall:", df["game_date"].min(), df["game_date"].max())
print("\nRows per game_year (batted balls):")
print(bip["game_year"].value_counts().sort_index())
