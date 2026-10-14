"""
Year-over-year "stickiness" test: for each metric (custom xwOBAcon, Statcast's
own xwOBAcon, and actual wOBA-on-contact), correlate a batter's value in
season N against their value in season N+1. A metric that's picking up more
true talent signal (and less noise) should correlate more strongly with
itself next year -- this is the standard way to argue one expected-stat
model is "better" than another, independent of which one fits closer to
THIS season's actual outcomes.

Requires >= MIN_BBE batted balls in BOTH seasons of a pair, to keep small-
sample noise from dominating.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

IN = "results/batter_xwoba_by_season.csv"
MIN_BBE = 100
METRICS = ["custom_xwobacon", "statcast_xwobacon", "actual_wobacon"]
METRIC_LABELS = {
    "custom_xwobacon": "Custom xwOBAcon (tuned LightGBM)",
    "statcast_xwobacon": "Statcast xwOBAcon",
    "actual_wobacon": "Actual wOBA on contact",
}

df = pd.read_csv(IN)
seasons = sorted(df["season"].unique())
pairs = list(zip(seasons[:-1], seasons[1:]))
print(f"Seasons: {seasons}")
print(f"Consecutive-season pairs: {pairs}")
print(f"Min batted balls required in both seasons of a pair: {MIN_BBE}\n")

per_pair_results = []
pooled = {m: {"y1": [], "y2": []} for m in METRICS}

for y1, y2 in pairs:
    left = df[(df["season"] == y1) & (df["n_bbe"] >= MIN_BBE)]
    right = df[(df["season"] == y2) & (df["n_bbe"] >= MIN_BBE)]
    merged = left.merge(right, on="batter", suffixes=(f"_{y1}", f"_{y2}"))
    n = len(merged)
    print(f"=== {y1} -> {y2}  (n = {n} batters qualified in both seasons) ===")
    row = {"pair": f"{y1}->{y2}", "n": n}
    for m in METRICS:
        v1 = merged[f"{m}_{y1}"].values
        v2 = merged[f"{m}_{y2}"].values
        r, p = pearsonr(v1, v2)
        print(f"  {METRIC_LABELS[m]:38s} r = {r:.4f}   r^2 = {r**2:.4f}   (p={p:.2e})")
        row[f"{m}_r"] = r
        row[f"{m}_r2"] = r ** 2
        pooled[m]["y1"].extend(v1.tolist())
        pooled[m]["y2"].extend(v2.tolist())
    per_pair_results.append(row)
    print()

print(f"=== POOLED across all {len(pairs)} season-pairs (n = {len(pooled[METRICS[0]]['y1'])} batter-transitions) ===")
pooled_row = {"pair": "POOLED", "n": len(pooled[METRICS[0]]["y1"])}
for m in METRICS:
    v1 = np.array(pooled[m]["y1"])
    v2 = np.array(pooled[m]["y2"])
    r, p = pearsonr(v1, v2)
    print(f"  {METRIC_LABELS[m]:38s} r = {r:.4f}   r^2 = {r**2:.4f}   (p={p:.2e})")
    pooled_row[f"{m}_r"] = r
    pooled_row[f"{m}_r2"] = r ** 2

results_df = pd.DataFrame(per_pair_results + [pooled_row])
results_df.to_csv("results/autocorrelation_results.csv", index=False)
print("\nSaved autocorrelation_results.csv")

print("\n=== Ranking (pooled r, higher = stickier = more signal, less noise) ===")
ranked = sorted(METRICS, key=lambda m: -pooled_row[f"{m}_r"])
for i, m in enumerate(ranked, 1):
    print(f"  {i}. {METRIC_LABELS[m]:38s} r = {pooled_row[f'{m}_r']:.4f}")
