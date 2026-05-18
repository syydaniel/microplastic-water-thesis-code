#!/usr/bin/env python3
"""Regenerate Chapter 4 distribution and analysis figures.

Figures produced:
- 4.10 — distribution of R_new (histogram, three-way: All / Calculated / Imputed)
- 4.11 — cumulative distribution of R_new vs R_MARINA
- 4.12 — mean ΔR by sub-basin-size class (SUPERVISOR FIX: x-axis label)
- 4.13 — ΔR vs MP_inside scatter (SUPERVISOR FIX: x-axis label)

Inputs: Chapter_4_Summary.xlsx (per_basin sheet, 10,226 rows)
"""
import sys
sys.path.insert(0, "/tmp")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import thesis_style as ts
ts.apply_style()

XLSX = Path("/Users/a1-6/Desktop/Thesis_Organized/03_Code_Final/Chapter_4/V7_Comparison_Analysis/outputs/Chapter_4_Summary.xlsx")
OUT_DIRS = [
    Path("/Users/a1-6/Desktop/MASTER THESIS/Figures/Chapter 4"),
    Path("/Users/a1-6/Desktop/Thesis_Organized/02_Figures_Final/Chapter_4"),
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

def save(fig, fname):
    for d in OUT_DIRS:
        fig.savefig(d / fname)
    print(f"  Saved {fname}")
    plt.close(fig)


print("Loading Ch4 data ...")
df = pd.read_excel(XLSX, sheet_name="per_basin")
by_size = pd.read_excel(XLSX, sheet_name="by_size")
top10 = pd.read_excel(XLSX, sheet_name="top10_load")
print(f"  per_basin: {df.shape}, by_size: {by_size.shape}, top10: {top10.shape}")
print(df["method"].value_counts())


# ============================================================
# Figure 4.10 — distribution of R_new
# ============================================================
print("\n=== Figure 4.10 ===")
fig, ax = plt.subplots(figsize=(11, 5.5))

valid = df[df["R_new"].notna()].copy()
calc = valid[valid["method"] == "Calculated"]["R_new"].values
imp = valid[valid["method"] == "Imputed"]["R_new"].values
all_v = valid["R_new"].values

bins = np.linspace(0, 1, 51)
ax.hist(all_v, bins=bins, color="white", edgecolor="#252525", lw=1.0, label=f"All (n = {len(all_v):,})", zorder=1)
ax.hist(calc, bins=bins, color="#1F78B4", alpha=0.75, label=f"Calculated (n = {len(calc):,})", zorder=2)
ax.hist(imp, bins=bins, color="#FB9A29", alpha=0.85, label=f"Imputed (n = {len(imp):,})", zorder=3)

ax.axvline(np.mean(all_v), color="#D7191C", lw=2.0, ls="--", label=f"Mean = {np.mean(all_v):.3f}")
ax.axvline(np.median(all_v), color="#1A9850", lw=2.0, ls="-.", label=f"Median = {np.median(all_v):.3f}")
ax.set_xlabel("New microplastic retention rate, $R_{new}$ (dimensionless)")
ax.set_ylabel("Number of MARINA sub-basins (log scale)")
ax.set_xlim(-0.02, 1.02)
ax.set_yscale("log")
ax.set_ylim(0.5, len(all_v) * 0.4)
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save(fig, "Figure_4_10.png")


# ============================================================
# Figure 4.11 — cumulative distribution R_new vs R_MARINA
# ============================================================
print("\n=== Figure 4.11 ===")
fig, ax = plt.subplots(figsize=(11, 5.5))

both = df.dropna(subset=["R_new", "R_MARINA"])
r_new = np.sort(both["R_new"].values)
r_mar = np.sort(both["R_MARINA"].values)
n = len(r_new)
ecdf_y = np.arange(1, n + 1) / n * 100

ax.plot(r_new, ecdf_y, color="#1F78B4", lw=2.0, label=f"$R_{{new}}$ (this chapter)\nmean = {np.mean(r_new):.3f}, median = {np.median(r_new):.3f}")
ax.plot(r_mar, ecdf_y, color="#D7191C", ls="--", lw=2.0, label=f"$R_{{MARINA}}$ (Micella et al., 2024)\nmean = {np.mean(r_mar):.3f}, median = {np.median(r_mar):.3f}")

# Mark the two L_MIP rule fixed points: 0.75 and 0.90
for v, lab in [(0.75, "$L_{MIP}$ = 0.75"), (0.90, "$L_{MIP}$ = 0.90")]:
    ax.axvline(v, color="#666", ls=":", lw=0.8, alpha=0.7)
    ax.text(v, 5, lab, color="#666", ha="center", va="bottom", fontsize=9, rotation=90)

ax.set_xlabel("Sub-basin retention rate (dimensionless)")
ax.set_ylabel("Cumulative share of MARINA sub-basins (%)")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 100)
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
save(fig, "Figure_4_11.png")


# ============================================================
# Figure 4.12 — mean ΔR by sub-basin-size class (SUPERVISOR FIX)
# ============================================================
print("\n=== Figure 4.12 ===")
fig, ax = plt.subplots(figsize=(10, 5.5))

# size class order from the by_size sheet
sc_order = ["≤ 4 cells", "5–10 cells", "11–50 cells", "51–200 cells", "> 200 cells"]
by_size_ordered = by_size.set_index("size_class").reindex(sc_order).reset_index()
mean_dr = by_size_ordered["delta_R_mean"].values
ns = by_size_ordered["n_basins"].astype(int).values

# Bar colours: red if positive (R_new > R_MARINA), blue if negative
colors = [ts.DELTA_R_RED if v > 0 else ts.DELTA_R_BLUE for v in mean_dr]
bars = ax.bar(np.arange(len(sc_order)), mean_dr, color=colors, edgecolor="#252525", lw=0.8)

# Label bars with value and n
for i, (b, v, n) in enumerate(zip(bars, mean_dr, ns)):
    sgn = "+" if v > 0 else ""
    y = v + (0.005 if v > 0 else -0.005)
    va = "bottom" if v > 0 else "top"
    ax.text(i, y, f"{sgn}{v:.3f}", ha="center", va=va, fontsize=10, fontweight="bold")
    ax.text(i, -0.13, f"n = {n:,}", ha="center", va="top", fontsize=9)

ax.axhline(0, color="#404040", lw=1.0)
ax.set_xticks(np.arange(len(sc_order)))
ax.set_xticklabels(sc_order)
# SUPERVISOR FIX: x-axis label
ax.set_xlabel("MARINA sub-basin size class (number of 0.5° grid cells)")
ax.set_ylabel("Mean $\\Delta R$ ($R_{new}$ − $R_{MARINA}$)")
ax.set_ylim(-0.16, 0.10)

# Legend
red_patch = mpatches.Patch(color=ts.DELTA_R_RED, label="$R_{new}$ > $R_{MARINA}$ (mean ΔR > 0)")
blue_patch = mpatches.Patch(color=ts.DELTA_R_BLUE, label="$R_{MARINA}$ > $R_{new}$ (mean ΔR < 0)")
ax.legend(handles=[red_patch, blue_patch], loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save(fig, "Figure_4_12.png")


# ============================================================
# Figure 4.13 — ΔR vs MP_inside scatter (SUPERVISOR FIX)
# ============================================================
print("\n=== Figure 4.13 ===")
fig, ax = plt.subplots(figsize=(11, 6))

both = df.dropna(subset=["delta_R", "gen_flux"]).copy()
both = both[both["gen_flux"] > 0].copy()
both["log_gen"] = np.log10(both["gen_flux"])
both = both[np.isfinite(both["log_gen"])]

# Background scatter
ax.scatter(both["log_gen"], both["delta_R"], c="#A0A0A0", s=4, alpha=0.35, edgecolors="none", zorder=1)

# Decile bins
both["dec"] = pd.qcut(both["log_gen"], 10, labels=False, duplicates="drop")
dec_stats = both.groupby("dec").agg(
    log_gen_mid=("log_gen", "mean"),
    dr_mean=("delta_R", "mean"),
    dr_std=("delta_R", "std"),
    n=("delta_R", "size"),
).reset_index()
ax.errorbar(dec_stats["log_gen_mid"], dec_stats["dr_mean"], yerr=dec_stats["dr_std"],
            fmt="o", color="#1F78B4", ecolor="#1F78B4", capsize=3, lw=1.5, ms=8,
            zorder=4, label="Decile mean ± 1 SD")

# Top-10 highest-load points (red rings)
for i, row in top10.iterrows():
    rank = i + 1
    log_gen = np.log10(row.get("gen_flux", row.get("MP_inside", np.nan)))
    dr = row.get("delta_R", np.nan)
    if np.isfinite(log_gen) and np.isfinite(dr):
        ax.scatter([log_gen], [dr], facecolor="none", edgecolor="#D7191C", s=180, lw=1.6, zorder=5)
        ax.text(log_gen, dr + 0.02, str(rank), color="#D7191C", ha="center", va="bottom",
                fontsize=10, fontweight="bold", zorder=6)

ax.axhline(0, color="#404040", lw=0.8, ls="--")
# SUPERVISOR FIX: x-axis label uses MP_inside instead of "annual microplastic generation load"
ax.set_xlabel("$\\log_{10}$($MP_{inside}$, items y$^{-1}$)")
ax.set_ylabel("$\\Delta R$ = $R_{new}$ − $R_{MARINA}$")
ax.legend(loc="lower left", fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(-1.0, 0.4)

plt.tight_layout()
save(fig, "Figure_4_13.png")

print("\nAll Ch4 distribution figures regenerated.")
