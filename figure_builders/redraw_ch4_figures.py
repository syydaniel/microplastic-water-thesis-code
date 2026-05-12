"""Regenerate Ch4 figures 4.8-4.14 with canonical MP_inside,j / MP_out,j labels.

Adapts code from:
- /Users/a1-6/Desktop/01_MSc_Thesis/MASTER THESIS/Code_Figures/build_maps.py (4.8, 4.9, 4.14)
- /Users/a1-6/Desktop/01_MSc_Thesis/MASTER THESIS/Code_Figures/build_ch4_figures.py (4.10-4.13)

Updates relative to those scripts:
- Paths fixed for /Users/a1-6/Desktop/01_MSc_Thesis/Thesis_Organized/...
- Output to /tmp/ch4_work/media/ (where Ch4 markdown expects them)
- Axis labels use MP_inside,j / MP_out,j (with j subscript) per the canonical
  symbol decided in Ch4 revision.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch

# Style
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Paths
MARINA_SHP = Path("/Users/a1-6/Desktop/01_MSc_Thesis/Thesis_Organized/03_Code_Final/Chapter_4/Active_V6_Pipeline/04_Final_Outputs/Final_Deliverables/Shapefile_Retention_Clear.shp")
XLSX = Path("/Users/a1-6/Desktop/01_MSc_Thesis/Thesis_Organized/03_Code_Final/Chapter_4/V7_Comparison_Analysis/outputs/Chapter_4_Summary.xlsx")
OUT_DIR = Path("/tmp/ch4_work/media")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading MARINA shapefile ...")
marina = gpd.read_file(MARINA_SHP).to_crs("EPSG:4326")
print(f"  marina: {len(marina):,}")
# Inspect columns
print(f"  columns: {list(marina.columns)[:20]}")

print("Loading Ch4 summary xlsx ...")
df = pd.read_excel(XLSX, sheet_name="per_basin")
by_size = pd.read_excel(XLSX, sheet_name="by_size")
top10 = pd.read_excel(XLSX, sheet_name="top10_load")
print(f"  per_basin: {df.shape}, by_size: {by_size.shape}, top10: {top10.shape}")

def map_axes(figsize=(14, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    return fig, ax


def save(fig, fname):
    out = OUT_DIR / fname
    fig.savefig(out)
    print(f"  Saved {out.name} ({out.stat().st_size//1024} KB)")
    plt.close(fig)


# ===========================================================
# Figure 4.8 — global MP_inside,j
# ===========================================================
print("\n=== Figure 4.8 — MP_inside,j ===")
# The shapefile column for generation load
col = None
for cand in ['gen_flux', 'genload', 'MP_inside', 'mp_inside', 'gen_load']:
    if cand in marina.columns:
        col = cand
        break
if col is None:
    print(f"  WARNING: no generation column found, columns = {list(marina.columns)}")
else:
    print(f"  using column: {col}")
    fig, ax = map_axes()
    vp = marina.copy()
    vp["log_gen"] = np.log10(vp[col].clip(lower=0) + 1)
    na = vp[vp[col].isna()]
    na.plot(ax=ax, color="#E0E0E0", edgecolor="none", aspect=None)
    valid = vp[vp[col].notna()]
    vmax = np.percentile(valid["log_gen"], 99.5)
    valid.plot(ax=ax, column="log_gen", cmap="YlOrRd", vmin=0, vmax=vmax,
               edgecolor="none",
               legend=True,
               legend_kwds={"label": r"log$_{10}$($MP_{inside,j}$ + 1) [items y$^{-1}$]",
                            "shrink": 0.6, "pad": 0.02})
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_title(r"Global distribution of $MP_{inside,j}$ across 10,226 MARINA sub-basins",
                 fontweight="bold", pad=8)
    save(fig, "image8.png")


# ===========================================================
# Figure 4.9 — global MP_out,j  (outlet load as points)
# ===========================================================
print("\n=== Figure 4.9 — MP_out,j ===")
col_out = None
for cand in ['out_flux', 'outload', 'MP_out', 'mp_out', 'out_load']:
    if cand in marina.columns:
        col_out = cand
        break
if col_out:
    fig, ax = map_axes()
    marina.plot(ax=ax, color="#F4F4F4", edgecolor="none", aspect=None)
    out_pts = marina.copy()
    out_pts["log_out"] = np.log10(out_pts[col_out].clip(lower=0) + 1)
    out_pts["lon"] = out_pts.geometry.centroid.x
    out_pts["lat"] = out_pts.geometry.centroid.y
    v = out_pts[out_pts[col_out].notna() & out_pts[col_out].gt(0)]
    vmax = np.percentile(v["log_out"], 99.5)
    sc = ax.scatter(v["lon"], v["lat"], c=v["log_out"], cmap="YlOrRd",
                    s=np.clip(v["log_out"] * 6, 2, 60), vmin=0, vmax=vmax,
                    edgecolors="none", alpha=0.9)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(r"log$_{10}$($MP_{out,j}$ + 1) [items y$^{-1}$]", fontweight="bold")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_title(r"Global distribution of $MP_{out,j}$ at MARINA sub-basin outlets",
                 fontweight="bold", pad=8)
    save(fig, "image9.png")


# ===========================================================
# Figure 4.10 — distribution of R_new
# ===========================================================
print("\n=== Figure 4.10 — R_new distribution ===")
fig, ax = plt.subplots(figsize=(11, 5.5))
valid_r = df[df["R_new"].notna()].copy()
calc = valid_r[valid_r["method"] == "Calculated"]["R_new"].values
imp = valid_r[valid_r["method"] == "Imputed"]["R_new"].values
all_v = valid_r["R_new"].values
bins = np.linspace(0, 1, 51)
ax.hist(all_v, bins=bins, color="white", edgecolor="#252525", lw=1.0, label=f"All (n = {len(all_v):,})", zorder=1)
ax.hist(calc, bins=bins, color="#1F78B4", alpha=0.75, label=f"Calculated (n = {len(calc):,})", zorder=2)
ax.hist(imp, bins=bins, color="#FB9A29", alpha=0.85, label=f"Imputed (n = {len(imp):,})", zorder=3)
ax.axvline(np.mean(all_v), color="#D7191C", lw=2.0, ls="--", label=f"Mean = {np.mean(all_v):.3f}")
ax.axvline(np.median(all_v), color="#1A9850", lw=2.0, ls="-.", label=f"Median = {np.median(all_v):.3f}")
ax.set_xlabel(r"New microplastic retention rate, $R_{new,j}$ (dimensionless)")
ax.set_ylabel("Number of MARINA sub-basins (log scale)")
ax.set_xlim(-0.02, 1.02)
ax.set_yscale("log")
ax.set_ylim(0.5, len(all_v) * 0.4)
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save(fig, "image10.png")


# ===========================================================
# Figure 4.11 — cumulative distribution R_new vs R_MARINA
# ===========================================================
print("\n=== Figure 4.11 — cumulative R_new vs R_MARINA ===")
fig, ax = plt.subplots(figsize=(8, 5.5))
rn = np.sort(df[df["R_new"].notna()]["R_new"].values)
rm = np.sort(df[df["R_MARINA"].notna()]["R_MARINA"].values)
ax.plot(rn, np.linspace(0, 100, len(rn)), color="#1F78B4", lw=2.2, label=r"$R_{new,j}$ (this study)")
ax.plot(rm, np.linspace(0, 100, len(rm)), color="#D7191C", lw=2.2, ls="--", label=r"$R_{MARINA,j}$ (baseline)")
ax.set_xlabel("Sub-basin retention rate (dimensionless)")
ax.set_ylabel("Cumulative share of MARINA sub-basins (%)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
save(fig, "image11.png")


# ===========================================================
# Figure 4.12 — mean ΔR by sub-basin-size class
# ===========================================================
print("\n=== Figure 4.12 — mean ΔR by size class ===")
fig, ax = plt.subplots(figsize=(10, 5.5))
sizes = by_size["size_class"].tolist()
means = by_size["mean_delta_R"].tolist()
ns = by_size["n"].tolist()
colors = ["#1F78B4" if m < 0 else "#D7191C" for m in means]
bars = ax.bar(range(len(sizes)), means, color=colors, edgecolor="#252525", lw=1.0)
ax.axhline(0, color="black", lw=0.8)
for i, (m, n) in enumerate(zip(means, ns)):
    ax.text(i, m + (0.005 if m > 0 else -0.005),
            f"n = {n:,}\n{m:+.3f}",
            ha="center", va="bottom" if m > 0 else "top", fontsize=9)
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([str(s) for s in sizes])
ax.set_xlabel("MARINA sub-basin size class (number of 0.5° grid cells)")
ax.set_ylabel(r"Mean $\Delta R_j$ ($R_{new,j}$ − $R_{MARINA,j}$)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save(fig, "image12.png")


# ===========================================================
# Figure 4.13 — ΔR vs log MP_inside scatter
# ===========================================================
print("\n=== Figure 4.13 — ΔR vs MP_inside scatter ===")
fig, ax = plt.subplots(figsize=(11, 6))
v_both = df[(df["R_new"].notna()) & (df["R_MARINA"].notna()) & (df["gen_flux"].notna()) & (df["gen_flux"] > 0)].copy()
v_both["log_gen"] = np.log10(v_both["gen_flux"])
v_both["delta_R"] = v_both["R_new"] - v_both["R_MARINA"]
ax.scatter(v_both["log_gen"], v_both["delta_R"], s=4, color="#888888", alpha=0.35, edgecolors="none", label=f"Individual sub-basins (n = {len(v_both):,})")
# Decile binning
v_both["decile"] = pd.qcut(v_both["log_gen"], q=10, labels=False, duplicates="drop")
dec_means = v_both.groupby("decile").agg({"log_gen": "mean", "delta_R": ["mean", "std"]})
dec_means.columns = ["log_gen_mean", "delta_R_mean", "delta_R_sd"]
ax.errorbar(dec_means["log_gen_mean"], dec_means["delta_R_mean"],
            yerr=dec_means["delta_R_sd"], color="#1F78B4", marker="o", lw=2.0,
            label="Decile-bin mean $\\Delta R_j$ ± 1 SD")
# Top-10 highest-load sub-basins
for i, row in top10.iterrows():
    if i >= 10:
        break
    rank = i + 1
    lg = np.log10(row.get("gen_flux", np.nan)) if pd.notna(row.get("gen_flux", np.nan)) else np.nan
    dr = row.get("delta_R", np.nan)
    if np.isfinite(lg) and np.isfinite(dr):
        ax.scatter([lg], [dr], facecolor="none", edgecolor="#D7191C", s=180, lw=1.6, zorder=5)
        ax.text(lg, dr + 0.02, str(rank), color="#D7191C", ha="center", va="bottom",
                fontsize=10, fontweight="bold", zorder=6)
ax.axhline(0, color="#404040", lw=0.8, ls="--")
ax.set_xlabel(r"log$_{10}$($MP_{inside,j}$, items y$^{-1}$)")
ax.set_ylabel(r"$\Delta R_j$ = $R_{new,j}$ − $R_{MARINA,j}$")
ax.legend(loc="lower left", fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(-1.0, 0.4)
plt.tight_layout()
save(fig, "image13.png")


# ===========================================================
# Figure 4.14 — 3-panel R_MARINA / R_new / ΔR maps
# ===========================================================
print("\n=== Figure 4.14 — 3-panel retention maps ===")
fig, axes = plt.subplots(3, 1, figsize=(14, 18))
for i, (ax, col_n, label, cmap, vmin, vmax) in enumerate([
    (axes[0], "R_MARINA", r"(a) MARINA-Multi baseline retention, $R_{MARINA,j}$ = 1 − $FE_{rivo,j}$", "Blues", 0.0, 1.0),
    (axes[1], "R_new",    r"(b) This study retention, $R_{new,j}$ = 1 − $MP_{out,j}$ / $MP_{inside,j}$", "Blues", 0.0, 1.0),
    (axes[2], "delta_R",  r"(c) Per-basin retention difference, $\Delta R_j$ = $R_{new,j}$ − $R_{MARINA,j}$", "RdBu", -0.5, 0.5),
]):
    vp = marina.copy()
    if col_n == "delta_R":
        # Compute delta_R on the fly if not in shapefile
        # Use df merge - assume R_new and R_MARINA cols
        if "delta_R" not in vp.columns:
            vp = vp.merge(df[["basin_id", "delta_R"]], on="basin_id", how="left") if "basin_id" in vp.columns else vp
    na = vp[vp[col_n].isna()] if col_n in vp.columns else vp.iloc[:0]
    if len(na):
        na.plot(ax=ax, color="#E0E0E0", edgecolor="none", aspect=None)
    if col_n in vp.columns:
        valid = vp[vp[col_n].notna()]
        valid.plot(ax=ax, column=col_n, cmap=cmap, vmin=vmin, vmax=vmax,
                   edgecolor="none", legend=True,
                   legend_kwds={"label": col_n, "shrink": 0.6, "pad": 0.02})
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_title(label, fontweight="bold", pad=4, loc="left")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

plt.tight_layout()
save(fig, "image14.png")

print("\nAll Ch4 figures regenerated with canonical MP_inside,j / MP_out,j labels.")
