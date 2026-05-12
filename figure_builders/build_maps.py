#!/usr/bin/env python3
"""Build all global map figures with unified style and supervisor edits.

Maps produced:
- Fig 3.4 — global mask (Valid / Desert / NA)
- Fig 3.5 — global natural discharge (HydroBASINS Level 6)
- Fig 3.7 — predicted abundance + top-10 hotspots
- Fig 3.8 — predicted annual load + top-10 hotspots
- Fig 3.13 — six-config CoV uncertainty map
- Fig 4.4 — Step-1 filter outcome map
- Fig 4.6 — calculated / imputed / NA per MARINA sub-basin
- Fig 4.8 — global MP_inside (SUPERVISOR FIX: colorbar 'Genload' → 'MP_inside')
- Fig 4.9 — global MP_out (SUPERVISOR FIX: colorbar 'Outload' → 'MP_out')
- Fig 4.14 — 3-panel R_MARINA / R_new / ΔR (SUPERVISOR FIX: panel-b drop "Calculated")
"""
import sys
sys.path.insert(0, "/tmp")
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

import thesis_style as ts
ts.apply_style()

# Inputs
LEV6_SHP = Path("/Users/a1-6/Desktop/Thesis_Organized/03_Code_Final/Chapter_3/3.3.1 reult/data/BasinATLAS_v10_lev06.shp")
LEV6_STATS = Path("/Users/a1-6/Desktop/Thesis_Organized/03_Code_Final/Chapter_3/02_Training/03_Global_Results_Refined/Global_Stats_Lev6.csv")
LEV6_AGREE = Path("/Users/a1-6/Desktop/Thesis_Organized/03_Code_Final/Chapter_3/04_Model_Comparison/Global_Results_Agreement_Levels.csv")
LEV6_LOAD = Path("/Users/a1-6/Desktop/Thesis_Organized/03_Code_Final/Chapter_3/3.3.1 reult/data/Global_Load_Lev6.csv")
TOP10_ABUND = Path("/Users/a1-6/Desktop/Thesis_Organized/03_Code_Final/Chapter_3/05_Top10_Basin_Analysis/Top10_Subbasins_Detailed.csv")
MARINA_SHP = Path("/Users/a1-6/Desktop/Thesis_Organized/03_Code_Final/Chapter_4/Active_V6_Pipeline/04_Final_Outputs/Final_Deliverables/Shapefile_Retention_Clear.shp")
MARINA_XLSX = Path("/Users/a1-6/Desktop/Thesis_Organized/03_Code_Final/Chapter_4/V7_Comparison_Analysis/outputs/Chapter_4_Summary.xlsx")

CH3_OUT = [Path("/Users/a1-6/Desktop/MASTER THESIS/Figures/Chapter 3"),
           Path("/Users/a1-6/Desktop/Thesis_Organized/02_Figures_Final/Chapter_3")]
CH4_OUT = [Path("/Users/a1-6/Desktop/MASTER THESIS/Figures/Chapter 4"),
           Path("/Users/a1-6/Desktop/Thesis_Organized/02_Figures_Final/Chapter_4")]

def save(fig, fname, dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / fname, dpi=200, bbox_inches="tight")
    print(f"  Saved {fname}")
    plt.close(fig)


def base_world(ax, gdf):
    """Add light grey land background outline (just plot all polygons)."""
    gdf.plot(ax=ax, color="#F0F0F0", edgecolor="#D0D0D0", linewidth=0.1, alpha=0.7, aspect=None)


def map_axes(figsize=(14, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_aspect(1.0)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    return fig, ax


# ============================================================
# Load Lev6 base + stats
# ============================================================
print("Loading HydroBASINS Level 6 ...")
lev6 = gpd.read_file(LEV6_SHP)[["HYBAS_ID", "PFAF_ID", "SUB_AREA", "dis_m3_pyr", "geometry"]]
print(f"  L6: {len(lev6):,} rows")

stats = pd.read_csv(LEV6_STATS)
agree = pd.read_csv(LEV6_AGREE)
print(f"  stats: {len(stats):,}, agreement: {len(agree):,}")

# Merge
lev6_m = lev6.merge(stats[["Lev6_ID", "Mean_Linear_Conc", "Mean_Log_Conc", "Anomaly_Linear"]],
                    left_on="PFAF_ID", right_on="Lev6_ID", how="left")
lev6_m = lev6_m.merge(agree[["Lev6_ID", "CoV"]], on="Lev6_ID", how="left")
print(f"  merged: {len(lev6_m):,}")


# ============================================================
# Figure 3.4 — global mask (Valid / Desert / NA)
# ============================================================
print("\n=== Figure 3.4 — global mask ===")
def classify(row):
    if pd.notna(row["Mean_Linear_Conc"]):
        return "Valid"
    return "Not Available"

lev6_m["mask_class"] = lev6_m.apply(classify, axis=1)

fig, ax = map_axes(figsize=(14, 7))
colors_map = {"Valid": "#2C9F2C", "Not Available": "#A0A0A0"}
for cls, c in colors_map.items():
    sub = lev6_m[lev6_m["mask_class"] == cls]
    sub.plot(ax=ax, color=c, edgecolor="none", linewidth=0, aspect=None)

ax.set_xlim(-180, 180)
ax.set_ylim(-60, 80)
ax.set_aspect("equal")

handles = [Patch(color=c, label=cls) for cls, c in colors_map.items()]
ax.legend(handles=handles, loc="lower left", fontsize=10, frameon=True, facecolor="white", edgecolor="#404040")
ax.set_title("Spatial coverage of the global microplastic-abundance modelling framework",
             fontweight="bold", pad=8)
save(fig, "Figure_3_4.png", CH3_OUT)


# ============================================================
# Figure 3.5 — natural discharge
# ============================================================
print("\n=== Figure 3.5 — natural discharge ===")
fig, ax = map_axes(figsize=(14, 7))
disc = lev6_m.copy()
disc["log_dis"] = np.log10(disc["dis_m3_pyr"].clip(lower=1e-6) + 1)
disc.plot(ax=ax, column="log_dis", cmap="Blues", edgecolor="none",
          legend=True, legend_kwds={"label": "log₁₀(natural discharge, m³ s⁻¹, aspect=None)",
                                     "shrink": 0.6, "pad": 0.02})
ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect("equal")
ax.set_title("Global natural river discharge at HydroBASINS Level 6",
             fontweight="bold", pad=8)
save(fig, "Figure_3_5.png", CH3_OUT)


# ============================================================
# Figure 3.7 — predicted abundance + top-10 hotspots (main + 3 zooms)
# ============================================================
# Layout (v4, May 2026): each panel is a SEPARATE 4:3 PNG. The main global
# map (Figure_3_7.png) holds all 10 hotspots; three regional zooms
# (Figure_3_7a/b/c.png) re-render the same polygon data at the crowded
# clusters so the rank numbers remain readable. The colour scale and
# vmin/vmax are identical across all four panels.
print("\n=== Figure 3.7 — predicted abundance ===")

VMIN_ABUND, VMAX_ABUND = 0, 7

valid = lev6_m[lev6_m["Mean_Linear_Conc"].notna()].copy()
nodata = lev6_m[lev6_m["Mean_Linear_Conc"].isna()]
valid["log_abund"] = np.log10(valid["Mean_Linear_Conc"] + 1)

top10_abund = (valid.nlargest(10, "Mean_Linear_Conc")
                    .reset_index(drop=True).copy())
top10_abund["rank"] = range(1, 11)
top10_abund["lon"] = top10_abund.geometry.centroid.x
top10_abund["lat"] = top10_abund.geometry.centroid.y


def draw_rank_marker(ax, lon, lat, rank, value_log, vmin, vmax,
                     offset_pt, marker_size, font_size):
    """Plot one labelled leader line that ends in an arrowhead at (lon, lat).

    No circle marker. The arrowhead alone identifies the sub-basin centroid;
    the underlying YlOrRd polygon already shows the abundance / load value.
    Dropping the circle avoids the layering issue where one rank's leader
    line overlaps another rank's circle.

    offset_pt is (dx, dy) in pixel-points so leader-line lengths stay
    consistent across the global map and the regional zooms. The
    marker_size, value_log, vmin and vmax arguments are accepted for
    backward-compatible call sites; they are now unused.
    """
    _ = marker_size, value_log, vmin, vmax
    ax.annotate(str(rank), xy=(lon, lat),
                xytext=offset_pt, textcoords="offset points",
                fontsize=font_size, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec="black", lw=0.6, alpha=0.95),
                arrowprops=dict(
                    arrowstyle="->,head_length=0.45,head_width=0.30",
                    color="black", lw=1.0, shrinkA=0, shrinkB=0),
                zorder=10)


# Pixel-offset table for the global abundance map. Offsets are chosen so
# that no two leader lines cross each other.
MAIN37_OFFSETS = {
    1:  (-55,  28), 2:  ( 38,  10), 3:  ( 38, -22), 4:  ( 38, -20),
    5:  (-45,  20), 6:  ( 38, -28), 7:  (-50,  -2), 8:  (  0,  32),
    9:  (-45,  28), 10: ( 38, -22),
}

# --- Figure 3.7 main map ---
fig, ax = plt.subplots(figsize=(8, 6))   # 4:3
nodata.plot(ax=ax, color="#E0E0E0", edgecolor="none", aspect=None)
valid.plot(ax=ax, column="log_abund", cmap="YlOrRd",
           vmin=VMIN_ABUND, vmax=VMAX_ABUND, edgecolor="none",
           legend=True,
           legend_kwds={"label": "log₁₀(Predicted abundance + 1) [items m⁻³]",
                        "shrink": 0.6, "pad": 0.02})
for _, h in top10_abund.iterrows():
    r = int(h["rank"])
    draw_rank_marker(ax, h["lon"], h["lat"], r, h["log_abund"],
                     VMIN_ABUND, VMAX_ABUND,
                     offset_pt=MAIN37_OFFSETS[r],
                     marker_size=max(110, 220 - (r - 1) * 10),
                     font_size=10)
ax.set_xlim(-180, 180); ax.set_ylim(-58, 82); ax.set_aspect("equal")
ax.set_xlabel("Longitude (°)", fontsize=10)
ax.set_ylabel("Latitude (°)", fontsize=10)
save(fig, "Figure_3_7.png", CH3_OUT)

# --- Figure 3.7 zoom panels ---
FIG37_ZOOMS = [
    {"out": "Figure_3_7a.png",
     "title": "(a) Maharashtra cluster (near Pune)",
     "extent": (70, 78, 16, 22), "ranks": [1, 6],
     "label_offsets": {1: (-55,  30), 6: ( 45, -30)}},
    {"out": "Figure_3_7b.png",
     "title": "(b) Mississippi cluster (Missouri)",
     "extent": (-97, -85, 33, 42), "ranks": [3, 7, 8],
     "label_offsets": {3: ( 45, -30), 7: (-55,  30), 8: ( 45,  30)}},
    {"out": "Figure_3_7c.png",
     "title": "(c) Kaduna cluster (Nigeria)",
     "extent": (4, 12, 8, 14), "ranks": [9, 10],
     "label_offsets": {9: (-55,  30), 10: ( 45, -30)}},
]
for z in FIG37_ZOOMS:
    lon_min, lon_max, lat_min, lat_max = z["extent"]
    sub_v = valid.cx[lon_min:lon_max, lat_min:lat_max]
    sub_n = nodata.cx[lon_min:lon_max, lat_min:lat_max]
    fig, ax = plt.subplots(figsize=(8, 6))
    if not sub_n.empty:
        sub_n.plot(ax=ax, color="#E0E0E0", edgecolor="none", aspect=None)
    sub_v.plot(ax=ax, column="log_abund", cmap="YlOrRd",
               vmin=VMIN_ABUND, vmax=VMAX_ABUND, edgecolor="none",
               legend=True,
               legend_kwds={"label": "log₁₀(Abundance + 1) [items m⁻³]",
                            "shrink": 0.7, "pad": 0.02})
    for r in z["ranks"]:
        h = top10_abund[top10_abund["rank"] == r].iloc[0]
        draw_rank_marker(ax, h["lon"], h["lat"], r, h["log_abund"],
                         VMIN_ABUND, VMAX_ABUND,
                         offset_pt=z["label_offsets"][r],
                         marker_size=max(160, 320 - (r - 1) * 18),
                         font_size=13)
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.set_title(z["title"], fontweight="bold", pad=10)
    save(fig, z["out"], CH3_OUT)


# ============================================================
# Figure 3.8 — predicted annual load + top-10 hotspots (main + 2 zooms)
# ============================================================
print("\n=== Figure 3.8 — predicted annual load ===")

VMIN_LOAD, VMAX_LOAD = 0, 11
SECS_PER_YEAR = 31_557_600
valid["annual_load"] = (valid["Mean_Linear_Conc"]
                        * valid["dis_m3_pyr"] * SECS_PER_YEAR)
valid["log_load"] = np.log10(valid["annual_load"].clip(lower=1) + 1)

top10_load = (valid.nlargest(10, "annual_load")
                   .reset_index(drop=True).copy())
top10_load["rank"] = range(1, 11)
top10_load["lon"] = top10_load.geometry.centroid.x
top10_load["lat"] = top10_load.geometry.centroid.y

MAIN38_OFFSETS = {
    1:  ( 38,   6), 2:  (-55,  20), 3:  ( 40,  18), 4:  ( 40, -16),
    5:  (-55,  28), 6:  (-55,   2), 7:  (-55, -28), 8:  ( 30,  26),
    9:  ( 38,   0), 10: ( 38, -20),
}

# --- Figure 3.8 main map ---
fig, ax = plt.subplots(figsize=(8, 6))
nodata.plot(ax=ax, color="#E0E0E0", edgecolor="none", aspect=None)
valid.plot(ax=ax, column="log_load", cmap="YlOrRd",
           vmin=VMIN_LOAD, vmax=VMAX_LOAD, edgecolor="none",
           legend=True,
           legend_kwds={"label": "log₁₀(Predicted annual load + 1) [items y⁻¹]",
                        "shrink": 0.6, "pad": 0.02})
for _, h in top10_load.iterrows():
    r = int(h["rank"])
    draw_rank_marker(ax, h["lon"], h["lat"], r, h["log_load"],
                     VMIN_LOAD, VMAX_LOAD,
                     offset_pt=MAIN38_OFFSETS[r],
                     marker_size=max(110, 220 - (r - 1) * 10),
                     font_size=10)
ax.set_xlim(-180, 180); ax.set_ylim(-58, 82); ax.set_aspect("equal")
ax.set_xlabel("Longitude (°)", fontsize=10)
ax.set_ylabel("Latitude (°)", fontsize=10)
save(fig, "Figure_3_8.png", CH3_OUT)

# --- Figure 3.8 zoom panels ---
FIG38_ZOOMS = [
    {"out": "Figure_3_8a.png",
     "title": "(a) North America cluster (Mississippi system)",
     "extent": (-96, -84, 34, 43), "ranks": [2, 5, 6, 7, 8, 10],
     "label_offsets": {2: ( 38, -30), 5: (-55,  30), 6: (-55,  -2),
                       7: (-55, -30), 8: ( 38,  30), 10: ( 55,  0)}},
    {"out": "Figure_3_8b.png",
     "title": "(b) East Asia cluster (Yangtze delta, Jiangsu)",
     "extent": (115, 123, 29, 35), "ranks": [1, 3, 4],
     "label_offsets": {1: ( 45,   0), 3: (-55,  28), 4: ( 45, -26)}},
]
for z in FIG38_ZOOMS:
    lon_min, lon_max, lat_min, lat_max = z["extent"]
    sub_v = valid.cx[lon_min:lon_max, lat_min:lat_max]
    sub_n = nodata.cx[lon_min:lon_max, lat_min:lat_max]
    fig, ax = plt.subplots(figsize=(8, 6))
    if not sub_n.empty:
        sub_n.plot(ax=ax, color="#E0E0E0", edgecolor="none", aspect=None)
    sub_v.plot(ax=ax, column="log_load", cmap="YlOrRd",
               vmin=VMIN_LOAD, vmax=VMAX_LOAD, edgecolor="none",
               legend=True,
               legend_kwds={"label": "log₁₀(Annual load + 1) [items y⁻¹]",
                            "shrink": 0.7, "pad": 0.02})
    for r in z["ranks"]:
        h = top10_load[top10_load["rank"] == r].iloc[0]
        draw_rank_marker(ax, h["lon"], h["lat"], r, h["log_load"],
                         VMIN_LOAD, VMAX_LOAD,
                         offset_pt=z["label_offsets"][r],
                         marker_size=max(160, 320 - (r - 1) * 18),
                         font_size=13)
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.set_title(z["title"], fontweight="bold", pad=10)
    save(fig, z["out"], CH3_OUT)


# ============================================================
# Figure 3.13 — six-config CoV uncertainty map
# ============================================================
print("\n=== Figure 3.13 — CoV uncertainty ===")
fig, ax = map_axes(figsize=(14, 7))

def cov_bin(c):
    if pd.isna(c): return "NA"
    if c < 0.5: return "Low (0–0.5)"
    if c < 1.0: return "Relatively low (0.5–1.0)"
    if c < 1.5: return "Medium (1.0–1.5)"
    return "High (>1.5)"

lev6_m["cov_class"] = lev6_m["CoV"].apply(cov_bin)
cov_cmap = {"Low (0–0.5)": "#1A9850", "Relatively low (0.5–1.0)": "#A6DBA0",
            "Medium (1.0–1.5)": "#FEE08B", "High (>1.5)": "#D7191C", "NA": "#E0E0E0"}

# Plot each class with its colour
for cls, c in cov_cmap.items():
    sub = lev6_m[lev6_m["cov_class"] == cls]
    if len(sub) > 0:
        sub.plot(ax=ax, color=c, edgecolor="none", aspect=None)

ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect("equal")
handles = [Patch(color=c, label=cls) for cls, c in cov_cmap.items()]
ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, facecolor="white", edgecolor="#404040")
ax.set_title("Per-sub-basin Coefficient of Variation (CoV) across six model configurations",
             fontweight="bold", pad=8)
save(fig, "Figure_3_13.png", CH3_OUT)


# ============================================================
# Now Ch4 maps — load MARINA shapefile and Chapter_4_Summary
# ============================================================
print("\n\n=== Loading MARINA sub-basins ===")
marina = gpd.read_file(MARINA_SHP)
print(f"  MARINA: {len(marina):,} rows")

ch4 = pd.read_excel(MARINA_XLSX, sheet_name="per_basin")
print(f"  per_basin: {len(ch4):,} rows")

marina = marina.drop(columns=[c for c in ["method","gen_flux","out_flux","ar_sqkm","grd_cells"] if c in marina.columns])
marina_m = marina.merge(ch4, left_on="subbasn", right_on="basin_id", how="left")
print(f"  merged: {len(marina_m):,}, R_MARINA non-na: {marina_m['R_MARINA'].notna().sum():,}, R_new non-na: {marina_m['R_new'].notna().sum():,}")


# ============================================================
# Figure 4.4 — Step-1 filter outcome
# ============================================================
print("\n=== Figure 4.4 — Step-1 filter outcome ===")
def filter_class(row):
    if row["is_masked"]: return "Not to sea"
    if pd.isna(row["R_new"]): return "Greenland"
    if row["method"] == "Calculated": return "Valid"
    return "Lack data or single grid"

marina_m["filter_class"] = marina_m.apply(filter_class, axis=1)
print(marina_m["filter_class"].value_counts())

fig, ax = map_axes(figsize=(14, 7))
fc_colors = {"Valid": "#2C9F2C", "Lack data or single grid": "#FB9A29",
             "Not to sea": "#9F76C9", "Greenland": "#A0A0A0"}
for cls, c in fc_colors.items():
    sub = marina_m[marina_m["filter_class"] == cls]
    if len(sub) > 0:
        sub.plot(ax=ax, color=c, edgecolor="none", aspect=None)

ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect("equal")
handles = [Patch(color=c, label=cls) for cls, c in fc_colors.items()]
ax.legend(handles=handles, loc="lower left", fontsize=10, frameon=True, facecolor="white", edgecolor="#404040")
ax.set_title("Global outcome of the two Step-1 data-quality filters across 10,226 MARINA sub-basins",
             fontweight="bold", pad=8)
save(fig, "Figure_4_4.png", CH4_OUT)


# ============================================================
# Figure 4.6 — calculated / imputed / NA
# ============================================================
print("\n=== Figure 4.6 — calculated / imputed / NA ===")
def method_class(row):
    if row["is_masked"]: return "Not Available"
    if pd.isna(row["R_new"]): return "Not Available"
    if row["method"] == "Calculated": return "Calculated"
    return "Imputed"

marina_m["m_class"] = marina_m.apply(method_class, axis=1)
print(marina_m["m_class"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), gridspec_kw={"width_ratios": [3, 1]})

ax = axes[0]
mc = {"Calculated": "#1F78B4", "Imputed": "#FB9A29", "Not Available": "#A0A0A0"}
for cls, c in mc.items():
    sub = marina_m[marina_m["m_class"] == cls]
    if len(sub) > 0:
        sub.plot(ax=ax, color=c, edgecolor="none", aspect=None)

ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect("equal")
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
handles = [Patch(color=c, label=cls) for cls, c in mc.items()]
ax.legend(handles=handles, loc="lower left", fontsize=10, frameon=True, facecolor="white", edgecolor="#404040")
ax.set_title("(a) Map of method per MARINA sub-basin", fontweight="bold")

# Right pie/bar: by area
ax2 = axes[1]
counts = marina_m.groupby("m_class")["ar_sqkm"].sum() / marina_m["ar_sqkm"].sum() * 100
counts = counts.reindex(["Calculated", "Imputed", "Not Available"], fill_value=0)
n_counts = marina_m["m_class"].value_counts().reindex(["Calculated", "Imputed", "Not Available"], fill_value=0)
ax2.barh(np.arange(3), counts.values, color=[mc[k] for k in counts.index], edgecolor="#252525", lw=0.8)
for i, (lab, pct, n) in enumerate(zip(counts.index, counts.values, n_counts.values)):
    ax2.text(pct + 1, i, f"{pct:.0f}% (n = {n:,})", va="center", fontsize=9)
ax2.set_yticks(np.arange(3))
ax2.set_yticklabels(counts.index)
ax2.set_xlabel("Share of total sub-basin area (%)")
ax2.set_xlim(0, 100)
ax2.set_title("(b) Area share", fontweight="bold")
ax2.invert_yaxis()

plt.tight_layout()
save(fig, "Figure_4_6.png", CH4_OUT)


# ============================================================
# Figure 4.8 — global MP_inside (SUPERVISOR FIX colorbar)
# ============================================================
print("\n=== Figure 4.8 — MP_inside ===")
fig, ax = map_axes(figsize=(14, 7))

vp = marina_m.copy()
vp["log_gen"] = np.log10(vp["gen_flux"].clip(lower=0) + 1)

# Mask NA
na = vp[vp["gen_flux"].isna()]
na.plot(ax=ax, color="#E0E0E0", edgecolor="none", aspect=None)

valid_v = vp[vp["gen_flux"].notna()]
vmax = np.percentile(valid_v["log_gen"], 99.5)
valid_v.plot(ax=ax, column="log_gen", cmap="YlOrRd", vmin=0, vmax=vmax,
             edgecolor="none",
             # SUPERVISOR FIX: colorbar label uses MP_inside
             legend=True, legend_kwds={"label": "log₁₀($MP_{inside}$ + 1, aspect=None) [items y⁻¹]",
                                        "shrink": 0.6, "pad": 0.02})

ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect("equal")
ax.set_title("Global distribution of $MP_{inside}$ across 10,226 MARINA sub-basins",
             fontweight="bold", pad=8)
save(fig, "Figure_4_8.png", CH4_OUT)


# ============================================================
# Figure 4.9 — global MP_out (SUPERVISOR FIX colorbar) — outlets as points
# ============================================================
print("\n=== Figure 4.9 — MP_out ===")
fig, ax = map_axes(figsize=(14, 7))

# Plot land background
marina_m.plot(ax=ax, color="#F4F4F4", edgecolor="none", aspect=None)

# Plot outlets as points coloured by log MP_out
out_pts = marina_m.copy()
out_pts["log_out"] = np.log10(out_pts["out_flux"].clip(lower=0) + 1)
out_pts["lon"] = out_pts.geometry.centroid.x
out_pts["lat"] = out_pts.geometry.centroid.y
v_pts = out_pts[out_pts["out_flux"].notna() & out_pts["out_flux"].gt(0)]
vmax = np.percentile(v_pts["log_out"], 99.5)

sc = ax.scatter(v_pts["lon"], v_pts["lat"], c=v_pts["log_out"], cmap="YlOrRd",
                s=np.clip(v_pts["log_out"] * 6, 2, 60), vmin=0, vmax=vmax,
                edgecolors="none", alpha=0.9)
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
# SUPERVISOR FIX: colorbar label uses MP_out
cbar.set_label("log₁₀($MP_{out}$ + 1) [items y⁻¹]", fontweight="bold")

ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect("equal")
ax.set_title("Global distribution of $MP_{out}$ at MARINA sub-basin outlets",
             fontweight="bold", pad=8)
save(fig, "Figure_4_9.png", CH4_OUT)


# ============================================================
# Figure 4.14 — three-panel R_MARINA / R_new / ΔR  (SUPERVISOR FIX panel-b)
# ============================================================
print("\n=== Figure 4.14 — three-panel retention ===")
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# Panel a — R_MARINA
ax = axes[0]
mp = marina_m.copy()
ax.set_aspect("equal")
ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
boundaries = [0.85, 0.88, 0.91, 0.94, 0.97, 1.00]
cmap = plt.cm.YlOrBr
norm = BoundaryNorm(boundaries, cmap.N)
mp.plot(ax=ax, column="R_MARINA", cmap=cmap, norm=norm, edgecolor="none",
        missing_kwds={"color": "#E0E0E0"}, aspect=None)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, ticks=boundaries)
cbar.set_label("$R_{MARINA}$ = 1 − $FE_{riv,o}$", fontweight="bold")
ax.set_title("(a) MARINA-Multi baseline retention rate $R_{MARINA}$",
             fontweight="bold", loc="left")

# Panel b — R_new (SUPERVISOR FIX: drop "(Calculated)")
ax = axes[1]
ax.set_aspect("equal")
ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
mp.plot(ax=ax, column="R_new", cmap=cmap, norm=norm, edgecolor="none",
        missing_kwds={"color": "#E0E0E0"}, aspect=None)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, ticks=boundaries)
# SUPERVISOR FIX: drop "(Calculated)" — just use the symbolic equation in MP_inside / MP_out
cbar.set_label("$R_{new,j}$ = 1 − $MP_{out}$ / $MP_{inside}$", fontweight="bold")
ax.set_title("(b) New retention rate of this chapter $R_{new}$",
             fontweight="bold", loc="left")

# Panel c — ΔR (diverging map)
ax = axes[2]
ax.set_aspect("equal")
ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
dr_boundaries = [-0.30, -0.20, -0.10, -0.02, 0.02, 0.10, 0.20, 0.30]
dr_cmap = plt.cm.RdBu_r
dr_norm = BoundaryNorm(dr_boundaries, dr_cmap.N)
mp.plot(ax=ax, column="delta_R", cmap=dr_cmap, norm=dr_norm, edgecolor="none",
        missing_kwds={"color": "#E0E0E0"}, aspect=None)
sm = plt.cm.ScalarMappable(cmap=dr_cmap, norm=dr_norm)
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, ticks=dr_boundaries)
cbar.set_label("$\\Delta R$ = $R_{new}$ − $R_{MARINA}$", fontweight="bold")
ax.set_title("(c) Per-MARINA-sub-basin retention difference $\\Delta R$",
             fontweight="bold", loc="left")

plt.tight_layout()
save(fig, "Figure_4_14.png", CH4_OUT)

print("\nAll maps regenerated.")
