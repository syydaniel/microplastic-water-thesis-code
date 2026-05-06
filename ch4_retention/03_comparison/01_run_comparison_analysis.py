"""Chapter 4 — reproducible analysis pipeline.

Revision 2026-04-20b (comparison-axis, sub-basin scope):
    Both sides of the comparison describe the SAME quantity:
    the share of microplastic generated inside a DDM30 sub-basin that
    stays in the sub-basin (i.e. does not leave through the outlet).
    FEriv_m is NOT applied on either side. This matches the scope
    of the imputation rule of Section 4.2.2 (which only concerns the
    sub-basin outlet).

    This study retention   R_new(j)    = L_new(j)
                                       = 1 - out_flux(j) / gen_flux(j)
    MARINA retention       R_MARINA(j) = 1 - FEriv_o(j)
    delta_R(j)             = R_new(j) - R_MARINA(j)

where out_flux / gen_flux comes from the Chapter 3 flux-reconciliation
pipeline. FEriv_o and FEriv_m are taken AS-IS from the MARINA-Multi
supplementary dataset (2010 / SSP1-RCP2.6; Micella et al., 2024b).

Reminder on terminology: retention = fraction STAYING. If 70% of the
emitted microplastic stays inside the basin system and 30% reaches the
ocean, then retention = 0.70. A positive delta_R means this study retains MORE
microplastic than MARINA at the compared scope; a negative delta_R means
the opposite.

Inputs
------
- retention_info.xlsx          MARINA-Multi reference data (Micella et al., 2024b)
- Shapefile_Retention_Clear.shp  output of this study's flux pipeline

Outputs (in outputs/)
---------------------
- Chapter_4_Summary.xlsx
- Figure_4_3_retention_distribution.png  ECDF of R_new vs R_MARINA + histogram
- Figure_4_4_retention_map.png           3-row map (MARINA | mine | delta_R)
- Figure_4_5_deltaR_map.png              single-panel delta_R map
- Figure_4_6_size_class_bars.png         |delta_R| and |delta_L| by size class
- Figure_4_7_agreement_by_size.png       scatter R_new vs R_MARINA
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Global Times New Roman styling for every figure this pipeline produces.
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
})

MAP_DPI = 600

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

THESIS_ROOT = HERE.parents[2]
RETENTION_XLSX = THESIS_ROOT / "retention_info.xlsx"
V6_SHP = (THESIS_ROOT / "03_Code_Final" / "Chapter_4" / "Active_V6_Pipeline"
          / "04_Final_Outputs" / "Final_Deliverables" / "Shapefile_Retention_Clear.shp")

SIZE_BINS = [0, 4, 10, 50, 200, np.inf]
SIZE_LABELS = ["≤ 4 cells", "5–10 cells", "11–50 cells",
               "51–200 cells", "> 200 cells"]


def continent_from_latlon(lat: float, lon: float) -> str:
    """Rough continent labelling from a polygon centroid.

    Used only to make the top-10 sub-basins table readable. It is not
    a formal continent classification; basins on continental borders
    (e.g. the Caucasus, Suez, the Bering Strait) may be mis-labelled.
    """
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return "Unknown"
    if lat < -60:
        return "Antarctica"
    # Oceania: Australia, New Zealand, New Guinea, Pacific islands
    if -50 < lat < 10 and 110 < lon <= 180:
        return "Oceania"
    if -50 < lat < -10 and -180 <= lon <= -140:
        return "Oceania"
    # Europe
    if 35 < lat <= 72 and -11 < lon <= 40:
        return "Europe"
    # Asia: includes Middle East and Siberia
    if 10 < lat <= 82 and 40 < lon <= 180:
        return "Asia"
    if -12 < lat <= 10 and 95 < lon <= 180:
        return "Asia"
    # Africa
    if -36 < lat <= 37 and -18 < lon <= 52:
        return "Africa"
    # North America (+ Central America)
    if 14 < lat <= 85 and -170 <= lon <= -52:
        return "North America"
    if 7 < lat <= 14 and -105 < lon <= -60:
        return "North America"
    # South America
    if -56 < lat <= 14 and -82 < lon <= -34:
        return "South America"
    return "Other"

COL_MINE = "#1E64C8"   # blue
COL_MAR = "#C74F3B"    # brick
COL_IMP = "#E08A1F"    # orange
COL_GREY = "#D0D0D0"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_inputs() -> gpd.GeoDataFrame:
    gdf = (gpd.read_file(V6_SHP)
           .rename(columns={"subbasn": "basin_id", "ret_fin": "ret_final"}))
    gdf["basin_id"] = gdf["basin_id"].astype(int)

    feriv = pd.read_excel(RETENTION_XLSX, sheet_name="FEriv")
    params = pd.read_excel(RETENTION_XLSX, sheet_name="Retention parameters")
    ref = params.merge(feriv, on="basin_id", how="outer")
    return gdf.merge(ref, on="basin_id", how="left")


def derive_columns(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    L_new = df["ret_final"].astype(float)
    ratio_new = 1.0 - L_new
    L_mar = df["Lmip_dst_10_s1"].astype(float)
    feriv_o_marina = df["FEriv_MIP_dst_o_10_s1"].astype(float)
    feriv_m = df["FEriv_MIP_dst_m_10_s1"].astype(float)

    df["ratio_final"] = ratio_new
    df["L_new"] = L_new
    df["L_MARINA"] = L_mar
    df["delta_L"] = L_new - L_mar

    # MARINA sub-basin retention: 1 - outlet export fraction.
    # FEriv_m is NOT applied, so the two sides share the same
    # sub-basin scope (matches the imputation scope of Section 4.2.2).
    df["R_MARINA"] = 1.0 - feriv_o_marina

    # This study uses the sub-basin retention directly.
    df["R_new"] = L_new
    df["delta_R"] = df["R_new"] - df["R_MARINA"]

    # Kept for reference only (river-mouth-level quantities from
    # earlier revisions; not used in any Section 4.3 statistic).
    df["FE_MARINA_rivermouth"] = feriv_o_marina * feriv_m
    df["R_MARINA_rivermouth"] = 1.0 - df["FE_MARINA_rivermouth"]

    # Single "not available" mask used across ALL panels so the MARINA
    # and this-study maps visually agree: a basin missing on either side
    # is shown grey everywhere.
    df["is_masked"] = df["R_new"].isna() | df["R_MARINA"].isna()

    cells = df["grd_cells"].astype(float)
    df["size_class"] = pd.cut(cells, bins=SIZE_BINS, labels=SIZE_LABELS,
                              right=True, include_lowest=True)
    return df


def _descriptive_stats(df: gpd.GeoDataFrame) -> pd.DataFrame:
    """Descriptive statistics for the three retention quantities.

    Computed on the inner-join of R_new and R_MARINA so that the same
    basin set underpins every row.
    """
    valid = df[~df["is_masked"]]
    rows = []
    for col, label in [("R_new", "This study (R_new)"),
                       ("R_MARINA", "MARINA (1 - FEriv_o)"),
                       ("delta_R", "delta_R (this study - MARINA)")]:
        s = valid[col].dropna()
        rows.append({
            "quantity": label,
            "n": int(s.size),
            "min": float(s.min()),
            "Q1": float(s.quantile(0.25)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "Q3": float(s.quantile(0.75)),
            "max": float(s.max()),
            "std": float(s.std()),
            "IQR": float(s.quantile(0.75) - s.quantile(0.25)),
        })
    return pd.DataFrame(rows)


def summarise(df: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Summarise ONLY over basins with valid values on both sides.

    Inner-join semantics: any basin where either R_new or R_MARINA is NaN
    is excluded from every statistic reported in Section 4.3. The counts
    of included / excluded basins are returned so they can be stated in
    the methodology.
    """
    total = len(df)
    valid = df[~df["is_masked"]].copy()
    n_valid = len(valid)
    n_masked = total - n_valid

    agg = {
        "delta_R":  ["mean", lambda s: s.abs().mean(), "std"],
        "delta_L":  ["mean", lambda s: s.abs().mean(), "std"],
        "R_new":    "mean",
        "R_MARINA": "mean",
        "basin_id": "count",
    }
    by_size = valid.groupby("size_class", observed=True).agg(agg)
    by_size.columns = ["_".join([c for c in col if c]).replace("<lambda_0>", "mean_abs")
                       for col in by_size.columns]
    by_size = by_size.rename(columns={"basin_id_count": "n_basins"}).reset_index()

    def _stats(col):
        s = valid[col].dropna()
        return {"mean": s.mean(), "mean_abs": s.abs().mean(),
                "rmse_vs_zero": float(np.sqrt((s ** 2).mean())), "std": s.std()}
    glob = pd.DataFrame([
        {"quantity": c, **_stats(c)} for c in ["delta_R", "delta_L"]
    ])

    coverage = {
        "n_total": total,
        "n_valid_both": n_valid,
        "n_excluded": n_masked,
        "n_missing_this_thesis": int(df["R_new"].isna().sum()),
        "n_missing_marina": int(df["R_MARINA"].isna().sum()),
    }
    return by_size, glob, coverage


def write_excel(df: gpd.GeoDataFrame, by_size: pd.DataFrame,
                glob: pd.DataFrame, coverage: dict) -> Path:
    out = OUT / "Chapter_4_Summary.xlsx"
    cols = ["basin_id", "ar_sqkm", "grd_cells", "size_class", "method",
            "gen_flux", "out_flux",
            "L_new", "ratio_final", "L_MARINA",
            "FEriv_MIP_dst_o_10_s1", "FEriv_MIP_dst_m_10_s1", "FQrem_10_s1",
            "R_MARINA", "R_new",
            "R_MARINA_rivermouth",
            "delta_R", "delta_L",
            "is_masked"]
    per_basin = pd.DataFrame(df[cols])
    cov_df = pd.DataFrame([
        {"key": "total basins (DDM30)", "value": coverage["n_total"]},
        {"key": "valid on both sides (used for statistics)",
         "value": coverage["n_valid_both"]},
        {"key": "excluded (missing on at least one side)",
         "value": coverage["n_excluded"]},
        {"key": "  - missing on this study side",
         "value": coverage["n_missing_this_thesis"]},
        {"key": "  - missing on MARINA side",
         "value": coverage["n_missing_marina"]},
    ])
    stats_df = _descriptive_stats(df)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        per_basin.to_excel(writer, sheet_name="per_basin", index=False)
        by_size.to_excel(writer, sheet_name="by_size", index=False)
        glob.to_excel(writer, sheet_name="global", index=False)
        stats_df.to_excel(writer, sheet_name="descriptive_stats",
                          index=False)
        cov_df.to_excel(writer, sheet_name="coverage", index=False)
    return out


# ---------------------------------------------------------------------------
# Retention colour scale — the bulk of the data lies in [0.5, 1.0]. The
# [0, 0.5] range is sparse and is folded into a single bin so that the
# remaining six bins span the data-rich part of the distribution.
# ---------------------------------------------------------------------------
RETENTION_BOUNDS = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]

# Finer retention bins, used only by the Appendix version of Figure 4.9.
# The extra breakpoints sit in the data-rich [0.5, 1.0] range so the
# near-constant MARINA values and the continuous L_new field can be told
# apart visually even where both fields sit in the mid-70s to high-90s.
FINE_RETENTION_BOUNDS = [0.0, 0.3, 0.5, 0.65, 0.75,
                         0.82, 0.88, 0.92, 0.95, 0.97, 0.99, 1.0]

# Diverging ΔR bins used in Figure 4.9(c). The narrow central bins
# (|ΔR| ≤ 0.02, 0.05, 0.10) pull the near-zero majority out of a single
# pale-yellow block and give every sub-basin a distinct colour class.
DELTA_R_BOUNDS = [-0.5, -0.2, -0.1, -0.05, -0.02,
                  0.02, 0.05, 0.1, 0.2, 0.5]


def _retention_norm_and_cmap():
    cmap = plt.get_cmap("Spectral_r", len(RETENTION_BOUNDS) - 1)
    norm = mpl.colors.BoundaryNorm(RETENTION_BOUNDS, ncolors=cmap.N,
                                   clip=True)
    return cmap, norm


# ---------------------------------------------------------------------------
# Map panel helpers
# ---------------------------------------------------------------------------
def _panel_style(ax):
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim([-180, 180]); ax.set_ylim([-60, 90])
    # Per author request, the Longitude / Latitude axis labels and the
    # numeric tick labels are bumped up so they are readable at the
    # zoom level of the printed thesis.
    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.4)


def _panel_cbar(fig, ax, norm, cmap, label, ticks=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.12)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax, ticks=ticks)
    # Enlarged to match the axis-label / tick-label sizes above.
    cbar.set_label(label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    return cbar


def _plot_one_map_panel(ax, fig, df, col, cmap, norm, title, cbar_label,
                        ticks=None):
    _panel_style(ax)
    masked = df[df["is_masked"] | df[col].isna()]
    valid = df[~df["is_masked"] & df[col].notna()]
    if len(masked):
        masked.plot(ax=ax, color=COL_GREY, edgecolor="grey", linewidth=0.05)
    valid.plot(column=col, ax=ax, cmap=cmap, norm=norm,
               legend=False, edgecolor="grey", linewidth=0.05)
    # Per author request, the (a)/(b)/(c) marker sits *above* the
    # black axes frame (left-aligned, no outlining box), instead of
    # floating in the top-left corner inside the plot area.
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold",
                 color="#111111", pad=6)
    _panel_cbar(fig, ax, norm, cmap, cbar_label, ticks=ticks)


# ---------------------------------------------------------------------------
# Figure 4.3 — distribution of R (retention = 1 - export fraction)
# ---------------------------------------------------------------------------
def plot_retention_distribution(df: gpd.GeoDataFrame) -> Path:
    """Figure 4.8 — single-panel ECDF comparing the two retention fields.

    The per-method histogram of L_new is the V6 Plot_Retention_Distribution
    figure (reused as Figure 4.7). Figure 4.8 is therefore kept strictly
    comparative: it shows the cumulative distributions of L_new and
    R_MARINA on the inner-join basin set so the two can be read off the
    same curve axis.
    """
    fig, ax = plt.subplots(figsize=(9, 6.4))

    mine = df["R_new"].dropna().sort_values().to_numpy()
    mar = df["R_MARINA"].dropna().sort_values().to_numpy()
    ax.plot(mine, np.linspace(0, 1, len(mine)),
            color=COL_MINE, linewidth=2.8,
            label=r"This study ($R_{\mathrm{new},j}$)")
    ax.plot(mar, np.linspace(0, 1, len(mar)),
            color=COL_MAR, linewidth=2.8, linestyle="--",
            label=(r"MARINA-Multi "
                   r"($R_{\mathrm{MARINA},j}=1-FE_{\mathrm{riv},o,j}$)"))

    # Per user feedback, the two median vertical lines and the 0.5
    # horizontal reference line are no longer drawn on the figure: the
    # medians (both 0.75) and means are reported directly in the
    # Results prose of Section 4.3.3, so annotating them on the figure
    # adds visual clutter without adding information.

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Per author request: keep the x-axis as a dimensionless fraction
    # (0.0, 0.2, ..., 1.0) to match the colour-bar conventions used in
    # the other retention figures (4.9, 4.10). The y-axis stays in
    # percent because "cumulative share of sub-basins" is naturally a
    # share quantity.
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Microplastics retention", fontsize=15)
    ax.set_ylabel("Cumulative share of sub-basins (%)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=13)
    fig.suptitle("Figure 4.8  Cumulative distribution of sub-basin "
                 "microplastic retention: this study vs MARINA-Multi "
                 "(2010)",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    out = OUT / "Figure_4_3_retention_distribution.png"
    fig.savefig(out, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig); return out


# ---------------------------------------------------------------------------
# Figure 4.4 — vertical 3-panel retention comparison
# ---------------------------------------------------------------------------
def plot_retention_map(df: gpd.GeoDataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14, 20), dpi=300,
                             facecolor="white",
                             gridspec_kw={"hspace": 0.28})

    r_cmap, r_norm = _retention_norm_and_cmap()

    _plot_one_map_panel(axes[0], fig, df, "R_MARINA",
                        r_cmap, r_norm,
                        "(a)",
                        r"$R_{\mathrm{MARINA},j}"
                        r"=1-FE_{\mathrm{riv},o,j}$",
                        ticks=RETENTION_BOUNDS)
    _plot_one_map_panel(axes[1], fig, df, "R_new",
                        r_cmap, r_norm,
                        "(b)",
                        r"$R_{\mathrm{new},j}"
                        r"=1-\mathrm{MP}_{out,j}/\mathrm{MP}_{inside,j}$",
                        ticks=RETENTION_BOUNDS)

    # Panel (c): per-basin ΔR with narrow central bins, so the near-zero
    # majority of sub-basins no longer collapses into one pale-yellow
    # block. The bins sit at ±0.02, ±0.05, ±0.10, ±0.20, ±0.50.
    d_cmap = plt.get_cmap("RdBu_r", len(DELTA_R_BOUNDS) - 1)
    d_norm = mpl.colors.BoundaryNorm(DELTA_R_BOUNDS, ncolors=d_cmap.N,
                                     clip=True)

    _plot_one_map_panel(axes[2], fig, df, "delta_R",
                        d_cmap, d_norm,
                        "(c)",
                        r"$\Delta R_{j}"
                        r"=R_{\mathrm{new},j}-R_{\mathrm{MARINA},j}$",
                        ticks=DELTA_R_BOUNDS)

    # Per author request, attach the "Not available" grey-patch legend
    # to every panel (not only panel (a)), and make it slightly larger
    # so it can be read alongside the new 15 pt tick labels.
    for _ax in axes:
        _ax.legend(handles=[mpatches.Patch(color=COL_GREY,
                                           label="Not available")],
                   loc="lower left", frameon=True, framealpha=0.9,
                   fontsize=14)

    fig.suptitle(
        "Figure 4.9  Global sub-basin microplastic retention: "
        r"MARINA-Multi baseline ($1-FE_{\mathrm{riv},o,j}$), this study "
        r"($R_{\mathrm{new},j}$), and their difference",
        fontsize=16, y=0.995)
    out = OUT / "Figure_4_4_retention_map.png"
    fig.savefig(out, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig); return out


# ---------------------------------------------------------------------------
# Figure 4.5 — single-panel global delta_R map
# ---------------------------------------------------------------------------
def plot_delta_retention_map(df: gpd.GeoDataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(15, 9), dpi=300, facecolor="white")
    _panel_style(ax)

    valid = df[~df["is_masked"] & df["delta_R"].notna()]
    masked = df[df["is_masked"] | df["delta_R"].isna()]
    dmax = max(abs(valid["delta_R"].quantile(0.01)),
               abs(valid["delta_R"].quantile(0.99))) or 1e-6
    dmax = float(np.ceil(dmax * 20) / 20)
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)

    if len(masked):
        masked.plot(ax=ax, color=COL_GREY, edgecolor="grey", linewidth=0.05)
    valid.plot(column="delta_R", ax=ax, cmap=cmap, norm=norm,
               legend=False, edgecolor="grey", linewidth=0.05)

    _panel_cbar(fig, ax, norm, cmap,
                "Microplastic retention difference "
                "(this study minus MARINA-Multi)",
                ticks=np.round(np.linspace(-dmax, dmax, 7), 2))

    ax.legend(handles=[mpatches.Patch(color=COL_GREY,
                                      label="Not available")],
              loc="lower left", frameon=True, framealpha=0.9, fontsize=12)

    ax.set_title("Per sub-basin difference in microplastic retention "
                 "between this study and the MARINA-Multi baseline "
                 "(auxiliary; same data as Figure 4.9c)",
                 fontsize=14, pad=10)
    plt.tight_layout()
    out = OUT / "Figure_4_5_deltaR_map.png"
    fig.savefig(out, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig); return out


# ---------------------------------------------------------------------------
# Appendix retention map — finer colour bins for MARINA and L_new
# ---------------------------------------------------------------------------
def plot_retention_map_fine_appendix(df: gpd.GeoDataFrame) -> Path:
    """Appendix counterpart of Figure 4.9 (a) and (b) with finer bins.

    Figure 4.9(a) groups MARINA values into eight colour classes, which
    tends to render as a handful of broad monochrome patches because
    MARINA only uses a small number of discrete size-class constants.
    The appendix map reruns the same two fields through a colour ramp
    with additional breakpoints inside [0.5, 1.0], so the within-class
    spread of the data-driven L_new field and the discrete steps of the
    MARINA field can be told apart visually.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 14), dpi=300,
                             facecolor="white",
                             gridspec_kw={"hspace": 0.28})

    cmap = plt.get_cmap("Spectral_r", len(FINE_RETENTION_BOUNDS) - 1)
    norm = mpl.colors.BoundaryNorm(FINE_RETENTION_BOUNDS,
                                   ncolors=cmap.N, clip=True)

    _plot_one_map_panel(axes[0], fig, df, "R_MARINA",
                        cmap, norm,
                        "(a)  MARINA-Multi retention "
                        r"$R_{\mathrm{MARINA},j}=1-FE_{\mathrm{riv},o,j}$"
                        "  (fine colour bins)",
                        "Microplastic retention (fraction staying)",
                        ticks=FINE_RETENTION_BOUNDS)
    _plot_one_map_panel(axes[1], fig, df, "R_new",
                        cmap, norm,
                        "(b)  This study retention "
                        r"$R_{\mathrm{new},j}=1-\mathrm{MP}_{out,j}"
                        r"/\mathrm{MP}_{inside,j}$"
                        "  (fine colour bins)",
                        "Microplastic retention (fraction staying)",
                        ticks=FINE_RETENTION_BOUNDS)

    axes[0].legend(handles=[mpatches.Patch(color=COL_GREY,
                                           label="Not available")],
                   loc="lower left", frameon=True, framealpha=0.9,
                   fontsize=11)

    fig.suptitle("Appendix Figure 4.A.1  Retention maps with finer "
                 "colour bins (MARINA and this study)",
                 fontsize=16, y=0.995)
    out = OUT / "Figure_4_A1_retention_map_fine.png"
    fig.savefig(out, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig); return out


# ---------------------------------------------------------------------------
# Figure 4.6 — agreement by basin-size class, bar chart
# ---------------------------------------------------------------------------
def plot_size_class_bars(by_size: pd.DataFrame) -> Path:
    """Figure 4.10 — signed mean retention difference by DDM30 size class.

    Per author request, bars use the same red-for-positive / blue-for-
    negative convention as the retention-difference map of Figure 4.9(c):
    red means R_new,j > R_MARINA,j on average in that size class (this
    study retains more of the generated load), blue means the opposite.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(by_size))
    # strip trailing " cells" from label for compact x-tick display
    labels = [str(s).replace(" cells", "").strip()
              for s in by_size["size_class"]]

    # Red for positive (R_new,j > R_MARINA,j), blue for negative
    # (R_new,j < R_MARINA,j). Colours match the diverging RdBu_r ramp
    # used in Figure 4.9(c).
    POS_COLOR = "#B2182B"  # red
    NEG_COLOR = "#2166AC"  # blue
    means = by_size["delta_R_mean"].to_numpy()
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in means]

    bars = ax.bar(x, means, color=colors,
                  edgecolor="white", width=0.7)
    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("MARINA sub-basin size class (number of 0.5° grid cells)", fontsize=13)
    ax.set_ylabel(
        "Mean retention difference\n"
        r"$\Delta R_{j} = R_{\mathrm{new},j} - R_{\mathrm{MARINA},j}$",
        fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for xi, rect, n, m in zip(x, bars, by_size["n_basins"], means):
        # Put the text label on the outer side of each bar so it never
        # overlaps the zero line for small-magnitude classes.
        if m >= 0:
            ty = rect.get_height()
            va = "bottom"
        else:
            ty = rect.get_height()
            va = "top"
        ax.text(xi, ty,
                f"n = {int(n):,}\n" + r"$\Delta R_{j}$ = " + f"{m:+.3f}",
                ha="center", va=va, fontsize=10)

    # Symmetric y-limits around zero so the sign is visually obvious.
    ymax_abs = float(np.abs(means).max())
    pad = ymax_abs * 0.45
    ax.set_ylim(-ymax_abs - pad, ymax_abs + pad)
    fig.suptitle("Figure 4.10  Mean retention difference between this "
                 "study and MARINA-Multi, by basin size",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    out = OUT / "Figure_4_6_size_class_bars.png"
    fig.savefig(out, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig); return out


# ---------------------------------------------------------------------------
# Figure 4.7 — agreement scatter of FE values by basin-size class
# ---------------------------------------------------------------------------
def plot_agreement_scatter(df: gpd.GeoDataFrame) -> Path:
    # 2 rows x 3 columns so each panel is large enough to be legible.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, label in zip(axes_flat, SIZE_LABELS):
        sub = df[df["size_class"] == label].dropna(subset=["R_MARINA", "R_new"])
        ax.scatter(sub["R_MARINA"], sub["R_new"],
                   s=14, alpha=0.40, color=COL_MINE,
                   edgecolor="none")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="1:1 line")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(label, fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.4)
        if len(sub) >= 2:
            r = float(np.corrcoef(sub["R_MARINA"], sub["R_new"])[0, 1])
            rmse = float(np.sqrt(((sub["R_new"] - sub["R_MARINA"]) ** 2).mean()))
            ax.text(0.03, 0.97,
                    f"n = {len(sub):,}\nr = {r:.2f}\nRMSE = {rmse:.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888"))

    # drop the unused sixth axis
    axes_flat[-1].set_axis_off()

    for ax in axes[-1, :]:
        ax.set_xlabel("MARINA-Multi retention  "
                      r"$R_{\mathrm{MARINA},j}=1-FE_{\mathrm{riv},o,j}$",
                      fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(
            "This study retention  "
            r"$R_{\mathrm{new},j}=1-\mathrm{Outload}_{j}/\mathrm{Genload}_{j}$",
            fontsize=12)

    fig.suptitle("Figure 4.A.2  Per sub-basin agreement in microplastic retention, "
                 "by basin-size class (Appendix)",
                 fontsize=15, y=1.00)
    plt.tight_layout()
    out = OUT / "Figure_4_7_agreement_by_size.png"
    fig.savefig(out, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig); return out


# ---------------------------------------------------------------------------
# Figure 4.11 — relationship between ΔR and the annual microplastic generation load
# ---------------------------------------------------------------------------
def plot_delta_vs_load(
        df: gpd.GeoDataFrame
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Scatter + decile-binned curve of the retention difference.

    Returns (figure path, binned DataFrame, top-10 DataFrame). The
    binned curve uses ten equal-count bins on log10(gen_flux) (i.e.,
    deciles), and the ten sub-basins with the largest gen_flux are
    ringed and labelled 1..10 on the scatter so the reader can see
    where the highest-load sub-basins sit on the retention-difference
    axis. The top-10 metadata is written to Excel for the Appendix
    table.
    """
    valid = df[(~df["is_masked"]) & (df["gen_flux"] > 0)].copy()
    valid["log_gen"] = np.log10(valid["gen_flux"].astype(float))

    # Ten equal-count bins: quantile edges at 0/10, 1/10, 2/10, ..., 10/10
    # of log10(gen_flux). Every bin therefore carries ~n/10 sub-basins.
    n_bins = 10
    edges = np.quantile(valid["log_gen"], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    valid["_bin"] = pd.cut(valid["log_gen"], bins=edges,
                           labels=False, include_lowest=True)
    binned = valid.groupby("_bin", observed=True).agg(
        x_mid=("log_gen", "mean"),
        dR_mean=("delta_R", "mean"),
        dR_std=("delta_R", "std"),
        n=("delta_R", "size"),
        gen_low=("gen_flux", "min"),
        gen_high=("gen_flux", "max"),
    ).reset_index()

    # Top-10 sub-basins by annual microplastic generation load.
    top10 = valid.nlargest(10, "gen_flux").copy().reset_index(drop=True)
    top10["rank"] = range(1, len(top10) + 1)
    # Compute centroids on an equal-area projection (Mollweide) so the
    # centroid math stays well-defined, then take lon/lat from WGS84.
    centroids_ea = gpd.GeoSeries(top10.geometry, crs=top10.crs) \
        .to_crs("ESRI:54009").centroid.to_crs(top10.crs)
    # Full precision kept per author instruction ("不要round"); display
    # formatting happens inside 03_build_chapter4_docx.py.
    top10["centroid_lat"] = centroids_ea.y
    top10["centroid_lon"] = centroids_ea.x
    top10["continent"] = [
        continent_from_latlon(lat, lon)
        for lat, lon in zip(top10["centroid_lat"], top10["centroid_lon"])
    ]

    fig, ax = plt.subplots(figsize=(10, 6.2))

    # Single-colour scatter: the size-class split belongs to Figure 4.10
    # and Figure 4.A.2. Here we want the reader's eye on the binned
    # mean points and their ±1 SD error bars, not on basin-size
    # categories.
    ax.scatter(valid["log_gen"], valid["delta_R"],
               s=9, color="#6F6F6F", alpha=0.22,
               edgecolor="none",
               label="Individual sub-basins")

    ax.axhline(0.0, color="#555555", linestyle="--",
               linewidth=1.0, alpha=0.8)

    # Per user feedback: the ten decile bins are discrete summaries of
    # independent sub-basin populations, so they are drawn as discrete
    # error-bar points rather than as a connected line with a shaded
    # envelope. Each point is one decile; the error bar is ±1 SD of
    # the retention difference within that decile.
    ax.errorbar(binned["x_mid"].to_numpy(),
                binned["dR_mean"].to_numpy(),
                yerr=binned["dR_std"].to_numpy(),
                fmt="o", color=COL_MINE, ecolor=COL_MINE,
                elinewidth=1.6, capsize=5, capthick=1.4,
                markersize=9, markerfacecolor=COL_MINE,
                markeredgecolor="white", markeredgewidth=0.9,
                zorder=4,
                label=r"Decile-bin mean $\Delta R_{j}$ $\pm$ 1 SD")

    # Mark the top-10 highest-load sub-basins with red rings + numbered
    # tags. The tags are placed with purely vertical arrows (dx = 0)
    # so no two arrows can cross each other. The vertical offset is
    # cycled through three magnitudes within each sign group so that
    # labels at close x-values do not collide, and the largest |dR|
    # points are assigned the shortest offset (so their labels stay
    # inside the plotting frame).
    ax.scatter(top10["log_gen"], top10["delta_R"],
               s=95, facecolors="none", edgecolors="#C74F3B",
               linewidth=1.8, zorder=5)

    # Sort each sign group so that the most extreme |dR| point gets
    # the smallest vertical offset (label closest to its data point).
    # This guarantees no label exits the axis frame.
    pos_ordered = (top10[top10["delta_R"] >= 0]
                   .sort_values("delta_R", ascending=False).index.tolist())
    neg_ordered = (top10[top10["delta_R"] < 0]
                   .sort_values("delta_R", ascending=True).index.tolist())
    # Pure-vertical (dx = 0) offsets in display points.
    pos_dy_cycle = [40, 78, 116]
    neg_dy_cycle = [-40, -78, -116]

    def _place_tag(i, dy):
        row = top10.loc[i]
        rank = int(row["rank"])
        ax.annotate(
            str(rank),
            xy=(row["log_gen"], row["delta_R"]),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#C74F3B",
            bbox=dict(boxstyle="circle,pad=0.25",
                      fc="white", ec="#C74F3B", lw=1.1),
            arrowprops=dict(arrowstyle="->",
                            color="#C74F3B", lw=0.9, alpha=0.85),
            zorder=6,
        )

    for k, i in enumerate(pos_ordered):
        _place_tag(i, pos_dy_cycle[k % len(pos_dy_cycle)])
    for k, i in enumerate(neg_ordered):
        _place_tag(i, neg_dy_cycle[k % len(neg_dy_cycle)])

    ax.set_xlabel(
        r"$\log_{10}$(MP$_{inside}$, items y$^{-1}$)",
        fontsize=14)
    ax.set_ylabel(
        r"Retention difference $\Delta R_{j} = R_{\mathrm{new},j} - "
        r"R_{\mathrm{MARINA},j}$",
        fontsize=14)
    ax.tick_params(axis="both", labelsize=11)

    # Tighten the y-axis to the data-relevant range. The 1-99 percentile
    # of the raw scatter plus the ±1 SD error-bar extent of the binned
    # points defines a readable window with no outlier cropping.
    dR = valid["delta_R"].dropna().to_numpy()
    lo = float(np.nanmin([np.quantile(dR, 0.01),
                          (binned["dR_mean"] - binned["dR_std"]).min()]))
    hi = float(np.nanmax([np.quantile(dR, 0.99),
                          (binned["dR_mean"] + binned["dR_std"]).max()]))
    half = max(abs(lo), abs(hi)) * 1.05
    half = float(np.ceil(half * 20) / 20)  # round up to nearest 0.05
    # Guarantee every top-10 label stays inside the axis frame: the
    # largest |dR| point is paired with the smallest |dy| (40 pt),
    # which is roughly 0.10 data units at this figure size. The extra
    # 0.20 padding leaves a small visual margin above/below.
    half = max(half, float(np.abs(top10["delta_R"]).max()) + 0.20)
    ax.set_ylim(-half, half)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)

    fig.suptitle(
        r"Figure 4.11  Retention difference ($\Delta R_{j}$) against "
        "annual microplastic generation load",
        fontsize=15, y=1.00)
    plt.tight_layout()
    out = OUT / "Figure_4_8_deltaR_vs_load.png"
    fig.savefig(out, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    binned_out = binned.assign(
        gen_low=binned["gen_low"].map(lambda v: f"{v:.2e}"),
        gen_high=binned["gen_high"].map(lambda v: f"{v:.2e}"),
    )

    # All numeric columns are kept at full float precision per author
    # instruction ("不要round"): downstream docx formatting decides
    # how many decimals to show, but the Excel deliverable carries the
    # raw values so borderline cases such as R_new = 0.99998 are not
    # silently promoted to exactly 1.
    top10_out = pd.DataFrame({
        "rank": top10["rank"].astype(int),
        "basin_id": top10["basin_id"].astype(int),
        "continent": top10["continent"],
        "centroid_lat": top10["centroid_lat"],
        "centroid_lon": top10["centroid_lon"],
        "size_class": top10["size_class"].astype(str),
        "gen_flux_items_yr": top10["gen_flux"],
        "R_new_this_study": top10["R_new"],
        "R_MARINA_1_minus_FEriv_o": top10["R_MARINA"],
        "retention_difference": top10["R_new"] - top10["R_MARINA"],
    })
    return out, binned_out, top10_out


# ---------------------------------------------------------------------------
def main():
    print("Loading inputs...")
    df = load_inputs(); print(f"  {len(df)} sub-basins loaded")
    print("Deriving comparison columns...")
    df = derive_columns(df)
    print("Summarising (valid on BOTH sides only)...")
    by_size, glob, coverage = summarise(df)
    print(f"  total basins: {coverage['n_total']}, "
          f"valid-on-both: {coverage['n_valid_both']}, "
          f"excluded: {coverage['n_excluded']} "
          f"(this-thesis NaN={coverage['n_missing_this_thesis']}, "
          f"MARINA NaN={coverage['n_missing_marina']})")
    print("Writing Excel summary...")
    print(f"  -> {write_excel(df, by_size, glob, coverage)}")
    print("Making figures...")
    # Figure 4.A.1 (finer-colour retention map) has been dropped: at
    # the sub-basin scale the two retention fields are visually too
    # close to tell apart even with the finer ramp. The comparative
    # message now lives in Figure 4.9(c) (difference map) and
    # Figure 4.11 (retention difference vs load), which both resolve
    # the disagreement directly.
    for p in [plot_retention_distribution(df),
              plot_retention_map(df),
              plot_delta_retention_map(df),
              plot_size_class_bars(by_size),
              plot_agreement_scatter(df)]:
        print(f"  -> {p}")
    p_load, binned_load, top10_load = plot_delta_vs_load(df)
    print(f"  -> {p_load}")

    # append the retention-difference vs load binned + top-10 tables
    # to the Excel deliverable
    xlsx_path = OUT / "Chapter_4_Summary.xlsx"
    with pd.ExcelWriter(xlsx_path, mode="a", engine="openpyxl",
                        if_sheet_exists="replace") as writer:
        binned_load.to_excel(writer, sheet_name="deltaR_vs_load",
                             index=False)
        top10_load.to_excel(writer, sheet_name="top10_load",
                            index=False)

    print("\nGlobal summary:")
    print(glob.to_string(index=False))
    print("\nBy size class:")
    print(by_size[["size_class", "n_basins",
                   "delta_R_mean_abs", "delta_L_mean_abs"]].to_string(index=False))
    print("\nRetention difference vs annual microplastic generation load "
          "(10 decile log-bins):")
    print(binned_load[["x_mid", "dR_mean", "dR_std", "n",
                       "gen_low", "gen_high"]].to_string(index=False))
    print("\nTop 10 sub-basins by annual microplastic generation load:")
    print(top10_load.to_string(index=False))


if __name__ == "__main__":
    main()
