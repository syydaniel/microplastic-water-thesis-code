"""Chapter 4 — Figure 4.1 (methodology overview, three-step structure).

Three numbered boxes:
  1. Retention calculation   (inputs -> Yiyang's sub-basin retention L_new)
  2. Descriptive analysis    (distribution + global map of L_new)
  3. MARINA comparison       (delta maps, size-class agreement)

Render a single PNG + PDF that the Ch4 docx embeds as Figure 4.1.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

COL_STEP = "#1f4e79"
COL_IN = "#5b9bd5"
COL_OUT = "#70ad47"
COL_CMP = "#c55a11"
COL_BG = "#f7f7f7"


def box(ax, xy, w, h, text, face, edge="black", fontsize=9, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color="black", weight=weight, wrap=True)


def arrow(ax, start, end, color="black"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color=color,
    ))


def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_axis_off()
    ax.set_facecolor(COL_BG)

    # Title
    ax.text(6, 7.55, "Figure 4.1  Chapter 4 analysis workflow",
            ha="center", fontsize=13, weight="bold")
    ax.text(6, 7.15,
            "Three sequential steps: (1) retention calculation, "
            "(2) descriptive analysis, (3) MARINA comparison.",
            ha="center", fontsize=9, style="italic", color="#444")

    # ---- Step 1 : retention calculation ----
    ax.text(2.0, 6.5, "Step 1 — Retention calculation",
            ha="center", fontsize=11, weight="bold", color=COL_STEP)
    # inputs column
    box(ax, (0.2, 5.2), 1.6, 0.8,
        "Sub-basin shapefile\n(DDM30, n = 10,226)",
        COL_IN, fontsize=8)
    box(ax, (0.2, 4.3), 1.6, 0.8,
        "Microplastic emission flux\n(gen_flux per basin)",
        COL_IN, fontsize=8)
    box(ax, (0.2, 3.4), 1.6, 0.8,
        "Basin outflow flux\n(out_flux per basin)",
        COL_IN, fontsize=8)
    # processing box
    box(ax, (2.2, 4.0), 1.8, 1.5,
        "V6 pipeline\nL_new = 1 -\nout_flux / gen_flux\n(calc. / imputed)",
        "#dbe5f1", fontsize=8, weight="bold")
    arrow(ax, (1.8, 5.6), (2.2, 5.2))
    arrow(ax, (1.8, 4.7), (2.2, 4.75))
    arrow(ax, (1.8, 3.8), (2.2, 4.3))

    # ---- Step 2 : descriptive analysis ----
    ax.text(6.0, 6.5, "Step 2 — Descriptive analysis",
            ha="center", fontsize=11, weight="bold", color=COL_STEP)
    box(ax, (4.8, 4.7), 2.4, 0.8,
        "Histogram of L_new\n(Figure 4.3)", COL_OUT, fontsize=9)
    box(ax, (4.8, 3.7), 2.4, 0.8,
        "Global map of L_new\n(Figure 4.4)", COL_OUT, fontsize=9)
    box(ax, (4.8, 2.7), 2.4, 0.8,
        "Summary statistics by\nbasin-size class", COL_OUT, fontsize=9)
    arrow(ax, (4.0, 4.75), (4.8, 5.1))
    arrow(ax, (4.0, 4.75), (4.8, 4.1))
    arrow(ax, (4.0, 4.75), (4.8, 3.1))

    # ---- Step 3 : MARINA comparison ----
    ax.text(10.0, 6.5, "Step 3 — MARINA comparison",
            ha="center", fontsize=11, weight="bold", color=COL_STEP)
    box(ax, (8.2, 5.4), 3.6, 0.9,
        "MARINA retention\nR_MARINA = 1 - FEriv_o",
        "#fbe5d6", fontsize=9)
    box(ax, (8.2, 4.4), 3.6, 0.9,
        "This study retention\nR_new = L_new = 1 - out_flux / gen_flux",
        "#fbe5d6", fontsize=9)
    box(ax, (8.2, 3.3), 3.6, 0.9,
        "delta_R map (Fig 4.5), size-class\nbars (Fig 4.6), scatter (Fig 4.7)",
        COL_CMP, fontsize=9, weight="bold")
    arrow(ax, (7.2, 4.1), (8.2, 4.85))
    arrow(ax, (8.2, 5.85), (8.2, 5.3))  # decorative
    arrow(ax, (10.0, 4.4), (10.0, 4.2))

    # ---- Bottom band: deliverables ----
    box(ax, (0.5, 1.3), 11.0, 1.4,
        "Deliverables  |  Chapter_4_Summary.xlsx (per_basin, by_size, global)  "
        "+  Figures 4.3-4.7  +  methodology Figure 4.1\n"
        "Single-command run:  python 01_run_analysis.py  ->  outputs/",
        "#efefef", edge="#888", fontsize=9)

    # ---- Footnotes ----
    ax.text(6, 0.6,
            "Terminology: retention = fraction of emitted microplastic that STAYS inside the sub-basin. "
            "Export fraction = fraction that leaves the outlet. retention = 1 - export fraction.\n"
            "L_new = this study sub-basin retention (1 - out_flux / gen_flux); "
            "ratio_final = 1 - L_new = sub-basin export ratio; "
            "FEriv_o = MARINA sub-basin outlet export fraction.\n"
            "delta_R = R_new - R_MARINA = L_new - (1 - FEriv_o) "
            "(positive = this study retains more microplastic than MARINA).",
            ha="center", fontsize=8, style="italic", color="#333")

    fig.tight_layout()
    png = OUT / "Figure_4_1_methodology.png"
    pdf = OUT / "Figure_4_1_methodology.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=COL_BG)
    fig.savefig(pdf, bbox_inches="tight", facecolor=COL_BG)
    plt.close(fig)
    print(f"  -> {png}")
    print(f"  -> {pdf}")


if __name__ == "__main__":
    main()
