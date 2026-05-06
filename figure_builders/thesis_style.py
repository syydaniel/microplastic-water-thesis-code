"""Unified matplotlib style for the MSc thesis.

Apply at the top of any plotting script:
    from thesis_style import apply_style
    apply_style()

The style enforces:
- Times New Roman 12pt body, bold axis labels (per supervisor)
- 1.2pt axis lines and tick lines for clean print
- 300 dpi save with tight bbox
- A consistent palette for the recurring categories
"""
import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_style():
    """Apply the thesis-wide style. Idempotent — safe to call multiple times."""
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 10,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
        "mathtext.fontset": "cm",
        "mathtext.default": "regular",
    })


# A consistent palette for the recurring categorical groups
PALETTE = {
    # Sampling gear (Ch2)
    "Bulk Water": "#3B82C4",
    "Pumps": "#2C7BB6",
    "Nets": "#D7191C",
    "Others/Unknown": "#999999",
    # Density-separation chemical (Ch2)
    "NaCl": "#1F78B4",
    "ZnCl2": "#33A02C",
    "ZnCl₂": "#33A02C",
    "NaI": "#E31A1C",
    # Mesh-size bins (Ch2)
    "< 1 µm": "#3288BD",
    "1–10 µm": "#66C2A5",
    "10–100 µm": "#FEC44F",
    "> 100 µm": "#D7191C",
    # HydroSHEDS Level 1 regions (Ch2 §2.3.2.1)
    "Africa": "#E69F00",
    "Arctic (North America)": "#56B4E9",
    "Asia": "#009E73",
    "Australia": "#F0E442",
    "Europe": "#0072B2",
    "North America": "#D55E00",
    "South America": "#CC79A7",
    # Sub-basin status (Ch4)
    "Calculated": "#1F78B4",
    "Imputed": "#FB9A29",
    "Not Available": "#A0A0A0",
    "Valid": "#2C9F2C",
    "Lack data or single grid": "#FB9A29",
    "Not to sea": "#9F76C9",
    "Greenland": "#A0A0A0",
}

# Group-letter colour for ΔR sign (Ch4 Fig 4.12, 4.14)
DELTA_R_RED = "#D7191C"
DELTA_R_BLUE = "#2C7BB6"
DELTA_R_GREY = "#A0A0A0"


def set_title(ax, title, loc="left", **kw):
    """Set a panel title (a, b, c) using bold Times New Roman."""
    ax.set_title(title, loc=loc, fontweight="bold", **kw)


def add_panel_label(ax, label, **kw):
    """Add (a) / (b) / (c) label inside a panel, top-left."""
    ax.text(
        0.02, 0.97, label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=14,
        va="top", ha="left",
        **kw,
    )


# Auto-apply when imported (so external scripts only need `import thesis_style`)
apply_style()
