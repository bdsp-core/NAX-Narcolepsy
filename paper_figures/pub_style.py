"""
Shared publication style for all manuscript figures.

Import this at the top of every figure script to ensure consistency.

Usage:
    from pub_style import apply_style, SITE_COLORS, CASE_COLOR, CTRL_COLOR, ...
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Journal specs (JAMA Neurology)
# ---------------------------------------------------------------------------
SINGLE_COL_IN = 3.25
DOUBLE_COL_IN = 6.875
MAX_HEIGHT_IN = 9.5
TARGET_DPI = 600
FIG_FORMAT = 'tiff'  # JAMA prefers TIFF; keep PNG as fallback

# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
FONT_FAMILY = 'Arial'
FONT_SIZE_BASE = 9        # body text, tick labels
FONT_SIZE_AXIS_LABEL = 10
FONT_SIZE_TITLE = 11
FONT_SIZE_PANEL_LABEL = 12
FONT_SIZE_SUPTITLE = 12
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATION = 7

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette for SITES
# https://jfly.uni-koeln.de/color/
# ---------------------------------------------------------------------------
OKABE_ITO = {
    'orange':    '#E69F00',
    'sky_blue':  '#56B4E9',
    'green':     '#009E73',
    'yellow':    '#F0E442',
    'blue':      '#0072B2',
    'vermilion': '#D55E00',
    'purple':    '#CC79A7',
    'black':     '#000000',
}

# Site colors — colorblind-safe, distinct
SITE_COLORS = {
    'BCH':      OKABE_ITO['blue'],       # #0072B2
    'BIDMC':    OKABE_ITO['vermilion'],   # #D55E00
    'Emory':    OKABE_ITO['sky_blue'],    # #56B4E9
    'MGB':      OKABE_ITO['green'],       # #009E73
    'Stanford': OKABE_ITO['purple'],      # #CC79A7
}

# Semantic colors for cases vs controls (consistent across all figures)
CASE_COLOR = '#0072B2'      # Okabe-Ito blue — same as BCH but OK, different context
CTRL_COLOR = '#E69F00'      # Okabe-Ito orange

# Accent colors for specific uses
AUROC_COLOR = '#009E73'     # Okabe-Ito green
SENS_COLOR = '#D55E00'      # Okabe-Ito vermilion
NNT_COLOR = '#0072B2'       # matches case color (blue)
FEAT_POS_COLOR = '#0072B2'  # positive feature = higher risk
FEAT_NEG_COLOR = '#E69F00'  # negative feature = lower risk

# Bar chart colors (performance comparison)
BAR_COLORS = ['#0072B2', '#D55E00', '#009E73']  # CV, LOSO, Final

# ---------------------------------------------------------------------------
# Line & marker defaults
# ---------------------------------------------------------------------------
LINE_WIDTH = 1.5
LINE_WIDTH_THICK = 2.5
LINE_WIDTH_THIN = 0.8
REFERENCE_LINE_WIDTH = 1.0
SPINE_WIDTH = 0.6

# ---------------------------------------------------------------------------
# Apply global style
# ---------------------------------------------------------------------------
def apply_style():
    """Apply publication-ready matplotlib style. Call once at script start."""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT_FAMILY, 'Helvetica', 'DejaVu Sans'],
        'font.size': FONT_SIZE_BASE,
        # Axes
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_AXIS_LABEL,
        'axes.linewidth': SPINE_WIDTH,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': 'white',
        # Ticks
        'xtick.labelsize': FONT_SIZE_BASE,
        'ytick.labelsize': FONT_SIZE_BASE,
        'xtick.major.width': SPINE_WIDTH,
        'ytick.major.width': SPINE_WIDTH,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,
        # Legend
        'legend.fontsize': FONT_SIZE_LEGEND,
        'legend.frameon': False,
        # Figure
        'figure.facecolor': 'white',
        'figure.dpi': 100,
        'savefig.dpi': TARGET_DPI,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        # Grid
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


def add_panel_label(ax, label, x=-0.08, y=1.05):
    """Add a bold panel label (A, B, C...) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONT_SIZE_PANEL_LABEL, fontweight='bold',
            va='bottom', ha='right')


def savefig(fig, path, dpi=None, fmt=None):
    """Save figure with publication defaults. Saves both TIFF and PNG."""
    import os
    dpi = dpi or TARGET_DPI
    # Always save PNG for quick viewing
    png_path = path if path.endswith('.png') else os.path.splitext(path)[0] + '.png'
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    # Also save TIFF for submission
    tiff_path = os.path.splitext(png_path)[0] + '.tiff'
    fig.savefig(tiff_path, dpi=dpi, bbox_inches='tight', facecolor='white',
                format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
