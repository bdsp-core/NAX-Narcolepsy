#!/usr/bin/env python3
"""
Generate CONSORT-style flow diagrams for the narcolepsy manuscript.

Produces two figures:
  1. Cross-sectional classification pipeline (Figure 1A)
  2. Longitudinal prediction pipeline (Figure 1B)

Usage:
    python consort_diagrams.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_box(ax, x, y, w, h, text, fontsize=8, bold_first_line=False,
             facecolor='#E8F0FE', edgecolor='#333333', linewidth=1.0):
    """Draw a rounded-rectangle box with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        transform=ax.transData, zorder=2
    )
    ax.add_patch(box)

    if bold_first_line and '\n' in text:
        lines = text.split('\n', 1)
        ax.text(x, y + 0.01, lines[0], ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold', zorder=3)
        ax.text(x, y - 0.01, lines[1], ha='center', va='top',
                fontsize=fontsize, zorder=3)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, zorder=3, linespacing=1.3)


def draw_arrow(ax, x1, y1, x2, y2, color='#333333'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=12,
        color=color, linewidth=1.2, zorder=1
    )
    ax.add_patch(arrow)


def draw_side_box(ax, x_main, y_from, x_side, y_side, w, h, text,
                  fontsize=7.5, facecolor='#FFF3E0'):
    """Draw an exclusion side-branch box with L-shaped connector."""
    # Horizontal line from main flow to side box
    mid_y = y_from
    ax.plot([x_main, x_side], [mid_y, mid_y], color='#333333',
            linewidth=1.0, zorder=1)
    # Arrow into side box
    arrow = FancyArrowPatch(
        (x_side, mid_y), (x_side, y_side + h/2),
        arrowstyle='->', mutation_scale=10,
        color='#333333', linewidth=1.0, zorder=1
    )
    ax.add_patch(arrow)
    draw_box(ax, x_side, y_side, w, h, text, fontsize=fontsize,
             facecolor=facecolor, edgecolor='#999999', linewidth=0.8)


# =============================================================================
# Figure 1A: Cross-Sectional Classification Pipeline
# =============================================================================
def make_cross_sectional_consort():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    cx = 0.45  # Center x for main flow
    bw = 0.38  # Box width
    bh = 0.055  # Box height

    # Title
    ax.text(0.5, 0.97, 'Figure 1A: Cross-Sectional Classification Pipeline',
            ha='center', va='top', fontsize=12, fontweight='bold')

    # --- Row 1: EHR Data Source ---
    y1 = 0.90
    draw_box(ax, cx, y1, bw, bh,
             'EHR Data from 5 BDSP Sites\n'
             'BCH, BIDMC, Emory, MGB, Stanford',
             fontsize=9, bold_first_line=True,
             facecolor='#D4E6F1')

    # --- Row 2: Stratified Sampling ---
    y2 = 0.79
    draw_arrow(ax, cx, y1 - bh/2, cx, y2 + bh/2)
    draw_box(ax, cx, y2, bw, bh + 0.015,
             'Stratified Enrichment Sampling\n'
             '3 groups per task: "almost certainly positive",\n'
             '"almost certainly negative", "maybe"',
             fontsize=8, bold_first_line=True)

    # --- Row 3: Note Selection ---
    y3 = 0.67
    draw_arrow(ax, cx, y2 - (bh + 0.015)/2, cx, y3 + bh/2)
    draw_box(ax, cx, y3, bw, bh,
             'Note Selection\n'
             '~250 patients/site, notes >500 words\n'
             '~1,800 notes/site, ~300/category',
             fontsize=8, bold_first_line=True)

    # --- Row 4: Total Notes ---
    y4 = 0.56
    draw_arrow(ax, cx, y3 - bh/2, cx, y4 + bh/2)
    draw_box(ax, cx, y4, bw, bh,
             'Total Cohort\n'
             '6,498 patients, 8,990 annotated notes',
             fontsize=9, bold_first_line=True,
             facecolor='#D5F5E3')

    # --- Row 5: Annotation ---
    y5 = 0.45
    draw_arrow(ax, cx, y4 - bh/2, cx, y5 + bh/2)
    draw_box(ax, cx, y5, bw, bh + 0.01,
             'Manual Annotation\n'
             '6 physician annotators\n'
             'NT1: 620 | NT2/IH: 360 | Absent: 7,714 | Unclear: 296',
             fontsize=8, bold_first_line=True)

    # --- Exclusion side branch ---
    excl_x = 0.82
    excl_y = 0.39
    draw_side_box(ax, cx + bw/2, y5 - (bh + 0.01)/2 + 0.01, excl_x, excl_y,
                  0.22, 0.04,
                  'Excluded: 296 "Unclear"\n(50-80% confidence)',
                  fontsize=7.5)

    # --- Row 6: Final Cohort ---
    y6 = 0.33
    draw_arrow(ax, cx, y5 - (bh + 0.01)/2, cx, y6 + bh/2)
    draw_box(ax, cx, y6, bw, bh,
             'Final Classification Cohort\n'
             '8,694 notes with definitive labels',
             fontsize=9, bold_first_line=True,
             facecolor='#D5F5E3')

    # --- Split into two tasks ---
    y7 = 0.21
    left_x = 0.25
    right_x = 0.65
    split_bw = 0.28

    # Arrows from center to both sides
    draw_arrow(ax, cx, y6 - bh/2, left_x, y7 + bh/2)
    draw_arrow(ax, cx, y6 - bh/2, right_x, y7 + bh/2)

    draw_box(ax, left_x, y7, split_bw, bh + 0.01,
             'NT1 vs. Others\n'
             'Positive: 620 notes\n'
             'Negative: 8,074 notes (NT2/IH + Absent)',
             fontsize=8, bold_first_line=True,
             facecolor='#EDE7F6')

    draw_box(ax, right_x, y7, split_bw, bh + 0.01,
             'NT2/IH vs. Others\n'
             'Positive: 360 notes\n'
             'Negative: 8,334 notes (NT1 + Absent)',
             fontsize=8, bold_first_line=True,
             facecolor='#EDE7F6')

    # --- Evaluation ---
    y8 = 0.08
    draw_arrow(ax, left_x, y7 - (bh + 0.01)/2, left_x, y8 + bh/2)
    draw_arrow(ax, right_x, y7 - (bh + 0.01)/2, right_x, y8 + bh/2)

    draw_box(ax, left_x, y8, split_bw, bh,
             'LOSO Cross-Validation\n'
             '4 classifiers: LR, RF, GBT, XGB\n'
             'Best: GBT (AUROC 0.994, Sens 0.876)',
             fontsize=7.5, bold_first_line=True,
             facecolor='#FCE4EC')

    draw_box(ax, right_x, y8, split_bw, bh,
             'LOSO Cross-Validation\n'
             '4 classifiers: LR, RF, GBT, XGB\n'
             'Best: XGB (AUROC 0.984, Sens 0.570)',
             fontsize=7.5, bold_first_line=True,
             facecolor='#FCE4EC')

    fig.tight_layout()
    os.makedirs('../manuscript/figures', exist_ok=True)
    fig.savefig('../manuscript/figures/figure1a_consort_cross_sectional.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved: manuscript/figures/figure1a_consort_cross_sectional.png')


# =============================================================================
# Figure 1B: Longitudinal Prediction Pipeline
# =============================================================================
def make_longitudinal_consort():
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    cx = 0.42  # Center x
    bw = 0.42  # Box width
    bh = 0.05  # Box height
    excl_x = 0.82

    # Title
    ax.text(0.5, 0.97, 'Figure 1B: Longitudinal Prediction Pipeline',
            ha='center', va='top', fontsize=12, fontweight='bold')

    # --- Row 1: Initial Cohort ---
    y1 = 0.90
    draw_box(ax, cx, y1, bw, bh + 0.01,
             'Initial Cohort\n'
             '13,342 patients | 1,022,458 visits\n'
             '596 cases (282 NT1, 314 NT2/IH) | 12,746 controls',
             fontsize=8.5, bold_first_line=True,
             facecolor='#D4E6F1')

    # --- Row 2: Gap Exclusion ---
    y2 = 0.76
    draw_arrow(ax, cx, y1 - (bh + 0.01)/2, cx, y2 + bh/2)
    draw_box(ax, cx, y2, bw, bh + 0.005,
             'After Gap Exclusion (>5 yr between visits)\n'
             '11,588 patients | 876,318 visits\n'
             '539 cases (258 NT1, 281 NT2/IH) | 11,049 controls',
             fontsize=8.5, bold_first_line=True)

    # Exclusion side branch
    draw_side_box(ax, cx + bw/2, y2 + 0.005, excl_x, y2 - 0.03,
                  0.24, 0.04,
                  'Excluded: 1,754 patients\n'
                  '24 NT1, 33 NT2/IH, 1,697 controls',
                  fontsize=7.5)

    # --- Row 3: Visit Subsampling ---
    y3 = 0.63
    draw_arrow(ax, cx, y2 - (bh + 0.005)/2, cx, y3 + bh/2)
    draw_box(ax, cx, y3, bw, bh + 0.005,
             'After Visit Subsampling (max 20/patient)\n'
             '11,588 patients | 164,383 visits\n'
             '539 cases (258 NT1, 281 NT2/IH) | 11,049 controls',
             fontsize=8.5, bold_first_line=True)

    # Side note
    draw_side_box(ax, cx + bw/2, y3 + 0.005, excl_x, y3 - 0.03,
                  0.24, 0.035,
                  'Visits reduced: 876,318 → 164,383\n'
                  'First and last visits preserved',
                  fontsize=7.5, facecolor='#F3E5F5')

    # --- Row 4: Sparse Feature Removal ---
    y4 = 0.51
    draw_arrow(ax, cx, y3 - (bh + 0.005)/2, cx, y4 + bh/2)
    draw_box(ax, cx, y4, bw, bh,
             'Sparse Feature Removal\n'
             'Features with <50 non-zero values excluded\n'
             'Chi-squared top 100 features selected per fold',
             fontsize=8.5, bold_first_line=True)

    # --- Row 5: Temporal Windowing ---
    y5 = 0.39
    draw_arrow(ax, cx, y4 - bh/2, cx, y5 + bh/2)
    draw_box(ax, cx, y5, bw, bh + 0.01,
             'Temporal Windowing\n'
             'Training window: [-2.5 yr, -0.5 yr] before diagnosis\n'
             '0.5-year horizon exclusion to avoid diagnostic workup',
             fontsize=8.5, bold_first_line=True)

    # Exclusion
    draw_side_box(ax, cx + bw/2, y5 + 0.005, excl_x, y5 - 0.035,
                  0.24, 0.04,
                  'Cases excluded:\n'
                  '343 any-narcolepsy, 192 NT1\n'
                  '(insufficient pre-dx visits)',
                  fontsize=7.5)

    # --- Split into two outcome models ---
    y6 = 0.24
    left_x = 0.22
    right_x = 0.62
    split_bw = 0.32

    draw_arrow(ax, cx, y5 - (bh + 0.01)/2, left_x, y6 + bh/2)
    draw_arrow(ax, cx, y5 - (bh + 0.01)/2, right_x, y6 + bh/2)

    draw_box(ax, left_x, y6, split_bw, bh + 0.01,
             'Any Narcolepsy Model\n'
             '196 cases (NT1 + NT2/IH)\n'
             '11,049 controls\n'
             '155,613 visits',
             fontsize=8.5, bold_first_line=True,
             facecolor='#D5F5E3')

    draw_box(ax, right_x, y6, split_bw, bh + 0.01,
             'NT1-Only Model\n'
             '66 NT1 cases\n'
             '11,049 controls\n'
             '(NT2/IH excluded from cases & controls)',
             fontsize=8.5, bold_first_line=True,
             facecolor='#D5F5E3')

    # --- Evaluation ---
    y7 = 0.10
    draw_arrow(ax, left_x, y6 - (bh + 0.01)/2, left_x, y7 + bh/2)
    draw_arrow(ax, right_x, y6 - (bh + 0.01)/2, right_x, y7 + bh/2)

    draw_box(ax, left_x, y7, split_bw, bh + 0.01,
             'Evaluation\n'
             '5-fold CV: AUC 0.835, AUPRC 0.377\n'
             'LOSO: AUC 0.797, AUPRC 0.428',
             fontsize=8.5, bold_first_line=True,
             facecolor='#FCE4EC')

    draw_box(ax, right_x, y7, split_bw, bh + 0.01,
             'Evaluation\n'
             '5-fold CV: AUC 0.838, AUPRC 0.298\n'
             'LOSO: AUC 0.788, AUPRC 0.285',
             fontsize=8.5, bold_first_line=True,
             facecolor='#FCE4EC')

    fig.tight_layout()
    os.makedirs('../manuscript/figures', exist_ok=True)
    fig.savefig('../manuscript/figures/figure1b_consort_longitudinal.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved: manuscript/figures/figure1b_consort_longitudinal.png')


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_cross_sectional_consort()
    make_longitudinal_consort()
    print('Done. Both CONSORT diagrams generated.')
