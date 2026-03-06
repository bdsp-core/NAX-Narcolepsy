#!/usr/bin/env python3
"""
Generate CONSORT-style flow diagrams for the narcolepsy manuscript.

Edward Tufte style: no filled boxes, just text and arrows.

Produces two figures:
  1. Cross-sectional classification pipeline (eFigure 1)
  2. Longitudinal prediction pipeline (eFigure 2)

Both start from the same EHR source to emphasize shared data origin.

Usage:
    python consort_diagrams.py
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def draw_text(ax, x, y, title, body, fontsize=8.5, title_size=None,
              color='#222222', ha='center'):
    """Draw a title line (bold) with body text below, no box."""
    if title_size is None:
        title_size = fontsize + 0.5
    ax.text(x, y, title, ha=ha, va='bottom',
            fontsize=title_size, fontweight='bold', color=color, zorder=3)
    ax.text(x, y - 0.005, body, ha=ha, va='top',
            fontsize=fontsize, color=color, zorder=3, linespacing=1.4)


def draw_arrow(ax, x1, y1, x2, y2, color='#555555', lw=0.8):
    """Draw a thin arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=10,
        color=color, linewidth=lw, zorder=1
    )
    ax.add_patch(arrow)


def draw_line(ax, x1, y1, x2, y2, color='#555555', lw=0.8):
    """Draw a simple line (no arrowhead)."""
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=1)


def draw_side_text(ax, x_main, y_from, x_side, y_side, text,
                   fontsize=7.5, color='#666666'):
    """Draw an exclusion side-branch: horizontal line, down arrow, text."""
    draw_line(ax, x_main, y_from, x_side, y_from)
    draw_arrow(ax, x_side, y_from, x_side, y_side + 0.015)
    ax.text(x_side, y_side, text, ha='center', va='top',
            fontsize=fontsize, color=color, zorder=3, linespacing=1.3,
            style='italic')


# =============================================================================
# Shared starting point
# =============================================================================
SHARED_START_TITLE = 'EHR Data from 5 BDSP Sites'
SHARED_START_BODY = 'BCH, BIDMC, Emory, MGB, Stanford\n13,342 patients  |  596 narcolepsy cases (282 NT1, 314 NT2/IH)'

# Original manual annotation counts (pre-reconciliation)
# NT1: 620 notes (271 patients), NT2/IH: 360 (280 patients)
# Unclear: 296 (221 patients), Absent: 7,714 (6,019 patients)
# Total: 8,990, Usable (excl. Unclear): 8,694


# =============================================================================
# eFigure 1: Cross-Sectional Classification Pipeline
# =============================================================================
def make_cross_sectional_consort():
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    cx = 0.44
    excl_x = 0.82

    # --- Row 1: Shared EHR Data Source ---
    y1 = 0.93
    draw_text(ax, cx, y1, SHARED_START_TITLE, SHARED_START_BODY)

    # --- Row 2: Stratified Sampling ---
    y2 = 0.82
    draw_arrow(ax, cx, y1 - 0.04, cx, y2 + 0.025)
    draw_text(ax, cx, y2,
              'Stratified Enrichment Sampling',
              '~250 patients/site, notes >500 words\n'
              '~1,800 notes/site across 3 enrichment strata')

    # --- Row 3: Total Annotated Notes ---
    y3 = 0.70
    draw_arrow(ax, cx, y2 - 0.045, cx, y3 + 0.025)
    draw_text(ax, cx, y3,
              'Manual Annotation',
              '6,498 patients, 8,990 notes annotated\n'
              'by 6 physician annotators')

    # --- Row 4: Annotation breakdown ---
    y4 = 0.58
    draw_arrow(ax, cx, y3 - 0.04, cx, y4 + 0.025)
    draw_text(ax, cx, y4,
              'Manual Annotation by Physician Reviewers',
              'NT1: 620 notes (271 patients)  |  NT2/IH: 360 notes (280 patients)\n'
              'Absent: 7,714 notes  |  Unclear: 296 notes')

    # --- Exclusion side branch ---
    draw_side_text(ax, cx + 0.20, y4 - 0.035, excl_x, y4 - 0.07,
                   'Excluded: 296 "Unclear"',
                   fontsize=7.5)

    # --- Row 5: Final Cohort ---
    y5 = 0.44
    draw_arrow(ax, cx, y4 - 0.06, cx, y5 + 0.025)
    draw_text(ax, cx, y5,
              'Final Classification Cohort',
              '8,694 notes with definitive labels\n'
              '(271 NT1 patients, 280 NT2/IH patients)')

    # --- Split into two tasks ---
    y6 = 0.30
    left_x = 0.24
    right_x = 0.66

    draw_arrow(ax, cx, y5 - 0.04, left_x, y6 + 0.025)
    draw_arrow(ax, cx, y5 - 0.04, right_x, y6 + 0.025)

    draw_text(ax, left_x, y6,
              'NT1 vs. Others',
              'Positive: 620 notes\n'
              'Negative: 8,074 notes (NT2/IH + Absent)')

    draw_text(ax, right_x, y6,
              'NT2/IH vs. Others',
              'Positive: 360 notes\n'
              'Negative: 8,334 notes (NT1 + Absent)')

    # --- Evaluation ---
    # Performance numbers will be filled in after retraining
    y7 = 0.14
    draw_arrow(ax, left_x, y6 - 0.045, left_x, y7 + 0.025)
    draw_arrow(ax, right_x, y6 - 0.045, right_x, y7 + 0.025)

    draw_text(ax, left_x, y7,
              'LOSO Cross-Validation',
              '4 classifiers: LR, RF, GBT, XGB\n'
              'Best: RF (AUROC 0.995, Sens 0.890)',
              fontsize=8)

    draw_text(ax, right_x, y7,
              'LOSO Cross-Validation',
              '4 classifiers: LR, RF, GBT, XGB\n'
              'Best: XGB (AUROC 0.977, Sens 0.586)',
              fontsize=8)

    fig.tight_layout()
    os.makedirs('../manuscript/figures', exist_ok=True)
    fig.savefig('../manuscript/figures/efigure1_consort_cross_sectional.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved: manuscript/figures/efigure1_consort_cross_sectional.png')


# =============================================================================
# eFigure 2: Longitudinal Prediction Pipeline
# =============================================================================
def make_longitudinal_consort():
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    cx = 0.42
    excl_x = 0.82

    # --- Row 1: Shared EHR Data Source ---
    y1 = 0.93
    draw_text(ax, cx, y1, SHARED_START_TITLE, SHARED_START_BODY)

    # --- Row 2: Full longitudinal cohort ---
    y2 = 0.82
    draw_arrow(ax, cx, y1 - 0.04, cx, y2 + 0.025)
    draw_text(ax, cx, y2,
              'Longitudinal Visit Extraction',
              '13,342 patients  |  1,022,458 visits\n'
              '596 cases (282 NT1, 314 NT2/IH)  |  12,746 controls')

    # --- Row 3: Gap Exclusion ---
    y3 = 0.70
    draw_arrow(ax, cx, y2 - 0.045, cx, y3 + 0.025)
    draw_text(ax, cx, y3,
              'After Gap Exclusion (>5 yr between visits)',
              '11,588 patients  |  876,318 visits\n'
              '539 cases (258 NT1, 281 NT2/IH)  |  11,049 controls')

    draw_side_text(ax, cx + 0.22, y3 - 0.005, excl_x, y3 - 0.04,
                   'Excluded: 1,754 patients\n'
                   '24 NT1, 33 NT2/IH, 1,697 controls',
                   fontsize=7.5)

    # --- Row 4: Visit Subsampling ---
    y4 = 0.57
    draw_arrow(ax, cx, y3 - 0.045, cx, y4 + 0.025)
    draw_text(ax, cx, y4,
              'After Visit Subsampling (max 20/patient)',
              '11,588 patients  |  164,383 visits\n'
              '539 cases (258 NT1, 281 NT2/IH)  |  11,049 controls')

    draw_side_text(ax, cx + 0.22, y4 - 0.005, excl_x, y4 - 0.035,
                   'Visits reduced: 876,318 → 164,383\n'
                   'First and last visits preserved',
                   fontsize=7.5, color='#7B68AE')

    # --- Row 5: Sparse Feature Removal ---
    y5 = 0.45
    draw_arrow(ax, cx, y4 - 0.045, cx, y5 + 0.025)
    draw_text(ax, cx, y5,
              'Sparse Feature Removal',
              'Features with <50 non-zero values excluded\n'
              'Chi-squared top 100 features selected per fold')

    # --- Row 6: Temporal Windowing ---
    y6 = 0.34
    draw_arrow(ax, cx, y5 - 0.04, cx, y6 + 0.025)
    draw_text(ax, cx, y6,
              'Temporal Windowing',
              'Training window: [−2.5 yr, −0.5 yr] before diagnosis\n'
              '0.5-year horizon exclusion to avoid diagnostic workup')

    draw_side_text(ax, cx + 0.22, y6 - 0.005, excl_x, y6 - 0.04,
                   'Cases excluded:\n'
                   '343 any-narcolepsy, 192 NT1\n'
                   '(insufficient pre-dx visits)',
                   fontsize=7.5)

    # --- Split into two outcome models ---
    y7 = 0.19
    left_x = 0.22
    right_x = 0.64
    draw_arrow(ax, cx, y6 - 0.055, left_x, y7 + 0.025)
    draw_arrow(ax, cx, y6 - 0.055, right_x, y7 + 0.025)

    draw_text(ax, left_x, y7,
              'Any Narcolepsy Model',
              '196 cases (NT1 + NT2/IH)\n'
              '11,049 controls\n'
              '155,613 visits')

    draw_text(ax, right_x, y7,
              'NT1-Only Model',
              '66 NT1 cases\n'
              '11,049 controls\n'
              '(NT2/IH excluded from cases & controls)')

    # --- Evaluation ---
    y8 = 0.04
    draw_arrow(ax, left_x, y7 - 0.055, left_x, y8 + 0.025)
    draw_arrow(ax, right_x, y7 - 0.055, right_x, y8 + 0.025)

    draw_text(ax, left_x, y8,
              'Evaluation',
              '5-fold CV: AUC 0.835, AUPRC 0.377\n'
              'LOSO: AUC 0.797, AUPRC 0.428')

    draw_text(ax, right_x, y8,
              'Evaluation',
              '5-fold CV: AUC 0.838, AUPRC 0.298\n'
              'LOSO: AUC 0.788, AUPRC 0.285')

    fig.tight_layout()
    os.makedirs('../manuscript/figures', exist_ok=True)
    fig.savefig('../manuscript/figures/efigure2_consort_longitudinal.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved: manuscript/figures/efigure2_consort_longitudinal.png')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_cross_sectional_consort()
    make_longitudinal_consort()
    print('Done. Both CONSORT diagrams generated.')
