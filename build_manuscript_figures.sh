#!/usr/bin/env bash
#
# Regenerate all manuscript figures from code.
#
# All output goes to manuscript/figures/ — the single canonical location.
# Run this from the repo root.
#
# Main text figures (3):
#   1       - ROC & PRC curves (best model) (paper_figures/roc_prc.ipynb)
#   2       - Risk score trajectories       (predictive-modeling/risk_score_v2/risk_score_v2.py)
#   3       - NNT analysis                  (predictive-modeling/risk_score_v2/risk_score_v2.py)
#
# Supplementary figures (12):
#   eFig 1  - CONSORT cross-sectional       (paper_figures/consort_diagrams.py)
#   eFig 2  - CONSORT longitudinal          (paper_figures/consort_diagrams.py)
#   eFig 3  - ROC & PRC NT1 (all models)    (paper_figures/roc_prc.ipynb)
#   eFig 4  - ROC & PRC NT2/IH (all models) (paper_figures/roc_prc.ipynb)
#   eFig 5  - Confusion matrices NT1        (paper_figures/confusion_matrices.ipynb)
#   eFig 6  - Confusion matrices NT2/IH     (paper_figures/confusion_matrices.ipynb)
#   eFig 7  - Predictive model performance  (predictive-modeling/risk_score_v2/risk_score_v2.py)
#   eFig 8  - Risk score distributions      (predictive-modeling/risk_score_v2/risk_score_v2.py)
#   eFig 9  - Top predictive features       (predictive-modeling/risk_score_v2/risk_score_v2.py)
#   eFig 10 - Feature heatmap (any narco)   (paper_figures/feature_heatmap.py)
#   eFig 11 - Feature heatmap (NT1)         (paper_figures/feature_heatmap.py)
#   eFig 12 - Swimmer plot                  (paper_figures/swimmer_plot.py)

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Clearing old figures ==="
rm -f manuscript/figures/figure*.png manuscript/figures/efigure*.png

echo ""
echo "=== 1. CONSORT diagrams (eFigures 1, 2) ==="
cd paper_figures
python3 consort_diagrams.py
cd ..

echo ""
echo "=== 2. ROC & PRC curves (Figure 1; eFigures 3, 4) ==="
python3 -m jupyter nbconvert --to notebook --execute --inplace paper_figures/roc_prc.ipynb

echo ""
echo "=== 3. Confusion matrices (eFigures 5, 6) ==="
python3 -m jupyter nbconvert --to notebook --execute --inplace paper_figures/confusion_matrices.ipynb

echo ""
echo "=== 4. Predictive model figures (Figure 2, 3; eFigures 7, 8, 9) ==="
cd predictive-modeling/risk_score_v2
python3 risk_score_v2.py both
cd ../..

echo ""
echo "=== 5. Feature evolution heatmaps (eFigures 10, 11) ==="
cd paper_figures
python3 feature_heatmap.py
cd ..

echo ""
echo "=== 6. Swimmer plot (eFigure 12) ==="
cd paper_figures
python3 swimmer_plot.py
cd ..

echo ""
echo "=== Done! ==="
echo "All figures in manuscript/figures/:"
ls -1 manuscript/figures/*.png
