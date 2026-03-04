#!/usr/bin/env bash
#
# Regenerate all manuscript figures from code.
#
# All output goes to manuscript/figures/ — the single canonical location.
# Run this from the repo root.
#
# Figures produced:
#   1A, 1B  - CONSORT diagrams           (paper_figures/consort_diagrams.py)
#   2, 3    - ROC & PRC curves            (paper_figures/roc_prc.ipynb)
#   4, 5    - Confusion matrices          (paper_figures/confusion_matrices.ipynb)
#   6-9, 12 - Predictive model figures    (predictive-modeling/risk_score_v2/risk_score_v2.py)
#   10, 11  - Feature evolution heatmaps  (paper_figures/feature_heatmap.py)

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Clearing old figures ==="
rm -f manuscript/figures/figure*.png

echo ""
echo "=== 1. CONSORT diagrams (Figures 1A, 1B) ==="
cd paper_figures
python3 consort_diagrams.py
cd ..

echo ""
echo "=== 2. ROC & PRC curves (Figures 2, 3) ==="
python3 -m jupyter nbconvert --to notebook --execute --inplace paper_figures/roc_prc.ipynb

echo ""
echo "=== 3. Confusion matrices (Figures 4, 5) ==="
python3 -m jupyter nbconvert --to notebook --execute --inplace paper_figures/confusion_matrices.ipynb

echo ""
echo "=== 4. Predictive model figures (Figures 6-9, 12) ==="
cd predictive-modeling/risk_score_v2
python3 risk_score_v2.py both
cd ../..

echo ""
echo "=== 5. Feature evolution heatmaps (Figures 10, 11) ==="
cd paper_figures
python3 feature_heatmap.py
cd ..

echo ""
echo "=== Done! ==="
echo "All figures in manuscript/figures/:"
ls -1 manuscript/figures/figure*.png
