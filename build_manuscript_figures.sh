#!/usr/bin/env bash
#
# Regenerate all manuscript figures from code.
#
# Each script/notebook saves its output directly to manuscript/figures/
# with the correct filename. Run this from the repo root.
#
# Figures produced:
#   1A, 1B  - CONSORT diagrams           (consort_diagrams.py)
#   2, 3    - ROC & PRC curves            (roc_prc.ipynb)
#   4, 5    - Confusion matrices          (confusion_matrices.ipynb)
#   6-9, 12 - Predictive model figures    (risk_score_v2.py)
#   10, 11  - Feature evolution heatmaps  (feature_heatmap.py)

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
jupyter nbconvert --to notebook --execute --inplace paper_figures/roc_prc.ipynb

echo ""
echo "=== 3. Confusion matrices (Figures 4, 5) ==="
jupyter nbconvert --to notebook --execute --inplace paper_figures/confusion_matrices.ipynb

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
echo "Figures in manuscript/figures/:"
ls -1 manuscript/figures/figure*.png
