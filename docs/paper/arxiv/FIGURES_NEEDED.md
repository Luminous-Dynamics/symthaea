# Figures Needed for arXiv Submission

## fig1.pdf — AUC Comparison Bar Chart

- **Type**: Grouped bar chart with value labels
- **X-axis**: Three architecture variants (V1 Tabula Rasa, V2 Temporal, V3 Physics-Informed)
- **Y-axis**: AUC (range 0.5 to 1.0, with 0.5 baseline marked as "random")
- **Data**:
  - V1: AUC = 0.778
  - V2: AUC = 0.820
  - V3: AUC = 0.830
- **Annotations**: Delta labels between bars (+0.042 V1->V2, +0.010 V2->V3), horizontal dashed line at 0.5 (random baseline), optional reference lines for Kates-Harbeck LSTM (~0.76) and Tinguely RF (~0.80)
- **Tool**: matplotlib
- **Notes**: Keep clean, no grid clutter. Use colorblind-safe palette.

## fig2.pdf — O(1) Temporal Scaling

- **Type**: Line plot, log-scale x-axis (temporal horizon), linear y-axis (prediction time in ms)
- **X-axis**: Temporal horizon (1 ms to 10,000 s), logarithmic
- **Y-axis**: Mean prediction time (ms), range ~0 to 5 ms
- **Data**:
  | Horizon | Mean Time (ms) |
  |---------|---------------|
  | 0.001 s | 1.5 |
  | 0.01 s  | 1.6 |
  | 0.1 s   | 1.7 |
  | 1 s     | 1.8 |
  | 10 s    | 1.9 |
  | 100 s   | 2.0 |
  | 1000 s  | 2.2 |
  | 10000 s | 2.4 |
- **Additional elements**:
  - Dashed diagonal line showing hypothetical O(T) LSTM scaling (for contrast)
  - Horizontal reference band at ~2 ms showing "O(1) regime"
  - Annotation: "1.534x ratio across 7 orders of magnitude"
- **Tool**: matplotlib
- **Notes**: The near-flat line is the key visual. Make LSTM diagonal steep and clearly labeled.

## fig3.pdf — ROC Curves

- **Type**: ROC curve plot (FPR vs TPR)
- **X-axis**: False Positive Rate (0 to 1)
- **Y-axis**: True Positive Rate (0 to 1)
- **Data**: Reconstructed from confusion matrices at the reported operating points:
  - V1: TPR=0.461, FPR=0.048 at threshold=0.05, AUC=0.778
  - V2: TPR=0.491, FPR=0.016 at threshold=0.03, AUC=0.820
  - V3: TPR=0.480, FPR=0.013 at threshold=0.02, AUC=0.830
- **Additional elements**:
  - Diagonal dashed line (random classifier, AUC=0.5)
  - Operating point markers on each curve
  - Legend with AUC values
- **Tool**: matplotlib
- **Notes**: Full ROC curves are not available from the manuscript data (only single operating points). The script generates approximate curves using a beta-distribution model calibrated to the known AUC and operating point. Mark these as "illustrative" if exact ROC data is not available. Replace with actual ROC data if the evaluation code can export FPR/TPR arrays.

## fig4.pdf — Energy Efficiency Comparison

- **Type**: Horizontal bar chart, logarithmic x-axis
- **X-axis**: Energy per inference (Joules), log scale
- **Y-axis**: System labels
- **Data**:
  | System | Energy/Inference (J) |
  |--------|---------------------|
  | HDC (laptop, 15W) | 0.144 |
  | HDC (desktop, 65W) | 0.622 |
  | Deep learning SOTA (GPU) | ~5.0 (midpoint of 1-10) |
  | GPT-3 (175B params) | 35.5 |
- **Annotations**: Efficiency multipliers (247x, 57x) labeled on bars
- **Tool**: matplotlib
- **Notes**: Log scale essential to show the magnitude difference. Use green shading for HDC bars, gray for baselines.
