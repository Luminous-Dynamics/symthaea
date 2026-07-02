# Zero-TrustML Paper - Figure Specifications

**Date**: November 5, 2025
**Status**: Ready for generation and embedding
**Total Figures**: 3 (core) + 2 (optional supplementary)

---

## Core Figures (Required for Paper)

### Figure 1: Mode 1 Performance Across BFT Levels

**Location in Paper**: Section 5.1 (Mode 1 Boundary Testing)

**Data Source**: Table 1 results (20% to 50% BFT)

**Figure Type**: Dual-axis line chart with error bands

**LaTeX Caption**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/mode1_performance_bft.pdf}
\caption{Mode 1 (Ground Truth - PoGQ) performance across Byzantine ratios. \textbf{Left axis (red)}: Detection rate remains 100\% at 20-50\% BFT, exceeding the 33\% peer-comparison ceiling. \textbf{Right axis (blue)}: False positive rate increases from 0\% at 20\% to 10\% at 50\% BFT, staying within acceptable target ($\leq$10\%). Shaded regions show $\pm1\sigma$ across 3 random seeds. \textbf{Key finding}: Ground truth validation maintains reliable Byzantine detection beyond the theoretical 33\% barrier for peer-comparison methods.}
\label{fig:mode1_performance}
\end{figure}
```

**Plot Specifications**:
```python
import matplotlib.pyplot as plt
import numpy as np

# Data from Table 1
bft_ratios = [20, 25, 30, 35, 40, 45, 50]
detection_rates = [100, 100, 100, 100, 100, 100, 100]
fprs = [0, 0, 0, 7.7, 7.7, 9.1, 10.0]
detection_stds = [0, 0, 0, 0, 0, 0, 0]  # Perfect consistency
fpr_stds = [0, 0, 0, 4.3, 4.3, 4.3, 4.3]  # From multi-seed validation

fig, ax1 = plt.subplots(figsize=(7, 5))

# Detection rate (left axis)
color = 'tab:red'
ax1.set_xlabel('Byzantine Ratio (%)', fontsize=12)
ax1.set_ylabel('Detection Rate (%)', color=color, fontsize=12)
line1 = ax1.plot(bft_ratios, detection_rates, 'o-', color=color, linewidth=2,
                 markersize=8, label='Detection Rate')
ax1.fill_between(bft_ratios,
                 np.array(detection_rates) - detection_stds,
                 np.array(detection_rates) + detection_stds,
                 color=color, alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, 110])
ax1.grid(axis='y', alpha=0.3)

# FPR (right axis)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('False Positive Rate (%)', color=color, fontsize=12)
line2 = ax2.plot(bft_ratios, fprs, 's-', color=color, linewidth=2,
                 markersize=8, label='False Positive Rate')
ax2.fill_between(bft_ratios,
                 np.array(fprs) - fpr_stds,
                 np.array(fprs) + fpr_stds,
                 color=color, alpha=0.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, 15])

# 33% barrier line
ax1.axvline(x=33, color='gray', linestyle='--', linewidth=1.5,
            label='33% Peer-Comparison Ceiling')

# 10% FPR target line
ax2.axhline(y=10, color='gray', linestyle=':', linewidth=1.5,
            label='10% FPR Target')

# Legend
lines = line1 + line2 + [plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5),
                          plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5)]
labels = ['Detection Rate (100%)', 'False Positive Rate',
          '33% Ceiling', '10% Target']
ax1.legend(lines, labels, loc='center left', fontsize=10)

plt.title('Mode 1 Performance: Breaking the 33% Barrier', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/mode1_performance_bft.pdf', dpi=300, bbox_inches='tight')
```

---

### Figure 2: Confusion Matrix Across BFT Levels

**Location in Paper**: Section 5.1.3 (Confusion Matrix Analysis)

**Data Source**: Table 1 confusion matrix data

**Figure Type**: 2×2 grid of confusion matrices

**LaTeX Caption**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/confusion_matrix_grid.pdf}
\caption{Confusion matrices for Mode 1 detection across Byzantine ratios. Each matrix shows True Positives (green), False Positives (orange), True Negatives (green), and False Negatives (red). \textbf{20\% BFT}: Perfect discrimination (0\% FPR). \textbf{35\% BFT}: 1 false positive (7.7\% FPR), representing natural quality score overlap with heterogeneous data. \textbf{45\% BFT}: 1 false positive (9.1\% FPR). \textbf{50\% BFT}: 1 false positive (10\% FPR, boundary case). All results maintain 100\% detection (recall) with acceptable precision ($\geq$87.5\%).}
\label{fig:confusion_matrices}
\end{figure}
```

**Plot Specifications**:
```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

configs = [
    (20, 16, 4, 16, 0, 4, 0, "20% BFT", "100% Precision"),
    (35, 13, 7, 12, 1, 7, 0, "35% BFT", "87.5% Precision"),
    (45, 11, 9, 10, 1, 9, 0, "45% BFT", "90.0% Precision"),
    (50, 10, 10, 9, 1, 10, 0, "50% BFT", "90.9% Precision")
]

for idx, (bft, hn, bn, tn, fp, tp, fn, title, precision) in enumerate(configs):
    ax = axes[idx // 2, idx % 2]

    # Create confusion matrix
    matrix = np.array([[tp, fn], [fp, tn]])

    # Plot
    im = ax.imshow(matrix, cmap='YlGn', alpha=0.6, vmin=0, vmax=max(hn, bn))

    # Add values
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, matrix[i, j],
                          ha="center", va="center", color="black", fontsize=20,
                          fontweight='bold')

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nByzantine', 'Predicted\nHonest'], fontsize=10)
    ax.set_yticklabels(['Actual\nByzantine', 'Actual\nHonest'], fontsize=10)

    # Title
    ax.set_title(f'{title}\n{precision}', fontsize=12, fontweight='bold')

    # Annotations
    ax.text(0, -0.4, f'TP={tp}', ha='center', fontsize=9, color='green')
    ax.text(1, -0.4, f'FN={fn}', ha='center', fontsize=9, color='red')
    ax.text(0, 1.4, f'FP={fp}', ha='center', fontsize=9, color='orange')
    ax.text(1, 1.4, f'TN={tn}', ha='center', fontsize=9, color='green')

plt.tight_layout()
plt.savefig('figures/confusion_matrix_grid.pdf', dpi=300, bbox_inches='tight')
```

---

### Figure 3: Mode 0 vs Mode 1 Direct A/B Comparison

**Location in Paper**: Section 5.2 (Mode 0 vs Mode 1)

**Data Source**: `tests/test_mode0_vs_mode1.py` results, `DAY2_DIRECT_COMPARISON_RESULTS.md`

**Figure Type**: Side-by-side confusion matrices

**LaTeX Caption**:
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/mode0_vs_mode1_comparison.pdf}
\caption{Direct A/B comparison of Mode 0 (Peer-Comparison) vs Mode 1 (Ground Truth - PoGQ) at 35\% BFT on identical experimental setup (seed=42, heterogeneous data with Dirichlet $\alpha$=0.1 label skew). \textbf{Left}: Mode 0 exhibits \textit{complete detector inversion}, flagging all 13 honest nodes as Byzantine (100\% FPR) while correctly identifying 7 Byzantine nodes. \textbf{Right}: Mode 1 achieves perfect discrimination with 0\% FPR (seed 42) to 7.7\% FPR (multi-seed), maintaining 100\% detection across all tests. Both detectors tested on same 20 client gradients, same pre-trained model, and same validation set. This demonstrates that peer-comparison methods fundamentally fail with heterogeneous data at Byzantine ratios $\geq$35\%, while ground truth validation remains reliable.}
\label{fig:mode0_vs_mode1}
\end{figure*}
```

**Plot Specifications**:
See `FIGURE_3_MODE0_VS_MODE1_COMPARISON.md` for multiple format options (confusion matrices, bar charts, table).

**Recommended**: Use side-by-side confusion matrix format for maximum visual impact.

---

## Optional Supplementary Figures

### Figure S1: Adaptive Threshold Sweep

**Location**: Section 5.5 (Ablation Study) or Appendix

**Data Source**: Table 9 (Adaptive vs Fixed Threshold)

**LaTeX Caption**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/threshold_sweep.pdf}
\caption{Impact of threshold selection on Mode 1 performance at 35\% BFT. \textbf{Fixed threshold} (τ=0.5) produces 84.6\% FPR, flagging 11/13 honest nodes. \textbf{Adaptive threshold} (τ=0.497, gap-based + MAD) reduces FPR to 7.7\% (91\% reduction), achieving 10.9× improvement while maintaining 100\% detection. Shaded region shows quality score distribution overlap between Byzantine [0.488-0.497] and honest [0.495-0.541] nodes. Adaptive threshold optimally places τ in the gap between distributions.}
\label{fig:threshold_sweep}
\end{figure}
```

---

### Figure S2: Temporal Signal Response (Sleeper Agent)

**Location**: Section 5.4 (Temporal Signal Evaluation) or Appendix

**Data Source**: Table 8 (Sleeper Agent detection)

**LaTeX Caption**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/temporal_signal.pdf}
\caption{Temporal consistency signal response to Sleeper Agent activation at round 5 (30\% BFT). \textbf{Pre-activation} (rounds 1-5): Low temporal confidence (0.17-0.25) as Sleepers behave honestly. \textbf{Activation} (round 6): Temporal confidence increases 2.6× (0.17 $\rightarrow$ 0.45), correctly detecting behavioral change. \textbf{Post-activation} (rounds 7-10): Confidence stabilizes at 0.25-0.45 but remains below ensemble threshold (0.6). While the temporal signal successfully identifies behavioral changes, signal strength is insufficient for reliable detection when combined with weak similarity and magnitude signals due to heterogeneous data. This demonstrates that even temporal consistency requires ground truth validation in realistic federated learning scenarios.}
\label{fig:temporal_signal}
\end{figure}
```

---

## Figure Generation Instructions

### For LaTeX Integration:
1. Generate PDFs using provided Python code
2. Place in `figures/` subdirectory
3. Use `\includegraphics{figures/filename.pdf}` in LaTeX
4. Captions are ready to copy-paste

### For Matplotlib Generation:
```bash
# Navigate to project root
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Create figures directory
mkdir -p paper-submission/figures

# Run figure generation scripts (if available)
python scripts/generate_figures.py

# Or generate individually using specifications above
```

### Figure Format Requirements:
- **Format**: PDF (vector) for LaTeX, PNG (300 dpi) for preview
- **Size**: Single column (3.5"), double column (7.16") for IEEE
- **Fonts**: 8-10pt minimum, matching paper font family
- **Colors**: Colorblind-friendly palette (use seaborn or viridis)
- **Resolution**: 300 dpi for raster elements, vector for lines/text

---

## Figure Placement Guidelines

### IEEE S&P / USENIX Security Format:
- **Figure 1**: Section 5.1, after Table 1 (page ~6)
- **Figure 2**: Section 5.1.3, after confusion matrix discussion (page ~7)
- **Figure 3**: Section 5.2, after Table 3 (page ~8) - **WIDE FORMAT** (span 2 columns)
- **Figure S1**: Appendix or Section 5.5 (page ~10)
- **Figure S2**: Appendix or Section 5.4 (page ~9)

### Placement Code Example:
```latex
% After Table 1 in Section 5.1
\input{figures/mode1_performance_bft}

% After Table 3 in Section 5.2
\begin{figure*}[t]  % Wide figure spanning 2 columns
  \centering
  \includegraphics[width=0.95\textwidth]{figures/mode0_vs_mode1_comparison.pdf}
  \caption{[Caption from Figure 3 spec above]}
  \label{fig:mode0_vs_mode1}
\end{figure*}
```

---

## Quick Reference: Figure Numbers

| Figure | Section | Data Source | Status |
|--------|---------|-------------|--------|
| Figure 1 | 5.1 | Table 1 | ✅ Specified |
| Figure 2 | 5.1.3 | Table 1 | ✅ Specified |
| Figure 3 | 5.2 | Direct test results | ✅ Complete (see FIGURE_3 doc) |
| Figure S1 | 5.5/Appendix | Table 9 | ✅ Specified |
| Figure S2 | 5.4/Appendix | Table 8 | ✅ Specified |

**Total**: 3 core + 2 supplementary = 5 figures

---

**Status**: All figure specifications complete and ready for generation
**Date**: November 5, 2025
**Next Action**: Generate PDFs using provided Python code or create in LaTeX with TikZ
