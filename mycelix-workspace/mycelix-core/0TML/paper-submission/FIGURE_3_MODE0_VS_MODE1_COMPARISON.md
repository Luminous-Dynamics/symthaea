# Figure 3: Mode 0 vs Mode 1 Direct A/B Comparison at 35% BFT

**Figure Caption**: Direct comparison of Mode 0 (Peer-Comparison) and Mode 1 (Ground Truth - PoGQ) at 35% BFT with identical experimental setup (seed=42, heterogeneous data). Mode 0 experiences complete detector inversion (100% FPR), flagging all honest nodes as Byzantine. Mode 1 achieves reliable detection (0-7.7% FPR) by measuring gradient quality against external validation loss, demonstrating that ground truth validation is necessary for Byzantine-robust federated learning with heterogeneous data.

---

## ASCII Visualization (for LaTeX integration)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Mode 0 vs Mode 1: Direct A/B Comparison (35% BFT, Seed 42)        │
│  Configuration: 20 clients (13 honest, 7 Byzantine), heterogeneous  │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┬────────────────────┬─────────────────────┐
│ Metric                   │ Mode 0 (Peer)      │ Mode 1 (Ground)     │
├──────────────────────────┼────────────────────┼─────────────────────┤
│ Detection Rate           │ ████████████ 100%  │ ████████████  100%  │
│                          │       ✅            │        ✅            │
├──────────────────────────┼────────────────────┼─────────────────────┤
│ False Positive Rate      │ ████████████ 100%  │ ░░░░░░░░░░░░   0%   │
│                          │       ❌            │        ✅            │
├──────────────────────────┼────────────────────┼─────────────────────┤
│ True Positives           │ 7/7                │ 7/7                 │
├──────────────────────────┼────────────────────┼─────────────────────┤
│ False Positives          │ 13/13 (ALL!)       │ 0/13                │
├──────────────────────────┼────────────────────┼─────────────────────┤
│ True Negatives           │ 0/13 (NONE!)       │ 13/13 (ALL!)        │
├──────────────────────────┼────────────────────┼─────────────────────┤
│ Precision                │ 35.0%              │ 100.0%              │
├──────────────────────────┼────────────────────┼─────────────────────┤
│ F1-Score                 │ 51.9%              │ 100.0%              │
└──────────────────────────┴────────────────────┴─────────────────────┘

Key Finding: Mode 0 flags ALL honest nodes as Byzantine (complete
detector inversion), while Mode 1 achieves perfect discrimination.
```

---

## LaTeX Table (for paper inclusion)

```latex
\begin{table}[t]
\centering
\caption{Direct A/B Comparison: Mode 0 (Peer-Comparison) vs Mode 1 (Ground Truth - PoGQ) at 35\% BFT with Identical Experimental Setup (Seed=42, Heterogeneous Data)}
\label{tab:mode0_vs_mode1_direct}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Mode 0 (Peer)} & \textbf{Mode 1 (Ground)} & \textbf{Target} \\
\midrule
Detection Rate & \textcolor{green}{\textbf{100.0\%}} (7/7) & \textcolor{green}{\textbf{100.0\%}} (7/7) & $\geq$80\% \\
False Positive Rate & \textcolor{red}{\textbf{100.0\%}} (13/13) & \textcolor{green}{\textbf{0.0\%}} (0/13) & $\leq$10\% \\
\midrule
True Positives & 7 & 7 & -- \\
False Positives & \textcolor{red}{\textbf{13}} (all honest!) & \textcolor{green}{\textbf{0}} & -- \\
True Negatives & \textcolor{red}{\textbf{0}} (none!) & \textcolor{green}{\textbf{13}} (all!) & -- \\
False Negatives & 0 & 0 & -- \\
\midrule
Precision & 35.0\% & 100.0\% & -- \\
Recall & 100.0\% & 100.0\% & -- \\
F1-Score & 51.9\% & 100.0\% & -- \\
\midrule
Status & \textcolor{red}{\textbf{FAILED}} & \textcolor{green}{\textbf{PASSED}} & -- \\
\bottomrule
\end{tabular}
\vspace{0.5em}
\begin{minipage}{\linewidth}
\footnotesize
\textbf{Configuration:} 20 clients (13 honest, 7 Byzantine = 35\% BFT), heterogeneous data (Dirichlet $\alpha$=0.1 label skew), sign flip attack, pre-trained SimpleCNN model. Both detectors tested on \textbf{identical data} (same seed, same gradients, same validation set). \\[0.3em]
\textbf{Key Finding:} Mode 0 (peer-comparison) experiences \textit{complete detector inversion}, flagging every single honest node as Byzantine (100\% FPR). Mode 1 (ground truth) achieves reliable detection with perfect discrimination (0\% FPR). This demonstrates that ground truth validation is \textbf{necessary} for Byzantine-robust federated learning with heterogeneous data at BFT ratios $\geq$35\%.
\end{minipage}
\end{table}
```

---

## Bar Chart Visualization (for presentation/poster)

### Detection Rate (Both Pass)

```
Detection Rate (Higher is Better)
┌─────────────────────────────────────────────────┐
│ Mode 0:  ████████████████████████████████ 100%  │ ✅
│ Mode 1:  ████████████████████████████████ 100%  │ ✅
└─────────────────────────────────────────────────┘
         0%        25%        50%        75%      100%
```

### False Positive Rate (Lower is Better)

```
False Positive Rate (Lower is Better)
┌─────────────────────────────────────────────────┐
│ Mode 0:  ████████████████████████████████ 100%  │ ❌ COMPLETE FAILURE
│ Mode 1:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%  │ ✅ PERFECT
└─────────────────────────────────────────────────┘
         0%        25%        50%        75%      100%
                               ▲
                          Target: <10%
```

### Confusion Matrix Comparison

```
Mode 0 (Peer-Comparison)              Mode 1 (Ground Truth - PoGQ)
┌───────────────────────────┐         ┌───────────────────────────┐
│        Predicted          │         │        Predicted          │
│     Byz    Honest         │         │     Byz    Honest         │
│   ┌─────────────────┐     │         │   ┌─────────────────┐     │
│ B │  7  │     0     │     │         │ B │  7  │     0     │     │
│ y │ (TP)│   (FN)    │     │         │ y │ (TP)│   (FN)    │     │
│ z ├─────┼───────────┤     │         │ z ├─────┼───────────┤     │
│   │ 13  │     0     │     │         │   │  0  │    13     │     │
│ H │(FP) │   (TN)    │     │         │ H │(FP) │   (TN)    │     │
│   └─────────────────┘     │         │   └─────────────────┘     │
│                           │         │                           │
│ FPR: 100% (13/13) ❌      │         │ FPR: 0% (0/13) ✅         │
│ All honest flagged!       │         │ All honest protected!     │
└───────────────────────────┘         └───────────────────────────┘
```

---

## Analysis: Detector Inversion Explained

```
With Heterogeneous Data at 35% BFT:

Honest Nodes (13):
  ↓ Each has unique local data (Dirichlet α=0.1)
  ↓ Gradients naturally differ
  ↓ Low cosine similarity (~0.05)
  ↓ Appear "spread out" to detector

Byzantine Nodes (7):
  ↓ Sign-flip attack on their local gradients
  ↓ Form a cluster (all flipped)
  ↓ Higher mutual similarity
  ↓ Appear as "majority" to detector

Mode 0 Decision:
  ┌─────────────────────────────┐
  │ Byzantine cluster = "Normal"│
  │ Honest spread = "Outliers"  │
  └─────────────────────────────┘
            ↓
    Complete Inversion
    (100% FPR)

Mode 1 Decision:
  ┌─────────────────────────────┐
  │ Honest: Improve loss = High │
  │ Byzantine: Degrade = Low    │
  └─────────────────────────────┘
            ↓
    Perfect Separation
    (0-7.7% FPR)
```

---

## Recommended Figure Style for IEEE/USENIX

**Option 1: Side-by-Side Bar Chart**
- X-axis: Metric (Detection Rate, FPR, Precision, F1)
- Y-axis: Percentage (0-100%)
- Two bars per metric (Mode 0 in red, Mode 1 in green)
- Horizontal line at 10% for FPR target

**Option 2: Confusion Matrix Grid (2x2)**
- Left: Mode 0 confusion matrix (shows 13 false positives)
- Right: Mode 1 confusion matrix (shows 0 false positives)
- Color-coded cells (green=correct, red=incorrect)

**Option 3: Stacked Metrics Table**
- Table similar to LaTeX version above
- Bold/color-coded pass/fail indicators
- Compact format for space-constrained venues

**Recommendation**: Use **Option 2** (confusion matrix grid) for maximum visual impact. The dramatic difference (13 vs 0 false positives) is immediately clear.

---

## Implementation Notes

### For LaTeX/IEEE Template:
```latex
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
```

### For TikZ/PGFPlots Visualization:
```latex
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
```

### For Matplotlib (if generating figure programmatically):
```python
import matplotlib.pyplot as plt
import numpy as np

metrics = ['Detection\nRate', 'False Positive\nRate', 'Precision', 'F1-Score']
mode0 = [100, 100, 35, 51.9]
mode1 = [100, 0, 100, 100]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, mode0, width, label='Mode 0 (Peer)', color='#e74c3c')
bars2 = ax.bar(x + width/2, mode1, width, label='Mode 1 (Ground)', color='#27ae60')

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Mode 0 vs Mode 1: Direct A/B Comparison at 35% BFT', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper right', fontsize=11)
ax.axhline(y=10, color='gray', linestyle='--', linewidth=1, label='Target FPR (<10%)')
ax.set_ylim([0, 110])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_mode0_vs_mode1.pdf', dpi=300, bbox_inches='tight')
```

---

**File Status**: Ready for paper integration
**Created**: November 5, 2025
**Purpose**: Figure 3 for Section 5.2 direct comparison
