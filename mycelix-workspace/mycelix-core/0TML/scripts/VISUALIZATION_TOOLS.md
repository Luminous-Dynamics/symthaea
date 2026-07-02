# Zero-TrustML Visualization Tools

Comprehensive guide to plotting and analysis tools for experiment results.

## Overview

This directory contains publication-quality plotting tools for the Zero-TrustML paper:

1. **`plot_roc_det_curves.py`** - ROC and DET curves for detection performance
2. **`plot_guard_activations.py`** - CM-Safe guard activation heatmaps
3. **`plot_ttd_histograms.py`** - Time-to-detection distribution analysis
4. **`generate_all_figures.sh`** - Master script to generate all figures
5. **`summarize_incremental.py`** - Rolling Table-II preview as experiments complete
6. **`spot_check_artifacts.sh`** - Mid-run CI gate verification

## Quick Start

### Generate All Figures (PNG, 300 DPI)
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML
./scripts/generate_all_figures.sh
```

### Generate Publication-Quality Figures (PDF, 600 DPI)
```bash
./scripts/generate_all_figures.sh pdf 600
```

### Check Experiment Progress
```bash
# Table-II preview
python scripts/summarize_incremental.py | column -t

# Verify CI gates
./scripts/spot_check_artifacts.sh
```

---

## Tool 1: ROC and DET Curves

**File**: `scripts/plot_roc_det_curves.py`
**Purpose**: Compare detection performance across defenses using ROC and DET curves

### Features
- **ROC curves** with AUROC scores
- **DET curves** (log-scale FPR/FNR) for low-error emphasis
- **Multi-seed averaging** with confidence intervals
- **Colorblind-safe palette**
- **Publication-quality** formatting

### Usage

```bash
# All dataset+attack combinations
python scripts/plot_roc_det_curves.py

# Filter by dataset
python scripts/plot_roc_det_curves.py --dataset emnist

# Filter by attack
python scripts/plot_roc_det_curves.py --attack sign_flip

# PDF format at 600 DPI
python scripts/plot_roc_det_curves.py --format pdf --dpi 600 --output-dir figures
```

### Output Files
- `figures/roc_<dataset>_<attack>.png` - ROC curve for each combination
- `figures/det_<dataset>_<attack>.png` - DET curve (log-scale)

### Customization

**Color Palette** (lines 22-33):
```python
DEFENSE_COLORS = {
    'fedavg': '#1f77b4',  # Blue (baseline)
    'pogq_v4_1': '#7f7f7f',  # Gray
    # ... add more
}
```

**Line Styles** (lines 35-45):
```python
DEFENSE_LINESTYLE = {
    'fedavg': '--',  # Baseline dashed
    'pogq_v4_1': '-',  # Solid
}
```

### Example Plot

ROC curve shows AUROC scores in legend:
```
pogq_v4_1 (AUROC=0.982)
cbf (AUROC=0.951)
coord_median_safe (AUROC=0.887)
fedavg (AUROC=0.502)  # Random baseline
```

DET curve emphasizes low-error regions critical for security applications.

---

## Tool 2: Guard Activation Heatmaps

**File**: `scripts/plot_guard_activations.py`
**Purpose**: Visualize when and why CM-Safe guards trigger

### Features
- **Heatmap** of guard activations by round and experiment
- **Summary bar chart** of total activations per guard type
- **4 guard types**: min_clients, norm_clamp, direction_check, reject_all
- **Color-coded by severity**: Red (critical) → Orange (moderate) → Blue (informational)

### Usage

```bash
# All attacks
python scripts/plot_guard_activations.py

# Specific attack
python scripts/plot_guard_activations.py --attack sign_flip

# PDF format
python scripts/plot_guard_activations.py --format pdf --output-dir figures
```

### Output Files
- `figures/guard_heatmap_<attack>.png` - 2x2 heatmap (one per guard type)
- `figures/guard_summary_<attack>.png` - Bar chart of activation counts

### Guard Types

1. **min_clients** (Red): Too few honest clients (< 50%)
2. **norm_clamp** (Orange): Gradient norm exceeds safe threshold
3. **direction_check** (Blue): Gradient direction deviates from median
4. **reject_all** (Gray): Fallback when all guards fail

### Interpretation

- **High activation early** → Attack detected quickly
- **Consistent activation** → Persistent Byzantine behavior
- **Zero activations** → Guards not needed (honest clients) or failed to trigger

---

## Tool 3: Time-to-Detection Histograms

**File**: `scripts/plot_ttd_histograms.py`
**Purpose**: Analyze how quickly each defense detects Byzantine clients

### Features
- **Histogram** of detection round distribution
- **Color-coded bins**: Green (1-3 rounds) → Yellow (4-6) → Orange (7-10) → Red (11+)
- **Statistics overlay**: Mean, median, std dev, min/max
- **Violin plots** for cross-defense comparison

### Usage

```bash
# All defenses and attacks
python scripts/plot_ttd_histograms.py

# Specific defense
python scripts/plot_ttd_histograms.py --defense pogq_v4_1

# Specific attack
python scripts/plot_ttd_histograms.py --attack sign_flip

# Custom max round range
python scripts/plot_ttd_histograms.py --max-rounds 30
```

### Output Files
- `figures/ttd_histogram_<defense>_<attack>.png` - Individual histogram
- `figures/ttd_comparison_<attack>.png` - Violin plot comparing all defenses

### Interpretation

**Fast detection (green bins)**:
- Mean TTD < 3 rounds → Excellent
- Median TTD = 1 → Attack visible immediately

**Slow detection (red bins)**:
- Mean TTD > 10 rounds → Poor
- High std dev → Inconsistent performance

**Comparison plots**:
- Narrow violin → Consistent TTD
- Wide violin → High variance
- Lower position → Faster detection

---

## Tool 4: Incremental Summary

**File**: `scripts/summarize_incremental.py`
**Purpose**: Monitor Table-II results as experiments complete

### Features
- **Rolling preview** of detection metrics (AUROC, TPR, FPR)
- **TSV and CSV output** formats
- **Non-invasive**: Reads completed artifacts without touching running processes

### Usage

```bash
# Formatted table (human-readable)
python scripts/summarize_incremental.py | column -t

# CSV export
python scripts/summarize_incremental.py --csv > table_ii_preview.csv

# Watch mode (refresh every 5 minutes)
watch -n 300 'python scripts/summarize_incremental.py | column -t'
```

### Output Format

**TSV (default)**:
```
dataset  attack      defense          auroc  tpr    fpr
emnist   sign_flip   pogq_v4_1        0.982  0.95   0.08
emnist   sign_flip   cbf              0.951  0.91   0.10
emnist   sign_flip   coord_median_safe 0.887  0.82   0.12
```

**CSV** (`--csv` flag):
```csv
dataset,attack,defense,auroc,tpr,fpr
emnist,sign_flip,pogq_v4_1,0.982,0.95,0.08
...
```

### Integration with Excel/Google Sheets

```bash
# Export to CSV
python scripts/summarize_incremental.py --csv > results.csv

# Open in Google Sheets
# File → Import → Upload → results.csv
```

---

## Tool 5: Spot-Check Artifacts

**File**: `scripts/spot_check_artifacts.sh`
**Purpose**: Verify 3 CI gates on latest completed artifacts

### CI Gates

1. **Gate 1: Conformal FPR ≤ 0.12** (all buckets)
2. **Gate 2: EMA Stability** (flap count ≤ 5)
3. **Gate 3: CM-Safe Guard Coverage** (≥ 1 guard activated)

### Usage

```bash
# Check latest artifacts
./scripts/spot_check_artifacts.sh

# Watch mode (check every 5 minutes)
watch -n 300 ./scripts/spot_check_artifacts.sh
```

### Output

**All gates pass**:
```
✅ Gate 1 PASSED: Conformal FPR (all buckets ≤ 0.12)
✅ Gate 2 PASSED: EMA Stability (all flap counts ≤ 5)
✅ Gate 3 PASSED: CM-Safe Guard Coverage (≥ 1 guard activated)

====================================================================
🎉 ALL CI GATES PASSED
====================================================================
```

**Gate failure**:
```
❌ Gate 1 FAILED: Conformal FPR violations
   label_3: FPR=0.15 n=142 (bucket violated threshold)
   label_7: FPR=0.14 n=98
```

### Integration with CI/CD

```yaml
# .github/workflows/ci.yml
- name: Run sanity slice
  run: python experiments/matrix_runner.py --config configs/sanity_slice.yaml

- name: Verify CI gates
  run: |
    ./scripts/spot_check_artifacts.sh
    if [ $? -ne 0 ]; then
      echo "CI gates failed!"
      exit 1
    fi
```

---

## Tool 6: Master Figure Generator

**File**: `scripts/generate_all_figures.sh`
**Purpose**: One-command generation of all publication figures

### Features
- **Runs all 3 plotting tools** in sequence
- **Configurable format** (PNG, PDF, SVG)
- **Configurable DPI** (300, 600, 1200)
- **Automatic dependency checking**

### Usage

```bash
# Default: PNG at 300 DPI
./scripts/generate_all_figures.sh

# PDF at 600 DPI (publication-quality)
./scripts/generate_all_figures.sh pdf 600

# SVG (vector graphics)
./scripts/generate_all_figures.sh svg
```

### Output Summary

```
✅ Figure generation complete!
Output directory: figures/

Generated figures:
  figures/roc_emnist_sign_flip.png (245K)
  figures/det_emnist_sign_flip.png (198K)
  figures/guard_heatmap_sign_flip.png (412K)
  figures/guard_summary_sign_flip.png (156K)
  figures/ttd_histogram_pogq_v4_1_sign_flip.png (187K)
  figures/ttd_comparison_sign_flip.png (223K)

Figure count: 48

Next steps:
  1. Review figures: open figures/
  2. Include in paper LaTeX: \includegraphics{figures/roc_emnist_sign_flip.png}
  3. Generate PDF versions: ./scripts/generate_all_figures.sh pdf 600
```

---

## Dependencies

### Required Python Packages
```bash
pip install matplotlib seaborn numpy
```

Or using the nix environment:
```bash
nix develop  # All dependencies included
```

### Optional: High-DPI Rendering
For 600+ DPI figures, ensure you have sufficient memory:
- **Minimum**: 4GB RAM
- **Recommended**: 8GB RAM for large datasets

---

## File Structure

```
scripts/
├── plot_roc_det_curves.py        # ROC/DET curve generator
├── plot_guard_activations.py     # Guard heatmap generator
├── plot_ttd_histograms.py        # TTD histogram generator
├── generate_all_figures.sh       # Master figure script
├── summarize_incremental.py      # Rolling Table-II preview
├── spot_check_artifacts.sh       # CI gate verification
└── VISUALIZATION_TOOLS.md        # This file

figures/                           # Output directory
├── roc_*.png                     # ROC curves
├── det_*.png                     # DET curves
├── guard_heatmap_*.png           # Guard activation heatmaps
├── guard_summary_*.png           # Guard activation summaries
├── ttd_histogram_*.png           # TTD histograms
└── ttd_comparison_*.png          # TTD comparisons
```

---

## Common Workflows

### Workflow 1: Monitor Running Experiments

```bash
# Terminal 1: Watch Table-II preview
watch -n 300 'python scripts/summarize_incremental.py | column -t'

# Terminal 2: Check CI gates
watch -n 300 './scripts/spot_check_artifacts.sh'
```

### Workflow 2: Generate Figures for Paper

```bash
# Wait for experiments to complete
tail -f logs/sanity_slice.log

# Generate all figures
./scripts/generate_all_figures.sh pdf 600

# Review figures
open figures/
```

### Workflow 3: Debug Specific Defense

```bash
# Check detection performance
python scripts/plot_roc_det_curves.py --defense pogq_v4_1

# Check guard activations (if CM-Safe)
python scripts/plot_guard_activations.py

# Check detection speed
python scripts/plot_ttd_histograms.py --defense pogq_v4_1
```

### Workflow 4: Compare Attack Scenarios

```bash
# ROC curves for specific attack
python scripts/plot_roc_det_curves.py --attack sign_flip

# TTD comparison across defenses
python scripts/plot_ttd_histograms.py --attack sign_flip
# → Generates ttd_comparison_sign_flip.png
```

---

## Troubleshooting

### No figures generated
**Problem**: Scripts run but produce no output

**Solution**:
```bash
# Check for artifact directories
ls results/artifacts_*/

# Check for detection_metrics.json
find results -name "detection_metrics.json" | head -n 5

# If missing, experiments may not have completed or artifact generation failed
```

### Import errors (matplotlib, seaborn)
**Problem**: `ModuleNotFoundError: No module named 'matplotlib'`

**Solution**:
```bash
# Use nix environment
nix develop
python scripts/plot_roc_det_curves.py

# Or install manually
pip install matplotlib seaborn numpy
```

### Figures look blurry/pixelated
**Problem**: Low-resolution output

**Solution**:
```bash
# Increase DPI
./scripts/generate_all_figures.sh png 600

# Or use vector format
./scripts/generate_all_figures.sh pdf
./scripts/generate_all_figures.sh svg
```

### Memory errors during figure generation
**Problem**: `MemoryError` or system freezes

**Solution**:
```bash
# Reduce DPI
./scripts/generate_all_figures.sh png 150

# Generate figures individually (one attack at a time)
python scripts/plot_roc_det_curves.py --attack sign_flip
python scripts/plot_roc_det_curves.py --attack scaling_x100
# ...
```

---

## Customization Guide

### Change Color Palette

Edit `DEFENSE_COLORS` in each plotting script:

```python
# Use your institutional colors
DEFENSE_COLORS = {
    'fedavg': '#003262',      # Berkeley Blue
    'pogq_v4_1': '#FDB515',   # California Gold
}
```

### Add New Defense

1. Generate results for the new defense
2. Color palette will auto-extend with default colors
3. Optionally add custom color in `DEFENSE_COLORS` dict

### Change Figure Size

```python
# In each plotting script
plt.rcParams['figure.figsize'] = (10, 6)  # Width, Height in inches
```

### Export to Different Formats

All scripts support `--format` flag:
- `png` - Raster (default)
- `pdf` - Vector (recommended for publication)
- `svg` - Vector (for web/presentations)

---

## Paper Integration

### LaTeX Include

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\columnwidth]{figures/roc_emnist_sign_flip.pdf}
  \caption{ROC curves for sign-flip attack on EMNIST.}
  \label{fig:roc_sign_flip}
\end{figure}
```

### Figure Captions

**ROC curves**:
> Figure X: ROC curves comparing detection performance of 8 defenses against sign-flip attack on EMNIST. PoGQ-v4.1 achieves AUROC=0.982, significantly outperforming coordinate-median (0.887) and approaching perfect detection.

**TTD histograms**:
> Figure Y: Time-to-detection distribution for PoGQ-v4.1 against sign-flip attack. Mean TTD of 2.3 rounds demonstrates rapid threat identification, with 95% of Byzantine clients detected within 5 rounds.

**Guard activations**:
> Figure Z: CM-Safe guard activation patterns across 20 rounds. Norm-clamp guard triggers in rounds 2-4 during sign-flip attack, demonstrating effective gradient magnitude anomaly detection.

---

## Future Enhancements

- [ ] **Interactive plots** with Plotly (hover tooltips, zoom)
- [ ] **Confidence bands** for multi-seed ROC curves (bootstrap CI)
- [ ] **Animation** of guard activations over time
- [ ] **3D plots** for multi-dimensional metric space
- [ ] **Automated LaTeX table generation** from JSON artifacts
- [ ] **Statistical significance markers** (Wilcoxon p-values on plots)

---

**Author**: Luminous Dynamics
**Date**: November 8, 2025
**Status**: Production-ready visualization suite
**Paper**: Zero-TrustML: Comprehensive Byzantine Defense for Federated Learning
