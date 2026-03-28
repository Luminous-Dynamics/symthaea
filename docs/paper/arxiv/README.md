# arXiv Submission: Fusion HDC Disruption Prediction

## Compilation

```bash
pdflatex fusion_hdc
pdflatex fusion_hdc  # twice for references
```

## Figure Generation

```bash
python generate_figures.py
```

Generates fig1.pdf through fig4.pdf in the current directory.

Requirements: `pip install matplotlib numpy`

## Submission Checklist

- [ ] Generate figures: `python generate_figures.py` (produces fig1.pdf through fig4.pdf)
- [ ] Replace fig3.pdf with actual ROC data if available (current version uses approximate curves)
- [ ] Verify compilation: `pdflatex fusion_hdc && pdflatex fusion_hdc`
- [ ] Check all numbers match manuscript (Table 1: AUC 0.778/0.820/0.830; Table 2: timing 1.5-2.4 ms; Table 3: energy 0.144-0.622 J)
- [ ] Verify reference count matches (33 references)
- [ ] Upload to arxiv.org
- [ ] Select primary category: **physics.plasm-ph**
- [ ] Select cross-list categories: **cs.LG**, **cs.AI**

## Files

| File | Description |
|------|-------------|
| `fusion_hdc.tex` | Main LaTeX manuscript (revtex4-2, two-column) |
| `fig1.pdf` | AUC comparison bar chart (V1/V2/V3) |
| `fig2.pdf` | O(1) temporal scaling (log-scale horizon vs prediction time) |
| `fig3.pdf` | ROC curves for all three variants |
| `fig4.pdf` | Energy efficiency comparison (log-scale bar chart) |
| `generate_figures.py` | Python script to produce all figures |
| `FIGURES_NEEDED.md` | Detailed figure specifications |
