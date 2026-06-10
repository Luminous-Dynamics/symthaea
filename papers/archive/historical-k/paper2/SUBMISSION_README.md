# P3: Coherence-Guided Control

**Title**: Coherence-Guided Control: Corridor Discovery and Physics-Respecting Rescue in a Bioelectric Grid

## Submission Details

**Target Journal**: PLoS Computational Biology
**Format**: PLoS LaTeX template (inline `\begin{thebibliography}`)
**Date Prepared**: November 12, 2025; audited February 17, 2026
**Compilation**: `pdflatex manuscript.tex && pdflatex manuscript.tex` (no bibtex needed)

## Contents

- `manuscript.tex` — Main LaTeX source (PLoS format, 518 lines)
- `manuscript.pdf` — Compiled PDF (13 pages, ~258 KB)
- `references.bib` — NOT USED by manuscript (manuscript uses inline bibliography)

## Research Summary

### Track B: SAC Controller with K-Index Feedback
- **+27.8 pp** corridor discovery improvement (50.0% vs 22.2% baseline, p < 0.001)
- ROC AUC = 0.943, PR AUC = 0.934, Brier = 0.164
- N = 48 SAC seeds, N = 8 baseline seeds

### K-Feedback Ablation
- **+37.5 pp** (+300% relative) with K-feedback vs without
- SAC(K): 50.0% [46.1, 54.4], No-K: 12.5% [8.0, 17.0]

### Track C: Morphological Rescue
- Rescue interventions REDUCED mean performance vs passive dynamics
- Cliff's delta = 0.730 (large effect favoring no-rescue)
- N = 10 per arm

## Key Statistics
- 25 references (inline bibliography)
- 4 tables, 0 main-text figures, 5 supplementary figures referenced
- BCa confidence intervals (10,000 stratified resamples)
- Effect sizes: Cliff's delta, Vargha-Delaney A

## Known Issues
- OSF DOI placeholder: `[insert]` appears 3 times (need DOI before submission)
- No main-text figures embedded (supplementary S1-S5 described but not generated as separate files)
- Figure generation from 48-seed data needed (existing track_b_analysis/ figures show earlier 45.6% result)

## Recompilation

```bash
cd /srv/luminous-dynamics/kosmic-lab/papers/paper2
nix-shell -p texliveFull --run "pdflatex -interaction=nonstopmode manuscript.tex && pdflatex -interaction=nonstopmode manuscript.tex"
```
