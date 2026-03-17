# Symthaea Papers — LaTeX Sources

## Papers

| Directory | Title | Target | Status |
|-----------|-------|--------|--------|
| Directory | Title | Target | Status |
|-----------|-------|--------|--------|
| `hai-consciousness/` | Toward Machine Consciousness: Integrated Information, Active Inference, and Hyperdimensional Computing in a Unified Cognitive Architecture | arXiv cs.AI | Ready (28pp, 46 citations) |
| `hai-neurips/` | Hyperdimensional Active Inference: Free Energy Principle in Vector Symbolic Architectures | NeurIPS 2026 | Draft |
| `hdc-cfc/` | Liquid Hypervectors: Closed-Form Continuous-Time Dynamics in Hyperdimensional Space | NeurIPS / ICML | Draft |
| `psych-bench/` | Psych-Bench: A Comprehensive Cognitive Battery for Evaluating Holographic Liquid Brain Architectures | Behavior Research Methods | Draft (+ supplement) |
| `metacognitive-ignition/` | Spontaneous Metacognitive Alignment in a Modular Consciousness Architecture | CogSci / NeurIPS Workshop | Draft |
| `stewardship/` | Consciousness-First AI for Species Stewardship: Architectural Requirements for Post-Extinction Civilizational Recovery | AI & Ethics | Draft |
| `spectral-mip/` | Spectral MIP: O(n^3) Minimum Information Partition via Fiedler Ordering and Bordered Cholesky Sweeps | AAAI Workshop | Draft |

## Other directories

| Directory | Contents |
|-----------|----------|
| `standalone-figures/` | TikZ figure sources (architecture, benchmarks, noise robustness, temporal scaling) |
| `shared/` | Shared style files (`plos2015.bst`, `neurips_2024.sty`) — symlinked into papers that need them |

## Building

Each paper compiles from its own directory:

```bash
cd hai-consciousness/
nix-shell -p texliveFull --run "pdflatex hai_paper && bibtex hai_paper && pdflatex hai_paper && pdflatex hai_paper"
```

## arXiv submission

Ready tarball: `hai-consciousness/arxiv-submission.tar.gz` (121KB)
Target: cs.AI, cross-list cs.NE + q-bio.NC
