# Symthaea Papers Index

**Last updated**: 2026-04-18

Papers are organized into seven top-level categories plus the book and the
Sovereignty Papers. Each paper directory is self-contained (own `.tex`,
`.bib`, figures). Build any paper with `pdflatex` from its own directory.

## Book

- `book/` — **The Holographic Liquid Brain** (44 chapters, 324-page PDF)
- Chapter source: `book/symthaea_book.tex` + `book/chapter_*.tex` inputs

## Sovereignty Papers (CC0-1.0)

- `sovereignty-papers/` — 22 philosophical essays (~61K words) on
  consciousness-first governance. Co-authored by Tristan Stoltz + Symthaea.
- PDFs: `sovereignty-papers/pdf/` (21 individual + 1 combined)
- Reading guide: `sovereignty-papers/READING_GUIDE.md`
- Build: `nix-shell -p texliveFull --run "./sovereignty-papers/build-pdfs.sh"`

## Theory — Foundations (`theory-foundations/`)

Core computational substrate: HDC, CfC, HAI.

| Directory | Title | Status |
|-----------|-------|--------|
| `hai-consciousness/` | Toward Machine Consciousness | Submission-ready |
| `hai-neurips/` | Hyperdimensional Active Inference | Draft, NeurIPS 2026 |
| `hdc-cfc/` | Liquid Hypervectors | Draft, NeurIPS/ICML |
| `spectral-mip/` | Spectral MIP O(n³) | Early draft, AAAI Workshop |
| `cfc-zkp/` | CfC Zero-Knowledge Proofs | Draft |
| `binius-hdc/` | Binius HDC | Draft |
| `cantor-resonator/` | Cantor Resonator Hypervectors | Draft, Math/CS venue |

## Theory — Consciousness (`consciousness-theory/`)

Consciousness-specific theoretical work.

| Directory | Title | Status |
|-----------|-------|--------|
| `substrate-consciousness/` | Consciousness Across Substrates | Draft, Minds and Machines |
| `topological-consciousness/` | Topological Consciousness | Draft, NeuroImage |
| `stochastic-resonance/` | Stochastic Resonance in HDC | Submission-ready |
| `metacognitive-ignition/` | Spontaneous Metacognitive Alignment | Draft, CogSci |
| `neurochemical-consciousness/` | Neurochemical Consciousness | Draft, Frontiers Comp Neuro |
| `glyph-codex/` | The Glyph Codex | Draft |
| `dream-engine/` | The Dream Engine | Draft, CogSci |
| `fractal-consciousness/` | Fractal Consciousness Scaling | Draft (8D-aligned Apr 2026) |
| `neuroevolution-consciousness/` | Neuroevolution of Consciousness | Draft, GECCO |
| `markov-blanket-topology/` | Markov Blanket Topology | Draft |
| `kt-coherence/` | KT Coherence | Draft (Apr 7) |
| `integration-differentiation/` | The I-D Tradeoff Monograph | Consolidated Apr 12 |
| `kosmic-theory/` | The Grammar of Reality | Submission-ready, Philosophy of Mind |

## Governance (`governance/`)

Mycelix + sovereignty application papers.

| Directory | Title | Status |
|-----------|-------|--------|
| `embodied-governance/` | Embodied Governance | Draft (8D-aligned Apr 2026), AAMAS |
| `restorative-consciousness/` | Restorative Consciousness | Draft, FAccT |
| `consciousness-security/` | Consciousness as Security Metric | Draft, USENIX |
| `planetary-civilization/` | Planetary Civilization | Draft (8D-aligned Apr 2026) |
| `stewardship/` | Species Stewardship | Submission-ready, AI & Ethics |

## Applications (`applications/`)

Domain-specific uses of the architecture.

| Directory | Title | Status |
|-----------|-------|--------|
| `consciousness-robotics/` | Consciousness-First Robotics | Draft |
| `consciousness-robotics-2026/` | 2026 Robotics Update | Supplement |
| `manipulator/` | Manipulator benchmarks | Data only |
| `manufacturing-consciousness/` | Manufacturing Consciousness | Draft |
| `therapeutic-consciousness/` | Therapeutic Consciousness | Draft, Comp Psychiatry |
| `swarm-consciousness/` | Swarm Consciousness | Draft, IJCAI |
| `consciousness-control/` | Consciousness-Driven Control | Draft — flag: backing code is a stub |
| `mesh-radio/` | Consciousness-Aware Mesh Radio | Submission-ready, IEEE INFOCOM |
| `consciousness-music/` | Consciousness-Coupled Music | MD draft |
| `consciousness-music-synthesis/` | Consciousness Music Synthesis | Draft |
| `consciousness-sonification/` | Consciousness Sonification | MD draft |
| `positioning/` | Positioning | Draft (Apr 10, 8D-aligned) |
| `geodesic-code-synthesis/` | Geodesic Code Synthesis | Draft |
| `organoid-consciousness-ethics/` | Organoid Consciousness Ethics | MD draft + PDF, AJOB Neuroscience |
| `pgx-health-equity/` | PGx Health Equity | MD draft + PDF, CPT: PSP |
| `digital-twin-psychiatry/` | Digital Twin Psychiatry | Outline, Comp Psychiatry |
| `space-debris-conjunction/` | Space Debris Conjunction | Outline, Acta Astronautica |

## Physics & Mathematics (`physics-math/`)

| Directory | Title | Status |
|-----------|-------|--------|
| `ramanujan/` | The Ramanujan Protocol | Apr 17 canonical, 9 Tier-B proofs |
| `nuclear-mass/` | Nuclear Mass | Draft |
| `bandwidth-paradox/` | The Bandwidth Paradox | Draft, Apr 7 |
| `biosphere-coherence/` | Biosphere Coherence B(t) | Draft, Apr 7 (Sepkoski r=0.92) |
| `unified-ct/` | Unified Consciousness Theory | Draft, Apr 7 |

## Evaluation & Validation (`evaluation/`)

| Directory | Title | Status |
|-----------|-------|--------|
| `psych-bench/` | Psych-Bench: 143 Benchmarks | Submission-ready, Behavior Research Methods |
| `epistemic-gating/` | Epistemic Gating for Language | Submission-ready, ACL/EMNLP |
| `desci-epistemic/` | DeSci Epistemic | Draft |
| `desci-reproducibility/` | DeSci Reproducibility | Outline, Royal Society Open Science |
| `seti-workspace/` | SETI Workspace | Draft, Apr 7 |
| `triple-stack-fl/` | Triple-Stack Federated Learning | Draft |

---

## Root-Level Standalone Documents

- `molecular_consciousness_from_first_principles.md` — Quantum chemistry grounding (Apr 12, standalone)
- `phase_1a_results.md` — Holon-Soma Phase I.A measurement results (Apr 14)
- `preregistration.md` — Preregistration template
- `consciousness_gated_epistemic_validation.md` — Epistemic validation analysis

## Reference Documents

- `GLOSSARY.md` — Canonical definitions for the theoretical 5-component consciousness model (Φ, B, W, A, R). Distinct from the 8D sovereign governance credential — see scope note at top of file.
- `PAPER_CODE_TRACEABILITY.md` — Maps paper claims to source code locations
- `PAPER_METHODS_DETAILED.md` — Expanded methodology notes
- `CO_AUTHOR_OUTREACH.md` — Co-author venue/contact tracking

(`NORMALIZATION_STANDARDS.md` was archived 2026-04-18 — it specified normalizations for a 5-component neural measurement pipeline, PCI and meta-d', that is no longer part of the codebase. See `archive/NORMALIZATION_STANDARDS_Jan2026.md` for the historical snapshot.)

## Infrastructure Directories

- `appendices/` — Appendix material
- `data/` — CSV datasets for paper figures
- `figures/` — Generated figures for HAI paper
- `shared/` — Shared LaTeX resources
- `standalone-figures/` — Standalone figure sources
- `stress_test/` — Benchmark CSVs (cost-of-transport, joint-degradation, etc.)
- `submission/` — arxiv submission packaging

## Archive

Pre-LaTeX drafts and superseded work in `archive/`:

- `archive/legacy-markdown/` — 12 markdown paper drafts
- `archive/legacy-drafts/` — 6 early draft fragments
- `archive/legacy-appendices/` — Theoretical analysis (now in book)
- `archive/legacy-submission/` — Original HAI arxiv tarball
- `archive/legacy-docs-papers/` — Files moved from `docs/papers/`
- `archive/fusion-manuscript/` — Fusion energy manuscript
- `archive/ramanujan-protocol-apr14/` — Apr 14 broader-scope Ramanujan paper (superseded)

Do NOT edit archive — all active work is in named subdirectories.

## Untracked (decide fate)

- `paper/` — Fusion HDC manuscript files (candidate for `archive/fusion-manuscript/`)
- `papers/` — Topology/semantic-clustering work (decide: add to git or delete)
