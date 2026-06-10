# Symthaea Papers — LaTeX Sources

## Papers (27 total)

**See `../PAPERS_INDEX.md` for the complete index with venues and submission status.**

### Core Architecture Papers (7 — previously written)

| Directory | Title | Target | Status |
|-----------|-------|--------|--------|
| `hai-consciousness/` | Toward Machine Consciousness: Integrated Information, Active Inference, and Hyperdimensional Computing in a Unified Cognitive Architecture | arXiv cs.AI | Ready (28pp, 46 citations) |
| `hai-neurips/` | Hyperdimensional Active Inference: Free Energy Principle in Vector Symbolic Architectures | NeurIPS 2026 | Draft |
| `hdc-cfc/` | Liquid Hypervectors: Closed-Form Continuous-Time Dynamics in Hyperdimensional Space | NeurIPS / ICML | Draft |
| `psych-bench/` | Psych-Bench: A Comprehensive Cognitive Battery for Evaluating Holographic Liquid Brain Architectures | Behavior Research Methods | Draft (38pp, arXiv tarball ready) |
| `metacognitive-ignition/` | Spontaneous Metacognitive Alignment in a Modular Consciousness Architecture | CogSci / NeurIPS Workshop | Draft |
| `stewardship/` | Consciousness-First AI for Species Stewardship: Architectural Requirements for Post-Extinction Civilizational Recovery | AI & Ethics | Draft |
| `spectral-mip/` | Spectral MIP: O(n^3) Minimum Information Partition via Fiedler Ordering and Bordered Cholesky Sweeps | AAAI Workshop | Draft |

### New Research Papers (10 — drafted 2026-03-22)

| Directory | Title | Target | ~Pages |
|-----------|-------|--------|--------|
| `stochastic-resonance/` | Stochastic Resonance in Hyperdimensional Consciousness: Noise-Induced Integration in HDC Systems | Consciousness and Cognition | ~12 |
| `epistemic-gating/` | Epistemic Gating for Consciousness-Aware Language Generation: Eliminating Hallucination via Logit Masking and HDC Binding | ACL / EMNLP | ~13 |
| `substrate-consciousness/` | Consciousness Across Substrates: Feasibility Metrics, Validation Overlays, and Multi-Region Hybrid Implementations | Minds and Machines | ~15 |
| `topological-consciousness/` | Topological Signatures of Consciousness: Persistent Homology, Betti Numbers, and State Classification via Simplicial Complexes | NeuroImage / PLOS Comp Bio | ~14 |
| `embodied-governance/` | Embodied Governance: From Holochain Protocols to Consciousness-Aware Reward Signals | AAMAS | ~13 |
| `consciousness-control/` | Consciousness-Driven Flight and Humanoid Control: Active Inference with Integrated Information Dynamics | CoRL / ICRA | ~14 |
| `neurochemical-consciousness/` | Neurochemical Consciousness: Pharmacokinetic Dynamics, Receptor Subtypes, and Allostatic Load in Embodied Cognition | Frontiers in Computational Neuroscience | ~12 |
| `swarm-consciousness/` | Swarm Consciousness: Federated Trust-Weighted Learning from Peer Phi Distributions | IJCAI | ~10 |
| `consciousness-security/` | Consciousness as a Security Metric: Anomaly Detection via Phi Distributions and Governance Event Analysis | USENIX Security Workshop | ~11 |
| `restorative-consciousness/` | Restorative Consciousness: Moral Topology, Institutional Compliance, and the Ethics of Earning Back Trust | FAccT / AAAI Ethics | ~12 |

## Other directories

| Directory | Contents |
|-----------|----------|
| `standalone-figures/` | TikZ figure sources (architecture, benchmarks, noise robustness, temporal scaling) |
| `shared/` | Shared style files (`plos2015.bst`, `neurips_2024.sty`) — symlinked into papers that need them |

## Building

Each paper compiles from its own directory:

```bash
cd <paper-directory>/
nix-shell -p texliveFull --run "pdflatex <main> && bibtex <main> && pdflatex <main> && pdflatex <main>"
```

Or compile all 10 new papers at once:

```bash
for dir in stochastic-resonance epistemic-gating substrate-consciousness topological-consciousness embodied-governance consciousness-control neurochemical-consciousness swarm-consciousness consciousness-security restorative-consciousness; do
  cd $dir && nix-shell -p texliveFull --run "pdflatex *.tex && bibtex $(basename *.tex .tex) 2>/dev/null; pdflatex *.tex && pdflatex *.tex" && cd ..
done
```

## arXiv Submissions

| Paper | Tarball | Status |
|-------|---------|--------|
| hai-consciousness | `hai-consciousness/arxiv-submission.tar.gz` (121KB) | Ready — cs.AI + cs.NE + q-bio.NC |
| psych-bench | `psych-bench/arxiv-submission.tar.gz` (937KB) | Ready — cs.AI + cs.NE + q-bio.NC |

## Research Program Overview

The 17 papers span 6 research themes:

1. **Core Architecture** (3 papers): HAI, HDC-CfC, Spectral MIP
2. **Consciousness Measurement** (3 papers): Psych-Bench, Topological Consciousness, Stochastic Resonance
3. **Embodied Cognition** (3 papers): Consciousness Control, Neurochemical Consciousness, Substrate Consciousness
4. **Language & Ethics** (3 papers): Epistemic Gating, Restorative Consciousness, Stewardship
5. **Distributed Intelligence** (3 papers): Embodied Governance, Swarm Consciousness, Consciousness Security
6. **Meta-Science** (2 papers): Metacognitive Ignition, HAI-NeurIPS (theoretical foundations)
