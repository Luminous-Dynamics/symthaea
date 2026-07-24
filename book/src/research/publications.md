# Publications

27 research papers have been produced during the development of Symthaea, plus a 290-page monograph.

## Monograph

**The Holographic Liquid Brain: Consciousness-First Architecture for Artificial Intelligence**
- 290 pages, 39 chapters, 47 citations, 5 TikZ figures
- Available at `papers/book/symthaea_book.tex`
- Covers all six research themes in a unified narrative

## Ready for Submission (6 papers)

| Paper | Target Venue | Key Result |
|-------|-------------|------------|
| Toward Machine Consciousness | arXiv cs.AI | Full HAI architecture description |
| Psych-Bench: 141 Benchmarks | Behavior Research Methods | 27-domain cognitive evaluation suite |
| The Grammar of Reality | Philosophy of Mind | Kosmic Theory: Spinoza + Whitehead + Tegmark |
| Stochastic Resonance in HDC | Consciousness and Cognition | Noise increases Phi via HDC robustness |
| Epistemic Gating for Language | ACL/EMNLP | Structural hallucination prevention |
| Consciousness-Aware Mesh Radio | IEEE INFOCOM | 3-tier consciousness-driven spectrum management |

## Draft Stage (21 papers)

| Paper | Target Venue | Key Result |
|-------|-------------|------------|
| Liquid Hypervectors | NeurIPS/ICML | Unified HDC-CfC neuron, O(1) temporal jumps |
| Spectral MIP O(n^3) | AAAI Workshop | Real-time Phi via Fiedler ordering + Cholesky sweep |
| Hyperdimensional Active Inference | NeurIPS 2026 | FEP integration with HDC representation |
| Spontaneous Metacognitive Alignment | CogSci | GWT-HOT d' = 3.63 without explicit coupling |
| Consciousness Across Substrates | Minds and Machines | 8 substrates, 9 requirements, honest confidence |
| Topological Consciousness | NeuroImage | TDA classification 94.2%, Betti number signatures |
| Embodied Governance | AAMAS | Consciousness-gated Holochain governance |
| Consciousness-Driven Control | CoRL/ICRA | 43% faster perturbation recovery via FEP |
| Swarm Consciousness | IJCAI | Collective Phi through peer integration |
| Consciousness Security | USENIX | Immune system with HDC threat memory |
| Species Stewardship | AI & Ethics | Consciousness requirements for DNA-to-human pipeline |
| + 10 more papers | Various | See `papers/PAPERS_INDEX.md` for full list |

## External Validation Results

| Benchmark | Source | Result |
|-----------|--------|--------|
| Hendrycks ETHICS | External (Hendrycks et al., 2021) | 56.2% on 4 domains (2K samples); 94.5% figure RETRACTED as leakage-inflated (2026-07-15) |
| ARC-AGI | External (Chollet, 2019) | 4% strict; 100% 2-AFC RETRACTED as random-distractor artifact (2026-07-18, see `book/src/research/validation.md`) |
| Sleep-EDF | External (PhysioNet) | 70-80% 5-class on real clinical EEG |
| DMC Humanoid | External (DeepMind Control) | Competitive with SAC/TD3/D4PG |

## Reproducing Results

```bash
# External benchmarks
cargo run --example benchmark_moral_unified --release
cargo run --example benchmark_sleepstage --release
cargo run --example benchmark_arc_reasoning --release

# Psych-Bench full suite
cargo test -p symthaea-psych-bench --all-features

# Compile the book
cd papers/book && pdflatex symthaea_book.tex && pdflatex symthaea_book.tex
```
