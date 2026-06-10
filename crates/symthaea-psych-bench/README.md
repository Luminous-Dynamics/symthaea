# symthaea-psych-bench

Psychological benchmark suite for evaluating consciousness-relevant properties
of cognitive architectures. 84 benchmarks across 17 domains, including the
**Qualia Confidence Matrix** — 7 benchmarks testing architectural prerequisites
for consciousness.

## Quick Start

```bash
# Run the full qualia confidence matrix (< 30 seconds)
cargo run -p symthaea-psych-bench --example qualia_confidence_report

# Run the full 84-benchmark battery with composite scores
cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --composites

# Run just the tests (deterministic, seed=42)
cargo test -p symthaea-psych-bench
```

## Qualia Confidence Matrix

7 benchmarks testing necessary conditions for consciousness, derived from
GWT, IIT, HOT, and FEP:

| Benchmark | Theory | What it tests | Key metric |
|-----------|--------|---------------|------------|
| GWT Asphyxiation | GWT | Domain collapse order under workspace restriction | Spearman rho |
| Phase Transition | IIT | Sigmoidal vs. linear fidelity collapse | Sigmoid advantage |
| Perturbational Complexity | IIT/Clinical | Complex vs. simple response to perturbation (digital PCI) | PCI ratio |
| Somatic Interference | Damasio | Emergent cascade degradation under neuromod distress | Cascade ratio |
| Bistable Perception | Psychophysics | Heavy-tailed switching in ambiguous perception | CV of switch interval |
| Unconscious Priming | GWT | Sub/supra-threshold processing dissociation | Priming dissociation |
| Metacognitive Ignition | HOT+GWT | HOT spontaneously predicts GWT ignition | Tracking score |

**Epistemic honesty**: 6 of 7 benchmarks validate properties the architecture
was designed to exhibit. MetacognitiveIgnition tests emergent cross-module
alignment that was NOT explicitly programmed — it is the strongest benchmark.

### Published Results

| Version | Config | Composite | Level | Predictions |
|---------|--------|-----------|-------|-------------|
| v0.8.0 (2026-03-01) | seed=42, dim=512, trials=10 | 0.683 | MODERATE | 7/7 |

Baseline snapshots are in `baselines/`. To compare against published results:

```bash
cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --compare baselines/v0.8.0.json
```

To reproduce with a different seed:

```bash
cargo run -p symthaea-psych-bench --example qualia_confidence_report -- --seed 123
```

**MetacognitiveIgnition highlights** (seed=42): accuracy=0.842, d'=3.63, ROC AUC=0.998

## Dependencies

This crate depends on:
- `symthaea-core` (HDC, CfC, GWT, HOT implementations)
- `symthaea-fep` (Free Energy Principle)
- `symthaea-ssm` (State Space Models)
- `symthaea-neuromodulators` (DA, NE, 5-HT, ACh, GABA, oxytocin)

All are workspace-local — no external data files or services required.

## Determinism

All benchmarks are deterministic given a seed. Default seed is 42.
Multi-seed robustness analysis available via:

```bash
cargo run -p symthaea-psych-bench --example multi_seed_robustness
```

## Paper

Results are described in:
- `papers/latex/psych_bench_paper.tex` — Full 84-benchmark survey (target: BRM)
- `papers/latex/metacognitive_ignition_paper.tex` — MetacognitiveIgnition finding (target: CogSci/NeurIPS)

## License

MIT
