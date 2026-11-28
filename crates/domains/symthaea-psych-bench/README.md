# symthaea-psych-bench

A Rust benchmark suite for evaluating cognitive architectures through a mixture
of synthetic mechanism probes, externally grounded tasks, and selected
human-anchored comparisons. The benchmark inventory is evolving; exact counts
must be generated from the registered implementations rather than copied into
documentation. The suite also includes the **Qualia Confidence Matrix**, a set
of computational probes inspired by consciousness theories and psychophysics.

> **Scope honesty:** most benchmarks operate on synthetic HDC stimuli and
> validate architectural mechanics, not human-normed task performance. In
> particular the `creativity/` domain is HDC-algebra sanity checking (the RAT
> adapter plants the association it recovers) — see
> `src/benchmarks/creativity/mod.rs` for details. Externally-grounded
> exceptions include Hendrycks ETHICS, Sleep-EDF, and ARC-AGI (see the
> examples that load real datasets).

## Quick Start

```bash
# Run the full qualia confidence matrix (< 30 seconds)
cargo run -p symthaea-psych-bench --example qualia_confidence_report

# Run the registered benchmark battery with composite summaries
cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --composites

# Run just the tests (deterministic, seed=42)
cargo test -p symthaea-psych-bench
```

## Qualia Confidence Matrix

7 computational probes inspired by GWT, IIT, HOT, FEP, Damasio, and
psychophysics. These probes test operationalized architectural behaviors; they
do not establish necessary or sufficient conditions for consciousness, and
successful results do not establish subjective experience:

| Benchmark | Theory | What it tests | Key metric |
|-----------|--------|---------------|------------|
| GWT Asphyxiation | GWT | Domain collapse order under workspace restriction | Spearman rho |
| Phase Transition | IIT | Sigmoidal vs. linear fidelity collapse | Sigmoid advantage |
| Perturbational Complexity | IIT/Clinical | Complex vs. simple response to perturbation (digital PCI) | PCI ratio |
| Somatic Interference | Damasio | Emergent cascade degradation under neuromod distress | Cascade ratio |
| Bistable Perception | Psychophysics | Heavy-tailed switching in ambiguous perception | CV of switch interval |
| Unconscious Priming | GWT | Sub/supra-threshold processing dissociation | Priming dissociation |
| Metacognitive Ignition | HOT+GWT | Shared-input HOT/GWT decision alignment under workspace competition | Tracking score |

**Evidence boundary**: 6 of 7 benchmarks primarily validate properties the
architecture was designed to exhibit. MetacognitiveIgnition does not receive
GWT's outcome or filler state, but HOT and GWT share the target activation
signal; its ROC/AUC is therefore an input-score discrimination diagnostic,
not an independent HOT-generated metacognitive readout. The suite provides no
independent evidence of human-like cognition or subjective experience.

### Published Results

| Version | Config | Composite | Level | Predictions |
|---------|--------|-----------|-------|-------------|
| v0.8.0 (2026-03-01) | seed=42, dim=512, trials=10 | 0.683 (legacy composite) | Historical label only | 7/7 |

Baseline snapshots are in `baselines/`. To compare against published results:

```bash
cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --compare baselines/v0.8.0.json
```

To reproduce with a different seed:

```bash
cargo run -p symthaea-psych-bench --example qualia_confidence_report -- --seed 123
```

**MetacognitiveIgnition illustration** (seed=42): accuracy=0.842, d'=3.63,
ROC AUC=0.998. This is a seed- and configuration-specific synthetic diagnostic,
not a human comparison or state-of-the-art result.

## Dependencies

This crate depends on:
- `symthaea-core` (HDC, CfC, GWT, HOT implementations)
- `symthaea-fep` (Free Energy Principle)
- `symthaea-ssm` (State Space Models)
- `symthaea-neuromodulators` (DA, NE, 5-HT, ACh, GABA, oxytocin)

All are workspace-local — no external data files or services required.

## Grounded creativity benchmarks (external data, optional)

The exception to "no external data" is the honestly-grounded creativity module
(`benchmarks/creativity/grounded.rs`). Unlike the HDC-algebra creativity
benchmarks (whose adapters plant the associations they recover — see the scope
note in `benchmarks/creativity/mod.rs`), the grounded RAT/DAT use an external
embedding space as their only knowledge source. To run them with real data:

1. **Embeddings**: download GloVe from <https://nlp.stanford.edu/projects/glove/>
   (e.g. `glove.6B.zip`, extract `glove.6B.300d.txt`) and load with
   `GloVeFileEmbedder::from_path("path/to/glove.6B.300d.txt")`. The bundled
   `DeterministicHashEmbedder` is a semantics-free mock for testing the
   harness — numbers from it are meaningless.
2. **RAT norms**: the published 144-item normed compound-remote-associate set
   is Bowden, E. M. & Jung-Beeman, M. (2003), "Normative data for 144
   compound remote associate problems", *Behavior Research Methods,
   Instruments, & Computers* 35, 634–639 (doi:10.3758/BF03195543). Convert
   the published item table to TSV — one item per line,
   `cue1<TAB>cue2<TAB>cue3<TAB>solution` — and load with
   `RatItemSet::from_tsv_file(path)`. The items are NOT bundled here to avoid
   transcription errors; the built-in `RatItemSet::demo()` is a ~12-item
   canonical subset for smoke tests only and must not be reported as RAT
   performance.
3. **DAT**: `score_dat(&words, &mut embedder)` implements Olson et al. (2021),
   *PNAS* 118(25) e2022340118 (mean pairwise cosine distance of the first 7
   valid words, × 100). Human norms (≈78 mean) were computed with
   GloVe-840B — scores from other embedding spaces are internally comparable
   but not directly comparable to the published norms.

These benchmarks characterize the embedding substrate (and, for DAT, whoever
produced the word list) — they do not yet measure Symthaea's own generation.

## Determinism

All benchmarks are deterministic given a seed. Default seed is 42.
Multi-seed robustness analysis available via:

```bash
cargo run -p symthaea-psych-bench --example multi_seed_robustness
```

## Paper

Results and methods are described in:
- `../../../papers/evaluation/psych-bench/psych_bench_paper.tex`
- `../../../papers/evaluation/psych-bench/psych_bench_supplement.tex`

The paper's reported benchmark totals and human-comparison claims are under
reconciliation with the current implementation inventory. Do not cite a
historical benchmark count or composite until the result is regenerated from a
single versioned registry and its measurement class is stated.

## License

MIT
