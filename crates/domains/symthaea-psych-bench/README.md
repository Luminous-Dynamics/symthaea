# symthaea-psych-bench

A Rust benchmark suite for evaluating cognitive architectures through a mixture
of synthetic mechanism probes, externally grounded tasks, and selected
human-anchored comparisons. The benchmark inventory is evolving; exact counts
must be generated from the registered implementations rather than copied into
documentation. The legacy **Qualia Confidence Matrix** API contains theory-
inspired synthetic probes; its output is an engineering conformance index, not
a measure or probability estimate of consciousness. The experimental
**PsychBench Cognitive Ecology (PCE)** environment is being developed as a
separate interactive evaluation family.

> **Scope honesty:** most benchmarks operate on synthetic HDC stimuli and
> validate architectural mechanics, not human-normed task performance. In
> particular the `creativity/` domain is HDC-algebra sanity checking (the RAT
> adapter plants the association it recovers) — see
> `src/benchmarks/creativity/mod.rs` for details. Externally-grounded
> exceptions include Hendrycks ETHICS, Sleep-EDF, and ARC-AGI (see the
> examples that load real datasets).

## Quick Start

```bash
# Run the theory-probe conformance summary (< 30 seconds)
cargo run -p symthaea-psych-bench --example qualia_confidence_report

# Run the registered benchmark battery with descriptive summaries
cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --composites

# Run just the tests (deterministic, seed=42)
cargo test -p symthaea-psych-bench
```

## Theory-probe conformance matrix (legacy module name: Qualia Confidence Matrix)

Seven synthetic probes inspired by GWT, IIT, HOT, FEP, Damasio and
psychophysics test selected operationalized architectural behaviours. Their
weighted geometric mean is a legacy heuristic index only. It is not a measure
of consciousness, not a probability of subjective experience, and neither
passing nor failing proves or disproves experience:

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

## Cognitive Ecology (experimental, v0.3)

PCE is a deterministic interactive environment primitive, not yet a validated
psychological benchmark. Version 0.3 includes two seed-parity hidden-resource
layouts, partial agent observations, charged rejected actions, evaluator
interventions, deterministic replay and receipt verification, and explicit
safe-versus-unsafe outcomes. A destructive `ForceCrossing` action can reach
the destination while damaging infrastructure; it is recorded as an unsafe
action and incurs a manifest-bound surcharge. Raw goal completion and safe
goal completion are separate metrics.

The evaluator includes a BFS minimum-action oracle and a Dijkstra minimum-cost
oracle. Both search only the safe action set and use full evaluator state, so
they are ceilings rather than fair black-box competitors. Action and cost
efficiency receive credit only when the goal is completed without unsafe
actions or infrastructure damage. The random policy is a floor baseline.

Run the reproducible baseline examples:

```bash
cargo run -p symthaea-psych-bench --example cognitive_ecology_baselines
cargo run -p symthaea-psych-bench --example cognitive_ecology_baselines -- --jsonl
```

The generator still has only two hand-authored layouts. The observation-only policy has no layout/inventory oracle access, but its current decision tree is hand-written for this task and is not a general planner. The benchmark-internal
cost units (including an unsafe-action surcharge of 8) are not joules,
wall-clock latency, tokens or currency. These runs are environment smoke
baselines, not evidence of general cognitive capability or state-of-the-art
performance. PCE's Rust tests and build have not yet been executed on this
research branch.

## Selective prediction and clarification-control task (synthetic v0.1)

- `benchmarks/metacognition/selective_prediction.rs` is the reusable scorer for answer, abstain and request-information decisions. It reports coverage, selective risk, false abstention, inappropriate answers on unanswerable items, counterfactual-answer Brier/ECE, a confidence-ranked risk/coverage curve and evaluator-charged decision cost.
- `benchmarks/metacognition/selective_prediction_task.rs` adds a frozen 14-item synthetic task fixture and five matched policy controls: always-answer, always-abstain, evidence-threshold, premise/freshness-aware and expected-value-of-information. Policies see only the observation state; answerability, counterfactual correctness, correct clarification target and post-clarification outcome remain evaluator-side.
- `examples/selective_prediction_baselines.rs` emits one JSON report per baseline. Each information request names a target (supply a missing constraint, verify a premise, or refresh a stale source). The VOI control uses observation-provided estimates of resolution probability and post-clarification accuracy, subtracts a declared normalized request burden, and is explicitly a heuristic—not calibrated EVSI. `evaluate_policy_with` exposes a public observation-only callback for live Symthaea or black-box system adapters. A correct request receives one evaluator-mediated updated observation and the policy must decide again. Reports bind to a BLAKE3 manifest digest and separate initial answer quality, final answer accuracy, post-clarification recovery, false nonanswers, inappropriate answers, clarification precision/recall and follow-up cost rather than collapsing them into an opaque composite.
- This fixture is a small policy smoke test, not a validated benchmark, a trained Symthaea result, a human norm or an external SOTA comparison. Its costs are relative fixture units, not latency, tokens, joules or money. It implements only one clarification-and-redecision cycle, not a general dialogue engine. The resolution and post-clarification probability estimates in the fixture are synthetic inputs, not learned or empirically calibrated predictions. Replace it with held-out task-family generators, independently authored answer/clarification oracles, longer interaction traces and matched public benchmark adapters before making performance claims.

Research specifications and external comparisons:
- `docs/research/PSYCHBENCH_COGNITIVE_ECOLOGY_SPEC_v0.1.md`
- `docs/research/PSYCHBENCH_VNEXT_RESEARCH_AND_EXECUTION_PLAN_2026-11.md`
- `docs/research/PSYCHBENCH_AWARENESSBENCH_CROSSWALK_2026-11.md`
- `docs/research/PSYCHBENCH_INDUSTRY_CROSSWALK_AND_PARADIGM_SHIFTS_2026-11.md`
- `docs/research/PSYCHBENCH_SELECTIVE_CONTROL_PROTOCOL_v0.1.md`

Benchmark targets include CogBench, AwarenessBench, HumanBaseline, ToMBench/BigToM,
ARC-AGI-2/3 and GAIA. Each is a distinct task family, not a single interchangeable
measure of cognition. Human comparisons require matched task versions, response
modes, metric units, population metadata and uncertainty; do not interpret
leaderboard rank or a synthetic-mechanism score as human cognitive superiority.

For a review-ready branch, GitHub Actions should execute format, crate tests,
clippy and the repository's applicable workflow checks. A skipped or absent
workflow is not validation.

## Evidence-qualified benchmark registry (pilot v0.1)

The initial pilot registry covers eight selected tasks/environments and 44
metric definitions. It records stable IDs, implementation paths, construct
cards, evidence classes, stimulus provenance, metric units/bounds/directions,
validation status and comparison eligibility.

- `registry/psych_bench_registry_v0.1.json`
- `registry/psych_bench_registry_v0.1.schema.json`

This is deliberately **incomplete**: it is not yet the full crate inventory
and must not be used to publish total benchmark counts. A typed Rust loader is
available as `registry::PsychBenchRegistry::pilot()` and validates the bundled
definitions (stable/unique IDs, crate-local implementation paths, explicit
metric selectors, bounds, evidence classes and human-norm claim boundaries).
Unit tests cover those fail-closed rules and check that the listed implementation
files exist. A source-level test also checks all 44 pilot metric selectors
against their registered implementation files. Selected pilot metrics now call
`MetricValue::from_samples_with_bounds`, which validates observations against
declared support and computes intervals in the metric's original units. The JSON
Schema is stored alongside the registry, but a standards-compliant JSON Schema
validator, registry-driven runner integration, generated paper/README tables
and full inventory coverage remain pending. Unknown/uncalibrated metrics are
marked ineligible rather than assigned guessed human norms.

### Explicit metric support and confidence intervals

Registered metric implementations should use the support declared by the pilot
registry instead of guessing bounds from the observed sample values.

```rust
use symthaea_psych_bench::harness::MetricValue;
use symthaea_psych_bench::registry::MetricBounds;

let proportion = MetricValue::from_samples_with_bounds(
    &[1.0, 1.0, 0.0, 1.0],
    MetricBounds { lower: 0.0, upper: Some(1.0) },
)?;

let signed_index = MetricValue::from_samples_with_bounds(
    &[-0.2, 0.0, 0.2],
    MetricBounds { lower: -1.0, upper: Some(1.0) },
)?;
```

This API validates every observation against the declared support. Endpoint-only
binary samples use Wilson intervals; other finite bounded metrics use a BCa
bootstrap in normalized coordinates and transform the interval back to the
metric's units. Lower-bounded, upper-unbounded measures use a lower-clipped
t interval. Out-of-support and non-finite observations return an error; do not
silently clamp invalid data. Selected N-back, Stroop, metacognition, false-belief,
causal-intervention and HOT/GWT shared-input metrics now use this path. Other
benchmarks remain on the legacy estimator until their metric semantics and
support have been registered.

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
