# Validation Studies

## Internal Validation: Psych-Bench

141 benchmarks across 27 cognitive domains test executive function, attention, reasoning, social cognition, motor learning, metacognition, and more. Grand mean z-score: +1.190 (all domains above human mean).

**Honest caveat**: All 141 benchmarks were designed by the same team that built the architecture. The z-scores should be read as "performance on our benchmarks relative to our baselines," not as independently validated cognitive capability.

## External Validation

Four external benchmarks partially address the circular validation concern:

| Benchmark | Result | What It Validates |
|-----------|--------|-------------------|
| **Hendrycks ETHICS** (4 domains, 2K samples) | 56.2% (94.5% figure **RETRACTED** as leakage-inflated, 2026-07-15) | Learned HDC moral classification |
| **Sleep-EDF** (PhysioNet clinical EEG) | 70-80% 5-class | LTC integration on real EEG |
| **ARC-AGI** (Chollet) | 4% strict; 2-AFC **RETRACTED** (see note) | HDC algebraic rule transfer |
| **DMC Humanoid** (vs SAC/TD3/D4PG) | Competitive | FEP perturbation recovery |

Major suites remain unevaluated: MMLU, GSM8K, HellaSwag, HumanEval.

**ARC-AGI 2-AFC retraction (2026-07-18)**: the previously reported "100% 2-AFC" was a
discriminability artifact, not a rule-transfer score — it compared the predicted output's
similarity to the actual answer against its similarity to a **literally random** distractor
vector, and any structured grid encoding beats random noise regardless of whether the rule
was learned correctly. Re-run on the real 400-task ARC-AGI training set with fair (equally
structured) distractors: random-distractor 99.0%, but identity-distractor (test input
unchanged) 13.8% — *below* chance — and reflect_x/reflect_y distractors 64.9%/67.8%, well
short of "100%". The 4% strict (pixel-perfect) figure is unaffected and stands. See
`crates/domains/symthaea-psych-bench/src/benchmarks/reasoning/arc_dataset.rs` and
`examples/arc_2afc_reaudit.rs` for the full methodology — same inflation class as the
retracted Hendrycks ETHICS figure below.

## Butlin Consciousness Indicators

12 of 14 indicators from Butlin et al. (2023) are satisfied: recurrent processing, global workspace, higher-order representation, attention schema, temporal integration, binding, embodiment, agency, metacognition, affect, learning, and social cognition.

## Running Validation

```bash
# Psych-Bench full suite
cargo test -p symthaea-psych-bench --all-features

# External benchmarks
cargo run --example benchmark_moral_unified --release
cargo run --example benchmark_sleepstage --release
cargo run --example benchmark_arc_reasoning --release
```
