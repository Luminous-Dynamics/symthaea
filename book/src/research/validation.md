# Validation Studies

## Internal Validation: Psych-Bench

141 benchmarks across 27 cognitive domains test executive function, attention, reasoning, social cognition, motor learning, metacognition, and more. Grand mean z-score: +1.190 (all domains above human mean).

**Honest caveat**: All 141 benchmarks were designed by the same team that built the architecture. The z-scores should be read as "performance on our benchmarks relative to our baselines," not as independently validated cognitive capability.

## External Validation

Four external benchmarks partially address the circular validation concern:

| Benchmark | Result | What It Validates |
|-----------|--------|-------------------|
| **Hendrycks ETHICS** (4 domains, 2K samples) | 94.5% | Learned HDC moral classification |
| **Sleep-EDF** (PhysioNet clinical EEG) | 70-80% 5-class | LTC integration on real EEG |
| **ARC-AGI** (Chollet) | 100% 2-AFC, 4% strict | HDC algebraic rule transfer |
| **DMC Humanoid** (vs SAC/TD3/D4PG) | Competitive | FEP perturbation recovery |

Major suites remain unevaluated: MMLU, GSM8K, HellaSwag, HumanEval.

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
