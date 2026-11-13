# Vocal causal-control evaluator

A dependency-light Rust evaluator for one-factor vocal interventions. It measures whether a requested change was realized in the intended phrase or region while preserving protected features and regions.

It consumes feature measurements from independent audio-analysis tools and listener studies. It does not inspect WAV bytes, infer F0/phonemes, or judge naturalness. Its job is to score supplied measurements against a provenance-bound intervention contract. It accepts both a single matched pair and a multi-seed suite with a Student-t 95% confidence interval.

## Paradigm

Treat a singer as a controllable dynamical system, not just a sample generator:

    requested control change -> rendered response -> collateral-drift audit

Examples:
- Increase vibrato depth in phrase 2 while retaining its rate and pitch center.
- Increase breathiness in one line while preserving phone identity and note timing.
- Move one breath without perturbing the next phrase.
- Correct one phone while retaining the rest of the score, voice identity and unaffected phrases.

## Input contract

Each JSON case contains:
- exactly one target feature dimension and a non-zero requested delta;
- target and protected region IDs;
- required measured dimensions and explicit per-dimension tolerances;
- baseline and variant feature vectors for every declared region;
- baseline/variant receipts containing source revision, backend, model-artifact digest, voice ID, invariant-task digest, exact performance digest, exact WAV digest, seed and sample rate.

Both render receipts must use the same code, backend/model artifacts, voice, invariant task, seed and sample rate. Their performance and output digests must differ. Missing measurements, malformed hashes, invalid regions, non-finite values and mismatched renderer provenance fail closed.

## Scoring

- Target response ratio: observed mean target delta / requested delta.
- Direction consistency: fraction of target regions that move in the requested direction.
- Collateral drift: absolute unintended delta divided by the declared tolerance, across untargeted features in target regions and every feature in protected regions.
- Locality: when required, protected regions must be explicitly supplied; unintended changes there are reported separately.

The report includes independent values and exact receipts. There is no single quality composite. A case qualifies only when target response, direction consistency and collateral constraints all pass.

Before choosing intervention tolerances, run a no-op calibration with at least five unique seeds, the same serialized performance, and identical code, renderer, model artifact, voice and task. For every feature and region, it computes the 95th-percentile pairwise absolute drift across repeated renders; the worst region's value is combined with an explicit measurement-resolution floor to recommend a tolerance. This calibration output is empirical and must be frozen before any intervention result is observed.

For multi-seed suites, every replicate must keep the same code, renderer, model artifact, voice, invariant task and sample rate, while using a unique seed. The suite report includes mean target-response ratio, sample standard deviation, a two-sided 95% Student-t confidence interval, direction consistency, worst collateral drift, and the number of individually qualified replicates. A suite qualifies only when every pair qualifies and the aggregate interval is wholly inside the predeclared response bounds. Three seeds are the minimum contract; five or more are recommended for pilot work. Confirmatory sample sizes should be based on a separate power analysis.

## Run

~~~bash
cargo run --manifest-path tools/vocal-control-eval/Cargo.toml -- control-case.json
cargo test --manifest-path tools/vocal-control-eval/Cargo.toml

# A multi-seed suite uses a top-level replicates array and requires at least 3 unique seeds.
cargo run --manifest-path tools/vocal-control-eval/Cargo.toml -- control-suite.json
~~~

Exit code 0 means the supplied case passed its machine-defined control contract. Exit code 2 means it was evaluated but failed. Exit code 1 means the file or contract was invalid. None of these means the singing audio is perceptually good; human naturalness, intelligibility and preference remain separate assessments.

## Important limits

- Feature extractors and annotators must be independent of the renderer being assessed.
- The control target and protected-region boundaries must be frozen before generation.
- Tolerances must be calibrated on repeatability/no-op runs, then frozen before confirmatory tests.
- Report missing estimates as missing; do not replace them with zero.
- Use multiple seeds and held-out songs/singers for generalization, not only one deterministic fixture.
- Model-artifact digests are required; backend names and paths are not sufficient.
- Human preference and perceived expression need a blinded study; the CLI cannot replace it.
