# NewtonBench Native Scorecard Importer v0.1

A standalone Rust CLI for paired importing of raw NewtonBench trial artifacts, preserving the benchmark's native metrics and failure states.

This is a data-ingestion and pairing layer—not a NewtonBench execution environment, an agent, a symbolic-equivalence verifier, or a SOTA result.

## Run

```sh
cargo test --locked --manifest-path research/newtonbench-native-scorecard/Cargo.toml
cargo run --locked --manifest-path research/newtonbench-native-scorecard/Cargo.toml -- \
  MANIFEST.json VANILLA_TRIALS_DIR CODE_ASSISTED_TRIALS_DIR
```

The CLI recursively reads raw `trial*.json` files from each supplied trial directory. The manifest freezes the upstream repository revision, model and judge identities, runner-configuration digest, backend labels, and every expected paired task cell. The expected manifest—not the set of files found—defines the denominator.

## Native metric contract

- Preserve `evaluation.rmsle` as **RMSLE**. It is never relabeled or converted into NRMSE.
- Preserve `evaluation.exact_accuracy` as the upstream per-trial 0/1 result from an LLM symbolic-equivalence judge. It is not a formal proof.
- Preserve `num_experiments`, `rounds`, and `total_tokens` as separate costs.
- Report how many completed trials supplied each cost field separately from sums of available values; a missing field is never silently presented as a complete total.
- Retain failures and missing counterpart trials; paired deltas use only pairs with both trials completed and the corresponding metric available.
- Report paired effects by module, noise, equation difficulty, system type and law version as well as the pooled scorecard, so pooled gains cannot hide a regression on a difficult stratum.
- Include approximate paired Student-t 95% intervals and standard errors for mean deltas at pooled and stratum level; intervals are null when fewer than two complete pairs exist. They are descriptive intervals, not causal proof or a substitute for a preregistered hierarchical analysis.
- The importer records source-file digests, native metric availability, and import problems. A malformed artifact does not silently disappear from the report.
- Python's default JSON serialization may emit non-standard NaN/Infinity values. The importer normalizes standalone non-finite tokens outside strings to null, records the normalization, and does not convert an unavailable RMSLE into zero.
On upstream `failed` or `invalid` rows, evaluation metrics are withheld even when the failure writer emitted placeholder `exact_accuracy = 0.0`; the terminal state and raw-file digest remain visible. Recorded cost fields remain visible with availability counts, and paired cost deltas require values in both completed arms.

NewtonBench's upstream RMSLE implementation masks NaN points and uses absolute values before applying log1p. The importer preserves its result and records this semantic caveat rather than recomputing or overstating it. Upstream output also labels the backend; the importer does not authenticate that a tool actually executed. False-confidence and premature-commitment outcomes are not inferred because the native trial schema does not directly provide them.

## Synthetic CLI fixture

Run:

```sh
cargo run --locked --manifest-path research/newtonbench-native-scorecard/Cargo.toml -- \
  research/newtonbench-native-scorecard/fixtures/manifest.json \
  research/newtonbench-native-scorecard/fixtures/vanilla \
  research/newtonbench-native-scorecard/fixtures/code_assisted
```

The fixture intentionally includes one paired completed trial, one failed vanilla trial with a Python-style NaN RMSLE, and a missing code-assisted counterpart. These values exist only to test importer behavior; they are not empirical benchmark results.

For actual imports, set `upstream_revision` to a pinned 40-character lowercase NewtonBench Git commit SHA (not `main`); the validator rejects other revision labels. The literal `fixture-only` is reserved for explicitly synthetic fixtures. Hash the complete runner/model/judge/tool configuration in `runner_configuration_sha256`. The report can establish native artifact pairing and arithmetic, not independent scientific validity or authenticated execution.
