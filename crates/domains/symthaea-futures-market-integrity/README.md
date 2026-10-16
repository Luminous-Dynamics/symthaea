# Symthaea Futures: Prediction-Market Integrity Lab

**Status:** early research harness; synthetic-only; not a trading system.

This is a bounded audit kernel for eight structural questions:

1. Does a large count of forecast accounts conceal repeated reports from one declared source group?
2. How much gross turnover is explicitly same-beneficial-owner, assuming ownership labels are already qualified?
3. Does an event contract include required fields for a structured resolution policy?
4. How much of the terminal outcome-aligned price move occurred before a supplied public-information timestamp, under declared scope conditions?
5. How sensitive is that timing statistic to plausible timestamp alternatives, with exclusions explicit rather than zero-filled?
6. How sensitive is a source-capped forecast aggregate to alternative declared source partitions?
7. Can the evaluator select only a forecast available by a frozen horizon cutoff and preserve missing/stale exclusions?
8. Can paired Brier differences be summarized with a deterministic declared-cluster bootstrap rather than treating repeated snapshots as independent?

It deliberately does not connect to a venue, execute orders, infer criminal intent, prove source independence, authenticate identities, resolve events, or infer meaningful net risk transfer from gross turnover.

## Run

From the Symthaea workspace:

~~~sh
cargo fmt --manifest-path crates/domains/symthaea-futures-market-integrity/Cargo.toml --check
cargo test --manifest-path crates/domains/symthaea-futures-market-integrity/Cargo.toml
cargo clippy --manifest-path crates/domains/symthaea-futures-market-integrity/Cargo.toml --all-targets -- -D warnings
cargo run --manifest-path crates/domains/symthaea-futures-market-integrity/Cargo.toml --example replay_benchmark_v1
~~~

These are reproduction commands, not evidence of passing checks. The example emits synthetic Brier/log-loss summaries and a future-only oracle exclusion control; it is not a real forecast-quality result. Only exact-head completed runs count as qualification.

The example emits a CSV summary to stdout and reliability-bin, paired-common-cohort, negative-control, and manifest records to stderr. In CI, the manifest binds the actual expected PR head SHA supplied by the workflow. A local run without that environment variable explicitly reports `unbound-local-run` rather than inventing provenance.

## New comparative replay

The `replay` module selects the latest observation at or before a caller-frozen event/horizon cutoff, rejects duplicate observation keys, and records missing/stale combinations as explicit exclusions. The new `calibration_report` builds equal-width reliability bins for one method and one horizon, preserves empty bins, and publishes bin counts, mean forecast probabilities, observed frequencies, and a secondary bin-dependent weighted absolute calibration error. It rejects mixed horizons/methods to avoid a silently pooled reliability diagram.

The paired cluster bootstrap computes candidate-minus-baseline Brier differences, averages rows within each declared event/parent-event cluster, then resamples clusters with replacement.

Call the bootstrap separately for horizon-specific inference. Mixing horizons produces a cluster-weighted average over the supplied rows, which is only meaningful if that combined estimand was preregistered. Negative Brier differences favor the candidate. The fraction of bootstrap means below zero is descriptive, **not a p-value**. The percentile interval depends on defensible cluster labels and a valid sampling design; this module cannot infer statistical independence.

See [PREDICTION_MARKET_EPISTEMIC_INFRASTRUCTURE_V1.md](../../../docs/research/finance/PREDICTION_MARKET_EPISTEMIC_INFRASTRUCTURE_V1.md) for the broader research agenda and [PREDICTION_MARKET_COMPARATIVE_REPLAY_BENCHMARK_V1.md](../../../docs/research/finance/PREDICTION_MARKET_COMPARATIVE_REPLAY_BENCHMARK_V1.md) for the versioned benchmark protocol.

## Design boundaries

- Source-capped pooling is one sensitivity baseline, not a universally optimal estimator.
- Source groups are supplied labels, not authenticated independence. Unknown/correlated sources must remain visible.
- Same-owner turnover is a structural alert, not proof of intent or a complete detector of round trips across multiple owners.
- Resolution-policy qualification checks required fields and simple bounds; it does not authenticate a source, establish adjudicator authority, or interpret the law.
- Information-timing diagnostics require qualified public-event timestamps and category-matched controls. They are not an insider-trading probability or proof of misconduct.
- Timestamp sensitivity describes the supplied finite candidate sample; it is not a continuous uncertainty bound without a justified timestamp grid.
- Prequential cutoffs and outcomes are caller-supplied data; their authenticity is outside this crate's trust boundary.
- The bootstrap resamples declared clusters with a fixed seed for replay. It is non-cryptographic, and incorrectly defined/non-independent clusters can still yield overconfident intervals.
- All outputs are diagnostics, never a trading recommendation or autonomous market action.
