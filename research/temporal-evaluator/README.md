# Symthaea Temporal Evaluator

This is a separate, evaluator-side Rust package for paired retain/evict effects and exhaustive small-subset upper bounds. It is not a candidate runtime dependency.

The current source provides:

- seeded, reproducible retain-first / evict-first order assignment using evaluator-held randomness (SplitMix64 for reproducibility only; not cryptographic secrecy);

- paired intervention checks that require identical episode/task/future-stream/model/initial-state/candidate-feature digests and identical byte/compute/refresh/observation budgets;
- componentwise wrong-belief, abstention and missing-output exposure differences;
- group interventions to expose non-additive memory interactions;
- exact subset enumeration up to 16 candidate items;
- separate best-known subsets for each exposure component and a Pareto frontier, with no weighted scalar objective;
- hard checks for duplicate IDs, unordered selections, incorrect bytes, budget overruns, incomplete subset enumeration and exposure totals exceeding episode duration.

The exact subset result is an evaluator-only clairvoyant upper bound, not a deployable candidate policy. It must never be exposed to the candidate or used to train against confirmatory outcomes.

Package separation is an architectural boundary only. It does not provide OS-level confidentiality. Before confirmatory use, complete #9437: separate OS identities and protected oracle storage, candidate sandboxing, evaluator-owned signing keys, signed receipts, replay protection and adversarial filesystem/descriptor tests.

No Cargo build, format, Clippy or test execution is claimed until a Rust toolchain is available.

## Temporal decision-preserving compression

The `decision_sufficient_representation` module compares full-state and compressed-state information partitions using the existing decision-observability oracle. The `temporal_representation_preservation` module composes that certificate over a frozen sequence of future decision epochs.

The temporal profile:
- preserves the same evaluator hidden-world cohort and hidden-world digests across every epoch;
- freezes model/configuration identity and deterministic policy seed;
- permits candidate observation history and recurrent state to evolve;
- rejects skipped epochs, dropped/aliased worlds and identity drift;
- reports the contiguous representation-preservation prefix, maximum added regret and first failure;
- reports candidate-action quality separately from representation information preservation.

It does not implement the calibrated selected-action horizon (#11646/#11650), episode-level calibration (#11655/#11659), a candidate compressor, or actuation authority. See `docs/research/TEMPORAL_DECISION_PRESERVATION_BENCHMARK_V0_1.md` and issue #13255.

Rust 1.96 formatting, full-crate unit tests and strict Clippy are covered by `.github/workflows/temporal-evaluator-qualification.yml`. Treat each result as exact-head evidence; a skipped job is not a pass.

The qualification workflow is supplied by the parent decision-preserving representation branch so stacked pull requests execute the same Rust 1.96 format/test/Clippy contract from their base revision.

The single-epoch compression evaluator is formatted and maintained by the parent #13239 branch; this temporal module delegates to it rather than forking its decision-regret logic.
