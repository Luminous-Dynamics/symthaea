# symthaea-energy-bench

Independent semantic validation primitives for the Energy-Conversion Benchmark Contract.

## Run the tests

From the repository root:

```bash
cargo test -p symthaea-energy-bench
cargo fmt --all -- --check
```

Run the Rust unit tests and formatting check locally, or verify a successful check on the exact commit under review. Tests authored but not executed are unverified; queued, pending, skipped, or cancelled jobs are not passes.

## Deterministic experiment selection (Phase A)

The `experiment_selection` module implements a transparent queue baseline before active learning. It ranks proposals lexicographically by registered decision relevance, expected fractional uncertainty reduction for a named observable, that reduction per unit of a common preregistered resource, reproducibility, and stable experiment ID. The queue rejects proposals with missing competing hypotheses, absent pre-observation registration, unapproved safety review, missing operating envelope or abort criteria, incomplete provenance planning, non-finite estimates, duplicate IDs, or inconsistent cost units. Rejections include explicit reasons.

Example API: `experiment_selection::rank_experiments(&policy, &proposals)`. The queue policy itself must be registered before ranking. Resource units are compared exactly; conversions must happen before selection. This is **not** Bayesian optimization, expected information gain, a calibrated uncertainty model, or a safety approval system. Ranking is a screening aid and cannot replace engineering review.

## Oracle-backed selector benchmark

The `experiment_benchmark` module evaluates Phase A against random, cheapest-first, one-dimensional space-filling, and an exhaustive oracle upper bound on the same admissible proposal set and resource budget. Its deterministic generator makes six-proposal known-answer scenarios and deliberately includes cases where declared uncertainty-reduction proxies are misleading.

The report includes selected IDs, rejected proposals and reasons, budget use, achieved synthetic resolution utility, oracle utility, decision regret, resolution rate, median/maximum regret, resource cost among resolved scenarios, and an exact oracle lower bound on cost to resolution. Heuristic strategies stop selecting once their registered synthetic resolution threshold is met. The oracle enumerates subsets and is capped at 16 eligible experiments per scenario. For generated scenarios, utility is a toy independent-information model; these are algorithmic stress tests only, not a physical energy-conversion model or evidence of engineering superiority.

Run with `cargo test -p symthaea-energy-bench experiment_benchmark::tests::reference_suite_runs_one_thousand_known_answer_scenarios`. This test is authored but must be run on a Rust-enabled environment; source presence is not a passing result.
The harness can emit a replayable JSON receipt with `experiment_benchmark::benchmark_report_json(&report)`. It embeds complete scenario inputs and records SHA-256 digests for the registered policy, each scenario, suite configuration, and computed result set. `parse_and_verify_benchmark_report_json` checks digests and replays each strategy to reject inconsistent/tampered outputs. Store the exact Symthaea source commit alongside the receipt; these input digests do not identify which source revision executed the benchmark.

## Candidate-decision negative controls

The `candidate_decision_benchmark` module separates synthetic oracle truth from the observed candidate record, and computes false-superiority and false-negative rates over a 720-case grid of observed bias, uncertainty scales, correlation declarations, and registered worst-case differential-bias bounds. The comparator subtracts that bound linearly after expanded measurement uncertainty. Its required basis text documents the assumption but does not authenticate it. This deliberately tests a failure mode that accounting closure alone cannot catch: biased observations can close the energy ledger yet still imply the wrong conclusion.

Run the standalone report generator:

```bash
cargo run -p symthaea-energy-bench --example candidate_decision_benchmark > candidate-decision-benchmark.json
```

The receipt embeds its scenario inputs, records fixture/suite/result hashes, and can be replay-checked with `parse_and_verify_candidate_decision_report_json`. Its records are software-generated negative controls, not measured or simulated physical-engine results. A nonzero false-superiority rate here is an identified limitation to address, not an efficiency finding.

## Benchmark acceptance targets

The research acceptance ladder is specified in [ENERGY_CONVERSION_BENCHMARK_TARGETS.md](../../../docs/research/ENERGY_CONVERSION_BENCHMARK_TARGETS.md). It separates code/accounting conformance, oracle-backed selector quality, calibrated multifidelity evaluation, and independent physical validation. Proposed percentage improvements are unproven go/no-go targets, not current performance claims. Synthetic cases never establish physical superiority.

## What it checks

- schema version and key identifiers;
- required mission phases and contiguous, ordered intervals from time zero through declared mission duration;
- finite/non-negative external input, useful output, auxiliary consumption, rejected/lost energy, and uncertainty quantities;
- per-interval first-law residuals and aggregate closure using the preregistered tolerance;
- charge-sustaining storage before exposing net cycle efficiency;
- cached totals/efficiency against independently recomputed values;
- provenance artifact IDs, hash syntax, references, and RFC 3339 timestamps;
- preregistration timestamp ordering and synthetic-fixture decision guards;
- SHA-256 for exact artifact bytes when the caller supplies those bytes;
- candidate-versus-baseline comparison under identical registered missions, interval schedules, energy boundaries, solver method/version/revision/fidelity, calibration datasets, and policies;
- explicit v1 metric declaration: `net_cycle_efficiency`. Unsupported metrics such as heat-pump coefficient of performance are rejected rather than treated as efficiency;
- first-order propagation of efficiency uncertainty only when primitive channel/interval and candidate/baseline uncertainties are explicitly declared independent;
- a conservative comparison lower bound that must exceed a preregistered absolute efficiency margin before returning `LowerBoundClearsRegisteredMargin`.

## Explicit limitations

This crate is not a JSON Schema engine. Apply the published JSON Schema (Draft 2020-12) to reject unknown fields and enforce full structural constraints before semantic validation. The typed Rust parser intentionally ignores extra fields, so semantic validation alone does not replace schema validation.

The v1 comparator only supports charge-sustaining net cycle efficiency. It is not a universal score for all energy technologies: heat-pump coefficient of performance, specific fuel consumption, thrust-specific metrics, cost, and lifecycle outcomes require separate metric definitions and implementations. Unsupported metrics are rejected.

The crate does not fetch URIs, read arbitrary filesystem paths, check external signatures, or prove that a timestamp was not falsified. A real run's benchmark policy and manifest digest must be immutably anchored before candidate search (for example in a reviewed pre-search commit or signed append-only registry); a self-reported boolean or timestamp is not proof of preregistration. Uncertainty propagation currently uses a first-order delta-method approximation for the ratio of aggregate useful output to aggregate external input, with root-sum-of-squares and an explicit mutual-independence assumption across channels, intervals, and candidate/baseline measurements. Correlated or unknown cases are `Indeterminate`; covariance-matrix and Monte Carlo propagation are not implemented. The decision enum reports only whether the calculated lower bound clears the preregistered margin after subtracting the registered worst-case differential-bias bound. The required basis text is auditable documentation, not authenticated evidence. The verifier does not establish that the bound is physically defensible, authenticate pre-search anchoring, or prove physical superiority. The output is model-level evidence only: the crate does not validate the physical solver or dataset representativeness.

The synthetic fixture under docs/research/examples is for tests only. It is marked `not_for_decision: true` and must never be reported as measured performance.
Run the full example benchmark (1,000 scenarios, Phase A, cheapest-first, space-filling, oracle upper bound, and 30 seeded random baselines) from the repository root:

```bash
cargo run -p symthaea-energy-bench --example selector_benchmark > energy-selector-benchmark.json
```

This writes the complete input-and-results receipt to JSON. Record `git rev-parse HEAD` alongside the artifact; the input digests identify the policy and scenarios, not the executable revision. The example has been authored but is not claimed to have compiled or run until this exact revision is tested.
