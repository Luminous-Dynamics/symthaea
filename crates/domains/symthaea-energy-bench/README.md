# symthaea-energy-bench

Independent semantic validation primitives for the Energy-Conversion Benchmark Contract.

## Run the tests

From the repository root:

```bash
cargo test -p symthaea-energy-bench
cargo fmt --all -- --check
```

Current validation is performed by the Rust unit tests and exact-head CI. Do not treat a queued job as a pass.

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
- candidate-versus-baseline comparison under identical registered missions, boundaries, and policies;
- first-order propagation of efficiency uncertainty only when primitive channel/interval and candidate/baseline uncertainties are explicitly declared independent;
- a conservative comparison lower bound that must exceed a preregistered absolute efficiency margin before returning `LowerBoundClearsRegisteredMargin`.

## Explicit limitations

This crate is not a JSON Schema engine. Apply the published JSON Schema (Draft 2020-12) to reject unknown fields and enforce full structural constraints before semantic validation. The typed Rust parser intentionally ignores extra fields, so semantic validation alone does not replace schema validation.

The crate does not fetch URIs, read arbitrary filesystem paths, check external signatures, or prove that a timestamp was not falsified. A real run's benchmark policy and manifest digest must be immutably anchored before candidate search (for example in a reviewed pre-search commit or signed append-only registry); a self-reported boolean or timestamp is not proof of preregistration. Uncertainty propagation currently uses a first-order delta-method approximation for the ratio of aggregate useful output to aggregate external input, with root-sum-of-squares and an explicit mutual-independence assumption across channels, intervals, and candidate/baseline measurements. Correlated or unknown cases are `Indeterminate`; covariance-matrix and Monte Carlo propagation are not implemented. The decision enum reports only whether the calculated lower bound clears the preregistered margin; it does not authenticate pre-search anchoring or prove physical superiority. The output is model-level evidence only: the crate does not validate the physical solver or dataset representativeness.

The synthetic fixture under docs/research/examples is for tests only. It is marked `not_for_decision: true` and must never be reported as measured performance.
