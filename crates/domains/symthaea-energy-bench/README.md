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
- SHA-256 for exact artifact bytes when the caller supplies those bytes.

## Explicit limitations

This crate is not a JSON Schema engine. Apply the published JSON Schema (Draft 2020-12) to reject unknown fields and enforce full structural constraints before semantic validation. The typed Rust parser intentionally ignores extra fields, so semantic validation alone does not replace schema validation.

The crate does not fetch URIs, read arbitrary filesystem paths, check external signatures, prove that a timestamp was not falsified, or propagate uncertainty distributions. A real run's benchmark policy and manifest digest must be immutably anchored before candidate search (for example in a reviewed pre-search commit or signed append-only registry); a self-reported boolean or timestamp is not proof of preregistration. The crate does not validate the physical solver, prove dataset representativeness, or establish superiority over a baseline.

The synthetic fixture under docs/research/examples is for tests only. It is marked `not_for_decision: true` and must never be reported as measured performance.
