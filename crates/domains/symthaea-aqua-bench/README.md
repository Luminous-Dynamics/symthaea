# symthaea-aqua-bench

Deterministic, dependency-free accounting primitives for Luminous Aqua benchmark runs.

## Scope

This crate calculates complete-service fresh-water totals, purchased-energy totals, matched-control water savings, and water-balance closure against an explicit expanded measurement uncertainty.

It deliberately does **not** assess microbiological quality, recirculation hygiene, shower comfort, spray performance, safety, regulatory compliance, or physical performance. It is not a hydraulic or thermal solver. Those claims remain with the relevant independent methods and existing Luminous subsystem owners.

## Evidence semantics

- Every quantity needed for a whole-service result is an `Option<f64>`; missing is not zero.
- Explicit `Some(0.0)` records a measured zero for an operation that does not apply.
- Startup, priming, active fresh water, post-use cleaning, purge, periodic-cleaning allocation, and retry/recovery water are all counted.
- In-session recirculated delivery is recorded separately and is not counted as fresh water or external water input.
- Electricity, fuel input, and useful recovered heat are reported separately. Recovered heat is not subtracted from purchased energy.
- Internal recirculation is excluded from whole-boundary water source/output balance because it is an internal transfer.
- Water-balance closure is accepted only when absolute residual is within the supplied expanded uncertainty. This arithmetic does not create instrument calibration or uncertainty authority.

## Run

```sh
cargo test -p symthaea-aqua-bench
```

The crate is automatically a workspace member through the repository's `crates/domains/*` glob. Run the tests on the exact branch head before promoting any code claim.
