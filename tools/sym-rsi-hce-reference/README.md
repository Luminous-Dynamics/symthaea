# HCE and asynchronous worker-trace reference

Standalone, offline Rust reference tooling for issue #10980.

It has two narrow functions:

1. Validate a synthetic Hidden Consistent Evaluation (HCE) epoch contract with distinct train, search, post-search selection, confirmation and fresh-family transfer roles.
2. Analyze already-recorded synchronous/asynchronous worker traces for valid candidate throughput, busy/idle capacity, utilization, p50/p95 attempt latency, straggler ratio, failures and declared resource cost.

It does **not** run workers, schedule tasks, evaluate models, access hidden labels, authenticate a custodian, issue evidence receipts or authorize promotion.

## Run

```bash
cargo fmt --manifest-path tools/sym-rsi-hce-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-hce-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-hce-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-hce-reference/Cargo.toml -- --fixtures
```

## HCE contract

The five role-specific partitions are:

- `train`: features and labels visible to the agent;
- `search`: features available, labels hidden; an external search evaluator returns only a score;
- `selection`: hidden from the search process and used only after search stops;
- `confirmation`: accessible only to an independent evaluator after the candidate is frozen;
- `transfer`: an independent fresh task family evaluated after the candidate is frozen.

The checker rejects role/visibility contradictions, duplicate roles, duplicate split IDs or membership commitments, incorrect family assignments, an unseparated selection/confirmation policy, or a missing/invalid independent overlap-audit declaration.

**Important:** membership hashes and a boolean overlap-audit declaration do not prove the underlying sets are disjoint. The named independent custodian must inspect the actual membership manifests, data mounts, labels and access logs. This crate checks the declaration and structural policy only.

## Worker trace contract

The trace analyzer requires unique attempt IDs, in-range worker identities, positive bounded intervals, no overlapping runs on the same worker, finite resource-cost values and a declared wall-clock budget.

It counts unique valid candidate IDs separately from valid attempts, retaining repeated attempts and failed/invalid/cancelled/timed-out work in the denominator. Worker utilization is measured as total busy milliseconds divided by worker-count × wall-clock-budget. p50/p95 are nearest-rank percentiles over all recorded attempts.

These are descriptive metrics over a supplied trace. A synthetic trace result is not a throughput benchmark and does not prove that an asynchronous scheduler ran.

## Claim boundary

- Both checked-in fixtures are synthetic.
- The HCE fixture remains `SYNTHETIC_HCE_FIXTURE_NOT_PREREGISTERED`.
- No real split membership, sealed labels, evaluator service, model, worker, GPU, or campaign was accessed.
- No Rust compilation, rustfmt, Clippy, test execution or CI pass is claimed until the commands run on an available toolchain.
