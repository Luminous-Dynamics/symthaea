# POAR-RX Lab: Deterministic Recovery Microbenchmark

This standalone Rust crate is the first executable fault-injection slice for issue #9428. It depends only on the existing zero-third-party-dependency POD-Meta lab and is intentionally outside the root Cargo workspace.

## Run

~~~sh
cargo test --manifest-path research/poar-rx-lab/Cargo.toml
cargo run --manifest-path research/poar-rx-lab/Cargo.toml --bin poar-rx-lab
cargo fmt --manifest-path research/poar-rx-lab/Cargo.toml -- --check
cargo clippy --manifest-path research/poar-rx-lab/Cargo.toml --all-targets -- -D warnings
~~~

The default runner emits a tab-separated single-action matrix over three controller policies, four deterministic fault schedules, and two tool idempotency contracts.

## Compared policies

- blind_retry_once: retries after a missing acknowledgement without checking state; reports completion when the retry is acknowledged.
- stop_on_unknown: never retries an ambiguous non-idempotent action; preserves the unknown terminal state.
- reconcile_first: uses the POAR-RX lifecycle, performs read-back, retries only after authoritative non-commit evidence, and reports completion only after the postcondition is verified.

The matrix also includes an enforced idempotency-key contract. That contract is a property of the simulated tool, not a promise that a client-side key makes an arbitrary external API idempotent.

## Fault schedules

- No fault.
- Timeout before commit.
- Commit followed by lost acknowledgement.
- Commit followed by unavailable authoritative read-back.

The effect store counts actual non-idempotent effects and deduplicates only when the simulated server enforces the same idempotency key. This makes duplicate effects observable even when the controller's second request receives a normal acknowledgement.

## Metrics

Every row preserves separate fields for dispatch attempts, read-back calls, actual effects, completion reports, postcondition verification, duplicate effects, false completion, and unverified completion claims. No weighted aggregate is computed.

For this microbenchmark:

- duplicate_effect means more than one effect was applied for the one logical action.
- false_completion means the controller reported completion while the oracle effect count was not exactly one.
- unverified_completion_claim means the controller reported completion without authoritative postcondition verification.

An acknowledged retry can therefore be a false completion even when the transport returns success.

## Current claim boundary

This v0.1 single-action simulator is deliberately narrow. It is not a production adapter, a protected benchmark service, or evidence of a Symthaea SOTA result. Its oracle is local to the simulator.

## Two-target workflow experiment

The additional `workflow` module compares blind whole-workflow replay, saga compensation, and proof-gated reconciliation over eight deterministic fault families and two compensation-authorization states (48 rows). It detects lost concurrent updates, unauthorized effects, duplicate logical writes, false completion and failed compensation. See `research/pod-meta-lab/POAR_RX_PARALLEL_WORKFLOW_V1.md` for the protocol, external comparisons, metrics and claim limits.

Run it with:

~~~sh
cargo run --manifest-path research/poar-rx-lab/Cargo.toml --bin poar-rx-workflow-matrix
~~~

The two-target simulator is still a deterministic microbenchmark. It does not model durable crash consistency, real distributed systems, real IAM, arbitrary asynchronous workflows or production external APIs. It uses a public local oracle; no general recovery or SOTA claim follows from it.
