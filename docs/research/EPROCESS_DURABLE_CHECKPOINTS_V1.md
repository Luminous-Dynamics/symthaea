# Durable Resume Checkpoints for E-Process Monitors V1

Status: isolated Rust prototype with unit-level resume-equivalence and corruption-rejection tests. It does not change the live Symthaea controller. Exact-head compilation, formatting, Clippy and tests are required before calling it qualified.

## Why checkpoint correctness is part of the statistical guarantee

An anytime-valid monitor is only continuous while its evidence state and allocated error budget are continuous. A service restart that reconstructs an e-process at M0 = 1 and reuses the same alpha allocation effectively restarts the statistical test. Repeated restarts can quietly turn a stream-level or lifetime guarantee into a per-process-run guarantee.

This branch adds a versioned checkpoint codec to the three-component mixture e-process used by the runtime-control experiments. It stores the calibrated contract and exact component log-e-values, not just the aggregate statistic. Resume validation recomputes the mixture and threshold from the saved parameters and rejects inconsistent state.

## Stored fields

The checkpoint contains:
- a versioned magic header;
- risk bound, three bet fractions, global delta and simultaneous-monitor count;
- the one-based alpha-spending episode index, if the monitor uses episode-spent alpha;
- the configured log threshold;
- each component's log-e-value and the aggregate mixture log-e-value;
- the latched alarm state; and
- an FNV-1a 64-bit checksum over the canonical body.

Floating-point values are encoded by their exact IEEE-754 bit patterns in canonical hexadecimal, so a text decode/encode does not round or alter the evidence state. The restored monitor checks the saved threshold against a threshold recomputed from the contract, checks component values are finite, recomputes the log-mixture, checks aggregate/component consistency, and rejects an unlatched state whose evidence is already above its alarm boundary.

## Security boundary

The FNV checksum detects accidental corruption, truncated writes, and partial/malformed checkpoints. It is not cryptographic authentication: someone able to rewrite the checkpoint can also recompute FNV. Production deployment should place this payload inside an authenticated storage envelope (for example a keyed MAC or digital signature), bind it to monitor/controller identity and software version, and store it atomically with replay protection. The dependency-free benchmark codec does not pretend to supply those production properties.

The alpha-spending episode index is retained in the checkpoint. A resumed episode must continue under that same index; a new episode must be assigned a fresh predeclared index. Both fixed and mixture e-process APIs now refuse an in-place reset when an alpha-spending index is attached; the call returns false without mutating evidence or the latch. A new episode must construct a fresh monitor with its next predeclared index. This makes it harder for a caller to accidentally reuse an already-spent budget through the public reset API. Resetting state must never mint an additional alpha budget. Checkpoints are versioned so a semantic change to the evidence recurrence or threshold contract requires a version bump and explicit migration policy.

## Tests

Unit tests compare uninterrupted and checkpoint-restored execution bit-for-bit over the remaining loss stream, including every component value, aggregate log-evidence and alarm state. Additional tests reject truncated, corrupted and wrong-version payloads and verify that a latched alarm and episode alpha boundary survive restoration.

The tests cover only in-process codec behavior. A later integration test should write to a temporary file, sync file and parent directory where applicable, atomically replace the previous checkpoint, kill/restart the monitor process, and verify equivalence. That crash-consistency test is intentionally separate from this dependency-free unit test so it can exercise the deployment store's actual durability semantics.

## Next production requirements

1. Authenticated envelope: MAC/signature, key rotation, controller identity and compatible code-version identifier.
2. Atomic write protocol: write temporary file, sync file and parent directory where applicable, rename atomically, and keep a recovery generation.
3. Replay protection: durable monotone generation and episode index; reject stale snapshots after newer state was committed.
4. Fail-closed recovery: a missing, corrupt, stale or incompatible checkpoint must not silently reset to M = 1. The supervisor should enter a declared degraded/no-certified-action state or reinitialize under a new explicitly accounted budget.
5. Test crash points before write, after write, after sync, and around rename; verify restored behavior against an uninterrupted reference trace.

## Relationship to external research

The e-process is motivated by Timans et al., [On Continuous Monitoring of Risk Violations under Unknown Shift, UAI 2025](https://proceedings.mlr.press/v286/timans25a.html). Lifetime alpha spending and persistent-versus-reset experiments are described in [EPROCESS_LIFETIME_ALPHA_SPENDING_V1.md](EPROCESS_LIFETIME_ALPHA_SPENDING_V1.md) and [EPROCESS_PERSISTENT_VS_RESET_V1.md](EPROCESS_PERSISTENT_VS_RESET_V1.md). The checkpoint work preserves statistical evidence across restarts; it does not improve the underlying risk model or establish physical safety.
