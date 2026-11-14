# symthaea-organizational-closure

Evidence-oriented primitives for the AOBench runtime self-maintenance track.

## Scope

This crate verifies a bounded functional self-maintenance contract:

- typed component classes prevent immutable roots and external resources from being counted as internally regenerated;
- process versions and required/optional input/output contracts are explicit;
- artifact versions are immutable and keyed by component ID plus version;
- SHA-256 is recomputed over the stored bytes;
- a downstream consumer must reference the exact earlier verified output and producer execution;
- external dependency observations must use the process allow-list;
- failed, abstained or interrupted events cannot expose committed outputs;
- required regenerative components need both verified producer and consumer evidence.

The report predicate is VerificationReport::qualifies_functional_self_maintenance(). It does not claim runtime organizational recurrence, causal closure, biological autopoiesis, life or consciousness.

## Run

From the workspace root:

```bash
cargo fmt -p symthaea-organizational-closure -- --check
cargo check --locked -p symthaea-organizational-closure --all-targets
cargo test --locked -p symthaea-organizational-closure
cargo clippy --locked -p symthaea-organizational-closure --all-targets -- -D warnings
```

The focused workflow is .github/workflows/aobench-runtime.yml.

## Trust boundary

The verifier recomputes content digests but does not authenticate the trace producer, verify that a process actually ran on the claimed executable, validate external receipt signatures, or make authorization decisions. The authorization_policy_digest is an identity reference only. Production adapters must bind events to an independently trusted runtime/logger and keep the verifier outside the agent's write authority.

This crate is a first implementation tranche. See docs/research/autopoiesis/RUNTIME_EVIDENCE_LAYER_2026-11-14.md and the [AOBENCH-001 issue](https://github.com/Luminous-Dynamics/symthaea/issues/11980).
