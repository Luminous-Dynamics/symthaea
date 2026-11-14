# symthaea-organizational-closure

Evidence-oriented primitives for the AOBench runtime self-maintenance track.

## Scope

This crate verifies a bounded functional self-maintenance contract:

- typed component classes prevent immutable roots and external resources from being counted as internally regenerated;
- process versions and frozen expected executable digests are explicit;
- artifact versions are immutable and keyed by component ID plus version;
- SHA-256 is recomputed over the stored bytes;
- a downstream consumer must reference the exact earlier verified output and producer execution;
- required external dependencies must produce an explicit runtime observation; optional dependencies may be absent;
- observed dependencies must use the process allow-list and available observations need a receipt digest;
- failed, abstained or interrupted events cannot expose committed outputs;
- required regenerative components need both verified producer and consumer evidence.

The report predicate is VerificationReport::qualifies_functional_self_maintenance(). It does not claim runtime organizational recurrence, causal closure, biological autopoiesis, life or consciousness.

## Run

From the package directory (this is an independently locked nested Cargo workspace):

```bash
cargo fmt --all -- --check
cargo check --locked --all-targets
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

The focused workflow is .github/workflows/aobench-runtime.yml.

## Trust boundary

The verifier recomputes artifact content digests and checks the event's executable digest against the frozen process specification, but it does not authenticate the trace producer or prove that the executable actually ran. It does not validate external receipt signatures or make authorization decisions. The authorization_policy_digest is an identity reference only. Production adapters must bind events to an independently trusted runtime/logger and keep the verifier outside the agent's write authority.

This crate is a first implementation tranche. See docs/research/autopoiesis/RUNTIME_EVIDENCE_LAYER_2026-11-14.md and the [AOBENCH-001 issue](https://github.com/Luminous-Dynamics/symthaea/issues/11980).

## Organizational reconfiguration gate

The `reconfiguration` module compares a frozen `OrganizationSpec` with a candidate and reports the exact component/process/edge diff. It enforces explicit addition/removal/contract-change allowlists, a structural-edit budget, referential integrity, required-versus-optional edge consistency, and hard protection for provisioned roots, constraints, viability-critical components and operator-designated processes. Existing process input edges to provisioned roots and constraints cannot be severed or weakened, and candidate processes cannot produce provisioned roots or externally governed constraints.

This is only a **structural admission** result. A passing candidate is not thereby authorized, viable, causally beneficial, or proven to have executed. Promotion still requires external authority and durable receipts; viability, runtime recurrence, intervention isolation and capability preservation remain separate evidence planes.

Focused API: `evaluate_reconfiguration(baseline, candidate, policy)`.
