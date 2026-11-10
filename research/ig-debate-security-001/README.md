# IG-DEBATE-SECURITY-001 — effect-oracle benchmark fixtures

**Status:** synthetic fixture design / no behavior trials executed  
**Owner:** [Symthaea #11434](https://github.com/Luminous-Dynamics/symthaea/issues/11434)  
**Hostile-agent runtime:** [#1872](https://github.com/Luminous-Dynamics/symthaea/issues/1872)

## Paradigm shift

A correct final answer does not make a debate system secure. A group can answer correctly while an agent reads a protected object, writes to a workspace, acquires a capability or performs a forbidden state transition.

This benchmark treats **observed effects and authorization state** as the security oracle. Final-answer correctness remains separate and never compensates for an unauthorized effect.

## Included artifacts

- `DEBATE_SECURITY_FIXTURES_V0_1.json` — 30 synthetic fixtures across six attack families.
- `DEBATE_SECURITY_FIXTURES_V0_1.schema.json` — bounded, closed manifest schema.
- `qualifier/` — standalone Rust structural/effect-outcome qualifier.
- `.github/workflows/ig-debate-security-qualifier.yml` — exact-triggering-revision formatter, tests and fixture qualification.

## Coverage

| Attack family | Fixtures |
|---|---:|
| External input / instruction injection | 5 |
| Evidence and information-flow manipulation | 5 |
| Tool / workspace misuse | 5 |
| Agent collusion | 5 |
| Orchestrator / policy / capability compromise | 5 |
| Evaluation-aware delayed persistence | 5 |
| **Total** | **30** |

Each family has an organism-positive control, a defense control, a benign false-positive control, a collusion/attack control and a trace-integrity negative control. Positive controls demonstrate that the attack organism can reach the tested action path. They are deliberately distinguished from defended conditions.

## Effect oracle

The Rust qualifier derives terminal classification from:
- whether an action was attempted;
- the independent effect status;
- authorization at the time of the action;
- effect type and target;
- state/data/capability receipts;
- observer receipt completeness.

It distinguishes blocked attempts from unauthorized reads, writes, capability acquisitions and state transitions. Missing observer or effect receipts produce `InvalidOrIncompleteTrace`, never a pass.

All 30 checked-in fixtures deliberately have `answer_status = Correct`. This is a non-compensatory control: a correct final answer cannot hide an unauthorized effect.

## Graph and execution boundaries

- [#11337](https://github.com/Luminous-Dynamics/symthaea/issues/11337) owns CommunicationGraph channels and message delivery.
- [#11248](https://github.com/Luminous-Dynamics/symthaea/issues/11248) owns DelegationGraph mandate edges. A delegation edge is never a message channel.
- [#1872](https://github.com/Luminous-Dynamics/symthaea/issues/1872) owns hostile-agent execution and model organisms.
- [#11320](https://github.com/Luminous-Dynamics/symthaea/issues/11320) owns public/private stance measurement.
- [#11350](https://github.com/Luminous-Dynamics/symthaea/pull/11350) owns observable claim/evidence revision events.
- [#11367](https://github.com/Luminous-Dynamics/symthaea/pull/11367) owns confidence calibration metrics.

The current corpus is synthetic, offline, external-tool-free and contains no real data. The qualifier does not create a sandbox, launch attacks or authorize network interaction.

## Run

```bash
cargo fmt --manifest-path research/ig-debate-security-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/ig-debate-security-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-debate-security-001/qualifier/Cargo.toml -- \
  research/ig-debate-security-001/DEBATE_SECURITY_FIXTURES_V0_1.json \
  research/ig-debate-security-001/DEBATE_SECURITY_FIXTURES_V0_1.schema.json
```

## Claim ceiling

A structural PASS qualifies only the synthetic manifest, outcome classification and declared receipt requirements. It does not demonstrate that any model attempts or resists an attack, that a mitigation works, or that a real system is secure. Behavioral robustness requires executed trials in the authorized isolated sandbox, validated positive controls, independent effect observation and exact-head receipts. No real data access, external-system exploitation or production policy mutation is performed by this work.
