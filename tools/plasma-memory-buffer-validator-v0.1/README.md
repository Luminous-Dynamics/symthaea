# Plasma Memory Buffer Validator v0.1

**Status:** Standalone reference validator with authored unit tests. Rust compilation and test execution are pending exact-head CI.

## Purpose

Validate causal availability and semantic labeling for history slots consumed by online plasma moment-closure models.

The key distinction is between:

- **Physical abscissa**: the rational Runge–Kutta stage coordinate (c), describing which physical time a provisional state represents.
- **Computation order**: the order in which the integrator actually evaluated the stages.

These are not interchangeable. SSPRK-style stages can have (c = 0, 1, 1/2) while being computed in stage order 1, 2, 3. A later-computed stage can therefore represent a later physical time than the current query. Silently sorting only by time would destroy the solver dependency; silently treating computation order as physical chronology would hide the non-monotone time semantics.

## Three policies

- `PaperReported`: permits previously computed provisional stages to reproduce the published buffer behavior. It explicitly reports same-step provisional slots with later (c)-values.
- `AcceptedStateOnly`: allows only committed physical-time states; provisional stage values invalidate this policy.
- `StageExplicit`: requires an exact tableau digest and explicit declaration of same-step prior-stage dependencies when their (c)-value is later than the query.

Every policy rejects a slot from another solver run/trajectory, future step, not-yet-computed stage, stale step, duplicate evaluation position or malformed rational abscissa.

Empty history and insufficient history return `Unknown`, not `Valid`. The validator rejects unsorted computational-order inputs rather than silently rewriting the original history.

## What this is not

It does not train or execute an FNO, solve a Vlasov/fluid equation, decide whether a closure is physically valid, or prove solver stability. It validates caller-supplied identities and stage metadata only. The exact paper reproduction, matched solver runs and conservation/stability checks remain owned by Symthaea #12907 and Symtropy #1685.

## Known-answer fixture

The unit tests exercise a three-stage pattern with abscissae 0, 1, 1/2. They prove only that the validator preserves computation order while surfacing the future-abscissa stage explicitly. They do not prove that the paper's published model uses this exact buffer for every configuration or that another buffer policy is scientifically superior.

## Run

```bash
cargo fmt --manifest-path tools/plasma-memory-buffer-validator-v0.1/Cargo.toml -- --check
cargo test --manifest-path tools/plasma-memory-buffer-validator-v0.1/Cargo.toml --locked
cargo clippy --manifest-path tools/plasma-memory-buffer-validator-v0.1/Cargo.toml --locked --all-targets -- -D warnings
```
