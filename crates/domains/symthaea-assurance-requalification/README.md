# symthaea-assurance-requalification

Fail-closed, hysteretic recovery for evidence-bearing assurance states.

The central rule is simple:

> one favorable sample cannot erase a prior restricted, incomplete, unsafe, or fail-closed state.

Recovery requires an explicit reviewed requalification authorization plus a sustained sequence of fresh, distinct nominal evidence satisfying deployment-reviewed count, span, and gap requirements.

## Monotonic restriction

While a restriction is latched, later non-nominal observations can only retain or increase the effective restriction. For example, an `Unsafe` latch cannot become merely `Restricted` because one later subsystem report looks less severe.

An explicit requalification authorization does **not** reduce the effective restriction. The old restriction remains externally visible while evidence accumulates.

Only successful completion of the reviewed recovery criteria returns the gate to `Nominal`.

## Required evidence

Every recovery sample requires:

- unique sample id
- monotonic observation time
- explicit upstream assurance state
- non-empty evidence references

Requalification authorization requires:

- unique authorization id
- reviewed recovery procedure reference
- reviewer/process reference
- authorization time after the latched failure
- supporting evidence references

## Deliberate boundaries

This crate does not infer identity, intent, risk, targeting, engagement, or physical authority. It only controls whether a previously degraded assurance state may return to nominal.

```bash
cargo test -p symthaea-assurance-requalification
```
