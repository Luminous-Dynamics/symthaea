# symthaea-assurance-requalification

Fail-closed, hysteretic recovery for evidence-bearing assurance states.

The central rule is simple:

> one favorable sample cannot erase a prior restricted, incomplete, unsafe, or fail-closed state.

Recovery requires an explicit reviewed requalification authorization plus a sustained sequence of fresh, distinct nominal evidence satisfying deployment-reviewed count, span, gap, and authorization-age requirements.

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

Structurally valid sample IDs are tombstoned before temporal admission. A sample rejected for non-monotonic timing therefore cannot later be replayed with a modified timestamp and become favorable evidence.

Requalification authorization requires:

- unique authorization id
- reviewed recovery procedure reference
- reviewer/process reference
- authorization time after the latched failure and latest observation
- supporting evidence references
- completion before the policy's maximum authorization age

The authorization window must be at least as long as the required nominal evidence span; impossible recovery policies fail validation.

## Deliberate boundaries

This crate does not infer identity, intent, risk, targeting, engagement, or physical authority. It only controls whether a previously degraded assurance state may return to nominal.

```bash
cargo test -p symthaea-assurance-requalification
```
