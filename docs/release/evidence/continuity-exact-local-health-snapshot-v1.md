# Exact local healthy snapshot v1 — qualification subject

Status: **implementation subject only; not qualified evidence**.

Exact parent:

`architecture/continuity-crash-reconciliation-v1`

Parent exact head at branch creation:

`b0d65c0e3086ac990d5564799519ae190a417778`

The theorem under qualification is:

```text
qualified exact-B target health == Healthy
    -> healthy-local snapshot with PostExecutionTarget basis

qualified crash reconciliation == exact source A observed
+ authenticated exact source-A health == Healthy
    -> healthy-local snapshot with CrashReconciledSource basis
```

Required fail-closed properties include source-reconciliation mismatch, source/attempt mismatch, health predating identity, verifier-root substitution, health-profile substitution, zero/missing evidence digests, stale/future health evidence, non-Healthy snapshot input, and basis-dependent snapshot identity.

The snapshot grants no distributed-health, recovery, promotion, retry, or physical execution authority.

No test or CI pass is claimed here. Exact-head CI and parent qualification remain required.
