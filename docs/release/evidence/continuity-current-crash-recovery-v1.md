# Current crash recovery v1 — qualification subject

Status: **implementation subject only; not qualified evidence**.

Exact parent:

`architecture/continuity-active-lkg-currentness-v1`

Parent exact head at branch creation:

`a027cfa24b7f090e8f535e59827409fd747aa94b`

Exact code subject frozen before this evidence-only commit:

`e4dcb7fdf4233473441f5c9e9984ef7c4bc08234`

This subject composes two deliberately separate facts:

```text
QualifiedCrashRecoveryToActiveKnownGoodV1
+ QualifiedActiveLkgCurrentnessV1
    -> QualifiedCurrentCrashRecoveryV1
```

The first proves that exact source A was independently rediscovered after an intent-only crash, proven locally Healthy, and surrounded by a freshly qualified distributed-health world. The second proves that a fresh rollback-resistant challenge/attestation names one exact active-LKG selection as current.

The composition exists only when the currentness proof names the exact same active selection id, source checkpoint id, continuity subject, and recovered realization A as the crash-recovery proof. The currentness anchor must be at or after the recovery qualification time.

This prevents a stale original A selection from being treated as the authoritative recovery head after a later LKG promotion has superseded it.

The persistent `CurrentCrashRecoveryRecordV1` records exact parent proof identities, currentness platform profile/root and anchor sequence, active selection/checkpoint, subject/realization, fresh challenge digest, recovery time, and currentness anchor time. The record is audit evidence only; `rebind()` requires both exact live parent proofs.

This proof means “recovered to A, and a fresh protected-head attestation still names A as active at this decision boundary.” It does not prove arbitrary external side effects were undone and grants no execution, retry, promotion, selection advancement, or bootstrap authority.

No test or CI pass is claimed here. Exact-head CI and all stacked parent qualification remain required.
