# Engineering Trust Clock Governance Time Envelope V1

## Purpose

Freeze the temporal theorem required before existing threshold-ceremony, policy-migration, and trust-rotation machinery can safely consume the trusted operational clock.

The existing fabrication governance paths evaluate approval, key, snapshot, and activation timing against a scalar caller-supplied `now_unix_s`. For clock governance, replacing that scalar with the candidate clock's consensus timestamp would discard the uncertainty interval and could make a transition appear valid even though it is invalid for some possible true time inside the accepted window.

## Governing theorem

```text
OperationalClockBasisV1
        ↓
GovernanceEvaluationEnvelopeV1
[lower_unix_ms, upper_unix_ms]
        ↓
all temporal predicates must hold
for every possible true time in the interval
```

For a governance action to be valid throughout `[lower, upper]`:

- approval / key / trust-snapshot validity must cover the entire interval;
- `not_before <= lower`;
- `not_after > upper`;
- an activation that must not be in the past requires `activation >= upper`;
- an activation constrained by `maximum_delay` requires `activation <= lower + maximum_delay`.

The consensus timestamp remains useful evidence, but it is not substituted for the interval in authority-critical temporal predicates.

## Frozen fixture

```text
OperationalClockBasisV1
f9d30baa62bb317c52393f0fa181c8c0ceabe77bbe06d16ca8dc79e10a51443b

VerifiedClockWindow V1
5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229

TrustSnapshot
609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633

lower_unix_ms     1499920
upper_unix_ms     1500100
consensus_unix_ms 1500010

GovernanceEvaluationEnvelopeV1
afbbee8cf9de3d39600c3d3a51901e3563467b13be664a681e4b5cd84c90082d
```

## Adversarial findings

The exact oracle rejects:

- an approval that expires at second 1500, because the trusted interval extends to 1500.100 s;
- a snapshot that expires at second 1500;
- a key that only becomes valid at second 1500;
- inactive or wrong-purpose keys;
- activation at second 1500 because it may already be in the past at the upper bound;
- activation at second 1510 under a 10-second maximum delay because it is too late when measured conservatively from the lower bound.

The last two cases show why simply evaluating against one consensus timestamp is insufficient.

## Exact-byte evidence

```text
Python                     3.13.5
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         6ead2d86232f6556f84f242815c84705ea87b73eaee32457841a430fde581bbb
locally computed Git blob  4aafbf5ab9be5283446db1ba696fa4dfca5f5a3b
GitHub stored Git blob     4aafbf5ab9be5283446db1ba696fa4dfca5f5a3b
```

Repository CI remains separate qualification evidence.

## Production consequence

Clock-governed policy migration and trust rotation should adapt existing governance semantics to an opaque interval-safe evaluation capability derived from `OperationalClockBasisV1`.

They should **not** accept an unconstrained caller `now_unix_s`, and they should not collapse the trusted interval to its consensus timestamp for signer/snapshot/approval eligibility or activation-delay checks.

This reference does not itself authorize a policy migration or trust rotation. It only freezes the temporal authority boundary those adapters must satisfy.

## Deliberate nonclaims

No policy migration, trust rotation, signer/provider qualification, cryptographic threshold primitive, ETK currentness, engineering requirement satisfaction, deployment approval, or physical actuation authority is established here.
