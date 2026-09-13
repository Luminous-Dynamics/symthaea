# Engineering Trust Clock Operational Production V1

## Scope

Production counterpart to #2663's independent recursive operational-clock theorem.

This tranche replaces the one-shot normal-operation type progression with one opaque capability:

```text
AcceptedClockBasisV5
        ↓ bind exact witness records
OperationalClockBasisV1(epoch 42)
        ↓ pre-candidate permit
OperationalClockBasisV1(epoch 43)
        ↓ pre-candidate permit
OperationalClockBasisV1(epoch 44)
        ↓ ...
```

The same `OperationalClockBasisV1` type is both successor output and next-transition authority input.

## Authority properties

- Bootstrap conversion accepts policy/snapshot **witness records**, but they must reproduce IDs already committed by the opaque V5 basis.
- Normal successor permit derivation takes only `&OperationalClockBasisV1`; it has no policy, snapshot, observation, timestamp, window or witness parameter.
- Successor acceptance takes only the private permit, original signed observations, and the cryptographic observation verifier.
- Quorum and continuity policies are retained privately and revalidated before use.
- Trust-snapshot contents are retained privately and revalidated before each future transition envelope.
- The current verified window and evaluation witness are retained; a continuous basis also retains the immediately prior window and verified continuity witness.
- Authority-output IDs/capabilities expose no public parse/deserialization constructor.
- Policy migration and trust rotation are deliberately outside this normal path.

## Frozen public-API vectors

The Rust corpus pins the exact #2663 independent reference values:

```text
OperationalClockBasisV1 epoch 42
f9d30baa62bb317c52393f0fa181c8c0ceabe77bbe06d16ca8dc79e10a51443b

ClockSuccessorEvaluationPermitV2 42→43
750c044e48a5395ee04a457ba7b89cb2397c2bf2cea76e96e71e50dcd338b73b

OperationalClockBasisV1 epoch 43
0e36e238e300ae1d626846220bbba84f5e87a50be7ffa2ac3b8e7822f4d47810

ClockSuccessorEvaluationPermitV2 43→44
68b6dd92d42f3b42093f33d310d69eb8ac1bb259b02bd698256f67ea872b8a3d

OperationalClockBasisV1 epoch 44
9373fac49b1b488859de56cd6cd588b8e5868cf9fb455c8b3546b46974d8ffce
```

The corpus also pins the legacy successor window/witness/continuity vectors for both generations.

## Fail-closed corpus

Source tests cover:

- bootstrap policy witness substitution;
- bootstrap trust-snapshot substitution;
- retained quorum enforcement on the second successor generation;
- out-of-envelope successor windows;
- retained continuity-policy epoch-step enforcement;
- snapshot expiry before a later transition envelope;
- cryptographic signature rejection;
- source-level API ratchets forbidding policy/snapshot/time/window/witness arguments on normal successor functions;
- source-level ratchets forbidding `Serialize`/`Deserialize` on operational authority capabilities.

## Current evidence boundary

Source and vector parity are implemented. This environment has no local `rustc`, `cargo`, or `rustfmt`, so this document makes no compilation, formatting, Clippy or Rust-test claim.

Exact-head repository CI is the qualification gate.

## Deliberate nonclaims

This tranche establishes no policy migration authority, trust-snapshot rotation authority, real clock-provider qualification, ETK currentness, engineering requirement satisfaction, design qualification, deployment approval or physical actuation authority.
