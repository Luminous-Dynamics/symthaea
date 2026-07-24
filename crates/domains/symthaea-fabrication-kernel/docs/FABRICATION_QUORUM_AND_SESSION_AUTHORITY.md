# Fabrication Quorum and Session Authority

Version 0.12 adds an operational authority layer above cryptographic manifest
verification and below physical printer submission.

## Authority chain

```text
AttestedFabricationManifest
  -> lifecycle-verified VerifiedAttestation
  -> role-aware ReleaseAuthorization
  -> fresh TimedMachineSession
  -> accepted and consumed MachineSessionLease
  -> GovernedAuthorizedPrintJob
  -> GovernedSubmittedJobReceipt
  -> runtime fault evidence and signed audit anchor
  -> OperationalFabricationReplayContract
```

Each arrow is fail-closed. The descriptive inputs remain available for
inspection, but only the capability-bearing output of the preceding stage may
cross the next high-trust boundary.

## Release quorum

`ReleasePolicy` maps exact `(algorithm, key_id)` identities to independent roles.
A policy can require:

- a minimum number of distinct recognized signers;
- one or more signers for design, manufacturing, safety, operations, or named
  custom roles;
- cryptographic algorithm diversity;
- bounded signer and policy inventories.

A valid signature from an active key does not satisfy a role unless the exact
identity is bound to that role. The canonical policy digest is retained in print
and replay evidence.

## Delegation

A delegation grant is deliberately narrow:

- one delegator;
- one already lifecycle-verified manifest signer;
- one release role;
- one exact manifest digest;
- one bounded validity window;
- one canonical nonce.

The delegator must itself be an active policy role holder and eligible for
fabrication-manifest authority. Delegation cannot create a trusted key, cross a
manifest boundary, survive revocation, or outlive its explicit window.

## Timed machine sessions

Legacy untimed sessions remain available to old callers, but the governed path
requires `TimedMachineSession` evidence. It binds:

- machine identity and capabilities;
- session nonce;
- monotonically increasing session sequence;
- issue and expiry timestamps;
- a deterministic SHA-256 digest.

`MachineSessionTracker` rejects sequence rollback, same-sequence substitution,
nonce reuse, superseded sessions, expired leases, and repeated consumption.
The tracker is serializable so a machine gateway can persist anti-replay state.

## Runtime containment evidence

The standard fault matrix executes the actual `ExecutionGuard` against:

- heartbeat loss;
- progress stall and regression;
- nozzle and bed runaway;
- nozzle and bed control deviation;
- time regression;
- non-finite sensor input.

Every report contains the observed decisions and a deterministic digest. The
operational replay contract requires one intact report for every standard
scenario; partial matrices are not accepted as release evidence.

## Audit export and anchoring

Audit journals can be exported as independently verifiable bounded segments.
A segment retains its expected predecessor head, exact sequence range, event
hashes, and segment digest.

A signed audit anchor binds the complete journal digest, chain head, event count,
trust snapshot, timestamp, and external anchor identifier. Anchor signers require
the dedicated `AuditAnchor` key usage. Adding, deleting, reordering, or altering
journal events invalidates an earlier anchor.

## Operational replay

`OperationalFabricationReplayContract` extends governed deterministic replay
with the exact:

- release policy and optional delegation evidence;
- timed machine-session digest and sequence;
- signed audit anchor;
- complete fault-matrix digest.

This contract is evidence of identity and deterministic agreement. It is not a
substitute for authenticated machine transport, durable state storage, real
cryptographic provider validation, supervised actuator fault injection, or a
successful workspace build and test run.
