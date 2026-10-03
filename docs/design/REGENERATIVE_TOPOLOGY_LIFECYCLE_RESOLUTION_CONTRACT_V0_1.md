# Regenerative Topology Lifecycle Resolution Contract v0.1

## Purpose

Topology epoch continuity detects rollback, skips, unrelated predecessors, and concurrent successor forks. It does not decide which branch is authoritative.

This contract defines the next boundary:

> When distributed lifecycle statements disagree, how can an authoritative resolution be represented without erasing the competing history?

The answer is an explicit resolution record that selects one observed successor while preserving every competing branch reference supplied to the local decision.

## Resolution model

A topology lifecycle is represented as an append-only chain:

**predecessor epoch + digest → lifecycle event → successor epoch + digest**

A distributed system can produce:

**epoch 1 / v1 → epoch 2 / v2a**

and concurrently:

**epoch 1 / v1 → epoch 2 / v2b**

Neither branch becomes current merely because it arrived first, has a newer local timestamp, or has more local copies.

The lifecycle resolution record instead binds:

- asset identity;
- component identity;
- predecessor epoch and digest;
- selected successor epoch and digest;
- the complete set of observed competing successor branches;
- stable resolution identity;
- authority identity;
- authority statement digest;
- verification reference;
- resolution time; and
- configuration identity.

## Deterministic local rules

The resolution gate:

1. rejects malformed identities and invalid epoch transitions;
2. requires the configured authority identity;
3. rejects future-dated resolution records;
4. requires the selected successor to be one of the observed branches;
5. requires every locally observed branch to be represented in the resolution record;
6. preserves all observed branches in the resulting decision;
7. reports an unresolved fork as Conflicted;
8. reports a well-formed but unresolved single successor as InsufficientEvidence;
9. never chooses a branch on its own.

The local gate therefore validates the shape and completeness of a resolution. It does not perform cryptographic verification of the authority statement.

## Historical preservation

A selected branch is not substituted for history.

For example, if the observed set is:

- v2a
- v2b

and the authority selects v2a, the resulting decision retains both references.

The semantic result is:

**current candidate = v2a; historical competing evidence = v2b**

The losing branch remains available for audit, incident analysis, later revocation, or a subsequent authoritative correction.

This prevents the common failure mode in which conflict resolution mutates state until the conflict is no longer visible.

## Authority boundary

This contract deliberately stops before trust-anchor verification.

Symthaea can enforce:

**expected authority identity → resolution statement identity → selected observed branch → preserved competing branches**

but the authoritative provenance system must establish whether that authority statement is actually valid.

This maps cleanly to the broader remote-attestation separation between evidence, verification, and relying-party decisions. Current IETF Epoch Marker work similarly treats shared freshness as something that should be established explicitly and securely conveyed rather than inferred solely from local clocks. 

## Safety invariant

**Never make a distributed lifecycle conflict disappear by overwriting one branch with another.**

A resolution may make one branch current, but the existence of the competing branch remains part of the evidence history.

## Non-goals

This contract does not:

- prove that the selected topology is physically correct;
- prove that the authority is honest;
- perform cryptographic signature verification;
- diagnose physical damage;
- certify repair or recovery;
- establish regulatory safety;
- automatically reconcile incompatible physical histories.

Those remain separate authority, physics, and recovery boundaries.


## Resolution continuity

The authority layer now has its own monotonic resolution epoch.

A later authority decision does not overwrite an earlier decision. Instead:

**resolution epoch N + resolution digest → authority event → resolution epoch N+1 + new resolution digest**

For the initial resolution, no predecessor resolution digest is permitted. For later resolutions, the expected predecessor digest must match policy.

This closes the replay seam in which an older, still-validly formatted authority decision could be presented again as though it were the current resolution.

The resulting lifecycle has two linked histories:

**physical topology history**
predecessor topology digest → successor topology digest

**authority decision history**
previous resolution digest → new resolution digest

These histories are intentionally separate. A new authority resolution may change which topology branch is considered current without rewriting the underlying physical/topology statements.

The architecture therefore preserves:

- the original topology statements;
- all competing successor branches;
- the authority decision that resolved a fork;
- later authority decisions that supersede or correct earlier decisions.

A future Mycelix implementation can bind these records to signed lifecycle events and an append-only transparency mechanism. SCITT's published architecture provides a useful external analogy: signed statements can be recorded in an append-only verifiable structure and accompanied by receipts, while the content semantics remain application-specific.

### Security invariant

**An older authority decision must never regain current authority merely because its signature or validity interval remains acceptable.**

Currentness is a lifecycle property, not merely a validity-window property.


## Downstream audit identity

A resolved decision exposes three stable identifiers for downstream evidence chaining:

- the resolution ID;
- the monotonic resolution epoch; and
- the authority statement digest.

These are deliberately distinct. The resolution ID names the lifecycle decision, the epoch establishes its position in the authority-decision history, and the statement digest identifies the exact authority statement.

A downstream evidence envelope can therefore reference the decision without copying or reinterpreting the authority statement itself.

The contract still does not calculate or cryptographically verify that digest. It treats the digest as an authoritative-layer identity claim and leaves verification to the trust/provenance system.
