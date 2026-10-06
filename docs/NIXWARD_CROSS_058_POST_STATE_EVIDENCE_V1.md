# Nixward CROSS-058 — Effect-Bound Post-State Receipt V1

## Purpose

CROSS-058 defines a closed evidence protocol for proving the observed outcome of an already-authorized systemd service effect.

The protocol deliberately separates four questions:

1. Was the effect authorized?
2. Was the requested typed effect dispatched?
3. What state did the host actually expose afterward?
4. Did that observed state satisfy the effect-specific postcondition strongly enough to claim proof?

Phi, confidence, prediction quality, anomaly scores, and other semantic values are outside this authority chain.

## Binding

A NixPostStateReceiptV1 binds:

- exact action-intent digest;
- exact authorization-record digest;
- an effect digest derived from operation, unit, authorized generation, authorized definition identity, pre-invocation identity, and stability contract;
- target unit;
- authorized and observed generation;
- authorized and observed systemd definition identity;
- systemd job identity/type/result when a lifecycle job is required;
- pre/post invocation IDs;
- effect-specific postcondition assessment;
- observed timestamp;
- stability evidence when required;
- observer identity/version.

Receipt integrity is represented by a domain-separated BLAKE3 digest. The digest is an integrity identifier; consumers must still preserve the trusted receipt/reference boundary rather than treating a self-reported digest as a signature.

## Claim ladder

The protocol intentionally distinguishes:

- Observed: the effect-specific postcondition was satisfied at an observation point, but no stability contract was required.
- Proven: the effect-specific postcondition was satisfied and the declared stability contract was satisfied.
- Violated: authoritative evidence contradicts the requested postcondition.
- Unproven: available evidence is insufficient to establish the requested postcondition.

Therefore:

dispatch success != state observed != stable state != proven effect

## Effect-specific predicates

Lifecycle effects require matching systemd job evidence and done as the JobRemoved result.

- Start: job type start + result done + ActiveState=active.
- Stop: job type stop + result done + ActiveState=inactive.
- Restart: job type restart + result done + ActiveState=active + a post invocation identity different from the captured pre invocation identity.
- Reload: job type reload + result done + ActiveState=active.
- Enable: UnitFileState=enabled.
- Disable: UnitFileState=disabled.

Unknown future job-result vocabulary does not become success merely because it is non-empty.

## Stability

Stability is explicit evidence rather than an arbitrary sleep.

A stability record must:

- contain a non-zero required window;
- cover at least the required duration;
- contain at least two samples;
- report no state-change timestamp later than the beginning of the stability window;
- end no later than the observation timestamp.

This is deliberately a conservative protocol. A future observer may strengthen it with actual repeated state snapshots, event subscriptions, or stronger monotonic-clock evidence.

## Definition identity

NixSystemdUnitDefinitionIdentityV1 commits to the exact observed FragmentPath and sorted DropInPaths set. This is source identity, not a content hash.

That distinction is intentional:

> path identity proves which definition sources systemd reported; it does not prove that arbitrary file contents were independently hashed.

A later hardening tranche should add a read-only definition-content evidence layer if the threat model requires content equivalence rather than source-path equivalence.

## Current transport boundary

CROSS-058 is transport-neutral. Existing Nixward service observation currently uses a strict systemctl show transport. A future CROSS-059 should add a narrowly scoped read-only D-Bus observer using the existing optional zbus dependency and feed its structured observations into this protocol.

Current systemd source exposes unit-level definition metadata such as FragmentPath, DropInPaths, and state-change timestamps, while the job path exposes job identity/type and emits JobRemoved results. See:

- https://github.com/systemd/systemd/blob/main/src/core/dbus-unit.c
- https://github.com/systemd/systemd/blob/main/src/core/dbus-job.c
- https://github.com/systemd/systemd/blob/main/src/systemctl/systemctl-start-unit.c

The current systemd documentation also describes normalized microsecond time properties and distinguishes backing-file/source inspection from runtime state. See:

- https://github.com/systemd/systemd/blob/main/man/systemctl.xml

## Security boundary

This module does not:

- mint execution authority;
- accept Phi/confidence as permission;
- perform host mutation;
- convert observations into authorization;
- establish file-content authenticity;
- eliminate all TOCTOU between dispatch and observation.

The intended chain is:

typed authority -> typed effect -> systemd execution -> independent observation -> postcondition evaluation -> receipt

The remaining architectural goal is to ensure the executor and observer have different capabilities so that mutation authority is never required merely to inspect outcome state.

## Qualification vectors

The minimum hostile suite for this protocol is:

1. valid intent + valid authorization + matching job/state -> accepted;
2. mismatched intent effect -> rejected;
3. authorization bound to a different intent -> rejected;
4. denied authorization -> rejected;
5. wrong unit -> rejected;
6. wrong generation -> rejected;
7. wrong definition identity -> rejected;
8. missing lifecycle job -> unproven;
9. wrong job type -> violated;
10. failed job result -> violated;
11. wrong post-state despite successful job -> unproven;
12. restart without new invocation -> unproven;
13. stability window shorter than required -> rejected;
14. state change during stability window -> rejected;
15. forged Proven claim with removed/shortened stability -> rejected;
16. receipt field mutation -> different digest.

## Qualification rule

Do not call this protocol PASS solely because a branch exists or a workflow is queued.

The exact commit under test must have an executed, successful focused qualification run before the corresponding claim is promoted to PASS evidence.
