# Regenerative Sensor Temporal Corroboration Contract v0.1\n\n## Purpose\n\nThe first fusion boundary answers a cross-sectional question:\n\n> Do multiple currently trusted sensors agree closely enough to corroborate the present observation?\n\nThat is necessary but insufficient. A stuck, drifting, or slowly biased sensor can remain individually trusted and still agree with a cohort at one instant.\n\nTemporal corroboration adds a second evidence boundary:\n\n> Do the changes reported by independent trusted sensors agree over time?\n\nThis is deliberately separate from structural health inference.\n\n## Deterministic pipeline\n\nPhysical state → sensor field → per-sensor qualification → cross-sectional corroboration → temporal corroboration → structural-health inference → intervention → independent recovery verification.\n\nTemporal corroboration compares sensor deltas, not absolute physical health.\n\nFor each sensor: delta_i = residual_current_i - residual_previous_i.\n\nThe cohort consensus delta is the deterministic median of trusted sensor deltas.\n\nA sensor is temporally inconsistent when abs(delta_i - consensus_delta) exceeds maximum_delta_disagreement.\n\nThe policy threshold is expressed in milli-residual units for deterministic integer comparison after normalization.\n\n## What this distinguishes\n\n### Common physical change\n\nIf three independently qualified sensors move approximately:\n\n- 0.5 → 1.5\n- 0.6 → 1.6\n- 0.4 → 1.4\n\nthe temporal deltas are all approximately +1.0. The result is Corroborated.\n\nThis does not mean the structure is healthy. It means the sensors consistently report a changing state. Downstream structural-health logic must interpret the consensus residual.\n\n### Stuck sensor during physical change\n\nIf two sensors move by approximately +1.0 but a third remains unchanged, the third sensor's delta departs materially from the cohort. The result is Conflicted.\n\n### Gradual drift during physical change\n\nIf two sensors move by +1.0 while another moves by only +0.5, the deviation is exposed even though the drifting sensor may still be below its individual absolute residual threshold.\n\n## Safety properties\n\n- Sensor identity must remain stable across the pair.\n- Current timestamps must be strictly later than previous timestamps.\n- Only sensors trusted at both observations participate in temporal corroboration.\n- Duplicate sensor identities are conflicting evidence.\n- Insufficient trusted sensors never become corroborated by extrapolation.\n- Temporal disagreement remains visible; it is never averaged away.\n- A corroborated physical transition is not itself a recovery verdict.\n- Temporal corroboration does not authorize repair or operation.\n\n
## Topology attestation boundary

Independence is now bound to a structured sensor-topology attestation rather than an unqualified digest string. The attestation binds:

- asset identity;
- component identity;
- topology identity and version;
- topology/dependency digest;
- configuration digest;
- issuance and expiry timestamps; and
- provenance evidence identity.

A temporal sensor pair may contribute to quorum only when the attestation is valid at both the previous and current observation timestamps and its asset, component, and configuration identities agree with the binding. Stale, future-dated, substituted, or configuration-incompatible attestations are excluded and surfaced as explicit evidence-quality faults.

This is deliberately a validity boundary, not a trust oracle. A syntactically and temporally valid attestation does not prove that the declared physical topology is truthful or that an issuer is honest. That stronger property belongs to the provenance/attestation layer. Recent remote-attestation work emphasizes continuous verification rather than treating an initial trust state as permanently sufficient.

The resulting separation is:

**topology declaration → topology attestation validity → independence quorum → temporal corroboration → physics/model evidence.**

The local contract therefore remains deterministic and dependency-light while leaving room for a later Mycelix-backed authoritative topology attestation.

## Authoritative attestation-reference boundary

The topology attestation now carries a structured reference into the authoritative provenance layer:

- attestation identity;
- issuer identity;
- attestation statement digest; and
- verification-result/evidence reference.

The local contract validates reference completeness and expected issuer identity, but does **not** perform or claim cryptographic verification. This is an intentional boundary: Symthaea consumes an attestation reference and enforces its local consistency, while Mycelix (or another configured authority) remains responsible for authoritative provenance and verification.

The reference is also time-bounded through the enclosing topology attestation. Future-dated and expired attestations cannot qualify a temporal pair. Configuration or topology changes therefore require a compatible attestation reference rather than silently reusing an unrelated declaration.

This follows the broader RATS separation between attesters, verifiers, and attestation results: the sensing domain should consume the result/reference rather than impersonate the authoritative verifier.

The resulting evidence chain is:

**sensor observation → sensor qualification → topology declaration → authoritative attestation reference → local validity → independence quorum → temporal corroboration → physics/model evidence.**

A valid reference is still not physical truth. It establishes an auditable trust-layer dependency, not a self-issued claim of correctness.

Within one temporal corroboration decision, admitted sensor pairs must also resolve to the same authoritative topology attestation reference. This prevents multiple independently well-formed references from silently creating a quorum over different topology statements.

## Verification-result boundary

The authoritative reference is now paired with an explicit verification result. The result binds:

- the verifier identity;
- the exact attestation ID and statement digest evaluated;
- the verification-result reference; and
- a validity interval for the verification result.

The local sensing contract checks these bindings and the result's temporal validity. It still does **not** perform cryptographic verification or establish the verifier's trust anchor. That remains the responsibility of the authoritative provenance layer.

This distinction prevents a particularly dangerous failure mode: a sensor can no longer become locally admissible merely by naming an attestation reference; the temporal gate also requires a coherent, time-valid verification result for that exact reference.

The intended trust progression is therefore:

**attestation reference → authoritative verification result → local temporal validity → sensor independence quorum.**

A verification result is evidence about the topology statement, not evidence that the physical asset itself is healthy.

## Adversarial boundary\n\nThis layer improves resilience against isolated stuck/drifting channels, but it cannot solve a coordinated majority attack by itself. If a quorum of sensors is compromised and reports a coherent false trajectory, temporal agreement alone cannot distinguish that trajectory from a genuine physical transition.\n\nThat case requires additional independent evidence such as physics/model residuals, heterogeneous sensing modalities, provenance and authorization controls, independent verification, or stronger Byzantine/fault assumptions.\n\n## Research basis\n\nRecent 2026 structural-health-monitoring research explicitly models sensor faults alongside evolving structural state and identifies bias, drift, gain variation, saturation, dropout, and stuck signals as distinct sensing degradations. It also emphasizes that structural degradation and sensor faults can otherwise produce ambiguous measurement anomalies. The present contract adopts that architectural distinction while keeping the core decision deterministic and platform-neutral.\n\n## Non-goals\n\nThis contract does not:\n\n- diagnose the physical failure mechanism;\n- declare a vehicle safe;\n- certify a repair;\n- replace regulated engineering authority;\n- infer physical recovery from sensor agreement;\n- assume that majority agreement is proof of truth.\n\nThe strongest invariant remains:\n\n**agreement is evidence quality, not physical truth.**

## Topology epoch continuity

Topology identity is now also bound to an explicit lifecycle epoch. The epoch binding carries:

- a monotonic epoch number;
- the predecessor topology digest for epochs after the initial epoch;
- the lifecycle time at which the epoch became effective; and
- a stable lifecycle event identifier.

The temporal policy pins the currently admissible topology epoch in addition to the topology ID, version, and digest. This makes a rollback to an older topology statement, a forked digest under the expected epoch, or an otherwise malformed epoch binding fail closed at the sensing boundary.

This is deliberately a **lifecycle continuity boundary**, not a claim that the local policy itself is authoritative. A legitimate topology/configuration change should create a new authoritative topology statement and advance the policy/reference to that epoch. The local gate therefore avoids silently treating an old attestation as current merely because its timestamp has not yet expired.

The distinction matters for offline and distributed systems: freshness alone does not establish that a statement is the *current* lifecycle state. RFC 9334 likewise treats freshness as a separate architectural concern and identifies replay, delay, reordering, and freezing participants on a past epoch as threats that must be handled by the attestation design. citeturn1search0

The intended trust progression is now:

**attestation reference → authoritative verification result → topology epoch continuity → local temporal validity → independence quorum.**

The epoch remains evidence about topology lifecycle state. It does not establish physical health, safety, repair success, or recovery.



## Distributed topology lifecycle convergence

Epoch continuity prevents an individual observation from silently rolling back to an older topology statement, but distributed/offline operation creates a second problem: two actors can independently produce successors from the same admitted predecessor.

The lifecycle boundary therefore treats topology transitions as append-only statements:

**predecessor epoch + predecessor digest → lifecycle event → successor epoch + successor digest**

The local convergence gate pins the predecessor that is currently admissible and requires the successor to advance exactly one epoch. A successor from an unrelated predecessor, an epoch skip, or an epoch rollback is quarantined.

Most importantly, two distinct successor digests for the same predecessor are a **concurrent successor fork**. The gate reports Conflicted; it does not select the newer timestamp, larger epoch, first-arriving statement, or any other local winner.

Duplicate copies of the same successor are not a fork. They may be recorded as a duplicate-evidence condition while retaining the same lifecycle interpretation.

This distinction is important for offline-first assets. Network reconnection can produce multiple validly formed lifecycle statements that cannot safely be reconciled by timestamp alone. The safe local behavior is to preserve the competing branches and require an authoritative lifecycle resolution before one branch becomes current.

The boundary therefore separates four questions:

1. **Validity:** is this transition structurally well formed?
2. **Continuity:** does it descend directly from the currently admitted predecessor?
3. **Convergence:** do all observed statements agree on one successor?
4. **Authority:** which lifecycle statement is authoritative when distributed actors disagree?

Only the first three are addressed here. Authority remains outside the local sensing contract.

This also aligns with the RATS freshness model: freshness is not the same thing as current lifecycle state, and distributed epoch mechanisms must account for propagation races and stale/frozen participants. Epoch Markers explicitly provide a shared freshness mechanism without requiring every participant to trust its local clock. citeturn0search0turn0search1

### Security invariant

**Never silently merge or choose between competing topology successors.**

A fork is evidence of unresolved lifecycle divergence, not evidence that either branch is correct.

