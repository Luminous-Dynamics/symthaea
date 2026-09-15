# Scientific View Coherent Capture + Cross-Root Closure v1

Status: SCI-014 architecture child of #3341 and prerequisite for durable scientific high-water in #3279 / #3280.

Related issue: #3345.

## 1. Purpose

A complete scientific authority roster is necessary but not sufficient.

Individually valid mutable heads may be read at incompatible moments and describe a scientific world state that never existed. Even if those heads did coexist, their cross-references may be semantically inconsistent.

V1 therefore freezes two separate theorems:

```text
COHERENCE
Did these exact mutable authority occurrences coexist?

CLOSURE
Do those coherent occurrences and their immutable dependencies form one valid scientific view under the exact active profile?
```

Neither theorem establishes live currentness forever, scientific support, proposition truth, or action authority.

## 2. Core non-equivalences

Freeze:

```text
individually valid/current-looking roots
    != coherent scientific authority vector

coherent authority vector
    != cross-root closure

closed historical scientific view
    != selected current scientific view

small timestamp skew
    != coherence

same friendly generations
    != same scientific state
```

## 3. Capture starts from the active control plane

The capture owner first resolves the exact SCI-014P control-plane head.

That head binds:

```text
active semantic-profile identity
active deployment-binding identity
canonical required mutable-role roster
exact source/provisioning identities
allowed capture theorem(s)
closure policy/profile
```

The caller does not choose a subset of roots and does not supply an alternate roster.

If the control-plane head changes during capture, the capture attempt cannot silently continue under the old roster.

## 4. Coherent observation record

A successful coherence theorem may produce a serializable historical record such as:

```text
CoherentScientificAuthorityObservationV1
```

binding at least:

```text
exact control-plane head
semantic-profile ID
deployment-binding ID
canonical role-roster commitment
exact source occurrence per required role
capture theorem/profile ID
capture evidence / attempt identity
observation commitment
```

This object means only:

> these exact authority occurrences were proven to coexist under this exact capture theorem.

It is not a live currentness capability.

## 5. V1 theorem A: owner-controlled atomic capture

Prefer:

```text
OwnerAtomicCaptureV1
```

when one qualified owner/store can atomically expose the complete exact control-plane + required-role vector.

A valid atomic capture should bind one exact immutable transaction/snapshot identity and every required role occurrence derived from that same snapshot.

A public store/transport interface claiming `atomic = true` is not sufficient authority.

The exact capture provider must be admitted/qualified under the deployment/profile boundary that owns the scientific view.

## 6. V1 theorem B: stabilized double collect

Where no atomic multi-root snapshot exists, permit:

```text
StabilizedDoubleCollectV1
```

only when every required mutable source satisfies the theorem prerequisites.

For each source role, the qualified provider must establish:

1. a linearizable read of its current exact occurrence/head;
2. immutable anti-ABA occurrence identity;
3. no scientifically material mutable state change without occurrence/head advancement;
4. exact source/provisioning identity in every observation;
5. explicit unavailable/error semantics instead of cached success fallback;
6. bounded observation behavior compatible with the capture profile.

If any required source cannot establish those properties, this capture theorem is unavailable for the profile.

## 7. Double-collect algorithm

Use a canonical required-role order.

Conceptually:

```text
read exact active control-plane C_A
derive exact roster R from C_A

SCAN A:
    read every role occurrence in canonical order
all A reads complete

read exact active control-plane C_B
require C_B == C_A before accepting the same roster

SCAN B:
    read every role occurrence in the same canonical order

require A(role) == B(role) for every role
require exact control-plane occurrence unchanged
```

All first-scan role reads finish before the second-scan role phase begins.

With linearizable reads plus anti-ABA exact occurrence identity, equality proves each source remained at its observed occurrence across the inter-scan boundary. Therefore the complete vector existed simultaneously at that boundary.

## 8. Why anti-ABA identity is mandatory

A mutable source may evolve:

```text
O17 -> O18 -> O19
```

while O19 happens to contain the same visible state bytes as O17.

Therefore:

```text
A.state_bytes == B.state_bytes
    != A.occurrence == B.occurrence
```

SCI-014P provisioning/source occurrence identity provides the stronger comparison.

A change away and back to equal content must fail exact double-collect equality.

## 9. Control-plane changes during capture

If:

```text
C_A = profile P3 / binding B8
C_B = profile P4 / binding B9
```

then the capture is not a valid P3/P4 mixed vector.

It fails or retries from the new exact control plane.

Do not compare role vectors derived from different source rosters as though they were one scientific state.

## 10. Bounded attempts

Capture under sustained churn must not retry forever.

The exact capture profile should commit a finite attempt/work bound.

After the bound is exhausted, return an explicit result such as:

```text
ConcurrentAuthorityMutation
WorldDidNotStabilize
```

Do not silently select:

- the most common vector;
- the highest sequence;
- the latest timestamp;
- the nearest snapshot;
- a partial roster.

## 11. Timestamp proximity is not coherence

Reject a theorem such as:

```text
all roots observed within N milliseconds
    -> coherent snapshot
```

Wall-clock proximity does not prove that the vector existed simultaneously and does not solve interleaving writes.

Trusted timestamps may remain audit metadata, but they are not the coherence primitive.

## 12. Coherence and closure remain separate

Once a coherent vector exists, SCI-014 separately verifies cross-root closure.

Conceptually:

```text
CoherentScientificAuthorityObservationV1
+ exact immutable dependencies
+ active profile closure policy
+ owner-qualified role/domain closure verifiers
        ↓
QualifiedHistoricalScientificViewCaptureV1
```

A stable coherent vector can still fail closure.

## 13. Closure theorem

Closure verifies every scientifically material relationship required by the active profile.

Examples, when the relevant roles exist, include:

```text
selected view head
    agrees with effective selection-decision lifecycle

evidence admission
    names exact research / execution / view identities

subject selection
    matches exact subject/view scope

verifier policy references
    match the exact admitted verifier-policy head

discovery/completeness state
    matches the active profile's discovery scope

immutable execution/result/artifact dependencies
    resolve and validate under exact content/lineage identity
```

A required reference to another deployment/profile/source occurrence fails closure.

## 14. Domain semantics remain domain-owned

SCI-014 composes role-specific validation; it does not duplicate every domain's semantics.

For example:

```text
ResearchSemanticHead
```

should eventually be validated by the research-lineage owner from #1946.

SCI-014's closure theorem consumes that validated identity and verifies cross-root agreement.

The same ownership rule applies to admissions, verifier policy, subject selection, discovery/completeness, and other future roles.

## 15. Hidden mutable dependency rule

Freeze:

```text
positive scientific use requires dependency D
+ D is mutable
+ D is absent from active profile roster
    -> closure/currentness unavailable
```

Scientific evaluators must not consult hidden ambient mutable state after capture.

If a newly discovered mutable dependency is scientifically material, the semantic profile/role roster must change explicitly.

## 16. Immutable dependencies remain explicit

Exact immutable dependencies may stay outside the mutable-role roster, but they remain named and verified by closure.

Examples may include:

```text
execution receipts
historical admissions
candidate/result manifests
content-addressed artifacts
exact research events already fixed by lineage occurrence
```

Missing or invalid immutable dependencies produce closure failure, not a fabricated mutable-currentness coordinate.

## 17. Closure policy versioning

Changing the set of scientifically material cross-root relationships changes the semantic profile/closure-policy identity.

A stronger new closure profile does not rewrite an older historical capture as fraudulent.

Instead:

```text
historical view valid under old closure profile
    != automatically satisfies stronger new profile
```

## 18. Owner-qualified closure path

Avoid downstream-mintable authority such as:

```text
trait ScientificClosureVerifier {
    fn is_qualified(&self) -> bool;
}
```

Pluggable parsers, transports, and non-authorizing validators may exist below the boundary.

Only the owner/control-plane-authorized closure path may mint the positive closed-view capability.

## 19. Failure taxonomy

Preserve distinct failure classes such as:

```text
MissingRequiredRole
UnknownRoleSemanticRevision
SourceUnavailable
ControlPlaneChangedDuringCapture
ConcurrentAuthorityMutation
WorldDidNotStabilize
SourceOccurrenceRollbackOrABA
CoherenceProviderUnqualified
ImmutableDependencyMissing
ImmutableDependencyInvalid
CrossRootReferenceMismatch
ClosurePolicyMismatch
ClosureUnavailable
```

Do not reduce these to:

```text
not current
```

and never reinterpret them as scientific opposition.

## 20. Relationship to scientific view lineage

A closed coherent capture is an immutable historical basis for a view-state transition/head.

The exact transition/head must bind the exact closed capture identity.

A later selected scientific-view head must not be paired after the fact with a different capture that merely looks compatible.

Freeze:

```text
same visible roots
    != interchangeable capture provenance
```

## 21. Relationship to SCI-014A high-water

SCI-014A should construct/persist a scientific high-water only from a precisely bound closed coherent authority/view state.

Recommended chain:

```text
active control-plane/profile/binding
        ↓
complete exact mutable-role roster
        ↓
coherent capture
        ↓
cross-root closure
        ↓
QualifiedHistoricalScientificViewCaptureV1
        ↓
view-state transition / selected head
        ↓
scientific authority vector
        ↓
local durable high-water
        ↓
external witness / freshness classification
```

The high-water must retain enough identity to prove which exact closed capture/view head it protects.

## 22. Coherent capture is still historical

Even a perfectly coherent and closed capture does not prove no successor exists.

Keep the later stages explicit:

```text
closed coherent historical view
    != terminal in supplied lineage
    != externally complete/current-through-checkpoint
    != point-of-use current
```

Those currentness theorems remain downstream.

## 23. Pure historical replay does not need present currentness

Historical inspection/reanalysis may consume an exact retained closed capture directly, subject to the requested historical-use policy.

Present-tense scientific claims require the downstream currentness/high-water path.

This prevents unnecessary global/current-source dependencies for deterministic historical replay.

## 24. Adversarial qualification

A future implementation should prove at least:

1. individually valid heads assembled from states that never coexisted cannot mint a coherent capture;
2. exact two-pass occurrence equality establishes one coherent boundary vector;
3. source change away/back to equal bytes with a new occurrence is detected;
4. control-plane/profile change between scans prevents cross-profile capture;
5. omitted required role prevents capture;
6. tiny timestamp skew provides no coherence credit;
7. stable coherent vector with selection decision bound to another view head fails closure;
8. missing immutable dependency fails closure;
9. repeated churn exhausts bounded attempts rather than spinning forever;
10. downstream/unqualified capture implementation cannot mint positive coherent/closed authority;
11. historical closed capture remains replayable after later current heads advance;
12. capture/closure artifacts grant no scientific support, truth, recommendation, governance, medical/resource, deployment, or physical-effect authority.

## 25. Non-claims

This architecture does not implement:

- the domain-specific mutable authority sources;
- #1946 research lineage;
- durable high-water persistence;
- external witness selection;
- global evidence completeness;
- scientific truth;
- action authority.

It defines the exact coherence and closure theorem those layers must consume.