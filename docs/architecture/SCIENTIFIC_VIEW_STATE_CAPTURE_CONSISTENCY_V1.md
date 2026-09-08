# Scientific View State Capture Consistency v1

Status: hardening companion to `SCIENTIFIC_VIEW_STATE_LINEAGE_V1.md`.

## 1. Purpose

A scientific-view state snapshot may bind exact identities for every required registry/root and still fail to represent one coherent scientific state.

Example:

```text
proposition root       read at state A
evidence root          read after state B
lifecycle root         read after state C
argument root          read before state D
```

Every root can be individually valid while their assembled combination was never simultaneously admitted by the scientific view.

Freeze:

```text
exact root identities
    != coherent cross-root snapshot
```

and:

```text
all roots individually current when fetched
    != root set atomically/currently coherent
```

Currentness needs a view-state capture/consistency theorem in addition to root identity and lineage.

## 2. Mixed-root state is not a harmless implementation detail

Cross-root inconsistency can create scientifically impossible or authority-distorting states.

Examples:

- evidence contribution E exists in the evidence registry but its proposition target is from an older proposition registry root;
- a source retraction is visible in lifecycle state while the evidence-view root still reflects a pre-retraction admission result;
- a new dependency edge is visible while triangulation state remains from before that edge existed;
- a disposition policy root is newer than the predicate/evaluator registry it assumes;
- an argument graph references an evidence contribution absent from the captured evidence root;
- an equivalence receipt makes evidence newly relevant while the evidence-view/disposition slice remains pre-equivalence.

A deterministic hash over that root vector would faithfully identify an incoherent state rather than make it coherent.

## 3. State capture profile

A scientific-view profile should declare how coherent root sets are captured.

Conceptually:

```text
ScientificViewStateCaptureProfileV1
```

Candidate profile families may include:

```text
AtomicSnapshot
QualifiedConsistencyFrontier
```

The exact production enum is deferred.

Do not collapse these into a generic `consistent = true` flag.

## 4. Atomic snapshot profile

For a view whose required registries can be read under one owner-controlled transactional snapshot, prefer the simplest theorem:

```text
one immutable snapshot token/revision
    -> resolve every required root
    -> canonical root set
```

Conceptually the capture receipt binds:

```text
view namespace/profile/epoch
capture profile
owner snapshot token/revision
complete canonical root set
capture implementation artifact
capture execution lineage
```

All required roots must resolve from the same immutable storage snapshot or equivalent atomic read boundary.

A caller-supplied list of root digests plus a claimed snapshot number cannot mint the capability.

## 5. Federated/distributed views may need a consistency frontier

A federated scientific view may intentionally aggregate multiple independently versioned registries for which no single global atomic database transaction exists.

Do not pretend they share one instantaneous wall-clock state.

Instead define an explicit owner-qualified consistency frontier conceptually like:

```text
ScientificViewConsistencyFrontierV1
```

binding, as appropriate:

```text
view namespace/profile/epoch
per-registry exact root/head identity
per-registry lineage position
cross-registry dependency constraints
information/availability cutoff semantics
causal/checkpoint dependencies
frontier profile identity
verification implementation/execution lineage
```

The frontier means only that the selected roots form an admissible coherent view under that exact profile.

It does not imply universal distributed consensus or synchronized physical time.

## 6. Root set must be closed under required cross-root references

A qualified view-state capture must reject a root set when one registry contains a scientifically material reference that cannot be resolved under the captured state/profile.

Examples include:

```text
evidence -> proposition
lifecycle event -> evidence contribution
argument edge -> evidence/proposition/reason object
dependency edge -> lineage/evidence object
triangulation receipt -> compatibility/dependency state
policy -> predicate/evaluator profile
predicate derivation -> reason/evidence state
```

The exact reference graph is profile-defined.

Freeze:

```text
referenced object exists somewhere
    != referenced object is admitted in this captured view state
```

## 7. Forward-reference discipline

A captured root must not depend on a scientifically material object that appears only in a later root/frontier position unless the view profile explicitly models that reference as a foreign/pending assertion rather than current local state.

For ordinary current-state qualification:

```text
root A references object X
+ X only appears after captured frontier
    -> incoherent capture
```

This prevents hidden future-state leakage into historical or current snapshots.

## 8. Historical information cutoff and capture frontier

Historical replay introduces another boundary:

```text
logical/effective event time
    != event availability time
    != view capture time
```

A historical scientific-view state at cutoff `t0` may include only objects admitted by the view's historical-availability semantics at `t0`.

A later backdated event must not enter the older view merely because its effective timestamp is earlier.

The capture/frontier receipt therefore needs to bind the exact information-cutoff semantics already required by #783/#808.

## 9. Capture receipt is verifier-owned capability

Portable state metadata may be serializable for audit:

```text
PersistedScientificViewStateSnapshotV1
```

but a positive coherence object should be verifier-owned, conceptually:

```text
QualifiedScientificViewStateCaptureV1
```

with:

- private construction;
- no public raw constructor;
- no Deserialize reconstruction into positive capability;
- exact view namespace/profile/epoch;
- exact canonical root set;
- exact capture/frontier identity;
- capture profile identity;
- verification execution lineage.

A persisted root bag cannot regain coherence authority merely by deserialization.

## 10. Capture coherence vs selected live head

Even a qualified coherent state capture does not establish that the view owner selected it as the live head.

Preserve:

```text
QualifiedScientificViewStateCaptureV1
    != QualifiedScientificViewHeadV1
```

The head-selection/lineage layer must still prove exact transition history and current selected head under the declared view policy.

Likewise:

```text
valid transition referencing a coherent snapshot
    != selected live transition
```

## 11. Selected head should commit the exact capture identity

A `ScientificViewStateTransitionV1` should ultimately bind the exact qualified/canonical state capture material, not a caller-assembled root bag.

Conceptually:

```text
QualifiedScientificViewStateCaptureV1
        ↓ canonical snapshot identity
ScientificViewStateTransitionV1
        ↓ exact predecessor lineage
QualifiedScientificViewHeadV1
```

The lower serializable transition may contain only the canonical capture/snapshot identity, while the privileged head verifier independently resolves and revalidates the referenced capture material.

## 12. Capture profile drift

Changing the capture semantics changes what one root vector means.

Examples:

- moving from atomic MVCC snapshot to causal-frontier semantics;
- changing cross-root closure requirements;
- changing historical-availability cutoff rules;
- changing mandatory registry roles;
- changing accepted pending/foreign-reference treatment.

Such changes require a new immutable capture/view profile identity and ordinarily a view epoch migration theorem.

Do not silently reinterpret old state snapshots under new consistency rules.

## 13. Split-brain registry reads

If an owner-local storage layer exposes two competing heads for one required registry, state capture must not choose one by:

```text
latest timestamp
highest sequence
first response
lowest latency
lexicographic digest
```

unless the exact view profile has independently qualified that resolution rule.

An unresolved component-registry fork means the composite view capture is unavailable/contested.

## 14. Partial availability

If one required registry/root cannot be obtained or verified, the view cannot mint a complete current-state capture merely from the remaining roots.

Return an explicit unavailable/incomplete result.

Optional roots are allowed only when the exact immutable view profile declares them optional and defines the semantics of absence.

Freeze:

```text
required registry unavailable
    != unchanged
    != empty
    != no evidence
```

## 15. Capture TOCTOU

Even a coherent capture can become stale relative to the live source registries immediately after capture.

That is acceptable when the consumer operates against the immutable captured snapshot/frontier.

It is not sufficient for live currentness after any required source head advances.

The later exact-head currentness verifier binds to the selected immutable view transition/head created from that capture.

The later #886 continuity theorem handles qualified preservation across newer heads.

## 16. Incremental capture is a later optimization

A future large Atlas may update one composite view root incrementally rather than reconstructing all registries on every transition.

That optimization must prove equivalence to a full coherent capture under the exact profile.

Freeze:

```text
incremental root update
    != coherent composite view by default
```

The incremental algorithm may reuse prior roots only when owner-qualified delta/impact semantics prove they remain valid under the destination frontier.

This should compose with #886's impact-frontier theorem rather than create a separate weaker path.

## 17. Relation to material continuity

There are two different incremental questions:

```text
A. Is H18 itself a coherent scientific-view state?
B. Did H17 -> H18 preserve the material dependencies of proposition P/use U?
```

This document owns A.

#886 owns B.

Material continuity must never compensate for an incoherent destination head.

Therefore:

```text
QualifiedMaterialContinuity(H17, H18)
    requires QualifiedScientificViewStateCapture(H18)
```

or an equivalent qualified destination-state theorem.

## 18. Qualification vectors

A first executable state-capture qualification should prove at minimum:

1. individually valid roots from incompatible capture revisions fail coherence;
2. missing required root fails;
3. duplicate root role fails;
4. root ordering does not change canonical identity;
5. evidence referencing a proposition absent from the captured proposition root fails;
6. lifecycle event targeting evidence absent from the captured evidence root fails;
7. dependency/triangulation state from inconsistent generations fails under the profile;
8. caller-supplied `consistent=true` cannot mint the capability;
9. caller-supplied snapshot token cannot substitute owner-resolved atomic state;
10. unresolved component-registry fork prevents positive capture;
11. required registry unavailability fails closed;
12. later/backdated information does not leak into an earlier historical cutoff;
13. same canonical roots under a different capture profile are not silently the same qualified capture;
14. coherent capture does not mint selected live-head authority;
15. foreign-view roots cannot silently enter a local root role without an explicit profile/bridge rule;
16. capture qualification never implies proposition truth, global evidence completeness, recommendation, governance, medical/resource, or physical-effect authority.

## 19. First implementation order

Prefer:

```text
1. canonical root-role schema/profile
2. synthetic AtomicSnapshot capture fixture
3. cross-root reference-closure validation
4. private QualifiedScientificViewStateCaptureV1
5. ScientificViewStateTransitionV1 binds capture identity
6. owner-selected QualifiedScientificViewHeadV1
7. federated consistency-frontier profile later
8. incremental capture later
```

Do not begin with distributed consensus.

## 20. Exit gate

The capture theorem is complete when the repository can truthfully state:

> A scientific-view state head cannot be built from an arbitrary bag of individually valid registry roots. The root set must be captured or proven coherent under one exact immutable view/capture profile, closed over every required cross-root scientific reference and information cutoff, before that state may enter view-head lineage or currentness verification.

This is a state-coherence theorem, not a truth theorem.
