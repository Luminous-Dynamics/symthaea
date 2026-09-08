# Scientific View State Lineage v1

Status: architecture candidate. This document refines the SCI-014 current-within-view boundary defined by #808 by specifying what an exact scientific-view state/head must mean before any currentness capability is implemented.

## 1. Purpose

#808 correctly requires current scientific disposition to bind to an exact namespace-scoped scientific-view state rather than caller-supplied matching generation identifiers.

That leaves a prior identity problem:

> What exactly is a scientific-view state head, and how does a verifier know which history it belongs to?

Generation numbers, timestamps, friendly registry names, and a bag of current root digests are not sufficient.

Two different state histories can share:

```text
view_name = "default"
generation = 17
```

and even two distinct histories can temporarily reach the same state payload.

Currentness therefore needs both:

```text
exact state identity
+
exact lineage identity
```

before it can mean anything.

## 2. Core theorem

Freeze:

```text
view namespace
    != view profile
    != view epoch
    != view-state snapshot
    != view-state transition
    != view-state head
    != owner-qualified live head
    != current scientific disposition
    != truth
```

and:

```text
generation equality
    != lineage equality

timestamp ordering
    != lineage authority

same state payload
    != same lineage position

same friendly registry names
    != same scientific-view state
```

A currentness verifier must never infer scientific-view identity from display metadata alone.

## 3. View namespace and profile

A scientific view is scoped by an exact namespace and immutable semantic profile.

Conceptually:

```text
ScientificViewNamespaceId
ScientificViewProfileId
```

The namespace answers:

> Which independently administered/federated scientific view is this?

The profile answers:

> Under which exact state/registry semantics is this view interpreted?

A profile should bind at least the schema/roles required to interpret the view's root set, including which registry classes are authoritative for this view and how state snapshots are canonicalized.

Friendly labels such as:

```text
"public-atlas"
"economics-view"
"institution-A"
```

may be retained for UX, but they do not establish namespace/profile identity.

## 4. View epoch

Ordinary state succession should not be forced to encode semantic-profile migration, namespace transfer, or root-policy replacement.

Define a view epoch as one continuous state lineage under one exact namespace/profile authority context.

Conceptually:

```text
ScientificViewEpochId
```

Within one ordinary epoch, the following remain invariant unless an explicitly qualified epoch-transition theorem says otherwise:

- view namespace identity;
- view profile identity;
- root-role schema/profile;
- lineage interpretation rules.

A profile migration is not an ordinary successor merely because it increments a generation number.

Freeze:

```text
ordinary successor
    != view-profile migration
    != namespace transfer
    != bootstrap of a new view epoch
```

## 5. Canonical state snapshot

The view state at one lineage position should be represented by a canonical material snapshot whose identity covers every registry/root required by the exact view profile.

Conceptually:

```text
ScientificViewStateSnapshotV1 {
    namespace_id,
    view_profile_id,
    epoch_id,
    canonical_root_set,
}
```

The canonical root set may include, according to the profile:

```text
proposition registry root
evidence/discovery registry root
source lifecycle root
argument/adjudication root
dependency graph root
compatibility/triangulation root
disposition-policy registry root
evaluator registry root
measurement/admission registry root
learned-prior/model registry root, when scientifically material
other explicitly profile-declared scientific roots
```

The exact production shape is deferred.

The key invariant is that the profile determines the complete required root-role set. A caller cannot omit a required root merely because it is inconvenient.

## 6. Canonical root-role set

The state snapshot must have one canonical representation.

Require conceptually:

```text
unique root role
+ exact content identity per role
+ deterministic role ordering
+ no unknown privileged role unless the profile admits it
+ no duplicate role aliases
```

Two snapshots with the same semantic root set in different container order should have the same canonical identity.

Two snapshots that differ in any required scientifically material root must have different state identities.

Freeze:

```text
same root count
    != same state

same display name
    != same root role

same generation number
    != same root material
```

## 7. State snapshot identity vs transition identity

State content and lineage position are different identities.

Conceptually:

```text
ScientificViewStateSnapshotDigest
ScientificViewStateTransitionDigest
```

The state snapshot digest identifies the canonical scientific root material.

The transition digest identifies the exact lineage-bearing transition that reached that snapshot.

This distinction is required because a later explicit rollback/recovery transition may intentionally point to root material equal to an earlier snapshot while remaining a new lineage event.

Freeze:

```text
same snapshot digest
    != same transition digest
```

## 8. Exact predecessor binding

Generation numbers do not authenticate history.

Ordinary state lineage therefore follows the same generic anti-equivocation pattern already used elsewhere in Symthaea:

```text
sequence 1
    -> explicit Bootstrap

sequence > 1
    -> Previous(exact predecessor transition digest)
```

Conceptually:

```text
ScientificViewStateTransitionV1 {
    transition_profile_id,
    namespace_id,
    view_profile_id,
    epoch_id,
    sequence,
    predecessor,
    state_snapshot_digest,
}
```

The transition's content identity must cover the predecessor relation and the exact destination state snapshot.

A valid successor therefore proves which exact prior transition it descends from.

## 9. Sequence number is secondary metadata

A monotone sequence remains useful for diagnostics and exhaustion checks, but it is not history identity.

Require exact +1 succession inside ordinary lineage, but preserve:

```text
sequence == 18
    != unique history position
```

Two transitions may both claim sequence 18 while descending from the same sequence-17 predecessor and carrying different state snapshots.

That is an explicit fork, not two interchangeable representations of "generation 18".

## 10. Forks are first-class

Suppose:

```text
H17
├── H18a
└── H18b
```

Both successors can be structurally valid lineage transitions.

Structural validity does not choose which one is the live head.

Freeze:

```text
valid successor
    != selected live head

higher timestamp
    != fork resolution

lexicographically larger digest
    != fork resolution

majority of observers
    != universal scientific truth
```

If one scientific-view owner/profile permits fork resolution, that decision must come through an explicit qualified view-head-selection/fork-resolution policy.

Until then, live currentness for that view is unavailable/contested.

## 11. Owner-qualified live head

The currentness path must not accept:

```text
caller_supplied_head = H18a
```

and treat that as the live scientific state merely because H18a is structurally valid.

The privileged currentness path needs a verifier-owned result conceptually like:

```text
QualifiedScientificViewStateV1
```

or:

```text
QualifiedScientificViewHeadV1
```

created only after an owner-local/registered view-state resolver proves:

- exact namespace/profile/epoch;
- exact transition identity;
- exact canonical snapshot identity;
- exact predecessor/lineage validity;
- exact selected live-head policy for that view;
- no unresolved fork under that policy;
- source/currentness evidence required by the profile.

The exact storage mechanism may be pluggable below this boundary. The privileged live-head capability is not caller-mintable through a generic `TrustedProvider` promise.

## 12. View-head selection is not scientific truth selection

Even a fully qualified live head means only:

> this is the current selected state of this exact declared scientific view under this exact view-state policy.

It does not mean:

- globally complete literature;
- universal consensus;
- objectively true propositions;
- superiority over another qualified view;
- recommendation or action authority.

Multiple scientific views may legitimately select different current heads and disagree in content.

Cross-view comparison remains a separate Theory Atlas operation.

## 13. Rollback and reorganization

A view may need recovery, correction, rollback, or reorganization semantics.

Do not model rollback by mutating the live pointer to an old transition and pretending history rewound.

Prefer an explicit new lineage event that descends from the currently selected head and names the retained target state/recovery semantics.

Conceptually:

```text
H20 current
    -> RecoveryTransition(target_snapshot = snapshot(H17))
    -> H21
```

Even if:

```text
snapshot(H21) == snapshot(H17)
```

we still require:

```text
transition(H21) != transition(H17)
```

This prevents old currentness witnesses from becoming current again merely because root material or generation metadata reappears.

The exact rollback/recovery profile is deferred and should not be smuggled into ordinary successor semantics.

## 14. No generation reuse authority

A later branch may never resurrect an old witness through:

```text
generation = 17
```

or equivalent timestamps/labels.

Currentness binds to exact immutable transition/head identity and selected lineage.

Therefore:

```text
same generation
+ same snapshot payload
    != same live-head identity
```

unless an explicit qualified identity/equivalence theorem says so.

## 15. Epoch/profile migration

Changing the view's semantic profile can change the meaning of its root roles even when many underlying digests remain identical.

A future migration should therefore use an explicit epoch-transition receipt conceptually binding:

```text
source namespace/profile/epoch/head
+ destination namespace/profile/epoch/bootstrap head
+ migration profile
+ canonical root-role mapping
+ migration assumptions
+ verification evidence
```

The destination starts a distinct epoch lineage.

Old transitions remain historical objects in the old profile.

Freeze:

```text
profile version string changed
    != migration qualified

same root digests under new role semantics
    != same scientific-view state
```

## 16. Retention, pruning, and availability

A head identity may remain valid even if some referenced historical material later becomes unavailable, but that unavailability affects what can be verified/replayed.

Do not rewrite old transition/snapshot identities during compaction.

A storage system may retain checkpoints/commitments while pruning detail, but then the scientific layer must distinguish:

```text
identity known
history path known
material fully replayable
material unavailable
```

Unavailable historical material must not silently become currentness or opposition evidence.

If exact predecessor or required snapshot material cannot be verified under the view profile, positive live-head qualification fails closed.

## 17. Federation and foreign heads

A head from another scientific-view namespace remains foreign state by type/identity.

Importing or observing:

```text
View B head HB
```

inside View A does not make HB the current head of A.

Cross-view bridge/federation receipts may later establish relationships such as:

```text
View A observes View B head HB
View A imports selected evidence from HB
View A compares proposition dispositions with HB
```

without collapsing namespaces.

Freeze:

```text
foreign qualified head
    != local qualified head
```

and:

```text
same proposition IDs across views
    != same view state
```

unless proposition identity/profile equivalence has independently qualified.

## 18. State-head identity must be content-addressed, not label-addressed

Friendly IDs, sequence numbers, and timestamps are navigation metadata.

A durable transition/head identity should derive from canonical material including at least:

```text
transition profile identity
namespace identity
view profile identity
epoch identity
sequence
predecessor identity/bootstrap tag
state snapshot identity
```

The specific digest algorithm/profile should be explicit and evolvable.

A raw caller-provided digest is not authoritative merely because it has the expected length.

Verifier paths recompute or independently verify canonical identities.

## 19. State-profile drift

A profile label such as:

```text
scientific-view-v1
```

must not remain semantically mutable.

The actual view-profile definition should itself have immutable content identity.

Changing:

- required root roles;
- root-role semantics;
- canonical ordering;
- admissible optional roles;
- transition interpretation;
- fork-resolution semantics;
- rollback semantics;
- bootstrap semantics;

creates a new profile identity and ordinarily a new epoch/migration problem.

## 20. Interaction with exact-head currentness

Once full scientific replay exists, the simplest currentness theorem becomes:

```text
ReplayVerifiedScientificDispositionV1 {
    reconstructed against view head H
}
+
QualifiedScientificViewHeadV1 {
    selected live head H
}
    -> CurrentWithinScientificViewDispositionV1
```

Exact head equality means exact transition/head identity equality, not merely matching generation numbers or matching root-set digests.

If the selected live head is H+1, the old witness is stale for the live view by default.

#886 may later optimize this with independently qualified material continuity.

## 21. Interaction with material continuity

#886's future continuity optimization needs this lineage theorem underneath it.

A continuity proof from H17 to H18 requires:

- both heads qualify under the same exact namespace/profile/epoch or an explicit compatible transition theorem;
- H18 is a verified descendant of H17 under the selected view lineage;
- no unresolved fork or rollback is hidden;
- the ancestry/delta proof refers to exact transition identities, not sequence numbers alone.

Therefore:

```text
material continuity
    != currentness without lineage
```

## 22. Currentness result space

A future view-head/currentness resolver should prefer explicit outcomes rather than Boolean state.

Candidate distinctions include:

```text
QualifiedLiveHead
UnknownViewNamespace
ProfileMismatch
EpochMismatch
UnsupportedTransitionProfile
LineageGap
PredecessorMismatch
SequenceMismatch
ForkUnresolved
HeadSelectionUnavailable
HistoricalMaterialUnavailable
RollbackProfileRequired
MigrationProfileRequired
```

The exact enum is deferred.

Unknown/unavailable state must not become currentness.

## 23. Adversarial qualification vectors

A first executable view-state lineage qualification should prove at minimum:

1. sequence 1 requires explicit bootstrap;
2. non-initial ordinary transition requires exact predecessor digest;
3. sequence skip fails;
4. same sequence with different state snapshots produces distinct transition identities;
5. same state snapshot reached through different predecessor lineages remains distinct transition identity;
6. namespace change fails ordinary succession;
7. view-profile change fails ordinary succession;
8. epoch change fails ordinary succession;
9. missing required root role fails snapshot qualification;
10. duplicate root role fails canonical state qualification;
11. root container ordering does not change canonical snapshot identity;
12. caller-supplied matching generation/head labels cannot mint qualified live-head capability;
13. two valid successors of one predecessor create an unresolved fork unless owner policy resolves it;
14. latest timestamp cannot resolve the fork by default;
15. generation reuse cannot resurrect an old live witness;
16. rollback to old root material produces a new transition identity;
17. foreign-view head cannot satisfy local-view currentness;
18. unsupported profile/schema drift fails closed;
19. unavailable required lineage material prevents positive live-head verification;
20. qualified live head never implies truth, global consensus, recommendation, governance, resource, medical, or physical-effect authority.

## 24. First implementation order

Do not begin by implementing a distributed consensus system.

Preferred order:

```text
1. #825 disposition evaluation replay qualifies
2. #839/#848 full scientific reconstruction qualifies
3. pure canonical ScientificViewStateSnapshotV1
4. pure exact-predecessor ScientificViewStateTransitionV1
5. synthetic fork/rollback/profile-migration negative qualification
6. owner-local QualifiedScientificViewHeadV1 resolver
7. exact-head CurrentWithinScientificViewDispositionV1
8. TOCTOU point-of-use qualification
9. #886 material-continuity optimization later
```

This keeps identity/lineage semantics independent of storage/network implementation.

## 25. Exit gate

This architecture is complete when the repository can truthfully state:

> A scientific-view currentness witness binds to one exact namespace/profile/epoch, one canonical scientific-state snapshot, one exact predecessor-authenticated transition lineage, and one owner-qualified selected live head. Sequence numbers, timestamps, friendly labels, matching root bags, caller-selected heads, forks, rollback metadata, or foreign-view state cannot mint currentness by themselves.

This is a state-lineage/currentness theorem, not a truth theorem.
