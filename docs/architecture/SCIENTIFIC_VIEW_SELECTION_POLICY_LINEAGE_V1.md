# Scientific View Selection Policy Lineage v1

Status: hardening companion to `SCIENTIFIC_VIEW_HEAD_SELECTION_PROVENANCE_V1.md` under #908.

## 1. Purpose

#908 already freezes a critical non-circular rule:

```text
candidate head
    != policy that selects that candidate head
```

Bootstrap selection must come from an independently established source, and ordinary succession must use policy qualified before the candidate.

One authority boundary remains:

> How do we know that the pre-existing selection policy itself is the exact policy currently authorized for this view transition?

A policy can be historically valid yet no longer current. Its lineage can fork. A later policy can revoke, supersede, restrict, or migrate an older policy. A caller can possess an authentic old policy and replay it against a new candidate.

Freeze:

```text
valid selection policy
    != current selection policy

historically authorized policy
    != authorized policy for this transition

policy revision number
    != policy lineage identity
```

## 2. Core theorem

The scientific-view currentness chain must preserve:

```text
selection-policy profile
    != selection-policy artifact
    != selection-policy lineage transition
    != selected/current policy head
    != head-selection receipt
    != scientific-view head
    != scientific truth
```

A positive view-head selection must bind to one exact verifier-owned policy-currentness witness, not merely to a policy ID or digest supplied by the caller.

## 3. Selection policy is privileged currentness state

Selection policy does not decide scientific truth, but it does decide which structurally valid view transition becomes the live state of one declared view.

Therefore policy currentness is privileged administration/currentness state.

It must receive the same anti-equivocation discipline as the view heads it selects.

Do not permit:

```text
policy_id = "default"
revision = 7
```

to serve as sufficient current-policy identity.

Two different policy histories can share those labels.

## 4. Canonical selection-policy artifact

A policy artifact should have immutable semantic identity.

Conceptually:

```text
ScientificViewSelectionPolicyArtifactV1 {
    policy_profile_id,
    view_namespace_id,
    view_profile_id,
    epoch_scope,
    transition_classes,
    candidate_set_requirements,
    fork_resolution_semantics,
    bootstrap/succession/recovery/migration permissions,
    administrator/quorum semantics where applicable,
    validity/freshness semantics where applicable,
    other profile-declared material rules,
}
```

The exact production fields are deferred.

Friendly names do not define semantics.

Changing a scientifically/currentness-material rule creates a new artifact identity.

## 5. Policy lineage uses exact predecessor binding

Policy history must not be inferred from revision counters.

Use the generic anti-equivocation pattern:

```text
policy generation 1
    -> explicit Bootstrap

policy generation > 1
    -> Previous(exact predecessor policy-transition digest)
```

Conceptually:

```text
ScientificViewSelectionPolicyTransitionV1 {
    transition_profile_id,
    view_namespace_id,
    view_profile_id,
    policy_epoch_id,
    generation,
    predecessor,
    destination_policy_artifact_digest,
}
```

Require exact +1 generation inside ordinary policy succession, while preserving:

```text
generation equality
    != policy history equality
```

## 6. Policy artifact identity and policy lineage position differ

The same policy artifact may intentionally be reinstated later.

For example:

```text
P1 -> P2 -> P3(reinstates P1 semantics)
```

Even if:

```text
artifact(P3) == artifact(P1)
```

we require:

```text
transition(P3) != transition(P1)
```

An old current-policy witness cannot become current again merely because equivalent bytes reappear.

## 7. Policy currentness is verifier-owned

Portable records may retain historical policy transitions and selected labels.

Positive policy currentness should instead be an opaque verifier-owned capability conceptually like:

```text
QualifiedScientificViewSelectionPolicyHeadV1
```

It should bind at least:

```text
view namespace/profile/epoch
policy profile/epoch
exact selected policy transition
exact policy artifact
exact predecessor lineage
policy-head selection/currentness source
verification implementation/execution lineage
```

No raw constructor from:

```text
policy digest
revision
generation
timestamp
selected=true
```

## 8. Head selection consumes policy currentness capability

The future view-head resolver should consume:

```text
QualifiedScientificViewSelectionPolicyHeadV1
```

not:

```text
ScientificViewSelectionPolicyArtifactV1
```

and not a caller-selected enum/string/profile.

Conceptually:

```text
Qualified live scientific-view head H17
+ Qualified current selection-policy head P17
+ coherent candidate H18
+ closed candidate/fork context
    -> QualifiedScientificViewHeadSelectionV1
```

The selection receipt retains exact P17 identity.

## 9. Policy update is effective only after prior-policy authorization

Suppose H18 proposes policy P18.

The transition into H18 still uses P17.

Only after H18 is selected under P17 may P18 become eligible for the policy-currentness transition governing H18 -> H19.

Freeze:

```text
candidate contains P18
    != P18 currently authorized
```

and:

```text
P18 authored
    != P18 selected policy head
```

A separate policy-transition theorem must establish P17 -> P18.

## 10. Policy transition itself cannot self-authorize

Invalid:

```text
P18 says replacing P17 with P18 is allowed
    -> P18 becomes current
```

The authorization rule for the P17 -> P18 transition must come from P17 or another independently qualified pre-existing source.

This is the same anti-circularity theorem applied recursively at the policy layer.

## 11. Policy forks are first-class

Suppose:

```text
P17
├── P18a
└── P18b
```

Both may be structurally valid successors.

Neither is current merely because it is newer, observed first, or has a larger generation/digest.

Until an exact higher/pre-existing policy-currentness theorem resolves the fork:

```text
SelectionPolicyForkUnresolved
```

and positive scientific-view head selection must fail closed if it depends on that policy lineage.

## 12. Policy fork cannot be resolved by a branch-local policy

P18a cannot contain a rule saying P18a wins policy forks and use that rule to defeat P18b.

Likewise P18b cannot do the reverse.

Policy-fork resolution must derive from common prior authority or an independently provisioned higher/root policy source.

## 13. Candidate-set closure applies to policy forks too

A resolver that sees only one supplied policy successor has not proven uniqueness.

Require a qualified successor-set theorem for the exact policy lineage scope.

Freeze:

```text
one observed policy successor
    != only valid policy successor
```

An unresolved or incomplete policy-successor inventory yields Unknown/Unavailable, not currentness.

## 14. Revocation and supersession are not deletion

If P17 is superseded or revoked, retain its historical artifact and transition.

Historical replay may still need to prove that P17 correctly selected H18 at time t.

But:

```text
historically valid P17
    != currently usable P17
```

The policy-currentness witness must be time/view-state scoped.

## 15. Policy lifecycle and scientific-view state capture

If selection-policy state is represented inside the canonical scientific-view root set, distinguish:

```text
policy registry captured in candidate H18
```

from:

```text
policy currentness used to admit H18
```

The former is candidate destination state.

The latter must come from H17/pre-existing authority.

A coherent capture can prove what policy state H18 contains; it cannot make that policy retroactively authoritative for H18 admission.

## 16. Policy currentness must bind to exact view lineage

A policy head qualified for:

```text
namespace A / profile X / epoch 3
```

cannot select a head in:

```text
namespace B / profile X / epoch 3
```

or:

```text
namespace A / profile Y / epoch 3
```

without an explicit migration/equivalence theorem.

Friendly policy IDs shared across views do not bridge authority domains.

## 17. Policy epoch migration

Some policy changes are too semantic to be ordinary successors.

Changing, for example:

- administrator identity model;
- quorum semantics;
- fork-resolution semantics;
- policy artifact schema meaning;
- view namespace/profile scope;
- bootstrap root semantics;

may require a distinct policy epoch/migration theorem.

Do not hide a policy authority-domain migration behind `generation + 1`.

## 18. Bootstrap root and regress termination

Policy currentness cannot recurse forever.

Each deployed scientific view needs an explicit termination point for policy trust/currentness, such as an independently established bootstrap/provisioning source.

The shared scientific kernel should expose this root provenance rather than invent a generic `TrustedPolicyProvider` trait whose downstream implementation can promise trust.

Freeze:

```text
trait implementation says trusted
    != qualified policy root
```

## 19. Administrator/quorum changes

For views using multiple administrators, changing the administrator set or threshold is itself a policy-authority transition.

The new administrator set cannot authorize its own installation unless the previous/current policy explicitly authorizes that transition.

For example:

```text
old quorum = {A,B,C}, threshold 2
new quorum = {D}, threshold 1
```

must be authorized under the old qualified rules, not merely signed by D.

## 20. Revocation/currentness races

Suppose P17 is read as current, then revoked before H18 selection completes.

The selection pipeline needs point-of-use currentness semantics.

The first safe implementation should bind policy-head currentness and candidate selection to one immutable/atomic policy-view state or perform an atomic compare-and-use/revalidation before issuing the head-selection capability.

Freeze:

```text
policy current when read
    != policy current when used
```

## 21. TOCTOU capability binding

A successful head-selection receipt should retain the exact policy-currentness/head witness or its immutable identity.

If policy state advances before selection commits, fail stale/reprepare unless the operation was executed against an immutable qualified snapshot.

A later continuity optimization for policy currentness must be independently qualified; do not reuse #886 automatically because view-state continuity and policy-authority continuity are distinct propositions.

## 22. Historical replay

Historical Atlas replay should be able to reconstruct:

```text
which policy head was current
under which policy lineage/root
when H18 was selected
```

A policy revoked in 2030 must not make a 2028 selection historically nonexistent.

Likewise a policy introduced in 2030 cannot be used to reinterpret which branch was selected in 2028 unless a separate historical correction/adjudication theorem says so.

## 23. Policy availability

Knowing a historical policy digest is not sufficient if its semantics/lineage material cannot be recovered and reverified under the exact profile.

Distinguish:

```text
policy identity known
policy lineage known
policy material replayable
policy currently qualified
```

Unavailable required material blocks positive replay/currentness rather than becoming an implicit unchanged/default policy.

## 24. Cross-view/federated policy import

A policy head from View B remains foreign to View A.

Observation or import does not make it selection authority for A.

A bridge may later establish a scoped delegated/recognized relationship, but that requires its own exact authorization theorem.

Freeze:

```text
foreign qualified policy
    != local current policy
```

## 25. Learned policy proposals

A model/LLM may propose a selection-policy change or help diagnose forks.

It cannot directly mint policy-currentness.

The proposed policy must cross the same prior-authority transition boundary as any human-authored policy change.

## 26. Failure result space

A future policy-currentness resolver should preserve explicit outcomes, for example:

```text
QualifiedCurrentPolicyHead
UnknownPolicyRoot
PolicyProfileMismatch
PolicyEpochMismatch
UnsupportedPolicyTransition
PolicyLineageGap
PolicyPredecessorMismatch
PolicyGenerationMismatch
PolicyForkUnresolved
PolicyCandidateSetIncomplete
PolicyRevoked
PolicySuperseded
PolicySelectionSourceUnavailable
PolicyMaterialUnavailable
PolicyStaleAtUse
PolicyMigrationRequired
```

The exact enum is deferred.

Do not collapse these into `valid=false` if the distinction matters for audit/recovery.

## 27. Adversarial qualification vectors

A first executable policy-lineage/currentness qualification should prove at minimum:

1. policy generation 1 requires explicit bootstrap;
2. later ordinary policy transition requires exact predecessor digest;
3. generation skip fails;
4. two same-generation policy successors remain distinct forks;
5. same policy artifact reintroduced later has a different transition/head identity;
6. historically valid but superseded policy cannot select a new live head;
7. candidate H18 cannot choose a stale earlier policy that admits it more easily;
8. P18 cannot authorize its own P17 -> P18 transition;
9. policy introduced inside H18 cannot select H18;
10. policy introduced inside selected H18 may affect H19 only after a valid policy transition;
11. branch-local P18a cannot resolve P18a/P18b policy fork;
12. incomplete policy-successor candidate set cannot yield a current policy head;
13. latest timestamp/highest generation cannot implicitly resolve policy fork;
14. stored `policy_current=true` cannot deserialize into current-policy capability;
15. policy qualified for another namespace/profile/epoch cannot select local head;
16. administrator-set downgrade requires authorization under prior current policy;
17. policy revoked between read and use causes stale/fail-closed result;
18. unavailable required historical policy material prevents positive replay;
19. foreign qualified policy cannot become local current policy by import;
20. current selection policy never implies proposition truth, evidence superiority, recommendation, governance, resource, medical, or physical-effect authority.

## 28. Implementation order

Keep this behind the already queued replay/reconstruction gates.

After #825 and #839/#848 qualify, the currentness path should prefer:

```text
1. pure canonical selection-policy artifact
2. pure exact-predecessor policy transition
3. synthetic independently provisioned policy bootstrap root
4. private QualifiedScientificViewSelectionPolicyHeadV1
5. policy fork/revocation/TOCTOU negative qualification
6. #908 coherent view-state capture + exact view transition
7. head selection consumes current-policy capability
8. QualifiedScientificViewHeadV1
9. exact-head disposition currentness
10. #886 material continuity later
```

Do not begin with a universal policy governance engine.

## 29. Exit gate

This theorem is complete when the repository can truthfully state:

> Scientific-view head selection consumes one exact currently qualified selection-policy lineage. A historically valid, superseded, revoked, forked, foreign, caller-selected, or self-authorizing policy cannot mint a live head. Policy changes are themselves predecessor-authenticated transitions authorized under prior current policy or an independently established root, and point-of-use policy currentness is revalidated before head-selection authority is issued.

This is a view-currentness administration theorem, not a scientific truth theorem.
