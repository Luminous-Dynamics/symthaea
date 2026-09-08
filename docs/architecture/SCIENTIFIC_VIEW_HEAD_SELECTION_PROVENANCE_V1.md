# Scientific View Head Selection Provenance v1

Status: hardening companion to `SCIENTIFIC_VIEW_STATE_LINEAGE_V1.md` and `SCIENTIFIC_VIEW_STATE_CAPTURE_CONSISTENCY_V1.md`.

## 1. Purpose

A structurally valid, coherently captured scientific-view transition still does not establish that it is the selected live head.

The policy that selects/accepts the candidate head is therefore scientifically privileged currentness state.

That creates a circularity risk:

```text
candidate H18 contains permissive head-selection policy P18
        ↓
P18 says H18 is selected
        ↓
H18 becomes current
```

Freeze:

```text
candidate state
    != authority to choose how candidate state is selected
```

and:

```text
valid transition
    != selected live head
```

The selection policy used for a transition must predate, or be independently qualified from outside, the candidate authority it selects.

## 2. Selection authority is not proposition truth authority

This document uses "selection" only in the narrow current-view sense:

> Which exact structurally valid scientific-view transition is the current head of this exact declared view under its exact administration/lineage policy?

It does not select which proposition is true.

Freeze:

```text
view-head selection authority
    != scientific truth authority
    != evidence admission authority
    != recommendation authority
    != governance/effect authority
```

## 3. Bootstrap requires a pre-existing source

The first head of a scientific-view epoch has no predecessor head from which to derive selection policy.

Therefore bootstrap must consume an independently established bootstrap selection source.

Conceptually:

```text
QualifiedScientificViewBootstrapPolicyV1
        ↓
coherent bootstrap capture
        ↓
bootstrap transition
        ↓
QualifiedScientificViewHeadV1
```

The bootstrap candidate cannot mint or choose the bootstrap policy from its own state payload.

Invalid shape:

```text
bootstrap H1 contains bootstrap policy P1
P1 validates H1
H1 selects itself
```

The source may eventually be embedded configuration from a previously qualified release, a separately provisioned view policy, or another explicit owner-local policy source. Exact source families are deferred.

## 4. Ordinary succession uses predecessor-qualified policy

For ordinary succession:

```text
Qualified live head H17
        ↓
selection policy qualified/current at H17
        ↓
validate candidate successor H18
        ↓
select H18
```

The candidate H18 may contain policy-registry changes, but those changes do not gain authority to select H18 itself.

If H18 validly changes the head-selection policy under rules authorized by H17, the new policy becomes eligible only for later transitions such as H18 -> H19.

Freeze:

```text
policy introduced by candidate H18
    != policy that selects H18
```

## 5. Selection policy provenance must be explicit

A future owner-qualified selection capability should bind at least:

```text
view namespace/profile/epoch
selection policy profile identity
selection policy artifact/content identity
selection policy revision/lineage identity
source qualified head or bootstrap source
requested transition class
selection implementation artifact
selection execution lineage
```

Friendly policy names and caller-supplied `trusted=true` values are not authority.

## 6. Candidate cannot weaken its own selection requirements

A candidate may not choose a weaker profile under which it is evaluated.

Invalid:

```text
H18 candidate
    -> selects Development/PermissiveSelectionProfile
    -> passes
```

The applicable profile must be fixed by predecessor-qualified or independently provisioned policy before candidate evaluation.

This mirrors the broader repository rule:

```text
subject under evaluation
    != source of the assurance profile used to evaluate itself
```

## 7. Fork resolution must use common prior authority

Suppose:

```text
H17
├── H18a
└── H18b
```

Neither branch may use a policy introduced only inside itself to prove it wins the fork.

Fork resolution should derive from policy already qualified before the fork, conceptually:

```text
policy/currentness state at H17
    + exact candidate set {H18a, H18b}
    + exact fork-resolution profile
        -> resolved selected head
```

or fail closed to:

```text
ForkUnresolved
```

A branch-local later policy may affect subsequent descendants only after that branch has already been selected under non-circular prior authority.

## 8. Candidate set closure

A fork resolver is unsafe if the caller can hide a competing valid successor.

Therefore the resolver also needs an exact candidate-set/lineage-view theorem for the scope it claims to resolve.

Freeze:

```text
resolver sees one supplied candidate
    != only valid successor exists
```

For a closed owner-local transition store, the resolver may use an atomic/qualified successor index at the predecessor head.

For a federated view, an explicit bounded discovery/frontier theorem is required.

Without candidate-set closure, the result is incomplete/unknown rather than selected-by-default.

## 9. No latest-wins fallback

Do not use these as implicit fork authority:

```text
latest timestamp
highest sequence
highest wall-clock generation
lexicographically smallest/largest digest
first observed
lowest network latency
majority response
```

unless the exact pre-existing qualified view policy explicitly defines and qualifies such semantics for that view.

Even then the result is view-head selection, not truth.

## 10. Policy update transition

Changing head-selection policy is itself a privileged lineage event.

Prefer a transition theorem conceptually like:

```text
current qualified policy P17
+ proposed policy P18
+ exact authorized policy-transition rule
+ candidate H18
        ↓
H18 selected under P17
        ↓
P18 becomes effective for successors of H18
```

This prevents simultaneous self-authorization.

The exact policy-transition mechanism is deferred.

## 11. Profile/epoch migration

View-profile or epoch migration is even more sensitive.

The destination profile cannot define the rule that authorizes migration into itself unless an independent earlier source already authenticates that destination profile and migration rule.

Conceptually:

```text
qualified source-epoch head
+ source-qualified migration policy
+ independently identified destination profile
+ destination bootstrap capture
        -> qualified epoch-transition receipt
        -> destination qualified bootstrap head
```

Freeze:

```text
destination profile says migration is valid
    != source authorized migration
```

## 12. Recovery and rollback

Recovery must also avoid self-selection.

If H20 is current and a recovery transition creates H21 with snapshot material equal to old H17, the recovery policy must come from H20's already-qualified authority context or an independently established recovery policy.

Old H17 policy metadata does not automatically reactivate merely because the state payload is restored.

Likewise, H21 cannot select a new permissive recovery policy and use it retroactively to justify its own recovery.

## 13. Selection-policy root inside view state

A scientific view may include a policy registry root in its canonical state snapshot.

That is useful and may define the policy effective for *future* transitions.

But the current candidate snapshot's policy root must not be conflated with the selection policy used to admit that snapshot.

Distinguish conceptually:

```text
admission/selection policy for transition into H18
    = qualified predecessor/external policy

policy registry contained by H18
    = candidate future policy state
```

An exact transition receipt should preserve both identities when they differ.

## 14. Selection receipt

A successful head selection should produce an explicit verifier-owned receipt/capability conceptually like:

```text
QualifiedScientificViewHeadSelectionV1
```

binding:

```text
view namespace/profile/epoch
predecessor selected-head identity or bootstrap source
candidate transition identity
candidate coherent-capture identity
applicable pre-existing selection-policy identity
candidate-set/fork context identity
selection result
selection implementation/execution lineage
```

A later `QualifiedScientificViewHeadV1` may retain or compose this receipt.

No public constructor from candidate + strings/hashes.

## 15. Stored selection labels are not live authority

A persisted record may say:

```text
selected = true
```

for audit/history.

That remains ordinary data.

Positive runtime currentness requires replay/revalidation of the exact selection theorem against the qualified source policy and current selected lineage.

Freeze:

```text
stored selected=true
    != QualifiedScientificViewHeadV1
```

## 16. Selection policy currentness

The selection policy itself has lifecycle/currentness.

A historically valid policy can remain replayable after a newer policy takes effect.

Therefore:

```text
historically valid selection policy
    != current selection policy for live transition
```

The policy lineage/profile needs exact currentness semantics before it can be used to admit a new candidate.

## 17. Selection-policy equivocation

If the policy source itself forks or equivocates, head selection fails closed unless a higher/pre-existing policy layer resolves that fork.

Avoid infinite regress by terminating the trust chain in an explicit bootstrap/provisioning source whose authority is independently defined by the view deployment profile.

The scientific kernel should expose that root rather than pretending currentness can arise from pure self-reference.

## 18. Multiple administrators / quorum policies

Some scientific views may use threshold/quorum administration.

That is allowed only through an explicit qualified selection profile defining:

- administrator identity set/lineage;
- threshold semantics;
- signature/attestation profile;
- revocation/currentness;
- exact candidate transition subject;
- anti-replay domain separation.

Do not equate:

```text
more signatures
    == more scientific truth
```

Quorum proves only the configured view-head administration rule.

## 19. Learned/model-assisted selection

A model may assist with fork diagnostics or operator review, but model output must not become view-head selection authority unless the exact pre-existing selection policy explicitly defines and qualifies that role.

Default rule:

```text
LLM/HDC/model preference
    != live-head selection capability
```

This keeps learned science and administrative lineage separate.

## 20. Interaction with state capture

The candidate must first be coherently captured under the exact view/capture profile.

Selection never repairs incoherent candidate state.

Require:

```text
QualifiedScientificViewStateCaptureV1
    before
QualifiedScientificViewHeadSelectionV1
```

or an equivalent theorem.

## 21. Interaction with lineage transition

The candidate transition must already prove exact predecessor/bootstrap semantics and immutable namespace/profile/epoch rules for its transition class.

Selection does not make an invalid lineage transition valid.

Require:

```text
valid exact-predecessor transition
+ coherent candidate capture
+ pre-existing selection policy
    -> selected head
```

not:

```text
selection policy accepted candidate
    -> therefore transition valid
```

## 22. Interaction with #886 material continuity

Material continuity compares already qualified selected heads.

A continuity proof cannot be used to select a fork branch or bootstrap a new live head.

Freeze:

```text
QualifiedMaterialContinuity(H17,H18)
    != proof H18 is selected live head
```

Head selection precedes continuity/currentness optimization.

## 23. Historical replay

Historical Atlas reconstruction may replay which head was selected under the policy and candidate set available at time `t0`.

A later policy change or newly discovered fork must not silently rewrite the historical selection record.

Current live-head selection and historical selected-head reconstruction therefore use different cutoff/currentness contexts while preserving the same provenance theorem.

## 24. Qualification vectors

A first executable selection-provenance qualification should prove at minimum:

1. candidate cannot supply its own selection profile;
2. bootstrap candidate cannot self-provide bootstrap policy;
3. ordinary H18 is evaluated under policy already qualified at H17 or independent source;
4. policy introduced only in H18 cannot select H18;
5. policy introduced in selected H18 may be eligible for H19 if transition rules allow;
6. fork branch A cannot use branch-A-only policy to defeat branch B;
7. incomplete fork candidate set cannot yield positive resolved selection;
8. latest timestamp fails as implicit resolver;
9. highest generation fails as implicit resolver;
10. caller-supplied `selected=true` cannot mint live-head capability;
11. stored historical selection record cannot deserialize into live-head authority;
12. destination epoch/profile cannot self-authorize migration;
13. rollback candidate cannot choose its own recovery policy;
14. selection-policy fork fails closed without higher/pre-existing resolution;
15. learned/model preference cannot directly select a head;
16. coherent capture remains required before selection;
17. valid lineage transition remains required before selection;
18. selection receipt remains view-scoped and cannot satisfy another namespace/profile;
19. selected head never implies proposition truth/global consensus;
20. selected head never grants recommendation, governance, resource, medical, or physical-effect authority.

## 25. First implementation order

Prefer:

```text
1. pure canonical capture + transition validation
2. synthetic pre-existing bootstrap selection source
3. bootstrap head selection
4. predecessor-qualified ordinary successor selection
5. fork closure + fail-closed unresolved result
6. private QualifiedScientificViewHeadSelectionV1
7. compose QualifiedScientificViewHeadV1
8. exact-head disposition currentness
9. policy transition/migration/recovery profiles later
```

Do not begin with a generic pluggable `TrustedHeadSelector` trait that downstream code can implement to mint currentness by promise.

## 26. Exit gate

The selection-provenance theorem is complete when the repository can truthfully state:

> A candidate scientific-view head cannot choose the policy, assurance profile, fork resolver, migration rule, or recovery rule that makes itself current. Bootstrap selection comes from an independently established source; ordinary succession uses policy already qualified before the candidate; policy changes become effective only after admission under prior authority; and unresolved candidate-set/fork ambiguity fails closed.

This is a view-head currentness-administration theorem, not a scientific truth theorem.
