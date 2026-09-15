# Scientific View Profile Authority Roster v1

Status: SCI-014 architecture child of #908 and implementation prerequisite for #3279 / #3280.

Related issue: #3338.

## 1. Purpose

SCI-014 needs an exact answer to a deceptively simple question before durable currentness can be implemented:

> Which mutable authorities are scientifically material for this exact view, and which exact deployment-local sources currently satisfy those roles?

A high-water checkpoint cannot safely protect a scientific view until that question has a non-circular answer.

This document freezes a v1 authority surface that separates:

```text
immutable scientific-view semantics
    != active deployment authority
    != concrete mutable source occurrences
    != coherent current-state capture
    != durable high-water currentness
    != scientific support
    != truth
    != action authority
```

## 2. The circularity to avoid

The following design is invalid:

```text
candidate profile P2 defines required authorities
        ↓
read only the authorities P2 says matter
        ↓
those authorities say P2 is current
        ↓
activate P2
```

P2 could silently remove a material role, weaken capture semantics, change discovery scope, or otherwise redefine the theorem used to activate P2.

Freeze:

```text
candidate profile semantics
    != authority to activate candidate profile
```

and:

```text
profile-defined role roster
    != proof that the profile is the active profile
```

The active-profile decision must come from authority that predates the candidate profile.

## 3. Three objects with three owners

V1 keeps three concepts distinct.

### 3.1 `ScientificViewSemanticProfileV1`

An immutable, content-addressed definition of scientifically material view semantics.

It should bind at least:

```text
view namespace / profile family
profile schema / semantic revision
canonical mutable authority-role set
discovery-scope semantics
cross-root closure policy
permitted capture theorem(s)
requested-use policy family
role-specific semantic revisions
immutable-dependency treatment
profile commitment
```

A materially changed role set, closure rule, discovery universe, capture theorem, or use-policy semantics produces a new profile identity.

A semantic profile is ordinary persisted evidence. It is not current authority.

### 3.2 `ScientificViewDeploymentBindingV1`

A deployment-scoped binding from the immutable semantic profile's roles to exact authority-source identities.

It should bind at least:

```text
deployment identity
view namespace
semantic-profile identity
canonical required role set
per-role source logical identity
per-role provisioning/source epoch
per-role source qualification profile
capture-provider binding/profile
binding predecessor / lineage identity
binding commitment
```

A URL, database path, process name, hostname, friendly label, or matching data bytes are not source authority by themselves.

### 3.3 `ScientificViewControlPlaneHeadV1`

A tiny non-configurable administrative lineage that establishes which semantic profile and deployment binding are active.

This control plane sits outside the candidate profile's mutable scientific-role roster.

It is rooted in SCI-014's explicit deployment/bootstrap authority and remains administrative rather than scientific truth authority.

## 4. Minimal mandatory control plane

Every scientific view requires a small authority layer that no semantic profile may remove from itself.

V1 should bind at least:

```text
deployment/bootstrap trust-root lineage identity
active profile/binding lineage identity
exact control-plane predecessor/current head identity
```

The profile then defines the scientific mutable-role roster below this mandatory layer.

Freeze:

```text
control plane
    != scientific evidence
    != proposition truth
```

Its role is only to establish which declared scientific-view semantics and source binding are active.

## 5. Initial activation

The first profile/binding pair cannot self-provision.

Conceptually:

```text
DeploymentProvisioningSourceV1
+ immutable semantic profile P1
+ complete deployment binding B1
        ↓
ScientificViewControlPlaneHeadV1 C1(P1, B1)
```

Invalid:

```text
P1 says P1 is active
        ↓
P1 becomes active
```

or:

```text
B1's configured sources certify B1 as the authoritative binding
        ↓
B1 becomes active
```

The provisioning theorem exists outside the candidate profile/binding state.

## 6. Successor activation

Profile/binding changes use exact predecessor authority.

Conceptually:

```text
current C17(P3, B9)
+ transition policy already authorized under C17 / its prior bootstrap authority
+ candidate immutable profile P4
+ complete candidate binding B10
+ exact migration/activation evidence
        ↓
C18(P4, B10)
```

The candidate destination profile/binding cannot authorize the transition that installs itself.

Same sequence with different successor commitments is a control-plane fork/equivocation, not a latest-wins race.

## 7. Scientific authority roles are semantic

A role describes what scientific fact a mutable authority owns.

Candidate V1 roles include, as real implementations become available:

```text
ResearchSemanticHead
EvidenceAdmissionHead
SubjectSelectionHead
VerifierPolicyHead
ScientificViewSelectionLifecycleHead
DiscoveryCompletenessHead
```

The V1 role vocabulary should be narrow and owner-controlled.

Do not let arbitrary downstream crates add free-form role strings and thereby expand or weaken the scientific authority theorem without a profile semantic revision.

A role is not a store.

```text
same role
    != same source
```

## 8. Source logical identity is not source occurrence

A deployment may reuse the same logical source identity across restarts, reprovisioning, migration, or restoration.

Currentness therefore binds a stronger occurrence identity.

Conceptually:

```text
AuthoritySourceOccurrenceV1 {
    role,
    source_identity,
    provisioning_epoch,
    lineage_position,
    predecessor_occurrence,
    state_commitment,
    occurrence_commitment,
}
```

V1 requirements:

- exact deployment/view/role scope;
- explicit source logical identity;
- provisioning/source epoch as anti-ABA material;
- exact lineage position / current-head occurrence;
- exact predecessor occurrence where the source semantics provide one;
- exact state commitment;
- domain-separated occurrence commitment.

Freeze:

```text
same source logical identity
    != same source occurrence
```

and:

```text
same state bytes
+ new provisioning epoch
    != same authority occurrence
```

## 9. Anti-ABA source semantics

Suppose a source advances:

```text
S7/O17 -> S7/O18
```

and an old filesystem/database snapshot later restores bytes corresponding to O17.

The restored bytes do not revive O17 currentness.

Similarly, reprovisioning a source with identical visible state must create a new provisioning epoch/occurrence identity.

A positive capability bound to the old occurrence remains historical only.

## 10. Immutable dependencies stay outside the live roster

Scientifically material dependencies fall into two categories only:

```text
1. exact immutable dependency
2. authoritative mutable dependency represented by one active profile role
```

Examples of immutable dependencies may include:

- execution receipts;
- historical evidence admissions;
- result artifacts;
- candidate manifests;
- immutable scientific-view definitions;
- exact research events already named by a validated lineage occurrence.

Those are checked through exact identity / cross-root closure rather than treated as live mutable authority heads.

Freeze:

```text
scientifically material dependency
    != automatically a mutable currentness source
```

and:

```text
hidden ambient mutable scientific state
    -> currentness unavailable / closure failure
```

## 11. Roster completeness is profile-derived

The caller does not choose the source roster.

The current control-plane head chooses the active immutable profile and deployment binding; those determine the required canonical role set.

Example:

```text
P3 requires:
Research
Admission
VerifierPolicy
SelectionLifecycle
DiscoveryCompleteness
```

If the runtime supplies only four roles, the result is an explicit missing-role/currentness failure.

```text
coherent supplied subset
    != complete scientific authority roster
```

Unknown required role semantics fail closed.

## 12. Profile changes are scientifically material state

If P3 becomes P4 and P4 adds `DiscoveryCompletenessHead`, that is not a hidden implementation refactor.

It is a new scientific-view semantics identity.

Historical P3 views remain historically valid under P3.

They do not satisfy a P4 requested-use theorem merely because every overlapping coordinate is byte-identical.

Freeze:

```text
old profile historically valid
    != satisfies stronger new profile
```

## 13. Source binding changes need explicit migration

Moving a role from one authoritative source occurrence to another should use a predecessor-bound migration theorem.

Conceptually:

```text
QualifiedAuthoritySourceMigrationV1 {
    role,
    old_source_occurrence,
    new_source_identity,
    new_provisioning_epoch,
    migrated_state_commitment,
    migration_profile,
    migration_evidence,
    migration_commitment,
}
```

The migration is authorized under the prior active control-plane/binding state.

The destination source cannot self-authorize its own installation.

Historical views remain bound to the old source occurrence. New current views bind the new one.

## 14. Equal state bytes do not erase migration

A migration may intentionally reproduce identical logical state:

```text
old source state commitment == new source state commitment
```

That does not make the source occurrences interchangeable.

The new provisioning/source occurrence identity remains distinct and must appear in new current scientific-view/high-water state.

This makes store migration, restore, failover, and reprovisioning observable rather than silently ABA-equivalent.

## 15. Capture theorem is profile semantics, runtime qualification is separate

An immutable profile may require or allow capture semantics such as:

```text
OwnerAtomicCaptureV1
StabilizedDoubleCollectV1
```

But naming a capture algorithm is not proof that runtime prerequisites hold.

For example, a `StabilizedDoubleCollectV1` runtime provider must independently establish for every required mutable source role that the source offers the exact observation/anti-ABA semantics needed by that theorem.

Freeze:

```text
profile names capture theorem
    != runtime capture provider qualified
```

Downstream code must not mint scientific currentness by implementing a public semantic-promise trait such as:

```text
CurrentScientificAuthorityProvider
```

Pluggable transport can exist below the owner-qualified authority boundary.

## 16. Control-plane currentness and scientific-role currentness are distinct

A coherent scientific authority capture needs both:

```text
exact active control-plane head
+ complete current role occurrences required by that active profile/binding
```

The control-plane head answers:

> Which profile and source binding are active?

The profile role coordinates answer:

> What is current within those declared scientific mutable authorities?

Do not let either substitute for the other.

## 17. Profile/binding state belongs in the later high-water

SCI-014A's scientific high-water should eventually commit at least:

```text
control-plane head identity
active semantic-profile identity
active deployment-binding identity
complete canonical role roster commitment
exact source occurrence identities
selected scientific-view head
effective selection decision
predecessor high-water
```

This means rollback of the **definition of which authorities matter** is itself detectable.

An attacker cannot restore an older profile that omitted a later material role and then claim the smaller vector is current.

## 18. Relationship to #1946 research lineage

This contract does not implement research-semantic lineage.

When #1946 qualifies, `ResearchSemanticHead` should bind that research-owned currentness/lineage theorem rather than duplicating research events in SCI-014.

Until then, profile/roster semantics can be implemented without pretending the research role is already executable-qualified.

## 19. Relationship to evidence admission and ASSURE

The authority roster owns **current mutable scientific state**, not epistemic support.

Evidence admission remains historical evidence/provenance.

ASSURE remains the owner of claim-strength interpretation.

Freeze:

```text
current authority coordinate
    != evidence admission
    != claim support
```

## 20. Relationship to external high-water witnessing

External high-water publication should commit the control-plane/profile/binding/roster identities through the scientific high-water checkpoint.

It should not publish underlying scientific payloads merely to prove rollback resistance.

A fresh-machine verifier can then detect both:

- rollback of scientific mutable authority coordinates; and
- rollback to an older definition/binding of the authority surface itself.

## 21. Process-local positive authority remains separate

`ScientificViewSemanticProfileV1`, `ScientificViewDeploymentBindingV1`, source occurrence records, migration records, and control-plane records may all be serializable evidence.

None should deserialize into a live current-use capability.

Positive runtime currentness remains owner-local, non-Serde, and point-of-use qualified later in the SCI-014 chain.

## 22. Adversarial qualification

A future implementation should prove at least:

1. candidate P2 removing a material role cannot self-activate;
2. prior Cn authority can activate P2 only with a complete P2 binding;
3. missing one profile-required role fails roster completeness;
4. unknown role semantic revision fails closed;
5. same source bytes under a new provisioning epoch create a different occurrence;
6. restoring an old source occurrence does not revive old currentness;
7. destination source cannot self-authorize migration;
8. explicit source migration preserves historical old-source verification and changes present source authority;
9. immutable artifacts do not become accidental mutable roots;
10. hidden ambient mutable state cannot enter positive currentness silently;
11. same semantic profile deployed elsewhere does not inherit this deployment's source authority;
12. profile activation fork/equivocation does not resolve by highest sequence/timestamp;
13. serialized profile/binding/source evidence cannot recreate a live currentness capability;
14. no control-plane/profile/source object grants truth, recommendation, governance, deployment, medical/resource, or physical-effect authority.

## 23. Implementation order

Recommended SCI-014 order after this refinement:

```text
#908 bootstrap/currentness architecture
        ↓
immutable ScientificViewSemanticProfileV1
        ↓
non-circular control-plane activation
        ↓
ScientificViewDeploymentBindingV1
        ↓
AuthoritySourceOccurrenceV1 / roster completeness
        ↓
coherent capture + cross-root closure
        ↓
#3279 / #3280 scientific high-water
        ↓
#3281 continuation proof
        ↓
#3282 accepted external target selection
        ↓
#3284 point-of-use current-use bracket
        ↓
ASSURE / research-result consumers
```

## 24. Non-claims

This architecture does not establish:

- scientific truth;
- global scientific consensus;
- global evidence completeness;
- a concrete #1946 research-lineage implementation;
- durable scientific high-water persistence;
- external witness provider qualification;
- action authority.

It defines the exact, non-circular mutable authority surface those later theorems must protect.