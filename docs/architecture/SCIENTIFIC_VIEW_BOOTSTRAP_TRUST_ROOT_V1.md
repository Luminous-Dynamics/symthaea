# Scientific View Bootstrap Trust Root v1

Status: hardening companion to the SCI-014 scientific-view currentness architecture in #908.

## 1. Purpose

The selection-policy lineage contract correctly requires an independently established bootstrap source. That requirement must not terminate in an implicit promise such as:

```text
TrustedPolicyProvider
trusted = true
caller says this is the root
```

A scientific view therefore needs an explicit, auditable deployment/bootstrap trust-root theorem.

The root answers only:

> Which local administrative source is allowed to bootstrap or rotate the selection-policy lineage for this exact scientific view deployment?

It does not answer which scientific propositions are true.

## 2. Core theorem

Freeze:

```text
bootstrap root material
    != installed bootstrap trust root
    != current bootstrap trust root
    != selection-policy head
    != scientific-view head
    != scientific disposition
    != truth
```

and:

```text
valid key / quorum / artifact
    != authorized root for this deployment

same root bytes
    != same deployment authority

old root historically valid
    != current root

new root says it is valid
    != root rotation authorized
```

The scientific kernel must expose the bootstrap trust choice rather than pretend it can derive authority from purely internal scientific state.

## 3. Deployment scope is first-class

A bootstrap trust root is scoped to one exact deployment/view administration context.

Conceptually:

```text
ScientificViewDeploymentId
ScientificViewBootstrapTrustProfileId
ScientificViewBootstrapTrustRootId
```

The installed root should bind at least:

```text
deployment identity
view namespace identity
allowed initial view profile / epoch class
bootstrap trust profile identity
root subject / administrator / quorum identity
cryptographic or attestation profile, if used
allowed policy-bootstrap transition classes
installation/provisioning evidence identity
root lineage position
```

A root valid for deployment A cannot silently bootstrap deployment B.

Freeze:

```text
same public key
+ different deployment
    != same authority
```

unless an explicit portability profile says otherwise.

## 4. Bootstrap trust is an explicit local choice

There is no purely mathematical theorem that proves which human, institution, device, quorum, release, or provisioning ceremony should administer a scientific view.

The architecture should therefore say this plainly.

A deployment must terminate the authority chain in an explicit source such as, depending on the deployment profile:

- a pinned configuration distributed in a qualified release;
- an owner-local provisioned root;
- a threshold administrator set;
- a hardware-backed trust anchor;
- a separately verified organization/deployment identity;
- another profile-defined bootstrap ceremony.

The shared scientific kernel verifies conformance to that declared source. It does not manufacture legitimacy out of self-reference.

## 5. Persisted root record vs installed capability

A serializable root record is ordinary data.

Conceptually:

```text
PersistedScientificViewBootstrapRootV1
```

may contain the deployment/profile/root identities needed for audit and replay.

A positive runtime authority object should instead be verifier-owned, conceptually:

```text
QualifiedScientificViewBootstrapTrustRootV1
```

with:

- private construction;
- no public raw constructor;
- no Deserialize path into the positive capability;
- exact deployment/view/profile scope;
- exact root-lineage identity;
- exact installation/provisioning evidence;
- currentness/revocation state;
- verifier implementation/execution lineage where material.

Freeze:

```text
bytes from disk
    != installed current bootstrap authority
```

without the owner-local bootstrap-root verifier.

## 6. First installation cannot self-provision

The first root of a deployment has no predecessor root.

Therefore the initial installation is a distinct provisioning theorem, not an ordinary root successor.

Conceptually:

```text
DeploymentProvisioningSourceV1
        ↓
root material R1
        ↓
QualifiedScientificViewBootstrapTrustRootV1(R1)
```

Invalid:

```text
R1 contains statement "R1 is trusted"
        ↓
R1 becomes trusted
```

or:

```text
first policy candidate P1 embeds R1
        ↓
R1 authenticates P1
        ↓
P1 authenticates R1
```

The provisioning source must exist outside the candidate scientific/policy state it authorizes.

## 7. Root lineage uses exact predecessor identity

After bootstrap, root rotation should use an append-only transition lineage rather than mutable replacement.

Conceptually:

```text
ScientificViewBootstrapRootTransitionV1 {
    transition_profile_id,
    deployment_id,
    view_namespace_id,
    sequence,
    predecessor,
    source_root_id,
    destination_root_id,
    transition_class,
}
```

with:

```text
sequence 1 -> BootstrapProvisioning
sequence > 1 -> Previous(exact predecessor root-transition digest)
```

Sequence remains diagnostic metadata, not history identity.

Two same-sequence successors are a fork/equivocation until an already-authorized recovery/rotation theorem resolves them.

## 8. Rotation is authorized by the old root, not the new root

Ordinary root rotation follows:

```text
current root R17
    + proposed root R18
    + rotation profile already authorized under R17
        ↓
qualified R17 -> R18 transition
        ↓
R18 becomes current
```

The destination root cannot authorize its own installation.

Freeze:

```text
R18 approves R18
    != root rotation
```

and:

```text
new administrator set signs its own admission
    != source-authorized administrator rotation
```

## 9. Reintroducing old root bytes does not revive old authority

Suppose:

```text
R1 -> R2 -> R3
```

and R3 intentionally restores the same key/material as R1.

Then:

```text
root_material(R3) == root_material(R1)
```

may hold, but:

```text
root_lineage_position(R3) != root_lineage_position(R1)
```

must hold.

Old positive current-root witnesses cannot become live again merely because identical bytes reappear.

## 10. Revocation and supersession are lineage/currentness events

A historically valid root remains part of the audit trail after it is superseded or revoked.

Preserve:

```text
historically valid root
    != currently authorized root
```

A current root capability must bind exact root-lineage currentness and fail when a later qualified rotation/revocation becomes current.

Stored `current=true` or `revoked=false` fields are not live authority.

## 11. Compromise recovery needs predeclared independent recovery authority

If the current root is suspected or proven compromised, asking that root alone to authorize its replacement may be unsafe.

A deployment profile may therefore predeclare a separate recovery source, for example:

```text
recovery quorum
hardware recovery anchor
offline recovery key set
independent deployment administrator ceremony
```

The exact mechanisms are deployment-specific.

The important theorem is:

```text
compromised current root
    != sufficient recovery authority by default
```

A future recovery receipt should bind:

```text
deployment/view scope
compromised/superseded root lineage
pre-existing recovery profile identity
recovery authority identity
replacement root identity
recovery reason/evidence
recovery execution lineage
```

A recovery profile introduced only after compromise cannot retroactively prove that it was the legitimate recovery source.

## 12. Recovery does not erase compromise history

If recovery installs R21 after compromise of R20, do not mutate history to pretend R20 never existed.

Preserve:

```text
R19 -> R20 -> Recovery(R21)
```

and the exact reason/receipt linking the recovery event.

Scientific-view heads or policy transitions admitted under the compromised interval may need separate revalidation or adjudication. Root recovery does not automatically establish that every downstream object signed/selected during the compromised period is valid or invalid.

That is a separate impact-analysis theorem.

## 13. Root compromise impact is not scientific falsification

A compromised administrative root affects the provenance/currentness of a scientific view.

It does not imply:

```text
all scientific propositions in the view are false
```

or:

```text
opposite propositions are true
```

The Theory Atlas may mark affected view-state/currentness lineages as unqualified, contested, or requiring reconstruction while preserving the underlying evidence artifacts for independent reassessment.

## 14. Root portability must be explicit

Copying a root record, key file, database, VM image, or Atlas snapshot to another deployment must not silently preserve bootstrap authority unless the exact deployment profile allows portability.

Candidate policies may distinguish:

```text
NonPortableDeploymentRoot
PortableFederationRoot
DelegatedSubViewRoot
```

The exact taxonomy is deferred.

Default conservative theorem:

```text
copied root material
    != installed authority in destination deployment
```

## 15. Delegation is not root replacement

A bootstrap trust root may eventually delegate narrowly scoped administration without replacing itself.

Keep delegation separate from rotation.

A delegation receipt should bind at least:

```text
source current root
delegate identity
exact allowed capability/transition classes
view/deployment scope
validity/currentness semantics
revocation path
```

A delegate cannot enlarge its own scope.

Freeze:

```text
delegate for policy updates
    != bootstrap root
```

unless an explicitly qualified root-transition theorem says otherwise.

## 16. Threshold/quorum roots need exact membership lineage

If a deployment uses quorum administration, the root identity must cover the exact member set and threshold semantics.

Changing:

```text
members
threshold
signature/attestation algorithm
revocation semantics
```

changes the root/profile identity and ordinarily requires a qualified root transition.

`2-of-3` and `3-of-5` are not interchangeable merely because some individuals overlap.

The destination quorum cannot self-authorize its own installation.

## 17. Cryptographic validity is necessary but not sufficient

Where cryptography is used:

```text
valid signature
    != authorized root transition
```

The verifier must also bind:

```text
exact deployment/view subject
exact transition/profile domain separation
exact predecessor root lineage
exact candidate destination root
exact transition class
anti-replay/currentness state
```

Algorithm/profile identifiers must have immutable semantics. A label such as `root-signature-v1` cannot silently change the scheme or wire representation.

## 18. Root-profile migration

Changing the bootstrap trust semantics themselves is not an ordinary root rotation if the meaning of the authority system changes.

Examples:

- single administrator -> quorum governance;
- software key -> hardware-backed root;
- one cryptographic scheme -> another;
- local root -> federated delegation profile;
- changed recovery semantics.

Use an explicit migration theorem binding source profile/current root, destination profile/root, migration policy, and qualification evidence.

The destination profile cannot self-authorize migration into itself.

## 19. Root currentness is point-of-use state

A root can rotate or be revoked between verification and use.

Therefore:

```text
root current when checked
    != root current when transition issued
```

The first implementation should either:

- perform bootstrap/policy transition under one immutable owner-local root-state snapshot; or
- compare-and-use/revalidate the exact current root immediately before issuing the positive downstream capability.

This is the bootstrap-root analogue of the policy/head TOCTOU rule.

## 20. Multiple deployments/views may disagree administratively

Different scientific-view deployments may choose different roots and selection policies.

The shared scientific kernel does not majority-vote those roots into a universal authority.

Cross-view federation/import should preserve root/view namespaces and use explicit bridge receipts.

Freeze:

```text
root accepted by many views
    != scientific truth root
```

## 21. Trust-root observability and audit

The Atlas should make administrative provenance inspectable enough that a user can determine, for a current view:

```text
which deployment/view namespace
which current bootstrap-root lineage
which current selection-policy lineage
which view-state lineage/head
which transitions/rotations/recoveries occurred
which assumptions are administrative rather than scientific
```

This makes the trust choice explicit rather than hiding it behind a generic `trusted` flag.

## 22. Qualification vectors

A first executable bootstrap-root qualification should prove at minimum:

1. persisted root record cannot deserialize into positive root capability;
2. caller-supplied `trusted=true` cannot mint root authority;
3. deployment A root cannot bootstrap deployment B;
4. first root requires independent provisioning source;
5. destination root cannot authorize its own installation;
6. ordinary rotation binds exact current predecessor root transition;
7. sequence skip fails;
8. same-sequence competing rotations produce an unresolved fork;
9. historically valid superseded root cannot authorize a new policy transition;
10. reintroduced identical root bytes do not resurrect an old current-root witness;
11. root revocation invalidates later point-of-use currentness;
12. root rotation between check and use fails TOCTOU qualification;
13. compromised root alone cannot invoke an independent-recovery profile unless that profile explicitly allows it;
14. recovery source must predate or be independently provisioned from the compromised candidate state;
15. recovery preserves compromise/rotation history;
16. copied root material does not become destination authority by default;
17. delegate cannot expand its own capability scope;
18. quorum/member/threshold changes require qualified transition/migration;
19. valid cryptographic signature with wrong deployment/view/transition subject fails;
20. current bootstrap root never implies scientific truth, global consensus, recommendation, governance, medical/resource, or physical-effect authority.

## 23. First implementation order

After the lower replay/reconstruction SCI seams qualify, prefer:

```text
1. pure bootstrap-root material/profile identity
2. synthetic deployment-scoped provisioning fixture
3. private QualifiedScientificViewBootstrapTrustRootV1
4. exact-predecessor root transition
5. rotation/revocation/fork negative qualification
6. recovery profile as a separate later theorem
7. selection-policy bootstrap consumes current root capability
8. selection-policy lineage/currentness
9. view-head selection/currentness
```

Do not begin with a universal PKI, global scientific identity system, or generic downstream-implementable `TrustedRootProvider` trait.

## 24. Exit gate

This contract is complete when the repository can truthfully state:

> The scientific-view administrative trust chain terminates in an explicit deployment-scoped bootstrap source whose positive authority is verifier-owned and whose identity, lineage, currentness, rotation, revocation, recovery, and portability semantics are independently auditable. Neither a candidate view, candidate policy, new root, copied key, stale root, or caller promise can mint bootstrap authority for itself.

This is an administrative provenance/currentness theorem for one scientific view. It is not a scientific truth theorem.
