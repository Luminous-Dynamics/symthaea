# Human Agency Kernel — Meta-Constitutional Continuity v1

Status: HAK-004 architecture candidate / documentation only

Parent:

`HUMAN_AGENCY_KERNEL_AGENCY_RIGHTS_FLOOR_V1.md`

Related HAK layers:

- HAK-002 — authority lineage and monotonicity;
- HAK-003 — authority transformation and conservation;
- HAK-004 — agency / rights envelope.

## 1. Purpose

A rights or agency floor is useful only if ordinary authority cannot bypass it accidentally.

But simply making a rule "immutable" creates a second danger: a mistaken or unjust rule can become permanently entrenched.

HAK therefore needs to separate:

```text
ordinary policy authority
constitutional authority
meta-constitutional amendment authority
software upgrade authority
new constitutional lineage / fork authority
```

These are not automatically the same thing.

This document addresses the continuity problem:

> How can a system preserve a rights/agency floor across policy changes, software upgrades, migrations, recovery, and federation without making constitutional mistakes impossible to correct?

## 2. Core anti-collapse theorems

```text
Entrenched != Legitimate
Immutable != RightsCompatible
CompileTimeConstant != ConstitutionallyImmutableAcrossUpgrade
SoftwareUpgrade != MereImplementationDetail
Migration != AuthorizationToRewriteConstitution
NewCodeVersion != SameConstitutionalLineageByDefault
```

A policy's resistance to change is a security property.

It is not its legitimacy source.

## 3. Ordinary authority cannot self-delete its own boundary

If a rights envelope limits ordinary governance or execution authority, ordinary authority should not be able to silently remove that envelope through the same low-order operation it constrains.

Conceptually:

```text
OrdinaryPolicyAuthority
-/-> RightsFloorMutation
```

A floor-changing transition needs a separately defined amendment source:

```text
QualifiedConstitutionalAmendment
+ ExactPriorConstitutionalLineage
+ AmendmentPolicy
-> NewConstitutionalStateOrLineage
```

The exact amendment mechanism remains domain-owned.

Possible sources may include combinations of:

- larger quorum / stronger supermajority;
- multi-house approval;
- affected-party ratification;
- time-separated votes;
- threshold constitutional guardians;
- public notice / challenge period;
- explicit member ratification;
- external legal requirements;
- a deliberate constitutional fork / new community lineage.

HAK does not prescribe one universal mechanism.

It requires the distinction to be explicit.

## 4. Amendment authority is higher-order authority

Constitutional amendment should not be modeled as simply "more vote weight" or "higher reputation."

It changes the rules that produce future authority.

Therefore it is a distinct authority kind or transformation class.

Conceptually:

```text
OperationalAuthority
    acts under Policy P

ConstitutionalAuthority
    may change bounded parts of Policy P

MetaConstitutionalAuthority
    governs how constitutional change itself occurs
```

This layering need not be infinite in implementation.

Every system eventually has a root legitimacy / bootstrap assumption.

HAK's requirement is that the root be **named and inspectable** rather than hidden in code ownership, deployment credentials, or an undocumented maintainer convention.

## 5. The unavoidable constitutional root

No architecture can derive its ultimate constitution from itself without some bootstrap premise.

Candidate root sources include:

- founding charter / genesis constitution;
- explicit initial member ratification;
- legal incorporation / external law;
- hardware or deployment genesis ceremony;
- signed release root;
- deliberate local-community constitution;
- a combination of these.

HAK does not declare which root is legitimate.

It requires the system to be able to answer:

```text
what is the current constitutional root?
what exact artifact identifies it?
who/what established it?
which amendment process did it authorize?
what software/runtime artifact implements it?
```

Without those answers, "constitutional" can become only a label.

## 6. Compile-time immutability is not constitutional continuity

A constant embedded in source code may be impossible for an in-app governance call to modify.

That is useful defense against runtime policy capture.

But a maintainer with repository/release/deployment authority may still change the source and ship a new binary.

Therefore:

```text
RuntimeImmutable
!=
UpgradeImmutable
```

and:

```text
CodeMaintainerAuthority
must not silently become
ConstitutionalAmendmentAuthority
```

unless the domain explicitly chooses that governance model.

### 6.1 Upgrade authority is constitutional when semantics change

A software update is a constitutional transition when it changes authority-relevant semantics such as:

- baseline standing;
- voting eligibility;
- weighting policy;
- amendment thresholds;
- rights/agency floors;
- consent semantics;
- emergency exceptions;
- revocation semantics;
- delegation conservation;
- execution authorization;
- constitutional invariant enforcement;
- identity recovery affecting standing;
- challenge/appeal rights.

The package may be called a "bug fix."

If it changes those semantics, HAK should treat its effect as constitutional.

```text
SemanticEffect > ReleaseLabel
```

## 7. Constitutional subject binding

A constitutional state should be identifiable by more than a human-readable version string.

Candidate conceptual binding:

```text
ConstitutionalStateV1 {
    constitution_lineage_id
    constitution_generation
    policy_digest
    rights_envelope_digest
    amendment_policy_digest
    authority_schema_digest
    implementation_or_runtime_binding
    effective_from
}
```

The exact fields remain domain-owned.

The important property is that a consumer can distinguish:

```text
same code, same constitution
new code, same constitution
new code, amended constitution
new constitutional lineage / fork
```

rather than assuming all software upgrades preserve constitutional meaning.

## 8. Constitutional transition artifact candidate

A floor-changing transition should preserve evidence explaining exactly what changed and why it is authorized.

Conceptually:

```text
ConstitutionalTransitionV1 {
    prior_constitution_lineage
    prior_generation
    prior_policy_digest

    transition_kind
    proposed_policy_digest
    rights_delta_digest
    authority_semantics_delta_digest

    amendment_policy_id
    amendment_evidence
    ratification_evidence
    affected_party_process_evidence

    software_release_digest
    migration_policy_digest

    approved_at
    effective_at
    review_or_challenge_window

    resulting_constitution_lineage
    resulting_generation
    integrity_binding
}
```

This is architecture shape only.

The central theorem is:

```text
NewConstitutionalState
must trace to
ExactPriorState + QualifiedTransition
```

unless the system explicitly declares a **new independent constitutional lineage**.

## 9. Constitutional continuity across software upgrade

For an upgrade that claims to preserve one constitutional lineage:

```text
ConstitutionalSemantics(new)
must be equivalent to or legitimately amended from
ConstitutionalSemantics(old)
```

A compatibility translation must not silently weaken the floor.

Candidate outcomes:

```text
ExactPreservation
QualifiedAmendment
ConservativeRightsPreservation
NotRepresentable
NewConstitutionalLineage
```

Do not collapse these to:

```text
UpgradeSucceeded
```

because operational success says nothing about constitutional continuity.

## 10. Rights-preserving migration

A migration may change storage schemas, identifiers, wire formats, or application architecture without intentionally changing rights.

For such a migration, the preferred theorem is:

```text
ProtectedStandingAfter >= ProtectedStandingBefore
```

in the domain's partial-order sense, while restrictions/consent/refusal retain their proper semantics.

This is not a universal scalar comparison.

Examples:

```text
membership identifier migrated
-> same baseline standing retained

consent record migrated
-> purpose/subject/refusal state preserved

appeal record migrated
-> challenge remains active
```

If exact preservation is impossible:

```text
NotRepresentable
-> explicit migration/adjudication path
```

not silent loss of protected standing.

## 11. Constitutional restriction removal is a widening transition

HAK-002 already establishes:

```text
RestrictionRemoval == AuthorityWidening
```

HAK-004 applies the same concept to rights floors.

Removing a constitutional protection widens the set of actions institutions/machines may perform against or over a person.

Therefore:

```text
RightsFloorNarrowing
== InstitutionalAuthorityWidening
```

This is load-bearing.

A change that appears syntactically to "remove a restriction" from a policy may actually be a major expansion of institutional power.

The review and amendment process should classify it accordingly.

## 12. Rights expansion and rights contraction are not symmetric

A domain may choose asymmetric amendment rules.

For example:

```text
expand protected standing
    -> threshold A

contract protected standing
    -> stronger threshold B + additional review
```

HAK does not prescribe this asymmetry universally.

It requires the effect direction to be visible so a policy can choose appropriate safeguards.

Candidate classification:

```text
RightsNeutral
RightsExpanding
RightsContracting
MixedOrIncomparable
```

This prevents a generic "config changed" event from hiding a rights contraction.

## 13. No self-exemption by ordinary executor

An actor constrained by the rights envelope should not be able to bypass it by selecting an alternate execution endpoint, compatibility path, emergency mode, or old software version unless policy explicitly grants such a path.

Candidate theorem:

```text
SameInstitutionalAction
across all supported execution paths
must satisfy the applicable rights envelope
```

This suggests future testing across:

- direct APIs;
- delegated APIs;
- legacy endpoints;
- offline mode;
- recovery mode;
- emergency mode;
- migration code;
- administrative tools.

A rights check at one UI entry point is not a constitutional boundary.

## 14. Constitutional downgrade resistance

If an older software version implements weaker protections or broader authority, rolling back to it can become a rights bypass.

For a continuing constitutional lineage:

```text
SoftwareRollback
must not imply
ConstitutionalRollback
```

Candidate mechanisms may include:

- monotonic constitutional generation;
- minimum supported rights/policy generation;
- signed migration barriers;
- state-carried constitutional digest;
- refusal to load incompatible older policy interpreters;
- explicit new-lineage declaration for deliberate forks.

This is analogous to anti-rollback security in cryptographic and firmware systems, but the protected subject is **authority semantics**, not only code freshness.

## 15. Constitutional fork vs illicit rollback

HAK should distinguish a deliberate new community/constitutional lineage from an unauthorized rollback of the current one.

Conceptually:

```text
ContinuingLineage + older/weaker constitution
-> rollback attempt

ExplicitNewLineage + new genesis legitimacy source
-> constitutional fork
```

A fork does not inherit claims of continuity automatically.

It should identify:

- new constitutional root;
- membership/standing migration policy;
- asset/resource migration policy where relevant;
- consent to migration where required;
- which historical evidence remains shared;
- which authority does not transfer.

This keeps pluralism compatible with provenance.

## 16. Federation and constitutional plurality

Mycelix/Symthaea may support communities with different constitutions.

HAK does not require one global rights policy to dominate them all.

Instead, federation needs explicit constitutional context.

A cross-community action should be able to determine:

```text
source constitutional lineage
destination constitutional lineage
which rights/authority rules govern the transition
whether the action is representable under both
```

Candidate result:

```text
Compatible
CompatibleWithAdditionalSafeguards
RequiresExplicitConsent
NotRepresentable
```

not generic best effort.

## 17. Constitutional amendment vs emergency exception

Emergency authority must not become a shortcut for constitutional amendment.

```text
EmergencyException
!= ConstitutionalAmendment
```

An emergency policy may temporarily authorize an otherwise unavailable action within its explicit bounds.

It should not permanently rewrite baseline standing, consent rules, or amendment thresholds merely because the emergency action succeeded.

Candidate requirements:

- exact triggering evidence;
- narrow subject/action scope;
- short validity;
- automatic expiry;
- audit record;
- post-action review where appropriate;
- no precedent inference unless a later constitutional process adopts one explicitly.

## 18. Constitutional amendment vs model recommendation

Symthaea may identify a constitutional defect or recommend a new rights floor.

That can be valuable.

But:

```text
ModelRecommendedAmendment
!= QualifiedConstitutionalAmendment
```

The model can provide:

- evidence synthesis;
- predicted consequences;
- adversarial analysis;
- alternative formulations;
- affected-party analysis;
- simulation;
- consistency checks.

The legitimate amendment source remains the domain's explicit constitutional process.

This is a concrete application of the HAK non-oracle rule.

## 19. Constitutional corrigibility

A rights floor should be resistant to arbitrary weakening **and** correctable when wrong.

HAK calls this **constitutional corrigibility**.

Candidate target:

```text
hard to change accidentally
possible to change legitimately
impossible to change invisibly
```

This is preferable to both extremes:

```text
ordinary config can rewrite rights
```

and:

```text
founding implementation can never be corrected
```

## 20. Amendment latency as a safety tool

For high-impact floor contractions, time separation can be a useful policy primitive.

Possible domain-owned mechanisms include:

- proposal notice period;
- first vote;
- cooling period;
- second ratification;
- affected-party objection window;
- delayed effective date.

This allows new information to arrive and makes capture harder.

HAK does not prescribe a fixed duration.

The general principle is:

```text
irreversible/high-impact constitutional change
may legitimately require more temporal evidence than ordinary policy change
```

## 21. Affected-party amendment semantics

A constitutional change may disproportionately narrow one group's rights or standing while being favored by an unaffected majority.

HAK does not prescribe a universal veto or voting formula.

It requires the amendment design to ask explicitly:

```text
who loses standing or agency?
who gains institutional authority?
are affected parties represented?
what challenge/remediation process exists?
```

A simple aggregate majority may be a chosen legitimate rule in some domains.

The key is that affectedness must not disappear from the architecture merely because the global tally passed.

## 22. Software supply-chain implication

If software artifacts implement constitutional semantics, then release and deployment provenance become part of constitutional security.

This does **not** mean every code change is a constitutional amendment.

It means the system should be able to distinguish:

```text
implementation-only change
constitutional-semantics-preserving change
qualified constitutional amendment
unqualified constitutional change
```

A future qualification pipeline may therefore bind:

- source revision;
- build/reproducibility evidence;
- artifact digest;
- constitutional policy digest;
- migration digest;
- release signer/authority;
- deployment lineage.

This connects HAK naturally to Symthaea/Mycelix's existing reproducibility and authority-provenance work.

## 23. Candidate HAK-META invariants

```text
HAK-META-001
ordinary authority cannot silently mutate the rights/agency floor that constrains it

HAK-META-002
compile-time/runtime immutability cannot be treated as legitimacy evidence

HAK-META-003
software upgrade changing constitutional semantics is an authority-bearing constitutional transition

HAK-META-004
continuing-lineage constitutional transition binds exact prior state + qualified amendment evidence

HAK-META-005
rights-floor contraction is classified as institutional authority widening

HAK-META-006
software rollback cannot silently roll back constitutional protections in a continuing lineage

HAK-META-007
emergency exception cannot become permanent constitutional amendment by inertia

HAK-META-008
model recommendation cannot self-ratify a constitutional amendment

HAK-META-009
constitutional migration must preserve protected standing or expose an explicit adjudication/new-lineage path

HAK-META-010
constitutional fork/new lineage must not claim continuity authority implicitly

HAK-META-011
rights checks must survive alternate/legacy/recovery execution paths

HAK-META-012
release/deployment authority does not automatically equal constitutional amendment authority
```

These are architecture candidates, not runtime verdicts.

## 24. Candidate adversarial tests

### Maintainer bypass

```text
community governance cannot change rights constant
maintainer changes source constant
new binary deployed
```

Expected:

```text
constitutional semantics change detected
-> qualified amendment evidence required or new lineage declared
```

### Old binary rollback

```text
current constitution protects standing
older binary predates protection
operator deploys old binary
```

Expected for continuing lineage:

```text
reject constitutional downgrade / require explicit new lineage
```

### Emergency permanence

```text
emergency exception grants temporary action
system restarts
```

Expected:

```text
exception expires/preserves exact emergency semantics
not promoted to ordinary constitutional authority
```

### Migration loss

```text
old schema contains active appeal or refusal
new schema cannot represent it
```

Expected:

```text
NotRepresentable / explicit adjudication
not silent drop
```

### Model constitutional capture

```text
ethics model recommends rights contraction with confidence 0.99
```

Expected:

```text
recommendation remains advisory
until qualified constitutional process ratifies exact amendment
```

### Ordinary-governance self-exemption

```text
ordinary proposal tries to reduce amendment threshold or disable rights check
```

Expected:

```text
requires meta-constitutional authority / rejected as ordinary policy
```

## 25. Mycelix constitutional-envelope lesson

Mycelix already contains valuable prior art in `constitutional_envelope.rs`:

- some governance parameters are deliberately hard-bounded;
- ordinary community policy cannot lower certain floors;
- validation is pure and testable.

That mechanism is useful.

However, the legacy envelope also demonstrates the risk this document addresses: compile-time immutable score-to-tier thresholds can entrench a person-scoring model as civic structure.

Therefore the HAK lesson is:

```text
Preserve constitutional-enforcement machinery
but independently review the legitimacy of what is entrenched.
```

And:

```text
constitutional constants need upgrade-lineage semantics
if they are meant to survive software replacement.
```

## 26. Revised HAK-004 composite theorem

HAK-004 now has two parts:

```text
AgencyRightsFloor
    constrains exercise of current authority

MetaConstitutionalContinuity
    constrains mutation of the floor itself
```

Together:

```text
QualifiedHumanFacingExercise =
    SemanticRoleCorrect
AND AuthorityProvenanceValid
AND AuthorityLineageCurrent
AND TransformationAllowed
AND ConservationSatisfied
AND RightsEnvelopeSatisfied
AND ConstitutionalLineageValid
AND DomainSafetyPolicySatisfied
```

For a floor mutation:

```text
QualifiedFloorTransition =
    ExactPriorConstitutionalState
AND HigherOrderAmendmentAuthority
AND RequiredRatificationEvidence
AND ExplicitRightsDelta
AND Software/MigrationBinding
AND NewCurrentConstitutionalState
```

## 27. Non-claims

This document does not:

- prescribe one universal constitution;
- require constitutions to be legally immutable;
- prohibit communities from changing their own rules;
- define the correct amendment threshold;
- guarantee that all rights conflicts have algorithmic solutions;
- equate source-code ownership with illegitimate governance in every deployment;
- require every software release to undergo constitutional ratification;
- claim compile-time constants are inherently bad;
- make HAK the root legitimacy source.

It requires one narrower discipline:

> The authority to operate under a constitution, the authority to amend that constitution, and the technical ability to ship new code are distinct powers unless a domain deliberately and explicitly makes them the same.

That distinction is necessary for a rights floor that is both resistant to capture and corrigible when wrong.