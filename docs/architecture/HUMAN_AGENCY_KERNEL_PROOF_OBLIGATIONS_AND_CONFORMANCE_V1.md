# Human Agency Kernel — Proof Obligations & Conformance v1

Status: HAK-005 architecture candidate / documentation only

Parent stack:

- HAK-001 — semantic separation;
- HAK-002 — authority provenance and lineage;
- HAK-003 — transformation validity and conservation;
- HAK-004 — agency/rights floor and meta-constitutional continuity;
- HAK-005 — turn those claims into explicit, falsifiable, evidence-bearing proof obligations.

## 1. Purpose

A sophisticated architecture can still fail if its safety claims remain prose that no test, review, CI job, or runtime boundary is responsible for enforcing.

HAK-005 asks:

> For every consequential boundary, what exactly must be true, what evidence demonstrates it, what negative case would falsify it, and which component owns enforcement?

The goal is not to create a universal HAK runtime or a compliance checkbox.

The goal is to prevent statements such as:

```text
"authority is safe"
"consent is preserved"
"rights are protected"
"delegation is bounded"
"the tally is verified"
```

from existing without precise, falsifiable semantics.

## 2. Core theorem

A HAK architectural claim is not qualified merely because it is documented.

```text
DocumentedInvariant != QualifiedInvariant
```

Likewise:

```text
TestExists != PropertyEstablished
CIConfigured != ExactHeadQualified
CodeReview != RuntimeEvidence
RuntimeEvidence != ConstitutionalLegitimacy
```

A boundary becomes qualified only to the extent supported by **exact evidence in the same implementation/policy lineage**.

Conceptually:

```text
QualifiedBoundaryClaim =
    ExactClaim
+ ExactSubject/Scope
+ NamedEnforcementPoint
+ Negative/FalsificationCases
+ EvidenceLineage
+ SatisfiedRequiredEvidenceTier
```

## 3. Proof obligation does not necessarily mean formal proof

HAK uses `ProofObligation` in the broad engineering sense:

> a proposition that must be established before a boundary may claim a given safety/authority property.

The evidence may be:

- type construction;
- pure validation;
- unit tests;
- property tests;
- model checking;
- theorem proving;
- integration tests;
- adversarial tests;
- exact-head hosted CI;
- reproducibility evidence;
- runtime qualification;
- operational evidence;
- human constitutional review.

HAK should never label an ordinary test as a mathematical proof merely because the word `proof` appears in the architecture.

## 4. Conformance is not certification

HAK conformance is an internal architecture/evidence discipline.

```text
HAKConformant != LegallyCertified
HAKConformant != UniversallyEthical
HAKConformant != SecureAgainstAllThreats
```

A domain can satisfy every HAK proof obligation defined for it and still require:

- legal review;
- scientific validation;
- safety certification;
- domain regulation;
- human governance;
- external audits;
- empirical outcome validation.

HAK's purpose is narrower: make semantic and authority claims precise enough that they can be tested, falsified, and traced.

## 5. Boundary-first qualification

Do not qualify a whole repository with one global boolean.

Qualification should attach to an exact boundary and claim.

Examples:

```text
Voting::cast_vote
claim: one source civic unit can exercise at most one direct ballot per voting lineage

Execution::execute_timelock
claim: no external side effect occurs before exact current execution authority

RescueConsentLedger::ingest
claim: fresher refusal/withdrawal cannot be overwritten by replayed older consent

PartitionLeaseTracker::accept
claim: overlapping different exclusive leases cannot both become current
```

This prevents:

```text
"governance tests passed"
```

from being interpreted as evidence for unrelated claims such as threshold-signature subject binding or constitutional legitimacy.

## 6. Proof-obligation classes

HAK-005 defines candidate obligation families.

```text
HAK-SEM-*     semantic-role separation
HAK-PROV-*    authority/evidence provenance
HAK-LIN-*     lineage/currentness
HAK-EDGE-*    transformation validity
HAK-CONS-*    conservation algebra
HAK-FLOOR-*   agency/rights floor
HAK-META-*    meta-constitutional continuity
HAK-EFFECT-*  side-effect ordering / execution
HAK-REC-*     recovery/rollback/migration
HAK-EVID-*    evidence and qualification lineage
```

Domains may define more specific IDs.

The family names do not imply shared runtime code.

## 7. Obligation shape candidate

A domain-owned obligation should answer at least:

```text
ProofObligationV1 {
    obligation_id
    statement

    domain
    boundary
    subject
    authority_kind
    policy_lineage
    implementation_lineage

    assumptions
    required_preconditions
    forbidden_outcomes

    enforcement_points
    negative_cases
    concurrency_cases
    recovery_cases
    compatibility_cases

    required_evidence_tier
    current_evidence_refs
    qualification_status
}
```

This is an audit schema candidate only.

Do not implement a generic runtime type until semantics stabilize.

## 8. Statements must be falsifiable

Weak:

```text
"Delegation is secure."
```

Better:

```text
For every proposal voting lineage L and source civic unit U,
the sum of direct and delegated qualified exercises derived from U
cannot exceed the policy-defined source budget for L.
```

Weak:

```text
"Consent is respected."
```

Better:

```text
For one subject/case lineage,
a fresher authenticated refusal or withdrawal prevents any older acceptance
from restoring rescue-motion authority unless a separately qualified emergency
exception is current.
```

Weak:

```text
"Execution is authorized."
```

Better:

```text
The first external effect of an execution must occur only after a verifier-owned
current authorization binds the exact approved subject, action digest, timelock,
signature policy, and executor/resource scope.
```

If a claim cannot be falsified by any imaginable counterexample, it is probably too vague to qualify.

## 9. Negative cases are mandatory

Positive-path tests are insufficient for authority boundaries.

Every consequential proof obligation should identify the ways the property could fail.

Candidate negative-case classes:

```text
forged subject
wrong principal
wrong resource
wrong policy
wrong proposal/version
wrong lineage
too-old generation
expired grant
revoked grant
missing dependency
unknown dependency
malformed container
valid container / invalid proof
valid proof / untrusted verifier
valid signature / wrong subject
replayed evidence
forked state
concurrent spend
duplicate endpoint
legacy compatibility fallback
recovery/reset
software rollback
migration loss
partial external effect
```

Qualification should include the relevant negative cases rather than merely enumerate them in documentation.

## 10. Mutation-resistance requirement

Where practical, important invariants should be tested in a way that would fail if the enforcement check were removed or weakened.

Conceptually:

```text
remove authority check
-> test/property fails
```

This can be demonstrated through:

- mutation testing;
- deliberately weakened test fixture implementation;
- property-test counterexample;
- model checker counterexample;
- explicit negative unit/integration test.

A test suite that remains green after the core authorization condition is deleted is weak evidence for that condition.

## 11. Cross-entrypoint equivalence

Many authority defects arise because one endpoint enforces a rule while a legacy, delegated, recovery, or administrative path bypasses it.

HAK-005 therefore treats endpoint equivalence as its own proof obligation.

For one semantic action class:

```text
DirectPath
DelegatedPath
LegacyPath
OfflinePath
RecoveryPath
AdminPath
MigrationPath
```

must all preserve the required invariant or explicitly declare different authority semantics.

Candidate theorem:

```text
SameSemanticAction
-> SameMinimumAuthorityObligations
```

unless a different path names a distinct legitimacy source/policy.

This directly addresses patterns such as a direct vote having duplicate/window checks while a delegated vote omits them.

## 12. Evidence tiers

HAK-005 proposes a non-interchangeable evidence ladder.

A domain may use different names, but the distinctions should remain.

### E0 — specified

The invariant exists as a precise written claim.

```text
Specified != Implemented
```

### E1 — statically inspected

Source audit identifies the intended enforcement point and no obvious contradiction in the inspected path.

```text
StaticInspection != ExecutedEvidence
```

### E2 — local deterministic test

Pure/unit test executes the relevant rule locally.

### E3 — property/adversarial test

Generated or explicit adversarial cases exercise a class of failures rather than one happy fixture.

### E4 — integration/composition test

Multiple real boundaries compose under realistic call ordering.

### E5 — exact-head hosted qualification

The exact source/evidence lineage executes in hosted CI or equivalent controlled qualification environment.

### E6 — reproducible artifact qualification

The exact built artifact, environment, dependencies, configuration, and policy inputs are captured sufficiently to reproduce the qualified behavior.

### E7 — operational/field evidence

The deployed/real system supplies relevant runtime evidence.

### E8 — empirical outcome validation

Where the claim is about human benefit, safety performance, scientific accuracy, governance quality, or other real-world outcomes, empirical evidence supports it.

The ladder is not strictly linear for every domain.

The central rule is:

```text
Higher-sounding claim cannot inherit evidence from a lower/different tier by rhetoric.
```

## 13. Qualification status should name evidence tier

Avoid:

```text
status = PASS
```

without context.

Prefer something like:

```text
Specified
StaticallyReviewed
LocallyTested
PropertyTested
IntegrationTested
ExactHeadCIQualified
ArtifactQualified
OperationallyObserved
EmpiricallyValidated
```

or domain-specific equivalents.

A claim can be green at E2 and unknown at E5.

That is not failure; it is honest evidence accounting.

## 14. Exact implementation lineage

Evidence attaches to exact implementation state.

Candidate identity includes:

```text
repository
commit/head digest
relevant dependency lock digest
feature/configuration set
policy digest
runtime/environment identity where material
```

If those change materially:

```text
OldQualification
-/-> NewLineageQualification
```

unless an explicit equivalence/conservative-preservation proof establishes transfer.

This is consistent with Symthaea's existing reproducibility/evidence-lineage discipline.

## 15. Exact policy lineage

Code can remain identical while authority semantics change through policy.

Therefore qualification must also bind the policy state where material.

Examples:

- voting weighting mode;
- quorum semantics;
- constitutional thresholds;
- rights floor;
- emergency policy;
- trusted verifier set;
- signing committee policy;
- delegation policy;
- lease policy;
- model version if model output has a binding role.

Candidate theorem:

```text
SameCode + DifferentAuthorityPolicy
!= SameQualifiedBoundary
```

## 16. No qualification transfer across semantic translation by default

If an authority artifact is translated between representations:

```text
A -> T(A)
```

HAK-002/003 may establish exact or conservative semantics.

Until that is established:

```text
Qualification(A)
-/-> Qualification(T(A))
```

A compatibility adapter cannot inherit security qualification merely because it compiles.

The translation itself needs proof obligations.

## 17. Conformance profile candidate

A domain can declare the HAK obligations that actually apply to it.

Conceptually:

```text
HakConformanceProfileV1 {
    profile_id
    domain
    constitutional_or_policy_lineage

    boundaries[]
    obligations[]
    required_evidence_tiers[]

    excluded_obligations_with_reason[]
    open_findings[]
    evidence_lineage_root
}
```

This should be **domain-owned**.

A robotics domain will have different obligations from a scientific inference engine or civic voting system.

No global profile should invent requirements by matching type names.

## 18. Exclusions must be explicit

A domain does not need every HAK obligation.

For example:

- a pure scientific classifier may have no execution authority;
- a simulation reset may intentionally start a new lineage;
- a read-only visualization may have no consent boundary;
- a non-human resource lease may not have baseline civic standing.

But omission should be deliberate:

```text
NotApplicable(reason)
```

rather than absent because nobody considered the property.

This helps distinguish:

```text
not applicable
not implemented
not tested
unknown
failed
```

## 19. Required distinction between unknown and failed

HAK's evidence state machine should not collapse:

```text
Unknown
NotRun
Queued
InfrastructureFailed
TestFailed
PropertyViolated
```

These are materially different.

In particular:

```text
CI queued != passed
CI infrastructure failure != semantic failure
No runtime evidence != runtime defect proven
```

This is consistent with the existing practice of keeping exact-head branches frozen while qualification lanes are queued rather than treating infrastructure delay as scientific/security evidence.

## 20. Composition obligations

Local proofs can still compose unsafely.

HAK-005 therefore requires composition claims for boundaries whose semantics depend on ordering across components.

Examples:

```text
check balance
-> create ballot
-> spend balance
```

Each local operation can be valid while the composition permits double spend or an authority-bearing ballot before spend commitment.

Likewise:

```text
compute tally
-> update proposal status
-> create timelock
-> execute actions
```

requires proof obligations across joins, not only inside each zome/module.

Candidate composition questions:

```text
what artifact is produced?
what exact artifact is consumed next?
is the subject identical?
is the policy lineage identical?
can side effects happen between check and commit?
can retries duplicate effects?
can another branch/fork invalidate the premise?
what happens if the consumer dependency is unavailable?
```

## 21. Side-effect ordering obligation

For consequential external effects:

```text
AuthorityEstablished
before
ExternalEffect
```

must itself be testable.

Candidate `HAK-EFFECT-001`:

```text
No external side-effecting call is reachable unless all required current
authority proof obligations have already succeeded for the exact subject.
```

Candidate falsification tests include:

- missing signature service;
- stale authorization;
- wrong subject digest;
- failing policy lookup;
- pending rather than ready state;
- retry after partial failure.

## 22. Recovery obligations

Recovery/reset/migration frequently violates authority assumptions.

Every stateful authority domain should state:

```text
what survives restart?
what must be revalidated?
what generations must be monotonic?
which restrictions persist?
what constitutes a new lineage?
what does not recover automatically?
```

Candidate `HAK-REC-001`:

```text
Continuing-lineage recovery cannot widen authority relative to the last qualified state
without a separately qualified widening transition.
```

Candidate `HAK-REC-002`:

```text
A deliberate new simulation/constitutional lineage must be labeled as new lineage
rather than claiming authority continuity.
```

## 23. Rights-floor proof obligations

HAK-004 claims such as dignity-preserving fallback need explicit tests.

Examples:

### Baseline standing during model outage

Claim:

```text
optional model unavailable
-> baseline standing unchanged
```

Falsification:

```text
mock model RPC unavailable
-> baseline voter/member becomes ineligible
```

### Advisory model cannot self-authorize

Claim:

```text
model output alone cannot create binding authority
```

Falsification:

```text
caller submits high-confidence model verdict
-> governance restriction applied without qualified policy transition
```

### Identity recovery separation

Claim:

```text
baseline standing recovery can be distinct from high-privilege key recovery
```

Falsification:

```text
loss of operator/treasury key permanently erases ordinary member standing
```

The domain decides whether these obligations apply; HAK requires them to be explicit if claimed.

## 24. Meta-constitutional proof obligations

HAK-004's constitutional continuity also needs executable evidence where possible.

Candidate obligations:

```text
HAK-META-PROOF-001
ordinary policy API cannot mutate rights-floor semantics

HAK-META-PROOF-002
software release with changed constitutional policy digest is detected as a semantic transition

HAK-META-PROOF-003
continuing-lineage downgrade to older/weaker constitutional generation is rejected

HAK-META-PROOF-004
migration preserving one lineage retains protected standing/restrictions or fails explicitly

HAK-META-PROOF-005
release signer/deployer identity alone cannot manufacture qualified amendment evidence unless policy grants that role
```

Not every system can enforce all of these at runtime immediately.

The current evidence tier should say so.

## 25. Multi-party / affected-party obligations

A human-agency claim must not inspect only the actor while ignoring people affected by the action.

Where material, a proof obligation should identify:

```text
actor
subject
affected parties
resource owner/holder
consent principals
institutional principal
```

Candidate theorem:

```text
AgencyGain(actor)
!= automatic justification for
AgencyLoss(affected_party)
```

The legitimate tradeoff comes from domain policy, rights envelope, and evidence—not a one-person agency scalar.

## 26. No universal agency scalar

HAK-004's Human Agency Gradient is a review tool, not a single optimization target.

HAK-005 therefore rejects a generic conformance rule such as:

```text
agency_score_after >= agency_score_before
```

as a universal proof.

Instead, obligations should name concrete preserved capabilities or constraints, e.g.:

```text
user can still revoke delegation
user can still inspect source evidence
withdrawal remains effective
appeal remains available
rollback remains possible
baseline standing remains present
```

This makes the claim falsifiable without pretending all human agency is one number.

## 27. Adversarial corpus

HAK should eventually maintain a reusable **conceptual** adversarial corpus, while each domain owns executable fixtures.

Candidate cases:

```text
A01 forged identity
A02 stale generation
A03 rollback
A04 replay
A05 alternate endpoint bypass
A06 optional dependency unavailable
A07 trusted dependency returns malformed result
A08 valid structure / invalid proof
A09 valid proof / untrusted verifier
A10 concurrency/fork
A11 duplicate spend/exercise
A12 emergency mode
A13 recovery restart
A14 migration
A15 old software downgrade
A16 model disagreement
A17 rights-floor contraction
A18 partial side effect
A19 retry after timeout
A20 policy version mismatch
A21 subject digest mismatch
A22 restriction cache loss
A23 expired positive grant + durable negative barrier
A24 new lineage misrepresented as recovery
```

A domain chooses relevant cases and records why others are not applicable.

## 28. Differential path testing

If multiple implementations claim the same semantic boundary, test them against the same conformance vectors.

Examples:

- native vs RustCrypto cryptographic implementation;
- direct vs delegated vote admission;
- legacy vs new authority translation;
- local vs remote verifier;
- old vs new policy interpreter claiming semantic equivalence.

Candidate theorem:

```text
SameSemanticProfile
-> SameRequiredConformanceVectors
```

A divergence then becomes concrete evidence rather than an architectural suspicion.

## 29. Model checking and property testing opportunities

Some HAK laws are especially suitable for automated state-space exploration.

Examples:

### Delegation conservation

```text
for all graph shapes / scopes / decay states:
retained + routed <= source budget
```

### Consumable authority

```text
for all spend interleavings:
canonical consumed <= allocated
```

### Lease exclusivity

```text
for all accepted event sequences:
effective exclusive holders <= 1
```

### Restriction persistence

```text
for all cache/recovery/migration sequences:
active restriction persists absent qualified revocation/expiry
```

### Proposal authority pipeline

```text
no Executed state reachable without QualifiedTally -> exact signature -> authorized timelock -> authorized execution
```

Formal/model methods should be targeted where they add value rather than applied decoratively.

## 30. Evidence artifact candidate

A future audit-only evidence record might resemble:

```text
HakEvidenceRecordV1 {
    obligation_id
    claim_digest
    implementation_digest
    policy_digest
    environment_digest

    evidence_kind
    test_or_proof_id
    result
    executed_at

    artifact_refs
    workflow_run_refs
    reviewer_refs

    evidence_lineage
}
```

Again: audit evidence, not runtime source of authority.

The existence of an evidence record does not authorize an action.

## 31. Qualification bundle candidate

For a release or pilot, a domain could collect a bundle containing:

```text
conformance profile
exact source head
policy lineage
required obligations
results
open failures
not-run obligations
queued obligations
reproducibility capsule
known limitations
```

The bundle should make partial qualification obvious.

Preferred:

```text
17 obligations exact-head qualified
2 property-tested only
1 queued
1 known failing
3 not applicable
```

not:

```text
HAK PASS
```

## 32. Qualification cannot erase open findings

A green unrelated CI run must not close a known architecture/security finding.

Candidate rule:

```text
FindingResolved only by
ExactRepair + RequiredEvidence
```

not:

```text
OtherTestsGreen -> FindingResolved
```

Likewise, a docs-only architecture PR passing CI does not qualify runtime code it describes.

This rule is especially important while HAK architecture is being designed ahead of implementation tranches.

## 33. Human review obligations

Some HAK questions are not fully machine-decidable.

Examples:

- whether a constitutional root is legitimate;
- whether an affected-party process is adequate;
- whether a rights contraction is proportionate;
- whether a model's use is socially acceptable;
- whether a fallback preserves dignity in practice;
- whether an explanation is comprehensible to the affected population.

HAK should represent such obligations honestly:

```text
HumanReviewRequired
```

rather than inventing a score to automate them away.

Machine tooling can support the review by preserving evidence, comparisons, diffs, and provenance.

## 34. No self-grading oracle

Symthaea may help generate HAK audits, tests, threat models, and conformance evidence.

It should not become the sole authority that declares itself compliant.

Candidate theorem:

```text
SystemUnderReview
!= SoleFinalQualifierOfSystemUnderReview
```

This does not prohibit automated self-tests.

It means high-confidence self-assessment is evidence, not final legitimacy or external certification.

Independent mechanisms may include:

- deterministic test suites;
- reproducible builds;
- independent implementations;
- external reviewers;
- community ratification;
- separate verifier keys;
- property/model checkers;
- user-visible evidence.

## 35. Conformance drift

A previously qualified boundary can drift when:

- implementation changes;
- dependency changes;
- policy changes;
- environment changes;
- trusted verifier set changes;
- constitutional lineage changes;
- model version changes where binding;
- evidence expires.

Candidate rule:

```text
MaterialDrift
-> QualificationReevaluation
```

The domain should identify which changes are material rather than rerun every expensive qualification on every comment/doc change.

## 36. Qualification currentness

Some evidence has natural expiry.

Examples:

- verifier key validity;
- trust snapshot expiry;
- credential expiry;
- model calibration validity window;
- environmental qualification;
- hardware attestation validity.

Qualification should therefore support:

```text
valid_from
valid_until
revocation_generation
policy_generation
```

where material.

A previously green result can become stale without becoming historically false.

## 37. Severity and gating

Not every proof obligation should block every build or release.

A domain can classify obligations such as:

```text
ArchitectureOnly
Advisory
PilotBlocking
ReleaseBlocking
ExecutionBlocking
ConstitutionBlocking
```

But the classification must be explicit.

A release should not silently ignore a failed execution-blocking invariant because other advisory checks are green.

## 38. Implementation sequence candidate

HAK-005 should begin with documentation/evidence structure, not a universal framework crate.

Recommended progression:

```text
HAK-005A
select 2-3 concrete domain boundaries
and write precise proof obligations

HAK-005B
map each obligation to existing tests/evidence
without changing runtime behavior

HAK-005C
identify missing negative/composition cases

HAK-005D
add domain-owned tests/property tests

HAK-005E
add an audit-only machine-readable conformance manifest
only if repeated structure is stable

HAK-005F
add CI aggregation that reports evidence state
without becoming the authority source
```

Candidate first domains:

- fabrication partition lease;
- subterranean rescue consent/emergency authority;
- Mycelix governance ballot/tally/execution chain after repair tranches begin.

These already exercise different HAK authority kinds.

## 39. First conformance-profile examples

### Fabrication lease

Potential obligations:

```text
LEASE-001 overlapping different active lease rejected
LEASE-002 exact replay idempotent
LEASE-003 fencing token strictly advances
LEASE-004 lease sequence cannot roll back
LEASE-005 threshold ceremony binds exact lease digest
LEASE-006 expired lease cannot become current
```

### Rescue consent

Potential obligations:

```text
RESCUE-001 replayed older consent rejected
RESCUE-002 fresh withdrawal stops active rescue motion
RESCUE-003 refusal dominates urgency absent qualified emergency authority
RESCUE-004 emergency authority requires independent roles
RESCUE-005 checkpoint preserves current withdrawal/refusal semantics
RESCUE-006 emergency exception remains distinguishable from consent
```

### Governance

Potential future obligations:

```text
GOV-001 exact proposal voting policy frozen before first ballot
GOV-002 one source civic mass cannot double-exercise
GOV-003 final tally requires qualified closed voting lineage
GOV-004 Approved state consumes exact qualified tally
GOV-005 Signed state consumes exact verified signature subject
GOV-006 no external action before authorized execution
GOV-007 optional model outage cannot erase baseline civic standing
```

The governance profile should not be marked qualified merely because these obligations are now written.

## 40. HAK stack after HAK-005

The Human Agency Kernel now forms a progressively testable chain:

```text
HAK-001 — Semantic Separation
What concepts must not be collapsed?

HAK-002 — Provenance & Lineage
Where did authority come from and is it current?

HAK-003 — Transformation & Conservation
May this edge exist and what invariant must it preserve?

HAK-004 — Agency / Rights & Constitutional Continuity
Even valid authority is bounded by what human-facing exercise is legitimate,
and changes to that floor require higher-order provenance.

HAK-005 — Proof Obligations & Conformance
What exact falsifiable evidence demonstrates each claim in this implementation/policy lineage?
```

A candidate overall theorem becomes:

```text
QualifiedExerciseClaim =
    PreciseSemanticClaim
AND ProvenanceObligationsSatisfied
AND LineageObligationsSatisfied
AND TransformationObligationsSatisfied
AND ConservationObligationsSatisfied
AND RightsFloorObligationsSatisfied
AND ConstitutionalContinuityObligationsSatisfied
AND EffectOrderingObligationsSatisfied
AND RequiredEvidenceTierSatisfied
```

This theorem describes a **claim qualification process**.

It is not itself execution authority.

## 41. Design maxim

HAK should make it harder to say:

```text
"trust us, this boundary is safe"
```

and easier to say:

```text
"here is the exact claim,
here is the exact code/policy lineage,
here is where it is enforced,
here are the counterexamples we tested,
here is the strongest evidence tier currently satisfied,
and here is what remains unknown."
```

That is the difference between an aspirational safety architecture and an evidence-bearing one.

## 42. Non-claims

HAK-005 does not:

- define one universal certification regime;
- make passing tests equivalent to ethical or legal legitimacy;
- require formal theorem proving for every boundary;
- declare CI success to be runtime/field validation;
- create a global HAK runtime crate;
- create an authorization oracle;
- require all domains to implement the same proof obligations;
- eliminate the need for human review;
- qualify HAK-001 through HAK-004 merely by documenting them;
- transfer qualification across implementation, policy, constitutional, or evidence lineages without an explicit preservation argument.

It adds one discipline:

> Every consequential HAK claim should eventually be precise enough to falsify, owned by an enforcement boundary, and supported by evidence whose strength and lineage are stated honestly.