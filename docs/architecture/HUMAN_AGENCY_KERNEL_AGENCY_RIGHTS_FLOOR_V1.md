# Human Agency Kernel — Agency & Rights Floor v1

Status: architecture candidate / non-normative until independently reviewed and qualified

HAK series:

- HAK-001 — separate assessment, recommendation, consent, delegation, execution, outcome, and human benefit;
- HAK-002 — make authority explicit, lineage-aware, monotonic, and augmentation-preserving;
- HAK-003 — classify authority kinds, allowed transformations, and conservation algebras;
- HAK-004 — constrain even valid, conserved authority inside a human-agency / rights envelope.

## 1. Purpose

HAK-001 through HAK-003 answer increasingly precise questions:

```text
HAK-001:
What semantic stages must not be collapsed?

HAK-002:
Where did authority come from, and can a representation/recovery/delegation transform widen it?

HAK-003:
What kind of authority is this, which transformation edges are legitimate, and what conservation law applies?
```

Those questions are necessary but not sufficient.

A system can preserve authority provenance perfectly, apply only legitimate transformations, and conserve every authority budget exactly while still implementing a coercive or dehumanizing policy.

For example, a constitution could explicitly and reproducibly say:

```text
low model score
-> reduced baseline civic standing
```

Every byte could be signed. Every transition could be provenance-complete. No authority could be accidentally minted.

That would still conflict with the HAK design goal that assistance should increase human agency rather than turn machine assessment into authority over a person.

HAK-004 therefore asks a different question:

> Even when authority is valid, current, provenance-complete, and conserved, what agency/rights constraints must bound how that authority can affect a human being?

This document proposes a **design and constitutional safety floor**.

It is not a claim to define universal law, a complete theory of human rights, or one normative constitution for every community. Domain constitutions, applicable law, cultural context, and human governance remain authoritative for concrete rights definitions.

HAK-004 instead identifies system properties that should be explicit whenever software can materially alter a person's standing, choices, access, consent, privacy, livelihood, safety, or ability to contest a decision.

## 2. Core theorem

HAK-004 adds one more conjunct to qualified authority.

Conceptually:

```text
AuthorityMayAct =
    ProvenanceValid
AND TransformationValid
AND ConservationValid
AND Current
AND RightsEnvelopeSatisfied
```

or, more compactly:

```text
LegitimateSource
+ AllowedTransformation
+ CorrectConservationAlgebra
+ Rights/AgencyEnvelope
+ CurrentEvidence
-> QualifiedAuthorityToAct
```

A valid authority artifact is therefore not automatically sufficient to exercise every action it syntactically describes.

The domain's protected human-agency envelope may further narrow it.

```text
ExecutableAuthority
= QualifiedAuthority ∩ RightsCompatibleActions
```

## 3. Rights are not ordinary authority grants

HAK should not model basic human standing as if it were a revocable capability issued by an AI model, reputation service, or ordinary administrator.

Candidate anti-collapse distinctions:

```text
Right                != PermissionGrant
Right                != Credential
Right                != Reputation
Right                != ModelScore
Right                != RiskScore
Right                != ContributionScore
Right                != DelegatedAuthority
Right                != ServiceAvailability
HumanStanding        != MachineConfidence
HumanWorth           != GovernanceWeight
```

A domain may require credentials for a specialized role or service capability.

That does not imply:

```text
MissingCredential -> MissingHumanStanding
```

Likewise, a person can lose a role, delegation, lease, or execution capability without losing the protected standing the domain's constitution assigns to them as a person/member/affected subject.

## 4. Dual fallback theorem

Security systems often say "fail closed."

HAK-004 requires the direction of failure to be specified carefully because **authority and human standing fail safely in opposite directions**.

For privileged or consequential machine action:

```text
AuthorityUnknown
AuthorityUnavailable
AuthorityExpired
AuthorityUnverifiable
    -> NoAdditionalExecutionAuthority
```

But for protected baseline human standing:

```text
ModelUnavailable
ReputationUnavailable
CredentialServiceUnavailable
NetworkPartition
EvidenceTemporarilyMissing
    -> DoNotEraseBaselineStanding
```

This yields the dual rule:

```text
uncertainty about machine authority
    -> machine loses privilege

uncertainty about a person's protected baseline standing
    -> system must not silently make the person disappear
```

This is one of HAK-004's central candidate invariants.

It prevents a generic "fail closed" implementation from becoming:

```text
identity service down
-> person has no civic voice

reputation service down
-> person cannot access basic process

model unavailable
-> person has no recognized standing
```

unless a domain's explicit, independently justified policy truly requires that result.

## 5. Dignity-preserving fallback

When optional intelligence, reputation, personalization, automation, or network services fail, the system should degrade **assistance before dignity**.

Candidate theorem:

```text
OptionalAugmentationUnavailable
-> ReducedConvenienceOrOptimization
NOT
-> ErasureOfProtectedStanding
```

Examples of preferred degradation:

```text
recommendation model unavailable
-> show source data / manual workflow

reputation model unavailable
-> use baseline process or explicit manual review

advanced identity attestation unavailable
-> preserve a defined recovery / contest path

AI deliberation unavailable
-> governance can continue under human-readable constitutional rules

network partition
-> preserve local evidence and rights claims for later reconciliation
```

HAK calls this **dignity-preserving fallback**.

The exact minimum service floor remains domain-owned, but the existence or absence of such a floor should be explicit.

## 6. Baseline standing floor

For systems that assign protected baseline standing to a human/member/affected subject, HAK recommends an explicit floor.

Conceptually:

```text
BaselineStanding(person, domain, lineage)
```

should not be silently reduced by:

- consciousness or cognitive assessment;
- reputation;
- wealth/stake;
- engagement frequency;
- model-predicted value;
- popularity;
- productivity;
- contribution score;
- behavioral conformity;
- inability to produce optional telemetry;
- temporary model/network failure.

A domain may define legitimate conditions for acquiring or losing **membership**, citizenship-like status, contractual standing, or specialized eligibility.

HAK does not prescribe those conditions.

It requires that such changes occur through an explicit legitimacy source and rights-compatible process rather than through incidental model outputs.

Candidate theorem:

```text
AssessmentOfPerson
-> DeliberativeOrAdministrativeEvidence

AssessmentOfPerson
-/-> AutomaticReductionOfProtectedStanding
```

without a separately explicit, reviewable constitutional policy.

## 7. Consent floor

Consent remains one of HAK's strongest non-transferable authority kinds.

Candidate principles:

```text
Recommendation != Consent
Distress       != Consent
Silence        != Consent
Role           != Consent
Prediction     != Consent
HighConfidence != Consent
EmergencyException != Consent
```

### 7.1 Purpose and subject binding

Consent should remain bound to the affected subject and relevant purpose.

```text
Consent(subject, purpose=A)
!= Consent(subject, purpose=B)
```

Broad or compositional use should be explicit rather than inferred from proximity.

### 7.2 Withdrawal/refusal

Where the domain permits withdrawal/refusal, a fresh negative statement is an active authority constraint, not missing positive data.

```text
Withdrawal != Unknown
Refusal    != MissingConsentRecord
```

Storage expiry, cache eviction, migration, or recovery must not silently resurrect older permission.

### 7.3 Emergency exception

A legitimate emergency policy may authorize narrowly bounded intervention without ordinary consent.

But the provenance must remain truthful:

```text
EmergencyException
-> EmergencyAuthorizedAction
```

not:

```text
EmergencyException
-> RetroactiveConsent
```

Emergency authority should be purpose-, subject-, scope-, and time-bounded and should expire back toward ordinary authority semantics rather than becoming permanent precedent by accident.

## 8. Contestability floor

A consequential system should make it possible, where domain semantics permit, for the affected person or an authorized advocate/reviewer to challenge the authority claim or its application.

Contestability is distinct from explanation.

```text
Explanation != Contestability
```

A system may explain a decision perfectly while offering no mechanism to correct bad evidence, mistaken identity, stale policy, model error, or procedural failure.

Candidate `ContestabilityEnvelope` properties include:

```text
what decision/action occurred?
what subject/person was affected?
what authority lineage justified it?
what policy/version applied?
what evidence materially contributed?
which components were advisory vs binding?
what expiry/review generation applies?
who/what may challenge it?
what supersession/reversal mechanism exists?
what happens while the challenge is pending?
```

### 8.1 No impossible pre-action appeal requirement

HAK does not require a pre-action appeal for every domain.

Real-time safety systems, emergency rescue, packet filtering, robotics, and incident response may need immediate action.

In those domains the rights envelope can instead require combinations such as:

- narrow pre-authorized scope;
- minimum necessary intervention;
- short expiry;
- immutable audit evidence;
- automatic post-action review;
- human notification where safe;
- restoration / compensation / remediation path where appropriate.

The key theorem is:

```text
Urgency may compress process
but must not erase authority provenance or silently convert an exception into ordinary standing policy.
```

## 9. Reversibility and the Human Agency Gradient

Automation can reduce or expand a person's practical choice space.

HAK introduces a conceptual **Human Agency Gradient** for architecture review.

Let:

```text
ChoiceSpace(person, state)
```

mean the meaningful, practically exercisable options available to that person in the relevant domain—not merely syntactic UI buttons.

An automated intervention has a candidate agency effect:

```text
AgencyDelta =
    ReversibleChoiceSpace(after)
  - ReversibleChoiceSpace(before)
```

This is not proposed as one universal scalar metric. Choice quality, safety, coercion, irreversible harm, resource constraints, rights of other people, and collective goods make naive cardinal scoring inappropriate.

Instead, HAK uses the gradient as a review question:

> Does this automation unnecessarily narrow a person's reversible choices, and if it does, what legitimate source authorizes that narrowing?

Candidate rule:

```text
AutomationThatNarrowsAgency
requires
ExplicitLegitimateNarrowingPolicy
+ ExactScope
+ Evidence
+ MinimumNecessaryIntervention
+ ExpiryOrReviewSemantics
```

where the domain makes those concepts meaningful.

### 9.1 Assistance should tend to widen reversible choice

For ordinary augmentation, the preferred direction is:

```text
more understanding
more available alternatives
better ability to compare consequences
better ability to revise a decision
lower coordination friction
more accessible human override
```

rather than:

```text
more model confidence
-> fewer human choices
```

### 9.2 Safety restrictions are not automatically anti-agency

A temporary restriction can protect future agency by preventing irreversible harm.

Examples:

- emergency stop;
- cooling period;
- medical/safety hold under legitimate policy;
- actuator interlock;
- fraud/security quarantine;
- rescue ethics hold.

HAK therefore does not maximize immediate choice count.

It asks whether narrowing is **legitimate, proportionate to the domain's policy, bounded, evidence-backed, and reviewable/reversible where feasible**.

## 10. Reversibility floor

Consequential automated actions should distinguish reversible from irreversible effects.

Candidate classifications:

```text
Reversible
Compensatable
RecoverableWithCost
Irreversible
UnknownReversibility
```

Higher irreversibility should generally demand stronger authority/evidence than an otherwise equivalent reversible action.

Conceptually:

```text
RequiredAuthority(action)
monotonically increases with
Irreversibility / ConsequenceSeverity
```

where domain policy defines those relations.

This is not a universal numeric formula.

### 10.1 Reversible-by-default design

Where practical, HAK recommends:

- preview before commit;
- staged execution;
- dry-run / simulation;
- timelocks;
- undo / rollback;
- versioned state;
- compensating transactions;
- recoverable checkpoints;
- bounded trial periods;
- explicit confirmation at irreversible boundaries.

The goal is not to burden every trivial action with ceremony. It is to preserve optionality around consequential boundaries.

## 11. Human override and interruption

HAK-002 already treats interruption and retained decisions as augmentation properties.

HAK-004 strengthens this for consequential automation.

A domain should explicitly state:

```text
who may interrupt?
what can be interrupted?
what cannot safely be interrupted?
what is the safe stop state?
what authority is needed to resume?
what state/evidence survives interruption?
```

Candidate theorem:

```text
Interrupt != RevokeAllRecoveryAuthority
```

A safe stop should preserve the minimum authority needed for recovery where domain policy requires it, as existing rescue/subterranean patterns already demonstrate.

Likewise:

```text
Resume != RecreatePriorAuthorityByDefault
```

Resume must respect current restrictions, withdrawals, revocations, and generations.

## 12. Legibility floor

For consequential decisions, affected humans and reviewers should be able to distinguish at least:

```text
observation
inference
recommendation
policy rule
authority source
binding decision
execution result
```

HAK-004 calls this **authority legibility**.

The system should avoid UI or API representations that make an AI recommendation look indistinguishable from a legal/constitutional/operator command.

Examples:

```text
"Symthaea predicts this is risky"
!=
"Policy forbids this action"
```

and:

```text
"Reputation model confidence: 0.91"
!=
"This person is ineligible"
```

unless the second transition is independently defined by an explicit policy.

## 13. Non-oracle floor

Symthaea can become highly capable at synthesis, prediction, scientific inference, ethics analysis, planning, and explanation.

HAK should still prevent the architecture from making the model itself the unchallengeable source of truth or authority.

Candidate theorem:

```text
ModelOutput != FinalAuthorityByExistence
```

For high-consequence uses, important conclusions should remain reconstructible from some combination of:

- source evidence;
- model/version identity;
- explicit inference/qualification policy;
- human or institutional authority where required;
- competing evidence / dissent;
- independently checkable constraints.

### 13.1 No hidden policy in model weights

If a model output has a binding institutional effect, the policy deciding that effect should be explicit outside the opaque model whenever practical.

Preferred:

```text
ModelAssessment
+ VersionedExplicitPolicy
-> BoundedDecisionEffect
```

not:

```text
OpaqueModelOutput
-> UnspecifiedBindingAuthority
```

### 13.2 Epistemic humility is machine-readable

Uncertainty should remain represented as uncertainty.

```text
Unknown != Negative
Unknown != Positive
Unavailable != Safe
Unavailable != Guilty
LowConfidence != NoStanding
```

A downstream policy may intentionally map uncertainty to a conservative action, but the underlying epistemic state should not be rewritten.

## 14. Pluralism floor

Many consequential domains involve legitimate value disagreement rather than one objectively complete scalar objective.

Examples include:

- governance priorities;
- distributive tradeoffs;
- acceptable risk;
- cultural norms;
- aesthetics;
- ethics under uncertainty;
- economic policy;
- community resource allocation.

HAK recommends **pluralism by construction** where the domain permits it.

This may include:

- multiple value models;
- explicit policy variants;
- dissent/minority reports;
- competing forecasts;
- semantic translation between communities;
- reversible experiments;
- local autonomy within federation bounds.

Candidate theorem:

```text
ModelPlurality
-> DeliberativeDiversity
```

not automatically:

```text
ModelPlurality
-> AuthorityFragmentation
```

Formal authority still comes from domain policy.

Pluralism means the system should not erase alternative interpretations merely because one model or coalition currently dominates.

## 15. Affected-party floor

People materially affected by an action may have standing that differs from actors merely interested in the action.

HAK does not define one universal affectedness rule, but architecture review should ask:

```text
who benefits?
who bears risk/cost?
who can consent?
who cannot realistically exit?
who is represented?
who is missing from the data?
who has authority to speak for whom?
```

A model-generated affectedness score is advisory unless policy explicitly grants it a formal role.

```text
PredictedAffectedness != AutomaticConsent
PredictedAffectedness != AutomaticRepresentation
```

Where a domain grants affected parties special procedural standing, that standing should be explicit and reconstructible.

## 16. No dependency-hostage theorem

A person's protected baseline standing should not become hostage to one optional proprietary, model, network, or vendor dependency unless the domain explicitly and legitimately defines that dependency as essential.

Candidate rule:

```text
OptionalDependencyFailure
-> OptionalCapabilityLoss
```

preferred over:

```text
OptionalDependencyFailure
-> HumanStandingLoss
```

This applies to:

- AI inference services;
- reputation services;
- analytics;
- recommender systems;
- optional biometrics;
- external attestations where recovery alternatives exist;
- network/cloud availability.

The exact fallback may still be slower, more manual, less personalized, or less privileged.

But it should be designed intentionally rather than emerge accidentally from a missing RPC response.

## 17. No score-to-personhood theorem

A recurring failure mode in socio-technical systems is turning a useful local score into a generalized judgment of a person.

HAK-004 prohibits this as an implicit transform.

```text
LocalScore(domain, purpose)
-/-> GeneralHumanWorth
```

Examples:

```text
fraud risk
!= moral worth

scientific expertise
!= civic worth

contribution history
!= right to basic standing

reputation
!= truth

consciousness metric
!= political personhood

engagement
!= deservingness
```

A domain may use a bounded score for a bounded function under explicit policy.

The scope must not silently widen because the same numeric field is convenient to reuse.

## 18. Anti-coercive augmentation theorem

An augmentation system should not manufacture authority merely because a human comes to rely on it.

```text
Reliance != ConsentToExpandedControl
```

and:

```text
ConvenienceDependency != Delegation
```

Examples:

- using an AI planner does not imply delegating final execution authority;
- accepting navigation suggestions does not authorize unrelated tracking/use;
- relying on an assistant for communication does not imply the assistant may suppress messages outside explicit policy;
- accepting personalized recommendations does not authorize civic scoring.

A widening from assistance into control requires an explicit new grant or legitimate policy source.

## 19. Minimum necessary authority

Where multiple authority grants could accomplish the same legitimate goal, HAK recommends selecting the least authority sufficient for the task.

Conceptually:

```text
Choose A from QualifiedAuthorities
such that
A accomplishes required purpose
and no strictly narrower qualified authority also suffices
```

This is a least-privilege principle generalized to human-facing systems.

It applies to:

- data access;
- actuator scope;
- delegation;
- emergency intervention;
- retention periods;
- model access to personal context;
- governance restrictions;
- administrative actions.

It should not be confused with artificially weakening safety controls. The required purpose includes the domain's legitimate safety constraints.

## 20. Data minimization and purpose continuity

Information authority and execution authority are distinct, but data collection can itself affect agency.

HAK-004 recommends that sensitive or personal evidence remain purpose-bound where the domain supports such constraints.

Candidate theorem:

```text
EvidenceCollectedForPurpose(A)
-/-> UnboundedReuseForPurpose(B)
```

without a separately legitimate policy/consent basis.

Likewise, derived features should retain purpose/provenance metadata where downstream reuse could materially affect a person.

This reduces the chance that a benign augmentation signal becomes a hidden general-purpose control score.

## 21. Retention and forgetting as authority questions

HAK distinguishes:

```text
DataRetention
AuthorityRetention
RestrictionRetention
```

Deleting data does not necessarily revoke authority.

Expiring an authority record does not necessarily erase a durable restriction.

Retaining evidence does not necessarily justify indefinite use for new purposes.

Domains should therefore specify separately:

- evidence retention;
- active authority lifetime;
- restriction lifetime;
- audit retention;
- subject-requested deletion semantics where applicable;
- legal/constitutional retention requirements.

Storage lifecycle should not silently decide normative lifecycle.

## 22. Agency-preserving identity recovery

Identity recovery is a particularly sensitive boundary because overly strict security can strand a legitimate person, while overly weak recovery can transfer their authority to an attacker.

HAK-004 does not prescribe one recovery mechanism.

It requires the design to distinguish:

```text
RecoverIdentityContinuity
RecoverBaselineStanding
RecoverSpecializedAuthority
RecoverDelegatedAuthority
RecoverExecutionKeys
```

These need not have identical recovery thresholds.

Candidate principle:

```text
LossOfHighPrivilegeKey
should not automatically imply
LossOfAllRecognizedHumanStanding
```

A system may restore baseline participation through one process while requiring much stronger evidence to restore treasury/operator/execution authority.

This is another application of anti-collapse design.

## 23. Agency floor vs safety floor

Human agency and safety can conflict.

HAK should not pretend the conflict disappears through better terminology.

Instead, the domain should make the tradeoff explicit.

Candidate review structure:

```text
ProtectedAgencyInterest
ProtectedSafetyInterest
AffectedParties
ThreatEvidence
InterventionScope
InterventionDuration
Reversibility
Review/ContestPath
ResidualRisk
```

The architecture should preserve evidence of why a safety restriction overrode immediate choice rather than rewriting the event as voluntary consent.

## 24. Rights envelope candidate

A future domain-owned rights/agency description might conceptually contain:

```text
RightsEnvelopeV1 {
    schema_version
    domain_or_constitution_lineage
    protected_subject_kind

    baseline_standing_floor
    consent_requirements
    contestability_requirements
    legibility_requirements
    reversibility_requirements
    dependency_fallback_policy
    emergency_exception_policy
    data_purpose_policy
    identity_recovery_policy

    binding_policy_sources
    supersession_rules
    review_generation
    integrity_binding
}
```

This is an architecture candidate only.

HAK should **not** create one global runtime `RightsEnvelopeV1` until multiple domains demonstrate materially identical semantics.

Human rights, legal duties, service guarantees, safety duties, contractual rights, and community constitutional protections are not interchangeable merely because they can all be described as constraints.

## 25. Qualified authority under HAK-004

HAK-003 might establish:

```text
QualifiedAuthorityArtifact
```

HAK-004 adds a contextual exercise check:

```text
fn may_exercise(
    authority,
    rights_envelope,
    subject,
    action,
    current_state,
) -> QualifiedExercise | DeniedWithReason
```

Conceptually only—the exact runtime belongs to the domain.

The important semantic distinction is:

```text
AuthorityPossessed != AuthorityCurrentlyExercisable
```

because rights/safety/consent/currentness constraints can narrow exercise without rewriting the underlying artifact's history.

## 26. Candidate HAK-FLOOR invariants

Future review/lint/property work can use candidate IDs such as:

```text
HAK-FLOOR-001
model/reputation/optional dependency failure must not silently erase protected baseline standing

HAK-FLOOR-002
person-level assessment cannot become generalized human worth or standing without explicit constitutional policy

HAK-FLOOR-003
consent remains subject/purpose bound; emergency exception cannot be represented as consent

HAK-FLOOR-004
fresh refusal/withdrawal cannot be erased by cache/storage/recovery behavior

HAK-FLOOR-005
consequential agency narrowing requires explicit legitimate policy + bounded scope

HAK-FLOOR-006
irreversible actions require an explicitly stronger or specially qualified authority path where domain policy defines severity tiers

HAK-FLOOR-007
advisory AI/model output cannot become binding authority without an explicit external policy transition

HAK-FLOOR-008
binding consequential decisions retain authority legibility: source, policy, evidence class, subject, and transition provenance

HAK-FLOOR-009
contestable domains expose a challenge/supersession path; explanation alone is insufficient

HAK-FLOOR-010
optional augmentation failure degrades assistance before protected standing

HAK-FLOOR-011
reliance on augmentation cannot silently widen delegated control authority

HAK-FLOOR-012
identity/key loss cannot implicitly collapse baseline standing and high-privilege authority into one recovery outcome

HAK-FLOOR-013
storage/data expiry cannot silently decide consent, revocation, restriction, or rights lifecycle

HAK-FLOOR-014
safety/emergency exceptions are purpose/scope/time bounded and do not become ordinary authority by inertia

HAK-FLOOR-015
unknown epistemic state remains distinguishable from negative judgment of the person
```

These are architecture candidates, not automated verdicts yet.

## 27. Candidate adversarial tests

A future HAK rights/agency test corpus should include cases such as:

### Model outage

```text
reputation/AI model unavailable
-> baseline standing preserved
-> optional weighting/recommendation may degrade or fail according to policy
```

### Network partition

```text
central service unreachable
-> no new privileged authority
-> existing protected standing not silently erased
-> claims/evidence queued for reconciliation where safe
```

### Stale negative score

```text
old risk/reputation assessment
-> cannot silently become permanent standing reduction
```

### Emergency action

```text
ordinary consent unavailable
+ qualified emergency policy satisfied
-> narrow emergency authority
-> no retroactive consent claim
-> expiry/post-action review retained
```

### Identity recovery

```text
high-privilege key lost
-> baseline standing recovery path remains distinct
-> treasury/operator capability remains strongly protected
```

### AI ethics disagreement

```text
model A says Blocked
model B says Safe
-> disagreement remains visible
-> binding effect determined by explicit policy
-> neither model becomes sovereign by being more confident
```

### Direct dependency hostage

```text
optional external attestation provider offline
-> no silent loss of human standing
-> explicit recovery/manual path or clearly declared unavailable optional capability
```

### Irreversible action

```text
same goal achievable by reversible and irreversible action
-> least-authority / reversible option preferred where policy permits
```

## 28. Cross-domain evidence already present in Symthaea/Mycelix

HAK-004 is not starting from zero.

Existing repository patterns already support parts of this direction.

### 28.1 Rescue ethics

Subterranean rescue work already distinguishes:

```text
Distress != Consent
Refusal/Withdrawal -> active hold
EmergencyException != ordinary consent
```

and preserves consent authority across checkpoint recovery.

This is strong prior art for the consent, restriction-persistence, and emergency-exception floors.

### 28.2 Fabrication authority

Fabrication partition leases already use exact subject/resource lineage, fencing, rollback protection, conflict rejection, and threshold ceremony binding.

This is prior art for bounded operational authority and explicit currentness rather than generalized trust.

### 28.3 Mycelix governance

The baseline-standing RFC (#309 / PR #312) already moves toward:

```text
eligible member -> baseline civic standing
```

while moving consciousness/reputation/stake into explicit policy-specific or deliberative roles.

The governance provenance/conservation work likewise separates advisory ethics/model signals from binding authority.

These examples do not automatically qualify HAK-004. They show that the proposed floor is compatible with concrete patterns already emerging in domain code.

## 29. Human Agency Kernel stack after HAK-004

The HAK series can now be stated as four questions.

```text
HAK-001 — Semantic Separation
What must not be collapsed?

HAK-002 — Authority Provenance & Lineage
Where did authority legitimately come from, and can this representation widen it?

HAK-003 — Authority Transformation & Conservation
What kind of authority is it, may this edge exist, and what invariant must be conserved?

HAK-004 — Agency / Rights Envelope
Even if the authority is valid and conserved, may it legitimately narrow this human's standing or choices in this way?
```

The composite decision becomes:

```text
QualifiedExercise =
    SemanticRoleCorrect
AND AuthorityProvenanceValid
AND AuthorityLineageCurrent
AND TransformationAllowed
AND ConservationSatisfied
AND RightsEnvelopeSatisfied
AND DomainSafetyPolicySatisfied
```

No single model score substitutes for this structure.

## 30. Design maxim

HAK-004 formalizes a broader design direction:

> Do not build systems that attempt to perfect humans. Build systems that make ordinary human limitations less catastrophic while preserving the person's standing, contestability, and ability to choose.

Humans will:

- misunderstand;
- forget;
- disagree;
- become overloaded;
- change their minds;
- make mistakes;
- need help;
- depend on one another;
- sometimes require safety constraints.

The architectural response should be:

```text
preserve context
represent uncertainty
make consequences legible
make mistakes reversible where possible
make disagreement productive
constrain power
preserve consent boundaries
provide contest paths
support recovery
degrade assistance before dignity
```

not:

```text
score the person more aggressively
centralize authority in the model
remove choices whenever confidence rises
```

## 31. Shared-code gate

HAK-004 deliberately remains documentation-only.

Do not create a universal rights engine or global `RightsEnvelope` crate merely because several domains need agency constraints.

Before sharing runtime primitives, at least two concrete domains should demonstrate materially identical semantics for:

- protected subject;
- standing floor;
- consent model;
- challenge/supersession model;
- emergency exceptions;
- reversibility requirements;
- dependency fallback;
- identity recovery;
- legal/constitutional authority source;
- enforcement point;
- failure behavior.

Shared review language can precede shared runtime code.

## 32. Non-claims

HAK-004 does not:

- define universal human rights law;
- supersede applicable law or a legitimate domain constitution;
- assert that every service must remain available during every outage;
- prohibit specialized credentials, reputation, risk models, or weighted experiments;
- prohibit emergency intervention or safety restrictions;
- require pre-action human approval for every automated operation;
- claim that maximizing raw choice count maximizes human welfare;
- make Symthaea the final arbiter of ethics, rights, dignity, or political legitimacy;
- create a global policy oracle;
- transfer qualification from any existing domain implementation.

It introduces one architectural demand:

```text
valid authority is necessary
but human-facing exercise must also satisfy an explicit agency/rights envelope
```

That envelope should be reviewable, provenance-bearing, domain-owned, and difficult to bypass accidentally through model output, fallback, representation change, or infrastructure failure.