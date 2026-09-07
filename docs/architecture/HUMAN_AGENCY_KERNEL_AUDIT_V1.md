# Human Agency Kernel Audit v1

**Status:** architecture audit only; non-authorizing; non-qualifying.

**Base:** `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`

## 1. Purpose

Symthaea and Mycelix increasingly contain mechanisms that can help people understand complex situations, coordinate collective action, delegate bounded work, preserve provenance, and interact with institutions. They also contain mechanisms that can score, rank, gate, recommend, persuade, automate, and act.

Those two directions create the central Human Agency Kernel (HAK) problem:

> How do we increase machine and institutional capability while ensuring that humans retain legitimate authority over themselves, preserve meaningful alternatives, remain able to contest important conclusions, and ideally become more capable rather than merely more dependent?

This audit does **not** define a universal theory of human flourishing, a social-credit system, a production `symthaea-human-agency` crate, a new governance authority, or a claim that AI assistance is beneficial merely because a task becomes easier.

It freezes a common/non-common map before implementation.

The core design target is:

```text
human intent
    -> explicit augmentation relationship
    -> assistance / deliberation / bounded delegation
    -> action
    -> outcome evidence
    -> capability evaluation
    -> learning / revision
```

with **agency, evidence, pluralism, and reversibility** preserved across the chain.

Human protections in this audit are floors. This document makes no claim that only humans can possess moral standing; broader non-human moral-patient or digital-sentience questions remain separate.

---

## 2. Kernel theorem

A future Human Agency Kernel should preserve at least these distinct semantic layers:

```text
BaselineStanding
    != ScopedQualification
    != Recommendation
    != Consent
    != DelegatedAuthority
    != Execution
    != Outcome
    != HumanBenefit
    != LongitudinalCapabilityChange
```

and:

```text
assistance                      != understanding
understanding                   != competence
immediate task success          != transferable skill
recommendation                  != consent
consent                         != delegated authority
confidence                      != calibrated trustworthiness
trustworthiness                 != legitimacy
reputation                      != human worth
expertise                       != civic worth
agreement                       != collective intelligence
consensus                       != truth
minority position               != defect
explanation                     != verifiability
transparency                    != contestability
personalization                 != value authority
model assessment of a person    != authority over that person
scientific evidence             != action authority
```

No common scalar `agency_score`, `flourishing_score`, `citizen_score`, `fitness_score`, or `confidence: f64` should be allowed to erase these distinctions.

---

## 3. Constitutional asymmetry

The Human Agency Kernel should be intentionally asymmetric.

### 3.1 Human baseline standing is not a performance credential

Within a polity, cooperative, family-like institution, organization, or other governed community, baseline rights/standing should not silently vary as a function of model inference, participation frequency, wealth, physical contribution, ideological agreement, AI-estimated consciousness, or generalized reputation.

Candidate invariant:

```text
BaselineStanding(person, community)
    is not derived from
    CombinedBehavioralScore(person)
```

This does not prohibit communities from defining membership boundaries, age/capacity rules, conflict-of-interest rules, or lawful role eligibility. It prohibits an accidental collapse from descriptive/behavioral assessment into generalized human standing.

At minimum, HAK should distinguish baseline protections such as:

```text
Know
Inspect
Contest
HumanReview
Appeal
Exit
Privacy
IndependentExpression
BasicParticipation
```

from scoped operational powers.

### 3.2 Scoped operational capability may be qualified

Some powers legitimately require evidence:

- operating hazardous equipment;
- signing for treasury funds;
- performing clinical or engineering work;
- acting as an emergency responder;
- accessing protected data;
- executing production changes;
- administering cryptographic roots;
- exercising narrowly defined fiduciary authority.

Those should be modeled as scoped capabilities, not as general superiority.

```text
ScopedAuthority
    = exact subject
    + exact domain
    + exact capability
    + exact provenance / qualification
    + explicit constraints
    + currentness / expiry
    + revocation / challenge path
```

A highly qualified water engineer may receive a water-infrastructure operational role without receiving greater human or general civic worth.

### 3.3 Machine authority should be narrower still

Delegated machine authority should normally satisfy:

```text
MachineAuthority
    subset_of ExplicitHumanOrInstitutionalGrant
```

and should be:

- purpose-bound;
- capability-bound;
- time-bounded;
- revocable;
- auditable;
- non-transitive unless explicitly permitted;
- no broader than the delegator can lawfully grant;
- incapable of manufacturing new authority from model confidence.

This is the constitutional asymmetry:

```text
human baseline protections  -> durable floor
human scoped power          -> evidence / role constrained
machine delegated power     -> narrower + expiring + revocable
```

---

## 4. Existing architecture that HAK should learn from

### 4.1 Symthaea rescue consent

`crates/domains/symthaea-subterranean/src/rescue_consent.rs` already implements a strong domain theorem:

```text
silence / distress / identity / role
    != consent

fresh withdrawal / refusal
    > older acceptance
```

The ledger is case-specific, expiring, sequence/epoch ordered, and replay-resistant.

This is excellent evidence for HAK semantics, but HAK should **reuse the theorem, not depend on the subterranean rescue crate**. Rescue consent, research consent, AI delegation, family guardianship, medical consent, and governance consent remain substantively different domains.

### 4.2 Mycelix reciprocal accountability

Mycelix draft PR #28 introduces a useful cross-domain pattern for person-linked access:

```text
attempted person-linked lookup
    -> AccessReceipt before disclosure
    -> subject notice / bounded delayed notice
    -> Inspect / Contest / HumanReview / Appeal / ProofOfPolicy
```

This is one of the strongest existing seeds for a general agency rule:

> consequential observation or inference about a person should create reciprocal accountability, not invisible one-way legibility.

HAK should not duplicate the access-ledger protocol. It should specify when an augmentation or decision system must compose with such accountability infrastructure.

### 4.3 Mycelix AI sub-passports

`crates/mycelix-bridge-common/src/sub_passport.rs` already expresses several valuable delegation ideas:

- human delegator and AI agent are distinct identities;
- action classes are bounded;
- authority expires;
- authority can be revoked;
- delegation has a stated purpose;
- violations can reduce effective privilege.

However, the current signable transcript is not yet a canonical HAK delegation contract. In particular, the current `canonical_bytes()` binds agent/delegator identity, tier, maximum action, severity and expiry, but does not bind every security-relevant field such as `ahimsa_enforced` or the human-readable purpose.

Therefore:

```text
current SubPassport
    != canonical HAK delegation authority
```

A later integration must first freeze and qualify exact transcript semantics.

### 4.4 Symthaea metacognitive calibration

`symthaea-psych-bench` already measures whether internal confidence tracks actual performance through metacognitive calibration metrics such as expected calibration error and discrimination.

That provides a natural experimental substrate for HAK, but system self-calibration is only one side of the problem.

HAK additionally needs **human reliance calibration**:

```text
model confidence calibration
    != human reliance calibration
```

A well-calibrated model can still be presented in a way that encourages excessive deference.

### 4.5 Existing authority/evidence separation

Recent Symthaea architecture repeatedly preserves boundaries such as:

```text
evidence != disposition != belief != action authority
```

and:

```text
simulation result != successful claim != safety != execution permission
```

HAK should inherit this discipline. Human-centeredness must become an authority theorem, not a UI slogan.

---

## 5. A major governance tension discovered by this audit

The current Sovereignty Papers and Mycelix sovereign-profile implementation contain an important design tension.

The architecture correctly rejects capital as the sole governance input and tries to recognize many forms of real contribution. The current sovereign profile uses eight dimensions:

```text
EpistemicIntegrity
ThermodynamicYield
NetworkResilience
EconomicVelocity
CivicParticipation
StewardshipCare
SemanticResonance
DomainCompetence
```

It then composes those dimensions into a score/tier that can gate participation and change vote weight.

That design solves some problems of plutocracy and single-axis reputation, but the Human Agency audit finds that **dimensionality does not by itself solve legitimacy**.

A multi-dimensional behavioral score can still become a generalized civic-ranking system if it controls basic participation.

Several axes are particularly unsuitable as prerequisites for baseline civic standing:

- physical-energy contribution can track access to property, capital, infrastructure, health, geography, or physical ability;
- node uptime/bandwidth can track hardware and connectivity access;
- economic-velocity rules encode a specific economic policy preference;
- participation scores can penalize people with limited discretionary time, including caregivers;
- semantic resonance with community consensus risks rewarding convergence exactly where collective intelligence may need principled dissent;
- domain competence is valuable for scoped technical authority but does not establish greater general civic worth.

Zero-knowledge privacy improves confidentiality of the score. It does not answer whether the score should control the right in the first place.

Therefore HAK-001 proposes the following correction for future review:

```text
Contribution / competence evidence
    -> expertise routing
    -> scoped role qualification
    -> advisory credibility within domain

NOT

Contribution / competence evidence
    -> generalized human worth
    -> automatic baseline-vote multiplier
```

### 5.1 Separate legitimacy from epistemic competence

The underlying governance problem is real: complex decisions benefit from expertise, while legitimate governance requires voice for affected people.

Instead of multiplying civic worth by an expertise score, HAK recommends preserving separate channels:

```text
Democratic / member legitimacy
Affected-party representation
Domain expertise
Evidence quality
Constitutional constraints
```

A governance process may combine them procedurally without pretending they are interchangeable quantities.

One candidate shape is:

```text
proposal
    -> independent evidence review
    -> scoped expert / stewardship review
    -> affected-party review
    -> member / citizen decision
    -> timelock / contestation / appeal
    -> bounded execution
    -> outcome audit
```

Expert panels may have advisory authority, narrow operational authority, or a temporary/suspensive safety mechanism with a strong override path. That is materially different from saying an expert is a more valuable citizen.

### 5.2 A possible future for the eight-dimensional profile

The existing work need not be discarded.

Its safer long-term role may be closer to a private, multidimensional **Contribution and Role Evidence Vector** used for:

- expertise discovery;
- matching people to projects;
- voluntary delegation;
- scoped operational roles;
- stewardship recognition;
- eligibility for specific resource responsibilities;
- adaptive learning recommendations.

Its semantics should be independently audited before any migration. This document does not rename or rewrite the profile.

---

## 6. Research constraints that should shape HAK

### 6.1 Dependent versus autonomous cognitive offloading

Recent research distinguishes **dependent cognitive offloading**, where AI substitutes for core thinking, from **autonomous cognitive offloading**, where AI scaffolds a user's own reasoning while the user retains cognitive agency.

Reference: Zhu et al. (2026), *Not all cognitive offloading is equal: distinguishing dependent and autonomous offloading to generative AI*, Frontiers in Psychology, DOI `10.3389/fpsyg.2026.1878629`.

HAK implication:

```text
assistance amount
    != augmentation quality
```

The system should be able to measure whether assistance increases long-term independent capability.

### 6.2 Explanation can increase over-reliance

Research on AI-advised decision making has repeatedly shown that an explanation can increase reliance on both correct and incorrect answers. Useful explanation is therefore not merely persuasive detail; it must help a person **verify** or appropriately challenge the recommendation.

References:

- Kim et al. (CHI 2025), *Fostering Appropriate Reliance on Large Language Models: The Role of Explanations, Sources, and Inconsistencies*.
- Fok et al. (2024), *In search of verifiability: Explanations rarely enable complementary performance in AI-advised decision making*, AI Magazine.
- *More is not better: Visual uncertainty cues and the fragility of trust calibration in LLM-assisted decision making* (2026), Computers in Human Behavior: Artificial Humans.

HAK implication:

```text
explanation quality
    should be evaluated by
human error-detection / verification behavior
```

not by length, fluency, or subjective persuasiveness.

### 6.3 Human-AI feedback can amplify bias

Glickman & Sharot (Nature Human Behaviour, 2025) report experiments where human-AI feedback loops altered human judgments and amplified biases, with participants often unaware of the AI's influence.

HAK implication:

```text
personalization / repeated advice
    -> possible preference or judgment drift
```

Value influence and belief drift should be measured longitudinally rather than assumed harmless.

### 6.4 Individual creativity can rise while collective diversity falls

Research has found that AI-assisted brainstorming can increase individual idea quality while reducing diversity across the group.

References include Meincke, Nave & Terwiesch (Nature Human Behaviour, 2025) and later work on LLM-driven homogenization.

HAK implication:

```text
individual performance gain
    != collective-intelligence gain
```

Independent elicitation and dissent preservation should become first-class protocols.

### 6.5 Public-sector AI still under-measures impact and contestability

OECD's *Digital Government Outlook 2026* reports broad government AI adoption but limited impact measurement, user engagement, and complaint/feedback mechanisms.

HAK implication:

> impact measurement and contestability should be architecture requirements rather than optional maturity features.

---

## 7. Candidate common semantics

These are candidates for later implementation. HAK-001 does not establish that they belong in one crate.

### 7.1 Augmentation contract

```text
AugmentationContractV1 {
    principal
    goal / purpose
    interaction_mode
    decision_owner
    allowed_actions
    forbidden_actions
    evidence_policy
    uncertainty_policy
    privacy / data-use policy
    contestation_path
    reversibility_requirements
    competence_retention_target
    dependency_budget
    validity_window
    revocation / withdrawal semantics
}
```

Candidate modes:

```text
Reflect
Scaffold
Collaborate
Delegate
```

`Autopilot` should not be a rhetorical synonym for `Delegate`; it should require a separately qualified bounded-execution theorem.

### 7.2 Capability outcome vector

HAK should avoid one flourishing scalar.

A candidate experiment-level vector is:

```text
CapabilityOutcomeVectorV1 {
    task_performance
    understanding
    unaided_transfer
    calibration
    meaningful_option_preservation
    reversibility
    cognitive_or_administrative_burden
    dependency
    viewpoint_diversity
    coordination_quality
}
```

Not every experiment needs every dimension. Missing dimensions must remain missing rather than silently defaulting to success.

### 7.3 Dissent envelope

```text
DissentEnvelopeV1 {
    proposition / decision identity
    independently_elicited_positions
    position provenance
    supporting evidence
    contradicting evidence
    confidence / uncertainty representation
    falsifiers / discriminating observations
    elicited_before_shared_ai_synthesis
    synthesis lineage
}
```

Required theorem:

```text
minority position retained
    != minority position correct
```

The purpose is to prevent accidental epistemic erasure, not to give every claim equal evidentiary weight.

### 7.4 Agency-impact experiment contract

Human-benefit claims should be preregisterable:

```text
AgencyImpactClaimV1 {
    target population / task
    augmentation condition
    comparator
    immediate outcomes
    delayed outcomes
    non-inferiority constraints
    dependency / autonomy constraints
    measurement schedule
    decision rule
    missing-data policy
}
```

For example:

```text
Scaffold mode improves 7-day unaided transfer by >= X
relative to answer-only mode
without immediate accuracy degrading by > Y
and without dependency measure worsening beyond Z.
```

The numeric thresholds must come from an independently justified experiment protocol, not from this architecture audit.

### 7.5 Intent graph

Mycelix is a natural future owner for a coordination representation such as:

```text
Intent
    -> Outcomes
    -> Stakeholders
    -> Constraints
    -> Capabilities
    -> Resources
    -> Roles
    -> Commitments
    -> Actions
    -> Evidence
    -> Review
```

HAK should define the agency constraints around such a graph; it should not own domain truth for finance, care, identity, governance, or law.

---

## 8. Concepts that must remain non-common

Do not collapse these merely because they all involve human choice or authority.

### 8.1 Adult delegation != guardianship / youth autonomy

Mycelix Hearth's graduated-autonomy system concerns guardian/youth capability transitions. Adult AI delegation has different authority, legal, developmental, and consent semantics.

A shared abstract word such as `capability` does not make the two domains interchangeable.

### 8.2 Research consent != medical consent != rescue consent != data consent

A reusable temporal/withdrawal primitive may emerge, but the substantive authorization rule remains domain-owned.

### 8.3 Expertise != legitimacy

Expert evidence may affect recommendations or scoped roles. It must not silently become general person-ranking authority.

### 8.4 Reputation != trust for every purpose

Trust is contextual. A reliable software maintainer is not thereby a reliable clinician, mediator, treasurer, or constitutional interpreter.

### 8.5 Moral-model output != permission

Symthaea moral reasoning can identify concerns and support reflection. A model-generated moral verdict is not by itself consent, law, governance legitimacy, or execution authority.

### 8.6 Human benefit != preference satisfaction

A system can satisfy a short-term preference while increasing dependency, narrowing options, reducing skill, manipulating values, or externalizing costs onto other people.

---

## 9. Institutional Linter direction

A future advisory Institutional Linter should detect structural risk patterns without becoming a policy oracle.

Initial candidate findings include:

```text
affected party absent from deliberation
baseline civic standing depends on behavioral score
model-inferred trait changes fundamental rights
consequential delegated authority has no expiry
withdrawal path absent
no appeal / contestation path
no preregistered success criterion
irreversible action has no stronger evidence threshold
no rollback / compensation path
single metric controls reward and evaluation
expertise is converted into generalized civic worth
consensus similarity is rewarded where dissent is decision-relevant
minority view removed before synthesis
decision maker insulated from downside
benefits concentrated while externalities are distributed
prediction was not recorded before outcome
claimed intervention benefit lacks delayed measurement
```

Every linter result should retain:

```text
rule id
triggering evidence
scope
severity
uncertainty / limitations
possible mitigations
non-authoritative status
```

Required theorem:

```text
linter finding != moral verdict != governance decision
```

---

## 10. Human Agency benchmark program

`symthaea-psych-bench` is the natural experimental host for several HAK questions.

### 10.1 Assistance allocation benchmark

Compare:

```text
NoAI
AnswerOnly
Scaffold
Collaborate
Delegate
```

Measure immediate performance and delayed unaided transfer.

### 10.2 Reliance calibration benchmark

Vary:

```text
AI correct / incorrect
explanation / no explanation
source / no source
internally inconsistent / consistent
uncertainty communication mode
```

Measure whether humans appropriately accept correct advice and reject incorrect advice.

### 10.3 Value-influence / judgment-drift benchmark

Measure whether repeated interaction causes preferences or judgments to converge toward a model's prior position even when such convergence was not requested.

### 10.4 Collective-diversity benchmark

Compare group idea/decision diversity under:

```text
independent humans first
AI synthesis first
independent human + independent AI then synthesis
multi-model / multi-perspective synthesis
```

Measure both answer quality and diversity retained.

### 10.5 Contestability benchmark

Measure whether users can:

- identify what the AI actually decided versus merely recommended;
- locate supporting evidence;
- challenge an assumption;
- select an alternative;
- revoke delegated authority;
- recover after a wrong recommendation.

These are research measurements only. They must not become new human-ranking credentials.

---

## 11. Proposed dependency-gated implementation sequence

```text
HAK-001  Human Agency Kernel Audit v1                         [this audit]

HAK-002  canonical augmentation-contract semantics           [non-authoritative]
HAK-003  capability-outcome vector + missingness semantics    [measurement only]
HAK-004  agency-impact preregistration contract               [research only]
HAK-005  assistance / reliance psych-benchmark suite          [research only]
HAK-006  independent elicitation + DissentEnvelopeV1          [deliberative only]
HAK-007  collective-intelligence experiment harness           [research only]
HAK-008  Institutional Linter v0 advisory rules               [advisory only]

MYC-HAK-001  delegation transcript hardening                  [precondition]
MYC-HAK-002  baseline-standing vs scoped-role governance RFC  [docs / simulation first]
MYC-HAK-003  IntentGraphV1                                    [descriptive]
MYC-HAK-004  intent -> scoped commitment compiler             [explicit authority]
MYC-HAK-005  accountability / contestability composition      [cross-domain]
```

Do not create `HAK-002+` merely because this audit exists. Each implementation tranche should cite the exact HAK theorem it materializes and receive independent qualification.

---

## 12. Migration rules

Future HAK implementation should obey:

1. **No score-to-personhood shortcut.** Model or behavioral scores do not mutate baseline human standing.
2. **No qualification inheritance.** A qualified rescue-consent or physical-agency theorem does not automatically qualify HAK semantics.
3. **No domain laundering.** Adult delegation, guardianship, medical consent, research consent, and rescue consent remain distinct.
4. **No persuasive-explanation proxy.** Explanation quality must not be inferred from verbosity or user agreement.
5. **No hidden authority widening.** Recommendation, confidence, reputation, and moral evaluation cannot manufacture action authority.
6. **No unbound delegation transcript.** Every security-relevant grant field must be canonically bound before it can serve as authority evidence.
7. **No one-number flourishing claim.** Outcome dimensions remain explicit and may conflict.
8. **No post-hoc benefit claim.** Confirmatory human-benefit claims require preregistration.
9. **No collective-performance shortcut.** Group correctness does not erase diversity, minority information, or participant learning.
10. **No forced optimization.** Assistance should preserve meaningful exit and refusal whenever the domain permits it.
11. **No baseline-right decay from inactivity.** Expiring role authority is acceptable; silent decay of fundamental standing is a separate and much stronger claim.
12. **No AI synthesis before independence when independence matters.** Preserve pre-synthesis human judgments in consequential collective reasoning.

---

## 13. Exit gates for HAK-001

HAK-001 is complete as an audit when review agrees that it:

- distinguishes baseline standing from scoped qualification;
- separates expertise from legitimacy;
- separates assistance from long-term capability;
- preserves consent, delegation, execution and outcome as distinct states;
- identifies the current sovereign-profile governance tension without requiring an immediate destructive migration;
- identifies the current SubPassport transcript gap before reuse;
- defines the first research-backed benchmark families;
- maps common semantics without forcing them into one crate;
- provides a dependency-gated implementation sequence;
- introduces no product authority.

HAK-001 does **not** prove that any proposed human-agency intervention improves real human outcomes.

That requires later experiments with real participants, appropriate ethics/research review, preregistered outcomes, and delayed measurement.
