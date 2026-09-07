# Scientific Disposition Argument Graph v1

**Status:** architecture refinement only; non-authorizing; non-qualifying.

**Parent:** `#668@6d06b48d25e8da29fb848cf6e80b231d177e923f`

## 1. Purpose

SCI-001 defines the broad Scientific Method Kernel layers, and #668 specifies the missing target-compatibility / triangulation contract between SCI-006 and SCI-014.

The next semantic gap is the final transition from a triangulated evidence structure to a Theory Atlas scientific disposition.

A flat status field such as:

```text
supported = true
confidence = 0.91
```

would erase exactly the information the preceding architecture has worked to preserve.

The Theory Atlas must instead behave as a **versioned scientific argument graph** whose current disposition is derived from retained evidence, defeaters, falsifiers, retractions, supersession, target compatibility, dependency topology, and unresolved discriminating experiments.

This document does not implement a shared kernel, a disposition engine, a truth evaluator, canonical belief, workspace authority, action authority, or self-improvement promotion.

---

## 2. Core theorem

The shared scientific layer must preserve:

```text
scientific proposition identity
    != evidence contribution
    != evidence lifecycle state
    != argument relation
    != defeater
    != falsifier outcome
    != triangulation assessment
    != scientific disposition
    != truth
    != canonical belief
    != execution / governance authority
```

and:

```text
new supporting evidence
    does not monotonically increase a scalar confidence

new opposing evidence
    does not erase prior support

qualified defeater
    does not erase the evidence it defeats

retraction
    does not delete historical lineage

supersession
    does not rewrite predecessor evidence

contestation
    is a legitimate scientific state

underdetermination
    is a legitimate scientific state
```

The Theory Atlas therefore needs non-monotonic **current interpretation over append-only scientific history**.

---

## 3. Reuse the strongest RCA lesson without importing RCA policy

The RCA shadow-disposition work already establishes several strong general principles:

- qualified defeaters may block a positive disposition;
- unqualified defeater labels have no veto authority;
- support and opposition may coexist as contestation;
- bilateral disagreement need not be converted into a winner;
- reason traces retain predicates that remain true even when a higher-precedence rule controls the primary class;
- disposition remains separate from canonical belief and action authority.

A shared scientific kernel should generalize those **structural principles**, but must not import RCA's exact thresholds, root-count rules, proposition policies, or runtime-specific decision lattice into all scientific domains.

The common contract should define argument objects and required evidence retention. Domain-owned or experiment-contract-owned policy should define when those objects justify a particular bounded disposition.

---

## 4. Proposition versions are immutable scientific targets

The Theory Atlas should treat a proposition revision as a new target identity.

Candidate shape:

```text
ScientificPropositionVersion {
    proposition_id
    proposition_version_id
    semantic_scope_identity
    parent_version_id?
    revision_relation?
}
```

Possible revision relations include:

```text
Clarifies
NarrowsScope
BroadensScope
ChangesOperationalization
ChangesCausalEstimand
ChangesFalsifier
CorrectsError
Supersedes
```

These are descriptive relations, not automatic evidence migrations.

Core rule:

```text
old proposition version
    != new proposition version
```

Evidence bound to version A remains bound to A unless a separate domain-owned target-compatibility / migration receipt establishes how it may inform version B.

Changing population, horizon, estimand, outcome definition, operationalization, or falsifier therefore cannot silently preserve scientific disposition.

---

## 5. Evidence contribution and evidence lifecycle are different axes

An `EvidenceContribution` says what an evidence object contributes to an exact target.

Its lifecycle says whether that contribution remains eligible for current disposition.

These must not be one enum.

### 5.1 Contribution relation

Illustrative shared vocabulary:

```text
Supports
Opposes
FalsifiesWithinDeclaredScope
FailsToFalsify
ReplicatesWithinDeclaredScope
ChallengesGeneralization
InformsMechanism
InformsMeasurement
```

No relation above establishes independence or truth by itself.

`ReplicatesWithinDeclaredScope` still requires SCI-006 dependency analysis and the target-compatibility contract from #668 before stronger replication language is admitted.

### 5.2 Lifecycle state

Candidate lifecycle semantics:

```text
Active
Superseded { successor_contribution_id }
Retracted { retraction_event_id }
Invalidated { invalidation_event_id }
Withdrawn { withdrawal_event_id }
```

The exact vocabulary may be refined later, but the semantic split is mandatory.

A retracted contribution remains in history but is ineligible to support a current disposition unless a separate policy explicitly asks for historical state reconstruction.

A superseded contribution remains inspectable and may still matter for understanding scientific development, but the current disposition should use the designated successor under the exact supersession policy.

No lifecycle transition deletes the original artifact.

---

## 6. Retraction and correction must be append-only

The Atlas should model scientific history as append-only events:

```text
ContributionIssued
ContributionCorrected
ContributionSuperseded
ContributionRetracted
ContributionInvalidated
PropositionRevised
DefeaterIssued
DefeaterResolved
FalsifierTriggered
FalsifierReanalyzed
```

The current view is a deterministic interpretation of this event history under one exact disposition policy version.

Therefore:

```text
current scientific state
    = function(append-only history, current eligibility, current policy)
```

not:

```text
current scientific state
    = mutable row overwritten in place
```

This preserves the ability to reconstruct what the Atlas believed the evidence structure was at an earlier time without pretending the older state remains current.

---

## 7. Defeaters must target the inference they actually attack

A generic `defeated: bool` is too coarse.

At minimum the shared argument graph should distinguish:

```text
RebuttingDefeater
UndercuttingDefeater
ScopeDefeater
```

### 7.1 Rebutting defeater

Targets the proposition-level conclusion by providing qualified evidence for an incompatible claim or outcome.

Example form:

```text
claim: X increases Y
rebutter: qualified evidence supports X decreases Y
```

### 7.2 Undercutting defeater

Targets the inference from evidence to proposition without necessarily supporting the proposition's negation.

Typical targets include:

```text
measurement validity
sampling design
source provenance
execution integrity
identification assumption
estimator implementation
outcome definition
calibration / scoring integrity
historical timing / custody
independence claim
```

This distinction is critical.

For example:

```text
"the instrument is invalid"
```

may undercut an IV estimate without establishing the opposite causal effect.

### 7.3 Scope defeater

Targets transfer/generalization rather than the local result.

Example:

```text
result valid in population A
    + scope defeater for population B
        -> do not generalize to B
```

A scope defeater should not rewrite the result for population A.

---

## 8. Defeater qualification is separate from defeater declaration

The RCA work correctly establishes that a defeater label alone cannot veto evidence.

The shared kernel should therefore preserve:

```text
DefeaterDeclaration
    != QualifiedDefeater
```

A qualified defeater should retain at least:

```text
exact target
exact attacked object / relation
kind
provenance
qualification scope
currentness / validity
supporting evidence identities
dependency inventory
limitations
```

A producer may not become the sole authority for qualifying its own defeater merely because it emitted the label.

Unknown or unqualified defeaters remain visible as unresolved challenges but do not automatically block a disposition.

---

## 9. Defeaters may themselves be challenged

Scientific argument is non-monotonic and may contain attack-on-attack structure.

The shared graph should therefore permit a defeater to be challenged by another qualified argument object.

Example:

```text
Evidence E supports Claim C
    <- Undercutting Defeater D: measurement instrument drift
        <- Challenge R: independent calibration proves no drift in the qualified window
```

The Atlas should not implement this by deleting D.

It should retain:

```text
D exists
D was qualified
R challenges / resolves D under scope S
current disposition policy treats D as resolved under S
```

The scientific argument graph is therefore not required to be a DAG. Cyclic challenge structures may exist and should result in explicit unresolved/contested states unless a registered domain policy resolves them.

---

## 10. Falsifiers are special preregistered attacks

SCI-007 proposes a claim/falsifier graph. Economic Science already requires at least one predeclared falsification criterion per prediction.

The Theory Atlas should keep:

```text
ordinary opposition
    != preregistered falsifier outcome
```

A falsifier assessment should retain:

```text
exact target claim/prediction
falsifier specification id
experiment contract id
qualified observation/result
analysis implementation identity
scope
outcome:
    Triggered
    NotTriggered
    Inconclusive
    NotEvaluable
```

`NotTriggered` means the registered falsifier did not fire under that experiment. It does not mean the claim is true.

Whether `Triggered` yields a disposition such as `RefutedWithinDeclaredScope` belongs to the exact claim/falsifier policy, not a universal kernel rule.

---

## 11. Current eligibility is distinct from historical existence

Every argument object should have a current eligibility assessment separate from its immutable historical identity.

Illustrative reasons for current ineligibility include:

```text
Retracted
Superseded
ExpiredQualification
BrokenProvenance
FailedExecutionVerification
FailedMeasurementAdmission
OutcomePolicyViolation
PostCutoffInformationLeakage
DependencyInventoryIncompleteForRequiredUse
TargetVersionMismatch
```

A contribution may remain perfectly valid historical evidence while being ineligible for a stronger current use.

Example:

```text
retrospective backtest
    may remain valid retrospective evidence
    but is ineligible for prospective evidence credit
```

---

## 12. Disposition must be derived, not stored as an unquestioned label

A Theory Atlas entry may cache a disposition for performance, but scientific authority must come from a reproducible assessment over exact inputs.

Candidate assessment shape:

```text
ScientificDispositionAssessmentV1 {
    proposition_version_id
    disposition_policy_id
    argument_graph_generation_id

    active_support_contributions
    active_opposition_contributions
    active_falsifier_results

    active_rebutting_defeaters
    active_undercutting_defeaters
    active_scope_defeaters
    unresolved_defeaters
    resolved_defeaters

    target_compatibility_receipts
    dependency_analysis
    triangulation_assessments
    pooling_dispositions

    superseded_contributions
    retracted_contributions
    invalidated_contributions

    unresolved_contradictions
    unresolved_assumptions
    unanswered_discriminating_experiments

    primary_disposition
    reason_trace
}
```

The exact Rust shape is deliberately not frozen here.

The key requirement is that the primary disposition never erases the retained reason topology.

---

## 13. Candidate bounded disposition vocabulary

The shared kernel may eventually support a small bounded vocabulary, but these values must not be treated as a total ordering.

Illustrative candidates:

```text
NoAdmissibleEvidence
Underdetermined
TentativelySupportedWithinScope
SupportedWithinScope
TentativelyOpposedWithinScope
OpposedWithinScope
Contested
BlockedByQualifiedDefeater
RefutedWithinDeclaredFalsifierScope
```

This is **not** yet a frozen universal decision lattice.

Different domains may require different admission criteria before producing one of these states.

In particular:

```text
SupportedWithinScope
    != true

OpposedWithinScope
    != false

BlockedByQualifiedDefeater
    != proposition negated

RefutedWithinDeclaredFalsifierScope
    != metaphysically impossible
```

The disposition always remains scoped to the exact proposition version, evidence generation, policy, and scientific context.

---

## 14. No monotone confidence accumulator

The Theory Atlas must not compute scientific disposition by doing:

```text
confidence += supporting_study
confidence -= opposing_study
```

or:

```text
confidence *= replication_factor
```

or:

```text
confidence = average(model_confidences)
```

because this erases:

- target mismatch;
- shared dependencies;
- assumption overlap;
- different evidence channels;
- defeaters;
- retractions;
- falsifier outcomes;
- chronology;
- measurement class;
- causal-identification state;
- non-comparable results.

Local statistical probabilities/confidences remain valid domain quantities. They simply do not become universal Atlas authority.

---

## 15. Agreement does not erase dependency, disagreement does not erase evidence

The #668 triangulation theorem remains binding.

For concordant evidence:

```text
agreement
    + shared data / assumptions / model ancestry
        != independent corroboration
```

For discordant evidence:

```text
disagreement
    != choose winner
    != average until disagreement disappears
```

The Atlas should retain candidate explanations for discordance, including:

```text
regime difference
population difference
estimand mismatch
measurement mismatch
assumption failure
model misspecification
implementation defect
outcome-definition mismatch
time variation
true mechanism heterogeneity
falsification
```

Those explanations should feed SCI-009/SCI-013 as proposed discriminating experiments or falsification attacks.

---

## 16. Evidence dependency and argument attack are different graphs

The kernel should not collapse:

```text
EvidenceDependencyGraph
```

and:

```text
ScientificArgumentGraph
```

Dependency edges answer:

> Which scientific artifacts share ancestry or failure modes?

Argument edges answer:

> What does one scientific object say about another proposition, inference, or evidence path?

The same contribution can therefore be:

- argumentatively supportive of a claim;
- scientifically dependent on another supportive contribution;
- undercut by a third contribution;
- superseded by a later reanalysis.

All four facts matter simultaneously.

---

## 17. Interpretation lineage remains distinct from evidence lineage

RCA also establishes:

```text
evidence-root independence
    != interpretation-root independence
```

The Theory Atlas should preserve the analogous distinction.

Ten independent experiments interpreted by the same exact analysis/model/rule are not ten independent interpretations.

A future shared dependency vocabulary should therefore be able to project both:

```text
observation / evidence ancestry
```

and:

```text
interpretation / adjudication ancestry
```

without converting either into a count-based authority shortcut.

---

## 18. Corrections should create explicit successor lineage

A correction is not necessarily a retraction.

Candidate semantics:

```text
Contribution A
    -> CorrectedBy B
```

where B states exactly what changed:

```text
input data
measurement definition
analysis implementation
reported numerical value
scope
interpretation
```

The Atlas should retain A for historical audit and use B for current interpretation when the exact correction/supersession policy says B replaces A.

A correction that changes the scientific target should instead bind to a new proposition version and should not masquerade as a same-target numerical update.

---

## 19. Negative/null/inconclusive outcomes remain active scientific objects

SCI-001 already requires retaining nulls and failures.

The Theory Atlas should therefore represent outcomes such as:

```text
NoDetectedEffect
NoDetectedViolation
FailedToFalsify
Inconclusive
NonIdentifiable
ObservationallyIndistinguishable
EqualFit
Incomparable
NoExchangeabilityGuarantee
UnavailableEvidence
```

without converting them into either support or opposition when the underlying scientific contract does not justify that relation.

A null result may be highly informative about one mechanism while remaining inconclusive about another.

---

## 20. Time-indexed Atlas views

Because scientific state is append-only and non-monotonic, the Atlas should eventually support:

```text
view_at(scientific_time_cutoff)
```

subject to qualified historical availability/provenance semantics.

This is useful for:

- reconstructing historical scientific consensus;
- SCI-015 historical discovery benchmarks;
- understanding when a proposition became contested or defeated;
- reproducing which evidence was available before a decision or experiment.

A historical view must use evidence/artifacts actually eligible at that cutoff; today's correction/retraction state may be shown separately but must not silently rewrite the historical information set.

---

## 21. Theory Atlas entry shape

A future proposition-centered Atlas entry should roughly preserve:

```text
PropositionVersion
    |
    +-- active supporting contributions
    +-- active opposing contributions
    +-- falsifier specifications / results
    +-- rebutting defeaters
    +-- undercutting defeaters
    +-- scope defeaters
    +-- assumptions / diagnostics
    +-- measurement bindings
    +-- causal-identification state
    +-- uncertainty / calibration state
    +-- dependency topology
    +-- target compatibility
    +-- triangulation assessments
    +-- superseded / corrected contributions
    +-- retractions / invalidations
    +-- unresolved contradictions
    +-- discriminating experiments
    +-- exact current disposition assessment
```

This is a scientific argument graph, not a flat knowledge-base row.

---

## 22. Discriminating experiments are part of disposition, not an optional note

For `Contested`, `Underdetermined`, or defeater-blocked states, the Atlas should retain the experiments most capable of changing the disposition.

Candidate object:

```text
DispositionDiscriminator {
    target_proposition_version
    unresolved_question
    competing_argument_ids
    experiment_contract_family
    expected_discrimination_relation
    prerequisites
    cost/risk/duration/novelty metadata
}
```

SCI-009 may later prioritize these across multiple dimensions.

The Atlas should therefore answer not only:

> What do we currently know?

but:

> What exact evidence would most efficiently resolve what we do not know?

---

## 23. No action authority

The strongest possible scientific disposition remains non-authorizing.

```text
ScientificDispositionAssessment
    != canonical belief
    != workspace / GWT admission
    != governance decision
    != safety approval
    != physical actuation authority
    != Forge capability
    != self-improvement promotion
```

A separate deliberative or governance layer may consume a scientific disposition together with values, risk preferences, rights, obligations, and authority.

The scientific kernel may not perform that transition implicitly.

---

## 24. Proposed SCI-014 prerequisite refinement

The SCI-014 path should now be understood as:

```text
SCI-006 evidence dependency graph
    -> #668 target compatibility / result comparison / triangulation
    -> scientific argument graph
    -> evidence lifecycle / correction / retraction semantics
    -> defeater qualification semantics
    -> disposition policy / reason-trace contract
    -> SCI-014 Theory Atlas v1
```

A production Theory Atlas should not precede these boundaries, because otherwise its first schema is likely to bake in flat support counts or mutable truth-status rows that later migrations would have to unwind.

---

## 25. Implementation guidance

Do not immediately create one giant `symthaea-science-kernel` crate containing every object in this document.

Prefer narrow implementation tranches:

```text
SCI-014a proposition-version identity
SCI-014b evidence-contribution lifecycle
SCI-014c generic argument-edge / defeater contract
SCI-014d disposition assessment / reason trace
SCI-014e Theory Atlas storage/query projection
```

Each tranche should have independent qualification and should migrate existing domain objects only after proving no semantic weakening.

RCA should remain a conformance source for defeater/contestation reasoning; Economic Science should remain a conformance source for measurement/causal/triangulation semantics; Matter should remain a conformance source for blind validation/calibration/OOD semantics; Futures should remain a conformance source for timing/provenance; other domains should contribute similarly.

---

## 26. Review checklist

Review this refinement only on whether it preserves the following boundaries:

1. proposition revision does not rewrite evidence identity;
2. evidence relation is separate from evidence lifecycle;
3. retraction/correction/supersession are append-only history;
4. rebutting, undercutting, and scope defeaters remain distinct;
5. a defeater label is not automatically a qualified blocker;
6. defeaters can themselves be challenged without deletion;
7. falsifier results retain preregistered scope;
8. historical existence is separate from current eligibility;
9. primary disposition does not erase reason topology;
10. disposition is not a total score or truth probability;
11. dependency graph is separate from argument graph;
12. evidence lineage is separate from interpretation lineage;
13. null/inconclusive/non-identifiable outcomes remain first-class;
14. contested/underdetermined states generate discriminating-experiment targets;
15. scientific disposition grants no downstream action authority.

---

## 27. Non-claims

This document does **not** establish:

- a universal scientific disposition algorithm;
- universal defeater precedence;
- universal pooling rules;
- calibrated truth probabilities;
- that every domain must use the illustrative disposition vocabulary;
- that RCA's current shadow policy is qualified or suitable unchanged for scientific claims;
- that any existing evidence contribution is independent;
- that any proposition is true, false, supported, defeated, or refuted;
- that a Theory Atlas implementation should store a mutable canonical truth field;
- that scientific results may authorize action.

Its sole purpose is to prevent SCI-014 from collapsing a rich evidence/triangulation structure into a monotone confidence score or mutable truth label.
