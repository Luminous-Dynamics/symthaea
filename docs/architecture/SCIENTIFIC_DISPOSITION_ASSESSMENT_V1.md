# Scientific Disposition Assessment v1

**Status:** architecture contract candidate only; non-authorizing; non-qualifying.

**Parent:** `#769@9e73cb4bdf340b0cc58639812584e8e3115131fc`

## 1. Purpose

SCI-006 establishes evidence dependency topology, #668 separates target compatibility and triangulation, #701 defines a defeater-aware scientific argument graph, #729 freezes immutable proposition identity, and #769 defines append-only evidence-contribution lifecycle with source lifecycle separate from external adjudication.

The next missing SCI-014 boundary is the derivation of a bounded current scientific disposition from those exact objects without collapsing the scientific reason topology into one mutable label or scalar confidence.

This document specifies a candidate `ScientificDispositionAssessmentV1` contract. It does not implement a Theory Atlas, define universal truth probabilities, select universal evidentiary thresholds, or grant governance/action authority.

---

## 2. Core theorem

The shared scientific layer must preserve:

```text
scientific proposition
    != eligible evidence set
    != scientific argument graph
    != dependency topology
    != triangulation assessment
    != lifecycle/adjudication state
    != disposition policy
    != derived scientific disposition
    != reason topology
    != truth
    != canonical belief
    != recommendation
    != action authority
```

and:

```text
primary disposition != scientific record
summary label != reason topology
more supporting artifacts != monotone confidence
no visible opposition != proof
blocked evidence != opposing evidence
ineligible evidence != nonexistent evidence
unresolved contradiction != averaging target
cached disposition != durable authority
```

A disposition is a deterministic, replayable projection over exact immutable inputs under one exact registered policy and one exact information cutoff.

---

## 3. Candidate object shape

Conceptually:

```text
ScientificDispositionAssessmentV1 {
    assessment_id,
    proposition_id,
    requested_scientific_use,
    information_cutoff,

    disposition_policy_id,
    proposition_profile_id,

    source_lifecycle_generation_id,
    argument_adjudication_generation_id,
    dependency_graph_generation_id,
    triangulation_generation_id,

    admitted_contribution_ids,
    excluded_contribution_receipts,

    support_relations,
    opposition_relations,
    falsifier_results,
    active_defeaters,
    resolved_defeaters,
    unresolved_defeaters,

    compatibility_receipts,
    triangulation_receipts,
    pooling_dispositions,

    unresolved_assumptions,
    unresolved_contradictions,
    unresolved_scope_questions,
    discriminating_experiment_refs,

    primary_disposition,
    reason_topology_commitment,
}
```

The exact Rust shape is not frozen here.

The primary disposition is an output field derived from the complete retained reason topology. It is not the sole scientific state.

---

## 4. Exact input-generation binding

A disposition must bind every mutable projection layer by exact generation or content identity.

At minimum, the future assessment should be able to prove which versions of the following were used:

```text
proposition semantic identity
source lifecycle graph
external argument/adjudication graph
evidence dependency graph
target-compatibility / triangulation graph
disposition policy
historical/current information cutoff
scientific-use profile
```

If any bound generation changes, the previous disposition may remain a historical assessment but is stale for the new generation.

Therefore:

```text
same proposition + changed evidence graph
    -> new disposition assessment

same proposition + changed lifecycle graph
    -> new disposition assessment

same proposition + changed disposition policy
    -> new disposition assessment
```

No mutable `current_disposition` field on the proposition may silently update in place.

---

## 5. Disposition policy identity

The shared kernel should not contain one universal policy that decides all scientific fields.

A future `DispositionPolicyProfile` should have immutable identity and should declare the rules it uses to derive bounded disposition labels from qualified inputs.

The policy identity must be stronger than a display string such as:

```text
"default-science-v1"
```

Conceptually it should be content-addressed or otherwise bound to exact immutable policy semantics.

Changing any authority-relevant rule requires a new policy identity, including changes to:

- admissibility requirements;
- defeater precedence;
- treatment of unresolved assumptions;
- required evidence channels;
- falsifier interpretation;
- triangulation requirements;
- treatment of dependency overlap;
- rules for contested evidence;
- rules for scope/generalization;
- historical-information policy.

The policy may be domain-owned. The shared kernel must not infer economics, neuroscience, physics, or RCA thresholds as universal scientific law.

---

## 6. Candidate bounded dispositions

Illustrative non-total-order states include:

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

These labels are candidates, not a universal lattice.

The shared kernel must not assume:

```text
NoAdmissibleEvidence < Underdetermined < Tentative < Supported
```

as one scalar evidentiary ordering.

For example, `BlockedByQualifiedDefeater` and `Contested` are structurally different states, not values on one confidence axis.

---

## 7. Complete reason topology is mandatory

A disposition must preserve the evidence and reasoning that produced it.

The system must not return only:

```text
SupportedWithinScope
```

without retaining at least the exact qualified inputs considered, exact exclusions, argument relations, defeaters, dependency information, compatibility/triangulation receipts, unresolved assumptions, contradictions, and policy identity.

The reason topology should support questions such as:

- Which contributions currently support this proposition?
- Which are ineligible and why?
- Which active defeaters block an inference?
- Which opposing results remain unresolved?
- Which supporting studies share data, assumptions, models, or interpretation roots?
- Which results are not target-compatible?
- Which falsifiers triggered, failed to trigger, were inconclusive, or were not evaluable?
- Which observations were excluded because they were not available by the historical cutoff?
- Which discriminating experiment could change the current disposition?

A human-facing summary may compress this topology, but the underlying assessment must retain it.

---

## 8. Included and excluded evidence are both part of the audit

A scientifically useful disposition cannot list only evidence that survived admission.

It should also retain exclusion receipts for evidence that was considered but not admitted for the requested use.

Illustrative reasons include:

```text
TargetMismatch
MeasurementNotQualified
SamplingNotQualified
UncertaintyNotQualified
HistoricalCutoffViolation
SourceLifecycleIneligible
ExternalAdjudicationBlock
ExecutionNotVerified
DependencyPolicyExclusion
UnsupportedEvidenceChannel
SupersededForRequestedUse
MalformedOrUnresolvedLineage
```

This prevents silent selection from disappearing into the projection.

The exclusion receipt itself is not opposition to the proposition.

```text
Evidence E excluded from use U
    !=
Evidence E supports not-P
```

---

## 9. Nulls, failures, and abstentions must survive disposition

The disposition layer must preserve scientifically meaningful non-positive outcomes such as:

```text
null result
failed replication
forecast abstention
inconclusive diagnostic
not-evaluable falsifier
non-identifiable mechanism
observational equivalence
missing outcome under preregistered policy
```

They must not disappear merely because they do not produce a support/opposition edge.

A future policy may decide how they constrain the disposition, but the shared kernel must retain them as exact reason objects.

---

## 10. No support-count arithmetic

The disposition engine must not infer:

```text
supporting contributions > opposing contributions
    -> supported
```

without exact policy and dependency/compatibility handling.

Ten supporting contributions may form one known-dependency component. One opposing contribution may directly trigger a preregistered falsifier. Several results may target related-but-not-poolable estimands.

Therefore raw counts are descriptive only.

```text
N_support
N_oppose
```

must never become hidden authority.

---

## 11. No hidden scalarization

The shared kernel must not silently compute a universal value such as:

```text
confidence = 0.84
truth_probability = 0.91
support_score = 17.2
```

from heterogeneous scientific evidence.

If a domain explicitly defines a quantitative aggregation model, that model belongs to a versioned policy/assessment lineage and must expose its assumptions, dependency handling, target-compatibility requirements, uncertainty model, and calibration evidence.

Even then, the scalar is a derived model output, not the scientific record or truth authority.

---

## 12. Defeater-aware semantics

A disposition must distinguish at least:

```text
rebutting defeater
undercutting defeater
scope defeater
```

A qualified undercutting defeater may block the inference from evidence `E` to proposition `P` without generating an opposing proposition edge.

Example:

```text
E --Supports--> P
D --Undercuts--> inference(E -> P)
```

The current disposition may become blocked or underdetermined, but it must not infer:

```text
D -> Supports(not-P)
```

unless D actually carries that separate scientific relation.

---

## 13. Contradiction is a retained state

The Atlas must not force every qualified disagreement into immediate resolution.

Possible states include:

```text
supporting evidence remains admissible
opposing evidence remains admissible
no qualified defeater resolves either side
no policy-authorized pooling/precedence applies
    -> Contested
```

or:

```text
available evidence cannot discriminate candidate mechanisms
    -> Underdetermined
```

The reason topology must retain exactly what remains unresolved.

No averaging, latest-wins, majority vote, institutional prestige, author identity, or model popularity may resolve contradiction unless an explicit qualified policy says so.

---

## 14. Falsifier semantics remain scoped

Preregistered falsifier outcomes remain distinct from ordinary opposition.

```text
Triggered
NotTriggered
Inconclusive
NotEvaluable
```

A policy may derive `RefutedWithinDeclaredFalsifierScope` from a qualified `Triggered` result only when the exact proposition/falsifier contract authorizes that interpretation.

`NotTriggered` never implies truth.

A falsifier for proposition version P1 does not automatically transfer to P2 merely because they share a family.

---

## 15. Triangulation is evidence structure, not disposition

A strong triangulation assessment may influence a domain policy, but it must not itself mint:

```text
SupportedWithinScope
```

without the registered disposition rule.

Likewise:

```text
DeclaredDisjointWithinScope
```

from dependency analysis is not equivalent to verified independent replication.

The disposition assessment retains the triangulation receipt and the exact dependency topology used to interpret it.

---

## 16. Current view and historical view use the same machinery

A current disposition is just one assessment with a current information cutoff.

A historical disposition is the same derivation under an earlier cutoff.

Conceptually:

```text
assess(P, as_of=t0, policy=Q)
```

must exclude lifecycle events, adjudication findings, evidence contributions, target-equivalence receipts, or other information not admissibly available by `t0`.

Today's correction, retraction, invalidation, or replication must not leak backward into an earlier Atlas state.

The resulting historical assessment remains immutable even after later information arrives.

---

## 17. Assessment currentness

A positive disposition receipt must never be interpreted as eternally current.

A future point-of-use check should require the bound generations and policy identity still match the requested view.

Conceptually:

```text
assessment.is_current_against(
    proposition_identity,
    lifecycle_generation,
    adjudication_generation,
    dependency_generation,
    triangulation_generation,
    policy_identity,
    information_cutoff_policy,
)
```

A stale assessment remains historical evidence that the system once derived that disposition; it does not remain authoritative for the present view.

---

## 18. Reason topology commitment

A future durable assessment should bind a canonical commitment over its exact reason topology so that a primary disposition cannot be detached from the reasons that produced it.

Conceptually:

```text
reason_topology_commitment = H(
    proposition_id
    + policy_id
    + input generations
    + admitted contribution IDs
    + exclusion receipts
    + argument/defeater relations
    + falsifier outcomes
    + dependency/triangulation receipts
    + unresolved questions
    + discriminating experiments
    + primary disposition
)
```

The exact canonical encoding and digest primitive belong to the Scientific Artifact/Identity layer and are not chosen here.

A caller-supplied digest does not self-qualify the topology.

---

## 19. Determinism and reproducibility

For the same exact immutable inputs and policy, disposition derivation should be deterministic unless the policy explicitly declares an external stochastic procedure and binds its execution evidence.

The target invariant is:

```text
same exact proposition
+ same exact qualified evidence view
+ same exact argument graph
+ same exact dependency/triangulation view
+ same exact policy
+ same exact information cutoff
    -> same disposition + same reason topology
```

This is replay determinism, not a claim that the scientific conclusion is eternally correct.

---

## 20. Unknown/incomplete inputs fail epistemically closed

Missing scientific information must not default to support.

Examples:

```text
missing dependency inventory
missing target compatibility
unresolved lifecycle branch
unknown proposition migration relation
missing outcome under preregistered policy
unqualified source authority
incomplete measurement admission
```

should yield explicit unresolved/excluded reason states according to policy.

The generic kernel must not convert `Unknown` into `Satisfied` merely to produce a primary disposition.

---

## 21. Disposition does not grant action authority

Even a disposition such as:

```text
SupportedWithinScope
```

is a scientific assessment only.

It does not itself authorize:

```text
policy enactment
medical treatment
resource allocation
physical control
governance action
Mycelix execution
self-modification
```

The authority boundary remains:

```text
ScientificDispositionAssessment
    != recommendation
    != normative choice
    != governance authorization
    != effect capability
```

Any later decision system must explicitly consume scientific evidence together with values, rights, constraints, safety policy, and authorized human/institutional governance.

---

## 22. Candidate reason-edge families

A future reason topology may need typed edges such as:

```text
AdmittedForUse
ExcludedForUse
Supports
Opposes
FalsifiesWithinScope
FailsToFalsify
Undercuts
Rebuts
LimitsScope
DependsOn
CompatibleWith
TransformableTo
NotPoolableWith
Supersedes
Corrects
Retracts
ChallengesLifecycleAuthority
ResolvesDefeater
RequiresAssumption
FailsDiagnostic
LeavesUnresolved
SuggestsDiscriminatingExperiment
```

These are illustrative. The first implementation should not attempt to universalize every domain relation at once.

---

## 23. Suggested SCI-014 implementation split

After #729 and #769, implementation should remain narrow:

```text
SCI-014a  immutable proposition identity
SCI-014b  evidence contribution + lifecycle primitives
SCI-014c  disposition input snapshot / generation binding
SCI-014d  reason-topology receipt
SCI-014e  bounded disposition projection under one registered policy
SCI-014f  replay/currentness verification
SCI-014g  time-indexed storage/query projection
```

Only after this should a broader interactive Theory Atlas UI or automated experiment-planning loop consume disposition assessments.

---

## 24. Qualification adversarial cases

A future focused qualification should include at least:

1. same proposition + identical exact inputs/policy yields byte-identical disposition/reason topology;
2. changing policy identity yields a new assessment even when the primary label stays equal;
3. changing lifecycle/adjudication generation makes the predecessor assessment stale;
4. excluded evidence remains auditable and cannot silently disappear;
5. excluded evidence is not converted to opposition;
6. one active undercutting defeater can block an inference without creating support for the negation;
7. contradictory admissible evidence can yield `Contested` rather than forced winner;
8. incomplete dependency inventory cannot be promoted to independent replication;
9. `NotTriggered` falsifier does not become support/truth;
10. later correction/retraction does not alter an earlier historical assessment;
11. raw support counts cannot bypass dependency/compatibility policy;
12. a caller cannot mint `SupportedWithinScope` without the exact reason topology and policy path;
13. a stale cached disposition cannot pass currentness verification;
14. primary disposition can be reproduced from the retained immutable inputs;
15. disposition exposes no governance/effect authority.

---

## 25. Important non-claims

This contract does not define universal scientific truth, universal Bayesian priors, universal evidence weights, universal falsification thresholds, one correct disposition policy, legal/research governance, action authority, or a production Theory Atlas database.

It only freezes the requirement that a scientific disposition be a **bounded, replayable, policy-bound projection over a complete auditable reason topology**, with contradictions, exclusions, nulls, defeaters, dependencies, lifecycle changes, and unresolved questions preserved rather than silently compressed away.
