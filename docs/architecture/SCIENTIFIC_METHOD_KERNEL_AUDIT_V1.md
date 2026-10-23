# Scientific Method Kernel Audit v1

**Status:** architecture audit only; non-authorizing; non-qualifying.

**Base:** `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`

## 1. Purpose

Symthaea now contains multiple independently developed scientific-evidence boundaries across discovery, neuroscience, economics, futures, matter/nuclear physics, ALife, physical agency, causal reasoning, and recursive cognition.

Those domains repeatedly enforce the same high-level theorem:

```text
valid artifact
    != observation
    != interpretation
    != scientific claim
    != evidentiary support
    != independent replication
    != execution authority
```

The immediate architectural risk is no longer absence of scientific rigor. It is **semantic duplication**: each domain may continue inventing neighboring representations for provenance, preregistration, execution identity, evidence dependency, uncertainty, claim status, and replication semantics.

This audit defines what may eventually be generalized into a Scientific Method Kernel and, equally importantly, what must **not** be collapsed merely because two objects sound similar.

This document does not introduce a production `symthaea-science-kernel` crate, migrate any consumer, change any evidence status, or transfer qualification from any referenced PR.

---

## 2. Kernel theorem

A future common kernel should preserve at least these distinct semantic layers:

```text
ArtifactIdentity
    -> Observation
    -> Interpretation
    -> Hypothesis / Claim
    -> ExperimentContract
    -> ExecutionEvidence
    -> MeasurementEvidence
    -> Adjudication
    -> EvidenceContribution
    -> Dependency / Replication Analysis
    -> Scientific Disposition
```

with explicit non-equivalences:

```text
content digest syntax          != verified bytes-to-digest equality
verified bytes                 != trustworthy source
closed derivation graph        != verified transform execution
simulation execution           != experiment
experiment                     != successful outcome
successful outcome             != safety
numerical fit                  != formal proof
no detected violation          != assumption true
multiple implementations       != independent replication
multiple publications          != independent evidence lineages
historical availability        != live possession
high confidence                != scientific authority
scientific authority           != action authority
```

No common scalar `confidence: f64` should be allowed to erase these distinctions.

---

## 3. Existing domain evidence

### 3.1 Conjecture Engine / Ramanujan Protocol

Current main already distinguishes evidence states for conjectures:

```text
Proposed
NumericallyTested
BoundedChecked
SmtSamplesChecked
SymbolicallyChecked
FormallyVerified
Refuted
```

This is a strong seed for a general rule:

> finite observations, solver-assisted sample checking, symbolic identities, and proof-producing verification are different evidence classes.

The active experiment selector also already treats multiple live hypotheses as competing predictive objects and chooses candidate experiments by predictive disagreement. That selector is a useful baseline, not yet a complete decision-theoretic experiment planner.

### 3.2 Web research / conjecture feed

`ResearchProvenance` retains source URL, claim text, source type, epistemic status, confidence, supporting sources, and contradicting sources. Low-trust research data can quarantine downstream macro promotion.

This is valuable, but current web verification uses heuristic source/domain credibility and should not become the canonical scientific-verification authority without a separate qualification effort.

### 3.3 Evidence Plane

The evidence-plane work establishes an especially important general pattern:

```text
declared mechanism
    != measured execution
```

Ablations and metric implementations must be evidenced where the mechanism actually executes, rather than trusted from configuration labels.

The future kernel should preserve this distinction as a reusable execution/measurement theorem rather than a neuroscience-specific convention.

### 3.4 NeuroBridge / execution capsules

The NeuroBridge stack is converging on:

```text
source selection
+ exact input bytes
+ exact runtime closure
+ fixed process environment
+ platform
    -> execution-capsule identity
```

and separately:

```text
execution-capsule identity
    != observed execution
    != independently verified receipt
    != scientific qualification
```

This is the leading candidate for a future generic `ScientificExecutionCapsuleV1`, but no migration should occur until the Workbench lineage itself qualifies and the generic contract can be proven without weakening the neuro-specific one.

### 3.5 Matter / nuclear science

The Matter Observatory stack introduces several important general boundaries:

```text
matter scale
    != validation stage
    != solver capability
    != uncertainty
    != epistemic authority
```

The nuclear blind-validation work adds further reusable principles:

```text
training membership must be explicit
training values/configuration belong to model lineage
calibration must be disjoint from fitting
structural OOD holdout coverage != conformal guarantee
model disagreement != consensus
shared ancestry != independent confirmation
preregistered metric bounds != universal scientific validity
```

These should inform generic kernel semantics, but domain-specific statistical assumptions such as exchangeability must remain domain-owned.

### 3.6 Economic Science

The economics qualification branches strongly separate:

```text
theory variable
    != measurement specification
    != measurement evidence

sampling design
    != sample size
    != representativeness

reported point
    != exact realized state

causal estimand
    != identification strategy
    != estimator
    != estimate

identification assumption
    != diagnostic result
    != assumption truth

publication count
    != evidence-lineage count
    != verified independent replication
```

This is one of the strongest arguments against a universal scalar evidence score.

The future kernel should be able to carry these typed dimensions without claiming that one domain's causal or measurement vocabulary applies unchanged to another.

### 3.7 Futures

The Futures work contributes general-purpose provenance/temporal theorems:

```text
canonical digest syntax
    != bytes-to-digest verification

provenance inventory
    != closed derivation topology

closed derivation topology
    != verified transform execution

semantic/reference time
    != source availability time
    != local acquisition/custody time

historical availability
    != live prospective possession

resolution linkage
    != preregistered outcome definition
```

These are directly relevant to historical-discovery benchmarks and real-world scientific data ingestion.

### 3.8 Physical Agency

The strict simulation stack provides an unusually good template for confirmatory science:

```text
selected candidate
    -> preregistered outcome claim
    -> preregistered safety obligations
    -> exact context-bound solver execution
    -> outcome adjudication
    -> strict simulation qualification
```

and preserves:

```text
solver result != successful claimed outcome
successful simulated outcome != safety
simulation qualification != physical execution authority
```

The preregistration pattern should inform a generic `ExperimentContractV1`, but physical safety/actuation authority must remain outside the scientific kernel.

### 3.9 Recursive cognition / epistemics

The RCA stack contributes the strongest evidence-lineage and disposition separation:

```text
legacy graph label
    != canonical evidence-lineage generation identity

individually valid evidence artifacts
    != coherent joint evaluation input

coherent joint input
    != disposition

disposition
    != canonical belief
    != workspace authority
    != action authority
    != self-improvement promotion
```

A future Theory Atlas should reuse these ideas rather than inventing a second evidence topology.

### 3.10 ALife / Genesis

The ALife observatory correctly refuses to infer lifecycle facts absent from the event stream, while the perturbation substrate creates deterministic controlled interventions.

That pair is a promising scientific laboratory boundary:

```text
observed event stream
    -> analysis-only observatory
    -> candidate causal mechanism
    -> controlled perturbation
    -> new observation stream
    -> causal update
```

The future kernel should support this loop without embedding Genesis-specific event semantics.

---

## 4. Concepts that are candidates for generalization

The following concepts appear repeatedly enough to justify a future common contract.

### 4.1 Content-addressed artifact identity

Candidate common semantics:

```text
ArtifactRef {
    namespace
    artifact_id
    digest_algorithm
    digest
    byte_len?
}
```

Required theorem:

```text
ArtifactRef syntax != verified content admission
```

Do not migrate until neutral digest computation and bytes-to-digest verification have their own qualified boundary.

### 4.2 Scientific execution identity

Candidate common semantics:

```text
ScientificExecutionCapsuleV1 {
    source_identity
    dependency/runtime_closure_identity
    toolchain_identity
    process_environment_identity
    platform_identity
    input_snapshot_identity
    configuration_identity
    rng_identity
    external_tool_identities
}
```

Required theorem:

```text
capsule identity != execution receipt != scientific qualification
```

### 4.3 Preregistered experiment contract

Candidate common semantics:

```text
ExperimentContractV1 {
    target_claim
    hypothesis_set
    admissible inputs
    intervention / treatment semantics
    outcome definitions
    measurement specification
    analysis implementation identity
    decision rules / thresholds
    stopping rule
    missing-data policy
    preregistration evidence identity
}
```

The contract must be immutable before confirmatory outcome observation.

Required theorem:

```text
exploratory result != confirmatory evidence
```

### 4.4 Evidence contribution

Candidate common semantics:

```text
EvidenceContribution {
    target_claim_id
    relation
    evidence_artifact_ids
    execution_receipt_id?
    measurement_evidence_id?
    assumptions
    limitations
    lineage_inventory
}
```

Possible relation vocabulary:

```text
Supports
Opposes
Falsifies
FailsToFalsify
ReplicatesWithinDeclaredScope
Supersedes
Retracts
```

Do not use `independent: bool`.

### 4.5 Evidence dependency inventory

Generalize the economics/RCA direction into typed dependency domains such as:

```text
SourceData
SourceVintage
MeasurementSpecification
SamplingDesign
TransformationArtifact
ModelArtifact
EstimatorArtifact
IdentificationStrategy
ExecutionCapsule
OutcomePolicy
EvaluationProtocol
LearnedGrammarArtifact
```

The strongest generic no-overlap claim should remain scoped, e.g.:

```text
DeclaredDisjointWithinScope
```

not:

```text
IndependentReplication
```

unless a separate authority has qualified inventory completeness and independence semantics.

### 4.6 Scientific claim / disposition coordinates

A future claim should carry multiple typed coordinates rather than one total score.

Illustrative dimensions only:

```text
provenance state
execution state
measurement state
uncertainty/calibration state
causal-identification state
replication/dependency state
novelty-search state
temporal/preregistration state
```

No v1 total ordering is proposed.

---

## 5. Concepts that must remain distinct

### 5.1 Observation vs interpretation

Raw observation and downstream interpretation must retain separate identities.

### 5.2 Hypothesis vs claim

A generated explanatory candidate is not automatically a publication-grade claim.

### 5.3 Exploratory vs confirmatory evidence

An exploratory process that has seen an outcome may not create a confirmatory experiment contract for that same outcome and claim prospective credit.

### 5.4 Uncertainty vs epistemic authority

A calibrated interval or posterior describes an uncertainty model; it does not itself establish source trust, experimental status, causal identification, or independent replication.

### 5.5 Causal structure vs causal identification

A DAG edge, text-extracted causal relation, or learned structural relation is not an identified causal effect.

### 5.6 Scientific evidence vs safety/effect authority

No scientific evidence object may silently become an actuation permit, safety authorization, execution capability, or self-improvement promotion capability.

### 5.7 Replication vs independence

Different code, model architecture, author, institution, or paper identity is insufficient to establish independent replication.

### 5.8 Confidence vs disposition

Confidence values may remain local statistical/model quantities. They must not act as universal scientific-disposition authority.

---

## 6. Discovery-engine-specific hardening

Before the Conjecture Engine becomes a primary consumer of a future kernel, several repairs are desirable.

### 6.1 Status-report exhaustiveness

The discovery-health/reporting path must be ratcheted against every `ConjectureStatus` variant so newly introduced evidence states cannot silently disappear from reporting or make examples stale.

### 6.2 Uncertainty-bearing observations

The current `(x, y)` `ObservedSequence` should eventually gain an additive richer observation path carrying measurement uncertainty, method identity, provenance, and sample identity.

The old tuple API may remain as an explicitly simple/exact adapter.

### 6.3 Experiment selection baseline vs advanced policy

Prediction variance should remain a deterministic baseline.

A later experiment-design API may add, without hiding dimensions:

```text
expected information gain
falsification power
cost
risk
duration
novelty
```

Selection should preferably expose a Pareto frontier rather than requiring a universal scalar utility.

### 6.4 Symbolic-discovery ensemble

The architecture should permit multiple discovery backends:

```text
GeneticProgramming
ParallelEnumeration
SparseRegression
ResidualRepair
NeuralProposal
PhysicsPrior
```

All backends should be adjudicated against the same frozen observations/holdouts and evidence contract.

### 6.5 Learned grammar provenance

Dynamic grammar/macros learned from previous discoveries are scientific dependencies.

A downstream discovery using a learned primitive must inherit the originating discovery/dataset/verification lineage. Otherwise apparent independent rediscovery can be contaminated by prior learned structure.

Candidate future primitive classes:

```text
AxiomaticBase
ProvisionalLearned
VerifiedLearned
Quarantined
```

---

## 7. Causal-science hardening

### 7.1 Unify causal graph representations deliberately

The repository currently contains more than one causal-DAG representation. A future unification must preserve all semantics required by:

```text
graph topology
d-separation
identification
intervention
counterfactual reasoning
relation provenance
assumptions
confidence / uncertainty
```

Do not collapse them solely to remove duplicate structs.

### 7.2 Cycle protection

Any type named/used as a DAG should reject structural updates that create directed cycles unless a different cyclic-graph model is explicitly selected.

### 7.3 Relation provenance

Text- or knowledge-derived causal edges should not become unconditional hard causal facts. Edge lineage should retain source, extraction method, confidence/uncertainty, assumptions, and supporting/opposing evidence.

### 7.4 Intervention loop

Genesis provides the strongest near-term environment for a complete causal-discovery loop because interventions are cheap, controlled, and replayable.

---

## 8. Falsification as a first-class scientific object

Every confirmatory claim should eventually be able to name admissible falsifiers.

Candidate semantics:

```text
FalsifierSpecification {
    target_claim_id
    experiment_contract_family
    contradictory_observation_predicate
    scope
    analysis_identity
}
```

The scientific planner should search not only for supportive experiments but for experiments expected to discriminate among competing hypotheses or efficiently falsify them.

A claim surviving a falsification campaign should retain every attack/result as explicit lineage; it should not merely receive a larger scalar confidence.

---

## 9. Historical discovery benchmark direction

A future benchmark should evaluate scientific discovery under historical information constraints.

Protocol shape:

```text
frozen source corpus available by time t0
    -> verified historical availability / custody class
    -> Symthaea discovery process
    -> preregistered candidate result
    -> compare with knowledge first published after t0
```

The benchmark should never claim "independent rediscovery" if the model, learned grammar, embeddings, or external tools carry post-cutoff knowledge.

All model/tool provenance must therefore be part of benchmark eligibility.

---

## 10. Novelty boundary

No generic `novel: bool` should exist.

A defensible novelty result should be scoped to a search receipt:

```text
NoveltySearchReceipt {
    search_cutoff
    corpus_snapshot_ids
    query / retrieval policy identity
    model / embedding identities
    retrieved candidate prior work
    nearest materially similar results
}
```

The strongest default conclusion should be of the form:

> No materially equivalent prior result was found within the declared search scope.

not:

> First-ever discovery.

---

## 11. Theory Atlas direction

The Theory Atlas should become a scientific argument graph rather than a flat knowledge base.

For each proposition it should preserve:

```text
supporting evidence
opposing evidence
defeaters
assumptions
falsifiers
replication/dependency topology
causal status
unanswered discriminating experiments
superseded versions
retractions
exact artifact/execution provenance
```

Scientific disposition should consume these typed structures without turning publication count, model count, or confidence values into a vote.

---

## 12. Proposed implementation sequence

### SCI-001 — this audit

**Scope:** documentation only.

Freeze common/non-common semantic boundaries before creating shared production types.

### SCI-002 — canonical scientific artifact identity

**Prerequisites:** neutral digest computation + bytes-to-digest verification qualification.

Introduce a narrowly scoped shared artifact identity without source-trust authority.

### SCI-003 — scientific execution-capsule contract

Generalize the qualified NeuroBridge execution identity after proving the common contract does not weaken its existing semantics.

### SCI-004 — `ExperimentContractV1`

Introduce a non-result-bearing preregistration contract for confirmatory computation/simulation/experimentation.

### SCI-005 — exploratory / confirmatory separation

Make it impossible by API shape for an exploratory discovery object to directly mint confirmatory evidence.

### SCI-006 — evidence dependency graph core

Generalize declared dependency inventories and known shared-dependency components. Do not emit `independent: bool`.

### SCI-007 — scientific claim / falsifier graph

Introduce typed evidence relationships and falsifier specifications without a global confidence score.

### SCI-008 — uncertainty-bearing observations

Add a richer observation interface and preserve the legacy `(x, y)` adapter.

### SCI-009 — experiment-design frontier

Retain prediction-disagreement variance as baseline; add information-gain/falsification/cost/risk/duration/novelty dimensions.

### SCI-010 — symbolic-discovery tournament

Create backend-neutral evaluation for GP, enumeration, residual repair, and later additional discovery methods.

### SCI-011 — learned grammar provenance

Bind promoted discovery primitives to their originating evidence dependencies.

### SCI-012 — Genesis causal laboratory

Join observatory evidence, causal candidate generation, controlled perturbation, and causal update.

### SCI-013 — falsification campaign

Standardize negative controls, perturbations, alternative specifications, seed/noise sweeps, OOD tests, and explicit retained results.

### SCI-014 — Theory Atlas v1

Build proposition-centered evidence graphs consuming qualified scientific objects.

### SCI-015 — historical discovery benchmark

Evaluate whether Symthaea could have accelerated past discoveries using only provenance-qualified pre-cutoff knowledge.

---

## 13. Migration rules

1. **No qualification inheritance.** A common abstraction receives no scientific authority merely because one source domain qualified a similar local type.
2. **No semantic weakening.** Migration is permitted only when the common type expresses every domain invariant required at the migrated boundary.
3. **No flag inflation.** Do not replace typed evidence states with booleans such as `verified`, `valid`, `independent`, `causal`, `novel`, or `safe` when the real semantics are multidimensional.
4. **No post-hoc authority.** Confirmatory contracts, thresholds, outcome definitions, and stopping rules must be fixed before protected outcome observation.
5. **No hidden lineage.** Learned models, transforms, grammars, solver profiles, and external tools are dependencies and must be retained where they can affect scientific independence.
6. **No science-to-action shortcut.** Scientific evidence may inform deliberation but cannot itself mint safety, effect, actuator, Forge, or self-improvement authority.
7. **Retain nulls and failures.** Negative, null, inconclusive, indeterminate, contradicted, and refuted outcomes are evidence and must survive reporting.
8. **Prefer scoped statements.** `DeclaredDisjointWithinScope`, `NoDetectedViolation`, `NoExchangeabilityGuarantee`, and similar bounded language are preferred over universal labels.

---

## 14. Exit criteria for SCI-001

SCI-001 is complete when reviewers agree that:

- the listed semantic equivalences/non-equivalences are correct enough to guide shared-type design;
- no domain-specific scientific authority is accidentally generalized;
- the proposed SCI-002..SCI-015 order has explicit prerequisite boundaries;
- the kernel is understood as a small scientific-method substrate, not a replacement for domain science crates;
- future implementation PRs can cite this audit and state exactly which theorem they are materializing.

A green documentation/formatting check, if any, would establish only repository/document integrity. It would not scientifically qualify the proposed architecture.
