# Symthaea Scientific Research Authority Contract v1

**Status:** architecture freeze / no scientific authority

**Program ID:** `SCI-000R`

**Base:** `main@bf0d394a28baf5f7359b4a7986e6a31ea9416083`

## Purpose

Freeze the authority boundary for future scientific and mathematical research infrastructure before introducing a shared research kernel.

This document does not change any current scientific result, mathematical claim, cosmology result, benchmark result, causal conclusion, or qualification status. It defines the vocabulary and invariants that future executable research components must preserve.

The goal is not to make scientific claims easy to produce. The goal is to make it difficult for a discovery, execution, replication, or verification artifact to silently acquire more authority than its evidence supports.

## Governing theorem

```text
generation != evidence

evidence != execution

execution != qualification

qualification != replication

replication != independence

agreement != correctness

correlation != causation

simulation != observation

finite verification != proof

formal proof != correct formalization
```

No downstream adapter, UI, report, agent, benchmark, solver, or orchestration layer may collapse these distinctions into one undifferentiated confidence score.

## Core model

Every authority-bearing scientific object must separate at least four concepts:

1. **Subject identity** — what proposition, model, protocol, dataset, theorem, or experiment is being discussed.
2. **Evidence identity** — what observations, artifacts, executions, reviews, proofs, or replications bear on that subject.
3. **Qualification lineage** — what exact review/evaluation process grants a bounded authority claim over that evidence.
4. **Authority facets** — what kinds of statements the qualified object is actually permitted to support.

A stable subject may be requalified under multiple independent lineages without changing the subject identity. A new qualification lineage does not inherit authority from an older lineage unless that transfer is explicitly represented and valid.

## Authority is multidimensional

Scientific authority must not be represented as a single scalar.

The shared research kernel should support independent facets such as:

- **Provenance authority** — whether source identity and lineage are bound.
- **Execution authority** — whether the declared computation or experiment actually executed under the bound environment.
- **Empirical authority** — whether observations support the stated empirical relationship.
- **Causal authority** — whether the evidence design supports causal language.
- **Formal authority** — whether a formal checker authenticated the exact formal statement under a bound environment.
- **Replication authority** — whether the claim survived a separately identified replication attempt.
- **Independence authority** — how much correlated failure surface remains between supporting evidence lineages.

A result can be strong on one facet and weak or inapplicable on another.

Examples:

```text
Lean kernel acceptance
    -> may provide formal authority
    -> does not by itself provide formalization-correctness authority

independent numerical implementation over the same released dataset
    -> may improve implementation-independence authority
    -> does not establish independent data reduction

randomized intervention
    -> may support causal authority
    -> does not establish independent replication
```

## Required evidence-state distinctions

Shared research APIs must preserve explicit states rather than infer success from absence of failure.

At minimum:

```text
Pass
Fail
Unknown
Incomplete
NotApplicable
Invalid
```

`Unknown`, `Incomplete`, and `NotApplicable` are not aliases for `Pass`.

A missing uncertainty term is not zero uncertainty. A missing replication is not a failed replication. An invalid execution is not a negative scientific result.

## Claim authority ceiling

Every evidence object and qualification result must have a bounded claim ceiling.

The ceiling must answer:

- what statements this object may support;
- what statements it explicitly may not support;
- which upstream identities it depends on;
- which authority facets remain absent;
- which failures invalidate the object rather than count against the hypothesis.

A downstream composition may never silently raise an upstream claim ceiling.

Conceptually:

```text
authority(compose(a, b)) <= explicitly justified authority from a + b
```

No adapter is allowed to promote a structural declaration into authenticated execution authority, a simulation into empirical observation, or a repeated execution into independent replication.

## Scientific subject classes

The first shared research kernel should be general enough to represent subjects including:

- empirical hypotheses;
- causal hypotheses;
- mathematical conjectures;
- formal theorem statements;
- physical models;
- statistical models;
- datasets;
- experimental protocols;
- numerical reproduction targets;
- scientific measurements;
- benchmark tasks;
- literature-derived claims.

Domain-specific types remain free to impose stronger invariants.

## Evidence classes

The shared model should distinguish evidence such as:

- raw observation;
- calibrated observation;
- derived measurement;
- deterministic computation;
- numerical simulation;
- symbolic derivation;
- bounded verification;
- counterexample;
- formal proof;
- formal proof rejection;
- causal study;
- replication study;
- literature evidence;
- human review;
- semantic/formalization review;
- negative control;
- ablation;
- robustness test;
- out-of-distribution test.

These evidence classes do not share one universal ordering.

## Independence is not binary

Future research infrastructure must treat independence as a structured property rather than a boolean.

Useful dimensions include:

```text
SameExecutionReplay
IndependentExecutionSameBinary
IndependentImplementationSharedInputs
IndependentMethodSharedRawData
IndependentDataReduction
IndependentDataset
IndependentSite
IndependentOrganization
```

More than one dimension may apply simultaneously.

Two result files with different names are not independent evidence. Two implementations sharing the same raw dataset are more independent than repeated executions of the same binary, but less independent than separate data acquisition and reduction.

The system must preserve shared roots in the evidence graph so a report cannot double-count correlated evidence as independent confirmation.

## Provenance graph

Future shared science infrastructure should maintain an acyclic provenance graph for authority-bearing derivations:

```text
source/data
    -> preprocessing
    -> analysis/execution
    -> observation/measurement
    -> evidence
    -> claim evaluation
    -> qualification
```

Every edge must have typed semantics.

The graph must fail closed on:

- cycles in authority-bearing provenance;
- unknown referenced identities;
- subject substitution;
- qualification-lineage substitution;
- orphan authority-bearing evidence;
- digest mismatch;
- ambiguous predecessor identity;
- silent evidence replacement.

Scientific relations such as `supports`, `contradicts`, or `alternative-to` may live in a separate relation graph and need not be acyclic.

## Preregistration boundary

Confirmatory studies should bind their protocol before protected observations are consumed.

A frozen protocol should be able to bind, where applicable:

- falsifiable hypothesis;
- null hypothesis;
- alternative hypotheses;
- independent variables;
- dependent variables;
- primary endpoint;
- secondary/exploratory endpoints;
- predicted direction;
- predicted magnitude;
- statistical test;
- alpha / interval policy;
- sample target;
- stopping rule;
- data cutoff;
- controls;
- falsification conditions;
- model/checkpoint identity;
- verifier identity;
- analysis-plan identity;
- resource budget;
- allowed deviations;
- prohibited deviations.

Changing an authority-bearing frozen field requires a new protocol identity. Historical protocol text and evidence must not be overwritten.

## Falsification asymmetry

The research architecture must treat falsification conservatively.

```text
valid falsifier finds a failure
    -> claim authority may decrease

valid falsifier finds no failure
    -> claim authority does not automatically increase to proof
```

A falsification plan may include:

- null models;
- negative controls;
- placebo tests;
- shuffled controls;
- ablations;
- boundary-value attacks;
- parameter perturbations;
- alternative mechanisms;
- alternative preprocessing;
- model-family substitution;
- sensitivity analysis;
- distribution shift;
- cross-dataset generalization.

## Uncertainty boundary

Future shared research representations should preserve separate uncertainty sources where applicable:

```text
measurement
sampling
aleatoric
epistemic
parameter
numerical
structural
model_form
distribution_shift
formalization
provenance
```

Unknown uncertainty must remain unknown. It must not be silently encoded as `0`, omitted from an aggregate, or converted into a stronger confidence statement.

## Mathematics boundary

The existing mathematical research train remains authoritative for mathematical specification and formal verification.

Future science infrastructure must preserve this separation:

```text
mathematical subject identity
    != specification qualification identity
    != proof-search state
    != generated Lean artifact
    != authenticated Lean execution
    != semantic/formalization correctness
```

The Conjecture Engine may generate candidate claims and candidate formalizations. It must not be able to mint qualification-bound mathematical authority by itself.

A finite numerical check, bounded SMT evaluation, or sampled equality test must never be represented as universal proof authority.

## Physics boundary

Physics discovery and recognition may use permissive heuristics, but authority-bearing validation must fail closed.

In particular:

- unknown units are not dimensionless evidence;
- dimensional inconsistency is not dimensionless validity;
- catalog similarity is not physical correctness;
- solver agreement is not correctness without independence and model-contract checks;
- numerical convergence is not empirical validation;
- a simulation is not an observation.

A future physical-model contract should be able to bind dimensions, units, parameter regime, conservation laws, invariants, symmetries, positivity constraints, causality constraints, initial/boundary conditions, limiting behavior, and numerical convergence requirements.

## Autonomous research boundary

Autonomy comes after authority infrastructure.

A future autonomous research loop may orchestrate:

```text
observe
  -> hypothesize
  -> formalize
  -> design experiment
  -> execute
  -> falsify
  -> reproduce
  -> qualify
  -> generate new questions
```

but orchestration itself grants no scientific authority.

An LLM, HDC module, LTC module, search policy, or planning agent may propose artifacts. Durable evidence and qualification should preferentially flow through deterministic, replayable artifacts rather than require an opaque model invocation for reproduction.

## Compatibility with existing lines

This contract intentionally does not rewrite or reinterpret existing frozen evidence lines.

### Mathematics

The current `MATH-SPEC`, `MATH-EVID`, `MATH-STATE`, `MATH-VERIFY`, and related qualification work remains unchanged. Future shared-science adapters must consume their public bounded outputs without weakening their authority boundaries.

### DE-001A cosmology

The current dark-energy/cosmology chain remains unchanged. A future generic evidence-graph adapter may represent its existing receipts read-only, but must preserve distinctions such as:

```text
implementation-independent != independent data reduction
numerical convergence != scientific confirmation
bundle integrity != scientific truth
scientific_claim=NONE remains NONE
```

### Existing domain-local scientific validation

Domain-local validation such as the aesthetic scientific-validation and causal layers should initially be adapted into the generic kernel rather than rewritten wholesale. Migration must preserve or lower authority, never raise it.

### Spark experiment design

Existing expected-information-gain experiment design should be generalized through compatibility adapters rather than replaced with a weaker heuristic.

## First implementation train

The intended initial sequence is:

```text
SCI-000R  this architecture contract
    ↓
SCI-001A  dependency-light science-research kernel
    ↓
SCI-001AQ exact-head source qualification
    ↓
SCI-002A  typed provenance + independence DAG
    ↓
SCI-002AQ exact-head graph qualification
    ↓
SCI-003A  domain ScientificClaim compatibility adapter
```

Later work may add preregistration, replication, experiment design, falsification, uncertainty, physics contracts, solver federation, math adapters, research benchmarks, and autonomous orchestration.

## SCI-001A minimum surface

The first executable crate should remain intentionally small and deterministic.

Suggested modules:

```text
identity.rs
authority.rs
independence.rs
subject.rs
evidence.rs
claim.rs
```

It should contain no network access, LLM calls, HDC execution, solver execution, database access, or domain-specific scientific policy.

Its responsibility is only to make unsafe authority conflation difficult to represent.

## Qualification requirements for the first executable kernel

Before the new kernel can be treated as qualified infrastructure, a dedicated exact-head qualifier should establish at least:

- exact parent and source scope;
- pinned Rust/Cargo toolchain;
- locked compile/test/strict Clippy where the workspace lock permits;
- canonical digest vectors;
- malformed deserialization rejection;
- subject/qualification substitution rejection;
- independence classification tests;
- no authority promotion through generic adapters;
- postflight source immutability;
- retained qualification receipt;
- `scientific_claim=NONE`.

## Nonclaims

SCI-000R does not establish that:

- Symthaea has discovered any new scientific fact;
- any existing scientific hypothesis is true or false;
- any current cosmology result is qualified;
- any current mathematical conjecture is proved;
- any current causal claim is valid;
- any solver is independent;
- any existing evidence is replicated;
- the future architecture is complete;
- autonomous research is safe or scientifically trustworthy.

It freezes only the authority vocabulary and migration constraints for the executable work that follows.
