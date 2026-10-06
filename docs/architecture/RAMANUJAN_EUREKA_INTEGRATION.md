# Ramanujan × EUREKA Integration

Status: Architecture RFC
Related: #2045, #6868, #6869, #6871, #6874

## Thesis

Symthaea should combine Ramanujan and EUREKA at the scientific workflow level, not collapse them into one search algorithm.

Ramanujan is the hypothesis-generation / mathematical-discovery loop.
EUREKA is the independent challenge / qualification / experiment-selection loop.

## Dual-loop architecture

```text
observations
    -> Ramanujan discovery
    -> PhysicalType gate
    -> typed hypothesis
    -> EUREKA challenge
    -> hidden-world / simulator / experiment
    -> independent holdout evidence
    -> symbolic / formal qualification
    -> EUREKA result
    -> model revision
    -> new campaign
```

## Why not merge the mechanisms

A single component that generates a hypothesis, selects its confirmatory evidence, and judges the outcome creates a self-referential validation problem.

Keep roles separate while sharing canonical semantics and provenance.

## Canonical handoff

Ramanujan should emit a typed scientific-model candidate carrying, as applicable:

- candidate expression/model digest;
- physical type judgment;
- model maturity;
- source observation identity/digest;
- search grammar/configuration digest;
- search seed;
- discovery initial-condition manifest;
- candidate complexity;
- symbolic status;
- recognition/analogy status;
- explicit non-claims.

EUREKA consumes this candidate as a hypothesis, never as evidence.

EUREKA independently selects or validates holdout consequences, interventions/counterfactuals, transfer tests, active experiments, baselines, and replication requirements.

## Anti-circularity theorem

```text
Ramanujan proposal != EUREKA qualification
Ramanujan confidence != EUREKA evidence
catalog similarity != validation
solver convergence != physical truth
symbolic identity != empirical validity
EUREKA outcome != Ramanujan search-objective feedback
```

EUREKA HeldOut / External outcomes must not enter the original Ramanujan search ranking, grammar, comparator selection, or objective.

Only after a campaign is closed may its result become an explicit model-revision input for a new campaign.

## Why EUREKA is unusually valuable for Ramanujan

Low residual error is necessary but weak evidence for a scientific model.

EUREKA adds the questions that mathematical curve fitting alone cannot answer:

- Does the model predict fresh consequences?
- Does it survive interventions or perturbations?
- Does it transfer across surface domains?
- Does it choose informative experiments prospectively?
- Does it revise when contradicted?
- Does it outperform shortcut baselines?

That turns Ramanujan from a formula finder into the hypothesis-generation stage of a scientific discovery loop.

## Physics pipeline

```text
ODE/PDE/telemetry
 -> Ramanujan invariant/equation discovery
 -> PhysicalType
 -> independently frozen candidate
 -> EUREKA discriminative experiment
 -> simulator / digital twin
 -> fresh initial conditions
 -> intervention / perturbation
 -> holdout residuals
 -> symbolic check
 -> formal proof where supported
 -> transfer / replication
 -> engineering qualification only within the declared model envelope
```

## Active inquiry

Ramanujan can produce competing models H1, H2, and H3.

EUREKA should be able to select an experiment that is predicted to distinguish them, but the selection itself is not evidence.

The proper loop is:

```text
predicted disagreement score
 -> experiment
 -> observed outcome
 -> evidence
 -> model update
```

This preserves the boundary between prospective challenge selection and realized evidence.

## Concept Genesis

Ramanujan's compact formulas can become candidate abstractions, but naming or compressing training data is not sufficient to establish a concept.

A promoted abstraction should improve a preregistered combination of held-out compression, prospective prediction, intervention prediction, structural transfer, sample efficiency, active inquiry, and nuisance robustness.

## Model maturity

Keep maturity independent from discovery status:

```text
TextbookAnalytic
ValidatedNumerical
CalibratedEmpirical
ResearchPrototype
FrontierHypothesis
SyntheticInstrumental
```

A frontier hypothesis may be scientifically valuable while remaining outside an engineering safety gate.
A mature engineering model may be highly qualified without being novel.

Therefore:

```text
novelty != maturity
maturity != truth
truth claim != authority
```

## Physical units and temperature semantics

The shared PhysicalType layer distinguishes absolute thermodynamic temperature from a temperature difference. They share the same SI base dimension but are different quantity kinds.

```text
Temperature - Temperature -> TemperatureDifference
Temperature + TemperatureDifference -> Temperature
TemperatureDifference + TemperatureDifference -> TemperatureDifference
TemperatureDifference - Temperature -> INVALID
```

Affine offsets such as Celsius/Fahrenheit-to-Kelvin are valid for absolute temperature units, but not for a temperature difference. Delta units must use a zero offset and an appropriate scale.

Semantic compatibility and executable numeric transport are also separate:

```text
semantic compatibility
    -> same physical meaning / dimension

numeric transport compatibility
    -> semantic compatibility
    -> explicit units on both sides or neither
    -> explicit conversion when units differ
```

A solver boundary must therefore never infer a numeric unit merely because the dimensions match.

## Canonical prediction frame

Active inquiry has one additional requirement beyond typed connectivity: predictions must be compared in one explicitly declared physical frame.

```text
hypothesis prediction
    -> source PhysicalType
    -> explicit conversion
    -> canonical prediction frame
    -> finite-value check
    -> disagreement score
```

The strict selector rejects the challenge when any live hypothesis cannot produce a finite value that is physically convertible into the frame. This prevents unit-scale artifacts such as `1000 J` versus `1 kJ` from being mistaken for scientific disagreement.

The inquiry receipt binds the prediction-frame digest as well as the hypothesis-set, challenge-space, and selected-challenge identities.

Execution is a separate provenance event:

```text
ScientificInquirySelectionReceipt
    -> ScientificInquiryExecutionReceipt
    -> existing observation/evidence artifact
```

The execution receipt binds the exact selection receipt, selected challenge, prediction frame, execution manifest, observation artifact, observation physical type, and evaluator revision. It does not itself declare the scientific result true, false, novel, or qualified.

## Formal verification boundary

Formal proof establishes a theorem about the formalized model when the theorem and implementation are correctly bound.

It does not by itself establish that the model describes the physical world.

Ramanujan supplies candidate mathematics; EUREKA supplies independent empirical/operational challenge.

## Evidence architecture

Do not create a second generic EUREKA result system.

Prefer the existing evidence/research-result infrastructure:

```text
existing evidence infrastructure
  + Ramanujan discovery evidence
  + EUREKA qualification evidence
  + simulation evidence
  + formal evidence
  + engineering evidence
```

The handoff binds identities and provenance but preserves each stage's independent evidence lineage.

## Recommended implementation order

1. Stabilize #6869 PhysicalType semantics.
2. Define a canonical Ramanujan hypothesis envelope using existing evidence/result infrastructure. The initial AST-free `ScientificHypothesisHandoff` prototype now lives in `symthaea-types`.
3. Add a one-way Ramanujan -> EUREKA handoff receipt. The first `ScientificHypothesisHandoff` implementation is now available in `symthaea-types` and is emitted directly from `Conjecture` without consulting EUREKA.
4. Make EUREKA experiment selection independent of candidate search ranking. The first `ScientificInquirySelectionReceipt` implementation records the exact handoff/set/challenge identities, selector revision, seed, and predicted information value, while remaining explicitly non-outcome.
5. Add physics-specific discriminative experiments.
6. Permit model-revision feedback only into a new Ramanujan campaign.
7. Add cross-domain structural-transfer tests.
8. Build an end-to-end scientific-discovery runner only after those boundaries are qualified independently.

## Architectural success criterion

```text
discover
 -> type
 -> challenge
 -> experiment
 -> observe
 -> explain
 -> predict
 -> intervene
 -> transfer
 -> revise
 -> replicate
 -> qualify
```

Ramanujan provides the generative engine.
EUREKA provides the adversarial scientific discipline around it.
Neither replaces the other.


Implementation note: the first AST-free handoff envelope is now available as `symthaea_types::discovery_handoff::ScientificHypothesisHandoff`. It binds candidate identity, PhysicalType identity, model maturity, observation/search/discovery provenance, complexity, source, explicit non-claims, and an optional immutable EUREKA challenge-manifest commitment. It contains no EUREKA outcome field.


## Current executable integration slice

The current branch implements the following value-level chain:

```text
Ramanujan Conjecture
   -> exact candidate identity digest
   -> PhysicalType identity digest
   -> ScientificHypothesisHandoff
   -> independent experiment selector
   -> ScientificInquirySelectionReceipt
```

No EUREKA outcome is written into the handoff or selection receipt.

The candidate digest is derived from the exact symbolic expression AST, source identity, domain tag, and complexity rather than the human-readable formula formatter. This avoids collisions caused by display rounding of floating-point constants.

The experiment selector now explicitly preserves first-occurrence tie breaking, and a dedicated regression covers exact ties.

The current workflow therefore has an intentionally one-way evidence boundary:

```text
Ramanujan -> hypothesis identity -> EUREKA challenge
EUREKA outcome -X-> original Ramanujan campaign
```

A later model-revision campaign may consume the closed result as a new input, but it receives a new campaign/search identity. The first `ScientificHypothesisRevisionReceipt` implementation enforces this at the value level by requiring distinct prior/new handoff digests and a distinct campaign digest.

The current `ScientificInquirySelectionReceipt` implementation freezes which hypothesis set, challenge space, and canonical prediction frame produced an experiment-selection decision. Its disagreement score is a selection heuristic, not realized evidence; the actual experiment must be evaluated separately.
