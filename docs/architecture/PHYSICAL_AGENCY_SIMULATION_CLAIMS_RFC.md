# Symthaea Physical Agency — Simulation Outcome Claims RFC v0.1

**Status:** Draft architecture RFC  
**Stack:** PA-10, above PA-09 (#562)  
**Execution posture:** simulation-only; no new physical authority

## Purpose

The Physical Agency stack now distinguishes:

- desired physical transitions;
- unqualified mechanism proposals;
- model-ensemble disagreement;
- Pareto deliberation;
- immutable world-snapshot lineage;
- registry-validated external-solver provenance;
- proposal-bound simulation safety cases.

One semantic gap remains before a real physical-domain adapter should be admitted:

> a structurally valid solver result is not automatically evidence that the proposed
> physical effect succeeds.

`SimulationResult::is_engineering_evidence()` establishes that a result is valid,
converged, contains metrics, and carries complete external-solver provenance. It does not
establish that those metrics are the ones needed by a particular `DesiredTransition`, or
that their values satisfy a predeclared success condition.

PA-10 defines the missing claim layer.

The target evidence chain becomes:

```text
DesiredTransition
        |
        v
CandidateAssessment
        |
        +--> SimulationOutcomeClaim
        |       |
        |       +--> typed metric criteria
        |       +--> units
        |       +--> uncertainty policy
        |       +--> aggregation rule
        |
        v
SelectedCandidate
        |
        v
context-bound SimulationRequest
        |
        v
RegistryValidatedSimulationEvidence + SimulationResult
        |
        v
ClaimEvaluation
        |
        +--> SATISFIED
        +--> NOT_SATISFIED
        +--> INDETERMINATE
        |
        v
simulation qualification
```

A claim is predeclared before the solver result is observed. The result does not get to
define its own success criterion after execution.

## Core invariant

```text
ValidSimulationResult != EvidenceOfClaim
```

and:

```text
EvidenceOfClaim != PhysicalExecutionAuthority
```

PA-10 remains entirely on the simulation/evidence side of the architecture.

## Non-goals

PA-10 does not:

- prove that the external solver is honest;
- prove that its numerical model is physically correct outside its validity region;
- turn a simulation success into authority to act on hardware;
- infer success thresholds automatically from observed results;
- permit a free-form natural-language statement to discharge a machine gate;
- define hazardous operational parameters or weapon-specific objectives.

## `SimulationOutcomeClaim`

A selected candidate that is intended to advance through strict simulation qualification
must carry a typed, immutable claim describing what solver evidence would support it.

Conceptually:

```text
SimulationOutcomeClaim {
    schema_version,
    claim_id,
    transition_id,
    proposal_id,
    criteria: Vec<MetricCriterion>,
    aggregation,
}
```

The claim is ordinary serializable planning data. It is not authority.

### Identity binding

`transition_id` and `proposal_id` must exactly match the selected candidate lineage.
Claims cannot be reused across neighboring proposals merely because they ask for the same
metric names.

### Precommitment

The claim must exist before strict solver evidence is evaluated. A result-dependent API
that first reads the output and then constructs thresholds from it would defeat the
purpose of this layer.

The strict path should therefore accept a claim as part of the request/selection lineage,
not as a callback allowed to inspect the result before defining success.

## Metric criteria

The initial schema should remain intentionally small and deterministic.

Conceptually:

```text
MetricCriterion {
    metric_name,
    unit,
    predicate,
    uncertainty_policy,
}
```

### Exact metric identity

A criterion identifies a metric by exact canonical name and exact unit. Silent unit
conversion should not occur inside claim evaluation.

If conversion is necessary, an upstream adapter must normalize the result into the
canonical unit and record that conversion in existing solver warnings/provenance.

### Ambiguity fails closed

For a strict criterion:

- zero matching metrics => `INDETERMINATE` / fail qualification;
- more than one matching metric with the same canonical selector => ambiguous, fail closed;
- non-finite metric value => invalid evidence;
- empty name/unit => invalid claim/evidence.

The evaluator must not silently select the first matching metric.

## Initial predicates

The first implementation needs only deterministic scalar predicates:

```text
AtLeast(x)
AtMost(x)
InsideClosedInterval { lower, upper }
OutsideOpenInterval { lower, upper }
```

Thresholds must be finite. Interval bounds must be ordered.

Equality against floating-point engineering results should not be a first-class strict
predicate. If equality-like behavior is needed, it should be represented as an explicit
accepted interval.

## Uncertainty-aware evaluation

Point estimates alone can make a marginal result appear stronger than its evidence.
Claim evaluation should therefore make uncertainty handling explicit.

Conceptually:

```text
MetricUncertaintyPolicy {
    RequireInterval,
    AllowPointEstimate,
}
```

`RequireInterval` is the conservative default for strict qualification.

### Conservative interval semantics

For a result metric with confidence/uncertainty interval `[L, U]`:

```text
AtLeast(x)     satisfied iff L >= x
AtMost(x)      satisfied iff U <= x
Inside[a,b]    satisfied iff L >= a && U <= b
Outside(a,b)   satisfied iff U <= a || L >= b
```

If only part of the interval satisfies a predicate, the result is `INDETERMINATE`, not
satisfied.

This intentionally distinguishes:

```text
point estimate supports claim
```

from:

```text
uncertainty envelope supports claim
```

### Point-estimate policy

Some exploratory simulations may not expose a metric interval. `AllowPointEstimate` may
be useful for research comparison, but the resulting claim evaluation must retain that
weaker evidence tier.

A later physical-execution architecture must be free to reject point-only claim evidence
even if simulation-only research qualification accepts it.

## Three-valued claim result

Do not force every result into pass/fail.

Use:

```text
ClaimCriterionResult {
    Satisfied,
    NotSatisfied,
    Indeterminate,
}
```

`Indeterminate` covers cases such as:

- required metric absent;
- uncertainty interval absent when required;
- interval straddles the threshold;
- duplicate/ambiguous metric selector;
- unsupported schema/predicate;
- insufficient evidence tier.

This is epistemically more honest than mapping missing evidence to `false` while losing
why it failed.

For strict qualification, both `NotSatisfied` and `Indeterminate` fail closed.

## Claim aggregation

Initial strict qualification should support only:

```text
AllCriteria
```

Every required criterion must be `Satisfied`.

Avoid adding arbitrary boolean expression trees, weights, or threshold-count voting in
v0.1. Those mechanisms make it too easy to weaken a safety/physical-success contract by
configuration complexity.

If richer logic is later needed, it should be introduced as a separately versioned claim
schema with explicit tests.

## Claim evaluation receipt

Successful evaluation should produce a non-serializable runtime receipt rather than a
caller-constructible boolean.

Conceptually:

```text
SatisfiedSimulationClaim {
    claim_id,
    transition_id,
    proposal_id,
    simulation_request_id,
    request_lineage_digest,
    output_digest,
    evaluated_criteria,
    evidence_tier,
}
```

Only the evaluator constructs this type.

This receipt becomes an input to strict Physical Agency simulation qualification.
Serialized bytes must not mint it.

The evaluation should retain each criterion's observed value/interval so later audit can
answer not merely "pass" but **why** it passed.

## Bind claim evaluation to exact solver lineage

The claim receipt must include the exact simulation identities already being hardened by
PA-09:

```text
simulation request id
normalized request lineage digest
solver output digest
context set / world snapshot lineage
```

Therefore this must fail:

```text
claim evaluated against run A
        +
safety/evidence from run B
        -> qualification
```

Even if both runs happened to return numerically identical values.

## Interaction with `CandidateAssessment`

`CandidateAssessment::proposal.predicted_outcome.success_probability` remains a model-side
prediction used for deliberation.

It must not be confused with a solver-backed success claim.

```text
predicted success probability
        = model forecast used before evidence

SatisfiedSimulationClaim
        = post-solver evidence that predeclared criteria were met
```

A useful future benchmark can measure calibration between the two, but one must not
silently substitute for the other.

## Interaction with `SafetyCase`

`SimulationOutcomeClaim` answers:

> Did the external simulation produce evidence consistent with the proposed physical
> outcome?

`SafetyCase` answers a different question:

> Were the required safety/proof obligations discharged for this candidate/evidence
> lineage?

Strict simulation qualification should eventually require both:

```text
SatisfiedSimulationClaim
        +
proposal-bound discharged SafetyCase
        -> simulation-qualified candidate
```

A safety case cannot substitute for missing success evidence, and successful performance
metrics cannot substitute for a safety case.

## No post-hoc threshold selection

The implementation should make post-hoc success criteria difficult by API shape.

Preferred ordering:

```text
SelectedCandidate
    + predeclared SimulationOutcomeClaim
    + context-bound SimulationRequest
            |
            v
run solver
            |
            v
evaluate exact predeclared claim
```

Avoid an API shaped like:

```text
run solver -> inspect output -> construct claim -> qualify
```

where success criteria can be moved after seeing the answer.

For research workflows that intentionally explore results before defining a hypothesis,
store those runs as exploratory evidence and require a new preregistered lineage for
confirmatory qualification.

## Exploratory vs confirmatory evidence

This distinction would make Symthaea's scientific reasoning stronger.

Conceptually:

```text
ExploratorySimulation
    result may generate hypotheses
    cannot retrospectively become confirmatory evidence

ConfirmatorySimulation
    claim/context/request frozen before execution
    eligible for strict claim qualification
```

That mirrors good experimental methodology and prevents automated p-hacking-like behavior
inside physical planning.

PA-10 does not need to implement the full research-governance machinery yet, but the
claim schema should not prevent it.

## Required adversarial tests

### Irrelevant metric

Claim requires metric `diagnostic_quality`; solver returns only an unrelated valid metric.
Result is not evidence of the claim.

### Wrong unit

Claim expects canonical unit `m`; result supplies the same metric name in `mm` without a
normalized adapter conversion. Fail closed rather than comparing raw numbers.

### Duplicate metric ambiguity

Result contains two metrics matching the same strict `(name, unit)` selector. Fail closed.

### Threshold straddle

Claim requires `AtLeast(0.8)`, result estimate is `0.85` but required uncertainty interval
is `[0.72, 0.93]`. Result is `Indeterminate`, not satisfied.

### Conservative maximum

Claim requires `AtMost(5.0)`, interval `[4.2, 5.4]`. `Indeterminate`.

### Exact interval success

Claim requires `InsideClosedInterval[2, 4]`, result interval `[2.2, 3.8]`. Satisfied.

### Missing interval

Criterion requires interval; result has only point estimate. `Indeterminate`.

### Post-hoc neighboring claim

Run is bound to claim A. A caller presents claim B after execution, even with compatible
thresholds. Strict confirmatory qualification rejects the lineage mismatch.

### Run substitution

A satisfied claim receipt from output digest A cannot qualify output digest B.

### World-snapshot substitution

Claim/request/selection use snapshot A; evidence uses a request lineage containing
snapshot B. Reject before claim satisfaction can discharge qualification.

### Dry-run success spoof

A dry run reports numerically perfect metrics. It may be useful for orchestration tests,
but cannot mint strict external-solver claim evidence.

### Non-finite thresholds and values

NaN/Inf anywhere in claim thresholds or result metrics fail validation.

## PHYSIS v1 direction

After PA-09 and PA-10 are implemented, PHYSIS should grow from architecture-only lineage
tests into a small confirmatory-evidence benchmark.

A safe first fixture could compare simulated diagnostic mechanisms where the success
criteria are generic information/measurement metrics rather than high-energy physical
effects.

PHYSIS should test at least:

1. predeclared claim freezes before solver execution;
2. model ensemble predicts candidate outcome;
3. Pareto deliberation selects a candidate;
4. exact world snapshot enters request lineage;
5. external simulation returns structurally valid evidence;
6. typed metrics satisfy or fail the predeclared claim under uncertainty;
7. exact claim receipt and exact safety-case evidence are both required;
8. no artifact produced anywhere in the sequence grants hardware authority.

## Migration plan

1. freeze this RFC;
2. implement PA-09 typed simulation context/request lineage first;
3. introduce claim/criterion types in Physical Agency;
4. add result-side exact metric lookup and conservative uncertainty evaluation;
5. create a non-serializable satisfied-claim receipt;
6. require that receipt in the public strict simulation qualifier;
7. update PHYSIS to exercise adversarial semantic-evidence cases;
8. keep all real modality adapters out until the hosted qualification line is green.

## Phase exit gate

PA-10 is complete only when a solver result cannot become strict evidence merely by being
valid and well-provenanced.

The qualified path must establish:

```text
predeclared claim
    + exact selected candidate
    + exact world/context lineage
    + exact solver request lineage
    + exact solver output
    + conservative metric satisfaction
    + exact safety-case lineage
        -> simulation-qualified evidence
```

and every missing, ambiguous, substituted, post-hoc, uncertainty-straddling, or dry-run
path must fail closed.

Only after that should the first real acoustic diagnostic adapter be considered part of
the strict Physical Agency evidence path.
