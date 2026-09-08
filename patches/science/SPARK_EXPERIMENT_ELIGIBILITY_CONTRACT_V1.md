# Spark Experiment Eligibility Contract v1 — staged qualification design

Status: **queue-neutral design artifact only**

Base: `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`

Tracking issue: #890

Related architecture: SCI-004 / SCI-009 (#850)

This file is not an implementation patch, does not interpret historical `go_nogo` prose as executable policy, and carries no experiment/action authority.

## Problem statement

Spark currently has two different experiment structures:

```text
ExperimentalProgram / ExperimentPhase
    -> phase goals
    -> phase order
    -> human-readable go/no-go conditions

optimal_experiment
    -> flatten every phase
    -> rank by model-relative EIG / cost
    -> greedy budget-constrained sequence
```

Flattening is useful for portfolio visibility, but it erases the distinction between:

```text
candidate in the long-range program
and
eligible next experiment
```

## Core theorem

```text
CandidateExperiment
    !=
StructurallyValidExperiment
    !=
EligibleNow
    !=
PlannerPreferred
    !=
AuthorizedForExecution
```

and:

```text
BlockedByPrerequisite
    !=
LowScientificValue
```

A blocked experiment may still have high prospective scientific or operational value.

## 1. Machine-readable program state

The first implementation should introduce a small Spark-local program state rather than parsing historical prose.

Conceptual types:

```text
ExperimentStageIdV1
ExperimentCandidateIdV1
ExperimentPrerequisiteIdV1
ExperimentProgramStateV1
```

An exact experiment candidate should bind its stage/program identity separately from the experiment's scientific/predictive content.

## 2. Eligibility state

Minimum conservative vocabulary:

```text
ExperimentEligibilityStateV1 {
    EligibleNow,
    BlockedByPrerequisites {
        unmet,
    },
    PrerequisiteStateUnknown {
        required,
    },
    Inadmissible {
        reasons,
    },
    HistoricalCompleted,
}
```

Unknown prerequisite state fails closed for **current eligibility**.

It does not delete the experiment from the candidate inventory.

## 3. Prerequisite semantics

Prerequisites should reference typed evidence/state, not arbitrary caller booleans.

Long-term shape:

```text
ExperimentPrerequisiteV1 {
    prerequisite_id,
    required_scientific_state_ref,
    evaluation_profile_id,
}
```

The first pilot can be intentionally narrower and use one explicit test-owned transition state.

Do not create a generic parser from free-text `go_nogo` into authority.

## 4. Historical authored program remains evidence, not authority

Current `ExperimentPhase.go_nogo: String` establishes that the program was designed with stage gates in mind.

It does **not** prove:

```text
the exact criterion is machine-readable
the criterion was prospectively frozen
the criterion has been satisfied
the next stage is authorized
```

A migration should therefore add exact typed gate semantics rather than overwrite/reinterpret the old text.

## 5. Candidate inventory versus active frontier

SCI-009 should preserve two views:

```text
LongRangeCandidateInventory
ActiveEligibleFrontier
```

Long-range inventory may include blocked later-stage experiments for:

```text
budget planning
resource planning
instrument procurement
future value estimation
contingency design
```

The active frontier contains only experiments whose hard prerequisites are currently satisfied.

## 6. Planning flow

Preferred Spark sequence:

```text
Raw candidate inventory
  -> structural planning validation                 (#885 contract)
  -> exact target / prediction coverage diagnostics (#885 / #868)
  -> prerequisite / feasibility evaluation          (#890)
  -> ActiveEligibleFrontier
  -> observation-profile discrimination             (#857)
  -> scientific/cost/risk coordinates
  -> Pareto or exact preference policy               (SCI-009)
  -> planner proposal
  -> separate experiment authorization
```

The planner may never turn `BlockedByPrerequisites` into eligible by adding enough EIG, lowering cost, or changing preference weights.

## 7. Adaptive transition semantics

The architecture must support non-linear/adaptive science.

A future prospective SCI-004 program may define transitions such as:

```text
validation supported       -> mechanism branch eligible
validation falsified       -> anomaly-mechanism branch remains blocked
measurement invalid        -> measurement-repair/repeat branch eligible
mechanism unresolved       -> discrimination branch eligible
mechanism supported        -> selected optimization branch eligible
```

Every transition creates a new immutable planning snapshot.

Past eligibility states remain historical records.

## 8. Planner output semantics

A future plan record should distinguish:

```text
recommended_from_eligible_frontier
blocked_but_high_value
ineligible_due_to_unknown_prerequisite
inadmissible
```

Do not present a raw all-program ranking as an executable recommended sequence unless the report explicitly says phase/prerequisite eligibility was ignored.

## Required negative controls for future qualification

### A. High-EIG blocked experiment

Create:

```text
Phase1Experiment: eligible, modest EIG
Phase3Experiment: blocked, arbitrarily huge EIG/cost ratio
```

Post-repair requirement:

```text
Phase3 remains blocked
Phase1 can be selected
```

### B. Unknown prerequisite

A required prerequisite with no evaluable state must produce:

```text
PrerequisiteStateUnknown
```

not `EligibleNow`.

### C. Successful transition

Start with Phase 2 blocked.

Supply the exact machine-readable satisfied prerequisite in a new program-state snapshot.

Phase 2 becomes eligible without modifying the historical candidate or prior planning snapshot.

### D. Failed prerequisite

A failed Phase-1 gate must not unlock later stages.

### E. Audit preservation

Blocked experiments must remain present in the long-range inventory/report rather than disappearing from evidence.

### F. Utility non-bypass

Change EIG/cost/utility values across extreme ranges.

Hard prerequisite state must remain invariant.

## First implementation tranche

Keep the first Rust change deliberately small and non-authorizing:

```text
ExperimentStageIdV1
ExperimentEligibilityStateV1
ExperimentProgramStateV1
```

plus a pure eligibility evaluator over an exact test-owned prerequisite representation.

Do not initially:

- translate all current `go_nogo` strings;
- execute physical experiments;
- infer gate satisfaction from model confidence;
- grant action capabilities;
- claim the current three-stage program is scientifically optimal.

## Relationship to safety / rights constraints

This stage prerequisite layer is not a complete safety or ethics system.

SCI-009's hard constraints may include independent safety, consent, rights, environmental, regulatory, and governance admissibility checks.

Conceptually:

```text
scientific-stage eligibility
AND safety/rights/governance admissibility
```

may both be required before an experiment reaches an executable authorization layer.

## Deliberate non-claims

This contract does not establish:

- that the historical LCF phase transitions are correct;
- that any current go/no-go has been satisfied;
- that EIG is calibrated;
- that an eligible experiment is safe;
- that an eligible experiment should be executed;
- that a planner recommendation grants action authority.

It defines only the queue-neutral target for keeping long-range candidate value distinct from current scientific eligibility.