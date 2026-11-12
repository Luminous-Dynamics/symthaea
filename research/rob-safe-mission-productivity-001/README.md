# ROB-SAFE-MISSION-PRODUCTIVITY-001

Owner: Symthaea issue #11627 (fleet-serving benchmark).

This standalone, dependency-free Rust reference evaluates whether a fleet-serving arm produces more **SLO-qualified useful work** without hiding safety, evidence, deadline, recovery or per-robot service regressions.

It is a retrospective benchmark evaluator. It does not schedule inference, control robots, estimate physical state, authorize actuators or establish hard real-time guarantees.

## Paradigm: productivity is conditional on safe and evidenced work

Raw requests/second, GPU utilization and task completion can reward a scheduler that silently starves one robot, misses a critical action deadline, loses recovery reserve, or applies actions without complete physical lineage.

The evaluator therefore reports separate measures:

- task completions and physical safe-success;
- evidence-qualified safe-success (qualified thermal and recovery receipts plus causal application closure);
- SLO-qualified missions (evidence-qualified safe-success, no critical deadline miss and no service-floor breach);
- task-class useful work per GPU-hour and per robot-hour, using the task class's exact declared work unit;
- per-robot starvation, service-floor breaches, hard-safety/deadline misses and causal closure;
- energy evidence coverage and measured joules, kept separate from productivity;
- actuator-application causal closure, weighted by application counts rather than averaging mission percentages.

Failed, incomplete and unsafe missions remain in the denominator for throughput/GPU cost. Only SLO-qualified completed work receives productivity credit. Unknown energy is not entered as zero joules. Unknown thermal, recovery or causal evidence cannot receive evidence-qualified success credit.

## Comparison contract

Candidate/baseline comparisons require one-to-one pair_id matches and exact equality of the frozen workload, scenario, seed, initial state, robot, model, local control policy, hardware, observation budget and compute budget. Only the scheduler arm is allowed to differ.

The report retains a vector per task class and per robot, including per-robot hard-safety regressions. It does not combine unlike task units or produce one fleet-wide weighted winner. A candidate is non-inferior only when declared safe-success, SLO-success, deadline, hard-violation, service-floor and causal-closure margins hold for every task class and robot. Productivity dominance is reported only when non-inferiority holds, no task class loses productivity, and at least one task class improves.

Energy, joules, EVSI, regret, latency and task work remain distinct quantities. Energy is a separate evidence/measurement coverage vector, not a productivity conversion factor.

## Core types and semantics

- MissionTrial is one independent mission / sequential perturbation episode for one scheduler arm.
- MatchIdentity freezes every comparison dimension that must remain matched.
- EvidenceState::{Qualified, Unknown, Invalid} prevents missing evidence from becoming a zero-valued pass.
- ScorecardContract sets the minimum per-mission causal-closure threshold for productivity credit.
- ComparisonContract declares allowed per-metric non-inferiority margins.
- FleetScorecard reports per-task-class and per-robot results.
- FleetComparison reports matched deltas and Pareto outcomes without cross-class scalarization.

Causal closure is derived as:

    sum(applications_causally_closed) / sum(applications_examined)

It is weighted by the actual application denominator. It is not the unweighted mean of mission percentages. No actuator application means no causal-closure percentage and cannot earn SLO productivity credit.

Task-class productivity is reported in two independent rates: fixed-point milli-work-units per GPU-hour and per robot-hour. Both denominators include all consumed GPU service / elapsed mission time for that task class, including failed missions. Work units from different task classes are never pooled.

## Industry comparison matrix

Use the same robot/task/plant, model weights, local control policy, observation history, compute budget, GPU hardware, network conditions, initial state and held-out workload seeds wherever the comparison permits.

Required external and internal arms:

1. Round Robin and EDF.
2. Static active-episode cap.
3. Armory Lookahead and RED where their published scheduling assumptions and implementations can be matched.
4. Shared-GPU-pool serving (ROSA-style) under its published system-level SLO objective.
5. Luminous constrained mission-aware admission.
6. Evaluator-only clairvoyant oracle, labeled upper bound and never deployable.

Keep published results on different models, tasks and GPUs as external reference points only. Do not compare their raw scores as though they were matched Luminous replications.

## Prospective targets (not achieved)

- Zero local hard-safety displacement.
- Zero candidate regression in hard safety, critical deadlines, service floors or terminal safe-stop under the declared promotion margins.
- At least 100% causal closure on applications credited as SLO-qualified work when a 100% closure contract is selected.
- At least 10% more task-class SLO-qualified work per fixed GPU-hour than the strongest reproducible matched shared-pool baseline as a stretch target.
- No per-robot safe-success, deadline or starvation regression hidden by fleet-average productivity.
- Publish every task-class and robot stratum, including null results and cases where Round Robin, EDF, static caps, Armory or RED performs better.

The 10% target is prospective. It must be preregistered after baseline variance and measurement boundaries are established and confirmed on held-out paired mission sequences with confidence intervals clustered at the independent mission level.

## Run

    cargo fmt --manifest-path research/rob-safe-mission-productivity-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-safe-mission-productivity-001/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-safe-mission-productivity-001/Cargo.toml --all-targets -- -D warnings

The crate contains 20 authored deterministic unit tests covering paired workload identity, per-robot safety regressions, Unknown versus Invalid evidence, energy-boundary missingness, mixed task-unit rejection, causal-closure weighting and productivity denominators. Exact-head Rust 1.96 CI is the execution gate. These tests are not physical results or reproduced SOTA comparisons.
