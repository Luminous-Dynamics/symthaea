# Aircraft Factory-Line Scheduling Benchmark

**Date:** 2026-10-16  
**Status:** pure-Rust operations-research baseline and deterministic fixture; not a production planning authority.  
**Tracking issue:** [FACTORY-OPS-001 #7677](https://github.com/Luminous-Dynamics/symthaea/issues/7677)

## Why this belongs in the existing crate

Symthaea already has \`symthaea-operations-research\`, a small dependency-free library for queueing, inventory and shortest paths. Extend that owner instead of creating another optimization stack. Factory realization remains owned by [BUILT-ENV-001 #6032](https://github.com/Luminous-Dynamics/symthaea/issues/6032), engineering requirements/design evidence by [ENG-DESIGN-001 #6005](https://github.com/Luminous-Dynamics/symthaea/issues/6005), and manufacturing semantics/resource provenance by Mycelix [MFG-EVID-001 #3084](https://github.com/Luminous-Dynamics/mycelix/issues/3084) and [MFG-PROC-001 #3085](https://github.com/Luminous-Dynamics/mycelix/issues/3085).

The purpose of the first slice is narrow: give the aircraft-factory study a deterministic, inspectable baseline for whether a process graph can be scheduled with the declared worker skills, resource/tool capabilities and availability windows.

## Pilot product/process

The fixture represents a low-volume light-aircraft/glider assembly pilot. It is a scheduling fixture, not a licensed manufacturing route or a declaration that the operations are sufficient to build a flightworthy aircraft.

| Operation | Duration (s) | Predecessors | Required skill | Required resource |
|---|---:|---|---|---|
| Incoming inspection | 10 | — | Inspection | Metrology station |
| Cut structural members | 15 | Incoming inspection | Machining | CNC machine |
| Build electrical harness | 10 | Incoming inspection | Electrical | Electrical test bench |
| Structural subassembly | 20 | Cut structural members | Assembly | Assembly fixture |
| Final assembly | 30 | Structural subassembly, harness | Assembly | Assembly fixture |
| Final inspection | 15 | Final assembly | Inspection | Metrology station |

For the fixed test fixture, the precedence-only critical path is 90 seconds. The deterministic earliest-finish planner is expected to find the same 90-second makespan, with the harness operation running in parallel with the cutting operation. This is a tiny correctness fixture, not a credible factory takt-time or commercial throughput claim.

## API and semantics

\`plan_factory_line\` takes immutable operation definitions, qualified worker manifests and qualified resource manifests. All operations must be scheduled exactly once. Each operation reserves one worker and one resource for its full duration. Predecessor completion, skills/capabilities, interval availability, unique IDs and overflow are validated. The result reports:
- scheduled interval and selected worker/resource for each operation;
- makespan;
- a precedence-only critical-path lower bound;
- heuristic gap (makespan minus lower bound).

Algorithm is deterministic earliest-finish list scheduling with stable tie-breaking. **It is a heuristic, not an optimal job-shop solver.** Under complex time windows, failure to place an operation means this heuristic did not produce a schedule; it does not necessarily prove that no other ordering or resource assignment could work. The result/error wording must preserve this distinction. Future resource calendars, shift patterns, setup/changeover, maintenance windows, multi-worker tasks, uncertainty and exact/MILP solving are separate extensions, each with tests.

## Paradigm shift

The scheduler's unit is not just a machine and a duration. An operation is feasible only when the exact process revision, qualified people, calibrated resources, inputs, environment and inspection capacity all satisfy the declared requirements. This supports **skill-aware manufacturing**: the schedule cannot improve nominal throughput by inventing qualifications, skipping an inspection hold point or assuming a resource is available when its calendar says otherwise.

The next integration should bind the process plan to:
- the design/product configuration and revision;
- operation/process definition revision;
- material/component lot identity and approved substitutions;
- resource/tool/fixture and calibration evidence;
- required and recorded skill/authorization evidence;
- inspection/test acceptance criteria and hold points;
- work-order/manufactured-serial lineage and nonconformance/rework records.

Praxis, if selected as the training provider, can later consume versioned task/scenario refs and export assessment evidence against these same skill definitions. The exact Praxis repository/API was not found in accessible Luminous-Dynamics public repository searches; do not invent an integration or make training completion equal practical authorization.

## Benchmark and adversarial gates

1. **Validity:** all operations scheduled once; known predecessors; no cycles; positive bounded durations.
2. **Qualification:** required skills and resource capabilities must match exact declared IDs; no synonym/role-name inference.
3. **Feasibility:** every assignment lies inside the declared worker and resource availability windows; no worker/resource overlaps; every predecessor finishes before its dependent begins.
4. **Determinism:** same normalized inputs produce identical output ordering and timings.
5. **Independent verification:** a separate validator recomputes the stated invariants from the emitted plan, and detects schedule mutation.
6. **Negative controls:** missing inspector, missing metrology, missing calibration/evidence (in later schema integration), unknown predecessor, dependency cycle, duplicate IDs, invalid duration, unavailable window, integer overflow, process revision mismatch and unauthorized substitution.
7. **Optimization honesty:** report lower bound and heuristic gap. Do not call this optimal or globally infeasible when a heuristic dead-end has not been disproved by exhaustive search.
8. **Production target:** after a frozen, measured baseline exists, attempt >=10% reduction in rework hours or end-to-end lead time without increased escaped defects, weakened acceptance criteria, skipped inspection or worse worker-safety results.

Report first-pass yield, escape defects, rework, scrap, labor hours, lead time, utilization, queue delay, downtime and the cost of tooling/spares as separate measures. A better makespan does not establish better quality, aircraft safety or lower total cost.

## Validation and claim ceiling

The code includes deterministic tests and a test-side independent schedule checker. This development environment did not execute Cargo tests or a factory simulation binary; test-source presence is not a passing result. Recheck with \`cargo test -p symthaea-operations-research\` on the exact PR head. The fixture is only an algorithmic baseline, not a qualified manufacturing process, factory layout, aircraft design or airworthiness approval.
