# SX-032: Evidence-Carrying Mission Task Verification v0.1

**Status:** experimental software-contract verifier; synthetic tests only  
**Implementation:** suit_mission_tasks  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/7530  
**Predecessor:** SX-031 mission configuration graph, PR #7527

## Paradigm shift: verify the work, not just the wiring

SX-031 checks that declared components and pairwise interfaces compose under a policy. SX-032 adds an explicit task plan: a mission can have a connected suit/lander graph and still be unable to perform a required task because the suit lacks the claimed capability, a support resource is missing, the required connection is incompatible, or the declared connection evidence does not cover the task's operating environment.

Each task pins:
- a stable task ID and required/optional status;
- required suit capability claims;
- exact required component IDs;
- exact required connection IDs;
- a non-empty set of operating environments.

No requirements, evidence, resources, or connections are inferred from task names or model output. A task without declared dependencies is rejected to prevent vacuous acceptance.

## Evaluation contract

Task dependencies are checked against the SX-031 mission report and the SX-028 scoped suit capability findings:

| Dependency | Required task failure | Optional task failure |
|---|---|---|
| Capability claim absent | Blocker | Degradation |
| Capability evidence below the caller's floor | Blocker | Degradation |
| Conflicting capability claims | Blocker | Degradation |
| Required component absent | Blocker | Degradation |
| Component not reachable over policy-compatible edges in the task environment | Blocker | Degradation |
| Required connection absent or blocked | Blocker | Degradation |
| Task environment outside the connection's declared policy scope | Blocker | Degradation |

An optional task cannot waive a blocker in the base mission graph. An incompatible connection cannot be used as a path merely because an edge declaration exists. SX-031's connectivity check is likewise restricted to edges that meet their pairwise software policy.

Aggregate status is Blocked when the base graph or any required task is blocked, Degraded when no blocker exists but an optional task or base configuration is degraded, and MeetsDeclaredTaskPolicy otherwise.

## Evidence and receipt semantics

The task plan and the full embedded mission-configuration report receive domain-separated SHA-256 digests. The receipt binds the exact configuration digest, canonicalized task-plan digest, mission-report digest and task-report digest. Verification re-runs the evaluator and compares the typed outputs/receipt.

These digests provide deterministic change evidence only. They are not signatures, provenance authentication, evidence-source validation or qualification. A caller-declared BenchTested or Qualified label is still a claim supplied to the existing model; SX-032 does not verify the underlying artifact or promote evidence automatically.

Every report sets represents_physical_qualification = false. Nothing in this module authorizes EVA, actuators, life support or emergency operations.

## Current synthetic corpus

The module includes unit cases for successful explicit dependencies, missing capability evidence, below-floor evidence, missing components/connections, task environment mismatch, optional-task degradation, empty/vacuous and duplicate task rejection, task-vector permutation invariance, component-vector permutation invariance and receipt tampering. An integration replay contains a positive lunar-shadow task fixture and a negative fixture with missing capability/resource/connection prerequisites.

All fixture names, capabilities, revisions and evidence references are synthetic. These tests do not simulate astronaut workload, lunar grounding physics, dust, life support, thermal response, physical resource flow or rescue feasibility.

## Proposed benchmark targets

These targets are not measured results. Use a documented reference runner, frozen Rust toolchain, deterministic generator seeds and retained manifests before establishing performance gates.

| Benchmark | Initial target |
|---|---|
| Required task negative controls | 100% of seeded missing/blocked/out-of-scope prerequisites become blockers |
| Optional task negative controls | 100% degrade rather than block when all required tasks and base graph pass |
| Vacuous task declarations | 100% rejected |
| Mutation controls | 100% receipt rejection after changing tasks, evidence, graph, policy, revision or returned report |
| Task input order | Identical report and receipt across 1,000 semantically equivalent permutations |
| Generated task-graph corpus | 2,000 seeded valid/invalid tasks; zero unexpected panics |
| 500 tasks × 20 prerequisites | p95 below 50 ms |
| 5,000 tasks × 20 prerequisites | p95 below 500 ms |

For any 100% claim, report numerator and denominator, and preserve positive cases as well as negative controls. Timing goals are only meaningful alongside semantic coverage and false-acceptance counts.

## Next research/engineering steps

1. **Requirement traceability graph:** link each task to source requirement IDs, verification method, artifact digest, test run, exact hardware/software revision and the reviewer who accepted it.
2. **Resource quantities and reservations:** represent time-bounded consumables, power, cooling, communications, workspace and crew capacity as explicit resource claims with units and uncertainty intervals. Do not treat a graph path as proof of resource sufficiency.
3. **Fault-campaign matrix:** inject one or multiple missing resources, changed interface revisions, failed evidence artifacts, lost relays and environment changes; freeze expected task outcomes in a separate corpus.
4. **Minimal blocker explanation:** identify the smallest set of missing task prerequisites that blocks the declared task set; verify the explanation independently before presenting it.
5. **Human-system validation:** maintain a separate route for task feasibility, workload, dexterity, procedure usability and human-in-the-loop findings; graph consistency cannot establish these outcomes.
6. **Evidence provenance:** only add signed provenance after defining identity, trust roots, key rotation/revocation, timestamps, artifact retention and acceptance authority.

## Research basis

NASA's Systems Engineering Handbook appendix describes requirements verification matrices keyed by unique identifiers and guidance for requirements validation and traceability. NASA-STD-3001 Volume 2 Revision F covers human performance, environmental conditions, equipment, crew interfaces and spacesuits; NASA software guidance distinguishes validating that the right system is being built from verifying that the implementation meets requirements. These practices motivate explicit task traceability, not a claim that this module implements or satisfies NASA requirements.

- NASA Systems Engineering Handbook appendix: https://www.nasa.gov/reference/system-engineering-handbook-appendix/
- NASA-STD-3001 Volume 2 Revision F: https://standards.nasa.gov/node/237
- NASA SWE-055 Requirements Validation: https://swehb.nasa.gov/spaces/7150/pages/16449673/SWE-055%2B-%2BRequirements+Validation
