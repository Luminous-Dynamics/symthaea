# SFEB task-family expansion plan

**Date:** 2026-11-23  
**Status:** 30-task candidate blueprint; one candidate fixture-ready, zero runnable tasks

The repository currently has 12 staged task definitions and two checked-in public-pilot reference-fixture bundles. SFEB-L1-004 has the service-dashboard fixture; SFEB-L2-005 has the search/filter explorer fixture, synthetic dataset, Rust reference-model source and Leptos candidate source. In task-family-plan-v1.json, SFEB-PILOT-INT-003 is the single fixture-ready candidate promoted to SFEB-L2-005; the other 29 candidate briefs remain planned. The plan's runnable_task_count remains zero because Rust compilation/tests, WASM build, candidate browser journeys, isolated execution and independent browser-oracle replay are not complete.

## Proposed distribution

### Static visual (10)
Operations dashboard, editorial landing, financial table, inventory catalogue, resource calendar, analytics charts, support inbox, multilingual RTL page, administration settings and system status.

### Interactive multi-route (10)
Project routing/history, accessible multi-step form, search/filter/sort, keyboard-accessible work board, role-based workspace, live telemetry/reconnect, document import, persistent settings, resource scheduling and command palette.

### Fullstack and maintenance (10)
Multi-tenant CRUD, typed API connector, backward-compatible migration, mock-payment lifecycle, append-only audit history, feature maintenance, async race repair, secure file service, framework upgrade and operational resilience.

Every candidate names oracle families, seeded defect families, metamorphic controls and required evidence. Only INT-003 has fixture assets and authored golden/model source; the other 29 fixtures and their independent oracle campaigns remain unauthored. The 11/11 golden, 5,000-state, 9/9 fault-model and 8/8 URL-omission results belong to a separate JavaScript reference model and do not promote the Rust or browser task to runnable.


## Current inventory and evidence boundary (2026-11-23)

| Inventory item | Current state |
|---|---|
| Staged SFEB task definitions | 12 |
| Original public-pilot fixture bundles | 2 |
| Candidate task briefs in the 30-task plan | 30 |
| Candidate fixture-ready tasks | 1 (INT-003 → SFEB-L2-005) |
| Remaining candidate fixtures planned | 29 |
| Runnable tasks | **0** |

The service-dashboard bundle is present, but SFEB-L1-004 remains specification_only in the task index. SFEB-L2-005 is fixture_ready; its independent JavaScript reference preflight is not a Rust test receipt and not a Leptos/WASM browser qualification. Do not report either as a fully runnable or benchmark-scored task.

The 30-task public plan and the two fixture bundles are development material. The controller-only and hidden stages must be authored and stored outside the agent-readable public repository; public fixtures must never later be relabeled as hidden holdouts.

## Counterfactual task-family design

Each eventual task should have:
- a baseline implementation that satisfies its contract;
- a reproducible seeded-defect variant;
- a minimal intervention or repair hypothesis;
- a placebo/no-op or unrelated intervention;
- invariants for unaffected requirements;
- at least one semantics-preserving positive or metamorphic control.

This supports causal repair evaluation rather than simple before/after screenshot scoring.


## Two-sided task validity and external benchmark lessons

SWE-Bench Pro V2 provides an important current precedent: its September 22, 2026 public release dropped 89 tasks after review, corrected 529 task statements so graded assertions trace to the prompt, required a reviewed reference patch to pass and an empty patch to fail, restricted agent network access to the model endpoint, and re-graded candidate diffs on pristine images. This does not make SWE-Bench Pro a frontend benchmark, but it demonstrates that task validity and evaluator isolation must be release gates.

SFEB should adopt the underlying validity controls without copying repository-specific implementation:
- task brief ↔ requirement ↔ oracle traceability in both directions;
- independent human solvability before hidden tests are exposed;
- a passing reviewed witness implementation and a failing no-op/seeded-defect control;
- regrade on a clean evaluator image, preventing edits to tests, caches, environment or verification scripts from inflating credit;
- no access to fixing commits or solution-bearing sources;
- preserve partial, blocked and inconclusive execution outcomes separately.

The exact protocol is defined in benchmarks/frontend/task-qualification/two-sided-task-validity-v1.json. It is a design artifact; no task has passed this new protocol yet.

The adjacent benchmark landscape has also shifted:
- **FrontierCode 2.0** (Cognition, October 27, 2026) has 132 maintainer-authored repository tasks and a mergeability-oriented mix of tests, rubrics and verifiers. It is a useful comparator for maintenance and patch quality, not a frontend-only leaderboard.
- **OSWorld 2.0** includes 108 long-horizon workflows, 31 self-hosted websites and over 250 average agent steps. It is a useful adjacent stress test for complex UI interaction and state tracking, not code-generation quality.
- **SWE-Bench Pro V2** is a stronger task-validity and clean-room evaluation precedent, but its task success numbers must not be pooled with SFEB.

Sources: [SWE-Bench Pro V2 release](https://labs.scale.com/blog/swe-bench-pro-v2), [V2 task/release gate](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md), [FrontierCode 2.0](https://cognition.com/blog/frontier-code-2.0), [OSWorld 2.0](https://osworld-v2.xlang.ai/).

## Exposure and scale

- 30 public pilot tasks for harness development;
- 150 matched comparative tasks, 50 per level;
- at least 500 publication tasks;
- at least 30% controller-only/hidden holdout at the mature stage.

Public tasks are development material, not contamination-resistant evidence. Hold out whole task families and interaction patterns; do not commit hidden source, gold labels or oracles to the public repository.

## Promotion gates

A task is not runnable until original assets, fixture services, environment pins, cleanup protocol, semantic anchors, executable contract, independent oracle rules and evidence capture requirements exist. It must also pass the two-sided task-validity gate: a reviewed witness patch passes, an empty/no-op or seeded-defect baseline fails for the intended reason, the brief is independently solvable, and the patch is re-graded on a pristine evaluator image. Oracles must pass positive/negative/metamorphic qualification. A checked-in brief, fixture or candidate source alone does not count.

Implementation is tracked in issue [#13251](https://github.com/Luminous-Dynamics/symthaea/issues/13251), coordinated with oracle replay #13247, execution #13228 and judge qualification #13250.
