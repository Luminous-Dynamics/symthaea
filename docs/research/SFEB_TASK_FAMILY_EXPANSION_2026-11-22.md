# SFEB task-family expansion plan

**Date:** 2026-11-22  
**Status:** 30-task candidate blueprint authored; fixture and oracle work pending

The repository currently has 11 task definitions, but only SFEB-L1-004 has a complete checked-in fixture/task bundle. This note and `task-family-plan-v1.json` make the gap explicit: the next 30 are candidate briefs, not runnable tasks.

## Proposed distribution

### Static visual (10)
Operations dashboard, editorial landing, financial table, inventory catalogue, resource calendar, analytics charts, support inbox, multilingual RTL page, administration settings and system status.

### Interactive multi-route (10)
Project routing/history, accessible multi-step form, search/filter/sort, keyboard-accessible work board, role-based workspace, live telemetry/reconnect, document import, persistent settings, resource scheduling and command palette.

### Fullstack and maintenance (10)
Multi-tenant CRUD, typed API connector, backward-compatible migration, mock-payment lifecycle, append-only audit history, feature maintenance, async race repair, secure file service, framework upgrade and operational resilience.

Every task names oracle families, seeded defect families, metamorphic controls and required evidence. All candidate fixtures and oracle policies are explicitly not authored. The runnable count remains zero for this plan.

## Counterfactual task-family design

Each eventual task should have:
- a baseline implementation that satisfies its contract;
- a reproducible seeded-defect variant;
- a minimal intervention or repair hypothesis;
- a placebo/no-op or unrelated intervention;
- invariants for unaffected requirements;
- at least one semantics-preserving positive or metamorphic control.

This supports causal repair evaluation rather than simple before/after screenshot scoring.

## Exposure and scale

- 30 public pilot tasks for harness development;
- 150 matched comparative tasks, 50 per level;
- at least 500 publication tasks;
- at least 30% controller-only/hidden holdout at the mature stage.

Public tasks are development material, not contamination-resistant evidence. Hold out whole task families and interaction patterns; do not commit hidden source, gold labels or oracles to the public repository.

## Promotion gates

A task is not runnable until original assets, fixture services, environment pins, cleanup protocol, semantic anchors, executable contract, independent oracle rules and evidence capture requirements exist. Its oracles must pass positive/negative/metamorphic qualification. A checked-in brief alone does not count.

Implementation is tracked in issue [#13251](https://github.com/Luminous-Dynamics/symthaea/issues/13251), coordinated with oracle replay #13247, execution #13228 and judge qualification #13250.
