# SCI-006 — Scientific Evidence Dependency Graph v1 — Summary

**Status:** architecture-only; non-authorizing; non-qualifying.

SCI-006 defines how Symthaea should reason about shared scientific foundations without turning publication/model/agent count into fake replication.

## Core rule

Evidence multiplicity must come from exact target-compatible lineage and dependency topology, not from paper count, model count, organization count, or producer-supplied independence flags.

The strongest generic no-overlap conclusion is intentionally:

`DeclaredDisjointWithinScope`

not universal `IndependentReplication`.

## Three graph questions

SCI-006 separates:

1. derivation ancestry — what exact artifacts were derived from what;
2. scientific dependency inventory — what data/method/model/measurement/policy/tool foundations a contribution relies on;
3. replication/triangulation assessment — a later derived interpretation over target-compatible contributions and sufficiently complete inventories.

RCA's closed evidence-root DAG is a strong typed input, not something the generic layer should weaken or overgeneralize.

## Common dependency axes

A shared inventory can represent exact identities for source data/vintage, sampling, measurement/instrument/calibration, transformations/features, models/training data, estimators/identification strategies, SCI-003 execution/analysis, SCI-004 outcome/evaluation/decision policies, learned grammar, retrieval/embedding artifacts, external tools/databases, and domain-specific extensions.

These categories are not a strength ranking.

## Incomplete inventory fails closed

No known overlap is not enough.

If required dependency domains are incomplete, the pair remains `IncompleteInventory`.

Known overlap wins even when other metadata is missing: one known shared dependency is sufficient to reject declared disjointness.

## Pairwise relation is not component identity

Known shared dependencies may connect A-B and B-C without A-C sharing the same dependency.

SCI-006 retains exact pairwise relations plus conservative known-dependency connected components. Component cardinality is descriptive topology, not evidence weight or replication score.

## Methodological diversity is useful but separate

Shared data with different estimators, different data with the same measurement design, cross-instrument observations, or formal/numerical/empirical triangulation can all be scientifically informative.

The architecture should expose that diversity without pretending it erases shared dependencies or yields one scalar independence score.

## Replication class is not independence

Exact computational reproduction may intentionally share nearly everything. Conceptual replication may differ widely while retaining hidden upstream dependencies.

Prospective cleanliness (SCI-005), replication class, target relation, and dependency topology are separate axes.

## First implementation slice

Start with inventory only:

- `ScientificDependencyDomainV1`;
- `ScientificDependencyRefV1`;
- `DependencyInventoryScopeV1`;
- `DependencyInventoryCoverageV1`;
- `ScientificDependencyInventoryV1`.

Then add conservative pairwise comparison and known dependency components. Still do not issue `IndependentReplication`.

Economic Science is the strongest initial pairwise-comparison pilot; RCA is the strongest closed-ancestry adapter. Neither transfers qualification into the shared kernel.

## Dependency order

SCI-001 -> SCI-002 artifact identity -> SCI-003 execution capsule -> SCI-004 experiment contract -> SCI-005 exposure/use separation -> SCI-006 dependency graph -> later falsifier, replication/triangulation, and Theory Atlas layers.
