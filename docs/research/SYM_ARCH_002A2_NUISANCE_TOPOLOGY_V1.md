# SYM-ARCH-002A2 — Nuisance Topology Boundary v1

**Status:** statistical design hardening; no architecture result

**Tracks:** #55, stacked in #58

## Why this exists

A representation/learner/stream realization can enter an experiment in two materially different ways:

1. **nested-independent** — each nuisance realization belongs to only one generated environment;
2. **crossed-shared** — the same nuisance realization is reused across multiple generated environments.

Those observations do not have the same dependence structure.

If the same representation seed, learner seed, stream ordering, or other nuisance realization is reused across environments, independently resampling nuisance runs inside each environment can treat a shared source of variation as though it were independent. That can distort uncertainty and later power calculations.

The topology is therefore part of the frozen statistical design, not an implementation detail.

## `NestedIndependent`

The scientific guard requires every nuisance digest to be globally unique across environments.

The existing environment-first + nested-run hierarchical percentile bootstrap is appropriate only after this condition is established.

A repeated nuisance digest across two environments is rejected instead of being silently treated as nested.

## `CrossedShared`

A crossed design must expose the same complete nuisance-identity grid in every environment. Missing or substituted nuisance cells make the design invalid for the v1 crossed estimator.

A2 now supplies a two-way environment × nuisance percentile bootstrap:

1. sample environments with replacement;
2. sample nuisance identities with replacement **once per replicate**;
3. apply those sampled nuisance identities across all sampled environments;
4. average the sampled Cartesian product;
5. report the 2.5% and 97.5% bootstrap percentiles.

Sampling the nuisance identities jointly across environments preserves the fact that a reused nuisance realization is shared rather than independent within each environment.

The v1 crossed estimator treats the combined representation/learner/stream nuisance digest as one crossed cluster. If later experiments need separate variance components for representation seed, learner seed, and stream seed, that is a distinct multiway-random-effects extension and must not be inferred from this combined digest.

## Prospective power boundary

The existing v1 prospective-power simulator was written for nested-independent nuisance runs.

The topology-checked scientific entry point therefore behaves as follows:

- `NestedIndependent` → validate global uniqueness, then use the existing DEV-based prospective-power simulator;
- `CrossedShared` → **fail closed** with an explicit unsupported-design error.

A crossed CONFIRM design must not use the nested power recommendation.

Before crossed prospective power is allowed, a later implementation must simulate both independent environment sampling and shared nuisance-cluster sampling inside each future study and validate that procedure separately.

This restriction is intentional. Missing power support is safer than a precise-looking sample-size recommendation from the wrong dependence model.

## Confirmatory freeze requirements

Before opening CONFIRM, freeze at least:

- nuisance topology (`nested_independent` or `crossed_shared`);
- exact definition of the nuisance digest;
- whether representation, learner, and stream seeds are reused across environments;
- environment count;
- nuisance count or runs per environment;
- missing-cell policy (v1 crossed requires none);
- uncertainty procedure;
- prospective-power procedure and version;
- bootstrap seeds/resample counts;
- primary metric, comparator, SESOI, and direction.

Changing from nested to crossed after observing results creates a new analysis specification; it is not a harmless re-labeling.

## Claim ceiling

This hardening supports only:

> SYM-ARCH-002A2 distinguishes nested-independent from balanced crossed nuisance designs, applies topology-specific uncertainty checks, and refuses to use the v1 nested prospective-power simulator for crossed data.

It does not establish that either design is preferable, does not choose the eventual CONFIRM topology, and carries no Symthaea capability claim.