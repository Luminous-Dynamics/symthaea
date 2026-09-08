# SCI-005 — Exploratory / Confirmatory Separation v1 — Summary

**Status:** architecture-only; non-authorizing; non-qualifying.

SCI-005 prevents outcome-aware exploratory/tuning material from being relabeled as fresh confirmatory evidence while preserving legitimate cumulative and adaptive science.

## Core rule

Exploratory evidence is not confirmatory evidence, and a later label/timestamp/copy does not reset outcome exposure.

The architecture separates four things:

1. exact artifact identity;
2. information projection actually visible;
3. exposure event to a decision process for a declared target/use;
4. prospective evidence eligibility derived from the relevant lineage.

## Exposure is contextual

The same raw artifact is not globally “fresh” or “spent” for every possible question.

Eligibility depends on what information was exposed, to whom/what, for which target and evidentiary use.

This avoids both extremes:

- treating any historical access as permanent contamination for all future science;
- treating copied/transformed/repacked data as magically fresh.

## Decision process includes machines

Exposure can flow through:

- humans;
- model training;
- hyperparameter/model selection;
- retrieval systems;
- learned grammar/macros;
- prompts/context builders;
- leaderboard feedback;
- persistent institutional pipelines.

“The current analyst never saw the labels” is insufficient if the model/tool lineage already carries them.

## No direct conversion

There should be no authority path from exploratory result directly to confirmatory evidence.

The legitimate path is:

exploration -> new hypothesis/design -> new SCI-004 contract -> verified preregistration -> eligible prospective evidence -> SCI-003 execution -> later measurement/adjudication.

The exploratory result remains useful provenance for hypothesis/prior/design construction.

## Cumulative priors are allowed

Exploratory data may legitimately inform priors, hypotheses, models, or experiment design when that dependency is explicit.

Fresh prospective evidence can still provide confirmation, but the exploratory source must not be counted again as independent fresh likelihood evidence.

## Hidden benchmark lifecycle

A benchmark can move from sealed to partially exposed, score-exposed, fully revealed, and historical/spent for a declared scope.

Relevant exposure is historical and cannot be undone. Access revocation affects future access only.

A fresh confirmatory generation requires substantive new eligible evidence, not merely a renamed split or copied file.

## Repeated-query leakage matters

Hidden labels are not the only information channel. Repeated leaderboard/score queries can enable adaptive overfitting, especially for autonomous systems.

Query budget, score precision, subgroup feedback, and adaptation policy may therefore belong to exposure semantics.

## Confirmatory failure remains evidence

A null/negative confirmatory result remains evidence for its campaign.

If it is then used to tune a successor, that successor enters a new development lineage and needs a new prospective contract/evidence path for confirmation.

## Cross-validation and adaptive science remain possible

SCI-005 does not impose a simplistic one-view rule.

Prospectively frozen nested cross-validation, sequential experiments, and SCI-004 adaptive experiment planners can remain confirmatory within their declared scope. The key requirement is that adaptation semantics are precommitted and the resulting exposure lineage is retained.

## First implementation slice

Start with append-only exposure history only:

- `InformationProjectionV1`;
- `DecisionProcessIdentityV1`;
- `EvidenceUseV1`;
- `ExposureEventV1`;
- `ExposureLedgerV1`.

Do not issue positive prospective eligibility in the first tranche.

A later pilot may derive a private-fielded `ProspectiveEvidenceEligibilityV1` from exact SCI-004 contract/preregistration + exposure/custody/dependency evidence.

## Dependency order

SCI-001 -> SCI-002 artifact identity -> SCI-003 execution capsule -> SCI-004 experiment contract -> SCI-005 exposure/use separation -> SCI-006 dependency graph.

SCI-006 is needed to detect indirect information ancestry that exposure history alone cannot prove, such as labels -> trained model -> downstream evaluation or prior discovery -> learned grammar -> rediscovery.
