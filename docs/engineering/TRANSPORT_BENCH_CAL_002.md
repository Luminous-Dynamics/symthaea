# TRANSPORT-BENCH-CAL-002 — Decision-bound benchmark evidence ledger

## Status
Source contract only. This document defines metadata semantics; it does not record benchmark performance and does not grant operational authority.
Parent programs: #6264, #6270, #6255.

## Core theorem

benchmark result
NaN
NaN
NaN

A benchmark may be useful evidence for one narrowly declared engineering proposition while remaining irrelevant to another. The same benchmark result must not be silently reused after its benchmark, evaluator, model, ODD, physical article, measurement basis, rights, or transfer calibration changes.

## Evidence-use record
Each future ledger record is bound to:
1. exact external registry generation and benchmark entry;
2. exact benchmark version, task, split and evaluator;
3. exact Symthaea subject/model/controller generation;
4. declared ODD and applicability envelope;
5. one engineering proposition;
6. a transfer hypothesis frozen before confirmatory results are revealed;
7. one or more independent transfer targets;
8. common-mode lineage;
9. baseline and ablation identities;
10. development and confirmatory exposure;
11. rights/use disposition;
12. resource and execution context;
13. observed transfer result and failure modes;
14. currentness and expiry conditions;
15. an explicit claim ceiling.

The ledger describes **evidence relevance**. It never creates legal, safety, certification, dispatch, actuator, regulatory or mission authority.

## Evidence-use states
ObservedOnly | Calibrating | ConditionallyDecisionRelevant | TransferFailed | OutOfProfile | Expired | Retired

These are descriptive lifecycle states. None means safe, approved, certified, deployable, authoritative, or physically executable.

### State meanings
- **ObservedOnly** — a result exists, but no qualified transfer relationship has been established for the declared proposition.
- **Calibrating** — prospective transfer evidence is being collected under a frozen hypothesis.
- **ConditionallyDecisionRelevant** — the declared benchmark-to-target relationship has met its exact preregistered calibration rule for the exact profile.
- **TransferFailed** — the preregistered transfer target did not support the expected relationship.
- **OutOfProfile** — later evidence is outside the previously declared applicability envelope; this is not a retroactive explanation for an in-profile failure.
- **Expired** — a declared temporal/currentness/measurement/rights condition has lapsed.
- **Retired** — intentionally removed from future decision use while preserving historical records.

## Prospective calibration
Before confirmatory evidence is observed, freeze: hypothesis_id, benchmark_subject, decision_proposition, predicted_observable, expected_direction, uncertainty_interval, independent_target, common_mode_roots, development_exposure, confirmatory_exposure, baseline, and ablations.
After execution, append: observed_result, transfer_class, direction_correct, effect_estimate, interval_coverage, failure_mode, applicability_boundary, and decision_consequence.
A result may be null, contradictory, or non-transferable. Those outcomes remain first-class evidence.

## Common-mode dependence
Different benchmark names are not automatically independent witnesses.
Track shared roots across source dataset, annotation/oracle, simulator, scenario generator, sensor assumptions, evaluator implementation, metric definition, pretraining exposure, and leaderboard/community feedback.
Removing a common-mode edge after observing an unfavorable result is invalid.

## Requalification triggers
A record is no longer Current when a material field changes, including benchmark version/task/split/evaluator; simulator or scenario generator; sensor/input contract; metric definition; Symthaea model/controller generation; ODD or applicability expansion; physical article/configuration generation; calibration or measurement expiry; newly discovered common-mode dependency; preregistered repeated-transfer failure rule; benchmark rights/use terms; confirmatory target; or development/leaderboard exposure assumptions.
Historical records remain immutable. A changed condition creates a new generation rather than rewriting the old result.

## Anti-Goodhart controls
benchmark optimization exposure != independent evidence != transfer calibration
Controls include frozen confirmatory target, simple baseline retention, candidate/baseline parity under transfer, exposure recording, evaluator-specific tuning disclosure, held-out hostile/metamorphic cases, preservation of worsened secondary metrics, benchmark-version generationing, null-transfer retention, and no post-result benchmark substitution.

## Decision consequence
A calibrated record may inform a later engineering experiment or design discussion only within its exact declared proposition/profile and only while its currentness conditions hold.
It does not rank transport systems; establish physical capability; establish safety; establish certification; establish regulatory approval; establish commercial rights; authorize an actuator, vehicle, dispatch or mission; or establish human benefit.

## Cross-domain intent
The semantics are deliberately portable to later materials, biology, manufacturing, sensing and productive-loop evidence ledgers. Domain adapters must retain their own physics, evidence owners, rights and claim ceilings.

## Claim ceiling
A future independent qualifier can establish only deterministic representation and mutation resistance of decision-bound benchmark calibration metadata. It cannot establish that a benchmark is generally valid, predictive, safe, certified, commercially usable, physically capable, or operationally authoritative.