# Subterra Survival-Conditioned Scientific Return Bench v0.1

**Date:** 2026-11-16  
**Status:** Proposed protocol and targets; no results claimed  
**Owner:** Symthaea / Luminous Dynamics

## 1. Research question

Does a scientific exploration policy acquire and return more decision-relevant evidence when future sensing opportunities can be lost through safe retreat, mission interruption, robot loss, relay failure or resource exhaustion?

This benchmark is distinct from SLAM accuracy. A mapping algorithm estimates geometry and pose. This benchmark asks whether a planner uses available estimates and uncertainty to choose a better sequence of scientific observations while preserving the conditions required to finish and return evidence.

## 2. Benchmark outcome classes

Every measurement action must define a complete conditional outcome distribution for each hypothesis. Each outcome belongs to one of the following:

| Outcome | Future sensing | Evidence disposition | Risk interpretation |
|---|---|---|---|
| Observation | Continue with posterior update | Observation receipt must be returned to the planner | No interruption assumed by this class |
| Safe retreat | Stops this survey branch | Previously acquired evidence may be returned | Not robot loss; terminal science loss is explicit |
| Mission interrupted | No continuation | Partial evidence only; disposition must be explicit | Counts against mission-interruption ceiling |
| Robot lost | No continuation | No return is assumed unless a separate recovery model provides evidence | Counts against both interruption and robot-loss ceilings |
| Relay lost / store-and-forward | Stops the declared live survey sequence | Evidence retained locally; later retrieval must be separately modelled or charged in terminal loss | Not automatically equated with evidence loss |

Terminal branch losses are caller supplied and must account for truncated science value, lost future observations and data-delivery consequences. The planner does not invent terminal rewards.

## 3. Baselines

All methods must use the same prior, likelihood model, terminal-loss model, feasible action set, resource budget and mission-risk limits:

1. Stop-now decision (no additional measurement).
2. Passive / fixed measurement sequence.
3. Random feasible measurement selection.
4. Frontier / map-coverage greedy.
5. Shannon expected information gain greedy.
6. Information gain per declared resource cost.
7. Risk-penalized information gain.
8. Chance-constrained information-gain planning.
9. CVaR / tail-risk-constrained planning.
10. One-step EVSI.
11. Receding-horizon greedy EVSI.
12. Finite-horizon survival-conditioned EVSI.
13. Exact dynamic-programming oracle on small frozen problems only; this is a reference bound, not a deployable competitor.

The baseline implementations must be independently identified and hashed. A method is not reproduced merely because its name appears in a comparison table.

## 4. Required metrics

### Decision quality
- expected terminal decision loss;
- realised decision regret after complete or interrupted runs;
- EVSI and entropy reduction, reported separately;
- probability-calibration and proper scoring rules for outcome likelihoods;
- abstention and open-world coverage.

### Survival / continuation
- mission-interruption probability;
- robot-loss probability;
- safe-retreat frequency;
- conditional probability of reaching each planned follow-up observation;
- future observations lost due to absorbing outcomes;
- reserve / return feasibility from the independent mission supervisor.

### Scientific return
- verified observations acquired;
- verified observations returned to the surface;
- fraction of acquired evidence retained under store-and-forward;
- provenance / calibration completeness;
- unique science value rather than raw observation count;
- decision relevance of returned evidence.

### Resource consumption
- energy, duration, data volume and exposure;
- onboard memory and planning expansions;
- inter-robot bytes, latency, dropped packets and relay dependence;
- p50/p95/p99 planner runtime on declared target compute.

Do not merge these into a single opaque score. Publish the metric vector and all hard-gate outcomes.

## 5. Proposed targets

All targets are hypotheses for a preregistered frozen suite, not universal safety standards or achieved results.

| Metric | Initial target |
|---|---|
| Decision regret | At least 20% lower than the strongest matched passive / uniform / information-gain baseline |
| Survival-adjusted scientific return | At least 15% more verified, decision-relevant observations delivered under equal expected energy/time/data budgets, with no relaxation of risk ceilings |
| Hard-risk constraints | Zero policy outputs above configured mission-interruption or robot-loss limits |
| Future-opportunity accounting | No continuation value assigned after SafeRetreat, MissionInterrupted, RobotLost or RelayLostStoreAndForward terminal branches |
| Budget correctness | Zero path-wise energy, time, data or exposure budget violations |
| Calibration | Proper scoring rules and reliability curves reported on site-held-out data before any superiority claim |
| Search boundedness | Deterministic reproducibility and explicit non-pass when the configured node budget is exceeded |
| Evidence preservation | Zero unknown / inferred claims silently promoted to directly observed in the frozen adversarial suite |
| Return and data custody | No "delivered" label without a verified observation/data-return receipt |

The 15% science-return target is meaningful only after the measurement definition, data-return boundary, budget, and strongest comparator have been frozen.

## 6. Mandatory adversarial scenarios

- early rockfall / blocked passage before planned follow-up scan;
- safe retreat after the first positive/negative observation;
- low-probability but high-consequence robot loss;
- relay node loss with verified store-and-forward;
- relay loss with unverified buffer state (must not assume data preserved);
- battery reserve decline after the action was proposed;
- delayed, stale, duplicated and reordered observation messages;
- common-mode sensor bias represented by a shared latent fault hypothesis;
- apparent independent products derived from one physical acquisition;
- likelihood misspecification and an omitted geological hypothesis;
- a measurement that reduces entropy but cannot alter the optimal decision;
- a high-value measurement that exceeds one hard resource budget;
- risk estimate outside the validated domain;
- search-node exhaustion and numerical overflow;
- contradictory evidence that remains unresolved until a discriminating observation.

Each failure case must have a positive control and a negative control. Hard risk constraints remain enabled in all physical tests; only offline/synthetic mathematical ablations may vary objective terms.

## 7. Experiment design

1. Freeze the hypothesis set, explicit `unknown` mass, decision loss matrix, likelihood model and terminal losses before evaluation.
2. Hold out whole caves / sites. Random scan fragments from one cave must not be split across train and evaluation.
3. Independently calibrate likelihoods; do not use evaluation outcomes to fit likelihoods or tune planner thresholds.
4. Compare all methods on the same scenario seeds and paired action/outcome samples where mathematically appropriate.
5. Include both expected metrics and tail outcomes. Report confidence intervals and per-site dispersion.
6. Freeze the resource budget and risk ceilings before comparing planners.
7. Run the exact dynamic-programming oracle on tiny examples to check finite-horizon search and pruning.
8. Publish deterministic policies, scenario traces, all terminal branches, evidence receipts and configuration digests.
9. Report planner runtime, search expansions and memory; no unconstrained server-side oracle is assumed.
10. Separate simulated hazard probabilities from empirically measured hardware failure rates.

## 8. Limits of the v0.1 reference planner

The standalone Rust planner currently implements:
- a finite static hypothesis set with a mandatory nonzero `unknown` hypothesis;
- one set of caller-supplied conditional outcome likelihoods per action;
- terminal safe retreat, interruption, robot loss and relay-store-forward outcomes;
- separate mission-interruption and robot-loss probability ceilings;
- branch-specific remaining resource budgets;
- stop-now decisions and a bounded contingent policy tree;
- deterministic action ordering and an expansion cap.

It does **not** yet implement:
- likelihood calibration or model-discrepancy estimation;
- automatic joint-likelihood dependence / common-mode sensor models;
- geometric path planning, dynamic topology or continuous robot state;
- explicit relay buffer-capacity, data exfiltration or custody verification;
- robust optimization over uncertainty sets;
- CVaR or distributionally robust objectives;
- multi-agent task allocation;
- actual Symtropy scenario execution or field data ingestion.

The current model's conditional-independence assumption for repeated observations must be tested and then replaced or explicitly bounded before the planner can consume real correlated sensor streams.

## 9. Promotion gate

Promotion from mathematical reference to benchmark claim requires:
- executable exact-head formatting, compile, tests and strict Clippy;
- differential comparison with a separate small exact oracle;
- hand-computed cases and metamorphic tests;
- frozen scenario corpus and positive/negative controls;
- calibration and leakage-resistant evaluation;
- reproducible resource / terminal-branch accounting;
- independent review of risk and terminal-loss assumptions.

Robot-in-the-loop, analogue, flight and planetary evidence are separate higher tiers and cannot be inferred from this finite software benchmark.
