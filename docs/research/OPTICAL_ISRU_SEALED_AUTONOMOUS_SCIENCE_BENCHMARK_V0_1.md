# Optical ISRU — sealed autonomous-science benchmark contract v0.1

**Status:** benchmark contract and standalone invariant tests authored; exact-head validation pending  
**Parent:** Symthaea #12570 — ISRU-BENCH-001  
**Related:** #12538 stage workflow; #12549 stage workflow reference; #12550 asynchronous buffer; #12552 asynchronous PR; Symtropy #5724 evidence interoperability; Symtropy #5734 evidence profile PR

## 1. What changed in the benchmark state of the art

The **Autonomous Science Bench (ASB) 0.1** was published on 1 October 2026. It evaluates the complete scientific-agent configuration—model, agent, tools and harness—on working lab tasks. It separates two final tests:

- **Discovery:** quality of the designs the system actually selected and measured.
- **Learning:** quality of its ranking of designs it never measured.

ASB 0.1 reports six pilot tasks across biology, chemistry and materials. Ten configurations ran three trials per task. The leading listed configuration scored 72.2% resolution, but the maintainers explicitly caution that small gaps are not meaningful with only three trials per task. The first benchmark version therefore offers a valuable protocol pattern, not an ISRU leaderboard we can compare directly.

ASB's backend progression is equally important:
- **B0 replay oracle:** recorded experimental outcomes, replayed deterministically;
- **B1 digital twin:** stochastic outcomes, physical limits, failures and noisy measurements;
- **B2 live laboratory:** actual instruments, queues, resource costs and faults.

The benchmark must keep one task schema and action interface while progressively moving the outcome/execution backend closer to physical reality.

Primary source: https://asb.open-ssi.org/  
Preprint: https://arxiv.org/abs/2610.18449

## 2. Field comparison: what an ISRU benchmark must include

| Reference | What it measures / contributes | What we add for multi-stage ISRU |
|---|---|---|
| ASB 0.1 (2026) | Full model-agent-tool-harness score; discovery and held-out learning gates; replay/twin/live progression | Causal stage graphs, qualified physical evidence, resource vectors, asynchronous jobs, assay-backed product acceptance |
| SDL benchmarking review (Digital Discovery 2026) | Acceleration factor (AF) and enhancement factor (EF); warns that both depend on objective-space and reference strategy | Separate experiment-count and wall-clock acceleration, independently accepted product per campaign hour, and matched stage-aware baselines |
| SDL platform metrics (Nature Communications 2024) | Autonomy, lifetime, throughput, precision, material use, accessible parameter space, optimization efficiency | Full evidence quality, calibration validity, maintenance/fouling/recovery, operator minutes, actual resource meter boundary |
| ADePT (Communications Chemistry 2026) | Adaptability/learning, dexterity, perception, task complexity | Retain these as separate hardware/robotic capability dimensions instead of a single autonomy label |
| Multi-stage BO (Digital Discovery 2026) | Intermediate proxies, partial stage evaluations, resumable sample inventory, nested stage-aware acquisition | Evidence-typed output stages, checkpoint lineage, pending-result handling and independent product assay gate |
| Pessimistic async BO (Digital Discovery 2026) | Placeholder policies for concurrent high-cost experiments | Explicit synthetic fantasy isolation, asynchronous instrument reservations, actual metrology receipts and real heterogeneous waits |

Sources:
- ASB 0.1: https://asb.open-ssi.org/
- Adesiji et al., *Benchmarking self-driving labs*, DOI: https://doi.org/10.1039/D5DD00337G
- Volk & Abolhasani, *Performance metrics to unleash the power of self-driving labs*, DOI: https://doi.org/10.1038/s41467-024-45569-5
- Salazar-Villacis & Benyahia, *The ADePT framework*, DOI: https://doi.org/10.1038/s42004-026-01932-9
- Torresi & Friederich, multi-stage BO, DOI: https://doi.org/10.1039/D5DD00572H
- Volk et al., pessimistic asynchronous BO, DOI: https://doi.org/10.1039/D5DD00477B

## 3. Paradigm shift: benchmark scientific capability, not optimizer cleverness

The unit of competition should be a **qualified scientific decision**, not a prediction, model score, completed experiment or maximum proxy.

The full planner's action space should eventually include:
- start a new material candidate;
- advance an existing candidate to its next causally admissible stage;
- request a calibration/reference measurement;
- pause while awaiting evidence;
- schedule maintenance/recovery/requalification;
- select an independent assay;
- reject a candidate or refuse a proposed action.

That expands the optimization problem from "choose the next point" into "choose the next evidence-producing action under causal, metrology, resource and authority constraints."

The benchmark applies two independent scientific gates:
1. Discovery gate — did the campaign measure and independently qualify useful outcomes?
2. Learning gate — can it rank/predict the never-measured candidate set?

A run is not resolved if either gate is unresolved. Safety, evidence and hard resource compliance are non-compensable hard gates. No weighted score can average away a false product acceptance or prohibited stage start.

## 4. Required baselines

All methods use the same task, candidate pool, feasible actions, starting state, objective, resource budget, sealed oracle, evidence gates and paired seeds.

1. Fixed expert/reference workflow.
2. Random feasible sampling.
3. GP-EI / conventional one-stage BO.
4. Batch BO.
5. Multi-stage BO with intermediate proxy and resumable partial work.
6. Asynchronous realistic-mean fantasy.
7. Pessimistic constant-liar / confidence-bound fantasy.
8. Seeded-random fantasy.
9. No-fantasy asynchronous BO.
10. Symthaea EIG / EVSI.
11. Stage-aware EIG / EVSI.
12. Full Luminous planner combining stage DAG, evidence qualification, asynchronous scheduling and resource Pareto constraints.

The full configuration must be named: model, agent, tools, harness, toolchain, build commit and policy/configuration digests. "Model X" by itself is not a reproducible benchmark result.

## 5. Metrics — publish separate panels

### Scientific performance
- discovery gate success and independent product/assay quality;
- best qualified value versus experiment count and elapsed wall time;
- learning gate success and held-out ranking quality (Spearman/Kendall or task-calibrated metric);
- held-out predictive interval coverage and proper log score;
- proxy-to-final calibration on whole-batch and whole-family holdouts;
- Pareto hypervolume/regret for genuine multi-objective tasks.

### Efficiency and platform performance
- experiment acceleration at a fixed independently qualified outcome threshold;
- wall-clock acceleration at that same threshold;
- enhancement at a fixed experiment count and at a fixed wall-clock budget;
- accepted product per total campaign hour;
- time-to-qualified-decision, including waits, calibration, assays, maintenance and queueing;
- demonstrated throughput including preparation and measurement;
- rig utilization, queue time, idle time, maintenance downtime;
- total electrical input at a declared meter plane, feedstock, assay spend, consumables and maintenance;
- human intervention time and operational lifetime;
- demonstrated versus merely theoretical accessible parameter space.

### Trust and robustness
- synthetic-fantasy to physical-observation contamination (target zero);
- proxy/captured/purified to accepted-product promotion (target zero);
- use of unqualified or out-of-scope calibration evidence (target zero);
- unauthorized stage starts and hard-budget overruns (target zero);
- claim-level provenance completeness;
- late/out-of-order/duplicate/missing result behavior;
- stale/revoked/scope-mismatched calibration behavior;
- compatible partial-work salvage and incompatible-lineage refusal;
- replay determinism and independent audit time.

### Robotic capability where relevant
Report ADePT axes individually: adaptability/learning, dexterity, perception and task complexity. Do not infer dexterity or perception capability from a strong planning score.

## 6. AF and EF interpretation

The SDL literature uses two complementary views:

- **Acceleration factor:** the ratio of reference experiments required to hit a fixed outcome threshold to candidate experiments required to hit that same threshold. Report a separate wall-clock version using complete end-to-end elapsed time.
- **Enhancement factor:** the improvement over the same baseline outcome at a fixed campaign budget. Report it at both a fixed number of experiments and a fixed wall-clock duration.

For Optical ISRU, the threshold must be an independently qualified outcome, not a raw predicted value or intermediate proxy. The reference must be declared before the sealed run. Also report the full best-so-far curves, distributions and confidence intervals. AF/EF are not context-free universal properties of an algorithm; they depend on the chosen task, property space and reference strategy.

The SDL benchmarking review reports a median AF around 6 relative to reference strategies in reviewed studies and finds EF strongly dependent on the property space, often peaking around 10–20 experiments per input dimension. These are literature summaries, not an ISRU performance target and not a fair basis for comparing unlike tasks.

## 7. Statistical and anti-contamination protocol

### B0/B1
- At least 30 paired stochastic seeds per primary simulation benchmark cell; increase using a power analysis where required.
- Same seed/world/candidate pool for every paired method.
- Public development set; sealed final hidden set.
- Freeze oracle commitment, task manifest, budgets, feasible actions, gates and analysis plan before sealed evaluation.
- Use confidence intervals and full distributions; cluster bootstrap by material batch/family/lab where applicable.
- Preregister primary endpoints and multiple-comparison handling.
- Preserve failed, abandoned, censored and missing outcomes.

### B2
- Physical replicate counts come from a prospective power analysis, not from the simulation seed rule.
- Randomized or blocked run order; independent assay and sample/custody records.
- Record assisted and unassisted operating time separately.
- No simulation run can be presented as a physical replicate or live-lab result.

### Statistical claim eligibility
Do not claim superiority from one seed, one batch, one material family, one lab or a point estimate with an interval crossing parity. Require the preregistered trial count, a paired comparison and a confidence interval that clears the claim threshold.

## 8. Initial improvement hypotheses (not achieved results)

- At least 1.5× faster time-to-qualified-decision than the strongest fair eligible baseline, with paired 95% lower confidence bound above 1.0.
- At least 20% more independently accepted outcome per total wall-clock hour than the strongest stage-aware asynchronous baseline, while preserving non-inferior held-out learning/calibration and staying inside every hard resource budget.
- Zero false evidence promotions and zero unauthorized stage starts on the frozen adversarial suite.
- At least 90% salvage of partial work when exact lineage remains physically compatible; separate incompatible resume cases.
- Nominal 90% predictive intervals achieve coverage within a preregistered tolerance on held-out batches/material families.
- Cross-lab transfer reports required adaptation cost and uncertainty shift; no zero-shot generalization claim without evidence.

These are target hypotheses. They are not current measured results.

## 9. B0 replay oracle implementation

`research/asteroid-optical-isru-v0.2/replay_oracle.rs` is the first executable backend slice.

- The agent-side API sees candidate metadata, measurable candidates and holdout candidate metadata, but receives outcome values only after it requests an allowed measurable candidate.
- Holdout candidates cannot be measured. Their labels remain in a separate private store held by the evaluator.
- Request IDs are idempotent; reusing one request ID for another candidate is rejected.
- A candidate cannot be measured twice in B0 v0, and the maximum experiment count is a hard limit.
- Campaign closeout is required before discovery or learning grading.
- Discovery grading uses only the selected, replayed outcomes.
- Learning grading requires predictions for the complete sealed holdout and returns aggregate rank correlation and interval coverage, never the raw holdout labels.
- Candidate/outcome mapping, candidate-set overlap, duplicate IDs, invalid numbers and missing holdout records are rejected at construction or grading.

The agent and grader are separate Rust interfaces. For an actual sealed tournament, the grader and hidden outcome store must run in a custodian-controlled process or service; merely hiding a Rust field is not a security boundary against arbitrary code in the same process.

B0 is deterministic by design. It does not simulate heterogeneous durations, noise, fouling, failure, calibration drift or asynchronous queues. Those belong in B1, not in the replay oracle.

## 10. Executable contract

`research/asteroid-optical-isru-v0.2/benchmark_contract.rs` implements:
- backend/task/algorithm identities;
- exact configuration identity;
- a frozen benchmark manifest and sealed-oracle commitment;
- a comparability key requiring identical task/backend/candidate/action/start/objective/budget/gates/oracle/paired seed;
- separate discovery, learning, safety, evidence and resource gates;
- a run disposition where hard failures cannot be offset by high yield;
- separate experiment-count and wall-clock acceleration helpers;
- enhancement factor with explicit invalid-reference rejection;
- minimum 30 unique paired simulation seeds;
- a separate physical replication plan that requires power, randomization, blocking and independent assay plans;
- 16 benchmark-contract invariant tests plus 15 B0 replay-oracle tests.

### Reproduce locally

    rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/benchmark_contract.rs -o /tmp/isru-benchmark-contract-tests
    /tmp/isru-benchmark-contract-tests
    rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/replay_oracle.rs -o /tmp/isru-replay-oracle-tests
    /tmp/isru-replay-oracle-tests

The contract is not the benchmark dataset or an executed tournament. No scientific superiority, physical yield, lab throughput or live-rig qualification is claimed.

## 10. Work needed before a public leaderboard

1. Seal an Optical ISRU B0 task with fixed candidate pool and hidden outcomes.
2. Add a B1 stateful twin with heterogenous stage durations, queues, fouling, calibration drift, failed jobs and maintenance.
3. Connect the existing stage workflow and async buffer to one action API.
4. Build adapters for GP-EI, standard BO, MSBO-style stage-aware BO and async fantasy baselines.
5. Add the Symtropy portable evidence profile and external RO-Crate/PROV validation.
6. Ask an independent lab domain expert to verify task definitions, physical limits and outcome gates.
7. Run the sealed paired benchmark; publish all trial records and uncertainty with hidden-oracle custody attestation.
8. Only after B0/B1 invariants hold, progress toward B2 with external rig safety authority.

The objective is a benchmark other laboratories can reproduce and challenge, not a private score that only Luminous can interpret.
