# SDCB scientific-discovery benchmark interoperability v1

**Status:** research protocol / adapter plan  
**Parent:** [SDCB v1](SPECTRAL_DYNAMICAL_COGNITION_BRIDGE_V1.md)  
**Date:** 25 November 2026

## 1. Objective

Do not build a competing miniature scientific-agent leaderboard in isolation. Reuse established scientific-discovery task suites where their licenses and harnesses permit, then add narrowly defined SDCB measurements for identifiability, active experiment selection, surprise recovery, evidence provenance and unmeasured-candidate learning.

The target system is the complete research loop:

question → competing hypotheses → experiment design → budgeted measurement → observation → model revision → prediction of unmeasured cases → independent audit

A correct final answer is insufficient when the path involved a poor experiment, ignored disconfirming observations, miscalibrated uncertainty, or unsupported claims.

## 2. Existing benchmark map

| Benchmark | Scope / published information | SDCB integration | Important limitation |
|---|---|---|---|
| Autonomous Science Bench (ASB) 0.1 | Real lab workflows from active labs; six pilot tasks across three domains; grades experiment discovery and learning about designs never measured. Its current page lists GPT-6 Astra Codex 72.2% and Opus 5.5 Claude Code 66.7%. | First priority for sequential experiment design and unmeasured-design ranking. Reuse its distinction between design quality and learning quality. | Six tasks are a pilot, not a high-powered general capability estimate. Results are full-system and harness-specific. |
| FIRE-Bench | Agents rediscover established, verifiable insights from recent ML research by independently designing and running experiments. The ICML 2026 paper reports strongest evaluated agents below 50 F1, substantial run-to-run variance and failures in design, execution and evidence-based reasoning. | Use as an evidence-grounded scientific rediscovery track. Add explicit provenance and intervention receipts, but preserve FIRE's source task and scoring. | Rediscovery of known findings is not the same as genuinely novel discovery. |
| SciAgentArena | Approximately 200 interactive scientific research tasks with stepwise verification, drawn from real research scenarios across domains. | Adapter for interactive end-to-end research. Record tool actions, intermediate evidence, model changes and cost. | Current reported weakness in novel insight and self-directed exploration makes it a useful challenge set, but not a proof that a single architecture is best. |
| AstaBench | More than 2,400 scientific problems spanning literature understanding, code execution, analysis and discovery. Stanford AI Index 2026 reports best agent score around 0.53 at about $3.40 per problem in the captured evaluation summary. | Broad task coverage and cost curve. Use the exact current snapshot and official scoring protocol. | Aggregated score does not isolate active experiment design or proof/evidence integrity. |
| SorryDB | A 1,000-task January 2026 snapshot of open Lean obligations from 78 real formalization projects; tasks are pinned to source revision and Lean environment, and solutions are independently compiled. | Realistic formal proof-completion track and fresh source-lineage evaluation. Attach SDCB scope/obligation accounting and explanation quality as separate measures. | The public leaderboard is an evolving snapshot; compare the exact snapshot and prover budgets, not a changing top row. |
| Goedel-Prover-V2 | ICLR 2026 open-source Lean prover: 8B model reports 84.6% MiniF2F pass@32; 32B reports 88.1% pass@32 standard and 90.4% self-correction; PutnamBench result 86 problems at pass@184. | Strong formal-proving baseline, with exact pass@k and search budget pinned. | Proof completion scores do not measure scope explanation, independent checker diversity or human transfer. |
| Open Agent Leaderboard / Exgentic | Full-system agent evaluation across tools, tasks and constraints; exposes quality and cost, with trajectory/run-level data. The project reports failures can cost 20–54% more than successful runs. | Adopt agent-scaffold and tool-shortlisting ablations; record failed-run cost as well as successful-run cost. | General agent tasks do not replace scientific or formal-mathematics task sets. |

## 3. SDCB contribution beyond the upstream suites

The SDCB adapter should not change upstream task labels or quietly re-score the original benchmark. It should produce a parallel evidence report with the following dimensions:

1. **Hypothesis quality:** were competing explanations explicit, discriminable and tied to observable predictions?
2. **Experiment value:** expected information gain or expected reduction in decision-relevant uncertainty per unit cost, computed using only information available before the experiment.
3. **Budget efficiency:** best-so-far performance, cumulative regret, experiment count, wall time, tool calls and cost.
4. **Surprise response:** did the agent detect when observations contradicted its model, update the hypothesis set and change the next experiment?
5. **Unmeasured-candidate learning:** can it rank/predict candidates never measured, with calibration and uncertainty?
6. **Identifiability discipline:** did it distinguish a predictable observable from an unidentifiable hidden mechanism?
7. **Evidence integrity:** are observations source-pinned, independently captured and bound to the exact task/system/run?
8. **Reproducibility:** can another evaluator replay the run and reproduce the score and conclusion?

Keep the discovery outcome and learning outcome as separate gates. A method that chooses a good experiment but fails to learn the response model has not completed the loop; a method that predicts unmeasured designs without conducting informative experiments has not demonstrated autonomous discovery.

## 4. Sequential experiment-design protocol

For each task:
- publish the allowed design space, observation interface, prior information, measurement costs and hard constraints;
- withhold the reward/response surface and the task-specific hidden seed until the evaluation round is committed;
- provide observations only after the agent selects an experiment;
- allow a fixed number of adaptive rounds and a fixed resource budget;
- at the end, score both selected/observed designs and a held-out pool of never-measured designs;
- reveal the hidden task after the evaluation window so the run can be replayed.

Baselines:
- uniform random allocation;
- fixed space-filling design;
- one-shot greedy selection;
- uncertainty sampling;
- UCB / Thompson sampling for bandit-like tasks;
- Gaussian-process Bayesian optimization for suitable continuous design spaces;
- random forest / ensemble surrogate optimization;
- oracle acquisition function with exact hidden response (upper bound only);
- human-designed experiment under the same information and budget, where a valid human study is available.

No one baseline is suitable for every task. The comparison must match the problem structure, observation noise, prior, feedback, stopping rule and cost model.

## 5. Metrics

### Discovery gate
- best observed objective versus budget;
- simple-regret and cumulative-regret curves;
- fraction of budget spent on informative experiments;
- constraint violations / invalid experiment count;
- time and cost to first useful discovery;
- robustness under noise, missing measurements and hidden regime change.

### Learning gate
- held-out rank correlation and top-k recall for never-measured designs;
- predictive log score / proper scoring rule;
- calibration and interval coverage;
- ranking stability across seeds;
- out-of-domain performance on shifted design families;
- accuracy of model-change / surprise detection.

### Integrity and deployment
- evidence identity and provenance failures;
- reproducibility by independent runner;
- wall-clock, peak memory, tool/API calls and measured energy;
- failed-run cost and recovery time;
- abstention correctness when the response is not identifiable.

Do not collapse discovery, learning, evidence integrity and resource use into one weighted score. Report a Pareto frontier and non-compensatory gates.

## 6. Provisional targets (not results)

- Beat random and fixed space-filling baselines on simple-regret under the same measurement budget.
- Demonstrate lower simple-regret than greedy selection on at least one preregistered multimodal/noisy design family.
- Detect deliberately injected prior misspecification before wasting more than one further experiment in the frozen control cases.
- On never-measured candidates, beat the training-only baseline and report proper scoring/calibration, not only ranking.
- Preserve a correct abstention when two hypotheses are observationally equivalent.
- Select a discriminating sensor/probe when the experiment space contains one, and account for its measurement cost.
- Replicate results across at least five seeds and a held-out design family.
- Keep discovery quality and learning quality as separate outcomes.

These are project targets and must be revised only by a versioned amendment before a confirmatory run.

## 7. Implementation sequence without waiting for hosted runners

1. Add the four design-only sequential-experiment tasks to the SDCB pilot manifest.
2. Reuse the current sensor-selection reference control as a tiny active-measurement oracle.
3. The Rust reference harness now includes deterministic Beta–Bernoulli posterior updates and UCB1-style arm selection with conjugate-posterior, untried-arm, greedy-limit and invalid-input controls.
4. Add a small synthetic response-surface fixture with fixed cost/observation noise and hidden held-out candidates.
5. Define a common run/observation/experiment receipt that composes with the existing SDCB receipt and the Luminous evidence architecture.
6. Build read-only adapters for ASB, FIRE-Bench, SciAgentArena, AstaBench and SorryDB; no adapter is allowed to change upstream scoring.
7. Reproduce one upstream result per family before any superiority claim.
8. Run the synthetic pilot only after the local Rust toolchain is available; in the meantime, validate schemas, source pins, reference vectors and negative controls.

## 8. Claim boundaries

- ASB's six-task result is a pilot snapshot, not universal agent capability.
- FIRE-Bench rediscovery is not novelty.
- SciAgentArena performance is not a laboratory deployment certification.
- AstaBench aggregate score is not a direct measure of causal experiment quality.
- SorryDB / MiniF2F / PutnamBench proof completion is not proof comprehension or human transfer.
- A good predicted acquisition score is not evidence of a correct measured outcome.
- An experiment receipt proves only what its independent measurement and attestation chain actually establishes.

No SDCB score, reproduced baseline, physical experiment, independent evaluation or SOTA result is claimed by this adapter plan.
