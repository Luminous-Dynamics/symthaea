# CMAQ-002 WorldTest / AutumnBench adapter v0.1

**Status:** metadata-only source lock, custody fixture and Rust reset/replay trace diagnostic. No data or simulator has been downloaded or run.  
**Owners:** #8908 (active causal learning), #9094 (external benchmark bridge), #9176 (shared frozen-run manifest).

## Motivation

WorldTest asks whether an agent learns a reusable environment model that can answer new environment-level queries, not merely predict the next observation or maximize reward on the same rollout distribution. Its two-phase protocol separates reward-free exploration from a query in a related but different challenge environment.

The ICML 2026 paper reports 43 environments, 129 tasks, 19 stochastic environments, 517 human participants and five frontier models. The public release is reported as 20 environments / 60 tasks; do not describe a public-subset evaluation as a full-paper reproduction. The withheld 23/69 counts are derived differences, not a claim that those tasks can be downloaded. Sources: [PMLR paper](https://proceedings.mlr.press/v306/warrier26a.html), [author project page](https://archanarw.github.io/projects/autumnbench.html), [WorldTest presentation](https://datvo06.github.io/assets/talks/worldtest/).

This complements CMAQ-002's finite EVoI simulator: EVoI measures which safe probe to perform inside a small known hypothesis catalog; WorldTest asks whether reward-free exploration creates a model that generalizes to new queries in a related environment.

## Source and data boundary

The lock pins:
- Autumn.cpp at 0929d175c3d296de091ac4b998d09b9b55a38f67 (MIT; C++20 native interpreter, plus WASM/Python/Julia).
- MARAProtocol at d0d4e9251151778415cc5195c5dfb8296a5ded24 (MIT; protocol buffers pinned by file blob SHA).
- The public data manifest endpoint, without claiming immutable dataset identity or per-file checksums.

The two pinned commits are unsigned. The public data license is reported as CC BY 4.0 by a secondary catalogue, but remains pending per-file manifest and attribution review. The dataset is not downloaded. The MARA reference runner requires Python 3.12 and Protobuf tooling; Symthaea should implement a thin Rust adapter to the pinned protobuf contract rather than importing the Python runner into production. Use the native Autumn interpreter as a pinned external process; do not vendor code or data.

## Protocol custody

Exploration:
- Reward is absent.
- Candidate sees only the configured observation/action interface and actual observations.
- Reset is a first-class action and is charged to the interaction budget.
- Predicted future observations are never promoted to acquired observations.

Test:
- Use a related but different challenge program; the exploration and test program digests must differ.
- Query family is masked-frame prediction, change detection or planning.
- Task prompt may be visible as the task requires; answer key, true change timestep and evaluator scoring artifacts remain evaluator-only.
- Program-visible reference agents are a separate interface arm, not a matched comparison to observation-only agents.

The JSON Schema fixture validates this metadata/custody contract only. It does not establish actual dataset integrity or a successful simulator adapter.

## Rust reset/replay diagnostic

The new module src/worldtest_exploration.rs audits a bounded exploration trace and requires a terminal test-phase transition. It reports:
- action count, reset count and reset share of all action-plus-reset interactions;
- nonempty exploration segments separated by resets;
- adjacent segment pairs and their longest-common-subsequence (LCS) ratio;
- mean and median reset-replay LCS ratio.

The LCS normalization is Symthaea-defined: LCS length divided by the longer segment. It is not claimed to reproduce the paper's exact LCS normalization. Reset count and replay similarity are descriptive, not objectives; resetting more often is not rewarded.

The WorldTest presentation reports human reset share around 12.5% of actions, with at least one reset in every environment, versus roughly 1.4%-7.1% for the evaluated models. It reports human reset-adjacent LCS around 0.827 mean / 0.900 median. These are descriptive external references, not Symthaea measurements or targets. Source: [WorldTest presentation](https://datvo06.github.io/assets/talks/worldtest/).

## Benchmark arms and measurements

Use matched task/action access and interaction, runtime, memory and compute budgets:
1. MARA random agent.
2. MARA LLM ReAct agent with frozen prompt and model.
3. AutumnSynth/simple-world-model reference, with program visibility explicitly declared.
4. Direct history-only/reactive Symthaea policy.
5. Matched recurrent baseline (GRU/RSSM).
6. HDC-only, CfC/LTC-only and HDC+CfC/LTC.
7. HDC+CfC/LTC with event/individual relational memory.
8. Reset-disabled, fixed-reset, random-reset and decision-value reset policies.
9. Evaluator-only environment/answer oracle as an upper bound, never as a candidate baseline.

Report separately:
- Masked-frame exact accuracy and proper score where probabilities exist.
- Change-detection delay, false alarms, misses and post-change quality.
- Planning success, action count, path cost, constraint violations and failures.
- Query performance versus exploration interactions; reset count/share, replay LCS, observation coverage, action diversity and no-op rate.
- Belief revision timing and predictive score before/after contradiction, with no-change false-revision controls.
- p50/p95 latency, model queries, memory and measured energy.

An agent that resets frequently but does not improve held-out query results must not gain a better score. Do not optimize entropy reduction or reset count as a proxy for task success.

## Negative controls

- Reject reward leakage into exploration.
- Reject equal exploration/test program digests.
- Reject candidate-visible answer keys or true change times.
- Reject forecasts marked as acquired observations.
- Count reset events in the resource denominator.
- Include no-change tests and true rule-change tests.
- Keep public development data separate from withheld qualification tasks.

## Proposed graduation gates

- Reproduce official query-family scoring and a pinned public baseline on the public subset, where task/runtime/seed details can be recovered.
- Show positive held-out query improvement against the strongest matched observation-only baseline, with paired uncertainty intervals at the environment/task-family unit.
- Publish exploration-budget curves and net benefit of adaptive probing/reset versus matched fixed/random controls.
- Report each query family separately; do not pool three different capabilities into one unqualified score.
- Any human comparison must match the task release, instructions and budget and remain separate from the original 517-person study.

These are targets, not results.

## Execution order

1. Qualify the metadata lock, custody schema and Rust trace tests.
2. Recover and hash the public file manifest and each program/prompt/answer file; verify file-level license/attribution before import.
3. Implement a Rust adapter against pinned MARA protobuf definitions and a pinned Autumn.cpp process boundary.
4. Run the public 20/60 subset and verify task counts/results against MARA baselines; keep 43/129 paper coverage distinct.
5. Add Symthaea arms and matched reset/active-experiment ablations.
6. Keep withheld tasks under independent evaluator custody.

## Claim ceiling

This increment adds metadata, phase custody and a reset/replay trace diagnostic. It does not reproduce AutumnBench results, establish a general world model, demonstrate human-level performance, prove causal understanding or establish SOTA.