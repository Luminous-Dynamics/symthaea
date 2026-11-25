# CMAQ-002 WorldTest / AutumnBench adapter v0.1

**Status:** metadata-only source lock, custody fixture and Rust reset/replay trace diagnostic. No data or simulator has been downloaded or run.  
**Owners:** #8908 (active causal learning), #9094 (external benchmark bridge), #9176 (shared frozen-run manifest).

## Motivation

WorldTest asks whether an agent learns a reusable environment model that can answer new environment-level queries, not merely predict the next observation or maximize reward on the same rollout distribution. Its two-phase protocol separates reward-free exploration from a query in a related but different challenge environment.

The ICML 2026 paper reports 43 environments, 129 tasks, 19 stochastic environments, 517 human participants and five frontier models. The paper-linked public Zenodo record is v1 (DOI 10.5281/zenodo.19498269) and lists a 71.1 kB archive, autumnbench_public.zip, with published MD5 f276390c6ea102187c072a0fafda9888. Its description says the archive contains prompts, ground-truth answers, environment programs and color_dict.yaml. The PMLR landing page does not display a license value. A secondary catalogue reports 20 environments / 60 tasks and CC-BY-4.0, but neither the counts nor license are verified from the archive/manifest. Do not use the inferred 23/69 withheld counts as a fact. Sources: [PMLR paper](https://proceedings.mlr.press/v306/warrier26a.html), [author project page](https://archanarw.github.io/projects/autumnbench.html), [WorldTest presentation](https://datvo06.github.io/assets/talks/worldtest/).

This complements CMAQ-002's finite EVoI simulator: EVoI measures which safe probe to perform inside a small known hypothesis catalog; WorldTest asks whether reward-free exploration creates a model that generalizes to new queries in a related environment.

## Source and data boundary

The PMLR paper links directly to the Zenodo v1 archive record. Separately, the pinned MARAProtocol download script (blob a0eb99dfd39b8cab9ffa5e577d2e2e1b738301e9) points to a Firebase Storage manifest_public.json endpoint. These are currently two independent distribution routes; we have not downloaded either manifest/archive or matched their bytes. The Firebase manifest SHA-256 and Zenodo archive SHA-256 remain null. A download attempt from this execution environment failed at DNS resolution.

The lock pins:
- Autumn.cpp at 0929d175c3d296de091ac4b998d09b9b55a38f67 (MIT; C++20 native interpreter, plus WASM/Python/Julia).
- MARAProtocol at d0d4e9251151778415cc5195c5dfb8296a5ded24 (MIT; protocol buffers pinned by file blob SHA).
- The public data manifest endpoint, without claiming immutable dataset identity or per-file checksums.

The two pinned commits are unsigned. The official Zenodo landing page currently shows no license value. A secondary catalogue reports CC-BY-4.0, but this remains unverified and must not be treated as permission to redistribute. The dataset is not downloaded; per-file license and attribution review are still required. The MARA reference runner requires Python 3.12 and Protobuf tooling; Symthaea should implement a thin Rust adapter to the pinned protobuf contract rather than importing the Python runner into production. Use the native Autumn interpreter as a pinned external process; do not vendor code or data.

## Protocol custody

Exploration:
- The pinned MARA controller forwards a generic agent-feedback reward field on every step. In the reference reward-free exploration environment, ordinary actions, explicit reset and the go-to-test phase-exit action return exactly zero. The field is therefore present but carries no task-reward signal. The adapter must verify every exploration reward is zero and fail closed on any nonzero value; do not describe the feedback field itself as absent.
- Candidate sees only the configured observation/action interface and actual observations.
- Reset is a first-class action and is charged to the interaction budget. The candidate-issued go-to-test phase-exit action is also charged as an exploration interaction before the terminal test-phase marker. Each action/reset record now carries the exact IEEE-754 reward bits; the MARA wrapper rejects nonzero, NaN and infinite reward values before evaluating the trace.
- Predicted future observations are never promoted to acquired observations.

Test:
- Use a related but different challenge program; the exploration and test program digests must differ.
- Query family is masked-frame prediction, change detection or planning.
- Task prompt may be visible as the task requires; answer key, true change timestep and evaluator scoring artifacts remain evaluator-only.
- Program-visible reference agents are a separate interface arm, not a matched comparison to observation-only agents.

The JSON Schema fixture validates this metadata/custody contract only. It does not establish actual dataset integrity or a successful simulator adapter. The Zenodo archive's MD5 is recorded as published metadata only; it has not been verified locally, and SHA-256 is unknown.

## Pinned MARA runner reward and phase audit

The pinned reference controller calls the agent Feedback RPC after every environment step and forwards StepResponse.reward into FeedbackRequest.reward. The interactive exploration implementation returns exactly zero for ordinary actions, explicit reset and go-to-test. This is reward-free by **constant-zero signal semantics**, not by omitting the feedback field. A qualified adapter must check the returned reward at every exploration step and reject nonzero values before forwarding feedback.

The transition action go-to-test is issued by the candidate and is a charged exploration interaction. The trusted adapter appends the non-interaction BeginTest marker only after the environment has accepted the phase transition. The qualified Rust entry point is evaluate_mara_worldtest_exploration in src/worldtest_phase_contract.rs; it requires exactly one go-to-test action immediately before BeginTest.

Pinned source identities at MARAProtocol commit d0d4e9251151778415cc5195c5dfb8296a5ded24:
- mara_agent.proto: 70ae75dad22be364af45417fb24007ddbfbe61a3 (reward field in FeedbackRequest)
- mara_environment_service.proto: 58069aa74cb1bf084155c115111dc2a2aa6b58c6 (StepResponse reward)
- evaluation_controller.py: 8650b314b3f0e5f9301340dce89aed5aa4de0ae8 (reward forwarded on each Feedback call)
- environment_interfaces.py: 7048358744b74f042db4c43f734d9271468f727b (interactive-to-test transition)
- concrete_envs.py: 961858e7f47ad1e572bc20936d27ae0b4fee000b (interactive action/reset/go-to-test returns zero)

The source lock and machine-readable audit are in worldtest-runner-audit.lock.json and worldtest-runner-audit.schema.json. This is a pinned source audit only; no MARA process or Autumn interpreter was run.

## Rust reset/replay diagnostic

The bounded module src/worldtest_exploration.rs audits resets and replay; the phase wrapper src/worldtest_phase_contract.rs additionally requires the charged go-to-test action followed immediately by a terminal BeginTest marker. Use evaluate_mara_worldtest_exploration as the qualified entry point. It reports:
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