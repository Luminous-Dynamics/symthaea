# IG-HIDDEN-INFO-001 — collective information discovery under latent information asymmetry

**Status:** v0.1 structural fixture prototype; no LLM-agent campaign has been executed.  
**Owning issue:** [#12762](https://github.com/Luminous-Dynamics/symthaea/issues/12762)  
**Branch:** `research/ig-hidden-info-001-v0-1`

## Why this benchmark exists

[HiddenBench (ICML 2026)](https://proceedings.mlr.press/v306/li26ej.html) isolates collective reasoning when decisive facts are privately distributed across agents. Its published abstract reports 30.1% multi-agent accuracy under distributed information versus 80.7% for a single agent given complete information, across its 65-task/15-model evaluation.

The benchmark's central failure mode is not simply disagreement or persuasion. It is **failure to discover what another agent knows but has not yet said**.

This v0.1 slice begins with a deliberately narrow, deterministic evidence-flow contract. It validates whether a task's decisive evidence can be transferred to a designated decision agent, whether delivery provenance is valid, whether the final answer cites evidence available to that agent, and whether missing evidence remains incomplete rather than being silently treated as success.

## Included

- `IG_HIDDEN_INFO_PROTOCOL_V0_1.md`: experimental protocol, factors, baselines, metrics and claim ceiling.
- `HIDDEN_INFO_FIXTURES_V0_1.json`: evaluator-only synthetic evidence-transfer fixtures.
- `qualifier/`: standalone Rust structural qualifier and unit tests.

The fixtures are synthetic structural controls. They are not a reproduction of HiddenBench, not LLM-agent trials, and not performance evidence.

## Published result files pinned

The original experiment-result release is separately catalogued in [PUBLIC_RESULT_SOURCE_MANIFEST_V0_1.json](external/hiddenbench/PUBLIC_RESULT_SOURCE_MANIFEST_V0_1.json). It identifies the three GPT-4.1 files needed for the original complete comparison: 80 hidden runs over eight manual/adapted tasks, 570 hidden runs over 57 generated tasks, and 650 full-profile runs over all 65 tasks. The publisher states these JSON result payloads are copied without modification. The result payloads themselves have not yet been downloaded or independently re-scored in this branch; the manifest is a source locator, not a reproduced score.

The Rust scorer accepts both hidden files in one invocation and validates that their model/profile metadata agree. The full-profile file is optional and supplied with `--full`.

## Official corpus pinned

The exact upstream `data/benchmark.json` is vendored under `external/hiddenbench/` with its MIT license and source manifest:
- upstream: `Yassellee/HiddenBench_ICML`
- source commit: `3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1`
- source Git blob SHA: `918366d953ecfe7e86a0e8cf7906cf757f574012`
- 65 tasks: **58 four-agent tasks and 7 three-agent tasks**.

The Rust `validate_hiddenbench` binary validates task identities, answer-option membership, non-empty fields, and this pinned group-size distribution. The copied benchmark file's Git blob SHA matches the upstream SHA exactly. The license is preserved in the adjacent `LICENSE`.

The answer key and hidden-fact allocation are evaluator-only. Do not pass the corpus JSON, `correct_answer`, or source manifest to agents.

## Rust validation and scoring commands

```bash
# Validate the exact upstream corpus and its 65-task / 3- and 4-agent profile
cargo run --manifest-path research/ig-hidden-info-001/qualifier/Cargo.toml --bin validate_hiddenbench -- \
  research/ig-hidden-info-001/external/hiddenbench/benchmark.json

# Reproduce the official average-vote and strict-majority score fields
cargo run --manifest-path research/ig-hidden-info-001/qualifier/Cargo.toml --bin score_hiddenbench -- \
  research/ig-hidden-info-001/external/hiddenbench/benchmark.json \
  path/to/hidden-manual-adapted-results.json \
  path/to/hidden-generated-results.json

# Add the full-profile pre-discussion comparison
cargo run --manifest-path research/ig-hidden-info-001/qualifier/Cargo.toml --bin score_hiddenbench -- \
  research/ig-hidden-info-001/external/hiddenbench/benchmark.json \
  path/to/hidden-manual-adapted-results.json \
  path/to/hidden-generated-results.json \
  --full path/to/full-profile-results.json
```

The scorer supports the upstream result shapes (`metadata -> runs` and legacy `conditions -> runs`). It reports the official per-run average and strict-majority accuracy fields, then adds task-clustered paired bootstrap intervals (10,000 resamples, pinned seed). It distinguishes the official run-weighted Hidden-post minus Full-pre gap from the task-paired gap. A small evaluator-only smoke result is checked in at `fixtures/HIDDENBENCH_SCORE_SMOKE_V0_1.json`; it is not model evidence.
 Agent prompts should be constructed from only the assigned task description, shared facts and that agent's assigned private fact.

## Official protocol details to reproduce

The upstream reference simulator uses a 15-round discussion by default, randomized order of facts and private fact allocation under a recorded seed, an initial private vote, sequential/rotating agent turns, and a final vote prompted with the last round's transcript. The standard 65-task corpus does **not** have a single uniform group size: group size is determined by the number of hidden facts (3 or 4). The local reproduction adapter must preserve this rather than silently forcing a fixed team size.

Official scoring reports:
- individual-vote average accuracy before and after discussion;
- strict-majority accuracy (strictly more than half correct);
- post-minus-pre improvement;
- Hidden-profile post accuracy minus Full-profile pre accuracy.

Our planned scorer will preserve these official fields and additionally report task-clustered uncertainty. Duplicated runs on the same task are not independent task samples.

 The `final.answer_correct` labels and expected metrics are evaluator-only fixture annotations; they must never be sent to an agent.

## Run locally

From the repository root:

```bash
cargo fmt --manifest-path research/ig-hidden-info-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/ig-hidden-info-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-hidden-info-001/qualifier/Cargo.toml -- \
  research/ig-hidden-info-001/HIDDEN_INFO_FIXTURES_V0_1.json
```

The qualifier uses exact evidence IDs and ordered delivery events. It checks that a sender possessed an evidence atom before sending it, that delivery endpoints exist, that sequence numbers are strictly increasing, and that the designated decision agent only cites evidence it could actually access. Repeated delivery of the same atom cannot inflate decisive-evidence coverage.

## What it does not yet do

- It does not call any model provider or execute agent conversations.
- It does not infer evidence atoms from natural-language messages.
- It does not judge whether a cited fact logically entails an answer.
- It does not estimate causal effects, benchmark accuracy, latency, cost, or statistical significance.
- It does not replace the #3095 experiment runner, #11337 CommunicationGraph, #11350 revision ledger, #11367 confidence evaluator, #11383 deliberation controller, or #11683 phantom-consensus qualifier.

The next tranche should add a license-aware HiddenBench adapter, reproduce its structured-communication baseline, capture actual provider/model receipts, and run matched-budget comparisons. Every published-model result must be backed by exact prompts, model revisions, task splits, event logs, scoring code and uncertainty intervals.

## Industry and SOTA comparison

See [SOTA_COMPARISON_V0_1.md](SOTA_COMPARISON_V0_1.md) for the direct HiddenBench/CSCW comparators, Free-MAD and Meta-Moderator decision layers, MAST and MultiAgentBench reliability/transfer baselines, OpenAI Agents SDK / Microsoft Agent Framework / Google ADK industry patterns, current model-backbone scope, matched experimental arms and proposed decision gates.

## Existing work to compose

- [#11320 / #11323](https://github.com/Luminous-Dynamics/symthaea/issues/11320): private/public stance transitions and exposure.
- [#11337 / #11360](https://github.com/Luminous-Dynamics/symthaea/issues/11337): communication channels, topology and delivery identity.
- [#11340 / #11350](https://github.com/Luminous-Dynamics/symthaea/issues/11340): evidence-carrying revision provenance.
- [#11367](https://github.com/Luminous-Dynamics/symthaea/pull/11367): confidence calibration.
- [#11383](https://github.com/Luminous-Dynamics/symthaea/issues/11383): evidence-aware deliberation control and stopping.
- [#11682 / #11683](https://github.com/Luminous-Dynamics/symthaea/issues/11682): phantom consensus.
- [#5471 / #5478](https://github.com/Luminous-Dynamics/symthaea/issues/5471): semantic-progress stopping and causal metareasoning ablations.

Keep this experiment separate from delegation/mandate graphs (#11248): a mandate edge is not a message channel.

## Claim ceiling

A passing local qualifier establishes only that the checked synthetic evidence-flow fixtures obey their declared structural invariants. It does not establish collective intelligence, model robustness, human-group validity, consciousness, or SOTA performance.