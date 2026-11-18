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

The fixtures are synthetic structural controls. They are not a reproduction of HiddenBench, not LLM-agent trials, and not performance evidence. The `final.answer_correct` labels and expected metrics are evaluator-only fixture annotations; they must never be sent to an agent.

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