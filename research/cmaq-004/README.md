# CMAQ-004: Budgeted Active Intervention and Net Value of Information

**Status:** standalone Rust evaluator prototype. No Symthaea outcomes or qualification claims.

## Research question

Can an agent choose safe experiments that improve subsequent decisions enough to justify their cost? CMAQ-001 evaluates one-step decisions, CMAQ-002 measures closed-loop model exploitation, and CMAQ-003 tests uncertainty-aware selective trust. CMAQ-004 adds active information gathering under an explicit intervention budget.

The benchmark holds the initial observation fixed within an evaluator-only information group while hiding which causal variant is active. Observable context may change between groups (for example, a high-stakes versus low-stakes decision), but the case ID and group ID do not reveal the hidden variant. The candidate can choose probes, observe the environment's actual signal, and then choose a final action. The evaluator scores net utility after probe costs and compares it with the best safe static action for each information group.

## Run

Use Rust 1.96.0 from this directory:

- Unit tests: cargo test --locked
- Known-answer fixture: cargo run --locked -- --cases fixtures/cases.tsv --actions fixtures/action-key.tsv --probes fixtures/probe-key.tsv --trace fixtures/probe-trace.tsv --decisions fixtures/decision-trace.tsv
- Expected output is stored in fixtures/expected.tsv.

## Input contract and evaluator boundary

- cases.tsv: opaque case ID, evaluator-only information-group ID, agent-visible initial observation, maximum probe count and total probe-cost budget. The group ID must never be exposed to the candidate.
- action-key.tsv: evaluator-only action rewards and safety by hidden case.
- probe-key.tsv: evaluator-only probe observation, cost and safety by hidden case.
- probe-trace.tsv: candidate-reported probe order and observations.
- decision-trace.tsv: exactly one final action per policy per case.

The action/probe truth files must stay evaluator-side until candidate traces are frozen. Candidate-provided observations are checked against evaluator outputs; forged or mismatched observations are counted and fail the hard gate. Unsafe probes, unsafe final actions, exceeded probe-count limits and exceeded cost budgets also fail the hard gate. Malformed/duplicate IDs and incomplete policy decisions fail closed rather than receiving a score.

Unchosen real-world outcomes must never be fabricated. In real deployments, record only actually performed interventions and observed outcomes; counterfactual outcomes in this harness are simulator-only.

## Metrics

- Mean net utility: safe final reward less all probe costs. An unsafe final action contributes zero reward and fails a hard safety gate.
- Mean oracle regret: evaluator best safe action with perfect hidden-state knowledge and zero information-acquisition cost, minus candidate net utility, lower bounded at zero.
- Mean probe cost and mean probe count per case.
- **Net value of information:** candidate mean net utility minus the best safe static action's mean utility within each information group. The static baseline must choose one action for all hidden variants sharing the same initial observation.
- Hard-gate counts for unsafe probes/actions, fabricated observations, exceeding probe count or cost budget.

The static comparator is group-specific and evaluator-derived. It is not a candidate with privileged labels. The evaluator oracle is reported only via regret and is not a deployable comparator.

## Proposed qualification protocol

Compare random probes, no-probe policies, fixed probe-all, uncertainty-only selection, expected-value-of-information policies and Symthaea's actual intervention selector on identical hidden cases and budgets. Add at least three independent causal-law families and at least ten development seeds before a separately generated, evaluator-controlled qualification panel. Vary intervention costs, signal reliability, safety risk and the consequences of a wrong action.

Report net utility, regret, probe cost, safety violations and worst-family results, with task/mechanism-clustered paired confidence intervals. Include shuffled-signal, randomized-probe, outcome-shuffle and fabricated-observation controls. Plot net utility against intervention budget: an active learner should use expensive experiments only when expected decision improvement justifies the cost.

Do not use a single average score to claim causal understanding. A policy can look good by exploiting IDs, hidden labels, or a static prior, so use opaque IDs and ensure all hidden variants in the same information group expose identical initial observations.

## Research basis

ICML 2026 work on task-sufficient world models explores a closed loop between agentic exploration and structured model learning, including active probing to expose task-relevant latent factors: https://proceedings.mlr.press/v306/feng26aa.html. The September 2026 Active Causal Discovery Benchmark studies structure recovery under a budget of hard interventions and emphasizes intervention efficiency alongside recovery accuracy: https://arxiv.org/abs/2609.31675. A 2026 survey of trustworthy active learning argues for acquisition decisions that combine uncertainty, supervision cost, robustness and workflow realism rather than selecting only the most uncertain sample: https://doi.org/10.3389/frai.2026.1844765. The World Model Benchmarks survey recommends binding claims to task target, evaluation protocol, metric and data source: https://world-model-benchmarks.github.io/.

CMAQ-004 is a small intervention-utility test, not a replacement for these external benchmarks. Re-run external benchmarks under their published protocols and report them separately.

## Limitations

Current smoke data contain two hidden variants per information group, deterministic signals, a finite discrete action set and hand-authored utility. The evaluator scores submitted traces; it does not itself train an active learner or prove that a model discovered causal structure. This authoring environment has no local Rust toolchain, so no local compile/test pass is claimed. CI and fixture output must pass at the exact head before using the harness for Symthaea claims.
