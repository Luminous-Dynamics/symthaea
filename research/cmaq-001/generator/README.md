# CMAQ causal factor-composition generator

This is an isolated, dependency-free Rust research generator. It creates paired **agent-visible cases** and **evaluator-only keys** for synthetic causal transfer tasks. The hidden mechanism is a sparse parity function over binary factors; each agent case contains randomized action-intervention observations from that case's mechanism, then asks for the correct action on an unseen context vector.

## Run

```bash
cargo test --locked
cargo run --quiet --locked -- --seed 4242 --factors 8 --radius 2 --out /tmp/cmaq-factor-pilot
```

Outputs:

- `agent/cases.jsonl`: observation, public event history, allowed actions, and budgets. It must be the only task file shown to a candidate.
- `evaluator/answer-key.jsonl`: hidden active factor mask, parity offset, target action, and outcome labels for evaluator use only.
- `evaluator/manifest.json`: generation parameters and SHA-256 digests of the exact agent/key files.
- stdout: compact machine-readable generation receipt.

`factors` is the number of binary context variables. `radius` is the number of active variables in the parity mechanism (the initial implementation selects exactly that many). There are `factors * 8` cases, each with `max(factors * 4, 32)` historical interventions and a current context absent from its own history.

The generator is not a model evaluation: it establishes a reproducible task contract and a known-answer control. The data are synthetic, the mechanism family is disclosed, and the current version does not simulate continuous regime changes, noisy outcomes, heterogeneous action costs, external tool execution, or real-world safety. The public generation seed and keys are development artifacts, not a blind qualification set. Do not report the generated cases as SOTA results.

For agent comparisons, hold all histories/budgets constant and compare stateless choice, majority/global-action heuristic, nearest-context retrieval, explicit history reasoning, and a mechanism-learning candidate. Split at the hidden mechanism/task level before tuning, and report action accuracy, counterfactual prediction calibration, regret, query cost, retained memory, and latency. A future scorer adapter should join the generated answer key to independently recorded action/outcome receipts rather than accept agent-reported outcome labels.
