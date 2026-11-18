# IG-COORD-SELECT-001 — when and how to coordinate

**Owner:** [#12786](https://github.com/Luminous-Dynamics/symthaea/issues/12786)  
**Status:** deterministic synthetic selector prototype; no model campaign executed.

## Research thesis

Collaboration is an intervention with both benefits and costs, not a universal default. The system should choose a single agent, a team architecture, or a calibration experiment based on evidence about the task and the model's capability on that task family.

The key external anchor is Kim et al., *Capable language models can outgrow the benefits of collaboration*, Nature Machine Intelligence (24 July 2026): https://www.nature.com/articles/s42256-026-01268-y

The study compares 260 configurations over six benchmarks, five architectures and three model families. It reports that stronger single-agent performance often predicts less benefit from coordination, while decomposability, task dependencies and error verification change the outcome. Its fitted ~45% threshold is explicitly empirical and not a universal law. This implementation intentionally contains **no hard-coded capability threshold**.

Other architecture treatments to reproduce or compare:
- AgendaFlow, Findings EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.322/
- Free-MAD, Findings ACL 2026: https://aclanthology.org/2026.findings-acl.1600/
- Meta-Moderator, Findings EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.1250/
- HiddenBench, ICML 2026: https://proceedings.mlr.press/v306/li26ej.html

## Included

- `COORD_SELECT_PROTOCOL_V0_1.md`: task factors, experimental design, metrics and claim boundaries.
- `COORD_SELECT_FIXTURES_V0_1.json`: 12 deterministic known-truth selector scenarios.
- JSON Schema for evaluator/selector separation.
- Standalone Rust robust-gain selector, tests and exact-head workflow.

## Decision semantics

The selector has three outcomes:
- `SELECT`: a team architecture has a conservative lower-bound improvement over the single-agent baseline after token, latency and risk penalties.
- `SINGLE`: no available team has a sufficiently strong and cost-adjusted lower-bound advantage.
- `CALIBRATE`: task/model estimates are out of distribution, under-sampled or too uncertain, and a team might plausibly help.

This is a conservative structural policy prototype. It does not infer estimates from model outputs, learn the architecture policy, or prove that any architecture is better. Calibration records must come from randomized task/model evaluation, not answer-key leakage.

## Run locally

```bash
cargo fmt --manifest-path research/ig-coord-select-001/Cargo.toml --check
cargo test --manifest-path research/ig-coord-select-001/Cargo.toml
cargo clippy --manifest-path research/ig-coord-select-001/Cargo.toml --all-targets -- -D warnings
cargo run --manifest-path research/ig-coord-select-001/Cargo.toml -- \
  research/ig-coord-select-001/COORD_SELECT_FIXTURES_V0_1.json
```

## Claim ceiling

A passing synthetic selector fixture only demonstrates that the rule implements the stated robust-gain/cost/calibration semantics. It does not establish that the rule predicts collaboration gains, selects an optimal architecture in production, beats any SOTA framework, or generalizes across models and task families.
