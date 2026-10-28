# Broca pipeline-evolution and promotion gate v1

**Status:** Rust contract and unit tests authored; no compile/run is claimed in this environment. This module does not itself generate or optimize a memory pipeline and does not establish a benchmark result.

## Why this increment exists

Recent work makes it clear that memory research is moving beyond static retrieval:

- [MemPro: Agentic Memory Systems as Evolvable Programs](https://aclanthology.org/2026.emnlp-main.1364/) treats the full memory construction–retrieval pipeline as an evolvable program with a version tree, rather than merely tuning a memory bank.
- [MSCE: From Memory to Skills](https://aclanthology.org/2026.findings-emnlp.1637/) promotes evidence-backed procedures into reusable skills with applicability bounds, verification rules and reliability estimates.
- [PerMemSafe](https://aclanthology.org/2026.findings-acl.320/) tests implicit personalized safety as user-specific context accumulates and changes over time.
- [MemProbe](https://aclanthology.org/2026.emnlp-main.2473/) diagnoses interference, misinformation, consolidation and reconsolidation rather than collapsing memory behavior into final-answer accuracy.
- [HELM](https://aclanthology.org/2026.emnlp-main.721/) emphasizes explicit provenance and epistemic state as a control on invalid memories.
- [SkillsVote](https://aclanthology.org/2026.emnlp-main.281/) studies skill collection, attribution and evidence-gated evolution.

Those papers report their own methods and results. They are not comparable to each other or to Broca without matched benchmark, reader model, evaluator, budgets and split.

**Research hypothesis:** the whole memory pipeline should be allowed to evolve, but an optimizer must never see the held-out scores used to decide whether the resulting system generalizes. Every version is immutable, every comparison is matched, and promotion is blocked by critical governance incidents.

## What the Rust module validates

`crates/domains/symthaea-broca/src/memory_pipeline_evolution.rs` defines:

- **Candidate lineage:** each candidate has a stable ID, parent candidate, increasing edit iteration, candidate-manifest digest, source/config/prompt digests and lifecycle state. Duplicate IDs, missing parents, self-parenting, cycles and malformed digests fail validation.
- **Optimizer visibility:** candidate generation can reference only completed development/calibration receipts. Held-out receipts may exist for audit but cannot appear in the optimizer-visible allowlist or in a candidate's build inputs.
- **Receipt binding:** an evaluation receipt names the exact candidate manifest, benchmark, dataset, case set, answer model, reader prompt, evaluator, harness and shared protocol.
- **Matched promotion:** candidate and baseline must use the same held-out case set, reader/model/prompt/evaluator/harness and protocol; candidate must be frozen before evaluation and both receipts must be completed.
- **Non-compensatory risk gate:** a candidate with any critical governance event cannot be promoted, no matter its task accuracy.
- **Two promotion routes:**
  1. **Quality win:** paired held-out task-success lift is at least +3 percentage points and the lower bound of its preregistered 95% grouped paired interval exceeds zero.
  2. **Efficiency win:** the lower endpoint of the paired 95% confidence interval is at least -1 percentage point and measured total lifecycle cost is at most 75% of baseline. Point-estimate non-inferiority alone is insufficient.
- **Reject otherwise:** a candidate that does not meet either route is not promoted. The policy thresholds are proposed internal defaults, not industry standards.

The module validates receipt structure and consistency and applies the reported paired confidence bound. It does not compute that confidence interval, verify that the runner is honest, or cryptographically attest to the model/evaluator. The external runner must produce the metric and artifact receipts; the top-level run verifier must independently hash/validate those artifacts.

## Run focused Rust tests

From the repository root when a Rust toolchain is available:

```sh
cargo test -p symthaea-broca --lib memory_pipeline_evolution
```

Ten unit tests are authored for: valid candidate tree; held-out leakage to optimizer; candidate built from held-out feedback; missing parent/cycle; quality promotion; cost/non-inferiority promotion; mismatched protocol/case set; governance incident blocking promotion; insufficient independent groups; and iteration/manifest-digest pinning.

No Rust compiler is available in the code-authoring environment, so these tests are authored, not passed. CI results are not counted unless an executed, current-head run is inspected.

## How to use this gate in a real pipeline-evolution experiment

1. Freeze baseline candidate and shared evaluation protocol, then calculate its candidate-manifest/source/config/prompt digests.
2. Permit the optimizer to inspect only development/calibration receipts and immutable parent candidate manifests. Store every candidate in the version tree—including failed candidates and edits that regress.
3. Freeze the selected candidate before any held-out evaluation. A changed source/config/prompt produces a new candidate ID/manifest digest and requires a new held-out receipt.
4. Run candidate and baseline with the same LongMemEval-V2 protocol and exact held-out question set. Preserve official LAFS scoring separately from paired task-success and cost metrics.
5. Call the promotion gate. A candidate with a critical stale-memory, unauthorized-scope or deletion-resurrection event is rejected even if answer accuracy increases.
6. Keep native benchmark tracks separate: LongMemEval-V2 for memory recall/latency, Mem2ActBench for tool/action grounding, TimlMemBench/FAMA for governed updates, PerMemSafe for implicit personalized safety, MemProbe for stability–plasticity, and LoCoMo/EvoAgentBench/other skill benchmarks for transfer. Never pool them into one synthetic “SOTA” number.

## Intended next stage

This is the control-plane contract around pipeline evolution, not an implementation of MemPro. Next stages should add a deterministic candidate manifest writer, a trusted receipt verifier, an evaluator that owns held-out data outside the optimizer process, and an execution harness that measures the full lifecycle cost. Only after the control plane passes negative tests should it be connected to a pipeline-generation agent.

The correct success criterion is not “the optimizer found a higher number.” It is a repeatable held-out quality or lifecycle-cost advantage, with the full lineage and all negative results available for independent review.
