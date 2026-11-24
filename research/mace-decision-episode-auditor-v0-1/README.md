# MACE Decision Episode Auditor v0.1

**Status:** standalone research evaluator prototype. It is not yet wired into the hosted AML service, the LMEV2 sidecar, or the independent evaluator boundary. The checked-in fixture is synthetic and is not a benchmark result.

## Why this exists

A retrieval score does not tell us whether retrieved memory improved a decision. A response-preparation receipt does not prove that a consumer used the evidence. A self-reported outcome does not establish causal influence.

This crate creates a strict paired-data contract for measuring the next portion of the chain:

`stored → retrieved → qualified → exposed → selected → decision → outcome`

It audits **decision episodes** as matched control/treatment pairs, preserving a fixed task, model, prompt, evaluator, environment, stochastic seed and token budgets within each pair. It then reports treatment-minus-control differences in success, utility, harm, unauthorized exposure/actions, tokens, latency and total cost.

The contract is intended to feed the decision-influence protocol in [#13397](https://github.com/Luminous-Dynamics/symthaea/issues/13397) and the matched baseline programme in [#13261](https://github.com/Luminous-Dynamics/symthaea/issues/13261). It does not replace either programme.

## What is implemented

- Strict Serde decoding with unknown-field rejection.
- Versioned JSON dataset schema and a synthetic example fixture.
- Four typed contrasts: memory vs no memory; clean vs poisoned; poisoned vs repaired; baseline vs candidate. Poisoning and repair contrasts hold the retrieval-policy digest fixed while changing the memory snapshot, isolating the memory-state intervention.
- Per-pair frozen task/model/prompt/evaluator/environment/seed/budget binding.
- Mandatory decision and outcome receipt digest references for each arm.
- Pair and memory-condition validation, unique pair IDs, unique selected evidence IDs, and strict SHA-256 syntax checks.
- Treatment-minus-control metrics for task success, utility, harm, unauthorized exposure/actions, input/output tokens, latency and all-in cost.
- Equal weighting by independent cluster, rather than allowing a large cluster to dominate the mean.
- Deterministic percentile cluster bootstrap intervals when at least five independent clusters are present. The report explicitly declines to estimate an interval below that threshold.
- Non-compensable observed-count gates for unauthorized evidence exposure and unauthorized actions.
- A CLI that emits a JSON report bound to the exact source dataset bytes by SHA-256.

## Run

From the repository root, with the pinned Rust toolchain installed:

```bash
cargo test --manifest-path research/mace-decision-episode-auditor-v0-1/Cargo.toml

cargo run --manifest-path research/mace-decision-episode-auditor-v0-1/Cargo.toml -- \
  research/mace-decision-episode-auditor-v0-1/data/fixtures/mace_decision_episode_dataset_v1.example.json \
  > decision-episode-report.json
```

The example contains five synthetic independent clusters. It is only a schema and arithmetic fixture. It must never be presented as an empirical MACE result.

## Pair design and estimand

Each pair shares these immutable fields across both arms: task digest, task family, model revision, prompt digest, evaluator revision, environment revision, stochastic seed, context budget and output budget. The arm-specific fields record the memory state, retrieval policy digest, memory snapshot digest, decision/outcome receipt references and observed resource/outcome metrics.

The report computes:

`delta = treatment - control`

The treatment arm is always the system variant under test. A positive difference is favorable only for metrics marked `higher_is_better`; a negative difference is favorable for `lower_is_better`.

Cluster-level means are resampled as units. The reported 95% interval is a percentile cluster-bootstrap interval, with the configured seed and replicate count. This is a descriptive uncertainty interval, not a causal test. The v0.1 evaluator does not implement stratified randomization checks, multiplicity correction, sequential testing or preregistered stopping-rule enforcement.

## What this does not prove

This boundary is deliberate:

1. A SHA-256 receipt field is a reference/commitment, not signature verification. The crate does not verify signatures, source issuers, consent, purpose, revocation, corrections, erasure or transitive provenance.
2. A valid decision receipt digest does not prove that a person or agent actually read, attended to, or used the returned evidence.
3. A valid outcome receipt digest does not prove that the referenced outcome was independently measured.
4. The input is not independently randomized or sealed. The auditor cannot establish evaluator custody, assignment integrity, or prevent a candidate from inspecting hidden labels.
5. Pair matching is structural: common task/model/prompt/evaluator/environment/seed/budgets are fields on the pair envelope. It does not certify that the referenced artifacts truly correspond to those digests.
6. A cluster bootstrap with five or more clusters is an initial descriptive interval, not a substitute for a powered study, family-aware sampling plan, or external replication.
7. Zero observed unauthorized events in a finite fixture is not proof of zero real-world risk.
8. A candidate that abstains on everything may pass a safety count gate; useful-evidence retention and task utility must be read beside safety gates.

**Do not claim causal influence, security qualification, benchmark superiority, or production readiness from this crate alone.** For those claims, require independently held evaluator truth, cryptographically verified decision and outcome receipts, randomized or otherwise defensible assignment, current-at-use rights qualification, transitive evidence lineage, and exact-head reproducibility.

## External benchmark portfolio

These tracks should remain separate. Scores from different benchmark protocols, answer models, judges or task splits must not be pooled.

| Track | What it measures | MACE use |
|---|---|---|
| Agent Memory Leaderboard Cycle 2 | Official Textual, Coding and Multimodal Add/Search performance | Reproduce the official protocol and report each track separately |
| LongMemEval / LongMemEval-V2 | Multi-session retrieval, reasoning, updates and abstention | Matched-backbone memory retrieval baseline |
| MemoryArena | Interdependent Memory-Agent-Environment tasks | Decision and action utility under multi-session dependencies |
| WorldMemArena | 400 multi-session multimodal tasks; write, maintain, retrieve and use | Visual/procedural evidence and lifecycle-stage diagnosis |
| AMA-Bench | Agentic trajectories across recall, causal inference, state updating and abstraction | Causal/state-first retrieval and tool-augmented evidence use |
| MemSecBench | Poison persistence, downstream execution and selective repair | Poisoned-vs-clean and repaired-vs-poisoned contrasts |
| MemoryDojo | Long-horizon poisoning propagation across 155 trajectories | Trigger-conditioned and multi-session adversarial influence |

Useful external references:

- Agent Memory Leaderboard Cycle 2 results (published 2026-11-19): https://agentmemoryleaderboard.ai/results/cycle-2/
- Redis LongMemEval reproduction report (2026-09-21): https://redis.io/resources/agent-memory-benchmark-report/
- MemoryArena, ICML 2026: https://proceedings.mlr.press/v306/he26am.html
- WorldMemArena paper: https://arxiv.org/abs/2605.29341
- AMA-Bench, ICML 2026: https://proceedings.mlr.press/v306/zhao26bs.html
- MemSecBench: https://arxiv.org/abs/2607.27080
- MemoryDojo: https://arxiv.org/abs/2611.09721

### Current public reference points (not MACE results)

The official Cycle 2 open-method leaders are Resurface Textual 61.22, InsightMem Coding 63.33% resolved, and Khipu Multimodal 43.26. MemoraX leads the commercial divisions at 69.26, 68.00% resolved and 53.68. A proposed next-cycle open-method stretch target is at least +5 absolute points over the current open leaders: 66.22 Textual, 68.33% Coding and 48.26 Multimodal. These are targets, not achieved results and not guaranteed to be directly attainable under a future cycle's protocol.

Redis reports 86.5% task-averaged accuracy on LongMemEval Small for its Remis + Instruct strategy using a common GPT-4o backbone in its September 2026 report. This is a vendor-authored reproduction on a distinct benchmark and should not be compared numerically with AML Cycle 2.

## Proposed MACE success criteria

Treat these as preregistered project targets, not industry standards:

- At matched model, task, context and compute budget: at least +5 percentage points in task utility or at least 20% lower decision regret versus the strongest reproduced baseline.
- At the chosen operating point: zero unauthorized exposure and zero unauthorized actions on deterministic conformance/adversarial fixtures.
- Repair: 100% invalidation closure on deterministic derived-evidence DAG fixtures, zero stale/retracted/erased item reuse after repair, and at least 99% benign-memory utility retention as a stretch target.
- Efficiency: demonstrate a Pareto gain in completed-task utility per token and per dollar, including write/consolidation/qualification/index costs; do not report query latency alone.
- Robustness: maintain the same conclusions on held-out task clusters and at least one independently operated evaluator run.
- Ablations: compare no memory, full history, BM25, dense, hybrid RRF, typed facts, timeline, event/causal graph, static HDC, HDC-LTC, and the complete MACE policy under matched budgets.

## Next qualification increment

1. Bind the hosted service's search receipts to decision receipts emitted by a separate consumer process.
2. Add a verified Mycelix rights/lifecycle qualification step before evidence exposure; the service's local `active` flag must never be treated as consent or authority.
3. Verify receipt signatures and source lineage under an independently provisioned trust profile.
4. Move held-out labels and score computation to a separate evaluator custody domain.
5. Add randomized assignment manifests and family-aware sampling/power analysis.
6. Reproduce external baselines under their native data, model, prompt and evaluator protocols, then run the paired MACE contrasts.

The next implementation should compose with [#13446](https://github.com/Luminous-Dynamics/symthaea/issues/13446), [#13455](https://github.com/Luminous-Dynamics/symthaea/issues/13455), [#13319](https://github.com/Luminous-Dynamics/symthaea/issues/13319), and [#13287](https://github.com/Luminous-Dynamics/symthaea/issues/13287); it must not create a parallel rights ledger or self-qualifying candidate backend.
