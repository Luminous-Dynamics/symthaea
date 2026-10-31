# PITB execution and research frontier update — 2026-10-31

**Scope:** changes on draft PR #8154, branch research/proof-insight-transfer-v0. Synthetic contract validation is reported separately from Lean compilation, real model outcomes, and CI.

## Executed implementation

### Support antichain analysis

The exact-source analyzer is scripts/research/pitb_support_antichain_v0.py. It now:
- receives the configured exhaustive variable cap through the full validator path (default 12, configurable; hard max 16);
- recovers all inclusion-minimal successful subsets, preserving alternative routes rather than collapsing them into one support set;
- derives declarations common to every observed minimal route versus declarations present in at least one route;
- rejects incomplete subset lattices, mixed target/environment/toolchain/fixed-context/budget/policy rows, repeated context digests, unverified success labels, and absent full-context success;
- scores support predictions and proposed sufficient routes, requires one model ID/version per ledger and complete target coverage, and uses an explicitly named route-sufficiency probability for calibration.

Exact-source local checks on 31 October:
- py_compile passed;
- --self-test passed for alternative and conjunctive routes, prediction scoring, incomplete lattice rejection, budget identity, and kernel/semantic-gate enforcement;
- CLI smoke test passed on two exhaustive synthetic lattices (16 subsets total), recovering [{h}, {hEq}] and [{h1,h2}], scoring both route predictions, and rejecting a configured variable cap that was too low.

This is synthetic analyzer evidence only. The source's referenced state/checker/semantic receipt digests are syntax-checked and bound to rows, but the analyzer does not open/authenticate their bytes.

### Typed proof-repair trajectory

The operation schema is data/benchmarks/proof-insight-transfer-v0/typed-proof-repair-operation.schema.json. The validator is scripts/research/pitb_typed_repair_trajectory_v0.py.

It validates target statement digest invariance, immutable target node membership, operation-node existence, pre/post node inventories, affected-node scope, per-episode sequence numbering, unique operation IDs/receipts, continuous blueprint digest/node-state chains, and no state change for rejected or abstained actions. Applied edits must be marked Lean-passed and must change the blueprint digest.

A repair trajectory integrity report is intentionally not a theorem-success report. Receipt bytes still need independent authentication; actual edit semantics and downstream correctness remain the responsibility of Lean plus successor/invariant tests.

Exact-source local checks on 31 October:
- py_compile passed;
- --self-test passed with six negative controls: changed target digest, mutation after rejection, target listed as affected, broken digest chain, applied operation without Lean success, and unknown operation node;
- CLI smoke test passed on a synthetic rejected-then-applied trajectory.

### JSON Schema and workflow checks

The typed-repair operation schema passed local Draft 2020-12 meta-schema validation. Synthetic applied and rejected operations were accepted; controls for missing state inventory, applied-without-Lean-pass, rejected-with-affected-nodes, and abstention-with-Lean-run were rejected.

The PR branch includes:
- .github/workflows/pitb-typed-proof-repair.yml — focused typed-repair schema and trajectory checks;
- .github/workflows/proof-insight-transfer-v0.yml — support-antichain and typed-repair trajectory self-tests integrated into the primary workflow.

Local checks do not establish Actions results. At the latest exact-head check, GitHub reported no PR-triggered workflow runs and no commit status entries for the PR head.

## Research frontier refresh

| Frontier | Reported result | PITB response |
|---|---|---|
| Goedel-Prover-V2, ICLR 2026 | 32B reports 90.4% MiniF2F pass@32 with self-correction | Maintain a distinct proof-generation comparison; this is not a support/repair result |
| FormalMATH, EMNLP 2026 Findings | 5,560 semantically validated Lean statements; DeepSeek-Prover-V2-671B reports 28.31% pass@32 | Use as a contextual research-level formal reasoning reference; do not compare percentages across corpora |
| LeanGraphSearch, Oct 2026 | MathlibQR Recall@100 94.7% versus 82.7% for LeanSearch v2; MathlibMPR Covered@20 improves 3.7 pp | Make multi-hop dependency retrieval a serious pinned baseline |
| T², ACL 2026 Industry | 2,206 tasks; best reported Testing Accuracy 38.9% | Keep successor/integration semantics separate from kernel compilation |
| BlueprintRepair / BlueprintTrace, MathNLP 2026 | 142 controlled failures; typed edits solve roughly similar counts but are 1.30x cheaper than exact patches and 2.06x cheaper than full rewrites per solved state | Add a separate target-preserving repair endpoint |
| VeriSoftBench | 500 Lean tasks from 23 projects; reports 41% success with curated context versus 35% from raw repositories | Add a repository-scale, cross-project validation cohort |
| Keep the Proof State Live, 2026 | Reports 5.6–50x wall-time speedups on its 48-problem setup | Continue a separate live-state branching systems track; current PITB JSON snapshots are not live states |

Sources:
- [Goedel-Prover-V2, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html)
- [FormalMATH, EMNLP 2026 Findings](https://aclanthology.org/2026.findings-emnlp.579/)
- [LeanGraphSearch paper](https://arxiv.org/abs/2610.22339) and [results](https://pluslabnlp.github.io/LeanGraphSearch/)
- [T², ACL 2026 Industry](https://aclanthology.org/2026.acl-industry.150/)
- [BlueprintRepair, MathNLP 2026](https://aclanthology.org/2026.mathnlp-main.3/)
- [VeriSoftBench](https://utopia-group.github.io/VeriSoftBench/)
- [Keep the Proof State Live](https://arxiv.org/abs/2605.25556)

## Paradigm shifts and qualification strategy

PITB should keep five independent endpoints:

1. **Proving:** exact-target Lean proof success, with statement correspondence and semantic/successor gates.
2. **Transfer:** paired increment in verified held-out target success from typed insights over the strongest validation-selected retrieval baseline.
3. **Support explanation:** prediction of finite, policy- and budget-specific minimal-support antichains, including alternative routes.
4. **Repair:** minimal typed blueprint edits that preserve the exact target and unaffected sibling obligations.
5. **Systems efficiency:** live elaborated-state reuse, branch startup, verified branches per second and success-cost frontier.

Do not pool endpoint scores. Higher proof pass@k on MiniF2F does not establish better support prediction, semantic transfer or repair. A SOTA claim requires matched targets and budgets, identical checkers, independent task semantics, uncertainty intervals, mutation controls and independent replication.

### Proposed new qualification goals

- Support: necessary-declaration F1 >= 0.90; sufficient-route accuracy >= 0.95 at >= 0.80 prediction coverage; exact minimal-route recovery >= 0.80.
- Counterfactual decisions: zero registered invalid move applications after required support removal; >= 0.95 stability for reviewed benign transformations.
- Repair: 100% target identity preservation; 100% Lean validation of applied operations; invalid-edit rejection recall >= 0.99; false rejection <= 0.05; zero unaffected-sibling regressions in the finite adversarial suite.
- Repair efficiency: at least 20% lower cost per verified repair than the strongest non-typed baseline while remaining within 2 percentage points of its verified repair rate.
- Transfer: >= +10 percentage points paired exact-target semantic-gate success versus the strongest validation-selected baseline at matched budgets, with a positive component-clustered uncertainty interval.
- Systems: >= 5x median branch-start improvement and >= 2x verified branch attempts at equal wall time, with zero accepted stale-state identity mismatches.

All are proposed targets, not achieved results.

## Remaining scientific blockers

- The 372-family relationship inventory and independently governed coverage trust root remain incomplete; the 56-family source test split is not yet qualified as an independent mathematical holdout.
- No real ablation lattice, support prediction, typed repair episode, or model transfer outcome has been recorded.
- The Lean proof-term metaprogram and fixed-target pilot have not been compiled in this session.
- Local Python/schema controls are not GitHub Actions passes. Exact-head Actions runs/statuses must be observed before CI claims.
- No measured transfer gain or SOTA win is claimed.

## Next work sequence

1. Run the eight-state Lean calibration for {h, hEq, hIrrelevant}, with exact fixed goal/environment/toolchain and independent kernel receipts.
2. Extend to 2–6 independently reviewed theorem components with disjunctive and conjunctive support, near-miss and distractor mutations.
3. Build a first target-preserving repair corpus from reviewed training families; include accepted, rejected and abstained trajectories.
4. Add the full receipt-byte authentication and semantic-successor harness for repair operations.
5. Complete the real cross-family relationship audit before any confirmatory held-out model experiment.
6. Compare typed insight, graph retrieval, support-antichain prediction, typed repair, patching and rewriting under identical budgets, then replicate on a temporally later or independent repository cohort.
