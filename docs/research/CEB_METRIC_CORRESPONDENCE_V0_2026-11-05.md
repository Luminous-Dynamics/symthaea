# CEB Target-to-Comparator Metric Correspondence V0 — 2026-11-05

**Status:** Design-only; no matched comparison or CEB performance claim is established.  
**Machine-readable crosswalk:** [`benchmarks/ceb/metric-correspondence.v0.json`](../../benchmarks/ceb/metric-correspondence.v0.json)  
**Schema:** [`benchmarks/ceb/metric-correspondence.schema.json`](../../benchmarks/ceb/metric-correspondence.schema.json)  
**Validator:** [`scripts/validate_ceb_metric_correspondence.py`](../../scripts/validate_ceb_metric_correspondence.py)

## Why this layer exists

The external-observation registry binds a reported number to a benchmark split, a metric, a model/harness configuration and its source. It does not, by itself, prove that an external metric has the same operational meaning as a CEB target. A number can be well sourced and still be the wrong comparator.

This crosswalk gives every CEB primary metric exactly one row and one explicit correspondence class:

- **exact_semantics** — the metric's operational meaning aligns closely enough to use as a matched-test candidate.
- **close_semantics** — same broad construct, but specific definition differences must be normalized.
- **related_but_distinct** — useful context, not the same metric.
- **not_comparable** — no defensible numeric counterpart currently registered.

The crosswalk deliberately allows **zero direct superiority claims**. Even an exact-semantics match does not mean the benchmark tasks, split, harness, model, budgets, environment, adjudication or regrade are matched. Those conditions must be established by a separate preregistration.

## Important examples

| CEB metric | External observation | Classification | Why |
|---|---|---|---|
| `CEB-CONTEXT/recall_at_10` | ContextBench Recall | Related, not exact | CEB's top-10 relevance unit and external aggregate recall are not yet shown equivalent |
| `CEB-DEBUG/bug_recall_at_k` | PDB bug-level recall | Related, not exact | PDB does not supply CEB's top-five finding ranking |
| `CEB-DEBUG/edit_precision` | PDB edit precision | Close | PDB uses epsilon-relaxed edit matching; tolerance must be matched |
| `CEB-REVIEW/defect_recall_at_5` | CRBugBench top-five-comment recall | Close | Similar top-five finding construct, different output/comment semantics and task sets |
| `CEB-SECURE/joint_secure_functional_pass` | Agent Security League SecPass | Close | Security pass requires functional and security checks, but tasks/language/rubric differ |
| `CEB-HUMAN/paired_act_abstain_accuracy` | AgentAbstain paired accuracy | Exact metric semantics | Both require both near-identical should-act/should-abstain variants to be correct; task/harness matching remains outstanding |
| `CEB-RUST/rust_issue_resolution` | RUSTFORGER resolution | Close | Historical model/harness and task version, not a current matched frontier result |
| `CEB-OPT/expert_normalized_speedup` | SWE-fficiency expert speedup | Close | Same general expert-relative construct, different workloads/hardware/expert baseline |
| `CEB-EVAL/false_accept_rate` | Epoch Terminal-Bench audit | Methodology only | Defect-bearing task count is not a controlled false-accept rate |

Metrics with no legitimate external counterpart are explicitly marked not_comparable. This includes, for example, cost per accepted repair, fabricated compiler authority, regression-free repair, critical evaluator false accepts, within-environment performance variance, escalation precision and documentation false verification.

## Target intent is not a result

Every mapping also labels the CEB target's intended role: matched-comparison goal, pilot guardrail, sealed-holdout goal, hard gate, efficiency goal, quality goal, measurement-quality goal, complementarity goal or coverage goal. All thresholds remain prospective. They are not evidence of a measured capability and should be recalibrated from pilot variance before confirmatory use.

## Negative controls

The validator must reject:
- CEB Recall@10 mapped to ContextBench Context F1.
- CEB bug Recall@5 promoted to exact semantics against PDB's aggregate bug recall.
- CEB joint secure-functional pass mapped to FuncPass instead of SecPass.
- AgentAbstain paired accuracy downgraded or altered without versioning its exact paired semantics.
- A missing/duplicated CEB metric row, wrong-lane observation, stale metric definition or unknown observation.
- Any direct `beats_sota` / direct superiority flag in this V0 crosswalk.

## Claim protocol

For a formal superiority claim, a future result must separately establish:
1. Same benchmark revision, exact task split, task instructions and source snapshot.
2. Same metric definition, denominator, aggregation and task-level adjudication.
3. Matched model, harness, prompt, tools, turns, tokens, compute, wall time and cost, or a preregistered single-variable ablation.
4. A preregistered primary endpoint, stop rule and sample-size plan.
5. Paired uncertainty with repository/root-cause clustering.
6. Independent pristine regrade with exact evidence digests.
7. No critical gate violation.

The CEB execution/trust requirements remain unchanged: no candidate-controlled execution before #10064 qualifies isolation; no independent regrade claim before #10369 qualifies signer custody and infrastructure/failure-domain independence.


## HiL-Bench selective escalation

The CEB-HUMAN lane now includes three exact-semantic replication candidates against HiL-Bench: Blocker Recall, ASK-F1 and help-enabled Pass@3. These metrics retain separate denominators and meanings. Question Precision has no standalone published scalar observation in the current source snapshot, so it is explicitly `not_comparable`; the 44% ASK-F1 aggregate cannot be used to infer its value. The 24% Pass@3 result is not an ASK-F1 score. The SWE (150 tasks) and SQL (150 tasks) blocker-recall observations are not combined as a single system result.
