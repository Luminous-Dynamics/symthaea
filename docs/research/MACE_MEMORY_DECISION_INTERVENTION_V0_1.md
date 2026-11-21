# MACE memory-to-decision intervention protocol v0.1

**Status:** research protocol and record contract. This document reports no executable HDC-to-MACE adapter, compiled test, benchmark run, measured utility gain, official benchmark result, or SOTA claim.

**Parent work:** [#13127 — qualify HDC-LTC temporal retrieval as a MACE candidate arm](https://github.com/Luminous-Dynamics/symthaea/issues/13127), [draft PR #13128](https://github.com/Luminous-Dynamics/symthaea/pull/13128), [#9516 — causal experience transfer across dependent agent sessions](https://github.com/Luminous-Dynamics/symthaea/issues/9516), and [draft MACE implementation #12743](https://github.com/Luminous-Dynamics/symthaea/pull/12743).

**Owners reused:** #13099 (HDC-LTC retention), #9516/#12743 (MACE causal-transfer policy and evaluator), #13019 (slow-context refresh), #12846 (independent source/execution verification), #11816 (oracle independence), #8427 (all-in cost frontier), and #9139/#12731 (consolidation, privacy and reversibility).

## 1. Research shift: memory is an intervention

A memory benchmark should not stop when a candidate is retrieved. It should test what happens to a later decision when the agent is given, denied, corrected, or deliberately misled by that memory.

The primary question is:

> Under the same task, initial agent state, environment evidence and resource budget, does exposing valid retained evidence improve the decision—and can the evaluator show that the improvement came from that evidence rather than from the model, prompt, task order or an easier world?

This protocol adds a **memory-access intervention** to the existing MACE evaluation. It does not create a second memory store, causal evaluator, agent policy, refresh controller, or official benchmark adapter.

A retrieval event is not proof of memory use. A decision change is not proof of decision improvement. A decision improvement is not proof that the retrieved memory caused it. All three links need separately observable evidence.

## 2. Four causal questions, four estimands

For each held-out world-cluster/query block, clone the same initial candidate state and randomize the intervention arm. Let U(a) be the evaluator's net utility under arm a, including declared task rewards, failure penalties, probe costs and action costs.

1. **Does memory help?**
   `ΔU_memory = E[U(hdc_ltc) - U(no_memory)]`

2. **Is the information itself relevant, rather than the mere presence of extra context?**
   `ΔU_content = E[U(hdc_ltc) - U(semantic_placebo)]`

3. **Does the HDC-LTC retrieval substrate earn its complexity against conventional methods?**
   `ΔU_substrate = E[U(hdc_ltc) - U(best_matched_conventional_retrieval)]`

4. **Does lifecycle maintenance prevent obsolete knowledge from influencing current actions?**
   `ΔU_freshness = E[U(post_correction_snapshot) - U(pre_correction_snapshot)]`

These contrasts answer different questions. Do not combine them into one score or treat an HDC-vs-no-memory gain as evidence that HDC is better than BM25, dense retrieval, a key-value store, or an exact-scope scan.

The exact-scope scan is a candidate-visible evidence ceiling only when it uses the same candidate-visible records. An evaluator oracle that can see hidden labels remains evaluator-only.

## 3. Matched intervention arms

Run all feasible arms on the same world-cluster/query blocks:

| Arm | Candidate receives | Question answered |
|---|---|---|
| `no_memory` | No prior memory; only the current observation/context | Does any memory help? |
| `exact_scope_scan` | Exact-scope candidate-visible records | What is the retrieval ceiling? |
| `conventional_retrieval` | Best matched KV/episodic, BM25, dense or hybrid baseline | Does HDC-LTC beat practical retrieval? |
| `static_hdc` | Static HDC associations with the same encoder, dimension and fan-in | Does temporal evolution add value? |
| `hdc_ltc` | Actual HDC-LTC candidate retrieval | Does this proposed arm improve decisions? |
| `semantic_placebo` | A byte/token/count-matched set of valid but task-irrelevant, semantically confusable records from the same permitted candidate corpus | Is useful content responsible for the effect? |
| `pre_correction_snapshot` | A historically valid pre-correction memory snapshot while the current canonical registry knows the correction | Does stale retrieval get rejected? |
| `post_correction_snapshot` | The corrected current snapshot | Does correction propagation restore correct action? |

The placebo must be constructed from real, valid candidate-visible records. Prefer within-cluster distractors with mismatched objective, action, regime, valid-time interval or dependency chain. Do not create malformed identities and then count their rejection as evidence of semantic robustness. Cross-tenant and cross-user records are prohibited, not benchmark distractors.

The pre-correction arm is a lifecycle negative control, not a fair efficacy competitor. The current canonical resolver must still expose the correction/retraction state at the query's declared knowledge time. The candidate may retrieve the old event; MACE must reject its use when it is no longer admissible. Preserve both the old as-known snapshot and the current snapshot.

An oracle index may be retained as an evaluator-only diagnostic ceiling. It must never be candidate-visible or used to set thresholds on the held-out split.

## 4. Experimental procedure and leakage controls

1. Freeze the task corpus, candidate-visible interaction history, world-cluster split, model/policy revision, HDC profile, conventional retrievers, prompts, tools, action budget, memory budget, and all random seeds. Hash every manifest.
2. Partition by independent world cluster and hold out task families, source roots and mechanism/regime variants. Never split correlated episodes from the same world across train, calibration and final test.
3. Clone the same initial agent state for each arm. Randomize arm order within each pair block. Use common random numbers or deterministic replay where supported; record any residual nondeterminism.
4. Give every arm identical current observations, tool permissions, objective, time budget and model context budget. The candidate must not receive hidden evaluator labels, gold event IDs, arm-specific hints or the expected action.
5. Keep arm assignment and evaluator metadata outside the candidate prompt. Log the candidate-visible input digest so parity is auditable.
6. Allow only the assigned memory input to differ. Charge every method for its full index, metadata, codebook, evidence sidecar, retrieval, validation, writes, corrections, rebuilds and replay.
7. Run the candidate to a terminal decision: action A, action B, abstain, or request a bounded probe. Record the exact event IDs it retrieved and the subset it says it relied on. The evaluator independently resolves and scores those IDs; model-generated explanations are not evidence.
8. Evaluate the result against independent outcome receipts and hidden labels. Do not expose evaluator scores to the candidate or use them to adapt the same run.
9. For a sampled audit subset, replay from the same initial state with the decision-supporting evidence removed (evidence knockout) and with a matched placebo substituted. This estimates whether the cited memory was necessary for the decision change. It is a secondary causal diagnostic, not a substitute for randomized arm contrasts.
10. Preserve raw per-query rows, failures, invalid rows and all arm outputs. No post-hoc threshold tuning on the final split.

If the agent has online state that cannot be cloned, randomize at the world-cluster level and use disjoint replicas. Do not let one arm learn from another arm's actions or evaluator feedback.

## 5. Evidence-to-action trace

For each decision, preserve the following candidate-side chain:

`query → retrieved event IDs → canonical record resolution → lifecycle/scope/causal gates → decision-support event IDs → proposed action or abstention → dispatch/observation → reconciled outcome`

Each link has a separate receipt or digest. In particular:

- A retrieved ID is not automatically a decision-support ID.
- A source that resolves may still fail trust, estimand, objective, regime, time or retraction checks.
- A valid event may be irrelevant to the current decision.
- A model explanation that names an event does not prove the event changed the decision.
- The final action outcome is reconciled against the environment, not the agent's generated narrative.

For each arm, record at least: pair block, world cluster, query, arm, initial-state digest, candidate-input digest, memory-snapshot digest, policy/model revision, retrieved IDs, resolved IDs, decision-support IDs, final choice, abstention/probe reason, action receipt, outcome receipt, latency, token/tool use, index bytes and evidence-sidecar bytes.

The evaluator separately records hidden relevance, decision utility, regret, invalid/stale/wrong-scope evidence use, false causal claims and whether an observed decision change was justified by admissible evidence. Candidate records must not contain evaluator-only labels.

## 6. Metrics: report a vector, not a victory score

### Primary decision metrics

- Paired net-utility lift and regret reduction for each named contrast.
- Dependent-task and final-chain success.
- Decision agreement with exact-scope retrieval.
- Harmful memory dependence: fraction of paired cases where memory lowers utility relative to no memory.
- Justified influence rate: fraction of memory-induced decision changes supported by correctly resolved, admissible evidence.
- Placebo sensitivity: rate at which irrelevant memory changes the action or increases regret.
- Evidence-knockout effect on the audited subset.
- Worst-family and worst-regime utility, not only the mean.

### Calibration and safe non-use

- Selective risk versus decision coverage as abstention thresholds vary.
- Abstention precision/recall for insufficient, conflicting, stale and out-of-scope evidence.
- Probe-request rate, probe-resolution rate, probe cost and post-probe utility.
- False causal-claim rate and unsupported-action reuse.
- Calibration of retrieval confidence/margin against *decision-relevant* correctness, not merely event identity.

### Lifecycle metrics

- Correction-to-index-invalidation latency.
- Stale retrieval rate and stale-use rate, reported separately.
- Retraction propagation and historical as-known replay parity.
- Number of later decisions exposed to an obsolete item before correction is applied.
- Recovery after correction: queries or elapsed event time until decisions return to the corrected reference.
- False deletion of still-valid evidence after a correction.

### Resource frontier

- Total memory bytes including the canonical evidence sidecar, codebooks, indexes, metadata, journals and tombstones.
- Bytes read/written/moved per query and per update.
- Build, insert, correction, retraction, rebuild and replay cost.
- p50/p95/p99 retrieval and end-to-end decision latency.
- CPU/GPU time, allocations and concurrency behavior.
- Tokens, tool calls, provider cost and physical joules only when measured on a declared hardware boundary.

A high recall score with negative `ΔU_memory`, high placebo sensitivity or unsafe stale use is a failure. A fast retriever that increases all-in cost after canonical verification is not an efficiency win.

## 7. Statistical analysis

- The independent statistical unit is the world cluster, not the query, episode, retrieved item or event.
- Use paired cluster bootstrap intervals for each preregistered contrast; retain per-family and per-regime results.
- Pre-register one primary utility contrast and a limited set of safety co-primary outcomes. Correct secondary contrast families for multiplicity (for example, Holm correction).
- Report the point estimate, two-sided 95% interval, cluster count, effective sample size, missing/invalid blocks and sensitivity to cluster weighting.
- Treat missing or invalid evaluator outcomes as invalid runs, not zero-cost successes. Report attrition by arm.
- Freeze the evaluator, utility function and decision threshold before the final split.
- Report all arms, including controls that lose. Do not cherry-pick a single workload where HDC-LTC wins.

## 8. Promotion gates

These are proposed qualification criteria, not current measurements:

**Integrity gates (hard constraints)**

1. 100% of selected decision-support IDs resolve to exact canonical records.
2. Zero invalid, untrusted, unreconciled, retracted-at-known-time, wrong-estimand or wrong-scope evidence is used by the MACE decision policy.
3. Similarity, memory persistence and model explanations cannot override any existing MACE admission predicate.
4. The candidate cannot access evaluator-only labels, outcome tables or arm scores.
5. Corrections preserve historical as-known replay while preventing stale use in current decisions.

**Utility gates**

1. For a positive memory-value claim, the preregistered held-out paired `ΔU_memory` must have a 95% cluster-bootstrap interval whose lower bound is above zero.
2. For a content-specificity claim, `ΔU_content` must be positive; otherwise the result may be context-volume or model-variance rather than useful memory.
3. For an HDC-LTC superiority claim, compare `ΔU_substrate` against the strongest reproduced conventional retrieval baseline under matched all-in memory, latency and compute budgets. Do not infer superiority from comparison with no-memory alone.
4. The preregistered worst-family non-regression margin is at most 2 percentage points in task success, with safety gates remaining absolute. A family-level safety regression cannot be averaged away.
5. Publish risk/coverage and utility/cost Pareto curves. Compression is successful only when decision quality and evidence closure survive at the declared operating point.

A result can pass integrity and fail utility. It can also improve utility and fail integrity. Keep those outcomes distinct.

## 9. External benchmark alignment

This is a MACE-specific causal intervention extension, not a replacement score for other benchmarks:

- **MemoryArena (ICML 2026):** preserve its interdependent multi-session Memory-Agent-Environment task and scoring contract; its central lesson is that recall alone does not ensure downstream agentic task success.
- **AMA-Bench (ICML 2026):** preserve the long-horizon trajectory and official evaluation protocol. The public repository's test path must remain sealed; do not tune against test answers.
- **WorldMemArena (2026):** map diagnostics to its write, maintain, retrieve and use lifecycle and include genuine observation/action trajectories for multimodal testing. Do not convert its 400-task benchmark into a synthetic MACE score.
- **Continual Learning Bench (CL-Bench):** use its experience-gain framing as an adjacent check that the system improves across sequential task instances, while keeping its domains and official scoring unchanged.
- **Agent Memory Challenge Cycle 2:** published 2026-11-19, the open-method leaders are Resurface 61.22/100 (Textual), InsightMem 63.33% resolved (Coding) and Khipu 43.26/100 (Multimodal). Commercial/closed-source reference leaders are MemoraX 69.26, 68.00% resolved and 53.68 respectively. These remain separate tracks/divisions; the values are not directly comparable with one another or with MACE utility.
- **NeuroBench:** separate hardware-independent algorithmic correctness/complexity from deployment-aware system timing/efficiency. A CPU/GPU simulation or vector-byte count is not a neuromorphic energy claim.

The official AML stretch targets already recorded in the parent HDC-LTC protocol remain aspirational: 66.22 Textual, 68.33% Coding resolution and 48.26 Multimodal for the next eligible open-method comparison. Stage 2 commercial-frontier targets remain >70.26, >69.00% and >54.68, respectively. These can only be claimed after the official Add/Search deployment and evaluation process.

## 10. Implementation sequence

1. Add this intervention record to the #13128 protocol/schema branch without changing MACE policy or the evaluator.
2. Implement a deterministic evaluator-side arm scheduler over the existing #12743 MACE policies and world-cluster fixtures.
3. Start with no-memory, exact-scope, conventional, HDC-LTC and semantic-placebo arms; add pre/post-correction snapshots next.
4. Emit candidate decisions and evaluator scores as separate JSONL streams using [`data/schemas/mace_memory_decision_intervention_v0_1.schema.json`](https://github.com/Luminous-Dynamics/symthaea/blob/main/data/schemas/mace_memory_decision_intervention_v0_1.schema.json). The candidate stream contains no evaluator-only labels; the evaluator stream is held outside the candidate's read boundary.
5. Add tests that fail if candidate input contains hidden labels, if pair blocks differ in any frozen field other than the assigned memory input, or if a decision-support ID is unresolved.
6. Add the sampled evidence-knockout audit and cluster-bootstrap summary.
7. Only after a local executable baseline exists, add official benchmark adapters or hardware-specific profiling.

## 11. Current execution boundary

This increment is a research design and machine-readable record contract. It does not yet add an executable scheduler, Rust adapter, policy mutation, benchmark result or measured performance gain. The GitHub PR remains draft until those implementation and validation gates are satisfied. CI skipped/pending is not a pass.
