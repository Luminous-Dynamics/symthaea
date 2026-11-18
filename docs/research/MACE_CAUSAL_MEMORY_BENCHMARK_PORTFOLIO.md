# Causal Agent Memory: Benchmark Portfolio and SOTA Targets

**Snapshot date:** 2026-11-18  
**Programme:** Broca / Symthaea MACE — Memory-Augmented Causal Experience transfer  
**Status:** Research protocol and targets. Targets below are aspirational; they are not achieved results.

## 1. North-star claim

A capable agent memory system must do more than recover facts. It must improve later decisions by retaining *which action caused which outcome, under which conditions, for which objective, with what evidence and uncertainty*—and then refuse to apply that knowledge when its assumptions no longer hold.

The target architecture is therefore:

`raw events → provenance-bound experience → causal/effect contract → applicability test → decision or safe abstention → observed outcome → correction/revocation`

Memory is treated as an evidence-bearing control surface, not merely a retrieval index. A comparative action choice requires qualified evidence for both alternatives; one-sided evidence must lead to abstention or a bounded information-gathering request.

## 2. Current external benchmark landscape

Scores below are **not interchangeable**. Different benchmarks, tracks, models, judges, and interaction protocols support different claims.

| Benchmark / track | Published result to anchor against | What it measures | What it misses for MACE |
|---|---|---|---|
| [MemoryArena (ICML 2026)](https://proceedings.mlr.press/v306/he26am.html) | No single universal headline score in the paper abstract; the key finding is that strong long-context recall does not ensure success on interdependent multi-session tasks | Memory-Agent-Environment loops across web navigation, preference-constrained planning, progressive information search, and sequential formal reasoning | Explicit source-family attacks, revocation latency, and statistical transportability gates are not its central metric |
| [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html) | AMA-Agent: **57.22%** average accuracy, **+11.16 percentage points** over the strongest baseline reported by the authors | Long-horizon real/synthetic agent trajectories, including recall, causal inference, state updating, and state abstraction | Governed revocation and intervention-effect transfer under controlled regime changes are not the sole evaluation focus |
| [Agent Memory Leaderboard — Cycle 1, Textual / Commercial](https://agentmemories.ai/results/cycle-1/) | MemoraX v0.5: **58.02 overall**; **29.99** on Context Learning & Execution | Explicit fact recall, compositional inference, temporal reasoning, governance, personalization, execution, safety/privacy | A composite leaderboard score does not itself prove safe causal transfer in an interactive environment |
| [Agent Memory Leaderboard — Cycle 1, Textual / Open-source](https://agentmemories.ai/results/cycle-1/) | InvMem: **45.06 overall** | Same textual track, open-source division | Same limitation: high recall/composite score is not equivalent to lower decision regret after interventions |
| [Agent Memory Leaderboard — Cycle 1, Coding](https://agentmemories.ai/results/cycle-1/) | Commercial: MemoraX **62.00% resolved**; open-source: AM-Link **52.67% resolved** | New-feature and bug-fix resolution using persistent coding memory, with resource categories | Does not isolate evidence qualification, causal mechanisms, or stale-knowledge revocation as primary outcomes |
| [TimlMemBench (Findings of EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.507/) | **208** reviewed samples, **1,566** sessions, **1,715** questions, and **1,821** governance/activation cases | Raw-native governed memory: revocation, supersession, scope/boundary control, provenance, activation, restraint | Dialogue-oriented; does not by itself measure action-effect transport in continuous agent-environment loops |
| [StateMemBench (Aug 2026)](https://arxiv.org/abs/2608.19652) | 234 multi-session state-tracking scenarios; StateMem reports **0.205 → 0.363** on DeepSeek-V4-Flash (1.8×) and **0.149 → 0.233** on Qwen-3.5-9B against its strongest reported baselines | Current vs superseded facts and relational dependencies over long interactions | Not primarily a causal intervention-transfer or signed-provenance benchmark; backbone, task split and protocol-specific scores must remain attached |
| [Memora / FAMA (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.1337/) | No single universal score in the abstract; introduces **Forgetting-Aware Memory Accuracy (FAMA)** | Weeks-to-months user conversations across remembering, reasoning and recommending, with penalties for obsolete-memory use | Personal memory benchmark; not an environment-intervention benchmark |
| [ConsolidationBench (Sep 2026)](https://arxiv.org/abs/2609.33013) | One reported phase: **+22.7% task success at 7× compression**. A later external-transfer study reports pooled Spearman **ρ = −0.24**, n=12, confidence interval spanning zero, between consolidation-quality score and real transfer accuracy | Decisions to retain, compress, promote, abstract or forget; governance and auditability | Its key warning is directly relevant: consolidation quality is not a validated proxy for downstream transfer; score quality, governance and transfer independently |
| [AgentMemoryBench (Lifelong Agent @ ICLR 2026)](https://openreview.net/attachment?id=MSXbrNExax&name=pdf) | Workshop benchmark; no single cross-track headline score used here | Joint system/experiential memory and personal/factual memory across online learning, transfer and forgetting | Useful for co-evolving procedural and personal memory; its protocol is distinct from MACE causal transfer |
| [LongMemEval reproduction (Redis, 2026-09-21; page updated 2026-11-17)](https://redis.io/resources/agent-memory-benchmark-report/) | Redis Remis + Instruct: **86.5%** task-averaged accuracy on LongMemEval; reported at about **$0.0703/session** on the report's common GPT-4o reproduction | Long-term question answering and memory retrieval, including cost and query latency | QA accuracy should not be read as multi-session action competence; this is a vendor-authored reproduction report, not a cross-benchmark SOTA |
| [Survey of World Model Benchmarks (snapshot 2026-11-08)](https://world-model-benchmarks.github.io/) | Catalogues **130** benchmarks across seven target categories, two protocol classes, two metric levels, and four data sources | Clarifies that claims must specify target, protocol, metric family and data source | A taxonomy, not a direct memory-agent score |

### Interpretation

The industry now has distinct work on retrieval, state tracking, obsolete-memory penalties, consolidation decisions and interdependent agent-environment loops. Those are complementary—not interchangeable—capabilities.

StateMemBench is especially important because it isolates current-vs-superseded state rather than treating all prior facts as equally useful. FAMA turns obsolete-memory use into an explicit penalty. ConsolidationBench warns that optimizing an internal consolidation score may not improve external transfer at all. MACE should therefore report at least four independent axes: state correctness, governance/lifecycle correctness, causal transfer and downstream task utility.

### Prior interpretation (retained)

The field has moved beyond “can the system remember a conversation?” MemoryArena couples memory with later action; AMA-Bench emphasizes machine-generated trajectories and causal/objective information; TimlMemBench tests governed use from raw sessions; the Agent Memory Leaderboard separates capability tracks; LongMemEval remains a useful retrieval/QA control.

The opportunity is not to claim that MACE's synthetic score beats one of these scores. The opportunity is to add a missing, independently testable capability layer: **evidence-qualified causal transfer under distribution shift, source corruption, and lifecycle correction**, then reproduce the official benchmarks under their own protocols.

## 3. Target scorecard

These are programme targets for future matched evaluations, not current measured performance.

### External benchmark targets

| Benchmark | Baseline anchor | MACE target | Qualification rule |
|---|---:|---:|---|
| MemoryArena | Use the official baseline under the selected agent/backbone; do not infer a single SOTA from abstract | **+5 pp** mean task success over no-memory and strongest reproduced memory baseline; no task family worse by more than **2 pp** | Same base agent, task episodes, tool permissions, and token/compute budget; paired 95% interval for improvement excludes zero |
| AMA-Bench | AMA-Agent **57.22%** reported | **≥62%** with the exact published model, judge, data and evaluation protocol | Reproduce the 57.22 result first; publish recall, causal, state-update and state-abstraction sub-scores separately |
| Agent Memory Leaderboard — Textual | MemoraX v0.5 **58.02 overall**, **29.99 Context Learning & Execution** | **≥65 overall**, **≥45 Context Learning & Execution**, and **≥70** on both Governance and Safety/Privacy | Same Cycle/track/division, prompt, model class, memory budget, and evaluator version |
| Agent Memory Leaderboard — Coding | MemoraX **62.00%** resolved; AM-Link **52.67%** open-source | **≥70%** resolved overall; report new-feature and bug-fix rates separately | Same coding benchmark, repository/task split, toolchain, model and resource budget |
| TimlMemBench | 1,821 governance/activation cases | **Zero known unauthorized activations** on the supplied revoked/scope-incompatible cases; report provenance recovery and legitimate activation separately | Run the raw-native track; system must ingest raw sessions itself, not receive a prebuilt canonical memory pool |
| LongMemEval | Redis reproduction **86.5%**, about **$0.0703/session** | **≥87%** at **≤$0.07/session** on the same common GPT-4o setup | Separate QA/retrieval track; report ingest time, query latency, cost and accuracy together |

### Additional external qualification targets

| Benchmark | Proposed target | Qualification rule |
|---|---:|---|
| StateMemBench | At least **+5 percentage points** absolute current-state accuracy over the strongest reproduced same-backbone baseline | Official split and scoring; matched context/token budget; no future-knowledge leakage; separately report superseded-state errors |
| Memora / FAMA | At least **+5 points** over a matched reproduced memory baseline; **zero** obsolete-memory reuse on deterministic retraction controls | Run the official task protocol and FAMA scoring; report forgetting/false-deletion trade-offs |
| ConsolidationBench | Reproduce the official protocol, then target **+5 pp** downstream task success at a matched memory/storage budget without governance regressions | Do not optimize or claim victory on consolidation-quality score alone; require held-out external transfer, independent labels and separate poison/reversibility/audit metrics |
| AgentMemoryBench | Reproduce its procedural/personal memory protocol, then report per-track gains under matched backbone and budget | Keep personal-memory privacy/governance separate from procedural transfer and causal evidence claims |

### MACE-specific qualification targets

| Dimension | Target |
|---|---:|
| Duplicate event-ID acceptance as independent evidence | **0** |
| Use of retracted, unreconciled, untrusted, confounded, or incomplete-horizon evidence in a qualified decision | **0** in known-control suites |
| Transfer across a different optimization objective | **0** |
| Known Session-C retraction reused by the next decision | **0** |
| False acceptance of a deliberately changed mechanism | **≤0.1%** across at least **10,000** held-out negative trials; report a binomial confidence bound |
| Acceptance of genuinely invariant, independently witnessed few-shot transfers | **≥80%** on held-out positive cases |
| Mean normalized utility lift vs no-memory | **≥0.05**, with paired cluster-bootstrap 95% lower bound **>0** |
| Worst-family regret | **≤0.05** normalized utility units |
| Confidence calibration (once the policy emits probabilities) | ECE **≤0.03** and Brier score better than frequency-only and retrieval-only baselines |
| Decision-time overhead | Report p50/p95/p99 latency and CPU/energy per decision; target p95 deterministic policy overhead **<100 ms** on the declared reference host |

Do not report an internal metric as a leaderboard win. Synthetic controls qualify implementation properties; external benchmarks qualify external validity.

## 4. Paradigm shifts to pursue

### A. From text retrieval to causal contracts

A memory item should compile to a typed contract:

- action and outcome;
- objective and operating regime;
- preconditions and excluded conditions;
- measured effect distribution and uncertainty;
- intervention and confounder-control status;
- source event IDs and source-family lineage;
- observation horizon and delayed consequences;
- revocation/supersession state;
- applicable policy and authority scope.

A vector similarity hit is a candidate pointer, not permission to apply a memory.

### B. From exact-scope rejection to evidence-tested transportability

Exact-scope matching is safe but can under-generalize. Blind similarity transfer is unsafe. The intermediate capability is to test whether the action-effect relationship remains invariant in the target regime, with target-regime witnesses, disjoint source families and upstream roots, and conservative measurement-error intervals.

The current `invariant_transport` policy is only a heuristic prototype. It requires the worst-case difference between source and target effect intervals to fit within a fixed 0.12 normalized-utility margin. This deterministic bound is not a statistical confidence interval; the measurement bounds and source-root IDs require independent attestation. A later version should use preregistered statistical equivalence tests and calibrated uncertainty before any real deployment claim.

### C. From accuracy to regret, calibration, and governed-use

A memory system should be scored on:
- utility and regret of downstream actions;
- wrongful action and wrongful abstention;
- calibration and selective risk;
- unsupported causal claims;
- stale evidence reuse after correction;
- cost and latency across ingestion, consolidation, retrieval, verification and update.

A system that answers more questions but takes worse actions is not a better memory system.

### D. From passive memory to active information acquisition

When no memory is applicable, the agent should identify the cheapest safe experiment that distinguishes plausible mechanisms. Measure value of information, experiment cost, time-to-identification, and whether the experiment changes the next decision. Abstention is a safety property; active, low-risk probing is the capability that makes abstention useful.

### E. From trust labels to independently verifiable provenance

A different source-family string is not proof of independence. The production gate must use source-ledger attestations, event identity, common-cause analysis, and replayable receipts. Evaluator custody must also be independent of the candidate package.

## 5. Protocol needed to make a credible SOTA claim

1. **Reproduce each official benchmark first.** Lock model/backbone, prompts, judges, data versions, token/tool budgets, and stopping rules. No score pooling across benchmarks.
2. **Use matched baselines.** No-memory, full-context, vector retrieval, frequency-only, exact-scope causal memory, graph/tool-augmented memory, invariant transport, and active-probe policies must share the same agent and resource budgets.
3. **Separate development from evaluation.** Split by world seed, source family, mechanism, task template and environment—not random rows. Keep evaluator labels outside candidate dependencies and ultimately in a separately controlled repository/service.
4. **Use paired cluster-level statistics.** Resample environments/tasks, not correlated questions from the same environment. Publish effect sizes, confidence intervals, worst-family results, and failure examples.
5. **Attack the evaluator and evidence pipeline.** Include event-ID aliases, copied/near-duplicate sources, common-cause sources, fabricated success receipts, delayed effects, retractions, supersession, wrong objectives and adversarial regime changes.
6. **Measure the full lifecycle.** Include ingestion/consolidation, index/storage growth, retrieval, verification, tool calls, tokens, CPU/GPU, energy, p50/p95/p99 latency, and cost per successful downstream task.
7. **Publish the negative results.** A transport policy that refuses a true invariant effect or accepts a false one must be visible in the report, not hidden in an aggregate score.

## 6. Current MACE implementation status

- **MACE-lite fixed fixtures:** 17 deterministic scenarios and six candidate policies: no-memory, frequency-only, scope-aware, exact-scope causal-qualified, invariant transport, and active probe.
- **Generated stress suite:** 256 deterministic clusters × 16 scenario families = **4,096** cases. Includes confounding, regime/objective shifts, source retraction, unreconciled tool success, untrusted evidence, incomplete delayed horizons, correlated duplicates, event-ID aliasing, shared-common-root attack, error-interval negative control, negative experience, invariant few-shot transport, mechanism break, and an unseen-target active-probe case.
- **Statistics:** paired cluster-bootstrap utility lift versus no-memory (2,000 resamples); per-family reporting for causal-qualified, invariant transport and active probe; Session-B/Session-C known-retraction control; probe requests/resolutions and net utility after the 0.12 normalized-utility probe cost. The active policy emits a typed, scope-bound request with both actions, a replicate count and maximum normalized-utility budget; the evaluator rejects mismatched-scope, unsupported-action, unsupported-replicate and under-budget requests.
- **Raw-native ingress and consolidation handoff:** broca-mace-ingress plus broca-mace-consolidation; ingress has 32 deterministic mock-verifier tests, the consolidation policy has 7 unit tests and one cross-crate Session-A→B→C integration test, tracked by [#12771](https://github.com/Luminous-Dynamics/symthaea/issues/12771). It admits only Mycelix-style `receipt_attested` authority, uses signed server `received_at` for `known_at`, distinguishes Active/Retracted/Superseded lifecycle, validates source/calibration/causal/outcome receipt material, re-verifies duplicates and compares full receipt bundles. Ingress results are non-constructible outside the crate. `build_consolidation_evidence_pack` produces a deterministic receipt-preserving pack with source-family/root counts derived from verified lineage. The MACE-specific proposal consumer now consumes action-scoped summaries and binds all promotion thresholds into each source-linked proposal; invalid policy configuration fails closed. #12731's broader retention/privacy/reversibility policy does **not yet consume the pack**. No live Mycelix cryptographic adapter is claimed.
- **Candidate/evaluator separation:** candidate policy and evaluator are separate Cargo packages; the evaluator source remains public in the same repository, so it is not a sealed independent evaluator.
- **Verification:** code has been committed to draft PR #12743. Rust compile/tests/rustfmt have not run in the current execution environment (no Cargo/Rust toolchain is installed); GitHub draft checks are skipped. No test pass or SOTA result is claimed.

## 7. Immediate next tranche

1. Implement a production Mycelix verifier for canonical event bytes, signed authority receipt chain, source lineage, calibration, outcome reconciliation and lifecycle receipts (#12771, #9075, #297).
2. Wire ActionConsolidationEvidenceSummaryV1 into the separately governed lifecycle policy (#9139 / #12731), retaining source ancestry and reversible state transitions; explicitly model privacy erasure and supersession events rather than treating active-only ingress as the entire lifecycle. Before calling this raw-native transcript evaluation, add transcript turn/claim extraction with exact source-span receipts. Require per-action root/family support rather than pooled global episode counts.
3. Run one end-to-end raw Session-A → consolidation → Session-B decision → Session-C correction pipeline using unchanged event and receipt identities (#9516).
4. Move hidden labels and evaluator ownership to independent custody (#11816), and add source-/seed-/mechanism-held-out worlds.
5. Replace the fixed deterministic interval margin with calibrated uncertainty and preregistered statistical equivalence; retain the deterministic check as an inexpensive negative control.
6. Add official MemoryArena / AMA-Bench / StateMemBench / FAMA adapters only after licensing and evaluator contracts are reviewed; prepare Add/Search for the next eligible Agent Memory Challenge cycle.
7. Measure paired utility, worst-family regret, calibration/coverage, lifecycle cost, supersession/retraction latency and p50/p95/p99 performance under matched budgets (#8427).

## References

- He et al., [Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks](https://proceedings.mlr.press/v306/he26am.html), ICML 2026.
- Zhao et al., [AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications](https://proceedings.mlr.press/v306/zhao26bs.html), ICML 2026.
- [Agent Memory Leaderboard — Cycle 1 results](https://agentmemories.ai/results/cycle-1/), published 2026-08-12.
- Zhou et al., [TimlMemBench](https://aclanthology.org/2026.findings-emnlp.507/), Findings of EMNLP 2026.
- [Redis Agent Memory benchmark report](https://redis.io/resources/agent-memory-benchmark-report/), 2026-09-21; page updated 2026-11-17.
- Zhang et al., [A Survey of World Model Benchmarks](https://world-model-benchmarks.github.io/), final manuscript snapshot 2026-11-08.


Each experience now carries a claimed absolute measurement-error bound. Invariant transport requires:

`abs(mean_source - mean_target) + source_mean_error_bound + target_mean_error_bound <= 0.12`

This is a conservative deterministic interval-containment check, not a statistical confidence interval. Error bounds are candidate-visible claims in this prototype and need sensor/protocol attestation; future work should add calibrated uncertainty models, preregistered equivalence tests and independent target-regime witnesses.

## 8. Provider and benchmark ecosystem snapshot (2026-11-18)

OpenAI's Agents SDK distinguishes durable sandbox memory from conversational session history. Its documented design uses a small summary for initial context, progressive retrieval of detailed rollout summaries, and two-stage extraction/consolidation of memories. It supports separate memory layouts for different agents. Anthropic's Managed Agents describes independent session-log, harness and sandbox interfaces, and cautions that harness assumptions can become stale as models improve. Anthropic's long-running-agent pattern uses an initializer plus incremental sessions that leave durable handoff artifacts.

For Symthaea, these are useful architectural patterns, not systems to copy wholesale:
- retain append-only, bitemporal event history and provenance in Mycelix;
- have Symthaea produce versioned, evidence-bound, progressively disclosed memory views;
- keep OpenAI/Anthropic/local model adapters replaceable;
- keep Symtropy execution isolated and separate from reasoning;
- treat model-generated memory and tool responses as untrusted until verified;
- rerun behavioral qualification on provider/model/SDK upgrades.

TimlMemBench's raw-native track exposes a remaining gap: MACE now tests receipt-gated structured interaction envelopes, receipt-preserving pack projection, action-scoped evidence summaries and a proposal-only consolidation consumer. The fixed/generated MACE evaluators still use structured Experience fixtures, and ingress does not yet parse natural-language transcript turns or bind extracted claims to exact text spans. The broader governed lifecycle policy in #12731 (privacy erasure, reversible storage transitions, retention and forgetting) still does not consume the MACE evidence pack. The next seam is transcript/span provenance plus a single raw transcript → verified evidence → governed lifecycle → Session-B decision → Session-C correction evaluator.

The Agent Memory Challenge Cycle 2 official page, checked on November 18, 2026, says results are **expected November 19, 2026**. The October 31 submission deadline and November 4 evaluation close have passed; Cycle 2 is not open for new submissions. Its Add/Search → platform Answer/Eval contract is a target for a future MACE/Mycelix adapter. Do not claim a Cycle 2 rank before official publication.


### Mycelix lineage and measurement assurance

The synthetic `source_root_id` and `measurement_error_bound` fields are not trust anchors. Mycelix's FIN-SYS source profile provides a useful existing pattern: multiple publications reusing one upstream BIS/IMF series count as one provenance family. Production MACE should consume equivalent verified lineage receipts, including acquisition mechanism, artifact digest, source family, upstream series/root, transformation and supersession history.

A measurement-error bound must be tied to an identified sensor/method, units, calibration revision, operating interval and evidence record. It must not be an agent-editable confidence claim. This connects MACE to the bitemporal source-event ledger (#9075) and independent evaluator provenance (#11816).
