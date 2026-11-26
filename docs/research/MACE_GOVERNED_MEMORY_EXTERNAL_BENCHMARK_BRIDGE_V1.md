# MACE Governed-Memory External Benchmark Bridge V1

**Snapshot:** 2026-11-26  
**Status:** Benchmark integration specification. This document does not claim official benchmark runs, reproduced baselines, or MACE scores.

## 1. Objective

MACE should be evaluated as an evidence-qualified memory and decision system, not only as a retriever. The external evaluation plan therefore measures distinct transitions:

`evidence acquired → qualified → integrated → decision changed or intentionally unchanged → feedback qualified → future behavior updated`

A result at one stage must not be used as a proxy for the next. Retrieval recall, provenance correctness, permission correctness, answer utility, natural integration, lifecycle correctness and later feedback reuse are separate outcomes.

This bridge preserves each benchmark's own task definitions, graders, denominators and reporting units. It does not combine unrelated scores into a synthetic universal leaderboard.

## 2. Priority 0: TimlMemBench

Official project: https://github.com/lyirs/TimlMemBench

### Why first

The public release includes data and evaluation code, two distinct tracks, a documented JSONL adapter protocol, and explicit lifecycle/governance dimensions that map directly onto MACE's bitemporal ledger and invalidation work.

Release scope (verify again when pinning a run):
- 208 bilingual samples.
- 1,566 sessions and 1,715 questions.
- 1,040 governance cases and 781 activation cases (1,821 governed-memory cases).
- Nine diagnostic axes: Write, Revoke, Supersede, Preserve, Scope, Provenance, Boundary, Activation and Proactive.

### Required tracks

**A. Canonical-pool LLM-QA control**

Every system receives the same sample-local memory pool. This isolates selection, retrieval and answer generation over prepared evidence. Retain the official measures separately:
- Judge Accuracy.
- Answerable Judge.
- Abstention.
- Retrieval Accuracy (all expected IDs and no forbidden IDs).
- Combined Accuracy (judge and retrieval both correct).

**B. Raw-native governed-memory evaluation**

Reset for each sample, ingest raw sessions chronologically using the official adapter protocol, and answer governance/activation queries from MACE's own state. The published adapter command sequence is `reset`, `ingest_session`, `query`, `close`.

Do not expose gold expected answers, expected behavior, forbidden spans, gold state identifiers or source evidence annotations to MACE before prediction. Preserve the official label-leakage boundary. Record provider/model, judge version, adapter command, system commit, language, split and run date.

Report all nine axes separately. The benchmark documentation explicitly treats them as diagnostics rather than defining one aggregate leaderboard score.

### MACE-specific preregistered safety strata

In addition to official scoring, record a separate safety sidecar:
- revoked memory used as current;
- superseded value selected over a current value;
- cross-user or cross-role scope leak;
- unsupported provenance assertion;
- forbidden memory retrieved or surfaced;
- prohibited proactive disclosure;
- action/answer issued while a required evidence obligation is unresolved.

These do not replace official labels or redefine the benchmark. Count severe violations as counts and rates with denominators, not as a weighted score that can be averaged away.

**Proposed target:** at least +5 absolute percentage points on preregistered lifecycle/governance dimensions versus the strongest reproducible matched baseline, while retaining zero observed critical unauthorized-use or cross-scope disclosure events in the evaluated sample. This is a target, not a measured MACE result. Zero observed events in a finite test is not proof of zero risk.

## 3. Priority 1: StreamMemBench

Official project: https://github.com/landian60/StreamMemBench

### Why second

StreamMemBench adds the longitudinal behavior that a static memory QA score cannot establish. Its v1 release is described as 6 participants over 7 days, 3,347 stream segments, 2,854 evidence-task items, 8,107 evidence anchors and 16,214 tasks. It includes English and Chinese releases derived from EgoLife.

The official trajectory is:
1. Ingest the chronological stream segment.
2. Audit newly saved records for fidelity.
3. Answer the initial request.
4. Receive simulated or supplied feedback.
5. Revise the answer when feedback requires revision.
6. Store the evaluated interaction.
7. Answer the follow-up request.
8. Score the evidence-anchored trajectory.

Use the four canonical metrics without renaming:
- `fidelity`
- `initial_evidence_use`
- `feedback_incorporation`
- `followup_reuse`

Fidelity's default `saved_records` mode is not interchangeable with `retrieve_probe`. Do not mix those modes in a matched comparison. Follow the official non-applicable feedback denominator rule. The memory system must not receive `expected_behavior`, gold evidence statements/spans, metric scores or judge rationales.

### MACE extension

Attach a MACE evidence capsule to every scored trajectory, mapping:
- initial evidence records to qualified source/derivation roots;
- correction/feedback to a new bitemporal event;
- invalidated derived projections to explicit dependency roots;
- follow-up answer to the post-feedback memory snapshot;
- no-change outcomes to an explicit "feedback evaluated; no memory mutation required" receipt.

This tests whether MACE can learn from feedback without treating every user assertion as ground truth. Feedback must be checked for authority, scope and evidence before it changes shared memory.

**Proposed target:** +5 absolute percentage points on initial evidence use, feedback incorporation and follow-up reuse against the strongest matched baseline; no degradation in fidelity. Report each metric and its applicable denominator separately. These are research targets, not MACE results.

Privacy boundary: StreamMemBench is derived from egocentric daily-life data. Follow its data terms, avoid re-identification, and do not infer sensitive personal attributes beyond the released annotations. Its dataset card explicitly says the release is not intended to score clinical, legal, financial or safety-critical assistant quality.

## 4. Priority 2: COLLABMEM

Paper: https://aclanthology.org/2026.emnlp-main.1194/

COLLABMEM isolates the collaborative-memory layer with a matched memory-only harness. Its published scope includes 58 hand-authored scenarios (1,118 core probes), a 174-scenario scaled split (3,359 probes), adversarial/compositional hard sets and six transactive-memory competencies. TMS-GRAPH++ reports a 69.6% non-Oracle average across five evaluation splits. The paper says the data, evaluator and reproduced outputs will be released; availability and exact version must be verified before a matched run.

### MACE comparison

Run the official matched harness, keeping trace, target role, answerer, judge and gold event fixed. Compare against official SDK baselines (Mem0, Letta and A-Mem) as well as the strongest published non-Oracle architecture once the artifact is public.

**Proposed stretch target:** 74.6% non-Oracle average on the exact official five-split protocol (a +5.0 point target over the reported 69.6%), with no decline in role access, provenance, authority, routing, conflict-resolution or retention strata. This is not an achieved score. Do not tune on hidden/adversarial splits.

MACE's candidate differentiator is not simply graph structure. It is independently enforced visibility and capability scope, with each derived claim bound to writer identity, origin evidence and current lifecycle state.

## 5. Priority 3: UtilMem and MemUse

### UtilMem

Official repository: https://github.com/peijunallin/UtilMem  
Dataset: https://huggingface.co/datasets/KrisQ/utilmem

The project README describes 1,717 instances across learning support, finance guidance, mental wellness, fitness coaching and document analysis, with distributed evidence synthesis and semantically similar distractors. At this snapshot, the repository says the dataset is released but evaluation and benchmark-construction code are still forthcoming.

Therefore:
- preserve the public dataset version and licensing;
- do not claim official reproduction until the official scorer/harness is available;
- build a local diagnostic adapter only if clearly labelled as non-paper-comparable;
- test evidence coverage, contradiction retention, distractor rejection and long-form synthesis separately.

### MemUse

Paper: https://aclanthology.org/2026.emnlp-main.804/

MemUse distinguishes direct QA from natural integration in conversation and reports that recall accuracy did not track user satisfaction in its evaluated conditions. Its reported 78.8% direct-QA / 7.9% natural-reference example is a warning against rewarding memory mentions as an objective.

MACE should add two complementary diagnostics:
- **Appropriate integration:** a relevant memory is used naturally when it improves the response.
- **Appropriate non-use:** a memory is withheld when irrelevant, stale, sensitive for the current purpose, unauthorized or likely to distort the answer.

Do not maximize the raw count of memory references. Measure task usefulness, unnecessary personalization, unsupported personalization, privacy-boundary violations and satisfaction using a matched, consented study. Treat satisfaction associations as observational unless randomized and powered.

## 6. Priority 4: DeepControl and ServeLearnBench

### DeepControl

Official code: https://github.com/xiongsiheng/DeepControl  
Paper: https://arxiv.org/abs/2602.01672

DeepControl controls retrieval continuation and granularity, and reports +9.4 points over Search-R1 on Qwen2.5-7B and +8.6 points on Qwen2.5-3B across seven benchmarks.

MACE must reproduce the paper's exact model, benchmark splits, retriever, prompts, decoding settings and resource budgets before making a direct numerical comparison. Run fixed-top-k, one-shot information-value, DeepControl-style continuation/granularity and MACE adaptive EVSI/robust-policy arms with the same action space and terminal decision gate.

The official implementation describes an approximately 80 GB RAM CPU index or about 32 GB GPU index. A full reproduction is therefore a separate, resource-intensive task, not a local test to silently substitute with a smaller model or truncated corpus.

**Research target:** first reproduce the published comparison; only then preregister a >2-point improvement on the same primary task metric under matched or lower all-in resource cost. Never compare different model sizes or resource budgets as though they were equivalent.

### ServeLearnBench

Official project: https://infini-ai-lab.github.io/ServeLearnBench/

This benchmark tests agent learning from serving experience, future behavior and the cost of that learning. Its published leaderboard exposes hidden reward, fully specified reward and total API cost. Use it as a later feedback-learning benchmark after MACE has a qualified outcome/feedback ingestion path.

Compare on the cost-performance Pareto frontier, and report retention/regression after each learning window. A lower API cost with degraded hidden-task performance is not a win; a higher score bought with substantially more tokens/calls is not automatically a better system.

## 7. Additional research tracks

| Track | Why it matters | MACE experiment |
|---|---|---|
| Knowing the Store | Agents can only reason correctly about memory lifecycle if the backend exposes truthful state metadata | Evaluate metadata completeness, lifecycle-state summaries and whether store-state errors affect later decisions |
| CMSC-E memory canonicalization | The same stored claim can be interpreted differently across model families | Measure factual and emotional semantic drift across readers; preserve qualifiers and uncertainty |
| DyadMem | Agent-user relationship memory evolves beyond static user facts | Test interaction protocol, shared-history updates, correction and relationship-specific retention |
| Memory-benchmark registry | Broad landscape discovery; its entries have different reproduction levels | Use it to discover tests, but verify every benchmark through its official paper/repository |

CMSC-E's reported pilot uses 176 synthetic memory objects and three model families. Its emotional-consistency result did not survive correction for multiple comparisons, and no factual-drift comparison was significant. Treat it as an exploratory diagnostic, not evidence that canonicalization is already solved.

The public memory-benchmark registry lists 207 benchmarks in this snapshot and marks reproducibility levels separately. Its counts include 139 official-script entries, but no entries marked smoke-tested or full-run in the registry snapshot. An official script is not a reproduced result.

## 8. Cross-benchmark outcome model

Use a layered scorecard rather than one composite number.

| Layer | Primary questions | Example metrics |
|---|---|---|
| Acquisition | Did the system get the needed evidence efficiently? | useful evidence acquired, retrieval recall, cost, latency, bytes |
| Qualification | Is the evidence authentic, current, in scope and permitted? | provenance validity, independent-root sufficiency, stale-use rate, rights violations |
| Integration | Did the answer synthesize evidence faithfully? | evidence coverage, contradiction preservation, distractor resistance, calibration |
| Decision | Did evidence change the right decision, or appropriately leave it unchanged? | task utility, regret, abstention quality, unsafe-action count |
| Governance | Did the right agent see and use the right memory? | role leak, scope leak, revocation compliance, provenance and routing |
| Learning | Did feedback improve later behavior without poisoning state? | feedback incorporation, follow-up reuse, post-update regression |
| Experience | Was memory use useful and natural? | satisfaction, unnecessary personalization, appropriate non-use |

Every reported aggregate must retain its underlying per-axis counts and denominators. Severe permission and rights failures are veto conditions, not costs that can be offset by accuracy elsewhere.

## 9. Benchmark execution order

1. Pin and run TimlMemBench canonical-pool and raw-native tracks against no-memory, full-context, BM25/RAG, and external memory-system baselines.
2. Run StreamMemBench with raw, extracted, and hybrid retrieval arms, ensuring exact chronological ingestion and feedback reuse.
3. Freeze a MACE capability/authority manifest and run COLLABMEM as soon as the exact official artifact is available.
4. Add UtilMem once its official evaluator is released; until then, label all runs local diagnostics.
5. Run matched DeepControl and ServeLearnBench tracks with exact model/runtime/resource commitments.
6. Add CMSC-E, Knowing the Store and DyadMem as targeted cross-model, metacognitive and relational diagnostics.

## 10. Evidence capsule requirements

Every benchmark run must use the MACE Benchmark Result Evidence Capsule V1 introduced in the adaptive-oracle work. At minimum pin:
- official benchmark release, harness commit and digest;
- dataset split and task-ID digests;
- scorer/grader and prompt digests;
- candidate repository and exact commit;
- model/provider/checkpoint and decoding configuration;
- tool, memory-adapter and runtime versions;
- task denominator, attempted/completed/invalid/skipped/timeouts/abstentions;
- safety-stratum counts and excluded cases;
- token, tool-call, storage/byte, latency and monetary-cost measurements;
- matched baseline run IDs, pairing method and uncertainty interval;
- independent replay and claim status.

Use claim levels honestly: planned, executed, officially scored, independently replayed, or field validated. A planned manifest, authored adapter or skipped CI run is not an execution result.

## 11. External comparison principles

Compare MACE to Mem0, Letta, A-Mem, Graphiti/Zep, MemoryOS and other systems through their official implementations under the same harness, model, dataset, permissions and resource limits. Report both:
- **capability comparison:** what operations and guarantees the system supports;
- **measured comparison:** exact benchmark results under matched conditions.

Do not claim a general industry win from an architecture diagram or one benchmark. The defensible research claim is narrower: MACE is attempting to make provenance, authority, lifecycle and decision-value constraints enforceable across the complete evidence-to-action path, then measure whether that increases useful performance per unit cost without creating governance failures.

