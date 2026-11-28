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

## 12. Research update: non-regressive learning, claim-relative invalidation and metamemory

This addendum supersedes the earlier sequencing where necessary. These benchmarks expose failure modes that map directly to the existing MACE temporal ledger, invalidation and feedback architecture.

### 12.1 ServeLearnBench: adaptation must not destroy prior competence

Official project and leaderboard:
- https://infini-ai-lab.github.io/ServeLearnBench/
- https://infini-ai-lab.github.io/ServeLearnBench/leaderboard.html
- Code: https://github.com/Infini-AI-Lab/ServeLearnBench

The public leaderboard identifies a results snapshot dated 2026-09-25 and says there is one run per model/method cell. Its leading entries include:
- CH with Opus 5: Avg. Hidden 73.4, Avg. Fully Specified 96.2, API cost $2,449.
- CH with GLM-5.3: 69.8, 96.7, $1,077.
- Prime with GLM-5.3 Flash: 61.1, 95.2, $172.
- CH with GPT-5.6 Terra: 55.8, 94.2, $239.

These rows use different backbones and methods; they are not a matched model comparison. The relevant comparison is the Pareto frontier, with exact costs and task populations recorded.

A more important result for MACE is negative transfer: the benchmark reports that on Fully Specified Retail tasks every learning harness falls below the Blind baseline's 98.9%, reaching 89.5–95.5% in the reported aggregate. The ability to learn hidden policies does not guarantee retention of already-correct behavior.

**MACE paradigm shift: continual learning must be retention-constrained.**

Track separately:
- Hidden-task reward after each policy/environment change.
- Fully Specified/stable-task reward before and after the update.
- Stable-task regression and recovery time.
- AULC10 or equivalent learning-speed curve.
- Exploration diversity, discovery efficiency and cost.
- Reversal handling when a previously learned policy becomes invalid.

**Proposed MACE safety target:** no more than 1.0 absolute percentage point regression on the predeclared stable/fully-specified strata after a qualified update, while improving the hidden/changed strata. A run that exceeds the regression budget must trigger rollback, versioned specialization or explicit review. This is a proposed MACE criterion, not a ServeLearnBench requirement or current result.

The objective is not simply reward-after-learning. It is:

`adaptation gain subject to a retained-competence floor and bounded total cost`.

Do not optimize a single scalar that permits gains on hidden tasks to compensate for serious regression in known tasks.

### 12.2 Impact Is Not Invalidation: judge the claim, not the diff

Paper: https://arxiv.org/abs/2609.25130  
HTML: https://arxiv.org/html/2609.25130

The public memory-benchmark registry records 10,369 candidate claims mined from 23 Python libraries, with truth changes verified across 184 changed values and 187 commits. The paper reports that five models asked whether a commit preserves behavior fired on 59–72% of real commits and reached precision 0.291–0.329, while asking whether one specific claim still holds reached precision 0.705–0.974 on the same diffs. The registry classifies this as paper-only; no official downloadable dataset/evaluator was identified in its 2026-10-11 review.

This is directly relevant to MACE derived-memory invalidation. A file changed does not imply every claim sourced from it became false; a diff judged behavior-preserving does not prove every particular stored claim remains true.

**MACE experiment:** make invalidation claim-relative and evidence-dependent:
1. Bind each derived memory to its exact claim, test/assertion, source spans, dependency roots and derivation recipe.
2. Ask whether that claim still holds after a change, rather than classifying the diff globally.
3. Re-run the claim's executable check where possible.
4. Distinguish `still_valid`, `invalidated`, `requires_revalidation` and `unknown`.
5. Propagate invalidation only through the true dependency closure.
6. Never treat missing tests, skipped CI or unknown execution as a passing claim.

Report invalidation precision/recall, coverage, false invalidation rate, stale-claim escape rate, revalidation burden and time-to-correct-projection. This is a high-priority research comparator for PR #13736, but not an executable official comparison until its artifacts are released or reconstructed under an explicitly non-paper-comparable protocol.

### 12.3 Knowing the Store: metamemory depends on truthful backend metadata

Paper: https://arxiv.org/abs/2610.04794  
Official code and data: https://github.com/ansh348/knowing-the-store

The public registry describes 148 unique store questions, each evaluated against three minimally different store versions, for 444 scored items. Its counterfactual obligations include 88 supersession, 30 retraction and 30 expiry cases. The evaluation exposes value-free store summaries and tests whether a reader can tell what should not be used. It is classified as official-script in the registry snapshot.

**MACE opportunity:** lifecycle state must be a machine-readable property supplied by the authoritative store, not a conclusion the language model guesses from content. Expose:
- active/current, superseded, retracted/revoked, expired, quarantined and unresolved states;
- versioned counts and per-record identifiers where the caller is authorized;
- valid-time and observation-time boundaries;
- exact metadata schema/version and completeness receipt;
- correction/revocation and deletion events;
- scope and use-purpose attributes.

Run the official test, then add metamemory fault injection: omit a field, stale a count, contradict a record state, delay a revocation, or provide a wrong schema version. Measure both metadata-fault detection and downstream stale-use escape. A 100% metadata conformance goal is appropriate for controlled exact-schema tests; model-level question accuracy remains a separate metric.

### 12.4 DyadMem: relationship-specific memory is not ordinary user profiling

Paper and project: https://arxiv.org/abs/2610.03020

The paper describes 3,065 episodes, 50,961 sessions and 61,210 QA instances, with session-level Capture and Update supervision, query-level Recall support, and Gold-Memory versus Full-Pipeline QA. It reports Full-Pipeline weaknesses including incomplete capture/recall and unsafe deletion. The public benchmark registry classifies the release as paper-only because it did not identify released episode/evaluator artifacts at its 2026-10-11 review; recheck before any attempted run.

MACE should distinguish:
- facts about a user;
- preferences that may change;
- relationship-specific interaction protocols;
- collaborative commitments and role boundaries;
- temporary working agreements;
- inferred expectations, which must not silently become confirmed facts.

This is a later research direction, not an immediate official reproduction target until artifacts and their licensing are verifiably available.

### 12.5 Updated priority order

1. **TimlMemBench:** first executable external qualification for lifecycle, scope, provenance, activation and restraint. The Rust protocol smoke adapter is a wiring aid only; it is not governed MACE.
2. **StreamMemBench:** first executable longitudinal feedback/use test.
3. **Knowing the Store:** executable metamemory/backend-state qualification and fault injection.
4. **ServeLearnBench:** matched learning-versus-retention/cost frontier, with a predeclared stable-task regression floor.
5. **COLLABMEM:** role-aware multi-agent evaluation when its exact official artifact release is available.
6. **Impact Is Not Invalidation:** claim-relative invalidation research target; paper-only unless the official artifact appears.
7. **UtilMem and MemUse:** integration and appropriate-use tests, respecting their separate release and judging protocols.
8. **DyadMem and CMSC-E:** relationship-specific and cross-model semantic diagnostics after verifying reproducible artifacts.

All results remain disaggregated. A gain in adaptation, recall, relational personalization or answer quality cannot compensate for critical scope, revocation, rights or unauthorized-disclosure failures.

## 13. Evidence refresh: benchmark releases and industry comparison (2026-11-27)

This section supersedes older availability notes where the official status has changed. Benchmark numbers below remain claims of their respective papers/projects, not MACE results.

### 13.1 TimlMemBench is now a peer-reviewed executable target

The paper is published in Findings of EMNLP 2026:
https://aclanthology.org/2026.findings-emnlp.507/

Official repository:
https://github.com/lyirs/TimlMemBench

The official release reports:
- 208 reviewed Chinese/English samples;
- 1,566 sessions and 1,715 questions;
- 1,764 memory events and 1,694 gold states;
- 1,040 governance cases and 781 activation cases (1,821 governed-memory cases);
- two complementary tracks: canonical-pool LLM-QA control and raw-native governed-memory evaluation;
- nine separate diagnostic axes: Write, Revoke, Supersede, Preserve, Scope, Provenance, Boundary, Activation and Proactive.

The official repository provides release validation, a standard-library offline lexical protocol baseline, a heuristic smoke judge, and the complete JSONL process contract. Paper-comparable semantic judging uses the released prompts and the paper's judge model. Do not report the bundled lexical adapter or heuristic judge as a paper baseline.

The MACE Rust protocol adapter implements the reset/ingest/query/close message family and both request shapes. The next qualification step is an exact protocol conformance test against the official runner, followed by raw-native and canonical-pool end-to-end predictions with official semantic judging. Preserve the official split and denominator. The benchmark itself warns that its diagnostic dimensions should not be collapsed into a universal ranking.

### 13.2 Knowing the Store has a public artifact, but its paper is under review

Official artifact:
https://github.com/ansh348/knowing-the-store

The repository README identifies the associated work as an ICLR 2027 submission under review; the arXiv identifier is still marked pending. It is therefore an available artifact and research comparator, but not a peer-reviewed ICLR paper at this snapshot.

Its artifact describes:
- 148 unique store questions, each applied to three store versions differing in one attribute (444 store-version items);
- value-free summaries containing lifecycle-state counts and attribute names;
- counterfactual lifecycle cases for supersession, retraction and expiry;
- an offline supplementary archive of 1,438 files;
- a one-command reproduction path that does not need model APIs.

The artifact's own reproduction documentation is careful about evidence levels. It says some score files are recomputed, some table fragments are regenerated, some figures are independently re-derived, and remaining printed numbers are checked against ledgers. The README reports 232 checks and zero mismatches for its number gate, while the supplementary README explicitly states that one set-based ledger-accounting layer is weaker than independent recomputation. MACE should preserve that distinction when citing reproducibility.

The paper's central result for MACE is not that a reader can always infer lifecycle state. It reports that on the short attribute lists generated by the tested stores, none of five LLM readers reliably beats a cosine-similarity lookup over attribute names; counts and longer lists affect particular lifecycle distinctions. The backend's declared metadata can limit every reader.

MACE's value-free metamemory summary now exposes exact status and projection-kind counts, an ordered eight-field metadata attribute inventory, matched ledger/index sequence numbers and integrity commitments. This is a design alignment, not yet a benchmark reproduction or a claim of outperforming the study. The next run must compare the official name-list/cosine controls, count-only controls, full metadata and model readers on the exact released questions, then add metadata fault injection without altering official metrics.

### 13.3 COLLABMEM is now a published EMNLP 2026 main-conference benchmark

Official publication:
https://aclanthology.org/2026.emnlp-main.1194/

The paper reports 58 hand-authored scenarios (1,118 core probes), a 174-scenario scaled split (3,359 probes), adversarial/compositional hard sets and six transactive-memory competencies. TMS-GRAPH++ reports the highest non-Oracle average at 69.6% across five splits. The evaluated dimensions include sharing, access, routing, provenance, conflict resolution and retention.

The paper says the authors will release the dataset, evaluation code and reproduced outputs. In this repository search, I did not locate a linked official code/data repository that could yet be pinned. Keep COLLABMEM at “published paper / official artifact not pinned” until that exact artifact is discovered and verified; do not treat the promised release as an available run package.

### 13.4 ServeLearnBench leaderboard must be treated as a one-run snapshot

Official project and leaderboard:
- https://infini-ai-lab.github.io/ServeLearnBench/
- https://infini-ai-lab.github.io/ServeLearnBench/leaderboard.html

The current public design page reports nine streams, 53 environment windows and 7,718 tasks: 4,508 serving tasks and 3,210 held-out test tasks, across three domains and nine scenarios. The leaderboard labels its results snapshot 2026-09-25 and states one run per model/method cell.

Representative leaderboard rows:
- CH / Opus 5: Hidden 73.4, Fully Specified 96.2, API cost $2,449.
- CH / GLM-5.3: Hidden 69.8, Fully Specified 96.7, API cost $1,077.
- Prime / GLM-5.3 Flash: Hidden 61.1, Fully Specified 95.2, API cost $172.
- CH / GPT-5.6 Terra: Hidden 55.8, Fully Specified 94.2, API cost $239.

These are different model/method combinations, not matched head-to-head model comparisons. The benchmark itself reports that inferring a hidden policy can cost 4–101 times as much as following a disclosed one and can degrade behavior on tasks the agent could already solve. Our stable-task regression budget is a proposed MACE admission condition, not an official benchmark metric. The new four-run gate is intended to test that condition on paired task sets, not to claim a ServeLearnBench result.

### 13.5 Industry memory comparisons require two different scoreboards

Useful public comparators now include:

| Source | What it contributes | Limitation to preserve |
|---|---|---|
| AgentOS / agentos-bench | LongMemEval S/M, LOCOMO, cost, latency, CIs, judge false-positive-rate diagnostics and cross-vendor transparency | AgentOS's public S headline uses a reader router calibrated on the same 500 scored questions; its own documentation reports 84.2% with a fixed gpt-4o reader. Per-case result JSON availability has caveats in the public repository |
| agent-memory-bench | A matched retrieval-focused comparison across GoodMem, Letta, Mem0, LangMem and Zep/Graphiti using one reader model | Smaller 419-turn/120-probe harness; independent community benchmark, not an official universal leaderboard |
| COLLABMEM | Multi-agent sharing, access, routing, provenance, conflict resolution and retention | Official release is not pinned in this snapshot |
| TimlMemBench | Governed dialogue-memory lifecycle and use control | Small synthetic bilingual release; official dimensions are diagnostic |
| ServeLearnBench | Continual learning and its cost, including stable-task regression | Public results are one run per model/method cell |

AgentOS's published docs report 85.6% on LongMemEval-S with a per-category reader router, but explicitly disclose that its routing table was calibrated on the same 500 questions; it reports 84.2% with gpt-4o answering every question. Its reported LongMemEval-M score is 70.2%. The docs report $/correct and latency, but their own transparency page flags missing per-case JSON objects in Git LFS for some runs. Do not compare the routed S score with fixed-reader systems as if they were matched.

The community agent-memory-bench repository reports one matched retrieval setup with gpt-4o-mini and top-10 results: GoodMem vendor configuration 57.6% recall, Letta 52.2%, Mem0 50.0%, LangMem 45.7%, and Zep/Graphiti 37.0%. This is useful as a reproducibility-oriented secondary comparison, but its 120 probes and its own evaluation design do not replace official benchmarks.

### 13.6 DeepControl is a serious comparator for adaptive acquisition

Publication:
https://aclanthology.org/2026.findings-emnlp.52/

DeepControl reports improvements over Search-R1 of +9.4 and +8.6 points on Qwen2.5-7B and Qwen2.5-3B respectively across seven benchmarks, using adaptive retrieval continuation and granularity control.

MACE's finite-world adaptive policy solver is not directly comparable to those neural training results. Its initial claim is exactness and equivalence to an exhaustive oracle under a bounded V1 model, not higher end-task accuracy. To compete, MACE needs a matched downstream experiment with identical model, task set, reader, tool budget and resource accounting. Compare fixed top-k, fixed-cost RAG, marginal information-value continuation, DeepControl-style continuation/granularity and MACE's expected-value/robust-policy selectors under the same action space.

### 13.7 Updated priority and claim levels

1. TimlMemBench: official published release available; protocol smoke adapter authored; paper-comparable MACE run not yet executed.
2. Knowing the Store: public offline artifact available; associated paper under review; value-free MACE metamemory summary authored; official reproduction not yet executed by MACE.
3. StreamMemBench: official public artifact available; longitudinal integration and feedback-reuse run remains pending.
4. ServeLearnBench: official public code/leaderboard available; MACE's non-regression gate authored; paired run remains pending.
5. COLLABMEM: peer-reviewed EMNLP 2026 paper available; official evaluator/data package not yet pinned in this audit.
6. DeepControl: peer-reviewed Findings EMNLP 2026 comparator; no matched MACE reproduction yet.
7. Impact Is Not Invalidation: claim-relative research comparator; do not claim paper-comparable reproduction until its official artifact is verifiably available.

For every comparison, label separately: artifact available, protocol inspected, adapter authored, test executed, official scorer executed, independent replay completed, and field validated. A paper, a public dataset, a smoke adapter and a fully reproduced benchmark are different evidence levels.


## 13. Benchmark landscape update — 2026-11-28

This update incorporates official results released after the original 2026-11-26 snapshot. It does not claim that MACE has entered or reproduced any of these leaderboards.

### 13.1 Agent Memory Leaderboard — Cycle 2

Official results were published on 2026-11-19. The page reports 9 separate leaderboards and 289 track entries. Track, division and metric definitions differ; never combine these into one average.

| Track | Open-source leader | Open-source score | Commercial/closed leader | Score |
|---|---|---:|---|---:|
| Textual | Resurface | 61.22 | MemoraX | 69.26 |
| Multimodal | Khipu | 43.26 | MemoraX | 53.68 |
| Coding | InsightMem | 63.33% resolved | MemoraX | 68.00% resolved |

The textual board exposes explicit fact recall, relational/multi-hop reasoning, temporal sequencing, governance, personalization, workflow execution, epistemic safety/privacy and streaming. The multimodal board uses retrieval, relational reasoning, temporal reasoning, continual learning, personalization and evidence governance. The coding board reports resolution rates under relevant/noisy conditions and shows resource/time tiers. Use the official track-specific denominator and rubric. Track subdimensions matter: Resurface reports 76.55 for textual Memory Governance and 68.59 for Epistemic Safety & Privacy, while Khipu reports 31.29 for multimodal Evidence Governance. Those are capability subscores, not the overall score; MACE should publish them beside any overall result. The corresponding commercial textual MemoraX subscores are 78.10 and 81.98, respectively.

Sources:
- https://agentmemoryleaderboard.ai/results/cycle-2/
- https://agentmemories.ai/blog?post=phase2-results-20261119

**MACE target, proposed:** first enter the open-source division and reach at least +3 absolute points over the current open-source leader on each matched track, without worse privacy/governance submetrics. Stretch: approach or exceed the commercial/closed leader under the same official task, model, API, denominator and compliance rules. Do not infer a general industry win from one track.

### 13.2 LongMemEval-V2 — accuracy/latency frontier

The official V2 harness models memory as chronological trajectory insertion followed by \`Query(q) -> bounded context\`, then a fixed reader. It reports answer accuracy and memory-query latency, not a single unconstrained end-to-end agent score.

The current public page reports:
- AgentRunbook-C: 74.9% at 108.3s on Small; 70.1% at 139.9s on Medium.
- AgentRunbook-R: 58.6% at 26.9s on Small; 57.0% at 25.8s on Medium.
- RAG query-to-slice plus notes: 51.0% at 0.2s on Small; 45.9% at 0.3s on Medium.
- The leaderboard uses LAFS Gain, the increase in reachable accuracy over a fixed accuracy/latency reference frontier. The current page highlights AgentRunbook-C v2 Online at +2.823 LAFS Gain, 75.6% accuracy and 130.5s; Perpetual Recall V3 at +1.096, 53.2% and 1.96s; and OPS-Agentic-Memory-v1 at +0.322, 81.4% and 153.6s. These illustrate that higher raw accuracy does not necessarily mean a better accuracy-latency frontier.

Sources:
- https://xiaowu0162.github.io/longmemeval-v2/
- https://github.com/xiaowu0162/LongMemEval-V2/

**MACE target, proposed:** first beat the RAG slice+notes frontier while preserving the evidence-governance sidecar; then target positive LAFS gain, and finally compete with the current +2.823 reference improvement using a valid official submission package. Report Small and Medium separately. Do not use the original LongMemEval result as a substitute for V2.

### 13.3 Same-harness industry reproduction is not optional

Vendor-reported scores are useful leads, not a fair cross-system ranking. For example:
- Redis reports 86.5% task-averaged accuracy on original LongMemEval for its combined ReMi + Instruct strategy on a common GPT-4o backbone; this is a vendor-authored reproduction.
- An OmniMemEval snapshot run by the MemOS team reports LongMemEval-S results including MemOS 89.20, Letta 77.67, Zep/Graphiti 79.80 and Mem0 Cloud 56.00 with a common GPT-4.1-mini answerer and GPT-4o-mini judge; this is a system-maker's benchmark, not a neutral third-party leaderboard.
- A public third-party reproduction on 419 turns and 120 probes, with GPT-4o-mini and top-10 retrieval fixed, reports GoodMem vendor-config 57.6% recall, Letta 52.2%, Mem0 50.0%, LangMem 45.7%, and Zep/Graphiti 37.0%. It also records refusal on adversarial probes, p50 search latency, memory tokens and ingest time.

Sources:
- https://redis.io/resources/agent-memory-benchmark-report/
- https://aiagentmemory.org/articles/agent-memory-leaderboard/
- https://github.com/hamza-dev-tech/agent-memory-bench

The point is not that one of these numbers is the universal truth. It is that model, judge, retrieval depth, configuration, corpus, prompt, abstention rubric and latency accounting can move results dramatically. MACE should publish its own reproducible harness and match these variables before making performance claims.

### 13.4 A benchmark package MACE should submit

For every official benchmark, keep the official score untouched and attach a separate MACE evidence-governance capsule.

**Frozen comparison arms**
1. No-memory / reader-only baseline.
2. Full-context or official reference baseline.
3. BM25 or keyword RAG.
4. Dense-vector RAG.
5. A competitive external memory backend (Mem0, Letta, Zep/Graphiti or the best supported open implementation).
6. The same MACE candidate with evidence gating disabled (diagnostic ablation only).
7. MACE with use-time revalidation and exposure lineage enabled.

The same reader, judge, prompt, model revision, task order, query budget, context budget and resource envelope must be used wherever the official protocol permits. Mark any exception and do not compare it as matched.

**Paired execution**
- Run at least three fixed seeds where the benchmark permits.
- Pair by task and preserve the benchmark's true independent cluster (person, workspace, repository or trajectory family).
- Report paired difference, a cluster-aware confidence interval, missing/timeout/abstention counts and the number of independent clusters.
- Freeze candidate code, memory adapter, prompt, reader, grader, tool contracts and resource budget before unsealing evaluation outcomes.
- Use the calibration/evaluation split firewall introduced in the selective-risk-calibrator child PR before claiming held-out calibration performance.

**Separate official metrics from governance sidecars**
- Official answer/resolution accuracy, latency and token/time tiers.
- Useful evidence recall and forbidden-evidence retrieval.
- Stale, revoked, expired and superseded evidence exposed.
- Cross-scope / unauthorized evidence exposure.
- Unsupported provenance claims.
- Correct abstention and appropriate non-use.
- Evidence-span marginal utility from paired removal and neutral-marker interventions.
- End-to-end cost, including ingestion, indexing, verification, re-observation, storage, retrieval and answer generation.

Hard governance violations must remain veto counts/rates with explicit denominators. Never fold them into a weighted score that a high recall number can compensate for.

### 13.5 Proposed MACE success gates

These are internal targets, not published benchmark requirements or measured MACE results:

| Dimension | Initial qualification gate | Stretch target |
|---|---:|---:|
| AML textual, open-source | Match the 61.22 Resurface reference | +3 points |
| AML multimodal, open-source | Match the 43.26 Khipu reference | +3 points |
| AML coding, open-source | Match 63.33% resolved | +3 points |
| LongMemEval-V2 | Positive LAFS gain over the fixed official reference | Exceed +2.823 under exact official method |
| Stale/revoked/wrong-scope evidence escapes | 0 observed in adversarial qualification suite | Maintain 0 under expanded hidden testing |
| Unauthorized side effects / cross-scope disclosure | 0 observed | 0 under independent red-team evaluation |
| Erasure resurrection | 0 observed through derived-memory and cache paths | 0 under independent lifecycle audit |
| Cost-adjusted utility | Non-inferior official score at lower all-in cost | Better accuracy/latency/cost Pareto frontier |

Do not claim a zero population risk from zero observed test failures. Report confidence bounds and finite-sample limits, including cluster counts.

### 13.6 MACE's actual differentiation hypothesis

Conventional memory systems compete primarily on extraction, retrieval, graph/entity structure, context compression, update mechanisms and answer accuracy. MACE should compete there, but its differentiator is the ability to attach enforceable evidence lineage and authority to a particular decision-use event.

The testable hypothesis is:

> At the same reader and resource budget, MACE can preserve or improve benchmark utility while reducing stale/unauthorized evidence exposure, false provenance, unnecessary personalization and post-erasure resurrection—and can demonstrate the marginal utility of each exposed evidence span.

The research path is to test this hypothesis through matched benchmarks and independent instrumentation. The architecture alone does not establish it.


### 13.8 ServeLearnBench — continual learning under hidden environment change

ServeLearnBench is a separate, highly relevant challenge: agents must infer latent environment rules from service interactions, apply them, revise them as policies change, and measure the cost and regression caused by adaptation. The benchmark spans retail support, banking and sales-pitch generation; the 2026 release reports 53 environment windows and 7,718 tasks across 28 model/harness pairs and 252 learning runs.

The current public board reports examples such as:
- CH·Opus 5: 73.4 Avg. Hidden reward, 96.2 Avg. FS and $2,449 API cost.
- CH·GLM-5.3: 69.8 Avg. Hidden reward, 96.7 Avg. FS and $1,077.
- Prime·GLM-5.3 Flash: 61.1 Avg. Hidden reward, 95.2 Avg. FS and $172.

The benchmark paper identifies three important failure modes for memory/learning systems: continual adaptation remains difficult, adaptation can be costly and harm previously correct behavior, and inadequate exploration can prevent learning hidden policies.

Sources:
- https://infini-ai-lab.github.io/ServeLearnBench/
- https://arxiv.org/abs/2610.07792

**MACE-specific hypothesis:** a governed memory system should not only learn a new environment rule; it should know when the rule has changed, record what evidence supports the revision, distinguish temporary exceptions from persistent policy, and detect when the environment reverts.

Proposed additional benchmark arms:
1. Static memory / no adaptation.
2. Naive append-only memory.
3. Retrieval-only adaptation.
4. Unconstrained continual update.
5. MACE with evidence lineage, use-time revalidation, calibrated sufficiency, and reversible belief/procedure updates.

Keep the official Avg. Hidden, Avg. FS and API cost as separate primary metrics. Add a MACE sidecar for stale-rule use after change, correct revision latency, false update rate, rollback/reversion accuracy, unsupported procedural claims, exploration coverage and evidence acquisition cost.

**Prospective target:** improve hidden reward at matched API cost while preserving or improving Avg. FS, with zero observed unauthorized or stale-policy execution in the adversarial change/revert suite. Do not compare its scores numerically with AML or LongMemEval-V2; it tests a different task distribution and outcome.

The MACE experiment should include explicit state trajectories: policy A → policy B → policy A, partial rollout, conflicting sources, delayed notices, and a deceptive but non-authoritative suggestion. A correct memory system must update, scope, suspend and revert without treating the newest statement as automatically authoritative.


