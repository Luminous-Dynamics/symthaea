# MACE benchmark parity protocol — 2026-11

**Status:** research protocol and benchmark inventory. No MACE score on these external benchmarks is claimed. Current leaderboard values below are external reference results, not Symthaea measurements.

## Executive position

MACE should compete on two complementary dimensions, never collapse them into one weighted headline:

1. **Memory capability:** answer quality, temporal reasoning, procedure/workflow recall, personalization, collaboration, multimodal recall and coding task success.
2. **Evidence governance:** whether the correct evidence was admissible at the relevant valid/known time, exposed within scope, reconciled against contradictory evidence, retained/forgotten lawfully, and linked to a decision without unauthorized use.

The research opportunity is not to claim a new general memory algorithm. It is to integrate bitemporal, authority-qualified evidence with deterministic set governance, use-time revalidation, exact exposure lineage, selective-risk control and controlled counterfactual evaluation.

## External reference points — snapshot 2026-11-27

| Benchmark | Latest relevant published reference | What it measures | MACE use |
|---|---:|---|---|
| Agent Memory Leaderboard Cycle 2, textual | Open-source Resurface 61.22; commercial MemoraX 69.26 | Fact recall, relational/multi-hop reasoning, temporal sequencing, governance, personalization, workflow, epistemic safety/privacy and streaming | First public multi-axis baseline; target open-source parity, then matched +3 absolute points; commercial leader is stretch |
| Agent Memory Leaderboard Cycle 2, multimodal | Open-source Khipu 43.26; commercial MemoraX 53.68 | Multimodal retrieval, relational/temporal reasoning, continual learning, personalization and evidence governance | Add only when the image/context ingestion and evidence representation adapters are truly implemented |
| Agent Memory Leaderboard Cycle 2, coding | Open-source InsightMem 63.33%; commercial MemoraX 68.00% | Coding-memory success over relevant/noisy feature and bug-fix tasks | High-priority Symthaea repository-memory track; compare identical coding agent/model/harness |
| LongMemEval-V2 | Official benchmark uses 451 questions, five memory abilities, web and enterprise domains, and small/medium tiers; reports accuracy and memory-query latency, summarized as LAFS gain | Long histories and trajectory-derived experience; static/dynamic state, procedures/workflows, gotchas and premise awareness | Optimize the quality–latency Pareto frontier, not accuracy alone |
| AMA-Bench | AMA-Agent reports 57.22% average accuracy and +11.16 points over the strongest baseline in its paper setup | Long-horizon agent/environment interaction memory; causal/objective information | Test whether source lineage and causal memory improve machine-generated trajectory recall under matched setup |
| COLLABMEM | 58 scenarios / 1,118 core probes; expanded 174 scenarios / 3,359 probes; TMS-GRAPH++ 69.6% highest non-Oracle average in paper setup | Sharing, access, role routing, provenance, conflict resolution and retention across agents | Direct test for Mycelix-backed rights-aware evidence sharing; target >69.6% on the same splits and zero unauthorized visibility |
| MemSecBench | 310 cases, 48 contexts, seven lifecycle checkpoints; paper reports malicious memory persisted in 84.2% and full write–execute chain succeeded in 50.3% across its evaluated configurations | Memory poisoning from write through consequence and selective repair | Hard-security gate: zero unauthorized execution and zero resurrection in MACE's declared threat model; retain benign memories during repair |
| TimlMemBench | 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases | Raw-native governed use, lifecycle update, scope, provenance recovery, activation and restraint | Measure behavior from raw ingestion, not just answers over a supplied canonical memory pool |
| Memora / FAMA | FAMA explicitly penalizes reliance on obsolete or invalidated memory | Remembering, reasoning and recommendations under long-horizon updates | Report stale-use penalty alongside ordinary answer accuracy |
| MemGov (AACL 2026) | Deterministic pairwise must-link/cannot-link memory governance with repair and obstruction certificates | Unsafe combinations of individually retrieved evidence | Closest conceptual comparator for MACE set governance; distinguish MACE by bitemporal verified receipts, trust/lifecycle/lineage, purpose/scope and exposure binding |

These numbers are not directly cross-comparable. Different tracks, answerers, model backends, scoring functions and domains are distinct. MACE must publish a matched-run record before claiming parity or superiority.

## Benchmark references

- Agent Memory Leaderboard Cycle 2 official results (published 2026-11-19): https://agentmemoryleaderboard.ai/results/cycle-2/
- LongMemEval-V2 official repository and harness: https://github.com/xiaowu0162/LongMemEval-V2/
- LongMemEval-V2 official leaderboard: https://xiaowu0162.github.io/longmemeval-v2/
- AMA-Bench paper: https://arxiv.org/abs/2602.22769
- COLLABMEM (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.1194/
- MemSecBench paper: https://arxiv.org/abs/2607.27080
- TimlMemBench (Findings EMNLP 2026): https://aclanthology.org/2026.findings-emnlp.507/
- From Recall to Forgetting / FAMA (Findings ACL 2026): https://aclanthology.org/2026.findings-acl.1337/
- MemGov (AACL 2026): https://aclanthology.org/2026.aacl-long.246/
- Fixed-sequence confidence-interval calibration (Stats 2026): https://doi.org/10.3390/stats9050109

## MACE-specific benchmark: evidence decision governance

External leaderboards do not yet fully isolate all of MACE's intended invariants. Add a small, public, reproducible MACE benchmark suite with independent evidence clusters and gold governance outcomes. Each test case should include a frozen raw event stream, signed/verified authority fixtures, valid-time and known-time history, rights/scope policy, dependency graph, candidate evidence representations, a downstream task and an evaluator-only outcome.

### Required adversarial families

1. **Bitemporal replay:** evidence valid at one time but not yet known, later known but not valid, future-recorded, expired, retracted or superseded.
2. **Authority/source drift:** same payload with changed issuer, root, acquisition method, trust policy or lineage manifest.
3. **Joint inconsistency:** two individually admissible items that contradict, supersede, violate privacy together, or must travel as a dependency bundle.
4. **Scope and role:** evidence available to one principal/agent but forbidden to another; purpose and consumer-policy mismatch.
5. **Poisoning and repair:** malicious write, later recall, action attempt, deletion request, selective repair and benign-memory retention.
6. **Exposure fidelity:** retrieved-but-not-exposed, exposed-but-not-cited, cited-but-not-exposed, altered byte span, candidate-action mismatch and request instrumentation failure.
7. **Decision sufficiency:** evidence-set removal/neutral-marker ablation, re-observation versus inference, acquire/retrieve/abstain/act decisions.
8. **Distribution shift:** new source family, language, task, policy, group definition or model revision; calibration must suspend instead of silently transfer.

### Report a scorecard, not a composite

| Axis | Required metrics |
|---|---|
| Answer utility | Accuracy / task success, exact-match and judge agreement, regret, task class breakdown |
| Evidence governance | Invalid evidence use escapes, stale/superseded exposure, scope/privacy leakage, unresolved contradiction, unauthorized actions |
| Exposure lineage | Retrieved→admissible→exposed→cited proportions; missing-link and mismatch rates; span fidelity |
| Set quality | Constraint violations, set coverage, selected evidence count, token cost, obstruction rate, optimality replay agreement |
| Lifecycle security | Poison persistence, write→execute success, selective repair, benign-memory retention, erased-memory resurrection |
| Calibration | Cluster-level any-unsafe upper bound, coverage, threshold, group count, independent cluster count, calibration drift |
| Efficiency | Retrieval expansions, tokens, p50/p95 query and verification latency, end-to-end wall time, memory storage and acquisition cost |
| Robustness | Worst-group / language / source-family results, cluster-bootstrap intervals, missingness and shift strata |

Hard governance gates are reported as counts and rates with denominators and confidence bounds. They are not traded against answer quality.

## Matched experiment ladder

Use one fixed candidate memory pool and one fixed downstream model/answerer. Run these variants on the same clusters:

0. **Top-k baseline:** ordinary retrieval with fixed k and prompt.
1. **Lifecycle baseline:** lifecycle/scope filtering only.
2. **MACE item evidence:** use-time revalidation receipts (#13967).
3. **MACE exposure evidence:** exact request-span and candidate-action links (#13973).
4. **MACE set governance:** must-link/cannot-link selection with obstruction outputs (#14071).
5. **MACE calibrated sufficiency:** calibrated accept/retrieve/acquire/abstain/act policy (#13999).
6. **MACE full:** controlled span ablation, independent randomization/outcome custody and all preceding layers.

Every comparison must hold constant model revision, answerer, judge, prompt template, tools, memory corpus, context/token budget, retrieval budget, seed policy and evaluation split. Save exact code/model/data/API hashes. Publish failed cases, not only successful runs.

### Statistical protocol

- Cluster by independent user/profile, project/repository, task trajectory or source-root as appropriate. Repeated turns and evidence spans inside one cluster are not independent trials.
- Preregister the calibration cluster split, score definition, loss threshold, group tree and threshold grid before evaluation labels are inspected.
- Use the fixed-sequence cluster Clopper–Pearson method when the threshold family is a preregistered nested prefix; use simultaneous Bonferroni when later non-monotone safe regions must also be explored.
- The current selector's 20-candidate exact enumeration cap is an implementation limit, not a benchmark limit. Record pool size and whether exact optimum was returned.
- An unadjusted one-sided 95% zero-unsafe Clopper–Pearson upper bound is about 9.5% at 30 independent clusters and drops below 5% at 59; group multiplicity requires more. Do not advertise a 30-cluster floor as a 5% guarantee.
- Use independent holdout/evaluator custody for confirmatory claims. The exchangeability flag and SHA-256 manifests are not proof of sampling validity or issuer identity.

## Targets — prospective, not achieved

1. **Agent Memory Leaderboard:** first achieve reproducible official submission in each implemented track; then beat the open-source leader by at least 3 absolute points with uncertainty reported; commercial leader is a stretch target. No multimodal entry until the multimodal adapter is actually implemented.
2. **LongMemEval-V2:** move the accuracy–latency frontier outward and achieve positive LAFS gain; report small and medium separately, web and enterprise raw results, gotchas/static/dynamic/procedure breakdowns and query latency.
3. **COLLABMEM:** beat the 69.6% non-Oracle reference under the same harness while achieving zero unauthorized role disclosure and zero invalid provenance acceptance.
4. **MemSecBench:** zero unauthorized write→execute outcomes in the declared threat model, zero post-forget resurrection, and near-perfect selective repair without collateral loss of benign required memory. Report attack resistance and repair as distinct endpoints.
5. **TimlMemBench / FAMA:** beat matched raw-native lifecycle, scope, provenance, activation/restraint and obsolete-memory performance—not only canonical-pool QA.
6. **MACE governance suite:** zero stale/revoked/wrong-scope/cannot-link escapes in exhaustive adversarial fixtures; positive evidence-span utility/regret interval on confirmatory clusters; no worst-group regression; full replay agreement between selected set and independent checker.
7. **Operations:** local use-time revalidation p95 <10 ms (remote authority calls reported separately), reduce unnecessary retrieval/re-observation cost, and improve utility per token without relaxing hard gates.

## What would count as a paradigm shift

- A memory API should return a **governed evidence-set object**, not an unstructured list of snippets.
- A decision request should have a **proof-carrying context manifest**: the exact admissible evidence span, scope, purpose, transformation, valid/known snapshot and consumer-policy digest.
- The system should optimize **minimum sufficient evidence** under constraints, and emit an auditable obstruction instead of guessing when requirements conflict.
- It should know when evidence is insufficient and choose whether to retrieve, ask, acquire, re-observe or abstain using calibrated risk/expected value of information.
- Forgetting should be end-to-end: all derived projections, summaries, caches, procedure memories and future context views must be checked for resurrection.

The differentiator is the full path from source authority to evidence-set governance to actual model exposure to independently measured decision consequence. No single benchmark number proves that path.