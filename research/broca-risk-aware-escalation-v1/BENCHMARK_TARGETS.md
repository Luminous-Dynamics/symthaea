# Broca external comparison plan (checked 2026-10-28)

Do not reduce memory to one leaderboard percentage. Separate evidence retrieval, answer/task success, action execution, governance integrity, cost and latency. Preserve each benchmark's official metric; don't merge incompatible tracks.

## Priority 0: LongMemEval-V2

Official repository: https://github.com/xiaowu0162/LongMemEval-V2

The release describes 451 questions across web and enterprise, up to 500 trajectories and 115M tokens in large haystacks, small/medium tracks, and tests of static state recall, dynamic state tracking, workflow knowledge, environment gotchas and premise awareness. It scores answer accuracy alongside memory-query latency and maintains an accuracy/latency frontier.

Minimum matched arms: `no_retrieval`, `rag_query_to_slice`, and Broca. Add `rag_query_to_slice_notes`, AgentRunbook-R, vanilla Codex and AgentRunbook-C where feasible. Pin upstream code, dataset and harness digests, model revisions, prompts, token limits, tier, per-domain outputs and official aggregate. Report the official frontier metric plus paired per-question outcomes, lifecycle costs and p50/p95 latency.

**Current-upstream baseline caveat (checked 2026-10-28):** the official repository also has an `agentrunbook_c_v2` implementation and runner, but `leaderboard/compute_lafs.py` still defines the released fixed reference frontier using the historical AgentRunbook-C point. For a defensible comparison, pin one upstream commit and (a) compute the official LAFS gain against that commit's unchanged fixed frontier, and (b) report AgentRunbook-C V2 as a separately named measured arm / supplementary operating point. Do not silently replace the reference frontier or label C-V2 scores as the official historical C result. Keep any later official leaderboard revision as a new benchmark protocol version, not a retroactive substitution.

The existing receipt gate PR #8653 validates artifact completeness and digest consistency only. It is not the adapter, scorer or benchmark result.

## Priority 1: Mem2ActBench

Paper: https://aclanthology.org/2026.acl-long.370/
Code: https://github.com/Cantaloupe-M/Mem2ActBench

Measures whether agents apply historical context to tool selection and argument binding. The paper describes 400 tool-use tasks derived from 2,029 sessions and reports that 91.3% were strongly memory-dependent under human evaluation. Measure exact tool choice, arguments, environment success and harmful false-memory-induced actions—not retrieval alone.

## Priority 1: TimlMemBench and Memora/FAMA

TimlMemBench: https://aclanthology.org/2026.findings-emnlp.507/
Memora/FAMA: https://aclanthology.org/2026.findings-acl.1337/

These address governed use of long-term memory: revocation, supersession, scope, relevance/activation and obsolete facts. TimlMemBench describes bilingual reviewed examples and both canonical-pool and raw-native evaluation. Report revoked-fact reuse, collateral forgetting, source-lineage completeness, cross-scope disclosure and correction propagation as independent integrity outcomes.

## Priority 1: AgentAbstain; Sieve and Sage

AgentAbstain: https://agentabstain.github.io/
Sieve and Sage: https://aclanthology.org/2026.findings-emnlp.2356/

These adjacent directions test when an agent should abstain or recognize evidence absence versus being distracted by contradictory evidence. The proposed Broca contribution is not “invent abstention”; it is to evaluate **sequential answer/escalate/abstain selection** with risk/coverage/cost frontiers plus memory provenance and update governance.

## Priority 1: Evolving the memory pipeline (MemPro)

Paper: https://aclanthology.org/2026.emnlp-main.1364/  
Code: https://github.com/wanghai673/MemPro

MemPro's core direction is to treat the **whole memory pipeline as a versioned, evolvable program** instead of assuming that one fixed retrieve/summarize/consolidate recipe is optimal. Each candidate pipeline should be treated as an independently versioned system and evaluated under a fixed development budget; it must not be tuned on the final held-out set.

Broca experiment:
- Compare fixed raw-slice RAG, fixed BM25, fixed dense retrieval, current Broca policy, and an evolved-pipeline controller under the same reader, context budget, compute budget and seed/data split.
- Store every pipeline version as an immutable manifest: modules, parameters, source/model digests, prompt and data digests, resource budget, parent version, rationale and evaluation receipt.
- Optimize a declared objective such as held-out task success subject to cost and governance constraints. Keep the entire candidate tree; do not report only the winning run.
- Report gain per iteration / per unit of optimization compute, outer-loop overfitting, cost-quality Pareto frontier, stability across seeds, and post-shift regression.
- Include negative controls: random edits, shuffled feedback, held-out labels injected into optimizer metadata, and repeated tuning on validation until it overfits.

This is a **system-level optimization** lane, not a new retrieval metric. LongMemEval, LoCoMo, HotpotQA and NarrativeQA are not interchangeable; evaluate each with its own official protocol where compatible. Paper claims remain paper-reported until matched runs are completed.

## Priority 1: Compile evidence into guarded reusable skills (MSCE)

Paper: https://aclanthology.org/2026.findings-emnlp.1637/

The recent Memory-to-Skill Co-Evolution line argues that repeatedly useful experience can become a callable skill, but a memory-derived skill must preserve its evidence, applicability boundary, verification requirements and reliability. This is distinct from storing a text summary.

Broca experiment:
- Actions: retain raw episode, summarize, propose skill, activate skill, revise skill, quarantine, or retire.
- Every promoted skill must carry source evidence IDs/digests, supported task families, explicit preconditions, prohibited contexts, a verification check and rollback lineage.
- Test both skill reuse within its source environment and transfer to a new environment; separately report false skill activation, unsupported generalization, tool/argument correctness, execution success, and safe fallback.
- Compare always-retrieve, summary-only, fixed skill templates, evidence-grounded skill promotion, and an oracle that is strictly diagnostic.
- A skill may be promoted only on development data. Evaluate transfer on a held-out environment and prevent a generated skill from laundering an unverified or revoked claim into executable policy.

The contribution worth testing is not “skills are novel”; it is **auditable promotion with constrained transfer and rollback**.

## Priority 1: Implicit personalized safety (PerMemSafe)

Paper: https://aclanthology.org/2026.findings-acl.320/

Personalization can improve helpfulness while also making a system more likely to act on unsafe inferred preferences or to overgeneralize from past user behavior. A memory policy should therefore be evaluated for helpfulness and safety independently.

Broca experiment:
- Separate explicit, user-confirmed preferences from implicit patterns inferred from behavior.
- Test conflict cases: old preference vs current request, harmful inferred preference, sensitive attribute inference, context/scope shift, user correction, revocation, and uncertainty about identity or authorization.
- Include no-personalization, explicit-only memory, unrestricted personalization, evidence-gated personalization and a conservative safety memory baseline.
- Report safety success, helpfulness, unnecessary refusal, unsafe compliance, inferred-preference calibration, consent/revocation latency and cross-context leakage separately.
- Any claimed safety advantage must use the benchmark's own evaluation protocol and matched model/prompt settings. A paper-reported gain for SentinelMem or another method is not a Broca result and must not be merged into LongMemEval accuracy.

## The unifying research hypothesis

These directions suggest a stronger thesis than "better retrieval": **memory should be a governed, versioned decision system that can evolve without losing evidence, safety or reproducibility**.

Model it as four coupled but separately testable decisions:

1. **Acquire:** what source evidence is worth retaining?
2. **Retrieve / act:** what evidence should influence this decision, and at what cost?
3. **Compile / evolve:** when is evidence strong enough to become a reusable summary, procedure or skill?
4. **Govern / revise:** what must be scoped, corrected, revoked, quarantined or forgotten?

Do not optimize these all at once in an opaque reward. First qualify each subsystem with a simple baseline and contract tests; then evaluate the joint system against frozen baselines. The joint policy should be allowed to lose to simpler retrieval. Complexity is justified only by measured held-out task, cost or safety gains.

## Priority 1: Stability–plasticity diagnostics (MemProbe)

Paper: https://aclanthology.org/2026.emnlp-main.2473/

MemProbe isolates a common hidden trade-off in memory systems: whether an update corrects a stale fact or overwrites a still-valid one. It introduces diagnostic paradigms for interference, misinformation, consolidation strength, and reconsolidation windows, with behavioral profiles rather than only aggregate accuracy.

Broca experiment:
- Use a separate memory-maintenance suite for update, preservation, source attribution, and temporal ordering.
- Cross correction authority and recency: an old fact can be authoritative, while a newer observation can be weak or mistaken.
- Measure correct update rate, destructive overwrite rate, misinformation susceptibility, source-attribution accuracy, correction propagation latency, and cross-session generalization.
- Evaluate a grid of update strength and consolidation frequency. Plot accuracy and destructive forgetting against write cost and retrieval cost.
- Keep the full-context and exact-state controller baselines; don't assume a sophisticated consolidation algorithm is better.

MemProbe should be treated as a diagnostic track, not merged into LongMemEval's end-to-end score.

## Priority 1: Epistemic memory governance (HELM)

Paper: https://aclanthology.org/2026.emnlp-main.721/

HELM's direction is closely related to the evidence ABI here: memory records carry provenance and epistemic status, and retrieval is filtered/reranked using status, freshness, consistency, recency and provenance. The paper also argues that governance-only ablations account for a large share of reliability improvement.

Broca experiment:
- Compare BM25 alone, BM25 plus source/validity gating, and the full sequential answer/escalate/abstain policy under the same corpus and reader.
- Ablate freshness, contradiction handling, provenance checks and calibration separately.
- Include a governance-only arm so any quality gain from retrieval ranking isn't misattributed to policy checks.
- Track invalid-evidence exposure, stale-use events, epistemic error rate, retrieved context tokens and end-to-end success on held-out tasks.

This is the clearest empirical bridge from the current lexical baseline to a later HDC/hybrid architecture: first show the value of evidence governance with retrieval held constant.

## Priority 1: Evidence-gated skill lifecycle (SkillsVote)

Paper: https://aclanthology.org/2026.emnlp-main.281/  
Code: https://github.com/MemTensor/skills-vote

SkillsVote targets collection, recommendation, attribution and evolution of reusable skills, and tests agentic coding tasks such as Terminal-Bench 2.0 and SWE-Bench Pro. It adds another useful baseline for the memory-to-skill part of Broca: successful execution alone is not sufficient to justify a general reusable skill.

Broca experiment:
- Require evidence-backed preconditions, environment/applicability bounds, verification tests, observed reuse count and rollback lineage for each promoted skill.
- Attribute outcomes among skill-guided steps, free exploration, tool/environment changes and task result.
- Measure false promotion, harmful skill activation, cross-environment transfer, regression after skill updates, and the cost of discovering/verifying a skill.
- Evaluate on coding/tool-use benchmarks separately from LongMemEval. LongMemEval can establish recall; it does not establish safe execution or coding-agent improvement.

## Priority 2: LongMemEval-V1 and LoCoMo

Original LongMemEval: https://github.com/xiaowu0162/LongMemEval
LoCoMo: https://aclanthology.org/2024.acl-long.747/

Use for long conversational memory and temporal reasoning, but keep V1 and V2 as separate benchmark tracks. Never compare raw percentages from incompatible task/model/evaluator setups.

## Internal research gates (not industry standards)

1. **Task value:** at least +3 percentage points paired task success at a matched cost budget, with a 95% grouped-bootstrap interval excluding zero.
2. **Efficiency:** alternatively, at least 25% lower measured total lifecycle cost while remaining within a 1 pp non-inferiority margin.
3. **Risk:** publish risk/coverage/calibration curves; a single IID calibration split is not a guarantee under shift.
4. **Integrity:** zero observed stale reuse, unauthorized scope exposure and deleted-memory resurrection in qualifying tests; expand to 10,000+ adversarial transitions across seeds before broad empirical claims.
5. **Reproducibility:** pinned dataset/source/model/prompt/evaluator, paired per-case outputs, digest-bound receipt, negative controls and explicit PASS/FAIL/NOT RUN/INVALID/SKIPPED statuses.

## Next experiments

1. Finish and qualify the source-bound text adapter: run Rust/Python tests, verify the pinned harness installer on a clean checkout, and preserve all failures and receipts.
2. Run the LongMemEval-V2 small tier on both web and enterprise with matched `no_retrieval`, `rag_query_to_slice`, BM25, and separately named AgentRunbook-C V2 arms. Use the official evaluator and unchanged reference frontier.
3. Add an evolutionary outer-loop experiment only after the fixed baselines are stable. Freeze candidate-generation budget, manifest format, development feedback and held-out suite before optimization.
4. Add Mem2ActBench for action grounding; TimlMemBench/FAMA, MemProbe and HELM-style governance ablations for state maintenance; PerMemSafe for personalized safety; and MSCE/SkillsVote-style evidence-gated skill promotion in a separate skill-transfer lane. Keep all metric families separate.
5. Publish quality–latency–lifecycle-cost Pareto curves and a version tree of tried pipelines. If conventional raw-slice RAG wins at equal cost/quality, simplify Broca. If a complex winner is fragile across seeds or shifts, do not promote it.
