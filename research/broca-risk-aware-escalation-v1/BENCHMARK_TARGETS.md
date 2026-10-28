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

1. Implement the risk selector and telemetry in the actual Rust-backed MemoryBridge; keep Python only as a thin official-harness adapter.
2. Run LongMemEval-V2 small tier across both domains with the three matched arms and official evaluator.
3. Replace synthetic features/outcomes with observed retrieval evidence and outcome traces under the same answer backbone.
4. Add Mem2ActBench for action grounding and TimlMemBench/FAMA for governance.
5. Publish the full Pareto frontier. If conventional raw-slice RAG wins at equal cost/quality, simplify Broca accordingly.
