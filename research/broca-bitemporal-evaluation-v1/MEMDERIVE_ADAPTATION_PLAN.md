# MemDerive adaptation plan for Broca

Status: benchmark integration plan. The cited external numbers are published results, not Broca measurements.

## Why this changes the research priority

MemDerive (October 18, 2026) defines an agent-memory problem that conventional retrieval benchmarks do not expose: **derived state over evolving data**. A derived state is not stated by any one event; it is computed from multiple records and versioned rules. A new event can change that answer implicitly, without directly mentioning the derived state.

The benchmark contains 10 scenarios, 4,692 events, and 1,503 queries. It records exact Datalog-derived answers, intermediate-state DAGs, and minimal supporting-event witnesses. Difficulty is controlled along three independent axes: access to support, evolution of facts/rules, and composition depth.

Published findings:

| Measurement | Published result |
|---|---:|
| Four memory systems, derived-state exact accuracy | 30.41–48.30% |
| BM25 retrieval baseline, derived-state accuracy | 58.28% |
| Whole-history answer-model reference | 79.84% |
| Supporting-events answer-model reference | 96.01% |
| Unchanged-answer preservation | 91.03–92.86% |
| Explicit-change refresh | 82.4% |
| Implicit-change refresh | 41.1% |

The gap between the full-history and supporting-events references is an important diagnostic: systems must gather enough of the right premises, not merely reason over a large prompt. The 41.1% implicit-change refresh result makes dependency-aware invalidation and rule-version tracking central capabilities.

Primary source: [MemDerive, arXiv:2610.25373](https://arxiv.org/abs/2610.25373). Dataset and evaluator links are listed in the paper.

## Paradigm shift: from retrieval-first to proof-carrying state

The research direction should not be “retrieve more memories.” It should be:

1. Convert accepted observations into versioned facts with source, valid time, transaction time, authority, and lifecycle.
2. Store rules as versioned, first-class objects—not just prose embedded in reflections.
3. Build derived states with explicit dependency edges and support witnesses.
4. Plan a query against its required output state, recursively gather missing premises, and return a derivation trace with the answer.
5. Maintain affected derived states incrementally when facts or rules change.
6. Gate every premise by authority, scope, time, revocation, and role authorization before it reaches the answerer.
7. Calibrate the influence of each admitted memory: ignore irrelevant atoms, use bounded support where appropriate, and let controlling evidence constrain the answer.

This makes memory a small, queryable knowledge-maintenance system with a language interface, rather than a similarity search bolted onto a prompt.

## What the current Broca line covers—and does not cover

- `memory_governance.rs`: policy-pinned source authority, bitemporal query cutoffs, source-scoped assertions/retractions, and explicit conflict handling.
- `memory_attestation.rs`: Ed25519 source/policy signatures and trusted-ingress receipts; the new role-scope envelope adds separate signed grants, mandatory expiry, and a caller-supplied anti-rollback floor.
- `memory_evidence.rs`: evidence payload/source digests, lifecycle filtering, BM25, and role-authorized context materialization.
- `memory_derivation.rs` / #8998: derived-fact premise dependencies and invalidation.
- #9074: alternative proof supports with AND-premises and OR-justifications, interval validity, and a deterministic oracle fixture.
- #9107: a point-query incremental support session; it is not yet a full arbitrary-time incremental Datalog engine.
- #9076: memory-use calibration cases, including multi-agent role, provenance, conflict, and retention probes.

Important scope boundary: the current support engine is not a full Datalog implementation. It does not yet establish parity with MemDerive's stratified negation, aggregation, completeness/closed-world semantics, dynamic rule declarations, or full derivation-DAG diagnostic protocol. Those are explicit research gaps, not capabilities to infer from the existing Boolean support prototype.

## Benchmark portfolio

| Track | What to measure | External reference | Broca target (hypothesis) |
|---|---|---|---|
| Derived-state answer | Exact answer accuracy on the official MemDerive history/query split | BM25 58.28%; whole history 79.84%; supporting events 96.01% | At least 70% overall and no workload axis below BM25 |
| Implicit maintenance | Correctly refresh answers after updates that never mention the derived state | 41.1% published | At least 90% |
| Explicit maintenance | Refresh after an explicit fact revision | 82.4% published | At least 95% |
| Stability | Preserve answers where no supporting fact/rule changed | 91.03–92.86% published | At least 98% |
| Proof provenance | Precision/recall/F1 of cited supporting events and derivation witnesses | MemDerive witness gold | F1 at least 0.90 |
| Dependency correctness | Incremental result vs independent full fixed-point oracle | Existing Broca differential oracles | 100% oracle agreement on generated finite cases |
| Incremental work | Recomputed propositions/rules vs full rebuild under sparse updates | Full rebuild is the control | At least 5x less work on sparse-update workloads at equal outputs |
| Retrieval authorization | Unauthorized, expired, revoked, or out-of-scope payloads materialized into a model context | COLLABMEM access/provenance dimensions | Zero unauthorized materializations in the adversarial suite |
| Memory-use calibration | Ignore / bounded / controlling influence, over-use and under-use | MemCalib | Report separate over-use and under-use; no aggregate score may hide either |
| Long-horizon task memory | History recall, workflow gotchas, query latency | LongMemEval-V2 | Report accuracy and latency together; preserve a matched-model baseline |
| Memory evolution safety | Repeated poisoning, noisy tool outputs, biased feedback | MemEvoBench | Zero critical privilege escalation; quantify drift and recovery |

The numerical targets are research gates, not forecasts. They should be frozen before a held-out run, and revised only with a versioned protocol change.

## How to run a fair comparison

### Tier 0: reproduce benchmark references

Use the official MemDerive data and evaluation scripts; pin dataset revision, code revision, model identifier, prompt, tokenizer, and all inference parameters. First reproduce the published BM25 and whole-history references as controls. Do not compare a Broca component score to an end-to-end memory-system score.

### Tier 1: matched ablations

Run these systems with the same answer model, context budget, decoding, and compute budget:

- BM25-only;
- existing Broca HDC candidate retrieval + evidence resolver;
- retrieval plus query-time premise gathering;
- retrieval plus versioned rules and dependency graph;
- full incremental support maintenance;
- oracle-support upper bound.

For each, report exact answer accuracy, abstention/unknown accuracy, citation precision/recall, and all latency/token metrics.

### Tier 2: update and diagnosis tests

For every tracked query, inject explicit fact changes, implicit premise changes, rule changes, support retractions, alternative-proof removal, and irrelevant distractors. Probe intermediate nodes of the gold derivation DAG. Record the first incorrect node, whether a missing premise was stored but ungathered, and whether repairing that node repairs the final answer.

Compare incremental results against a separately implemented full fixed-point oracle. Do not reuse the production dependency algorithm inside the oracle.

### Tier 3: adversarial collaboration

Run COLLABMEM-style matched memory-only tests for sharing, access, routing, provenance, conflict resolution, and retention. Add cross-role poisoning, role confusion, expired grants, policy rollback, and provenance handoff. Report any unauthorized materialization as a critical failure even when final answer text happens not to reveal the secret.

### Tier 4: consolidation and calibration

Run MemCalib-style Ignore / Bound / Control atom influence tests separately from retrieval accuracy. Run MemEvoBench-style repeated injections across multiple turns. Measure both short-term answer quality and long-term policy/knowledge drift. An intrinsic consolidation score must not substitute for downstream transfer: *The Epistemics of Agent Memory* reports that its consolidation-quality score did not predict real transfer accuracy in its two-benchmark sample.

## External benchmark context

- **MemDerive**: [paper](https://arxiv.org/abs/2610.25373). Exact derived-state answer, proof-DAG, witness, and maintenance evaluation.
- **COLLABMEM**: [EMNLP 2026 paper](https://aclanthology.org/2026.emnlp-main.1194/). Its 174-scenario scaled set has 3,359 probes; TMS-GRAPH++ reports a 69.6% highest non-oracle average across five splits. This score is not directly comparable with MemDerive or AML.
- **MemCalib**: [arXiv:2609.24259](https://arxiv.org/abs/2609.24259). Separates correct memory influence from over-use and under-use; its experiments include Qwen3, Ministral, and Qwen3.5 model families.
- **MemEvoBench**: [arXiv:2604.15774](https://arxiv.org/abs/2604.15774). Evaluates long-horizon memory corruption from adversarial injection, noisy tools, and biased feedback across seven domains and 36 risk types.
- **The Epistemics of Agent Memory**: [arXiv:2609.33013](https://arxiv.org/abs/2609.33013). A warning against treating intrinsic consolidation quality as a proxy for task transfer.
- **Agent Memory Leaderboard (AML)**: [official leaderboard and Cycle 2 guide](https://agentmemoryleaderboard.ai/home). Cycle 1's published Textual board reports overall scores of 58.02 (MemoraX), 45.89 (MemOS), 42.61 (Cognee), and 41.40 (Mem0); these are AML composite scores and must not be compared numerically with MemDerive accuracy or COLLABMEM's split average. Cycle 2 covers Textual, Coding, and Multimodal tracks. Materials close October 31, 2026 at 23:59 UTC+8 (17:59 SAST); evaluation closes November 4 at 23:59 UTC+8 and official results are expected mid-November. AML requires participant-hosted Add/Search endpoints, an issued evaluation key, and a passing Smoke before Full evaluation; repository-only and Docker-only submissions are not accepted. The current Broca LongMemEval worker is a JSONL CLI, not this hosted API, so an AML entry is not ready merely because the repository code exists.

## Success criteria for a claim of SOTA

A Broca result should be described as better than a baseline only when it:

- beats that baseline on the same official split, answer model, context budget, and evaluator;
- reports confidence intervals and per-axis results, not just the aggregate;
- preserves or improves latency, token use, write cost, and memory footprint;
- includes negative controls and independent oracle verification;
- publishes exact dataset/code/model/prompt hashes and all failures;
- passes the adversarial authorization and stale-state tests independently of answer accuracy.

Until then, the defensible claim is architectural progress and authored verification coverage—not a benchmark win.
