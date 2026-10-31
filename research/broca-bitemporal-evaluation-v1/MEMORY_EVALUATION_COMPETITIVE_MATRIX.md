# Broca memory evaluation: 2026 competitive matrix and qualification plan

**Protocol family:** BROCA-MEMORY-EVAL-2026-V1  
**Purpose:** compare memory systems without mixing incompatible metrics, expose failures hidden by aggregate QA, and set measurable targets for proof-grounded long-horizon memory.

## Strategic shift

Long-term memory is no longer one benchmark axis. The 2026 literature separates at least eight capabilities:

1. conversational recall and multi-session reasoning;
2. long-horizon scale from 100K to 10M+ tokens;
3. multi-party belief tracking and speaker attribution;
4. heterogeneous, evolving sources such as conversations, documents and email;
5. forgetting, supersession and avoidance of obsolete facts;
6. experience-to-action transfer in multi-session agent-environment loops;
7. multimodal retention and reasoning;
8. operation-level diagnosis of encoding, retrieval and generation failures.

Broca should not optimize a single leaderboard number. It should produce a capability vector with matched baselines, per-task confidence intervals, proof and lifecycle error rates, and all-in cost.

## Benchmark landscape

| Capability | Benchmark / reference | What it measures | Published evidence and limitations |
|---|---|---|---|
| Conversation memory | LoCoMo | Single-hop, multi-hop, temporal, adversarial conversational recall | APEX-MEM reports 88.88% answer accuracy under its ACL 2026 paper protocol; do not pool with vendor scores or other reader/judge configurations |
| Conversation memory | LongMemEval / LongMemEval-S | Knowledge updates, temporal reasoning, multi-session reasoning, abstention and retrieval | APEX-MEM reports 86.2% LongMemEval accuracy. supermemory reports 97% Recall@20 on LongMemEval-S (500 questions, six categories); Recall@20 is retrieval recall, not answer accuracy |
| Long histories | BEAM | 100 conversations, 2,000 validated questions, histories through 10M tokens, ten memory abilities | LIGHT reports 3.5–12.69 percentage-point improvements over strongest baselines depending on backbone. Past.dev's published 10M run is 85.03% aggregate but 28.17% on its 20-question multi-session slice; this is a vendor-specific diagnostic |
| Long histories, same evaluator | OmniMemEval | BEAM nugget scores with GPT-4.1-mini answerer and GPT-4o-mini judge across reproduced systems | Same-harness 10M snapshot: Hindsight 59.75, MemOS 56.75, Zep/Graphiti 56.11, Cognee 56.02, Letta 52.30, Mem0 43.33. Maintained by MemTensor (also MemOS); disclose this conflict and retain the separate nugget-score metric |
| Multi-party memory | GroupMemBench | Group dynamics, per-speaker belief tracking, asker-conditioned queries, audience adaptation | Strongest evaluated system is 46.0% average; knowledge update 27.1%, term ambiguity 37.7%. Simple BM25 matches or exceeds most agent systems in the paper |
| Heterogeneous evolving memory | RHELM | Conversations plus documents/email aligned to evolving user timelines; 27 memory characteristics and seven inquiry types | Valuable coverage reference for cross-source aggregation and context evolution; no aggregate percentage recorded from the abstract |
| Forgetting and lifecycle | Memora / FAMA | Remembering, reasoning and recommending over weeks/months; penalties for obsolete memory use | ACL Findings 2026 reports frequent stale-memory reuse across six memory agents. FAMA is the right countermetric to raw recall; no abstract-level headline percentage is available |
| Agentic memory | AMA-Bench | Real and synthetic environment trajectories: states, actions, observations, tool outputs; 2,478 QA pairs across six domains | AMA-Agent reports 57.22% average, +11.16 percentage points over the strongest baseline using causality graphs and tool-augmented retrieval. This is not dialogue QA accuracy |
| Multi-session action | MemoryArena | Interdependent task chains where early experience must guide later web navigation, planning, search and formal reasoning | ICML 2026 reports near-saturated LoCoMo agents still perform poorly in this setting; task-action coupling is missing from passive recall benchmarks |
| Memory-grounded tool use | Mem2ActBench | 2,029 sessions and 400 tool-use tasks; selects tools and grounds parameters from earlier interactions | 91.3% of tasks were human-rated as strongly memory-dependent. This is task construction validity, not system success rate |
| Multimodal memory | Mem-Gallery | Multi-session image+text memory; extraction/adaptation, reasoning and knowledge management | ACL 2026 evaluates 12 memory systems and identifies gaps in visual retention, organization and efficiency. Keep answer F1, clue recall and visual token/compute budgets distinct |
| Operation diagnostics | EvalMem / DynaMem-Bench | Three independent examiners attribute encoding, retrieval and generation defects | On default LoCoMo, retrieval defects 22.1%, encoding defects 7.7%, generation defects 6.5%; recall-first probing raises evidence recall 70.2%→95.6%. These are diagnosis/coverage metrics, not final answer accuracy |
| Procedural agent memory | APEX-EM | Reuse of typed procedural/episodic traces for coding and structured tasks | Same-backbone held-out BigCodeBench gain +7.6pp over no memory; 3.3× MemRL's +2.3pp under that shared setup. It is procedural transfer, not conversational memory |

Sources are recorded individually in [industry-baseline-ledger-v1.json](industry-baseline-ledger-v1.json). Vendor self-reports, peer-reviewed papers, leaderboard reproductions and different metric types must remain separate.

## Fair-comparison rules

- Keep each benchmark revision, split, answer model, judge model, prompt, tokenizer, retrieval budget and maximum context fixed across arms.
- Run one shared answerer/judge cohort for the primary head-to-head. Separately reported vendor scores are landscape evidence, not direct ranking evidence.
- Never compare Recall@K, answer accuracy, strict evidence recall-all@K, BEAM nugget score, FAMA, action success and task-construction validity as the same metric.
- Publish per-category results and per-question receipts. Aggregate scores may be included, but must not hide multi-session, knowledge-update, temporal, speaker-attribution, abstention or stale-memory failures.
- Use paired per-question analysis and bootstrap at the independent source/session/conversation/user cluster. Report confidence intervals, not only point estimates.
- Measure total cost: retrieval/write/read/maintenance calls, embedding/extraction, tokens, storage, latency, cache state and cost per correct supported answer or successful action.
- Do not use an LLM judge as the only quality oracle. Preserve deterministic gold checks and blinded human adjudication on a stratified sample.

## Broca's differentiator: answer from a closed evidence world

Broca's hypothesis is not that a richer memory graph alone beats vector search. It is that the system can state exactly what it inspected, what it did not inspect, which source claims are admissible, and when it must abstain.

The current implementation work builds these layers:

1. **Snapshot-bound inventory:** exhaustively enumerate all active evidence within a scope and valid-time coordinate. A response cap yields an explicit incomplete receipt; a prefix cannot be called complete.
2. **One disposition per source:** every inventory item gets exactly one extraction row, including explicit empty results, unsupported sources or parse failures. Missing rows invalidate closed-world completeness.
3. **Speaker-grounded events:** each extracted event carries explicit speaker identity/role; a query requiring user authorship abstains if a relevant event has unknown attribution.
4. **Bitemporal and lifecycle checks:** admissions bind to the exact EvidenceIndex state, source trajectory, source revision, authorization, scope, valid interval and knowledge-time coordinate.
5. **Proof coverage:** claim dependencies require independent source families and, when requested, distinct trusted sessions; one axis cannot substitute for the other.
6. **Action gate:** only a complete, independently re-verifiable result can be used for a count/answer action. Incomplete extraction, unknown speaker or failed proof diversity is an abstention/continue condition.
7. **Operation diagnosis:** classify encoding, retrieval and generation defects independently, and report multi-label defects so a single query can expose several failure stages.

### Important limitation

A complete inventory plus one extraction disposition per source proves that the extractor processed every indexed source; it does **not** prove that the extractor semantically found every event. Event extraction false negatives remain an empirical precision/recall question against annotated gold. Closed-world completeness therefore means source-processing coverage, not omniscient semantic completeness.

## Prospective qualification targets

These are engineering targets, not measured Broca results:

| Track | Target | Guardrail |
|---|---|---|
| BEAM 10M multi-session | ≥70% on the exact official 10M multi-session slice | Report all ten BEAM abilities; compare on a common answerer/judge; no statistically significant regression in other slices |
| LongMemEval-S retrieval | ≥97% Recall@20 as a retrieval target | Report answer accuracy separately; the target is aligned to a published vendor retrieval result, not an answer-accuracy claim |
| GroupMemBench | ≥60% average under matched protocol | Speaker attribution precision/recall and knowledge-update / term-ambiguity slices must be published separately |
| Memora / FAMA | ≥95% FAMA and zero stale-only answers in deterministic conformance tests | FAMA must be computed on the official revision; never optimize raw recall at the expense of obsolete-memory use |
| AMA-Bench | Beat 57.22% by ≥3pp under the exact published backbone and scoring protocol | Measure both causal information retrieval and downstream task utility |
| MemoryArena / Mem2ActBench | ≥5pp improvement over matched no-memory and retrieval-only baselines on action success | Same model/tool environment, task seeds and allowed action budget; report tool-choice and parameter-grounding errors |
| Mem-Gallery | Improve answer F1 and clue Recall@K over matched multimodal baselines | Match visual encoder, caption budget, image tokens and perception model; report hallucinated visual facts |
| Closed-world source coverage | 100% inventory/extraction-row coverage for any answerable count query | Zero missing source rows; any truncation, parse failure or unknown required speaker disables answer permission |
| Proof soundness | 0 false accepted proof certificates in deterministic/adversarial suites | Independent verifier, exact source and claim bindings, negative controls, changed-snapshot tests |
| Efficiency | ≥2× lower total context tokens than full-context baseline at non-inferior answer/action quality | Include write, retrieval, extraction, reranker and maintenance costs; no hidden work moved off-budget |

## Next experiments

1. Replay official BEAM, LongMemEval-S and LoCoMo with a common answerer/judge and stage-wise defect diagnosis.
2. Run GroupMemBench with user/assistant/tool/external speaker roles and ask-specific authorization; stress homonyms, pronouns, role changes and cross-channel aliases.
3. Add RHELM-style documents/email and test multi-source event reconciliation with valid-time/known-time corrections.
4. Evaluate Memora/FAMA counterfactuals: superseded, retracted, corrected, forgotten and deliberately misleading historical facts.
5. Run AMA-Bench, MemoryArena and Mem2ActBench with action traces. Measure whether memory changes later tool choice, arguments and final task success.
6. Use Mem-Gallery as a separate multimodal track, never mix image evidence with text-only scores.
7. Use operation-level diagnosis to route engineering effort: repair encoding when facts are absent, retrieval when present facts are missed/late/noisy, and reasoning when oracle evidence is available but unused.

## Current artefacts

- [Cross-session proof coverage](CROSS_SESSION_PROOF_COVERAGE.md)
- [Closed-world event enumeration](CLOSED_WORLD_EVENT_ENUMERATION.md)
- [Operation-level diagnostics](MEMORY_OPERATION_DIAGNOSTICS.md)
- [Retrieval ablation protocol](RETRIEVAL_ABLATION_PROTOCOL.md)
- [Structured evidence-grade industry ledger](industry-baseline-ledger-v1.json)

## External references

- APEX-MEM, ACL 2026: https://aclanthology.org/2026.acl-long.749/
- GroupMemBench, Microsoft Research / May 2026: https://www.microsoft.com/en-us/research/publication/groupmembench-benchmarking-llm-agent-memory-in-multi-party-conversations/
- RHELM, May 2026: https://arxiv.org/abs/2605.31086
- Memora and FAMA, Findings ACL 2026: https://aclanthology.org/2026.findings-acl.1337/
- AMA-Bench, ICML 2026: https://proceedings.mlr.press/v306/zhao26bs.html
- MemoryArena, ICML 2026: https://proceedings.mlr.press/v306/he26am.html
- Mem2ActBench, ACL 2026: https://aclanthology.org/2026.acl-long.370/
- Mem-Gallery, ACL 2026: https://aclanthology.org/2026.acl-long.1892/
- EvalMem, Findings EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.1134/
- APEX-EM, Findings EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.1996/
- OmniMemEval same-harness snapshot: https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/results.md

