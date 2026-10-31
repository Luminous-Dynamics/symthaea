# PITB capability vector and industry benchmark plan

**Date:** 2026-10-31  
**Status:** benchmark design and public comparator inventory. No shared-task head-to-head run or transfer gain is claimed.

## Executive decision

PITB should not publish a single composite score. Its research proposition spans several distinct jobs: learning a reusable proof strategy, finding the right dependency route, proving an exact target, preserving downstream semantics, repairing an existing proof blueprint, predicting reliable context support, and doing all of this economically.

Those outputs are related, but a model can improve one while regressing another. Report a **capability vector and a cost/reliability Pareto frontier**, with a separately governed holdout and independent proof/semantic gates.

## Public comparator cohorts

| Cohort / system | Scope and reported result | PITB use |
|---|---|---|
| MathlibQR / Legendre leaderboard | The source benchmark contains up to 1,200 query slots; the currently scored table covers 894 resolved queries. Eleven target declarations absent from the pinned Mathlib revision and blank query slots are treated as unresolved, not misses. Results are fetched from hosted services rather than all reproduced from frozen model binaries. Source: https://www.legendre-leaderboard.com/ | Preserve the exact eligibility ledger and unresolved denominator; archive query responses and service revisions for repeatability. |
| SorryDB_2601 | 1,000 real-world Lean proof obligations. Its January 2026 snapshot displays Aleph Prover at 57.2% Pass@1, Gemini Flash 3 agentic at 30.3% Pass@1, Claude Opus 4.5 at 7.8% Pass@1 / 15.4% Pass@32, and 63.2% combined listed-system coverage. Source: https://sorrydb.org/ | Practical repository-proof cohort. Pin the exact snapshot, eligibility rules, source commits and model versions; do not compare PITB results with these figures unless the same tasks and budgets are reproduced. |
| VeriSoftBench | 500 Lean tasks from 23 real software-verification projects. The best listed result is 41% with curated context versus 35% with raw repository context. Source: https://utopia-group.github.io/VeriSoftBench/ | Test project-specific dependency/context transfer, whole-repository generalization and verifier integration. Use repository-clustered inference. |
| TheoremBench | Nearly 100 classical Lean theorems, provided in plain-main and premised forms; reports theorem-level coverage and token efficiency, and highlights bias toward easy subtheorems and long tactic traces. Source: https://arxiv.org/abs/2606.09450 | Measure progress across all parent-theorem obligations, not just easy subgoals or isolated completed lemmas. |
| Lean Finder V2 | October 22, 2026 paper introduces adversarial training and query rewriting for short/ambiguous mathematical queries. The live Legendre comparison reports 97.3% Recall@100 (n=894) and 89.4% Recall@10 on the resolved MathlibQR queries. Sources: https://arxiv.org/abs/2610.30843 · https://www.legendre-leaderboard.com/ | Current semantic-retrieval frontier. Reproduce exact query set, hosted model revision, query rewriting and evaluation denominator before claiming a head-to-head. |
| LeanGraphSearch | On its 810-query MathlibQR fair subset, Recall@100 is 94.7% versus 82.7% for LeanSearch v2. MathlibMPR complete-route Covered@20 is 42.8% versus 39.1%. Source: https://pluslabnlp.github.io/LeanGraphSearch/ | Strong graph-retrieval and route-completeness baseline. Compare multi-hop support recovery and complete-route coverage, then verify downstream proofs at matched budgets. |
| T² semantic theorem testing | 2,206 tasks from five repositories with about 41 dependent successor theorems per task on average; best reported Testing Accuracy is 38.9%. Source: https://aclanthology.org/2026.acl-industry.150/ | Semantic-preservation gate. Report compilation and successor acceptance separately. |
| BlueprintRepair / BlueprintTrace | 142 controlled Lean blueprint failures; ten typed local operations, target-node protection, and complete accepted/rejected trajectories. Typed operations are cheapest per solved state in the reported matched setup (patching 1.30×, rewriting 2.06× as expensive). Source: https://aclanthology.org/2026.mathnlp-main.3/ | Repair interface baseline. Reproduce its typed vocabulary and compare operation choice, search cost, target preservation and successor regressions. |
| FormalMATH | Research-level formal mathematics. Its published leaderboard reports DeepSeek-Prover-V2-671B at 28.31% Pass@32 on FormalMATH-All in the stated setup. Source: https://github.com/Sphere-AI-Lab/FormalMATH-Bench | Separate high-difficulty research-math proving reference. It is not a substitute for transfer, repair or semantic-preservation scores. |
| Benchmark integrity audit | The ICML 2026 audit reports 4,833 findings, including 398 mechanically certified issues, across five Lean benchmark corpora. Source: https://proceedings.mlr.press/v306/ammanamanchi26a.html | Treat statement fidelity, axiom/sorry audit, mutation controls and contamination as benchmark metrics. |

The numbers above are published context, not a common leaderboard. Task distributions, models, sampling budgets, toolchains, context availability and scoring differ. A figure is a matched baseline only after its task set and execution contract are reproduced.

## Capability vector

Every release should publish these endpoints independently.

| Endpoint | Primary measurement | Required guard |
|---|---|---|
| Strategy transfer | Paired change in exact-target, semantic-gate-accepted success when a typed insight is added to the strongest validation-selected baseline | Same targets, model, budget and checker; family-component clustering; exposure/leakage audit |
| Premise/dependency retrieval | Recall@k, nDCG@k, complete-route Covered@k, ambiguity/intent robustness, and verified proof completion after retrieval | Pinned index/corpus and annotated alternative premise routes |
| Proof generation | Pass@1 and prespecified Pass@k, theorem-level completion, tokens/calls/time/cost per accepted proof | Exact target identity, kernel checking, denominator/eligibility audit |
| Semantic preservation | Dependent-successor/invariant acceptance and false-accept rate under target/specification mutations | A failed semantic gate is not a pass even when the target compiles |
| Typed repair | Verified repairs per task, accepted/rejected operation trajectory completeness, target preservation, unaffected-sibling regressions, cost per accepted repair | Fixed immutable target; operation schema; Lean and successor checks |
| Support explanation | Necessary-declaration prediction, sufficient-route prediction, exact minimal-antichain recovery, selective risk/coverage | Compare against exhaustive finite observed lattice; distinguish direct proof-term occurrence from logical necessity |
| Reliability landscape | Simultaneous exact lower bound for each well-formed context, reliable antichain and context/resource Pareto frontier | Full valid lattice, distinct paired seeds, preregistered familywise inference |
| Systems efficiency | Verified branches per second, live-state reuse, wall time, peak memory, checker time, model calls and spend | Exact proof-state identity and stale-state/replay rejection |
| Evidence trust | False certificate acceptance, provenance completeness, signature/role verification, replay rejection | Independent verifier and trust root not controlled solely by the outcome producer |

Do not turn these into a weighted score unless a real deployment utility function and its weights are independently preregistered. A capability vector prevents strong results in one domain from hiding failures in another.

## Shared baseline arms

### Research-family transfer cohort

Retain the existing nine-arm protocol and make the interpretation explicit:
- no retrieval;
- BM25/lexical retrieval;
- typed syntax/AST retrieval;
- dependency-graph retrieval;
- raw proof retrieval;
- typed insight retrieval;
- insight plus source proof;
- HDC/embedding ranker;
- strongest reproducible premise-retrieval baseline.

Add a separate support-context policy comparison:
- full context;
- Lean MVarId.cleanup;
- direct typed dependency closure;
- random valid context;
- minimal-support antichain search;
- LeanGraphSearch-style multi-hop retrieval;
- HDC/LTC-ranked candidate context;
- adaptive portfolio with abstention.

Keep the model, checker, theorem state, context token budget and call/time/spend caps fixed. The same frozen seed is used at each replicate index across the context lattice, and distinct replicate indices must use distinct seeds. Use the exact one-sided Clopper-Pearson lower bound with Bonferroni familywise correction for the reliability gate; matched context-versus-full outcomes are reported separately using paired analysis.

### Real-world repository cohort

Import two public, separately versioned cohorts where redistribution/license rules allow:
- SorryDB_2601 for open proof obligations.
- VeriSoftBench for software-verification obligations from complete projects.

Add a later SorryDB snapshot and held-out repositories for temporal/generalization testing. Do not mix these task IDs into the 372-family source split. If the exact task bytes or environment cannot be reproduced, report the public result as contextual only.

### Repair cohort

Reproduce BlueprintTrace's accepted/rejected repair trajectories and operation budget. Add:
- support-limited repair (only declared support edges can change);
- graph-constrained repair with verifier feedback;
- typed repair with an adaptive operation portfolio;
- patch and full-rewrite baselines.

Record both outcome and trajectory cost. The benchmark must include false/too-strong statements, missing hypotheses, missing dependencies, redundant dependencies, dead nodes, monolithic nodes and unrepairable cases.

## Qualification targets

Prospective targets, not achieved results:

| Goal | Initial target |
|---|---:|
| Transfer uplift | At least +10 percentage points paired semantic-gate success versus strongest validation-selected baseline, with the component-clustered 95% interval above zero |
| Repository proof engineering | At least +5 percentage points Pass@1 over the strongest reproducible matched baseline, with repository-clustered 95% interval above zero |
| Retrieval | Reproduce and then exceed Lean Finder V2's 97.3% Recall@100 on the same 894 resolved MathlibQR queries; separately exceed LeanGraphSearch's 42.8% MathlibMPR Covered@20 on the same route benchmark and budget |
| Semantic correctness | Exceed T² Testing Accuracy on a common reproduced task set; zero target mutation false accepts in the registered finite suite |
| Repair | Match the typed-edit completion frontier, then achieve at least 10% lower cost per verified repair without a material completion-rate loss |
| Support prediction | Necessary-declaration F1 at least 0.90; sufficient-route accuracy at least 0.95 at prediction coverage at least 0.80; exact minimal-route recovery at least 0.80 |
| Reliability | Simultaneous 95% familywise lower bound at least 0.80 for any context labelled reliable |
| Certificate verification | Zero false accepts in 10,000 generated corrupt/replayed/incomplete/over-budget/target-mutated certificate cases |
| Evidence coverage | No confirmatory held-out claim while any hard cross-family relationship is unresolved or any required independent-review receipt is absent |
| Generalization | Replication on a later source snapshot or held-out external Lean repositories |

A threshold is only meaningful with its denominator, exclusions, confidence procedure, and task distribution attached. For high Pass@k sampling budgets, report calls and tokens rather than pretending Pass@32 and Pass@1 have equal cost.

## Current blockers and stop conditions

1. The 372-family cross-family relationship review is still pending. The generated 744-entry docket is a worklist, not a completed audit. No confirmatory test-split claim is allowed until complete independent review and a real signed coverage receipt exist.
2. The Lean context exporter and adapter are code additions only until the exact pinned Lean 4.34.1 workflow passes and emits a traceable artifact.
3. The target statement digest currently binds the pinned toolchain's pretty-printed expression. It is not a universal canonical hash of Lean Expr; cross-toolchain statement identity needs structured expression serialization or equivalent kernel-level evidence.
4. A valid local receipt digest does not identify a trusted producer. Run/kernel/semantic receipts need verifiable producer identity, role separation and replay controls.
5. Any cohort import must preserve all raw, eligible, excluded, unknown and defective task counts. Missing tasks cannot silently disappear from the denominator.

## Next execution sequence

1. Obtain exact Lean build and context-export evidence at the current commit; do not hand-author a dependency graph.
2. Freeze reviewed fixed/mutable role plans for 2–4 declaration calibration components.
3. Run a real complete support lattice and verify every row, context receipt, seed, kernel result, semantic gate and cost.
4. Reproduce Lean Finder V2's eligible MathlibQR query set and LeanGraphSearch's 810-query fair subset separately; include Lean Finder V2's query-rewriting stage and freeze service/model revisions.
5. Compare full context, cleanup, typed closure, graph retrieval, random-valid contexts and HDC/LTC proposals under identical frozen budgets.
6. Import SorryDB_2601 and VeriSoftBench through license-compliant snapshot manifests; reproduce available baselines before comparing.
7. Complete and independently sign the full family relationship audit; then perform confirmatory family-transfer experiments.
8. Publish per-capability leaderboards and Pareto frontiers, not one composite score.

No SOTA win is claimed by this protocol.


## Updated retrieval frontier (2026-10-31)

The retrieval target is now higher than the earlier LeanGraphSearch-only plan. Lean Finder V2 (paper dated 22 October 2026) uses adversarial training and query rewriting to handle short, ambiguous mathematical queries. The current Legendre leaderboard reports 97.3% Recall@100 and 89.4% Recall@10 over 894 resolved MathlibQR queries. This is not directly comparable to LeanGraphSearch's 94.7% on its 810-query fair subset: denominators and service configurations differ, and the Legendre board queries hosted system endpoints rather than requiring every result to be reproduced from frozen model binaries. Sources: https://arxiv.org/abs/2610.30843 · https://www.legendre-leaderboard.com/ · https://pluslabnlp.github.io/LeanGraphSearch/.

This suggests a further PITB research question: **intent-conditioned graph retrieval**. Improve semantic seed quality with ambiguity-aware query rewriting, then traverse the formal dependency graph to recover entire premise routes. Do not stop at a single target declaration: PITB's differentiator should be (a) ambiguous-query resolution, (b) multi-hop route coverage, (c) reliable proof completion, and (d) cost per semantically verified result.

The evaluation must have separate tracks:
- *single-declaration retrieval*: exact eligible MathlibQR query denominator, including unresolved/absent targets;
- *multi-premise route retrieval*: exact complete-route coverage on MathlibMPR or the research-premise set.

Freeze hosted API responses, service revision/date, query rewriting prompts, index revision, model versions, and query eligibility. An unversioned hosted endpoint is contextual evidence, not a reproducible SOTA baseline.
