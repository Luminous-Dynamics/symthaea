# VLEB industry comparison and next benchmark frontier — 2026-10-28

## Decision

Do not try to win only one public coding-agent leaderboard. VLEB should measure four separable outcomes:

1. **Task solving:** did a locked engineering task pass the independent behavioral contract?
2. **Assurance:** did the verifier detect wrong-but-plausible patches, incomplete evidence, and unsafe changes?
3. **Legacy fidelity:** did the patch or migration preserve data layout, numeric behavior, ordering, transaction effects, and restart/replay behavior?
4. **Economics:** what was the fully loaded cost and elapsed time per independently verified pass?

Keep model, agent scaffold, tools, budget, runtime image and task set attached to every reported result. The research literature now gives a concrete warning against reading tiny leaderboard gaps as robust ranking: an audit of 254 SWE-bench submissions found highly shared successes, no statistically significant pairwise differences between adjacent top-30 Verified entries under its exact McNemar procedure, and within-model scaffold ranges that were larger than the reported top-30 spread. This is an observational audit, not proof that all systems are equivalent. [Source](https://arxiv.org/abs/2609.17394).

## Relevant benchmarks: what to borrow and what not to claim

| External work | Reported scope/result | What VLEB should adopt | Comparability boundary |
|---|---|---|---|
| [Legacy-Bench](https://fireworks.ai/specialized-intelligence-index/benchmarks/code-generation/legacy-bench) | Listing updated 2026-10-22 reports 62.0% avg@3 for Claude Opus 5.5 and Muse Spark 1.3; DeepSeek V4.1 Flash 57.7% at $0.14/task | Full authorized cohort, fixed attempt policy, cost/time alongside score, language-family breakdown | Its task distribution and harness differ from VLEB. avg@3 is not pass@1. A 10-sample public view is not a full benchmark run. |
| [SWE-bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2) | Updated 2026-09-22: 642 tasks/11 repositories; 89 invalid tasks removed; 69 contradictory prompts corrected; agent web tools disabled; every diff re-graded on a pristine image; every task must pass reference and fail empty-patch controls | Two-sided oracle admission; offline/allowlisted agent environment; clean checkout re-grading; do not trust candidate-side test caches or mutable build state | Its scores are not numerically comparable with Legacy-Bench or VLEB. Adopt its validity controls, not its headline percentage. |
| [LegacyCodeBench](https://kalmantic.github.io/legacycodebench/) | Benchmark authors report 200 real-world COBOL programs across four complexity tiers, claim-based documentation checks validated against execution of the original program, and 86–92% score bands for evaluated model classes | Turn explanations into atomic claims with source spans and executable witnesses; check claims against original behavior instead of relying on LLM-as-judge alone | The reported results are from that benchmark's own dataset and evaluation. Treat as a research comparator, not a direct VLEB baseline or independently verified industry consensus. |
| [IBM Research: enterprise COBOL-to-Java translation, ICSE 2026](https://research.ibm.com/publications/enterprise-scale-cobol-to-java-translation-llms-augmented-with-program-analysis) | Authors describe 20 GenAPP programs plus 10 proprietary customer applications; report median structural-quality score above 80% and functional score above 75%, using an LLM-as-judge framework | Combine program analysis with LLM generation; infer data/control-flow metadata first; compare functional behavior separately from structural/idiomatic quality | Different application sample, proprietary access, pipeline and judge. The reported scores should not be compared directly to hidden-test pass rates. |
| [Agent Security League](https://www.endorlabs.com/research/ai-code-security-benchmark) | Its published results distinguish functional correctness from security correctness | Count a task as jointly successful only if functional tests and security tests both pass; disclose critical/high escape rates | Separate benchmark and scoring process; not a VLEB security score. |
| [DualGauge-Bench](https://aclanthology.org/2026.findings-emnlp.1335/) and [Last Secure Code Benchmark](https://arxiv.org/abs/2610.14766) | Paired functional/security tasks report a large gap between functional and joint secure success in their respective setups | Keep functional-only success, joint functional-and-secure success, and security regression/escape rates distinct | Separate task sets, languages, scoring rules and maturity; use as motivation, not direct percent-to-percent comparisons. |

## Proposed scorecard and gates

### Primary headline

Use the existing **Assurance-Adjusted Semantic Success Rate (AASSR)** definition:

**AASSR = independently verified passes / all tasks assigned to the locked confirmatory cohort.**

A pass requires the behavior contract, all required tests, all declared differential observables, the pre-registered task-level mutation gate, artifact/checkout/runtime identity, trusted run attestation, and independent receipt audit to pass. Failed, unresolved and abstained tasks remain non-passes; report each category separately. No weighted composite should hide security or correctness failures.

### Targets (all proposals, none measured yet)

- **Task admission:** 100% of tasks admitted to scored evaluation have independently reviewed requirements, a baseline/empty-patch failure control, a reference-solution pass control, and pinned evidence hashes.
- **Engineering score:** first establish a matched baseline. Then target at least 65% avg@3 on the full authorized Legacy-Bench under its exact protocol; keep pass@1 and avg@3 separate.
- **Cost frontier:** target at least 60% avg@3 for no more than $0.25/task as a separate quality/cost goal. Do not claim success using a cheaper partial benchmark.
- **Oracle resistance:** initially require 0 false accepts in the locked pilot negative-control set, report binomial confidence bounds, and scale the number of independent controls before broader reliability claims.
- **Mutation resistance:** target at least 90% killed among reviewed, non-equivalent, compilable mutants; publish by defect family and retain compile failures/equivalent mutants/unresolved mutants outside the kill numerator.
- **Cost per verified fix:** target at least 20% lower fully loaded cost against a matched baseline, with the correctness non-inferiority margin pre-registered before trials.
- **COBOL source-index pilot:** 99% precision and 99% recall is a proposed gate for the exact-anchor indexing task only, on independently annotated data. It is not a target for parsing, code generation, program equivalence, or production modernization.
- **Banking/mainframe migration:** require 100% pass on the task's mandatory financial/state invariants and no silent exceptions. Any tolerance for rounding or floating-point behavior must be explicit and approved per business rule—not inferred after seeing results.
- **Security:** score paired functional-and-security success separately; pre-register maximum critical/high vulnerability escape rates before the confirmatory evaluation.

## Concrete next work, in order

1. **Finish and independently review the source index.** Current Rust pilot is fixed-format-only and source-anchor-only. Grow a licensed/synthetic fixture corpus with annotations for COPY/REPLACING, PIC, USAGE, REDEFINES, FD/SD, SQL/CICS, literals, comments, continuation and columns 73–80. Score precision/recall and diagnostic recall by anchor type. Do not infer that the indexer supports the whole COBOL language.
2. **Add a copybook graph as a separate artifact.** Resolve copybook candidates only within declared roots; reject path traversal and ambiguous case-insensitive matches; retain source and expansion hashes; identify missing dependencies and cycles. Do not silently expand REPLACING without a dialect-aware implementation.
3. **Add a claim-verification lane.** The branch now contains a machine-readable [atomic claim evidence contract](../../tools/vleb/legacy-claim-evidence.schema.json), [claim auditor](../../tools/vleb/audit_legacy_claim.py), and adversarial tests. Generate atomic claims such as “amount is stored as signed packed decimal,” “copybook X is referenced at line Y,” or “this branch executes before commit.” The auditor recomputes source-file/source-span hashes, verifies witness artifact bytes, compares verifier hashes to out-of-band pins, rejects conflicting witnesses, and requires independent-execution evidence for runtime-behavior claims. Its passing verdict validates the evidence chain; it does not itself re-run compiler or transaction behavior. These new tests are not called validated until the exact-head workflow passes. A fluent explanation without a verified witness scores as unsupported, not correct.
4. **Add actual compiler-backed equivalence only after the static lane is stable.** Pin compiler/dialect versions and use explicitly authorized fixtures. Differentially test decimal sign/rounding/overflow, EBCDIC conversion, collation and sort order, fixed records, copybook expansion, return codes, Db2 commit/rollback, JCL restart, duplicate processing and replay idempotency. A GnuCOBOL pass alone is not evidence of IBM z/OS equivalence.
5. **Run matched baseline experiments.** Keep base model, system prompt, scaffold, tools, budgets, network policy, runtime and attempt count identical. Add/remove Symthaea planning, memory and verifier components one at a time. Report paired task outcomes, confidence intervals, cost, p50/p90 time, retries, abstentions and failure families. Analyze disagreements, not just the aggregate leaderboard delta.
6. **Keep each result bound to the current head.** The last baseline pass predates the new trusted manifest and COBOL indexer. No newly added control is validated until the workflow passes at the exact final commit and the raw artifact is inspected.

## Interpretation

The differentiator is not the claim that Symthaea “understands COBOL.” It is an auditable chain from source evidence to an explicit claim, from the claim to a reproducible behavioral witness, and from that witness to an independently pinned engineering result. This gives VLEB a chance to outperform generic coding agents on trustworthiness and legacy-specific fidelity even before the model's raw task-solving percentage surpasses the top comparator.
