# PITB: from proof-support retrieval to independently verifiable support decisions

**Date:** 2026-10-31  
**Status:** research design and benchmark proposal; not an implemented or measured system.

## The paradigm shift

A retrieval system can return plausible premises. A proof searcher can discover a proof. A Lean kernel can accept a proof term. None of those alone establishes that a selected support context is the smallest reliably sufficient context under a frozen prover, budget, model, and environment.

PITB should make a stronger object its central output: a **support-decision certificate** that a small independent verifier can check without trusting the optimizer that selected the context.

The certificate makes four separate claims:

1. **Validity:** the proposed context is a well-formed ordered Lean local context, including dependency closure for both retained mutable declarations and retained fixed declarations.
2. **Operational sufficiency:** under the frozen execution policy and budget, the context clears a preregistered repeated-trial reliability floor using simultaneous exact inference.
3. **Minimality:** no proper well-formed subcontext in the fully enumerated finite lattice also clears that same reliability gate.
4. **Efficiency:** the chosen context is not dominated on the preregistered resource dimensions by another reliability-qualified context.

This is not a proof that a premise is logically necessary for every possible proof of the theorem. It is a finite, operational claim about a declared search system and frozen conditions. Logical necessity, proof-term occurrence, and bounded-prover reliability remain distinct concepts.

## Architecture: proposer, optimizer, verifier

### Evidence producers

- A pinned Lean elaboration/export step emits the complete ordered local context, declaration IDs, types, let-values, and direct local-declaration references.
- A frozen runner records each context-by-replicate execution, including target, model/policy/toolchain, budget, seed, proof output digest, Lean kernel result, semantic successor-gate result, and measured resources.
- Each producer emits a canonical receipt payload. The payload is signed by a key whose public key and role are pinned in the study trust policy.

### Optimizer

The optimizer can use HDC/LTC embeddings, graph traversal, retrieval, learned cost models, and adaptive search to propose candidate contexts and proof routes. Its output is untrusted. It may be fast, approximate, stochastic, or buggy; it cannot authorize itself.

### Independent verifier

A separate, deliberately smaller implementation checks the certificate from the raw evidence ledger. It must not trust the optimizer's report fields. It independently:

- verifies signatures, key roles, revocation/rotation policy, and receipt-chain bindings;
- reconstructs the target, environment, telescope, and complete dependency graph from pinned source evidence;
- confirms every valid context and replicate is present exactly once, with unique seed per replicate and common paired seeds across contexts;
- rejects invalid contexts as invalid states, never as mathematical non-solves;
- checks kernel and semantic gate receipts and the frozen resource caps;
- recomputes exact one-sided Clopper-Pearson lower bounds and the Bonferroni familywise adjustment;
- recomputes the reliable minimal-support antichain and Pareto frontier from all qualified contexts;
- verifies target immutability and all successor/invariant obligations.

The verifier should use an independently written confidence-bound implementation (for example, log-space tail summation rather than the analyzer's recurrence) and separately published test vectors. The trusted computing base should be the signature verifier, canonicalization rules, Lean kernel/checker identity, and compact certificate verifier—not the HDC ranker or search optimizer.

## A missing measurement: use the paired design

The v3 ledger pairs every context with the same seed at the same replicate index. The current analyzer uses this to check schedule identity, but does not yet exploit the pairing in the report.

The next analysis should report matched outcomes against the full-context baseline for every candidate context:

- both candidate and full context succeed;
- candidate succeeds while full context fails;
- full context succeeds while candidate fails;
- both fail.

These discordant counts expose whether an ablation consistently changes outcomes under matched randomness. An exact McNemar test or a preregistered paired confidence procedure can be added after checking its assumptions. Until then, the four cell counts and paired resource deltas should be descriptive only; they are not proof of causality. Pairing is useful only when seeds are independently sampled across replicate indices and all other randomized policy inputs are frozen.

## Semantic correctness is a separate gate

A kernel-accepted proof establishes the encoded Lean proposition, not that the proposition or benchmark faithfully captures the intended task. PITB should therefore maintain separate evidence for:
- target statement identity and benchmark translation review;
- kernel proof acceptance;
- successor-theorem or invariant regression checks;
- target immutability during repair;
- declared proof dependencies and actual proof-term support.

A compile-only score must never be substituted for semantic correctness.

## Competitive landscape and fair comparisons

Recent work sets useful but non-interchangeable comparators:

| System / benchmark | Published evidence | What PITB should compare |
|---|---|---|
| LeanGraphSearch | On its 810-query MathlibQR fair subset, Recall@100 is 94.7% vs 82.7% for LeanSearch v2; complete-route coverage on MathlibMPR is also reported. | Direct retrieval quality, multi-hop support recovery, route completeness, and end-to-end proof conversion under the same corpus and call budget. |
| Compiler-guided adaptive Lean search | On seven miniCTX-v2 projects, +12.8 percentage points average pass rate and -21.9% LLM calls vs pass@k within a pass@32 budget. | Seed scheduling, best-state retention, cross-model diversity, proof success and calls at fixed budgets. |
| BlueprintRepair / BlueprintTrace | 142 controlled blueprint failures; typed edits cost 1.30x less per solved state than source patching and 2.06x less than full rewrites in the reported setup. | Typed repair vs patch vs rewrite on identical states, models, feedback, and token budgets. |
| T² semantic testing | 2,206 problems from five Lean repositories; best reported Testing Accuracy is 38.9% in its setup despite high compilation success. | Kernel acceptance vs successor/invariant preservation; false acceptance under mutation. |
| Formal benchmark audit | An audit of five commonly used Lean benchmark corpora reports 4,833 findings, including 398 mechanically certified issues. | Dataset/specification audits, adversarial negative controls, vacuity/axiom scans, split leakage, and independent replication. |

These numbers are not directly rank-comparable: tasks, models, data splits, inference budgets, and scoring protocols differ. PITB should reproduce each baseline on shared frozen tasks before making leaderboard claims.

## Proposed qualification targets (aspirational, not results)

1. **Decision soundness:** zero false accepts across the full hand-built mutation suite and at least 10,000 generated corrupt-certificate mutations; every forged, replayed, incomplete, target-mutating, over-budget, or semantically rejected certificate must fail closed.
2. **Reliability:** selected contexts clear a simultaneous one-sided 95% familywise confidence gate with lower bound at least 0.80. Use power planning for the full well-formed context family; do not use 20-trial pilots as final evidence.
3. **Efficiency:** at least 10% lower cost per independently verified success than the strongest matched baseline, with no loss of the reliability floor; separately report tokens, calls, checker time, wall time, and cost.
4. **Search quality:** improve complete proof-route coverage and end-to-end solve rate over direct retrieval, graph retrieval, MVarId.cleanup, dependency closure, random valid contexts, and the strongest adaptive controller under identical budgets.
5. **Repair quality:** typed local repair should preserve the target and untouched subproofs, with successor/invariant checks and full accepted/rejected trajectories.
6. **Generalization:** frozen test components must be held out by audited mathematical family and repository, not merely by file path. Publish cross-family relationship evidence and the external reviewer coverage.
7. **Reproducibility:** exact source SHA, toolchain, model/version, prompt/policy digest, seed derivation, budget, receipt signatures, and verifier version accompany every reported number.

A good first superiority result is not a huge headline pass rate. It is a statistically supported improvement in verified successes per dollar or a lower-cost support context at the same reliability, on a contamination-audited held-out split.

## Immediate sequence

1. Land and run the v3 seed-integrity and fixture regression controls.
2. Build a real Lean local-context exporter; derive the dependency graph from elaborated types and let-values rather than hand-authored lists.
3. Add cryptographically authenticated runner/kernel/semantic-gate receipts with pinned trust roots.
4. Implement the independent support-decision verifier and a disjoint set of mutation tests.
5. Add paired candidate-vs-full-context outcome tables and exact paired analysis.
6. Run the frozen 2–4 mutable-declaration calibration, then power the larger lattice using exact simultaneous inference.
7. Compare baselines at matched tasks, models, token/call/time budgets, and independent semantic checks.

No SOTA win is claimed until these tests are run on exact committed code and independently reproduced.
