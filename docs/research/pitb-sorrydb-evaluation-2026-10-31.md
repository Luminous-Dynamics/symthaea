# PITB research note: SorryDB as a second external evaluation axis

**Date:** 2026-10-31  
**Status:** design proposal and source review; no PITB SorryDB execution or result.

## Finding

SorryDB should be added as a separate external benchmark track. Its January 2026 SorryDB_2601 snapshot contains a fixed 1,000-task evaluation slice of real Lean sorry obligations, with repository/commit/Lean-version/source-location metadata and an independent original-project verification pipeline. The ICML 2026 paper describes a dynamic dataset from 78 real-world formalization projects and emphasizes that off-the-shelf, specialized, and agentic approaches are complementary.

This changes the PITB comparison strategy. The frozen OpenAI-math family split measures transfer of mathematical strategies across curated problem families. SorryDB measures practical completion of open obligations in the context of the actual projects that contain them. The task distribution and primary estimand differ; keep separate reports, denominators, confidence intervals and leaderboards.

## What the current external frontier says

The hosted SorryDB leaderboard for SorryDB_2601 reports:
- Aleph Prover: 57.2% Pass@1.
- Gemini Flash 3 agentic: 30.3% Pass@1.
- Gemini Flash 3 self-correcting: 27.9% Pass@1.
- Claude Opus 4.5 self-correcting: 27.1% Pass@1.
- General-purpose Gemini Pro 3: 11.0% Pass@1 / 20.5% Pass@32.
- General-purpose Gemini Flash 3: 10.8% Pass@1 / 20.5% Pass@32.
- General-purpose Claude Opus 4.5: 7.8% Pass@1 / 15.4% Pass@32.
- General-purpose GPT-5.2: 6.2% Pass@1 / 13.2% Pass@32.

The displayed 63.2% combined figure is an ensemble union (tasks solved by at least one listed approach), not a single prover result. The benchmark's live leaderboard and repository materials should be version-pinned when used: a union statistic is especially sensitive to which approaches and dataset snapshot are included. Never set 63.2% as a single-system baseline.

The current hosted table is a useful external reference, but the different attempt regimes matter. In particular, the 30.3% agentic row is Pass@1; the 15.4% Claude general-purpose result is Pass@32. A comparison that puts those numbers on one unqualified accuracy axis is invalid.

## PITB's distinct value proposition

PITB should not claim to compete with specialized prover models merely by adding more generations. The research hypothesis is that proof-state evidence and transferable mathematical strategies improve the decision policy used to complete repository obligations:

- Capture the actual elaborated proof state, including ordered local declarations, binder information, auxiliary declarations, and the local-instance registry.
- Retrieve intent-relevant premises and then traverse their formal dependency routes. The latest Legendre table gives Lean Finder v2 97.3% Recall@100 on 894 resolved MathlibQR queries; LeanGraphSearch reports 94.7% on a different 810-query fair subset and 42.8% complete-route coverage at Covered@20 on MathlibMPR. Those recall results are not head-to-head or downstream SorryDB completion scores.
- Attach proof insights as typed contracts with preconditions, evidence anchors, state transitions, and failure boundaries; abstain on incompatible targets.
- Generate candidate support/context projections and typed repair actions, then use Lean's checker as the independent feedback and acceptance boundary.
- Allocate the next token/tool/search/repair budget according to marginal expected verified success per unit cost; stop when the remaining actions are not justified by calibrated evidence.

The central design change is from "generate a proof string" to "perform an evidence-backed, reversible proof-hole transaction." The proposer may be an LLM, HDC retriever, dependency graph, tactic engine or portfolio. None of them may decide that their own output is valid. A separate checker validates the resulting patch, exact target-type identity, source scope and trust tier.

## Fair experiment design

The matched causal arms and task protocol live in the SorryDB track README. In summary:

- Freeze the exact 1,000-task SorryDB_2601 evaluation file, its SHA-256, upstream source commit and task IDs. Do not regenerate or edit the cohort.
- Pin each source repository commit and Lean toolchain. Rebuild the original project and reconstruct the goal from Lean; treat SorryDB's pretty-printed debug_info.goal as a debugging convenience, not an authoritative proof-state object.
- Compare same-model, same-prompt, same-checker, same-task arms: baseline; baseline + retrieval; baseline + local-instance-aware context evidence; baseline + typed insight transfer; and the combined system.
- Freeze token, attempt, tool-call, wall-time and monetary budgets. Report Pass@1, Pass@8 and Pass@32 separately, with cost and latency curves.
- Allow a candidate to replace only the designated sorry span for the primary integrity-qualified endpoint. Independently rebuild/check from a clean checkout; confirm the elaborated target type is unchanged under a versioned serializer; reject admitted proofs and unapproved axiom/trust changes.
- Derive scores from attempt receipts. Account for every assigned task, including timeouts, unavailable runners and non-attempts. Unknown evidence blocks confirmatory status.
- Cluster uncertainty by repository, not only by individual theorem, because goals in a project share imports, conventions, maintainers and build dependencies. Report task-weighted and repository-weighted aggregates.
- Treat the public January slice as a historical external test, not proof of contamination-free generalization. Freeze a later, temporally separate SorryDB cohort for forward evaluation.

## Targets

These are qualification targets, not results:
1. 100% task/arm/replicate denominator accounting and 100% independent replay of claimed successes.
2. Zero false accepts across the registered statement, patch-scope, sorry/axiom, and verifier-receipt mutation suite.
3. At least +5 percentage points absolute Pass@1 over the matched PITB S0 baseline, with a repository-clustered 95% interval excluding zero.
4. At least 10% lower cost per verified success at matched solve rate, or a strictly better success-cost frontier.
5. A stretch goal of at least 60% Pass@1 on the frozen January cohort, above the published 57.2% Aleph Prover Pass@1. This is only a fair head-to-head if attempt budgets, checker semantics, and trust tiers are comparable.
6. Reproduce the direction of improvement on a separately frozen post-January-2026 cohort without tuning on it.

## Important gaps before a benchmark run

- The source split bytes and exact upstream source commit have not yet been hashed into a PITB cohort manifest.
- No importer has yet converted the SorryDB JSON records into a validated PITB task manifest.
- No real task has passed a clean project rebuild, proof-hole-only patch audit, type-digest comparison and independent verification through PITB.
- No baseline or PITB model has been run on the fixed split.
- The current Lean v2 evidence exporter and the broader PITB exact-head workflow are themselves still unverified.

The new schema and workflow in this commit validate only the shape and fail-closed behavior of run receipts. They do not validate real benchmark rows, compile Lean, or establish a pass rate.

## Sources

- SorryDB leaderboard: https://sorrydb.org/
- SorryDB ICML 2026 paper: https://proceedings.mlr.press/v306/letson26a.html
- SorryDB source/database format: https://github.com/SorryDB/SorryDB and https://github.com/SorryDB/SorryDB/blob/master/doc/DATABASE.md
- SorryDB strategy contract: https://github.com/SorryDB/SorryDB/blob/master/doc/STRATEGIES.md
- Legendre Mathlib retrieval board: https://www.legendre-leaderboard.com/
- LeanGraphSearch: https://pluslabnlp.github.io/LeanGraphSearch/
- Lean 4 local instance synthesis: https://lean-lang.org/doc/reference/latest/Type-Classes/Instance-Synthesis/


## Source pin and initial manifest audit

The January evaluation file is identified at SorryDB source commit \`74b3b63f8ba890d675f362924b21ac55db410383\`, path \`data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json\`, Git blob SHA-1 \`af3b663429dc4689ee728d2fb5c7d8d5034476f7\`, byte count 1,599,271, and SHA-256 \`26bfd11099f6b8db7f80d49bdea5d9f8ea0d23ba65d44d93f67ed42ba1530062\`. A local implementation of the Git blob object hash over the retrieved UTF-8 payload reproduced the exact GitHub tree blob SHA-1. Structural inspection found 1,000 task records, 78 unique repository remotes, zero duplicate task IDs, and no unsafe repository-relative paths.

Two rows use a zero-based start column; therefore the importer allows column zero while retaining one-based line numbers and strictly ordered source spans. This is a source-format detail that the generic documentation example does not make obvious.

A Rust importer has been added to the PITB crate. It verifies byte count and SHA-256 before parsing, checks the fixed task/project counts, validates every repository pin and source coordinate, rejects duplicate IDs/path traversal, and emits a deterministic task manifest with provenance hashes. Its self-tests and real-snapshot run are wired into the focused SorryDB workflow; no compile or workflow pass is claimed until observed.


## Environment readiness and local-instance coverage

SorryDB_2601 contains 14 Lean toolchain tags, not one homogeneous compiler environment. Separate source/toolchain/dependency/project reconstruction from proof completion. A task enters the model-evaluation denominator only after the original pinned project passes the preflight build. Also publish operational yield over the entire assigned cohort, retaining preflight failures as non-solves rather than silently dropping them. See pitb-sorrydb-toolchain-and-smoke-cohort-2026-10-31.md for the exact version distribution and the three pinned smoke-task candidates.

A textual scan of the dataset's debugging goal previews finds 426 instance-like binder strings. This is only a selection heuristic. The first candidate exercising this path uses clap-lean, with a preview binder inst✝ : Fact (Nat.Prime p); actual local instance resources must be captured from Lean's metavariable declaration, not inferred from the rendered preview.
