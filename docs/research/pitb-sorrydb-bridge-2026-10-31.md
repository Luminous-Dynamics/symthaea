# PITB real-world repository proof cohort: SorryDB bridge (2026-10-31)

**Status:** evaluation protocol proposal; no SorryDB task snapshot has been imported, normalized, or scored by PITB.

## Why add this cohort?

PITB's existing mathematical-family cohort is useful for transfer of proof strategies across theorem families. It does not, by itself, measure whether those strategies help practitioners close proof obligations embedded in real Lean projects.

SorryDB publishes a January 2026 snapshot of 1,000 open proof obligations and a leaderboard with Pass@1 and Pass@32 comparisons. The displayed snapshot reports Aleph Prover at 57.2% Pass@1; Gemini Flash 3 agentic iteration at 30.3% Pass@1; Claude Opus 4.5 at 7.8% Pass@1 and 15.4% Pass@32; and 63.2% combined coverage across listed approaches. Source: https://sorrydb.org/ (snapshot shown as SorryDB_2601).

These are contextual figures from the SorryDB leaderboard, not directly comparable PITB scores. The leaderboard includes proprietary systems, and PITB must not claim superiority without access to the same task set, environment, system versions, and resource accounting.

## Separate endpoints

Do not merge the SorryDB cohort into the 372-family research-math transfer split. Publish two separately versioned cohorts:

1. **Research-family transfer:** typed insight transfer to unseen mathematical families, with semantic statement correspondence and family/component leakage control.
2. **Repository proof engineering:** closure of real project sorry obligations under pinned repository snapshots and exact Lean environments.

For the repository cohort, report:
- exact-target kernel-accepted completion rate at Pass@1 and Pass@k;
- task-level coverage and unresolved/invalid task counts;
- semantic successor/integration acceptance where the project exposes downstream obligations;
- model calls, generated tokens, wall time, checker time, memory if measured, and spend;
- cost per independently verified completion;
- context-source ablations: full project context, Lean cleanup, typed dependency closure, retrieval-only, typed insights, and adaptive repair;
- task snapshot, commit, Lean/Lake toolchain, Mathlib/project dependencies, model/policy version, prompt digest, seed, and receipt digests.

A proof attempt is not a success because the source text compiles. The exact target must be kernel checked, benchmark statement fidelity must be audited, and downstream semantic checks must be reported separately.

## Fair comparison design

Before any comparison:
1. Import a named SorryDB snapshot and pin its exact bytes, version, task IDs, source commits, and license/redistribution constraints.
2. Reproduce the baseline task eligibility and scoring denominator. Missing, unavailable, invalid, or unknown tasks must remain visible; do not silently drop them.
3. Re-run available baselines under matched task, model, context, token/call/time, and checker limits. Published numbers from systems that cannot be rerun under matched conditions remain contextual only.
4. Use task-clustered or repository-clustered uncertainty, depending on the task-generation hierarchy. Avoid treating many obligations from one repository as independent.
5. Audit every task for target drift, trivial/vacuous proof paths, axioms/sorries, dependency leakage, and hidden tests that do not match the declared theorem.
6. Hold out entire repositories and later snapshots for generalization.

## Proposed targets

These are prospective qualification goals, not results:
- at least +5 percentage points in Pass@1 over the strongest reproducible matched baseline, with a positive repository-clustered 95% interval;
- at least 10% lower cost per verified completion at no more than a 1 percentage-point loss in verified completion rate;
- 100% exact-target preservation in repair episodes;
- 100% kernel checking of accepted completions;
- no unreviewed axiom/sorry leakage;
- zero false accepts across registered target-mutation and trivial-solution controls;
- replication on a later snapshot or held-out repositories.

The primary first win should be lower cost at fixed reliability, not an inflated Pass@k score. A more ambitious outcome is improved proof completion from strategy transfer without exposing the full repository context to the prover.

## Positioning

SorryDB is a practical proof-engineering cohort, not a replacement for MiniF2F, PutnamBench, FormalMATH, LeanGraphSearch, T², BlueprintTrace, or PITB's own support-landscape benchmark. Each measures a different capability. A useful PITB release should publish a capability vector and resource frontier rather than collapse these endpoints into one composite score.
