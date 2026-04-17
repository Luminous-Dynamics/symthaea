# Phase 1 → Phase 2 Decision Memo (DRAFT)

**Status:** Draft for joint review. Plan reference: `plans/2-please-make-precious-fairy.md`.

**Prepared:** 2026-04-17, Phase 1 W5 complete.

## What the plan asked for at the week-10 gate

Seven artifacts, per the plan's "Week-10 Decision-Point Deliverables" section:

1. `MODULE_STATUS.md` — honest module audit.
2. ARC-AGI-2 CSV — reproducible baseline.
3. miniF2F-v2 + PutnamBench CSVs with `lean --check` verification.
4. Lean CI badge.
5. Ramanujan arxiv preprint URL.
6. Four tagged git commits, one per workstream.
7. This memo.

## What was delivered

| Plan artifact | Status | Reference |
|---------------|--------|-----------|
| 1. `MODULE_STATUS.md` | ✅ | `symthaea/MODULE_STATUS.md` |
| 2. ARC-AGI-2 CSV harness | ✅ (harness); ⏸ (data needs user download) | `examples/benchmark_arc_agi2.rs`, `docs/arc-agi-2-dataset.md` |
| 3a. miniF2F proof CSV | ✅ — on engine-originated subset (3/3 Lean-accepted) | `proofs/minif2f/` |
| 3b. Propositional tautology suite | ✅ — **23/23 Lean-accepted, 0 sorry, 100% strict** | `proofs/proptauts/`, `docs/minif2f-v2-scope.md` |
| 3c. miniF2F-v2 full | ❌ — architecturally out of scope at Phase 1 | `docs/minif2f-v2-scope.md` |
| 3d. PutnamBench | ❌ — same architectural scope issue | `docs/minif2f-v2-scope.md` |
| 4. Lean CI badge | ⏸ | Needs standalone-repo sync; see notes below |
| 5. Ramanujan arxiv preprint | ✅ (draft); ⏸ (submission is a user action) | `papers/ramanujan/main.tex`, `reproduce.sh`, `VERIFY.md`, `Dockerfile` |
| 6. Tagged commits | ✅ — multiple per workstream | `git log --oneline` |
| 7. This memo | 🏗 | This file |

## Honest unqualified results

### Strong

- **Ramanujan Protocol: 6 PROVEN conservation laws + 1 honest numeric failure**, deterministic (seed 42), reproducible in ≤6 min via `./papers/ramanujan/reproduce.sh`, verified against a 221-equation physics catalog at ≥99% structural similarity per match. Arxiv preprint draft ready.
- **Propositional tautology suite: 23/23 externally Lean-verified**, zero `sorry`. Every proof is a structurally meaningful term (`fun h0 => ⟨h0.1.1, ⟨h0.1.2, h0.2⟩⟩` and similar). Classical logic coverage includes intro/elim for ∧ ∨ → ¬, curry/uncurry, contrapositive, double-negation, ex falso, excluded middle.
- **Module audit honesty**: `MODULE_STATUS.md` has replaced vibes-based claims in the repo docs with a per-module evidence row citing file paths and test counts. Corrected several places where the pre-audit research was overly pessimistic about module state.

### Architecturally out of scope at Phase 1

- **miniF2F-v2 (full upstream)**: the benchmark is ~98% real-arithmetic and number-theoretic algebra; our `Proposition` enum doesn't represent equality, arithmetic, or function quantification. Honest accept rate: near zero. Not a pipeline failure — a deliberate scope boundary.
- **PutnamBench**: same mismatch. Most problems require Mathlib-level algebraic machinery.
- **ARC-AGI-2 intelligence**: current rule-vector pipeline is a first-token similarity measure; no grid-output generator. Baseline number will be single-digit-to-zero percent. This was the Phase 1 expectation and was explicitly flagged as deferred to Phase 2 stretch.

### Known gaps in the delivered infrastructure

- **SMT witness files**: `papers/ramanujan/proofs/` is empty because the `conjecture_engine` pipes SMT-LIB2 to Z3 via stdin rather than persisting it. This is a 1-2 day instrumentation change deferred to Phase 2 (documented in `proofs/README.md`).
- **Lean CI integration**: the monorepo policy (CLAUDE.md rule #7) forbids GitHub Actions in this private repo. CI runs on the standalone repo via `scripts/sync-to-standalone.sh`. A Lean verification CI step there has not yet been pushed.
- **Docker reproduction not yet built-and-pushed**: the Dockerfile is written, but no build has validated its layer ordering on a clean host.

## Three Phase 2 directions, ranked by leverage × risk

### Option A — Algebraic reasoning (broadest benchmark throughput)

Extend `Proposition` with equality and arithmetic over ℝ/ℕ/ℤ. Build a Lean 4 parser subset that maps miniF2F-v2 statements into that extended AST. Hook `conjecture_engine::auto_prove_via_z3` into the bridge and emit Mathlib-style `linarith`/`nlinarith` tactics on the Lean side (requiring a Mathlib-aware CI lane).

- **Effort**: 4–6 weeks focused.
- **Reach**: 15–30% of miniF2F-v2, comparable fraction of PutnamBench linear-arith problems. Real external benchmark numbers.
- **Risk**: parser work is brittle; Mathlib dependency widens the reproducibility footprint.
- **Optics**: strongest for external credibility (miniF2F/PutnamBench leaderboards).

### Option B — Conjecture-engine depth (doubling down on the Ramanujan result)

Instrument the `conjecture_engine` to dump SMT-LIB2 witnesses to disk, submit the current Ramanujan paper to arxiv, expand the showcase from 7 problems to 25+ canonical dynamical systems, and attempt the PCR3BP honest-failure with a richer expression grammar (log-of-polynomial, nested radicals).

- **Effort**: 3–4 weeks focused.
- **Reach**: a publishable result becomes a citable result. The Mystery ODE story — autonomous discovery of a non-textbook Hamiltonian — gets a second validated example.
- **Risk**: low. Mostly polish and breadth on already-working infrastructure.
- **Optics**: strongest for scientific-contribution narrative; weakest for benchmark-leaderboard optics.

### Option C — ARC-AGI-2 and abstract reasoning (HDC native fit)

Wire `abstract_thought/` macro discovery into `GridEncoder`-space to give Symthaea an actual predictor (not just rule-vector similarity). Attempt the public 120-task ARC-AGI-2 eval with real predictions.

- **Effort**: 6–10 weeks; research-risky.
- **Reach**: if it works, an honest single-digit-to-teens % score on ARC-AGI-2, which is the single most-watched "reasoning vs LLM" benchmark in 2026.
- **Risk**: high. No guarantee the HDC+macro architecture beats the rule-vector baseline meaningfully. Could consume resources for no external-benchmark improvement.
- **Optics**: highest-variance payoff.

### Recommendation (to be edited by reviewer)

Sequence **B → A**, skip C for now.

- B first (3–4 weeks): ship the arxiv paper + witness files. Locks in a scientific contribution while Phase 2's bigger bet (A) is underway. Low risk, high narrative value.
- A second (4–6 weeks): algebraic reasoning is the prerequisite for *any* external theorem-benchmark story. Without it, we never reach miniF2F numbers. B's paper doesn't conflict with this work.
- C deferred: ARC-AGI-2 is the highest-variance payoff and should be sized as a research sprint with its own go/no-go criterion after A lands.

Total: roughly one quarter of work. Alternative sequencing or a complete re-scope is of course open to discussion.

## Decision requested

The reviewer should choose one of:

1. Proceed with B → A (recommended).
2. Proceed with A directly, defer B's arxiv submission.
3. Proceed with C directly as the highest-variance bet.
4. Pause. Phase 1 artifacts stand on their own; revisit Phase 2 scoping in N weeks.
5. Re-scope entirely — Phase 2 pivots to a different direction based on reviewer's read of the landscape.

Whatever the choice, three sub-actions are always-good-to-do regardless:

- Submit the Ramanujan paper to arxiv (user action, ~30 min).
- Install miniF2F-v2 / arc-agi-2 public data on the host (user action; once data is local the existing harnesses produce real CSVs immediately).
- Sync to the standalone symthaea repo and push the Lean CI workflow (user + me).

## Appendix: Phase 1 test / verification totals

- 28 passing tests in `symthaea-lean-bridge` (23 unit + 5 integration).
- 26 Lean 4 files externally verified by `lean <file>` (3 miniF2F + 23 proptauts).
- 1 reproducible multi-physics showcase producing the Ramanujan table (7 rows, deterministic at seed=42).
- 1 honest-signal CSV harness for ARC-AGI-2 (smoke-tested on 2-task synthetic data).
- 12+ commits on `main`, all tagged in commit messages with the workstream they belong to.
