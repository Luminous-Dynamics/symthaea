# Phase 3 miniF2F-v2 Failure Analysis — Mapping 7 Rejections → Phase 4 Fixes

**Source:** `docs/phase3-results/minif2f_curated_results.csv` (32 hand-translated miniF2F-v2 problems, commit `7908cf86d9`). Re-measured on main post-Phase-3-Move-1 cross-term hints: **25/32 = 78.1%** accepted, 7 rejected. This doc categorizes the 7 rejections precisely so Phase 4 can target the right fixes without overlap.

Already known from the commit message: Phase 4 intake = "sq_nonneg offset widening + And-splitter." This analysis confirms those as the two biggest levers (closing 5/7) and identifies the remaining 2.

## Rejection-by-rejection

### Group A — And-splitter needed (2 rejections, each closeable by a single cascade alternative)

| # | Problem | Statement shape | Why it rejects |
|---|---------|-----------------|----------------|
| 1 | `mathd_algebra_126` | `x = 15 ∧ y = -11` | Goal is a conjunction. The cascade's single-tactic alternatives close the WHOLE goal or fail; there's no `constructor <;> linarith` form. Z3 says `unsat` so the math is sound — only the Lean-side emission pattern is wrong. |
| 2 | `mathd_algebra_101` | `(hypothesis on quadratic) → x ≥ -2 ∧ x ≤ 7` | Same shape. Quadratic hypothesis is the nonlinear content; once `constructor` splits, each branch closes with `nlinarith`. |

**Fix:** add `(constructor <;> (first | linarith | nlinarith [...] | omega | ring); done)` as an alternative near the top of the cascade, or a generic `all_goals` wrapper that runs the whole cascade on each subgoal. ~15 LOC.

**Expected delta:** +2/32 → 27/32 (84%).

### Group B — Offset sq_nonneg widening (2 rejections, solvable by ~10 LOC in `build_nlinarith_hints`)

| # | Problem | Statement shape | Missing hint |
|---|---------|-----------------|--------------|
| 3 | `mathd_algebra_113` | `∀ x : ℝ, x² − 14x + 3 ≥ 7² − 14·7 + 3` | `sq_nonneg (x - 7)` — current generator emits `sq_nonneg (x - 1)` and `sq_nonneg (x + 1)` only. |
| 4 | `mathd_algebra_410` | `y = x² − 6x + 13 → 4 ≤ y` (equivalently `(x - 3)² ≥ 0`) | `sq_nonneg (x - 3)`. |

**Fix:** in `build_nlinarith_hints` (currently at `crates/symthaea-lean-bridge/src/fol_ext_bridge.rs` around line 210), replace the hard-coded `n - 1` / `n + 1` with a loop over small integer offsets — `for k in 1..=10 { parts.push(format!("sq_nonneg ({} - {})", n, k)); parts.push(format!("sq_nonneg ({} + {})", n, k)); }`. Cost: ~5 extra hints per binder, but nlinarith filters these efficiently. ~10 LOC.

**Expected delta:** +2/32 → 29/32 (90.6%) with Group A also applied.

### Group C — Harder, needs targeted infrastructure (3 rejections)

| # | Problem | Statement shape | What's needed |
|---|---------|-----------------|---------------|
| 5 | `mathd_algebra_55` | `q = Σ...` ; `p = Σ...` ; `q/p = 2/3` | Division in the goal. `nlinarith` can't directly handle the quotient. Cascade needs `field_simp` before `linarith`/`nlinarith` (clears denominators by multiplying through). ~5 LOC: add `(field_simp; linarith; done)` and `(field_simp; nlinarith [...]; done)` alternatives. |
| 6 | `mathd_algebra_338` | 3 linear eqs in `(a, b, c)` → `a·b·c = -56` | After Gaussian elimination gives `a = -7, b = 3, c = 8`, the product is a numeric computation. `linear_combination` tactic plus `norm_num` is the idiomatic Mathlib tool. 1 new cascade alternative + careful ordering. ~15 LOC. |
| 7 | `mathd_numbertheory_326` | `(n-1)·n·(n+1) = 720 → n + 1 = 10` over ℤ | Nonlinear integer Diophantine. `omega` doesn't handle 3-term products. This is genuinely hard for automated tactics; `decide` works if Lean can bound `n`, but the bound isn't syntactic. Probably needs `interval_cases` + `decide`. ~20 LOC, but this category is brittle. |

**Expected delta:** +3/32 → 32/32 (100%) only if all three infrastructure items land. More realistic: +1-2/32 in the short term.

## Phase 4 intake recommendation

Priority order (biggest wins first, each is ~10-20 LOC in one file):

1. **And-splitter** (Group A) → +2 fixtures. Cleanest win; changes the cascade structure.
2. **Offset widening** (Group B) → +2 fixtures. One-line loop change; safest.
3. **field_simp fallback** (Group C #5) → +1 fixture. Pair with the existing linarith/nlinarith alternatives.
4. **linear_combination** (Group C #6) → +1 fixture. New cascade alternative; worth it for polynomial-system problems.
5. **interval_cases + decide** (Group C #7) → +1 fixture. Phase 5 scope; lower priority because Diophantine coverage is a long tail.

If Phase 4 lands steps 1-4, measurement should hit **29/32 = 90.6%**. Step 5 pushes it to 32/32 = 100% but is the highest-risk addition (Mathlib's `interval_cases` has non-obvious failure modes).

## Comparison to Lean's native `grind` baseline

Per the April 2026 Lean 4 release notes, `grind` reaches ~32.4% on miniF2F-v2. Symthaea currently sits at **78.1% on a curated 32-problem subset** — roughly 2.4× the `grind` baseline, caveat that the curation biases toward problems the cascade has the right shape for (see `scripts/filter_minif2f.sh` for the filter predicate).

Phase 4 target after steps 1-4: ≥90% on the same curated set, which would be ~2.8× `grind` on the apples-to-apples comparison. A fair cross-system comparison requires running `grind` on the same 32 files — worth doing as a companion measurement when Phase 4 lands.

## What this doc is NOT

- Not a code change to the cascade. The parallel Phase 3/4 session has uncommitted cascade work in their worktree; adding to the cascade from this tree would conflict.
- Not a commitment to any Phase 4 timeline. Lists the LOC estimates but the schedule is the parallel session's call.
- Not a claim that 29/32 is guaranteed. Expected deltas assume the cascade alternatives don't interfere with each other; `first`'s backtracking means the alternatives compose cleanly in principle, but real-world tactic interactions need measurement.

## Artifacts referenced

- Results CSV: `docs/phase3-results/minif2f_curated_results.csv`
- Emitted proofs: `proofs/minif2f_curated/*.lean` (32 files; 25 Lake-accepted)
- Filter script: `scripts/filter_minif2f.sh` (reproduces the 178-candidate → 32-final curation)
- Current cascade emitter: `crates/symthaea-lean-bridge/src/fol_ext_bridge.rs` — `synthesize_arith_tactic` + `build_nlinarith_hints`
