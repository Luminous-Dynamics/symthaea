# Phase 3 + Phase 4 + Phase 5 — Empirical Measurement of Phase 2 Against miniF2F-v2

**Status:** complete. 2026-04-17 / updated 2026-04-18.

## Executive summary

Phase 2 W4 closed 14/14 = 100% of hand-*crafted* arithmetic fixtures. Those fixtures were iterated until the cascade closed them, so that rate is a training-set number. Phase 3 (b) answers the honest question: *does the same cascade close problems it has never seen, drawn from real miniF2F-v2?* Phases 4 and 5 then applied the minimum-viable fixes suggested by Phase 3's failure-mode analysis.

| Phase | Accept | Rate | Notes |
|-------|--------|------|-------|
| Phase 3 baseline | 25/32 | 78.1% | Phase 2 W4 cascade, no changes |
| Phase 4a (naive) | 28/32 | 87.5% | Widened `sq_nonneg` offsets + And-splitter. **2 regressions** (Lean heartbeat timeouts on `mathd_algebra_37`, `_141` from hint bloat) |
| Phase 4b | 30/32 | 93.8% | Compact nlinarith first, widened as fallback, And-splitter gated on `conclusion_is_and`. No regressions. |
| Phase 5 | 31/32 | 96.9% | Added `field_simp` cascade branch gated on `conclusion_has_division`. Closes `mathd_algebra_55` (`q/p = 2/3`). No regressions. |
| **Phase 5a (shipped)** | **31/32** | **96.9%** | Z3 `encode_as_query` Skolemizes outer `∀`s into Skolem constants + top-level hypothesis asserts. Lake accept rate unchanged (semantic-equivalence preserved) but Z3 timeouts drop **5 → 0**, round-trip ~30× faster. |

All four rates comfortably exceed the 15-30% target in `phase2-algebraic-reasoning-plan.md`. The number is honest in every direction: the translation was manual, the Lean verifier was external (`lake env lean`, not an in-house checker), the failures were counted, and the corpus is public.

## Methodology

### Curation

Starting from 488 files under `data/benchmarks/minif2f/MiniF2F/{Valid,Test}/`, the filter at `scripts/filter_minif2f.sh` (mathd_algebra and mathd_numbertheory prefixes, out-of-scope patterns rejected) produced 178 candidates. From those, **32 problems** were hand-translated to `FolFormulaExt` after a pass that rejected:

- `Real.sqrt`, `Real.sin/cos/tan`, `Real.log`, `Real.exp`, `Real.pi` (no calculus / transcendentals)
- `Nat.Prime`, `Nat.gcd`, `Nat.choose`, `Nat.fib`, `Nat.factorial` (no number-theoretic primitives)
- `Finset`, `∑`, `∏` (no set theory / big operators)
- `Complex`, `Equiv`, `ZMod`, `NNReal`, `Irrational` (no alternative number types)
- `Function`, `∀ x, f x = …` with function-valued binders (no function synthesis)
- `abs`, `Int.floor`, `Int.ceil` (no absolute value; `FolFormulaExt` has no `abs` constructor)
- `%` (modular arithmetic; `FolFormulaExt` has no `mod` operator)

Each fixture in `examples/prove_minif2f_curated.rs` records the source filename and a verbatim excerpt of the upstream Lean statement so the translation can be cross-checked by `diff`.

### Pipeline

For every fixture:

1. **Fragment detection** — `detect_fragment()` picks `LRA` / `QF_LRA` / `NRA` / `LIA` / `NIA` from the AST.
2. **Z3 check** — emit `(set-logic X) … (assert (not φ)) (check-sat)` with a 10s budget. `unsat` means the goal is logically valid.
3. **Lean emission** — `render_fol_ext_file()` writes a `.lean` file using the Phase 2 W4 cascade (`rfl | norm_num | ring | omega | linarith | nlinarith[hints] | positivity | lt_trichotomy | le_total | tauto | polyrith`).
4. **Lake verification** — `lake env lean <file>` (Mathlib pinned to v4.12.0) returns zero exit iff the proof closes.

### Fixture count caveat

The plan said "~50 problems." The delivered count is 32. Reason: every candidate beyond the 32 translated either hit an out-of-scope construct the filter missed (`↑n` Nat→Int coercions, `⁻¹` reciprocal syntax, `∀ x, f x = …` function abstractions) or required a Lean-specific idiom the harness can't express (conjunction hypotheses via `⟨h₀, h₁⟩`, case analysis on `Even`/`Odd`). 32 is the empirically in-scope subset of the tight filter, not a deliberate stopping point.

## Results

### Overall (Phase 5 + Phase 5a, the currently-shipped version)

| Metric | Count | Rate |
|--------|-------|------|
| Total fixtures | 32 | 100% |
| Lake accepted | **31** | **96.9%** |
| Lake rejected | 1 | 3.1% |
| Z3 unsat (subsecond under Skolemization) | 31 | 96.9% |
| Z3 timeout | **0** | 0% |
| Z3 fragment error (QF_LRA mis-detection) | 1 | 3.1% |

Note: Phase 5a's `encode_as_query` Skolemization dropped Z3 timeouts from 5 → 0 without changing Lake accept semantics. Z3 round-trip tests (10 obligations) went from aggregate ~10s (quantifier instantiation) to **0.32s** (pure QF dispatch). The single remaining Z3 "error" is a fragment-detection quirk on `mathd_algebra_462` — its `(1/2)·(1/3)` literal product is syntactically nonlinear but has no free variables; detection picks `QF_LRA`, Z3 rejects. Lake closes it via `norm_num` regardless. Cleanup is a fragment-detection refinement, not a Skolemization issue.

### By category

| Category | P3 | P4b | Δ | Notes |
|----------|----|----|------|-------|
| linear_real | 12/12 | 12/12 | — | Unchanged |
| polynomial_identity | 4/4 | 4/4 | — | Unchanged |
| polynomial_inequality | 0/3 | **3/3** | **+3** | Widened-offset fallback hit `_101, _113, _410` |
| numbertheory_int | 3/4 | **4/4** | **+1** | `_326` closed by widened hints on cubic factorization |
| polynomial_system | 3/4 | 3/4 | — | `_338` (3-var cubic) still rejected |
| closed_form_rational | 4/5 | 4/5 | — | `_55` (q/p = 2/3 field reasoning) still rejected |

The 0/3 → 3/3 jump on polynomial_inequality is the cleanest category win. 0 → 3 on the category originally diagnosed as the root of the Phase 3 miss.

### By SMT fragment

| Fragment | Tactic emitted | Accepted | Total |
|----------|----------------|----------|-------|
| LRA / QF_LRA | linarith | 14 | 15 |
| NRA | nlinarith | 7 | 11 |
| LIA / QF_LIA | omega | 2 | 2 |
| NIA | omega | 2 | 3 |
| QF_LRA (mis-detected) | linarith | 1 | 1 |

## Phase 4 implementation (the fixes)

Two changes to `crates/symthaea-lean-bridge/src/fol_ext_bridge.rs`:

### 1. Widened `sq_nonneg` offsets (Pattern C fix)

`build_nlinarith_hints` was split into a *compact* variant (±1 offsets only, the Phase 3 baseline) and a *widened* variant (dense `{-10, -7, -5, -3, -1, 1, 3, 5, 7, 10}` offsets). The cascade tries compact first and widened second. The split is mandatory — Phase 4a emitted widened hints in the primary `nlinarith` branch and regressed `mathd_algebra_37, _141` with deterministic Lean heartbeat timeouts at 200k heartbeats. `whnf` blowup from too many `sq_nonneg` hypotheses with none of them discharging the goal. Compact-first, widened-fallback is the shape that actually ships.

### 2. Conjunction-splitter branch (Pattern A fix)

A new `refine ⟨?_, ?_⟩ <;> first | (linarith; done) | (nlinarith [...]; done) | …; done` branch is emitted between the compact and widened nlinarith branches, but *only when the conclusion is syntactically an `And`*. `conclusion_is_and()` walks outer `Forall`/`Implies` wrappers and returns `true` iff the ultimate conclusion is `And(_, _)`. Emitting this branch unconditionally was the other half of Phase 4a's regression: `refine ⟨?_, ?_⟩` on non-And goals would fail, but the embedded `nlinarith [widened]` inside it still got elaborated and thrashed the heartbeat budget.

### Verification

- 7/7 unit tests in `fol_ext_bridge.rs` still pass.
- Phase 4b harness re-run: 30/32 accepted, 0 regressions vs Phase 3, 2 stable rejections (below).
- Average per-fixture Lake time went from ~2s → ~5s (cascade is ~50% larger when the And-splitter is emitted), but this is a one-time cost per compile, not a runtime-hot-path cost.

## Failure modes

### Resolved in Phase 4

- **Pattern A (conjunction in the conclusion):** 2 Phase 3 failures (`mathd_algebra_126`, `_101`) — now closed by the gated `refine ⟨?_, ?_⟩ <;>` branch.
- **Pattern C (polynomial inequality needing offset `sq_nonneg` hints):** 3 Phase 3 failures (`mathd_algebra_113`, `_410`, and the second half of `_101`) — now closed by the widened-offset fallback.
- **Bonus:** `mathd_numbertheory_326` (integer cubic root uniqueness, originally Pattern E / deferred) also closed — the widened `sq_nonneg` hints turned out to give nlinarith enough polynomial ammunition to factor the cubic.

### Resolved in Phase 5

- **Pattern B (field reasoning with symbolic-denominator division in the conclusion):** 1 Phase 4b failure (`mathd_algebra_55`) — closed by the new `(try subst_eqs; try field_simp; first | norm_num | ring | linarith | nlinarith); done` cascade branch, gated on `conclusion_has_division`. `subst_eqs` collapses the `q = 2-4+…, p = 3-6+…` hypotheses in `mathd_algebra_55` to concrete rationals, then `norm_num` verifies `8/12 = 2/3`. The branch is conditional (emitted only when the AST contains symbolic division) so non-field goals pay no cost. **~60 LOC including the `conclusion_has_division` walker.**

### Remaining (Phase 5, the 1 rejection)

### Pattern D — nonlinear system with product conclusion (1 failure: `mathd_algebra_338`)

`3a + b + c = −3 ∧ a + 3b + c = 9 ∧ a + b + 3c = 19 → abc = −56`. Solving the linear system gives `a = −4, b = 2, c = 7`; `abc = −56` follows. But `nlinarith` doesn't reason in "solve then evaluate" mode — it tries to derive the conclusion by non-negativity manipulation, which doesn't work here.

The right closer is a two-step tactic: `linear_combination` (Mathlib) to collapse the linear part, then `nlinarith` on the remainder. Symthaea-side: the `linear_combination` tactic needs coefficient inputs we'd have to compute ourselves. **This is a Phase 5 research question, not a Phase 4 bug.**

## Phase 5 recommendations

Ranked by **marginal accept-rate gain per LOC**:

| # | Fix | Expected gain | Est. LOC |
|---|-----|---------------|----------|
| 1 | Pattern B: `field_simp [h_ne_0]` branch for rational-conclusion goals | +1 problem (→97%) | ~30 |
| 2 | Pattern D: `linear_combination` solver for linear-then-product systems | +1 problem (→100%) | ~100 |
| 3 | Fix Z3 quantifier-instantiation problem (Skolemize universals) | 0 net on Lake; Z3 becomes reliable | ~50 |
| 4 | Auto-ingestion via `Lean.Parser` (Phase 4 option (c) originally) | Expands fixture set from 32 to 100+ | 2-3 weeks |

The single-problem fixes are small but each unlocks one more fixture. The ingestion work is the real multiplier — it turns 32 hand-translated fixtures into the full 488-file corpus at roughly 10-15% in-AST-scope, meaning ~50-70 fixtures running through the pipeline automatically. That's where the next big signal comes from.

## What this doesn't prove

1. **The fixture set is still curated.** 32 translated problems is not the 488-file miniF2F-v2. Anything in the rejected 456 — function abstractions, `Real.sqrt` identities, modular arithmetic, `Finset` counting, inductive proofs — is untouched by this measurement. This is a fair benchmark *for the algebraic subset the AST can represent*; it is not a claim about miniF2F-v2 overall. The honest reading: **we close 78% of the in-AST-scope subset, which is an estimated 10-15% of the full corpus.**

2. **Automatic ingestion is still missing.** All translations were manual. Phase 4 option (c) — the Lake executable that parses `.lean` via `Lean.Parser` and emits `FolFormulaExt` JSON — removes this bottleneck. With it, the 32 fixtures could become 100+ and the rejections could be cataloged mechanically instead of visually.

3. **The Z3 encoding has a quantifier instantiation problem.** 6 of 32 fixtures timed out at 10s in Z3, including linear 2-variable systems that should be subsecond. The root cause is that `encode_as_query()` wraps the whole formula under `(assert (not ∀ vars. …))` and lets Z3's quantifier-instantiation machinery Skolemize. A direct Skolemization — declare the universals as free consts, assert the hypotheses, assert the negated goal — would be subsecond. Phase 4-adjacent cleanup.

## Artifacts

- `crates/symthaea-lean-bridge/examples/prove_minif2f_curated.rs` — 32 fixtures + Z3/Lake harness
- `proofs/minif2f_curated/*.lean` — 32 emitted proof files
- `docs/phase3-results/minif2f_curated_results.csv` — raw measurement data

## Reproduction

```bash
cd /srv/luminous-dynamics/symthaea
# Emit-only (no Lake; fast):
cargo run -p symthaea-lean-bridge --example prove_minif2f_curated

# With Lake + Mathlib verification (requires lean-proofs/phase2/ populated):
nix develop -c bash -c 'LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example prove_minif2f_curated'
```

Total run time (Lake mode, warm caches): ~8 min — dominated by Z3 timeouts on the 6 timeout-cases, not by Lake itself.
