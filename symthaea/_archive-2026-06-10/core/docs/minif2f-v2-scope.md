# miniF2F-v2: Phase 1 Scope Analysis

**Bottom line:** Symthaea's current Lean 4 bridge targets propositional and first-order classical logic. **miniF2F-v2 is almost entirely algebra, arithmetic, and number theory over ℝ, ℕ, and ℤ** — outside our current scope. This document explains the gap precisely so the Phase 1 → Phase 2 decision is made with open eyes.

## What the bridge can prove today

The `symthaea-lean-bridge` crate's `synthesize_proof_term` closes classical propositional tautologies up to arbitrary nesting of ∧, ∨, →, ¬, ↔, ⊤, ⊥. The W4 tautology suite (`prove_proptauts`, 23 fixtures) demonstrates this with **23/23 Lean-accepted, 0 `sorry`**. The synthesizer handles:

- Identity, K, and S-style combinators
- Curry/uncurry between `A → B → C` and `A ∧ B → C`
- Nested conjunction projection (`h.1.2.1` etc.) via `collect_projections`
- Multi-argument `h a₁ a₂ … aₙ` application via `try_curry_apply`
- Classical excluded middle (`A ∨ ¬A`) via `Classical.em`
- Ex falso (`False → P`) via `False.elim`

No Mathlib dependency. All proofs are term-mode in core Lean 4.

## What miniF2F-v2 demands

A representative miniF2F-v2 problem (`mathd_algebra_206`):

```lean
theorem mathd_algebra_206 (a b : ℝ)
    (f : ℝ → ℝ → ℝ)
    (h₀ : ∀ x, f x b = x^2 + b * x + 1)
    (h₁ : ∀ y, f a y = y^2 + a * y + 1)
    : a + b = -2 := by sorry
```

None of this is propositional. It requires:

1. **Real-number arithmetic**: `x^2`, `b * x`, `a + b = -2`.
2. **Universally-quantified function equations**: `∀ x, f x b = …`.
3. **Algebraic manipulation**: deriving `a = b` from `f x b = f a y` pattern-matching.
4. **Linear arithmetic on the goal**: `a + b = -2` is a linear inequality Z3's `QF_LRA` can decide (after the preceding algebraic rearrangement).

Our `Proposition` enum doesn't include `Eq`, `Mul`, `Pow`, or quantification over functions — making the problem statement unrepresentable, let alone provable.

## Scope spectrum

miniF2F-v2 problems, as a rough reading of the public set:

| Category | Fraction (est.) | Our Phase 1 reach |
|----------|----------------|-------------------|
| Algebra over ℝ (polynomials, linear/quadratic manipulation) | ~55% | None |
| Number theory over ℕ, ℤ (gcd, Fermat's little, factorizations) | ~25% | None |
| Inequalities (AM-GM, Cauchy-Schwarz, Jensen) | ~10% | None |
| Combinatorics (finite counting, Pigeonhole) | ~5% | None |
| Pure propositional / FOL | ~1-2% | **All** |

A generous estimate of our Phase 1 accept rate: **well under 2%**. Most of that 2% would also require parsing infrastructure we don't yet have: miniF2F statements are written in Lean 4, and we'd need to map them onto `symthaea_core::hdc::logic_engine::Proposition` before `tactics_for_goal` could even see them.

## What's needed to actually attack miniF2F-v2

1. **Extend `Proposition` (or a new `FOLFormula+`)** to include `Eq`, arithmetic operators, and domain-specific types (ℝ, ℕ, ℤ). This is several KLOC.
2. **Lean 4 parser** for the subset of Lean 4 syntax miniF2F uses. Even a restricted parser is a real undertaking — roughly the size of `logic_engine.rs`.
3. **Arithmetic decision procedures** to dispatch to. Z3 (`QF_LRA`, `QF_LIA`, `QF_NIA`) closes many algebra goals; hooking the existing `conjecture_engine::auto_prove_via_z3` into the Lean bridge is the most direct path.
4. **Term synthesis for arithmetic proofs**: Z3 typically returns `unsat` without a proof term; translating that into a Lean proof requires either (a) Mathlib's `linarith`/`nlinarith` tactics (which need Mathlib as a dependency) or (b) extracting a Positivstellensatz certificate from Z3 and reconstructing a Lean proof manually.

All four items together are a Phase 2 scope. Best-case timeline: one fully-focused sprint (4–6 weeks) gets item (1) + partial (2) + (3) into a working state with `linarith`-style Mathlib tactics. Target realistic accept rate post-Phase-2: 15–30% of miniF2F-v2.

## Honest decision-point framing

At the week-10 review, the artifact to show is:

- **Proposition level:** strong (23/23 strict, 100% on classical propositional tautologies).
- **miniF2F-v2:** architecturally out-of-scope at Phase 1; 0/X accepted.

That second number isn't a failure of the pipeline — it's a scope declaration. The right follow-up question is whether Phase 2 should prioritize the `linarith`-bridge path (algebraic mass production on benchmarks) or stay focused on propositional/FOL depth (novel theorem discovery via conjecture_engine's Z3 integration). These two paths don't conflict, but they have different staffing profiles, and one should lead.

## Placeholder harness

The `prove_minif2f_v2` example, when `MINIF2F_V2_DIR` is set, iterates the downloaded corpus and reports which problems the **bridge's statement representation can accept** (a strict upper bound on Phase 1 achievable accept rate). This number is expected to be near zero and should be read as a scope signal, not a quality signal.

```bash
# Install miniF2F-v2 first, e.g.:
#   git clone https://github.com/openai/miniF2F /tmp/miniF2F
#
# Then:
MINIF2F_V2_DIR=/tmp/miniF2F/lean4 \
    cargo run -p symthaea-lean-bridge --example prove_minif2f_v2 \
    > minif2f_v2_results.csv
```
