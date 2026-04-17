# Phase 2 Option A — Algebraic Reasoning (Technical Scoping)

**Status:** scoping draft. Sibling doc to `phase1-decision-memo.md`. User approval required before implementation; this file describes *what* would be built if the user chooses path A.

**Why it matters:** Phase 1's Lean bridge closes classical propositional tautologies at 100% (23/23). miniF2F-v2's upstream 490 files land at 0/490 in-scope. The gap is not pipeline quality — it's AST expressiveness. Every miniF2F problem involves real-number equality, integer arithmetic, or function-valued quantification. `Proposition` has none of those. This doc describes the minimum additions that would take our in-scope count from 0% toward a realistic Phase 2 target of 15–30%.

## Estimated effort

- **Minimum viable:** 4 weeks focused (one engineer, one context). Delivers a working `linarith`-style bridge for the algebra subset of miniF2F-v2.
- **Comfortable:** 6 weeks. Adds non-linear arithmetic via `QF_NIA` + `polyrith` and a Mathlib-aware CI lane.
- **Stretch:** 8 weeks. Adds the Lean-metaprogramming ingestion layer (see §5) so theorems are ingested automatically via Lean's own parser rather than hand-translated.

## Scope matrix — what's in and what's out

| Topic | Phase 2 Option A | Deferred (Phase 3+) |
|-------|------------------|---------------------|
| Equality (`=`) over ℝ, ℤ, ℕ | ✅ | — |
| Linear arithmetic (+, −, ×const) | ✅ | — |
| Polynomial arithmetic (finite degree) | ✅ | — |
| Non-linear Z3 via `QF_NIA`/`QF_NRA` | ✅ (bounded by Z3's practical reach) | unbounded non-linear |
| Induction over ℕ | 🟡 partial (via Z3's `:induction` pragma where it works) | full strong induction |
| Ordering (`<`, `≤`) | ✅ | — |
| Universal quantification over bounded integers | ✅ (unrolling) | unbounded `∀ n : ℕ` |
| Set theory (`Finset`, `Set`) | ❌ | ✅ |
| Algebraic structures (`Ring`, `Field`, `Group`) | ❌ | ✅ |
| Calculus (`deriv`, `integral`) | ❌ | ✅ |
| Number theory lemmas (`Nat.prime`, `gcd`) | ❌ | ✅ |

This matches the "easy miniF2F problems" criterion: problems that reduce to linear or low-degree polynomial arithmetic over ℝ/ℤ after a few symbolic manipulations.

## Proposed module changes

### 1. Extend `Proposition` → introduce `FolFormulaExt`

Current `Proposition` (at `symthaea-core/src/hdc/logic_engine.rs:26-47`) is purely boolean. Adding arithmetic directly to that enum would break every existing user. Cleaner: introduce `FolFormulaExt` as a sibling type that subsumes the existing `Proposition` via a `Base(Proposition)` variant.

```rust
// New file or co-located: symthaea-core/src/hdc/fol_formula_ext.rs
pub enum Term {
    Var(String),
    IntLit(i64),
    RealLit(f64),
    BinOp(ArithOp, Box<Term>, Box<Term>),   // Add, Sub, Mul, Div, Pow
    Neg(Box<Term>),
}

pub enum FolFormulaExt {
    Base(Proposition),                       // inherits Phase 1 coverage
    Eq(Term, Term),
    Lt(Term, Term),
    Le(Term, Term),
    And(Box<FolFormulaExt>, Box<FolFormulaExt>),
    Or(Box<FolFormulaExt>, Box<FolFormulaExt>),
    Not(Box<FolFormulaExt>),
    Implies(Box<FolFormulaExt>, Box<FolFormulaExt>),
    Forall(String, NumericType, Box<FolFormulaExt>),
    Exists(String, NumericType, Box<FolFormulaExt>),
}

pub enum NumericType { Int, Real, Nat }
```

Size estimate: ~600 LOC including tests.

### 2. Term → SMT-LIB2 serializer

Extend `expr_to_smtlib2` (conjecture_engine.rs:4447) to handle `FolFormulaExt`. The existing polynomial subset already works; add:
- Integer vs Real type inference (declare-const Int vs Real)
- Handling of `Lt`/`Le` (map to SMT `<` `<=`)
- Handling of quantifiers via SMT `forall`/`exists` (QF_LIA → LIA for quantifier support)
- Detection of fragment (QF_LRA if pure linear real, QF_LIA if pure linear integer, QF_NRA if polynomial real, etc.) to drive tactic choice.

Size estimate: ~300 LOC; lots of unit tests against known Z3-decidable inputs.

### 3. Bridge extension: `tactics_for_fol_ext`

Add a sibling to `synthesize_proof_term` that handles `FolFormulaExt`. Strategy:
- If pure propositional → delegate to existing `synthesize_proof_term`.
- Otherwise → emit `by linarith` or `by omega` or `by nlinarith` depending on SMT fragment.
  - These are **Mathlib tactics**. The emitted file needs `import Mathlib.Tactic.Linarith`.
  - This means the Phase 2 Lean CI lane needs Mathlib installed (adds ~5 min to cold CI, ~20s to warm).
- For quantifier-free problems Z3 can close, emit a `sorry` with a `-- Z3 unsat witness: proofs/<name>.smt2` comment. The SMT witness IS the proof; the Lean `sorry` is a placeholder acknowledging that Lean-native reconstruction is deferred.
- For problems needing induction: emit `induction n with …` scaffolding + `linarith` on each case.

Size estimate: ~400 LOC plus Mathlib-aware CI infra.

### 4. Lean-side project setup

New directory `lean-proofs/phase2/` with:
- `lakefile.lean` importing Mathlib (pin a specific commit for reproducibility).
- `lean-toolchain` pinned to a Mathlib-compatible version.
- Auto-emission target: `cargo run --example prove_minif2f_v2 --release` writes `.lean` files under this directory; `lake build` compiles them.
- CI lane uses `lake build` for the Mathlib-dependent proofs and direct `lean <file>` for the existing core-Lean proofs. Two lanes, clean separation.

This is the biggest infrastructural lift. Mathlib's full elaboration is ~10 min cold; warm cache is <1 min.

### 5. miniF2F ingestion — Lean metaprogramming, NOT a custom parser

**User decision (April 17):** do NOT write a recursive-descent parser for Lean 4. Lean 4's syntax is highly extensible and changes across versions; any custom parser would be a maintenance trap.

**Better approach** — use Lean 4's own metaprogramming API:

1. Write a small Lake executable in Lean that uses `Lean.Parser` to parse a `.lean` file (guaranteed-accurate because it's the same parser Lean uses itself).
2. The executable walks the resulting `Syntax` tree and emits either an S-expression or JSON that mirrors our `FolFormulaExt` shape.
3. Symthaea's Rust side invokes this as a subprocess and deserializes into `FolFormulaExt`.

Advantages:
- Zero custom parsing in Rust (~800 LOC saved vs the previous plan).
- Syntactic accuracy matches whatever Lean version we pin via `lean-toolchain`.
- Same mechanism scales to PutnamBench and future benchmarks without incremental parser work.

**Deferred to a later sprint.** Phase 2 MVP (weeks 1–4) measures accept rate against a hand-translated ~50-problem curated miniF2F subset. Full auto-ingestion via the Lean executable is the 8-week stretch.

## Measurement plan

Each of these becomes a CSV column after Phase 2 lands:

- `in_scope_per_filter`: our parser accepts the statement (if the parser stretch is done).
- `smt_fragment`: QF_LIA / QF_LRA / QF_NIA / QF_NRA / LIA / NIA / unsupported.
- `z3_result`: unsat / sat / unknown / timeout (10s budget).
- `lean_accepted`: `lean <file>` returns zero exit (needs Mathlib if `smt_fragment` non-trivial).
- `lean_sorry_free`: the `.lean` file has zero `sorry` occurrences.

Target at Phase 2 close:
- ≥ 30% `z3_result=unsat` on miniF2F-v2 (realistic).
- ≥ 15% `lean_sorry_free` (harder, gated by Mathlib tactic fidelity).

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Z3 timeouts on hard polynomial problems | High | Per-query budget of 10s; `sat`/`unknown` reported honestly, not retried to fake success |
| Mathlib version churn breaking proofs | Medium | Pin Mathlib commit in `lakefile.lean`; refresh quarterly |
| Parser edge cases (notation, unicode) | High | Explicitly scope the 60–70% supported subset; fail-fast on unsupported syntax |
| Induction handling is brittle | Medium | Restrict to strong-induction-via-Z3 cases; escalate to `sorry` otherwise |
| "linarith" loses on honest non-linear | Medium | Cascade: try `linarith` → `nlinarith` → `polyrith` → `sorry` |

Everything above is a known-bounded risk — nothing in the architecture requires new research.

## What stays the same

All Phase 1 infrastructure is preserved:
- `symthaea-core::hdc::logic_engine::Proposition` untouched.
- `proofs/proptauts/*.lean` untouched.
- `papers/ramanujan/*` untouched (paper already submitted, in theory, by this point).
- `MODULE_STATUS.md`'s Lean-propositional row stays at 23/23.

Phase 2 adds a second row for "Lean-FOL-arithmetic" with its own per-problem accept rate.

## Go / no-go checkpoints

- **Week 1 end:** `FolFormulaExt` + SMT serializer compiled, 10 hand-built unit tests pass.
- **Week 2 end:** First end-to-end miniF2F-v2 problem closes with Mathlib `linarith`.
- **Week 3 end:** 30 problems close; rejection reasons catalogued for iteration.
- **Week 4 end (MVP):** ≥ 15% accept rate on the full 490-file corpus. Decision: continue to 6-week comfortable scope or declare Phase 2 shipped.

## Decision requested from user

1. **Approve Option A.** I start on week 1 with the `FolFormulaExt` type definitions. Hard gate at week 4: if the MVP doesn't pass, we pause and re-scope.
2. **Skip Option A for now, pivot to something else** (Option B polish, Option C research sprint, or a different direction).
3. **Defer decision.** Phase 1 stands on its own; revisit Phase 2 scoping later.

Whichever you pick, the Phase 1 artifacts (MODULE_STATUS.md, 9 formal proofs, 23 propositional proofs, compiled paper, benchmark CSVs) are already in the repo and stable.
