# Phase 5 — Final Two miniF2F Rejections + Expansion Paths

**Status:** plan draft. 2026-04-18.

## Starting position

Phase 3 + Phase 4 landed 30/32 = 93.8% Lake accept on the hand-curated miniF2F-v2 subset. See `docs/phase3-findings.md` for the full arc. Two rejections remain; both are single-problem but require different machinery.

## The two rejections

| Fixture | Category | Why it fails | Phase 5 fix |
|---------|----------|--------------|-------------|
| `mathd_algebra_55` | closed_form_rational | `q/p = 2/3`: goal contains division by a free variable; `linarith`/`nlinarith` don't handle fields | Pattern B: `field_simp [hp_ne_zero] + ring` |
| `mathd_algebra_338` | polynomial_system | `3a+b+c=-3, a+3b+c=9, a+b+3c=19 → abc=-56`: needs linear solve *then* multiplicative evaluation — nlinarith's Positivstellensatz search doesn't reason this way | Pattern D: `linear_combination <coefficients> + nlinarith` |

## Pattern B — `field_simp` branch (~30 LOC)

### Problem shape

Any goal of the form `expr_with_division = literal_rational` where one or more denominators are *bound variables* that cannot be syntactically recognized as nonzero. `mathd_algebra_55` is the single example in our fixture set, but the pattern generalizes to any `q/p = r/s` with symbolic denominators.

### Tactic

Mathlib's `field_simp` clears denominators given nonzero-witness hypotheses. The `[h1, h2, …]` argument list takes nonzero hypotheses:

```lean
field_simp [hp_ne_zero, hq_ne_zero]
ring  -- or nlinarith, depending on the rearranged goal
```

### What we need to add

1. **Gate: detect field-goal shape.** New helper `conclusion_has_division(phi)` walks the conclusion after stripping `Forall`/`Implies` and checks for any `Term::BinOp(Div, _, b)` where `b` contains a free variable. Roughly 15 LOC.

2. **Hypothesis scan: collect nonzero witnesses.** Walk the hypothesis chain (everything on the left of the outermost `Implies` chain) and collect `h_ne` names wherever a hypothesis shape matches `Not(Eq(var, 0))`. In Symthaea's AST, `Not(Eq(Var("p"), IntLit(0)))`. The bridge emits `intro` names for these; we need to thread them back as `field_simp` arguments.

3. **Cascade branch.** Add a new alternative after `nlinarith [compact]` but before the And-splitter:

   ```
   | (field_simp [{nonzero_hyps}] <;> first | (ring; done) | (linarith; done) | (nlinarith [{hints_compact}]; done); done)
   ```

   If `nonzero_hyps` is empty, emit `field_simp` with no arguments — it'll still succeed on closed-form rationals.

### Expected cost

~30 LOC in `fol_ext_bridge.rs`: 1 helper, 1 hypothesis scan, 1 cascade branch with conditional arg-list emission. Unit tests adding 2 fixtures (`q/p = 2/3` plus a closed-form `1/3 + 1/4 = 7/12` type). No new Mathlib dep (field_simp is already available through `Mathlib.Tactic`).

### Risk

`field_simp` can *partially* simplify the goal without closing it, a known issue with our `; done`-terminated cascade design (we learned this lesson with `norm_num` in Phase 2 W4). The `; done` termination should still catch this correctly, but worth re-running the full fixture set to confirm nothing regresses.

### Gain

+1 problem (→31/32 = 96.9%).

## Pattern D — `linear_combination` solver (~100 LOC)

### Problem shape

Systems where the conclusion is a *function* of values obtained by solving a *linear sub-system* in the hypotheses. `mathd_algebra_338` is the canonical example: three linear equations determine `a = −4, b = 2, c = 7`, then the conclusion is the product `abc = −56`.

`nlinarith` can't do this because it tries non-negativity-based Positivstellensatz search, not linear solving + substitution. Mathlib's `linear_combination` tactic takes an explicit linear combination of hypotheses that rearranges to the goal:

```lean
linear_combination 2 * h₀ + h₁ - 3 * h₂
```

### What we need to add

Here's where it gets harder — `linear_combination` needs the *coefficients*. For a goal `g(a, b, c) = v` and hypotheses `h_i : l_i(a, b, c) = r_i`, we'd need to solve a linear system to find the coefficients `c_i` such that `sum c_i · (l_i − r_i) = g(a, b, c) − v`.

Two paths:

**Path D.1 — explicit solve via gaussian elimination.** Extract the linear parts of the hypotheses and goal, build a matrix, RREF it, emit the coefficients. Much heavier — roughly what a CAS does. ~100+ LOC.

**Path D.2 — use Mathlib's `polyrith` instead.** `polyrith` is already in our cascade (last alternative). It *should* be able to close `mathd_algebra_338` via Gröbner basis. The fact that it didn't in our Phase 3 measurement suggests either:
- `polyrith` hit a resource limit (`polyrith` can be slow),
- the cascade ordering means earlier branches simplified the goal in a way that broke `polyrith`,
- `polyrith` needs explicit coefficients like `linear_combination` does.

**Recommendation: try D.2 first.** Give `polyrith` more heartbeat budget explicitly in the emitter (`set_option maxHeartbeats 400000 in polyrith`). If that doesn't close 338, fall back to D.1.

### Expected cost

If D.2 works: ~20 LOC (bump heartbeat budget for the `polyrith` branch, or introduce a separate late-cascade `polyrith_strong` alternative). If D.1 is needed: 100+ LOC including a linear-system extractor and RREF.

### Gain

+1 problem (→32/32 = 100%) if it lands.

## Phase 5 expansion paths (beyond the 2 fixtures)

If Patterns B + D both land, the 32-fixture set is saturated. The next levers:

### 5a — Fix Z3's quantifier-instantiation problem

Current `encode_as_query()` wraps `(assert (not (forall vars. hyps → goal)))`. Z3's quantifier instantiation machinery times out on trivial linear goals. Replace with direct Skolemization: declare universals as free constants, assert hypotheses, assert the negated goal. Z3 then decides pure QF_LRA/QF_LIA/QF_NRA.

Affects 5 of 32 Phase 4b fixtures (Z3 timeout; Lake accepts them anyway). Won't change Lake accept rate, but makes Z3 reliable as a pre-filter and enables real-time conjecture testing against Z3.

~50 LOC in `symthaea-core/src/hdc/fol_ext_smt.rs`.

### 5b — Lean metaprogramming ingestion (the big multiplier)

Phase 2 scoping doc's option (c). Write a Lake executable in Lean that parses `.lean` files via `Lean.Parser` and emits JSON matching `FolFormulaExt`. This removes the hand-translation bottleneck and expands the fixture set from 32 to ~50-70 in-AST-scope problems (roughly 10-15% of the 488-file corpus).

2-3 weeks. Highest-leverage Phase 5 work — turns an isolated 93.8% on 32 fixtures into a continuously-measurable accept rate against the full in-scope slice.

### 5c — AST extension (the ceiling work)

Past 10-15% of miniF2F-v2, we need `abs`, `mod`, `Finset`, function abstraction, and more in `FolFormulaExt`. Each is a significant type-system extension. Scope and value TBD.

## Ranked plan

1. **Pattern B (`field_simp`)** — ~30 LOC → 96.9%
2. **Pattern D.2 (`polyrith` with bigger budget)** — ~20 LOC → possibly 100%
3. **5a (Z3 Skolemization)** — ~50 LOC → no Lake delta, better Z3 oracle
4. **5b (Lean ingestion)** — 2-3 weeks → fixture set 32 → ~60, continuous measurement
5. **5c (AST extension)** — scoping work, deferred

Items 1 and 2 are single-sitting fixes. Item 5b is where the real research value sits.

## What Phase 5 does NOT cover

- No new goal shapes beyond the 2 existing failures — this is a *close the known gaps* plan, not a scope expansion.
- No benchmark beyond miniF2F-v2 — PutnamBench and IMO-style problems are Phase 6.
- No AST overhaul (5c is scoped here but deferred in time).
