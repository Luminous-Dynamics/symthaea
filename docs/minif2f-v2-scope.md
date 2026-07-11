# miniF2F-v2: Scope & Status

> **2026-07-06 correction.** An earlier version of this document declared
> real-number arithmetic *out of scope* ("Phase 2, not yet done", "well under
> 2% accept rate"). **That is stale and wrong.** The arithmetic bridge has since
> been implemented through "Phase 6". This document now records the *verified*
> state and, importantly, the one gap that is actually real: **the emitted Lean
> is never checked by an actual Lean toolchain in this repo, so the accept rate
> is unmeasured.**

## What is actually built (verified against source, 2026-07-06)

The bridge is no longer propositional-only. Real-number / integer arithmetic is
implemented across two crates:

**`symthaea-core::hdc::fol_ext_smt`** — the arithmetic formula layer:
- `FolFormulaExt` with `Eq`, `Lt`, `Le`, implication, and quantifiers over typed
  binders (`NumericType::{Int, Real, Nat}`).
- `Term` with `Add`, `Mul`, `Pow`, `IntLit`, and exact rational literals
  (rendered `(1 : ℝ) / (3 : ℝ)`).
- `SmtFragment` (`QfLia`/`Lia`/`QfLra`/`Lra`/`QfNia`/`Nia`/`QfNra`/`Nra`) with
  `detect_fragment()` and `suggested_lean_tactic()`:
  LIA→`omega`, LRA→`linarith`, NRA→`nlinarith`, NIA→`omega_nat` (bounded).

**`symthaea-lean-bridge::fol_ext_bridge`** — the emitter:
- `render_fol_ext_file()` routes pure-propositional goals to the Phase-1 term
  synthesizer and arithmetic goals to `synthesize_arith_tactic()`.
- The arithmetic cascade emits real Mathlib:
  `first | rfl | norm_num | ring | omega | linarith | nlinarith [hints] |
  positivity | tauto | polyrith`, with:
  - **named-variable threading** (Phase 3) — concrete `intro x y`, and
    `sq_nonneg x` / `mul_self_nonneg y` hints using real names, not `_`.
  - **conjunction splitter** (Phase 4) — gated on `conclusion_is_and`.
  - **field-simp branch** (Phase 5) — gated on symbolic division, with
    `subst_eqs` / `field_simp`.
  - **`sub_ne_zero` witness derivation** (Phase 6a) — turns `x ≠ c` hypotheses
    into the `x - c ≠ 0` witnesses `field_simp` needs.

The representative problem the old doc called "unrepresentable"
(`mathd_algebra_206`, `a + b = -2` over ℝ) is now representable and routes to the
`linarith`/`nlinarith` cascade.

## What the unit tests actually prove — and what they don't

`fol_ext_bridge.rs` unit tests assert **emission shape**, not Lean acceptance:
- `linear_int_cascade_includes_omega`: `∀ n:ℤ, n < n+1` → output contains
  `omega`, no `sorry`.
- `nonlinear_real_cascade_includes_nlinarith`: `∀ x:ℝ, 0 ≤ x·x` → contains
  `nlinarith`, no `sorry`.
- rendering: unicode `≤`, exact rational `(1:ℝ)/(3:ℝ)`, implication `→`,
  Phase-1 vs arithmetic routing.

**These prove the emitter picks the right Mathlib tactic for the fragment and
produces well-formed ℝ/ℤ Lean syntax. They do NOT prove the emitted proof
type-checks and closes the goal.** A perfectly-shaped `nlinarith [...]` cascade
can still fail inside Lean. This is the Phase 0 grounding lesson one level up:
the tests verify *output shape*, not *output correctness*.

## The one real gap: no external verification

- **Lean toolchain is absent in this environment** (`lean`/`lake` not on PATH),
  so `runner::check_with_lean4` and the `prove_minif2f*` examples cannot run
  here.
- **No committed, reproducible accept-rate artifact exists.** The "Phase 3/4/5/6
  measurement" claims live only in code comments; there is no results CSV in the
  crate. The true miniF2F-v2 accept rate is therefore **currently unknown /
  unverified in-repo**, in either direction.

## Concrete next step to actually close Phase 1

The arithmetic *synthesis* is done; the missing work is *verification*, not
capability:

1. Provision Lean 4 + Mathlib (a `nix develop` devShell input or a pinned
   `elan`/`lake` toolchain). This is the blocker — everything else exists.
2. Run the committed examples against the corpus and **commit the results CSV**:
   ```bash
   MINIF2F_V2_DIR=/path/to/miniF2F/lean4 \
     cargo run -p symthaea-lean-bridge --example prove_minif2f_v2 \
     > minif2f_v2_results.csv
   ```
   (`prove_minif2f_curated`, `prove_minif2f`, and `prove_fol_arith` also exist;
   `prove_fol_arith` already optionally shells out to `lake env lean`.)
3. Add a **Lean-verified gate** — the analog of the Phase 0
   `SolverCorrectnessGate` — that runs a small fixed battery of arithmetic goals
   through the real Lean toolchain and asserts they close `sorry`-free. Gate it
   behind Lean availability so CI without Lean skips rather than fails.

Until step 2 lands a committed accept rate, describe this capability as
**"real-number arithmetic proof synthesis, implemented but externally
unverified"** — not as a percentage.
