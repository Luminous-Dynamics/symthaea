# SMT-LIB2 Proof Witnesses (Phase 1: currently empty)

This directory is reserved for SMT-LIB2 proof obligations corresponding to each `PROVEN` row in the paper's results table.

## Phase 1 state

**Empty.** The `conjecture_engine` in `symthaea-core` calls Z3 as a subprocess (see `conjecture_engine.rs:3157 auto_prove_via_z3`) and pipes the SMT-LIB2 via stdin without saving to disk. Committing witnesses requires a small instrumentation change — Phase 2 work.

## What lands here in Phase 2

For each `PROVEN` row, one file named `<problem>_<invariant>.smt2` containing:

```smt2
(set-logic QF_NRA)
; Problem: Harmonic oscillator
; Discovered invariant: C(x,v) = x^2 + v^2
; Dynamics: dx/dt = v, dv/dt = -x
; Obligation: dC/dt = 0

(declare-const x Real)
(declare-const v Real)
(declare-const dx_dt Real)
(declare-const dv_dt Real)
(assert (= dx_dt v))
(assert (= dv_dt (- x)))

; dC/dt = 2x·(dx/dt) + 2v·(dv/dt)
(assert (not (= (+ (* 2 x dx_dt) (* 2 v dv_dt)) 0)))

(check-sat)
; expected: unsat
```

A reader with any SMT-LIB2 solver (Z3, CVC5, MathSAT, etc.) can re-check each file independently.

## Verification without witness files (Phase 1 workaround)

Until Phase 2 lands, use `../reproduce.sh` — it re-invokes Z3 on the reader's host with the same SMT-LIB2 the engine built internally. Provided Z3 is installed and agrees with ours on `unsat`, this is equivalent to the per-file check.
