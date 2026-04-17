(set-logic QF_NRA)
; Ramanujan Protocol formal-verification obligation
; Problem: E = x² + v²; dx/dt = v, dv/dt = -x
; Invariant (Lean-ready): (+ (* x x) (* v v))
; Claim: dE/dt = 0 identically.
; Z3 query: ∃v, x : dE/dt ≠ 0 (expected UNSAT).

(declare-const v Real)
(declare-const x Real)

(assert (not (= (+ (* (* 2 x) v) (* (* 2 v) (- x))) 0)))
(check-sat)
