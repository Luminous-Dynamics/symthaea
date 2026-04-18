(set-logic QF_NRA)
; Ramanujan Protocol formal-verification obligation
; Problem: 4E = 2v² + x⁴ (quartic anharmonic oscillator; ×4 for integer coefs)
; Invariant (Lean-ready): (+ (* 2 (* v v)) (* x (* x (* x x))))
; Claim: dE/dt = 0 identically.
; Z3 query: ∃v, x : dE/dt ≠ 0 (expected UNSAT).

(declare-const v Real)
(declare-const x Real)

(assert (not (= (+ (* (* 4 (* x (* x x))) v) (* (* 2 (* 2 v)) (- (* x (* x x))))) 0)))
(check-sat)
