(set-logic QF_NRA)
; Ramanujan Protocol formal-verification obligation
; Problem: 4E = 2v² + 2x² + x⁴ (Duffing oscillator, conservative; ×4 for integer coefs)
; Invariant (Lean-ready): (+ (+ (* 2 (* v v)) (* 2 (* x x))) (* x (* x (* x x))))
; Claim: dE/dt = 0 identically.
; Z3 query: ∃v, x : dE/dt ≠ 0 (expected UNSAT).

(declare-const v Real)
(declare-const x Real)

(assert (not (= (+ (* (+ (* 2 (* 2 x)) (* 4 (* x (* x x)))) v) (* (* 2 (* 2 v)) (- (+ x (* x (* x x)))))) 0)))
(check-sat)
