(set-logic QF_NRA)
; Ramanujan Protocol formal-verification obligation
; Problem: 2E = vx² + vy² + x² + y² (2D isotropic harmonic; ×2)
; Invariant (Lean-ready): (+ (+ (* vx vx) (* vy vy)) (+ (* x x) (* y y)))
; Claim: dE/dt = 0 identically.
; Z3 query: ∃vx, vy, x, y : dE/dt ≠ 0 (expected UNSAT).

(declare-const vx Real)
(declare-const vy Real)
(declare-const x Real)
(declare-const y Real)

(assert (not (= (+ (+ (+ (* (* 2 x) vx) (* (* 2 y) vy)) (* (* 2 vx) (- x))) (* (* 2 vy) (- y))) 0)))
(check-sat)
