(set-logic QF_NRA)
; Ramanujan Protocol formal-verification obligation
; Problem: L = x·vy − y·vx (2D isotropic harmonic oscillator)
; Invariant (Lean-ready): (+ (* x vy) (- (* y vx)))
; Claim: dE/dt = 0 identically.
; Z3 query: ∃vx, vy, x, y : dE/dt ≠ 0 (expected UNSAT).

(declare-const vx Real)
(declare-const vy Real)
(declare-const x Real)
(declare-const y Real)

(assert (not (= (+ (+ (+ (* vy vx) (* (- vx) vy)) (* (- y) (- x))) (* x (- y))) 0)))
(check-sat)
