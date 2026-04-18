(set-logic QF_NRA)
; Ramanujan Protocol formal-verification obligation
; Problem: 2E = v1² + v2² + 2x1² + 2x2² − 2·x1·x2 (linear coupled oscillators, k=1; ×2)
; Invariant (Lean-ready): (+ (+ (* v1 v1) (* v2 v2)) (+ (+ (* 2 (* x1 x1)) (* 2 (* x2 x2))) (- (* 2 (* x1 x2)))))
; Claim: dE/dt = 0 identically.
; Z3 query: ∃v1, v2, x1, x2 : dE/dt ≠ 0 (expected UNSAT).

(declare-const v1 Real)
(declare-const v2 Real)
(declare-const x1 Real)
(declare-const x2 Real)

(assert (not (= (+ (+ (+ (* (+ (* 2 (* 2 x1)) (- (* 2 x2))) v1) (* (+ (* 2 (* 2 x2)) (- (* 2 x1))) v2)) (* (* 2 v1) (+ (- (* 2 x1)) x2))) (* (* 2 v2) (+ (- (* 2 x2)) x1))) 0)))
(check-sat)
