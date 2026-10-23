(set-logic QF_NRA)
; Ramanujan Protocol formal-verification obligation
; Problem: H = 1/2(px² + py²) + 1/2(x² + y²) + x²y - y³/3 (exact rational literals)
; Invariant (Lean-ready): (+ (+ (* (/ 1.0 2.0) (+ (* px px) (* py py))) (* (/ 1.0 2.0) (+ (* x x) (* y y)))) (+ (* (* x x) y) (- (* (/ 1.0 3.0) (* y (* y y))))))
; Claim: dE/dt = 0 identically.
; Z3 query: ∃px, py, x, y : dE/dt ≠ 0 (expected UNSAT).

(declare-const px Real)
(declare-const py Real)
(declare-const x Real)
(declare-const y Real)

(assert (not (= (+ (+ (+ (* (+ (* (/ 1.0 2.0) (* 2 x)) (* (* 2 x) y)) px) (* (+ (* (/ 1.0 2.0) (* 2 y)) (+ (* x x) (- (* (/ 1.0 3.0) (* 3 (* y y)))))) py)) (* (* (/ 1.0 2.0) (* 2 px)) (- (+ x (* 2 (* x y)))))) (* (* (/ 1.0 2.0) (* 2 py)) (- (+ y (+ (* x x) (- (* y y))))))) 0)))
(check-sat)
