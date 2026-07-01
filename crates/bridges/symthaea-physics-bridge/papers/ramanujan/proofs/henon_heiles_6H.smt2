(set-logic QF_NRA)
; Ramanujan Protocol formal-verification obligation
; Problem: 6H = 3(px² + py²) + 3(x² + y²) + 6x²y − 2y³ (Hénon-Heiles, ×6 to avoid f64/rational mismatch)
; Invariant (Lean-ready): (+ (+ (* 3 (+ (* px px) (* py py))) (* 3 (+ (* x x) (* y y)))) (+ (* 6 (* (* x x) y)) (- (* 2 (* y (* y y))))))
; Claim: dE/dt = 0 identically.
; Z3 query: ∃px, py, x, y : dE/dt ≠ 0 (expected UNSAT).

(declare-const px Real)
(declare-const py Real)
(declare-const x Real)
(declare-const y Real)

(assert (not (= (+ (+ (+ (* (+ (* 3 (* 2 x)) (* 6 (* (* 2 x) y))) px) (* (+ (* 3 (* 2 y)) (+ (* 6 (* x x)) (- (* 2 (* 3 (* y y)))))) py)) (* (* 3 (* 2 px)) (- (+ x (* 2 (* x y)))))) (* (* 3 (* 2 py)) (- (+ y (+ (* x x) (- (* y y))))))) 0)))
(check-sat)
