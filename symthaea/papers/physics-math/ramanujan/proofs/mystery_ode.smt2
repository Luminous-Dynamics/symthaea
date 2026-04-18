(set-logic QF_NRA)
; Ramanujan Protocol formal-verification obligation
; Problem: H = ½(px² + py²) + x² + y² + xy (anisotropic coupled oscillator)
; Invariant (Lean-ready): (+ (* 0.5 (+ (* px px) (* py py))) (+ (+ (* x x) (* y y)) (* x y)))
; Claim: dE/dt = 0 identically.
; Z3 query: ∃px, py, x, y : dE/dt ≠ 0 (expected UNSAT).

(declare-const px Real)
(declare-const py Real)
(declare-const x Real)
(declare-const y Real)

(assert (not (= (+ (+ (+ (* (+ (* 2 x) y) px) (* (+ (* 2 y) x) py)) (* (* 0.5 (* 2 px)) (- (+ (* 2 x) y)))) (* (* 0.5 (* 2 py)) (- (+ (* 2 y) x)))) 0)))
(check-sat)
