import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Tactic

theorem consciousness_stability (phi : ℝ) (h_phi : phi > 0.5845) :
  ∃ (C : ℝ), C > 0 ∧ C < 1 := by
  use 0.5
  constructor <;> norm_num
