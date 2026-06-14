/-- A k-tuple is admissible if for every prime p, 
   there is at least one residue class modulo p not occupied by the tuple. -/
def IsAdmissible (tuple : Finset ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → ∃ r : ℕ, r < p ∧ ∀ h ∈ tuple, h % p ≠ r

/-- The Twin Prime Conjecture as a formal statement stub. -/
axiom twin_prime_conjecture : ∃ᶠ (n : ℕ), n.Prime ∧ (n + 2).Prime

/-- Bounded Gaps Theorem (Zhang/Maynard-Tao) as a formal statement stub. -/
axiom bounded_gaps_theorem : ∃ h : ℕ, 0 < h ∧ ∃ᶠ (n : ℕ), n.Prime ∧ (n + h).Prime ∧ h ≤ 246
