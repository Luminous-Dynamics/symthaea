import Lean

namespace ScopeEquivalenceTier2AndComm

/-- Forward context entailment: the local context supplies the global context. -/
theorem local_context_to_global (p q : Prop) :
    p ∧ q → q ∧ p := And.symm

/-- Reverse context entailment: the global context supplies the local context. -/
theorem global_context_to_local (p q : Prop) :
    q ∧ p → p ∧ q := And.symm

/-- Two-way context equivalence, assembled from the explicit directions. -/
theorem context_equivalence (p q : Prop) :
    (p ∧ q) ↔ (q ∧ p) := by
  exact Iff.intro
    (local_context_to_global p q)
    (global_context_to_local p q)

/-- Transport a warrant proved under the local context to the reordered context. -/
theorem warrant_local_to_global (p q r : Prop) :
    (p ∧ q → r) → (q ∧ p → r) := by
  intro local_warrant global_assumptions
  exact local_warrant (And.symm global_assumptions)

/-- Transport a warrant proved under the reordered context back to the local context. -/
theorem warrant_global_to_local (p q r : Prop) :
    (q ∧ p → r) → (p ∧ q → r) := by
  intro global_warrant local_assumptions
  exact global_warrant (And.symm local_assumptions)

/-- Bidirectional warrant transport for the exact same target r. -/
theorem warrant_transport_equivalence (p q r : Prop) :
    (p ∧ q → r) ↔ (q ∧ p → r) := by
  exact Iff.intro
    (warrant_local_to_global p q r)
    (warrant_global_to_local p q r)

end ScopeEquivalenceTier2AndComm

#check ScopeEquivalenceTier2AndComm.local_context_to_global
#check ScopeEquivalenceTier2AndComm.global_context_to_local
#check ScopeEquivalenceTier2AndComm.context_equivalence
#check ScopeEquivalenceTier2AndComm.warrant_local_to_global
#check ScopeEquivalenceTier2AndComm.warrant_global_to_local
#check ScopeEquivalenceTier2AndComm.warrant_transport_equivalence

#print axioms ScopeEquivalenceTier2AndComm.local_context_to_global
#print axioms ScopeEquivalenceTier2AndComm.global_context_to_local
#print axioms ScopeEquivalenceTier2AndComm.context_equivalence
#print axioms ScopeEquivalenceTier2AndComm.warrant_local_to_global
#print axioms ScopeEquivalenceTier2AndComm.warrant_global_to_local
#print axioms ScopeEquivalenceTier2AndComm.warrant_transport_equivalence
