import Lean

/--
A bounded Lean-kernel pilot for the provisional ProofPlanV1 `And.comm` plan.

The declarations below certify the propositions attached to the three formal
plan actions. They do not certify the provisional natural-language role labels
or prove that this plan is the unique/canonical explanation of `And.comm`.
-/
namespace ProofPlanActionCertificatePilot

/-- A `split_iff` action is obligation-preserving: the parent iff goal is
logically equivalent to the conjunction of its two directional subgoals. -/
theorem and_comm_split_iff (a b : Prop) :
    (a ∧ b ↔ b ∧ a) ↔ ((a ∧ b → b ∧ a) ∧ (b ∧ a → a ∧ b)) := by
  constructor
  · intro h
    exact ⟨h.mp, h.mpr⟩
  · intro h
    exact ⟨h.1, h.2⟩

/-- Local certificate for ProofPlan node `A_FORWARD`. -/
theorem and_comm_forward (a b : Prop) :
    a ∧ b → b ∧ a := by
  intro h
  exact And.intro h.right h.left

/-- Local certificate for ProofPlan node `A_REVERSE`. -/
theorem and_comm_reverse (a b : Prop) :
    b ∧ a → a ∧ b := by
  intro h
  exact And.intro h.right h.left

/-- Root theorem assembled only from the two locally certified actions. -/
theorem and_comm_plan_replay (a b : Prop) :
    a ∧ b ↔ b ∧ a := by
  exact Iff.intro (and_comm_forward a b) (and_comm_reverse a b)

end ProofPlanActionCertificatePilot

#check And.comm
#check ProofPlanActionCertificatePilot.and_comm_split_iff
#check ProofPlanActionCertificatePilot.and_comm_forward
#check ProofPlanActionCertificatePilot.and_comm_reverse
#check ProofPlanActionCertificatePilot.and_comm_plan_replay

#print axioms And.comm
#print axioms ProofPlanActionCertificatePilot.and_comm_split_iff
#print axioms ProofPlanActionCertificatePilot.and_comm_forward
#print axioms ProofPlanActionCertificatePilot.and_comm_reverse
#print axioms ProofPlanActionCertificatePilot.and_comm_plan_replay
