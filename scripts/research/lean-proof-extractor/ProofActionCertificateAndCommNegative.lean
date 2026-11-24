import Lean

/-- Negative control: this proposition is not the forward action certificate.
    Reusing And.symm must be rejected because the target is identity, not swap. -/
namespace ProofPlanActionCertificatePilotNegative

theorem wrong_forward_proposition (a b : Prop) :
    a ∧ b → a ∧ b := And.symm

end ProofPlanActionCertificatePilotNegative
