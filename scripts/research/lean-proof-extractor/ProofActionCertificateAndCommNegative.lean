import Lean

namespace ProofPlanActionCertificatePilotNegative

theorem wrong_forward_proposition (a b : Prop) :
    a ∧ b → a ∧ b := And.symm

end ProofPlanActionCertificatePilotNegative
