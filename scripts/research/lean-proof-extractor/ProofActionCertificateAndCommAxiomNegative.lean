import Lean

namespace ProofActionCertificateAxiomControl

/-- A deliberately injected axiom used only to test axiom-dependency reporting. -/
axiom injected_premise : False

theorem axiom_dependent_certificate (p : Prop) : p :=
  False.elim injected_premise

end ProofActionCertificateAxiomControl

#print axioms ProofActionCertificateAxiomControl.axiom_dependent_certificate
