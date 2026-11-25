noncomputable def pick : Nat := Classical.choice ⟨0⟩

structure S9 where
  x : Fin (pick + 1)

axiom auditDirectAxiom : False

theorem auditTransitiveAxiom : False := auditDirectAxiom

mutual
  inductive Even : Nat → Prop where
    | zero : Even 0
    | succ : {n : Nat} → Odd n → Even (n + 1)

  inductive Odd : Nat → Prop where
    | succ : {n : Nat} → Even n → Odd (n + 1)
end

#print axioms S9
#print axioms S9.mk
#print axioms auditDirectAxiom
#print axioms auditTransitiveAxiom
#print axioms Even
#print axioms Odd
