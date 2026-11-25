noncomputable def pick : Nat := Classical.choice ⟨0⟩

structure S9 where
  x : Fin (pick + 1)

#print axioms S9
#print axioms S9.mk
