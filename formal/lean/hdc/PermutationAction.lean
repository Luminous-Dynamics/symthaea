namespace Symthaea.Formal.HDC

/--
A permutation of BinaryHV bit positions represented explicitly by mutually
inverse index maps. This first layer is intentionally generic: cyclic-offset
arithmetic is introduced only in SYM-FV-002P-B.
-/
structure IndexPermutation where
  forward : BitIndex → BitIndex
  inverse : BitIndex → BitIndex
  left_inv : ∀ i, inverse (forward i) = i
  right_inv : ∀ i, forward (inverse i) = i

/-- Identity permutation of the fixed BinaryHV index space. -/
def identityPermutation : IndexPermutation where
  forward := fun i => i
  inverse := fun i => i
  left_inv := by intro i; rfl
  right_inv := by intro i; rfl

/-- Inverse of a reviewed index permutation. -/
def inversePermutation (p : IndexPermutation) : IndexPermutation where
  forward := p.inverse
  inverse := p.forward
  left_inv := p.right_inv
  right_inv := p.left_inv

/--
Composition whose forward direction applies `p` and then `q`.
The inverse therefore applies `q⁻¹` and then `p⁻¹`.
-/
def composePermutation (p q : IndexPermutation) : IndexPermutation where
  forward := fun i => q.forward (p.forward i)
  inverse := fun i => p.inverse (q.inverse i)
  left_inv := by
    intro i
    rw [q.left_inv, p.left_inv]
  right_inv := by
    intro i
    rw [p.right_inv, q.right_inv]

/--
Permute a BinaryHV by reading each destination bit from the corresponding
inverse-mapped source position. This fixes the abstract orientation.
-/
def permuteBy (p : IndexPermutation) (v : BinaryHV) : BinaryHV :=
  fun i => v (p.inverse i)

/-- Exact bit observation rule for the generic permutation action. -/
theorem bit_permuteBy (p : IndexPermutation) (v : BinaryHV) (i : BitIndex) :
    bit (permuteBy p v) i = bit v (p.inverse i) := by
  rfl

/-- Identity permutation leaves every BinaryHV unchanged. -/
theorem permuteBy_identity (v : BinaryHV) :
    permuteBy identityPermutation v = v := by
  funext i
  rfl

/-- Permutations preserve the all-zero BinaryHV. -/
theorem permuteBy_zero (p : IndexPermutation) :
    permuteBy p zero = zero := by
  funext i
  rfl

/-- Permutation is a homomorphism of BinaryHV XOR binding. -/
theorem permuteBy_bind (p : IndexPermutation) (a b : BinaryHV) :
    permuteBy p (bind a b) = bind (permuteBy p a) (permuteBy p b) := by
  funext i
  rfl

/-- Applying `p` and then `q` equals one application of their composition. -/
theorem permuteBy_compose (p q : IndexPermutation) (v : BinaryHV) :
    permuteBy q (permuteBy p v) = permuteBy (composePermutation p q) v := by
  funext i
  rfl

/-- Applying a permutation and then its inverse recovers the original vector. -/
theorem inverse_after_permute (p : IndexPermutation) (v : BinaryHV) :
    permuteBy (inversePermutation p) (permuteBy p v) = v := by
  funext i
  change v (p.inverse (p.forward i)) = v i
  exact congrArg v (p.left_inv i)

/-- Applying the inverse and then the permutation also recovers the original. -/
theorem permute_after_inverse (p : IndexPermutation) (v : BinaryHV) :
    permuteBy p (permuteBy (inversePermutation p) v) = v := by
  funext i
  change v (p.forward (p.inverse i)) = v i
  exact congrArg v (p.right_inv i)

/-- A fixed reviewed permutation cannot collapse two distinct BinaryHVs. -/
theorem permuteBy_injective (p : IndexPermutation) :
    Function.Injective (permuteBy p) := by
  intro a b h
  funext i
  have hi := congrFun h (p.forward i)
  change a (p.inverse (p.forward i)) = b (p.inverse (p.forward i)) at hi
  rw [p.left_inv] at hi
  exact hi

/-- Every BinaryHV has a preimage under a fixed reviewed permutation. -/
theorem permuteBy_surjective (p : IndexPermutation) :
    Function.Surjective (permuteBy p) := by
  intro v
  refine ⟨permuteBy (inversePermutation p) v, ?_⟩
  exact permute_after_inverse p v

/-- The generic BinaryHV permutation action is bijective. -/
theorem permuteBy_bijective (p : IndexPermutation) :
    Function.Bijective (permuteBy p) := by
  exact ⟨permuteBy_injective p, permuteBy_surjective p⟩

#print axioms bit_permuteBy
#print axioms permuteBy_identity
#print axioms permuteBy_zero
#print axioms permuteBy_bind
#print axioms permuteBy_compose
#print axioms inverse_after_permute
#print axioms permute_after_inverse
#print axioms permuteBy_injective
#print axioms permuteBy_surjective
#print axioms permuteBy_bijective

end Symthaea.Formal.HDC
