namespace Symthaea.Formal.HDC

/-- Convert one Boolean bit to its exact natural-number population contribution. -/
def bitCount (b : Bool) : Nat := if b then 1 else 0

/--
Read an abstract fixed-width BinaryHV at a natural index.
Indices outside the production width are mapped to false so recursive prefix
counting has a total Nat-indexed observation function.
-/
def bitAtNat (v : BinaryHV) (i : Nat) : Bool :=
  if h : i < HdcDimension then v ⟨i, h⟩ else false

/-- Count set bits in the first `n` natural positions of an abstract BinaryHV. -/
def popcountPrefix : Nat → BinaryHV → Nat
  | 0, _ => 0
  | n + 1, v => popcountPrefix n v + bitCount (bitAtNat v n)

/-- Exact abstract population count over all 16,384 BinaryHV positions. -/
def popcount (v : BinaryHV) : Nat := popcountPrefix HdcDimension v

/--
Exact abstract Hamming distance. For binary hypervectors, distance is the
population count of their pointwise XOR/bind result.
-/
def hammingDistance (a b : BinaryHV) : Nat := popcount (bind a b)

/-- Hamming distance is exactly the population count of XOR binding. -/
theorem hamming_eq_popcount_bind (a b : BinaryHV) :
    hammingDistance a b = popcount (bind a b) := by
  rfl

/-- Population count of the all-zero abstract vector is zero. -/
theorem popcountPrefix_zero (n : Nat) : popcountPrefix n zero = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp [popcountPrefix, bitCount, bitAtNat, zero, ih]

/-- Full-width population count of zero is zero. -/
theorem popcount_zero : popcount zero = 0 := by
  unfold popcount
  exact popcountPrefix_zero HdcDimension

/-- A vector has zero Hamming distance from itself. -/
theorem hamming_self (a : BinaryHV) : hammingDistance a a = 0 := by
  rw [hammingDistance, bind_self_inverse, popcount_zero]

/-- Hamming distance is symmetric. -/
theorem hamming_comm (a b : BinaryHV) :
    hammingDistance a b = hammingDistance b a := by
  unfold hammingDistance
  rw [bind_comm]

/-- Binding both operands by the same left mask preserves pairwise XOR exactly. -/
theorem common_left_bind_pairwise_xor (mask a b : BinaryHV) :
    bind (bind mask a) (bind mask b) = bind a b := by
  funext i
  cases hm : mask i <;> cases ha : a i <;> cases hb : b i <;>
    simp [bind, hm, ha, hb]

/-- Common-mask XOR binding is an exact Hamming isometry. -/
theorem hamming_common_left_bind_invariant (mask a b : BinaryHV) :
    hammingDistance (bind mask a) (bind mask b) = hammingDistance a b := by
  unfold hammingDistance
  rw [common_left_bind_pairwise_xor]

/-- Binding both operands by the same right mask is also an exact Hamming isometry. -/
theorem hamming_common_right_bind_invariant (a b mask : BinaryHV) :
    hammingDistance (bind a mask) (bind b mask) = hammingDistance a b := by
  rw [bind_comm a mask, bind_comm b mask]
  exact hamming_common_left_bind_invariant mask a b

#print axioms hamming_eq_popcount_bind
#print axioms popcount_zero
#print axioms hamming_self
#print axioms hamming_comm
#print axioms common_left_bind_pairwise_xor
#print axioms hamming_common_left_bind_invariant
#print axioms hamming_common_right_bind_invariant

end Symthaea.Formal.HDC
