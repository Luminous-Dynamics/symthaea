namespace Symthaea.Formal.HDC

/-- Convert one Boolean bit to its exact natural-number contribution. -/
def bitNat (b : Bool) : Nat := if b then 1 else 0

/--
Count the true bits of the fixed-width abstract BinaryHV.

This is an exact mathematical fold over all 16,384 abstract bit positions. It is
not yet a theorem about Rust bytewise `count_ones`, SIMD popcount, floating-point
similarity, or native execution.
-/
def bitCount (v : BinaryHV) : Nat :=
  Fin.foldl HdcDimension (fun acc i => acc + bitNat (v i)) 0

/--
Pointwise mismatch vector: a bit is true exactly when the corresponding input
bits differ. This definition is intentionally stated independently from `bind`;
the theorem below establishes their equality.
-/
def mismatchVector (a b : BinaryHV) : BinaryHV :=
  fun i => a i != b i

/-- Exact abstract Hamming distance: count the mismatching bit positions. -/
def hammingDistance (a b : BinaryHV) : Nat :=
  bitCount (mismatchVector a b)

/-- A mismatch bit is exactly Boolean XOR of the corresponding input bits. -/
theorem mismatch_bit (a b : BinaryHV) (i : BitIndex) :
    bit (mismatchVector a b) i = Bool.xor (bit a i) (bit b i) := by
  cases ha : a i <;> cases hb : b i <;> rfl

/-- The independently defined mismatch vector is exactly HDC XOR binding. -/
theorem mismatchVector_eq_bind (a b : BinaryHV) :
    mismatchVector a b = bind a b := by
  funext i
  cases ha : a i <;> cases hb : b i <;> rfl

/-- Hamming distance is exactly the true-bit count of the XOR-bound vector. -/
theorem hamming_eq_bitCount_bind (a b : BinaryHV) :
    hammingDistance a b = bitCount (bind a b) := by
  unfold hammingDistance
  rw [mismatchVector_eq_bind]

/-- Hamming distance is symmetric. -/
theorem hamming_symm (a b : BinaryHV) :
    hammingDistance a b = hammingDistance b a := by
  rw [hamming_eq_bitCount_bind, hamming_eq_bitCount_bind, bind_comm]

/--
Binding the same right-hand mask into both vectors preserves the entire mismatch
vector. This is the structural reason common-mask XOR preserves Hamming distance.
-/
theorem mismatch_bind_right_invariant (a b mask : BinaryHV) :
    mismatchVector (bind a mask) (bind b mask) = mismatchVector a b := by
  funext i
  cases ha : a i <;> cases hb : b i <;> cases hm : mask i <;> rfl

/-- Common right-hand XOR masking is an exact Hamming isometry. -/
theorem hamming_bind_right_invariant (a b mask : BinaryHV) :
    hammingDistance (bind a mask) (bind b mask) = hammingDistance a b := by
  unfold hammingDistance
  rw [mismatch_bind_right_invariant]

/-- Common left-hand XOR masking is also an exact Hamming isometry. -/
theorem hamming_bind_left_invariant (mask a b : BinaryHV) :
    hammingDistance (bind mask a) (bind mask b) = hammingDistance a b := by
  rw [bind_comm mask a, bind_comm mask b]
  exact hamming_bind_right_invariant a b mask

-- Retain the exact trusted-axiom census in ordinary Lean checker output.
#print axioms mismatch_bit
#print axioms mismatchVector_eq_bind
#print axioms hamming_eq_bitCount_bind
#print axioms hamming_symm
#print axioms mismatch_bind_right_invariant
#print axioms hamming_bind_right_invariant
#print axioms hamming_bind_left_invariant

end Symthaea.Formal.HDC
