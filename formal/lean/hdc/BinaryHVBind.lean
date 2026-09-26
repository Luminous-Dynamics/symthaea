namespace Symthaea.Formal.HDC

/-- Production BinaryHV has exactly 16,384 bits. -/
def HdcDimension : Nat := 16_384

/-- Bit index for the fixed production-width abstract hypervector. -/
abbrev BitIndex : Type := Fin HdcDimension

/--
Abstract BinaryHV semantics for SYM-FV-002.

This is intentionally a bit-level specification. It is not the Rust byte-array
representation and it does not claim refinement of `BinaryHV::bind_scalar`.
That relationship belongs to SYM-FV-003.
-/
abbrev BinaryHV : Type := BitIndex → Bool

/-- All-zero abstract hypervector. -/
def zero : BinaryHV := fun _ => false

/-- Observe one abstract bit. -/
def bit (v : BinaryHV) (i : BitIndex) : Bool := v i

/-- Binary HDC binding is pointwise XOR. -/
def bind (a b : BinaryHV) : BinaryHV := fun i => Bool.xor (a i) (b i)

/-- Binding exposes exactly pointwise XOR at every bit. -/
theorem bit_bind (a b : BinaryHV) (i : BitIndex) :
    bit (bind a b) i = Bool.xor (bit a i) (bit b i) := by
  rfl

/-- Zero is a right identity for binding. -/
theorem bind_zero_right (a : BinaryHV) : bind a zero = a := by
  funext i
  cases h : a i <;> simp [bind, zero, h]

/-- Zero is a left identity for binding. -/
theorem bind_zero_left (a : BinaryHV) : bind zero a = a := by
  funext i
  cases h : a i <;> simp [bind, zero, h]

/-- Every BinaryHV is its own inverse under XOR binding. -/
theorem bind_self_inverse (a : BinaryHV) : bind a a = zero := by
  funext i
  cases h : a i <;> simp [bind, zero, h]

/-- XOR binding is commutative. -/
theorem bind_comm (a b : BinaryHV) : bind a b = bind b a := by
  funext i
  cases ha : a i <;> cases hb : b i <;> simp [bind, ha, hb]

/-- XOR binding is associative. -/
theorem bind_assoc (a b c : BinaryHV) :
    bind (bind a b) c = bind a (bind b c) := by
  funext i
  cases ha : a i <;> cases hb : b i <;> cases hc : c i <;>
    simp [bind, ha, hb, hc]

/-- Binding with the same right operand recovers the original vector. -/
theorem unbind_right (a b : BinaryHV) : bind (bind a b) b = a := by
  funext i
  cases ha : a i <;> cases hb : b i <;> simp [bind, ha, hb]

/-- Binding with the same left operand recovers the other vector. -/
theorem unbind_left (a b : BinaryHV) : bind (bind a b) a = b := by
  funext i
  cases ha : a i <;> cases hb : b i <;> simp [bind, ha, hb]

-- Retain the exact trusted-axiom census in the Lean checker output.
#print axioms bit_bind
#print axioms bind_zero_right
#print axioms bind_zero_left
#print axioms bind_self_inverse
#print axioms bind_comm
#print axioms bind_assoc
#print axioms unbind_right
#print axioms unbind_left

end Symthaea.Formal.HDC
