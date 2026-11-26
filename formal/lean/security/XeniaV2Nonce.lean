-- Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- SYM-FV-XENIA-001
-- Pure bit-level theorem subject for the XEN-WIRE-V2-001 static-IV nonce shape.
-- No claim is made here about byte order, Rust refinement, key derivation,
-- allocator ownership, reset/recovery safety, or AEAD usage limits.

namespace Symthaea.Formal.XeniaV2Nonce

/-- Fixed-width bitvector model used only for this nonce algebra. -/
abbrev Bits (n : Nat) : Type := Fin n → Bool

/-- Pointwise Boolean XOR. -/
def xorBits {n : Nat} (a b : Bits n) : Bits n :=
  fun i => Bool.xor (a i) (b i)

/-- XOR by one fixed bitvector is injective. -/
theorem xor_fixed_injective {n : Nat} (fixed x y : Bits n)
    (h : xorBits fixed x = xorBits fixed y) : x = y := by
  funext i
  have hi := congrFun h i
  cases hf : fixed i <;> cases hx : x i <;> cases hy : y i <;>
    simp [xorBits, hf, hx, hy] at hi ⊢

abbrev Bits32 : Type := Bits 32
abbrev Bits64 : Type := Bits 64

/-- 96-bit nonce modeled explicitly as the 32-bit zero-prefix region plus the
    64-bit sequence region used by the proposed Xenia V2 profile. -/
structure Nonce96 where
  high32 : Bits32
  low64 : Bits64

/-- Constant all-zero 32-bit prefix. -/
def zero32 : Bits32 := fun _ => false

/-- Abstract `0^32 || sequence64` padding. -/
def pad64 (sequence : Bits64) : Nonce96 :=
  { high32 := zero32, low64 := sequence }

/-- XEN-V2-NONCE-001: zero-prefix padding preserves the entire sequence. -/
theorem pad64_injective (s1 s2 : Bits64)
    (h : pad64 s1 = pad64 s2) : s1 = s2 := by
  exact congrArg Nonce96.low64 h

/-- Componentwise 96-bit XOR. -/
def xorNonce (iv input : Nonce96) : Nonce96 :=
  {
    high32 := xorBits iv.high32 input.high32
    low64 := xorBits iv.low64 input.low64
  }

/-- XEN-V2-NONCE-002: XOR by one fixed 96-bit IV is injective. -/
theorem xor_nonce_fixed_injective (iv x y : Nonce96)
    (h : xorNonce iv x = xorNonce iv y) : x = y := by
  have hhi : xorBits iv.high32 x.high32 = xorBits iv.high32 y.high32 :=
    congrArg Nonce96.high32 h
  have hlo : xorBits iv.low64 x.low64 = xorBits iv.low64 y.low64 :=
    congrArg Nonce96.low64 h
  have ehi : x.high32 = y.high32 :=
    xor_fixed_injective iv.high32 x.high32 y.high32 hhi
  have elo : x.low64 = y.low64 :=
    xor_fixed_injective iv.low64 x.low64 y.low64 hlo
  cases x with
  | mk xhi xlo =>
    cases y with
    | mk yhi ylo =>
      simp only at ehi elo
      cases ehi
      cases elo
      rfl

/-- Proposed abstract V2 nonce function for one fixed traffic IV. -/
def nonceFromSequence (iv : Nonce96) (sequence : Bits64) : Nonce96 :=
  xorNonce iv (pad64 sequence)

/-- Equality of two nonces under the same IV implies equality of sequences. -/
theorem nonce_from_sequence_injective (iv : Nonce96) (s1 s2 : Bits64)
    (h : nonceFromSequence iv s1 = nonceFromSequence iv s2) : s1 = s2 := by
  apply pad64_injective s1 s2
  exact xor_nonce_fixed_injective iv (pad64 s1) (pad64 s2) h

/-- XEN-V2-NONCE-003: two distinct sequence values cannot produce the same
    nonce inside one fixed-IV traffic context. -/
theorem distinct_sequences_distinct_nonces (iv : Nonce96) (s1 s2 : Bits64)
    (hneq : s1 ≠ s2) : nonceFromSequence iv s1 ≠ nonceFromSequence iv s2 := by
  intro hsame
  exact hneq (nonce_from_sequence_injective iv s1 s2 hsame)

#print axioms xor_fixed_injective
#print axioms pad64_injective
#print axioms xor_nonce_fixed_injective
#print axioms nonce_from_sequence_injective
#print axioms distinct_sequences_distinct_nonces

end Symthaea.Formal.XeniaV2Nonce
