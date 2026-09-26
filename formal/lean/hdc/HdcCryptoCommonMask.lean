-- Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- SYM-HDC-CRYPTO-FV-001B
-- This file is typechecked after the exact SYM-FV-002 BinaryHVBind theorem
-- source and SYM-HDC-CRYPTO-FV-001A attack theorem source are prepended by
-- the qualification workflow. It deliberately does not redefine BinaryHV,
-- bind, zero, or the HDC dimension.

namespace Symthaea.Formal.HDC.CryptoCommonMask

open Symthaea.Formal.HDC

/-- Abstract the exact XOR-mask shape used by the quarantined EncryptedHV
    transform: plaintext bound with one mask. This is algebra only and does not
    claim production Rust refinement. -/
def maskWith (plaintext mask : BinaryHV) : BinaryHV :=
  bind plaintext mask

/-- Reusing one XOR mask for two plaintexts preserves their entire pairwise XOR
    relation. This is stronger than a Hamming-distance statement but does not
    itself define or prove any Hamming semantics. -/
theorem common_mask_preserves_pairwise_xor
    (x y mask : BinaryHV) :
    bind (maskWith x mask) (maskWith y mask) = bind x y := by
  unfold maskWith
  funext i
  cases hx : x i <;> cases hy : y i <;> cases hm : mask i <;> rfl

/-- Equality of pairwise XOR relations. This names the exact structural
    information preserved by common-mask reuse without assigning a stronger
    cryptographic security notion to it. -/
def SamePairwiseXor (x y x' y' : BinaryHV) : Prop :=
  bind x y = bind x' y'

/-- Common-mask reuse always satisfies the exact structural leakage relation. -/
theorem common_mask_leaks_pairwise_relation
    (x y mask : BinaryHV) :
    SamePairwiseXor x y (maskWith x mask) (maskWith y mask) := by
  unfold SamePairwiseXor
  symm
  exact common_mask_preserves_pairwise_xor x y mask

/-- The leakage theorem is symmetric in the two plaintext operands. -/
theorem common_mask_preserves_pairwise_xor_swapped
    (x y mask : BinaryHV) :
    bind (maskWith y mask) (maskWith x mask) = bind y x := by
  exact common_mask_preserves_pairwise_xor y x mask

#print axioms common_mask_preserves_pairwise_xor
#print axioms common_mask_leaks_pairwise_relation
#print axioms common_mask_preserves_pairwise_xor_swapped

end Symthaea.Formal.HDC.CryptoCommonMask
