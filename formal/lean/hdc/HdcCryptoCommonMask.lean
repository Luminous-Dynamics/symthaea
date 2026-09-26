-- Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- SYM-HDC-CRYPTO-FV-001B v2
-- Typechecked after the repaired SYM-FV-002 algebra and rebuilt 001A v2
-- attack theorem source. No HDC semantic root is redefined here.

namespace Symthaea.Formal.HDC.CryptoCommonMask

open Symthaea.Formal.HDC

/-- Abstract the XOR-mask shape used by the quarantined EncryptedHV transform.
    This is algebra only and does not claim production Rust refinement. -/
def maskWith (plaintext mask : BinaryHV) : BinaryHV :=
  bind plaintext mask

/-- Reusing one XOR mask for two plaintexts preserves their complete pairwise
    XOR relation. This deliberately precedes the canonical Hamming corollary. -/
theorem common_mask_preserves_pairwise_xor
    (x y mask : BinaryHV) :
    bind (maskWith x mask) (maskWith y mask) = bind x y := by
  unfold maskWith
  funext i
  cases hx : x i <;> cases hy : y i <;> cases hm : mask i <;>
    simp [bind, hx, hy, hm]

/-- Equality of pairwise XOR relations. -/
def SamePairwiseXor (x y x' y' : BinaryHV) : Prop :=
  bind x y = bind x' y'

/-- Common-mask reuse always satisfies the structural leakage relation. -/
theorem common_mask_leaks_pairwise_relation
    (x y mask : BinaryHV) :
    SamePairwiseXor x y (maskWith x mask) (maskWith y mask) := by
  unfold SamePairwiseXor
  symm
  exact common_mask_preserves_pairwise_xor x y mask

/-- The relation is symmetric in the plaintext operands. -/
theorem common_mask_preserves_pairwise_xor_swapped
    (x y mask : BinaryHV) :
    bind (maskWith y mask) (maskWith x mask) = bind y x := by
  exact common_mask_preserves_pairwise_xor y x mask

#print axioms common_mask_preserves_pairwise_xor
#print axioms common_mask_leaks_pairwise_relation
#print axioms common_mask_preserves_pairwise_xor_swapped

end Symthaea.Formal.HDC.CryptoCommonMask
