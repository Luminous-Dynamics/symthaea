-- Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- SYM-HDC-CRYPTO-FV-001C / CI-002 threshold-independence strengthening.
-- This file is typechecked after the exact repaired SYM-FV-002 BinaryHVBind
-- and exact SYM-HDC-CRYPTO-FV-001A v2 attack theorem sources are prepended.
-- It deliberately reuses AbstractShare/makeShare/recoverOne rather than
-- introducing another XOR or share algebra.

namespace Symthaea.Formal.HDC.CryptoThresholdLeakage

open Symthaea.Formal.HDC
open Symthaea.Formal.HDC.CryptoAttacks

/-- Exact abstract split precondition mirrored from the quarantined Rust API:
    k is nonzero, k does not exceed n, and k is odd. -/
def splitAdmissible (k n : Nat) : Prop :=
  1 ≤ k ∧ k ≤ n ∧ k % 2 = 1

/-- Add the production record's public index without changing the inherited
    share/mask semantics. The declared threshold is intentionally not stored in
    each record, matching the quarantined HdcShare shape. -/
structure IndexedShare where
  index : Nat
  payload : AbstractShare

/-- Abstract one-record constructor corresponding to one element emitted by the
    quarantined split/split_secure loops after a mask has been chosen. -/
def makeIndexedShare (secret mask : BinaryHV) (index : Nat) : IndexedShare :=
  { index := index, payload := makeShare secret mask }

/-- Public one-record recovery ignores the index and applies the inherited XOR
    unbinding rule to the stored share/mask pair. -/
def recoverIndexedOne (s : IndexedShare) : BinaryHV :=
  recoverOne s.payload

/-- Every indexed record still reveals the complete secret. -/
theorem indexed_share_recovers_secret
    (secret mask : BinaryHV) (index : Nat) :
    recoverIndexedOne (makeIndexedShare secret mask index) = secret := by
  unfold recoverIndexedOne makeIndexedShare
  exact one_share_recovers_secret secret mask

/-- The declared (k,n) threshold cannot affect recovery from one returned
    record: for every production-admissible declaration, one record suffices. -/
theorem one_share_recovery_ignores_threshold_declaration
    (secret mask : BinaryHV) (index k n : Nat)
    (_valid : splitAdmissible k n) :
    recoverIndexedOne (makeIndexedShare secret mask index) = secret := by
  exact indexed_share_recovers_secret secret mask index

/-- Even when the caller requests a threshold strictly greater than one, the
    exact abstract record shape still permits complete one-record recovery. -/
theorem threshold_gt_one_still_allows_one_share_recovery
    (secret mask : BinaryHV) (index k n : Nat)
    (valid : splitAdmissible k n) (_threshold_gt_one : 1 < k) :
    recoverIndexedOne (makeIndexedShare secret mask index) = secret := by
  exact one_share_recovery_ignores_threshold_declaration
    secret mask index k n valid

/-- Record numbering contributes no confidentiality: changing only the public
    index leaves the recovered secret unchanged. -/
theorem share_index_is_irrelevant_to_recovery
    (secret mask : BinaryHV) (i j : Nat) :
    recoverIndexedOne (makeIndexedShare secret mask i) =
      recoverIndexedOne (makeIndexedShare secret mask j) := by
  rw [indexed_share_recovers_secret, indexed_share_recovers_secret]

#print axioms indexed_share_recovers_secret
#print axioms one_share_recovery_ignores_threshold_declaration
#print axioms threshold_gt_one_still_allows_one_share_recovery
#print axioms share_index_is_irrelevant_to_recovery

end Symthaea.Formal.HDC.CryptoThresholdLeakage
