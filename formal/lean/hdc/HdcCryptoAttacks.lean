-- Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- SYM-HDC-CRYPTO-FV-001A
-- This file is typechecked after the exact SYM-FV-002 BinaryHVBind theorem
-- source is prepended by the qualification workflow. It deliberately does not
-- redefine BinaryHV or bind.

namespace Symthaea.Formal.HDC.CryptoAttacks

open Symthaea.Formal.HDC

/-- Algebraic core of the quarantined HdcMac construction after public key
    permutation has produced a derived mask. -/
def macTag (message derivedMask : BinaryHV) : BinaryHV :=
  bind message derivedMask

/-- Recover the reusable derived mask from one known message/tag pair. -/
def recoverDerived (knownMessage knownTag : BinaryHV) : BinaryHV :=
  bind knownMessage knownTag

/-- Forge a tag for an arbitrary chosen message from one known pair. -/
def forgeTag (knownMessage knownTag chosenMessage : BinaryHV) : BinaryHV :=
  bind chosenMessage (recoverDerived knownMessage knownTag)

/-- One valid message/tag pair reveals the entire derived XOR mask. -/
theorem recover_derived_from_known_pair (message derivedMask : BinaryHV) :
    recoverDerived message (macTag message derivedMask) = derivedMask := by
  unfold recoverDerived macTag
  rw [← bind_assoc, bind_self_inverse, bind_zero_left]

/-- The recovered mask forges a valid tag for every chosen message. -/
theorem universal_known_pair_forgery
    (knownMessage chosenMessage derivedMask : BinaryHV) :
    forgeTag knownMessage (macTag knownMessage derivedMask) chosenMessage =
      macTag chosenMessage derivedMask := by
  unfold forgeTag
  rw [recover_derived_from_known_pair]
  rfl

/-- Abstract verifier predicate for the exact lossless tag relation. -/
def macAccepts (message derivedMask tag : BinaryHV) : Prop :=
  tag = macTag message derivedMask

/-- Every tag produced by the universal known-pair forgery is accepted by the
    abstract exact verifier relation. -/
theorem forged_tag_is_accepted
    (knownMessage chosenMessage derivedMask : BinaryHV) :
    macAccepts chosenMessage derivedMask
      (forgeTag knownMessage (macTag knownMessage derivedMask) chosenMessage) := by
  unfold macAccepts
  exact universal_known_pair_forgery knownMessage chosenMessage derivedMask

/-- Abstract shape of one returned legacy HDC share record. -/
structure AbstractShare where
  share : BinaryHV
  mask : BinaryHV

/-- The legacy record stores `secret XOR mask` together with that same mask. -/
def makeShare (secret mask : BinaryHV) : AbstractShare :=
  { share := bind secret mask, mask := mask }

/-- Anyone holding one returned record can apply the public XOR unbinding rule. -/
def recoverOne (s : AbstractShare) : BinaryHV :=
  bind s.share s.mask

/-- Every single returned record recovers the secret exactly. -/
theorem one_share_recovers_secret (secret mask : BinaryHV) :
    recoverOne (makeShare secret mask) = secret := by
  unfold recoverOne makeShare
  exact unbind_right secret mask

#print axioms recover_derived_from_known_pair
#print axioms universal_known_pair_forgery
#print axioms forged_tag_is_accepted
#print axioms one_share_recovers_secret

end Symthaea.Formal.HDC.CryptoAttacks
