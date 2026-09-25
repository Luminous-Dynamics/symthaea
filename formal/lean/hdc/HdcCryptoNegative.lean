namespace HdcCryptoNegative

/-- Production HDC dimension: 16,384 bits. -/
def HdcDimension : Nat := 16_384

abbrev BitIndex := Fin HdcDimension
abbrev BinaryHV := BitIndex → Bool

/-- Abstract XOR bind matching the algebra used by the quarantined HDC constructions. -/
def bind (a b : BinaryHV) : BinaryHV := fun i => Bool.xor (a i) (b i)

/-- Abstract compatibility transform: tag = message XOR effective_mask. -/
def mac (message effectiveMask : BinaryHV) : BinaryHV :=
  bind message effectiveMask

/-- Recover the effective mask from one known message/tag pair. -/
def recoverEffectiveMask (knownMessage knownTag : BinaryHV) : BinaryHV :=
  bind knownTag knownMessage

/-- Forge a tag for an arbitrary replacement message from one known pair. -/
def forgeTag (newMessage knownMessage knownTag : BinaryHV) : BinaryHV :=
  bind newMessage (recoverEffectiveMask knownMessage knownTag)

/-- One known valid pair reveals exactly the effective XOR mask. -/
theorem known_pair_recovers_effective_mask
    (message mask : BinaryHV) :
    recoverEffectiveMask message (mac message mask) = mask := by
  funext i
  simp [recoverEffectiveMask, mac, bind]
  cases message i <;> cases mask i <;> rfl

/-- The recovered mask deterministically forges a valid tag for every new message. -/
theorem one_pair_forges_arbitrary_message
    (knownMessage newMessage mask : BinaryHV) :
    forgeTag newMessage knownMessage (mac knownMessage mask) = mac newMessage mask := by
  funext i
  simp [forgeTag, recoverEffectiveMask, mac, bind]
  cases knownMessage i <;> cases newMessage i <;> cases mask i <;> rfl

/-- Abstract insecure share record: both secret XOR mask and mask are present. -/
structure Share where
  share : BinaryHV
  mask : BinaryHV

/-- Construct the abstract record used by the quarantined sharing scheme. -/
def makeShare (secret mask : BinaryHV) : Share :=
  { share := bind secret mask, mask := mask }

/-- Recover from one record, without any threshold. -/
def recoverOne (record : Share) : BinaryHV :=
  bind record.share record.mask

/-- Every single record deterministically reconstructs the original secret. -/
theorem one_share_recovers_secret
    (secret mask : BinaryHV) :
    recoverOne (makeShare secret mask) = secret := by
  funext i
  simp [recoverOne, makeShare, bind]
  cases secret i <;> cases mask i <;> rfl

#print axioms known_pair_recovers_effective_mask
#print axioms one_pair_forges_arbitrary_message
#print axioms one_share_recovers_secret

end HdcCryptoNegative
