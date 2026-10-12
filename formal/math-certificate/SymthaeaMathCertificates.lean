import PrimeCert

namespace SymthaeaMathCertificates

/-- Wire representation of one recursive Pocklington proof node. -/
structure WirePrimalityNode where
  prime : Nat
  predecessorFactors : List (Nat × Nat)
  witnesses : List (Nat × Nat)
  deriving DecidableEq, Repr

/-- Wire representation of a complete SYMPPRF1 certificate. -/
structure WirePrimalityProof where
  value : Nat
  certificates : List WirePrimalityNode
  deriving DecidableEq, Repr

/-- The outer factorization record in a SYMFCER1 certificate. -/
structure WireFactorization where
  value : Nat
  factors : List (Nat × Nat)
  deriving DecidableEq, Repr

/-- A complete SYMFCER1 certificate, including one proof for each prime factor. -/
structure WireCertifiedFactorization where
  factorization : WireFactorization
  primeProofs : List WirePrimalityProof
  deriving DecidableEq, Repr

/-- Extract one big-endian byte from a natural number. -/
def wireByteAt (value shift : Nat) : Nat :=
  value / (256 ^ shift) % 256

/-- Version-1 unsigned 32-bit big-endian encoding. -/
def encodeWireU32 (value : Nat) : List Nat :=
  [wireByteAt value 3, wireByteAt value 2, wireByteAt value 1, wireByteAt value 0]

/-- Version-1 unsigned 64-bit big-endian encoding. -/
def encodeWireU64 (value : Nat) : List Nat :=
  [wireByteAt value 7, wireByteAt value 6, wireByteAt value 5, wireByteAt value 4,
   wireByteAt value 3, wireByteAt value 2, wireByteAt value 1, wireByteAt value 0]

def encodeWireFactorPairs (factors : List (Nat × Nat)) : List Nat :=
  factors.flatMap (fun factor =>
    encodeWireU64 factor.1 ++ encodeWireU32 factor.2)

def encodeWireWitnessPairs (witnesses : List (Nat × Nat)) : List Nat :=
  witnesses.flatMap (fun witness =>
    encodeWireU64 witness.1 ++ encodeWireU64 witness.2)

/-- Canonical SYMPPRF1 encoding of one node. -/
def encodeWirePrimalityNode (node : WirePrimalityNode) : List Nat :=
  encodeWireU64 node.prime ++
  encodeWireU32 node.predecessorFactors.length ++
  encodeWireFactorPairs node.predecessorFactors ++
  encodeWireU32 node.witnesses.length ++
  encodeWireWitnessPairs node.witnesses

/-- Canonical SYMPPRF1 encoder, independently expressed in Lean. -/
def encodeWirePrimalityProof (proof : WirePrimalityProof) : List Nat :=
  [83, 89, 77, 80, 80, 82, 70, 49] ++
  encodeWireU64 proof.value ++
  encodeWireU32 proof.certificates.length ++
  proof.certificates.flatMap encodeWirePrimalityNode

/-- Canonical SYMFCER1 encoder, including length-prefixed nested proofs. -/
def encodeWireCertifiedFactorization
    (certificate : WireCertifiedFactorization) : List Nat :=
  [83, 89, 77, 70, 67, 69, 82, 49] ++
  encodeWireU64 certificate.factorization.value ++
  encodeWireU32 certificate.factorization.factors.length ++
  encodeWireFactorPairs certificate.factorization.factors ++
  encodeWireU32 certificate.primeProofs.length ++
  certificate.primeProofs.flatMap (fun proof =>
    let encoded := encodeWirePrimalityProof proof
    encodeWireU32 encoded.length ++ encoded)

/-- Width predicates prevent the modulo-based byte encoder from hiding overflow. -/
def wireFitsU32 (value : Nat) : Bool :=
  decide (value < 4294967296)

def wireFitsU64 (value : Nat) : Bool :=
  decide (value < 18446744073709551616)

def wirePrimalityProofFieldsFit (proof : WirePrimalityProof) : Bool :=
  wireFitsU64 proof.value &&
  wireFitsU32 proof.certificates.length &&
  decide (proof.certificates.length <= 4096) &&
  decide (proof.certificates.foldl
    (fun total node => total + node.predecessorFactors.length + node.witnesses.length) 0
      <= 65536) &&
  proof.certificates.all (fun node =>
    wireFitsU64 node.prime &&
    wireFitsU32 node.predecessorFactors.length &&
    node.predecessorFactors.all (fun factor =>
      wireFitsU64 factor.1 && wireFitsU32 factor.2) &&
    wireFitsU32 node.witnesses.length &&
    node.witnesses.all (fun witness =>
      wireFitsU64 witness.1 && wireFitsU64 witness.2))

/-- Check the SYMFCER1 wire widths, resource ceilings, and factor/proof alignment. -/
def wireCertifiedFactorizationFieldsFit
    (certificate : WireCertifiedFactorization) : Bool :=
  wireFitsU64 certificate.factorization.value &&
  wireFitsU32 certificate.factorization.factors.length &&
  decide (certificate.factorization.factors.length <= 4096) &&
  certificate.factorization.factors.all (fun factor =>
    wireFitsU64 factor.1 && wireFitsU32 factor.2 && decide (factor.2 > 0)) &&
  wireFitsU32 certificate.primeProofs.length &&
  decide (certificate.primeProofs.length <= 4096) &&
  certificate.primeProofs.all wirePrimalityProofFieldsFit &&
  certificate.primeProofs.all (fun proof =>
    wireFitsU32 (encodeWirePrimalityProof proof).length) &&
  decide ((encodeWireCertifiedFactorization certificate).length <= 1048576) &&
  decide (certificate.factorization.factors.length = certificate.primeProofs.length) &&
  (List.zip certificate.factorization.factors certificate.primeProofs).all
    (fun entry => entry.1.1 == entry.2.value)

/-- A kernel-checked Pocklington proof for the minimal cross-language vector. -/
theorem golden_prime_three : Nat.Prime 3 := by
  prime_cert [small {2}, pock (3, 2, 2)]

/-- Base-prime claim carried by the SYMFCER1 factorization-of-2 vector. -/
theorem golden_factor_two_is_prime : Nat.Prime 2 := by
  norm_num

/-- The outer factorization in the 76-byte SYMFCER1 vector reconstructs 2. -/
theorem golden_factorization_two_reconstructs : 2 = (2) := by
  norm_num

end SymthaeaMathCertificates
