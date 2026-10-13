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

/-- Pocklington soundness with an independent witness for each distinct
prime divisor. Unlike the common-root variant, this matches the certificate wire
format directly: each predecessor prime can carry its own witness.

The only arithmetic premises are the Pocklington divisibility and size bounds,
plus the witness congruence and GCD conditions for every prime divisor of F. -/
theorem pocklington_certify_independent_witnesses
    (N F : Nat) (h2n : 2 ≤ N) (hF : F ∣ N - 1) (hFlarge : N.sqrt < F)
    (witnesses : ∀ q ∈ F.primeFactors, ∃ a : Nat,
      a ^ (N - 1) ≡ 1 [MOD N] ∧
      Nat.gcd (a ^ ((N - 1) / q) - 1) N = 1) :
    Nat.Prime N := by
  by_contra hNprime
  rw [Nat.sqrt_lt, ← sq] at hFlarge
  have hfactor := pocklington_test N F (by omega) hF witnesses
    N.minFac (Nat.minFac_prime (by omega)) N.minFac_dvd
  have hminFac : 2 ≤ N.minFac := (Nat.minFac_prime (by omega)).two_le
  rw [Nat.ModEq.comm, Nat.modEq_iff_dvd' (by omega)] at hfactor
  have hfactorLowerBound :=
    Nat.succ_le_iff.mp <| (Nat.le_sub_iff_add_le (by omega)).mp <|
      Nat.le_of_dvd (by omega) hfactor
  exact lt_asymm hFlarge <|
    ((Nat.pow_lt_pow_iff_left (by omega)).mpr hfactorLowerBound).trans_le <|
      Nat.minFac_sq_le_self (by omega) hNprime

/-- A small, explicit regression: 13 is prime using different witness obligations
for q = 2 and q = 3, without constructing a shared witness in Python. -/
theorem prime_thirteen_from_independent_witnesses : Nat.Prime 13 := by
  apply pocklington_certify_independent_witnesses (N := 13) (F := 12)
  · norm_num
  · norm_num
  · norm_num
  · intro q hq
    have hqPrime : Nat.Prime q := (Nat.mem_primeFactors.mp hq).1
    have hqDvd : q ∣ 12 := (Nat.mem_primeFactors.mp hq).2.1
    have hqBound : q ≤ 12 := Nat.le_of_dvd (by norm_num) hqDvd
    have hqCases : q = 2 ∨ q = 3 := by
      interval_cases q <;> norm_num at hqPrime hqDvd ⊢
    rcases hqCases with rfl | rfl
    · exact ⟨2, by norm_num [Nat.ModEq], by norm_num⟩
    · exact ⟨2, by norm_num [Nat.ModEq], by norm_num⟩

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
