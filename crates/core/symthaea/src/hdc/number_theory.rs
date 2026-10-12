// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{PrimitiveSystem, seed_from_name};

/// Prime factorization of a number with HDC encoding
#[derive(Debug, Clone)]
pub struct PrimeFactorization {
    pub value: u64,
    pub factors: Vec<(u64, u32)>,
    pub encoding: BinaryHV,
    /// Floating representation of the exact Euler totient, rounded once from `u64`.
    /// Values above 2^53 may not be exactly representable as `f64`.
    pub phi: f64,
}

/// Compute Euler's totient exactly from a value and its distinct prime factors.
///
/// The caller must provide the complete, distinct prime factors of `value`.
/// Divide-before-multiply keeps every intermediate integral and no larger than `value`.
fn phi_from_distinct_prime_factors(value: u64, factors: &[(u64, u32)]) -> u64 {
    let mut phi = value;
    for &(prime, _) in factors {
        phi = (phi / prime) * (prime - 1);
    }
    phi
}

/// A compact, canonical integer-factorization certificate.
///
/// The prime factors must be strictly increasing and each exponent must be non-zero.
/// The certificate deliberately excludes HDC encodings and floating-point totients:
/// both are derived representations, not evidence that the integer factorization is valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorizationCertificate {
    pub value: u64,
    pub factors: Vec<(u64, u32)>,
}

/// One node in a recursive Pocklington primality proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimalityCertificate {
    /// The prime claimed by this node.
    pub prime: u64,
    /// Complete prime factorization of prime - 1 as (factor, exponent) pairs.
    pub predecessor_factors: Vec<(u64, u32)>,
    /// One (prime factor of prime - 1, witness base) pair per distinct factor.
    pub witnesses: Vec<(u64, u64)>,
}

/// A compact proof bundle. Certificate nodes are sorted by prime value so all
/// dependencies are expected to appear before the node that uses them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimalityProof {
    pub value: u64,
    pub certificates: Vec<PrimalityCertificate>,
}

/// Typed errors from building or verifying a recursive primality proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimalityProofError {
    InvalidRoot { value: u64 },
    TooManyCertificates { count: usize },
    TooManyEntries { count: usize },
    CertificateTableNotStrictlyIncreasing { previous: u64, current: u64 },
    MissingPrimeCertificate { prime: u64 },
    CompositeCandidate { prime: u64 },
    MalformedPrimeTwoCertificate,
    PredecessorFactorsNotStrictlyIncreasing { prime: u64, previous: u64, current: u64 },
    ZeroPredecessorExponent { prime: u64, factor: u64 },
    PredecessorFactorNotYetProved { prime: u64, factor: u64 },
    PredecessorProductOverflow { prime: u64, factor: u64 },
    PredecessorProductMismatch { prime: u64, expected: u64, actual: u64 },
    PredecessorProductNotLargeEnough { prime: u64, factor_product: u64 },
    WitnessesDoNotMatchPredecessorFactors { prime: u64 },
    InvalidWitnessBase { prime: u64, factor: u64, base: u64 },
    InvalidPocklingtonWitness { prime: u64, factor: u64, base: u64 },
    UnreachableCertificate { prime: u64 },
    GenerationFactorizationInvalid { value: u64 },
    WitnessSearchExhausted { prime: u64, factor: u64 },
}

/// Versioned, bounded canonical encoding errors for recursive primality proofs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimalityProofWireError {
    InvalidMagicOrVersion,
    NonCanonicalEncoding,
    TruncatedInput,
    TrailingBytes { count: usize },
    InputTooLarge { length: usize },
    TooManyCertificates { count: u32 },
    TooManyEntries { count: u64 },
    LengthOverflow,
    InvalidProof(PrimalityProofError),
}

const PRIMALITY_PROOF_WIRE_MAGIC: &[u8; 8] = b"SYMPPRF1";
const PRIMALITY_PROOF_MAX_WIRE_BYTES: usize = 1_048_576;
const PRIMALITY_PROOF_MAX_CERTIFICATES: u32 = 4_096;
const PRIMALITY_PROOF_MAX_ENTRIES: u64 = 65_536;

impl PrimalityProof {
    /// Verify this proof using only integer arithmetic and recursive Pocklington
    /// conditions. The checker does not call Miller-Rabin or Pollard-Brent.
    ///
    /// Each non-base node includes a full factorization of p - 1, recursive proof
    /// nodes for every distinct factor, and a witness for each such factor. The
    /// verifier checks product identities and Pocklington congruences directly.
    pub fn verify(&self) -> Result<(), PrimalityProofError> {
        if self.value < 2 {
            return Err(PrimalityProofError::InvalidRoot { value: self.value });
        }

        // Enforce the same ceilings for public in-memory objects as for wire input.
        // Do this before allocating the lookup table or evaluating any witnesses.
        if self.certificates.len() > PRIMALITY_PROOF_MAX_CERTIFICATES as usize {
            return Err(PrimalityProofError::TooManyCertificates {
                count: self.certificates.len(),
            });
        }
        let mut total_entries = 0usize;
        for certificate in &self.certificates {
            total_entries = total_entries
                .saturating_add(certificate.predecessor_factors.len())
                .saturating_add(certificate.witnesses.len());
            if total_entries > PRIMALITY_PROOF_MAX_ENTRIES as usize {
                return Err(PrimalityProofError::TooManyEntries { count: total_entries });
            }
        }

        let mut nodes = std::collections::BTreeMap::new();
        let mut verified = std::collections::BTreeSet::new();
        let mut previous_node = 0u64;

        for certificate in &self.certificates {
            if certificate.prime <= previous_node {
                return Err(PrimalityProofError::CertificateTableNotStrictlyIncreasing {
                    previous: previous_node,
                    current: certificate.prime,
                });
            }
            previous_node = certificate.prime;
            nodes.insert(certificate.prime, certificate);
        }

        if !nodes.contains_key(&self.value) {
            return Err(PrimalityProofError::MissingPrimeCertificate { prime: self.value });
        }

        for certificate in &self.certificates {
            let prime = certificate.prime;
            if prime == 2 {
                if !certificate.predecessor_factors.is_empty() || !certificate.witnesses.is_empty() {
                    return Err(PrimalityProofError::MalformedPrimeTwoCertificate);
                }
                verified.insert(prime);
                continue;
            }
            if prime < 3 || prime.is_multiple_of(2) {
                return Err(PrimalityProofError::CompositeCandidate { prime });
            }

            let mut product = 1u64;
            let mut previous_factor = 1u64;
            for &(factor, exponent) in &certificate.predecessor_factors {
                if factor <= previous_factor {
                    return Err(PrimalityProofError::PredecessorFactorsNotStrictlyIncreasing {
                        prime,
                        previous: previous_factor,
                        current: factor,
                    });
                }
                if exponent == 0 {
                    return Err(PrimalityProofError::ZeroPredecessorExponent { prime, factor });
                }
                if !verified.contains(&factor) {
                    return Err(PrimalityProofError::PredecessorFactorNotYetProved { prime, factor });
                }
                previous_factor = factor;
                for _ in 0..exponent {
                    product = product.checked_mul(factor).ok_or(
                        PrimalityProofError::PredecessorProductOverflow { prime, factor },
                    )?;
                }
            }

            let expected = prime - 1;
            if product != expected {
                return Err(PrimalityProofError::PredecessorProductMismatch {
                    prime,
                    expected,
                    actual: product,
                });
            }
            // Pocklington requires a fully certified factor F of p - 1 with F² > p.
            // Here F = p - 1, but check the theorem's size condition explicitly.
            if u128::from(product) * u128::from(product) <= u128::from(prime) {
                return Err(PrimalityProofError::PredecessorProductNotLargeEnough {
                    prime,
                    factor_product: product,
                });
            }

            if certificate.witnesses.len() != certificate.predecessor_factors.len() {
                return Err(PrimalityProofError::WitnessesDoNotMatchPredecessorFactors { prime });
            }

            for (&(factor, _), &(witness_factor, base)) in certificate
                .predecessor_factors
                .iter()
                .zip(certificate.witnesses.iter())
            {
                if factor != witness_factor {
                    return Err(PrimalityProofError::WitnessesDoNotMatchPredecessorFactors { prime });
                }
                if base <= 1 || base >= prime {
                    return Err(PrimalityProofError::InvalidWitnessBase { prime, factor, base });
                }

                let full_power = certificate_pow_mod(base, prime - 1, prime);
                let partial_power = certificate_pow_mod(base, (prime - 1) / factor, prime);
                if full_power != 1
                    || certificate_gcd(partial_power.abs_diff(1), prime) != 1
                {
                    return Err(PrimalityProofError::InvalidPocklingtonWitness { prime, factor, base });
                }
            }

            verified.insert(prime);
        }

        // Reject unrelated proof nodes so the bundle is canonical and contains
        // only the dependency closure needed to establish the requested root.
        let mut reachable = std::collections::BTreeSet::new();
        let mut pending = vec![self.value];
        while let Some(prime) = pending.pop() {
            if !reachable.insert(prime) {
                continue;
            }
            let certificate = nodes
                .get(&prime)
                .ok_or(PrimalityProofError::MissingPrimeCertificate { prime })?;
            pending.extend(certificate.predecessor_factors.iter().map(|&(factor, _)| factor));
        }
        if let Some(certificate) = self.certificates.iter().find(|c| !reachable.contains(&c.prime)) {
            return Err(PrimalityProofError::UnreachableCertificate { prime: certificate.prime });
        }
        Ok(())
    }
}

impl PrimalityProof {
    /// Encode a verified proof as canonical, versioned big-endian bytes.
    ///
    /// Fixed-width integers avoid alternate textual spellings. The whole proof is
    /// verified before encoding and capped at 1 MiB with bounded table/entry counts.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, PrimalityProofWireError> {
        let certificate_count = u32::try_from(self.certificates.len())
            .map_err(|_| PrimalityProofWireError::LengthOverflow)?;
        if certificate_count > PRIMALITY_PROOF_MAX_CERTIFICATES {
            return Err(PrimalityProofWireError::TooManyCertificates {
                count: certificate_count,
            });
        }

        let mut total_entries = 0u64;
        let mut encoded_len = 8usize + 8 + 4;
        for certificate in &self.certificates {
            let factors = u64::try_from(certificate.predecessor_factors.len())
                .map_err(|_| PrimalityProofWireError::LengthOverflow)?;
            let witnesses = u64::try_from(certificate.witnesses.len())
                .map_err(|_| PrimalityProofWireError::LengthOverflow)?;
            total_entries = total_entries
                .checked_add(factors)
                .and_then(|count| count.checked_add(witnesses))
                .ok_or(PrimalityProofWireError::LengthOverflow)?;
            encoded_len = encoded_len
                .checked_add(8 + 4 + 4)
                .and_then(|len| len.checked_add(certificate.predecessor_factors.len().checked_mul(12)?))
                .and_then(|len| len.checked_add(certificate.witnesses.len().checked_mul(16)?))
                .ok_or(PrimalityProofWireError::LengthOverflow)?;
        }

        if total_entries > PRIMALITY_PROOF_MAX_ENTRIES {
            return Err(PrimalityProofWireError::TooManyEntries { count: total_entries });
        }
        if encoded_len > PRIMALITY_PROOF_MAX_WIRE_BYTES {
            return Err(PrimalityProofWireError::InputTooLarge { length: encoded_len });
        }
        self.verify().map_err(PrimalityProofWireError::InvalidProof)?;

        let mut bytes = Vec::with_capacity(encoded_len);
        bytes.extend_from_slice(PRIMALITY_PROOF_WIRE_MAGIC);
        bytes.extend_from_slice(&self.value.to_be_bytes());
        bytes.extend_from_slice(&certificate_count.to_be_bytes());
        for certificate in &self.certificates {
            bytes.extend_from_slice(&certificate.prime.to_be_bytes());
            let factor_count = u32::try_from(certificate.predecessor_factors.len())
                .map_err(|_| PrimalityProofWireError::LengthOverflow)?;
            bytes.extend_from_slice(&factor_count.to_be_bytes());
            for &(factor, exponent) in &certificate.predecessor_factors {
                bytes.extend_from_slice(&factor.to_be_bytes());
                bytes.extend_from_slice(&exponent.to_be_bytes());
            }
            let witness_count = u32::try_from(certificate.witnesses.len())
                .map_err(|_| PrimalityProofWireError::LengthOverflow)?;
            bytes.extend_from_slice(&witness_count.to_be_bytes());
            for &(factor, base) in &certificate.witnesses {
                bytes.extend_from_slice(&factor.to_be_bytes());
                bytes.extend_from_slice(&base.to_be_bytes());
            }
        }

        debug_assert_eq!(bytes.len(), encoded_len);
        Ok(bytes)
    }

    /// Decode canonical proof bytes with size limits, strict framing, and verification.
    ///
    /// No allocation is driven by an unchecked count: every table and cumulative entry
    /// count is bounded before capacity is reserved. Trailing data and invalid proofs
    /// are rejected instead of normalized.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, PrimalityProofWireError> {
        if bytes.len() > PRIMALITY_PROOF_MAX_WIRE_BYTES {
            return Err(PrimalityProofWireError::InputTooLarge { length: bytes.len() });
        }

        let mut reader = PrimalityProofByteReader { bytes, offset: 0 };
        if reader.read(8)? != &PRIMALITY_PROOF_WIRE_MAGIC[..] {
            return Err(PrimalityProofWireError::InvalidMagicOrVersion);
        }
        let value = reader.read_u64()?;
        let certificate_count = reader.read_u32()?;
        if certificate_count > PRIMALITY_PROOF_MAX_CERTIFICATES {
            return Err(PrimalityProofWireError::TooManyCertificates {
                count: certificate_count,
            });
        }

        let mut total_entries = 0u64;
        let mut certificates = Vec::with_capacity(certificate_count as usize);
        for _ in 0..certificate_count {
            let prime = reader.read_u64()?;
            let factor_count = reader.read_u32()?;
            total_entries = total_entries
                .checked_add(u64::from(factor_count))
                .ok_or(PrimalityProofWireError::LengthOverflow)?;
            if total_entries > PRIMALITY_PROOF_MAX_ENTRIES {
                return Err(PrimalityProofWireError::TooManyEntries { count: total_entries });
            }

            let mut predecessor_factors = Vec::with_capacity(factor_count as usize);
            for _ in 0..factor_count {
                predecessor_factors.push((reader.read_u64()?, reader.read_u32()?));
            }

            let witness_count = reader.read_u32()?;
            total_entries = total_entries
                .checked_add(u64::from(witness_count))
                .ok_or(PrimalityProofWireError::LengthOverflow)?;
            if total_entries > PRIMALITY_PROOF_MAX_ENTRIES {
                return Err(PrimalityProofWireError::TooManyEntries { count: total_entries });
            }

            let mut witnesses = Vec::with_capacity(witness_count as usize);
            for _ in 0..witness_count {
                witnesses.push((reader.read_u64()?, reader.read_u64()?));
            }

            certificates.push(PrimalityCertificate {
                prime,
                predecessor_factors,
                witnesses,
            });
        }

        if reader.offset != bytes.len() {
            return Err(PrimalityProofWireError::TrailingBytes {
                count: bytes.len() - reader.offset,
            });
        }

        let proof = Self { value, certificates };
        proof.verify().map_err(PrimalityProofWireError::InvalidProof)?;
        let canonical = proof.to_canonical_bytes()?;
        if canonical.as_slice() != bytes {
            return Err(PrimalityProofWireError::NonCanonicalEncoding);
        }
        Ok(proof)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProofByteReaderError {
    TruncatedInput,
    LengthOverflow,
}

impl From<ProofByteReaderError> for PrimalityProofWireError {
    fn from(error: ProofByteReaderError) -> Self {
        match error {
            ProofByteReaderError::TruncatedInput => Self::TruncatedInput,
            ProofByteReaderError::LengthOverflow => Self::LengthOverflow,
        }
    }
}

impl From<ProofByteReaderError> for CertifiedFactorizationWireError {
    fn from(error: ProofByteReaderError) -> Self {
        match error {
            ProofByteReaderError::TruncatedInput => Self::TruncatedInput,
            ProofByteReaderError::LengthOverflow => Self::LengthOverflow,
        }
    }
}

struct PrimalityProofByteReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PrimalityProofByteReader<'a> {
    fn read(&mut self, length: usize) -> Result<&'a [u8], ProofByteReaderError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ProofByteReaderError::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(ProofByteReaderError::TruncatedInput);
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32, ProofByteReaderError> {
        let bytes = self.read(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, ProofByteReaderError> {
        let bytes = self.read(8)?;
        Ok(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
}

/// GCD helper used by the primality-proof checker, separate from the search engine.
fn certificate_gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Why a factorization certificate could not be verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactorizationVerificationError {
    /// Zero has no prime factorization under this certificate format.
    ZeroInput,
    /// In-memory factor lists use the same resource ceiling as the wire format.
    TooManyFactors { count: usize },
    /// Canonical factors must be strictly increasing; duplicates belong in the exponent.
    FactorsNotStrictlyIncreasing { previous: u64, current: u64 },
    /// A factor with exponent zero is not a canonical factorization entry.
    ZeroExponent { factor: u64 },
    /// The claimed factor is not prime under the deterministic u64 primality verifier.
    CompositeFactor { factor: u64 },
    /// The claimed prime powers cannot be multiplied without exceeding u64.
    ProductOverflow { factor: u64 },
    /// The exact product of the claimed prime powers differs from the certified input.
    ProductMismatch { expected: u64, actual: u64 },
}

impl PrimeFactorization {
    /// Discard derived HDC/float fields and retain the arithmetic certificate only.
    pub fn certificate(&self) -> FactorizationCertificate {
        FactorizationCertificate {
            value: self.value,
            factors: self.factors.clone(),
        }
    }
}

/// A factorization certificate whose distinct prime factors each carry a
/// recursively checkable Pocklington proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertifiedFactorizationCertificate {
    pub factorization: FactorizationCertificate,
    /// Proofs align one-to-one with factorization factors in strictly increasing order.
    pub prime_proofs: Vec<PrimalityProof>,
}

/// Typed failure while checking or constructing a proof-carrying factorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertifiedFactorizationError {
    InvalidFactorization(FactorizationVerificationError),
    TooManyPrimeProofs { count: usize },
    PrimeProofCountMismatch { factors: usize, proofs: usize },
    PrimeProofRootMismatch { factor: u64, proof_root: u64 },
    InvalidPrimeProof { factor: u64, reason: PrimalityProofError },
    PrimeProofGenerationFailed { factor: u64, reason: PrimalityProofError },
}

/// Versioned, bounded wire-format failures for proof-carrying factorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertifiedFactorizationWireError {
    InvalidMagicOrVersion,
    NonCanonicalEncoding,
    TruncatedInput,
    TrailingBytes { count: usize },
    InputTooLarge { length: usize },
    TooManyFactors { count: u32 },
    TooManyProofs { count: u32 },
    LengthOverflow,
    InvalidCertificate(CertifiedFactorizationError),
    InvalidNestedProof { index: u32, reason: PrimalityProofWireError },
}

const CERTIFIED_FACTORIZATION_WIRE_MAGIC: &[u8; 8] = b"SYMFCER1";
const CERTIFIED_FACTORIZATION_MAX_FACTORS: u32 = 4_096;
const CERTIFIED_FACTORIZATION_MAX_PROOFS: u32 = 4_096;
const CERTIFIED_FACTORIZATION_MAX_WIRE_BYTES: usize = 1_048_576;

/// Check the canonical factor list and exact product without making any primality
/// assumption. The certified variant follows this with a Pocklington proof per factor.
fn verify_factorization_structure(
    value: u64,
    factors: &[(u64, u32)],
) -> Result<(), FactorizationVerificationError> {
    if value == 0 {
        return Err(FactorizationVerificationError::ZeroInput);
    }
    if factors.len() > CERTIFIED_FACTORIZATION_MAX_FACTORS as usize {
        return Err(FactorizationVerificationError::TooManyFactors { count: factors.len() });
    }
    let mut previous = 1u64;
    let mut reconstructed = 1u64;
    for &(factor, exponent) in factors {
        if factor <= previous {
            return Err(FactorizationVerificationError::FactorsNotStrictlyIncreasing {
                previous,
                current: factor,
            });
        }
        if exponent == 0 {
            return Err(FactorizationVerificationError::ZeroExponent { factor });
        }
        previous = factor;
        for _ in 0..exponent {
            reconstructed = reconstructed.checked_mul(factor).ok_or(
                FactorizationVerificationError::ProductOverflow { factor },
            )?;
        }
    }
    if reconstructed != value {
        return Err(FactorizationVerificationError::ProductMismatch {
            expected: value,
            actual: reconstructed,
        });
    }
    Ok(())
}

impl CertifiedFactorizationCertificate {
    /// Verify exact factor reconstruction and every factor's recursive primality proof.
    ///
    /// Unlike FactorizationCertificate::verify, this path does not use the deterministic
    /// Miller-Rabin helper: primality is established only by the explicit Pocklington
    /// certificate bundles.
    pub fn verify(&self) -> Result<(), CertifiedFactorizationError> {
        if self.prime_proofs.len() > CERTIFIED_FACTORIZATION_MAX_PROOFS as usize {
            return Err(CertifiedFactorizationError::TooManyPrimeProofs {
                count: self.prime_proofs.len(),
            });
        }
        verify_factorization_structure(
            self.factorization.value,
            &self.factorization.factors,
        )
        .map_err(CertifiedFactorizationError::InvalidFactorization)?;

        if self.prime_proofs.len() != self.factorization.factors.len() {
            return Err(CertifiedFactorizationError::PrimeProofCountMismatch {
                factors: self.factorization.factors.len(),
                proofs: self.prime_proofs.len(),
            });
        }

        for (&(factor, _), proof) in self.factorization.factors.iter().zip(&self.prime_proofs) {
            if proof.value != factor {
                return Err(CertifiedFactorizationError::PrimeProofRootMismatch {
                    factor,
                    proof_root: proof.value,
                });
            }
            proof.verify().map_err(|reason| {
                CertifiedFactorizationError::InvalidPrimeProof { factor, reason }
            })?;
        }
        Ok(())
    }
    /// Encode a verified factorization plus all per-factor Pocklington proofs.
    /// Nested primality proofs use their own versioned canonical encoding.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, CertifiedFactorizationWireError> {
        let factor_count = u32::try_from(self.factorization.factors.len())
            .map_err(|_| CertifiedFactorizationWireError::LengthOverflow)?;
        let proof_count = u32::try_from(self.prime_proofs.len())
            .map_err(|_| CertifiedFactorizationWireError::LengthOverflow)?;
        if factor_count > CERTIFIED_FACTORIZATION_MAX_FACTORS {
            return Err(CertifiedFactorizationWireError::TooManyFactors { count: factor_count });
        }
        if proof_count > CERTIFIED_FACTORIZATION_MAX_PROOFS {
            return Err(CertifiedFactorizationWireError::TooManyProofs { count: proof_count });
        }
        self.verify().map_err(CertifiedFactorizationWireError::InvalidCertificate)?;

        let mut nested = Vec::with_capacity(self.prime_proofs.len());
        let mut encoded_len = 8usize + 8 + 4 + 4;
        encoded_len = encoded_len
            .checked_add(self.factorization.factors.len().checked_mul(12).ok_or(
                CertifiedFactorizationWireError::LengthOverflow,
            )?)
            .ok_or(CertifiedFactorizationWireError::LengthOverflow)?;

        for (index, proof) in self.prime_proofs.iter().enumerate() {
            let proof_bytes = proof.to_canonical_bytes().map_err(|reason| {
                CertifiedFactorizationWireError::InvalidNestedProof {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    reason,
                }
            })?;
            encoded_len = encoded_len
                .checked_add(4)
                .and_then(|len| len.checked_add(proof_bytes.len()))
                .ok_or(CertifiedFactorizationWireError::LengthOverflow)?;
            if encoded_len > CERTIFIED_FACTORIZATION_MAX_WIRE_BYTES {
                return Err(CertifiedFactorizationWireError::InputTooLarge { length: encoded_len });
            }
            nested.push(proof_bytes);
        }

        if encoded_len > CERTIFIED_FACTORIZATION_MAX_WIRE_BYTES {
            return Err(CertifiedFactorizationWireError::InputTooLarge { length: encoded_len });
        }

        let mut bytes = Vec::with_capacity(encoded_len);
        bytes.extend_from_slice(CERTIFIED_FACTORIZATION_WIRE_MAGIC);
        bytes.extend_from_slice(&self.factorization.value.to_be_bytes());
        bytes.extend_from_slice(&factor_count.to_be_bytes());
        for &(factor, exponent) in &self.factorization.factors {
            bytes.extend_from_slice(&factor.to_be_bytes());
            bytes.extend_from_slice(&exponent.to_be_bytes());
        }
        bytes.extend_from_slice(&proof_count.to_be_bytes());
        for proof_bytes in nested {
            let length = u32::try_from(proof_bytes.len())
                .map_err(|_| CertifiedFactorizationWireError::LengthOverflow)?;
            bytes.extend_from_slice(&length.to_be_bytes());
            bytes.extend_from_slice(&proof_bytes);
        }

        debug_assert_eq!(bytes.len(), encoded_len);
        Ok(bytes)
    }

    /// Decode a proof-carrying factorization with strict version, size, framing,
    /// canonical factor ordering, and recursive proof verification.
    pub fn from_canonical_bytes(
        bytes: &[u8],
    ) -> Result<Self, CertifiedFactorizationWireError> {
        if bytes.len() > CERTIFIED_FACTORIZATION_MAX_WIRE_BYTES {
            return Err(CertifiedFactorizationWireError::InputTooLarge { length: bytes.len() });
        }

        let mut reader = PrimalityProofByteReader { bytes, offset: 0 };
        if reader.read(8)? != &CERTIFIED_FACTORIZATION_WIRE_MAGIC[..] {
            return Err(CertifiedFactorizationWireError::InvalidMagicOrVersion);
        }
        let value = reader.read_u64()?;
        let factor_count = reader.read_u32()?;
        if factor_count > CERTIFIED_FACTORIZATION_MAX_FACTORS {
            return Err(CertifiedFactorizationWireError::TooManyFactors { count: factor_count });
        }

        let mut factors = Vec::with_capacity(factor_count as usize);
        for _ in 0..factor_count {
            factors.push((reader.read_u64()?, reader.read_u32()?));
        }

        let proof_count = reader.read_u32()?;
        if proof_count > CERTIFIED_FACTORIZATION_MAX_PROOFS {
            return Err(CertifiedFactorizationWireError::TooManyProofs { count: proof_count });
        }
        if proof_count != factor_count {
            return Err(CertifiedFactorizationWireError::InvalidCertificate(
                CertifiedFactorizationError::PrimeProofCountMismatch {
                    factors: factor_count as usize,
                    proofs: proof_count as usize,
                },
            ));
        }

        let mut prime_proofs = Vec::with_capacity(proof_count as usize);
        for index in 0..proof_count {
            let length = reader.read_u32()? as usize;
            if length > PRIMALITY_PROOF_MAX_WIRE_BYTES {
                return Err(CertifiedFactorizationWireError::InputTooLarge { length });
            }
            let proof_bytes = reader.read(length)?;
            let proof = PrimalityProof::from_canonical_bytes(proof_bytes).map_err(|reason| {
                CertifiedFactorizationWireError::InvalidNestedProof { index, reason }
            })?;
            prime_proofs.push(proof);
        }

        if reader.offset != bytes.len() {
            return Err(CertifiedFactorizationWireError::TrailingBytes {
                count: bytes.len() - reader.offset,
            });
        }

        let certificate = Self {
            factorization: FactorizationCertificate { value, factors },
            prime_proofs,
        };
        certificate
            .verify()
            .map_err(CertifiedFactorizationWireError::InvalidCertificate)?;
        let canonical = certificate.to_canonical_bytes()?;
        if canonical.as_slice() != bytes {
            return Err(CertifiedFactorizationWireError::NonCanonicalEncoding);
        }
        Ok(certificate)
    }

}

impl FactorizationCertificate {
    /// Verify canonical form, every prime factor, and exact reconstruction.
    ///
    /// Prime checks use a separate deterministic Miller–Rabin implementation with
    /// the seven-base u64 witness set, not Pollard–Brent or the search engine's
    /// selected witness schedule. This is a reproducible machine-integer verifier;
    /// it is not a recursively checkable formal proof object.
    pub fn verify(&self) -> Result<(), FactorizationVerificationError> {
        verify_factorization_structure(self.value, &self.factors)?;

        for &(prime, _) in &self.factors {
            if !certificate_is_prime_u64(prime) {
                return Err(FactorizationVerificationError::CompositeFactor { factor: prime });
            }
        }
        Ok(())
    }
}

// These modular-arithmetic helpers are intentionally separate from ModularRing and
// NumberTheoryEngine::miller_rabin, so certificate checking does not reuse the search
// engine's arithmetic path. A u64-by-u64 product fits exactly in u128.
fn certificate_mul_mod(a: u64, b: u64, modulus: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(modulus)) as u64
}

fn certificate_pow_mod(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = certificate_mul_mod(result, base, modulus);
        }
        base = certificate_mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    result
}

/// Deterministic primality check over the complete u64 domain using a witness set
/// different from the factorization search engine's first-prime schedule.
fn certificate_is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }

    // These include every prime divisor of the seven witness constants. Testing
    // equality first handles those primes explicitly; other multiples are composite.
    const SMALL_PRIMES: [u64; 16] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 73, 193, 407_521, 299_210_837,
    ];
    for prime in SMALL_PRIMES {
        if n == prime {
            return true;
        }
        if n % prime == 0 {
            return false;
        }
    }

    let mut odd_part = n - 1;
    let mut powers_of_two = 0u32;
    while odd_part & 1 == 0 {
        odd_part >>= 1;
        powers_of_two += 1;
    }

    const WITNESSES: [u64; 7] = [2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022];
    'witness: for witness in WITNESSES {
        let base = witness % n;
        if base == 0 {
            continue;
        }

        let mut x = certificate_pow_mod(base, odd_part, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..powers_of_two {
            x = certificate_mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }

    true
}

/// Number theory engine for HDC-based arithmetic
pub struct NumberTheoryEngine {
    multiplication: BinaryHV,
    addition: BinaryHV,
    zero: BinaryHV,
    successor: BinaryHV,
}

impl NumberTheoryEngine {
    pub fn new() -> Self {
        let prims = PrimitiveSystem::global();
        Self {
            multiplication: prims
                .get("MULTIPLICATION")
                .expect("MULTIPLICATION primitive must exist")
                .encoding,
            addition: prims
                .get("ADDITION")
                .expect("ADDITION primitive must exist")
                .encoding,
            zero: prims
                .get("ZERO")
                .expect("ZERO primitive must exist")
                .encoding,
            successor: prims
                .get("SUCCESSOR")
                .expect("SUCCESSOR primitive must exist")
                .encoding,
        }
    }

    /// Bounded deterministic-seed Pollard-Brent attempt to find a nontrivial factor.
    ///
    /// Returns `None` if the fixed work budget is exhausted; callers retain a
    /// trial-division fallback so the factorization API remains complete.
    fn pollard_brent_factor(&self, n: u64) -> Option<u64> {
        if n.is_multiple_of(2) {
            return Some(2);
        }
        if n.is_multiple_of(3) {
            return Some(3);
        }
        if n < 5 {
            return None;
        }

        const ATTEMPTS: u64 = 16;
        const ITERATION_BUDGET: u64 = 250_000;
        const BLOCK_SIZE: u64 = 128;
        let ring = ModularRing::new(n);

        for attempt in 0..ATTEMPTS {
            // Fixed seeds make behavior reproducible across runs and CI machines.
            let mut y = 2 + attempt % (n - 3);
            let c = 1 + 2 * attempt;
            let mut g = 1u64;
            let mut r = 1u64;
            let mut q = 1u64;
            let mut x = 0u64;
            let mut ys = 0u64;
            let mut iterations = 0u64;

            while g == 1 && iterations < ITERATION_BUDGET {
                x = y;

                // Keep each doubling phase inside the per-attempt work budget.
                if r > ITERATION_BUDGET - iterations {
                    break;
                }
                for _ in 0..r {
                    y = ring.add(ring.multiply(y, y), c);
                }
                iterations += r;

                let mut k = 0u64;
                while k < r && g == 1 {
                    ys = y;
                    let limit = BLOCK_SIZE.min(r - k).min(ITERATION_BUDGET - iterations);
                    if limit == 0 {
                        break;
                    }
                    for _ in 0..limit {
                        y = ring.add(ring.multiply(y, y), c);
                        q = ring.multiply(q, x.abs_diff(y));
                    }
                    iterations += limit;
                    g = self.gcd(q, n);
                    k += limit;
                }

                if g != 1 || iterations >= ITERATION_BUDGET || r > ITERATION_BUDGET / 2 {
                    break;
                }
                r *= 2;
            }

            // A batch can make q divisible by n. Recover by checking each step
            // from the beginning of the final batch, still under the same budget.
            if g == n {
                while iterations < ITERATION_BUDGET {
                    ys = ring.add(ring.multiply(ys, ys), c);
                    iterations += 1;
                    g = self.gcd(x.abs_diff(ys), n);
                    if g > 1 {
                        break;
                    }
                }
            }

            if g > 1 && g < n {
                return Some(g);
            }
        }

        None
    }

    /// Recursively collect prime factors. Miller-Rabin handles prime leaves;
    /// Pollard-Brent handles hard composites, with exact trial division as fallback.
    fn collect_prime_factors(&self, n: u64, out: &mut Vec<u64>) {
        if n <= 1 {
            return;
        }
        if self.miller_rabin(n) {
            out.push(n);
            return;
        }

        if let Some(factor) = self.pollard_brent_factor(n) {
            self.collect_prime_factors(factor, out);
            self.collect_prime_factors(n / factor, out);
            return;
        }

        // Completeness fallback: this may be slower for an unusually difficult
        // input, but it preserves exactness if the bounded rho attempts fail.
        let mut remaining = n;
        let mut divisor = 2u64;
        while divisor <= remaining / divisor {
            while remaining.is_multiple_of(divisor) {
                out.push(divisor);
                remaining /= divisor;
            }
            divisor += if divisor == 2 { 1 } else { 2 };
        }
        if remaining > 1 {
            out.push(remaining);
        }
    }

    /// Prime factorization using deterministic primality testing and Pollard-Brent,
    /// returning an HDC encoding of the factorization.
    pub fn factorize(&self, n: u64) -> PrimeFactorization {
        if n == 0 {
            return PrimeFactorization {
                value: 0,
                factors: vec![],
                encoding: self.zero,
                phi: 0.0,
            };
        }

        if n == 1 {
            return PrimeFactorization {
                value: 1,
                factors: vec![],
                encoding: self.successor.bind(&self.zero),
                phi: 1.0,
            };
        }

        let mut prime_factors = Vec::new();
        self.collect_prime_factors(n, &mut prime_factors);
        prime_factors.sort_unstable();

        let mut factors: Vec<(u64, u32)> = Vec::new();
        for prime in prime_factors {
            if let Some((last_prime, exponent)) = factors.last_mut() {
                if *last_prime == prime {
                    *exponent += 1;
                    continue;
                }
            }
            factors.push((prime, 1));
        }

        // Encode as bundle of MULTIPLICATION.bind(prime_enc).bind(exp_enc)
        let mut factor_encodings = Vec::new();
        for (prime, exponent) in &factors {
            let prime_enc = BinaryHV::random(seed_from_name(&format!("prime_{prime}")));
            let exp_enc = BinaryHV::random(seed_from_name(&format!("exp_{exponent}")));
            let factor_enc = self.multiplication.bind(&prime_enc).bind(&exp_enc);
            factor_encodings.push(factor_enc);
        }

        let encoding = if factor_encodings.is_empty() {
            self.successor.bind(&self.zero)
        } else {
            BinaryHV::bundle(&factor_encodings)
        };

        // Compute in integer arithmetic first, then convert once for the legacy API.
        let phi = phi_from_distinct_prime_factors(n, &factors) as f64;

        PrimeFactorization {
            value: n,
            factors,
            encoding,
            phi,
        }
    }

    /// Create a proof-carrying factorization certificate for a u64 input.
    ///
    /// The result can be checked without trusting the factorization search routine or
    /// its Miller-Rabin primality schedule. Zero is rejected; one has an empty
    /// factorization and needs no prime proofs.
    pub fn certified_factorization(
        &self,
        value: u64,
    ) -> Result<CertifiedFactorizationCertificate, CertifiedFactorizationError> {
        let factorization = self.factorize(value).certificate();
        verify_factorization_structure(factorization.value, &factorization.factors)
            .map_err(CertifiedFactorizationError::InvalidFactorization)?;

        let mut prime_proofs = Vec::with_capacity(factorization.factors.len());
        for &(factor, _) in &factorization.factors {
            let proof = self.primality_proof(factor).map_err(|reason| {
                CertifiedFactorizationError::PrimeProofGenerationFailed { factor, reason }
            })?;
            prime_proofs.push(proof);
        }

        let certificate = CertifiedFactorizationCertificate {
            factorization,
            prime_proofs,
        };
        certificate.verify()?;
        Ok(certificate)
    }

    /// Build a recursive Pocklington primality proof for a prime in the u64 domain.
    ///
    /// Generation factors each predecessor (p - 1), then searches bounded witness
    /// bases. Failure to find a witness within the search cap returns a typed error;
    /// it never emits an unchecked or incomplete proof.
    pub fn primality_proof(&self, value: u64) -> Result<PrimalityProof, PrimalityProofError> {
        if value < 2 {
            return Err(PrimalityProofError::InvalidRoot { value });
        }
        let mut certificates = std::collections::BTreeMap::new();
        self.collect_primality_proof_nodes(value, &mut certificates)?;
        let proof = PrimalityProof {
            value,
            certificates: certificates.into_values().collect(),
        };
        proof.verify()?;
        Ok(proof)
    }

    fn collect_primality_proof_nodes(
        &self,
        value: u64,
        certificates: &mut std::collections::BTreeMap<u64, PrimalityCertificate>,
    ) -> Result<(), PrimalityProofError> {
        if certificates.contains_key(&value) {
            return Ok(());
        }
        if value == 2 {
            certificates.insert(value, PrimalityCertificate {
                prime: value,
                predecessor_factors: Vec::new(),
                witnesses: Vec::new(),
            });
            return Ok(());
        }
        if value < 3 || value.is_multiple_of(2) || !self.is_prime(value) {
            return Err(PrimalityProofError::CompositeCandidate { prime: value });
        }

        let predecessor = self.factorize(value - 1);
        if predecessor.certificate().verify().is_err() {
            return Err(PrimalityProofError::GenerationFactorizationInvalid { value });
        }

        for &(factor, _) in &predecessor.factors {
            self.collect_primality_proof_nodes(factor, certificates)?;
        }

        let mut witnesses = Vec::with_capacity(predecessor.factors.len());
        const MAX_WITNESS_BASE: u64 = 4096;
        for &(factor, _) in &predecessor.factors {
            let max_base = (value - 1).min(MAX_WITNESS_BASE);
            let mut found = None;
            for base in 2..=max_base {
                if certificate_pow_mod(base, value - 1, value) != 1 {
                    continue;
                }
                let partial = certificate_pow_mod(base, (value - 1) / factor, value);
                if certificate_gcd(partial.abs_diff(1), value) == 1 {
                    found = Some(base);
                    break;
                }
            }
            let base = found.ok_or(PrimalityProofError::WitnessSearchExhausted {
                prime: value,
                factor,
            })?;
            witnesses.push((factor, base));
        }

        certificates.insert(value, PrimalityCertificate {
            prime: value,
            predecessor_factors: predecessor.factors,
            witnesses,
        });
        Ok(())
    }

    /// Widened extended Euclidean algorithm over the full i64 input domain.
    ///
    /// Returns (g, x, y) with non-negative g and a*x + b*y = g exactly.
    /// A u64 is required for g = abs(i64::MIN); i128 coefficients avoid
    /// intermediate overflow in the Euclidean recurrence.
    pub fn extended_gcd_wide(&self, a: i64, b: i64) -> (u64, i128, i128) {
        let mut old_r = i128::from(a);
        let mut r = i128::from(b);
        let mut old_s = 1i128;
        let mut s = 0i128;
        let mut old_t = 0i128;
        let mut t = 1i128;

        while r != 0 {
            let quotient = old_r / r;
            let next_r = old_r - quotient * r;
            old_r = r;
            r = next_r;

            let next_s = old_s - quotient * s;
            old_s = s;
            s = next_s;

            let next_t = old_t - quotient * t;
            old_t = t;
            t = next_t;
        }

        if old_r < 0 {
            old_r = -old_r;
            old_s = -old_s;
            old_t = -old_t;
        }

        // The absolute GCD of two i64 values is at most 2^63, which fits u64.
        let gcd = u64::try_from(old_r).expect("absolute GCD of i64 values fits u64");
        (gcd, old_s, old_t)
    }

    /// Extended Euclidean algorithm: returns (gcd, x, y) where ax + by = gcd.
    ///
    /// This compatibility signature uses i64 for all outputs and therefore
    /// panics explicitly when the exact result cannot be represented. Use
    /// `extended_gcd_wide` to support the entire i64 input domain.
    pub fn extended_gcd(&self, a: i64, b: i64) -> (i64, i64, i64) {
        let (gcd, x, y) = self.extended_gcd_wide(a, b);
        (
            i64::try_from(gcd)
                .expect("GCD exceeds i64::MAX; use extended_gcd_wide to preserve the exact result"),
            i64::try_from(x)
                .expect("extended GCD coefficient exceeds i64 range; use extended_gcd_wide"),
            i64::try_from(y)
                .expect("extended GCD coefficient exceeds i64 range; use extended_gcd_wide"),
        )
    }

    /// Greatest common divisor using Euclidean algorithm
    pub fn gcd(&self, a: u64, b: u64) -> u64 {
        let mut a = a;
        let mut b = b;
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Checked least common multiple.
    ///
    /// Returns `None` when the exact result is greater than `u64::MAX`.
    /// Dividing before multiplying minimizes the intermediate and avoids wraparound.
    pub fn checked_lcm(&self, a: u64, b: u64) -> Option<u64> {
        if a == 0 || b == 0 {
            return Some(0);
        }
        (a / self.gcd(a, b)).checked_mul(b)
    }

    /// Least common multiple, panicking rather than silently wrapping on overflow.
    ///
    /// Use `checked_lcm` when the inputs may be outside a known-safe range.
    pub fn lcm(&self, a: u64, b: u64) -> u64 {
        self.checked_lcm(a, b)
            .expect("least common multiple exceeds u64::MAX; use checked_lcm to handle overflow")
    }

    /// Sieve of Eratosthenes
    pub fn primes_up_to(&self, n: u64) -> Vec<(u64, BinaryHV)> {
        if n < 2 {
            return vec![];
        }

        let n_size = n as usize + 1;
        let mut is_prime = vec![true; n_size];
        is_prime[0] = false;
        is_prime[1] = false;

        let limit = (n as f64).sqrt() as usize;
        for i in 2..=limit {
            if is_prime[i] {
                let mut j = i * i;
                while j < n_size {
                    is_prime[j] = false;
                    j += i;
                }
            }
        }

        is_prime
            .iter()
            .enumerate()
            .filter(|&(_, &prime)| prime)
            .map(|(i, _)| {
                let p = i as u64;
                let encoding = BinaryHV::random(seed_from_name(&format!("prime_{p}")));
                (p, encoding)
            })
            .collect()
    }

    /// Deterministic primality test for the entire u64 domain.
    ///
    /// Uses Miller–Rabin with witness sets selected by proven numeric bounds,
    /// avoiding floating-point square-root rounding and unbounded O(sqrt(n))
    /// trial division for large inputs.
    pub fn is_prime(&self, n: u64) -> bool {
        self.miller_rabin(n)
    }

    /// Fermat primality test: for each witness a, check a^(n-1) mod n == 1
    pub fn fermat_test(&self, n: u64, witnesses: &[u64]) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n.is_multiple_of(2) {
            return false;
        }

        let ring = ModularRing::new(n);
        for &a in witnesses {
            if a >= n {
                continue;
            }
            if self.gcd(a, n) != 1 {
                return false;
            }
            if ring.power(a, n - 1) != 1 {
                return false;
            }
        }
        true
    }

    /// Miller-Rabin primality test — deterministic for n < 3.3×10^24.
    ///
    /// Unlike Fermat, this correctly rejects Carmichael numbers (561, 1105, 1729).
    /// Uses known witness sets that make the test deterministic (not probabilistic)
    /// for numbers up to specific bounds:
    /// - n < 2,047: witnesses {2}
    /// - n < 1,373,653: witnesses {2, 3}
    /// - n < 3,215,031,751: witnesses {2, 3, 5, 7}
    /// - n < 3,317,044,064,679,887,385,961,981: witnesses {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
    pub fn miller_rabin(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        // Write n-1 = 2^r * d where d is odd
        let mut d = n - 1;
        let mut r = 0u32;
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }

        // Deterministic witness sets by range
        let witnesses: &[u64] = if n < 2_047 {
            &[2]
        } else if n < 1_373_653 {
            &[2, 3]
        } else if n < 3_215_031_751 {
            &[2, 3, 5, 7]
        } else {
            // Covers all u64 values (deterministic for n < 3.3×10^24)
            &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        };

        let ring = ModularRing::new(n);

        'witness: for &a in witnesses {
            if a >= n {
                continue;
            }

            // Compute a^d mod n
            let mut x = ring.power(a, d);

            if x == 1 || x == n - 1 {
                continue 'witness;
            }

            // Square r-1 times
            for _ in 0..r - 1 {
                x = ring.multiply(x, x);
                if x == n - 1 {
                    continue 'witness;
                }
            }

            // If we never hit n-1, n is composite
            return false;
        }

        true
    }
}

impl Default for NumberTheoryEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Phase 1: IMO-level number theory primitives ─────────────────────────────

impl NumberTheoryEngine {
    /// Chinese Remainder Theorem. Given residues `[(a_i, m_i)]` representing
    /// `x ≡ a_i (mod m_i)`, returns `Some((x, M))` with `M = lcm(m_i)` and
    /// `0 ≤ x < M`, or `None` if the system is inconsistent.
    ///
    /// Moduli do NOT need to be pairwise coprime — a consistent overlap is
    /// merged via extended gcd.
    pub fn crt(&self, residues: &[(i64, i64)]) -> Option<(i64, i64)> {
        if residues.is_empty() {
            return Some((0, 1));
        }
        if residues[0].1 <= 0 {
            return None;
        }
        let mut x = residues[0].0.rem_euclid(residues[0].1);
        let mut m = residues[0].1;
        for &(a_i, m_i) in &residues[1..] {
            if m_i <= 0 {
                return None;
            }
            let a_i = a_i.rem_euclid(m_i);
            let (g, p, _q) = self.extended_gcd(m, m_i);
            let diff = a_i - x;
            if diff % g != 0 {
                return None;
            }
            let lcm = m / g * m_i;
            let step = m_i / g;
            let k = ((diff / g) % step * (p % step)).rem_euclid(step);
            x = (x + k * m).rem_euclid(lcm);
            m = lcm;
        }
        Some((x, m))
    }

    /// Legendre symbol (a/p) for odd prime p. Returns -1, 0, or 1.
    pub fn legendre_symbol(&self, a: i64, p: i64) -> i32 {
        assert!(p > 2 && p % 2 == 1, "legendre_symbol requires odd prime p");
        let a_mod = a.rem_euclid(p);
        if a_mod == 0 {
            return 0;
        }
        let ring = ModularRing::new(p as u64);
        let v = ring.power(a_mod as u64, ((p - 1) / 2) as u64);
        if v == 1 {
            1
        } else if v == (p - 1) as u64 {
            -1
        } else {
            0
        }
    }

    /// Jacobi symbol (a/n) for odd positive n ≥ 1. Returns -1, 0, or 1.
    /// Generalizes Legendre to composite n via quadratic reciprocity.
    pub fn jacobi_symbol(&self, a: i64, n: i64) -> i32 {
        assert!(n > 0 && n % 2 == 1, "jacobi_symbol requires odd positive n");
        let mut a = a.rem_euclid(n);
        let mut n = n;
        let mut result: i32 = 1;
        while a != 0 {
            while a % 2 == 0 {
                a /= 2;
                let r = n % 8;
                if r == 3 || r == 5 {
                    result = -result;
                }
            }
            std::mem::swap(&mut a, &mut n);
            if a % 4 == 3 && n % 4 == 3 {
                result = -result;
            }
            a %= n;
        }
        if n == 1 { result } else { 0 }
    }

    /// Tonelli–Shanks: modular square root of `a` mod prime `p`.
    /// Returns `Some(r)` with r² ≡ a (mod p), or `None` if a is a non-residue.
    pub fn tonelli_shanks(&self, a: u64, p: u64) -> Option<u64> {
        if p == 2 {
            return Some(a % 2);
        }
        let a = a % p;
        if a == 0 {
            return Some(0);
        }
        if self.legendre_symbol(a as i64, p as i64) != 1 {
            return None;
        }
        let ring = ModularRing::new(p);
        if p % 4 == 3 {
            return Some(ring.power(a, (p + 1) / 4));
        }
        let mut q = p - 1;
        let mut s = 0u64;
        while q % 2 == 0 {
            q /= 2;
            s += 1;
        }
        let mut z = 2u64;
        while self.legendre_symbol(z as i64, p as i64) != -1 {
            z += 1;
        }
        let mut m = s;
        let mut c = ring.power(z, q);
        let mut t = ring.power(a, q);
        let mut r = ring.power(a, q.div_ceil(2));
        loop {
            if t == 1 {
                return Some(r);
            }
            let mut i = 0u64;
            let mut temp = t;
            while temp != 1 && i < m {
                temp = ring.multiply(temp, temp);
                i += 1;
            }
            if i == m {
                return None;
            }
            let mut b = c;
            for _ in 0..(m - i - 1) {
                b = ring.multiply(b, b);
            }
            m = i;
            c = ring.multiply(b, b);
            t = ring.multiply(t, c);
            r = ring.multiply(r, b);
        }
    }

    /// p-adic valuation v_p(n): the largest k such that p^k divides n.
    /// For n = 0 returns u32::MAX (conventionally ∞).
    pub fn p_adic_valuation(&self, n: i64, p: i64) -> u32 {
        if n == 0 {
            return u32::MAX;
        }
        let mut k = 0u32;
        let mut m = n.abs();
        while m % p == 0 {
            m /= p;
            k += 1;
        }
        k
    }

    /// Lifting the Exponent (LTE): computes v_p(a^n − b^n) under LTE hypotheses.
    /// Returns `Some(v)` if the lemma applies, `None` otherwise.
    ///
    /// p odd, p ∤ a, p ∤ b, p | (a−b):
    ///   v_p(a^n − b^n) = v_p(a−b) + v_p(n)
    /// p = 2, n odd, 2 | (a−b):
    ///   v_2(a^n − b^n) = v_2(a−b)
    /// p = 2, n even, 4 | (a−b):
    ///   v_2(a^n − b^n) = v_2(a−b) + v_2(n)
    /// p = 2, n even, 2 | (a−b) but 4 ∤ (a−b):
    ///   v_2(a^n − b^n) = v_2(a−b) + v_2(a+b) + v_2(n) − 1
    pub fn lifting_the_exponent(&self, p: i64, a: i64, b: i64, n: u32) -> Option<u32> {
        if p < 2 || n == 0 {
            return None;
        }
        if a % p == 0 || b % p == 0 {
            return None;
        }
        if (a - b) % p != 0 {
            return None;
        }
        let v_diff = self.p_adic_valuation(a - b, p);
        let v_n = self.p_adic_valuation(n as i64, p);
        if p % 2 == 1 {
            return Some(v_diff + v_n);
        }
        if n % 2 == 1 {
            return Some(v_diff);
        }
        if (a - b) % 4 == 0 {
            return Some(v_diff + v_n);
        }
        let v_sum = self.p_adic_valuation(a + b, 2);
        Some(v_diff + v_sum + v_n - 1)
    }

    /// Linear Diophantine equation ax + by = c.
    ///
    /// Returns `Some((x0, y0, dx, dy))`: particular solution `(x0, y0)` plus
    /// increment `(dx, dy)` so the general solution is
    /// `(x0 + k·dx, y0 − k·dy)` for k ∈ ℤ. Returns `None` when gcd(a,b) ∤ c
    /// or when a = b = 0 and c ≠ 0.
    pub fn linear_diophantine(&self, a: i64, b: i64, c: i64) -> Option<(i64, i64, i64, i64)> {
        if a == 0 && b == 0 {
            return if c == 0 { Some((0, 0, 1, 0)) } else { None };
        }
        let (g, p, q) = self.extended_gcd(a, b);
        if c % g != 0 {
            return None;
        }
        let mult = c / g;
        let x0 = p * mult;
        let y0 = q * mult;
        let dx = b / g;
        let dy = a / g;
        Some((x0, y0, dx, dy))
    }
}

/// Modular arithmetic ring Z/nZ
pub struct ModularRing {
    pub modulus: u64,
    pub modulus_encoding: BinaryHV,
}

impl ModularRing {
    /// Construct a modular ring only when its modulus is positive.
    pub fn try_new(modulus: u64) -> Option<Self> {
        if modulus == 0 {
            return None;
        }
        let modulus_encoding = BinaryHV::random(seed_from_name(&format!("mod_{modulus}")));
        Some(Self {
            modulus,
            modulus_encoding,
        })
    }

    /// Construct a modular ring, failing explicitly for the invalid zero modulus.
    pub fn new(modulus: u64) -> Self {
        Self::try_new(modulus).expect("modular arithmetic requires a non-zero modulus")
    }

    // The fields remain public for compatibility, so validate the invariant at
    // operation boundaries as well as at construction.
    fn assert_valid_modulus(&self) {
        assert!(self.modulus > 0, "modular arithmetic requires a non-zero modulus");
    }

    /// Modular addition without u64 intermediate overflow.
    pub fn add(&self, a: u64, b: u64) -> u64 {
        self.assert_valid_modulus();
        let modulus = u128::from(self.modulus);
        let a = u128::from(a % self.modulus);
        let b = u128::from(b % self.modulus);
        ((a + b) % modulus) as u64
    }

    /// Modular multiplication
    pub fn multiply(&self, a: u64, b: u64) -> u64 {
        self.assert_valid_modulus();
        let a = a % self.modulus;
        let b = b % self.modulus;
        ((a as u128 * b as u128) % self.modulus as u128) as u64
    }

    /// Modular exponentiation by squaring
    pub fn power(&self, base: u64, exp: u64) -> u64 {
        self.assert_valid_modulus();
        if exp == 0 {
            return 1 % self.modulus;
        }

        let mut result = 1u64;
        let mut base = base % self.modulus;
        let mut exp = exp;

        while exp > 0 {
            if exp % 2 == 1 {
                result = self.multiply(result, base);
            }
            base = self.multiply(base, base);
            exp /= 2;
        }

        result
    }

    /// Modular multiplicative inverse for the full u64 modulus domain.
    ///
    /// Inputs are reduced modulo the ring modulus before the Euclidean algorithm;
    /// no u64-to-i64 casts or signed remainder corrections are used.
    pub fn inverse(&self, a: u64) -> Option<u64> {
        self.assert_valid_modulus();
        let modulus = i128::from(self.modulus);
        let mut old_r = i128::from(a % self.modulus);
        let mut r = modulus;
        let mut old_s = 1i128;
        let mut s = 0i128;

        while r != 0 {
            let quotient = old_r / r;
            let next_r = old_r - quotient * r;
            old_r = r;
            r = next_r;

            let next_s = old_s - quotient * s;
            old_s = s;
            s = next_s;
        }

        if old_r != 1 {
            return None;
        }

        Some(old_s.rem_euclid(modulus) as u64)
    }

    /// Check if element is a unit (has multiplicative inverse)
    pub fn is_unit(&self, a: u64) -> bool {
        self.assert_valid_modulus();
        let engine = NumberTheoryEngine::new();
        engine.gcd(a, self.modulus) == 1
    }

    /// Euler's totient function, computed from the distinct prime factors of the modulus.
    ///
    /// Uses phi(n) = n * product(1 - 1/p) over distinct prime factors p of n,
    /// avoiding the previous linear scan over every residue. Prime moduli therefore
    /// benefit from the deterministic Miller-Rabin fast path in factorize().
    /// By convention, phi(1) = 1.
    pub fn euler_totient(&self) -> u64 {
        self.assert_valid_modulus();

        let factorization = NumberTheoryEngine::new().factorize(self.modulus);
        phi_from_distinct_prime_factors(self.modulus, &factorization.factors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorize_60() {
        let engine = NumberTheoryEngine::new();
        let f = engine.factorize(60);
        assert_eq!(f.factors, vec![(2, 2), (3, 1), (5, 1)]);
    }

    #[test]
    fn test_factorize_1() {
        let engine = NumberTheoryEngine::new();
        let f = engine.factorize(1);
        assert!(f.factors.is_empty());
    }

    #[test]
    fn test_factorize_prime() {
        let engine = NumberTheoryEngine::new();
        let f = engine.factorize(17);
        assert_eq!(f.factors, vec![(17, 1)]);
    }

    #[test]
    fn test_factorize_large_semiprime_uses_pollard_brent() {
        let engine = NumberTheoryEngine::new();
        let p = 1_000_000_007u64;
        let q = 1_000_000_009u64;
        let semiprime = p * q;
        let factorization = engine.factorize(semiprime);
        assert_eq!(factorization.value, semiprime);
        assert_eq!(factorization.factors, vec![(p, 1u32), (q, 1u32)]);

        let reconstructed = factorization
            .factors
            .iter()
            .fold(1u128, |product, &(prime, exponent)| {
                product * u128::from(prime).pow(exponent)
            });
        assert_eq!(reconstructed, u128::from(semiprime));
    }

    #[test]
    fn test_factorize_near_u64_limit_semiprime() {
        let engine = NumberTheoryEngine::new();
        let p = 4_294_967_291u64;
        let q = 4_294_967_279u64;
        assert!(engine.miller_rabin(p));
        assert!(engine.miller_rabin(q));

        let semiprime = p * q;
        assert!(semiprime > u64::MAX - 100_000_000_000);
        let factorization = engine.factorize(semiprime);
        assert_eq!(factorization.factors, vec![(q, 1u32), (p, 1u32)]);

        let reconstructed = factorization
            .factors
            .iter()
            .fold(1u128, |product, &(prime, exponent)| {
                product * u128::from(prime).pow(exponent)
            });
        assert_eq!(reconstructed, u128::from(semiprime));
    }

    #[test]
    fn test_certified_factorization_proves_each_prime_factor() {
        let engine = NumberTheoryEngine::new();
        for value in [
            1u64,
            2,
            30,
            3_215_031_751,
            1_000_000_007u64 * 1_000_000_009,
            18_446_744_073_709_551_557,
        ] {
            let certificate = engine.certified_factorization(value).unwrap();
            assert_eq!(certificate.factorization.value, value);
            assert_eq!(certificate.verify(), Ok(()), "value={value}");
        }
    }

    #[test]
    fn test_certified_factorization_canonical_bytes_golden_vector_is_stable() {
        let certificate = CertifiedFactorizationCertificate {
            factorization: FactorizationCertificate {
                value: 2,
                factors: vec![(2, 1)],
            },
            prime_proofs: vec![PrimalityProof {
                value: 2,
                certificates: vec![PrimalityCertificate {
                    prime: 2,
                    predecessor_factors: vec![],
                    witnesses: vec![],
                }],
            }],
        };
        let golden: [u8; 76] = [
            83, 89, 77, 70, 67, 69, 82, 49, // SYMFCER1
            0, 0, 0, 0, 0, 0, 0, 2, // value = 2
            0, 0, 0, 1, // factor count
            0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, // (2, 1)
            0, 0, 0, 1, // proof count
            0, 0, 0, 36, // nested proof length
            83, 89, 77, 80, 80, 82, 70, 49, // SYMPPRF1
            0, 0, 0, 0, 0, 0, 0, 2, // proof value = 2
            0, 0, 0, 1, // certificate count
            0, 0, 0, 0, 0, 0, 0, 2, // candidate = 2
            0, 0, 0, 0, // no predecessor factors
            0, 0, 0, 0, // no witnesses
        ];
        assert_eq!(certificate.to_canonical_bytes().unwrap(), golden);
        assert_eq!(
            CertifiedFactorizationCertificate::from_canonical_bytes(&golden).unwrap(),
            certificate
        );
    }

    #[test]
    fn test_certified_factorization_canonical_bytes_round_trip() {
        let engine = NumberTheoryEngine::new();
        for value in [
            1u64,
            2,
            30,
            1_000_000_007u64 * 1_000_000_009,
            18_446_744_073_709_551_557,
        ] {
            let certificate = engine.certified_factorization(value).unwrap();
            let bytes = certificate.to_canonical_bytes().unwrap();
            assert_eq!(&bytes[..8], b"SYMFCER1");
            let decoded = CertifiedFactorizationCertificate::from_canonical_bytes(&bytes).unwrap();
            assert_eq!(decoded, certificate);
            assert_eq!(decoded.to_canonical_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn test_certified_factorization_wire_rejects_malformed_and_tampered_bytes() {
        let engine = NumberTheoryEngine::new();
        let certificate = engine.certified_factorization(30).unwrap();
        let bytes = certificate.to_canonical_bytes().unwrap();

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 1;
        assert_eq!(
            CertifiedFactorizationCertificate::from_canonical_bytes(&bad_magic),
            Err(CertifiedFactorizationWireError::InvalidMagicOrVersion)
        );
        assert_eq!(
            CertifiedFactorizationCertificate::from_canonical_bytes(&bytes[..bytes.len() - 1]),
            Err(CertifiedFactorizationWireError::TruncatedInput)
        );

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            CertifiedFactorizationCertificate::from_canonical_bytes(&trailing),
            Err(CertifiedFactorizationWireError::TrailingBytes { count: 1 })
        );

        let mut too_many_factors = bytes.clone();
        too_many_factors[16..20]
            .copy_from_slice(&(CERTIFIED_FACTORIZATION_MAX_FACTORS + 1).to_be_bytes());
        assert!(matches!(
            CertifiedFactorizationCertificate::from_canonical_bytes(&too_many_factors),
            Err(CertifiedFactorizationWireError::TooManyFactors { .. })
        ));

        let mut missing_proof_count = bytes.clone();
        missing_proof_count[56..60].copy_from_slice(&2u32.to_be_bytes());
        assert!(matches!(
            CertifiedFactorizationCertificate::from_canonical_bytes(&missing_proof_count),
            Err(CertifiedFactorizationWireError::InvalidCertificate(
                CertifiedFactorizationError::PrimeProofCountMismatch { .. }
            ))
        ));

        let mut wrong_value = bytes.clone();
        wrong_value[15] ^= 1;
        assert!(matches!(
            CertifiedFactorizationCertificate::from_canonical_bytes(&wrong_value),
            Err(CertifiedFactorizationWireError::InvalidCertificate(
                CertifiedFactorizationError::InvalidFactorization(
                    FactorizationVerificationError::ProductMismatch { .. }
                )
            ))
        ));

        let last = bytes.len() - 1;
        let mut bad_nested_witness = bytes;
        bad_nested_witness[last] = 1;
        assert!(matches!(
            CertifiedFactorizationCertificate::from_canonical_bytes(&bad_nested_witness),
            Err(CertifiedFactorizationWireError::InvalidNestedProof { .. })
        ));
    }

    #[test]
    fn test_certified_factorization_rejects_missing_or_tampered_prime_proof() {
        let engine = NumberTheoryEngine::new();
        let mut certificate = engine.certified_factorization(30).unwrap();
        certificate.prime_proofs.pop();
        assert!(matches!(
            certificate.verify(),
            Err(CertifiedFactorizationError::PrimeProofCountMismatch { .. })
        ));

        let mut certificate = engine.certified_factorization(30).unwrap();
        certificate.prime_proofs[0].value = 19;
        assert!(matches!(
            certificate.verify(),
            Err(CertifiedFactorizationError::PrimeProofRootMismatch { .. })
        ));

        let mut certificate = engine.certified_factorization(30).unwrap();
        certificate.factorization.factors[0].1 = 0;
        assert!(matches!(
            certificate.verify(),
            Err(CertifiedFactorizationError::InvalidFactorization(
                FactorizationVerificationError::ZeroExponent { .. }
            ))
        ));
    }

    #[test]
    fn test_primality_proof_canonical_bytes_round_trip() {
        let engine = NumberTheoryEngine::new();

        // Shared hand-authored golden vector also consumed by the standalone Python
        // verifier. This prevents matching encoder/decoder bugs from redefining v1.
        let proof_three = engine.primality_proof(3).unwrap();
        let golden: &[u8] = &[
            b'S', b'Y', b'M', b'P', b'P', b'R', b'F', b'1',
            0, 0, 0, 0, 0, 0, 0, 3, // root = 3
            0, 0, 0, 2, // two certificate nodes
            0, 0, 0, 0, 0, 0, 0, 2, // base node: prime = 2
            0, 0, 0, 0, // no predecessor factors
            0, 0, 0, 0, // no witnesses
            0, 0, 0, 0, 0, 0, 0, 3, // root node: prime = 3
            0, 0, 0, 1, // one predecessor factor
            0, 0, 0, 0, 0, 0, 0, 2, // factor = 2
            0, 0, 0, 1, // exponent = 1
            0, 0, 0, 1, // one witness
            0, 0, 0, 0, 0, 0, 0, 2, // witness factor = 2
            0, 0, 0, 0, 0, 0, 0, 2, // witness base = 2
        ];
        assert_eq!(proof_three.to_canonical_bytes().unwrap().as_slice(), golden);
        assert_eq!(
            PrimalityProof::from_canonical_bytes(golden).unwrap(),
            proof_three
        );

        for value in [
            2u64,
            17,
            97,
            1_000_003,
            1_000_000_007,
            18_446_744_073_709_551_557,
        ] {
            let proof = engine.primality_proof(value).unwrap();
            let bytes = proof.to_canonical_bytes().unwrap();
            assert_eq!(&bytes[..8], b"SYMPPRF1");
            let decoded = PrimalityProof::from_canonical_bytes(&bytes).unwrap();
            assert_eq!(decoded, proof);
            assert_eq!(decoded.to_canonical_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn test_primality_proof_decoder_rejects_malformed_wire_inputs() {
        let engine = NumberTheoryEngine::new();
        let proof = engine.primality_proof(17).unwrap();
        let bytes = proof.to_canonical_bytes().unwrap();

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 1;
        assert_eq!(
            PrimalityProof::from_canonical_bytes(&bad_magic),
            Err(PrimalityProofWireError::InvalidMagicOrVersion)
        );
        assert_eq!(
            PrimalityProof::from_canonical_bytes(&bytes[..bytes.len() - 1]),
            Err(PrimalityProofWireError::TruncatedInput)
        );

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            PrimalityProof::from_canonical_bytes(&trailing),
            Err(PrimalityProofWireError::TrailingBytes { count: 1 })
        );

        assert!(matches!(
            PrimalityProof::from_canonical_bytes(&vec![0u8; PRIMALITY_PROOF_MAX_WIRE_BYTES + 1]),
            Err(PrimalityProofWireError::InputTooLarge { .. })
        ));

        let mut too_many_certificates = bytes.clone();
        too_many_certificates[16..20]
            .copy_from_slice(&(PRIMALITY_PROOF_MAX_CERTIFICATES + 1).to_be_bytes());
        assert!(matches!(
            PrimalityProof::from_canonical_bytes(&too_many_certificates),
            Err(PrimalityProofWireError::TooManyCertificates { .. })
        ));

        let mut too_many_entries = bytes;
        too_many_entries[28..32]
            .copy_from_slice(&(PRIMALITY_PROOF_MAX_ENTRIES as u32 + 1).to_be_bytes());
        assert!(matches!(
            PrimalityProof::from_canonical_bytes(&too_many_entries),
            Err(PrimalityProofWireError::TooManyEntries { .. })
        ));
    }

    #[test]
    fn test_primality_proof_wire_rejects_tampered_witness() {
        let engine = NumberTheoryEngine::new();
        let proof = engine.primality_proof(17).unwrap();
        let mut bytes = proof.to_canonical_bytes().unwrap();
        let last = bytes.len() - 1;
        bytes[last] = 1; // base 1 is invalid; reject rather than normalize
        assert!(matches!(
            PrimalityProof::from_canonical_bytes(&bytes),
            Err(PrimalityProofWireError::InvalidProof(
                PrimalityProofError::InvalidWitnessBase { .. }
            ))
        ));

        let mut malformed = proof;
        malformed.certificates.reverse();
        assert!(matches!(
            malformed.to_canonical_bytes(),
            Err(PrimalityProofWireError::InvalidProof(
                PrimalityProofError::CertificateTableNotStrictlyIncreasing { .. }
            ))
        ));
    }

    #[test]
    fn test_pocklington_primality_proofs_verify_small_and_large_primes() {
        let engine = NumberTheoryEngine::new();
        for prime in [
            2u64,
            3,
            5,
            17,
            97,
            1_000_003,
            1_000_000_007,
            18_446_744_073_709_551_557,
        ] {
            let proof = engine.primality_proof(prime).expect("prime should have a proof");
            assert_eq!(proof.value, prime);
            assert_eq!(proof.verify(), Ok(()), "prime={prime}");
        }
    }

    #[test]
    fn test_primality_proof_verifier_enforces_in_memory_resource_budgets() {
        let engine = NumberTheoryEngine::new();
        let valid = engine.primality_proof(3).unwrap();

        let mut too_many_nodes = valid.clone();
        too_many_nodes.certificates = vec![
            PrimalityCertificate {
                prime: 2,
                predecessor_factors: vec![],
                witnesses: vec![],
            };
            PRIMALITY_PROOF_MAX_CERTIFICATES as usize + 1
        ];
        assert!(matches!(
            too_many_nodes.verify(),
            Err(PrimalityProofError::TooManyCertificates { count })
                if count == PRIMALITY_PROOF_MAX_CERTIFICATES as usize + 1
        ));

        let too_many_entries = PrimalityProof {
            value: 3,
            certificates: vec![
                PrimalityCertificate {
                    prime: 2,
                    predecessor_factors: vec![],
                    witnesses: vec![],
                },
                PrimalityCertificate {
                    prime: 3,
                    predecessor_factors: vec![
                        (2, 1);
                        PRIMALITY_PROOF_MAX_ENTRIES as usize + 1
                    ],
                    witnesses: vec![],
                },
            ],
        };
        assert!(matches!(
            too_many_entries.verify(),
            Err(PrimalityProofError::TooManyEntries { count })
                if count == PRIMALITY_PROOF_MAX_ENTRIES as usize + 1
        ));
    }

    #[test]
    fn test_pocklington_primality_proof_rejects_composites_and_invalid_roots() {
        let engine = NumberTheoryEngine::new();
        for value in [0u64, 1, 4, 15, 341_550_071_728_321, u64::MAX] {
            assert!(
                engine.primality_proof(value).is_err(),
                "value={value} must not receive a prime proof"
            );
        }
    }

    #[test]
    fn test_pocklington_proof_rejects_carmichael_when_fermat_passes() {
        let engine = NumberTheoryEngine::new();
        let mut nodes = std::collections::BTreeMap::new();
        for prime in [2u64, 3, 5, 7] {
            let proof = engine.primality_proof(prime).expect("small prime proof");
            for certificate in proof.certificates {
                nodes.insert(certificate.prime, certificate);
            }
        }

        // 561 is composite but Carmichael. A Fermat-only checker accepts base 2:
        // 2^560 = 1 (mod 561). The Pocklington GCD condition for q=2 must reject it.
        assert_eq!(certificate_pow_mod(2, 560, 561), 1);
        assert_eq!(certificate_pow_mod(2, 280, 561), 1);
        assert_eq!(certificate_gcd(certificate_pow_mod(2, 280, 561).abs_diff(1), 561), 561);

        let mut certificates: Vec<PrimalityCertificate> = nodes.into_values().collect();
        certificates.push(PrimalityCertificate {
            prime: 561,
            predecessor_factors: vec![(2, 4), (5, 1), (7, 1)],
            witnesses: vec![(2, 2), (5, 2), (7, 2)],
        });
        let claim = PrimalityProof { value: 561, certificates };
        assert!(matches!(
            claim.verify(),
            Err(PrimalityProofError::InvalidPocklingtonWitness {
                prime: 561,
                factor: 2,
                base: 2
            })
        ));
    }

    #[test]
    fn test_pocklington_proof_rejects_unreachable_certificate_nodes() {
        let engine = NumberTheoryEngine::new();
        let mut proof = engine.primality_proof(17).unwrap();
        let unrelated = engine.primality_proof(3).unwrap();
        proof.certificates.insert(1, unrelated.certificates[1].clone());
        assert!(matches!(
            proof.verify(),
            Err(PrimalityProofError::UnreachableCertificate { prime: 3 })
        ));
    }

    #[test]
    fn test_pocklington_primality_proof_rejects_tampered_witness_and_table() {
        let engine = NumberTheoryEngine::new();
        let mut proof = engine.primality_proof(17).unwrap();
        let root = proof.certificates.iter_mut().find(|c| c.prime == 17).unwrap();
        root.witnesses[0].1 = 1;
        assert!(matches!(
            proof.verify(),
            Err(PrimalityProofError::InvalidWitnessBase { prime: 17, .. })
        ));

        let mut proof = engine.primality_proof(17).unwrap();
        proof.certificates.reverse();
        assert!(matches!(
            proof.verify(),
            Err(PrimalityProofError::CertificateTableNotStrictlyIncreasing { .. })
        ));
    }

    #[test]
    fn test_factorization_certificate_verifiers_enforce_in_memory_resource_budgets() {
        let too_many_factors = FactorizationCertificate {
            value: 1,
            factors: (2u64..(2 + CERTIFIED_FACTORIZATION_MAX_FACTORS as u64 + 1))
                .map(|prime| (prime, 1))
                .collect(),
        };
        assert!(matches!(
            too_many_factors.verify(),
            Err(FactorizationVerificationError::TooManyFactors { count })
                if count == CERTIFIED_FACTORIZATION_MAX_FACTORS as usize + 1
        ));

        let engine = NumberTheoryEngine::new();
        let prime_two = engine.primality_proof(2).unwrap();
        let too_many_proofs = CertifiedFactorizationCertificate {
            factorization: FactorizationCertificate { value: 1, factors: vec![] },
            prime_proofs: vec![
                prime_two;
                CERTIFIED_FACTORIZATION_MAX_PROOFS as usize + 1
            ],
        };
        assert!(matches!(
            too_many_proofs.verify(),
            Err(CertifiedFactorizationError::TooManyPrimeProofs { count })
                if count == CERTIFIED_FACTORIZATION_MAX_PROOFS as usize + 1
        ));
    }

    #[test]
    fn test_factorization_certificates_verify_search_results() {
        let engine = NumberTheoryEngine::new();
        for value in [
            1u64,
            2,
            30,
            3_215_031_751,
            1_000_000_007u64 * 1_000_000_009,
            4_294_967_291u64 * 4_294_967_279,
            18_446_744_073_709_551_557,
        ] {
            let factorization = engine.factorize(value);
            let certificate = factorization.certificate();
            assert_eq!(certificate.value, value);
            assert_eq!(certificate.verify(), Ok(()), "value={value}");
        }
    }

    #[test]
    fn test_factorization_certificate_rejects_tampering_and_noncanonical_forms() {
        let cases = [
            (
                FactorizationCertificate { value: 15, factors: vec![(15, 1)] },
                FactorizationVerificationError::CompositeFactor { factor: 15 },
            ),
            (
                FactorizationCertificate { value: 6, factors: vec![(3, 1), (2, 1)] },
                FactorizationVerificationError::FactorsNotStrictlyIncreasing { previous: 3, current: 2 },
            ),
            (
                FactorizationCertificate { value: 2, factors: vec![(2, 0)] },
                FactorizationVerificationError::ZeroExponent { factor: 2 },
            ),
            (
                FactorizationCertificate { value: 11, factors: vec![(2, 1)] },
                FactorizationVerificationError::ProductMismatch { expected: 11, actual: 2 },
            ),
            (
                FactorizationCertificate { value: u64::MAX, factors: vec![(2, 64)] },
                FactorizationVerificationError::ProductOverflow { factor: 2 },
            ),
            (
                FactorizationCertificate { value: 0, factors: vec![] },
                FactorizationVerificationError::ZeroInput,
            ),
        ];

        for (certificate, expected_error) in cases {
            assert_eq!(certificate.verify(), Err(expected_error), "certificate={certificate:?}");
        }
    }

    #[test]
    fn test_certificate_primality_verifier_matches_search_engine_through_10000() {
        let engine = NumberTheoryEngine::new();
        for n in 0u64..=10_000 {
            assert_eq!(
                certificate_is_prime_u64(n),
                engine.is_prime(n),
                "n={n}"
            );
        }
        for n in [
            18_446_744_073_709_551_557u64,
            3_215_031_751,
            u64::MAX,
            341_550_071_728_321,
        ] {
            assert_eq!(certificate_is_prime_u64(n), engine.is_prime(n), "n={n}");
        }
    }

    #[test]
    fn test_factorization_phi_uses_exact_integer_totient_before_float_conversion() {
        let engine = NumberTheoryEngine::new();
        // Repeated floating products historically drifted for these exact small values:
        // phi(30) could be 8.000000000000002 rather than 8.0.
        for (n, expected_phi) in [
            (30u64, 8u64),
            (210, 48),
            (2_310, 480),
            (30_030, 5_760),
            (510_510, 92_160),
            (9_699_690, 1_658_880),
        ] {
            assert_eq!(engine.factorize(n).phi, expected_phi as f64, "n={n}");
        }
    }

    #[test]
    fn test_factorize_large_prime_powers_and_mixed_composite() {
        let engine = NumberTheoryEngine::new();
        for (n, expected) in [
            (2u64.pow(40), vec![(2u64, 40u32)]),
            (3u64.pow(20), vec![(3u64, 20u32)]),
            (
                3_215_031_751u64,
                vec![(151u64, 1u32), (751u64, 1u32), (28_351u64, 1u32)],
            ),
        ] {
            let result = engine.factorize(n);
            assert_eq!(result.factors, expected, "n={n}");
        }
    }

    #[test]
    fn test_extended_gcd() {
        let engine = NumberTheoryEngine::new();
        let (g, x, y) = engine.extended_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }

    #[test]
    fn test_extended_gcd_wide_satisfies_bezout_identity_on_signed_grid() {
        let engine = NumberTheoryEngine::new();
        for a in -40i64..=40 {
            for b in -40i64..=40 {
                let (gcd, x, y) = engine.extended_gcd_wide(a, b);
                assert_eq!(i128::from(a) * x + i128::from(b) * y, i128::from(gcd), "a={a}, b={b}");
                assert_eq!(gcd, engine.gcd(a.unsigned_abs(), b.unsigned_abs()));
            }
        }
    }

    #[test]
    fn test_extended_gcd_wide_handles_signed_integer_boundaries() {
        let engine = NumberTheoryEngine::new();
        for (a, b) in [
            (i64::MIN, 0),
            (0, i64::MIN),
            (i64::MIN, i64::MAX),
            (i64::MAX, i64::MIN),
            (i64::MIN, i64::MIN),
            (i64::MAX, i64::MAX),
            (0, 0),
        ] {
            let (gcd, x, y) = engine.extended_gcd_wide(a, b);
            assert_eq!(i128::from(a) * x + i128::from(b) * y, i128::from(gcd), "a={a}, b={b}");
            assert_eq!(gcd, engine.gcd(a.unsigned_abs(), b.unsigned_abs()));
        }
        assert_eq!(engine.extended_gcd_wide(i64::MIN, 0), (1u64 << 63, -1, 0));
    }

    #[test]
    #[should_panic(expected = "GCD exceeds i64::MAX; use extended_gcd_wide")]
    fn test_legacy_extended_gcd_rejects_unrepresentable_gcd_explicitly() {
        let engine = NumberTheoryEngine::new();
        let _ = engine.extended_gcd(i64::MIN, 0);
    }

    #[test]
    fn test_modular_inverse() {
        let ring = ModularRing::new(7);
        let inv = ring.inverse(3).unwrap();
        assert_eq!(inv, 5); // 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!(ring.multiply(3, inv), 1);
    }

    #[test]
    fn test_no_inverse() {
        let ring = ModularRing::new(6);
        assert!(ring.inverse(2).is_none()); // gcd(2, 6) = 2 != 1
    }

    #[test]
    fn test_modular_ring_rejects_zero_modulus_explicitly() {
        assert!(ModularRing::try_new(0).is_none());
        let ring = ModularRing::try_new(1).unwrap();
        assert_eq!(ring.add(u64::MAX, u64::MAX), 0);
        assert_eq!(ring.multiply(u64::MAX, u64::MAX), 0);
        assert_eq!(ring.power(u64::MAX, 0), 0);
    }

    #[test]
    #[should_panic(expected = "modular arithmetic requires a non-zero modulus")]
    fn test_modular_ring_legacy_constructor_fails_on_zero_modulus() {
        let _ = ModularRing::new(0);
    }

    #[test]
    #[should_panic(expected = "modular arithmetic requires a non-zero modulus")]
    fn test_modular_ring_operations_revalidate_public_mutable_modulus() {
        let mut ring = ModularRing::new(7);
        ring.modulus = 0;
        let _ = ring.add(1, 2);
    }

    #[test]
    fn test_modular_addition_does_not_overflow_near_u64_max() {
        let ring = ModularRing::new(u64::MAX);
        assert_eq!(ring.add(u64::MAX - 1, u64::MAX - 1), u64::MAX - 2);
        assert_eq!(ring.add(u64::MAX, 1), 1);
        assert_eq!(ring.add(u64::MAX - 1, 2), 1);
    }

    #[test]
    fn test_modular_inverse_supports_moduli_above_i64_max() {
        let modulus = u64::MAX - 1;
        let ring = ModularRing::new(modulus);
        let inverse = ring.inverse(3).expect("gcd(3, modulus) = 1");
        assert_eq!(ring.multiply(3, inverse), 1);

        let large_modulus = (1u64 << 63) + 29;
        let prime_ring = ModularRing::new(large_modulus);
        let inv = prime_ring.inverse(prime_ring.modulus - 1).unwrap();
        assert_eq!(inv, prime_ring.modulus - 1);
        assert_eq!(prime_ring.multiply(prime_ring.modulus - 1, inv), 1);
    }

    #[test]
    fn test_modular_power() {
        let ring = ModularRing::new(13);
        assert_eq!(ring.power(2, 10), 10); // 2^10 = 1024, 1024 mod 13 = 10
    }

    #[test]
    fn test_euler_totient() {
        let ring = ModularRing::new(12);
        assert_eq!(ring.euler_totient(), 4); // {1, 5, 7, 11}
        assert_eq!(ModularRing::new(1).euler_totient(), 1); // phi(1) = 1 by convention
    }

    #[test]
    fn test_euler_totient_matches_coprime_count_for_small_moduli() {
        let engine = NumberTheoryEngine::new();
        for modulus in 1u64..=128 {
            let expected = (1..=modulus)
                .filter(|&a| engine.gcd(a, modulus) == 1)
                .count() as u64;
            assert_eq!(
                ModularRing::new(modulus).euler_totient(),
                expected,
                "modulus={modulus}"
            );
        }
    }

    #[test]
    fn test_euler_totient_handles_large_u64_prime_without_residue_scan() {
        let prime = 18_446_744_073_709_551_557u64;
        assert_eq!(ModularRing::new(prime).euler_totient(), prime - 1);
    }

    #[test]
    fn test_euler_totient_handles_near_u64_semiprime() {
        let p = 4_294_967_291u64;
        let q = 4_294_967_279u64;
        let modulus = p * q;
        assert_eq!(
            ModularRing::new(modulus).euler_totient(),
            (p - 1) * (q - 1)
        );
    }

    #[test]
    fn test_primes_up_to() {
        let engine = NumberTheoryEngine::new();
        let primes = engine.primes_up_to(20);
        let prime_values: Vec<u64> = primes.iter().map(|(p, _)| *p).collect();
        assert_eq!(prime_values, vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_is_prime() {
        let engine = NumberTheoryEngine::new();
        assert!(engine.is_prime(17));
        assert!(!engine.is_prime(15));
        assert!(!engine.is_prime(1));
        assert!(engine.is_prime(2));
    }

    #[test]
    fn test_is_prime_is_deterministic_near_u64_limits() {
        let engine = NumberTheoryEngine::new();
        // Largest known prime below 2^64.
        assert!(engine.is_prime(18_446_744_073_709_551_557));
        assert!(!engine.is_prime(u64::MAX));
        assert!(!engine.is_prime(3_215_031_751)); // strong pseudoprime to small bases
        assert!(!engine.is_prime(561)); // Carmichael number
    }

    #[test]
    fn test_factorize_large_u64_prime_uses_fast_primality_path() {
        let engine = NumberTheoryEngine::new();
        let prime = 18_446_744_073_709_551_557;
        let factorization = engine.factorize(prime);
        assert_eq!(factorization.value, prime);
        assert_eq!(factorization.factors, vec![(prime, 1)]);
    }

    #[test]
    fn test_fermat_test() {
        let engine = NumberTheoryEngine::new();
        assert!(engine.fermat_test(17, &[2, 3, 5]));
        assert!(!engine.fermat_test(15, &[2]));
    }

    #[test]
    fn test_lcm() {
        let engine = NumberTheoryEngine::new();
        assert_eq!(engine.lcm(12, 18), 36);
        assert_eq!(engine.checked_lcm(12, 18), Some(36));
        assert_eq!(engine.checked_lcm(0, u64::MAX), Some(0));
        assert_eq!(engine.checked_lcm(u64::MAX, 1), Some(u64::MAX));
        assert_eq!(engine.checked_lcm(u64::MAX, 2), None);
    }

    #[test]
    #[should_panic(expected = "least common multiple exceeds u64::MAX")]
    fn test_lcm_fails_explicitly_instead_of_wrapping_on_overflow() {
        let engine = NumberTheoryEngine::new();
        let _ = engine.lcm(u64::MAX, 2);
    }

    // ── Miller-Rabin tests ──────────────────────────────────────────────

    #[test]
    fn test_is_prime_matches_independent_trial_division_through_10000() {
        let engine = NumberTheoryEngine::new();
        for n in 0u64..=10_000 {
            let expected = if n < 2 {
                false
            } else {
                let mut divisor = 2u64;
                let mut prime = true;
                while divisor <= n / divisor {
                    if n.is_multiple_of(divisor) {
                        prime = false;
                        break;
                    }
                    divisor += if divisor == 2 { 1 } else { 2 };
                }
                prime
            };
            assert_eq!(engine.is_prime(n), expected, "n={n}");
        }
    }

    #[test]
    fn test_factorization_reconstructs_small_integers_and_uses_prime_factors() {
        let engine = NumberTheoryEngine::new();
        for n in 2u64..=2_000 {
            let result = engine.factorize(n);
            let mut product = 1u128;
            for &(prime, exponent) in &result.factors {
                assert!(engine.miller_rabin(prime), "factor {prime} of {n} must be prime");
                assert!(exponent > 0);
                for _ in 0..exponent {
                    product *= u128::from(prime);
                }
            }
            assert_eq!(product, u128::from(n), "factorization of {n}: {:?}", result.factors);
        }
    }

    #[test]
    fn test_miller_rabin_small_primes() {
        let engine = NumberTheoryEngine::new();
        let primes = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ];
        for &p in &primes {
            assert!(engine.miller_rabin(p), "{} should be prime", p);
        }
    }

    #[test]
    fn test_miller_rabin_composites() {
        let engine = NumberTheoryEngine::new();
        let composites = [4, 6, 8, 9, 10, 12, 15, 21, 25, 49, 100, 121, 1000];
        for &c in &composites {
            assert!(!engine.miller_rabin(c), "{} should be composite", c);
        }
    }

    /// Carmichael numbers fool the Fermat test but NOT Miller-Rabin.
    /// This is the critical advantage of Miller-Rabin.
    #[test]
    fn test_miller_rabin_rejects_carmichael_numbers() {
        let engine = NumberTheoryEngine::new();
        // First 10 Carmichael numbers
        let carmichael = [561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341];
        for &c in &carmichael {
            assert!(
                !engine.miller_rabin(c),
                "Carmichael number {} should be rejected by Miller-Rabin",
                c
            );
        }

        // Verify that Fermat INCORRECTLY accepts some Carmichael numbers
        // (561 = 3×11×17 passes Fermat for witness 2)
        let fermat_561 = engine.fermat_test(561, &[2]);
        // Note: 561 actually DOES pass Fermat with witness 2
        // because 2^560 mod 561 = 1 (this is why Carmichael numbers are dangerous)
        eprintln!(
            "Fermat(561, [2]) = {} (Carmichael — Fermat may be fooled)",
            fermat_561
        );
        eprintln!(
            "Miller-Rabin(561) = {} (correctly rejects)",
            engine.miller_rabin(561)
        );
    }

    #[test]
    fn test_miller_rabin_large_primes() {
        let engine = NumberTheoryEngine::new();
        // Large known primes
        assert!(engine.miller_rabin(104729)); // 10000th prime
        assert!(engine.miller_rabin(1_000_003)); // prime just above 1M
        assert!(engine.miller_rabin(15_485_863)); // 1 millionth prime
        assert!(!engine.miller_rabin(15_485_864)); // not prime
    }

    #[test]
    fn test_miller_rabin_edge_cases() {
        let engine = NumberTheoryEngine::new();
        assert!(!engine.miller_rabin(0));
        assert!(!engine.miller_rabin(1));
        assert!(engine.miller_rabin(2));
        assert!(engine.miller_rabin(3));
    }

    // ── Phase 1 primitives ──────────────────────────────────────────────

    #[test]
    fn test_crt_coprime() {
        let e = NumberTheoryEngine::new();
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → x = 23 mod 105
        let (x, m) = e.crt(&[(2, 3), (3, 5), (2, 7)]).unwrap();
        assert_eq!(m, 105);
        assert_eq!(x, 23);
    }

    #[test]
    fn test_crt_non_coprime_consistent() {
        let e = NumberTheoryEngine::new();
        // x ≡ 2 (mod 6), x ≡ 8 (mod 10) → agree mod gcd(6,10)=2; lcm=30, x=8
        let (x, m) = e.crt(&[(2, 6), (8, 10)]).unwrap();
        assert_eq!(m, 30);
        assert_eq!(x.rem_euclid(6), 2);
        assert_eq!(x.rem_euclid(10), 8);
    }

    #[test]
    fn test_crt_inconsistent() {
        let e = NumberTheoryEngine::new();
        // x ≡ 1 (mod 4), x ≡ 2 (mod 6): disagree mod gcd=2
        assert!(e.crt(&[(1, 4), (2, 6)]).is_none());
    }

    #[test]
    fn test_legendre_symbol() {
        let e = NumberTheoryEngine::new();
        // QRs mod 7: {1, 2, 4}; NQRs: {3, 5, 6}
        assert_eq!(e.legendre_symbol(1, 7), 1);
        assert_eq!(e.legendre_symbol(2, 7), 1);
        assert_eq!(e.legendre_symbol(3, 7), -1);
        assert_eq!(e.legendre_symbol(4, 7), 1);
        assert_eq!(e.legendre_symbol(5, 7), -1);
        assert_eq!(e.legendre_symbol(6, 7), -1);
        assert_eq!(e.legendre_symbol(7, 7), 0);
    }

    #[test]
    fn test_jacobi_symbol() {
        let e = NumberTheoryEngine::new();
        // Jacobi = Legendre when n is prime
        for a in 1..11 {
            assert_eq!(e.jacobi_symbol(a, 11), e.legendre_symbol(a, 11));
        }
        // (2/15) = (2/3)(2/5) = (-1)(-1) = 1
        assert_eq!(e.jacobi_symbol(2, 15), 1);
        // (7/15) = (7/3)(7/5) = (1)(-1) = -1
        assert_eq!(e.jacobi_symbol(7, 15), -1);
    }

    #[test]
    fn test_tonelli_shanks_p3_mod_4() {
        let e = NumberTheoryEngine::new();
        // sqrt(2) mod 7: 7 ≡ 3 (mod 4). 2^((7+1)/4) = 2^2 = 4. 4² = 16 ≡ 2. ✓
        let r = e.tonelli_shanks(2, 7).unwrap();
        assert_eq!((r * r) % 7, 2);
    }

    #[test]
    fn test_tonelli_shanks_p1_mod_4() {
        let e = NumberTheoryEngine::new();
        // sqrt(10) mod 13: 13 ≡ 1 (mod 4), full algorithm path
        let r = e.tonelli_shanks(10, 13).unwrap();
        assert_eq!((r * r) % 13, 10);
    }

    #[test]
    fn test_tonelli_shanks_larger() {
        let e = NumberTheoryEngine::new();
        // sqrt(5) mod 41 exists since (5/41)=1
        let r = e.tonelli_shanks(5, 41).unwrap();
        assert_eq!((r * r) % 41, 5);
        // sqrt(3) mod 41: (3/41)=?  check both exist or not
        if e.legendre_symbol(3, 41) == 1 {
            let r = e.tonelli_shanks(3, 41).unwrap();
            assert_eq!((r * r) % 41, 3);
        } else {
            assert!(e.tonelli_shanks(3, 41).is_none());
        }
    }

    #[test]
    fn test_tonelli_shanks_non_residue() {
        let e = NumberTheoryEngine::new();
        // 3 is non-residue mod 7: (3/7) = -1
        assert!(e.tonelli_shanks(3, 7).is_none());
    }

    #[test]
    fn test_p_adic_valuation() {
        let e = NumberTheoryEngine::new();
        assert_eq!(e.p_adic_valuation(12, 2), 2); // 12 = 4·3
        assert_eq!(e.p_adic_valuation(12, 3), 1);
        assert_eq!(e.p_adic_valuation(12, 5), 0);
        assert_eq!(e.p_adic_valuation(-48, 2), 4);
        assert_eq!(e.p_adic_valuation(81, 3), 4);
    }

    #[test]
    fn test_lte_odd_prime() {
        let e = NumberTheoryEngine::new();
        // v_3(5^6 − 2^6) = v_3(5−2) + v_3(6) = 1 + 1 = 2
        // 5^6 − 2^6 = 15625 − 64 = 15561 = 3² · 7 · 13 · 19
        assert_eq!(e.lifting_the_exponent(3, 5, 2, 6), Some(2));
        // v_5(7^10 − 2^10) = v_5(5) + v_5(10) = 1 + 1 = 2
        assert_eq!(e.lifting_the_exponent(5, 7, 2, 10), Some(2));
    }

    #[test]
    fn test_lte_precondition_violation() {
        let e = NumberTheoryEngine::new();
        // p | a: fails precondition
        assert!(e.lifting_the_exponent(3, 6, 2, 4).is_none());
        // p ∤ (a−b): fails
        assert!(e.lifting_the_exponent(5, 7, 3, 4).is_none());
    }

    #[test]
    fn test_lte_p2_n_even_4_divides() {
        let e = NumberTheoryEngine::new();
        // a=5, b=1, a−b=4 (4 divides), n=2
        // v_2(5^2 − 1^2) = v_2(24) = 3
        // Formula: v_2(a−b) + v_2(n) = 2 + 1 = 3 ✓
        assert_eq!(e.lifting_the_exponent(2, 5, 1, 2), Some(3));
    }

    #[test]
    fn test_lte_p2_n_even_only_2_divides() {
        let e = NumberTheoryEngine::new();
        // a=3, b=1, a−b=2, a+b=4, n=2
        // v_2(3^2 − 1^2) = v_2(8) = 3
        // Formula: v_2(a−b) + v_2(a+b) + v_2(n) − 1 = 1 + 2 + 1 − 1 = 3 ✓
        assert_eq!(e.lifting_the_exponent(2, 3, 1, 2), Some(3));
    }

    #[test]
    fn test_linear_diophantine_solvable() {
        let e = NumberTheoryEngine::new();
        // 12x + 8y = 20 → gcd=4 | 20. Particular + increments.
        let (x0, y0, dx, dy) = e.linear_diophantine(12, 8, 20).unwrap();
        assert_eq!(12 * x0 + 8 * y0, 20);
        // k=1 shift
        assert_eq!(12 * (x0 + dx) + 8 * (y0 - dy), 20);
        // dx = b/g = 2, dy = a/g = 3
        assert_eq!(dx, 2);
        assert_eq!(dy, 3);
    }

    #[test]
    fn test_linear_diophantine_unsolvable() {
        let e = NumberTheoryEngine::new();
        // 6x + 9y = 5: gcd=3 ∤ 5
        assert!(e.linear_diophantine(6, 9, 5).is_none());
    }

    #[test]
    fn test_linear_diophantine_trivial() {
        let e = NumberTheoryEngine::new();
        assert!(e.linear_diophantine(0, 0, 5).is_none());
        assert!(e.linear_diophantine(0, 0, 0).is_some());
    }
}
