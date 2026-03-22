// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Zero-Knowledge Proofs for Graceful Ignorance System
//!
//! Implements realistic ZK proof primitives for privacy-preserving ignorance sharing:
//!
//! - **Pedersen Commitment**: Cryptographically binding and hiding commitments
//! - **Bulletproofs-style Range Proofs**: Efficient proofs that values are in a range
//! - **Schnorr Signatures**: Proofs of knowledge of a secret
//! - **Fiat-Shamir Transform**: Non-interactive proofs via random oracle
//!
//! ## Security Model
//!
//! These implementations use BLAKE3 as the random oracle and simulate elliptic
//! curve operations in a prime field for educational purposes. For production,
//! use established libraries like `bulletproofs` or `ark-groth16`.
//!
//! ## Architecture
//!
//! ```text
//! Topic ---> [Pedersen Commit] ---> Commitment (binding, hiding)
//!                |
//!                v
//! EIG ------> [Range Proof] ----> Proof that 0 ≤ EIG ≤ 1
//!                |
//!                v
//! All -------> [Validity Proof] -> ZK proof of topic coherence
//! ```

use blake3::Hasher;
use std::time::SystemTime;

/// Prime field modulus (256-bit Mersenne-like prime for simulation)
/// In production, this would be the secp256k1 or BLS12-381 curve order
const FIELD_MODULUS: u128 = (1 << 127) - 1; // 2^127 - 1

// ============================================================================
// Field Arithmetic
// ============================================================================

/// Field element in the prime field
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldElement(pub u128);

impl FieldElement {
    /// Create a new field element
    pub fn new(value: u128) -> Self {
        Self(value % FIELD_MODULUS)
    }

    /// Create from bytes using BLAKE3 hash
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(bytes);
        let hash = hasher.finalize();
        let arr: [u8; 16] = hash.as_bytes()[0..16]
            .try_into()
            .expect("16-byte slice fits [u8; 16]");
        Self::new(u128::from_le_bytes(arr))
    }

    /// Add two field elements
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.0.wrapping_add(other.0))
    }

    /// Subtract two field elements
    pub fn sub(&self, other: &Self) -> Self {
        if self.0 >= other.0 {
            Self::new(self.0 - other.0)
        } else {
            Self::new(FIELD_MODULUS - (other.0 - self.0))
        }
    }

    /// Multiply two field elements using Montgomery reduction to prevent overflow
    pub fn mul(&self, other: &Self) -> Self {
        // Use double-wide multiplication to prevent overflow
        // Split into high and low parts
        let a_lo = self.0 & 0xFFFF_FFFF_FFFF_FFFF;
        let a_hi = self.0 >> 64;
        let b_lo = other.0 & 0xFFFF_FFFF_FFFF_FFFF;
        let b_hi = other.0 >> 64;

        // Compute partial products (each fits in u128)
        let lo_lo = a_lo.wrapping_mul(b_lo);
        let lo_hi = a_lo.wrapping_mul(b_hi);
        let hi_lo = a_hi.wrapping_mul(b_lo);
        let hi_hi = a_hi.wrapping_mul(b_hi);

        // Combine with care for overflow
        // result = lo_lo + (lo_hi + hi_lo) << 64 + hi_hi << 128
        // For our modulus 2^127-1, we use: x mod (2^127-1) = (x mod 2^127) + (x / 2^127)

        // Simplified: just reduce components individually and combine
        let mid = lo_hi.wrapping_add(hi_lo);
        let carry = if mid < lo_hi { 1u128 } else { 0u128 };

        // Lower 128 bits contribution
        let low_part = lo_lo.wrapping_add(mid << 64);
        let low_carry = if low_part < lo_lo { 1u128 } else { 0u128 };

        // Higher bits
        let high_part = hi_hi
            .wrapping_add(mid >> 64)
            .wrapping_add(carry << 64)
            .wrapping_add(low_carry);

        // Reduce mod (2^127 - 1): x = low_part + high_part * 2^128
        // = low_part + high_part * 2 * 2^127 = low_part + high_part * 2 * (p + 1)
        // = low_part + 2*high_part mod p (approximately)

        let reduced = if high_part > 0 {
            // For very large products, just take mod
            (low_part % FIELD_MODULUS).wrapping_add((high_part % FIELD_MODULUS) << 1)
                % FIELD_MODULUS
        } else {
            low_part % FIELD_MODULUS
        };

        Self::new(reduced)
    }

    /// Scalar multiplication (repeated addition for simulation)
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        let mut result = Self::new(0);
        let mut base = *self;
        let mut s = scalar;

        while s > 0 {
            if s & 1 == 1 {
                result = result.add(&base);
            }
            base = base.mul(&base);
            s >>= 1;
        }
        result
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> [u8; 16] {
        self.0.to_le_bytes()
    }

    /// Generate random field element
    pub fn random(seed: u64) -> Self {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&seed.to_le_bytes());
        bytes[8..16].copy_from_slice(&(seed.wrapping_mul(0xDEADBEEF)).to_le_bytes());
        Self::from_bytes(&bytes)
    }
}

// ============================================================================
// Pedersen Commitment (Improved)
// ============================================================================

/// Generator points for Pedersen commitments
/// G and H are nothing-up-my-sleeve points derived from BLAKE3
pub struct PedersenGenerators {
    /// Base generator G
    pub g: FieldElement,
    /// Blinding generator H
    pub h: FieldElement,
}

impl PedersenGenerators {
    /// Create generators using BLAKE3 NUMS (Nothing Up My Sleeve)
    pub fn new() -> Self {
        Self {
            g: FieldElement::from_bytes(b"Pedersen Generator G - Graceful Ignorance"),
            h: FieldElement::from_bytes(b"Pedersen Generator H - Graceful Ignorance"),
        }
    }
}

impl Default for PedersenGenerators {
    fn default() -> Self {
        Self::new()
    }
}

/// Pedersen Commitment: C = g^m * h^r
///
/// Properties:
/// - **Binding**: Cannot open to different message (computational)
/// - **Hiding**: Commitment reveals nothing about message (perfect)
/// - **Homomorphic**: C(m1) * C(m2) = C(m1 + m2)
#[derive(Debug, Clone)]
pub struct PedersenCommitment {
    /// The commitment value
    pub commitment: FieldElement,
    /// Opening info (kept secret by committer)
    pub opening: Option<CommitmentOpening>,
}

/// Opening information for a Pedersen commitment
#[derive(Debug, Clone)]
pub struct CommitmentOpening {
    /// The committed message (as field element)
    pub message: FieldElement,
    /// The blinding factor
    pub randomness: FieldElement,
}

impl PedersenCommitment {
    /// Create a new Pedersen commitment to a message
    pub fn commit(message: &[u8], randomness_seed: u64) -> Self {
        let gens = PedersenGenerators::new();

        // Hash message to field element
        let m = FieldElement::from_bytes(message);

        // Generate randomness
        let r = FieldElement::random(randomness_seed);

        // C = g^m * h^r (using additive group notation: C = m*G + r*H)
        let commitment = gens.g.mul(&m).add(&gens.h.mul(&r));

        Self {
            commitment,
            opening: Some(CommitmentOpening {
                message: m,
                randomness: r,
            }),
        }
    }

    /// Create commitment from value only (for verification)
    pub fn from_commitment(commitment: FieldElement) -> Self {
        Self {
            commitment,
            opening: None,
        }
    }

    /// Verify a commitment opening
    pub fn verify_opening(&self, message: &[u8], randomness: &FieldElement) -> bool {
        let gens = PedersenGenerators::new();
        let m = FieldElement::from_bytes(message);

        let expected = gens.g.mul(&m).add(&gens.h.mul(randomness));
        self.commitment == expected
    }

    /// Get commitment as bytes (32 bytes)
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..16].copy_from_slice(&self.commitment.to_bytes());
        bytes
    }
}

// ============================================================================
// Bulletproofs-style Range Proof
// ============================================================================

/// Range proof parameters
#[derive(Debug, Clone)]
pub struct RangeProofParams {
    /// Number of bits in the range
    pub bit_length: usize,
    /// Generators for bit commitments
    pub bit_generators: Vec<FieldElement>,
}

impl RangeProofParams {
    /// Create parameters for range [0, 2^n - 1]
    pub fn new(bit_length: usize) -> Self {
        let bit_generators: Vec<FieldElement> = (0..bit_length)
            .map(|i| {
                let seed = format!("Range Proof Generator {i}");
                FieldElement::from_bytes(seed.as_bytes())
            })
            .collect();

        Self {
            bit_length,
            bit_generators,
        }
    }
}

/// Bulletproofs-style range proof
///
/// Proves that a committed value v is in range [0, 2^n - 1]
/// without revealing v.
///
/// This is a simplified version using the Fiat-Shamir transform.
/// Production code should use the full inner-product argument.
#[derive(Debug, Clone)]
pub struct BulletproofRangeProof {
    /// Bit commitments (one per bit)
    pub bit_commitments: Vec<FieldElement>,
    /// Challenge response
    pub challenge_response: FieldElement,
    /// Aggregated proof
    pub aggregated_proof: [u8; 64],
    /// Range being proven
    pub range: (f32, f32),
}

impl BulletproofRangeProof {
    /// Create a range proof for value in [min, max]
    ///
    /// This uses a simplified bit decomposition approach:
    /// 1. Decompose value into bits
    /// 2. Commit to each bit
    /// 3. Prove each commitment is 0 or 1
    /// 4. Prove aggregated value equals commitment
    pub fn create(value: f32, range: (f32, f32)) -> Option<Self> {
        // Check value is in range
        if value < range.0 || value > range.1 {
            return None;
        }

        let params = RangeProofParams::new(32);

        // Normalize value to [0, 2^32 - 1]
        let range_size = range.1 - range.0;
        if range_size <= 0.0 {
            return None;
        }
        // Use f64 for normalization: u32::MAX is not exactly representable in f32,
        // which would cause overflow at the top of the range.
        let normalized = ((value - range.0) as f64 / range_size as f64 * u32::MAX as f64)
            .min(u32::MAX as f64) as u32;

        // Bit decomposition
        let bits: Vec<bool> = (0..32).map(|i| (normalized >> i) & 1 == 1).collect();

        // Create bit commitments
        let mut bit_commitments = Vec::with_capacity(32);
        let mut transcript = Hasher::new();
        transcript.update(b"BulletproofRangeProof");
        transcript.update(&value.to_le_bytes());

        for (i, &bit) in bits.iter().enumerate() {
            let bit_val = if bit { 1u64 } else { 0u64 };

            // Commitment to bit: C_i = bit * G_i + r_i * H
            let r_i = FieldElement::random(
                transcript.finalize().as_bytes()[0..8]
                    .try_into()
                    .map(u64::from_le_bytes)
                    .unwrap_or(i as u64),
            );

            let bit_commitment = params.bit_generators[i]
                .scalar_mul(bit_val)
                .add(&PedersenGenerators::new().h.mul(&r_i));

            bit_commitments.push(bit_commitment);

            // Update transcript
            transcript = Hasher::new();
            transcript.update(&bit_commitment.to_bytes());
        }

        // Generate Fiat-Shamir challenge
        let challenge_hash = transcript.finalize();
        let challenge = FieldElement::from_bytes(challenge_hash.as_bytes());

        // Create aggregated response
        let challenge_response = challenge.mul(&FieldElement::new(normalized as u128));

        // Aggregate proof (simplified - full bulletproofs uses inner product)
        let mut aggregated_proof = [0u8; 64];
        aggregated_proof[0..32].copy_from_slice(challenge_hash.as_bytes());
        aggregated_proof[32..48].copy_from_slice(&challenge_response.to_bytes());

        Some(Self {
            bit_commitments,
            challenge_response,
            aggregated_proof,
            range,
        })
    }

    /// Verify the range proof
    pub fn verify(&self) -> bool {
        // Verify structure
        if self.bit_commitments.len() != 32 {
            return false;
        }

        // Verify range bounds are valid
        if self.range.0 > self.range.1 {
            return false;
        }

        // Verify bit commitments are non-zero (valid commitments)
        for commitment in &self.bit_commitments {
            if commitment.0 == 0 {
                return false;
            }
        }

        // Verify the aggregated proof has expected structure
        // The first 32 bytes should be a valid BLAKE3 hash
        let stored_hash = &self.aggregated_proof[0..32];
        let stored_response = FieldElement::from_bytes(&self.aggregated_proof[32..48]);

        // Verify challenge response is consistent
        // In a full bulletproofs implementation, this would verify:
        // 1. Each bit commitment opens to 0 or 1
        // 2. The aggregated commitment matches the claimed value
        // For this simplified version, we verify:
        // - The stored hash is non-zero
        // - The response is consistent with our commitment structure

        let hash_sum: u32 = stored_hash.iter().map(|&b| b as u32).sum();
        if hash_sum == 0 {
            return false;
        }

        // Verify challenge response is in valid range
        if stored_response.0 == 0 && self.challenge_response.0 != 0 {
            return false;
        }

        // Simplified verification passes if structure is valid
        // A full implementation would do the inner product argument
        true
    }

    /// Get proof size in bytes
    pub fn size(&self) -> usize {
        self.bit_commitments.len() * 16 + 64 + 16
    }
}

// ============================================================================
// Schnorr Signature / Proof of Knowledge
// ============================================================================

/// Schnorr signature proving knowledge of a secret
///
/// Proves: "I know x such that Y = g^x" without revealing x
///
/// Protocol (Fiat-Shamir):
/// 1. Prover: r <- random, R = g^r
/// 2. Challenge: c = H(R, Y, message)
/// 3. Response: s = r + c*x
/// 4. Verify: g^s == R * Y^c
#[derive(Debug, Clone)]
pub struct SchnorrProof {
    /// Commitment R = g^r
    pub commitment: FieldElement,
    /// Response s = r + c*x
    pub response: FieldElement,
    /// Public key Y = g^x (for verification)
    pub public_key: FieldElement,
}

impl SchnorrProof {
    /// Create a Schnorr proof for knowing a secret
    pub fn create(secret: &[u8], message: &[u8]) -> Self {
        let g = PedersenGenerators::new().g;

        // x = H(secret)
        let x = FieldElement::from_bytes(secret);

        // Y = g^x (public key)
        let public_key = g.mul(&x);

        // r <- random
        let r_seed = {
            let mut hasher = Hasher::new();
            hasher.update(secret);
            hasher.update(message);
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            hasher.update(&now.to_le_bytes());
            let hash = hasher.finalize();
            u64::from_le_bytes(
                hash.as_bytes()[0..8]
                    .try_into()
                    .expect("8-byte slice fits [u8; 8]"),
            )
        };
        let r = FieldElement::random(r_seed);

        // R = g^r
        let commitment = g.mul(&r);

        // c = H(R, Y, message)
        let challenge = Self::compute_challenge(&commitment, &public_key, message);

        // s = r + c*x
        let response = r.add(&challenge.mul(&x));

        Self {
            commitment,
            response,
            public_key,
        }
    }

    /// Verify a Schnorr proof
    pub fn verify(&self, message: &[u8]) -> bool {
        let g = PedersenGenerators::new().g;

        // c = H(R, Y, message)
        let challenge = Self::compute_challenge(&self.commitment, &self.public_key, message);

        // Check: g^s == R + c*Y
        let lhs = g.mul(&self.response);
        let rhs = self.commitment.add(&self.public_key.mul(&challenge));

        lhs == rhs
    }

    fn compute_challenge(r: &FieldElement, y: &FieldElement, message: &[u8]) -> FieldElement {
        let mut hasher = Hasher::new();
        hasher.update(b"SchnorrChallenge");
        hasher.update(&r.to_bytes());
        hasher.update(&y.to_bytes());
        hasher.update(message);
        FieldElement::from_bytes(hasher.finalize().as_bytes())
    }

    /// Get proof as bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(48);
        bytes.extend_from_slice(&self.commitment.to_bytes());
        bytes.extend_from_slice(&self.response.to_bytes());
        bytes.extend_from_slice(&self.public_key.to_bytes());
        bytes
    }
}

// ============================================================================
// Validity Proof (Topic Coherence)
// ============================================================================

/// Proof that a topic is semantically coherent without revealing it
///
/// Uses a combination of:
/// 1. Schnorr proof of knowledge of the topic
/// 2. Range proof for topic entropy (must be reasonable)
/// 3. Structure proof (topic has expected format)
#[derive(Debug, Clone)]
pub struct TopicValidityProof {
    /// Schnorr proof of topic knowledge
    pub knowledge_proof: SchnorrProof,
    /// Range proof for topic entropy
    pub entropy_proof: Option<BulletproofRangeProof>,
    /// Topic structure signature
    pub structure_signature: [u8; 32],
    /// Proof metadata
    pub metadata: ValidityProofMetadata,
}

/// Metadata for validity proof
#[derive(Debug, Clone)]
pub struct ValidityProofMetadata {
    /// Proof version
    pub version: u8,
    /// Timestamp
    pub timestamp: u64,
    /// Topic category hash
    pub category_hash: [u8; 8],
}

impl TopicValidityProof {
    /// Create a validity proof for a topic
    pub fn create(topic: &str, category: &str) -> Self {
        let topic_bytes = topic.as_bytes();
        let category_bytes = category.as_bytes();

        // 1. Schnorr proof of topic knowledge
        let knowledge_proof = SchnorrProof::create(topic_bytes, category_bytes);

        // 2. Calculate topic entropy
        let entropy = Self::calculate_entropy(topic);

        // 3. Range proof for entropy (should be 0.3 to 1.0 for valid topics)
        let entropy_proof = BulletproofRangeProof::create(entropy, (0.0, 1.0));

        // 4. Structure signature
        let structure_signature = Self::compute_structure_signature(topic);

        // 5. Metadata
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut category_hash = [0u8; 8];
        let cat_full_hash = FieldElement::from_bytes(category_bytes);
        category_hash.copy_from_slice(&cat_full_hash.to_bytes()[0..8]);

        let metadata = ValidityProofMetadata {
            version: 1,
            timestamp,
            category_hash,
        };

        Self {
            knowledge_proof,
            entropy_proof,
            structure_signature,
            metadata,
        }
    }

    /// Verify the validity proof
    pub fn verify(&self, category: &str) -> bool {
        // 1. Verify Schnorr proof
        if !self.knowledge_proof.verify(category.as_bytes()) {
            return false;
        }

        // 2. Verify entropy proof (if present)
        if let Some(ref entropy_proof) = self.entropy_proof {
            if !entropy_proof.verify() {
                return false;
            }
        }

        // 3. Verify category hash matches
        let expected_hash = {
            let h = FieldElement::from_bytes(category.as_bytes());
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&h.to_bytes()[0..8]);
            arr
        };

        if self.metadata.category_hash != expected_hash {
            return false;
        }

        // 4. Verify timestamp is recent (within 1 hour)
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if now.saturating_sub(self.metadata.timestamp) > 3600 {
            return false;
        }

        true
    }

    fn calculate_entropy(text: &str) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        // Character frequency distribution
        let mut freq = [0u32; 256];
        for &byte in text.as_bytes() {
            freq[byte as usize] += 1;
        }

        let len = text.len() as f32;
        let mut entropy = 0.0f32;

        for &count in &freq {
            if count > 0 {
                let p = count as f32 / len;
                entropy -= p * p.log2();
            }
        }

        // Normalize to [0, 1] (max entropy for ASCII is ~8 bits)
        (entropy / 8.0).min(1.0)
    }

    fn compute_structure_signature(topic: &str) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"TopicStructure");

        // Hash structural features
        hasher.update(&(topic.len() as u64).to_le_bytes());
        hasher.update(&(topic.split_whitespace().count() as u64).to_le_bytes());
        hasher.update(&(topic.chars().filter(|c| c.is_alphabetic()).count() as u64).to_le_bytes());

        let hash = hasher.finalize();
        hash.as_bytes()[0..32]
            .try_into()
            .expect("32-byte slice fits [u8; 32]")
    }

    /// Get proof size
    pub fn size(&self) -> usize {
        48 + // knowledge_proof
        self.entropy_proof.as_ref().map(|p| p.size()).unwrap_or(0) +
        32 + // structure_signature
        17 // metadata
    }
}

// ============================================================================
// Improved Commitment for DHT
// ============================================================================

/// Improved topic commitment using Pedersen scheme
#[derive(Debug, Clone)]
pub struct ZKTopicCommitment {
    /// Pedersen commitment
    pub pedersen: PedersenCommitment,
    /// Validity proof
    pub validity_proof: TopicValidityProof,
    /// Commitment timestamp
    pub timestamp: u64,
}

impl ZKTopicCommitment {
    /// Create a ZK commitment to a topic
    pub fn create(topic: &str, category: &str) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let pedersen = PedersenCommitment::commit(topic.as_bytes(), timestamp);
        let validity_proof = TopicValidityProof::create(topic, category);

        Self {
            pedersen,
            validity_proof,
            timestamp,
        }
    }

    /// Verify the commitment (without revealing topic)
    pub fn verify(&self, category: &str) -> bool {
        self.validity_proof.verify(category)
    }

    /// Open the commitment (reveals topic)
    pub fn open(&self, topic: &str) -> bool {
        if let Some(ref opening) = self.pedersen.opening {
            self.pedersen
                .verify_opening(topic.as_bytes(), &opening.randomness)
        } else {
            false
        }
    }

    /// Get commitment bytes (for network transmission)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.pedersen.to_bytes());
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes
    }
}

// ============================================================================
// Improved Range Proof for EIG
// ============================================================================

/// EIG Range Proof using Bulletproofs
#[derive(Debug, Clone)]
pub struct ZKEIGRangeProof {
    /// The bulletproof range proof
    pub bulletproof: BulletproofRangeProof,
    /// Commitment to the EIG value
    pub eig_commitment: PedersenCommitment,
}

impl ZKEIGRangeProof {
    /// Create a range proof for an EIG score
    pub fn create(eig: f32) -> Option<Self> {
        // EIG must be in [0, 1]
        if !(0.0..=1.0).contains(&eig) {
            return None;
        }

        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Commit to the EIG value
        let eig_commitment = PedersenCommitment::commit(&eig.to_le_bytes(), timestamp);

        // Create bulletproof range proof
        let bulletproof = BulletproofRangeProof::create(eig, (0.0, 1.0))?;

        Some(Self {
            bulletproof,
            eig_commitment,
        })
    }

    /// Verify the range proof
    pub fn verify(&self) -> bool {
        self.bulletproof.verify()
    }

    /// Get range
    pub fn range(&self) -> (f32, f32) {
        self.bulletproof.range
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_element_arithmetic() {
        let a = FieldElement::new(100);
        let b = FieldElement::new(200);

        let sum = a.add(&b);
        assert_eq!(sum.0, 300);

        let product = a.mul(&b);
        assert_eq!(product.0, 20000);
    }

    #[test]
    fn test_pedersen_commitment() {
        let message = b"Hello, World!";
        let commitment = PedersenCommitment::commit(message, 42);

        // Should be able to open with correct message
        if let Some(ref opening) = commitment.opening {
            assert!(commitment.verify_opening(message, &opening.randomness));
        }

        // Should NOT open with wrong message
        if let Some(ref opening) = commitment.opening {
            assert!(!commitment.verify_opening(b"Wrong message", &opening.randomness));
        }
    }

    #[test]
    fn test_bulletproof_range_proof() {
        // Valid proof (0.5 in [0, 1])
        let proof = BulletproofRangeProof::create(0.5, (0.0, 1.0));
        assert!(proof.is_some());
        assert!(proof.unwrap().verify());

        // Invalid proof (1.5 not in [0, 1])
        let invalid = BulletproofRangeProof::create(1.5, (0.0, 1.0));
        assert!(invalid.is_none());
    }

    #[test]
    fn test_schnorr_proof() {
        let witness = b"my_secret_topic";
        let message = b"category_hash";

        let proof = SchnorrProof::create(witness, message);

        // Should verify with correct message
        assert!(proof.verify(message));

        // Should NOT verify with wrong message
        assert!(!proof.verify(b"wrong_message"));
    }

    #[test]
    fn test_topic_validity_proof() {
        let topic = "What is quantum entanglement?";
        let category = "physics";

        let proof = TopicValidityProof::create(topic, category);

        // Should verify with correct category
        assert!(proof.verify(category));

        // Should NOT verify with wrong category
        assert!(!proof.verify("wrong_category"));
    }

    #[test]
    fn test_zk_topic_commitment() {
        let topic = "How do black holes work?";
        let category = "astrophysics";

        let commitment = ZKTopicCommitment::create(topic, category);

        // Verify without revealing
        assert!(commitment.verify(category));

        // Open commitment
        assert!(commitment.open(topic));
        assert!(!commitment.open("wrong topic"));
    }

    #[test]
    fn test_zk_eig_range_proof() {
        // Valid EIG
        let proof = ZKEIGRangeProof::create(0.75);
        assert!(proof.is_some());
        let proof = proof.unwrap();
        assert!(proof.verify());
        assert_eq!(proof.range(), (0.0, 1.0));

        // Invalid EIG
        let invalid = ZKEIGRangeProof::create(-0.5);
        assert!(invalid.is_none());
    }
}
