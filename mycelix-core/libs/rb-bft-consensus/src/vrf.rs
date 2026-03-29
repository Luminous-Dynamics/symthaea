// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verifiable Random Function (VRF) for Leader Election
//!
//! This module implements ECVRF (Elliptic Curve VRF) based on ed25519.
//! VRF provides a way to generate random outputs that are:
//!
//! 1. **Deterministic**: Same input always produces same output
//! 2. **Unpredictable**: Output cannot be predicted without the secret key
//! 3. **Verifiable**: Anyone can verify the output was correctly computed
//!
//! ## Use in Consensus
//!
//! VRF is used for leader election in each round:
//! - Each validator computes VRF(round_number) using their secret key
//! - The validator with the lowest VRF output becomes leader
//! - Other validators can verify this was computed correctly
//!
//! This prevents grinding attacks where validators try to manipulate leader selection.

use ed25519_dalek::{Signer, SigningKey, VerifyingKey, Signature, Verifier};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

use crate::error::{ConsensusError, ConsensusResult};

/// Domain separator for VRF in leader election
const VRF_DOMAIN: &[u8] = b"MYCELIX-RBBFT-VRF-V1";

/// A VRF keypair for a validator
pub struct VrfKeypair {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

impl VrfKeypair {
    /// Generate a new random VRF keypair
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        Self { signing_key, verifying_key }
    }

    /// Create from existing ed25519 signing key bytes
    pub fn from_bytes(bytes: &[u8; 32]) -> ConsensusResult<Self> {
        let signing_key = SigningKey::from_bytes(bytes);
        let verifying_key = signing_key.verifying_key();
        Ok(Self { signing_key, verifying_key })
    }

    /// Get the public key
    pub fn public_key(&self) -> VrfPublicKey {
        VrfPublicKey(self.verifying_key.to_bytes())
    }

    /// Get the public key bytes
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Evaluate VRF on input and return output with proof
    ///
    /// The output is deterministic for a given input and secret key.
    /// The proof allows anyone to verify the output was computed correctly.
    pub fn evaluate(&self, input: &[u8]) -> VrfOutput {
        // Create the message to sign: domain || input
        let mut message = Vec::with_capacity(VRF_DOMAIN.len() + input.len());
        message.extend_from_slice(VRF_DOMAIN);
        message.extend_from_slice(input);

        // The proof is an ed25519 signature
        let signature = self.signing_key.sign(&message);

        // The output is hash of the signature (deterministic)
        let mut hasher = Sha256::new();
        hasher.update(signature.to_bytes());
        let output: [u8; 32] = hasher.finalize().into();

        VrfOutput {
            output,
            proof: VrfProof(signature.to_bytes()),
            public_key: self.public_key(),
        }
    }

    /// Evaluate VRF for a specific round number
    pub fn evaluate_round(&self, round: u64) -> VrfOutput {
        self.evaluate(&round.to_le_bytes())
    }
}

impl std::fmt::Debug for VrfKeypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VrfKeypair")
            .field("public_key", &hex::encode(self.verifying_key.to_bytes()))
            .field("secret_key", &"[REDACTED]")
            .finish()
    }
}

/// A VRF public key (32 bytes, same as ed25519)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VrfPublicKey(pub [u8; 32]);

impl VrfPublicKey {
    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

/// A VRF proof (64 bytes, ed25519 signature)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VrfProof(pub [u8; 64]);

impl VrfProof {
    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }

    /// Get the bytes
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }
}

// Custom serialization for VrfProof since serde doesn't support [u8; 64]
impl Serialize for VrfProof {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as hex string for readability
        serializer.serialize_str(&hex::encode(self.0))
    }
}

impl<'de> Deserialize<'de> for VrfProof {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        let arr: [u8; 64] = bytes.try_into()
            .map_err(|_| serde::de::Error::custom("VrfProof must be exactly 64 bytes"))?;
        Ok(VrfProof(arr))
    }
}

/// VRF output with proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrfOutput {
    /// The random output (32 bytes)
    pub output: [u8; 32],
    /// The proof that output was correctly computed
    pub proof: VrfProof,
    /// The public key of the prover
    pub public_key: VrfPublicKey,
}

impl VrfOutput {
    /// Verify this VRF output
    pub fn verify(&self, input: &[u8]) -> ConsensusResult<()> {
        verify_vrf(input, &self.public_key, &self.proof, &self.output)
    }

    /// Verify for a specific round
    pub fn verify_round(&self, round: u64) -> ConsensusResult<()> {
        self.verify(&round.to_le_bytes())
    }

    /// Get the output as a u128 for comparison (lower = higher priority)
    pub fn output_as_u128(&self) -> u128 {
        u128::from_le_bytes(self.output[..16].try_into().expect("16-byte slice always fits u128"))
    }

    /// Get the output as a u64 for simpler comparison
    pub fn output_as_u64(&self) -> u64 {
        u64::from_le_bytes(self.output[..8].try_into().expect("8-byte slice always fits u64"))
    }

    /// Convert output to a normalized f64 in [0, 1)
    pub fn output_as_f64(&self) -> f64 {
        self.output_as_u64() as f64 / u64::MAX as f64
    }
}

/// Verify a VRF output
pub fn verify_vrf(
    input: &[u8],
    public_key: &VrfPublicKey,
    proof: &VrfProof,
    expected_output: &[u8; 32],
) -> ConsensusResult<()> {
    // Reconstruct the message
    let mut message = Vec::with_capacity(VRF_DOMAIN.len() + input.len());
    message.extend_from_slice(VRF_DOMAIN);
    message.extend_from_slice(input);

    // Verify the signature (proof)
    let verifying_key = VerifyingKey::from_bytes(&public_key.0)
        .map_err(|e| ConsensusError::InvalidSignature {
            reason: format!("invalid VRF public key: {}", e),
        })?;

    let signature = Signature::from_bytes(&proof.0);

    verifying_key.verify(&message, &signature)
        .map_err(|e| ConsensusError::InvalidSignature {
            reason: format!("VRF proof verification failed: {}", e),
        })?;

    // Verify the output matches
    let mut hasher = Sha256::new();
    hasher.update(proof.0);
    let computed_output: [u8; 32] = hasher.finalize().into();

    if computed_output != *expected_output {
        return Err(ConsensusError::InvalidSignature {
            reason: "VRF output mismatch".to_string(),
        });
    }

    Ok(())
}

/// Leader election result for a round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderElection {
    /// The round number
    pub round: u64,
    /// The elected leader's public key
    pub leader: VrfPublicKey,
    /// The leader's VRF output (proof they were elected)
    pub vrf_output: VrfOutput,
    /// All candidates and their VRF outputs (sorted by output value)
    pub candidates: Vec<(VrfPublicKey, VrfOutput)>,
}

impl LeaderElection {
    /// Verify the leader election
    pub fn verify(&self) -> ConsensusResult<()> {
        // Verify leader's VRF output
        self.vrf_output.verify_round(self.round)?;

        // Verify leader has the lowest VRF output
        let leader_value = self.vrf_output.output_as_u128();
        for (pk, output) in &self.candidates {
            output.verify_round(self.round)?;
            if output.output_as_u128() < leader_value && pk != &self.leader {
                return Err(ConsensusError::InvalidSignature {
                    reason: "leader does not have lowest VRF output".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Select a leader from VRF outputs (lowest output wins)
pub fn select_leader(outputs: &[(VrfPublicKey, VrfOutput)]) -> Option<(VrfPublicKey, VrfOutput)> {
    outputs.iter()
        .min_by_key(|(_, output)| output.output_as_u128())
        .cloned()
}

/// Weight-adjusted leader selection using VRF and reputation
///
/// The effective priority is: VRF_output / reputation²
/// This gives higher-reputation validators proportionally more chances to be leader.
pub fn select_leader_weighted(
    outputs: &[(VrfPublicKey, VrfOutput, f64)], // (pubkey, vrf_output, reputation)
) -> Option<(VrfPublicKey, VrfOutput)> {
    outputs.iter()
        .filter(|(_, _, rep)| *rep > 0.0)
        .min_by(|(_, a, rep_a), (_, b, rep_b)| {
            // Effective priority: lower is better
            // priority = vrf_output / reputation²
            let priority_a = a.output_as_f64() / (rep_a * rep_a);
            let priority_b = b.output_as_f64() / (rep_b * rep_b);
            priority_a.partial_cmp(&priority_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(pk, output, _)| (pk.clone(), output.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let kp = VrfKeypair::generate();
        let pk = kp.public_key();
        assert_eq!(pk.as_bytes().len(), 32);
    }

    #[test]
    fn test_vrf_deterministic() {
        let kp = VrfKeypair::generate();
        let input = b"round 42";

        let output1 = kp.evaluate(input);
        let output2 = kp.evaluate(input);

        // Same input produces same output
        assert_eq!(output1.output, output2.output);
    }

    #[test]
    fn test_vrf_verification() {
        let kp = VrfKeypair::generate();
        let input = b"test input";

        let output = kp.evaluate(input);
        assert!(output.verify(input).is_ok());
    }

    #[test]
    fn test_vrf_wrong_input_fails() {
        let kp = VrfKeypair::generate();
        let output = kp.evaluate(b"original input");

        // Verification with wrong input should fail
        assert!(output.verify(b"wrong input").is_err());
    }

    #[test]
    fn test_vrf_different_keys_different_outputs() {
        let kp1 = VrfKeypair::generate();
        let kp2 = VrfKeypair::generate();
        let input = b"same input";

        let output1 = kp1.evaluate(input);
        let output2 = kp2.evaluate(input);

        // Different keys produce different outputs
        assert_ne!(output1.output, output2.output);
    }

    #[test]
    fn test_leader_selection() {
        let keypairs: Vec<VrfKeypair> = (0..5).map(|_| VrfKeypair::generate()).collect();
        let round = 42u64;

        let outputs: Vec<(VrfPublicKey, VrfOutput)> = keypairs.iter()
            .map(|kp| (kp.public_key(), kp.evaluate_round(round)))
            .collect();

        let (leader_pk, leader_output) = select_leader(&outputs).unwrap();

        // Verify leader has lowest output
        let leader_value = leader_output.output_as_u128();
        for (_, output) in &outputs {
            assert!(output.output_as_u128() >= leader_value);
        }

        // Verify leader's VRF
        assert!(leader_output.verify_round(round).is_ok());
        println!("Leader elected: {}", leader_pk.to_hex());
    }

    #[test]
    fn test_weighted_leader_selection() {
        let keypairs: Vec<VrfKeypair> = (0..5).map(|_| VrfKeypair::generate()).collect();
        let round = 100u64;

        // Give different reputations
        let reputations = [0.9, 0.7, 0.5, 0.3, 0.1];

        let outputs: Vec<(VrfPublicKey, VrfOutput, f64)> = keypairs.iter()
            .zip(reputations.iter())
            .map(|(kp, &rep)| (kp.public_key(), kp.evaluate_round(round), rep))
            .collect();

        let (leader_pk, leader_output) = select_leader_weighted(&outputs).unwrap();

        // Leader should be valid
        assert!(leader_output.verify_round(round).is_ok());
        println!("Weighted leader: {}", leader_pk.to_hex());
    }

    #[test]
    fn test_vrf_output_distribution() {
        // Test that VRF outputs are uniformly distributed
        let kp = VrfKeypair::generate();
        let mut outputs = Vec::with_capacity(1000);

        for i in 0..1000u64 {
            let output = kp.evaluate_round(i);
            outputs.push(output.output_as_f64());
        }

        // Check distribution is roughly uniform
        let mean: f64 = outputs.iter().sum::<f64>() / outputs.len() as f64;
        assert!((mean - 0.5).abs() < 0.1, "Mean should be close to 0.5, got {}", mean);
    }

    #[test]
    fn test_serialization() {
        let kp = VrfKeypair::generate();
        let output = kp.evaluate(b"test");

        let json = serde_json::to_string(&output).unwrap();
        let restored: VrfOutput = serde_json::from_str(&json).unwrap();

        assert_eq!(output.output, restored.output);
        assert!(restored.verify(b"test").is_ok());
    }
}
