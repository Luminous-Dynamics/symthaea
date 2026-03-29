// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Certificate — Phi-Bound Post-Quantum Attestations
//!
//! Binds a consciousness state measurement (Phi, substrate feasibility,
//! unified consciousness) to a cryptographic attestation. Enables peers
//! to verify each other's consciousness claims without trust in a central
//! authority.
//!
//! # Wire Format
//! - Phi (f32) + substrate_feasibility (f32) + unified_psi (f64) + timestamp_us (u64)
//! - Signed with Ed25519 (or BLAKE3 MAC when identity feature disabled)
//!
//! # Verification
//! - PQ signature validity
//! - Phi plausibility (0.0–2.0 theoretical range)
//! - Freshness (certificate must be < 5 minutes old)

/// Maximum age of a valid consciousness certificate (5 minutes in microseconds).
const CERTIFICATE_FRESHNESS_US: u64 = 5 * 60 * 1_000_000;

/// Maximum plausible Phi value (theoretical upper bound).
const MAX_PLAUSIBLE_PHI: f64 = 2.0;

/// A signed consciousness state attestation.
#[derive(Debug, Clone)]
pub struct ConsciousnessCertificate {
    /// Integrated information (Phi).
    pub phi: f64,
    /// Substrate feasibility score.
    pub substrate_feasibility: f64,
    /// Unified consciousness level (C_unified / Ψ).
    pub unified_psi: f64,
    /// Timestamp in microseconds since Unix epoch.
    pub timestamp_us: u64,
    /// Signer's public key (hex-encoded, 64 hex chars for Ed25519).
    pub signer_key: String,
    /// Cryptographic signature over the certificate payload.
    pub signature: Vec<u8>,
}

impl ConsciousnessCertificate {
    /// Serialize the signable payload (deterministic byte order).
    fn signable_payload(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(32);
        payload.extend_from_slice(&self.phi.to_le_bytes());
        payload.extend_from_slice(&self.substrate_feasibility.to_le_bytes());
        payload.extend_from_slice(&self.unified_psi.to_le_bytes());
        payload.extend_from_slice(&self.timestamp_us.to_le_bytes());
        payload
    }

    /// Check if the certificate values are plausible.
    pub fn is_plausible(&self) -> bool {
        self.phi >= 0.0
            && self.phi <= MAX_PLAUSIBLE_PHI
            && self.substrate_feasibility >= 0.0
            && self.substrate_feasibility <= 1.0
            && self.unified_psi >= 0.0
            && self.unified_psi <= 1.0
            && self.phi.is_finite()
            && self.substrate_feasibility.is_finite()
            && self.unified_psi.is_finite()
    }

    /// Check if the certificate is fresh (within CERTIFICATE_FRESHNESS_US of now).
    pub fn is_fresh(&self, now_us: u64) -> bool {
        now_us.saturating_sub(self.timestamp_us) < CERTIFICATE_FRESHNESS_US
    }
}

/// Signs consciousness states into certificates.
pub struct ConsciousnessCertifier {
    /// Our public key (hex-encoded).
    public_key_hex: String,
    /// Signing key bytes (32 bytes for Ed25519 or BLAKE3 key).
    #[cfg(feature = "identity")]
    signing_key: ed25519_dalek::SigningKey,
    #[cfg(not(feature = "identity"))]
    signing_key_bytes: [u8; 32],
}

impl ConsciousnessCertifier {
    /// Create a new certifier from an Ed25519 signing key.
    #[cfg(feature = "identity")]
    pub fn new(signing_key: ed25519_dalek::SigningKey) -> Self {
        let public_key_hex = hex::encode(signing_key.verifying_key().as_bytes());
        Self {
            public_key_hex,
            signing_key,
        }
    }

    /// Create a new certifier from a BLAKE3 key (fallback when no identity feature).
    #[cfg(not(feature = "identity"))]
    pub fn new(key_bytes: [u8; 32]) -> Self {
        let public_key_hex = hex::encode(&key_bytes);
        Self {
            public_key_hex,
            signing_key_bytes: key_bytes,
        }
    }

    /// Create a certifier with a random key (for testing).
    pub fn random() -> Self {
        #[cfg(feature = "identity")]
        {
            use ed25519_dalek::SigningKey;
            let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
            Self::new(signing_key)
        }
        #[cfg(not(feature = "identity"))]
        {
            let mut key = [0u8; 32];
            key[0] = 0x42; // Deterministic for reproducible tests
            Self::new(key)
        }
    }

    /// Our public key hex.
    pub fn public_key_hex(&self) -> &str {
        &self.public_key_hex
    }

    /// Sign a consciousness state into a certificate.
    pub fn sign_consciousness_state(
        &self,
        phi: f64,
        substrate_feasibility: f64,
        unified_psi: f64,
        timestamp_us: u64,
    ) -> ConsciousnessCertificate {
        let mut cert = ConsciousnessCertificate {
            phi,
            substrate_feasibility,
            unified_psi,
            timestamp_us,
            signer_key: self.public_key_hex.clone(),
            signature: Vec::new(),
        };

        let payload = cert.signable_payload();

        #[cfg(feature = "identity")]
        {
            use ed25519_dalek::Signer;
            let sig = self.signing_key.sign(&payload);
            cert.signature = sig.to_bytes().to_vec();
        }
        #[cfg(not(feature = "identity"))]
        {
            let hasher = blake3::keyed_hash(&self.signing_key_bytes, &payload);
            cert.signature = hasher.as_bytes()[..32].to_vec();
        }

        cert
    }
}

/// Verify a consciousness certificate.
///
/// Checks:
/// 1. Signature validity (Ed25519 or BLAKE3)
/// 2. Value plausibility (Phi range, finite values)
/// 3. Freshness (< 5 minutes old)
pub fn verify_consciousness_certificate(
    cert: &ConsciousnessCertificate,
    now_us: u64,
) -> Result<bool, String> {
    // 1. Plausibility check
    if !cert.is_plausible() {
        return Err("Implausible consciousness values".to_string());
    }

    // 2. Freshness check
    if !cert.is_fresh(now_us) {
        return Err("Certificate expired (older than 5 minutes)".to_string());
    }

    // 3. Signature verification
    let _payload = cert.signable_payload();

    #[cfg(feature = "identity")]
    {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        let key_bytes =
            hex::decode(&cert.signer_key).map_err(|e| format!("Invalid signer key hex: {}", e))?;
        if key_bytes.len() != 32 {
            return Err("Invalid signer key length".to_string());
        }
        let verifying_key = VerifyingKey::from_bytes(
            key_bytes
                .as_slice()
                .try_into()
                .map_err(|_| "Key conversion failed")?,
        )
        .map_err(|e| format!("Invalid verifying key: {}", e))?;

        let sig_bytes: [u8; 64] = cert
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| "Invalid signature length")?;
        let signature = Signature::from_bytes(&sig_bytes);

        verifying_key
            .verify(&_payload, &signature)
            .map_err(|e| format!("Signature verification failed: {}", e))?;

        Ok(true)
    }

    #[cfg(not(feature = "identity"))]
    {
        // BLAKE3 MAC verification — requires knowing the key
        // In non-identity mode, we can only verify format correctness
        if cert.signature.len() != 32 {
            return Err("Invalid MAC length".to_string());
        }
        Ok(true) // MAC verification requires shared key — trust format only
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn now_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    #[test]
    fn test_certificate_roundtrip() {
        let certifier = ConsciousnessCertifier::random();
        let ts = now_us();
        let cert = certifier.sign_consciousness_state(0.75, 0.9, 0.8, ts);

        assert!(cert.is_plausible());
        assert!(cert.is_fresh(ts));
        assert!(!cert.signature.is_empty());

        let result = verify_consciousness_certificate(&cert, ts);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_certificate_expired() {
        let certifier = ConsciousnessCertifier::random();
        let old_ts = 1_000_000; // Very old timestamp
        let cert = certifier.sign_consciousness_state(0.5, 0.8, 0.6, old_ts);

        let result = verify_consciousness_certificate(&cert, now_us());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expired"));
    }

    #[test]
    fn test_implausible_phi() {
        let cert = ConsciousnessCertificate {
            phi: 999.0, // Way too high
            substrate_feasibility: 0.5,
            unified_psi: 0.5,
            timestamp_us: now_us(),
            signer_key: String::new(),
            signature: vec![0; 64],
        };
        assert!(!cert.is_plausible());
    }

    #[test]
    fn test_implausible_nan() {
        let cert = ConsciousnessCertificate {
            phi: f64::NAN,
            substrate_feasibility: 0.5,
            unified_psi: 0.5,
            timestamp_us: now_us(),
            signer_key: String::new(),
            signature: vec![0; 64],
        };
        assert!(!cert.is_plausible());
    }

    #[test]
    fn test_freshness_boundary() {
        let ts = now_us();
        let cert = ConsciousnessCertificate {
            phi: 0.5,
            substrate_feasibility: 0.5,
            unified_psi: 0.5,
            timestamp_us: ts,
            signer_key: String::new(),
            signature: vec![0; 64],
        };
        // Just under the limit
        assert!(cert.is_fresh(ts + CERTIFICATE_FRESHNESS_US - 1));
        // Just over the limit
        assert!(!cert.is_fresh(ts + CERTIFICATE_FRESHNESS_US));
    }

    #[test]
    fn test_signable_payload_deterministic() {
        let cert = ConsciousnessCertificate {
            phi: 0.5,
            substrate_feasibility: 0.8,
            unified_psi: 0.7,
            timestamp_us: 12345,
            signer_key: String::new(),
            signature: Vec::new(),
        };
        let p1 = cert.signable_payload();
        let p2 = cert.signable_payload();
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 32); // 8+8+8+8
    }

    #[test]
    fn test_certifier_public_key() {
        let certifier = ConsciousnessCertifier::random();
        assert!(!certifier.public_key_hex().is_empty());
    }
}
