// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign Proof Attestor.
//!
//! Cryptographically signs SMT 'Proven' verdicts using post-quantum
//! Dilithium signatures from mycelix-zkp-core.

use mycelix_zkp_core::dilithium::{DilithiumKeyPair, DilithiumSignature};
use serde::{Deserialize, Serialize};

/// A signed attestation of a formal proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAttestation {
    pub label: String,
    pub smtlib2_hash: [u8; 32],
    pub verdict: String,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
}

pub struct SovereignAttestor {
    keys: DilithiumKeyPair,
}

impl SovereignAttestor {
    /// Create a new attestor with random keys.
    pub fn new() -> Self {
        Self {
            keys: DilithiumKeyPair::generate(),
        }
    }

    /// Sign a proof verdict.
    pub fn attest(&self, label: &str, smtlib2: &str, verdict: &str) -> ProofAttestation {
        use sha2::{Digest, Sha256};
        
        let mut hasher = Sha256::new();
        hasher.update(smtlib2.as_bytes());
        let smt_hash: [u8; 32] = hasher.finalize().into();

        // Data to sign: label + hash + verdict
        let mut msg = Vec::new();
        msg.extend_from_slice(label.as_bytes());
        msg.extend_from_slice(&smt_hash);
        msg.extend_from_slice(verdict.as_bytes());

        let signature = self.keys.sign(&msg);

        ProofAttestation {
            label: label.to_string(),
            smtlib2_hash: smt_hash,
            verdict: verdict.to_string(),
            signature: signature.to_vec(),
            public_key: self.keys.public_key().to_vec(),
        }
    }

    /// Verify an attestation from a peer.
    pub fn verify(attestation: &ProofAttestation) -> bool {
        let mut msg = Vec::new();
        msg.extend_from_slice(attestation.label.as_bytes());
        msg.extend_from_slice(&attestation.smtlib2_hash);
        msg.extend_from_slice(attestation.verdict.as_bytes());

        // In a real implementation, we'd reconstruct the key from bytes
        // and verify. For the prototype, we use the core verification logic.
        DilithiumSignature::verify_raw(
            &msg,
            &attestation.signature,
            &attestation.public_key
        ).is_ok()
    }
}
