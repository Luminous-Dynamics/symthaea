// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign Proof Attestor.
//!
//! Cryptographically signs SMT 'Proven' verdicts using post-quantum
//! Dilithium signatures from mycelix-zkp-core.

use mycelix_zkp_core::dilithium::{DilithiumKeypair, verify_signature};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// A signed attestation of a formal proof and its compiled binary realization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAttestation {
    pub label: String,
    pub smtlib2_hash: [u8; 32],
    /// Cryptographic hash of the compiled binary artifact (supply-chain integrity).
    pub binary_hash: Option<[u8; 32]>,
    pub verdict: String,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
}

pub struct SovereignAttestor {
    keys: DilithiumKeypair,
}

static PROCESS_ATTESTOR: Lazy<Mutex<SovereignAttestor>> =
    Lazy::new(|| Mutex::new(SovereignAttestor::from_env_or_generate()));

impl SovereignAttestor {
    /// Create a new attestor with random keys.
    pub fn new() -> Self {
        Self {
            keys: DilithiumKeypair::generate(),
        }
    }

    /// Create an attestor from stable environment key material when present.
    pub fn from_env_or_generate() -> Self {
        match (
            std::env::var("SYMTHAEA_ATTESTOR_PUBLIC_KEY_HEX"),
            std::env::var("SYMTHAEA_ATTESTOR_SECRET_KEY_HEX"),
        ) {
            (Ok(public_hex), Ok(secret_hex)) => {
                let public = decode_hex(&public_hex);
                let secret = decode_hex(&secret_hex);
                if let (Ok(public), Ok(secret)) = (public, secret) {
                    if let Ok(keys) = DilithiumKeypair::from_bytes(public, secret) {
                        return Self { keys };
                    }
                }
                Self::new()
            }
            _ => Self::new(),
        }
    }

    /// Sign with the stable process attestor.
    pub fn attest_with_process_key(
        label: &str,
        smtlib2: &str,
        verdict: &str,
        binary_data: Option<&[u8]>,
    ) -> ProofAttestation {
        PROCESS_ATTESTOR
            .lock()
            .attest(label, smtlib2, verdict, binary_data)
    }

    /// Sign a proof verdict and an optional binary realization.
    pub fn attest(
        &self,
        label: &str,
        smtlib2: &str,
        verdict: &str,
        binary_data: Option<&[u8]>,
    ) -> ProofAttestation {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(smtlib2.as_bytes());
        let smt_hash: [u8; 32] = hasher.finalize().into();

        let mut binary_hash: Option<[u8; 32]> = None;
        if let Some(data) = binary_data {
            let mut bin_hasher = Sha256::new();
            bin_hasher.update(data);
            binary_hash = Some(bin_hasher.finalize().into());
        }

        // Data to sign: label + smt_hash + binary_hash (if any) + verdict
        let mut msg = Vec::new();
        msg.extend_from_slice(label.as_bytes());
        msg.extend_from_slice(&smt_hash);
        if let Some(bh) = binary_hash {
            msg.extend_from_slice(&bh);
        }
        msg.extend_from_slice(verdict.as_bytes());

        let signature = self.keys.sign(&msg).unwrap_or_default();

        ProofAttestation {
            label: label.to_string(),
            smtlib2_hash: smt_hash,
            binary_hash,
            verdict: verdict.to_string(),
            signature,
            public_key: self.keys.public_key().to_vec(),
        }
    }

    /// Verify an attestation from a peer, including supply-chain binary integrity.
    pub fn verify(attestation: &ProofAttestation, actual_binary: Option<&[u8]>) -> bool {
        use sha2::{Digest, Sha256};

        // 1. Verify binary integrity if present
        if let Some(expected_hash) = attestation.binary_hash {
            if let Some(binary) = actual_binary {
                let mut hasher = Sha256::new();
                hasher.update(binary);
                let actual_hash: [u8; 32] = hasher.finalize().into();
                if actual_hash != expected_hash {
                    tracing::error!(
                        "SUPPLY-CHAIN BREACH: Attested binary hash does not match physical artifact."
                    );
                    return false;
                }
            }
        }

        // 2. Verify Dilithium signature
        let mut msg = Vec::new();
        msg.extend_from_slice(attestation.label.as_bytes());
        msg.extend_from_slice(&attestation.smtlib2_hash);
        if let Some(bh) = attestation.binary_hash {
            msg.extend_from_slice(&bh);
        }
        msg.extend_from_slice(attestation.verdict.as_bytes());

        verify_signature(&msg, &attestation.signature, &attestation.public_key).unwrap_or(false)
    }
}

fn decode_hex(input: &str) -> Result<Vec<u8>, ()> {
    let cleaned = input.trim();
    if cleaned.len() % 2 != 0 {
        return Err(());
    }

    (0..cleaned.len())
        .step_by(2)
        .map(|idx| u8::from_str_radix(&cleaned[idx..idx + 2], 16).map_err(|_| ()))
        .collect()
}
