// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign Proof Attestor.
//!
//! Cryptographically signs SMT 'Proven' verdicts using post-quantum
//! Dilithium signatures from mycelix-zkp-core.

use mycelix_zkp_core::dilithium::{
    verify_detached_signature as verify_mldsa_detached_signature, verify_signature,
    DilithiumKeypair,
};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const TRANSCRIPT_VERSION_V2: u16 = 2;

/// Missing version on historical JSON attestations means the legacy v1
/// transcript. Legacy attestations are rejected by the verifier; the default
/// preserves the ability to load and inspect old JSON records without treating
/// their signatures as valid v2 attestations.
fn legacy_transcript_version() -> u16 {
    1
}

/// A signed attestation of a formal proof and its compiled binary realization.
///
/// `transcript_version == 2` uses a domain-separated, length-prefixed
/// transcript with an explicit presence tag for `binary_hash`. Version 1
/// records used an ambiguous concatenation and are rejected for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAttestation {
    pub label: String,
    pub smtlib2_hash: [u8; 32],
    /// Cryptographic hash of the compiled binary artifact (supply-chain integrity).
    pub binary_hash: Option<[u8; 32]>,
    pub verdict: String,
    /// Detached signature over the canonical transcript for the selected format version.
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    /// Signature transcript format. Historical JSON without this field loads as v1.
    #[serde(default = "legacy_transcript_version")]
    pub transcript_version: u16,
}

pub struct SovereignAttestor {
    keys: DilithiumKeypair,
}

static PROCESS_ATTESTOR: Lazy<Result<Mutex<SovereignAttestor>, String>> =
    Lazy::new(|| SovereignAttestor::from_env_or_generate().map(Mutex::new));

impl SovereignAttestor {
    /// Create a new attestor with random keys.
    pub fn new() -> Self {
        Self {
            keys: DilithiumKeypair::generate(),
        }
    }

    /// Create an attestor from stable environment key material when present.
    ///
    /// If neither variable is set, create a process-local random key. If either
    /// variable is configured, both must be present and valid. Invalid configured
    /// identity material is an error, never a reason to silently rotate identity.
    pub fn from_env_or_generate() -> Result<Self, String> {
        use std::env::VarError;

        let public_env = std::env::var("SYMTHAEA_ATTESTOR_PUBLIC_KEY_HEX");
        let secret_env = std::env::var("SYMTHAEA_ATTESTOR_SECRET_KEY_HEX");
        let public_absent = matches!(&public_env, Err(VarError::NotPresent));
        let secret_absent = matches!(&secret_env, Err(VarError::NotPresent));

        if public_absent && secret_absent {
            return Ok(Self::new());
        }

        let (public_hex, secret_hex) = match (public_env, secret_env) {
            (Ok(public), Ok(secret)) => (public, secret),
            _ => {
                return Err(
                    "configured attestor identity requires both public and secret key variables"
                        .to_string(),
                );
            }
        };

        let public = decode_hex(&public_hex)
            .map_err(|_| "configured attestor public key is not valid hex".to_string())?;
        let secret = decode_hex(&secret_hex)
            .map_err(|_| "configured attestor secret key is not valid hex".to_string())?;
        let keys = DilithiumKeypair::from_bytes(public, secret)
            .map_err(|_| "configured attestor key material failed validation".to_string())?;

        // Verify pair consistency even on branches that do not yet contain the
        // stronger from_bytes guard. This avoids treating same-length mismatched
        // key material as a configured identity.
        const CONFIG_CHECK: &[u8] = b"SYMTHAEA-ATTESTOR-CONFIG-CHECK/v1";
        let signed = keys
            .sign(CONFIG_CHECK)
            .map_err(|_| "configured attestor keypair cannot sign".to_string())?;
        let valid = verify_signature(CONFIG_CHECK, &signed, keys.public_key())
            .map_err(|_| "configured attestor keypair cannot verify".to_string())?;
        if !valid {
            return Err("configured attestor public and secret keys do not match".to_string());
        }

        Ok(Self { keys })
    }

    /// Sign with the stable process attestor.
    ///
    /// Configuration or signing errors are returned instead of silently
    /// generating an attestation under a different process identity.
    pub fn attest_with_process_key(
        label: &str,
        smtlib2: &str,
        verdict: &str,
        binary_data: Option<&[u8]>,
    ) -> Result<ProofAttestation, String> {
        let attestor = PROCESS_ATTESTOR
            .as_ref()
            .map_err(|error| error.clone())?;
        attestor.lock().attest(label, smtlib2, verdict, binary_data)
    }

    /// Public key for this attestor; callers should provision it as a trust anchor.
    pub fn public_key(&self) -> &[u8] {
        self.keys.public_key()
    }

    /// Public key associated with the configured process attestor.
    ///
    /// Consumers that verify process attestations should obtain this through a
    /// trusted provisioning channel, not by trusting the key embedded in an
    /// incoming attestation.
    pub fn process_public_key() -> Result<Vec<u8>, String> {
        let attestor = PROCESS_ATTESTOR
            .as_ref()
            .map_err(|error| error.clone())?;
        let public_key = attestor.lock().keys.public_key().to_vec();
        Ok(public_key)
    }

    /// Sign a proof verdict and an optional binary realization.
    pub fn attest(
        &self,
        label: &str,
        smtlib2: &str,
        verdict: &str,
        binary_data: Option<&[u8]>,
    ) -> Result<ProofAttestation, String> {
        let smt_hash: [u8; 32] = Sha256::digest(smtlib2.as_bytes()).into();
        let binary_hash: Option<[u8; 32]> =
            binary_data.map(|data| Sha256::digest(data).into());

        let mut attestation = ProofAttestation {
            label: label.to_string(),
            smtlib2_hash: smt_hash,
            binary_hash,
            verdict: verdict.to_string(),
            signature: Vec::new(),
            public_key: self.keys.public_key().to_vec(),
            transcript_version: TRANSCRIPT_VERSION_V2,
        };

        // Do not return an apparently complete attestation with an empty
        // signature when the signing operation fails.
        let message = signing_message_v2(&attestation);
        attestation.signature = self
            .keys
            .sign_detached(&message)
            .map_err(|_| "attestation signing failed".to_string())?;
        Ok(attestation)
    }

    /// Verify an attestation against a caller-provisioned trusted signer key.
    ///
    /// The public key embedded in the attestation is untrusted input; a valid
    /// signature under a self-supplied key does not establish signer identity.
    /// Callers must obtain `trusted_public_key` from an independently trusted
    /// policy/configuration channel. If the attestation commits to a binary, that
    /// exact binary is also required and checked. Legacy v1 records fail closed.
    pub fn verify(
        attestation: &ProofAttestation,
        actual_binary: Option<&[u8]>,
        trusted_public_key: &[u8],
    ) -> bool {
        if attestation.transcript_version != TRANSCRIPT_VERSION_V2 {
            return false;
        }
        if attestation.public_key.as_slice() != trusted_public_key {
            return false;
        }

        match (attestation.binary_hash, actual_binary) {
            (Some(expected_hash), Some(binary)) => {
                let actual_hash: [u8; 32] = Sha256::digest(binary).into();
                if actual_hash != expected_hash {
                    tracing::error!(
                        "SUPPLY-CHAIN BREACH: Attested binary hash does not match physical artifact."
                    );
                    return false;
                }
            }
            (Some(_), None) | (None, Some(_)) => return false,
            (None, None) => {}
        }

        let message = signing_message_v2(attestation);
        verify_mldsa_detached_signature(
            &message,
            &attestation.signature,
            trusted_public_key,
        )
        .unwrap_or(false)
    }
}

/// Canonical v2 transcript. Length prefixes and an explicit option tag prevent
/// ambiguous concatenations across variable-size fields and optional hashes.
fn signing_message_v2(attestation: &ProofAttestation) -> Vec<u8> {
    const DOMAIN: &[u8] = b"SYMTHAEA-SOVEREIGN-ATTESTATION/v2\0";

    let mut message = Vec::with_capacity(
        DOMAIN.len() + 2 + 8 + attestation.label.len() + 32 + 1 + 32 + 8
            + attestation.verdict.len(),
    );
    message.extend_from_slice(DOMAIN);
    message.extend_from_slice(&attestation.transcript_version.to_le_bytes());

    message.extend_from_slice(&(attestation.label.len() as u64).to_le_bytes());
    message.extend_from_slice(attestation.label.as_bytes());
    message.extend_from_slice(&attestation.smtlib2_hash);

    match attestation.binary_hash {
        Some(hash) => {
            message.push(1);
            message.extend_from_slice(&hash);
        }
        None => message.push(0),
    }

    message.extend_from_slice(&(attestation.verdict.len() as u64).to_le_bytes());
    message.extend_from_slice(attestation.verdict.as_bytes());
    message
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

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SMT: &str = "(check-sat)";

    #[test]
    fn v2_attestation_without_binary_verifies_without_binary() {
        let attestor = SovereignAttestor::new();
        let trusted_public_key = attestor.public_key().to_vec();
        let attestation = attestor.attest("unit-test", TEST_SMT, "Proven", None).unwrap();
        assert_eq!(attestation.transcript_version, TRANSCRIPT_VERSION_V2);
        assert!(SovereignAttestor::verify(&attestation, None, &trusted_public_key));
    }

    #[test]
    fn v2_attestation_with_binary_requires_the_matching_binary() {
        let attestor = SovereignAttestor::new();
        let trusted_public_key = attestor.public_key().to_vec();
        let binary = b"compiled artifact";
        let attestation = attestor.attest("unit-test", TEST_SMT, "Proven", Some(binary)).unwrap();

        assert!(SovereignAttestor::verify(&attestation, Some(binary), &trusted_public_key));
        assert!(!SovereignAttestor::verify(&attestation, Some(b"tampered artifact"), &trusted_public_key));
        assert!(
            !SovereignAttestor::verify(&attestation, None, &trusted_public_key),
            "an attestation that commits to a binary must not silently skip binary checking"
        );
    }

    #[test]
    fn v2_attestation_rejects_binary_presence_substitution() {
        let attestor = SovereignAttestor::new();
        let trusted_public_key = attestor.public_key().to_vec();
        let attestation = attestor.attest("unit-test", TEST_SMT, "Proven", Some(b"artifact")).unwrap();

        let mut changed = attestation.clone();
        changed.binary_hash = None;
        assert!(
            !SovereignAttestor::verify(&changed, None, &trusted_public_key),
            "removing the optional hash must invalidate the signature transcript"
        );

        let mut changed = attestor.attest("unit-test", TEST_SMT, "Proven", None).unwrap();
        assert!(
            !SovereignAttestor::verify(&changed, Some(b"artifact"), &trusted_public_key),
            "a supplied artifact must not be treated as verified if the attestation did not commit to it"
        );
        changed.verdict.push_str(" changed");
        assert!(!SovereignAttestor::verify(&changed, None, &trusted_public_key));
    }

    #[test]
    fn v2_transcript_binds_label_and_verdict() {
        let attestor = SovereignAttestor::new();
        let trusted_public_key = attestor.public_key().to_vec();
        let attestation = attestor.attest("unit-test", TEST_SMT, "Proven", None).unwrap();

        let mut changed_label = attestation.clone();
        changed_label.label.push_str("-other");
        assert!(!SovereignAttestor::verify(&changed_label, None, &trusted_public_key));

        let mut changed_verdict = attestation;
        changed_verdict.verdict = "Unknown".to_string();
        assert!(!SovereignAttestor::verify(&changed_verdict, None, &trusted_public_key));
    }

    #[test]
    fn self_signed_attestation_is_rejected_by_an_independent_trust_anchor() {
        let trusted_attestor = SovereignAttestor::new();
        let trusted_public_key = trusted_attestor.public_key().to_vec();
        let attacker = SovereignAttestor::new();
        let attacker_public_key = attacker.public_key().to_vec();
        let forged = attacker.attest("unit-test", TEST_SMT, "Proven", None).unwrap();

        assert!(
            !SovereignAttestor::verify(&forged, None, &trusted_public_key),
            "a valid signature from a self-supplied untrusted key must not authenticate the claim"
        );
        assert!(
            SovereignAttestor::verify(&forged, None, &attacker_public_key),
            "the same signature should verify only when the caller explicitly trusts that signer"
        );
    }

    #[test]
    fn legacy_v1_attestation_is_rejected() {
        let attestor = SovereignAttestor::new();
        let trusted_public_key = attestor.public_key().to_vec();
        let mut attestation = attestor.attest("unit-test", TEST_SMT, "Proven", None).unwrap();
        attestation.transcript_version = 1;
        assert!(!SovereignAttestor::verify(&attestation, None, &trusted_public_key));
    }

    #[test]
    fn legacy_v1_transcript_collision_is_reproduced_and_rejected() {
        // Both records produce the exact same v1 concatenated message, despite
        // having different labels, hashes, binary claims, and verdicts.
        // The repeated bytes make the boundary shift deterministic and keep
        // both variable string fields valid UTF-8.
        let attestor = SovereignAttestor::new();
        let trusted_public_key = attestor.public_key().to_vec();
        let original = ProofAttestation {
            label: "A".to_string(),
            smtlib2_hash: [b'B'; 32],
            binary_hash: Some([b'C'; 32]),
            verdict: "D".to_string(),
            signature: Vec::new(),
            public_key: attestor.keys.public_key().to_vec(),
            transcript_version: 1,
        };
        let mut changed = ProofAttestation {
            label: "AB".to_string(),
            smtlib2_hash: {
                let mut bytes = [b'B'; 32];
                bytes[31] = b'C';
                bytes
            },
            binary_hash: Some({
                let mut bytes = [b'C'; 32];
                bytes[31] = b'D';
                bytes
            }),
            verdict: String::new(),
            signature: Vec::new(),
            public_key: attestor.keys.public_key().to_vec(),
            transcript_version: 1,
        };

        fn legacy_message(attestation: &ProofAttestation) -> Vec<u8> {
            let mut message = Vec::new();
            message.extend_from_slice(attestation.label.as_bytes());
            message.extend_from_slice(&attestation.smtlib2_hash);
            if let Some(hash) = attestation.binary_hash {
                message.extend_from_slice(&hash);
            }
            message.extend_from_slice(attestation.verdict.as_bytes());
            message
        }

        let original_message = legacy_message(&original);
        let changed_message = legacy_message(&changed);
        assert_eq!(original_message, changed_message);

        let signature = attestor.keys.sign(&original_message).unwrap();
        changed.signature = signature.clone();
        let legacy_public_key = attestor.keys.public_key().to_vec();

        // This is the exact condition that made the old verifier accept both
        // records: the bytes being verified are identical and actual_binary=None
        // skips the optional binary comparison.
        assert!(verify_signature(&changed_message, &signature, &legacy_public_key).unwrap());

        // The new verifier rejects v1 format rather than pretending the legacy
        // ambiguous transcript can be interpreted unambiguously.
        assert!(!SovereignAttestor::verify(&changed, None, &trusted_public_key));
    }

    #[test]
    fn v2_transcript_binds_version_and_option_tag() {
        let attestor = SovereignAttestor::new();
        let attestation = attestor.attest("unit-test", TEST_SMT, "Proven", Some(b"artifact")).unwrap();

        let original = signing_message_v2(&attestation);
        let mut changed = attestation.clone();
        changed.binary_hash = None;
        assert_ne!(original, signing_message_v2(&changed));

        changed = attestation;
        changed.transcript_version = 3;
        assert_ne!(original, signing_message_v2(&changed));
    }
}
