// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Concrete RFC 8032 Ed25519 verification profile for hardware-interlock evidence.
//!
//! The generic interlock-trust layer intentionally keeps hardware signature formats
//! behind [`InterlockControllerEvidenceVerifier`]. A privileged actuation guard should
//! not accept an arbitrary implementation of that trait from an untrusted caller.
//! This crate provides one fixed, reviewable profile using the repository's already
//! pinned `ed25519-dalek` dependency.
//!
//! A controller using this profile signs exactly:
//!
//! `b"symthaea-iot-interlock-ed25519-v1\0" || physical_interlock_report_digest`
//!
//! where the report digest is already domain-separated by `symthaea-iot-final-gate`.
//! The extra profile domain prevents a valid signature over the same 32 bytes in a
//! different protocol from being silently reinterpreted as interlock evidence.

#![deny(unsafe_code)]

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use symthaea_authority::Digest32;
use symthaea_iot_interlock_trust::InterlockControllerEvidenceVerifier;

/// Exact algorithm/profile label accepted by this verifier.
pub const INTERLOCK_ED25519_ALGORITHM: &str = "ed25519-rfc8032";
/// RFC 8032 Ed25519 public-key length.
pub const INTERLOCK_ED25519_PUBLIC_KEY_LEN: usize = 32;
/// RFC 8032 Ed25519 signature length.
pub const INTERLOCK_ED25519_SIGNATURE_LEN: usize = 64;
/// Domain separator for the message authenticated by hardware controllers.
pub const INTERLOCK_ED25519_MESSAGE_DOMAIN: &[u8] = b"symthaea-iot-interlock-ed25519-v1\0";

/// Build the exact bytes a controller must sign for one physical-interlock report.
///
/// This helper is public so firmware/PLC adapters can implement the same canonical
/// signing contract without duplicating domain construction.
pub fn interlock_ed25519_signing_message(report_digest: Digest32) -> Vec<u8> {
    let Digest32(digest) = report_digest;
    let mut message = Vec::with_capacity(INTERLOCK_ED25519_MESSAGE_DOMAIN.len() + digest.len());
    message.extend_from_slice(INTERLOCK_ED25519_MESSAGE_DOMAIN);
    message.extend_from_slice(&digest);
    message
}

/// Fixed RFC 8032 verifier suitable for a privileged actuation guard.
///
/// This type contains no mutable trust configuration. Controller identity, key
/// lifecycle and exact public-key bytes remain governed by the anti-rollback
/// `InterlockTrustRegistry`; this verifier only performs the concrete signature check.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ed25519Rfc8032InterlockVerifier;

impl InterlockControllerEvidenceVerifier for Ed25519Rfc8032InterlockVerifier {
    fn verify_controller_evidence(
        &self,
        _controller_id: &str,
        _key_id: &str,
        algorithm: &str,
        public_key: &[u8],
        report_digest: Digest32,
        raw_evidence: &[u8],
    ) -> bool {
        if algorithm != INTERLOCK_ED25519_ALGORITHM
            || public_key.len() != INTERLOCK_ED25519_PUBLIC_KEY_LEN
            || raw_evidence.len() != INTERLOCK_ED25519_SIGNATURE_LEN
        {
            return false;
        }

        let Ok(public_key_bytes) = <&[u8; INTERLOCK_ED25519_PUBLIC_KEY_LEN]>::try_from(public_key)
        else {
            return false;
        };
        let Ok(verifying_key) = VerifyingKey::from_bytes(public_key_bytes) else {
            return false;
        };
        let Ok(signature) = Signature::from_slice(raw_evidence) else {
            return false;
        };

        let message = interlock_ed25519_signing_message(report_digest);
        verifying_key.verify(&message, &signature).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn signature_for(signing_key: &SigningKey, report_digest: Digest32) -> Signature {
        signing_key.sign(&interlock_ed25519_signing_message(report_digest))
    }

    #[test]
    fn exact_current_profile_verifies() {
        let signing_key = SigningKey::from_bytes(&[0x31; 32]);
        let report_digest = digest(7);
        let signature = signature_for(&signing_key, report_digest);
        assert!(Ed25519Rfc8032InterlockVerifier.verify_controller_evidence(
            "safety-plc:field-a",
            "plc-key-1",
            INTERLOCK_ED25519_ALGORITHM,
            signing_key.verifying_key().as_bytes(),
            report_digest,
            &signature.to_bytes(),
        ));
    }

    #[test]
    fn altered_report_digest_is_rejected() {
        let signing_key = SigningKey::from_bytes(&[0x31; 32]);
        let signature = signature_for(&signing_key, digest(7));
        assert!(!Ed25519Rfc8032InterlockVerifier.verify_controller_evidence(
            "safety-plc:field-a",
            "plc-key-1",
            INTERLOCK_ED25519_ALGORITHM,
            signing_key.verifying_key().as_bytes(),
            digest(8),
            &signature.to_bytes(),
        ));
    }

    #[test]
    fn wrong_public_key_is_rejected() {
        let signing_key = SigningKey::from_bytes(&[0x31; 32]);
        let other_key = SigningKey::from_bytes(&[0x32; 32]);
        let report_digest = digest(7);
        let signature = signature_for(&signing_key, report_digest);
        assert!(!Ed25519Rfc8032InterlockVerifier.verify_controller_evidence(
            "safety-plc:field-a",
            "plc-key-1",
            INTERLOCK_ED25519_ALGORITHM,
            other_key.verifying_key().as_bytes(),
            report_digest,
            &signature.to_bytes(),
        ));
    }

    #[test]
    fn wrong_algorithm_is_rejected_before_signature_acceptance() {
        let signing_key = SigningKey::from_bytes(&[0x31; 32]);
        let report_digest = digest(7);
        let signature = signature_for(&signing_key, report_digest);
        assert!(!Ed25519Rfc8032InterlockVerifier.verify_controller_evidence(
            "safety-plc:field-a",
            "plc-key-1",
            "ed25519",
            signing_key.verifying_key().as_bytes(),
            report_digest,
            &signature.to_bytes(),
        ));
    }

    #[test]
    fn malformed_key_or_signature_length_is_rejected() {
        let verifier = Ed25519Rfc8032InterlockVerifier;
        assert!(!verifier.verify_controller_evidence(
            "safety-plc:field-a",
            "plc-key-1",
            INTERLOCK_ED25519_ALGORITHM,
            &[0x44; 31],
            digest(7),
            &[0x55; 64],
        ));
        assert!(!verifier.verify_controller_evidence(
            "safety-plc:field-a",
            "plc-key-1",
            INTERLOCK_ED25519_ALGORITHM,
            &[0x44; 32],
            digest(7),
            &[0x55; 63],
        ));
    }

    #[test]
    fn profile_domain_is_part_of_signature_message() {
        let signing_key = SigningKey::from_bytes(&[0x31; 32]);
        let report_digest = digest(7);
        let Digest32(raw_digest) = report_digest;
        let signature: Signature = signing_key.sign(&raw_digest);
        assert!(!Ed25519Rfc8032InterlockVerifier.verify_controller_evidence(
            "safety-plc:field-a",
            "plc-key-1",
            INTERLOCK_ED25519_ALGORITHM,
            signing_key.verifying_key().as_bytes(),
            report_digest,
            &signature.to_bytes(),
        ));
    }
}
