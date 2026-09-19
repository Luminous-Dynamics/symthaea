// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ED25519ADAPTER-516: concrete strict Ed25519 verifier for the
//! crypto-agile presentation-attestation interface defined by QUAL-ATTESTERAUTH-515.
//!
//! This module is enabled only with Soma's existing `pairing` feature. It does
//! not alter the attestation message semantics: `symthaea-core` still defines
//! the exact message identity, key binding, role, epoch, and key-state rules.
//! This adapter contributes only one proposition: signatures under the exact
//! suite ID below are verified with `ed25519-dalek`'s strict verification path.

use ed25519_dalek::{Signature, VerifyingKey};
use symthaea_core::assurance_attester_auth::{AttestationVerifier, SignatureSuiteId};

const SUITE_DOMAIN: &[u8] =
    b"symthaea.signature-suite.v1/ed25519-dalek-2.1.0/verify-strict\0";

/// Returns the exact signature-suite identity implemented by this adapter.
pub fn ed25519_strict_suite_id() -> SignatureSuiteId {
    SignatureSuiteId(*blake3::hash(SUITE_DOMAIN).as_bytes())
}

/// Dependency-light adapter object for QUAL-ATTESTERAUTH-515.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ed25519StrictVerifier;

impl AttestationVerifier for Ed25519StrictVerifier {
    fn verify(
        &self,
        suite_id: SignatureSuiteId,
        public_key: &[u8],
        message: &[u8; 32],
        signature: &[u8],
    ) -> bool {
        if suite_id != ed25519_strict_suite_id() {
            return false;
        }

        let Ok(public_key_bytes): Result<[u8; 32], _> = public_key.try_into() else {
            return false;
        };
        let Ok(signature_bytes): Result<[u8; 64], _> = signature.try_into() else {
            return false;
        };
        let Ok(verifying_key) = VerifyingKey::from_bytes(&public_key_bytes) else {
            return false;
        };
        let signature = Signature::from_bytes(&signature_bytes);

        verifying_key.verify_strict(message, &signature).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn message(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    #[test]
    fn exact_signature_verifies() {
        let signing = SigningKey::from_bytes(&[7; 32]);
        let verifying = signing.verifying_key();
        let message = message(11);
        let signature = signing.sign(&message);

        assert!(Ed25519StrictVerifier.verify(
            ed25519_strict_suite_id(),
            verifying.as_bytes(),
            &message,
            &signature.to_bytes(),
        ));
    }

    #[test]
    fn modified_message_is_rejected() {
        let signing = SigningKey::from_bytes(&[8; 32]);
        let verifying = signing.verifying_key();
        let original = message(12);
        let changed = message(13);
        let signature = signing.sign(&original);

        assert!(!Ed25519StrictVerifier.verify(
            ed25519_strict_suite_id(),
            verifying.as_bytes(),
            &changed,
            &signature.to_bytes(),
        ));
    }

    #[test]
    fn modified_signature_is_rejected() {
        let signing = SigningKey::from_bytes(&[9; 32]);
        let verifying = signing.verifying_key();
        let message = message(14);
        let mut signature = signing.sign(&message).to_bytes();
        signature[0] ^= 1;

        assert!(!Ed25519StrictVerifier.verify(
            ed25519_strict_suite_id(),
            verifying.as_bytes(),
            &message,
            &signature,
        ));
    }

    #[test]
    fn wrong_suite_is_rejected_before_crypto() {
        let signing = SigningKey::from_bytes(&[10; 32]);
        let verifying = signing.verifying_key();
        let message = message(15);
        let signature = signing.sign(&message);

        assert!(!Ed25519StrictVerifier.verify(
            SignatureSuiteId([0xAA; 32]),
            verifying.as_bytes(),
            &message,
            &signature.to_bytes(),
        ));
    }

    #[test]
    fn malformed_lengths_are_rejected() {
        let message = message(16);
        assert!(!Ed25519StrictVerifier.verify(
            ed25519_strict_suite_id(),
            &[1; 31],
            &message,
            &[2; 64],
        ));
        assert!(!Ed25519StrictVerifier.verify(
            ed25519_strict_suite_id(),
            &[1; 32],
            &message,
            &[2; 63],
        ));
    }
}
