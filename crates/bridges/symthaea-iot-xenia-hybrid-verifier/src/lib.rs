// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed cryptographic verifier for Xenia authenticated-payload receipts.
//!
//! Xenia signs the same domain-separated 32-byte receipt-body digest with both
//! Ed25519 and ML-DSA-65. Symthaea deliberately verifies those signatures with a
//! concrete, non-configurable relying-party implementation:
//!
//! - Ed25519 through the workspace-pinned `ed25519-dalek` implementation; and
//! - ML-DSA-65 through `fips204`, independently from Xenia's RustCrypto `ml-dsa`
//!   signer/verifier implementation.
//!
//! This crate owns no trust registry, algorithm negotiation, policy, clock, device
//! state, or HAL handle. It only implements the exact signature-provider boundary
//! required by `symthaea-iot-transport-receipt`. Both signature checks remain
//! mandatory in that caller; there is no classical-only or provider-fallback path.

#![deny(unsafe_code)]

use ed25519_dalek::{Signature, Verifier as Ed25519Verifier, VerifyingKey};
use fips204::{
    ml_dsa_65,
    traits::{SerDes, Verifier as MlDsaVerifier},
};
use symthaea_iot_transport_receipt::{
    HybridReceiptSignatureVerifier, TransportReceiptError, TransportTrustRegistry,
    VerifiedTransportEnvelope, XENIA_ED25519_SIGNATURE_LEN, XENIA_ML_DSA_65_PUBLIC_KEY_LEN,
    XENIA_ML_DSA_65_SIGNATURE_LEN, verify_xenia_transport_receipt,
};

const _: () = assert!(XENIA_ED25519_SIGNATURE_LEN == 64);
const _: () = assert!(XENIA_ML_DSA_65_PUBLIC_KEY_LEN == ml_dsa_65::PK_LEN);
const _: () = assert!(XENIA_ML_DSA_65_SIGNATURE_LEN == ml_dsa_65::SIG_LEN);

/// Fixed production verifier for Xenia's mandatory hybrid receipt signature suite.
///
/// This zero-sized type intentionally has no runtime configuration or algorithm
/// selection surface. Trust/key lifecycle belongs to `TransportTrustRegistry`; this
/// type only answers whether the supplied exact key verifies the supplied exact digest.
#[derive(Debug, Clone, Copy, Default)]
pub struct XeniaHybridReceiptVerifier;

impl HybridReceiptSignatureVerifier for XeniaHybridReceiptVerifier {
    fn verify_ed25519(
        &self,
        public_key: &[u8; 32],
        digest: &[u8; 32],
        signature: &[u8; XENIA_ED25519_SIGNATURE_LEN],
    ) -> bool {
        let Ok(verifying_key) = VerifyingKey::from_bytes(public_key) else {
            return false;
        };
        let signature = Signature::from_bytes(signature);
        verifying_key.verify(digest, &signature).is_ok()
    }

    fn verify_ml_dsa_65(
        &self,
        public_key: &[u8],
        digest: &[u8; 32],
        signature: &[u8; XENIA_ML_DSA_65_SIGNATURE_LEN],
    ) -> bool {
        let Ok(public_key_bytes) = <[u8; ml_dsa_65::PK_LEN]>::try_from(public_key) else {
            return false;
        };
        let Ok(verifying_key) = ml_dsa_65::PublicKey::try_from_bytes(public_key_bytes) else {
            return false;
        };
        let Ok(signature_bytes) = <[u8; ml_dsa_65::SIG_LEN]>::try_from(signature.as_slice()) else {
            return false;
        };

        // Xenia signs its already-domain-separated receipt-body digest as the ML-DSA
        // message with no ML-DSA context. Do not hash or domain-separate it again here.
        verifying_key.verify(digest, &signature_bytes, &[])
    }
}

/// Guard-facing Xenia verification path with no caller-selectable crypto provider.
///
/// A privileged actuation guard should call this function rather than the lower-level
/// generic `verify_xenia_transport_receipt`. The concrete hybrid verifier is selected
/// here, inside the reviewed TCB, so unprivileged request data and ordinary call sites
/// cannot substitute a permissive verifier implementation.
pub fn verify_xenia_physical_effect_receipt(
    registry: &TransportTrustRegistry,
    raw_receipt: &[u8],
    raw_payload: &[u8],
    now_unix_ms: u64,
) -> Result<VerifiedTransportEnvelope, TransportReceiptError> {
    verify_xenia_transport_receipt(
        registry,
        raw_receipt,
        raw_payload,
        now_unix_ms,
        &XeniaHybridReceiptVerifier,
    )
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::{Signer as Ed25519Signer, SigningKey};
    use fips204::traits::{KeyGen, SerDes, Signer as MlDsaSigner};

    use super::*;

    fn digest() -> [u8; 32] {
        [0xA5; 32]
    }

    #[test]
    fn ed25519_verifies_exact_digest_and_rejects_mutation() {
        let signing_key = SigningKey::from_bytes(&[0x66; 32]);
        let public_key = signing_key.verifying_key().to_bytes();
        let digest = digest();
        let signature = signing_key.sign(&digest).to_bytes();
        let verifier = XeniaHybridReceiptVerifier;

        assert!(verifier.verify_ed25519(&public_key, &digest, &signature));

        let mut altered_digest = digest;
        altered_digest[0] ^= 1;
        assert!(!verifier.verify_ed25519(&public_key, &altered_digest, &signature));

        let mut altered_signature = signature;
        altered_signature[0] ^= 1;
        assert!(!verifier.verify_ed25519(&public_key, &digest, &altered_signature));
    }

    #[test]
    fn independent_ml_dsa_verifier_accepts_exact_fips204_signature() {
        let (public_key, private_key) = ml_dsa_65::KG::keygen_from_seed(&[0x77; 32]);
        let digest = digest();
        let signature = private_key
            .try_sign_with_seed(&[0x88; 32], &digest, &[])
            .expect("deterministic ML-DSA-65 test signature");
        let public_key_bytes = public_key.into_bytes();
        let verifier = XeniaHybridReceiptVerifier;

        assert!(verifier.verify_ml_dsa_65(&public_key_bytes, &digest, &signature));

        let mut altered_digest = digest;
        altered_digest[0] ^= 1;
        assert!(!verifier.verify_ml_dsa_65(
            &public_key_bytes,
            &altered_digest,
            &signature
        ));

        let mut altered_signature = signature;
        altered_signature[0] ^= 1;
        assert!(!verifier.verify_ml_dsa_65(
            &public_key_bytes,
            &digest,
            &altered_signature
        ));
    }

    #[test]
    fn malformed_ml_dsa_public_key_fails_without_fallback() {
        let verifier = XeniaHybridReceiptVerifier;
        assert!(!verifier.verify_ml_dsa_65(&[0x42; 64], &digest(), &[0x55; 3_309]));
        assert!(!verifier.verify_ml_dsa_65(
            &[0u8; XENIA_ML_DSA_65_PUBLIC_KEY_LEN],
            &digest(),
            &[0x55; XENIA_ML_DSA_65_SIGNATURE_LEN]
        ));
    }
}
