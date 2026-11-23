// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pinned Ed25519 trust profiles for MACE evidence-qualification receipts.
//!
//! The trust profile is a directly pinned leaf-key set. Its expected SHA-256
//! must be provisioned outside the candidate backend and outside the memory
//! store's rollback domain. A digest copied from the profile being verified is
//! not a trust bootstrap.
//!
//! Qualification signatures cover the 64-byte lowercase-hex
//! signing_payload_sha256() value. That digest is itself domain-separated and
//! binds the complete typed qualification receipt. This is a versioned MACE
//! signature profile, not a DSSE envelope. A later wire adapter may wrap the
//! same canonical receipt payload in DSSE/in-toto without changing semantics.

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::evidence_qualification_gate::{
    EvidenceQualificationRejectV1,
    EvidenceQualificationSignatureVerifierV1,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum QualificationKeyStatusV1 {
    Active,
    Revoked,
}

impl QualificationKeyStatusV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Active => 1,
            Self::Revoked => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualificationTrustKeyV1 {
    /// Derived as SHA-256 over the versioned key-id preimage.
    pub key_id: String,
    pub public_key: [u8; 32],
    pub status: QualificationKeyStatusV1,
    pub not_before_unix: u64,
    pub not_after_unix: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualificationTrustProfileV1 {
    pub schema_version: u32,
    pub verifier_id: String,
    pub profile_version: u64,
    /// Keys must be strictly ascending by key_id. Reordering is rejected.
    pub keys: Vec<QualificationTrustKeyV1>,
}

impl QualificationTrustProfileV1 {
    /// Deterministic profile digest used for out-of-band pinning.
    /// Commits to verifier identity, profile version, every key, status and
    /// validity interval.
    pub fn profile_sha256(&self) -> Result<String, EvidenceQualificationRejectV1> {
        validate_profile(self)?;
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:qualification-trust-profile:v1\0");
        hash_field(&mut hasher, &self.verifier_id);
        hasher.update(self.schema_version.to_be_bytes());
        hasher.update(self.profile_version.to_be_bytes());
        hasher.update((self.keys.len() as u64).to_be_bytes());
        for key in &self.keys {
            hash_field(&mut hasher, &key.key_id);
            hasher.update(key.public_key);
            hasher.update([key.status.tag()]);
            hasher.update(key.not_before_unix.to_be_bytes());
            hasher.update(key.not_after_unix.to_be_bytes());
        }
        Ok(hex_digest(&hasher.finalize()))
    }
}

/// Compute the stable key identifier for a raw Ed25519 public key.
pub fn qualification_key_id_v1(public_key: &[u8; 32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:qualification-key-id:v1\0");
    hasher.update(public_key);
    hex_digest(&hasher.finalize())
}

#[derive(Clone, Debug)]
struct PinnedQualificationKeyV1 {
    verifying_key: VerifyingKey,
    status: QualificationKeyStatusV1,
    not_before_unix: u64,
    not_after_unix: u64,
}

/// Immutable verifier constructed only from a profile whose exact digest
/// matches an out-of-band provisioned pin.
#[derive(Clone, Debug)]
pub struct PinnedEd25519QualificationVerifierV1 {
    profile_sha256: String,
    verifier_id: String,
    profile_version: u64,
    keys: BTreeMap<String, PinnedQualificationKeyV1>,
}

impl PinnedEd25519QualificationVerifierV1 {
    pub fn from_pinned_profile(
        profile: &QualificationTrustProfileV1,
        expected_profile_sha256: &str,
    ) -> Result<Self, EvidenceQualificationRejectV1> {
        if !valid_sha256(expected_profile_sha256) {
            return Err(EvidenceQualificationRejectV1::TrustProfileInvalid);
        }
        let actual = profile.profile_sha256()?;
        if actual != expected_profile_sha256 {
            return Err(EvidenceQualificationRejectV1::TrustProfileDigestMismatch);
        }

        let mut keys = BTreeMap::new();
        for key in &profile.keys {
            let verifying_key = VerifyingKey::from_bytes(&key.public_key)
                .map_err(|_| EvidenceQualificationRejectV1::InvalidPublicKey)?;
            if keys.insert(
                key.key_id.clone(),
                PinnedQualificationKeyV1 {
                    verifying_key,
                    status: key.status,
                    not_before_unix: key.not_before_unix,
                    not_after_unix: key.not_after_unix,
                },
            ).is_some() {
                return Err(EvidenceQualificationRejectV1::TrustProfileInvalid);
            }
        }

        Ok(Self {
            profile_sha256: actual,
            verifier_id: profile.verifier_id.clone(),
            profile_version: profile.profile_version,
            keys,
        })
    }

    pub fn profile_sha256(&self) -> &str { &self.profile_sha256 }
    pub fn verifier_id(&self) -> &str { &self.verifier_id }
    pub fn profile_version(&self) -> u64 { self.profile_version }
}

impl EvidenceQualificationSignatureVerifierV1 for PinnedEd25519QualificationVerifierV1 {
    fn verify_signature(
        &self,
        verifier_profile_sha256: &str,
        key_id: &str,
        payload_sha256: &str,
        signature: &[u8],
        issued_at_unix: u64,
        now_unix: u64,
    ) -> Result<(), EvidenceQualificationRejectV1> {
        if verifier_profile_sha256 != self.profile_sha256 {
            return Err(EvidenceQualificationRejectV1::TrustProfileDigestMismatch);
        }
        if !valid_sha256(payload_sha256) {
            return Err(EvidenceQualificationRejectV1::InvalidDigest);
        }
        let key = self.keys.get(key_id)
            .ok_or(EvidenceQualificationRejectV1::UnknownKey)?;

        if key.status == QualificationKeyStatusV1::Revoked {
            return Err(EvidenceQualificationRejectV1::RevokedKey);
        }
        if issued_at_unix < key.not_before_unix || now_unix < key.not_before_unix {
            return Err(EvidenceQualificationRejectV1::KeyNotYetValid);
        }
        if issued_at_unix >= key.not_after_unix || now_unix >= key.not_after_unix {
            return Err(EvidenceQualificationRejectV1::KeyExpired);
        }
        if signature.len() != 64 {
            return Err(EvidenceQualificationRejectV1::InvalidSignatureEncoding);
        }
        let signature_bytes: &[u8; 64] = signature
            .try_into()
            .map_err(|_| EvidenceQualificationRejectV1::InvalidSignatureEncoding)?;
        let signature = Signature::from_bytes(signature_bytes);
        key.verifying_key
            .verify_strict(payload_sha256.as_bytes(), &signature)
            .map_err(|_| EvidenceQualificationRejectV1::SignatureRejected)
    }
}

fn validate_profile(
    profile: &QualificationTrustProfileV1,
) -> Result<(), EvidenceQualificationRejectV1> {
    if profile.schema_version != 1
        || profile.profile_version == 0
        || !valid_atom(&profile.verifier_id)
        || profile.keys.is_empty()
        || profile.keys.len() > 1_024
    {
        return Err(EvidenceQualificationRejectV1::TrustProfileInvalid);
    }

    let mut seen_key_ids = BTreeSet::new();
    let mut seen_public_keys = BTreeSet::new();
    for (index, key) in profile.keys.iter().enumerate() {
        if !valid_sha256(&key.key_id)
            || key.key_id != qualification_key_id_v1(&key.public_key)
            || key.not_before_unix >= key.not_after_unix
            || VerifyingKey::from_bytes(&key.public_key).is_err()
        {
            return Err(EvidenceQualificationRejectV1::TrustProfileInvalid);
        }
        if !seen_key_ids.insert(key.key_id.as_str())
            || !seen_public_keys.insert(key.public_key)
        {
            return Err(EvidenceQualificationRejectV1::TrustProfileInvalid);
        }
        if index > 0 && profile.keys[index - 1].key_id >= key.key_id {
            return Err(EvidenceQualificationRejectV1::TrustProfileInvalid);
        }
    }
    Ok(())
}

fn valid_atom(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn signing_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn profile_with_key(
        signing_key: &SigningKey,
        status: QualificationKeyStatusV1,
        not_before_unix: u64,
        not_after_unix: u64,
    ) -> QualificationTrustProfileV1 {
        let public_key = signing_key.verifying_key().to_bytes();
        let key = QualificationTrustKeyV1 {
            key_id: qualification_key_id_v1(&public_key),
            public_key,
            status,
            not_before_unix,
            not_after_unix,
        };
        QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: "mace-test-verifier".to_owned(),
            profile_version: 1,
            keys: vec![key],
        }
    }

    fn signed_payload(signing_key: &SigningKey, payload_sha256: &str) -> Vec<u8> {
        signing_key.sign(payload_sha256.as_bytes()).to_bytes().to_vec()
    }

    #[test]
    fn pinned_active_ed25519_key_verifies_exact_qualification_digest() {
        let signer = signing_key(7);
        let profile = profile_with_key(&signer, QualificationKeyStatusV1::Active, 10, 1_000);
        let pin = profile.profile_sha256().expect("profile digest");
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile, &pin,
        ).expect("pinned trust");
        let payload = "a".repeat(64);
        let signature = signed_payload(&signer, &payload);
        assert!(verifier.verify_signature(
            &pin, &profile.keys[0].key_id, &payload, &signature, 100, 101,
        ).is_ok());
    }

    #[test]
    fn profile_must_match_independently_provisioned_digest() {
        let signer = signing_key(8);
        let profile = profile_with_key(&signer, QualificationKeyStatusV1::Active, 10, 1_000);
        assert_eq!(
            PinnedEd25519QualificationVerifierV1::from_pinned_profile(
                &profile, &"0".repeat(64),
            ).err(),
            Some(EvidenceQualificationRejectV1::TrustProfileDigestMismatch),
        );
    }

    #[test]
    fn unknown_key_is_rejected() {
        let signer = signing_key(9);
        let profile = profile_with_key(&signer, QualificationKeyStatusV1::Active, 10, 1_000);
        let pin = profile.profile_sha256().expect("profile digest");
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile, &pin,
        ).expect("pinned trust");
        assert_eq!(
            verifier.verify_signature(
                &pin, &"f".repeat(64), &"a".repeat(64), &[0; 64], 100, 101,
            ),
            Err(EvidenceQualificationRejectV1::UnknownKey),
        );
    }

    #[test]
    fn revoked_key_is_rejected_even_for_a_previously_valid_receipt() {
        let signer = signing_key(10);
        let profile = profile_with_key(&signer, QualificationKeyStatusV1::Revoked, 10, 1_000);
        let pin = profile.profile_sha256().expect("profile digest");
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile, &pin,
        ).expect("pinned trust");
        let payload = "a".repeat(64);
        let signature = signed_payload(&signer, &payload);
        assert_eq!(
            verifier.verify_signature(
                &pin, &profile.keys[0].key_id, &payload, &signature, 100, 200,
            ),
            Err(EvidenceQualificationRejectV1::RevokedKey),
        );
    }

    #[test]
    fn key_not_yet_valid_at_receipt_issuance_is_rejected() {
        let signer = signing_key(11);
        let profile = profile_with_key(&signer, QualificationKeyStatusV1::Active, 200, 1_000);
        let pin = profile.profile_sha256().expect("profile digest");
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile, &pin,
        ).expect("pinned trust");
        let payload = "a".repeat(64);
        let signature = signed_payload(&signer, &payload);
        assert_eq!(
            verifier.verify_signature(
                &pin, &profile.keys[0].key_id, &payload, &signature, 100, 250,
            ),
            Err(EvidenceQualificationRejectV1::KeyNotYetValid),
        );
    }

    #[test]
    fn expired_key_is_rejected_at_consumption_time() {
        let signer = signing_key(12);
        let profile = profile_with_key(&signer, QualificationKeyStatusV1::Active, 10, 200);
        let pin = profile.profile_sha256().expect("profile digest");
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile, &pin,
        ).expect("pinned trust");
        let payload = "a".repeat(64);
        let signature = signed_payload(&signer, &payload);
        assert_eq!(
            verifier.verify_signature(
                &pin, &profile.keys[0].key_id, &payload, &signature, 100, 200,
            ),
            Err(EvidenceQualificationRejectV1::KeyExpired),
        );
    }

    #[test]
    fn profile_requires_canonical_key_order() {
        let signer_a = signing_key(14);
        let signer_b = signing_key(15);
        let make_key = |signer: &SigningKey| {
            let public_key = signer.verifying_key().to_bytes();
            QualificationTrustKeyV1 {
                key_id: qualification_key_id_v1(&public_key),
                public_key,
                status: QualificationKeyStatusV1::Active,
                not_before_unix: 10,
                not_after_unix: 1_000,
            }
        };
        let mut keys = vec![make_key(&signer_a), make_key(&signer_b)];
        keys.sort_by(|left, right| left.key_id.cmp(&right.key_id));
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: "mace-test-verifier".to_owned(),
            profile_version: 2,
            keys: keys.into_iter().rev().collect(),
        };
        assert_eq!(
            profile.profile_sha256(),
            Err(EvidenceQualificationRejectV1::TrustProfileInvalid)
        );
    }

    #[test]
    fn profile_rejects_duplicate_key_and_public_key_aliases() {
        let signer = signing_key(16);
        let key = {
            let public_key = signer.verifying_key().to_bytes();
            QualificationTrustKeyV1 {
                key_id: qualification_key_id_v1(&public_key),
                public_key,
                status: QualificationKeyStatusV1::Active,
                not_before_unix: 10,
                not_after_unix: 1_000,
            }
        };
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: "mace-test-verifier".to_owned(),
            profile_version: 2,
            keys: vec![key.clone(), key],
        };
        assert_eq!(
            profile.profile_sha256(),
            Err(EvidenceQualificationRejectV1::TrustProfileInvalid)
        );
    }

    #[test]
    fn altered_signature_is_rejected() {
        let signer = signing_key(13);
        let profile = profile_with_key(&signer, QualificationKeyStatusV1::Active, 10, 1_000);
        let pin = profile.profile_sha256().expect("profile digest");
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile, &pin,
        ).expect("pinned trust");
        let payload = "a".repeat(64);
        let mut signature = signed_payload(&signer, &payload);
        signature[0] ^= 1;
        assert_eq!(
            verifier.verify_signature(
                &pin, &profile.keys[0].key_id, &payload, &signature, 100, 101,
            ),
            Err(EvidenceQualificationRejectV1::SignatureRejected),
        );
    }
}
