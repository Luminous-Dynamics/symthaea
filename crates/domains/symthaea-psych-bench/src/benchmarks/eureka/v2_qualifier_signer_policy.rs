// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure signer-policy mechanics for future EUREKA-002 V2 governance
//! admissions.
//!
//! This module contains no real governance key and performs no signature
//! verification. It only defines deterministic signer-set identity, threshold
//! rules, and append-only policy rotation. Human admission of the real genesis
//! policy and external OpenSSH signature verification remain separate tranches.

#![allow(dead_code)]

pub(super) const V2_QUALIFIER_SIGNER_POLICY_SCHEMA: &str =
    "EUREKA.002.V2.QUALIFIER_SIGNER_POLICY.v1";
pub(super) const V2_QUALIFIER_ADMISSION_SSH_NAMESPACE: &str =
    "eureka-v2-qualifier-admission@luminousdynamics.org";
pub(super) const V2_QUALIFIER_SIGNER_POLICY_SSH_NAMESPACE: &str =
    "eureka-v2-signer-policy@luminousdynamics.org";
pub(super) const V2_SIGNER_PUBLIC_KEY_MATERIAL_ENCODING: &str =
    "OPENSSH.PUBLIC_KEY_MATERIAL.keytype-space-base64.no-comment.no-newline.v1";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct V2GovernanceSigner {
    principal: String,
    key_type: String,
    public_key_material_sha256: [u8; 32],
}

impl V2GovernanceSigner {
    /// `public_key_material_sha256` is SHA-256 over the exact UTF-8 bytes:
    ///
    /// `<key_type><single ASCII space><OpenSSH base64 public-key blob>`
    ///
    /// with no comment and no trailing newline. The encoding revision is bound
    /// into every signer-policy commitment so a future representation change
    /// cannot silently reinterpret an old admitted key digest.
    pub(super) fn from_hex(
        principal: &str,
        key_type: &str,
        public_key_material_sha256: &str,
    ) -> Result<Self, V2QualifierSignerPolicyError> {
        validate_principal(principal)?;
        validate_key_type(key_type)?;
        Ok(Self {
            principal: principal.to_string(),
            key_type: key_type.to_string(),
            public_key_material_sha256: decode_hex32(public_key_material_sha256)?,
        })
    }

    pub(super) fn principal(&self) -> &str {
        &self.principal
    }

    pub(super) fn key_type(&self) -> &str {
        &self.key_type
    }

    pub(super) const fn public_key_material_sha256(&self) -> [u8; 32] {
        self.public_key_material_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2QualifierSignerPolicy {
    sequence: u64,
    predecessor_commitment: Option<[u8; 32]>,
    threshold: u16,
    signers: Vec<V2GovernanceSigner>,
    commitment: [u8; 32],
}

impl V2QualifierSignerPolicy {
    pub(super) fn genesis(
        claimed_sequence: u64,
        threshold: u16,
        signers: Vec<V2GovernanceSigner>,
    ) -> Result<Self, V2QualifierSignerPolicyError> {
        if claimed_sequence != 1 {
            return Err(V2QualifierSignerPolicyError::WrongSequence);
        }
        let signers = canonical_signers(signers)?;
        validate_threshold(threshold, signers.len())?;
        let commitment = signer_policy_commitment(claimed_sequence, None, threshold, &signers);
        Ok(Self {
            sequence: claimed_sequence,
            predecessor_commitment: None,
            threshold,
            signers,
            commitment,
        })
    }

    pub(super) fn rotate(
        predecessor: &Self,
        claimed_predecessor_commitment: [u8; 32],
        claimed_sequence: u64,
        threshold: u16,
        signers: Vec<V2GovernanceSigner>,
    ) -> Result<Self, V2QualifierSignerPolicyError> {
        if claimed_predecessor_commitment != predecessor.commitment {
            return Err(V2QualifierSignerPolicyError::WrongPredecessor);
        }
        let expected_sequence = predecessor
            .sequence
            .checked_add(1)
            .ok_or(V2QualifierSignerPolicyError::SequenceExhausted)?;
        if claimed_sequence != expected_sequence {
            return Err(V2QualifierSignerPolicyError::WrongSequence);
        }

        let signers = canonical_signers(signers)?;
        validate_threshold(threshold, signers.len())?;
        if threshold == predecessor.threshold && signers == predecessor.signers {
            return Err(V2QualifierSignerPolicyError::NoOpRotation);
        }

        let commitment = signer_policy_commitment(
            claimed_sequence,
            Some(claimed_predecessor_commitment),
            threshold,
            &signers,
        );
        Ok(Self {
            sequence: claimed_sequence,
            predecessor_commitment: Some(claimed_predecessor_commitment),
            threshold,
            signers,
            commitment,
        })
    }

    pub(super) const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(super) const fn predecessor_commitment(&self) -> Option<[u8; 32]> {
        self.predecessor_commitment
    }

    pub(super) const fn threshold(&self) -> u16 {
        self.threshold
    }

    pub(super) fn signers(&self) -> &[V2GovernanceSigner] {
        &self.signers
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualifierSignerPolicyError {
    InvalidHex,
    InvalidPrincipal,
    UnsupportedKeyType,
    EmptySignerSet,
    InvalidThreshold,
    DuplicatePrincipal,
    DuplicateKey,
    WrongSequence,
    SequenceExhausted,
    WrongPredecessor,
    NoOpRotation,
}

fn canonical_signers(
    mut signers: Vec<V2GovernanceSigner>,
) -> Result<Vec<V2GovernanceSigner>, V2QualifierSignerPolicyError> {
    if signers.is_empty() {
        return Err(V2QualifierSignerPolicyError::EmptySignerSet);
    }
    signers.sort();

    for pair in signers.windows(2) {
        if pair[0].principal == pair[1].principal {
            return Err(V2QualifierSignerPolicyError::DuplicatePrincipal);
        }
    }

    let mut keys: Vec<[u8; 32]> = signers
        .iter()
        .map(V2GovernanceSigner::public_key_material_sha256)
        .collect();
    keys.sort_unstable();
    if keys.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(V2QualifierSignerPolicyError::DuplicateKey);
    }
    Ok(signers)
}

fn validate_threshold(
    threshold: u16,
    signer_count: usize,
) -> Result<(), V2QualifierSignerPolicyError> {
    if threshold == 0 || usize::from(threshold) > signer_count {
        return Err(V2QualifierSignerPolicyError::InvalidThreshold);
    }
    Ok(())
}

fn validate_principal(value: &str) -> Result<(), V2QualifierSignerPolicyError> {
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'@' | b'.' | b'_' | b'-')
        })
    {
        return Err(V2QualifierSignerPolicyError::InvalidPrincipal);
    }
    Ok(())
}

fn validate_key_type(value: &str) -> Result<(), V2QualifierSignerPolicyError> {
    match value {
        "ssh-ed25519" | "sk-ssh-ed25519@openssh.com" => Ok(()),
        _ => Err(V2QualifierSignerPolicyError::UnsupportedKeyType),
    }
}

fn signer_policy_commitment(
    sequence: u64,
    predecessor_commitment: Option<[u8; 32]>,
    threshold: u16,
    signers: &[V2GovernanceSigner],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_QUALIFIER_SIGNER_POLICY_SCHEMA.as_bytes());
    encode_bytes(
        &mut bytes,
        V2_SIGNER_PUBLIC_KEY_MATERIAL_ENCODING.as_bytes(),
    );
    encode_bytes(
        &mut bytes,
        V2_QUALIFIER_ADMISSION_SSH_NAMESPACE.as_bytes(),
    );
    encode_bytes(
        &mut bytes,
        V2_QUALIFIER_SIGNER_POLICY_SSH_NAMESPACE.as_bytes(),
    );
    bytes.extend_from_slice(&sequence.to_le_bytes());
    match predecessor_commitment {
        Some(predecessor) => {
            bytes.push(1);
            bytes.extend_from_slice(&predecessor);
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&threshold.to_le_bytes());
    bytes.extend_from_slice(&(signers.len() as u64).to_le_bytes());
    for signer in signers {
        encode_bytes(&mut bytes, signer.principal.as_bytes());
        encode_bytes(&mut bytes, signer.key_type.as_bytes());
        bytes.extend_from_slice(&signer.public_key_material_sha256);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn decode_hex32(value: &str) -> Result<[u8; 32], V2QualifierSignerPolicyError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2QualifierSignerPolicyError::InvalidHex);
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(V2QualifierSignerPolicyError::InvalidHex)?;
        let low = hex_nibble(chunk[1]).ok_or(V2QualifierSignerPolicyError::InvalidHex)?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signer(principal: &str, key: char) -> V2GovernanceSigner {
        V2GovernanceSigner::from_hex(principal, "ssh-ed25519", &key.to_string().repeat(64))
            .unwrap()
    }

    #[test]
    fn genesis_is_order_independent_but_has_one_canonical_signer_order() {
        let first = V2QualifierSignerPolicy::genesis(
            1,
            2,
            vec![signer("beta@example.org", 'b'), signer("alpha@example.org", 'a')],
        )
        .unwrap();
        let second = V2QualifierSignerPolicy::genesis(
            1,
            2,
            vec![signer("alpha@example.org", 'a'), signer("beta@example.org", 'b')],
        )
        .unwrap();

        assert_eq!(first.commitment(), second.commitment());
        assert_eq!(first.signers()[0].principal(), "alpha@example.org");
        assert_eq!(first.signers()[1].principal(), "beta@example.org");
        assert_ne!(first.commitment(), [0_u8; 32]);
    }

    #[test]
    fn empty_invalid_threshold_duplicate_principal_and_duplicate_key_fail() {
        assert_eq!(
            V2QualifierSignerPolicy::genesis(1, 1, vec![]).unwrap_err(),
            V2QualifierSignerPolicyError::EmptySignerSet
        );
        assert_eq!(
            V2QualifierSignerPolicy::genesis(1, 0, vec![signer("a@example.org", 'a')])
                .unwrap_err(),
            V2QualifierSignerPolicyError::InvalidThreshold
        );
        assert_eq!(
            V2QualifierSignerPolicy::genesis(1, 2, vec![signer("a@example.org", 'a')])
                .unwrap_err(),
            V2QualifierSignerPolicyError::InvalidThreshold
        );
        assert_eq!(
            V2QualifierSignerPolicy::genesis(
                1,
                1,
                vec![signer("a@example.org", 'a'), signer("a@example.org", 'b')],
            )
            .unwrap_err(),
            V2QualifierSignerPolicyError::DuplicatePrincipal
        );
        assert_eq!(
            V2QualifierSignerPolicy::genesis(
                1,
                1,
                vec![signer("a@example.org", 'a'), signer("b@example.org", 'a')],
            )
            .unwrap_err(),
            V2QualifierSignerPolicyError::DuplicateKey
        );
    }

    #[test]
    fn malformed_or_unsupported_signer_identity_fails_closed() {
        assert_eq!(
            V2GovernanceSigner::from_hex("bad principal", "ssh-ed25519", &"a".repeat(64))
                .unwrap_err(),
            V2QualifierSignerPolicyError::InvalidPrincipal
        );
        assert_eq!(
            V2GovernanceSigner::from_hex("a@example.org", "ssh ed25519", &"a".repeat(64))
                .unwrap_err(),
            V2QualifierSignerPolicyError::UnsupportedKeyType
        );
        assert_eq!(
            V2GovernanceSigner::from_hex("a@example.org", "ssh-rsa", &"a".repeat(64))
                .unwrap_err(),
            V2QualifierSignerPolicyError::UnsupportedKeyType
        );
        assert_eq!(
            V2GovernanceSigner::from_hex("a@example.org", "ssh-ed25519", &"A".repeat(64))
                .unwrap_err(),
            V2QualifierSignerPolicyError::InvalidHex
        );
        assert!(V2GovernanceSigner::from_hex(
            "hardware@example.org",
            "sk-ssh-ed25519@openssh.com",
            &"b".repeat(64),
        )
        .is_ok());
    }

    #[test]
    fn rotation_requires_exact_predecessor_next_sequence_and_real_change() {
        let genesis = V2QualifierSignerPolicy::genesis(
            1,
            1,
            vec![signer("primary@example.org", 'a')],
        )
        .unwrap();

        assert_eq!(
            V2QualifierSignerPolicy::rotate(
                &genesis,
                [9_u8; 32],
                2,
                1,
                vec![signer("replacement@example.org", 'b')],
            )
            .unwrap_err(),
            V2QualifierSignerPolicyError::WrongPredecessor
        );
        assert_eq!(
            V2QualifierSignerPolicy::rotate(
                &genesis,
                genesis.commitment(),
                3,
                1,
                vec![signer("replacement@example.org", 'b')],
            )
            .unwrap_err(),
            V2QualifierSignerPolicyError::WrongSequence
        );
        assert_eq!(
            V2QualifierSignerPolicy::rotate(
                &genesis,
                genesis.commitment(),
                2,
                1,
                vec![signer("primary@example.org", 'a')],
            )
            .unwrap_err(),
            V2QualifierSignerPolicyError::NoOpRotation
        );

        let successor = V2QualifierSignerPolicy::rotate(
            &genesis,
            genesis.commitment(),
            2,
            1,
            vec![signer("replacement@example.org", 'b')],
        )
        .unwrap();
        assert_eq!(successor.sequence(), 2);
        assert_eq!(successor.predecessor_commitment(), Some(genesis.commitment()));
        assert_ne!(successor.commitment(), genesis.commitment());
    }

    #[test]
    fn sequence_exhaustion_fails_closed() {
        let signers = vec![signer("primary@example.org", 'a')];
        let predecessor = V2QualifierSignerPolicy {
            sequence: u64::MAX,
            predecessor_commitment: Some([7_u8; 32]),
            threshold: 1,
            commitment: signer_policy_commitment(u64::MAX, Some([7_u8; 32]), 1, &signers),
            signers,
        };
        assert_eq!(
            V2QualifierSignerPolicy::rotate(
                &predecessor,
                predecessor.commitment(),
                u64::MAX,
                1,
                vec![signer("replacement@example.org", 'b')],
            )
            .unwrap_err(),
            V2QualifierSignerPolicyError::SequenceExhausted
        );
    }

    #[test]
    fn signer_or_threshold_changes_policy_identity() {
        let baseline = V2QualifierSignerPolicy::genesis(
            1,
            1,
            vec![signer("a@example.org", 'a'), signer("b@example.org", 'b')],
        )
        .unwrap();
        let threshold_changed = V2QualifierSignerPolicy::genesis(
            1,
            2,
            vec![signer("a@example.org", 'a'), signer("b@example.org", 'b')],
        )
        .unwrap();
        let signer_changed = V2QualifierSignerPolicy::genesis(
            1,
            1,
            vec![signer("a@example.org", 'a'), signer("c@example.org", 'c')],
        )
        .unwrap();

        assert_ne!(baseline.commitment(), threshold_changed.commitment());
        assert_ne!(baseline.commitment(), signer_changed.commitment());
    }

    #[test]
    fn namespaces_and_key_encoding_are_domain_separated_and_source_has_no_real_key() {
        assert_ne!(
            V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
            V2_QUALIFIER_SIGNER_POLICY_SSH_NAMESPACE
        );
        assert!(V2_QUALIFIER_ADMISSION_SSH_NAMESPACE.ends_with("@luminousdynamics.org"));
        assert!(V2_QUALIFIER_SIGNER_POLICY_SSH_NAMESPACE.ends_with("@luminousdynamics.org"));
        assert!(V2_SIGNER_PUBLIC_KEY_MATERIAL_ENCODING.contains("no-comment.no-newline"));

        let production = include_str!("v2_qualifier_signer_policy.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "BEGIN OPENSSH PRIVATE KEY",
            "PRIVATE KEY",
            "ssh-keygen -Y sign",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "execution_authority_granted=true",
        ] {
            assert!(!production.contains(forbidden), "forbidden signer-policy surface: {forbidden}");
        }
    }
}
