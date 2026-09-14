// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical, non-authorizing bytes for future signed EUREKA-002 V2 qualifier
//! admissions.
//!
//! This module deliberately performs no signing and contains no real trusted
//! root, governance key, or execution capability. Its only job is to make the
//! bytes that a future external governance signer approves deterministic and
//! unambiguous.

#![allow(dead_code)]

use super::v2_qualification_receipt::{
    V2_QUALIFICATION_CLAIM_SCOPE, V2_QUALIFICATION_COMMAND_CONTRACT_REVISION,
    V2_QUALIFICATION_RECEIPT_SCHEMA, V2_QUALIFICATION_REVISION,
};
use super::v2_qualifier_authority::V2_QUALIFIER_AUTHORITY_SCHEMA;

pub(super) const V2_QUALIFIER_ADMISSION_MANIFEST_SCHEMA: &str =
    "EUREKA.002.V2.QUALIFIER_ADMISSION_MANIFEST.v1";
pub(super) const V2_QUALIFIER_ADMISSION_MANIFEST_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.QUALIFIER_ADMISSION_MANIFEST_COMMITMENT.v1";
pub(super) const V2_REPOSITORY_IDENTITY: &str = "Luminous-Dynamics/symthaea";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualifierAdmissionKind {
    Genesis,
    Rotation,
}

impl V2QualifierAdmissionKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Genesis => "genesis",
            Self::Rotation => "rotation",
        }
    }
}

/// Revision/profile facts are stored in each manifest rather than re-derived
/// from whatever constants happen to be current when an old signature is later
/// audited. This keeps historical signed bytes reconstructable after upgrades.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2QualifierAdmissionRevisions {
    authority_schema_revision: String,
    receipt_schema_revision: String,
    qualification_revision: String,
    command_contract_revision: String,
    claim_scope: String,
    environment_contract_revision: String,
}

impl V2QualifierAdmissionRevisions {
    pub(super) fn current(
        environment_contract_revision: &str,
    ) -> Result<Self, V2QualifierAdmissionManifestError> {
        Self::explicit(
            V2_QUALIFIER_AUTHORITY_SCHEMA,
            V2_QUALIFICATION_RECEIPT_SCHEMA,
            V2_QUALIFICATION_REVISION,
            V2_QUALIFICATION_COMMAND_CONTRACT_REVISION,
            V2_QUALIFICATION_CLAIM_SCOPE,
            environment_contract_revision,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn explicit(
        authority_schema_revision: &str,
        receipt_schema_revision: &str,
        qualification_revision: &str,
        command_contract_revision: &str,
        claim_scope: &str,
        environment_contract_revision: &str,
    ) -> Result<Self, V2QualifierAdmissionManifestError> {
        for value in [
            authority_schema_revision,
            receipt_schema_revision,
            qualification_revision,
            command_contract_revision,
            claim_scope,
            environment_contract_revision,
        ] {
            validate_token(value)?;
        }
        Ok(Self {
            authority_schema_revision: authority_schema_revision.to_string(),
            receipt_schema_revision: receipt_schema_revision.to_string(),
            qualification_revision: qualification_revision.to_string(),
            command_contract_revision: command_contract_revision.to_string(),
            claim_scope: claim_scope.to_string(),
            environment_contract_revision: environment_contract_revision.to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2QualifierAdmissionManifest {
    kind: V2QualifierAdmissionKind,
    subject_head: String,
    subject_tree: String,
    revisions: V2QualifierAdmissionRevisions,
    authority_sequence: u64,
    predecessor_authority_commitment: Option<[u8; 32]>,
    predecessor_currentness_commitment: Option<[u8; 32]>,
    authority_commitment: [u8; 32],
    signer_policy_sequence: u64,
    signer_policy_commitment: [u8; 32],
    workflow_sha256: [u8; 32],
    command_contract_sha256: [u8; 32],
    environment_commitment: [u8; 32],
    evidence_capsule_sha256: [u8; 32],
}

impl V2QualifierAdmissionManifest {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_hex(
        kind: V2QualifierAdmissionKind,
        subject_head: &str,
        subject_tree: &str,
        revisions: V2QualifierAdmissionRevisions,
        authority_sequence: u64,
        predecessor_authority_commitment: Option<&str>,
        predecessor_currentness_commitment: Option<&str>,
        authority_commitment: &str,
        signer_policy_sequence: u64,
        signer_policy_commitment: &str,
        workflow_sha256: &str,
        command_contract_sha256: &str,
        environment_commitment: &str,
        evidence_capsule_sha256: &str,
    ) -> Result<Self, V2QualifierAdmissionManifestError> {
        match kind {
            V2QualifierAdmissionKind::Genesis => {
                if authority_sequence != 1
                    || predecessor_authority_commitment.is_some()
                    || predecessor_currentness_commitment.is_some()
                {
                    return Err(V2QualifierAdmissionManifestError::InvalidGenesisShape);
                }
            }
            V2QualifierAdmissionKind::Rotation => {
                if authority_sequence < 2
                    || predecessor_authority_commitment.is_none()
                    || predecessor_currentness_commitment.is_none()
                {
                    return Err(V2QualifierAdmissionManifestError::InvalidRotationShape);
                }
            }
        }
        if signer_policy_sequence == 0 {
            return Err(V2QualifierAdmissionManifestError::InvalidSignerPolicySequence);
        }

        Ok(Self {
            kind,
            subject_head: canonical_hex(subject_head, 40)?.to_string(),
            subject_tree: canonical_hex(subject_tree, 40)?.to_string(),
            revisions,
            authority_sequence,
            predecessor_authority_commitment: predecessor_authority_commitment
                .map(decode_hex32)
                .transpose()?,
            predecessor_currentness_commitment: predecessor_currentness_commitment
                .map(decode_hex32)
                .transpose()?,
            authority_commitment: decode_hex32(authority_commitment)?,
            signer_policy_sequence,
            signer_policy_commitment: decode_hex32(signer_policy_commitment)?,
            workflow_sha256: decode_hex32(workflow_sha256)?,
            command_contract_sha256: decode_hex32(command_contract_sha256)?,
            environment_commitment: decode_hex32(environment_commitment)?,
            evidence_capsule_sha256: decode_hex32(evidence_capsule_sha256)?,
        })
    }

    /// Exact bytes intended for external governance signing.
    ///
    /// The encoding is deliberately boring: fixed UTF-8 key order, one
    /// `key=value\n` record per field, canonical lowercase hex, and explicit
    /// `none` for absent predecessor commitments. No timestamps, PR numbers,
    /// usernames, URLs, locale-sensitive formatting, or map serialization are
    /// admitted into the signed bytes.
    pub(super) fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::new();
        push_field(
            &mut out,
            "manifest_schema_revision",
            V2_QUALIFIER_ADMISSION_MANIFEST_SCHEMA,
        );
        push_field(&mut out, "admission_kind", self.kind.as_str());
        push_field(&mut out, "repository", V2_REPOSITORY_IDENTITY);
        push_field(&mut out, "subject_head", &self.subject_head);
        push_field(&mut out, "subject_tree", &self.subject_tree);
        push_field(
            &mut out,
            "authority_schema_revision",
            &self.revisions.authority_schema_revision,
        );
        push_field(
            &mut out,
            "authority_sequence",
            &self.authority_sequence.to_string(),
        );
        push_field(
            &mut out,
            "predecessor_authority_commitment",
            &optional_hex(self.predecessor_authority_commitment),
        );
        push_field(
            &mut out,
            "predecessor_currentness_commitment",
            &optional_hex(self.predecessor_currentness_commitment),
        );
        push_field(
            &mut out,
            "authority_commitment",
            &hex32(self.authority_commitment),
        );
        push_field(
            &mut out,
            "signer_policy_sequence",
            &self.signer_policy_sequence.to_string(),
        );
        push_field(
            &mut out,
            "signer_policy_commitment",
            &hex32(self.signer_policy_commitment),
        );
        push_field(
            &mut out,
            "receipt_schema_revision",
            &self.revisions.receipt_schema_revision,
        );
        push_field(
            &mut out,
            "qualification_revision",
            &self.revisions.qualification_revision,
        );
        push_field(
            &mut out,
            "command_contract_revision",
            &self.revisions.command_contract_revision,
        );
        push_field(&mut out, "claim_scope", &self.revisions.claim_scope);
        push_field(&mut out, "workflow_sha256", &hex32(self.workflow_sha256));
        push_field(
            &mut out,
            "command_contract_sha256",
            &hex32(self.command_contract_sha256),
        );
        push_field(
            &mut out,
            "environment_contract_revision",
            &self.revisions.environment_contract_revision,
        );
        push_field(
            &mut out,
            "environment_commitment",
            &hex32(self.environment_commitment),
        );
        push_field(
            &mut out,
            "evidence_capsule_sha256",
            &hex32(self.evidence_capsule_sha256),
        );
        push_field(&mut out, "execution_authority_granted", "false");
        out.into_bytes()
    }

    pub(super) fn commitment(&self) -> [u8; 32] {
        let bytes = self.canonical_bytes();
        let mut domain_separated = Vec::new();
        encode_bytes(
            &mut domain_separated,
            V2_QUALIFIER_ADMISSION_MANIFEST_COMMITMENT_REVISION.as_bytes(),
        );
        encode_bytes(&mut domain_separated, &bytes);
        *blake3::hash(&domain_separated).as_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualifierAdmissionManifestError {
    InvalidHex,
    InvalidToken,
    InvalidGenesisShape,
    InvalidRotationShape,
    InvalidSignerPolicySequence,
}

fn validate_token(value: &str) -> Result<(), V2QualifierAdmissionManifestError> {
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
        })
    {
        return Err(V2QualifierAdmissionManifestError::InvalidToken);
    }
    Ok(())
}

fn canonical_hex(
    value: &str,
    expected_len: usize,
) -> Result<&str, V2QualifierAdmissionManifestError> {
    if value.len() != expected_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2QualifierAdmissionManifestError::InvalidHex);
    }
    Ok(value)
}

fn decode_hex32(value: &str) -> Result<[u8; 32], V2QualifierAdmissionManifestError> {
    canonical_hex(value, 64)?;
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(V2QualifierAdmissionManifestError::InvalidHex)?;
        let low = hex_nibble(chunk[1]).ok_or(V2QualifierAdmissionManifestError::InvalidHex)?;
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

fn optional_hex(value: Option<[u8; 32]>) -> String {
    value.map(hex32).unwrap_or_else(|| "none".to_string())
}

fn hex32(value: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in value {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn push_field(output: &mut String, key: &str, value: &str) {
    output.push_str(key);
    output.push('=');
    output.push_str(value);
    output.push('\n');
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn head(byte: char) -> String {
        byte.to_string().repeat(40)
    }

    fn current_revisions() -> V2QualifierAdmissionRevisions {
        V2QualifierAdmissionRevisions::current(
            "EUREKA.002.V2.QUALIFIER_ENVIRONMENT.v1",
        )
        .unwrap()
    }

    fn genesis() -> V2QualifierAdmissionManifest {
        V2QualifierAdmissionManifest::from_hex(
            V2QualifierAdmissionKind::Genesis,
            &head('1'),
            &head('2'),
            current_revisions(),
            1,
            None,
            None,
            &hex('a'),
            1,
            &hex('b'),
            &hex('c'),
            &hex('d'),
            &hex('e'),
            &hex('f'),
        )
        .unwrap()
    }

    #[test]
    fn canonical_genesis_bytes_are_fixed_and_language_neutral() {
        let manifest = genesis();
        let expected = format!(
            "manifest_schema_revision={V2_QUALIFIER_ADMISSION_MANIFEST_SCHEMA}\n\
admission_kind=genesis\n\
repository={V2_REPOSITORY_IDENTITY}\n\
subject_head={}\n\
subject_tree={}\n\
authority_schema_revision={V2_QUALIFIER_AUTHORITY_SCHEMA}\n\
authority_sequence=1\n\
predecessor_authority_commitment=none\n\
predecessor_currentness_commitment=none\n\
authority_commitment={}\n\
signer_policy_sequence=1\n\
signer_policy_commitment={}\n\
receipt_schema_revision={V2_QUALIFICATION_RECEIPT_SCHEMA}\n\
qualification_revision={V2_QUALIFICATION_REVISION}\n\
command_contract_revision={V2_QUALIFICATION_COMMAND_CONTRACT_REVISION}\n\
claim_scope={V2_QUALIFICATION_CLAIM_SCOPE}\n\
workflow_sha256={}\n\
command_contract_sha256={}\n\
environment_contract_revision=EUREKA.002.V2.QUALIFIER_ENVIRONMENT.v1\n\
environment_commitment={}\n\
evidence_capsule_sha256={}\n\
execution_authority_granted=false\n",
            head('1'),
            head('2'),
            hex('a'),
            hex('b'),
            hex('c'),
            hex('d'),
            hex('e'),
            hex('f'),
        );
        assert_eq!(manifest.canonical_bytes(), expected.into_bytes());
        assert_ne!(manifest.commitment(), [0_u8; 32]);
    }

    #[test]
    fn historical_revisions_are_stored_instead_of_rederived() {
        let revisions = V2QualifierAdmissionRevisions::explicit(
            "EUREKA.002.V2.TRUSTED_QUALIFIER_AUTHORITY.v0",
            "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v0",
            "EUREKA.002.V2.BACKEND_QUALIFICATION.v0",
            "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v0",
            "historical-build-only",
            "EUREKA.002.V2.QUALIFIER_ENVIRONMENT.v0",
        )
        .unwrap();
        let manifest = V2QualifierAdmissionManifest::from_hex(
            V2QualifierAdmissionKind::Genesis,
            &head('1'),
            &head('2'),
            revisions,
            1,
            None,
            None,
            &hex('a'),
            1,
            &hex('b'),
            &hex('c'),
            &hex('d'),
            &hex('e'),
            &hex('f'),
        )
        .unwrap();
        let text = String::from_utf8(manifest.canonical_bytes()).unwrap();
        assert!(text.contains("authority_schema_revision=EUREKA.002.V2.TRUSTED_QUALIFIER_AUTHORITY.v0\n"));
        assert!(text.contains("receipt_schema_revision=EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v0\n"));
        assert!(text.contains("command_contract_revision=EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v0\n"));
        assert!(text.contains("environment_contract_revision=EUREKA.002.V2.QUALIFIER_ENVIRONMENT.v0\n"));
        assert!(text.contains("claim_scope=historical-build-only\n"));
    }

    #[test]
    fn genesis_and_rotation_shapes_fail_closed() {
        assert_eq!(
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('1'),
                &head('2'),
                current_revisions(),
                2,
                None,
                None,
                &hex('a'),
                1,
                &hex('b'),
                &hex('c'),
                &hex('d'),
                &hex('e'),
                &hex('f'),
            )
            .unwrap_err(),
            V2QualifierAdmissionManifestError::InvalidGenesisShape
        );
        assert_eq!(
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Rotation,
                &head('1'),
                &head('2'),
                current_revisions(),
                2,
                Some(&hex('1')),
                None,
                &hex('a'),
                1,
                &hex('b'),
                &hex('c'),
                &hex('d'),
                &hex('e'),
                &hex('f'),
            )
            .unwrap_err(),
            V2QualifierAdmissionManifestError::InvalidRotationShape
        );
    }

    #[test]
    fn rotation_binds_both_predecessor_identities_and_domain() {
        let predecessor_authority = hex('1');
        let predecessor_currentness = hex('2');
        let rotation = V2QualifierAdmissionManifest::from_hex(
            V2QualifierAdmissionKind::Rotation,
            &head('3'),
            &head('4'),
            current_revisions(),
            2,
            Some(&predecessor_authority),
            Some(&predecessor_currentness),
            &hex('a'),
            2,
            &hex('b'),
            &hex('c'),
            &hex('d'),
            &hex('e'),
            &hex('f'),
        )
        .unwrap();
        let text = String::from_utf8(rotation.canonical_bytes()).unwrap();
        assert!(text.contains("admission_kind=rotation\n"));
        assert!(text.contains(&format!(
            "predecessor_authority_commitment={predecessor_authority}\n"
        )));
        assert!(text.contains(&format!(
            "predecessor_currentness_commitment={predecessor_currentness}\n"
        )));
        assert_ne!(rotation.commitment(), genesis().commitment());
    }

    #[test]
    fn malformed_or_noncanonical_inputs_fail_closed() {
        let upper = "A".repeat(64);
        assert_eq!(
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('1'),
                &head('2'),
                current_revisions(),
                1,
                None,
                None,
                &upper,
                1,
                &hex('b'),
                &hex('c'),
                &hex('d'),
                &hex('e'),
                &hex('f'),
            )
            .unwrap_err(),
            V2QualifierAdmissionManifestError::InvalidHex
        );
        assert_eq!(
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('1'),
                &head('2'),
                current_revisions(),
                1,
                None,
                None,
                &hex('a'),
                0,
                &hex('b'),
                &hex('c'),
                &hex('d'),
                &hex('e'),
                &hex('f'),
            )
            .unwrap_err(),
            V2QualifierAdmissionManifestError::InvalidSignerPolicySequence
        );
        assert_eq!(
            V2QualifierAdmissionRevisions::explicit(
                V2_QUALIFIER_AUTHORITY_SCHEMA,
                V2_QUALIFICATION_RECEIPT_SCHEMA,
                V2_QUALIFICATION_REVISION,
                V2_QUALIFICATION_COMMAND_CONTRACT_REVISION,
                V2_QUALIFICATION_CLAIM_SCOPE,
                "bad\nrevision",
            )
            .unwrap_err(),
            V2QualifierAdmissionManifestError::InvalidToken
        );
    }

    #[test]
    fn every_authority_bearing_identity_changes_manifest_commitment() {
        let baseline = genesis();
        let variants = [
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('3'),
                &head('2'),
                current_revisions(),
                1,
                None,
                None,
                &hex('a'),
                1,
                &hex('b'),
                &hex('c'),
                &hex('d'),
                &hex('e'),
                &hex('f'),
            )
            .unwrap(),
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('1'),
                &head('2'),
                current_revisions(),
                1,
                None,
                None,
                &hex('9'),
                1,
                &hex('b'),
                &hex('c'),
                &hex('d'),
                &hex('e'),
                &hex('f'),
            )
            .unwrap(),
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('1'),
                &head('2'),
                current_revisions(),
                1,
                None,
                None,
                &hex('a'),
                2,
                &hex('8'),
                &hex('c'),
                &hex('d'),
                &hex('e'),
                &hex('f'),
            )
            .unwrap(),
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('1'),
                &head('2'),
                current_revisions(),
                1,
                None,
                None,
                &hex('a'),
                1,
                &hex('b'),
                &hex('7'),
                &hex('d'),
                &hex('e'),
                &hex('f'),
            )
            .unwrap(),
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('1'),
                &head('2'),
                current_revisions(),
                1,
                None,
                None,
                &hex('a'),
                1,
                &hex('b'),
                &hex('c'),
                &hex('6'),
                &hex('e'),
                &hex('f'),
            )
            .unwrap(),
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('1'),
                &head('2'),
                V2QualifierAdmissionRevisions::current(
                    "EUREKA.002.V2.QUALIFIER_ENVIRONMENT.v2",
                )
                .unwrap(),
                1,
                None,
                None,
                &hex('a'),
                1,
                &hex('b'),
                &hex('c'),
                &hex('d'),
                &hex('5'),
                &hex('f'),
            )
            .unwrap(),
            V2QualifierAdmissionManifest::from_hex(
                V2QualifierAdmissionKind::Genesis,
                &head('1'),
                &head('2'),
                current_revisions(),
                1,
                None,
                None,
                &hex('a'),
                1,
                &hex('b'),
                &hex('c'),
                &hex('d'),
                &hex('e'),
                &hex('4'),
            )
            .unwrap(),
        ];

        for variant in variants {
            assert_ne!(baseline.commitment(), variant.commitment());
        }
    }

    #[test]
    fn admission_manifest_source_contains_no_signing_key_or_execution_surface() {
        let production = include_str!("v2_qualifier_admission_manifest.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "PRIVATE KEY",
            "BEGIN OPENSSH PRIVATE KEY",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "V2CanaryAuthorization",
            "V2RealHeldOutPairedFreeze",
            "execution_authority_granted=true",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden admission surface: {forbidden}"
            );
        }
    }
}
