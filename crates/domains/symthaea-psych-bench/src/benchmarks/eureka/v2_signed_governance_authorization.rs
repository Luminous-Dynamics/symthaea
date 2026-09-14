// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Signed-governance authorization mechanics for one exact EUREKA-002 V2
//! qualifier-authority rotation.
//!
//! This module bridges cryptographically verified canonical admission bytes to
//! the existing authority/currentness mechanics. It deliberately stops before
//! trusted-root admission: a signed governance authorization still requires the
//! independent qualification/environment/evidence-capsule ceremony before it
//! can become an admitted qualifier root.

#![allow(dead_code)]

use super::v2_openssh_admission_verifier::{
    V2OpenSshAdmissionSignature, V2OpenSshAdmissionVerifierError,
    verify_openssh_admission_signatures,
};
use super::v2_qualification_receipt::{
    V2_QUALIFICATION_CLAIM_SCOPE, V2_QUALIFICATION_COMMAND_CONTRACT_REVISION,
    V2_QUALIFICATION_RECEIPT_SCHEMA, V2_QUALIFICATION_REVISION,
};
use super::v2_qualifier_admission_manifest::V2_QUALIFIER_ADMISSION_MANIFEST_COMMITMENT_REVISION;
use super::v2_qualifier_admission_manifest_parser::{
    V2QualifierAdmissionParseError, parse_canonical_admission_manifest,
};
use super::v2_qualifier_authority::{
    V2_QUALIFIER_AUTHORITY_SCHEMA, V2QualifierAuthorityError, V2QualifierAuthorityProfile,
    V2QualifierAuthorityRecord,
};
use super::v2_qualifier_currentness::V2QualifierAuthorityLineage;
use super::v2_qualifier_signer_policy::V2QualifierSignerPolicy;

pub(super) const V2_SIGNED_GOVERNANCE_AUTHORIZATION_REVISION: &str =
    "EUREKA.002.V2.SIGNED_GOVERNANCE_AUTHORIZATION.v1";

/// Governance approval of exactly one authority rotation.
///
/// This type is intentionally move-only and intentionally weaker than an
/// admitted qualifier root. It authenticates who approved which exact canonical
/// transition bytes; it does not establish that the referenced qualification,
/// environment closure, or evidence capsule has passed the independent root-
/// admission ceremony.
#[derive(Debug)]
pub(super) struct V2SignedGovernanceAuthorization {
    manifest_commitment: [u8; 32],
    signature_set_commitment: [u8; 32],
    signer_policy_sequence: u64,
    signer_policy_commitment: [u8; 32],
    predecessor_authority_commitment: [u8; 32],
    predecessor_currentness_commitment: [u8; 32],
    authority_sequence: u64,
    authority_commitment: [u8; 32],
    subject_head: String,
    subject_tree: String,
    environment_commitment: [u8; 32],
    evidence_capsule_sha256: [u8; 32],
    commitment: [u8; 32],
}

impl V2SignedGovernanceAuthorization {
    pub(super) const fn manifest_commitment(&self) -> [u8; 32] {
        self.manifest_commitment
    }

    pub(super) const fn signature_set_commitment(&self) -> [u8; 32] {
        self.signature_set_commitment
    }

    pub(super) const fn signer_policy_sequence(&self) -> u64 {
        self.signer_policy_sequence
    }

    pub(super) const fn signer_policy_commitment(&self) -> [u8; 32] {
        self.signer_policy_commitment
    }

    pub(super) const fn predecessor_authority_commitment(&self) -> [u8; 32] {
        self.predecessor_authority_commitment
    }

    pub(super) const fn predecessor_currentness_commitment(&self) -> [u8; 32] {
        self.predecessor_currentness_commitment
    }

    pub(super) const fn authority_sequence(&self) -> u64 {
        self.authority_sequence
    }

    pub(super) const fn authority_commitment(&self) -> [u8; 32] {
        self.authority_commitment
    }

    pub(super) fn subject_head(&self) -> &str {
        &self.subject_head
    }

    pub(super) fn subject_tree(&self) -> &str {
        &self.subject_tree
    }

    pub(super) const fn environment_commitment(&self) -> [u8; 32] {
        self.environment_commitment
    }

    pub(super) const fn evidence_capsule_sha256(&self) -> [u8; 32] {
        self.evidence_capsule_sha256
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2SignedGovernanceAuthorizationError {
    Manifest(V2QualifierAdmissionParseError),
    Signature(V2OpenSshAdmissionVerifierError),
    Authority(V2QualifierAuthorityError),
    WrongAdmissionKind,
    UnsupportedAuthorityRevision,
    UnsupportedReceiptRevision,
    UnsupportedQualificationRevision,
    UnsupportedCommandContractRevision,
    UnsupportedClaimScope,
    SignerPolicyMismatch,
    PredecessorAuthorityMismatch,
    PredecessorCurrentnessMismatch,
    CandidateAuthorityMismatch,
    ManifestCommitmentMismatch,
    InvalidNumber,
    InvalidHex,
}

/// Verify and bind one signed rotation manifest to the exact active authority
/// predecessor and exact mechanically reconstructable successor.
///
/// The current v1 bridge intentionally accepts only the *current* authority /
/// receipt / qualification / command-contract revisions. Historical signed
/// manifests remain auditable, but admitting historical revision transitions is
/// a separate theorem rather than an implicit fallback.
pub(super) fn authorize_signed_authority_rotation(
    manifest_bytes: &[u8],
    signer_policy: &V2QualifierSignerPolicy,
    signatures: &[V2OpenSshAdmissionSignature<'_>],
    active_lineage: &V2QualifierAuthorityLineage,
    predecessor: &V2QualifierAuthorityRecord,
    candidate: &V2QualifierAuthorityRecord,
) -> Result<V2SignedGovernanceAuthorization, V2SignedGovernanceAuthorizationError> {
    let manifest = parse_canonical_admission_manifest(manifest_bytes)
        .map_err(V2SignedGovernanceAuthorizationError::Manifest)?;
    let verified = verify_openssh_admission_signatures(manifest_bytes, signer_policy, signatures)
        .map_err(V2SignedGovernanceAuthorizationError::Signature)?;

    let text = std::str::from_utf8(manifest_bytes).map_err(|_| {
        V2SignedGovernanceAuthorizationError::Manifest(V2QualifierAdmissionParseError::InvalidUtf8)
    })?;

    if exact_field(text, "admission_kind")? != "rotation" {
        return Err(V2SignedGovernanceAuthorizationError::WrongAdmissionKind);
    }
    if exact_field(text, "authority_schema_revision")? != V2_QUALIFIER_AUTHORITY_SCHEMA {
        return Err(V2SignedGovernanceAuthorizationError::UnsupportedAuthorityRevision);
    }
    if exact_field(text, "receipt_schema_revision")? != V2_QUALIFICATION_RECEIPT_SCHEMA {
        return Err(V2SignedGovernanceAuthorizationError::UnsupportedReceiptRevision);
    }
    if exact_field(text, "qualification_revision")? != V2_QUALIFICATION_REVISION {
        return Err(V2SignedGovernanceAuthorizationError::UnsupportedQualificationRevision);
    }
    if exact_field(text, "command_contract_revision")?
        != V2_QUALIFICATION_COMMAND_CONTRACT_REVISION
    {
        return Err(V2SignedGovernanceAuthorizationError::UnsupportedCommandContractRevision);
    }
    if exact_field(text, "claim_scope")? != V2_QUALIFICATION_CLAIM_SCOPE {
        return Err(V2SignedGovernanceAuthorizationError::UnsupportedClaimScope);
    }

    let signer_policy_sequence = parse_u64(exact_field(text, "signer_policy_sequence")?)?;
    let signer_policy_commitment = decode_hex32(exact_field(text, "signer_policy_commitment")?)?;
    if signer_policy_sequence != signer_policy.sequence()
        || signer_policy_sequence != verified.signer_policy_sequence()
        || signer_policy_commitment != signer_policy.commitment()
        || signer_policy_commitment != verified.signer_policy_commitment()
    {
        return Err(V2SignedGovernanceAuthorizationError::SignerPolicyMismatch);
    }

    let predecessor_authority_commitment = decode_hex32(exact_field(
        text,
        "predecessor_authority_commitment",
    )?)?;
    let predecessor_currentness_commitment = decode_hex32(exact_field(
        text,
        "predecessor_currentness_commitment",
    )?)?;
    let authority_sequence = parse_u64(exact_field(text, "authority_sequence")?)?;
    let authority_commitment = decode_hex32(exact_field(text, "authority_commitment")?)?;

    if predecessor.sequence() != active_lineage.current_sequence()
        || predecessor.commitment() != active_lineage.current_authority_commitment()
        || predecessor.commitment() != predecessor_authority_commitment
    {
        return Err(V2SignedGovernanceAuthorizationError::PredecessorAuthorityMismatch);
    }
    if active_lineage.commitment() != predecessor_currentness_commitment {
        return Err(V2SignedGovernanceAuthorizationError::PredecessorCurrentnessMismatch);
    }

    let workflow_sha256 = exact_field(text, "workflow_sha256")?;
    let command_contract_sha256 = exact_field(text, "command_contract_sha256")?;
    let profile =
        V2QualifierAuthorityProfile::current_from_hex(workflow_sha256, command_contract_sha256)
            .map_err(V2SignedGovernanceAuthorizationError::Authority)?;
    let reconstructed = V2QualifierAuthorityRecord::rotate(
        predecessor,
        predecessor_authority_commitment,
        authority_sequence,
        profile,
    )
    .map_err(V2SignedGovernanceAuthorizationError::Authority)?;

    if &reconstructed != candidate
        || candidate.sequence() != authority_sequence
        || candidate.predecessor_commitment() != Some(predecessor_authority_commitment)
        || candidate.commitment() != authority_commitment
        || reconstructed.commitment() != authority_commitment
    {
        return Err(V2SignedGovernanceAuthorizationError::CandidateAuthorityMismatch);
    }

    if manifest.commitment() != verified.manifest_commitment() {
        return Err(V2SignedGovernanceAuthorizationError::ManifestCommitmentMismatch);
    }

    let subject_head = exact_field(text, "subject_head")?.to_string();
    let subject_tree = exact_field(text, "subject_tree")?.to_string();
    let environment_commitment = decode_hex32(exact_field(text, "environment_commitment")?)?;
    let evidence_capsule_sha256 = decode_hex32(exact_field(text, "evidence_capsule_sha256")?)?;

    let mut authorization = V2SignedGovernanceAuthorization {
        manifest_commitment: manifest.commitment(),
        signature_set_commitment: verified.commitment(),
        signer_policy_sequence,
        signer_policy_commitment,
        predecessor_authority_commitment,
        predecessor_currentness_commitment,
        authority_sequence,
        authority_commitment,
        subject_head,
        subject_tree,
        environment_commitment,
        evidence_capsule_sha256,
        commitment: [0_u8; 32],
    };
    authorization.commitment = authorization_commitment(&authorization);
    Ok(authorization)
}

fn authorization_commitment(authorization: &V2SignedGovernanceAuthorization) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_SIGNED_GOVERNANCE_AUTHORIZATION_REVISION.as_bytes(),
    );
    encode_bytes(
        &mut bytes,
        V2_QUALIFIER_ADMISSION_MANIFEST_COMMITMENT_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&authorization.manifest_commitment);
    bytes.extend_from_slice(&authorization.signature_set_commitment);
    bytes.extend_from_slice(&authorization.signer_policy_sequence.to_le_bytes());
    bytes.extend_from_slice(&authorization.signer_policy_commitment);
    bytes.extend_from_slice(&authorization.predecessor_authority_commitment);
    bytes.extend_from_slice(&authorization.predecessor_currentness_commitment);
    bytes.extend_from_slice(&authorization.authority_sequence.to_le_bytes());
    bytes.extend_from_slice(&authorization.authority_commitment);
    encode_bytes(&mut bytes, authorization.subject_head.as_bytes());
    encode_bytes(&mut bytes, authorization.subject_tree.as_bytes());
    bytes.extend_from_slice(&authorization.environment_commitment);
    bytes.extend_from_slice(&authorization.evidence_capsule_sha256);
    *blake3::hash(&bytes).as_bytes()
}

fn exact_field<'a>(
    text: &'a str,
    key: &'static str,
) -> Result<&'a str, V2SignedGovernanceAuthorizationError> {
    let prefix = format!("{key}=");
    text.lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .ok_or(V2SignedGovernanceAuthorizationError::Manifest(
            V2QualifierAdmissionParseError::WrongField(key),
        ))
}

fn parse_u64(value: &str) -> Result<u64, V2SignedGovernanceAuthorizationError> {
    value
        .parse::<u64>()
        .map_err(|_| V2SignedGovernanceAuthorizationError::InvalidNumber)
}

fn decode_hex32(value: &str) -> Result<[u8; 32], V2SignedGovernanceAuthorizationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2SignedGovernanceAuthorizationError::InvalidHex);
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(V2SignedGovernanceAuthorizationError::InvalidHex)?;
        let low = hex_nibble(chunk[1]).ok_or(V2SignedGovernanceAuthorizationError::InvalidHex)?;
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
    use super::super::v2_qualifier_signer_policy::V2GovernanceSigner;
    use super::*;

    const MANIFEST: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-rotation.env");
    const ALPHA_PUB: &str = include_str!("fixtures/eureka-v2-signed-transition-alpha.pub");
    const BETA_PUB: &str = include_str!("fixtures/eureka-v2-signed-transition-beta.pub");
    const ALPHA_SIG: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-alpha.sig");
    const BETA_SIG: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-beta.sig");

    const ALPHA_PRINCIPAL: &str = "fixture-transition-alpha@example.invalid";
    const BETA_PRINCIPAL: &str = "fixture-transition-beta@example.invalid";
    const ALPHA_KEY_SHA256: &str =
        "98c9c52c03393af5fabb34d1a41a60a87073525c52f5fb7ca0a51c7de931da2d";
    const BETA_KEY_SHA256: &str =
        "70535ce88cf2b7575de01e7e98559e95fa66d18dd7317f9411500f3930dcfbd7";
    const POLICY_COMMITMENT: &str =
        "da130b66f57e0084c1de91468f0fa2de67b4506925a3807686297aa2305805ae";
    const GENESIS_AUTHORITY: &str =
        "ce53583b605d9a01ef1a290b5af37de077c3b2958aa80d29d7a035cd65edcfb2";
    const GENESIS_CURRENTNESS: &str =
        "b11a31922977bd740943af81b926570da08d7ad8f074d2af8c7e3cebe0c1eb9b";
    const SUCCESSOR_AUTHORITY: &str =
        "aad37878240ef98d51a173d5a22113ea9761b532efb8638acb9d11c342a75f3d";

    fn policy() -> V2QualifierSignerPolicy {
        let alpha = V2GovernanceSigner::from_hex(
            ALPHA_PRINCIPAL,
            "ssh-ed25519",
            ALPHA_KEY_SHA256,
        )
        .unwrap();
        let beta = V2GovernanceSigner::from_hex(
            BETA_PRINCIPAL,
            "ssh-ed25519",
            BETA_KEY_SHA256,
        )
        .unwrap();
        V2QualifierSignerPolicy::genesis(1, 2, vec![beta, alpha]).unwrap()
    }

    fn signatures() -> [V2OpenSshAdmissionSignature<'static>; 2] {
        [
            V2OpenSshAdmissionSignature::new(ALPHA_PRINCIPAL, ALPHA_PUB, ALPHA_SIG),
            V2OpenSshAdmissionSignature::new(BETA_PRINCIPAL, BETA_PUB, BETA_SIG),
        ]
    }

    fn predecessor() -> V2QualifierAuthorityRecord {
        V2QualifierAuthorityRecord::genesis(
            1,
            V2QualifierAuthorityProfile::current_from_hex(&"c".repeat(64), &"d".repeat(64))
                .unwrap(),
        )
        .unwrap()
    }

    fn candidate(predecessor: &V2QualifierAuthorityRecord) -> V2QualifierAuthorityRecord {
        V2QualifierAuthorityRecord::rotate(
            predecessor,
            predecessor.commitment(),
            2,
            V2QualifierAuthorityProfile::current_from_hex(&"e".repeat(64), &"d".repeat(64))
                .unwrap(),
        )
        .unwrap()
    }

    fn active_lineage() -> V2QualifierAuthorityLineage {
        V2QualifierAuthorityLineage::activate_genesis(predecessor()).unwrap()
    }

    #[test]
    fn synthetic_fixture_is_derived_from_real_authority_currentness_and_policy_mechanics() {
        let predecessor = predecessor();
        let candidate = candidate(&predecessor);
        let lineage = active_lineage();
        let policy = policy();

        assert_eq!(hex32(predecessor.commitment()), GENESIS_AUTHORITY);
        assert_eq!(hex32(lineage.commitment()), GENESIS_CURRENTNESS);
        assert_eq!(hex32(candidate.commitment()), SUCCESSOR_AUTHORITY);
        assert_eq!(hex32(policy.commitment()), POLICY_COMMITMENT);
    }

    #[test]
    fn two_of_two_signatures_authorize_exact_rotation_but_not_a_root() {
        let predecessor = predecessor();
        let candidate = candidate(&predecessor);
        let lineage = active_lineage();
        let policy = policy();
        let authorization = authorize_signed_authority_rotation(
            MANIFEST,
            &policy,
            &signatures(),
            &lineage,
            &predecessor,
            &candidate,
        )
        .unwrap();

        assert_eq!(authorization.authority_sequence(), 2);
        assert_eq!(authorization.authority_commitment(), candidate.commitment());
        assert_eq!(
            authorization.predecessor_authority_commitment(),
            predecessor.commitment()
        );
        assert_eq!(
            authorization.predecessor_currentness_commitment(),
            lineage.commitment()
        );
        assert_eq!(authorization.signer_policy_commitment(), policy.commitment());
        assert_eq!(authorization.signer_policy_sequence(), 1);
        assert_eq!(authorization.subject_head(), "1".repeat(40));
        assert_eq!(authorization.subject_tree(), "2".repeat(40));
        assert_eq!(authorization.environment_commitment(), [0xff; 32]);
        assert_eq!(authorization.evidence_capsule_sha256(), [0xaa; 32]);
        assert_ne!(authorization.manifest_commitment(), [0_u8; 32]);
        assert_ne!(authorization.signature_set_commitment(), [0_u8; 32]);
        assert_ne!(authorization.commitment(), [0_u8; 32]);
    }

    #[test]
    fn signed_manifest_cannot_authorize_sibling_or_different_profile() {
        let predecessor = predecessor();
        let lineage = active_lineage();
        let policy = policy();
        let sibling = V2QualifierAuthorityRecord::rotate(
            &predecessor,
            predecessor.commitment(),
            2,
            V2QualifierAuthorityProfile::current_from_hex(&"e".repeat(64), &"f".repeat(64))
                .unwrap(),
        )
        .unwrap();

        assert_eq!(
            authorize_signed_authority_rotation(
                MANIFEST,
                &policy,
                &signatures(),
                &lineage,
                &predecessor,
                &sibling,
            )
            .unwrap_err(),
            V2SignedGovernanceAuthorizationError::CandidateAuthorityMismatch
        );
    }

    #[test]
    fn signed_rotation_is_bound_to_exact_predecessor_currentness() {
        let predecessor = predecessor();
        let candidate = candidate(&predecessor);
        let wrong_predecessor = V2QualifierAuthorityRecord::genesis(
            1,
            V2QualifierAuthorityProfile::current_from_hex(&"b".repeat(64), &"d".repeat(64))
                .unwrap(),
        )
        .unwrap();
        let wrong_lineage =
            V2QualifierAuthorityLineage::activate_genesis(wrong_predecessor).unwrap();
        let policy = policy();

        assert_eq!(
            authorize_signed_authority_rotation(
                MANIFEST,
                &policy,
                &signatures(),
                &wrong_lineage,
                &predecessor,
                &candidate,
            )
            .unwrap_err(),
            V2SignedGovernanceAuthorizationError::PredecessorAuthorityMismatch
        );
    }

    #[test]
    fn one_signature_cannot_create_governance_authorization() {
        let predecessor = predecessor();
        let candidate = candidate(&predecessor);
        let lineage = active_lineage();
        let policy = policy();
        let one = [V2OpenSshAdmissionSignature::new(
            ALPHA_PRINCIPAL,
            ALPHA_PUB,
            ALPHA_SIG,
        )];

        assert_eq!(
            authorize_signed_authority_rotation(
                MANIFEST,
                &policy,
                &one,
                &lineage,
                &predecessor,
                &candidate,
            )
            .unwrap_err(),
            V2SignedGovernanceAuthorizationError::Signature(
                V2OpenSshAdmissionVerifierError::InsufficientSignatures
            )
        );
    }

    #[test]
    fn changing_signed_environment_bytes_invalidates_signatures() {
        let predecessor = predecessor();
        let candidate = candidate(&predecessor);
        let lineage = active_lineage();
        let policy = policy();
        let mutated = std::str::from_utf8(MANIFEST)
            .unwrap()
            .replacen(
                &format!("environment_commitment={}", "f".repeat(64)),
                &format!("environment_commitment={}", "e".repeat(64)),
                1,
            );

        assert_eq!(
            authorize_signed_authority_rotation(
                mutated.as_bytes(),
                &policy,
                &signatures(),
                &lineage,
                &predecessor,
                &candidate,
            )
            .unwrap_err(),
            V2SignedGovernanceAuthorizationError::Signature(
                V2OpenSshAdmissionVerifierError::SignatureRejected
            )
        );
    }

    #[test]
    fn authorization_source_contains_no_root_permit_or_execution_surface() {
        let production = include_str!("v2_signed_governance_authorization.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "execution_authority_granted=true",
            "-Y sign",
            "PRIVATE KEY",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden signed-governance surface: {forbidden}"
            );
        }
        assert!(production.contains("verify_openssh_admission_signatures"));
        assert!(production.contains("V2QualifierAuthorityRecord::rotate"));
        assert!(production.contains("active_lineage.commitment()"));
    }

    fn hex32(value: [u8; 32]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in value {
            output.push(HEX[usize::from(byte >> 4)] as char);
            output.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        output
    }
}
