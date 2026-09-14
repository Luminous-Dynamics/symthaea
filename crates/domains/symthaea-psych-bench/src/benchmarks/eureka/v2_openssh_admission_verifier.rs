// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! OpenSSH-backed verification mechanics for future signed EUREKA-002 V2
//! qualifier admissions.
//!
//! This module is test-only qualification machinery. It contains no real
//! governance key, performs no root admission, and grants no execution
//! authority. OpenSSH verifies detached SSHSIG cryptography; this module
//! independently binds those signatures to the exact canonical admission
//! bytes and exact signer policy before they can count toward threshold.
//!
//! The external `ssh-keygen` and `sha256sum` executables are part of the
//! verification TCB. A real admission ceremony must therefore bind their exact
//! environment/tool identity through the environment/provenance theorem rather
//! than treating this fixture qualification as sufficient authority.

#![allow(dead_code)]

use super::v2_openssh_sshsig_metadata::validate_admission_sshsig_v1;
use super::v2_qualifier_admission_manifest_parser::parse_canonical_admission_manifest;
use super::v2_qualifier_signer_policy::{
    V2GovernanceSigner, V2QualifierSignerPolicy, V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
};
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const V2_VERIFIED_ADMISSION_SIGNATURE_SET_REVISION: &str =
    "EUREKA.002.V2.VERIFIED_ADMISSION_SIGNATURE_SET.v1";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy)]
pub(super) struct V2OpenSshAdmissionSignature<'a> {
    principal: &'a str,
    public_key_line: &'a str,
    armored_signature: &'a [u8],
}

impl<'a> V2OpenSshAdmissionSignature<'a> {
    pub(super) const fn new(
        principal: &'a str,
        public_key_line: &'a str,
        armored_signature: &'a [u8],
    ) -> Self {
        Self {
            principal,
            public_key_line,
            armored_signature,
        }
    }
}

/// Cryptographically verified admission signatures bound to one canonical
/// manifest and one exact signer-policy state.
///
/// This is deliberately not an admitted qualifier root. A future admission
/// tranche must additionally establish human/governance authorization,
/// environment/evidence prerequisites, currentness, durable provenance and
/// anti-rollback state.
#[derive(Debug)]
pub(super) struct V2VerifiedAdmissionSignatureSet {
    manifest_commitment: [u8; 32],
    signer_policy_commitment: [u8; 32],
    signer_policy_sequence: u64,
    verified_principals: Vec<String>,
    commitment: [u8; 32],
}

impl V2VerifiedAdmissionSignatureSet {
    pub(super) const fn manifest_commitment(&self) -> [u8; 32] {
        self.manifest_commitment
    }

    pub(super) const fn signer_policy_commitment(&self) -> [u8; 32] {
        self.signer_policy_commitment
    }

    pub(super) const fn signer_policy_sequence(&self) -> u64 {
        self.signer_policy_sequence
    }

    pub(super) fn verified_principals(&self) -> &[String] {
        &self.verified_principals
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2OpenSshAdmissionVerifierError {
    ManifestRejected,
    ManifestPolicySequenceMismatch,
    ManifestPolicyCommitmentMismatch,
    InsufficientSignatures,
    DuplicateSigner,
    UnknownSigner,
    MalformedPublicKey,
    KeyTypeMismatch,
    PublicKeyDigestMismatch,
    SignatureMetadataRejected,
    ExternalHashUnavailable,
    ExternalHashFailed,
    OpenSshUnavailable,
    SignatureRejected,
    TemporaryIo,
}

pub(super) fn verify_openssh_admission_signatures(
    manifest_bytes: &[u8],
    policy: &V2QualifierSignerPolicy,
    signatures: &[V2OpenSshAdmissionSignature<'_>],
) -> Result<V2VerifiedAdmissionSignatureSet, V2OpenSshAdmissionVerifierError> {
    let manifest = parse_canonical_admission_manifest(manifest_bytes)
        .map_err(|_| V2OpenSshAdmissionVerifierError::ManifestRejected)?;
    verify_manifest_policy_binding(manifest_bytes, policy)?;

    if signatures.len() < usize::from(policy.threshold()) {
        return Err(V2OpenSshAdmissionVerifierError::InsufficientSignatures);
    }

    let mut verified = BTreeSet::new();
    for signature in signatures {
        if !verified.insert(signature.principal.to_string()) {
            return Err(V2OpenSshAdmissionVerifierError::DuplicateSigner);
        }

        let policy_signer = policy
            .signers()
            .iter()
            .find(|candidate| candidate.principal() == signature.principal)
            .ok_or(V2OpenSshAdmissionVerifierError::UnknownSigner)?;

        let (key_type, key_material) = canonical_public_key_material(signature.public_key_line)?;
        verify_policy_key(policy_signer, &key_type, key_material.as_bytes())?;
        validate_admission_sshsig_v1(
            signature.armored_signature,
            &key_material,
            &key_type,
            V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
        )
        .map_err(|_| V2OpenSshAdmissionVerifierError::SignatureMetadataRejected)?;
        verify_with_openssh(
            manifest_bytes,
            signature.principal,
            &key_material,
            signature.armored_signature,
        )?;
    }

    if verified.len() < usize::from(policy.threshold()) {
        return Err(V2OpenSshAdmissionVerifierError::InsufficientSignatures);
    }

    let verified_principals: Vec<String> = verified.into_iter().collect();
    let commitment = verified_signature_set_commitment(
        manifest.commitment(),
        policy.sequence(),
        policy.commitment(),
        &verified_principals,
    );

    Ok(V2VerifiedAdmissionSignatureSet {
        manifest_commitment: manifest.commitment(),
        signer_policy_commitment: policy.commitment(),
        signer_policy_sequence: policy.sequence(),
        verified_principals,
        commitment,
    })
}

fn verify_manifest_policy_binding(
    manifest_bytes: &[u8],
    policy: &V2QualifierSignerPolicy,
) -> Result<(), V2OpenSshAdmissionVerifierError> {
    let text = std::str::from_utf8(manifest_bytes)
        .map_err(|_| V2OpenSshAdmissionVerifierError::ManifestRejected)?;
    let sequence = exact_field(text, "signer_policy_sequence")?
        .parse::<u64>()
        .map_err(|_| V2OpenSshAdmissionVerifierError::ManifestRejected)?;
    if sequence != policy.sequence() {
        return Err(V2OpenSshAdmissionVerifierError::ManifestPolicySequenceMismatch);
    }

    let commitment = exact_field(text, "signer_policy_commitment")?;
    if commitment != hex32(policy.commitment()) {
        return Err(V2OpenSshAdmissionVerifierError::ManifestPolicyCommitmentMismatch);
    }
    Ok(())
}

fn exact_field<'a>(
    text: &'a str,
    key: &str,
) -> Result<&'a str, V2OpenSshAdmissionVerifierError> {
    let prefix = format!("{key}=");
    let mut matches = text.lines().filter_map(|line| line.strip_prefix(&prefix));
    let value = matches
        .next()
        .ok_or(V2OpenSshAdmissionVerifierError::ManifestRejected)?;
    if matches.next().is_some() || value.is_empty() {
        return Err(V2OpenSshAdmissionVerifierError::ManifestRejected);
    }
    Ok(value)
}

fn canonical_public_key_material(
    public_key_line: &str,
) -> Result<(String, String), V2OpenSshAdmissionVerifierError> {
    let line = public_key_line.strip_suffix('\n').unwrap_or(public_key_line);
    if line.contains('\n') || line.contains('\r') {
        return Err(V2OpenSshAdmissionVerifierError::MalformedPublicKey);
    }

    let mut fields = line.split_ascii_whitespace();
    let key_type = fields
        .next()
        .ok_or(V2OpenSshAdmissionVerifierError::MalformedPublicKey)?;
    let base64_blob = fields
        .next()
        .ok_or(V2OpenSshAdmissionVerifierError::MalformedPublicKey)?;
    if key_type.is_empty()
        || base64_blob.is_empty()
        || !base64_blob.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'=')
        })
    {
        return Err(V2OpenSshAdmissionVerifierError::MalformedPublicKey);
    }

    Ok((
        key_type.to_string(),
        format!("{key_type} {base64_blob}"),
    ))
}

fn verify_policy_key(
    signer: &V2GovernanceSigner,
    key_type: &str,
    key_material: &[u8],
) -> Result<(), V2OpenSshAdmissionVerifierError> {
    if key_type != signer.key_type() {
        return Err(V2OpenSshAdmissionVerifierError::KeyTypeMismatch);
    }
    let digest = sha256_external(key_material)?;
    if digest != signer.public_key_material_sha256() {
        return Err(V2OpenSshAdmissionVerifierError::PublicKeyDigestMismatch);
    }
    Ok(())
}

fn sha256_external(input: &[u8]) -> Result<[u8; 32], V2OpenSshAdmissionVerifierError> {
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| V2OpenSshAdmissionVerifierError::ExternalHashUnavailable)?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or(V2OpenSshAdmissionVerifierError::ExternalHashFailed)?;
    stdin
        .write_all(input)
        .map_err(|_| V2OpenSshAdmissionVerifierError::ExternalHashFailed)?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .map_err(|_| V2OpenSshAdmissionVerifierError::ExternalHashFailed)?;
    if !output.status.success() {
        return Err(V2OpenSshAdmissionVerifierError::ExternalHashFailed);
    }
    let stdout = std::str::from_utf8(&output.stdout)
        .map_err(|_| V2OpenSshAdmissionVerifierError::ExternalHashFailed)?;
    let digest = stdout
        .split_ascii_whitespace()
        .next()
        .ok_or(V2OpenSshAdmissionVerifierError::ExternalHashFailed)?;
    decode_hex32(digest).ok_or(V2OpenSshAdmissionVerifierError::ExternalHashFailed)
}

fn verify_with_openssh(
    manifest_bytes: &[u8],
    principal: &str,
    key_material: &str,
    armored_signature: &[u8],
) -> Result<(), V2OpenSshAdmissionVerifierError> {
    let temp_dir = temporary_verification_dir();
    fs::create_dir(&temp_dir).map_err(|_| V2OpenSshAdmissionVerifierError::TemporaryIo)?;
    let allowed_signers = temp_dir.join("allowed_signers");
    let signature_path = temp_dir.join("admission.sig");

    let result = (|| {
        fs::write(
            &allowed_signers,
            format!("{principal} {key_material}\n"),
        )
        .map_err(|_| V2OpenSshAdmissionVerifierError::TemporaryIo)?;
        fs::write(&signature_path, armored_signature)
            .map_err(|_| V2OpenSshAdmissionVerifierError::TemporaryIo)?;

        let mut child = Command::new("ssh-keygen")
            .args(["-Y", "verify", "-f"])
            .arg(&allowed_signers)
            .args([
                "-I",
                principal,
                "-n",
                V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
                "-s",
            ])
            .arg(&signature_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| V2OpenSshAdmissionVerifierError::OpenSshUnavailable)?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or(V2OpenSshAdmissionVerifierError::SignatureRejected)?;
        stdin
            .write_all(manifest_bytes)
            .map_err(|_| V2OpenSshAdmissionVerifierError::SignatureRejected)?;
        drop(stdin);

        let status = child
            .wait()
            .map_err(|_| V2OpenSshAdmissionVerifierError::SignatureRejected)?;
        if !status.success() {
            return Err(V2OpenSshAdmissionVerifierError::SignatureRejected);
        }
        Ok(())
    })();

    let _ = fs::remove_dir_all(&temp_dir);
    result
}

fn temporary_verification_dir() -> PathBuf {
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "symthaea-eureka-v2-sshsig-{}-{id}",
        std::process::id()
    ))
}

fn verified_signature_set_commitment(
    manifest_commitment: [u8; 32],
    signer_policy_sequence: u64,
    signer_policy_commitment: [u8; 32],
    principals: &[String],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_VERIFIED_ADMISSION_SIGNATURE_SET_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&manifest_commitment);
    bytes.extend_from_slice(&signer_policy_sequence.to_le_bytes());
    bytes.extend_from_slice(&signer_policy_commitment);
    bytes.extend_from_slice(&(principals.len() as u64).to_le_bytes());
    for principal in principals {
        encode_bytes(&mut bytes, principal.as_bytes());
    }
    *blake3::hash(&bytes).as_bytes()
}

fn decode_hex32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0])?;
        let low = hex_nibble(chunk[1])?;
        output[index] = (high << 4) | low;
    }
    Some(output)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
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

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &[u8] =
        include_bytes!("fixtures/eureka-v2-signed-admission-2of2.env");
    const ALPHA_PUB: &str =
        include_str!("fixtures/eureka-v2-signed-admission-alpha.pub");
    const BETA_PUB: &str =
        include_str!("fixtures/eureka-v2-signed-admission-beta.pub");
    const ALPHA_SIG: &[u8] =
        include_bytes!("fixtures/eureka-v2-signed-admission-alpha.sig");
    const BETA_SIG: &[u8] =
        include_bytes!("fixtures/eureka-v2-signed-admission-beta.sig");
    const ALPHA_WRONG_NAMESPACE_SIG: &[u8] = include_bytes!(
        "fixtures/eureka-v2-signed-admission-alpha-wrong-namespace.sig"
    );

    const ALPHA_PRINCIPAL: &str = "fixture-alpha@example.invalid";
    const BETA_PRINCIPAL: &str = "fixture-beta@example.invalid";
    const ALPHA_KEY_SHA256: &str =
        "b82ecaf327614d6054d3ea0b36456a01411b16dd33cea2ffccd09a95be84bf2a";
    const BETA_KEY_SHA256: &str =
        "cb097f5b4a59e052c5e9c1389e0becc74418c850665990b66f74c6b740c376fd";
    const POLICY_COMMITMENT: &str =
        "968816409929b3885121e6ac26b5284a463e370c5b5f7c84bbc7d90120e674c2";

    fn fixture_policy(threshold: u16) -> V2QualifierSignerPolicy {
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
        V2QualifierSignerPolicy::genesis(1, threshold, vec![beta, alpha]).unwrap()
    }

    fn signatures() -> [V2OpenSshAdmissionSignature<'static>; 2] {
        [
            V2OpenSshAdmissionSignature::new(ALPHA_PRINCIPAL, ALPHA_PUB, ALPHA_SIG),
            V2OpenSshAdmissionSignature::new(BETA_PRINCIPAL, BETA_PUB, BETA_SIG),
        ]
    }

    #[test]
    fn fixture_policy_commitment_matches_signed_manifest_exactly() {
        let policy = fixture_policy(2);
        assert_eq!(hex32(policy.commitment()), POLICY_COMMITMENT);
        verify_manifest_policy_binding(MANIFEST, &policy).unwrap();
    }

    #[test]
    fn two_of_two_namespaced_signatures_verify_and_order_is_canonical() {
        let policy = fixture_policy(2);
        let first =
            verify_openssh_admission_signatures(MANIFEST, &policy, &signatures()).unwrap();
        let reversed = [signatures()[1], signatures()[0]];
        let second =
            verify_openssh_admission_signatures(MANIFEST, &policy, &reversed).unwrap();

        assert_eq!(first.signer_policy_sequence(), 1);
        assert_eq!(first.signer_policy_commitment(), policy.commitment());
        assert_eq!(first.manifest_commitment(), second.manifest_commitment());
        assert_eq!(first.commitment(), second.commitment());
        assert_eq!(
            first.verified_principals(),
            &[ALPHA_PRINCIPAL.to_string(), BETA_PRINCIPAL.to_string()]
        );
        assert_ne!(first.commitment(), [0_u8; 32]);
    }

    #[test]
    fn threshold_is_enforced_after_manifest_policy_binding() {
        let policy = fixture_policy(2);
        let only_alpha = [V2OpenSshAdmissionSignature::new(
            ALPHA_PRINCIPAL,
            ALPHA_PUB,
            ALPHA_SIG,
        )];
        assert_eq!(
            verify_openssh_admission_signatures(MANIFEST, &policy, &only_alpha).unwrap_err(),
            V2OpenSshAdmissionVerifierError::InsufficientSignatures
        );
    }

    #[test]
    fn duplicate_signer_cannot_satisfy_threshold_twice() {
        let policy = fixture_policy(2);
        let duplicated = [
            V2OpenSshAdmissionSignature::new(ALPHA_PRINCIPAL, ALPHA_PUB, ALPHA_SIG),
            V2OpenSshAdmissionSignature::new(ALPHA_PRINCIPAL, ALPHA_PUB, ALPHA_SIG),
        ];
        assert_eq!(
            verify_openssh_admission_signatures(MANIFEST, &policy, &duplicated).unwrap_err(),
            V2OpenSshAdmissionVerifierError::DuplicateSigner
        );
    }

    #[test]
    fn wrong_namespace_signature_is_rejected_before_crypto_verification() {
        let policy = fixture_policy(2);
        let wrong = [
            V2OpenSshAdmissionSignature::new(
                ALPHA_PRINCIPAL,
                ALPHA_PUB,
                ALPHA_WRONG_NAMESPACE_SIG,
            ),
            V2OpenSshAdmissionSignature::new(BETA_PRINCIPAL, BETA_PUB, BETA_SIG),
        ];
        assert_eq!(
            verify_openssh_admission_signatures(MANIFEST, &policy, &wrong).unwrap_err(),
            V2OpenSshAdmissionVerifierError::SignatureMetadataRejected
        );
    }

    #[test]
    fn valid_signature_over_different_canonical_bytes_is_rejected() {
        let policy = fixture_policy(2);
        let tampered = std::str::from_utf8(MANIFEST)
            .unwrap()
            .replacen(
                "subject_head=1111111111111111111111111111111111111111",
                "subject_head=3111111111111111111111111111111111111111",
                1,
            );
        assert!(parse_canonical_admission_manifest(tampered.as_bytes()).is_ok());
        assert_eq!(
            verify_openssh_admission_signatures(tampered.as_bytes(), &policy, &signatures())
                .unwrap_err(),
            V2OpenSshAdmissionVerifierError::SignatureRejected
        );
    }

    #[test]
    fn manifest_must_name_exact_signer_policy_before_signatures_count() {
        let one_of_two = fixture_policy(1);
        assert_ne!(hex32(one_of_two.commitment()), POLICY_COMMITMENT);
        assert_eq!(
            verify_openssh_admission_signatures(MANIFEST, &one_of_two, &signatures())
                .unwrap_err(),
            V2OpenSshAdmissionVerifierError::ManifestPolicyCommitmentMismatch
        );
    }

    #[test]
    fn unknown_principal_and_wrong_public_key_fail_before_threshold() {
        let policy = fixture_policy(2);
        let unknown = [
            V2OpenSshAdmissionSignature::new(
                "fixture-unknown@example.invalid",
                ALPHA_PUB,
                ALPHA_SIG,
            ),
            signatures()[1],
        ];
        assert_eq!(
            verify_openssh_admission_signatures(MANIFEST, &policy, &unknown).unwrap_err(),
            V2OpenSshAdmissionVerifierError::UnknownSigner
        );

        let wrong_key = ALPHA_PUB.replacen('A', "B", 1);
        let wrong_key_bundle = [
            V2OpenSshAdmissionSignature::new(ALPHA_PRINCIPAL, &wrong_key, ALPHA_SIG),
            signatures()[1],
        ];
        assert_eq!(
            verify_openssh_admission_signatures(MANIFEST, &policy, &wrong_key_bundle)
                .unwrap_err(),
            V2OpenSshAdmissionVerifierError::PublicKeyDigestMismatch
        );
    }

    #[test]
    fn verifier_source_has_no_signing_root_admission_or_execution_surface() {
        let production = include_str!("v2_openssh_admission_verifier.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "-Y sign",
            "PRIVATE KEY",
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden verifier production surface: {forbidden}"
            );
        }
        assert!(production.contains("validate_admission_sshsig_v1"));
        assert!(production.contains("-Y\", \"verify"));
        assert!(production.contains("V2_QUALIFIER_ADMISSION_SSH_NAMESPACE"));
    }
}
