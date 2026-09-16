// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent signed-narinfo graph cross-check for a qualified Nix runtime closure.
//!
//! Nix signs the ValidPathInfo fingerprint
//! `1;StorePath;NarHash;NarSize;References`. This crate reproduces that byte
//! string and Nix's Ed25519 key/signature encoding directly in Rust, without a
//! Nix subprocess at verification time. The public #3245 entry vector is first
//! rebound to its opaque closure identity before any signed-capsule comparison.
//!
//! Positive claims are deliberately bounded to one exact reviewed signed
//! capsule. The capsule is not claimed to be globally latest/current, and Nix
//! content-address, transport, compression, URL, time, or cache-availability
//! metadata are not promoted into the signed theorem.

#![deny(unsafe_code)]

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use symthaea_assurance_nix_runtime_closure::{
    NixClosureEntry, NixRuntimeClosurePolicy, NixRuntimeClosureQualification,
};

pub const SIGNED_NARINFO_CLOSURE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.signed-narinfo-runtime-closure-policy.v1";
pub const SIGNED_NARINFO_CLOSURE_CAPSULE_SCHEMA_V1: &str =
    "symthaea.assurance.signed-narinfo-runtime-closure-capsule.v1";
pub const SIGNED_NARINFO_CLOSURE_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.signed-narinfo-runtime-closure-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-runtime-closure-policy.digest.v1\0";
const CAPSULE_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-runtime-closure-capsule.digest.v1\0";
const SIGNED_GRAPH_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-runtime-closure-graph.digest.v1\0";
const TRUSTED_KEYS_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-runtime-closure-trusted-keys.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-runtime-closure-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-runtime-closure-qualification.digest.v1\0";

// Must stay byte-identical to #3245 while rebinding its public entry vector.
const LOCAL_CLOSURE_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-runtime-closure-content.digest.v1\0";

const MAX_TEXT_BYTES: usize = 4096;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_TRUSTED_KEYS: usize = 64;
const MAX_SIGNATURES_PER_RECORD: usize = 128;
const MAX_PATHS_HARD: u32 = 65_536;
const MAX_TOTAL_NAR_BYTES_HARD: u64 = 1u64 << 50;
const NIX_BASE32: &[u8] = b"0123456789abcdfghijklmnpqrsvwxyz";
const SHA256_NIX32_LEN: usize = 52;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoRuntimeClosurePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub root_store_path: String,
    pub expected_local_closure_policy_digest: String,
    pub trusted_public_keys: Vec<String>,
    pub signatures_needed: u8,
    pub max_paths: u32,
    pub max_total_nar_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl SignedNarInfoRuntimeClosurePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == SIGNED_NARINFO_CLOSURE_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_store_object_path(&self.root_store_path)
            && valid_blake3(&self.expected_local_closure_policy_digest)
            && !self.trusted_public_keys.is_empty()
            && self.trusted_public_keys.len() <= MAX_TRUSTED_KEYS
            && self.signatures_needed > 0
            && usize::from(self.signatures_needed) <= self.trusted_public_keys.len()
            && parse_trusted_keys(&self.trusted_public_keys).is_some()
            && (1..=MAX_PATHS_HARD).contains(&self.max_paths)
            && (1..=MAX_TOTAL_NAR_BYTES_HARD).contains(&self.max_total_nar_bytes)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut h = blake3::Hasher::new();
        h.update(POLICY_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.root_store_path.as_str(),
            self.expected_local_closure_policy_digest.as_str(),
        ] {
            field_le(&mut h, value);
        }
        let mut keys = self.trusted_public_keys.clone();
        keys.sort();
        h.update(&(keys.len() as u64).to_le_bytes());
        for key in keys {
            field_le(&mut h, &key);
        }
        h.update(&[self.signatures_needed]);
        h.update(&self.max_paths.to_le_bytes());
        h.update(&self.max_total_nar_bytes.to_le_bytes());
        sorted_strings_le(&mut h, &self.evidence_refs);
        Some(blake3_text(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoRecord {
    pub store_path: String,
    /// Exact Nix fingerprint spelling: `sha256:<52 Nix32 characters>`.
    pub nar_hash: String,
    pub nar_size: u64,
    /// Full `/nix/store/...` paths. Order is nonsemantic; Nix signs StorePathSet order.
    pub references: Vec<String>,
    /// Nix detached signatures: `<key-name>:<base64(64 raw Ed25519 bytes)>`.
    pub signatures: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoClosureCapsule {
    pub schema_version: String,
    pub capsule_id: String,
    pub records: Vec<SignedNarInfoRecord>,
    pub evidence_refs: Vec<String>,
}

impl SignedNarInfoClosureCapsule {
    pub fn validate_shape(&self, maximum_paths: u32) -> bool {
        self.schema_version == SIGNED_NARINFO_CLOSURE_CAPSULE_SCHEMA_V1
            && canonical_text(&self.capsule_id)
            && !self.records.is_empty()
            && self.records.len() <= maximum_paths as usize
            && self.records.len() <= MAX_PATHS_HARD as usize
            && valid_refs(&self.evidence_refs)
            && unique_by(self.records.iter().map(|record| record.store_path.as_str()))
            && self.records.iter().all(|record| {
                valid_store_object_path(&record.store_path)
                    && parse_signed_nar_hash(&record.nar_hash).is_some()
                    && record.nar_size > 0
                    && record.references.iter().all(|value| valid_store_object_path(value))
                    && unique_by(record.references.iter().map(String::as_str))
                    && !record.signatures.is_empty()
                    && record.signatures.len() <= MAX_SIGNATURES_PER_RECORD
                    && unique_by(record.signatures.iter().map(String::as_str))
                    && record
                        .signatures
                        .iter()
                        .all(|value| parse_named_base64(value, 64).is_some())
            })
    }

    pub fn canonical_digest(&self, maximum_paths: u32) -> Option<String> {
        if !self.validate_shape(maximum_paths) {
            return None;
        }
        let mut h = blake3::Hasher::new();
        h.update(CAPSULE_DOMAIN);
        field_le(&mut h, &self.schema_version);
        field_le(&mut h, &self.capsule_id);
        let mut records = self.records.iter().collect::<Vec<_>>();
        records.sort_by(|left, right| left.store_path.cmp(&right.store_path));
        h.update(&(records.len() as u64).to_le_bytes());
        for record in records {
            field_le(&mut h, &record.store_path);
            field_le(&mut h, &record.nar_hash);
            h.update(&record.nar_size.to_le_bytes());
            sorted_strings_le(&mut h, &record.references);
            sorted_strings_le(&mut h, &record.signatures);
        }
        sorted_strings_le(&mut h, &self.evidence_refs);
        Some(blake3_text(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignedNarInfoClosureIssue {
    InvalidPolicy,
    InvalidCapsule,
    LocalPolicyDigestMismatch,
    LocalRootMismatch,
    LocalTrustedKeysMismatch,
    LocalSignatureThresholdMismatch,
    LocalPublicClosureDigestMismatch,
    LocalPathCountMismatch,
    LocalTotalNarBytesOverflow,
    LocalTotalNarBytesMismatch,
    TooManyPaths { observed: u64, maximum: u32 },
    TotalNarBytesOverflow,
    TotalNarBytesExceeded { observed: u64, maximum: u64 },
    SignedRootMissing,
    SignedPathSetMismatch,
    SignedReferenceMissing { from: String, reference: String },
    SignedUnreachablePath(String),
    NarHashMismatch(String),
    NarSizeMismatch(String),
    ReferenceSetMismatch(String),
    MalformedSignature { path: String, key_name: String },
    InvalidTrustedSignature { path: String, key_name: String },
    SignatureThresholdNotMet {
        path: String,
        observed: u64,
        required: u8,
    },
}

impl SignedNarInfoClosureIssue {
    fn invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::InvalidCapsule
                | Self::LocalPolicyDigestMismatch
                | Self::LocalRootMismatch
                | Self::LocalTrustedKeysMismatch
                | Self::LocalSignatureThresholdMismatch
                | Self::LocalPublicClosureDigestMismatch
                | Self::LocalPathCountMismatch
                | Self::LocalTotalNarBytesOverflow
                | Self::LocalTotalNarBytesMismatch
                | Self::TooManyPaths { .. }
                | Self::TotalNarBytesOverflow
                | Self::SignedRootMissing
                | Self::SignedPathSetMismatch
                | Self::SignedReferenceMissing { .. }
                | Self::SignedUnreachablePath(_)
                | Self::NarHashMismatch(_)
                | Self::NarSizeMismatch(_)
                | Self::ReferenceSetMismatch(_)
                | Self::MalformedSignature { .. }
                | Self::InvalidTrustedSignature { .. }
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidCapsule => "invalid-capsule".into(),
            Self::LocalPolicyDigestMismatch => "local-policy-digest-mismatch".into(),
            Self::LocalRootMismatch => "local-root-mismatch".into(),
            Self::LocalTrustedKeysMismatch => "local-trusted-keys-mismatch".into(),
            Self::LocalSignatureThresholdMismatch => "local-signature-threshold-mismatch".into(),
            Self::LocalPublicClosureDigestMismatch => "local-public-closure-digest-mismatch".into(),
            Self::LocalPathCountMismatch => "local-path-count-mismatch".into(),
            Self::LocalTotalNarBytesOverflow => "local-total-nar-bytes-overflow".into(),
            Self::LocalTotalNarBytesMismatch => "local-total-nar-bytes-mismatch".into(),
            Self::TooManyPaths { observed, maximum } => {
                format!("too-many-paths:{observed}:{maximum}")
            }
            Self::TotalNarBytesOverflow => "total-nar-bytes-overflow".into(),
            Self::TotalNarBytesExceeded { observed, maximum } => {
                format!("total-nar-bytes-exceeded:{observed}:{maximum}")
            }
            Self::SignedRootMissing => "signed-root-missing".into(),
            Self::SignedPathSetMismatch => "signed-path-set-mismatch".into(),
            Self::SignedReferenceMissing { from, reference } => {
                format!("signed-reference-missing:{from}:{reference}")
            }
            Self::SignedUnreachablePath(path) => format!("signed-unreachable-path:{path}"),
            Self::NarHashMismatch(path) => format!("nar-hash-mismatch:{path}"),
            Self::NarSizeMismatch(path) => format!("nar-size-mismatch:{path}"),
            Self::ReferenceSetMismatch(path) => format!("reference-set-mismatch:{path}"),
            Self::MalformedSignature { path, key_name } => {
                format!("malformed-signature:{path}:{key_name}")
            }
            Self::InvalidTrustedSignature { path, key_name } => {
                format!("invalid-trusted-signature:{path}:{key_name}")
            }
            Self::SignatureThresholdNotMet {
                path,
                observed,
                required,
            } => format!("signature-threshold-not-met:{path}:{observed}:{required}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignedNarInfoClosureDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoClosureReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub capsule_id: String,
    pub capsule_digest: Option<String>,
    pub root_store_path: String,
    pub local_closure_policy_digest: String,
    pub local_closure_qualification_digest: String,
    pub local_closure_digest: String,
    pub rebound_local_closure_digest: Option<String>,
    pub signed_graph_digest: Option<String>,
    pub trusted_key_set_digest: Option<String>,
    pub path_count: u64,
    pub total_nar_bytes: u64,
    pub minimum_valid_trusted_signers: u64,
    pub signatures_needed: u8,
    pub disposition: SignedNarInfoClosureDisposition,
    pub issues: Vec<SignedNarInfoClosureIssue>,
}

impl SignedNarInfoClosureReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.capsule_id.as_str(),
            self.capsule_digest.as_deref().unwrap_or("-"),
            self.root_store_path.as_str(),
            self.local_closure_policy_digest.as_str(),
            self.local_closure_qualification_digest.as_str(),
            self.local_closure_digest.as_str(),
            self.rebound_local_closure_digest.as_deref().unwrap_or("-"),
            self.signed_graph_digest.as_deref().unwrap_or("-"),
            self.trusted_key_set_digest.as_deref().unwrap_or("-"),
        ] {
            field_le(&mut h, value);
        }
        h.update(&self.path_count.to_le_bytes());
        h.update(&self.total_nar_bytes.to_le_bytes());
        h.update(&self.minimum_valid_trusted_signers.to_le_bytes());
        h.update(&[self.signatures_needed]);
        field_le(
            &mut h,
            match self.disposition {
                SignedNarInfoClosureDisposition::Invalid => "invalid",
                SignedNarInfoClosureDisposition::Blocked => "blocked",
                SignedNarInfoClosureDisposition::Qualified => "qualified",
            },
        );
        h.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field_le(&mut h, &issue.code());
        }
        blake3_text(h.finalize())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedNarInfoBackedNixRuntimeClosure {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    capsule_digest: String,
    local_closure_qualification_digest: String,
    local_closure_digest: String,
    signed_graph_digest: String,
    trusted_key_set_digest: String,
    root_store_path: String,
    path_count: u64,
    total_nar_bytes: u64,
    signatures_needed: u8,
}

impl SignedNarInfoBackedNixRuntimeClosure {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn capsule_digest(&self) -> &str { &self.capsule_digest }
    pub fn local_closure_qualification_digest(&self) -> &str {
        &self.local_closure_qualification_digest
    }
    pub fn local_closure_digest(&self) -> &str { &self.local_closure_digest }
    pub fn signed_graph_digest(&self) -> &str { &self.signed_graph_digest }
    pub fn trusted_key_set_digest(&self) -> &str { &self.trusted_key_set_digest }
    pub fn root_store_path(&self) -> &str { &self.root_store_path }
    pub const fn path_count(&self) -> u64 { self.path_count }
    pub const fn total_nar_bytes(&self) -> u64 { self.total_nar_bytes }
    pub const fn signatures_needed(&self) -> u8 { self.signatures_needed }

    pub const fn local_public_closure_rebound_to_opaque_identity(&self) -> bool { true }
    pub const fn every_path_nar_identity_signed_by_reviewed_threshold(&self) -> bool { true }
    pub const fn signed_reference_graph_matches_local_closure(&self) -> bool { true }
    pub const fn local_graph_cross_checked_against_signed_external_graph(&self) -> bool { true }
    pub const fn local_graph_non_equivocation_against_this_capsule(&self) -> bool { true }
    pub const fn content_address_metadata_covered_by_external_signatures(&self) -> bool { false }
    pub const fn signed_capsule_global_currentness_established(&self) -> bool { false }
    pub const fn nix_database_global_currentness_established(&self) -> bool { false }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedNarInfoClosureQualification {
    pub report: SignedNarInfoClosureReport,
    verified: SignedNarInfoBackedNixRuntimeClosure,
}

impl SignedNarInfoClosureQualification {
    pub fn verified(&self) -> &SignedNarInfoBackedNixRuntimeClosure { &self.verified }
    pub fn into_verified(self) -> SignedNarInfoBackedNixRuntimeClosure { self.verified }
}

#[derive(Debug, Clone)]
struct VerifiedRecord {
    store_path: String,
    nar_hash: String,
    nar_size: u64,
    references: Vec<String>,
    valid_signers: Vec<String>,
}

pub fn qualify_signed_narinfo_runtime_closure(
    policy: &SignedNarInfoRuntimeClosurePolicy,
    local_policy: &NixRuntimeClosurePolicy,
    local: &NixRuntimeClosureQualification,
    capsule: &SignedNarInfoClosureCapsule,
) -> Result<SignedNarInfoClosureQualification, SignedNarInfoClosureReport> {
    let policy_digest = policy.canonical_digest();
    let capsule_digest = capsule.canonical_digest(policy.max_paths);
    let local_verified = local.verified();
    let mut report = SignedNarInfoClosureReport {
        schema_version: SIGNED_NARINFO_CLOSURE_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        capsule_id: capsule.capsule_id.clone(),
        capsule_digest: capsule_digest.clone(),
        root_store_path: policy.root_store_path.clone(),
        local_closure_policy_digest: local_verified.policy_digest().into(),
        local_closure_qualification_digest: local_verified.qualification_digest().into(),
        local_closure_digest: local_verified.closure_digest().into(),
        rebound_local_closure_digest: None,
        signed_graph_digest: None,
        trusted_key_set_digest: None,
        path_count: 0,
        total_nar_bytes: 0,
        minimum_valid_trusted_signers: 0,
        signatures_needed: policy.signatures_needed,
        disposition: SignedNarInfoClosureDisposition::Invalid,
        issues: Vec::new(),
    };

    if policy_digest.is_none() {
        report.issues.push(SignedNarInfoClosureIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if capsule_digest.is_none() {
        report.issues.push(SignedNarInfoClosureIssue::InvalidCapsule);
        return Err(finalize(report));
    }

    let local_policy_digest = local_policy.canonical_digest();
    if local_policy_digest.as_deref() != Some(policy.expected_local_closure_policy_digest.as_str())
        || local_verified.policy_digest() != policy.expected_local_closure_policy_digest
    {
        report
            .issues
            .push(SignedNarInfoClosureIssue::LocalPolicyDigestMismatch);
    }
    if local_policy.root_store_path != policy.root_store_path
        || local_verified.root_store_path() != policy.root_store_path
    {
        report.issues.push(SignedNarInfoClosureIssue::LocalRootMismatch);
    }
    if sorted_copy(&local_policy.trusted_public_keys) != sorted_copy(&policy.trusted_public_keys) {
        report
            .issues
            .push(SignedNarInfoClosureIssue::LocalTrustedKeysMismatch);
    }
    if local_policy.signatures_needed != policy.signatures_needed {
        report
            .issues
            .push(SignedNarInfoClosureIssue::LocalSignatureThresholdMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let rebound = local_closure_digest(&policy.root_store_path, &local.entries);
    report.rebound_local_closure_digest = Some(rebound.clone());
    if rebound != local_verified.closure_digest() {
        report
            .issues
            .push(SignedNarInfoClosureIssue::LocalPublicClosureDigestMismatch);
    }
    if local.entries.len() as u64 != local_verified.path_count() {
        report
            .issues
            .push(SignedNarInfoClosureIssue::LocalPathCountMismatch);
    }
    let local_total = match total_nar_bytes(&local.entries) {
        Some(value) => value,
        None => {
            report
                .issues
                .push(SignedNarInfoClosureIssue::LocalTotalNarBytesOverflow);
            return Err(finalize(report));
        }
    };
    if local_total != local_verified.total_nar_bytes() {
        report
            .issues
            .push(SignedNarInfoClosureIssue::LocalTotalNarBytesMismatch);
    }
    if local.entries.len() > policy.max_paths as usize {
        report.issues.push(SignedNarInfoClosureIssue::TooManyPaths {
            observed: local.entries.len() as u64,
            maximum: policy.max_paths,
        });
    }
    if local_total > policy.max_total_nar_bytes {
        report
            .issues
            .push(SignedNarInfoClosureIssue::TotalNarBytesExceeded {
                observed: local_total,
                maximum: policy.max_total_nar_bytes,
            });
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let trusted = parse_trusted_keys(&policy.trusted_public_keys).expect("validated policy keys");
    let trusted_key_set_digest = trusted_keys_digest(&policy.trusted_public_keys);
    report.trusted_key_set_digest = Some(trusted_key_set_digest.clone());

    let local_by_path = local
        .entries
        .iter()
        .map(|entry| (entry.store_path.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    let signed_by_path = capsule
        .records
        .iter()
        .map(|record| (record.store_path.as_str(), record))
        .collect::<BTreeMap<_, _>>();

    if !signed_by_path.contains_key(policy.root_store_path.as_str()) {
        report.issues.push(SignedNarInfoClosureIssue::SignedRootMissing);
    }
    if local_by_path.keys().copied().collect::<BTreeSet<_>>()
        != signed_by_path.keys().copied().collect::<BTreeSet<_>>()
    {
        report
            .issues
            .push(SignedNarInfoClosureIssue::SignedPathSetMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let mut verified_records = Vec::with_capacity(capsule.records.len());
    let mut total = 0u64;
    let mut minimum_signers = u64::MAX;

    let mut ordered_records = capsule.records.iter().collect::<Vec<_>>();
    ordered_records.sort_by(|left, right| left.store_path.cmp(&right.store_path));
    for record in ordered_records {
        let local_entry = local_by_path
            .get(record.store_path.as_str())
            .expect("equal path sets established");

        let local_hash = parse_sha256_hash(&local_entry.nar_hash);
        let signed_hash = parse_signed_nar_hash(&record.nar_hash);
        if local_hash.is_none() || signed_hash.is_none() || local_hash != signed_hash {
            report
                .issues
                .push(SignedNarInfoClosureIssue::NarHashMismatch(record.store_path.clone()));
        }
        if local_entry.nar_size != record.nar_size {
            report
                .issues
                .push(SignedNarInfoClosureIssue::NarSizeMismatch(record.store_path.clone()));
        }
        if sorted_copy(&local_entry.references) != sorted_copy(&record.references) {
            report
                .issues
                .push(SignedNarInfoClosureIssue::ReferenceSetMismatch(record.store_path.clone()));
        }
        for reference in &record.references {
            if !signed_by_path.contains_key(reference.as_str()) {
                report
                    .issues
                    .push(SignedNarInfoClosureIssue::SignedReferenceMissing {
                        from: record.store_path.clone(),
                        reference: reference.clone(),
                    });
            }
        }
        if !report.issues.is_empty() {
            return Err(finalize(report));
        }

        total = match total.checked_add(record.nar_size) {
            Some(value) => value,
            None => {
                report.issues.push(SignedNarInfoClosureIssue::TotalNarBytesOverflow);
                return Err(finalize(report));
            }
        };

        let fingerprint = nix_fingerprint(record);
        let mut valid_signers = BTreeSet::new();
        for encoded_signature in &record.signatures {
            let Some((key_name, signature_bytes)) = parse_named_base64(encoded_signature, 64) else {
                report
                    .issues
                    .push(SignedNarInfoClosureIssue::MalformedSignature {
                        path: record.store_path.clone(),
                        key_name: signature_key_name(encoded_signature),
                    });
                return Err(finalize(report));
            };
            let Some(key) = trusted.get(key_name.as_str()) else {
                continue;
            };
            let signature_array: [u8; 64] = signature_bytes
                .as_slice()
                .try_into()
                .expect("signature length was checked");
            let signature = Signature::from_bytes(&signature_array);
            if key.verify(fingerprint.as_bytes(), &signature).is_err() {
                report
                    .issues
                    .push(SignedNarInfoClosureIssue::InvalidTrustedSignature {
                        path: record.store_path.clone(),
                        key_name,
                    });
                return Err(finalize(report));
            }
            valid_signers.insert(key_name);
        }
        if valid_signers.len() < usize::from(policy.signatures_needed) {
            report
                .issues
                .push(SignedNarInfoClosureIssue::SignatureThresholdNotMet {
                    path: record.store_path.clone(),
                    observed: valid_signers.len() as u64,
                    required: policy.signatures_needed,
                });
            return Err(finalize(report));
        }
        minimum_signers = minimum_signers.min(valid_signers.len() as u64);
        verified_records.push(VerifiedRecord {
            store_path: record.store_path.clone(),
            nar_hash: record.nar_hash.clone(),
            nar_size: record.nar_size,
            references: sorted_copy(&record.references),
            valid_signers: valid_signers.into_iter().collect(),
        });
    }

    if total > policy.max_total_nar_bytes {
        report
            .issues
            .push(SignedNarInfoClosureIssue::TotalNarBytesExceeded {
                observed: total,
                maximum: policy.max_total_nar_bytes,
            });
        return Err(finalize(report));
    }

    if let Some(unreachable) = first_unreachable(&policy.root_store_path, &verified_records) {
        report
            .issues
            .push(SignedNarInfoClosureIssue::SignedUnreachablePath(unreachable));
        return Err(finalize(report));
    }

    let signed_graph_digest = signed_graph_digest(&policy.root_store_path, &verified_records);
    report.signed_graph_digest = Some(signed_graph_digest.clone());
    report.path_count = verified_records.len() as u64;
    report.total_nar_bytes = total;
    report.minimum_valid_trusted_signers = minimum_signers;
    report.disposition = SignedNarInfoClosureDisposition::Qualified;

    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let capsule_digest = capsule_digest.expect("validated capsule has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        local_verified.qualification_digest(),
        local_verified.closure_digest(),
        &capsule_digest,
        &signed_graph_digest,
        &trusted_key_set_digest,
        policy.signatures_needed,
        &report_digest,
    );
    let verified = SignedNarInfoBackedNixRuntimeClosure {
        qualification_digest,
        report_digest,
        policy_digest,
        capsule_digest,
        local_closure_qualification_digest: local_verified.qualification_digest().into(),
        local_closure_digest: local_verified.closure_digest().into(),
        signed_graph_digest,
        trusted_key_set_digest,
        root_store_path: policy.root_store_path.clone(),
        path_count: verified_records.len() as u64,
        total_nar_bytes: total,
        signatures_needed: policy.signatures_needed,
    };
    Ok(SignedNarInfoClosureQualification { report, verified })
}

fn finalize(mut report: SignedNarInfoClosureReport) -> SignedNarInfoClosureReport {
    report.disposition = if report.issues.iter().any(SignedNarInfoClosureIssue::invalid) {
        SignedNarInfoClosureDisposition::Invalid
    } else {
        SignedNarInfoClosureDisposition::Blocked
    };
    report
}

fn parse_trusted_keys(values: &[String]) -> Option<BTreeMap<String, VerifyingKey>> {
    let mut keys = BTreeMap::new();
    for value in values {
        let (name, bytes) = parse_named_base64(value, 32)?;
        let raw: [u8; 32] = bytes.as_slice().try_into().ok()?;
        let key = VerifyingKey::from_bytes(&raw).ok()?;
        if keys.insert(name, key).is_some() {
            return None;
        }
    }
    Some(keys)
}

fn parse_named_base64(value: &str, expected_len: usize) -> Option<(String, Vec<u8>)> {
    let (name, encoded) = value.split_once(':')?;
    if !key_name(name) || encoded.is_empty() || encoded.chars().any(char::is_whitespace) {
        return None;
    }
    let decoded = BASE64.decode(encoded.as_bytes()).ok()?;
    (decoded.len() == expected_len).then(|| (name.to_string(), decoded))
}

fn key_name(value: &str) -> bool {
    canonical_text(value) && !value.contains(':') && !value.chars().any(char::is_whitespace)
}

fn signature_key_name(value: &str) -> String {
    value.split_once(':').map(|(name, _)| name).unwrap_or("?").to_string()
}

fn nix_fingerprint(record: &SignedNarInfoRecord) -> String {
    let references = sorted_copy(&record.references).join(",");
    format!(
        "1;{};{};{};{}",
        record.store_path, record.nar_hash, record.nar_size, references
    )
}

fn parse_signed_nar_hash(value: &str) -> Option<[u8; 32]> {
    let encoded = value.strip_prefix("sha256:")?;
    if encoded.len() != SHA256_NIX32_LEN {
        return None;
    }
    let decoded = decode_nix32(encoded)?;
    let raw: [u8; 32] = decoded.as_slice().try_into().ok()?;
    // Enforce canonical Nix32 spelling, not merely a decodable spelling.
    (encode_nix32(&raw) == encoded).then_some(raw)
}

fn parse_sha256_hash(value: &str) -> Option<[u8; 32]> {
    if value.starts_with("sha256:") {
        return parse_signed_nar_hash(value);
    }
    let encoded = value.strip_prefix("sha256-")?;
    let decoded = BASE64.decode(encoded.as_bytes()).ok()?;
    decoded.as_slice().try_into().ok()
}

// Byte-for-byte transcription of Nix BaseNix32's unusual reversed digit order.
fn decode_nix32(value: &str) -> Option<Vec<u8>> {
    if value.is_empty() {
        return Some(Vec::new());
    }
    let mut result = Vec::<u8>::with_capacity((value.len() * 5 + 7) / 8);
    let bytes = value.as_bytes();
    for n in 0..bytes.len() {
        let character = bytes[bytes.len() - n - 1];
        let digit = NIX_BASE32.iter().position(|candidate| *candidate == character)? as u8;
        let bit = n * 5;
        let index = bit / 8;
        let shift = bit % 8;
        if result.len() <= index {
            result.resize(index + 1, 0);
        }
        result[index] |= ((digit as u16) << shift) as u8;
        if shift != 0 {
            let high = ((digit as u16) >> (8 - shift)) as u8;
            if high != 0 {
                if result.len() <= index + 1 {
                    result.resize(index + 2, 0);
                }
                result[index + 1] |= high;
            }
        }
    }
    Some(result)
}

fn encode_nix32(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }
    let length = (bytes.len() * 8).div_ceil(5);
    let mut output = String::with_capacity(length);
    for n in (0..length).rev() {
        let bit = n * 5;
        let index = bit / 8;
        let shift = bit % 8;
        let low = (bytes[index] as u16) >> shift;
        let high = if index >= bytes.len() - 1 {
            0
        } else {
            (bytes[index + 1] as u16) << (8 - shift)
        };
        output.push(NIX_BASE32[((low | high) & 0x1f) as usize] as char);
    }
    output
}

fn first_unreachable(root: &str, records: &[VerifiedRecord]) -> Option<String> {
    let by_path = records
        .iter()
        .map(|record| (record.store_path.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    if !by_path.contains_key(root) {
        return Some(root.to_string());
    }
    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::from([root.to_string()]);
    while let Some(path) = queue.pop_front() {
        if !reachable.insert(path.clone()) {
            continue;
        }
        let record = by_path.get(path.as_str())?;
        for reference in &record.references {
            if reference != &path {
                queue.push_back(reference.clone());
            }
        }
    }
    records
        .iter()
        .find(|record| !reachable.contains(&record.store_path))
        .map(|record| record.store_path.clone())
}

fn signed_graph_digest(root: &str, records: &[VerifiedRecord]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(SIGNED_GRAPH_DOMAIN);
    field_le(&mut h, root);
    h.update(&(records.len() as u64).to_le_bytes());
    let mut records = records.iter().collect::<Vec<_>>();
    records.sort_by(|left, right| left.store_path.cmp(&right.store_path));
    for record in records {
        field_le(&mut h, &record.store_path);
        field_le(&mut h, &record.nar_hash);
        h.update(&record.nar_size.to_le_bytes());
        sorted_strings_le(&mut h, &record.references);
        sorted_strings_le(&mut h, &record.valid_signers);
    }
    blake3_text(h.finalize())
}

fn trusted_keys_digest(keys: &[String]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(TRUSTED_KEYS_DOMAIN);
    sorted_strings_le(&mut h, keys);
    blake3_text(h.finalize())
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    policy_digest: &str,
    local_qualification_digest: &str,
    local_closure_digest: &str,
    capsule_digest: &str,
    signed_graph_digest: &str,
    trusted_key_set_digest: &str,
    signatures_needed: u8,
    report_digest: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [
        policy_digest,
        local_qualification_digest,
        local_closure_digest,
        capsule_digest,
        signed_graph_digest,
        trusted_key_set_digest,
        report_digest,
    ] {
        field_le(&mut h, value);
    }
    h.update(&[signatures_needed]);
    blake3_text(h.finalize())
}

fn local_closure_digest(root: &str, entries: &[NixClosureEntry]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(LOCAL_CLOSURE_DOMAIN);
    field_le(&mut h, root);
    h.update(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        field_le(&mut h, &entry.store_path);
        field_le(&mut h, &entry.nar_hash);
        h.update(&entry.nar_size.to_le_bytes());
        match &entry.content_address {
            Some(value) => {
                h.update(&[1]);
                field_le(&mut h, value);
            }
            None => h.update(&[0]),
        }
        h.update(&(entry.references.len() as u64).to_le_bytes());
        for reference in &entry.references {
            field_le(&mut h, reference);
        }
    }
    blake3_text(h.finalize())
}

fn total_nar_bytes(entries: &[NixClosureEntry]) -> Option<u64> {
    entries
        .iter()
        .try_fold(0u64, |total, entry| total.checked_add(entry.nar_size))
}

fn valid_store_object_path(path: &str) -> bool {
    let Some(component) = path.strip_prefix("/nix/store/") else {
        return false;
    };
    if component.contains('/') || component.len() <= 33 || component.as_bytes()[32] != b'-' {
        return false;
    }
    component.as_bytes()[..32]
        .iter()
        .all(|byte| NIX_BASE32.contains(byte))
        && component[33..].bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.' | b'_' | b'?' | b'=')
        })
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_blake3(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && unique_by(values.iter().map(String::as_str))
}

fn unique_by<'a>(values: impl Iterator<Item = &'a str>) -> bool {
    let mut seen = BTreeSet::new();
    values.into_iter().all(|value| seen.insert(value))
}

fn sorted_copy(values: &[String]) -> Vec<String> {
    let mut output = values.to_vec();
    output.sort();
    output
}

fn field_le(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn sorted_strings_le(hasher: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        field_le(hasher, &value);
    }
}

fn blake3_text(hash: blake3::Hash) -> String {
    format!("blake3:{}", hash.to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    const ROOT: &str = "/nix/store/00000000000000000000000000000000-root";
    const DEP: &str = "/nix/store/11111111111111111111111111111111-dep";

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    #[test]
    fn nix32_round_trip_matches_raw_sha256_width() {
        let mut raw = [0u8; 32];
        for (index, byte) in raw.iter_mut().enumerate() {
            *byte = index as u8;
        }
        let encoded = encode_nix32(&raw);
        assert_eq!(encoded.len(), SHA256_NIX32_LEN);
        assert_eq!(decode_nix32(&encoded).unwrap(), raw);
        assert_eq!(parse_signed_nar_hash(&format!("sha256:{encoded}")), Some(raw));
    }

    #[test]
    fn sri_and_nix32_sha256_compare_on_raw_bytes() {
        let raw = [0x5au8; 32];
        let nix = format!("sha256:{}", encode_nix32(&raw));
        let sri = format!("sha256-{}", BASE64.encode(raw));
        assert_eq!(parse_sha256_hash(&nix), parse_sha256_hash(&sri));
    }

    #[test]
    fn fingerprint_sorts_store_path_references() {
        let record = SignedNarInfoRecord {
            store_path: ROOT.into(),
            nar_hash: format!("sha256:{}", encode_nix32(&[1u8; 32])),
            nar_size: 42,
            references: vec![DEP.into(), ROOT.into()],
            signatures: Vec::new(),
        };
        assert_eq!(
            nix_fingerprint(&record),
            format!("1;{ROOT};{};42;{ROOT},{DEP}", record.nar_hash)
        );
    }

    #[test]
    fn nix_named_ed25519_signature_verifies_exact_fingerprint() {
        let signing = SigningKey::from_bytes(&[7u8; 32]);
        let verifying = signing.verifying_key();
        let key_wire = format!("cache.example-1:{}", BASE64.encode(verifying.to_bytes()));
        let keys = parse_trusted_keys(&[key_wire]).unwrap();

        let record = SignedNarInfoRecord {
            store_path: ROOT.into(),
            nar_hash: format!("sha256:{}", encode_nix32(&[2u8; 32])),
            nar_size: 123,
            references: vec![DEP.into()],
            signatures: Vec::new(),
        };
        let fingerprint = nix_fingerprint(&record);
        let signature = signing.sign(fingerprint.as_bytes());
        let wire = format!("cache.example-1:{}", BASE64.encode(signature.to_bytes()));
        let (name, raw) = parse_named_base64(&wire, 64).unwrap();
        let raw: [u8; 64] = raw.as_slice().try_into().unwrap();
        let signature = Signature::from_bytes(&raw);
        assert!(keys[&name].verify(fingerprint.as_bytes(), &signature).is_ok());
        assert!(keys[&name].verify(b"different", &signature).is_err());
    }

    #[test]
    fn policy_identity_binds_threshold_and_trusted_keys() {
        let signing = SigningKey::from_bytes(&[9u8; 32]);
        let key = format!(
            "cache.example-1:{}",
            BASE64.encode(signing.verifying_key().to_bytes())
        );
        let mut policy = SignedNarInfoRuntimeClosurePolicy {
            schema_version: SIGNED_NARINFO_CLOSURE_POLICY_SCHEMA_V1.into(),
            policy_id: "signed-narinfo:v1".into(),
            root_store_path: ROOT.into(),
            expected_local_closure_policy_digest: d("local-policy"),
            trusted_public_keys: vec![key],
            signatures_needed: 1,
            max_paths: 32,
            max_total_nar_bytes: 1_000_000,
            evidence_refs: vec!["review:signed-narinfo".into()],
        };
        let first = policy.canonical_digest().unwrap();
        policy.max_paths = 31;
        assert_ne!(first, policy.canonical_digest().unwrap());
    }

    #[test]
    fn capsule_order_is_nonsemantic_but_signature_bytes_are_semantic() {
        let signing = SigningKey::from_bytes(&[11u8; 32]);
        let mut root = SignedNarInfoRecord {
            store_path: ROOT.into(),
            nar_hash: format!("sha256:{}", encode_nix32(&[3u8; 32])),
            nar_size: 10,
            references: vec![DEP.into()],
            signatures: vec![format!("cache.example-1:{}", BASE64.encode([0u8; 64]))],
        };
        let dep = SignedNarInfoRecord {
            store_path: DEP.into(),
            nar_hash: format!("sha256:{}", encode_nix32(&[4u8; 32])),
            nar_size: 5,
            references: Vec::new(),
            signatures: vec![format!("cache.example-1:{}", BASE64.encode([1u8; 64]))],
        };
        let mut left = SignedNarInfoClosureCapsule {
            schema_version: SIGNED_NARINFO_CLOSURE_CAPSULE_SCHEMA_V1.into(),
            capsule_id: "capsule:1".into(),
            records: vec![root.clone(), dep.clone()],
            evidence_refs: vec!["source:a".into(), "source:b".into()],
        };
        let first = left.canonical_digest(32).unwrap();
        left.records.reverse();
        left.evidence_refs.reverse();
        assert_eq!(first, left.canonical_digest(32).unwrap());
        root.signatures = vec![format!(
            "cache.example-1:{}",
            BASE64.encode(signing.sign(b"different").to_bytes())
        )];
        left.records = vec![root, dep];
        assert_ne!(first, left.canonical_digest(32).unwrap());
    }
}
