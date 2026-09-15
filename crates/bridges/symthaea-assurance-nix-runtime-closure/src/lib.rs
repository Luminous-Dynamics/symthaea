// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verified-at-assessment Nix runtime closure assurance.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use symthaea_assurance_fd_pinned_executable::FdPinnedExecutablePolicy;

#[cfg(target_os = "linux")]
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use symthaea_assurance_fd_pinned_executable::FdPinnedExecutable;

pub const NIX_RUNTIME_CLOSURE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.nix-runtime-closure-policy.v1";
pub const NIX_RUNTIME_CLOSURE_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.nix-runtime-closure-report.v1";
pub const NIX_PATH_INFO_JSON_FORMAT_V1: u64 = 1;
pub const NIX_STORE_DIR_V1: &str = "/nix/store";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-runtime-closure-policy.digest.v1\0";
const CLOSURE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-runtime-closure-content.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-runtime-closure-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-runtime-closure-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 4096;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_TRUSTED_KEYS: usize = 64;
const MAX_PATHS_HARD: u32 = 65_536;
const MAX_TOTAL_NAR_BYTES_HARD: u64 = 1u64 << 50;
const NIX_BASE32: &[u8] = b"0123456789abcdfghijklmnpqrsvwxyz";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixRuntimeClosurePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub root_store_path: String,
    pub expected_fd_pinned_nix_policy_digest: String,
    pub trusted_public_keys: Vec<String>,
    pub signatures_needed: u8,
    pub max_paths: u32,
    pub max_total_nar_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl NixRuntimeClosurePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == NIX_RUNTIME_CLOSURE_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_store_object_path(&self.root_store_path)
            && valid_blake3_digest(&self.expected_fd_pinned_nix_policy_digest)
            && !self.trusted_public_keys.is_empty()
            && self.trusted_public_keys.len() <= MAX_TRUSTED_KEYS
            && self.trusted_public_keys.iter().all(|key| valid_trusted_key(key))
            && unique(&self.trusted_public_keys)
            && (1..=8).contains(&self.signatures_needed)
            && (1..=MAX_PATHS_HARD).contains(&self.max_paths)
            && (1..=MAX_TOTAL_NAR_BYTES_HARD).contains(&self.max_total_nar_bytes)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.root_store_path.as_str(),
            self.expected_fd_pinned_nix_policy_digest.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        let mut keys = self.trusted_public_keys.clone();
        keys.sort();
        hasher.update(&(keys.len() as u64).to_le_bytes());
        for key in keys {
            push_field(&mut hasher, &key);
        }
        hasher.update(&[self.signatures_needed]);
        hasher.update(&self.max_paths.to_le_bytes());
        hasher.update(&self.max_total_nar_bytes.to_le_bytes());
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixClosureEntry {
    pub store_path: String,
    pub nar_hash: String,
    pub nar_size: u64,
    pub references: Vec<String>,
    pub content_address: Option<String>,
}

impl NixClosureEntry {
    fn validate(&self) -> bool {
        valid_store_object_path(&self.store_path)
            && valid_nar_hash(&self.nar_hash)
            && self.nar_size > 0
            && self.references.iter().all(|reference| valid_store_object_path(reference))
            && unique(&self.references)
            && self
                .content_address
                .as_ref()
                .map(|value| canonical_text(value))
                .unwrap_or(true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixRuntimeClosureDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixRuntimeClosureIssue {
    InvalidPolicy,
    NixExecutionPolicyMismatch,
    NixExecutionUnavailable(String),
    StoreVerifyRejected(Option<i32>),
    StoreVerifyEmittedOutput,
    PathInfoRejected(Option<i32>),
    PathInfoEmittedStderr,
    MalformedPathInfoJson(String),
    JsonVersionMismatch,
    StoreDirMismatch,
    TooManyPaths { observed: u64, maximum: u32 },
    InvalidClosureEntry(String),
    RootMissing,
    MissingReference { from: String, reference: String },
    UnreachableClosureEntry(String),
    TotalNarSizeOverflow,
    TotalNarSizeExceeded { observed: u64, maximum: u64 },
    InvocationReceiptInvalid,
}

impl NixRuntimeClosureIssue {
    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::NixExecutionPolicyMismatch
                | Self::MalformedPathInfoJson(_)
                | Self::JsonVersionMismatch
                | Self::StoreDirMismatch
                | Self::TooManyPaths { .. }
                | Self::InvalidClosureEntry(_)
                | Self::RootMissing
                | Self::MissingReference { .. }
                | Self::UnreachableClosureEntry(_)
                | Self::TotalNarSizeOverflow
                | Self::InvocationReceiptInvalid
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::NixExecutionPolicyMismatch => "nix-execution-policy-mismatch".into(),
            Self::NixExecutionUnavailable(reason) => format!("nix-execution-unavailable:{reason}"),
            Self::StoreVerifyRejected(code) => format!("store-verify-rejected:{code:?}"),
            Self::StoreVerifyEmittedOutput => "store-verify-emitted-output".into(),
            Self::PathInfoRejected(code) => format!("path-info-rejected:{code:?}"),
            Self::PathInfoEmittedStderr => "path-info-emitted-stderr".into(),
            Self::MalformedPathInfoJson(reason) => format!("malformed-json:{reason}"),
            Self::JsonVersionMismatch => "json-version-mismatch".into(),
            Self::StoreDirMismatch => "store-dir-mismatch".into(),
            Self::TooManyPaths { observed, maximum } => {
                format!("too-many-paths:{observed}:{maximum}")
            }
            Self::InvalidClosureEntry(path) => format!("invalid-closure-entry:{path}"),
            Self::RootMissing => "root-missing".into(),
            Self::MissingReference { from, reference } => {
                format!("missing-reference:{from}:{reference}")
            }
            Self::UnreachableClosureEntry(path) => format!("unreachable-entry:{path}"),
            Self::TotalNarSizeOverflow => "total-nar-size-overflow".into(),
            Self::TotalNarSizeExceeded { observed, maximum } => {
                format!("total-nar-size-exceeded:{observed}:{maximum}")
            }
            Self::InvocationReceiptInvalid => "invocation-receipt-invalid".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixRuntimeClosureReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub root_store_path: String,
    pub nix_execution_policy_digest: String,
    pub nix_execution_identity_digest: String,
    pub store_verify_invocation_digest: Option<String>,
    pub path_info_invocation_digest: Option<String>,
    pub closure_digest: Option<String>,
    pub path_count: u64,
    pub total_nar_bytes: u64,
    pub disposition: NixRuntimeClosureDisposition,
    pub issues: Vec<NixRuntimeClosureIssue>,
}

impl NixRuntimeClosureReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.root_store_path.as_str(),
            self.nix_execution_policy_digest.as_str(),
            self.nix_execution_identity_digest.as_str(),
            self.store_verify_invocation_digest.as_deref().unwrap_or("-"),
            self.path_info_invocation_digest.as_deref().unwrap_or("-"),
            self.closure_digest.as_deref().unwrap_or("-"),
        ] {
            push_field(&mut hasher, field);
        }
        hasher.update(&self.path_count.to_le_bytes());
        hasher.update(&self.total_nar_bytes.to_le_bytes());
        push_field(
            &mut hasher,
            match self.disposition {
                NixRuntimeClosureDisposition::Invalid => "invalid",
                NixRuntimeClosureDisposition::Blocked => "blocked",
                NixRuntimeClosureDisposition::Qualified => "qualified",
            },
        );
        for issue in &self.issues {
            push_field(&mut hasher, &issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedNixRuntimeClosure {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    root_store_path: String,
    closure_digest: String,
    path_count: u64,
    total_nar_bytes: u64,
    nix_execution_policy_digest: String,
    nix_execution_identity_digest: String,
    store_verify_invocation_digest: String,
    path_info_invocation_digest: String,
}

impl VerifiedNixRuntimeClosure {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }
    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }
    pub fn root_store_path(&self) -> &str {
        &self.root_store_path
    }
    pub fn closure_digest(&self) -> &str {
        &self.closure_digest
    }
    pub const fn path_count(&self) -> u64 {
        self.path_count
    }
    pub const fn total_nar_bytes(&self) -> u64 {
        self.total_nar_bytes
    }
    pub fn nix_execution_policy_digest(&self) -> &str {
        &self.nix_execution_policy_digest
    }
    pub fn nix_execution_identity_digest(&self) -> &str {
        &self.nix_execution_identity_digest
    }
    pub fn store_verify_invocation_digest(&self) -> &str {
        &self.store_verify_invocation_digest
    }
    pub fn path_info_invocation_digest(&self) -> &str {
        &self.path_info_invocation_digest
    }
    pub const fn recursive_contents_verified_at_assessment(&self) -> bool {
        true
    }
    pub const fn reviewed_nix_trust_policy_satisfied_at_assessment(&self) -> bool {
        true
    }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool {
        false
    }
    pub const fn nix_database_currentness_established(&self) -> bool {
        false
    }
    pub const fn root_resistant_immutability_established(&self) -> bool {
        false
    }
    pub const fn trusted_time_established(&self) -> bool {
        false
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixRuntimeClosureQualification {
    pub report: NixRuntimeClosureReport,
    pub entries: Vec<NixClosureEntry>,
    verified: VerifiedNixRuntimeClosure,
}

impl NixRuntimeClosureQualification {
    pub fn verified(&self) -> &VerifiedNixRuntimeClosure {
        &self.verified
    }
    pub fn into_verified(self) -> VerifiedNixRuntimeClosure {
        self.verified
    }
}

#[cfg(target_os = "linux")]
pub fn qualify_nix_runtime_closure(
    policy: &NixRuntimeClosurePolicy,
    nix: &mut FdPinnedExecutable,
) -> Result<NixRuntimeClosureQualification, NixRuntimeClosureReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = base_report(policy, policy_digest.clone(), nix);
    if policy_digest.is_none() {
        report.issues.push(NixRuntimeClosureIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if nix.policy_digest() != policy.expected_fd_pinned_nix_policy_digest {
        report
            .issues
            .push(NixRuntimeClosureIssue::NixExecutionPolicyMismatch);
        return Err(finalize(report));
    }

    let trusted_keys = {
        let mut keys = policy.trusted_public_keys.clone();
        keys.sort();
        keys.join(" ")
    };
    let env = vec![
        (OsString::from("LANG"), OsString::from("C")),
        (OsString::from("LC_ALL"), OsString::from("C")),
        (OsString::from("TZ"), OsString::from("UTC0")),
    ];
    let prefix = vec![
        OsString::from("--extra-experimental-features"),
        OsString::from("nix-command"),
        OsString::from("--offline"),
        OsString::from("--option"),
        OsString::from("trusted-public-keys"),
        OsString::from(trusted_keys),
    ];

    let mut verify_args = prefix.clone();
    verify_args.extend([
        OsString::from("store"),
        OsString::from("verify"),
        OsString::from("--recursive"),
        OsString::from("--sigs-needed"),
        OsString::from(policy.signatures_needed.to_string()),
        OsString::from(&policy.root_store_path),
    ]);
    let verify = match nix.execute(&verify_args, &env) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(NixRuntimeClosureIssue::NixExecutionUnavailable(format!(
                    "store-verify:{error:?}"
                )));
            return Err(finalize(report));
        }
    };
    report.store_verify_invocation_digest = verify.receipt.canonical_digest();
    if report.store_verify_invocation_digest.is_none() {
        report
            .issues
            .push(NixRuntimeClosureIssue::InvocationReceiptInvalid);
        return Err(finalize(report));
    }
    if verify.exit_code != Some(0) {
        report
            .issues
            .push(NixRuntimeClosureIssue::StoreVerifyRejected(verify.exit_code));
        return Err(finalize(report));
    }
    if !verify.stdout.is_empty() || !verify.stderr.is_empty() {
        report
            .issues
            .push(NixRuntimeClosureIssue::StoreVerifyEmittedOutput);
        return Err(finalize(report));
    }

    let mut path_info_args = prefix;
    path_info_args.extend([
        OsString::from("path-info"),
        OsString::from("--recursive"),
        OsString::from("--json"),
        OsString::from("--json-format"),
        OsString::from(NIX_PATH_INFO_JSON_FORMAT_V1.to_string()),
        OsString::from("--no-pretty"),
        OsString::from(&policy.root_store_path),
    ]);
    let path_info = match nix.execute(&path_info_args, &env) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(NixRuntimeClosureIssue::NixExecutionUnavailable(format!(
                    "path-info:{error:?}"
                )));
            return Err(finalize(report));
        }
    };
    report.path_info_invocation_digest = path_info.receipt.canonical_digest();
    if report.path_info_invocation_digest.is_none() {
        report
            .issues
            .push(NixRuntimeClosureIssue::InvocationReceiptInvalid);
        return Err(finalize(report));
    }
    if path_info.exit_code != Some(0) {
        report
            .issues
            .push(NixRuntimeClosureIssue::PathInfoRejected(path_info.exit_code));
        return Err(finalize(report));
    }
    if !path_info.stderr.is_empty() {
        report
            .issues
            .push(NixRuntimeClosureIssue::PathInfoEmittedStderr);
        return Err(finalize(report));
    }

    let entries = match parse_path_info_v1(&path_info.stdout, policy) {
        Ok(value) => value,
        Err(issue) => {
            report.issues.push(issue);
            return Err(finalize(report));
        }
    };
    let path_count = entries.len() as u64;
    let total_nar_bytes = match entries.iter().try_fold(0u64, |total, entry| {
        total.checked_add(entry.nar_size)
    }) {
        Some(value) => value,
        None => {
            report.issues.push(NixRuntimeClosureIssue::TotalNarSizeOverflow);
            return Err(finalize(report));
        }
    };
    if total_nar_bytes > policy.max_total_nar_bytes {
        report
            .issues
            .push(NixRuntimeClosureIssue::TotalNarSizeExceeded {
                observed: total_nar_bytes,
                maximum: policy.max_total_nar_bytes,
            });
        return Err(finalize(report));
    }
    let closure_digest = closure_digest(&policy.root_store_path, &entries);
    report.closure_digest = Some(closure_digest.clone());
    report.path_count = path_count;
    report.total_nar_bytes = total_nar_bytes;
    report.disposition = NixRuntimeClosureDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let verify_digest = report
        .store_verify_invocation_digest
        .clone()
        .expect("qualified report has verify receipt");
    let path_info_digest = report
        .path_info_invocation_digest
        .clone()
        .expect("qualified report has path-info receipt");
    let qualification_digest = qualification_digest(
        &policy_digest,
        nix.identity_digest(),
        &verify_digest,
        &path_info_digest,
        &closure_digest,
        &report_digest,
    );
    let verified = VerifiedNixRuntimeClosure {
        qualification_digest,
        report_digest,
        policy_digest,
        root_store_path: policy.root_store_path.clone(),
        closure_digest,
        path_count,
        total_nar_bytes,
        nix_execution_policy_digest: nix.policy_digest().to_string(),
        nix_execution_identity_digest: nix.identity_digest().to_string(),
        store_verify_invocation_digest: verify_digest,
        path_info_invocation_digest: path_info_digest,
    };
    Ok(NixRuntimeClosureQualification {
        report,
        entries,
        verified,
    })
}

#[cfg(not(target_os = "linux"))]
pub fn qualify_nix_runtime_closure(
    policy: &NixRuntimeClosurePolicy,
    _nix: &mut symthaea_assurance_fd_pinned_executable::FdPinnedExecutable,
) -> Result<NixRuntimeClosureQualification, NixRuntimeClosureReport> {
    let mut report = NixRuntimeClosureReport {
        schema_version: NIX_RUNTIME_CLOSURE_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy.canonical_digest(),
        root_store_path: policy.root_store_path.clone(),
        nix_execution_policy_digest: String::new(),
        nix_execution_identity_digest: String::new(),
        store_verify_invocation_digest: None,
        path_info_invocation_digest: None,
        closure_digest: None,
        path_count: 0,
        total_nar_bytes: 0,
        disposition: NixRuntimeClosureDisposition::Blocked,
        issues: vec![NixRuntimeClosureIssue::NixExecutionUnavailable(
            "Linux fd-pinned execution is required".into(),
        )],
    };
    report = finalize(report);
    Err(report)
}

#[cfg(target_os = "linux")]
fn base_report(
    policy: &NixRuntimeClosurePolicy,
    policy_digest: Option<String>,
    nix: &FdPinnedExecutable,
) -> NixRuntimeClosureReport {
    NixRuntimeClosureReport {
        schema_version: NIX_RUNTIME_CLOSURE_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        root_store_path: policy.root_store_path.clone(),
        nix_execution_policy_digest: nix.policy_digest().to_string(),
        nix_execution_identity_digest: nix.identity_digest().to_string(),
        store_verify_invocation_digest: None,
        path_info_invocation_digest: None,
        closure_digest: None,
        path_count: 0,
        total_nar_bytes: 0,
        disposition: NixRuntimeClosureDisposition::Invalid,
        issues: Vec::new(),
    }
}

fn finalize(mut report: NixRuntimeClosureReport) -> NixRuntimeClosureReport {
    report.disposition = if report.issues.iter().any(NixRuntimeClosureIssue::is_invalid) {
        NixRuntimeClosureDisposition::Invalid
    } else {
        NixRuntimeClosureDisposition::Blocked
    };
    report
}

fn parse_path_info_v1(
    bytes: &[u8],
    policy: &NixRuntimeClosurePolicy,
) -> Result<Vec<NixClosureEntry>, NixRuntimeClosureIssue> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| NixRuntimeClosureIssue::MalformedPathInfoJson(error.to_string()))?;
    let object = value.as_object().ok_or_else(|| {
        NixRuntimeClosureIssue::MalformedPathInfoJson("top level is not an object".into())
    })?;
    if object.get("version").and_then(Value::as_u64) != Some(NIX_PATH_INFO_JSON_FORMAT_V1) {
        return Err(NixRuntimeClosureIssue::JsonVersionMismatch);
    }
    if object.get("storeDir").and_then(Value::as_str) != Some(NIX_STORE_DIR_V1) {
        return Err(NixRuntimeClosureIssue::StoreDirMismatch);
    }

    let path_keys = object
        .keys()
        .filter(|key| key.as_str() != "version" && key.as_str() != "storeDir")
        .cloned()
        .collect::<Vec<_>>();
    if path_keys.len() as u64 > policy.max_paths as u64 {
        return Err(NixRuntimeClosureIssue::TooManyPaths {
            observed: path_keys.len() as u64,
            maximum: policy.max_paths,
        });
    }

    let mut entries = Vec::with_capacity(path_keys.len());
    for path in path_keys {
        let raw = object
            .get(&path)
            .and_then(Value::as_object)
            .ok_or_else(|| NixRuntimeClosureIssue::InvalidClosureEntry(path.clone()))?;
        let nar_hash = raw
            .get("narHash")
            .and_then(Value::as_str)
            .ok_or_else(|| NixRuntimeClosureIssue::InvalidClosureEntry(path.clone()))?
            .to_string();
        let nar_size = raw
            .get("narSize")
            .and_then(Value::as_u64)
            .ok_or_else(|| NixRuntimeClosureIssue::InvalidClosureEntry(path.clone()))?;
        let references = raw
            .get("references")
            .and_then(Value::as_array)
            .ok_or_else(|| NixRuntimeClosureIssue::InvalidClosureEntry(path.clone()))?
            .iter()
            .map(|value| value.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| NixRuntimeClosureIssue::InvalidClosureEntry(path.clone()))?;
        let content_address = match raw.get("ca") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(value.clone()),
            Some(_) => return Err(NixRuntimeClosureIssue::InvalidClosureEntry(path.clone())),
        };
        let mut entry = NixClosureEntry {
            store_path: path.clone(),
            nar_hash,
            nar_size,
            references,
            content_address,
        };
        entry.references.sort();
        if !entry.validate() {
            return Err(NixRuntimeClosureIssue::InvalidClosureEntry(path));
        }
        entries.push(entry);
    }
    entries.sort_by(|left, right| left.store_path.cmp(&right.store_path));
    validate_complete_closure(entries, policy)
}

fn validate_complete_closure(
    entries: Vec<NixClosureEntry>,
    policy: &NixRuntimeClosurePolicy,
) -> Result<Vec<NixClosureEntry>, NixRuntimeClosureIssue> {
    let by_path = entries
        .iter()
        .map(|entry| (entry.store_path.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    if !by_path.contains_key(policy.root_store_path.as_str()) {
        return Err(NixRuntimeClosureIssue::RootMissing);
    }
    for entry in &entries {
        for reference in &entry.references {
            if !by_path.contains_key(reference.as_str()) {
                return Err(NixRuntimeClosureIssue::MissingReference {
                    from: entry.store_path.clone(),
                    reference: reference.clone(),
                });
            }
        }
    }

    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::from([policy.root_store_path.clone()]);
    while let Some(path) = queue.pop_front() {
        if !reachable.insert(path.clone()) {
            continue;
        }
        let entry = by_path
            .get(path.as_str())
            .expect("all reachable references were validated");
        for reference in &entry.references {
            if reference != &path {
                queue.push_back(reference.clone());
            }
        }
    }
    if let Some(extra) = entries
        .iter()
        .find(|entry| !reachable.contains(&entry.store_path))
    {
        return Err(NixRuntimeClosureIssue::UnreachableClosureEntry(
            extra.store_path.clone(),
        ));
    }
    Ok(entries)
}

fn closure_digest(root: &str, entries: &[NixClosureEntry]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CLOSURE_DIGEST_DOMAIN);
    push_field(&mut hasher, root);
    hasher.update(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        push_field(&mut hasher, &entry.store_path);
        push_field(&mut hasher, &entry.nar_hash);
        hasher.update(&entry.nar_size.to_le_bytes());
        match &entry.content_address {
            Some(value) => {
                hasher.update(&[1]);
                push_field(&mut hasher, value);
            }
            None => hasher.update(&[0]),
        }
        hasher.update(&(entry.references.len() as u64).to_le_bytes());
        for reference in &entry.references {
            push_field(&mut hasher, reference);
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn qualification_digest(
    policy_digest: &str,
    nix_identity_digest: &str,
    verify_invocation_digest: &str,
    path_info_invocation_digest: &str,
    closure_digest: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        nix_identity_digest,
        verify_invocation_digest,
        path_info_invocation_digest,
        closure_digest,
        report_digest,
    ] {
        push_field(&mut hasher, field);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
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

fn valid_nar_hash(value: &str) -> bool {
    (value.starts_with("sha256-") || value.starts_with("sha256:"))
        && value.len() <= 256
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

fn valid_trusted_key(value: &str) -> bool {
    value.len() <= MAX_TEXT_BYTES
        && value == value.trim()
        && value.contains(':')
        && !value.chars().any(char::is_whitespace)
        && !value.chars().any(char::is_control)
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_blake3_digest(value: &str) -> bool {
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
        && unique(values)
}

fn unique(values: &[String]) -> bool {
    values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for reference in refs {
        push_field(hasher, &reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOT: &str = "/nix/store/00000000000000000000000000000000-root";
    const DEP: &str = "/nix/store/11111111111111111111111111111111-dep";
    const EXTRA: &str = "/nix/store/22222222222222222222222222222222-extra";

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> NixRuntimeClosurePolicy {
        NixRuntimeClosurePolicy {
            schema_version: NIX_RUNTIME_CLOSURE_POLICY_SCHEMA_V1.into(),
            policy_id: "nix-closure:checkquote:1".into(),
            root_store_path: ROOT.into(),
            expected_fd_pinned_nix_policy_digest: d("nix-exec-policy"),
            trusted_public_keys: vec!["cache.example-1:AAAA".into()],
            signatures_needed: 1,
            max_paths: 32,
            max_total_nar_bytes: 1_000_000,
            evidence_refs: vec!["review:nix-closure".into()],
        }
    }

    fn fixture(root_refs: &[&str], include_extra: bool) -> Vec<u8> {
        let mut map = serde_json::Map::new();
        map.insert("version".into(), Value::from(1));
        map.insert("storeDir".into(), Value::from("/nix/store"));
        map.insert(
            ROOT.into(),
            serde_json::json!({
                "narHash": "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                "narSize": 100,
                "references": root_refs,
                "ca": null
            }),
        );
        map.insert(
            DEP.into(),
            serde_json::json!({
                "narHash": "sha256-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=",
                "narSize": 50,
                "references": [DEP],
                "ca": "fixed:r:sha256:example"
            }),
        );
        if include_extra {
            map.insert(
                EXTRA.into(),
                serde_json::json!({
                    "narHash": "sha256-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC=",
                    "narSize": 25,
                    "references": [],
                    "ca": null
                }),
            );
        }
        serde_json::to_vec(&Value::Object(map)).unwrap()
    }

    #[test]
    fn complete_reachable_closure_parses_and_has_stable_digest() {
        let entries = parse_path_info_v1(&fixture(&[DEP], false), &policy()).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].store_path, ROOT);
        assert_eq!(entries[1].store_path, DEP);
        let first = closure_digest(ROOT, &entries);

        let reordered = parse_path_info_v1(&fixture(&[DEP], false), &policy()).unwrap();
        assert_eq!(first, closure_digest(ROOT, &reordered));
    }

    #[test]
    fn missing_referenced_path_is_invalid() {
        let json = serde_json::json!({
            "version": 1,
            "storeDir": "/nix/store",
            ROOT: {
                "narHash": "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                "narSize": 100,
                "references": [DEP],
                "ca": null
            }
        });
        let error = parse_path_info_v1(&serde_json::to_vec(&json).unwrap(), &policy()).unwrap_err();
        assert!(matches!(error, NixRuntimeClosureIssue::MissingReference { .. }));
    }

    #[test]
    fn unrelated_extra_entry_is_invalid() {
        let error = parse_path_info_v1(&fixture(&[DEP], true), &policy()).unwrap_err();
        assert!(matches!(
            error,
            NixRuntimeClosureIssue::UnreachableClosureEntry(path) if path == EXTRA
        ));
    }

    #[test]
    fn json_version_and_store_dir_are_authoritative() {
        let mut value: Value = serde_json::from_slice(&fixture(&[DEP], false)).unwrap();
        value["version"] = Value::from(2);
        assert!(matches!(
            parse_path_info_v1(&serde_json::to_vec(&value).unwrap(), &policy()),
            Err(NixRuntimeClosureIssue::JsonVersionMismatch)
        ));
        let mut value: Value = serde_json::from_slice(&fixture(&[DEP], false)).unwrap();
        value["storeDir"] = Value::from("/other/store");
        assert!(matches!(
            parse_path_info_v1(&serde_json::to_vec(&value).unwrap(), &policy()),
            Err(NixRuntimeClosureIssue::StoreDirMismatch)
        ));
    }

    #[test]
    fn policy_digest_commits_trust_and_resource_limits() {
        let first = policy().canonical_digest().unwrap();
        let mut changed = policy();
        changed.signatures_needed = 2;
        assert_ne!(first, changed.canonical_digest().unwrap());
        let mut changed = policy();
        changed.max_paths += 1;
        assert_ne!(first, changed.canonical_digest().unwrap());
    }

    #[test]
    fn trusted_key_order_is_nonsemantic() {
        let mut left = policy();
        left.trusted_public_keys = vec!["a:AAAA".into(), "b:BBBB".into()];
        let mut right = left.clone();
        right.trusted_public_keys.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn expected_nix_execution_policy_is_explicitly_bound() {
        let fd_policy = FdPinnedExecutablePolicy {
            schema_version:
                symthaea_assurance_fd_pinned_executable::FD_PINNED_EXECUTABLE_POLICY_SCHEMA_V1
                    .into(),
            policy_id: "nix-cli:reviewed".into(),
            executable_path:
                "/nix/store/33333333333333333333333333333333-nix/bin/nix".into(),
            expected_executable_blake3: d("nix-bin"),
            max_executable_bytes: 128 * 1024 * 1024,
            max_output_bytes: 16 * 1024 * 1024,
            max_invocations: 2,
            evidence_refs: vec!["review:nix-cli".into()],
        };
        let mut closure_policy = policy();
        closure_policy.expected_fd_pinned_nix_policy_digest = fd_policy.canonical_digest().unwrap();
        assert!(closure_policy.validate());
    }
}
