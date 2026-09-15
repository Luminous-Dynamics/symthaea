// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact binding between the current in-process TPM verifier host, a verified
//! Nix runtime closure, and a reviewed verifier-runtime continuity policy.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_current_in_process_linux_tpm_ima_anchor::
    CurrentInProcessLinuxTpmImaRuntimeAnchor;
use symthaea_assurance_nix_runtime_closure::VerifiedNixRuntimeClosure;
use symthaea_evidence_verifier_runtime_continuity::VerifierRuntimeContinuityPolicy;

pub const NIX_BOUND_IN_PROCESS_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.nix-bound-in-process-runtime-policy.v1";
pub const NIX_BOUND_IN_PROCESS_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.nix-bound-in-process-runtime-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-bound-in-process-runtime-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-bound-in-process-runtime-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-bound-in-process-runtime-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 1024;
const MAX_EVIDENCE_REFS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixBoundInProcessPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_provider_policy_digest: String,
    pub expected_closure_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub evidence_refs: Vec<String>,
}

impl NixBoundInProcessPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == NIX_BOUND_IN_PROCESS_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_provider_policy_digest)
            && valid_blake3_digest(&self.expected_closure_policy_digest)
            && valid_blake3_digest(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
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
            self.expected_provider_policy_digest.as_str(),
            self.expected_closure_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixBoundInProcessDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixBoundInProcessIssue {
    InvalidPolicy,
    InvalidRuntimePolicy,
    ProviderPolicyMismatch,
    ClosurePolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierRefMismatch,
    BackendMismatch,
    StoreRootMismatch,
    ExecutableDigestMismatch,
    DependencyClosureDigestMismatch,
}

impl NixBoundInProcessIssue {
    fn is_invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy | Self::InvalidRuntimePolicy)
    }

    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::InvalidRuntimePolicy => "invalid-runtime-policy",
            Self::ProviderPolicyMismatch => "provider-policy-mismatch",
            Self::ClosurePolicyMismatch => "closure-policy-mismatch",
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch",
            Self::RuntimeVerifierRefMismatch => "runtime-verifier-ref-mismatch",
            Self::BackendMismatch => "backend-mismatch",
            Self::StoreRootMismatch => "store-root-mismatch",
            Self::ExecutableDigestMismatch => "executable-digest-mismatch",
            Self::DependencyClosureDigestMismatch => "dependency-closure-digest-mismatch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixBoundInProcessReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub provider_policy_digest: String,
    pub provider_qualification_digest: String,
    pub closure_policy_digest: String,
    pub closure_qualification_digest: String,
    pub runtime_policy_digest: Option<String>,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub host_identity_digest: String,
    pub host_executable_digest: String,
    pub host_executable_path: String,
    pub provider_nix_store_root: String,
    pub closure_root_store_path: String,
    pub closure_digest: String,
    pub runtime_expected_executable_digest: String,
    pub runtime_expected_dependency_closure_digest: String,
    pub provider_use_at_ms: u64,
    pub disposition: NixBoundInProcessDisposition,
    pub issues: Vec<NixBoundInProcessIssue>,
}

impl NixBoundInProcessReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.provider_policy_digest.as_str(),
            self.provider_qualification_digest.as_str(),
            self.closure_policy_digest.as_str(),
            self.closure_qualification_digest.as_str(),
            self.runtime_policy_digest.as_deref().unwrap_or("-"),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.host_identity_digest.as_str(),
            self.host_executable_digest.as_str(),
            self.host_executable_path.as_str(),
            self.provider_nix_store_root.as_str(),
            self.closure_root_store_path.as_str(),
            self.closure_digest.as_str(),
            self.runtime_expected_executable_digest.as_str(),
            self.runtime_expected_dependency_closure_digest.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        hasher.update(&self.provider_use_at_ms.to_le_bytes());
        push_field(
            &mut hasher,
            match self.disposition {
                NixBoundInProcessDisposition::Invalid => "invalid",
                NixBoundInProcessDisposition::Blocked => "blocked",
                NixBoundInProcessDisposition::Qualified => "qualified",
            },
        );
        hasher.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            push_field(&mut hasher, issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixBoundInProcessRuntimePolicy {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    provider_policy_digest: String,
    provider_qualification_digest: String,
    closure_policy_digest: String,
    closure_qualification_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    host_identity_digest: String,
    executable_digest: String,
    executable_path: String,
    nix_store_root: String,
    dependency_closure_digest: String,
    provider_use_at_ms: u64,
}

impl NixBoundInProcessRuntimePolicy {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn provider_policy_digest(&self) -> &str { &self.provider_policy_digest }
    pub fn provider_qualification_digest(&self) -> &str { &self.provider_qualification_digest }
    pub fn closure_policy_digest(&self) -> &str { &self.closure_policy_digest }
    pub fn closure_qualification_digest(&self) -> &str { &self.closure_qualification_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn host_identity_digest(&self) -> &str { &self.host_identity_digest }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn executable_path(&self) -> &str { &self.executable_path }
    pub fn nix_store_root(&self) -> &str { &self.nix_store_root }
    pub fn dependency_closure_digest(&self) -> &str { &self.dependency_closure_digest }
    pub const fn provider_use_at_ms(&self) -> u64 { self.provider_use_at_ms }
    pub const fn signature_verification_in_process(&self) -> bool { true }
    pub const fn external_checkquote_process_required(&self) -> bool { false }
    pub const fn runtime_policy_bound_to_verified_closure(&self) -> bool { true }
    pub const fn provider_host_bound_to_runtime_policy(&self) -> bool { true }
    pub const fn closure_verified_at_assessment(&self) -> bool { true }
    pub const fn temporal_coobservation_established(&self) -> bool { false }
    pub const fn mapped_memory_identity_established(&self) -> bool { false }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool { false }
    pub const fn continuous_execution_established(&self) -> bool { false }
    pub const fn nix_database_currentness_established(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixBoundInProcessQualification {
    pub report: NixBoundInProcessReport,
    bound: NixBoundInProcessRuntimePolicy,
}

impl NixBoundInProcessQualification {
    pub fn bound(&self) -> &NixBoundInProcessRuntimePolicy { &self.bound }
    pub fn into_bound(self) -> NixBoundInProcessRuntimePolicy { self.bound }
}

pub fn bind_in_process_nix_runtime_policy(
    policy: &NixBoundInProcessPolicy,
    provider: &CurrentInProcessLinuxTpmImaRuntimeAnchor,
    closure: &VerifiedNixRuntimeClosure,
    runtime_policy: &VerifierRuntimeContinuityPolicy,
) -> Result<NixBoundInProcessQualification, NixBoundInProcessReport> {
    let policy_digest = policy.canonical_digest();
    let runtime_policy_digest = runtime_policy.canonical_digest();
    let mut report = NixBoundInProcessReport {
        schema_version: NIX_BOUND_IN_PROCESS_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        provider_policy_digest: provider.policy_digest().into(),
        provider_qualification_digest: provider.qualification_digest().into(),
        closure_policy_digest: closure.policy_digest().into(),
        closure_qualification_digest: closure.qualification_digest().into(),
        runtime_policy_digest: runtime_policy_digest.clone(),
        runtime_verifier_ref: runtime_policy.verifier_ref.clone(),
        backend_id: provider.backend_id().into(),
        host_identity_digest: provider.host_identity_digest().into(),
        host_executable_digest: provider.host_executable_blake3().into(),
        host_executable_path: provider.host_executable_path().into(),
        provider_nix_store_root: provider.host_nix_store_root().into(),
        closure_root_store_path: closure.root_store_path().into(),
        closure_digest: closure.closure_digest().into(),
        runtime_expected_executable_digest: runtime_policy.expected_executable_digest.clone(),
        runtime_expected_dependency_closure_digest: runtime_policy
            .expected_dependency_closure_digest
            .clone(),
        provider_use_at_ms: provider.use_at_ms(),
        disposition: NixBoundInProcessDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(NixBoundInProcessIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if !runtime_policy.validate() || runtime_policy_digest.is_none() {
        report.issues.push(NixBoundInProcessIssue::InvalidRuntimePolicy);
        return Err(finalize(report));
    }
    if provider.policy_digest() != policy.expected_provider_policy_digest {
        report.issues.push(NixBoundInProcessIssue::ProviderPolicyMismatch);
    }
    if closure.policy_digest() != policy.expected_closure_policy_digest {
        report.issues.push(NixBoundInProcessIssue::ClosurePolicyMismatch);
    }
    if runtime_policy_digest.as_deref() != Some(policy.expected_runtime_policy_digest.as_str()) {
        report.issues.push(NixBoundInProcessIssue::RuntimePolicyMismatch);
    }
    if provider.backend_id() != policy.expected_backend_id {
        report.issues.push(NixBoundInProcessIssue::BackendMismatch);
    }
    report.issues.extend(semantic_binding_issues(
        &policy.expected_runtime_verifier_ref,
        &runtime_policy.verifier_ref,
        provider.host_nix_store_root(),
        closure.root_store_path(),
        provider.host_executable_blake3(),
        &runtime_policy.expected_executable_digest,
        closure.closure_digest(),
        &runtime_policy.expected_dependency_closure_digest,
    ));

    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    report.disposition = NixBoundInProcessDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let runtime_policy_digest = runtime_policy_digest.expect("validated runtime policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        provider.qualification_digest(),
        closure.qualification_digest(),
        &runtime_policy_digest,
        provider.host_identity_digest(),
        provider.host_executable_blake3(),
        provider.host_nix_store_root(),
        closure.closure_digest(),
        provider.backend_id(),
        &report_digest,
    );
    let bound = NixBoundInProcessRuntimePolicy {
        qualification_digest,
        report_digest,
        policy_digest,
        provider_policy_digest: provider.policy_digest().into(),
        provider_qualification_digest: provider.qualification_digest().into(),
        closure_policy_digest: closure.policy_digest().into(),
        closure_qualification_digest: closure.qualification_digest().into(),
        runtime_policy_digest,
        runtime_verifier_ref: runtime_policy.verifier_ref.clone(),
        backend_id: provider.backend_id().into(),
        host_identity_digest: provider.host_identity_digest().into(),
        executable_digest: provider.host_executable_blake3().into(),
        executable_path: provider.host_executable_path().into(),
        nix_store_root: provider.host_nix_store_root().into(),
        dependency_closure_digest: closure.closure_digest().into(),
        provider_use_at_ms: provider.use_at_ms(),
    };

    Ok(NixBoundInProcessQualification { report, bound })
}

#[allow(clippy::too_many_arguments)]
fn semantic_binding_issues(
    expected_verifier_ref: &str,
    observed_verifier_ref: &str,
    provider_store_root: &str,
    closure_store_root: &str,
    provider_executable_digest: &str,
    runtime_expected_executable_digest: &str,
    closure_digest: &str,
    runtime_expected_closure_digest: &str,
) -> Vec<NixBoundInProcessIssue> {
    let mut issues = Vec::new();
    if observed_verifier_ref != expected_verifier_ref {
        issues.push(NixBoundInProcessIssue::RuntimeVerifierRefMismatch);
    }
    if provider_store_root != closure_store_root {
        issues.push(NixBoundInProcessIssue::StoreRootMismatch);
    }
    if provider_executable_digest != runtime_expected_executable_digest {
        issues.push(NixBoundInProcessIssue::ExecutableDigestMismatch);
    }
    if closure_digest != runtime_expected_closure_digest {
        issues.push(NixBoundInProcessIssue::DependencyClosureDigestMismatch);
    }
    issues
}

fn finalize(mut report: NixBoundInProcessReport) -> NixBoundInProcessReport {
    report.disposition = if report.issues.iter().any(NixBoundInProcessIssue::is_invalid) {
        NixBoundInProcessDisposition::Invalid
    } else {
        NixBoundInProcessDisposition::Blocked
    };
    report
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    policy_digest: &str,
    provider_qualification_digest: &str,
    closure_qualification_digest: &str,
    runtime_policy_digest: &str,
    host_identity_digest: &str,
    executable_digest: &str,
    nix_store_root: &str,
    closure_digest: &str,
    backend_id: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        provider_qualification_digest,
        closure_qualification_digest,
        runtime_policy_digest,
        host_identity_digest,
        executable_digest,
        nix_store_root,
        closure_digest,
        backend_id,
        report_digest,
    ] {
        push_field(&mut hasher, field);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn valid_blake3_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else { return false; };
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_refs(refs: &[String]) -> bool {
    if refs.len() > MAX_EVIDENCE_REFS || !refs.iter().all(|value| canonical_text(value)) {
        return false;
    }
    let mut seen = BTreeSet::new();
    refs.iter().all(|value| seen.insert(value.as_str()))
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for reference in refs { push_field(hasher, &reference); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: &str) -> String {
        format!("blake3:{}", blake3::hash(seed.as_bytes()).to_hex())
    }

    fn policy() -> NixBoundInProcessPolicy {
        NixBoundInProcessPolicy {
            schema_version: NIX_BOUND_IN_PROCESS_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:nix-bound-in-process:1".into(),
            expected_provider_policy_digest: digest("provider"),
            expected_closure_policy_digest: digest("closure"),
            expected_runtime_policy_digest: digest("runtime"),
            expected_runtime_verifier_ref: "verifier:in-process-host".into(),
            expected_backend_id: "aws-lc-rs-1.17.0-ecdsa-p256-sha256-fixed-v1".into(),
            evidence_refs: vec!["review:closure".into(), "review:runtime".into()],
        }
    }

    #[test]
    fn policy_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn verifier_role_and_backend_are_semantic() {
        let base = policy();
        let base_digest = base.canonical_digest().unwrap();
        let mut changed = base.clone();
        changed.expected_runtime_verifier_ref = "verifier:other".into();
        assert_ne!(base_digest, changed.canonical_digest().unwrap());
        changed = base.clone();
        changed.expected_backend_id = "backend:other".into();
        assert_ne!(base_digest, changed.canonical_digest().unwrap());
    }

    #[test]
    fn exact_semantic_binding_has_no_issues() {
        assert!(semantic_binding_issues(
            "verifier:host",
            "verifier:host",
            "/nix/store/root-host",
            "/nix/store/root-host",
            "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        .is_empty());
    }

    #[test]
    fn every_binding_axis_fails_independently() {
        let base = semantic_binding_issues(
            "verifier:a",
            "verifier:b",
            "root:a",
            "root:b",
            "exe:a",
            "exe:b",
            "closure:a",
            "closure:b",
        );
        assert_eq!(base.len(), 4);
        assert!(base.contains(&NixBoundInProcessIssue::RuntimeVerifierRefMismatch));
        assert!(base.contains(&NixBoundInProcessIssue::StoreRootMismatch));
        assert!(base.contains(&NixBoundInProcessIssue::ExecutableDigestMismatch));
        assert!(base.contains(&NixBoundInProcessIssue::DependencyClosureDigestMismatch));
    }

    #[test]
    fn qualification_identity_binds_provider_closure_runtime_and_host() {
        let base = qualification_digest(
            &digest("policy"),
            &digest("provider"),
            &digest("closure-qualification"),
            &digest("runtime"),
            &digest("host-identity"),
            &digest("executable"),
            "/nix/store/root-host",
            &digest("closure-content"),
            "backend:1",
            &digest("report"),
        );
        let changed = qualification_digest(
            &digest("policy"),
            &digest("provider:other"),
            &digest("closure-qualification"),
            &digest("runtime"),
            &digest("host-identity"),
            &digest("executable"),
            "/nix/store/root-host",
            &digest("closure-content"),
            "backend:1",
            &digest("report"),
        );
        assert_ne!(base, changed);
    }
}
