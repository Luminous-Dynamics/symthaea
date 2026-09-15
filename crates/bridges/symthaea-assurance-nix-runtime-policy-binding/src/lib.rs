// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact binding between a qualified fd-pinned provider executable, a verified
//! Nix runtime closure, and a reviewed verifier-runtime continuity policy.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_current_fd_pinned_linux_tpm_ima_anchor::
    CurrentFdPinnedLinuxTpmImaRuntimeAnchor;
use symthaea_assurance_nix_runtime_closure::VerifiedNixRuntimeClosure;
use symthaea_evidence_verifier_runtime_continuity::VerifierRuntimeContinuityPolicy;

pub const NIX_RUNTIME_POLICY_BINDING_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.nix-runtime-policy-binding-policy.v1";
pub const NIX_RUNTIME_POLICY_BINDING_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.nix-runtime-policy-binding-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-runtime-policy-binding-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-runtime-policy-binding-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-runtime-policy-binding-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 1024;
const MAX_EVIDENCE_REFS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixRuntimePolicyBindingPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_provider_policy_digest: String,
    pub expected_closure_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub evidence_refs: Vec<String>,
}

impl NixRuntimePolicyBindingPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == NIX_RUNTIME_POLICY_BINDING_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_provider_policy_digest)
            && valid_blake3_digest(&self.expected_closure_policy_digest)
            && valid_blake3_digest(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
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
        ] {
            push_field(&mut hasher, field);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixRuntimePolicyBindingDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixRuntimePolicyBindingIssue {
    InvalidPolicy,
    InvalidRuntimePolicy,
    ProviderPolicyMismatch,
    ClosurePolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierRefMismatch,
    StoreRootMismatch,
    ExecutableDigestMismatch,
    DependencyClosureDigestMismatch,
}

impl NixRuntimePolicyBindingIssue {
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
            Self::StoreRootMismatch => "store-root-mismatch",
            Self::ExecutableDigestMismatch => "executable-digest-mismatch",
            Self::DependencyClosureDigestMismatch => "dependency-closure-digest-mismatch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixRuntimePolicyBindingReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub provider_policy_digest: String,
    pub provider_qualification_digest: String,
    pub closure_policy_digest: String,
    pub closure_qualification_digest: String,
    pub runtime_policy_digest: Option<String>,
    pub runtime_verifier_ref: String,
    pub provider_executable_digest: String,
    pub canonical_executable_path: String,
    pub provider_nix_store_root: String,
    pub closure_root_store_path: String,
    pub closure_digest: String,
    pub runtime_expected_executable_digest: String,
    pub runtime_expected_dependency_closure_digest: String,
    pub provider_use_at_ms: u64,
    pub disposition: NixRuntimePolicyBindingDisposition,
    pub issues: Vec<NixRuntimePolicyBindingIssue>,
}

impl NixRuntimePolicyBindingReport {
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
            self.provider_executable_digest.as_str(),
            self.canonical_executable_path.as_str(),
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
                NixRuntimePolicyBindingDisposition::Invalid => "invalid",
                NixRuntimePolicyBindingDisposition::Blocked => "blocked",
                NixRuntimePolicyBindingDisposition::Qualified => "qualified",
            },
        );
        hasher.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            push_field(&mut hasher, issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixBoundRuntimePolicy {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    provider_policy_digest: String,
    provider_qualification_digest: String,
    closure_policy_digest: String,
    closure_qualification_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    executable_digest: String,
    canonical_executable_path: String,
    nix_store_root: String,
    dependency_closure_digest: String,
    provider_use_at_ms: u64,
}

impl NixBoundRuntimePolicy {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }
    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }
    pub fn provider_policy_digest(&self) -> &str {
        &self.provider_policy_digest
    }
    pub fn provider_qualification_digest(&self) -> &str {
        &self.provider_qualification_digest
    }
    pub fn closure_policy_digest(&self) -> &str {
        &self.closure_policy_digest
    }
    pub fn closure_qualification_digest(&self) -> &str {
        &self.closure_qualification_digest
    }
    pub fn runtime_policy_digest(&self) -> &str {
        &self.runtime_policy_digest
    }
    pub fn runtime_verifier_ref(&self) -> &str {
        &self.runtime_verifier_ref
    }
    pub fn executable_digest(&self) -> &str {
        &self.executable_digest
    }
    pub fn canonical_executable_path(&self) -> &str {
        &self.canonical_executable_path
    }
    pub fn nix_store_root(&self) -> &str {
        &self.nix_store_root
    }
    pub fn dependency_closure_digest(&self) -> &str {
        &self.dependency_closure_digest
    }
    pub const fn provider_use_at_ms(&self) -> u64 {
        self.provider_use_at_ms
    }
    pub const fn runtime_policy_bound_to_verified_closure(&self) -> bool {
        true
    }
    pub const fn provider_executable_bound_to_runtime_policy(&self) -> bool {
        true
    }
    pub const fn closure_verified_at_assessment(&self) -> bool {
        true
    }
    pub const fn temporal_coobservation_established(&self) -> bool {
        false
    }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool {
        false
    }
    pub const fn continuous_execution_established(&self) -> bool {
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
pub struct NixRuntimePolicyBindingQualification {
    pub report: NixRuntimePolicyBindingReport,
    bound: NixBoundRuntimePolicy,
}

impl NixRuntimePolicyBindingQualification {
    pub fn bound(&self) -> &NixBoundRuntimePolicy {
        &self.bound
    }

    pub fn into_bound(self) -> NixBoundRuntimePolicy {
        self.bound
    }
}

pub fn bind_nix_runtime_policy(
    policy: &NixRuntimePolicyBindingPolicy,
    provider: &CurrentFdPinnedLinuxTpmImaRuntimeAnchor,
    closure: &VerifiedNixRuntimeClosure,
    runtime_policy: &VerifierRuntimeContinuityPolicy,
) -> Result<NixRuntimePolicyBindingQualification, NixRuntimePolicyBindingReport> {
    let policy_digest = policy.canonical_digest();
    let runtime_policy_digest = runtime_policy.canonical_digest();
    let mut report = NixRuntimePolicyBindingReport {
        schema_version: NIX_RUNTIME_POLICY_BINDING_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        provider_policy_digest: provider.policy_digest().into(),
        provider_qualification_digest: provider.qualification_digest().into(),
        closure_policy_digest: closure.policy_digest().into(),
        closure_qualification_digest: closure.qualification_digest().into(),
        runtime_policy_digest: runtime_policy_digest.clone(),
        runtime_verifier_ref: runtime_policy.verifier_ref.clone(),
        provider_executable_digest: provider.executable_digest().into(),
        canonical_executable_path: provider.canonical_executable_path().into(),
        provider_nix_store_root: provider.nix_store_root().into(),
        closure_root_store_path: closure.root_store_path().into(),
        closure_digest: closure.closure_digest().into(),
        runtime_expected_executable_digest: runtime_policy.expected_executable_digest.clone(),
        runtime_expected_dependency_closure_digest: runtime_policy
            .expected_dependency_closure_digest
            .clone(),
        provider_use_at_ms: provider.use_at_ms(),
        disposition: NixRuntimePolicyBindingDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report
            .issues
            .push(NixRuntimePolicyBindingIssue::InvalidPolicy);
        return Err(finalize_report(report));
    }
    if !runtime_policy.validate() || runtime_policy_digest.is_none() {
        report
            .issues
            .push(NixRuntimePolicyBindingIssue::InvalidRuntimePolicy);
        return Err(finalize_report(report));
    }
    if provider.policy_digest() != policy.expected_provider_policy_digest {
        report
            .issues
            .push(NixRuntimePolicyBindingIssue::ProviderPolicyMismatch);
    }
    if closure.policy_digest() != policy.expected_closure_policy_digest {
        report
            .issues
            .push(NixRuntimePolicyBindingIssue::ClosurePolicyMismatch);
    }
    if runtime_policy_digest.as_deref() != Some(policy.expected_runtime_policy_digest.as_str()) {
        report
            .issues
            .push(NixRuntimePolicyBindingIssue::RuntimePolicyMismatch);
    }
    report.issues.extend(semantic_binding_issues(
        &policy.expected_runtime_verifier_ref,
        &runtime_policy.verifier_ref,
        provider.nix_store_root(),
        closure.root_store_path(),
        provider.executable_digest(),
        &runtime_policy.expected_executable_digest,
        closure.closure_digest(),
        &runtime_policy.expected_dependency_closure_digest,
    ));

    if !report.issues.is_empty() {
        return Err(finalize_report(report));
    }

    report.disposition = NixRuntimePolicyBindingDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let runtime_policy_digest = runtime_policy_digest.expect("validated runtime policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        provider.qualification_digest(),
        closure.qualification_digest(),
        &runtime_policy_digest,
        provider.executable_digest(),
        provider.nix_store_root(),
        closure.closure_digest(),
        &report_digest,
    );
    let bound = NixBoundRuntimePolicy {
        qualification_digest,
        report_digest,
        policy_digest,
        provider_policy_digest: provider.policy_digest().into(),
        provider_qualification_digest: provider.qualification_digest().into(),
        closure_policy_digest: closure.policy_digest().into(),
        closure_qualification_digest: closure.qualification_digest().into(),
        runtime_policy_digest,
        runtime_verifier_ref: runtime_policy.verifier_ref.clone(),
        executable_digest: provider.executable_digest().into(),
        canonical_executable_path: provider.canonical_executable_path().into(),
        nix_store_root: provider.nix_store_root().into(),
        dependency_closure_digest: closure.closure_digest().into(),
        provider_use_at_ms: provider.use_at_ms(),
    };

    Ok(NixRuntimePolicyBindingQualification { report, bound })
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
) -> Vec<NixRuntimePolicyBindingIssue> {
    let mut issues = Vec::new();
    if observed_verifier_ref != expected_verifier_ref {
        issues.push(NixRuntimePolicyBindingIssue::RuntimeVerifierRefMismatch);
    }
    if provider_store_root != closure_store_root {
        issues.push(NixRuntimePolicyBindingIssue::StoreRootMismatch);
    }
    if provider_executable_digest != runtime_expected_executable_digest {
        issues.push(NixRuntimePolicyBindingIssue::ExecutableDigestMismatch);
    }
    if closure_digest != runtime_expected_closure_digest {
        issues.push(NixRuntimePolicyBindingIssue::DependencyClosureDigestMismatch);
    }
    issues
}

fn finalize_report(mut report: NixRuntimePolicyBindingReport) -> NixRuntimePolicyBindingReport {
    report.disposition = if report
        .issues
        .iter()
        .any(NixRuntimePolicyBindingIssue::is_invalid)
    {
        NixRuntimePolicyBindingDisposition::Invalid
    } else {
        NixRuntimePolicyBindingDisposition::Blocked
    };
    report
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    policy_digest: &str,
    provider_qualification_digest: &str,
    closure_qualification_digest: &str,
    runtime_policy_digest: &str,
    executable_digest: &str,
    nix_store_root: &str,
    dependency_closure_digest: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        provider_qualification_digest,
        closure_qualification_digest,
        runtime_policy_digest,
        executable_digest,
        nix_store_root,
        dependency_closure_digest,
        report_digest,
    ] {
        push_field(&mut hasher, field);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn valid_blake3_digest(value: &str) -> bool {
    value
        .strip_prefix("blake3:")
        .map(|hex| {
            hex.len() == 64
                && hex
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        })
        .unwrap_or(false)
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TEXT_BYTES
        && value == value.trim()
        && !value.contains('\0')
}

fn valid_refs(values: &[String]) -> bool {
    if values.len() > MAX_EVIDENCE_REFS || values.iter().any(|value| !canonical_text(value)) {
        return false;
    }
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.as_str()))
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

    fn digest(byte: char) -> String {
        format!("blake3:{}", byte.to_string().repeat(64))
    }

    fn policy() -> NixRuntimePolicyBindingPolicy {
        NixRuntimePolicyBindingPolicy {
            schema_version: NIX_RUNTIME_POLICY_BINDING_POLICY_SCHEMA_V1.into(),
            policy_id: "nix-runtime-binding-v1".into(),
            expected_provider_policy_digest: digest('a'),
            expected_closure_policy_digest: digest('b'),
            expected_runtime_policy_digest: digest('c'),
            expected_runtime_verifier_ref: "tpm2-checkquote-verifier".into(),
            evidence_refs: vec!["review:A".into(), "review:B".into()],
        }
    }

    #[test]
    fn evidence_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn runtime_policy_and_role_are_semantic_commitments() {
        let base = policy();
        let base_digest = base.canonical_digest();

        let mut changed_policy = base.clone();
        changed_policy.expected_runtime_policy_digest = digest('d');
        assert_ne!(base_digest, changed_policy.canonical_digest());

        let mut changed_role = base.clone();
        changed_role.expected_runtime_verifier_ref = "some-other-verifier".into();
        assert_ne!(base_digest, changed_role.canonical_digest());
    }

    #[test]
    fn exact_semantic_binding_has_no_issues() {
        assert!(semantic_binding_issues(
            "role",
            "role",
            "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-tool",
            "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-tool",
            "blake3:exe",
            "blake3:exe",
            "blake3:closure",
            "blake3:closure",
        )
        .is_empty());
    }

    #[test]
    fn every_semantic_substitution_is_exposed() {
        let issues = semantic_binding_issues(
            "expected-role",
            "other-role",
            "provider-root",
            "closure-root",
            "provider-exe",
            "runtime-exe",
            "closure-a",
            "closure-b",
        );
        assert_eq!(
            issues,
            vec![
                NixRuntimePolicyBindingIssue::RuntimeVerifierRefMismatch,
                NixRuntimePolicyBindingIssue::StoreRootMismatch,
                NixRuntimePolicyBindingIssue::ExecutableDigestMismatch,
                NixRuntimePolicyBindingIssue::DependencyClosureDigestMismatch,
            ]
        );
    }

    #[test]
    fn qualification_digest_binds_both_parent_qualifications() {
        let base = qualification_digest(
            &digest('a'),
            &digest('b'),
            &digest('c'),
            &digest('d'),
            &digest('e'),
            "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-tool",
            &digest('f'),
            &digest('0'),
        );
        let changed = qualification_digest(
            &digest('a'),
            &digest('9'),
            &digest('c'),
            &digest('d'),
            &digest('e'),
            "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-tool",
            &digest('f'),
            &digest('0'),
        );
        assert_ne!(base, changed);
    }
}
