// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Continuous verifier execution bound to an exact Nix-closure-backed runtime policy.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_nix_runtime_policy_binding::NixBoundRuntimePolicy;
use symthaea_evidence_verifier_runtime_continuity::ContinuousVerifierExecution;

pub const CLOSURE_BACKED_CONTINUOUS_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.closure-backed-continuous-verifier-policy.v1";
pub const CLOSURE_BACKED_CONTINUOUS_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.closure-backed-continuous-verifier-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.closure-backed-continuous-verifier-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.closure-backed-continuous-verifier-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.closure-backed-continuous-verifier-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 1024;
const MAX_EVIDENCE_REFS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureBackedContinuousPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_nix_binding_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub evidence_refs: Vec<String>,
}

impl ClosureBackedContinuousPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == CLOSURE_BACKED_CONTINUOUS_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_nix_binding_policy_digest)
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
            self.expected_nix_binding_policy_digest.as_str(),
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
pub enum ClosureBackedContinuousDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClosureBackedContinuousIssue {
    InvalidPolicy,
    NixBindingPolicyMismatch,
    BoundRuntimePolicyMismatch,
    BoundVerifierRefMismatch,
    ContinuousRuntimePolicyMismatch,
    ContinuousVerifierRefMismatch,
}

impl ClosureBackedContinuousIssue {
    fn is_invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy)
    }

    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::NixBindingPolicyMismatch => "nix-binding-policy-mismatch",
            Self::BoundRuntimePolicyMismatch => "bound-runtime-policy-mismatch",
            Self::BoundVerifierRefMismatch => "bound-verifier-ref-mismatch",
            Self::ContinuousRuntimePolicyMismatch => "continuous-runtime-policy-mismatch",
            Self::ContinuousVerifierRefMismatch => "continuous-verifier-ref-mismatch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureBackedContinuousReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub nix_binding_policy_digest: String,
    pub nix_binding_qualification_digest: String,
    pub bound_runtime_policy_digest: String,
    pub bound_runtime_verifier_ref: String,
    pub continuous_runtime_policy_digest: String,
    pub continuous_verifier_ref: String,
    pub continuous_execution_digest: String,
    pub process_instance_id: String,
    pub runtime_trace_digest: String,
    pub executable_digest: String,
    pub nix_store_root: String,
    pub dependency_closure_digest: String,
    pub provider_use_at_ms: u64,
    pub computation_started_at_ms: u64,
    pub computation_completed_at_ms: u64,
    pub continuous_assessed_at_ms: u64,
    pub disposition: ClosureBackedContinuousDisposition,
    pub issues: Vec<ClosureBackedContinuousIssue>,
}

impl ClosureBackedContinuousReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.nix_binding_policy_digest.as_str(),
            self.nix_binding_qualification_digest.as_str(),
            self.bound_runtime_policy_digest.as_str(),
            self.bound_runtime_verifier_ref.as_str(),
            self.continuous_runtime_policy_digest.as_str(),
            self.continuous_verifier_ref.as_str(),
            self.continuous_execution_digest.as_str(),
            self.process_instance_id.as_str(),
            self.runtime_trace_digest.as_str(),
            self.executable_digest.as_str(),
            self.nix_store_root.as_str(),
            self.dependency_closure_digest.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        hasher.update(&self.provider_use_at_ms.to_le_bytes());
        hasher.update(&self.computation_started_at_ms.to_le_bytes());
        hasher.update(&self.computation_completed_at_ms.to_le_bytes());
        hasher.update(&self.continuous_assessed_at_ms.to_le_bytes());
        push_field(
            &mut hasher,
            match self.disposition {
                ClosureBackedContinuousDisposition::Invalid => "invalid",
                ClosureBackedContinuousDisposition::Blocked => "blocked",
                ClosureBackedContinuousDisposition::Qualified => "qualified",
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
pub struct ClosureBackedContinuousVerifierExecution {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    nix_binding_policy_digest: String,
    nix_binding_qualification_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    continuous_execution_digest: String,
    process_instance_id: String,
    runtime_trace_digest: String,
    input_digest: String,
    output_digest: String,
    executable_digest: String,
    nix_store_root: String,
    dependency_closure_digest: String,
    provider_use_at_ms: u64,
    computation_started_at_ms: u64,
    computation_completed_at_ms: u64,
    continuous_assessed_at_ms: u64,
}

impl ClosureBackedContinuousVerifierExecution {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }
    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }
    pub fn nix_binding_policy_digest(&self) -> &str {
        &self.nix_binding_policy_digest
    }
    pub fn nix_binding_qualification_digest(&self) -> &str {
        &self.nix_binding_qualification_digest
    }
    pub fn runtime_policy_digest(&self) -> &str {
        &self.runtime_policy_digest
    }
    pub fn runtime_verifier_ref(&self) -> &str {
        &self.runtime_verifier_ref
    }
    pub fn continuous_execution_digest(&self) -> &str {
        &self.continuous_execution_digest
    }
    pub fn process_instance_id(&self) -> &str {
        &self.process_instance_id
    }
    pub fn runtime_trace_digest(&self) -> &str {
        &self.runtime_trace_digest
    }
    pub fn input_digest(&self) -> &str {
        &self.input_digest
    }
    pub fn output_digest(&self) -> &str {
        &self.output_digest
    }
    pub fn executable_digest(&self) -> &str {
        &self.executable_digest
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
    pub const fn computation_started_at_ms(&self) -> u64 {
        self.computation_started_at_ms
    }
    pub const fn computation_completed_at_ms(&self) -> u64 {
        self.computation_completed_at_ms
    }
    pub const fn continuous_assessed_at_ms(&self) -> u64 {
        self.continuous_assessed_at_ms
    }
    pub const fn runtime_policy_bound_to_verified_closure(&self) -> bool {
        true
    }
    pub const fn continuous_execution_under_bound_runtime_policy(&self) -> bool {
        true
    }
    pub const fn closure_verified_at_assessment(&self) -> bool {
        true
    }
    pub const fn closure_temporally_coobserved_at_computation(&self) -> bool {
        false
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
pub struct ClosureBackedContinuousQualification {
    pub report: ClosureBackedContinuousReport,
    verified: ClosureBackedContinuousVerifierExecution,
}

impl ClosureBackedContinuousQualification {
    pub fn verified(&self) -> &ClosureBackedContinuousVerifierExecution {
        &self.verified
    }

    pub fn into_verified(self) -> ClosureBackedContinuousVerifierExecution {
        self.verified
    }
}

pub fn bind_continuous_execution_to_nix_runtime_policy(
    policy: &ClosureBackedContinuousPolicy,
    bound: &NixBoundRuntimePolicy,
    continuous: &ContinuousVerifierExecution,
) -> Result<ClosureBackedContinuousQualification, ClosureBackedContinuousReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = ClosureBackedContinuousReport {
        schema_version: CLOSURE_BACKED_CONTINUOUS_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        nix_binding_policy_digest: bound.policy_digest().into(),
        nix_binding_qualification_digest: bound.qualification_digest().into(),
        bound_runtime_policy_digest: bound.runtime_policy_digest().into(),
        bound_runtime_verifier_ref: bound.runtime_verifier_ref().into(),
        continuous_runtime_policy_digest: continuous.policy_digest().into(),
        continuous_verifier_ref: continuous.verifier_ref().into(),
        continuous_execution_digest: continuous.continuity_digest().into(),
        process_instance_id: continuous.process_instance_id().into(),
        runtime_trace_digest: continuous.runtime_trace_digest().into(),
        executable_digest: bound.executable_digest().into(),
        nix_store_root: bound.nix_store_root().into(),
        dependency_closure_digest: bound.dependency_closure_digest().into(),
        provider_use_at_ms: bound.provider_use_at_ms(),
        computation_started_at_ms: continuous.started_at_ms(),
        computation_completed_at_ms: continuous.completed_at_ms(),
        continuous_assessed_at_ms: continuous.assessed_at_ms(),
        disposition: ClosureBackedContinuousDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report
            .issues
            .push(ClosureBackedContinuousIssue::InvalidPolicy);
        return Err(finalize_report(report));
    }

    report.issues.extend(semantic_binding_issues(
        &policy.expected_nix_binding_policy_digest,
        bound.policy_digest(),
        &policy.expected_runtime_policy_digest,
        bound.runtime_policy_digest(),
        &policy.expected_runtime_verifier_ref,
        bound.runtime_verifier_ref(),
        continuous.policy_digest(),
        continuous.verifier_ref(),
    ));

    if !report.issues.is_empty() {
        return Err(finalize_report(report));
    }

    report.disposition = ClosureBackedContinuousDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        bound.qualification_digest(),
        continuous.continuity_digest(),
        bound.runtime_policy_digest(),
        bound.runtime_verifier_ref(),
        bound.executable_digest(),
        bound.nix_store_root(),
        bound.dependency_closure_digest(),
        &report_digest,
    );
    let verified = ClosureBackedContinuousVerifierExecution {
        qualification_digest,
        report_digest,
        policy_digest,
        nix_binding_policy_digest: bound.policy_digest().into(),
        nix_binding_qualification_digest: bound.qualification_digest().into(),
        runtime_policy_digest: bound.runtime_policy_digest().into(),
        runtime_verifier_ref: bound.runtime_verifier_ref().into(),
        continuous_execution_digest: continuous.continuity_digest().into(),
        process_instance_id: continuous.process_instance_id().into(),
        runtime_trace_digest: continuous.runtime_trace_digest().into(),
        input_digest: continuous.input_digest().into(),
        output_digest: continuous.output_digest().into(),
        executable_digest: bound.executable_digest().into(),
        nix_store_root: bound.nix_store_root().into(),
        dependency_closure_digest: bound.dependency_closure_digest().into(),
        provider_use_at_ms: bound.provider_use_at_ms(),
        computation_started_at_ms: continuous.started_at_ms(),
        computation_completed_at_ms: continuous.completed_at_ms(),
        continuous_assessed_at_ms: continuous.assessed_at_ms(),
    };

    Ok(ClosureBackedContinuousQualification { report, verified })
}

#[allow(clippy::too_many_arguments)]
fn semantic_binding_issues(
    expected_binding_policy_digest: &str,
    observed_binding_policy_digest: &str,
    expected_runtime_policy_digest: &str,
    bound_runtime_policy_digest: &str,
    expected_verifier_ref: &str,
    bound_verifier_ref: &str,
    continuous_runtime_policy_digest: &str,
    continuous_verifier_ref: &str,
) -> Vec<ClosureBackedContinuousIssue> {
    let mut issues = Vec::new();
    if observed_binding_policy_digest != expected_binding_policy_digest {
        issues.push(ClosureBackedContinuousIssue::NixBindingPolicyMismatch);
    }
    if bound_runtime_policy_digest != expected_runtime_policy_digest {
        issues.push(ClosureBackedContinuousIssue::BoundRuntimePolicyMismatch);
    }
    if bound_verifier_ref != expected_verifier_ref {
        issues.push(ClosureBackedContinuousIssue::BoundVerifierRefMismatch);
    }
    if continuous_runtime_policy_digest != bound_runtime_policy_digest {
        issues.push(ClosureBackedContinuousIssue::ContinuousRuntimePolicyMismatch);
    }
    if continuous_verifier_ref != bound_verifier_ref {
        issues.push(ClosureBackedContinuousIssue::ContinuousVerifierRefMismatch);
    }
    issues
}

fn finalize_report(mut report: ClosureBackedContinuousReport) -> ClosureBackedContinuousReport {
    report.disposition = if report
        .issues
        .iter()
        .any(ClosureBackedContinuousIssue::is_invalid)
    {
        ClosureBackedContinuousDisposition::Invalid
    } else {
        ClosureBackedContinuousDisposition::Blocked
    };
    report
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    policy_digest: &str,
    nix_binding_qualification_digest: &str,
    continuous_execution_digest: &str,
    runtime_policy_digest: &str,
    runtime_verifier_ref: &str,
    executable_digest: &str,
    nix_store_root: &str,
    dependency_closure_digest: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        nix_binding_qualification_digest,
        continuous_execution_digest,
        runtime_policy_digest,
        runtime_verifier_ref,
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

    fn policy() -> ClosureBackedContinuousPolicy {
        ClosureBackedContinuousPolicy {
            schema_version: CLOSURE_BACKED_CONTINUOUS_POLICY_SCHEMA_V1.into(),
            policy_id: "closure-backed-continuous-v1".into(),
            expected_nix_binding_policy_digest: digest('a'),
            expected_runtime_policy_digest: digest('b'),
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
        changed_policy.expected_runtime_policy_digest = digest('c');
        assert_ne!(base_digest, changed_policy.canonical_digest());

        let mut changed_role = base.clone();
        changed_role.expected_runtime_verifier_ref = "other-verifier".into();
        assert_ne!(base_digest, changed_role.canonical_digest());
    }

    #[test]
    fn exact_semantic_binding_has_no_issues() {
        assert!(semantic_binding_issues(
            "binding-policy",
            "binding-policy",
            "runtime-policy",
            "runtime-policy",
            "role",
            "role",
            "runtime-policy",
            "role",
        )
        .is_empty());
    }

    #[test]
    fn every_parent_or_runtime_substitution_is_exposed() {
        let issues = semantic_binding_issues(
            "expected-binding",
            "other-binding",
            "expected-runtime",
            "other-runtime",
            "expected-role",
            "other-bound-role",
            "continuous-runtime",
            "continuous-role",
        );
        assert_eq!(
            issues,
            vec![
                ClosureBackedContinuousIssue::NixBindingPolicyMismatch,
                ClosureBackedContinuousIssue::BoundRuntimePolicyMismatch,
                ClosureBackedContinuousIssue::BoundVerifierRefMismatch,
                ClosureBackedContinuousIssue::ContinuousRuntimePolicyMismatch,
                ClosureBackedContinuousIssue::ContinuousVerifierRefMismatch,
            ]
        );
    }

    #[test]
    fn qualification_digest_binds_continuous_execution() {
        let base = qualification_digest(
            &digest('a'),
            &digest('b'),
            &digest('c'),
            &digest('d'),
            "role",
            &digest('e'),
            "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-tool",
            &digest('f'),
            &digest('0'),
        );
        let changed = qualification_digest(
            &digest('a'),
            &digest('b'),
            &digest('9'),
            &digest('d'),
            "role",
            &digest('e'),
            "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-tool",
            &digest('f'),
            &digest('0'),
        );
        assert_ne!(base, changed);
    }
}
