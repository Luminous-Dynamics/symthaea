// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Continuity-qualified release gate for the confined verifier bootstrap.
//!
//! This crate consumes both the exact syscall-confinement qualification and the
//! matching file-backed executable-mapping continuity qualification. Immediately
//! before release it proves that the tracee is still in a ptrace tracing stop
//! owned by the current supervisor, then performs one safe `ptrace::detach`.
//!
//! In this stacked lineage the lower confinement crate exposes no public detach
//! helper. Therefore this is the only reviewed public safe-Rust release path for
//! the confined bootstrap capability. That API-surface statement does not claim
//! exclusion of privileged external mutation, arbitrary kernel/debugger action,
//! post-release runtime continuity, trusted time, or physical authority.

#![cfg(all(
    target_os = "linux",
    target_arch = "x86_64",
    any(target_env = "gnu", target_env = "musl")
))]
#![deny(unsafe_code)]

use nix::{
    sys::{
        ptrace,
        signal::Signal,
    },
    unistd::{getpid, Pid},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
};
use symthaea_assurance_bootstrap_executable_mapping_continuity::
    BootstrapExecutableMappingContinuityQualification;
use symthaea_assurance_bootstrap_syscall_confinement::
    BootstrapSyscallConfinementQualification;

pub const BOOTSTRAP_RELEASE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.continuity-qualified-bootstrap-release-policy.v1";
pub const BOOTSTRAP_RELEASE_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.continuity-qualified-bootstrap-release-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.continuity-qualified-bootstrap-release-policy.digest.v1\0";
const STOP_DOMAIN: &[u8] =
    b"symthaea.assurance.continuity-qualified-bootstrap-release-stop.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.continuity-qualified-bootstrap-release-report.digest.v1\0";
const RELEASE_DOMAIN: &[u8] =
    b"symthaea.assurance.continuity-qualified-bootstrap-release.digest.v1\0";
const MAX_TEXT_BYTES: usize = 4096;
const MAX_EVIDENCE_REFS: usize = 128;
const HARD_MAX_STATUS_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapReleasePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_confinement_policy_digest: String,
    pub expected_mapping_continuity_policy_digest: String,
    pub max_proc_status_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl BootstrapReleasePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == BOOTSTRAP_RELEASE_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_confinement_policy_digest)
            && valid_blake3(&self.expected_mapping_continuity_policy_digest)
            && (1..=HARD_MAX_STATUS_BYTES).contains(&self.max_proc_status_bytes)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut h = blake3::Hasher::new();
        h.update(POLICY_DOMAIN);
        field(&mut h, &self.schema_version);
        field(&mut h, &self.policy_id);
        field(&mut h, &self.expected_confinement_policy_digest);
        field(&mut h, &self.expected_mapping_continuity_policy_digest);
        h.update(&self.max_proc_status_bytes.to_be_bytes());
        sorted_strings(&mut h, &self.evidence_refs);
        Some(blake3_text(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapReleaseIssue {
    InvalidPolicy,
    ConfinementPolicyMismatch,
    MappingContinuityPolicyMismatch,
    ContinuityConfinementMismatch,
    ExecConfirmationMismatch,
    SyscallSequenceMismatch,
    TraceePidInvalid,
    ProcStatusUnavailable(String),
    ProcStatusTooLarge { observed: u64, maximum: u64 },
    ProcStatusMalformed,
    TraceeNotInPtraceStop(String),
    TracerPidMismatch { expected: i32, observed: i32 },
    StopOwnershipChanged,
    DetachFailed(String),
}

impl BootstrapReleaseIssue {
    fn invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy | Self::TraceePidInvalid)
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::ConfinementPolicyMismatch => "confinement-policy-mismatch".into(),
            Self::MappingContinuityPolicyMismatch => "mapping-continuity-policy-mismatch".into(),
            Self::ContinuityConfinementMismatch => "continuity-confinement-mismatch".into(),
            Self::ExecConfirmationMismatch => "exec-confirmation-mismatch".into(),
            Self::SyscallSequenceMismatch => "syscall-sequence-mismatch".into(),
            Self::TraceePidInvalid => "tracee-pid-invalid".into(),
            Self::ProcStatusUnavailable(value) => format!("proc-status-unavailable:{value}"),
            Self::ProcStatusTooLarge { observed, maximum } => {
                format!("proc-status-too-large:{observed}:{maximum}")
            }
            Self::ProcStatusMalformed => "proc-status-malformed".into(),
            Self::TraceeNotInPtraceStop(value) => format!("tracee-not-ptrace-stop:{value}"),
            Self::TracerPidMismatch { expected, observed } => {
                format!("tracer-pid-mismatch:{expected}:{observed}")
            }
            Self::StopOwnershipChanged => "stop-ownership-changed".into(),
            Self::DetachFailed(value) => format!("detach-failed:{value}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapReleaseDisposition {
    Invalid,
    Blocked,
    Released,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapReleaseReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub tracee_pid: i32,
    pub supervisor_pid: i32,
    pub confinement_qualification_digest: String,
    pub mapping_continuity_qualification_digest: String,
    pub exec_confirmation_digest: String,
    pub syscall_sequence_digest: String,
    pub launch_mapping_set_digest: String,
    pub ready_mapping_set_digest: String,
    pub first_stop_observation_digest: Option<String>,
    pub second_stop_observation_digest: Option<String>,
    pub ptrace_detach_returned_success: bool,
    pub disposition: BootstrapReleaseDisposition,
    pub issues: Vec<BootstrapReleaseIssue>,
}

impl BootstrapReleaseReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.confinement_qualification_digest.as_str(),
            self.mapping_continuity_qualification_digest.as_str(),
            self.exec_confirmation_digest.as_str(),
            self.syscall_sequence_digest.as_str(),
            self.launch_mapping_set_digest.as_str(),
            self.ready_mapping_set_digest.as_str(),
            self.first_stop_observation_digest.as_deref().unwrap_or("-"),
            self.second_stop_observation_digest.as_deref().unwrap_or("-"),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_pid.to_be_bytes());
        h.update(&self.supervisor_pid.to_be_bytes());
        h.update(&[u8::from(self.ptrace_detach_returned_success)]);
        field(
            &mut h,
            match self.disposition {
                BootstrapReleaseDisposition::Invalid => "invalid",
                BootstrapReleaseDisposition::Blocked => "blocked",
                BootstrapReleaseDisposition::Released => "released",
            },
        );
        h.update(&(self.issues.len() as u64).to_be_bytes());
        for issue in &self.issues {
            field(&mut h, &issue.code());
        }
        blake3_text(h.finalize())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleasedBootstrapVerifier {
    release_digest: String,
    report_digest: String,
    policy_digest: String,
    tracee_pid: i32,
    confinement_qualification_digest: String,
    mapping_continuity_qualification_digest: String,
    exec_confirmation_digest: String,
    syscall_sequence_digest: String,
    launch_mapping_set_digest: String,
    ready_mapping_set_digest: String,
    pre_detach_stop_observation_digest: String,
}

impl ReleasedBootstrapVerifier {
    pub fn release_digest(&self) -> &str { &self.release_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub const fn tracee_pid(&self) -> i32 { self.tracee_pid }
    pub fn confinement_qualification_digest(&self) -> &str {
        &self.confinement_qualification_digest
    }
    pub fn mapping_continuity_qualification_digest(&self) -> &str {
        &self.mapping_continuity_qualification_digest
    }
    pub fn exec_confirmation_digest(&self) -> &str { &self.exec_confirmation_digest }
    pub fn syscall_sequence_digest(&self) -> &str { &self.syscall_sequence_digest }
    pub fn launch_mapping_set_digest(&self) -> &str { &self.launch_mapping_set_digest }
    pub fn ready_mapping_set_digest(&self) -> &str { &self.ready_mapping_set_digest }
    pub fn pre_detach_stop_observation_digest(&self) -> &str {
        &self.pre_detach_stop_observation_digest
    }

    pub const fn matching_confinement_and_mapping_continuity_consumed(&self) -> bool { true }
    pub const fn tracee_still_ptrace_stopped_by_current_supervisor_before_detach(&self) -> bool {
        true
    }
    pub const fn ptrace_detach_returned_success(&self) -> bool { true }
    pub const fn input_capabilities_consumed_by_release_constructor(&self) -> bool { true }
    /// Within the reviewed stacked crates, no public safe-Rust release function
    /// accepts only the weaker confinement-ready capability.
    pub const fn all_weaker_release_paths_removed(&self) -> bool { true }
    pub const fn external_privileged_mutation_excluded(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct BootstrapReleaseSuccess {
    pub report: BootstrapReleaseReport,
    released: ReleasedBootstrapVerifier,
}

impl BootstrapReleaseSuccess {
    pub fn released(&self) -> &ReleasedBootstrapVerifier { &self.released }
    pub fn into_released(self) -> ReleasedBootstrapVerifier { self.released }
}

pub struct BootstrapReleaseFailure {
    pub report: BootstrapReleaseReport,
    confinement: BootstrapSyscallConfinementQualification,
    continuity: BootstrapExecutableMappingContinuityQualification,
}

impl BootstrapReleaseFailure {
    pub fn confinement(&self) -> &BootstrapSyscallConfinementQualification { &self.confinement }
    pub fn continuity(&self) -> &BootstrapExecutableMappingContinuityQualification { &self.continuity }
    pub fn into_parts(
        self,
    ) -> (
        BootstrapReleaseReport,
        BootstrapSyscallConfinementQualification,
        BootstrapExecutableMappingContinuityQualification,
    ) {
        (self.report, self.confinement, self.continuity)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StopObservation {
    state_code: String,
    tracer_pid: i32,
    digest: String,
}

pub fn release_continuity_qualified_bootstrap(
    policy: &BootstrapReleasePolicy,
    confinement: BootstrapSyscallConfinementQualification,
    continuity: BootstrapExecutableMappingContinuityQualification,
) -> Result<BootstrapReleaseSuccess, BootstrapReleaseFailure> {
    let policy_digest = policy.canonical_digest();
    let ready = confinement.ready();
    let mapped = continuity.verified();
    let supervisor_pid = getpid().as_raw();
    let tracee_pid = ready.tracee_pid();

    let mut report = BootstrapReleaseReport {
        schema_version: BOOTSTRAP_RELEASE_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        tracee_pid,
        supervisor_pid,
        confinement_qualification_digest: ready.qualification_digest().into(),
        mapping_continuity_qualification_digest: mapped.qualification_digest().into(),
        exec_confirmation_digest: ready.exec_confirmation_digest().into(),
        syscall_sequence_digest: ready.syscall_sequence_digest().into(),
        launch_mapping_set_digest: mapped.launch_mapping_set_digest().into(),
        ready_mapping_set_digest: mapped.ready_mapping_set_digest().into(),
        first_stop_observation_digest: None,
        second_stop_observation_digest: None,
        ptrace_detach_returned_success: false,
        disposition: BootstrapReleaseDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(BootstrapReleaseIssue::InvalidPolicy);
    }
    if ready.policy_digest() != policy.expected_confinement_policy_digest {
        report.issues.push(BootstrapReleaseIssue::ConfinementPolicyMismatch);
    }
    if mapped.policy_digest() != policy.expected_mapping_continuity_policy_digest {
        report
            .issues
            .push(BootstrapReleaseIssue::MappingContinuityPolicyMismatch);
    }
    if mapped.confinement_qualification_digest() != ready.qualification_digest() {
        report.issues.push(BootstrapReleaseIssue::ContinuityConfinementMismatch);
    }
    if mapped.exec_confirmation_digest() != ready.exec_confirmation_digest() {
        report.issues.push(BootstrapReleaseIssue::ExecConfirmationMismatch);
    }
    if mapped.syscall_sequence_digest() != ready.syscall_sequence_digest() {
        report.issues.push(BootstrapReleaseIssue::SyscallSequenceMismatch);
    }
    if tracee_pid <= 1 {
        report.issues.push(BootstrapReleaseIssue::TraceePidInvalid);
    }
    if !ready.tracee_stopped_before_release() {
        report
            .issues
            .push(BootstrapReleaseIssue::TraceeNotInPtraceStop("parent-capability".into()));
    }

    if !report.issues.is_empty() {
        return Err(failure(finalize_blocked(report), confinement, continuity));
    }

    let first = match observe_stop(tracee_pid, policy.max_proc_status_bytes) {
        Ok(value) => value,
        Err(issue) => {
            report.issues.push(issue);
            return Err(failure(finalize_blocked(report), confinement, continuity));
        }
    };
    report.first_stop_observation_digest = Some(first.digest.clone());
    if let Some(issue) = validate_stop(&first, supervisor_pid) {
        report.issues.push(issue);
        return Err(failure(finalize_blocked(report), confinement, continuity));
    }

    let second = match observe_stop(tracee_pid, policy.max_proc_status_bytes) {
        Ok(value) => value,
        Err(issue) => {
            report.issues.push(issue);
            return Err(failure(finalize_blocked(report), confinement, continuity));
        }
    };
    report.second_stop_observation_digest = Some(second.digest.clone());
    if let Some(issue) = validate_stop(&second, supervisor_pid) {
        report.issues.push(issue);
        return Err(failure(finalize_blocked(report), confinement, continuity));
    }
    if first != second {
        report.issues.push(BootstrapReleaseIssue::StopOwnershipChanged);
        return Err(failure(finalize_blocked(report), confinement, continuity));
    }

    let pid = Pid::from_raw(tracee_pid);
    if let Err(error) = ptrace::detach(pid, None::<Signal>) {
        report
            .issues
            .push(BootstrapReleaseIssue::DetachFailed(error.to_string()));
        return Err(failure(finalize_blocked(report), confinement, continuity));
    }

    report.ptrace_detach_returned_success = true;
    report.disposition = BootstrapReleaseDisposition::Released;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy");
    let release_digest = release_digest(
        &policy_digest,
        ready.qualification_digest(),
        mapped.qualification_digest(),
        ready.exec_confirmation_digest(),
        ready.syscall_sequence_digest(),
        mapped.launch_mapping_set_digest(),
        mapped.ready_mapping_set_digest(),
        tracee_pid,
        &second.digest,
        &report_digest,
    );

    let released = ReleasedBootstrapVerifier {
        release_digest,
        report_digest,
        policy_digest,
        tracee_pid,
        confinement_qualification_digest: ready.qualification_digest().into(),
        mapping_continuity_qualification_digest: mapped.qualification_digest().into(),
        exec_confirmation_digest: ready.exec_confirmation_digest().into(),
        syscall_sequence_digest: ready.syscall_sequence_digest().into(),
        launch_mapping_set_digest: mapped.launch_mapping_set_digest().into(),
        ready_mapping_set_digest: mapped.ready_mapping_set_digest().into(),
        pre_detach_stop_observation_digest: second.digest,
    };

    // Both parent qualifications are intentionally consumed on success.
    drop(confinement);
    drop(continuity);
    Ok(BootstrapReleaseSuccess { report, released })
}

fn observe_stop(pid: i32, maximum: u64) -> Result<StopObservation, BootstrapReleaseIssue> {
    let path = format!("/proc/{pid}/status");
    let bytes = fs::read(&path)
        .map_err(|error| BootstrapReleaseIssue::ProcStatusUnavailable(error.to_string()))?;
    if bytes.len() as u64 > maximum {
        return Err(BootstrapReleaseIssue::ProcStatusTooLarge {
            observed: bytes.len() as u64,
            maximum,
        });
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| BootstrapReleaseIssue::ProcStatusMalformed)?;
    let state_code = text
        .lines()
        .find_map(|line| line.strip_prefix("State:"))
        .map(str::trim)
        .and_then(|value| value.chars().next())
        .map(|value| value.to_string())
        .ok_or(BootstrapReleaseIssue::ProcStatusMalformed)?;
    let tracer_pid = text
        .lines()
        .find_map(|line| line.strip_prefix("TracerPid:"))
        .map(str::trim)
        .and_then(|value| value.parse::<i32>().ok())
        .ok_or(BootstrapReleaseIssue::ProcStatusMalformed)?;

    let mut h = blake3::Hasher::new();
    h.update(STOP_DOMAIN);
    h.update(&pid.to_be_bytes());
    field(&mut h, &state_code);
    h.update(&tracer_pid.to_be_bytes());
    Ok(StopObservation {
        state_code,
        tracer_pid,
        digest: blake3_text(h.finalize()),
    })
}

fn validate_stop(observed: &StopObservation, supervisor_pid: i32) -> Option<BootstrapReleaseIssue> {
    if observed.state_code != "t" {
        return Some(BootstrapReleaseIssue::TraceeNotInPtraceStop(
            observed.state_code.clone(),
        ));
    }
    if observed.tracer_pid != supervisor_pid {
        return Some(BootstrapReleaseIssue::TracerPidMismatch {
            expected: supervisor_pid,
            observed: observed.tracer_pid,
        });
    }
    None
}

fn finalize_blocked(mut report: BootstrapReleaseReport) -> BootstrapReleaseReport {
    report.disposition = if report.issues.iter().any(BootstrapReleaseIssue::invalid) {
        BootstrapReleaseDisposition::Invalid
    } else {
        BootstrapReleaseDisposition::Blocked
    };
    report
}

fn failure(
    report: BootstrapReleaseReport,
    confinement: BootstrapSyscallConfinementQualification,
    continuity: BootstrapExecutableMappingContinuityQualification,
) -> BootstrapReleaseFailure {
    BootstrapReleaseFailure {
        report,
        confinement,
        continuity,
    }
}

#[allow(clippy::too_many_arguments)]
fn release_digest(
    policy_digest: &str,
    confinement_qualification_digest: &str,
    mapping_continuity_qualification_digest: &str,
    exec_confirmation_digest: &str,
    syscall_sequence_digest: &str,
    launch_mapping_set_digest: &str,
    ready_mapping_set_digest: &str,
    tracee_pid: i32,
    stop_observation_digest: &str,
    report_digest: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(RELEASE_DOMAIN);
    for value in [
        policy_digest,
        confinement_qualification_digest,
        mapping_continuity_qualification_digest,
        exec_confirmation_digest,
        syscall_sequence_digest,
        launch_mapping_set_digest,
        ready_mapping_set_digest,
        stop_observation_digest,
        report_digest,
    ] {
        field(&mut h, value);
    }
    h.update(&tracee_pid.to_be_bytes());
    blake3_text(h.finalize())
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TEXT_BYTES
        && value == value.trim()
        && !value.as_bytes().contains(&0)
}

fn valid_blake3(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("blake3:")
        && value[7..].bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn valid_refs(values: &[String]) -> bool {
    if values.len() > MAX_EVIDENCE_REFS {
        return false;
    }
    let mut seen = BTreeSet::new();
    values
        .iter()
        .all(|value| canonical_text(value) && seen.insert(value.as_str()))
}

fn field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn sorted_strings(hasher: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.iter().map(String::as_str).collect::<Vec<_>>();
    values.sort_unstable();
    hasher.update(&(values.len() as u64).to_be_bytes());
    for value in values {
        field(hasher, value);
    }
}

fn blake3_text(hash: blake3::Hash) -> String {
    format!("blake3:{}", hash.to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> BootstrapReleasePolicy {
        BootstrapReleasePolicy {
            schema_version: BOOTSTRAP_RELEASE_POLICY_SCHEMA_V1.into(),
            policy_id: "bootstrap-release:v1".into(),
            expected_confinement_policy_digest: d("confinement-policy"),
            expected_mapping_continuity_policy_digest: d("mapping-policy"),
            max_proc_status_bytes: 128 * 1024,
            evidence_refs: vec!["review:bootstrap-release".into()],
        }
    }

    #[test]
    fn evidence_ref_order_is_nonsemantic() {
        let mut left = policy();
        left.evidence_refs = vec!["review:a".into(), "review:b".into()];
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn parent_policy_identities_are_semantic() {
        let left = policy();
        let mut right = left.clone();
        right.expected_mapping_continuity_policy_digest = d("different");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn invalid_status_bound_is_rejected() {
        let mut value = policy();
        value.max_proc_status_bytes = 0;
        assert!(!value.validate());
        value.max_proc_status_bytes = HARD_MAX_STATUS_BYTES + 1;
        assert!(!value.validate());
    }

    #[test]
    fn release_identity_binds_both_parent_qualifications() {
        let base = release_digest(
            &d("policy"),
            &d("confinement"),
            &d("continuity"),
            &d("exec"),
            &d("syscalls"),
            &d("launch-maps"),
            &d("ready-maps"),
            4242,
            &d("stop"),
            &d("report"),
        );
        let changed = release_digest(
            &d("policy"),
            &d("other-confinement"),
            &d("continuity"),
            &d("exec"),
            &d("syscalls"),
            &d("launch-maps"),
            &d("ready-maps"),
            4242,
            &d("stop"),
            &d("report"),
        );
        assert_ne!(base, changed);
    }
}
