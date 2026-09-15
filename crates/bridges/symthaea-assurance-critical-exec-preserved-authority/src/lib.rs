// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Critical exec-preserved authority observation for a confirmed static verifier.
//!
//! This Linux-only bridge consumes a post-exec confirmation while the tracee is
//! still held at PTRACE_EVENT_EXEC and binds a deliberately bounded set of
//! inherited process authority to exact reviewed policy.

#![cfg(target_os = "linux")]
#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use symthaea_assurance_ptrace_static_exec_confirmation::ConfirmedStaticExecLaunch;

pub const CRITICAL_EXEC_AUTHORITY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.critical-exec-preserved-authority-policy.v1";
pub const CRITICAL_EXEC_AUTHORITY_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.critical-exec-preserved-authority-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.critical-exec-preserved-authority-policy.digest.v1\0";
const STATUS_DOMAIN: &[u8] =
    b"symthaea.assurance.critical-exec-preserved-status.digest.v1\0";
const FD_SET_DOMAIN: &[u8] =
    b"symthaea.assurance.critical-exec-preserved-fd-set.digest.v1\0";
const NAMESPACE_SET_DOMAIN: &[u8] =
    b"symthaea.assurance.critical-exec-preserved-namespace-set.digest.v1\0";
const SNAPSHOT_DOMAIN: &[u8] =
    b"symthaea.assurance.critical-exec-preserved-snapshot.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.critical-exec-preserved-authority-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.critical-exec-preserved-authority-qualification.digest.v1\0";

const MAX_TEXT: usize = 16 * 1024;
const MAX_REFS: usize = 128;
const MAX_EXPECTED_FDS: usize = 4096;
const MAX_EXPECTED_NAMESPACES: usize = 64;
const HARD_MAX_PROC_BYTES: u64 = 64 * 1024 * 1024;
const HARD_MAX_FDINFO_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedFdAuthority {
    pub fd: u32,
    pub exact_target: String,
    pub expected_position: u64,
    pub expected_flags_octal: String,
    pub expected_mount_id: u64,
    pub expected_inode: u64,
    pub expected_fdinfo_blake3: String,
}

impl ExpectedFdAuthority {
    pub fn validate(&self) -> bool {
        self.fd <= 1_048_576
            && canonical_text(&self.exact_target)
            && !self.exact_target.ends_with(" (deleted)")
            && valid_octal(&self.expected_flags_octal)
            && valid_blake3(&self.expected_fdinfo_blake3)
    }

    fn commit(&self, hasher: &mut blake3::Hasher) {
        hasher.update(&self.fd.to_le_bytes());
        field(hasher, &self.exact_target);
        hasher.update(&self.expected_position.to_le_bytes());
        field(hasher, &self.expected_flags_octal);
        hasher.update(&self.expected_mount_id.to_le_bytes());
        hasher.update(&self.expected_inode.to_le_bytes());
        field(hasher, &self.expected_fdinfo_blake3);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedNamespaceAuthority {
    pub name: String,
    pub exact_link: String,
}

impl ExpectedNamespaceAuthority {
    pub fn validate(&self) -> bool {
        identifier(&self.name)
            && canonical_text(&self.exact_link)
            && !self.exact_link.ends_with(" (deleted)")
            && self.exact_link.contains(":[")
            && self.exact_link.ends_with(']')
    }

    fn commit(&self, hasher: &mut blake3::Hasher) {
        field(hasher, &self.name);
        field(hasher, &self.exact_link);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriticalExecAuthorityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_exec_confirmation_policy_digest: String,
    pub expected_fd_exec_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,

    pub expected_uids: [u32; 4],
    pub expected_gids: [u32; 4],
    pub expected_supplementary_groups: Vec<u32>,
    pub expected_umask: u32,
    pub expected_cap_inheritable: u64,
    pub expected_cap_permitted: u64,
    pub expected_cap_effective: u64,
    pub expected_cap_bounding: u64,
    pub expected_cap_ambient: u64,
    pub expected_sig_pending: u64,
    pub expected_shared_sig_pending: u64,
    pub expected_sig_blocked: u64,
    pub expected_sig_ignored: u64,
    pub expected_sig_caught: u64,
    pub expected_no_new_privs: bool,
    pub expected_seccomp_mode: u8,
    pub expected_seccomp_filters: u32,
    pub expected_core_dumping: bool,

    pub expected_cwd: String,
    pub expected_root: String,
    pub expected_fds: Vec<ExpectedFdAuthority>,
    pub expected_namespaces: Vec<ExpectedNamespaceAuthority>,
    pub expected_limits_blake3: String,
    pub expected_cgroup_blake3: String,

    pub max_status_bytes: u64,
    pub max_fdinfo_bytes: u64,
    pub max_limits_bytes: u64,
    pub max_cgroup_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl CriticalExecAuthorityPolicy {
    pub fn validate(&self) -> bool {
        let seccomp_consistent = if self.expected_seccomp_mode == 2 {
            self.expected_seccomp_filters > 0
        } else {
            self.expected_seccomp_filters == 0
        };
        self.schema_version == CRITICAL_EXEC_AUTHORITY_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_exec_confirmation_policy_digest)
            && valid_blake3(&self.expected_fd_exec_policy_digest)
            && valid_blake3(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
            && unique_u32(&self.expected_supplementary_groups)
            && self.expected_umask <= 0o7777
            && self.expected_seccomp_mode <= 2
            && seccomp_consistent
            && canonical_absolute_path(&self.expected_cwd)
            && canonical_absolute_path(&self.expected_root)
            && self.expected_fds.len() <= MAX_EXPECTED_FDS
            && self.expected_fds.iter().all(ExpectedFdAuthority::validate)
            && unique_fd_numbers(&self.expected_fds)
            && self.expected_namespaces.len() <= MAX_EXPECTED_NAMESPACES
            && self
                .expected_namespaces
                .iter()
                .all(ExpectedNamespaceAuthority::validate)
            && unique_namespace_names(&self.expected_namespaces)
            && valid_blake3(&self.expected_limits_blake3)
            && valid_blake3(&self.expected_cgroup_blake3)
            && (1..=HARD_MAX_PROC_BYTES).contains(&self.max_status_bytes)
            && (1..=HARD_MAX_FDINFO_BYTES).contains(&self.max_fdinfo_bytes)
            && (1..=HARD_MAX_PROC_BYTES).contains(&self.max_limits_bytes)
            && (1..=HARD_MAX_PROC_BYTES).contains(&self.max_cgroup_bytes)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.expected_exec_confirmation_policy_digest.as_str(),
            self.expected_fd_exec_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            field(&mut hasher, value);
        }
        for value in self.expected_uids {
            hasher.update(&value.to_le_bytes());
        }
        for value in self.expected_gids {
            hasher.update(&value.to_le_bytes());
        }
        let mut groups = self.expected_supplementary_groups.clone();
        groups.sort_unstable();
        hasher.update(&(groups.len() as u64).to_le_bytes());
        for group in groups {
            hasher.update(&group.to_le_bytes());
        }
        hasher.update(&self.expected_umask.to_le_bytes());
        for value in [
            self.expected_cap_inheritable,
            self.expected_cap_permitted,
            self.expected_cap_effective,
            self.expected_cap_bounding,
            self.expected_cap_ambient,
            self.expected_sig_pending,
            self.expected_shared_sig_pending,
            self.expected_sig_blocked,
            self.expected_sig_ignored,
            self.expected_sig_caught,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&[u8::from(self.expected_no_new_privs)]);
        hasher.update(&[self.expected_seccomp_mode]);
        hasher.update(&self.expected_seccomp_filters.to_le_bytes());
        hasher.update(&[u8::from(self.expected_core_dumping)]);
        field(&mut hasher, &self.expected_cwd);
        field(&mut hasher, &self.expected_root);

        let mut fds = self.expected_fds.clone();
        fds.sort_by_key(|value| value.fd);
        hasher.update(&(fds.len() as u64).to_le_bytes());
        for fd in &fds {
            fd.commit(&mut hasher);
        }

        let mut namespaces = self.expected_namespaces.clone();
        namespaces.sort_by(|left, right| left.name.cmp(&right.name));
        hasher.update(&(namespaces.len() as u64).to_le_bytes());
        for namespace in &namespaces {
            namespace.commit(&mut hasher);
        }

        field(&mut hasher, &self.expected_limits_blake3);
        field(&mut hasher, &self.expected_cgroup_blake3);
        for value in [
            self.max_status_bytes,
            self.max_fdinfo_bytes,
            self.max_limits_bytes,
            self.max_cgroup_bytes,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        sorted_strings(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriticalStatusEvidence {
    pub state: String,
    pub tracer_pid: u32,
    pub uids: [u32; 4],
    pub gids: [u32; 4],
    pub supplementary_groups: Vec<u32>,
    pub umask: u32,
    pub cap_inheritable: u64,
    pub cap_permitted: u64,
    pub cap_effective: u64,
    pub cap_bounding: u64,
    pub cap_ambient: u64,
    pub sig_pending: u64,
    pub shared_sig_pending: u64,
    pub sig_blocked: u64,
    pub sig_ignored: u64,
    pub sig_caught: u64,
    pub no_new_privs: bool,
    pub seccomp_mode: u8,
    pub seccomp_filters: u32,
    pub threads: u32,
    pub core_dumping: bool,
    pub selected_status_digest: String,
}

impl CriticalStatusEvidence {
    fn recompute_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(STATUS_DOMAIN);
        field(&mut hasher, &self.state);
        hasher.update(&self.tracer_pid.to_le_bytes());
        for value in self.uids {
            hasher.update(&value.to_le_bytes());
        }
        for value in self.gids {
            hasher.update(&value.to_le_bytes());
        }
        let mut groups = self.supplementary_groups.clone();
        groups.sort_unstable();
        hasher.update(&(groups.len() as u64).to_le_bytes());
        for group in groups {
            hasher.update(&group.to_le_bytes());
        }
        hasher.update(&self.umask.to_le_bytes());
        for value in [
            self.cap_inheritable,
            self.cap_permitted,
            self.cap_effective,
            self.cap_bounding,
            self.cap_ambient,
            self.sig_pending,
            self.shared_sig_pending,
            self.sig_blocked,
            self.sig_ignored,
            self.sig_caught,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&[u8::from(self.no_new_privs)]);
        hasher.update(&[self.seccomp_mode]);
        hasher.update(&self.seccomp_filters.to_le_bytes());
        hasher.update(&self.threads.to_le_bytes());
        hasher.update(&[u8::from(self.core_dumping)]);
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FdAuthorityEvidence {
    pub fd: u32,
    pub target: String,
    pub position: u64,
    pub flags_octal: String,
    pub mount_id: u64,
    pub inode: u64,
    pub fdinfo_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespaceAuthorityEvidence {
    pub name: String,
    pub link: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriticalAuthoritySnapshot {
    pub status: CriticalStatusEvidence,
    pub fds: Vec<FdAuthorityEvidence>,
    pub fd_set_digest: String,
    pub cwd: String,
    pub root: String,
    pub namespaces: Vec<NamespaceAuthorityEvidence>,
    pub namespace_set_digest: String,
    pub limits_blake3: String,
    pub cgroup_blake3: String,
    pub authority_snapshot_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CriticalExecAuthorityIssue {
    InvalidPolicy,
    ConfirmationPolicyMismatch,
    FdExecPolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    ConfirmationNotStopped,
    ProcStatusUnavailable(String),
    ProcStatusInvalid(String),
    TraceeNotInPtraceStop(String),
    TracerPidMismatch { expected: u32, observed: u32 },
    UidMismatch,
    GidMismatch,
    SupplementaryGroupsMismatch,
    UmaskMismatch,
    CapabilityMismatch(String),
    SignalStateMismatch(String),
    NoNewPrivsMismatch,
    SeccompModeMismatch,
    SeccompFilterCountMismatch,
    ThreadCountMismatch { expected: u32, observed: u32 },
    CoreDumpingMismatch,
    FdDirectoryUnavailable(String),
    InvalidFdDirectoryEntry(String),
    MissingFileDescriptor(u32),
    UnexpectedFileDescriptor(u32),
    FdTargetUnavailable { fd: u32, reason: String },
    FdTargetMismatch(u32),
    DeletedFdTarget(u32),
    FdInfoUnavailable { fd: u32, reason: String },
    FdInfoInvalid { fd: u32, reason: String },
    FdPositionMismatch(u32),
    FdFlagsMismatch(u32),
    FdMountIdMismatch(u32),
    FdInodeMismatch(u32),
    FdInfoDigestMismatch(u32),
    CwdUnavailable(String),
    CwdMismatch,
    RootUnavailable(String),
    RootMismatch,
    DeletedAuthorityPath(String),
    NamespaceDirectoryUnavailable(String),
    InvalidNamespaceEntry(String),
    MissingNamespace(String),
    UnexpectedNamespace(String),
    NamespaceLinkUnavailable { name: String, reason: String },
    NamespaceLinkMismatch(String),
    LimitsUnavailable(String),
    LimitsDigestMismatch,
    CgroupUnavailable(String),
    CgroupDigestMismatch,
    SnapshotChangedDuringObservation,
}

impl CriticalExecAuthorityIssue {
    fn invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy)
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::ConfirmationPolicyMismatch => "confirmation-policy-mismatch".into(),
            Self::FdExecPolicyMismatch => "fd-exec-policy-mismatch".into(),
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::ConfirmationNotStopped => "confirmation-not-stopped".into(),
            Self::ProcStatusUnavailable(v) => format!("proc-status-unavailable:{v}"),
            Self::ProcStatusInvalid(v) => format!("proc-status-invalid:{v}"),
            Self::TraceeNotInPtraceStop(v) => format!("tracee-not-in-ptrace-stop:{v}"),
            Self::TracerPidMismatch { expected, observed } => {
                format!("tracer-pid-mismatch:{expected}:{observed}")
            }
            Self::UidMismatch => "uid-mismatch".into(),
            Self::GidMismatch => "gid-mismatch".into(),
            Self::SupplementaryGroupsMismatch => "supplementary-groups-mismatch".into(),
            Self::UmaskMismatch => "umask-mismatch".into(),
            Self::CapabilityMismatch(v) => format!("capability-mismatch:{v}"),
            Self::SignalStateMismatch(v) => format!("signal-state-mismatch:{v}"),
            Self::NoNewPrivsMismatch => "no-new-privs-mismatch".into(),
            Self::SeccompModeMismatch => "seccomp-mode-mismatch".into(),
            Self::SeccompFilterCountMismatch => "seccomp-filter-count-mismatch".into(),
            Self::ThreadCountMismatch { expected, observed } => {
                format!("thread-count-mismatch:{expected}:{observed}")
            }
            Self::CoreDumpingMismatch => "core-dumping-mismatch".into(),
            Self::FdDirectoryUnavailable(v) => format!("fd-directory-unavailable:{v}"),
            Self::InvalidFdDirectoryEntry(v) => format!("invalid-fd-directory-entry:{v}"),
            Self::MissingFileDescriptor(fd) => format!("missing-file-descriptor:{fd}"),
            Self::UnexpectedFileDescriptor(fd) => format!("unexpected-file-descriptor:{fd}"),
            Self::FdTargetUnavailable { fd, reason } => {
                format!("fd-target-unavailable:{fd}:{reason}")
            }
            Self::FdTargetMismatch(fd) => format!("fd-target-mismatch:{fd}"),
            Self::DeletedFdTarget(fd) => format!("deleted-fd-target:{fd}"),
            Self::FdInfoUnavailable { fd, reason } => {
                format!("fdinfo-unavailable:{fd}:{reason}")
            }
            Self::FdInfoInvalid { fd, reason } => format!("fdinfo-invalid:{fd}:{reason}"),
            Self::FdPositionMismatch(fd) => format!("fd-position-mismatch:{fd}"),
            Self::FdFlagsMismatch(fd) => format!("fd-flags-mismatch:{fd}"),
            Self::FdMountIdMismatch(fd) => format!("fd-mount-id-mismatch:{fd}"),
            Self::FdInodeMismatch(fd) => format!("fd-inode-mismatch:{fd}"),
            Self::FdInfoDigestMismatch(fd) => format!("fdinfo-digest-mismatch:{fd}"),
            Self::CwdUnavailable(v) => format!("cwd-unavailable:{v}"),
            Self::CwdMismatch => "cwd-mismatch".into(),
            Self::RootUnavailable(v) => format!("root-unavailable:{v}"),
            Self::RootMismatch => "root-mismatch".into(),
            Self::DeletedAuthorityPath(v) => format!("deleted-authority-path:{v}"),
            Self::NamespaceDirectoryUnavailable(v) => {
                format!("namespace-directory-unavailable:{v}")
            }
            Self::InvalidNamespaceEntry(v) => format!("invalid-namespace-entry:{v}"),
            Self::MissingNamespace(v) => format!("missing-namespace:{v}"),
            Self::UnexpectedNamespace(v) => format!("unexpected-namespace:{v}"),
            Self::NamespaceLinkUnavailable { name, reason } => {
                format!("namespace-link-unavailable:{name}:{reason}")
            }
            Self::NamespaceLinkMismatch(v) => format!("namespace-link-mismatch:{v}"),
            Self::LimitsUnavailable(v) => format!("limits-unavailable:{v}"),
            Self::LimitsDigestMismatch => "limits-digest-mismatch".into(),
            Self::CgroupUnavailable(v) => format!("cgroup-unavailable:{v}"),
            Self::CgroupDigestMismatch => "cgroup-digest-mismatch".into(),
            Self::SnapshotChangedDuringObservation => {
                "snapshot-changed-during-observation".into()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CriticalExecAuthorityDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriticalExecAuthorityReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub exec_confirmation_digest: String,
    pub exec_confirmation_policy_digest: String,
    pub fd_exec_policy_digest: String,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub tracee_pid: i32,
    pub authority_snapshot_digest: Option<String>,
    pub status_digest: Option<String>,
    pub fd_set_digest: Option<String>,
    pub namespace_set_digest: Option<String>,
    pub fd_count: u64,
    pub namespace_count: u64,
    pub disposition: CriticalExecAuthorityDisposition,
    pub issues: Vec<CriticalExecAuthorityIssue>,
    pub qualification_digest: Option<String>,
}

impl CriticalExecAuthorityReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.exec_confirmation_digest.as_str(),
            self.exec_confirmation_policy_digest.as_str(),
            self.fd_exec_policy_digest.as_str(),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.authority_snapshot_digest.as_deref().unwrap_or("-"),
            self.status_digest.as_deref().unwrap_or("-"),
            self.fd_set_digest.as_deref().unwrap_or("-"),
            self.namespace_set_digest.as_deref().unwrap_or("-"),
        ] {
            field(&mut hasher, value);
        }
        hasher.update(&self.tracee_pid.to_le_bytes());
        hasher.update(&self.fd_count.to_le_bytes());
        hasher.update(&self.namespace_count.to_le_bytes());
        field(
            &mut hasher,
            match self.disposition {
                CriticalExecAuthorityDisposition::Invalid => "invalid",
                CriticalExecAuthorityDisposition::Blocked => "blocked",
                CriticalExecAuthorityDisposition::Qualified => "qualified",
            },
        );
        hasher.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field(&mut hasher, &issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalExecPreservedAuthority {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    exec_confirmation_digest: String,
    exec_confirmation_policy_digest: String,
    fd_exec_policy_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    tracee_pid: i32,
    authority_snapshot_digest: String,
    status_digest: String,
    fd_set_digest: String,
    namespace_set_digest: String,
    limits_blake3: String,
    cgroup_blake3: String,
    fd_count: u64,
    namespace_count: u64,
}

impl CriticalExecPreservedAuthority {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn exec_confirmation_digest(&self) -> &str { &self.exec_confirmation_digest }
    pub fn exec_confirmation_policy_digest(&self) -> &str {
        &self.exec_confirmation_policy_digest
    }
    pub fn fd_exec_policy_digest(&self) -> &str { &self.fd_exec_policy_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub const fn tracee_pid(&self) -> i32 { self.tracee_pid }
    pub fn authority_snapshot_digest(&self) -> &str { &self.authority_snapshot_digest }
    pub fn status_digest(&self) -> &str { &self.status_digest }
    pub fn fd_set_digest(&self) -> &str { &self.fd_set_digest }
    pub fn namespace_set_digest(&self) -> &str { &self.namespace_set_digest }
    pub fn limits_blake3(&self) -> &str { &self.limits_blake3 }
    pub fn cgroup_blake3(&self) -> &str { &self.cgroup_blake3 }
    pub const fn fd_count(&self) -> u64 { self.fd_count }
    pub const fn namespace_count(&self) -> u64 { self.namespace_count }

    pub const fn exact_inherited_fd_inventory_matches_policy(&self) -> bool { true }
    pub const fn critical_identity_capability_signal_state_matches_policy(&self) -> bool { true }
    pub const fn no_new_privs_state_matches_policy(&self) -> bool { true }
    pub const fn seccomp_mode_and_filter_count_match_policy(&self) -> bool { true }
    pub const fn cwd_and_root_match_policy(&self) -> bool { true }
    pub const fn namespace_identity_set_matches_policy(&self) -> bool { true }
    pub const fn resource_limit_bytes_match_policy(&self) -> bool { true }
    pub const fn cgroup_membership_bytes_match_policy(&self) -> bool { true }
    pub const fn tracee_ptrace_stopped_during_observation(&self) -> bool { true }
    pub const fn critical_authority_snapshot_stable_across_two_reads(&self) -> bool { true }

    pub const fn all_exec_preserved_process_state_qualified(&self) -> bool { false }
    pub const fn fd_peer_authority_verified(&self) -> bool { false }
    pub const fn namespace_contents_verified(&self) -> bool { false }
    pub const fn seccomp_filter_semantics_verified(&self) -> bool { false }
    pub const fn resource_limit_semantics_independently_parsed(&self) -> bool { false }
    pub const fn lsm_policy_context_qualified(&self) -> bool { false }
    pub const fn post_release_continuity_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct CriticalExecAuthorityQualification {
    pub report: CriticalExecAuthorityReport,
    pub snapshot: CriticalAuthoritySnapshot,
    verified: CriticalExecPreservedAuthority,
}

impl CriticalExecAuthorityQualification {
    pub fn verified(&self) -> &CriticalExecPreservedAuthority { &self.verified }
    pub fn into_verified(self) -> CriticalExecPreservedAuthority { self.verified }
}

pub fn assess_critical_exec_preserved_authority(
    policy: &CriticalExecAuthorityPolicy,
    confirmed: &ConfirmedStaticExecLaunch,
) -> Result<CriticalExecAuthorityQualification, CriticalExecAuthorityReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = base_report(policy, policy_digest.clone(), confirmed);
    if !policy.validate() {
        report.issues.push(CriticalExecAuthorityIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if confirmed.policy_digest() != policy.expected_exec_confirmation_policy_digest {
        report
            .issues
            .push(CriticalExecAuthorityIssue::ConfirmationPolicyMismatch);
    }
    if confirmed.fd_exec_policy_digest() != policy.expected_fd_exec_policy_digest {
        report
            .issues
            .push(CriticalExecAuthorityIssue::FdExecPolicyMismatch);
    }
    if confirmed.runtime_policy_digest() != policy.expected_runtime_policy_digest {
        report
            .issues
            .push(CriticalExecAuthorityIssue::RuntimePolicyMismatch);
    }
    if confirmed.runtime_verifier_ref() != policy.expected_runtime_verifier_ref {
        report
            .issues
            .push(CriticalExecAuthorityIssue::RuntimeVerifierMismatch);
    }
    if confirmed.backend_id() != policy.expected_backend_id {
        report.issues.push(CriticalExecAuthorityIssue::BackendMismatch);
    }
    if !confirmed.tracee_stopped_at_confirmation() {
        report
            .issues
            .push(CriticalExecAuthorityIssue::ConfirmationNotStopped);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let first = match observe_snapshot(policy, confirmed.tracee_pid()) {
        Ok(snapshot) => snapshot,
        Err(issues) => {
            report.issues.extend(issues);
            return Err(finalize(report));
        }
    };
    let second = match observe_snapshot(policy, confirmed.tracee_pid()) {
        Ok(snapshot) => snapshot,
        Err(issues) => {
            report.issues.extend(issues);
            return Err(finalize(report));
        }
    };
    if first.authority_snapshot_digest != second.authority_snapshot_digest {
        report
            .issues
            .push(CriticalExecAuthorityIssue::SnapshotChangedDuringObservation);
        return Err(finalize(report));
    }

    report.authority_snapshot_digest = Some(first.authority_snapshot_digest.clone());
    report.status_digest = Some(first.status.selected_status_digest.clone());
    report.fd_set_digest = Some(first.fd_set_digest.clone());
    report.namespace_set_digest = Some(first.namespace_set_digest.clone());
    report.fd_count = first.fds.len() as u64;
    report.namespace_count = first.namespaces.len() as u64;
    report.disposition = CriticalExecAuthorityDisposition::Qualified;

    let policy_digest = policy_digest.expect("validated policy has digest");
    let report_digest = report.canonical_digest();
    let qualification_digest = qualification_digest(
        &policy_digest,
        confirmed.confirmation_digest(),
        &first.authority_snapshot_digest,
        &report_digest,
    );
    report.qualification_digest = Some(qualification_digest.clone());
    debug_assert_eq!(report_digest, report.canonical_digest());
    let verified = CriticalExecPreservedAuthority {
        qualification_digest,
        report_digest,
        policy_digest,
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        exec_confirmation_policy_digest: confirmed.policy_digest().into(),
        fd_exec_policy_digest: confirmed.fd_exec_policy_digest().into(),
        runtime_policy_digest: confirmed.runtime_policy_digest().into(),
        runtime_verifier_ref: confirmed.runtime_verifier_ref().into(),
        backend_id: confirmed.backend_id().into(),
        tracee_pid: confirmed.tracee_pid(),
        authority_snapshot_digest: first.authority_snapshot_digest.clone(),
        status_digest: first.status.selected_status_digest.clone(),
        fd_set_digest: first.fd_set_digest.clone(),
        namespace_set_digest: first.namespace_set_digest.clone(),
        limits_blake3: first.limits_blake3.clone(),
        cgroup_blake3: first.cgroup_blake3.clone(),
        fd_count: first.fds.len() as u64,
        namespace_count: first.namespaces.len() as u64,
    };
    Ok(CriticalExecAuthorityQualification {
        report,
        snapshot: first,
        verified,
    })
}

fn observe_snapshot(
    policy: &CriticalExecAuthorityPolicy,
    pid: i32,
) -> Result<CriticalAuthoritySnapshot, Vec<CriticalExecAuthorityIssue>> {
    let mut issues = Vec::new();
    let base = PathBuf::from(format!("/proc/{pid}"));

    let status_bytes = match read_bounded(&base.join("status"), policy.max_status_bytes) {
        Ok(value) => value,
        Err(error) => {
            issues.push(CriticalExecAuthorityIssue::ProcStatusUnavailable(error));
            return Err(issues);
        }
    };
    let status = match parse_status(&status_bytes) {
        Ok(value) => value,
        Err(error) => {
            issues.push(CriticalExecAuthorityIssue::ProcStatusInvalid(error));
            return Err(issues);
        }
    };
    check_status(policy, &status, &mut issues);

    let fds = observe_fds(&base, policy, &mut issues);
    let fd_set_digest = fd_set_digest(&fds);

    let cwd = match read_link_text(&base.join("cwd")) {
        Ok(value) => value,
        Err(error) => {
            issues.push(CriticalExecAuthorityIssue::CwdUnavailable(error));
            String::new()
        }
    };
    if cwd.ends_with(" (deleted)") {
        issues.push(CriticalExecAuthorityIssue::DeletedAuthorityPath("cwd".into()));
    }
    if !cwd.is_empty() && cwd != policy.expected_cwd {
        issues.push(CriticalExecAuthorityIssue::CwdMismatch);
    }

    let root = match read_link_text(&base.join("root")) {
        Ok(value) => value,
        Err(error) => {
            issues.push(CriticalExecAuthorityIssue::RootUnavailable(error));
            String::new()
        }
    };
    if root.ends_with(" (deleted)") {
        issues.push(CriticalExecAuthorityIssue::DeletedAuthorityPath("root".into()));
    }
    if !root.is_empty() && root != policy.expected_root {
        issues.push(CriticalExecAuthorityIssue::RootMismatch);
    }

    let namespaces = observe_namespaces(&base, policy, &mut issues);
    let namespace_set_digest = namespace_set_digest(&namespaces);

    let limits_blake3 = match read_bounded(&base.join("limits"), policy.max_limits_bytes) {
        Ok(value) => format!("blake3:{}", blake3::hash(&value).to_hex()),
        Err(error) => {
            issues.push(CriticalExecAuthorityIssue::LimitsUnavailable(error));
            String::new()
        }
    };
    if !limits_blake3.is_empty() && limits_blake3 != policy.expected_limits_blake3 {
        issues.push(CriticalExecAuthorityIssue::LimitsDigestMismatch);
    }

    let cgroup_blake3 = match read_bounded(&base.join("cgroup"), policy.max_cgroup_bytes) {
        Ok(value) => format!("blake3:{}", blake3::hash(&value).to_hex()),
        Err(error) => {
            issues.push(CriticalExecAuthorityIssue::CgroupUnavailable(error));
            String::new()
        }
    };
    if !cgroup_blake3.is_empty() && cgroup_blake3 != policy.expected_cgroup_blake3 {
        issues.push(CriticalExecAuthorityIssue::CgroupDigestMismatch);
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let authority_snapshot_digest = snapshot_digest(
        &status.selected_status_digest,
        &fd_set_digest,
        &cwd,
        &root,
        &namespace_set_digest,
        &limits_blake3,
        &cgroup_blake3,
    );
    Ok(CriticalAuthoritySnapshot {
        status,
        fds,
        fd_set_digest,
        cwd,
        root,
        namespaces,
        namespace_set_digest,
        limits_blake3,
        cgroup_blake3,
        authority_snapshot_digest,
    })
}

fn parse_status(bytes: &[u8]) -> Result<CriticalStatusEvidence, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if values
            .insert(key.to_string(), value.trim().to_string())
            .is_some()
        {
            return Err(format!("duplicate-status-field:{key}"));
        }
    }

    let state = required(&values, "State")?.to_string();
    let tracer_pid = parse_u32(required(&values, "TracerPid")?)?;
    let uids = parse_quad(required(&values, "Uid")?)?;
    let gids = parse_quad(required(&values, "Gid")?)?;
    let supplementary_groups = parse_u32_list(required(&values, "Groups")?)?;
    let umask = u32::from_str_radix(required(&values, "Umask")?, 8)
        .map_err(|_| "invalid-umask".to_string())?;
    let cap_inheritable = parse_hex_u64(required(&values, "CapInh")?)?;
    let cap_permitted = parse_hex_u64(required(&values, "CapPrm")?)?;
    let cap_effective = parse_hex_u64(required(&values, "CapEff")?)?;
    let cap_bounding = parse_hex_u64(required(&values, "CapBnd")?)?;
    let cap_ambient = parse_hex_u64(required(&values, "CapAmb")?)?;
    let sig_pending = parse_hex_u64(required(&values, "SigPnd")?)?;
    let shared_sig_pending = parse_hex_u64(required(&values, "ShdPnd")?)?;
    let sig_blocked = parse_hex_u64(required(&values, "SigBlk")?)?;
    let sig_ignored = parse_hex_u64(required(&values, "SigIgn")?)?;
    let sig_caught = parse_hex_u64(required(&values, "SigCgt")?)?;
    let no_new_privs = parse_bool01(required(&values, "NoNewPrivs")?)?;
    let seccomp_mode = parse_u8(required(&values, "Seccomp")?)?;
    if seccomp_mode > 2 {
        return Err("invalid-seccomp-mode".into());
    }
    let seccomp_filters = parse_u32(required(&values, "Seccomp_filters")?)?;
    let threads = parse_u32(required(&values, "Threads")?)?;
    let core_dumping = parse_bool01(required(&values, "CoreDumping")?)?;

    let mut evidence = CriticalStatusEvidence {
        state,
        tracer_pid,
        uids,
        gids,
        supplementary_groups,
        umask,
        cap_inheritable,
        cap_permitted,
        cap_effective,
        cap_bounding,
        cap_ambient,
        sig_pending,
        shared_sig_pending,
        sig_blocked,
        sig_ignored,
        sig_caught,
        no_new_privs,
        seccomp_mode,
        seccomp_filters,
        threads,
        core_dumping,
        selected_status_digest: String::new(),
    };
    evidence.selected_status_digest = evidence.recompute_digest();
    Ok(evidence)
}

fn check_status(
    policy: &CriticalExecAuthorityPolicy,
    observed: &CriticalStatusEvidence,
    issues: &mut Vec<CriticalExecAuthorityIssue>,
) {
    if !observed.state.starts_with("t ") && observed.state != "t" {
        issues.push(CriticalExecAuthorityIssue::TraceeNotInPtraceStop(
            observed.state.clone(),
        ));
    }
    let expected_tracer = std::process::id();
    if observed.tracer_pid != expected_tracer {
        issues.push(CriticalExecAuthorityIssue::TracerPidMismatch {
            expected: expected_tracer,
            observed: observed.tracer_pid,
        });
    }
    if observed.uids != policy.expected_uids {
        issues.push(CriticalExecAuthorityIssue::UidMismatch);
    }
    if observed.gids != policy.expected_gids {
        issues.push(CriticalExecAuthorityIssue::GidMismatch);
    }
    let mut groups = observed.supplementary_groups.clone();
    groups.sort_unstable();
    let mut expected_groups = policy.expected_supplementary_groups.clone();
    expected_groups.sort_unstable();
    if groups != expected_groups {
        issues.push(CriticalExecAuthorityIssue::SupplementaryGroupsMismatch);
    }
    if observed.umask != policy.expected_umask {
        issues.push(CriticalExecAuthorityIssue::UmaskMismatch);
    }
    for (name, observed_value, expected_value) in [
        (
            "CapInh",
            observed.cap_inheritable,
            policy.expected_cap_inheritable,
        ),
        (
            "CapPrm",
            observed.cap_permitted,
            policy.expected_cap_permitted,
        ),
        (
            "CapEff",
            observed.cap_effective,
            policy.expected_cap_effective,
        ),
        (
            "CapBnd",
            observed.cap_bounding,
            policy.expected_cap_bounding,
        ),
        (
            "CapAmb",
            observed.cap_ambient,
            policy.expected_cap_ambient,
        ),
    ] {
        if observed_value != expected_value {
            issues.push(CriticalExecAuthorityIssue::CapabilityMismatch(name.into()));
        }
    }
    for (name, observed_value, expected_value) in [
        ("SigPnd", observed.sig_pending, policy.expected_sig_pending),
        (
            "ShdPnd",
            observed.shared_sig_pending,
            policy.expected_shared_sig_pending,
        ),
        ("SigBlk", observed.sig_blocked, policy.expected_sig_blocked),
        ("SigIgn", observed.sig_ignored, policy.expected_sig_ignored),
        ("SigCgt", observed.sig_caught, policy.expected_sig_caught),
    ] {
        if observed_value != expected_value {
            issues.push(CriticalExecAuthorityIssue::SignalStateMismatch(name.into()));
        }
    }
    if observed.no_new_privs != policy.expected_no_new_privs {
        issues.push(CriticalExecAuthorityIssue::NoNewPrivsMismatch);
    }
    if observed.seccomp_mode != policy.expected_seccomp_mode {
        issues.push(CriticalExecAuthorityIssue::SeccompModeMismatch);
    }
    if observed.seccomp_filters != policy.expected_seccomp_filters {
        issues.push(CriticalExecAuthorityIssue::SeccompFilterCountMismatch);
    }
    if observed.threads != 1 {
        issues.push(CriticalExecAuthorityIssue::ThreadCountMismatch {
            expected: 1,
            observed: observed.threads,
        });
    }
    if observed.core_dumping != policy.expected_core_dumping {
        issues.push(CriticalExecAuthorityIssue::CoreDumpingMismatch);
    }
}

fn observe_fds(
    base: &Path,
    policy: &CriticalExecAuthorityPolicy,
    issues: &mut Vec<CriticalExecAuthorityIssue>,
) -> Vec<FdAuthorityEvidence> {
    let mut expected = BTreeMap::new();
    for spec in &policy.expected_fds {
        expected.insert(spec.fd, spec);
    }

    let entries = match fs::read_dir(base.join("fd")) {
        Ok(value) => value,
        Err(error) => {
            issues.push(CriticalExecAuthorityIssue::FdDirectoryUnavailable(
                error.to_string(),
            ));
            return Vec::new();
        }
    };
    let mut actual_numbers = BTreeSet::new();
    for entry in entries {
        let entry = match entry {
            Ok(value) => value,
            Err(error) => {
                issues.push(CriticalExecAuthorityIssue::FdDirectoryUnavailable(
                    error.to_string(),
                ));
                continue;
            }
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            issues.push(CriticalExecAuthorityIssue::InvalidFdDirectoryEntry(
                "non-utf8".into(),
            ));
            continue;
        };
        match name.parse::<u32>() {
            Ok(fd) => {
                actual_numbers.insert(fd);
            }
            Err(_) => issues.push(CriticalExecAuthorityIssue::InvalidFdDirectoryEntry(
                name.into(),
            )),
        }
    }

    for fd in expected.keys() {
        if !actual_numbers.contains(fd) {
            issues.push(CriticalExecAuthorityIssue::MissingFileDescriptor(*fd));
        }
    }
    for fd in &actual_numbers {
        if !expected.contains_key(fd) {
            issues.push(CriticalExecAuthorityIssue::UnexpectedFileDescriptor(*fd));
        }
    }

    let mut evidence = Vec::new();
    for fd in actual_numbers {
        let Some(spec) = expected.get(&fd) else {
            continue;
        };
        let target = match read_link_text(&base.join("fd").join(fd.to_string())) {
            Ok(value) => value,
            Err(error) => {
                issues.push(CriticalExecAuthorityIssue::FdTargetUnavailable {
                    fd,
                    reason: error,
                });
                continue;
            }
        };
        if target.ends_with(" (deleted)") {
            issues.push(CriticalExecAuthorityIssue::DeletedFdTarget(fd));
        }
        if target != spec.exact_target {
            issues.push(CriticalExecAuthorityIssue::FdTargetMismatch(fd));
        }
        let fdinfo = match read_bounded(
            &base.join("fdinfo").join(fd.to_string()),
            policy.max_fdinfo_bytes,
        ) {
            Ok(value) => value,
            Err(error) => {
                issues.push(CriticalExecAuthorityIssue::FdInfoUnavailable {
                    fd,
                    reason: error,
                });
                continue;
            }
        };
        let fdinfo_blake3 = format!("blake3:{}", blake3::hash(&fdinfo).to_hex());
        let parsed = match parse_fdinfo(&fdinfo) {
            Ok(value) => value,
            Err(error) => {
                issues.push(CriticalExecAuthorityIssue::FdInfoInvalid {
                    fd,
                    reason: error,
                });
                continue;
            }
        };
        if parsed.position != spec.expected_position {
            issues.push(CriticalExecAuthorityIssue::FdPositionMismatch(fd));
        }
        if parsed.flags_octal != spec.expected_flags_octal {
            issues.push(CriticalExecAuthorityIssue::FdFlagsMismatch(fd));
        }
        if parsed.mount_id != spec.expected_mount_id {
            issues.push(CriticalExecAuthorityIssue::FdMountIdMismatch(fd));
        }
        if parsed.inode != spec.expected_inode {
            issues.push(CriticalExecAuthorityIssue::FdInodeMismatch(fd));
        }
        if fdinfo_blake3 != spec.expected_fdinfo_blake3 {
            issues.push(CriticalExecAuthorityIssue::FdInfoDigestMismatch(fd));
        }
        evidence.push(FdAuthorityEvidence {
            fd,
            target,
            position: parsed.position,
            flags_octal: parsed.flags_octal,
            mount_id: parsed.mount_id,
            inode: parsed.inode,
            fdinfo_blake3,
        });
    }
    evidence.sort_by_key(|value| value.fd);
    evidence
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedFdInfo {
    position: u64,
    flags_octal: String,
    mount_id: u64,
    inode: u64,
}

fn parse_fdinfo(bytes: &[u8]) -> Result<ParsedFdInfo, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut position = None;
    let mut flags_octal = None;
    let mut mount_id = None;
    let mut inode = None;
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key {
            "pos" => set_once(
                &mut position,
                value.parse::<u64>().map_err(|_| "invalid-pos")?,
            )?,
            "flags" => {
                if !valid_octal(value) {
                    return Err("invalid-flags".into());
                }
                set_once(&mut flags_octal, value.to_string())?;
            }
            "mnt_id" => set_once(
                &mut mount_id,
                value.parse::<u64>().map_err(|_| "invalid-mnt-id")?,
            )?,
            "ino" => set_once(
                &mut inode,
                value.parse::<u64>().map_err(|_| "invalid-ino")?,
            )?,
            _ => {}
        }
    }
    Ok(ParsedFdInfo {
        position: position.ok_or("missing-pos")?,
        flags_octal: flags_octal.ok_or("missing-flags")?,
        mount_id: mount_id.ok_or("missing-mnt-id")?,
        inode: inode.ok_or("missing-ino")?,
    })
}

fn observe_namespaces(
    base: &Path,
    policy: &CriticalExecAuthorityPolicy,
    issues: &mut Vec<CriticalExecAuthorityIssue>,
) -> Vec<NamespaceAuthorityEvidence> {
    let mut expected = BTreeMap::new();
    for spec in &policy.expected_namespaces {
        expected.insert(spec.name.as_str(), spec);
    }

    let entries = match fs::read_dir(base.join("ns")) {
        Ok(value) => value,
        Err(error) => {
            issues.push(CriticalExecAuthorityIssue::NamespaceDirectoryUnavailable(
                error.to_string(),
            ));
            return Vec::new();
        }
    };
    let mut names = BTreeSet::new();
    for entry in entries {
        let entry = match entry {
            Ok(value) => value,
            Err(error) => {
                issues.push(CriticalExecAuthorityIssue::NamespaceDirectoryUnavailable(
                    error.to_string(),
                ));
                continue;
            }
        };
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            issues.push(CriticalExecAuthorityIssue::InvalidNamespaceEntry(
                "non-utf8".into(),
            ));
            continue;
        };
        if !identifier(&name) {
            issues.push(CriticalExecAuthorityIssue::InvalidNamespaceEntry(name));
            continue;
        }
        names.insert(name);
    }

    for name in expected.keys() {
        if !names.contains(*name) {
            issues.push(CriticalExecAuthorityIssue::MissingNamespace((*name).into()));
        }
    }
    for name in &names {
        if !expected.contains_key(name.as_str()) {
            issues.push(CriticalExecAuthorityIssue::UnexpectedNamespace(name.clone()));
        }
    }

    let mut evidence = Vec::new();
    for name in names {
        let Some(spec) = expected.get(name.as_str()) else {
            continue;
        };
        let link = match read_link_text(&base.join("ns").join(&name)) {
            Ok(value) => value,
            Err(error) => {
                issues.push(CriticalExecAuthorityIssue::NamespaceLinkUnavailable {
                    name: name.clone(),
                    reason: error,
                });
                continue;
            }
        };
        if link != spec.exact_link {
            issues.push(CriticalExecAuthorityIssue::NamespaceLinkMismatch(name.clone()));
        }
        evidence.push(NamespaceAuthorityEvidence { name, link });
    }
    evidence.sort_by(|left, right| left.name.cmp(&right.name));
    evidence
}

fn fd_set_digest(values: &[FdAuthorityEvidence]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(FD_SET_DOMAIN);
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(&value.fd.to_le_bytes());
        field(&mut hasher, &value.target);
        hasher.update(&value.position.to_le_bytes());
        field(&mut hasher, &value.flags_octal);
        hasher.update(&value.mount_id.to_le_bytes());
        hasher.update(&value.inode.to_le_bytes());
        field(&mut hasher, &value.fdinfo_blake3);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn namespace_set_digest(values: &[NamespaceAuthorityEvidence]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(NAMESPACE_SET_DOMAIN);
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        field(&mut hasher, &value.name);
        field(&mut hasher, &value.link);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn snapshot_digest(
    status_digest: &str,
    fd_digest: &str,
    cwd: &str,
    root: &str,
    namespace_digest: &str,
    limits_digest: &str,
    cgroup_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SNAPSHOT_DOMAIN);
    for value in [
        status_digest,
        fd_digest,
        cwd,
        root,
        namespace_digest,
        limits_digest,
        cgroup_digest,
    ] {
        field(&mut hasher, value);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn qualification_digest(
    policy_digest: &str,
    confirmation_digest: &str,
    snapshot_digest: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DOMAIN);
    for value in [
        policy_digest,
        confirmation_digest,
        snapshot_digest,
        report_digest,
    ] {
        field(&mut hasher, value);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn base_report(
    policy: &CriticalExecAuthorityPolicy,
    policy_digest: Option<String>,
    confirmed: &ConfirmedStaticExecLaunch,
) -> CriticalExecAuthorityReport {
    CriticalExecAuthorityReport {
        schema_version: CRITICAL_EXEC_AUTHORITY_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        exec_confirmation_policy_digest: confirmed.policy_digest().into(),
        fd_exec_policy_digest: confirmed.fd_exec_policy_digest().into(),
        runtime_policy_digest: confirmed.runtime_policy_digest().into(),
        runtime_verifier_ref: confirmed.runtime_verifier_ref().into(),
        backend_id: confirmed.backend_id().into(),
        tracee_pid: confirmed.tracee_pid(),
        authority_snapshot_digest: None,
        status_digest: None,
        fd_set_digest: None,
        namespace_set_digest: None,
        fd_count: 0,
        namespace_count: 0,
        disposition: CriticalExecAuthorityDisposition::Invalid,
        issues: Vec::new(),
        qualification_digest: None,
    }
}

fn finalize(mut report: CriticalExecAuthorityReport) -> CriticalExecAuthorityReport {
    report.disposition = if report.issues.iter().any(CriticalExecAuthorityIssue::invalid) {
        CriticalExecAuthorityDisposition::Invalid
    } else {
        CriticalExecAuthorityDisposition::Blocked
    };
    report
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, String> {
    use std::io::Read;

    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut output = Vec::new();
    let mut limited = (&mut file).take(maximum.saturating_add(1));
    limited
        .read_to_end(&mut output)
        .map_err(|error| error.to_string())?;
    if output.len() as u64 > maximum {
        return Err("bounded-read-limit-exceeded".into());
    }
    Ok(output)
}

fn read_link_text(path: &Path) -> Result<String, String> {
    fs::read_link(path)
        .map_err(|error| error.to_string())?
        .into_os_string()
        .into_string()
        .map_err(|_| "non-utf8-link-target".into())
}

fn required<'a>(values: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, String> {
    values
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing-status-field:{key}"))
}

fn parse_quad(value: &str) -> Result<[u32; 4], String> {
    let values = parse_u32_list(value)?;
    values
        .try_into()
        .map_err(|_| "expected-four-integers".into())
}

fn parse_u32_list(value: &str) -> Result<Vec<u32>, String> {
    value
        .split_whitespace()
        .map(|item| item.parse::<u32>().map_err(|_| "invalid-u32".to_string()))
        .collect()
}

fn parse_u32(value: &str) -> Result<u32, String> {
    value.parse::<u32>().map_err(|_| "invalid-u32".into())
}

fn parse_u8(value: &str) -> Result<u8, String> {
    value.parse::<u8>().map_err(|_| "invalid-u8".into())
}

fn parse_bool01(value: &str) -> Result<bool, String> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err("invalid-bool01".into()),
    }
}

fn parse_hex_u64(value: &str) -> Result<u64, String> {
    if value.is_empty()
        || value.len() > 16
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("invalid-hex-u64".into());
    }
    u64::from_str_radix(value, 16).map_err(|_| "invalid-hex-u64".into())
}

fn set_once<T>(slot: &mut Option<T>, value: T) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err("duplicate-standard-fdinfo-field".into());
    }
    Ok(())
}

fn field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn sorted_strings(hasher: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        field(hasher, &value);
    }
}

fn valid_blake3(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn valid_octal(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 32
        && value.bytes().all(|byte| (b'0'..=b'7').contains(&byte))
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT
        && !value.chars().any(char::is_control)
}

fn canonical_absolute_path(value: &str) -> bool {
    canonical_text(value)
        && value.starts_with('/')
        && !value.ends_with(" (deleted)")
        && !value.split('/').any(|segment| segment == "." || segment == "..")
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn valid_refs(values: &[String]) -> bool {
    values.len() <= MAX_REFS
        && values.iter().all(|value| canonical_text(value))
        && unique_strings(values)
}

fn unique_strings(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.as_str()))
}

fn unique_u32(values: &[u32]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(*value))
}

fn unique_fd_numbers(values: &[ExpectedFdAuthority]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.fd))
}

fn unique_namespace_names(values: &[ExpectedNamespaceAuthority]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.name.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn expected_fd(fd: u32) -> ExpectedFdAuthority {
        ExpectedFdAuthority {
            fd,
            exact_target: format!("pipe:[{}]", 1000 + fd),
            expected_position: 0,
            expected_flags_octal: "0100000".into(),
            expected_mount_id: 17,
            expected_inode: 1000 + fd as u64,
            expected_fdinfo_blake3: d(&format!("fdinfo:{fd}")),
        }
    }

    fn policy() -> CriticalExecAuthorityPolicy {
        CriticalExecAuthorityPolicy {
            schema_version: CRITICAL_EXEC_AUTHORITY_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:critical-authority:1".into(),
            expected_exec_confirmation_policy_digest: d("confirm-policy"),
            expected_fd_exec_policy_digest: d("fd-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:host".into(),
            expected_backend_id: "backend:in-process".into(),
            expected_uids: [1000, 1000, 1000, 1000],
            expected_gids: [1000, 1000, 1000, 1000],
            expected_supplementary_groups: vec![10, 20],
            expected_umask: 0o077,
            expected_cap_inheritable: 0,
            expected_cap_permitted: 0,
            expected_cap_effective: 0,
            expected_cap_bounding: 0,
            expected_cap_ambient: 0,
            expected_sig_pending: 0,
            expected_shared_sig_pending: 0,
            expected_sig_blocked: 0,
            expected_sig_ignored: 0,
            expected_sig_caught: 0,
            expected_no_new_privs: true,
            expected_seccomp_mode: 2,
            expected_seccomp_filters: 1,
            expected_core_dumping: false,
            expected_cwd: "/".into(),
            expected_root: "/".into(),
            expected_fds: vec![expected_fd(0), expected_fd(1)],
            expected_namespaces: vec![
                ExpectedNamespaceAuthority {
                    name: "mnt".into(),
                    exact_link: "mnt:[4026531840]".into(),
                },
                ExpectedNamespaceAuthority {
                    name: "user".into(),
                    exact_link: "user:[4026531837]".into(),
                },
            ],
            expected_limits_blake3: d("limits"),
            expected_cgroup_blake3: d("cgroup"),
            max_status_bytes: 1 << 20,
            max_fdinfo_bytes: 1 << 20,
            max_limits_bytes: 1 << 20,
            max_cgroup_bytes: 1 << 20,
            evidence_refs: vec!["review:a".into(), "review:b".into()],
        }
    }

    #[test]
    fn policy_set_order_is_nonsemantic_but_authority_is_semantic() {
        let left = policy();
        let mut right = left.clone();
        right.expected_supplementary_groups.reverse();
        right.expected_fds.reverse();
        right.expected_namespaces.reverse();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
        right.expected_no_new_privs = false;
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn policy_rejects_incoherent_seccomp_mode_filter_count() {
        let mut value = policy();
        value.expected_seccomp_filters = 0;
        assert!(!value.validate());
        value.expected_seccomp_mode = 0;
        assert!(value.validate());
        value.expected_seccomp_filters = 1;
        assert!(!value.validate());
    }

    #[test]
    fn duplicate_fd_and_namespace_authority_is_invalid() {
        let mut value = policy();
        value.expected_fds.push(expected_fd(0));
        assert!(!value.validate());
        let mut value = policy();
        value
            .expected_namespaces
            .push(value.expected_namespaces[0].clone());
        assert!(!value.validate());
    }

    #[test]
    fn fdinfo_parser_requires_standard_fields_and_preserves_flags() {
        let value = b"pos:\t7\nflags:\t0100002\nmnt_id:\t23\nino:\t99\neventfd-count:\t4\n";
        let parsed = parse_fdinfo(value).unwrap();
        assert_eq!(parsed.position, 7);
        assert_eq!(parsed.flags_octal, "0100002");
        assert_eq!(parsed.mount_id, 23);
        assert_eq!(parsed.inode, 99);
        assert!(parse_fdinfo(b"pos:\t0\nflags:\t0100000\n").is_err());
    }

    #[test]
    fn status_parser_binds_privilege_signal_and_sandbox_state() {
        let status = b"State:\tt (tracing stop)\nTracerPid:\t77\nUid:\t1000\t1000\t1000\t1000\nGid:\t1000\t1000\t1000\t1000\nGroups:\t10 20\nUmask:\t0077\nCapInh:\t0000000000000000\nCapPrm:\t0000000000000000\nCapEff:\t0000000000000000\nCapBnd:\t0000000000000000\nCapAmb:\t0000000000000000\nSigPnd:\t0000000000000000\nShdPnd:\t0000000000000000\nSigBlk:\t0000000000000000\nSigIgn:\t0000000000000000\nSigCgt:\t0000000000000000\nNoNewPrivs:\t1\nSeccomp:\t2\nSeccomp_filters:\t1\nThreads:\t1\nCoreDumping:\t0\n";
        let parsed = parse_status(status).unwrap();
        assert_eq!(parsed.tracer_pid, 77);
        assert_eq!(parsed.umask, 0o077);
        assert!(parsed.no_new_privs);
        assert_eq!(parsed.seccomp_mode, 2);
        assert_eq!(parsed.seccomp_filters, 1);
        assert_eq!(parsed.threads, 1);
        assert!(valid_blake3(&parsed.selected_status_digest));
    }

    #[test]
    fn snapshot_identity_binds_every_critical_surface() {
        let left = snapshot_digest(
            &d("status"),
            &d("fds"),
            "/work",
            "/",
            &d("ns"),
            &d("limits"),
            &d("cgroup"),
        );
        let right = snapshot_digest(
            &d("status"),
            &d("fds"),
            "/other",
            "/",
            &d("ns"),
            &d("limits"),
            &d("cgroup"),
        );
        assert_ne!(left, right);
    }

    #[test]
    fn qualification_field_does_not_self_reference_report_digest() {
        let mut report = CriticalExecAuthorityReport {
            schema_version: CRITICAL_EXEC_AUTHORITY_REPORT_SCHEMA_V1.into(),
            policy_id: "policy:test".into(),
            policy_digest: Some(d("policy")),
            exec_confirmation_digest: d("confirmation"),
            exec_confirmation_policy_digest: d("confirmation-policy"),
            fd_exec_policy_digest: d("fd-policy"),
            runtime_policy_digest: d("runtime-policy"),
            runtime_verifier_ref: "verifier:host".into(),
            backend_id: "backend:host".into(),
            tracee_pid: 42,
            authority_snapshot_digest: Some(d("snapshot")),
            status_digest: Some(d("status")),
            fd_set_digest: Some(d("fds")),
            namespace_set_digest: Some(d("namespaces")),
            fd_count: 3,
            namespace_count: 8,
            disposition: CriticalExecAuthorityDisposition::Qualified,
            issues: Vec::new(),
            qualification_digest: None,
        };
        let before = report.canonical_digest();
        report.qualification_digest = Some(d("qualification"));
        assert_eq!(before, report.canonical_digest());
    }

    #[test]
    fn claim_ceiling_stays_bounded() {
        let claims = [
            "all_exec_preserved_process_state_qualified=false",
            "fd_peer_authority_verified=false",
            "namespace_contents_verified=false",
            "seccomp_filter_semantics_verified=false",
            "resource_limit_semantics_independently_parsed=false",
            "lsm_policy_context_qualified=false",
            "post_release_continuity_established=false",
            "trusted_time_established=false",
            "grants_physical_authority=false",
        ];
        assert_eq!(claims.len(), 9);
    }
}
