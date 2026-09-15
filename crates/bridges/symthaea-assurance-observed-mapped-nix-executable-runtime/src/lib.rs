// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded observation of the verifier's actually mapped executable Nix objects.
//!
//! This bridge observes `/proc/self/maps`, binds every non-kernel executable
//! mapping to its current backing inode and the already-qualified Nix closure,
//! and compares executable bytes read through `/proc/self/mem` with the
//! corresponding backing-file bytes. The observation is deliberately bounded
//! to one stable snapshot; it does not claim loader atomicity or continuity
//! since `execve`.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Component, Path};
use symthaea_assurance_nix_bound_in_process_runtime::NixBoundInProcessRuntimePolicy;
use symthaea_assurance_nix_runtime_closure::{
    NixClosureEntry, NixRuntimeClosureQualification,
};

#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::{Read, Take};
#[cfg(target_os = "linux")]
use std::os::unix::fs::{FileExt, MetadataExt};

pub const MAPPED_EXECUTABLE_RUNTIME_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.mapped-nix-executable-runtime-policy.v1";
pub const MAPPED_EXECUTABLE_RUNTIME_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.mapped-nix-executable-runtime-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.mapped-nix-executable-runtime-policy.digest.v1\0";
const MAPPED_OBJECT_SET_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.mapped-nix-executable-object-set.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.mapped-nix-executable-runtime-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.mapped-nix-executable-runtime-qualification.digest.v1\0";
const NIX_CLOSURE_CONTENT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.nix-runtime-closure-content.digest.v1\0";

const MAX_TEXT_BYTES: usize = 4096;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_ALLOWED_KERNEL_EXEC_MAPPINGS: usize = 16;
const MAX_MAPS_BYTES_HARD: u64 = 64 * 1024 * 1024;
const MAX_MAP_ENTRIES_HARD: u32 = 262_144;
const MAX_EXEC_MAPPINGS_HARD: u32 = 65_536;
const MAX_MAPPED_FILES_HARD: u32 = 16_384;
const MAX_SINGLE_FILE_BYTES_HARD: u64 = 8 * 1024 * 1024 * 1024;
const MAX_TOTAL_BACKING_FILE_BYTES_HARD: u64 = 64 * 1024 * 1024 * 1024;
const MAX_TOTAL_EXECUTABLE_BYTES_HARD: u64 = 16 * 1024 * 1024 * 1024;
const NIX_BASE32: &[u8] = b"0123456789abcdfghijklmnpqrsvwxyz";
const IO_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedExecutableRuntimePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_nix_binding_policy_digest: String,
    pub expected_closure_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub allowed_kernel_executable_mappings: Vec<String>,
    pub max_maps_bytes: u64,
    pub max_map_entries: u32,
    pub max_executable_mappings: u32,
    pub max_mapped_files: u32,
    pub max_single_file_bytes: u64,
    pub max_total_backing_file_bytes: u64,
    pub max_total_executable_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl MappedExecutableRuntimePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == MAPPED_EXECUTABLE_RUNTIME_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_nix_binding_policy_digest)
            && valid_blake3_digest(&self.expected_closure_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
            && !self.allowed_kernel_executable_mappings.is_empty()
            && self.allowed_kernel_executable_mappings.len()
                <= MAX_ALLOWED_KERNEL_EXEC_MAPPINGS
            && self
                .allowed_kernel_executable_mappings
                .iter()
                .all(|value| valid_kernel_pseudo_mapping(value))
            && unique(&self.allowed_kernel_executable_mappings)
            && (1..=MAX_MAPS_BYTES_HARD).contains(&self.max_maps_bytes)
            && (1..=MAX_MAP_ENTRIES_HARD).contains(&self.max_map_entries)
            && (1..=MAX_EXEC_MAPPINGS_HARD).contains(&self.max_executable_mappings)
            && (1..=MAX_MAPPED_FILES_HARD).contains(&self.max_mapped_files)
            && (1..=MAX_SINGLE_FILE_BYTES_HARD).contains(&self.max_single_file_bytes)
            && (1..=MAX_TOTAL_BACKING_FILE_BYTES_HARD)
                .contains(&self.max_total_backing_file_bytes)
            && (1..=MAX_TOTAL_EXECUTABLE_BYTES_HARD)
                .contains(&self.max_total_executable_bytes)
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
            self.expected_closure_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        let mut pseudo = self.allowed_kernel_executable_mappings.clone();
        pseudo.sort();
        hasher.update(&(pseudo.len() as u64).to_le_bytes());
        for value in pseudo {
            push_field(&mut hasher, &value);
        }
        hasher.update(&self.max_maps_bytes.to_le_bytes());
        hasher.update(&self.max_map_entries.to_le_bytes());
        hasher.update(&self.max_executable_mappings.to_le_bytes());
        hasher.update(&self.max_mapped_files.to_le_bytes());
        hasher.update(&self.max_single_file_bytes.to_le_bytes());
        hasher.update(&self.max_total_backing_file_bytes.to_le_bytes());
        hasher.update(&self.max_total_executable_bytes.to_le_bytes());
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MappedExecutableRuntimeDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MappedExecutableRuntimeIssue {
    InvalidPolicy,
    InvalidObservationTime,
    NixBindingPolicyMismatch,
    ClosurePolicyMismatch,
    ClosureQualificationMismatch,
    ClosureDigestMismatch,
    RuntimeVerifierRefMismatch,
    BackendMismatch,
    ClosureManifestInvalid(String),
    ClosureManifestDigestMismatch,
    ProcMapsUnavailable(String),
    ProcMapsTooLarge,
    MalformedProcMapsLine(u64),
    TooManyMapEntries { observed: u64, maximum: u32 },
    TooManyExecutableMappings { observed: u64, maximum: u32 },
    TooManyMappedFiles { observed: u64, maximum: u32 },
    WritableExecutableMapping(String),
    AnonymousExecutableMapping,
    UnapprovedKernelExecutableMapping(String),
    DeletedExecutableMapping(String),
    NonNixExecutableMapping(String),
    MappedStoreObjectOutsideClosure(String),
    MappedPathIdentityConflict(String),
    NonCanonicalMappedPath(String),
    BackingFileUnavailable { path: String, reason: String },
    BackingFileMetadataMismatch(String),
    BackingFileNotRootOwned(String),
    BackingFileWritable(String),
    BackingFileTooLarge { path: String, observed: u64, maximum: u64 },
    TotalBackingFileBytesExceeded { observed: u64, maximum: u64 },
    ExecutableBytesExceeded { observed: u64, maximum: u64 },
    MappingOutsideBackingFile(String),
    ProcMemUnavailable(String),
    ProcMemReadUnavailable { path: String, reason: String },
    BackingFileReadUnavailable { path: String, reason: String },
    ExecutableMappingBytesMismatch(String),
    ExecutableMappingBytesUnstable(String),
    HostExecutableNotMapped,
    HostExecutableDigestMismatch,
    ProcMapsChangedDuringObservation,
}

impl MappedExecutableRuntimeIssue {
    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::InvalidObservationTime
                | Self::ClosureManifestInvalid(_)
                | Self::ClosureManifestDigestMismatch
                | Self::MalformedProcMapsLine(_)
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidObservationTime => "invalid-observation-time".into(),
            Self::NixBindingPolicyMismatch => "nix-binding-policy-mismatch".into(),
            Self::ClosurePolicyMismatch => "closure-policy-mismatch".into(),
            Self::ClosureQualificationMismatch => "closure-qualification-mismatch".into(),
            Self::ClosureDigestMismatch => "closure-digest-mismatch".into(),
            Self::RuntimeVerifierRefMismatch => "runtime-verifier-ref-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::ClosureManifestInvalid(reason) => {
                format!("closure-manifest-invalid:{reason}")
            }
            Self::ClosureManifestDigestMismatch => "closure-manifest-digest-mismatch".into(),
            Self::ProcMapsUnavailable(reason) => format!("proc-maps-unavailable:{reason}"),
            Self::ProcMapsTooLarge => "proc-maps-too-large".into(),
            Self::MalformedProcMapsLine(line) => format!("malformed-proc-maps-line:{line}"),
            Self::TooManyMapEntries { observed, maximum } => {
                format!("too-many-map-entries:{observed}:{maximum}")
            }
            Self::TooManyExecutableMappings { observed, maximum } => {
                format!("too-many-executable-mappings:{observed}:{maximum}")
            }
            Self::TooManyMappedFiles { observed, maximum } => {
                format!("too-many-mapped-files:{observed}:{maximum}")
            }
            Self::WritableExecutableMapping(path) => {
                format!("writable-executable-mapping:{path}")
            }
            Self::AnonymousExecutableMapping => "anonymous-executable-mapping".into(),
            Self::UnapprovedKernelExecutableMapping(path) => {
                format!("unapproved-kernel-executable-mapping:{path}")
            }
            Self::DeletedExecutableMapping(path) => {
                format!("deleted-executable-mapping:{path}")
            }
            Self::NonNixExecutableMapping(path) => {
                format!("non-nix-executable-mapping:{path}")
            }
            Self::MappedStoreObjectOutsideClosure(path) => {
                format!("mapped-store-object-outside-closure:{path}")
            }
            Self::MappedPathIdentityConflict(path) => {
                format!("mapped-path-identity-conflict:{path}")
            }
            Self::NonCanonicalMappedPath(path) => format!("noncanonical-mapped-path:{path}"),
            Self::BackingFileUnavailable { path, reason } => {
                format!("backing-file-unavailable:{path}:{reason}")
            }
            Self::BackingFileMetadataMismatch(path) => {
                format!("backing-file-metadata-mismatch:{path}")
            }
            Self::BackingFileNotRootOwned(path) => {
                format!("backing-file-not-root-owned:{path}")
            }
            Self::BackingFileWritable(path) => format!("backing-file-writable:{path}"),
            Self::BackingFileTooLarge {
                path,
                observed,
                maximum,
            } => format!("backing-file-too-large:{path}:{observed}:{maximum}"),
            Self::TotalBackingFileBytesExceeded { observed, maximum } => {
                format!("total-backing-file-bytes-exceeded:{observed}:{maximum}")
            }
            Self::ExecutableBytesExceeded { observed, maximum } => {
                format!("executable-bytes-exceeded:{observed}:{maximum}")
            }
            Self::MappingOutsideBackingFile(path) => {
                format!("mapping-outside-backing-file:{path}")
            }
            Self::ProcMemUnavailable(reason) => format!("proc-mem-unavailable:{reason}"),
            Self::ProcMemReadUnavailable { path, reason } => {
                format!("proc-mem-read-unavailable:{path}:{reason}")
            }
            Self::BackingFileReadUnavailable { path, reason } => {
                format!("backing-file-read-unavailable:{path}:{reason}")
            }
            Self::ExecutableMappingBytesMismatch(path) => {
                format!("executable-mapping-bytes-mismatch:{path}")
            }
            Self::ExecutableMappingBytesUnstable(path) => {
                format!("executable-mapping-bytes-unstable:{path}")
            }
            Self::HostExecutableNotMapped => "host-executable-not-mapped".into(),
            Self::HostExecutableDigestMismatch => "host-executable-digest-mismatch".into(),
            Self::ProcMapsChangedDuringObservation => {
                "proc-maps-changed-during-observation".into()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedExecutableSegmentEvidence {
    pub file_offset: u64,
    pub length: u64,
    pub permissions: String,
    pub mapped_bytes_blake3: String,
    pub backing_bytes_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedExecutableObjectEvidence {
    pub canonical_path: String,
    pub nix_store_root: String,
    pub device_major: u64,
    pub device_minor: u64,
    pub inode: u64,
    pub file_size: u64,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub mtime_seconds: i64,
    pub mtime_nanoseconds: i64,
    pub full_file_blake3: String,
    pub executable_segments: Vec<MappedExecutableSegmentEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedExecutableRuntimeReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub nix_binding_policy_digest: String,
    pub nix_binding_qualification_digest: String,
    pub closure_policy_digest: String,
    pub closure_qualification_digest: String,
    pub closure_digest: String,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub host_identity_digest: String,
    pub host_executable_path: String,
    pub host_executable_digest: String,
    pub proc_maps_digest: Option<String>,
    pub mapped_object_set_digest: Option<String>,
    pub map_entry_count: u64,
    pub executable_mapping_count: u64,
    pub kernel_executable_mapping_count: u64,
    pub mapped_file_count: u64,
    pub total_backing_file_bytes: u64,
    pub total_executable_bytes: u64,
    pub observed_at_ms: u64,
    pub disposition: MappedExecutableRuntimeDisposition,
    pub issues: Vec<MappedExecutableRuntimeIssue>,
}

impl MappedExecutableRuntimeReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.nix_binding_policy_digest.as_str(),
            self.nix_binding_qualification_digest.as_str(),
            self.closure_policy_digest.as_str(),
            self.closure_qualification_digest.as_str(),
            self.closure_digest.as_str(),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.host_identity_digest.as_str(),
            self.host_executable_path.as_str(),
            self.host_executable_digest.as_str(),
            self.proc_maps_digest.as_deref().unwrap_or("-"),
            self.mapped_object_set_digest.as_deref().unwrap_or("-"),
        ] {
            push_field(&mut hasher, field);
        }
        for value in [
            self.map_entry_count,
            self.executable_mapping_count,
            self.kernel_executable_mapping_count,
            self.mapped_file_count,
            self.total_backing_file_bytes,
            self.total_executable_bytes,
            self.observed_at_ms,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        push_field(
            &mut hasher,
            match self.disposition {
                MappedExecutableRuntimeDisposition::Invalid => "invalid",
                MappedExecutableRuntimeDisposition::Blocked => "blocked",
                MappedExecutableRuntimeDisposition::Qualified => "qualified",
            },
        );
        hasher.update(&(self.issues.len() as u64).to_le_bytes());
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
pub struct ObservedMappedNixExecutableRuntime {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    nix_binding_qualification_digest: String,
    closure_qualification_digest: String,
    closure_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    host_identity_digest: String,
    host_executable_path: String,
    host_executable_digest: String,
    proc_maps_digest: String,
    mapped_object_set_digest: String,
    mapped_file_count: u64,
    executable_mapping_count: u64,
    total_executable_bytes: u64,
    observed_at_ms: u64,
}

impl ObservedMappedNixExecutableRuntime {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn nix_binding_qualification_digest(&self) -> &str { &self.nix_binding_qualification_digest }
    pub fn closure_qualification_digest(&self) -> &str { &self.closure_qualification_digest }
    pub fn closure_digest(&self) -> &str { &self.closure_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn host_identity_digest(&self) -> &str { &self.host_identity_digest }
    pub fn host_executable_path(&self) -> &str { &self.host_executable_path }
    pub fn host_executable_digest(&self) -> &str { &self.host_executable_digest }
    pub fn proc_maps_digest(&self) -> &str { &self.proc_maps_digest }
    pub fn mapped_object_set_digest(&self) -> &str { &self.mapped_object_set_digest }
    pub const fn mapped_file_count(&self) -> u64 { self.mapped_file_count }
    pub const fn executable_mapping_count(&self) -> u64 { self.executable_mapping_count }
    pub const fn total_executable_bytes(&self) -> u64 { self.total_executable_bytes }
    pub const fn observed_at_ms(&self) -> u64 { self.observed_at_ms }
    pub const fn proc_maps_snapshot_stable_during_observation(&self) -> bool { true }
    pub const fn no_writable_executable_mappings_observed(&self) -> bool { true }
    pub const fn all_non_kernel_executable_mappings_nix_backed(&self) -> bool { true }
    pub const fn all_mapped_executable_store_objects_in_verified_closure(&self) -> bool { true }
    pub const fn executable_mapping_bytes_match_backing_files_at_observation(&self) -> bool { true }
    pub const fn executable_mapping_bytes_stable_across_two_reads(&self) -> bool { true }
    pub const fn host_executable_mapping_matches_bound_identity(&self) -> bool { true }
    pub const fn mapped_memory_identity_established(&self) -> bool { false }
    pub const fn mapping_continuity_since_exec_established(&self) -> bool { false }
    pub const fn closure_reverified_at_observation(&self) -> bool { false }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappedExecutableRuntimeQualification {
    pub report: MappedExecutableRuntimeReport,
    pub objects: Vec<MappedExecutableObjectEvidence>,
    observed: ObservedMappedNixExecutableRuntime,
}

impl MappedExecutableRuntimeQualification {
    pub fn observed(&self) -> &ObservedMappedNixExecutableRuntime { &self.observed }
    pub fn into_observed(self) -> ObservedMappedNixExecutableRuntime { self.observed }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedMap {
    start: u64,
    end: u64,
    permissions: String,
    file_offset: u64,
    device_major: u64,
    device_minor: u64,
    inode: u64,
    pathname: String,
}

impl ParsedMap {
    fn executable(&self) -> bool { self.permissions.as_bytes().get(2) == Some(&b'x') }
    fn writable(&self) -> bool { self.permissions.as_bytes().get(1) == Some(&b'w') }
    fn length(&self) -> u64 { self.end - self.start }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExecutableFilePlan {
    canonical_path: String,
    nix_store_root: String,
    device_major: u64,
    device_minor: u64,
    inode: u64,
    segments: Vec<ParsedMap>,
}

#[cfg(target_os = "linux")]
pub fn observe_mapped_nix_executable_runtime(
    policy: &MappedExecutableRuntimePolicy,
    bound: &NixBoundInProcessRuntimePolicy,
    closure: &NixRuntimeClosureQualification,
    observed_at_ms: u64,
) -> Result<MappedExecutableRuntimeQualification, MappedExecutableRuntimeReport> {
    let policy_digest = policy.canonical_digest();
    let verified_closure = closure.verified();
    let mut report = base_report(policy, policy_digest.clone(), bound, closure, observed_at_ms);

    if !policy.validate() {
        report.issues.push(MappedExecutableRuntimeIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if observed_at_ms == 0 {
        report.issues.push(MappedExecutableRuntimeIssue::InvalidObservationTime);
        return Err(finalize(report));
    }
    if bound.policy_digest() != policy.expected_nix_binding_policy_digest {
        report.issues.push(MappedExecutableRuntimeIssue::NixBindingPolicyMismatch);
    }
    if verified_closure.policy_digest() != policy.expected_closure_policy_digest {
        report.issues.push(MappedExecutableRuntimeIssue::ClosurePolicyMismatch);
    }
    if verified_closure.qualification_digest() != bound.closure_qualification_digest() {
        report.issues.push(MappedExecutableRuntimeIssue::ClosureQualificationMismatch);
    }
    if verified_closure.closure_digest() != bound.dependency_closure_digest() {
        report.issues.push(MappedExecutableRuntimeIssue::ClosureDigestMismatch);
    }
    if bound.runtime_verifier_ref() != policy.expected_runtime_verifier_ref {
        report.issues.push(MappedExecutableRuntimeIssue::RuntimeVerifierRefMismatch);
    }
    if bound.backend_id() != policy.expected_backend_id {
        report.issues.push(MappedExecutableRuntimeIssue::BackendMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let (manifest_digest, closure_roots) = match recompute_closure_manifest_digest(
        verified_closure.root_store_path(),
        &closure.entries,
    ) {
        Ok(value) => value,
        Err(reason) => {
            report.issues.push(MappedExecutableRuntimeIssue::ClosureManifestInvalid(reason));
            return Err(finalize(report));
        }
    };
    if manifest_digest != verified_closure.closure_digest() {
        report.issues.push(MappedExecutableRuntimeIssue::ClosureManifestDigestMismatch);
        return Err(finalize(report));
    }

    let maps_before = match read_bounded_file("/proc/self/maps", policy.max_maps_bytes) {
        Ok(value) => value,
        Err(BoundedReadError::TooLarge) => {
            report.issues.push(MappedExecutableRuntimeIssue::ProcMapsTooLarge);
            return Err(finalize(report));
        }
        Err(BoundedReadError::Io(reason)) => {
            report.issues.push(MappedExecutableRuntimeIssue::ProcMapsUnavailable(reason));
            return Err(finalize(report));
        }
    };
    let maps_digest = digest_bytes(&maps_before);
    report.proc_maps_digest = Some(maps_digest.clone());

    let parsed = match parse_proc_maps(&maps_before, policy.max_map_entries) {
        Ok(value) => value,
        Err(issue) => {
            report.issues.push(issue);
            return Err(finalize(report));
        }
    };
    report.map_entry_count = parsed.len() as u64;

    let (plans, executable_count, kernel_exec_count, total_exec_bytes) =
        match executable_file_plans(policy, &parsed, &closure_roots) {
            Ok(value) => value,
            Err(issue) => {
                report.issues.push(issue);
                return Err(finalize(report));
            }
        };
    report.executable_mapping_count = executable_count;
    report.kernel_executable_mapping_count = kernel_exec_count;
    report.mapped_file_count = plans.len() as u64;
    report.total_executable_bytes = total_exec_bytes;

    if plans.len() as u64 > policy.max_mapped_files as u64 {
        report.issues.push(MappedExecutableRuntimeIssue::TooManyMappedFiles {
            observed: plans.len() as u64,
            maximum: policy.max_mapped_files,
        });
        return Err(finalize(report));
    }

    let proc_mem = match File::open("/proc/self/mem") {
        Ok(value) => value,
        Err(error) => {
            report.issues.push(MappedExecutableRuntimeIssue::ProcMemUnavailable(error.to_string()));
            return Err(finalize(report));
        }
    };

    let mut objects = Vec::with_capacity(plans.len());
    let mut total_backing_file_bytes = 0u64;
    let mut host_seen = false;
    for plan in plans.values() {
        let object = match inspect_executable_file(policy, plan, &proc_mem) {
            Ok(value) => value,
            Err(issue) => {
                report.issues.push(issue);
                return Err(finalize(report));
            }
        };
        total_backing_file_bytes = match total_backing_file_bytes.checked_add(object.file_size) {
            Some(value) => value,
            None => {
                report.issues.push(MappedExecutableRuntimeIssue::TotalBackingFileBytesExceeded {
                    observed: u64::MAX,
                    maximum: policy.max_total_backing_file_bytes,
                });
                return Err(finalize(report));
            }
        };
        if total_backing_file_bytes > policy.max_total_backing_file_bytes {
            report.issues.push(MappedExecutableRuntimeIssue::TotalBackingFileBytesExceeded {
                observed: total_backing_file_bytes,
                maximum: policy.max_total_backing_file_bytes,
            });
            return Err(finalize(report));
        }

        if object.canonical_path == bound.executable_path() {
            host_seen = true;
            if object.full_file_blake3 != bound.executable_digest() {
                report.issues.push(MappedExecutableRuntimeIssue::HostExecutableDigestMismatch);
                return Err(finalize(report));
            }
        }
        objects.push(object);
    }
    if !host_seen {
        report.issues.push(MappedExecutableRuntimeIssue::HostExecutableNotMapped);
        return Err(finalize(report));
    }
    report.total_backing_file_bytes = total_backing_file_bytes;

    let maps_after = match read_bounded_file("/proc/self/maps", policy.max_maps_bytes) {
        Ok(value) => value,
        Err(BoundedReadError::TooLarge) => {
            report.issues.push(MappedExecutableRuntimeIssue::ProcMapsTooLarge);
            return Err(finalize(report));
        }
        Err(BoundedReadError::Io(reason)) => {
            report.issues.push(MappedExecutableRuntimeIssue::ProcMapsUnavailable(reason));
            return Err(finalize(report));
        }
    };
    if maps_after != maps_before {
        report.issues.push(MappedExecutableRuntimeIssue::ProcMapsChangedDuringObservation);
        return Err(finalize(report));
    }

    objects.sort_by(|left, right| left.canonical_path.cmp(&right.canonical_path));
    let object_set_digest = mapped_object_set_digest(&objects);
    report.mapped_object_set_digest = Some(object_set_digest.clone());
    report.disposition = MappedExecutableRuntimeDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        bound.qualification_digest(),
        verified_closure.qualification_digest(),
        &maps_digest,
        &object_set_digest,
        observed_at_ms,
        &report_digest,
    );
    let observed = ObservedMappedNixExecutableRuntime {
        qualification_digest,
        report_digest,
        policy_digest,
        nix_binding_qualification_digest: bound.qualification_digest().into(),
        closure_qualification_digest: verified_closure.qualification_digest().into(),
        closure_digest: verified_closure.closure_digest().into(),
        runtime_policy_digest: bound.runtime_policy_digest().into(),
        runtime_verifier_ref: bound.runtime_verifier_ref().into(),
        backend_id: bound.backend_id().into(),
        host_identity_digest: bound.host_identity_digest().into(),
        host_executable_path: bound.executable_path().into(),
        host_executable_digest: bound.executable_digest().into(),
        proc_maps_digest: maps_digest,
        mapped_object_set_digest: object_set_digest,
        mapped_file_count: objects.len() as u64,
        executable_mapping_count: executable_count,
        total_executable_bytes: total_exec_bytes,
        observed_at_ms,
    };

    Ok(MappedExecutableRuntimeQualification { report, objects, observed })
}

#[cfg(not(target_os = "linux"))]
pub fn observe_mapped_nix_executable_runtime(
    policy: &MappedExecutableRuntimePolicy,
    bound: &NixBoundInProcessRuntimePolicy,
    closure: &NixRuntimeClosureQualification,
    observed_at_ms: u64,
) -> Result<MappedExecutableRuntimeQualification, MappedExecutableRuntimeReport> {
    let mut report = base_report(policy, policy.canonical_digest(), bound, closure, observed_at_ms);
    report.issues.push(MappedExecutableRuntimeIssue::ProcMapsUnavailable(
        "Linux /proc/self/maps and /proc/self/mem are required".into(),
    ));
    Err(finalize(report))
}

fn base_report(
    policy: &MappedExecutableRuntimePolicy,
    policy_digest: Option<String>,
    bound: &NixBoundInProcessRuntimePolicy,
    closure: &NixRuntimeClosureQualification,
    observed_at_ms: u64,
) -> MappedExecutableRuntimeReport {
    MappedExecutableRuntimeReport {
        schema_version: MAPPED_EXECUTABLE_RUNTIME_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        nix_binding_policy_digest: bound.policy_digest().into(),
        nix_binding_qualification_digest: bound.qualification_digest().into(),
        closure_policy_digest: closure.verified().policy_digest().into(),
        closure_qualification_digest: closure.verified().qualification_digest().into(),
        closure_digest: closure.verified().closure_digest().into(),
        runtime_policy_digest: bound.runtime_policy_digest().into(),
        runtime_verifier_ref: bound.runtime_verifier_ref().into(),
        backend_id: bound.backend_id().into(),
        host_identity_digest: bound.host_identity_digest().into(),
        host_executable_path: bound.executable_path().into(),
        host_executable_digest: bound.executable_digest().into(),
        proc_maps_digest: None,
        mapped_object_set_digest: None,
        map_entry_count: 0,
        executable_mapping_count: 0,
        kernel_executable_mapping_count: 0,
        mapped_file_count: 0,
        total_backing_file_bytes: 0,
        total_executable_bytes: 0,
        observed_at_ms,
        disposition: MappedExecutableRuntimeDisposition::Invalid,
        issues: Vec::new(),
    }
}

fn finalize(mut report: MappedExecutableRuntimeReport) -> MappedExecutableRuntimeReport {
    report.disposition = if report.issues.iter().any(MappedExecutableRuntimeIssue::is_invalid) {
        MappedExecutableRuntimeDisposition::Invalid
    } else {
        MappedExecutableRuntimeDisposition::Blocked
    };
    report
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
enum BoundedReadError {
    TooLarge,
    Io(String),
}

#[cfg(target_os = "linux")]
fn read_bounded_file(path: &str, maximum: u64) -> Result<Vec<u8>, BoundedReadError> {
    let file = File::open(path).map_err(|error| BoundedReadError::Io(error.to_string()))?;
    let mut reader: Take<File> = file.take(maximum.saturating_add(1));
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).map_err(|error| BoundedReadError::Io(error.to_string()))?;
    if bytes.len() as u64 > maximum {
        return Err(BoundedReadError::TooLarge);
    }
    Ok(bytes)
}

fn parse_proc_maps(
    bytes: &[u8],
    max_entries: u32,
) -> Result<Vec<ParsedMap>, MappedExecutableRuntimeIssue> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| MappedExecutableRuntimeIssue::MalformedProcMapsLine(0))?;
    let mut maps = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        if maps.len() as u64 >= max_entries as u64 {
            return Err(MappedExecutableRuntimeIssue::TooManyMapEntries {
                observed: maps.len() as u64 + 1,
                maximum: max_entries,
            });
        }
        maps.push(parse_proc_maps_line(line).ok_or(
            MappedExecutableRuntimeIssue::MalformedProcMapsLine(index as u64 + 1),
        )?);
    }
    Ok(maps)
}

fn parse_proc_maps_line(line: &str) -> Option<ParsedMap> {
    let mut cursor = 0usize;
    let address = next_ascii_token(line, &mut cursor)?;
    let permissions = next_ascii_token(line, &mut cursor)?;
    let offset = next_ascii_token(line, &mut cursor)?;
    let device = next_ascii_token(line, &mut cursor)?;
    let inode = next_ascii_token(line, &mut cursor)?;
    while cursor < line.len() && line.as_bytes()[cursor].is_ascii_whitespace() {
        cursor += 1;
    }
    let pathname = line.get(cursor..).unwrap_or("").to_string();

    let (start_text, end_text) = address.split_once('-')?;
    let start = u64::from_str_radix(start_text, 16).ok()?;
    let end = u64::from_str_radix(end_text, 16).ok()?;
    if end <= start || !valid_permissions(permissions) {
        return None;
    }
    let file_offset = u64::from_str_radix(offset, 16).ok()?;
    let (major_text, minor_text) = device.split_once(':')?;
    let device_major = u64::from_str_radix(major_text, 16).ok()?;
    let device_minor = u64::from_str_radix(minor_text, 16).ok()?;
    let inode = inode.parse::<u64>().ok()?;

    Some(ParsedMap {
        start,
        end,
        permissions: permissions.into(),
        file_offset,
        device_major,
        device_minor,
        inode,
        pathname,
    })
}

fn next_ascii_token<'a>(line: &'a str, cursor: &mut usize) -> Option<&'a str> {
    let bytes = line.as_bytes();
    while *cursor < bytes.len() && bytes[*cursor].is_ascii_whitespace() {
        *cursor += 1;
    }
    let start = *cursor;
    while *cursor < bytes.len() && !bytes[*cursor].is_ascii_whitespace() {
        *cursor += 1;
    }
    (start < *cursor).then(|| &line[start..*cursor])
}

fn valid_permissions(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 4
        && matches!(bytes[0], b'r' | b'-')
        && matches!(bytes[1], b'w' | b'-')
        && matches!(bytes[2], b'x' | b'-')
        && matches!(bytes[3], b'p' | b's')
}

fn executable_file_plans(
    policy: &MappedExecutableRuntimePolicy,
    maps: &[ParsedMap],
    closure_roots: &BTreeSet<String>,
) -> Result<(BTreeMap<String, ExecutableFilePlan>, u64, u64, u64), MappedExecutableRuntimeIssue> {
    let allowed_kernel = policy
        .allowed_kernel_executable_mappings
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut plans = BTreeMap::<String, ExecutableFilePlan>::new();
    let mut executable_count = 0u64;
    let mut kernel_exec_count = 0u64;
    let mut total_exec_bytes = 0u64;

    for mapping in maps {
        if mapping.executable() && mapping.writable() {
            return Err(MappedExecutableRuntimeIssue::WritableExecutableMapping(
                if mapping.pathname.is_empty() { "<anonymous>".into() } else { mapping.pathname.clone() },
            ));
        }
        if !mapping.executable() {
            continue;
        }
        executable_count += 1;
        if executable_count > policy.max_executable_mappings as u64 {
            return Err(MappedExecutableRuntimeIssue::TooManyExecutableMappings {
                observed: executable_count,
                maximum: policy.max_executable_mappings,
            });
        }
        total_exec_bytes = total_exec_bytes.checked_add(mapping.length()).ok_or(
            MappedExecutableRuntimeIssue::ExecutableBytesExceeded {
                observed: u64::MAX,
                maximum: policy.max_total_executable_bytes,
            },
        )?;
        if total_exec_bytes > policy.max_total_executable_bytes {
            return Err(MappedExecutableRuntimeIssue::ExecutableBytesExceeded {
                observed: total_exec_bytes,
                maximum: policy.max_total_executable_bytes,
            });
        }

        if mapping.pathname.is_empty() {
            return Err(MappedExecutableRuntimeIssue::AnonymousExecutableMapping);
        }
        if mapping.pathname.starts_with('[') {
            if mapping.device_major != 0 || mapping.device_minor != 0 || mapping.inode != 0 {
                return Err(MappedExecutableRuntimeIssue::UnapprovedKernelExecutableMapping(
                    mapping.pathname.clone(),
                ));
            }
            if !allowed_kernel.contains(mapping.pathname.as_str()) {
                return Err(MappedExecutableRuntimeIssue::UnapprovedKernelExecutableMapping(
                    mapping.pathname.clone(),
                ));
            }
            kernel_exec_count += 1;
            continue;
        }
        if mapping.pathname.ends_with(" (deleted)") {
            return Err(MappedExecutableRuntimeIssue::DeletedExecutableMapping(
                mapping.pathname.clone(),
            ));
        }
        let store_root = direct_nix_store_root(&mapping.pathname).ok_or_else(|| {
            MappedExecutableRuntimeIssue::NonNixExecutableMapping(mapping.pathname.clone())
        })?;
        if !closure_roots.contains(&store_root) {
            return Err(MappedExecutableRuntimeIssue::MappedStoreObjectOutsideClosure(store_root));
        }

        match plans.get_mut(&mapping.pathname) {
            Some(plan) => {
                if plan.device_major != mapping.device_major
                    || plan.device_minor != mapping.device_minor
                    || plan.inode != mapping.inode
                {
                    return Err(MappedExecutableRuntimeIssue::MappedPathIdentityConflict(
                        mapping.pathname.clone(),
                    ));
                }
                plan.segments.push(mapping.clone());
            }
            None => {
                plans.insert(
                    mapping.pathname.clone(),
                    ExecutableFilePlan {
                        canonical_path: mapping.pathname.clone(),
                        nix_store_root: store_root,
                        device_major: mapping.device_major,
                        device_minor: mapping.device_minor,
                        inode: mapping.inode,
                        segments: vec![mapping.clone()],
                    },
                );
            }
        }
    }
    Ok((plans, executable_count, kernel_exec_count, total_exec_bytes))
}

#[cfg(target_os = "linux")]
fn inspect_executable_file(
    policy: &MappedExecutableRuntimePolicy,
    plan: &ExecutableFilePlan,
    proc_mem: &File,
) -> Result<MappedExecutableObjectEvidence, MappedExecutableRuntimeIssue> {
    let canonical = std::fs::canonicalize(&plan.canonical_path).map_err(|error| {
        MappedExecutableRuntimeIssue::BackingFileUnavailable {
            path: plan.canonical_path.clone(),
            reason: format!("canonicalize: {error}"),
        }
    })?;
    let canonical = canonical.to_str().ok_or_else(|| {
        MappedExecutableRuntimeIssue::NonCanonicalMappedPath(plan.canonical_path.clone())
    })?;
    if canonical != plan.canonical_path {
        return Err(MappedExecutableRuntimeIssue::NonCanonicalMappedPath(
            plan.canonical_path.clone(),
        ));
    }

    let mut file = File::open(&plan.canonical_path).map_err(|error| {
        MappedExecutableRuntimeIssue::BackingFileUnavailable {
            path: plan.canonical_path.clone(),
            reason: error.to_string(),
        }
    })?;
    let before = file.metadata().map_err(|error| {
        MappedExecutableRuntimeIssue::BackingFileUnavailable {
            path: plan.canonical_path.clone(),
            reason: format!("metadata: {error}"),
        }
    })?;
    if !before.is_file()
        || linux_dev_major(before.dev()) != plan.device_major
        || linux_dev_minor(before.dev()) != plan.device_minor
        || before.ino() != plan.inode
    {
        return Err(MappedExecutableRuntimeIssue::BackingFileMetadataMismatch(
            plan.canonical_path.clone(),
        ));
    }
    if before.uid() != 0 {
        return Err(MappedExecutableRuntimeIssue::BackingFileNotRootOwned(
            plan.canonical_path.clone(),
        ));
    }
    if before.mode() & 0o222 != 0 {
        return Err(MappedExecutableRuntimeIssue::BackingFileWritable(
            plan.canonical_path.clone(),
        ));
    }
    if before.len() == 0 || before.len() > policy.max_single_file_bytes {
        return Err(MappedExecutableRuntimeIssue::BackingFileTooLarge {
            path: plan.canonical_path.clone(),
            observed: before.len(),
            maximum: policy.max_single_file_bytes,
        });
    }

    let full_file_blake3 = hash_full_file(&mut file, &plan.canonical_path)?;
    let after_full_hash = file.metadata().map_err(|error| {
        MappedExecutableRuntimeIssue::BackingFileUnavailable {
            path: plan.canonical_path.clone(),
            reason: format!("post-hash metadata: {error}"),
        }
    })?;
    if !stable_metadata(&before, &after_full_hash) {
        return Err(MappedExecutableRuntimeIssue::BackingFileMetadataMismatch(
            plan.canonical_path.clone(),
        ));
    }

    let mut segments = plan.segments.clone();
    segments.sort_by_key(|segment| (segment.file_offset, segment.start));
    let mut evidence = Vec::with_capacity(segments.len());
    for segment in &segments {
        let end_offset = segment.file_offset.checked_add(segment.length()).ok_or_else(|| {
            MappedExecutableRuntimeIssue::MappingOutsideBackingFile(plan.canonical_path.clone())
        })?;
        if end_offset > before.len() {
            return Err(MappedExecutableRuntimeIssue::MappingOutsideBackingFile(
                plan.canonical_path.clone(),
            ));
        }
        let mapped_first = hash_region(
            proc_mem,
            segment.start,
            segment.length(),
            &plan.canonical_path,
            true,
        )?;
        let backing_digest = hash_region(
            &file,
            segment.file_offset,
            segment.length(),
            &plan.canonical_path,
            false,
        )?;
        let mapped_second = hash_region(
            proc_mem,
            segment.start,
            segment.length(),
            &plan.canonical_path,
            true,
        )?;
        if mapped_first != mapped_second {
            return Err(MappedExecutableRuntimeIssue::ExecutableMappingBytesUnstable(
                plan.canonical_path.clone(),
            ));
        }
        if mapped_first != backing_digest {
            return Err(MappedExecutableRuntimeIssue::ExecutableMappingBytesMismatch(
                plan.canonical_path.clone(),
            ));
        }
        evidence.push(MappedExecutableSegmentEvidence {
            file_offset: segment.file_offset,
            length: segment.length(),
            permissions: segment.permissions.clone(),
            mapped_bytes_blake3: mapped_first,
            backing_bytes_blake3: backing_digest,
        });
    }

    let after_segments = file.metadata().map_err(|error| {
        MappedExecutableRuntimeIssue::BackingFileUnavailable {
            path: plan.canonical_path.clone(),
            reason: format!("post-segment metadata: {error}"),
        }
    })?;
    if !stable_metadata(&before, &after_segments) {
        return Err(MappedExecutableRuntimeIssue::BackingFileMetadataMismatch(
            plan.canonical_path.clone(),
        ));
    }

    Ok(MappedExecutableObjectEvidence {
        canonical_path: plan.canonical_path.clone(),
        nix_store_root: plan.nix_store_root.clone(),
        device_major: plan.device_major,
        device_minor: plan.device_minor,
        inode: plan.inode,
        file_size: before.len(),
        mode: before.mode(),
        uid: before.uid(),
        gid: before.gid(),
        mtime_seconds: before.mtime(),
        mtime_nanoseconds: before.mtime_nsec(),
        full_file_blake3,
        executable_segments: evidence,
    })
}

#[cfg(target_os = "linux")]
fn hash_full_file(file: &mut File, path: &str) -> Result<String, MappedExecutableRuntimeIssue> {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; IO_CHUNK_BYTES];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            MappedExecutableRuntimeIssue::BackingFileReadUnavailable {
                path: path.into(),
                reason: error.to_string(),
            }
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

#[cfg(target_os = "linux")]
fn hash_region(
    file: &File,
    offset: u64,
    length: u64,
    path: &str,
    proc_mem: bool,
) -> Result<String, MappedExecutableRuntimeIssue> {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; IO_CHUNK_BYTES];
    let mut done = 0u64;
    while done < length {
        let wanted = std::cmp::min(IO_CHUNK_BYTES as u64, length - done) as usize;
        let read = file.read_at(&mut buffer[..wanted], offset + done).map_err(|error| {
            if proc_mem {
                MappedExecutableRuntimeIssue::ProcMemReadUnavailable {
                    path: path.into(),
                    reason: error.to_string(),
                }
            } else {
                MappedExecutableRuntimeIssue::BackingFileReadUnavailable {
                    path: path.into(),
                    reason: error.to_string(),
                }
            }
        })?;
        if read == 0 {
            return Err(if proc_mem {
                MappedExecutableRuntimeIssue::ProcMemReadUnavailable {
                    path: path.into(),
                    reason: "unexpected EOF".into(),
                }
            } else {
                MappedExecutableRuntimeIssue::BackingFileReadUnavailable {
                    path: path.into(),
                    reason: "unexpected EOF".into(),
                }
            });
        }
        hasher.update(&buffer[..read]);
        done += read as u64;
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

#[cfg(target_os = "linux")]
fn stable_metadata(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.mode() == right.mode()
        && left.uid() == right.uid()
        && left.gid() == right.gid()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
}

#[cfg(target_os = "linux")]
fn linux_dev_major(dev: u64) -> u64 {
    ((dev >> 8) & 0x0fff) | ((dev >> 32) & 0xffff_f000)
}

#[cfg(target_os = "linux")]
fn linux_dev_minor(dev: u64) -> u64 {
    (dev & 0x00ff) | ((dev >> 12) & 0xffff_ff00)
}

fn recompute_closure_manifest_digest(
    root: &str,
    entries: &[NixClosureEntry],
) -> Result<(String, BTreeSet<String>), String> {
    if !valid_store_object_path(root) || entries.is_empty() {
        return Err("invalid-root-or-empty".into());
    }
    let mut normalized = entries.to_vec();
    for entry in &mut normalized {
        if !valid_store_object_path(&entry.store_path)
            || !valid_nar_hash(&entry.nar_hash)
            || entry.nar_size == 0
            || !entry
                .content_address
                .as_ref()
                .map(|value| canonical_text(value))
                .unwrap_or(true)
            || entry
                .references
                .iter()
                .any(|reference| !valid_store_object_path(reference))
        {
            return Err(format!("invalid-entry:{}", entry.store_path));
        }
        entry.references.sort();
        if !unique(&entry.references) {
            return Err(format!("duplicate-reference:{}", entry.store_path));
        }
    }
    normalized.sort_by(|left, right| left.store_path.cmp(&right.store_path));
    if normalized.windows(2).any(|pair| pair[0].store_path == pair[1].store_path) {
        return Err("duplicate-store-path".into());
    }

    let by_path = normalized
        .iter()
        .map(|entry| (entry.store_path.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    if !by_path.contains_key(root) {
        return Err("root-missing".into());
    }
    for entry in &normalized {
        for reference in &entry.references {
            if !by_path.contains_key(reference.as_str()) {
                return Err(format!("missing-reference:{}:{reference}", entry.store_path));
            }
        }
    }
    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::from([root.to_string()]);
    while let Some(path) = queue.pop_front() {
        if !reachable.insert(path.clone()) {
            continue;
        }
        let entry = by_path.get(path.as_str()).ok_or_else(|| format!("unresolved:{path}"))?;
        for reference in &entry.references {
            if reference != &path {
                queue.push_back(reference.clone());
            }
        }
    }
    if reachable.len() != normalized.len() {
        return Err("unreachable-store-object".into());
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(NIX_CLOSURE_CONTENT_DIGEST_DOMAIN);
    push_field(&mut hasher, root);
    hasher.update(&(normalized.len() as u64).to_le_bytes());
    for entry in &normalized {
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
    let roots = normalized.iter().map(|entry| entry.store_path.clone()).collect();
    Ok((format!("blake3:{}", hasher.finalize().to_hex()), roots))
}

fn mapped_object_set_digest(objects: &[MappedExecutableObjectEvidence]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MAPPED_OBJECT_SET_DIGEST_DOMAIN);
    hasher.update(&(objects.len() as u64).to_le_bytes());
    for object in objects {
        for field in [
            object.canonical_path.as_str(),
            object.nix_store_root.as_str(),
            object.full_file_blake3.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        for value in [object.device_major, object.device_minor, object.inode, object.file_size] {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&object.mode.to_le_bytes());
        hasher.update(&object.uid.to_le_bytes());
        hasher.update(&object.gid.to_le_bytes());
        hasher.update(&object.mtime_seconds.to_le_bytes());
        hasher.update(&object.mtime_nanoseconds.to_le_bytes());
        hasher.update(&(object.executable_segments.len() as u64).to_le_bytes());
        for segment in &object.executable_segments {
            hasher.update(&segment.file_offset.to_le_bytes());
            hasher.update(&segment.length.to_le_bytes());
            push_field(&mut hasher, &segment.permissions);
            push_field(&mut hasher, &segment.mapped_bytes_blake3);
            push_field(&mut hasher, &segment.backing_bytes_blake3);
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn qualification_digest(
    policy_digest: &str,
    nix_binding_qualification_digest: &str,
    closure_qualification_digest: &str,
    proc_maps_digest: &str,
    mapped_object_set_digest: &str,
    observed_at_ms: u64,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        nix_binding_qualification_digest,
        closure_qualification_digest,
        proc_maps_digest,
        mapped_object_set_digest,
        report_digest,
    ] {
        push_field(&mut hasher, field);
    }
    hasher.update(&observed_at_ms.to_le_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn direct_nix_store_root(path: &str) -> Option<String> {
    let suffix = path.strip_prefix("/nix/store/")?;
    let component = suffix.split('/').next()?;
    if component.len() <= 33 || component.as_bytes()[32] != b'-' {
        return None;
    }
    if !component.as_bytes()[..32].iter().all(|byte| NIX_BASE32.contains(byte)) {
        return None;
    }
    if component[33..].is_empty()
        || !component[33..].bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.' | b'_' | b'?' | b'=')
        })
    {
        return None;
    }
    if Path::new(path).components().any(|part| matches!(part, Component::ParentDir)) {
        return None;
    }
    Some(format!("/nix/store/{component}"))
}

fn valid_store_object_path(path: &str) -> bool {
    let Some(component) = path.strip_prefix("/nix/store/") else { return false; };
    if component.contains('/') || component.len() <= 33 || component.as_bytes()[32] != b'-' {
        return false;
    }
    component.as_bytes()[..32].iter().all(|byte| NIX_BASE32.contains(byte))
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

fn valid_kernel_pseudo_mapping(value: &str) -> bool {
    canonical_text(value)
        && value.starts_with('[')
        && value.ends_with(']')
        && !value.chars().any(char::is_whitespace)
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_blake3_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else { return false; };
    hex.len() == 64
        && hex.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn valid_refs(refs: &[String]) -> bool {
    refs.len() <= MAX_EVIDENCE_REFS
        && refs.iter().all(|value| canonical_text(value))
        && unique(refs)
}

fn unique(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.as_str()))
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
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

    fn blake(value: &[u8]) -> String {
        format!("blake3:{}", blake3::hash(value).to_hex())
    }

    fn policy() -> MappedExecutableRuntimePolicy {
        MappedExecutableRuntimePolicy {
            schema_version: MAPPED_EXECUTABLE_RUNTIME_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:mapped-runtime:1".into(),
            expected_nix_binding_policy_digest: blake(b"binding"),
            expected_closure_policy_digest: blake(b"closure"),
            expected_runtime_verifier_ref: "verifier:host".into(),
            expected_backend_id: "backend:in-process".into(),
            allowed_kernel_executable_mappings: vec!["[vdso]".into(), "[vsyscall]".into()],
            max_maps_bytes: 1024 * 1024,
            max_map_entries: 4096,
            max_executable_mappings: 1024,
            max_mapped_files: 512,
            max_single_file_bytes: 1024 * 1024 * 1024,
            max_total_backing_file_bytes: 4 * 1024 * 1024 * 1024,
            max_total_executable_bytes: 2 * 1024 * 1024 * 1024,
            evidence_refs: vec!["review:mapped-runtime".into(), "review:procfs".into()],
        }
    }

    fn store(name: &str) -> String {
        format!("/nix/store/0123456789abcdfghijklmnpqrsvwxyz-{name}")
    }

    fn entry(name: &str, refs: Vec<String>) -> NixClosureEntry {
        NixClosureEntry {
            store_path: store(name),
            nar_hash: "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=".into(),
            nar_size: 4096,
            references: refs,
            content_address: None,
        }
    }

    #[test]
    fn policy_reference_and_kernel_mapping_order_are_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        right.allowed_kernel_executable_mappings.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn proc_maps_parser_preserves_path() {
        let path = store("host") + "/bin/symthaea";
        let line = format!("00400000-00452000 r-xp 00001000 08:02 173521 {path}");
        let parsed = parse_proc_maps_line(&line).unwrap();
        assert_eq!(parsed.start, 0x0040_0000);
        assert_eq!(parsed.end, 0x0045_2000);
        assert_eq!(parsed.file_offset, 0x1000);
        assert_eq!(parsed.device_major, 8);
        assert_eq!(parsed.device_minor, 2);
        assert_eq!(parsed.inode, 173521);
        assert_eq!(parsed.pathname, path);
        assert!(parsed.executable());
        assert!(!parsed.writable());
    }

    #[test]
    fn writable_executable_mapping_fails_closed() {
        let path = store("host") + "/bin/symthaea";
        let bytes = format!("00400000-00401000 rwxp 00000000 08:02 1 {path}\n");
        let maps = parse_proc_maps(bytes.as_bytes(), 16).unwrap();
        let roots = BTreeSet::from([store("host")]);
        let issue = executable_file_plans(&policy(), &maps, &roots).unwrap_err();
        assert!(matches!(issue, MappedExecutableRuntimeIssue::WritableExecutableMapping(_)));
    }

    #[test]
    fn unapproved_anonymous_or_kernel_executable_mapping_fails_closed() {
        let anon = parse_proc_maps(b"00400000-00401000 r-xp 00000000 00:00 0\n", 16).unwrap();
        let issue = executable_file_plans(&policy(), &anon, &BTreeSet::new()).unwrap_err();
        assert_eq!(issue, MappedExecutableRuntimeIssue::AnonymousExecutableMapping);

        let jit = parse_proc_maps(
            b"00400000-00401000 r-xp 00000000 00:00 0 [anon:jit]\n",
            16,
        )
        .unwrap();
        let issue = executable_file_plans(&policy(), &jit, &BTreeSet::new()).unwrap_err();
        assert!(matches!(
            issue,
            MappedExecutableRuntimeIssue::UnapprovedKernelExecutableMapping(_)
        ));
    }

    #[test]
    fn approved_vdso_is_not_misclassified_as_nix_file() {
        let maps = parse_proc_maps(
            b"7fff0000-7fff1000 r-xp 00000000 00:00 0 [vdso]\n",
            16,
        )
        .unwrap();
        let (plans, executable, kernel, bytes) =
            executable_file_plans(&policy(), &maps, &BTreeSet::new()).unwrap();
        assert!(plans.is_empty());
        assert_eq!(executable, 1);
        assert_eq!(kernel, 1);
        assert_eq!(bytes, 4096);
    }

    #[test]
    fn non_nix_or_out_of_closure_executable_mapping_fails_closed() {
        let foreign = parse_proc_maps(
            b"00400000-00401000 r-xp 00000000 08:02 1 /usr/bin/example\n",
            16,
        )
        .unwrap();
        let issue = executable_file_plans(&policy(), &foreign, &BTreeSet::new()).unwrap_err();
        assert!(matches!(issue, MappedExecutableRuntimeIssue::NonNixExecutableMapping(_)));

        let path = store("host") + "/bin/symthaea";
        let bytes = format!("00400000-00401000 r-xp 00000000 08:02 1 {path}\n");
        let maps = parse_proc_maps(bytes.as_bytes(), 16).unwrap();
        let issue = executable_file_plans(&policy(), &maps, &BTreeSet::new()).unwrap_err();
        assert!(matches!(
            issue,
            MappedExecutableRuntimeIssue::MappedStoreObjectOutsideClosure(_)
        ));
    }

    #[test]
    fn closure_manifest_rebinding_is_order_independent_but_complete() {
        let dep = entry("dep", vec![]);
        let root = entry("host", vec![dep.store_path.clone()]);
        let root_path = root.store_path.clone();
        let (left, roots_left) = recompute_closure_manifest_digest(
            &root_path,
            &[root.clone(), dep.clone()],
        )
        .unwrap();
        let (right, roots_right) = recompute_closure_manifest_digest(
            &root_path,
            &[dep.clone(), root.clone()],
        )
        .unwrap();
        assert_eq!(left, right);
        assert_eq!(roots_left, roots_right);

        let extra = entry("extra", vec![]);
        assert!(recompute_closure_manifest_digest(&root_path, &[root, dep, extra]).is_err());
    }

    #[test]
    fn object_set_digest_changes_when_mapped_bytes_change() {
        let object = MappedExecutableObjectEvidence {
            canonical_path: store("host") + "/bin/symthaea",
            nix_store_root: store("host"),
            device_major: 8,
            device_minor: 2,
            inode: 1,
            file_size: 8192,
            mode: 0o100555,
            uid: 0,
            gid: 0,
            mtime_seconds: 1,
            mtime_nanoseconds: 2,
            full_file_blake3: blake(b"file"),
            executable_segments: vec![MappedExecutableSegmentEvidence {
                file_offset: 4096,
                length: 4096,
                permissions: "r-xp".into(),
                mapped_bytes_blake3: blake(b"mapped"),
                backing_bytes_blake3: blake(b"mapped"),
            }],
        };
        let left = mapped_object_set_digest(std::slice::from_ref(&object));
        let mut changed = object;
        changed.executable_segments[0].mapped_bytes_blake3 = blake(b"changed");
        let right = mapped_object_set_digest(&[changed]);
        assert_ne!(left, right);
    }
}
