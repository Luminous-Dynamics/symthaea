// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact static / loader-free ELF profile assurance for a Nix-bound verifier.
//!
//! This crate parses the exact executable bytes already named by a qualified
//! `NixBoundInProcessRuntimePolicy`. Qualification requires an ELF64
//! little-endian executable for the reviewed machine, no `PT_INTERP`, no
//! `DT_NEEDED`, no writable+executable `PT_LOAD`, and exactly one explicit
//! non-executable `PT_GNU_STACK`.
//!
//! This is an assessment theorem only. It does not execute the binary, prove
//! fd-pinned `execveat`, exclude post-launch `dlopen`/manual executable maps,
//! establish mapping continuity, trusted time, or physical authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_nix_bound_in_process_runtime::NixBoundInProcessRuntimePolicy;

#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::Read;
#[cfg(target_os = "linux")]
use std::os::unix::fs::MetadataExt;

pub const STATIC_ELF_LAUNCH_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.static-elf-launch-profile-policy.v1";
pub const STATIC_ELF_LAUNCH_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.static-elf-launch-profile-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.static-elf-launch-profile-policy.digest.v1\0";
const ELF_PROFILE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.static-elf-launch-profile.content.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.static-elf-launch-profile-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.static-elf-launch-profile-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 4096;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_EXECUTABLE_BYTES_HARD: u64 = 2 * 1024 * 1024 * 1024;
const MAX_PROGRAM_HEADERS_HARD: u16 = 4096;

const ELF_MAGIC: &[u8; 4] = b"\x7fELF";
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EV_CURRENT: u8 = 1;
const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;
const PN_XNUM: u16 = 0xffff;
const ELF64_EHDR_SIZE: u16 = 64;
const ELF64_PHDR_SIZE: u16 = 56;

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_INTERP: u32 = 3;
const PT_SHLIB: u32 = 5;
const PT_GNU_STACK: u32 = 0x6474_e551;
const PF_X: u32 = 1;
const PF_W: u32 = 2;
const DT_NULL: i64 = 0;
const DT_NEEDED: i64 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticElfLaunchPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_nix_binding_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub expected_elf_machine: u16,
    pub allow_static_pie: bool,
    pub max_executable_bytes: u64,
    pub max_program_headers: u16,
    pub evidence_refs: Vec<String>,
}

impl StaticElfLaunchPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == STATIC_ELF_LAUNCH_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_nix_binding_policy_digest)
            && valid_blake3_digest(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
            && self.expected_elf_machine != 0
            && (1..=MAX_EXECUTABLE_BYTES_HARD).contains(&self.max_executable_bytes)
            && (1..=MAX_PROGRAM_HEADERS_HARD).contains(&self.max_program_headers)
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
            self.expected_backend_id.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        hasher.update(&self.expected_elf_machine.to_le_bytes());
        hasher.update(&[u8::from(self.allow_static_pie)]);
        hasher.update(&self.max_executable_bytes.to_le_bytes());
        hasher.update(&self.max_program_headers.to_le_bytes());
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaticElfLaunchDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaticElfLaunchIssue {
    InvalidPolicy,
    NixBindingPolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    ExecutableUnavailable(String),
    ExecutableMetadataMismatch,
    ExecutableTooLarge { observed: u64, maximum: u64 },
    ExecutableDigestMismatch,
    MalformedElf(String),
    ElfMachineMismatch { observed: u16, expected: u16 },
    ElfTypeUnsupported(u16),
    StaticPieForbidden,
    ProgramHeaderCountExceeded { observed: u16, maximum: u16 },
    ExtendedProgramHeaderCountUnsupported,
    InterpreterPresent,
    NeededDependencyPresent,
    WritableExecutableLoadSegment,
    ExecutableLoadSegmentMissing,
    DynamicSegmentCountExceeded,
    DynamicSegmentMalformed,
    DynamicTerminatorMissing,
    ReservedShlibSegmentPresent,
    GnuStackMissing,
    MultipleGnuStackSegments,
    ExecutableGnuStack,
}

impl StaticElfLaunchIssue {
    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::MalformedElf(_)
                | Self::ProgramHeaderCountExceeded { .. }
                | Self::ExtendedProgramHeaderCountUnsupported
                | Self::DynamicSegmentMalformed
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::NixBindingPolicyMismatch => "nix-binding-policy-mismatch".into(),
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::ExecutableUnavailable(reason) => format!("executable-unavailable:{reason}"),
            Self::ExecutableMetadataMismatch => "executable-metadata-mismatch".into(),
            Self::ExecutableTooLarge { observed, maximum } => {
                format!("executable-too-large:{observed}:{maximum}")
            }
            Self::ExecutableDigestMismatch => "executable-digest-mismatch".into(),
            Self::MalformedElf(reason) => format!("malformed-elf:{reason}"),
            Self::ElfMachineMismatch { observed, expected } => {
                format!("elf-machine-mismatch:{observed}:{expected}")
            }
            Self::ElfTypeUnsupported(value) => format!("elf-type-unsupported:{value}"),
            Self::StaticPieForbidden => "static-pie-forbidden".into(),
            Self::ProgramHeaderCountExceeded { observed, maximum } => {
                format!("program-header-count-exceeded:{observed}:{maximum}")
            }
            Self::ExtendedProgramHeaderCountUnsupported => {
                "extended-program-header-count-unsupported".into()
            }
            Self::InterpreterPresent => "pt-interp-present".into(),
            Self::NeededDependencyPresent => "dt-needed-present".into(),
            Self::WritableExecutableLoadSegment => "writable-executable-load-segment".into(),
            Self::ExecutableLoadSegmentMissing => "executable-load-segment-missing".into(),
            Self::DynamicSegmentCountExceeded => "dynamic-segment-count-exceeded".into(),
            Self::DynamicSegmentMalformed => "dynamic-segment-malformed".into(),
            Self::DynamicTerminatorMissing => "dynamic-terminator-missing".into(),
            Self::ReservedShlibSegmentPresent => "reserved-pt-shlib-present".into(),
            Self::GnuStackMissing => "gnu-stack-missing".into(),
            Self::MultipleGnuStackSegments => "multiple-gnu-stack-segments".into(),
            Self::ExecutableGnuStack => "executable-gnu-stack".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticElfAnalysis {
    pub elf_type: u16,
    pub elf_machine: u16,
    pub entry_point: u64,
    pub program_header_count: u16,
    pub load_segment_count: u16,
    pub executable_load_segment_count: u16,
    pub dynamic_segment_count: u16,
    pub needed_dependency_count: u16,
    pub interpreter_segment_count: u16,
    pub gnu_stack_segment_count: u16,
    pub gnu_stack_flags: u32,
    pub profile_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticElfLaunchReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub nix_binding_policy_digest: String,
    pub nix_binding_qualification_digest: String,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub executable_path: String,
    pub expected_executable_digest: String,
    pub observed_executable_digest: Option<String>,
    pub executable_size: u64,
    pub analysis: Option<StaticElfAnalysis>,
    pub disposition: StaticElfLaunchDisposition,
    pub issues: Vec<StaticElfLaunchIssue>,
}

impl StaticElfLaunchReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.nix_binding_policy_digest.as_str(),
            self.nix_binding_qualification_digest.as_str(),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.executable_path.as_str(),
            self.expected_executable_digest.as_str(),
            self.observed_executable_digest.as_deref().unwrap_or("-"),
        ] {
            push_field(&mut hasher, field);
        }
        hasher.update(&self.executable_size.to_le_bytes());
        if let Some(analysis) = &self.analysis {
            hasher.update(&[1]);
            push_analysis(&mut hasher, analysis);
        } else {
            hasher.update(&[0]);
        }
        push_field(
            &mut hasher,
            match self.disposition {
                StaticElfLaunchDisposition::Invalid => "invalid",
                StaticElfLaunchDisposition::Blocked => "blocked",
                StaticElfLaunchDisposition::Qualified => "qualified",
            },
        );
        hasher.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            push_field(&mut hasher, &issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticElfLaunchProfile {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    nix_binding_qualification_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    executable_path: String,
    executable_digest: String,
    executable_size: u64,
    elf_profile_digest: String,
    elf_type: u16,
    elf_machine: u16,
    entry_point: u64,
    program_header_count: u16,
}

impl StaticElfLaunchProfile {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn nix_binding_qualification_digest(&self) -> &str {
        &self.nix_binding_qualification_digest
    }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn executable_path(&self) -> &str { &self.executable_path }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub const fn executable_size(&self) -> u64 { self.executable_size }
    pub fn elf_profile_digest(&self) -> &str { &self.elf_profile_digest }
    pub const fn elf_type(&self) -> u16 { self.elf_type }
    pub const fn elf_machine(&self) -> u16 { self.elf_machine }
    pub const fn entry_point(&self) -> u64 { self.entry_point }
    pub const fn program_header_count(&self) -> u16 { self.program_header_count }
    pub const fn exact_executable_content_matches_nix_bound_identity(&self) -> bool { true }
    pub const fn external_elf_interpreter_declared(&self) -> bool { false }
    pub const fn declared_dynamic_dependencies_present(&self) -> bool { false }
    pub const fn writable_executable_load_segment_present(&self) -> bool { false }
    pub const fn explicit_non_executable_gnu_stack(&self) -> bool { true }
    pub const fn static_or_static_pie_launch_profile_established(&self) -> bool { true }
    pub const fn fd_pinned_execveat_launch_performed(&self) -> bool { false }
    pub const fn post_launch_dynamic_loading_excluded(&self) -> bool { false }
    pub const fn mapping_continuity_since_exec_established(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticElfLaunchQualification {
    pub report: StaticElfLaunchReport,
    profile: StaticElfLaunchProfile,
}

impl StaticElfLaunchQualification {
    pub fn profile(&self) -> &StaticElfLaunchProfile { &self.profile }
    pub fn into_profile(self) -> StaticElfLaunchProfile { self.profile }
}

#[cfg(target_os = "linux")]
pub fn qualify_static_elf_launch_profile(
    policy: &StaticElfLaunchPolicy,
    bound: &NixBoundInProcessRuntimePolicy,
) -> Result<StaticElfLaunchQualification, StaticElfLaunchReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = base_report(policy, policy_digest.clone(), bound);

    if !policy.validate() {
        report.issues.push(StaticElfLaunchIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if bound.policy_digest() != policy.expected_nix_binding_policy_digest {
        report.issues.push(StaticElfLaunchIssue::NixBindingPolicyMismatch);
    }
    if bound.runtime_policy_digest() != policy.expected_runtime_policy_digest {
        report.issues.push(StaticElfLaunchIssue::RuntimePolicyMismatch);
    }
    if bound.runtime_verifier_ref() != policy.expected_runtime_verifier_ref {
        report.issues.push(StaticElfLaunchIssue::RuntimeVerifierMismatch);
    }
    if bound.backend_id() != policy.expected_backend_id {
        report.issues.push(StaticElfLaunchIssue::BackendMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let mut file = match File::open(bound.executable_path()) {
        Ok(value) => value,
        Err(error) => {
            report.issues.push(StaticElfLaunchIssue::ExecutableUnavailable(error.to_string()));
            return Err(finalize(report));
        }
    };
    let before = match file.metadata() {
        Ok(value) => value,
        Err(error) => {
            report.issues.push(StaticElfLaunchIssue::ExecutableUnavailable(error.to_string()));
            return Err(finalize(report));
        }
    };
    report.executable_size = before.len();
    if !before.is_file()
        || before.uid() != 0
        || before.mode() & 0o222 != 0
        || before.mode() & 0o111 == 0
    {
        report.issues.push(StaticElfLaunchIssue::ExecutableMetadataMismatch);
        return Err(finalize(report));
    }
    if before.len() == 0 || before.len() > policy.max_executable_bytes {
        report.issues.push(StaticElfLaunchIssue::ExecutableTooLarge {
            observed: before.len(),
            maximum: policy.max_executable_bytes,
        });
        return Err(finalize(report));
    }

    let mut bytes = Vec::with_capacity(before.len() as usize);
    if let Err(error) = file.read_to_end(&mut bytes) {
        report.issues.push(StaticElfLaunchIssue::ExecutableUnavailable(error.to_string()));
        return Err(finalize(report));
    }
    let after = match file.metadata() {
        Ok(value) => value,
        Err(error) => {
            report.issues.push(StaticElfLaunchIssue::ExecutableUnavailable(error.to_string()));
            return Err(finalize(report));
        }
    };
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.mode() != after.mode()
        || before.uid() != after.uid()
        || before.gid() != after.gid()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
        || bytes.len() as u64 != before.len()
    {
        report.issues.push(StaticElfLaunchIssue::ExecutableMetadataMismatch);
        return Err(finalize(report));
    }

    let executable_digest = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    report.observed_executable_digest = Some(executable_digest.clone());
    if executable_digest != bound.executable_digest() {
        report.issues.push(StaticElfLaunchIssue::ExecutableDigestMismatch);
        return Err(finalize(report));
    }

    let (analysis, issues) = analyze_elf(&bytes, policy);
    report.issues.extend(issues);
    report.analysis = analysis.clone();
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }
    let analysis = analysis.expect("issue-free ELF analysis exists");

    report.disposition = StaticElfLaunchDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        bound.qualification_digest(),
        &executable_digest,
        &analysis.profile_digest,
        &report_digest,
    );
    let profile = StaticElfLaunchProfile {
        qualification_digest,
        report_digest,
        policy_digest,
        nix_binding_qualification_digest: bound.qualification_digest().into(),
        runtime_policy_digest: bound.runtime_policy_digest().into(),
        runtime_verifier_ref: bound.runtime_verifier_ref().into(),
        backend_id: bound.backend_id().into(),
        executable_path: bound.executable_path().into(),
        executable_digest,
        executable_size: before.len(),
        elf_profile_digest: analysis.profile_digest.clone(),
        elf_type: analysis.elf_type,
        elf_machine: analysis.elf_machine,
        entry_point: analysis.entry_point,
        program_header_count: analysis.program_header_count,
    };

    Ok(StaticElfLaunchQualification { report, profile })
}

#[cfg(not(target_os = "linux"))]
pub fn qualify_static_elf_launch_profile(
    policy: &StaticElfLaunchPolicy,
    bound: &NixBoundInProcessRuntimePolicy,
) -> Result<StaticElfLaunchQualification, StaticElfLaunchReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = base_report(policy, policy_digest, bound);
    report.issues.push(StaticElfLaunchIssue::ExecutableUnavailable(
        "Linux Nix runtime host required".into(),
    ));
    Err(finalize(report))
}

fn analyze_elf(
    bytes: &[u8],
    policy: &StaticElfLaunchPolicy,
) -> (Option<StaticElfAnalysis>, Vec<StaticElfLaunchIssue>) {
    let mut issues = Vec::new();
    if bytes.len() < ELF64_EHDR_SIZE as usize {
        issues.push(StaticElfLaunchIssue::MalformedElf("header-too-short".into()));
        return (None, issues);
    }
    if &bytes[..4] != ELF_MAGIC {
        issues.push(StaticElfLaunchIssue::MalformedElf("bad-magic".into()));
        return (None, issues);
    }
    if bytes[4] != ELFCLASS64 {
        issues.push(StaticElfLaunchIssue::MalformedElf("not-elf64".into()));
        return (None, issues);
    }
    if bytes[5] != ELFDATA2LSB {
        issues.push(StaticElfLaunchIssue::MalformedElf("not-little-endian".into()));
        return (None, issues);
    }
    if bytes[6] != EV_CURRENT {
        issues.push(StaticElfLaunchIssue::MalformedElf("bad-ident-version".into()));
        return (None, issues);
    }

    let elf_type = read_u16(bytes, 16).expect("checked ELF header");
    let elf_machine = read_u16(bytes, 18).expect("checked ELF header");
    let elf_version = read_u32(bytes, 20).expect("checked ELF header");
    let entry_point = read_u64(bytes, 24).expect("checked ELF header");
    let program_header_offset = read_u64(bytes, 32).expect("checked ELF header");
    let elf_header_size = read_u16(bytes, 52).expect("checked ELF header");
    let program_header_entry_size = read_u16(bytes, 54).expect("checked ELF header");
    let program_header_count = read_u16(bytes, 56).expect("checked ELF header");

    if elf_version != 1
        || elf_header_size != ELF64_EHDR_SIZE
        || program_header_entry_size != ELF64_PHDR_SIZE
        || program_header_offset < ELF64_EHDR_SIZE as u64
        || entry_point == 0
    {
        issues.push(StaticElfLaunchIssue::MalformedElf("invalid-elf-header-fields".into()));
        return (None, issues);
    }
    if program_header_count == PN_XNUM {
        issues.push(StaticElfLaunchIssue::ExtendedProgramHeaderCountUnsupported);
        return (None, issues);
    }
    if program_header_count == 0 || program_header_count > policy.max_program_headers {
        issues.push(StaticElfLaunchIssue::ProgramHeaderCountExceeded {
            observed: program_header_count,
            maximum: policy.max_program_headers,
        });
        return (None, issues);
    }
    if elf_machine != policy.expected_elf_machine {
        issues.push(StaticElfLaunchIssue::ElfMachineMismatch {
            observed: elf_machine,
            expected: policy.expected_elf_machine,
        });
    }
    match elf_type {
        ET_EXEC => {}
        ET_DYN if policy.allow_static_pie => {}
        ET_DYN => issues.push(StaticElfLaunchIssue::StaticPieForbidden),
        other => issues.push(StaticElfLaunchIssue::ElfTypeUnsupported(other)),
    }

    let table_bytes = match (program_header_count as u64).checked_mul(ELF64_PHDR_SIZE as u64) {
        Some(value) => value,
        None => {
            issues.push(StaticElfLaunchIssue::MalformedElf("program-header-size-overflow".into()));
            return (None, issues);
        }
    };
    let table_end = match program_header_offset.checked_add(table_bytes) {
        Some(value) => value,
        None => {
            issues.push(StaticElfLaunchIssue::MalformedElf("program-header-end-overflow".into()));
            return (None, issues);
        }
    };
    if table_end > bytes.len() as u64 {
        issues.push(StaticElfLaunchIssue::MalformedElf("program-header-table-out-of-bounds".into()));
        return (None, issues);
    }

    let mut load_count = 0u16;
    let mut executable_load_count = 0u16;
    let mut dynamic_count = 0u16;
    let mut needed_count = 0u16;
    let mut interp_count = 0u16;
    let mut gnu_stack_count = 0u16;
    let mut gnu_stack_flags = 0u32;
    let mut previous_load_vaddr: Option<u64> = None;

    for index in 0..program_header_count as usize {
        let offset = program_header_offset as usize + index * ELF64_PHDR_SIZE as usize;
        let p_type = read_u32(bytes, offset).expect("program header table bounded");
        let p_flags = read_u32(bytes, offset + 4).expect("program header table bounded");
        let p_offset = read_u64(bytes, offset + 8).expect("program header table bounded");
        let p_vaddr = read_u64(bytes, offset + 16).expect("program header table bounded");
        let p_filesz = read_u64(bytes, offset + 32).expect("program header table bounded");
        let p_memsz = read_u64(bytes, offset + 40).expect("program header table bounded");
        let p_align = read_u64(bytes, offset + 48).expect("program header table bounded");

        if p_filesz > 0 {
            let Some(segment_end) = p_offset.checked_add(p_filesz) else {
                issues.push(StaticElfLaunchIssue::MalformedElf(format!(
                    "segment-{index}-end-overflow"
                )));
                continue;
            };
            if segment_end > bytes.len() as u64 {
                issues.push(StaticElfLaunchIssue::MalformedElf(format!(
                    "segment-{index}-out-of-bounds"
                )));
                continue;
            }
        }

        match p_type {
            PT_LOAD => {
                load_count = load_count.saturating_add(1);
                if p_filesz > p_memsz {
                    issues.push(StaticElfLaunchIssue::MalformedElf(format!(
                        "load-{index}-filesz-exceeds-memsz"
                    )));
                }
                if p_align > 1
                    && (!p_align.is_power_of_two() || p_vaddr % p_align != p_offset % p_align)
                {
                    issues.push(StaticElfLaunchIssue::MalformedElf(format!(
                        "load-{index}-invalid-alignment"
                    )));
                }
                if previous_load_vaddr.is_some_and(|previous| p_vaddr < previous) {
                    issues.push(StaticElfLaunchIssue::MalformedElf(format!(
                        "load-{index}-vaddr-order"
                    )));
                }
                previous_load_vaddr = Some(p_vaddr);
                if p_flags & PF_X != 0 {
                    executable_load_count = executable_load_count.saturating_add(1);
                    if p_flags & PF_W != 0 {
                        issues.push(StaticElfLaunchIssue::WritableExecutableLoadSegment);
                    }
                }
            }
            PT_DYNAMIC => {
                dynamic_count = dynamic_count.saturating_add(1);
                if dynamic_count > 1 {
                    issues.push(StaticElfLaunchIssue::DynamicSegmentCountExceeded);
                }
                match parse_dynamic_segment(bytes, p_offset, p_filesz) {
                    Ok(count) => needed_count = needed_count.saturating_add(count),
                    Err(DynamicParseError::Malformed) => {
                        issues.push(StaticElfLaunchIssue::DynamicSegmentMalformed)
                    }
                    Err(DynamicParseError::TerminatorMissing) => {
                        issues.push(StaticElfLaunchIssue::DynamicTerminatorMissing)
                    }
                }
            }
            PT_INTERP => {
                interp_count = interp_count.saturating_add(1);
                issues.push(StaticElfLaunchIssue::InterpreterPresent);
            }
            PT_SHLIB => issues.push(StaticElfLaunchIssue::ReservedShlibSegmentPresent),
            PT_GNU_STACK => {
                gnu_stack_count = gnu_stack_count.saturating_add(1);
                gnu_stack_flags = p_flags;
                if gnu_stack_count > 1 {
                    issues.push(StaticElfLaunchIssue::MultipleGnuStackSegments);
                }
                if p_flags & PF_X != 0 {
                    issues.push(StaticElfLaunchIssue::ExecutableGnuStack);
                }
            }
            _ => {}
        }
    }

    if executable_load_count == 0 {
        issues.push(StaticElfLaunchIssue::ExecutableLoadSegmentMissing);
    }
    if needed_count > 0 {
        issues.push(StaticElfLaunchIssue::NeededDependencyPresent);
    }
    if gnu_stack_count == 0 {
        issues.push(StaticElfLaunchIssue::GnuStackMissing);
    }

    let profile_digest = elf_profile_digest(
        elf_type,
        elf_machine,
        entry_point,
        program_header_count,
        load_count,
        executable_load_count,
        dynamic_count,
        needed_count,
        interp_count,
        gnu_stack_count,
        gnu_stack_flags,
    );
    let analysis = StaticElfAnalysis {
        elf_type,
        elf_machine,
        entry_point,
        program_header_count,
        load_segment_count: load_count,
        executable_load_segment_count: executable_load_count,
        dynamic_segment_count: dynamic_count,
        needed_dependency_count: needed_count,
        interpreter_segment_count: interp_count,
        gnu_stack_segment_count: gnu_stack_count,
        gnu_stack_flags,
        profile_digest,
    };
    (Some(analysis), issues)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DynamicParseError {
    Malformed,
    TerminatorMissing,
}

fn parse_dynamic_segment(
    bytes: &[u8],
    offset: u64,
    size: u64,
) -> Result<u16, DynamicParseError> {
    if size == 0 || size % 16 != 0 {
        return Err(DynamicParseError::Malformed);
    }
    let end = offset.checked_add(size).ok_or(DynamicParseError::Malformed)?;
    if end > bytes.len() as u64 {
        return Err(DynamicParseError::Malformed);
    }
    let mut needed = 0u16;
    let mut cursor = offset as usize;
    let end = end as usize;
    while cursor < end {
        let tag = read_i64(bytes, cursor).ok_or(DynamicParseError::Malformed)?;
        if tag == DT_NULL {
            return Ok(needed);
        }
        if tag == DT_NEEDED {
            needed = needed.saturating_add(1);
        }
        cursor += 16;
    }
    Err(DynamicParseError::TerminatorMissing)
}

#[allow(clippy::too_many_arguments)]
fn elf_profile_digest(
    elf_type: u16,
    elf_machine: u16,
    entry_point: u64,
    ph_count: u16,
    load_count: u16,
    executable_load_count: u16,
    dynamic_count: u16,
    needed_count: u16,
    interp_count: u16,
    gnu_stack_count: u16,
    gnu_stack_flags: u32,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ELF_PROFILE_DIGEST_DOMAIN);
    for value in [
        elf_type,
        elf_machine,
        ph_count,
        load_count,
        executable_load_count,
        dynamic_count,
        needed_count,
        interp_count,
        gnu_stack_count,
    ] {
        hasher.update(&value.to_le_bytes());
    }
    hasher.update(&entry_point.to_le_bytes());
    hasher.update(&gnu_stack_flags.to_le_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn push_analysis(hasher: &mut blake3::Hasher, analysis: &StaticElfAnalysis) {
    for value in [
        analysis.elf_type,
        analysis.elf_machine,
        analysis.program_header_count,
        analysis.load_segment_count,
        analysis.executable_load_segment_count,
        analysis.dynamic_segment_count,
        analysis.needed_dependency_count,
        analysis.interpreter_segment_count,
        analysis.gnu_stack_segment_count,
    ] {
        hasher.update(&value.to_le_bytes());
    }
    hasher.update(&analysis.entry_point.to_le_bytes());
    hasher.update(&analysis.gnu_stack_flags.to_le_bytes());
    push_field(hasher, &analysis.profile_digest);
}

fn qualification_digest(
    policy_digest: &str,
    nix_binding_qualification_digest: &str,
    executable_digest: &str,
    elf_profile_digest: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        nix_binding_qualification_digest,
        executable_digest,
        elf_profile_digest,
        report_digest,
    ] {
        push_field(&mut hasher, field);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn base_report(
    policy: &StaticElfLaunchPolicy,
    policy_digest: Option<String>,
    bound: &NixBoundInProcessRuntimePolicy,
) -> StaticElfLaunchReport {
    StaticElfLaunchReport {
        schema_version: STATIC_ELF_LAUNCH_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        nix_binding_policy_digest: bound.policy_digest().into(),
        nix_binding_qualification_digest: bound.qualification_digest().into(),
        runtime_policy_digest: bound.runtime_policy_digest().into(),
        runtime_verifier_ref: bound.runtime_verifier_ref().into(),
        backend_id: bound.backend_id().into(),
        executable_path: bound.executable_path().into(),
        expected_executable_digest: bound.executable_digest().into(),
        observed_executable_digest: None,
        executable_size: 0,
        analysis: None,
        disposition: StaticElfLaunchDisposition::Invalid,
        issues: Vec::new(),
    }
}

fn finalize(mut report: StaticElfLaunchReport) -> StaticElfLaunchReport {
    report.disposition = if report.issues.iter().any(StaticElfLaunchIssue::is_invalid) {
        StaticElfLaunchDisposition::Invalid
    } else {
        StaticElfLaunchDisposition::Blocked
    };
    report
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(offset..offset + 2)?.try_into().ok()?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(bytes.get(offset..offset + 8)?.try_into().ok()?))
}

fn read_i64(bytes: &[u8], offset: usize) -> Option<i64> {
    Some(i64::from_le_bytes(bytes.get(offset..offset + 8)?.try_into().ok()?))
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
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

    const EM_X86_64: u16 = 62;
    const PF_R: u32 = 4;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> StaticElfLaunchPolicy {
        StaticElfLaunchPolicy {
            schema_version: STATIC_ELF_LAUNCH_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:static-elf:1".into(),
            expected_nix_binding_policy_digest: d("binding-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:host".into(),
            expected_backend_id: "backend:in-process".into(),
            expected_elf_machine: EM_X86_64,
            allow_static_pie: true,
            max_executable_bytes: 1024 * 1024,
            max_program_headers: 32,
            evidence_refs: vec!["review:elf".into(), "review:static".into()],
        }
    }

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i64(bytes: &mut [u8], offset: usize, value: i64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[allow(clippy::too_many_arguments)]
    fn phdr(
        bytes: &mut [u8],
        index: usize,
        p_type: u32,
        flags: u32,
        offset: u64,
        vaddr: u64,
        filesz: u64,
        memsz: u64,
        align: u64,
    ) {
        let base = ELF64_EHDR_SIZE as usize + index * ELF64_PHDR_SIZE as usize;
        put_u32(bytes, base, p_type);
        put_u32(bytes, base + 4, flags);
        put_u64(bytes, base + 8, offset);
        put_u64(bytes, base + 16, vaddr);
        put_u64(bytes, base + 24, vaddr);
        put_u64(bytes, base + 32, filesz);
        put_u64(bytes, base + 40, memsz);
        put_u64(bytes, base + 48, align);
    }

    fn base_elf(phnum: u16, extra: usize) -> Vec<u8> {
        let header_bytes = ELF64_EHDR_SIZE as usize + phnum as usize * ELF64_PHDR_SIZE as usize;
        let mut bytes = vec![0u8; header_bytes + extra];
        bytes[..4].copy_from_slice(ELF_MAGIC);
        bytes[4] = ELFCLASS64;
        bytes[5] = ELFDATA2LSB;
        bytes[6] = EV_CURRENT;
        put_u16(&mut bytes, 16, ET_EXEC);
        put_u16(&mut bytes, 18, EM_X86_64);
        put_u32(&mut bytes, 20, 1);
        put_u64(&mut bytes, 24, 0x401000);
        put_u64(&mut bytes, 32, ELF64_EHDR_SIZE as u64);
        put_u16(&mut bytes, 52, ELF64_EHDR_SIZE);
        put_u16(&mut bytes, 54, ELF64_PHDR_SIZE);
        put_u16(&mut bytes, 56, phnum);
        bytes
    }

    fn minimal_static_elf() -> Vec<u8> {
        let mut bytes = base_elf(2, 64);
        let file_len = bytes.len() as u64;
        phdr(
            &mut bytes,
            0,
            PT_LOAD,
            PF_R | PF_X,
            0,
            0x400000,
            file_len,
            file_len,
            0x1000,
        );
        phdr(&mut bytes, 1, PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, 0, 16);
        bytes
    }

    #[test]
    fn policy_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
        right.expected_elf_machine = 183;
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn minimal_static_elf_qualifies_profile_analysis() {
        let bytes = minimal_static_elf();
        let (analysis, issues) = analyze_elf(&bytes, &policy());
        assert!(issues.is_empty(), "{issues:?}");
        let analysis = analysis.unwrap();
        assert_eq!(analysis.interpreter_segment_count, 0);
        assert_eq!(analysis.needed_dependency_count, 0);
        assert_eq!(analysis.executable_load_segment_count, 1);
        assert_eq!(analysis.gnu_stack_segment_count, 1);
        assert_eq!(analysis.gnu_stack_flags & PF_X, 0);
    }

    #[test]
    fn interpreter_segment_is_blocking() {
        let mut bytes = base_elf(3, 128);
        let file_len = bytes.len() as u64;
        phdr(&mut bytes, 0, PT_LOAD, PF_R | PF_X, 0, 0x400000, file_len, file_len, 0x1000);
        let interp_offset = file_len - 32;
        bytes[interp_offset as usize..interp_offset as usize + 8].copy_from_slice(b"/ld.so\0\0");
        phdr(&mut bytes, 1, PT_INTERP, PF_R, interp_offset, 0, 8, 8, 1);
        phdr(&mut bytes, 2, PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, 0, 16);
        let (_, issues) = analyze_elf(&bytes, &policy());
        assert!(issues.contains(&StaticElfLaunchIssue::InterpreterPresent));
    }

    #[test]
    fn dt_needed_is_blocking_even_without_interpreter() {
        let mut bytes = base_elf(3, 64);
        let file_len = bytes.len() as u64;
        phdr(&mut bytes, 0, PT_LOAD, PF_R | PF_X, 0, 0x400000, file_len, file_len, 0x1000);
        let dynamic_offset = file_len - 32;
        put_i64(&mut bytes, dynamic_offset as usize, DT_NEEDED);
        put_u64(&mut bytes, dynamic_offset as usize + 8, 1);
        put_i64(&mut bytes, dynamic_offset as usize + 16, DT_NULL);
        put_u64(&mut bytes, dynamic_offset as usize + 24, 0);
        phdr(&mut bytes, 1, PT_DYNAMIC, PF_R | PF_W, dynamic_offset, 0x500000, 32, 32, 8);
        phdr(&mut bytes, 2, PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, 0, 16);
        let (analysis, issues) = analyze_elf(&bytes, &policy());
        assert!(issues.contains(&StaticElfLaunchIssue::NeededDependencyPresent));
        assert_eq!(analysis.unwrap().needed_dependency_count, 1);
    }

    #[test]
    fn wx_load_and_executable_stack_are_blocking() {
        let mut bytes = minimal_static_elf();
        let first = ELF64_EHDR_SIZE as usize;
        put_u32(&mut bytes, first + 4, PF_R | PF_W | PF_X);
        let second = first + ELF64_PHDR_SIZE as usize;
        put_u32(&mut bytes, second + 4, PF_R | PF_W | PF_X);
        let (_, issues) = analyze_elf(&bytes, &policy());
        assert!(issues.contains(&StaticElfLaunchIssue::WritableExecutableLoadSegment));
        assert!(issues.contains(&StaticElfLaunchIssue::ExecutableGnuStack));
    }

    #[test]
    fn static_pie_is_policy_semantic() {
        let mut bytes = minimal_static_elf();
        put_u16(&mut bytes, 16, ET_DYN);
        let (analysis, issues) = analyze_elf(&bytes, &policy());
        assert!(issues.is_empty(), "{issues:?}");
        assert_eq!(analysis.unwrap().elf_type, ET_DYN);

        let mut strict = policy();
        strict.allow_static_pie = false;
        let (_, issues) = analyze_elf(&bytes, &strict);
        assert!(issues.contains(&StaticElfLaunchIssue::StaticPieForbidden));
    }

    #[test]
    fn machine_mismatch_is_blocking() {
        let bytes = minimal_static_elf();
        let mut other = policy();
        other.expected_elf_machine = 183;
        let (_, issues) = analyze_elf(&bytes, &other);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            StaticElfLaunchIssue::ElfMachineMismatch { observed: EM_X86_64, expected: 183 }
        )));
    }
}
