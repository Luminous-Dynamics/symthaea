// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Retained-fd `execveat(AT_EMPTY_PATH)` launch for a qualified static verifier.
//!
//! Preparation opens the exact qualified executable once, binds the retained
//! inode/content to the static ELF + Nix runtime identities, and commits an
//! explicit argv/environment/nonce plan. Execution rechecks that same fd and
//! then uses the safe `nix` 0.26.4 `execveat` wrapper. A successful exec never
//! returns, so this crate deliberately emits no post-exec success capability.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::ffi::CString;
use symthaea_assurance_nix_bound_in_process_runtime::NixBoundInProcessRuntimePolicy;
use symthaea_assurance_static_elf_launch_profile::StaticElfLaunchProfile;

#[cfg(target_os = "linux")]
use nix::{fcntl::AtFlags, unistd::execveat};
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::{Read, Seek, SeekFrom};
#[cfg(target_os = "linux")]
use std::os::unix::{fs::MetadataExt, io::AsRawFd};
#[cfg(target_os = "linux")]
use std::path::Path;

pub const STATIC_FD_EXEC_POLICY_SCHEMA_V1: &str = "symthaea.assurance.static-fd-exec-policy.v1";
pub const STATIC_FD_EXEC_REQUEST_SCHEMA_V1: &str = "symthaea.assurance.static-fd-exec-request.v1";
pub const STATIC_FD_EXEC_PREPARATION_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.static-fd-exec-preparation-report.v1";
pub const STATIC_FD_EXEC_FAILURE_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.static-fd-exec-failure-report.v1";

const POLICY_DOMAIN: &[u8] = b"symthaea.assurance.static-fd-exec-policy.digest.v1\0";
const REQUEST_DOMAIN: &[u8] = b"symthaea.assurance.static-fd-exec-request.digest.v1\0";
const FD_DOMAIN: &[u8] = b"symthaea.assurance.static-fd-exec-fd-identity.digest.v1\0";
const ARGV_DOMAIN: &[u8] = b"symthaea.assurance.static-fd-exec-argv.digest.v1\0";
const ENV_DOMAIN: &[u8] = b"symthaea.assurance.static-fd-exec-environment.digest.v1\0";
const PLAN_DOMAIN: &[u8] = b"symthaea.assurance.static-fd-exec-plan.digest.v1\0";
const PREP_REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.static-fd-exec-preparation-report.digest.v1\0";
const FAILURE_REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.static-fd-exec-failure-report.digest.v1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;
const HARD_MAX_ARGS: u32 = 4096;
const HARD_MAX_ENV: u32 = 4096;
const HARD_MAX_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticFdExecPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_static_profile_policy_digest: String,
    pub expected_nix_binding_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub expected_argv0: String,
    pub allowed_environment_keys: Vec<String>,
    pub required_environment_keys: Vec<String>,
    pub max_arguments: u32,
    pub max_environment_entries: u32,
    pub max_total_argument_bytes: u64,
    pub max_total_environment_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl StaticFdExecPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == STATIC_FD_EXEC_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && digest(&self.expected_static_profile_policy_digest)
            && digest(&self.expected_nix_binding_policy_digest)
            && digest(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
            && argument(&self.expected_argv0)
            && self.allowed_environment_keys.len() <= HARD_MAX_ENV as usize
            && self.required_environment_keys.len() <= HARD_MAX_ENV as usize
            && self.allowed_environment_keys.iter().all(|v| env_key(v))
            && self.required_environment_keys.iter().all(|v| env_key(v))
            && unique(&self.allowed_environment_keys)
            && unique(&self.required_environment_keys)
            && self
                .required_environment_keys
                .iter()
                .all(|v| self.allowed_environment_keys.contains(v))
            && (1..=HARD_MAX_ARGS).contains(&self.max_arguments)
            && self.max_environment_entries <= HARD_MAX_ENV
            && (1..=HARD_MAX_BYTES).contains(&self.max_total_argument_bytes)
            && self.max_total_environment_bytes <= HARD_MAX_BYTES
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() { return None; }
        let mut h = blake3::Hasher::new();
        h.update(POLICY_DOMAIN);
        for v in [
            self.schema_version.as_str(), self.policy_id.as_str(),
            self.expected_static_profile_policy_digest.as_str(),
            self.expected_nix_binding_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(), self.expected_backend_id.as_str(),
            self.expected_argv0.as_str(),
        ] { field(&mut h, v); }
        sorted(&mut h, &self.allowed_environment_keys);
        sorted(&mut h, &self.required_environment_keys);
        h.update(&self.max_arguments.to_le_bytes());
        h.update(&self.max_environment_entries.to_le_bytes());
        h.update(&self.max_total_argument_bytes.to_le_bytes());
        h.update(&self.max_total_environment_bytes.to_le_bytes());
        sorted(&mut h, &self.evidence_refs);
        Some(format!("blake3:{}", h.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticFdExecRequest {
    pub schema_version: String,
    pub argv: Vec<String>,
    pub environment: Vec<String>,
    pub launch_nonce_blake3_hex: String,
}

impl StaticFdExecRequest {
    pub fn validate_against(&self, policy: &StaticFdExecPolicy) -> bool {
        if self.schema_version != STATIC_FD_EXEC_REQUEST_SCHEMA_V1
            || !policy.validate()
            || self.argv.is_empty()
            || self.argv.len() > policy.max_arguments as usize
            || self.argv.first() != Some(&policy.expected_argv0)
            || !self.argv.iter().all(|v| argument(v))
            || self.environment.len() > policy.max_environment_entries as usize
            || !lower_hex(&self.launch_nonce_blake3_hex, 64)
        { return false; }
        if total_bytes(&self.argv).is_none_or(|n| n > policy.max_total_argument_bytes) { return false; }
        let Some(env) = parse_env(&self.environment) else { return false; };
        if total_bytes(&self.environment).is_none_or(|n| n > policy.max_total_environment_bytes) { return false; }
        if !env.keys().all(|k| policy.allowed_environment_keys.contains(k)) { return false; }
        policy.required_environment_keys.iter().all(|k| env.contains_key(k))
    }

    pub fn canonical_digest(&self, policy: &StaticFdExecPolicy) -> Option<String> {
        if !self.validate_against(policy) { return None; }
        let env = canonical_env(&self.environment)?;
        let mut h = blake3::Hasher::new();
        h.update(REQUEST_DOMAIN);
        field(&mut h, &argv_digest(&self.argv));
        field(&mut h, &env_digest(&env));
        field(&mut h, &self.launch_nonce_blake3_hex);
        Some(format!("blake3:{}", h.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaticFdExecPreparationIssue {
    InvalidPolicy, InvalidRequest, StaticProfilePolicyMismatch, NixBindingPolicyMismatch,
    NixBindingQualificationMismatch, RuntimePolicyMismatch, RuntimeVerifierMismatch,
    BackendMismatch, ExecutableIdentityMismatch, ExecutableUnavailable(String),
    CanonicalPathMismatch, RetainedFdMetadataMismatch, RetainedFdDigestMismatch,
    ArgumentEncodingInvalid, EnvironmentEncodingInvalid,
}
impl StaticFdExecPreparationIssue {
    fn code(&self) -> String { match self {
        Self::InvalidPolicy => "invalid-policy".into(), Self::InvalidRequest => "invalid-request".into(),
        Self::StaticProfilePolicyMismatch => "static-profile-policy-mismatch".into(),
        Self::NixBindingPolicyMismatch => "nix-binding-policy-mismatch".into(),
        Self::NixBindingQualificationMismatch => "nix-binding-qualification-mismatch".into(),
        Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
        Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
        Self::BackendMismatch => "backend-mismatch".into(),
        Self::ExecutableIdentityMismatch => "executable-identity-mismatch".into(),
        Self::ExecutableUnavailable(v) => format!("executable-unavailable:{v}"),
        Self::CanonicalPathMismatch => "canonical-path-mismatch".into(),
        Self::RetainedFdMetadataMismatch => "retained-fd-metadata-mismatch".into(),
        Self::RetainedFdDigestMismatch => "retained-fd-digest-mismatch".into(),
        Self::ArgumentEncodingInvalid => "argument-encoding-invalid".into(),
        Self::EnvironmentEncodingInvalid => "environment-encoding-invalid".into(),
    }}
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetainedExecutableFdIdentity {
    pub canonical_path: String, pub executable_blake3: String,
    pub device: u64, pub inode: u64, pub file_size: u64,
    pub mode: u32, pub uid: u32, pub gid: u32,
    pub mtime_seconds: i64, pub mtime_nanoseconds: i64,
    pub identity_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticFdExecPreparationReport {
    pub schema_version: String, pub policy_id: String,
    pub policy_digest: Option<String>, pub request_digest: Option<String>,
    pub static_profile_policy_digest: String, pub static_profile_qualification_digest: String,
    pub nix_binding_policy_digest: String, pub nix_binding_qualification_digest: String,
    pub runtime_policy_digest: String, pub runtime_verifier_ref: String, pub backend_id: String,
    pub executable_path: String, pub executable_digest: String, pub elf_profile_digest: String,
    pub fd_identity_digest: Option<String>, pub argv_digest: Option<String>,
    pub environment_digest: Option<String>, pub launch_nonce_blake3_hex: String,
    pub plan_digest: Option<String>, pub issues: Vec<StaticFdExecPreparationIssue>,
}
impl StaticFdExecPreparationReport {
    pub fn qualified(&self) -> bool { self.issues.is_empty() && self.plan_digest.is_some() }
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new(); h.update(PREP_REPORT_DOMAIN);
        for v in [self.schema_version.as_str(), self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"), self.request_digest.as_deref().unwrap_or("-"),
            self.static_profile_policy_digest.as_str(), self.static_profile_qualification_digest.as_str(),
            self.nix_binding_policy_digest.as_str(), self.nix_binding_qualification_digest.as_str(),
            self.runtime_policy_digest.as_str(), self.runtime_verifier_ref.as_str(), self.backend_id.as_str(),
            self.executable_path.as_str(), self.executable_digest.as_str(), self.elf_profile_digest.as_str(),
            self.fd_identity_digest.as_deref().unwrap_or("-"), self.argv_digest.as_deref().unwrap_or("-"),
            self.environment_digest.as_deref().unwrap_or("-"), self.launch_nonce_blake3_hex.as_str(),
            self.plan_digest.as_deref().unwrap_or("-")] { field(&mut h, v); }
        h.update(&(self.issues.len() as u64).to_le_bytes());
        for i in &self.issues { field(&mut h, &i.code()); }
        format!("blake3:{}", h.finalize().to_hex())
    }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[cfg(target_os = "linux")]
pub struct PreparedStaticFdExec {
    executable: File, plan_digest: String, policy_digest: String, request_digest: String,
    static_profile_qualification_digest: String, nix_binding_qualification_digest: String,
    runtime_policy_digest: String, runtime_verifier_ref: String, backend_id: String,
    executable_path: String, executable_digest: String, elf_profile_digest: String,
    fd_identity: RetainedExecutableFdIdentity, argv: Vec<CString>, environment: Vec<CString>,
    argv_digest: String, environment_digest: String, launch_nonce_blake3_hex: String,
}
#[cfg(target_os = "linux")]
impl std::fmt::Debug for PreparedStaticFdExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedStaticFdExec").field("plan_digest", &self.plan_digest)
            .field("fd_identity", &self.fd_identity).field("executable_path", &self.executable_path)
            .field("argv_digest", &self.argv_digest).field("environment_digest", &self.environment_digest)
            .finish_non_exhaustive()
    }
}
#[cfg(target_os = "linux")]
impl PreparedStaticFdExec {
    pub fn plan_digest(&self) -> &str { &self.plan_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn request_digest(&self) -> &str { &self.request_digest }
    pub fn static_profile_qualification_digest(&self) -> &str { &self.static_profile_qualification_digest }
    pub fn nix_binding_qualification_digest(&self) -> &str { &self.nix_binding_qualification_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn executable_path(&self) -> &str { &self.executable_path }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn elf_profile_digest(&self) -> &str { &self.elf_profile_digest }
    pub fn fd_identity(&self) -> &RetainedExecutableFdIdentity { &self.fd_identity }
    pub fn argv_digest(&self) -> &str { &self.argv_digest }
    pub fn environment_digest(&self) -> &str { &self.environment_digest }
    pub fn launch_nonce_blake3_hex(&self) -> &str { &self.launch_nonce_blake3_hex }
    pub const fn exact_retained_fd_content_bound_to_static_profile(&self) -> bool { true }
    pub const fn main_executable_path_lookup_required_at_exec(&self) -> bool { false }
    pub const fn external_elf_interpreter_declared(&self) -> bool { false }
    pub const fn declared_dynamic_dependencies_present(&self) -> bool { false }
    pub const fn ambient_environment_inherited(&self) -> bool { false }
    pub const fn launch_success_established(&self) -> bool { false }
    pub const fn post_exec_identity_established(&self) -> bool { false }
    pub const fn mapping_continuity_since_exec_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[cfg(target_os = "linux")]
pub struct StaticFdExecPreparation { pub report: StaticFdExecPreparationReport, prepared: PreparedStaticFdExec }
#[cfg(target_os = "linux")]
impl StaticFdExecPreparation {
    pub fn prepared(&self) -> &PreparedStaticFdExec { &self.prepared }
    pub fn into_prepared(self) -> PreparedStaticFdExec { self.prepared }
}

#[cfg(target_os = "linux")]
pub fn prepare_static_fd_exec(
    policy: &StaticFdExecPolicy, request: &StaticFdExecRequest,
    profile: &StaticElfLaunchProfile, bound: &NixBoundInProcessRuntimePolicy,
) -> Result<StaticFdExecPreparation, StaticFdExecPreparationReport> {
    let pd = policy.canonical_digest(); let rd = request.canonical_digest(policy);
    let mut report = base_report(policy, pd.clone(), request, rd.clone(), profile, bound);
    if !policy.validate() { report.issues.push(StaticFdExecPreparationIssue::InvalidPolicy); return Err(report); }
    if rd.is_none() { report.issues.push(StaticFdExecPreparationIssue::InvalidRequest); return Err(report); }
    if profile.policy_digest() != policy.expected_static_profile_policy_digest { report.issues.push(StaticFdExecPreparationIssue::StaticProfilePolicyMismatch); }
    if bound.policy_digest() != policy.expected_nix_binding_policy_digest { report.issues.push(StaticFdExecPreparationIssue::NixBindingPolicyMismatch); }
    if profile.nix_binding_qualification_digest() != bound.qualification_digest() { report.issues.push(StaticFdExecPreparationIssue::NixBindingQualificationMismatch); }
    if profile.runtime_policy_digest() != policy.expected_runtime_policy_digest || bound.runtime_policy_digest() != policy.expected_runtime_policy_digest { report.issues.push(StaticFdExecPreparationIssue::RuntimePolicyMismatch); }
    if profile.runtime_verifier_ref() != policy.expected_runtime_verifier_ref || bound.runtime_verifier_ref() != policy.expected_runtime_verifier_ref { report.issues.push(StaticFdExecPreparationIssue::RuntimeVerifierMismatch); }
    if profile.backend_id() != policy.expected_backend_id || bound.backend_id() != policy.expected_backend_id { report.issues.push(StaticFdExecPreparationIssue::BackendMismatch); }
    if profile.executable_path() != bound.executable_path() || profile.executable_digest() != bound.executable_digest() { report.issues.push(StaticFdExecPreparationIssue::ExecutableIdentityMismatch); }
    if !report.issues.is_empty() { return Err(report); }

    let canonical = match std::fs::canonicalize(profile.executable_path()) { Ok(v) => v, Err(e) => { report.issues.push(StaticFdExecPreparationIssue::ExecutableUnavailable(e.to_string())); return Err(report); } };
    if canonical != Path::new(profile.executable_path()) { report.issues.push(StaticFdExecPreparationIssue::CanonicalPathMismatch); return Err(report); }
    let mut executable = match File::open(profile.executable_path()) { Ok(v) => v, Err(e) => { report.issues.push(StaticFdExecPreparationIssue::ExecutableUnavailable(e.to_string())); return Err(report); } };
    let fd_identity = match observe_fd(&mut executable, profile) { Ok(v) => v, Err(i) => { report.issues.push(i); return Err(report); } };
    let argv = match cstrings(&request.argv) { Some(v) => v, None => { report.issues.push(StaticFdExecPreparationIssue::ArgumentEncodingInvalid); return Err(report); } };
    let canonical_environment = match canonical_env(&request.environment) { Some(v) => v, None => { report.issues.push(StaticFdExecPreparationIssue::EnvironmentEncodingInvalid); return Err(report); } };
    let environment = match cstrings(&canonical_environment) { Some(v) => v, None => { report.issues.push(StaticFdExecPreparationIssue::EnvironmentEncodingInvalid); return Err(report); } };
    let ad = argv_digest(&request.argv); let ed = env_digest(&canonical_environment);
    let pd = pd.expect("valid policy"); let rd = rd.expect("valid request");
    let plan = plan_digest(&pd, &rd, profile.qualification_digest(), bound.qualification_digest(), &fd_identity.identity_digest, &ad, &ed, &request.launch_nonce_blake3_hex);
    report.fd_identity_digest = Some(fd_identity.identity_digest.clone()); report.argv_digest = Some(ad.clone()); report.environment_digest = Some(ed.clone()); report.plan_digest = Some(plan.clone());
    Ok(StaticFdExecPreparation { report, prepared: PreparedStaticFdExec {
        executable, plan_digest: plan, policy_digest: pd, request_digest: rd,
        static_profile_qualification_digest: profile.qualification_digest().into(), nix_binding_qualification_digest: bound.qualification_digest().into(),
        runtime_policy_digest: bound.runtime_policy_digest().into(), runtime_verifier_ref: bound.runtime_verifier_ref().into(), backend_id: bound.backend_id().into(),
        executable_path: profile.executable_path().into(), executable_digest: profile.executable_digest().into(), elf_profile_digest: profile.elf_profile_digest().into(),
        fd_identity, argv, environment, argv_digest: ad, environment_digest: ed, launch_nonce_blake3_hex: request.launch_nonce_blake3_hex.clone(),
    }})
}

#[cfg(target_os = "linux")]
fn observe_fd(file: &mut File, profile: &StaticElfLaunchProfile) -> Result<RetainedExecutableFdIdentity, StaticFdExecPreparationIssue> {
    let before = file.metadata().map_err(|e| StaticFdExecPreparationIssue::ExecutableUnavailable(e.to_string()))?;
    if !before.is_file() || before.uid() != 0 || before.mode() & 0o222 != 0 || before.mode() & 0o111 == 0 || before.len() != profile.executable_size() { return Err(StaticFdExecPreparationIssue::RetainedFdMetadataMismatch); }
    let actual = hash_file(file)?;
    let after = file.metadata().map_err(|e| StaticFdExecPreparationIssue::ExecutableUnavailable(e.to_string()))?;
    if !same_meta(&before, &after) { return Err(StaticFdExecPreparationIssue::RetainedFdMetadataMismatch); }
    if actual != profile.executable_digest() { return Err(StaticFdExecPreparationIssue::RetainedFdDigestMismatch); }
    let id = fd_digest(profile.executable_path(), &actual, &before);
    Ok(RetainedExecutableFdIdentity { canonical_path: profile.executable_path().into(), executable_blake3: actual,
        device: before.dev(), inode: before.ino(), file_size: before.len(), mode: before.mode(), uid: before.uid(), gid: before.gid(),
        mtime_seconds: before.mtime(), mtime_nanoseconds: before.mtime_nsec(), identity_digest: id })
}

#[cfg(target_os = "linux")]
fn hash_file(file: &mut File) -> Result<String, StaticFdExecPreparationIssue> {
    file.seek(SeekFrom::Start(0)).map_err(|e| StaticFdExecPreparationIssue::ExecutableUnavailable(e.to_string()))?;
    let mut h = blake3::Hasher::new(); let mut buf = [0u8; 64 * 1024];
    loop { let n = file.read(&mut buf).map_err(|e| StaticFdExecPreparationIssue::ExecutableUnavailable(e.to_string()))?; if n == 0 { break; } h.update(&buf[..n]); }
    Ok(format!("blake3:{}", h.finalize().to_hex()))
}

#[cfg(target_os = "linux")]
fn same_meta(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    a.dev()==b.dev() && a.ino()==b.ino() && a.len()==b.len() && a.mode()==b.mode()
        && a.uid()==b.uid() && a.gid()==b.gid() && a.mtime()==b.mtime() && a.mtime_nsec()==b.mtime_nsec()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaticFdExecFailureStage { PreExecRecheck, Execveat }
impl StaticFdExecFailureStage { fn code(self) -> &'static str { match self { Self::PreExecRecheck => "pre-exec-recheck", Self::Execveat => "execveat" } } }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticFdExecFailureReport {
    pub schema_version: String, pub stage: StaticFdExecFailureStage,
    pub plan_digest: String, pub policy_digest: String, pub request_digest: String,
    pub static_profile_qualification_digest: String, pub nix_binding_qualification_digest: String,
    pub fd_identity_digest: String, pub argv_digest: String, pub environment_digest: String,
    pub launch_nonce_blake3_hex: String, pub error: String, pub report_digest: String,
}
impl StaticFdExecFailureReport {
    pub const fn execveat_attempted(&self) -> bool { matches!(self.stage, StaticFdExecFailureStage::Execveat) }
    pub const fn execveat_succeeded(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[cfg(target_os = "linux")]
pub fn execute_prepared_static_fd_exec(mut p: PreparedStaticFdExec) -> Result<Infallible, StaticFdExecFailureReport> {
    if let Err(issue) = recheck(&mut p) { return Err(failure(&p, StaticFdExecFailureStage::PreExecRecheck, issue.code())); }
    let empty = CString::new(Vec::<u8>::new()).expect("empty CString");
    match execveat(p.executable.as_raw_fd(), &empty, &p.argv, &p.environment, AtFlags::AT_EMPTY_PATH) {
        Ok(never) => match never {},
        Err(error) => Err(failure(&p, StaticFdExecFailureStage::Execveat, error.to_string())),
    }
}

#[cfg(target_os = "linux")]
fn recheck(p: &mut PreparedStaticFdExec) -> Result<(), StaticFdExecPreparationIssue> {
    let before = p.executable.metadata().map_err(|e| StaticFdExecPreparationIssue::ExecutableUnavailable(e.to_string()))?;
    if before.dev()!=p.fd_identity.device || before.ino()!=p.fd_identity.inode || before.len()!=p.fd_identity.file_size
        || before.mode()!=p.fd_identity.mode || before.uid()!=p.fd_identity.uid || before.gid()!=p.fd_identity.gid
        || before.mtime()!=p.fd_identity.mtime_seconds || before.mtime_nsec()!=p.fd_identity.mtime_nanoseconds
    { return Err(StaticFdExecPreparationIssue::RetainedFdMetadataMismatch); }
    let actual = hash_file(&mut p.executable)?;
    let after = p.executable.metadata().map_err(|e| StaticFdExecPreparationIssue::ExecutableUnavailable(e.to_string()))?;
    if !same_meta(&before, &after) { return Err(StaticFdExecPreparationIssue::RetainedFdMetadataMismatch); }
    if actual != p.executable_digest { return Err(StaticFdExecPreparationIssue::RetainedFdDigestMismatch); }
    Ok(())
}

#[cfg(target_os = "linux")]
fn failure(p: &PreparedStaticFdExec, stage: StaticFdExecFailureStage, error: String) -> StaticFdExecFailureReport {
    let mut r = StaticFdExecFailureReport { schema_version: STATIC_FD_EXEC_FAILURE_REPORT_SCHEMA_V1.into(), stage,
        plan_digest: p.plan_digest.clone(), policy_digest: p.policy_digest.clone(), request_digest: p.request_digest.clone(),
        static_profile_qualification_digest: p.static_profile_qualification_digest.clone(), nix_binding_qualification_digest: p.nix_binding_qualification_digest.clone(),
        fd_identity_digest: p.fd_identity.identity_digest.clone(), argv_digest: p.argv_digest.clone(), environment_digest: p.environment_digest.clone(),
        launch_nonce_blake3_hex: p.launch_nonce_blake3_hex.clone(), error, report_digest: String::new() };
    r.report_digest = failure_digest(&r); r
}
fn failure_digest(r: &StaticFdExecFailureReport) -> String {
    let mut h=blake3::Hasher::new(); h.update(FAILURE_REPORT_DOMAIN); field(&mut h,r.stage.code());
    for v in [r.schema_version.as_str(),r.plan_digest.as_str(),r.policy_digest.as_str(),r.request_digest.as_str(),r.static_profile_qualification_digest.as_str(),r.nix_binding_qualification_digest.as_str(),r.fd_identity_digest.as_str(),r.argv_digest.as_str(),r.environment_digest.as_str(),r.launch_nonce_blake3_hex.as_str(),r.error.as_str()] { field(&mut h,v); }
    format!("blake3:{}",h.finalize().to_hex())
}

fn base_report(policy:&StaticFdExecPolicy,pd:Option<String>,request:&StaticFdExecRequest,rd:Option<String>,profile:&StaticElfLaunchProfile,bound:&NixBoundInProcessRuntimePolicy)->StaticFdExecPreparationReport{
    StaticFdExecPreparationReport{schema_version:STATIC_FD_EXEC_PREPARATION_REPORT_SCHEMA_V1.into(),policy_id:policy.policy_id.clone(),policy_digest:pd,request_digest:rd,
        static_profile_policy_digest:profile.policy_digest().into(),static_profile_qualification_digest:profile.qualification_digest().into(),nix_binding_policy_digest:bound.policy_digest().into(),nix_binding_qualification_digest:bound.qualification_digest().into(),runtime_policy_digest:bound.runtime_policy_digest().into(),runtime_verifier_ref:bound.runtime_verifier_ref().into(),backend_id:bound.backend_id().into(),executable_path:profile.executable_path().into(),executable_digest:profile.executable_digest().into(),elf_profile_digest:profile.elf_profile_digest().into(),fd_identity_digest:None,argv_digest:None,environment_digest:None,launch_nonce_blake3_hex:request.launch_nonce_blake3_hex.clone(),plan_digest:None,issues:Vec::new()}
}

fn fd_digest(path:&str,d:&str,m:&std::fs::Metadata)->String{
    #[cfg(target_os="linux")]{let mut h=blake3::Hasher::new();h.update(FD_DOMAIN);field(&mut h,path);field(&mut h,d);for v in[m.dev(),m.ino(),m.len()]{h.update(&v.to_le_bytes());}for v in[m.mode(),m.uid(),m.gid()]{h.update(&v.to_le_bytes());}h.update(&m.mtime().to_le_bytes());h.update(&m.mtime_nsec().to_le_bytes());format!("blake3:{}",h.finalize().to_hex())}
    #[cfg(not(target_os="linux"))]{let _=(path,d,m);String::new()}
}
fn plan_digest(pd:&str,rd:&str,pq:&str,bq:&str,fd:&str,ad:&str,ed:&str,nonce:&str)->String{let mut h=blake3::Hasher::new();h.update(PLAN_DOMAIN);for v in[pd,rd,pq,bq,fd,ad,ed,nonce]{field(&mut h,v);}format!("blake3:{}",h.finalize().to_hex())}
fn argv_digest(v:&[String])->String{let mut h=blake3::Hasher::new();h.update(ARGV_DOMAIN);h.update(&(v.len()as u64).to_le_bytes());for x in v{field(&mut h,x);}format!("blake3:{}",h.finalize().to_hex())}
fn env_digest(v:&[String])->String{let mut h=blake3::Hasher::new();h.update(ENV_DOMAIN);h.update(&(v.len()as u64).to_le_bytes());for x in v{field(&mut h,x);}format!("blake3:{}",h.finalize().to_hex())}
fn cstrings(v:&[String])->Option<Vec<CString>>{v.iter().map(|x|CString::new(x.as_bytes()).ok()).collect()}
fn canonical_env(v:&[String])->Option<Vec<String>>{Some(parse_env(v)?.into_iter().map(|(k,val)|format!("{k}={val}")).collect())}
fn parse_env(v:&[String])->Option<BTreeMap<String,String>>{let mut out=BTreeMap::new();for e in v{let(k,val)=e.split_once('=')?;if !env_key(k)||e.len()>MAX_TEXT||e.bytes().any(|b|b==0||b.is_ascii_control())||out.insert(k.into(),val.into()).is_some(){return None;}}Some(out)}
fn env_key(v:&str)->bool{let mut b=v.bytes();let Some(f)=b.next()else{return false;};(f.is_ascii_alphabetic()||f==b'_')&&b.all(|x|x.is_ascii_alphanumeric()||x==b'_')}
fn argument(v:&str)->bool{!v.is_empty()&&v.len()<=MAX_TEXT&&!v.bytes().any(|b|b==0||b.is_ascii_control())}
fn total_bytes(v:&[String])->Option<u64>{v.iter().try_fold(0u64,|s,x|s.checked_add(x.len()as u64+1))}
fn canonical_text(v:&str)->bool{!v.is_empty()&&v==v.trim()&&v.len()<=MAX_TEXT&&!v.chars().any(char::is_control)}
fn digest(v:&str)->bool{v.strip_prefix("blake3:").is_some_and(|x|lower_hex(x,64))}
fn lower_hex(v:&str,n:usize)->bool{v.len()==n&&v.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b))}
fn valid_refs(v:&[String])->bool{v.len()<=MAX_REFS&&v.iter().all(|x|canonical_text(x))&&unique(v)}
fn unique(v:&[String])->bool{let mut s=BTreeSet::new();v.iter().all(|x|s.insert(x.as_str()))}
fn field(h:&mut blake3::Hasher,v:&str){h.update(&(v.len()as u64).to_le_bytes());h.update(v.as_bytes());}
fn sorted(h:&mut blake3::Hasher,v:&[String]){let mut x=v.to_vec();x.sort();h.update(&(x.len()as u64).to_le_bytes());for s in x{field(h,&s);}}

#[cfg(test)]
mod tests{
use super::*;
fn d(x:&str)->String{format!("blake3:{}",blake3::hash(x.as_bytes()).to_hex())}
fn policy()->StaticFdExecPolicy{StaticFdExecPolicy{schema_version:STATIC_FD_EXEC_POLICY_SCHEMA_V1.into(),policy_id:"policy:exec:1".into(),expected_static_profile_policy_digest:d("profile-policy"),expected_nix_binding_policy_digest:d("binding-policy"),expected_runtime_policy_digest:d("runtime-policy"),expected_runtime_verifier_ref:"verifier:host".into(),expected_backend_id:"backend:in-process".into(),expected_argv0:"verifier".into(),allowed_environment_keys:vec!["LANG".into(),"TZ".into()],required_environment_keys:vec!["LANG".into()],max_arguments:8,max_environment_entries:8,max_total_argument_bytes:4096,max_total_environment_bytes:4096,evidence_refs:vec!["review:a".into(),"review:b".into()]}}
fn request()->StaticFdExecRequest{StaticFdExecRequest{schema_version:STATIC_FD_EXEC_REQUEST_SCHEMA_V1.into(),argv:vec!["verifier".into(),"--serve".into()],environment:vec!["TZ=UTC0".into(),"LANG=C".into()],launch_nonce_blake3_hex:"11".repeat(32)}}
#[test]fn policy_set_order_nonsemantic(){let a=policy();let mut b=a.clone();b.evidence_refs.reverse();b.allowed_environment_keys.reverse();assert_eq!(a.canonical_digest(),b.canonical_digest());}
#[test]fn env_order_nonsemantic_argv_order_semantic(){let p=policy();let a=request();let mut b=a.clone();b.environment.reverse();assert_eq!(a.canonical_digest(&p),b.canonical_digest(&p));b=a.clone();b.argv.swap(0,1);assert!(b.canonical_digest(&p).is_none());}
#[test]fn unknown_duplicate_and_missing_required_env_rejected(){let p=policy();let mut a=request();a.environment.push("PATH=/bin".into());assert!(!a.validate_against(&p));let mut b=request();b.environment.push("LANG=en".into());assert!(!b.validate_against(&p));let mut c=request();c.environment.retain(|x|!x.starts_with("LANG="));assert!(!c.validate_against(&p));}
#[test]fn nonce_changes_request_identity(){let p=policy();let a=request();let mut b=a.clone();b.launch_nonce_blake3_hex="22".repeat(32);assert_ne!(a.canonical_digest(&p),b.canonical_digest(&p));}
#[test]fn failure_stage_controls_attempt_claim(){let mk=|stage|StaticFdExecFailureReport{schema_version:STATIC_FD_EXEC_FAILURE_REPORT_SCHEMA_V1.into(),stage,plan_digest:d("p"),policy_digest:d("pol"),request_digest:d("r"),static_profile_qualification_digest:d("s"),nix_binding_qualification_digest:d("b"),fd_identity_digest:d("fd"),argv_digest:d("a"),environment_digest:d("e"),launch_nonce_blake3_hex:"11".repeat(32),error:"x".into(),report_digest:String::new()};assert!(!mk(StaticFdExecFailureStage::PreExecRecheck).execveat_attempted());assert!(mk(StaticFdExecFailureStage::Execveat).execveat_attempted());}
}
