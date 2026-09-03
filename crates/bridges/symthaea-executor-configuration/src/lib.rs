// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only systemd/Nix executor configuration measurement.
//!
//! A witnessed executable identity is not enough if the same binary can be
//! launched under materially different service policy. V1 derives a canonical
//! configuration commitment from the *current process's own systemd unit* and
//! the unit content systemd currently exposes.
//!
//! The production measurement path accepts a reviewed policy containing the
//! expected executor unit; callers do not choose an arbitrary unit at
//! measurement time. It discovers the current unit from `/proc/self/cgroup`,
//! reads selected security properties in one `systemctl show`, hashes the full
//! `systemctl cat` output without retaining it, resolves the fragment path, and
//! repeats the entire measurement. A changed second snapshot fails closed.
//!
//! This crate is read-only and creates no authority. V1 trusts the running
//! kernel/systemd manager view. IMA/fs-verity/TPM evidence can strengthen that
//! root later without changing the canonical configuration digest.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use thiserror::Error;

pub const EXECUTOR_CONFIGURATION_SCHEMA_VERSION: u16 = 1;
pub const MAX_SYSTEMD_UNIT_BYTES: usize = 512;
const POLICY_DOMAIN: &[u8] = b"symthaea.executor-configuration.policy.v1\0";
const CONFIG_DOMAIN: &[u8] = b"symthaea.executor-configuration.v1\0";

const SHOW_PROPERTIES: &[&str] = &[
    "Id",
    "FragmentPath",
    "ExecStart",
    "User",
    "Group",
    "DynamicUser",
    "NoNewPrivileges",
    "ProtectSystem",
    "ProtectHome",
    "PrivateTmp",
    "PrivateDevices",
    "PrivateNetwork",
    "MemoryDenyWriteExecute",
    "LockPersonality",
    "ProtectKernelTunables",
    "ProtectKernelModules",
    "ProtectControlGroups",
    "RestrictSUIDSGID",
    "RestrictRealtime",
    "CapabilityBoundingSet",
    "AmbientCapabilities",
    "SystemCallFilter",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorConfigurationPolicyV1 {
    pub schema_version: u16,
    /// Exact service unit expected to contain the executor process.
    pub expected_unit: String,
    /// Require the resolved unit fragment to live under `/nix/store`.
    pub require_nix_store_fragment: bool,
    pub require_no_new_privileges: bool,
    pub require_protect_system_strict: bool,
    pub require_protect_home: bool,
    pub require_private_tmp: bool,
    pub require_private_devices: bool,
    pub require_memory_deny_write_execute: bool,
    pub require_lock_personality: bool,
    pub require_protect_kernel_tunables: bool,
    pub require_protect_kernel_modules: bool,
    pub require_protect_control_groups: bool,
    pub require_restrict_suid_sgid: bool,
    pub require_restrict_realtime: bool,
}

impl ExecutorConfigurationPolicyV1 {
    pub fn validate(&self) -> Result<(), ExecutorConfigurationError> {
        if self.schema_version != EXECUTOR_CONFIGURATION_SCHEMA_VERSION
            || self.expected_unit.is_empty()
            || self.expected_unit.len() > MAX_SYSTEMD_UNIT_BYTES
            || !self.expected_unit.ends_with(".service")
            || self.expected_unit.bytes().any(|b| b.is_ascii_control())
        {
            return Err(ExecutorConfigurationError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, ExecutorConfigurationError> {
        self.validate()?;
        let mut t = Transcript::new(POLICY_DOMAIN);
        t.u16(self.schema_version);
        t.bytes(self.expected_unit.as_bytes())?;
        for flag in [
            self.require_nix_store_fragment,
            self.require_no_new_privileges,
            self.require_protect_system_strict,
            self.require_protect_home,
            self.require_private_tmp,
            self.require_private_devices,
            self.require_memory_deny_write_execute,
            self.require_lock_personality,
            self.require_protect_kernel_tunables,
            self.require_protect_kernel_modules,
            self.require_protect_control_groups,
            self.require_restrict_suid_sgid,
            self.require_restrict_realtime,
        ] {
            t.u8(u8::from(flag));
        }
        Ok(Digest32(t.finish()))
    }
}

/// Canonical privacy-minimized identity of systemd's current executor policy.
///
/// Raw `ExecStart`, unit contents, user/group strings, capability sets and
/// syscall filters are committed by digest rather than retained in the proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorConfigurationSnapshotV1 {
    pub schema_version: u16,
    pub unit: String,
    pub fragment_path_digest: Digest32,
    pub resolved_fragment_path_digest: Digest32,
    pub fragment_in_nix_store: bool,
    pub unit_text_digest: Digest32,
    pub exec_start_digest: Digest32,
    pub user_digest: Digest32,
    pub group_digest: Digest32,
    pub capability_bounding_set_digest: Digest32,
    pub ambient_capabilities_digest: Digest32,
    pub system_call_filter_digest: Digest32,
    pub dynamic_user: bool,
    pub no_new_privileges: bool,
    pub protect_system: ProtectSystemV1,
    pub protect_home: ProtectHomeV1,
    pub private_tmp: bool,
    pub private_devices: bool,
    pub private_network: bool,
    pub memory_deny_write_execute: bool,
    pub lock_personality: bool,
    pub protect_kernel_tunables: bool,
    pub protect_kernel_modules: bool,
    pub protect_control_groups: bool,
    pub restrict_suid_sgid: bool,
    pub restrict_realtime: bool,
}

impl ExecutorConfigurationSnapshotV1 {
    pub fn digest(&self) -> Result<Digest32, ExecutorConfigurationError> {
        self.validate()?;
        let mut t = Transcript::new(CONFIG_DOMAIN);
        t.u16(self.schema_version);
        t.bytes(self.unit.as_bytes())?;
        for digest in [
            self.fragment_path_digest,
            self.resolved_fragment_path_digest,
            self.unit_text_digest,
            self.exec_start_digest,
            self.user_digest,
            self.group_digest,
            self.capability_bounding_set_digest,
            self.ambient_capabilities_digest,
            self.system_call_filter_digest,
        ] {
            t.fixed(&digest.0);
        }
        t.u8(u8::from(self.fragment_in_nix_store));
        t.u8(u8::from(self.dynamic_user));
        t.u8(u8::from(self.no_new_privileges));
        t.u8(self.protect_system as u8);
        t.u8(self.protect_home as u8);
        for flag in [
            self.private_tmp,
            self.private_devices,
            self.private_network,
            self.memory_deny_write_execute,
            self.lock_personality,
            self.protect_kernel_tunables,
            self.protect_kernel_modules,
            self.protect_control_groups,
            self.restrict_suid_sgid,
            self.restrict_realtime,
        ] {
            t.u8(u8::from(flag));
        }
        Ok(Digest32(t.finish()))
    }

    pub fn validate(&self) -> Result<(), ExecutorConfigurationError> {
        if self.schema_version != EXECUTOR_CONFIGURATION_SCHEMA_VERSION
            || self.unit.is_empty()
            || self.unit.len() > MAX_SYSTEMD_UNIT_BYTES
            || !self.unit.ends_with(".service")
        {
            return Err(ExecutorConfigurationError::InvalidSnapshot);
        }
        for digest in [
            self.fragment_path_digest,
            self.resolved_fragment_path_digest,
            self.unit_text_digest,
            self.exec_start_digest,
            self.user_digest,
            self.group_digest,
            self.capability_bounding_set_digest,
            self.ambient_capabilities_digest,
            self.system_call_filter_digest,
        ] {
            if digest.0 == [0; 32] {
                return Err(ExecutorConfigurationError::InvalidSnapshot);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ProtectSystemV1 {
    No = 0,
    Yes = 1,
    Full = 2,
    Strict = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ProtectHomeV1 {
    No = 0,
    Yes = 1,
    ReadOnly = 2,
    Tmpfs = 3,
}

/// Opaque proof produced only by the live read-only measurement path.
#[derive(Debug)]
pub struct VerifiedExecutorConfiguration {
    policy_digest: Digest32,
    snapshot: ExecutorConfigurationSnapshotV1,
}

impl VerifiedExecutorConfiguration {
    pub fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }

    pub fn snapshot(&self) -> &ExecutorConfigurationSnapshotV1 {
        &self.snapshot
    }

    pub fn configuration_digest(&self) -> Result<Digest32, ExecutorConfigurationError> {
        self.snapshot.digest()
    }

    /// Re-run the entire measurement and require exact equality.
    pub fn require_current(
        &self,
        policy: &ExecutorConfigurationPolicyV1,
    ) -> Result<(), ExecutorConfigurationError> {
        if policy.digest()? != self.policy_digest {
            return Err(ExecutorConfigurationError::PolicyMismatch);
        }
        let current = measure_current_executor_configuration(policy)?;
        if current.snapshot != self.snapshot {
            return Err(ExecutorConfigurationError::ConfigurationChanged);
        }
        Ok(())
    }
}

/// Measure the current process's reviewed systemd service configuration twice.
///
/// The double snapshot turns ordinary daemon-reload/file races into a fail-
/// closed error instead of silently binding a mixed configuration.
pub fn measure_current_executor_configuration(
    policy: &ExecutorConfigurationPolicyV1,
) -> Result<VerifiedExecutorConfiguration, ExecutorConfigurationError> {
    policy.validate()?;
    let unit = discover_systemd_service_for_current_process()?;
    if unit != policy.expected_unit {
        return Err(ExecutorConfigurationError::UnexpectedExecutorUnit {
            expected: policy.expected_unit.clone(),
            actual: unit,
        });
    }

    let first = measure_unit_once(&policy.expected_unit)?;
    let second = measure_unit_once(&policy.expected_unit)?;
    if first != second {
        return Err(ExecutorConfigurationError::ConfigurationChangedDuringMeasurement);
    }
    enforce_policy(policy, &first)?;
    Ok(VerifiedExecutorConfiguration {
        policy_digest: policy.digest()?,
        snapshot: first,
    })
}

pub fn discover_systemd_service_for_current_process() -> Result<String, ExecutorConfigurationError> {
    let cgroup = fs::read_to_string("/proc/self/cgroup")?;
    discover_systemd_service_from_cgroup(&cgroup)
}

fn discover_systemd_service_from_cgroup(cgroup: &str) -> Result<String, ExecutorConfigurationError> {
    let mut candidates = BTreeSet::new();
    for line in cgroup.lines() {
        let Some((_prefix, path)) = line.rsplit_once(':') else {
            continue;
        };
        for component in path.split('/') {
            if component.ends_with(".service")
                && !component.is_empty()
                && component.len() <= MAX_SYSTEMD_UNIT_BYTES
            {
                candidates.insert(component.to_string());
            }
        }
    }
    match candidates.len() {
        1 => Ok(candidates.into_iter().next().expect("len checked")),
        0 => Err(ExecutorConfigurationError::NoSystemdServiceForCurrentProcess),
        _ => Err(ExecutorConfigurationError::AmbiguousSystemdService),
    }
}

fn measure_unit_once(unit: &str) -> Result<ExecutorConfigurationSnapshotV1, ExecutorConfigurationError> {
    let properties_arg = format!("--property={}", SHOW_PROPERTIES.join(","));
    let show = Command::new("systemctl")
        .args(["show", unit, "--no-pager", &properties_arg])
        .output()?;
    if !show.status.success() {
        return Err(ExecutorConfigurationError::SystemctlShowFailed(
            String::from_utf8_lossy(&show.stderr).trim().to_string(),
        ));
    }
    let properties = parse_show(&show.stdout)?;
    let id = required(&properties, "Id")?;
    if id != unit {
        return Err(ExecutorConfigurationError::UnitIdentityMismatch);
    }
    let fragment = required(&properties, "FragmentPath")?;
    if fragment.is_empty() {
        return Err(ExecutorConfigurationError::MissingFragmentPath);
    }
    let fragment_path = PathBuf::from(fragment);
    let resolved_fragment = fs::canonicalize(&fragment_path)?;

    let cat = Command::new("systemctl")
        .args(["cat", unit, "--no-pager"])
        .output()?;
    if !cat.status.success() {
        return Err(ExecutorConfigurationError::SystemctlCatFailed(
            String::from_utf8_lossy(&cat.stderr).trim().to_string(),
        ));
    }
    if cat.stdout.is_empty() {
        return Err(ExecutorConfigurationError::EmptyUnitText);
    }

    Ok(ExecutorConfigurationSnapshotV1 {
        schema_version: EXECUTOR_CONFIGURATION_SCHEMA_VERSION,
        unit: unit.to_string(),
        fragment_path_digest: digest_bytes(fragment.as_bytes()),
        resolved_fragment_path_digest: digest_path(&resolved_fragment),
        fragment_in_nix_store: path_is_in_nix_store(&resolved_fragment),
        unit_text_digest: digest_bytes(&cat.stdout),
        exec_start_digest: digest_bytes(required(&properties, "ExecStart")?.as_bytes()),
        user_digest: digest_bytes(required(&properties, "User")?.as_bytes()),
        group_digest: digest_bytes(required(&properties, "Group")?.as_bytes()),
        capability_bounding_set_digest: digest_bytes(
            required(&properties, "CapabilityBoundingSet")?.as_bytes(),
        ),
        ambient_capabilities_digest: digest_bytes(
            required(&properties, "AmbientCapabilities")?.as_bytes(),
        ),
        system_call_filter_digest: digest_bytes(required(&properties, "SystemCallFilter")?.as_bytes()),
        dynamic_user: parse_bool(required(&properties, "DynamicUser")?)?,
        no_new_privileges: parse_bool(required(&properties, "NoNewPrivileges")?)?,
        protect_system: parse_protect_system(required(&properties, "ProtectSystem")?)?,
        protect_home: parse_protect_home(required(&properties, "ProtectHome")?)?,
        private_tmp: parse_bool(required(&properties, "PrivateTmp")?)?,
        private_devices: parse_bool(required(&properties, "PrivateDevices")?)?,
        private_network: parse_bool(required(&properties, "PrivateNetwork")?)?,
        memory_deny_write_execute: parse_bool(required(&properties, "MemoryDenyWriteExecute")?)?,
        lock_personality: parse_bool(required(&properties, "LockPersonality")?)?,
        protect_kernel_tunables: parse_bool(required(&properties, "ProtectKernelTunables")?)?,
        protect_kernel_modules: parse_bool(required(&properties, "ProtectKernelModules")?)?,
        protect_control_groups: parse_bool(required(&properties, "ProtectControlGroups")?)?,
        restrict_suid_sgid: parse_bool(required(&properties, "RestrictSUIDSGID")?)?,
        restrict_realtime: parse_bool(required(&properties, "RestrictRealtime")?)?,
    })
}

fn enforce_policy(
    policy: &ExecutorConfigurationPolicyV1,
    snapshot: &ExecutorConfigurationSnapshotV1,
) -> Result<(), ExecutorConfigurationError> {
    snapshot.validate()?;
    if snapshot.unit != policy.expected_unit {
        return Err(ExecutorConfigurationError::UnitIdentityMismatch);
    }
    let failed = (policy.require_nix_store_fragment && !snapshot.fragment_in_nix_store)
        || (policy.require_no_new_privileges && !snapshot.no_new_privileges)
        || (policy.require_protect_system_strict
            && snapshot.protect_system != ProtectSystemV1::Strict)
        || (policy.require_protect_home && snapshot.protect_home == ProtectHomeV1::No)
        || (policy.require_private_tmp && !snapshot.private_tmp)
        || (policy.require_private_devices && !snapshot.private_devices)
        || (policy.require_memory_deny_write_execute && !snapshot.memory_deny_write_execute)
        || (policy.require_lock_personality && !snapshot.lock_personality)
        || (policy.require_protect_kernel_tunables && !snapshot.protect_kernel_tunables)
        || (policy.require_protect_kernel_modules && !snapshot.protect_kernel_modules)
        || (policy.require_protect_control_groups && !snapshot.protect_control_groups)
        || (policy.require_restrict_suid_sgid && !snapshot.restrict_suid_sgid)
        || (policy.require_restrict_realtime && !snapshot.restrict_realtime);
    if failed {
        return Err(ExecutorConfigurationError::HardeningPolicyNotSatisfied);
    }
    Ok(())
}

fn parse_show(bytes: &[u8]) -> Result<BTreeMap<String, String>, ExecutorConfigurationError> {
    let text = std::str::from_utf8(bytes).map_err(|_| ExecutorConfigurationError::InvalidUtf8)?;
    let expected: BTreeSet<&str> = SHOW_PROPERTIES.iter().copied().collect();
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let (key, value) = line
            .split_once('=')
            .ok_or(ExecutorConfigurationError::MalformedSystemctlShow)?;
        if !expected.contains(key) {
            return Err(ExecutorConfigurationError::UnexpectedSystemctlProperty(key.to_string()));
        }
        if out.insert(key.to_string(), value.to_string()).is_some() {
            return Err(ExecutorConfigurationError::DuplicateSystemctlProperty(key.to_string()));
        }
    }
    if out.len() != expected.len() {
        return Err(ExecutorConfigurationError::MissingSystemctlProperties);
    }
    Ok(out)
}

fn required<'a>(
    properties: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, ExecutorConfigurationError> {
    properties
        .get(key)
        .map(String::as_str)
        .ok_or(ExecutorConfigurationError::MissingSystemctlProperties)
}

fn parse_bool(value: &str) -> Result<bool, ExecutorConfigurationError> {
    match value {
        "yes" | "true" | "1" => Ok(true),
        "no" | "false" | "0" => Ok(false),
        _ => Err(ExecutorConfigurationError::InvalidSystemdBoolean(value.to_string())),
    }
}

fn parse_protect_system(value: &str) -> Result<ProtectSystemV1, ExecutorConfigurationError> {
    match value {
        "no" | "false" => Ok(ProtectSystemV1::No),
        "yes" | "true" => Ok(ProtectSystemV1::Yes),
        "full" => Ok(ProtectSystemV1::Full),
        "strict" => Ok(ProtectSystemV1::Strict),
        _ => Err(ExecutorConfigurationError::InvalidProtectSystem(value.to_string())),
    }
}

fn parse_protect_home(value: &str) -> Result<ProtectHomeV1, ExecutorConfigurationError> {
    match value {
        "no" | "false" => Ok(ProtectHomeV1::No),
        "yes" | "true" => Ok(ProtectHomeV1::Yes),
        "read-only" => Ok(ProtectHomeV1::ReadOnly),
        "tmpfs" => Ok(ProtectHomeV1::Tmpfs),
        _ => Err(ExecutorConfigurationError::InvalidProtectHome(value.to_string())),
    }
}

fn digest_bytes(bytes: &[u8]) -> Digest32 {
    Digest32(*blake3::hash(bytes).as_bytes())
}

fn digest_path(path: &Path) -> Digest32 {
    digest_bytes(path.as_os_str().to_string_lossy().as_bytes())
}

fn path_is_in_nix_store(path: &Path) -> bool {
    path.starts_with("/nix/store/")
}

struct Transcript {
    bytes: Vec<u8>,
}
impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(&(domain.len() as u32).to_be_bytes());
        bytes.extend_from_slice(domain);
        Self { bytes }
    }
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }
    fn bytes(&mut self, value: &[u8]) -> Result<(), ExecutorConfigurationError> {
        let len = u32::try_from(value.len()).map_err(|_| ExecutorConfigurationError::Encoding)?;
        self.bytes.extend_from_slice(&len.to_be_bytes());
        self.bytes.extend_from_slice(value);
        Ok(())
    }
    fn fixed<const N: usize>(&mut self, value: &[u8; N]) {
        self.bytes.extend_from_slice(value);
    }
    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.bytes).as_bytes()
    }
}

#[derive(Debug, Error)]
pub enum ExecutorConfigurationError {
    #[error("executor configuration policy is invalid")]
    InvalidPolicy,
    #[error("executor configuration snapshot is invalid")]
    InvalidSnapshot,
    #[error("executor configuration policy does not match the verified proof")]
    PolicyMismatch,
    #[error("current process does not belong to a unique systemd service")]
    NoSystemdServiceForCurrentProcess,
    #[error("current process maps to multiple systemd service candidates")]
    AmbiguousSystemdService,
    #[error("executor service unit mismatch: expected {expected}, actual {actual}")]
    UnexpectedExecutorUnit { expected: String, actual: String },
    #[error("systemd unit identity differs from the expected service")]
    UnitIdentityMismatch,
    #[error("systemd unit has no fragment path")]
    MissingFragmentPath,
    #[error("systemctl show failed: {0}")]
    SystemctlShowFailed(String),
    #[error("systemctl cat failed: {0}")]
    SystemctlCatFailed(String),
    #[error("systemctl cat returned empty unit text")]
    EmptyUnitText,
    #[error("systemctl show output is not valid UTF-8")]
    InvalidUtf8,
    #[error("systemctl show returned malformed output")]
    MalformedSystemctlShow,
    #[error("systemctl show returned unexpected property {0}")]
    UnexpectedSystemctlProperty(String),
    #[error("systemctl show returned duplicate property {0}")]
    DuplicateSystemctlProperty(String),
    #[error("systemctl show omitted one or more required properties")]
    MissingSystemctlProperties,
    #[error("invalid systemd boolean value {0}")]
    InvalidSystemdBoolean(String),
    #[error("invalid ProtectSystem value {0}")]
    InvalidProtectSystem(String),
    #[error("invalid ProtectHome value {0}")]
    InvalidProtectHome(String),
    #[error("executor systemd hardening policy is not satisfied")]
    HardeningPolicyNotSatisfied,
    #[error("executor configuration changed while it was being measured")]
    ConfigurationChangedDuringMeasurement,
    #[error("executor configuration changed since verification")]
    ConfigurationChanged,
    #[error("executor configuration canonical encoding failed")]
    Encoding,
    #[error("executor configuration measurement I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show_fixture() -> Vec<u8> {
        let mut rows = BTreeMap::new();
        for key in SHOW_PROPERTIES {
            rows.insert(*key, String::new());
        }
        rows.insert("Id", "symthaea-system-broker.service".into());
        rows.insert("FragmentPath", "/etc/systemd/system/symthaea-system-broker.service".into());
        rows.insert("ExecStart", "{ path=/nix/store/abc/bin/broker ; argv[]=/nix/store/abc/bin/broker ; }".into());
        rows.insert("User", "symthaea-broker".into());
        rows.insert("Group", "symthaea-broker".into());
        rows.insert("DynamicUser", "no".into());
        rows.insert("NoNewPrivileges", "yes".into());
        rows.insert("ProtectSystem", "strict".into());
        rows.insert("ProtectHome", "yes".into());
        rows.insert("PrivateTmp", "yes".into());
        rows.insert("PrivateDevices", "yes".into());
        rows.insert("PrivateNetwork", "no".into());
        rows.insert("MemoryDenyWriteExecute", "yes".into());
        rows.insert("LockPersonality", "yes".into());
        rows.insert("ProtectKernelTunables", "yes".into());
        rows.insert("ProtectKernelModules", "yes".into());
        rows.insert("ProtectControlGroups", "yes".into());
        rows.insert("RestrictSUIDSGID", "yes".into());
        rows.insert("RestrictRealtime", "yes".into());
        rows.insert("CapabilityBoundingSet", "cap_chown cap_dac_override".into());
        rows.insert("AmbientCapabilities", "".into());
        rows.insert("SystemCallFilter", "@system-service".into());
        rows.into_iter()
            .map(|(k, v)| format!("{k}={v}\n"))
            .collect::<String>()
            .into_bytes()
    }

    #[test]
    fn cgroup_parser_binds_unique_service_and_rejects_ambiguity() {
        assert_eq!(
            discover_systemd_service_from_cgroup(
                "0::/system.slice/symthaea-system-broker.service\n"
            )
            .unwrap(),
            "symthaea-system-broker.service"
        );
        assert!(matches!(
            discover_systemd_service_from_cgroup(
                "0::/system.slice/a.service/b.service\n"
            ),
            Err(ExecutorConfigurationError::AmbiguousSystemdService)
        ));
    }

    #[test]
    fn show_parser_requires_exact_unique_property_set() {
        let parsed = parse_show(&show_fixture()).unwrap();
        assert_eq!(parsed.len(), SHOW_PROPERTIES.len());
        assert_eq!(parsed["ProtectSystem"], "strict");

        let mut duplicate = show_fixture();
        duplicate.extend_from_slice(b"Id=symthaea-system-broker.service\n");
        assert!(matches!(
            parse_show(&duplicate),
            Err(ExecutorConfigurationError::DuplicateSystemctlProperty(_))
        ));
    }

    #[test]
    fn configuration_digest_changes_for_security_relevant_state() {
        let props = parse_show(&show_fixture()).unwrap();
        let base = ExecutorConfigurationSnapshotV1 {
            schema_version: 1,
            unit: "symthaea-system-broker.service".into(),
            fragment_path_digest: digest_bytes(props["FragmentPath"].as_bytes()),
            resolved_fragment_path_digest: Digest32([1; 32]),
            fragment_in_nix_store: true,
            unit_text_digest: Digest32([2; 32]),
            exec_start_digest: digest_bytes(props["ExecStart"].as_bytes()),
            user_digest: digest_bytes(props["User"].as_bytes()),
            group_digest: digest_bytes(props["Group"].as_bytes()),
            capability_bounding_set_digest: digest_bytes(props["CapabilityBoundingSet"].as_bytes()),
            ambient_capabilities_digest: digest_bytes(props["AmbientCapabilities"].as_bytes()),
            system_call_filter_digest: digest_bytes(props["SystemCallFilter"].as_bytes()),
            dynamic_user: false,
            no_new_privileges: true,
            protect_system: ProtectSystemV1::Strict,
            protect_home: ProtectHomeV1::Yes,
            private_tmp: true,
            private_devices: true,
            private_network: false,
            memory_deny_write_execute: true,
            lock_personality: true,
            protect_kernel_tunables: true,
            protect_kernel_modules: true,
            protect_control_groups: true,
            restrict_suid_sgid: true,
            restrict_realtime: true,
        };
        let mut changed = base.clone();
        changed.no_new_privileges = false;
        assert_ne!(base.digest().unwrap(), changed.digest().unwrap());
    }

    #[test]
    fn strict_policy_rejects_missing_hardening() {
        let policy = ExecutorConfigurationPolicyV1 {
            schema_version: 1,
            expected_unit: "symthaea-system-broker.service".into(),
            require_nix_store_fragment: true,
            require_no_new_privileges: true,
            require_protect_system_strict: true,
            require_protect_home: true,
            require_private_tmp: true,
            require_private_devices: true,
            require_memory_deny_write_execute: true,
            require_lock_personality: true,
            require_protect_kernel_tunables: true,
            require_protect_kernel_modules: true,
            require_protect_control_groups: true,
            require_restrict_suid_sgid: true,
            require_restrict_realtime: true,
        };
        let mut snapshot = ExecutorConfigurationSnapshotV1 {
            schema_version: 1,
            unit: policy.expected_unit.clone(),
            fragment_path_digest: Digest32([1; 32]),
            resolved_fragment_path_digest: Digest32([2; 32]),
            fragment_in_nix_store: true,
            unit_text_digest: Digest32([3; 32]),
            exec_start_digest: Digest32([4; 32]),
            user_digest: Digest32([5; 32]),
            group_digest: Digest32([6; 32]),
            capability_bounding_set_digest: Digest32([7; 32]),
            ambient_capabilities_digest: Digest32([8; 32]),
            system_call_filter_digest: Digest32([9; 32]),
            dynamic_user: false,
            no_new_privileges: true,
            protect_system: ProtectSystemV1::Strict,
            protect_home: ProtectHomeV1::Yes,
            private_tmp: true,
            private_devices: true,
            private_network: false,
            memory_deny_write_execute: true,
            lock_personality: true,
            protect_kernel_tunables: true,
            protect_kernel_modules: true,
            protect_control_groups: true,
            restrict_suid_sgid: true,
            restrict_realtime: true,
        };
        enforce_policy(&policy, &snapshot).unwrap();
        snapshot.protect_kernel_modules = false;
        assert!(matches!(
            enforce_policy(&policy, &snapshot),
            Err(ExecutorConfigurationError::HardeningPolicyNotSatisfied)
        ));
    }
}
