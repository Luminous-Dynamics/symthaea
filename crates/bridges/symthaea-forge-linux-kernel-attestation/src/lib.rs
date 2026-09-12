// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Parent-side Linux kernel observations for a paused Forge evaluator sandbox.
//!
//! This crate does not launch Bubblewrap. It observes one exact host process through `/proc` and
//! content-addresses what the kernel exposes about that process. The intended caller is a future
//! launcher that obtains Bubblewrap's host-visible `child-pid`, pauses setup with `--block-fd`,
//! performs this observation, and releases the child only after a strong gate passes.
//!
//! A strong gate requires all requested namespaces to differ from the observer, every capability
//! set to be zero, the root/Nix-store/model mounts to be read-only, `/tmp` to remain writable, the
//! sandbox PID to be PID 1 in its inner PID namespace, the nested-userns limit to be observed as 1,
//! and the expected sandbox hostname. The process start-time is sampled before and after all reads
//! to reject PID reuse or process replacement during observation.

use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use symthaea_algorithms::ContentId;
use thiserror::Error;

const MAX_PROC_TEXT_BYTES: usize = 8 * 1024 * 1024;
const EXPECTED_HOSTNAME: &str = "symthaea-forge-evaluator";
const REQUIRED_NAMESPACES: [&str; 7] = ["mnt", "user", "pid", "ipc", "uts", "net", "cgroup"];
const REQUIRED_MOUNTS: [&str; 4] = ["/", "/nix/store", "/runner", "/tmp"];

#[derive(Debug, Error)]
pub enum KernelAttestationError {
    #[error("kernel attestation is supported only on Linux")]
    UnsupportedPlatform,
    #[error("procfs IO error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("procfs observation exceeded the bounded text limit: {0:?}")]
    ProcTextTooLarge(PathBuf),
    #[error("could not parse process start-time from /proc/{0}/stat")]
    InvalidStat(u32),
    #[error("required process status field is missing or invalid: {0}")]
    InvalidStatusField(&'static str),
    #[error("required mount was not visible in process mountinfo: {0}")]
    MissingMount(&'static str),
    #[error("process identity changed while kernel state was being observed")]
    ProcessChangedDuringObservation,
    #[error("kernel observation identity does not match canonical fields")]
    ObservationIdentityMismatch,
    #[error("required Linux namespaces are not all independently observed as distinct")]
    NamespacesNotIsolated,
    #[error("sandbox capability sets are not all zero")]
    CapabilitiesRemain,
    #[error("sandbox mount policy does not match the strict evaluator filesystem theorem")]
    MountPolicyMismatch,
    #[error("sandbox process is not observed as PID 1 in its inner PID namespace")]
    PidNamespaceInitMismatch,
    #[error("nested user namespaces are not independently observed as disabled")]
    NestedUserNamespacesNotDisabled,
    #[error("sandbox hostname does not match the precommitted evaluator hostname")]
    HostnameMismatch,
    #[error("kernel isolation gate identity does not match canonical fields")]
    GateIdentityMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NamespaceObservation {
    name: String,
    observer_link: String,
    target_link: String,
    differs: bool,
}

impl NamespaceObservation {
    pub fn name(&self) -> &str { &self.name }
    pub fn observer_link(&self) -> &str { &self.observer_link }
    pub fn target_link(&self) -> &str { &self.target_link }
    pub fn differs(&self) -> bool { self.differs }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapabilityObservation {
    inheritable: u64,
    permitted: u64,
    effective: u64,
    bounding: u64,
    ambient: u64,
}

impl CapabilityObservation {
    pub fn inheritable(self) -> u64 { self.inheritable }
    pub fn permitted(self) -> u64 { self.permitted }
    pub fn effective(self) -> u64 { self.effective }
    pub fn bounding(self) -> u64 { self.bounding }
    pub fn ambient(self) -> u64 { self.ambient }

    pub fn all_zero(self) -> bool {
        self.inheritable == 0
            && self.permitted == 0
            && self.effective == 0
            && self.bounding == 0
            && self.ambient == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MountObservation {
    mount_point: String,
    options: Vec<String>,
    read_only: bool,
    read_write: bool,
}

impl MountObservation {
    pub fn mount_point(&self) -> &str { &self.mount_point }
    pub fn options(&self) -> &[String] { &self.options }
    pub fn read_only(&self) -> bool { self.read_only }
    pub fn read_write(&self) -> bool { self.read_write }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelSandboxObservation {
    id: ContentId,
    host_pid: u32,
    process_start_time_ticks: u64,
    namespaces: Vec<NamespaceObservation>,
    capabilities: CapabilityObservation,
    mounts: Vec<MountObservation>,
    namespace_pids: Vec<u32>,
    nested_user_namespace_limit: Option<u64>,
    hostname: Option<String>,
    seccomp_mode: Option<u32>,
    no_new_privs: Option<u32>,
}

impl KernelSandboxObservation {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn host_pid(&self) -> u32 { self.host_pid }
    pub fn process_start_time_ticks(&self) -> u64 { self.process_start_time_ticks }
    pub fn namespaces(&self) -> &[NamespaceObservation] { &self.namespaces }
    pub fn capabilities(&self) -> CapabilityObservation { self.capabilities }
    pub fn mounts(&self) -> &[MountObservation] { &self.mounts }
    pub fn namespace_pids(&self) -> &[u32] { &self.namespace_pids }
    pub fn nested_user_namespace_limit(&self) -> Option<u64> {
        self.nested_user_namespace_limit
    }
    pub fn hostname(&self) -> Option<&str> { self.hostname.as_deref() }
    pub fn seccomp_mode(&self) -> Option<u32> { self.seccomp_mode }
    pub fn no_new_privs(&self) -> Option<u32> { self.no_new_privs }

    pub fn mount(&self, mount_point: &str) -> Option<&MountObservation> {
        self.mounts.iter().find(|mount| mount.mount_point == mount_point)
    }

    pub fn validate(&self) -> Result<(), KernelAttestationError> {
        let expected = derive_observation_id(
            self.host_pid,
            self.process_start_time_ticks,
            &self.namespaces,
            self.capabilities,
            &self.mounts,
            &self.namespace_pids,
            self.nested_user_namespace_limit,
            self.hostname.as_deref(),
            self.seccomp_mode,
            self.no_new_privs,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(KernelAttestationError::ObservationIdentityMismatch)
        }
    }
}

/// Strong parent-side kernel observation gate. This is intentionally independent of Bubblewrap's
/// command-line recipe: it describes state actually visible in procfs for the exact host process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelIsolationGate {
    id: ContentId,
    observation_id: ContentId,
    host_pid: u32,
    process_start_time_ticks: u64,
}

impl KernelIsolationGate {
    pub fn issue(observation: &KernelSandboxObservation) -> Result<Self, KernelAttestationError> {
        observation.validate()?;
        validate_strong_state(observation)?;
        let id = derive_gate_id(
            observation.id(),
            observation.host_pid(),
            observation.process_start_time_ticks(),
        );
        Ok(Self {
            id,
            observation_id: observation.id().clone(),
            host_pid: observation.host_pid(),
            process_start_time_ticks: observation.process_start_time_ticks(),
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn observation_id(&self) -> &ContentId { &self.observation_id }
    pub fn host_pid(&self) -> u32 { self.host_pid }
    pub fn process_start_time_ticks(&self) -> u64 { self.process_start_time_ticks }

    pub fn validate_for(
        &self,
        observation: &KernelSandboxObservation,
    ) -> Result<(), KernelAttestationError> {
        observation.validate()?;
        validate_strong_state(observation)?;
        if self.observation_id != *observation.id()
            || self.host_pid != observation.host_pid()
            || self.process_start_time_ticks != observation.process_start_time_ticks()
        {
            return Err(KernelAttestationError::GateIdentityMismatch);
        }
        let expected = derive_gate_id(
            &self.observation_id,
            self.host_pid,
            self.process_start_time_ticks,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(KernelAttestationError::GateIdentityMismatch)
        }
    }
}

/// Observe one exact host PID twice around the full procfs read set, rejecting any identity change.
pub fn observe_sandbox_process(host_pid: u32) -> Result<KernelSandboxObservation, KernelAttestationError> {
    if !cfg!(target_os = "linux") {
        return Err(KernelAttestationError::UnsupportedPlatform);
    }
    let start_before = read_start_time(host_pid)?;
    let namespaces = observe_namespaces(host_pid)?;
    let status = read_proc_text(PathBuf::from(format!("/proc/{host_pid}/status")))?;
    let capabilities = parse_capabilities(&status)?;
    let namespace_pids = parse_nspid(&status)?;
    let seccomp_mode = parse_decimal_field(&status, "Seccomp:");
    let no_new_privs = parse_decimal_field(&status, "NoNewPrivs:");
    let mounts = observe_mounts(host_pid)?;
    let nested_user_namespace_limit = read_optional_u64(PathBuf::from(format!(
        "/proc/{host_pid}/root/proc/sys/user/max_user_namespaces"
    )));
    let hostname = read_optional_trimmed(PathBuf::from(format!(
        "/proc/{host_pid}/root/proc/sys/kernel/hostname"
    )));
    let start_after = read_start_time(host_pid)?;
    if start_before != start_after {
        return Err(KernelAttestationError::ProcessChangedDuringObservation);
    }
    let id = derive_observation_id(
        host_pid,
        start_before,
        &namespaces,
        capabilities,
        &mounts,
        &namespace_pids,
        nested_user_namespace_limit,
        hostname.as_deref(),
        seccomp_mode,
        no_new_privs,
    );
    let observation = KernelSandboxObservation {
        id,
        host_pid,
        process_start_time_ticks: start_before,
        namespaces,
        capabilities,
        mounts,
        namespace_pids,
        nested_user_namespace_limit,
        hostname,
        seccomp_mode,
        no_new_privs,
    };
    observation.validate()?;
    Ok(observation)
}

fn validate_strong_state(observation: &KernelSandboxObservation) -> Result<(), KernelAttestationError> {
    if REQUIRED_NAMESPACES.iter().any(|required| {
        !observation
            .namespaces()
            .iter()
            .any(|namespace| namespace.name() == *required && namespace.differs())
    }) {
        return Err(KernelAttestationError::NamespacesNotIsolated);
    }
    if !observation.capabilities().all_zero() {
        return Err(KernelAttestationError::CapabilitiesRemain);
    }
    let root = observation.mount("/").ok_or(KernelAttestationError::MissingMount("/"))?;
    let nix = observation
        .mount("/nix/store")
        .ok_or(KernelAttestationError::MissingMount("/nix/store"))?;
    let runner = observation
        .mount("/runner")
        .ok_or(KernelAttestationError::MissingMount("/runner"))?;
    let tmp = observation
        .mount("/tmp")
        .ok_or(KernelAttestationError::MissingMount("/tmp"))?;
    if !root.read_only() || !nix.read_only() || !runner.read_only() || !tmp.read_write() {
        return Err(KernelAttestationError::MountPolicyMismatch);
    }
    if observation.namespace_pids().last().copied() != Some(1) {
        return Err(KernelAttestationError::PidNamespaceInitMismatch);
    }
    if observation.nested_user_namespace_limit() != Some(1) {
        return Err(KernelAttestationError::NestedUserNamespacesNotDisabled);
    }
    if observation.hostname() != Some(EXPECTED_HOSTNAME) {
        return Err(KernelAttestationError::HostnameMismatch);
    }
    Ok(())
}

fn observe_namespaces(host_pid: u32) -> Result<Vec<NamespaceObservation>, KernelAttestationError> {
    let mut observations = Vec::with_capacity(REQUIRED_NAMESPACES.len());
    for name in REQUIRED_NAMESPACES {
        let observer_path = PathBuf::from(format!("/proc/self/ns/{name}"));
        let target_path = PathBuf::from(format!("/proc/{host_pid}/ns/{name}"));
        let observer = fs::read_link(&observer_path).map_err(|source| KernelAttestationError::Io {
            path: observer_path,
            source,
        })?;
        let target = fs::read_link(&target_path).map_err(|source| KernelAttestationError::Io {
            path: target_path,
            source,
        })?;
        let observer_link = observer.to_string_lossy().into_owned();
        let target_link = target.to_string_lossy().into_owned();
        observations.push(NamespaceObservation {
            name: name.to_string(),
            differs: observer_link != target_link,
            observer_link,
            target_link,
        });
    }
    observations.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(observations)
}

fn observe_mounts(host_pid: u32) -> Result<Vec<MountObservation>, KernelAttestationError> {
    let path = PathBuf::from(format!("/proc/{host_pid}/mountinfo"));
    let text = read_proc_text(path)?;
    let all = parse_mountinfo(&text);
    let mut required = Vec::with_capacity(REQUIRED_MOUNTS.len());
    for mount_point in REQUIRED_MOUNTS {
        let mount = all
            .get(mount_point)
            .cloned()
            .ok_or(KernelAttestationError::MissingMount(mount_point))?;
        required.push(mount);
    }
    required.sort_by(|a, b| a.mount_point.cmp(&b.mount_point));
    Ok(required)
}

fn parse_mountinfo(text: &str) -> BTreeMap<String, MountObservation> {
    let mut mounts = BTreeMap::new();
    for line in text.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 6 {
            continue;
        }
        let mount_point = fields[4].to_string();
        let mut options = fields[5]
            .split(',')
            .map(str::to_string)
            .collect::<Vec<_>>();
        options.sort();
        options.dedup();
        let read_only = options.iter().any(|value| value == "ro");
        let read_write = options.iter().any(|value| value == "rw");
        mounts.insert(
            mount_point.clone(),
            MountObservation {
                mount_point,
                options,
                read_only,
                read_write,
            },
        );
    }
    mounts
}

fn parse_capabilities(status: &str) -> Result<CapabilityObservation, KernelAttestationError> {
    Ok(CapabilityObservation {
        inheritable: parse_hex_field(status, "CapInh:")
            .ok_or(KernelAttestationError::InvalidStatusField("CapInh"))?,
        permitted: parse_hex_field(status, "CapPrm:")
            .ok_or(KernelAttestationError::InvalidStatusField("CapPrm"))?,
        effective: parse_hex_field(status, "CapEff:")
            .ok_or(KernelAttestationError::InvalidStatusField("CapEff"))?,
        bounding: parse_hex_field(status, "CapBnd:")
            .ok_or(KernelAttestationError::InvalidStatusField("CapBnd"))?,
        ambient: parse_hex_field(status, "CapAmb:")
            .ok_or(KernelAttestationError::InvalidStatusField("CapAmb"))?,
    })
}

fn parse_hex_field(status: &str, key: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .and_then(|value| u64::from_str_radix(value.trim(), 16).ok())
}

fn parse_decimal_field(status: &str, key: &str) -> Option<u32> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .and_then(|value| value.trim().parse().ok())
}

fn parse_nspid(status: &str) -> Result<Vec<u32>, KernelAttestationError> {
    let value = status
        .lines()
        .find_map(|line| line.strip_prefix("NSpid:"))
        .ok_or(KernelAttestationError::InvalidStatusField("NSpid"))?;
    let pids = value
        .split_whitespace()
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| KernelAttestationError::InvalidStatusField("NSpid"))?;
    if pids.is_empty() {
        Err(KernelAttestationError::InvalidStatusField("NSpid"))
    } else {
        Ok(pids)
    }
}

fn read_start_time(host_pid: u32) -> Result<u64, KernelAttestationError> {
    let path = PathBuf::from(format!("/proc/{host_pid}/stat"));
    let stat = read_proc_text(path)?;
    let close_paren = stat.rfind(')').ok_or(KernelAttestationError::InvalidStat(host_pid))?;
    let remainder = stat
        .get(close_paren + 1..)
        .ok_or(KernelAttestationError::InvalidStat(host_pid))?;
    // `remainder` begins with field 3 (state), so field 22 (starttime) is zero-based index 19.
    remainder
        .split_whitespace()
        .nth(19)
        .and_then(|value| value.parse().ok())
        .ok_or(KernelAttestationError::InvalidStat(host_pid))
}

fn read_proc_text(path: PathBuf) -> Result<String, KernelAttestationError> {
    let text = fs::read_to_string(&path).map_err(|source| KernelAttestationError::Io {
        path: path.clone(),
        source,
    })?;
    if text.len() > MAX_PROC_TEXT_BYTES {
        return Err(KernelAttestationError::ProcTextTooLarge(path));
    }
    Ok(text)
}

fn read_optional_u64(path: PathBuf) -> Option<u64> {
    read_proc_text(path).ok()?.trim().parse().ok()
}

fn read_optional_trimmed(path: PathBuf) -> Option<String> {
    let value = read_proc_text(path).ok()?.trim().to_string();
    (!value.is_empty()).then_some(value)
}

#[allow(clippy::too_many_arguments)]
fn derive_observation_id(
    host_pid: u32,
    process_start_time_ticks: u64,
    namespaces: &[NamespaceObservation],
    capabilities: CapabilityObservation,
    mounts: &[MountObservation],
    namespace_pids: &[u32],
    nested_user_namespace_limit: Option<u64>,
    hostname: Option<&str>,
    seccomp_mode: Option<u32>,
    no_new_privs: Option<u32>,
) -> ContentId {
    let mut parts = vec![
        host_pid.to_be_bytes().to_vec(),
        process_start_time_ticks.to_be_bytes().to_vec(),
    ];
    parts.push((namespaces.len() as u64).to_be_bytes().to_vec());
    for namespace in namespaces {
        parts.push(namespace.name.as_bytes().to_vec());
        parts.push(namespace.observer_link.as_bytes().to_vec());
        parts.push(namespace.target_link.as_bytes().to_vec());
        parts.push(vec![u8::from(namespace.differs)]);
    }
    for capability in [
        capabilities.inheritable,
        capabilities.permitted,
        capabilities.effective,
        capabilities.bounding,
        capabilities.ambient,
    ] {
        parts.push(capability.to_be_bytes().to_vec());
    }
    parts.push((mounts.len() as u64).to_be_bytes().to_vec());
    for mount in mounts {
        parts.push(mount.mount_point.as_bytes().to_vec());
        parts.push((mount.options.len() as u64).to_be_bytes().to_vec());
        for option in &mount.options {
            parts.push(option.as_bytes().to_vec());
        }
        parts.push(vec![u8::from(mount.read_only), u8::from(mount.read_write)]);
    }
    parts.push((namespace_pids.len() as u64).to_be_bytes().to_vec());
    for pid in namespace_pids {
        parts.push(pid.to_be_bytes().to_vec());
    }
    parts.push(nested_user_namespace_limit.unwrap_or(u64::MAX).to_be_bytes().to_vec());
    parts.push(hostname.unwrap_or("").as_bytes().to_vec());
    parts.push(seccomp_mode.unwrap_or(u32::MAX).to_be_bytes().to_vec());
    parts.push(no_new_privs.unwrap_or(u32::MAX).to_be_bytes().to_vec());
    ContentId::derive(
        "symthaea.forge-kernel-sandbox-observation.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn derive_gate_id(observation_id: &ContentId, host_pid: u32, start_time_ticks: u64) -> ContentId {
    ContentId::derive(
        "symthaea.forge-kernel-isolation-gate.v1",
        [
            observation_id.as_str().as_bytes(),
            host_pid.to_be_bytes().as_slice(),
            start_time_ticks.to_be_bytes().as_slice(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mountinfo_parser_preserves_required_rw_semantics() {
        let sample = "36 25 0:32 / / ro,relatime - tmpfs tmpfs ro\n\
41 36 0:40 / /nix/store ro,nosuid,nodev - tmpfs tmpfs ro\n\
42 36 0:41 / /runner ro,nosuid,nodev - tmpfs tmpfs ro\n\
43 36 0:42 / /tmp rw,nosuid,nodev - tmpfs tmpfs rw\n";
        let mounts = parse_mountinfo(sample);
        assert!(mounts["/"].read_only());
        assert!(mounts["/nix/store"].read_only());
        assert!(mounts["/runner"].read_only());
        assert!(mounts["/tmp"].read_write());
    }

    #[test]
    fn capability_parser_requires_all_five_sets() {
        let status = "CapInh:\t0000000000000000\nCapPrm:\t0000000000000000\nCapEff:\t0000000000000000\nCapBnd:\t0000000000000000\nCapAmb:\t0000000000000000\nNSpid:\t123\t1\n";
        let capabilities = parse_capabilities(status).unwrap();
        assert!(capabilities.all_zero());
        assert_eq!(parse_nspid(status).unwrap(), vec![123, 1]);
    }

    #[test]
    fn nonzero_bounding_capability_fails_zero_test() {
        let capabilities = CapabilityObservation {
            inheritable: 0,
            permitted: 0,
            effective: 0,
            bounding: 1,
            ambient: 0,
        };
        assert!(!capabilities.all_zero());
    }

    #[test]
    fn stat_start_time_parser_handles_parenthesized_command_names() {
        // Fields after ')' begin at field 3. Field 22/starttime is therefore zero-based index 19.
        let stat = "77 (name with spaces) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 424242 21";
        let close = stat.rfind(')').unwrap();
        let value = stat[close + 1..]
            .split_whitespace()
            .nth(19)
            .unwrap()
            .parse::<u64>()
            .unwrap();
        assert_eq!(value, 424242);
    }
}
