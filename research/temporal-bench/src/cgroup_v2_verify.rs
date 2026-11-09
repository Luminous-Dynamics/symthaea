//! Read back and verify effective cgroup-v2 limits for a candidate scope.
//!
//! Evaluator-side only: reads host cgroup control files outside the candidate
//! mount namespace. This is an observation, not a signature or hardware attestation.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use super::digest::CanonicalHasher;

const REQUIRED_CONTROLLERS: [&str; 3] = ["cpu", "memory", "pids"];
const MAX_CGROUP_CONTROL_FILE_BYTES: u64 = 64 * 1024;
const MAX_SYSTEMD_SHOW_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CgroupV2Error {
    InvalidControlGroupPath,
    InvalidMountPath,
    Cgroup2MountNotFound,
    AmbiguousCgroup2Mount,
    Cgroup2MountRootNotUnified,
    InvalidMountInfoEscape,
    InvalidMountInfoUtf8,
    ReadFailed { path: String, reason: String },
    ControlFileTooLarge { path: String, maximum_bytes: u64 },
    ScopePathMismatch,
    MissingController(&'static str),
    ControllerNotEnabled(&'static str),
    InvalidControlValue(&'static str),
    MemoryLimitMismatch { expected: u64, actual: u64 },
    SwapLimitMismatch { expected: u64, actual: u64 },
    CpuQuotaMismatch {
        expected_percent: u32,
        actual_quota: u64,
        actual_period: u64,
    },
    TaskLimitMismatch { expected: u32, actual: u32 },
    OomGroupNotEnabled,
    CandidateNotInScope { pid: u32 },
    ScopeNotPopulated,
    SystemdPropertyParse,
    MissingSystemdProperty(&'static str),
    DuplicateSystemdProperty(&'static str),
    InvalidSystemdProperty(&'static str),
    SystemdPropertyMismatch(&'static str),
    SystemdScopeNotActive,
    InvalidExpectedLimit(&'static str),
}

impl fmt::Display for CgroupV2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidControlGroupPath => write!(f, "systemd control-group path is not a safe absolute path"),
            Self::InvalidMountPath => write!(f, "cgroup2 mount path must be absolute"),
            Self::Cgroup2MountNotFound => write!(f, "requested mount point is not a unified cgroup2 mount"),
            Self::AmbiguousCgroup2Mount => write!(f, "multiple cgroup2 mounts match the requested mount point"),
            Self::Cgroup2MountRootNotUnified => write!(f, "cgroup2 mount does not expose the unified hierarchy root"),
            Self::InvalidMountInfoEscape => write!(f, "mountinfo contains an invalid escaped path field"),
            Self::InvalidMountInfoUtf8 => write!(f, "mountinfo path field is not valid UTF-8"),
            Self::ReadFailed { path, reason } => write!(f, "could not read cgroup evidence at {path}: {reason}"),
            Self::ControlFileTooLarge { path, maximum_bytes } => write!(f, "cgroup control file {path} exceeds {maximum_bytes} bytes"),
            Self::ScopePathMismatch => write!(f, "observed cgroup path differs from the expected systemd scope"),
            Self::MissingController(name) => write!(f, "parent cgroup does not expose required {name} controller"),
            Self::ControllerNotEnabled(name) => write!(f, "parent cgroup has not enabled {name} for its children"),
            Self::InvalidControlValue(name) => write!(f, "invalid or missing cgroup-v2 control value: {name}"),
            Self::MemoryLimitMismatch { expected, actual } => write!(f, "memory.max is {actual}, expected {expected}"),
            Self::SwapLimitMismatch { expected, actual } => write!(f, "memory.swap.max is {actual}, expected {expected}"),
            Self::CpuQuotaMismatch { expected_percent, actual_quota, actual_period } => {
                write!(f, "cpu.max quota/period {actual_quota}/{actual_period} does not equal {expected_percent}%")
            }
            Self::TaskLimitMismatch { expected, actual } => write!(f, "pids.max is {actual}, expected {expected}"),
            Self::OomGroupNotEnabled => write!(f, "memory.oom.group is not enabled"),
            Self::CandidateNotInScope { pid } => write!(f, "candidate PID {pid} is not a member of the expected scope cgroup"),
            Self::ScopeNotPopulated => write!(f, "candidate scope cgroup is not populated"),
            Self::SystemdPropertyParse => write!(f, "systemd show output is not valid KEY=VALUE data"),
            Self::MissingSystemdProperty(name) => write!(f, "systemd show output is missing required property {name}"),
            Self::DuplicateSystemdProperty(name) => write!(f, "systemd show output repeats property {name}"),
            Self::InvalidSystemdProperty(name) => write!(f, "systemd show property {name} has an invalid value"),
            Self::SystemdPropertyMismatch(name) => write!(f, "systemd show property {name} does not match the requested scope policy"),
            Self::SystemdScopeNotActive => write!(f, "systemd scope is not active/running"),
            Self::InvalidExpectedLimit(name) => write!(f, "invalid expected cgroup limit: {name}"),
        }
    }
}

impl Error for CgroupV2Error {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedCgroupV2Limits {
    /// Exact ControlGroup property returned by the trusted systemd manager.
    pub control_group: String,
    /// PID of the Bubblewrap process after it has entered the systemd scope.
    pub candidate_pid: u32,
    pub memory_max_bytes: u64,
    pub cpu_quota_percent: u32,
    pub tasks_max: u32,
}

/// Expected properties returned by the trusted systemd manager for one transient scope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedSystemdScopeProperties {
    pub unit_name: String,
    pub control_group: String,
    pub memory_max_bytes: u64,
    pub memory_swap_max_bytes: u64,
    pub cpu_quota_percent: u32,
    pub tasks_max: u32,
    pub runtime_max_usec: u64,
}

/// Parsed, normalized fields from a bounded `systemctl show` KEY=VALUE response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SystemdScopeProperties {
    pub unit_name: String,
    pub active_state: String,
    pub sub_state: String,
    pub control_group: String,
    pub memory_max_bytes: u64,
    pub memory_swap_max_bytes: u64,
    pub cpu_quota_per_sec_usec: u64,
    pub tasks_max: u64,
    pub runtime_max_usec: u64,
    pub oom_policy: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CgroupV2Snapshot {
    /// Exact systemd ControlGroup value used to locate this snapshot.
    pub control_group: String,
    /// cgroup.controllers from the candidate scope's immediate parent.
    pub parent_controllers: String,
    /// cgroup.subtree_control from the candidate scope's immediate parent.
    pub parent_subtree_control: String,
    /// Raw contents of the candidate scope's effective control files.
    pub memory_max: String,
    pub memory_swap_max: String,
    pub cpu_max: String,
    pub pids_max: String,
    pub memory_oom_group: String,
    pub cgroup_procs: String,
    pub cgroup_events: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CgroupV2Attestation {
    pub control_group: String,
    pub candidate_pid: u32,
    pub memory_max_bytes: u64,
    pub memory_swap_max_bytes: u64,
    pub cpu_quota_usec: u64,
    pub cpu_period_usec: u64,
    pub tasks_max: u32,
    pub oom_group: bool,
    /// Digest of the exact verified read-back values. This is not a signature.
    pub limits_sha256: [u8; 32],
}

fn safe_control_group_path(value: &str) -> bool {
    if !value.starts_with('/')
        || value.starts_with("//")
        || value == "/"
        || value.ends_with('/')
        || value.contains('\0')
    {
        return false;
    }

    let mut segments = value.split('/');
    if segments.next() != Some("") {
        return false;
    }
    let segments: Vec<&str> = segments.collect();
    if segments.is_empty() || segments.iter().any(|segment| {
        segment.is_empty() || *segment == "." || *segment == ".."
    }) {
        return false;
    }
    segments.last().is_some_and(|name| name.ends_with(".scope"))
}

fn decode_mountinfo_path(value: &str) -> Result<String, CgroupV2Error> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] != b'\\' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }

        let escaped = bytes
            .get(index + 1..index + 4)
            .ok_or(CgroupV2Error::InvalidMountInfoEscape)?;
        if !escaped.iter().all(|byte| (b'0'..=b'7').contains(byte)) {
            return Err(CgroupV2Error::InvalidMountInfoEscape);
        }
        let byte = (u16::from(escaped[0] - b'0') << 6)
            | (u16::from(escaped[1] - b'0') << 3)
            | u16::from(escaped[2] - b'0');
        if byte > u16::from(u8::MAX) {
            return Err(CgroupV2Error::InvalidMountInfoEscape);
        }
        decoded.push(byte as u8);
        index += 4;
    }

    String::from_utf8(decoded).map_err(|_| CgroupV2Error::InvalidMountInfoUtf8)
}

/// Require exactly one cgroup2 mount at expected_mount, rooted at the unified hierarchy.
/// The input must be read from the trusted supervisor's own /proc/self/mountinfo.
pub fn verify_cgroup2_mount(
    mountinfo: &str,
    expected_mount: &Path,
) -> Result<(), CgroupV2Error> {
    if !expected_mount.is_absolute() {
        return Err(CgroupV2Error::InvalidMountPath);
    }

    let mut matches = 0_usize;
    for line in mountinfo.lines() {
        let Some((before_separator, after_separator)) = line.split_once(" - ") else {
            continue;
        };
        let left: Vec<&str> = before_separator.split_whitespace().collect();
        let right: Vec<&str> = after_separator.split_whitespace().collect();
        if left.len() < 6 || right.len() < 3 || right[0] != "cgroup2" {
            continue;
        }

        let root = decode_mountinfo_path(left[3])?;
        let mount_point = decode_mountinfo_path(left[4])?;
        if Path::new(&mount_point) != expected_mount {
            continue;
        }
        if root != "/" {
            return Err(CgroupV2Error::Cgroup2MountRootNotUnified);
        }
        matches += 1;
    }

    match matches {
        0 => Err(CgroupV2Error::Cgroup2MountNotFound),
        1 => Ok(()),
        _ => Err(CgroupV2Error::AmbiguousCgroup2Mount),
    }
}

fn read_control_file(path: &Path) -> Result<String, CgroupV2Error> {
    let display_path = path.display().to_string();
    let file = File::open(path).map_err(|error| CgroupV2Error::ReadFailed {
        path: display_path.clone(),
        reason: error.to_string(),
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_CGROUP_CONTROL_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| CgroupV2Error::ReadFailed {
            path: display_path.clone(),
            reason: error.to_string(),
        })?;
    if bytes.len() as u64 > MAX_CGROUP_CONTROL_FILE_BYTES {
        return Err(CgroupV2Error::ControlFileTooLarge {
            path: display_path,
            maximum_bytes: MAX_CGROUP_CONTROL_FILE_BYTES,
        });
    }
    String::from_utf8(bytes).map_err(|error| CgroupV2Error::ReadFailed {
        path: display_path,
        reason: format!("control file is not UTF-8: {error}"),
    })
}

/// Read a bounded snapshot from the trusted host cgroup mount.
///
/// control_group must come from the trusted systemd manager's ControlGroup
/// property. Canonicalized paths must remain beneath cgroup_mount.
pub fn read_cgroup_v2_snapshot(
    cgroup_mount: &Path,
    control_group: &str,
) -> Result<CgroupV2Snapshot, CgroupV2Error> {
    if !safe_control_group_path(control_group) {
        return Err(CgroupV2Error::InvalidControlGroupPath);
    }
    if !cgroup_mount.is_absolute() {
        return Err(CgroupV2Error::InvalidMountPath);
    }

    let canonical_mount = fs::canonicalize(cgroup_mount).map_err(|error| {
        CgroupV2Error::ReadFailed {
            path: cgroup_mount.display().to_string(),
            reason: error.to_string(),
        }
    })?;
    let relative = control_group
        .strip_prefix('/')
        .ok_or(CgroupV2Error::InvalidControlGroupPath)?;
    let requested_scope = canonical_mount.join(relative);
    let canonical_scope = fs::canonicalize(&requested_scope).map_err(|error| {
        CgroupV2Error::ReadFailed {
            path: requested_scope.display().to_string(),
            reason: error.to_string(),
        }
    })?;

    if !canonical_scope.starts_with(&canonical_mount) || canonical_scope == canonical_mount {
        return Err(CgroupV2Error::InvalidControlGroupPath);
    }
    let parent = canonical_scope
        .parent()
        .ok_or(CgroupV2Error::InvalidControlGroupPath)?;
    if !parent.starts_with(&canonical_mount) {
        return Err(CgroupV2Error::InvalidControlGroupPath);
    }

    Ok(CgroupV2Snapshot {
        control_group: control_group.to_string(),
        parent_controllers: read_control_file(&parent.join("cgroup.controllers"))?,
        parent_subtree_control: read_control_file(&parent.join("cgroup.subtree_control"))?,
        memory_max: read_control_file(&canonical_scope.join("memory.max"))?,
        memory_swap_max: read_control_file(&canonical_scope.join("memory.swap.max"))?,
        cpu_max: read_control_file(&canonical_scope.join("cpu.max"))?,
        pids_max: read_control_file(&canonical_scope.join("pids.max"))?,
        memory_oom_group: read_control_file(&canonical_scope.join("memory.oom.group"))?,
        cgroup_procs: read_control_file(&canonical_scope.join("cgroup.procs"))?,
        cgroup_events: read_control_file(&canonical_scope.join("cgroup.events"))?,
    })
}

fn required_systemd_property<'a>(
    properties: &'a std::collections::BTreeMap<String, String>,
    name: &'static str,
) -> Result<&'a str, CgroupV2Error> {
    properties
        .get(name)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(CgroupV2Error::MissingSystemdProperty(name))
}

fn parse_systemd_u64(
    properties: &std::collections::BTreeMap<String, String>,
    name: &'static str,
) -> Result<u64, CgroupV2Error> {
    required_systemd_property(properties, name)?
        .parse::<u64>()
        .map_err(|_| CgroupV2Error::InvalidSystemdProperty(name))
}

/// Parse systemd's normalized integer-component timespan representation into
/// microseconds. Examples include "190ms", "1s", and "1min 30s".
///
/// Fractional, unknown, unbounded ("infinity"), empty and overflowing values
/// fail closed. Sub-microsecond precision is rejected rather than rounded.
fn parse_systemd_timespan_usec(
    value: &str,
    name: &'static str,
) -> Result<u64, CgroupV2Error> {
    let value = value.trim();
    if value.is_empty() || value == "infinity" {
        return Err(CgroupV2Error::InvalidSystemdProperty(name));
    }

    const NS_PER_US: u128 = 1_000;
    let mut total_ns = 0_u128;

    for component in value.split_whitespace() {
        let digit_end = component
            .find(|character: char| !character.is_ascii_digit())
            .unwrap_or(component.len());
        if digit_end == 0 || digit_end == component.len() {
            return Err(CgroupV2Error::InvalidSystemdProperty(name));
        }

        let amount = component[..digit_end]
            .parse::<u128>()
            .map_err(|_| CgroupV2Error::InvalidSystemdProperty(name))?;
        let unit = &component[digit_end..];
        let ns_per_unit = match unit {
            "ns" => 1_u128,
            "us" | "µs" => 1_000_u128,
            "ms" => 1_000_000_u128,
            "s" => 1_000_000_000_u128,
            "min" => 60_000_000_000_u128,
            "h" => 3_600_000_000_000_u128,
            "d" => 86_400_000_000_000_u128,
            "w" => 604_800_000_000_000_u128,
            _ => return Err(CgroupV2Error::InvalidSystemdProperty(name)),
        };
        let component_ns = amount
            .checked_mul(ns_per_unit)
            .ok_or(CgroupV2Error::InvalidSystemdProperty(name))?;
        total_ns = total_ns
            .checked_add(component_ns)
            .ok_or(CgroupV2Error::InvalidSystemdProperty(name))?;
    }

    if total_ns % NS_PER_US != 0 {
        return Err(CgroupV2Error::InvalidSystemdProperty(name));
    }
    u64::try_from(total_ns / NS_PER_US)
        .map_err(|_| CgroupV2Error::InvalidSystemdProperty(name))
}

/// Parse the exact `KEY=VALUE` response selected by the trusted caller.
///
/// Query only the listed properties with `systemctl show --property=...`.
/// Unknown keys are ignored for forward compatibility, while duplicate keys,
/// missing required keys, empty values and nonnumeric limit fields fail closed.
pub fn parse_systemd_scope_properties(
    output: &str,
) -> Result<SystemdScopeProperties, CgroupV2Error> {
    if output.len() > MAX_SYSTEMD_SHOW_BYTES {
        return Err(CgroupV2Error::SystemdPropertyParse);
    }
    let mut fields = std::collections::BTreeMap::<String, String>::new();
    for line in output.lines() {
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or(CgroupV2Error::SystemdPropertyParse)?;
        if key.is_empty() || value.contains('\0') {
            return Err(CgroupV2Error::SystemdPropertyParse);
        }
        if fields.insert(key.to_string(), value.to_string()).is_some() {
            return Err(CgroupV2Error::DuplicateSystemdProperty(match key {
                "Id" => "Id",
                "ActiveState" => "ActiveState",
                "SubState" => "SubState",
                "ControlGroup" => "ControlGroup",
                "MemoryMax" => "MemoryMax",
                "MemorySwapMax" => "MemorySwapMax",
                "CPUQuotaPerSecUSec" => "CPUQuotaPerSecUSec",
                "TasksMax" => "TasksMax",
                "RuntimeMaxUSec" => "RuntimeMaxUSec",
                "OOMPolicy" => "OOMPolicy",
                _ => "unknown",
            }));
        }
    }

    Ok(SystemdScopeProperties {
        unit_name: required_systemd_property(&fields, "Id")?.to_string(),
        active_state: required_systemd_property(&fields, "ActiveState")?.to_string(),
        sub_state: required_systemd_property(&fields, "SubState")?.to_string(),
        control_group: required_systemd_property(&fields, "ControlGroup")?.to_string(),
        memory_max_bytes: parse_systemd_u64(&fields, "MemoryMax")?,
        memory_swap_max_bytes: parse_systemd_u64(&fields, "MemorySwapMax")?,
        cpu_quota_per_sec_usec: parse_systemd_timespan_usec(required_systemd_property(&fields, "CPUQuotaPerSecUSec")?, "CPUQuotaPerSecUSec")?,
        tasks_max: parse_systemd_u64(&fields, "TasksMax")?,
        runtime_max_usec: parse_systemd_timespan_usec(required_systemd_property(&fields, "RuntimeMaxUSec")?, "RuntimeMaxUSec")?,
        oom_policy: required_systemd_property(&fields, "OOMPolicy")?.to_string(),
    })
}

/// Require the systemd manager's view of the scope to match the requested
/// policy before reading the actual kernel cgroup controls.
pub fn verify_systemd_scope_properties(
    expected: &ExpectedSystemdScopeProperties,
    observed: &SystemdScopeProperties,
) -> Result<(), CgroupV2Error> {
    if expected.unit_name.is_empty()
        || expected.unit_name.contains('/')
        || expected.unit_name.contains('\\0')
        || !expected.unit_name.ends_with(".scope")
        || expected.control_group.rsplit('/').next() != Some(expected.unit_name.as_str())
        || expected.memory_max_bytes == 0
        || expected.memory_swap_max_bytes != 0
        || !(1..=1_600).contains(&expected.cpu_quota_percent)
        || expected.tasks_max == 0
        || expected.runtime_max_usec == 0
        || !safe_control_group_path(&expected.control_group)
    {
        return Err(CgroupV2Error::InvalidExpectedLimit("systemd_scope_properties"));
    }

    if observed.active_state != "active" || observed.sub_state != "running" {
        return Err(CgroupV2Error::SystemdScopeNotActive);
    }
    if observed.unit_name != expected.unit_name {
        return Err(CgroupV2Error::SystemdPropertyMismatch("Id"));
    }
    if observed.control_group != expected.control_group {
        return Err(CgroupV2Error::SystemdPropertyMismatch("ControlGroup"));
    }
    if observed.memory_max_bytes != expected.memory_max_bytes {
        return Err(CgroupV2Error::SystemdPropertyMismatch("MemoryMax"));
    }
    if observed.memory_swap_max_bytes != expected.memory_swap_max_bytes {
        return Err(CgroupV2Error::SystemdPropertyMismatch("MemorySwapMax"));
    }
    let expected_cpu_usec = u64::from(expected.cpu_quota_percent)
        .checked_mul(10_000)
        .ok_or(CgroupV2Error::InvalidExpectedLimit("cpu_quota_percent"))?;
    if observed.cpu_quota_per_sec_usec != expected_cpu_usec {
        return Err(CgroupV2Error::SystemdPropertyMismatch("CPUQuotaPerSecUSec"));
    }
    if observed.tasks_max != u64::from(expected.tasks_max) {
        return Err(CgroupV2Error::SystemdPropertyMismatch("TasksMax"));
    }
    if observed.runtime_max_usec != expected.runtime_max_usec {
        return Err(CgroupV2Error::SystemdPropertyMismatch("RuntimeMaxUSec"));
    }
    if observed.oom_policy != "kill" {
        return Err(CgroupV2Error::SystemdPropertyMismatch("OOMPolicy"));
    }
    Ok(())
}

fn parse_single_u64(value: &str, name: &'static str) -> Result<u64, CgroupV2Error> {
    let value = value.trim();
    if value.is_empty() || value == "max" || value.split_whitespace().count() != 1 {
        return Err(CgroupV2Error::InvalidControlValue(name));
    }
    value
        .parse::<u64>()
        .map_err(|_| CgroupV2Error::InvalidControlValue(name))
}

fn parse_cpu_max(value: &str) -> Result<(u64, u64), CgroupV2Error> {
    let mut fields = value.split_whitespace();
    let quota = fields
        .next()
        .ok_or(CgroupV2Error::InvalidControlValue("cpu.max"))?;
    let period = fields
        .next()
        .ok_or(CgroupV2Error::InvalidControlValue("cpu.max"))?;
    if fields.next().is_some() || quota == "max" {
        return Err(CgroupV2Error::InvalidControlValue("cpu.max"));
    }
    let quota = quota
        .parse::<u64>()
        .map_err(|_| CgroupV2Error::InvalidControlValue("cpu.max"))?;
    let period = period
        .parse::<u64>()
        .map_err(|_| CgroupV2Error::InvalidControlValue("cpu.max"))?;
    if quota == 0 || period == 0 {
        return Err(CgroupV2Error::InvalidControlValue("cpu.max"));
    }
    Ok((quota, period))
}

fn parse_key_value_lines(
    value: &str,
    name: &'static str,
) -> Result<Vec<(String, String)>, CgroupV2Error> {
    let mut output = Vec::new();
    let mut seen = BTreeSet::new();
    for line in value.lines() {
        let mut fields = line.split_whitespace();
        let Some(key) = fields.next() else { continue };
        let Some(value) = fields.next() else {
            return Err(CgroupV2Error::InvalidControlValue(name));
        };
        if fields.next().is_some() || !seen.insert(key.to_string()) {
            return Err(CgroupV2Error::InvalidControlValue(name));
        }
        output.push((key.to_string(), value.to_string()));
    }
    Ok(output)
}

fn contains_controller(value: &str, expected: &str) -> bool {
    value.split_whitespace().any(|token| token == expected)
}

fn verify_parent_controllers(snapshot: &CgroupV2Snapshot) -> Result<(), CgroupV2Error> {
    for controller in REQUIRED_CONTROLLERS {
        if !contains_controller(&snapshot.parent_controllers, controller) {
            return Err(CgroupV2Error::MissingController(controller));
        }
        if !contains_controller(&snapshot.parent_subtree_control, controller) {
            return Err(CgroupV2Error::ControllerNotEnabled(controller));
        }
    }
    Ok(())
}

fn verify_candidate_membership(
    snapshot: &CgroupV2Snapshot,
    candidate_pid: u32,
) -> Result<(), CgroupV2Error> {
    let mut observed = BTreeSet::new();
    for line in snapshot.cgroup_procs.lines() {
        let pid = line
            .trim()
            .parse::<u32>()
            .map_err(|_| CgroupV2Error::InvalidControlValue("cgroup.procs"))?;
        if pid == 0 || !observed.insert(pid) {
            return Err(CgroupV2Error::InvalidControlValue("cgroup.procs"));
        }
    }
    if !observed.contains(&candidate_pid) {
        return Err(CgroupV2Error::CandidateNotInScope { pid: candidate_pid });
    }

    let events = parse_key_value_lines(&snapshot.cgroup_events, "cgroup.events")?;
    let populated = events
        .iter()
        .find(|(key, _)| key == "populated")
        .map(|(_, value)| value.as_str());
    if populated != Some("1") {
        return Err(CgroupV2Error::ScopeNotPopulated);
    }
    Ok(())
}

/// Verify read-back values against the exact requested policy.
///
/// This consistency check does not prove that the kernel, systemd manager, or
/// procfs view is uncompromised.
pub fn verify_cgroup_v2_snapshot(
    expected: &ExpectedCgroupV2Limits,
    snapshot: &CgroupV2Snapshot,
) -> Result<CgroupV2Attestation, CgroupV2Error> {
    if expected.candidate_pid == 0 {
        return Err(CgroupV2Error::InvalidExpectedLimit("candidate_pid"));
    }
    if expected.memory_max_bytes == 0 {
        return Err(CgroupV2Error::InvalidExpectedLimit("memory_max_bytes"));
    }
    if !(1..=1_600).contains(&expected.cpu_quota_percent) {
        return Err(CgroupV2Error::InvalidExpectedLimit("cpu_quota_percent"));
    }
    if expected.tasks_max == 0 {
        return Err(CgroupV2Error::InvalidExpectedLimit("tasks_max"));
    }
    if !safe_control_group_path(&expected.control_group) {
        return Err(CgroupV2Error::InvalidControlGroupPath);
    }
    if snapshot.control_group != expected.control_group {
        return Err(CgroupV2Error::ScopePathMismatch);
    }

    verify_parent_controllers(snapshot)?;
    verify_candidate_membership(snapshot, expected.candidate_pid)?;

    let memory_max = parse_single_u64(&snapshot.memory_max, "memory.max")?;
    if memory_max != expected.memory_max_bytes {
        return Err(CgroupV2Error::MemoryLimitMismatch {
            expected: expected.memory_max_bytes,
            actual: memory_max,
        });
    }

    let memory_swap_max = parse_single_u64(&snapshot.memory_swap_max, "memory.swap.max")?;
    if memory_swap_max != 0 {
        return Err(CgroupV2Error::SwapLimitMismatch {
            expected: 0,
            actual: memory_swap_max,
        });
    }

    let (cpu_quota_usec, cpu_period_usec) = parse_cpu_max(&snapshot.cpu_max)?;
    if u128::from(cpu_quota_usec) * 100
        != u128::from(cpu_period_usec) * u128::from(expected.cpu_quota_percent)
    {
        return Err(CgroupV2Error::CpuQuotaMismatch {
            expected_percent: expected.cpu_quota_percent,
            actual_quota: cpu_quota_usec,
            actual_period: cpu_period_usec,
        });
    }

    let tasks_max_raw = parse_single_u64(&snapshot.pids_max, "pids.max")?;
    let tasks_max = u32::try_from(tasks_max_raw)
        .map_err(|_| CgroupV2Error::InvalidControlValue("pids.max"))?;
    if tasks_max != expected.tasks_max {
        return Err(CgroupV2Error::TaskLimitMismatch {
            expected: expected.tasks_max,
            actual: tasks_max,
        });
    }

    if snapshot.memory_oom_group.trim() != "1" {
        return Err(CgroupV2Error::OomGroupNotEnabled);
    }

    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.cgroup-v2-attestation.v1");
    hasher.bytes(snapshot.control_group.as_bytes());
    hasher.u32(expected.candidate_pid);
    hasher.u64(memory_max);
    hasher.u64(memory_swap_max);
    hasher.u64(cpu_quota_usec);
    hasher.u64(cpu_period_usec);
    hasher.u32(tasks_max);
    hasher.u8(1);

    Ok(CgroupV2Attestation {
        control_group: snapshot.control_group.clone(),
        candidate_pid: expected.candidate_pid,
        memory_max_bytes: memory_max,
        memory_swap_max_bytes: memory_swap_max,
        cpu_quota_usec,
        cpu_period_usec,
        tasks_max,
        oom_group: true,
        limits_sha256: hasher.finish(),
    })
}

/// Read the host cgroup mount and exact systemd scope, then verify all limits.
pub fn read_and_verify_cgroup_v2(
    mountinfo: &str,
    cgroup_mount: &Path,
    expected: &ExpectedCgroupV2Limits,
) -> Result<CgroupV2Attestation, CgroupV2Error> {
    verify_cgroup2_mount(mountinfo, cgroup_mount)?;
    let snapshot = read_cgroup_v2_snapshot(cgroup_mount, &expected.control_group)?;
    verify_cgroup_v2_snapshot(expected, &snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected() -> ExpectedCgroupV2Limits {
        ExpectedCgroupV2Limits {
            control_group: "/user.slice/user-1000.slice/user@1000.service/app.slice/ceb-candidate-test.scope".into(),
            candidate_pid: 420,
            memory_max_bytes: 128 * 1024 * 1024,
            cpu_quota_percent: 100,
            tasks_max: 16,
        }
    }

    fn snapshot() -> CgroupV2Snapshot {
        CgroupV2Snapshot {
            control_group: expected().control_group,
            parent_controllers: "cpu io memory pids".into(),
            parent_subtree_control: "cpu memory pids".into(),
            memory_max: "134217728".into(),
            memory_swap_max: "0".into(),
            cpu_max: "100000 100000".into(),
            pids_max: "16".into(),
            memory_oom_group: "1".into(),
            cgroup_procs: "420\n421\n".into(),
            cgroup_events: "populated 1\nfrozen 0\n".into(),
        }
    }

    fn systemd_show_output() -> String {
        format!(
            "Id=ceb-candidate-test.scope\nActiveState=active\nSubState=running\nControlGroup={}\nMemoryMax=134217728\nMemorySwapMax=0\nCPUQuotaPerSecUSec=1s\nTasksMax=16\nRuntimeMaxUSec=30s\nOOMPolicy=kill\n",
            expected().control_group
        )
    }

    fn expected_systemd() -> ExpectedSystemdScopeProperties {
        ExpectedSystemdScopeProperties {
            unit_name: "ceb-candidate-test.scope".into(),
            control_group: expected().control_group,
            memory_max_bytes: 128 * 1024 * 1024,
            memory_swap_max_bytes: 0,
            cpu_quota_percent: 100,
            tasks_max: 16,
            runtime_max_usec: 30_000_000,
        }
    }

    #[test]
    fn systemd_timespan_parser_handles_documented_integer_units_exactly() {
        assert_eq!(parse_systemd_timespan_usec("190ms", "test"), Ok(190_000));
        assert_eq!(parse_systemd_timespan_usec("1s", "test"), Ok(1_000_000));
        assert_eq!(parse_systemd_timespan_usec("1min 30s", "test"), Ok(90_000_000));
        assert_eq!(parse_systemd_timespan_usec("1s 500ms", "test"), Ok(1_500_000));
        assert_eq!(parse_systemd_timespan_usec("1000us", "test"), Ok(1_000));
    }

    #[test]
    fn systemd_timespan_parser_rejects_unbounded_malformed_and_inexact_values() {
        for invalid in ["", "infinity", "1.5s", "1fortnight", "1ns", "1s garbage"] {
            assert_eq!(
                parse_systemd_timespan_usec(invalid, "RuntimeMaxUSec"),
                Err(CgroupV2Error::InvalidSystemdProperty("RuntimeMaxUSec")),
                "unexpectedly accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn systemd_properties_match_the_cgroup_policy() {
        let observed = parse_systemd_scope_properties(&systemd_show_output()).unwrap();
        assert_eq!(verify_systemd_scope_properties(&expected_systemd(), &observed), Ok(()));
    }

    #[test]
    fn systemd_properties_reject_unit_scope_and_resource_drift() {
        let expected_policy = expected_systemd();
        let mut observed = parse_systemd_scope_properties(&systemd_show_output()).unwrap();
        observed.unit_name = "other.scope".into();
        assert_eq!(
            verify_systemd_scope_properties(&expected_policy, &observed),
            Err(CgroupV2Error::SystemdPropertyMismatch("Id"))
        );

        observed = parse_systemd_scope_properties(&systemd_show_output()).unwrap();
        observed.control_group = "/user.slice/other.scope".into();
        assert_eq!(
            verify_systemd_scope_properties(&expected_policy, &observed),
            Err(CgroupV2Error::SystemdPropertyMismatch("ControlGroup"))
        );

        observed = parse_systemd_scope_properties(&systemd_show_output()).unwrap();
        observed.cpu_quota_per_sec_usec = 500_000;
        assert_eq!(
            verify_systemd_scope_properties(&expected_policy, &observed),
            Err(CgroupV2Error::SystemdPropertyMismatch("CPUQuotaPerSecUSec"))
        );

        observed = parse_systemd_scope_properties(&systemd_show_output()).unwrap();
        observed.active_state = "inactive".into();
        assert_eq!(
            verify_systemd_scope_properties(&expected_policy, &observed),
            Err(CgroupV2Error::SystemdScopeNotActive)
        );
    }

    #[test]
    fn systemd_scope_policy_binds_unit_id_to_control_group_basename() {
        let mut policy = expected_systemd();
        policy.unit_name = "another.scope".into();
        let observed = parse_systemd_scope_properties(&systemd_show_output()).unwrap();
        assert_eq!(
            verify_systemd_scope_properties(&policy, &observed),
            Err(CgroupV2Error::InvalidExpectedLimit("systemd_scope_properties"))
        );
    }

    #[test]
    fn systemd_show_response_is_bounded() {
        let oversized = "x".repeat(MAX_SYSTEMD_SHOW_BYTES + 1);
        assert_eq!(
            parse_systemd_scope_properties(&oversized),
            Err(CgroupV2Error::SystemdPropertyParse)
        );
    }

    #[test]
    fn systemd_show_rejects_duplicate_missing_and_infinite_properties() {
        let duplicate = format!("{}Id=other.scope\n", systemd_show_output());
        assert_eq!(
            parse_systemd_scope_properties(&duplicate),
            Err(CgroupV2Error::DuplicateSystemdProperty("Id"))
        );

        let missing = systemd_show_output().replace("OOMPolicy=kill\n", "");
        assert_eq!(
            parse_systemd_scope_properties(&missing),
            Err(CgroupV2Error::MissingSystemdProperty("OOMPolicy"))
        );

        let infinite = systemd_show_output().replace("MemoryMax=134217728", "MemoryMax=infinity");
        assert_eq!(
            parse_systemd_scope_properties(&infinite),
            Err(CgroupV2Error::InvalidSystemdProperty("MemoryMax"))
        );
    }

    #[test]
    fn exact_readback_qualifies_the_requested_cgroup_policy() {
        let proof = verify_cgroup_v2_snapshot(&expected(), &snapshot()).unwrap();
        assert_eq!(proof.memory_max_bytes, 134_217_728);
        assert_eq!(proof.memory_swap_max_bytes, 0);
        assert_eq!(proof.cpu_quota_usec, 100_000);
        assert_eq!(proof.cpu_period_usec, 100_000);
        assert_eq!(proof.tasks_max, 16);
        assert!(proof.oom_group);
        assert_ne!(proof.limits_sha256, [0; 32]);
    }

    #[test]
    fn scope_identity_must_match_the_systemd_control_group() {
        let mut observed = snapshot();
        observed.control_group = "/user.slice/other.scope".into();
        assert_eq!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::ScopePathMismatch)
        );
    }

    #[test]
    fn memory_limit_mismatch_is_rejected() {
        let mut observed = snapshot();
        observed.memory_max = "max".into();
        assert!(matches!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::InvalidControlValue("memory.max"))
        ));
        observed.memory_max = "268435456".into();
        assert!(matches!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::MemoryLimitMismatch { .. })
        ));
    }

    #[test]
    fn swap_must_be_disabled() {
        let mut observed = snapshot();
        observed.memory_swap_max = "1".into();
        assert!(matches!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::SwapLimitMismatch { .. })
        ));
    }

    #[test]
    fn cpu_must_be_finite_and_match_the_requested_ratio() {
        let mut observed = snapshot();
        observed.cpu_max = "max 100000".into();
        assert!(matches!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::InvalidControlValue("cpu.max"))
        ));
        observed.cpu_max = "50000 100000".into();
        assert!(matches!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::CpuQuotaMismatch { .. })
        ));
    }

    #[test]
    fn task_limit_must_match_exactly() {
        let mut observed = snapshot();
        observed.pids_max = "max".into();
        assert!(matches!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::InvalidControlValue("pids.max"))
        ));
        observed.pids_max = "128".into();
        assert!(matches!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::TaskLimitMismatch { .. })
        ));
    }

    #[test]
    fn parent_controllers_must_exist_and_be_enabled() {
        let mut observed = snapshot();
        observed.parent_controllers = "cpu memory".into();
        assert_eq!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::MissingController("pids"))
        );
        observed = snapshot();
        observed.parent_subtree_control = "cpu memory".into();
        assert_eq!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::ControllerNotEnabled("pids"))
        );
    }

    #[test]
    fn oom_group_kill_policy_must_be_effective() {
        let mut observed = snapshot();
        observed.memory_oom_group = "0".into();
        assert_eq!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::OomGroupNotEnabled)
        );
    }

    #[test]
    fn candidate_must_be_in_the_scope_and_the_scope_must_be_populated() {
        let mut observed = snapshot();
        observed.cgroup_procs = "421\n".into();
        assert_eq!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::CandidateNotInScope { pid: 420 })
        );
        observed = snapshot();
        observed.cgroup_events = "populated 0\nfrozen 0\n".into();
        assert_eq!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::ScopeNotPopulated)
        );
    }

    #[test]
    fn control_group_paths_reject_root_traversal_and_noncanonical_segments() {
        for invalid in [
            "/",
            "//user.slice/job.scope",
            "/user.slice/../secret.scope",
            "/user.slice/./job.scope",
            "/user.slice//job.scope",
            "/user.slice/job.scope/",
        ] {
            let mut observed = snapshot();
            observed.control_group = invalid.into();
            let mut policy = expected();
            policy.control_group = invalid.into();
            assert_eq!(
                verify_cgroup_v2_snapshot(&policy, &observed),
                Err(CgroupV2Error::InvalidControlGroupPath)
            );
        }
    }

    #[test]
    fn only_exact_systemd_scope_paths_are_accepted() {
        for invalid in [
            "/user.slice/ceb-candidate.service",
            "/user.slice/ceb-candidate.scope/child",
        ] {
            let mut observed = snapshot();
            observed.control_group = invalid.into();
            let mut policy = expected();
            policy.control_group = invalid.into();
            assert_eq!(
                verify_cgroup_v2_snapshot(&policy, &observed),
                Err(CgroupV2Error::InvalidControlGroupPath)
            );
        }
    }

    #[test]
    fn oversized_control_file_is_rejected_before_full_read() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "symthaea-cgroup-control-{}-{nonce}",
            std::process::id()
        ));
        fs::write(&path, vec![b'x'; (MAX_CGROUP_CONTROL_FILE_BYTES + 1) as usize]).unwrap();
        let result = read_control_file(&path);
        let _ = fs::remove_file(&path);
        assert!(matches!(result, Err(CgroupV2Error::ControlFileTooLarge { .. })));
    }

    #[test]
    fn mountinfo_requires_a_real_unified_cgroup2_mount() {
        let mountinfo = "36 25 0:32 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n";
        assert_eq!(
            verify_cgroup2_mount(mountinfo, Path::new("/sys/fs/cgroup")),
            Ok(())
        );
        assert_eq!(
            verify_cgroup2_mount(mountinfo, Path::new("/sys/fs/cgroup-v1")),
            Err(CgroupV2Error::Cgroup2MountNotFound)
        );
    }

    #[test]
    fn mountinfo_rejects_subtree_mounts_and_decodes_escaped_mountpoints() {
        let subtree = "36 25 0:32 /user.slice /sys/fs/cgroup rw - cgroup2 cgroup rw\n";
        assert_eq!(
            verify_cgroup2_mount(subtree, Path::new("/sys/fs/cgroup")),
            Err(CgroupV2Error::Cgroup2MountRootNotUnified)
        );
        let escaped = "36 25 0:32 / /sys/fs/group\\040root rw - cgroup2 cgroup rw\n";
        assert_eq!(
            verify_cgroup2_mount(escaped, Path::new("/sys/fs/group root")),
            Ok(())
        );
    }

    #[test]
    fn mountinfo_rejects_octal_escapes_outside_single_byte_range() {
        let invalid = "36 25 0:32 / /sys/fs/group\\777 rw - cgroup2 cgroup rw\\n";
        assert_eq!(
            verify_cgroup2_mount(invalid, Path::new("/sys/fs/group")),
            Err(CgroupV2Error::InvalidMountInfoEscape)
        );
    }

    #[test]
    fn duplicate_or_malformed_cgroup_event_fields_fail_closed() {
        let mut observed = snapshot();
        observed.cgroup_events = "populated 1\npopulated 1\n".into();
        assert_eq!(
            verify_cgroup_v2_snapshot(&expected(), &observed),
            Err(CgroupV2Error::InvalidControlValue("cgroup.events"))
        );
    }
}
