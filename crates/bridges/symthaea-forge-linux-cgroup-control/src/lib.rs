// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Host-side cgroup v2 resource enforcement for Forge evaluator sandboxes.
//!
//! This bridge assumes a system manager has already delegated a non-root cgroup v2 subtree with
//! `cpu`, `memory`, and `pids` enabled for children. It deliberately does not mutate the parent's
//! `cgroup.subtree_control`: delegation and top-down controller ownership remain host policy.
//!
//! A successful lease creates one leaf cgroup, writes bounded `memory.max`, `pids.max`, and
//! `cpu.max`, migrates the exact host PID through `cgroup.procs`, then reads the controller files and
//! `/proc/<pid>/cgroup` back before issuing a content-addressed receipt. This proves those limits were
//! configured on that leaf at admission time; it does not claim IO limits, swap limits, PSI quality,
//! OOM behavior, or that ancestors are less restrictive.

use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use symthaea_algorithms::ContentId;
use thiserror::Error;

const CGROUP_V2_ROOT: &str = "/sys/fs/cgroup";
const REQUIRED_CONTROLLERS: [&str; 3] = ["cpu", "memory", "pids"];
const MIN_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MEMORY_BYTES: u64 = 256 * 1024 * 1024 * 1024;
const MAX_PIDS: u64 = 8192;
const MIN_CPU_PERIOD_US: u64 = 1_000;
const MAX_CPU_PERIOD_US: u64 = 1_000_000;
const MAX_CPU_EQUIVALENTS: u64 = 256;

static LEAF_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum CgroupV2ResourceError {
    #[error("cgroup v2 resource control is supported only on Linux")]
    UnsupportedPlatform,
    #[error("invalid cgroup v2 resource policy")]
    InvalidPolicy,
    #[error("host PID must be nonzero")]
    InvalidPid,
    #[error("delegation root is outside /sys/fs/cgroup: {0:?}")]
    OutsideCgroupRoot(PathBuf),
    #[error("the global cgroup root is not accepted as an evaluator delegation root")]
    GlobalRootNotDelegated,
    #[error("required cgroup v2 controller is not available: {0}")]
    ControllerUnavailable(&'static str),
    #[error("required cgroup v2 controller is not enabled for children: {0}")]
    ControllerNotDelegated(&'static str),
    #[error("cgroup IO error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cgroup controller value is malformed: {0}")]
    MalformedControllerValue(&'static str),
    #[error("written cgroup resource limits did not read back exactly")]
    LimitReadbackMismatch,
    #[error("sandbox PID is not observed in the evaluator leaf cgroup")]
    MembershipMismatch,
    #[error("failed to roll sandbox PID back to the delegated parent after admission failure")]
    RollbackFailed,
    #[error("resource receipt identity does not match canonical fields")]
    ReceiptIdentityMismatch,
    #[error("cannot remove a populated evaluator cgroup")]
    CgroupStillPopulated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CgroupV2ResourcePolicy {
    id: ContentId,
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
}

impl CgroupV2ResourcePolicy {
    pub fn new(
        memory_max_bytes: u64,
        pids_max: u64,
        cpu_quota_us: u64,
        cpu_period_us: u64,
    ) -> Result<Self, CgroupV2ResourceError> {
        if memory_max_bytes < MIN_MEMORY_BYTES
            || memory_max_bytes > MAX_MEMORY_BYTES
            || pids_max == 0
            || pids_max > MAX_PIDS
            || cpu_quota_us == 0
            || cpu_period_us < MIN_CPU_PERIOD_US
            || cpu_period_us > MAX_CPU_PERIOD_US
        {
            return Err(CgroupV2ResourceError::InvalidPolicy);
        }
        let max_quota = cpu_period_us
            .checked_mul(MAX_CPU_EQUIVALENTS)
            .ok_or(CgroupV2ResourceError::InvalidPolicy)?;
        if cpu_quota_us > max_quota {
            return Err(CgroupV2ResourceError::InvalidPolicy);
        }
        let id = derive_policy_id(memory_max_bytes, pids_max, cpu_quota_us, cpu_period_us);
        Ok(Self {
            id,
            memory_max_bytes,
            pids_max,
            cpu_quota_us,
            cpu_period_us,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn memory_max_bytes(&self) -> u64 {
        self.memory_max_bytes
    }
    pub fn pids_max(&self) -> u64 {
        self.pids_max
    }
    pub fn cpu_quota_us(&self) -> u64 {
        self.cpu_quota_us
    }
    pub fn cpu_period_us(&self) -> u64 {
        self.cpu_period_us
    }

    pub fn validate(&self) -> Result<(), CgroupV2ResourceError> {
        if Self::new(
            self.memory_max_bytes,
            self.pids_max,
            self.cpu_quota_us,
            self.cpu_period_us,
        )? == *self
        {
            Ok(())
        } else {
            Err(CgroupV2ResourceError::InvalidPolicy)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CgroupV2ResourceReceipt {
    id: ContentId,
    policy_id: ContentId,
    delegation_root: String,
    leaf_path: String,
    host_pid: u32,
    available_controllers: Vec<String>,
    delegated_controllers: Vec<String>,
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
    proc_membership: String,
}

impl CgroupV2ResourceReceipt {
    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn policy_id(&self) -> &ContentId {
        &self.policy_id
    }
    pub fn delegation_root(&self) -> &str {
        &self.delegation_root
    }
    pub fn leaf_path(&self) -> &str {
        &self.leaf_path
    }
    pub fn host_pid(&self) -> u32 {
        self.host_pid
    }
    pub fn available_controllers(&self) -> &[String] {
        &self.available_controllers
    }
    pub fn delegated_controllers(&self) -> &[String] {
        &self.delegated_controllers
    }
    pub fn memory_max_bytes(&self) -> u64 {
        self.memory_max_bytes
    }
    pub fn pids_max(&self) -> u64 {
        self.pids_max
    }
    pub fn cpu_quota_us(&self) -> u64 {
        self.cpu_quota_us
    }
    pub fn cpu_period_us(&self) -> u64 {
        self.cpu_period_us
    }
    pub fn proc_membership(&self) -> &str {
        &self.proc_membership
    }

    pub fn validate_for(
        &self,
        policy: &CgroupV2ResourcePolicy,
    ) -> Result<(), CgroupV2ResourceError> {
        policy.validate()?;
        if self.policy_id != *policy.id()
            || self.memory_max_bytes != policy.memory_max_bytes()
            || self.pids_max != policy.pids_max()
            || self.cpu_quota_us != policy.cpu_quota_us()
            || self.cpu_period_us != policy.cpu_period_us()
            || REQUIRED_CONTROLLERS.iter().any(|controller| {
                !self
                    .available_controllers
                    .iter()
                    .any(|value| value == controller)
            })
            || REQUIRED_CONTROLLERS.iter().any(|controller| {
                !self
                    .delegated_controllers
                    .iter()
                    .any(|value| value == controller)
            })
        {
            return Err(CgroupV2ResourceError::ReceiptIdentityMismatch);
        }
        let expected = derive_receipt_id(
            &self.policy_id,
            &self.delegation_root,
            &self.leaf_path,
            self.host_pid,
            &self.available_controllers,
            &self.delegated_controllers,
            self.memory_max_bytes,
            self.pids_max,
            self.cpu_quota_us,
            self.cpu_period_us,
            &self.proc_membership,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(CgroupV2ResourceError::ReceiptIdentityMismatch)
        }
    }
}

/// Owns one live evaluator leaf. Dropping the lease makes a best-effort removal attempt; callers
/// should use `cleanup()` after the sandbox has exited when they need cleanup evidence.
#[derive(Debug)]
pub struct CgroupV2ResourceLease {
    receipt: CgroupV2ResourceReceipt,
    leaf_path: PathBuf,
    cleaned: bool,
}

impl CgroupV2ResourceLease {
    pub fn receipt(&self) -> &CgroupV2ResourceReceipt {
        &self.receipt
    }
    pub fn leaf_path(&self) -> &Path {
        &self.leaf_path
    }

    pub fn cleanup(mut self) -> Result<CgroupV2ResourceReceipt, CgroupV2ResourceError> {
        ensure_unpopulated(&self.leaf_path)?;
        fs::remove_dir(&self.leaf_path).map_err(|source| CgroupV2ResourceError::Io {
            path: self.leaf_path.clone(),
            source,
        })?;
        self.cleaned = true;
        Ok(self.receipt.clone())
    }
}

impl Drop for CgroupV2ResourceLease {
    fn drop(&mut self) {
        if !self.cleaned {
            let _ = ensure_unpopulated(&self.leaf_path).and_then(|_| {
                fs::remove_dir(&self.leaf_path).map_err(|source| CgroupV2ResourceError::Io {
                    path: self.leaf_path.clone(),
                    source,
                })
            });
        }
    }
}

pub fn apply_cgroup_v2_resource_policy(
    policy: &CgroupV2ResourcePolicy,
    delegation_root: impl AsRef<Path>,
    host_pid: u32,
) -> Result<CgroupV2ResourceLease, CgroupV2ResourceError> {
    if !cfg!(target_os = "linux") {
        return Err(CgroupV2ResourceError::UnsupportedPlatform);
    }
    policy.validate()?;
    if host_pid == 0 {
        return Err(CgroupV2ResourceError::InvalidPid);
    }

    let root = canonical_delegation_root(delegation_root.as_ref())?;
    let available = read_controller_set(&root.join("cgroup.controllers"))?;
    let delegated = read_controller_set(&root.join("cgroup.subtree_control"))?;
    for controller in REQUIRED_CONTROLLERS {
        if !available.iter().any(|value| value == controller) {
            return Err(CgroupV2ResourceError::ControllerUnavailable(controller));
        }
        if !delegated.iter().any(|value| value == controller) {
            return Err(CgroupV2ResourceError::ControllerNotDelegated(controller));
        }
    }

    let nonce = LEAF_NONCE.fetch_add(1, Ordering::Relaxed);
    let leaf = root.join(format!("symthaea-forge-evaluator-{host_pid}-{nonce}"));
    fs::create_dir(&leaf).map_err(|source| CgroupV2ResourceError::Io {
        path: leaf.clone(),
        source,
    })?;

    match configure_and_admit(policy, &root, &leaf, host_pid, &available, &delegated) {
        Ok(receipt) => Ok(CgroupV2ResourceLease {
            receipt,
            leaf_path: leaf,
            cleaned: false,
        }),
        Err(error) => {
            fs::remove_dir(&leaf).map_err(|source| CgroupV2ResourceError::Io {
                path: leaf,
                source,
            })?;
            Err(error)
        }
    }
}

fn configure_and_admit(
    policy: &CgroupV2ResourcePolicy,
    root: &Path,
    leaf: &Path,
    host_pid: u32,
    available: &[String],
    delegated: &[String],
) -> Result<CgroupV2ResourceReceipt, CgroupV2ResourceError> {
    write_value(&leaf.join("memory.max"), &policy.memory_max_bytes().to_string())?;
    write_value(&leaf.join("pids.max"), &policy.pids_max().to_string())?;
    write_value(
        &leaf.join("cpu.max"),
        &format!("{} {}", policy.cpu_quota_us(), policy.cpu_period_us()),
    )?;

    let memory_max = read_u64_value(&leaf.join("memory.max"), "memory.max")?;
    let pids_max = read_u64_value(&leaf.join("pids.max"), "pids.max")?;
    let (cpu_quota, cpu_period) = read_cpu_max(&leaf.join("cpu.max"))?;
    if memory_max != policy.memory_max_bytes()
        || pids_max != policy.pids_max()
        || cpu_quota != policy.cpu_quota_us()
        || cpu_period != policy.cpu_period_us()
    {
        return Err(CgroupV2ResourceError::LimitReadbackMismatch);
    }

    write_value(&leaf.join("cgroup.procs"), &host_pid.to_string())?;

    let post_admission = (|| {
        let members = read_text(&leaf.join("cgroup.procs"))?;
        let process_present = members
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .any(|pid| pid == host_pid);
        let proc_membership = read_proc_membership(host_pid)?;
        let expected_membership = cgroup_membership_for_leaf(leaf)?;
        if !process_present || proc_membership != expected_membership {
            return Err(CgroupV2ResourceError::MembershipMismatch);
        }

        let delegation_root = root.to_string_lossy().into_owned();
        let leaf_path = leaf.to_string_lossy().into_owned();
        let id = derive_receipt_id(
            policy.id(),
            &delegation_root,
            &leaf_path,
            host_pid,
            available,
            delegated,
            memory_max,
            pids_max,
            cpu_quota,
            cpu_period,
            &proc_membership,
        );
        let receipt = CgroupV2ResourceReceipt {
            id,
            policy_id: policy.id().clone(),
            delegation_root,
            leaf_path,
            host_pid,
            available_controllers: available.to_vec(),
            delegated_controllers: delegated.to_vec(),
            memory_max_bytes: memory_max,
            pids_max,
            cpu_quota_us: cpu_quota,
            cpu_period_us: cpu_period,
            proc_membership,
        };
        receipt.validate_for(policy)?;
        Ok(receipt)
    })();

    match post_admission {
        Ok(receipt) => Ok(receipt),
        Err(error) => {
            rollback_pid(root, host_pid)?;
            Err(error)
        }
    }
}

fn canonical_delegation_root(path: &Path) -> Result<PathBuf, CgroupV2ResourceError> {
    let canonical = path
        .canonicalize()
        .map_err(|source| CgroupV2ResourceError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let global = Path::new(CGROUP_V2_ROOT);
    if !canonical.starts_with(global) {
        return Err(CgroupV2ResourceError::OutsideCgroupRoot(canonical));
    }
    if canonical == global {
        return Err(CgroupV2ResourceError::GlobalRootNotDelegated);
    }
    Ok(canonical)
}

fn read_controller_set(path: &Path) -> Result<Vec<String>, CgroupV2ResourceError> {
    let text = read_text(path)?;
    let mut set = BTreeSet::new();
    for value in text.split_whitespace() {
        set.insert(value.trim_start_matches('+').to_string());
    }
    Ok(set.into_iter().collect())
}

fn write_value(path: &Path, value: &str) -> Result<(), CgroupV2ResourceError> {
    fs::write(path, format!("{value}\n")).map_err(|source| CgroupV2ResourceError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_text(path: &Path) -> Result<String, CgroupV2ResourceError> {
    fs::read_to_string(path).map_err(|source| CgroupV2ResourceError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_u64_value(path: &Path, name: &'static str) -> Result<u64, CgroupV2ResourceError> {
    read_text(path)?
        .trim()
        .parse::<u64>()
        .map_err(|_| CgroupV2ResourceError::MalformedControllerValue(name))
}

fn read_cpu_max(path: &Path) -> Result<(u64, u64), CgroupV2ResourceError> {
    let text = read_text(path)?;
    parse_cpu_max(&text)
}

fn parse_cpu_max(text: &str) -> Result<(u64, u64), CgroupV2ResourceError> {
    let mut fields = text.split_whitespace();
    let quota = fields
        .next()
        .ok_or(CgroupV2ResourceError::MalformedControllerValue("cpu.max"))?
        .parse::<u64>()
        .map_err(|_| CgroupV2ResourceError::MalformedControllerValue("cpu.max"))?;
    let period = fields
        .next()
        .ok_or(CgroupV2ResourceError::MalformedControllerValue("cpu.max"))?
        .parse::<u64>()
        .map_err(|_| CgroupV2ResourceError::MalformedControllerValue("cpu.max"))?;
    if fields.next().is_some() {
        return Err(CgroupV2ResourceError::MalformedControllerValue("cpu.max"));
    }
    Ok((quota, period))
}

fn read_proc_membership(host_pid: u32) -> Result<String, CgroupV2ResourceError> {
    let path = PathBuf::from(format!("/proc/{host_pid}/cgroup"));
    let text = read_text(&path)?;
    parse_unified_membership(&text).ok_or(CgroupV2ResourceError::MembershipMismatch)
}

fn parse_unified_membership(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        if hierarchy == "0" && controllers.is_empty() && path.starts_with('/') {
            Some(path.trim().to_string())
        } else {
            None
        }
    })
}

fn cgroup_membership_for_leaf(leaf: &Path) -> Result<String, CgroupV2ResourceError> {
    let relative = leaf
        .strip_prefix(Path::new(CGROUP_V2_ROOT))
        .map_err(|_| CgroupV2ResourceError::OutsideCgroupRoot(leaf.to_path_buf()))?;
    Ok(format!(
        "/{}",
        relative.to_string_lossy().trim_start_matches('/')
    ))
}

fn rollback_pid(root: &Path, host_pid: u32) -> Result<(), CgroupV2ResourceError> {
    write_value(&root.join("cgroup.procs"), &host_pid.to_string())
        .map_err(|_| CgroupV2ResourceError::RollbackFailed)
}

fn ensure_unpopulated(leaf: &Path) -> Result<(), CgroupV2ResourceError> {
    let events = read_text(&leaf.join("cgroup.events"))?;
    let populated = events.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next()) {
            (Some("populated"), Some(value)) => value.parse::<u32>().ok(),
            _ => None,
        }
    });
    if populated == Some(0) {
        Ok(())
    } else {
        Err(CgroupV2ResourceError::CgroupStillPopulated)
    }
}

fn derive_policy_id(
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-v2-resource-policy.v1",
        [
            memory_max_bytes.to_be_bytes().as_slice(),
            pids_max.to_be_bytes().as_slice(),
            cpu_quota_us.to_be_bytes().as_slice(),
            cpu_period_us.to_be_bytes().as_slice(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_receipt_id(
    policy_id: &ContentId,
    delegation_root: &str,
    leaf_path: &str,
    host_pid: u32,
    available_controllers: &[String],
    delegated_controllers: &[String],
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
    proc_membership: &str,
) -> ContentId {
    let mut parts = vec![
        policy_id.as_str().as_bytes().to_vec(),
        delegation_root.as_bytes().to_vec(),
        leaf_path.as_bytes().to_vec(),
        host_pid.to_be_bytes().to_vec(),
        memory_max_bytes.to_be_bytes().to_vec(),
        pids_max.to_be_bytes().to_vec(),
        cpu_quota_us.to_be_bytes().to_vec(),
        cpu_period_us.to_be_bytes().to_vec(),
        proc_membership.as_bytes().to_vec(),
        (available_controllers.len() as u64).to_be_bytes().to_vec(),
    ];
    parts.extend(
        available_controllers
            .iter()
            .map(|value| value.as_bytes().to_vec()),
    );
    parts.push(
        (delegated_controllers.len() as u64)
            .to_be_bytes()
            .to_vec(),
    );
    parts.extend(
        delegated_controllers
            .iter()
            .map(|value| value.as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-cgroup-v2-resource-receipt.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_rejects_unbounded_or_zero_values() {
        assert!(CgroupV2ResourcePolicy::new(0, 1, 10_000, 100_000).is_err());
        assert!(CgroupV2ResourcePolicy::new(MIN_MEMORY_BYTES, 0, 10_000, 100_000).is_err());
        assert!(CgroupV2ResourcePolicy::new(MIN_MEMORY_BYTES, 1, 0, 100_000).is_err());
        assert!(CgroupV2ResourcePolicy::new(MIN_MEMORY_BYTES, 1, 1, 999).is_err());
    }

    #[test]
    fn cpu_max_parser_is_closed_shape() {
        assert_eq!(
            parse_cpu_max("50000 100000\n").unwrap(),
            (50_000, 100_000)
        );
        assert!(parse_cpu_max("max 100000").is_err());
        assert!(parse_cpu_max("50000 100000 extra").is_err());
    }

    #[test]
    fn unified_membership_parser_requires_v2_shape() {
        assert_eq!(
            parse_unified_membership("0::/user.slice/example\n"),
            Some("/user.slice/example".to_string())
        );
        assert_eq!(parse_unified_membership("1:cpu:/legacy\n"), None);
    }

    #[test]
    fn controller_sets_are_canonicalized_by_parser_shape() {
        let mut set = BTreeSet::new();
        for value in "pids cpu memory cpu".split_whitespace() {
            set.insert(value.to_string());
        }
        assert_eq!(
            set.into_iter().collect::<Vec<_>>(),
            vec!["cpu", "memory", "pids"]
        );
    }
}
