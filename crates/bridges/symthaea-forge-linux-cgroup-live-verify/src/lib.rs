// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(target_os = "linux")]
//! Read-only live verification of Forge evaluator cgroup-v2 containment.
//!
//! Admission-time receipts prove what was configured when the sandbox entered its evaluator leaf.
//! This bridge answers a narrower later question without mutating policy: are the same resource
//! limits, strict no-swap/OOM hardening, and exact unified-cgroup membership still observable now?
//!
//! The result is intentionally a re-observation theorem, not an independent proof of Linux cgroup
//! semantics, host delegation correctness, IO isolation, PSI behavior, or OOM-selection behavior.

use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use symthaea_algorithms::ContentId;
use symthaea_forge_linux_cgroup_control::{
    CgroupV2ResourceError, CgroupV2ResourcePolicy, CgroupV2ResourceReceipt,
};
use symthaea_forge_linux_cgroup_strict::{StrictCgroupV2Error, StrictCgroupV2Receipt};
use thiserror::Error;

const CGROUP_V2_ROOT: &str = "/sys/fs/cgroup";

#[derive(Debug, Error)]
pub enum CgroupLiveVerifyError {
    #[error(transparent)]
    Resource(#[from] CgroupV2ResourceError),
    #[error(transparent)]
    Strict(#[from] StrictCgroupV2Error),
    #[error("live cgroup verification IO error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("live cgroup controller value is malformed: {0}")]
    Malformed(&'static str),
    #[error("live cgroup resource state no longer matches the frozen receipts")]
    ResourceMismatch,
    #[error("live cgroup strict state no longer matches the frozen strict receipt")]
    StrictMismatch,
    #[error("live cgroup PID membership no longer matches admission evidence")]
    MembershipMismatch,
    #[error("live cgroup verification receipt identity is non-canonical")]
    ReceiptIdentityMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CgroupLiveVerificationReceipt {
    id: ContentId,
    policy_id: ContentId,
    base_receipt_id: ContentId,
    strict_receipt_id: ContentId,
    leaf_path: String,
    host_pid: u32,
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
    swap_max_bytes: u64,
    oom_group: bool,
    proc_membership: String,
}

impl CgroupLiveVerificationReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn base_receipt_id(&self) -> &ContentId { &self.base_receipt_id }
    pub fn strict_receipt_id(&self) -> &ContentId { &self.strict_receipt_id }
    pub fn leaf_path(&self) -> &str { &self.leaf_path }
    pub fn host_pid(&self) -> u32 { self.host_pid }
    pub fn proc_membership(&self) -> &str { &self.proc_membership }

    pub fn validate_for(
        &self,
        policy: &CgroupV2ResourcePolicy,
        base: &CgroupV2ResourceReceipt,
        strict: &StrictCgroupV2Receipt,
    ) -> Result<(), CgroupLiveVerifyError> {
        base.validate_for(policy)?;
        strict.validate_for(policy, base)?;
        if self.policy_id != *policy.id()
            || self.base_receipt_id != *base.id()
            || self.strict_receipt_id != *strict.id()
            || self.leaf_path != base.leaf_path()
            || self.leaf_path != strict.leaf_path()
            || self.host_pid != base.host_pid()
            || self.memory_max_bytes != policy.memory_max_bytes()
            || self.pids_max != policy.pids_max()
            || self.cpu_quota_us != policy.cpu_quota_us()
            || self.cpu_period_us != policy.cpu_period_us()
            || self.swap_max_bytes != strict.swap_max_bytes()
            || self.swap_max_bytes != 0
            || self.oom_group != strict.oom_group()
            || !self.oom_group
            || self.proc_membership != base.proc_membership()
        {
            return Err(CgroupLiveVerifyError::ResourceMismatch);
        }
        let expected = derive_receipt_id(
            &self.policy_id,
            &self.base_receipt_id,
            &self.strict_receipt_id,
            &self.leaf_path,
            self.host_pid,
            self.memory_max_bytes,
            self.pids_max,
            self.cpu_quota_us,
            self.cpu_period_us,
            self.swap_max_bytes,
            self.oom_group,
            &self.proc_membership,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(CgroupLiveVerifyError::ReceiptIdentityMismatch)
        }
    }
}

pub fn verify_live_cgroup_v2(
    policy: &CgroupV2ResourcePolicy,
    base: &CgroupV2ResourceReceipt,
    strict: &StrictCgroupV2Receipt,
) -> Result<CgroupLiveVerificationReceipt, CgroupLiveVerifyError> {
    base.validate_for(policy)?;
    strict.validate_for(policy, base)?;

    let leaf = Path::new(base.leaf_path())
        .canonicalize()
        .map_err(|source| CgroupLiveVerifyError::Io {
            path: PathBuf::from(base.leaf_path()),
            source,
        })?;
    let expected_leaf = Path::new(base.leaf_path());
    if leaf != expected_leaf || !leaf.starts_with(Path::new(CGROUP_V2_ROOT)) {
        return Err(CgroupLiveVerifyError::ResourceMismatch);
    }

    let memory_max = read_u64(&leaf.join("memory.max"), "memory.max")?;
    let pids_max = read_u64(&leaf.join("pids.max"), "pids.max")?;
    let (cpu_quota, cpu_period) = read_cpu_max(&leaf.join("cpu.max"))?;
    if memory_max != policy.memory_max_bytes()
        || pids_max != policy.pids_max()
        || cpu_quota != policy.cpu_quota_us()
        || cpu_period != policy.cpu_period_us()
    {
        return Err(CgroupLiveVerifyError::ResourceMismatch);
    }

    let swap_max = read_u64(&leaf.join("memory.swap.max"), "memory.swap.max")?;
    let oom_group = read_trimmed(&leaf.join("memory.oom.group"))?;
    let oom_group = match oom_group.as_str() {
        "1" => true,
        "0" => false,
        _ => return Err(CgroupLiveVerifyError::Malformed("memory.oom.group")),
    };
    if swap_max != strict.swap_max_bytes() || swap_max != 0 || oom_group != strict.oom_group() || !oom_group {
        return Err(CgroupLiveVerifyError::StrictMismatch);
    }

    let members = read_trimmed(&leaf.join("cgroup.procs"))?;
    let pid_present = members
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .any(|pid| pid == base.host_pid());
    let proc_membership = read_proc_membership(base.host_pid())?;
    if !pid_present || proc_membership != base.proc_membership() {
        return Err(CgroupLiveVerifyError::MembershipMismatch);
    }

    let id = derive_receipt_id(
        policy.id(),
        base.id(),
        strict.id(),
        base.leaf_path(),
        base.host_pid(),
        memory_max,
        pids_max,
        cpu_quota,
        cpu_period,
        swap_max,
        oom_group,
        &proc_membership,
    );
    let receipt = CgroupLiveVerificationReceipt {
        id,
        policy_id: policy.id().clone(),
        base_receipt_id: base.id().clone(),
        strict_receipt_id: strict.id().clone(),
        leaf_path: base.leaf_path().to_string(),
        host_pid: base.host_pid(),
        memory_max_bytes: memory_max,
        pids_max,
        cpu_quota_us: cpu_quota,
        cpu_period_us: cpu_period,
        swap_max_bytes: swap_max,
        oom_group,
        proc_membership,
    };
    receipt.validate_for(policy, base, strict)?;
    Ok(receipt)
}

fn read_trimmed(path: &Path) -> Result<String, CgroupLiveVerifyError> {
    fs::read_to_string(path)
        .map(|value| value.trim().to_string())
        .map_err(|source| CgroupLiveVerifyError::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn read_u64(path: &Path, name: &'static str) -> Result<u64, CgroupLiveVerifyError> {
    read_trimmed(path)?
        .parse::<u64>()
        .map_err(|_| CgroupLiveVerifyError::Malformed(name))
}

fn read_cpu_max(path: &Path) -> Result<(u64, u64), CgroupLiveVerifyError> {
    let text = read_trimmed(path)?;
    let mut fields = text.split_whitespace();
    let quota = fields
        .next()
        .ok_or(CgroupLiveVerifyError::Malformed("cpu.max"))?
        .parse::<u64>()
        .map_err(|_| CgroupLiveVerifyError::Malformed("cpu.max"))?;
    let period = fields
        .next()
        .ok_or(CgroupLiveVerifyError::Malformed("cpu.max"))?
        .parse::<u64>()
        .map_err(|_| CgroupLiveVerifyError::Malformed("cpu.max"))?;
    if fields.next().is_some() {
        return Err(CgroupLiveVerifyError::Malformed("cpu.max"));
    }
    Ok((quota, period))
}

fn read_proc_membership(host_pid: u32) -> Result<String, CgroupLiveVerifyError> {
    let path = PathBuf::from(format!("/proc/{host_pid}/cgroup"));
    let text = fs::read_to_string(&path).map_err(|source| CgroupLiveVerifyError::Io {
        path: path.clone(),
        source,
    })?;
    text.lines()
        .find_map(|line| {
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
        .ok_or(CgroupLiveVerifyError::MembershipMismatch)
}

#[allow(clippy::too_many_arguments)]
fn derive_receipt_id(
    policy_id: &ContentId,
    base_receipt_id: &ContentId,
    strict_receipt_id: &ContentId,
    leaf_path: &str,
    host_pid: u32,
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
    swap_max_bytes: u64,
    oom_group: bool,
    proc_membership: &str,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-v2-live-verification-receipt.v1",
        [
            policy_id.as_str().as_bytes(),
            base_receipt_id.as_str().as_bytes(),
            strict_receipt_id.as_str().as_bytes(),
            leaf_path.as_bytes(),
            host_pid.to_be_bytes().as_slice(),
            memory_max_bytes.to_be_bytes().as_slice(),
            pids_max.to_be_bytes().as_slice(),
            cpu_quota_us.to_be_bytes().as_slice(),
            cpu_period_us.to_be_bytes().as_slice(),
            swap_max_bytes.to_be_bytes().as_slice(),
            &[u8::from(oom_group)],
            proc_membership.as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_identity_changes_with_membership() {
        let policy = ContentId::derive("test-policy", [b"p".as_slice()]);
        let base = ContentId::derive("test-base", [b"b".as_slice()]);
        let strict = ContentId::derive("test-strict", [b"s".as_slice()]);
        let a = derive_receipt_id(
            &policy, &base, &strict, "/sys/fs/cgroup/delegated/eval", 42,
            1024, 4, 50_000, 100_000, 0, true, "/delegated/eval",
        );
        let b = derive_receipt_id(
            &policy, &base, &strict, "/sys/fs/cgroup/delegated/eval", 42,
            1024, 4, 50_000, 100_000, 0, true, "/delegated/other",
        );
        assert_ne!(a, b);
    }
}
