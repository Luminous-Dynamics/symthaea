// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parent-owned cgroup v2 resource leases for simulation workers.
//!
//! The host must supply an explicitly delegated cgroup-v2 directory. This crate
//! never discovers, mounts, or seizes a hierarchy. Required controllers must
//! already be enabled for children. A lease creates one fresh child, programs
//! exact hard resource limits, verifies kernel-visible readback, and may then
//! place one known child PID into that exact leaf.
//!
//! A lease is resource-containment state, not admission or execution authority.
//! Persisted [`CgroupV2Evidence`] is audit evidence only and cannot recreate a
//! writable lease. Missing delegation is an error, never permission to silently
//! substitute rlimits.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
#[cfg(target_os = "linux")]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const CGROUP_V2_PROFILE_V1: &str = "symthaea.simulation.worker-cgroup.v2-v1";
const EVIDENCE_DOMAIN_V1: &[u8] = b"symthaea.simulation.worker-cgroup.v2-v1\0";
const REQUIRED_CONTROLLERS: [&str; 3] = ["cpu", "memory", "pids"];
const MIN_CPU_PERIOD_US: u64 = 1_000;
const MAX_CPU_PERIOD_US: u64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CgroupV2Limits {
    pub memory_max_bytes: u64,
    pub pids_max: u64,
    pub cpu_quota_us: u64,
    pub cpu_period_us: u64,
}

impl Default for CgroupV2Limits {
    fn default() -> Self {
        Self {
            memory_max_bytes: 2 * 1024 * 1024 * 1024,
            pids_max: 64,
            cpu_quota_us: 100_000,
            cpu_period_us: 100_000,
        }
    }
}

impl CgroupV2Limits {
    pub fn validate(self) -> Result<Self, CgroupV2Error> {
        if self.memory_max_bytes == 0
            || self.pids_max == 0
            || self.cpu_quota_us == 0
            || !(MIN_CPU_PERIOD_US..=MAX_CPU_PERIOD_US).contains(&self.cpu_period_us)
        {
            return Err(CgroupV2Error::InvalidLimits);
        }
        Ok(self)
    }

    fn cpu_max(self) -> String {
        format!("{} {}", self.cpu_quota_us, self.cpu_period_us)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CgroupV2Evidence {
    pub profile: String,
    pub delegated_root: String,
    pub delegated_root_device: u64,
    pub delegated_root_inode: u64,
    pub leaf: String,
    pub leaf_device: u64,
    pub leaf_inode: u64,
    pub controllers: Vec<String>,
    pub memory_max_bytes: u64,
    pub pids_max: u64,
    pub cpu_quota_us: u64,
    pub cpu_period_us: u64,
    pub memory_oom_group: bool,
    pub evidence_sha256: String,
}

impl CgroupV2Evidence {
    /// Verify internal canonical structure and digest only. This establishes no
    /// writable cgroup authority and does not inspect current kernel state.
    pub fn verify(&self) -> Result<(), CgroupV2Error> {
        if self.profile != CGROUP_V2_PROFILE_V1
            || self.controllers != canonical_controllers()
            || !self.memory_oom_group
        {
            return Err(CgroupV2Error::InvalidEvidence);
        }
        CgroupV2Limits {
            memory_max_bytes: self.memory_max_bytes,
            pids_max: self.pids_max,
            cpu_quota_us: self.cpu_quota_us,
            cpu_period_us: self.cpu_period_us,
        }
        .validate()?;
        let expected = hex_digest(evidence_sha256_v1(self)?);
        if self.evidence_sha256 != expected {
            return Err(CgroupV2Error::EvidenceDigestMismatch);
        }
        Ok(())
    }
}

/// Live host-owned lease on one exact cgroup-v2 leaf.
///
/// Deliberately non-serializable and non-cloneable.
#[derive(Debug)]
pub struct CgroupV2Lease {
    delegated_root: PathBuf,
    delegated_root_device: u64,
    delegated_root_inode: u64,
    leaf: PathBuf,
    leaf_device: u64,
    leaf_inode: u64,
    limits: CgroupV2Limits,
}

impl CgroupV2Lease {
    pub fn create(
        delegated_root: impl AsRef<Path>,
        leaf_name: &str,
        limits: CgroupV2Limits,
    ) -> Result<Self, CgroupV2Error> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (delegated_root, leaf_name, limits);
            Err(CgroupV2Error::UnsupportedPlatform)
        }

        #[cfg(target_os = "linux")]
        {
            let limits = limits.validate()?;
            validate_leaf_name(leaf_name)?;
            let delegated_root = fs::canonicalize(delegated_root.as_ref())
                .map_err(CgroupV2Error::DelegatedRoot)?;
            require_utf8_path(&delegated_root)?;
            require_control_file(&delegated_root.join("cgroup.controllers"))?;
            require_control_file(&delegated_root.join("cgroup.subtree_control"))?;
            require_control_file(&delegated_root.join("cgroup.procs"))?;

            let available = read_words(&delegated_root.join("cgroup.controllers"))?;
            let enabled = read_words(&delegated_root.join("cgroup.subtree_control"))?;
            for controller in REQUIRED_CONTROLLERS {
                if !available.contains(controller) {
                    return Err(CgroupV2Error::ControllerUnavailable(controller));
                }
                if !enabled.contains(controller) {
                    return Err(CgroupV2Error::ControllerNotEnabled(controller));
                }
            }

            let root_metadata = fs::metadata(&delegated_root).map_err(CgroupV2Error::RootMetadata)?;
            let root_device = root_metadata.dev();
            let root_inode = root_metadata.ino();
            let leaf = delegated_root.join(leaf_name);
            fs::create_dir(&leaf).map_err(CgroupV2Error::CreateLeaf)?;
            let canonical_leaf = match fs::canonicalize(&leaf) {
                Ok(path) => path,
                Err(error) => {
                    let _ = fs::remove_dir(&leaf);
                    return Err(CgroupV2Error::LeafMetadata(error));
                }
            };

            let result = Self::configure_created_leaf(
                delegated_root,
                root_device,
                root_inode,
                canonical_leaf.clone(),
                limits,
            );
            if result.is_err() {
                // Safe best-effort rollback: no PID is written by configuration.
                let _ = fs::remove_dir(&canonical_leaf);
            }
            result
        }
    }

    #[cfg(target_os = "linux")]
    fn configure_created_leaf(
        delegated_root: PathBuf,
        delegated_root_device: u64,
        delegated_root_inode: u64,
        leaf: PathBuf,
        limits: CgroupV2Limits,
    ) -> Result<Self, CgroupV2Error> {
        require_utf8_path(&leaf)?;
        for file in [
            "cgroup.procs",
            "cgroup.events",
            "memory.max",
            "memory.oom.group",
            "pids.max",
            "cpu.max",
        ] {
            require_control_file(&leaf.join(file))?;
        }

        write_control(&leaf.join("memory.max"), &limits.memory_max_bytes.to_string())?;
        write_control(&leaf.join("memory.oom.group"), "1")?;
        write_control(&leaf.join("pids.max"), &limits.pids_max.to_string())?;
        write_control(&leaf.join("cpu.max"), &limits.cpu_max())?;

        let metadata = fs::metadata(&leaf).map_err(CgroupV2Error::LeafMetadata)?;
        let lease = Self {
            delegated_root,
            delegated_root_device,
            delegated_root_inode,
            leaf,
            leaf_device: metadata.dev(),
            leaf_inode: metadata.ino(),
            limits,
        };
        lease.verify_configuration()?;
        Ok(lease)
    }

    pub fn limits(&self) -> CgroupV2Limits {
        self.limits
    }

    pub fn leaf(&self) -> &Path {
        &self.leaf
    }

    pub fn delegated_root(&self) -> &Path {
        &self.delegated_root
    }

    pub fn verify_configuration(&self) -> Result<(), CgroupV2Error> {
        #[cfg(not(target_os = "linux"))]
        {
            Err(CgroupV2Error::UnsupportedPlatform)
        }

        #[cfg(target_os = "linux")]
        {
            let root = fs::metadata(&self.delegated_root).map_err(CgroupV2Error::RootMetadata)?;
            if root.dev() != self.delegated_root_device || root.ino() != self.delegated_root_inode {
                return Err(CgroupV2Error::RootIdentityChanged);
            }
            let leaf = fs::metadata(&self.leaf).map_err(CgroupV2Error::LeafMetadata)?;
            if leaf.dev() != self.leaf_device || leaf.ino() != self.leaf_inode {
                return Err(CgroupV2Error::LeafIdentityChanged);
            }
            require_exact_u64(
                &self.leaf.join("memory.max"),
                self.limits.memory_max_bytes,
                "memory.max",
            )?;
            require_exact_u64(
                &self.leaf.join("pids.max"),
                self.limits.pids_max,
                "pids.max",
            )?;
            if read_trimmed(&self.leaf.join("cpu.max"))? != self.limits.cpu_max() {
                return Err(CgroupV2Error::ReadbackMismatch("cpu.max"));
            }
            if read_trimmed(&self.leaf.join("memory.oom.group"))? != "1" {
                return Err(CgroupV2Error::ReadbackMismatch("memory.oom.group"));
            }
            Ok(())
        }
    }

    /// Move the entire process containing `pid` into this leaf and verify that
    /// the target cgroup reports it afterwards.
    pub fn place_pid(&self, pid: u32) -> Result<(), CgroupV2Error> {
        if pid == 0 {
            return Err(CgroupV2Error::InvalidPid);
        }
        self.verify_configuration()?;
        write_control(&self.leaf.join("cgroup.procs"), &pid.to_string())?;
        if !self.contains_pid(pid)? {
            return Err(CgroupV2Error::PlacementNotObserved(pid));
        }
        self.verify_configuration()?;
        Ok(())
    }

    pub fn contains_pid(&self, pid: u32) -> Result<bool, CgroupV2Error> {
        let contents = fs::read_to_string(self.leaf.join("cgroup.procs"))
            .map_err(CgroupV2Error::ReadControl)?;
        Ok(contents
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .any(|candidate| candidate == pid))
    }

    pub fn populated(&self) -> Result<bool, CgroupV2Error> {
        let contents = fs::read_to_string(self.leaf.join("cgroup.events"))
            .map_err(CgroupV2Error::ReadControl)?;
        for line in contents.lines() {
            let mut fields = line.split_whitespace();
            if fields.next() == Some("populated") {
                return match fields.next() {
                    Some("0") => Ok(false),
                    Some("1") => Ok(true),
                    _ => Err(CgroupV2Error::InvalidEvents),
                };
            }
        }
        Err(CgroupV2Error::InvalidEvents)
    }

    /// Explicit emergency termination. Never invoked implicitly from `Drop`.
    pub fn kill_all(&self) -> Result<(), CgroupV2Error> {
        self.verify_configuration()?;
        let path = self.leaf.join("cgroup.kill");
        if !path.exists() {
            return Err(CgroupV2Error::KillUnavailable);
        }
        write_control(&path, "1")
    }

    /// Remove an already-unpopulated leaf. Never kills or migrates tasks.
    pub fn remove_empty(self) -> Result<(), CgroupV2Error> {
        self.verify_configuration()?;
        if self.populated()? {
            return Err(CgroupV2Error::StillPopulated);
        }
        fs::remove_dir(&self.leaf).map_err(CgroupV2Error::RemoveLeaf)
    }

    pub fn evidence(&self) -> Result<CgroupV2Evidence, CgroupV2Error> {
        self.verify_configuration()?;
        let mut evidence = CgroupV2Evidence {
            profile: CGROUP_V2_PROFILE_V1.into(),
            delegated_root: require_utf8_path(&self.delegated_root)?.to_owned(),
            delegated_root_device: self.delegated_root_device,
            delegated_root_inode: self.delegated_root_inode,
            leaf: require_utf8_path(&self.leaf)?.to_owned(),
            leaf_device: self.leaf_device,
            leaf_inode: self.leaf_inode,
            controllers: canonical_controllers(),
            memory_max_bytes: self.limits.memory_max_bytes,
            pids_max: self.limits.pids_max,
            cpu_quota_us: self.limits.cpu_quota_us,
            cpu_period_us: self.limits.cpu_period_us,
            memory_oom_group: true,
            evidence_sha256: String::new(),
        };
        evidence.evidence_sha256 = hex_digest(evidence_sha256_v1(&evidence)?);
        evidence.verify()?;
        Ok(evidence)
    }
}

fn canonical_controllers() -> Vec<String> {
    REQUIRED_CONTROLLERS.into_iter().map(str::to_owned).collect()
}

fn validate_leaf_name(value: &str) -> Result<(), CgroupV2Error> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-'))
        || matches!(value, "." | "..")
    {
        return Err(CgroupV2Error::InvalidLeafName);
    }
    Ok(())
}

fn require_utf8_path(path: &Path) -> Result<&str, CgroupV2Error> {
    path.to_str().ok_or(CgroupV2Error::NonUtf8Path)
}

fn require_control_file(path: &Path) -> Result<(), CgroupV2Error> {
    let metadata = fs::metadata(path).map_err(CgroupV2Error::ReadControl)?;
    if !metadata.is_file() {
        return Err(CgroupV2Error::NotCgroupV2);
    }
    Ok(())
}

fn read_words(path: &Path) -> Result<BTreeSet<String>, CgroupV2Error> {
    Ok(read_trimmed(path)?
        .split_whitespace()
        .map(str::to_owned)
        .collect())
}

fn read_trimmed(path: &Path) -> Result<String, CgroupV2Error> {
    fs::read_to_string(path)
        .map(|value| value.trim().to_owned())
        .map_err(CgroupV2Error::ReadControl)
}

fn write_control(path: &Path, value: &str) -> Result<(), CgroupV2Error> {
    let mut file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(CgroupV2Error::WriteControl)?;
    file.write_all(value.as_bytes())
        .map_err(CgroupV2Error::WriteControl)?;
    file.flush().map_err(CgroupV2Error::WriteControl)
}

fn require_exact_u64(
    path: &Path,
    expected: u64,
    field: &'static str,
) -> Result<(), CgroupV2Error> {
    let actual = read_trimmed(path)?
        .parse::<u64>()
        .map_err(|_| CgroupV2Error::ReadbackMismatch(field))?;
    if actual != expected {
        return Err(CgroupV2Error::ReadbackMismatch(field));
    }
    Ok(())
}

fn evidence_sha256_v1(evidence: &CgroupV2Evidence) -> Result<[u8; 32], CgroupV2Error> {
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_DOMAIN_V1);
    put_string(&mut hasher, &evidence.profile)?;
    put_string(&mut hasher, &evidence.delegated_root)?;
    hasher.update(evidence.delegated_root_device.to_le_bytes());
    hasher.update(evidence.delegated_root_inode.to_le_bytes());
    put_string(&mut hasher, &evidence.leaf)?;
    hasher.update(evidence.leaf_device.to_le_bytes());
    hasher.update(evidence.leaf_inode.to_le_bytes());
    let count = u64::try_from(evidence.controllers.len()).map_err(|_| CgroupV2Error::LengthOverflow)?;
    hasher.update(count.to_le_bytes());
    for controller in &evidence.controllers {
        put_string(&mut hasher, controller)?;
    }
    hasher.update(evidence.memory_max_bytes.to_le_bytes());
    hasher.update(evidence.pids_max.to_le_bytes());
    hasher.update(evidence.cpu_quota_us.to_le_bytes());
    hasher.update(evidence.cpu_period_us.to_le_bytes());
    hasher.update([u8::from(evidence.memory_oom_group)]);
    Ok(hasher.finalize().into())
}

fn put_string(hasher: &mut Sha256, value: &str) -> Result<(), CgroupV2Error> {
    let len = u64::try_from(value.len()).map_err(|_| CgroupV2Error::LengthOverflow)?;
    hasher.update(len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(TABLE[(byte >> 4) as usize] as char);
        output.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Debug, Error)]
pub enum CgroupV2Error {
    #[error("cgroup v2 worker leases are supported only on Linux")]
    UnsupportedPlatform,
    #[error("cgroup v2 limits are invalid")]
    InvalidLimits,
    #[error("cgroup leaf name is invalid")]
    InvalidLeafName,
    #[error("cgroup paths must be UTF-8 for canonical evidence")]
    NonUtf8Path,
    #[error("failed to resolve delegated cgroup root: {0}")]
    DelegatedRoot(#[source] io::Error),
    #[error("delegated path does not expose required cgroup-v2 files")]
    NotCgroupV2,
    #[error("required cgroup controller is unavailable: {0}")]
    ControllerUnavailable(&'static str),
    #[error("required cgroup controller is not enabled for children: {0}")]
    ControllerNotEnabled(&'static str),
    #[error("failed creating fresh cgroup leaf: {0}")]
    CreateLeaf(#[source] io::Error),
    #[error("failed reading delegated-root metadata: {0}")]
    RootMetadata(#[source] io::Error),
    #[error("failed reading cgroup-leaf metadata: {0}")]
    LeafMetadata(#[source] io::Error),
    #[error("delegated cgroup root identity changed")]
    RootIdentityChanged,
    #[error("cgroup leaf identity changed")]
    LeafIdentityChanged,
    #[error("failed reading cgroup control state: {0}")]
    ReadControl(#[source] io::Error),
    #[error("failed writing cgroup control state: {0}")]
    WriteControl(#[source] io::Error),
    #[error("cgroup control readback mismatch: {0}")]
    ReadbackMismatch(&'static str),
    #[error("worker PID must be nonzero")]
    InvalidPid,
    #[error("worker PID {0} was not observed in target cgroup after placement")]
    PlacementNotObserved(u32),
    #[error("cgroup.events did not contain a valid populated field")]
    InvalidEvents,
    #[error("cgroup.kill is unavailable")]
    KillUnavailable,
    #[error("cgroup lease is still populated")]
    StillPopulated,
    #[error("failed removing empty cgroup leaf: {0}")]
    RemoveLeaf(#[source] io::Error),
    #[error("serialized cgroup evidence is structurally invalid")]
    InvalidEvidence,
    #[error("serialized cgroup evidence digest does not match")]
    EvidenceDigestMismatch,
    #[error("cgroup evidence length overflow")]
    LengthOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> CgroupV2Evidence {
        let mut evidence = CgroupV2Evidence {
            profile: CGROUP_V2_PROFILE_V1.into(),
            delegated_root: "/sys/fs/cgroup/delegated".into(),
            delegated_root_device: 1,
            delegated_root_inode: 2,
            leaf: "/sys/fs/cgroup/delegated/invocation-1".into(),
            leaf_device: 1,
            leaf_inode: 3,
            controllers: canonical_controllers(),
            memory_max_bytes: 1024,
            pids_max: 8,
            cpu_quota_us: 50_000,
            cpu_period_us: 100_000,
            memory_oom_group: true,
            evidence_sha256: String::new(),
        };
        evidence.evidence_sha256 = hex_digest(evidence_sha256_v1(&evidence).unwrap());
        evidence
    }

    #[test]
    fn rejects_invalid_limits() {
        assert!(matches!(
            CgroupV2Limits {
                memory_max_bytes: 0,
                ..CgroupV2Limits::default()
            }
            .validate(),
            Err(CgroupV2Error::InvalidLimits)
        ));
        assert!(matches!(
            CgroupV2Limits {
                cpu_period_us: 999,
                ..CgroupV2Limits::default()
            }
            .validate(),
            Err(CgroupV2Error::InvalidLimits)
        ));
    }

    #[test]
    fn rejects_path_shaped_or_noncanonical_leaf_names() {
        for name in ["", ".", "..", "a/b", "a b", "a\nb", "é"] {
            assert!(matches!(
                validate_leaf_name(name),
                Err(CgroupV2Error::InvalidLeafName)
            ));
        }
        assert!(validate_leaf_name("simulation-abc_123.v1").is_ok());
    }

    #[test]
    fn evidence_digest_binds_limits_and_identity() {
        let first = evidence();
        first.verify().unwrap();
        let mut changed = first.clone();
        changed.memory_max_bytes += 1;
        assert!(matches!(
            changed.verify(),
            Err(CgroupV2Error::EvidenceDigestMismatch)
        ));
        let mut changed = first;
        changed.leaf_inode += 1;
        assert!(matches!(
            changed.verify(),
            Err(CgroupV2Error::EvidenceDigestMismatch)
        ));
    }

    #[test]
    fn serialized_evidence_cannot_widen_controller_claim() {
        let mut changed = evidence();
        changed.controllers.push("io".into());
        changed.evidence_sha256 = hex_digest(evidence_sha256_v1(&changed).unwrap());
        assert!(matches!(
            changed.verify(),
            Err(CgroupV2Error::InvalidEvidence)
        ));
    }
}
