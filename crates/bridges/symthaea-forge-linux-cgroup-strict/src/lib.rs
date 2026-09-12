// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(target_os = "linux")]
//! Strict hardening for a live Forge evaluator cgroup v2 lease.
//!
//! The base resource-control bridge establishes finite `memory.max`, `pids.max`, and `cpu.max` on
//! one delegated evaluator leaf. This bridge strengthens that live leaf by requiring
//! `memory.swap.max = 0` and `memory.oom.group = 1`, then adds an explicit descendant-wide teardown
//! path through `cgroup.kill` with a verified `populated 0` barrier before leaf removal.
//!
//! This remains narrower than a full resource-isolation theorem: ancestor cgroups may be stricter,
//! IO bandwidth/IOPS are not controlled, kernel OOM selection semantics are not independently
//! proven, and controller implementation correctness remains part of the Linux kernel TCB.

use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use symthaea_algorithms::ContentId;
use symthaea_forge_linux_cgroup_control::{
    CgroupV2ResourceError, CgroupV2ResourceLease, CgroupV2ResourcePolicy,
    CgroupV2ResourceReceipt,
};
use thiserror::Error;

const MAX_TEARDOWN_TIMEOUT_MS: u64 = 30_000;
const POLL_INTERVAL_MS: u64 = 10;

#[derive(Debug, Error)]
pub enum StrictCgroupV2Error {
    #[error(transparent)]
    Base(#[from] CgroupV2ResourceError),
    #[error("strict cgroup hardening IO error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("strict cgroup controller value did not read back exactly")]
    ReadbackMismatch,
    #[error("strict cgroup receipt does not match supplied base evidence")]
    ScopeMismatch,
    #[error("strict cgroup receipt identity is non-canonical")]
    ReceiptIdentityMismatch,
    #[error("strict cgroup teardown timeout is invalid")]
    InvalidTeardownTimeout,
    #[error("strict cgroup remained populated after descendant-wide teardown")]
    TeardownUnverified,
    #[error("strict cgroup teardown receipt identity is non-canonical")]
    TeardownIdentityMismatch,
    #[error("strict cgroup lease was already consumed")]
    LeaseConsumed,
    #[error("failed strict hardening could not be cleaned up fail-closed")]
    HardeningCleanupFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StrictCgroupV2Receipt {
    id: ContentId,
    base_receipt_id: ContentId,
    leaf_path: String,
    swap_max_bytes: u64,
    oom_group: bool,
}

impl StrictCgroupV2Receipt {
    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn base_receipt_id(&self) -> &ContentId {
        &self.base_receipt_id
    }
    pub fn leaf_path(&self) -> &str {
        &self.leaf_path
    }
    pub fn swap_max_bytes(&self) -> u64 {
        self.swap_max_bytes
    }
    pub fn oom_group(&self) -> bool {
        self.oom_group
    }

    pub fn validate_for(
        &self,
        base_policy: &CgroupV2ResourcePolicy,
        base_receipt: &CgroupV2ResourceReceipt,
    ) -> Result<(), StrictCgroupV2Error> {
        base_receipt.validate_for(base_policy)?;
        if self.base_receipt_id != *base_receipt.id()
            || self.leaf_path != base_receipt.leaf_path()
            || self.swap_max_bytes != 0
            || !self.oom_group
        {
            return Err(StrictCgroupV2Error::ScopeMismatch);
        }
        let expected = derive_strict_receipt_id(
            &self.base_receipt_id,
            &self.leaf_path,
            self.swap_max_bytes,
            self.oom_group,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(StrictCgroupV2Error::ReceiptIdentityMismatch)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StrictCgroupV2TeardownReceipt {
    id: ContentId,
    strict_receipt_id: ContentId,
    base_receipt_id: ContentId,
    descendant_kill_requested: bool,
    populated_zero_verified: bool,
    wait_ms: u64,
}

impl StrictCgroupV2TeardownReceipt {
    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn strict_receipt_id(&self) -> &ContentId {
        &self.strict_receipt_id
    }
    pub fn base_receipt_id(&self) -> &ContentId {
        &self.base_receipt_id
    }
    pub fn descendant_kill_requested(&self) -> bool {
        self.descendant_kill_requested
    }
    pub fn populated_zero_verified(&self) -> bool {
        self.populated_zero_verified
    }
    pub fn wait_ms(&self) -> u64 {
        self.wait_ms
    }

    pub fn validate_for(
        &self,
        strict: &StrictCgroupV2Receipt,
        base_policy: &CgroupV2ResourcePolicy,
        base_receipt: &CgroupV2ResourceReceipt,
    ) -> Result<(), StrictCgroupV2Error> {
        strict.validate_for(base_policy, base_receipt)?;
        if self.strict_receipt_id != *strict.id()
            || self.base_receipt_id != *base_receipt.id()
            || !self.populated_zero_verified
        {
            return Err(StrictCgroupV2Error::ScopeMismatch);
        }
        let expected = derive_teardown_receipt_id(
            &self.strict_receipt_id,
            &self.base_receipt_id,
            self.descendant_kill_requested,
            self.populated_zero_verified,
            self.wait_ms,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(StrictCgroupV2Error::TeardownIdentityMismatch)
        }
    }
}

/// Owns the underlying live evaluator cgroup until an explicit terminal operation consumes it.
#[derive(Debug)]
pub struct StrictCgroupV2Lease {
    base: Option<CgroupV2ResourceLease>,
    receipt: StrictCgroupV2Receipt,
}

impl StrictCgroupV2Lease {
    pub fn receipt(&self) -> &StrictCgroupV2Receipt {
        &self.receipt
    }
    pub fn leaf_path(&self) -> Option<&Path> {
        self.base.as_ref().map(CgroupV2ResourceLease::leaf_path)
    }

    /// Normal terminal path after the evaluator is already known to have exited.
    pub fn cleanup(mut self) -> Result<StrictCgroupV2TeardownReceipt, StrictCgroupV2Error> {
        let base = self.base.take().ok_or(StrictCgroupV2Error::LeaseConsumed)?;
        let started = Instant::now();
        require_populated_zero(base.leaf_path())?;
        let base_receipt = base.cleanup()?;
        let wait_ms = elapsed_ms(started)?;
        Ok(build_teardown_receipt(
            &self.receipt,
            &base_receipt,
            false,
            wait_ms,
        ))
    }

    /// Terminal path after the exact sandbox process has exited. If no descendants remain, this is
    /// the graceful path. If the leaf is still populated, it automatically upgrades to
    /// descendant-wide `cgroup.kill`, verifies `populated 0`, and records that fact in the receipt.
    pub fn finalize_after_sandbox_exit(
        mut self,
        timeout_ms: u64,
    ) -> Result<StrictCgroupV2TeardownReceipt, StrictCgroupV2Error> {
        if timeout_ms == 0 || timeout_ms > MAX_TEARDOWN_TIMEOUT_MS {
            return Err(StrictCgroupV2Error::InvalidTeardownTimeout);
        }
        let leaf = self
            .base
            .as_ref()
            .ok_or(StrictCgroupV2Error::LeaseConsumed)?
            .leaf_path()
            .to_path_buf();
        let started = Instant::now();
        let descendant_kill_requested = if read_populated(&leaf)? == 0 {
            false
        } else {
            write_exact(&leaf.join("cgroup.kill"), "1\n")?;
            wait_populated_zero(&leaf, Duration::from_millis(timeout_ms))?;
            true
        };
        let base = self.base.take().ok_or(StrictCgroupV2Error::LeaseConsumed)?;
        let base_receipt = base.cleanup()?;
        let wait_ms = elapsed_ms(started)?;
        Ok(build_teardown_receipt(
            &self.receipt,
            &base_receipt,
            descendant_kill_requested,
            wait_ms,
        ))
    }

    /// Fail-closed terminal path for timeout/error handling. `cgroup.kill` targets every process in
    /// the leaf, including descendants that are no longer safely enumerable by PID from userspace.
    pub fn kill_and_cleanup(
        mut self,
        timeout_ms: u64,
    ) -> Result<StrictCgroupV2TeardownReceipt, StrictCgroupV2Error> {
        if timeout_ms == 0 || timeout_ms > MAX_TEARDOWN_TIMEOUT_MS {
            return Err(StrictCgroupV2Error::InvalidTeardownTimeout);
        }
        let base = self.base.take().ok_or(StrictCgroupV2Error::LeaseConsumed)?;
        let leaf = base.leaf_path().to_path_buf();
        write_exact(&leaf.join("cgroup.kill"), "1\n")?;
        let started = Instant::now();
        wait_populated_zero(&leaf, Duration::from_millis(timeout_ms))?;
        let wait_ms = elapsed_ms(started)?;
        let base_receipt = base.cleanup()?;
        Ok(build_teardown_receipt(
            &self.receipt,
            &base_receipt,
            true,
            wait_ms,
        ))
    }
}

impl Drop for StrictCgroupV2Lease {
    fn drop(&mut self) {
        if let Some(base) = self.base.as_ref() {
            let _ = fs::write(base.leaf_path().join("cgroup.kill"), b"1\n");
        }
        // The base lease's Drop remains responsible for best-effort leaf removal.
    }
}

/// Upgrade one live base lease before evaluator release.
///
/// Any hardening failure is fail-closed: the entire evaluator leaf is killed, `populated 0` is
/// verified, and the base leaf is removed before the original hardening error is returned. If that
/// cleanup cannot itself be verified, `HardeningCleanupFailed` becomes the primary error.
pub fn harden_cgroup_v2_lease(
    base_policy: &CgroupV2ResourcePolicy,
    base: CgroupV2ResourceLease,
) -> Result<StrictCgroupV2Lease, StrictCgroupV2Error> {
    let hardening = (|| {
        base.receipt().validate_for(base_policy)?;
        let leaf = base.leaf_path().to_path_buf();
        write_exact(&leaf.join("memory.swap.max"), "0\n")?;
        write_exact(&leaf.join("memory.oom.group"), "1\n")?;

        let swap = read_trimmed(&leaf.join("memory.swap.max"))?;
        let oom = read_trimmed(&leaf.join("memory.oom.group"))?;
        if swap != "0" || oom != "1" {
            return Err(StrictCgroupV2Error::ReadbackMismatch);
        }

        let base_receipt = base.receipt();
        let leaf_path = base_receipt.leaf_path().to_string();
        let id = derive_strict_receipt_id(base_receipt.id(), &leaf_path, 0, true);
        let receipt = StrictCgroupV2Receipt {
            id,
            base_receipt_id: base_receipt.id().clone(),
            leaf_path,
            swap_max_bytes: 0,
            oom_group: true,
        };
        receipt.validate_for(base_policy, base_receipt)?;
        Ok(receipt)
    })();

    match hardening {
        Ok(receipt) => Ok(StrictCgroupV2Lease {
            base: Some(base),
            receipt,
        }),
        Err(error) => {
            if cleanup_failed_hardening(base).is_err() {
                return Err(StrictCgroupV2Error::HardeningCleanupFailed);
            }
            Err(error)
        }
    }
}

fn cleanup_failed_hardening(base: CgroupV2ResourceLease) -> Result<(), StrictCgroupV2Error> {
    let leaf = base.leaf_path().to_path_buf();
    fs::write(leaf.join("cgroup.kill"), b"1\n").map_err(|source| StrictCgroupV2Error::Io {
        path: leaf.join("cgroup.kill"),
        source,
    })?;
    wait_populated_zero(&leaf, Duration::from_millis(MAX_TEARDOWN_TIMEOUT_MS))?;
    base.cleanup()?;
    Ok(())
}

fn build_teardown_receipt(
    strict: &StrictCgroupV2Receipt,
    base: &CgroupV2ResourceReceipt,
    descendant_kill_requested: bool,
    wait_ms: u64,
) -> StrictCgroupV2TeardownReceipt {
    let populated_zero_verified = true;
    let id = derive_teardown_receipt_id(
        strict.id(),
        base.id(),
        descendant_kill_requested,
        populated_zero_verified,
        wait_ms,
    );
    StrictCgroupV2TeardownReceipt {
        id,
        strict_receipt_id: strict.id().clone(),
        base_receipt_id: base.id().clone(),
        descendant_kill_requested,
        populated_zero_verified,
        wait_ms,
    }
}

fn write_exact(path: &Path, value: &str) -> Result<(), StrictCgroupV2Error> {
    fs::write(path, value.as_bytes()).map_err(|source| StrictCgroupV2Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let observed = read_trimmed(path)?;
    if observed != value.trim() {
        return Err(StrictCgroupV2Error::ReadbackMismatch);
    }
    Ok(())
}

fn read_trimmed(path: &Path) -> Result<String, StrictCgroupV2Error> {
    fs::read_to_string(path)
        .map(|value| value.trim().to_string())
        .map_err(|source| StrictCgroupV2Error::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn require_populated_zero(leaf: &Path) -> Result<(), StrictCgroupV2Error> {
    if read_populated(leaf)? == 0 {
        Ok(())
    } else {
        Err(StrictCgroupV2Error::TeardownUnverified)
    }
}

fn wait_populated_zero(leaf: &Path, timeout: Duration) -> Result<(), StrictCgroupV2Error> {
    let deadline = Instant::now() + timeout;
    loop {
        if read_populated(leaf)? == 0 {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(StrictCgroupV2Error::TeardownUnverified);
        }
        thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
}

fn read_populated(leaf: &Path) -> Result<u64, StrictCgroupV2Error> {
    let text = read_trimmed(&leaf.join("cgroup.events"))?;
    parse_populated(&text).ok_or(StrictCgroupV2Error::ReadbackMismatch)
}

fn parse_populated(text: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        if parts.next()? == "populated" {
            parts.next()?.parse().ok()
        } else {
            None
        }
    })
}

fn elapsed_ms(started: Instant) -> Result<u64, StrictCgroupV2Error> {
    u64::try_from(started.elapsed().as_millis()).map_err(|_| StrictCgroupV2Error::ReadbackMismatch)
}

fn derive_strict_receipt_id(
    base_receipt_id: &ContentId,
    leaf_path: &str,
    swap_max_bytes: u64,
    oom_group: bool,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-v2-strict-receipt.v1",
        [
            base_receipt_id.as_str().as_bytes(),
            leaf_path.as_bytes(),
            swap_max_bytes.to_be_bytes().as_slice(),
            &[u8::from(oom_group)],
        ],
    )
}

fn derive_teardown_receipt_id(
    strict_receipt_id: &ContentId,
    base_receipt_id: &ContentId,
    descendant_kill_requested: bool,
    populated_zero_verified: bool,
    wait_ms: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-v2-strict-teardown-receipt.v1",
        [
            strict_receipt_id.as_str().as_bytes(),
            base_receipt_id.as_str().as_bytes(),
            &[u8::from(descendant_kill_requested)],
            &[u8::from(populated_zero_verified)],
            wait_ms.to_be_bytes().as_slice(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn populated_parser_is_closed_shape() {
        assert_eq!(parse_populated("populated 0\nfrozen 0\n"), Some(0));
        assert_eq!(parse_populated("populated 1\nfrozen 0\n"), Some(1));
        assert_eq!(parse_populated("frozen 0\n"), None);
        assert_eq!(parse_populated("populated many\n"), None);
    }

    #[test]
    fn strict_identity_separates_swap_and_oom_semantics() {
        let base = ContentId::derive("test-base", [b"base".as_slice()]);
        let a = derive_strict_receipt_id(&base, "/sys/fs/cgroup/delegated/leaf", 0, true);
        let b = derive_strict_receipt_id(&base, "/sys/fs/cgroup/delegated/leaf", 1, true);
        let c = derive_strict_receipt_id(&base, "/sys/fs/cgroup/delegated/leaf", 0, false);
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn teardown_identity_distinguishes_descendant_kill() {
        let strict = ContentId::derive("test-strict", [b"strict".as_slice()]);
        let base = ContentId::derive("test-base", [b"base".as_slice()]);
        let graceful = derive_teardown_receipt_id(&strict, &base, false, true, 0);
        let killed = derive_teardown_receipt_id(&strict, &base, true, true, 0);
        assert_ne!(graceful, killed);
    }
}
