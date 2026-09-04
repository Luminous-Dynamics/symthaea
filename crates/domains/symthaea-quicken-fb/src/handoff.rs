// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Renderer-side display handoff receipt.
//!
//! This module is deliberately presentation-only. A receipt is written only
//! after the renderer has explicitly attempted display restoration and released
//! its DRM framebuffer. It is never interpreted as boot, login, recovery, or
//! generation authority.

#![forbid(unsafe_code)]

use crate::framebuffer::DisplayRestoreOutcome;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const HANDOFF_RECEIPT_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    Signal,
    InstallComplete,
    Natural,
}

impl ExitReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Signal => "signal",
            Self::InstallComplete => "install-complete",
            Self::Natural => "natural",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayReleaseReceipt {
    pub renderer_pid: u32,
    pub release_us: u64,
    pub renderer_uptime_us: u64,
    pub reason: ExitReason,
    pub restore_outcome: DisplayRestoreOutcome,
}

impl DisplayReleaseReceipt {
    pub fn new(
        release: Duration,
        renderer_uptime: Duration,
        reason: ExitReason,
        restore_outcome: DisplayRestoreOutcome,
    ) -> Self {
        Self {
            renderer_pid: std::process::id(),
            release_us: saturating_micros(release),
            renderer_uptime_us: saturating_micros(renderer_uptime),
            reason,
            restore_outcome,
        }
    }

    /// Persist a tiny atomic acknowledgement after the caller has released DRM.
    ///
    /// The file is diagnostic evidence only. Future handoff coordination must
    /// additionally verify renderer process/unit state before granting a display
    /// manager access to the device. In particular, `restore_succeeded=false`
    /// must never become a reason for presentation to hold login or recovery.
    pub fn write_atomic(&self, path: &Path) -> io::Result<()> {
        let parent = path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "handoff receipt path has no parent")
        })?;
        fs::create_dir_all(parent)?;

        let tmp = temporary_path(path, self.renderer_pid);
        let restore_error_kind = self
            .restore_outcome
            .error_kind()
            .map(|kind| format!("{kind:?}"));
        let value = serde_json::json!({
            "version": HANDOFF_RECEIPT_VERSION,
            "renderer_pid": self.renderer_pid,
            "release_us": self.release_us,
            "renderer_uptime_us": self.renderer_uptime_us,
            "reason": self.reason.as_str(),
            "restore_attempted": true,
            "restore_succeeded": self.restore_outcome.succeeded(),
            "restore_status": self.restore_outcome.as_str(),
            "restore_error_kind": restore_error_kind,
        });
        let bytes = serde_json::to_vec(&value)
            .map_err(|error| io::Error::other(format!("serialize handoff receipt: {error}")))?;
        fs::write(&tmp, bytes)?;
        fs::rename(tmp, path)?;
        Ok(())
    }
}

fn temporary_path(path: &Path, pid: u32) -> PathBuf {
    let mut temporary = path.as_os_str().to_os_string();
    temporary.push(format!(".tmp-{pid}"));
    PathBuf::from(temporary)
}

fn saturating_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn reason_names_are_stable() {
        assert_eq!(ExitReason::Signal.as_str(), "signal");
        assert_eq!(ExitReason::InstallComplete.as_str(), "install-complete");
        assert_eq!(ExitReason::Natural.as_str(), "natural");
    }

    #[test]
    fn durations_saturate_to_u64() {
        let receipt = DisplayReleaseReceipt::new(
            Duration::from_micros(42),
            Duration::from_micros(9001),
            ExitReason::Natural,
            DisplayRestoreOutcome::Restored,
        );
        assert_eq!(receipt.release_us, 42);
        assert_eq!(receipt.renderer_uptime_us, 9001);
        assert!(receipt.restore_outcome.succeeded());
    }

    #[test]
    fn atomic_receipt_reports_failed_restore_without_raw_error_text() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "symthaea-handoff-test-{}-{nonce}",
            std::process::id()
        ));
        let path = directory.join("released.json");

        let receipt = DisplayReleaseReceipt::new(
            Duration::from_micros(73),
            Duration::from_millis(250),
            ExitReason::Signal,
            DisplayRestoreOutcome::RestoreFailed(io::ErrorKind::PermissionDenied),
        );
        receipt.write_atomic(&path).unwrap();

        let value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["version"], HANDOFF_RECEIPT_VERSION);
        assert_eq!(value["reason"], "signal");
        assert_eq!(value["release_us"], 73);
        assert_eq!(value["restore_attempted"], true);
        assert_eq!(value["restore_succeeded"], false);
        assert_eq!(value["restore_status"], "restore-failed");
        assert_eq!(value["restore_error_kind"], "PermissionDenied");

        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn atomic_receipt_reports_success_only_for_successful_ioctl() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "symthaea-handoff-success-test-{}-{nonce}",
            std::process::id()
        ));
        let path = directory.join("released.json");

        DisplayReleaseReceipt::new(
            Duration::from_micros(11),
            Duration::from_millis(100),
            ExitReason::Natural,
            DisplayRestoreOutcome::Restored,
        )
        .write_atomic(&path)
        .unwrap();

        let value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["restore_attempted"], true);
        assert_eq!(value["restore_succeeded"], true);
        assert_eq!(value["restore_status"], "restored");
        assert!(value["restore_error_kind"].is_null());

        let _ = fs::remove_dir_all(directory);
    }
}
