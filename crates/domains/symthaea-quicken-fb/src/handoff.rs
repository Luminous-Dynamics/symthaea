// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Renderer-side display handoff receipt.
//!
//! This module is deliberately presentation-only. A receipt is written only
//! after the DRM framebuffer has been dropped by the caller, and it is never
//! interpreted as boot, login, recovery, or generation authority.

#![forbid(unsafe_code)]

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const HANDOFF_RECEIPT_VERSION: u16 = 1;

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
}

impl DisplayReleaseReceipt {
    pub fn new(release: Duration, renderer_uptime: Duration, reason: ExitReason) -> Self {
        Self {
            renderer_pid: std::process::id(),
            release_us: saturating_micros(release),
            renderer_uptime_us: saturating_micros(renderer_uptime),
            reason,
        }
    }

    /// Persist a tiny atomic acknowledgement after the caller has released DRM.
    ///
    /// The file is diagnostic evidence only. Future handoff coordination must
    /// additionally verify the renderer process/unit state before granting the
    /// display manager access to the device.
    pub fn write_atomic(&self, path: &Path) -> io::Result<()> {
        let parent = path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "handoff receipt path has no parent")
        })?;
        fs::create_dir_all(parent)?;

        let tmp = temporary_path(path, self.renderer_pid);
        let value = serde_json::json!({
            "version": HANDOFF_RECEIPT_VERSION,
            "renderer_pid": self.renderer_pid,
            "release_us": self.release_us,
            "renderer_uptime_us": self.renderer_uptime_us,
            "reason": self.reason.as_str(),
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
        );
        assert_eq!(receipt.release_us, 42);
        assert_eq!(receipt.renderer_uptime_us, 9001);
    }
}
