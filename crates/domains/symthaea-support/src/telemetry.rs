// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! System telemetry collection for the support sub-crate.
//!
//! Provides live collection of host-level metrics (RSS memory, disk free space,
//! network latency) and stubs for Holochain-specific counters (gossip peers,
//! error rate) that will be wired to the conductor in a future iteration.

use crate::types::SystemTelemetry;

/// Collect a snapshot of current system telemetry.
///
/// Fills in real values for process RSS and disk free percentage on Linux;
/// gossip peers, network latency, and error rate are either measured locally
/// or stubbed to safe defaults.
pub fn collect_telemetry() -> SystemTelemetry {
    SystemTelemetry {
        conductor_memory_mb: get_process_rss_mb(),
        gossip_peers: 0, // stub — needs conductor API
        disk_free_pct: get_disk_free_pct(),
        network_latency_ms: measure_network_latency(),
        error_rate_per_min: 0.0, // stub — needs error counter
    }
}

// ---------------------------------------------------------------------------
// Process RSS
// ---------------------------------------------------------------------------

/// Read the current process's Resident Set Size from `/proc/self/status`.
///
/// Returns the value in megabytes, or 0.0 on any failure.
#[cfg(target_os = "linux")]
fn get_process_rss_mb() -> f64 {
    use std::fs;

    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            // Format: "VmRSS:    <number> kB"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2
                && let Ok(kb) = parts[1].parse::<f64>()
            {
                return kb / 1024.0;
            }
        }
    }
    0.0
}

#[cfg(not(target_os = "linux"))]
fn get_process_rss_mb() -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Disk free percentage
// ---------------------------------------------------------------------------

/// Calculate the percentage of free disk space on `/` using `statvfs`.
///
/// Returns a value in the range `[0.0, 100.0]`, or 100.0 on non-Linux
/// platforms (conservative: "no alarm").
#[cfg(target_os = "linux")]
fn get_disk_free_pct() -> f64 {
    use std::ffi::CString;
    use std::mem::MaybeUninit;

    let path = CString::new("/").expect("CString::new(\"/\") failed");
    let mut buf = MaybeUninit::<libc::statvfs>::uninit();

    // SAFETY: `statvfs` writes into the provided buffer and `path` is a
    // valid NUL-terminated C string pointing to a mount point that always
    // exists on Linux.
    let ret = unsafe { libc::statvfs(path.as_ptr(), buf.as_mut_ptr()) };

    if ret != 0 {
        return 100.0;
    }

    // SAFETY: `statvfs` returned 0, so the buffer is fully initialised.
    let stat = unsafe { buf.assume_init() };

    let total = stat.f_blocks as f64;
    if total == 0.0 {
        return 100.0;
    }

    let free = stat.f_bfree as f64;
    (free / total) * 100.0
}

#[cfg(not(target_os = "linux"))]
fn get_disk_free_pct() -> f64 {
    100.0
}

// ---------------------------------------------------------------------------
// Network latency
// ---------------------------------------------------------------------------

/// Attempt a TCP connect to `127.0.0.1:8888` with a 2-second timeout and
/// measure the elapsed time in milliseconds.
///
/// Returns 0.0 if the connection fails (e.g. nothing is listening on that
/// port), which is the common case; a non-zero value indicates a reachable
/// local service and its connection latency.
fn measure_network_latency() -> f64 {
    use std::net::{SocketAddr, TcpStream};
    use std::time::{Duration, Instant};

    let addr: SocketAddr = "127.0.0.1:8888".parse().expect("valid socket address");
    let timeout = Duration::from_secs(2);

    let start = Instant::now();
    match TcpStream::connect_timeout(&addr, timeout) {
        Ok(_stream) => {
            let elapsed = start.elapsed();
            elapsed.as_secs_f64() * 1000.0
        }
        Err(_) => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_telemetry_returns_finite() {
        let t = collect_telemetry();
        assert!(
            t.conductor_memory_mb.is_finite(),
            "conductor_memory_mb must be finite"
        );
        assert!(t.disk_free_pct.is_finite(), "disk_free_pct must be finite");
        assert!(
            t.network_latency_ms.is_finite(),
            "network_latency_ms must be finite"
        );
        assert!(
            t.error_rate_per_min.is_finite(),
            "error_rate_per_min must be finite"
        );
    }

    #[test]
    fn test_disk_free_reasonable() {
        let pct = get_disk_free_pct();
        assert!(pct >= 0.0, "disk_free_pct must be >= 0.0, got {pct}");
        assert!(pct <= 100.0, "disk_free_pct must be <= 100.0, got {pct}");
    }

    #[test]
    fn test_network_latency_non_negative() {
        let latency = measure_network_latency();
        assert!(
            latency >= 0.0,
            "network_latency_ms must be >= 0.0, got {latency}"
        );
    }
}
