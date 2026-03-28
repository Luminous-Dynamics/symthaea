// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sweettest Harness — Memory-Efficient Conductor Sharing
//!
//! Provides a shared conductor pattern for Mycelix sweettests. Instead of
//! creating a fresh `SweetConductor` per test (which OOMs on large DNAs),
//! this crate provides a `SharedConductor` that is initialized once per
//! test file and reused across all tests.
//!
//! ## Problem
//!
//! 322 sweettest functions × fresh conductor each = unbounded memory growth.
//! Commons (90 zomes, 30MB DNA) OOMs immediately. Identity (25 zomes, 6.4MB)
//! gets through 17/67 before OOM on 31GB RAM.
//!
//! ## Solution
//!
//! `SharedConductor` wraps a single `SweetConductor` behind `tokio::sync::OnceCell`.
//! The first test to call `shared_conductor()` initializes it; all subsequent
//! tests in the same file reuse it. Each test gets a unique app install
//! (unique agent key) to avoid state conflicts.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sweettest_harness::SharedConductor;
//!
//! static CONDUCTOR: SharedConductor = SharedConductor::new();
//!
//! fn dna_path() -> std::path::PathBuf {
//!     std::path::PathBuf::from("../dna/mycelix_identity_dna.dna")
//! }
//!
//! #[tokio::test(flavor = "multi_thread")]
//! #[ignore = "requires Holochain conductor"]
//! async fn test_create_did() {
//!     let (conductor, alice) = CONDUCTOR.get_or_init(&dna_path()).await;
//!     let result: Record = conductor.call(&alice.zome("did_registry"), "create_did", ()).await;
//!     assert!(result.action().author() == alice.agent_pubkey());
//! }
//! ```

use holochain::sweettest::{SweetConductor, SweetCell, SweetDnaFile};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::OnceCell;

/// Counter for generating unique app IDs per test.
static APP_COUNTER: AtomicU32 = AtomicU32::new(0);

/// A shared conductor that is initialized once and reused across tests.
///
/// Use one `static SharedConductor` per test file. The first test to call
/// `get_or_init()` creates the conductor; all subsequent tests reuse it.
pub struct SharedConductor {
    inner: OnceCell<SweetConductor>,
    dna: OnceCell<SweetDnaFile>,
}

impl SharedConductor {
    /// Create a new uninitialized shared conductor.
    /// Call this in a `static` at the top of your test file.
    pub const fn new() -> Self {
        Self {
            inner: OnceCell::const_new(),
            dna: OnceCell::const_new(),
        }
    }

    /// Get or initialize the shared conductor with the given DNA path.
    ///
    /// The first call creates the conductor and loads the DNA. Subsequent
    /// calls return the existing conductor. Each call creates a unique
    /// app install with a fresh agent key to avoid state conflicts.
    ///
    /// Returns `(conductor_ref, cell)` where `cell` is the installed app's cell.
    pub async fn get_or_init(
        &self,
        dna_path: &Path,
    ) -> (&SweetConductor, SweetCell) {
        let conductor = self
            .inner
            .get_or_init(|| async {
                log_memory("conductor_init_start");
                SweetConductor::from_standard_config().await
            })
            .await;

        let dna_file = self
            .dna
            .get_or_init(|| async {
                SweetDnaFile::from_bundle(dna_path).await.unwrap()
            })
            .await;

        // Each test gets a unique app install (fresh agent, isolated state)
        let app_id = format!(
            "test-app-{}",
            APP_COUNTER.fetch_add(1, Ordering::Relaxed)
        );

        let app = conductor
            .setup_app(&app_id, &[dna_file.clone()])
            .await
            .unwrap();

        let (cell,) = app.into_tuple();

        log_memory(&format!("after_install_{}", app_id));

        (conductor, cell)
    }

    /// Get the conductor if already initialized (no new app install).
    pub fn get(&self) -> Option<&SweetConductor> {
        self.inner.get()
    }
}

// ============================================================================
// Memory Monitoring
// ============================================================================

/// Log current RSS (Resident Set Size) for memory debugging.
///
/// Reads from `/proc/self/status` on Linux. No-op on other platforms.
pub fn log_memory(label: &str) {
    #[cfg(target_os = "linux")]
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        if let Some(line) = status.lines().find(|l| l.starts_with("VmRSS:")) {
            tracing::info!(target: "sweettest_harness", "[MEM] {}: {}", label, line.trim());
            eprintln!("[MEM] {}: {}", label, line.trim());
        }
    }
}

/// Get current RSS in kilobytes. Returns 0 on non-Linux.
pub fn get_rss_kb() -> usize {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse().ok())
                    })
            })
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Panic if RSS exceeds the given limit in megabytes.
///
/// Call at the start of each test to catch memory leaks early.
/// Default recommendation: 4096 MB (4GB) for sweettest suites.
///
/// ```rust,ignore
/// sweettest_harness::assert_memory_under(4096);
/// ```
pub fn assert_memory_under(limit_mb: usize) {
    let rss_kb = get_rss_kb();
    let rss_mb = rss_kb / 1024;
    if rss_mb > limit_mb {
        panic!(
            "MEMORY GUARD: RSS {}MB exceeds {}MB limit. \
             Likely cause: conductor not cleaned up from previous test. \
             Consider using SharedConductor to reuse conductors across tests.",
            rss_mb, limit_mb
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_logging() {
        log_memory("test");
        let rss = get_rss_kb();
        // Should be non-zero on Linux
        #[cfg(target_os = "linux")]
        assert!(rss > 0, "RSS should be non-zero on Linux");
    }

    #[test]
    fn test_assert_memory_under_passes() {
        // 100GB limit should always pass
        assert_memory_under(100_000);
    }

    #[test]
    #[should_panic(expected = "MEMORY GUARD")]
    fn test_assert_memory_under_fails() {
        // 0MB limit should always fail
        assert_memory_under(0);
    }

    #[test]
    fn test_app_counter_increments() {
        let a = APP_COUNTER.load(Ordering::Relaxed);
        let _ = APP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let b = APP_COUNTER.load(Ordering::Relaxed);
        assert_eq!(b, a + 1);
    }
}
