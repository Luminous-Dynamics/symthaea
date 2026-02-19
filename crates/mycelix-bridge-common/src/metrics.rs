//! Lightweight in-memory bridge metrics collection.
//!
//! Provides WASM-compatible metrics tracking for bridge dispatch calls.
//! Since Holochain WASM zomes are single-threaded, we use `RefCell` for
//! interior mutability instead of `Mutex` or atomics.
//!
//! ## Usage
//!
//! ```ignore
//! use mycelix_bridge_common::metrics::{bridge_metrics, BridgeMetricsSnapshot};
//!
//! // Record a successful call
//! bridge_metrics().record_success("property_registry", "verify_ownership", 42);
//!
//! // Record an error
//! bridge_metrics().record_error("property_registry", "verify_ownership", "BRG-006");
//!
//! // Record a rate limit hit
//! bridge_metrics().record_rate_limit_hit();
//!
//! // Get a serializable snapshot
//! let snapshot: BridgeMetricsSnapshot = bridge_metrics().snapshot();
//! ```

use core::cell::RefCell;
use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Maximum number of latency samples retained for percentile computation.
///
/// Keeping this small ensures bounded memory usage in WASM. 256 samples
/// is enough for p50/p95/p99 approximation without excessive overhead.
const MAX_LATENCY_SAMPLES: usize = 256;

/// Maximum number of distinct (zome, fn_name) pairs tracked.
///
/// Prevents unbounded memory growth from diverse call patterns. Once the
/// limit is reached, new call keys are tracked under a catch-all `"_overflow"` key.
const MAX_CALL_KEYS: usize = 128;

// ============================================================================
// Internal counters
// ============================================================================

/// Per-function call counters.
#[derive(Debug, Clone)]
struct CallCounter {
    /// Fully qualified key: "zome::fn_name"
    key: String,
    /// Total successful calls.
    success_count: u64,
    /// Total failed calls.
    error_count: u64,
}

/// Per-error-variant counters.
#[derive(Debug, Clone)]
struct ErrorCounter {
    /// Error code (e.g. "BRG-001").
    code: String,
    /// Number of times this error has occurred.
    count: u64,
}

/// Ring buffer of latency samples (in microseconds).
///
/// Uses a fixed-size array with a write cursor for O(1) insertion.
/// When full, the oldest sample is overwritten.
#[derive(Debug, Clone)]
struct LatencyRing {
    /// Latency values in microseconds.
    samples: Vec<u64>,
    /// Next write position (wraps around).
    cursor: usize,
    /// Total number of samples ever recorded (may exceed `MAX_LATENCY_SAMPLES`).
    total_recorded: u64,
}

impl LatencyRing {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(MAX_LATENCY_SAMPLES),
            cursor: 0,
            total_recorded: 0,
        }
    }

    fn push(&mut self, latency_us: u64) {
        if self.samples.len() < MAX_LATENCY_SAMPLES {
            self.samples.push(latency_us);
        } else {
            self.samples[self.cursor] = latency_us;
        }
        self.cursor = (self.cursor + 1) % MAX_LATENCY_SAMPLES;
        self.total_recorded += 1;
    }

    /// Compute percentiles from the current samples.
    ///
    /// Returns `(p50, p95, p99)` in microseconds, or `None` if no samples exist.
    fn percentiles(&self) -> Option<LatencyPercentiles> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let len = sorted.len();

        let p50 = sorted[len * 50 / 100];
        let p95 = sorted[(len * 95 / 100).min(len - 1)];
        let p99 = sorted[(len * 99 / 100).min(len - 1)];

        Some(LatencyPercentiles {
            p50_us: p50,
            p95_us: p95,
            p99_us: p99,
            sample_count: len as u32,
            total_recorded: self.total_recorded,
        })
    }
}

// ============================================================================
// Core metrics store
// ============================================================================

/// In-memory metrics store for bridge dispatch operations.
///
/// Thread-local to the WASM instance (single-threaded). The store accumulates
/// counters and latency samples for the lifetime of the zome instance. Metrics
/// reset when the conductor restarts or the cell is re-created.
#[derive(Debug)]
pub struct BridgeMetrics {
    /// Per-function call counters, keyed by "zome::fn_name".
    call_counters: Vec<CallCounter>,
    /// Per-error-code counters.
    error_counters: Vec<ErrorCounter>,
    /// Latency ring buffer for dispatch call durations.
    latency: LatencyRing,
    /// Total number of rate limit hits.
    rate_limit_hits: u64,
    /// Total successful dispatches (all functions combined).
    total_success: u64,
    /// Total failed dispatches (all functions combined).
    total_errors: u64,
    /// Total cross-cluster calls (subset of total_success + total_errors).
    total_cross_cluster: u64,
}

impl BridgeMetrics {
    fn new() -> Self {
        Self {
            call_counters: Vec::new(),
            error_counters: Vec::new(),
            latency: LatencyRing::new(),
            rate_limit_hits: 0,
            total_success: 0,
            total_errors: 0,
            total_cross_cluster: 0,
        }
    }

    /// Find or create a call counter for the given zome::fn_name key.
    fn get_or_create_call_counter(&mut self, zome: &str, fn_name: &str) -> &mut CallCounter {
        let key_str = format!("{}::{}", zome, fn_name);

        // Check if it already exists
        let pos = self.call_counters.iter().position(|c| c.key == key_str);
        if let Some(idx) = pos {
            return &mut self.call_counters[idx];
        }

        // If we're at the limit, use "_overflow::_overflow" as a catch-all
        let effective_key = if self.call_counters.len() >= MAX_CALL_KEYS {
            "_overflow::_overflow".to_string()
        } else {
            key_str
        };

        // Check again for overflow key
        let pos = self.call_counters.iter().position(|c| c.key == effective_key);
        if let Some(idx) = pos {
            return &mut self.call_counters[idx];
        }

        self.call_counters.push(CallCounter {
            key: effective_key,
            success_count: 0,
            error_count: 0,
        });
        let len = self.call_counters.len();
        &mut self.call_counters[len - 1]
    }

    /// Find or create an error counter for the given error code.
    fn get_or_create_error_counter(&mut self, error_code: &str) -> &mut ErrorCounter {
        let pos = self.error_counters.iter().position(|c| c.code == error_code);
        if let Some(idx) = pos {
            return &mut self.error_counters[idx];
        }

        self.error_counters.push(ErrorCounter {
            code: error_code.to_string(),
            count: 0,
        });
        let len = self.error_counters.len();
        &mut self.error_counters[len - 1]
    }

    /// Record a successful dispatch call.
    ///
    /// - `zome`: target zome name
    /// - `fn_name`: target function name
    /// - `latency_us`: call duration in microseconds (0 if timing is not available)
    pub fn record_success(&mut self, zome: &str, fn_name: &str, latency_us: u64) {
        self.total_success += 1;
        let counter = self.get_or_create_call_counter(zome, fn_name);
        counter.success_count += 1;
        if latency_us > 0 {
            self.latency.push(latency_us);
        }
    }

    /// Record a failed dispatch call.
    ///
    /// - `zome`: target zome name
    /// - `fn_name`: target function name
    /// - `error_code`: the BRG-xxx error code from the `BridgeError`
    pub fn record_error(&mut self, zome: &str, fn_name: &str, error_code: &str) {
        self.total_errors += 1;
        let counter = self.get_or_create_call_counter(zome, fn_name);
        counter.error_count += 1;
        let ec = self.get_or_create_error_counter(error_code);
        ec.count += 1;
    }

    /// Record a rate limit hit (counted separately from per-function errors).
    pub fn record_rate_limit_hit(&mut self) {
        self.rate_limit_hits += 1;
        let ec = self.get_or_create_error_counter("BRG-001");
        ec.count += 1;
        self.total_errors += 1;
    }

    /// Increment the cross-cluster call counter.
    pub fn record_cross_cluster(&mut self) {
        self.total_cross_cluster += 1;
    }

    /// Take a serializable snapshot of the current metrics.
    pub fn snapshot(&self) -> BridgeMetricsSnapshot {
        let call_counts: Vec<CallCountSnapshot> = self.call_counters.iter().map(|c| {
            CallCountSnapshot {
                key: c.key.clone(),
                success_count: c.success_count,
                error_count: c.error_count,
            }
        }).collect();

        let error_counts: Vec<ErrorCountSnapshot> = self.error_counters.iter().map(|c| {
            ErrorCountSnapshot {
                code: c.code.clone(),
                count: c.count,
            }
        }).collect();

        BridgeMetricsSnapshot {
            total_success: self.total_success,
            total_errors: self.total_errors,
            total_cross_cluster: self.total_cross_cluster,
            rate_limit_hits: self.rate_limit_hits,
            call_counts,
            error_counts,
            latency: self.latency.percentiles(),
        }
    }

    /// Reset all metrics to zero. Useful for testing.
    pub fn reset(&mut self) {
        self.call_counters.clear();
        self.error_counters.clear();
        self.latency = LatencyRing::new();
        self.rate_limit_hits = 0;
        self.total_success = 0;
        self.total_errors = 0;
        self.total_cross_cluster = 0;
    }
}

// ============================================================================
// Thread-local singleton
// ============================================================================

thread_local! {
    static METRICS: RefCell<BridgeMetrics> = RefCell::new(BridgeMetrics::new());
}

/// Access the thread-local bridge metrics store.
///
/// In Holochain WASM, each zome instance gets its own thread-local store.
/// Metrics accumulate for the lifetime of the cell and reset on conductor
/// restart.
///
/// # Panics
///
/// Panics if called re-entrantly (should not happen in normal zome code).
pub fn with_bridge_metrics<F, R>(f: F) -> R
where
    F: FnOnce(&mut BridgeMetrics) -> R,
{
    METRICS.with(|cell| {
        let mut guard = cell.borrow_mut();
        f(&mut *guard)
    })
}

/// Convenience: record a successful dispatch.
pub fn record_success(zome: &str, fn_name: &str, latency_us: u64) {
    with_bridge_metrics(|m| m.record_success(zome, fn_name, latency_us));
}

/// Convenience: record a failed dispatch.
pub fn record_error(zome: &str, fn_name: &str, error_code: &str) {
    with_bridge_metrics(|m| m.record_error(zome, fn_name, error_code));
}

/// Convenience: record a rate limit hit.
pub fn record_rate_limit_hit() {
    with_bridge_metrics(|m| m.record_rate_limit_hit());
}

/// Convenience: record a cross-cluster call.
pub fn record_cross_cluster() {
    with_bridge_metrics(|m| m.record_cross_cluster());
}

/// Convenience: get a serializable metrics snapshot.
pub fn metrics_snapshot() -> BridgeMetricsSnapshot {
    with_bridge_metrics(|m| m.snapshot())
}

// ============================================================================
// Serializable snapshot types (returned to callers)
// ============================================================================

/// Per-function call count in the metrics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CallCountSnapshot {
    /// Fully qualified key: "zome::fn_name"
    pub key: String,
    /// Number of successful calls.
    pub success_count: u64,
    /// Number of failed calls.
    pub error_count: u64,
}

/// Per-error-code count in the metrics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErrorCountSnapshot {
    /// Error code (e.g. "BRG-001").
    pub code: String,
    /// Number of occurrences.
    pub count: u64,
}

/// Latency percentile summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatencyPercentiles {
    /// 50th percentile (median) in microseconds.
    pub p50_us: u64,
    /// 95th percentile in microseconds.
    pub p95_us: u64,
    /// 99th percentile in microseconds.
    pub p99_us: u64,
    /// Number of samples currently in the ring buffer.
    pub sample_count: u32,
    /// Total number of samples ever recorded (may exceed sample_count).
    pub total_recorded: u64,
}

/// Serializable snapshot of all bridge metrics.
///
/// Returned by [`with_bridge_metrics`] / [`metrics_snapshot`] and by the
/// `get_bridge_metrics()` extern fn in both bridge coordinators.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BridgeMetricsSnapshot {
    /// Total successful dispatches.
    pub total_success: u64,
    /// Total failed dispatches.
    pub total_errors: u64,
    /// Total cross-cluster calls (subset of success + errors).
    pub total_cross_cluster: u64,
    /// Number of rate limit rejections.
    pub rate_limit_hits: u64,
    /// Per-function call counts.
    pub call_counts: Vec<CallCountSnapshot>,
    /// Per-error-code counts.
    pub error_counts: Vec<ErrorCountSnapshot>,
    /// Latency percentiles (None if no latency data collected).
    pub latency: Option<LatencyPercentiles>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a fresh BridgeMetrics for isolated testing.
    fn fresh_metrics() -> BridgeMetrics {
        BridgeMetrics::new()
    }

    #[test]
    fn empty_snapshot() {
        let m = fresh_metrics();
        let snap = m.snapshot();
        assert_eq!(snap.total_success, 0);
        assert_eq!(snap.total_errors, 0);
        assert_eq!(snap.total_cross_cluster, 0);
        assert_eq!(snap.rate_limit_hits, 0);
        assert!(snap.call_counts.is_empty());
        assert!(snap.error_counts.is_empty());
        assert!(snap.latency.is_none());
    }

    #[test]
    fn record_success_increments_counters() {
        let mut m = fresh_metrics();
        m.record_success("property_registry", "verify_ownership", 100);
        m.record_success("property_registry", "verify_ownership", 200);
        m.record_success("property_registry", "get_property", 50);

        let snap = m.snapshot();
        assert_eq!(snap.total_success, 3);
        assert_eq!(snap.total_errors, 0);
        assert_eq!(snap.call_counts.len(), 2);

        let verify = snap.call_counts.iter()
            .find(|c| c.key == "property_registry::verify_ownership")
            .unwrap();
        assert_eq!(verify.success_count, 2);
        assert_eq!(verify.error_count, 0);

        let get = snap.call_counts.iter()
            .find(|c| c.key == "property_registry::get_property")
            .unwrap();
        assert_eq!(get.success_count, 1);
    }

    #[test]
    fn record_error_increments_counters() {
        let mut m = fresh_metrics();
        m.record_error("property_registry", "verify_ownership", "BRG-006");
        m.record_error("property_registry", "verify_ownership", "BRG-006");
        m.record_error("housing_units", "get_all", "BRG-003");

        let snap = m.snapshot();
        assert_eq!(snap.total_errors, 3);
        assert_eq!(snap.total_success, 0);

        let verify = snap.call_counts.iter()
            .find(|c| c.key == "property_registry::verify_ownership")
            .unwrap();
        assert_eq!(verify.error_count, 2);

        let brg006 = snap.error_counts.iter()
            .find(|c| c.code == "BRG-006")
            .unwrap();
        assert_eq!(brg006.count, 2);

        let brg003 = snap.error_counts.iter()
            .find(|c| c.code == "BRG-003")
            .unwrap();
        assert_eq!(brg003.count, 1);
    }

    #[test]
    fn record_rate_limit_hit() {
        let mut m = fresh_metrics();
        m.record_rate_limit_hit();
        m.record_rate_limit_hit();
        m.record_rate_limit_hit();

        let snap = m.snapshot();
        assert_eq!(snap.rate_limit_hits, 3);
        assert_eq!(snap.total_errors, 3);

        let brg001 = snap.error_counts.iter()
            .find(|c| c.code == "BRG-001")
            .unwrap();
        assert_eq!(brg001.count, 3);
    }

    #[test]
    fn record_cross_cluster() {
        let mut m = fresh_metrics();
        m.record_cross_cluster();
        m.record_cross_cluster();
        let snap = m.snapshot();
        assert_eq!(snap.total_cross_cluster, 2);
    }

    #[test]
    fn latency_percentiles_single_sample() {
        let mut m = fresh_metrics();
        m.record_success("z", "f", 1000);
        let snap = m.snapshot();
        let lat = snap.latency.unwrap();
        assert_eq!(lat.p50_us, 1000);
        assert_eq!(lat.p95_us, 1000);
        assert_eq!(lat.p99_us, 1000);
        assert_eq!(lat.sample_count, 1);
        assert_eq!(lat.total_recorded, 1);
    }

    #[test]
    fn latency_percentiles_multiple_samples() {
        let mut m = fresh_metrics();
        // Push 100 samples: 1, 2, ..., 100 (microseconds)
        for i in 1..=100 {
            m.record_success("z", "f", i);
        }
        let snap = m.snapshot();
        let lat = snap.latency.unwrap();
        assert_eq!(lat.sample_count, 100);
        assert_eq!(lat.total_recorded, 100);
        // p50 of [1..100] at index 50 = 51
        assert_eq!(lat.p50_us, 51);
        // p95 at index 95 = 96
        assert_eq!(lat.p95_us, 96);
        // p99 at index 99 = 100
        assert_eq!(lat.p99_us, 100);
    }

    #[test]
    fn latency_ring_wraps_around() {
        let mut ring = LatencyRing::new();
        // Fill beyond capacity
        for i in 0..(MAX_LATENCY_SAMPLES as u64 + 10) {
            ring.push(i);
        }
        assert_eq!(ring.samples.len(), MAX_LATENCY_SAMPLES);
        assert_eq!(ring.total_recorded, MAX_LATENCY_SAMPLES as u64 + 10);
        // Percentiles should still be computable
        assert!(ring.percentiles().is_some());
    }

    #[test]
    fn zero_latency_not_recorded() {
        let mut m = fresh_metrics();
        m.record_success("z", "f", 0);
        let snap = m.snapshot();
        assert_eq!(snap.total_success, 1);
        assert!(snap.latency.is_none(), "zero latency should not be pushed");
    }

    #[test]
    fn overflow_key_used_when_limit_reached() {
        let mut m = fresh_metrics();
        // Fill up to MAX_CALL_KEYS distinct keys
        for i in 0..MAX_CALL_KEYS {
            m.record_success(&format!("zome_{}", i), "fn", 10);
        }
        assert_eq!(m.call_counters.len(), MAX_CALL_KEYS);

        // Next call should go to overflow
        m.record_success("zome_new", "fn", 10);
        // Should have one more entry for overflow
        assert_eq!(m.call_counters.len(), MAX_CALL_KEYS + 1);
        let overflow = m.call_counters.iter()
            .find(|c| c.key == "_overflow::_overflow")
            .unwrap();
        assert_eq!(overflow.success_count, 1);
    }

    #[test]
    fn reset_clears_everything() {
        let mut m = fresh_metrics();
        m.record_success("z", "f", 100);
        m.record_error("z", "f", "BRG-003");
        m.record_rate_limit_hit();
        m.record_cross_cluster();
        m.reset();

        let snap = m.snapshot();
        assert_eq!(snap.total_success, 0);
        assert_eq!(snap.total_errors, 0);
        assert_eq!(snap.total_cross_cluster, 0);
        assert_eq!(snap.rate_limit_hits, 0);
        assert!(snap.call_counts.is_empty());
        assert!(snap.error_counts.is_empty());
        assert!(snap.latency.is_none());
    }

    #[test]
    fn mixed_success_and_error_same_function() {
        let mut m = fresh_metrics();
        m.record_success("z", "f", 50);
        m.record_success("z", "f", 60);
        m.record_error("z", "f", "BRG-006");

        let snap = m.snapshot();
        assert_eq!(snap.total_success, 2);
        assert_eq!(snap.total_errors, 1);

        let counter = snap.call_counts.iter()
            .find(|c| c.key == "z::f")
            .unwrap();
        assert_eq!(counter.success_count, 2);
        assert_eq!(counter.error_count, 1);
    }

    #[test]
    fn snapshot_serde_roundtrip() {
        let mut m = fresh_metrics();
        m.record_success("property_registry", "verify_ownership", 100);
        m.record_error("housing_units", "get_all", "BRG-003");
        m.record_rate_limit_hit();
        m.record_cross_cluster();

        let snap = m.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let snap2: BridgeMetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snap, snap2);
    }

    #[test]
    fn thread_local_record_and_snapshot() {
        // Verify the thread-local convenience functions work by doing
        // all operations inside a single with_bridge_metrics call, then
        // verifying via the snapshot function.
        with_bridge_metrics(|m| {
            m.reset();
            m.record_success("a", "b", 10);
            m.record_error("a", "b", "BRG-004");
            m.record_rate_limit_hit();
            m.record_cross_cluster();
        });

        let snap = metrics_snapshot();
        assert_eq!(snap.total_success, 1);
        assert_eq!(snap.total_errors, 2); // 1 error + 1 rate limit
        assert_eq!(snap.total_cross_cluster, 1);
        assert_eq!(snap.rate_limit_hits, 1);

        // Clean up
        with_bridge_metrics(|m| m.reset());
    }

    #[test]
    fn convenience_functions_via_with_bridge_metrics() {
        // Test that with_bridge_metrics correctly accumulates state
        // across multiple calls (validates the thread-local pattern).
        with_bridge_metrics(|m| m.reset());

        with_bridge_metrics(|m| m.record_success("a", "b", 50));
        with_bridge_metrics(|m| m.record_error("a", "b", "BRG-004"));
        with_bridge_metrics(|m| m.record_rate_limit_hit());
        with_bridge_metrics(|m| m.record_cross_cluster());

        let snap = metrics_snapshot();
        assert_eq!(snap.total_success, 1);
        assert_eq!(snap.total_errors, 2); // 1 error + 1 rate limit
        assert_eq!(snap.total_cross_cluster, 1);
        assert_eq!(snap.rate_limit_hits, 1);

        with_bridge_metrics(|m| m.reset());
    }

    #[test]
    fn module_level_convenience_functions() {
        // Test the module-level convenience functions that delegate
        // to with_bridge_metrics. Use fully-qualified paths to avoid
        // any name resolution issues in the test module scope.
        with_bridge_metrics(|m| m.reset());

        crate::metrics::record_success("x", "y", 10);
        crate::metrics::record_error("x", "y", "BRG-003");
        crate::metrics::record_rate_limit_hit();
        crate::metrics::record_cross_cluster();

        let snap = crate::metrics::metrics_snapshot();
        assert_eq!(snap.total_success, 1);
        assert_eq!(snap.total_errors, 2); // 1 error + 1 rate limit
        assert_eq!(snap.total_cross_cluster, 1);
        assert_eq!(snap.rate_limit_hits, 1);

        with_bridge_metrics(|m| m.reset());
    }
}
