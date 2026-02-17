//! OpenTelemetry Integration for Proof Metrics
//!
//! Distributed tracing and metrics for proof generation and verification.
//!
//! ## Features
//!
//! - Proof generation timing
//! - Verification latency tracking
//! - Cache hit/miss rates
//! - Prometheus-compatible metrics export
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::telemetry::{ProofTelemetry, TelemetryStats};
//!
//! // Create telemetry instance
//! let telemetry = ProofTelemetry::new();
//!
//! // Record metrics
//! telemetry.record_generation("Range", Duration::from_millis(150), 1024);
//! telemetry.record_verification("Range", Duration::from_millis(20), true);
//!
//! // Export stats
//! let stats = telemetry.stats();
//! println!("{}", stats.to_prometheus());
//! ```

use std::time::Duration;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "proofs-telemetry")]
use opentelemetry::{
    global,
    metrics::{Counter, Histogram, Meter},
    KeyValue,
};

/// Proof telemetry collector
pub struct ProofTelemetry {
    /// Total proofs generated
    proofs_generated: AtomicU64,
    /// Total proofs verified
    proofs_verified: AtomicU64,
    /// Verification failures
    verification_failures: AtomicU64,
    /// Cache hits
    cache_hits: AtomicU64,
    /// Cache misses
    cache_misses: AtomicU64,

    #[cfg(feature = "proofs-telemetry")]
    generation_histogram: Option<Histogram<f64>>,
    #[cfg(feature = "proofs-telemetry")]
    verification_histogram: Option<Histogram<f64>>,
    #[cfg(feature = "proofs-telemetry")]
    proof_size_histogram: Option<Histogram<u64>>,
    #[cfg(feature = "proofs-telemetry")]
    generation_counter: Option<Counter<u64>>,
    #[cfg(feature = "proofs-telemetry")]
    verification_counter: Option<Counter<u64>>,
}

impl ProofTelemetry {
    /// Create a new telemetry collector
    pub fn new() -> Self {
        Self {
            proofs_generated: AtomicU64::new(0),
            proofs_verified: AtomicU64::new(0),
            verification_failures: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            #[cfg(feature = "proofs-telemetry")]
            generation_histogram: None,
            #[cfg(feature = "proofs-telemetry")]
            verification_histogram: None,
            #[cfg(feature = "proofs-telemetry")]
            proof_size_histogram: None,
            #[cfg(feature = "proofs-telemetry")]
            generation_counter: None,
            #[cfg(feature = "proofs-telemetry")]
            verification_counter: None,
        }
    }

    /// Create with OpenTelemetry meter
    #[cfg(feature = "proofs-telemetry")]
    pub fn with_meter(meter: &Meter) -> Self {
        let generation_histogram = meter
            .f64_histogram("proof.generation.duration")
            .with_description("Proof generation duration in milliseconds")
            .with_unit("ms")
            .build();

        let verification_histogram = meter
            .f64_histogram("proof.verification.duration")
            .with_description("Proof verification duration in milliseconds")
            .with_unit("ms")
            .build();

        let proof_size_histogram = meter
            .u64_histogram("proof.size")
            .with_description("Proof size in bytes")
            .with_unit("bytes")
            .build();

        let generation_counter = meter
            .u64_counter("proof.generation.count")
            .with_description("Total proofs generated")
            .build();

        let verification_counter = meter
            .u64_counter("proof.verification.count")
            .with_description("Total proofs verified")
            .build();

        Self {
            proofs_generated: AtomicU64::new(0),
            proofs_verified: AtomicU64::new(0),
            verification_failures: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            generation_histogram: Some(generation_histogram),
            verification_histogram: Some(verification_histogram),
            proof_size_histogram: Some(proof_size_histogram),
            generation_counter: Some(generation_counter),
            verification_counter: Some(verification_counter),
        }
    }

    /// Create with global meter (convenience method)
    #[cfg(feature = "proofs-telemetry")]
    pub fn with_global_meter(meter_name: &'static str) -> Self {
        let meter = global::meter(meter_name);
        Self::with_meter(&meter)
    }

    /// Record proof generation
    pub fn record_generation(&self, proof_type: &str, duration: Duration, size_bytes: usize) {
        self.proofs_generated.fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "proofs-telemetry")]
        {
            let attrs = [KeyValue::new("proof_type", proof_type.to_string())];

            if let Some(ref histogram) = self.generation_histogram {
                histogram.record(duration.as_secs_f64() * 1000.0, &attrs);
            }

            if let Some(ref histogram) = self.proof_size_histogram {
                histogram.record(size_bytes as u64, &attrs);
            }

            if let Some(ref counter) = self.generation_counter {
                counter.add(1, &attrs);
            }
        }

        // Silence unused variable warnings when feature not enabled
        let _ = (proof_type, duration, size_bytes);
    }

    /// Record proof verification
    pub fn record_verification(&self, proof_type: &str, duration: Duration, success: bool) {
        self.proofs_verified.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.verification_failures.fetch_add(1, Ordering::Relaxed);
        }

        #[cfg(feature = "proofs-telemetry")]
        {
            let attrs = [
                KeyValue::new("proof_type", proof_type.to_string()),
                KeyValue::new("success", success),
            ];

            if let Some(ref histogram) = self.verification_histogram {
                histogram.record(duration.as_secs_f64() * 1000.0, &attrs);
            }

            if let Some(ref counter) = self.verification_counter {
                counter.add(1, &attrs);
            }
        }

        // Silence unused variable warnings when feature not enabled
        let _ = (proof_type, duration, success);
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get statistics snapshot
    pub fn stats(&self) -> TelemetryStats {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        TelemetryStats {
            proofs_generated: self.proofs_generated.load(Ordering::Relaxed),
            proofs_verified: self.proofs_verified.load(Ordering::Relaxed),
            verification_failures: self.verification_failures.load(Ordering::Relaxed),
            cache_hits: hits,
            cache_misses: misses,
            cache_hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
}

impl Default for ProofTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

/// Telemetry statistics snapshot
#[derive(Clone, Debug)]
pub struct TelemetryStats {
    pub proofs_generated: u64,
    pub proofs_verified: u64,
    pub verification_failures: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
}

impl TelemetryStats {
    /// Verification success rate
    pub fn verification_success_rate(&self) -> f64 {
        if self.proofs_verified == 0 {
            1.0
        } else {
            (self.proofs_verified - self.verification_failures) as f64 / self.proofs_verified as f64
        }
    }
}

/// Initialize telemetry with a global meter
///
/// This sets up the global meter provider. For custom OTLP configuration,
/// set up the meter provider directly using opentelemetry_otlp.
#[cfg(feature = "proofs-telemetry")]
pub fn init_telemetry(service_name: &str) {
    // Users should configure their own MeterProvider and TracerProvider
    // using opentelemetry_sdk and opentelemetry_otlp directly.
    // This function is a placeholder for documentation purposes.
    let _ = service_name;
    tracing::info!("Telemetry initialized. Use ProofTelemetry::with_global_meter() to create collectors.");
}

/// Shutdown telemetry
#[cfg(feature = "proofs-telemetry")]
pub fn shutdown_telemetry() {
    global::shutdown_tracer_provider();
}

/// Prometheus-compatible output
impl TelemetryStats {
    /// Export as Prometheus metrics
    pub fn to_prometheus(&self) -> String {
        format!(
            r#"# HELP proof_generation_total Total proofs generated
# TYPE proof_generation_total counter
proof_generation_total {}

# HELP proof_verification_total Total proofs verified
# TYPE proof_verification_total counter
proof_verification_total {}

# HELP proof_verification_failures_total Total verification failures
# TYPE proof_verification_failures_total counter
proof_verification_failures_total {}

# HELP proof_cache_hits_total Total cache hits
# TYPE proof_cache_hits_total counter
proof_cache_hits_total {}

# HELP proof_cache_misses_total Total cache misses
# TYPE proof_cache_misses_total counter
proof_cache_misses_total {}

# HELP proof_cache_hit_rate Cache hit rate
# TYPE proof_cache_hit_rate gauge
proof_cache_hit_rate {}

# HELP proof_verification_success_rate Verification success rate
# TYPE proof_verification_success_rate gauge
proof_verification_success_rate {}
"#,
            self.proofs_generated,
            self.proofs_verified,
            self.verification_failures,
            self.cache_hits,
            self.cache_misses,
            self.cache_hit_rate,
            self.verification_success_rate(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_recording() {
        let telemetry = ProofTelemetry::new();

        telemetry.record_generation("Range", Duration::from_millis(100), 1024);
        telemetry.record_generation("Range", Duration::from_millis(150), 1024);
        telemetry.record_verification("Range", Duration::from_millis(20), true);
        telemetry.record_verification("Range", Duration::from_millis(25), false);

        let stats = telemetry.stats();
        assert_eq!(stats.proofs_generated, 2);
        assert_eq!(stats.proofs_verified, 2);
        assert_eq!(stats.verification_failures, 1);
    }

    #[test]
    fn test_cache_tracking() {
        let telemetry = ProofTelemetry::new();

        telemetry.record_cache_hit();
        telemetry.record_cache_hit();
        telemetry.record_cache_miss();

        let stats = telemetry.stats();
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 1);
        assert!((stats.cache_hit_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_prometheus_output() {
        let telemetry = ProofTelemetry::new();
        telemetry.record_generation("Range", Duration::from_millis(100), 1024);

        let stats = telemetry.stats();
        let prometheus = stats.to_prometheus();

        assert!(prometheus.contains("proof_generation_total 1"));
    }
}
