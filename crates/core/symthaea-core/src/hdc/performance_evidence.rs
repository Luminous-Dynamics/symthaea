//! Feature-neutral measured performance evidence.
//!
//! This contract records measurements; it does not manufacture them. Missing
//! measurements remain optional, and logical byte counts are explicitly distinct
//! from observed physical memory traffic.

use serde::{Deserialize, Serialize};

pub const PERFORMANCE_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkIdentity {
    pub benchmark: String,
    pub workload: String,
    pub resolution: usize,
    pub representation: String,
    pub operation: String,
    pub implementation: String,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProvenance {
    pub commit_sha: String,
    pub toolchain: String,
    pub compiler: String,
    pub target: String,
    pub operating_system: String,
    pub hardware: String,
    pub runner: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub sample_count: u64,
    pub warmup_count: u64,
    /// Number of benchmark-body invocations represented by the measured interval.
    ///
    /// This is deliberately distinct from Criterion's statistical sample count:
    /// one statistical sample may contain many benchmark-body iterations.
    pub iterations: u64,
    /// Number of logical workload items processed by one benchmark-body invocation.
    ///
    /// This is workload semantics, not a claim about physical memory traffic.
    pub batch_size: u64,
    /// Logical bytes processed by one benchmark-body invocation.
    ///
    /// This makes the accounting convention explicit: total logical bytes are
    /// derived as bytes_per_iteration * iterations.
    pub logical_bytes_per_iteration: u64,
    /// Total logical bytes across the measured interval.
    ///
    /// This is stored explicitly so serialized evidence can be independently
    /// audited without reconstructing the value from surrounding metadata.
    pub total_logical_bytes: u64,
    pub elapsed_seconds: f64,
    pub throughput_bytes_per_second: Option<f64>,
    pub allocations: Option<u64>,
    pub peak_resident_bytes: Option<u64>,
    pub physical_memory_bytes: Option<u64>,
    pub energy_joules: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceEvidenceRecord {
    pub schema_version: u32,
    pub identity: BenchmarkIdentity,
    pub provenance: ExecutionProvenance,
    pub measurement: Measurement,
    pub execution_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceEvidenceError {
    UnsupportedSchema(u32),
    EmptyIdentity(&'static str),
    InvalidResolution,
    ZeroSamples,
    InvalidElapsedSeconds,
    ThroughputMismatch,
    IterationCountZero,
    BatchSizeZero,
    LogicalBytesOverflow,
    LogicalBytesTotalMismatch,
    NonFiniteMeasurement(&'static str),
    NegativeMeasurement(&'static str),
    EmptyCommitSha,
    EmptyProvenance(&'static str),
    BenchmarkNotExecuted,
}

impl std::fmt::Display for PerformanceEvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported performance evidence schema: {v}"),
            Self::EmptyIdentity(field) => write!(f, "benchmark identity field is empty: {field}"),
            Self::InvalidResolution => write!(f, "benchmark resolution must be non-zero"),
            Self::ZeroSamples => write!(f, "benchmark sample count must be non-zero"),
            Self::InvalidElapsedSeconds => write!(f, "elapsed seconds must be finite and positive"),
            Self::ThroughputMismatch => write!(f, "throughput does not match total logical bytes / elapsed time"),
            Self::IterationCountZero => write!(f, "measured iteration count must be non-zero"),
            Self::BatchSizeZero => write!(f, "batch size must be non-zero"),
            Self::LogicalBytesOverflow => write!(f, "logical byte accounting overflowed"),
            Self::LogicalBytesTotalMismatch => write!(f, "total logical bytes do not match bytes-per-iteration * iterations"),
            Self::NonFiniteMeasurement(field) => write!(f, "measurement is non-finite: {field}"),
            Self::NegativeMeasurement(field) => write!(f, "measurement is negative: {field}"),
            Self::EmptyCommitSha => write!(f, "commit SHA is empty"),
            Self::EmptyProvenance(field) => write!(f, "provenance field is empty: {field}"),
            Self::BenchmarkNotExecuted => write!(f, "benchmark evidence is not an executed benchmark result"),
        }
    }
}

impl std::error::Error for PerformanceEvidenceError {}

impl PerformanceEvidenceRecord {
    pub fn validate(&self) -> Result<(), PerformanceEvidenceError> {
        if self.schema_version != PERFORMANCE_EVIDENCE_SCHEMA_VERSION {
            return Err(PerformanceEvidenceError::UnsupportedSchema(self.schema_version));
        }
        for (value, name) in [
            (&self.identity.benchmark, "benchmark"),
            (&self.identity.workload, "workload"),
            (&self.identity.representation, "representation"),
            (&self.identity.operation, "operation"),
            (&self.identity.implementation, "implementation"),
        ] {
            if value.is_empty() {
                return Err(PerformanceEvidenceError::EmptyIdentity(name));
            }
        }
        if self.identity.resolution == 0 {
            return Err(PerformanceEvidenceError::InvalidResolution);
        }
        if self.provenance.commit_sha.is_empty() {
            return Err(PerformanceEvidenceError::EmptyCommitSha);
        }
        for (value, name) in [
            (&self.provenance.toolchain, "toolchain"),
            (&self.provenance.compiler, "compiler"),
            (&self.provenance.target, "target"),
            (&self.provenance.operating_system, "operating_system"),
            (&self.provenance.hardware, "hardware"),
            (&self.provenance.runner, "runner"),
        ] {
            if value.is_empty() {
                return Err(PerformanceEvidenceError::EmptyProvenance(name));
            }
        }
        if self.execution_status != "executed" {
            return Err(PerformanceEvidenceError::BenchmarkNotExecuted);
        }
        let m = &self.measurement;
        if m.sample_count == 0 {
            return Err(PerformanceEvidenceError::ZeroSamples);
        }
        if m.iterations == 0 {
            return Err(PerformanceEvidenceError::IterationCountZero);
        }
        if m.batch_size == 0 {
            return Err(PerformanceEvidenceError::BatchSizeZero);
        }
        let expected_total = m
            .logical_bytes_per_iteration
            .checked_mul(m.iterations)
            .ok_or(PerformanceEvidenceError::LogicalBytesOverflow)?;
        if m.total_logical_bytes != expected_total {
            return Err(PerformanceEvidenceError::LogicalBytesTotalMismatch);
        }
        if !m.elapsed_seconds.is_finite() || m.elapsed_seconds <= 0.0 {
            return Err(PerformanceEvidenceError::InvalidElapsedSeconds);
        }
        if let Some(throughput) = m.throughput_bytes_per_second {
            if !throughput.is_finite() || throughput < 0.0 {
                return Err(PerformanceEvidenceError::NonFiniteMeasurement("throughput"));
            }
            let expected = m.total_logical_bytes as f64 / m.elapsed_seconds;
            if (throughput - expected).abs() > expected.max(1.0) * 1e-9 {
                return Err(PerformanceEvidenceError::ThroughputMismatch);
            }
        }
        for (value, name) in [
            (m.allocations.map(|v| v as f64), "allocations"),
            (m.peak_resident_bytes.map(|v| v as f64), "peak_resident_bytes"),
            (m.physical_memory_bytes.map(|v| v as f64), "physical_memory_bytes"),
            (m.energy_joules, "energy_joules"),
        ] {
            if let Some(value) = value {
                if !value.is_finite() {
                    return Err(PerformanceEvidenceError::NonFiniteMeasurement(name));
                }
                if value < 0.0 {
                    return Err(PerformanceEvidenceError::NegativeMeasurement(name));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> PerformanceEvidenceRecord {
        let elapsed = 0.25;
        let bytes_per_iteration = 2 * 131_072 * 4;
        let iterations = 100;
        let total_bytes = bytes_per_iteration * iterations;
        PerformanceEvidenceRecord {
            schema_version: PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
            identity: BenchmarkIdentity {
                benchmark: "simd_continuous".to_owned(),
                workload: "dot".to_owned(),
                resolution: 131_072,
                representation: "continuous_f32".to_owned(),
                operation: "dot".to_owned(),
                implementation: "avx2".to_owned(),
                seed: 42,
            },
            provenance: ExecutionProvenance {
                commit_sha: "deadbeef".to_owned(),
                toolchain: "rustc 1.96.0".to_owned(),
                compiler: "rustc".to_owned(),
                target: "x86_64-unknown-linux-gnu".to_owned(),
                operating_system: "linux".to_owned(),
                hardware: "fixture-hardware".to_owned(),
                runner: "declared-runner".to_owned(),
            },
            measurement: Measurement {
                sample_count: 100,
                warmup_count: 10,
                iterations,
                batch_size: 2,
                logical_bytes_per_iteration: bytes_per_iteration,
                total_logical_bytes: total_bytes,
                elapsed_seconds: elapsed,
                throughput_bytes_per_second: Some(total_bytes as f64 / elapsed),
                allocations: Some(0),
                peak_resident_bytes: Some(2 * 1024 * 1024),
                physical_memory_bytes: None,
                energy_joules: None,
            },
            execution_status: "executed".to_owned(),
        }
    }

    #[test]
    fn executed_record_validates_with_unknown_optional_metrics() {
        record().validate().expect("fixture should validate");
    }

    #[test]
    fn compile_only_status_fails_closed() {
        let mut r = record();
        r.execution_status = "compile-only".to_owned();
        assert!(matches!(r.validate(), Err(PerformanceEvidenceError::BenchmarkNotExecuted)));
    }

    #[test]
    fn missing_energy_is_not_zero() {
        assert!(record().measurement.energy_joules.is_none());
    }

    #[test]
    fn throughput_mismatch_fails_closed() {
        let mut r = record();
        r.measurement.throughput_bytes_per_second = Some(1.0);
        assert!(matches!(r.validate(), Err(PerformanceEvidenceError::ThroughputMismatch)));
    }

    #[test]
    fn total_logical_bytes_are_derived_from_iteration_accounting() {
        let r = record();
        assert_eq!(
            r.measurement.total_logical_bytes,
            r.measurement.logical_bytes_per_iteration * r.measurement.iterations
        );
    }

    #[test]
    fn inconsistent_total_logical_bytes_fails_closed() {
        let mut r = record();
        r.measurement.total_logical_bytes -= 1;
        assert!(matches!(
            r.validate(),
            Err(PerformanceEvidenceError::LogicalBytesTotalMismatch)
        ));
    }

    #[test]
    fn iteration_overflow_fails_closed() {
        let mut r = record();
        r.measurement.iterations = u64::MAX;
        assert!(matches!(
            r.validate(),
            Err(PerformanceEvidenceError::LogicalBytesOverflow)
        ));
    }

    #[test]
    fn zero_batch_size_fails_closed() {
        let mut r = record();
        r.measurement.batch_size = 0;
        assert!(matches!(
            r.validate(),
            Err(PerformanceEvidenceError::BatchSizeZero)
        ));
    }

    #[test]
    fn provenance_is_required {
        let mut r = record();
        r.provenance.hardware.clear();
        assert!(matches!(r.validate(), Err(PerformanceEvidenceError::EmptyProvenance("hardware"))));
    }
}

// Evidence schema remains measurement-only; no benchmark values are embedded here.
