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
    pub elapsed_seconds: f64,
    pub logical_bytes: u64,
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
            Self::ThroughputMismatch => write!(f, "throughput does not match logical bytes / elapsed time"),
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
        if !m.elapsed_seconds.is_finite() || m.elapsed_seconds <= 0.0 {
            return Err(PerformanceEvidenceError::InvalidElapsedSeconds);
        }
        if let Some(throughput) = m.throughput_bytes_per_second {
            if !throughput.is_finite() || throughput < 0.0 {
                return Err(PerformanceEvidenceError::NonFiniteMeasurement("throughput"));
            }
            let expected = m.logical_bytes as f64 / m.elapsed_seconds;
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
        let bytes = 2 * 131_072 * 4;
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
                elapsed_seconds: elapsed,
                logical_bytes: bytes,
                throughput_bytes_per_second: Some(bytes as f64 / elapsed),
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
    fn provenance_is_required() {
        let mut r = record();
        r.provenance.hardware.clear();
        assert!(matches!(r.validate(), Err(PerformanceEvidenceError::EmptyProvenance("hardware"))));
    }
}

// Evidence schema remains measurement-only; no benchmark values are embedded here.
