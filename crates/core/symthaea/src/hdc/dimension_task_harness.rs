//! Deterministic task-quality/performance/resource harness across HDC dimensions.
//!
//! This is deliberately a synthetic retrieval/classification fixture, not a
//! claim about a production dataset. It holds the task protocol constant while
//! changing only the continuous-HV resolution. The output uses the existing
//! evidence contracts and produces content-addressed artifact references that
//! can be consumed by `DimensionFrontierManifest`.
//!
//! The task is intentionally simple: each class owns a deterministic prototype;
//! held-out queries are generated as a fixed mixture of the class prototype and
//! independent noise. Classification is nearest-prototype cosine similarity.
//! This measures whether the representation preserves a controlled signal as
//! dimensionality changes without importing external data or user state.

use super::cost_quality_join::EvidenceReference as JoinEvidenceReference;
use super::performance_evidence::{
    BenchmarkIdentity, ExecutionProvenance as PerformanceProvenance, Measurement,
    PerformanceEvidenceRecord, PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
};
use super::resource_evidence::{
    qualify_resource, ResourceBudget, ResourceEvidenceRecord, ResourceWorkload,
    RESOURCE_EVIDENCE_SCHEMA_VERSION, RESOURCE_QUALIFIED_STATUS,
};
use super::task_quality_evidence::{
    ExecutionProvenance as QualityProvenance, MetricDirection, QualityMeasurement,
    QualityMetric, TaskIdentity, TaskQualityEvidenceRecord, UncertaintyInterval,
    TASK_QUALITY_EVIDENCE_SCHEMA_VERSION,
};
use super::dimension_frontier::{
    DimensionFrontierManifest, DimensionFrontierRow, FrontierReference,
    DIMENSION_FRONTIER_SCHEMA_VERSION,
};
use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub const DIMENSION_TASK_HARNESS_SCHEMA_VERSION: u32 = 1;
pub const DIMENSION_TASK_HARNESS_DOMAIN: &[u8] = b"symthaea:hdc-dimension-task-harness";
pub const TASK_NAME: &str = "hdc_synthetic_prototype_retrieval";
pub const SCENARIO_SET: &str = "synthetic-prototype-retrieval-v1";
pub const SCENARIO_REVISION: &str = "sha256:deterministic-fixture-v1";
pub const SPLIT: &str = "held-out";
pub const PROTOCOL: &str = "4-class-4-query-nearest-prototype-cosine";
pub const REPRESENTATION: &str = "continuous_f32";
pub const MODEL_REVISION: &str = "prototype-mixture-v1";
pub const DEFAULT_SEED: u64 = 0x4844_4352_4554_5226;
pub const DEFAULT_DIMENSIONS: &[usize] =
    &[1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144];
pub const CLASS_COUNT: usize = 4;
pub const QUERIES_PER_CLASS: usize = 4;
pub const NOISE_WEIGHT: f32 = 0.20;
pub const PROTOTYPE_WEIGHT: f32 = 0.80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionTaskSpec {
    pub schema_version: u32,
    pub dimensions: &'static [usize],
    pub seed: u64,
    pub queries_per_class: usize,
}

impl Default for DimensionTaskSpec {
    fn default() -> Self {
        Self {
            schema_version: DIMENSION_TASK_HARNESS_SCHEMA_VERSION,
            dimensions: DEFAULT_DIMENSIONS,
            seed: DEFAULT_SEED,
            queries_per_class: QUERIES_PER_CLASS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionTaskRow {
    pub resolution: usize,
    pub correct: u64,
    pub total: u64,
    pub accuracy: f64,
    pub mean_margin: f64,
    pub elapsed_seconds: f64,
    pub logical_bytes: u64,
    pub task_quality: TaskQualityEvidenceRecord,
    pub performance: PerformanceEvidenceRecord,
    pub resource: ResourceEvidenceRecord,
    pub task_quality_digest: String,
    pub performance_digest: String,
    pub resource_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionTaskEvidence {
    pub schema_version: u32,
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub seed: u64,
    pub rows: Vec<DimensionTaskRow>,
    pub frontier: DimensionFrontierManifest,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimensionTaskError {
    UnsupportedSchema(u32),
    EmptyDimensions,
    InvalidDimension(usize),
    UnsortedDimensions,
    ZeroQueries,
    InvalidScore,
    Resource(String),
    Evidence(String),
}

impl std::fmt::Display for DimensionTaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported dimension task schema: {v}"),
            Self::EmptyDimensions => f.write_str("dimension task has no dimensions"),
            Self::InvalidDimension(d) => write!(f, "invalid dimension: {d}"),
            Self::UnsortedDimensions => f.write_str("dimensions must be strictly ascending"),
            Self::ZeroQueries => f.write_str("queries_per_class must be non-zero"),
            Self::InvalidScore => f.write_str("task score is non-finite"),
            Self::Resource(e) => write!(f, "resource evidence failed: {e}"),
            Self::Evidence(e) => write!(f, "evidence validation failed: {e}"),
        }
    }
}

impl std::error::Error for DimensionTaskError {}

impl DimensionTaskSpec {
    pub fn validate(&self) -> Result<(), DimensionTaskError> {
        if self.schema_version != DIMENSION_TASK_HARNESS_SCHEMA_VERSION {
            return Err(DimensionTaskError::UnsupportedSchema(self.schema_version));
        }
        if self.dimensions.is_empty() {
            return Err(DimensionTaskError::EmptyDimensions);
        }
        if self.queries_per_class == 0 {
            return Err(DimensionTaskError::ZeroQueries);
        }
        let mut previous = 0;
        for &dimension in self.dimensions {
            if dimension == 0 || !dimension.is_power_of_two() {
                return Err(DimensionTaskError::InvalidDimension(dimension));
            }
            if dimension <= previous {
                return Err(DimensionTaskError::UnsortedDimensions);
            }
            previous = dimension;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(DIMENSION_TASK_HARNESS_DOMAIN);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.seed.to_be_bytes());
        bytes.extend_from_slice(&(self.queries_per_class as u64).to_be_bytes());
        bytes.extend_from_slice(&(self.dimensions.len() as u64).to_be_bytes());
        for &dimension in self.dimensions {
            bytes.extend_from_slice(&(dimension as u64).to_be_bytes());
        }
        bytes
    }

    pub fn identity_digest(&self) -> String {
        let digest = Sha256::digest(self.canonical_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

fn provenance() -> QualityProvenance {
    QualityProvenance {
        commit_sha: option_env!("GITHUB_SHA").unwrap_or("local").to_owned(),
        toolchain: option_env!("RUSTUP_TOOLCHAIN").unwrap_or("rust").to_owned(),
        compiler: "rustc".to_owned(),
        target: std::env::consts::ARCH.to_owned(),
        operating_system: std::env::consts::OS.to_owned(),
        hardware: "runtime-declared".to_owned(),
        runner: "dimension-task-harness".to_owned(),
    }
}

fn performance_provenance() -> PerformanceProvenance {
    PerformanceProvenance {
        commit_sha: option_env!("GITHUB_SHA").unwrap_or("local").to_owned(),
        toolchain: option_env!("RUSTUP_TOOLCHAIN").unwrap_or("rust").to_owned(),
        compiler: "rustc".to_owned(),
        target: std::env::consts::ARCH.to_owned(),
        operating_system: std::env::consts::OS.to_owned(),
        hardware: "runtime-declared".to_owned(),
        runner: "dimension-task-harness".to_owned(),
    }
}

fn artifact_digest<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("evidence structs must serialize");
    let digest = Sha256::digest(bytes);
    let mut out = String::from("sha256:");
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn fixture_seed(seed: u64, class_index: usize, query_index: usize) -> u64 {
    let mut x = seed
        ^ (class_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (query_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn wilson_95(correct: u64, total: u64) -> (f64, f64) {
    let n = total as f64;
    let z = 1.959_963_984_540_054;
    let p = correct as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width =
        z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    ((center - half_width).max(0.0), (center + half_width).min(1.0))
}

fn make_query(prototype: &ContinuousHV, seed: u64) -> ContinuousHV {
    let noise = ContinuousHV::random(prototype.dim(), seed);
    let values = prototype
        .values
        .iter()
        .zip(noise.values.iter())
        .map(|(&p, &n)| PROTOTYPE_WEIGHT * p + NOISE_WEIGHT * n)
        .collect();
    ContinuousHV::from_vec(values)
}

fn reference(kind: &str, schema_version: u32, digest: String, id: String) -> FrontierReference {
    FrontierReference {
        kind: kind.to_owned(),
        schema_version,
        artifact_digest: digest,
        artifact_id: id,
    }
}

fn run_row(
    resolution: usize,
    seed: u64,
    queries_per_class: usize,
) -> Result<DimensionTaskRow, DimensionTaskError> {
    let prototypes: Vec<ContinuousHV> = (0..CLASS_COUNT)
        .map(|class| ContinuousHV::random(resolution, seed.wrapping_add(class as u64 + 1)))
        .collect();

    let total_queries = CLASS_COUNT * queries_per_class;
    let logical_bytes_per_query = (CLASS_COUNT + 1)
        .checked_mul(resolution)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(DimensionTaskError::InvalidDimension(resolution))?;
    let start = Instant::now();

    let mut correct = 0u64;
    let mut margin_sum = 0.0f64;
    for class in 0..CLASS_COUNT {
        for query_index in 0..queries_per_class {
            let query = make_query(
                &prototypes[class],
                fixture_seed(seed, class, query_index),
            );
            let mut best_class = 0usize;
            let mut best = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for (candidate, prototype) in prototypes.iter().enumerate() {
                let similarity = query.similarity(prototype);
                if similarity > best {
                    second = best;
                    best = similarity;
                    best_class = candidate;
                } else if similarity > second {
                    second = similarity;
                }
            }
            if best_class == class {
                correct += 1;
            }
            margin_sum += f64::from(best - second);
        }
    }

    let elapsed_seconds = start.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
    let accuracy = correct as f64 / total_queries as f64;
    if !accuracy.is_finite() {
        return Err(DimensionTaskError::InvalidScore);
    }

    let task_quality = TaskQualityEvidenceRecord {
        schema_version: TASK_QUALITY_EVIDENCE_SCHEMA_VERSION,
        identity: TaskIdentity {
            task: TASK_NAME.to_owned(),
            scenario_set: SCENARIO_SET.to_owned(),
            scenario_revision: SCENARIO_REVISION.to_owned(),
            split: SPLIT.to_owned(),
            protocol: PROTOCOL.to_owned(),
            resolution,
            representation: REPRESENTATION.to_owned(),
            model_revision: MODEL_REVISION.to_owned(),
            seed,
        },
        metric: QualityMetric {
            name: "accuracy".to_owned(),
            direction: MetricDirection::HigherIsBetter,
            unit: "fraction".to_owned(),
        },
        measurement: QualityMeasurement {
            score: accuracy,
            sample_count: total_queries as u64,
            uncertainty: Some({
                let (lower, upper) = wilson_95(correct, total_queries as u64);
                UncertaintyInterval {
                    method: "wilson-95".to_owned(),
                    lower,
                    upper,
                }
            }),
        },
        provenance: provenance(),
        upstream_evidence: Vec::new(),
        execution_status: "executed".to_owned(),
        qualification_status: "fixture-qualified".to_owned(),
    };
    task_quality
        .validate()
        .map_err(|e| DimensionTaskError::Evidence(e.to_string()))?;

    let logical_bytes = logical_bytes_per_query
        .checked_mul(total_queries as u64)
        .ok_or(DimensionTaskError::InvalidDimension(resolution))?;
    let performance = PerformanceEvidenceRecord {
        schema_version: PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
        identity: BenchmarkIdentity {
            benchmark: TASK_NAME.to_owned(),
            workload: "nearest-prototype".to_owned(),
            resolution,
            representation: REPRESENTATION.to_owned(),
            operation: "cosine-similarity".to_owned(),
            implementation: "continuous-hv".to_owned(),
            seed,
        },
        provenance: performance_provenance(),
        measurement: Measurement {
            sample_count: 1,
            warmup_count: 0,
            iterations: 1,
            batch_size: total_queries as u64,
            logical_bytes_per_iteration: logical_bytes,
            total_logical_bytes: logical_bytes,
            elapsed_seconds,
            throughput_bytes_per_second: Some(logical_bytes as f64 / elapsed_seconds),
            allocations: None,
            peak_resident_bytes: None,
            physical_memory_bytes: None,
            energy_joules: None,
        },
        execution_status: "executed".to_owned(),
    };
    performance
        .validate()
        .map_err(|e| DimensionTaskError::Evidence(e.to_string()))?;

    let workload = ResourceWorkload {
        resolution,
        representation: REPRESENTATION.to_owned(),
        element_size_bytes: std::mem::size_of::<f32>(),
        resident_vectors: CLASS_COUNT + 1,
    };
    let vector_bytes = workload
        .vector_bytes()
        .map_err(|e| DimensionTaskError::Resource(e.to_string()))?;
    let resident_bytes = workload
        .resident_bytes()
        .map_err(|e| DimensionTaskError::Resource(e.to_string()))?;
    let resource = ResourceEvidenceRecord {
        schema_version: RESOURCE_EVIDENCE_SCHEMA_VERSION,
        workload,
        budget: Some(ResourceBudget::new(
            vector_bytes,
            resident_bytes,
            None,
        )),
        vector_bytes,
        resident_bytes,
        peak_temporary_bytes: None,
        conversion_bytes: Some(0),
        provenance_id: format!("{}-{}-{}", TASK_NAME, resolution, seed),
        qualification_status: RESOURCE_QUALIFIED_STATUS.to_owned(),
    };
    qualify_resource(&resource, resolution, REPRESENTATION)
        .map_err(|e| DimensionTaskError::Resource(e.to_string()))?;

    let task_quality_digest = artifact_digest(&task_quality);
    let performance_digest = artifact_digest(&performance);
    let resource_digest = artifact_digest(&resource);

    Ok(DimensionTaskRow {
        resolution,
        correct,
        total: total_queries as u64,
        accuracy,
        mean_margin: margin_sum / total_queries as f64,
        elapsed_seconds,
        logical_bytes,
        task_quality,
        performance,
        resource,
        task_quality_digest,
        performance_digest,
        resource_digest,
    })
}

pub fn run_dimension_task(
    spec: DimensionTaskSpec,
) -> Result<DimensionTaskEvidence, DimensionTaskError> {
    spec.validate()?;
    let rows: Vec<DimensionTaskRow> = spec
        .dimensions
        .iter()
        .copied()
        .map(|dimension| run_row(dimension, spec.seed, spec.queries_per_class))
        .collect::<Result<_, _>>()?;

    let frontier_rows = rows
        .iter()
        .map(|row| DimensionFrontierRow {
            resolution: row.resolution,
            task_quality: reference(
                "task_quality",
                TASK_QUALITY_EVIDENCE_SCHEMA_VERSION,
                row.task_quality_digest.clone(),
                format!("{}:task-quality:{}", TASK_NAME, row.resolution),
            ),
            performance: reference(
                "performance",
                PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
                row.performance_digest.clone(),
                format!("{}:performance:{}", TASK_NAME, row.resolution),
            ),
            resource: reference(
                "resource",
                RESOURCE_EVIDENCE_SCHEMA_VERSION,
                row.resource_digest.clone(),
                format!("{}:resource:{}", TASK_NAME, row.resolution),
            ),
            structural_sweep: None,
        })
        .collect();

    let frontier = DimensionFrontierManifest {
        schema_version: DIMENSION_FRONTIER_SCHEMA_VERSION,
        task: TASK_NAME.to_owned(),
        scenario_set: SCENARIO_SET.to_owned(),
        scenario_revision: SCENARIO_REVISION.to_owned(),
        split: SPLIT.to_owned(),
        protocol: PROTOCOL.to_owned(),
        representation: REPRESENTATION.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        rows: frontier_rows,
    };
    frontier
        .validate()
        .map_err(|e| DimensionTaskError::Evidence(e.to_string()))?;

    let mut evidence = DimensionTaskEvidence {
        schema_version: DIMENSION_TASK_HARNESS_SCHEMA_VERSION,
        task: TASK_NAME.to_owned(),
        scenario_set: SCENARIO_SET.to_owned(),
        scenario_revision: SCENARIO_REVISION.to_owned(),
        seed: spec.seed,
        rows,
        frontier,
        artifact_digest: String::new(),
    };
    evidence.artifact_digest = artifact_digest(&evidence);
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_ordered_and_identity_stable() {
        let spec = DimensionTaskSpec::default();
        spec.validate().unwrap();
        assert_eq!(spec.identity_digest(), DimensionTaskSpec::default().identity_digest());
    }

    #[test]
    fn task_is_deterministic() {
        let spec = DimensionTaskSpec {
            dimensions: &[1_024, 2_048],
            ..Default::default()
        };
        let a = run_dimension_task(spec).unwrap();
        let b = run_dimension_task(spec).unwrap();
        assert_eq!(a.rows.iter().map(|r| r.accuracy).collect::<Vec<_>>(),
                   b.rows.iter().map(|r| r.accuracy).collect::<Vec<_>>());
        assert_eq!(
            a.rows.iter().map(|r| r.task_quality_digest).collect::<Vec<_>>(),
            b.rows.iter().map(|r| r.task_quality_digest).collect::<Vec<_>>()
        );
        assert_eq!(a.frontier.artifact_digest(), b.frontier.artifact_digest());
    }

    #[test]
    fn malformed_dimension_fails_closed() {
        let spec = DimensionTaskSpec {
            dimensions: &[1_024, 3_000],
            ..Default::default()
        };
        assert!(matches!(spec.validate(), Err(DimensionTaskError::InvalidDimension(3_000))));
    }

    #[test]
    fn quality_is_a_real_task_measurement_not_structural_geometry() {
        let spec = DimensionTaskSpec {
            dimensions: &[1_024],
            queries_per_class: 2,
            ..Default::default()
        };
        let evidence = run_dimension_task(spec).unwrap();
        assert_eq!(evidence.rows[0].total, 8);
        assert!(evidence.rows[0].accuracy.is_finite());
        assert!(evidence.rows[0].task_quality.measurement.uncertainty.is_some());
    }

    #[test]
    fn frontier_references_match_rows() {
        let spec = DimensionTaskSpec {
            dimensions: &[1_024, 2_048],
            queries_per_class: 1,
            ..Default::default()
        };
        let evidence = run_dimension_task(spec).unwrap();
        for (row, frontier) in evidence.rows.iter().zip(evidence.frontier.rows.iter()) {
            assert_eq!(row.resolution, frontier.resolution);
            assert_eq!(row.task_quality_digest, frontier.task_quality.artifact_digest);
            assert_eq!(row.performance_digest, frontier.performance.artifact_digest);
            assert_eq!(row.resource_digest, frontier.resource.artifact_digest);
        }
    }
}
