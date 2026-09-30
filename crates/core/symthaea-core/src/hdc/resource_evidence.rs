//! Feature-neutral resource admissibility evidence for HDC workloads.
//!
//! This layer answers only: "does this declared workload fit inside the declared
//! resource envelope?" It deliberately does not claim latency, throughput,
//! bandwidth, allocation efficiency, or energy efficiency.

use serde::{Deserialize, Serialize};

use super::resolution_space::{HdcResolution, ResolutionError};

pub const RESOURCE_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const RESOURCE_QUALIFIED_STATUS: &str = "qualified";
pub const RESOURCE_REJECTED_STATUS: &str = "rejected";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub max_vector_bytes: usize,
    pub max_resident_bytes: usize,
    pub max_peak_temporary_bytes: Option<usize>,
}

impl ResourceBudget {
    pub const fn new(
        max_vector_bytes: usize,
        max_resident_bytes: usize,
        max_peak_temporary_bytes: Option<usize>,
    ) -> Self {
        Self {
            max_vector_bytes,
            max_resident_bytes,
            max_peak_temporary_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceWorkload {
    pub resolution: usize,
    pub representation: String,
    pub element_size_bytes: usize,
    pub resident_vectors: usize,
}

impl ResourceWorkload {
    pub fn validate(&self) -> Result<HdcResolution, ResourceEvidenceError> {
        let resolution = HdcResolution::new(self.resolution)
            .map_err(ResourceEvidenceError::Resolution)?;
        if self.representation.is_empty() {
            return Err(ResourceEvidenceError::EmptyRepresentation);
        }
        if self.element_size_bytes == 0 {
            return Err(ResourceEvidenceError::InvalidElementSize);
        }
        if self.resident_vectors == 0 {
            return Err(ResourceEvidenceError::ZeroResidentVectors);
        }
        Ok(resolution)
    }

    pub fn vector_bytes(&self) -> Result<usize, ResourceEvidenceError> {
        let resolution = self.validate()?;
        resolution
            .checked_bytes(self.element_size_bytes)
            .map_err(ResourceEvidenceError::Resolution)
    }

    pub fn resident_bytes(&self) -> Result<usize, ResourceEvidenceError> {
        self.vector_bytes()?
            .checked_mul(self.resident_vectors)
            .ok_or(ResourceEvidenceError::ResidentBytesOverflow)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceEvidenceRecord {
    pub schema_version: u32,
    pub workload: ResourceWorkload,
    pub budget: ResourceBudget,
    pub vector_bytes: usize,
    pub resident_bytes: usize,
    pub peak_temporary_bytes: Option<usize>,
    pub conversion_bytes: Option<usize>,
    pub provenance_id: String,
    pub qualification_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceEvidenceError {
    Resolution(ResolutionError),
    EmptyRepresentation,
    InvalidElementSize,
    ZeroResidentVectors,
    ResidentBytesOverflow,
    VectorBudgetExceeded { required: usize, limit: usize },
    ResidentBudgetExceeded { required: usize, limit: usize },
    PeakTemporaryBudgetExceeded { required: usize, limit: usize },
    MissingProvenance,
    UnsupportedSchema(u32),
    RepresentationMismatch { expected: String, observed: String },
    ResolutionMismatch { expected: usize, observed: usize },
    VectorBytesMismatch { expected: usize, observed: usize },
    ResidentBytesMismatch { expected: usize, observed: usize },
}

impl std::fmt::Display for ResourceEvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolution(e) => write!(f, "invalid HDC resolution: {e:?}"),
            Self::EmptyRepresentation => write!(f, "resource representation is empty"),
            Self::InvalidElementSize => write!(f, "resource element size must be non-zero"),
            Self::ZeroResidentVectors => write!(f, "resident vector count must be non-zero"),
            Self::ResidentBytesOverflow => write!(f, "resident working-set byte calculation overflowed"),
            Self::VectorBudgetExceeded { required, limit } =>
                write!(f, "vector budget exceeded: required {required}, limit {limit}"),
            Self::ResidentBudgetExceeded { required, limit } =>
                write!(f, "resident budget exceeded: required {required}, limit {limit}"),
            Self::PeakTemporaryBudgetExceeded { required, limit } =>
                write!(f, "peak temporary budget exceeded: required {required}, limit {limit}"),
            Self::MissingProvenance => write!(f, "resource evidence provenance is missing"),
            Self::UnsupportedSchema(v) => write!(f, "unsupported resource evidence schema: {v}"),
            Self::RepresentationMismatch { expected, observed } =>
                write!(f, "resource representation mismatch: expected {expected}, observed {observed}"),
            Self::ResolutionMismatch { expected, observed } =>
                write!(f, "resource resolution mismatch: expected {expected}, observed {observed}"),
            Self::VectorBytesMismatch { expected, observed } =>
                write!(f, "resource vector bytes mismatch: expected {expected}, observed {observed}"),
            Self::ResidentBytesMismatch { expected, observed } =>
                write!(f, "resource resident bytes mismatch: expected {expected}, observed {observed}"),
        }
    }
}

impl std::error::Error for ResourceEvidenceError {}

pub fn qualify_resource(
    record: &ResourceEvidenceRecord,
    expected_resolution: usize,
    expected_representation: &str,
) -> Result<(), ResourceEvidenceError> {
    if record.schema_version != RESOURCE_EVIDENCE_SCHEMA_VERSION {
        return Err(ResourceEvidenceError::UnsupportedSchema(record.schema_version));
    }
    if record.provenance_id.is_empty() {
        return Err(ResourceEvidenceError::MissingProvenance);
    }
    if record.workload.resolution != expected_resolution {
        return Err(ResourceEvidenceError::ResolutionMismatch {
            expected: expected_resolution,
            observed: record.workload.resolution,
        });
    }
    if record.workload.representation != expected_representation {
        return Err(ResourceEvidenceError::RepresentationMismatch {
            expected: expected_representation.to_owned(),
            observed: record.workload.representation.clone(),
        });
    }

    let expected_vector_bytes = record.workload.vector_bytes()?;
    let expected_resident_bytes = record.workload.resident_bytes()?;
    if record.vector_bytes != expected_vector_bytes {
        return Err(ResourceEvidenceError::VectorBytesMismatch {
            expected: expected_vector_bytes,
            observed: record.vector_bytes,
        });
    }
    if record.resident_bytes != expected_resident_bytes {
        return Err(ResourceEvidenceError::ResidentBytesMismatch {
            expected: expected_resident_bytes,
            observed: record.resident_bytes,
        });
    }
    if expected_vector_bytes > record.budget.max_vector_bytes {
        return Err(ResourceEvidenceError::VectorBudgetExceeded {
            required: expected_vector_bytes,
            limit: record.budget.max_vector_bytes,
        });
    }
    if expected_resident_bytes > record.budget.max_resident_bytes {
        return Err(ResourceEvidenceError::ResidentBudgetExceeded {
            required: expected_resident_bytes,
            limit: record.budget.max_resident_bytes,
        });
    }
    if let (Some(required), Some(limit)) =
        (record.peak_temporary_bytes, record.budget.max_peak_temporary_bytes)
    {
        if required > limit {
            return Err(ResourceEvidenceError::PeakTemporaryBudgetExceeded {
                required,
                limit,
            });
        }
    }

    if record.qualification_status != RESOURCE_QUALIFIED_STATUS {
        return Err(ResourceEvidenceError::UnsupportedSchema(
            record.schema_version,
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> ResourceEvidenceRecord {
        let workload = ResourceWorkload {
            resolution: 131_072,
            representation: "continuous_f32".to_owned(),
            element_size_bytes: 4,
            resident_vectors: 4,
        };
        ResourceEvidenceRecord {
            schema_version: RESOURCE_EVIDENCE_SCHEMA_VERSION,
            vector_bytes: workload.vector_bytes().unwrap(),
            resident_bytes: workload.resident_bytes().unwrap(),
            peak_temporary_bytes: Some(64 * 1024),
            conversion_bytes: Some(0),
            provenance_id: "fixture-resource-v1".to_owned(),
            qualification_status: RESOURCE_QUALIFIED_STATUS.to_owned(),
            budget: ResourceBudget::new(512 * 1024, 2 * 1024 * 1024, Some(64 * 1024)),
            workload,
        }
    }

    #[test]
    fn exact_fit_qualifies_without_performance_claim() {
        let record = record();
        assert_eq!(record.vector_bytes, 512 * 1024);
        assert_eq!(record.resident_bytes, 2 * 1024 * 1024);
        qualify_resource(&record, 131_072, "continuous_f32").expect("exact fit qualifies");
    }

    #[test]
    fn vector_budget_excess_fails_closed() {
        let mut record = record();
        record.budget.max_vector_bytes = 512 * 1024 - 1;
        assert!(matches!(
            qualify_resource(&record, 131_072, "continuous_f32"),
            Err(ResourceEvidenceError::VectorBudgetExceeded { .. })
        ));
    }

    #[test]
    fn resident_budget_excess_fails_closed() {
        let mut record = record();
        record.budget.max_resident_bytes = 2 * 1024 * 1024 - 1;
        assert!(matches!(
            qualify_resource(&record, 131_072, "continuous_f32"),
            Err(ResourceEvidenceError::ResidentBudgetExceeded { .. })
        ));
    }

    #[test]
    fn stale_vector_byte_metadata_fails_closed() {
        let mut record = record();
        record.vector_bytes += 4;
        assert!(matches!(
            qualify_resource(&record, 131_072, "continuous_f32"),
            Err(ResourceEvidenceError::VectorBytesMismatch { .. })
        ));
    }

    #[test]
    fn stale_resolution_fails_closed() {
        let record = record();
        assert!(matches!(
            qualify_resource(&record, 262_144, "continuous_f32"),
            Err(ResourceEvidenceError::ResolutionMismatch { .. })
        ));
    }

    #[test]
    fn missing_provenance_fails_closed() {
        let mut record = record();
        record.provenance_id.clear();
        assert!(matches!(
            qualify_resource(&record, 131_072, "continuous_f32"),
            Err(ResourceEvidenceError::MissingProvenance)
        ));
    }

    #[test]
    fn overflow_fails_closed() {
        let workload = ResourceWorkload {
            resolution: usize::MAX,
            representation: "continuous_f32".to_owned(),
            element_size_bytes: usize::MAX,
            resident_vectors: usize::MAX,
        };
        assert!(matches!(
            workload.resident_bytes(),
            Err(ResourceEvidenceError::Resolution(_))
        ));
    }
}
