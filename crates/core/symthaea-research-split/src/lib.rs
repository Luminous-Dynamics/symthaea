//! Content-addressed research split contracts for leakage-resistant evaluation.
//!
//! This crate records and enforces **declared** sample separation. It can prove facts such as
//! "no evaluation sample shares the declared `spatial-block` or `acquisition` group with
//! development data" and "evaluation begins only after the declared forward-time embargo".
//!
//! It deliberately does **not** prove that the chosen group definitions are scientifically
//! adequate, that two blocks are statistically independent, or that a buffer distance exceeds
//! the process autocorrelation scale. Those stronger claims require domain-specific evidence and
//! remain visible through [`SeparationEvidence`] rather than being collapsed into a trust score.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

const MANIFEST_SCHEMA: &str = "symthaea-research-split/v1";

pub type Result<T> = std::result::Result<T, SplitError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SplitError {
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("split manifest requires at least one development sample")]
    MissingDevelopment,
    #[error("split manifest requires at least one evaluation sample")]
    MissingEvaluation,
    #[error("duplicate sample id: {0}")]
    DuplicateSampleId(String),
    #[error("sample {sample_id} repeats group dimension {dimension}")]
    DuplicateGroupDimension { sample_id: String, dimension: String },
    #[error("group policy repeats dimension: {0}")]
    DuplicatePolicyDimension(String),
    #[error("group-constrained policy requires at least one dimension")]
    EmptyPolicyDimensions,
    #[error("sample {sample_id} is missing required group dimension {dimension}")]
    MissingRequiredGroupDimension { sample_id: String, dimension: String },
    #[error(
        "declared group leakage on {dimension}={value}: {first_role:?} and {second_role:?} are not allowed to share that group"
    )]
    GroupLeakage {
        dimension: String,
        value: String,
        first_role: PartitionRole,
        second_role: PartitionRole,
    },
    #[error(
        "forward evaluation violates embargo: latest development={latest_development_unix_ms}, earliest evaluation={earliest_evaluation_unix_ms}, embargo_ms={embargo_ms}"
    )]
    TemporalEmbargoViolation {
        latest_development_unix_ms: i64,
        earliest_evaluation_unix_ms: i64,
        embargo_ms: u64,
    },
    #[error("split manifest digest mismatch")]
    ManifestDigestMismatch,
    #[error("split manifest serialization failed: {0}")]
    Serialization(String),
}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        Err(SplitError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PartitionRole {
    Training,
    Calibration,
    Evaluation,
}

impl PartitionRole {
    pub fn is_development(self) -> bool {
        matches!(self, Self::Training | Self::Calibration)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupRef {
    /// Group namespace, e.g. `spatial-block`, `acquisition`, `watershed`, or `subject`.
    pub dimension: String,
    /// Caller-defined value within the dimension.
    pub value: String,
}

impl GroupRef {
    pub fn new(dimension: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let dimension = dimension.into();
        let value = value.into();
        non_empty(&dimension, "group dimension")?;
        non_empty(&value, "group value")?;
        Ok(Self { dimension, value })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitUnit {
    pub sample_id: String,
    /// Timestamp used only when a temporal policy is selected. The contract does not prescribe
    /// whether this is acquisition, event, publication, or another domain clock; that meaning
    /// belongs in the dataset/split documentation and must remain consistent within a manifest.
    pub observed_at_unix_ms: i64,
    /// Content/provenance digest of the immutable sample record or source artifact.
    pub content_digest: String,
    /// At most one value per dimension for this unit.
    pub groups: Vec<GroupRef>,
}

impl SplitUnit {
    pub fn new(
        sample_id: impl Into<String>,
        observed_at_unix_ms: i64,
        content_digest: impl Into<String>,
        groups: Vec<GroupRef>,
    ) -> Result<Self> {
        let unit = Self {
            sample_id: sample_id.into(),
            observed_at_unix_ms,
            content_digest: content_digest.into(),
            groups,
        };
        unit.validate()?;
        Ok(unit)
    }

    fn validate(&self) -> Result<()> {
        non_empty(&self.sample_id, "sample id")?;
        non_empty(&self.content_digest, "sample content digest")?;
        let mut dimensions = HashSet::new();
        for group in &self.groups {
            non_empty(&group.dimension, "group dimension")?;
            non_empty(&group.value, "group value")?;
            if !dimensions.insert(group.dimension.as_str()) {
                return Err(SplitError::DuplicateGroupDimension {
                    sample_id: self.sample_id.clone(),
                    dimension: group.dimension.clone(),
                });
            }
        }
        Ok(())
    }

    fn group_value(&self, dimension: &str) -> Option<&str> {
        self.groups
            .iter()
            .find(|group| group.dimension == dimension)
            .map(|group| group.value.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssignedUnit {
    pub unit: SplitUnit,
    pub role: PartitionRole,
}

impl AssignedUnit {
    pub fn new(unit: SplitUnit, role: PartitionRole) -> Self {
        Self { unit, role }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupSeparationPolicy {
    /// No group-exclusivity assertion. Group metadata may still be retained for audit.
    None,
    /// Evaluation may not share any configured group value with Training or Calibration.
    /// Training and Calibration may share groups with each other.
    EvaluationDisjoint { dimensions: Vec<String> },
    /// A configured group value may occur in only one partition role across the manifest.
    AllRolesDisjoint { dimensions: Vec<String> },
}

impl GroupSeparationPolicy {
    fn dimensions(&self) -> &[String] {
        match self {
            Self::None => &[],
            Self::EvaluationDisjoint { dimensions } | Self::AllRolesDisjoint { dimensions } => {
                dimensions
            }
        }
    }

    fn validate(&self) -> Result<()> {
        let dimensions = self.dimensions();
        if !matches!(self, Self::None) && dimensions.is_empty() {
            return Err(SplitError::EmptyPolicyDimensions);
        }
        let mut seen = HashSet::new();
        for dimension in dimensions {
            non_empty(dimension, "group policy dimension")?;
            if !seen.insert(dimension.as_str()) {
                return Err(SplitError::DuplicatePolicyDimension(dimension.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalSeparationPolicy {
    None,
    /// Every evaluation sample must be at least `embargo_ms` later than the latest Training or
    /// Calibration sample. This is intentionally stricter than merely checking mean dates.
    ForwardEvaluation { embargo_ms: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SeparationEvidenceKind {
    SpatialBuffer,
    TemporalAutocorrelation,
    AcquisitionSeparation,
    GroupDefinition,
    ExternalAudit,
    Other,
}

/// Evidence about separation adequacy that cannot be proven from equality/timestamps alone.
///
/// Example: an artifact may show that 20 km exceeds an empirically estimated spatial
/// autocorrelation range. Recording that artifact makes the claim attributable; this type does
/// not independently establish that the artifact or conclusion is correct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeparationEvidence {
    pub kind: SeparationEvidenceKind,
    pub statement: String,
    pub artifact_digest: String,
}

impl SeparationEvidence {
    pub fn new(
        kind: SeparationEvidenceKind,
        statement: impl Into<String>,
        artifact_digest: impl Into<String>,
    ) -> Result<Self> {
        let value = Self {
            kind,
            statement: statement.into(),
            artifact_digest: artifact_digest.into(),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<()> {
        non_empty(&self.statement, "separation evidence statement")?;
        non_empty(&self.artifact_digest, "separation evidence artifact digest")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchSplitManifest {
    pub manifest_id: String,
    pub assignments: Vec<AssignedUnit>,
    pub group_policy: GroupSeparationPolicy,
    pub temporal_policy: TemporalSeparationPolicy,
    pub separation_evidence: Vec<SeparationEvidence>,
    pub manifest_digest: String,
}

#[derive(Serialize)]
struct ManifestDigestView<'a> {
    schema: &'static str,
    manifest_id: &'a str,
    assignments: &'a [AssignedUnit],
    group_policy: &'a GroupSeparationPolicy,
    temporal_policy: &'a TemporalSeparationPolicy,
    separation_evidence: &'a [SeparationEvidence],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitSummary {
    pub training_count: usize,
    pub calibration_count: usize,
    pub evaluation_count: usize,
    pub latest_development_unix_ms: Option<i64>,
    pub earliest_evaluation_unix_ms: Option<i64>,
    /// Signed observed gap `earliest evaluation - latest development` before comparing it to an
    /// embargo. Kept as i128 so extreme i64 endpoints cannot overflow the diagnostic.
    pub observed_temporal_gap_ms: Option<i128>,
}

impl ResearchSplitManifest {
    pub fn new(
        manifest_id: impl Into<String>,
        assignments: Vec<AssignedUnit>,
        group_policy: GroupSeparationPolicy,
        temporal_policy: TemporalSeparationPolicy,
        separation_evidence: Vec<SeparationEvidence>,
    ) -> Result<Self> {
        let mut manifest = Self {
            manifest_id: manifest_id.into(),
            assignments,
            group_policy,
            temporal_policy,
            separation_evidence,
            manifest_digest: String::new(),
        };
        manifest.validate()?;
        manifest.manifest_digest = manifest.compute_digest()?;
        Ok(manifest)
    }

    fn digest_view(&self) -> ManifestDigestView<'_> {
        ManifestDigestView {
            schema: MANIFEST_SCHEMA,
            manifest_id: &self.manifest_id,
            assignments: &self.assignments,
            group_policy: &self.group_policy,
            temporal_policy: &self.temporal_policy,
            separation_evidence: &self.separation_evidence,
        }
    }

    pub fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&self.digest_view())
            .map_err(|error| SplitError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn verify_digest(&self) -> Result<()> {
        self.validate()?;
        if self.compute_digest()? != self.manifest_digest {
            return Err(SplitError::ManifestDigestMismatch);
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        non_empty(&self.manifest_id, "split manifest id")?;
        self.group_policy.validate()?;
        for evidence in &self.separation_evidence {
            evidence.validate()?;
        }

        let mut sample_ids = HashSet::new();
        let mut has_development = false;
        let mut has_evaluation = false;
        for assignment in &self.assignments {
            assignment.unit.validate()?;
            if !sample_ids.insert(assignment.unit.sample_id.as_str()) {
                return Err(SplitError::DuplicateSampleId(
                    assignment.unit.sample_id.clone(),
                ));
            }
            if assignment.role.is_development() {
                has_development = true;
            } else if assignment.role == PartitionRole::Evaluation {
                has_evaluation = true;
            }
        }
        if !has_development {
            return Err(SplitError::MissingDevelopment);
        }
        if !has_evaluation {
            return Err(SplitError::MissingEvaluation);
        }

        self.validate_required_groups()?;
        self.validate_group_separation()?;
        self.validate_temporal_separation()?;
        Ok(())
    }

    fn validate_required_groups(&self) -> Result<()> {
        for dimension in self.group_policy.dimensions() {
            for assignment in &self.assignments {
                if assignment.unit.group_value(dimension).is_none() {
                    return Err(SplitError::MissingRequiredGroupDimension {
                        sample_id: assignment.unit.sample_id.clone(),
                        dimension: dimension.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_group_separation(&self) -> Result<()> {
        match &self.group_policy {
            GroupSeparationPolicy::None => Ok(()),
            GroupSeparationPolicy::EvaluationDisjoint { dimensions } => {
                for dimension in dimensions {
                    let mut seen: HashMap<&str, PartitionRole> = HashMap::new();
                    for assignment in &self.assignments {
                        let value = assignment.unit.group_value(dimension).expect(
                            "required group dimensions are checked before group separation",
                        );
                        if let Some(first_role) = seen.get(value).copied() {
                            let crosses_evaluation = (first_role == PartitionRole::Evaluation)
                                != (assignment.role == PartitionRole::Evaluation);
                            if crosses_evaluation {
                                return Err(SplitError::GroupLeakage {
                                    dimension: dimension.clone(),
                                    value: value.to_string(),
                                    first_role,
                                    second_role: assignment.role,
                                });
                            }
                        } else {
                            seen.insert(value, assignment.role);
                        }
                    }
                }
                Ok(())
            }
            GroupSeparationPolicy::AllRolesDisjoint { dimensions } => {
                for dimension in dimensions {
                    let mut seen: HashMap<&str, PartitionRole> = HashMap::new();
                    for assignment in &self.assignments {
                        let value = assignment.unit.group_value(dimension).expect(
                            "required group dimensions are checked before group separation",
                        );
                        if let Some(first_role) = seen.get(value).copied() {
                            if first_role != assignment.role {
                                return Err(SplitError::GroupLeakage {
                                    dimension: dimension.clone(),
                                    value: value.to_string(),
                                    first_role,
                                    second_role: assignment.role,
                                });
                            }
                        } else {
                            seen.insert(value, assignment.role);
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn validate_temporal_separation(&self) -> Result<()> {
        let TemporalSeparationPolicy::ForwardEvaluation { embargo_ms } = self.temporal_policy else {
            return Ok(());
        };
        let summary = self.summary();
        let latest_development_unix_ms = summary
            .latest_development_unix_ms
            .expect("development presence validated before temporal policy");
        let earliest_evaluation_unix_ms = summary
            .earliest_evaluation_unix_ms
            .expect("evaluation presence validated before temporal policy");
        let required_evaluation =
            i128::from(latest_development_unix_ms) + i128::from(embargo_ms);
        if i128::from(earliest_evaluation_unix_ms) < required_evaluation {
            return Err(SplitError::TemporalEmbargoViolation {
                latest_development_unix_ms,
                earliest_evaluation_unix_ms,
                embargo_ms,
            });
        }
        Ok(())
    }

    pub fn summary(&self) -> SplitSummary {
        let mut training_count = 0;
        let mut calibration_count = 0;
        let mut evaluation_count = 0;
        let mut latest_development_unix_ms: Option<i64> = None;
        let mut earliest_evaluation_unix_ms: Option<i64> = None;

        for assignment in &self.assignments {
            match assignment.role {
                PartitionRole::Training => training_count += 1,
                PartitionRole::Calibration => calibration_count += 1,
                PartitionRole::Evaluation => evaluation_count += 1,
            }
            if assignment.role.is_development() {
                latest_development_unix_ms = Some(
                    latest_development_unix_ms
                        .map_or(assignment.unit.observed_at_unix_ms, |current| {
                            current.max(assignment.unit.observed_at_unix_ms)
                        }),
                );
            } else {
                earliest_evaluation_unix_ms = Some(
                    earliest_evaluation_unix_ms
                        .map_or(assignment.unit.observed_at_unix_ms, |current| {
                            current.min(assignment.unit.observed_at_unix_ms)
                        }),
                );
            }
        }

        let observed_temporal_gap_ms = latest_development_unix_ms
            .zip(earliest_evaluation_unix_ms)
            .map(|(latest, earliest)| i128::from(earliest) - i128::from(latest));

        SplitSummary {
            training_count,
            calibration_count,
            evaluation_count,
            latest_development_unix_ms,
            earliest_evaluation_unix_ms,
            observed_temporal_gap_ms,
        }
    }
}
