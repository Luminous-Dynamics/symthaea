//! Content-addressed model-selection provenance for Symthaea research.
//!
//! A leakage-safe train/calibration/evaluation split and leakage-safe fitted artifacts are still
//! not enough if candidate selection looks at final Evaluation outcomes. This crate binds model
//! selection to an exact frozen split, exact fitted artifacts, one declared selection metric,
//! and an explicit development-role policy. Evaluation samples are always forbidden.
//!
//! The initial contract deliberately supports one scalar selection metric and deterministic
//! tie-breaking. Multi-objective scientific comparison belongs in higher-level outcome analysis;
//! this layer records the narrower engineering act of choosing one frozen candidate before final
//! evaluation.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};
use symthaea_research_fit::FitArtifactManifest;
use symthaea_research_split::{PartitionRole, ResearchSplitManifest};
use thiserror::Error;

const SELECTION_SCHEMA: &str = "symthaea-research-selection/v1";

pub type Result<T> = std::result::Result<T, SelectionError>;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SelectionError {
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("model selection requires at least two candidates")]
    TooFewCandidates,
    #[error("duplicate candidate id: {0}")]
    DuplicateCandidate(String),
    #[error("duplicate candidate/sample metric observation: {candidate_id}/{sample_id}")]
    DuplicateObservation { candidate_id: String, sample_id: String },
    #[error("selection observation references unknown candidate: {0}")]
    UnknownCandidate(String),
    #[error("selection observation references sample outside split: {0}")]
    UnknownSample(String),
    #[error("evaluation sample must not influence model selection: {0}")]
    EvaluationLeakage(String),
    #[error("training sample is not allowed by CalibrationOnly selection policy: {0}")]
    TrainingLeakage(String),
    #[error("selection metric value must be finite for {candidate_id}/{sample_id}")]
    NonFiniteMetric { candidate_id: String, sample_id: String },
    #[error("candidate {0} has no selection observations")]
    MissingCandidateObservations(String),
    #[error("candidate selection sample sets differ: {candidate_id}")]
    UnequalCandidateSampleSet { candidate_id: String },
    #[error("split manifest digest does not match selection manifest")]
    SplitManifestMismatch,
    #[error("fit manifest not supplied for candidate: {0}")]
    MissingFitManifest(String),
    #[error("fit manifest digest does not match candidate: {0}")]
    FitManifestMismatch(String),
    #[error("fit output artifact digest does not match candidate: {0}")]
    FitArtifactMismatch(String),
    #[error("selection observation content digest changed for sample {0}")]
    SampleDigestMismatch(String),
    #[error("selection observation role changed for sample {sample_id}: recorded={recorded:?}, actual={actual:?}")]
    SampleRoleMismatch {
        sample_id: String,
        recorded: PartitionRole,
        actual: PartitionRole,
    },
    #[error("recorded aggregate differs from recomputed aggregate for candidate {0}")]
    AggregateMismatch(String),
    #[error("recorded winner differs from deterministic recomputation")]
    WinnerMismatch,
    #[error("selection manifest digest mismatch")]
    ManifestDigestMismatch,
    #[error("selection serialization failed: {0}")]
    Serialization(String),
    #[error("split validation failed: {0}")]
    Split(String),
    #[error("fit validation failed: {0}")]
    Fit(String),
}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        Err(SelectionError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionDirection {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionRolePolicy {
    /// Only the explicit Calibration partition may influence candidate choice.
    CalibrationOnly,
    /// Training and Calibration may influence candidate choice. Evaluation remains forbidden.
    TrainingAndCalibration,
}

impl SelectionRolePolicy {
    fn allows(self, role: PartitionRole) -> bool {
        match self {
            Self::CalibrationOnly => role == PartitionRole::Calibration,
            Self::TrainingAndCalibration => {
                matches!(role, PartitionRole::Training | PartitionRole::Calibration)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TieBreakPolicy {
    /// If the declared aggregate metric is exactly equal, choose the lexicographically smallest
    /// candidate id. This is intentionally boring and deterministic.
    LexicographicCandidateId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionCandidate {
    pub candidate_id: String,
    pub fit_manifest_digest: String,
    pub output_artifact_digest: String,
}

impl SelectionCandidate {
    pub fn from_fit(
        candidate_id: impl Into<String>,
        fit: &FitArtifactManifest,
        split: &ResearchSplitManifest,
    ) -> Result<Self> {
        fit.verify_against_split(split)
            .map_err(|error| SelectionError::Fit(error.to_string()))?;
        let candidate_id = candidate_id.into();
        non_empty(&candidate_id, "candidate id")?;
        Ok(Self {
            candidate_id,
            fit_manifest_digest: fit.manifest_digest.clone(),
            output_artifact_digest: fit.output_artifact_digest.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectionObservation {
    pub candidate_id: String,
    pub sample_id: String,
    pub sample_role: PartitionRole,
    pub sample_content_digest: String,
    pub metric_value: f64,
}

impl SelectionObservation {
    pub fn from_split(
        candidate_id: impl Into<String>,
        sample_id: &str,
        metric_value: f64,
        split: &ResearchSplitManifest,
        role_policy: SelectionRolePolicy,
    ) -> Result<Self> {
        split
            .verify_digest()
            .map_err(|error| SelectionError::Split(error.to_string()))?;
        let candidate_id = candidate_id.into();
        non_empty(&candidate_id, "candidate id")?;
        non_empty(sample_id, "selection sample id")?;
        if !metric_value.is_finite() {
            return Err(SelectionError::NonFiniteMetric {
                candidate_id,
                sample_id: sample_id.to_string(),
            });
        }
        let assignment = split
            .assignments
            .iter()
            .find(|assignment| assignment.unit.sample_id == sample_id)
            .ok_or_else(|| SelectionError::UnknownSample(sample_id.to_string()))?;
        if assignment.role == PartitionRole::Evaluation {
            return Err(SelectionError::EvaluationLeakage(sample_id.to_string()));
        }
        if !role_policy.allows(assignment.role) {
            return Err(SelectionError::TrainingLeakage(sample_id.to_string()));
        }
        Ok(Self {
            candidate_id,
            sample_id: assignment.unit.sample_id.clone(),
            sample_role: assignment.role,
            sample_content_digest: assignment.unit.content_digest.clone(),
            metric_value,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateAggregate {
    pub candidate_id: String,
    pub sample_count: usize,
    pub mean_metric: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResearchSelectionManifest {
    pub selection_id: String,
    pub split_manifest_digest: String,
    pub metric_id: String,
    pub direction: SelectionDirection,
    pub role_policy: SelectionRolePolicy,
    pub tie_break_policy: TieBreakPolicy,
    pub candidates: Vec<SelectionCandidate>,
    pub observations: Vec<SelectionObservation>,
    pub aggregates: Vec<CandidateAggregate>,
    pub selected_candidate_id: String,
    pub selected_at_unix_ms: i64,
    pub manifest_digest: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct ResearchSelectionManifestRepr {
    selection_id: String,
    split_manifest_digest: String,
    metric_id: String,
    direction: SelectionDirection,
    role_policy: SelectionRolePolicy,
    tie_break_policy: TieBreakPolicy,
    candidates: Vec<SelectionCandidate>,
    observations: Vec<SelectionObservation>,
    aggregates: Vec<CandidateAggregate>,
    selected_candidate_id: String,
    selected_at_unix_ms: i64,
    manifest_digest: String,
}

#[derive(Serialize)]
struct SelectionDigestView<'a> {
    schema: &'static str,
    selection_id: &'a str,
    split_manifest_digest: &'a str,
    metric_id: &'a str,
    direction: SelectionDirection,
    role_policy: SelectionRolePolicy,
    tie_break_policy: TieBreakPolicy,
    candidates: &'a [SelectionCandidate],
    observations: &'a [SelectionObservation],
    aggregates: &'a [CandidateAggregate],
    selected_candidate_id: &'a str,
    selected_at_unix_ms: i64,
}

impl ResearchSelectionManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        split: &ResearchSplitManifest,
        selection_id: impl Into<String>,
        metric_id: impl Into<String>,
        direction: SelectionDirection,
        role_policy: SelectionRolePolicy,
        candidates: Vec<SelectionCandidate>,
        observations: Vec<SelectionObservation>,
        selected_at_unix_ms: i64,
    ) -> Result<Self> {
        split
            .verify_digest()
            .map_err(|error| SelectionError::Split(error.to_string()))?;
        let selection_id = selection_id.into();
        let metric_id = metric_id.into();
        non_empty(&selection_id, "selection id")?;
        non_empty(&metric_id, "selection metric id")?;
        if candidates.len() < 2 {
            return Err(SelectionError::TooFewCandidates);
        }

        let mut candidate_ids = HashSet::new();
        for candidate in &candidates {
            non_empty(&candidate.candidate_id, "candidate id")?;
            non_empty(&candidate.fit_manifest_digest, "candidate fit manifest digest")?;
            non_empty(&candidate.output_artifact_digest, "candidate output artifact digest")?;
            if !candidate_ids.insert(candidate.candidate_id.clone()) {
                return Err(SelectionError::DuplicateCandidate(candidate.candidate_id.clone()));
            }
        }

        validate_observations(split, role_policy, &candidate_ids, &observations)?;
        let aggregates = compute_aggregates(&candidates, &observations)?;
        let selected_candidate_id = choose_winner(&aggregates, direction)?;

        let mut result = Self {
            selection_id,
            split_manifest_digest: split.manifest_digest.clone(),
            metric_id,
            direction,
            role_policy,
            tie_break_policy: TieBreakPolicy::LexicographicCandidateId,
            candidates,
            observations,
            aggregates,
            selected_candidate_id,
            selected_at_unix_ms,
            manifest_digest: String::new(),
        };
        result.candidates.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
        result.observations.sort_by(|a, b| {
            a.candidate_id
                .cmp(&b.candidate_id)
                .then_with(|| a.sample_id.cmp(&b.sample_id))
        });
        result.aggregates.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
        result.manifest_digest = result.compute_digest()?;
        Ok(result)
    }

    fn digest_view(&self) -> SelectionDigestView<'_> {
        SelectionDigestView {
            schema: SELECTION_SCHEMA,
            selection_id: &self.selection_id,
            split_manifest_digest: &self.split_manifest_digest,
            metric_id: &self.metric_id,
            direction: self.direction,
            role_policy: self.role_policy,
            tie_break_policy: self.tie_break_policy,
            candidates: &self.candidates,
            observations: &self.observations,
            aggregates: &self.aggregates,
            selected_candidate_id: &self.selected_candidate_id,
            selected_at_unix_ms: self.selected_at_unix_ms,
        }
    }

    pub fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&self.digest_view())
            .map_err(|error| SelectionError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn verify_digest(&self) -> Result<()> {
        validate_internal(self)?;
        if self.compute_digest()? != self.manifest_digest {
            return Err(SelectionError::ManifestDigestMismatch);
        }
        Ok(())
    }

    /// Revalidate a persisted selection against the authoritative split and candidate fit manifests.
    pub fn verify_against(
        &self,
        split: &ResearchSplitManifest,
        fits: &[FitArtifactManifest],
    ) -> Result<()> {
        self.verify_digest()?;
        split
            .verify_digest()
            .map_err(|error| SelectionError::Split(error.to_string()))?;
        if self.split_manifest_digest != split.manifest_digest {
            return Err(SelectionError::SplitManifestMismatch);
        }

        for candidate in &self.candidates {
            let fit = fits
                .iter()
                .find(|fit| fit.manifest_digest == candidate.fit_manifest_digest)
                .ok_or_else(|| SelectionError::MissingFitManifest(candidate.candidate_id.clone()))?;
            fit.verify_against_split(split)
                .map_err(|error| SelectionError::Fit(error.to_string()))?;
            if fit.manifest_digest != candidate.fit_manifest_digest {
                return Err(SelectionError::FitManifestMismatch(candidate.candidate_id.clone()));
            }
            if fit.output_artifact_digest != candidate.output_artifact_digest {
                return Err(SelectionError::FitArtifactMismatch(candidate.candidate_id.clone()));
            }
        }

        let ids: HashSet<_> = self
            .candidates
            .iter()
            .map(|candidate| candidate.candidate_id.clone())
            .collect();
        validate_observations(split, self.role_policy, &ids, &self.observations)?;
        Ok(())
    }
}

impl TryFrom<ResearchSelectionManifestRepr> for ResearchSelectionManifest {
    type Error = SelectionError;

    fn try_from(value: ResearchSelectionManifestRepr) -> Result<Self> {
        let result = Self {
            selection_id: value.selection_id,
            split_manifest_digest: value.split_manifest_digest,
            metric_id: value.metric_id,
            direction: value.direction,
            role_policy: value.role_policy,
            tie_break_policy: value.tie_break_policy,
            candidates: value.candidates,
            observations: value.observations,
            aggregates: value.aggregates,
            selected_candidate_id: value.selected_candidate_id,
            selected_at_unix_ms: value.selected_at_unix_ms,
            manifest_digest: value.manifest_digest,
        };
        result.verify_digest()?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for ResearchSelectionManifest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = ResearchSelectionManifestRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

fn validate_internal(manifest: &ResearchSelectionManifest) -> Result<()> {
    non_empty(&manifest.selection_id, "selection id")?;
    non_empty(&manifest.split_manifest_digest, "split manifest digest")?;
    non_empty(&manifest.metric_id, "selection metric id")?;
    non_empty(&manifest.selected_candidate_id, "selected candidate id")?;
    if manifest.candidates.len() < 2 {
        return Err(SelectionError::TooFewCandidates);
    }
    let mut candidate_ids = HashSet::new();
    for candidate in &manifest.candidates {
        non_empty(&candidate.candidate_id, "candidate id")?;
        non_empty(&candidate.fit_manifest_digest, "candidate fit manifest digest")?;
        non_empty(&candidate.output_artifact_digest, "candidate output artifact digest")?;
        if !candidate_ids.insert(candidate.candidate_id.clone()) {
            return Err(SelectionError::DuplicateCandidate(candidate.candidate_id.clone()));
        }
    }
    if !candidate_ids.contains(&manifest.selected_candidate_id) {
        return Err(SelectionError::WinnerMismatch);
    }

    let recomputed = compute_aggregates(&manifest.candidates, &manifest.observations)?;
    let recorded: BTreeMap<_, _> = manifest
        .aggregates
        .iter()
        .map(|aggregate| (aggregate.candidate_id.as_str(), aggregate))
        .collect();
    if recorded.len() != recomputed.len() {
        return Err(SelectionError::AggregateMismatch("aggregate-count".into()));
    }
    for aggregate in &recomputed {
        let Some(existing) = recorded.get(aggregate.candidate_id.as_str()) else {
            return Err(SelectionError::AggregateMismatch(aggregate.candidate_id.clone()));
        };
        if existing.sample_count != aggregate.sample_count
            || existing.mean_metric.to_bits() != aggregate.mean_metric.to_bits()
        {
            return Err(SelectionError::AggregateMismatch(aggregate.candidate_id.clone()));
        }
    }
    if choose_winner(&recomputed, manifest.direction)? != manifest.selected_candidate_id {
        return Err(SelectionError::WinnerMismatch);
    }
    Ok(())
}

fn validate_observations(
    split: &ResearchSplitManifest,
    role_policy: SelectionRolePolicy,
    candidate_ids: &HashSet<String>,
    observations: &[SelectionObservation],
) -> Result<()> {
    let mut seen = HashSet::new();
    for observation in observations {
        non_empty(&observation.candidate_id, "observation candidate id")?;
        non_empty(&observation.sample_id, "observation sample id")?;
        non_empty(&observation.sample_content_digest, "observation sample digest")?;
        if !candidate_ids.contains(&observation.candidate_id) {
            return Err(SelectionError::UnknownCandidate(observation.candidate_id.clone()));
        }
        if !observation.metric_value.is_finite() {
            return Err(SelectionError::NonFiniteMetric {
                candidate_id: observation.candidate_id.clone(),
                sample_id: observation.sample_id.clone(),
            });
        }
        let key = (observation.candidate_id.clone(), observation.sample_id.clone());
        if !seen.insert(key) {
            return Err(SelectionError::DuplicateObservation {
                candidate_id: observation.candidate_id.clone(),
                sample_id: observation.sample_id.clone(),
            });
        }
        let assignment = split
            .assignments
            .iter()
            .find(|assignment| assignment.unit.sample_id == observation.sample_id)
            .ok_or_else(|| SelectionError::UnknownSample(observation.sample_id.clone()))?;
        if assignment.role == PartitionRole::Evaluation {
            return Err(SelectionError::EvaluationLeakage(observation.sample_id.clone()));
        }
        if !role_policy.allows(assignment.role) {
            return Err(SelectionError::TrainingLeakage(observation.sample_id.clone()));
        }
        if assignment.role != observation.sample_role {
            return Err(SelectionError::SampleRoleMismatch {
                sample_id: observation.sample_id.clone(),
                recorded: observation.sample_role,
                actual: assignment.role,
            });
        }
        if assignment.unit.content_digest != observation.sample_content_digest {
            return Err(SelectionError::SampleDigestMismatch(observation.sample_id.clone()));
        }
    }
    Ok(())
}

fn compute_aggregates(
    candidates: &[SelectionCandidate],
    observations: &[SelectionObservation],
) -> Result<Vec<CandidateAggregate>> {
    let mut by_candidate: BTreeMap<&str, Vec<&SelectionObservation>> = BTreeMap::new();
    for observation in observations {
        by_candidate
            .entry(observation.candidate_id.as_str())
            .or_default()
            .push(observation);
    }

    let mut reference_sample_set: Option<BTreeSet<&str>> = None;
    let mut aggregates = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let values = by_candidate
            .get(candidate.candidate_id.as_str())
            .ok_or_else(|| SelectionError::MissingCandidateObservations(candidate.candidate_id.clone()))?;
        if values.is_empty() {
            return Err(SelectionError::MissingCandidateObservations(candidate.candidate_id.clone()));
        }
        let sample_set: BTreeSet<&str> = values.iter().map(|value| value.sample_id.as_str()).collect();
        match &reference_sample_set {
            None => reference_sample_set = Some(sample_set),
            Some(reference) if reference != &sample_set => {
                return Err(SelectionError::UnequalCandidateSampleSet {
                    candidate_id: candidate.candidate_id.clone(),
                });
            }
            Some(_) => {}
        }

        let mut mean = 0.0_f64;
        for (index, value) in values.iter().enumerate() {
            let n = (index + 1) as f64;
            mean += (value.metric_value - mean) / n;
        }
        if !mean.is_finite() {
            return Err(SelectionError::NonFiniteMetric {
                candidate_id: candidate.candidate_id.clone(),
                sample_id: "aggregate".into(),
            });
        }
        aggregates.push(CandidateAggregate {
            candidate_id: candidate.candidate_id.clone(),
            sample_count: values.len(),
            mean_metric: mean,
        });
    }
    aggregates.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    Ok(aggregates)
}

fn choose_winner(
    aggregates: &[CandidateAggregate],
    direction: SelectionDirection,
) -> Result<String> {
    let mut ordered: Vec<&CandidateAggregate> = aggregates.iter().collect();
    ordered.sort_by(|a, b| {
        let metric_order = a
            .mean_metric
            .partial_cmp(&b.mean_metric)
            .unwrap_or(Ordering::Equal);
        let metric_order = match direction {
            SelectionDirection::Minimize => metric_order,
            SelectionDirection::Maximize => metric_order.reverse(),
        };
        metric_order.then_with(|| a.candidate_id.cmp(&b.candidate_id))
    });
    ordered
        .first()
        .map(|value| value.candidate_id.clone())
        .ok_or(SelectionError::TooFewCandidates)
}
