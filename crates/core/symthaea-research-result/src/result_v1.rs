//! Immutable research-result manifests bound to frozen preregistration and run lineage.
//!
//! A result is not just a number or a prose claim. This crate binds reported outcomes to the
//! exact frozen protocol, registered source/data/environment/seed lineage, amendments,
//! deviations, analysis artifacts, preregistered metrics, and hypothesis references that gave
//! the result meaning.
//!
//! Load-bearing rules:
//! - every preregistered primary metric must be represented, including missing/not-computed ones;
//! - unknown metric/hypothesis/artifact ids fail closed;
//! - post-unblinding amendments and primary-analysis deviations downgrade interpretation via
//!   `symthaea-research-protocol`, rather than being hidden by this layer;
//! - null and inconclusive findings are first-class dispositions;
//! - the manifest has a versioned content digest and contains no universal evidence/trust score.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use symthaea_research_protocol::{
    classify_result, FrozenProtocol, HypothesisRole, MetricRole, ProtocolAmendment,
    ProtocolDeviation, ResearchRunRegistration, ResultInterpretation,
};

const MANIFEST_SCHEMA: &str = "symthaea-research-result/v1";

pub type Result<T> = std::result::Result<T, ResultManifestError>;

#[derive(Debug, Clone, PartialEq)]
pub enum ResultManifestError {
    EmptyField(&'static str),
    Protocol(String),
    ProtocolDigestMismatch,
    ResultBeforeRunRegistration,
    DuplicateId(String),
    UnknownMetric(String),
    UnknownHypothesis(String),
    UnknownArtifact(String),
    MissingPrimaryMetric(String),
    UnitMismatch {
        metric_id: String,
        expected: String,
        got: String,
    },
    NonFiniteMetric {
        metric_id: String,
        value: f64,
    },
    EmptyMetricReason(String),
    MissingAnalysisArtifact,
    ClaimWithoutEvidence(String),
    ClaimMetricNotReported {
        claim_id: String,
        metric_id: String,
    },
    ExploratoryHypothesisClaimMarkedConfirmatory(String),
    ManifestDigestMismatch,
    Serialization(String),
}

impl Display for ResultManifestError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::Protocol(message) => write!(f, "research protocol validation failed: {message}"),
            Self::ProtocolDigestMismatch => {
                write!(f, "run/amendment lineage does not match the frozen protocol")
            }
            Self::ResultBeforeRunRegistration => {
                write!(f, "result completion cannot precede run registration")
            }
            Self::DuplicateId(id) => write!(f, "duplicate result-manifest id: {id}"),
            Self::UnknownMetric(id) => write!(f, "unknown preregistered metric id: {id}"),
            Self::UnknownHypothesis(id) => write!(f, "unknown preregistered hypothesis id: {id}"),
            Self::UnknownArtifact(id) => write!(f, "unknown result artifact id: {id}"),
            Self::MissingPrimaryMetric(id) => {
                write!(f, "preregistered primary metric {id} is absent from the result")
            }
            Self::UnitMismatch {
                metric_id,
                expected,
                got,
            } => write!(
                f,
                "metric {metric_id} unit mismatch: expected {expected}, got {got}"
            ),
            Self::NonFiniteMetric { metric_id, value } => {
                write!(f, "metric {metric_id} must be finite, got {value}")
            }
            Self::EmptyMetricReason(id) => {
                write!(f, "metric {id} missing/not-computed state requires a reason")
            }
            Self::MissingAnalysisArtifact => {
                write!(f, "result manifest requires at least one analysis artifact")
            }
            Self::ClaimWithoutEvidence(id) => write!(
                f,
                "claim {id} must reference at least one reported metric or result artifact"
            ),
            Self::ClaimMetricNotReported {
                claim_id,
                metric_id,
            } => write!(
                f,
                "claim {claim_id} references metric {metric_id}, but that metric has no result entry"
            ),
            Self::ExploratoryHypothesisClaimMarkedConfirmatory(id) => write!(
                f,
                "claim {id} targets a preregistered exploratory hypothesis and cannot be confirmatory"
            ),
            Self::ManifestDigestMismatch => write!(f, "research result manifest digest mismatch"),
            Self::Serialization(message) => {
                write!(f, "research result manifest serialization failed: {message}")
            }
        }
    }
}

impl Error for ResultManifestError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ResultManifestError::EmptyField(field));
    }
    Ok(())
}

fn unique_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(ResultManifestError::DuplicateId(id.to_string()));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResultArtifactKind {
    Analysis,
    Metrics,
    RawOutput,
    Figure,
    Table,
    Model,
    ForecastLedger,
    Verification,
    Log,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultArtifactRef {
    pub id: String,
    pub kind: ResultArtifactKind,
    pub digest: String,
    pub media_type: Option<String>,
    pub description: String,
}

impl ResultArtifactRef {
    pub fn new(
        id: impl Into<String>,
        kind: ResultArtifactKind,
        digest: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<Self> {
        let id = id.into();
        let digest = digest.into();
        let description = description.into();
        non_empty(&id, "result artifact id")?;
        non_empty(&digest, "result artifact digest")?;
        non_empty(&description, "result artifact description")?;
        Ok(Self {
            id,
            kind,
            digest,
            media_type: None,
            description,
        })
    }

    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Result<Self> {
        let media_type = media_type.into();
        non_empty(&media_type, "result artifact media type")?;
        self.media_type = Some(media_type);
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricOutcome {
    Numeric { value: f64, unit: String },
    Boolean(bool),
    Categorical(String),
    Missing { reason: String },
    NotComputed { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricResult {
    pub metric_id: String,
    pub outcome: MetricOutcome,
    pub artifact_ids: Vec<String>,
    pub notes: Option<String>,
}

impl MetricResult {
    pub fn new(metric_id: impl Into<String>, outcome: MetricOutcome) -> Result<Self> {
        let metric_id = metric_id.into();
        non_empty(&metric_id, "metric result id")?;
        match &outcome {
            MetricOutcome::Numeric { value, unit } => {
                if !value.is_finite() {
                    return Err(ResultManifestError::NonFiniteMetric {
                        metric_id: metric_id.clone(),
                        value: *value,
                    });
                }
                non_empty(unit, "metric result unit")?;
            }
            MetricOutcome::Categorical(value) => {
                non_empty(value, "categorical metric result")?;
            }
            MetricOutcome::Missing { reason } | MetricOutcome::NotComputed { reason } => {
                if reason.trim().is_empty() {
                    return Err(ResultManifestError::EmptyMetricReason(metric_id.clone()));
                }
            }
            MetricOutcome::Boolean(_) => {}
        }
        Ok(Self {
            metric_id,
            outcome,
            artifact_ids: Vec::new(),
            notes: None,
        })
    }

    pub fn with_artifact(mut self, artifact_id: impl Into<String>) -> Result<Self> {
        let artifact_id = artifact_id.into();
        non_empty(&artifact_id, "metric result artifact id")?;
        self.artifact_ids.push(artifact_id);
        Ok(self)
    }

    pub fn with_notes(mut self, notes: impl Into<String>) -> Result<Self> {
        let notes = notes.into();
        non_empty(&notes, "metric result notes")?;
        self.notes = Some(notes);
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimDisposition {
    ConsistentWithHypothesis,
    InconsistentWithHypothesis,
    NullResult,
    Inconclusive,
    DescriptiveOnly,
    NotEvaluated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimInterpretation {
    Confirmatory,
    Exploratory,
    Invalidated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultClaim {
    pub claim_id: String,
    pub statement: String,
    pub hypothesis_id: Option<String>,
    pub disposition: ClaimDisposition,
    pub metric_ids: Vec<String>,
    pub artifact_ids: Vec<String>,
    pub interpretation: ClaimInterpretation,
}

impl ResultClaim {
    pub fn new(
        claim_id: impl Into<String>,
        statement: impl Into<String>,
        disposition: ClaimDisposition,
        interpretation: ClaimInterpretation,
    ) -> Result<Self> {
        let claim_id = claim_id.into();
        let statement = statement.into();
        non_empty(&claim_id, "result claim id")?;
        non_empty(&statement, "result claim statement")?;
        Ok(Self {
            claim_id,
            statement,
            hypothesis_id: None,
            disposition,
            metric_ids: Vec::new(),
            artifact_ids: Vec::new(),
            interpretation,
        })
    }

    pub fn for_hypothesis(mut self, hypothesis_id: impl Into<String>) -> Result<Self> {
        let hypothesis_id = hypothesis_id.into();
        non_empty(&hypothesis_id, "result claim hypothesis id")?;
        self.hypothesis_id = Some(hypothesis_id);
        Ok(self)
    }

    pub fn with_metric(mut self, metric_id: impl Into<String>) -> Result<Self> {
        let metric_id = metric_id.into();
        non_empty(&metric_id, "result claim metric id")?;
        self.metric_ids.push(metric_id);
        Ok(self)
    }

    pub fn with_artifact(mut self, artifact_id: impl Into<String>) -> Result<Self> {
        let artifact_id = artifact_id.into();
        non_empty(&artifact_id, "result claim artifact id")?;
        self.artifact_ids.push(artifact_id);
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchResultManifest {
    pub manifest_id: String,
    pub protocol_digest: String,
    pub run: ResearchRunRegistration,
    pub completed_at_unix_ms: i64,
    pub amendments: Vec<ProtocolAmendment>,
    pub deviations: Vec<ProtocolDeviation>,
    pub interpretation: ResultInterpretation,
    pub artifacts: Vec<ResultArtifactRef>,
    pub metrics: Vec<MetricResult>,
    pub claims: Vec<ResultClaim>,
    pub manifest_digest: String,
}

#[derive(Serialize)]
struct ManifestDigestView<'a> {
    schema: &'static str,
    manifest_id: &'a str,
    protocol_digest: &'a str,
    run: &'a ResearchRunRegistration,
    completed_at_unix_ms: i64,
    amendments: &'a [ProtocolAmendment],
    deviations: &'a [ProtocolDeviation],
    interpretation: ResultInterpretation,
    artifacts: &'a [ResultArtifactRef],
    metrics: &'a [MetricResult],
    claims: &'a [ResultClaim],
}

impl ResearchResultManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frozen: &FrozenProtocol,
        run: ResearchRunRegistration,
        manifest_id: impl Into<String>,
        completed_at_unix_ms: i64,
        amendments: Vec<ProtocolAmendment>,
        deviations: Vec<ProtocolDeviation>,
        invalidated: bool,
        artifacts: Vec<ResultArtifactRef>,
        metrics: Vec<MetricResult>,
        claims: Vec<ResultClaim>,
    ) -> Result<Self> {
        frozen
            .verify_digest()
            .map_err(|error| ResultManifestError::Protocol(error.to_string()))?;
        let manifest_id = manifest_id.into();
        non_empty(&manifest_id, "result manifest id")?;

        if run.protocol_digest != frozen.digest() {
            return Err(ResultManifestError::ProtocolDigestMismatch);
        }
        if completed_at_unix_ms < run.registered_at_unix_ms {
            return Err(ResultManifestError::ResultBeforeRunRegistration);
        }
        if amendments
            .iter()
            .any(|amendment| amendment.parent_protocol_digest != frozen.digest())
        {
            return Err(ResultManifestError::ProtocolDigestMismatch);
        }

        unique_ids(artifacts.iter().map(|value| value.id.as_str()))?;
        unique_ids(metrics.iter().map(|value| value.metric_id.as_str()))?;
        unique_ids(claims.iter().map(|value| value.claim_id.as_str()))?;
        unique_ids(amendments.iter().map(|value| value.amendment_id.as_str()))?;
        unique_ids(deviations.iter().map(|value| value.deviation_id.as_str()))?;

        if !artifacts
            .iter()
            .any(|artifact| artifact.kind == ResultArtifactKind::Analysis)
        {
            return Err(ResultManifestError::MissingAnalysisArtifact);
        }

        let protocol = frozen.protocol();
        let metric_specs: HashMap<&str, _> = protocol
            .metrics
            .iter()
            .map(|metric| (metric.id.as_str(), metric))
            .collect();
        let hypothesis_specs: HashMap<&str, _> = protocol
            .hypotheses
            .iter()
            .map(|hypothesis| (hypothesis.id.as_str(), hypothesis))
            .collect();
        let artifact_ids: HashSet<&str> = artifacts.iter().map(|artifact| artifact.id.as_str()).collect();
        let reported_metric_ids: HashSet<&str> =
            metrics.iter().map(|metric| metric.metric_id.as_str()).collect();

        for metric in &metrics {
            let Some(spec) = metric_specs.get(metric.metric_id.as_str()) else {
                return Err(ResultManifestError::UnknownMetric(metric.metric_id.clone()));
            };
            if let MetricOutcome::Numeric { value, unit } = &metric.outcome {
                if !value.is_finite() {
                    return Err(ResultManifestError::NonFiniteMetric {
                        metric_id: metric.metric_id.clone(),
                        value: *value,
                    });
                }
                if unit != &spec.unit {
                    return Err(ResultManifestError::UnitMismatch {
                        metric_id: metric.metric_id.clone(),
                        expected: spec.unit.clone(),
                        got: unit.clone(),
                    });
                }
            }
            for artifact_id in &metric.artifact_ids {
                if !artifact_ids.contains(artifact_id.as_str()) {
                    return Err(ResultManifestError::UnknownArtifact(artifact_id.clone()));
                }
            }
        }

        for spec in protocol
            .metrics
            .iter()
            .filter(|metric| metric.role == MetricRole::Primary)
        {
            if !reported_metric_ids.contains(spec.id.as_str()) {
                return Err(ResultManifestError::MissingPrimaryMetric(spec.id.clone()));
            }
        }

        let overall_interpretation = classify_result(&amendments, &deviations, invalidated);

        for claim in &claims {
            if claim.metric_ids.is_empty() && claim.artifact_ids.is_empty() {
                return Err(ResultManifestError::ClaimWithoutEvidence(
                    claim.claim_id.clone(),
                ));
            }
            for metric_id in &claim.metric_ids {
                if !metric_specs.contains_key(metric_id.as_str()) {
                    return Err(ResultManifestError::UnknownMetric(metric_id.clone()));
                }
                if !reported_metric_ids.contains(metric_id.as_str()) {
                    return Err(ResultManifestError::ClaimMetricNotReported {
                        claim_id: claim.claim_id.clone(),
                        metric_id: metric_id.clone(),
                    });
                }
            }
            for artifact_id in &claim.artifact_ids {
                if !artifact_ids.contains(artifact_id.as_str()) {
                    return Err(ResultManifestError::UnknownArtifact(artifact_id.clone()));
                }
            }
            if let Some(hypothesis_id) = &claim.hypothesis_id {
                let Some(hypothesis) = hypothesis_specs.get(hypothesis_id.as_str()) else {
                    return Err(ResultManifestError::UnknownHypothesis(hypothesis_id.clone()));
                };
                if hypothesis.role == HypothesisRole::Exploratory
                    && claim.interpretation == ClaimInterpretation::Confirmatory
                {
                    return Err(
                        ResultManifestError::ExploratoryHypothesisClaimMarkedConfirmatory(
                            claim.claim_id.clone(),
                        ),
                    );
                }
            }

            match overall_interpretation {
                ResultInterpretation::Confirmatory => {}
                ResultInterpretation::ExploratoryDueToPostUnblindingAmendment
                | ResultInterpretation::ExploratoryDueToPrimaryDeviation => {
                    if claim.interpretation == ClaimInterpretation::Confirmatory {
                        return Err(ResultManifestError::Protocol(
                            "overall result is exploratory but a claim is marked confirmatory"
                                .into(),
                        ));
                    }
                }
                ResultInterpretation::Invalidated => {
                    if claim.interpretation != ClaimInterpretation::Invalidated {
                        return Err(ResultManifestError::Protocol(
                            "invalidated result requires invalidated claim interpretations"
                                .into(),
                        ));
                    }
                }
            }
        }

        let mut manifest = Self {
            manifest_id,
            protocol_digest: frozen.digest().to_string(),
            run,
            completed_at_unix_ms,
            amendments,
            deviations,
            interpretation: overall_interpretation,
            artifacts,
            metrics,
            claims,
            manifest_digest: String::new(),
        };
        manifest.manifest_digest = manifest.compute_digest()?;
        Ok(manifest)
    }

    fn digest_view(&self) -> ManifestDigestView<'_> {
        ManifestDigestView {
            schema: MANIFEST_SCHEMA,
            manifest_id: &self.manifest_id,
            protocol_digest: &self.protocol_digest,
            run: &self.run,
            completed_at_unix_ms: self.completed_at_unix_ms,
            amendments: &self.amendments,
            deviations: &self.deviations,
            interpretation: self.interpretation,
            artifacts: &self.artifacts,
            metrics: &self.metrics,
            claims: &self.claims,
        }
    }

    pub fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&self.digest_view())
            .map_err(|error| ResultManifestError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn verify_digest(&self) -> Result<()> {
        if self.compute_digest()? != self.manifest_digest {
            return Err(ResultManifestError::ManifestDigestMismatch);
        }
        Ok(())
    }

    pub fn primary_metric_results<'a>(
        &'a self,
        frozen: &'a FrozenProtocol,
    ) -> Result<Vec<&'a MetricResult>> {
        frozen
            .verify_digest()
            .map_err(|error| ResultManifestError::Protocol(error.to_string()))?;
        if frozen.digest() != self.protocol_digest {
            return Err(ResultManifestError::ProtocolDigestMismatch);
        }
        let primary: HashSet<&str> = frozen
            .protocol()
            .metrics
            .iter()
            .filter(|metric| metric.role == MetricRole::Primary)
            .map(|metric| metric.id.as_str())
            .collect();
        Ok(self
            .metrics
            .iter()
            .filter(|metric| primary.contains(metric.metric_id.as_str()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_research_protocol::{
        AnalysisPlanRef, BaselineSpec, HypothesisDirection, HypothesisSpec, MetricSpec,
        MultiplicityPolicy, ResearchProtocol, StoppingRule,
    };

    fn frozen() -> FrozenProtocol {
        ResearchProtocol::new(
            "wetland-watch-v1",
            "1",
            "Does semantic prioritization preserve more mission-relevant information?",
            vec![
                HypothesisSpec::new(
                    "h-primary",
                    "semantic prioritization beats the simple ROI baseline",
                    HypothesisRole::Primary,
                    HypothesisDirection::GreaterThan,
                )
                .unwrap(),
                HypothesisSpec::new(
                    "h-explore",
                    "regional HDC improves retrieval",
                    HypothesisRole::Exploratory,
                    HypothesisDirection::GreaterThan,
                )
                .unwrap(),
            ],
            vec![
                MetricSpec::new(
                    "mission-bits",
                    "mission-relevant information per byte",
                    "utility/byte",
                    MetricRole::Primary,
                    "held-out mean",
                )
                .unwrap(),
                MetricSpec::new(
                    "energy",
                    "encoding energy",
                    "joule",
                    MetricRole::Secondary,
                    "held-out mean",
                )
                .unwrap(),
            ],
            vec![BaselineSpec::new(
                "simple-roi",
                "conventional codec plus simple cloud/change ROI",
                "bench/simple_roi_v1",
            )
            .unwrap()],
            vec![],
            StoppingRule::FixedSampleCount(100),
            MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
            AnalysisPlanRef::new("analysis", "1", "sha256:analysis-plan").unwrap(),
            "100 frozen paired Sentinel scenes",
            "fixed seed manifest",
        )
        .unwrap()
        .freeze(1_000)
        .unwrap()
    }

    fn run(frozen: &FrozenProtocol) -> ResearchRunRegistration {
        ResearchRunRegistration::new(
            frozen,
            "run-001",
            1_100,
            "deadbeef",
            "sha256:dataset",
            "sha256:repro",
            "sha256:seeds",
        )
        .unwrap()
    }

    fn analysis_artifact() -> ResultArtifactRef {
        ResultArtifactRef::new(
            "analysis",
            ResultArtifactKind::Analysis,
            "sha256:analysis-result",
            "preregistered analysis output",
        )
        .unwrap()
    }

    #[test]
    fn primary_metric_cannot_disappear() {
        let frozen = frozen();
        let err = ResearchResultManifest::new(
            &frozen,
            run(&frozen),
            "result-001",
            2_000,
            vec![],
            vec![],
            false,
            vec![analysis_artifact()],
            vec![],
            vec![],
        )
        .unwrap_err();
        assert_eq!(
            err,
            ResultManifestError::MissingPrimaryMetric("mission-bits".into())
        );
    }

    #[test]
    fn explicit_missing_primary_metric_is_retained() {
        let frozen = frozen();
        let result = ResearchResultManifest::new(
            &frozen,
            run(&frozen),
            "result-001",
            2_000,
            vec![],
            vec![],
            false,
            vec![analysis_artifact()],
            vec![MetricResult::new(
                "mission-bits",
                MetricOutcome::Missing {
                    reason: "sensor fixture checksum failed before analysis".into(),
                },
            )
            .unwrap()],
            vec![],
        )
        .unwrap();
        assert_eq!(result.interpretation, ResultInterpretation::Confirmatory);
        result.verify_digest().unwrap();
    }

    #[test]
    fn null_result_is_a_first_class_claim() {
        let frozen = frozen();
        let result = ResearchResultManifest::new(
            &frozen,
            run(&frozen),
            "result-null",
            2_000,
            vec![],
            vec![],
            false,
            vec![analysis_artifact()],
            vec![MetricResult::new(
                "mission-bits",
                MetricOutcome::Numeric {
                    value: 0.0,
                    unit: "utility/byte".into(),
                },
            )
            .unwrap()],
            vec![ResultClaim::new(
                "claim-null",
                "No advantage was observed on the preregistered primary metric.",
                ClaimDisposition::NullResult,
                ClaimInterpretation::Confirmatory,
            )
            .unwrap()
            .for_hypothesis("h-primary")
            .unwrap()
            .with_metric("mission-bits")
            .unwrap()],
        )
        .unwrap();
        assert_eq!(result.claims[0].disposition, ClaimDisposition::NullResult);
    }

    #[test]
    fn exploratory_hypothesis_cannot_be_labeled_confirmatory() {
        let frozen = frozen();
        let err = ResearchResultManifest::new(
            &frozen,
            run(&frozen),
            "result-explore",
            2_000,
            vec![],
            vec![],
            false,
            vec![analysis_artifact()],
            vec![MetricResult::new(
                "mission-bits",
                MetricOutcome::Numeric {
                    value: 1.2,
                    unit: "utility/byte".into(),
                },
            )
            .unwrap()],
            vec![ResultClaim::new(
                "claim-explore",
                "HDC retrieval improved.",
                ClaimDisposition::ConsistentWithHypothesis,
                ClaimInterpretation::Confirmatory,
            )
            .unwrap()
            .for_hypothesis("h-explore")
            .unwrap()
            .with_artifact("analysis")
            .unwrap()],
        )
        .unwrap_err();
        assert_eq!(
            err,
            ResultManifestError::ExploratoryHypothesisClaimMarkedConfirmatory(
                "claim-explore".into()
            )
        );
    }

    #[test]
    fn post_unblinding_amendment_prevents_confirmatory_claim() {
        let frozen = frozen();
        let amendment = symthaea_research_protocol::ProtocolAmendment::new(
            &frozen,
            "a1",
            1_500,
            symthaea_research_protocol::AmendmentTiming::AfterOutcomeUnblinding,
            "added a favorable subgroup after seeing outcomes",
            vec!["add subgroup analysis".into()],
        )
        .unwrap();
        let err = ResearchResultManifest::new(
            &frozen,
            run(&frozen),
            "result-amended",
            2_000,
            vec![amendment],
            vec![],
            false,
            vec![analysis_artifact()],
            vec![MetricResult::new(
                "mission-bits",
                MetricOutcome::Numeric {
                    value: 1.2,
                    unit: "utility/byte".into(),
                },
            )
            .unwrap()],
            vec![ResultClaim::new(
                "claim-primary",
                "Semantic prioritization improved the primary metric.",
                ClaimDisposition::ConsistentWithHypothesis,
                ClaimInterpretation::Confirmatory,
            )
            .unwrap()
            .for_hypothesis("h-primary")
            .unwrap()
            .with_metric("mission-bits")
            .unwrap()],
        )
        .unwrap_err();
        assert!(matches!(err, ResultManifestError::Protocol(_)));
    }

    #[test]
    fn artifact_and_metric_references_are_checked() {
        let frozen = frozen();
        let err = ResearchResultManifest::new(
            &frozen,
            run(&frozen),
            "result-bad-ref",
            2_000,
            vec![],
            vec![],
            false,
            vec![analysis_artifact()],
            vec![MetricResult::new(
                "mission-bits",
                MetricOutcome::Numeric {
                    value: 1.0,
                    unit: "utility/byte".into(),
                },
            )
            .unwrap()
            .with_artifact("missing-artifact")
            .unwrap()],
            vec![],
        )
        .unwrap_err();
        assert_eq!(
            err,
            ResultManifestError::UnknownArtifact("missing-artifact".into())
        );
    }

    #[test]
    fn digest_detects_manifest_mutation() {
        let frozen = frozen();
        let mut result = ResearchResultManifest::new(
            &frozen,
            run(&frozen),
            "result-digest",
            2_000,
            vec![],
            vec![],
            false,
            vec![analysis_artifact()],
            vec![MetricResult::new(
                "mission-bits",
                MetricOutcome::Numeric {
                    value: 1.0,
                    unit: "utility/byte".into(),
                },
            )
            .unwrap()],
            vec![],
        )
        .unwrap();
        result.verify_digest().unwrap();
        result.metrics[0].outcome = MetricOutcome::Numeric {
            value: 99.0,
            unit: "utility/byte".into(),
        };
        assert_eq!(
            result.verify_digest().unwrap_err(),
            ResultManifestError::ManifestDigestMismatch
        );
    }
}
