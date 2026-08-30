//! Replication and lineage-comparison evidence for Symthaea research.
//!
//! Reproducibility, direct replication, conceptual replication, and reanalysis answer different
//! questions. This crate makes those distinctions explicit and records factual lineage equality
//! from result manifests instead of exposing a vague `independent = true` flag.

use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use symthaea_research_result::ResearchResultManifest;

const ASSESSMENT_SCHEMA: &str = "symthaea-research-replication/v1";

pub type Result<T> = std::result::Result<T, ReplicationError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationError {
    EmptyField(&'static str),
    OriginalManifestInvalid(String),
    FollowupManifestInvalid(String),
    SameResultManifest,
    ExactReproductionProtocolChanged,
    ExactReproductionSourceChanged,
    ExactReproductionDatasetChanged,
    ExactReproductionEnvironmentChanged,
    ExactReproductionSeedsChanged,
    DirectReplicationProtocolChanged,
    DirectReplicationReusedDataset,
    ReanalysisDatasetChanged,
    AssessmentDigestMismatch,
    Serialization(String),
}

impl Display for ReplicationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::OriginalManifestInvalid(message) => {
                write!(f, "original result manifest is invalid: {message}")
            }
            Self::FollowupManifestInvalid(message) => {
                write!(f, "follow-up result manifest is invalid: {message}")
            }
            Self::SameResultManifest => write!(f, "replication assessment requires two distinct result manifests"),
            Self::ExactReproductionProtocolChanged => write!(f, "exact reproduction requires the same frozen protocol digest"),
            Self::ExactReproductionSourceChanged => write!(f, "exact reproduction requires the same source commit"),
            Self::ExactReproductionDatasetChanged => write!(f, "exact reproduction requires the same dataset manifest digest"),
            Self::ExactReproductionEnvironmentChanged => write!(f, "exact reproduction requires the same reproducibility capsule digest"),
            Self::ExactReproductionSeedsChanged => write!(f, "exact reproduction requires the same seed manifest digest"),
            Self::DirectReplicationProtocolChanged => write!(f, "direct replication requires the same frozen protocol digest"),
            Self::DirectReplicationReusedDataset => write!(f, "direct replication requires a different dataset lineage; same-data reruns are reproduction/reanalysis evidence"),
            Self::ReanalysisDatasetChanged => write!(f, "reanalysis requires the same dataset manifest digest"),
            Self::AssessmentDigestMismatch => write!(f, "replication assessment digest mismatch"),
            Self::Serialization(message) => write!(f, "replication assessment serialization failed: {message}"),
        }
    }
}

impl Error for ReplicationError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ReplicationError::EmptyField(field));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationDesign {
    /// Same protocol, source, data, environment capsule, and seeds. Tests deterministic/replay
    /// reproducibility of an exact lineage; it is not an independent replication.
    ExactReproduction,
    /// Same frozen protocol/question with a distinct dataset lineage. Implementation may be the
    /// same or different; those dimensions remain visible separately.
    DirectReplication,
    /// Related question tested under a deliberately different protocol/design/population/model.
    ConceptualReplication,
    /// Same dataset, analyzed again (possibly with a different implementation/analysis path).
    Reanalysis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineageRelation {
    Same,
    Different,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactualLineageComparison {
    pub protocol: LineageRelation,
    pub source_commit: LineageRelation,
    pub dataset_manifest: LineageRelation,
    pub reproducibility_capsule: LineageRelation,
    pub seed_manifest: LineageRelation,
}

impl FactualLineageComparison {
    pub fn between(original: &ResearchResultManifest, followup: &ResearchResultManifest) -> Self {
        fn relation(left: &str, right: &str) -> LineageRelation {
            if left == right {
                LineageRelation::Same
            } else {
                LineageRelation::Different
            }
        }
        Self {
            protocol: relation(&original.protocol_digest, &followup.protocol_digest),
            source_commit: relation(&original.run.source_commit, &followup.run.source_commit),
            dataset_manifest: relation(
                &original.run.dataset_manifest_digest,
                &followup.run.dataset_manifest_digest,
            ),
            reproducibility_capsule: relation(
                &original.run.reproducibility_capsule_digest,
                &followup.run.reproducibility_capsule_digest,
            ),
            seed_manifest: relation(
                &original.run.seed_manifest_digest,
                &followup.run.seed_manifest_digest,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndependenceDimension {
    DataAcquisition,
    Implementation,
    Analyst,
    Institution,
    Hardware,
    MeasurementSystem,
    ValidationTeam,
    Other,
}

/// Evidence-backed claim about an independence dimension that cannot be established merely by
/// comparing repository/data digests. A statement and evidence digest are retained, but this
/// type does not prove the claim is true.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceEvidence {
    pub dimension: IndependenceDimension,
    pub statement: String,
    pub evidence_digest: String,
}

impl IndependenceEvidence {
    pub fn new(
        dimension: IndependenceDimension,
        statement: impl Into<String>,
        evidence_digest: impl Into<String>,
    ) -> Result<Self> {
        let statement = statement.into();
        let evidence_digest = evidence_digest.into();
        non_empty(&statement, "independence evidence statement")?;
        non_empty(&evidence_digest, "independence evidence digest")?;
        Ok(Self {
            dimension,
            statement,
            evidence_digest,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationOutcome {
    Concordant,
    Discordant,
    Mixed,
    Inconclusive,
    NotComparable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationComparisonEvidence {
    pub method: String,
    pub artifact_digest: String,
    pub notes: String,
}

impl ReplicationComparisonEvidence {
    pub fn new(
        method: impl Into<String>,
        artifact_digest: impl Into<String>,
        notes: impl Into<String>,
    ) -> Result<Self> {
        let method = method.into();
        let artifact_digest = artifact_digest.into();
        let notes = notes.into();
        non_empty(&method, "replication comparison method")?;
        non_empty(&artifact_digest, "replication comparison artifact digest")?;
        non_empty(&notes, "replication comparison notes")?;
        Ok(Self {
            method,
            artifact_digest,
            notes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationAssessment {
    pub assessment_id: String,
    pub design: ReplicationDesign,
    pub original_result_digest: String,
    pub followup_result_digest: String,
    pub factual_lineage: FactualLineageComparison,
    pub independence_evidence: Vec<IndependenceEvidence>,
    pub outcome: ReplicationOutcome,
    pub comparison: ReplicationComparisonEvidence,
    pub assessment_digest: String,
}

#[derive(Serialize)]
struct AssessmentDigestView<'a> {
    schema: &'static str,
    assessment_id: &'a str,
    design: ReplicationDesign,
    original_result_digest: &'a str,
    followup_result_digest: &'a str,
    factual_lineage: &'a FactualLineageComparison,
    independence_evidence: &'a [IndependenceEvidence],
    outcome: ReplicationOutcome,
    comparison: &'a ReplicationComparisonEvidence,
}

impl ReplicationAssessment {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        assessment_id: impl Into<String>,
        design: ReplicationDesign,
        original: &ResearchResultManifest,
        followup: &ResearchResultManifest,
        independence_evidence: Vec<IndependenceEvidence>,
        outcome: ReplicationOutcome,
        comparison: ReplicationComparisonEvidence,
    ) -> Result<Self> {
        original
            .verify_digest()
            .map_err(|error| ReplicationError::OriginalManifestInvalid(error.to_string()))?;
        followup
            .verify_digest()
            .map_err(|error| ReplicationError::FollowupManifestInvalid(error.to_string()))?;

        if original.manifest_digest == followup.manifest_digest {
            return Err(ReplicationError::SameResultManifest);
        }

        let assessment_id = assessment_id.into();
        non_empty(&assessment_id, "replication assessment id")?;
        let factual_lineage = FactualLineageComparison::between(original, followup);

        match design {
            ReplicationDesign::ExactReproduction => {
                if factual_lineage.protocol != LineageRelation::Same {
                    return Err(ReplicationError::ExactReproductionProtocolChanged);
                }
                if factual_lineage.source_commit != LineageRelation::Same {
                    return Err(ReplicationError::ExactReproductionSourceChanged);
                }
                if factual_lineage.dataset_manifest != LineageRelation::Same {
                    return Err(ReplicationError::ExactReproductionDatasetChanged);
                }
                if factual_lineage.reproducibility_capsule != LineageRelation::Same {
                    return Err(ReplicationError::ExactReproductionEnvironmentChanged);
                }
                if factual_lineage.seed_manifest != LineageRelation::Same {
                    return Err(ReplicationError::ExactReproductionSeedsChanged);
                }
            }
            ReplicationDesign::DirectReplication => {
                if factual_lineage.protocol != LineageRelation::Same {
                    return Err(ReplicationError::DirectReplicationProtocolChanged);
                }
                if factual_lineage.dataset_manifest != LineageRelation::Different {
                    return Err(ReplicationError::DirectReplicationReusedDataset);
                }
            }
            ReplicationDesign::Reanalysis => {
                if factual_lineage.dataset_manifest != LineageRelation::Same {
                    return Err(ReplicationError::ReanalysisDatasetChanged);
                }
            }
            ReplicationDesign::ConceptualReplication => {}
        }

        let mut assessment = Self {
            assessment_id,
            design,
            original_result_digest: original.manifest_digest.clone(),
            followup_result_digest: followup.manifest_digest.clone(),
            factual_lineage,
            independence_evidence,
            outcome,
            comparison,
            assessment_digest: String::new(),
        };
        assessment.assessment_digest = assessment.compute_digest()?;
        Ok(assessment)
    }

    fn digest_view(&self) -> AssessmentDigestView<'_> {
        AssessmentDigestView {
            schema: ASSESSMENT_SCHEMA,
            assessment_id: &self.assessment_id,
            design: self.design,
            original_result_digest: &self.original_result_digest,
            followup_result_digest: &self.followup_result_digest,
            factual_lineage: &self.factual_lineage,
            independence_evidence: &self.independence_evidence,
            outcome: self.outcome,
            comparison: &self.comparison,
        }
    }

    pub fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&self.digest_view())
            .map_err(|error| ReplicationError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn verify_digest(&self) -> Result<()> {
        if self.compute_digest()? != self.assessment_digest {
            return Err(ReplicationError::AssessmentDigestMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_research_result::{
        MetricOutcome, MetricResult, ResultArtifactKind, ResultArtifactRef, ResearchResultManifest,
    };
    use symthaea_research_result::symthaea_research_protocol::{
        AnalysisPlanRef, BaselineSpec, FrozenProtocol, HypothesisDirection, HypothesisRole,
        HypothesisSpec, MetricRole, MetricSpec, MultiplicityPolicy, ResearchProtocol,
        ResearchRunRegistration, StoppingRule,
    };

    fn frozen() -> FrozenProtocol {
        ResearchProtocol::new(
            "p",
            "1",
            "question",
            vec![HypothesisSpec::new(
                "h",
                "effect exists",
                HypothesisRole::Primary,
                HypothesisDirection::TwoSided,
            )
            .unwrap()],
            vec![MetricSpec::new("m", "metric", "unit", MetricRole::Primary, "mean").unwrap()],
            vec![BaselineSpec::new("b", "baseline", "impl").unwrap()],
            vec![],
            StoppingRule::FixedSampleCount(10),
            MultiplicityPolicy::NotApplicable,
            AnalysisPlanRef::new("a", "1", "digest:a").unwrap(),
            "dataset plan",
            "seed plan",
        )
        .unwrap()
        .freeze(1)
        .unwrap()
    }

    fn result(
        frozen: &FrozenProtocol,
        id: &str,
        source: &str,
        dataset: &str,
        environment: &str,
        seeds: &str,
    ) -> ResearchResultManifest {
        let run = ResearchRunRegistration::new(
            frozen,
            format!("run-{id}"),
            2,
            source,
            dataset,
            environment,
            seeds,
        )
        .unwrap();
        ResearchResultManifest::new(
            frozen,
            run,
            id,
            3,
            vec![],
            vec![],
            false,
            vec![ResultArtifactRef::new(
                "analysis",
                ResultArtifactKind::Analysis,
                format!("digest:analysis:{id}"),
                "analysis",
            )
            .unwrap()],
            vec![MetricResult::new(
                "m",
                MetricOutcome::Numeric {
                    value: 1.0,
                    unit: "unit".into(),
                },
            )
            .unwrap()],
            vec![],
        )
        .unwrap()
    }

    fn comparison() -> ReplicationComparisonEvidence {
        ReplicationComparisonEvidence::new(
            "frozen comparison plan v1",
            "digest:comparison",
            "effect direction and interval overlap",
        )
        .unwrap()
    }

    #[test]
    fn direct_replication_requires_new_data() {
        let frozen = frozen();
        let a = result(&frozen, "a", "source", "data", "env", "seeds-a");
        let b = result(&frozen, "b", "source", "data", "env", "seeds-b");
        let err = ReplicationAssessment::new(
            "r",
            ReplicationDesign::DirectReplication,
            &a,
            &b,
            vec![],
            ReplicationOutcome::Concordant,
            comparison(),
        )
        .unwrap_err();
        assert_eq!(err, ReplicationError::DirectReplicationReusedDataset);
    }

    #[test]
    fn direct_replication_can_reuse_implementation_but_not_data() {
        let frozen = frozen();
        let a = result(&frozen, "a", "source", "data-a", "env", "seeds-a");
        let b = result(&frozen, "b", "source", "data-b", "env", "seeds-b");
        let assessment = ReplicationAssessment::new(
            "r",
            ReplicationDesign::DirectReplication,
            &a,
            &b,
            vec![],
            ReplicationOutcome::Concordant,
            comparison(),
        )
        .unwrap();
        assert_eq!(assessment.factual_lineage.source_commit, LineageRelation::Same);
        assert_eq!(assessment.factual_lineage.dataset_manifest, LineageRelation::Different);
        assessment.verify_digest().unwrap();
    }

    #[test]
    fn exact_reproduction_rejects_environment_change() {
        let frozen = frozen();
        let a = result(&frozen, "a", "source", "data", "env-a", "seeds");
        let b = result(&frozen, "b", "source", "data", "env-b", "seeds");
        let err = ReplicationAssessment::new(
            "r",
            ReplicationDesign::ExactReproduction,
            &a,
            &b,
            vec![],
            ReplicationOutcome::Concordant,
            comparison(),
        )
        .unwrap_err();
        assert_eq!(err, ReplicationError::ExactReproductionEnvironmentChanged);
    }

    #[test]
    fn reanalysis_requires_same_dataset() {
        let frozen = frozen();
        let a = result(&frozen, "a", "source-a", "data-a", "env", "seeds");
        let b = result(&frozen, "b", "source-b", "data-b", "env", "seeds");
        let err = ReplicationAssessment::new(
            "r",
            ReplicationDesign::Reanalysis,
            &a,
            &b,
            vec![],
            ReplicationOutcome::Mixed,
            comparison(),
        )
        .unwrap_err();
        assert_eq!(err, ReplicationError::ReanalysisDatasetChanged);
    }

    #[test]
    fn human_or_institutional_independence_is_evidence_backed_not_inferred() {
        let evidence = IndependenceEvidence::new(
            IndependenceDimension::Institution,
            "follow-up was performed by a separate laboratory",
            "digest:attestation",
        )
        .unwrap();
        assert_eq!(evidence.dimension, IndependenceDimension::Institution);
        assert_eq!(evidence.evidence_digest, "digest:attestation");
    }
}
