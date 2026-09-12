//! Outcome-blind preregistration, run binding, and deviation contracts.
//!
//! The goal is not bureaucracy. It is to make it difficult to rewrite a
//! hypothesis, primary metric, exclusion rule, baseline, or stopping rule after
//! seeing outcomes while still permitting transparent amendments and deviations.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, ProtocolError>;

const FINGERPRINT_SCHEMA: &str = "symthaea-research-protocol/v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    EmptyField(&'static str),
    MissingHypothesis,
    MissingPrimaryMetric,
    MissingBaseline,
    MissingStoppingRule,
    DuplicateId(String),
    InvalidSampleCount,
    InvalidTimeWindow { start_unix_ms: i64, end_unix_ms: i64 },
    InvalidTickHorizon,
    MissingArtifactDigest(&'static str),
    Serialization(String),
    ProtocolDigestMismatch,
    AmendmentBeforeFreeze,
    RunBeforeFreeze,
}

impl Display for ProtocolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::MissingHypothesis => write!(f, "research protocol requires at least one hypothesis"),
            Self::MissingPrimaryMetric => write!(f, "research protocol requires at least one primary metric"),
            Self::MissingBaseline => write!(f, "research protocol requires at least one baseline"),
            Self::MissingStoppingRule => write!(f, "research protocol requires an explicit stopping rule"),
            Self::DuplicateId(id) => write!(f, "duplicate protocol id: {id}"),
            Self::InvalidSampleCount => write!(f, "fixed sample/episode count must be > 0"),
            Self::InvalidTimeWindow { start_unix_ms, end_unix_ms } => write!(
                f,
                "time window requires end >= start, got {start_unix_ms}..={end_unix_ms}"
            ),
            Self::InvalidTickHorizon => write!(f, "fixed tick horizon must be > 0"),
            Self::MissingArtifactDigest(field) => write!(f, "{field} requires an artifact digest"),
            Self::Serialization(message) => write!(f, "protocol serialization failed: {message}"),
            Self::ProtocolDigestMismatch => write!(f, "run/amendment protocol digest does not match frozen protocol"),
            Self::AmendmentBeforeFreeze => write!(f, "protocol amendment requires a frozen parent protocol"),
            Self::RunBeforeFreeze => write!(f, "research run registration requires a frozen protocol"),
        }
    }
}

impl Error for ProtocolError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ProtocolError::EmptyField(field));
    }
    Ok(())
}

fn unique_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(ProtocolError::DuplicateId(id.to_string()));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisRole {
    Primary,
    Secondary,
    Exploratory,
    Safety,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisDirection {
    TwoSided,
    GreaterThan,
    LessThan,
    Equivalence,
    NonInferiority,
    Qualitative,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisSpec {
    pub id: String,
    pub statement: String,
    pub role: HypothesisRole,
    pub direction: HypothesisDirection,
}

impl HypothesisSpec {
    pub fn new(
        id: impl Into<String>,
        statement: impl Into<String>,
        role: HypothesisRole,
        direction: HypothesisDirection,
    ) -> Result<Self> {
        let id = id.into();
        let statement = statement.into();
        non_empty(&id, "hypothesis id")?;
        non_empty(&statement, "hypothesis statement")?;
        Ok(Self { id, statement, role, direction })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricRole {
    Primary,
    Secondary,
    Safety,
    Exploratory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricSpec {
    pub id: String,
    pub label: String,
    pub unit: String,
    pub role: MetricRole,
    pub aggregation: String,
    pub success_criterion: Option<String>,
}

impl MetricSpec {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        unit: impl Into<String>,
        role: MetricRole,
        aggregation: impl Into<String>,
    ) -> Result<Self> {
        let id = id.into();
        let label = label.into();
        let unit = unit.into();
        let aggregation = aggregation.into();
        non_empty(&id, "metric id")?;
        non_empty(&label, "metric label")?;
        non_empty(&unit, "metric unit")?;
        non_empty(&aggregation, "metric aggregation")?;
        Ok(Self { id, label, unit, role, aggregation, success_criterion: None })
    }

    pub fn with_success_criterion(mut self, criterion: impl Into<String>) -> Result<Self> {
        let criterion = criterion.into();
        non_empty(&criterion, "metric success criterion")?;
        self.success_criterion = Some(criterion);
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselineSpec {
    pub id: String,
    pub description: String,
    pub implementation_ref: String,
}

impl BaselineSpec {
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        implementation_ref: impl Into<String>,
    ) -> Result<Self> {
        let id = id.into();
        let description = description.into();
        let implementation_ref = implementation_ref.into();
        non_empty(&id, "baseline id")?;
        non_empty(&description, "baseline description")?;
        non_empty(&implementation_ref, "baseline implementation ref")?;
        Ok(Self { id, description, implementation_ref })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExclusionRule {
    pub id: String,
    pub criterion: String,
}

impl ExclusionRule {
    pub fn new(id: impl Into<String>, criterion: impl Into<String>) -> Result<Self> {
        let id = id.into();
        let criterion = criterion.into();
        non_empty(&id, "exclusion rule id")?;
        non_empty(&criterion, "exclusion criterion")?;
        Ok(Self { id, criterion })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoppingRule {
    FixedSampleCount(u64),
    FixedEpisodeCount(u64),
    FixedTickHorizon(u64),
    FixedTimeWindow { start_unix_ms: i64, end_unix_ms: i64 },
    SafetyStopOnly { safety_condition: String },
}

impl StoppingRule {
    fn validate(&self) -> Result<()> {
        match self {
            Self::FixedSampleCount(0) | Self::FixedEpisodeCount(0) => Err(ProtocolError::InvalidSampleCount),
            Self::FixedTickHorizon(0) => Err(ProtocolError::InvalidTickHorizon),
            Self::FixedTimeWindow { start_unix_ms, end_unix_ms } if end_unix_ms < start_unix_ms => {
                Err(ProtocolError::InvalidTimeWindow { start_unix_ms: *start_unix_ms, end_unix_ms: *end_unix_ms })
            }
            Self::SafetyStopOnly { safety_condition } => non_empty(safety_condition, "safety stop condition"),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultiplicityPolicy {
    NotApplicable,
    SeparateConfirmatoryFromExploratory,
    Bonferroni,
    Holm,
    FalseDiscoveryRate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisPlanRef {
    pub id: String,
    pub version: String,
    pub artifact_digest: String,
}

impl AnalysisPlanRef {
    pub fn new(
        id: impl Into<String>,
        version: impl Into<String>,
        artifact_digest: impl Into<String>,
    ) -> Result<Self> {
        let id = id.into();
        let version = version.into();
        let artifact_digest = artifact_digest.into();
        non_empty(&id, "analysis plan id")?;
        non_empty(&version, "analysis plan version")?;
        if artifact_digest.trim().is_empty() {
            return Err(ProtocolError::MissingArtifactDigest("analysis plan"));
        }
        Ok(Self { id, version, artifact_digest })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchProtocol {
    pub protocol_id: String,
    pub protocol_version: String,
    pub research_question: String,
    pub hypotheses: Vec<HypothesisSpec>,
    pub metrics: Vec<MetricSpec>,
    pub baselines: Vec<BaselineSpec>,
    pub exclusions: Vec<ExclusionRule>,
    pub stopping_rule: StoppingRule,
    pub multiplicity_policy: MultiplicityPolicy,
    pub analysis_plan: AnalysisPlanRef,
    pub dataset_plan: String,
    pub seed_plan: String,
    pub null_result_policy: String,
}

impl ResearchProtocol {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        protocol_id: impl Into<String>,
        protocol_version: impl Into<String>,
        research_question: impl Into<String>,
        hypotheses: Vec<HypothesisSpec>,
        metrics: Vec<MetricSpec>,
        baselines: Vec<BaselineSpec>,
        exclusions: Vec<ExclusionRule>,
        stopping_rule: StoppingRule,
        multiplicity_policy: MultiplicityPolicy,
        analysis_plan: AnalysisPlanRef,
        dataset_plan: impl Into<String>,
        seed_plan: impl Into<String>,
    ) -> Result<Self> {
        let protocol_id = protocol_id.into();
        let protocol_version = protocol_version.into();
        let research_question = research_question.into();
        let dataset_plan = dataset_plan.into();
        let seed_plan = seed_plan.into();
        non_empty(&protocol_id, "protocol id")?;
        non_empty(&protocol_version, "protocol version")?;
        non_empty(&research_question, "research question")?;
        non_empty(&dataset_plan, "dataset plan")?;
        non_empty(&seed_plan, "seed plan")?;
        if hypotheses.is_empty() {
            return Err(ProtocolError::MissingHypothesis);
        }
        if !metrics.iter().any(|metric| metric.role == MetricRole::Primary) {
            return Err(ProtocolError::MissingPrimaryMetric);
        }
        if baselines.is_empty() {
            return Err(ProtocolError::MissingBaseline);
        }
        stopping_rule.validate()?;
        unique_ids(hypotheses.iter().map(|value| value.id.as_str()))?;
        unique_ids(metrics.iter().map(|value| value.id.as_str()))?;
        unique_ids(baselines.iter().map(|value| value.id.as_str()))?;
        unique_ids(exclusions.iter().map(|value| value.id.as_str()))?;
        Ok(Self {
            protocol_id,
            protocol_version,
            research_question,
            hypotheses,
            metrics,
            baselines,
            exclusions,
            stopping_rule,
            multiplicity_policy,
            analysis_plan,
            dataset_plan,
            seed_plan,
            null_result_policy: "retain_and_report_all_confirmatory_results".into(),
        })
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&(FINGERPRINT_SCHEMA, self))
            .map_err(|error| ProtocolError::Serialization(error.to_string()))
    }

    pub fn freeze(self, frozen_at_unix_ms: i64) -> Result<FrozenProtocol> {
        let digest = blake3::hash(&self.canonical_bytes()?).to_hex().to_string();
        Ok(FrozenProtocol { protocol: self, frozen_at_unix_ms, digest })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenProtocol {
    protocol: ResearchProtocol,
    frozen_at_unix_ms: i64,
    digest: String,
}

impl FrozenProtocol {
    pub fn protocol(&self) -> &ResearchProtocol { &self.protocol }
    pub fn frozen_at_unix_ms(&self) -> i64 { self.frozen_at_unix_ms }
    pub fn digest(&self) -> &str { &self.digest }

    pub fn verify_digest(&self) -> Result<()> {
        let actual = blake3::hash(&self.protocol.canonical_bytes()?).to_hex().to_string();
        if actual != self.digest {
            return Err(ProtocolError::ProtocolDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmendmentTiming {
    BeforeDataCollection,
    BeforeOutcomeUnblinding,
    AfterOutcomeUnblinding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolAmendment {
    pub amendment_id: String,
    pub parent_protocol_digest: String,
    pub amended_at_unix_ms: i64,
    pub timing: AmendmentTiming,
    pub reason: String,
    pub changes: Vec<String>,
}

impl ProtocolAmendment {
    pub fn new(
        frozen: &FrozenProtocol,
        amendment_id: impl Into<String>,
        amended_at_unix_ms: i64,
        timing: AmendmentTiming,
        reason: impl Into<String>,
        changes: Vec<String>,
    ) -> Result<Self> {
        frozen.verify_digest()?;
        let amendment_id = amendment_id.into();
        let reason = reason.into();
        non_empty(&amendment_id, "amendment id")?;
        non_empty(&reason, "amendment reason")?;
        if amended_at_unix_ms < frozen.frozen_at_unix_ms() {
            return Err(ProtocolError::AmendmentBeforeFreeze);
        }
        for change in &changes {
            non_empty(change, "amendment change")?;
        }
        Ok(Self {
            amendment_id,
            parent_protocol_digest: frozen.digest().to_string(),
            amended_at_unix_ms,
            timing,
            reason,
            changes,
        })
    }

    pub fn is_confirmatory_safe(&self) -> bool {
        self.timing != AmendmentTiming::AfterOutcomeUnblinding
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchRunRegistration {
    pub run_id: String,
    pub protocol_digest: String,
    pub registered_at_unix_ms: i64,
    pub source_commit: String,
    pub dataset_manifest_digest: String,
    pub reproducibility_capsule_digest: String,
    pub seed_manifest_digest: String,
}

impl ResearchRunRegistration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frozen: &FrozenProtocol,
        run_id: impl Into<String>,
        registered_at_unix_ms: i64,
        source_commit: impl Into<String>,
        dataset_manifest_digest: impl Into<String>,
        reproducibility_capsule_digest: impl Into<String>,
        seed_manifest_digest: impl Into<String>,
    ) -> Result<Self> {
        frozen.verify_digest()?;
        if registered_at_unix_ms < frozen.frozen_at_unix_ms() {
            return Err(ProtocolError::RunBeforeFreeze);
        }
        let run_id = run_id.into();
        let source_commit = source_commit.into();
        let dataset_manifest_digest = dataset_manifest_digest.into();
        let reproducibility_capsule_digest = reproducibility_capsule_digest.into();
        let seed_manifest_digest = seed_manifest_digest.into();
        non_empty(&run_id, "run id")?;
        non_empty(&source_commit, "source commit")?;
        if dataset_manifest_digest.trim().is_empty() {
            return Err(ProtocolError::MissingArtifactDigest("dataset manifest"));
        }
        if reproducibility_capsule_digest.trim().is_empty() {
            return Err(ProtocolError::MissingArtifactDigest("reproducibility capsule"));
        }
        if seed_manifest_digest.trim().is_empty() {
            return Err(ProtocolError::MissingArtifactDigest("seed manifest"));
        }
        Ok(Self {
            run_id,
            protocol_digest: frozen.digest().to_string(),
            registered_at_unix_ms,
            source_commit,
            dataset_manifest_digest,
            reproducibility_capsule_digest,
            seed_manifest_digest,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolDeviation {
    pub deviation_id: String,
    pub description: String,
    pub detected_at_unix_ms: i64,
    pub affects_primary_analysis: bool,
}

impl ProtocolDeviation {
    pub fn new(
        deviation_id: impl Into<String>,
        description: impl Into<String>,
        detected_at_unix_ms: i64,
        affects_primary_analysis: bool,
    ) -> Result<Self> {
        let deviation_id = deviation_id.into();
        let description = description.into();
        non_empty(&deviation_id, "deviation id")?;
        non_empty(&description, "deviation description")?;
        Ok(Self { deviation_id, description, detected_at_unix_ms, affects_primary_analysis })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResultInterpretation {
    Confirmatory,
    ExploratoryDueToPostUnblindingAmendment,
    ExploratoryDueToPrimaryDeviation,
    Invalidated,
}

pub fn classify_result(
    amendments: &[ProtocolAmendment],
    deviations: &[ProtocolDeviation],
    invalidated: bool,
) -> ResultInterpretation {
    if invalidated {
        return ResultInterpretation::Invalidated;
    }
    if amendments.iter().any(|amendment| !amendment.is_confirmatory_safe()) {
        return ResultInterpretation::ExploratoryDueToPostUnblindingAmendment;
    }
    if deviations.iter().any(|deviation| deviation.affects_primary_analysis) {
        return ResultInterpretation::ExploratoryDueToPrimaryDeviation;
    }
    ResultInterpretation::Confirmatory
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protocol() -> ResearchProtocol {
        ResearchProtocol::new(
            "wetland-watch-semantic-downlink-v1",
            "1.0.0",
            "Does semantic prioritization improve mission-relevant information per transmitted byte?",
            vec![HypothesisSpec::new(
                "h1",
                "semantic scheduling outperforms simple ROI baseline on held-out scenes",
                HypothesisRole::Primary,
                HypothesisDirection::GreaterThan,
            ).unwrap()],
            vec![MetricSpec::new(
                "utility-per-byte",
                "Mission-relevant information per transmitted byte",
                "utility/byte",
                MetricRole::Primary,
                "mean over frozen held-out scenes",
            ).unwrap()],
            vec![BaselineSpec::new(
                "simple-roi",
                "conventional codec plus simple cloud/change ROI policy",
                "benchmark/simple-roi-v1",
            ).unwrap()],
            vec![ExclusionRule::new("corrupt-product", "exclude only products failing frozen checksum validation").unwrap()],
            StoppingRule::FixedSampleCount(100),
            MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
            AnalysisPlanRef::new("analysis", "1", "sha256:analysis").unwrap(),
            "frozen Sentinel scene manifest v1",
            "fixed scene/seed manifest",
        ).unwrap()
    }

    #[test]
    fn protocol_requires_primary_metric_and_baseline() {
        let mut value = protocol();
        value.metrics[0].role = MetricRole::Secondary;
        assert!(matches!(
            ResearchProtocol::new(
                value.protocol_id,
                value.protocol_version,
                value.research_question,
                value.hypotheses,
                value.metrics,
                value.baselines,
                value.exclusions,
                value.stopping_rule,
                value.multiplicity_policy,
                value.analysis_plan,
                value.dataset_plan,
                value.seed_plan,
            ),
            Err(ProtocolError::MissingPrimaryMetric)
        ));
    }

    #[test]
    fn frozen_protocol_detects_tampering() {
        let frozen = protocol().freeze(1_000).unwrap();
        assert!(frozen.verify_digest().is_ok());
        let mut tampered = frozen.clone();
        tampered.protocol.metrics[0].aggregation = "best 10 scenes only".into();
        assert_eq!(tampered.verify_digest().unwrap_err(), ProtocolError::ProtocolDigestMismatch);
    }

    #[test]
    fn post_unblinding_amendment_downgrades_confirmatory_status() {
        let frozen = protocol().freeze(1_000).unwrap();
        let amendment = ProtocolAmendment::new(
            &frozen,
            "a1",
            2_000,
            AmendmentTiming::AfterOutcomeUnblinding,
            "change metric after seeing outcomes",
            vec!["replace primary metric".into()],
        ).unwrap();
        assert_eq!(
            classify_result(&[amendment], &[], false),
            ResultInterpretation::ExploratoryDueToPostUnblindingAmendment
        );
    }

    #[test]
    fn run_binds_protocol_code_data_environment_and_seeds() {
        let frozen = protocol().freeze(1_000).unwrap();
        let run = ResearchRunRegistration::new(
            &frozen,
            "run-1",
            1_100,
            "deadbeef",
            "sha256:data",
            "sha256:capsule",
            "sha256:seeds",
        ).unwrap();
        assert_eq!(run.protocol_digest, frozen.digest());
    }

    #[test]
    fn primary_deviation_prevents_confirmatory_label() {
        let deviation = ProtocolDeviation::new(
            "d1",
            "primary metric was computed with wrong mask",
            2_000,
            true,
        ).unwrap();
        assert_eq!(
            classify_result(&[], &[deviation], false),
            ResultInterpretation::ExploratoryDueToPrimaryDeviation
        );
    }
}
