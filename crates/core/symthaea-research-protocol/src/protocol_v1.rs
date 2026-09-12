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
const FROZEN_PROTOCOL_SCHEMA: &str = "symthaea-research-protocol/frozen-v1";
const NULL_RESULT_POLICY: &str = "retain_and_report_all_confirmatory_results";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    EmptyField(&'static str),
    MissingHypothesis,
    MissingPrimaryMetric,
    MissingBaseline,
    MissingStoppingRule,
    DuplicateId(String),
    InvalidSampleCount,
    InvalidTimeWindow {
        start_unix_ms: i64,
        end_unix_ms: i64,
    },
    InvalidTickHorizon,
    MissingArtifactDigest(&'static str),
    InvalidNullResultPolicy,
    Serialization(String),
    ProtocolDigestMismatch,
    AmendmentBeforeFreeze,
    RunBeforeFreeze,
}

impl Display for ProtocolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::MissingHypothesis => {
                write!(f, "research protocol requires at least one hypothesis")
            }
            Self::MissingPrimaryMetric => {
                write!(f, "research protocol requires at least one primary metric")
            }
            Self::MissingBaseline => {
                write!(f, "research protocol requires at least one baseline")
            }
            Self::MissingStoppingRule => {
                write!(f, "research protocol requires an explicit stopping rule")
            }
            Self::DuplicateId(id) => write!(f, "duplicate protocol id: {id}"),
            Self::InvalidSampleCount => {
                write!(f, "fixed sample/episode count must be > 0")
            }
            Self::InvalidTimeWindow {
                start_unix_ms,
                end_unix_ms,
            } => write!(
                f,
                "time window requires end >= start, got {start_unix_ms}..={end_unix_ms}"
            ),
            Self::InvalidTickHorizon => write!(f, "fixed tick horizon must be > 0"),
            Self::MissingArtifactDigest(field) => {
                write!(f, "{field} requires an artifact digest")
            }
            Self::InvalidNullResultPolicy => write!(
                f,
                "research protocol null-result policy must retain and report all confirmatory results"
            ),
            Self::Serialization(message) => {
                write!(f, "protocol serialization failed: {message}")
            }
            Self::ProtocolDigestMismatch => write!(
                f,
                "run/amendment protocol digest does not match frozen protocol"
            ),
            Self::AmendmentBeforeFreeze => {
                write!(f, "protocol amendment requires a frozen parent protocol")
            }
            Self::RunBeforeFreeze => {
                write!(f, "research run registration requires a frozen protocol")
            }
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

fn artifact_digest(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ProtocolError::MissingArtifactDigest(field));
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
        let value = Self {
            id: id.into(),
            statement: statement.into(),
            role,
            direction,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        non_empty(&self.id, "hypothesis id")?;
        non_empty(&self.statement, "hypothesis statement")?;
        Ok(())
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
        let value = Self {
            id: id.into(),
            label: label.into(),
            unit: unit.into(),
            role,
            aggregation: aggregation.into(),
            success_criterion: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_success_criterion(mut self, criterion: impl Into<String>) -> Result<Self> {
        self.success_criterion = Some(criterion.into());
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        non_empty(&self.id, "metric id")?;
        non_empty(&self.label, "metric label")?;
        non_empty(&self.unit, "metric unit")?;
        non_empty(&self.aggregation, "metric aggregation")?;
        if let Some(criterion) = &self.success_criterion {
            non_empty(criterion, "metric success criterion")?;
        }
        Ok(())
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
        let value = Self {
            id: id.into(),
            description: description.into(),
            implementation_ref: implementation_ref.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        non_empty(&self.id, "baseline id")?;
        non_empty(&self.description, "baseline description")?;
        non_empty(&self.implementation_ref, "baseline implementation ref")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExclusionRule {
    pub id: String,
    pub criterion: String,
}

impl ExclusionRule {
    pub fn new(id: impl Into<String>, criterion: impl Into<String>) -> Result<Self> {
        let value = Self {
            id: id.into(),
            criterion: criterion.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        non_empty(&self.id, "exclusion rule id")?;
        non_empty(&self.criterion, "exclusion criterion")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoppingRule {
    FixedSampleCount(u64),
    FixedEpisodeCount(u64),
    FixedTickHorizon(u64),
    FixedTimeWindow {
        start_unix_ms: i64,
        end_unix_ms: i64,
    },
    SafetyStopOnly {
        safety_condition: String,
    },
}

impl StoppingRule {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::FixedSampleCount(0) | Self::FixedEpisodeCount(0) => {
                Err(ProtocolError::InvalidSampleCount)
            }
            Self::FixedTickHorizon(0) => Err(ProtocolError::InvalidTickHorizon),
            Self::FixedTimeWindow {
                start_unix_ms,
                end_unix_ms,
            } if end_unix_ms < start_unix_ms => Err(ProtocolError::InvalidTimeWindow {
                start_unix_ms: *start_unix_ms,
                end_unix_ms: *end_unix_ms,
            }),
            Self::SafetyStopOnly { safety_condition } => {
                non_empty(safety_condition, "safety stop condition")
            }
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
        let value = Self {
            id: id.into(),
            version: version.into(),
            artifact_digest: artifact_digest.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        non_empty(&self.id, "analysis plan id")?;
        non_empty(&self.version, "analysis plan version")?;
        artifact_digest(&self.artifact_digest, "analysis plan")?;
        Ok(())
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
        let value = Self {
            protocol_id: protocol_id.into(),
            protocol_version: protocol_version.into(),
            research_question: research_question.into(),
            hypotheses,
            metrics,
            baselines,
            exclusions,
            stopping_rule,
            multiplicity_policy,
            analysis_plan,
            dataset_plan: dataset_plan.into(),
            seed_plan: seed_plan.into(),
            null_result_policy: NULL_RESULT_POLICY.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        non_empty(&self.protocol_id, "protocol id")?;
        non_empty(&self.protocol_version, "protocol version")?;
        non_empty(&self.research_question, "research question")?;
        non_empty(&self.dataset_plan, "dataset plan")?;
        non_empty(&self.seed_plan, "seed plan")?;

        if self.hypotheses.is_empty() {
            return Err(ProtocolError::MissingHypothesis);
        }
        if !self
            .metrics
            .iter()
            .any(|metric| metric.role == MetricRole::Primary)
        {
            return Err(ProtocolError::MissingPrimaryMetric);
        }
        if self.baselines.is_empty() {
            return Err(ProtocolError::MissingBaseline);
        }
        if self.null_result_policy != NULL_RESULT_POLICY {
            return Err(ProtocolError::InvalidNullResultPolicy);
        }

        for hypothesis in &self.hypotheses {
            hypothesis.validate()?;
        }
        for metric in &self.metrics {
            metric.validate()?;
        }
        for baseline in &self.baselines {
            baseline.validate()?;
        }
        for exclusion in &self.exclusions {
            exclusion.validate()?;
        }
        self.stopping_rule.validate()?;
        self.analysis_plan.validate()?;

        unique_ids(self.hypotheses.iter().map(|value| value.id.as_str()))?;
        unique_ids(self.metrics.iter().map(|value| value.id.as_str()))?;
        unique_ids(self.baselines.iter().map(|value| value.id.as_str()))?;
        unique_ids(self.exclusions.iter().map(|value| value.id.as_str()))?;
        Ok(())
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&(FINGERPRINT_SCHEMA, self))
            .map_err(|error| ProtocolError::Serialization(error.to_string()))
    }

    pub fn freeze(self, frozen_at_unix_ms: i64) -> Result<FrozenProtocol> {
        self.validate()?;
        let digest = frozen_protocol_digest(&self, frozen_at_unix_ms)?;
        Ok(FrozenProtocol {
            protocol: self,
            frozen_at_unix_ms,
            digest,
        })
    }
}

fn frozen_protocol_digest(protocol: &ResearchProtocol, frozen_at_unix_ms: i64) -> Result<String> {
    let canonical_protocol = protocol.canonical_bytes()?;
    let bytes = serde_json::to_vec(&(
        FROZEN_PROTOCOL_SCHEMA,
        canonical_protocol,
        frozen_at_unix_ms,
    ))
    .map_err(|error| ProtocolError::Serialization(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenProtocol {
    protocol: ResearchProtocol,
    frozen_at_unix_ms: i64,
    digest: String,
}

impl FrozenProtocol {
    pub fn protocol(&self) -> &ResearchProtocol {
        &self.protocol
    }

    pub fn frozen_at_unix_ms(&self) -> i64 {
        self.frozen_at_unix_ms
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Verify only the content-addressed frozen-record identity.
    ///
    /// A matching digest does not imply that the contained protocol is semantically valid;
    /// callers accepting imported evidence should use [`Self::validate`].
    pub fn verify_digest(&self) -> Result<()> {
        let actual = frozen_protocol_digest(&self.protocol, self.frozen_at_unix_ms)?;
        if actual != self.digest {
            return Err(ProtocolError::ProtocolDigestMismatch);
        }
        Ok(())
    }

    /// Verify both frozen-record identity and all recursively checkable protocol semantics.
    pub fn validate(&self) -> Result<()> {
        self.verify_digest()?;
        self.protocol.validate()
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
        let value = Self {
            amendment_id: amendment_id.into(),
            parent_protocol_digest: frozen.digest().to_string(),
            amended_at_unix_ms,
            timing,
            reason: reason.into(),
            changes,
        };
        value.validate_against(frozen)?;
        Ok(value)
    }

    pub fn validate_against(&self, frozen: &FrozenProtocol) -> Result<()> {
        frozen.validate()?;
        non_empty(&self.amendment_id, "amendment id")?;
        non_empty(&self.reason, "amendment reason")?;
        if self.parent_protocol_digest != frozen.digest() {
            return Err(ProtocolError::ProtocolDigestMismatch);
        }
        if self.amended_at_unix_ms < frozen.frozen_at_unix_ms() {
            return Err(ProtocolError::AmendmentBeforeFreeze);
        }
        for change in &self.changes {
            non_empty(change, "amendment change")?;
        }
        Ok(())
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
        let value = Self {
            run_id: run_id.into(),
            protocol_digest: frozen.digest().to_string(),
            registered_at_unix_ms,
            source_commit: source_commit.into(),
            dataset_manifest_digest: dataset_manifest_digest.into(),
            reproducibility_capsule_digest: reproducibility_capsule_digest.into(),
            seed_manifest_digest: seed_manifest_digest.into(),
        };
        value.validate_against(frozen)?;
        Ok(value)
    }

    pub fn validate_against(&self, frozen: &FrozenProtocol) -> Result<()> {
        frozen.validate()?;
        if self.protocol_digest != frozen.digest() {
            return Err(ProtocolError::ProtocolDigestMismatch);
        }
        if self.registered_at_unix_ms < frozen.frozen_at_unix_ms() {
            return Err(ProtocolError::RunBeforeFreeze);
        }
        non_empty(&self.run_id, "run id")?;
        non_empty(&self.source_commit, "source commit")?;
        artifact_digest(&self.dataset_manifest_digest, "dataset manifest")?;
        artifact_digest(
            &self.reproducibility_capsule_digest,
            "reproducibility capsule",
        )?;
        artifact_digest(&self.seed_manifest_digest, "seed manifest")?;
        Ok(())
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
        let value = Self {
            deviation_id: deviation_id.into(),
            description: description.into(),
            detected_at_unix_ms,
            affects_primary_analysis,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        non_empty(&self.deviation_id, "deviation id")?;
        non_empty(&self.description, "deviation description")?;
        Ok(())
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
    if amendments
        .iter()
        .any(|amendment| !amendment.is_confirmatory_safe())
    {
        return ResultInterpretation::ExploratoryDueToPostUnblindingAmendment;
    }
    if deviations
        .iter()
        .any(|deviation| deviation.affects_primary_analysis)
    {
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
            vec![
                HypothesisSpec::new(
                    "h1",
                    "semantic scheduling outperforms simple ROI baseline on held-out scenes",
                    HypothesisRole::Primary,
                    HypothesisDirection::GreaterThan,
                )
                .unwrap(),
            ],
            vec![
                MetricSpec::new(
                    "utility-per-byte",
                    "Mission-relevant information per transmitted byte",
                    "utility/byte",
                    MetricRole::Primary,
                    "mean over frozen held-out scenes",
                )
                .unwrap(),
            ],
            vec![
                BaselineSpec::new(
                    "simple-roi",
                    "conventional codec plus simple cloud/change ROI policy",
                    "benchmark/simple-roi-v1",
                )
                .unwrap(),
            ],
            vec![
                ExclusionRule::new(
                    "corrupt-product",
                    "exclude only products failing frozen checksum validation",
                )
                .unwrap(),
            ],
            StoppingRule::FixedSampleCount(100),
            MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
            AnalysisPlanRef::new("analysis", "1", "sha256:analysis").unwrap(),
            "frozen Sentinel scene manifest v1",
            "fixed scene/seed manifest",
        )
        .unwrap()
    }

    #[test]
    fn protocol_requires_primary_metric_and_baseline() {
        let mut value = protocol();
        value.metrics[0].role = MetricRole::Secondary;
        assert_eq!(
            value.validate().unwrap_err(),
            ProtocolError::MissingPrimaryMetric
        );
    }

    #[test]
    fn frozen_protocol_detects_protocol_tampering() {
        let frozen = protocol().freeze(1_000).unwrap();
        assert!(frozen.validate().is_ok());
        let mut tampered = frozen.clone();
        tampered.protocol.metrics[0].aggregation = "best 10 scenes only".into();
        assert_eq!(
            tampered.verify_digest().unwrap_err(),
            ProtocolError::ProtocolDigestMismatch
        );
    }

    #[test]
    fn frozen_protocol_identity_commits_freeze_boundary() {
        let frozen = protocol().freeze(1_000).unwrap();
        let mut tampered = frozen.clone();
        tampered.frozen_at_unix_ms = 999;
        assert_eq!(
            tampered.verify_digest().unwrap_err(),
            ProtocolError::ProtocolDigestMismatch
        );
    }

    #[test]
    fn recomputed_digest_does_not_rescue_semantically_invalid_protocol() {
        let mut imported = protocol().freeze(1_000).unwrap();
        imported.protocol.metrics[0].role = MetricRole::Secondary;
        imported.digest =
            frozen_protocol_digest(&imported.protocol, imported.frozen_at_unix_ms).unwrap();

        assert!(imported.verify_digest().is_ok());
        assert_eq!(
            imported.validate().unwrap_err(),
            ProtocolError::MissingPrimaryMetric
        );
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
        )
        .unwrap();
        assert_eq!(
            classify_result(&[amendment], &[], false),
            ResultInterpretation::ExploratoryDueToPostUnblindingAmendment
        );
    }

    #[test]
    fn imported_amendment_revalidates_parent_and_freeze_boundary() {
        let frozen = protocol().freeze(1_000).unwrap();
        let mut amendment = ProtocolAmendment::new(
            &frozen,
            "a1",
            1_100,
            AmendmentTiming::BeforeOutcomeUnblinding,
            "clarify logging",
            vec!["record additional diagnostics".into()],
        )
        .unwrap();
        amendment.parent_protocol_digest = "foreign-protocol".into();
        assert_eq!(
            amendment.validate_against(&frozen).unwrap_err(),
            ProtocolError::ProtocolDigestMismatch
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
        )
        .unwrap();
        assert_eq!(run.protocol_digest, frozen.digest());
        run.validate_against(&frozen).unwrap();
    }

    #[test]
    fn imported_run_cannot_bypass_constructor_time_boundary() {
        let frozen = protocol().freeze(1_000).unwrap();
        let run = ResearchRunRegistration::new(
            &frozen,
            "run-1",
            1_100,
            "deadbeef",
            "sha256:data",
            "sha256:capsule",
            "sha256:seeds",
        )
        .unwrap();
        let mut serialized = serde_json::to_value(run).unwrap();
        serialized["registered_at_unix_ms"] = serde_json::json!(999);
        let imported: ResearchRunRegistration = serde_json::from_value(serialized).unwrap();

        assert_eq!(
            imported.validate_against(&frozen).unwrap_err(),
            ProtocolError::RunBeforeFreeze
        );
    }

    #[test]
    fn recursive_validation_rejects_invalid_nested_fields() {
        let mut imported = protocol().freeze(1_000).unwrap();
        imported.protocol.hypotheses[0].statement.clear();
        imported.digest =
            frozen_protocol_digest(&imported.protocol, imported.frozen_at_unix_ms).unwrap();

        assert!(imported.verify_digest().is_ok());
        assert_eq!(
            imported.validate().unwrap_err(),
            ProtocolError::EmptyField("hypothesis statement")
        );
    }

    #[test]
    fn null_result_retention_policy_is_semantically_enforced() {
        let mut imported = protocol().freeze(1_000).unwrap();
        imported.protocol.null_result_policy = "report_only_positive_results".into();
        imported.digest =
            frozen_protocol_digest(&imported.protocol, imported.frozen_at_unix_ms).unwrap();

        assert!(imported.verify_digest().is_ok());
        assert_eq!(
            imported.validate().unwrap_err(),
            ProtocolError::InvalidNullResultPolicy
        );
    }

    #[test]
    fn primary_deviation_prevents_confirmatory_label() {
        let deviation = ProtocolDeviation::new(
            "d1",
            "primary metric was computed with wrong mask",
            2_000,
            true,
        )
        .unwrap();
        assert_eq!(
            classify_result(&[], &[deviation], false),
            ResultInterpretation::ExploratoryDueToPrimaryDeviation
        );
    }
}
