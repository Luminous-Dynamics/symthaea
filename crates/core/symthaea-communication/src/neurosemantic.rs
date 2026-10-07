//! Neurosemantic communication protocol primitives.
//!
//! Protocol infrastructure for exchanging derived cognitive representations.
//! This is not a claim that arbitrary thoughts can be decoded or written.

use crate::{content_hash, RepresentationFamily};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Stable neurosemantic protocol version.
pub const NEUROSEMANTIC_PROTOCOL_VERSION: u16 = 1;

/// Maximum encoded payload size before content hashing.
pub const MAX_NEUROSEMANTIC_PAYLOAD_BYTES: usize = 1_048_576;

/// Maximum number of replay-tracker keys retained in memory.
pub const MAX_TRACKED_NEUROSEMANTIC_SESSIONS: usize = 4096;

/// Maximum identifier size accepted by protocol constructors and validators.
pub const MAX_NEUROSEMANTIC_ID_BYTES: usize = 4096;

/// Maximum serialized packet/authorization artifact size before JSON materialization.
pub const MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES: usize = MAX_NEUROSEMANTIC_PAYLOAD_BYTES;
pub const MAX_NEUROSEMANTIC_JURISDICTION_ID_BYTES: usize = 64;
pub const MAX_NEUROSEMANTIC_SECONDARY_USE_CLASSES: usize = 16;
pub const MAX_NEUROSEMANTIC_DESTINATION_JURISDICTIONS: usize = 16;

const NEUROSEMANTIC_POLICY_PROVENANCE_DOMAIN: &[u8] =
    b"symthaea-neurosemantic-policy-provenance-v1\0";
const NEUROSEMANTIC_DERIVATION_PROVENANCE_DOMAIN: &[u8] =
    b"symthaea-neurosemantic-derivation-provenance-v1\0";
const NEUROSEMANTIC_STATUS_SOURCE_DOMAIN: &[u8] =
    b"symthaea-neurosemantic-status-source-v1\0";
const NEUROSEMANTIC_LIFECYCLE_VERIFICATION_SCOPE_DOMAIN: &[u8] =
    b"symthaea-neurosemantic-lifecycle-verification-scope-v1\0";
const NEUROSEMANTIC_POLICY_ATTESTATION_DOMAIN: &[u8] =
    b"symthaea-neurosemantic-policy-attestation-v1\0";
const MAX_NEUROSEMANTIC_AUTHORITY_REF_BYTES: usize = 4096;
const MAX_NEUROSEMANTIC_AUTHORITY_KEY_REF_BYTES: usize = 4096;
const MAX_NEUROSEMANTIC_AUTHORITY_SIGNATURE_BYTES: usize = 64;
const MAX_NEUROSEMANTIC_RESOLVER_REF_BYTES: usize = 4096;
const MAX_NEUROSEMANTIC_STATUS_SOURCE_REF_BYTES: usize = 4096;
const MAX_NEUROSEMANTIC_DERIVATION_INPUT_ARTIFACTS: usize = 32;
const MAX_NEUROSEMANTIC_LIFECYCLE_VERIFICATION_TARGETS: usize = 4096;
const MAX_NEUROSEMANTIC_AUTHORITY_RESOLUTION_TTL_S: u64 = 86_400;
const NEUROSEMANTIC_AUTHORITY_RESOLUTION_DOMAIN: &[u8] =
    b"symthaea-neurosemantic-authority-resolution-v1\0";

/// Representation channel used for routing and authorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CognitiveChannel {
    Semantic,
    Affective,
    Spatial,
    Temporal,
    Procedural,
    Sensory,
}

/// Direction of information flow from the subject's perspective.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelDirection {
    /// Subject -> peer.
    Read,
    /// Peer -> subject.
    Write,
}

/// Purpose binding prevents ambient authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CommunicationPurpose {
    AssistiveCommunication,
    HumanCollaboration,
    AgentCoordination,
    Research,
}

/// Machine-readable class of the cognitive data being transported.
/// Unknown is the conservative legacy/default state and cannot authorize
/// neurosemantic access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticDataClass {
    #[default]
    Unknown,
    RawNeuralRecording,
    DerivedNeuralFeature,
    SemanticRepresentation,
    DecodedClaim,
    PersonalizedDecoderModel,
}

/// Machine-readable inference classes a data product may expose or enable.
/// These remain separate from data class and transport sensitivity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticInferenceClass {
    Unknown,
    SignalPattern,
    UnitPattern,
    LinguisticContent,
    SemanticContent,
    AffectiveState,
    Intent,
    Identity,
}

pub const NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION: u16 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticSecondaryUse {
    Research,
    ModelTraining,
    ProductDevelopment,
    CommercialAnalytics,
    BehavioralProfiling,
    AffectiveInference,
    IdentityInference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeurosemanticRetentionPolicy {
    Ephemeral,
    UntilUnixS(u64),
}

pub const NEUROSEMANTIC_DERIVATION_LINEAGE_SCHEMA_VERSION: u16 = 2;

/// Structured external lineage evidence for a derived cognitive artifact.
///
/// This is intentionally a compact protocol boundary, not a full provenance ontology:
/// input entities, the transformation activity, its execution revision, and the output
/// artifact are explicit so an independent verifier can check identity relationships.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticDerivationLineageRecord {
    pub schema_version: u16,
    pub lineage_ref: String,
    #[serde(default)]
    pub input_artifact_refs: Vec<String>,
    #[serde(default)]
    pub input_artifact_hashes: Vec<String>,
    pub activity_ref: String,
    pub activity_revision: String,
    pub output_artifact_hash: String,
    /// Canonical Git object ID for the execution context that produced the artifact.
    pub execution_revision: String,
    pub generated_at_unix_s: u64,
}

/// Post-generation lifecycle action for a derived artifact. This records an
/// external effect/evidence claim; it is not an authorization primitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticArtifactLifecycleAction {
    AccessRevocation,
    Retention,
    Erasure,
    Rectification,
    Supersession,
}

/// State reported for an external lifecycle action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticArtifactLifecycleState {
    Requested,
    Accepted,
    Processing,
    Applied,
    IndependentlyVerified,
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticArtifactLifecycleVerificationScope {
    /// Only the exact derived artifact named by the receipt was independently inspected.
    ArtifactOnly,
    /// A separately content-addressed target-set artifact enumerated the descendants inspected.
    EnumeratedTargetSet,
}

pub const NEUROSEMANTIC_ARTIFACT_LIFECYCLE_RECEIPT_SCHEMA_VERSION: u16 = 3;
pub const NEUROSEMANTIC_ARTIFACT_LIFECYCLE_VERIFICATION_SCOPE_SCHEMA_VERSION: u16 = 1;
pub const NEUROSEMANTIC_REMEDIATION_IMPACT_ARTIFACT_SCHEMA_VERSION: u16 = 1;
const MAX_NEUROSEMANTIC_IMPACT_DIMENSION_REFS: usize = 32;
const MAX_NEUROSEMANTIC_REMEDIATION_EVALUATION_MEMBERS: usize = 4096;
const MAX_NEUROSEMANTIC_REMEDIATION_METHOD_REF_BYTES: usize = 4096;


/// Frozen membership manifest for a remediation evaluation population.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticRemediationEvaluationSetKind {
    Forget,
    Retain,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationEvaluationSetManifest {
    pub schema_version: u16,
    pub set_ref: String,
    pub set_kind: NeurosemanticRemediationEvaluationSetKind,
    pub source_dataset_manifest_hash: String,
    pub member_artifact_hashes: Vec<String>,
}

pub const NEUROSEMANTIC_REMEDIATION_EVALUATION_SET_SCHEMA_VERSION: u16 = 1;

impl NeurosemanticRemediationEvaluationSetManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_EVALUATION_SET_SCHEMA_VERSION
            || !valid_identifier(&self.set_ref)
            || !valid_blake3_digest(&self.source_dataset_manifest_hash)
            || self.member_artifact_hashes.is_empty()
            || self.member_artifact_hashes.len() > MAX_NEUROSEMANTIC_REMEDIATION_EVALUATION_MEMBERS
            || self.member_artifact_hashes.iter().any(|hash| !valid_blake3_digest(hash))
        {
            return Err("neurosemantic remediation evaluation set manifest fields are invalid".into());
        }
        let unique_members: BTreeSet<&str> =
            self.member_artifact_hashes.iter().map(String::as_str).collect();
        if unique_members.len() != self.member_artifact_hashes.len() {
            return Err("neurosemantic remediation evaluation set manifest contains duplicate members".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation evaluation set manifest JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let manifest: Self = serde_json::from_slice(bytes).map_err(|error| {
            format!("neurosemantic remediation evaluation set manifest JSON: {error}")
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.member_artifact_hashes.sort();
        Ok(content_hash(
            &serde_json::to_vec(&canonical)
                .map_err(|error| format!("neurosemantic remediation evaluation set serialization: {error}"))?,
        ))
    }
}

/// One exact member/group binding in the fairness evaluation split.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationEvaluationSplitMember {
    pub subject_artifact_hash: String,
    pub group_ref: String,
}

impl NeurosemanticRemediationEvaluationSplitMember {
    fn validate(&self) -> Result<(), String> {
        if !valid_blake3_digest(&self.subject_artifact_hash)
            || !valid_identifier(&self.group_ref)
        {
            return Err("neurosemantic remediation evaluation split member fields are invalid".into());
        }
        Ok(())
    }
}

/// Typed identity of the exact evaluation split used for subgroup/fairness analysis.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationEvaluationSplitManifest {
    pub schema_version: u16,
    pub split_ref: String,
    pub source_dataset_manifest_hash: String,
    pub members: Vec<NeurosemanticRemediationEvaluationSplitMember>,
}

pub const NEUROSEMANTIC_REMEDIATION_EVALUATION_SPLIT_SCHEMA_VERSION: u16 = 1;

impl NeurosemanticRemediationEvaluationSplitManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_EVALUATION_SPLIT_SCHEMA_VERSION
            || !valid_identifier(&self.split_ref)
            || !valid_blake3_digest(&self.source_dataset_manifest_hash)
            || self.members.is_empty()
            || self.members.len() > MAX_NEUROSEMANTIC_REMEDIATION_EVALUATION_MEMBERS
        {
            return Err("neurosemantic remediation evaluation split manifest fields are invalid".into());
        }
        let mut subjects = BTreeSet::new();
        for member in &self.members {
            member.validate()?;
            if !subjects.insert(member.subject_artifact_hash.as_str()) {
                return Err("neurosemantic remediation evaluation split contains duplicate subjects".into());
            }
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation evaluation split manifest JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let manifest: Self = serde_json::from_slice(bytes).map_err(|error| {
            format!("neurosemantic remediation evaluation split manifest JSON: {error}")
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.members.sort_by(|left, right| {
            left.subject_artifact_hash
                .cmp(&right.subject_artifact_hash)
                .then_with(|| left.group_ref.cmp(&right.group_ref))
        });
        Ok(content_hash(&serde_json::to_vec(&canonical).map_err(|error| {
            format!("neurosemantic remediation evaluation split serialization: {error}")
        })?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticRemediationEvaluationMethodKind {
    RecoveryAttack,
    RepresentationResidualProbe,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationEvaluationMethod {
    pub schema_version: u16,
    pub method_ref: String,
    pub kind: NeurosemanticRemediationEvaluationMethodKind,
    pub protocol_hash: String,
    pub implementation_revision: String,
}

pub const NEUROSEMANTIC_REMEDIATION_EVALUATION_METHOD_SCHEMA_VERSION: u16 = 1;
pub const NEUROSEMANTIC_REMEDIATION_EVALUATION_MANIFEST_SCHEMA_VERSION: u16 = 1;
pub const NEUROSEMANTIC_REMEDIATION_EVALUATION_ENVIRONMENT_SCHEMA_VERSION: u16 = 1;

/// Machine-readable descriptor for the runtime environment in which remediation impact was evaluated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationEvaluationEnvironment {
    pub schema_version: u16,
    pub environment_ref: String,
    pub platform_ref: String,
    pub runtime_ref: String,
    pub toolchain_ref: String,
    pub dependency_lock_hash: String,
    pub configuration_hash: String,
    pub execution_revision: String,
}


/// Typed structural record of what each remediation evaluation dimension actually produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticRemediationMeasurementKind {
    Forgetfulness,
    UtilityImpact,
    FairnessImpact,
    RecoveryRisk,
    RepresentationResidual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeurosemanticRemediationMetricDirection {
    HigherIsBetter,
    LowerIsBetter,
    DescriptiveOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeurosemanticRemediationUncertainty {
    NotEstimated,
    Interval {
        lower_numerator: i64,
        upper_numerator: i64,
        scale: u32,
        confidence_level_bps: u16,
        uncertainty_method_ref: String,
        uncertainty_computation_artifact_hash: String,
    },
}

/// Target population model for an uncertainty calculation.
///
/// FixedEvaluationPopulation describes uncertainty only within the exact frozen
/// evaluation population. Superpopulation permits inferential claims beyond that
/// frozen population under the declared sampling assumptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticRemediationUncertaintyInferenceScope {
    FixedEvaluationPopulation,
    Superpopulation,
}

/// Sampling design declared for statistical inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticRemediationStatisticalSamplingDesign {
    ProbabilitySample,
    CensusOfTargetPopulation,
    NonProbabilitySample,
    Unknown,
}

/// Dependence structure declared for the observed analysis units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticRemediationStatisticalDependenceModel {
    IndependentObservationUnits,
    Clustered,
    RepeatedMeasures,
    Unknown,
}

/// Content-addressed statistical-design contract for an uncertainty calculation.
///
/// This makes method applicability explicit and independently checkable. It records what
/// the evaluator claims about the analysis unit, outcome model, sampling design, and
/// dependence structure; it does not itself prove those declarations are empirically true.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationStatisticalDesignArtifact {
    pub schema_version: u16,
    pub design_ref: String,
    pub metric_ref: String,
    pub metric_definition_hash: String,
    pub observation_set_hash: String,
    pub inference_scope: NeurosemanticRemediationUncertaintyInferenceScope,
    pub sampling_design: NeurosemanticRemediationStatisticalSamplingDesign,
    pub dependence_model: NeurosemanticRemediationStatisticalDependenceModel,
    pub analysis_unit_ref: String,
    pub outcome_model_ref: String,
    pub study_protocol_hash: String,
    pub assumptions_hash: String,
    pub execution_revision: String,
}

pub const NEUROSEMANTIC_REMEDIATION_STATISTICAL_DESIGN_SCHEMA_VERSION: u16 = 1;

impl NeurosemanticRemediationStatisticalDesignArtifact {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_STATISTICAL_DESIGN_SCHEMA_VERSION
            || !valid_identifier(&self.design_ref)
            || !valid_identifier(&self.metric_ref)
            || !valid_blake3_digest(&self.metric_definition_hash)
            || !valid_blake3_digest(&self.observation_set_hash)
            || !valid_identifier(&self.analysis_unit_ref)
            || !valid_identifier(&self.outcome_model_ref)
            || !valid_blake3_digest(&self.study_protocol_hash)
            || !valid_blake3_digest(&self.assumptions_hash)
            || !valid_execution_revision(&self.execution_revision)
        {
            return Err("neurosemantic remediation statistical design fields are invalid".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation statistical design JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let artifact: Self = serde_json::from_slice(bytes).map_err(|error| {
            format!("neurosemantic remediation statistical design JSON: {error}")
        })?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        Ok(content_hash(&serde_json::to_vec(self).map_err(|error| {
            format!("neurosemantic remediation statistical design serialization: {error}")
        })?))
    }
}

/// Content-addressed output record for a declared uncertainty calculation.
///
/// This binds the uncertainty result to the exact metric definition, observation set,
/// point estimate, method identity, inference scope, statistical-design artifact,
/// assumptions, and execution revision. It does not, by itself, prove that the declared
/// statistical design assumptions are empirically true.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationUncertaintyComputationArtifact {
    pub schema_version: u16,
    pub uncertainty_ref: String,
    pub metric_ref: String,
    pub metric_definition_hash: String,
    pub observation_set_hash: String,
    pub point_estimate_numerator: i64,
    pub point_estimate_scale: u32,
    pub lower_numerator: i64,
    pub upper_numerator: i64,
    pub scale: u32,
    pub confidence_level_bps: u16,
    pub inference_scope: NeurosemanticRemediationUncertaintyInferenceScope,
    pub statistical_design_hash: String,
    pub method_ref: String,
    /// Content-addressed assumptions statement used by the uncertainty procedure.
    pub assumptions_hash: String,
    pub assumptions_ref: String,
    pub execution_revision: String,
}

impl NeurosemanticRemediationUncertaintyComputationArtifact {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_UNCERTAINTY_COMPUTATION_SCHEMA_VERSION
            || !valid_identifier(&self.uncertainty_ref)
            || !valid_identifier(&self.metric_ref)
            || !valid_blake3_digest(&self.metric_definition_hash)
            || !valid_blake3_digest(&self.observation_set_hash)
            || self.point_estimate_scale > 12
            || self.scale > 12
            || self.lower_numerator > self.upper_numerator
            || self.point_estimate_numerator < self.lower_numerator
            || self.point_estimate_numerator > self.upper_numerator
            || self.scale != self.point_estimate_scale
            || self.confidence_level_bps == 0
            || self.confidence_level_bps > 10_000
            || !valid_blake3_digest(&self.statistical_design_hash)
            || !valid_identifier(&self.method_ref)
            || !valid_blake3_digest(&self.assumptions_hash)
            || !valid_identifier(&self.assumptions_ref)
            || !valid_execution_revision(&self.execution_revision)
        {
            return Err("neurosemantic remediation uncertainty computation fields are invalid".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation uncertainty computation JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let artifact: Self = serde_json::from_slice(bytes).map_err(|error| {
            format!("neurosemantic remediation uncertainty computation JSON: {error}")
        })?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        Ok(content_hash(&serde_json::to_vec(self).map_err(|error| {
            format!("neurosemantic remediation uncertainty computation serialization: {error}")
        })?))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationMetricDefinition {
    pub schema_version: u16,
    pub metric_ref: String,
    pub kind: NeurosemanticRemediationMeasurementKind,
    pub estimand_ref: String,
    pub scope_ref: String,
    pub unit_ref: String,
    pub aggregation_ref: String,
    pub direction: NeurosemanticRemediationMetricDirection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationMeasurement {
    pub metric_ref: String,
    pub kind: NeurosemanticRemediationMeasurementKind,
    pub status: NeurosemanticRemediationImpactDisposition,
    pub estimate_numerator: i64,
    pub estimate_scale: u32,
    pub uncertainty: NeurosemanticRemediationUncertainty,
    /// Number of cases eligible for this measurement.
    pub eligible_sample_count: u64,
    /// Number of eligible cases actually observed by the measurement procedure.
    pub observed_sample_count: u64,
    /// Number of observed cases counted as failures by the declared metric procedure.
    pub failure_count: u64,
}

/// One observed case contributing to a declared remediation metric.
///
/// This protocol boundary records a binary metric-defined failure event for one concrete
/// subject artifact. Scientific meaning remains bounded by the metric definition and study protocol.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationObservationRecord {
    pub observation_ref: String,
    pub subject_artifact_hash: String,
    pub failure_observed: bool,
    /// Required for subgroup-gap metrics so subgroup membership is itself auditable.
    #[serde(default)]
    pub group_ref: Option<String>,
}

impl NeurosemanticRemediationObservationRecord {
    fn validate(&self, require_group: bool) -> Result<(), String> {
        if !valid_identifier(&self.observation_ref)
            || !valid_blake3_digest(&self.subject_artifact_hash)
            || (require_group
                && self
                    .group_ref
                    .as_deref()
                    .is_none_or(|group| !valid_identifier(group)))
            || self
                .group_ref
                .as_deref()
                .is_some_and(|group| !valid_identifier(group))
        {
            return Err("neurosemantic remediation observation record fields are invalid".into());
        }
        Ok(())
    }
}

/// Frozen eligible population plus the exact observations actually evaluated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationObservationSetArtifact {
    pub schema_version: u16,
    pub observation_set_ref: String,
    pub metric_ref: String,
    pub kind: NeurosemanticRemediationMeasurementKind,
    pub scope_ref: String,
    /// Content-addressed identity of the canonical eligible population/split used by the metric.
    pub population_manifest_hash: String,
    pub eligible_subject_artifact_hashes: Vec<String>,
    pub observations: Vec<NeurosemanticRemediationObservationRecord>,
}

impl NeurosemanticRemediationObservationSetArtifact {
    pub fn validate(&self, aggregation_ref: &str) -> Result<(), String> {
        let require_group = aggregation_ref == "worst-subgroup-gap";
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_OBSERVATION_SET_SCHEMA_VERSION
            || !valid_identifier(&self.observation_set_ref)
            || !valid_identifier(&self.metric_ref)
            || self.scope_ref.is_empty()
            || self.scope_ref.len() > MAX_NEUROSEMANTIC_REMEDIATION_METRIC_TEXT_BYTES
            || !valid_blake3_digest(&self.population_manifest_hash)
            || self.eligible_subject_artifact_hashes.is_empty()
            || self.eligible_subject_artifact_hashes.len() > MAX_NEUROSEMANTIC_REMEDIATION_OBSERVATIONS
            || self.observations.len() > MAX_NEUROSEMANTIC_REMEDIATION_OBSERVATIONS
            || self
                .eligible_subject_artifact_hashes
                .iter()
                .any(|hash| !valid_blake3_digest(hash))
        {
            return Err("neurosemantic remediation observation set fields are invalid".into());
        }

        let eligible: BTreeSet<&str> = self
            .eligible_subject_artifact_hashes
            .iter()
            .map(String::as_str)
            .collect();
        if eligible.len() != self.eligible_subject_artifact_hashes.len() {
            return Err("neurosemantic remediation observation set contains duplicate eligible subjects".into());
        }

        let mut observed_subjects = BTreeSet::new();
        let mut observation_refs = BTreeSet::new();
        for observation in &self.observations {
            observation.validate(require_group)?;
            if !eligible.contains(observation.subject_artifact_hash.as_str())
                || !observed_subjects.insert(observation.subject_artifact_hash.as_str())
                || !observation_refs.insert(observation.observation_ref.as_str())
            {
                return Err("neurosemantic remediation observation set contains an invalid or duplicate observation".into());
            }
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8], aggregation_ref: &str) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation observation set JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let artifact: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic remediation observation set JSON: {error}"))?;
        artifact.validate(aggregation_ref)?;
        Ok(artifact)
    }

    pub fn fingerprint(&self, aggregation_ref: &str) -> Result<String, String> {
        self.validate(aggregation_ref)?;
        let mut canonical = self.clone();
        canonical.eligible_subject_artifact_hashes.sort();
        canonical.observations.sort_by(|left, right| {
            left.subject_artifact_hash
                .cmp(&right.subject_artifact_hash)
                .then_with(|| left.observation_ref.cmp(&right.observation_ref))
        });
        Ok(content_hash(&serde_json::to_vec(&canonical).map_err(|error| {
            format!("neurosemantic remediation observation set serialization: {error}")
        })?))
    }
}

/// Independently reproducible point-estimate computation over one observation set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationMetricComputationArtifact {
    pub schema_version: u16,
    pub computation_ref: String,
    pub metric_ref: String,
    pub kind: NeurosemanticRemediationMeasurementKind,
    pub metric_definition_hash: String,
    pub observation_set_hash: String,
    pub aggregation_ref: String,
    pub execution_revision: String,
    /// Fixed-point result: estimate_numerator / 10^estimate_scale.
    pub estimate_numerator: i64,
    pub estimate_scale: u32,
    pub eligible_sample_count: u64,
    pub observed_sample_count: u64,
    pub failure_count: u64,
}

impl NeurosemanticRemediationMetricComputationArtifact {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_METRIC_COMPUTATION_SCHEMA_VERSION
            || !valid_identifier(&self.computation_ref)
            || !valid_identifier(&self.metric_ref)
            || !valid_blake3_digest(&self.metric_definition_hash)
            || !valid_blake3_digest(&self.observation_set_hash)
            || !valid_identifier(&self.aggregation_ref)
            || !valid_execution_revision(&self.execution_revision)
            || self.estimate_scale > 12
        {
            return Err("neurosemantic remediation metric computation fields are invalid".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation metric computation JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let artifact: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic remediation metric computation JSON: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        Ok(content_hash(&serde_json::to_vec(self).map_err(|error| {
            format!("neurosemantic remediation metric computation serialization: {error}")
        })?))
    }
}

pub const NEUROSEMANTIC_REMEDIATION_MEASUREMENT_SCHEMA_VERSION: u16 = 3;
pub const NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION: u16 = 1;
pub const NEUROSEMANTIC_REMEDIATION_OBSERVATION_SET_SCHEMA_VERSION: u16 = 1;
pub const NEUROSEMANTIC_REMEDIATION_METRIC_COMPUTATION_SCHEMA_VERSION: u16 = 1;
pub const NEUROSEMANTIC_REMEDIATION_UNCERTAINTY_COMPUTATION_SCHEMA_VERSION: u16 = 3;
const MAX_NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITIONS: usize = 32;
const MAX_NEUROSEMANTIC_REMEDIATION_METRIC_TEXT_BYTES: usize = 256;
const MAX_NEUROSEMANTIC_REMEDIATION_OBSERVATIONS: usize = 4096;
const NEUROSEMANTIC_REMEDIATION_WILSON_95_METHOD_REF: &str = "wilson-score-95-v1";
const NEUROSEMANTIC_REMEDIATION_WILSON_95_ASSUMPTIONS_REF: &str =
    "independent-bernoulli-trials-v1";
const NEUROSEMANTIC_REMEDIATION_WILSON_95_ASSUMPTIONS_BYTES: &[u8] =
    b"independent Bernoulli trials; fixed binary outcome; no clustering correction declared";

fn validate_uncertainty_method_application(
    method_ref: &str,
    definition: &NeurosemanticRemediationMetricDefinition,
    design: &NeurosemanticRemediationStatisticalDesignArtifact,
) -> Result<(), String> {
    if method_ref != NEUROSEMANTIC_REMEDIATION_WILSON_95_METHOD_REF {
        return Ok(());
    }
    if definition.unit_ref != "proportion"
        || !matches!(
            definition.aggregation_ref.as_str(),
            "per-item-rate" | "attack-success-rate" | "probe-detection-rate"
        )
    {
        return Err(
            "neurosemantic remediation Wilson uncertainty method is only valid for single-proportion failure-rate metrics"
                .into(),
        );
    }
    if design.inference_scope != NeurosemanticRemediationUncertaintyInferenceScope::Superpopulation
        || design.sampling_design
            != NeurosemanticRemediationStatisticalSamplingDesign::ProbabilitySample
        || design.dependence_model
            != NeurosemanticRemediationStatisticalDependenceModel::IndependentObservationUnits
        || design.analysis_unit_ref != "subject-artifact"
        || design.outcome_model_ref != "binary-failure-indicator-v1"
    {
        return Err(
            "neurosemantic remediation Wilson uncertainty method requires a probability sample of independent subject-level binary outcomes"
                .into(),
        );
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationMeasurementArtifact {
    pub schema_version: u16,
    pub measurement_ref: String,
    pub metric_definitions: Vec<NeurosemanticRemediationMetricDefinition>,
    pub measurements: Vec<NeurosemanticRemediationMeasurement>,
    /// Exact point-estimate computation artifacts, one per measurement metric.
    pub metric_computation_artifact_hashes: Vec<String>,
    pub worst_case_disposition: NeurosemanticRemediationImpactDisposition,
}

impl NeurosemanticRemediationMeasurementArtifact {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_MEASUREMENT_SCHEMA_VERSION
            || !valid_identifier(&self.measurement_ref)
            || self.metric_definitions.is_empty()
            || self.metric_definitions.len() > MAX_NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITIONS
            || self.measurements.is_empty()
            || self.measurements.len() > 32
            || self.metric_computation_artifact_hashes.len() != self.measurements.len()
            || self.metric_computation_artifact_hashes.iter().any(|hash| !valid_blake3_digest(hash))
        {
            return Err("neurosemantic remediation measurement artifact fields are invalid".into());
        }

        let mut definition_refs = BTreeSet::new();
        for definition in &self.metric_definitions {
            if definition.schema_version != NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION
                || !valid_identifier(&definition.metric_ref)
                || !valid_identifier(&definition.estimand_ref)
                || definition.scope_ref.is_empty()
                || definition.scope_ref.len() > MAX_NEUROSEMANTIC_REMEDIATION_METRIC_TEXT_BYTES
                || !valid_identifier(&definition.unit_ref)
                || !valid_identifier(&definition.aggregation_ref)
                || !definition_refs.insert(definition.metric_ref.as_str())
            {
                return Err("neurosemantic remediation metric definition fields are invalid".into());
            }
        }

        let computation_hashes: BTreeSet<&str> = self
            .metric_computation_artifact_hashes
            .iter()
            .map(String::as_str)
            .collect();
        if computation_hashes.len() != self.metric_computation_artifact_hashes.len() {
            return Err("neurosemantic remediation measurement contains duplicate computation artifact hashes".into());
        }

        let mut measurement_refs = BTreeSet::new();
        let mut kinds = BTreeSet::new();
        for measurement in &self.measurements {
            if measurement.eligible_sample_count == 0
                || measurement.observed_sample_count > measurement.eligible_sample_count
                || measurement.failure_count > measurement.observed_sample_count
                || measurement.estimate_scale > 12
                || (measurement.observed_sample_count < measurement.eligible_sample_count
                    && measurement.status != NeurosemanticRemediationImpactDisposition::Inconclusive)
                || !measurement_refs.insert(measurement.metric_ref.as_str())
                || !kinds.insert(measurement.kind)
            {
                return Err("neurosemantic remediation measurement fields are invalid".into());
            }
            let definition = self.metric_definitions.iter().find(|definition| {
                definition.metric_ref == measurement.metric_ref
            }).ok_or_else(|| "neurosemantic remediation measurement references unknown metric definition".to_string())?;
            if definition.kind != measurement.kind {
                return Err("neurosemantic remediation measurement kind disagrees with metric definition".into());
            }
            match &measurement.uncertainty {
                NeurosemanticRemediationUncertainty::NotEstimated => {}
                NeurosemanticRemediationUncertainty::Interval {
                    lower_numerator,
                    upper_numerator,
                    scale,
                    confidence_level_bps,
                    uncertainty_method_ref,
                    uncertainty_computation_artifact_hash,
                } => {
                    if lower_numerator > upper_numerator
                        || scale > 12
                        || scale != measurement.estimate_scale
                        || measurement.estimate_numerator < lower_numerator
                        || measurement.estimate_numerator > upper_numerator
                        || confidence_level_bps == 0
                        || confidence_level_bps > 10_000
                        || !valid_identifier(&uncertainty_method_ref)
                        || !valid_blake3_digest(&uncertainty_computation_artifact_hash)
                    {
                        return Err("neurosemantic remediation measurement uncertainty is invalid".into());
                    }
                    if definition.unit_ref == "proportion" {
                        let scale_factor = 10_i64
                            .checked_pow(scale)
                            .ok_or_else(|| {
                                "neurosemantic remediation uncertainty proportion scale is out of range"
                                    .to_string()
                            })?;
                        if lower_numerator < 0 || upper_numerator > scale_factor {
                            return Err(
                                "neurosemantic remediation proportion uncertainty is outside [0,1]"
                                    .into(),
                            );
                        }
                    }
                    // The concrete uncertainty computation artifact is required to
                    // determine inference scope; method applicability is therefore checked
                    // during strong bundle verification.
                }
            }
        }
        if definition_refs
            .iter()
            .any(|metric_ref| !measurement_refs.contains(*metric_ref))
        {
            return Err("neurosemantic remediation measurement contains an unused metric definition".into());
        }
        for required in [
            NeurosemanticRemediationMeasurementKind::Forgetfulness,
            NeurosemanticRemediationMeasurementKind::UtilityImpact,
            NeurosemanticRemediationMeasurementKind::RecoveryRisk,
            NeurosemanticRemediationMeasurementKind::RepresentationResidual,
        ] {
            if !kinds.contains(&required) {
                return Err("neurosemantic remediation measurement omits a required dimension".into());
            }
        }
        let recomputed = self.recompute_worst_case_unchecked()?;
        if recomputed != self.worst_case_disposition {
            return Err("neurosemantic remediation worst-case disposition does not match dimensions".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation measurement artifact JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let artifact: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic remediation measurement artifact JSON: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    fn recompute_worst_case_unchecked(
        &self,
    ) -> Result<NeurosemanticRemediationImpactDisposition, String> {
        if self.measurements.is_empty() {
            return Err("neurosemantic remediation measurement set is empty".into());
        }
        if self
            .measurements
            .iter()
            .any(|measurement| measurement.status == NeurosemanticRemediationImpactDisposition::Inconclusive)
        {
            return Ok(NeurosemanticRemediationImpactDisposition::Inconclusive);
        }
        if self
            .measurements
            .iter()
            .any(|measurement| measurement.status == NeurosemanticRemediationImpactDisposition::OutsideDeclaredBounds)
        {
            return Ok(NeurosemanticRemediationImpactDisposition::OutsideDeclaredBounds);
        }
        Ok(NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds)
    }

    pub fn recomputed_worst_case_disposition(&self) -> Result<NeurosemanticRemediationImpactDisposition, String> {
        self.validate()?;
        self.recompute_worst_case_unchecked()
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        Ok(content_hash(
            &serde_json::to_vec(self).map_err(|error| {
                format!("neurosemantic remediation measurement serialization: {error}")
            })?,
        ))
    }
}

impl NeurosemanticRemediationMetricDefinition {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION
            || !valid_identifier(&self.metric_ref)
            || !valid_identifier(&self.estimand_ref)
            || self.scope_ref.is_empty()
            || self.scope_ref.len() > MAX_NEUROSEMANTIC_REMEDIATION_METRIC_TEXT_BYTES
            || !valid_identifier(&self.unit_ref)
            || !valid_identifier(&self.aggregation_ref)
        {
            return Err("neurosemantic remediation metric definition fields are invalid".into());
        }
        if matches!(
            self.aggregation_ref.as_str(),
            "per-item-rate" | "attack-success-rate" | "probe-detection-rate" | "worst-subgroup-gap"
        ) && self.direction != NeurosemanticRemediationMetricDirection::LowerIsBetter
        {
            return Err(
                "neurosemantic remediation failure-rate metrics must declare LowerIsBetter"
                    .into(),
            );
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        Ok(content_hash(&serde_json::to_vec(self).map_err(|error| {
            format!("neurosemantic remediation metric definition serialization: {error}")
        })?))
    }
}

impl NeurosemanticRemediationEvaluationEnvironment {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_EVALUATION_ENVIRONMENT_SCHEMA_VERSION
            || !valid_identifier(&self.environment_ref)
            || !valid_identifier(&self.platform_ref)
            || !valid_identifier(&self.runtime_ref)
            || !valid_identifier(&self.toolchain_ref)
            || !valid_blake3_digest(&self.dependency_lock_hash)
            || !valid_blake3_digest(&self.configuration_hash)
            || !valid_execution_revision(&self.execution_revision)
        {
            return Err("neurosemantic remediation evaluation environment fields are invalid".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation evaluation environment JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let environment: Self = serde_json::from_slice(bytes).map_err(|error| {
            format!("neurosemantic remediation evaluation environment JSON: {error}")
        })?;
        environment.validate()?;
        Ok(environment)
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        Ok(content_hash(&serde_json::to_vec(self).map_err(|error| {
            format!("neurosemantic remediation evaluation environment serialization: {error}")
        })?))
    }
}


/// Canonical identity for the frozen remediation evaluation design.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationEvaluationManifest {
    pub schema_version: u16,
    pub evaluation_ref: String,
    pub source_dataset_manifest_hash: String,
    pub forget_set_manifest_hash: String,
    pub retain_set_manifest_hash: String,
    pub study_protocol_hash: String,
    pub evaluation_split_manifest_hash: String,
    pub recovery_method_hash: String,
    pub representation_probe_method_hash: String,
}

impl NeurosemanticRemediationEvaluationManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_EVALUATION_MANIFEST_SCHEMA_VERSION
            || !valid_identifier(&self.evaluation_ref)
            || !valid_blake3_digest(&self.source_dataset_manifest_hash)
            || !valid_blake3_digest(&self.forget_set_manifest_hash)
            || !valid_blake3_digest(&self.retain_set_manifest_hash)
            || !valid_blake3_digest(&self.study_protocol_hash)
            || !valid_blake3_digest(&self.evaluation_split_manifest_hash)
            || !valid_blake3_digest(&self.recovery_method_hash)
            || !valid_blake3_digest(&self.representation_probe_method_hash)
            || self.forget_set_manifest_hash == self.retain_set_manifest_hash
            || self.recovery_method_hash == self.representation_probe_method_hash
        {
            return Err("neurosemantic remediation evaluation manifest fields are invalid".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation evaluation manifest JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic remediation evaluation manifest JSON: {error}"))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        Ok(content_hash(&serde_json::to_vec(self).map_err(|error| {
            format!("neurosemantic remediation evaluation manifest serialization: {error}")
        })?))
    }
}

impl NeurosemanticRemediationEvaluationMethod {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_EVALUATION_METHOD_SCHEMA_VERSION
            || !valid_identifier(&self.method_ref)
            || self.method_ref.len() > MAX_NEUROSEMANTIC_REMEDIATION_METHOD_REF_BYTES
            || !valid_blake3_digest(&self.protocol_hash)
            || !valid_execution_revision(&self.implementation_revision)
        {
            return Err("neurosemantic remediation evaluation method fields are invalid".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation evaluation method JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let method: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic remediation evaluation method JSON: {error}"))?;
        method.validate()?;
        Ok(method)
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        Ok(content_hash(
            &serde_json::to_vec(self)
                .map_err(|error| format!("neurosemantic remediation evaluation method serialization: {error}"))?,
        ))
    }
}

/// Content-addressed evidence that a remediation was evaluated for its declared impact dimensions.
/// This is an evidence binding, not a theorem that the model forgot information or is safe.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticRemediationImpactArtifact {
    pub schema_version: u16,
    pub impact_ref: String,
    pub pre_remediation_model_hash: String,
    pub post_remediation_model_hash: String,
    pub pre_remediation_lineage_ref: String,
    pub pre_remediation_lineage_hash: String,
    pub post_remediation_lineage_ref: String,
    pub post_remediation_lineage_hash: String,
    /// Exact fingerprint of the lifecycle receipt whose remediation effect is being evaluated.
    pub lifecycle_receipt_hash: String,
    /// Canonical identity of the full frozen evaluation design.
    pub evaluation_manifest_hash: String,
    /// Content-addressed structural measurement record for all required evaluation dimensions.
    pub measurement_artifact_hash: String,
    /// Identity of the agent that conducted the remediation impact evaluation.
    pub evaluation_agent_ref: String,
    /// Identity of the independent agent that verified the evaluation result.
    pub evaluation_verifier_ref: String,
    /// Exact evidence bytes supporting independent evaluation verification.
    pub evaluation_verification_evidence_hash: String,
    /// Content-addressed evaluation runtime/environment descriptor.
    pub evaluation_environment_hash: String,
    pub remediation_action: NeurosemanticArtifactLifecycleAction,
    pub study_protocol_hash: String,
    pub evaluation_split_manifest_hash: String,
    /// Frozen forget/retain population identities used by the evaluation.
    pub forget_set_manifest_hash: String,
    pub retain_set_manifest_hash: String,
    /// Content-addressed identity of the recovery-attack methodology.
    pub recovery_method_hash: String,
    /// Content-addressed identity of the representation-level residual probe methodology.
    pub representation_probe_method_hash: String,
    /// Evidence over the intended forget/removal target.
    pub forget_evidence_hash: String,
    /// Evidence over retained-task utility/behavior.
    pub utility_impact_evidence_hash: String,
    /// Evidence produced by recovery-attack evaluation.
    pub recovery_evidence_hash: String,
    /// Evidence produced by representation-level residual probing.
    pub representation_residual_evidence_hash: String,
    /// Optional subgroup/fairness impact evidence; absence means it was not evaluated.
    pub fairness_impact_evidence_hash: Option<String>,
    /// Evidence over recovery or residual-influence checks.
    pub residual_risk_evidence_hash: String,
    /// Exact execution revision producing the impact artifact.
    pub execution_revision: String,
    pub observed_at_unix_s: u64,
    /// Declared interpretation of the evaluation result.
    pub disposition: NeurosemanticRemediationImpactDisposition,
    /// Explicit names of impact dimensions covered by the artifact.
    pub dimensions: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticRemediationImpactDisposition {
    WithinDeclaredBounds,
    OutsideDeclaredBounds,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticRemediationImpactEvidenceKind {
    Forgetfulness,
    UtilityImpact,
    FairnessImpact,
    ResidualRisk,
    RecoveryRisk,
    RepresentationResidual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticRemediationImpactLineageSide {
    PreRemediation,
    PostRemediation,
}

impl NeurosemanticRemediationImpactArtifact {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_REMEDIATION_IMPACT_ARTIFACT_SCHEMA_VERSION
            || !valid_identifier(&self.impact_ref)
            || !valid_blake3_digest(&self.pre_remediation_model_hash)
            || !valid_blake3_digest(&self.post_remediation_model_hash)
            || self.pre_remediation_model_hash == self.post_remediation_model_hash
            || !valid_identifier(&self.pre_remediation_lineage_ref)
            || !valid_blake3_digest(&self.pre_remediation_lineage_hash)
            || !valid_identifier(&self.post_remediation_lineage_ref)
            || !valid_blake3_digest(&self.post_remediation_lineage_hash)
            || !valid_blake3_digest(&self.lifecycle_receipt_hash)
            || !valid_blake3_digest(&self.evaluation_manifest_hash)
            || !valid_blake3_digest(&self.measurement_artifact_hash)
            || !valid_identifier(&self.evaluation_agent_ref)
            || !valid_identifier(&self.evaluation_verifier_ref)
            || self.evaluation_agent_ref == self.evaluation_verifier_ref
            || !valid_blake3_digest(&self.evaluation_verification_evidence_hash)
            || !valid_blake3_digest(&self.evaluation_environment_hash)
            || self.pre_remediation_lineage_ref == self.post_remediation_lineage_ref
            || self.pre_remediation_lineage_hash == self.post_remediation_lineage_hash
            || !valid_blake3_digest(&self.study_protocol_hash)
            || !valid_blake3_digest(&self.evaluation_split_manifest_hash)
            || !valid_blake3_digest(&self.forget_set_manifest_hash)
            || !valid_blake3_digest(&self.retain_set_manifest_hash)
            || !valid_blake3_digest(&self.recovery_method_hash)
            || !valid_blake3_digest(&self.representation_probe_method_hash)
            || !valid_blake3_digest(&self.forget_evidence_hash)
            || !valid_blake3_digest(&self.utility_impact_evidence_hash)
            || !valid_blake3_digest(&self.recovery_evidence_hash)
            || !valid_blake3_digest(&self.representation_residual_evidence_hash)
            || self.fairness_impact_evidence_hash.as_ref().is_some_and(|hash| !valid_blake3_digest(hash))
            || !valid_blake3_digest(&self.residual_risk_evidence_hash)
            || !valid_execution_revision(&self.execution_revision)
            || self.dimensions.is_empty()
            || self.dimensions.len() > MAX_NEUROSEMANTIC_IMPACT_DIMENSION_REFS
            || self.dimensions.iter().any(|dimension| !valid_identifier(dimension))
        {
            return Err("neurosemantic remediation impact artifact fields are invalid".into());
        }
        let unique_dimensions: BTreeSet<&str> = self.dimensions.iter().map(String::as_str).collect();
        if unique_dimensions.len() != self.dimensions.len() {
            return Err("neurosemantic remediation impact artifact contains duplicate dimensions".into());
        }
        for required in ["forgetfulness", "utility-impact", "residual-risk", "recovery-attack", "representation-residual", "forget-set", "retain-set"] {
            if !unique_dimensions.contains(required) {
                return Err(format!("neurosemantic remediation impact artifact omits required dimension: {required}"));
            }
        }
        let fairness_declared = unique_dimensions.contains("fairness-impact");
        if fairness_declared != self.fairness_impact_evidence_hash.is_some() {
            return Err("neurosemantic remediation fairness dimension and evidence must be declared together".into());
        }
        match self.remediation_action {
            NeurosemanticArtifactLifecycleAction::Rectification
            | NeurosemanticArtifactLifecycleAction::Supersession
            | NeurosemanticArtifactLifecycleAction::Erasure => {}
            NeurosemanticArtifactLifecycleAction::AccessRevocation
            | NeurosemanticArtifactLifecycleAction::Retention => {
                return Err("neurosemantic remediation impact artifact requires model-affecting remediation".into());
            }
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic remediation impact artifact JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let artifact: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic remediation impact artifact JSON: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn verify_evidence_bytes(
        &self,
        kind: NeurosemanticRemediationImpactEvidenceKind,
        evidence_bytes: &[u8],
    ) -> Result<(), String> {
        self.validate()?;
        if evidence_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic remediation impact evidence exceeds the serialized artifact limit".into());
        }
        let expected_hash = match kind {
            NeurosemanticRemediationImpactEvidenceKind::Forgetfulness => &self.forget_evidence_hash,
            NeurosemanticRemediationImpactEvidenceKind::UtilityImpact => &self.utility_impact_evidence_hash,
            NeurosemanticRemediationImpactEvidenceKind::FairnessImpact => self
                .fairness_impact_evidence_hash
                .as_ref()
                .ok_or_else(|| "neurosemantic remediation fairness evidence is not declared".to_string())?,
            NeurosemanticRemediationImpactEvidenceKind::ResidualRisk => &self.residual_risk_evidence_hash,
            NeurosemanticRemediationImpactEvidenceKind::RecoveryRisk => &self.recovery_evidence_hash,
            NeurosemanticRemediationImpactEvidenceKind::RepresentationResidual =>
                &self.representation_residual_evidence_hash,
        };
        if content_hash(evidence_bytes) != *expected_hash {
            return Err("neurosemantic remediation impact evidence hash mismatch".into());
        }
        Ok(())
    }

    pub fn verify_lifecycle_binding(&self, lifecycle_receipt: &NeurosemanticArtifactLifecycleReceipt) -> Result<(), String> {
        self.validate()?;
        lifecycle_receipt.validate()?;
        if lifecycle_receipt.state != NeurosemanticArtifactLifecycleState::IndependentlyVerified {
            return Err("neurosemantic remediation impact evidence requires an independently-verified lifecycle receipt".into());
        }
        if lifecycle_receipt.action != self.remediation_action {
            return Err("neurosemantic remediation impact action does not match the lifecycle receipt action".into());
        }
        if Some(self.evaluation_agent_ref.as_str()) == lifecycle_receipt.effect_agent_ref.as_deref()
            || Some(self.evaluation_verifier_ref.as_str()) == lifecycle_receipt.effect_agent_ref.as_deref()
            || Some(self.evaluation_agent_ref.as_str())
                == lifecycle_receipt.verification_agent_ref.as_deref()
            || Some(self.evaluation_verifier_ref.as_str())
                == lifecycle_receipt.verification_agent_ref.as_deref()
        {
            return Err(
                "neurosemantic remediation evaluation roles must be independent from lifecycle roles"
                    .into(),
            );
        }
        if lifecycle_receipt.execution_revision != self.execution_revision {
            return Err("neurosemantic remediation impact execution revision does not match the lifecycle receipt".into());
        }
        let fingerprint = lifecycle_receipt.fingerprint()?;
        if fingerprint != self.lifecycle_receipt_hash {
            return Err("neurosemantic remediation impact artifact is bound to a different lifecycle receipt".into());
        }
        Ok(())
    }

    pub fn verify_measurement_artifact_bytes(
        &self,
        measurement_bytes: &[u8],
    ) -> Result<NeurosemanticRemediationMeasurementArtifact, String> {
        self.validate()?;
        let measurement = NeurosemanticRemediationMeasurementArtifact::from_json_bytes(measurement_bytes)?;
        if measurement.fingerprint()? != self.measurement_artifact_hash {
            return Err("neurosemantic remediation measurement artifact hash mismatch".into());
        }
        if measurement.recomputed_worst_case_disposition()? != self.disposition {
            return Err(
                "neurosemantic remediation impact disposition does not match measurement worst case"
                    .into(),
            );
        }
        for declared in &self.dimensions {
            let required_kind = match declared.as_str() {
                "forgetfulness" => Some(NeurosemanticRemediationMeasurementKind::Forgetfulness),
                "utility-impact" => Some(NeurosemanticRemediationMeasurementKind::UtilityImpact),
                "fairness-impact" => Some(NeurosemanticRemediationMeasurementKind::FairnessImpact),
                "recovery-attack" => Some(NeurosemanticRemediationMeasurementKind::RecoveryRisk),
                "representation-residual" => Some(NeurosemanticRemediationMeasurementKind::RepresentationResidual),
                _ => None,
            };
            if let Some(required_kind) = required_kind
                && !measurement.measurements.iter().any(|item| item.kind == required_kind)
            {
                return Err("neurosemantic remediation measurement artifact omits a declared impact dimension".into());
            }
        }
        Ok(measurement)
    }

    /// Verify every reported point estimate against explicit, content-addressed observations.
    ///
    /// The caller supplies the exact observation-set bytes; no ambient repository/global
    /// resolver is consulted.
    pub fn verify_measurement_computation_bundle_bytes(
        &self,
        measurement_bytes: &[u8],
        computation_bytes: &[&[u8]],
        observation_set_bytes: &[&[u8]],
        population_manifest_bytes: &[&[u8]],
        uncertainty_computation_bytes: &[&[u8]],
        uncertainty_assumption_bytes: &[&[u8]],
        statistical_design_bytes: &[&[u8]],
    ) -> Result<(), String> {
        self.validate()?;
        let measurement = self.verify_measurement_artifact_bytes(measurement_bytes)?;
        if computation_bytes.len() != measurement.metric_computation_artifact_hashes.len() {
            return Err("neurosemantic remediation computation bundle cardinality mismatch".into());
        }

        let mut seen_computation_hashes = BTreeSet::new();
        let mut seen_metric_refs = BTreeSet::new();

        for bytes in computation_bytes {
            let computation = NeurosemanticRemediationMetricComputationArtifact::from_json_bytes(bytes)?;
            let computation_hash = computation.fingerprint()?;
            if !measurement
                .metric_computation_artifact_hashes
                .iter()
                .any(|expected| expected == &computation_hash)
                || !seen_computation_hashes.insert(computation_hash)
                || !seen_metric_refs.insert(computation.metric_ref.clone())
            {
                return Err("neurosemantic remediation computation bundle contains an unexpected or duplicate computation".into());
            }

            let definition = measurement
                .metric_definitions
                .iter()
                .find(|definition| definition.metric_ref == computation.metric_ref)
                .ok_or_else(|| "neurosemantic remediation computation references an unknown metric".to_string())?;
            if definition.kind != computation.kind
                || definition.aggregation_ref != computation.aggregation_ref
                || definition.fingerprint()? != computation.metric_definition_hash
            {
                return Err("neurosemantic remediation metric computation definition binding mismatch".into());
            }

            let observation_set_bytes = observation_set_bytes
                .iter()
                .copied()
                .find(|candidate| content_hash(candidate) == computation.observation_set_hash)
                .ok_or_else(|| "neurosemantic remediation computation observation set is missing".to_string())?;
            let observation_set = NeurosemanticRemediationObservationSetArtifact::from_json_bytes(
                observation_set_bytes,
                computation.aggregation_ref.as_str(),
            )?;
            if observation_set.metric_ref != computation.metric_ref
                || observation_set.kind != computation.kind
                || observation_set.fingerprint(computation.aggregation_ref.as_str())? != computation.observation_set_hash
                || observation_set.scope_ref != definition.scope_ref
            {
                return Err("neurosemantic remediation observation set binding mismatch".into());
            }

            let population_bytes = population_manifest_bytes
                .iter()
                .copied()
                .find(|candidate| content_hash(candidate) == observation_set.population_manifest_hash)
                .ok_or_else(|| "neurosemantic remediation population manifest is missing".to_string())?;

            let expected_population_hash = match computation.kind {
                NeurosemanticRemediationMeasurementKind::Forgetfulness
                | NeurosemanticRemediationMeasurementKind::RecoveryRisk
                | NeurosemanticRemediationMeasurementKind::RepresentationResidual => {
                    &self.forget_set_manifest_hash
                }
                NeurosemanticRemediationMeasurementKind::UtilityImpact => &self.retain_set_manifest_hash,
                NeurosemanticRemediationMeasurementKind::FairnessImpact => &self.evaluation_split_manifest_hash,
            };
            if observation_set.population_manifest_hash != *expected_population_hash {
                return Err(
                    "neurosemantic remediation observation set population does not match the canonical impact population"
                        .into(),
                );
            }

            match computation.kind {
                NeurosemanticRemediationMeasurementKind::Forgetfulness
                | NeurosemanticRemediationMeasurementKind::RecoveryRisk
                | NeurosemanticRemediationMeasurementKind::RepresentationResidual => {
                    let population =
                        NeurosemanticRemediationEvaluationSetManifest::from_json_bytes(population_bytes)?;
                    if population.set_kind != NeurosemanticRemediationEvaluationSetKind::Forget
                        || population.fingerprint()? != observation_set.population_manifest_hash
                        || sorted_hashes(&population.member_artifact_hashes)
                            != sorted_hashes(&observation_set.eligible_subject_artifact_hashes)
                    {
                        return Err("neurosemantic remediation observation set is not exactly bound to the forget population".into());
                    }
                }
                NeurosemanticRemediationMeasurementKind::UtilityImpact => {
                    let population =
                        NeurosemanticRemediationEvaluationSetManifest::from_json_bytes(population_bytes)?;
                    if population.set_kind != NeurosemanticRemediationEvaluationSetKind::Retain
                        || population.fingerprint()? != observation_set.population_manifest_hash
                        || sorted_hashes(&population.member_artifact_hashes)
                            != sorted_hashes(&observation_set.eligible_subject_artifact_hashes)
                    {
                        return Err("neurosemantic remediation observation set is not exactly bound to the retain population".into());
                    }
                }
                NeurosemanticRemediationMeasurementKind::FairnessImpact => {
                    let split =
                        NeurosemanticRemediationEvaluationSplitManifest::from_json_bytes(population_bytes)?;
                    if split.fingerprint()? != observation_set.population_manifest_hash
                        || sorted_hashes(
                            &split
                                .members
                                .iter()
                                .map(|member| member.subject_artifact_hash.clone())
                                .collect::<Vec<_>>(),
                        ) != sorted_hashes(&observation_set.eligible_subject_artifact_hashes)
                    {
                        return Err(
                            "neurosemantic remediation observation set is not exactly bound to the fairness split population"
                                .into(),
                        );
                    }
                    let observed_groups: BTreeMap<&str, &str> = observation_set
                        .observations
                        .iter()
                        .filter_map(|observation| {
                            observation
                                .group_ref
                                .as_deref()
                                .map(|group| (observation.subject_artifact_hash.as_str(), group))
                        })
                        .collect();
                    let expected_groups: BTreeMap<&str, &str> = split
                        .members
                        .iter()
                        .map(|member| {
                            (
                                member.subject_artifact_hash.as_str(),
                                member.group_ref.as_str(),
                            )
                        })
                        .collect();
                    if observed_groups
                        .iter()
                        .any(|(subject, group)| expected_groups.get(subject) != Some(group))
                    {
                        return Err(
                            "neurosemantic remediation fairness observation group binding mismatch"
                                .into(),
                        );
                    }
                }
            }

            let (eligible_count, observed_count, failure_count, ratio_numerator, ratio_denominator) =
                recompute_metric_ratio(&observation_set, computation.aggregation_ref.as_str())?;
            if computation.execution_revision != self.execution_revision
                || computation.eligible_sample_count != eligible_count
                || computation.observed_sample_count != observed_count
                || computation.failure_count != failure_count
                || !fixed_point_equals_ratio(
                    computation.estimate_numerator,
                    computation.estimate_scale,
                    ratio_numerator,
                    ratio_denominator,
                )
            {
                return Err("neurosemantic remediation metric computation does not reproduce its observations".into());
            }

            let measurement_item = measurement
                .measurements
                .iter()
                .find(|item| item.metric_ref == computation.metric_ref)
                .ok_or_else(|| "neurosemantic remediation computation lacks a measurement".to_string())?;

            if let NeurosemanticRemediationUncertainty::Interval {
                lower_numerator,
                upper_numerator,
                scale,
                confidence_level_bps,
                uncertainty_method_ref,
                uncertainty_computation_artifact_hash,
            } = &measurement_item.uncertainty
            {
                let uncertainty_bytes = uncertainty_computation_bytes
                    .iter()
                    .copied()
                    .find(|candidate| content_hash(candidate) == *uncertainty_computation_artifact_hash)
                    .ok_or_else(|| "neurosemantic remediation uncertainty computation is missing".to_string())?;
                let uncertainty =
                    NeurosemanticRemediationUncertaintyComputationArtifact::from_json_bytes(uncertainty_bytes)?;
                let assumptions_bytes = uncertainty_assumption_bytes
                    .iter()
                    .copied()
                    .find(|candidate| content_hash(candidate) == uncertainty.assumptions_hash)
                    .ok_or_else(|| "neurosemantic remediation uncertainty assumptions are missing".to_string())?;
                if content_hash(assumptions_bytes) != uncertainty.assumptions_hash
                    || uncertainty.fingerprint()? != *uncertainty_computation_artifact_hash
                    || uncertainty.metric_ref != computation.metric_ref
                    || uncertainty.metric_definition_hash != computation.metric_definition_hash
                    || uncertainty.observation_set_hash != computation.observation_set_hash
                    || uncertainty.point_estimate_numerator != computation.estimate_numerator
                    || uncertainty.point_estimate_scale != computation.estimate_scale
                    || uncertainty.lower_numerator != *lower_numerator
                    || uncertainty.upper_numerator != *upper_numerator
                    || uncertainty.scale != *scale
                    || uncertainty.confidence_level_bps != *confidence_level_bps
                    || uncertainty.method_ref != *uncertainty_method_ref
                    || uncertainty.execution_revision != self.execution_revision
                {
                    return Err("neurosemantic remediation uncertainty computation binding mismatch".into());
                }

                let design_bytes = statistical_design_bytes
                    .iter()
                    .copied()
                    .find(|candidate| content_hash(candidate) == uncertainty.statistical_design_hash)
                    .ok_or_else(|| "neurosemantic remediation statistical design is missing".to_string())?;
                let design =
                    NeurosemanticRemediationStatisticalDesignArtifact::from_json_bytes(design_bytes)?;
                if design.fingerprint()? != uncertainty.statistical_design_hash
                    || design.metric_ref != computation.metric_ref
                    || design.metric_definition_hash != computation.metric_definition_hash
                    || design.observation_set_hash != computation.observation_set_hash
                    || design.inference_scope != uncertainty.inference_scope
                    || design.assumptions_hash != uncertainty.assumptions_hash
                    || design.study_protocol_hash != self.study_protocol_hash
                    || design.execution_revision != self.execution_revision
                {
                    return Err("neurosemantic remediation statistical design binding mismatch".into());
                }

                if uncertainty.method_ref == NEUROSEMANTIC_REMEDIATION_WILSON_95_METHOD_REF {
                    validate_uncertainty_method_application(
                        uncertainty.method_ref.as_str(),
                        definition,
                        &design,
                    )?;
                    if uncertainty.inference_scope
                            != NeurosemanticRemediationUncertaintyInferenceScope::Superpopulation
                        || uncertainty.confidence_level_bps != 9_500
                        || uncertainty.assumptions_ref
                            != NEUROSEMANTIC_REMEDIATION_WILSON_95_ASSUMPTIONS_REF
                        || uncertainty.assumptions_hash
                            != content_hash(NEUROSEMANTIC_REMEDIATION_WILSON_95_ASSUMPTIONS_BYTES)
                    {
                        return Err(
                            "neurosemantic remediation Wilson uncertainty assumptions are invalid"
                                .into(),
                        );
                    }
                    let (expected_lower, expected_upper, expected_scale) =
                        recompute_wilson_score_95_interval(
                            computation.failure_count,
                            computation.observed_sample_count,
                            uncertainty.scale,
                        )?;
                    if uncertainty.lower_numerator != expected_lower
                        || uncertainty.upper_numerator != expected_upper
                        || uncertainty.scale != expected_scale
                    {
                        return Err(
                            "neurosemantic remediation Wilson uncertainty calculation mismatch"
                                .into(),
                        );
                    }
                }
            }

            if measurement_item.kind != computation.kind
                || measurement_item.eligible_sample_count != computation.eligible_sample_count
                || measurement_item.observed_sample_count != computation.observed_sample_count
                || measurement_item.failure_count != computation.failure_count
                || measurement_item.estimate_numerator != computation.estimate_numerator
                || measurement_item.estimate_scale != computation.estimate_scale
            {
                return Err("neurosemantic remediation measurement does not match independent computation".into());
            }
        }

        if seen_computation_hashes.len() != measurement.metric_computation_artifact_hashes.len()
            || seen_metric_refs.len() != measurement.measurements.len()
        {
            return Err("neurosemantic remediation computation bundle is incomplete".into());
        }
        Ok(())
    }

    pub fn verify_evaluation_verification_evidence_bytes(
        &self,
        verifier_agent_ref: &str,
        evidence_bytes: &[u8],
    ) -> Result<(), String> {
        self.validate()?;
        if verifier_agent_ref != self.evaluation_verifier_ref {
            return Err("neurosemantic remediation evaluation verifier identity mismatch".into());
        }
        if verifier_agent_ref == self.evaluation_agent_ref {
            return Err("neurosemantic remediation evaluation verifier must differ from evaluator".into());
        }
        if evidence_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic remediation evaluation verification evidence exceeds the serialized artifact limit".into());
        }
        if content_hash(evidence_bytes) != self.evaluation_verification_evidence_hash {
            return Err("neurosemantic remediation evaluation verification evidence hash mismatch".into());
        }
        Ok(())
    }

    pub fn verify_evaluation_environment_bytes(
        &self,
        environment_bytes: &[u8],
    ) -> Result<(), String> {
        self.validate()?;
        let environment =
            NeurosemanticRemediationEvaluationEnvironment::from_json_bytes(environment_bytes)?;
        if environment.execution_revision != self.execution_revision {
            return Err(
                "neurosemantic remediation evaluation environment execution revision does not match the impact artifact"
                    .into(),
            );
        }
        if environment.fingerprint()? != self.evaluation_environment_hash {
            return Err("neurosemantic remediation evaluation environment hash mismatch".into());
        }
        Ok(())
    }

    pub fn verify_evaluation_manifest_bytes(&self, manifest_bytes: &[u8]) -> Result<(), String> {
        self.validate()?;
        let manifest = NeurosemanticRemediationEvaluationManifest::from_json_bytes(manifest_bytes)?;
        if manifest.forget_set_manifest_hash != self.forget_set_manifest_hash
            || manifest.retain_set_manifest_hash != self.retain_set_manifest_hash
            || manifest.study_protocol_hash != self.study_protocol_hash
            || manifest.evaluation_split_manifest_hash != self.evaluation_split_manifest_hash
            || manifest.recovery_method_hash != self.recovery_method_hash
            || manifest.representation_probe_method_hash != self.representation_probe_method_hash
            || manifest.fingerprint()? != self.evaluation_manifest_hash
        {
            return Err("neurosemantic remediation evaluation manifest binding mismatch".into());
        }
        Ok(())
    }

    pub fn verify_evaluation_set_manifest_bytes(
        &self,
        kind: NeurosemanticRemediationEvaluationSetKind,
        manifest_bytes: &[u8],
    ) -> Result<(), String> {
        self.validate()?;
        let manifest = NeurosemanticRemediationEvaluationSetManifest::from_json_bytes(manifest_bytes)?;
        if manifest.set_kind != kind {
            return Err("neurosemantic remediation evaluation set kind mismatch".into());
        }
        let expected_hash = match kind {
            NeurosemanticRemediationEvaluationSetKind::Forget => &self.forget_set_manifest_hash,
            NeurosemanticRemediationEvaluationSetKind::Retain => &self.retain_set_manifest_hash,
        };
        if manifest.fingerprint()? != *expected_hash {
            return Err("neurosemantic remediation evaluation set manifest hash mismatch".into());
        }
        Ok(())
    }

    pub fn verify_evaluation_method_bytes(
        &self,
        kind: NeurosemanticRemediationEvaluationMethodKind,
        method_bytes: &[u8],
    ) -> Result<(), String> {
        self.validate()?;
        let method = NeurosemanticRemediationEvaluationMethod::from_json_bytes(method_bytes)?;
        if method.kind != kind {
            return Err("neurosemantic remediation evaluation method kind mismatch".into());
        }
        if method.protocol_hash != self.study_protocol_hash {
            return Err("neurosemantic remediation evaluation method is bound to a different study protocol".into());
        }
        if method.implementation_revision != self.execution_revision {
            return Err("neurosemantic remediation evaluation method execution revision does not match the impact artifact".into());
        }
        let expected_hash = match kind {
            NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack =>
                &self.recovery_method_hash,
            NeurosemanticRemediationEvaluationMethodKind::RepresentationResidualProbe =>
                &self.representation_probe_method_hash,
        };
        if method.fingerprint()? != *expected_hash {
            return Err("neurosemantic remediation evaluation method hash mismatch".into());
        }
        Ok(())
    }

    pub fn verify_evaluation_bundle_identity(
        &self,
        evaluation_manifest_bytes: &[u8],
        forget_manifest_bytes: &[u8],
        retain_manifest_bytes: &[u8],
    ) -> Result<(), String> {
        self.validate()?;
        let manifest = NeurosemanticRemediationEvaluationManifest::from_json_bytes(evaluation_manifest_bytes)?;
        if manifest.fingerprint()? != self.evaluation_manifest_hash {
            return Err("neurosemantic remediation evaluation manifest fingerprint mismatch".into());
        }
        let forget = NeurosemanticRemediationEvaluationSetManifest::from_json_bytes(forget_manifest_bytes)?;
        let retain = NeurosemanticRemediationEvaluationSetManifest::from_json_bytes(retain_manifest_bytes)?;
        self.verify_evaluation_set_pair_bytes(forget_manifest_bytes, retain_manifest_bytes)?;
        if manifest.source_dataset_manifest_hash != forget.source_dataset_manifest_hash
            || manifest.source_dataset_manifest_hash != retain.source_dataset_manifest_hash
        {
            return Err("neurosemantic remediation evaluation manifest source dataset does not match evaluation sets".into());
        }
        Ok(())
    }
    pub fn verify_evaluation_set_pair_bytes(
        &self,
        forget_manifest_bytes: &[u8],
        retain_manifest_bytes: &[u8],
    ) -> Result<(), String> {
        self.validate()?;
        let forget = NeurosemanticRemediationEvaluationSetManifest::from_json_bytes(forget_manifest_bytes)?;
        let retain = NeurosemanticRemediationEvaluationSetManifest::from_json_bytes(retain_manifest_bytes)?;
        if forget.set_kind != NeurosemanticRemediationEvaluationSetKind::Forget
            || retain.set_kind != NeurosemanticRemediationEvaluationSetKind::Retain
        {
            return Err("neurosemantic remediation evaluation set roles are invalid".into());
        }
        if forget.source_dataset_manifest_hash != retain.source_dataset_manifest_hash {
            return Err("neurosemantic remediation forget/retain sets use different source dataset manifests".into());
        }
        let forget_members: BTreeSet<&str> =
            forget.member_artifact_hashes.iter().map(String::as_str).collect();
        if retain.member_artifact_hashes.iter().any(|hash| forget_members.contains(hash.as_str())) {
            return Err("neurosemantic remediation forget and retain sets overlap".into());
        }
        self.verify_evaluation_set_manifest_bytes(
            NeurosemanticRemediationEvaluationSetKind::Forget,
            forget_manifest_bytes,
        )?;
        self.verify_evaluation_set_manifest_bytes(
            NeurosemanticRemediationEvaluationSetKind::Retain,
            retain_manifest_bytes,
        )?;
        Ok(())
    }
    pub fn verify_study_protocol_bytes(&self, protocol_bytes: &[u8]) -> Result<(), String> {
        self.validate()?;
        if protocol_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic remediation study protocol exceeds the serialized artifact limit".into());
        }
        if content_hash(protocol_bytes) != self.study_protocol_hash {
            return Err("neurosemantic remediation study protocol hash mismatch".into());
        }
        Ok(())
    }

    pub fn verify_evaluation_split_manifest_bytes(
        &self,
        split_manifest_bytes: &[u8],
    ) -> Result<(), String> {
        self.validate()?;
        if split_manifest_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic remediation split manifest exceeds the serialized artifact limit".into());
        }
        if content_hash(split_manifest_bytes) != self.evaluation_split_manifest_hash {
            return Err("neurosemantic remediation split manifest hash mismatch".into());
        }
        Ok(())
    }

    pub fn verify_lineage_bytes(
        &self,
        side: NeurosemanticRemediationImpactLineageSide,
        lineage_record_bytes: &[u8],
    ) -> Result<NeurosemanticDerivationLineageRecord, String> {
        self.validate()?;
        if lineage_record_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic remediation lineage evidence exceeds the serialized artifact limit".into());
        }
        let record = NeurosemanticDerivationLineageRecord::from_json_bytes(lineage_record_bytes)?;
        let (expected_ref, expected_hash, expected_model_hash) = match side {
            NeurosemanticRemediationImpactLineageSide::PreRemediation => (
                &self.pre_remediation_lineage_ref,
                &self.pre_remediation_lineage_hash,
                &self.pre_remediation_model_hash,
            ),
            NeurosemanticRemediationImpactLineageSide::PostRemediation => (
                &self.post_remediation_lineage_ref,
                &self.post_remediation_lineage_hash,
                &self.post_remediation_model_hash,
            ),
        };
        if record.lineage_ref != *expected_ref
            || record.output_artifact_hash != *expected_model_hash
            || record.execution_revision != self.execution_revision
            || compute_derivation_provenance_hash(&record.lineage_ref, lineage_record_bytes)
                != *expected_hash
        {
            return Err("neurosemantic remediation lineage binding mismatch".into());
        }
        Ok(record)
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("neurosemantic remediation impact artifact serialization: {error}"))?;
        Ok(content_hash(&bytes))
    }
}

/// Machine-readable enumeration of the exact artifact identities independently inspected.
/// The record is content-addressed separately from the lifecycle receipt so coverage scope
/// cannot be reduced to an opaque prose assertion.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticArtifactLifecycleVerificationTargetSet {
    pub schema_version: u16,
    pub scope_ref: String,
    pub root_artifact_hash: String,
    pub target_artifact_hashes: Vec<String>,
}

impl NeurosemanticArtifactLifecycleVerificationTargetSet {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_ARTIFACT_LIFECYCLE_VERIFICATION_SCOPE_SCHEMA_VERSION
            || !valid_identifier(&self.scope_ref)
            || !valid_blake3_digest(&self.root_artifact_hash)
            || self.target_artifact_hashes.is_empty()
            || self.target_artifact_hashes.len() > MAX_NEUROSEMANTIC_LIFECYCLE_VERIFICATION_TARGETS
            || self.target_artifact_hashes.iter().any(|hash| !valid_blake3_digest(hash))
            || !self.target_artifact_hashes.iter().any(|hash| hash == &self.root_artifact_hash)
        {
            return Err("neurosemantic lifecycle verification target set is invalid".into());
        }
        let unique_targets: BTreeSet<&str> =
            self.target_artifact_hashes.iter().map(String::as_str).collect();
        if unique_targets.len() != self.target_artifact_hashes.len() {
            return Err("neurosemantic lifecycle verification target set contains duplicates".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic lifecycle verification target set JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let target_set: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic lifecycle verification target set JSON: {error}"))?;
        target_set.validate()?;
        Ok(target_set)
    }
}

/// Content-addressed evidence that a downstream lifecycle effect was observed.
/// Validation establishes exact binding of the receipt to the artifact, lineage,
/// and supplied effect evidence. It does not establish that the external effect
/// actually occurred or that all copies/derivatives were changed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticArtifactLifecycleReceipt {
    pub schema_version: u16,
    pub receipt_ref: String,
    pub artifact_hash: String,
    pub derivation_provenance_ref: String,
    pub derivation_provenance_hash: String,
    /// Monotonic event sequence when receipts are chained. Sequence zero is the first event.
    pub event_sequence: u64,
    /// Content hash of the immediately preceding lifecycle receipt, when chained.
    pub previous_receipt_hash: Option<String>,
    pub action: NeurosemanticArtifactLifecycleAction,
    pub state: NeurosemanticArtifactLifecycleState,
    /// Content-addressed evidence of the applied downstream effect. Absent before Applied.
    pub effect_evidence_ref: Option<String>,
    pub effect_evidence_hash: Option<String>,
    /// Identity of the actor/process that emitted the applied effect evidence.
    /// This is not an authority assertion; it exists so independence is mechanically checkable.
    pub effect_agent_ref: Option<String>,
    /// Identity of the independently verifying actor/process. This is not a trust assertion.
    pub verification_agent_ref: Option<String>,
    /// Content-addressed evidence produced by the independent verifier.
    pub verification_evidence_ref: Option<String>,
    pub verification_evidence_hash: Option<String>,
    /// Exact effect-evidence hash the verifier attests to have inspected.
    pub verification_target_effect_evidence_hash: Option<String>,
    /// Explicit scope of what was independently inspected.
    pub verification_scope: Option<NeurosemanticArtifactLifecycleVerificationScope>,
    /// For EnumeratedTargetSet, a separate artifact enumerating the checked descendants.
    pub verification_scope_ref: Option<String>,
    pub verification_scope_hash: Option<String>,
    /// Canonical Git object ID for the execution context that emitted the receipt.
    pub execution_revision: String,
    pub observed_at_unix_s: u64,
    /// Replacement/superseding artifact identity for rectification or supersession.
    pub resulting_artifact_hash: Option<String>,
    /// Lineage identity for the replacement/superseding artifact.
    pub resulting_derivation_provenance_ref: Option<String>,
    pub resulting_derivation_provenance_hash: Option<String>,
}

/// Compute the content identity of a lifecycle verification target-set artifact.
pub fn compute_lifecycle_verification_scope_hash(scope_ref: &str, scope_bytes: &[u8]) -> String {
    let mut bytes = Vec::with_capacity(
        NEUROSEMANTIC_LIFECYCLE_VERIFICATION_SCOPE_DOMAIN.len()
            + scope_ref.len()
            + scope_bytes.len()
            + 1,
    );
    bytes.extend_from_slice(NEUROSEMANTIC_LIFECYCLE_VERIFICATION_SCOPE_DOMAIN);
    bytes.extend_from_slice(scope_ref.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(scope_bytes);
    content_hash(&bytes)
}

impl NeurosemanticArtifactLifecycleReceipt {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_ARTIFACT_LIFECYCLE_RECEIPT_SCHEMA_VERSION
            || !valid_identifier(&self.receipt_ref)
            || !valid_blake3_digest(&self.artifact_hash)
            || !valid_identifier(&self.derivation_provenance_ref)
            || !valid_blake3_digest(&self.derivation_provenance_hash)
            || !valid_execution_revision(&self.execution_revision)
        {
            return Err("neurosemantic lifecycle receipt fields are invalid".into());
        }

        match (self.event_sequence, &self.previous_receipt_hash) {
            (0, None) => {}
            (0, Some(_)) => {
                return Err("neurosemantic lifecycle sequence zero cannot reference a previous receipt".into());
            }
            (_, None) => {
                return Err("neurosemantic lifecycle nonzero sequence requires a previous receipt hash".into());
            }
            (_, Some(previous_hash)) if !valid_blake3_digest(previous_hash) => {
                return Err("neurosemantic lifecycle previous receipt hash is invalid".into());
            }
            _ => {}
        }

        let has_effect = match (&self.effect_evidence_ref, &self.effect_evidence_hash) {
            (Some(reference), Some(hash)) if valid_identifier(reference) && valid_blake3_digest(hash) => true,
            (None, None) => false,
            _ => return Err("neurosemantic lifecycle effect evidence reference/hash must be present together".into()),
        };

        if let Some(effect_agent) = &self.effect_agent_ref {
            if !valid_identifier(effect_agent) {
                return Err("neurosemantic lifecycle effect agent reference is invalid".into());
            }
        }

        let has_verification = match (
            &self.verification_agent_ref,
            &self.verification_evidence_ref,
            &self.verification_evidence_hash,
            &self.verification_target_effect_evidence_hash,
            self.verification_scope,
            &self.verification_scope_ref,
            &self.verification_scope_hash,
        ) {
            (
                Some(agent),
                Some(reference),
                Some(hash),
                Some(target_hash),
                Some(scope),
                scope_ref,
                scope_hash,
            ) if valid_identifier(agent)
                && valid_identifier(reference)
                && valid_blake3_digest(hash)
                && valid_blake3_digest(target_hash)
                && match scope {
                    NeurosemanticArtifactLifecycleVerificationScope::ArtifactOnly =>
                        scope_ref.is_none() && scope_hash.is_none(),
                    NeurosemanticArtifactLifecycleVerificationScope::EnumeratedTargetSet => matches!(
                        (scope_ref, scope_hash),
                        (Some(scope_reference), Some(scope_digest))
                            if valid_identifier(scope_reference) && valid_blake3_digest(scope_digest)
                    ),
                } => true,
            (None, None, None, None, None, None, None) => false,
            _ => return Err("neurosemantic lifecycle verification evidence fields are incomplete".into()),
        };

        match self.state {
            NeurosemanticArtifactLifecycleState::Requested
            | NeurosemanticArtifactLifecycleState::Accepted
            | NeurosemanticArtifactLifecycleState::Processing => {
                if has_effect || has_verification || self.effect_agent_ref.is_some() {
                    return Err("neurosemantic lifecycle pre-application state cannot claim effect or independent verification evidence".into());
                }
            }
            NeurosemanticArtifactLifecycleState::Applied => {
                if !has_effect || has_verification || self.effect_agent_ref.is_none() {
                    return Err("neurosemantic applied lifecycle state requires effect evidence, an effect agent, and no independent verification".into());
                }
            }
            NeurosemanticArtifactLifecycleState::IndependentlyVerified => {
                if !has_effect || !has_verification || self.effect_agent_ref.is_none() {
                    return Err("neurosemantic independently-verified lifecycle state requires effect evidence, an effect agent, and verifier evidence".into());
                }
                if self.verification_target_effect_evidence_hash != self.effect_evidence_hash {
                    return Err("neurosemantic verifier target does not match the applied effect evidence".into());
                }
                if self.verification_agent_ref == self.effect_agent_ref {
                    return Err("neurosemantic independent verifier must differ from the effect agent".into());
                }
            }
            NeurosemanticArtifactLifecycleState::Rejected => {
                if has_verification || self.effect_agent_ref.is_some() {
                    return Err("neurosemantic rejected lifecycle state cannot claim independent verification or applied effect identity".into());
                }
            }
        }

        match self.action {
            NeurosemanticArtifactLifecycleAction::Rectification
            | NeurosemanticArtifactLifecycleAction::Supersession => {
                match (
                    &self.resulting_artifact_hash,
                    &self.resulting_derivation_provenance_ref,
                    &self.resulting_derivation_provenance_hash,
                ) {
                    (Some(artifact_hash), Some(lineage_ref), Some(lineage_hash))
                        if valid_blake3_digest(artifact_hash)
                            && artifact_hash != &self.artifact_hash
                            && valid_identifier(lineage_ref)
                            && valid_blake3_digest(lineage_hash)
                            && lineage_ref != &self.derivation_provenance_ref
                            && lineage_hash != &self.derivation_provenance_hash => {}
                    _ => {
                        return Err("neurosemantic lifecycle replacement action requires distinct resulting artifact and lineage identities".into());
                    }
                }
            }
            NeurosemanticArtifactLifecycleAction::AccessRevocation
            | NeurosemanticArtifactLifecycleAction::Retention
            | NeurosemanticArtifactLifecycleAction::Erasure => {
                if self.resulting_artifact_hash.is_some()
                    || self.resulting_derivation_provenance_ref.is_some()
                    || self.resulting_derivation_provenance_hash.is_some()
                {
                    return Err("neurosemantic lifecycle non-replacement action cannot carry replacement artifact lineage".into());
                }
            }
        }

        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic lifecycle receipt JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let receipt: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic lifecycle receipt JSON: {error}"))?;
        receipt.validate()?;
        Ok(receipt)
    }

    /// Verify the concrete effect-evidence bytes against the exact receipt hash.
    /// This proves evidence identity, not the truth of the external effect claim.
    pub fn verify_effect_evidence_bytes(&self, evidence_bytes: &[u8]) -> bool {
        self.validate().is_ok()
            && evidence_bytes.len() <= MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            && self.effect_evidence_hash.is_some()
            && content_hash(evidence_bytes)
                == self.effect_evidence_hash.as_deref().unwrap_or_default()
    }

    /// Verify independent-verifier evidence against the exact effect evidence and scope.
    /// This proves the verification artifact and target bindings, not verifier trustworthiness.
    pub fn verify_independent_verification_bytes(
        &self,
        verification_evidence_bytes: &[u8],
        now_unix_s: u64,
    ) -> Result<(), String> {
        self.validate()?;
        if self.state != NeurosemanticArtifactLifecycleState::IndependentlyVerified {
            return Err("neurosemantic lifecycle is not in independently-verified state".into());
        }
        if self.observed_at_unix_s > now_unix_s {
            return Err("neurosemantic lifecycle verification receipt is dated in the future".into());
        }
        if verification_evidence_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic lifecycle verification evidence exceeds the serialized artifact limit".into());
        }
        if content_hash(verification_evidence_bytes)
            != self.verification_evidence_hash.as_deref().unwrap_or_default()
        {
            return Err("neurosemantic lifecycle verification evidence hash mismatch".into());
        }
        if self.verification_target_effect_evidence_hash != self.effect_evidence_hash {
            return Err("neurosemantic lifecycle verifier target does not match effect evidence".into());
        }
        Ok(())
    }

    /// Verify the optional content-addressed target-set artifact for descendant coverage.
    pub fn verify_verification_scope_bytes(&self, scope_bytes: &[u8]) -> Result<(), String> {
        self.validate()?;
        if self.verification_scope
            != Some(NeurosemanticArtifactLifecycleVerificationScope::EnumeratedTargetSet)
        {
            return Err("neurosemantic lifecycle receipt does not declare an enumerated target-set scope".into());
        }
        if scope_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic lifecycle verification scope exceeds the serialized artifact limit".into());
        }
        let scope_ref = self.verification_scope_ref.as_deref().ok_or_else(||
            "neurosemantic lifecycle receipt has no verification scope reference".to_string(),
        )?;
        let expected_hash = self.verification_scope_hash.as_deref().ok_or_else(||
            "neurosemantic lifecycle receipt has no verification scope hash".to_string(),
        )?;
        if compute_lifecycle_verification_scope_hash(scope_ref, scope_bytes) != expected_hash {
            return Err("neurosemantic lifecycle verification scope hash mismatch".into());
        }
        let target_set = NeurosemanticArtifactLifecycleVerificationTargetSet::from_json_bytes(scope_bytes)?;
        if target_set.scope_ref != scope_ref || target_set.root_artifact_hash != self.artifact_hash {
            return Err("neurosemantic lifecycle verification target set does not match the receipt root".into());
        }
        Ok(())
    }

    /// Content-address the exact receipt, including lifecycle state and chain linkage.
    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("neurosemantic lifecycle receipt serialization: {error}"))?;
        Ok(content_hash(&bytes))
    }

    /// Verify the append-only state transition from one receipt to this receipt.
    /// This prevents sequence gaps, predecessor substitution, skipped lifecycle stages,
    /// and impossible state regressions.
    pub fn verify_transition(&self, previous: &Self) -> Result<(), String> {
        self.validate()?;
        previous.validate()?;
        let previous_fingerprint = previous.fingerprint()?;
        let expected_sequence = previous.event_sequence.checked_add(1)
            .ok_or_else(|| "neurosemantic lifecycle event sequence overflow".to_string())?;
        if self.event_sequence != expected_sequence
            || self.previous_receipt_hash.as_deref() != Some(previous_fingerprint.as_str())
            || self.artifact_hash != previous.artifact_hash
            || self.derivation_provenance_ref != previous.derivation_provenance_ref
            || self.derivation_provenance_hash != previous.derivation_provenance_hash
            || self.action != previous.action
            || self.observed_at_unix_s < previous.observed_at_unix_s
        {
            return Err("neurosemantic lifecycle receipt transition is not a valid append-only continuation".into());
        }
        if previous.state == NeurosemanticArtifactLifecycleState::Applied
            && self.state == NeurosemanticArtifactLifecycleState::IndependentlyVerified
            && (self.effect_agent_ref != previous.effect_agent_ref
                || self.effect_evidence_ref != previous.effect_evidence_ref
                || self.effect_evidence_hash != previous.effect_evidence_hash)
        {
            return Err("neurosemantic independent verification cannot substitute the applied effect identity or evidence".into());
        }
        match (previous.state, self.state) {
            (NeurosemanticArtifactLifecycleState::Requested, NeurosemanticArtifactLifecycleState::Accepted)
            | (NeurosemanticArtifactLifecycleState::Accepted, NeurosemanticArtifactLifecycleState::Processing)
            | (NeurosemanticArtifactLifecycleState::Processing, NeurosemanticArtifactLifecycleState::Applied)
            | (NeurosemanticArtifactLifecycleState::Applied, NeurosemanticArtifactLifecycleState::IndependentlyVerified)
            | (NeurosemanticArtifactLifecycleState::Requested, NeurosemanticArtifactLifecycleState::Rejected)
            | (NeurosemanticArtifactLifecycleState::Accepted, NeurosemanticArtifactLifecycleState::Rejected)
            | (NeurosemanticArtifactLifecycleState::Processing, NeurosemanticArtifactLifecycleState::Rejected) => Ok(()),
            _ => Err("neurosemantic lifecycle state transition is invalid".into()),
        }
    }

    /// Verify the concrete replacement lineage record when a replacement action is declared.
    pub fn verify_resulting_lineage_binding_bytes(
        &self,
        lineage_record_bytes: &[u8],
    ) -> Result<NeurosemanticDerivationLineageRecord, String> {
        self.validate()?;
        let expected_artifact_hash = self.resulting_artifact_hash.as_deref().ok_or_else(||
            "neurosemantic lifecycle receipt has no resulting artifact for lineage verification".to_string())?;
        let expected_lineage_ref = self.resulting_derivation_provenance_ref.as_deref().ok_or_else(||
            "neurosemantic lifecycle receipt has no resulting lineage reference".to_string())?;
        let expected_lineage_hash = self.resulting_derivation_provenance_hash.as_deref().ok_or_else(||
            "neurosemantic lifecycle receipt has no resulting lineage hash".to_string())?;
        if lineage_record_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic replacement lineage record exceeds the serialized artifact limit".into());
        }
        let record = NeurosemanticDerivationLineageRecord::from_json_bytes(lineage_record_bytes)?;
        if record.lineage_ref != expected_lineage_ref
            || record.output_artifact_hash != expected_artifact_hash
            || compute_derivation_provenance_hash(expected_lineage_ref, lineage_record_bytes)
                != expected_lineage_hash
        {
            return Err("neurosemantic lifecycle replacement lineage binding mismatch".into());
        }
        Ok(record)
    }

    /// Bind a lifecycle receipt to the exact derived artifact and lineage record already
    /// carried by the policy boundary, while rejecting future-dated receipts.
    pub fn verify_binding(
        &self,
        expected_artifact_hash: &str,
        expected_derivation_provenance_ref: &str,
        expected_derivation_provenance_hash: &str,
        evidence_bytes: &[u8],
        now_unix_s: u64,
    ) -> Result<(), String> {
        self.validate()?;
        if self.artifact_hash != expected_artifact_hash
            || self.derivation_provenance_ref != expected_derivation_provenance_ref
            || self.derivation_provenance_hash != expected_derivation_provenance_hash
        {
            return Err("neurosemantic lifecycle receipt does not match the derived artifact lineage".into());
        }
        if self.observed_at_unix_s > now_unix_s {
            return Err("neurosemantic lifecycle receipt is dated in the future".into());
        }
        if !self.verify_effect_evidence_bytes(evidence_bytes) {
            return Err("neurosemantic lifecycle effect evidence hash mismatch".into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeurosemanticHandlingAction {
    Transmit,
    Persist,
    SecondaryUse(NeurosemanticSecondaryUse),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticHandlingPolicy {
    pub schema_version: u16,
    /// Opaque reference to the externally authoritative policy/consent record.
    /// Symthaea binds this reference into packet integrity but does not authenticate
    /// the authority; the deployment policy layer remains responsible for verification.
    pub policy_provenance_ref: String,
    /// BLAKE3-256 digest of the exact externally authoritative policy/consent record bytes
    /// (or its separately specified canonical form). This is a binding, not an authority signature.
    pub policy_provenance_hash: String,
    /// Opaque reference to the external derivation/data-lineage record for this artifact.
    pub derivation_provenance_ref: String,
    /// BLAKE3-256 binding over the exact derivation reference and exact lineage record bytes.
    pub derivation_provenance_hash: String,
    /// Exact payload/artifact hash declared by the external lineage record.
    #[serde(default)]
    pub derivation_output_artifact_hash: String,
    /// Jurisdiction identifier asserted for the originating data/controller context.
    /// This is an interoperable policy identifier, not a legal determination.
    pub origin_jurisdiction: String,
    /// Explicit destination allow-list. Empty means deny-all.
    #[serde(default)]
    pub permitted_destination_jurisdictions: BTreeSet<String>,
    /// Explicitly authorized downstream uses beyond the packet's primary purpose.
    /// Empty means no secondary use.
    #[serde(default)]
    pub permitted_secondary_uses: BTreeSet<NeurosemanticSecondaryUse>,
    pub retention: NeurosemanticRetentionPolicy,
    /// Maximum age of an external authority/status resolution accepted by this policy.
    /// This is distinct from the protocol-wide 24-hour defensive upper bound.
    pub max_authority_resolution_age_s: u64,
}

impl Default for NeurosemanticHandlingPolicy {
    fn default() -> Self {
        Self {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: String::new(),
            policy_provenance_hash: String::new(),
            derivation_provenance_ref: String::new(),
            derivation_provenance_hash: String::new(),
            derivation_output_artifact_hash: String::new(),
            origin_jurisdiction: String::new(),
            permitted_destination_jurisdictions: BTreeSet::new(),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
            max_authority_resolution_age_s: 300,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticPolicyAuthorityAttestation {
    pub schema_version: u16,
    /// External authority identifier. Symthaea does not establish trust in this
    /// identifier; the deployment trust layer must map it to an authorized issuer.
    pub authority_ref: String,
    /// External key identifier used by the authority's trust registry.
    pub key_ref: String,
    /// Exact policy fingerprint that the authority attests.
    pub handling_policy_fingerprint: String,
    /// Exact provenance digest that the authority attests.
    pub policy_provenance_hash: String,
    /// Inclusive validity start for the attestation.
    pub issued_at_unix_s: u64,
    /// Exclusive validity end for the attestation.
    pub expires_at_unix_s: u64,
    /// Ed25519 signature over the domain-separated attestation message.
    pub signature: Vec<u8>,
}

pub const NEUROSEMANTIC_POLICY_ATTESTATION_SCHEMA_VERSION: u16 = 1;

/// Explicit external authority states. Only Active may become a handling capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeurosemanticAuthorityStatus {
    Active,
    Suspended,
    Revoked,
    Unknown,
    Unavailable,
}

/// The exact consent context to which an external authority resolution is bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeurosemanticConsentBindingContext {
    pub subject_ref: String,
    pub peer_ref: String,
    pub lease_id: String,
    pub consent_epoch: u64,
    pub consent_lease_fingerprint: String,
    pub purpose: CommunicationPurpose,
    pub channel: CognitiveChannel,
    pub direction: ChannelDirection,
}

impl NeurosemanticConsentBindingContext {
    pub fn validate(&self) -> Result<(), String> {
        if !valid_identifier(&self.subject_ref)
            || !valid_identifier(&self.peer_ref)
            || !valid_identifier(&self.lease_id)
            || !valid_blake3_digest(&self.consent_lease_fingerprint)
        {
            return Err("neurosemantic consent binding context identifiers or lease fingerprint are invalid".into());
        }
        Ok(())
    }
}

/// Signed current-state resolution from the external identity/policy bridge.
///
/// The signature authenticates the exact resolution to the configured resolver
/// key. It does not establish that the resolver key is trustworthy; that mapping
/// remains an outer deployment responsibility.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticAuthorityResolutionAttestation {
    pub schema_version: u16,
    pub resolver_ref: String,
    pub resolver_key_ref: String,
    pub authority_ref: String,
    pub authority_key_ref: String,
    pub authority_attestation_fingerprint: String,
    pub handling_policy_fingerprint: String,
    pub policy_provenance_ref: String,
    pub policy_provenance_hash: String,
    pub subject_ref: String,
    pub peer_ref: String,
    pub lease_id: String,
    pub consent_epoch: u64,
    pub consent_lease_fingerprint: String,
    pub purpose: CommunicationPurpose,
    pub channel: CognitiveChannel,
    pub direction: ChannelDirection,
    pub status: NeurosemanticAuthorityStatus,
    pub status_source_ref: String,
    pub status_source_hash: String,
    pub checked_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub signature: Vec<u8>,
}

pub const NEUROSEMANTIC_AUTHORITY_RESOLUTION_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeurosemanticPolicyProvenanceBinding {
    policy_provenance_ref: String,
    policy_provenance_hash: String,
    derivation_provenance_ref: String,
    derivation_provenance_hash: String,
    derivation_output_artifact_hash: String,
    handling_policy_fingerprint: String,
    authority_ref: String,
    key_ref: String,
    attestation_fingerprint: String,
    attestation_expires_at_unix_s: u64,
    authority_resolution_fingerprint: String,
    authority_resolution_checked_at_unix_s: u64,
    authority_resolution_expires_at_unix_s: u64,
    authority_resolution_subject_ref: String,
    authority_resolution_peer_ref: String,
    authority_resolution_lease_id: String,
    authority_resolution_consent_epoch: u64,
    authority_resolution_lease_fingerprint: String,
    authority_resolution_status_source_ref: String,
    authority_resolution_status_source_hash: String,
    authority_resolution_purpose: CommunicationPurpose,
    authority_resolution_channel: CognitiveChannel,
    authority_resolution_direction: ChannelDirection,
    authority_resolution_status: NeurosemanticAuthorityStatus,
}

impl NeurosemanticAuthorityResolutionAttestation {
    pub fn message_bytes(&self) -> Result<Vec<u8>, String> {
        // Validate the signature-independent fields before signing so callers
        // cannot manufacture a cryptographically valid proof over an invalid
        // authority-resolution record.
        let mut unsigned = self.clone();
        unsigned.signature = vec![0; MAX_NEUROSEMANTIC_AUTHORITY_SIGNATURE_BYTES];
        unsigned.validate()?;
        unsigned.signature.clear();
        let encoded = serde_json::to_vec(&unsigned)
            .map_err(|error| format!("authority resolution serialization: {error}"))?;
        let mut bytes =
            Vec::with_capacity(NEUROSEMANTIC_AUTHORITY_RESOLUTION_DOMAIN.len() + encoded.len() + 8);
        bytes.extend_from_slice(NEUROSEMANTIC_AUTHORITY_RESOLUTION_DOMAIN);
        bytes.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&encoded);
        Ok(bytes)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_AUTHORITY_RESOLUTION_SCHEMA_VERSION
            || !valid_identifier(&self.resolver_ref)
            || self.resolver_ref.len() > MAX_NEUROSEMANTIC_RESOLVER_REF_BYTES
            || !valid_identifier(&self.resolver_key_ref)
            || self.resolver_key_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_KEY_REF_BYTES
            || !valid_identifier(&self.authority_ref)
            || self.authority_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_REF_BYTES
            || !valid_identifier(&self.authority_key_ref)
            || self.authority_key_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_KEY_REF_BYTES
            || !valid_blake3_digest(&self.authority_attestation_fingerprint)
            || !valid_blake3_digest(&self.handling_policy_fingerprint)
            || !valid_identifier(&self.policy_provenance_ref)
            || !valid_blake3_digest(&self.policy_provenance_hash)
            || !valid_identifier(&self.subject_ref)
            || !valid_identifier(&self.peer_ref)
            || !valid_identifier(&self.lease_id)
            || !valid_blake3_digest(&self.consent_lease_fingerprint)
            || !valid_identifier(&self.status_source_ref)
            || self.status_source_ref.len() > MAX_NEUROSEMANTIC_STATUS_SOURCE_REF_BYTES
            || !valid_blake3_digest(&self.status_source_hash)
            || self.checked_at_unix_s >= self.expires_at_unix_s
            || self.expires_at_unix_s.saturating_sub(self.checked_at_unix_s)
                > MAX_NEUROSEMANTIC_AUTHORITY_RESOLUTION_TTL_S
            || self.signature.len() != MAX_NEUROSEMANTIC_AUTHORITY_SIGNATURE_BYTES
        {
            return Err("neurosemantic authority resolution fields are invalid".into());
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic authority resolution JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let resolution: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic authority resolution JSON: {error}"))?;
        resolution.validate()?;
        Ok(resolution)
    }

    /// Verify that the exact external authority/status record is bound to the
    /// declared status-source reference and digest. The resolver signature still
    /// determines the asserted status; this check only prevents source-artifact
    /// substitution behind an otherwise unchanged source reference.
    pub fn verify_status_source_binding_bytes(&self, record_bytes: &[u8]) -> bool {
        self.validate().is_ok()
            && record_bytes.len() <= MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            && compute_status_source_hash(&self.status_source_ref, record_bytes)
                == self.status_source_hash
    }

    pub fn verify(
        &self,
        expected_policy_fingerprint: &str,
        expected_policy_provenance_ref: &str,
        expected_policy_provenance_hash: &str,
        authority_attestation: &NeurosemanticPolicyAuthorityAttestation,
        expected_context: &NeurosemanticConsentBindingContext,
        verifying_key: &VerifyingKey,
        now_unix_s: u64,
    ) -> Result<(), String> {
        self.validate()?;
        expected_context.validate()?;
        if self.authority_ref != authority_attestation.authority_ref
            || self.authority_key_ref != authority_attestation.key_ref
            || self.authority_attestation_fingerprint
                != authority_attestation.fingerprint_for_attestation()?
            || self.handling_policy_fingerprint != expected_policy_fingerprint
            || self.policy_provenance_ref != expected_policy_provenance_ref
            || self.policy_provenance_hash != expected_policy_provenance_hash
            || self.subject_ref != expected_context.subject_ref
            || self.peer_ref != expected_context.peer_ref
            || self.lease_id != expected_context.lease_id
            || self.consent_epoch != expected_context.consent_epoch
            || self.consent_lease_fingerprint != expected_context.consent_lease_fingerprint
            || self.purpose != expected_context.purpose
            || self.channel != expected_context.channel
            || self.direction != expected_context.direction
            || self.checked_at_unix_s > now_unix_s
            || now_unix_s >= self.expires_at_unix_s
        {
            return Err(
                "neurosemantic authority resolution does not match the current policy or consent context"
                    .into(),
            );
        }

        let message = self.message_bytes()?;
        let signature_bytes: [u8; 64] = self
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| "neurosemantic authority resolution signature has invalid length".to_string())?;
        let signature = Signature::from_bytes(&signature_bytes);
        verifying_key
            .verify(&message, &signature)
            .map_err(|_| "neurosemantic authority resolution signature verification failed".to_string())
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("authority resolution serialization: {error}"))?;
        Ok(content_hash(&bytes))
    }
}

impl NeurosemanticPolicyAuthorityAttestation {
    /// Deserialize an untrusted authority attestation only after enforcing the
    /// serialized byte ceiling and structural bounds.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic authority attestation JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let attestation: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic authority attestation JSON: {error}"))?;
        attestation.validate()?;
        Ok(attestation)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_POLICY_ATTESTATION_SCHEMA_VERSION
            || !valid_identifier(&self.authority_ref)
            || !valid_identifier(&self.key_ref)
            || self.authority_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_REF_BYTES
            || self.key_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_KEY_REF_BYTES
            || !valid_blake3_digest(&self.handling_policy_fingerprint)
            || !valid_blake3_digest(&self.policy_provenance_hash)
            || self.issued_at_unix_s >= self.expires_at_unix_s
            || self.signature.len() != MAX_NEUROSEMANTIC_AUTHORITY_SIGNATURE_BYTES
        {
            return Err("neurosemantic authority attestation fields are invalid".into());
        }
        Ok(())
    }

    pub fn message_bytes(
        authority_ref: &str,
        key_ref: &str,
        handling_policy_fingerprint: &str,
        policy_provenance_hash: &str,
        issued_at_unix_s: u64,
        expires_at_unix_s: u64,
    ) -> Result<Vec<u8>, String> {
        if !valid_identifier(authority_ref)
            || authority_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_REF_BYTES
            || !valid_identifier(key_ref)
            || key_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_KEY_REF_BYTES
            || !valid_blake3_digest(handling_policy_fingerprint)
            || !valid_blake3_digest(policy_provenance_hash)
            || issued_at_unix_s >= expires_at_unix_s
        {
            return Err("neurosemantic authority attestation fields are invalid".into());
        }
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(NEUROSEMANTIC_POLICY_ATTESTATION_DOMAIN);
        for value in [authority_ref.as_bytes(), key_ref.as_bytes(), handling_policy_fingerprint.as_bytes(), policy_provenance_hash.as_bytes()] {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value);
        }
        bytes.extend_from_slice(&issued_at_unix_s.to_le_bytes());
        bytes.extend_from_slice(&expires_at_unix_s.to_le_bytes());
        Ok(bytes)
    }

    pub fn verify(
        &self,
        expected_policy_fingerprint: &str,
        expected_policy_provenance_hash: &str,
        verifying_key: &VerifyingKey,
        now_unix_s: u64,
    ) -> Result<(), String> {
        self.validate()?;
        if self.handling_policy_fingerprint != expected_policy_fingerprint
            || self.policy_provenance_hash != expected_policy_provenance_hash
            || now_unix_s < self.issued_at_unix_s
            || now_unix_s >= self.expires_at_unix_s
        {
            return Err("neurosemantic authority attestation is invalid or outside its validity interval".into());
        }
        let message = Self::message_bytes(
            &self.authority_ref,
            &self.key_ref,
            &self.handling_policy_fingerprint,
            &self.policy_provenance_hash,
            self.issued_at_unix_s,
            self.expires_at_unix_s,
        )?;
        let signature_bytes: [u8; 64] = self.signature.as_slice()
            .try_into()
            .map_err(|_| "neurosemantic authority signature has invalid length".to_string())?;
        let signature = Signature::from_bytes(&signature_bytes);
        verifying_key
            .verify(&message, &signature)
            .map_err(|_| "neurosemantic authority signature verification failed".to_string())
    }

    /// Fingerprint the exact serialized authority proof artifact, including its signature.
    pub fn fingerprint_for_attestation(&self) -> Result<String, String> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("authority attestation serialization: {error}"))?;
        Ok(content_hash(&bytes))
    }
}

impl NeurosemanticPolicyProvenanceBinding {
    pub fn reference(&self) -> &str {
        &self.policy_provenance_ref
    }

    pub fn digest(&self) -> &str {
        &self.policy_provenance_hash
    }

    pub fn handling_policy_fingerprint(&self) -> &str {
        &self.handling_policy_fingerprint
    }

    pub fn derivation_provenance_ref(&self) -> &str {
        &self.derivation_provenance_ref
    }

    pub fn derivation_provenance_hash(&self) -> &str {
        &self.derivation_provenance_hash
    }

    pub fn derivation_output_artifact_hash(&self) -> &str {
        &self.derivation_output_artifact_hash
    }

    pub fn authority_ref(&self) -> &str {
        &self.authority_ref
    }

    pub fn key_ref(&self) -> &str {
        &self.key_ref
    }

    pub fn attestation_fingerprint(&self) -> &str {
        &self.attestation_fingerprint
    }

    pub fn attestation_expires_at_unix_s(&self) -> u64 {
        self.attestation_expires_at_unix_s
    }

    /// Exact content hash of the signed external authority-resolution snapshot.
    pub fn authority_resolution_fingerprint(&self) -> &str {
        &self.authority_resolution_fingerprint
    }

    pub fn authority_resolution_checked_at_unix_s(&self) -> u64 {
        self.authority_resolution_checked_at_unix_s
    }

    pub fn authority_resolution_expires_at_unix_s(&self) -> u64 {
        self.authority_resolution_expires_at_unix_s
    }

    pub fn authority_resolution_status(&self) -> NeurosemanticAuthorityStatus {
        self.authority_resolution_status
    }

    pub fn authority_resolution_subject_ref(&self) -> &str {
        &self.authority_resolution_subject_ref
    }

    pub fn authority_resolution_peer_ref(&self) -> &str {
        &self.authority_resolution_peer_ref
    }

    pub fn authority_resolution_lease_id(&self) -> &str {
        &self.authority_resolution_lease_id
    }

    pub fn authority_resolution_consent_epoch(&self) -> u64 {
        self.authority_resolution_consent_epoch
    }

    pub fn authority_resolution_lease_fingerprint(&self) -> &str {
        &self.authority_resolution_lease_fingerprint
    }

    pub fn authority_resolution_status_source_ref(&self) -> &str {
        &self.authority_resolution_status_source_ref
    }

    pub fn authority_resolution_status_source_hash(&self) -> &str {
        &self.authority_resolution_status_source_hash
    }
}

impl NeurosemanticHandlingPolicy {
    pub fn fingerprint_for_attestation(&self) -> Result<String, String> {
        let mut canonical = self.clone();
        canonical.policy_provenance_hash.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| format!("handling policy serialization: {error}"))?;
        Ok(content_hash(&bytes))
    }

    pub fn validates(&self) -> bool {
        self.schema_version == NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION
            && valid_identifier(&self.policy_provenance_ref)
            && valid_blake3_digest(&self.policy_provenance_hash)
            && valid_identifier(&self.derivation_provenance_ref)
            && valid_blake3_digest(&self.derivation_provenance_hash)
            && valid_blake3_digest(&self.derivation_output_artifact_hash)
            && valid_jurisdiction_id(&self.origin_jurisdiction)
            && !self.permitted_destination_jurisdictions.is_empty()
            && self.permitted_destination_jurisdictions.len() <= MAX_NEUROSEMANTIC_DESTINATION_JURISDICTIONS
            && self
                .permitted_destination_jurisdictions
                .iter()
                .all(|jurisdiction| valid_jurisdiction_id(jurisdiction))
            && self.permitted_destination_jurisdictions.contains(&self.origin_jurisdiction)
            && self.permitted_secondary_uses.len() <= MAX_NEUROSEMANTIC_SECONDARY_USE_CLASSES
            && self.max_authority_resolution_age_s > 0
            && self.max_authority_resolution_age_s <= MAX_NEUROSEMANTIC_AUTHORITY_RESOLUTION_TTL_S
            && matches!(
                self.retention,
                NeurosemanticRetentionPolicy::Ephemeral
                    | NeurosemanticRetentionPolicy::UntilUnixS(_)
            )
    }

    /// Verify that the exact external policy/consent record is bound to the
    /// declared provenance reference and digest. This does not produce a
    /// handling capability or authenticate the issuing authority.
    pub fn verify_policy_record_binding_bytes(&self, record_bytes: &[u8]) -> bool {
        self.validates()
            && record_bytes.len() <= MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            && compute_policy_provenance_hash(&self.policy_provenance_ref, record_bytes)
                == self.policy_provenance_hash
    }

    /// Verify that the exact external derivation/data-lineage record is bound to the declared
    /// reference, digest, and declared output artifact identity. This does not itself establish
    /// that the lineage assertions are truthful; it establishes only exact artifact binding.
    pub fn verify_derivation_provenance_record_bytes(
        &self,
        record_bytes: &[u8],
    ) -> Result<NeurosemanticDerivationLineageRecord, String> {
        if !self.validates() {
            return Err("neurosemantic derivation provenance policy state is invalid".into());
        }
        let record = NeurosemanticDerivationLineageRecord::from_json_bytes(record_bytes)?;
        if record.lineage_ref != self.derivation_provenance_ref {
            return Err("neurosemantic derivation lineage reference mismatch".into());
        }
        if compute_derivation_provenance_hash(&self.derivation_provenance_ref, record_bytes)
            != self.derivation_provenance_hash
        {
            return Err("neurosemantic derivation provenance record hash mismatch".into());
        }
        if record.output_artifact_hash != self.derivation_output_artifact_hash {
            return Err("neurosemantic derivation lineage output artifact mismatch".into());
        }
        Ok(record)
    }

    /// Verify the structured derivation binding as a boolean predicate.
    pub fn verify_derivation_provenance_binding_bytes(&self, record_bytes: &[u8]) -> bool {
        self.verify_derivation_provenance_record_bytes(record_bytes).is_ok()
    }

    /// Produce a handling capability only after the external policy record,
    /// authority attestation, and a fresh signed authority-resolution snapshot all
    /// agree on the same consent context.
    pub fn bind_policy_provenance_with_attestation_and_resolution(
        &self,
        record_bytes: &[u8],
        derivation_record_bytes: &[u8],
        status_record_bytes: &[u8],
        attestation: &NeurosemanticPolicyAuthorityAttestation,
        verifying_key: &VerifyingKey,
        resolution: &NeurosemanticAuthorityResolutionAttestation,
        resolution_verifying_key: &VerifyingKey,
        context: &NeurosemanticConsentBindingContext,
        now_unix_s: u64,
    ) -> Result<NeurosemanticPolicyProvenanceBinding, String> {
        if !self.validates() {
            return Err("neurosemantic policy provenance state is invalid".into());
        }
        if record_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            || derivation_record_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            || status_record_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
        {
            return Err("neurosemantic provenance record exceeds the serialized artifact limit".into());
        }
        let policy_fingerprint = self.fingerprint_for_attestation()?;
        if compute_policy_provenance_hash(&self.policy_provenance_ref, record_bytes)
            != self.policy_provenance_hash
        {
            return Err("neurosemantic policy provenance reference/record binding mismatch".into());
        }
        let derivation_lineage =
            self.verify_derivation_provenance_record_bytes(derivation_record_bytes)?;
        if derivation_lineage.generated_at_unix_s > now_unix_s {
            return Err("neurosemantic derivation lineage is dated in the future".into());
        }
        if !resolution.verify_status_source_binding_bytes(status_record_bytes) {
            return Err("neurosemantic authority status source reference/record binding mismatch".into());
        }
        context.validate()?;
        attestation.verify(
            &policy_fingerprint,
            &self.policy_provenance_hash,
            verifying_key,
            now_unix_s,
        )?;
        resolution.verify(
            &policy_fingerprint,
            &self.policy_provenance_ref,
            &self.policy_provenance_hash,
            attestation,
            context,
            resolution_verifying_key,
            now_unix_s,
        )?;
        if resolution.checked_at_unix_s < attestation.issued_at_unix_s {
            return Err("neurosemantic authority resolution predates its authority attestation".into());
        }
        if resolution.expires_at_unix_s > attestation.expires_at_unix_s {
            return Err("neurosemantic authority resolution outlives its authority attestation".into());
        }
        if now_unix_s.saturating_sub(resolution.checked_at_unix_s)
            > self.max_authority_resolution_age_s
        {
            return Err("neurosemantic authority resolution exceeds policy freshness bound".into());
        }
        if resolution.status != NeurosemanticAuthorityStatus::Active {
            return Err("neurosemantic authority resolution is not active".into());
        }

        Ok(NeurosemanticPolicyProvenanceBinding {
            policy_provenance_ref: self.policy_provenance_ref.clone(),
            policy_provenance_hash: self.policy_provenance_hash.clone(),
            derivation_provenance_ref: self.derivation_provenance_ref.clone(),
            derivation_provenance_hash: self.derivation_provenance_hash.clone(),
            derivation_output_artifact_hash: self.derivation_output_artifact_hash.clone(),
            handling_policy_fingerprint: policy_fingerprint,
            authority_ref: attestation.authority_ref.clone(),
            key_ref: attestation.key_ref.clone(),
            attestation_fingerprint: attestation.fingerprint_for_attestation()?,
            attestation_expires_at_unix_s: attestation.expires_at_unix_s,
            authority_resolution_fingerprint: resolution.fingerprint()?,
            authority_resolution_checked_at_unix_s: resolution.checked_at_unix_s,
            authority_resolution_expires_at_unix_s: resolution.expires_at_unix_s,
            authority_resolution_subject_ref: resolution.subject_ref.clone(),
            authority_resolution_peer_ref: resolution.peer_ref.clone(),
            authority_resolution_lease_id: resolution.lease_id.clone(),
            authority_resolution_consent_epoch: resolution.consent_epoch,
            authority_resolution_lease_fingerprint: resolution.consent_lease_fingerprint.clone(),
            authority_resolution_status_source_ref: resolution.status_source_ref.clone(),
            authority_resolution_status_source_hash: resolution.status_source_hash.clone(),
            authority_resolution_purpose: resolution.purpose,
            authority_resolution_channel: resolution.channel,
            authority_resolution_direction: resolution.direction,
            authority_resolution_status: resolution.status,
        })
    }

    pub fn allows_destination(&self, destination_jurisdiction: &str) -> bool {
        self.validates()
            && valid_jurisdiction_id(destination_jurisdiction)
            && self
                .permitted_destination_jurisdictions
                .contains(destination_jurisdiction)
    }

    pub fn allows_action(&self, action: NeurosemanticHandlingAction, now_unix_s: u64) -> bool {
        if !self.validates() {
            return false;
        }

        match action {
            NeurosemanticHandlingAction::Transmit => true,
            NeurosemanticHandlingAction::Persist => self.retention_allows_persistence(now_unix_s),
            NeurosemanticHandlingAction::SecondaryUse(secondary_use) => {
                self.retention_allows_persistence(now_unix_s)
                    && self.permitted_secondary_uses.contains(&secondary_use)
            }
        }
    }

    pub fn retention_allows_persistence(&self, now_unix_s: u64) -> bool {
        match self.retention {
            NeurosemanticRetentionPolicy::Ephemeral => false,
            NeurosemanticRetentionPolicy::UntilUnixS(expires_at) => now_unix_s < expires_at,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticDataPolicy {
    pub schema_version: u16,
    pub data_class: NeurosemanticDataClass,
    #[serde(default)]
    pub inference_classes: BTreeSet<NeurosemanticInferenceClass>,
    #[serde(default)]
    pub permitted_purposes: BTreeSet<CommunicationPurpose>,
    #[serde(default)]
    pub handling: NeurosemanticHandlingPolicy,
}

impl Default for NeurosemanticDataPolicy {
    fn default() -> Self {
        Self {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            data_class: NeurosemanticDataClass::Unknown,
            inference_classes: BTreeSet::new(),
            permitted_purposes: BTreeSet::new(),
            handling: NeurosemanticHandlingPolicy::default(),
        }
    }
}

impl NeurosemanticDataPolicy {
    pub fn transportable(&self) -> bool {
        matches!(
            self.data_class,
            NeurosemanticDataClass::DerivedNeuralFeature
                | NeurosemanticDataClass::SemanticRepresentation
                | NeurosemanticDataClass::DecodedClaim
        )
    }

    pub fn validates(&self) -> bool {
        self.schema_version == NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION
            && self.data_class != NeurosemanticDataClass::Unknown
            && !self.inference_classes.is_empty()
            && !self.inference_classes.contains(&NeurosemanticInferenceClass::Unknown)
            && !self.permitted_purposes.is_empty()
            && self.handling.validates()
    }

    pub fn allows_purpose(&self, purpose: CommunicationPurpose) -> bool {
        self.validates() && self.permitted_purposes.contains(&purpose)
    }

    /// Secondary uses that explicitly infer affective state or identity must also
    /// be represented in the packet's declared inference classes. This prevents a
    /// downstream-use flag from escalating the inference capability beyond the
    /// data product's declared boundary.
    fn allows_secondary_inference(&self, secondary_use: NeurosemanticSecondaryUse) -> bool {
        match secondary_use {
            NeurosemanticSecondaryUse::AffectiveInference => {
                self.inference_classes
                    .contains(&NeurosemanticInferenceClass::AffectiveState)
            }
            NeurosemanticSecondaryUse::IdentityInference => {
                self.inference_classes
                    .contains(&NeurosemanticInferenceClass::Identity)
            }
            _ => true,
        }
    }

    pub fn allows_handling(
        &self,
        destination_jurisdiction: &str,
        action: NeurosemanticHandlingAction,
        now_unix_s: u64,
    ) -> bool {
        self.validates()
            && match action {
                NeurosemanticHandlingAction::SecondaryUse(secondary_use) => {
                    self.allows_secondary_inference(secondary_use)
                }
                _ => true,
            }
            && self.handling.allows_destination(destination_jurisdiction)
            && self.handling.allows_action(action, now_unix_s)
    }
}

/// Derived representations only. Raw neural samples are intentionally absent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NeurosemanticPayload {
    SemanticGraph(Vec<u8>),
    Hypervector(Vec<i8>),
    /// A representation that has already crossed an explicit decode boundary.
    DecodedClaim(Vec<u8>),
    /// Legacy opaque representation retained for deserialization compatibility.
    /// It cannot be authorized under a declared v1 data policy because its
    /// semantic data class is not machine-verifiable from the enum variant.
    StructuredRepresentation(Vec<u8>),
    DerivedNeuralFeature(Vec<f32>),
}

impl NeurosemanticPayload {
    /// Return the data class that this concrete payload variant can authorize
    /// without inspecting opaque user-defined bytes.
    pub fn intrinsic_data_class(&self) -> Option<NeurosemanticDataClass> {
        match self {
            Self::SemanticGraph(_) | Self::Hypervector(_) => {
                Some(NeurosemanticDataClass::SemanticRepresentation)
            }
            Self::DecodedClaim(_) => Some(NeurosemanticDataClass::DecodedClaim),
            Self::StructuredRepresentation(_) => None,
            Self::DerivedNeuralFeature(_) => Some(NeurosemanticDataClass::DerivedNeuralFeature),
        }
    }
}

/// Sensitivity class for policy and data minimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CognitiveSensitivity {
    Public,
    Contextual,
    Private,
    HighlyPrivate,
}

/// Explicit, peer-specific, purpose-bound, time-bounded consent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitiveConsentLease {
    pub lease_id: String,
    pub subject_id: String,
    pub peer_id: String,
    pub purpose: CommunicationPurpose,
    pub read_scopes: BTreeSet<CognitiveChannel>,
    pub write_scopes: BTreeSet<CognitiveChannel>,
    #[serde(default = "default_public_sensitivity")]
    pub max_read_sensitivity: CognitiveSensitivity,
    #[serde(default = "default_public_sensitivity")]
    pub max_write_sensitivity: CognitiveSensitivity,
    #[serde(default)]
    pub read_data_classes: BTreeSet<NeurosemanticDataClass>,
    #[serde(default)]
    pub write_data_classes: BTreeSet<NeurosemanticDataClass>,
    #[serde(default)]
    pub read_inference_classes: BTreeSet<NeurosemanticInferenceClass>,
    #[serde(default)]
    pub write_inference_classes: BTreeSet<NeurosemanticInferenceClass>,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub consent_epoch: u64,
    pub revoked: bool,
    /// Effective timestamp for a revocation record. A revoked lease must carry this value.
    #[serde(default)]
    pub revoked_at_unix_s: Option<u64>,
}

impl CognitiveConsentLease {
    /// Deserialize a persisted lease only after enforcing the serialized size ceiling.
    /// Untrusted callers should use this entry point rather than unbounded `serde_json`
    /// deserialization.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic consent lease JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let lease: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic consent lease JSON: {error}"))?;
        lease.validate()?;
        Ok(lease)
    }

    /// Fingerprint the exact serialized consent lease, including scopes, sensitivity
    /// ceilings, data/inference permissions, validity, revocation state, and epoch.
    /// Any consent mutation therefore produces a new capability identity.
    pub fn fingerprint_for_authorization(&self) -> Result<String, String> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("consent lease serialization: {error}"))?;
        Ok(content_hash(&bytes))
    }

    pub fn validate(&self) -> Result<(), String> {
        if !valid_identifier(&self.lease_id)
            || !valid_identifier(&self.subject_id)
            || !valid_identifier(&self.peer_id)
            || self.issued_at_unix_s >= self.expires_at_unix_s
        {
            return Err("consent lease identity or time bounds are invalid".into());
        }
        if self.revoked && self.revoked_at_unix_s.is_none() {
            return Err("revoked consent leases require an explicit effective timestamp".into());
        }
        if let Some(revoked_at) = self.revoked_at_unix_s {
            if revoked_at < self.issued_at_unix_s {
                return Err("consent revocation effective time cannot precede lease issuance".into());
            }
        }
        Ok(())
    }

    pub fn authorizes_sensitivity(
        &self,
        direction: ChannelDirection,
        sensitivity: CognitiveSensitivity,
    ) -> bool {
        let maximum = match direction {
            ChannelDirection::Read => self.max_read_sensitivity,
            ChannelDirection::Write => self.max_write_sensitivity,
        };
        sensitivity <= maximum
    }

    pub fn authorizes_data_policy(
        &self,
        direction: ChannelDirection,
        policy: &NeurosemanticDataPolicy,
    ) -> bool {
        if !policy.validates() {
            return false;
        }

        let (allowed_data_classes, allowed_inference_classes) = match direction {
            ChannelDirection::Read => (&self.read_data_classes, &self.read_inference_classes),
            ChannelDirection::Write => (&self.write_data_classes, &self.write_inference_classes),
        };

        allowed_data_classes.contains(&policy.data_class)
            && policy
                .inference_classes
                .iter()
                .all(|inference| allowed_inference_classes.contains(inference))
    }

    pub fn authorizes(
        &self,
        channel: CognitiveChannel,
        direction: ChannelDirection,
        purpose: CommunicationPurpose,
        peer_id: &str,
        now_unix_s: u64,
    ) -> bool {
        if self.revoked
            || self
                .revoked_at_unix_s
                .is_some_and(|revoked_at| now_unix_s >= revoked_at)
            || self.peer_id != peer_id
            || self.purpose != purpose
            || now_unix_s < self.issued_at_unix_s
            || now_unix_s >= self.expires_at_unix_s
        {
            return false;
        }

        match direction {
            ChannelDirection::Read => self.read_scopes.contains(&channel),
            ChannelDirection::Write => self.write_scopes.contains(&channel),
        }
    }
}

/// Content-addressed packet for derived cognitive representations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeurosemanticPacket {
    pub protocol_version: u16,
    pub sequence: u64,
    pub sender_id: String,
    pub recipient_id: String,
    pub purpose: CommunicationPurpose,
    pub channel: CognitiveChannel,
    pub direction: ChannelDirection,
    pub representation: RepresentationFamily,
    pub sensitivity: CognitiveSensitivity,
    #[serde(default)]
    pub data_policy: NeurosemanticDataPolicy,
    pub confidence: f32,
    pub payload: NeurosemanticPayload,
    pub payload_hash: String,
    pub packet_hash: String,
}

impl NeurosemanticPacket {
    /// Deserialize an untrusted packet only after enforcing the raw byte ceiling.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic packet JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let packet: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic packet JSON: {error}"))?;
        packet.validate_integrity()?;
        Ok(packet)
    }

    pub fn new(
        sequence: u64,
        sender_id: impl Into<String>,
        recipient_id: impl Into<String>,
        purpose: CommunicationPurpose,
        channel: CognitiveChannel,
        direction: ChannelDirection,
        representation: RepresentationFamily,
        sensitivity: CognitiveSensitivity,
        confidence: f32,
        payload: NeurosemanticPayload,
    ) -> Result<Self, String> {
        if sequence == 0 {
            return Err("packet sequence must be non-zero".into());
        }
        let sender_id = sender_id.into();
        let recipient_id = recipient_id.into();
        if !valid_identifier(&sender_id) || !valid_identifier(&recipient_id) {
            return Err("packet sender and recipient identifiers are invalid or oversized".into());
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err("confidence must be finite and in [0, 1]".into());
        }
        validate_payload(&payload)?;

        let mut packet = Self {
            protocol_version: NEUROSEMANTIC_PROTOCOL_VERSION,
            sequence,
            sender_id,
            recipient_id,
            purpose,
            channel,
            direction,
            representation,
            sensitivity,
            data_policy: NeurosemanticDataPolicy::default(),
            confidence,
            payload,
            payload_hash: String::new(),
            packet_hash: String::new(),
        };
        packet.refresh_hashes()?;
        Ok(packet)
    }

    pub fn new_with_policy(
        sequence: u64,
        sender_id: impl Into<String>,
        recipient_id: impl Into<String>,
        purpose: CommunicationPurpose,
        channel: CognitiveChannel,
        direction: ChannelDirection,
        representation: RepresentationFamily,
        sensitivity: CognitiveSensitivity,
        data_policy: NeurosemanticDataPolicy,
        confidence: f32,
        payload: NeurosemanticPayload,
    ) -> Result<Self, String> {
        if !data_policy.validates() || !data_policy.transportable() {
            return Err("neurosemantic data policy is invalid or not transportable in protocol v1".into());
        }
        if payload.intrinsic_data_class() != Some(data_policy.data_class) {
            return Err("neurosemantic payload type does not match its declared data class".into());
        }
        let expected_output_artifact_hash = payload_hash(&payload)?;
        if data_policy.handling.derivation_output_artifact_hash != expected_output_artifact_hash {
            return Err("neurosemantic derivation lineage output does not match the payload".into());
        }
        let mut packet = Self::new(
            sequence,
            sender_id,
            recipient_id,
            purpose,
            channel,
            direction,
            representation,
            sensitivity,
            confidence,
            payload,
        )?;
        packet.data_policy = data_policy;
        packet.refresh_hashes()?;
        Ok(packet)
    }

    pub fn validate_integrity(&self) -> Result<(), String> {
        if self.protocol_version != NEUROSEMANTIC_PROTOCOL_VERSION
            || !valid_identifier(&self.sender_id)
            || !valid_identifier(&self.recipient_id)
        {
            return Err("packet identity or protocol version is invalid".into());
        }
        if self.sequence == 0 {
            return Err("packet sequence must be non-zero".into());
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err("packet confidence must be finite and in [0, 1]".into());
        }

        if self.data_policy.schema_version != NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION {
            return Err("unsupported neurosemantic data policy schema version".into());
        }
        if self.data_policy.validates()
            && self.payload.intrinsic_data_class() != Some(self.data_policy.data_class)
        {
            return Err("neurosemantic payload type does not match its declared data class".into());
        }
        validate_payload(&self.payload)?;
        let expected_payload_hash = payload_hash(&self.payload)?;
        if self.payload_hash != expected_payload_hash {
            return Err("payload hash mismatch".into());
        }
        if self.data_policy.validates()
            && self.packet_derivation_output_artifact_hash() != expected_payload_hash
        {
            return Err("neurosemantic derivation lineage output does not match the payload".into());
        }

        let mut canonical = self.clone();
        canonical.packet_hash.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| format!("packet serialization: {error}"))?;
        if self.packet_hash != content_hash(&bytes) {
            return Err("packet hash mismatch".into());
        }
        Ok(())
    }

    fn packet_derivation_output_artifact_hash(&self) -> &str {
        &self.data_policy.handling.derivation_output_artifact_hash
    }

    pub fn refresh_hashes(&mut self) -> Result<(), String> {
        validate_payload(&self.payload)?;
        self.payload_hash = payload_hash(&self.payload)?;
        let mut canonical = self.clone();
        canonical.packet_hash.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| format!("packet serialization: {error}"))?;
        self.packet_hash = content_hash(&bytes);
        Ok(())
    }
}

/// A packet plus the exact consent epoch used to authorize it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthorizedNeurosemanticMessage {
    pub packet: NeurosemanticPacket,
    pub consent_epoch: u64,
    pub lease_id: String,
}

impl AuthorizedNeurosemanticMessage {
    /// Deserialize an authorized message only after enforcing the raw byte ceiling
    /// and re-running packet + consent validation.
    pub fn from_json_bytes(
        bytes: &[u8],
        lease: &CognitiveConsentLease,
        now_unix_s: u64,
    ) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "authorized neurosemantic message JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let message: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("authorized neurosemantic message JSON: {error}"))?;
        message.validate(lease, now_unix_s)?;
        Ok(message)
    }

    pub fn validate(
        &self,
        lease: &CognitiveConsentLease,
        now_unix_s: u64,
    ) -> Result<(), String> {
        self.packet.validate_integrity()?;
        lease.validate()?;

        if self.lease_id != lease.lease_id || self.consent_epoch != lease.consent_epoch {
            return Err("authorization epoch or lease id mismatch".into());
        }

        let (expected_sender, expected_recipient) = match self.packet.direction {
            ChannelDirection::Read => (lease.subject_id.as_str(), lease.peer_id.as_str()),
            ChannelDirection::Write => (lease.peer_id.as_str(), lease.subject_id.as_str()),
        };

        if self.packet.sender_id != expected_sender || self.packet.recipient_id != expected_recipient
        {
            return Err("packet endpoints do not match the consent direction".into());
        }

        if !self.packet.data_policy.transportable() {
            return Err("packet data class is not transportable in neurosemantic protocol v1".into());
        }
        if !self.packet.data_policy.allows_purpose(self.packet.purpose) {
            return Err("packet data policy does not permit the requested purpose".into());
        }
        if !lease.authorizes_data_policy(self.packet.direction, &self.packet.data_policy) {
            return Err("packet data class or inference class is not authorized by the consent lease".into());
        }

        if !lease.authorizes(
            self.packet.channel,
            self.packet.direction,
            self.packet.purpose,
            &lease.peer_id,
            now_unix_s,
        ) {
            return Err("communication is not authorized by the active consent lease".into());
        }
        if !lease.authorizes_sensitivity(self.packet.direction, self.packet.sensitivity) {
            return Err("packet sensitivity exceeds the consent lease ceiling".into());
        }

        Ok(())
    }

    /// Apply the declared handling policy to a concrete downstream request.
    /// Mycelix/another policy authority must supply the deployment context and
    /// independently authenticate the policy provenance; this method does not
    /// establish legal compliance or signed authorization by itself.
    pub fn validate_for_handling(
        &self,
        lease: &CognitiveConsentLease,
        provenance: &NeurosemanticPolicyProvenanceBinding,
        destination_jurisdiction: &str,
        action: NeurosemanticHandlingAction,
        now_unix_s: u64,
    ) -> Result<(), String> {
        if provenance.policy_provenance_ref
            != self.packet.data_policy.handling.policy_provenance_ref
            || provenance.policy_provenance_hash
                != self.packet.data_policy.handling.policy_provenance_hash
            || provenance.handling_policy_fingerprint
                != self.packet.data_policy.handling.fingerprint_for_attestation()?
            || provenance.authority_ref.is_empty()
            || provenance.key_ref.is_empty()
            || now_unix_s >= provenance.attestation_expires_at_unix_s
            || provenance.attestation_fingerprint.is_empty()
            || provenance.authority_resolution_fingerprint.is_empty()
            || provenance.authority_resolution_status != NeurosemanticAuthorityStatus::Active
            || provenance.authority_resolution_subject_ref != lease.subject_id
            || provenance.authority_resolution_peer_ref != lease.peer_id
            || provenance.authority_resolution_lease_id != lease.lease_id
            || provenance.authority_resolution_consent_epoch != lease.consent_epoch
            || provenance.authority_resolution_lease_fingerprint
                != lease.fingerprint_for_authorization()?
            || provenance.authority_resolution_purpose != self.packet.purpose
            || provenance.authority_resolution_channel != self.packet.channel
            || provenance.authority_resolution_direction != self.packet.direction
            || now_unix_s < provenance.authority_resolution_checked_at_unix_s
            || now_unix_s.saturating_sub(provenance.authority_resolution_checked_at_unix_s)
                > self.packet.data_policy.handling.max_authority_resolution_age_s
            || now_unix_s >= provenance.authority_resolution_expires_at_unix_s
        {
            return Err("policy provenance capability does not match the current policy, consent context, or authority status".into());
        }
        self.validate(lease, now_unix_s)?;
        if !self.packet.data_policy.allows_handling(
            destination_jurisdiction,
            action,
            now_unix_s,
        ) {
            return Err(
                "neurosemantic handling request exceeds destination, retention, or secondary-use policy"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Persistent per-lease sequence guard for replay/collision detection.
///
/// The tracker deliberately lives above packet integrity: integrity answers
/// "was this packet altered?", while this tracker answers "have we already
/// accepted this packet sequence in this consent epoch?".
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticReplayTracker {
    latest: BTreeMap<(String, String, String, u64), ReplayState>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ReplayState {
    sequence: u64,
    packet_hash: String,
    expires_at_unix_s: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayDecision {
    Accept,
    Duplicate,
}

impl NeurosemanticReplayTracker {
    /// Authorize first, then apply replay/collision protection.
    pub fn observe_authorized(
        &mut self,
        message: &AuthorizedNeurosemanticMessage,
        lease: &CognitiveConsentLease,
        now_unix_s: u64,
    ) -> Result<ReplayDecision, String> {
        message.validate(lease, now_unix_s)?;
        self.prune_expired(now_unix_s);
        self.observe_with_expiry(message, lease.expires_at_unix_s)
    }

    /// Explicitly reclaim replay state whose authorization lease has expired.
    /// This makes the bounded tracker a reclaimable resource rather than a
    /// permanent accumulation vector across short-lived leases.
    pub fn prune_expired(&mut self, now_unix_s: u64) -> usize {
        let before = self.latest.len();
        self.latest
            .retain(|_, state| now_unix_s < state.expires_at_unix_s);
        before - self.latest.len()
    }

    fn observe_with_expiry(
        &mut self,
        message: &AuthorizedNeurosemanticMessage,
        expires_at_unix_s: u64,
    ) -> Result<ReplayDecision, String> {
        message.packet.validate_integrity()?;
        if message.packet.sequence == 0 {
            return Err("packet sequence must be non-zero".into());
        }

        let key = (
            message.packet.sender_id.clone(),
            message.packet.recipient_id.clone(),
            message.lease_id.clone(),
            message.consent_epoch,
        );

        match self.latest.get(&key) {
            None => {
                if self.latest.len() >= MAX_TRACKED_NEUROSEMANTIC_SESSIONS {
                    return Err("replay tracker capacity exceeded".into());
                }
                self.latest.insert(
                    key,
                    ReplayState {
                        sequence: message.packet.sequence,
                        packet_hash: message.packet.packet_hash.clone(),
                        expires_at_unix_s,
                    },
                );
                Ok(ReplayDecision::Accept)
            }
            Some(state) if message.packet.sequence > state.sequence => {
                self.latest.insert(
                    key,
                    ReplayState {
                        sequence: message.packet.sequence,
                        packet_hash: message.packet.packet_hash.clone(),
                        expires_at_unix_s,
                    },
                );
                Ok(ReplayDecision::Accept)
            }
            Some(state)
                if message.packet.sequence == state.sequence
                    && message.packet.packet_hash == state.packet_hash =>
            {
                Ok(ReplayDecision::Duplicate)
            }
            Some(state) => Err(format!(
                "replay or sequence collision: latest={}, proposed={}",
                state.sequence, message.packet.sequence
            )),
        }
    }
}

fn default_public_sensitivity() -> CognitiveSensitivity {
    CognitiveSensitivity::Public
}

pub fn compute_policy_provenance_hash(provenance_ref: &str, record_bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(NEUROSEMANTIC_POLICY_PROVENANCE_DOMAIN);
    hasher.update(&(provenance_ref.len() as u64).to_le_bytes());
    hasher.update(provenance_ref.as_bytes());
    hasher.update(&(record_bytes.len() as u64).to_le_bytes());
    hasher.update(record_bytes);
    hasher.finalize().to_hex().to_string()
}

impl NeurosemanticDerivationLineageRecord {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic derivation lineage JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let record: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic derivation lineage JSON: {error}"))?;
        record.validate()?;
        Ok(record)
    }

    /// Verify one concrete input artifact against the exact content hash recorded
    /// for that lineage input. The reference identifies the intended entity; the hash
    /// makes the supplied bytes independently checkable.
    pub fn verify_input_artifact_bytes(
        &self,
        index: usize,
        artifact_bytes: &[u8],
    ) -> Result<(), String> {
        self.validate()?;
        let expected_hash = self
            .input_artifact_hashes
            .get(index)
            .ok_or_else(|| "neurosemantic derivation lineage input index out of bounds".to_string())?;
        if artifact_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic derivation lineage input artifact exceeds the serialized artifact limit".into());
        }
        if content_hash(artifact_bytes) != *expected_hash {
            return Err("neurosemantic derivation lineage input artifact hash mismatch".into());
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_DERIVATION_LINEAGE_SCHEMA_VERSION
            || !valid_identifier(&self.lineage_ref)
            || !valid_identifier(&self.activity_ref)
            || !valid_identifier(&self.activity_revision)
            || !valid_blake3_digest(&self.output_artifact_hash)
            || !valid_execution_revision(&self.execution_revision)
            || self.input_artifact_refs.is_empty()
            || self.input_artifact_refs.len() > MAX_NEUROSEMANTIC_DERIVATION_INPUT_ARTIFACTS
            || self.input_artifact_refs.len() != self.input_artifact_hashes.len()
            || self.input_artifact_refs.iter().any(|id| !valid_identifier(id))
            || self.input_artifact_hashes.iter().any(|hash| !valid_blake3_digest(hash))
        {
            return Err("neurosemantic derivation lineage fields are invalid".into());
        }
        let unique_inputs: BTreeSet<&str> =
            self.input_artifact_refs.iter().map(String::as_str).collect();
        if unique_inputs.len() != self.input_artifact_refs.len() {
            return Err("neurosemantic derivation lineage contains duplicate input artifacts".into());
        }
        Ok(())
    }
}

pub fn compute_derivation_provenance_hash(provenance_ref: &str, record_bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(NEUROSEMANTIC_DERIVATION_PROVENANCE_DOMAIN);
    hasher.update(&(provenance_ref.len() as u64).to_le_bytes());
    hasher.update(provenance_ref.as_bytes());
    hasher.update(&(record_bytes.len() as u64).to_le_bytes());
    hasher.update(record_bytes);
    hasher.finalize().to_hex().to_string()
}

pub fn compute_status_source_hash(provenance_ref: &str, record_bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(NEUROSEMANTIC_STATUS_SOURCE_DOMAIN);
    hasher.update(&(provenance_ref.len() as u64).to_le_bytes());
    hasher.update(provenance_ref.as_bytes());
    hasher.update(&(record_bytes.len() as u64).to_le_bytes());
    hasher.update(record_bytes);
    hasher.finalize().to_hex().to_string()
}

fn valid_identifier(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= MAX_NEUROSEMANTIC_ID_BYTES
}

fn valid_blake3_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && value.bytes().any(|byte| byte != b'0')
}

fn valid_execution_revision(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value.bytes().any(|byte| byte != b'0')
}

fn valid_jurisdiction_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 2
        && bytes[0].is_ascii_uppercase()
        && bytes[1].is_ascii_uppercase()
        && bytes.len() <= MAX_NEUROSEMANTIC_JURISDICTION_ID_BYTES
}

fn validate_payload(payload: &NeurosemanticPayload) -> Result<(), String> {
    match payload {
        NeurosemanticPayload::DerivedNeuralFeature(values)
            if values.iter().any(|value| !value.is_finite()) =>
        {
            Err("derived neural features must be finite".into())
        }
        _ => {
            let bytes = serde_json::to_vec(payload)
                .map_err(|error| format!("payload serialization: {error}"))?;
            if bytes.len() > MAX_NEUROSEMANTIC_PAYLOAD_BYTES {
                return Err(format!(
                    "neurosemantic payload exceeds {} bytes",
                    MAX_NEUROSEMANTIC_PAYLOAD_BYTES
                ));
            }
            Ok(())
        }
    }
}

fn sorted_hashes(values: &[String]) -> Vec<&str> {
    let mut sorted: Vec<&str> = values.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    sorted
}

fn recompute_wilson_score_95_interval(
    failure_count: u64,
    observed_count: u64,
    scale: u32,
) -> Result<(i64, i64, u32), String> {
    if observed_count == 0 || failure_count > observed_count || scale > 9 {
        return Err(
            "neurosemantic remediation Wilson interval requires 1..=9 scale and observations"
                .into(),
        );
    }
    let n = observed_count as f64;
    let p = failure_count as f64 / n;
    let z = 1.959963984540054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let radical = (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    let half_width = z * radical / denominator;
    let lower = (center - half_width).clamp(0.0, 1.0);
    let upper = (center + half_width).clamp(0.0, 1.0);
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err("neurosemantic remediation Wilson interval is not finite".into());
    }
    let factor = 10_f64.powi(i32::try_from(scale).map_err(|_| {
        "neurosemantic remediation Wilson scale conversion failed".to_string()
    })?);
    if !factor.is_finite() || factor <= 0.0 {
        return Err("neurosemantic remediation Wilson scale factor is invalid".into());
    }
    let lower_numerator = (lower * factor).floor();
    let upper_numerator = (upper * factor).ceil();
    if lower_numerator < i64::MIN as f64
        || upper_numerator > i64::MAX as f64
    {
        return Err("neurosemantic remediation Wilson interval exceeds fixed-point range".into());
    }
    Ok((lower_numerator as i64, upper_numerator as i64, scale))
}

fn recompute_metric_ratio(
    observation_set: &NeurosemanticRemediationObservationSetArtifact,
    aggregation_ref: &str,
) -> Result<(u64, u64, u64, i128, i128), String> {
    let eligible_count = observation_set.eligible_subject_artifact_hashes.len() as u64;
    let observed_count = observation_set.observations.len() as u64;
    let failure_count = observation_set
        .observations
        .iter()
        .filter(|observation| observation.failure_observed)
        .count() as u64;

    match aggregation_ref {
        "per-item-rate" | "attack-success-rate" | "probe-detection-rate" => {
            if observed_count == 0 {
                return Err("neurosemantic remediation point-estimate computation has no observations".into());
            }
            Ok((
                eligible_count,
                observed_count,
                failure_count,
                failure_count as i128,
                observed_count as i128,
            ))
        }
        "worst-subgroup-gap" => {
            let mut groups: BTreeMap<String, (u64, u64)> = BTreeMap::new();
            for observation in &observation_set.observations {
                let group = observation.group_ref.as_ref().ok_or_else(|| {
                    "neurosemantic remediation subgroup computation requires group identities".to_string()
                })?;
                let entry = groups.entry(group.clone()).or_default();
                entry.0 += u64::from(observation.failure_observed);
                entry.1 += 1;
            }
            if groups.len() < 2 {
                return Err("neurosemantic remediation subgroup gap requires at least two groups".into());
            }
            let mut rates: Vec<(u64, u64)> = groups.values().copied().collect();
            rates.sort_by(|left, right| {
                (left.0 as u128 * right.1 as u128)
                    .cmp(&(right.0 as u128 * left.1 as u128))
            });
            let (min_numerator, min_denominator) = rates
                .first()
                .copied()
                .ok_or_else(|| "neurosemantic remediation subgroup rates are empty".to_string())?;
            let (max_numerator, max_denominator) = rates
                .last()
                .copied()
                .ok_or_else(|| "neurosemantic remediation subgroup rates are empty".to_string())?;
            let gap_numerator =
                max_numerator as i128 * min_denominator as i128
                    - min_numerator as i128 * max_denominator as i128;
            let gap_denominator = max_denominator as i128 * min_denominator as i128;
            Ok((
                eligible_count,
                observed_count,
                failure_count,
                gap_numerator,
                gap_denominator,
            ))
        }
        _ => Err("neurosemantic remediation aggregation is not reproducibly computable at this protocol boundary".into()),
    }
}

fn fixed_point_equals_ratio(
    estimate_numerator: i64,
    estimate_scale: u32,
    ratio_numerator: i128,
    ratio_denominator: i128,
) -> bool {
    if ratio_denominator <= 0 || estimate_numerator < 0 {
        return false;
    }
    let scale_factor = 10_i128.checked_pow(estimate_scale).unwrap_or(0);
    scale_factor > 0
        && estimate_numerator as i128 * ratio_denominator == ratio_numerator * scale_factor
}

fn payload_hash(payload: &NeurosemanticPayload) -> Result<String, String> {
    let bytes = serde_json::to_vec(payload)
        .map_err(|error| format!("payload serialization: {error}"))?;
    Ok(content_hash(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_output_artifact_hash() -> String {
        payload_hash(&NeurosemanticPayload::Hypervector(vec![1, -1])).unwrap()
    }

    fn synthetic_derivation_lineage_record() -> NeurosemanticDerivationLineageRecord {
        NeurosemanticDerivationLineageRecord {
            schema_version: NEUROSEMANTIC_DERIVATION_LINEAGE_SCHEMA_VERSION,
            lineage_ref: "synthetic-derivation-record-1".into(),
            input_artifact_refs: vec!["input-artifact-1".into(), "input-artifact-2".into()],
            input_artifact_hashes: vec![
                content_hash(b"synthetic-input-artifact-1"),
                content_hash(b"synthetic-input-artifact-2"),
            ],
            activity_ref: "synthetic-transform".into(),
            activity_revision: "transform-v1".into(),
            output_artifact_hash: synthetic_output_artifact_hash(),
            execution_revision: "1".repeat(40),
            generated_at_unix_s: 120,
        }
    }

    fn semantic_policy() -> NeurosemanticDataPolicy {
        NeurosemanticDataPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            data_class: NeurosemanticDataClass::SemanticRepresentation,
            inference_classes: BTreeSet::from([NeurosemanticInferenceClass::SemanticContent]),
            permitted_purposes: BTreeSet::from([CommunicationPurpose::HumanCollaboration]),
            handling: NeurosemanticHandlingPolicy {
                schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
                policy_provenance_ref: "synthetic-policy-record-1".into(),
            policy_provenance_hash: compute_policy_provenance_hash(
                "synthetic-policy-record-1",
                b"synthetic-policy-record-1",
            ),
            derivation_provenance_ref: "synthetic-derivation-record-1".into(),
            derivation_provenance_hash: compute_derivation_provenance_hash(
                "synthetic-derivation-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
            ),
            derivation_output_artifact_hash: synthetic_output_artifact_hash(),
            origin_jurisdiction: "ZA".into(),
                permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
                permitted_secondary_uses: BTreeSet::new(),
                retention: NeurosemanticRetentionPolicy::UntilUnixS(200),
                max_authority_resolution_age_s: 600,
            },
        }
    }

    fn binding_context() -> NeurosemanticConsentBindingContext {
        NeurosemanticConsentBindingContext {
            subject_ref: "subject".into(),
            peer_ref: "peer".into(),
            lease_id: "lease-1".into(),
            consent_epoch: 7,
            consent_lease_fingerprint: lease().fingerprint_for_authorization().unwrap(),
            purpose: CommunicationPurpose::HumanCollaboration,
            channel: CognitiveChannel::Semantic,
            direction: ChannelDirection::Write,
        }
    }

    fn authority_resolution(
        policy: &NeurosemanticDataPolicy,
        attestation: &NeurosemanticPolicyAuthorityAttestation,
        context: &NeurosemanticConsentBindingContext,
        checked_at_unix_s: u64,
        expires_at_unix_s: u64,
    ) -> (NeurosemanticAuthorityResolutionAttestation, VerifyingKey) {
        use ed25519_dalek::{Signer, SigningKey};
        let signing_key = SigningKey::from_bytes(&[9u8; 32]);
        let mut resolution = NeurosemanticAuthorityResolutionAttestation {
            schema_version: NEUROSEMANTIC_AUTHORITY_RESOLUTION_SCHEMA_VERSION,
            resolver_ref: "mycelix-identity-policy-bridge".into(),
            resolver_key_ref: "resolver-key-1".into(),
            authority_ref: attestation.authority_ref.clone(),
            authority_key_ref: attestation.key_ref.clone(),
            authority_attestation_fingerprint: attestation.fingerprint_for_attestation().unwrap(),
            handling_policy_fingerprint: policy.handling.fingerprint_for_attestation().unwrap(),
            policy_provenance_ref: policy.handling.policy_provenance_ref.clone(),
            policy_provenance_hash: policy.handling.policy_provenance_hash.clone(),
            subject_ref: context.subject_ref.clone(),
            peer_ref: context.peer_ref.clone(),
            lease_id: context.lease_id.clone(),
            consent_epoch: context.consent_epoch,
            consent_lease_fingerprint: context.consent_lease_fingerprint.clone(),
            purpose: context.purpose,
            channel: context.channel,
            direction: context.direction,
            status: NeurosemanticAuthorityStatus::Active,
            status_source_ref: "mycelix-status:synthetic-1".into(),
            status_source_hash: compute_status_source_hash(
                "mycelix-status:synthetic-1",
                b"synthetic-status-record-1",
            ),
            checked_at_unix_s,
            expires_at_unix_s,
            signature: Vec::new(),
        };
        resolution.signature =
            signing_key.sign(&resolution.message_bytes().unwrap()).to_bytes().to_vec();
        (resolution, signing_key.verifying_key())
    }

    fn policy_provenance_binding() -> NeurosemanticPolicyProvenanceBinding {
        let policy = semantic_policy();
        let (attestation, verifying_key) = authority_attestation(&policy);
        let context = binding_context();
        let (resolution, resolution_key) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);
        policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &attestation,
                &verifying_key,
                &resolution,
                &resolution_key,
                &context,
                150,
            )
            .unwrap()
    }

    fn synthetic_lifecycle_receipt_chain() -> (
        NeurosemanticArtifactLifecycleReceipt,
        NeurosemanticArtifactLifecycleReceipt,
        NeurosemanticArtifactLifecycleReceipt,
        NeurosemanticArtifactLifecycleReceipt,
        NeurosemanticArtifactLifecycleReceipt,
        Vec<u8>,
        Vec<u8>,
        NeurosemanticDerivationLineageRecord,
        Vec<u8>,
    ) {
        let evidence = b"synthetic-lifecycle-effect-v1".to_vec();
        let lineage = synthetic_derivation_lineage_record();
        let derivation_hash = compute_derivation_provenance_hash(
            &lineage.lineage_ref,
            &serde_json::to_vec(&lineage).unwrap(),
        );
        let base = NeurosemanticArtifactLifecycleReceipt {
            schema_version: NEUROSEMANTIC_ARTIFACT_LIFECYCLE_RECEIPT_SCHEMA_VERSION,
            receipt_ref: "synthetic-lifecycle-receipt-1".into(),
            artifact_hash: lineage.output_artifact_hash.clone(),
            derivation_provenance_ref: lineage.lineage_ref.clone(),
            derivation_provenance_hash: derivation_hash.clone(),
            event_sequence: 0,
            previous_receipt_hash: None,
            action: NeurosemanticArtifactLifecycleAction::Erasure,
            state: NeurosemanticArtifactLifecycleState::Requested,
            effect_evidence_ref: None,
            effect_evidence_hash: None,
            verification_agent_ref: None,
            verification_evidence_ref: None,
            verification_evidence_hash: None,
            verification_target_effect_evidence_hash: None,
            verification_scope: None,
            verification_scope_ref: None,
            verification_scope_hash: None,
            execution_revision: "2".repeat(40),
            observed_at_unix_s: 158,
            resulting_artifact_hash: None,
            resulting_derivation_provenance_ref: None,
            resulting_derivation_provenance_hash: None,
        };
        let accepted = NeurosemanticArtifactLifecycleReceipt {
            receipt_ref: "synthetic-lifecycle-receipt-2".into(),
            event_sequence: 1,
            previous_receipt_hash: Some(base.fingerprint().unwrap()),
            state: NeurosemanticArtifactLifecycleState::Accepted,
            observed_at_unix_s: 159,
            ..base.clone()
        };
        let processing = NeurosemanticArtifactLifecycleReceipt {
            receipt_ref: "synthetic-lifecycle-receipt-3".into(),
            event_sequence: 2,
            previous_receipt_hash: Some(accepted.fingerprint().unwrap()),
            state: NeurosemanticArtifactLifecycleState::Processing,
            observed_at_unix_s: 160,
            ..accepted.clone()
        };
        let effect_evidence_hash = content_hash(&evidence);
        let applied = NeurosemanticArtifactLifecycleReceipt {
            receipt_ref: "synthetic-lifecycle-receipt-4".into(),
            event_sequence: 3,
            previous_receipt_hash: Some(processing.fingerprint().unwrap()),
            state: NeurosemanticArtifactLifecycleState::Applied,
            effect_agent_ref: Some("synthetic-effect-worker-1".into()),
            effect_evidence_ref: Some("synthetic-lifecycle-effect-1".into()),
            effect_evidence_hash: Some(effect_evidence_hash.clone()),
            observed_at_unix_s: 161,
            ..processing.clone()
        };
        let verification_evidence = b"synthetic-independent-verification-v1".to_vec();
        let verified = NeurosemanticArtifactLifecycleReceipt {
            receipt_ref: "synthetic-lifecycle-receipt-5".into(),
            event_sequence: 4,
            previous_receipt_hash: Some(applied.fingerprint().unwrap()),
            state: NeurosemanticArtifactLifecycleState::IndependentlyVerified,
            effect_evidence_ref: applied.effect_evidence_ref.clone(),
            effect_evidence_hash: applied.effect_evidence_hash.clone(),
            verification_agent_ref: Some("synthetic-independent-verifier-1".into()),
            verification_evidence_ref: Some("synthetic-independent-verification-1".into()),
            verification_evidence_hash: Some(content_hash(&verification_evidence)),
            verification_target_effect_evidence_hash: Some(effect_evidence_hash),
            verification_scope: Some(NeurosemanticArtifactLifecycleVerificationScope::ArtifactOnly),
            observed_at_unix_s: 162,
            ..applied.clone()
        };
        let replacement_hash = content_hash(b"replacement-artifact");
        let replacement_lineage = NeurosemanticDerivationLineageRecord {
            lineage_ref: "synthetic-replacement-lineage".into(),
            output_artifact_hash: replacement_hash,
            ..lineage.clone()
        };
        let replacement_bytes = serde_json::to_vec(&replacement_lineage).unwrap();
        (base, accepted, processing, applied, verified, evidence, verification_evidence, replacement_lineage, replacement_bytes)
    }

    #[test]
    fn lifecycle_receipt_binds_artifact_lineage_effect_and_independent_verification() {
        let (requested, accepted, processing, applied, verified, evidence, verification_evidence, _, _) =
            synthetic_lifecycle_receipt_chain();
        let lineage = synthetic_derivation_lineage_record();
        let lineage_bytes = serde_json::to_vec(&lineage).unwrap();
        assert!(verified.validate().is_ok());
        assert_eq!(
            NeurosemanticArtifactLifecycleReceipt::from_json_bytes(
                &serde_json::to_vec(&verified).unwrap()
            ).unwrap(),
            verified
        );
        assert!(verified.verify_effect_evidence_bytes(&evidence));
        assert!(verified.verify_independent_verification_bytes(&verification_evidence, 170).is_ok());
        assert!(verified.verify_binding(
            &lineage.output_artifact_hash,
            &lineage.lineage_ref,
            &compute_derivation_provenance_hash(&lineage.lineage_ref, &lineage_bytes),
            &evidence,
            170,
        ).is_ok());
        assert!(accepted.verify_transition(&requested).is_ok());
        assert!(processing.verify_transition(&accepted).is_ok());
        assert!(applied.verify_transition(&processing).is_ok());
        assert!(verified.verify_transition(&applied).is_ok());
        assert!(!verified.verify_effect_evidence_bytes(b"synthetic-lifecycle-effect-tampered"));
        assert!(verified.verify_independent_verification_bytes(
            b"synthetic-independent-verification-tampered",
            170
        ).is_err());
        let mut forged = verified.clone();
        forged.previous_receipt_hash = Some(content_hash(b"wrong-previous-receipt"));
        assert!(forged.verify_transition(&applied).is_err());
    }

    #[test]
    fn lifecycle_receipt_rejects_future_ambiguous_replacement_and_unverified_verification_claims() {
        let (requested, accepted, processing, applied, mut receipt, evidence, verification_evidence, replacement_lineage, replacement_bytes) =
            synthetic_lifecycle_receipt_chain();
        assert!(receipt.verify_binding(
            &receipt.artifact_hash,
            &receipt.derivation_provenance_ref,
            &receipt.derivation_provenance_hash,
            &evidence,
            159,
        ).is_err());
        assert!(receipt.verify_transition(&applied).is_ok());
        assert!(applied.verify_transition(&processing).is_ok());
        assert!(processing.verify_transition(&accepted).is_ok());
        assert!(accepted.verify_transition(&requested).is_ok());

        let mut unbound = receipt.clone();
        unbound.verification_target_effect_evidence_hash = Some(content_hash(b"other-effect"));
        assert!(unbound.validate().is_err());

        let mut missing_verifier = receipt.clone();
        missing_verifier.verification_agent_ref = None;
        assert!(missing_verifier.validate().is_err());

        let mut same_agent = receipt.clone();
        same_agent.verification_agent_ref = same_agent.effect_agent_ref.clone();
        assert!(same_agent.validate().is_err());

        let mut substituted_effect = receipt.clone();
        substituted_effect.effect_evidence_hash = Some(content_hash(b"other-effect"));
        assert!(substituted_effect.verify_transition(&applied).is_err());

        let mut invalid_effect_agent = receipt.clone();
        invalid_effect_agent.effect_agent_ref = Some(String::new());
        assert!(invalid_effect_agent.validate().is_err());

        receipt.action = NeurosemanticArtifactLifecycleAction::Rectification;
        receipt.resulting_artifact_hash = None;
        receipt.resulting_derivation_provenance_ref = None;
        receipt.resulting_derivation_provenance_hash = None;
        assert!(receipt.validate().is_err());

        receipt.resulting_artifact_hash = Some(receipt.artifact_hash.clone());
        receipt.resulting_derivation_provenance_ref = Some(replacement_lineage.lineage_ref.clone());
        receipt.resulting_derivation_provenance_hash = Some(content_hash(&replacement_bytes));
        assert!(receipt.validate().is_err());

        receipt.resulting_artifact_hash = Some(replacement_lineage.output_artifact_hash.clone());
        receipt.resulting_derivation_provenance_ref = Some(replacement_lineage.lineage_ref.clone());
        receipt.resulting_derivation_provenance_hash = Some(
            compute_derivation_provenance_hash(&replacement_lineage.lineage_ref, &replacement_bytes)
        );
        assert!(receipt.validate().is_ok());
        assert!(receipt.verify_resulting_lineage_binding_bytes(&replacement_bytes).is_ok());

        let mut forged_lineage = replacement_lineage.clone();
        forged_lineage.output_artifact_hash = content_hash(b"other-artifact");
        let forged_bytes = serde_json::to_vec(&forged_lineage).unwrap();
        assert!(receipt.verify_resulting_lineage_binding_bytes(&forged_bytes).is_err());

        let _ = verification_evidence;
    }

    #[test]
    fn lifecycle_receipt_bounded_parser_and_schema_are_fail_closed() {
        let (_, _, _, _, receipt, _, _, _, _) = synthetic_lifecycle_receipt_chain();
        let mut value = serde_json::to_value(&receipt).unwrap();
        value.as_object_mut()
            .unwrap()
            .insert(
                "schema_version".into(),
                serde_json::json!(NEUROSEMANTIC_ARTIFACT_LIFECYCLE_RECEIPT_SCHEMA_VERSION - 1),
            );
        assert!(NeurosemanticArtifactLifecycleReceipt::from_json_bytes(
            &serde_json::to_vec(&value).unwrap()
        ).is_err());
        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticArtifactLifecycleReceipt::from_json_bytes(&oversized).is_err());
    }

    #[test]
    fn lifecycle_verification_target_set_is_machine_validated() {
        let root = content_hash(b"root-artifact");
        let target = content_hash(b"descendant-artifact");
        let target_set = NeurosemanticArtifactLifecycleVerificationTargetSet {
            schema_version: NEUROSEMANTIC_ARTIFACT_LIFECYCLE_VERIFICATION_SCOPE_SCHEMA_VERSION,
            scope_ref: "synthetic-scope-1".into(),
            root_artifact_hash: root.clone(),
            target_artifact_hashes: vec![root.clone(), target],
        };
        let encoded = serde_json::to_vec(&target_set).unwrap();
        assert!(NeurosemanticArtifactLifecycleVerificationTargetSet::from_json_bytes(&encoded).is_ok());

        let mut duplicate = target_set.clone();
        duplicate.target_artifact_hashes.push(root);
        assert!(duplicate.validate().is_err());

        let mut missing_root = target_set.clone();
        missing_root.target_artifact_hashes = vec![content_hash(b"other")];
        assert!(missing_root.validate().is_err());

        let mut wrong_schema = target_set.clone();
        wrong_schema.schema_version = 0;
        assert!(wrong_schema.validate().is_err());

        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticArtifactLifecycleVerificationTargetSet::from_json_bytes(&oversized).is_err());
    }

    #[test]
    fn remediation_evaluation_set_fingerprint_is_order_insensitive() {
        let mut manifest = NeurosemanticRemediationEvaluationSetManifest {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_SET_SCHEMA_VERSION,
            set_ref: "set-order-test".into(),
            set_kind: NeurosemanticRemediationEvaluationSetKind::Forget,
            source_dataset_manifest_hash: content_hash(b"dataset"),
            member_artifact_hashes: vec![content_hash(b"b"), content_hash(b"a")],
        };
        let first = manifest.fingerprint().unwrap();
        manifest.member_artifact_hashes.reverse();
        let second = manifest.fingerprint().unwrap();
        assert_eq!(first, second);
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn remediation_impact_artifact_binds_model_lineage_and_evidence_dimensions() {
        let pre_model_hash = content_hash(b"model-pre");
        let post_model_hash = content_hash(b"model-post");
        let pre_lineage = NeurosemanticDerivationLineageRecord {
            lineage_ref: "model-lineage-pre".into(),
            output_artifact_hash: pre_model_hash.clone(),
            ..synthetic_derivation_lineage_record()
        };
        let post_lineage = NeurosemanticDerivationLineageRecord {
            lineage_ref: "model-lineage-post".into(),
            output_artifact_hash: post_model_hash.clone(),
            ..synthetic_derivation_lineage_record()
        };
        let pre_bytes = serde_json::to_vec(&pre_lineage).unwrap();
        let post_bytes = serde_json::to_vec(&post_lineage).unwrap();
        let lifecycle = synthetic_lifecycle_receipt_chain().4;
        let source_dataset_manifest_hash = content_hash(b"source-dataset");
        let forget_set = NeurosemanticRemediationEvaluationSetManifest {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_SET_SCHEMA_VERSION,
            set_ref: "forget-set".into(),
            set_kind: NeurosemanticRemediationEvaluationSetKind::Forget,
            source_dataset_manifest_hash: source_dataset_manifest_hash.clone(),
            member_artifact_hashes: vec![content_hash(b"forget-1")],
        };
        let retain_set = NeurosemanticRemediationEvaluationSetManifest {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_SET_SCHEMA_VERSION,
            set_ref: "retain-set".into(),
            set_kind: NeurosemanticRemediationEvaluationSetKind::Retain,
            source_dataset_manifest_hash,
            member_artifact_hashes: vec![content_hash(b"retain-1")],
        };
        let forget_bytes = serde_json::to_vec(&forget_set).unwrap();
        let retain_bytes = serde_json::to_vec(&retain_set).unwrap();
        let protocol_bytes = b"protocol-1";
        let recovery_method = NeurosemanticRemediationEvaluationMethod {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_METHOD_SCHEMA_VERSION,
            method_ref: "recovery-method".into(),
            kind: NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack,
            protocol_hash: content_hash(protocol_bytes),
            implementation_revision: "3".repeat(40),
        };
        let representation_method = NeurosemanticRemediationEvaluationMethod {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_METHOD_SCHEMA_VERSION,
            method_ref: "representation-method".into(),
            kind: NeurosemanticRemediationEvaluationMethodKind::RepresentationResidualProbe,
            protocol_hash: content_hash(protocol_bytes),
            implementation_revision: "3".repeat(40),
        };
        let recovery_bytes = serde_json::to_vec(&recovery_method).unwrap();
        let representation_bytes = serde_json::to_vec(&representation_method).unwrap();
        let evaluation_manifest = NeurosemanticRemediationEvaluationManifest {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_MANIFEST_SCHEMA_VERSION,
            evaluation_ref: "evaluation-1".into(),
            source_dataset_manifest_hash: forget_set.source_dataset_manifest_hash.clone(),
            forget_set_manifest_hash: forget_set.fingerprint().unwrap(),
            retain_set_manifest_hash: retain_set.fingerprint().unwrap(),
            study_protocol_hash: content_hash(protocol_bytes),
            evaluation_split_manifest_hash: content_hash(b"split-1"),
            recovery_method_hash: recovery_method.fingerprint().unwrap(),
            representation_probe_method_hash: representation_method.fingerprint().unwrap(),
        };
        let evaluation_manifest_bytes = serde_json::to_vec(&evaluation_manifest).unwrap();
        let evaluation_environment = NeurosemanticRemediationEvaluationEnvironment {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_ENVIRONMENT_SCHEMA_VERSION,
            environment_ref: "environment-1".into(),
            platform_ref: "linux-x86_64".into(),
            runtime_ref: "rust-runtime".into(),
            toolchain_ref: "rust-1.96".into(),
            dependency_lock_hash: content_hash(b"deps"),
            configuration_hash: content_hash(b"config"),
            execution_revision: "2".repeat(40),
        };
        let evaluation_environment_bytes = serde_json::to_vec(&evaluation_environment).unwrap();
        let measurement = NeurosemanticRemediationMeasurementArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_MEASUREMENT_SCHEMA_VERSION,
            measurement_ref: "measurement-1".into(),
            metric_definitions: vec![
                NeurosemanticRemediationMetricDefinition {
                    schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
                    metric_ref: "metric-forgetfulness".into(),
                    kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
                    estimand_ref: "forgetfulness-on-forget-set".into(),
                    scope_ref: "forget-set-v1".into(),
                    unit_ref: "proportion".into(),
                    aggregation_ref: "per-item-rate".into(),
                    direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
                },
                NeurosemanticRemediationMetricDefinition {
                    schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
                    metric_ref: "metric-utility".into(),
                    kind: NeurosemanticRemediationMeasurementKind::UtilityImpact,
                    estimand_ref: "utility-on-retain-set".into(),
                    scope_ref: "retain-set-v1".into(),
                    unit_ref: "proportion".into(),
                    aggregation_ref: "per-item-rate".into(),
                    direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
                },
                NeurosemanticRemediationMetricDefinition {
                    schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
                    metric_ref: "metric-recovery".into(),
                    kind: NeurosemanticRemediationMeasurementKind::RecoveryRisk,
                    estimand_ref: "recovery-risk-on-forget-set".into(),
                    scope_ref: "forget-set-v1".into(),
                    unit_ref: "proportion".into(),
                    aggregation_ref: "attack-success-rate".into(),
                    direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
                },
                NeurosemanticRemediationMetricDefinition {
                    schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
                    metric_ref: "metric-representation".into(),
                    kind: NeurosemanticRemediationMeasurementKind::RepresentationResidual,
                    estimand_ref: "representation-residual-on-forget-set".into(),
                    scope_ref: "forget-set-v1".into(),
                    unit_ref: "proportion".into(),
                    aggregation_ref: "probe-detection-rate".into(),
                    direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
                },
                NeurosemanticRemediationMetricDefinition {
                    schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
                    metric_ref: "metric-fairness".into(),
                    kind: NeurosemanticRemediationMeasurementKind::FairnessImpact,
                    estimand_ref: "fairness-impact-on-declared-subgroups".into(),
                    scope_ref: "fairness-split-v1".into(),
                    unit_ref: "proportion".into(),
                    aggregation_ref: "worst-subgroup-gap".into(),
                    direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
                },
            ],
            metric_computation_artifact_hashes: vec![
                content_hash(b"integration-computation-forgetfulness"),
                content_hash(b"integration-computation-utility"),
                content_hash(b"integration-computation-recovery"),
                content_hash(b"integration-computation-representation"),
                content_hash(b"integration-computation-fairness"),
            ],
            measurements: vec![
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-forgetfulness".into(),
                    kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 2,
                    observed_sample_count: 2,
                    failure_count: 0,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-utility".into(),
                    kind: NeurosemanticRemediationMeasurementKind::UtilityImpact,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 2,
                    observed_sample_count: 2,
                    failure_count: 0,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-recovery".into(),
                    kind: NeurosemanticRemediationMeasurementKind::RecoveryRisk,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 2,
                    observed_sample_count: 2,
                    failure_count: 0,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-representation".into(),
                    kind: NeurosemanticRemediationMeasurementKind::RepresentationResidual,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 2,
                    observed_sample_count: 2,
                    failure_count: 0,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-fairness".into(),
                    kind: NeurosemanticRemediationMeasurementKind::FairnessImpact,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 2,
                    observed_sample_count: 2,
                    failure_count: 0,
                },
            ],
            worst_case_disposition: NeurosemanticRemediationImpactDisposition::Inconclusive,
        };
        let measurement_bytes = serde_json::to_vec(&measurement).unwrap();
        let impact = NeurosemanticRemediationImpactArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_IMPACT_ARTIFACT_SCHEMA_VERSION,
            impact_ref: "impact-1".into(),
            pre_remediation_model_hash: pre_model_hash.clone(),
            post_remediation_model_hash: post_model_hash.clone(),
            pre_remediation_lineage_ref: pre_lineage.lineage_ref.clone(),
            pre_remediation_lineage_hash: compute_derivation_provenance_hash(&pre_lineage.lineage_ref, &pre_bytes),
            post_remediation_lineage_ref: post_lineage.lineage_ref.clone(),
            post_remediation_lineage_hash: compute_derivation_provenance_hash(&post_lineage.lineage_ref, &post_bytes),
            lifecycle_receipt_hash: lifecycle.fingerprint().unwrap(),
            evaluation_manifest_hash: evaluation_manifest.fingerprint().unwrap(),
            measurement_artifact_hash: measurement.fingerprint().unwrap(),
            evaluation_agent_ref: "evaluation-agent-1".into(),
            evaluation_verifier_ref: "evaluation-verifier-1".into(),
            evaluation_verification_evidence_hash: content_hash(b"evaluation-verification"),
            evaluation_environment_hash: evaluation_environment.fingerprint().unwrap(),
            remediation_action: NeurosemanticArtifactLifecycleAction::Erasure,
            study_protocol_hash: content_hash(protocol_bytes),
            evaluation_split_manifest_hash: content_hash(b"split-1"),
            forget_set_manifest_hash: forget_set.fingerprint().unwrap(),
            retain_set_manifest_hash: retain_set.fingerprint().unwrap(),
            recovery_method_hash: recovery_method.fingerprint().unwrap(),
            representation_probe_method_hash: representation_method.fingerprint().unwrap(),
            forget_evidence_hash: content_hash(b"forget"),
            utility_impact_evidence_hash: content_hash(b"utility"),
            recovery_evidence_hash: content_hash(b"recovery"),
            representation_residual_evidence_hash: content_hash(b"representation"),
            fairness_impact_evidence_hash: Some(content_hash(b"fairness")),
            residual_risk_evidence_hash: content_hash(b"residual"),
            execution_revision: "2".repeat(40),
            observed_at_unix_s: 170,
            disposition: NeurosemanticRemediationImpactDisposition::Inconclusive,
            dimensions: vec![
                "forgetfulness".into(),
                "utility-impact".into(),
                "fairness-impact".into(),
                "residual-risk".into(),
                "forget-set".into(),
                "retain-set".into(),
                "recovery-attack".into(),
                "representation-residual".into(),
            ],
        };
        assert!(impact.validate().is_ok());
        assert!(impact.verify_lifecycle_binding(&lifecycle).is_ok());
        assert!(impact.verify_evaluation_manifest_bytes(&evaluation_manifest_bytes).is_ok());
        assert_eq!(
            impact.verify_measurement_artifact_bytes(&measurement_bytes).unwrap(),
            measurement
        );
        let mut impact_status_forged = impact.clone();
        impact_status_forged.disposition =
            NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds;
        assert!(impact_status_forged
            .verify_measurement_artifact_bytes(&measurement_bytes)
            .is_err());
        let mut measurement_subset = measurement.clone();
        measurement_subset.measurements.retain(|item| {
            item.kind != NeurosemanticRemediationMeasurementKind::RepresentationResidual
        });
        measurement_subset.worst_case_disposition =
            measurement_subset.recomputed_worst_case_disposition().unwrap();
        assert!(
            impact
                .verify_measurement_artifact_bytes(
                    &serde_json::to_vec(&measurement_subset).unwrap()
                )
                .is_err()
        );
        assert!(impact
            .verify_evaluation_verification_evidence_bytes(
                "evaluation-verifier-1",
                b"evaluation-verification",
            )
            .is_ok());
        assert!(impact
            .verify_evaluation_environment_bytes(&evaluation_environment_bytes)
            .is_ok());
        let mut role_collision = impact.clone();
        role_collision.evaluation_verifier_ref = lifecycle.verification_agent_ref.clone().unwrap();
        assert!(role_collision.verify_lifecycle_binding(&lifecycle).is_err());
        let mut evaluator_collision = impact.clone();
        evaluator_collision.evaluation_agent_ref = lifecycle.effect_agent_ref.clone().unwrap();
        assert!(evaluator_collision.verify_lifecycle_binding(&lifecycle).is_err());
        assert!(impact
            .verify_evaluation_verification_evidence_bytes(
                "evaluation-agent-1",
                b"evaluation-verification",
            )
            .is_err());
        assert!(impact
            .verify_evaluation_verification_evidence_bytes(
                "evaluation-verifier-1",
                b"tampered-evaluation-verification",
            )
            .is_err());
        assert!(impact
            .verify_evaluation_environment_bytes(b"other-environment")
            .is_err());
        let mut environment_revision = evaluation_environment.clone();
        environment_revision.execution_revision = "4".repeat(40);
        assert!(impact
            .verify_evaluation_environment_bytes(
                &serde_json::to_vec(&environment_revision).unwrap()
            )
            .is_err());
        let mut measurement_status_forged = measurement.clone();
        measurement_status_forged.measurements[0].status =
            NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds;
        assert!(measurement_status_forged.validate().is_err());
        let mut measurement_count_forged = measurement.clone();
        measurement_count_forged.measurements[0].failure_count = 3;
        assert!(measurement_count_forged.validate().is_err());
        assert!(impact.verify_lineage_bytes(NeurosemanticRemediationImpactLineageSide::PreRemediation, &pre_bytes).is_ok());
        assert!(impact.verify_lineage_bytes(NeurosemanticRemediationImpactLineageSide::PostRemediation, &post_bytes).is_ok());
        assert!(impact.verify_evaluation_set_pair_bytes(&forget_bytes, &retain_bytes).is_ok());
        assert!(impact.verify_evaluation_bundle_identity(
            &evaluation_manifest_bytes,
            &forget_bytes,
            &retain_bytes,
        ).is_ok());
        let mut source_swap = retain_set.clone();
        source_swap.source_dataset_manifest_hash = content_hash(b"other-dataset");
        assert!(impact.verify_evaluation_bundle_identity(
            &evaluation_manifest_bytes,
            &forget_bytes,
            &serde_json::to_vec(&source_swap).unwrap(),
        ).is_err());
        assert!(impact.verify_evaluation_method_bytes(NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack, &recovery_bytes).is_ok());
        assert!(impact.verify_evaluation_method_bytes(NeurosemanticRemediationEvaluationMethodKind::RepresentationResidualProbe, &representation_bytes).is_ok());
        assert!(impact.verify_study_protocol_bytes(protocol_bytes).is_ok());
        assert!(impact.verify_evaluation_split_manifest_bytes(b"split-1").is_ok());
        assert!(impact.verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::Forgetfulness, b"forget").is_ok());
        assert!(impact.verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::UtilityImpact, b"utility").is_ok());
        assert!(impact.verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::FairnessImpact, b"fairness").is_ok());
        assert!(impact.verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::ResidualRisk, b"residual").is_ok());
        assert!(impact.verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::RecoveryRisk, b"recovery").is_ok());
        assert!(impact.verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::RepresentationResidual, b"representation").is_ok());

        let mut evaluation_swap = evaluation_manifest.clone();
        evaluation_swap.retain_set_manifest_hash = content_hash(b"other-retain-set");
        assert!(impact.verify_evaluation_manifest_bytes(&serde_json::to_vec(&evaluation_swap).unwrap()).is_err());

        let mut overlap = retain_set.clone();
        overlap.member_artifact_hashes = vec![forget_set.member_artifact_hashes[0].clone()];
        assert!(impact.verify_evaluation_set_pair_bytes(&forget_bytes, &serde_json::to_vec(&overlap).unwrap()).is_err());

        let mut role_swap = forget_set.clone();
        role_swap.set_kind = NeurosemanticRemediationEvaluationSetKind::Retain;
        assert!(impact.verify_evaluation_set_manifest_bytes(NeurosemanticRemediationEvaluationSetKind::Forget, &serde_json::to_vec(&role_swap).unwrap()).is_err());

        let mut method_swap = recovery_method.clone();
        method_swap.kind = NeurosemanticRemediationEvaluationMethodKind::RepresentationResidualProbe;
        assert!(impact.verify_evaluation_method_bytes(NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack, &serde_json::to_vec(&method_swap).unwrap()).is_err());

        let mut protocol_swap = recovery_method.clone();
        protocol_swap.protocol_hash = content_hash(b"other-protocol");
        assert!(impact.verify_evaluation_method_bytes(NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack, &serde_json::to_vec(&protocol_swap).unwrap()).is_err());

        let mut revision_swap = recovery_method.clone();
        revision_swap.implementation_revision = "4".repeat(40);
        assert!(impact.verify_evaluation_method_bytes(NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack, &serde_json::to_vec(&revision_swap).unwrap()).is_err());

        let mut legacy = impact.clone();
        legacy.schema_version = 1;
        assert!(NeurosemanticRemediationImpactArtifact::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).is_err());

        let mut invalid_roles = impact.clone();
        invalid_roles.evaluation_agent_ref.clear();
        assert!(invalid_roles.validate().is_err());
    }
    #[test]
    fn remediation_evaluation_manifest_schema_is_fail_closed() {
        let manifest = NeurosemanticRemediationEvaluationManifest {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_MANIFEST_SCHEMA_VERSION,
            evaluation_ref: "evaluation-schema-test".into(),
            source_dataset_manifest_hash: content_hash(b"dataset"),
            forget_set_manifest_hash: content_hash(b"forget-set"),
            retain_set_manifest_hash: content_hash(b"retain-set"),
            study_protocol_hash: content_hash(b"protocol"),
            evaluation_split_manifest_hash: content_hash(b"split"),
            recovery_method_hash: content_hash(b"recovery"),
            representation_probe_method_hash: content_hash(b"representation"),
        };
        let encoded = serde_json::to_vec(&manifest).unwrap();
        assert_eq!(NeurosemanticRemediationEvaluationManifest::from_json_bytes(&encoded).unwrap(), manifest);

        let mut legacy = manifest.clone();
        legacy.schema_version = 0;
        assert!(NeurosemanticRemediationEvaluationManifest::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).is_err());

        let mut ambiguous = manifest.clone();
        ambiguous.forget_set_manifest_hash = ambiguous.retain_set_manifest_hash.clone();
        assert!(ambiguous.validate().is_err());

        let mut same_method = manifest.clone();
        same_method.recovery_method_hash = same_method.representation_probe_method_hash.clone();
        assert!(same_method.validate().is_err());

        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticRemediationEvaluationManifest::from_json_bytes(&oversized).is_err());
    }

    #[test]
    fn remediation_evaluation_environment_schema_is_fail_closed() {
        let environment = NeurosemanticRemediationEvaluationEnvironment {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_ENVIRONMENT_SCHEMA_VERSION,
            environment_ref: "environment-schema-test".into(),
            platform_ref: "linux-x86_64".into(),
            runtime_ref: "rust-runtime".into(),
            toolchain_ref: "rust-1.96".into(),
            dependency_lock_hash: content_hash(b"deps"),
            configuration_hash: content_hash(b"config"),
            execution_revision: "6".repeat(40),
        };
        let encoded = serde_json::to_vec(&environment).unwrap();
        assert_eq!(NeurosemanticRemediationEvaluationEnvironment::from_json_bytes(&encoded).unwrap(), environment);
        let mut legacy = environment.clone();
        legacy.schema_version = 0;
        assert!(NeurosemanticRemediationEvaluationEnvironment::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).is_err());
        let mut bad_revision = environment.clone();
        bad_revision.execution_revision = "0".repeat(40);
        assert!(bad_revision.validate().is_err());
        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticRemediationEvaluationEnvironment::from_json_bytes(&oversized).is_err());
    }

    #[test]
    fn remediation_evaluation_method_schema_is_fail_closed() {
        let method = NeurosemanticRemediationEvaluationMethod {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_METHOD_SCHEMA_VERSION,
            method_ref: "recovery-method-schema-test".into(),
            kind: NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack,
            protocol_hash: content_hash(b"protocol"),
            implementation_revision: "5".repeat(40),
        };
        assert!(NeurosemanticRemediationEvaluationMethod::from_json_bytes(&serde_json::to_vec(&method).unwrap()).is_ok());
        let mut invalid = method.clone();
        invalid.method_ref = String::new();
        assert!(invalid.validate().is_err());
        let mut future = method.clone();
        future.implementation_revision = "0".repeat(40);
        assert!(future.validate().is_err());
    }
    #[test]
    fn remediation_metric_definition_and_uncertainty_are_typed() {
        let definition = NeurosemanticRemediationMetricDefinition {
            schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
            metric_ref: "metric-typed".into(),
            kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
            estimand_ref: "forgetfulness-on-forget-set".into(),
            scope_ref: "forget-set-v1".into(),
            unit_ref: "proportion".into(),
            aggregation_ref: "per-item-rate".into(),
            direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
        };
        let artifact = NeurosemanticRemediationMeasurementArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_MEASUREMENT_SCHEMA_VERSION,
            measurement_ref: "measurement-typed".into(),
            metric_definitions: vec![definition],
            metric_computation_artifact_hashes: vec![
                content_hash(b"typed-computation-forgetfulness"),
                content_hash(b"typed-computation-utility"),
                content_hash(b"typed-computation-recovery"),
                content_hash(b"typed-computation-representation"),
            ],
            measurements: vec![
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-typed".into(),
                    kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::Interval {
                        lower_numerator: -1,
                        upper_numerator: 1,
                        scale: 4,
                        confidence_level_bps: 9500,
                        uncertainty_method_ref: "synthetic-structural-interval-v1".into(),
                        uncertainty_computation_artifact_hash: content_hash(b"typed-uncertainty-computation"),
                    },
                    eligible_sample_count: 10,
                    observed_sample_count: 10,
                    failure_count: 0,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-utility".into(),
                    kind: NeurosemanticRemediationMeasurementKind::UtilityImpact,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 10,
                    observed_sample_count: 10,
                    failure_count: 0,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-recovery".into(),
                    kind: NeurosemanticRemediationMeasurementKind::RecoveryRisk,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 10,
                    observed_sample_count: 10,
                    failure_count: 0,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-representation".into(),
                    kind: NeurosemanticRemediationMeasurementKind::RepresentationResidual,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 10,
                    observed_sample_count: 10,
                    failure_count: 0,
                },
            ],
            worst_case_disposition: NeurosemanticRemediationImpactDisposition::Inconclusive,
        };
        assert!(artifact.validate().is_err());

        let missing_definition = NeurosemanticRemediationMetricDefinition {
            schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
            metric_ref: "metric-utility".into(),
            kind: NeurosemanticRemediationMeasurementKind::UtilityImpact,
            estimand_ref: "utility-on-retain-set".into(),
            scope_ref: "retain-set-v1".into(),
            unit_ref: "proportion".into(),
            aggregation_ref: "per-item-rate".into(),
            direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
        };
        let mut valid = artifact.clone();
        valid.metric_definitions.extend([
            missing_definition,
            NeurosemanticRemediationMetricDefinition {
                schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
                metric_ref: "metric-recovery".into(),
                kind: NeurosemanticRemediationMeasurementKind::RecoveryRisk,
                estimand_ref: "recovery-risk-on-forget-set".into(),
                scope_ref: "forget-set-v1".into(),
                unit_ref: "proportion".into(),
                aggregation_ref: "attack-success-rate".into(),
                direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
            },
            NeurosemanticRemediationMetricDefinition {
                schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
                metric_ref: "metric-representation".into(),
                kind: NeurosemanticRemediationMeasurementKind::RepresentationResidual,
                estimand_ref: "representation-residual-on-forget-set".into(),
                scope_ref: "forget-set-v1".into(),
                unit_ref: "proportion".into(),
                aggregation_ref: "probe-detection-rate".into(),
                direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
            },
        ]);
        assert!(valid.validate().is_ok());
        let mut legacy = valid.clone();
        legacy.schema_version = 1;
        assert!(
            NeurosemanticRemediationMeasurementArtifact::from_json_bytes(
                &serde_json::to_vec(&legacy).unwrap()
            )
            .is_err()
        );
        let mut legacy_v2 = valid.clone();
        legacy_v2.schema_version = 2;
        assert!(
            NeurosemanticRemediationMeasurementArtifact::from_json_bytes(
                &serde_json::to_vec(&legacy_v2).unwrap()
            )
            .is_err()
        );
        let mut bad_uncertainty = valid.clone();
        if let NeurosemanticRemediationUncertainty::Interval { upper_numerator, .. } =
            &mut bad_uncertainty.measurements[0].uncertainty
        {
            *upper_numerator = -2;
        }
        assert!(bad_uncertainty.validate().is_err());

        let mut missing_observation = valid.clone();
        missing_observation.measurements[0].observed_sample_count = 9;
        assert!(missing_observation.validate().is_ok());
        missing_observation.measurements[0].status =
            NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds;
        assert!(missing_observation.validate().is_err());

        let mut impossible_failure_count = valid.clone();
        impossible_failure_count.measurements[0].failure_count = 11;
        assert!(impossible_failure_count.validate().is_err());
    }

    #[test]
    fn remediation_metric_computation_reproduces_exact_observations() {
        let definition = NeurosemanticRemediationMetricDefinition {
            schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
            metric_ref: "metric-computation".into(),
            kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
            estimand_ref: "forgetfulness-on-forget-set".into(),
            scope_ref: "forget-set-v1".into(),
            unit_ref: "proportion".into(),
            aggregation_ref: "per-item-rate".into(),
            direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
        };
        let observation_set = NeurosemanticRemediationObservationSetArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_OBSERVATION_SET_SCHEMA_VERSION,
            observation_set_ref: "observation-set-computation".into(),
            metric_ref: definition.metric_ref.clone(),
            kind: definition.kind,
            scope_ref: definition.scope_ref.clone(),
            population_manifest_hash: content_hash(b"population-for-computation-test"),
            eligible_subject_artifact_hashes: vec![
                content_hash(b"item-1"),
                content_hash(b"item-2"),
                content_hash(b"item-3"),
            ],
            observations: vec![
                NeurosemanticRemediationObservationRecord {
                    observation_ref: "observation-1".into(),
                    subject_artifact_hash: content_hash(b"item-1"),
                    failure_observed: false,
                    group_ref: None,
                },
                NeurosemanticRemediationObservationRecord {
                    observation_ref: "observation-2".into(),
                    subject_artifact_hash: content_hash(b"item-2"),
                    failure_observed: true,
                    group_ref: None,
                },
            ],
        };
        let observation_bytes = serde_json::to_vec(&observation_set).unwrap();
        let observation_roundtrip =
            NeurosemanticRemediationObservationSetArtifact::from_json_bytes(
                &observation_bytes,
                &definition.aggregation_ref,
            )
            .unwrap();
        let (eligible, observed, failures, ratio_numerator, ratio_denominator) =
            recompute_metric_ratio(&observation_roundtrip, &definition.aggregation_ref).unwrap();
        assert_eq!((eligible, observed, failures), (3, 2, 1));
        assert!(fixed_point_equals_ratio(5, 1, ratio_numerator, ratio_denominator));

        let computation = NeurosemanticRemediationMetricComputationArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_COMPUTATION_SCHEMA_VERSION,
            computation_ref: "computation-exact-observations".into(),
            metric_ref: definition.metric_ref.clone(),
            kind: definition.kind,
            metric_definition_hash: definition.fingerprint().unwrap(),
            observation_set_hash: observation_set
                .fingerprint(&definition.aggregation_ref)
                .unwrap(),
            aggregation_ref: definition.aggregation_ref.clone(),
            execution_revision: "7".repeat(40),
            estimate_numerator: 5,
            estimate_scale: 1,
            eligible_sample_count: 3,
            observed_sample_count: 2,
            failure_count: 1,
        };
        assert!(computation.validate().is_ok());
        assert!(fixed_point_equals_ratio(
            computation.estimate_numerator,
            computation.estimate_scale,
            ratio_numerator,
            ratio_denominator,
        ));

        let mut forged = computation.clone();
        forged.estimate_numerator = 6;
        assert!(!fixed_point_equals_ratio(
            forged.estimate_numerator,
            forged.estimate_scale,
            ratio_numerator,
            ratio_denominator,
        ));

        let mut duplicate_subject = observation_set.clone();
        duplicate_subject.observations[1].subject_artifact_hash =
            duplicate_subject.observations[0].subject_artifact_hash.clone();
        assert!(duplicate_subject
            .validate(&definition.aggregation_ref)
            .is_err());

        let mut out_of_population = observation_set.clone();
        out_of_population.observations[1].subject_artifact_hash = content_hash(b"item-4");
        assert!(out_of_population
            .validate(&definition.aggregation_ref)
            .is_err());
    }

    #[test]
    fn remediation_alternate_population_identity_is_not_membership_equivalent() {
        let canonical = NeurosemanticRemediationEvaluationSetManifest {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_SET_SCHEMA_VERSION,
            set_ref: "forget-canonical".into(),
            set_kind: NeurosemanticRemediationEvaluationSetKind::Forget,
            source_dataset_manifest_hash: content_hash(b"dataset"),
            member_artifact_hashes: vec![content_hash(b"a"), content_hash(b"b")],
        };
        let mut alternate = canonical.clone();
        alternate.set_ref = "forget-alternate".into();
        assert_eq!(
            sorted_hashes(&canonical.member_artifact_hashes),
            sorted_hashes(&alternate.member_artifact_hashes)
        );
        assert_ne!(canonical.fingerprint().unwrap(), alternate.fingerprint().unwrap());
    }

    #[test]
    fn remediation_evaluation_split_manifest_is_typed_and_canonical() {
        let split = NeurosemanticRemediationEvaluationSplitManifest {
            schema_version: NEUROSEMANTIC_REMEDIATION_EVALUATION_SPLIT_SCHEMA_VERSION,
            split_ref: "split-typed".into(),
            source_dataset_manifest_hash: content_hash(b"dataset"),
            members: vec![
                NeurosemanticRemediationEvaluationSplitMember {
                    subject_artifact_hash: content_hash(b"subject-a"),
                    group_ref: "group-a".into(),
                },
                NeurosemanticRemediationEvaluationSplitMember {
                    subject_artifact_hash: content_hash(b"subject-b"),
                    group_ref: "group-b".into(),
                },
            ],
        };
        let bytes = serde_json::to_vec(&split).unwrap();
        assert_eq!(
            NeurosemanticRemediationEvaluationSplitManifest::from_json_bytes(&bytes).unwrap(),
            split
        );
        let mut reordered = split.clone();
        reordered.members.reverse();
        assert_eq!(split.fingerprint().unwrap(), reordered.fingerprint().unwrap());

        let mut duplicate = split.clone();
        duplicate.members[1].subject_artifact_hash = duplicate.members[0].subject_artifact_hash.clone();
        assert!(duplicate.validate().is_err());

        let mut bad_schema = split.clone();
        bad_schema.schema_version = 0;
        assert!(
            NeurosemanticRemediationEvaluationSplitManifest::from_json_bytes(
                &serde_json::to_vec(&bad_schema).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn remediation_observation_set_subgroup_gap_requires_explicit_groups() {
        let set = NeurosemanticRemediationObservationSetArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_OBSERVATION_SET_SCHEMA_VERSION,
            observation_set_ref: "subgroup-gap-observations".into(),
            metric_ref: "metric-fairness-gap".into(),
            kind: NeurosemanticRemediationMeasurementKind::FairnessImpact,
            scope_ref: "fairness-split-v1".into(),
            population_manifest_hash: content_hash(b"fairness-population-placeholder"),
            eligible_subject_artifact_hashes: vec![
                content_hash(b"fair-1"),
                content_hash(b"fair-2"),
            ],
            observations: vec![
                NeurosemanticRemediationObservationRecord {
                    observation_ref: "fair-observation-1".into(),
                    subject_artifact_hash: content_hash(b"fair-1"),
                    failure_observed: false,
                    group_ref: None,
                },
                NeurosemanticRemediationObservationRecord {
                    observation_ref: "fair-observation-2".into(),
                    subject_artifact_hash: content_hash(b"fair-2"),
                    failure_observed: false,
                    group_ref: None,
                },
            ],
        };
        assert!(set.validate("worst-subgroup-gap").is_err());
    }

    #[test]
    fn remediation_statistical_design_artifact_is_bounded_and_canonical() {
        let design = NeurosemanticRemediationStatisticalDesignArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_STATISTICAL_DESIGN_SCHEMA_VERSION,
            design_ref: "design-statistical-1".into(),
            metric_ref: "metric-statistical-1".into(),
            metric_definition_hash: content_hash(b"metric-definition"),
            observation_set_hash: content_hash(b"observation-set"),
            inference_scope: NeurosemanticRemediationUncertaintyInferenceScope::Superpopulation,
            sampling_design: NeurosemanticRemediationStatisticalSamplingDesign::ProbabilitySample,
            dependence_model:
                NeurosemanticRemediationStatisticalDependenceModel::IndependentObservationUnits,
            analysis_unit_ref: "subject-artifact".into(),
            outcome_model_ref: "binary-failure-indicator-v1".into(),
            study_protocol_hash: content_hash(b"protocol"),
            assumptions_hash: content_hash(b"assumptions"),
            execution_revision: "a".repeat(40),
        };
        let bytes = serde_json::to_vec(&design).unwrap();
        assert_eq!(
            NeurosemanticRemediationStatisticalDesignArtifact::from_json_bytes(&bytes).unwrap(),
            design
        );
        assert_eq!(design.fingerprint().unwrap(), content_hash(&bytes));

        let mut legacy = design.clone();
        legacy.schema_version = 0;
        assert!(
            NeurosemanticRemediationStatisticalDesignArtifact::from_json_bytes(
                &serde_json::to_vec(&legacy).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn remediation_uncertainty_computation_is_typed_and_bound() {
        let metric_definition = NeurosemanticRemediationMetricDefinition {
            schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
            metric_ref: "metric-uncertainty".into(),
            kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
            estimand_ref: "forgetfulness-on-forget-set".into(),
            scope_ref: "forget-set-v1".into(),
            unit_ref: "proportion".into(),
            aggregation_ref: "per-item-rate".into(),
            direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
        };
        let observation_set_hash = content_hash(b"observation-set-uncertainty");
        let assumptions_hash = content_hash(b"synthetic-structural-assumptions");
        let artifact = NeurosemanticRemediationUncertaintyComputationArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_UNCERTAINTY_COMPUTATION_SCHEMA_VERSION,
            uncertainty_ref: "uncertainty-typed".into(),
            metric_ref: metric_definition.metric_ref.clone(),
            metric_definition_hash: metric_definition.fingerprint().unwrap(),
            observation_set_hash,
            point_estimate_numerator: 0,
            point_estimate_scale: 4,
            lower_numerator: 0,
            upper_numerator: 1,
            scale: 4,
            confidence_level_bps: 9_500,
            inference_scope: NeurosemanticRemediationUncertaintyInferenceScope::Superpopulation,
            statistical_design_hash: content_hash(b"synthetic-statistical-design"),
            method_ref: "synthetic-structural-interval-v1".into(),
            assumptions_hash,
            assumptions_ref: "synthetic-structural-assumptions-v1".into(),
            execution_revision: "8".repeat(40),
        };
        assert!(artifact.validate().is_ok());
        let bytes = serde_json::to_vec(&artifact).unwrap();
        assert_eq!(
            NeurosemanticRemediationUncertaintyComputationArtifact::from_json_bytes(&bytes).unwrap(),
            artifact
        );

        let mut legacy = artifact.clone();
        legacy.schema_version = 2;
        assert!(
            NeurosemanticRemediationUncertaintyComputationArtifact::from_json_bytes(
                &serde_json::to_vec(&legacy).unwrap()
            )
            .is_err()
        );

        let mut assumptions_invalid = artifact.clone();
        assumptions_invalid.assumptions_hash.clear();
        assert!(assumptions_invalid.validate().is_err());

        let mut out_of_domain = artifact.clone();
        out_of_domain.lower_numerator = -1;
        assert!(out_of_domain.validate().is_err());
        out_of_domain.lower_numerator = 0;
        out_of_domain.upper_numerator = 10_001;
        assert!(out_of_domain.validate().is_err());

        let _ = metric_definition;
    }

    #[test]
    fn remediation_failure_rate_direction_cannot_claim_higher_is_better() {
        let mut definition = NeurosemanticRemediationMetricDefinition {
            schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
            metric_ref: "metric-direction".into(),
            kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
            estimand_ref: "forgetfulness-failure-rate".into(),
            scope_ref: "forget-set-v1".into(),
            unit_ref: "proportion".into(),
            aggregation_ref: "per-item-rate".into(),
            direction: NeurosemanticRemediationMetricDirection::HigherIsBetter,
        };
        assert!(definition.validate().is_err());
        definition.direction = NeurosemanticRemediationMetricDirection::LowerIsBetter;
        assert!(definition.validate().is_ok());
    }

    #[test]
    fn remediation_wilson_method_cannot_target_incompatible_designs() {
        let definition = NeurosemanticRemediationMetricDefinition {
            schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
            metric_ref: "metric-direction".into(),
            kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
            estimand_ref: "forgetfulness-failure-rate".into(),
            scope_ref: "forget-set-v1".into(),
            unit_ref: "proportion".into(),
            aggregation_ref: "per-item-rate".into(),
            direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
        };
        let mut design = NeurosemanticRemediationStatisticalDesignArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_STATISTICAL_DESIGN_SCHEMA_VERSION,
            design_ref: "design-wilson".into(),
            metric_ref: definition.metric_ref.clone(),
            metric_definition_hash: definition.fingerprint().unwrap(),
            observation_set_hash: content_hash(b"observations"),
            inference_scope: NeurosemanticRemediationUncertaintyInferenceScope::Superpopulation,
            sampling_design: NeurosemanticRemediationStatisticalSamplingDesign::ProbabilitySample,
            dependence_model:
                NeurosemanticRemediationStatisticalDependenceModel::IndependentObservationUnits,
            analysis_unit_ref: "subject-artifact".into(),
            outcome_model_ref: "binary-failure-indicator-v1".into(),
            study_protocol_hash: content_hash(b"protocol"),
            assumptions_hash: content_hash(b"assumptions"),
            execution_revision: "8".repeat(40),
        };
        assert!(validate_uncertainty_method_application(
            NEUROSEMANTIC_REMEDIATION_WILSON_95_METHOD_REF,
            &definition,
            &design,
        )
        .is_ok());

        design.dependence_model =
            NeurosemanticRemediationStatisticalDependenceModel::Clustered;
        assert!(validate_uncertainty_method_application(
            NEUROSEMANTIC_REMEDIATION_WILSON_95_METHOD_REF,
            &definition,
            &design,
        )
        .is_err());

        design.dependence_model =
            NeurosemanticRemediationStatisticalDependenceModel::IndependentObservationUnits;
        design.sampling_design =
            NeurosemanticRemediationStatisticalSamplingDesign::NonProbabilitySample;
        assert!(validate_uncertainty_method_application(
            NEUROSEMANTIC_REMEDIATION_WILSON_95_METHOD_REF,
            &definition,
            &design,
        )
        .is_err());

        design.sampling_design =
            NeurosemanticRemediationStatisticalSamplingDesign::ProbabilitySample;
        design.outcome_model_ref = "continuous-outcome-v1".into();
        assert!(validate_uncertainty_method_application(
            NEUROSEMANTIC_REMEDIATION_WILSON_95_METHOD_REF,
            &definition,
            &design,
        )
        .is_err());
    }

    #[test]
    fn remediation_wilson_method_requires_superpopulation_scope() {
        let definition = NeurosemanticRemediationMetricDefinition {
            schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
            metric_ref: "metric-scope".into(),
            kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
            estimand_ref: "forgetfulness-failure-rate".into(),
            scope_ref: "forget-set-v1".into(),
            unit_ref: "proportion".into(),
            aggregation_ref: "per-item-rate".into(),
            direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
        };
        let mut design = NeurosemanticRemediationStatisticalDesignArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_STATISTICAL_DESIGN_SCHEMA_VERSION,
            design_ref: "design-scope".into(),
            metric_ref: definition.metric_ref.clone(),
            metric_definition_hash: definition.fingerprint().unwrap(),
            observation_set_hash: content_hash(b"observations"),
            inference_scope: NeurosemanticRemediationUncertaintyInferenceScope::FixedEvaluationPopulation,
            sampling_design: NeurosemanticRemediationStatisticalSamplingDesign::ProbabilitySample,
            dependence_model:
                NeurosemanticRemediationStatisticalDependenceModel::IndependentObservationUnits,
            analysis_unit_ref: "subject-artifact".into(),
            outcome_model_ref: "binary-failure-indicator-v1".into(),
            study_protocol_hash: content_hash(b"protocol"),
            assumptions_hash: content_hash(b"assumptions"),
            execution_revision: "9".repeat(40),
        };
        assert!(validate_uncertainty_method_application(
            NEUROSEMANTIC_REMEDIATION_WILSON_95_METHOD_REF,
            &definition,
            &design,
        )
        .is_err());

        design.inference_scope =
            NeurosemanticRemediationUncertaintyInferenceScope::Superpopulation;
        assert!(validate_uncertainty_method_application(
            NEUROSEMANTIC_REMEDIATION_WILSON_95_METHOD_REF,
            &definition,
            &design,
        )
        .is_ok());
    }

    #[test]
    fn remediation_wilson_score_interval_is_numerically_reproducible() {
        assert_eq!(
            recompute_wilson_score_95_interval(0, 2, 4).unwrap(),
            (0, 6_577, 4)
        );
        let (lower, upper, scale) =
            recompute_wilson_score_95_interval(1, 2, 4).unwrap();
        assert_eq!(scale, 4);
        assert!(lower > 0);
        assert!(upper < 10_000);
    }

    #[test]
    fn remediation_measurement_worst_case_is_recomputed_not_supplied() {
        let definitions = [
            (
                "metric-forgetfulness",
                NeurosemanticRemediationMeasurementKind::Forgetfulness,
            ),
            (
                "metric-utility",
                NeurosemanticRemediationMeasurementKind::UtilityImpact,
            ),
            (
                "metric-recovery",
                NeurosemanticRemediationMeasurementKind::RecoveryRisk,
            ),
            (
                "metric-representation",
                NeurosemanticRemediationMeasurementKind::RepresentationResidual,
            ),
        ];
        let metric_definitions = definitions
            .iter()
            .map(|(metric_ref, kind)| NeurosemanticRemediationMetricDefinition {
                schema_version: NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
                metric_ref: (*metric_ref).into(),
                kind: *kind,
                estimand_ref: format!("{metric_ref}-estimand"),
                scope_ref: "evaluation-set-v1".into(),
                unit_ref: "proportion".into(),
                aggregation_ref: "per-item-rate".into(),
                direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
            })
            .collect();
        let measured = NeurosemanticRemediationMeasurementArtifact {
            schema_version: NEUROSEMANTIC_REMEDIATION_MEASUREMENT_SCHEMA_VERSION,
            measurement_ref: "measurement-worst-case".into(),
            metric_definitions,
            metric_computation_artifact_hashes: vec![
                content_hash(b"worst-case-computation-1"),
                content_hash(b"worst-case-computation-2"),
                content_hash(b"worst-case-computation-3"),
                content_hash(b"worst-case-computation-4"),
            ],
            measurements: vec![
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-forgetfulness".into(),
                    kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
                    status: NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 10,
                    observed_sample_count: 10,
                    failure_count: 0,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-utility".into(),
                    kind: NeurosemanticRemediationMeasurementKind::UtilityImpact,
                    status: NeurosemanticRemediationImpactDisposition::OutsideDeclaredBounds,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 10,
                    observed_sample_count: 10,
                    failure_count: 1,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-recovery".into(),
                    kind: NeurosemanticRemediationMeasurementKind::RecoveryRisk,
                    status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 10,
                    observed_sample_count: 10,
                    failure_count: 0,
                },
                NeurosemanticRemediationMeasurement {
                    metric_ref: "metric-representation".into(),
                    kind: NeurosemanticRemediationMeasurementKind::RepresentationResidual,
                    status: NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds,
                    estimate_numerator: 0,
                    estimate_scale: 4,
                    uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                    eligible_sample_count: 10,
                    observed_sample_count: 10,
                    failure_count: 0,
                },
            ],
            worst_case_disposition: NeurosemanticRemediationImpactDisposition::Inconclusive,
        };
        assert!(measured.validate().is_ok());
        let mut forged = measured.clone();
        forged.worst_case_disposition = NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds;
        assert!(forged.validate().is_err());
        assert_eq!(
            measured.recomputed_worst_case_disposition().unwrap(),
            NeurosemanticRemediationImpactDisposition::Inconclusive
        );
        let mut outside_only = measured.clone();
        for item in &mut outside_only.measurements {
            if item.status == NeurosemanticRemediationImpactDisposition::Inconclusive {
                item.status = NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds;
            }
        }
        outside_only.worst_case_disposition = NeurosemanticRemediationImpactDisposition::OutsideDeclaredBounds;
        assert!(outside_only.validate().is_ok());
        let mut invalid_counts = measured.clone();
        invalid_counts.measurements[0].failure_count = 11;
        assert!(invalid_counts.validate().is_err());
        let mut missing_observation = measured.clone();
        missing_observation.measurements[0].observed_sample_count = 9;
        assert!(missing_observation.validate().is_ok());
        missing_observation.measurements[0].status =
            NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds;
        assert!(missing_observation.validate().is_err());
    }

    #[test]
    fn lifecycle_receipt_action_schema_is_stable() {
        for (action, expected) in [
            (NeurosemanticArtifactLifecycleAction::AccessRevocation, "AccessRevocation"),
            (NeurosemanticArtifactLifecycleAction::Retention, "Retention"),
            (NeurosemanticArtifactLifecycleAction::Erasure, "Erasure"),
            (NeurosemanticArtifactLifecycleAction::Rectification, "Rectification"),
            (NeurosemanticArtifactLifecycleAction::Supersession, "Supersession"),
        ] {
            assert_eq!(serde_json::to_string(&action).unwrap(), format!("\"{expected}\""));
        }
    }

    #[test]
    fn lifecycle_receipt_state_schema_is_stable() {
        for (state, expected) in [
            (NeurosemanticArtifactLifecycleState::Requested, "Requested"),
            (NeurosemanticArtifactLifecycleState::Accepted, "Accepted"),
            (NeurosemanticArtifactLifecycleState::Processing, "Processing"),
            (NeurosemanticArtifactLifecycleState::Applied, "Applied"),
            (NeurosemanticArtifactLifecycleState::IndependentlyVerified, "IndependentlyVerified"),
            (NeurosemanticArtifactLifecycleState::Rejected, "Rejected"),
        ] {
            assert_eq!(serde_json::to_string(&state).unwrap(), format!("\"{expected}\""));
        }
    }

    #[test]
    fn lifecycle_receipt_verification_scope_schema_is_stable() {
        for (scope, expected) in [
            (NeurosemanticArtifactLifecycleVerificationScope::ArtifactOnly, "ArtifactOnly"),
            (NeurosemanticArtifactLifecycleVerificationScope::EnumeratedTargetSet, "EnumeratedTargetSet"),
        ] {
            assert_eq!(serde_json::to_string(&scope).unwrap(), format!("\"{expected}\""));
        }
    }

    fn authority_attestation(
        policy: &NeurosemanticDataPolicy,
    ) -> (NeurosemanticPolicyAuthorityAttestation, VerifyingKey) {
        use ed25519_dalek::{Signer, SigningKey};
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let fingerprint = policy.handling.fingerprint_for_attestation().unwrap();
        let hash = policy.handling.policy_provenance_hash.clone();
        let message = NeurosemanticPolicyAuthorityAttestation::message_bytes(
            "mycelix-policy-authority",
            "test-key-1",
            &fingerprint,
            &hash,
            100,
            2_000,
        )
        .unwrap();
        let signature = signing_key.sign(&message).to_bytes().to_vec();
        (
            NeurosemanticPolicyAuthorityAttestation {
                schema_version: NEUROSEMANTIC_POLICY_ATTESTATION_SCHEMA_VERSION,
                authority_ref: "mycelix-policy-authority".into(),
                key_ref: "test-key-1".into(),
                handling_policy_fingerprint: fingerprint,
                policy_provenance_hash: hash,
                issued_at_unix_s: 100,
                expires_at_unix_s: 2_000,
                signature,
            },
            signing_key.verifying_key(),
        )
    }

    fn lease() -> CognitiveConsentLease {
        CognitiveConsentLease {
            lease_id: "lease-1".into(),
            subject_id: "subject".into(),
            peer_id: "peer".into(),
            purpose: CommunicationPurpose::HumanCollaboration,
            read_scopes: BTreeSet::from([CognitiveChannel::Semantic]),
            write_scopes: BTreeSet::from([CognitiveChannel::Semantic]),
            max_read_sensitivity: CognitiveSensitivity::Private,
            max_write_sensitivity: CognitiveSensitivity::Private,
            read_data_classes: BTreeSet::from([NeurosemanticDataClass::SemanticRepresentation]),
            write_data_classes: BTreeSet::from([NeurosemanticDataClass::SemanticRepresentation]),
            read_inference_classes: BTreeSet::from([NeurosemanticInferenceClass::SemanticContent]),
            write_inference_classes: BTreeSet::from([NeurosemanticInferenceClass::SemanticContent]),
            issued_at_unix_s: 100,
            expires_at_unix_s: 200,
            consent_epoch: 7,
            revoked: false,
            revoked_at_unix_s: None,
        }
    }

    #[test]
    fn v2_transport_rejects_raw_and_personalized_data_classes() {
        let mut policy = semantic_policy();
        policy.data_class = NeurosemanticDataClass::RawNeuralRecording;
        assert!(!policy.transportable());
        let mut policy = semantic_policy();
        policy.data_class = NeurosemanticDataClass::PersonalizedDecoderModel;
        assert!(!policy.transportable());
    }

    #[test]
    fn data_policy_is_deny_by_default() {
        let policy = NeurosemanticDataPolicy::default();
        assert!(!policy.validates());
        assert!(!policy.allows_purpose(CommunicationPurpose::HumanCollaboration));
    }

    #[test]
    fn data_policy_requires_explicit_inference_and_purpose() {
        let mut policy = semantic_policy();
        assert!(policy.validates());
        policy.inference_classes.clear();
        assert!(!policy.validates());
        let mut policy = semantic_policy();
        policy.permitted_purposes.clear();
        assert!(!policy.validates());
    }

    #[test]
    fn legacy_lease_data_permissions_default_to_empty_and_deny() {
        let lease = lease();
        let mut value = serde_json::to_value(&lease).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("read_data_classes");
        object.remove("write_data_classes");
        object.remove("read_inference_classes");
        object.remove("write_inference_classes");
        let restored: CognitiveConsentLease = serde_json::from_value(value).unwrap();
        assert!(restored.read_data_classes.is_empty());
        assert!(restored.write_data_classes.is_empty());
        assert!(!restored.authorizes_data_policy(ChannelDirection::Read, &semantic_policy()));
    }

    #[test]
    fn data_class_and_inference_authorization_are_independent() {
        let lease = lease();
        let policy = semantic_policy();
        assert!(lease.authorizes_data_policy(ChannelDirection::Read, &policy));
        let mut identity = policy.clone();
        identity.inference_classes = BTreeSet::from([NeurosemanticInferenceClass::Identity]);
        assert!(!lease.authorizes_data_policy(ChannelDirection::Read, &identity));
        let mut decoded_claim = policy;
        decoded_claim.data_class = NeurosemanticDataClass::DecodedClaim;
        assert!(!lease.authorizes_data_policy(ChannelDirection::Read, &decoded_claim));
    }

    #[test]
    fn legacy_data_policy_schema_is_rejected_by_v8_validator() {
        let policy = semantic_policy();
        let mut value = serde_json::to_value(&policy).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert(
            "schema_version".into(),
            serde_json::json!(NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION - 1),
        );
        let restored: NeurosemanticDataPolicy = serde_json::from_value(value).unwrap();
        assert!(!restored.validates());
        assert!(!restored.handling.validates());
    }

    #[test]
    fn data_class_schema_is_stable() {
        for (class, expected) in [
            (NeurosemanticDataClass::Unknown, "Unknown"),
            (NeurosemanticDataClass::RawNeuralRecording, "RawNeuralRecording"),
            (NeurosemanticDataClass::DerivedNeuralFeature, "DerivedNeuralFeature"),
            (NeurosemanticDataClass::SemanticRepresentation, "SemanticRepresentation"),
            (NeurosemanticDataClass::DecodedClaim, "DecodedClaim"),
            (NeurosemanticDataClass::PersonalizedDecoderModel, "PersonalizedDecoderModel"),
        ] {
            assert_eq!(serde_json::to_string(&class).unwrap(), format!("\"{expected}\""));
        }
    }

    #[test]
    fn inference_class_schema_is_stable() {
        for (class, expected) in [
            (NeurosemanticInferenceClass::Unknown, "Unknown"),
            (NeurosemanticInferenceClass::SignalPattern, "SignalPattern"),
            (NeurosemanticInferenceClass::UnitPattern, "UnitPattern"),
            (NeurosemanticInferenceClass::LinguisticContent, "LinguisticContent"),
            (NeurosemanticInferenceClass::SemanticContent, "SemanticContent"),
            (NeurosemanticInferenceClass::AffectiveState, "AffectiveState"),
            (NeurosemanticInferenceClass::Intent, "Intent"),
            (NeurosemanticInferenceClass::Identity, "Identity"),
        ] {
            assert_eq!(serde_json::to_string(&class).unwrap(), format!("\"{expected}\""));
        }
    }

    #[test]
    fn authority_attestation_rejects_tampering_wrong_key_and_policy() {
        let policy = semantic_policy();
        let (mut attestation, verifying_key) = authority_attestation(&policy);
        attestation.signature[0] ^= 0x01;
        assert!(attestation
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_err());

        let (good_attestation, _) = authority_attestation(&policy);
        let wrong_key = ed25519_dalek::SigningKey::from_bytes(&[8u8; 32]);
        assert!(good_attestation
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &wrong_key.verifying_key(),
                150
            )
            .is_err());

        let mut changed_lineage_policy = policy.clone();
        changed_lineage_policy.handling.derivation_provenance_ref =
            "synthetic-derivation-record-2".into();
        assert!(good_attestation
            .verify(
                &changed_lineage_policy.handling.fingerprint_for_attestation().unwrap(),
                &changed_lineage_policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_err());

        let mut changed_policy = policy.clone();
        changed_policy.handling.retention = NeurosemanticRetentionPolicy::Ephemeral;
        assert!(good_attestation
            .verify(
                &changed_policy.handling.fingerprint_for_attestation().unwrap(),
                &changed_policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_err());
    }

    #[test]
    fn authority_attestation_bounded_json_parser_fails_closed() {
        let policy = semantic_policy();
        let (attestation, _) = authority_attestation(&policy);
        let encoded = serde_json::to_vec(&attestation).unwrap();
        assert_eq!(
            NeurosemanticPolicyAuthorityAttestation::from_json_bytes(&encoded).unwrap(),
            attestation
        );

        let mut malformed = attestation.clone();
        malformed.signature = vec![0u8; MAX_NEUROSEMANTIC_AUTHORITY_SIGNATURE_BYTES - 1];
        let malformed_bytes = serde_json::to_vec(&malformed).unwrap();
        assert!(NeurosemanticPolicyAuthorityAttestation::from_json_bytes(&malformed_bytes).is_err());

        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticPolicyAuthorityAttestation::from_json_bytes(&oversized).is_err());
    }

    #[test]
    fn authority_attestation_fingerprint_binds_exact_signature() {
        let policy = semantic_policy();
        let (attestation, verifying_key) = authority_attestation(&policy);
        let fingerprint = attestation.fingerprint_for_attestation().unwrap();
        let mut tampered = attestation.clone();
        tampered.signature[0] ^= 0x01;
        assert_ne!(
            tampered.fingerprint_for_attestation().unwrap(),
            fingerprint
        );
        assert!(tampered
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_err());
    }

    #[test]
    fn authority_attestation_verifies_exact_policy_and_provenance() {
        let policy = semantic_policy();
        let (attestation, verifying_key) = authority_attestation(&policy);
        assert!(attestation
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_ok());

        let mut changed = attestation.clone();
        changed.policy_provenance_hash = compute_policy_provenance_hash(
            "synthetic-policy-record-1",
            b"synthetic-policy-record-2",
        );
        assert!(changed
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_err());
    }

    #[test]
    fn authority_attestation_expires_and_invalidates_binding() {
        let policy = semantic_policy();
        let (attestation, verifying_key) = authority_attestation(&policy);
        let binding = policy_provenance_binding();
        assert_eq!(binding.authority_ref(), "mycelix-policy-authority");
        let context = binding_context();
        let (resolution, resolution_key) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);
        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &attestation,
                &verifying_key,
                &resolution,
                &resolution_key,
                &context,
                2_000,
            )
            .is_err());
    }

    #[test]
    fn handling_rejects_stale_or_unattested_provenance_capability() {
        let policy = semantic_policy();
        let original_binding = policy_provenance_binding();

        let packet = NeurosemanticPacket::new_with_policy(
            19,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease().consent_epoch,
            lease_id: lease().lease_id,
        };
        assert!(message
            .validate_for_handling(
                &lease(),
                &original_binding,
                "ZA",
                NeurosemanticHandlingAction::Transmit,
                150,
            )
            .is_ok());

        let mut expired = original_binding.clone();
        expired.attestation_expires_at_unix_s = 150;
        assert!(message
            .validate_for_handling(
                &lease(),
                &expired,
                "ZA",
                NeurosemanticHandlingAction::Transmit,
                150,
            )
            .is_err());
    }

    #[test]
    fn authority_resolution_rejects_valid_but_different_authority_proof() {
        let policy = semantic_policy();
        let (attestation, authority_key) = authority_attestation(&policy);
        let context = binding_context();
        let (resolution, resolver_key) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);

        use ed25519_dalek::Signer;
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
        let changed_message = NeurosemanticPolicyAuthorityAttestation::message_bytes(
            &attestation.authority_ref,
            &attestation.key_ref,
            &attestation.handling_policy_fingerprint,
            &attestation.policy_provenance_hash,
            110,
            1_900,
        )
        .unwrap();
        let substituted_attestation = NeurosemanticPolicyAuthorityAttestation {
            issued_at_unix_s: 110,
            expires_at_unix_s: 1_900,
            signature: signing_key.sign(&changed_message).to_bytes().to_vec(),
            ..attestation.clone()
        };

        assert!(substituted_attestation
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &authority_key,
                150
            )
            .is_ok());

        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &substituted_attestation,
                &authority_key,
                &resolution,
                &resolver_key,
                &context,
                150,
            )
            .is_err());
    }

    #[test]
    fn authority_resolution_binds_status_and_consent_context() {
        let policy = semantic_policy();
        let (attestation, authority_key) = authority_attestation(&policy);
        let context = binding_context();
        let (resolution, resolver_key) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);
        let binding = policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &attestation,
                &authority_key,
                &resolution,
                &resolver_key,
                &context,
                150,
            )
            .unwrap();
        assert_eq!(binding.authority_resolution_status, NeurosemanticAuthorityStatus::Active);
        assert_eq!(binding.authority_resolution_consent_epoch, 7);
        assert_eq!(binding.authority_resolution_subject_ref, "subject");
    }

    #[test]
    fn authority_resolution_status_schema_is_stable() {
        for (status, expected) in [
            (NeurosemanticAuthorityStatus::Active, "Active"),
            (NeurosemanticAuthorityStatus::Suspended, "Suspended"),
            (NeurosemanticAuthorityStatus::Revoked, "Revoked"),
            (NeurosemanticAuthorityStatus::Unknown, "Unknown"),
            (NeurosemanticAuthorityStatus::Unavailable, "Unavailable"),
        ] {
            assert_eq!(
                serde_json::to_string(&status).unwrap(),
                format!("\"{expected}\"")
            );
        }
    }

    #[test]
    fn authority_resolution_schema_version_fails_closed() {
        let policy = semantic_policy();
        let (attestation, _) = authority_attestation(&policy);
        let context = binding_context();
        let (resolution, _) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);
        let mut value = serde_json::to_value(&resolution).unwrap();
        value.as_object_mut()
            .unwrap()
            .insert("schema_version".into(), serde_json::json!(999));
        let encoded = serde_json::to_vec(&value).unwrap();
        assert!(NeurosemanticAuthorityResolutionAttestation::from_json_bytes(&encoded).is_err());

        let mut legacy = serde_json::to_value(&resolution).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .insert("schema_version".into(), serde_json::json!(1));
        let legacy_encoded = serde_json::to_vec(&legacy).unwrap();
        assert!(NeurosemanticAuthorityResolutionAttestation::from_json_bytes(&legacy_encoded).is_err());
    }

    #[test]
    fn authority_resolution_message_requires_valid_unsigned_fields() {
        let policy = semantic_policy();
        let (attestation, _) = authority_attestation(&policy);
        let context = binding_context();
        let (mut resolution, _) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);

        resolution.expires_at_unix_s = resolution.checked_at_unix_s;
        assert!(resolution.message_bytes().is_err());

        let (mut resolution, _) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);
        resolution.resolver_ref.clear();
        assert!(resolution.message_bytes().is_err());
    }

    #[test]
    fn authority_resolution_cannot_predate_authority_attestation() {
        let policy = semantic_policy();
        let (attestation, authority_key) = authority_attestation(&policy);
        let context = binding_context();
        let (mut resolution, resolver_key) =
            authority_resolution(&policy, &attestation, &context, 50, 1_900);
        let resolver_signing_key = ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]);
        resolution.signature = ed25519_dalek::Signer::sign(
            &resolver_signing_key,
            &resolution.message_bytes().unwrap(),
        )
        .to_bytes()
        .to_vec();

        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &attestation,
                &authority_key,
                &resolution,
                &resolver_key,
                &context,
                150,
            )
            .is_err());
    }

    #[test]
    fn authority_resolution_cannot_outlive_authority_attestation() {
        let policy = semantic_policy();
        let (attestation, authority_key) = authority_attestation(&policy);
        let context = binding_context();
        let mut resolution = authority_resolution(&policy, &attestation, &context, 100, 2_100).0;
        let resolver_key = ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]);
        resolution.signature =
            ed25519_dalek::Signer::sign(&resolver_key, &resolution.message_bytes().unwrap())
                .to_bytes()
                .to_vec();
        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &attestation,
                &authority_key,
                &resolution,
                &resolver_key.verifying_key(),
                &context,
                1_500,
            )
            .is_err());
    }

    #[test]
    fn authority_resolution_rejects_mutated_lease_without_epoch_change() {
        let policy = semantic_policy();
        let binding = policy_provenance_binding();

        let mut mutated_lease = lease();
        mutated_lease.max_write_sensitivity = CognitiveSensitivity::HighlyPrivate;

        let packet = NeurosemanticPacket::new_with_policy(
            20,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: mutated_lease.consent_epoch,
            lease_id: mutated_lease.lease_id.clone(),
        };
        assert!(message
            .validate_for_handling(
                &mutated_lease,
                &binding,
                "ZA",
                NeurosemanticHandlingAction::Transmit,
                150,
            )
            .is_err());
    }

    #[test]
    fn authority_resolution_rejects_wrong_context_status_and_stale_time() {
        let policy = semantic_policy();
        let (attestation, authority_key) = authority_attestation(&policy);
        let context = binding_context();
        let (resolution, resolver_key) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);

        let mut wrong_context = context.clone();
        wrong_context.subject_ref = "other-subject".into();
        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &attestation,
                &authority_key,
                &resolution,
                &resolver_key,
                &wrong_context,
                150,
            )
            .is_err());

        let mut suspended = resolution.clone();
        suspended.status = NeurosemanticAuthorityStatus::Suspended;
        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &attestation,
                &authority_key,
                &suspended,
                &resolver_key,
                &context,
                150,
            )
            .is_err());

        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &attestation,
                &authority_key,
                &resolution,
                &resolver_key,
                &context,
                2_000,
            )
            .is_err());
    }

    #[test]
    fn authority_resolution_signature_and_json_boundary_are_fail_closed() {
        let policy = semantic_policy();
        let (attestation, authority_key) = authority_attestation(&policy);
        let context = binding_context();
        let (resolution, resolver_key) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);

        let encoded = serde_json::to_vec(&resolution).unwrap();
        assert_eq!(
            NeurosemanticAuthorityResolutionAttestation::from_json_bytes(&encoded).unwrap(),
            resolution
        );

        let mut tampered = resolution.clone();
        tampered.signature[0] ^= 1;
        assert!(tampered
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_ref,
                &policy.handling.policy_provenance_hash,
                &attestation,
                &context,
                &resolver_key,
                150,
            )
            .is_err());

        let wrong_key = ed25519_dalek::SigningKey::from_bytes(&[10u8; 32]);
        assert!(resolution
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_ref,
                &policy.handling.policy_provenance_hash,
                &attestation,
                &context,
                &wrong_key.verifying_key(),
                150,
            )
            .is_err());

        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticAuthorityResolutionAttestation::from_json_bytes(&oversized).is_err());

        let _ = authority_key;
    }

    #[test]
    fn handling_policy_freshness_bound_is_machine_enforced() {
        let mut policy = semantic_policy().handling;
        assert!(policy.validates());
        policy.max_authority_resolution_age_s = 0;
        assert!(!policy.validates());
        policy.max_authority_resolution_age_s = MAX_NEUROSEMANTIC_AUTHORITY_RESOLUTION_TTL_S + 1;
        assert!(!policy.validates());
        policy.max_authority_resolution_age_s = 1;
        assert!(policy.validates());
    }

    #[test]
    fn handling_policy_requires_exact_derivation_provenance() {
        let mut policy = semantic_policy().handling;
        assert!(policy.verify_derivation_provenance_binding_bytes(
            &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap()
        ));
        assert!(!policy.verify_derivation_provenance_binding_bytes(
            b"synthetic-derivation-record-2"
        ));
        policy.derivation_provenance_ref = "synthetic-derivation-record-2".into();
        assert!(!policy.verify_derivation_provenance_binding_bytes(
            &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap()
        ));
    }

    #[test]
    fn authority_resolution_requires_exact_status_source_record() {
        let policy = semantic_policy();
        let (attestation, authority_key) = authority_attestation(&policy);
        let context = binding_context();
        let (resolution, resolver_key) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);
        assert!(resolution.verify_status_source_binding_bytes(b"synthetic-status-record-1"));
        assert!(!resolution.verify_status_source_binding_bytes(b"synthetic-status-record-2"));

        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-1",
                &attestation,
                &authority_key,
                &resolution,
                &resolver_key,
                &context,
                150,
            )
            .is_ok());
        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                b"synthetic-status-record-2",
                &attestation,
                &authority_key,
                &resolution,
                &resolver_key,
                &context,
                150,
            )
            .is_err());
    }

    #[test]
    fn derivation_lineage_record_is_bounded_and_machine_verifiable() {
        let record = synthetic_derivation_lineage_record();
        let encoded = serde_json::to_vec(&record).unwrap();
        assert_eq!(NeurosemanticDerivationLineageRecord::from_json_bytes(&encoded).unwrap(), record);

        let mut duplicate = record.clone();
        duplicate.input_artifact_refs.push("input-artifact-1".into());
        assert!(duplicate.validate().is_err());

        let mut missing_hash = record.clone();
        missing_hash.input_artifact_hashes.pop();
        assert!(missing_hash.validate().is_err());

        let mut bad_hash = record.clone();
        bad_hash.input_artifact_hashes[0] = "not-a-blake3-digest".into();
        assert!(bad_hash.validate().is_err());

        let mut bad_revision = record.clone();
        bad_revision.execution_revision = "placeholder".into();
        assert!(bad_revision.validate().is_err());

        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticDerivationLineageRecord::from_json_bytes(&oversized).is_err());
    }

    #[test]
    fn derivation_lineage_verifies_concrete_input_artifacts() {
        let record = synthetic_derivation_lineage_record();
        assert!(record.verify_input_artifact_bytes(0, b"synthetic-input-artifact-1").is_ok());
        assert!(record.verify_input_artifact_bytes(1, b"synthetic-input-artifact-2").is_ok());
        assert!(record.verify_input_artifact_bytes(0, b"synthetic-input-artifact-tampered").is_err());
        assert!(record.verify_input_artifact_bytes(2, b"synthetic-input-artifact-3").is_err());
        assert!(record
            .verify_input_artifact_bytes(
                0,
                &vec![b'x'; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1],
            )
            .is_err());
    }

    #[test]
    fn packet_rejects_derivation_lineage_output_substitution() {
        let mut policy = semantic_policy();
        policy.handling.derivation_output_artifact_hash =
            content_hash(&serde_json::to_vec(&NeurosemanticPayload::Hypervector(vec![9, 9])).unwrap());
        assert!(NeurosemanticPacket::new_with_policy(
            111,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .is_err());
    }

    #[test]
    fn derivation_lineage_future_timestamp_is_fail_closed() {
        let mut policy = semantic_policy();
        let mut record = synthetic_derivation_lineage_record();
        record.generated_at_unix_s = 151;
        let record_bytes = serde_json::to_vec(&record).unwrap();
        policy.handling.derivation_provenance_hash = compute_derivation_provenance_hash(
            &policy.handling.derivation_provenance_ref,
            &record_bytes,
        );
        let (attestation, authority_key) = authority_attestation(&policy);
        let context = binding_context();
        let (resolution, resolver_key) =
            authority_resolution(&policy, &attestation, &context, 100, 2_000);
        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &record_bytes,
                b"synthetic-status-record-1",
                &attestation,
                &authority_key,
                &resolution,
                &resolver_key,
                &context,
                150,
            )
            .is_err());
    }

    #[test]
    fn handling_policy_defaults_to_deny() {
        let policy = NeurosemanticHandlingPolicy::default();
        assert!(!policy.validates());
        assert!(!policy.allows_destination("ZA"));
        assert!(!policy.allows_action(NeurosemanticHandlingAction::Persist, 1));
        assert!(!policy.allows_action(
            NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::Research),
            1
        ));
    }

    #[test]
    fn handling_policy_provenance_reference_is_bounded() {
        let mut policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: "x".repeat(MAX_NEUROSEMANTIC_ID_BYTES + 1),
            policy_provenance_hash: compute_policy_provenance_hash(
                "synthetic-policy-record-1",
                b"synthetic-policy-record-1",
            ),
            derivation_provenance_ref: "synthetic-derivation-record-1".into(),
            derivation_provenance_hash: compute_derivation_provenance_hash(
                "synthetic-derivation-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
            ),
            derivation_output_artifact_hash: synthetic_output_artifact_hash(),
            origin_jurisdiction: "ZA".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
            max_authority_resolution_age_s: 300,
        };
        assert!(!policy.validates());
        policy.policy_provenance_ref = "synthetic-policy-record-1".into();
        assert!(policy.validates());
    }

    #[test]
    fn handling_policy_requires_machine_verifiable_provenance_hash() {
        let mut policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: "synthetic-policy-record-1".into(),
            policy_provenance_hash: String::new(),
            derivation_provenance_ref: "synthetic-derivation-record-1".into(),
            derivation_provenance_hash: compute_derivation_provenance_hash(
                "synthetic-derivation-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
            ),
            derivation_output_artifact_hash: synthetic_output_artifact_hash(),
            origin_jurisdiction: "ZA".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
            max_authority_resolution_age_s: 300,
        };
        assert!(!policy.validates());
        policy.policy_provenance_hash = compute_policy_provenance_hash(
            "synthetic-policy-record-1",
            b"synthetic-policy-record-1",
        );
        assert!(policy.validates());
        policy.policy_provenance_hash = "not-a-blake3-digest".into();
        assert!(!policy.validates());
        policy.policy_provenance_hash = "A".repeat(64);
        assert!(!policy.validates());
    }

    #[test]
    fn handling_policy_provenance_digest_verifies_exact_record_and_reference() {
        let policy = semantic_policy().handling;
        assert!(policy.verify_policy_record_binding_bytes(b"synthetic-policy-record-1"));
        assert!(!policy.verify_policy_record_binding_bytes(b"synthetic-policy-record-2"));
        assert!(!policy.verify_policy_record_binding_bytes(
            &vec![b'x'; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1]
        ));
    }

    #[test]
    fn handling_policy_provenance_reference_cannot_be_swapped_under_existing_digest() {
        let mut policy = semantic_policy().handling;
        assert!(policy.verify_policy_record_binding_bytes(b"synthetic-policy-record-1"));
        policy.policy_provenance_ref = "synthetic-policy-record-2".into();
        assert!(!policy.verify_policy_record_binding_bytes(b"synthetic-policy-record-1"));
    }

    #[test]
    fn handling_policy_provenance_binding_becomes_stale_after_policy_mutation() {
        let policy = semantic_policy();
        let original_binding = {
            let (attestation, verifying_key) = authority_attestation(&policy);
            policy_provenance_binding()
        };

        let mut mutated = policy.clone();
        mutated
            .handling
            .permitted_secondary_uses
            .insert(NeurosemanticSecondaryUse::Research);

        let (fresh_attestation, fresh_key) = authority_attestation(&mutated);
        let fresh_binding = {
            let context = binding_context();
            let (resolution, resolution_key) =
                authority_resolution(&mutated, &fresh_attestation, &context, 100, 2_000);
            mutated
                .handling
                .bind_policy_provenance_with_attestation_and_resolution(
                    b"synthetic-policy-record-1",
                    &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
                    b"synthetic-status-record-1",
                    &fresh_attestation,
                    &fresh_key,
                    &resolution,
                    &resolution_key,
                    &context,
                    150,
                )
                .unwrap()
        };

        let mut packet = NeurosemanticPacket::new_with_policy(
            18,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            NeurosemanticDataPolicy {
                handling: mutated,
                ..semantic_policy()
            },
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();

        packet.refresh_hashes().unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease().consent_epoch,
            lease_id: lease().lease_id,
        };
        assert!(message
            .validate_for_handling(
                &lease(),
                &fresh_binding,
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::Research),
                150,
            )
            .is_ok());
        assert!(message
            .validate_for_handling(
                &lease(),
                &original_binding,
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::Research),
                150,
            )
            .is_err());
    }

    #[test]
    fn handling_policy_requires_provenance_reference() {
        let mut policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: String::new(),
            policy_provenance_hash: String::new(),
            derivation_provenance_ref: "synthetic-derivation-record-1".into(),
            derivation_provenance_hash: compute_derivation_provenance_hash(
                "synthetic-derivation-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
            ),
            derivation_output_artifact_hash: synthetic_output_artifact_hash(),
            origin_jurisdiction: "ZA".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
            max_authority_resolution_age_s: 300,
        };
        assert!(!policy.validates());
        policy.policy_provenance_ref = "synthetic-policy-record-1".into();
        assert!(policy.validates());
    }

    #[test]
    fn handling_policy_requires_explicit_destination_and_retention() {
        let policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: "synthetic-policy-record-1".into(),
            policy_provenance_hash: compute_policy_provenance_hash(
                "synthetic-policy-record-1",
                b"synthetic-policy-record-1",
            ),
            derivation_provenance_ref: "synthetic-derivation-record-1".into(),
            derivation_provenance_hash: compute_derivation_provenance_hash(
                "synthetic-derivation-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
            ),
            derivation_output_artifact_hash: synthetic_output_artifact_hash(),
            origin_jurisdiction: "ZA".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["ZA".into(), "GB".into()]),
            permitted_secondary_uses: BTreeSet::from([NeurosemanticSecondaryUse::Research]),
            retention: NeurosemanticRetentionPolicy::UntilUnixS(200),
            max_authority_resolution_age_s: 300,
        };
        assert!(policy.validates());
        assert!(policy.allows_destination("GB"));
        assert!(!policy.allows_destination("US"));
        assert!(policy.allows_action(NeurosemanticHandlingAction::Persist, 150));
        assert!(!policy.allows_action(NeurosemanticHandlingAction::Persist, 200));
        assert!(policy.allows_action(
            NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::Research),
            150
        ));
        assert!(!policy.allows_action(
            NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::ModelTraining),
            150
        ));
    }

    #[test]
    fn handling_policy_rejects_noncanonical_jurisdictions() {
        let mut policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: "synthetic-policy-record-1".into(),
            policy_provenance_hash: compute_policy_provenance_hash(
                "synthetic-policy-record-1",
                b"synthetic-policy-record-1",
            ),
            derivation_provenance_ref: "synthetic-derivation-record-1".into(),
            derivation_provenance_hash: compute_derivation_provenance_hash(
                "synthetic-derivation-record-1",
                &serde_json::to_vec(&synthetic_derivation_lineage_record()).unwrap(),
            ),
            derivation_output_artifact_hash: synthetic_output_artifact_hash(),
            origin_jurisdiction: "za".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["za".into()]),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
            max_authority_resolution_age_s: 300,
        };
        assert!(!policy.validates());
        policy.origin_jurisdiction = "ZA".into();
        policy.permitted_destination_jurisdictions = BTreeSet::from(["ZA".into()]);
        assert!(policy.validates());
    }

    #[test]
    fn inference_sensitive_secondary_uses_cannot_escalate_declared_inference_classes() {
        let lease = lease();
        let mut policy = semantic_policy();
        policy.handling.permitted_secondary_uses =
            BTreeSet::from([NeurosemanticSecondaryUse::AffectiveInference]);
        let packet = NeurosemanticPacket::new_with_policy(
            109,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy.clone(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id.clone(),
        };
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(
                    NeurosemanticSecondaryUse::AffectiveInference
                ),
                150
            )
            .is_err());

        policy.inference_classes =
            BTreeSet::from([NeurosemanticInferenceClass::AffectiveState]);
        policy.permitted_secondary_uses =
            BTreeSet::from([NeurosemanticSecondaryUse::AffectiveInference]);
        let packet = NeurosemanticPacket::new_with_policy(
            110,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id,
        };
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(
                    NeurosemanticSecondaryUse::AffectiveInference
                ),
                150
            )
            .is_ok());
    }

    #[test]
    fn authorized_handling_respects_secondary_use_jurisdiction_and_retention() {
        let lease = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            107,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id.clone(),
        };

        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::Transmit,
                150
            )
            .is_ok());
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "GB",
                NeurosemanticHandlingAction::Transmit,
                150
            )
            .is_err());
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::Persist,
                200
            )
            .is_err());

        let mut secondary_policy = semantic_policy();
        secondary_policy.handling.permitted_secondary_uses =
            BTreeSet::from([NeurosemanticSecondaryUse::Research]);
        let packet = NeurosemanticPacket::new_with_policy(
            108,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            secondary_policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id,
        };
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(
                    NeurosemanticSecondaryUse::Research
                ),
                150
            )
            .is_ok());
    }

    #[test]
    fn data_policy_serialization_roundtrips() {
        let policy = semantic_policy();
        let encoded = serde_json::to_vec(&policy).unwrap();
        let decoded: NeurosemanticDataPolicy = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, policy);
    }

    #[test]
    fn bounded_json_parsers_reject_oversized_artifacts() {
        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticPacket::from_json_bytes(&oversized).is_err());
        assert!(CognitiveConsentLease::from_json_bytes(&oversized,).is_err());
        let lease = lease();
        assert!(AuthorizedNeurosemanticMessage::from_json_bytes(&oversized, &lease, 150).is_err());
    }

    #[test]
    fn identifiers_are_bounded_and_nonempty() {
        let oversized = "x".repeat(MAX_NEUROSEMANTIC_ID_BYTES + 1);
        let packet = NeurosemanticPacket::new(
            105,
            oversized,
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        );
        assert!(packet.is_err());

        let mut l = lease();
        l.peer_id.clear();
        assert!(l.validate().is_err());
        l.peer_id = "x".repeat(MAX_NEUROSEMANTIC_ID_BYTES + 1);
        assert!(l.validate().is_err());
    }

    #[test]
    fn bounded_json_parsers_roundtrip_valid_artifacts() {
        let lease = lease();
        let encoded_lease = serde_json::to_vec(&lease).unwrap();
        assert_eq!(CognitiveConsentLease::from_json_bytes(&encoded_lease).unwrap(), lease);

        let packet = NeurosemanticPacket::new_with_policy(
            106,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        ).unwrap();
        let encoded_packet = serde_json::to_vec(&packet).unwrap();
        assert_eq!(NeurosemanticPacket::from_json_bytes(&encoded_packet).unwrap(), packet);

        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id.clone(),
        };
        let encoded_message = serde_json::to_vec(&message).unwrap();
        assert_eq!(AuthorizedNeurosemanticMessage::from_json_bytes(&encoded_message, &lease, 150).unwrap(), message);
    }
    #[test]
    fn policy_binds_payload_to_declared_data_class() {
        let result = NeurosemanticPacket::new_with_policy(
            101,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            NeurosemanticDataPolicy {
                data_class: NeurosemanticDataClass::DerivedNeuralFeature,
                ..semantic_policy()
            },
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        );
        assert!(result.is_err());

        let mut packet = NeurosemanticPacket::new_with_policy(
            102,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        packet.data_policy.data_class = NeurosemanticDataClass::DecodedClaim;
        packet.refresh_hashes().unwrap();
        assert!(packet.validate_integrity().is_err());
    }

    #[test]
    fn opaque_legacy_structured_representation_cannot_cross_policy_boundary() {
        assert_eq!(
            NeurosemanticPayload::StructuredRepresentation(b"hello".to_vec())
                .intrinsic_data_class(),
            None
        );
        assert!(NeurosemanticPacket::new_with_policy(
            103,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::StructuredRepresentation(b"hello".to_vec()),
        )
        .is_err());
    }

    #[test]
    fn decoded_claim_is_a_typed_transport_payload() {
        let policy = NeurosemanticDataPolicy {
            data_class: NeurosemanticDataClass::DecodedClaim,
            ..semantic_policy()
        };
        let packet = NeurosemanticPacket::new_with_policy(
            104,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::DecodedClaim(b"claim".to_vec()),
        )
        .unwrap();
        assert!(packet.validate_integrity().is_ok());
    }

    #[test]
    fn legacy_packet_data_policy_defaults_to_unknown_and_cannot_authorize() {
        let packet = NeurosemanticPacket::new(
            1,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        assert_eq!(packet.data_policy.data_class, NeurosemanticDataClass::Unknown);
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: 7,
            lease_id: "lease-1".into(),
        };
        assert!(message.validate(&lease(), 150).is_err());
    }

    #[test]
    fn packet_policy_purpose_must_match_packet_purpose() {
        let mut policy = semantic_policy();
        policy.permitted_purposes = BTreeSet::from([CommunicationPurpose::Research]);
        let packet = NeurosemanticPacket::new_with_policy(
            2,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: 7,
            lease_id: "lease-1".into(),
        };
        assert!(message.validate(&lease(), 150).is_err());
    }
    #[test]
    fn consent_is_deny_by_default() {
        let l = lease();
        assert!(!l.authorizes(
            CognitiveChannel::Affective,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            150
        ));
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::Research,
            "peer",
            150
        ));
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "other-peer",
            150
        ));
    }

    #[test]
    fn expiry_and_revocation_block_authority() {
        let mut l = lease();
        assert!(l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            150
        ));
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            200
        ));
        l.revoked = true;
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            150
        ));
    }

    #[test]
    fn policy_provenance_ref_is_bound_to_packet_integrity() {
        let mut packet = NeurosemanticPacket::new_with_policy(
            17,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        assert!(packet.validate_integrity().is_ok());
        packet.data_policy.handling.policy_provenance_ref = "synthetic-policy-record-2".into();
        assert!(packet.validate_integrity().is_err());
    }

    #[test]
    fn policy_provenance_hash_is_bound_to_packet_integrity() {
        let mut packet = NeurosemanticPacket::new_with_policy(
            17,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        assert!(packet.validate_integrity().is_ok());
        packet.data_policy.handling.policy_provenance_hash = compute_policy_provenance_hash(
            "synthetic-policy-record-1",
            b"synthetic-policy-record-2",
        );
        assert!(packet.validate_integrity().is_err());
    }

    #[test]
    fn packet_hashes_detect_tampering() {
        let mut packet = NeurosemanticPacket::new(
            1,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.91,
            NeurosemanticPayload::Hypervector(vec![1, -1, 1]),
        )
        .unwrap();

        assert!(packet.validate_integrity().is_ok());
        if let NeurosemanticPayload::Hypervector(values) = &mut packet.payload {
            values[0] = -values[0];
        }
        assert!(packet.validate_integrity().is_err());
    }

    #[test]
    fn authorized_message_requires_scope_epoch_and_subject_binding() {
        let packet = NeurosemanticPacket::new_with_policy(
            3,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.88,
            NeurosemanticPayload::Hypervector(vec![1, -1, 1]),
        )
        .unwrap();

        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: 7,
            lease_id: "lease-1".into(),
        };
        assert!(message.validate(&lease(), 150).is_ok());

        let mut wrong_epoch = message.clone();
        wrong_epoch.consent_epoch = 8;
        assert!(wrong_epoch.validate(&lease(), 150).is_err());
    }

    #[test]
    fn read_direction_flows_from_subject_to_peer() {
        let base = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            9,
            "subject",
            "peer",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.77,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();

        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: base.consent_epoch,
            lease_id: base.lease_id,
        };

        assert!(message.validate(&lease(), 150).is_ok());
    }

    #[test]
    fn read_direction_rejects_peer_to_subject_packet() {
        let base = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            10,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.77,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();

        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: base.consent_epoch,
            lease_id: base.lease_id,
        };

        assert!(message.validate(&lease(), 150).is_err());
    }

    #[test]
    fn nonfinite_derived_features_are_rejected() {
        let result = NeurosemanticPacket::new(
            11,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::DerivedNeuralFeature(vec![0.1, f32::NAN]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn unsupported_protocol_versions_are_rejected() {
        let mut packet = NeurosemanticPacket::new(
            12,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        packet.protocol_version += 1;
        assert!(packet.validate_integrity().is_err());
    }

    #[test]
    fn payload_size_is_bounded() {
        let values = vec![1_i8; MAX_NEUROSEMANTIC_PAYLOAD_BYTES];
        let result = NeurosemanticPacket::new(
            13,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(values),
        );
        assert!(result.is_err());
    }

    #[test]
    fn zero_sequence_is_rejected_at_construction() {
        let result = NeurosemanticPacket::new(
            0,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn authorized_replay_path_requires_active_consent() {
        let base = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            14,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: base.consent_epoch,
            lease_id: base.lease_id.clone(),
        };
        let mut tracker = NeurosemanticReplayTracker::default();
        assert_eq!(
            tracker.observe_authorized(&message, &base, 150).unwrap(),
            ReplayDecision::Accept
        );
        let mut revoked = base.clone();
        revoked.revoked = true;
        revoked.revoked_at_unix_s = Some(150);
        assert!(tracker
            .observe_authorized(&message, &revoked, 150)
            .is_err());

        let mut malformed_revoked = base.clone();
        malformed_revoked.revoked = true;
        malformed_revoked.revoked_at_unix_s = None;
        assert!(malformed_revoked.validate().is_err());
    }

    #[test]
    fn replay_state_is_reclaimed_after_lease_expiry() {
        let base = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            16,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: base.consent_epoch,
            lease_id: base.lease_id.clone(),
        };
        let mut tracker = NeurosemanticReplayTracker::default();
        assert_eq!(
            tracker.observe_authorized(&message, &base, 150).unwrap(),
            ReplayDecision::Accept
        );
        assert_eq!(tracker.prune_expired(199), 0);
        assert_eq!(tracker.prune_expired(200), 1);
        assert_eq!(tracker.prune_expired(200), 0);
    }

    #[test]
    fn sensitivity_ceiling_is_enforced() {
        let base = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            15,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::HighlyPrivate,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: base.consent_epoch,
            lease_id: base.lease_id.clone(),
        };
        assert!(message.validate(&base, 150).is_err());
        let mut elevated = base.clone();
        elevated.max_write_sensitivity = CognitiveSensitivity::HighlyPrivate;
        assert!(message.validate(&elevated, 150).is_ok());
    }

    #[test]
    fn revoked_leases_require_effective_timestamp() {
        let mut l = lease();
        l.revoked = true;
        assert!(l.validate().is_err());
        l.revoked_at_unix_s = Some(150);
        assert!(l.validate().is_ok());
        assert!(l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            149
        ));
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            150
        ));
    }

    #[test]
    fn legacy_leases_default_to_public_sensitivity() {
        let lease = lease();
        let mut value = serde_json::to_value(&lease).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("max_read_sensitivity");
        object.remove("max_write_sensitivity");
        let restored: CognitiveConsentLease = serde_json::from_value(value).unwrap();
        assert_eq!(restored.max_read_sensitivity, CognitiveSensitivity::Public);
        assert_eq!(restored.max_write_sensitivity, CognitiveSensitivity::Public);
    }

    #[test]
    fn raw_neural_samples_are_not_a_payload_variant() {
        let encoded = serde_json::to_string(
            &NeurosemanticPayload::DerivedNeuralFeature(vec![0.1, 0.2]),
        )
        .unwrap();
        assert!(encoded.contains("DerivedNeuralFeature"));
        assert!(!encoded.contains("RawNeural"));
    }
}
