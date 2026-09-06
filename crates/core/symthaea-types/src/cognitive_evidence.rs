// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Canonical cognitive proposal and evidence-authority types for RCA v1.
//!
//! This module deliberately separates four transitions:
//!
//! ```text
//! cognitive production
//!         !=
//! epistemic admission
//!         !=
//! action authority
//!         !=
//! self-improvement promotion
//! ```
//!
//! A cognitive subsystem may produce a proposal or evidence record. That record
//! must validate before downstream use. Validation still does not grant belief,
//! action, or recursive-improvement authority.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;

/// Current serialized schema version for RCA v1 cognitive evidence.
pub const COGNITIVE_EVIDENCE_SCHEMA_VERSION: u16 = 1;

/// Fixed-point scale for confidence and uncertainty values.
pub const COGNITIVE_PROBABILITY_SCALE: u32 = 1_000_000;

/// Epistemic origin of one cognitive evidence record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveEvidenceAuthorityV1 {
    /// Deterministic or stochastic software/test fixture.
    SyntheticFixture,
    /// Inference produced internally by a cognitive mechanism.
    InternalInference,
    /// Result produced by an explicit internal simulation/model execution.
    InternalSimulation,
    /// Claim retrieved from an external source but not itself an observation.
    RetrievedExternalClaim,
    /// Observation admitted as empirical input at an external evidence boundary.
    EmpiricalObservation,
    /// Derivation produced by an identified formal verifier/proof system.
    FormalDerivation,
}

impl CognitiveEvidenceAuthorityV1 {
    const fn requires_model_identity(self) -> bool {
        matches!(self, Self::InternalSimulation)
    }

    const fn requires_source_digest(self) -> bool {
        matches!(
            self,
            Self::RetrievedExternalClaim | Self::EmpiricalObservation | Self::FormalDerivation
        )
    }

    const fn requires_observation_identity(self) -> bool {
        matches!(self, Self::EmpiricalObservation)
    }

    const fn requires_formal_verifier(self) -> bool {
        matches!(self, Self::FormalDerivation)
    }

    /// Whether this authority may be admitted for one bounded downstream use.
    ///
    /// No evidence authority in RCA-001 can directly grant canonical-belief,
    /// external-action, or self-improvement-promotion authority.
    pub const fn permits(self, use_case: CognitiveEvidenceUseV1) -> bool {
        match use_case {
            CognitiveEvidenceUseV1::SoftwareQualification => true,
            CognitiveEvidenceUseV1::HypothesisGeneration => true,
            CognitiveEvidenceUseV1::ModelBehavior => matches!(
                self,
                Self::InternalInference | Self::InternalSimulation
            ),
            CognitiveEvidenceUseV1::Deliberation => !matches!(self, Self::SyntheticFixture),
            CognitiveEvidenceUseV1::BeliefSupport => matches!(
                self,
                Self::InternalInference
                    | Self::RetrievedExternalClaim
                    | Self::EmpiricalObservation
                    | Self::FormalDerivation
            ),
            CognitiveEvidenceUseV1::SelfModelUpdate => matches!(
                self,
                Self::InternalInference | Self::InternalSimulation | Self::EmpiricalObservation
            ),
            CognitiveEvidenceUseV1::ImprovementEvaluation => matches!(
                self,
                Self::InternalInference
                    | Self::InternalSimulation
                    | Self::EmpiricalObservation
                    | Self::FormalDerivation
            ),
            CognitiveEvidenceUseV1::CanonicalBeliefAdmission
            | CognitiveEvidenceUseV1::ActionAuthority
            | CognitiveEvidenceUseV1::SelfImprovementPromotion => false,
        }
    }
}

/// Explicit downstream use requested for cognitive evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveEvidenceUseV1 {
    SoftwareQualification,
    HypothesisGeneration,
    ModelBehavior,
    Deliberation,
    BeliefSupport,
    SelfModelUpdate,
    ImprovementEvaluation,
    /// Reserved transition. No RCA-001 evidence authority grants this directly.
    CanonicalBeliefAdmission,
    /// Reserved transition. Cognitive evidence is never external-effect authority.
    ActionAuthority,
    /// Reserved transition. Self-generated evidence cannot promote self-modification.
    SelfImprovementPromotion,
}

/// One typed reference to evidence supporting or challenging a cognitive proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveEvidenceRefV1 {
    pub schema_version: u16,
    pub authority: CognitiveEvidenceAuthorityV1,
    /// Human-readable source/proposer identity.
    pub source: String,
    /// Optional immutable/reviewable source revision.
    pub source_version: Option<String>,
    /// Digest of the exact claim or observation represented by this record.
    pub claim_digest: String,
    /// Digest of the external/proof source artifact when required.
    pub source_digest: Option<String>,
    /// Explicit model identity for internal simulations.
    pub model_id: Option<String>,
    /// Explicit model revision for internal simulations.
    pub model_version: Option<String>,
    /// Observation identifier for empirical evidence.
    pub observation_id: Option<String>,
    /// Formal verifier/proof-system identity for formal derivations.
    pub formal_verifier: Option<String>,
    /// Optional caller-independent freshness generation/epoch supplied by the
    /// evidence-owning boundary. This field is metadata, not currentness proof.
    pub freshness_epoch: Option<u64>,
}

impl CognitiveEvidenceRefV1 {
    /// Validate schema, authority-specific metadata, and immutable digests.
    pub fn validate(self) -> Result<ValidatedCognitiveEvidenceRefV1, CognitiveEvidenceError> {
        ValidatedCognitiveEvidenceRefV1::try_from(self)
    }
}

/// Evidence reference whose structural authority contract has been validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedCognitiveEvidenceRefV1(CognitiveEvidenceRefV1);

impl ValidatedCognitiveEvidenceRefV1 {
    pub const fn authority(&self) -> CognitiveEvidenceAuthorityV1 {
        self.0.authority
    }

    pub fn claim_digest(&self) -> &str {
        &self.0.claim_digest
    }

    pub fn as_raw(&self) -> &CognitiveEvidenceRefV1 {
        &self.0
    }

    /// Consume this evidence record and admit it for one bounded use.
    ///
    /// The result deliberately does not implement `Deserialize`; persisted data
    /// must cross validation/admission again.
    pub fn authorize(
        self,
        use_case: CognitiveEvidenceUseV1,
    ) -> Result<AdmittedCognitiveEvidenceV1, CognitiveEvidenceAdmissionError> {
        if self.authority().permits(use_case) {
            Ok(AdmittedCognitiveEvidenceV1 {
                evidence: self,
                use_case,
            })
        } else {
            Err(CognitiveEvidenceAdmissionError {
                authority: self.authority(),
                requested_use: use_case,
            })
        }
    }
}

impl TryFrom<CognitiveEvidenceRefV1> for ValidatedCognitiveEvidenceRefV1 {
    type Error = CognitiveEvidenceError;

    fn try_from(value: CognitiveEvidenceRefV1) -> Result<Self, Self::Error> {
        if value.schema_version != COGNITIVE_EVIDENCE_SCHEMA_VERSION {
            return Err(CognitiveEvidenceError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        require_nonempty(&value.source, CognitiveEvidenceError::MissingSource)?;
        validate_digest(&value.claim_digest)?;

        if let Some(source_digest) = value.source_digest.as_deref() {
            validate_digest(source_digest)?;
        } else if value.authority.requires_source_digest() {
            return Err(CognitiveEvidenceError::MissingSourceDigest {
                authority: value.authority,
            });
        }

        if value.authority.requires_model_identity() {
            require_optional_nonempty(
                value.model_id.as_deref(),
                CognitiveEvidenceError::MissingModelId,
            )?;
            require_optional_nonempty(
                value.model_version.as_deref(),
                CognitiveEvidenceError::MissingModelVersion,
            )?;
        }

        if value.authority.requires_observation_identity() {
            require_optional_nonempty(
                value.observation_id.as_deref(),
                CognitiveEvidenceError::MissingObservationId,
            )?;
        }

        if value.authority.requires_formal_verifier() {
            require_optional_nonempty(
                value.formal_verifier.as_deref(),
                CognitiveEvidenceError::MissingFormalVerifier,
            )?;
        }

        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedCognitiveEvidenceRefV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        CognitiveEvidenceRefV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

/// Affine-style capability proving admission for one explicit epistemic use.
///
/// This is intentionally non-serializable and non-deserializable. It is not a
/// canonical-belief token, action permit, or self-improvement permit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedCognitiveEvidenceV1 {
    evidence: ValidatedCognitiveEvidenceRefV1,
    use_case: CognitiveEvidenceUseV1,
}

impl AdmittedCognitiveEvidenceV1 {
    pub const fn use_case(&self) -> CognitiveEvidenceUseV1 {
        self.use_case
    }

    pub fn evidence(&self) -> &ValidatedCognitiveEvidenceRefV1 {
        &self.evidence
    }

    pub fn into_evidence(self) -> ValidatedCognitiveEvidenceRefV1 {
        self.evidence
    }
}

/// Cognitive operation available to a future metacognitive controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetaActionV1 {
    Commit,
    ThinkMore,
    RetrieveMemory,
    SearchExternally,
    RunSimulation,
    InvokeCausalReasoner,
    InvokeFormalVerifier,
    GenerateAlternatives,
    SeekCounterexample,
    Critique,
    CrossCheck,
    AskHuman,
    Replan,
    Abstain,
}

/// Raw cognitive proposal before structural validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveProposalV1 {
    pub schema_version: u16,
    /// Immutable content-addressed proposal identifier.
    pub proposal_id: String,
    /// Cognitive subsystem/agent that produced the proposal.
    pub proposer: String,
    /// Digest of the exact proposition or recommendation payload.
    pub proposition_digest: String,
    /// Downstream use the proposer is requesting.
    pub requested_use: CognitiveEvidenceUseV1,
    /// Fixed-point confidence in [0, 1_000_000].
    pub confidence_ppm: u32,
    /// Fixed-point uncertainty in [0, 1_000_000]. Not required to complement confidence.
    pub uncertainty_ppm: u32,
    #[serde(default)]
    pub assumptions: Vec<String>,
    #[serde(default)]
    pub evidence: Vec<ValidatedCognitiveEvidenceRefV1>,
    pub expected_compute_microunits: Option<u64>,
    pub expected_latency_ms: Option<u64>,
    pub reversible: Option<bool>,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

impl CognitiveProposalV1 {
    pub fn validate(self) -> Result<ValidatedCognitiveProposalV1, CognitiveProposalError> {
        ValidatedCognitiveProposalV1::try_from(self)
    }
}

/// Structurally valid cognitive proposal. This still carries no truth or action authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedCognitiveProposalV1(CognitiveProposalV1);

impl ValidatedCognitiveProposalV1 {
    pub fn proposal_id(&self) -> &str {
        &self.0.proposal_id
    }

    pub fn proposer(&self) -> &str {
        &self.0.proposer
    }

    pub const fn requested_use(&self) -> CognitiveEvidenceUseV1 {
        self.0.requested_use
    }

    pub fn evidence(&self) -> &[ValidatedCognitiveEvidenceRefV1] {
        &self.0.evidence
    }

    pub fn as_raw(&self) -> &CognitiveProposalV1 {
        &self.0
    }
}

impl TryFrom<CognitiveProposalV1> for ValidatedCognitiveProposalV1 {
    type Error = CognitiveProposalError;

    fn try_from(value: CognitiveProposalV1) -> Result<Self, Self::Error> {
        if value.schema_version != COGNITIVE_EVIDENCE_SCHEMA_VERSION {
            return Err(CognitiveProposalError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        validate_digest(&value.proposal_id).map_err(CognitiveProposalError::Evidence)?;
        validate_digest(&value.proposition_digest).map_err(CognitiveProposalError::Evidence)?;
        if value.proposer.trim().is_empty() {
            return Err(CognitiveProposalError::MissingProposer);
        }
        if value.confidence_ppm > COGNITIVE_PROBABILITY_SCALE {
            return Err(CognitiveProposalError::ConfidenceOutOfRange {
                found: value.confidence_ppm,
            });
        }
        if value.uncertainty_ppm > COGNITIVE_PROBABILITY_SCALE {
            return Err(CognitiveProposalError::UncertaintyOutOfRange {
                found: value.uncertainty_ppm,
            });
        }
        reject_empty_strings(&value.assumptions, CognitiveProposalError::EmptyAssumption)?;
        reject_empty_strings(&value.dependencies, CognitiveProposalError::EmptyDependency)?;
        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedCognitiveProposalV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        CognitiveProposalV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

/// Kind of unresolved conflict between cognitive proposals/evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveDisagreementKindV1 {
    ContradictoryClaims,
    IncompatibleRecommendations,
    EvidenceConflict,
    ConfidenceConflict,
    AssumptionConflict,
}

/// First-class unresolved disagreement for workspace/metacognitive attention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveDisagreementV1 {
    pub schema_version: u16,
    pub disagreement_id: String,
    pub kind: CognitiveDisagreementKindV1,
    /// Content-addressed proposal identifiers participating in the conflict.
    pub proposal_ids: Vec<String>,
    /// Candidate information-seeking/control operations that may resolve it.
    #[serde(default)]
    pub resolving_actions: Vec<MetaActionV1>,
}

impl CognitiveDisagreementV1 {
    pub fn validate(self) -> Result<ValidatedCognitiveDisagreementV1, CognitiveDisagreementError> {
        ValidatedCognitiveDisagreementV1::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedCognitiveDisagreementV1(CognitiveDisagreementV1);

impl ValidatedCognitiveDisagreementV1 {
    pub fn as_raw(&self) -> &CognitiveDisagreementV1 {
        &self.0
    }
}

impl TryFrom<CognitiveDisagreementV1> for ValidatedCognitiveDisagreementV1 {
    type Error = CognitiveDisagreementError;

    fn try_from(value: CognitiveDisagreementV1) -> Result<Self, Self::Error> {
        if value.schema_version != COGNITIVE_EVIDENCE_SCHEMA_VERSION {
            return Err(CognitiveDisagreementError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        validate_digest(&value.disagreement_id)
            .map_err(CognitiveDisagreementError::Evidence)?;
        if value.proposal_ids.len() < 2 {
            return Err(CognitiveDisagreementError::RequiresMultipleProposals);
        }

        let mut unique = HashSet::with_capacity(value.proposal_ids.len());
        for proposal_id in &value.proposal_ids {
            validate_digest(proposal_id).map_err(CognitiveDisagreementError::Evidence)?;
            if !unique.insert(proposal_id.as_str()) {
                return Err(CognitiveDisagreementError::DuplicateProposalId);
            }
        }
        if value.resolving_actions.is_empty() {
            return Err(CognitiveDisagreementError::MissingResolvingAction);
        }
        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedCognitiveDisagreementV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        CognitiveDisagreementV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

/// Resolution status for a proposal after an observed/evaluated outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveOutcomeStatusV1 {
    Supported,
    Contradicted,
    Mixed,
    Unresolved,
    NotObserved,
}

/// Outcome record used for calibration and later self-model updates.
///
/// Recording an outcome is bookkeeping/evidence. It does not promote a belief,
/// action, architecture, or self-modification by itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveOutcomeV1 {
    pub schema_version: u16,
    pub proposal_id: String,
    pub status: CognitiveOutcomeStatusV1,
    #[serde(default)]
    pub evidence: Vec<ValidatedCognitiveEvidenceRefV1>,
    pub observed_compute_microunits: Option<u64>,
    pub observed_latency_ms: Option<u64>,
}

impl CognitiveOutcomeV1 {
    pub fn validate(self) -> Result<ValidatedCognitiveOutcomeV1, CognitiveOutcomeError> {
        ValidatedCognitiveOutcomeV1::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedCognitiveOutcomeV1(CognitiveOutcomeV1);

impl ValidatedCognitiveOutcomeV1 {
    pub fn as_raw(&self) -> &CognitiveOutcomeV1 {
        &self.0
    }
}

impl TryFrom<CognitiveOutcomeV1> for ValidatedCognitiveOutcomeV1 {
    type Error = CognitiveOutcomeError;

    fn try_from(value: CognitiveOutcomeV1) -> Result<Self, Self::Error> {
        if value.schema_version != COGNITIVE_EVIDENCE_SCHEMA_VERSION {
            return Err(CognitiveOutcomeError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        validate_digest(&value.proposal_id).map_err(CognitiveOutcomeError::Evidence)?;
        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedCognitiveOutcomeV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        CognitiveOutcomeV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

/// Evidence failed structural validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveEvidenceError {
    UnsupportedSchemaVersion { found: u16 },
    MissingSource,
    MalformedDigest,
    MissingSourceDigest { authority: CognitiveEvidenceAuthorityV1 },
    MissingModelId,
    MissingModelVersion,
    MissingObservationId,
    MissingFormalVerifier,
}

impl std::fmt::Display for CognitiveEvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported cognitive evidence schema version {found}; expected {COGNITIVE_EVIDENCE_SCHEMA_VERSION}"
            ),
            Self::MissingSource => f.write_str("cognitive evidence source must be explicit"),
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::MissingSourceDigest { authority } => write!(
                f,
                "{authority:?} cognitive evidence requires an immutable source digest"
            ),
            Self::MissingModelId => {
                f.write_str("internal simulation evidence requires an explicit model id")
            }
            Self::MissingModelVersion => {
                f.write_str("internal simulation evidence requires an explicit model version")
            }
            Self::MissingObservationId => {
                f.write_str("empirical observation evidence requires an observation id")
            }
            Self::MissingFormalVerifier => {
                f.write_str("formal derivation evidence requires a formal verifier id")
            }
        }
    }
}

impl std::error::Error for CognitiveEvidenceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CognitiveEvidenceAdmissionError {
    pub authority: CognitiveEvidenceAuthorityV1,
    pub requested_use: CognitiveEvidenceUseV1,
}

impl std::fmt::Display for CognitiveEvidenceAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} cognitive evidence is not admitted for {:?}",
            self.authority, self.requested_use
        )
    }
}

impl std::error::Error for CognitiveEvidenceAdmissionError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveProposalError {
    UnsupportedSchemaVersion { found: u16 },
    Evidence(CognitiveEvidenceError),
    MissingProposer,
    ConfidenceOutOfRange { found: u32 },
    UncertaintyOutOfRange { found: u32 },
    EmptyAssumption,
    EmptyDependency,
}

impl std::fmt::Display for CognitiveProposalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported cognitive proposal schema version {found}; expected {COGNITIVE_EVIDENCE_SCHEMA_VERSION}"
            ),
            Self::Evidence(error) => error.fmt(f),
            Self::MissingProposer => f.write_str("cognitive proposal requires a proposer"),
            Self::ConfidenceOutOfRange { found } => write!(
                f,
                "confidence {found} exceeds fixed-point scale {COGNITIVE_PROBABILITY_SCALE}"
            ),
            Self::UncertaintyOutOfRange { found } => write!(
                f,
                "uncertainty {found} exceeds fixed-point scale {COGNITIVE_PROBABILITY_SCALE}"
            ),
            Self::EmptyAssumption => f.write_str("proposal assumptions must be non-empty"),
            Self::EmptyDependency => f.write_str("proposal dependencies must be non-empty"),
        }
    }
}

impl std::error::Error for CognitiveProposalError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveDisagreementError {
    UnsupportedSchemaVersion { found: u16 },
    Evidence(CognitiveEvidenceError),
    RequiresMultipleProposals,
    DuplicateProposalId,
    MissingResolvingAction,
}

impl std::fmt::Display for CognitiveDisagreementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported cognitive disagreement schema version {found}; expected {COGNITIVE_EVIDENCE_SCHEMA_VERSION}"
            ),
            Self::Evidence(error) => error.fmt(f),
            Self::RequiresMultipleProposals => {
                f.write_str("a disagreement requires at least two distinct proposals")
            }
            Self::DuplicateProposalId => {
                f.write_str("a disagreement cannot contain duplicate proposal ids")
            }
            Self::MissingResolvingAction => {
                f.write_str("a disagreement requires at least one candidate resolving action")
            }
        }
    }
}

impl std::error::Error for CognitiveDisagreementError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveOutcomeError {
    UnsupportedSchemaVersion { found: u16 },
    Evidence(CognitiveEvidenceError),
}

impl std::fmt::Display for CognitiveOutcomeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported cognitive outcome schema version {found}; expected {COGNITIVE_EVIDENCE_SCHEMA_VERSION}"
            ),
            Self::Evidence(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for CognitiveOutcomeError {}

fn require_nonempty(value: &str, error: CognitiveEvidenceError) -> Result<(), CognitiveEvidenceError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(())
    }
}

fn require_optional_nonempty(
    value: Option<&str>,
    error: CognitiveEvidenceError,
) -> Result<(), CognitiveEvidenceError> {
    if value.is_none_or(|value| value.trim().is_empty()) {
        Err(error)
    } else {
        Ok(())
    }
}

fn reject_empty_strings(
    values: &[String],
    error: CognitiveProposalError,
) -> Result<(), CognitiveProposalError> {
    if values.iter().any(|value| value.trim().is_empty()) {
        Err(error)
    } else {
        Ok(())
    }
}

fn validate_digest(digest: &str) -> Result<(), CognitiveEvidenceError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(CognitiveEvidenceError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(CognitiveEvidenceError::MalformedDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const BLAKE_C: &str =
        "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn inference() -> CognitiveEvidenceRefV1 {
        CognitiveEvidenceRefV1 {
            schema_version: COGNITIVE_EVIDENCE_SCHEMA_VERSION,
            authority: CognitiveEvidenceAuthorityV1::InternalInference,
            source: "causal-reasoner".into(),
            source_version: Some("git:example".into()),
            claim_digest: SHA_A.into(),
            source_digest: None,
            model_id: None,
            model_version: None,
            observation_id: None,
            formal_verifier: None,
            freshness_epoch: Some(7),
        }
    }

    fn simulation() -> CognitiveEvidenceRefV1 {
        CognitiveEvidenceRefV1 {
            authority: CognitiveEvidenceAuthorityV1::InternalSimulation,
            source: "world-model".into(),
            model_id: Some("symthaea-world-model".into()),
            model_version: Some("v1".into()),
            ..inference()
        }
    }

    fn external_claim() -> CognitiveEvidenceRefV1 {
        CognitiveEvidenceRefV1 {
            authority: CognitiveEvidenceAuthorityV1::RetrievedExternalClaim,
            source: "external-source".into(),
            source_digest: Some(SHA_B.into()),
            ..inference()
        }
    }

    fn empirical() -> CognitiveEvidenceRefV1 {
        CognitiveEvidenceRefV1 {
            authority: CognitiveEvidenceAuthorityV1::EmpiricalObservation,
            source: "qualified-sensor".into(),
            source_digest: Some(BLAKE_C.into()),
            observation_id: Some("obs-001".into()),
            ..inference()
        }
    }

    fn formal() -> CognitiveEvidenceRefV1 {
        CognitiveEvidenceRefV1 {
            authority: CognitiveEvidenceAuthorityV1::FormalDerivation,
            source: "lean-bridge".into(),
            source_digest: Some(SHA_B.into()),
            formal_verifier: Some("lean4@reviewed-revision".into()),
            ..inference()
        }
    }

    fn proposal() -> CognitiveProposalV1 {
        CognitiveProposalV1 {
            schema_version: COGNITIVE_EVIDENCE_SCHEMA_VERSION,
            proposal_id: SHA_B.into(),
            proposer: "causal-reasoner".into(),
            proposition_digest: SHA_A.into(),
            requested_use: CognitiveEvidenceUseV1::Deliberation,
            confidence_ppm: 800_000,
            uncertainty_ppm: 150_000,
            assumptions: vec!["stationary environment".into()],
            evidence: vec![inference().validate().unwrap()],
            expected_compute_microunits: Some(42),
            expected_latency_ms: Some(3),
            reversible: Some(true),
            dependencies: vec!["world-model:v1".into()],
        }
    }

    #[test]
    fn internal_simulation_requires_explicit_model_identity() {
        let mut raw = simulation();
        raw.model_id = None;
        assert_eq!(raw.validate(), Err(CognitiveEvidenceError::MissingModelId));

        let mut raw = simulation();
        raw.model_version = None;
        assert_eq!(
            raw.validate(),
            Err(CognitiveEvidenceError::MissingModelVersion)
        );
    }

    #[test]
    fn external_empirical_and_formal_authorities_require_source_digest() {
        for mut raw in [external_claim(), empirical(), formal()] {
            let authority = raw.authority;
            raw.source_digest = None;
            assert_eq!(
                raw.validate(),
                Err(CognitiveEvidenceError::MissingSourceDigest { authority })
            );
        }
    }

    #[test]
    fn empirical_and_formal_metadata_fail_closed() {
        let mut raw = empirical();
        raw.observation_id = None;
        assert_eq!(
            raw.validate(),
            Err(CognitiveEvidenceError::MissingObservationId)
        );

        let mut raw = formal();
        raw.formal_verifier = None;
        assert_eq!(
            raw.validate(),
            Err(CognitiveEvidenceError::MissingFormalVerifier)
        );
    }

    #[test]
    fn digests_are_strict_and_versioned() {
        let mut raw = inference();
        raw.claim_digest = "sha256:decorative".into();
        assert_eq!(raw.validate(), Err(CognitiveEvidenceError::MalformedDigest));

        let mut raw = inference();
        raw.schema_version += 1;
        assert_eq!(
            raw.validate(),
            Err(CognitiveEvidenceError::UnsupportedSchemaVersion {
                found: COGNITIVE_EVIDENCE_SCHEMA_VERSION + 1
            })
        );
    }

    #[test]
    fn evidence_admission_is_bounded_by_authority() {
        assert!(
            simulation()
                .validate()
                .unwrap()
                .authorize(CognitiveEvidenceUseV1::ModelBehavior)
                .is_ok()
        );
        assert!(
            empirical()
                .validate()
                .unwrap()
                .authorize(CognitiveEvidenceUseV1::BeliefSupport)
                .is_ok()
        );
        assert!(
            formal()
                .validate()
                .unwrap()
                .authorize(CognitiveEvidenceUseV1::BeliefSupport)
                .is_ok()
        );
        assert!(
            CognitiveEvidenceRefV1 {
                authority: CognitiveEvidenceAuthorityV1::SyntheticFixture,
                ..inference()
            }
            .validate()
            .unwrap()
            .authorize(CognitiveEvidenceUseV1::Deliberation)
            .is_err()
        );
    }

    #[test]
    fn no_evidence_authority_self_grants_final_transitions() {
        let authorities = [
            CognitiveEvidenceAuthorityV1::SyntheticFixture,
            CognitiveEvidenceAuthorityV1::InternalInference,
            CognitiveEvidenceAuthorityV1::InternalSimulation,
            CognitiveEvidenceAuthorityV1::RetrievedExternalClaim,
            CognitiveEvidenceAuthorityV1::EmpiricalObservation,
            CognitiveEvidenceAuthorityV1::FormalDerivation,
        ];
        let forbidden = [
            CognitiveEvidenceUseV1::CanonicalBeliefAdmission,
            CognitiveEvidenceUseV1::ActionAuthority,
            CognitiveEvidenceUseV1::SelfImprovementPromotion,
        ];

        for authority in authorities {
            for use_case in forbidden {
                assert!(!authority.permits(use_case), "{authority:?} granted {use_case:?}");
            }
        }
    }

    #[test]
    fn proposal_validation_enforces_fixed_point_bounds() {
        assert!(proposal().validate().is_ok());

        let mut raw = proposal();
        raw.confidence_ppm = COGNITIVE_PROBABILITY_SCALE + 1;
        assert_eq!(
            raw.validate(),
            Err(CognitiveProposalError::ConfidenceOutOfRange {
                found: COGNITIVE_PROBABILITY_SCALE + 1
            })
        );

        let mut raw = proposal();
        raw.uncertainty_ppm = COGNITIVE_PROBABILITY_SCALE + 1;
        assert_eq!(
            raw.validate(),
            Err(CognitiveProposalError::UncertaintyOutOfRange {
                found: COGNITIVE_PROBABILITY_SCALE + 1
            })
        );
    }

    #[test]
    fn proposal_validation_rejects_empty_semantic_fields() {
        let mut raw = proposal();
        raw.proposer = " ".into();
        assert_eq!(raw.validate(), Err(CognitiveProposalError::MissingProposer));

        let mut raw = proposal();
        raw.assumptions.push(String::new());
        assert_eq!(raw.validate(), Err(CognitiveProposalError::EmptyAssumption));
    }

    #[test]
    fn validated_types_revalidate_on_deserialization() {
        let valid = inference().validate().unwrap();
        let json = serde_json::to_string(&valid).unwrap();
        let decoded: ValidatedCognitiveEvidenceRefV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, valid);

        let mut raw = inference();
        raw.claim_digest = "invalid".into();
        let json = serde_json::to_string(&raw).unwrap();
        assert!(serde_json::from_str::<ValidatedCognitiveEvidenceRefV1>(&json).is_err());

        let valid = proposal().validate().unwrap();
        let json = serde_json::to_string(&valid).unwrap();
        let decoded: ValidatedCognitiveProposalV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, valid);
    }

    #[test]
    fn disagreement_requires_distinct_proposals_and_resolution_path() {
        let valid = CognitiveDisagreementV1 {
            schema_version: COGNITIVE_EVIDENCE_SCHEMA_VERSION,
            disagreement_id: BLAKE_C.into(),
            kind: CognitiveDisagreementKindV1::ContradictoryClaims,
            proposal_ids: vec![SHA_A.into(), SHA_B.into()],
            resolving_actions: vec![MetaActionV1::SeekCounterexample],
        };
        assert!(valid.validate().is_ok());

        let duplicate = CognitiveDisagreementV1 {
            schema_version: COGNITIVE_EVIDENCE_SCHEMA_VERSION,
            disagreement_id: BLAKE_C.into(),
            kind: CognitiveDisagreementKindV1::EvidenceConflict,
            proposal_ids: vec![SHA_A.into(), SHA_A.into()],
            resolving_actions: vec![MetaActionV1::CrossCheck],
        };
        assert_eq!(
            duplicate.validate(),
            Err(CognitiveDisagreementError::DuplicateProposalId)
        );

        let no_resolution = CognitiveDisagreementV1 {
            schema_version: COGNITIVE_EVIDENCE_SCHEMA_VERSION,
            disagreement_id: BLAKE_C.into(),
            kind: CognitiveDisagreementKindV1::AssumptionConflict,
            proposal_ids: vec![SHA_A.into(), SHA_B.into()],
            resolving_actions: vec![],
        };
        assert_eq!(
            no_resolution.validate(),
            Err(CognitiveDisagreementError::MissingResolvingAction)
        );
    }

    #[test]
    fn outcome_is_bookkeeping_not_promotion_authority() {
        let outcome = CognitiveOutcomeV1 {
            schema_version: COGNITIVE_EVIDENCE_SCHEMA_VERSION,
            proposal_id: SHA_A.into(),
            status: CognitiveOutcomeStatusV1::Supported,
            evidence: vec![empirical().validate().unwrap()],
            observed_compute_microunits: Some(100),
            observed_latency_ms: Some(4),
        };
        assert!(outcome.validate().is_ok());

        // Even empirical support cannot mint one of RCA's reserved final transitions.
        let empirical = empirical().validate().unwrap();
        assert!(
            empirical
                .authorize(CognitiveEvidenceUseV1::SelfImprovementPromotion)
                .is_err()
        );
    }

    #[test]
    fn unknown_fields_fail_closed_for_raw_evidence() {
        let raw = inference();
        let mut value = serde_json::to_value(raw).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("invented_authority".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<CognitiveEvidenceRefV1>(value).is_err());
    }
}
