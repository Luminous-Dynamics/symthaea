//! Transport-neutral, read-only analysis protocol for bounded Symthaea integrations.
//!
//! This module deliberately models analytical requests and analytical artifacts only.
//! It contains no decision, authorization, certification, ledger-write, observation-
//! creation, or effect-execution output variant.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::content_hash;

pub const ANALYSIS_PROTOCOL_VERSION: &str = "symthaea.analysis/v1";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolRef {
    /// Semantic-reference profile understood by the adapter, for example
    /// `mycelix.semantic-ref/v1`. Symthaea does not reinterpret the profile.
    pub profile: String,
    pub canonical_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_commitment: Option<String>,
}

impl ProtocolRef {
    pub fn validate(&self) -> Result<(), AnalysisProtocolError> {
        if self.profile.trim().is_empty() || self.canonical_id.trim().is_empty() {
            return Err(AnalysisProtocolError::InvalidReference(
                "profile and canonical_id are required".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceState {
    Current,
    Stale,
    Unknown,
    Conflicting,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBinding {
    pub semantic_ref: ProtocolRef,
    pub schema_ref: ProtocolRef,
    pub state: SourceState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_at: Option<String>,
}

impl SourceBinding {
    fn validate(&self) -> Result<(), AnalysisProtocolError> {
        self.semantic_ref.validate()?;
        self.schema_ref.validate()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisKind {
    Diagnostic,
    Prediction,
    Counterfactual,
    Sensitivity,
    DesignSearch,
    Optimization,
    AnomalyDetection,
    OutcomeReview,
}

/// Analytical output classes only.
///
/// Intentionally absent: Observation, Decision, Authorization, CertifiedDesign,
/// LedgerEntry, ExecutionReceipt, EffectResult.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisResultKind {
    DiagnosticFinding,
    Prediction,
    CounterfactualResult,
    DesignCandidate,
    OptimizationCandidate,
    RiskOrConstraintFinding,
    Recommendation,
    Abstention,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfileRequirement {
    pub profile_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_version: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentnessPolicy {
    pub reject_stale: bool,
    pub require_known: bool,
    pub reject_conflicting: bool,
}

impl Default for CurrentnessPolicy {
    fn default() -> Self {
        Self {
            reject_stale: true,
            require_known: true,
            reject_conflicting: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRequest {
    pub protocol_version: String,
    pub request_id: String,
    pub analysis_kind: AnalysisKind,
    pub sources: Vec<SourceBinding>,
    pub objective: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    pub allowed_results: Vec<AnalysisResultKind>,
    pub model_requirement: ModelProfileRequirement,
    pub privacy_profile: String,
    pub require_calibration: bool,
    pub allow_abstention: bool,
    pub currentness_policy: CurrentnessPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requester_ref: Option<ProtocolRef>,
    #[serde(default)]
    pub provenance_refs: Vec<ProtocolRef>,
}

impl AnalysisRequest {
    pub fn computed_id(&self) -> Result<String, serde_json::Error> {
        let mut canonical = self.clone();
        canonical.request_id.clear();
        serde_json::to_vec(&canonical).map(|bytes| content_hash(&bytes))
    }

    pub fn refresh_id(&mut self) -> Result<(), serde_json::Error> {
        self.request_id = self.computed_id()?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), AnalysisProtocolError> {
        if self.protocol_version != ANALYSIS_PROTOCOL_VERSION {
            return Err(AnalysisProtocolError::UnsupportedProtocol(
                self.protocol_version.clone(),
            ));
        }
        if self.objective.trim().is_empty() {
            return Err(AnalysisProtocolError::InvalidRequest(
                "objective is required".into(),
            ));
        }
        if self.sources.is_empty() {
            return Err(AnalysisProtocolError::MissingSources);
        }
        if self.allowed_results.is_empty() {
            return Err(AnalysisProtocolError::InvalidRequest(
                "at least one allowed result kind is required".into(),
            ));
        }
        if self.model_requirement.profile_id.trim().is_empty()
            || self.privacy_profile.trim().is_empty()
        {
            return Err(AnalysisProtocolError::InvalidRequest(
                "model profile and privacy profile are required".into(),
            ));
        }

        let mut refs = BTreeSet::new();
        for source in &self.sources {
            source.validate()?;
            if !refs.insert(source.semantic_ref.clone()) {
                return Err(AnalysisProtocolError::DuplicateSource(
                    source.semantic_ref.canonical_id.clone(),
                ));
            }
            match source.state {
                SourceState::Stale if self.currentness_policy.reject_stale => {
                    return Err(AnalysisProtocolError::StaleSource(
                        source.semantic_ref.canonical_id.clone(),
                    ));
                }
                SourceState::Unknown if self.currentness_policy.require_known => {
                    return Err(AnalysisProtocolError::UnknownSourceState(
                        source.semantic_ref.canonical_id.clone(),
                    ));
                }
                SourceState::Conflicting if self.currentness_policy.reject_conflicting => {
                    return Err(AnalysisProtocolError::ConflictingSource(
                        source.semantic_ref.canonical_id.clone(),
                    ));
                }
                _ => {}
            }
        }

        if let Some(requester) = &self.requester_ref {
            requester.validate()?;
        }
        for provenance in &self.provenance_refs {
            provenance.validate()?;
        }

        let expected = self
            .computed_id()
            .map_err(|error| AnalysisProtocolError::Serialization(error.to_string()))?;
        if self.request_id != expected {
            return Err(AnalysisProtocolError::InvalidRequestIdentity);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub profile_id: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_commitment: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UncertaintyProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<[f64; 2]>,
}

impl UncertaintyProfile {
    fn validate(&self) -> Result<(), AnalysisProtocolError> {
        if let Some(confidence) = self.confidence
            && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
        {
            return Err(AnalysisProtocolError::InvalidUncertainty(
                "confidence must be finite and in [0, 1]".into(),
            ));
        }
        if let Some([low, high]) = self.interval
            && (!low.is_finite() || !high.is_finite() || low > high)
        {
            return Err(AnalysisProtocolError::InvalidUncertainty(
                "interval must be finite and ordered".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimCeiling {
    Exploratory,
    DerivedAnalysis,
    CalibratedEstimate,
    /// Verifies only the declared computation/profile, not real-world assumptions or truth.
    VerifiedComputation,
}

/// Analysis artifacts have no institutional or execution authority.
/// The single-variant enum makes authority widening an explicit protocol change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisAuthority {
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisArtifact {
    pub protocol_version: String,
    pub artifact_id: String,
    pub request_id: String,
    pub input_refs: Vec<ProtocolRef>,
    pub model: ModelIdentity,
    pub derivation_profile: String,
    pub result_kind: AnalysisResultKind,
    pub result: Value,
    #[serde(default)]
    pub assumptions: Vec<String>,
    pub uncertainty: UncertaintyProfile,
    #[serde(default)]
    pub applicability: BTreeMap<String, String>,
    #[serde(default)]
    pub limitations: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_receipt: Option<ProtocolRef>,
    pub claim_ceiling: ClaimCeiling,
    pub authority: AnalysisAuthority,
    #[serde(default)]
    pub provenance_refs: Vec<ProtocolRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

impl AnalysisArtifact {
    pub fn computed_id(&self) -> Result<String, serde_json::Error> {
        let mut canonical = self.clone();
        canonical.artifact_id.clear();
        serde_json::to_vec(&canonical).map(|bytes| content_hash(&bytes))
    }

    pub fn refresh_id(&mut self) -> Result<(), serde_json::Error> {
        self.artifact_id = self.computed_id()?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), AnalysisProtocolError> {
        if self.protocol_version != ANALYSIS_PROTOCOL_VERSION {
            return Err(AnalysisProtocolError::UnsupportedProtocol(
                self.protocol_version.clone(),
            ));
        }
        if self.request_id.trim().is_empty()
            || self.input_refs.is_empty()
            || self.model.profile_id.trim().is_empty()
            || self.model.version.trim().is_empty()
            || self.derivation_profile.trim().is_empty()
        {
            return Err(AnalysisProtocolError::InvalidArtifact(
                "request, inputs, model identity, and derivation profile are required".into(),
            ));
        }
        self.uncertainty.validate()?;
        for input in &self.input_refs {
            input.validate()?;
        }
        if let Some(proof) = &self.proof_receipt {
            proof.validate()?;
        }
        for provenance in &self.provenance_refs {
            provenance.validate()?;
        }

        let expected = self
            .computed_id()
            .map_err(|error| AnalysisProtocolError::Serialization(error.to_string()))?;
        if self.artifact_id != expected {
            return Err(AnalysisProtocolError::InvalidArtifactIdentity);
        }
        Ok(())
    }

    pub fn validate_against_request(
        &self,
        request: &AnalysisRequest,
    ) -> Result<(), AnalysisProtocolError> {
        request.validate()?;
        self.validate()?;
        if self.request_id != request.request_id {
            return Err(AnalysisProtocolError::RequestArtifactMismatch(
                "request identity differs".into(),
            ));
        }
        if !request.allowed_results.contains(&self.result_kind) {
            return Err(AnalysisProtocolError::ResultKindNotAllowed(self.result_kind));
        }
        if self.model.profile_id != request.model_requirement.profile_id {
            return Err(AnalysisProtocolError::RequestArtifactMismatch(
                "model profile differs".into(),
            ));
        }
        if let Some(version) = &request.model_requirement.exact_version
            && &self.model.version != version
        {
            return Err(AnalysisProtocolError::RequestArtifactMismatch(
                "model version differs".into(),
            ));
        }
        if request.require_calibration && self.uncertainty.calibration_profile.is_none() {
            return Err(AnalysisProtocolError::MissingCalibration);
        }
        if !request.allow_abstention
            && matches!(
                self.result_kind,
                AnalysisResultKind::Abstention | AnalysisResultKind::Unknown
            )
        {
            return Err(AnalysisProtocolError::ResultKindNotAllowed(self.result_kind));
        }

        let expected_inputs: Vec<_> = request
            .sources
            .iter()
            .map(|source| source.semantic_ref.clone())
            .collect();
        if self.input_refs != expected_inputs {
            return Err(AnalysisProtocolError::RequestArtifactMismatch(
                "artifact input frontier differs from request sources".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnalysisProtocolError {
    UnsupportedProtocol(String),
    InvalidReference(String),
    InvalidRequest(String),
    InvalidRequestIdentity,
    MissingSources,
    DuplicateSource(String),
    StaleSource(String),
    UnknownSourceState(String),
    ConflictingSource(String),
    InvalidArtifact(String),
    InvalidArtifactIdentity,
    InvalidUncertainty(String),
    Serialization(String),
    RequestArtifactMismatch(String),
    ResultKindNotAllowed(AnalysisResultKind),
    MissingCalibration,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic(id: &str) -> ProtocolRef {
        ProtocolRef {
            profile: "mycelix.semantic-ref/v1".into(),
            canonical_id: id.into(),
            version: Some("1".into()),
            content_commitment: None,
        }
    }

    fn schema(id: &str) -> ProtocolRef {
        ProtocolRef {
            profile: "mycelix.schema-ref/v1".into(),
            canonical_id: id.into(),
            version: Some("1".into()),
            content_commitment: None,
        }
    }

    fn request() -> AnalysisRequest {
        let mut request = AnalysisRequest {
            protocol_version: ANALYSIS_PROTOCOL_VERSION.into(),
            request_id: String::new(),
            analysis_kind: AnalysisKind::Prediction,
            sources: vec![SourceBinding {
                semantic_ref: semantic("water.observation.pressure"),
                schema_ref: schema("water.pressure-observation"),
                state: SourceState::Current,
                observed_at: Some("2026-09-27T10:00:00Z".into()),
            }],
            objective: "Estimate near-term pressure behavior".into(),
            constraints: vec!["read-only".into()],
            allowed_results: vec![
                AnalysisResultKind::Prediction,
                AnalysisResultKind::Abstention,
                AnalysisResultKind::Unknown,
            ],
            model_requirement: ModelProfileRequirement {
                profile_id: "symthaea.water.forecast".into(),
                exact_version: Some("1.0.0".into()),
            },
            privacy_profile: "synthetic-public".into(),
            require_calibration: true,
            allow_abstention: true,
            currentness_policy: CurrentnessPolicy::default(),
            valid_until: None,
            requester_ref: None,
            provenance_refs: Vec::new(),
        };
        request.refresh_id().unwrap();
        request
    }

    fn artifact(request: &AnalysisRequest) -> AnalysisArtifact {
        let mut artifact = AnalysisArtifact {
            protocol_version: ANALYSIS_PROTOCOL_VERSION.into(),
            artifact_id: String::new(),
            request_id: request.request_id.clone(),
            input_refs: request
                .sources
                .iter()
                .map(|source| source.semantic_ref.clone())
                .collect(),
            model: ModelIdentity {
                profile_id: "symthaea.water.forecast".into(),
                version: "1.0.0".into(),
                model_commitment: Some("model:abc".into()),
            },
            derivation_profile: "forecast-v1".into(),
            result_kind: AnalysisResultKind::Prediction,
            result: serde_json::json!({"pressure_kpa": 410.0}),
            assumptions: vec!["synthetic fixture".into()],
            uncertainty: UncertaintyProfile {
                calibration_profile: Some("heldout-brier-v1".into()),
                confidence: Some(0.72),
                interval: Some([390.0, 430.0]),
            },
            applicability: BTreeMap::new(),
            limitations: vec!["not a source observation".into()],
            proof_receipt: None,
            claim_ceiling: ClaimCeiling::CalibratedEstimate,
            authority: AnalysisAuthority::None,
            provenance_refs: Vec::new(),
            created_at: None,
        };
        artifact.refresh_id().unwrap();
        artifact
    }

    #[test]
    fn request_and_artifact_round_trip_with_no_authority_lane() {
        let request = request();
        request.validate().unwrap();
        let artifact = artifact(&request);
        artifact.validate_against_request(&request).unwrap();

        let encoded = serde_json::to_value(&artifact).unwrap();
        assert_eq!(encoded["authority"], "none");
        let decoded: AnalysisArtifact = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.authority, AnalysisAuthority::None);
    }

    #[test]
    fn missing_source_frontier_is_rejected() {
        let mut request = request();
        request.sources.clear();
        request.refresh_id().unwrap();
        assert_eq!(request.validate(), Err(AnalysisProtocolError::MissingSources));
    }

    #[test]
    fn stale_unknown_and_conflicting_sources_fail_closed_by_default() {
        for (state, expected) in [
            (
                SourceState::Stale,
                AnalysisProtocolError::StaleSource("water.observation.pressure".into()),
            ),
            (
                SourceState::Unknown,
                AnalysisProtocolError::UnknownSourceState("water.observation.pressure".into()),
            ),
            (
                SourceState::Conflicting,
                AnalysisProtocolError::ConflictingSource("water.observation.pressure".into()),
            ),
        ] {
            let mut request = request();
            request.sources[0].state = state;
            request.refresh_id().unwrap();
            assert_eq!(request.validate(), Err(expected));
        }
    }

    #[test]
    fn duplicate_source_identity_is_rejected() {
        let mut request = request();
        request.sources.push(request.sources[0].clone());
        request.refresh_id().unwrap();
        assert!(matches!(
            request.validate(),
            Err(AnalysisProtocolError::DuplicateSource(_))
        ));
    }

    #[test]
    fn protocol_has_no_observation_decision_authorization_or_certification_result_kind() {
        for kind in [
            AnalysisResultKind::DiagnosticFinding,
            AnalysisResultKind::Prediction,
            AnalysisResultKind::CounterfactualResult,
            AnalysisResultKind::DesignCandidate,
            AnalysisResultKind::OptimizationCandidate,
            AnalysisResultKind::RiskOrConstraintFinding,
            AnalysisResultKind::Recommendation,
            AnalysisResultKind::Abstention,
            AnalysisResultKind::Unknown,
        ] {
            let encoded = serde_json::to_string(&kind).unwrap();
            assert!(!encoded.contains("observation"));
            assert!(!encoded.contains("decision"));
            assert!(!encoded.contains("authorization"));
            assert!(!encoded.contains("certified_design"));
            assert!(!encoded.contains("ledger"));
            assert!(!encoded.contains("effect"));
        }
    }

    #[test]
    fn request_schema_cannot_carry_oracle_answer_fields() {
        let encoded = serde_json::to_value(request()).unwrap();
        let keys = encoded.as_object().unwrap();
        for forbidden in ["oracle", "answer_key", "expected_outcome", "expected_result"] {
            assert!(!keys.contains_key(forbidden));
        }
    }

    #[test]
    fn unknown_oracle_field_is_rejected_by_serde() {
        let mut encoded = serde_json::to_value(request()).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .insert("expected_outcome".into(), serde_json::json!("pressure drops"));
        assert!(serde_json::from_value::<AnalysisRequest>(encoded).is_err());
    }

    #[test]
    fn model_version_changes_artifact_identity() {
        let request = request();
        let a = artifact(&request);
        let mut b = a.clone();
        b.model.version = "1.0.1".into();
        b.refresh_id().unwrap();
        assert_ne!(a.artifact_id, b.artifact_id);
    }

    #[test]
    fn invalid_confidence_is_rejected() {
        let request = request();
        let mut artifact = artifact(&request);
        artifact.uncertainty.confidence = Some(1.5);
        artifact.refresh_id().unwrap();
        assert!(matches!(
            artifact.validate(),
            Err(AnalysisProtocolError::InvalidUncertainty(_))
        ));
    }

    #[test]
    fn exact_request_frontier_and_model_generation_are_enforced() {
        let request = request();
        let mut artifact = artifact(&request);
        artifact.input_refs[0].canonical_id = "different.source".into();
        artifact.refresh_id().unwrap();
        assert!(matches!(
            artifact.validate_against_request(&request),
            Err(AnalysisProtocolError::RequestArtifactMismatch(_))
        ));

        let mut artifact = artifact(&request);
        artifact.model.version = "2.0.0".into();
        artifact.refresh_id().unwrap();
        assert!(matches!(
            artifact.validate_against_request(&request),
            Err(AnalysisProtocolError::RequestArtifactMismatch(_))
        ));
    }

    #[test]
    fn proof_receipt_does_not_widen_claim_or_authority() {
        let request = request();
        let mut artifact = artifact(&request);
        artifact.proof_receipt = Some(ProtocolRef {
            profile: "symthaea.proof-receipt/v1".into(),
            canonical_id: "proof:123".into(),
            version: Some("1".into()),
            content_commitment: Some("sha256:abc".into()),
        });
        artifact.claim_ceiling = ClaimCeiling::VerifiedComputation;
        artifact.refresh_id().unwrap();
        artifact.validate_against_request(&request).unwrap();
        assert_eq!(artifact.authority, AnalysisAuthority::None);
    }
}
