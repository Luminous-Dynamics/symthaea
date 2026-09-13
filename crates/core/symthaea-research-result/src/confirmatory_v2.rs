//! Endpoint-bound V2 results layered over the immutable V1 result manifest.
//!
//! A V2 object represents the terminal evidence for a campaign that began with a canonical
//! confirmatory protocol. The campaign may still end Confirmatory, Exploratory after a recorded
//! amendment/deviation, or Invalidated. Endpoint completeness survives those downgrades; only
//! genuinely confirmatory claims are allowed to carry confirmatory support semantics.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use symthaea_research_protocol::{
    classify_result, CanonicalConfirmatoryProtocol, CanonicalConfirmatoryRunBinding,
    ConfirmatoryDecisionRule, HypothesisRole, MetricRole, MetricValueSchema, ResultInterpretation,
};

use crate::result_v1::{
    ClaimDisposition, ClaimInterpretation, MetricOutcome, MetricResult, ResearchResultManifest,
    ResultArtifactKind, ResultArtifactRef, ResultClaim,
};

const RESULT_SCHEMA: &str = "symthaea-research-result/endpoint-bound-v2";

pub type Result<T> = std::result::Result<T, ConfirmatoryResultError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfirmatoryResultError {
    EmptyField(&'static str),
    DuplicateId(String),
    InvalidV1(String),
    ProtocolDigestMismatch,
    RunBindingMismatch,
    ResultBeforeRunRegistration,
    UnknownMetric(String),
    UnknownHypothesis(String),
    UnknownArtifact(String),
    UnknownClaim(String),
    UnknownEndpoint(String),
    MissingPrimaryMetric(String),
    MissingPrimaryEndpointClaim(String),
    MetricSchemaMismatch(String),
    EndpointBindingMismatch(String),
    ClaimEvidenceInadequate(String),
    ExternalRuleEvidenceMissing(String),
    NonCanonicalBindingOrder,
    Serialization(String),
    DigestMismatch,
}

impl Display for ConfirmatoryResultError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::DuplicateId(id) => write!(f, "duplicate endpoint/result id: {id}"),
            Self::InvalidV1(message) => write!(f, "V1 result manifest invalid: {message}"),
            Self::ProtocolDigestMismatch => write!(f, "result protocol identity mismatch"),
            Self::RunBindingMismatch => write!(f, "result run identity does not match V2 run binding"),
            Self::ResultBeforeRunRegistration => {
                write!(f, "result completion cannot precede run registration")
            }
            Self::UnknownMetric(id) => write!(f, "unknown metric id: {id}"),
            Self::UnknownHypothesis(id) => write!(f, "unknown hypothesis id: {id}"),
            Self::UnknownArtifact(id) => write!(f, "unknown result artifact id: {id}"),
            Self::UnknownClaim(id) => write!(f, "unknown result claim id: {id}"),
            Self::UnknownEndpoint(id) => write!(f, "unknown confirmatory endpoint id: {id}"),
            Self::MissingPrimaryMetric(id) => {
                write!(f, "preregistered primary metric {id} is absent from the result")
            }
            Self::MissingPrimaryEndpointClaim(id) => {
                write!(f, "primary endpoint {id} has no terminal claim binding")
            }
            Self::MetricSchemaMismatch(message) => write!(f, "metric schema mismatch: {message}"),
            Self::EndpointBindingMismatch(message) => {
                write!(f, "endpoint binding mismatch: {message}")
            }
            Self::ClaimEvidenceInadequate(message) => {
                write!(f, "claim evidence inadequate: {message}")
            }
            Self::ExternalRuleEvidenceMissing(message) => {
                write!(f, "external-rule evidence missing: {message}")
            }
            Self::NonCanonicalBindingOrder => {
                write!(f, "endpoint claim bindings are not in canonical endpoint-id order")
            }
            Self::Serialization(message) => {
                write!(f, "V2 result serialization failed: {message}")
            }
            Self::DigestMismatch => write!(f, "V2 result digest mismatch"),
        }
    }
}

impl Error for ConfirmatoryResultError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ConfirmatoryResultError::EmptyField(field));
    }
    Ok(())
}

fn unique_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(ConfirmatoryResultError::DuplicateId(id.to_string()));
        }
    }
    Ok(())
}

fn is_observed(outcome: &MetricOutcome) -> bool {
    matches!(
        outcome,
        MetricOutcome::Numeric { .. }
            | MetricOutcome::Boolean(_)
            | MetricOutcome::Categorical(_)
    )
}

fn outcome_kind(outcome: &MetricOutcome) -> &'static str {
    match outcome {
        MetricOutcome::Numeric { .. } => "Numeric",
        MetricOutcome::Boolean(_) => "Boolean",
        MetricOutcome::Categorical(_) => "Categorical",
        MetricOutcome::Missing { .. } => "Missing",
        MetricOutcome::NotComputed { .. } => "NotComputed",
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryClaimBinding {
    pub claim_id: String,
    pub endpoint_id: String,
    /// For an external frozen analysis rule, a confirmatory positive/negative/null conclusion
    /// requires an Analysis artifact here and on the V1 claim. This binds an output artifact to
    /// the endpoint lineage; it does not by itself attest that the frozen rule was executed
    /// faithfully.
    pub analysis_artifact_id: Option<String>,
}

impl ConfirmatoryClaimBinding {
    pub fn new(claim_id: impl Into<String>, endpoint_id: impl Into<String>) -> Result<Self> {
        let value = Self {
            claim_id: claim_id.into(),
            endpoint_id: endpoint_id.into(),
            analysis_artifact_id: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_analysis_artifact(mut self, artifact_id: impl Into<String>) -> Result<Self> {
        self.analysis_artifact_id = Some(artifact_id.into());
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<()> {
        non_empty(&self.claim_id, "endpoint claim binding claim id")?;
        non_empty(&self.endpoint_id, "endpoint claim binding endpoint id")?;
        if let Some(id) = &self.analysis_artifact_id {
            non_empty(id, "endpoint claim binding analysis artifact id")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmatoryResearchResultV2 {
    pub result: ResearchResultManifest,
    /// Canonically sorted by endpoint_id. Every primary endpoint receives one terminal binding,
    /// including campaigns that were downgraded or invalidated.
    pub claim_bindings: Vec<ConfirmatoryClaimBinding>,
    pub confirmatory_protocol_digest: String,
    pub confirmatory_run_binding_digest: String,
    pub digest: String,
}

#[derive(Serialize)]
struct DigestView<'a> {
    schema: &'static str,
    v1_result_digest: &'a str,
    claim_bindings: &'a [ConfirmatoryClaimBinding],
    confirmatory_protocol_digest: &'a str,
    confirmatory_run_binding_digest: &'a str,
}

impl ConfirmatoryResearchResultV2 {
    pub fn new(
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
        result: ResearchResultManifest,
        mut claim_bindings: Vec<ConfirmatoryClaimBinding>,
    ) -> Result<Self> {
        for binding in &claim_bindings {
            binding.validate()?;
        }
        // Binding order has no scientific meaning. Normalize it before minting identity.
        claim_bindings.sort_by(|left, right| left.endpoint_id.cmp(&right.endpoint_id));

        let mut value = Self {
            result,
            claim_bindings,
            confirmatory_protocol_digest: protocol.digest().to_string(),
            confirmatory_run_binding_digest: run_binding.digest().to_string(),
            digest: String::new(),
        };
        value.validate_semantics(protocol, run_binding)?;
        value.digest = value.compute_digest()?;
        Ok(value)
    }

    pub fn verify_digest(&self) -> Result<()> {
        if self.compute_digest()? != self.digest {
            return Err(ConfirmatoryResultError::DigestMismatch);
        }
        Ok(())
    }

    /// Revalidate digest, canonical representation, V1 evidence, V2 schemas, endpoint census and
    /// claim semantics. Imported/deserialized evidence must use this boundary.
    pub fn validate_against(
        &self,
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
    ) -> Result<()> {
        self.verify_digest()?;
        self.validate_semantics(protocol, run_binding)
    }

    fn compute_digest(&self) -> Result<String> {
        let view = DigestView {
            schema: RESULT_SCHEMA,
            v1_result_digest: &self.result.manifest_digest,
            claim_bindings: &self.claim_bindings,
            confirmatory_protocol_digest: &self.confirmatory_protocol_digest,
            confirmatory_run_binding_digest: &self.confirmatory_run_binding_digest,
        };
        let bytes = serde_json::to_vec(&view)
            .map_err(|error| ConfirmatoryResultError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    fn validate_semantics(
        &self,
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
    ) -> Result<()> {
        protocol
            .validate()
            .map_err(|error| ConfirmatoryResultError::InvalidV1(error.to_string()))?;
        run_binding
            .validate_against(protocol)
            .map_err(|error| ConfirmatoryResultError::InvalidV1(error.to_string()))?;

        if self.confirmatory_protocol_digest != protocol.digest() {
            return Err(ConfirmatoryResultError::ProtocolDigestMismatch);
        }
        if self.confirmatory_run_binding_digest != run_binding.digest() {
            return Err(ConfirmatoryResultError::RunBindingMismatch);
        }

        validate_v1_result(protocol, run_binding, &self.result)?;

        for binding in &self.claim_bindings {
            binding.validate()?;
        }
        unique_ids(self.claim_bindings.iter().map(|value| value.claim_id.as_str()))?;
        unique_ids(self.claim_bindings.iter().map(|value| value.endpoint_id.as_str()))?;
        if !self
            .claim_bindings
            .windows(2)
            .all(|window| window[0].endpoint_id < window[1].endpoint_id)
        {
            return Err(ConfirmatoryResultError::NonCanonicalBindingOrder);
        }

        let claims: HashMap<&str, &ResultClaim> = self
            .result
            .claims
            .iter()
            .map(|claim| (claim.claim_id.as_str(), claim))
            .collect();
        let endpoints: HashMap<&str, _> = protocol
            .confirmatory_endpoints()
            .iter()
            .map(|endpoint| (endpoint.endpoint_id.as_str(), endpoint))
            .collect();
        let metric_results: HashMap<&str, &MetricResult> = self
            .result
            .metrics
            .iter()
            .map(|metric| (metric.metric_id.as_str(), metric))
            .collect();
        let artifacts: HashMap<&str, &ResultArtifactRef> = self
            .result
            .artifacts
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact))
            .collect();
        let hypothesis_roles: HashMap<&str, HypothesisRole> = protocol
            .frozen_protocol()
            .protocol()
            .hypotheses
            .iter()
            .map(|hypothesis| (hypothesis.id.as_str(), hypothesis.role))
            .collect();
        let bindings_by_claim: HashMap<&str, &ConfirmatoryClaimBinding> = self
            .claim_bindings
            .iter()
            .map(|binding| (binding.claim_id.as_str(), binding))
            .collect();
        let bound_endpoints: HashSet<&str> = self
            .claim_bindings
            .iter()
            .map(|binding| binding.endpoint_id.as_str())
            .collect();

        // Any claim asserting Confirmatory interpretation must be tied to one exact frozen endpoint.
        for claim in &self.result.claims {
            if claim.interpretation == ClaimInterpretation::Confirmatory {
                let Some(hypothesis_id) = claim.hypothesis_id.as_deref() else {
                    return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                        "confirmatory claim {} must identify a hypothesis",
                        claim.claim_id
                    )));
                };
                let binding = bindings_by_claim.get(claim.claim_id.as_str()).ok_or_else(|| {
                    ConfirmatoryResultError::EndpointBindingMismatch(format!(
                        "confirmatory claim {} lacks an exact endpoint binding",
                        claim.claim_id
                    ))
                })?;
                let endpoint = endpoints
                    .get(binding.endpoint_id.as_str())
                    .ok_or_else(|| ConfirmatoryResultError::UnknownEndpoint(binding.endpoint_id.clone()))?;
                if endpoint.hypothesis_id != hypothesis_id {
                    return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                        "claim {} targets hypothesis {}, endpoint {} targets {}",
                        claim.claim_id, hypothesis_id, endpoint.endpoint_id, endpoint.hypothesis_id
                    )));
                }
            }
        }

        for binding in &self.claim_bindings {
            let claim = claims
                .get(binding.claim_id.as_str())
                .ok_or_else(|| ConfirmatoryResultError::UnknownClaim(binding.claim_id.clone()))?;
            let endpoint = endpoints
                .get(binding.endpoint_id.as_str())
                .ok_or_else(|| ConfirmatoryResultError::UnknownEndpoint(binding.endpoint_id.clone()))?;

            validate_binding_interpretation(&self.result, claim)?;

            if claim.hypothesis_id.as_deref() != Some(endpoint.hypothesis_id.as_str()) {
                return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                    "claim {} hypothesis does not match endpoint {}",
                    claim.claim_id, endpoint.endpoint_id
                )));
            }
            if claim.metric_ids.as_slice() != endpoint.metric_ids.as_slice() {
                return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                    "claim {} metric set/order does not exactly match endpoint {}",
                    claim.claim_id, endpoint.endpoint_id
                )));
            }
            if claim.disposition == ClaimDisposition::DescriptiveOnly {
                return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                    "endpoint-bound claim {} cannot be DescriptiveOnly",
                    claim.claim_id
                )));
            }

            // Schema/type validation applies to every campaign. Scientific support semantics below
            // are enforced only while the claim is genuinely confirmatory.
            let mut all_observed = true;
            for metric_id in &endpoint.metric_ids {
                let metric = metric_results.get(metric_id.as_str()).ok_or_else(|| {
                    ConfirmatoryResultError::EndpointBindingMismatch(format!(
                        "endpoint {} required metric {} is not reported",
                        endpoint.endpoint_id, metric_id
                    ))
                })?;
                all_observed &= is_observed(&metric.outcome);
            }

            if claim.interpretation == ClaimInterpretation::Confirmatory {
                validate_confirmatory_endpoint_evidence(
                    claim,
                    binding,
                    endpoint,
                    &metric_results,
                    &artifacts,
                    all_observed,
                )?;
            } else if let Some(artifact_id) = binding.analysis_artifact_id.as_deref() {
                let artifact = artifacts
                    .get(artifact_id)
                    .ok_or_else(|| ConfirmatoryResultError::UnknownArtifact(artifact_id.to_string()))?;
                if artifact.kind != ResultArtifactKind::Analysis {
                    return Err(ConfirmatoryResultError::ExternalRuleEvidenceMissing(format!(
                        "downgraded endpoint {} references non-Analysis artifact {}",
                        endpoint.endpoint_id, artifact_id
                    )));
                }
            }
        }

        // Selection completeness: every preregistered primary endpoint receives exactly one
        // terminal binding even if the campaign was downgraded or invalidated.
        for endpoint in protocol.confirmatory_endpoints() {
            if hypothesis_roles.get(endpoint.hypothesis_id.as_str()) == Some(&HypothesisRole::Primary)
                && !bound_endpoints.contains(endpoint.endpoint_id.as_str())
            {
                return Err(ConfirmatoryResultError::MissingPrimaryEndpointClaim(
                    endpoint.endpoint_id.clone(),
                ));
            }
        }

        Ok(())
    }
}

fn validate_binding_interpretation(
    result: &ResearchResultManifest,
    claim: &ResultClaim,
) -> Result<()> {
    match result.interpretation {
        ResultInterpretation::Confirmatory => {
            if claim.interpretation != ClaimInterpretation::Confirmatory {
                return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                    "endpoint-bound claim {} must be Confirmatory when the campaign is Confirmatory",
                    claim.claim_id
                )));
            }
        }
        ResultInterpretation::ExploratoryDueToPostUnblindingAmendment
        | ResultInterpretation::ExploratoryDueToPrimaryDeviation => {
            if claim.interpretation != ClaimInterpretation::Exploratory {
                return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                    "endpoint-bound claim {} must be Exploratory after campaign downgrade",
                    claim.claim_id
                )));
            }
        }
        ResultInterpretation::Invalidated => {
            if claim.interpretation != ClaimInterpretation::Invalidated {
                return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                    "endpoint-bound claim {} must be Invalidated when the campaign is invalidated",
                    claim.claim_id
                )));
            }
        }
    }
    Ok(())
}

fn validate_confirmatory_endpoint_evidence(
    claim: &ResultClaim,
    binding: &ConfirmatoryClaimBinding,
    endpoint: &symthaea_research_protocol::ConfirmatoryEndpointSpec,
    metric_results: &HashMap<&str, &MetricResult>,
    artifacts: &HashMap<&str, &ResultArtifactRef>,
    all_observed: bool,
) -> Result<()> {
    match &endpoint.decision_rule {
        ConfirmatoryDecisionRule::BooleanMustBe(expected) => {
            if binding.analysis_artifact_id.is_some() {
                return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                    "direct endpoint {} must not substitute an external analysis artifact",
                    endpoint.endpoint_id
                )));
            }
            let metric = metric_results[endpoint.metric_ids[0].as_str()];
            match &metric.outcome {
                MetricOutcome::Boolean(observed) => {
                    let required = if observed == expected {
                        ClaimDisposition::ConsistentWithHypothesis
                    } else {
                        ClaimDisposition::InconsistentWithHypothesis
                    };
                    if claim.disposition != required {
                        return Err(ConfirmatoryResultError::ClaimEvidenceInadequate(format!(
                            "claim {} disposition {:?} disagrees with Boolean endpoint {}",
                            claim.claim_id, claim.disposition, endpoint.endpoint_id
                        )));
                    }
                }
                MetricOutcome::Missing { .. } | MetricOutcome::NotComputed { .. } => {
                    require_absence_compatible_disposition(claim)?;
                }
                other => {
                    return Err(ConfirmatoryResultError::MetricSchemaMismatch(format!(
                        "endpoint {} expected Boolean result, got {}",
                        endpoint.endpoint_id,
                        outcome_kind(other)
                    )));
                }
            }
        }
        ConfirmatoryDecisionRule::CategoricalMustEqual(expected) => {
            if binding.analysis_artifact_id.is_some() {
                return Err(ConfirmatoryResultError::EndpointBindingMismatch(format!(
                    "direct endpoint {} must not substitute an external analysis artifact",
                    endpoint.endpoint_id
                )));
            }
            let metric = metric_results[endpoint.metric_ids[0].as_str()];
            match &metric.outcome {
                MetricOutcome::Categorical(observed) => {
                    let required = if observed == expected {
                        ClaimDisposition::ConsistentWithHypothesis
                    } else {
                        ClaimDisposition::InconsistentWithHypothesis
                    };
                    if claim.disposition != required {
                        return Err(ConfirmatoryResultError::ClaimEvidenceInadequate(format!(
                            "claim {} disposition {:?} disagrees with categorical endpoint {}",
                            claim.claim_id, claim.disposition, endpoint.endpoint_id
                        )));
                    }
                }
                MetricOutcome::Missing { .. } | MetricOutcome::NotComputed { .. } => {
                    require_absence_compatible_disposition(claim)?;
                }
                other => {
                    return Err(ConfirmatoryResultError::MetricSchemaMismatch(format!(
                        "endpoint {} expected Categorical result, got {}",
                        endpoint.endpoint_id,
                        outcome_kind(other)
                    )));
                }
            }
        }
        ConfirmatoryDecisionRule::ExternalFrozenAnalysisRule { .. } => {
            if !all_observed
                && matches!(
                    claim.disposition,
                    ClaimDisposition::ConsistentWithHypothesis
                        | ClaimDisposition::InconsistentWithHypothesis
                        | ClaimDisposition::NullResult
                )
            {
                return Err(ConfirmatoryResultError::ClaimEvidenceInadequate(format!(
                    "claim {} cannot report positive/negative/null support because endpoint {} has Missing/NotComputed required evidence",
                    claim.claim_id, endpoint.endpoint_id
                )));
            }

            if matches!(
                claim.disposition,
                ClaimDisposition::ConsistentWithHypothesis
                    | ClaimDisposition::InconsistentWithHypothesis
                    | ClaimDisposition::NullResult
            ) {
                let artifact_id = binding.analysis_artifact_id.as_deref().ok_or_else(|| {
                    ConfirmatoryResultError::ExternalRuleEvidenceMissing(format!(
                        "claim {} requires an Analysis output for external-rule endpoint {}",
                        claim.claim_id, endpoint.endpoint_id
                    ))
                })?;
                let artifact = artifacts
                    .get(artifact_id)
                    .ok_or_else(|| ConfirmatoryResultError::UnknownArtifact(artifact_id.to_string()))?;
                if artifact.kind != ResultArtifactKind::Analysis {
                    return Err(ConfirmatoryResultError::ExternalRuleEvidenceMissing(format!(
                        "artifact {} for endpoint {} is not an Analysis artifact",
                        artifact_id, endpoint.endpoint_id
                    )));
                }
                if !claim.artifact_ids.iter().any(|id| id == artifact_id) {
                    return Err(ConfirmatoryResultError::ExternalRuleEvidenceMissing(format!(
                        "claim {} does not reference endpoint analysis artifact {}",
                        claim.claim_id, artifact_id
                    )));
                }
            } else if let Some(artifact_id) = binding.analysis_artifact_id.as_deref() {
                let artifact = artifacts
                    .get(artifact_id)
                    .ok_or_else(|| ConfirmatoryResultError::UnknownArtifact(artifact_id.to_string()))?;
                if artifact.kind != ResultArtifactKind::Analysis {
                    return Err(ConfirmatoryResultError::ExternalRuleEvidenceMissing(format!(
                        "artifact {} for endpoint {} is not an Analysis artifact",
                        artifact_id, endpoint.endpoint_id
                    )));
                }
            }
        }
    }
    Ok(())
}

fn require_absence_compatible_disposition(claim: &ResultClaim) -> Result<()> {
    if !matches!(
        claim.disposition,
        ClaimDisposition::Inconclusive | ClaimDisposition::NotEvaluated
    ) {
        return Err(ConfirmatoryResultError::ClaimEvidenceInadequate(format!(
            "claim {} has Missing/NotComputed endpoint evidence but disposition {:?}",
            claim.claim_id, claim.disposition
        )));
    }
    Ok(())
}

fn validate_v1_result(
    protocol: &CanonicalConfirmatoryProtocol,
    run_binding: &CanonicalConfirmatoryRunBinding,
    result: &ResearchResultManifest,
) -> Result<()> {
    result
        .verify_digest()
        .map_err(|error| ConfirmatoryResultError::InvalidV1(error.to_string()))?;

    let frozen = protocol.frozen_protocol();
    if result.protocol_digest != frozen.digest() {
        return Err(ConfirmatoryResultError::ProtocolDigestMismatch);
    }
    if &result.run != run_binding.run() {
        return Err(ConfirmatoryResultError::RunBindingMismatch);
    }
    if result.completed_at_unix_ms < result.run.registered_at_unix_ms {
        return Err(ConfirmatoryResultError::ResultBeforeRunRegistration);
    }

    result
        .run
        .validate_against(frozen)
        .map_err(|error| ConfirmatoryResultError::InvalidV1(error.to_string()))?;
    for amendment in &result.amendments {
        amendment
            .validate_against(frozen)
            .map_err(|error| ConfirmatoryResultError::InvalidV1(error.to_string()))?;
    }
    for deviation in &result.deviations {
        deviation
            .validate()
            .map_err(|error| ConfirmatoryResultError::InvalidV1(error.to_string()))?;
    }

    unique_ids(result.artifacts.iter().map(|value| value.id.as_str()))?;
    unique_ids(result.metrics.iter().map(|value| value.metric_id.as_str()))?;
    unique_ids(result.claims.iter().map(|value| value.claim_id.as_str()))?;
    unique_ids(result.amendments.iter().map(|value| value.amendment_id.as_str()))?;
    unique_ids(result.deviations.iter().map(|value| value.deviation_id.as_str()))?;

    if !result
        .artifacts
        .iter()
        .any(|artifact| artifact.kind == ResultArtifactKind::Analysis)
    {
        return Err(ConfirmatoryResultError::InvalidV1(
            "result requires at least one Analysis artifact".into(),
        ));
    }

    let artifact_ids: HashSet<&str> = result.artifacts.iter().map(|value| value.id.as_str()).collect();
    for artifact in &result.artifacts {
        non_empty(&artifact.id, "result artifact id")?;
        non_empty(&artifact.digest, "result artifact digest")?;
        non_empty(&artifact.description, "result artifact description")?;
        if let Some(media_type) = &artifact.media_type {
            non_empty(media_type, "result artifact media type")?;
        }
    }

    let protocol_v1 = frozen.protocol();
    let metric_specs: HashMap<&str, _> = protocol_v1
        .metrics
        .iter()
        .map(|metric| (metric.id.as_str(), metric))
        .collect();
    let hypothesis_specs: HashMap<&str, _> = protocol_v1
        .hypotheses
        .iter()
        .map(|hypothesis| (hypothesis.id.as_str(), hypothesis))
        .collect();
    let metric_schemas: HashMap<&str, _> = protocol
        .metric_schemas()
        .iter()
        .map(|binding| (binding.metric_id.as_str(), &binding.value_schema))
        .collect();
    let reported_metric_ids: HashSet<&str> =
        result.metrics.iter().map(|metric| metric.metric_id.as_str()).collect();

    for metric in &result.metrics {
        non_empty(&metric.metric_id, "metric result id")?;
        let spec = metric_specs
            .get(metric.metric_id.as_str())
            .ok_or_else(|| ConfirmatoryResultError::UnknownMetric(metric.metric_id.clone()))?;
        let schema = *metric_schemas
            .get(metric.metric_id.as_str())
            .ok_or_else(|| ConfirmatoryResultError::MetricSchemaMismatch(format!(
                "metric {} has no V2 schema",
                metric.metric_id
            )))?;

        match (&metric.outcome, schema) {
            (MetricOutcome::Numeric { value, unit }, MetricValueSchema::Numeric { unit: expected }) => {
                if !value.is_finite() {
                    return Err(ConfirmatoryResultError::MetricSchemaMismatch(format!(
                        "metric {} numeric value is not finite",
                        metric.metric_id
                    )));
                }
                non_empty(unit, "metric result unit")?;
                if unit != expected || unit != &spec.unit {
                    return Err(ConfirmatoryResultError::MetricSchemaMismatch(format!(
                        "metric {} unit mismatch: V1={}, V2={}, result={}",
                        metric.metric_id, spec.unit, expected, unit
                    )));
                }
            }
            (MetricOutcome::Boolean(_), MetricValueSchema::Boolean) => {}
            (
                MetricOutcome::Categorical(value),
                MetricValueSchema::Categorical { allowed_values },
            ) => {
                non_empty(value, "categorical metric result")?;
                if let Some(values) = allowed_values {
                    if !values.contains(value) {
                        return Err(ConfirmatoryResultError::MetricSchemaMismatch(format!(
                            "metric {} categorical value {:?} is outside frozen vocabulary",
                            metric.metric_id, value
                        )));
                    }
                }
            }
            (MetricOutcome::Missing { reason }, _) | (MetricOutcome::NotComputed { reason }, _) => {
                if reason.trim().is_empty() {
                    return Err(ConfirmatoryResultError::MetricSchemaMismatch(format!(
                        "metric {} Missing/NotComputed state requires a reason",
                        metric.metric_id
                    )));
                }
            }
            (outcome, schema) => {
                return Err(ConfirmatoryResultError::MetricSchemaMismatch(format!(
                    "metric {} reports {} but frozen schema is {:?}",
                    metric.metric_id,
                    outcome_kind(outcome),
                    schema
                )));
            }
        }

        for artifact_id in &metric.artifact_ids {
            non_empty(artifact_id, "metric result artifact id")?;
            if !artifact_ids.contains(artifact_id.as_str()) {
                return Err(ConfirmatoryResultError::UnknownArtifact(artifact_id.clone()));
            }
        }
        if let Some(notes) = &metric.notes {
            non_empty(notes, "metric result notes")?;
        }
    }

    for spec in protocol_v1
        .metrics
        .iter()
        .filter(|metric| metric.role == MetricRole::Primary)
    {
        if !reported_metric_ids.contains(spec.id.as_str()) {
            return Err(ConfirmatoryResultError::MissingPrimaryMetric(spec.id.clone()));
        }
    }

    let derived_interpretation = classify_result(
        &result.amendments,
        &result.deviations,
        result.interpretation == ResultInterpretation::Invalidated,
    );
    if derived_interpretation != result.interpretation {
        return Err(ConfirmatoryResultError::InvalidV1(format!(
            "stored overall interpretation {:?} disagrees with supplied amendment/deviation semantics {:?}",
            result.interpretation, derived_interpretation
        )));
    }

    for claim in &result.claims {
        non_empty(&claim.claim_id, "result claim id")?;
        non_empty(&claim.statement, "result claim statement")?;
        if claim.metric_ids.is_empty() && claim.artifact_ids.is_empty() {
            return Err(ConfirmatoryResultError::ClaimEvidenceInadequate(format!(
                "claim {} has no metric/artifact evidence",
                claim.claim_id
            )));
        }
        for metric_id in &claim.metric_ids {
            non_empty(metric_id, "result claim metric id")?;
            if !metric_specs.contains_key(metric_id.as_str()) {
                return Err(ConfirmatoryResultError::UnknownMetric(metric_id.clone()));
            }
            if !reported_metric_ids.contains(metric_id.as_str()) {
                return Err(ConfirmatoryResultError::ClaimEvidenceInadequate(format!(
                    "claim {} references unreported metric {}",
                    claim.claim_id, metric_id
                )));
            }
        }
        for artifact_id in &claim.artifact_ids {
            non_empty(artifact_id, "result claim artifact id")?;
            if !artifact_ids.contains(artifact_id.as_str()) {
                return Err(ConfirmatoryResultError::UnknownArtifact(artifact_id.clone()));
            }
        }
        if let Some(hypothesis_id) = &claim.hypothesis_id {
            non_empty(hypothesis_id, "result claim hypothesis id")?;
            let hypothesis = hypothesis_specs
                .get(hypothesis_id.as_str())
                .ok_or_else(|| ConfirmatoryResultError::UnknownHypothesis(hypothesis_id.clone()))?;
            if hypothesis.role == HypothesisRole::Exploratory
                && claim.interpretation == ClaimInterpretation::Confirmatory
            {
                return Err(ConfirmatoryResultError::InvalidV1(format!(
                    "claim {} targets exploratory hypothesis {} but is Confirmatory",
                    claim.claim_id, hypothesis_id
                )));
            }
        }

        match result.interpretation {
            ResultInterpretation::Confirmatory => {}
            ResultInterpretation::ExploratoryDueToPostUnblindingAmendment
            | ResultInterpretation::ExploratoryDueToPrimaryDeviation => {
                if claim.interpretation == ClaimInterpretation::Confirmatory {
                    return Err(ConfirmatoryResultError::InvalidV1(format!(
                        "claim {} is Confirmatory while overall result is exploratory",
                        claim.claim_id
                    )));
                }
            }
            ResultInterpretation::Invalidated => {
                if claim.interpretation != ClaimInterpretation::Invalidated {
                    return Err(ConfirmatoryResultError::InvalidV1(format!(
                        "claim {} is not Invalidated while overall result is invalidated",
                        claim.claim_id
                    )));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_research_protocol::{
        AnalysisPlanRef, BaselineSpec, CanonicalConfirmatoryProtocol, ConfirmatoryEndpointSpec,
        HypothesisDirection, HypothesisSpec, MetricSchemaBinding, MetricSpec, MultiplicityPolicy,
        ProtocolAmendment, AmendmentTiming, ResearchProtocol, ResearchRunRegistration, StoppingRule,
    };

    use crate::result_v1::{ResultArtifactRef, ResultClaim};

    fn bool_protocol() -> CanonicalConfirmatoryProtocol {
        let frozen = ResearchProtocol::new(
            "bool-protocol",
            "1",
            "Did the protected event occur?",
            vec![
                HypothesisSpec::new(
                    "h-event",
                    "the protected event has the preregistered Boolean outcome",
                    HypothesisRole::Primary,
                    HypothesisDirection::Qualitative,
                )
                .unwrap(),
            ],
            vec![
                MetricSpec::new(
                    "event",
                    "protected event occurred",
                    "bool",
                    MetricRole::Primary,
                    "exact",
                )
                .unwrap(),
            ],
            vec![
                BaselineSpec::new("baseline", "frozen comparator", "bench/baseline-v1").unwrap(),
            ],
            vec![],
            StoppingRule::FixedEpisodeCount(1),
            MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
            AnalysisPlanRef::new("analysis", "1", "sha256:analysis-plan").unwrap(),
            "frozen scenario",
            "frozen seed",
        )
        .unwrap()
        .freeze(1_000)
        .unwrap();

        CanonicalConfirmatoryProtocol::new(
            frozen,
            vec![MetricSchemaBinding::new("event", MetricValueSchema::Boolean).unwrap()],
            vec![
                ConfirmatoryEndpointSpec::new(
                    "event-endpoint",
                    "h-event",
                    vec!["event".into()],
                    vec![],
                    ConfirmatoryDecisionRule::BooleanMustBe(false),
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn numeric_protocol() -> CanonicalConfirmatoryProtocol {
        let frozen = ResearchProtocol::new(
            "numeric-protocol",
            "1",
            "Did the numeric outcome satisfy the frozen analysis?",
            vec![
                HypothesisSpec::new(
                    "h-score",
                    "score satisfies frozen analysis",
                    HypothesisRole::Primary,
                    HypothesisDirection::GreaterThan,
                )
                .unwrap(),
            ],
            vec![MetricSpec::new("score", "score", "points", MetricRole::Primary, "mean").unwrap()],
            vec![
                BaselineSpec::new("baseline", "frozen comparator", "bench/baseline-v1").unwrap(),
            ],
            vec![],
            StoppingRule::FixedEpisodeCount(1),
            MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
            AnalysisPlanRef::new("analysis", "1", "sha256:analysis-plan").unwrap(),
            "frozen scenario",
            "frozen seed",
        )
        .unwrap()
        .freeze(1_000)
        .unwrap();

        CanonicalConfirmatoryProtocol::new(
            frozen,
            vec![MetricSchemaBinding::new("score", MetricValueSchema::numeric("points").unwrap()).unwrap()],
            vec![
                ConfirmatoryEndpointSpec::new(
                    "score-endpoint",
                    "h-score",
                    vec!["score".into()],
                    vec!["baseline".into()],
                    ConfirmatoryDecisionRule::external("analysis/score", "sha256:score-rule").unwrap(),
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn run_and_binding(
        protocol: &CanonicalConfirmatoryProtocol,
    ) -> (symthaea_research_protocol::ResearchRunRegistration, CanonicalConfirmatoryRunBinding) {
        let run = ResearchRunRegistration::new(
            protocol.frozen_protocol(),
            "run-001",
            1_100,
            "deadbeef",
            "sha256:dataset",
            "sha256:repro",
            "sha256:seeds",
        )
        .unwrap();
        let binding = protocol.bind_run(run.clone()).unwrap();
        (run, binding)
    }

    fn analysis_artifact() -> ResultArtifactRef {
        ResultArtifactRef::new(
            "analysis",
            ResultArtifactKind::Analysis,
            "sha256:analysis-output",
            "frozen analysis output",
        )
        .unwrap()
    }

    fn bool_result(
        protocol: &CanonicalConfirmatoryProtocol,
        outcome: MetricOutcome,
        disposition: ClaimDisposition,
        interpretation: ClaimInterpretation,
        amendment: bool,
    ) -> (ResearchResultManifest, CanonicalConfirmatoryRunBinding) {
        let (run, binding) = run_and_binding(protocol);
        let amendments = if amendment {
            vec![ProtocolAmendment::new(
                protocol.frozen_protocol(),
                "a1",
                1_500,
                AmendmentTiming::AfterOutcomeUnblinding,
                "post-unblinding change",
                vec!["changed analysis handling".into()],
            )
            .unwrap()]
        } else {
            vec![]
        };
        let claim = ResultClaim::new(
            "claim-event",
            "event endpoint conclusion",
            disposition,
            interpretation,
        )
        .unwrap()
        .for_hypothesis("h-event")
        .unwrap()
        .with_metric("event")
        .unwrap();
        let result = ResearchResultManifest::new(
            protocol.frozen_protocol(),
            run,
            "result-001",
            2_000,
            amendments,
            vec![],
            false,
            vec![analysis_artifact()],
            vec![MetricResult::new("event", outcome).unwrap()],
            vec![claim],
        )
        .unwrap();
        (result, binding)
    }

    #[test]
    fn direct_boolean_endpoint_derives_exact_confirmatory_disposition() {
        let protocol = bool_protocol();
        let (result, binding) = bool_result(
            &protocol,
            MetricOutcome::Boolean(false),
            ClaimDisposition::ConsistentWithHypothesis,
            ClaimInterpretation::Confirmatory,
            false,
        );
        ConfirmatoryResearchResultV2::new(
            &protocol,
            &binding,
            result,
            vec![ConfirmatoryClaimBinding::new("claim-event", "event-endpoint").unwrap()],
        )
        .unwrap();
    }

    #[test]
    fn missing_endpoint_evidence_cannot_support_positive_confirmatory_claim() {
        let protocol = bool_protocol();
        let (result, binding) = bool_result(
            &protocol,
            MetricOutcome::Missing { reason: "sensor unavailable".into() },
            ClaimDisposition::ConsistentWithHypothesis,
            ClaimInterpretation::Confirmatory,
            false,
        );
        let error = ConfirmatoryResearchResultV2::new(
            &protocol,
            &binding,
            result,
            vec![ConfirmatoryClaimBinding::new("claim-event", "event-endpoint").unwrap()],
        )
        .unwrap_err();
        assert!(matches!(error, ConfirmatoryResultError::ClaimEvidenceInadequate(_)));
    }

    #[test]
    fn missing_endpoint_evidence_may_be_confirmatory_inconclusive() {
        let protocol = bool_protocol();
        let (result, binding) = bool_result(
            &protocol,
            MetricOutcome::Missing { reason: "sensor unavailable".into() },
            ClaimDisposition::Inconclusive,
            ClaimInterpretation::Confirmatory,
            false,
        );
        ConfirmatoryResearchResultV2::new(
            &protocol,
            &binding,
            result,
            vec![ConfirmatoryClaimBinding::new("claim-event", "event-endpoint").unwrap()],
        )
        .unwrap();
    }

    #[test]
    fn downgraded_campaign_keeps_endpoint_census_without_confirmatory_support() {
        let protocol = bool_protocol();
        let (result, binding) = bool_result(
            &protocol,
            MetricOutcome::Boolean(false),
            ClaimDisposition::ConsistentWithHypothesis,
            ClaimInterpretation::Exploratory,
            true,
        );
        assert_eq!(
            result.interpretation,
            ResultInterpretation::ExploratoryDueToPostUnblindingAmendment
        );
        ConfirmatoryResearchResultV2::new(
            &protocol,
            &binding,
            result,
            vec![ConfirmatoryClaimBinding::new("claim-event", "event-endpoint").unwrap()],
        )
        .unwrap();
    }

    #[test]
    fn numeric_metric_reported_as_boolean_is_rejected_by_v2_schema() {
        let protocol = numeric_protocol();
        let (run, binding) = run_and_binding(&protocol);
        let claim = ResultClaim::new(
            "claim-score",
            "score supports hypothesis",
            ClaimDisposition::ConsistentWithHypothesis,
            ClaimInterpretation::Confirmatory,
        )
        .unwrap()
        .for_hypothesis("h-score")
        .unwrap()
        .with_metric("score")
        .unwrap()
        .with_artifact("analysis")
        .unwrap();
        let result = ResearchResultManifest::new(
            protocol.frozen_protocol(),
            run,
            "result-score",
            2_000,
            vec![],
            vec![],
            false,
            vec![analysis_artifact()],
            vec![MetricResult::new("score", MetricOutcome::Boolean(true)).unwrap()],
            vec![claim],
        )
        .unwrap();
        let error = ConfirmatoryResearchResultV2::new(
            &protocol,
            &binding,
            result,
            vec![ConfirmatoryClaimBinding::new("claim-score", "score-endpoint")
                .unwrap()
                .with_analysis_artifact("analysis")
                .unwrap()],
        )
        .unwrap_err();
        assert!(matches!(error, ConfirmatoryResultError::MetricSchemaMismatch(_)));
    }

    #[test]
    fn external_positive_claim_requires_bound_analysis_output() {
        let protocol = numeric_protocol();
        let (run, binding) = run_and_binding(&protocol);
        let claim = ResultClaim::new(
            "claim-score",
            "score supports hypothesis",
            ClaimDisposition::ConsistentWithHypothesis,
            ClaimInterpretation::Confirmatory,
        )
        .unwrap()
        .for_hypothesis("h-score")
        .unwrap()
        .with_metric("score")
        .unwrap();
        let result = ResearchResultManifest::new(
            protocol.frozen_protocol(),
            run,
            "result-score",
            2_000,
            vec![],
            vec![],
            false,
            vec![analysis_artifact()],
            vec![MetricResult::new(
                "score",
                MetricOutcome::Numeric { value: 1.0, unit: "points".into() },
            )
            .unwrap()],
            vec![claim],
        )
        .unwrap();
        let error = ConfirmatoryResearchResultV2::new(
            &protocol,
            &binding,
            result,
            vec![ConfirmatoryClaimBinding::new("claim-score", "score-endpoint").unwrap()],
        )
        .unwrap_err();
        assert!(matches!(error, ConfirmatoryResultError::ExternalRuleEvidenceMissing(_)));
    }

    #[test]
    fn binding_order_is_canonicalized() {
        // One endpoint is enough to establish validation; canonical ordering itself is enforced by
        // constructor sorting plus post-deserialization `windows` validation.
        let protocol = bool_protocol();
        let (result, binding) = bool_result(
            &protocol,
            MetricOutcome::Boolean(false),
            ClaimDisposition::ConsistentWithHypothesis,
            ClaimInterpretation::Confirmatory,
            false,
        );
        let value = ConfirmatoryResearchResultV2::new(
            &protocol,
            &binding,
            result,
            vec![ConfirmatoryClaimBinding::new("claim-event", "event-endpoint").unwrap()],
        )
        .unwrap();
        value.validate_against(&protocol, &binding).unwrap();
    }
}
