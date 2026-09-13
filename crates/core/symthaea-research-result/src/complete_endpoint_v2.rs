//! Stronger endpoint-completeness boundary for V2 research results.
//!
//! `ConfirmatoryResearchResultV2` validates the historical V1 result against the canonical V2
//! protocol/run identity and preserves legitimate campaign downgrade/invalidation semantics.
//! This wrapper adds two cross-interpretation invariants that must remain true even after a
//! campaign is downgraded:
//!
//! 1. every frozen V2 endpoint receives exactly one terminal claim binding;
//! 2. Missing/NotComputed required observations cannot support positive, negative, or null
//!    conclusions, regardless of whether the terminal claim is Confirmatory or Exploratory.
//!
//! This is a validation boundary, not a new evidence identity and not a software/scientific
//! qualification claim. Its `digest()` is the exact inner V2 result digest.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use symthaea_research_protocol::{
    CanonicalConfirmatoryProtocol, CanonicalConfirmatoryRunBinding,
};

use crate::confirmatory_v2::{ConfirmatoryResearchResultV2, ConfirmatoryResultError};
use crate::result_v1::{ClaimDisposition, MetricOutcome, MetricResult, ResultClaim};

pub type Result<T> = std::result::Result<T, CompleteEndpointResultError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompleteEndpointResultError {
    Inner(String),
    MissingEndpointBinding(String),
    UnknownBoundClaim(String),
    MissingRequiredMetric {
        endpoint_id: String,
        metric_id: String,
    },
    MissingEvidenceSupportsConclusion {
        endpoint_id: String,
        claim_id: String,
        metric_id: String,
    },
}

impl Display for CompleteEndpointResultError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inner(message) => write!(f, "inner V2 result invalid: {message}"),
            Self::MissingEndpointBinding(endpoint_id) => {
                write!(f, "frozen endpoint {endpoint_id} has no terminal claim binding")
            }
            Self::UnknownBoundClaim(claim_id) => {
                write!(f, "endpoint binding references unknown claim {claim_id}")
            }
            Self::MissingRequiredMetric {
                endpoint_id,
                metric_id,
            } => write!(
                f,
                "endpoint {endpoint_id} requires metric {metric_id}, but no result entry exists"
            ),
            Self::MissingEvidenceSupportsConclusion {
                endpoint_id,
                claim_id,
                metric_id,
            } => write!(
                f,
                "claim {claim_id} for endpoint {endpoint_id} treats Missing/NotComputed metric {metric_id} as positive, negative, or null evidence"
            ),
        }
    }
}

impl Error for CompleteEndpointResultError {}

impl From<ConfirmatoryResultError> for CompleteEndpointResultError {
    fn from(value: ConfirmatoryResultError) -> Self {
        Self::Inner(value.to_string())
    }
}

/// Validation witness for a V2 result with complete frozen-endpoint census and evidence adequacy.
///
/// The inner result remains the content-addressed evidence object. This wrapper deliberately does
/// not mint a second digest merely for satisfying stronger validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompleteEndpointResearchResultV2 {
    inner: ConfirmatoryResearchResultV2,
}

impl CompleteEndpointResearchResultV2 {
    pub fn new(
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
        inner: ConfirmatoryResearchResultV2,
    ) -> Result<Self> {
        let value = Self { inner };
        value.validate_against(protocol, run_binding)?;
        Ok(value)
    }

    /// Required after deserialization as well as construction.
    pub fn validate_against(
        &self,
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
    ) -> Result<()> {
        self.inner.validate_against(protocol, run_binding)?;
        self.validate_complete_endpoint_census(protocol)
    }

    pub fn digest(&self) -> &str {
        &self.inner.digest
    }

    pub fn inner(&self) -> &ConfirmatoryResearchResultV2 {
        &self.inner
    }

    pub fn into_inner(self) -> ConfirmatoryResearchResultV2 {
        self.inner
    }

    fn validate_complete_endpoint_census(
        &self,
        protocol: &CanonicalConfirmatoryProtocol,
    ) -> Result<()> {
        let claims: HashMap<&str, &ResultClaim> = self
            .inner
            .result
            .claims
            .iter()
            .map(|claim| (claim.claim_id.as_str(), claim))
            .collect();
        let metrics: HashMap<&str, &MetricResult> = self
            .inner
            .result
            .metrics
            .iter()
            .map(|metric| (metric.metric_id.as_str(), metric))
            .collect();
        let bound_endpoints: HashSet<&str> = self
            .inner
            .claim_bindings
            .iter()
            .map(|binding| binding.endpoint_id.as_str())
            .collect();
        let binding_by_endpoint: HashMap<&str, _> = self
            .inner
            .claim_bindings
            .iter()
            .map(|binding| (binding.endpoint_id.as_str(), binding))
            .collect();

        for endpoint in protocol.confirmatory_endpoints() {
            if !bound_endpoints.contains(endpoint.endpoint_id.as_str()) {
                return Err(CompleteEndpointResultError::MissingEndpointBinding(
                    endpoint.endpoint_id.clone(),
                ));
            }

            let binding = binding_by_endpoint[endpoint.endpoint_id.as_str()];
            let claim = claims.get(binding.claim_id.as_str()).ok_or_else(|| {
                CompleteEndpointResultError::UnknownBoundClaim(binding.claim_id.clone())
            })?;

            for metric_id in &endpoint.metric_ids {
                let metric = metrics.get(metric_id.as_str()).ok_or_else(|| {
                    CompleteEndpointResultError::MissingRequiredMetric {
                        endpoint_id: endpoint.endpoint_id.clone(),
                        metric_id: metric_id.clone(),
                    }
                })?;

                let absent = matches!(
                    metric.outcome,
                    MetricOutcome::Missing { .. } | MetricOutcome::NotComputed { .. }
                );
                let evidentiary_conclusion = matches!(
                    claim.disposition,
                    ClaimDisposition::ConsistentWithHypothesis
                        | ClaimDisposition::InconsistentWithHypothesis
                        | ClaimDisposition::NullResult
                );
                if absent && evidentiary_conclusion {
                    return Err(
                        CompleteEndpointResultError::MissingEvidenceSupportsConclusion {
                            endpoint_id: endpoint.endpoint_id.clone(),
                            claim_id: claim.claim_id.clone(),
                            metric_id: metric_id.clone(),
                        },
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::confirmatory_v2::ConfirmatoryClaimBinding;
    use crate::result_v1::{
        ClaimInterpretation, MetricResult, ResearchResultManifest, ResultArtifactKind,
        ResultArtifactRef, ResultClaim,
    };
    use symthaea_research_protocol::{
        AnalysisPlanRef, BaselineSpec, CanonicalConfirmatoryProtocol,
        ConfirmatoryDecisionRule, ConfirmatoryEndpointSpec, HypothesisDirection, HypothesisRole,
        HypothesisSpec, MetricRole, MetricSchemaBinding, MetricSpec, MetricValueSchema,
        MultiplicityPolicy, ResearchProtocol, ResearchRunRegistration, StoppingRule,
    };

    fn protocol() -> CanonicalConfirmatoryProtocol {
        let frozen = ResearchProtocol::new(
            "complete-endpoints",
            "1",
            "Are both declared endpoint observations retained?",
            vec![
                HypothesisSpec::new(
                    "h-primary",
                    "primary endpoint",
                    HypothesisRole::Primary,
                    HypothesisDirection::Qualitative,
                )
                .unwrap(),
                HypothesisSpec::new(
                    "h-safety",
                    "safety endpoint",
                    HypothesisRole::Safety,
                    HypothesisDirection::Qualitative,
                )
                .unwrap(),
            ],
            vec![
                MetricSpec::new("primary", "primary", "bool", MetricRole::Primary, "exact")
                    .unwrap(),
                MetricSpec::new("safety", "safety", "bool", MetricRole::Safety, "exact")
                    .unwrap(),
            ],
            vec![
                BaselineSpec::new("baseline", "frozen comparator", "bench/baseline-v1").unwrap(),
            ],
            vec![],
            StoppingRule::FixedEpisodeCount(1),
            MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
            AnalysisPlanRef::new("analysis", "1", "sha256:analysis").unwrap(),
            "frozen scenario",
            "frozen seeds",
        )
        .unwrap()
        .freeze(1_000)
        .unwrap();

        CanonicalConfirmatoryProtocol::new(
            frozen,
            vec![
                MetricSchemaBinding::new("primary", MetricValueSchema::Boolean).unwrap(),
                MetricSchemaBinding::new("safety", MetricValueSchema::Boolean).unwrap(),
            ],
            vec![
                ConfirmatoryEndpointSpec::new(
                    "endpoint-primary",
                    "h-primary",
                    vec!["primary".into()],
                    vec![],
                    ConfirmatoryDecisionRule::BooleanMustBe(true),
                )
                .unwrap(),
                ConfirmatoryEndpointSpec::new(
                    "endpoint-safety",
                    "h-safety",
                    vec!["safety".into()],
                    vec![],
                    ConfirmatoryDecisionRule::BooleanMustBe(false),
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn run(
        protocol: &CanonicalConfirmatoryProtocol,
    ) -> (
        ResearchRunRegistration,
        CanonicalConfirmatoryRunBinding,
    ) {
        let run = ResearchRunRegistration::new(
            protocol.frozen_protocol(),
            "run-1",
            1_100,
            "deadbeef",
            "sha256:data",
            "sha256:repro",
            "sha256:seeds",
        )
        .unwrap();
        let binding = protocol.bind_run(run.clone()).unwrap();
        (run, binding)
    }

    fn analysis() -> ResultArtifactRef {
        ResultArtifactRef::new(
            "analysis",
            ResultArtifactKind::Analysis,
            "sha256:analysis-output",
            "analysis output",
        )
        .unwrap()
    }

    #[test]
    fn secondary_or_safety_endpoint_cannot_disappear() {
        let protocol = protocol();
        let (run, binding) = run(&protocol);
        let result = ResearchResultManifest::new(
            protocol.frozen_protocol(),
            run,
            "result",
            2_000,
            vec![],
            vec![],
            false,
            vec![analysis()],
            vec![
                MetricResult::new("primary", MetricOutcome::Boolean(true)).unwrap(),
                MetricResult::new("safety", MetricOutcome::Boolean(false)).unwrap(),
            ],
            vec![
                ResultClaim::new(
                    "claim-primary",
                    "primary endpoint",
                    ClaimDisposition::ConsistentWithHypothesis,
                    ClaimInterpretation::Confirmatory,
                )
                .unwrap()
                .for_hypothesis("h-primary")
                .unwrap()
                .with_metric("primary")
                .unwrap(),
            ],
        )
        .unwrap();
        let v2 = ConfirmatoryResearchResultV2::new(
            &protocol,
            &binding,
            result,
            vec![
                ConfirmatoryClaimBinding::new("claim-primary", "endpoint-primary").unwrap(),
            ],
        )
        .unwrap();

        assert_eq!(
            CompleteEndpointResearchResultV2::new(&protocol, &binding, v2).unwrap_err(),
            CompleteEndpointResultError::MissingEndpointBinding("endpoint-safety".into())
        );
    }
}
