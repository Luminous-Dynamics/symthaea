use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use symthaea_research_protocol::{
    CanonicalConfirmatoryProtocol, CanonicalConfirmatoryRunBinding, ConfirmatoryDecisionRule,
};
use symthaea_research_result::{
    ClaimDisposition, CompleteEndpointResearchResultV2, MetricResult, ResultArtifactKind,
    ResultArtifactRef, ResultClaim,
};

use crate::error::{ResearchAnalysisError, Result};
use crate::plan::FrozenExternalAnalysisPlanV1;
use crate::receipt::FrozenAnalysisExecutionReceiptV1;
use crate::types::AnalysisInputRole;

/// Strong validation witness for an endpoint-complete result whose external-rule endpoints have
/// an exact frozen invocation/receipt census.
///
/// This wrapper mints no new scientific evidence digest. The result, plans, and receipts retain
/// their own identities; this type proves that their relationships recursively validate together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalAnalysisQualifiedResultV1 {
    inner: CompleteEndpointResearchResultV2,
    plans: Vec<FrozenExternalAnalysisPlanV1>,
    receipts: Vec<FrozenAnalysisExecutionReceiptV1>,
}

impl ExternalAnalysisQualifiedResultV1 {
    pub fn new(
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
        inner: CompleteEndpointResearchResultV2,
        mut plans: Vec<FrozenExternalAnalysisPlanV1>,
        mut receipts: Vec<FrozenAnalysisExecutionReceiptV1>,
    ) -> Result<Self> {
        plans.sort_by(|left, right| left.endpoint_id.cmp(&right.endpoint_id));
        receipts.sort_by(|left, right| left.endpoint_id.cmp(&right.endpoint_id));
        let value = Self {
            inner,
            plans,
            receipts,
        };
        value.validate_against(protocol, run_binding)?;
        Ok(value)
    }

    pub fn validate_against(
        &self,
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
    ) -> Result<()> {
        self.inner
            .validate_against(protocol, run_binding)
            .map_err(|error| ResearchAnalysisError::Result(error.to_string()))?;

        validate_canonical_endpoint_order(
            self.plans.iter().map(|plan| plan.endpoint_id.as_str()),
            "analysis plans",
        )?;
        validate_canonical_endpoint_order(
            self.receipts.iter().map(|receipt| receipt.endpoint_id.as_str()),
            "analysis receipts",
        )?;

        let mut external_endpoint_ids: Vec<&str> = protocol
            .confirmatory_endpoints()
            .iter()
            .filter(|endpoint| {
                matches!(
                    &endpoint.decision_rule,
                    ConfirmatoryDecisionRule::ExternalFrozenAnalysisRule { .. }
                )
            })
            .map(|endpoint| endpoint.endpoint_id.as_str())
            .collect();
        external_endpoint_ids.sort_unstable();

        let plan_ids: Vec<&str> = self.plans.iter().map(|plan| plan.endpoint_id.as_str()).collect();
        let receipt_ids: Vec<&str> = self
            .receipts
            .iter()
            .map(|receipt| receipt.endpoint_id.as_str())
            .collect();

        enforce_exact_census(&external_endpoint_ids, &plan_ids, true)?;
        enforce_exact_census(&external_endpoint_ids, &receipt_ids, false)?;

        let plans: HashMap<&str, &FrozenExternalAnalysisPlanV1> = self
            .plans
            .iter()
            .map(|plan| (plan.endpoint_id.as_str(), plan))
            .collect();
        let receipts: HashMap<&str, &FrozenAnalysisExecutionReceiptV1> = self
            .receipts
            .iter()
            .map(|receipt| (receipt.endpoint_id.as_str(), receipt))
            .collect();

        let result = self.inner.inner();
        let artifacts: HashMap<&str, &ResultArtifactRef> = result
            .result
            .artifacts
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact))
            .collect();
        let metrics: HashMap<&str, &MetricResult> = result
            .result
            .metrics
            .iter()
            .map(|metric| (metric.metric_id.as_str(), metric))
            .collect();
        let claims: HashMap<&str, &ResultClaim> = result
            .result
            .claims
            .iter()
            .map(|claim| (claim.claim_id.as_str(), claim))
            .collect();
        let claim_bindings: HashMap<&str, _> = result
            .claim_bindings
            .iter()
            .map(|binding| (binding.endpoint_id.as_str(), binding))
            .collect();

        for endpoint_id in external_endpoint_ids {
            let plan = plans[endpoint_id];
            let receipt = receipts[endpoint_id];
            plan.validate_against(protocol)?;
            receipt.validate_against(protocol, run_binding, plan)?;

            for input in &receipt.ordered_inputs {
                let artifact = artifacts
                    .get(input.artifact_id.as_str())
                    .ok_or_else(|| ResearchAnalysisError::UnknownArtifact(input.artifact_id.clone()))?;
                if artifact.digest != input.artifact_digest {
                    return Err(ResearchAnalysisError::ArtifactDigestMismatch(
                        input.artifact_id.clone(),
                    ));
                }

                match &input.role {
                    AnalysisInputRole::EndpointMetric { metric_id } => {
                        let metric = metrics.get(metric_id.as_str()).ok_or_else(|| {
                            ResearchAnalysisError::MetricInputArtifactMismatch(format!(
                                "endpoint {endpoint_id} has no result for metric {metric_id}"
                            ))
                        })?;
                        if !metric.artifact_ids.iter().any(|id| id == &input.artifact_id) {
                            return Err(ResearchAnalysisError::MetricInputArtifactMismatch(format!(
                                "metric {metric_id} does not bind input artifact {}",
                                input.artifact_id
                            )));
                        }
                    }
                    AnalysisInputRole::BaselineObservation { .. } => {
                        if artifact.kind != ResultArtifactKind::Metrics {
                            return Err(ResearchAnalysisError::ArtifactKindMismatch(
                                input.artifact_id.clone(),
                            ));
                        }
                    }
                    AnalysisInputRole::AuxiliaryFrozenInput { .. } => {}
                }
            }

            let binding = claim_bindings
                .get(endpoint_id)
                .ok_or_else(|| ResearchAnalysisError::MissingBoundClaim(endpoint_id.to_string()))?;
            let claim = claims
                .get(binding.claim_id.as_str())
                .ok_or_else(|| ResearchAnalysisError::MissingBoundClaim(endpoint_id.to_string()))?;

            match &receipt.output {
                Some(output) => {
                    let artifact = artifacts.get(output.artifact_id.as_str()).ok_or_else(|| {
                        ResearchAnalysisError::UnknownArtifact(output.artifact_id.clone())
                    })?;
                    if artifact.digest != output.artifact_digest {
                        return Err(ResearchAnalysisError::ArtifactDigestMismatch(
                            output.artifact_id.clone(),
                        ));
                    }
                    if artifact.kind != ResultArtifactKind::Analysis {
                        return Err(ResearchAnalysisError::ArtifactKindMismatch(
                            output.artifact_id.clone(),
                        ));
                    }
                    if binding.analysis_artifact_id.as_deref()
                        != Some(output.artifact_id.as_str())
                        || !claim.artifact_ids.iter().any(|id| id == &output.artifact_id)
                    {
                        return Err(ResearchAnalysisError::MissingAnalysisOutput(
                            endpoint_id.to_string(),
                        ));
                    }
                }
                None => {
                    if binding.analysis_artifact_id.is_some() {
                        return Err(ResearchAnalysisError::MissingAnalysisOutput(
                            endpoint_id.to_string(),
                        ));
                    }
                }
            }

            let evidentiary_conclusion = matches!(
                claim.disposition,
                ClaimDisposition::ConsistentWithHypothesis
                    | ClaimDisposition::InconsistentWithHypothesis
                    | ClaimDisposition::NullResult
            );
            if evidentiary_conclusion {
                if !receipt.supports_evidentiary_conclusion(plan) {
                    return Err(ResearchAnalysisError::InsufficientExecutionVerification(
                        endpoint_id.to_string(),
                    ));
                }
                if receipt.output.is_none() {
                    return Err(ResearchAnalysisError::MissingAnalysisOutput(
                        endpoint_id.to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn result_digest(&self) -> &str {
        self.inner.digest()
    }

    pub fn plans(&self) -> &[FrozenExternalAnalysisPlanV1] {
        &self.plans
    }

    pub fn receipts(&self) -> &[FrozenAnalysisExecutionReceiptV1] {
        &self.receipts
    }

    pub fn inner(&self) -> &CompleteEndpointResearchResultV2 {
        &self.inner
    }
}

fn validate_canonical_endpoint_order<'a>(
    ids: impl IntoIterator<Item = &'a str>,
    kind: &str,
) -> Result<()> {
    let mut previous: Option<&str> = None;
    for id in ids {
        if previous.is_some_and(|prior| prior >= id) {
            return Err(ResearchAnalysisError::NonCanonicalOrder(kind.to_string()));
        }
        previous = Some(id);
    }
    Ok(())
}

fn enforce_exact_census(expected: &[&str], actual: &[&str], plans: bool) -> Result<()> {
    let expected_set: HashSet<&str> = expected.iter().copied().collect();
    let actual_set: HashSet<&str> = actual.iter().copied().collect();

    for endpoint in expected_set.difference(&actual_set) {
        return if plans {
            Err(ResearchAnalysisError::MissingPlan((*endpoint).to_string()))
        } else {
            Err(ResearchAnalysisError::MissingReceipt((*endpoint).to_string()))
        };
    }
    for endpoint in actual_set.difference(&expected_set) {
        return if plans {
            Err(ResearchAnalysisError::OrphanPlan((*endpoint).to_string()))
        } else {
            Err(ResearchAnalysisError::OrphanReceipt((*endpoint).to_string()))
        };
    }
    Ok(())
}
