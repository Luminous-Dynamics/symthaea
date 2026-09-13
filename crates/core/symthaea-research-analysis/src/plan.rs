use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use symthaea_research_protocol::{
    CanonicalConfirmatoryProtocol, ConfirmatoryDecisionRule,
};

use crate::error::{non_empty, ResearchAnalysisError, Result};
use crate::types::{AnalysisInputRole, AnalysisVerificationClass, FrozenAnalysisInputSpecV1};

const PLAN_SCHEMA: &str = "symthaea-research-analysis/frozen-invocation-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenExternalAnalysisPlanV1 {
    pub endpoint_id: String,
    pub confirmatory_protocol_digest: String,
    pub rule_id: String,
    pub rule_artifact_digest: String,
    pub executable_artifact_digest: String,
    /// Digest of exact argv/config/entrypoint semantics, separate from executable bytes.
    pub invocation_digest: String,
    pub toolchain_digest: String,
    pub execution_environment_digest: String,
    /// Minimum verification class frozen before execution.
    pub minimum_verification_class: AnalysisVerificationClass,
    /// Exact ordered inputs. Cross-category order is identity-bearing.
    pub ordered_inputs: Vec<FrozenAnalysisInputSpecV1>,
    pub digest: String,
}

#[derive(Serialize)]
struct PlanDigestView<'a> {
    schema: &'static str,
    endpoint_id: &'a str,
    confirmatory_protocol_digest: &'a str,
    rule_id: &'a str,
    rule_artifact_digest: &'a str,
    executable_artifact_digest: &'a str,
    invocation_digest: &'a str,
    toolchain_digest: &'a str,
    execution_environment_digest: &'a str,
    minimum_verification_class: AnalysisVerificationClass,
    ordered_inputs: &'a [FrozenAnalysisInputSpecV1],
}

impl FrozenExternalAnalysisPlanV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        protocol: &CanonicalConfirmatoryProtocol,
        endpoint_id: impl Into<String>,
        executable_artifact_digest: impl Into<String>,
        invocation_digest: impl Into<String>,
        toolchain_digest: impl Into<String>,
        execution_environment_digest: impl Into<String>,
        minimum_verification_class: AnalysisVerificationClass,
        ordered_inputs: Vec<FrozenAnalysisInputSpecV1>,
    ) -> Result<Self> {
        protocol
            .validate()
            .map_err(|error| ResearchAnalysisError::Protocol(error.to_string()))?;
        let endpoint_id = endpoint_id.into();
        let endpoint = protocol
            .confirmatory_endpoints()
            .iter()
            .find(|endpoint| endpoint.endpoint_id == endpoint_id)
            .ok_or_else(|| ResearchAnalysisError::UnknownEndpoint(endpoint_id.clone()))?;
        let (rule_id, rule_artifact_digest) = match &endpoint.decision_rule {
            ConfirmatoryDecisionRule::ExternalFrozenAnalysisRule {
                rule_id,
                artifact_digest,
            } => (rule_id.clone(), artifact_digest.clone()),
            _ => return Err(ResearchAnalysisError::EndpointIsNotExternalRule(endpoint_id)),
        };

        let mut value = Self {
            endpoint_id,
            confirmatory_protocol_digest: protocol.digest().to_string(),
            rule_id,
            rule_artifact_digest,
            executable_artifact_digest: executable_artifact_digest.into(),
            invocation_digest: invocation_digest.into(),
            toolchain_digest: toolchain_digest.into(),
            execution_environment_digest: execution_environment_digest.into(),
            minimum_verification_class,
            ordered_inputs,
            digest: String::new(),
        };
        value.validate_semantics(protocol)?;
        value.digest = value.compute_digest()?;
        Ok(value)
    }

    pub fn verify_digest(&self) -> Result<()> {
        if self.compute_digest()? != self.digest {
            return Err(ResearchAnalysisError::PlanDigestMismatch);
        }
        Ok(())
    }

    pub fn validate_against(&self, protocol: &CanonicalConfirmatoryProtocol) -> Result<()> {
        self.verify_digest()?;
        self.validate_semantics(protocol)
    }

    fn compute_digest(&self) -> Result<String> {
        let view = PlanDigestView {
            schema: PLAN_SCHEMA,
            endpoint_id: &self.endpoint_id,
            confirmatory_protocol_digest: &self.confirmatory_protocol_digest,
            rule_id: &self.rule_id,
            rule_artifact_digest: &self.rule_artifact_digest,
            executable_artifact_digest: &self.executable_artifact_digest,
            invocation_digest: &self.invocation_digest,
            toolchain_digest: &self.toolchain_digest,
            execution_environment_digest: &self.execution_environment_digest,
            minimum_verification_class: self.minimum_verification_class,
            ordered_inputs: &self.ordered_inputs,
        };
        let bytes = serde_json::to_vec(&view)
            .map_err(|error| ResearchAnalysisError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    fn validate_semantics(&self, protocol: &CanonicalConfirmatoryProtocol) -> Result<()> {
        protocol
            .validate()
            .map_err(|error| ResearchAnalysisError::Protocol(error.to_string()))?;
        non_empty(&self.endpoint_id, "external analysis endpoint id")?;
        non_empty(&self.rule_id, "external analysis rule id")?;
        non_empty(
            &self.rule_artifact_digest,
            "external analysis rule artifact digest",
        )?;
        non_empty(
            &self.executable_artifact_digest,
            "analysis executable artifact digest",
        )?;
        non_empty(&self.invocation_digest, "analysis invocation digest")?;
        non_empty(&self.toolchain_digest, "analysis toolchain digest")?;
        non_empty(
            &self.execution_environment_digest,
            "analysis execution environment digest",
        )?;
        if self.ordered_inputs.is_empty() {
            return Err(ResearchAnalysisError::InputRoleMismatch(
                "external analysis plan requires at least one frozen input".into(),
            ));
        }
        if self.confirmatory_protocol_digest != protocol.digest() {
            return Err(ResearchAnalysisError::RuleBindingMismatch(
                "confirmatory protocol digest differs from frozen plan".into(),
            ));
        }

        let endpoint = protocol
            .confirmatory_endpoints()
            .iter()
            .find(|endpoint| endpoint.endpoint_id == self.endpoint_id)
            .ok_or_else(|| ResearchAnalysisError::UnknownEndpoint(self.endpoint_id.clone()))?;
        match &endpoint.decision_rule {
            ConfirmatoryDecisionRule::ExternalFrozenAnalysisRule {
                rule_id,
                artifact_digest,
            } => {
                if rule_id != &self.rule_id || artifact_digest != &self.rule_artifact_digest {
                    return Err(ResearchAnalysisError::RuleBindingMismatch(format!(
                        "endpoint {} frozen rule identity differs from invocation plan",
                        self.endpoint_id
                    )));
                }
            }
            _ => {
                return Err(ResearchAnalysisError::EndpointIsNotExternalRule(
                    self.endpoint_id.clone(),
                ));
            }
        }

        let mut seen = HashSet::new();
        let mut metric_ids = Vec::new();
        let mut baseline_ids = Vec::new();
        for input in &self.ordered_inputs {
            input.validate()?;
            let key = input.role.key();
            if !seen.insert(key.clone()) {
                return Err(ResearchAnalysisError::DuplicateInputRole(key));
            }
            match &input.role {
                AnalysisInputRole::EndpointMetric { metric_id } => metric_ids.push(metric_id.clone()),
                AnalysisInputRole::BaselineObservation { baseline_id } => {
                    baseline_ids.push(baseline_id.clone());
                }
                AnalysisInputRole::AuxiliaryFrozenInput { .. } => {}
            }
        }
        if metric_ids != endpoint.metric_ids {
            return Err(ResearchAnalysisError::InputRoleMismatch(format!(
                "endpoint {} metric input order/content differs: expected {:?}, got {:?}",
                self.endpoint_id, endpoint.metric_ids, metric_ids
            )));
        }
        if baseline_ids != endpoint.baseline_ids {
            return Err(ResearchAnalysisError::InputRoleMismatch(format!(
                "endpoint {} baseline input order/content differs: expected {:?}, got {:?}",
                self.endpoint_id, endpoint.baseline_ids, baseline_ids
            )));
        }
        Ok(())
    }
}
