use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use symthaea_research_protocol::{
    CanonicalConfirmatoryProtocol, CanonicalConfirmatoryRunBinding,
};

use crate::error::{non_empty, optional_non_empty, ResearchAnalysisError, Result};
use crate::plan::FrozenExternalAnalysisPlanV1;
use crate::types::{
    AnalysisExecutionStatus, AnalysisInputRole, AnalysisVerificationClass, AnalysisVerifierResult,
    AnalysisVerifierV1, FrozenAnalysisInputSpecV1,
};

const RECEIPT_SCHEMA: &str = "symthaea-research-analysis/execution-receipt-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisInputBindingV1 {
    pub role: AnalysisInputRole,
    pub source_plan_digest: Option<String>,
    pub source_run_binding_digest: String,
    pub artifact_id: String,
    pub artifact_digest: String,
    pub schema_digest: String,
}

impl AnalysisInputBindingV1 {
    pub fn new(
        spec: &FrozenAnalysisInputSpecV1,
        run_binding: &CanonicalConfirmatoryRunBinding,
        artifact_id: impl Into<String>,
        artifact_digest: impl Into<String>,
    ) -> Result<Self> {
        let value = Self {
            role: spec.role.clone(),
            source_plan_digest: spec.source_plan_digest.clone(),
            source_run_binding_digest: run_binding.digest().to_string(),
            artifact_id: artifact_id.into(),
            artifact_digest: artifact_digest.into(),
            schema_digest: spec.schema_digest.clone(),
        };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.role.validate()?;
        optional_non_empty(&self.source_plan_digest, "analysis input source-plan digest")?;
        if self.role.requires_source_plan() && self.source_plan_digest.is_none() {
            return Err(ResearchAnalysisError::MissingSourcePlan(self.role.key()));
        }
        non_empty(
            &self.source_run_binding_digest,
            "analysis input run-binding digest",
        )?;
        non_empty(&self.artifact_id, "analysis input artifact id")?;
        non_empty(&self.artifact_digest, "analysis input artifact digest")?;
        non_empty(&self.schema_digest, "analysis input schema digest")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisOutputBindingV1 {
    pub artifact_id: String,
    pub artifact_digest: String,
}

impl AnalysisOutputBindingV1 {
    pub fn new(artifact_id: impl Into<String>, artifact_digest: impl Into<String>) -> Result<Self> {
        let value = Self {
            artifact_id: artifact_id.into(),
            artifact_digest: artifact_digest.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        non_empty(&self.artifact_id, "analysis output artifact id")?;
        non_empty(&self.artifact_digest, "analysis output artifact digest")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenAnalysisExecutionReceiptV1 {
    pub endpoint_id: String,
    pub confirmatory_protocol_digest: String,
    pub confirmatory_run_binding_digest: String,
    pub plan_digest: String,
    pub rule_id: String,
    pub rule_artifact_digest: String,
    pub executable_artifact_digest: String,
    pub invocation_digest: String,
    pub toolchain_digest: String,
    pub execution_environment_digest: String,
    pub ordered_inputs: Vec<AnalysisInputBindingV1>,
    pub output: Option<AnalysisOutputBindingV1>,
    pub execution_status: AnalysisExecutionStatus,
    pub execution_witness_digest: Option<String>,
    pub verifier: AnalysisVerifierV1,
    /// Optional references to external custody/access receipts (for example #193). These are
    /// identity hooks only; this crate does not validate custody policy or principal authenticity.
    pub custody_access_receipt_digests: Vec<String>,
    pub digest: String,
}

#[derive(Serialize)]
struct ReceiptDigestView<'a> {
    schema: &'static str,
    endpoint_id: &'a str,
    confirmatory_protocol_digest: &'a str,
    confirmatory_run_binding_digest: &'a str,
    plan_digest: &'a str,
    rule_id: &'a str,
    rule_artifact_digest: &'a str,
    executable_artifact_digest: &'a str,
    invocation_digest: &'a str,
    toolchain_digest: &'a str,
    execution_environment_digest: &'a str,
    ordered_inputs: &'a [AnalysisInputBindingV1],
    output: &'a Option<AnalysisOutputBindingV1>,
    execution_status: &'a AnalysisExecutionStatus,
    execution_witness_digest: &'a Option<String>,
    verifier: &'a AnalysisVerifierV1,
    custody_access_receipt_digests: &'a [String],
}

impl FrozenAnalysisExecutionReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
        plan: &FrozenExternalAnalysisPlanV1,
        ordered_inputs: Vec<AnalysisInputBindingV1>,
        output: Option<AnalysisOutputBindingV1>,
        execution_status: AnalysisExecutionStatus,
        execution_witness_digest: Option<String>,
        verifier: AnalysisVerifierV1,
        mut custody_access_receipt_digests: Vec<String>,
    ) -> Result<Self> {
        plan.validate_against(protocol)?;
        run_binding
            .validate_against(protocol)
            .map_err(|error| ResearchAnalysisError::Protocol(error.to_string()))?;

        let mut seen = HashSet::new();
        for digest in &custody_access_receipt_digests {
            non_empty(digest, "custody access receipt digest")?;
            if !seen.insert(digest.as_str()) {
                return Err(ResearchAnalysisError::DuplicateId(digest.clone()));
            }
        }
        custody_access_receipt_digests.sort();

        let mut value = Self {
            endpoint_id: plan.endpoint_id.clone(),
            confirmatory_protocol_digest: protocol.digest().to_string(),
            confirmatory_run_binding_digest: run_binding.digest().to_string(),
            plan_digest: plan.digest.clone(),
            rule_id: plan.rule_id.clone(),
            rule_artifact_digest: plan.rule_artifact_digest.clone(),
            executable_artifact_digest: plan.executable_artifact_digest.clone(),
            invocation_digest: plan.invocation_digest.clone(),
            toolchain_digest: plan.toolchain_digest.clone(),
            execution_environment_digest: plan.execution_environment_digest.clone(),
            ordered_inputs,
            output,
            execution_status,
            execution_witness_digest,
            verifier,
            custody_access_receipt_digests,
            digest: String::new(),
        };
        value.validate_semantics(protocol, run_binding, plan)?;
        value.digest = value.compute_digest()?;
        Ok(value)
    }

    pub fn verify_digest(&self) -> Result<()> {
        if self.compute_digest()? != self.digest {
            return Err(ResearchAnalysisError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
        plan: &FrozenExternalAnalysisPlanV1,
    ) -> Result<()> {
        self.verify_digest()?;
        self.validate_semantics(protocol, run_binding, plan)
    }

    pub fn supports_evidentiary_conclusion(&self, plan: &FrozenExternalAnalysisPlanV1) -> bool {
        self.execution_status.is_completed()
            && self.output.is_some()
            && self.execution_witness_digest.is_some()
            && self.verifier.result.is_passed()
            && self.verifier.verification_class != AnalysisVerificationClass::BindingOnly
            && self.verifier.verification_class >= plan.minimum_verification_class
    }

    fn compute_digest(&self) -> Result<String> {
        let view = ReceiptDigestView {
            schema: RECEIPT_SCHEMA,
            endpoint_id: &self.endpoint_id,
            confirmatory_protocol_digest: &self.confirmatory_protocol_digest,
            confirmatory_run_binding_digest: &self.confirmatory_run_binding_digest,
            plan_digest: &self.plan_digest,
            rule_id: &self.rule_id,
            rule_artifact_digest: &self.rule_artifact_digest,
            executable_artifact_digest: &self.executable_artifact_digest,
            invocation_digest: &self.invocation_digest,
            toolchain_digest: &self.toolchain_digest,
            execution_environment_digest: &self.execution_environment_digest,
            ordered_inputs: &self.ordered_inputs,
            output: &self.output,
            execution_status: &self.execution_status,
            execution_witness_digest: &self.execution_witness_digest,
            verifier: &self.verifier,
            custody_access_receipt_digests: &self.custody_access_receipt_digests,
        };
        let bytes = serde_json::to_vec(&view)
            .map_err(|error| ResearchAnalysisError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    fn validate_semantics(
        &self,
        protocol: &CanonicalConfirmatoryProtocol,
        run_binding: &CanonicalConfirmatoryRunBinding,
        plan: &FrozenExternalAnalysisPlanV1,
    ) -> Result<()> {
        plan.validate_against(protocol)?;
        run_binding
            .validate_against(protocol)
            .map_err(|error| ResearchAnalysisError::Protocol(error.to_string()))?;

        if self.endpoint_id != plan.endpoint_id
            || self.confirmatory_protocol_digest != protocol.digest()
            || self.confirmatory_run_binding_digest != run_binding.digest()
            || self.plan_digest != plan.digest
            || self.rule_id != plan.rule_id
            || self.rule_artifact_digest != plan.rule_artifact_digest
            || self.executable_artifact_digest != plan.executable_artifact_digest
            || self.invocation_digest != plan.invocation_digest
            || self.toolchain_digest != plan.toolchain_digest
            || self.execution_environment_digest != plan.execution_environment_digest
        {
            return Err(ResearchAnalysisError::ReceiptBindingMismatch(format!(
                "receipt identity does not exactly match frozen plan/run for endpoint {}",
                plan.endpoint_id
            )));
        }

        if self.ordered_inputs.len() != plan.ordered_inputs.len() {
            return Err(ResearchAnalysisError::ReceiptBindingMismatch(format!(
                "endpoint {} input count differs from frozen plan",
                plan.endpoint_id
            )));
        }
        for (index, (binding, spec)) in self
            .ordered_inputs
            .iter()
            .zip(plan.ordered_inputs.iter())
            .enumerate()
        {
            binding.validate()?;
            if binding.role != spec.role || binding.source_plan_digest != spec.source_plan_digest {
                return Err(ResearchAnalysisError::ReceiptBindingMismatch(format!(
                    "endpoint {} input {index} role/source plan differs from frozen invocation",
                    plan.endpoint_id
                )));
            }
            if binding.schema_digest != spec.schema_digest {
                return Err(ResearchAnalysisError::SchemaBindingMismatch(format!(
                    "endpoint {} input {index} schema digest differs from frozen invocation",
                    plan.endpoint_id
                )));
            }
            if binding.source_run_binding_digest != run_binding.digest() {
                return Err(ResearchAnalysisError::RunBindingMismatch);
            }
        }

        self.execution_status.validate()?;
        optional_non_empty(
            &self.execution_witness_digest,
            "analysis execution witness digest",
        )?;
        self.verifier.validate()?;
        if let Some(output) = &self.output {
            output.validate()?;
        }

        let mut previous: Option<&str> = None;
        for digest in &self.custody_access_receipt_digests {
            non_empty(digest, "custody access receipt digest")?;
            if previous.is_some_and(|prior| prior >= digest.as_str()) {
                return Err(ResearchAnalysisError::NonCanonicalOrder(
                    "custody access receipt digests".into(),
                ));
            }
            previous = Some(digest);
        }

        if self.execution_status.is_completed() {
            if self.output.is_none() {
                return Err(ResearchAnalysisError::InvalidExecutionStatus(
                    "Completed requires an exact output binding".into(),
                ));
            }
            if self.execution_witness_digest.is_none() {
                return Err(ResearchAnalysisError::InvalidExecutionStatus(
                    "Completed requires an execution witness digest".into(),
                ));
            }
        } else if self.verifier.result.is_passed() {
            return Err(ResearchAnalysisError::InvalidVerifier(
                "non-Completed execution cannot have a Passed verifier result".into(),
            ));
        }

        match (&self.execution_status, &self.verifier.result) {
            (AnalysisExecutionStatus::VerifierFailed { .. }, AnalysisVerifierResult::Failed { .. }) => {}
            (AnalysisExecutionStatus::VerifierFailed { .. }, _) => {
                return Err(ResearchAnalysisError::InvalidVerifier(
                    "VerifierFailed execution status requires Failed verifier result".into(),
                ));
            }
            (_, AnalysisVerifierResult::Failed { .. }) => {
                return Err(ResearchAnalysisError::InvalidExecutionStatus(
                    "Failed verifier result requires VerifierFailed execution status".into(),
                ));
            }
            _ => {}
        }

        if self.verifier.result.is_passed()
            && self.verifier.verification_class < plan.minimum_verification_class
        {
            return Err(ResearchAnalysisError::InvalidVerifier(format!(
                "verifier class {:?} is weaker than frozen minimum {:?}",
                self.verifier.verification_class, plan.minimum_verification_class
            )));
        }

        let exact_reexecution = matches!(
            self.verifier.verification_class,
            AnalysisVerificationClass::DeterministicReexecutionExact
                | AnalysisVerificationClass::IndependentReexecutionExact
        );
        if exact_reexecution && self.verifier.result.is_passed() {
            let output = self.output.as_ref().ok_or_else(|| {
                ResearchAnalysisError::InvalidVerifier(
                    "Passed exact re-execution requires an exact output binding".into(),
                )
            })?;
            if self.verifier.reexecuted_output_digest.as_deref()
                != Some(output.artifact_digest.as_str())
            {
                return Err(ResearchAnalysisError::ReexecutionOutputMismatch);
            }
        }

        Ok(())
    }
}
