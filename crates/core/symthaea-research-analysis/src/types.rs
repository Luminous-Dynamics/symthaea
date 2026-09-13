use serde::{Deserialize, Serialize};

use crate::error::{non_empty, optional_non_empty, ResearchAnalysisError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalysisInputRole {
    EndpointMetric { metric_id: String },
    BaselineObservation { baseline_id: String },
    AuxiliaryFrozenInput { input_id: String },
}

impl AnalysisInputRole {
    pub fn endpoint_metric(metric_id: impl Into<String>) -> Self {
        Self::EndpointMetric {
            metric_id: metric_id.into(),
        }
    }

    pub fn baseline_observation(baseline_id: impl Into<String>) -> Self {
        Self::BaselineObservation {
            baseline_id: baseline_id.into(),
        }
    }

    pub fn auxiliary(input_id: impl Into<String>) -> Self {
        Self::AuxiliaryFrozenInput {
            input_id: input_id.into(),
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::EndpointMetric { metric_id } => non_empty(metric_id, "endpoint metric input id"),
            Self::BaselineObservation { baseline_id } => {
                non_empty(baseline_id, "baseline observation input id")
            }
            Self::AuxiliaryFrozenInput { input_id } => {
                non_empty(input_id, "auxiliary frozen input id")
            }
        }
    }

    pub(crate) fn key(&self) -> String {
        match self {
            Self::EndpointMetric { metric_id } => format!("metric:{metric_id}"),
            Self::BaselineObservation { baseline_id } => format!("baseline:{baseline_id}"),
            Self::AuxiliaryFrozenInput { input_id } => format!("aux:{input_id}"),
        }
    }

    pub(crate) fn requires_source_plan(&self) -> bool {
        matches!(
            self,
            Self::BaselineObservation { .. } | Self::AuxiliaryFrozenInput { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenAnalysisInputSpecV1 {
    pub role: AnalysisInputRole,
    pub source_plan_digest: Option<String>,
    /// Exact schema/serialization contract expected by the frozen invocation.
    pub schema_digest: String,
}

impl FrozenAnalysisInputSpecV1 {
    pub fn new(
        role: AnalysisInputRole,
        source_plan_digest: Option<String>,
        schema_digest: impl Into<String>,
    ) -> Result<Self> {
        let value = Self {
            role,
            source_plan_digest,
            schema_digest: schema_digest.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.role.validate()?;
        optional_non_empty(&self.source_plan_digest, "analysis input source-plan digest")?;
        non_empty(&self.schema_digest, "analysis input schema digest")?;
        if self.role.requires_source_plan() && self.source_plan_digest.is_none() {
            return Err(ResearchAnalysisError::MissingSourcePlan(self.role.key()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnalysisVerificationClass {
    BindingOnly,
    ExecutionWitnessBindingValidated,
    DeterministicReexecutionExact,
    IndependentReexecutionExact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisVerifierResult {
    Passed,
    Failed { reason: String },
    Inconclusive { reason: String },
}

impl AnalysisVerifierResult {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Passed => Ok(()),
            Self::Failed { reason } | Self::Inconclusive { reason } => {
                non_empty(reason, "analysis verifier result reason")
            }
        }
    }

    pub(crate) fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisVerifierV1 {
    pub verifier_id: String,
    pub verifier_identity_digest: String,
    pub independent_of_analysis_author: bool,
    pub verification_class: AnalysisVerificationClass,
    pub verification_evidence_digest: String,
    /// Required for exact re-execution classes. The receipt validator compares it to the exact
    /// bound output digest when verifier status is Passed.
    pub reexecuted_output_digest: Option<String>,
    pub result: AnalysisVerifierResult,
}

impl AnalysisVerifierV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        verifier_id: impl Into<String>,
        verifier_identity_digest: impl Into<String>,
        independent_of_analysis_author: bool,
        verification_class: AnalysisVerificationClass,
        verification_evidence_digest: impl Into<String>,
        reexecuted_output_digest: Option<String>,
        result: AnalysisVerifierResult,
    ) -> Result<Self> {
        let value = Self {
            verifier_id: verifier_id.into(),
            verifier_identity_digest: verifier_identity_digest.into(),
            independent_of_analysis_author,
            verification_class,
            verification_evidence_digest: verification_evidence_digest.into(),
            reexecuted_output_digest,
            result,
        };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        non_empty(&self.verifier_id, "analysis verifier id")?;
        non_empty(
            &self.verifier_identity_digest,
            "analysis verifier identity digest",
        )?;
        non_empty(
            &self.verification_evidence_digest,
            "analysis verification evidence digest",
        )?;
        optional_non_empty(
            &self.reexecuted_output_digest,
            "analysis re-executed output digest",
        )?;
        self.result.validate()?;

        let exact_reexecution = matches!(
            self.verification_class,
            AnalysisVerificationClass::DeterministicReexecutionExact
                | AnalysisVerificationClass::IndependentReexecutionExact
        );
        if exact_reexecution && self.result.is_passed() && self.reexecuted_output_digest.is_none() {
            return Err(ResearchAnalysisError::InvalidVerifier(
                "Passed exact re-execution requires reexecuted_output_digest".into(),
            ));
        }
        if self.verification_class == AnalysisVerificationClass::IndependentReexecutionExact
            && !self.independent_of_analysis_author
        {
            return Err(ResearchAnalysisError::InvalidVerifier(
                "IndependentReexecutionExact requires a verifier independent of the analysis author"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisExecutionStatus {
    Completed,
    InputMissing { reason: String },
    InputSchemaMismatch { reason: String },
    RuleArtifactMismatch { reason: String },
    ExecutionFailed { reason: String },
    VerifierFailed { reason: String },
    InfrastructureFailure { reason: String },
    NotRunWithReason { reason: String },
}

impl AnalysisExecutionStatus {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Completed => Ok(()),
            Self::InputMissing { reason }
            | Self::InputSchemaMismatch { reason }
            | Self::RuleArtifactMismatch { reason }
            | Self::ExecutionFailed { reason }
            | Self::VerifierFailed { reason }
            | Self::InfrastructureFailure { reason }
            | Self::NotRunWithReason { reason } => {
                non_empty(reason, "analysis execution terminal reason")
            }
        }
    }

    pub(crate) fn is_completed(&self) -> bool {
        matches!(self, Self::Completed)
    }
}
