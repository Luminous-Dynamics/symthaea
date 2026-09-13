use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResearchAnalysisError {
    Protocol(String),
    Result(String),
    EmptyField(&'static str),
    DuplicateInputRole(String),
    DuplicateId(String),
    UnknownEndpoint(String),
    EndpointIsNotExternalRule(String),
    RuleBindingMismatch(String),
    InputRoleMismatch(String),
    MissingSourcePlan(String),
    SchemaBindingMismatch(String),
    PlanDigestMismatch,
    RunBindingMismatch,
    ReceiptBindingMismatch(String),
    InvalidExecutionStatus(String),
    InvalidVerifier(String),
    ReexecutionOutputMismatch,
    ReceiptDigestMismatch,
    MissingPlan(String),
    MissingReceipt(String),
    OrphanPlan(String),
    OrphanReceipt(String),
    UnknownArtifact(String),
    ArtifactDigestMismatch(String),
    ArtifactKindMismatch(String),
    MetricInputArtifactMismatch(String),
    MissingBoundClaim(String),
    MissingAnalysisOutput(String),
    InsufficientExecutionVerification(String),
    NonCanonicalOrder(String),
    Serialization(String),
}

impl Display for ResearchAnalysisError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Protocol(message) => write!(f, "confirmatory protocol invalid: {message}"),
            Self::Result(message) => write!(f, "endpoint-complete result invalid: {message}"),
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::DuplicateInputRole(role) => write!(f, "duplicate analysis input role: {role}"),
            Self::DuplicateId(id) => write!(f, "duplicate analysis id: {id}"),
            Self::UnknownEndpoint(id) => write!(f, "unknown confirmatory endpoint: {id}"),
            Self::EndpointIsNotExternalRule(id) => {
                write!(f, "endpoint {id} does not use ExternalFrozenAnalysisRule")
            }
            Self::RuleBindingMismatch(message) => write!(f, "analysis rule binding mismatch: {message}"),
            Self::InputRoleMismatch(message) => write!(f, "analysis input role mismatch: {message}"),
            Self::MissingSourcePlan(role) => write!(f, "analysis input role {role} requires a frozen source-plan digest"),
            Self::SchemaBindingMismatch(message) => write!(f, "analysis schema binding mismatch: {message}"),
            Self::PlanDigestMismatch => write!(f, "frozen analysis invocation digest mismatch"),
            Self::RunBindingMismatch => write!(f, "analysis receipt run binding mismatch"),
            Self::ReceiptBindingMismatch(message) => write!(f, "analysis receipt binding mismatch: {message}"),
            Self::InvalidExecutionStatus(message) => write!(f, "invalid execution status: {message}"),
            Self::InvalidVerifier(message) => write!(f, "invalid analysis verifier: {message}"),
            Self::ReexecutionOutputMismatch => write!(f, "exact re-execution output digest differs from bound analysis output"),
            Self::ReceiptDigestMismatch => write!(f, "analysis execution receipt digest mismatch"),
            Self::MissingPlan(endpoint) => write!(f, "external endpoint {endpoint} has no frozen invocation plan"),
            Self::MissingReceipt(endpoint) => write!(f, "external endpoint {endpoint} has no execution receipt"),
            Self::OrphanPlan(endpoint) => write!(f, "analysis plan {endpoint} does not correspond to a frozen external endpoint"),
            Self::OrphanReceipt(endpoint) => write!(f, "analysis receipt {endpoint} does not correspond to a frozen external endpoint"),
            Self::UnknownArtifact(id) => write!(f, "analysis binding references unknown result artifact {id}"),
            Self::ArtifactDigestMismatch(id) => write!(f, "analysis artifact digest mismatch for {id}"),
            Self::ArtifactKindMismatch(id) => write!(f, "analysis artifact kind is incompatible for {id}"),
            Self::MetricInputArtifactMismatch(message) => write!(f, "metric input artifact mismatch: {message}"),
            Self::MissingBoundClaim(endpoint) => write!(f, "endpoint {endpoint} has no bound terminal claim"),
            Self::MissingAnalysisOutput(endpoint) => write!(f, "endpoint {endpoint} has no exact bound Analysis output"),
            Self::InsufficientExecutionVerification(endpoint) => write!(f, "endpoint {endpoint} has an evidentiary conclusion without sufficient execution verification"),
            Self::NonCanonicalOrder(kind) => write!(f, "{kind} are not in canonical endpoint-id order"),
            Self::Serialization(message) => write!(f, "research-analysis serialization failed: {message}"),
        }
    }
}

impl Error for ResearchAnalysisError {}

pub type Result<T> = std::result::Result<T, ResearchAnalysisError>;

pub(crate) fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ResearchAnalysisError::EmptyField(field));
    }
    Ok(())
}

pub(crate) fn optional_non_empty(value: &Option<String>, field: &'static str) -> Result<()> {
    if let Some(value) = value {
        non_empty(value, field)?;
    }
    Ok(())
}
