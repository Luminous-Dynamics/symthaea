// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provider-neutral inference policy contracts.
//!
//! This module is intentionally runtime-neutral. It defines the semantic and
//! privacy boundary that a later inference router must satisfy before choosing
//! any local or remote execution backend.
//!
//! # Authority boundary
//!
//! Successful admission is **not** an execution capability and does not contact
//! a provider. A later tranche will mint short-lived, one-use inference permits
//! immediately before execution. External model output is likewise not action
//! authority; it remains untrusted data until separately admitted by the
//! relevant Symthaea capability/ethics boundary.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Semantic purpose of an inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferencePurpose {
    Translation,
    Summarization,
    Extraction,
    Classification,
    GeneralReasoning,
    MathematicalReasoning,
    ScientificReasoning,
    CodeGeneration,
    CodeRepair,
    CodeReview,
    CreativeWriting,
    Dialogue,
    ToolProposal,
    Embedding,
    Reranking,
}

/// Information-flow class carried by an inference payload.
///
/// The four `is_raw_non_exportable()` classes are hard remote boundaries.
/// Sending a minimized/approved derivative requires producing a new payload with
/// `RemoteSafeDerived`; merely changing policy flags cannot export the raw data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InformationClass {
    Public,
    UserProvided,
    Personal,
    PrivateMemory,
    EpisodicMemory,
    CognitiveState,
    IdentitySecret,
    AuthenticationSecret,
    RemoteSafeDerived,
}

impl InformationClass {
    /// Raw classes that may never cross the remote inference boundary.
    pub const fn is_raw_non_exportable(self) -> bool {
        matches!(
            self,
            Self::EpisodicMemory
                | Self::CognitiveState
                | Self::IdentitySecret
                | Self::AuthenticationSecret
        )
    }
}

/// Where inference is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionLocation {
    LocalProcess,
    LocalDevice,
    LocalNetwork,
    RemoteProvider,
    CommunityPeer,
}

impl ExecutionLocation {
    pub const fn is_remote(self) -> bool {
        matches!(self, Self::RemoteProvider | Self::CommunityPeer)
    }
}

/// Whether provider policy permits inference content to be used for training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderTrainingPolicy {
    Never,
    MayTrain,
    Unknown,
}

/// Provider-side retention semantics for inference content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderRetentionPolicy {
    ZeroRetention,
    MayRetain,
    Unknown,
}

/// Whether a provider may forward the request to another inference provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderRoutingPolicy {
    DirectOnly,
    MayRouteThirdParty,
    Unknown,
}

/// How strongly Symthaea can identify the model that executed a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ModelIdentity {
    /// Exact local content identity (for example a weights/tokenizer digest set).
    ContentVerified { digest: String },
    /// Remote provider asserts that this declared model served the request.
    ProviderAttested {
        provider: String,
        declared_model: String,
    },
    /// Endpoint does not provide a stable model identity claim.
    OpaqueEndpoint { endpoint_id: String },
}

/// Hard capability requirements for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRequirements {
    pub purpose: InferencePurpose,
    pub information_class: InformationClass,
    /// Estimated input tokens used only for admission/budget planning.
    pub estimated_input_tokens: u64,
    pub max_output_tokens: u64,
    pub require_streaming: bool,
    pub require_tools: bool,
    pub require_structured_output: bool,
}

impl InferenceRequirements {
    pub fn minimal(purpose: InferencePurpose, information_class: InformationClass) -> Self {
        Self {
            purpose,
            information_class,
            estimated_input_tokens: 0,
            max_output_tokens: 256,
            require_streaming: false,
            require_tools: false,
            require_structured_output: false,
        }
    }
}

/// Sensitive request payload plus its non-sensitive requirements.
///
/// Deliberately not `Serialize`: callers should not gain an accidental generic
/// persistence path for prompts/system context simply because metadata is serializable.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub requirements: InferenceRequirements,
}

impl InferenceRequest {
    pub fn new(prompt: impl Into<String>, requirements: InferenceRequirements) -> Self {
        Self {
            prompt: prompt.into(),
            system_prompt: None,
            requirements,
        }
    }
}

/// Provider/model capability snapshot used only for admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceCandidate {
    pub provider_id: String,
    pub model: ModelIdentity,
    pub location: ExecutionLocation,
    pub supported_purposes: Vec<InferencePurpose>,
    pub context_window_tokens: u64,
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_structured_output: bool,
    pub training_policy: ProviderTrainingPolicy,
    pub retention_policy: ProviderRetentionPolicy,
    pub routing_policy: ProviderRoutingPolicy,
    /// Maximum charge this candidate may impose for this request in micro-USD.
    /// `None` means the cost is not known strongly enough for fail-closed budgeting.
    pub max_charge_microusd: Option<u64>,
}

impl InferenceCandidate {
    pub fn local(provider_id: impl Into<String>, model: ModelIdentity) -> Self {
        Self {
            provider_id: provider_id.into(),
            model,
            location: ExecutionLocation::LocalDevice,
            supported_purposes: Vec::new(),
            context_window_tokens: 0,
            supports_streaming: false,
            supports_tools: false,
            supports_structured_output: false,
            training_policy: ProviderTrainingPolicy::Never,
            retention_policy: ProviderRetentionPolicy::ZeroRetention,
            routing_policy: ProviderRoutingPolicy::DirectOnly,
            max_charge_microusd: Some(0),
        }
    }

    fn supports_purpose(&self, purpose: InferencePurpose) -> bool {
        self.supported_purposes.contains(&purpose)
    }
}

/// User/deployment policy. Credentials never grant remote-execution authority by themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferencePolicy {
    pub allow_remote: bool,
    pub allow_provider_training: bool,
    pub allow_provider_retention: bool,
    pub allow_third_party_routing: bool,
    /// Maximum admitted charge per request, in micro-USD.
    pub max_charge_microusd: u64,
}

impl InferencePolicy {
    /// Local-first policy: no remote execution, no provider data use, zero spend.
    pub const fn sovereign_default() -> Self {
        Self {
            allow_remote: false,
            allow_provider_training: false,
            allow_provider_retention: false,
            allow_third_party_routing: false,
            max_charge_microusd: 0,
        }
    }

    /// Remote inference is allowed only when it is free and the candidate declares
    /// no training, zero retention, and no third-party routing.
    pub const fn free_private() -> Self {
        Self {
            allow_remote: true,
            allow_provider_training: false,
            allow_provider_retention: false,
            allow_third_party_routing: false,
            max_charge_microusd: 0,
        }
    }

    /// Fail-closed admission. This produces a route decision, not execution authority.
    pub fn admit(
        &self,
        request: &InferenceRequest,
        candidate: &InferenceCandidate,
    ) -> Result<AdmittedInferenceRoute, InferenceAdmissionError> {
        let req = &request.requirements;

        if !candidate.supports_purpose(req.purpose) {
            return Err(InferenceAdmissionError::PurposeUnsupported(req.purpose));
        }

        let required_context = req
            .estimated_input_tokens
            .saturating_add(req.max_output_tokens);
        if required_context > candidate.context_window_tokens {
            return Err(InferenceAdmissionError::ContextWindowTooSmall {
                required: required_context,
                available: candidate.context_window_tokens,
            });
        }

        if req.require_streaming && !candidate.supports_streaming {
            return Err(InferenceAdmissionError::StreamingUnsupported);
        }
        if req.require_tools && !candidate.supports_tools {
            return Err(InferenceAdmissionError::ToolsUnsupported);
        }
        if req.require_structured_output && !candidate.supports_structured_output {
            return Err(InferenceAdmissionError::StructuredOutputUnsupported);
        }

        if candidate.location.is_remote() {
            if req.information_class.is_raw_non_exportable() {
                return Err(InferenceAdmissionError::RawSensitiveDataCannotLeaveDevice(
                    req.information_class,
                ));
            }
            if !self.allow_remote {
                return Err(InferenceAdmissionError::RemoteExecutionForbidden);
            }
            if !self.allow_provider_training
                && candidate.training_policy != ProviderTrainingPolicy::Never
            {
                return Err(InferenceAdmissionError::ProviderTrainingPolicyRejected(
                    candidate.training_policy,
                ));
            }
            if !self.allow_provider_retention
                && candidate.retention_policy != ProviderRetentionPolicy::ZeroRetention
            {
                return Err(InferenceAdmissionError::ProviderRetentionPolicyRejected(
                    candidate.retention_policy,
                ));
            }
            if !self.allow_third_party_routing
                && candidate.routing_policy != ProviderRoutingPolicy::DirectOnly
            {
                return Err(InferenceAdmissionError::ProviderRoutingPolicyRejected(
                    candidate.routing_policy,
                ));
            }
        }

        let charge = candidate
            .max_charge_microusd
            .ok_or(InferenceAdmissionError::CostUnknown)?;
        if charge > self.max_charge_microusd {
            return Err(InferenceAdmissionError::CostExceedsBudget {
                candidate_microusd: charge,
                limit_microusd: self.max_charge_microusd,
            });
        }

        Ok(AdmittedInferenceRoute {
            provider_id: candidate.provider_id.clone(),
            model: candidate.model.clone(),
            location: candidate.location,
            information_class: req.information_class,
            purpose: req.purpose,
            admitted_max_charge_microusd: charge,
        })
    }
}

impl Default for InferencePolicy {
    fn default() -> Self {
        Self::sovereign_default()
    }
}

/// Policy-checked route metadata. This is explicitly not a permit/capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedInferenceRoute {
    provider_id: String,
    model: ModelIdentity,
    location: ExecutionLocation,
    information_class: InformationClass,
    purpose: InferencePurpose,
    admitted_max_charge_microusd: u64,
}

impl AdmittedInferenceRoute {
    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }
    pub fn model(&self) -> &ModelIdentity {
        &self.model
    }
    pub const fn location(&self) -> ExecutionLocation {
        self.location
    }
    pub const fn information_class(&self) -> InformationClass {
        self.information_class
    }
    pub const fn purpose(&self) -> InferencePurpose {
        self.purpose
    }
    pub const fn admitted_max_charge_microusd(&self) -> u64 {
        self.admitted_max_charge_microusd
    }
}

/// Explicit reasons a candidate cannot be admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceAdmissionError {
    PurposeUnsupported(InferencePurpose),
    ContextWindowTooSmall { required: u64, available: u64 },
    StreamingUnsupported,
    ToolsUnsupported,
    StructuredOutputUnsupported,
    RawSensitiveDataCannotLeaveDevice(InformationClass),
    RemoteExecutionForbidden,
    ProviderTrainingPolicyRejected(ProviderTrainingPolicy),
    ProviderRetentionPolicyRejected(ProviderRetentionPolicy),
    ProviderRoutingPolicyRejected(ProviderRoutingPolicy),
    CostUnknown,
    CostExceedsBudget {
        candidate_microusd: u64,
        limit_microusd: u64,
    },
}

impl fmt::Display for InferenceAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PurposeUnsupported(purpose) => write!(f, "inference purpose unsupported: {purpose:?}"),
            Self::ContextWindowTooSmall { required, available } => write!(
                f,
                "context window too small: required {required} tokens, available {available}"
            ),
            Self::StreamingUnsupported => write!(f, "streaming is required but unsupported"),
            Self::ToolsUnsupported => write!(f, "tool support is required but unsupported"),
            Self::StructuredOutputUnsupported => {
                write!(f, "structured output is required but unsupported")
            }
            Self::RawSensitiveDataCannotLeaveDevice(class) => {
                write!(f, "raw {class:?} data cannot cross a remote inference boundary")
            }
            Self::RemoteExecutionForbidden => write!(f, "remote inference is not authorized"),
            Self::ProviderTrainingPolicyRejected(policy) => {
                write!(f, "provider training policy rejected: {policy:?}")
            }
            Self::ProviderRetentionPolicyRejected(policy) => {
                write!(f, "provider retention policy rejected: {policy:?}")
            }
            Self::ProviderRoutingPolicyRejected(policy) => {
                write!(f, "provider routing policy rejected: {policy:?}")
            }
            Self::CostUnknown => write!(f, "candidate cost is unknown"),
            Self::CostExceedsBudget {
                candidate_microusd,
                limit_microusd,
            } => write!(
                f,
                "candidate cost {candidate_microusd} micro-USD exceeds limit {limit_microusd}"
            ),
        }
    }
}

impl std::error::Error for InferenceAdmissionError {}
