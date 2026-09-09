// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IF-10 freshness-bound provider/model registry.
//!
//! Provider configuration is not timeless truth. Executable candidates are
//! materialized only from source-attributed claims that are fresh, field-appropriate,
//! and locally versioned against rollback. Public pricing documentation alone cannot
//! mint account/request cost authority.

#[cfg(not(test))]
use super::inference_contract::{
    ExecutionLocation, InferenceCandidate, InferencePurpose, ModelIdentity,
    ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};
#[cfg(test)]
use crate::inference_contract::{
    ExecutionLocation, InferenceCandidate, InferencePurpose, ModelIdentity,
    ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};

use std::collections::BTreeMap;
use std::fmt;

const PROFILE_DOMAIN: &[u8] = b"symthaea.inference.provider-profile.v1";
const MAX_ID_BYTES: usize = 512;
const MAX_SOURCE_LOCATOR_BYTES: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderProtocol {
    OpenAiCompatibleV1,
}

/// Evidence source class. Some source kinds are representable for future
/// verification/observability but deliberately not executable authority in v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderClaimSourceKind {
    FirstPartyPolicyDocument,
    FirstPartyModelCatalog,
    FirstPartyPricingDocument,
    FirstPartyAccountState,
    SignedDeploymentManifest,
    ThirdPartyCurator,
}

/// Semantic role a claim serves. Source authority is role-specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderClaimRole {
    Endpoint,
    Lifecycle,
    Capabilities,
    DataPolicy,
    RequestCost,
}

/// Locally observed claim provenance. Source locators are deliberately omitted
/// from Debug because account-scoped API URLs can be correlating metadata.
#[derive(Clone, PartialEq, Eq)]
pub struct ProviderClaimEvidence {
    source_kind: ProviderClaimSourceKind,
    source_locator: String,
    snapshot_digest: [u8; 32],
    observed_at_tick: u64,
    valid_until_tick: u64,
}

impl fmt::Debug for ProviderClaimEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderClaimEvidence")
            .field("source_kind", &self.source_kind)
            .field("snapshot_digest", &self.snapshot_digest)
            .field("observed_at_tick", &self.observed_at_tick)
            .field("valid_until_tick", &self.valid_until_tick)
            .finish_non_exhaustive()
    }
}

impl ProviderClaimEvidence {
    pub fn new(
        source_kind: ProviderClaimSourceKind,
        source_locator: impl Into<String>,
        snapshot_digest: [u8; 32],
        observed_at_tick: u64,
        valid_until_tick: u64,
    ) -> Result<Self, ProviderRegistryError> {
        let source_locator = source_locator.into();
        if !valid_text(&source_locator, MAX_SOURCE_LOCATOR_BYTES) {
            return Err(ProviderRegistryError::InvalidSourceLocator);
        }
        if snapshot_digest == [0; 32] {
            return Err(ProviderRegistryError::ZeroSnapshotDigest);
        }
        if valid_until_tick <= observed_at_tick {
            return Err(ProviderRegistryError::InvalidClaimValidityWindow);
        }
        Ok(Self {
            source_kind,
            source_locator,
            snapshot_digest,
            observed_at_tick,
            valid_until_tick,
        })
    }

    pub const fn source_kind(&self) -> ProviderClaimSourceKind {
        self.source_kind
    }

    pub fn source_locator(&self) -> &str {
        &self.source_locator
    }

    pub const fn snapshot_digest(&self) -> &[u8; 32] {
        &self.snapshot_digest
    }

    pub const fn observed_at_tick(&self) -> u64 {
        self.observed_at_tick
    }

    pub const fn valid_until_tick(&self) -> u64 {
        self.valid_until_tick
    }

    fn validate_for(
        &self,
        role: ProviderClaimRole,
        now_tick: u64,
    ) -> Result<(), ProviderQualificationError> {
        if now_tick < self.observed_at_tick {
            return Err(ProviderQualificationError::ClaimNotYetValid(role));
        }
        if now_tick >= self.valid_until_tick {
            return Err(ProviderQualificationError::ClaimExpired(role));
        }
        if !source_allowed(role, self.source_kind) {
            return Err(ProviderQualificationError::SourceNotAuthoritative {
                role,
                source: self.source_kind,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedProviderClaim<T> {
    value: T,
    evidence: ProviderClaimEvidence,
}

impl<T> SourcedProviderClaim<T> {
    pub fn new(value: T, evidence: ProviderClaimEvidence) -> Self {
        Self { value, evidence }
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn evidence(&self) -> &ProviderClaimEvidence {
        &self.evidence
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderModelLifecycle {
    Production,
    Preview,
    Deprecated,
    Unavailable,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ProviderEndpointDescriptor {
    protocol: ProviderProtocol,
    base_url: String,
}

impl fmt::Debug for ProviderEndpointDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderEndpointDescriptor")
            .field("protocol", &self.protocol)
            .field("base_url_present", &true)
            .finish_non_exhaustive()
    }
}

impl ProviderEndpointDescriptor {
    pub fn openai_compatible(base_url: &str) -> Result<Self, ProviderRegistryError> {
        let mut parsed = reqwest::Url::parse(base_url)
            .map_err(|_| ProviderRegistryError::InvalidEndpointUrl)?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.cannot_be_a_base() {
            return Err(ProviderRegistryError::InvalidEndpointUrl);
        }
        if !parsed.path().ends_with('/') {
            let mut path = parsed.path().to_owned();
            path.push('/');
            parsed.set_path(&path);
        }
        Ok(Self {
            protocol: ProviderProtocol::OpenAiCompatibleV1,
            base_url: parsed.to_string(),
        })
    }

    pub const fn protocol(&self) -> ProviderProtocol {
        self.protocol
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderModelCapabilities {
    pub supported_purposes: Vec<InferencePurpose>,
    pub context_window_tokens: u64,
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_structured_output: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderDataPolicyClaim {
    pub training_policy: ProviderTrainingPolicy,
    pub retention_policy: ProviderRetentionPolicy,
    pub routing_policy: ProviderRoutingPolicy,
}

/// Executable request-cost ceiling for one deployment/account scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRequestCostBound {
    account_scope_id: String,
    max_charge_microusd: u64,
}

impl ProviderRequestCostBound {
    pub fn new(
        account_scope_id: impl Into<String>,
        max_charge_microusd: u64,
    ) -> Result<Self, ProviderRegistryError> {
        let account_scope_id = account_scope_id.into();
        if !valid_text(&account_scope_id, MAX_ID_BYTES) {
            return Err(ProviderRegistryError::InvalidAccountScopeId);
        }
        Ok(Self {
            account_scope_id,
            max_charge_microusd,
        })
    }

    pub fn account_scope_id(&self) -> &str {
        &self.account_scope_id
    }

    pub const fn max_charge_microusd(&self) -> u64 {
        self.max_charge_microusd
    }
}

/// v1 accepts only first-party source classes. `SignedDeploymentManifest` remains
/// representable but cannot become executable authority until a verifier produces
/// an opaque verified-manifest capability instead of a caller-selected enum value.
fn source_allowed(role: ProviderClaimRole, source: ProviderClaimSourceKind) -> bool {
    match role {
        ProviderClaimRole::Endpoint
        | ProviderClaimRole::Lifecycle
        | ProviderClaimRole::Capabilities => {
            source == ProviderClaimSourceKind::FirstPartyModelCatalog
        }
        ProviderClaimRole::DataPolicy => {
            source == ProviderClaimSourceKind::FirstPartyPolicyDocument
        }
        ProviderClaimRole::RequestCost => {
            source == ProviderClaimSourceKind::FirstPartyAccountState
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProviderQualificationPolicy {
    pub allow_preview_models: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderModelProfile {
    provider_id: String,
    deployment_id: String,
    model_id: String,
    profile_epoch: u64,
    endpoint: SourcedProviderClaim<ProviderEndpointDescriptor>,
    lifecycle: SourcedProviderClaim<ProviderModelLifecycle>,
    capabilities: SourcedProviderClaim<ProviderModelCapabilities>,
    data_policy: SourcedProviderClaim<ProviderDataPolicyClaim>,
    request_cost: SourcedProviderClaim<ProviderRequestCostBound>,
}

impl ProviderModelProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider_id: impl Into<String>,
        deployment_id: impl Into<String>,
        model_id: impl Into<String>,
        profile_epoch: u64,
        endpoint: SourcedProviderClaim<ProviderEndpointDescriptor>,
        lifecycle: SourcedProviderClaim<ProviderModelLifecycle>,
        capabilities: SourcedProviderClaim<ProviderModelCapabilities>,
        data_policy: SourcedProviderClaim<ProviderDataPolicyClaim>,
        request_cost: SourcedProviderClaim<ProviderRequestCostBound>,
    ) -> Result<Self, ProviderRegistryError> {
        let provider_id = provider_id.into();
        let deployment_id = deployment_id.into();
        let model_id = model_id.into();
        if !valid_text(&provider_id, MAX_ID_BYTES) {
            return Err(ProviderRegistryError::InvalidProviderId);
        }
        if !valid_text(&deployment_id, MAX_ID_BYTES) {
            return Err(ProviderRegistryError::InvalidDeploymentId);
        }
        if !valid_text(&model_id, MAX_ID_BYTES) {
            return Err(ProviderRegistryError::InvalidModelId);
        }
        if profile_epoch == 0 {
            return Err(ProviderRegistryError::ZeroProfileEpoch);
        }
        Ok(Self {
            provider_id,
            deployment_id,
            model_id,
            profile_epoch,
            endpoint,
            lifecycle,
            capabilities,
            data_policy,
            request_cost,
        })
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub const fn profile_epoch(&self) -> u64 {
        self.profile_epoch
    }

    pub fn key(&self) -> ProviderProfileKey {
        ProviderProfileKey {
            provider_id: self.provider_id.clone(),
            deployment_id: self.deployment_id.clone(),
            model_id: self.model_id.clone(),
        }
    }

    pub fn qualify(
        &self,
        now_tick: u64,
        policy: ProviderQualificationPolicy,
    ) -> Result<QualifiedProviderCandidate, ProviderQualificationError> {
        self.endpoint
            .evidence
            .validate_for(ProviderClaimRole::Endpoint, now_tick)?;
        self.lifecycle
            .evidence
            .validate_for(ProviderClaimRole::Lifecycle, now_tick)?;
        self.capabilities
            .evidence
            .validate_for(ProviderClaimRole::Capabilities, now_tick)?;
        self.data_policy
            .evidence
            .validate_for(ProviderClaimRole::DataPolicy, now_tick)?;
        self.request_cost
            .evidence
            .validate_for(ProviderClaimRole::RequestCost, now_tick)?;

        match self.lifecycle.value {
            ProviderModelLifecycle::Production => {}
            ProviderModelLifecycle::Preview if policy.allow_preview_models => {}
            ProviderModelLifecycle::Preview => {
                return Err(ProviderQualificationError::PreviewModelForbidden);
            }
            ProviderModelLifecycle::Deprecated => {
                return Err(ProviderQualificationError::ModelDeprecated);
            }
            ProviderModelLifecycle::Unavailable => {
                return Err(ProviderQualificationError::ModelUnavailable);
            }
        }

        if self.capabilities.value.context_window_tokens == 0 {
            return Err(ProviderQualificationError::ZeroContextWindow);
        }
        if self.capabilities.value.supported_purposes.is_empty() {
            return Err(ProviderQualificationError::NoSupportedPurposes);
        }

        let mut purposes = self.capabilities.value.supported_purposes.clone();
        purposes.sort_by_key(|purpose| purpose_tag(*purpose));
        purposes.dedup();

        let valid_until_tick = self
            .endpoint
            .evidence
            .valid_until_tick
            .min(self.lifecycle.evidence.valid_until_tick)
            .min(self.capabilities.evidence.valid_until_tick)
            .min(self.data_policy.evidence.valid_until_tick)
            .min(self.request_cost.evidence.valid_until_tick);

        let profile_digest = digest_profile(self);
        let candidate = InferenceCandidate {
            provider_id: self.provider_id.clone(),
            model: ModelIdentity::ProviderAttested {
                provider: self.provider_id.clone(),
                declared_model: self.model_id.clone(),
            },
            location: ExecutionLocation::RemoteProvider,
            supported_purposes: purposes,
            context_window_tokens: self.capabilities.value.context_window_tokens,
            supports_streaming: self.capabilities.value.supports_streaming,
            supports_tools: self.capabilities.value.supports_tools,
            supports_structured_output: self.capabilities.value.supports_structured_output,
            training_policy: self.data_policy.value.training_policy,
            retention_policy: self.data_policy.value.retention_policy,
            routing_policy: self.data_policy.value.routing_policy,
            max_charge_microusd: Some(self.request_cost.value.max_charge_microusd),
        };

        Ok(QualifiedProviderCandidate {
            deployment_id: self.deployment_id.clone(),
            account_scope_id: self.request_cost.value.account_scope_id.clone(),
            endpoint: self.endpoint.value.clone(),
            candidate,
            profile_digest,
            profile_epoch: self.profile_epoch,
            qualified_at_tick: now_tick,
            valid_until_tick,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProviderProfileKey {
    pub provider_id: String,
    pub deployment_id: String,
    pub model_id: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProviderProfileDigest([u8; 32]);

impl ProviderProfileDigest {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for ProviderProfileDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ProviderProfileDigest").field(&self.0).finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedProviderCandidate {
    deployment_id: String,
    account_scope_id: String,
    endpoint: ProviderEndpointDescriptor,
    candidate: InferenceCandidate,
    profile_digest: ProviderProfileDigest,
    profile_epoch: u64,
    qualified_at_tick: u64,
    valid_until_tick: u64,
}

impl QualifiedProviderCandidate {
    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn account_scope_id(&self) -> &str {
        &self.account_scope_id
    }

    pub fn endpoint(&self) -> &ProviderEndpointDescriptor {
        &self.endpoint
    }

    pub fn candidate(&self) -> &InferenceCandidate {
        &self.candidate
    }

    pub const fn profile_digest(&self) -> ProviderProfileDigest {
        self.profile_digest
    }

    pub const fn profile_epoch(&self) -> u64 {
        self.profile_epoch
    }

    pub const fn qualified_at_tick(&self) -> u64 {
        self.qualified_at_tick
    }

    pub const fn valid_until_tick(&self) -> u64 {
        self.valid_until_tick
    }

    pub const fn is_fresh_at(&self, now_tick: u64) -> bool {
        now_tick >= self.qualified_at_tick && now_tick < self.valid_until_tick
    }
}

#[derive(Debug, Default)]
pub struct ProviderRegistry {
    profiles: BTreeMap<ProviderProfileKey, ProviderModelProfile>,
}

impl ProviderRegistry {
    pub fn install(&mut self, profile: ProviderModelProfile) -> Result<(), ProviderRegistryError> {
        let key = profile.key();
        if self
            .profiles
            .get(&key)
            .is_some_and(|existing| profile.profile_epoch <= existing.profile_epoch)
        {
            return Err(ProviderRegistryError::ProfileEpochNotAdvanced);
        }
        self.profiles.insert(key, profile);
        Ok(())
    }

    pub fn get(&self, key: &ProviderProfileKey) -> Option<&ProviderModelProfile> {
        self.profiles.get(key)
    }

    pub fn qualify_all(
        &self,
        now_tick: u64,
        policy: ProviderQualificationPolicy,
    ) -> ProviderRegistrySnapshot {
        let mut qualified = Vec::new();
        let mut rejected = Vec::new();
        for (key, profile) in &self.profiles {
            match profile.qualify(now_tick, policy) {
                Ok(candidate) => qualified.push(candidate),
                Err(error) => rejected.push(RejectedProviderProfile {
                    key: key.clone(),
                    error,
                }),
            }
        }
        ProviderRegistrySnapshot {
            qualified,
            rejected,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedProviderProfile {
    pub key: ProviderProfileKey,
    pub error: ProviderQualificationError,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRegistrySnapshot {
    qualified: Vec<QualifiedProviderCandidate>,
    rejected: Vec<RejectedProviderProfile>,
}

impl ProviderRegistrySnapshot {
    pub fn qualified(&self) -> &[QualifiedProviderCandidate] {
        &self.qualified
    }

    pub fn rejected(&self) -> &[RejectedProviderProfile] {
        &self.rejected
    }
}

fn digest_profile(profile: &ProviderModelProfile) -> ProviderProfileDigest {
    let mut h = ProfileHasher::new();
    h.string(&profile.provider_id);
    h.string(&profile.deployment_id);
    h.string(&profile.model_id);
    h.u64(profile.profile_epoch);

    h.claim_evidence(&profile.endpoint.evidence);
    h.u8(protocol_tag(profile.endpoint.value.protocol));
    h.string(&profile.endpoint.value.base_url);

    h.claim_evidence(&profile.lifecycle.evidence);
    h.u8(lifecycle_tag(profile.lifecycle.value));

    h.claim_evidence(&profile.capabilities.evidence);
    let mut purposes: Vec<u8> = profile
        .capabilities
        .value
        .supported_purposes
        .iter()
        .copied()
        .map(purpose_tag)
        .collect();
    purposes.sort_unstable();
    purposes.dedup();
    h.u64(purposes.len() as u64);
    for purpose in purposes {
        h.u8(purpose);
    }
    h.u64(profile.capabilities.value.context_window_tokens);
    h.bool(profile.capabilities.value.supports_streaming);
    h.bool(profile.capabilities.value.supports_tools);
    h.bool(profile.capabilities.value.supports_structured_output);

    h.claim_evidence(&profile.data_policy.evidence);
    h.u8(training_tag(profile.data_policy.value.training_policy));
    h.u8(retention_tag(profile.data_policy.value.retention_policy));
    h.u8(routing_tag(profile.data_policy.value.routing_policy));

    h.claim_evidence(&profile.request_cost.evidence);
    h.string(&profile.request_cost.value.account_scope_id);
    h.u64(profile.request_cost.value.max_charge_microusd);

    ProviderProfileDigest(*h.finish().as_bytes())
}

pub fn digest_provider_snapshot(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

struct ProfileHasher {
    hasher: blake3::Hasher,
}

impl ProfileHasher {
    fn new() -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(PROFILE_DOMAIN.len() as u64).to_le_bytes());
        hasher.update(PROFILE_DOMAIN);
        Self { hasher }
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u8(&mut self, value: u8) {
        self.hasher.update(&[value]);
    }

    fn u64(&mut self, value: u64) {
        self.hasher.update(&value.to_le_bytes());
    }

    fn string(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.hasher.update(value.as_bytes());
    }

    fn claim_evidence(&mut self, evidence: &ProviderClaimEvidence) {
        self.u8(source_tag(evidence.source_kind));
        self.string(&evidence.source_locator);
        self.hasher.update(&evidence.snapshot_digest);
        self.u64(evidence.observed_at_tick);
        self.u64(evidence.valid_until_tick);
    }

    fn finish(self) -> blake3::Hash {
        self.hasher.finalize()
    }
}

fn valid_text(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_bytes
        && !value.chars().any(char::is_control)
}

fn protocol_tag(value: ProviderProtocol) -> u8 {
    match value {
        ProviderProtocol::OpenAiCompatibleV1 => 1,
    }
}

fn lifecycle_tag(value: ProviderModelLifecycle) -> u8 {
    match value {
        ProviderModelLifecycle::Production => 1,
        ProviderModelLifecycle::Preview => 2,
        ProviderModelLifecycle::Deprecated => 3,
        ProviderModelLifecycle::Unavailable => 4,
    }
}

fn source_tag(value: ProviderClaimSourceKind) -> u8 {
    match value {
        ProviderClaimSourceKind::FirstPartyPolicyDocument => 1,
        ProviderClaimSourceKind::FirstPartyModelCatalog => 2,
        ProviderClaimSourceKind::FirstPartyPricingDocument => 3,
        ProviderClaimSourceKind::FirstPartyAccountState => 4,
        ProviderClaimSourceKind::SignedDeploymentManifest => 5,
        ProviderClaimSourceKind::ThirdPartyCurator => 6,
    }
}

fn purpose_tag(value: InferencePurpose) -> u8 {
    match value {
        InferencePurpose::Translation => 1,
        InferencePurpose::Summarization => 2,
        InferencePurpose::Extraction => 3,
        InferencePurpose::Classification => 4,
        InferencePurpose::GeneralReasoning => 5,
        InferencePurpose::MathematicalReasoning => 6,
        InferencePurpose::ScientificReasoning => 7,
        InferencePurpose::CodeGeneration => 8,
        InferencePurpose::CodeRepair => 9,
        InferencePurpose::CodeReview => 10,
        InferencePurpose::CreativeWriting => 11,
        InferencePurpose::Dialogue => 12,
        InferencePurpose::ToolProposal => 13,
        InferencePurpose::Embedding => 14,
        InferencePurpose::Reranking => 15,
    }
}

fn training_tag(value: ProviderTrainingPolicy) -> u8 {
    match value {
        ProviderTrainingPolicy::Never => 1,
        ProviderTrainingPolicy::MayTrain => 2,
        ProviderTrainingPolicy::Unknown => 3,
    }
}

fn retention_tag(value: ProviderRetentionPolicy) -> u8 {
    match value {
        ProviderRetentionPolicy::ZeroRetention => 1,
        ProviderRetentionPolicy::MayRetain => 2,
        ProviderRetentionPolicy::Unknown => 3,
    }
}

fn routing_tag(value: ProviderRoutingPolicy) -> u8 {
    match value {
        ProviderRoutingPolicy::DirectOnly => 1,
        ProviderRoutingPolicy::MayRouteThirdParty => 2,
        ProviderRoutingPolicy::Unknown => 3,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderRegistryError {
    InvalidProviderId,
    InvalidDeploymentId,
    InvalidModelId,
    InvalidAccountScopeId,
    InvalidSourceLocator,
    InvalidEndpointUrl,
    ZeroSnapshotDigest,
    InvalidClaimValidityWindow,
    ZeroProfileEpoch,
    ProfileEpochNotAdvanced,
}

impl fmt::Display for ProviderRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "provider registry error: {self:?}")
    }
}

impl std::error::Error for ProviderRegistryError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderQualificationError {
    ClaimNotYetValid(ProviderClaimRole),
    ClaimExpired(ProviderClaimRole),
    SourceNotAuthoritative {
        role: ProviderClaimRole,
        source: ProviderClaimSourceKind,
    },
    PreviewModelForbidden,
    ModelDeprecated,
    ModelUnavailable,
    ZeroContextWindow,
    NoSupportedPurposes,
}

impl fmt::Display for ProviderQualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "provider qualification failed: {self:?}")
    }
}

impl std::error::Error for ProviderQualificationError {}
