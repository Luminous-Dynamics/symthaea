//! Caller-, session-, consent-, provider- and invocation-bound mobile
//! authorization leases.
//!
//! This module prepares one bounded, one-shot authorization for an explicitly
//! selected mobile observation. It does not execute ADB/MCP operations, verify
//! consent signatures, authenticate the provider executable, or attest device
//! state. The gateway must perform those independent checks and use a durable
//! replay ledger before dispatching the command.

use crate::mobile_tool_effects::{
    EffectCeiling, MobileProviderCapabilityReport, MobileTool, ProviderBoundToolAuthorization,
    ToolAuthorizationError, authorize_reported_mobile_tool,
};
use crate::{canonical_jcs_bytes, sha256_hex};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub const MOBILE_INVOCATION_MAX_TTL_MS: u64 = 60_000;
pub const MOBILE_INVOCATION_MAX_CLOCK_SKEW_MS: u64 = 5_000;
pub const MOBILE_INVOCATION_MAX_LOCAL_REPLAY_ENTRIES: usize = 65_536;

/// Explicitly selected, ephemeral device session. The serial-to-alias mapping
/// remains local to the gateway and is never included in the portable receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MobileDeviceSessionSnapshot {
    pub session_alias: String,
    pub generation: u64,
    pub session_nonce: String,
    /// Optional digest of a non-unique target/build scope, never a serial/IMEI.
    pub target_scope_sha256: Option<String>,
}

/// Caller-consent record already obtained by the gateway's consent workflow.
///
/// Its digest and verifier label bind the supplied metadata, but this module
/// does not validate a signature or decide whether the verifier is trusted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MobileConsentGrant {
    pub consent_id: String,
    pub consent_receipt_sha256: String,
    pub verifier_id: String,
    pub caller_id: String,
    pub session_alias: String,
    pub session_generation: u64,
    pub allowed_tools: Vec<MobileTool>,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    /// Populated only by the gateway's consent/revocation verifier.
    pub revoked_at_unix_ms: Option<u64>,
}

/// Exact, one-use request presented to the gateway.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MobileInvocationRequest {
    pub invocation_id: String,
    pub invocation_nonce: String,
    pub caller_id: String,
    pub session_alias: String,
    pub session_generation: u64,
    pub session_nonce: String,
    pub consent_id: String,
    pub consent_receipt_sha256: String,
    pub tool_name: String,
    pub arguments: Value,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

/// Metadata-safe receipt. Raw ephemeral session nonce and raw arguments are
/// deliberately excluded; their digests bind the invocation without exporting
/// a reusable session token or potentially sensitive command values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MobileInvocationReceipt {
    pub schema: String,
    pub invocation_id: String,
    pub invocation_nonce_sha256: String,
    pub caller_id: String,
    pub session_alias: String,
    pub session_generation: u64,
    pub session_nonce_sha256: String,
    pub target_scope_sha256: Option<String>,
    pub consent_id: String,
    pub consent_receipt_sha256: String,
    pub consent_verifier_id: String,
    pub tool: MobileTool,
    pub arguments_sha256: String,
    pub provider_id: String,
    pub provider_version: Option<String>,
    pub provider_revision: String,
    pub provider_executable_sha256: Option<String>,
    pub capability_report_sha256: String,
    pub effect_catalog_sha256: String,
    pub policy_id: String,
    pub policy_sha256: String,
    pub effect_sha256: String,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub prepared_at_unix_ms: u64,
    pub receipt_sha256: String,
}

/// Prepared authorization passed to a narrowly scoped execution adapter.
/// The arguments are kept separate from the portable receipt and are not
/// serializable through this type.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthorizedMobileInvocation {
    pub receipt: MobileInvocationReceipt,
    arguments: Value,
    pub provider_authorization: ProviderBoundToolAuthorization,
}

impl AuthorizedMobileInvocation {
    pub fn arguments(&self) -> &Value {
        &self.arguments
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MobileInvocationError {
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    InvalidTimestamp(&'static str),
    ConsentExpired,
    ConsentRevoked,
    ReplayCapacityExceeded,
    ConsentScopeMismatch(&'static str),
    SessionMismatch(&'static str),
    InvocationExpired,
    InvocationLifetimeExceeded,
    ToolNotConsented,
    InvalidArguments,
    Replay(&'static str),
    Provider(ToolAuthorizationError),
    Canonicalization(String),
}

impl fmt::Display for MobileInvocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier(field) => write!(f, "{field} is not a valid opaque identifier"),
            Self::InvalidDigest(field) => write!(f, "{field} is not a lowercase SHA-256 digest"),
            Self::InvalidTimestamp(field) => write!(f, "{field} is invalid"),
            Self::ConsentExpired => write!(f, "caller consent is expired or not yet valid"),
            Self::ConsentRevoked => write!(f, "caller consent has been revoked"),
            Self::ReplayCapacityExceeded => write!(f, "process-local replay guard reached its bounded capacity"),
            Self::ConsentScopeMismatch(field) => write!(f, "consent does not authorize {field}"),
            Self::SessionMismatch(field) => write!(f, "selected device session does not match {field}"),
            Self::InvocationExpired => write!(f, "invocation authorization has expired"),
            Self::InvocationLifetimeExceeded => write!(f, "invocation lifetime exceeds the configured bound"),
            Self::ToolNotConsented => write!(f, "requested tool is outside the caller-consented allowlist"),
            Self::InvalidArguments => write!(f, "invocation arguments must be a JSON object"),
            Self::Replay(field) => write!(f, "replayed mobile invocation {field} rejected"),
            Self::Provider(error) => write!(f, "provider/tool authorization failed: {error}"),
            Self::Canonicalization(error) => write!(f, "invocation receipt canonicalization failed: {error}"),
        }
    }
}

impl std::error::Error for MobileInvocationError {}

impl From<ToolAuthorizationError> for MobileInvocationError {
    fn from(value: ToolAuthorizationError) -> Self {
        Self::Provider(value)
    }
}

/// Validate the consent, selected device session, exact tool, provider report,
/// arguments and invocation time window; then produce a digest-bound receipt.
///
/// The gateway supplies the current time and current session snapshot rather
/// than allowing the agent to select time or device state itself.
pub fn prepare_mobile_invocation(
    request: &MobileInvocationRequest,
    consent: &MobileConsentGrant,
    session: &MobileDeviceSessionSnapshot,
    provider: &MobileProviderCapabilityReport,
    policy_id: &str,
    ceiling: EffectCeiling,
    now_unix_ms: u64,
) -> Result<AuthorizedMobileInvocation, MobileInvocationError> {
    validate_opaque_id(&request.invocation_id, "invocation_id", "inv_")?;
    validate_opaque_id(&request.invocation_nonce, "invocation_nonce", "nonce_")?;
    validate_opaque_id(&consent.consent_id, "consent_id", "consent_")?;
    validate_opaque_id(&session.session_alias, "session_alias", "ds_")?;
    validate_opaque_id(&session.session_nonce, "session_nonce", "session_")?;
    validate_sha256(&consent.consent_receipt_sha256, "consent_receipt_sha256")?;
    if let Some(target_digest) = &session.target_scope_sha256 {
        validate_sha256(target_digest, "target_scope_sha256")?;
    }
    if consent.caller_id.trim().is_empty() || request.caller_id.trim().is_empty() {
        return Err(MobileInvocationError::InvalidIdentifier("caller_id"));
    }
    if consent.verifier_id.trim().is_empty() {
        return Err(MobileInvocationError::InvalidIdentifier("consent verifier_id"));
    }

    if consent.issued_at_unix_ms > now_unix_ms
        || consent.expires_at_unix_ms <= now_unix_ms
        || consent.expires_at_unix_ms <= consent.issued_at_unix_ms
    {
        return Err(MobileInvocationError::ConsentExpired);
    }
    if let Some(revoked_at) = consent.revoked_at_unix_ms {
        if revoked_at < consent.issued_at_unix_ms {
            return Err(MobileInvocationError::InvalidTimestamp("revoked_at_unix_ms"));
        }
        if revoked_at <= now_unix_ms {
            return Err(MobileInvocationError::ConsentRevoked);
        }
    }
    if session.generation == 0 {
        return Err(MobileInvocationError::SessionMismatch("zero session generation"));
    }
    let mut consented_tools = HashSet::new();
    if consent.allowed_tools.is_empty()
        || !consent.allowed_tools.iter().all(|tool| consented_tools.insert(*tool))
    {
        return Err(MobileInvocationError::ConsentScopeMismatch(
            "empty or duplicate tool allowlist",
        ));
    }

    if request.issued_at_unix_ms > now_unix_ms.saturating_add(MOBILE_INVOCATION_MAX_CLOCK_SKEW_MS) {
        return Err(MobileInvocationError::InvalidTimestamp("issued_at_unix_ms is in the future"));
    }
    if request.issued_at_unix_ms < consent.issued_at_unix_ms
        || request.issued_at_unix_ms > consent.expires_at_unix_ms
    {
        return Err(MobileInvocationError::ConsentScopeMismatch("invocation issue time"));
    }
    if request.expires_at_unix_ms <= now_unix_ms {
        return Err(MobileInvocationError::InvocationExpired);
    }
    if request.expires_at_unix_ms <= request.issued_at_unix_ms
        || request.expires_at_unix_ms > consent.expires_at_unix_ms
    {
        return Err(MobileInvocationError::InvalidTimestamp("expires_at_unix_ms"));
    }
    if request.expires_at_unix_ms - request.issued_at_unix_ms > MOBILE_INVOCATION_MAX_TTL_MS {
        return Err(MobileInvocationError::InvocationLifetimeExceeded);
    }

    if request.caller_id != consent.caller_id {
        return Err(MobileInvocationError::ConsentScopeMismatch("caller_id"));
    }
    if request.session_alias != session.session_alias
        || consent.session_alias != session.session_alias
    {
        return Err(MobileInvocationError::SessionMismatch("session_alias"));
    }
    if request.session_generation != session.generation
        || consent.session_generation != session.generation
    {
        return Err(MobileInvocationError::SessionMismatch("session_generation"));
    }
    if request.session_nonce != session.session_nonce {
        return Err(MobileInvocationError::SessionMismatch("session_nonce"));
    }
    if request.consent_id != consent.consent_id
        || request.consent_receipt_sha256 != consent.consent_receipt_sha256
    {
        return Err(MobileInvocationError::ConsentScopeMismatch("consent receipt"));
    }

    let tool = MobileTool::parse(&request.tool_name)
        .ok_or_else(|| MobileInvocationError::Provider(ToolAuthorizationError::UnknownTool(request.tool_name.clone())))?;
    if !consent.allowed_tools.contains(&tool) {
        return Err(MobileInvocationError::ToolNotConsented);
    }
    if request.arguments.as_object().is_none() {
        return Err(MobileInvocationError::InvalidArguments);
    }

    let provider_authorization = authorize_reported_mobile_tool(
        &request.tool_name,
        provider,
        policy_id,
        ceiling,
    )?;

    let arguments_bytes = canonical_jcs_bytes(&request.arguments)
        .map_err(|error| MobileInvocationError::Canonicalization(error.to_string()))?;
    let arguments_sha256 = sha256_hex(&arguments_bytes);
    let invocation_nonce_sha256 = sha256_hex(request.invocation_nonce.as_bytes());
    let session_nonce_sha256 = sha256_hex(session.session_nonce.as_bytes());

    let mut receipt = MobileInvocationReceipt {
        schema: "luminous.mobile-invocation-receipt.v1".to_owned(),
        invocation_id: request.invocation_id.clone(),
        invocation_nonce_sha256,
        caller_id: request.caller_id.clone(),
        session_alias: session.session_alias.clone(),
        session_generation: session.generation,
        session_nonce_sha256,
        target_scope_sha256: session.target_scope_sha256.clone(),
        consent_id: consent.consent_id.clone(),
        consent_receipt_sha256: consent.consent_receipt_sha256.clone(),
        consent_verifier_id: consent.verifier_id.clone(),
        tool,
        arguments_sha256,
        provider_id: provider_authorization.provider_id.clone(),
        provider_version: provider_authorization.provider_version.clone(),
        provider_revision: provider_authorization.provider_revision.clone(),
        provider_executable_sha256: provider_authorization.provider_executable_sha256.clone(),
        capability_report_sha256: provider_authorization.capability_report_sha256.clone(),
        effect_catalog_sha256: provider.effect_catalog_sha256.clone(),
        policy_id: provider_authorization.authorization.policy_id.clone(),
        policy_sha256: provider_authorization.authorization.policy_sha256.clone(),
        effect_sha256: provider_authorization.authorization.effect_sha256.clone(),
        issued_at_unix_ms: request.issued_at_unix_ms,
        expires_at_unix_ms: request.expires_at_unix_ms,
        prepared_at_unix_ms: now_unix_ms,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = mobile_invocation_receipt_sha256(&receipt)?;

    Ok(AuthorizedMobileInvocation {
        receipt,
        arguments: request.arguments.clone(),
        provider_authorization,
    })
}

/// Recompute the canonical receipt digest. The receipt's own digest field is
/// excluded from its commitment.
pub fn mobile_invocation_receipt_sha256(
    receipt: &MobileInvocationReceipt,
) -> Result<String, MobileInvocationError> {
    let mut value = serde_json::to_value(receipt)
        .map_err(|error| MobileInvocationError::Canonicalization(error.to_string()))?;
    value.as_object_mut()
        .ok_or_else(|| MobileInvocationError::Canonicalization("receipt must serialize as object".to_owned()))?
        .remove("receipt_sha256");
    let bytes = canonical_jcs_bytes(&value)
        .map_err(|error| MobileInvocationError::Canonicalization(error.to_string()))?;
    Ok(sha256_hex(&bytes))
}

/// Verify the internal receipt commitment. This is not a signature check.
pub fn verify_mobile_invocation_receipt(
    receipt: &MobileInvocationReceipt,
) -> Result<(), MobileInvocationError> {
    validate_opaque_id(&receipt.invocation_id, "invocation_id", "inv_")?;
    validate_opaque_id(&receipt.session_alias, "session_alias", "ds_")?;
    validate_opaque_id(&receipt.consent_id, "consent_id", "consent_")?;
    if receipt.caller_id.trim().is_empty() || receipt.consent_verifier_id.trim().is_empty()
        || receipt.provider_id.trim().is_empty() || receipt.provider_revision.trim().is_empty()
    {
        return Err(MobileInvocationError::InvalidIdentifier("receipt principal/provider"));
    }
    validate_sha256(&receipt.invocation_nonce_sha256, "invocation_nonce_sha256")?;
    validate_sha256(&receipt.session_nonce_sha256, "session_nonce_sha256")?;
    validate_sha256(&receipt.consent_receipt_sha256, "consent_receipt_sha256")?;
    validate_sha256(&receipt.arguments_sha256, "arguments_sha256")?;
    validate_sha256(&receipt.capability_report_sha256, "capability_report_sha256")?;
    validate_sha256(&receipt.effect_catalog_sha256, "effect_catalog_sha256")?;
    validate_sha256(&receipt.policy_sha256, "policy_sha256")?;
    validate_sha256(&receipt.effect_sha256, "effect_sha256")?;
    validate_sha256(&receipt.receipt_sha256, "receipt_sha256")?;
    if receipt.caller_id.trim().is_empty()
        || receipt.consent_verifier_id.trim().is_empty()
        || receipt.provider_id.trim().is_empty()
        || receipt.provider_revision.trim().is_empty()
        || receipt.policy_id.trim().is_empty()
    {
        return Err(MobileInvocationError::InvalidIdentifier("receipt principal/provider/policy"));
    }
    if let Some(executable_digest) = &receipt.provider_executable_sha256 {
        validate_sha256(executable_digest, "provider_executable_sha256")?;
    }
    if let Some(target_digest) = &receipt.target_scope_sha256 {
        validate_sha256(target_digest, "target_scope_sha256")?;
    }
    if receipt.schema != "luminous.mobile-invocation-receipt.v1" {
        return Err(MobileInvocationError::InvalidIdentifier("receipt schema"));
    }
    if receipt.expires_at_unix_ms <= receipt.issued_at_unix_ms
        || receipt.expires_at_unix_ms - receipt.issued_at_unix_ms > MOBILE_INVOCATION_MAX_TTL_MS
        || receipt.prepared_at_unix_ms >= receipt.expires_at_unix_ms
        || receipt.issued_at_unix_ms
            > receipt.prepared_at_unix_ms.saturating_add(MOBILE_INVOCATION_MAX_CLOCK_SKEW_MS)
        || receipt.prepared_at_unix_ms < receipt.issued_at_unix_ms
    {
        return Err(MobileInvocationError::InvalidTimestamp("receipt time bounds"));
    }
    if mobile_invocation_receipt_sha256(receipt)? != receipt.receipt_sha256 {
        return Err(MobileInvocationError::Canonicalization(
            "receipt digest does not match its payload".to_owned(),
        ));
    }
    Ok(())
}

/// Process-local, fail-closed replay guard. Production gateways must persist
/// claims transactionally across restarts and share the store across adapters.
#[derive(Debug, Default)]
pub struct MobileInvocationReplayGuard {
    invocation_ids: HashMap<String, u64>,
    nonce_digests: HashMap<String, u64>,
}

impl MobileInvocationReplayGuard {
    fn discard_expired(&mut self, now_unix_ms: u64) {
        self.invocation_ids.retain(|_, expires_at| *expires_at > now_unix_ms);
        self.nonce_digests.retain(|_, expires_at| *expires_at > now_unix_ms);
    }

    /// Validate first, then claim both the invocation ID and nonce in this
    /// process. The in-memory guard is bounded and expiry-aware, but it is not
    /// crash-safe or shared across processes. Production gateways must replace
    /// it with a durable transactional replay store.
    pub fn prepare_once(
        &mut self,
        request: &MobileInvocationRequest,
        consent: &MobileConsentGrant,
        session: &MobileDeviceSessionSnapshot,
        provider: &MobileProviderCapabilityReport,
        policy_id: &str,
        ceiling: EffectCeiling,
        now_unix_ms: u64,
    ) -> Result<AuthorizedMobileInvocation, MobileInvocationError> {
        let prepared = prepare_mobile_invocation(
            request, consent, session, provider, policy_id, ceiling, now_unix_ms,
        )?;
        self.discard_expired(now_unix_ms);
        let nonce_digest = sha256_hex(request.invocation_nonce.as_bytes());
        if self.invocation_ids.contains_key(&request.invocation_id) {
            return Err(MobileInvocationError::Replay("invocation_id"));
        }
        if self.nonce_digests.contains_key(&nonce_digest) {
            return Err(MobileInvocationError::Replay("invocation_nonce"));
        }
        if self.invocation_ids.len() >= MOBILE_INVOCATION_MAX_LOCAL_REPLAY_ENTRIES
            || self.nonce_digests.len() >= MOBILE_INVOCATION_MAX_LOCAL_REPLAY_ENTRIES
        {
            return Err(MobileInvocationError::ReplayCapacityExceeded);
        }
        self.invocation_ids.insert(
            request.invocation_id.clone(),
            request.expires_at_unix_ms,
        );
        self.nonce_digests.insert(nonce_digest, request.expires_at_unix_ms);
        Ok(prepared)
    }

    pub fn claimed_invocation_count(&self) -> usize {
        self.invocation_ids.len()
    }
}

fn validate_opaque_id(
    value: &str,
    field: &'static str,
    prefix: &'static str,
) -> Result<(), MobileInvocationError> {
    let Some(suffix) = value.strip_prefix(prefix) else {
        return Err(MobileInvocationError::InvalidIdentifier(field));
    };
    if suffix.len() != 64
        || !suffix
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MobileInvocationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), MobileInvocationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MobileInvocationError::InvalidDigest(field));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mobile_tool_effects::{
        MOBILE_TOOL_EFFECTS, MOBILE_DEVICE_READ_ONLY_POLICY_ID, MobileProviderCapabilityReport,
        ProviderAvailability, ProviderOperationCapability, mobile_tool_catalog_sha256,
    };
    use serde_json::json;

    const NOW: u64 = 1_800_000_000_000;
    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn report() -> MobileProviderCapabilityReport {
        MobileProviderCapabilityReport {
            provider_id: "adb-local".to_owned(),
            provider_version: Some("35.0.2".to_owned()),
            provider_revision: "adb-r35".to_owned(),
            provider_executable_sha256: Some(HASH_A.to_owned()),
            upstream_rea_commit: crate::mobile_tool_effects::UPSTREAM_REA_COMMIT.to_owned(),
            effect_catalog_sha256: mobile_tool_catalog_sha256().unwrap(),
            availability: ProviderAvailability::Available,
            reason: None,
            operations: MOBILE_TOOL_EFFECTS
                .iter()
                .map(|entry| ProviderOperationCapability {
                    tool: entry.tool,
                    available: true,
                    reason: None,
                    effects: entry.effects,
                    returns_target_content: entry.returns_target_content,
                    limitations: Vec::new(),
                })
                .collect(),
            limitations: Vec::new(),
        }
    }

    fn aliases() -> (MobileDeviceSessionSnapshot, MobileConsentGrant) {
        let session = MobileDeviceSessionSnapshot {
            session_alias: format!("ds_{}", "1".repeat(64)),
            generation: 7,
            session_nonce: format!("session_{}", "2".repeat(64)),
            target_scope_sha256: Some(HASH_A.to_owned()),
        };
        let consent = MobileConsentGrant {
            consent_id: format!("consent_{}", "3".repeat(64)),
            consent_receipt_sha256: "4".repeat(64),
            verifier_id: "local-consent-ui-v1".to_owned(),
            caller_id: "human-operator".to_owned(),
            session_alias: session.session_alias.clone(),
            session_generation: session.generation,
            allowed_tools: vec![MobileTool::InspectAdbDevice],
            issued_at_unix_ms: NOW - 10_000,
            expires_at_unix_ms: NOW + 120_000,
            revoked_at_unix_ms: None,
        };
        (session, consent)
    }

    fn build_request(session: &MobileDeviceSessionSnapshot, consent: &MobileConsentGrant) -> MobileInvocationRequest {
        MobileInvocationRequest {
            invocation_id: format!("inv_{}", "5".repeat(64)),
            invocation_nonce: format!("nonce_{}", "6".repeat(64)),
            caller_id: consent.caller_id.clone(),
            session_alias: session.session_alias.clone(),
            session_generation: session.generation,
            session_nonce: session.session_nonce.clone(),
            consent_id: consent.consent_id.clone(),
            consent_receipt_sha256: consent.consent_receipt_sha256.clone(),
            tool_name: "inspect_adb_device".to_owned(),
            arguments: json!({"include_properties": false, "sections": ["build", "services"]}),
            issued_at_unix_ms: NOW - 1_000,
            expires_at_unix_ms: NOW + 5_000,
        }
    }

    #[test]
    fn prepares_exact_session_consent_provider_and_argument_bound_receipt() {
        let (session, consent) = aliases();
        let request = build_request(&session, &consent);
        let authorized = prepare_mobile_invocation(
            &request, &consent, &session, &report(),
            MOBILE_DEVICE_READ_ONLY_POLICY_ID,
            EffectCeiling::MOBILE_DEVICE_READ_ONLY,
            NOW,
        ).unwrap();

        assert_eq!(authorized.receipt.tool, MobileTool::InspectAdbDevice);
        assert_eq!(authorized.arguments(), &request.arguments);
        assert_eq!(authorized.receipt.session_alias, session.session_alias);
        assert_eq!(authorized.receipt.session_generation, 7);
        assert_eq!(authorized.receipt.target_scope_sha256, Some(HASH_A.to_owned()));
        assert_eq!(authorized.receipt.provider_id, "adb-local");
        assert_eq!(authorized.receipt.arguments_sha256, sha256_hex(
            &canonical_jcs_bytes(&request.arguments).unwrap()
        ));
        assert_eq!(authorized.receipt.session_nonce_sha256, sha256_hex(session.session_nonce.as_bytes()));
        verify_mobile_invocation_receipt(&authorized.receipt).unwrap();
    }

    #[test]
    fn rejects_session_alias_generation_or_nonce_substitution() {
        let (session, consent) = aliases();
        let mut request = build_request(&session, &consent);
        request.session_alias = format!("ds_{}", "9".repeat(64));
        assert!(matches!(
            prepare_mobile_invocation(&request, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW),
            Err(MobileInvocationError::SessionMismatch("session_alias"))
        ));

        let mut request = build_request(&session, &consent);
        request.session_generation += 1;
        assert!(matches!(
            prepare_mobile_invocation(&request, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW),
            Err(MobileInvocationError::SessionMismatch("session_generation"))
        ));

        let mut request = build_request(&session, &consent);
        request.session_nonce = format!("session_{}", "9".repeat(64));
        assert!(matches!(
            prepare_mobile_invocation(&request, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW),
            Err(MobileInvocationError::SessionMismatch("session_nonce"))
        ));
    }

    #[test]
    fn rejects_expired_or_overlong_invocations_and_stale_consent() {
        let (session, consent) = aliases();
        let mut request = build_request(&session, &consent);
        request.expires_at_unix_ms = NOW;
        assert!(matches!(
            prepare_mobile_invocation(&request, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW),
            Err(MobileInvocationError::InvocationExpired)
        ));

        let mut request = build_request(&session, &consent);
        request.expires_at_unix_ms = request.issued_at_unix_ms + MOBILE_INVOCATION_MAX_TTL_MS + 1;
        assert!(matches!(
            prepare_mobile_invocation(&request, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW),
            Err(MobileInvocationError::InvocationLifetimeExceeded)
        ));

        let mut consent = consent.clone();
        consent.expires_at_unix_ms = NOW;
        let request = build_request(&session, &consent);
        assert!(matches!(
            prepare_mobile_invocation(&request, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW),
            Err(MobileInvocationError::ConsentExpired)
        ));
    }

    #[test]
    fn rejects_revoked_consent_before_provider_authorization() {
        let (session, mut consent) = aliases();
        consent.revoked_at_unix_ms = Some(NOW - 1);
        let error = prepare_mobile_invocation(
            &build_request(&session, &consent), &consent, &session, &report(),
            MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW,
        ).unwrap_err();
        assert_eq!(error, MobileInvocationError::ConsentRevoked);
    }

    #[test]
    fn rejects_provider_operation_reported_unavailable() {
        let (session, consent) = aliases();
        let mut provider = report();
        provider.availability = ProviderAvailability::PartiallyAvailable;
        let operation = provider.operations.iter_mut()
            .find(|entry| entry.tool == MobileTool::InspectAdbDevice)
            .unwrap();
        operation.available = false;
        operation.reason = Some("adb transport disconnected".to_owned());

        let error = prepare_mobile_invocation(
            &build_request(&session, &consent), &consent, &session, &provider,
            MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW,
        ).unwrap_err();
        assert!(matches!(error, MobileInvocationError::Provider(
            ToolAuthorizationError::ProviderOperationUnavailable { tool: MobileTool::InspectAdbDevice, .. }
        )));
    }

    #[test]
    fn replay_guard_discards_expired_process_local_claims() {
        let (session, consent) = aliases();
        let mut guard = MobileInvocationReplayGuard::default();
        let first = build_request(&session, &consent);
        guard.prepare_once(
            &first, &consent, &session, &report(),
            MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW,
        ).unwrap();
        assert_eq!(guard.claimed_invocation_count(), 1);

        let mut second = build_request(&session, &consent);
        second.invocation_id = format!("inv_{}", "7".repeat(64));
        second.invocation_nonce = format!("nonce_{}", "8".repeat(64));
        second.issued_at_unix_ms = NOW + 6_000;
        second.expires_at_unix_ms = NOW + 11_000;
        guard.prepare_once(
            &second, &consent, &session, &report(),
            MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW + 6_000,
        ).unwrap();
        assert_eq!(guard.claimed_invocation_count(), 1);
    }

    #[test]
    fn rejects_caller_consent_tool_and_argument_substitution() {
        let (session, consent) = aliases();
        let mut request = build_request(&session, &consent);
        request.caller_id = "different-caller".to_owned();
        assert!(matches!(
            prepare_mobile_invocation(&request, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW),
            Err(MobileInvocationError::ConsentScopeMismatch("caller_id"))
        ));

        let mut request = build_request(&session, &consent);
        request.tool_name = "pull_adb_package".to_owned();
        assert!(matches!(
            prepare_mobile_invocation(&request, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW),
            Err(MobileInvocationError::ToolNotConsented)
        ));

        let mut request = build_request(&session, &consent);
        request.arguments = json!([]);
        assert_eq!(
            prepare_mobile_invocation(&request, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW)
            .unwrap_err(),
            MobileInvocationError::InvalidArguments
        );
    }

    #[test]
    fn rejects_tampered_portable_receipt() {
        let (session, consent) = aliases();
        let mut authorized = prepare_mobile_invocation(
            &build_request(&session, &consent), &consent, &session, &report(),
            MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW,
        ).unwrap();
        authorized.receipt.arguments_sha256 = "9".repeat(64);
        assert!(verify_mobile_invocation_receipt(&authorized.receipt).is_err());
    }

    #[test]
    fn replay_guard_consumes_only_valid_invocations_and_rejects_replay() {
        let (session, consent) = aliases();
        let req = build_request(&session, &consent);
        let mut guard = MobileInvocationReplayGuard::default();

        assert!(guard.prepare_once(
            &req, &consent, &session, &report(),
            MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW,
        ).is_ok());
        assert_eq!(guard.claimed_invocation_count(), 1);
        assert!(matches!(
            guard.prepare_once(
                &req, &consent, &session, &report(),
                MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW,
            ),
            Err(MobileInvocationError::Replay("invocation_id"))
        ));

        let mut invalid = build_request(&session, &consent);
        invalid.invocation_id = format!("inv_{}", "7".repeat(64));
        invalid.invocation_nonce = "bad".to_owned();
        assert!(guard.prepare_once(
            &invalid, &consent, &session, &report(),
            MOBILE_DEVICE_READ_ONLY_POLICY_ID, EffectCeiling::MOBILE_DEVICE_READ_ONLY, NOW,
        ).is_err());
        assert_eq!(guard.claimed_invocation_count(), 1);
    }
}
