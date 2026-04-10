// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;
use hdk::prelude::{hdk_extern, wasm_error, WasmErrorInner};
use serde::{Deserialize, Serialize};

// =============================================================================
// CROSS-CLUSTER DISPATCH TO PERSONAL
// =============================================================================

/// Allowed zomes in the personal cluster that governance can call
const ALLOWED_PERSONAL_ZOMES: &[&str] = &["personal_bridge"];

/// Dispatch a call to the personal cluster via OtherRole
///
/// Used by governance to request credential presentations (Phi, K-vector,
/// identity proofs) from the agent's personal vault via the personal bridge.
///
/// Note: This function takes a pre-encoded ExternIO payload, so it cannot use
/// governance_utils::call_role (which encodes internally). Kept as manual match.
#[hdk_extern]
pub fn dispatch_personal_call(input: DispatchPersonalCallInput) -> ExternResult<ExternIO> {
    // Validate zome is in allowlist
    if !ALLOWED_PERSONAL_ZOMES.contains(&input.zome_name.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Zome '{}' not in ALLOWED_PERSONAL_ZOMES",
            input.zome_name
        ))));
    }

    // Validate function name length
    if input.fn_name.is_empty() || input.fn_name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Function name must be 1-256 characters".into()
        )));
    }

    // Call the personal cluster via OtherRole (pre-encoded payload)
    // Circuit breaker: catch transport errors to return clear "cluster unavailable" messages
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from(input.zome_name.clone()),
        FunctionName::from(input.fn_name.clone()),
        None,
        input.payload,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => Ok(io),
        Ok(ZomeCallResponse::NetworkError(e)) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: personal cluster unavailable (network error calling {}::{}), operation suspended: {}",
            input.zome_name, input.fn_name, e
        )))),
        Ok(other) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: personal cluster returned unexpected response from {}::{}, operation suspended: {:?}",
            input.zome_name, input.fn_name, other
        )))),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: personal cluster unreachable (transport error calling {}::{}), operation suspended: {:?}",
            input.zome_name, input.fn_name, e
        )))),
    }
}

/// Input for dispatching a call to the personal cluster
#[derive(Serialize, Deserialize, Debug)]
pub struct DispatchPersonalCallInput {
    pub zome_name: String,
    pub fn_name: String,
    pub payload: ExternIO,
}

/// Request Phi credential from agent's personal vault
#[hdk_extern]
pub fn request_phi_credential(_: ()) -> ExternResult<ExternIO> {
    governance_utils::call_role("personal", "personal_bridge", "present_phi_credential", ())
}

/// Request K-vector trust credential from agent's personal vault
#[hdk_extern]
pub fn request_k_vector(_: ()) -> ExternResult<ExternIO> {
    governance_utils::call_role("personal", "personal_bridge", "present_k_vector", ())
}

/// Request identity proof from agent's personal vault
#[hdk_extern]
pub fn request_identity_proof(_: ()) -> ExternResult<ExternIO> {
    governance_utils::call_role("personal", "personal_bridge", "present_identity_proof", ())
}

// =============================================================================
// CROSS-CLUSTER DISPATCH TO IDENTITY
// =============================================================================

/// Allowed zomes in the identity cluster that governance can call
const ALLOWED_IDENTITY_ZOMES: &[&str] =
    &["identity_bridge", "did_registry", "verifiable_credential"];

/// Dispatch a call to the identity cluster via OtherRole
#[hdk_extern]
pub fn dispatch_identity_call(input: DispatchIdentityCallInput) -> ExternResult<ExternIO> {
    if !ALLOWED_IDENTITY_ZOMES.contains(&input.zome_name.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Zome '{}' not in ALLOWED_IDENTITY_ZOMES",
            input.zome_name
        ))));
    }

    if input.fn_name.is_empty() || input.fn_name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Function name must be 1-256 characters".into()
        )));
    }

    // Circuit breaker: catch transport errors to return clear "cluster unavailable" messages
    match call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::from(input.zome_name.clone()),
        FunctionName::from(input.fn_name.clone()),
        None,
        input.payload,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => Ok(io),
        Ok(ZomeCallResponse::NetworkError(e)) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: identity cluster unavailable (network error calling {}::{}), operation suspended: {}",
            input.zome_name, input.fn_name, e
        )))),
        Ok(other) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: identity cluster returned unexpected response from {}::{}, operation suspended: {:?}",
            input.zome_name, input.fn_name, other
        )))),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: identity cluster unreachable (transport error calling {}::{}), operation suspended: {:?}",
            input.zome_name, input.fn_name, e
        )))),
    }
}

/// Input for dispatching a call to the identity cluster
#[derive(Serialize, Deserialize, Debug)]
pub struct DispatchIdentityCallInput {
    pub zome_name: String,
    pub fn_name: String,
    pub payload: ExternIO,
}

/// Verify a voter's DID is active in the identity cluster.
///
/// Used by governance Phi gate to ensure the voter has a valid, non-deactivated
/// DID before allowing them to participate in governance actions.
#[hdk_extern]
pub fn verify_voter_did(did: String) -> ExternResult<ExternIO> {
    governance_utils::call_role("identity", "did_registry", "is_did_active", did)
}

/// Get a voter's MATL trust score from the identity bridge.
///
/// Returns the composite trust score (0.0-1.0) combining MFA assurance
/// and reputation. Can be used to weight votes or gate participation.
#[hdk_extern]
pub fn get_voter_matl_score(did: String) -> ExternResult<ExternIO> {
    governance_utils::call_role("identity", "identity_bridge", "get_matl_score", did)
}

/// Verify a credential via the identity cluster's verifiable_credential zome.
///
/// Used to validate that a credential presented during governance actions
/// (e.g., expertise claims for council membership) is genuine and not revoked.
#[hdk_extern]
pub fn verify_governance_credential(credential_id: String) -> ExternResult<ExternIO> {
    governance_utils::call_role(
        "identity",
        "verifiable_credential",
        "verify_credential",
        credential_id,
    )
}

/// Check enhanced trust for a DID — combines reputation + MFA requirements.
///
/// Used for high-stakes governance actions (treasury operations, key rotation)
/// that require stronger identity verification.
#[hdk_extern]
pub fn check_voter_trust(input: CheckVoterTrustInput) -> ExternResult<ExternIO> {
    governance_utils::call_role("identity", "identity_bridge", "check_enhanced_trust", input)
}

/// Input for enhanced trust verification of a governance participant
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckVoterTrustInput {
    pub did: String,
    pub min_reputation: f64,
    pub require_mfa: bool,
    pub min_assurance_level: Option<u8>,
}

// =============================================================================
// CROSS-CLUSTER DISPATCH TO COMMONS
// =============================================================================

/// Allowed zomes in the commons cluster that governance can call.
///
/// Used for property status checks before proposals, housing capacity
/// impact assessments, and water stewardship policy verification.
const ALLOWED_COMMONS_ZOMES: &[&str] = &[
    "commons_bridge",
    "property_registry",
    "housing_governance",
    "water_steward",
];

/// Dispatch a call to the commons cluster via OtherRole.
#[hdk_extern]
pub fn dispatch_commons_call(input: DispatchCommonsCallInput) -> ExternResult<ExternIO> {
    if !ALLOWED_COMMONS_ZOMES.contains(&input.zome_name.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Zome '{}' not in ALLOWED_COMMONS_ZOMES",
            input.zome_name
        ))));
    }

    if input.fn_name.is_empty() || input.fn_name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Function name must be 1-256 characters".into()
        )));
    }

    match call(
        CallTargetCell::OtherRole("commons".into()),
        ZomeName::from(input.zome_name.clone()),
        FunctionName::from(input.fn_name.clone()),
        None,
        input.payload,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => Ok(io),
        Ok(ZomeCallResponse::NetworkError(e)) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: commons cluster unavailable (network error calling {}::{}): {}",
            input.zome_name, input.fn_name, e
        )))),
        Ok(other) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: commons cluster unexpected response from {}::{}: {:?}",
            input.zome_name, input.fn_name, other
        )))),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: commons cluster unreachable (transport error calling {}::{}): {:?}",
            input.zome_name, input.fn_name, e
        )))),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DispatchCommonsCallInput {
    pub zome_name: String,
    pub fn_name: String,
    pub payload: ExternIO,
}

/// Check property status before proposals affecting land use.
///
/// Returns the property record if found, allowing governance to verify
/// ownership and enforcement status before accepting land-related proposals.
#[hdk_extern]
pub fn check_property_status(property_id: String) -> ExternResult<ExternIO> {
    governance_utils::call_role("commons", "property_registry", "get_property", property_id)
}

/// Query housing capacity in an area for impact assessments.
///
/// Used when evaluating proposals that affect housing — governance can
/// check current capacity before approving development or zoning changes.
#[hdk_extern]
pub fn query_housing_capacity(area: String) -> ExternResult<ExternIO> {
    governance_utils::call_role("commons", "housing_governance", "get_capacity_summary", area)
}

// =============================================================================
// CROSS-CLUSTER DISPATCH TO CIVIC
// =============================================================================

/// Allowed zomes in the civic cluster that governance can call.
///
/// Used for checking active justice disputes and emergency status
/// before treasury operations or policy proposals.
const ALLOWED_CIVIC_ZOMES: &[&str] = &[
    "civic_bridge",
    "justice_cases",
    "emergency_coordination",
];

/// Dispatch a call to the civic cluster via OtherRole.
#[hdk_extern]
pub fn dispatch_civic_call(input: DispatchCivicCallInput) -> ExternResult<ExternIO> {
    if !ALLOWED_CIVIC_ZOMES.contains(&input.zome_name.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Zome '{}' not in ALLOWED_CIVIC_ZOMES",
            input.zome_name
        ))));
    }

    if input.fn_name.is_empty() || input.fn_name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Function name must be 1-256 characters".into()
        )));
    }

    match call(
        CallTargetCell::OtherRole("civic".into()),
        ZomeName::from(input.zome_name.clone()),
        FunctionName::from(input.fn_name.clone()),
        None,
        input.payload,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => Ok(io),
        Ok(ZomeCallResponse::NetworkError(e)) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: civic cluster unavailable (network error calling {}::{}): {}",
            input.zome_name, input.fn_name, e
        )))),
        Ok(other) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: civic cluster unexpected response from {}::{}: {:?}",
            input.zome_name, input.fn_name, other
        )))),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: civic cluster unreachable (transport error calling {}::{}): {:?}",
            input.zome_name, input.fn_name, e
        )))),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DispatchCivicCallInput {
    pub zome_name: String,
    pub fn_name: String,
    pub payload: ExternIO,
}

/// Check for active justice cases that might affect a governance action.
///
/// Used before executing proposals that affect property or individuals
/// under active legal proceedings.
#[hdk_extern]
pub fn check_active_justice_cases(query: String) -> ExternResult<ExternIO> {
    governance_utils::call_role("civic", "justice_cases", "get_active_cases", query)
}

/// Check emergency status in an area before governance operations.
///
/// During active emergencies, certain governance operations may be
/// fast-tracked or suspended depending on the emergency type.
#[hdk_extern]
pub fn check_emergency_status(area: String) -> ExternResult<ExternIO> {
    governance_utils::call_role(
        "civic",
        "emergency_coordination",
        "get_active_emergencies_in_area",
        area,
    )
}

// =============================================================================
// CROSS-CLUSTER DISPATCH TO FINANCE
// =============================================================================

/// Allowed zomes in the finance cluster that governance can call.
///
/// Used for treasury operations approved by governance (budget proposals),
/// payment settlement queries, and staking requirement checks.
const ALLOWED_FINANCE_ZOMES: &[&str] = &[
    "finance_bridge",
    "treasury",
    "payments",
];

/// Dispatch a call to the finance cluster via OtherRole.
#[hdk_extern]
pub fn dispatch_finance_call(input: DispatchFinanceCallInput) -> ExternResult<ExternIO> {
    if !ALLOWED_FINANCE_ZOMES.contains(&input.zome_name.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Zome '{}' not in ALLOWED_FINANCE_ZOMES",
            input.zome_name
        ))));
    }

    if input.fn_name.is_empty() || input.fn_name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Function name must be 1-256 characters".into()
        )));
    }

    match call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::from(input.zome_name.clone()),
        FunctionName::from(input.fn_name.clone()),
        None,
        input.payload,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => Ok(io),
        Ok(ZomeCallResponse::NetworkError(e)) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: finance cluster unavailable (network error calling {}::{}): {}",
            input.zome_name, input.fn_name, e
        )))),
        Ok(other) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: finance cluster unexpected response from {}::{}: {:?}",
            input.zome_name, input.fn_name, other
        )))),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: finance cluster unreachable (transport error calling {}::{}): {:?}",
            input.zome_name, input.fn_name, e
        )))),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DispatchFinanceCallInput {
    pub zome_name: String,
    pub fn_name: String,
    pub payload: ExternIO,
}

/// Query current treasury balance for budget proposal validation.
///
/// Governance proposals requesting budget allocation should verify
/// sufficient funds exist before submitting.
#[hdk_extern]
pub fn check_treasury_balance(_: ()) -> ExternResult<ExternIO> {
    governance_utils::call_role("finance", "treasury", "get_current_balance", ())
}

/// Execute a treasury transfer approved by governance.
///
/// Called after a budget proposal passes voting. The transfer ID should
/// reference the approved proposal hash for auditability.
#[hdk_extern]
pub fn execute_approved_transfer(input: ApprovedTransferInput) -> ExternResult<ExternIO> {
    governance_utils::call_role("finance", "treasury", "execute_governance_transfer", input)
}

/// Input for governance-approved treasury transfers
#[derive(Serialize, Deserialize, Debug)]
pub struct ApprovedTransferInput {
    /// Hash of the approved governance proposal authorizing this transfer
    pub proposal_hash: String,
    /// Recipient DID
    pub recipient_did: String,
    /// Amount in SAP (smallest unit)
    pub amount_sap: u64,
    /// Purpose description for audit trail
    pub purpose: String,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn personal_allowlist_contains_bridge() {
        assert!(ALLOWED_PERSONAL_ZOMES.contains(&"personal_bridge"));
    }

    #[test]
    fn identity_allowlist_contains_required_zomes() {
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"identity_bridge"));
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"did_registry"));
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"verifiable_credential"));
    }

    #[test]
    fn commons_allowlist_contains_required_zomes() {
        assert!(ALLOWED_COMMONS_ZOMES.contains(&"commons_bridge"));
        assert!(ALLOWED_COMMONS_ZOMES.contains(&"property_registry"));
        assert!(ALLOWED_COMMONS_ZOMES.contains(&"housing_governance"));
        assert!(ALLOWED_COMMONS_ZOMES.contains(&"water_steward"));
    }

    #[test]
    fn civic_allowlist_contains_required_zomes() {
        assert!(ALLOWED_CIVIC_ZOMES.contains(&"civic_bridge"));
        assert!(ALLOWED_CIVIC_ZOMES.contains(&"justice_cases"));
        assert!(ALLOWED_CIVIC_ZOMES.contains(&"emergency_coordination"));
    }

    #[test]
    fn finance_allowlist_contains_required_zomes() {
        assert!(ALLOWED_FINANCE_ZOMES.contains(&"finance_bridge"));
        assert!(ALLOWED_FINANCE_ZOMES.contains(&"treasury"));
        assert!(ALLOWED_FINANCE_ZOMES.contains(&"payments"));
    }

    #[test]
    fn unknown_zome_not_in_any_allowlist() {
        assert!(!ALLOWED_PERSONAL_ZOMES.contains(&"rogue_zome"));
        assert!(!ALLOWED_IDENTITY_ZOMES.contains(&"rogue_zome"));
        assert!(!ALLOWED_COMMONS_ZOMES.contains(&"rogue_zome"));
        assert!(!ALLOWED_CIVIC_ZOMES.contains(&"rogue_zome"));
        assert!(!ALLOWED_FINANCE_ZOMES.contains(&"rogue_zome"));
    }

    #[test]
    fn all_five_clusters_have_dispatch() {
        // Verify we cover all 5 declared governance outbound routes
        assert_eq!(ALLOWED_PERSONAL_ZOMES.len(), 1);
        assert_eq!(ALLOWED_IDENTITY_ZOMES.len(), 3);
        assert_eq!(ALLOWED_COMMONS_ZOMES.len(), 4);
        assert_eq!(ALLOWED_CIVIC_ZOMES.len(), 3);
        assert_eq!(ALLOWED_FINANCE_ZOMES.len(), 3);
    }

    #[test]
    fn approved_transfer_input_serde_roundtrip() {
        let input = ApprovedTransferInput {
            proposal_hash: "QmABC123".to_string(),
            recipient_did: "did:mycelix:test".to_string(),
            amount_sap: 10_000,
            purpose: "Community garden funding".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: ApprovedTransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.amount_sap, 10_000);
        assert_eq!(back.proposal_hash, "QmABC123");
    }

    #[test]
    fn check_voter_trust_input_serde_roundtrip() {
        let input = CheckVoterTrustInput {
            did: "did:mycelix:voter".to_string(),
            min_reputation: 0.5,
            require_mfa: true,
            min_assurance_level: Some(2),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CheckVoterTrustInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.did, "did:mycelix:voter");
        assert!(back.require_mfa);
    }
}
