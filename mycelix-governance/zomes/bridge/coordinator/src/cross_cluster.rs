use super::*;

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
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from(input.zome_name),
        FunctionName::from(input.fn_name),
        None,
        input.payload,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling personal cluster: {}",
            e
        )))),
        other => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Personal cluster call failed: {:?}",
            other
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

    match call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::from(input.zome_name),
        FunctionName::from(input.fn_name),
        None,
        input.payload,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling identity cluster: {}",
            e
        )))),
        other => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Identity cluster call failed: {:?}",
            other
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
