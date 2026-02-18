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

    // Call the personal cluster via OtherRole
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from(input.zome_name),
        FunctionName::from(input.fn_name),
        None,
        input.payload,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling personal cluster: {}", e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Personal cluster call failed: {:?}", other)
        ))),
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
///
/// Convenience wrapper around dispatch_personal_call for the common case
/// of requesting a Phi attestation for consciousness-gated governance.
#[hdk_extern]
pub fn request_phi_credential(_: ()) -> ExternResult<ExternIO> {
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from("personal_bridge"),
        FunctionName::from("present_phi_credential"),
        None,
        ExternIO::encode(())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling personal cluster: {}", e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Personal cluster call failed: {:?}", other)
        ))),
    }
}

/// Request K-vector trust credential from agent's personal vault
///
/// Convenience wrapper for requesting K-vector trust data for weighted voting.
#[hdk_extern]
pub fn request_k_vector(_: ()) -> ExternResult<ExternIO> {
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from("personal_bridge"),
        FunctionName::from("present_k_vector"),
        None,
        ExternIO::encode(())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling personal cluster: {}", e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Personal cluster call failed: {:?}", other)
        ))),
    }
}

/// Request identity proof from agent's personal vault
///
/// Convenience wrapper for requesting ZK identity proof without full profile disclosure.
#[hdk_extern]
pub fn request_identity_proof(_: ()) -> ExternResult<ExternIO> {
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from("personal_bridge"),
        FunctionName::from("present_identity_proof"),
        None,
        ExternIO::encode(())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling personal cluster: {}", e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Personal cluster call failed: {:?}", other)
        ))),
    }
}
