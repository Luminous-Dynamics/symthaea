//! Shared utilities for governance coordinator zomes
//!
//! Provides standardized cross-zome call handling with consistent error messages.

use hdk::prelude::*;

/// Make a cross-zome call to a local zome and unwrap the response.
///
/// Handles `ZomeCallResponse` matching with consistent error messages:
/// - `Ok` → returns the `ExternIO` payload
/// - `NetworkError` → returns a guest error with zome/function context
/// - Any other variant → returns a descriptive guest error
///
/// # Example
///
/// ```ignore
/// let io = call_local("proposals", "get_proposal", proposal_id)?;
/// let record: Option<Record> = io.decode()
///     .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
/// ```
pub fn call_local(
    zome: &str,
    fn_name: &str,
    payload: impl serde::Serialize + std::fmt::Debug,
) -> ExternResult<ExternIO> {
    let encoded = ExternIO::encode(payload)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Failed to encode payload for {}::{}: {}", zome, fn_name, e)
        )))?;

    match call(
        CallTargetCell::Local,
        ZomeName::from(zome),
        FunctionName::from(fn_name),
        None,
        encoded,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling {}::{}: {}", zome, fn_name, e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Unexpected response from {}::{}: {:?}", zome, fn_name, other)
        ))),
    }
}

/// Make a cross-zome call to a zome in another role (cross-cluster/cross-DNA).
///
/// Same error handling as `call_local` but targets `CallTargetCell::OtherRole`.
pub fn call_role(
    role: &str,
    zome: &str,
    fn_name: &str,
    payload: impl serde::Serialize + std::fmt::Debug,
) -> ExternResult<ExternIO> {
    let encoded = ExternIO::encode(payload)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Failed to encode payload for {}::{}::{}: {}", role, zome, fn_name, e)
        )))?;

    match call(
        CallTargetCell::OtherRole(role.into()),
        ZomeName::from(zome),
        FunctionName::from(fn_name),
        None,
        encoded,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling {}::{}::{}: {}", role, zome, fn_name, e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Unexpected response from {}::{}::{}: {:?}", role, zome, fn_name, other)
        ))),
    }
}

/// Make a best-effort cross-zome call that logs failures but doesn't propagate errors.
///
/// Useful for non-critical operations like signal syncing, audit logging, etc.
/// Returns `Ok(Some(io))` on success, `Ok(None)` on any failure.
pub fn call_local_best_effort(
    zome: &str,
    fn_name: &str,
    payload: impl serde::Serialize + std::fmt::Debug,
) -> ExternResult<Option<ExternIO>> {
    match call_local(zome, fn_name, payload) {
        Ok(io) => Ok(Some(io)),
        Err(e) => {
            debug!("Best-effort call to {}::{} failed (non-fatal): {:?}", zome, fn_name, e);
            Ok(None)
        }
    }
}

/// Make a best-effort cross-role call that logs failures but doesn't propagate errors.
///
/// Same as `call_local_best_effort` but targets `CallTargetCell::OtherRole`.
pub fn call_role_best_effort(
    role: &str,
    zome: &str,
    fn_name: &str,
    payload: impl serde::Serialize + std::fmt::Debug,
) -> ExternResult<Option<ExternIO>> {
    match call_role(role, zome, fn_name, payload) {
        Ok(io) => Ok(Some(io)),
        Err(e) => {
            debug!("Best-effort call to {}::{}::{} failed (non-fatal): {:?}", role, zome, fn_name, e);
            Ok(None)
        }
    }
}
