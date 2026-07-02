// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    // Pass payload directly to call() — it handles serialization internally.
    // Do NOT pre-encode with ExternIO::encode(), which causes double-encoding
    // (call() would serialize the Vec<u8> as msgpack bin8, wrapping the already-encoded bytes).
    match call(
        CallTargetCell::Local,
        ZomeName::from(zome),
        FunctionName::from(fn_name),
        None,
        payload,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling {}::{}: {}",
            zome, fn_name, e
        )))),
        other => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Unexpected response from {}::{}: {:?}",
            zome, fn_name, other
        )))),
    }
}

/// Make a cross-zome call to a zome in another role (cross-cluster/cross-DNA).
///
/// Implements a circuit breaker pattern: if the target cluster is unreachable
/// (transport error or network error), returns a descriptive "cluster unavailable"
/// error instead of propagating raw Holochain internals. This ensures callers
/// get a clear, consistent signal that the operation is suspended due to
/// cluster unavailability.
pub fn call_role(
    role: &str,
    zome: &str,
    fn_name: &str,
    payload: impl serde::Serialize + std::fmt::Debug,
) -> ExternResult<ExternIO> {
    // Pass payload directly to call() — it handles serialization internally.
    // Do NOT pre-encode with ExternIO::encode(), which causes double-encoding
    // (call() would serialize the Vec<u8> as msgpack bin8, wrapping the already-encoded bytes).
    match call(
        CallTargetCell::OtherRole(role.into()),
        ZomeName::from(zome),
        FunctionName::from(fn_name),
        None,
        payload,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => Ok(io),
        Ok(ZomeCallResponse::NetworkError(e)) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: {} cluster unavailable (network error calling {}::{}), operation suspended: {}",
            role, zome, fn_name, e
        )))),
        Ok(other) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: {} cluster returned unexpected response from {}::{}, operation suspended: {:?}",
            role, zome, fn_name, other
        )))),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Circuit breaker: {} cluster unreachable (transport error calling {}::{}), operation suspended: {:?}",
            role, zome, fn_name, e
        )))),
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
            debug!(
                "Best-effort call to {}::{} failed (non-fatal): {:?}",
                zome, fn_name, e
            );
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
            debug!(
                "Best-effort call to {}::{}::{} failed (non-fatal): {:?}",
                role, zome, fn_name, e
            );
            Ok(None)
        }
    }
}
