// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use hearth_types::MemberRole;

// Re-export from mycelix-zome-helpers so existing consumers keep compiling.
pub use mycelix_zome_helpers::{get_latest_record, records_from_links};

/// Decode a typed value from a ZomeCallResponse, providing context for error messages.
pub fn decode_zome_response<T: serde::de::DeserializeOwned + std::fmt::Debug>(
    response: ZomeCallResponse,
    context: &str,
) -> ExternResult<T> {
    match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode {} response: {}",
                context, e
            )))
        }),
        other => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cross-zome call to {} failed: {:?}",
            context, other
        )))),
    }
}

/// Verify the caller is an active member of the given hearth.
/// Returns the caller's role on success, or an error if not a member.
pub fn require_membership(hearth_hash: &ActionHash) -> ExternResult<MemberRole> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("hearth_kinship"),
        FunctionName::new("get_caller_role"),
        None,
        hearth_hash,
    )?;
    let role: Option<MemberRole> = decode_zome_response(response, "get_caller_role")?;
    role.ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(
            "You are not an active member of this hearth".into()
        ))
    })
}

/// Verify the caller is a guardian (Founder, Elder, or Adult) of the given hearth.
/// Returns the caller's role on success, or an error if not a guardian.
pub fn require_guardian(hearth_hash: &ActionHash) -> ExternResult<MemberRole> {
    let role = require_membership(hearth_hash)?;
    if !role.is_guardian() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only guardians (Founder, Elder, or Adult) can perform this action".into()
        )));
    }
    Ok(role)
}
