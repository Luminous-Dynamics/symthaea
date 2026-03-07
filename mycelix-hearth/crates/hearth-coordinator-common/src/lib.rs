use hdk::prelude::*;
use hearth_types::MemberRole;

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

/// Follow the update chain from an action hash to get the latest version.
/// Returns the record at the tip of the chain, or the original if no updates exist.
pub fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                // Follow the most recent update (last in the list)
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                // Recursively follow in case there are chained updates
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

/// Resolve a list of links into their corresponding records,
/// following update chains to return the latest version of each entry.
/// Skips any targets whose records have been deleted.
pub fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
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
