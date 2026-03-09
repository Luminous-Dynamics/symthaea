//! Identity Vault Coordinator Zome
//!
//! CRUD operations for the agent's private identity data.
//! All entries are stored on the source chain (private by default).

use hdk::prelude::*;
use identity_vault_integrity::*;

/// Create or update the agent's profile.
///
/// Stores the profile on the source chain and creates a link from the
/// agent's pubkey for retrieval. If a profile already exists, it is updated.
#[hdk_extern]
pub fn set_profile(profile: Profile) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Profile(profile.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToProfile, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created profile".into()
    )))
}

/// Get the agent's current profile.
#[hdk_extern]
pub fn get_my_profile(_: ()) -> ExternResult<Option<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToProfile)?,
        GetStrategy::Local,
    )?;
    if let Some(link) = links.last() {
        let target = ActionHash::try_from(link.target.clone()).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Invalid link target: {:?}",
                e
            )))
        })?;
        get(target, GetOptions::default())
    } else {
        Ok(None)
    }
}

/// Register a master key for this agent.
#[hdk_extern]
pub fn register_key(key: MasterKey) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::MasterKey(key))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToKeys, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created key".into()
    )))
}

/// List all registered keys for this agent.
#[hdk_extern]
pub fn get_my_keys(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToKeys)?,
        GetStrategy::Local,
    )?;
    let mut records = Vec::new();
    for link in links {
        let target = ActionHash::try_from(link.target.clone()).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Invalid link target: {:?}",
                e
            )))
        })?;
        if let Some(record) = get(target, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Selective disclosure: return profile fields filtered by requested scope.
///
/// Used by personal_bridge to fulfill cross-cluster identity queries
/// without revealing the full profile.
#[hdk_extern]
pub fn disclose_profile(fields: Vec<String>) -> ExternResult<String> {
    let profile_record = get_my_profile(())?;
    let profile = match profile_record {
        Some(record) => {
            let p: Profile = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid profile entry".into()
                )))?;
            p
        }
        None => return Ok("{}".into()),
    };

    let mut disclosed = serde_json::Map::new();
    for field in &fields {
        match field.as_str() {
            "display_name" => {
                disclosed.insert(
                    "display_name".into(),
                    serde_json::Value::String(profile.display_name.clone()),
                );
            }
            "bio" => {
                if let Some(ref bio) = profile.bio {
                    disclosed.insert("bio".into(), serde_json::Value::String(bio.clone()));
                }
            }
            "avatar" => {
                if let Some(ref avatar) = profile.avatar {
                    disclosed.insert("avatar".into(), serde_json::Value::String(avatar.clone()));
                }
            }
            _ => {} // Unknown fields are silently ignored
        }
    }

    serde_json::to_string(&disclosed)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Serialization error: {}", e))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_entry_type_exists() {
        // Verify the entry type enum compiles and matches
        let _variant = UnitEntryTypes::Profile;
    }

    #[test]
    fn master_key_entry_type_exists() {
        let _variant = UnitEntryTypes::MasterKey;
    }

    #[test]
    fn link_types_exist() {
        let _profile = LinkTypes::AgentToProfile;
        let _keys = LinkTypes::AgentToKeys;
    }
}
