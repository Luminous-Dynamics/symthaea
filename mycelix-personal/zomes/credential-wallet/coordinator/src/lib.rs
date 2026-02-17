//! Credential Wallet Coordinator Zome
//!
//! Store, present, and verify verifiable credentials.
//! Credentials are stored privately; presentation is done via personal_bridge.

use hdk::prelude::*;
use credential_wallet_integrity::*;
use personal_types::CredentialType;

/// Store a new credential in the wallet.
#[hdk_extern]
pub fn store_credential(credential: StoredCredential) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::StoredCredential(credential.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToCredentials, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve stored credential".into()
    )))
}

/// Create a proof from a stored credential.
#[hdk_extern]
pub fn create_proof(proof: CredentialProof) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::CredentialProof(proof))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToProofs, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created proof".into()
    )))
}

/// Get all credentials stored by this agent.
#[hdk_extern]
pub fn get_my_credentials(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToCredentials)?,
        GetStrategy::Local,
    )?;
    let mut records = Vec::new();
    for link in links {
        let target = ActionHash::try_from(link.target.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Invalid link target: {:?}", e))))?;
        if let Some(record) = get(target, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Get credentials of a specific type.
#[hdk_extern]
pub fn get_credentials_by_type(credential_type: CredentialType) -> ExternResult<Vec<StoredCredential>> {
    let all = get_my_credentials(())?;
    let mut matched = Vec::new();
    for record in all {
        if let Some(cred) = record
            .entry()
            .to_app_option::<StoredCredential>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if cred.credential_type == credential_type && !cred.revoked {
                matched.push(cred);
            }
        }
    }
    Ok(matched)
}

/// Present a credential for cross-cluster verification.
///
/// Returns the credential data as a JSON string. The personal_bridge
/// wraps this into a CredentialPresentation with appropriate scope filtering.
#[hdk_extern]
pub fn present_credential(credential_type: CredentialType) -> ExternResult<String> {
    let creds = get_credentials_by_type(credential_type.clone())?;
    match creds.first() {
        Some(cred) => Ok(cred.credential_data.clone()),
        None => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "No active credential of type {:?} found",
            credential_type
        )))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_types_exist() {
        let _cred = UnitEntryTypes::StoredCredential;
        let _proof = UnitEntryTypes::CredentialProof;
    }

    #[test]
    fn link_types_exist() {
        let _creds = LinkTypes::AgentToCredentials;
        let _proofs = LinkTypes::AgentToProofs;
        let _type_to_cred = LinkTypes::CredentialTypeToCredential;
    }
}
