//! Credentials Coordinator Zome
//! Business logic for credential issuance, verification, and reference management.

use hdk::prelude::*;
use credentials_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Issue a new credential to a holder
#[hdk_extern]
pub fn issue_credential(credential: CareCredential) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::CareCredential(credential.clone()))?;

    // Link holder to credential
    let holder_anchor = ensure_anchor(&format!("agent_credentials:{}", credential.holder))?;
    create_link(holder_anchor, action_hash.clone(), LinkTypes::AgentToCredential, ())?;

    // Link credential type to credential
    let type_anchor = ensure_anchor(&format!("cred_type:{}", credential.credential_type.anchor_key()))?;
    create_link(type_anchor, action_hash.clone(), LinkTypes::TypeToCredential, ())?;

    // If verified, link to all verified credentials
    if credential.verified {
        let verified_anchor = ensure_anchor("all_verified_credentials")?;
        create_link(verified_anchor, action_hash.clone(), LinkTypes::AllVerifiedCredentials, ())?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created credential".into())))
}

/// Input for verifying a credential
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyCredentialInput {
    pub credential_hash: ActionHash,
}

/// Mark a credential as verified
#[hdk_extern]
pub fn verify_credential(input: VerifyCredentialInput) -> ExternResult<Record> {
    let record = get(input.credential_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credential not found".into())))?;

    let mut credential: CareCredential = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid credential entry".into())))?;

    if credential.verified {
        return Err(wasm_error!(WasmErrorInner::Guest("Credential is already verified".into())));
    }

    credential.verified = true;

    let updated_hash = update_entry(input.credential_hash, &EntryTypes::CareCredential(credential.clone()))?;

    // Link to verified credentials
    let verified_anchor = ensure_anchor("all_verified_credentials")?;
    create_link(verified_anchor, updated_hash.clone(), LinkTypes::AllVerifiedCredentials, ())?;

    get(updated_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated credential".into())))
}

/// Add a reference for a care provider
#[hdk_extern]
pub fn add_reference(reference: CareReference) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is the one giving the reference
    if caller != reference.from_recipient {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only add references from your own agent".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::CareReference(reference.clone()))?;

    // Link provider to received reference
    let provider_anchor = ensure_anchor(&format!("agent_references:{}", reference.provider))?;
    create_link(provider_anchor, action_hash.clone(), LinkTypes::AgentToReference, ())?;

    // Link recipient to given reference
    let giver_anchor = ensure_anchor(&format!("agent_given_refs:{}", reference.from_recipient))?;
    create_link(giver_anchor, action_hash.clone(), LinkTypes::AgentGivenReferences, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created reference".into())))
}

/// Get all credentials for a provider
#[hdk_extern]
pub fn get_provider_credentials(provider: AgentPubKey) -> ExternResult<Vec<Record>> {
    let holder_anchor = anchor_hash(&format!("agent_credentials:{}", provider))?;
    let links = get_links(
        LinkQuery::try_new(holder_anchor, LinkTypes::AgentToCredential)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all references for a provider
#[hdk_extern]
pub fn get_provider_references(provider: AgentPubKey) -> ExternResult<Vec<Record>> {
    let provider_anchor = anchor_hash(&format!("agent_references:{}", provider))?;
    let links = get_links(
        LinkQuery::try_new(provider_anchor, LinkTypes::AgentToReference)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get my credentials
#[hdk_extern]
pub fn get_my_credentials(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    get_provider_credentials(caller)
}

/// Get my references (received)
#[hdk_extern]
pub fn get_my_references(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    get_provider_references(caller)
}

/// Get credentials by type
#[hdk_extern]
pub fn get_credentials_by_type(credential_type: CredentialType) -> ExternResult<Vec<Record>> {
    let type_anchor = anchor_hash(&format!("cred_type:{}", credential_type.anchor_key()))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::TypeToCredential)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Provider reputation summary
#[derive(Serialize, Deserialize, Debug)]
pub struct ProviderReputation {
    pub provider: AgentPubKey,
    pub credential_count: u32,
    pub verified_credential_count: u32,
    pub reference_count: u32,
    pub average_rating: f32,
}

/// Get a provider's reputation summary
#[hdk_extern]
pub fn get_provider_reputation(provider: AgentPubKey) -> ExternResult<ProviderReputation> {
    let credentials = get_provider_credentials(provider.clone())?;
    let references = get_provider_references(provider.clone())?;

    let mut verified_count = 0u32;
    for record in &credentials {
        if let Some(cred) = record
            .entry()
            .to_app_option::<CareCredential>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if cred.verified {
                verified_count += 1;
            }
        }
    }

    let mut total_rating = 0u32;
    let mut rating_count = 0u32;
    for record in &references {
        if let Some(reference) = record
            .entry()
            .to_app_option::<CareReference>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            total_rating += reference.rating as u32;
            rating_count += 1;
        }
    }

    let average_rating = if rating_count > 0 {
        total_rating as f32 / rating_count as f32
    } else {
        0.0
    };

    Ok(ProviderReputation {
        provider,
        credential_count: credentials.len() as u32,
        verified_credential_count: verified_count,
        reference_count: references.len() as u32,
        average_rating,
    })
}
