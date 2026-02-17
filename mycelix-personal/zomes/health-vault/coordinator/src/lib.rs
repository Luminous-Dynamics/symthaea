//! Health Vault Coordinator Zome
//!
//! CRUD operations for the agent's private health data.
//! Access is gated by consent grants.

use hdk::prelude::*;
use health_vault_integrity::*;

/// Create a health record on the agent's source chain.
#[hdk_extern]
pub fn create_health_record(record: HealthRecord) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::HealthRecord(record.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent.clone(), action_hash.clone(), LinkTypes::AgentToRecords, ())?;

    let type_anchor_hash = hash_entry(&EntryTypes::HealthRecord(HealthRecord {
        record_type: record.record_type.clone(),
        data: String::new(),
        source: String::new(),
        event_date: Timestamp::from_micros(0),
        updated_at: Timestamp::from_micros(0),
    }))?;
    let _ = create_link(agent, type_anchor_hash, LinkTypes::RecordTypeToRecord, record.record_type.as_bytes().to_vec());

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created health record".into()
    )))
}

/// Store a biometric measurement.
#[hdk_extern]
pub fn record_biometric(biometric: Biometric) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Biometric(biometric))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToBiometrics, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created biometric".into()
    )))
}

/// Grant consent for a specific agent to read health records.
#[hdk_extern]
pub fn grant_consent(consent: ConsentGrant) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::ConsentGrant(consent))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToConsents, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created consent".into()
    )))
}

/// Get all health records for this agent.
#[hdk_extern]
pub fn get_my_records(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToRecords)?,
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

/// Get all biometric measurements for this agent.
#[hdk_extern]
pub fn get_my_biometrics(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToBiometrics)?,
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

/// Get all active consent grants for this agent.
#[hdk_extern]
pub fn get_my_consents(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToConsents)?,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_types_exist() {
        let _record = UnitEntryTypes::HealthRecord;
        let _bio = UnitEntryTypes::Biometric;
        let _consent = UnitEntryTypes::ConsentGrant;
    }

    #[test]
    fn link_types_exist() {
        let _records = LinkTypes::AgentToRecords;
        let _bio = LinkTypes::AgentToBiometrics;
        let _consents = LinkTypes::AgentToConsents;
        let _type_to_record = LinkTypes::RecordTypeToRecord;
    }
}
