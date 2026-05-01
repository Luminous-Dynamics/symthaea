// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Health Vault Coordinator Zome
//!
//! CRUD operations for the agent's private health data.
//! Access is gated by consent grants.

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use hdk::prelude::*;
use health_vault_integrity::*;

getrandom::register_custom_getrandom!(my_custom_getrandom);

pub fn my_custom_getrandom(buf: &mut [u8]) -> Result<(), getrandom::Error> {
    let bytes = random_bytes(buf.len() as u32).map_err(|_| getrandom::Error::UNSUPPORTED)?;
    buf.copy_from_slice(bytes.as_ref());
    Ok(())
}

use personal_leptos_types::{
    BiometricView, ConsentGrantInputView, ConsentGrantView, HealthRecordView,
};

fn decode_agent_pubkey(pubkey: &str) -> ExternResult<AgentPubKey> {
    let bytes = BASE64.decode(pubkey).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid agent pubkey encoding: {e}"
        )))
    })?;
    if bytes.len() != 39 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Agent pubkey wrong length: {} bytes (expected 39)",
            bytes.len()
        ))));
    }
    Ok(AgentPubKey::from_raw_39(bytes))
}

/// Create a health record on the agent's source chain.
#[hdk_extern]
pub fn create_health_record(record: HealthRecord) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::HealthRecord(record.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(
        agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToRecords,
        (),
    )?;

    let type_anchor_hash = hash_entry(&EntryTypes::HealthRecord(HealthRecord {
        record_type: record.record_type.clone(),
        data: String::new(),
        source: String::new(),
        event_date: Timestamp::from_micros(0),
        updated_at: Timestamp::from_micros(0),
    }))?;
    let _ = create_link(
        agent,
        type_anchor_hash,
        LinkTypes::RecordTypeToRecord,
        record.record_type.as_bytes().to_vec(),
    );

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

#[hdk_extern]
pub fn grant_consent_view(input: ConsentGrantInputView) -> ExternResult<Record> {
    let grantee = decode_agent_pubkey(&input.grantee)?;
    grant_consent(ConsentGrant {
        grantee,
        record_types: input.record_types,
        expires_at: input.expires_at.map(Timestamp::from_micros),
        active: input.active,
        created_at: sys_time()?,
    })
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

#[hdk_extern]
pub fn get_my_records_view(_: ()) -> ExternResult<Vec<HealthRecordView>> {
    let records = get_my_records(())?;
    let mut views = Vec::new();
    for record in records {
        if let Some(entry) = record
            .entry()
            .to_app_option::<HealthRecord>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            views.push(HealthRecordView {
                hash: record.action_address().to_string(),
                record_type: entry.record_type,
                data: entry.data,
                source: entry.source,
                event_date: entry.event_date.as_micros(),
                updated_at: entry.updated_at.as_micros(),
            });
        }
    }
    Ok(views)
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

#[hdk_extern]
pub fn get_my_biometrics_view(_: ()) -> ExternResult<Vec<BiometricView>> {
    let records = get_my_biometrics(())?;
    let mut views = Vec::new();
    for record in records {
        if let Some(entry) = record
            .entry()
            .to_app_option::<Biometric>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            views.push(BiometricView {
                hash: record.action_address().to_string(),
                metric_type: entry.metric_type,
                value: entry.value,
                unit: entry.unit,
                measured_at: entry.measured_at.as_micros(),
            });
        }
    }
    Ok(views)
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

#[hdk_extern]
pub fn get_my_consents_view(_: ()) -> ExternResult<Vec<ConsentGrantView>> {
    let records = get_my_consents(())?;
    let mut views = Vec::new();
    for record in records {
        if let Some(entry) = record
            .entry()
            .to_app_option::<ConsentGrant>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            views.push(ConsentGrantView {
                hash: record.action_address().to_string(),
                grantee: entry.grantee.to_string(),
                record_types: entry.record_types,
                expires_at: entry.expires_at.map(|ts| ts.as_micros()),
                active: entry.active,
                created_at: entry.created_at.as_micros(),
            });
        }
    }
    Ok(views)
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
