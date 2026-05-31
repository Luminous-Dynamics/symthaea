// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Purity Coordinator Zome
//! Business logic for water quality monitoring, alerts, and remediation

use hdk::prelude::*;
use purity_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
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

// ============================================================================
// QUALITY READINGS
// ============================================================================

/// Submit a new water quality reading
#[hdk_extern]
pub fn submit_reading(reading: QualityReading) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::QualityReading(reading.clone()))?;

    // Link source to reading
    create_link(
        reading.source_hash.clone(),
        action_hash.clone(),
        LinkTypes::SourceToReading,
        (),
    )?;

    // Link sampler to reading
    create_link(
        reading.sampler.clone(),
        action_hash.clone(),
        LinkTypes::SamplerToReading,
        (),
    )?;

    // Link to all readings
    create_entry(&EntryTypes::Anchor(Anchor("all_readings".to_string())))?;
    create_link(
        anchor_hash("all_readings")?,
        action_hash.clone(),
        LinkTypes::AllReadings,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created reading".into()
    )))
}

/// Get all quality readings for a water source
#[hdk_extern]
pub fn get_source_readings(source_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(source_hash, LinkTypes::SourceToReading)?,
        GetStrategy::default(),
    )?;
    let mut records = records_from_links(links)?;
    records.sort_by(|a, b| a.action().timestamp().cmp(&b.action().timestamp()));
    Ok(records)
}

/// Check potability of a water source based on latest reading
#[hdk_extern]
pub fn check_potability(source_hash: ActionHash) -> ExternResult<PotabilityResult> {
    let readings = get_source_readings(source_hash)?;

    if let Some(latest) = readings.into_iter().last() {
        let reading: QualityReading = latest
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid reading entry".into()
            )))?;

        let mut warnings = Vec::new();

        if let Some(ph) = reading.ph {
            if ph < 6.5 || ph > 8.5 {
                warnings.push(format!("pH out of range: {:.1}", ph));
            }
        }
        if let Some(turb) = reading.turbidity_ntu {
            if turb > 4.0 {
                warnings.push(format!("High turbidity: {:.1} NTU", turb));
            }
        }
        if let Some(lead) = reading.lead_ug_l {
            if lead > 10.0 {
                warnings.push(format!("Lead exceeds limit: {:.1} ug/L", lead));
            }
        }
        if let Some(arsenic) = reading.arsenic_ug_l {
            if arsenic > 10.0 {
                warnings.push(format!("Arsenic exceeds limit: {:.1} ug/L", arsenic));
            }
        }
        if let Some(ecoli) = reading.e_coli_cfu {
            if ecoli > 0 {
                warnings.push(format!("E. coli detected: {} CFU", ecoli));
            }
        }
        if let Some(coliform) = reading.total_coliform_cfu {
            if coliform > 0 {
                warnings.push(format!("Total coliform detected: {} CFU", coliform));
            }
        }

        Ok(PotabilityResult {
            is_potable: reading.meets_who_standards && reading.meets_epa_standards,
            potability_score: reading.potability_score,
            meets_who: reading.meets_who_standards,
            meets_epa: reading.meets_epa_standards,
            warnings,
            reading_timestamp: reading.timestamp,
        })
    } else {
        Ok(PotabilityResult {
            is_potable: false,
            potability_score: 0.0,
            meets_who: false,
            meets_epa: false,
            warnings: vec!["No quality readings available for this source".to_string()],
            reading_timestamp: Timestamp::from_micros(0),
        })
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PotabilityResult {
    pub is_potable: bool,
    pub potability_score: f32,
    pub meets_who: bool,
    pub meets_epa: bool,
    pub warnings: Vec<String>,
    pub reading_timestamp: Timestamp,
}

// ============================================================================
// CONTAMINATION ALERTS
// ============================================================================

/// Raise a contamination alert for a water source
#[hdk_extern]
pub fn raise_alert(alert: ContaminationAlert) -> ExternResult<Record> {
    if alert.contaminant.is_empty() || alert.contaminant.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Contaminant name must be 1-256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::ContaminationAlert(alert.clone()))?;

    // Link source to alert
    create_link(
        alert.source_hash.clone(),
        action_hash.clone(),
        LinkTypes::SourceToAlert,
        (),
    )?;

    // Link to active alerts anchor
    create_entry(&EntryTypes::Anchor(Anchor("active_alerts".to_string())))?;
    create_link(
        anchor_hash("active_alerts")?,
        action_hash.clone(),
        LinkTypes::ActiveAlerts,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created alert".into()
    )))
}

/// Get all active (unresolved) alerts
#[hdk_extern]
pub fn get_active_alerts(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("active_alerts")?, LinkTypes::ActiveAlerts)?,
        GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    // Filter to only unresolved alerts
    let mut active = Vec::new();
    for record in records {
        if let Some(alert) = record
            .entry()
            .to_app_option::<ContaminationAlert>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if alert.resolved_at.is_none() {
                active.push(record);
            }
        }
    }
    Ok(active)
}

/// Resolve an alert by marking it with a resolution timestamp
#[hdk_extern]
pub fn resolve_alert(input: ResolveAlertInput) -> ExternResult<Record> {
    let record = get(input.alert_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Alert not found".into())))?;
    let mut alert: ContaminationAlert = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid alert entry".into()
        )))?;

    if alert.resolved_at.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Alert is already resolved".into()
        )));
    }

    let now = sys_time()?;
    alert.resolved_at = Some(now);
    alert.remediation_hash = input.remediation_hash;

    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::ContaminationAlert(alert),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated alert".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveAlertInput {
    pub alert_hash: ActionHash,
    pub remediation_hash: Option<ActionHash>,
}

// ============================================================================
// REMEDIATION
// ============================================================================

/// Start a remediation action for a contamination alert
#[hdk_extern]
pub fn start_remediation(remediation: Remediation) -> ExternResult<Record> {
    if remediation.method.is_empty() || remediation.method.len() > 1024 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Remediation method must be 1-1024 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Remediation(remediation.clone()))?;

    // Link alert to remediation
    create_link(
        remediation.alert_hash.clone(),
        action_hash.clone(),
        LinkTypes::AlertToRemediation,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created remediation".into()
    )))
}

/// Complete a remediation with verification
#[hdk_extern]
pub fn complete_remediation(input: CompleteRemediationInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let record = get(input.remediation_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Remediation not found".into())
    ))?;
    let mut remediation: Remediation = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid remediation entry".into()
        )))?;

    if remediation.completed_at.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Remediation is already completed".into()
        )));
    }

    let now = sys_time()?;
    remediation.completed_at = Some(now);
    remediation.verified_by = Some(agent_info.agent_initial_pubkey);
    remediation.post_treatment_reading = input.post_treatment_reading;
    if let Some(notes) = input.notes {
        remediation.notes = notes;
    }

    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::Remediation(remediation),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated remediation".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteRemediationInput {
    pub remediation_hash: ActionHash,
    pub post_treatment_reading: Option<ActionHash>,
    pub notes: Option<String>,
}
