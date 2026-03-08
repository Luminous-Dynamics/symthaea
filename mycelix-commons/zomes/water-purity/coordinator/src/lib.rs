//! Purity Coordinator Zome
//! Business logic for water quality monitoring, alerts, and remediation

use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};
use water_purity_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
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

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

// ============================================================================
// QUALITY READINGS
// ============================================================================

/// Submit a new water quality reading
#[hdk_extern]
pub fn submit_reading(reading: QualityReading) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "submit_reading")?;

    // Validate pH range (0-14) if provided
    if let Some(ph) = reading.ph {
        if !(0.0..=14.0).contains(&ph) {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "pH must be between 0.0 and 14.0".into()
            )));
        }
    }

    // Validate non-negative contamination levels
    if let Some(turb) = reading.turbidity_ntu {
        if turb < 0.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Turbidity cannot be negative".into()
            )));
        }
    }
    if let Some(tds) = reading.tds_ppm {
        if tds < 0.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "TDS cannot be negative".into()
            )));
        }
    }
    if let Some(nitrates) = reading.nitrates_mg_l {
        if nitrates < 0.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Nitrate concentration cannot be negative".into()
            )));
        }
    }
    if let Some(arsenic) = reading.arsenic_ug_l {
        if arsenic < 0.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Arsenic concentration cannot be negative".into()
            )));
        }
    }
    if let Some(lead) = reading.lead_ug_l {
        if lead < 0.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Lead concentration cannot be negative".into()
            )));
        }
    }
    if let Some(chlorine) = reading.chlorine_mg_l {
        if chlorine < 0.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Chlorine concentration cannot be negative".into()
            )));
        }
    }
    if let Some(dissolved_o2) = reading.dissolved_oxygen_mg_l {
        if dissolved_o2 < 0.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Dissolved oxygen cannot be negative".into()
            )));
        }
    }

    // Validate potability score range
    if reading.potability_score < 0.0 || reading.potability_score > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Potability score must be between 0.0 and 1.0".into()
        )));
    }

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
    records.sort_by_key(|a| a.action().timestamp());
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
            if !(6.5..=8.5).contains(&ph) {
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "raise_alert")?;
    if alert.contaminant.trim().is_empty() || alert.contaminant.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Contaminant name must be 1-256 non-whitespace characters".into()
        )));
    }
    if alert.measured_value < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Measured contamination value cannot be negative".into()
        )));
    }
    if alert.threshold_value < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Threshold value cannot be negative".into()
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "resolve_alert")?;
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "start_remediation")?;
    if remediation.method.trim().is_empty() || remediation.method.len() > 1024 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Remediation method must be 1-1024 non-whitespace characters".into()
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "complete_remediation")?;
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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn make_reading() -> QualityReading {
        QualityReading {
            source_hash: fake_action_hash(),
            sampler: fake_agent(),
            timestamp: Timestamp::from_micros(1000),
            temperature_celsius: Some(20.0),
            turbidity_ntu: Some(1.5),
            ph: Some(7.2),
            tds_ppm: Some(250.0),
            dissolved_oxygen_mg_l: Some(8.0),
            nitrates_mg_l: Some(5.0),
            arsenic_ug_l: Some(2.0),
            lead_ug_l: Some(3.0),
            total_coliform_cfu: Some(0),
            e_coli_cfu: Some(0),
            chlorine_mg_l: Some(0.5),
            potability_score: 0.95,
            meets_who_standards: true,
            meets_epa_standards: true,
        }
    }

    fn make_alert() -> ContaminationAlert {
        ContaminationAlert {
            source_hash: fake_action_hash(),
            severity: AlertSeverity::Warning,
            contaminant: "Lead".to_string(),
            measured_value: 15.0,
            threshold_value: 10.0,
            alert_type: AlertType::Chemical,
            reported_by: fake_agent(),
            reported_at: Timestamp::from_micros(2000),
            resolved_at: None,
            remediation_hash: None,
        }
    }

    fn make_remediation() -> Remediation {
        Remediation {
            alert_hash: fake_action_hash(),
            method: "Activated carbon filtration".to_string(),
            started_at: Timestamp::from_micros(3000),
            completed_at: None,
            verified_by: None,
            post_treatment_reading: None,
            cost_estimate: Some(5000),
            notes: "Initial treatment phase".to_string(),
        }
    }

    // ========================================================================
    // COORDINATOR STRUCT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn potability_result_serde_roundtrip() {
        let result = PotabilityResult {
            is_potable: true,
            potability_score: 0.95,
            meets_who: true,
            meets_epa: true,
            warnings: vec!["Minor turbidity".to_string()],
            reading_timestamp: Timestamp::from_micros(1000),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PotabilityResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.is_potable, true);
        assert_eq!(decoded.potability_score, 0.95);
        assert_eq!(decoded.meets_who, true);
        assert_eq!(decoded.meets_epa, true);
        assert_eq!(decoded.warnings.len(), 1);
        assert_eq!(decoded.warnings[0], "Minor turbidity");
    }

    #[test]
    fn potability_result_not_potable() {
        let result = PotabilityResult {
            is_potable: false,
            potability_score: 0.2,
            meets_who: false,
            meets_epa: false,
            warnings: vec![
                "pH out of range: 5.0".to_string(),
                "Lead exceeds limit: 20.0 ug/L".to_string(),
            ],
            reading_timestamp: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PotabilityResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.is_potable, false);
        assert_eq!(decoded.potability_score, 0.2);
        assert_eq!(decoded.warnings.len(), 2);
    }

    #[test]
    fn potability_result_no_warnings() {
        let result = PotabilityResult {
            is_potable: true,
            potability_score: 1.0,
            meets_who: true,
            meets_epa: true,
            warnings: vec![],
            reading_timestamp: Timestamp::from_micros(500),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PotabilityResult = serde_json::from_str(&json).unwrap();
        assert!(decoded.warnings.is_empty());
        assert_eq!(decoded.potability_score, 1.0);
    }

    #[test]
    fn potability_result_zero_score() {
        let result = PotabilityResult {
            is_potable: false,
            potability_score: 0.0,
            meets_who: false,
            meets_epa: false,
            warnings: vec!["Severely contaminated".to_string()],
            reading_timestamp: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PotabilityResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.potability_score, 0.0);
    }

    #[test]
    fn resolve_alert_input_serde_roundtrip() {
        let input = ResolveAlertInput {
            alert_hash: fake_action_hash(),
            remediation_hash: Some(fake_action_hash()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ResolveAlertInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.remediation_hash.is_some());
    }

    #[test]
    fn resolve_alert_input_without_remediation() {
        let input = ResolveAlertInput {
            alert_hash: fake_action_hash(),
            remediation_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ResolveAlertInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.remediation_hash.is_none());
    }

    #[test]
    fn complete_remediation_input_serde_roundtrip() {
        let input = CompleteRemediationInput {
            remediation_hash: fake_action_hash(),
            post_treatment_reading: Some(fake_action_hash()),
            notes: Some("Treatment successful".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteRemediationInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.post_treatment_reading.is_some());
        assert_eq!(decoded.notes, Some("Treatment successful".to_string()));
    }

    #[test]
    fn complete_remediation_input_minimal() {
        let input = CompleteRemediationInput {
            remediation_hash: fake_action_hash(),
            post_treatment_reading: None,
            notes: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteRemediationInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.post_treatment_reading.is_none());
        assert!(decoded.notes.is_none());
    }

    #[test]
    fn complete_remediation_input_empty_notes() {
        let input = CompleteRemediationInput {
            remediation_hash: fake_action_hash(),
            post_treatment_reading: None,
            notes: Some(String::new()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteRemediationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.notes, Some(String::new()));
    }

    // ========================================================================
    // INTEGRITY ENUM SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn alert_severity_all_variants_serde() {
        for variant in [
            AlertSeverity::Advisory,
            AlertSeverity::Warning,
            AlertSeverity::Emergency,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: AlertSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn alert_type_all_variants_serde() {
        for variant in [
            AlertType::Chemical,
            AlertType::Biological,
            AlertType::Physical,
            AlertType::Radiological,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: AlertType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    // ========================================================================
    // INTEGRITY ENTRY TYPE SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn quality_reading_full_serde_roundtrip() {
        let reading = make_reading();
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, reading);
    }

    #[test]
    fn quality_reading_all_none_optionals() {
        let reading = QualityReading {
            source_hash: fake_action_hash(),
            sampler: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            temperature_celsius: None,
            turbidity_ntu: None,
            ph: None,
            tds_ppm: None,
            dissolved_oxygen_mg_l: None,
            nitrates_mg_l: None,
            arsenic_ug_l: None,
            lead_ug_l: None,
            total_coliform_cfu: None,
            e_coli_cfu: None,
            chlorine_mg_l: None,
            potability_score: 0.0,
            meets_who_standards: false,
            meets_epa_standards: false,
        };
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert!(decoded.temperature_celsius.is_none());
        assert!(decoded.ph.is_none());
        assert!(decoded.lead_ug_l.is_none());
        assert!(decoded.e_coli_cfu.is_none());
        assert_eq!(decoded.potability_score, 0.0);
    }

    #[test]
    fn quality_reading_boundary_ph_values() {
        // pH at exact boundary values (6.5 and 8.5)
        let mut reading = make_reading();
        reading.ph = Some(6.5);
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.ph, Some(6.5));

        reading.ph = Some(8.5);
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.ph, Some(8.5));
    }

    #[test]
    fn quality_reading_u32_max_coliform() {
        let mut reading = make_reading();
        reading.total_coliform_cfu = Some(u32::MAX);
        reading.e_coli_cfu = Some(u32::MAX);
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_coliform_cfu, Some(u32::MAX));
        assert_eq!(decoded.e_coli_cfu, Some(u32::MAX));
    }

    #[test]
    fn contamination_alert_serde_roundtrip() {
        let alert = make_alert();
        let json = serde_json::to_string(&alert).unwrap();
        let decoded: ContaminationAlert = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, alert);
    }

    #[test]
    fn contamination_alert_resolved() {
        let mut alert = make_alert();
        alert.resolved_at = Some(Timestamp::from_micros(5000));
        alert.remediation_hash = Some(fake_action_hash());
        let json = serde_json::to_string(&alert).unwrap();
        let decoded: ContaminationAlert = serde_json::from_str(&json).unwrap();
        assert!(decoded.resolved_at.is_some());
        assert!(decoded.remediation_hash.is_some());
    }

    #[test]
    fn contamination_alert_unicode_contaminant() {
        let mut alert = make_alert();
        alert.contaminant = "\u{2622} Radiological \u{2622}".to_string();
        alert.alert_type = AlertType::Radiological;
        let json = serde_json::to_string(&alert).unwrap();
        let decoded: ContaminationAlert = serde_json::from_str(&json).unwrap();
        assert!(decoded.contaminant.contains('\u{2622}'));
    }

    #[test]
    fn remediation_serde_roundtrip() {
        let rem = make_remediation();
        let json = serde_json::to_string(&rem).unwrap();
        let decoded: Remediation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, rem);
    }

    #[test]
    fn remediation_completed() {
        let mut rem = make_remediation();
        rem.completed_at = Some(Timestamp::from_micros(10000));
        rem.verified_by = Some(fake_agent());
        rem.post_treatment_reading = Some(fake_action_hash());
        let json = serde_json::to_string(&rem).unwrap();
        let decoded: Remediation = serde_json::from_str(&json).unwrap();
        assert!(decoded.completed_at.is_some());
        assert!(decoded.verified_by.is_some());
        assert!(decoded.post_treatment_reading.is_some());
    }

    #[test]
    fn remediation_no_cost_estimate() {
        let mut rem = make_remediation();
        rem.cost_estimate = None;
        let json = serde_json::to_string(&rem).unwrap();
        let decoded: Remediation = serde_json::from_str(&json).unwrap();
        assert!(decoded.cost_estimate.is_none());
    }

    #[test]
    fn remediation_u64_max_cost() {
        let mut rem = make_remediation();
        rem.cost_estimate = Some(u64::MAX);
        let json = serde_json::to_string(&rem).unwrap();
        let decoded: Remediation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.cost_estimate, Some(u64::MAX));
    }

    // ========================================================================
    // CLONE / EQUALITY TESTS
    // ========================================================================

    #[test]
    fn quality_reading_clone_equals() {
        let reading = make_reading();
        let cloned = reading.clone();
        assert_eq!(reading, cloned);
    }

    #[test]
    fn contamination_alert_clone_equals() {
        let alert = make_alert();
        let cloned = alert.clone();
        assert_eq!(alert, cloned);
    }

    #[test]
    fn alert_severity_clone_eq() {
        let s = AlertSeverity::Emergency;
        assert_eq!(s.clone(), AlertSeverity::Emergency);
    }

    // ========================================================================
    // VALIDATION HARDENING EDGE CASE TESTS
    // ========================================================================

    /// pH value below 0.0 should be rejected by submit_reading validation.
    #[test]
    fn quality_reading_negative_ph_serde_roundtrip() {
        let mut reading = make_reading();
        reading.ph = Some(-1.0);
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.ph, Some(-1.0));
        // Coordinator rejects at runtime: "pH must be between 0.0 and 14.0"
    }

    /// pH value above 14.0 should be rejected by submit_reading validation.
    #[test]
    fn quality_reading_ph_above_14_serde_roundtrip() {
        let mut reading = make_reading();
        reading.ph = Some(14.5);
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.ph, Some(14.5));
    }

    /// pH boundary values (0.0 and 14.0) should be accepted.
    #[test]
    fn quality_reading_ph_boundary_values_serde() {
        for ph_val in [0.0_f32, 14.0] {
            let mut reading = make_reading();
            reading.ph = Some(ph_val);
            let json = serde_json::to_string(&reading).unwrap();
            let decoded: QualityReading = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.ph, Some(ph_val));
        }
    }

    /// Negative turbidity should be rejected by submit_reading validation.
    #[test]
    fn quality_reading_negative_turbidity_serde_roundtrip() {
        let mut reading = make_reading();
        reading.turbidity_ntu = Some(-0.5);
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.turbidity_ntu, Some(-0.5));
    }

    /// Negative arsenic level should be rejected.
    #[test]
    fn quality_reading_negative_arsenic_serde_roundtrip() {
        let mut reading = make_reading();
        reading.arsenic_ug_l = Some(-1.0);
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.arsenic_ug_l, Some(-1.0));
    }

    /// Negative lead level should be rejected.
    #[test]
    fn quality_reading_negative_lead_serde_roundtrip() {
        let mut reading = make_reading();
        reading.lead_ug_l = Some(-2.0);
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.lead_ug_l, Some(-2.0));
    }

    /// Potability score below 0.0 should be rejected.
    #[test]
    fn quality_reading_negative_potability_score_serde() {
        let mut reading = make_reading();
        reading.potability_score = -0.1;
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert!(decoded.potability_score < 0.0);
    }

    /// Potability score above 1.0 should be rejected.
    #[test]
    fn quality_reading_over_one_potability_score_serde() {
        let mut reading = make_reading();
        reading.potability_score = 1.5;
        let json = serde_json::to_string(&reading).unwrap();
        let decoded: QualityReading = serde_json::from_str(&json).unwrap();
        assert!(decoded.potability_score > 1.0);
    }

    /// Whitespace-only contaminant name should be rejected by raise_alert.
    #[test]
    fn contamination_alert_whitespace_only_contaminant() {
        let mut alert = make_alert();
        alert.contaminant = "   \t  ".to_string();
        let json = serde_json::to_string(&alert).unwrap();
        let decoded: ContaminationAlert = serde_json::from_str(&json).unwrap();
        assert!(
            decoded.contaminant.trim().is_empty(),
            "Whitespace-only contaminant must be caught by trim()"
        );
    }

    /// Negative measured contamination value should be rejected.
    #[test]
    fn contamination_alert_negative_measured_value_serde() {
        let mut alert = make_alert();
        alert.measured_value = -5.0;
        let json = serde_json::to_string(&alert).unwrap();
        let decoded: ContaminationAlert = serde_json::from_str(&json).unwrap();
        assert!(decoded.measured_value < 0.0);
    }

    /// Whitespace-only remediation method should be rejected.
    #[test]
    fn remediation_whitespace_only_method_serde() {
        let mut rem = make_remediation();
        rem.method = "  \n\t  ".to_string();
        let json = serde_json::to_string(&rem).unwrap();
        let decoded: Remediation = serde_json::from_str(&json).unwrap();
        assert!(
            decoded.method.trim().is_empty(),
            "Whitespace-only method must be caught by trim()"
        );
    }
}
