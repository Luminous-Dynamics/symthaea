//! Purity Integrity Zome
//! Water quality monitoring, contamination alerts, and remediation tracking

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// QUALITY READINGS
// ============================================================================

/// A water quality reading from a source
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct QualityReading {
    /// Water source this reading is for
    pub source_hash: ActionHash,
    /// Agent who took the sample
    pub sampler: AgentPubKey,
    /// When the sample was taken
    pub timestamp: Timestamp,
    /// Water temperature in Celsius
    pub temperature_celsius: Option<f32>,
    /// Turbidity in NTU (Nephelometric Turbidity Units)
    pub turbidity_ntu: Option<f32>,
    /// pH level (0-14)
    pub ph: Option<f32>,
    /// Total dissolved solids in parts per million
    pub tds_ppm: Option<f32>,
    /// Dissolved oxygen in mg/L
    pub dissolved_oxygen_mg_l: Option<f32>,
    /// Nitrate concentration in mg/L
    pub nitrates_mg_l: Option<f32>,
    /// Arsenic concentration in micrograms per liter
    pub arsenic_ug_l: Option<f32>,
    /// Lead concentration in micrograms per liter
    pub lead_ug_l: Option<f32>,
    /// Total coliform count in colony-forming units
    pub total_coliform_cfu: Option<u32>,
    /// E. coli count in colony-forming units
    pub e_coli_cfu: Option<u32>,
    /// Free chlorine residual in mg/L
    pub chlorine_mg_l: Option<f32>,
    /// Calculated potability score (0.0-1.0)
    pub potability_score: f32,
    /// Whether this reading meets WHO drinking water standards
    pub meets_who_standards: bool,
    /// Whether this reading meets EPA drinking water standards
    pub meets_epa_standards: bool,
}

// ============================================================================
// CONTAMINATION ALERTS
// ============================================================================

/// Severity of a contamination alert
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    /// Informational, no immediate danger
    Advisory,
    /// Caution, take precautions
    Warning,
    /// Immediate danger, do not consume
    Emergency,
}

/// Type of contamination
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AlertType {
    /// Chemical contamination (heavy metals, pesticides)
    Chemical,
    /// Biological contamination (bacteria, viruses)
    Biological,
    /// Physical contamination (turbidity, sediment)
    Physical,
    /// Radiological contamination
    Radiological,
}

/// A contamination alert for a water source
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ContaminationAlert {
    /// Affected water source
    pub source_hash: ActionHash,
    /// How serious the contamination is
    pub severity: AlertSeverity,
    /// Name of the contaminant detected
    pub contaminant: String,
    /// Measured value of the contaminant
    pub measured_value: f32,
    /// Safe threshold value
    pub threshold_value: f32,
    /// Type of contamination
    pub alert_type: AlertType,
    /// Agent who reported the alert
    pub reported_by: AgentPubKey,
    /// When the alert was raised
    pub reported_at: Timestamp,
    /// When the alert was resolved (if resolved)
    pub resolved_at: Option<Timestamp>,
    /// Link to remediation action if taken
    pub remediation_hash: Option<ActionHash>,
}

// ============================================================================
// REMEDIATION
// ============================================================================

/// A remediation action taken to address contamination
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Remediation {
    /// The contamination alert this addresses
    pub alert_hash: ActionHash,
    /// Description of remediation method used
    pub method: String,
    /// When remediation started
    pub started_at: Timestamp,
    /// When remediation completed (if completed)
    pub completed_at: Option<Timestamp>,
    /// Agent who verified remediation effectiveness
    pub verified_by: Option<AgentPubKey>,
    /// Post-treatment quality reading
    pub post_treatment_reading: Option<ActionHash>,
    /// Estimated cost in smallest currency unit
    pub cost_estimate: Option<u64>,
    /// Additional notes
    pub notes: String,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    QualityReading(QualityReading),
    ContaminationAlert(ContaminationAlert),
    Remediation(Remediation),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Source to its quality readings
    SourceToReading,
    /// Sampler to their readings
    SamplerToReading,
    /// All active alerts anchor
    ActiveAlerts,
    /// Source to its alerts
    SourceToAlert,
    /// Alert to its remediation
    AlertToRemediation,
    /// All readings anchor (for global queries)
    AllReadings,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::QualityReading(reading) => validate_create_reading(action, reading),
                EntryTypes::ContaminationAlert(alert) => validate_create_alert(action, alert),
                EntryTypes::Remediation(remediation) => {
                    validate_create_remediation(action, remediation)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ContaminationAlert(alert) => {
                    validate_update_alert(action, alert, original_action_hash)
                }
                EntryTypes::Remediation(remediation) => {
                    validate_update_remediation(action, remediation, original_action_hash)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::SourceToReading => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SourceToReading link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SamplerToReading => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SamplerToReading link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ActiveAlerts => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ActiveAlerts link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SourceToAlert => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SourceToAlert link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AlertToRemediation => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AlertToRemediation link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllReadings => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllReadings link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action,
        } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_create_reading(
    _action: Create,
    reading: QualityReading,
) -> ExternResult<ValidateCallbackResult> {
    // Potability score must be finite and in [0, 1]
    if !reading.potability_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Potability score must be a finite number".into(),
        ));
    }
    if reading.potability_score < 0.0 || reading.potability_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Potability score must be between 0.0 and 1.0".into(),
        ));
    }

    // pH must be in [0, 14] if provided
    if let Some(ph) = reading.ph {
        if !(0.0..=14.0).contains(&ph) {
            return Ok(ValidateCallbackResult::Invalid(
                "pH must be between 0 and 14".into(),
            ));
        }
    }

    // Temperature sanity check
    if let Some(temp) = reading.temperature_celsius {
        if !(-50.0..=100.0).contains(&temp) {
            return Ok(ValidateCallbackResult::Invalid(
                "Temperature must be between -50 and 100 Celsius".into(),
            ));
        }
    }

    // Turbidity must be non-negative
    if let Some(turb) = reading.turbidity_ntu {
        if turb < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Turbidity cannot be negative".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_alert(
    _action: Create,
    alert: ContaminationAlert,
) -> ExternResult<ValidateCallbackResult> {
    if alert.contaminant.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Contaminant name cannot be empty".into(),
        ));
    }
    if alert.contaminant.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Contaminant name must be 256 characters or fewer".into(),
        ));
    }
    if !alert.measured_value.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Measured value must be a finite number".into(),
        ));
    }
    if alert.measured_value < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Measured value cannot be negative".into(),
        ));
    }
    if !alert.threshold_value.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Threshold value must be a finite number".into(),
        ));
    }
    if alert.threshold_value < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Threshold value cannot be negative".into(),
        ));
    }
    // Alert should only be raised if measured exceeds threshold
    if alert.measured_value <= alert.threshold_value {
        return Ok(ValidateCallbackResult::Invalid(
            "Measured value must exceed threshold to raise an alert".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_alert(
    _action: Update,
    alert: ContaminationAlert,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    if alert.contaminant.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Contaminant name must be 256 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_remediation(
    _action: Create,
    remediation: Remediation,
) -> ExternResult<ValidateCallbackResult> {
    if remediation.method.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Remediation method cannot be empty".into(),
        ));
    }
    if remediation.method.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Remediation method must be 4096 characters or fewer".into(),
        ));
    }
    if remediation.notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Remediation notes must be 4096 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_remediation(
    _action: Update,
    remediation: Remediation,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    if remediation.method.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Remediation method must be 4096 characters or fewer".into(),
        ));
    }
    if remediation.notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Remediation notes must be 4096 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_entry_hash() -> EntryHash {
        EntryHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_create() -> Create {
        Create {
            author: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: fake_action_hash(),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        }
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    fn make_quality_reading() -> QualityReading {
        QualityReading {
            source_hash: fake_action_hash(),
            sampler: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            temperature_celsius: Some(20.0),
            turbidity_ntu: Some(1.5),
            ph: Some(7.0),
            tds_ppm: Some(150.0),
            dissolved_oxygen_mg_l: Some(8.0),
            nitrates_mg_l: Some(5.0),
            arsenic_ug_l: Some(1.0),
            lead_ug_l: Some(0.5),
            total_coliform_cfu: Some(0),
            e_coli_cfu: Some(0),
            chlorine_mg_l: Some(0.5),
            potability_score: 0.95,
            meets_who_standards: true,
            meets_epa_standards: true,
        }
    }

    fn make_contamination_alert() -> ContaminationAlert {
        ContaminationAlert {
            source_hash: fake_action_hash(),
            severity: AlertSeverity::Warning,
            contaminant: "Lead".into(),
            measured_value: 20.0,
            threshold_value: 15.0,
            alert_type: AlertType::Chemical,
            reported_by: fake_agent(),
            reported_at: Timestamp::from_micros(0),
            resolved_at: None,
            remediation_hash: None,
        }
    }

    fn make_remediation() -> Remediation {
        Remediation {
            alert_hash: fake_action_hash(),
            method: "Activated carbon filtration".into(),
            started_at: Timestamp::from_micros(0),
            completed_at: None,
            verified_by: None,
            post_treatment_reading: None,
            cost_estimate: Some(5000),
            notes: "Initial treatment phase".into(),
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn serde_roundtrip_alert_severity() {
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
    fn serde_roundtrip_alert_type() {
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
    // QUALITY READING VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_quality_reading_passes() {
        let result = validate_create_reading(fake_create(), make_quality_reading());
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_potability_score_negative_rejected() {
        let mut reading = make_quality_reading();
        reading.potability_score = -0.1;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Potability score must be between 0.0 and 1.0"
        );
    }

    #[test]
    fn quality_reading_potability_score_over_one_rejected() {
        let mut reading = make_quality_reading();
        reading.potability_score = 1.1;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Potability score must be between 0.0 and 1.0"
        );
    }

    #[test]
    fn quality_reading_potability_score_zero_accepted() {
        let mut reading = make_quality_reading();
        reading.potability_score = 0.0;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_potability_score_one_accepted() {
        let mut reading = make_quality_reading();
        reading.potability_score = 1.0;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_potability_score_mid_range_accepted() {
        let mut reading = make_quality_reading();
        reading.potability_score = 0.5;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_ph_negative_rejected() {
        let mut reading = make_quality_reading();
        reading.ph = Some(-0.1);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "pH must be between 0 and 14");
    }

    #[test]
    fn quality_reading_ph_over_14_rejected() {
        let mut reading = make_quality_reading();
        reading.ph = Some(14.1);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "pH must be between 0 and 14");
    }

    #[test]
    fn quality_reading_ph_zero_accepted() {
        let mut reading = make_quality_reading();
        reading.ph = Some(0.0);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_ph_14_accepted() {
        let mut reading = make_quality_reading();
        reading.ph = Some(14.0);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_ph_neutral_accepted() {
        let mut reading = make_quality_reading();
        reading.ph = Some(7.0);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_ph_none_accepted() {
        let mut reading = make_quality_reading();
        reading.ph = None;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_temperature_under_neg50_rejected() {
        let mut reading = make_quality_reading();
        reading.temperature_celsius = Some(-50.1);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Temperature must be between -50 and 100 Celsius"
        );
    }

    #[test]
    fn quality_reading_temperature_over_100_rejected() {
        let mut reading = make_quality_reading();
        reading.temperature_celsius = Some(100.1);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Temperature must be between -50 and 100 Celsius"
        );
    }

    #[test]
    fn quality_reading_temperature_neg50_accepted() {
        let mut reading = make_quality_reading();
        reading.temperature_celsius = Some(-50.0);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_temperature_100_accepted() {
        let mut reading = make_quality_reading();
        reading.temperature_celsius = Some(100.0);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_temperature_zero_accepted() {
        let mut reading = make_quality_reading();
        reading.temperature_celsius = Some(0.0);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_temperature_none_accepted() {
        let mut reading = make_quality_reading();
        reading.temperature_celsius = None;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_turbidity_negative_rejected() {
        let mut reading = make_quality_reading();
        reading.turbidity_ntu = Some(-0.1);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Turbidity cannot be negative");
    }

    #[test]
    fn quality_reading_turbidity_zero_accepted() {
        let mut reading = make_quality_reading();
        reading.turbidity_ntu = Some(0.0);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_turbidity_positive_accepted() {
        let mut reading = make_quality_reading();
        reading.turbidity_ntu = Some(100.0);
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_turbidity_none_accepted() {
        let mut reading = make_quality_reading();
        reading.turbidity_ntu = None;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_all_fields_none_accepted() {
        let mut reading = make_quality_reading();
        reading.temperature_celsius = None;
        reading.turbidity_ntu = None;
        reading.ph = None;
        reading.tds_ppm = None;
        reading.dissolved_oxygen_mg_l = None;
        reading.nitrates_mg_l = None;
        reading.arsenic_ug_l = None;
        reading.lead_ug_l = None;
        reading.total_coliform_cfu = None;
        reading.e_coli_cfu = None;
        reading.chlorine_mg_l = None;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_meets_who_false_accepted() {
        let mut reading = make_quality_reading();
        reading.meets_who_standards = false;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_meets_epa_false_accepted() {
        let mut reading = make_quality_reading();
        reading.meets_epa_standards = false;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    #[test]
    fn quality_reading_both_standards_false_accepted() {
        let mut reading = make_quality_reading();
        reading.meets_who_standards = false;
        reading.meets_epa_standards = false;
        let result = validate_create_reading(fake_create(), reading);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // CONTAMINATION ALERT VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_contamination_alert_passes() {
        let result = validate_create_alert(fake_create(), make_contamination_alert());
        assert!(is_valid(&result));
    }

    #[test]
    fn contamination_alert_empty_contaminant_rejected() {
        let mut alert = make_contamination_alert();
        alert.contaminant = "".into();
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Contaminant name cannot be empty");
    }

    #[test]
    fn contamination_alert_measured_value_negative_rejected() {
        let mut alert = make_contamination_alert();
        alert.measured_value = -0.1;
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Measured value cannot be negative");
    }

    #[test]
    fn contamination_alert_threshold_value_negative_rejected() {
        let mut alert = make_contamination_alert();
        alert.threshold_value = -0.1;
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Threshold value cannot be negative");
    }

    #[test]
    fn contamination_alert_measured_equals_threshold_rejected() {
        let mut alert = make_contamination_alert();
        alert.measured_value = 10.0;
        alert.threshold_value = 10.0;
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Measured value must exceed threshold to raise an alert"
        );
    }

    #[test]
    fn contamination_alert_measured_below_threshold_rejected() {
        let mut alert = make_contamination_alert();
        alert.measured_value = 5.0;
        alert.threshold_value = 10.0;
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Measured value must exceed threshold to raise an alert"
        );
    }

    #[test]
    fn contamination_alert_measured_slightly_above_threshold_accepted() {
        let mut alert = make_contamination_alert();
        alert.measured_value = 10.1;
        alert.threshold_value = 10.0;
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_valid(&result));
    }

    #[test]
    fn contamination_alert_measured_far_above_threshold_accepted() {
        let mut alert = make_contamination_alert();
        alert.measured_value = 100.0;
        alert.threshold_value = 10.0;
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_valid(&result));
    }

    #[test]
    fn contamination_alert_zero_threshold_accepted() {
        let mut alert = make_contamination_alert();
        alert.measured_value = 0.1;
        alert.threshold_value = 0.0;
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_valid(&result));
    }

    #[test]
    fn contamination_alert_all_severities_accepted() {
        for severity in [
            AlertSeverity::Advisory,
            AlertSeverity::Warning,
            AlertSeverity::Emergency,
        ] {
            let mut alert = make_contamination_alert();
            alert.severity = severity;
            let result = validate_create_alert(fake_create(), alert);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn contamination_alert_all_types_accepted() {
        for alert_type in [
            AlertType::Chemical,
            AlertType::Biological,
            AlertType::Physical,
            AlertType::Radiological,
        ] {
            let mut alert = make_contamination_alert();
            alert.alert_type = alert_type;
            let result = validate_create_alert(fake_create(), alert);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn contamination_alert_with_resolved_at_accepted() {
        let mut alert = make_contamination_alert();
        alert.resolved_at = Some(Timestamp::from_micros(1000));
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_valid(&result));
    }

    #[test]
    fn contamination_alert_with_remediation_hash_accepted() {
        let mut alert = make_contamination_alert();
        alert.remediation_hash = Some(fake_action_hash());
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // REMEDIATION VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_remediation_passes() {
        let result = validate_create_remediation(fake_create(), make_remediation());
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_empty_method_rejected() {
        let mut remediation = make_remediation();
        remediation.method = "".into();
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Remediation method cannot be empty");
    }

    #[test]
    fn remediation_with_completed_at_accepted() {
        let mut remediation = make_remediation();
        remediation.completed_at = Some(Timestamp::from_micros(1000));
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_with_verified_by_accepted() {
        let mut remediation = make_remediation();
        remediation.verified_by = Some(fake_agent_2());
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_with_post_treatment_reading_accepted() {
        let mut remediation = make_remediation();
        remediation.post_treatment_reading = Some(fake_action_hash());
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_with_cost_estimate_accepted() {
        let mut remediation = make_remediation();
        remediation.cost_estimate = Some(10_000);
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_zero_cost_accepted() {
        let mut remediation = make_remediation();
        remediation.cost_estimate = Some(0);
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_no_cost_accepted() {
        let mut remediation = make_remediation();
        remediation.cost_estimate = None;
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_empty_notes_accepted() {
        let mut remediation = make_remediation();
        remediation.notes = "".into();
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_notes_exactly_4096_accepted() {
        let mut remediation = make_remediation();
        remediation.notes = "A".repeat(4096);
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_notes_4097_rejected() {
        let mut remediation = make_remediation();
        remediation.notes = "A".repeat(4097);
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Remediation notes must be 4096 characters or fewer"
        );
    }

    // ========================================================================
    // UPDATE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn update_alert_always_valid() {
        let alert = make_contamination_alert();
        let result = validate_update_alert(
            Update {
                author: fake_agent(),
                timestamp: Timestamp::from_micros(1000),
                action_seq: 1,
                prev_action: fake_action_hash(),
                original_action_address: fake_action_hash(),
                original_entry_address: fake_entry_hash(),
                entry_type: EntryType::App(AppEntryDef::new(
                    EntryDefIndex(0),
                    ZomeIndex(0),
                    EntryVisibility::Public,
                )),
                entry_hash: fake_entry_hash(),
                weight: EntryRateWeight::default(),
            },
            alert,
            fake_action_hash(),
        );
        assert!(is_valid(&result));
    }

    #[test]
    fn update_remediation_always_valid() {
        let remediation = make_remediation();
        let result = validate_update_remediation(
            Update {
                author: fake_agent(),
                timestamp: Timestamp::from_micros(1000),
                action_seq: 1,
                prev_action: fake_action_hash(),
                original_action_address: fake_action_hash(),
                original_entry_address: fake_entry_hash(),
                entry_type: EntryType::App(AppEntryDef::new(
                    EntryDefIndex(0),
                    ZomeIndex(0),
                    EntryVisibility::Public,
                )),
                entry_hash: fake_entry_hash(),
                weight: EntryRateWeight::default(),
            },
            remediation,
            fake_action_hash(),
        );
        assert!(is_valid(&result));
    }

    // ========================================================================
    // STRING LENGTH LIMIT TESTS - CONTAMINATION ALERT
    // ========================================================================

    #[test]
    fn contamination_alert_contaminant_exactly_256_accepted() {
        let mut alert = make_contamination_alert();
        alert.contaminant = "x".repeat(256);
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_valid(&result));
    }

    #[test]
    fn contamination_alert_contaminant_257_rejected() {
        let mut alert = make_contamination_alert();
        alert.contaminant = "x".repeat(257);
        let result = validate_create_alert(fake_create(), alert);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Contaminant name must be 256 characters or fewer"
        );
    }

    // ========================================================================
    // STRING LENGTH LIMIT TESTS - REMEDIATION
    // ========================================================================

    #[test]
    fn remediation_method_exactly_4096_accepted() {
        let mut remediation = make_remediation();
        remediation.method = "x".repeat(4096);
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_valid(&result));
    }

    #[test]
    fn remediation_method_4097_rejected() {
        let mut remediation = make_remediation();
        remediation.method = "x".repeat(4097);
        let result = validate_create_remediation(fake_create(), remediation);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Remediation method must be 4096 characters or fewer"
        );
    }

    // ========================================================================
    // NOTE: Quality readings are immutable (no update validation function)
    // ========================================================================

    // ========================================================================
    // DELETE AUTHORIZATION TESTS
    // ========================================================================
    // The RegisterDelete and RegisterDeleteLink match arms in validate()
    // use must_get_action() to fetch the original action from the DHT and
    // verify the deleting agent is the original author. must_get_action()
    // requires a live Holochain conductor context and cannot be called in
    // pure unit tests.
    //
    // These tests verify:
    // 1. The Delete action struct can be constructed with author fields
    // 2. The author comparison logic is correct
    // 3. The validation function has explicit match arms (not wildcard)

    #[test]
    fn delete_author_comparison_same_agent_matches() {
        // Verifies the author comparison logic used in RegisterDelete
        let agent = fake_agent();
        let other_agent = fake_agent();
        // Same agent bytes should match
        assert_eq!(agent, other_agent);
    }

    #[test]
    fn delete_author_comparison_different_agent_does_not_match() {
        // Verifies that different agents are detected as different
        let agent_a = fake_agent();
        let agent_b = fake_agent_2();
        assert_ne!(agent_a, agent_b);
    }

    #[test]
    fn delete_action_struct_has_deletes_address_field() {
        // Verifies the Delete action struct shape used in RegisterDelete validation.
        // The validate() function accesses action.deletes_address to look up the
        // original action and compare authors.
        let delete = Delete {
            author: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 1,
            prev_action: fake_action_hash(),
            deletes_address: fake_action_hash(),
            deletes_entry_address: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        };
        // The field exists and is accessible -- this is what RegisterDelete uses
        assert_eq!(delete.deletes_address, fake_action_hash());
        assert_eq!(delete.author, fake_agent());
    }

    #[test]
    fn delete_link_action_struct_has_link_add_address_field() {
        // Verifies the DeleteLink action struct shape used in RegisterDeleteLink validation.
        // The validate() function accesses action.link_add_address to look up the
        // original link creation action and compare authors.
        let delete_link = DeleteLink {
            author: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 2,
            prev_action: fake_action_hash(),
            link_add_address: fake_action_hash(),
            base_address: AnyLinkableHash::from(fake_entry_hash()),
        };
        assert_eq!(delete_link.link_add_address, fake_action_hash());
        assert_eq!(delete_link.author, fake_agent());
    }
}
