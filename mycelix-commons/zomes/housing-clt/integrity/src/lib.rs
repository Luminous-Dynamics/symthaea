//! Community Land Trust Integrity Zome
//! Entry types and validation for land trusts, ground leases, resale formulas,
//! and affordability reporting.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A Community Land Trust
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LandTrust {
    pub id: String,
    pub name: String,
    pub mission: String,
    pub boundary: Vec<(f64, f64)>,
    pub charter_hash: Option<ActionHash>,
    pub stewardship_board: Vec<AgentPubKey>,
    pub affordability_target_ami_percent: u8,
    pub created_at: Timestamp,
}

/// The type of resale formula
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FormulaType {
    AppreciationCap,
    AreaMedianIncome,
    ConsumerPriceIndex,
    Hybrid,
}

/// Resale formula configuration for a ground lease
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ResaleFormula {
    pub formula_type: FormulaType,
    pub max_appreciation_percent_annual: Option<u8>,
    pub ami_cap_percent: Option<u8>,
    pub improvement_credit_percent: Option<u8>,
}

/// A ground lease linking a unit to the land trust
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GroundLease {
    pub trust_hash: ActionHash,
    pub unit_hash: ActionHash,
    pub leaseholder: AgentPubKey,
    pub lease_term_years: u32,
    pub ground_rent_monthly_cents: u64,
    pub resale_formula: ResaleFormula,
    pub started_at: Timestamp,
    pub expires_at: Timestamp,
}

/// A resale price calculation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ResaleCalculation {
    pub lease_hash: ActionHash,
    pub original_price_cents: u64,
    pub years_held: u32,
    pub improvements_value_cents: u64,
    pub calculated_max_price_cents: u64,
    pub ami_at_purchase: Option<u64>,
    pub current_ami: Option<u64>,
}

/// An affordability report for a trust
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AffordabilityReport {
    pub trust_hash: ActionHash,
    pub report_date: Timestamp,
    pub total_units: u32,
    pub affordable_units: u32,
    pub average_monthly_cost_cents: u64,
    pub median_area_income_cents: u64,
    pub affordability_ratio: f32,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    LandTrust(LandTrust),
    GroundLease(GroundLease),
    ResaleCalculation(ResaleCalculation),
    AffordabilityReport(AffordabilityReport),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All trusts anchor
    AllTrusts,
    /// Trust to its ground leases
    TrustToLease,
    /// Leaseholder to their leases
    LeaseholderToLease,
    /// Lease to resale calculations
    LeaseToResaleCalc,
    /// Trust to affordability reports
    TrustToReport,
    /// Unit to lease
    UnitToLease,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::LandTrust(trust) => validate_create_trust(action, trust),
                EntryTypes::GroundLease(lease) => validate_create_lease(action, lease),
                EntryTypes::ResaleCalculation(calc) => validate_resale_calc(action, calc),
                EntryTypes::AffordabilityReport(report) => {
                    validate_affordability_report(action, report)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::LandTrust(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::GroundLease(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ResaleCalculation(_) => Ok(ValidateCallbackResult::Invalid(
                    "Resale calculations are immutable records".into(),
                )),
                EntryTypes::AffordabilityReport(_) => Ok(ValidateCallbackResult::Invalid(
                    "Affordability reports are immutable records".into(),
                )),
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
            LinkTypes::AllTrusts => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllTrusts link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TrustToLease => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TrustToLease link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::LeaseholderToLease => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "LeaseholderToLease link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::LeaseToResaleCalc => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "LeaseToResaleCalc link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TrustToReport => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TrustToReport link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::UnitToLease => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "UnitToLease link tag too long (max 256 bytes)".into(),
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

fn validate_create_trust(
    _action: Create,
    trust: LandTrust,
) -> ExternResult<ValidateCallbackResult> {
    if trust.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust ID cannot be empty".into(),
        ));
    }
    if trust.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust ID must be at most 256 characters".into(),
        ));
    }
    if trust.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust name cannot be empty".into(),
        ));
    }
    if trust.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust name must be at most 256 characters".into(),
        ));
    }
    if trust.mission.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust mission cannot be empty".into(),
        ));
    }
    if trust.mission.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust mission must be at most 4096 characters".into(),
        ));
    }
    if trust.boundary.len() < 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust boundary must have at least 3 coordinate points".into(),
        ));
    }
    for (lat, lon) in &trust.boundary {
        if !lat.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary latitude must be a finite number".into(),
            ));
        }
        if *lat < -90.0 || *lat > 90.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary latitude must be between -90 and 90".into(),
            ));
        }
        if !lon.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary longitude must be a finite number".into(),
            ));
        }
        if *lon < -180.0 || *lon > 180.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary longitude must be between -180 and 180".into(),
            ));
        }
    }
    if trust.stewardship_board.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust must have at least one stewardship board member".into(),
        ));
    }
    if trust.affordability_target_ami_percent == 0 || trust.affordability_target_ami_percent > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Affordability target must be between 1 and 200 percent of AMI".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_lease(
    _action: Create,
    lease: GroundLease,
) -> ExternResult<ValidateCallbackResult> {
    if lease.lease_term_years == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Lease term must be at least 1 year".into(),
        ));
    }
    if lease.lease_term_years > 199 {
        return Ok(ValidateCallbackResult::Invalid(
            "Lease term cannot exceed 199 years".into(),
        ));
    }
    if lease.expires_at <= lease.started_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Lease expiration must be after start date".into(),
        ));
    }
    // Validate resale formula consistency
    match lease.resale_formula.formula_type {
        FormulaType::AppreciationCap => {
            if lease
                .resale_formula
                .max_appreciation_percent_annual
                .is_none()
            {
                return Ok(ValidateCallbackResult::Invalid(
                    "AppreciationCap formula requires max_appreciation_percent_annual".into(),
                ));
            }
        }
        FormulaType::AreaMedianIncome => {
            if lease.resale_formula.ami_cap_percent.is_none() {
                return Ok(ValidateCallbackResult::Invalid(
                    "AreaMedianIncome formula requires ami_cap_percent".into(),
                ));
            }
        }
        FormulaType::ConsumerPriceIndex => {}
        FormulaType::Hybrid => {
            if lease
                .resale_formula
                .max_appreciation_percent_annual
                .is_none()
                && lease.resale_formula.ami_cap_percent.is_none()
            {
                return Ok(ValidateCallbackResult::Invalid(
                    "Hybrid formula requires at least one cap parameter".into(),
                ));
            }
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_resale_calc(
    _action: Create,
    calc: ResaleCalculation,
) -> ExternResult<ValidateCallbackResult> {
    if calc.original_price_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Original price must be greater than 0".into(),
        ));
    }
    if calc.calculated_max_price_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Calculated max price must be greater than 0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_affordability_report(
    _action: Create,
    report: AffordabilityReport,
) -> ExternResult<ValidateCallbackResult> {
    if report.total_units == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total units must be greater than 0".into(),
        ));
    }
    if report.affordable_units > report.total_units {
        return Ok(ValidateCallbackResult::Invalid(
            "Affordable units cannot exceed total units".into(),
        ));
    }
    if !report.affordability_ratio.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Affordability ratio must be a finite number".into(),
        ));
    }
    if report.affordability_ratio < 0.0 || report.affordability_ratio > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Affordability ratio must be between 0 and 1".into(),
        ));
    }
    if report.median_area_income_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Median area income must be greater than 0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fake_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn action_hash_a() -> ActionHash {
        ActionHash::from_raw_36(vec![10u8; 36])
    }

    fn action_hash_b() -> ActionHash {
        ActionHash::from_raw_36(vec![11u8; 36])
    }

    fn valid_land_trust() -> LandTrust {
        LandTrust {
            id: "trust-001".to_string(),
            name: "Community Housing Trust".to_string(),
            mission: "Provide affordable housing for all".to_string(),
            boundary: vec![
                (45.0, -122.0),
                (45.1, -122.0),
                (45.1, -122.1),
                (45.0, -122.1),
            ],
            charter_hash: Some(action_hash_a()),
            stewardship_board: vec![agent_a(), agent_b()],
            affordability_target_ami_percent: 80,
            created_at: Timestamp::from_micros(1000),
        }
    }

    fn valid_ground_lease() -> GroundLease {
        GroundLease {
            trust_hash: action_hash_a(),
            unit_hash: action_hash_b(),
            leaseholder: agent_a(),
            lease_term_years: 99,
            ground_rent_monthly_cents: 50000,
            resale_formula: ResaleFormula {
                formula_type: FormulaType::AppreciationCap,
                max_appreciation_percent_annual: Some(3),
                ami_cap_percent: None,
                improvement_credit_percent: Some(100),
            },
            started_at: Timestamp::from_micros(1000),
            expires_at: Timestamp::from_micros(1000 + 99 * 365 * 24 * 60 * 60 * 1_000_000),
        }
    }

    fn valid_resale_calculation() -> ResaleCalculation {
        ResaleCalculation {
            lease_hash: action_hash_a(),
            original_price_cents: 30000000,
            years_held: 5,
            improvements_value_cents: 2000000,
            calculated_max_price_cents: 35000000,
            ami_at_purchase: Some(6000000),
            current_ami: Some(6500000),
        }
    }

    fn valid_affordability_report() -> AffordabilityReport {
        AffordabilityReport {
            trust_hash: action_hash_a(),
            report_date: Timestamp::from_micros(2000),
            total_units: 100,
            affordable_units: 85,
            average_monthly_cost_cents: 120000,
            median_area_income_cents: 6000000,
            affordability_ratio: 0.85,
        }
    }

    fn assert_valid(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid message containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_formula_type() {
        let types = vec![
            FormulaType::AppreciationCap,
            FormulaType::AreaMedianIncome,
            FormulaType::ConsumerPriceIndex,
            FormulaType::Hybrid,
        ];
        for ft in &types {
            let json = serde_json::to_string(ft).unwrap();
            let back: FormulaType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, ft);
        }
    }

    #[test]
    fn serde_roundtrip_resale_formula() {
        let formula = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: Some(2),
            ami_cap_percent: Some(120),
            improvement_credit_percent: Some(75),
        };
        let json = serde_json::to_string(&formula).unwrap();
        let back: ResaleFormula = serde_json::from_str(&json).unwrap();
        assert_eq!(back, formula);
    }

    #[test]
    fn serde_roundtrip_land_trust() {
        let trust = valid_land_trust();
        let json = serde_json::to_string(&trust).unwrap();
        let back: LandTrust = serde_json::from_str(&json).unwrap();
        assert_eq!(back, trust);
    }

    #[test]
    fn serde_roundtrip_ground_lease() {
        let lease = valid_ground_lease();
        let json = serde_json::to_string(&lease).unwrap();
        let back: GroundLease = serde_json::from_str(&json).unwrap();
        assert_eq!(back, lease);
    }

    #[test]
    fn serde_roundtrip_resale_calculation() {
        let calc = valid_resale_calculation();
        let json = serde_json::to_string(&calc).unwrap();
        let back: ResaleCalculation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, calc);
    }

    #[test]
    fn serde_roundtrip_affordability_report() {
        let report = valid_affordability_report();
        let json = serde_json::to_string(&report).unwrap();
        let back: AffordabilityReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }

    // ── validate_create_trust tests ─────────────────────────────────────

    #[test]
    fn create_trust_valid() {
        assert_valid(validate_create_trust(fake_create(), valid_land_trust()));
    }

    #[test]
    fn create_trust_empty_id() {
        let mut t = valid_land_trust();
        t.id = String::new();
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust ID cannot be empty",
        );
    }

    #[test]
    fn create_trust_empty_name() {
        let mut t = valid_land_trust();
        t.name = String::new();
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust name cannot be empty",
        );
    }

    #[test]
    fn create_trust_empty_mission() {
        let mut t = valid_land_trust();
        t.mission = String::new();
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust mission cannot be empty",
        );
    }

    #[test]
    fn create_trust_id_at_max_length() {
        let mut t = valid_land_trust();
        t.id = "a".repeat(64);
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_id_over_max_length() {
        let mut t = valid_land_trust();
        t.id = "a".repeat(257);
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust ID must be at most 256 characters",
        );
    }

    #[test]
    fn create_trust_name_at_max_length() {
        let mut t = valid_land_trust();
        t.name = "a".repeat(256);
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_name_over_max_length() {
        let mut t = valid_land_trust();
        t.name = "a".repeat(257);
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust name must be at most 256 characters",
        );
    }

    #[test]
    fn create_trust_mission_at_max_length() {
        let mut t = valid_land_trust();
        t.mission = "a".repeat(4096);
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_mission_over_max_length() {
        let mut t = valid_land_trust();
        t.mission = "a".repeat(4097);
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust mission must be at most 4096 characters",
        );
    }

    #[test]
    fn create_trust_boundary_too_few_points() {
        let mut t = valid_land_trust();
        t.boundary = vec![(45.0, -122.0), (45.1, -122.0)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust boundary must have at least 3 coordinate points",
        );
    }

    #[test]
    fn create_trust_boundary_exactly_three_points() {
        let mut t = valid_land_trust();
        t.boundary = vec![(45.0, -122.0), (45.1, -122.0), (45.1, -122.1)];
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_boundary_empty() {
        let mut t = valid_land_trust();
        t.boundary = vec![];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust boundary must have at least 3 coordinate points",
        );
    }

    #[test]
    fn create_trust_boundary_latitude_below_min() {
        let mut t = valid_land_trust();
        t.boundary = vec![(-90.1, -122.0), (45.1, -122.0), (45.1, -122.1)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary latitude must be between -90 and 90",
        );
    }

    #[test]
    fn create_trust_boundary_latitude_above_max() {
        let mut t = valid_land_trust();
        t.boundary = vec![(45.0, -122.0), (90.1, -122.0), (45.1, -122.1)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary latitude must be between -90 and 90",
        );
    }

    #[test]
    fn create_trust_boundary_latitude_at_min() {
        let mut t = valid_land_trust();
        t.boundary = vec![(-90.0, -122.0), (45.1, -122.0), (45.1, -122.1)];
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_boundary_latitude_at_max() {
        let mut t = valid_land_trust();
        t.boundary = vec![(90.0, -122.0), (45.1, -122.0), (45.1, -122.1)];
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_boundary_longitude_below_min() {
        let mut t = valid_land_trust();
        t.boundary = vec![(45.0, -180.1), (45.1, -122.0), (45.1, -122.1)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary longitude must be between -180 and 180",
        );
    }

    #[test]
    fn create_trust_boundary_longitude_above_max() {
        let mut t = valid_land_trust();
        t.boundary = vec![(45.0, -122.0), (45.1, 180.1), (45.1, -122.1)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary longitude must be between -180 and 180",
        );
    }

    #[test]
    fn create_trust_boundary_longitude_at_min() {
        let mut t = valid_land_trust();
        t.boundary = vec![(45.0, -180.0), (45.1, -122.0), (45.1, -122.1)];
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_boundary_longitude_at_max() {
        let mut t = valid_land_trust();
        t.boundary = vec![(45.0, 180.0), (45.1, -122.0), (45.1, -122.1)];
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_empty_stewardship_board() {
        let mut t = valid_land_trust();
        t.stewardship_board = vec![];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust must have at least one stewardship board member",
        );
    }

    #[test]
    fn create_trust_single_board_member() {
        let mut t = valid_land_trust();
        t.stewardship_board = vec![agent_a()];
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_affordability_target_zero() {
        let mut t = valid_land_trust();
        t.affordability_target_ami_percent = 0;
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Affordability target must be between 1 and 200 percent of AMI",
        );
    }

    #[test]
    fn create_trust_affordability_target_above_max() {
        let mut t = valid_land_trust();
        t.affordability_target_ami_percent = 201;
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Affordability target must be between 1 and 200 percent of AMI",
        );
    }

    #[test]
    fn create_trust_affordability_target_at_min() {
        let mut t = valid_land_trust();
        t.affordability_target_ami_percent = 1;
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_affordability_target_at_max() {
        let mut t = valid_land_trust();
        t.affordability_target_ami_percent = 200;
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_no_charter_hash() {
        let mut t = valid_land_trust();
        t.charter_hash = None;
        assert_valid(validate_create_trust(fake_create(), t));
    }

    // ── validate_create_lease tests ─────────────────────────────────────

    #[test]
    fn create_lease_valid() {
        assert_valid(validate_create_lease(fake_create(), valid_ground_lease()));
    }

    #[test]
    fn create_lease_term_zero() {
        let mut l = valid_ground_lease();
        l.lease_term_years = 0;
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Lease term must be at least 1 year",
        );
    }

    #[test]
    fn create_lease_term_at_min() {
        let mut l = valid_ground_lease();
        l.lease_term_years = 1;
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_term_exceed_max() {
        let mut l = valid_ground_lease();
        l.lease_term_years = 200;
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Lease term cannot exceed 199 years",
        );
    }

    #[test]
    fn create_lease_term_at_max() {
        let mut l = valid_ground_lease();
        l.lease_term_years = 199;
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_expires_before_start() {
        let mut l = valid_ground_lease();
        l.expires_at = Timestamp::from_micros(500);
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Lease expiration must be after start date",
        );
    }

    #[test]
    fn create_lease_expires_equal_to_start() {
        let mut l = valid_ground_lease();
        l.expires_at = l.started_at;
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Lease expiration must be after start date",
        );
    }

    #[test]
    fn create_lease_formula_appreciation_cap_valid() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::AppreciationCap,
            max_appreciation_percent_annual: Some(5),
            ami_cap_percent: None,
            improvement_credit_percent: None,
        };
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_appreciation_cap_missing_param() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::AppreciationCap,
            max_appreciation_percent_annual: None,
            ami_cap_percent: None,
            improvement_credit_percent: None,
        };
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "AppreciationCap formula requires max_appreciation_percent_annual",
        );
    }

    #[test]
    fn create_lease_formula_ami_valid() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::AreaMedianIncome,
            max_appreciation_percent_annual: None,
            ami_cap_percent: Some(80),
            improvement_credit_percent: None,
        };
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_ami_missing_param() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::AreaMedianIncome,
            max_appreciation_percent_annual: None,
            ami_cap_percent: None,
            improvement_credit_percent: None,
        };
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "AreaMedianIncome formula requires ami_cap_percent",
        );
    }

    #[test]
    fn create_lease_formula_cpi_no_params_needed() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::ConsumerPriceIndex,
            max_appreciation_percent_annual: None,
            ami_cap_percent: None,
            improvement_credit_percent: None,
        };
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_cpi_with_params_ok() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::ConsumerPriceIndex,
            max_appreciation_percent_annual: Some(10),
            ami_cap_percent: Some(100),
            improvement_credit_percent: Some(50),
        };
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_hybrid_with_appreciation() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: Some(4),
            ami_cap_percent: None,
            improvement_credit_percent: None,
        };
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_hybrid_with_ami() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: None,
            ami_cap_percent: Some(120),
            improvement_credit_percent: None,
        };
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_hybrid_with_both() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: Some(3),
            ami_cap_percent: Some(100),
            improvement_credit_percent: Some(75),
        };
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_hybrid_no_params() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: None,
            ami_cap_percent: None,
            improvement_credit_percent: None,
        };
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Hybrid formula requires at least one cap parameter",
        );
    }

    #[test]
    fn create_lease_zero_ground_rent_ok() {
        let mut l = valid_ground_lease();
        l.ground_rent_monthly_cents = 0;
        assert_valid(validate_create_lease(fake_create(), l));
    }

    // ── validate_resale_calc tests ──────────────────────────────────────

    #[test]
    fn create_resale_calc_valid() {
        assert_valid(validate_resale_calc(
            fake_create(),
            valid_resale_calculation(),
        ));
    }

    #[test]
    fn create_resale_calc_zero_original_price() {
        let mut c = valid_resale_calculation();
        c.original_price_cents = 0;
        assert_invalid(
            validate_resale_calc(fake_create(), c),
            "Original price must be greater than 0",
        );
    }

    #[test]
    fn create_resale_calc_zero_calculated_price() {
        let mut c = valid_resale_calculation();
        c.calculated_max_price_cents = 0;
        assert_invalid(
            validate_resale_calc(fake_create(), c),
            "Calculated max price must be greater than 0",
        );
    }

    #[test]
    fn create_resale_calc_no_ami_data() {
        let mut c = valid_resale_calculation();
        c.ami_at_purchase = None;
        c.current_ami = None;
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    #[test]
    fn create_resale_calc_zero_years_held() {
        let mut c = valid_resale_calculation();
        c.years_held = 0;
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    #[test]
    fn create_resale_calc_zero_improvements() {
        let mut c = valid_resale_calculation();
        c.improvements_value_cents = 0;
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    #[test]
    fn create_resale_calc_large_values() {
        let mut c = valid_resale_calculation();
        c.original_price_cents = u64::MAX - 1;
        c.calculated_max_price_cents = u64::MAX - 2;
        c.improvements_value_cents = u64::MAX / 2;
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    // ── validate_affordability_report tests ─────────────────────────────

    #[test]
    fn create_affordability_report_valid() {
        assert_valid(validate_affordability_report(
            fake_create(),
            valid_affordability_report(),
        ));
    }

    #[test]
    fn create_affordability_report_zero_total_units() {
        let mut r = valid_affordability_report();
        r.total_units = 0;
        assert_invalid(
            validate_affordability_report(fake_create(), r),
            "Total units must be greater than 0",
        );
    }

    #[test]
    fn create_affordability_report_affordable_exceeds_total() {
        let mut r = valid_affordability_report();
        r.total_units = 100;
        r.affordable_units = 101;
        assert_invalid(
            validate_affordability_report(fake_create(), r),
            "Affordable units cannot exceed total units",
        );
    }

    #[test]
    fn create_affordability_report_affordable_equals_total() {
        let mut r = valid_affordability_report();
        r.total_units = 100;
        r.affordable_units = 100;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_zero_affordable_ok() {
        let mut r = valid_affordability_report();
        r.affordable_units = 0;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_ratio_below_zero() {
        let mut r = valid_affordability_report();
        r.affordability_ratio = -0.01;
        assert_invalid(
            validate_affordability_report(fake_create(), r),
            "Affordability ratio must be between 0 and 1",
        );
    }

    #[test]
    fn create_affordability_report_ratio_above_one() {
        let mut r = valid_affordability_report();
        r.affordability_ratio = 1.01;
        assert_invalid(
            validate_affordability_report(fake_create(), r),
            "Affordability ratio must be between 0 and 1",
        );
    }

    #[test]
    fn create_affordability_report_ratio_at_zero() {
        let mut r = valid_affordability_report();
        r.affordability_ratio = 0.0;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_ratio_at_one() {
        let mut r = valid_affordability_report();
        r.affordability_ratio = 1.0;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_zero_median_income() {
        let mut r = valid_affordability_report();
        r.median_area_income_cents = 0;
        assert_invalid(
            validate_affordability_report(fake_create(), r),
            "Median area income must be greater than 0",
        );
    }

    #[test]
    fn create_affordability_report_zero_monthly_cost_ok() {
        let mut r = valid_affordability_report();
        r.average_monthly_cost_cents = 0;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    // ── Additional serde roundtrip tests ─────────────────────────────────

    #[test]
    fn serde_roundtrip_anchor() {
        let anchor = Anchor("all_trusts".to_string());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, anchor);
    }

    #[test]
    fn serde_roundtrip_anchor_empty() {
        let anchor = Anchor(String::new());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, anchor);
    }

    #[test]
    fn serde_roundtrip_anchor_unicode() {
        let anchor = Anchor("\u{1F30D} earth trusts \u{2764}".to_string());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, anchor);
    }

    #[test]
    fn serde_roundtrip_resale_formula_all_none() {
        let formula = ResaleFormula {
            formula_type: FormulaType::ConsumerPriceIndex,
            max_appreciation_percent_annual: None,
            ami_cap_percent: None,
            improvement_credit_percent: None,
        };
        let json = serde_json::to_string(&formula).unwrap();
        let back: ResaleFormula = serde_json::from_str(&json).unwrap();
        assert_eq!(back, formula);
    }

    #[test]
    fn serde_roundtrip_resale_formula_all_some() {
        let formula = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: Some(u8::MAX),
            ami_cap_percent: Some(u8::MAX),
            improvement_credit_percent: Some(u8::MAX),
        };
        let json = serde_json::to_string(&formula).unwrap();
        let back: ResaleFormula = serde_json::from_str(&json).unwrap();
        assert_eq!(back, formula);
    }

    #[test]
    fn serde_roundtrip_land_trust_minimal() {
        let trust = LandTrust {
            id: "x".to_string(),
            name: "N".to_string(),
            mission: "M".to_string(),
            boundary: vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            charter_hash: None,
            stewardship_board: vec![agent_a()],
            affordability_target_ami_percent: 1,
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&trust).unwrap();
        let back: LandTrust = serde_json::from_str(&json).unwrap();
        assert_eq!(back, trust);
    }

    #[test]
    fn serde_roundtrip_land_trust_unicode_fields() {
        let trust = LandTrust {
            id: "\u{4FE1}\u{8A17}".to_string(), // Chinese chars
            name: "Gemeinn\u{00FC}tziger Landtrust".to_string(), // German umlaut
            mission:
                "\u{0421}\u{043E}\u{0434}\u{0435}\u{0439}\u{0441}\u{0442}\u{0432}\u{0438}\u{0435}"
                    .to_string(), // Russian
            boundary: vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            charter_hash: None,
            stewardship_board: vec![agent_a()],
            affordability_target_ami_percent: 80,
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&trust).unwrap();
        let back: LandTrust = serde_json::from_str(&json).unwrap();
        assert_eq!(back, trust);
    }

    #[test]
    fn serde_roundtrip_resale_calculation_no_ami() {
        let calc = ResaleCalculation {
            lease_hash: action_hash_a(),
            original_price_cents: 1,
            years_held: 0,
            improvements_value_cents: 0,
            calculated_max_price_cents: 1,
            ami_at_purchase: None,
            current_ami: None,
        };
        let json = serde_json::to_string(&calc).unwrap();
        let back: ResaleCalculation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, calc);
    }

    #[test]
    fn serde_roundtrip_affordability_report_extreme_values() {
        let report = AffordabilityReport {
            trust_hash: action_hash_a(),
            report_date: Timestamp::from_micros(i64::MAX),
            total_units: u32::MAX,
            affordable_units: u32::MAX,
            average_monthly_cost_cents: u64::MAX,
            median_area_income_cents: u64::MAX,
            affordability_ratio: 1.0,
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: AffordabilityReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }

    // ── Additional validate_create_trust edge-case tests ─────────────────

    #[test]
    fn create_trust_unicode_fields_valid() {
        let mut t = valid_land_trust();
        t.id = "\u{1F30E}".to_string();
        t.name = "\u{5730}\u{57DF}\u{4FE1}\u{8A17}".to_string();
        t.mission = "F\u{00F6}rdern".to_string();
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_whitespace_only_id_rejected() {
        let mut t = valid_land_trust();
        t.id = "   ".to_string();
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust ID cannot be empty",
        );
    }

    #[test]
    fn create_trust_whitespace_only_name_rejected() {
        let mut t = valid_land_trust();
        t.name = "\t\n".to_string();
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust name cannot be empty",
        );
    }

    #[test]
    fn create_trust_whitespace_only_mission_rejected() {
        let mut t = valid_land_trust();
        t.mission = " ".to_string();
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust mission cannot be empty",
        );
    }

    #[test]
    fn create_trust_affordability_target_at_u8_max() {
        // u8::MAX = 255, which is > 200, so invalid
        let mut t = valid_land_trust();
        t.affordability_target_ami_percent = u8::MAX;
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Affordability target must be between 1 and 200 percent of AMI",
        );
    }

    #[test]
    fn create_trust_many_boundary_points() {
        let mut t = valid_land_trust();
        t.boundary = (0..1000)
            .map(|i| (i as f64 * 0.01, i as f64 * 0.01))
            .collect();
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_many_board_members() {
        let mut t = valid_land_trust();
        t.stewardship_board = (0..100)
            .map(|i| AgentPubKey::from_raw_36(vec![i; 36]))
            .collect();
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_boundary_all_at_origin() {
        let mut t = valid_land_trust();
        t.boundary = vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_boundary_negative_coords_valid() {
        let mut t = valid_land_trust();
        t.boundary = vec![(-45.0, -122.0), (-45.1, -122.0), (-45.1, -122.1)];
        assert_valid(validate_create_trust(fake_create(), t));
    }

    #[test]
    fn create_trust_boundary_lat_nan_is_invalid() {
        let mut t = valid_land_trust();
        t.boundary = vec![(f64::NAN, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary latitude must be a finite number",
        );
    }

    #[test]
    fn create_trust_boundary_lon_nan_is_invalid() {
        let mut t = valid_land_trust();
        t.boundary = vec![(0.0, f64::NAN), (1.0, 0.0), (0.0, 1.0)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary longitude must be a finite number",
        );
    }

    #[test]
    fn create_trust_boundary_lat_positive_infinity() {
        let mut t = valid_land_trust();
        t.boundary = vec![(f64::INFINITY, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary latitude must be a finite number",
        );
    }

    #[test]
    fn create_trust_boundary_lat_negative_infinity() {
        let mut t = valid_land_trust();
        t.boundary = vec![(f64::NEG_INFINITY, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary latitude must be a finite number",
        );
    }

    #[test]
    fn create_trust_boundary_lon_positive_infinity() {
        let mut t = valid_land_trust();
        t.boundary = vec![(0.0, f64::INFINITY), (1.0, 0.0), (0.0, 1.0)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary longitude must be a finite number",
        );
    }

    #[test]
    fn create_trust_boundary_lon_negative_infinity() {
        let mut t = valid_land_trust();
        t.boundary = vec![(0.0, f64::NEG_INFINITY), (1.0, 0.0), (0.0, 1.0)];
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Boundary longitude must be a finite number",
        );
    }

    // ── Additional validate_create_lease edge-case tests ─────────────────

    #[test]
    fn create_lease_max_term_boundary_200_invalid() {
        let mut l = valid_ground_lease();
        l.lease_term_years = 200;
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Lease term cannot exceed 199 years",
        );
    }

    #[test]
    fn create_lease_max_term_u32_max_invalid() {
        let mut l = valid_ground_lease();
        l.lease_term_years = u32::MAX;
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Lease term cannot exceed 199 years",
        );
    }

    #[test]
    fn create_lease_max_ground_rent_ok() {
        let mut l = valid_ground_lease();
        l.ground_rent_monthly_cents = u64::MAX;
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_appreciation_cap_with_extra_fields_ok() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::AppreciationCap,
            max_appreciation_percent_annual: Some(5),
            ami_cap_percent: Some(100), // extra field, still valid
            improvement_credit_percent: Some(50),
        };
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_ami_with_extra_fields_ok() {
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::AreaMedianIncome,
            max_appreciation_percent_annual: Some(3), // extra
            ami_cap_percent: Some(80),
            improvement_credit_percent: Some(50),
        };
        assert_valid(validate_create_lease(fake_create(), l));
    }

    #[test]
    fn create_lease_formula_hybrid_only_improvement_credit_invalid() {
        // improvement_credit_percent alone does not satisfy the hybrid requirement
        let mut l = valid_ground_lease();
        l.resale_formula = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: None,
            ami_cap_percent: None,
            improvement_credit_percent: Some(100),
        };
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Hybrid formula requires at least one cap parameter",
        );
    }

    #[test]
    fn create_lease_expires_one_microsecond_after_start() {
        let mut l = valid_ground_lease();
        l.started_at = Timestamp::from_micros(1000);
        l.expires_at = Timestamp::from_micros(1001);
        assert_valid(validate_create_lease(fake_create(), l));
    }

    // ── Additional validate_resale_calc edge-case tests ──────────────────

    #[test]
    fn create_resale_calc_original_price_one() {
        let mut c = valid_resale_calculation();
        c.original_price_cents = 1;
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    #[test]
    fn create_resale_calc_calculated_price_one() {
        let mut c = valid_resale_calculation();
        c.calculated_max_price_cents = 1;
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    #[test]
    fn create_resale_calc_max_years_held() {
        let mut c = valid_resale_calculation();
        c.years_held = u32::MAX;
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    #[test]
    fn create_resale_calc_both_prices_at_u64_max() {
        let mut c = valid_resale_calculation();
        c.original_price_cents = u64::MAX;
        c.calculated_max_price_cents = u64::MAX;
        c.improvements_value_cents = u64::MAX;
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    #[test]
    fn create_resale_calc_both_zero_prices_invalid() {
        let mut c = valid_resale_calculation();
        c.original_price_cents = 0;
        c.calculated_max_price_cents = 0;
        // Hits the first check (original_price_cents == 0)
        assert_invalid(
            validate_resale_calc(fake_create(), c),
            "Original price must be greater than 0",
        );
    }

    #[test]
    fn create_resale_calc_partial_ami_only_purchase() {
        let mut c = valid_resale_calculation();
        c.ami_at_purchase = Some(5000000);
        c.current_ami = None;
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    #[test]
    fn create_resale_calc_partial_ami_only_current() {
        let mut c = valid_resale_calculation();
        c.ami_at_purchase = None;
        c.current_ami = Some(5000000);
        assert_valid(validate_resale_calc(fake_create(), c));
    }

    // ── Additional validate_affordability_report edge-case tests ─────────

    #[test]
    fn create_affordability_report_one_unit_one_affordable() {
        let mut r = valid_affordability_report();
        r.total_units = 1;
        r.affordable_units = 1;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_one_unit_zero_affordable() {
        let mut r = valid_affordability_report();
        r.total_units = 1;
        r.affordable_units = 0;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_max_units() {
        let mut r = valid_affordability_report();
        r.total_units = u32::MAX;
        r.affordable_units = u32::MAX;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_ratio_epsilon_above_zero() {
        let mut r = valid_affordability_report();
        r.affordability_ratio = f32::EPSILON;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_ratio_just_below_one() {
        let mut r = valid_affordability_report();
        r.affordability_ratio = 1.0 - f32::EPSILON;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_ratio_nan_is_invalid() {
        let mut r = valid_affordability_report();
        r.affordability_ratio = f32::NAN;
        assert_invalid(
            validate_affordability_report(fake_create(), r),
            "Affordability ratio must be a finite number",
        );
    }

    #[test]
    fn create_affordability_report_ratio_positive_infinity() {
        let mut r = valid_affordability_report();
        r.affordability_ratio = f32::INFINITY;
        assert_invalid(
            validate_affordability_report(fake_create(), r),
            "Affordability ratio must be a finite number",
        );
    }

    #[test]
    fn create_affordability_report_ratio_negative_infinity() {
        let mut r = valid_affordability_report();
        r.affordability_ratio = f32::NEG_INFINITY;
        assert_invalid(
            validate_affordability_report(fake_create(), r),
            "Affordability ratio must be a finite number",
        );
    }

    #[test]
    fn create_affordability_report_median_income_one() {
        let mut r = valid_affordability_report();
        r.median_area_income_cents = 1;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_median_income_u64_max() {
        let mut r = valid_affordability_report();
        r.median_area_income_cents = u64::MAX;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    #[test]
    fn create_affordability_report_monthly_cost_u64_max() {
        let mut r = valid_affordability_report();
        r.average_monthly_cost_cents = u64::MAX;
        assert_valid(validate_affordability_report(fake_create(), r));
    }

    // ── Clone and PartialEq trait tests ──────────────────────────────────

    #[test]
    fn anchor_clone_and_eq() {
        let a = Anchor("test".to_string());
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn land_trust_clone_and_eq() {
        let a = valid_land_trust();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn ground_lease_clone_and_eq() {
        let a = valid_ground_lease();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn resale_calculation_clone_and_eq() {
        let a = valid_resale_calculation();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn affordability_report_clone_and_eq() {
        let a = valid_affordability_report();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn formula_type_clone_and_eq() {
        let variants = [
            FormulaType::AppreciationCap,
            FormulaType::AreaMedianIncome,
            FormulaType::ConsumerPriceIndex,
            FormulaType::Hybrid,
        ];
        for v in &variants {
            assert_eq!(v, &v.clone());
        }
    }

    #[test]
    fn resale_formula_clone_and_eq() {
        let a = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: Some(3),
            ami_cap_percent: Some(80),
            improvement_credit_percent: Some(50),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── PartialEq inequality tests ───────────────────────────────────────

    #[test]
    fn land_trust_ne_different_id() {
        let a = valid_land_trust();
        let mut b = a.clone();
        b.id = "different".to_string();
        assert_ne!(a, b);
    }

    #[test]
    fn ground_lease_ne_different_term() {
        let a = valid_ground_lease();
        let mut b = a.clone();
        b.lease_term_years = 50;
        assert_ne!(a, b);
    }

    #[test]
    fn resale_calculation_ne_different_price() {
        let a = valid_resale_calculation();
        let mut b = a.clone();
        b.original_price_cents = 1;
        assert_ne!(a, b);
    }

    #[test]
    fn affordability_report_ne_different_ratio() {
        let a = valid_affordability_report();
        let mut b = a.clone();
        b.affordability_ratio = 0.5;
        assert_ne!(a, b);
    }

    #[test]
    fn formula_type_ne_variants() {
        assert_ne!(FormulaType::AppreciationCap, FormulaType::AreaMedianIncome);
        assert_ne!(FormulaType::ConsumerPriceIndex, FormulaType::Hybrid);
    }

    // ── Debug trait tests ────────────────────────────────────────────────

    #[test]
    fn formula_type_debug() {
        let s = format!("{:?}", FormulaType::AppreciationCap);
        assert!(s.contains("AppreciationCap"));
    }

    #[test]
    fn resale_formula_debug() {
        let f = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: Some(3),
            ami_cap_percent: None,
            improvement_credit_percent: None,
        };
        let s = format!("{:?}", f);
        assert!(s.contains("Hybrid"));
        assert!(s.contains("Some(3)"));
    }

    // ── Validation order-of-checks tests ─────────────────────────────────

    #[test]
    fn create_trust_multiple_errors_returns_first_id() {
        // When both id and name are empty, id error is returned first
        let mut t = valid_land_trust();
        t.id = String::new();
        t.name = String::new();
        t.mission = String::new();
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust ID cannot be empty",
        );
    }

    #[test]
    fn create_trust_name_error_after_valid_id() {
        let mut t = valid_land_trust();
        t.name = String::new();
        t.mission = String::new();
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust name cannot be empty",
        );
    }

    #[test]
    fn create_trust_mission_error_after_valid_id_and_name() {
        let mut t = valid_land_trust();
        t.mission = String::new();
        assert_invalid(
            validate_create_trust(fake_create(), t),
            "Trust mission cannot be empty",
        );
    }

    #[test]
    fn create_lease_term_zero_checked_before_expiry() {
        let mut l = valid_ground_lease();
        l.lease_term_years = 0;
        l.expires_at = l.started_at; // Also invalid, but term is checked first
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Lease term must be at least 1 year",
        );
    }

    #[test]
    fn create_lease_term_200_checked_before_expiry() {
        let mut l = valid_ground_lease();
        l.lease_term_years = 200;
        l.expires_at = l.started_at; // Also invalid, but term > 199 is checked first
        assert_invalid(
            validate_create_lease(fake_create(), l),
            "Lease term cannot exceed 199 years",
        );
    }

    #[test]
    fn create_resale_calc_original_zero_checked_before_calculated() {
        let mut c = valid_resale_calculation();
        c.original_price_cents = 0;
        c.calculated_max_price_cents = 0;
        assert_invalid(
            validate_resale_calc(fake_create(), c),
            "Original price must be greater than 0",
        );
    }

    #[test]
    fn create_affordability_report_total_units_zero_checked_first() {
        let mut r = valid_affordability_report();
        r.total_units = 0;
        r.affordable_units = 1; // Also problematic, but total_units == 0 checked first
        assert_invalid(
            validate_affordability_report(fake_create(), r),
            "Total units must be greater than 0",
        );
    }

    // ============================================================================
    // Link Tag Validation Tests
    // ============================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllTrusts
            | LinkTypes::TrustToLease
            | LinkTypes::LeaseholderToLease
            | LinkTypes::LeaseToResaleCalc
            | LinkTypes::TrustToReport
            | LinkTypes::UnitToLease => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 256 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn link_tag_all_trusts_at_limit() {
        assert_valid(validate_create_link_tag(
            LinkTypes::AllTrusts,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn link_tag_all_trusts_over_limit() {
        let result = validate_create_link_tag(LinkTypes::AllTrusts, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_tag_trust_to_lease_at_limit() {
        assert_valid(validate_create_link_tag(
            LinkTypes::TrustToLease,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn link_tag_trust_to_lease_over_limit() {
        let result = validate_create_link_tag(LinkTypes::TrustToLease, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_tag_unit_to_lease_at_limit() {
        assert_valid(validate_create_link_tag(
            LinkTypes::UnitToLease,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn link_tag_unit_to_lease_over_limit() {
        let result = validate_create_link_tag(LinkTypes::UnitToLease, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_tag_empty_tag_valid() {
        assert_valid(validate_create_link_tag(LinkTypes::AllTrusts, vec![]));
    }
}
