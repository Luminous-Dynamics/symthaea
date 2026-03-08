//! Transport Impact Integrity Zome
//! Entry types and validation for trip logging, emissions tracking, and carbon credits.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// TRIP LOG
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TripMode {
    Driving,
    Cycling,
    Walking,
    Transit,
    Carpool,
    ElectricVehicle,
    Flying,
    Water,
    Rail,
    Micromobility,
    Autonomous,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TripLog {
    pub vehicle_hash: Option<ActionHash>,
    pub route_hash: Option<ActionHash>,
    pub distance_km: f64,
    pub mode: TripMode,
    pub passengers: u32,
    pub cargo_kg: f64,
    pub emissions_kg_co2: f64,
    pub logged_at: u64,
}

// ============================================================================
// CARBON CREDIT
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CreditSource {
    Cycling,
    Transit,
    Carpool,
    ElectricVehicle,
    Walking,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CarbonCredit {
    pub holder: AgentPubKey,
    pub credits_kg_co2: f64,
    pub earned_from: CreditSource,
    pub earned_at: u64,
}

// ============================================================================
// CREDIT REDEMPTION
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CreditRedemption {
    pub holder: AgentPubKey,
    pub credits_redeemed: f64,
    pub redeemed_for: String,
    pub redeemed_at: u64,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    TripLog(TripLog),
    CarbonCredit(CarbonCredit),
    CreditRedemption(CreditRedemption),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllTrips,
    AgentToTrip,
    AgentToCredit,
    VehicleToTrip,
    AgentToRedemptions,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::TripLog(t) => validate_trip(t),
                EntryTypes::CarbonCredit(c) => validate_credit(c),
                EntryTypes::CreditRedemption(r) => validate_redemption(r),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::TripLog(t) => validate_trip(t),
                EntryTypes::CarbonCredit(c) => validate_credit(c),
                EntryTypes::CreditRedemption(r) => validate_redemption(r),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::AllTrips => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllTrips link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToTrip => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToTrip link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToCredit => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToCredit link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::VehicleToTrip => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "VehicleToTrip link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToRedemptions => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToRedemptions link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_trip(t: TripLog) -> ExternResult<ValidateCallbackResult> {
    if !t.distance_km.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "distance_km must be a finite number".into(),
        ));
    }
    if t.distance_km <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Distance must be positive".into(),
        ));
    }
    if !t.cargo_kg.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "cargo_kg must be a finite number".into(),
        ));
    }
    if !t.emissions_kg_co2.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "emissions_kg_co2 must be a finite number".into(),
        ));
    }
    if t.emissions_kg_co2 < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Emissions cannot be negative".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_credit(c: CarbonCredit) -> ExternResult<ValidateCallbackResult> {
    if !c.credits_kg_co2.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "credits_kg_co2 must be a finite number".into(),
        ));
    }
    if c.credits_kg_co2 <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credits must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_redemption(r: CreditRedemption) -> ExternResult<ValidateCallbackResult> {
    if !r.credits_redeemed.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "credits_redeemed must be a finite number".into(),
        ));
    }
    if r.credits_redeemed <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credits redeemed must be positive".into(),
        ));
    }
    if r.redeemed_for.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Redeemed-for description cannot be empty".into(),
        ));
    }
    if r.redeemed_for.len() > 1024 {
        return Ok(ValidateCallbackResult::Invalid(
            "Redeemed-for description too long (max 1024 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn valid_trip() -> TripLog {
        TripLog {
            vehicle_hash: None,
            route_hash: None,
            distance_km: 15.0,
            mode: TripMode::Cycling,
            passengers: 1,
            cargo_kg: 0.0,
            emissions_kg_co2: 0.0,
            logged_at: 1700000000,
        }
    }

    fn valid_carbon_credit() -> CarbonCredit {
        CarbonCredit {
            holder: fake_agent(),
            credits_kg_co2: 2.1,
            earned_from: CreditSource::Cycling,
            earned_at: 1700000000,
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
                    "Expected Invalid containing '{expected_substr}', got: '{msg}'"
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
    fn serde_roundtrip_trip_mode() {
        let modes = vec![
            TripMode::Driving,
            TripMode::Cycling,
            TripMode::Walking,
            TripMode::Transit,
            TripMode::Carpool,
            TripMode::ElectricVehicle,
            TripMode::Flying,
            TripMode::Water,
            TripMode::Rail,
            TripMode::Micromobility,
            TripMode::Autonomous,
        ];
        for mode in &modes {
            let json = serde_json::to_string(mode).unwrap();
            let back: TripMode = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, mode);
        }
    }

    #[test]
    fn serde_roundtrip_credit_source() {
        let sources = vec![
            CreditSource::Cycling,
            CreditSource::Transit,
            CreditSource::Carpool,
            CreditSource::ElectricVehicle,
            CreditSource::Walking,
        ];
        for src in &sources {
            let json = serde_json::to_string(src).unwrap();
            let back: CreditSource = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, src);
        }
    }

    #[test]
    fn serde_roundtrip_trip_log() {
        let t = valid_trip();
        let json = serde_json::to_string(&t).unwrap();
        let back: TripLog = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn serde_roundtrip_trip_log_with_hashes() {
        let mut t = valid_trip();
        t.vehicle_hash = Some(ActionHash::from_raw_36(vec![10u8; 36]));
        t.route_hash = Some(ActionHash::from_raw_36(vec![11u8; 36]));
        let json = serde_json::to_string(&t).unwrap();
        let back: TripLog = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn serde_roundtrip_carbon_credit() {
        let c = valid_carbon_credit();
        let json = serde_json::to_string(&c).unwrap();
        let back: CarbonCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
    }

    // ── validate_trip: distance_km ──────────────────────────────────────

    #[test]
    fn valid_trip_passes() {
        assert_valid(validate_trip(valid_trip()));
    }

    #[test]
    fn trip_zero_distance_rejected() {
        let mut t = valid_trip();
        t.distance_km = 0.0;
        assert_invalid(validate_trip(t), "Distance must be positive");
    }

    #[test]
    fn trip_negative_distance_rejected() {
        let mut t = valid_trip();
        t.distance_km = -5.0;
        assert_invalid(validate_trip(t), "Distance must be positive");
    }

    #[test]
    fn trip_barely_positive_distance_valid() {
        let mut t = valid_trip();
        t.distance_km = 0.001;
        assert_valid(validate_trip(t));
    }

    #[test]
    fn trip_large_distance_valid() {
        let mut t = valid_trip();
        t.distance_km = 50000.0;
        assert_valid(validate_trip(t));
    }

    // ── validate_trip: emissions_kg_co2 ─────────────────────────────────

    #[test]
    fn trip_negative_emissions_rejected() {
        let mut t = valid_trip();
        t.emissions_kg_co2 = -0.5;
        assert_invalid(validate_trip(t), "Emissions cannot be negative");
    }

    #[test]
    fn trip_negative_emissions_large_rejected() {
        let mut t = valid_trip();
        t.emissions_kg_co2 = -100.0;
        assert_invalid(validate_trip(t), "Emissions cannot be negative");
    }

    #[test]
    fn trip_zero_emissions_valid() {
        let mut t = valid_trip();
        t.emissions_kg_co2 = 0.0;
        assert_valid(validate_trip(t));
    }

    #[test]
    fn trip_positive_emissions_valid() {
        let mut t = valid_trip();
        t.emissions_kg_co2 = 42.5;
        assert_valid(validate_trip(t));
    }

    #[test]
    fn trip_barely_negative_emissions_rejected() {
        let mut t = valid_trip();
        t.emissions_kg_co2 = -0.001;
        assert_invalid(validate_trip(t), "Emissions cannot be negative");
    }

    // ── validate_trip: mode ─────────────────────────────────────────────

    #[test]
    fn all_trip_modes_valid() {
        for mode in [
            TripMode::Driving,
            TripMode::Cycling,
            TripMode::Walking,
            TripMode::Transit,
            TripMode::Carpool,
            TripMode::ElectricVehicle,
            TripMode::Flying,
            TripMode::Water,
            TripMode::Rail,
            TripMode::Micromobility,
            TripMode::Autonomous,
        ] {
            let mut t = valid_trip();
            t.mode = mode;
            assert_valid(validate_trip(t));
        }
    }

    // ── validate_trip: optional hashes ──────────────────────────────────

    #[test]
    fn trip_with_vehicle_hash_valid() {
        let mut t = valid_trip();
        t.vehicle_hash = Some(ActionHash::from_raw_36(vec![10u8; 36]));
        assert_valid(validate_trip(t));
    }

    #[test]
    fn trip_with_route_hash_valid() {
        let mut t = valid_trip();
        t.route_hash = Some(ActionHash::from_raw_36(vec![11u8; 36]));
        assert_valid(validate_trip(t));
    }

    #[test]
    fn trip_with_both_hashes_valid() {
        let mut t = valid_trip();
        t.vehicle_hash = Some(ActionHash::from_raw_36(vec![10u8; 36]));
        t.route_hash = Some(ActionHash::from_raw_36(vec![11u8; 36]));
        assert_valid(validate_trip(t));
    }

    // ── validate_trip: passengers and cargo ─────────────────────────────

    #[test]
    fn trip_zero_passengers_valid() {
        let mut t = valid_trip();
        t.passengers = 0;
        assert_valid(validate_trip(t));
    }

    #[test]
    fn trip_many_passengers_valid() {
        let mut t = valid_trip();
        t.passengers = 50;
        assert_valid(validate_trip(t));
    }

    #[test]
    fn trip_zero_cargo_valid() {
        let mut t = valid_trip();
        t.cargo_kg = 0.0;
        assert_valid(validate_trip(t));
    }

    #[test]
    fn trip_large_cargo_valid() {
        let mut t = valid_trip();
        t.cargo_kg = 10000.0;
        assert_valid(validate_trip(t));
    }

    // ── validate_trip: combined invalid ─────────────────────────────────

    #[test]
    fn trip_zero_distance_with_negative_emissions_rejects_distance_first() {
        let mut t = valid_trip();
        t.distance_km = 0.0;
        t.emissions_kg_co2 = -1.0;
        // Distance check comes first
        assert_invalid(validate_trip(t), "Distance must be positive");
    }

    // ── validate_credit: credits_kg_co2 ─────────────────────────────────

    #[test]
    fn valid_credit_passes() {
        assert_valid(validate_credit(valid_carbon_credit()));
    }

    #[test]
    fn credit_zero_rejected() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = 0.0;
        assert_invalid(validate_credit(c), "Credits must be positive");
    }

    #[test]
    fn credit_negative_rejected() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = -1.0;
        assert_invalid(validate_credit(c), "Credits must be positive");
    }

    #[test]
    fn credit_barely_positive_valid() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = 0.001;
        assert_valid(validate_credit(c));
    }

    #[test]
    fn credit_large_value_valid() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = 999999.0;
        assert_valid(validate_credit(c));
    }

    #[test]
    fn credit_barely_negative_rejected() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = -0.001;
        assert_invalid(validate_credit(c), "Credits must be positive");
    }

    // ── validate_credit: source variants ────────────────────────────────

    #[test]
    fn all_credit_sources_valid() {
        for src in [
            CreditSource::Cycling,
            CreditSource::Transit,
            CreditSource::Carpool,
            CreditSource::ElectricVehicle,
            CreditSource::Walking,
        ] {
            let mut c = valid_carbon_credit();
            c.earned_from = src;
            assert_valid(validate_credit(c));
        }
    }

    // ── Anchor tests ────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip_anchor() {
        let a = Anchor("all_trips".to_string());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn anchor_empty_string_ok() {
        let a = Anchor(String::new());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    // ── Update validation: TripLog ───────────────────────────────────────

    #[test]
    fn test_update_trip_invalid_distance_rejected() {
        let mut t = valid_trip();
        t.distance_km = 0.0;
        assert_invalid(validate_trip(t), "Distance must be positive");
    }

    #[test]
    fn test_update_trip_invalid_negative_distance_rejected() {
        let mut t = valid_trip();
        t.distance_km = -10.0;
        assert_invalid(validate_trip(t), "Distance must be positive");
    }

    #[test]
    fn test_update_trip_invalid_emissions_rejected() {
        let mut t = valid_trip();
        t.emissions_kg_co2 = -1.0;
        assert_invalid(validate_trip(t), "Emissions cannot be negative");
    }

    // ── Update validation: CarbonCredit ──────────────────────────────────

    #[test]
    fn test_update_credit_invalid_zero_rejected() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = 0.0;
        assert_invalid(validate_credit(c), "Credits must be positive");
    }

    #[test]
    fn test_update_credit_invalid_negative_rejected() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = -5.0;
        assert_invalid(validate_credit(c), "Credits must be positive");
    }

    // ── Link tag length validation tests ────────────────────────────────

    fn validate_link_tag(link_type: &LinkTypes, tag_len: usize) -> ValidateCallbackResult {
        let tag = LinkTag(vec![0u8; tag_len]);
        let max = match link_type {
            LinkTypes::AllTrips
            | LinkTypes::AgentToTrip
            | LinkTypes::AgentToCredit
            | LinkTypes::VehicleToTrip
            | LinkTypes::AgentToRedemptions => 256,
        };
        let name = match link_type {
            LinkTypes::AllTrips => "AllTrips",
            LinkTypes::AgentToTrip => "AgentToTrip",
            LinkTypes::AgentToCredit => "AgentToCredit",
            LinkTypes::VehicleToTrip => "VehicleToTrip",
            LinkTypes::AgentToRedemptions => "AgentToRedemptions",
        };
        if tag.0.len() > max {
            ValidateCallbackResult::Invalid(format!(
                "{} link tag too long (max {} bytes)",
                name, max
            ))
        } else {
            ValidateCallbackResult::Valid
        }
    }

    #[test]
    fn test_link_all_trips_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllTrips, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_trips_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllTrips, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_agent_to_trip_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AgentToTrip, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_agent_to_trip_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AgentToTrip, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_agent_to_credit_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AgentToCredit, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_agent_to_credit_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AgentToCredit, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_vehicle_to_trip_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::VehicleToTrip, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_vehicle_to_trip_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::VehicleToTrip, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_agent_to_redemptions_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AgentToRedemptions, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_agent_to_redemptions_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AgentToRedemptions, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── Serde roundtrip: CreditRedemption ──────────────────────────────

    fn valid_redemption() -> CreditRedemption {
        CreditRedemption {
            holder: fake_agent(),
            credits_redeemed: 5.0,
            redeemed_for: "Transit pass discount".into(),
            redeemed_at: 1700000000,
        }
    }

    #[test]
    fn serde_roundtrip_credit_redemption() {
        let r = valid_redemption();
        let json = serde_json::to_string(&r).unwrap();
        let back: CreditRedemption = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    // ── validate_redemption tests ──────────────────────────────────────

    #[test]
    fn valid_redemption_passes() {
        assert_valid(validate_redemption(valid_redemption()));
    }

    #[test]
    fn redemption_zero_credits_rejected() {
        let mut r = valid_redemption();
        r.credits_redeemed = 0.0;
        assert_invalid(validate_redemption(r), "Credits redeemed must be positive");
    }

    #[test]
    fn redemption_negative_credits_rejected() {
        let mut r = valid_redemption();
        r.credits_redeemed = -1.0;
        assert_invalid(validate_redemption(r), "Credits redeemed must be positive");
    }

    #[test]
    fn redemption_barely_positive_valid() {
        let mut r = valid_redemption();
        r.credits_redeemed = 0.001;
        assert_valid(validate_redemption(r));
    }

    #[test]
    fn redemption_empty_redeemed_for_rejected() {
        let mut r = valid_redemption();
        r.redeemed_for = String::new();
        assert_invalid(
            validate_redemption(r),
            "Redeemed-for description cannot be empty",
        );
    }

    #[test]
    fn redemption_whitespace_redeemed_for_rejected() {
        let mut r = valid_redemption();
        r.redeemed_for = "  ".into();
        assert_invalid(
            validate_redemption(r),
            "Redeemed-for description cannot be empty",
        );
    }

    #[test]
    fn redemption_redeemed_for_too_long_rejected() {
        let mut r = valid_redemption();
        r.redeemed_for = "x".repeat(1025);
        assert_invalid(validate_redemption(r), "Redeemed-for description too long");
    }

    #[test]
    fn redemption_redeemed_for_at_max_valid() {
        let mut r = valid_redemption();
        r.redeemed_for = "x".repeat(1024);
        assert_valid(validate_redemption(r));
    }

    // ── NaN/Infinity bypass hardening tests ────────────────────────────

    #[test]
    fn trip_nan_distance_rejected() {
        let mut t = valid_trip();
        t.distance_km = f64::NAN;
        assert_invalid(validate_trip(t), "distance_km must be a finite number");
    }

    #[test]
    fn trip_infinity_distance_rejected() {
        let mut t = valid_trip();
        t.distance_km = f64::INFINITY;
        assert_invalid(validate_trip(t), "distance_km must be a finite number");
    }

    #[test]
    fn trip_neg_infinity_distance_rejected() {
        let mut t = valid_trip();
        t.distance_km = f64::NEG_INFINITY;
        assert_invalid(validate_trip(t), "distance_km must be a finite number");
    }

    #[test]
    fn trip_nan_cargo_rejected() {
        let mut t = valid_trip();
        t.cargo_kg = f64::NAN;
        assert_invalid(validate_trip(t), "cargo_kg must be a finite number");
    }

    #[test]
    fn trip_infinity_cargo_rejected() {
        let mut t = valid_trip();
        t.cargo_kg = f64::INFINITY;
        assert_invalid(validate_trip(t), "cargo_kg must be a finite number");
    }

    #[test]
    fn trip_nan_emissions_rejected() {
        let mut t = valid_trip();
        t.emissions_kg_co2 = f64::NAN;
        assert_invalid(validate_trip(t), "emissions_kg_co2 must be a finite number");
    }

    #[test]
    fn trip_infinity_emissions_rejected() {
        let mut t = valid_trip();
        t.emissions_kg_co2 = f64::INFINITY;
        assert_invalid(validate_trip(t), "emissions_kg_co2 must be a finite number");
    }

    #[test]
    fn credit_nan_rejected() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = f64::NAN;
        assert_invalid(validate_credit(c), "credits_kg_co2 must be a finite number");
    }

    #[test]
    fn credit_infinity_rejected() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = f64::INFINITY;
        assert_invalid(validate_credit(c), "credits_kg_co2 must be a finite number");
    }

    #[test]
    fn credit_neg_infinity_rejected() {
        let mut c = valid_carbon_credit();
        c.credits_kg_co2 = f64::NEG_INFINITY;
        assert_invalid(validate_credit(c), "credits_kg_co2 must be a finite number");
    }

    #[test]
    fn redemption_nan_credits_rejected() {
        let mut r = valid_redemption();
        r.credits_redeemed = f64::NAN;
        assert_invalid(
            validate_redemption(r),
            "credits_redeemed must be a finite number",
        );
    }

    #[test]
    fn redemption_infinity_credits_rejected() {
        let mut r = valid_redemption();
        r.credits_redeemed = f64::INFINITY;
        assert_invalid(
            validate_redemption(r),
            "credits_redeemed must be a finite number",
        );
    }

    #[test]
    fn redemption_neg_infinity_credits_rejected() {
        let mut r = valid_redemption();
        r.credits_redeemed = f64::NEG_INFINITY;
        assert_invalid(
            validate_redemption(r),
            "credits_redeemed must be a finite number",
        );
    }
}
