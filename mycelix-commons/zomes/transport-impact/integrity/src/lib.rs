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
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    TripLog(TripLog),
    CarbonCredit(CarbonCredit),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllTrips,
    AgentToTrip,
    AgentToCredit,
    VehicleToTrip,
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
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_trip(t: TripLog) -> ExternResult<ValidateCallbackResult> {
    if t.distance_km <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Distance must be positive".into()));
    }
    if t.emissions_kg_co2 < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Emissions cannot be negative".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_credit(c: CarbonCredit) -> ExternResult<ValidateCallbackResult> {
    if c.credits_kg_co2 <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Credits must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    #[test]
    fn valid_trip_passes() {
        let t = TripLog {
            vehicle_hash: None,
            route_hash: None,
            distance_km: 15.0,
            mode: TripMode::Cycling,
            passengers: 1,
            cargo_kg: 0.0,
            emissions_kg_co2: 0.0,
            logged_at: 1700000000,
        };
        assert_eq!(validate_trip(t).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn trip_zero_distance_rejected() {
        let t = TripLog {
            vehicle_hash: None,
            route_hash: None,
            distance_km: 0.0,
            mode: TripMode::Walking,
            passengers: 1,
            cargo_kg: 0.0,
            emissions_kg_co2: 0.0,
            logged_at: 1700000000,
        };
        assert!(matches!(validate_trip(t).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn trip_negative_distance_rejected() {
        let t = TripLog {
            vehicle_hash: None,
            route_hash: None,
            distance_km: -5.0,
            mode: TripMode::Driving,
            passengers: 1,
            cargo_kg: 0.0,
            emissions_kg_co2: 0.0,
            logged_at: 1700000000,
        };
        assert!(matches!(validate_trip(t).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn trip_negative_emissions_rejected() {
        let t = TripLog {
            vehicle_hash: None,
            route_hash: None,
            distance_km: 10.0,
            mode: TripMode::Driving,
            passengers: 1,
            cargo_kg: 0.0,
            emissions_kg_co2: -0.5,
            logged_at: 1700000000,
        };
        assert!(matches!(validate_trip(t).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn all_trip_modes_valid() {
        for mode in [TripMode::Driving, TripMode::Cycling, TripMode::Walking,
                      TripMode::Transit, TripMode::Carpool, TripMode::ElectricVehicle] {
            let t = TripLog {
                vehicle_hash: None,
                route_hash: None,
                distance_km: 10.0,
                mode,
                passengers: 1,
                cargo_kg: 0.0,
                emissions_kg_co2: 0.0,
                logged_at: 1700000000,
            };
            assert_eq!(validate_trip(t).unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn valid_credit_passes() {
        let c = CarbonCredit {
            holder: fake_agent(),
            credits_kg_co2: 2.1,
            earned_from: CreditSource::Cycling,
            earned_at: 1700000000,
        };
        assert_eq!(validate_credit(c).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn credit_zero_rejected() {
        let c = CarbonCredit {
            holder: fake_agent(),
            credits_kg_co2: 0.0,
            earned_from: CreditSource::Transit,
            earned_at: 1700000000,
        };
        assert!(matches!(validate_credit(c).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn credit_negative_rejected() {
        let c = CarbonCredit {
            holder: fake_agent(),
            credits_kg_co2: -1.0,
            earned_from: CreditSource::Carpool,
            earned_at: 1700000000,
        };
        assert!(matches!(validate_credit(c).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn all_credit_sources_valid() {
        for src in [CreditSource::Cycling, CreditSource::Transit, CreditSource::Carpool,
                     CreditSource::ElectricVehicle, CreditSource::Walking] {
            let c = CarbonCredit {
                holder: fake_agent(),
                credits_kg_co2: 1.0,
                earned_from: src,
                earned_at: 1700000000,
            };
            assert_eq!(validate_credit(c).unwrap(), ValidateCallbackResult::Valid);
        }
    }
}
