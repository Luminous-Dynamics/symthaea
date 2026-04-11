// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transport Impact Coordinator Zome
//! Business logic for trip logging, emissions calculation, and carbon credits.

use hdk::prelude::*;
use mycelix_bridge_common::civic_requirement_basic;
use mycelix_zome_helpers::records_from_links;
use transport_impact_integrity::*;

// ============================================================================
// BRIDGE SIGNAL (for cross-domain UI notification)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub event_type: String,
    pub source_zome: String,
    pub payload: String,
}


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Average CO2 emissions per km by mode (kg CO2/km)
fn emissions_factor(mode: &TripMode) -> f64 {
    match mode {
        TripMode::Driving => 0.21,         // Average car
        TripMode::ElectricVehicle => 0.05, // EV with average grid
        TripMode::Transit => 0.089,        // Bus/train average
        TripMode::Carpool => 0.07,         // Car split among passengers
        TripMode::Cycling => 0.0,
        TripMode::Walking => 0.0,
        TripMode::Flying => 0.255,      // Helicopter/eVTOL average
        TripMode::Water => 0.19,        // Ferry average
        TripMode::Rail => 0.041,        // Train/tram average
        TripMode::Micromobility => 0.0, // Human-powered / negligible electric
        TripMode::Autonomous => 0.05,   // Autonomous EV
    }
}

// ============================================================================
// TRIP LOGGING
// ============================================================================

#[hdk_extern]
pub fn log_trip(mut trip: TripLog) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "log_trip")?;
    let agent = agent_info()?.agent_initial_pubkey;

    // Auto-calculate emissions if not provided (zero means calculate)
    if trip.emissions_kg_co2 == 0.0 {
        trip.emissions_kg_co2 = calculate_trip_emissions(&trip);
    }

    let action_hash = create_entry(&EntryTypes::TripLog(trip.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_trips".to_string())))?;
    create_link(
        anchor_hash("all_trips")?,
        action_hash.clone(),
        LinkTypes::AllTrips,
        (),
    )?;
    create_link(
        agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToTrip,
        (),
    )?;

    if let Some(ref vehicle_hash) = trip.vehicle_hash {
        create_link(
            vehicle_hash.clone(),
            action_hash.clone(),
            LinkTypes::VehicleToTrip,
            (),
        )?;
    }

    // Award carbon credits for low-emission modes
    let baseline_emissions = trip.distance_km * 0.21; // baseline: solo car
    let saved = baseline_emissions - trip.emissions_kg_co2;
    if saved > 0.0 {
        let credit_source = match trip.mode {
            TripMode::Cycling | TripMode::Micromobility => CreditSource::Cycling,
            TripMode::Walking => CreditSource::Walking,
            TripMode::Transit | TripMode::Rail | TripMode::Water => CreditSource::Transit,
            TripMode::Carpool => CreditSource::Carpool,
            TripMode::ElectricVehicle | TripMode::Autonomous => CreditSource::ElectricVehicle,
            _ => CreditSource::Transit,
        };

        let credit = CarbonCredit {
            holder: agent.clone(),
            credits_kg_co2: saved,
            earned_from: credit_source,
            earned_at: trip.logged_at,
        };
        let credit_hash = create_entry(&EntryTypes::CarbonCredit(credit))?;
        create_link(agent, credit_hash, LinkTypes::AgentToCredit, ())?;

        // Emit bridge signal so UI can show the carbon credit award
        let _ = emit_signal(&BridgeEventSignal {
            event_type: "carbon_credits_awarded".to_string(),
            source_zome: "transport_impact".to_string(),
            payload: format!(
                r#"{{"trip_hash":"{}","distance_km":{},"mode":"{:?}","credits_kg_co2":{}}}"#,
                action_hash, trip.distance_km, trip.mode, saved,
            ),
        });
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created trip".into()
    )))
}

fn calculate_trip_emissions(trip: &TripLog) -> f64 {
    let base = trip.distance_km * emissions_factor(&trip.mode);
    // Adjust for passengers in carpool
    if trip.passengers > 1 && matches!(trip.mode, TripMode::Driving | TripMode::Carpool) {
        base / trip.passengers as f64
    } else {
        base
    }
}

#[hdk_extern]
pub fn get_my_trips(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToTrip)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn get_my_carbon_credits(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToCredit)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EmissionsCalcInput {
    pub distance_km: f64,
    pub mode: TripMode,
    pub passengers: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EmissionsCalcResult {
    pub emissions_kg_co2: f64,
    pub baseline_emissions: f64,
    pub savings_kg_co2: f64,
}

#[hdk_extern]
pub fn calculate_emissions(input: EmissionsCalcInput) -> ExternResult<EmissionsCalcResult> {
    let base = input.distance_km * emissions_factor(&input.mode);
    let emissions =
        if input.passengers > 1 && matches!(input.mode, TripMode::Driving | TripMode::Carpool) {
            base / input.passengers as f64
        } else {
            base
        };
    let baseline = input.distance_km * 0.21;

    Ok(EmissionsCalcResult {
        emissions_kg_co2: emissions,
        baseline_emissions: baseline,
        savings_kg_co2: (baseline - emissions).max(0.0),
    })
}

// ============================================================================
// CREDIT REDEMPTION
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RedeemInput {
    pub credits_redeemed: f64,
    pub redeemed_for: String,
}

#[hdk_extern]
pub fn redeem_credits(input: RedeemInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "redeem_credits")?;
    let agent = agent_info()?.agent_initial_pubkey;

    // Verify sufficient balance before redeeming
    let balance = get_agent_carbon_balance(agent.clone())?;
    if input.credits_redeemed > balance.balance {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Insufficient carbon credits: requested {}, available {}",
            input.credits_redeemed, balance.balance
        ))));
    }

    let now = sys_time()?.as_micros() / 1_000_000;

    let redemption = CreditRedemption {
        holder: agent.clone(),
        credits_redeemed: input.credits_redeemed,
        redeemed_for: input.redeemed_for,
        redeemed_at: now as u64,
    };

    let action_hash = create_entry(&EntryTypes::CreditRedemption(redemption))?;
    create_link(
        agent,
        action_hash.clone(),
        LinkTypes::AgentToRedemptions,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created redemption".into()
    )))
}

#[hdk_extern]
pub fn get_my_redemptions(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToRedemptions)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CarbonBalance {
    pub total_earned: f64,
    pub total_redeemed: f64,
    pub balance: f64,
}

#[hdk_extern]
pub fn get_agent_carbon_balance(agent: AgentPubKey) -> ExternResult<CarbonBalance> {
    // Sum earned credits
    let credit_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToCredit)?,
        GetStrategy::default(),
    )?;
    let mut total_earned = 0.0;
    for link in credit_links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Ok(Some(credit)) = record.entry().to_app_option::<CarbonCredit>() {
                total_earned += credit.credits_kg_co2;
            }
        }
    }

    // Sum redeemed credits
    let redemption_links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToRedemptions)?,
        GetStrategy::default(),
    )?;
    let mut total_redeemed = 0.0;
    for link in redemption_links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Ok(Some(redemption)) = record.entry().to_app_option::<CreditRedemption>() {
                total_redeemed += redemption.credits_redeemed;
            }
        }
    }

    Ok(CarbonBalance {
        total_earned,
        total_redeemed,
        balance: total_earned - total_redeemed,
    })
}

#[hdk_extern]
pub fn get_vehicle_trip_history(vehicle_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(vehicle_hash, LinkTypes::VehicleToTrip)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CommunityImpactSummary {
    pub total_trips: u32,
    pub total_distance_km: f64,
    pub total_emissions_kg_co2: f64,
    pub total_credits_earned: f64,
}

#[hdk_extern]
pub fn get_community_impact_summary(_: ()) -> ExternResult<CommunityImpactSummary> {
    let trip_links = get_links(
        LinkQuery::try_new(anchor_hash("all_trips")?, LinkTypes::AllTrips)?,
        GetStrategy::default(),
    )?;

    let mut total_distance = 0.0;
    let mut total_emissions = 0.0;
    let mut trip_count: u32 = 0;

    for link in trip_links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Ok(Some(trip)) = record.entry().to_app_option::<TripLog>() {
                total_distance += trip.distance_km;
                total_emissions += trip.emissions_kg_co2;
                trip_count += 1;
            }
        }
    }

    let baseline = total_distance * 0.21;
    let credits = (baseline - total_emissions).max(0.0);

    Ok(CommunityImpactSummary {
        total_trips: trip_count,
        total_distance_km: total_distance,
        total_emissions_kg_co2: total_emissions,
        total_credits_earned: credits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn make_trip(mode: TripMode, distance: f64, passengers: u32) -> TripLog {
        TripLog {
            vehicle_hash: None,
            route_hash: None,
            distance_km: distance,
            mode,
            passengers,
            cargo_kg: 0.0,
            emissions_kg_co2: 0.0,
            logged_at: 0,
        }
    }

    // ========================================================================
    // Pure function: emissions_factor tests
    // ========================================================================

    #[test]
    fn emissions_factor_driving() {
        assert!((emissions_factor(&TripMode::Driving) - 0.21).abs() < f64::EPSILON);
    }

    #[test]
    fn emissions_factor_cycling_is_zero() {
        assert_eq!(emissions_factor(&TripMode::Cycling), 0.0);
    }

    #[test]
    fn emissions_factor_walking_is_zero() {
        assert_eq!(emissions_factor(&TripMode::Walking), 0.0);
    }

    #[test]
    fn emissions_factor_ev_lower_than_driving() {
        assert!(
            emissions_factor(&TripMode::ElectricVehicle) < emissions_factor(&TripMode::Driving)
        );
    }

    #[test]
    fn emissions_factor_carpool_lower_than_driving() {
        assert!(emissions_factor(&TripMode::Carpool) < emissions_factor(&TripMode::Driving));
    }

    #[test]
    fn emissions_factor_transit_value() {
        assert!((emissions_factor(&TripMode::Transit) - 0.089).abs() < f64::EPSILON);
    }

    #[test]
    fn emissions_factor_ev_value() {
        assert!((emissions_factor(&TripMode::ElectricVehicle) - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn emissions_factor_carpool_value() {
        assert!((emissions_factor(&TripMode::Carpool) - 0.07).abs() < f64::EPSILON);
    }

    #[test]
    fn emissions_factor_flying_value() {
        assert!((emissions_factor(&TripMode::Flying) - 0.255).abs() < f64::EPSILON);
    }

    #[test]
    fn emissions_factor_water_value() {
        assert!((emissions_factor(&TripMode::Water) - 0.19).abs() < f64::EPSILON);
    }

    #[test]
    fn emissions_factor_rail_value() {
        assert!((emissions_factor(&TripMode::Rail) - 0.041).abs() < f64::EPSILON);
    }

    #[test]
    fn emissions_factor_micromobility_is_zero() {
        assert_eq!(emissions_factor(&TripMode::Micromobility), 0.0);
    }

    #[test]
    fn emissions_factor_autonomous_value() {
        assert!((emissions_factor(&TripMode::Autonomous) - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn emissions_factor_flying_higher_than_driving() {
        assert!(emissions_factor(&TripMode::Flying) > emissions_factor(&TripMode::Driving));
    }

    #[test]
    fn emissions_factor_rail_lower_than_driving() {
        assert!(emissions_factor(&TripMode::Rail) < emissions_factor(&TripMode::Driving));
    }

    // ========================================================================
    // Pure function: calculate_trip_emissions tests
    // ========================================================================

    #[test]
    fn calculate_trip_emissions_cycling() {
        let trip = make_trip(TripMode::Cycling, 20.0, 1);
        assert_eq!(calculate_trip_emissions(&trip), 0.0);
    }

    #[test]
    fn calculate_trip_emissions_walking() {
        let trip = make_trip(TripMode::Walking, 5.0, 1);
        assert_eq!(calculate_trip_emissions(&trip), 0.0);
    }

    #[test]
    fn calculate_trip_emissions_solo_driving() {
        let trip = make_trip(TripMode::Driving, 100.0, 1);
        let emissions = calculate_trip_emissions(&trip);
        assert!((emissions - 21.0).abs() < 0.01); // 100km * 0.21
    }

    #[test]
    fn calculate_trip_emissions_carpool_splits() {
        let trip = make_trip(TripMode::Carpool, 100.0, 4);
        let emissions = calculate_trip_emissions(&trip);
        // 100km * 0.07 / 4 passengers = 1.75
        assert!((emissions - 1.75).abs() < 0.01);
    }

    #[test]
    fn calculate_trip_emissions_driving_multi_passenger_splits() {
        let trip = make_trip(TripMode::Driving, 100.0, 2);
        let emissions = calculate_trip_emissions(&trip);
        // 100km * 0.21 / 2 = 10.5
        assert!((emissions - 10.5).abs() < 0.01);
    }

    #[test]
    fn calculate_trip_emissions_transit_no_split() {
        let trip = make_trip(TripMode::Transit, 50.0, 3);
        let emissions = calculate_trip_emissions(&trip);
        // Transit doesn't split by passengers: 50 * 0.089 = 4.45
        assert!((emissions - 4.45).abs() < 0.01);
    }

    #[test]
    fn calculate_trip_emissions_ev_no_split_with_passengers() {
        // EV with multiple passengers should NOT split (not Driving/Carpool)
        let trip = make_trip(TripMode::ElectricVehicle, 100.0, 4);
        let emissions = calculate_trip_emissions(&trip);
        // 100km * 0.05 = 5.0 (no split)
        assert!((emissions - 5.0).abs() < 0.01);
    }

    #[test]
    fn calculate_trip_emissions_zero_distance() {
        let trip = make_trip(TripMode::Driving, 0.0, 1);
        assert_eq!(calculate_trip_emissions(&trip), 0.0);
    }

    #[test]
    fn calculate_trip_emissions_very_small_distance() {
        let trip = make_trip(TripMode::Driving, 0.001, 1);
        let emissions = calculate_trip_emissions(&trip);
        assert!((emissions - 0.00021).abs() < 1e-10);
    }

    #[test]
    fn calculate_trip_emissions_driving_single_passenger_no_split() {
        // passengers == 1 should NOT trigger split even for Driving
        let trip = make_trip(TripMode::Driving, 10.0, 1);
        let emissions = calculate_trip_emissions(&trip);
        assert!((emissions - 2.1).abs() < 0.01);
    }

    // ========================================================================
    // Coordinator input/output struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn emissions_calc_input_serde_roundtrip() {
        let input = EmissionsCalcInput {
            distance_km: 42.5,
            mode: TripMode::Transit,
            passengers: 1,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: EmissionsCalcInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.distance_km, 42.5);
        assert_eq!(decoded.mode, TripMode::Transit);
        assert_eq!(decoded.passengers, 1);
    }

    #[test]
    fn emissions_calc_input_all_modes() {
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
            let input = EmissionsCalcInput {
                distance_km: 10.0,
                mode: mode.clone(),
                passengers: 1,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: EmissionsCalcInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.mode, mode);
        }
    }

    #[test]
    fn emissions_calc_result_serde_roundtrip() {
        let result = EmissionsCalcResult {
            emissions_kg_co2: 2.1,
            baseline_emissions: 2.1,
            savings_kg_co2: 0.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: EmissionsCalcResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.emissions_kg_co2, 2.1);
        assert_eq!(decoded.savings_kg_co2, 0.0);
    }

    #[test]
    fn emissions_calc_result_with_savings() {
        let result = EmissionsCalcResult {
            emissions_kg_co2: 0.0,
            baseline_emissions: 2.1,
            savings_kg_co2: 2.1,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: EmissionsCalcResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.savings_kg_co2, 2.1);
    }

    #[test]
    fn community_impact_summary_serde_roundtrip() {
        let summary = CommunityImpactSummary {
            total_trips: 100,
            total_distance_km: 5000.0,
            total_emissions_kg_co2: 800.0,
            total_credits_earned: 250.0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: CommunityImpactSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_trips, 100);
        assert_eq!(decoded.total_distance_km, 5000.0);
        assert_eq!(decoded.total_emissions_kg_co2, 800.0);
        assert_eq!(decoded.total_credits_earned, 250.0);
    }

    #[test]
    fn community_impact_summary_zero_values() {
        let summary = CommunityImpactSummary {
            total_trips: 0,
            total_distance_km: 0.0,
            total_emissions_kg_co2: 0.0,
            total_credits_earned: 0.0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: CommunityImpactSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_trips, 0);
    }

    #[test]
    fn bridge_event_signal_serde_roundtrip() {
        let signal = BridgeEventSignal {
            event_type: "carbon_credits_awarded".to_string(),
            source_zome: "transport_impact".to_string(),
            payload: r#"{"credits":1.5}"#.to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.event_type, "carbon_credits_awarded");
        assert_eq!(decoded.source_zome, "transport_impact");
        assert!(decoded.payload.contains("credits"));
    }

    // ========================================================================
    // Integrity entry struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn trip_log_serde_roundtrip_with_hashes() {
        let trip = TripLog {
            vehicle_hash: Some(fake_hash()),
            route_hash: Some(fake_hash()),
            distance_km: 15.0,
            mode: TripMode::Driving,
            passengers: 2,
            cargo_kg: 50.0,
            emissions_kg_co2: 1.575,
            logged_at: 1700000000,
        };
        let json = serde_json::to_string(&trip).unwrap();
        let decoded: TripLog = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, trip);
    }

    #[test]
    fn trip_log_serde_roundtrip_no_hashes() {
        let trip = make_trip(TripMode::Walking, 3.0, 1);
        let json = serde_json::to_string(&trip).unwrap();
        let decoded: TripLog = serde_json::from_str(&json).unwrap();
        assert!(decoded.vehicle_hash.is_none());
        assert!(decoded.route_hash.is_none());
    }

    #[test]
    fn carbon_credit_serde_roundtrip() {
        let credit = CarbonCredit {
            holder: fake_agent(),
            credits_kg_co2: 3.15,
            earned_from: CreditSource::Cycling,
            earned_at: 1700000000,
        };
        let json = serde_json::to_string(&credit).unwrap();
        let decoded: CarbonCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, credit);
    }

    #[test]
    fn carbon_credit_all_sources() {
        for source in [
            CreditSource::Cycling,
            CreditSource::Transit,
            CreditSource::Carpool,
            CreditSource::ElectricVehicle,
            CreditSource::Walking,
        ] {
            let credit = CarbonCredit {
                holder: fake_agent(),
                credits_kg_co2: 1.0,
                earned_from: source.clone(),
                earned_at: 0,
            };
            let json = serde_json::to_string(&credit).unwrap();
            let decoded: CarbonCredit = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.earned_from, source);
        }
    }

    // ========================================================================
    // Clone / equality tests
    // ========================================================================

    #[test]
    fn trip_log_clone_equals() {
        let trip = make_trip(TripMode::Transit, 25.0, 1);
        let cloned = trip.clone();
        assert_eq!(trip, cloned);
    }

    #[test]
    fn trip_mode_clone_eq() {
        let mode = TripMode::ElectricVehicle;
        assert_eq!(mode.clone(), TripMode::ElectricVehicle);
    }

    // ========================================================================
    // Edge case / boundary value tests
    // ========================================================================

    #[test]
    fn emissions_calc_input_u32_max_passengers() {
        let input = EmissionsCalcInput {
            distance_km: 10.0,
            mode: TripMode::Carpool,
            passengers: u32::MAX,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: EmissionsCalcInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.passengers, u32::MAX);
    }

    #[test]
    fn community_impact_summary_u32_max_trips() {
        let summary = CommunityImpactSummary {
            total_trips: u32::MAX,
            total_distance_km: f64::MAX,
            total_emissions_kg_co2: f64::MAX,
            total_credits_earned: f64::MAX,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: CommunityImpactSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_trips, u32::MAX);
    }

    #[test]
    fn bridge_event_signal_empty_fields() {
        let signal = BridgeEventSignal {
            event_type: String::new(),
            source_zome: String::new(),
            payload: String::new(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert!(decoded.event_type.is_empty());
        assert!(decoded.source_zome.is_empty());
        assert!(decoded.payload.is_empty());
    }

    // ========================================================================
    // RedeemInput serde roundtrip
    // ========================================================================

    #[test]
    fn redeem_input_serde_roundtrip() {
        let input = RedeemInput {
            credits_redeemed: 5.0,
            redeemed_for: "Transit pass discount".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RedeemInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.credits_redeemed, 5.0);
        assert_eq!(decoded.redeemed_for, "Transit pass discount");
    }

    #[test]
    fn redeem_input_serde_small_credits() {
        let input = RedeemInput {
            credits_redeemed: 0.001,
            redeemed_for: "Micro reward".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RedeemInput = serde_json::from_str(&json).unwrap();
        assert!((decoded.credits_redeemed - 0.001).abs() < 1e-9);
    }

    // ========================================================================
    // CarbonBalance serde roundtrip
    // ========================================================================

    #[test]
    fn carbon_balance_serde_roundtrip() {
        let balance = CarbonBalance {
            total_earned: 100.0,
            total_redeemed: 30.0,
            balance: 70.0,
        };
        let json = serde_json::to_string(&balance).unwrap();
        let decoded: CarbonBalance = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_earned, 100.0);
        assert_eq!(decoded.total_redeemed, 30.0);
        assert_eq!(decoded.balance, 70.0);
    }

    #[test]
    fn carbon_balance_serde_zero_balance() {
        let balance = CarbonBalance {
            total_earned: 0.0,
            total_redeemed: 0.0,
            balance: 0.0,
        };
        let json = serde_json::to_string(&balance).unwrap();
        let decoded: CarbonBalance = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance, 0.0);
    }

    #[test]
    fn carbon_balance_serde_negative_balance() {
        let balance = CarbonBalance {
            total_earned: 10.0,
            total_redeemed: 15.0,
            balance: -5.0,
        };
        let json = serde_json::to_string(&balance).unwrap();
        let decoded: CarbonBalance = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance, -5.0);
    }

    // ========================================================================
    // CreditRedemption serde roundtrip
    // ========================================================================

    #[test]
    fn credit_redemption_serde_roundtrip() {
        let r = CreditRedemption {
            holder: fake_agent(),
            credits_redeemed: 10.5,
            redeemed_for: "Bike share membership".to_string(),
            redeemed_at: 1700000000,
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: CreditRedemption = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, r);
    }

    #[test]
    fn credit_redemption_serde_max_u64_time() {
        let r = CreditRedemption {
            holder: fake_agent(),
            credits_redeemed: 1.0,
            redeemed_for: "Test".to_string(),
            redeemed_at: u64::MAX,
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: CreditRedemption = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.redeemed_at, u64::MAX);
    }
}
