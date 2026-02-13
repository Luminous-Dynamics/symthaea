//! Transport Impact Coordinator Zome
//! Business logic for trip logging, emissions calculation, and carbon credits.

use transport_impact_integrity::*;
use hdk::prelude::*;

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

/// Average CO2 emissions per km by mode (kg CO2/km)
fn emissions_factor(mode: &TripMode) -> f64 {
    match mode {
        TripMode::Driving => 0.21,          // Average car
        TripMode::ElectricVehicle => 0.05,   // EV with average grid
        TripMode::Transit => 0.089,          // Bus/train average
        TripMode::Carpool => 0.07,           // Car split among passengers
        TripMode::Cycling => 0.0,
        TripMode::Walking => 0.0,
    }
}

// ============================================================================
// TRIP LOGGING
// ============================================================================

#[hdk_extern]
pub fn log_trip(mut trip: TripLog) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Auto-calculate emissions if not provided (zero means calculate)
    if trip.emissions_kg_co2 == 0.0 {
        trip.emissions_kg_co2 = calculate_trip_emissions(&trip);
    }

    let action_hash = create_entry(&EntryTypes::TripLog(trip.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_trips".to_string())))?;
    create_link(anchor_hash("all_trips")?, action_hash.clone(), LinkTypes::AllTrips, ())?;
    create_link(agent.clone(), action_hash.clone(), LinkTypes::AgentToTrip, ())?;

    if let Some(ref vehicle_hash) = trip.vehicle_hash {
        create_link(vehicle_hash.clone(), action_hash.clone(), LinkTypes::VehicleToTrip, ())?;
    }

    // Award carbon credits for low-emission modes
    let baseline_emissions = trip.distance_km * 0.21; // baseline: solo car
    let saved = baseline_emissions - trip.emissions_kg_co2;
    if saved > 0.0 {
        let credit_source = match trip.mode {
            TripMode::Cycling => CreditSource::Cycling,
            TripMode::Walking => CreditSource::Walking,
            TripMode::Transit => CreditSource::Transit,
            TripMode::Carpool => CreditSource::Carpool,
            TripMode::ElectricVehicle => CreditSource::ElectricVehicle,
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
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created trip".into())))
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
    let emissions = if input.passengers > 1 && matches!(input.mode, TripMode::Driving | TripMode::Carpool) {
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
