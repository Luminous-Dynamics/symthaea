// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;

// ─── Entry Types ─────────────────────────────────────────────────

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GeothermalNode {
    pub name: String,
    pub region: String,
    pub lat: f64,
    pub lon: f64,
    pub capacity_mw: f64,
    pub temperature_c: u32,
    pub node_type: String, // HighEnthalpy, SuperhotRock, EnhancedGeothermal
    pub status: String,    // Operational, UnderDevelopment, Planned, Potential
    pub created: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MaglevCorridor {
    pub name: String,
    pub from_name: String,
    pub from_lat: f64,
    pub from_lon: f64,
    pub to_name: String,
    pub to_lat: f64,
    pub to_lon: f64,
    pub distance_km: f64,
    pub travel_time_min: f64,
    pub submarine: bool,
    pub seismic_risk: String, // Low, Medium, High, Extreme
    pub cost_billion_usd: f64,
    pub capacity_pax_hr: u32,
    pub geothermal_powered: bool,
    pub blast_doors: u32,
    pub build_years: u32,
    pub created: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ResontiaVault {
    pub vault_id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub capacity_people: u32,
    pub heat_rejection_mw: f64,
    pub blast_doors: u32,
    pub status: String, // Planned, UnderConstruction, Operational, Dormant
    pub terra_lumina_id: String,
    pub created: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TerraLuminaSite {
    pub site_id: String,
    pub name: String,
    pub country: String,
    pub lat: f64,
    pub lon: f64,
    pub score: u32,
    pub tier: String, // Ultimate, Premium, Standard
    pub geothermal_gw: f64,
    pub solar_gw: f64,
    pub hydro_gw: f64,
    pub total_renewable_gw: f64,
    pub phase1_billion_eur: f64,
    pub total_billion_eur: f64,
    pub irr_percent: f64,
    pub created: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FossilDeposit {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub fuel_type: String,              // Oil, Gas, Coal, TarSands
    pub proven_reserves_mboe: f64,      // million barrels oil equivalent
    pub annual_production_mboe: f64,
    pub status: String,                 // Producing, Declining, Depleted, Undeveloped
    pub country: String,
    pub discovery_year: u32,
    pub created: Timestamp,
}

// ─── Entry & Link Registration ──────────────────────────────────

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    GeothermalNode(GeothermalNode),
    MaglevCorridor(MaglevCorridor),
    ResontiaVault(ResontiaVault),
    TerraLuminaSite(TerraLuminaSite),
    FossilDeposit(FossilDeposit),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllGeothermalNodes,
    AllCorridors,
    AllVaults,
    AllTerraLuminaSites,
    AllFossilDeposits,
    TypeToFossilDeposits,
    NodeToCorridors,
    VaultToCorridors,
    RegionToInfrastructure,
}

// ─── Validation ──────────────────────────────────────────────────

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::GeothermalNode(node) => validate_coordinates(node.lat, node.lon),
                EntryTypes::MaglevCorridor(corridor) => {
                    if let ValidateCallbackResult::Invalid(msg) = validate_coordinates(corridor.from_lat, corridor.from_lon)? {
                        return Ok(ValidateCallbackResult::Invalid(msg));
                    }
                    if let ValidateCallbackResult::Invalid(msg) = validate_coordinates(corridor.to_lat, corridor.to_lon)? {
                        return Ok(ValidateCallbackResult::Invalid(msg));
                    }
                    if corridor.distance_km <= 0.0 {
                        return Ok(ValidateCallbackResult::Invalid("Distance must be positive".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::ResontiaVault(vault) => {
                    if let ValidateCallbackResult::Invalid(msg) = validate_coordinates(vault.lat, vault.lon)? {
                        return Ok(ValidateCallbackResult::Invalid(msg));
                    }
                    if vault.capacity_people == 0 {
                        return Ok(ValidateCallbackResult::Invalid("Vault must have capacity > 0".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::TerraLuminaSite(site) => {
                    if let ValidateCallbackResult::Invalid(msg) = validate_coordinates(site.lat, site.lon)? {
                        return Ok(ValidateCallbackResult::Invalid(msg));
                    }
                    if site.score > 100 {
                        return Ok(ValidateCallbackResult::Invalid("Score must be 0-100".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::FossilDeposit(deposit) => validate_fossil_deposit(deposit),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_fossil_deposit(d: FossilDeposit) -> ExternResult<ValidateCallbackResult> {
    let coord_result = validate_coordinates(d.lat, d.lon)?;
    if let ValidateCallbackResult::Invalid(msg) = coord_result {
        return Ok(ValidateCallbackResult::Invalid(msg));
    }
    if d.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Name must not be empty".into()));
    }
    let valid_types = ["Oil", "Gas", "Coal", "TarSands"];
    if !valid_types.contains(&d.fuel_type.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Invalid fuel type: {}. Must be one of: {:?}", d.fuel_type, valid_types),
        ));
    }
    let valid_status = ["Producing", "Declining", "Depleted", "Undeveloped"];
    if !valid_status.contains(&d.status.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Invalid status: {}. Must be one of: {:?}", d.status, valid_status),
        ));
    }
    if d.proven_reserves_mboe < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Reserves must be non-negative".into()));
    }
    if d.annual_production_mboe < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Production must be non-negative".into()));
    }
    if d.discovery_year < 1700 || d.discovery_year > 2100 {
        return Ok(ValidateCallbackResult::Invalid("Discovery year must be 1700-2100".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_coordinates(lat: f64, lon: f64) -> ExternResult<ValidateCallbackResult> {
    if lat < -90.0 || lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid("Latitude must be between -90 and 90".into()));
    }
    if lon < -180.0 || lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid("Longitude must be between -180 and 180".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Coordinate Validation ──────────────────────────────────────

    #[test]
    fn valid_coordinates_equator_prime_meridian() {
        let result = validate_coordinates(0.0, 0.0).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn valid_coordinates_boundaries() {
        assert_eq!(validate_coordinates(90.0, 180.0).unwrap(), ValidateCallbackResult::Valid);
        assert_eq!(validate_coordinates(-90.0, -180.0).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn invalid_latitude_too_high() {
        let result = validate_coordinates(90.1, 0.0).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn invalid_latitude_too_low() {
        let result = validate_coordinates(-90.1, 0.0).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn invalid_longitude_too_high() {
        let result = validate_coordinates(0.0, 180.1).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn invalid_longitude_too_low() {
        let result = validate_coordinates(0.0, -180.1).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ─── Fossil Deposit Validation ──────────────────────────────────

    fn make_deposit(overrides: impl FnOnce(&mut FossilDeposit)) -> FossilDeposit {
        let mut d = FossilDeposit {
            name: "Ghawar Field".into(),
            lat: 25.4,
            lon: 49.6,
            fuel_type: "Oil".into(),
            proven_reserves_mboe: 100_000.0,
            annual_production_mboe: 3_500.0,
            status: "Producing".into(),
            country: "Saudi Arabia".into(),
            discovery_year: 1948,
            created: Timestamp::from_micros(0),
        };
        overrides(&mut d);
        d
    }

    #[test]
    fn valid_fossil_deposit() {
        let d = make_deposit(|_| {});
        assert_eq!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn fossil_deposit_empty_name_rejected() {
        let d = make_deposit(|d| d.name = String::new());
        assert!(matches!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn fossil_deposit_invalid_fuel_type_rejected() {
        let d = make_deposit(|d| d.fuel_type = "Uranium".into());
        assert!(matches!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn fossil_deposit_all_valid_fuel_types() {
        for fuel in &["Oil", "Gas", "Coal", "TarSands"] {
            let d = make_deposit(|d| d.fuel_type = fuel.to_string());
            assert_eq!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Valid,
                "fuel_type '{}' should be valid", fuel);
        }
    }

    #[test]
    fn fossil_deposit_invalid_status_rejected() {
        let d = make_deposit(|d| d.status = "Active".into());
        assert!(matches!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn fossil_deposit_all_valid_statuses() {
        for status in &["Producing", "Declining", "Depleted", "Undeveloped"] {
            let d = make_deposit(|d| d.status = status.to_string());
            assert_eq!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Valid,
                "status '{}' should be valid", status);
        }
    }

    #[test]
    fn fossil_deposit_negative_reserves_rejected() {
        let d = make_deposit(|d| d.proven_reserves_mboe = -1.0);
        assert!(matches!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn fossil_deposit_negative_production_rejected() {
        let d = make_deposit(|d| d.annual_production_mboe = -0.1);
        assert!(matches!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn fossil_deposit_discovery_year_too_early() {
        let d = make_deposit(|d| d.discovery_year = 1699);
        assert!(matches!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn fossil_deposit_discovery_year_too_late() {
        let d = make_deposit(|d| d.discovery_year = 2101);
        assert!(matches!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn fossil_deposit_boundary_years_valid() {
        let d1 = make_deposit(|d| d.discovery_year = 1700);
        assert_eq!(validate_fossil_deposit(d1).unwrap(), ValidateCallbackResult::Valid);
        let d2 = make_deposit(|d| d.discovery_year = 2100);
        assert_eq!(validate_fossil_deposit(d2).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn fossil_deposit_zero_reserves_valid() {
        let d = make_deposit(|d| d.proven_reserves_mboe = 0.0);
        assert_eq!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn fossil_deposit_invalid_lat_rejected() {
        let d = make_deposit(|d| d.lat = 100.0);
        assert!(matches!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn fossil_deposit_invalid_lon_rejected() {
        let d = make_deposit(|d| d.lon = -200.0);
        assert!(matches!(validate_fossil_deposit(d).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    // ─── Entry Type Construction ────────────────────────────────────

    #[test]
    fn geothermal_node_construction() {
        let node = GeothermalNode {
            name: "Olkaria".into(),
            region: "East Africa".into(),
            lat: -0.89,
            lon: 36.29,
            capacity_mw: 800.0,
            temperature_c: 350,
            node_type: "HighEnthalpy".into(),
            status: "Operational".into(),
            created: Timestamp::from_micros(0),
        };
        assert_eq!(node.name, "Olkaria");
        assert_eq!(node.capacity_mw, 800.0);
    }

    #[test]
    fn maglev_corridor_construction() {
        let corridor = MaglevCorridor {
            name: "Cape Town - Nairobi".into(),
            from_name: "Cape Town".into(),
            from_lat: -33.9,
            from_lon: 18.4,
            to_name: "Nairobi".into(),
            to_lat: -1.3,
            to_lon: 36.8,
            distance_km: 4800.0,
            travel_time_min: 120.0,
            submarine: false,
            seismic_risk: "Medium".into(),
            cost_billion_usd: 180.0,
            capacity_pax_hr: 2000,
            geothermal_powered: true,
            blast_doors: 48,
            build_years: 12,
            created: Timestamp::from_micros(0),
        };
        assert!(corridor.geothermal_powered);
        assert!(!corridor.submarine);
        assert_eq!(corridor.blast_doors, 48);
    }

    #[test]
    fn resontia_vault_construction() {
        let vault = ResontiaVault {
            vault_id: "RV-EC-001".into(),
            name: "Ecuador Prime".into(),
            lat: -2.2,
            lon: -79.9,
            capacity_people: 10_000,
            heat_rejection_mw: 50.0,
            blast_doors: 6,
            status: "Planned".into(),
            terra_lumina_id: "TL003".into(),
            created: Timestamp::from_micros(0),
        };
        assert_eq!(vault.capacity_people, 10_000);
    }

    #[test]
    fn terra_lumina_site_construction() {
        let site = TerraLuminaSite {
            site_id: "TL003".into(),
            name: "Ecuador".into(),
            country: "Ecuador".into(),
            lat: -2.2,
            lon: -79.9,
            score: 98,
            tier: "Ultimate".into(),
            geothermal_gw: 5.0,
            solar_gw: 3.0,
            hydro_gw: 4.0,
            total_renewable_gw: 12.0,
            phase1_billion_eur: 15.0,
            total_billion_eur: 85.0,
            irr_percent: 12.5,
            created: Timestamp::from_micros(0),
        };
        assert_eq!(site.score, 98);
        assert_eq!(site.total_renewable_gw, 12.0);
    }

    // ─── Serde Roundtrips ───────────────────────────────────────────

    #[test]
    fn geothermal_node_serde_roundtrip() {
        let node = GeothermalNode {
            name: "Test".into(),
            region: "Test".into(),
            lat: 45.0,
            lon: 90.0,
            capacity_mw: 100.0,
            temperature_c: 200,
            node_type: "SuperhotRock".into(),
            status: "Planned".into(),
            created: Timestamp::from_micros(12345),
        };
        let json = serde_json::to_string(&node).unwrap();
        let restored: GeothermalNode = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, node);
    }

    #[test]
    fn fossil_deposit_serde_roundtrip() {
        let d = make_deposit(|_| {});
        let json = serde_json::to_string(&d).unwrap();
        let restored: FossilDeposit = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.name, d.name);
        assert_eq!(restored.fuel_type, d.fuel_type);
    }
}
