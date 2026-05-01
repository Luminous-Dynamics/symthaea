// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Commons shared types — mirrors integrity zome entries without HDI dependency.

use serde::{Deserialize, Serialize};

// ── Property ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Property {
    pub id: String,
    pub property_type: PropertyType,
    pub title: String,
    pub description: String,
    pub owner_did: String,
    pub co_owners: Vec<CoOwner>,
    pub geolocation: Option<GeoLocation>,
    pub registered: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PropertyType {
    Land,
    Building,
    Unit,
    Equipment,
    Intellectual,
    Digital,
    Other,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoOwner {
    pub did: String,
    pub share_bps: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoLocation {
    pub lat: f64,
    pub lon: f64,
    pub area_sqm: Option<f64>,
}

// ── Housing ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Building {
    pub id: String,
    pub name: String,
    pub address: String,
    pub location_lat: f64,
    pub location_lon: f64,
    pub total_units: u16,
    pub year_built: Option<u16>,
    pub building_type: BuildingType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BuildingType {
    Apartment,
    Townhouse,
    SingleFamily,
    Duplex,
    CoHousing,
    MixedUse,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Unit {
    pub unit_number: String,
    pub unit_type: UnitType,
    pub square_meters: u32,
    pub floor: u8,
    pub bedrooms: u8,
    pub bathrooms: u8,
    pub status: UnitStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum UnitType {
    Studio,
    OneBedroom,
    TwoBedroom,
    ThreeBedroom,
    FourPlus,
    Accessible,
    Family,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum UnitStatus {
    Available,
    Occupied,
    UnderMaintenance,
    Reserved,
    Renovation,
}

// ── Care ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarePlan {
    pub title: String,
    pub description: String,
    pub care_type: CareType,
    pub schedule: String,
    pub status: PlanStatus,
    pub hours_per_week: f32,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CareType {
    Childcare,
    Eldercare,
    DisabilitySupport,
    PostSurgery,
    MentalHealth,
    Respite,
    Other,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PlanStatus {
    Draft,
    Active,
    Paused,
    Completed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CareSession {
    pub started_at: i64,
    pub ended_at: i64,
    pub hours: f32,
    pub notes: String,
    pub tasks_completed: Vec<String>,
}

// ── Mutual Aid ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AidRequest {
    pub id: String,
    pub requester_did: String,
    pub request_type: AidRequestType,
    pub description: String,
    pub urgency: Urgency,
    pub location: Option<String>,
    pub amount_needed: Option<u64>,
    pub fulfilled_amount: u64,
    pub status: AidRequestStatus,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AidRequestType {
    Financial,
    Housing,
    Food,
    Medical,
    Childcare,
    Transportation,
    Legal,
    Other,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Urgency {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AidRequestStatus {
    Open,
    PartiallyFulfilled,
    Fulfilled,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AidOffer {
    pub id: String,
    pub request_id: String,
    pub offerer_did: String,
    pub amount: Option<u64>,
    pub message: String,
    pub status: OfferStatus,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OfferStatus {
    Pending,
    Accepted,
    Completed,
    Withdrawn,
}

// ── Water ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterSource {
    pub id: String,
    pub name: String,
    pub source_type: WaterSourceType,
    pub max_capacity_liters: u64,
    pub recharge_rate_liters_per_day: u64,
    pub location_lat: f64,
    pub location_lon: f64,
    pub status: SourceStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum WaterSourceType {
    Municipal,
    Well,
    Spring,
    Rainwater,
    Aquifer,
    River,
    Lake,
    Recycled,
    Desalinated,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SourceStatus {
    Active,
    Seasonal,
    Depleted,
    Contaminated,
    UnderMaintenance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterShare {
    pub holder_did: String,
    pub allocation_type: AllocationType,
    pub volume_per_period_liters: u64,
    pub period_days: u32,
    pub priority: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AllocationType {
    Fixed,
    Proportional,
    Priority,
    Emergency,
}

// ── Food ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoodListing {
    pub product_name: String,
    pub quantity_kg: f64,
    pub price_per_kg: f64,
    pub available_from: i64,
    pub status: ListingStatus,
    pub organic: bool,
    pub allergen_flags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ListingStatus {
    Available,
    Reserved,
    Sold,
    Expired,
}

// ── Transport ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Route {
    pub id: String,
    pub name: String,
    pub waypoints: Vec<Waypoint>,
    pub distance_km: f64,
    pub estimated_minutes: u32,
    pub mode: TransportMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Waypoint {
    pub lat: f64,
    pub lon: f64,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TransportMode {
    Driving,
    Cycling,
    Walking,
    Transit,
    Mixed,
    Flying,
    Water,
    Rail,
    Micromobility,
    Autonomous,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aid_request_roundtrip() {
        let req = AidRequest {
            id: "ar-1".into(),
            requester_did: "did:t".into(),
            request_type: AidRequestType::Food,
            description: "test".into(),
            urgency: Urgency::High,
            location: None,
            amount_needed: Some(100),
            fulfilled_amount: 0,
            status: AidRequestStatus::Open,
            created_at: 0,
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let decoded: AidRequest = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.urgency, Urgency::High);
    }
}
