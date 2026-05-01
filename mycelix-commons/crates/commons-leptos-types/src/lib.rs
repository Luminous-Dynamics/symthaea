// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe view types for the Mycelix Commons frontend.
//!
//! These are lightweight serde-only UI models shared by the standalone
//! Commons app and any shell surfaces such as Portal.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeedCategory {
    Housing,
    Childcare,
    Transportation,
    Food,
    Skills,
    Healthcare,
    Other(String),
}

impl NeedCategory {
    pub fn label(&self) -> &str {
        match self {
            Self::Housing => "Housing",
            Self::Childcare => "Childcare",
            Self::Transportation => "Transportation",
            Self::Food => "Food",
            Self::Skills => "Skills",
            Self::Healthcare => "Healthcare",
            Self::Other(label) => label.as_str(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Urgency {
    Low,
    Medium,
    High,
    Critical,
}

impl Urgency {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Critical => "Critical",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Low => "urgency-low",
            Self::Medium => "urgency-medium",
            Self::High => "urgency-high",
            Self::Critical => "urgency-critical",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeedStatus {
    Open,
    Matched,
    Fulfilled,
    Withdrawn,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeedView {
    pub hash: String,
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: NeedCategory,
    pub requester_did: String,
    pub urgency: Urgency,
    pub status: NeedStatus,
    pub created: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OfferView {
    pub hash: String,
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: NeedCategory,
    pub offerer_did: String,
    pub created: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircleType {
    Neighborhood,
    Family,
    MutualAid,
    ElderCare,
    Childcare,
    Other(String),
}

impl CircleType {
    pub fn label(&self) -> &str {
        match self {
            Self::Neighborhood => "Neighborhood",
            Self::Family => "Family",
            Self::MutualAid => "Mutual Aid",
            Self::ElderCare => "Elder Care",
            Self::Childcare => "Childcare",
            Self::Other(label) => label.as_str(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CareCircleView {
    pub hash: String,
    pub name: String,
    pub description: String,
    pub circle_type: CircleType,
    pub member_count: u32,
    pub active: bool,
    pub created: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlotType {
    CommunityGarden,
    Rooftop,
    Orchard,
    Farm,
    Other(String),
}

impl PlotType {
    pub fn label(&self) -> &str {
        match self {
            Self::CommunityGarden => "Community Garden",
            Self::Rooftop => "Rooftop",
            Self::Orchard => "Orchard",
            Self::Farm => "Farm",
            Self::Other(label) => label.as_str(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlotView {
    pub hash: String,
    pub name: String,
    pub area_sqm: f64,
    pub plot_type: PlotType,
    pub steward_did: String,
    pub crop_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketType {
    Farmers,
    FoodBank,
    MutualAid,
    CoOp,
}

impl MarketType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Farmers => "Farmers Market",
            Self::FoodBank => "Food Bank",
            Self::MutualAid => "Mutual Aid",
            Self::CoOp => "Co-op",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketView {
    pub hash: String,
    pub name: String,
    pub market_type: MarketType,
    pub listing_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoodListingView {
    pub hash: String,
    pub product_name: String,
    pub quantity_kg: f64,
    pub price_per_kg: f64,
    pub organic: bool,
    pub producer_did: String,
    pub available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaterSystemType {
    RoofRainwater,
    GroundCatchment,
    Well,
    Reservoir,
}

impl WaterSystemType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::RoofRainwater => "Roof Rainwater",
            Self::GroundCatchment => "Ground Catchment",
            Self::Well => "Well",
            Self::Reservoir => "Reservoir",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterSystemView {
    pub hash: String,
    pub name: String,
    pub system_type: WaterSystemType,
    pub capacity_liters: u32,
    pub current_level_liters: u32,
    pub owner_did: String,
}

impl WaterSystemView {
    pub fn fill_pct(&self) -> f64 {
        if self.capacity_liters == 0 {
            0.0
        } else {
            (self.current_level_liters as f64 / self.capacity_liters as f64) * 100.0
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCategory {
    PowerTool,
    Garden,
    Kitchen,
    Mobility,
    Other(String),
}

impl ToolCategory {
    pub fn label(&self) -> &str {
        match self {
            Self::PowerTool => "Power Tool",
            Self::Garden => "Garden",
            Self::Kitchen => "Kitchen",
            Self::Mobility => "Mobility",
            Self::Other(label) => label.as_str(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCondition {
    Excellent,
    Good,
    Fair,
    NeedsRepair,
}

impl ToolCondition {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::NeedsRepair => "Needs Repair",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolView {
    pub hash: String,
    pub name: String,
    pub description: String,
    pub category: ToolCategory,
    pub condition: ToolCondition,
    pub available: bool,
    pub owner_did: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventView {
    pub hash: String,
    pub title: String,
    pub description: String,
    pub category: String,
    pub organizer_did: String,
    pub start_time: i64,
    pub end_time: i64,
    pub max_attendees: u32,
    pub rsvp_count: u32,
}
