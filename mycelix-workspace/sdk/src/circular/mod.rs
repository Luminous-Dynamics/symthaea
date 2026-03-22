// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Circular Economy SDK Module
//!
//! Client types for interacting with the waste/circular economy zomes:
//! - **waste-registry**: Waste stream tracking, classification, facility matching
//! - **waste-collection**: Collection requests, vehicle matching, run planning
//! - **compost-control**: Sensor-driven composting automation, carbon attribution
//! - **circular-marketplace**: Secondary material listings, demand matching, orders
//!
//! These mirror the Holochain entry types for client-side use without
//! importing the zome crates directly (avoids WASM symbol conflicts).

use serde::{Deserialize, Serialize};

// ============================================================================
// WASTE REGISTRY TYPES
// ============================================================================

/// Category of waste material.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WasteCategory {
    /// Food scraps, yard waste, wood, paper
    Organic,
    /// Plastics, metals, glass, cardboard
    Recyclable,
    /// Toxic chemicals, batteries, medical waste
    Hazardous,
    /// Computers, phones, circuit boards
    Electronic,
    /// Unsorted or multi-material waste
    Mixed,
    /// Construction debris, ceramics, non-reactive
    Inert,
}

/// Contamination level of a waste stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContaminationLevel {
    /// < 0.5% contaminant by weight
    Clean,
    /// 0.5–3% contaminant
    LightlyContaminated,
    /// 3–10% contaminant
    Contaminated,
    /// > 10% or toxic contaminants present
    Hazardous,
}

/// End-of-life strategy for waste material.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EndOfLifeStrategy {
    /// Can be mechanically recycled
    MechanicalRecycling,
    /// Can be chemically recycled
    ChemicalRecycling,
    /// Naturally biodegrades
    Biodegradable,
    /// Industrial composting required
    IndustrialCompost,
    /// Can be downcycled to lower-grade products
    Downcycle,
    /// Must go to landfill (discouraged)
    Landfill,
}

/// Lifecycle status of a waste stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WasteStreamStatus {
    /// Newly registered
    Registered,
    /// Classified by human or AI
    Classified,
    /// Matched to a facility
    Routed,
    /// Picked up by collection vehicle
    Collected,
    /// Being processed
    Processing,
    /// Fully processed
    Completed,
}

/// Type of waste processing facility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FacilityType {
    /// Materials Recovery Facility
    MRF,
    /// Composting operation
    Composter,
    /// Anaerobic digestion
    AnaerobicDigester,
    /// Chemical recycling
    ChemicalRecycler,
    /// Sanitary landfill
    Landfill,
    /// Transfer/consolidation station
    TransferStation,
}

// ============================================================================
// COMPOST CONTROL TYPES
// ============================================================================

/// Composting method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompostMethod {
    /// Open-air rows
    Windrow,
    /// Forced aeration
    StaticAerated,
    /// Enclosed vessel
    InVessel,
    /// Worm-mediated
    Vermicompost,
    /// Anaerobic fermentation
    Bokashi,
}

/// Lifecycle status of a compost batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompostBatchStatus {
    /// Initial mixing
    Mixing,
    /// Thermophilic phase (55-65C)
    ActiveComposting,
    /// Cooling/maturation (25-40C)
    Curing,
    /// Final screening
    Screening,
    /// Finished
    Complete,
    /// Process failed
    Failed,
}

/// Carbon attribution from composting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonAttribution {
    /// Total waste diverted from landfill (kg)
    pub waste_diverted_kg: f64,
    /// Estimated avoided CO2 equivalent (tonnes)
    pub co2e_avoided_tonnes: f64,
    /// Calculation methodology
    pub methodology: String,
}

/// Nutrient estimate for a compost batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NutrientEstimate {
    /// Total batch weight (kg)
    pub total_kg: f64,
    /// Nitrogen percentage
    pub nitrogen_pct: f64,
    /// Phosphorus percentage
    pub phosphorus_pct: f64,
    /// Potassium percentage
    pub potassium_pct: f64,
}

// ============================================================================
// MARKETPLACE TYPES
// ============================================================================

/// Quality grade for secondary materials.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaterialQualityGrade {
    /// Top quality
    Premium,
    /// Standard quality
    Standard,
    /// Lower grade, industrial use
    Industrial,
}

/// Type of secondary material.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SecondaryMaterialType {
    /// Finished compost
    Compost,
    /// Worm castings
    Vermicompost,
    /// Anaerobic digestion output
    Digestate,
    /// Biochar
    Biochar,
    /// Recycled plastic
    RecycledPlastic,
    /// Recycled metal
    RecycledMetal,
    /// Recycled glass
    RecycledGlass,
    /// Recycled paper/cardboard
    RecycledFiber,
    /// Mulch/woodchips
    Mulch,
}

/// Diversion rate summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversionRateResult {
    /// Total waste processed (kg)
    pub total_kg: f64,
    /// Composted (kg)
    pub composted_kg: f64,
    /// Recycled (kg)
    pub recycled_kg: f64,
    /// Landfilled (kg)
    pub landfilled_kg: f64,
    /// Diversion rate (0-100%)
    pub diversion_rate_pct: f64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waste_category_serde_roundtrip() {
        let categories = vec![
            WasteCategory::Organic,
            WasteCategory::Recyclable,
            WasteCategory::Hazardous,
            WasteCategory::Electronic,
            WasteCategory::Mixed,
            WasteCategory::Inert,
        ];
        for cat in categories {
            let json = serde_json::to_string(&cat).expect("serialize");
            let decoded: WasteCategory = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(cat, decoded);
        }
    }

    #[test]
    fn test_eol_strategy_serde_roundtrip() {
        let strategies = vec![
            EndOfLifeStrategy::MechanicalRecycling,
            EndOfLifeStrategy::ChemicalRecycling,
            EndOfLifeStrategy::Biodegradable,
            EndOfLifeStrategy::IndustrialCompost,
            EndOfLifeStrategy::Downcycle,
            EndOfLifeStrategy::Landfill,
        ];
        for strat in strategies {
            let json = serde_json::to_string(&strat).expect("serialize");
            let decoded: EndOfLifeStrategy = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(strat, decoded);
        }
    }

    #[test]
    fn test_carbon_attribution_serde() {
        let attr = CarbonAttribution {
            waste_diverted_kg: 1000.0,
            co2e_avoided_tonnes: 0.06,
            methodology: "EPA WARM".to_string(),
        };
        let json = serde_json::to_string(&attr).expect("serialize");
        let decoded: CarbonAttribution = serde_json::from_str(&json).expect("deserialize");
        assert!((decoded.co2e_avoided_tonnes - 0.06).abs() < 0.001);
    }

    #[test]
    fn test_nutrient_estimate_serde() {
        let nutrients = NutrientEstimate {
            total_kg: 100.0,
            nitrogen_pct: 1.5,
            phosphorus_pct: 0.5,
            potassium_pct: 1.0,
        };
        let json = serde_json::to_string(&nutrients).expect("serialize");
        let decoded: NutrientEstimate = serde_json::from_str(&json).expect("deserialize");
        assert!((decoded.nitrogen_pct - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_compost_method_serde_roundtrip() {
        let methods = vec![
            CompostMethod::Windrow,
            CompostMethod::StaticAerated,
            CompostMethod::InVessel,
            CompostMethod::Vermicompost,
            CompostMethod::Bokashi,
        ];
        for m in methods {
            let json = serde_json::to_string(&m).expect("serialize");
            let decoded: CompostMethod = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(m, decoded);
        }
    }

    #[test]
    fn test_quality_grade_serde_roundtrip() {
        let grades = vec![
            MaterialQualityGrade::Premium,
            MaterialQualityGrade::Standard,
            MaterialQualityGrade::Industrial,
        ];
        for g in grades {
            let json = serde_json::to_string(&g).expect("serialize");
            let decoded: MaterialQualityGrade = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(g, decoded);
        }
    }
}
