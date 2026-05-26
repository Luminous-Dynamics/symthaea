// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};

/// Discrete ecological features extracted from Earth observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EarthFeature {
    VegetationHealth(f64), // NDVI
    WaterExtent(f64),      // NDWI / SAR area
    FloodExtent(f64),      // Change in water extent
    Deforestation(f64),    // Canopy loss
    BurnSeverity(f64),     // NBR
    SoilMoisture(f64),     // SAR/Microwave
}

impl EarthFeature {
    pub fn name(&self) -> &'static str {
        match self {
            Self::VegetationHealth(_) => "VegetationHealth",
            Self::WaterExtent(_) => "WaterExtent",
            Self::FloodExtent(_) => "FloodExtent",
            Self::Deforestation(_) => "Deforestation",
            Self::BurnSeverity(_) => "BurnSeverity",
            Self::SoilMoisture(_) => "SoilMoisture",
        }
    }
}
