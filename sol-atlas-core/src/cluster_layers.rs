// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal cluster layers for Sol Atlas core (Simulation Patch).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoFeature {
    pub lat: f64,
    pub lon: f64,
    pub feature_type: String,
    pub name: String,
    pub properties: HashMap<String, String>,
    pub color: [f32; 4],
    pub size: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationLayer {
    pub cluster: String,
    pub display_name: String,
    pub toggle_key: String,
    pub features: Vec<GeoFeature>,
    pub enabled: bool,
}

pub fn energy_layer() -> VisualizationLayer {
    VisualizationLayer {
        cluster: "energy".into(),
        display_name: "Energy Sites".into(),
        toggle_key: "e".into(),
        features: vec![],
        enabled: true,
    }
}

pub fn climate_layer() -> VisualizationLayer {
    VisualizationLayer {
        cluster: "climate".into(),
        display_name: "Climate Projects".into(),
        toggle_key: "c".into(),
        features: vec![],
        enabled: true,
    }
}

pub fn commons_layer() -> VisualizationLayer {
    VisualizationLayer {
        cluster: "commons".into(),
        display_name: "Commons".into(),
        toggle_key: "m".into(),
        features: vec![],
        enabled: true,
    }
}

pub fn supply_chain_layer() -> VisualizationLayer {
    VisualizationLayer {
        cluster: "supplychain".into(),
        display_name: "Supply Chain".into(),
        toggle_key: "s".into(),
        features: vec![],
        enabled: true,
    }
}

pub fn orbital_layer() -> VisualizationLayer {
    VisualizationLayer {
        cluster: "orbital".into(),
        display_name: "Orbital Assets".into(),
        toggle_key: "o".into(),
        features: vec![],
        enabled: true,
    }
}
