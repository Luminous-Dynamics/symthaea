// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spatial Metabolism — Hexagonal zone-based settlement management.

use h3o::{CellIndex, Resolution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The role of a metabolic zone in the settlement.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ZoneRole {
    Generation,  // Power generation (Solar/Helios)
    Filtration,  // Water purification
    Fabrication, // 3D Printing / Manufacturing
    Storage,     // Battery / Material storage
}

/// A spatial zone with local metabolic state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolicZone {
    pub h3_index: u64,
    pub role: ZoneRole,
    pub efficiency: f32,
    pub load: f32,
    pub localized_surprise: f32,
}

/// A spatial grid of metabolic zones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialMetabolicGrid {
    pub zones: HashMap<u64, MetabolicZone>,
}

impl SpatialMetabolicGrid {
    /// Create a new grid around a central coordinate.
    pub fn new_hex_cluster(center_lat: f64, center_lon: f64) -> Self {
        let mut zones = HashMap::new();
        let resolution = Resolution::Eight;

        // In a real implementation, we would use h3o to find neighbors.
        // Here we mock a 7-cell hexagonal cluster (1 center + 6 neighbors).
        let roles = [
            ZoneRole::Generation,
            ZoneRole::Storage,
            ZoneRole::Filtration,
            ZoneRole::Fabrication,
            ZoneRole::Fabrication,
            ZoneRole::Filtration,
            ZoneRole::Storage,
        ];

        for i in 0..7 {
            // Mock unique indices
            let index = 0x8828308281ffffffu64 + i;
            zones.insert(
                index,
                MetabolicZone {
                    h3_index: index,
                    role: roles[i as usize % roles.len()],
                    efficiency: 1.0,
                    load: 0.0,
                    localized_surprise: 0.0,
                },
            );
        }

        Self { zones }
    }

    /// Update the grid state and propagate localized surprise.
    pub fn update(&mut self, global_demand: f32, global_available: f32) -> f32 {
        let mut total_surprise = 0.0;
        let zone_count = self.zones.len() as f32;

        // Simple propagation: if a Generation zone drops,
        // the Fabrication zones nearby feel the surprise.
        let avg_gen_efficiency: f32 = self
            .zones
            .values()
            .filter(|z| z.role == ZoneRole::Generation)
            .map(|z| z.efficiency)
            .sum::<f32>()
            / self
                .zones
                .values()
                .filter(|z| z.role == ZoneRole::Generation)
                .count()
                .max(1) as f32;

        for zone in self.zones.values_mut() {
            match zone.role {
                ZoneRole::Fabrication => {
                    // Localized surprise based on generation deficit
                    zone.localized_surprise = (1.0 - avg_gen_efficiency).max(0.0);
                    zone.load = (global_demand / zone_count) * avg_gen_efficiency;
                }
                ZoneRole::Generation => {
                    // Efficiency affected by global availability
                    zone.efficiency = (global_available / 15.0).min(1.0);
                }
                _ => {
                    zone.localized_surprise *= 0.9; // Dissipate surprise
                }
            }
            total_surprise += zone.localized_surprise;
        }

        total_surprise / zone_count
    }
}
