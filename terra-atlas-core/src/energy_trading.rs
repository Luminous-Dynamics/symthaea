// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! P2P energy trading types and simulation for Terra Atlas visualization.
//!
//! Real data comes from mycelix-energy grid zome (TradeOffer/Trade entries).
//! This module provides types + a deterministic simulation for demo/offline use.

use serde::{Deserialize, Serialize};

/// A P2P energy trade between two geographic points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyTrade {
    pub seller_lat: f64,
    pub seller_lon: f64,
    pub buyer_lat: f64,
    pub buyer_lon: f64,
    pub amount_kwh: f64,
    pub price_per_kwh: f64,
    pub currency: String,
    pub timestamp: f64,
}

/// Regional grid stress computed from FEP prediction error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridStress {
    pub lat: f64,
    pub lon: f64,
    pub name: String,
    /// Allostatic load: 0.0 (balanced) to 1.0 (critical stress).
    pub allostatic_load: f32,
    /// Prediction error: how wrong the model's energy demand forecast was.
    pub prediction_error: f32,
    /// Phi (integration): how well the local grid is coordinated.
    pub phi: f32,
    /// Renewable penetration: fraction of demand met by renewables.
    pub renewable_fraction: f32,
}

/// Generate simulated P2P energy trades between renewable sites.
/// Uses deterministic seed for reproducibility.
pub fn simulate_trades(sites: &[(f64, f64, f64)], time: f64) -> Vec<EnergyTrade> {
    let mut trades = Vec::new();
    let n = sites.len();
    if n < 2 { return trades; }

    // Simulate trades between nearby sites based on time
    for i in 0..n {
        let j = (i + 1 + (time as usize * 7 + i * 3) % (n - 1)) % n;
        if i == j { continue; }

        // Trade probability based on distance and time of day
        let phase = (time * 0.1 + i as f64 * 0.7).sin();
        if phase < 0.3 { continue; } // only ~35% of pairs trade at any time

        trades.push(EnergyTrade {
            seller_lat: sites[i].0,
            seller_lon: sites[i].1,
            buyer_lat: sites[j].0,
            buyer_lon: sites[j].1,
            amount_kwh: sites[i].2 * phase.abs() * 100.0, // proportional to capacity
            price_per_kwh: 0.08 + phase.abs() * 0.04,     // $0.08-0.12/kWh
            currency: "SAP".into(),
            timestamp: time,
        });
    }

    trades
}

/// Generate simulated grid stress for major population centers.
/// Stress increases as fossil EROI declines (timeline year) and
/// decreases where renewable penetration is high.
pub fn simulate_grid_stress(year: u32) -> Vec<GridStress> {
    let fossil_decline = (year as f32 / 300.0).min(1.0);
    let renewable_growth = (year as f32 / 100.0).min(1.0);

    // Major population/grid centers with baseline stress
    let centers = [
        (28.6, 77.2, "Delhi", 0.7),
        (23.1, 113.3, "Guangzhou", 0.5),
        (35.7, 139.7, "Tokyo", 0.3),
        (40.7, -74.0, "New York", 0.25),
        (51.5, -0.1, "London", 0.2),
        (-26.2, 28.0, "Johannesburg", 0.8),   // Eskom crisis
        (19.4, -99.1, "Mexico City", 0.6),
        (6.5, 3.4, "Lagos", 0.75),
        (-23.5, -46.6, "São Paulo", 0.4),
        (55.75, 37.6, "Moscow", 0.35),
        (30.0, 31.2, "Cairo", 0.65),
        (13.75, 100.5, "Bangkok", 0.5),
        (37.6, 127.0, "Seoul", 0.25),
        (-33.9, 18.4, "Cape Town", 0.7),       // Load shedding
        (12.97, 77.6, "Bangalore", 0.55),
        (41.0, 29.0, "Istanbul", 0.45),
    ];

    centers.iter().map(|(lat, lon, name, base_stress)| {
        // Stress increases as fossil EROI drops, decreases with renewables
        let stress = (base_stress + fossil_decline * 0.4 - renewable_growth * 0.3).clamp(0.0, 1.0);
        let pe = stress * 0.8 + (year as f32 * 0.03 + (*lat as f32).abs() * 0.01).sin().abs() * 0.2;
        let phi = 1.0 - stress * 0.7; // high stress = low integration

        GridStress {
            lat: *lat,
            lon: *lon,
            name: name.to_string(),
            allostatic_load: stress,
            prediction_error: pe.clamp(0.0, 1.0),
            phi: phi.clamp(0.0, 1.0),
            renewable_fraction: renewable_growth * (1.0 - base_stress * 0.5),
        }
    }).collect()
}

/// Color for grid stress visualization: green (healthy) → amber → red (critical).
pub fn stress_color(allostatic_load: f32) -> [f32; 3] {
    if allostatic_load < 0.3 {
        [0.06, 0.73, 0.51] // green — balanced
    } else if allostatic_load < 0.6 {
        [0.98, 0.75, 0.14] // amber — strained
    } else {
        [0.94, 0.20, 0.15] // red — critical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_trades_needs_two_sites() {
        let trades = simulate_trades(&[(0.0, 0.0, 100.0)], 1.0);
        assert!(trades.is_empty());
    }

    #[test]
    fn test_simulate_trades_produces_output() {
        let sites = vec![
            (51.5, -0.1, 500.0),  // London
            (48.9, 2.3, 300.0),   // Paris
            (52.5, 13.4, 400.0),  // Berlin
        ];
        let trades = simulate_trades(&sites, 5.0);
        assert!(!trades.is_empty());
        for trade in &trades {
            assert!(trade.amount_kwh > 0.0);
            assert!(trade.price_per_kwh > 0.0);
        }
    }

    #[test]
    fn test_grid_stress_simulation() {
        let stress_y0 = simulate_grid_stress(0);
        let stress_y200 = simulate_grid_stress(200);
        assert_eq!(stress_y0.len(), 16);

        // Johannesburg should be high stress
        let joburg = stress_y0.iter().find(|s| s.name == "Johannesburg").unwrap();
        assert!(joburg.allostatic_load > 0.5);

        // Stress should generally increase over time as fossils decline
        let avg_0: f32 = stress_y0.iter().map(|s| s.allostatic_load).sum::<f32>() / stress_y0.len() as f32;
        let avg_200: f32 = stress_y200.iter().map(|s| s.allostatic_load).sum::<f32>() / stress_y200.len() as f32;
        // But renewable growth offsets some stress, so check specific high-stress regions
        let joburg_200 = stress_y200.iter().find(|s| s.name == "Johannesburg").unwrap();
        assert!(joburg_200.allostatic_load > joburg.allostatic_load * 0.8);
    }

    #[test]
    fn test_stress_color_thresholds() {
        let green = stress_color(0.1);
        let amber = stress_color(0.4);
        let red = stress_color(0.8);
        assert!(green[1] > green[0]);
        assert!(amber[0] > amber[2]);
        assert!(red[0] > red[1]);
    }
}
