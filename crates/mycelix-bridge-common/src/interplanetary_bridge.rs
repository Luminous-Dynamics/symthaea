// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Interplanetary Bridge: transfer windows, cargo tracking, and trade routes.
//!
//! Implements the orbital mechanics constraints on inter-colony communication
//! and trade. All constants are derived from Keplerian orbital mechanics
//! (Bate, Mueller & White 1971; Vallado 2013).
//!
//! # Key Concept: Transfer Windows
//!
//! Physical cargo can only move between planets during Hohmann transfer windows
//! (every synodic period). Information travels at light speed but is blocked
//! during solar conjunction. Fusion Drive tech eliminates window constraints.

use crate::earth_colony_protocol::PlanetaryBody;
use serde::{Deserialize, Serialize};

// ============================================================================
// Orbital Constants
// ============================================================================

/// Synodic period: ticks between transfer windows (1 tick = 1 month).
/// Source: Keplerian synodic period formula, JPL Solar System Dynamics.
pub fn synodic_period(from: PlanetaryBody, to: PlanetaryBody) -> u32 {
    use PlanetaryBody::*;
    let pair = normalize_pair(from, to);
    match pair {
        (Earth, Moon) | (Moon, Earth) => 0,     // Continuous (3-day transfer)
        (Earth, Mars) | (Mars, Earth) => 26,    // 25.6 months
        (Moon, Mars) | (Mars, Moon) => 26,      // Same as Earth-Mars
        (Earth, Europa) | (Europa, Earth) => 13, // 13.1 months
        (Moon, Europa) | (Europa, Moon) => 13,
        (Earth, Titan) | (Titan, Earth) => 12,  // 12.4 months
        (Moon, Titan) | (Titan, Moon) => 12,
        (Mars, Europa) | (Europa, Mars) => 27,  // 26.8 months
        (Mars, Titan) | (Titan, Mars) => 24,    // 24.1 months
        (Earth, Ceres) | (Ceres, Earth) => 15,  // ~15.6 months
        _ => 0, // Unknown pairs: continuous (conservative)
    }
}

/// Hohmann transfer time in ticks (minimum energy trajectory).
/// Source: Vallado, Fundamentals of Astrodynamics 4th ed. (2013).
pub fn transfer_time(from: PlanetaryBody, to: PlanetaryBody) -> u32 {
    use PlanetaryBody::*;
    let pair = normalize_pair(from, to);
    match pair {
        (Earth, Moon) | (Moon, Earth) => 0,      // ~3 days ≈ 0 ticks
        (Earth, Mars) | (Mars, Earth) => 9,      // 258 days
        (Moon, Mars) | (Mars, Moon) => 9,
        (Earth, Europa) | (Europa, Earth) => 33,  // 2.73 years
        (Moon, Europa) | (Europa, Moon) => 33,
        (Earth, Titan) | (Titan, Earth) => 73,   // 6.05 years
        (Moon, Titan) | (Titan, Moon) => 73,
        (Mars, Europa) | (Europa, Mars) => 26,   // 2.16 years
        (Mars, Titan) | (Titan, Mars) => 59,     // 4.9 years
        (Earth, Ceres) | (Ceres, Earth) => 16,   // ~1.3 years
        _ => 0,
    }
}

/// Delta-v cost for Hohmann transfer (km/s, from LEO-equivalent).
pub fn hohmann_delta_v(from: PlanetaryBody, to: PlanetaryBody) -> f64 {
    use PlanetaryBody::*;
    let pair = normalize_pair(from, to);
    match pair {
        (Earth, Moon) | (Moon, Earth) => 4.0,
        (Earth, Mars) | (Mars, Earth) => 4.5,
        (Earth, Europa) | (Europa, Earth) => 7.2,
        (Earth, Titan) | (Titan, Earth) => 8.4,
        (Mars, Europa) | (Europa, Mars) => 3.9,
        (Mars, Titan) | (Titan, Mars) => 5.5,
        (Earth, Ceres) | (Ceres, Earth) => 5.0,
        _ => 3.0, // Conservative default
    }
}

/// Delta-v cost multiplier for off-window emergency transfer.
/// Launching outside the Hohmann window costs 2-5x more propellant.
pub const EMERGENCY_DV_MULTIPLIER: f64 = 2.5;

/// Solar conjunction blackout duration (ticks). Communication blocked.
pub const CONJUNCTION_BLACKOUT_TICKS: u32 = 1;

/// Normalize a body pair so the "smaller" enum variant is first.
fn normalize_pair(a: PlanetaryBody, b: PlanetaryBody) -> (PlanetaryBody, PlanetaryBody) {
    if (a as u8) <= (b as u8) { (a, b) } else { (b, a) }
}

// ============================================================================
// Transfer Window State
// ============================================================================

/// Transfer window status for a route between two bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowStatus {
    /// Window is open — cargo can launch at optimal delta-v.
    Open,
    /// Window is closed — emergency launch possible at 2.5x delta-v cost.
    Closed,
    /// Solar conjunction — no communication possible.
    Blackout,
    /// Fusion Drive available — windows are irrelevant.
    ContinuousAccess,
}

/// A trade route between two planetary bodies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterplanetaryRoute {
    pub from: PlanetaryBody,
    pub to: PlanetaryBody,
    /// Synodic period in ticks (0 = continuous).
    pub synodic_period: u32,
    /// Hohmann transfer time in ticks.
    pub transfer_time: u32,
    /// Delta-v cost (km/s).
    pub delta_v: f64,
    /// Transport cost per kg (kWh/kg), derived from delta-v.
    pub cost_per_kg: f64,
    /// Current window status.
    pub status: WindowStatus,
}

impl InterplanetaryRoute {
    /// Create a route between two bodies with computed orbital parameters.
    pub fn new(from: PlanetaryBody, to: PlanetaryBody) -> Self {
        let sp = synodic_period(from, to);
        let tt = transfer_time(from, to);
        let dv = hohmann_delta_v(from, to);
        Self {
            from, to,
            synodic_period: sp,
            transfer_time: tt,
            delta_v: dv,
            cost_per_kg: dv * dv * 0.001,
            status: if sp == 0 { WindowStatus::ContinuousAccess } else { WindowStatus::Closed },
        }
    }

    /// Update window status for current tick.
    pub fn update_status(&mut self, current_tick: u32, established_tick: u32, has_fusion_drive: bool) {
        if has_fusion_drive || self.synodic_period == 0 {
            self.status = WindowStatus::ContinuousAccess;
            return;
        }
        let elapsed = current_tick.saturating_sub(established_tick);
        if elapsed % self.synodic_period == 0 {
            self.status = WindowStatus::Open;
        } else {
            self.status = WindowStatus::Closed;
        }
        // TODO: conjunction blackout detection
    }
}

/// Cargo in transit between worlds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InTransitCargo {
    pub from: PlanetaryBody,
    pub to: PlanetaryBody,
    /// Resource type (matches ResourceManifest field names).
    pub resource_type: String,
    /// Amount in kg.
    pub amount_kg: f64,
    /// Tick when cargo departed.
    pub departure_tick: u32,
    /// Tick when cargo arrives.
    pub arrival_tick: u32,
    /// Whether this was an emergency (off-window) launch.
    pub is_emergency: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synodic_periods_match_orbital_mechanics() {
        assert_eq!(synodic_period(PlanetaryBody::Earth, PlanetaryBody::Mars), 26);
        assert_eq!(synodic_period(PlanetaryBody::Earth, PlanetaryBody::Europa), 13);
        assert_eq!(synodic_period(PlanetaryBody::Earth, PlanetaryBody::Titan), 12);
        // Moon-Earth is continuous
        assert_eq!(synodic_period(PlanetaryBody::Earth, PlanetaryBody::Moon), 0);
    }

    #[test]
    fn test_transfer_times() {
        assert_eq!(transfer_time(PlanetaryBody::Earth, PlanetaryBody::Mars), 9);
        assert_eq!(transfer_time(PlanetaryBody::Earth, PlanetaryBody::Titan), 73);
    }

    #[test]
    fn test_route_creation() {
        let route = InterplanetaryRoute::new(PlanetaryBody::Earth, PlanetaryBody::Europa);
        assert_eq!(route.synodic_period, 13);
        assert_eq!(route.transfer_time, 33);
        assert!(route.delta_v > 7.0);
    }

    #[test]
    fn test_symmetry() {
        // Route A→B should have same parameters as B→A
        assert_eq!(
            synodic_period(PlanetaryBody::Mars, PlanetaryBody::Europa),
            synodic_period(PlanetaryBody::Europa, PlanetaryBody::Mars),
        );
    }
}
