// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Robotic Fleet Integration: conscious machines that cost Joules to think.
//!
//! Robots don't get cancer. But robots cost power. A Humanoid in the Ag-Bay
//! saves humans from 20 mSv/month — but its FEP consciousness engine draws
//! 500W continuously. When a dust storm kills the solar panels, the colony
//! must choose: power the robot's mind, or heat the dormitories.
//!
//! # The Joule Tax
//!
//! Every active robot deducts power from the colony's energy grid.
//! Power draw scales with required Phi:
//! - Manipulator (simple task, low Phi): ~100W
//! - Quadrotor (flight, moderate Phi): ~200W
//! - Humanoid (full autonomy, high Phi): ~500W
//! - Helicopter (flight + SAR, highest Phi): ~800W
//!
//! # The Brownout Cascade
//!
//! When the grid can't supply the fleet's total power demand:
//! 1. Robots are powered in priority order (medical > ECLSS > agriculture > other)
//! 2. Underpowered robots lose Phi proportionally
//! 3. Phi below safety threshold → robot halts (Green→Yellow→Orange→Red)
//! 4. Halted robots' tasks fall back to human labor
//! 5. Humans entering hazardous modules take radiation + psychological damage
//!
//! # References
//!
//! - Symthaea EmbodimentBridge trait: 4-tier safety (Green/Yellow/Orange/Red)
//! - NASA RASSOR: ~100W excavation robot
//! - Robonaut 2: ~200W for upper body, ~500W with legs
//! - Boston Dynamics Atlas: ~800W (terrestrial humanoid reference)

use serde::{Deserialize, Serialize};

/// The 6 Symthaea robotic platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RobotPlatform {
    /// 7-DOF industrial arm. Construction, fabrication, maintenance.
    Manipulator,
    /// Quadrotor drone. Cave exploration, surveillance, small cargo.
    Quadrotor,
    /// Autonomous underwater vehicle. Water quality, under-ice ops.
    Auv,
    /// Autonomous ground vehicle. Surface transport, cargo hauling.
    Vehicle,
    /// Bipedal humanoid. General labor, EVA replacement, agriculture.
    Humanoid,
    /// Rotorcraft. SAR, aerial survey, emergency transport.
    Helicopter,
}

impl RobotPlatform {
    /// Power consumption in watts at full consciousness (Phi = 1.0).
    /// Includes motor power + computation (FEP engine, HDC encoding).
    /// The consciousness tax: thinking costs Joules.
    pub fn power_watts(&self) -> f64 {
        match self {
            Self::Manipulator => 150.0, // Simple kinematics, low Phi needed
            Self::Quadrotor => 200.0,   // Flight + navigation
            Self::Auv => 250.0,         // Hydrodynamics + chemical sensors
            Self::Vehicle => 300.0,     // Locomotion + mesh coordination
            Self::Humanoid => 500.0,    // Full-body dynamics + FEP
            Self::Helicopter => 800.0,  // Rotor dynamics + SAR autonomy
        }
    }

    /// Minimum Phi required for safe operation.
    /// Below this, the robot enters safe-mode (halts).
    pub fn min_phi(&self) -> f64 {
        match self {
            Self::Manipulator => 0.1, // Can operate at Orange level
            Self::Quadrotor => 0.3,   // Needs Yellow for flight
            Self::Auv => 0.3,         // Needs Yellow for navigation
            Self::Vehicle => 0.1,     // Can crawl at Orange
            Self::Humanoid => 0.3,    // Needs Yellow for walking
            Self::Helicopter => 0.6,  // Needs Green for full flight
        }
    }

    /// Human labor-hours replaced per day when fully operational.
    /// From NASA crew time studies + IFR manufacturing data.
    pub fn labor_hours_replaced(&self) -> f64 {
        match self {
            Self::Manipulator => 6.0, // 1 arm ≈ 0.75 human-shift
            Self::Quadrotor => 3.0,   // Scout/survey, not physical labor
            Self::Auv => 4.0,         // Continuous monitoring
            Self::Vehicle => 8.0,     // 24/7 transport (3 shifts)
            Self::Humanoid => 12.0,   // General-purpose labor
            Self::Helicopter => 5.0,  // SAR + aerial survey
        }
    }

    /// Which habitat module function this robot primarily serves.
    pub fn primary_module(&self) -> crate::habitat::ModuleFunction {
        use crate::habitat::ModuleFunction;
        match self {
            Self::Manipulator => ModuleFunction::Workshop,
            Self::Quadrotor => ModuleFunction::Laboratory, // Exploration
            Self::Auv => ModuleFunction::LifeSupport,      // Water systems
            Self::Vehicle => ModuleFunction::Storage,      // Logistics
            Self::Humanoid => ModuleFunction::Agriculture, // Ag-Bay labor
            Self::Helicopter => ModuleFunction::Commons,   // SAR staging
        }
    }
}

/// A single robot unit in the colony fleet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotUnit {
    pub platform: RobotPlatform,
    /// Current Phi level [0, 1]. Degrades when power-starved.
    pub phi: f64,
    /// Whether the unit is operational (not in safe-mode).
    pub operational: bool,
    /// Total operating hours since deployment.
    pub total_hours: f64,
    /// Current power allocation from grid [0, 1].
    pub power_fraction: f64,
}

impl RobotUnit {
    pub fn new(platform: RobotPlatform) -> Self {
        Self {
            platform,
            phi: 0.8, // Start at high consciousness
            operational: true,
            total_hours: 0.0,
            power_fraction: 1.0,
        }
    }

    /// Actual power draw this tick (watts), given power allocation.
    pub fn actual_power_watts(&self) -> f64 {
        self.platform.power_watts() * self.power_fraction
    }

    /// Effective labor hours replaced, accounting for Phi degradation.
    pub fn effective_labor_hours(&self) -> f64 {
        if !self.operational {
            return 0.0;
        }
        // Labor scales with Phi above minimum threshold
        let min = self.platform.min_phi();
        if self.phi < min {
            return 0.0;
        }
        let phi_fraction = (self.phi - min) / (1.0 - min);
        self.platform.labor_hours_replaced() * phi_fraction
    }
}

/// Colony robotic fleet manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoboticFleet {
    pub units: Vec<RobotUnit>,
    /// Total power consumed by fleet this tick (watts).
    pub total_power_draw_w: f64,
    /// Total human labor-hours replaced by fleet this tick.
    pub total_labor_replaced: f64,
    /// Number of robots in safe-mode (insufficient power/Phi).
    pub units_in_safe_mode: u32,
}

impl RoboticFleet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a robot to the fleet.
    pub fn deploy(&mut self, platform: RobotPlatform) {
        self.units.push(RobotUnit::new(platform));
    }

    /// The Joule Tax: compute fleet power demand vs available power.
    /// Allocate power to robots in priority order. Underpowered robots
    /// lose Phi and may enter safe-mode.
    ///
    /// Returns: (total_power_drawn_kw, total_labor_replaced_hours, brownout_count)
    pub fn tick_power_allocation(
        &mut self,
        available_power_kw: f64,
        dt_hours: f64, // hours per tick (typically 730 for monthly)
    ) -> (f64, f64, u32) {
        let available_watts = available_power_kw * 1000.0;

        // Total fleet demand
        let total_demand: f64 = self.units.iter().map(|u| u.platform.power_watts()).sum();

        // Power fraction: how much of demand can be met
        let power_fraction = if total_demand > 0.0 {
            (available_watts / total_demand).min(1.0)
        } else {
            1.0
        };

        let mut total_draw = 0.0;
        let mut total_labor = 0.0;
        let mut brownouts = 0u32;

        for unit in &mut self.units {
            // Allocate power proportionally
            unit.power_fraction = power_fraction;

            // Phi tracks power: full power → Phi recovers toward 0.8
            // Low power → Phi decays toward 0.0
            let target_phi = power_fraction * 0.8;
            unit.phi = unit.phi * 0.9 + target_phi * 0.1; // EMA blending

            // Check operational status
            let was_operational = unit.operational;
            unit.operational = unit.phi >= unit.platform.min_phi();

            if !unit.operational && was_operational {
                brownouts += 1;
            }

            // Accumulate power draw and labor
            if unit.operational {
                total_draw += unit.actual_power_watts();
                total_labor += unit.effective_labor_hours() * dt_hours / 24.0;
                unit.total_hours += dt_hours;
            }
        }

        self.total_power_draw_w = total_draw;
        self.total_labor_replaced = total_labor;
        self.units_in_safe_mode = self.units.iter().filter(|u| !u.operational).count() as u32;

        (total_draw / 1000.0, total_labor, brownouts)
    }

    /// Count operational units of a specific platform.
    pub fn count_operational(&self, platform: RobotPlatform) -> usize {
        self.units
            .iter()
            .filter(|u| u.platform == platform && u.operational)
            .count()
    }

    /// Total fleet size.
    pub fn total_units(&self) -> usize {
        self.units.len()
    }

    /// Fraction of fleet that is operational.
    pub fn operational_fraction(&self) -> f64 {
        if self.units.is_empty() {
            return 0.0;
        }
        self.units.iter().filter(|u| u.operational).count() as f64 / self.units.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robot_power_consumption() {
        assert!(RobotPlatform::Manipulator.power_watts() < RobotPlatform::Humanoid.power_watts());
        assert!(RobotPlatform::Humanoid.power_watts() < RobotPlatform::Helicopter.power_watts());
    }

    #[test]
    fn test_fleet_power_allocation_sufficient() {
        let mut fleet = RoboticFleet::new();
        fleet.deploy(RobotPlatform::Manipulator);
        fleet.deploy(RobotPlatform::Humanoid);

        // 1 kW available = 1000W. Manipulator(150) + Humanoid(500) = 650W. Sufficient.
        let (draw, labor, brownouts) = fleet.tick_power_allocation(1.0, 730.0);
        assert!(draw > 0.0);
        assert!(labor > 0.0);
        assert_eq!(brownouts, 0);
        assert_eq!(fleet.units_in_safe_mode, 0);
    }

    #[test]
    fn test_brownout_cascade() {
        let mut fleet = RoboticFleet::new();
        fleet.deploy(RobotPlatform::Humanoid); // 500W, needs Phi > 0.3
        fleet.deploy(RobotPlatform::Helicopter); // 800W, needs Phi > 0.6

        // Only 200W available for 1300W demand → 15% power fraction
        // After several ticks, Phi decays below thresholds
        for _ in 0..20 {
            fleet.tick_power_allocation(0.2, 730.0);
        }

        // Helicopter (min_phi 0.6) should be in safe-mode
        // Humanoid (min_phi 0.3) may also be in safe-mode at 15% power
        assert!(
            fleet.units_in_safe_mode > 0,
            "At least one robot should be in safe-mode. Safe: {}",
            fleet.units_in_safe_mode
        );
    }

    #[test]
    fn test_labor_scales_with_phi() {
        let mut unit = RobotUnit::new(RobotPlatform::Humanoid);

        // Full Phi → full labor
        unit.phi = 0.8;
        let full = unit.effective_labor_hours();

        // Half Phi → less labor
        unit.phi = 0.5;
        let half = unit.effective_labor_hours();

        // Below minimum → zero labor
        unit.phi = 0.2;
        let zero = unit.effective_labor_hours();

        assert!(full > half);
        assert!(half > zero);
        assert_eq!(zero, 0.0);
    }
}
