// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Modular Habitat Geometry: radiation dose and psychology per volume.
//!
//! A colony is not a point. It is a collection of physical modules, each with
//! distinct shielding, volume, function, and psychological character. An agent's
//! radiation dose depends on WHERE they spend their day. Their allostatic load
//! depends on whether they ever see light.
//!
//! # Architecture
//!
//! Each World contains a `HabitatComplex` — a collection of `HabitatModule`s.
//! Each module has:
//! - Volume (m³) and capacity (max occupants)
//! - Shielding type → radiation dose multiplier
//! - Window type → psychological light bonus
//! - Function → which professions work here
//!
//! Agents are assigned to modules based on their profession. Their per-tick
//! radiation dose and allostatic load modifier come from their assigned module.
//!
//! # The Commuter's Dose
//!
//! A botanist spends 10h/day in the unshielded Ag-Bay (20 mSv/month Mars)
//! and 14h in buried dormitories (4 mSv/month). Their weighted dose:
//!   (10/24) × 20 + (14/24) × 4 = 8.3 + 2.3 = 10.7 mSv/month
//!
//! A server admin spends 10h in the buried server room (4 mSv/month)
//! and 14h in dormitories (4 mSv/month) = 4.0 mSv/month, but their
//! allostatic load spikes from never seeing light.
//!
//! # References
//!
//! - Cucinotta et al. (2006) — water shielding effectiveness
//! - Blair et al. (2017) — lunar lava tube structural stability
//! - Hassler et al. (2014) — Mars surface radiation (MSL/RAD)
//! - Ulrich (1984) — window access and health outcomes
//! - Palinkas & Suedfeld (2008) — isolation psychology
//! - Flynn (2013) — NASA NIAC Water Walls

use serde::{Deserialize, Serialize};

// ============================================================================
// Shielding Types
// ============================================================================

/// How the module is shielded from radiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShieldingType {
    /// Surface habitat, no additional shielding. Full ambient dose.
    Unshielded,
    /// Regolith piled on top. Depth in meters determines reduction.
    /// 2m lunar regolith = ~70-80% GCR reduction (De Angelis 2002).
    RegolithBuried { depth_m: u8 },
    /// Water-filled walls. Thickness in cm determines reduction.
    /// 20cm water = 25-35% GCR, 80-90% SPE (Cucinotta 2006).
    WaterWall { thickness_cm: u8 },
    /// Inside a lava tube. Roof thickness >> any engineering solution.
    /// ~99% GCR reduction (hundreds of meters of basalt).
    LavaTube,
    /// Deep underground (excavated or natural cave, not lava tube).
    /// Similar to heavy regolith burial.
    Underground { depth_m: u8 },
}

impl ShieldingType {
    /// Fraction of ambient radiation that reaches the interior [0, 1].
    /// 0.0 = perfect shielding, 1.0 = no shielding.
    pub fn dose_fraction(&self) -> f64 {
        match self {
            Self::Unshielded => 1.0,
            Self::RegolithBuried { depth_m } => {
                // Exponential attenuation. 2m → 0.25, 5m → 0.05.
                // Ref: OLTARIS calculations, De Angelis 2002.
                (-0.7 * *depth_m as f64).exp().max(0.01)
            }
            Self::WaterWall { thickness_cm } => {
                // 20cm → ~0.70, 30cm → ~0.60, 50cm → ~0.45.
                // Ref: Cucinotta 2006, HZETRN transport code.
                (-0.018 * *thickness_cm as f64).exp().max(0.05)
            }
            Self::LavaTube => 0.01, // ~99% reduction
            Self::Underground { depth_m } => (-0.6 * *depth_m as f64).exp().max(0.01),
        }
    }
}

// ============================================================================
// Window / Light Types
// ============================================================================

/// What kind of natural light or view the module provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WindowType {
    /// No windows. Artificial light only.
    /// Allostatic load penalty: +20-30% after 90 days (Palinkas 2008).
    None,
    /// Simulated windows (screens showing external views).
    /// Recovers ~40% of no-window deficit (Kjellgren 2010).
    Simulated,
    /// Water-filled transparent walls. Natural light + shielding.
    /// Recovers ~80% of deficit + radiation protection.
    WaterWindow,
    /// Traditional pressure windows (small, structural constraint).
    /// Recovers ~70% of deficit. No radiation protection.
    PressureWindow,
    /// Large dome or cupola. Full sky view.
    /// Best psychological outcome but zero shielding.
    Dome,
}

impl WindowType {
    /// Allostatic load modifier from light/view access.
    /// Negative = reduces load (beneficial). Positive = increases load.
    /// Applied per tick to agents assigned to this module.
    pub fn load_modifier(&self) -> f64 {
        match self {
            Self::None => 0.003,         // +0.003/tick no-window stress
            Self::Simulated => 0.001,    // Partial recovery
            Self::WaterWindow => -0.001, // Net beneficial (light + beauty)
            Self::PressureWindow => -0.0005,
            Self::Dome => -0.002, // Best psychological outcome
        }
    }
}

// ============================================================================
// Habitat Module
// ============================================================================

/// A physical volume in the colony with specific shielding and function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HabitatModule {
    /// Human-readable name.
    pub name: String,
    /// Function: what happens in this module.
    pub function: ModuleFunction,
    /// Pressurized volume in m³.
    pub volume_m3: f64,
    /// Maximum simultaneous occupants.
    pub capacity: u32,
    /// Current occupancy.
    pub occupancy: u32,
    /// Radiation shielding.
    pub shielding: ShieldingType,
    /// Window / natural light type.
    pub windows: WindowType,
    /// Construction status: true = operational, false = under construction.
    pub operational: bool,
    /// Tick when construction started (None = pre-built).
    pub construction_start: Option<u32>,
    /// Ticks required to build.
    pub construction_duration: u32,
}

/// What the module is used for. Determines which professions are assigned here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleFunction {
    /// Living quarters / dormitories.
    Habitation,
    /// Food production (greenhouses, hydroponics).
    Agriculture,
    /// Medical facility.
    Medical,
    /// Engineering workshop / fabrication.
    Workshop,
    /// Command / governance / communications.
    Command,
    /// Science laboratory.
    Laboratory,
    /// Power systems (reactor, solar array control).
    Power,
    /// Social / recreation / dining.
    Commons,
    /// ECLSS (life support machinery).
    LifeSupport,
    /// Storage / logistics.
    Storage,
}

impl ModuleFunction {
    /// Which skill sector primarily works in this module (index 0-7).
    pub fn primary_sector(&self) -> usize {
        match self {
            Self::Agriculture => 1, // agriculture
            Self::Medical => 2,     // medicine
            Self::Command => 3,     // governance
            Self::Laboratory => 4,  // science
            Self::Workshop => 0,    // engineering
            Self::Power => 0,       // engineering
            Self::Commons => 6,     // art_culture
            Self::LifeSupport => 0, // engineering
            Self::Habitation => 7,  // logistics (everyone)
            Self::Storage => 7,     // logistics
        }
    }

    /// Hours per day a worker spends in this module.
    pub fn work_hours(&self) -> f64 {
        match self {
            Self::Habitation => 14.0, // Sleep + off-duty
            Self::Commons => 2.0,     // Meals + recreation
            _ => 8.0,                 // Standard work shift
        }
    }
}

// ============================================================================
// Habitat Complex (collection of modules)
// ============================================================================

/// The complete physical infrastructure of a colony.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HabitatComplex {
    pub modules: Vec<HabitatModule>,
}

impl HabitatComplex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total pressurized volume (operational modules only).
    pub fn total_volume(&self) -> f64 {
        self.modules
            .iter()
            .filter(|m| m.operational)
            .map(|m| m.volume_m3)
            .sum()
    }

    /// Total habitation capacity.
    pub fn total_capacity(&self) -> u32 {
        self.modules
            .iter()
            .filter(|m| m.operational)
            .map(|m| m.capacity)
            .sum()
    }

    /// Compute the weighted radiation dose for an agent based on their profession.
    ///
    /// The agent spends work_hours in their work module, rest_hours in habitation,
    /// and some time in commons. The dose is time-weighted across modules.
    ///
    /// Returns: dose_fraction [0, 1] — multiply by ambient dose to get actual dose.
    pub fn agent_dose_fraction(&self, primary_sector: usize) -> f64 {
        if self.modules.is_empty() {
            return 1.0;
        } // No modules = surface

        // Find the module where this agent works
        let work_module = self
            .modules
            .iter()
            .filter(|m| m.operational)
            .find(|m| m.function.primary_sector() == primary_sector);

        // Find habitation and commons modules
        let hab_module = self
            .modules
            .iter()
            .filter(|m| m.operational)
            .find(|m| m.function == ModuleFunction::Habitation);
        let commons_module = self
            .modules
            .iter()
            .filter(|m| m.operational)
            .find(|m| m.function == ModuleFunction::Commons);

        // Time-weighted dose
        let work_hours = 8.0;
        let commons_hours = 2.0;
        let hab_hours = 14.0;
        let total_hours = 24.0;

        let work_dose = work_module
            .map(|m| m.shielding.dose_fraction())
            .unwrap_or(1.0); // If no work module, surface exposure
        let hab_dose = hab_module
            .map(|m| m.shielding.dose_fraction())
            .unwrap_or(1.0);
        let commons_dose = commons_module
            .map(|m| m.shielding.dose_fraction())
            .unwrap_or(hab_dose); // Default to hab if no commons

        (work_hours * work_dose + commons_hours * commons_dose + hab_hours * hab_dose) / total_hours
    }

    /// Compute the psychological load modifier for an agent based on their daily module exposure.
    pub fn agent_psych_modifier(&self, primary_sector: usize) -> f64 {
        if self.modules.is_empty() {
            return 0.003;
        } // No modules = no-window stress

        let work_module = self
            .modules
            .iter()
            .filter(|m| m.operational)
            .find(|m| m.function.primary_sector() == primary_sector);
        let hab_module = self
            .modules
            .iter()
            .filter(|m| m.operational)
            .find(|m| m.function == ModuleFunction::Habitation);
        let commons_module = self
            .modules
            .iter()
            .filter(|m| m.operational)
            .find(|m| m.function == ModuleFunction::Commons);

        let work_mod = work_module
            .map(|m| m.windows.load_modifier())
            .unwrap_or(0.003);
        let hab_mod = hab_module
            .map(|m| m.windows.load_modifier())
            .unwrap_or(0.003);
        let commons_mod = commons_module
            .map(|m| m.windows.load_modifier())
            .unwrap_or(0.001);

        // Time-weighted
        (8.0 * work_mod + 2.0 * commons_mod + 14.0 * hab_mod) / 24.0
    }

    /// Build the default habitat complex for a new colony.
    pub fn default_surface_colony(location: &str) -> Self {
        let mut complex = Self::new();

        // Early colonies are austere: sealed pressure vessels with minimal windows.
        // "Tin cans in space" — the ISS aesthetic, not the sci-fi aesthetic.
        // Window upgrades come later as projects (Water Window Nexus, Observation Dome).
        complex.modules.push(HabitatModule {
            name: "Primary Hab".into(),
            function: ModuleFunction::Habitation,
            volume_m3: 500.0,
            capacity: 200,
            occupancy: 0,
            shielding: ShieldingType::Unshielded,
            windows: WindowType::Simulated, // Screens, not real windows (early colony)
            operational: true,
            construction_start: None,
            construction_duration: 0,
        });

        complex.modules.push(HabitatModule {
            name: "Ag-Bay".into(),
            function: ModuleFunction::Agriculture,
            volume_m3: 300.0,
            capacity: 20,
            occupancy: 0,
            shielding: ShieldingType::Unshielded,
            windows: WindowType::PressureWindow, // Needs some light for crops, but not a dome
            operational: true,
            construction_start: None,
            construction_duration: 0,
        });

        complex.modules.push(HabitatModule {
            name: "Workshop".into(),
            function: ModuleFunction::Workshop,
            volume_m3: 200.0,
            capacity: 30,
            occupancy: 0,
            shielding: ShieldingType::Unshielded,
            windows: WindowType::None,
            operational: true,
            construction_start: None,
            construction_duration: 0,
        });

        complex.modules.push(HabitatModule {
            name: "ECLSS Bay".into(),
            function: ModuleFunction::LifeSupport,
            volume_m3: 100.0,
            capacity: 10,
            occupancy: 0,
            shielding: ShieldingType::Unshielded,
            windows: WindowType::None,
            operational: true,
            construction_start: None,
            construction_duration: 0,
        });

        // Location-specific additions
        match location {
            "Moon" | "Mars" => {
                // Surface colonies start unshielded but can be upgraded
            }
            "Europa" => {
                // Europa: under-ice habitat is naturally shielded
                for m in &mut complex.modules {
                    m.shielding = ShieldingType::Underground { depth_m: 5 };
                    m.windows = WindowType::Simulated; // Under ice, no natural light
                }
            }
            "Titan" => {
                // Titan: thick atmosphere provides some shielding
                // But extreme cold requires sealed habitats
                for m in &mut complex.modules {
                    m.windows = WindowType::PressureWindow; // Can see Titan's landscape
                }
            }
            _ => {}
        }

        complex
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shielding_dose_fractions() {
        assert!((ShieldingType::Unshielded.dose_fraction() - 1.0).abs() < 0.01);
        assert!(ShieldingType::LavaTube.dose_fraction() < 0.02);
        assert!(ShieldingType::RegolithBuried { depth_m: 2 }.dose_fraction() < 0.30);
        assert!(ShieldingType::WaterWall { thickness_cm: 20 }.dose_fraction() < 0.75);
    }

    #[test]
    fn test_botanist_vs_admin_dose() {
        let mut complex = HabitatComplex::default_surface_colony("Mars");

        // Add buried dormitories
        complex.modules[0].shielding = ShieldingType::RegolithBuried { depth_m: 2 };

        // Botanist works in Ag-Bay (unshielded dome), sleeps in buried hab
        let botanist_dose = complex.agent_dose_fraction(1); // agriculture sector

        // Engineer works in Workshop (unshielded), sleeps in buried hab
        let engineer_dose = complex.agent_dose_fraction(0); // engineering sector

        // Both work in unshielded modules, sleep in buried hab
        // Botanist's Ag-Bay is unshielded (1.0), hab is buried (0.25)
        // Weighted: (8*1.0 + 14*0.25) / 24 = (8 + 3.5) / 24 = 0.479
        assert!(botanist_dose < 0.6, "Botanist dose: {}", botanist_dose);
        assert!(botanist_dose > 0.3, "Botanist dose: {}", botanist_dose);
    }

    #[test]
    fn test_water_window_psychology() {
        let mut complex = HabitatComplex::default_surface_colony("Mars");

        // Add a water-window commons module
        complex.modules.push(HabitatModule {
            name: "Nexus".into(),
            function: ModuleFunction::Commons,
            volume_m3: 150.0,
            capacity: 50,
            occupancy: 0,
            shielding: ShieldingType::WaterWall { thickness_cm: 20 },
            windows: WindowType::WaterWindow,
            operational: true,
            construction_start: None,
            construction_duration: 0,
        });

        // Agent with water-window commons should have better psych than without
        let psych_with = complex.agent_psych_modifier(0);
        // A complex without commons defaults to hab windows
        let mut no_commons = complex.clone();
        no_commons
            .modules
            .retain(|m| m.function != ModuleFunction::Commons);
        let psych_without = no_commons.agent_psych_modifier(0);

        assert!(
            psych_with < psych_without,
            "Water window commons should improve psychology: {} vs {}",
            psych_with,
            psych_without
        );
    }

    #[test]
    fn test_lava_tube_colony() {
        let mut complex = HabitatComplex::default_surface_colony("Moon");

        // Move everything into a lava tube
        for m in &mut complex.modules {
            m.shielding = ShieldingType::LavaTube;
            m.windows = WindowType::Simulated; // No natural light in tubes
        }

        // Dose should be near zero
        let dose = complex.agent_dose_fraction(0);
        assert!(dose < 0.02, "Lava tube dose: {}", dose);
    }
}
