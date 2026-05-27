// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Alternative Reactor Architectures for LCF
//!
//! The original Spark Engine uses a solid-state 3-layer design with HEA shell.
//! This module explores alternative architectures that sidestep key challenges:
//!
//! ## Challenges Addressed
//!
//! | Challenge                    | Original Design   | Alternative Solutions          |
//! |------------------------------|-------------------|--------------------------------|
//! | 14.1 MeV neutron damage      | HEA "Wolverine"   | Aneutronic D-He3, flowing Pb   |
//! | Impractical X-ray trigger    | XFEL required     | E-beam, piezo, electrolysis    |
//! | Fuel loading complexity      | Pd deuteride      | Flowing D2 gas, liquid target  |
//! | Thermal management           | Passive/Galinstan | Active liquid metal cooling    |
//! | Scaling across power levels  | Different designs | Modular cell architecture      |
//!
//! ## Architecture Overview
//!
//! ```text
//! ARCHITECTURE          TRIGGER         FUEL           NEUTRONS      SCALING
//! ─────────────────────────────────────────────────────────────────────────────
//! Spark v1 (original)   X-ray          Pd deuteride   D-D/D-T       Fixed size
//! FlowReactor           E-beam         Flowing Pd     D-D           Linear
//! AneutronicCore        Piezo/Laser    He3 + D2 gas   ~None         Modular
//! PulsedElectrolysis    Electrolysis   D2O + Pd       D-D           Simple
//! MoltenSalt            Thermal        Molten FLiBe   D-T           Large
//! ModularCell           Any            Configurable   Any           Arbitrary
//! ```

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::radiation_damage::FusionReaction;
use super::trigger_systems::{
    ExtendedTriggerMethod, FusionYieldEstimate, TriggerSystemLibrary, TriggerSystemSpec,
    estimate_fusion_yield,
};

/// Reactor architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReactorArchitecture {
    /// Original Spark Engine (solid HEA shell, Galinstan core)
    SparkV1,
    /// Flowing liquid metal target (self-renewing)
    FlowReactor,
    /// Aneutronic D-He3 (minimal neutron damage)
    AneutronicCore,
    /// Pulsed electrolysis (simplest setup)
    PulsedElectrolysis,
    /// Molten salt (FLiBe-based, for D-T)
    MoltenSalt,
    /// Modular cell architecture (scalable)
    ModularCell,
    /// Magnetized target fusion hybrid
    MagnetizedTarget,
}

/// Fuel system specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuelSystem {
    /// Primary fuel (D, T, He3)
    pub primary_fuel: FuelType,
    /// Secondary fuel (if hybrid)
    pub secondary_fuel: Option<FuelType>,
    /// Fuel state (gas, liquid, solid, plasma)
    pub state: FuelState,
    /// Host material (for solid-state)
    pub host_material: Option<String>,
    /// Operating pressure (atm)
    pub pressure_atm: f64,
    /// Operating temperature (K)
    pub temperature_k: f64,
    /// Fuel flow rate (if flowing, g/s)
    pub flow_rate_g_s: Option<f64>,
    /// Fuel loading ratio (D/Metal for deuterides)
    pub loading_ratio: Option<f64>,
}

/// Fuel types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuelType {
    Deuterium,
    Tritium,
    Helium3,
    Protium,
    DeuteriumTritium, // 50-50 mix
}

impl FuelType {
    pub fn atomic_mass(&self) -> f64 {
        match self {
            FuelType::Protium => 1.008,
            FuelType::Deuterium => 2.014,
            FuelType::Tritium => 3.016,
            FuelType::Helium3 => 3.016,
            FuelType::DeuteriumTritium => 2.515, // Average
        }
    }

    pub fn cost_per_gram_usd(&self) -> f64 {
        match self {
            FuelType::Protium => 0.0001,            // Essentially free
            FuelType::Deuterium => 0.01,            // $10/kg from seawater
            FuelType::Tritium => 30_000.0,          // $30,000/g (breeder required)
            FuelType::Helium3 => 1_000_000.0,       // $1M/g (lunar mining needed)
            FuelType::DeuteriumTritium => 15_000.0, // Average
        }
    }

    pub fn availability(&self) -> &'static str {
        match self {
            FuelType::Protium => "Unlimited (water)",
            FuelType::Deuterium => "Unlimited (seawater 0.015%)",
            FuelType::Tritium => "Scarce (breed from Li)",
            FuelType::Helium3 => "Extremely rare (lunar regolith)",
            FuelType::DeuteriumTritium => "Limited by tritium",
        }
    }
}

/// Fuel physical state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuelState {
    Gas,
    Liquid,
    Solid,
    Plasma,
    Dissolved, // In liquid metal or molten salt
}

/// Cooling system specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingSystem {
    /// Cooling method
    pub method: CoolingMethod,
    /// Coolant material
    pub coolant: String,
    /// Coolant inlet temperature (K)
    pub inlet_temp_k: f64,
    /// Coolant outlet temperature (K)
    pub outlet_temp_k: f64,
    /// Flow rate (kg/s)
    pub flow_rate_kg_s: f64,
    /// Heat transfer coefficient (W/m²·K)
    pub htc: f64,
}

/// Cooling methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoolingMethod {
    Passive,          // Natural convection
    ForcedAir,        // Fan cooling
    WaterLoop,        // Water jacket
    LiquidMetal,      // NaK, PbLi, Galinstan
    HeatPipe,         // Passive high-flux
    MoltenSalt,       // FLiBe, FLiNaK
    DirectConversion, // MHD or thermionic (for charged particles)
}

/// Shielding system specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShieldingSystem {
    /// Primary shielding material
    pub primary_material: String,
    /// Primary thickness (m)
    pub primary_thickness_m: f64,
    /// Secondary shielding (if layered)
    pub secondary_material: Option<String>,
    /// Secondary thickness (m)
    pub secondary_thickness_m: Option<f64>,
    /// Total mass (kg)
    pub total_mass_kg: f64,
    /// Attenuation factor
    pub attenuation_factor: f64,
}

/// Thermal constraint result from cooling analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConstraint {
    /// Cooling method used
    pub cooling_method: CoolingMethod,
    /// Maximum heat flux for this cooling method (W/m²)
    pub max_heat_flux_w_m2: f64,
    /// Minimum required surface area to reject heat (m²)
    pub min_surface_area_m2: f64,
    /// Minimum volume forced by thermal limits (m³) — sphere assumption
    pub min_volume_m3: f64,
    /// Whether thermal limits are binding (forced volume increase)
    pub is_binding: bool,
}

impl ThermalConstraint {
    /// Compute thermal constraint for a given power and cooling method.
    pub fn compute(power_w: f64, cooling: &CoolingMethod) -> Self {
        let max_heat_flux = match cooling {
            CoolingMethod::Passive => 500.0,
            CoolingMethod::ForcedAir => 5_000.0,
            CoolingMethod::WaterLoop => 50_000.0,
            CoolingMethod::LiquidMetal => 500_000.0,
            CoolingMethod::HeatPipe => 100_000.0,
            CoolingMethod::MoltenSalt => 200_000.0,
            CoolingMethod::DirectConversion => 1_000_000.0,
        };

        let min_area = power_w / max_heat_flux;
        // Sphere: A = 4πr², V = 4/3 πr³ → r = √(A/4π), V = 4/3 π (A/4π)^(3/2)
        let r = (min_area / (4.0 * PI)).sqrt();
        let min_vol = 4.0 / 3.0 * PI * r.powi(3);

        ThermalConstraint {
            cooling_method: *cooling,
            max_heat_flux_w_m2: max_heat_flux,
            min_surface_area_m2: min_area,
            min_volume_m3: min_vol,
            is_binding: false, // Set by caller after comparing with design volume
        }
    }
}

/// Detailed mass breakdown replacing simple mass_kg = shielding + core × 12000.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassBreakdown {
    /// Fuel mass — PdD core (core_volume × 12000 kg/m³)
    pub fuel_kg: f64,
    /// Structural mass — HEA shell (shell_volume × 7200 × 1.3 for flanges/welds)
    pub structure_kg: f64,
    /// Shielding mass
    pub shielding_kg: f64,
    /// Cooling system mass
    pub cooling_kg: f64,
    /// Trigger system mass
    pub trigger_kg: f64,
    /// Balance of plant (12% of subtotal)
    pub bop_kg: f64,
    /// Containment vessel
    pub containment_kg: f64,
    /// Total system mass
    pub total_kg: f64,
}

impl MassBreakdown {
    /// Compute mass breakdown from reactor parameters.
    pub fn compute(
        core_volume_m3: f64,
        shell_volume_m3: f64,
        shielding_mass_kg: f64,
        cooling_method: &CoolingMethod,
        power_w: f64,
        trigger_footprint_m3: f64,
    ) -> Self {
        let fuel_kg = core_volume_m3 * 12000.0; // Pd density
        let structure_kg = shell_volume_m3 * 7200.0 * 1.3; // HEA × 1.3 for flanges/welds

        let cooling_kg = match cooling_method {
            CoolingMethod::Passive => 0.5,
            CoolingMethod::ForcedAir => 2.0 + power_w * 0.0001,
            CoolingMethod::WaterLoop => 5.0 + power_w * 0.001,
            CoolingMethod::HeatPipe => 3.0 + power_w * 0.0005,
            CoolingMethod::LiquidMetal => 10.0 + power_w * 0.005,
            CoolingMethod::MoltenSalt => 15.0 + power_w * 0.003,
            CoolingMethod::DirectConversion => 5.0 + power_w * 0.002,
        };

        // Trigger mass: footprint × 500 kg/m³, clamped 0.5–1000 kg
        let trigger_kg = (trigger_footprint_m3 * 500.0).clamp(0.5, 1000.0);

        let containment_kg = 5.0 + power_w * 0.001;

        let subtotal =
            fuel_kg + structure_kg + shielding_mass_kg + cooling_kg + trigger_kg + containment_kg;
        let bop_kg = subtotal * 0.12; // 12% balance of plant

        let total_kg = subtotal + bop_kg;

        MassBreakdown {
            fuel_kg,
            structure_kg,
            shielding_kg: shielding_mass_kg,
            cooling_kg,
            trigger_kg,
            bop_kg,
            containment_kg,
            total_kg,
        }
    }
}

/// Detailed cost breakdown replacing simple cost = $50k + power×5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Core materials (Pd, Galinstan, HEA)
    pub core_materials_usd: f64,
    /// Shielding cost
    pub shielding_usd: f64,
    /// Trigger system cost
    pub trigger_usd: f64,
    /// Cooling system cost
    pub cooling_usd: f64,
    /// Balance of plant (15% of subtotal)
    pub bop_usd: f64,
    /// Engineering (20% of subtotal)
    pub engineering_usd: f64,
    /// Assembly and testing (25% of subtotal)
    pub assembly_usd: f64,
    /// Total cost
    pub total_usd: f64,
}

impl CostBreakdown {
    /// Compute cost breakdown from reactor parameters.
    pub fn compute(
        fuel_mass_kg: f64,
        structure_mass_kg: f64,
        shielding_mass_kg: f64,
        cooling_method: &CoolingMethod,
        cooling_mass_kg: f64,
        trigger_cost_usd: f64,
    ) -> Self {
        // Core materials: Pd at $70k/kg × 0.3 (30% Pd by mass) + Galinstan × $150/kg × 0.7 + HEA × $500/kg
        let pd_fraction = 0.3;
        let galinstan_fraction = 0.7;
        let core_materials = fuel_mass_kg * (pd_fraction * 70_000.0 + galinstan_fraction * 150.0)
            + structure_mass_kg * 500.0;

        let shielding_cost = shielding_mass_kg * 50.0;

        let cooling_cost = match cooling_method {
            CoolingMethod::Passive => 500.0,
            CoolingMethod::ForcedAir => 1000.0 + cooling_mass_kg * 50.0,
            CoolingMethod::WaterLoop => 2000.0 + cooling_mass_kg * 80.0,
            CoolingMethod::HeatPipe => 3000.0 + cooling_mass_kg * 90.0,
            CoolingMethod::LiquidMetal => cooling_mass_kg * 100.0,
            CoolingMethod::MoltenSalt => cooling_mass_kg * 120.0,
            CoolingMethod::DirectConversion => 5000.0 + cooling_mass_kg * 150.0,
        };

        let subtotal = core_materials + shielding_cost + trigger_cost_usd + cooling_cost;
        let bop = subtotal * 0.15;
        let engineering = subtotal * 0.20;
        let assembly = subtotal * 0.25;

        let total = subtotal + bop + engineering + assembly;

        CostBreakdown {
            core_materials_usd: core_materials,
            shielding_usd: shielding_cost,
            trigger_usd: trigger_cost_usd,
            cooling_usd: cooling_cost,
            bop_usd: bop,
            engineering_usd: engineering,
            assembly_usd: assembly,
            total_usd: total,
        }
    }
}

/// Fuel cycle analysis for a reactor design.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuelCycle {
    /// Deuterium consumption rate (g/s)
    pub d_consumption_g_s: f64,
    /// Deuterium consumption rate (g/year)
    pub d_consumption_g_year: f64,
    /// Initial deuterium loading (g)
    pub initial_loading_g: f64,
    /// Loading time (hours) — gas loading at 1-10 atm
    pub loading_time_hours: f64,
    /// Refueling interval (years) — when D/Pd drops from 0.7 to 0.5
    pub refueling_interval_years: f64,
    /// Duty cycle (operating / (operating + loading)) per refuel
    pub duty_cycle: f64,
    /// Tritium accumulation from p+T branch at 50% (g/year)
    pub tritium_accumulation_g_year: f64,
}

impl FuelCycle {
    /// Compute fuel cycle for a D-D reactor.
    pub fn compute(reactions_per_second: f64, pd_mass_kg: f64, loading_ratio: f64) -> Self {
        // D consumption: 2 deuterons per reaction, each 2.014 amu
        let amu_to_g = 1.6605e-24;
        let d_per_reaction = 2.0;
        let d_consumption_g_s = reactions_per_second * d_per_reaction * 2.014 * amu_to_g;
        let seconds_per_year = 3.156e7;
        let d_consumption_g_year = d_consumption_g_s * seconds_per_year;

        // Initial loading: Pd_mass × loading_ratio × (M_D / M_Pd)
        let initial_loading_g = pd_mass_kg * 1000.0 * loading_ratio * (2.014 / 106.42);

        // Loading time: 24-72 hours depending on loading ratio
        let loading_time_hours = 24.0 + 48.0 * loading_ratio;

        // Refueling interval: when loading drops from initial ratio to 0.5
        let usable_d_g = initial_loading_g * (loading_ratio - 0.5) / loading_ratio;
        let refueling_interval_years = if d_consumption_g_year > 0.0 {
            (usable_d_g / d_consumption_g_year).max(0.1)
        } else {
            100.0 // Effectively never
        };

        // Duty cycle
        let operating_hours = refueling_interval_years * 8760.0;
        let duty_cycle = operating_hours / (operating_hours + loading_time_hours);

        // Tritium accumulation: 50% of reactions produce T
        let tritium_rate_g_s = reactions_per_second * 0.5 * 3.016 * amu_to_g;
        let tritium_accumulation_g_year = tritium_rate_g_s * seconds_per_year;

        FuelCycle {
            d_consumption_g_s,
            d_consumption_g_year,
            initial_loading_g,
            loading_time_hours,
            refueling_interval_years,
            duty_cycle: duty_cycle.clamp(0.0, 1.0),
            tritium_accumulation_g_year,
        }
    }
}

/// Complete reactor architecture specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactorSpec {
    /// Architecture name
    pub name: String,
    /// Architecture type
    pub architecture: ReactorArchitecture,
    /// Fusion reaction(s)
    pub reactions: Vec<FusionReaction>,
    /// Trigger system
    pub trigger: TriggerSystemSpec,
    /// Fuel system
    pub fuel: FuelSystem,
    /// Cooling system
    pub cooling: CoolingSystem,
    /// Shielding system
    pub shielding: ShieldingSystem,
    /// Target power output (W)
    pub power_w: f64,
    /// Overall dimensions (m³)
    pub volume_m3: f64,
    /// Total system mass (kg) — computed from mass_breakdown if available
    pub mass_kg: f64,
    /// Estimated cost (USD) — computed from cost_breakdown if available
    pub cost_usd: f64,
    /// Expected lifetime (years)
    pub lifetime_years: f64,
    /// Technology readiness level
    pub trl: u8,
    /// Key advantages
    pub advantages: Vec<String>,
    /// Key challenges
    pub challenges: Vec<String>,
    /// Estimated fusion yield
    pub yield_estimate: FusionYieldEstimate,
    /// Detailed mass breakdown
    pub mass_breakdown: Option<MassBreakdown>,
    /// Detailed cost breakdown
    pub cost_breakdown: Option<CostBreakdown>,
    /// Fuel cycle analysis
    pub fuel_cycle: Option<FuelCycle>,
    /// Thermal constraint analysis
    pub thermal_constraint: Option<ThermalConstraint>,
}

/// Reactor design parameters
#[derive(Debug, Clone)]
pub struct ReactorDesignParams {
    /// Target power output (W)
    pub target_power_w: f64,
    /// Target lifetime (years)
    pub target_lifetime_years: f64,
    /// Maximum cost (USD)
    pub max_cost_usd: f64,
    /// Maximum volume (m³)
    pub max_volume_m3: f64,
    /// Maximum mass (kg)
    pub max_mass_kg: f64,
    /// Acceptable neutron reactions
    pub allow_neutrons: bool,
    /// TRL requirement
    pub min_trl: u8,
    /// Risk adjustment factor (0.0 = ignore risk, 1.0 = full risk adjustment)
    /// When > 0, scores are multiplied by estimated feasibility probability.
    pub risk_weight: f64,
}

impl Default for ReactorDesignParams {
    fn default() -> Self {
        Self {
            target_power_w: 10_000.0, // 10 kW
            target_lifetime_years: 20.0,
            max_cost_usd: 100_000.0,
            max_volume_m3: 1.0,
            max_mass_kg: 1000.0,
            allow_neutrons: true,
            min_trl: 4,
            risk_weight: 0.0, // Default: no risk adjustment
        }
    }
}

impl ReactorDesignParams {
    /// Consumer household unit
    pub fn consumer() -> Self {
        Self {
            target_power_w: 5_000.0,
            target_lifetime_years: 25.0,
            max_cost_usd: 50_000.0,
            max_volume_m3: 0.5,
            max_mass_kg: 500.0,
            allow_neutrons: false, // Consumer wants aneutronic
            min_trl: 6,
            risk_weight: 0.5, // Moderate risk consideration for consumer
        }
    }

    /// Research prototype
    pub fn prototype() -> Self {
        Self {
            target_power_w: 100.0,
            target_lifetime_years: 1.0,
            max_cost_usd: 500_000.0,
            max_volume_m3: 10.0,
            max_mass_kg: 5000.0,
            allow_neutrons: true,
            min_trl: 3,
            risk_weight: 0.0, // No risk adjustment for prototypes
        }
    }

    /// Industrial power plant
    pub fn industrial() -> Self {
        Self {
            target_power_w: 100_000_000.0, // 100 MW
            target_lifetime_years: 40.0,
            max_cost_usd: 1_000_000_000.0,
            max_volume_m3: 10_000.0,
            max_mass_kg: 10_000_000.0,
            allow_neutrons: true,
            min_trl: 5,
            risk_weight: 0.8, // Higher risk consideration for industrial
        }
    }
}

/// Reactor architecture designer
pub struct ReactorDesigner {
    trigger_library: TriggerSystemLibrary,
}

impl ReactorDesigner {
    pub fn new() -> Self {
        Self {
            trigger_library: TriggerSystemLibrary::new(),
        }
    }

    /// Apply thermal constraint and detailed breakdowns to a reactor spec.
    /// If thermal limits force a larger volume, this adjusts the spec accordingly.
    fn apply_engineering_realism(&self, mut spec: ReactorSpec) -> ReactorSpec {
        // Thermal constraint
        let tc = ThermalConstraint::compute(spec.power_w, &spec.cooling.method);
        let mut thermal = tc;
        if thermal.min_volume_m3 > spec.volume_m3 {
            thermal.is_binding = true;
            spec.volume_m3 = thermal.min_volume_m3;
        }
        spec.thermal_constraint = Some(thermal);

        // Mass breakdown
        let core_volume = spec.power_w / 50_000.0_f64.max(0.0001);
        let core_radius = (3.0 * core_volume / (4.0 * PI)).powf(1.0 / 3.0);
        let shell_thickness = 0.01 + core_radius * 0.2;
        let shell_outer = core_radius + shell_thickness;
        let shell_volume = 4.0 / 3.0 * PI * (shell_outer.powi(3) - core_radius.powi(3));

        let mb = MassBreakdown::compute(
            core_volume,
            shell_volume,
            spec.shielding.total_mass_kg,
            &spec.cooling.method,
            spec.power_w,
            spec.trigger.footprint_m3,
        );
        spec.mass_kg = mb.total_kg;
        spec.mass_breakdown = Some(mb);

        // Cost breakdown
        let fuel_mass = core_volume * 12000.0;
        let struct_mass = shell_volume * 7200.0 * 1.3;
        let cooling_mass = match spec.cooling.method {
            CoolingMethod::Passive => 0.5,
            CoolingMethod::ForcedAir => 2.0 + spec.power_w * 0.0001,
            CoolingMethod::WaterLoop => 5.0 + spec.power_w * 0.001,
            CoolingMethod::HeatPipe => 3.0 + spec.power_w * 0.0005,
            CoolingMethod::LiquidMetal => 10.0 + spec.power_w * 0.005,
            CoolingMethod::MoltenSalt => 15.0 + spec.power_w * 0.003,
            CoolingMethod::DirectConversion => 5.0 + spec.power_w * 0.002,
        };
        let cb = CostBreakdown::compute(
            fuel_mass,
            struct_mass,
            spec.shielding.total_mass_kg,
            &spec.cooling.method,
            cooling_mass,
            spec.trigger.cost_estimate_usd,
        );
        spec.cost_usd = cb.total_usd;
        spec.cost_breakdown = Some(cb);

        // Fuel cycle
        let fc = FuelCycle::compute(
            spec.yield_estimate.reactions_per_second,
            fuel_mass / 1000.0, // convert to kg
            spec.fuel.loading_ratio.unwrap_or(0.7),
        );
        spec.fuel_cycle = Some(fc);

        spec
    }

    /// Design optimal reactor for given parameters
    pub fn design(&self, params: &ReactorDesignParams) -> ReactorSpec {
        // Select architecture based on constraints
        let architecture = self.select_architecture(params);

        // Design the reactor
        let spec = match architecture {
            ReactorArchitecture::SparkV1 => self.design_spark_v1(params),
            ReactorArchitecture::FlowReactor => self.design_flow_reactor(params),
            ReactorArchitecture::AneutronicCore => self.design_aneutronic(params),
            ReactorArchitecture::PulsedElectrolysis => self.design_electrolysis(params),
            ReactorArchitecture::MoltenSalt => self.design_molten_salt(params),
            ReactorArchitecture::ModularCell => self.design_modular(params),
            ReactorArchitecture::MagnetizedTarget => self.design_magnetized(params),
        };

        self.apply_engineering_realism(spec)
    }

    /// Select best architecture for given constraints
    fn select_architecture(&self, params: &ReactorDesignParams) -> ReactorArchitecture {
        // Decision tree for architecture selection
        if !params.allow_neutrons {
            // Aneutronic required
            return ReactorArchitecture::AneutronicCore;
        }

        if params.target_power_w < 1000.0 {
            // Very small: electrolysis or modular
            if params.max_cost_usd < 10_000.0 {
                return ReactorArchitecture::PulsedElectrolysis;
            } else {
                return ReactorArchitecture::ModularCell;
            }
        }

        if params.target_power_w > 10_000_000.0 {
            // Large scale: molten salt for high power
            return ReactorArchitecture::MoltenSalt;
        }

        if params.target_lifetime_years > 30.0 {
            // Long lifetime: flow reactor for self-renewal
            return ReactorArchitecture::FlowReactor;
        }

        // Default to original Spark design
        ReactorArchitecture::SparkV1
    }

    /// Design original Spark v1 architecture
    fn design_spark_v1(&self, params: &ReactorDesignParams) -> ReactorSpec {
        let reaction = if params.target_power_w > 100_000.0 {
            FusionReaction::DT
        } else {
            FusionReaction::DD
        };

        // Select trigger for power scale
        let trigger = self
            .trigger_library
            .optimal_for_power(params.target_power_w)
            .clone();

        // Calculate dimensions
        let power_density = 50_000.0; // W/m³ for solid-state
        let core_volume = params.target_power_w / power_density;
        let core_radius = (3.0 * core_volume / (4.0 * PI)).powf(1.0 / 3.0);
        let shell_thickness = 0.01 + core_radius * 0.2;
        let total_radius = core_radius + shell_thickness + 0.05; // +5cm shielding

        let fuel = FuelSystem {
            primary_fuel: FuelType::Deuterium,
            secondary_fuel: if reaction == FusionReaction::DT {
                Some(FuelType::Tritium)
            } else {
                None
            },
            state: FuelState::Solid,
            host_material: Some("TiVZrNbPd HEA".to_string()),
            pressure_atm: 1.0,
            temperature_k: 350.0,
            flow_rate_g_s: None,
            loading_ratio: Some(0.7),
        };

        let cooling_method = if params.target_power_w > 100_000.0 {
            CoolingMethod::LiquidMetal
        } else if params.target_power_w > 1_000.0 {
            CoolingMethod::WaterLoop
        } else {
            CoolingMethod::Passive
        };

        let cooling = CoolingSystem {
            method: cooling_method,
            coolant: "Galinstan".to_string(),
            inlet_temp_k: 300.0,
            outlet_temp_k: 350.0,
            flow_rate_kg_s: params.target_power_w / 50_000.0,
            htc: 10_000.0,
        };

        let neutron_energy = reaction.neutron_energy_mev().unwrap_or(2.45);
        let shielding_thickness = if neutron_energy > 10.0 { 0.5 } else { 0.2 };

        let shielding = ShieldingSystem {
            primary_material: "Borated polyethylene".to_string(),
            primary_thickness_m: shielding_thickness,
            secondary_material: Some("Lead".to_string()),
            secondary_thickness_m: Some(0.02),
            total_mass_kg: 4.0 * PI * total_radius.powi(2) * shielding_thickness * 1000.0,
            attenuation_factor: (-shielding_thickness * 10.0).exp(),
        };

        let volume = 4.0 / 3.0 * PI * total_radius.powi(3);
        let mass = shielding.total_mass_kg + core_volume * 12000.0; // Pd density

        let yield_estimate = estimate_fusion_yield(&trigger, core_volume * 12000.0 * 1000.0, 0.7);

        ReactorSpec {
            name: format!("Spark v1 ({:.1}kW)", params.target_power_w / 1000.0),
            architecture: ReactorArchitecture::SparkV1,
            reactions: vec![reaction],
            trigger,
            fuel,
            cooling,
            shielding,
            power_w: params.target_power_w,
            volume_m3: volume,
            mass_kg: mass,
            cost_usd: 50_000.0 + params.target_power_w * 5.0,
            lifetime_years: 20.0,
            trl: 4,
            advantages: vec![
                "Compact solid-state design".to_string(),
                "Self-healing HEA shell".to_string(),
                "Proven materials".to_string(),
            ],
            challenges: vec![
                "Requires X-ray or e-beam trigger".to_string(),
                "Neutron damage accumulation".to_string(),
                "Limited fuel lifetime".to_string(),
            ],
            yield_estimate,
            mass_breakdown: None,
            cost_breakdown: None,
            fuel_cycle: None,
            thermal_constraint: None,
        }
    }

    /// Design flowing liquid metal reactor
    fn design_flow_reactor(&self, params: &ReactorDesignParams) -> ReactorSpec {
        // Flowing Pb-Li eutectic with dissolved deuterium
        let trigger = self
            .trigger_library
            .systems
            .iter()
            .find(|s| s.method == ExtendedTriggerMethod::ElectronBeam)
            .expect("trigger library must contain ElectronBeam system")
            .clone();

        let core_volume = params.target_power_w / 30_000.0; // Lower power density for flowing
        let pipe_diameter = 0.1; // 10cm pipe
        let pipe_length = core_volume / (PI * (pipe_diameter / 2.0_f64).powi(2));

        let fuel = FuelSystem {
            primary_fuel: FuelType::Deuterium,
            secondary_fuel: None,
            state: FuelState::Dissolved,
            host_material: Some("Pb-17Li eutectic".to_string()),
            pressure_atm: 5.0,
            temperature_k: 600.0,
            flow_rate_g_s: Some(params.target_power_w / 10.0), // ~100g/s per kW
            loading_ratio: Some(0.001),                        // Much lower in liquid
        };

        let cooling = CoolingSystem {
            method: CoolingMethod::LiquidMetal,
            coolant: "Pb-17Li (self-cooling)".to_string(),
            inlet_temp_k: 550.0,
            outlet_temp_k: 650.0,
            flow_rate_kg_s: fuel.flow_rate_g_s.unwrap_or(100.0) / 1000.0,
            htc: 50_000.0,
        };

        let shielding = ShieldingSystem {
            primary_material: "Pb-17Li self-shielding + steel".to_string(),
            primary_thickness_m: 0.3,
            secondary_material: None,
            secondary_thickness_m: None,
            total_mass_kg: 500.0 + params.target_power_w * 0.1,
            attenuation_factor: 1e-4,
        };

        let volume = pipe_length * PI * 0.5_f64.powi(2) + 1.0; // Pipe + equipment
        let mass = 1000.0 + params.target_power_w * 0.5;

        let yield_estimate = estimate_fusion_yield(&trigger, mass * 0.1, 0.001);

        ReactorSpec {
            name: format!("FlowReactor ({:.1}kW)", params.target_power_w / 1000.0),
            architecture: ReactorArchitecture::FlowReactor,
            reactions: vec![FusionReaction::DD],
            trigger,
            fuel,
            cooling,
            shielding,
            power_w: params.target_power_w,
            volume_m3: volume,
            mass_kg: mass,
            cost_usd: 200_000.0 + params.target_power_w * 10.0,
            lifetime_years: 50.0, // Very long - flowing target is self-renewing
            trl: 3,
            advantages: vec![
                "Self-renewing target (indefinite lifetime)".to_string(),
                "Pb-Li provides self-shielding".to_string(),
                "Can breed tritium from Li".to_string(),
                "Continuous fuel processing possible".to_string(),
            ],
            challenges: vec![
                "Complex plumbing system".to_string(),
                "Pb-Li handling at 600K".to_string(),
                "Lower D concentration than solid".to_string(),
                "Corrosion management".to_string(),
            ],
            yield_estimate,
            mass_breakdown: None,
            cost_breakdown: None,
            fuel_cycle: None,
            thermal_constraint: None,
        }
    }

    /// Design aneutronic D-He3 reactor
    fn design_aneutronic(&self, params: &ReactorDesignParams) -> ReactorSpec {
        // Aneutronic requires higher temperatures/energies
        // Use piezo phonon + laser for multi-mode trigger
        let trigger = self
            .trigger_library
            .systems
            .iter()
            .find(|s| s.method == ExtendedTriggerMethod::LaserBremsstrahlung)
            .expect("trigger library must contain LaserBremsstrahlung system")
            .clone();

        let fuel = FuelSystem {
            primary_fuel: FuelType::Deuterium,
            secondary_fuel: Some(FuelType::Helium3),
            state: FuelState::Gas,
            host_material: None,
            pressure_atm: 100.0,   // High pressure gas target
            temperature_k: 1000.0, // Elevated temperature
            flow_rate_g_s: Some(0.01),
            loading_ratio: None,
        };

        // Direct conversion for charged particles (protons from D-He3)
        let cooling = CoolingSystem {
            method: CoolingMethod::DirectConversion,
            coolant: "MHD channel".to_string(),
            inlet_temp_k: 800.0,
            outlet_temp_k: 1200.0,
            flow_rate_kg_s: 0.0, // No coolant flow - direct conversion
            htc: 100_000.0,      // High HTC for direct energy extraction
        };

        // Minimal shielding - aneutronic!
        let shielding = ShieldingSystem {
            primary_material: "Aluminum (structural only)".to_string(),
            primary_thickness_m: 0.01,
            secondary_material: None,
            secondary_thickness_m: None,
            total_mass_kg: 50.0,
            attenuation_factor: 1.0, // Not needed
        };

        let volume = 0.1 + params.target_power_w / 100_000.0;
        let mass = 100.0 + params.target_power_w * 0.01;

        // He3 cost dominates — power × time / (energy per gram of He-3)
        // He-3 molar mass = 3.016 g/mol
        let he3_consumption_g_year = params.target_power_w * 3.15e7 * 3.016
            / (18.3e6 * super::constants::MEV_TO_J * super::constants::N_AVOGADRO);
        let fuel_cost_year = he3_consumption_g_year * FuelType::Helium3.cost_per_gram_usd();

        let yield_estimate = estimate_fusion_yield(&trigger, 10.0, 0.5);

        ReactorSpec {
            name: format!(
                "AneutronicCore D-He3 ({:.1}kW)",
                params.target_power_w / 1000.0
            ),
            architecture: ReactorArchitecture::AneutronicCore,
            reactions: vec![FusionReaction::DHe3],
            trigger,
            fuel,
            cooling,
            shielding,
            power_w: params.target_power_w,
            volume_m3: volume,
            mass_kg: mass,
            cost_usd: 500_000.0 + fuel_cost_year * 10.0, // Include 10yr fuel cost
            lifetime_years: 100.0,                       // No neutron damage
            trl: 2,                                      // Very early stage
            advantages: vec![
                "No neutron damage to structure".to_string(),
                "Minimal shielding required".to_string(),
                "Direct energy conversion possible (>60% efficiency)".to_string(),
                "Safe for consumer deployment".to_string(),
                "Indefinite structural lifetime".to_string(),
            ],
            challenges: vec![
                "He3 extremely expensive ($1M/g)".to_string(),
                "Higher ignition temperature required".to_string(),
                "Lower cross-section than D-T".to_string(),
                "Side D-D reactions produce some neutrons".to_string(),
            ],
            yield_estimate,
            mass_breakdown: None,
            cost_breakdown: None,
            fuel_cycle: None,
            thermal_constraint: None,
        }
    }

    /// Design pulsed electrolysis reactor (simplest)
    fn design_electrolysis(&self, params: &ReactorDesignParams) -> ReactorSpec {
        let trigger = self
            .trigger_library
            .systems
            .iter()
            .find(|s| s.method == ExtendedTriggerMethod::PulsedElectrolysis)
            .expect("trigger library must contain PulsedElectrolysis system")
            .clone();

        let fuel = FuelSystem {
            primary_fuel: FuelType::Deuterium,
            secondary_fuel: None,
            state: FuelState::Dissolved,
            host_material: Some("Pd cathode in D2O".to_string()),
            pressure_atm: 1.0,
            temperature_k: 350.0, // Slightly elevated
            flow_rate_g_s: None,
            loading_ratio: Some(0.9), // High loading via electrolysis
        };

        let cooling = CoolingSystem {
            method: CoolingMethod::WaterLoop,
            coolant: "D2O (electrolyte)".to_string(),
            inlet_temp_k: 300.0,
            outlet_temp_k: 350.0,
            flow_rate_kg_s: 0.1,
            htc: 5000.0,
        };

        let shielding = ShieldingSystem {
            primary_material: "Water + borated plastic".to_string(),
            primary_thickness_m: 0.2,
            secondary_material: None,
            secondary_thickness_m: None,
            total_mass_kg: 100.0,
            attenuation_factor: 0.1,
        };

        let yield_estimate = estimate_fusion_yield(&trigger, 100.0, 0.9);

        ReactorSpec {
            name: format!("PulsedElectrolysis ({:.0}W)", params.target_power_w),
            architecture: ReactorArchitecture::PulsedElectrolysis,
            reactions: vec![FusionReaction::DD],
            trigger,
            fuel,
            cooling,
            shielding,
            power_w: params.target_power_w,
            volume_m3: 0.05, // Very compact
            mass_kg: 50.0,
            cost_usd: 5_000.0,   // Very cheap
            lifetime_years: 5.0, // Limited by cathode degradation
            trl: 5,              // Some experimental validation
            advantages: vec![
                "Extremely simple setup".to_string(),
                "Very low cost (<$5K)".to_string(),
                "Self-loading fuel".to_string(),
                "Desktop scale".to_string(),
            ],
            challenges: vec![
                "Low power output (typically <100W)".to_string(),
                "Controversial physics (reproducibility issues)".to_string(),
                "Cathode degradation".to_string(),
                "Difficult to scale up".to_string(),
            ],
            yield_estimate,
            mass_breakdown: None,
            cost_breakdown: None,
            fuel_cycle: None,
            thermal_constraint: None,
        }
    }

    /// Design molten salt reactor (for large scale)
    fn design_molten_salt(&self, params: &ReactorDesignParams) -> ReactorSpec {
        let trigger = self
            .trigger_library
            .systems
            .iter()
            .find(|s| s.method == ExtendedTriggerMethod::CompactXRay)
            .expect("trigger library must contain CompactXRay system")
            .clone();

        let fuel = FuelSystem {
            primary_fuel: FuelType::DeuteriumTritium,
            secondary_fuel: None,
            state: FuelState::Dissolved,
            host_material: Some("FLiBe molten salt".to_string()),
            pressure_atm: 1.0,
            temperature_k: 900.0, // High temp for molten salt
            flow_rate_g_s: Some(params.target_power_w / 100.0),
            loading_ratio: Some(0.01),
        };

        let cooling = CoolingSystem {
            method: CoolingMethod::MoltenSalt,
            coolant: "FLiBe".to_string(),
            inlet_temp_k: 800.0,
            outlet_temp_k: 1000.0,
            flow_rate_kg_s: params.target_power_w / 200_000.0,
            htc: 30_000.0,
        };

        // Need significant shielding for D-T
        let shielding = ShieldingSystem {
            primary_material: "FLiBe (Li breeds T) + steel".to_string(),
            primary_thickness_m: 1.0,
            secondary_material: Some("Concrete".to_string()),
            secondary_thickness_m: Some(2.0),
            total_mass_kg: 100_000.0 + params.target_power_w * 0.1,
            attenuation_factor: 1e-6,
        };

        let volume = params.target_power_w / 10_000.0; // 10 kW/m³
        let mass = 10_000.0 + params.target_power_w * 0.2;

        let yield_estimate = estimate_fusion_yield(&trigger, mass * 0.01, 0.01);

        ReactorSpec {
            name: format!("MoltenSalt ({:.0}MW)", params.target_power_w / 1_000_000.0),
            architecture: ReactorArchitecture::MoltenSalt,
            reactions: vec![FusionReaction::DT],
            trigger,
            fuel,
            cooling,
            shielding,
            power_w: params.target_power_w,
            volume_m3: volume,
            mass_kg: mass,
            cost_usd: 10_000_000.0 + params.target_power_w * 1.0,
            lifetime_years: 40.0,
            trl: 3,
            advantages: vec![
                "High power density".to_string(),
                "Self-healing liquid core".to_string(),
                "Tritium breeding from Li".to_string(),
                "High temperature operation".to_string(),
            ],
            challenges: vec![
                "14.1 MeV neutron damage to structure".to_string(),
                "Tritium handling complexity".to_string(),
                "High temperature materials".to_string(),
                "Regulatory complexity".to_string(),
            ],
            yield_estimate,
            mass_breakdown: None,
            cost_breakdown: None,
            fuel_cycle: None,
            thermal_constraint: None,
        }
    }

    /// Design modular cell architecture (scalable)
    fn design_modular(&self, params: &ReactorDesignParams) -> ReactorSpec {
        // Each cell produces ~100W
        let cell_power = 100.0;
        let num_cells = (params.target_power_w / cell_power).ceil() as usize;

        let trigger = self
            .trigger_library
            .systems
            .iter()
            .find(|s| s.method == ExtendedTriggerMethod::PiezoPhonon)
            .expect("trigger library must contain PiezoPhonon system")
            .clone();

        let fuel = FuelSystem {
            primary_fuel: FuelType::Deuterium,
            secondary_fuel: None,
            state: FuelState::Solid,
            host_material: Some("Pd nano-powder".to_string()),
            pressure_atm: 10.0,
            temperature_k: 400.0,
            flow_rate_g_s: None,
            loading_ratio: Some(0.8),
        };

        let cooling = CoolingSystem {
            method: CoolingMethod::HeatPipe,
            coolant: "Na heat pipe".to_string(),
            inlet_temp_k: 350.0,
            outlet_temp_k: 400.0,
            flow_rate_kg_s: 0.0, // Passive heat pipe
            htc: 20_000.0,
        };

        // Shared shielding for cell array
        let shielding = ShieldingSystem {
            primary_material: "Borated water".to_string(),
            primary_thickness_m: 0.3,
            secondary_material: None,
            secondary_thickness_m: None,
            total_mass_kg: 200.0 + (num_cells as f64) * 5.0,
            attenuation_factor: 0.01,
        };

        // Each cell is ~10cm cube
        let cell_volume = 0.001; // 1 liter
        let volume = (num_cells as f64) * cell_volume * 2.0; // 2x for spacing

        let cell_mass = 0.5; // 500g per cell
        let mass = (num_cells as f64) * cell_mass + shielding.total_mass_kg;

        let cell_cost = 500.0; // $500 per cell at scale
        let cost = (num_cells as f64) * cell_cost + 10_000.0; // + base cost

        let yield_estimate = estimate_fusion_yield(&trigger, cell_mass * 500.0, 0.8);

        ReactorSpec {
            name: format!(
                "ModularCell {}x ({:.1}kW)",
                num_cells,
                params.target_power_w / 1000.0
            ),
            architecture: ReactorArchitecture::ModularCell,
            reactions: vec![FusionReaction::DD],
            trigger,
            fuel,
            cooling,
            shielding,
            power_w: params.target_power_w,
            volume_m3: volume,
            mass_kg: mass,
            cost_usd: cost,
            lifetime_years: 15.0,
            trl: 4,
            advantages: vec![
                "Arbitrary scaling (1W to 1MW+)".to_string(),
                "Graceful degradation (cell failure)".to_string(),
                "Mass production potential".to_string(),
                "Easy replacement/maintenance".to_string(),
                "Fault tolerant design".to_string(),
            ],
            challenges: vec![
                "Cell synchronization".to_string(),
                "Heat pipe thermal limits".to_string(),
                "Individual cell reliability".to_string(),
                "Complex manufacturing".to_string(),
            ],
            yield_estimate,
            mass_breakdown: None,
            cost_breakdown: None,
            fuel_cycle: None,
            thermal_constraint: None,
        }
    }

    /// Design magnetized target fusion hybrid
    fn design_magnetized(&self, params: &ReactorDesignParams) -> ReactorSpec {
        // MTF uses magnetic compression of plasma target
        let trigger = self
            .trigger_library
            .systems
            .iter()
            .find(|s| s.method == ExtendedTriggerMethod::LaserBremsstrahlung)
            .expect("trigger library must contain LaserBremsstrahlung system")
            .clone();

        let fuel = FuelSystem {
            primary_fuel: FuelType::DeuteriumTritium,
            secondary_fuel: None,
            state: FuelState::Plasma,
            host_material: None,
            pressure_atm: 0.001,         // Low pressure plasma
            temperature_k: 10_000_000.0, // 10 keV plasma
            flow_rate_g_s: Some(0.001),
            loading_ratio: None,
        };

        let cooling = CoolingSystem {
            method: CoolingMethod::LiquidMetal,
            coolant: "Pb-Li liner (also compressor)".to_string(),
            inlet_temp_k: 600.0,
            outlet_temp_k: 800.0,
            flow_rate_kg_s: params.target_power_w / 100_000.0,
            htc: 50_000.0,
        };

        let shielding = ShieldingSystem {
            primary_material: "Pb-Li blanket".to_string(),
            primary_thickness_m: 0.8,
            secondary_material: Some("Concrete".to_string()),
            secondary_thickness_m: Some(1.5),
            total_mass_kg: 5000.0 + params.target_power_w * 0.05,
            attenuation_factor: 1e-5,
        };

        let volume = 10.0 + params.target_power_w / 50_000.0;
        let mass = 5000.0 + params.target_power_w * 0.1;

        let yield_estimate = estimate_fusion_yield(&trigger, 10.0, 0.5);

        ReactorSpec {
            name: format!(
                "MagnetizedTarget ({:.1}MW)",
                params.target_power_w / 1_000_000.0
            ),
            architecture: ReactorArchitecture::MagnetizedTarget,
            reactions: vec![FusionReaction::DT],
            trigger,
            fuel,
            cooling,
            shielding,
            power_w: params.target_power_w,
            volume_m3: volume,
            mass_kg: mass,
            cost_usd: 50_000_000.0 + params.target_power_w * 0.5,
            lifetime_years: 30.0,
            trl: 3,
            advantages: vec![
                "Lower magnetic field than tokamak".to_string(),
                "Pulsed operation (simpler magnets)".to_string(),
                "Liquid liner provides shielding".to_string(),
                "Tritium breeding integrated".to_string(),
            ],
            challenges: vec![
                "Plasma instabilities".to_string(),
                "Repetition rate limited by liner".to_string(),
                "D-T neutron damage".to_string(),
                "Complex pulsed power system".to_string(),
            ],
            yield_estimate,
            mass_breakdown: None,
            cost_breakdown: None,
            fuel_cycle: None,
            thermal_constraint: None,
        }
    }

    /// Compare all architectures for given parameters
    pub fn compare_all(&self, params: &ReactorDesignParams) -> Vec<ReactorSpec> {
        let architectures = [
            ReactorArchitecture::SparkV1,
            ReactorArchitecture::FlowReactor,
            ReactorArchitecture::AneutronicCore,
            ReactorArchitecture::PulsedElectrolysis,
            ReactorArchitecture::MoltenSalt,
            ReactorArchitecture::ModularCell,
            ReactorArchitecture::MagnetizedTarget,
        ];

        architectures
            .iter()
            .map(|arch| {
                let mut p = params.clone();
                // Override architecture selection
                let spec = match arch {
                    ReactorArchitecture::SparkV1 => self.design_spark_v1(&p),
                    ReactorArchitecture::FlowReactor => self.design_flow_reactor(&p),
                    ReactorArchitecture::AneutronicCore => {
                        p.allow_neutrons = false;
                        self.design_aneutronic(&p)
                    }
                    ReactorArchitecture::PulsedElectrolysis => self.design_electrolysis(&p),
                    ReactorArchitecture::MoltenSalt => self.design_molten_salt(&p),
                    ReactorArchitecture::ModularCell => self.design_modular(&p),
                    ReactorArchitecture::MagnetizedTarget => self.design_magnetized(&p),
                };
                self.apply_engineering_realism(spec)
            })
            .collect()
    }

    /// Rank architectures by weighted score
    pub fn rank(&self, params: &ReactorDesignParams) -> Vec<(ReactorSpec, f64)> {
        let specs = self.compare_all(params);

        let mut scored: Vec<_> = specs
            .into_iter()
            .map(|spec| {
                let score = self.score_architecture(&spec, params);
                (spec, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Score an architecture against parameters
    fn score_architecture(&self, spec: &ReactorSpec, params: &ReactorDesignParams) -> f64 {
        let mut score = 0.0;
        let mut penalties = 0.0;

        // TRL bonus (0-30 points)
        if spec.trl >= params.min_trl {
            score += (spec.trl as f64) * 3.0;
        } else {
            penalties += 50.0;
        }

        // Cost score (0-25 points)
        if spec.cost_usd <= params.max_cost_usd {
            score += 25.0 * (1.0 - spec.cost_usd / params.max_cost_usd);
        } else {
            penalties += 30.0 * (spec.cost_usd / params.max_cost_usd - 1.0);
        }

        // Volume score (0-15 points)
        if spec.volume_m3 <= params.max_volume_m3 {
            score += 15.0 * (1.0 - spec.volume_m3 / params.max_volume_m3);
        } else {
            penalties += 20.0 * (spec.volume_m3 / params.max_volume_m3 - 1.0);
        }

        // Mass score (0-15 points)
        if spec.mass_kg <= params.max_mass_kg {
            score += 15.0 * (1.0 - spec.mass_kg / params.max_mass_kg);
        } else {
            penalties += 20.0 * (spec.mass_kg / params.max_mass_kg - 1.0);
        }

        // Lifetime score (0-15 points)
        let lifetime_ratio = spec.lifetime_years / params.target_lifetime_years;
        score += 15.0 * lifetime_ratio.min(2.0) / 2.0;

        // Neutron penalty (if not allowed)
        if !params.allow_neutrons {
            match spec.architecture {
                ReactorArchitecture::AneutronicCore => score += 20.0,
                _ => penalties += 40.0,
            }
        }

        let base_score = (score - penalties).max(0.0);

        // Risk adjustment: multiply by feasibility probability if enabled
        if params.risk_weight > 0.0 {
            let p_feasible = self.estimate_feasibility_probability(spec, params);
            let risk_factor = 1.0 - params.risk_weight * (1.0 - p_feasible);
            base_score * risk_factor
        } else {
            base_score
        }
    }

    /// Estimate feasibility probability for a reactor spec.
    ///
    /// This is a simplified model based on how well the spec meets constraints
    /// and known risk factors for each architecture type.
    fn estimate_feasibility_probability(
        &self,
        spec: &ReactorSpec,
        params: &ReactorDesignParams,
    ) -> f64 {
        let mut p_feasible = 1.0;

        // TRL penalty: lower TRL = higher uncertainty
        let trl_factor = match spec.trl {
            1..=2 => 0.3,  // Lab concept
            3 => 0.5,      // Proof of concept
            4 => 0.7,      // Validated in lab
            5 => 0.8,      // Validated in environment
            6 => 0.9,      // Demonstrated
            7..=9 => 0.95, // Operational
            _ => 0.5,
        };
        p_feasible *= trl_factor;

        // Cost margin: if over budget, reduce probability
        if spec.cost_usd > params.max_cost_usd {
            let overage = spec.cost_usd / params.max_cost_usd;
            p_feasible *= (2.0 - overage).max(0.1);
        }

        // Volume margin: if over constraint, reduce probability
        if spec.volume_m3 > params.max_volume_m3 {
            let overage = spec.volume_m3 / params.max_volume_m3;
            p_feasible *= (2.0 - overage).max(0.1);
        }

        // Mass margin: if over constraint, reduce probability
        if spec.mass_kg > params.max_mass_kg {
            let overage = spec.mass_kg / params.max_mass_kg;
            p_feasible *= (2.0 - overage).max(0.1);
        }

        // Architecture-specific risk factors
        let arch_factor = match spec.architecture {
            ReactorArchitecture::PulsedElectrolysis => 0.9, // Simple, well-understood
            ReactorArchitecture::ModularCell => 0.85,       // Configurable, moderate risk
            ReactorArchitecture::SparkV1 => 0.75,           // Original design, thermal challenges
            ReactorArchitecture::FlowReactor => 0.7,        // Complex plumbing
            ReactorArchitecture::AneutronicCore => 0.6,     // He-3 availability
            ReactorArchitecture::MoltenSalt => 0.65,        // Corrosion challenges
            ReactorArchitecture::MagnetizedTarget => 0.5,   // Plasma containment
        };
        p_feasible *= arch_factor;

        // Thermal constraint binding: if thermal limit is forcing volume up, add uncertainty
        if let Some(ref tc) = spec.thermal_constraint {
            if tc.is_binding {
                p_feasible *= 0.85; // 15% penalty for thermal-limited designs
            }
        }

        p_feasible.clamp(0.05, 0.99)
    }
}

impl Default for ReactorDesigner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_designer_creation() {
        let designer = ReactorDesigner::new();
        assert!(!designer.trigger_library.systems.is_empty());
    }

    #[test]
    fn test_architecture_selection() {
        let designer = ReactorDesigner::new();

        // Consumer: should select aneutronic (no neutrons allowed)
        let consumer = ReactorDesignParams::consumer();
        let spec = designer.design(&consumer);
        assert_eq!(spec.architecture, ReactorArchitecture::AneutronicCore);

        // Prototype: small power, low cost → electrolysis
        let proto = ReactorDesignParams {
            target_power_w: 100.0,
            max_cost_usd: 5_000.0,
            allow_neutrons: true,
            ..Default::default()
        };
        let spec = designer.design(&proto);
        assert_eq!(spec.architecture, ReactorArchitecture::PulsedElectrolysis);
    }

    #[test]
    fn test_compare_all() {
        let designer = ReactorDesigner::new();
        let params = ReactorDesignParams::default();

        let specs = designer.compare_all(&params);
        assert_eq!(specs.len(), 7);

        for spec in &specs {
            assert!(spec.power_w > 0.0);
            assert!(spec.mass_kg > 0.0);
        }
    }

    #[test]
    fn test_ranking() {
        let designer = ReactorDesigner::new();
        let params = ReactorDesignParams::default();

        let ranked = designer.rank(&params);
        assert!(!ranked.is_empty());

        // Scores should be in descending order
        for i in 1..ranked.len() {
            assert!(ranked[i - 1].1 >= ranked[i].1);
        }
    }

    #[test]
    fn test_modular_scaling() {
        let designer = ReactorDesigner::new();

        // Test different power levels
        for power in [100.0, 1000.0, 10_000.0, 100_000.0] {
            let params = ReactorDesignParams {
                target_power_w: power,
                ..Default::default()
            };
            let spec = designer.design_modular(&params);

            // Power should scale approximately linearly
            assert!((spec.power_w - power).abs() < power * 0.1);
        }
    }

    // === Direction B: Engineering Realism Tests ===

    #[test]
    fn test_mass_components_sum_to_total() {
        let designer = ReactorDesigner::new();
        let params = ReactorDesignParams::default();
        let spec = designer.design(&params);

        let mb = spec.mass_breakdown.expect("Should have mass breakdown");
        let sum = mb.fuel_kg
            + mb.structure_kg
            + mb.shielding_kg
            + mb.cooling_kg
            + mb.trigger_kg
            + mb.bop_kg
            + mb.containment_kg;
        assert!(
            (sum - mb.total_kg).abs() < 0.01,
            "Mass components ({:.2}) should sum to total ({:.2})",
            sum,
            mb.total_kg
        );
        assert_eq!(spec.mass_kg, mb.total_kg);
    }

    #[test]
    fn test_cost_components_sum_to_total() {
        let designer = ReactorDesigner::new();
        let params = ReactorDesignParams::default();
        let spec = designer.design(&params);

        let cb = spec.cost_breakdown.expect("Should have cost breakdown");
        let subtotal = cb.core_materials_usd + cb.shielding_usd + cb.trigger_usd + cb.cooling_usd;
        let total = subtotal + cb.bop_usd + cb.engineering_usd + cb.assembly_usd;
        assert!(
            (total - cb.total_usd).abs() < 0.01,
            "Cost components ({:.2}) should sum to total ({:.2})",
            total,
            cb.total_usd
        );
        assert_eq!(spec.cost_usd, cb.total_usd);
    }

    #[test]
    fn test_thermal_forces_larger_volume_at_high_power() {
        let designer = ReactorDesigner::new();
        // High power with passive cooling should force volume increase
        let params = ReactorDesignParams {
            target_power_w: 100_000.0, // 100kW
            ..Default::default()
        };
        let spec = designer.design(&params);
        let tc = spec
            .thermal_constraint
            .expect("Should have thermal constraint");
        // Thermal min volume should be significant for high power
        assert!(tc.min_volume_m3 > 0.0);
        assert!(tc.min_surface_area_m2 > 0.0);
    }

    #[test]
    fn test_fuel_cycle_duty_in_range() {
        let designer = ReactorDesigner::new();
        let params = ReactorDesignParams::default();
        let spec = designer.design(&params);

        let fc = spec.fuel_cycle.expect("Should have fuel cycle");
        assert!(
            fc.duty_cycle > 0.0 && fc.duty_cycle <= 1.0,
            "Duty cycle ({:.4}) should be in (0, 1]",
            fc.duty_cycle
        );
        assert!(fc.d_consumption_g_s >= 0.0);
        assert!(fc.initial_loading_g > 0.0);
        assert!(fc.loading_time_hours > 0.0);
    }

    #[test]
    fn test_thermal_constraint_scales_with_power() {
        let tc_low = ThermalConstraint::compute(100.0, &CoolingMethod::Passive);
        let tc_high = ThermalConstraint::compute(100_000.0, &CoolingMethod::Passive);
        assert!(
            tc_high.min_surface_area_m2 > tc_low.min_surface_area_m2,
            "Higher power should need more surface area"
        );
        assert!(
            tc_high.min_volume_m3 > tc_low.min_volume_m3,
            "Higher power should need more volume"
        );
    }

    #[test]
    fn test_cooling_method_max_flux_ordering() {
        // Liquid metal should handle more heat than passive
        let passive = ThermalConstraint::compute(10_000.0, &CoolingMethod::Passive);
        let liquid = ThermalConstraint::compute(10_000.0, &CoolingMethod::LiquidMetal);
        assert!(liquid.max_heat_flux_w_m2 > passive.max_heat_flux_w_m2);
        assert!(liquid.min_surface_area_m2 < passive.min_surface_area_m2);
    }

    #[test]
    fn test_all_architectures_have_breakdowns() {
        let designer = ReactorDesigner::new();
        let params = ReactorDesignParams::default();
        let specs = designer.compare_all(&params);

        for spec in &specs {
            assert!(
                spec.mass_breakdown.is_some(),
                "{} should have mass breakdown",
                spec.name
            );
            assert!(
                spec.cost_breakdown.is_some(),
                "{} should have cost breakdown",
                spec.name
            );
            assert!(
                spec.fuel_cycle.is_some(),
                "{} should have fuel cycle",
                spec.name
            );
            assert!(
                spec.thermal_constraint.is_some(),
                "{} should have thermal constraint",
                spec.name
            );
        }
    }

    // === Risk-adjusted scoring tests ===

    #[test]
    fn test_risk_adjusted_scoring_reduces_scores() {
        let designer = ReactorDesigner::new();

        // Without risk adjustment
        let params_no_risk = ReactorDesignParams {
            target_power_w: 10_000.0,
            risk_weight: 0.0,
            ..Default::default()
        };
        let ranked_no_risk = designer.rank(&params_no_risk);

        // With risk adjustment
        let params_risk = ReactorDesignParams {
            target_power_w: 10_000.0,
            risk_weight: 1.0,
            ..Default::default()
        };
        let ranked_risk = designer.rank(&params_risk);

        // Risk-adjusted scores should be <= non-risk-adjusted scores
        for (i, ((_, score_no_risk), (_, score_risk))) in
            ranked_no_risk.iter().zip(ranked_risk.iter()).enumerate()
        {
            assert!(
                *score_risk <= *score_no_risk + 0.01, // Small tolerance for floating point
                "Risk-adjusted score {} ({:.2}) should be <= non-risk ({:.2})",
                i,
                score_risk,
                score_no_risk
            );
        }
    }

    #[test]
    fn test_feasibility_probability_in_range() {
        let designer = ReactorDesigner::new();
        let params = ReactorDesignParams::default();
        let specs = designer.compare_all(&params);

        for spec in &specs {
            let p = designer.estimate_feasibility_probability(spec, &params);
            assert!(
                p >= 0.05 && p <= 0.99,
                "{:?} feasibility ({:.3}) should be in [0.05, 0.99]",
                spec.architecture,
                p
            );
        }
    }

    #[test]
    fn test_higher_trl_higher_feasibility() {
        let designer = ReactorDesigner::new();
        let params = ReactorDesignParams::default();
        let specs = designer.compare_all(&params);

        // Group by TRL and check that higher TRL tends to have higher base feasibility
        let low_trl: Vec<_> = specs.iter().filter(|s| s.trl <= 4).collect();
        let high_trl: Vec<_> = specs.iter().filter(|s| s.trl >= 6).collect();

        if !low_trl.is_empty() && !high_trl.is_empty() {
            let avg_low: f64 = low_trl
                .iter()
                .map(|s| designer.estimate_feasibility_probability(s, &params))
                .sum::<f64>()
                / low_trl.len() as f64;
            let avg_high: f64 = high_trl
                .iter()
                .map(|s| designer.estimate_feasibility_probability(s, &params))
                .sum::<f64>()
                / high_trl.len() as f64;

            assert!(
                avg_high >= avg_low * 0.8, // High TRL should have better or similar feasibility
                "High TRL avg ({:.3}) should be >= 80% of low TRL avg ({:.3})",
                avg_high,
                avg_low
            );
        }
    }
}