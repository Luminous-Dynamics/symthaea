// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spark Engine Prototype Specification Generator
//!
//! Generates complete engineering specifications from coupled physics simulations:
//! - Bill of Materials (BOM)
//! - Dimensional drawings
//! - Manufacturing tolerances
//! - Safety analysis
//! - Cost estimates
//!
//! ## Output Format
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │         SPARK ENGINE PROTOTYPE SPECIFICATION            │
//! ├─────────────────────────────────────────────────────────┤
//! │  Model: SE-5K-DD-25                                     │
//! │  Power: 5 kW thermal                                    │
//! │  Fuel: Deuterium-Deuterium                              │
//! │  Lifetime: 25 years                                     │
//! ├─────────────────────────────────────────────────────────┤
//! │  DIMENSIONS                                             │
//! │  Core diameter: 60 mm                                   │
//! │  Shell thickness: 10 mm                                 │
//! │  Shielding: 200 mm                                      │
//! │  Total diameter: 540 mm                                 │
//! ├─────────────────────────────────────────────────────────┤
//! │  BILL OF MATERIALS                                      │
//! │  1. TiVZrNbPd HEA shell - 2.4 kg                        │
//! │  2. Zr/Nb nano-laminate - 0.8 kg                        │
//! │  3. Galinstan core - 1.2 kg                             │
//! │  4. LiH shielding - 45 kg                               │
//! └─────────────────────────────────────────────────────────┘
//! ```

use super::coupled_physics::{CoupledSimulationResult, OperatingConditions};
use super::radiation_damage::FusionReaction;

/// Model designation encoding
#[derive(Debug, Clone)]
pub struct ModelDesignation {
    /// Series (SE = Spark Engine)
    pub series: String,
    /// Power class (kW)
    pub power_class: String,
    /// Fuel type (DD, DT)
    pub fuel_type: String,
    /// Lifetime class (years)
    pub lifetime_class: String,
}

impl ModelDesignation {
    pub fn from_conditions(conditions: &OperatingConditions) -> Self {
        let power_class = if conditions.power_kw < 1000.0 {
            format!("{:.0}K", conditions.power_kw)
        } else {
            format!("{:.0}M", conditions.power_kw / 1000.0)
        };

        let fuel_type = match conditions.reaction {
            FusionReaction::DD => "DD",
            FusionReaction::DT => "DT",
            _ => "XX",
        }
        .to_string();

        let lifetime_class = format!("{:.0}", conditions.target_lifetime_years);

        Self {
            series: "SE".to_string(),
            power_class,
            fuel_type,
            lifetime_class,
        }
    }

    pub fn full_name(&self) -> String {
        format!(
            "{}-{}-{}-{}",
            self.series, self.power_class, self.fuel_type, self.lifetime_class
        )
    }
}

/// Dimensional specification with tolerances
#[derive(Debug, Clone)]
pub struct DimensionalSpec {
    /// Component name
    pub component: String,
    /// Nominal value (mm)
    pub nominal_mm: f64,
    /// Tolerance +/- (mm)
    pub tolerance_mm: f64,
    /// Surface finish (Ra, µm)
    pub surface_finish_um: f64,
    /// Notes
    pub notes: String,
}

/// Bill of materials entry
#[derive(Debug, Clone)]
pub struct BomEntry {
    /// Item number
    pub item: u32,
    /// Part name
    pub name: String,
    /// Material specification
    pub material_spec: String,
    /// Quantity
    pub quantity: f64,
    /// Unit
    pub unit: String,
    /// Mass (kg)
    pub mass_kg: f64,
    /// Unit cost (USD)
    pub unit_cost_usd: f64,
    /// Total cost (USD)
    pub total_cost_usd: f64,
    /// Lead time (weeks)
    pub lead_time_weeks: u32,
    /// Supplier notes
    pub supplier_notes: String,
}

/// Safety analysis entry
#[derive(Debug, Clone)]
pub struct SafetyEntry {
    /// Hazard category
    pub category: String,
    /// Hazard description
    pub hazard: String,
    /// Design mitigation
    pub mitigation: String,
    /// Residual risk level
    pub residual_risk: RiskLevel,
    /// Verification method
    pub verification: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Negligible,
    Low,
    Medium,
    High,
    Critical,
}

impl RiskLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            RiskLevel::Negligible => "Negligible",
            RiskLevel::Low => "Low",
            RiskLevel::Medium => "Medium",
            RiskLevel::High => "High",
            RiskLevel::Critical => "Critical",
        }
    }
}

/// Manufacturing process note
#[derive(Debug, Clone)]
pub struct ManufacturingNote {
    /// Process step
    pub step: u32,
    /// Process name
    pub process: String,
    /// Description
    pub description: String,
    /// Equipment required
    pub equipment: String,
    /// Critical parameters
    pub critical_params: Vec<String>,
}

/// Complete prototype specification
#[derive(Debug, Clone)]
pub struct PrototypeSpecification {
    /// Model designation
    pub model: ModelDesignation,
    /// Revision
    pub revision: String,
    /// Date generated
    pub date: String,

    /// Executive summary
    pub summary: String,

    /// Dimensional specifications
    pub dimensions: Vec<DimensionalSpec>,

    /// Bill of materials
    pub bom: Vec<BomEntry>,

    /// Cost summary
    pub total_material_cost_usd: f64,
    pub estimated_labor_cost_usd: f64,
    pub estimated_total_cost_usd: f64,

    /// Safety analysis
    pub safety: Vec<SafetyEntry>,

    /// Manufacturing notes
    pub manufacturing: Vec<ManufacturingNote>,

    /// Performance specifications
    pub thermal_power_kw: f64,
    pub electrical_output_kw: f64,
    pub efficiency_percent: f64,
    pub operating_temp_range_k: (f64, f64),
    pub design_lifetime_years: f64,
    pub predicted_lifetime_years: f64,

    /// Regulatory compliance notes
    pub regulatory_notes: Vec<String>,
}

impl PrototypeSpecification {
    /// Generate specification from coupled simulation result
    pub fn from_simulation(result: &CoupledSimulationResult) -> Self {
        let model = ModelDesignation::from_conditions(&result.conditions);

        // Generate dimensions
        let dimensions = Self::generate_dimensions(result);

        // Generate BOM
        let bom = Self::generate_bom(result);

        // Calculate costs
        let total_material: f64 = bom.iter().map(|e| e.total_cost_usd).sum();
        let labor_estimate = total_material * 2.5; // Assembly labor ~2.5x material
        let total_estimate = total_material + labor_estimate;

        // Generate safety analysis
        let safety = Self::generate_safety_analysis(result);

        // Generate manufacturing notes
        let manufacturing = Self::generate_manufacturing_notes(result);

        // Performance specs
        let thermal_power = result.conditions.power_kw;
        let electrical = thermal_power * 0.35; // ~35% conversion efficiency
        let efficiency = 35.0;

        let summary = format!(
            "Spark Engine {} - {:.1} kW thermal fusion power unit using {:?} fuel. \
             Designed for {:.0} year operational lifetime with self-healing HEA shell \
             and liquid metal core. {}",
            model.full_name(),
            thermal_power,
            result.conditions.reaction,
            result.conditions.target_lifetime_years,
            if result.feasible {
                "DESIGN VALIDATED"
            } else {
                "DESIGN REQUIRES REVISION"
            }
        );

        Self {
            model,
            revision: "A".to_string(),
            date: "2025-01-01".to_string(),
            summary,
            dimensions,
            bom,
            total_material_cost_usd: total_material,
            estimated_labor_cost_usd: labor_estimate,
            estimated_total_cost_usd: total_estimate,
            safety,
            manufacturing,
            thermal_power_kw: thermal_power,
            electrical_output_kw: electrical,
            efficiency_percent: efficiency,
            operating_temp_range_k: (
                result.conditions.ambient_temp_k,
                result.thermal_profile.t_max,
            ),
            design_lifetime_years: result.conditions.target_lifetime_years,
            predicted_lifetime_years: result.pulse_thermal.lifetime_years.min(100.0),
            regulatory_notes: Self::generate_regulatory_notes(result),
        }
    }

    fn generate_dimensions(result: &CoupledSimulationResult) -> Vec<DimensionalSpec> {
        let geo = &result.geometry_shielding.geometry;

        vec![
            DimensionalSpec {
                component: "Core cavity diameter".to_string(),
                nominal_mm: geo.core_radius * 2000.0,
                tolerance_mm: 0.5,
                surface_finish_um: 1.6,
                notes: "Must maintain sphericity for uniform loading".to_string(),
            },
            DimensionalSpec {
                component: "Interface layer thickness".to_string(),
                nominal_mm: geo.interface_thickness * 1000.0,
                tolerance_mm: 0.1,
                surface_finish_um: 0.8,
                notes: "Nano-laminate deposition thickness".to_string(),
            },
            DimensionalSpec {
                component: "Shell wall thickness".to_string(),
                nominal_mm: geo.shell_thickness * 1000.0,
                tolerance_mm: 0.2,
                surface_finish_um: 3.2,
                notes: "HEA shell, uniform thickness critical".to_string(),
            },
            DimensionalSpec {
                component: "Shielding thickness".to_string(),
                nominal_mm: result.geometry_shielding.shielding.thickness_m * 1000.0,
                tolerance_mm: 5.0,
                surface_finish_um: 12.5,
                notes: "Neutron shielding layer".to_string(),
            },
            DimensionalSpec {
                component: "Total outer diameter".to_string(),
                nominal_mm: (result.geometry_shielding.geometry.outer_radius
                    + result.geometry_shielding.shielding_thickness_m)
                    * 2000.0,
                tolerance_mm: 10.0,
                surface_finish_um: 6.3,
                notes: "External containment".to_string(),
            },
        ]
    }

    fn generate_bom(result: &CoupledSimulationResult) -> Vec<BomEntry> {
        let geo = &result.geometry_shielding.geometry;

        // Calculate volumes and masses
        let core_volume = 4.0 / 3.0 * std::f64::consts::PI * geo.core_radius.powi(3);
        let core_mass = core_volume * 6440.0; // Galinstan density

        let interface_r_inner = geo.core_radius;
        let interface_r_outer = geo.core_radius + geo.interface_thickness;
        let interface_volume = 4.0 / 3.0
            * std::f64::consts::PI
            * (interface_r_outer.powi(3) - interface_r_inner.powi(3));
        let interface_mass = interface_volume * 7500.0; // Zr/Nb density

        let shell_volume = geo.shell_volume();
        let shell_mass = shell_volume * 7200.0; // HEA density

        let shield_r_inner = geo.outer_radius;
        let shield_r_outer = geo.outer_radius + result.geometry_shielding.shielding_thickness_m;
        let shield_volume =
            4.0 / 3.0 * std::f64::consts::PI * (shield_r_outer.powi(3) - shield_r_inner.powi(3));
        let shield_mass = shield_volume * 820.0; // LiH density

        // Cost estimates (USD/kg)
        let hea_cost_per_kg = 500.0; // Custom alloy
        let laminate_cost_per_kg = 800.0; // PVD deposited
        let galinstan_cost_per_kg = 150.0; // Commercial
        let lih_cost_per_kg = 50.0; // Industrial grade

        vec![
            BomEntry {
                item: 1,
                name: "Shell - HEA hemisphere (x2)".to_string(),
                material_spec: format!("{} per ASTM B771", result.shell_material),
                quantity: 2.0,
                unit: "each".to_string(),
                mass_kg: shell_mass,
                unit_cost_usd: shell_mass * hea_cost_per_kg / 2.0,
                total_cost_usd: shell_mass * hea_cost_per_kg,
                lead_time_weeks: 12,
                supplier_notes: "Arc melted, HIP consolidated".to_string(),
            },
            BomEntry {
                item: 2,
                name: "Interface coating".to_string(),
                material_spec: format!("{}, 100-layer", result.interface_material),
                quantity: 1.0,
                unit: "coating".to_string(),
                mass_kg: interface_mass,
                unit_cost_usd: interface_mass * laminate_cost_per_kg,
                total_cost_usd: interface_mass * laminate_cost_per_kg,
                lead_time_weeks: 8,
                supplier_notes: "Magnetron sputtering, 5nm layers".to_string(),
            },
            BomEntry {
                item: 3,
                name: "Core fluid".to_string(),
                material_spec: format!("{}, >99.99% purity", result.core_material),
                quantity: core_mass,
                unit: "kg".to_string(),
                mass_kg: core_mass,
                unit_cost_usd: galinstan_cost_per_kg,
                total_cost_usd: core_mass * galinstan_cost_per_kg,
                lead_time_weeks: 2,
                supplier_notes: "Commercial grade, degassed".to_string(),
            },
            BomEntry {
                item: 4,
                name: "Neutron shielding".to_string(),
                material_spec: result.geometry_shielding.shielding.primary_material.clone(),
                quantity: shield_mass,
                unit: "kg".to_string(),
                mass_kg: shield_mass,
                unit_cost_usd: lih_cost_per_kg,
                total_cost_usd: shield_mass * lih_cost_per_kg,
                lead_time_weeks: 4,
                supplier_notes: "Sintered blocks, moisture sealed".to_string(),
            },
            BomEntry {
                item: 5,
                name: "Deuterium fuel charge".to_string(),
                material_spec: "D2 gas, 99.8% isotopic purity".to_string(),
                quantity: 0.1,
                unit: "kg".to_string(),
                mass_kg: 0.1,
                unit_cost_usd: 5000.0,
                total_cost_usd: 500.0,
                lead_time_weeks: 6,
                supplier_notes: "DOE licensed supplier".to_string(),
            },
            BomEntry {
                item: 6,
                name: "Instrumentation package".to_string(),
                material_spec: "Temperature, pressure, radiation sensors".to_string(),
                quantity: 1.0,
                unit: "set".to_string(),
                mass_kg: 2.0,
                unit_cost_usd: 5000.0,
                total_cost_usd: 5000.0,
                lead_time_weeks: 4,
                supplier_notes: "Radiation-hardened electronics".to_string(),
            },
            BomEntry {
                item: 7,
                name: "Control system".to_string(),
                material_spec: "Pulse timing, safety interlocks".to_string(),
                quantity: 1.0,
                unit: "system".to_string(),
                mass_kg: 5.0,
                unit_cost_usd: 10000.0,
                total_cost_usd: 10000.0,
                lead_time_weeks: 8,
                supplier_notes: "SIL-3 certified".to_string(),
            },
        ]
    }

    fn generate_safety_analysis(result: &CoupledSimulationResult) -> Vec<SafetyEntry> {
        let mut safety = vec![
            SafetyEntry {
                category: "Radiation".to_string(),
                hazard: "Neutron emission during operation".to_string(),
                mitigation: format!(
                    "{} shielding, {:.1} cm thickness",
                    result.geometry_shielding.shielding.primary_material,
                    result.geometry_shielding.shielding.thickness_m * 100.0
                ),
                residual_risk: if result.geometry_shielding.shielding.public_safe {
                    RiskLevel::Low
                } else {
                    RiskLevel::High
                },
                verification: "Dosimetry survey at commissioning".to_string(),
            },
            SafetyEntry {
                category: "Thermal".to_string(),
                hazard: "High temperature surfaces".to_string(),
                mitigation: "Insulated containment, warning labels".to_string(),
                residual_risk: RiskLevel::Low,
                verification: "Thermal imaging under operation".to_string(),
            },
            SafetyEntry {
                category: "Chemical".to_string(),
                hazard: "Liquid metal core (Ga, In, Sn)".to_string(),
                mitigation: "Sealed containment, leak detection".to_string(),
                residual_risk: RiskLevel::Low,
                verification: "Helium leak test < 1e-9 mbar·L/s".to_string(),
            },
            SafetyEntry {
                category: "Pressure".to_string(),
                hazard: "Deuterium gas under pressure".to_string(),
                mitigation: "Pressure relief, burst disc".to_string(),
                residual_risk: RiskLevel::Low,
                verification: "Hydrostatic test to 2x working pressure".to_string(),
            },
            SafetyEntry {
                category: "Electrical".to_string(),
                hazard: "High voltage pulse system".to_string(),
                mitigation: "Interlocks, grounding, enclosure".to_string(),
                residual_risk: RiskLevel::Low,
                verification: "Dielectric testing per IEC 61010".to_string(),
            },
        ];

        // Add specific risks based on conditions
        if result.conditions.reaction == FusionReaction::DT {
            safety.push(SafetyEntry {
                category: "Radiation".to_string(),
                hazard: "Tritium handling and containment".to_string(),
                mitigation: "Double containment, tritium monitors".to_string(),
                residual_risk: RiskLevel::Medium,
                verification: "DOE tritium handling certification".to_string(),
            });
        }

        safety
    }

    fn generate_manufacturing_notes(result: &CoupledSimulationResult) -> Vec<ManufacturingNote> {
        vec![
            ManufacturingNote {
                step: 1,
                process: "HEA Shell Fabrication".to_string(),
                description: "Arc melt constituent elements, cast into ingot, \
                             HIP consolidate, machine hemispheres"
                    .to_string(),
                equipment: "Vacuum arc furnace, HIP unit, 5-axis CNC".to_string(),
                critical_params: vec![
                    "Composition within ±0.5 at%".to_string(),
                    "Porosity < 0.1%".to_string(),
                    "Surface finish Ra < 3.2 µm".to_string(),
                ],
            },
            ManufacturingNote {
                step: 2,
                process: "Interface Coating".to_string(),
                description: "Magnetron sputter deposit Zr/Nb alternating layers, \
                             100 layer pairs at 5nm each"
                    .to_string(),
                equipment: "PVD chamber with dual targets".to_string(),
                critical_params: vec![
                    "Layer thickness 5 ± 0.5 nm".to_string(),
                    "Interface sharpness < 1 nm".to_string(),
                    "Coating adhesion > 50 MPa".to_string(),
                ],
            },
            ManufacturingNote {
                step: 3,
                process: "Shell Assembly".to_string(),
                description: "Electron beam weld hemispheres with fill port, \
                             leak test, radiograph weld"
                    .to_string(),
                equipment: "EB welder, He leak detector, X-ray".to_string(),
                critical_params: vec![
                    "Weld penetration 100%".to_string(),
                    "Leak rate < 1e-9 mbar·L/s".to_string(),
                    "No weld defects > 0.5 mm".to_string(),
                ],
            },
            ManufacturingNote {
                step: 4,
                process: "Core Fill".to_string(),
                description: "Evacuate shell, backfill with degassed Galinstan, \
                             seal fill port"
                    .to_string(),
                equipment: "Vacuum system, liquid metal handling".to_string(),
                critical_params: vec![
                    "Fill level 95-98%".to_string(),
                    "Oxygen content < 10 ppm".to_string(),
                    "No trapped gas bubbles".to_string(),
                ],
            },
            ManufacturingNote {
                step: 5,
                process: "Shielding Assembly".to_string(),
                description: format!(
                    "Stack {} blocks around core, seal in containment",
                    result.geometry_shielding.shielding.primary_material
                ),
                equipment: "Dry room (if LiH), lifting fixtures".to_string(),
                critical_params: vec![
                    "No gaps > 2 mm".to_string(),
                    "Moisture content < 0.1%".to_string(),
                    "Uniform density".to_string(),
                ],
            },
            ManufacturingNote {
                step: 6,
                process: "Final Integration".to_string(),
                description: "Install instrumentation, connect control system, \
                             commission and calibrate"
                    .to_string(),
                equipment: "Integration fixtures, test equipment".to_string(),
                critical_params: vec![
                    "All sensors responding".to_string(),
                    "Safety interlocks functional".to_string(),
                    "Pulse timing within spec".to_string(),
                ],
            },
        ]
    }

    fn generate_regulatory_notes(result: &CoupledSimulationResult) -> Vec<String> {
        let mut notes = vec![
            "NRC 10 CFR 30 - Byproduct material license required".to_string(),
            "DOE requirements for fusion devices may apply".to_string(),
            "OSHA radiation protection per 29 CFR 1910.1096".to_string(),
        ];

        if result.conditions.reaction == FusionReaction::DT {
            notes.push("DOE tritium handling license required".to_string());
            notes.push("EPA tritium release limits apply".to_string());
        }

        if result.conditions.power_kw > 1000.0 {
            notes.push("May require environmental impact assessment".to_string());
        }

        notes.push("Export controls (ITAR/EAR) may apply to technology".to_string());

        notes
    }

    /// Generate formatted specification document
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&"═".repeat(70));
        output.push('\n');
        output.push_str("  SPARK ENGINE PROTOTYPE SPECIFICATION\n");
        output.push_str(&format!(
            "  Model: {}  Rev: {}  Date: {}\n",
            self.model.full_name(),
            self.revision,
            self.date
        ));
        output.push_str(&"═".repeat(70));
        output.push_str("\n\n");

        // Summary
        output.push_str("EXECUTIVE SUMMARY\n");
        output.push_str(&"─".repeat(70));
        output.push('\n');
        output.push_str(&self.summary);
        output.push_str("\n\n");

        // Performance
        output.push_str("PERFORMANCE SPECIFICATIONS\n");
        output.push_str(&"─".repeat(70));
        output.push('\n');
        output.push_str(&format!(
            "  Thermal Power:     {:.1} kW\n",
            self.thermal_power_kw
        ));
        output.push_str(&format!(
            "  Electrical Output: {:.1} kW (est.)\n",
            self.electrical_output_kw
        ));
        output.push_str(&format!(
            "  Conversion Eff:    {:.0}%\n",
            self.efficiency_percent
        ));
        output.push_str(&format!(
            "  Operating Temp:    {:.0} - {:.0} K\n",
            self.operating_temp_range_k.0, self.operating_temp_range_k.1
        ));
        output.push_str(&format!(
            "  Design Lifetime:   {:.0} years\n",
            self.design_lifetime_years
        ));
        output.push_str(&format!(
            "  Predicted Life:    {:.1} years\n",
            self.predicted_lifetime_years
        ));
        output.push('\n');

        // Dimensions
        output.push_str("DIMENSIONAL SPECIFICATIONS\n");
        output.push_str(&"─".repeat(70));
        output.push('\n');
        output.push_str(&format!(
            "{:30} {:>12} {:>10} {:>12}\n",
            "Component", "Nominal", "Tol. ±", "Finish"
        ));
        for dim in &self.dimensions {
            output.push_str(&format!(
                "{:30} {:>10.1} mm {:>8.1} mm {:>10.1} µm\n",
                dim.component, dim.nominal_mm, dim.tolerance_mm, dim.surface_finish_um
            ));
        }
        output.push('\n');

        // BOM
        output.push_str("BILL OF MATERIALS\n");
        output.push_str(&"─".repeat(70));
        output.push('\n');
        output.push_str(&format!(
            "{:4} {:30} {:>10} {:>12}\n",
            "Item", "Description", "Mass (kg)", "Cost (USD)"
        ));
        for entry in &self.bom {
            output.push_str(&format!(
                "{:4} {:30} {:>10.2} {:>12.0}\n",
                entry.item, entry.name, entry.mass_kg, entry.total_cost_usd
            ));
        }
        output.push_str(&format!(
            "{:34} {:>10} {:>12}\n",
            "",
            "─".repeat(10),
            "─".repeat(12)
        ));
        let total_mass: f64 = self.bom.iter().map(|e| e.mass_kg).sum();
        output.push_str(&format!(
            "{:34} {:>10.1} {:>12.0}\n",
            "TOTAL", total_mass, self.total_material_cost_usd
        ));
        output.push_str(&format!(
            "{:34} {:>10} {:>12.0}\n",
            "Est. Labor", "", self.estimated_labor_cost_usd
        ));
        output.push_str(&format!(
            "{:34} {:>10} {:>12.0}\n",
            "EST. TOTAL", "", self.estimated_total_cost_usd
        ));
        output.push('\n');

        // Safety
        output.push_str("SAFETY ANALYSIS\n");
        output.push_str(&"─".repeat(70));
        output.push('\n');
        for entry in &self.safety {
            output.push_str(&format!(
                "[{}] {}: {}\n",
                entry.residual_risk.as_str(),
                entry.category,
                entry.hazard
            ));
            output.push_str(&format!("    Mitigation: {}\n", entry.mitigation));
        }
        output.push('\n');

        // Manufacturing
        output.push_str("MANUFACTURING PROCESS\n");
        output.push_str(&"─".repeat(70));
        output.push('\n');
        for note in &self.manufacturing {
            output.push_str(&format!("Step {}: {}\n", note.step, note.process));
            output.push_str(&format!("  {}\n", note.description));
        }
        output.push('\n');

        // Regulatory
        output.push_str("REGULATORY COMPLIANCE\n");
        output.push_str(&"─".repeat(70));
        output.push('\n');
        for note in &self.regulatory_notes {
            output.push_str(&format!("• {note}\n"));
        }

        output.push('\n');
        output.push_str(&"═".repeat(70));
        output.push_str("\n  END OF SPECIFICATION\n");
        output.push_str(&"═".repeat(70));
        output.push('\n');

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genesis::GenesisSeed;
    use crate::physics::CoupledPhysicsEngine;

    fn setup_result() -> CoupledSimulationResult {
        let genesis = GenesisSeed::from_phrase("prototype spec test");
        let engine = CoupledPhysicsEngine::from_genesis(&genesis);
        engine.simulate(&OperatingConditions::consumer())
    }

    #[test]
    fn test_model_designation() {
        let conditions = OperatingConditions::consumer();
        let model = ModelDesignation::from_conditions(&conditions);

        assert_eq!(model.series, "SE");
        assert_eq!(model.fuel_type, "DD");
        println!("Model: {}", model.full_name());
    }

    #[test]
    fn test_specification_generation() {
        let result = setup_result();
        let spec = PrototypeSpecification::from_simulation(&result);

        assert!(!spec.dimensions.is_empty());
        assert!(!spec.bom.is_empty());
        assert!(!spec.safety.is_empty());

        println!("Total cost: ${:.0}", spec.estimated_total_cost_usd);
    }

    #[test]
    fn test_specification_output() {
        let result = setup_result();
        let spec = PrototypeSpecification::from_simulation(&result);

        let output = spec.to_string();
        assert!(output.contains("SPARK ENGINE"));
        assert!(output.contains("BILL OF MATERIALS"));

        // Print first 2000 chars for inspection
        println!("{}", &output[..output.len().min(2000)]);
    }
}
