// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Manufacturing-Aware Design: From Physics to Fabrication
//!
//! Bridges the gap between "physics works" and "can actually build this".
//!
//! ## Design for Manufacturing (DFM) Considerations
//!
//! 1. **Tolerance Analysis**: GD&T stackup, form/position errors
//! 2. **Joining Methods**: Welding, brazing, mechanical fastening
//! 3. **Material Compatibility**: Galvanic corrosion, diffusion bonding
//! 4. **Inspection Requirements**: NDE methods, acceptance criteria
//! 5. **Assembly Sequence**: Order of operations, accessibility
//!
//! ## Key Challenges for Spark Engine
//!
//! - HEA shell fabrication (powder metallurgy vs casting vs AM)
//! - Galinstan containment (no aluminum, copper, or zinc contact)
//! - Neutron shielding joints (no streaming paths)
//! - Thermal expansion mismatch at interfaces

use std::collections::HashMap;

/// Manufacturing process capability
#[derive(Debug, Clone)]
pub struct ProcessCapability {
    /// Process name
    pub name: String,
    /// Achievable tolerance (mm)
    pub tolerance_mm: f64,
    /// Surface finish (Ra, µm)
    pub surface_finish_um: f64,
    /// Relative cost (1.0 = baseline)
    pub relative_cost: f64,
    /// Lead time (weeks)
    pub lead_time_weeks: f64,
    /// Compatible materials
    pub compatible_materials: Vec<String>,
}

impl ProcessCapability {
    /// CNC machining
    pub fn cnc_machining() -> Self {
        Self {
            name: "CNC Machining".to_string(),
            tolerance_mm: 0.025,
            surface_finish_um: 1.6,
            relative_cost: 1.0,
            lead_time_weeks: 2.0,
            compatible_materials: vec![
                "Steel".to_string(),
                "Aluminum".to_string(),
                "Titanium".to_string(),
                "Inconel".to_string(),
            ],
        }
    }

    /// Powder metallurgy (for HEAs)
    pub fn powder_metallurgy() -> Self {
        Self {
            name: "Powder Metallurgy (HIP)".to_string(),
            tolerance_mm: 0.5,
            surface_finish_um: 6.3,
            relative_cost: 3.0,
            lead_time_weeks: 8.0,
            compatible_materials: vec![
                "HEA".to_string(),
                "Tungsten".to_string(),
                "Molybdenum".to_string(),
            ],
        }
    }

    /// Additive manufacturing (DMLS/SLM)
    pub fn additive_manufacturing() -> Self {
        Self {
            name: "Metal AM (DMLS)".to_string(),
            tolerance_mm: 0.1,
            surface_finish_um: 10.0,
            relative_cost: 5.0,
            lead_time_weeks: 3.0,
            compatible_materials: vec![
                "Titanium".to_string(),
                "Inconel".to_string(),
                "Stainless".to_string(),
                "HEA".to_string(),
            ],
        }
    }

    /// Investment casting
    pub fn investment_casting() -> Self {
        Self {
            name: "Investment Casting".to_string(),
            tolerance_mm: 0.25,
            surface_finish_um: 3.2,
            relative_cost: 2.0,
            lead_time_weeks: 6.0,
            compatible_materials: vec![
                "Steel".to_string(),
                "Inconel".to_string(),
                "Bronze".to_string(),
            ],
        }
    }

    /// Sheet metal forming
    pub fn sheet_forming() -> Self {
        Self {
            name: "Sheet Metal Forming".to_string(),
            tolerance_mm: 0.5,
            surface_finish_um: 1.6,
            relative_cost: 0.5,
            lead_time_weeks: 2.0,
            compatible_materials: vec![
                "Steel".to_string(),
                "Aluminum".to_string(),
                "Stainless".to_string(),
            ],
        }
    }
}

/// Joining method specification
#[derive(Debug, Clone)]
pub struct JoiningMethod {
    /// Method name
    pub name: String,
    /// Joint efficiency (strength ratio vs base metal)
    pub efficiency: f64,
    /// Temperature limit (K)
    pub max_temp_k: f64,
    /// Hermetic seal capability
    pub hermetic: bool,
    /// Radiation resistance (relative)
    pub radiation_resistance: f64,
    /// Cost per joint (relative)
    pub relative_cost: f64,
    /// Compatible material pairs
    pub compatible_pairs: Vec<(String, String)>,
}

impl JoiningMethod {
    /// Electron beam welding (for HEAs)
    pub fn electron_beam_welding() -> Self {
        Self {
            name: "Electron Beam Welding".to_string(),
            efficiency: 0.95,
            max_temp_k: 1500.0,
            hermetic: true,
            radiation_resistance: 0.9,
            relative_cost: 3.0,
            compatible_pairs: vec![
                ("HEA".to_string(), "HEA".to_string()),
                ("Titanium".to_string(), "Titanium".to_string()),
                ("Steel".to_string(), "Steel".to_string()),
            ],
        }
    }

    /// Diffusion bonding
    pub fn diffusion_bonding() -> Self {
        Self {
            name: "Diffusion Bonding".to_string(),
            efficiency: 0.90,
            max_temp_k: 1200.0,
            hermetic: true,
            radiation_resistance: 1.0,
            relative_cost: 5.0,
            compatible_pairs: vec![
                ("HEA".to_string(), "Zr/Nb".to_string()),
                ("Titanium".to_string(), "Steel".to_string()),
            ],
        }
    }

    /// Brazing
    pub fn brazing() -> Self {
        Self {
            name: "Vacuum Brazing".to_string(),
            efficiency: 0.70,
            max_temp_k: 800.0,
            hermetic: true,
            radiation_resistance: 0.6,
            relative_cost: 2.0,
            compatible_pairs: vec![
                ("Steel".to_string(), "Steel".to_string()),
                ("Steel".to_string(), "Copper".to_string()),
            ],
        }
    }

    /// Mechanical seal (O-ring/gasket)
    pub fn mechanical_seal() -> Self {
        Self {
            name: "Mechanical Seal".to_string(),
            efficiency: 1.0,   // No strength reduction
            max_temp_k: 500.0, // Limited by elastomers
            hermetic: true,
            radiation_resistance: 0.3, // Elastomers degrade
            relative_cost: 0.5,
            compatible_pairs: vec![("Any".to_string(), "Any".to_string())],
        }
    }

    /// Bolted flange
    pub fn bolted_flange() -> Self {
        Self {
            name: "Bolted Flange".to_string(),
            efficiency: 0.8,
            max_temp_k: 800.0,
            hermetic: false, // Requires gasket
            radiation_resistance: 0.9,
            relative_cost: 1.0,
            compatible_pairs: vec![("Any".to_string(), "Any".to_string())],
        }
    }
}

/// Material compatibility assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compatibility {
    /// Fully compatible
    Compatible,
    /// Compatible with precautions
    Conditional,
    /// Not compatible
    Incompatible,
}

/// Material compatibility matrix
#[derive(Debug, Clone)]
pub struct MaterialCompatibility {
    /// Compatibility map: (material1, material2) -> compatibility
    pub matrix: HashMap<(String, String), Compatibility>,
    /// Notes for conditional compatibility
    pub notes: HashMap<(String, String), String>,
}

impl MaterialCompatibility {
    /// Create default compatibility matrix for Spark Engine materials
    pub fn spark_engine_default() -> Self {
        let mut matrix = HashMap::new();
        let mut notes = HashMap::new();

        // Galinstan compatibility (critical - attacks Al, Cu, Zn)
        matrix.insert(
            ("Galinstan".to_string(), "Aluminum".to_string()),
            Compatibility::Incompatible,
        );
        matrix.insert(
            ("Galinstan".to_string(), "Copper".to_string()),
            Compatibility::Incompatible,
        );
        matrix.insert(
            ("Galinstan".to_string(), "Zinc".to_string()),
            Compatibility::Incompatible,
        );
        matrix.insert(
            ("Galinstan".to_string(), "Steel".to_string()),
            Compatibility::Compatible,
        );
        matrix.insert(
            ("Galinstan".to_string(), "Stainless".to_string()),
            Compatibility::Compatible,
        );
        matrix.insert(
            ("Galinstan".to_string(), "Titanium".to_string()),
            Compatibility::Compatible,
        );
        matrix.insert(
            ("Galinstan".to_string(), "Tungsten".to_string()),
            Compatibility::Compatible,
        );
        matrix.insert(
            ("Galinstan".to_string(), "HEA".to_string()),
            Compatibility::Conditional,
        );
        notes.insert(
            ("Galinstan".to_string(), "HEA".to_string()),
            "Verify HEA composition excludes Al/Cu/Zn".to_string(),
        );

        // HEA compatibility
        matrix.insert(
            ("HEA".to_string(), "Zr/Nb".to_string()),
            Compatibility::Compatible,
        );
        matrix.insert(
            ("HEA".to_string(), "Steel".to_string()),
            Compatibility::Conditional,
        );
        notes.insert(
            ("HEA".to_string(), "Steel".to_string()),
            "Use diffusion barrier at high temperatures".to_string(),
        );

        // Zr/Nb nano-laminate
        matrix.insert(
            ("Zr/Nb".to_string(), "Galinstan".to_string()),
            Compatibility::Compatible,
        );
        matrix.insert(
            ("Zr/Nb".to_string(), "Steel".to_string()),
            Compatibility::Compatible,
        );

        // Shielding materials
        matrix.insert(
            ("Borated PE".to_string(), "Steel".to_string()),
            Compatibility::Compatible,
        );
        matrix.insert(
            ("Borated PE".to_string(), "Concrete".to_string()),
            Compatibility::Compatible,
        );
        matrix.insert(
            ("Lead".to_string(), "Steel".to_string()),
            Compatibility::Compatible,
        );

        Self { matrix, notes }
    }

    /// Check compatibility between two materials
    pub fn check(&self, mat1: &str, mat2: &str) -> (Compatibility, Option<&str>) {
        // Check both orderings
        let key1 = (mat1.to_string(), mat2.to_string());
        let key2 = (mat2.to_string(), mat1.to_string());

        let compat = self
            .matrix
            .get(&key1)
            .or_else(|| self.matrix.get(&key2))
            .copied()
            .unwrap_or(Compatibility::Compatible); // Default to compatible if not specified

        let note = self
            .notes
            .get(&key1)
            .or_else(|| self.notes.get(&key2))
            .map(|s| s.as_str());

        (compat, note)
    }
}

/// Tolerance stackup analysis
#[derive(Debug, Clone)]
pub struct ToleranceStackup {
    /// Feature name
    pub feature: String,
    /// Nominal dimension (mm)
    pub nominal_mm: f64,
    /// Contributing tolerances
    pub contributors: Vec<ToleranceContributor>,
    /// Total tolerance (worst case)
    pub worst_case_mm: f64,
    /// Total tolerance (RSS)
    pub rss_mm: f64,
    /// Is within specification?
    pub within_spec: bool,
    /// Specification limit (mm)
    pub spec_limit_mm: f64,
}

/// Individual tolerance contributor
#[derive(Debug, Clone)]
pub struct ToleranceContributor {
    /// Component name
    pub component: String,
    /// Tolerance (±mm)
    pub tolerance_mm: f64,
    /// Sensitivity (how much this affects the stackup)
    pub sensitivity: f64,
}

impl ToleranceStackup {
    /// Calculate stackup from contributors
    pub fn calculate(
        feature: &str,
        nominal_mm: f64,
        contributors: Vec<ToleranceContributor>,
        spec_limit_mm: f64,
    ) -> Self {
        // Worst case: sum of all tolerances
        let worst_case: f64 = contributors
            .iter()
            .map(|c| c.tolerance_mm * c.sensitivity.abs())
            .sum();

        // RSS: root sum of squares
        let rss: f64 = contributors
            .iter()
            .map(|c| (c.tolerance_mm * c.sensitivity).powi(2))
            .sum::<f64>()
            .sqrt();

        let within_spec = rss <= spec_limit_mm;

        Self {
            feature: feature.to_string(),
            nominal_mm,
            contributors,
            worst_case_mm: worst_case,
            rss_mm: rss,
            within_spec,
            spec_limit_mm,
        }
    }
}

/// NDE (Non-Destructive Examination) method
#[derive(Debug, Clone)]
pub struct NdeMethod {
    /// Method name
    pub name: String,
    /// Detectable flaw size (mm)
    pub min_flaw_size_mm: f64,
    /// Applicable to
    pub applicable_to: Vec<String>,
    /// Cost per inspection (relative)
    pub relative_cost: f64,
    /// Inspection time (hours)
    pub time_hours: f64,
}

impl NdeMethod {
    /// Radiographic testing (X-ray/gamma)
    pub fn radiography() -> Self {
        Self {
            name: "Radiographic Testing (RT)".to_string(),
            min_flaw_size_mm: 0.5,
            applicable_to: vec![
                "Welds".to_string(),
                "Castings".to_string(),
                "Thick sections".to_string(),
            ],
            relative_cost: 3.0,
            time_hours: 4.0,
        }
    }

    /// Ultrasonic testing
    pub fn ultrasonic() -> Self {
        Self {
            name: "Ultrasonic Testing (UT)".to_string(),
            min_flaw_size_mm: 0.25,
            applicable_to: vec![
                "Welds".to_string(),
                "Forgings".to_string(),
                "Bonds".to_string(),
            ],
            relative_cost: 2.0,
            time_hours: 2.0,
        }
    }

    /// Dye penetrant
    pub fn dye_penetrant() -> Self {
        Self {
            name: "Dye Penetrant (PT)".to_string(),
            min_flaw_size_mm: 0.1,
            applicable_to: vec!["Surface cracks".to_string(), "Porosity".to_string()],
            relative_cost: 0.5,
            time_hours: 1.0,
        }
    }

    /// Helium leak testing
    pub fn helium_leak() -> Self {
        Self {
            name: "Helium Leak Testing".to_string(),
            min_flaw_size_mm: 0.001, // Can detect molecular leaks
            applicable_to: vec!["Hermetic seals".to_string(), "Pressure vessels".to_string()],
            relative_cost: 2.5,
            time_hours: 3.0,
        }
    }
}

/// Assembly step
#[derive(Debug, Clone)]
pub struct AssemblyStep {
    /// Step number
    pub step: usize,
    /// Description
    pub description: String,
    /// Components involved
    pub components: Vec<String>,
    /// Joining method (if applicable)
    pub joining: Option<String>,
    /// NDE required
    pub nde_required: Vec<String>,
    /// Critical dimension to verify
    pub critical_dims: Vec<String>,
    /// Estimated time (hours)
    pub time_hours: f64,
}

/// Complete manufacturing assessment
#[derive(Debug, Clone)]
pub struct ManufacturingAssessment {
    /// Design name
    pub design_name: String,
    /// Recommended processes by component
    pub processes: HashMap<String, ProcessCapability>,
    /// Joining methods
    pub joints: Vec<(String, String, JoiningMethod)>,
    /// Material compatibility issues
    pub compatibility_issues: Vec<String>,
    /// Tolerance stackups
    pub stackups: Vec<ToleranceStackup>,
    /// Required NDE
    pub nde_plan: Vec<(String, NdeMethod)>,
    /// Assembly sequence
    pub assembly_sequence: Vec<AssemblyStep>,
    /// Total manufacturing cost (relative)
    pub total_cost: f64,
    /// Total lead time (weeks)
    pub lead_time_weeks: f64,
    /// Manufacturing risk score (0-1, lower is better)
    pub risk_score: f64,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Manufacturing assessment engine
pub struct ManufacturingEngine {
    pub processes: Vec<ProcessCapability>,
    pub joining_methods: Vec<JoiningMethod>,
    pub compatibility: MaterialCompatibility,
    pub nde_methods: Vec<NdeMethod>,
}

impl ManufacturingEngine {
    /// Create with default capabilities
    pub fn new() -> Self {
        Self {
            processes: vec![
                ProcessCapability::cnc_machining(),
                ProcessCapability::powder_metallurgy(),
                ProcessCapability::additive_manufacturing(),
                ProcessCapability::investment_casting(),
                ProcessCapability::sheet_forming(),
            ],
            joining_methods: vec![
                JoiningMethod::electron_beam_welding(),
                JoiningMethod::diffusion_bonding(),
                JoiningMethod::brazing(),
                JoiningMethod::mechanical_seal(),
                JoiningMethod::bolted_flange(),
            ],
            compatibility: MaterialCompatibility::spark_engine_default(),
            nde_methods: vec![
                NdeMethod::radiography(),
                NdeMethod::ultrasonic(),
                NdeMethod::dye_penetrant(),
                NdeMethod::helium_leak(),
            ],
        }
    }

    /// Assess manufacturing for consumer Spark Engine
    pub fn assess_consumer_spark(&self) -> ManufacturingAssessment {
        let mut processes = HashMap::new();
        let mut joints = Vec::new();
        let mut compatibility_issues = Vec::new();
        let mut recommendations = Vec::new();

        // HEA Shell - requires powder metallurgy or AM
        processes.insert(
            "HEA Shell".to_string(),
            ProcessCapability::powder_metallurgy(),
        );
        recommendations
            .push("HEA shell: Use HIP (Hot Isostatic Pressing) for full density".to_string());

        // Zr/Nb interface - sputtering/PVD
        processes.insert(
            "Zr/Nb Interface".to_string(),
            ProcessCapability {
                name: "Magnetron Sputtering".to_string(),
                tolerance_mm: 0.001,
                surface_finish_um: 0.1,
                relative_cost: 4.0,
                lead_time_weeks: 4.0,
                compatible_materials: vec!["Zr".to_string(), "Nb".to_string()],
            },
        );

        // Containment vessel - CNC machined stainless
        processes.insert(
            "Containment Vessel".to_string(),
            ProcessCapability::cnc_machining(),
        );

        // Shielding - standard processes
        processes.insert(
            "Neutron Shield".to_string(),
            ProcessCapability::sheet_forming(),
        );

        // Check material compatibility
        let (galinstan_hea, note) = self.compatibility.check("Galinstan", "HEA");
        if galinstan_hea == Compatibility::Conditional {
            compatibility_issues.push(format!(
                "Galinstan-HEA interface: {}",
                note.unwrap_or("Verify compatibility")
            ));
        }

        // Define joints
        joints.push((
            "HEA Shell".to_string(),
            "Containment".to_string(),
            JoiningMethod::electron_beam_welding(),
        ));
        joints.push((
            "Containment".to_string(),
            "Shield".to_string(),
            JoiningMethod::bolted_flange(),
        ));

        // Tolerance stackup for critical gap
        let core_gap_stackup = ToleranceStackup::calculate(
            "Core-Shell Gap",
            2.0, // 2mm nominal gap
            vec![
                ToleranceContributor {
                    component: "Shell ID".to_string(),
                    tolerance_mm: 0.5,
                    sensitivity: 1.0,
                },
                ToleranceContributor {
                    component: "Core OD".to_string(),
                    tolerance_mm: 0.25,
                    sensitivity: -1.0,
                },
                ToleranceContributor {
                    component: "Concentricity".to_string(),
                    tolerance_mm: 0.1,
                    sensitivity: 1.0,
                },
            ],
            1.0, // 1mm tolerance on gap
        );

        let stackups = vec![core_gap_stackup];

        // NDE plan
        let nde_plan = vec![
            ("HEA Shell welds".to_string(), NdeMethod::radiography()),
            ("Pressure boundary".to_string(), NdeMethod::helium_leak()),
            ("All welds".to_string(), NdeMethod::dye_penetrant()),
        ];

        // Assembly sequence
        let assembly_sequence = vec![
            AssemblyStep {
                step: 1,
                description: "Machine containment vessel".to_string(),
                components: vec!["Containment".to_string()],
                joining: None,
                nde_required: vec!["PT".to_string()],
                critical_dims: vec!["ID".to_string(), "Concentricity".to_string()],
                time_hours: 8.0,
            },
            AssemblyStep {
                step: 2,
                description: "HIP HEA shell blank".to_string(),
                components: vec!["HEA Shell".to_string()],
                joining: None,
                nde_required: vec!["UT".to_string()],
                critical_dims: vec!["Density".to_string()],
                time_hours: 40.0,
            },
            AssemblyStep {
                step: 3,
                description: "Machine HEA shell to final dimensions".to_string(),
                components: vec!["HEA Shell".to_string()],
                joining: None,
                nde_required: vec!["PT".to_string()],
                critical_dims: vec![
                    "ID".to_string(),
                    "OD".to_string(),
                    "Wall thickness".to_string(),
                ],
                time_hours: 16.0,
            },
            AssemblyStep {
                step: 4,
                description: "Deposit Zr/Nb nano-laminate".to_string(),
                components: vec!["HEA Shell".to_string(), "Zr/Nb Interface".to_string()],
                joining: Some("Sputtering".to_string()),
                nde_required: vec!["Thickness measurement".to_string()],
                critical_dims: vec!["Layer thickness".to_string()],
                time_hours: 24.0,
            },
            AssemblyStep {
                step: 5,
                description: "Weld shell to containment".to_string(),
                components: vec!["HEA Shell".to_string(), "Containment".to_string()],
                joining: Some("EBW".to_string()),
                nde_required: vec!["RT".to_string(), "PT".to_string()],
                critical_dims: vec!["Weld penetration".to_string()],
                time_hours: 8.0,
            },
            AssemblyStep {
                step: 6,
                description: "Fill with Galinstan and seal".to_string(),
                components: vec!["Galinstan".to_string()],
                joining: Some("Mechanical seal".to_string()),
                nde_required: vec!["Helium leak".to_string()],
                critical_dims: vec!["Fill level".to_string()],
                time_hours: 4.0,
            },
            AssemblyStep {
                step: 7,
                description: "Install neutron shielding".to_string(),
                components: vec!["Neutron Shield".to_string()],
                joining: Some("Bolted".to_string()),
                nde_required: vec![],
                critical_dims: vec!["Coverage".to_string()],
                time_hours: 8.0,
            },
            AssemblyStep {
                step: 8,
                description: "Final leak test and acceptance".to_string(),
                components: vec!["Complete assembly".to_string()],
                joining: None,
                nde_required: vec!["Helium leak".to_string()],
                critical_dims: vec!["Leak rate".to_string()],
                time_hours: 4.0,
            },
        ];

        // Calculate totals
        let total_cost: f64 = processes.values().map(|p| p.relative_cost).sum::<f64>()
            + joints.iter().map(|(_, _, j)| j.relative_cost).sum::<f64>();

        let lead_time_weeks: f64 = processes
            .values()
            .map(|p| p.lead_time_weeks)
            .fold(0.0f64, |a, b| a.max(b))
            + 2.0; // Assembly time

        // Risk assessment
        let mut risk_score: f64 = 0.0;
        if !compatibility_issues.is_empty() {
            risk_score += 0.2;
        }
        for stackup in &stackups {
            if !stackup.within_spec {
                risk_score += 0.3;
                recommendations.push(format!(
                    "Tolerance issue: {} (RSS {:.2}mm > spec {:.2}mm)",
                    stackup.feature, stackup.rss_mm, stackup.spec_limit_mm
                ));
            }
        }
        // HEA processing adds risk
        risk_score += 0.15;
        recommendations
            .push("HEA processing: Establish supplier qualification program".to_string());

        ManufacturingAssessment {
            design_name: "Consumer Spark Engine (5 kW)".to_string(),
            processes,
            joints,
            compatibility_issues,
            stackups,
            nde_plan,
            assembly_sequence,
            total_cost,
            lead_time_weeks,
            risk_score: risk_score.min(1.0),
            recommendations,
        }
    }

    /// Print manufacturing assessment
    pub fn print_assessment(&self, assessment: &ManufacturingAssessment) {
        println!("\n");
        println!("┌────────────────────────────────────────────────────────────────────┐");
        println!("│              MANUFACTURING ASSESSMENT                              │");
        println!(
            "│              {}                              │",
            assessment.design_name
        );
        println!("├────────────────────────────────────────────────────────────────────┤");

        println!("│                                                                    │");
        println!("│  PROCESSES BY COMPONENT                                            │");
        println!("│  ─────────────────────────────────────────────────────────────     │");
        for (component, process) in &assessment.processes {
            println!(
                "│  {:20} → {:25} (±{:.2}mm)  │",
                component, process.name, process.tolerance_mm
            );
        }

        println!("│                                                                    │");
        println!("│  JOINTS                                                            │");
        println!("│  ─────────────────────────────────────────────────────────────     │");
        for (comp1, comp2, method) in &assessment.joints {
            println!("│  {} ─── {} : {}        │", comp1, comp2, method.name);
        }

        if !assessment.compatibility_issues.is_empty() {
            println!("│                                                                    │");
            println!("│  ⚠ COMPATIBILITY ISSUES                                           │");
            println!("│  ─────────────────────────────────────────────────────────────     │");
            for issue in &assessment.compatibility_issues {
                println!("│  • {issue}  │");
            }
        }

        println!("│                                                                    │");
        println!("│  TOLERANCE ANALYSIS                                                │");
        println!("│  ─────────────────────────────────────────────────────────────     │");
        for stackup in &assessment.stackups {
            let status = if stackup.within_spec { "✓" } else { "✗" };
            println!(
                "│  {} {}: RSS={:.2}mm (spec={:.2}mm)            │",
                status, stackup.feature, stackup.rss_mm, stackup.spec_limit_mm
            );
        }

        println!("│                                                                    │");
        println!("│  SUMMARY                                                           │");
        println!("│  ─────────────────────────────────────────────────────────────     │");
        println!(
            "│  Relative cost factor:  {:.1}x                                      │",
            assessment.total_cost
        );
        println!(
            "│  Lead time:             {:.0} weeks                                  │",
            assessment.lead_time_weeks
        );
        println!(
            "│  Risk score:            {:.0}%                                       │",
            assessment.risk_score * 100.0
        );

        if !assessment.recommendations.is_empty() {
            println!("│                                                                    │");
            println!("│  RECOMMENDATIONS                                                   │");
            println!("│  ─────────────────────────────────────────────────────────────     │");
            for rec in &assessment.recommendations {
                println!("│  • {rec}  │");
            }
        }

        println!("│                                                                    │");
        println!("└────────────────────────────────────────────────────────────────────┘");
    }
}

impl Default for ManufacturingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_capabilities() {
        let cnc = ProcessCapability::cnc_machining();
        assert!(cnc.tolerance_mm < 0.1);
        assert!(cnc.relative_cost > 0.0);
    }

    #[test]
    fn test_material_compatibility() {
        let compat = MaterialCompatibility::spark_engine_default();

        // Galinstan should not contact aluminum
        let (result, _) = compat.check("Galinstan", "Aluminum");
        assert_eq!(result, Compatibility::Incompatible);

        // Galinstan-Steel should be compatible
        let (result, _) = compat.check("Galinstan", "Steel");
        assert_eq!(result, Compatibility::Compatible);
    }

    #[test]
    fn test_tolerance_stackup() {
        let stackup = ToleranceStackup::calculate(
            "Test Gap",
            10.0,
            vec![
                ToleranceContributor {
                    component: "Part A".to_string(),
                    tolerance_mm: 0.1,
                    sensitivity: 1.0,
                },
                ToleranceContributor {
                    component: "Part B".to_string(),
                    tolerance_mm: 0.1,
                    sensitivity: -1.0,
                },
            ],
            0.2,
        );

        // RSS should be sqrt(0.1^2 + 0.1^2) ≈ 0.141
        assert!((stackup.rss_mm - 0.141).abs() < 0.01);
        assert!(stackup.within_spec);
    }

    #[test]
    fn test_manufacturing_assessment() {
        let engine = ManufacturingEngine::new();
        let assessment = engine.assess_consumer_spark();

        assert!(!assessment.processes.is_empty());
        assert!(!assessment.joints.is_empty());
        assert!(!assessment.assembly_sequence.is_empty());
        assert!(assessment.total_cost > 0.0);
        assert!(assessment.lead_time_weeks > 0.0);
    }
}
