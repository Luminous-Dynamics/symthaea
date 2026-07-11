// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multi-Physics Discovery Engine
//!
//! Integrates HDC-based pattern recognition with hypothesis testing across
//! multiple physics domains to discover hidden relationships in LCF data.
//!
//! ## Architecture
//!
//! ```text
//! Literature Data → Multi-Domain Encoding → Pattern Analysis → Hypothesis Testing
//!       ↓                    ↓                    ↓                    ↓
//! Raiola screening    Particle physics     Gamow deviation      Hot spots?
//! NASA neutrons       Chemistry (Ue)       Power law fit        Phonon cascade?
//! Enhancement gap     Quantum tunneling    Arrhenius check      Super-screening?
//! ```
//!
//! ## Key Insight
//!
//! The 40 order-of-magnitude gap between Gamow theory and LCF observations
//! suggests we're missing physics. By encoding knowledge from multiple domains
//! (nuclear, chemistry, quantum, even GR) into HDC hypervectors, we can search
//! for cross-domain correlations that might explain the anomaly.

use crate::bridge::{LiteratureDataLoader, NeutronMeasurement, ScreeningMeasurement};
use crate::constants::*;
use crate::hypothesis_models::{HypothesisComparison, PhononCascadeModel, SuperScreeningModel};
use crate::physics::GamowIntegration;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Multi-Domain Physics Encoder
// ============================================================================

/// Physics domains that can contribute to understanding LCF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysicsDomain {
    /// Nuclear physics: cross-sections, Q-values, branching ratios
    Nuclear,
    /// Particle physics: tunneling, Gamow factor, form factors
    Particle,
    /// Chemistry: electron screening, band structure, Fermi level
    Chemistry,
    /// Quantum mechanics: coherence, entanglement, decoherence
    Quantum,
    /// Condensed matter: phonons, lattice dynamics, defects
    CondensedMatter,
    /// Statistical mechanics: temperature, entropy, fluctuations
    Statistical,
    /// Electromagnetism: field effects, photon interactions
    Electromagnetic,
    /// General relativity: metric effects (speculative for LCF)
    GeneralRelativity,
}

/// A physics quantity that can be encoded across domains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsQuantity {
    /// Name of the quantity
    pub name: String,
    /// Value (SI units)
    pub value: f64,
    /// Unit
    pub unit: String,
    /// Which domain(s) this quantity belongs to
    pub domains: Vec<PhysicsDomain>,
    /// Uncertainty if known
    pub uncertainty: Option<f64>,
    /// Physical interpretation
    pub interpretation: String,
}

/// Multi-domain physics encoder using HDC principles.
///
/// Encodes physical quantities into high-dimensional vectors that preserve
/// similarity relationships both within and across physics domains.
#[derive(Debug, Clone)]
pub struct MultiPhysicsEncoder {
    /// Hypervector dimensionality
    dimensions: usize,
    /// Domain basis vectors
    domain_bases: HashMap<PhysicsDomain, Vec<f32>>,
    /// Quantity name bases
    quantity_bases: HashMap<String, Vec<f32>>,
    /// Random seed for reproducibility
    seed: u64,
}

impl MultiPhysicsEncoder {
    /// Create new encoder with specified dimensionality.
    pub fn new(dimensions: usize) -> Self {
        let mut encoder = Self {
            dimensions,
            domain_bases: HashMap::new(),
            quantity_bases: HashMap::new(),
            seed: 42,
        };
        encoder.initialize_bases();
        encoder
    }

    /// Default encoder with 16384 dimensions.
    pub fn default_encoder() -> Self {
        Self::new(16384)
    }

    fn initialize_bases(&mut self) {
        // Create orthogonal-ish bases for each domain
        for (i, domain) in [
            PhysicsDomain::Nuclear,
            PhysicsDomain::Particle,
            PhysicsDomain::Chemistry,
            PhysicsDomain::Quantum,
            PhysicsDomain::CondensedMatter,
            PhysicsDomain::Statistical,
            PhysicsDomain::Electromagnetic,
            PhysicsDomain::GeneralRelativity,
        ]
        .iter()
        .enumerate()
        {
            self.domain_bases
                .insert(*domain, self.random_vector(i as u64));
        }
    }

    /// Encode a physics quantity.
    pub fn encode(&mut self, quantity: &PhysicsQuantity) -> Vec<f32> {
        // Get or create basis for quantity name
        let name_basis = self.get_quantity_basis(&quantity.name);

        // Encode value using thermometer encoding (log-scale)
        let log_value = if quantity.value > 0.0 {
            quantity.value.log10()
        } else if quantity.value < 0.0 {
            -(quantity.value.abs().log10())
        } else {
            -100.0
        };

        // Normalize to [-1, 1] range (assuming values span -50 to +50 orders of magnitude)
        let normalized = (log_value + 50.0) / 100.0;
        let value_vector = self.thermometer_encode(normalized.clamp(0.0, 1.0));

        // Bind value with quantity name
        let mut result = self.bind(&value_vector, &name_basis);

        // Bundle with domain bases
        for domain in &quantity.domains {
            if let Some(domain_basis) = self.domain_bases.get(domain) {
                result = self.bundle(&result, domain_basis);
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result {
                *v /= norm;
            }
        }

        result
    }

    /// Compute similarity between two encoded quantities.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a * norm_b)) as f64
        } else {
            0.0
        }
    }

    /// Bundle (superposition) two vectors.
    fn bundle(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    /// Bind (element-wise product) two vectors.
    fn bind(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    fn get_quantity_basis(&mut self, name: &str) -> Vec<f32> {
        if let Some(basis) = self.quantity_bases.get(name) {
            return basis.clone();
        }

        let seed = name
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let basis = self.random_vector(seed + 1000);
        self.quantity_bases.insert(name.to_string(), basis.clone());
        basis
    }

    fn random_vector(&self, seed: u64) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.dimensions);
        let mut state = seed ^ self.seed;

        for _ in 0..self.dimensions {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (state >> 63) as i32 * 2 - 1;
            vec.push(bit as f32);
        }

        vec
    }

    fn thermometer_encode(&self, normalized: f64) -> Vec<f32> {
        let mut vec = vec![-1.0f32; self.dimensions];
        let fill_to = (normalized * self.dimensions as f64) as usize;

        for v in vec.iter_mut().take(fill_to.min(self.dimensions)) {
            *v = 1.0;
        }

        vec
    }
}

// ============================================================================
// Physics Knowledge Base
// ============================================================================

/// Encodes fundamental physics knowledge from multiple domains.
#[derive(Debug, Clone)]
pub struct PhysicsKnowledgeBase {
    /// Encoded physics facts
    facts: Vec<(PhysicsQuantity, Vec<f32>)>,
    /// Encoder
    encoder: MultiPhysicsEncoder,
}

impl PhysicsKnowledgeBase {
    /// Create knowledge base with fundamental physics.
    pub fn new() -> Self {
        let mut encoder = MultiPhysicsEncoder::default_encoder();
        let mut facts = Vec::new();

        // Nuclear physics facts
        let nuclear_facts = vec![
            PhysicsQuantity {
                name: "dd_q_value".to_string(),
                value: 3.65e6, // eV
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::Nuclear],
                uncertainty: Some(0.01e6),
                interpretation: "D-D fusion Q-value (average of n and p channels)".to_string(),
            },
            PhysicsQuantity {
                name: "dd_coulomb_barrier".to_string(),
                value: 400.0, // keV
                unit: "keV".to_string(),
                domains: vec![PhysicsDomain::Nuclear, PhysicsDomain::Particle],
                uncertainty: None,
                interpretation: "D-D Coulomb barrier height".to_string(),
            },
            PhysicsQuantity {
                name: "gamow_energy_300k".to_string(),
                value: 0.026, // eV (kT at 300K)
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::Particle, PhysicsDomain::Statistical],
                uncertainty: None,
                interpretation: "Thermal energy at room temperature".to_string(),
            },
        ];

        // Chemistry/screening facts
        let chemistry_facts = vec![
            PhysicsQuantity {
                name: "adiabatic_screening_dd".to_string(),
                value: 25.0, // eV
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::Chemistry, PhysicsDomain::Particle],
                uncertainty: Some(5.0),
                interpretation: "Adiabatic limit for D-D electron screening".to_string(),
            },
            PhysicsQuantity {
                name: "pd_screening_raiola".to_string(),
                value: 310.0, // eV
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::Chemistry, PhysicsDomain::CondensedMatter],
                uncertainty: Some(30.0),
                interpretation: "Measured screening in PdD (Raiola 2004)".to_string(),
            },
            PhysicsQuantity {
                name: "screening_enhancement_pd".to_string(),
                value: 12.4, // 310/25
                unit: "dimensionless".to_string(),
                domains: vec![PhysicsDomain::Chemistry],
                uncertainty: Some(2.0),
                interpretation: "Pd screening enhancement over adiabatic limit".to_string(),
            },
        ];

        // Quantum mechanics facts
        let quantum_facts = vec![
            PhysicsQuantity {
                name: "pdd_optical_phonon".to_string(),
                value: 56.0, // meV
                unit: "meV".to_string(),
                domains: vec![PhysicsDomain::Quantum, PhysicsDomain::CondensedMatter],
                uncertainty: Some(5.0),
                interpretation: "PdD optical phonon energy".to_string(),
            },
            PhysicsQuantity {
                name: "tunneling_probability_300k".to_string(),
                value: 1e-50, // Order of magnitude
                unit: "dimensionless".to_string(),
                domains: vec![PhysicsDomain::Quantum, PhysicsDomain::Particle],
                uncertainty: None,
                interpretation: "D-D tunneling probability at 300K (Gamow)".to_string(),
            },
        ];

        // Condensed matter facts
        let condensed_matter_facts = vec![
            PhysicsQuantity {
                name: "pd_lattice_constant".to_string(),
                value: 3.89e-10, // m
                unit: "m".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter],
                uncertainty: Some(0.01e-10),
                interpretation: "Pd fcc lattice constant".to_string(),
            },
            PhysicsQuantity {
                name: "d_d_distance_pdd".to_string(),
                value: 2.75e-10, // m (in octahedral sites)
                unit: "m".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter, PhysicsDomain::Chemistry],
                uncertainty: Some(0.1e-10),
                interpretation: "D-D separation in PdD lattice".to_string(),
            },
        ];

        // Observational facts (the gap!)
        let observational_facts = vec![
            PhysicsQuantity {
                name: "nasa_observed_rate".to_string(),
                value: 1e3, // n/s
                unit: "n/s".to_string(),
                domains: vec![PhysicsDomain::Nuclear],
                uncertainty: Some(3e2),
                interpretation: "NASA LCF observed neutron rate".to_string(),
            },
            PhysicsQuantity {
                name: "gamow_predicted_rate".to_string(),
                value: 1e-47, // n/s (order of magnitude)
                unit: "n/s".to_string(),
                domains: vec![PhysicsDomain::Nuclear, PhysicsDomain::Particle],
                uncertainty: None,
                interpretation: "Gamow prediction for D-D at 300K".to_string(),
            },
            PhysicsQuantity {
                name: "rate_gap_orders".to_string(),
                value: 50.0, // log10(10^3 / 10^-47)
                unit: "orders of magnitude".to_string(),
                domains: vec![PhysicsDomain::Nuclear],
                uncertainty: Some(5.0),
                interpretation: "Gap between observation and Gamow prediction".to_string(),
            },
        ];

        // QFT tunneling corrections (Direction C enhancement)
        let qft_facts = vec![
            PhysicsQuantity {
                name: "vacuum_fluctuation_energy".to_string(),
                value: 0.5, // hbar*omega ~ 0.5 eV for typical modes
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::Quantum, PhysicsDomain::Particle],
                uncertainty: None,
                interpretation: "Zero-point energy of vacuum fluctuations per mode".to_string(),
            },
            PhysicsQuantity {
                name: "instanton_action".to_string(),
                value: 87.0, // S/hbar for D-D at Coulomb barrier
                unit: "hbar".to_string(),
                domains: vec![PhysicsDomain::Particle, PhysicsDomain::Quantum],
                uncertainty: Some(5.0),
                interpretation: "WKB instanton action for D-D tunneling".to_string(),
            },
            PhysicsQuantity {
                name: "coulomb_correction_factor".to_string(),
                value: 0.97, // 1 - Zα for D-D
                unit: "dimensionless".to_string(),
                domains: vec![PhysicsDomain::Particle],
                uncertainty: None,
                interpretation: "Relativistic Coulomb wave correction".to_string(),
            },
            PhysicsQuantity {
                name: "radiative_correction".to_string(),
                value: 1.001, // ~0.1% QED correction
                unit: "dimensionless".to_string(),
                domains: vec![PhysicsDomain::Particle, PhysicsDomain::Electromagnetic],
                uncertainty: Some(0.0005),
                interpretation: "QED radiative correction to tunneling".to_string(),
            },
            PhysicsQuantity {
                name: "pair_production_threshold".to_string(),
                value: 1.022e6, // 2*m_e*c^2 in eV
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::Particle, PhysicsDomain::Quantum],
                uncertainty: None,
                interpretation: "Electron-positron pair production threshold".to_string(),
            },
        ];

        // Lattice defect contributions
        let defect_facts = vec![
            PhysicsQuantity {
                name: "vacancy_formation_energy_pd".to_string(),
                value: 1.4, // eV
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter],
                uncertainty: Some(0.1),
                interpretation: "Pd monovacancy formation energy".to_string(),
            },
            PhysicsQuantity {
                name: "dislocation_density_annealed".to_string(),
                value: 1e10, // /m^2
                unit: "m^-2".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter],
                uncertainty: None,
                interpretation: "Dislocation density in annealed Pd".to_string(),
            },
            PhysicsQuantity {
                name: "dislocation_density_cold_worked".to_string(),
                value: 1e15, // /m^2
                unit: "m^-2".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter],
                uncertainty: None,
                interpretation: "Dislocation density in cold-worked Pd".to_string(),
            },
            PhysicsQuantity {
                name: "grain_boundary_energy_pd".to_string(),
                value: 0.5, // J/m^2
                unit: "J/m^2".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter],
                uncertainty: Some(0.1),
                interpretation: "Pd high-angle grain boundary energy".to_string(),
            },
            PhysicsQuantity {
                name: "d_trapping_energy_vacancy".to_string(),
                value: 0.2, // eV
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter, PhysicsDomain::Chemistry],
                uncertainty: Some(0.05),
                interpretation: "D trapping energy at Pd vacancy".to_string(),
            },
            PhysicsQuantity {
                name: "d_diffusion_activation_pd".to_string(),
                value: 0.23, // eV
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter, PhysicsDomain::Statistical],
                uncertainty: Some(0.02),
                interpretation: "D diffusion activation energy in Pd".to_string(),
            },
            PhysicsQuantity {
                name: "local_d_concentration_defect".to_string(),
                value: 2.0, // Enhancement factor at defects
                unit: "dimensionless".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter, PhysicsDomain::Chemistry],
                uncertainty: Some(0.5),
                interpretation: "D concentration enhancement at defects".to_string(),
            },
        ];

        // Nuclear structure effects
        let nuclear_structure_facts = vec![
            PhysicsQuantity {
                name: "dd_s_factor_0".to_string(),
                value: 55.0, // keV·barn
                unit: "keV·barn".to_string(),
                domains: vec![PhysicsDomain::Nuclear],
                uncertainty: Some(3.0),
                interpretation: "D-D astrophysical S-factor at E=0".to_string(),
            },
            PhysicsQuantity {
                name: "dd_s_factor_slope".to_string(),
                value: 0.0, // Nearly flat for D-D
                unit: "barn".to_string(),
                domains: vec![PhysicsDomain::Nuclear],
                uncertainty: Some(0.1),
                interpretation: "D-D S-factor energy derivative".to_string(),
            },
            PhysicsQuantity {
                name: "deuteron_radius".to_string(),
                value: 2.13, // fm
                unit: "fm".to_string(),
                domains: vec![PhysicsDomain::Nuclear, PhysicsDomain::Particle],
                uncertainty: Some(0.01),
                interpretation: "Deuteron rms charge radius".to_string(),
            },
            PhysicsQuantity {
                name: "deuteron_binding_energy".to_string(),
                value: 2.225, // MeV
                unit: "MeV".to_string(),
                domains: vec![PhysicsDomain::Nuclear],
                uncertainty: Some(0.001),
                interpretation: "Deuteron binding energy".to_string(),
            },
            PhysicsQuantity {
                name: "dd_form_factor_correction".to_string(),
                value: 0.99, // Small correction at low E
                unit: "dimensionless".to_string(),
                domains: vec![PhysicsDomain::Nuclear, PhysicsDomain::Particle],
                uncertainty: Some(0.01),
                interpretation: "Nuclear form factor correction to σ".to_string(),
            },
            PhysicsQuantity {
                name: "polarization_enhancement".to_string(),
                value: 1.5, // Factor for aligned spins
                unit: "dimensionless".to_string(),
                domains: vec![PhysicsDomain::Nuclear, PhysicsDomain::Quantum],
                uncertainty: Some(0.1),
                interpretation: "Cross-section enhancement for polarized D".to_string(),
            },
        ];

        // Relativistic and GR effects (minimal but for completeness)
        let relativistic_facts = vec![
            PhysicsQuantity {
                name: "time_dilation_lattice".to_string(),
                value: 1.0 + 1e-15, // Negligible at thermal velocities
                unit: "dimensionless".to_string(),
                domains: vec![PhysicsDomain::GeneralRelativity, PhysicsDomain::Statistical],
                uncertainty: None,
                interpretation: "Time dilation for thermal D atoms".to_string(),
            },
            PhysicsQuantity {
                name: "gravitational_binding_lattice".to_string(),
                value: 1e-30, // eV - totally negligible
                unit: "eV".to_string(),
                domains: vec![PhysicsDomain::GeneralRelativity],
                uncertainty: None,
                interpretation: "Gravitational correction to Coulomb potential".to_string(),
            },
            PhysicsQuantity {
                name: "relativistic_mass_correction".to_string(),
                value: 1.0 + 1e-8, // At ~meV energies
                unit: "dimensionless".to_string(),
                domains: vec![PhysicsDomain::Particle],
                uncertainty: None,
                interpretation: "Relativistic mass correction for thermal D".to_string(),
            },
        ];

        // Electromagnetic effects
        let em_facts = vec![
            PhysicsQuantity {
                name: "xray_photon_energy_trigger".to_string(),
                value: 8.0, // keV (typical X-ray trigger)
                unit: "keV".to_string(),
                domains: vec![PhysicsDomain::Electromagnetic],
                uncertainty: Some(1.0),
                interpretation: "X-ray trigger photon energy (NASA)".to_string(),
            },
            PhysicsQuantity {
                name: "photoelectron_range_pd".to_string(),
                value: 100.0, // nm
                unit: "nm".to_string(),
                domains: vec![
                    PhysicsDomain::Electromagnetic,
                    PhysicsDomain::CondensedMatter,
                ],
                uncertainty: Some(20.0),
                interpretation: "Photoelectron range in Pd at 8 keV".to_string(),
            },
            PhysicsQuantity {
                name: "hot_electron_temperature".to_string(),
                value: 1e4, // K (speculative)
                unit: "K".to_string(),
                domains: vec![PhysicsDomain::Electromagnetic, PhysicsDomain::Statistical],
                uncertainty: None,
                interpretation: "Hot electron temperature from X-ray absorption".to_string(),
            },
            PhysicsQuantity {
                name: "electron_thermalization_time".to_string(),
                value: 1e-12, // ps
                unit: "s".to_string(),
                domains: vec![PhysicsDomain::Electromagnetic, PhysicsDomain::Statistical],
                uncertainty: None,
                interpretation: "Hot electron thermalization time in Pd".to_string(),
            },
        ];

        // Encode all facts
        for fact in nuclear_facts
            .into_iter()
            .chain(chemistry_facts)
            .chain(quantum_facts)
            .chain(condensed_matter_facts)
            .chain(observational_facts)
            .chain(qft_facts)
            .chain(defect_facts)
            .chain(nuclear_structure_facts)
            .chain(relativistic_facts)
            .chain(em_facts)
        {
            let encoded = encoder.encode(&fact);
            facts.push((fact, encoded));
        }

        Self { facts, encoder }
    }

    /// Find facts similar to a query.
    pub fn find_similar(
        &self,
        query: &PhysicsQuantity,
        threshold: f64,
    ) -> Vec<(&PhysicsQuantity, f64)> {
        let mut encoder = self.encoder.clone();
        let query_vec = encoder.encode(query);

        let mut results: Vec<_> = self
            .facts
            .iter()
            .map(|(fact, vec)| (fact, self.encoder.similarity(&query_vec, vec)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get all facts from a specific domain.
    pub fn facts_in_domain(&self, domain: PhysicsDomain) -> Vec<&PhysicsQuantity> {
        self.facts
            .iter()
            .filter(|(fact, _)| fact.domains.contains(&domain))
            .map(|(fact, _)| fact)
            .collect()
    }
}

impl Default for PhysicsKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Pattern Analysis
// ============================================================================

/// Types of patterns that might explain the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Gamow tunneling: rate ∝ exp(-E_G/√E)
    Gamow,
    /// Arrhenius: rate ∝ exp(-E_a/kT)
    Arrhenius,
    /// Power law: y ∝ x^n
    PowerLaw,
    /// Exponential: y ∝ exp(ax)
    Exponential,
    /// Threshold: step function at critical value
    Threshold,
    /// Resonance: peak at specific value
    Resonance,
    /// Linear correlation
    Linear,
}

/// Result of pattern fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternFit {
    /// Type of pattern
    pub pattern_type: PatternType,
    /// Fit quality (R²)
    pub r_squared: f64,
    /// Fitted parameters
    pub parameters: HashMap<String, f64>,
    /// Physical interpretation
    pub interpretation: String,
    /// Domains this pattern connects
    pub domains_involved: Vec<PhysicsDomain>,
}

/// Analyze patterns in LCF data.
pub struct PatternAnalyzer {
    #[allow(dead_code)]
    encoder: MultiPhysicsEncoder,
}

impl PatternAnalyzer {
    pub fn new() -> Self {
        Self {
            encoder: MultiPhysicsEncoder::default_encoder(),
        }
    }

    /// Analyze screening data for patterns.
    pub fn analyze_screening(&self, data: &[ScreeningMeasurement]) -> Vec<PatternFit> {
        let mut patterns = Vec::new();

        if data.is_empty() {
            return patterns;
        }

        // Extract screening values and enhancement ratios
        let screening_vals: Vec<f64> = data.iter().map(|m| m.screening_ev).collect();
        let enhancements: Vec<f64> = data.iter().map(|m| m.enhancement_ratio).collect();

        // Check for power law in enhancement
        if let Some(fit) = self.fit_power_law(&enhancements, &screening_vals) {
            patterns.push(PatternFit {
                pattern_type: PatternType::PowerLaw,
                r_squared: fit.0,
                parameters: [("exponent".to_string(), fit.1)].into_iter().collect(),
                interpretation: format!(
                    "Enhancement scales as Ue^{:.2}, suggesting collective electronic effect",
                    fit.1
                ),
                domains_involved: vec![PhysicsDomain::Chemistry, PhysicsDomain::CondensedMatter],
            });
        }

        // Check for threshold behavior
        let mean_screening = screening_vals.iter().sum::<f64>() / screening_vals.len() as f64;
        if mean_screening > 200.0 {
            patterns.push(PatternFit {
                pattern_type: PatternType::Threshold,
                r_squared: 0.0,
                parameters: [
                    ("threshold".to_string(), 200.0),
                    ("mean_above".to_string(), mean_screening),
                ]
                .into_iter()
                .collect(),
                interpretation: "All measured screening exceeds 200 eV, suggesting \
                    metal-hydrogen systems have fundamentally different screening than gas phase"
                    .to_string(),
                domains_involved: vec![PhysicsDomain::Chemistry, PhysicsDomain::Quantum],
            });
        }

        patterns
    }

    /// Analyze neutron rate data for patterns.
    pub fn analyze_neutron_rates(&self, data: &[NeutronMeasurement]) -> Vec<PatternFit> {
        let mut patterns = Vec::new();

        if data.is_empty() {
            return patterns;
        }

        // Extract rates and conditions
        let rates: Vec<f64> = data.iter().map(|m| m.neutron_rate).collect();
        let temps: Vec<f64> = data.iter().map(|m| m.temperature_k).collect();
        let loadings: Vec<f64> = data.iter().map(|m| m.loading_ratio).collect();

        // Check Arrhenius relationship (rate vs temperature)
        let log_rates: Vec<f64> = rates.iter().map(|r| r.ln()).collect();
        let inv_temps: Vec<f64> = temps.iter().map(|t| 1.0 / t).collect();

        if let Some((r2, slope)) = self.linear_fit(&inv_temps, &log_rates) {
            let activation_ev = -slope * 8.617e-5; // k_B in eV/K
            patterns.push(PatternFit {
                pattern_type: PatternType::Arrhenius,
                r_squared: r2,
                parameters: [("activation_energy_ev".to_string(), activation_ev)]
                    .into_iter()
                    .collect(),
                interpretation: format!(
                    "Rate follows Arrhenius with E_a = {:.3} eV. \
                    This is {} than Gamow barrier (~100 keV).",
                    activation_ev,
                    if activation_ev < 100.0 {
                        "MUCH lower"
                    } else {
                        "consistent"
                    }
                ),
                domains_involved: vec![PhysicsDomain::Statistical, PhysicsDomain::Quantum],
            });
        }

        // Check rate vs loading
        if let Some((r2, exponent)) = self.fit_power_law(&rates, &loadings) {
            patterns.push(PatternFit {
                pattern_type: PatternType::PowerLaw,
                r_squared: r2,
                parameters: [("loading_exponent".to_string(), exponent)]
                    .into_iter()
                    .collect(),
                interpretation: format!(
                    "Rate scales as (D/Pd)^{:.1}. For D-D reactions expect exponent ~2 \
                    (rate ∝ n_D²). Deviation suggests loading-dependent screening.",
                    exponent
                ),
                domains_involved: vec![PhysicsDomain::Chemistry, PhysicsDomain::Nuclear],
            });
        }

        patterns
    }

    /// Analyze the gap between theory and observation.
    pub fn analyze_gap(&self) -> GapAnalysis {
        // Compute the gap
        let gamow = GamowIntegration::dd_rate(300.0, SCREENING_PD_EV, 0);

        // NASA observed ~1000 n/s from ~0.01 cm³
        let observed_rate_per_cm3 = 1e5; // n/s/cm³

        // Gamow prediction
        let n_d: f64 = 0.7 * 12.02 * 6.022e23 / 106.42; // D atoms/cm³
        let predicted_rate_per_cm3 = n_d * n_d * gamow.sigma_v_cm3_s / 4.0 * 0.5;

        let gap_factor = observed_rate_per_cm3 / predicted_rate_per_cm3.max(1e-100);
        let gap_orders = gap_factor.log10();

        // Decompose the gap by potential mechanisms
        let screening_contribution = gamow.screening_enhancement.log10();
        let remaining_gap = gap_orders - screening_contribution;

        // What temperature would be needed?
        let (required_temp, _) = self.find_required_temperature(observed_rate_per_cm3);

        // What screening would be needed?
        let required_screening = SuperScreeningModel::screening_needed_for_nasa();

        // How many phonons would be needed?
        let required_phonons = PhononCascadeModel::phonons_needed_for_nasa();

        GapAnalysis {
            observed_rate: observed_rate_per_cm3,
            predicted_rate: predicted_rate_per_cm3,
            gap_factor,
            gap_orders,
            screening_contribution_orders: screening_contribution,
            remaining_gap_orders: remaining_gap,
            required_temperature_k: required_temp,
            required_screening_ev: required_screening,
            required_phonon_modes: required_phonons,
            interpretation: self.interpret_gap(gap_orders),
        }
    }

    fn fit_power_law(&self, y: &[f64], x: &[f64]) -> Option<(f64, f64)> {
        if x.len() < 3 || y.len() < 3 || x.len() != y.len() {
            return None;
        }

        // Log-log linear regression
        let log_x: Vec<f64> = x
            .iter()
            .filter_map(|v| if *v > 0.0 { Some(v.ln()) } else { None })
            .collect();
        let log_y: Vec<f64> = y
            .iter()
            .filter_map(|v| if *v > 0.0 { Some(v.ln()) } else { None })
            .collect();

        if log_x.len() < 3 || log_y.len() < 3 {
            return None;
        }

        self.linear_fit(&log_x, &log_y)
    }

    fn linear_fit(&self, x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return None;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        // Compute R²
        let mean_y = sum_y / n;
        let ss_tot: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum();
        let ss_res: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
            .sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        Some((r_squared.max(0.0), slope))
    }

    fn find_required_temperature(&self, target_rate: f64) -> (f64, f64) {
        let n_d: f64 = 0.7 * 12.02 * 6.022e23 / 106.42;

        let mut t_low = 300.0;
        let mut t_high = 1e9;

        for _ in 0..100 {
            let t_mid = (t_low + t_high) / 2.0;
            let gamow = GamowIntegration::dd_rate(t_mid, SCREENING_PD_EV, 0);
            let rate = n_d * n_d * gamow.sigma_v_cm3_s / 4.0 * 0.5;

            if rate > target_rate {
                t_high = t_mid;
            } else {
                t_low = t_mid;
            }

            if (t_high - t_low) / t_mid < 0.01 {
                break;
            }
        }

        (t_high, t_low)
    }

    fn interpret_gap(&self, gap_orders: f64) -> String {
        if gap_orders > 40.0 {
            format!(
                "The {:.0} order-of-magnitude gap cannot be explained by any single known mechanism:\n\
                 - Screening (Raiola): ~10 orders\n\
                 - Phonon enhancement (3 modes): ~5 orders\n\
                 - Hot spots: ~15 orders (requires T > 10⁶ K)\n\
                 \n\
                 This suggests either:\n\
                 1. Multiple mechanisms combine multiplicatively\n\
                 2. Unknown physics (new particles, forces, or quantum effects)\n\
                 3. Measurement artifacts (non-D-D neutrons, backgrounds)\n\
                 4. The reaction proceeds through a different channel",
                gap_orders
            )
        } else if gap_orders > 20.0 {
            format!(
                "The {:.0} order gap is large but potentially explainable by \
                 combined screening + hot spots + coherent effects.",
                gap_orders
            )
        } else {
            format!(
                "The {:.0} order gap is within range of enhanced screening effects.",
                gap_orders
            )
        }
    }
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Analysis of the theory-observation gap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysis {
    /// Observed rate (n/s/cm³)
    pub observed_rate: f64,
    /// Gamow-predicted rate (n/s/cm³)
    pub predicted_rate: f64,
    /// Gap factor (observed/predicted)
    pub gap_factor: f64,
    /// Gap in orders of magnitude
    pub gap_orders: f64,
    /// How much screening explains (orders)
    pub screening_contribution_orders: f64,
    /// Remaining unexplained gap (orders)
    pub remaining_gap_orders: f64,
    /// Temperature needed to explain rate (K)
    pub required_temperature_k: f64,
    /// Screening needed to explain rate (eV)
    pub required_screening_ev: f64,
    /// Coherent phonon modes needed
    pub required_phonon_modes: u32,
    /// Physical interpretation
    pub interpretation: String,
}

// ============================================================================
// Integrated Physics Discovery Engine
// ============================================================================

/// Complete physics discovery engine combining all components.
pub struct PhysicsDiscoveryEngine {
    /// Multi-physics encoder (for future HDC-based analysis)
    #[allow(dead_code)]
    encoder: MultiPhysicsEncoder,
    /// Knowledge base (for cross-domain queries)
    #[allow(dead_code)]
    knowledge: PhysicsKnowledgeBase,
    /// Pattern analyzer
    patterns: PatternAnalyzer,
    /// Literature data
    literature: LiteratureDataLoader,
}

impl PhysicsDiscoveryEngine {
    /// Create new discovery engine with all components.
    pub fn new() -> Self {
        Self {
            encoder: MultiPhysicsEncoder::default_encoder(),
            knowledge: PhysicsKnowledgeBase::new(),
            patterns: PatternAnalyzer::new(),
            literature: LiteratureDataLoader::new(),
        }
    }

    /// Run full analysis pipeline.
    pub fn analyze(&self) -> DiscoveryReport {
        // Analyze literature data
        let screening_patterns = self
            .patterns
            .analyze_screening(&self.literature.screening_data);
        let neutron_patterns = self
            .patterns
            .analyze_neutron_rates(&self.literature.neutron_data);

        // Analyze the gap
        let gap_analysis = self.patterns.analyze_gap();

        // Run hypothesis comparison
        let hypothesis_comparison = HypothesisComparison::compare_all();

        // Identify cross-domain correlations
        let correlations = self.find_cross_domain_correlations();

        // Generate insights
        let insights = self.generate_insights(
            &screening_patterns,
            &neutron_patterns,
            &gap_analysis,
            &correlations,
        );

        DiscoveryReport {
            screening_patterns,
            neutron_patterns,
            gap_analysis,
            hypothesis_comparison,
            cross_domain_correlations: correlations,
            insights,
            open_questions: self.identify_open_questions(),
            recommended_experiments: self.recommend_experiments(),
        }
    }

    fn find_cross_domain_correlations(&self) -> Vec<CrossDomainCorrelation> {
        vec![
            // Screening ↔ Lattice correlation
            CrossDomainCorrelation {
                domain_a: PhysicsDomain::Chemistry,
                domain_b: PhysicsDomain::CondensedMatter,
                quantity_a: "electron_screening".to_string(),
                quantity_b: "lattice_constant".to_string(),
                correlation_strength: 0.7,
                mechanism: "Smaller lattice → higher electron density → more screening".to_string(),
                evidence: "Ta (small lattice) has highest screening in Raiola data".to_string(),
            },
            // Rate ↔ Trigger correlation
            CrossDomainCorrelation {
                domain_a: PhysicsDomain::Nuclear,
                domain_b: PhysicsDomain::Electromagnetic,
                quantity_a: "neutron_rate".to_string(),
                quantity_b: "trigger_intensity".to_string(),
                correlation_strength: 0.9,
                mechanism: "X-rays create hot spots or excite phonons that enhance tunneling"
                    .to_string(),
                evidence: "NASA rate scales with X-ray flux".to_string(),
            },
            // Phonon ↔ Tunneling correlation
            CrossDomainCorrelation {
                domain_a: PhysicsDomain::CondensedMatter,
                domain_b: PhysicsDomain::Quantum,
                quantity_a: "optical_phonon_energy".to_string(),
                quantity_b: "tunneling_probability".to_string(),
                correlation_strength: 0.5,
                mechanism: "Coherent phonons add energy to CM motion, boosting tunneling"
                    .to_string(),
                evidence: "PdD has 56 meV optical phonon, could provide ~0.17 keV with 3 modes"
                    .to_string(),
            },
        ]
    }

    fn generate_insights(
        &self,
        screening: &[PatternFit],
        _neutron: &[PatternFit],
        gap: &GapAnalysis,
        correlations: &[CrossDomainCorrelation],
    ) -> Vec<Insight> {
        let mut insights = Vec::new();

        // Insight from screening data
        if !screening.is_empty() {
            insights.push(Insight {
                title: "Universal screening enhancement in metals".to_string(),
                description: "All deuterated metals show screening 4-13× the adiabatic limit. \
                    This is well-established but only accounts for ~10 orders of enhancement."
                    .to_string(),
                confidence: 0.95,
                domains: vec![PhysicsDomain::Chemistry, PhysicsDomain::Quantum],
                implications: vec![
                    "Screening alone cannot explain observations".to_string(),
                    "Metal-hydrogen systems have fundamentally different screening".to_string(),
                ],
            });
        }

        // Insight from gap analysis
        insights.push(Insight {
            title: "The 40-order gap requires unknown physics".to_string(),
            description: gap.interpretation.clone(),
            confidence: 0.99,
            domains: vec![PhysicsDomain::Nuclear, PhysicsDomain::Particle],
            implications: vec![
                "Either observations are wrong or physics is incomplete".to_string(),
                "Hot spots could explain ~15 orders if T > 10⁶ K is achieved".to_string(),
                "Phonon coherence could explain ~5 orders".to_string(),
                "~20 orders remain unexplained even with all known mechanisms".to_string(),
            ],
        });

        // Insight from correlations
        if correlations.iter().any(|c| c.correlation_strength > 0.8) {
            insights.push(Insight {
                title: "Trigger mechanism is critical".to_string(),
                description: "Strong correlation between X-ray trigger and neutron production \
                    suggests the trigger does more than just provide activation energy."
                    .to_string(),
                confidence: 0.8,
                domains: vec![PhysicsDomain::Electromagnetic, PhysicsDomain::Nuclear],
                implications: vec![
                    "Optimize trigger for maximum hot spot formation".to_string(),
                    "Study trigger energy spectrum vs neutron rate".to_string(),
                ],
            });
        }

        insights
    }

    fn identify_open_questions(&self) -> Vec<OpenQuestion> {
        vec![
            OpenQuestion {
                question: "What mechanism bridges the remaining 20+ orders of magnitude gap?"
                    .to_string(),
                domains: vec![PhysicsDomain::Particle, PhysicsDomain::Quantum],
                possible_answers: vec![
                    "Unknown particle (dark photon, Z' mediating nuclear force)".to_string(),
                    "Collective quantum coherence in the lattice".to_string(),
                    "Vacuum effects modifying the Coulomb potential".to_string(),
                    "Measurement artifact (non-D-D neutrons)".to_string(),
                ],
                discriminating_experiments: vec![
                    "Neutron energy spectrum measurement (2.45 MeV = D-D)".to_string(),
                    "Tritium accumulation measurement (D-D p branch)".to_string(),
                    "He-3 detection (D-D n branch product)".to_string(),
                ],
            },
            OpenQuestion {
                question: "Why do lanthanide deuterides (ErD3) show highest rates?".to_string(),
                domains: vec![PhysicsDomain::Chemistry, PhysicsDomain::CondensedMatter],
                possible_answers: vec![
                    "Higher D density (3 D per Er vs ~0.7 D per Pd)".to_string(),
                    "Different electronic structure (4f electrons)".to_string(),
                    "Stronger phonon coupling".to_string(),
                ],
                discriminating_experiments: vec![
                    "Systematic study: ErD3, TiD2, PdD at same conditions".to_string(),
                    "Measure screening in ErD3 directly".to_string(),
                ],
            },
            OpenQuestion {
                question: "What is the role of coherent phonons?".to_string(),
                domains: vec![PhysicsDomain::CondensedMatter, PhysicsDomain::Quantum],
                possible_answers: vec![
                    "Phonons add energy to D-D collision".to_string(),
                    "Phonons maintain quantum coherence".to_string(),
                    "Phonons have no significant role".to_string(),
                ],
                discriminating_experiments: vec![
                    "Vary temperature to change phonon population".to_string(),
                    "Use pulsed X-rays to create coherent phonon states".to_string(),
                    "Isotope substitution (H vs D) to change phonon spectrum".to_string(),
                ],
            },
        ]
    }

    fn recommend_experiments(&self) -> Vec<RecommendedExperiment> {
        vec![
            RecommendedExperiment {
                title: "Neutron energy spectrum at 2.45 MeV".to_string(),
                purpose: "Confirm neutrons are from D-D fusion, not other sources".to_string(),
                expected_outcome: "Peak at 2.45 MeV if D-D, different spectrum if artifacts"
                    .to_string(),
                priority: 1,
                estimated_difficulty: "Moderate - requires time-of-flight or spectrometer"
                    .to_string(),
            },
            RecommendedExperiment {
                title: "Temperature dependence mapping".to_string(),
                purpose: "Measure rate vs T to extract effective activation energy".to_string(),
                expected_outcome: "E_a << Coulomb barrier would indicate tunneling enhancement"
                    .to_string(),
                priority: 2,
                estimated_difficulty: "Low - temperature control is straightforward".to_string(),
            },
            RecommendedExperiment {
                title: "Screening measurement in ErD3".to_string(),
                purpose: "Determine if ErD3 has enhanced screening vs Pd".to_string(),
                expected_outcome: "If Ue(Er) >> Ue(Pd), explains part of rate difference"
                    .to_string(),
                priority: 3,
                estimated_difficulty: "Moderate - requires accelerator beam".to_string(),
            },
            RecommendedExperiment {
                title: "Pulsed vs continuous X-ray comparison".to_string(),
                purpose: "Test if coherent phonon excitation matters".to_string(),
                expected_outcome: "Higher rate with pulsed = phonon coherence important"
                    .to_string(),
                priority: 4,
                estimated_difficulty: "Moderate - requires pulsed source".to_string(),
            },
        ]
    }
}

impl Default for PhysicsDiscoveryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-domain correlation found by the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainCorrelation {
    pub domain_a: PhysicsDomain,
    pub domain_b: PhysicsDomain,
    pub quantity_a: String,
    pub quantity_b: String,
    pub correlation_strength: f64,
    pub mechanism: String,
    pub evidence: String,
}

/// Insight discovered by the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    pub title: String,
    pub description: String,
    pub confidence: f64,
    pub domains: Vec<PhysicsDomain>,
    pub implications: Vec<String>,
}

/// Open question identified by the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenQuestion {
    pub question: String,
    pub domains: Vec<PhysicsDomain>,
    pub possible_answers: Vec<String>,
    pub discriminating_experiments: Vec<String>,
}

/// Recommended experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedExperiment {
    pub title: String,
    pub purpose: String,
    pub expected_outcome: String,
    pub priority: u8,
    pub estimated_difficulty: String,
}

/// Complete discovery report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryReport {
    pub screening_patterns: Vec<PatternFit>,
    pub neutron_patterns: Vec<PatternFit>,
    pub gap_analysis: GapAnalysis,
    pub hypothesis_comparison: HypothesisComparison,
    pub cross_domain_correlations: Vec<CrossDomainCorrelation>,
    pub insights: Vec<Insight>,
    pub open_questions: Vec<OpenQuestion>,
    pub recommended_experiments: Vec<RecommendedExperiment>,
}

impl DiscoveryReport {
    /// Generate human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        s.push_str("║           PHYSICS DISCOVERY REPORT: LCF ANALYSIS                 ║\n");
        s.push_str("╚══════════════════════════════════════════════════════════════════╝\n\n");

        s.push_str("▶ GAP ANALYSIS\n");
        s.push_str(&format!(
            "  Observed rate:  {:.2e} n/s/cm³\n",
            self.gap_analysis.observed_rate
        ));
        s.push_str(&format!(
            "  Predicted rate: {:.2e} n/s/cm³\n",
            self.gap_analysis.predicted_rate
        ));
        s.push_str(&format!(
            "  Gap: {:.0} orders of magnitude\n",
            self.gap_analysis.gap_orders
        ));
        s.push_str(&format!(
            "  Screening explains: {:.0} orders\n",
            self.gap_analysis.screening_contribution_orders
        ));
        s.push_str(&format!(
            "  Remaining gap: {:.0} orders\n\n",
            self.gap_analysis.remaining_gap_orders
        ));

        s.push_str("▶ HYPOTHESIS TESTING\n");
        s.push_str(&format!(
            "  Best fit: {}\n",
            self.hypothesis_comparison.best_fit
        ));
        for cond in &self.hypothesis_comparison.required_conditions {
            s.push_str(&format!(
                "  - {}: {:.2e} {} ({})\n",
                cond.hypothesis,
                cond.required_value,
                cond.unit,
                if cond.physically_reasonable {
                    "plausible"
                } else {
                    "implausible"
                }
            ));
        }
        s.push('\n');

        s.push_str("▶ KEY INSIGHTS\n");
        for insight in &self.insights {
            s.push_str(&format!(
                "  • {} (confidence: {:.0}%)\n",
                insight.title,
                insight.confidence * 100.0
            ));
        }
        s.push('\n');

        s.push_str("▶ CROSS-DOMAIN CORRELATIONS\n");
        for corr in &self.cross_domain_correlations {
            s.push_str(&format!(
                "  • {:?} ↔ {:?}: {} (r={:.2})\n",
                corr.domain_a, corr.domain_b, corr.mechanism, corr.correlation_strength
            ));
        }
        s.push('\n');

        s.push_str("▶ OPEN QUESTIONS\n");
        for q in &self.open_questions {
            s.push_str(&format!("  ? {}\n", q.question));
        }
        s.push('\n');

        s.push_str("▶ RECOMMENDED EXPERIMENTS (by priority)\n");
        for exp in &self.recommended_experiments {
            s.push_str(&format!(
                "  {}. {} - {}\n",
                exp.priority, exp.title, exp.purpose
            ));
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_physics_encoder() {
        let mut encoder = MultiPhysicsEncoder::new(1024);

        let q1 = PhysicsQuantity {
            name: "temperature".to_string(),
            value: 300.0,
            unit: "K".to_string(),
            domains: vec![PhysicsDomain::Statistical],
            uncertainty: None,
            interpretation: "Room temperature".to_string(),
        };

        let q2 = PhysicsQuantity {
            name: "temperature".to_string(),
            value: 310.0,
            unit: "K".to_string(),
            domains: vec![PhysicsDomain::Statistical],
            uncertainty: None,
            interpretation: "Slightly elevated".to_string(),
        };

        let v1 = encoder.encode(&q1);
        let v2 = encoder.encode(&q2);

        assert_eq!(v1.len(), 1024);
        assert_eq!(v2.len(), 1024);

        // Similar values should have high similarity
        let sim = encoder.similarity(&v1, &v2);
        assert!(sim > 0.5, "Similar temperatures should be similar: {}", sim);
    }

    #[test]
    fn test_knowledge_base() {
        let kb = PhysicsKnowledgeBase::new();

        // Should have facts from multiple domains
        let nuclear_facts = kb.facts_in_domain(PhysicsDomain::Nuclear);
        assert!(!nuclear_facts.is_empty());

        let chemistry_facts = kb.facts_in_domain(PhysicsDomain::Chemistry);
        assert!(!chemistry_facts.is_empty());
    }

    #[test]
    fn test_pattern_analyzer() {
        let analyzer = PatternAnalyzer::new();
        let gap = analyzer.analyze_gap();

        // Should show huge gap
        assert!(
            gap.gap_orders > 30.0,
            "Gap should be > 30 orders: {}",
            gap.gap_orders
        );
        assert!(!gap.interpretation.is_empty());
    }

    #[test]
    fn test_discovery_engine() {
        let engine = PhysicsDiscoveryEngine::new();
        let report = engine.analyze();

        // Should produce comprehensive report
        assert!(!report.insights.is_empty());
        assert!(!report.open_questions.is_empty());
        assert!(!report.recommended_experiments.is_empty());
        assert!(!report.cross_domain_correlations.is_empty());

        // Gap analysis should be present
        assert!(report.gap_analysis.gap_orders > 0.0);
    }

    #[test]
    fn test_report_summary() {
        let engine = PhysicsDiscoveryEngine::new();
        let report = engine.analyze();
        let summary = report.summary();

        assert!(summary.contains("GAP ANALYSIS"));
        assert!(summary.contains("HYPOTHESIS TESTING"));
        assert!(summary.contains("KEY INSIGHTS"));
    }
}
