// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Standard Model: The Axioms of Reality
//!
//! Layer 1 of the Physics Hierarchy: fundamental particles that cannot be
//! decomposed further. These are the "Genesis Atoms" from which all matter
//! is constructed.
//!
//! ## Particle Families
//!
//! ### Quarks (6 flavors, 3 generations)
//! - Generation 1: Up (+2/3), Down (-1/3)
//! - Generation 2: Charm (+2/3), Strange (-1/3)
//! - Generation 3: Top (+2/3), Bottom (-1/3)
//!
//! ### Leptons (6 flavors, 3 generations)
//! - Generation 1: Electron (-1), Electron Neutrino (0)
//! - Generation 2: Muon (-1), Muon Neutrino (0)
//! - Generation 3: Tau (-1), Tau Neutrino (0)
//!
//! ### Gauge Bosons (force carriers)
//! - Photon: Electromagnetic force
//! - Gluon: Strong force (8 color combinations)
//! - W+, W-, Z: Weak force
//!
//! ### Higgs Boson
//! - Gives mass to particles
//!
//! ## Design Principle
//!
//! Each fundamental particle is a **deterministic** hypervector derived from
//! the Genesis Seed. The domain label encodes the particle identity:
//! ```text
//! UP_QUARK = genesis.hv("standard_model::quark::up", dim)
//! ```
//!
//! Properties like charge and mass are **bound** to the particle vector,
//! creating rich representations that encode physics relationships.

use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use crate::physics::constants;
use serde::{Deserialize, Serialize};

/// Dimension for physics vectors (matches HDC standard)
pub const PHYSICS_DIM: usize = 16_384;

/// Quark flavors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuarkFlavor {
    Up,
    Down,
    Charm,
    Strange,
    Top,
    Bottom,
}

impl QuarkFlavor {
    /// Electric charge in units of e/3
    pub fn charge_thirds(&self) -> i8 {
        match self {
            QuarkFlavor::Up | QuarkFlavor::Charm | QuarkFlavor::Top => 2, // +2/3
            QuarkFlavor::Down | QuarkFlavor::Strange | QuarkFlavor::Bottom => -1, // -1/3
        }
    }

    /// Mass in MeV/c² (approximate)
    pub fn mass_mev(&self) -> f32 {
        match self {
            QuarkFlavor::Up => 2.2,
            QuarkFlavor::Down => 4.7,
            QuarkFlavor::Charm => 1_280.0,
            QuarkFlavor::Strange => 96.0,
            QuarkFlavor::Top => 173_100.0,
            QuarkFlavor::Bottom => 4_180.0,
        }
    }

    /// Generation (1, 2, or 3)
    pub fn generation(&self) -> u8 {
        match self {
            QuarkFlavor::Up | QuarkFlavor::Down => 1,
            QuarkFlavor::Charm | QuarkFlavor::Strange => 2,
            QuarkFlavor::Top | QuarkFlavor::Bottom => 3,
        }
    }

    /// Domain label for genesis derivation
    fn domain_label(&self) -> &'static str {
        match self {
            QuarkFlavor::Up => "standard_model::quark::up",
            QuarkFlavor::Down => "standard_model::quark::down",
            QuarkFlavor::Charm => "standard_model::quark::charm",
            QuarkFlavor::Strange => "standard_model::quark::strange",
            QuarkFlavor::Top => "standard_model::quark::top",
            QuarkFlavor::Bottom => "standard_model::quark::bottom",
        }
    }
}

/// Lepton flavors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LeptonFlavor {
    Electron,
    ElectronNeutrino,
    Muon,
    MuonNeutrino,
    Tau,
    TauNeutrino,
}

impl LeptonFlavor {
    /// Electric charge (-1, 0)
    pub fn charge(&self) -> i8 {
        match self {
            LeptonFlavor::Electron | LeptonFlavor::Muon | LeptonFlavor::Tau => -1,
            _ => 0,
        }
    }

    /// Mass in MeV/c² (neutrino masses are upper bounds)
    pub fn mass_mev(&self) -> f32 {
        match self {
            LeptonFlavor::Electron => 0.511,
            LeptonFlavor::ElectronNeutrino => 0.000001, // < 1 eV
            LeptonFlavor::Muon => 105.66,
            LeptonFlavor::MuonNeutrino => 0.00017,
            LeptonFlavor::Tau => 1776.86,
            LeptonFlavor::TauNeutrino => 0.0182,
        }
    }

    /// Generation (1, 2, or 3)
    pub fn generation(&self) -> u8 {
        match self {
            LeptonFlavor::Electron | LeptonFlavor::ElectronNeutrino => 1,
            LeptonFlavor::Muon | LeptonFlavor::MuonNeutrino => 2,
            LeptonFlavor::Tau | LeptonFlavor::TauNeutrino => 3,
        }
    }

    /// Domain label for genesis derivation
    fn domain_label(&self) -> &'static str {
        match self {
            LeptonFlavor::Electron => "standard_model::lepton::electron",
            LeptonFlavor::ElectronNeutrino => "standard_model::lepton::electron_neutrino",
            LeptonFlavor::Muon => "standard_model::lepton::muon",
            LeptonFlavor::MuonNeutrino => "standard_model::lepton::muon_neutrino",
            LeptonFlavor::Tau => "standard_model::lepton::tau",
            LeptonFlavor::TauNeutrino => "standard_model::lepton::tau_neutrino",
        }
    }
}

/// Gauge bosons (force carriers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GaugeBoson {
    /// Electromagnetic force carrier
    Photon,
    /// Strong force carrier (8 gluons, but we use one representative)
    Gluon,
    /// Weak force (positive charge)
    WPlus,
    /// Weak force (negative charge)
    WMinus,
    /// Weak force (neutral)
    Z,
    /// Graviton (hypothetical, for completeness)
    Graviton,
}

impl GaugeBoson {
    /// Mass in GeV/c² (0 for massless bosons)
    pub fn mass_gev(&self) -> f32 {
        match self {
            GaugeBoson::Photon => 0.0,
            GaugeBoson::Gluon => 0.0,
            GaugeBoson::WPlus | GaugeBoson::WMinus => 80.379,
            GaugeBoson::Z => 91.1876,
            GaugeBoson::Graviton => 0.0,
        }
    }

    /// Spin (all gauge bosons have spin 1)
    pub fn spin(&self) -> u8 {
        match self {
            GaugeBoson::Graviton => 2,
            _ => 1,
        }
    }

    /// Domain label for genesis derivation
    fn domain_label(&self) -> &'static str {
        match self {
            GaugeBoson::Photon => "standard_model::boson::photon",
            GaugeBoson::Gluon => "standard_model::boson::gluon",
            GaugeBoson::WPlus => "standard_model::boson::w_plus",
            GaugeBoson::WMinus => "standard_model::boson::w_minus",
            GaugeBoson::Z => "standard_model::boson::z",
            GaugeBoson::Graviton => "standard_model::boson::graviton",
        }
    }
}

/// Property vectors for encoding physical attributes
#[derive(Debug, Clone)]
pub struct PropertyVectors {
    /// Charge property vector (for binding with particles)
    pub charge: ContinuousHV,
    /// Mass property vector
    pub mass: ContinuousHV,
    /// Spin property vector
    pub spin: ContinuousHV,
    /// Color charge (for quarks/gluons)
    pub color: ContinuousHV,
    /// Generation (for fermions)
    pub generation: ContinuousHV,
    /// Positive/negative polarity
    pub positive: ContinuousHV,
    pub negative: ContinuousHV,
    pub neutral: ContinuousHV,
}

impl PropertyVectors {
    /// Create property vectors from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            charge: genesis.hv("property::charge", PHYSICS_DIM),
            mass: genesis.hv("property::mass", PHYSICS_DIM),
            spin: genesis.hv("property::spin", PHYSICS_DIM),
            color: genesis.hv("property::color", PHYSICS_DIM),
            generation: genesis.hv("property::generation", PHYSICS_DIM),
            positive: genesis.hv("property::positive", PHYSICS_DIM),
            negative: genesis.hv("property::negative", PHYSICS_DIM),
            neutral: genesis.hv("property::neutral", PHYSICS_DIM),
        }
    }

    /// Encode a charge value as a vector
    pub fn encode_charge(&self, charge_thirds: i8) -> ContinuousHV {
        let magnitude = (charge_thirds.abs() as f32) / 3.0;
        let polarity = if charge_thirds > 0 {
            &self.positive
        } else if charge_thirds < 0 {
            &self.negative
        } else {
            &self.neutral
        };

        // Bind charge property with polarity, scaled by magnitude
        self.charge.bind(polarity).scale(magnitude.max(0.01))
    }

    /// Encode a mass value (log scale for huge range)
    pub fn encode_mass(&self, mass_mev: f32) -> ContinuousHV {
        // Use log scale: electron = 0.511 MeV, top quark = 173,100 MeV
        let log_mass = (mass_mev.max(0.000001)).ln();
        let normalized = (log_mass + 10.0) / 20.0; // Rough normalization

        self.mass.scale(normalized.clamp(0.01, 1.0))
    }

    /// Encode generation (1, 2, 3)
    pub fn encode_generation(&self, r#gen: u8) -> ContinuousHV {
        let shift = (r#gen as usize - 1) * 1000;
        self.generation.permute(shift)
    }
}

/// The Standard Model of Particle Physics
///
/// Contains hypervector representations of all fundamental particles,
/// deterministically derived from a Genesis Seed.
#[derive(Debug, Clone)]
pub struct StandardModel {
    // Quarks
    pub up_quark: ContinuousHV,
    pub down_quark: ContinuousHV,
    pub charm_quark: ContinuousHV,
    pub strange_quark: ContinuousHV,
    pub top_quark: ContinuousHV,
    pub bottom_quark: ContinuousHV,

    // Leptons
    pub electron: ContinuousHV,
    pub electron_neutrino: ContinuousHV,
    pub muon: ContinuousHV,
    pub muon_neutrino: ContinuousHV,
    pub tau: ContinuousHV,
    pub tau_neutrino: ContinuousHV,

    // Gauge Bosons
    pub photon: ContinuousHV,
    pub gluon: ContinuousHV,
    pub w_plus: ContinuousHV,
    pub w_minus: ContinuousHV,
    pub z_boson: ContinuousHV,
    pub graviton: ContinuousHV,

    // Higgs
    pub higgs: ContinuousHV,

    // Property vectors (for encoding attributes)
    pub properties: PropertyVectors,

    // Antiparticle transform (bind with this to get antiparticle)
    pub antimatter: ContinuousHV,
}

impl StandardModel {
    /// Create the Standard Model from a Genesis Seed
    ///
    /// All particles are deterministically derived from the seed phrase.
    /// Same phrase → identical physics.
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        let properties = PropertyVectors::from_genesis(genesis);

        // Create base particle vectors
        let up_quark_base = genesis.hv(QuarkFlavor::Up.domain_label(), PHYSICS_DIM);
        let down_quark_base = genesis.hv(QuarkFlavor::Down.domain_label(), PHYSICS_DIM);
        let charm_quark_base = genesis.hv(QuarkFlavor::Charm.domain_label(), PHYSICS_DIM);
        let strange_quark_base = genesis.hv(QuarkFlavor::Strange.domain_label(), PHYSICS_DIM);
        let top_quark_base = genesis.hv(QuarkFlavor::Top.domain_label(), PHYSICS_DIM);
        let bottom_quark_base = genesis.hv(QuarkFlavor::Bottom.domain_label(), PHYSICS_DIM);

        let electron_base = genesis.hv(LeptonFlavor::Electron.domain_label(), PHYSICS_DIM);
        let electron_neutrino_base =
            genesis.hv(LeptonFlavor::ElectronNeutrino.domain_label(), PHYSICS_DIM);
        let muon_base = genesis.hv(LeptonFlavor::Muon.domain_label(), PHYSICS_DIM);
        let muon_neutrino_base = genesis.hv(LeptonFlavor::MuonNeutrino.domain_label(), PHYSICS_DIM);
        let tau_base = genesis.hv(LeptonFlavor::Tau.domain_label(), PHYSICS_DIM);
        let tau_neutrino_base = genesis.hv(LeptonFlavor::TauNeutrino.domain_label(), PHYSICS_DIM);

        let photon_base = genesis.hv(GaugeBoson::Photon.domain_label(), PHYSICS_DIM);
        let gluon_base = genesis.hv(GaugeBoson::Gluon.domain_label(), PHYSICS_DIM);
        let w_plus_base = genesis.hv(GaugeBoson::WPlus.domain_label(), PHYSICS_DIM);
        let w_minus_base = genesis.hv(GaugeBoson::WMinus.domain_label(), PHYSICS_DIM);
        let z_boson_base = genesis.hv(GaugeBoson::Z.domain_label(), PHYSICS_DIM);
        let graviton_base = genesis.hv(GaugeBoson::Graviton.domain_label(), PHYSICS_DIM);

        let higgs_base = genesis.hv("standard_model::higgs", PHYSICS_DIM);
        let antimatter = genesis.hv("standard_model::antimatter", PHYSICS_DIM);

        // Enrich particles with their properties
        // UP QUARK: +2/3 charge, 2.2 MeV, r#gen 1
        let up_quark = Self::enrich_particle(
            &up_quark_base,
            &properties,
            QuarkFlavor::Up.charge_thirds(),
            QuarkFlavor::Up.mass_mev(),
            QuarkFlavor::Up.generation(),
        );

        let down_quark = Self::enrich_particle(
            &down_quark_base,
            &properties,
            QuarkFlavor::Down.charge_thirds(),
            QuarkFlavor::Down.mass_mev(),
            QuarkFlavor::Down.generation(),
        );

        let charm_quark = Self::enrich_particle(
            &charm_quark_base,
            &properties,
            QuarkFlavor::Charm.charge_thirds(),
            QuarkFlavor::Charm.mass_mev(),
            QuarkFlavor::Charm.generation(),
        );

        let strange_quark = Self::enrich_particle(
            &strange_quark_base,
            &properties,
            QuarkFlavor::Strange.charge_thirds(),
            QuarkFlavor::Strange.mass_mev(),
            QuarkFlavor::Strange.generation(),
        );

        let top_quark = Self::enrich_particle(
            &top_quark_base,
            &properties,
            QuarkFlavor::Top.charge_thirds(),
            QuarkFlavor::Top.mass_mev(),
            QuarkFlavor::Top.generation(),
        );

        let bottom_quark = Self::enrich_particle(
            &bottom_quark_base,
            &properties,
            QuarkFlavor::Bottom.charge_thirds(),
            QuarkFlavor::Bottom.mass_mev(),
            QuarkFlavor::Bottom.generation(),
        );

        // Leptons
        let electron = Self::enrich_particle(
            &electron_base,
            &properties,
            LeptonFlavor::Electron.charge() * 3, // Convert to thirds
            LeptonFlavor::Electron.mass_mev(),
            LeptonFlavor::Electron.generation(),
        );

        let electron_neutrino = Self::enrich_particle(
            &electron_neutrino_base,
            &properties,
            0,
            LeptonFlavor::ElectronNeutrino.mass_mev(),
            LeptonFlavor::ElectronNeutrino.generation(),
        );

        let muon = Self::enrich_particle(
            &muon_base,
            &properties,
            LeptonFlavor::Muon.charge() * 3,
            LeptonFlavor::Muon.mass_mev(),
            LeptonFlavor::Muon.generation(),
        );

        let muon_neutrino = Self::enrich_particle(
            &muon_neutrino_base,
            &properties,
            0,
            LeptonFlavor::MuonNeutrino.mass_mev(),
            LeptonFlavor::MuonNeutrino.generation(),
        );

        let tau = Self::enrich_particle(
            &tau_base,
            &properties,
            LeptonFlavor::Tau.charge() * 3,
            LeptonFlavor::Tau.mass_mev(),
            LeptonFlavor::Tau.generation(),
        );

        let tau_neutrino = Self::enrich_particle(
            &tau_neutrino_base,
            &properties,
            0,
            LeptonFlavor::TauNeutrino.mass_mev(),
            LeptonFlavor::TauNeutrino.generation(),
        );

        Self {
            up_quark,
            down_quark,
            charm_quark,
            strange_quark,
            top_quark,
            bottom_quark,
            electron,
            electron_neutrino,
            muon,
            muon_neutrino,
            tau,
            tau_neutrino,
            photon: photon_base,
            gluon: gluon_base,
            w_plus: w_plus_base,
            w_minus: w_minus_base,
            z_boson: z_boson_base,
            graviton: graviton_base,
            higgs: higgs_base,
            properties,
            antimatter,
        }
    }

    /// Enrich a base particle vector with its physical properties
    fn enrich_particle(
        base: &ContinuousHV,
        properties: &PropertyVectors,
        charge_thirds: i8,
        mass_mev: f32,
        generation: u8,
    ) -> ContinuousHV {
        let charge_hv = properties.encode_charge(charge_thirds);
        let mass_hv = properties.encode_mass(mass_mev);
        let gen_hv = properties.encode_generation(generation);

        // Bundle: particle + charge + mass + generation
        // This creates a rich representation encoding all properties
        ContinuousHV::bundle(&[base, &charge_hv, &mass_hv, &gen_hv])
    }

    /// Get a quark by flavor
    pub fn quark(&self, flavor: QuarkFlavor) -> &ContinuousHV {
        match flavor {
            QuarkFlavor::Up => &self.up_quark,
            QuarkFlavor::Down => &self.down_quark,
            QuarkFlavor::Charm => &self.charm_quark,
            QuarkFlavor::Strange => &self.strange_quark,
            QuarkFlavor::Top => &self.top_quark,
            QuarkFlavor::Bottom => &self.bottom_quark,
        }
    }

    /// Get a lepton by flavor
    pub fn lepton(&self, flavor: LeptonFlavor) -> &ContinuousHV {
        match flavor {
            LeptonFlavor::Electron => &self.electron,
            LeptonFlavor::ElectronNeutrino => &self.electron_neutrino,
            LeptonFlavor::Muon => &self.muon,
            LeptonFlavor::MuonNeutrino => &self.muon_neutrino,
            LeptonFlavor::Tau => &self.tau,
            LeptonFlavor::TauNeutrino => &self.tau_neutrino,
        }
    }

    /// Get a gauge boson
    pub fn boson(&self, boson: GaugeBoson) -> &ContinuousHV {
        match boson {
            GaugeBoson::Photon => &self.photon,
            GaugeBoson::Gluon => &self.gluon,
            GaugeBoson::WPlus => &self.w_plus,
            GaugeBoson::WMinus => &self.w_minus,
            GaugeBoson::Z => &self.z_boson,
            GaugeBoson::Graviton => &self.graviton,
        }
    }

    /// Create an antiparticle (permute by half-dimension)
    ///
    /// Uses cyclic permutation by PHYSICS_DIM/2 to create an orthogonal vector.
    /// This is self-inverse: permute(permute(x, N/2), N/2) = x because N/2 + N/2 = N
    /// wraps around to identity.
    pub fn antiparticle(&self, particle: &ContinuousHV) -> ContinuousHV {
        // Permute by half dimension - self-inverse operation
        particle.permute(PHYSICS_DIM / 2)
    }

    /// Check if two particles are from the same generation
    pub fn same_generation(&self, a: &ContinuousHV, b: &ContinuousHV) -> f32 {
        // Extract generation component by unbinding
        let gen_a = a.bind(&self.properties.generation.inverse());
        let gen_b = b.bind(&self.properties.generation.inverse());
        gen_a.similarity(&gen_b)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RUNNING COUPLING CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// 1-loop QCD running coupling constant α_s(Q²).
///
/// Uses the 1-loop beta function:
///   α_s(Q²) = α_s(M_Z²) / (1 + b₀·α_s(M_Z²)·ln(Q²/M_Z²)/(2π))
///
/// where b₀ = (11·N_c - 2·n_f)/3 with N_c=3 colors and n_f active flavors.
///
/// # Arguments
/// * `q2_gev2` — momentum transfer squared in GeV² (must be > 0)
///
/// # Returns
/// α_s at the given scale. Returns `None` if Q² is at or below the Landau pole.
pub fn alpha_s_running(q2_gev2: f64) -> Option<f64> {
    if q2_gev2 <= 0.0 {
        return None;
    }
    let m_z_gev = constants::M_Z_BOSON / 1000.0; // Convert MeV to GeV
    let m_z2 = m_z_gev * m_z_gev;
    let alpha_s_mz = constants::ALPHA_S_MZ;

    // Determine active flavors at this scale (threshold crossings)
    let n_f = if q2_gev2 < 1.3_f64.powi(2) {
        3.0 // u, d, s
    } else if q2_gev2 < 4.18_f64.powi(2) {
        4.0 // + c
    } else if q2_gev2 < 173.1_f64.powi(2) {
        5.0 // + b
    } else {
        6.0 // + t
    };

    let b0 = (11.0 * 3.0 - 2.0 * n_f) / 3.0;
    let log_ratio = (q2_gev2 / m_z2).ln();
    let denominator = 1.0 + b0 * alpha_s_mz * log_ratio / (2.0 * std::f64::consts::PI);

    if denominator <= 0.0 {
        return None; // Landau pole
    }

    Some(alpha_s_mz / denominator)
}

/// 1-loop QED running coupling constant α_em(Q²).
///
/// Uses the 1-loop running:
///   α(Q²) = α(0) / (1 - Δα(Q²))
///
/// Simplified: only lepton contributions at low scale, hadronic at higher scale.
///
/// # Arguments
/// * `q2_gev2` — momentum transfer squared in GeV²
pub fn alpha_em_running(q2_gev2: f64) -> f64 {
    let alpha_0 = constants::ALPHA;
    if q2_gev2 <= 0.0 {
        return alpha_0;
    }

    // Lepton loop contribution: Δα_lep ≈ (α/3π)·Σ ln(Q²/m_l²) for m_l² < Q²
    let m_e2 = (constants::M_ELECTRON_MEV / 1000.0).powi(2); // GeV²
    let m_mu2 = (105.66 / 1000.0_f64).powi(2);
    let m_tau2 = (1776.86 / 1000.0_f64).powi(2);

    let prefactor = alpha_0 / (3.0 * std::f64::consts::PI);
    let mut delta_alpha = 0.0;

    if q2_gev2 > m_e2 {
        delta_alpha += prefactor * (q2_gev2 / m_e2).ln();
    }
    if q2_gev2 > m_mu2 {
        delta_alpha += prefactor * (q2_gev2 / m_mu2).ln();
    }
    if q2_gev2 > m_tau2 {
        delta_alpha += prefactor * (q2_gev2 / m_tau2).ln();
    }

    // Hadronic contribution (~0.02762 at M_Z, interpolated)
    let m_z_gev2 = (constants::M_Z_BOSON / 1000.0).powi(2);
    let hadronic = 0.02762 * (q2_gev2 / m_z_gev2).min(1.0);
    delta_alpha += hadronic;

    alpha_0 / (1.0 - delta_alpha)
}

// ═══════════════════════════════════════════════════════════════════════════════
// CKM & PMNS MIXING MATRICES
// ═══════════════════════════════════════════════════════════════════════════════

/// CKM quark mixing matrix magnitudes |V_ij|.
///
/// Rows: (u, c, t), Columns: (d, s, b). PDG 2024 global fit.
pub const CKM: [[f64; 3]; 3] = [
    [0.97435, 0.22500, 0.00369], // u → d, s, b
    [0.22486, 0.97349, 0.04182], // c → d, s, b
    [0.00857, 0.04110, 0.99912], // t → d, s, b
];

/// PMNS neutrino mixing matrix magnitudes |U_αi|.
///
/// Rows: (e, μ, τ), Columns: (ν₁, ν₂, ν₃). NuFIT 5.2 (2023).
pub const PMNS: [[f64; 3]; 3] = [
    [0.821, 0.550, 0.149], // e  → ν₁, ν₂, ν₃
    [0.352, 0.579, 0.737], // μ  → ν₁, ν₂, ν₃
    [0.449, 0.602, 0.660], // τ  → ν₁, ν₂, ν₃
];

/// Check approximate unitarity of a 3×3 mixing matrix.
/// Returns the maximum deviation of any row or column norm from 1.0.
pub fn mixing_matrix_unitarity_deviation(matrix: &[[f64; 3]; 3]) -> f64 {
    let mut max_dev = 0.0_f64;

    for row in matrix.iter() {
        let norm_sq: f64 = row.iter().map(|x| x * x).sum();
        max_dev = max_dev.max((norm_sq - 1.0).abs());
    }

    for col in 0..3 {
        let norm_sq: f64 = matrix.iter().map(|row| row[col] * row[col]).sum();
        max_dev = max_dev.max((norm_sq - 1.0).abs());
    }

    max_dev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_model_creation() {
        let genesis = GenesisSeed::from_phrase("test universe");
        let model = StandardModel::from_genesis(&genesis);

        // All particle vectors should have correct dimension
        assert_eq!(model.up_quark.dim(), PHYSICS_DIM);
        assert_eq!(model.electron.dim(), PHYSICS_DIM);
        assert_eq!(model.photon.dim(), PHYSICS_DIM);
    }

    #[test]
    fn test_deterministic_creation() {
        let genesis = GenesisSeed::from_phrase("E=mc²");

        let model1 = StandardModel::from_genesis(&genesis);
        let model2 = StandardModel::from_genesis(&genesis);

        // Same seed → identical particles
        assert!(
            model1.up_quark.similarity(&model2.up_quark) > 0.9999,
            "Same genesis should produce identical particles"
        );
    }

    #[test]
    fn test_different_universes() {
        let genesis1 = GenesisSeed::from_phrase("Universe A");
        let genesis2 = GenesisSeed::from_phrase("Universe B");

        let model1 = StandardModel::from_genesis(&genesis1);
        let model2 = StandardModel::from_genesis(&genesis2);

        // Different seeds → different particles
        let sim = model1.up_quark.similarity(&model2.up_quark);
        assert!(
            sim.abs() < 0.1,
            "Different genesis should produce orthogonal particles: {}",
            sim
        );
    }

    #[test]
    fn test_quark_relationships() {
        let genesis = GenesisSeed::from_phrase("quark test");
        let model = StandardModel::from_genesis(&genesis);

        // Same generation quarks should be more similar than cross-generation
        let up_down_sim = model.up_quark.similarity(&model.down_quark);
        let up_charm_sim = model.up_quark.similarity(&model.charm_quark);

        // Up and Down are r#gen 1, Charm is r#gen 2
        // They should have some similarity due to shared structure
        assert!(
            up_down_sim > up_charm_sim * 0.8,
            "Same-generation quarks should have related structure"
        );
    }

    #[test]
    fn test_antiparticle_transform() {
        let genesis = GenesisSeed::from_phrase("antimatter test");
        let model = StandardModel::from_genesis(&genesis);

        let positron = model.antiparticle(&model.electron);

        // Antiparticle should be different from particle
        let sim = model.electron.similarity(&positron);
        assert!(
            sim.abs() < 0.3,
            "Antiparticle should be dissimilar to particle: {}",
            sim
        );

        // Double transformation should return near-original (binding is self-inverse)
        let double_transform = model.antiparticle(&positron);
        let recovery_sim = model.electron.similarity(&double_transform);
        assert!(
            recovery_sim > 0.9,
            "Double antimatter transform should recover original: {}",
            recovery_sim
        );
    }

    #[test]
    fn test_alpha_s_at_m_z() {
        // At Q² = M_Z², α_s should recover the input value
        let m_z_gev = constants::M_Z_BOSON / 1000.0;
        let alpha = alpha_s_running(m_z_gev * m_z_gev).unwrap();
        assert!(
            (alpha - constants::ALPHA_S_MZ).abs() < 1e-6,
            "α_s(M_Z) = {}, expected {}",
            alpha,
            constants::ALPHA_S_MZ
        );
    }

    #[test]
    fn test_asymptotic_freedom() {
        // α_s should decrease with increasing Q² (asymptotic freedom)
        let alpha_low = alpha_s_running(100.0).unwrap(); // Q = 10 GeV
        let alpha_high = alpha_s_running(10000.0).unwrap(); // Q = 100 GeV
        assert!(
            alpha_low > alpha_high,
            "Asymptotic freedom violated: α_s(100)={} should be > α_s(10000)={}",
            alpha_low,
            alpha_high
        );
    }

    #[test]
    fn test_ckm_unitarity() {
        let dev = mixing_matrix_unitarity_deviation(&CKM);
        assert!(dev < 0.01, "CKM unitarity deviation too large: {}", dev);
    }

    #[test]
    fn test_pmns_unitarity() {
        let dev = mixing_matrix_unitarity_deviation(&PMNS);
        assert!(dev < 0.05, "PMNS unitarity deviation too large: {}", dev);
    }
}
