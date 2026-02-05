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
            QuarkFlavor::Up | QuarkFlavor::Charm | QuarkFlavor::Top => 2,      // +2/3
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
    pub fn encode_generation(&self, gen: u8) -> ContinuousHV {
        let shift = (gen as usize - 1) * 1000;
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
        let electron_neutrino_base = genesis.hv(LeptonFlavor::ElectronNeutrino.domain_label(), PHYSICS_DIM);
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
        // UP QUARK: +2/3 charge, 2.2 MeV, gen 1
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

        // Up and Down are gen 1, Charm is gen 2
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
}
