// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hadrons: Layer 2 - Composite Particles
//!
//! Hadrons are particles made of quarks bound by the strong force.
//! This module **constructs** hadrons from their constituent quarks
//! rather than defining them as independent vectors.
//!
//! ## Key Insight: Composition, Not Definition
//!
//! ```text
//! Proton  = Bundle(Up, Up, Down)     // uud, charge = +2/3 + 2/3 - 1/3 = +1
//! Neutron = Bundle(Up, Down, Down)   // udd, charge = +2/3 - 1/3 - 1/3 = 0
//! ```
//!
//! Because we COMPOSE hadrons, Symthaea understands:
//! - Protons and neutrons are "siblings" (share quarks)
//! - Beta decay: n → p + e⁻ + ν̄ₑ (neutron becomes proton)
//! - Isospin symmetry (up↔down exchange)
//!
//! ## Hadron Types
//!
//! ### Baryons (3 quarks)
//! - Proton (uud): stable, +1 charge
//! - Neutron (udd): β-decays with half-life 10.2 min
//! - Delta baryons (Δ++, Δ+, Δ0, Δ−): excited states
//!
//! ### Mesons (quark + antiquark)
//! - Pion (π+, π0, π−): mediates nuclear force
//! - Kaon (K): contains strange quark
//!
//! ## Nuclear Force Encoding
//!
//! The residual strong force between nucleons is encoded as:
//! ```text
//! Nuclear attraction ∝ similarity(proton, neutron)
//! ```
//!
//! This naturally emerges from shared quark content.

use super::standard_model::{PHYSICS_DIM, StandardModel};
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Baryon types (3-quark composite particles)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Baryon {
    Proton,        // uud
    Neutron,       // udd
    DeltaPlusPlus, // uuu
    DeltaPlus,     // uud (excited)
    DeltaZero,     // udd (excited)
    DeltaMinus,    // ddd
    Lambda,        // uds
    SigmaPlus,     // uus
    SigmaZero,     // uds
    SigmaMinus,    // dds
    XiZero,        // uss
    XiMinus,       // dss
    Omega,         // sss
}

impl Baryon {
    /// Electric charge
    pub fn charge(&self) -> i8 {
        match self {
            Baryon::DeltaPlusPlus => 2,
            Baryon::Proton | Baryon::DeltaPlus | Baryon::SigmaPlus => 1,
            Baryon::Neutron
            | Baryon::DeltaZero
            | Baryon::Lambda
            | Baryon::SigmaZero
            | Baryon::XiZero => 0,
            Baryon::DeltaMinus | Baryon::SigmaMinus | Baryon::XiMinus | Baryon::Omega => -1,
        }
    }

    /// Mass in MeV/c²
    pub fn mass_mev(&self) -> f32 {
        match self {
            Baryon::Proton => 938.272,
            Baryon::Neutron => 939.565,
            Baryon::DeltaPlusPlus | Baryon::DeltaPlus | Baryon::DeltaZero | Baryon::DeltaMinus => {
                1232.0
            }
            Baryon::Lambda => 1115.683,
            Baryon::SigmaPlus => 1189.37,
            Baryon::SigmaZero => 1192.642,
            Baryon::SigmaMinus => 1197.449,
            Baryon::XiZero => 1314.86,
            Baryon::XiMinus => 1321.71,
            Baryon::Omega => 1672.45,
        }
    }

    /// Strangeness quantum number
    pub fn strangeness(&self) -> i8 {
        match self {
            Baryon::Proton
            | Baryon::Neutron
            | Baryon::DeltaPlusPlus
            | Baryon::DeltaPlus
            | Baryon::DeltaZero
            | Baryon::DeltaMinus => 0,
            Baryon::Lambda | Baryon::SigmaPlus | Baryon::SigmaZero | Baryon::SigmaMinus => -1,
            Baryon::XiZero | Baryon::XiMinus => -2,
            Baryon::Omega => -3,
        }
    }
}

/// Meson types (quark-antiquark pairs)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Meson {
    PionPlus,  // u + d̄
    PionZero,  // (uū - dd̄)/√2
    PionMinus, // d + ū
    KaonPlus,  // u + s̄
    KaonZero,  // d + s̄
    KaonMinus, // s + ū
    Eta,       // (uū + dd̄ - 2ss̄)/√6
    EtaPrime,  // (uū + dd̄ + ss̄)/√3
}

impl Meson {
    /// Electric charge
    pub fn charge(&self) -> i8 {
        match self {
            Meson::PionPlus | Meson::KaonPlus => 1,
            Meson::PionZero | Meson::KaonZero | Meson::Eta | Meson::EtaPrime => 0,
            Meson::PionMinus | Meson::KaonMinus => -1,
        }
    }

    /// Mass in MeV/c²
    pub fn mass_mev(&self) -> f32 {
        match self {
            Meson::PionPlus | Meson::PionMinus => 139.570,
            Meson::PionZero => 134.977,
            Meson::KaonPlus | Meson::KaonMinus => 493.677,
            Meson::KaonZero => 497.611,
            Meson::Eta => 547.862,
            Meson::EtaPrime => 957.78,
        }
    }
}

/// Hadron system: baryons and mesons composed from quarks
#[derive(Debug, Clone)]
pub struct Hadrons {
    // Nucleons (most important for chemistry)
    pub proton: ContinuousHV,
    pub neutron: ContinuousHV,

    // Delta baryons
    pub delta_plus_plus: ContinuousHV,
    pub delta_plus: ContinuousHV,
    pub delta_zero: ContinuousHV,
    pub delta_minus: ContinuousHV,

    // Strange baryons
    pub lambda: ContinuousHV,
    pub sigma_plus: ContinuousHV,
    pub sigma_zero: ContinuousHV,
    pub sigma_minus: ContinuousHV,

    // Mesons
    pub pion_plus: ContinuousHV,
    pub pion_zero: ContinuousHV,
    pub pion_minus: ContinuousHV,
    pub kaon_plus: ContinuousHV,
    pub kaon_zero: ContinuousHV,

    // Binding energy concept (for nuclear physics)
    pub binding_energy: ContinuousHV,

    // Nuclear force mediator
    pub nuclear_force: ContinuousHV,
}

impl Hadrons {
    /// Construct all hadrons from the Standard Model
    ///
    /// This is the KEY function: hadrons are COMPOSED from quarks,
    /// not defined independently.
    pub fn from_model(model: &StandardModel, genesis: &GenesisSeed) -> Self {
        // Nucleons: the building blocks of all atoms
        // Proton = Bundle(Up, Up, Down)
        let proton = Self::compose_baryon(&[&model.up_quark, &model.up_quark, &model.down_quark]);

        // Neutron = Bundle(Up, Down, Down)
        let neutron =
            Self::compose_baryon(&[&model.up_quark, &model.down_quark, &model.down_quark]);

        // Delta baryons (excited states)
        let delta_plus_plus =
            Self::compose_baryon(&[&model.up_quark, &model.up_quark, &model.up_quark]);

        let delta_plus =
            Self::compose_baryon(&[&model.up_quark, &model.up_quark, &model.down_quark])
                .bind(&genesis.hv("hadron::delta_excited", PHYSICS_DIM));

        let delta_zero =
            Self::compose_baryon(&[&model.up_quark, &model.down_quark, &model.down_quark])
                .bind(&genesis.hv("hadron::delta_excited", PHYSICS_DIM));

        let delta_minus =
            Self::compose_baryon(&[&model.down_quark, &model.down_quark, &model.down_quark]);

        // Strange baryons
        let lambda =
            Self::compose_baryon(&[&model.up_quark, &model.down_quark, &model.strange_quark]);

        let sigma_plus =
            Self::compose_baryon(&[&model.up_quark, &model.up_quark, &model.strange_quark]);

        let sigma_zero =
            Self::compose_baryon(&[&model.up_quark, &model.down_quark, &model.strange_quark])
                .bind(&genesis.hv("hadron::sigma_state", PHYSICS_DIM));

        let sigma_minus =
            Self::compose_baryon(&[&model.down_quark, &model.down_quark, &model.strange_quark]);

        // Mesons (quark + antiquark)
        let anti_down = model.antiparticle(&model.down_quark);
        let anti_up = model.antiparticle(&model.up_quark);
        let anti_strange = model.antiparticle(&model.strange_quark);

        let pion_plus = Self::compose_meson(&model.up_quark, &anti_down);
        let pion_minus = Self::compose_meson(&model.down_quark, &anti_up);
        let pion_zero = ContinuousHV::bundle(&[
            &Self::compose_meson(&model.up_quark, &anti_up),
            &Self::compose_meson(&model.down_quark, &anti_down),
        ]);

        let kaon_plus = Self::compose_meson(&model.up_quark, &anti_strange);
        let kaon_zero = Self::compose_meson(&model.down_quark, &anti_strange);

        // Nuclear physics concepts
        let binding_energy = genesis.hv("nuclear::binding_energy", PHYSICS_DIM);
        let nuclear_force = genesis.hv("nuclear::force", PHYSICS_DIM);

        Self {
            proton,
            neutron,
            delta_plus_plus,
            delta_plus,
            delta_zero,
            delta_minus,
            lambda,
            sigma_plus,
            sigma_zero,
            sigma_minus,
            pion_plus,
            pion_zero,
            pion_minus,
            kaon_plus,
            kaon_zero,
            binding_energy,
            nuclear_force,
        }
    }

    /// Compose a baryon from three quarks
    ///
    /// Uses bundling to create superposition that preserves
    /// similarity to all constituent quarks.
    fn compose_baryon(quarks: &[&ContinuousHV; 3]) -> ContinuousHV {
        // Position-encode each quark to distinguish them
        let q1 = quarks[0].permute(0);
        let q2 = quarks[1].permute(1000);
        let q3 = quarks[2].permute(2000);

        // Bundle to create composite
        ContinuousHV::bundle(&[&q1, &q2, &q3])
    }

    /// Compose a meson from quark-antiquark pair
    fn compose_meson(quark: &ContinuousHV, antiquark: &ContinuousHV) -> ContinuousHV {
        // Bind quark and antiquark (creates association)
        // Then normalize
        quark.bind(antiquark).normalize()
    }

    /// Get nucleon similarity (proton-neutron)
    ///
    /// This measures the "isospin" relationship between nucleons.
    pub fn nucleon_similarity(&self) -> f32 {
        self.proton.similarity(&self.neutron)
    }

    /// Simulate beta decay: n → p + e⁻ + ν̄ₑ
    ///
    /// Returns the products of neutron decay.
    pub fn beta_decay(&self, model: &StandardModel) -> (ContinuousHV, ContinuousHV, ContinuousHV) {
        // Neutron decays to proton
        let proton = self.proton.clone();

        // Electron emitted
        let electron = model.electron.clone();

        // Antineutrino emitted
        let antineutrino = model.antiparticle(&model.electron_neutrino);

        (proton, electron, antineutrino)
    }

    /// Compute nuclear binding for a given nucleus
    ///
    /// Binding = base + (proton-neutron) interactions
    pub fn compute_binding(&self, protons: usize, neutrons: usize) -> ContinuousHV {
        // Nuclear binding is related to the number of nucleon pairs
        let total = protons + neutrons;
        let pair_factor = (protons * neutrons) as f32 / (total.max(1) as f32);

        // Bundle proton and neutron contributions
        let p_contrib = self.proton.scale(protons as f32);
        let n_contrib = self.neutron.scale(neutrons as f32);
        let nucleon_bundle = ContinuousHV::bundle(&[&p_contrib, &n_contrib]);

        // Bind with nuclear force and scale by pairing
        nucleon_bundle
            .bind(&self.nuclear_force)
            .bind(&self.binding_energy)
            .scale(pair_factor)
    }

    /// Get baryon by type
    pub fn baryon(&self, baryon: Baryon) -> &ContinuousHV {
        match baryon {
            Baryon::Proton => &self.proton,
            Baryon::Neutron => &self.neutron,
            Baryon::DeltaPlusPlus => &self.delta_plus_plus,
            Baryon::DeltaPlus => &self.delta_plus,
            Baryon::DeltaZero => &self.delta_zero,
            Baryon::DeltaMinus => &self.delta_minus,
            Baryon::Lambda => &self.lambda,
            Baryon::SigmaPlus => &self.sigma_plus,
            Baryon::SigmaZero => &self.sigma_zero,
            Baryon::SigmaMinus => &self.sigma_minus,
            _ => &self.proton, // Default for unimplemented
        }
    }

    /// Get meson by type
    pub fn meson(&self, meson: Meson) -> &ContinuousHV {
        match meson {
            Meson::PionPlus => &self.pion_plus,
            Meson::PionZero => &self.pion_zero,
            Meson::PionMinus => &self.pion_minus,
            Meson::KaonPlus => &self.kaon_plus,
            Meson::KaonZero => &self.kaon_zero,
            _ => &self.pion_zero, // Default for unimplemented
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadron_construction() {
        let genesis = GenesisSeed::from_phrase("hadron test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        assert_eq!(hadrons.proton.dim(), PHYSICS_DIM);
        assert_eq!(hadrons.neutron.dim(), PHYSICS_DIM);
    }

    #[test]
    fn test_nucleon_sibling_relationship() {
        let genesis = GenesisSeed::from_phrase("nucleon siblings");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        // Proton and neutron share quark content
        let sim = hadrons.nucleon_similarity();

        // They should have significant similarity (share 2 quarks)
        assert!(sim > 0.3, "Proton and neutron should be related: {}", sim);
    }

    #[test]
    fn test_delta_relationship() {
        let genesis = GenesisSeed::from_phrase("delta test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        // Delta++ is uuu, proton is uud
        // They should share structure (2 up quarks)
        let sim = hadrons.proton.similarity(&hadrons.delta_plus_plus);

        assert!(
            sim > 0.2,
            "Delta++ and proton should share structure: {}",
            sim
        );
    }

    #[test]
    fn test_strange_baryon_dissimilarity() {
        let genesis = GenesisSeed::from_phrase("strangeness test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        // Lambda has strange quark, proton doesn't
        // They should be less similar than proton-neutron
        let lambda_proton = hadrons.proton.similarity(&hadrons.lambda);
        let proton_neutron = hadrons.nucleon_similarity();

        assert!(
            lambda_proton < proton_neutron,
            "Strange baryons should be less similar to proton: lambda={}, nucleon={}",
            lambda_proton,
            proton_neutron
        );
    }

    #[test]
    fn test_meson_composition() {
        let genesis = GenesisSeed::from_phrase("meson test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        // Pion+ and pion- are antiparticles
        let sim = hadrons.pion_plus.similarity(&hadrons.pion_minus);

        // They should be dissimilar (different quark content)
        assert!(sim < 0.5, "Pion+ and pion- should be distinct: {}", sim);
    }

    #[test]
    fn test_beta_decay_products() {
        let genesis = GenesisSeed::from_phrase("decay test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        let (proton, electron, antineutrino) = hadrons.beta_decay(&model);

        // Decay products should be valid particles
        assert_eq!(proton.dim(), PHYSICS_DIM);
        assert_eq!(electron.dim(), PHYSICS_DIM);
        assert_eq!(antineutrino.dim(), PHYSICS_DIM);

        // Proton from decay should match hadrons.proton
        assert!(
            proton.similarity(&hadrons.proton) > 0.99,
            "Decay product should be proton"
        );
    }

    #[test]
    fn test_deterministic_hadrons() {
        let genesis = GenesisSeed::from_phrase("determinism");
        let model = StandardModel::from_genesis(&genesis);

        let h1 = Hadrons::from_model(&model, &genesis);
        let h2 = Hadrons::from_model(&model, &genesis);

        assert!(
            h1.proton.similarity(&h2.proton) > 0.9999,
            "Hadrons should be deterministic"
        );
    }
}
