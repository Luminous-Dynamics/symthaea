// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Antimatter Encoding
//!
//! This module implements antimatter particles using CPT symmetry transforms.
//!
//! ## CPT Conjugation
//!
//! The CPT theorem states that all physical laws are invariant under the combined
//! transformation of Charge conjugation (C), Parity inversion (P), and Time reversal (T).
//!
//! In HDC, we implement this as a half-dimension permutation, which is self-inverse:
//! ```text
//! antiparticle = particle.permute(DIM / 2)
//! particle = antiparticle.permute(DIM / 2)  // Self-inverse!
//! ```
//!
//! ## Antihadron Composition
//!
//! Antihadrons are composed from antiquarks:
//! - Antiproton (p̄) = ū + ū + d̄
//! - Antineutron (n̄) = ū + d̄ + d̄
//!
//! ## Annihilation
//!
//! When particle meets antiparticle, they annihilate producing energy (typically photons):
//! - e⁺ + e⁻ → 2γ (1.022 MeV total)

use super::hadrons::Hadrons;
use super::standard_model::{PHYSICS_DIM, StandardModel};
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;

/// Antimatter encoder
#[derive(Debug, Clone)]
pub struct Antimatter {
    /// CPT conjugation vector (used for verification)
    pub cpt_transform: ContinuousHV,

    // Antileptons
    /// Positron (e⁺)
    pub positron: ContinuousHV,
    /// Antimuon (μ⁺)
    pub antimuon: ContinuousHV,
    /// Antitau (τ⁺)
    pub antitau: ContinuousHV,

    // Antineutrinos
    /// Electron antineutrino (ν̄ₑ)
    pub antineutrino_e: ContinuousHV,
    /// Muon antineutrino (ν̄_μ)
    pub antineutrino_mu: ContinuousHV,
    /// Tau antineutrino (ν̄_τ)
    pub antineutrino_tau: ContinuousHV,

    // Antiquarks
    /// Anti-up quark (ū)
    pub anti_up: ContinuousHV,
    /// Anti-down quark (d̄)
    pub anti_down: ContinuousHV,
    /// Anti-charm quark (c̄)
    pub anti_charm: ContinuousHV,
    /// Anti-strange quark (s̄)
    pub anti_strange: ContinuousHV,
    /// Anti-top quark (t̄)
    pub anti_top: ContinuousHV,
    /// Anti-bottom quark (b̄)
    pub anti_bottom: ContinuousHV,

    // Antihadrons
    /// Antiproton (p̄)
    pub antiproton: ContinuousHV,
    /// Antineutron (n̄)
    pub antineutron: ContinuousHV,

    // Concept vectors
    /// Annihilation concept
    pub annihilation: ContinuousHV,
    /// Pair production concept
    pub pair_production: ContinuousHV,
}

impl Antimatter {
    /// Create antimatter encoder from Standard Model
    pub fn from_model(model: &StandardModel, _hadrons: &Hadrons, genesis: &GenesisSeed) -> Self {
        // CPT transform marker
        let cpt_transform = genesis.hv("antimatter::cpt_transform", PHYSICS_DIM);

        // Antileptons: CPT conjugate of leptons
        let positron = Self::conjugate(&model.electron);
        let antimuon = Self::conjugate(&model.muon);
        let antitau = Self::conjugate(&model.tau);

        // Antineutrinos
        let antineutrino_e = Self::conjugate(&model.electron_neutrino);
        let antineutrino_mu = Self::conjugate(&model.muon_neutrino);
        let antineutrino_tau = Self::conjugate(&model.tau_neutrino);

        // Antiquarks
        let anti_up = Self::conjugate(&model.up_quark);
        let anti_down = Self::conjugate(&model.down_quark);
        let anti_charm = Self::conjugate(&model.charm_quark);
        let anti_strange = Self::conjugate(&model.strange_quark);
        let anti_top = Self::conjugate(&model.top_quark);
        let anti_bottom = Self::conjugate(&model.bottom_quark);

        // Compose antihadrons from antiquarks
        let antiproton = Self::compose_antibaryon(&[&anti_up, &anti_up, &anti_down]);
        let antineutron = Self::compose_antibaryon(&[&anti_up, &anti_down, &anti_down]);

        // Concept vectors
        let annihilation = genesis.hv("antimatter::annihilation", PHYSICS_DIM);
        let pair_production = genesis.hv("antimatter::pair_production", PHYSICS_DIM);

        Self {
            cpt_transform,
            positron,
            antimuon,
            antitau,
            antineutrino_e,
            antineutrino_mu,
            antineutrino_tau,
            anti_up,
            anti_down,
            anti_charm,
            anti_strange,
            anti_top,
            anti_bottom,
            antiproton,
            antineutron,
            annihilation,
            pair_production,
        }
    }

    /// CPT transform: permute by half-dimension (self-inverse)
    ///
    /// This operation flips the "handedness" of the vector in a way that:
    /// 1. Is deterministic
    /// 2. Is self-inverse: conjugate(conjugate(x)) = x
    /// 3. Creates an orthogonal representation
    pub fn conjugate(particle: &ContinuousHV) -> ContinuousHV {
        particle.permute(PHYSICS_DIM / 2)
    }

    /// Verify CPT symmetry: double conjugation recovers original
    pub fn verify_cpt_symmetry(particle: &ContinuousHV) -> bool {
        let anti = Self::conjugate(particle);
        let recovered = Self::conjugate(&anti);
        particle.similarity(&recovered) > 0.99
    }

    /// Compose antibaryon from three antiquarks
    fn compose_antibaryon(antiquarks: &[&ContinuousHV; 3]) -> ContinuousHV {
        // Position encode each antiquark
        let q1 = antiquarks[0].permute(0);
        let q2 = antiquarks[1].permute(1000);
        let q3 = antiquarks[2].permute(2000);
        ContinuousHV::bundle(&[&q1, &q2, &q3])
    }

    /// Compose antimeson from antiquark and quark
    pub fn compose_antimeson(antiquark: &ContinuousHV, quark: &ContinuousHV) -> ContinuousHV {
        antiquark.bind(quark).normalize()
    }

    /// Get the antiparticle of any particle
    pub fn antiparticle(&self, particle: &ContinuousHV) -> ContinuousHV {
        Self::conjugate(particle)
    }
}

/// Annihilation event
#[derive(Debug, Clone)]
pub struct AnnihilationEvent {
    /// The particle involved
    pub particle: ContinuousHV,
    /// The antiparticle involved
    pub antiparticle: ContinuousHV,
    /// Products of annihilation (typically photons)
    pub products: Vec<ContinuousHV>,
    /// Total energy released (MeV)
    pub energy_mev: f64,
    /// Event vector representation
    pub vector: ContinuousHV,
}

impl Antimatter {
    /// Electron-positron annihilation: e⁺ + e⁻ → 2γ (1.022 MeV total)
    pub fn electron_positron_annihilation(&self, model: &StandardModel) -> AnnihilationEvent {
        // Two photons emitted back-to-back
        let photon1 = model.photon.permute(0);
        let photon2 = model.photon.permute(1000);

        // Event vector: bind particle + antiparticle, then bind with products
        let initial = model.electron.bind(&self.positron);
        let final_state = ContinuousHV::bundle(&[&photon1, &photon2]);
        let event_vector = initial.bind(&final_state).bind(&self.annihilation);

        AnnihilationEvent {
            particle: model.electron.clone(),
            antiparticle: self.positron.clone(),
            products: vec![photon1, photon2],
            energy_mev: 1.022, // 2 × 0.511 MeV (electron mass)
            vector: event_vector,
        }
    }

    /// Proton-antiproton annihilation: p + p̄ → mesons (typically pions)
    pub fn proton_antiproton_annihilation(&self, hadrons: &Hadrons) -> AnnihilationEvent {
        // Typically produces multiple pions
        let pion1 = hadrons.pion_plus.permute(0);
        let pion2 = hadrons.pion_minus.permute(1000);
        let pion3 = hadrons.pion_zero.permute(2000);

        let initial = hadrons.proton.bind(&self.antiproton);
        let final_state = ContinuousHV::bundle(&[&pion1, &pion2, &pion3]);
        let event_vector = initial.bind(&final_state).bind(&self.annihilation);

        AnnihilationEvent {
            particle: hadrons.proton.clone(),
            antiparticle: self.antiproton.clone(),
            products: vec![pion1, pion2, pion3],
            energy_mev: 1876.0, // 2 × 938 MeV (proton mass)
            vector: event_vector,
        }
    }

    /// Pair production: γ + γ → e⁺ + e⁻ (requires > 1.022 MeV)
    pub fn pair_production_event(&self, model: &StandardModel) -> AnnihilationEvent {
        // Reverse of annihilation
        let photon1 = model.photon.permute(0);
        let photon2 = model.photon.permute(1000);

        let initial = ContinuousHV::bundle(&[&photon1, &photon2]);
        let final_state = model.electron.bind(&self.positron);
        let event_vector = initial.bind(&final_state).bind(&self.pair_production);

        AnnihilationEvent {
            particle: photon1.clone(),
            antiparticle: photon2,
            products: vec![model.electron.clone(), self.positron.clone()],
            energy_mev: 1.022, // Minimum threshold
            vector: event_vector,
        }
    }
}

/// Antiatom structure
#[derive(Debug, Clone)]
pub struct Antiatom {
    /// Name of the antiatom
    pub name: String,
    /// Number of antiprotons
    pub antiprotons: u8,
    /// Number of antineutrons
    pub antineutrons: u8,
    /// Number of positrons
    pub positrons: u8,
    /// Vector representation
    pub vector: ContinuousHV,
}

impl Antimatter {
    /// Antihydrogen: p̄ + e⁺
    pub fn antihydrogen(&self) -> Antiatom {
        // Single antiproton with single positron
        let nucleus = self.antiproton.clone();
        let cloud = self.positron.clone();
        let vector = ContinuousHV::bundle(&[&nucleus, &cloud]);

        Antiatom {
            name: "Antihydrogen".to_string(),
            antiprotons: 1,
            antineutrons: 0,
            positrons: 1,
            vector,
        }
    }

    /// Antideuterium: p̄ + n̄ + e⁺
    pub fn antideuterium(&self) -> Antiatom {
        let nucleus = ContinuousHV::bundle(&[&self.antiproton, &self.antineutron]);
        let cloud = self.positron.clone();
        let vector = ContinuousHV::bundle(&[&nucleus, &cloud]);

        Antiatom {
            name: "Antideuterium".to_string(),
            antiprotons: 1,
            antineutrons: 1,
            positrons: 1,
            vector,
        }
    }

    /// Antihelium-3: 2p̄ + n̄ + 2e⁺
    pub fn antihelium3(&self) -> Antiatom {
        let nucleus =
            ContinuousHV::weighted_bundle(&[&self.antiproton, &self.antineutron], &[2.0, 1.0]);
        let cloud = self.positron.scale(2.0);
        let vector = ContinuousHV::bundle(&[&nucleus, &cloud]);

        Antiatom {
            name: "Antihelium-3".to_string(),
            antiprotons: 2,
            antineutrons: 1,
            positrons: 2,
            vector,
        }
    }

    /// Antihelium-4: 2p̄ + 2n̄ + 2e⁺
    pub fn antihelium4(&self) -> Antiatom {
        let nucleus =
            ContinuousHV::weighted_bundle(&[&self.antiproton, &self.antineutron], &[2.0, 2.0]);
        let cloud = self.positron.scale(2.0);
        let vector = ContinuousHV::bundle(&[&nucleus, &cloud]);

        Antiatom {
            name: "Antihelium-4".to_string(),
            antiprotons: 2,
            antineutrons: 2,
            positrons: 2,
            vector,
        }
    }

    /// Compose a general antiatom
    pub fn compose_antiatom(&self, antiprotons: u8, antineutrons: u8) -> Antiatom {
        let p = antiprotons as f32;
        let n = antineutrons as f32;

        let nucleus =
            ContinuousHV::weighted_bundle(&[&self.antiproton, &self.antineutron], &[p, n]);
        let cloud = self.positron.scale(p);
        let vector = ContinuousHV::bundle(&[&nucleus, &cloud]);

        Antiatom {
            name: format!("Anti-Z{antiprotons}"),
            antiprotons,
            antineutrons,
            positrons: antiprotons,
            vector,
        }
    }
}

/// Antimatter-matter asymmetry concepts
#[derive(Debug, Clone)]
pub struct BaryogenesisConcepts {
    /// CP violation
    pub cp_violation: ContinuousHV,
    /// Baryon number violation
    pub baryon_violation: ContinuousHV,
    /// Out of equilibrium
    pub out_of_equilibrium: ContinuousHV,
    /// Sakharov conditions
    pub sakharov_conditions: ContinuousHV,
}

impl BaryogenesisConcepts {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            cp_violation: genesis.hv("baryogenesis::cp_violation", PHYSICS_DIM),
            baryon_violation: genesis.hv("baryogenesis::baryon_violation", PHYSICS_DIM),
            out_of_equilibrium: genesis.hv("baryogenesis::out_of_equilibrium", PHYSICS_DIM),
            sakharov_conditions: genesis.hv("baryogenesis::sakharov_conditions", PHYSICS_DIM),
        }
    }

    /// Encode Sakharov conditions for baryogenesis
    pub fn encode_sakharov(&self) -> ContinuousHV {
        // All three conditions must be met
        ContinuousHV::bundle(&[
            &self.baryon_violation,
            &self.cp_violation,
            &self.out_of_equilibrium,
        ])
        .bind(&self.sakharov_conditions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (StandardModel, Hadrons, Antimatter, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("antimatter test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let antimatter = Antimatter::from_model(&model, &hadrons, &genesis);
        (model, hadrons, antimatter, genesis)
    }

    #[test]
    fn test_cpt_double_transform_identity() {
        let (model, hadrons, _, _) = setup();

        // Double conjugation should recover original
        assert!(Antimatter::verify_cpt_symmetry(&model.electron));
        assert!(Antimatter::verify_cpt_symmetry(&model.up_quark));
        assert!(Antimatter::verify_cpt_symmetry(&hadrons.proton));
    }

    #[test]
    fn test_antiparticle_distinct_from_particle() {
        let (model, _, antimatter, _) = setup();

        // Positron should be distinct from electron
        let sim = model.electron.similarity(&antimatter.positron);
        assert!(
            sim < 0.5,
            "Antiparticle should be distinct from particle: e⁻-e⁺={}",
            sim
        );

        // Anti-up should be distinct from up
        let sim_quark = model.up_quark.similarity(&antimatter.anti_up);
        assert!(
            sim_quark < 0.5,
            "Antiquark should be distinct from quark: u-ū={}",
            sim_quark
        );
    }

    #[test]
    fn test_antiproton_composition() {
        let (_, hadrons, antimatter, _) = setup();

        // Antiproton should be distinct from proton
        let sim = hadrons.proton.similarity(&antimatter.antiproton);
        assert!(
            sim < 0.5,
            "Antiproton should be distinct from proton: {}",
            sim
        );

        // Antiproton should be similar to CPT conjugate of proton
        let conjugated_proton = Antimatter::conjugate(&hadrons.proton);
        let sim_conjugate = conjugated_proton.similarity(&antimatter.antiproton);
        assert!(
            sim_conjugate > 0.5,
            "Antiproton should relate to CPT-conjugated proton: {}",
            sim_conjugate
        );
    }

    #[test]
    fn test_annihilation_energy_conservation() {
        let (model, _, antimatter, _) = setup();

        let event = antimatter.electron_positron_annihilation(&model);

        // Energy should be twice electron mass
        assert!((event.energy_mev - 1.022).abs() < 0.001);

        // Should produce two photons
        assert_eq!(event.products.len(), 2);
    }

    #[test]
    fn test_antihydrogen() {
        let (_, hadrons, antimatter, _) = setup();

        let anti_h = antimatter.antihydrogen();

        assert_eq!(anti_h.antiprotons, 1);
        assert_eq!(anti_h.antineutrons, 0);
        assert_eq!(anti_h.positrons, 1);

        // Should be distinct from regular hydrogen-like structure
        // (hypothetically comparing to matter hydrogen)
        let matter_h_approx = ContinuousHV::bundle(&[&hadrons.proton, &antimatter.positron]);
        let sim = anti_h.vector.similarity(&matter_h_approx);

        // They should share some structure but not be identical
        assert!(sim < 0.95, "AntiH should differ from approx matter H");
    }

    #[test]
    fn test_antihelium() {
        let (_, _, antimatter, _) = setup();

        let anti_he3 = antimatter.antihelium3();
        let anti_he4 = antimatter.antihelium4();

        assert_eq!(anti_he3.antiprotons, 2);
        assert_eq!(anti_he3.antineutrons, 1);

        assert_eq!(anti_he4.antiprotons, 2);
        assert_eq!(anti_he4.antineutrons, 2);

        // Isotopes should be similar
        let sim = anti_he3.vector.similarity(&anti_he4.vector);
        assert!(sim > 0.5, "Anti-helium isotopes should be similar: {}", sim);
    }

    #[test]
    fn test_baryogenesis_concepts() {
        let genesis = GenesisSeed::from_phrase("baryogenesis test");
        let concepts = BaryogenesisConcepts::from_genesis(&genesis);

        let sakharov = concepts.encode_sakharov();

        // Should have valid dimension
        assert_eq!(sakharov.dim(), PHYSICS_DIM);

        // Should have non-zero norm
        assert!(sakharov.norm() > 0.0);
    }

    #[test]
    fn test_pair_production() {
        let (model, _, antimatter, _) = setup();

        let event = antimatter.pair_production_event(&model);

        // Should produce electron and positron
        assert_eq!(event.products.len(), 2);
        assert!((event.energy_mev - 1.022).abs() < 0.001);
    }

    #[test]
    fn test_proton_antiproton_annihilation() {
        let (_, hadrons, antimatter, _) = setup();

        let event = antimatter.proton_antiproton_annihilation(&hadrons);

        // Should produce multiple pions
        assert!(event.products.len() >= 3);

        // Energy should be twice proton mass
        assert!((event.energy_mev - 1876.0).abs() < 1.0);
    }

    #[test]
    fn test_antimatter_determinism() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        let am1 = Antimatter::from_model(&model, &hadrons, &genesis);
        let am2 = Antimatter::from_model(&model, &hadrons, &genesis);

        assert!(
            am1.positron.similarity(&am2.positron) > 0.9999,
            "Antimatter should be deterministic"
        );
    }
}
