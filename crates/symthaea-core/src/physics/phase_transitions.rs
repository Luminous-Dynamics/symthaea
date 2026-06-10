// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phase Transitions
//!
//! This module encodes phase transitions and exotic quantum states of matter.
//!
//! ## Phase Transition Types
//!
//! - **First order**: Discontinuous (latent heat, e.g., ice → water)
//! - **Second order**: Continuous (critical point, e.g., ferromagnet → paramagnet)
//! - **Infinite order**: Kosterlitz-Thouless (topological)
//!
//! ## Quantum States of Matter
//!
//! - **Superconductivity**: Zero resistance, Meissner effect, Cooper pairs
//! - **Superfluidity**: Zero viscosity, quantized vortices
//! - **Bose-Einstein Condensate**: Macroscopic quantum state
//! - **Topological Insulators**: Bulk insulator, conducting surface

use super::periodic_table::PeriodicTable;
use super::standard_model::PHYSICS_DIM;
use super::thermodynamics::ThermoEncoder;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Phase transition order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseTransitionOrder {
    /// First order: discontinuous, latent heat
    First,
    /// Second order: continuous, critical exponents
    Second,
    /// Infinite order: Kosterlitz-Thouless type
    Infinite,
}

/// Phase transition
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    /// Name of the transition
    pub name: String,
    /// Order of the transition
    pub order: PhaseTransitionOrder,
    /// Critical temperature in Kelvin
    pub critical_temperature_k: f64,
    /// Order parameter vector
    pub order_parameter: ContinuousHV,
    /// Broken symmetry vector
    pub broken_symmetry: ContinuousHV,
    /// High temperature phase vector
    pub high_temp_phase: ContinuousHV,
    /// Low temperature phase vector
    pub low_temp_phase: ContinuousHV,
    /// Overall transition vector
    pub vector: ContinuousHV,
}

/// Phase encoder
#[derive(Debug, Clone)]
pub struct PhaseEncoder {
    // Universal concepts
    pub order_parameter: ContinuousHV,
    pub symmetry_breaking: ContinuousHV,
    pub critical_point: ContinuousHV,
    pub correlation_length: ContinuousHV,
    pub fluctuations: ContinuousHV,

    // Phase types
    pub solid: ContinuousHV,
    pub liquid: ContinuousHV,
    pub gas: ContinuousHV,
    pub plasma: ContinuousHV,

    // Quantum phases
    pub superconductor: ContinuousHV,
    pub superfluid: ContinuousHV,
    pub bec: ContinuousHV,
    pub topological: ContinuousHV,

    // Process concepts
    pub phonon: ContinuousHV,
    pub cooper_pair: ContinuousHV,
    pub condensate: ContinuousHV,
    pub vortex: ContinuousHV,
}

impl PhaseEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            order_parameter: genesis.hv("phase::order_parameter", PHYSICS_DIM),
            symmetry_breaking: genesis.hv("phase::symmetry_breaking", PHYSICS_DIM),
            critical_point: genesis.hv("phase::critical_point", PHYSICS_DIM),
            correlation_length: genesis.hv("phase::correlation_length", PHYSICS_DIM),
            fluctuations: genesis.hv("phase::fluctuations", PHYSICS_DIM),

            solid: genesis.hv("phase::solid", PHYSICS_DIM),
            liquid: genesis.hv("phase::liquid", PHYSICS_DIM),
            gas: genesis.hv("phase::gas", PHYSICS_DIM),
            plasma: genesis.hv("phase::plasma", PHYSICS_DIM),

            superconductor: genesis.hv("phase::superconductor", PHYSICS_DIM),
            superfluid: genesis.hv("phase::superfluid", PHYSICS_DIM),
            bec: genesis.hv("phase::bec", PHYSICS_DIM),
            topological: genesis.hv("phase::topological", PHYSICS_DIM),

            phonon: genesis.hv("phase::phonon", PHYSICS_DIM),
            cooper_pair: genesis.hv("phase::cooper_pair", PHYSICS_DIM),
            condensate: genesis.hv("phase::condensate", PHYSICS_DIM),
            vortex: genesis.hv("phase::vortex", PHYSICS_DIM),
        }
    }

    /// Create PhaseEncoder from thermodynamics module
    ///
    /// This constructor grounds phase concepts in thermodynamic principles,
    /// connecting phase transitions to entropy and free energy.
    pub fn from_thermo(thermo: &ThermoEncoder, genesis: &GenesisSeed) -> Self {
        // Order parameter relates to entropy reduction
        let order_parameter = thermo
            .entropy
            .bind(&genesis.hv("phase::ordering", PHYSICS_DIM));

        // Symmetry breaking is a free energy transition
        let symmetry_breaking = thermo
            .gibbs
            .bind(&genesis.hv("symmetry::broken", PHYSICS_DIM));

        // Critical point is where fluctuations diverge
        let critical_point = thermo
            .temperature
            .bind(&genesis.hv("critical::divergence", PHYSICS_DIM));

        // Phases related to thermodynamic states
        let solid = ContinuousHV::bundle(&[
            &thermo.entropy.scale(-1.0), // Low entropy
            &genesis.hv("phase::solid", PHYSICS_DIM),
        ]);

        let liquid = thermo
            .entropy
            .bind(&genesis.hv("phase::liquid", PHYSICS_DIM));

        let gas = ContinuousHV::bundle(&[
            &thermo.entropy.scale(2.0), // High entropy
            &genesis.hv("phase::gas", PHYSICS_DIM),
        ]);

        let plasma = ContinuousHV::bundle(&[
            &thermo.energy.scale(3.0), // High energy
            &genesis.hv("phase::plasma", PHYSICS_DIM),
        ]);

        Self {
            order_parameter,
            symmetry_breaking,
            critical_point,
            correlation_length: genesis.hv("phase::correlation_length", PHYSICS_DIM),
            fluctuations: thermo.uncertainty.clone(),

            solid,
            liquid,
            gas,
            plasma,

            superconductor: genesis.hv("phase::superconductor", PHYSICS_DIM),
            superfluid: genesis.hv("phase::superfluid", PHYSICS_DIM),
            bec: genesis.hv("phase::bec", PHYSICS_DIM),
            topological: genesis.hv("phase::topological", PHYSICS_DIM),

            phonon: genesis.hv("phase::phonon", PHYSICS_DIM),
            cooper_pair: genesis.hv("phase::cooper_pair", PHYSICS_DIM),
            condensate: genesis.hv("phase::condensate", PHYSICS_DIM),
            vortex: genesis.hv("phase::vortex", PHYSICS_DIM),
        }
    }

    /// Create a phase transition
    pub fn create_transition(
        &self,
        name: &str,
        order: PhaseTransitionOrder,
        critical_temperature_k: f64,
        high_phase: &ContinuousHV,
        low_phase: &ContinuousHV,
    ) -> PhaseTransition {
        let order_param = self.order_parameter.bind(low_phase);
        let symmetry = self.symmetry_breaking.bind(high_phase).bind(low_phase);

        let vector = ContinuousHV::weighted_bundle(
            &[&order_param, &symmetry, &self.critical_point],
            &[1.0, 1.0, 0.5],
        );

        PhaseTransition {
            name: name.to_string(),
            order,
            critical_temperature_k,
            order_parameter: order_param,
            broken_symmetry: symmetry,
            high_temp_phase: high_phase.clone(),
            low_temp_phase: low_phase.clone(),
            vector,
        }
    }

    /// Water: liquid → solid (ice)
    pub fn water_freezing(&self) -> PhaseTransition {
        self.create_transition(
            "Water freezing",
            PhaseTransitionOrder::First,
            273.15,
            &self.liquid,
            &self.solid,
        )
    }

    /// Ferromagnetic transition
    pub fn ferromagnetic_transition(&self, curie_temp_k: f64) -> PhaseTransition {
        let paramagnetic = self.gas.bind(&self.fluctuations); // Disordered
        let ferromagnetic = self.solid.bind(&self.order_parameter); // Ordered

        self.create_transition(
            "Ferromagnetic",
            PhaseTransitionOrder::Second,
            curie_temp_k,
            &paramagnetic,
            &ferromagnetic,
        )
    }
}

/// Superconductor type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuperconductorType {
    /// Type I: single critical field
    TypeI,
    /// Type II: two critical fields, vortex lattice
    TypeII,
}

/// Superconductor
#[derive(Debug, Clone)]
pub struct Superconductor {
    /// Material name
    pub material: String,
    /// Critical temperature in Kelvin
    pub critical_temp_k: f64,
    /// Critical magnetic field in Tesla
    pub critical_field_tesla: f64,
    /// Type (I or II)
    pub type_: SuperconductorType,
    /// BCS gap in meV
    pub gap_mev: f64,
    /// Cooper pair vector
    pub cooper_pair: ContinuousHV,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl PhaseEncoder {
    /// BCS Cooper pair: bound electron pair via phonon exchange
    pub fn create_cooper_pair(
        &self,
        electron: &ContinuousHV,
        phonon_: &ContinuousHV,
    ) -> ContinuousHV {
        // Two electrons with opposite spin and momentum
        let e1 = electron.permute(0);
        let e2 = electron.permute(PHYSICS_DIM / 2); // Opposite momentum (CPT-like)

        let pair = ContinuousHV::bundle(&[&e1, &e2]);
        pair.bind(phonon_).bind(&self.cooper_pair) // Phonon-mediated attraction
    }

    /// Create a superconductor
    pub fn create_superconductor(
        &self,
        material: &str,
        critical_temp_k: f64,
        critical_field_tesla: f64,
        type_: SuperconductorType,
        electron: &ContinuousHV,
    ) -> Superconductor {
        let gap_mev = 1.76 * 8.617e-2 * critical_temp_k; // BCS relation: Δ ≈ 1.76 k_B T_c
        let cooper_pair = self.create_cooper_pair(electron, &self.phonon);

        let type_factor = match type_ {
            SuperconductorType::TypeI => 0.5,
            SuperconductorType::TypeII => 1.5,
        };

        let vector = self
            .superconductor
            .bind(&cooper_pair)
            .scale((critical_temp_k / 100.0) as f32 * type_factor);

        Superconductor {
            material: material.to_string(),
            critical_temp_k,
            critical_field_tesla,
            type_,
            gap_mev,
            cooper_pair,
            vector,
        }
    }

    /// Meissner effect: flux expulsion
    pub fn meissner_state(&self, sc: &Superconductor) -> ContinuousHV {
        // Perfect diamagnetism: B = 0 inside
        sc.vector.bind(&self.order_parameter)
    }

    /// Abrikosov vortex lattice (Type II)
    pub fn abrikosov_vortex(&self, sc: &Superconductor) -> ContinuousHV {
        if sc.type_ != SuperconductorType::TypeII {
            return ContinuousHV::zero(PHYSICS_DIM);
        }

        sc.vector.bind(&self.vortex)
    }

    /// High-Tc cuprate superconductor (d-wave pairing)
    pub fn cuprate_superconductor(&self, electron: &ContinuousHV) -> Superconductor {
        self.create_superconductor(
            "YBCO",
            92.0, // 92 K
            100.0,
            SuperconductorType::TypeII,
            electron,
        )
    }

    /// Conventional BCS superconductor (s-wave)
    pub fn conventional_superconductor(
        &self,
        material: &str,
        tc: f64,
        electron: &ContinuousHV,
    ) -> Superconductor {
        self.create_superconductor(
            material,
            tc,
            0.1, // Low critical field for Type I
            SuperconductorType::TypeI,
            electron,
        )
    }
}

/// Bose-Einstein Condensate
#[derive(Debug, Clone)]
pub struct BoseEinsteinCondensate {
    /// Atom type (as vector)
    pub atom: ContinuousHV,
    /// Number of atoms
    pub atom_count: f64,
    /// Critical temperature in nanoKelvin
    pub critical_temp_nk: f64,
    /// Condensate fraction (0-1)
    pub condensate_fraction: f64,
    /// Coherence length in micrometers
    pub coherence_length_um: f64,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl PhaseEncoder {
    /// Create a Bose-Einstein condensate
    ///
    /// BEC: macroscopic ground state occupation for bosons
    pub fn create_bec(
        &self,
        atom: &ContinuousHV,
        n_atoms: f64,
        temp_nk: f64,
    ) -> BoseEinsteinCondensate {
        // Critical temperature estimate (simplified)
        let tc = 3.3125 * (n_atoms / 1e6).powf(2.0 / 3.0);

        // Condensate fraction below Tc
        let fraction = if temp_nk < tc {
            1.0 - (temp_nk / tc).powf(1.5)
        } else {
            0.0
        };

        // Coherence length (thermal de Broglie wavelength)
        let coherence_length_um = 1.0 / (temp_nk + 0.001).sqrt();

        // Condensate: all atoms in same quantum state
        let condensate = atom.scale((fraction * n_atoms) as f32);
        let vector = condensate.bind(&self.bec).bind(&self.condensate);

        BoseEinsteinCondensate {
            atom: atom.clone(),
            atom_count: n_atoms,
            critical_temp_nk: tc,
            condensate_fraction: fraction,
            coherence_length_um,
            vector,
        }
    }

    /// BEC interference pattern (two BECs)
    pub fn bec_interference(
        &self,
        bec1: &BoseEinsteinCondensate,
        bec2: &BoseEinsteinCondensate,
    ) -> ContinuousHV {
        // Interference requires phase coherence
        let phase_diff = bec1.atom.permute(1000);
        bec1.vector.bind(&bec2.vector).bind(&phase_diff)
    }

    /// Rubidium-87 BEC (first observed BEC)
    pub fn rb87_bec(
        &self,
        table: &PeriodicTable,
        n_atoms: f64,
        temp_nk: f64,
    ) -> BoseEinsteinCondensate {
        let rb = table
            .element(37)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        self.create_bec(&rb, n_atoms, temp_nk)
    }
}

/// Superfluid
#[derive(Debug, Clone)]
pub struct Superfluid {
    /// Material name (He-4, He-3)
    pub material: String,
    /// Critical temperature in Kelvin
    pub critical_temp_k: f64,
    /// Viscosity (zero below Tc)
    pub viscosity: f64,
    /// Quantized vortex vector
    pub quantized_vortex: ContinuousHV,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl PhaseEncoder {
    /// Helium-4 superfluid (boson condensate)
    pub fn helium4_superfluid(&self, table: &PeriodicTable) -> Superfluid {
        let he = table
            .element(2)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));

        let quantized_vortex = self.vortex.bind(&he);
        let vector = self.superfluid.bind(&he).bind(&self.bec);

        Superfluid {
            material: "He-4".to_string(),
            critical_temp_k: 2.17, // Lambda point
            viscosity: 0.0,
            quantized_vortex,
            vector,
        }
    }

    /// Helium-3 superfluid (fermion pairing, like BCS)
    pub fn helium3_superfluid(&self, table: &PeriodicTable) -> Superfluid {
        let he = table
            .element(2)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));

        // He-3 is a fermion, so it pairs like Cooper pairs
        let he3 = he.permute(3000); // Distinguish from He-4
        let paired = self.create_cooper_pair(&he3, &self.phonon);

        let quantized_vortex = self.vortex.bind(&he3);
        let vector = self.superfluid.bind(&paired);

        Superfluid {
            material: "He-3".to_string(),
            critical_temp_k: 0.0025, // 2.5 mK
            viscosity: 0.0,
            quantized_vortex,
            vector,
        }
    }

    /// Quantized vortex: circulation = n·h/m
    pub fn create_quantized_vortex(&self, sf: &Superfluid, winding: i32) -> ContinuousHV {
        sf.quantized_vortex
            .permute(winding.unsigned_abs() as usize * 1000)
            .scale(winding as f32)
    }

    /// Second sound (temperature wave in superfluid)
    pub fn second_sound(&self, sf: &Superfluid) -> ContinuousHV {
        // In superfluid, entropy can propagate as a wave
        sf.vector.bind(&self.fluctuations)
    }
}

/// Topological insulator
#[derive(Debug, Clone)]
pub struct TopologicalInsulator {
    /// Material name
    pub material: String,
    /// Band gap in eV
    pub band_gap_ev: f64,
    /// Z2 topological invariant (0 = trivial, 1 = topological)
    pub z2_invariant: i32,
    /// Surface state vector
    pub surface_state: ContinuousHV,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl PhaseEncoder {
    /// Create a topological insulator
    pub fn create_topological_insulator(
        &self,
        material: &str,
        band_gap_ev: f64,
        z2: i32,
    ) -> TopologicalInsulator {
        let surface_state = self
            .topological
            .permute((z2.unsigned_abs() as usize) * PHYSICS_DIM / 4);
        let vector = self
            .topological
            .bind(&surface_state)
            .scale(band_gap_ev as f32);

        TopologicalInsulator {
            material: material.to_string(),
            band_gap_ev,
            z2_invariant: z2,
            surface_state,
            vector,
        }
    }

    /// Quantum Hall state
    pub fn quantum_hall(&self, filling_fraction: f64) -> ContinuousHV {
        // Integer or fractional quantum Hall
        self.topological
            .scale(filling_fraction as f32)
            .bind(&self.order_parameter)
    }

    /// Majorana fermion (topological qubit candidate)
    pub fn majorana_mode(&self, ti: &TopologicalInsulator) -> ContinuousHV {
        // Majorana modes appear at boundaries of topological superconductors
        ti.surface_state.bind(&self.superconductor)
    }

    /// Bi2Se3 topological insulator
    pub fn bi2se3(&self) -> TopologicalInsulator {
        self.create_topological_insulator("Bi2Se3", 0.3, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (PhaseEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("phase transitions test");
        let encoder = PhaseEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_water_freezing() {
        let (encoder, _) = setup();

        let transition = encoder.water_freezing();

        assert_eq!(transition.order, PhaseTransitionOrder::First);
        assert!((transition.critical_temperature_k - 273.15).abs() < 0.01);
    }

    #[test]
    fn test_ferromagnetic_transition() {
        let (encoder, _) = setup();

        // Iron: Curie temperature ~1043 K
        let fe_transition = encoder.ferromagnetic_transition(1043.0);

        assert_eq!(fe_transition.order, PhaseTransitionOrder::Second);
        assert!((fe_transition.critical_temperature_k - 1043.0).abs() < 0.01);
    }

    #[test]
    fn test_cooper_pair_symmetric() {
        let (encoder, genesis) = setup();

        let electron = genesis.hv("test::electron", PHYSICS_DIM);
        let cp = encoder.create_cooper_pair(&electron, &encoder.phonon);

        // Cooper pair should be valid vector
        assert_eq!(cp.dim(), PHYSICS_DIM);
        assert!(cp.norm() > 0.0);
    }

    #[test]
    fn test_superconductor() {
        let (encoder, genesis) = setup();
        let electron = genesis.hv("test::electron", PHYSICS_DIM);

        // Aluminum: Type I, Tc = 1.2 K
        let al = encoder.conventional_superconductor("Al", 1.2, &electron);

        assert_eq!(al.type_, SuperconductorType::TypeI);
        assert!((al.critical_temp_k - 1.2).abs() < 0.01);
        assert!(al.gap_mev > 0.0);

        // YBCO: Type II, Tc = 92 K
        let ybco = encoder.cuprate_superconductor(&electron);

        assert_eq!(ybco.type_, SuperconductorType::TypeII);
        assert!(ybco.critical_temp_k > al.critical_temp_k);
    }

    #[test]
    fn test_bec_fraction() {
        let (encoder, genesis) = setup();
        let rb = genesis.hv("test::rb87", PHYSICS_DIM);

        // Above Tc: no condensate
        let bec_hot = encoder.create_bec(&rb, 1e6, 1000.0);
        assert!((bec_hot.condensate_fraction - 0.0).abs() < 0.01);

        // Below Tc: some condensate
        let bec_cold = encoder.create_bec(&rb, 1e6, 0.1);
        assert!(bec_cold.condensate_fraction > 0.5);
    }

    #[test]
    fn test_superfluid_viscosity_zero() {
        let genesis = GenesisSeed::from_phrase("superfluid test");
        let encoder = PhaseEncoder::from_genesis(&genesis);
        let model = super::super::StandardModel::from_genesis(&genesis);
        let hadrons = super::super::Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);

        let he4 = encoder.helium4_superfluid(&table);

        assert!((he4.viscosity - 0.0).abs() < 1e-10);
        assert!((he4.critical_temp_k - 2.17).abs() < 0.01);

        let he3 = encoder.helium3_superfluid(&table);
        assert!(he3.critical_temp_k < he4.critical_temp_k);
    }

    #[test]
    fn test_topological_z2_preserved() {
        let (encoder, _) = setup();

        let ti = encoder.bi2se3();

        assert_eq!(ti.z2_invariant, 1);
        assert!(ti.band_gap_ev > 0.0);

        // Trivial insulator
        let trivial = encoder.create_topological_insulator("Si", 1.1, 0);
        assert_eq!(trivial.z2_invariant, 0);

        // They should have different vectors
        let sim = ti.vector.similarity(&trivial.vector);
        assert!(sim < 0.9, "Topological and trivial should differ: {}", sim);
    }

    #[test]
    fn test_quantum_hall() {
        let (encoder, _) = setup();

        // Integer quantum Hall
        let iqhe = encoder.quantum_hall(1.0);
        assert!(iqhe.norm() > 0.0);

        // Fractional quantum Hall (1/3)
        let fqhe = encoder.quantum_hall(1.0 / 3.0);
        assert!(fqhe.norm() > 0.0);

        // Both should have topological character
        let iqhe_topo = iqhe.similarity(&encoder.topological);
        let fqhe_topo = fqhe.similarity(&encoder.topological);
        assert!(iqhe_topo > 0.0, "IQHE should have topological character");
        assert!(fqhe_topo > 0.0, "FQHE should have topological character");

        // FQHE has fractional charge, should differ from IQHE in magnitude effects
        // (scaling preserves direction but not magnitude)
        assert!(
            iqhe.norm() > fqhe.norm(),
            "IQHE norm > FQHE norm due to scaling"
        );
    }

    #[test]
    fn test_majorana() {
        let (encoder, _) = setup();

        let ti = encoder.bi2se3();
        let majorana = encoder.majorana_mode(&ti);

        assert_eq!(majorana.dim(), PHYSICS_DIM);
        assert!(majorana.norm() > 0.0);
    }

    #[test]
    fn test_vortex_winding() {
        let genesis = GenesisSeed::from_phrase("vortex test");
        let encoder = PhaseEncoder::from_genesis(&genesis);
        let model = super::super::StandardModel::from_genesis(&genesis);
        let hadrons = super::super::Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);

        let sf = encoder.helium4_superfluid(&table);

        let v1 = encoder.create_quantized_vortex(&sf, 1);
        let v2 = encoder.create_quantized_vortex(&sf, 2);
        let v_neg = encoder.create_quantized_vortex(&sf, -1);

        // Different windings should give different vectors
        let sim_12 = v1.similarity(&v2);
        let sim_1neg = v1.similarity(&v_neg);

        assert!(sim_12 < 0.99, "Different windings should differ");
        assert!(sim_1neg < 0.99, "Opposite windings should differ");
    }

    #[test]
    fn test_determinism() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let enc1 = PhaseEncoder::from_genesis(&genesis);
        let enc2 = PhaseEncoder::from_genesis(&genesis);

        assert!(
            enc1.superconductor.similarity(&enc2.superconductor) > 0.9999,
            "Phase encoder should be deterministic"
        );
    }
}
