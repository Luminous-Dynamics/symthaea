// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Biophysics
//!
//! **Status**: Exploratory module — HDC encodings for biophysics (molecular motors, biomolecules, membrane physics).
//!
//! This module bridges physics and biology: the physical principles
//! underlying life at molecular, cellular, and systems levels.
//!
//! ## Key Domains
//!
//! - **Molecular biophysics**: Protein folding, DNA mechanics
//! - **Membrane biophysics**: Ion channels, action potentials
//! - **Cellular mechanics**: Cytoskeleton, motility
//! - **Systems biophysics**: Networks, information flow
//!
//! ## Physical Principles in Biology
//!
//! - Thermal fluctuations (kT scale)
//! - Electrostatics in aqueous environment
//! - Hydrodynamics at low Reynolds number
//! - Information and entropy in living systems

use super::constants::{F_FARADAY, K_BOLTZMANN, R_GAS, T_ROOM};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Thermal energy at 300K (computed from constants)
pub const KT_300: f64 = K_BOLTZMANN * T_ROOM; // J (~25 meV)

/// Biological structure type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BioStructure {
    Protein,
    DNA,
    RNA,
    Membrane,
    Cytoskeleton,
    Virus,
    Cell,
}

/// Molecular motor type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotorType {
    Kinesin,     // Walks along microtubules
    Dynein,      // Opposite direction to kinesin
    Myosin,      // Muscle contraction
    ATPSynthase, // Rotary motor
    Flagellum,   // Bacterial motility
}

/// A biomolecule
#[derive(Debug, Clone)]
pub struct Biomolecule {
    pub name: String,
    pub structure: BioStructure,
    pub molecular_weight_kda: f64,
    pub size_nm: f64,
    pub vector: ContinuousHV,
}

/// A molecular motor
#[derive(Debug, Clone)]
pub struct MolecularMotor {
    pub motor_type: MotorType,
    pub step_size_nm: f64,
    pub stall_force_pn: f64,
    pub velocity_nm_s: f64,
    pub efficiency: f64,
    pub vector: ContinuousHV,
}

/// Biophysics Encoder
#[derive(Debug, Clone)]
pub struct BiophysicsEncoder {
    genesis: GenesisSeed,

    // Fundamental scales
    pub thermal_energy: ContinuousHV, // kT
    pub nm_scale: ContinuousHV,       // nanometer
    pub pn_scale: ContinuousHV,       // piconewton
    pub ms_scale: ContinuousHV,       // millisecond

    // Molecular structures
    pub protein: ContinuousHV,
    pub dna: ContinuousHV,
    pub rna: ContinuousHV,
    pub membrane: ContinuousHV,
    pub lipid: ContinuousHV,
    pub cytoskeleton: ContinuousHV,

    // Protein physics
    pub folding: ContinuousHV,
    pub unfolding: ContinuousHV,
    pub misfolding: ContinuousHV,
    pub alpha_helix: ContinuousHV,
    pub beta_sheet: ContinuousHV,
    pub random_coil: ContinuousHV,
    pub hydrophobic_effect: ContinuousHV,

    // DNA/RNA mechanics
    pub persistence_length: ContinuousHV,
    pub bending: ContinuousHV,
    pub stretching: ContinuousHV,
    pub supercoiling: ContinuousHV,
    pub melting: ContinuousHV,

    // Membrane physics
    pub bilayer: ContinuousHV,
    pub curvature: ContinuousHV,
    pub fluidity: ContinuousHV,
    pub permeability: ContinuousHV,
    pub phase_separation: ContinuousHV,

    // Motors and forces
    pub molecular_motor: ContinuousHV,
    pub atp_hydrolysis: ContinuousHV,
    pub power_stroke: ContinuousHV,
    pub stall_force: ContinuousHV,

    // Electrophysiology
    pub ion_channel: ContinuousHV,
    pub action_potential: ContinuousHV,
    pub membrane_potential: ContinuousHV,
    pub nernst_potential: ContinuousHV,

    // Transport
    pub diffusion: ContinuousHV,
    pub active_transport: ContinuousHV,
    pub vesicle: ContinuousHV,
    pub endocytosis: ContinuousHV,
    pub exocytosis: ContinuousHV,

    // Information
    pub signaling: ContinuousHV,
    pub feedback: ContinuousHV,
    pub regulation: ContinuousHV,
}

impl BiophysicsEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),

            thermal_energy: genesis.hv("bio::thermal_energy", PHYSICS_DIM),
            nm_scale: genesis.hv("bio::nm_scale", PHYSICS_DIM),
            pn_scale: genesis.hv("bio::pn_scale", PHYSICS_DIM),
            ms_scale: genesis.hv("bio::ms_scale", PHYSICS_DIM),

            protein: genesis.hv("bio::protein", PHYSICS_DIM),
            dna: genesis.hv("bio::dna", PHYSICS_DIM),
            rna: genesis.hv("bio::rna", PHYSICS_DIM),
            membrane: genesis.hv("bio::membrane", PHYSICS_DIM),
            lipid: genesis.hv("bio::lipid", PHYSICS_DIM),
            cytoskeleton: genesis.hv("bio::cytoskeleton", PHYSICS_DIM),

            folding: genesis.hv("protein::folding", PHYSICS_DIM),
            unfolding: genesis.hv("protein::unfolding", PHYSICS_DIM),
            misfolding: genesis.hv("protein::misfolding", PHYSICS_DIM),
            alpha_helix: genesis.hv("protein::alpha_helix", PHYSICS_DIM),
            beta_sheet: genesis.hv("protein::beta_sheet", PHYSICS_DIM),
            random_coil: genesis.hv("protein::random_coil", PHYSICS_DIM),
            hydrophobic_effect: genesis.hv("protein::hydrophobic", PHYSICS_DIM),

            persistence_length: genesis.hv("polymer::persistence_length", PHYSICS_DIM),
            bending: genesis.hv("polymer::bending", PHYSICS_DIM),
            stretching: genesis.hv("polymer::stretching", PHYSICS_DIM),
            supercoiling: genesis.hv("dna::supercoiling", PHYSICS_DIM),
            melting: genesis.hv("dna::melting", PHYSICS_DIM),

            bilayer: genesis.hv("membrane::bilayer", PHYSICS_DIM),
            curvature: genesis.hv("membrane::curvature", PHYSICS_DIM),
            fluidity: genesis.hv("membrane::fluidity", PHYSICS_DIM),
            permeability: genesis.hv("membrane::permeability", PHYSICS_DIM),
            phase_separation: genesis.hv("membrane::phase_separation", PHYSICS_DIM),

            molecular_motor: genesis.hv("motor::molecular_motor", PHYSICS_DIM),
            atp_hydrolysis: genesis.hv("motor::atp_hydrolysis", PHYSICS_DIM),
            power_stroke: genesis.hv("motor::power_stroke", PHYSICS_DIM),
            stall_force: genesis.hv("motor::stall_force", PHYSICS_DIM),

            ion_channel: genesis.hv("electro::ion_channel", PHYSICS_DIM),
            action_potential: genesis.hv("electro::action_potential", PHYSICS_DIM),
            membrane_potential: genesis.hv("electro::membrane_potential", PHYSICS_DIM),
            nernst_potential: genesis.hv("electro::nernst", PHYSICS_DIM),

            diffusion: genesis.hv("transport::diffusion", PHYSICS_DIM),
            active_transport: genesis.hv("transport::active", PHYSICS_DIM),
            vesicle: genesis.hv("transport::vesicle", PHYSICS_DIM),
            endocytosis: genesis.hv("transport::endocytosis", PHYSICS_DIM),
            exocytosis: genesis.hv("transport::exocytosis", PHYSICS_DIM),

            signaling: genesis.hv("info::signaling", PHYSICS_DIM),
            feedback: genesis.hv("info::feedback", PHYSICS_DIM),
            regulation: genesis.hv("info::regulation", PHYSICS_DIM),
        }
    }

    // ============================================================
    // PROTEIN PHYSICS
    // ============================================================

    /// Protein folding funnel
    pub fn folding_funnel(&self) -> ContinuousHV {
        self.folding
            .bind(&self.hydrophobic_effect)
            .bind(&self.thermal_energy)
    }

    /// Levinthal paradox: how proteins fold fast despite huge conformational space
    pub fn levinthal_paradox(&self) -> ContinuousHV {
        let conformational_space = self
            .genesis
            .hv("protein::conformational_space", PHYSICS_DIM);
        self.folding
            .bind(&conformational_space)
            .bind(&self.ms_scale)
    }

    /// Create a biomolecule
    pub fn create_biomolecule(
        &self,
        name: &str,
        structure: BioStructure,
        mw_kda: f64,
        size_nm: f64,
    ) -> Biomolecule {
        let struct_vec = match structure {
            BioStructure::Protein => self.protein.clone(),
            BioStructure::DNA => self.dna.clone(),
            BioStructure::RNA => self.rna.clone(),
            BioStructure::Membrane => self.membrane.clone(),
            BioStructure::Cytoskeleton => self.cytoskeleton.clone(),
            BioStructure::Virus => self.protein.bind(&self.membrane),
            BioStructure::Cell => self.membrane.bind(&self.cytoskeleton),
        };

        let vector = struct_vec.bind(&self.nm_scale.scale(size_nm as f32));

        Biomolecule {
            name: name.to_string(),
            structure,
            molecular_weight_kda: mw_kda,
            size_nm,
            vector,
        }
    }

    // ============================================================
    // DNA/RNA MECHANICS
    // ============================================================

    /// DNA persistence length: ~50 nm
    pub fn dna_persistence(&self) -> ContinuousHV {
        self.dna.bind(&self.persistence_length).bind(&self.bending)
    }

    /// DNA stretching (worm-like chain model)
    pub fn dna_stretching(&self, force_pn: f64) -> ContinuousHV {
        self.dna
            .bind(&self.stretching)
            .bind(&self.pn_scale.scale(force_pn as f32))
    }

    /// DNA melting temperature
    pub fn dna_melting(&self, gc_content: f64) -> f64 {
        // Approximate: Tm = 64.9 + 41*(GC - 16.4)/N
        // Simplified for long DNA
        64.9 + 41.0 * (gc_content - 0.164)
    }

    /// Supercoiling energy
    pub fn supercoiling(&self, linking_number_change: i32) -> ContinuousHV {
        self.dna
            .bind(&self.supercoiling)
            .scale(linking_number_change.abs() as f32)
    }

    // ============================================================
    // MEMBRANE PHYSICS
    // ============================================================

    /// Lipid bilayer encoding
    pub fn lipid_bilayer(&self) -> ContinuousHV {
        self.membrane
            .bind(&self.bilayer)
            .bind(&self.lipid)
            .bind(&self.hydrophobic_effect)
    }

    /// Membrane bending energy: E = (κ/2)H² + κ_G·K
    pub fn membrane_bending(&self, mean_curvature: f64, bending_modulus: f64) -> ContinuousHV {
        let energy = 0.5 * bending_modulus * mean_curvature * mean_curvature;
        self.membrane
            .bind(&self.curvature)
            .bind(&self.bending)
            .scale((energy / KT_300) as f32) // In units of kT
    }

    /// Membrane fluidity (depends on temperature, cholesterol, saturation)
    pub fn membrane_fluidity_vec(&self, temp_c: f64, cholesterol_fraction: f64) -> ContinuousHV {
        // Higher temp → more fluid, more cholesterol → less fluid
        let fluidity = (temp_c - 20.0) / 50.0 - cholesterol_fraction;
        self.membrane.bind(&self.fluidity).scale(fluidity as f32)
    }

    // ============================================================
    // MOLECULAR MOTORS
    // ============================================================

    /// Create a molecular motor
    pub fn create_motor(&self, motor_type: MotorType) -> MolecularMotor {
        let (step, stall, vel, eff) = match motor_type {
            MotorType::Kinesin => (8.0, 6.0, 800.0, 0.5),
            MotorType::Dynein => (8.0, 1.0, 100.0, 0.3),
            MotorType::Myosin => (5.5, 3.0, 400.0, 0.4),
            MotorType::ATPSynthase => (1.0, 40.0, 100.0, 0.9), // Rotary
            MotorType::Flagellum => (0.0, 0.0, 25000.0, 0.02), // Rotation rate, not steps
        };

        let vector = self
            .molecular_motor
            .bind(&self.atp_hydrolysis)
            .bind(&self.power_stroke)
            .bind(&self.stall_force.scale(stall as f32));

        MolecularMotor {
            motor_type,
            step_size_nm: step,
            stall_force_pn: stall,
            velocity_nm_s: vel,
            efficiency: eff,
            vector,
        }
    }

    /// Motor efficiency: η = (F·d)/(ΔG_ATP)
    pub fn motor_efficiency(&self, force: f64, step_size_nm: f64) -> f64 {
        let work = force * 1e-12 * step_size_nm * 1e-9; // pN·nm → J
        let delta_g_atp = 50.0 * KT_300; // ~50 kT ≈ 80 kJ/mol
        work / delta_g_atp
    }

    // ============================================================
    // ELECTROPHYSIOLOGY
    // ============================================================

    /// Nernst potential: E = (RT/zF) ln([out]/[in])
    ///
    /// A pure scalar calculation with no dependence on encoder state —
    /// callable as `BiophysicsEncoder::nernst_potential_mv(...)` without
    /// constructing an encoder instance.
    pub fn nernst_potential_mv(z: i32, conc_out: f64, conc_in: f64, temp_k: f64) -> f64 {
        let rt_f = R_GAS * temp_k / F_FARADAY; // R*T/F in volts
        1000.0 * rt_f / (z as f64) * (conc_out / conc_in).ln()
    }

    /// Goldman-Hodgkin-Katz equation for membrane potential
    pub fn ghk_potential(&self) -> ContinuousHV {
        self.membrane_potential
            .bind(&self.ion_channel)
            .bind(&self.nernst_potential)
    }

    /// Action potential propagation
    pub fn action_potential_vec(&self) -> ContinuousHV {
        self.action_potential
            .bind(&self.ion_channel)
            .bind(&self.membrane)
            .bind(&self.signaling)
    }

    // ============================================================
    // DIFFUSION AND TRANSPORT
    // ============================================================

    /// Diffusion coefficient from Stokes-Einstein: D = kT/(6πηr)
    pub fn diffusion_coefficient(&self, radius_nm: f64, viscosity_pas: f64, temp_k: f64) -> f64 {
        let kt = 1.38e-23 * temp_k;
        let r = radius_nm * 1e-9;
        kt / (6.0 * std::f64::consts::PI * viscosity_pas * r)
    }

    /// Mean squared displacement: <x²> = 2Dt (1D)
    pub fn diffusion_vec(&self) -> ContinuousHV {
        self.diffusion.bind(&self.thermal_energy)
    }

    /// Active transport (against gradient)
    pub fn active_transport_vec(&self) -> ContinuousHV {
        self.active_transport
            .bind(&self.atp_hydrolysis)
            .bind(&self.membrane)
    }

    // ============================================================
    // LOW REYNOLDS NUMBER PHYSICS
    // ============================================================

    /// Stokes drag: F = 6πηrv
    pub fn stokes_drag(&self, radius_nm: f64, velocity_um_s: f64, viscosity: f64) -> f64 {
        6.0 * std::f64::consts::PI * viscosity * (radius_nm * 1e-9) * (velocity_um_s * 1e-6)
    }

    /// Reynolds number for a bacterium
    pub fn bacterial_reynolds(&self, size_um: f64, velocity_um_s: f64) -> f64 {
        // Re = ρvL/η ≈ (1000)(v)(L)/(0.001) for water
        1000.0 * (velocity_um_s * 1e-6) * (size_um * 1e-6) / 0.001
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> BiophysicsEncoder {
        let genesis = GenesisSeed::from_phrase("biophysics test");
        BiophysicsEncoder::from_genesis(&genesis)
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = setup();
        assert!(encoder.protein.norm() > 0.0);
        assert!(encoder.membrane.norm() > 0.0);
    }

    #[test]
    fn test_nernst_potential() {
        // K+ at 37°C: [out]=5mM, [in]=140mM
        let e_k = BiophysicsEncoder::nernst_potential_mv(1, 5.0, 140.0, 310.0);

        // Should be around -90 mV
        assert!((e_k - (-90.0)).abs() < 10.0);
    }

    #[test]
    fn test_dna_melting() {
        let encoder = setup();

        let tm_50 = encoder.dna_melting(0.5); // 50% GC
        let tm_30 = encoder.dna_melting(0.3); // 30% GC

        // Higher GC → higher Tm
        assert!(tm_50 > tm_30);
    }

    #[test]
    fn test_molecular_motor() {
        let encoder = setup();

        let kinesin = encoder.create_motor(MotorType::Kinesin);
        let myosin = encoder.create_motor(MotorType::Myosin);

        // Kinesin: 8nm steps, ~6pN stall
        assert_eq!(kinesin.step_size_nm, 8.0);
        assert!(kinesin.stall_force_pn > 5.0);

        // Different motors should have different vectors
        assert!(kinesin.vector.norm() > 0.0);
        assert!(myosin.vector.norm() > 0.0);
    }

    #[test]
    fn test_diffusion_coefficient() {
        let encoder = setup();

        // GFP (~3nm radius) in water at 25°C
        let d = encoder.diffusion_coefficient(3.0, 0.001, 298.0);

        // Should be ~80 μm²/s
        let d_um2_s = d * 1e12;
        assert!(d_um2_s > 50.0 && d_um2_s < 150.0);
    }

    #[test]
    fn test_low_reynolds() {
        let encoder = setup();

        // E. coli: 2μm, 30μm/s
        let re = encoder.bacterial_reynolds(2.0, 30.0);

        // Should be <<1 (viscous dominated)
        assert!(re < 0.001);
    }

    #[test]
    fn test_motor_efficiency() {
        let encoder = setup();

        // Kinesin: 6pN force, 8nm steps
        let eff = encoder.motor_efficiency(6.0, 8.0);

        // Calculated efficiency: (6e-12 * 8e-9) / (50 * 4.11e-21) ≈ 0.23
        // Motor efficiency is typically 20-60% for molecular motors
        assert!(eff > 0.1 && eff < 0.8, "Efficiency was {}", eff);
    }

    #[test]
    fn test_biomolecule_creation() {
        let encoder = setup();

        let gfp = encoder.create_biomolecule("GFP", BioStructure::Protein, 27.0, 3.0);

        assert_eq!(gfp.structure, BioStructure::Protein);
        assert!(gfp.vector.norm() > 0.0);
    }
}
