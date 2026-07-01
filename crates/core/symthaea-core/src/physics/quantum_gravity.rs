// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Quantum Gravity Hints
//!
//! This module encodes speculative quantum gravity concepts including
//! Planck scale physics, loop quantum gravity, string theory, and the holographic principle.
//!
//! ## Planck Scale
//!
//! The Planck units define the scale where quantum and gravitational effects are both important:
//! - Planck length: l_P = √(ℏG/c³) ≈ 1.616×10⁻³⁵ m
//! - Planck time: t_P = √(ℏG/c⁵) ≈ 5.391×10⁻⁴⁴ s
//! - Planck mass: m_P = √(ℏc/G) ≈ 2.176×10⁻⁸ kg
//!
//! ## Loop Quantum Gravity
//!
//! LQG quantizes spacetime geometry:
//! - Spin networks: graphs with SU(2) spins on edges
//! - Area quantization: A = 8πγ√(j(j+1)) l_P²
//! - Volume quantization at nodes
//!
//! ## String Theory
//!
//! Fundamental strings with various excitation modes:
//! - T-duality: R ↔ α'/R
//! - S-duality: g ↔ 1/g
//! - D-branes: extended objects
//!
//! ## Holographic Principle
//!
//! AdS/CFT correspondence: bulk gravity ↔ boundary QFT
//! - Bekenstein-Hawking entropy: S = A/(4l_P²)
//! - ER = EPR: wormholes from entanglement

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Planck units
#[derive(Debug, Clone)]
pub struct PlanckUnits {
    /// Planck length in meters
    pub length_m: f64,
    /// Planck time in seconds
    pub time_s: f64,
    /// Planck mass in kg
    pub mass_kg: f64,
    /// Planck energy in Joules
    pub energy_j: f64,
    /// Planck temperature in Kelvin
    pub temperature_k: f64,
    /// Planck vector
    pub vector: ContinuousHV,
}

impl PlanckUnits {
    /// Create Planck units from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            length_m: 1.616255e-35,
            time_s: 5.391247e-44,
            mass_kg: 2.176434e-8,
            energy_j: 1.9561e9,
            temperature_k: 1.416784e32,
            vector: genesis.hv("planck::scale", PHYSICS_DIM),
        }
    }

    /// Check self-consistency: l_P = c × t_P
    pub fn is_self_consistent(&self) -> bool {
        use super::constants::C;
        let ratio = self.length_m / (C * self.time_s);
        (ratio - 1.0).abs() < 0.01
    }
}

/// Quantum gravity encoder
#[derive(Debug, Clone)]
pub struct QuantumGravityEncoder {
    pub planck: PlanckUnits,

    // Fundamental concepts
    pub spacetime: ContinuousHV,
    pub quantum_foam: ContinuousHV,
    pub holographic: ContinuousHV,
    pub emergence: ContinuousHV,

    // LQG concepts
    pub spin_network: ContinuousHV,
    pub spin_foam: ContinuousHV,
    pub area_quantization: ContinuousHV,
    pub volume_quantization: ContinuousHV,

    // String theory concepts
    pub string: ContinuousHV,
    pub brane: ContinuousHV,
    pub compactification: ContinuousHV,
    pub duality: ContinuousHV,

    // Constants
    pub immirzi_parameter: f64, // γ ≈ 0.2375
}

impl QuantumGravityEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            planck: PlanckUnits::from_genesis(genesis),

            spacetime: genesis.hv("qg::spacetime", PHYSICS_DIM),
            quantum_foam: genesis.hv("qg::quantum_foam", PHYSICS_DIM),
            holographic: genesis.hv("qg::holographic", PHYSICS_DIM),
            emergence: genesis.hv("qg::emergence", PHYSICS_DIM),

            spin_network: genesis.hv("lqg::spin_network", PHYSICS_DIM),
            spin_foam: genesis.hv("lqg::spin_foam", PHYSICS_DIM),
            area_quantization: genesis.hv("lqg::area_quantization", PHYSICS_DIM),
            volume_quantization: genesis.hv("lqg::volume_quantization", PHYSICS_DIM),

            string: genesis.hv("string::fundamental", PHYSICS_DIM),
            brane: genesis.hv("string::brane", PHYSICS_DIM),
            compactification: genesis.hv("string::compactification", PHYSICS_DIM),
            duality: genesis.hv("string::duality", PHYSICS_DIM),

            immirzi_parameter: 0.2375,
        }
    }
}

/// Spin network node
#[derive(Debug, Clone)]
pub struct SpinNetworkNode {
    /// Intertwiner quantum number (SU(2))
    pub intertwiner: usize,
    /// Number of edges meeting at this node
    pub valence: usize,
    /// Volume quanta (in Planck units)
    pub volume_quanta: f64,
    /// Node vector
    pub vector: ContinuousHV,
}

/// Spin network edge
#[derive(Debug, Clone)]
pub struct SpinNetworkEdge {
    /// Spin (half-integer: 1/2, 1, 3/2, ...)
    pub spin: f64,
    /// Area quanta (in Planck units²)
    pub area_quanta: f64,
    /// Edge vector
    pub vector: ContinuousHV,
}

/// Spin network (quantum state of spatial geometry in LQG)
#[derive(Debug, Clone)]
pub struct SpinNetwork {
    /// Nodes
    pub nodes: Vec<SpinNetworkNode>,
    /// Edges
    pub edges: Vec<SpinNetworkEdge>,
    /// Total area (in Planck units²)
    pub total_area: f64,
    /// Total volume (in Planck units³)
    pub total_volume: f64,
    /// Network vector
    pub vector: ContinuousHV,
}

impl QuantumGravityEncoder {
    /// Area eigenvalue in LQG: A = 8πγ√(j(j+1)) l_P²
    pub fn area_quantum(&self, spin_j: f64) -> f64 {
        let l_p_sq = self.planck.length_m * self.planck.length_m;
        8.0 * PI * self.immirzi_parameter * (spin_j * (spin_j + 1.0)).sqrt() * l_p_sq
    }

    /// Create a spin network edge
    pub fn create_edge(&self, spin_j: f64) -> SpinNetworkEdge {
        let area = self.area_quantum(spin_j);
        let vector = self
            .spin_network
            .permute((spin_j * 1000.0).max(0.0) as usize)
            .scale(spin_j as f32)
            .bind(&self.area_quantization);

        SpinNetworkEdge {
            spin: spin_j,
            area_quanta: area,
            vector,
        }
    }

    /// Create a spin network from spins
    pub fn create_spin_network(&self, spins: &[f64]) -> SpinNetwork {
        let edges: Vec<SpinNetworkEdge> = spins.iter().map(|&j| self.create_edge(j)).collect();

        let total_area: f64 = edges.iter().map(|e| e.area_quanta).sum();

        let edge_refs: Vec<&ContinuousHV> = edges.iter().map(|e| &e.vector).collect();
        let vector = if edge_refs.is_empty() {
            ContinuousHV::zero(PHYSICS_DIM)
        } else {
            ContinuousHV::bundle(&edge_refs).bind(&self.area_quantization)
        };

        SpinNetwork {
            nodes: vec![],
            edges,
            total_area,
            total_volume: 0.0,
            vector,
        }
    }

    /// Spin foam: history of spin network evolution
    pub fn create_spin_foam(&self, initial: &SpinNetwork, final_net: &SpinNetwork) -> ContinuousHV {
        initial.vector.bind(&final_net.vector).bind(&self.spin_foam)
    }

    /// Area eigenvalues are discrete
    pub fn area_is_discrete(&self) -> bool {
        let a_half = self.area_quantum(0.5);
        let a_one = self.area_quantum(1.0);
        let a_three_half = self.area_quantum(1.5);

        // Check that areas are discrete and increasing
        a_half < a_one && a_one < a_three_half
    }
}

/// String type (in string theory)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StringType {
    TypeI,
    TypeIIA,
    TypeIIB,
    HeteroticSO32,
    HeteroticE8,
}

/// String state
#[derive(Debug, Clone)]
pub struct StringState {
    /// Type of string
    pub string_type: StringType,
    /// Oscillator mode excitations
    pub oscillator_modes: Vec<i32>,
    /// Winding number
    pub winding: i32,
    /// Kaluza-Klein momentum
    pub momentum: i32,
    /// Mass squared (in string units)
    pub mass_squared: f64,
    /// State vector
    pub vector: ContinuousHV,
}

impl QuantumGravityEncoder {
    /// Fundamental string (ground state)
    pub fn fundamental_string(&self) -> StringState {
        StringState {
            string_type: StringType::TypeIIB,
            oscillator_modes: vec![],
            winding: 0,
            momentum: 0,
            mass_squared: 0.0, // Massless ground state (graviton)
            vector: self.string.clone(),
        }
    }

    /// Excited string state
    pub fn excited_string(&self, modes: &[i32]) -> StringState {
        let n: i32 = modes.iter().sum();

        let vector = self
            .string
            .scale(n as f32)
            .permute((n.unsigned_abs() as usize) * 100);

        StringState {
            string_type: StringType::TypeIIB,
            oscillator_modes: modes.to_vec(),
            winding: 0,
            momentum: 0,
            mass_squared: n as f64, // M² = N/α' in string units
            vector,
        }
    }

    /// String mass squared is non-negative
    pub fn mass_squared_nonnegative(&self, state: &StringState) -> bool {
        state.mass_squared >= 0.0
    }

    /// T-duality: swaps winding and momentum, R ↔ α'/R
    pub fn t_duality(&self, state: &StringState) -> StringState {
        StringState {
            string_type: state.string_type,
            oscillator_modes: state.oscillator_modes.clone(),
            winding: state.momentum,
            momentum: state.winding,
            mass_squared: state.mass_squared, // Mass unchanged under T-duality
            vector: state.vector.permute(PHYSICS_DIM / 4).bind(&self.duality),
        }
    }

    /// S-duality: strong ↔ weak coupling
    pub fn s_duality(&self, coupling: f64) -> f64 {
        1.0 / coupling
    }
}

/// D-brane
#[derive(Debug, Clone)]
pub struct Brane {
    /// Spatial dimension (D0 = point, D1 = string, D2 = membrane, ...)
    pub dimension: usize,
    /// RR charge
    pub charge: f64,
    /// Tension (energy per unit volume)
    pub tension: f64,
    /// Brane vector
    pub vector: ContinuousHV,
}

impl QuantumGravityEncoder {
    /// Create a D-brane
    pub fn d_brane(&self, dimension: usize) -> Brane {
        let tension = 1.0 / (2.0 * PI).powi(dimension as i32);

        let vector = self.brane.permute(dimension * 2000);

        Brane {
            dimension,
            charge: 1.0,
            tension,
            vector,
        }
    }

    /// D0-brane (point particle in string theory)
    pub fn d0_brane(&self) -> Brane {
        self.d_brane(0)
    }

    /// D3-brane (important for AdS/CFT)
    pub fn d3_brane(&self) -> Brane {
        self.d_brane(3)
    }
}

/// Holographic boundary (AdS/CFT)
#[derive(Debug, Clone)]
pub struct HolographicBoundary {
    /// Bulk (AdS) state
    pub bulk: ContinuousHV,
    /// Boundary (CFT) state
    pub boundary: ContinuousHV,
    /// AdS radius (in string units)
    pub ads_radius: f64,
    /// Boundary CFT dimension
    pub cft_dimension: usize,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl QuantumGravityEncoder {
    /// AdS/CFT correspondence: bulk gravity ↔ boundary QFT
    pub fn ads_cft(&self, bulk_state: &ContinuousHV, boundary_dim: usize) -> HolographicBoundary {
        // Boundary encodes bulk holographically
        let boundary = bulk_state.permute(boundary_dim * 1000);

        let vector = bulk_state.bind(&self.holographic);

        HolographicBoundary {
            bulk: bulk_state.clone(),
            boundary,
            ads_radius: 1.0,
            cft_dimension: boundary_dim,
            vector,
        }
    }

    /// Bekenstein-Hawking entropy: S = A / (4 l_P²)
    ///
    /// Returns entropy in natural units (k_B = 1)
    pub fn black_hole_entropy(&self, area_m2: f64) -> f64 {
        let l_p_sq = self.planck.length_m * self.planck.length_m;
        area_m2 / (4.0 * l_p_sq)
    }

    /// Holographic bound: S ≤ A / (4 l_P²)
    pub fn holographic_bound_satisfied(&self, region_area_m2: f64, entropy_bits: f64) -> bool {
        let max_entropy = self.black_hole_entropy(region_area_m2);
        entropy_bits <= max_entropy
    }

    /// Schwarzschild radius: r_s = 2GM/c²
    pub fn schwarzschild_radius(&self, mass_kg: f64) -> f64 {
        use super::constants::{C, G};
        2.0 * G * mass_kg / (C * C)
    }
}

/// Emergent spacetime
#[derive(Debug, Clone)]
pub struct EmergentSpacetime {
    /// Pre-geometric quantum substrate
    pub pre_geometric: ContinuousHV,
    /// Emerged classical spacetime
    pub geometric: ContinuousHV,
    /// Emergence scale
    pub emergence_scale: f64,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl QuantumGravityEncoder {
    /// ER = EPR: wormholes from entanglement
    pub fn er_epr(&self, entangled_a: &ContinuousHV, entangled_b: &ContinuousHV) -> ContinuousHV {
        // Entanglement creates geometric connection
        let entanglement = entangled_a.bind(entangled_b);
        entanglement.bind(&self.spacetime)
    }

    /// Spacetime from entanglement entropy
    pub fn spacetime_from_entanglement(
        &self,
        subsystem_a: &ContinuousHV,
        subsystem_b: &ContinuousHV,
    ) -> EmergentSpacetime {
        let entanglement = subsystem_a.bind(subsystem_b);
        let pre_geo = ContinuousHV::bundle(&[subsystem_a, subsystem_b, &entanglement]);

        let geometric = pre_geo.bind(&self.emergence);

        EmergentSpacetime {
            pre_geometric: pre_geo.clone(),
            geometric,
            emergence_scale: self.planck.length_m * 1000.0, // ~1000 Planck lengths
            vector: pre_geo.bind(&self.spacetime).bind(&self.emergence),
        }
    }

    /// Is spacetime emergent from entanglement?
    pub fn spacetime_is_emergent(&self, emergent: &EmergentSpacetime) -> bool {
        // Check that geometric differs from pre-geometric
        let sim = emergent.pre_geometric.similarity(&emergent.geometric);
        sim < 0.9 // Some transformation occurred
    }
}

/// Information paradox concepts
#[derive(Debug, Clone)]
pub struct InformationParadox {
    /// Black hole interior
    pub interior: ContinuousHV,
    /// Hawking radiation
    pub hawking_radiation: ContinuousHV,
    /// Is information preserved?
    pub information_preserved: bool,
    /// Vector
    pub vector: ContinuousHV,
}

impl QuantumGravityEncoder {
    /// Encode the black hole information paradox
    pub fn information_paradox(&self, genesis: &GenesisSeed) -> InformationParadox {
        let interior = genesis.hv("bh::interior", PHYSICS_DIM);
        let hawking_radiation = genesis.hv("bh::hawking_radiation", PHYSICS_DIM);

        // In modern understanding, information is preserved (Page curve)
        let vector = interior.bind(&hawking_radiation).bind(&self.holographic);

        InformationParadox {
            interior,
            hawking_radiation,
            information_preserved: true,
            vector,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (QuantumGravityEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("quantum gravity test");
        let encoder = QuantumGravityEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_planck_units_self_consistent() {
        let (encoder, _) = setup();

        assert!(encoder.planck.is_self_consistent());
    }

    #[test]
    fn test_area_eigenvalues_discrete() {
        let (encoder, _) = setup();

        assert!(encoder.area_is_discrete());

        // Specific values
        let a_half = encoder.area_quantum(0.5);
        let a_one = encoder.area_quantum(1.0);

        assert!(a_half > 0.0);
        assert!(a_one > a_half);
    }

    #[test]
    fn test_spin_network_creation() {
        let (encoder, _) = setup();

        let spins = vec![0.5, 1.0, 0.5, 1.5];
        let network = encoder.create_spin_network(&spins);

        assert_eq!(network.edges.len(), 4);
        assert!(network.total_area > 0.0);
    }

    #[test]
    fn test_string_mass_squared_nonnegative() {
        let (encoder, _) = setup();

        let ground = encoder.fundamental_string();
        assert!(encoder.mass_squared_nonnegative(&ground));

        let excited = encoder.excited_string(&[1, 1, 1]);
        assert!(encoder.mass_squared_nonnegative(&excited));
        assert!(excited.mass_squared > ground.mass_squared);
    }

    #[test]
    fn test_t_duality_swaps_winding_momentum() {
        let (encoder, _) = setup();

        let state = StringState {
            string_type: StringType::TypeIIB,
            oscillator_modes: vec![],
            winding: 2,
            momentum: 3,
            mass_squared: 0.0,
            vector: encoder.string.clone(),
        };

        let dual = encoder.t_duality(&state);

        assert_eq!(dual.winding, state.momentum);
        assert_eq!(dual.momentum, state.winding);
    }

    #[test]
    fn test_s_duality() {
        let (encoder, _) = setup();

        let g = 0.1; // Weak coupling
        let g_dual = encoder.s_duality(g);

        assert!((g_dual - 10.0).abs() < 0.001);
        assert!((encoder.s_duality(g_dual) - g).abs() < 0.001);
    }

    #[test]
    fn test_holographic_bound() {
        let (encoder, _) = setup();

        // Solar mass black hole
        let m_solar = 2e30; // kg
        let r_s = encoder.schwarzschild_radius(m_solar);
        let area = 4.0 * PI * r_s * r_s;

        let entropy = encoder.black_hole_entropy(area);

        // Bekenstein-Hawking entropy should be huge
        assert!(entropy > 1e70);

        // Holographic bound should be satisfied for the black hole itself
        assert!(encoder.holographic_bound_satisfied(area, entropy));
    }

    #[test]
    fn test_d_branes() {
        let (encoder, _) = setup();

        let d0 = encoder.d0_brane();
        let d3 = encoder.d3_brane();

        assert_eq!(d0.dimension, 0);
        assert_eq!(d3.dimension, 3);
        assert!(d0.tension > d3.tension); // Lower dimension = higher tension
    }

    #[test]
    fn test_ads_cft() {
        let (encoder, genesis) = setup();

        let bulk = genesis.hv("test::bulk", PHYSICS_DIM);
        let hologram = encoder.ads_cft(&bulk, 4); // 4D boundary

        assert_eq!(hologram.cft_dimension, 4);
        assert!(hologram.vector.norm() > 0.0);
    }

    #[test]
    fn test_er_epr() {
        let (encoder, genesis) = setup();

        let alice = genesis.hv("test::alice", PHYSICS_DIM);
        let bob = genesis.hv("test::bob", PHYSICS_DIM);

        let wormhole = encoder.er_epr(&alice, &bob);

        assert_eq!(wormhole.dim(), PHYSICS_DIM);
        assert!(wormhole.norm() > 0.0);
    }

    #[test]
    fn test_emergent_spacetime() {
        let (encoder, genesis) = setup();

        let a = genesis.hv("test::a", PHYSICS_DIM);
        let b = genesis.hv("test::b", PHYSICS_DIM);

        let emergent = encoder.spacetime_from_entanglement(&a, &b);

        assert!(encoder.spacetime_is_emergent(&emergent));
        assert!(emergent.emergence_scale > encoder.planck.length_m);
    }

    #[test]
    fn test_information_paradox() {
        let (encoder, genesis) = setup();

        let paradox = encoder.information_paradox(&genesis);

        // Modern understanding: information is preserved
        assert!(paradox.information_preserved);
    }

    #[test]
    fn test_determinism() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let enc1 = QuantumGravityEncoder::from_genesis(&genesis);
        let enc2 = QuantumGravityEncoder::from_genesis(&genesis);

        assert!(
            enc1.spacetime.similarity(&enc2.spacetime) > 0.9999,
            "QG encoder should be deterministic"
        );
    }
}
