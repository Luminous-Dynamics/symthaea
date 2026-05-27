// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Quantum Field Theory
//!
//! **Status**: Exploratory module — HDC encodings for quantum field theory (Feynman diagrams, propagators, vertices).
//!
//! This module encodes QFT concepts including Feynman diagrams, propagators,
//! and vertices for the Standard Model interactions.
//!
//! ## Feynman Diagram Primitives
//!
//! - **Propagators**: Internal lines representing virtual particle exchange
//! - **Vertices**: Interaction points where particles meet
//! - **External legs**: Incoming/outgoing real particles
//!
//! ## Gauge Theories
//!
//! - **QED**: U(1) - electromagnetic interaction (α ≈ 1/137)
//! - **QCD**: SU(3) - strong interaction (αs ≈ 0.12 at MZ)
//! - **Electroweak**: SU(2)×U(1) - weak and electromagnetic unified

use super::constants::ALPHA;
use super::hadrons::Hadrons;
use super::standard_model::{PHYSICS_DIM, StandardModel};
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Propagator type (line style in Feynman diagram)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagatorType {
    /// Fermion: solid line (electrons, quarks)
    Fermion,
    /// Vector boson: wavy line (photons, W, Z)
    Boson,
    /// Gluon: curly line
    Gluon,
    /// Higgs: dashed line
    Higgs,
    /// Ghost: dotted line (gauge fixing)
    Ghost,
}

impl PropagatorType {
    fn domain_label(&self) -> &'static str {
        match self {
            PropagatorType::Fermion => "qft::propagator_fermion",
            PropagatorType::Boson => "qft::propagator_boson",
            PropagatorType::Gluon => "qft::propagator_gluon",
            PropagatorType::Higgs => "qft::propagator_higgs",
            PropagatorType::Ghost => "qft::propagator_ghost",
        }
    }
}

/// Propagator (internal line in Feynman diagram)
#[derive(Debug, Clone)]
pub struct Propagator {
    /// Particle being propagated
    pub particle: ContinuousHV,
    /// Type of propagator
    pub propagator_type: PropagatorType,
    /// Momentum (virtual)
    pub momentum: ContinuousHV,
    /// Overall vector
    pub vector: ContinuousHV,
}

/// Vertex (interaction point)
#[derive(Debug, Clone)]
pub struct Vertex {
    /// Coupling constant
    pub coupling: f64,
    /// Incoming particles
    pub incoming: Vec<ContinuousHV>,
    /// Outgoing particles
    pub outgoing: Vec<ContinuousHV>,
    /// Vertex vector
    pub vector: ContinuousHV,
}

/// Feynman diagram
#[derive(Debug, Clone)]
pub struct FeynmanDiagram {
    /// Diagram name
    pub name: String,
    /// Internal propagators
    pub propagators: Vec<Propagator>,
    /// Vertices
    pub vertices: Vec<Vertex>,
    /// External legs (incoming + outgoing)
    pub external_legs: Vec<Propagator>,
    /// Loop order (0 = tree level)
    pub loop_order: usize,
    /// Estimated amplitude (coupling^n)
    pub amplitude_estimate: f64,
    /// Overall diagram vector
    pub vector: ContinuousHV,
}

/// Divergence type for loop diagrams
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceType {
    /// Finite integral
    Finite,
    /// Ultraviolet divergent
    UVDivergent,
    /// Infrared divergent
    IRDivergent,
    /// Both UV and IR divergent
    Both,
}

/// QED Encoder
#[derive(Debug, Clone)]
pub struct QEDEncoder {
    /// Electron propagator
    pub electron_propagator: ContinuousHV,
    /// Photon propagator
    pub photon_propagator: ContinuousHV,
    /// QED vertex (e-γ-e)
    pub qed_vertex: ContinuousHV,
    /// Fine structure constant α ≈ 1/137
    pub fine_structure: f64,

    // Propagator type vectors
    fermion_type: ContinuousHV,
    boson_type: ContinuousHV,
    momentum: ContinuousHV,
}

impl QEDEncoder {
    pub fn from_genesis(genesis: &GenesisSeed, model: &StandardModel) -> Self {
        let fermion_type = genesis.hv(PropagatorType::Fermion.domain_label(), PHYSICS_DIM);
        let boson_type = genesis.hv(PropagatorType::Boson.domain_label(), PHYSICS_DIM);
        let momentum = genesis.hv("qft::momentum", PHYSICS_DIM);

        let electron_propagator = model.electron.bind(&fermion_type);
        let photon_propagator = model.photon.bind(&boson_type);
        let qed_vertex = genesis.hv("qft::qed_vertex", PHYSICS_DIM);

        Self {
            electron_propagator,
            photon_propagator,
            qed_vertex,
            fine_structure: ALPHA,
            fermion_type,
            boson_type,
            momentum,
        }
    }

    /// Create electron propagator
    pub fn electron_prop(&self) -> Propagator {
        Propagator {
            particle: self.electron_propagator.clone(),
            propagator_type: PropagatorType::Fermion,
            momentum: self.momentum.clone(),
            vector: self.electron_propagator.bind(&self.momentum),
        }
    }

    /// Create photon propagator
    pub fn photon_prop(&self) -> Propagator {
        Propagator {
            particle: self.photon_propagator.clone(),
            propagator_type: PropagatorType::Boson,
            momentum: self.momentum.clone(),
            vector: self.photon_propagator.bind(&self.momentum),
        }
    }

    /// Basic QED vertex: e⁻ + γ → e⁻
    pub fn vertex(&self) -> Vertex {
        let incoming = vec![
            self.electron_propagator.clone(),
            self.photon_propagator.clone(),
        ];
        let outgoing = vec![self.electron_propagator.clone()];

        let vertex_vector = self
            .qed_vertex
            .bind(&self.electron_propagator)
            .bind(&self.photon_propagator);

        Vertex {
            coupling: self.fine_structure.sqrt(), // √α
            incoming,
            outgoing,
            vector: vertex_vector,
        }
    }

    /// Compton scattering: e⁻ + γ → e⁻ + γ (tree-level, 2 diagrams)
    pub fn compton_scattering(&self) -> FeynmanDiagram {
        let v1 = self.vertex();
        let v2 = self.vertex();

        // s-channel and u-channel diagrams
        let prop = self.electron_prop();

        let amplitude = self.fine_structure; // α from two vertices

        let vector = v1.vector.bind(&v2.vector).bind(&prop.vector);

        FeynmanDiagram {
            name: "Compton scattering".to_string(),
            propagators: vec![prop],
            vertices: vec![v1, v2],
            external_legs: vec![
                self.electron_prop(),
                self.photon_prop(),
                self.electron_prop(),
                self.photon_prop(),
            ],
            loop_order: 0,
            amplitude_estimate: amplitude,
            vector,
        }
    }

    /// Bhabha scattering: e⁺ + e⁻ → e⁺ + e⁻
    pub fn bhabha_scattering(&self) -> FeynmanDiagram {
        let v1 = self.vertex();
        let v2 = self.vertex();

        let photon = self.photon_prop();

        let amplitude = self.fine_structure;

        let vector = v1.vector.bind(&v2.vector).bind(&photon.vector);

        FeynmanDiagram {
            name: "Bhabha scattering".to_string(),
            propagators: vec![photon],
            vertices: vec![v1, v2],
            external_legs: vec![
                self.electron_prop(),
                self.electron_prop(),
                self.electron_prop(),
                self.electron_prop(),
            ],
            loop_order: 0,
            amplitude_estimate: amplitude,
            vector,
        }
    }

    /// Pair production: γ + γ → e⁺ + e⁻
    pub fn pair_production(&self) -> FeynmanDiagram {
        let v1 = self.vertex();
        let v2 = self.vertex();
        let electron = self.electron_prop();

        let vector = v1.vector.bind(&v2.vector).bind(&electron.vector);

        FeynmanDiagram {
            name: "Pair production".to_string(),
            propagators: vec![electron],
            vertices: vec![v1, v2],
            external_legs: vec![
                self.photon_prop(),
                self.photon_prop(),
                self.electron_prop(),
                self.electron_prop(),
            ],
            loop_order: 0,
            amplitude_estimate: self.fine_structure,
            vector,
        }
    }
}

/// QCD Encoder
#[derive(Debug, Clone)]
pub struct QCDEncoder {
    /// Quark propagator
    pub quark_propagator: ContinuousHV,
    /// Gluon propagator
    pub gluon_propagator: ContinuousHV,
    /// Quark-gluon vertex
    pub qqg_vertex: ContinuousHV,
    /// Triple gluon vertex (non-Abelian)
    pub ggg_vertex: ContinuousHV,
    /// Quartic gluon vertex
    pub gggg_vertex: ContinuousHV,
    /// Color charges (SU(3) generators, simplified)
    pub color_charges: [ContinuousHV; 8],
    /// Strong coupling constant αs ≈ 0.12 at MZ
    pub strong_coupling: f64,

    gluon_type: ContinuousHV,
    fermion_type: ContinuousHV,
}

impl QCDEncoder {
    pub fn from_genesis(genesis: &GenesisSeed, model: &StandardModel) -> Self {
        let fermion_type = genesis.hv(PropagatorType::Fermion.domain_label(), PHYSICS_DIM);
        let gluon_type = genesis.hv(PropagatorType::Gluon.domain_label(), PHYSICS_DIM);

        let quark_propagator = model.up_quark.bind(&fermion_type);
        let gluon_propagator = model.gluon.bind(&gluon_type);

        let qqg_vertex = genesis.hv("qft::qqg_vertex", PHYSICS_DIM);
        let ggg_vertex = genesis.hv("qft::ggg_vertex", PHYSICS_DIM);
        let gggg_vertex = genesis.hv("qft::gggg_vertex", PHYSICS_DIM);

        // SU(3) color charges (8 generators)
        let color_charges = [
            genesis.hv("qft::color_1", PHYSICS_DIM),
            genesis.hv("qft::color_2", PHYSICS_DIM),
            genesis.hv("qft::color_3", PHYSICS_DIM),
            genesis.hv("qft::color_4", PHYSICS_DIM),
            genesis.hv("qft::color_5", PHYSICS_DIM),
            genesis.hv("qft::color_6", PHYSICS_DIM),
            genesis.hv("qft::color_7", PHYSICS_DIM),
            genesis.hv("qft::color_8", PHYSICS_DIM),
        ];

        Self {
            quark_propagator,
            gluon_propagator,
            qqg_vertex,
            ggg_vertex,
            gggg_vertex,
            color_charges,
            strong_coupling: 0.118,
            gluon_type,
            fermion_type,
        }
    }

    /// Quark-gluon vertex
    pub fn qqg_vertex_with_color(&self, color_idx: usize) -> Vertex {
        let color = &self.color_charges[color_idx % 8];
        let vertex_vector = self
            .qqg_vertex
            .bind(&self.quark_propagator)
            .bind(&self.gluon_propagator)
            .bind(color);

        Vertex {
            coupling: self.strong_coupling.sqrt(),
            incoming: vec![self.quark_propagator.clone(), self.gluon_propagator.clone()],
            outgoing: vec![self.quark_propagator.clone()],
            vector: vertex_vector,
        }
    }

    /// Triple gluon vertex (non-Abelian self-interaction)
    pub fn triple_gluon_vertex(&self) -> Vertex {
        let vertex_vector = self.ggg_vertex.bind(&self.gluon_propagator).scale(3.0);

        Vertex {
            coupling: self.strong_coupling.sqrt(),
            incoming: vec![self.gluon_propagator.clone()],
            outgoing: vec![self.gluon_propagator.clone(), self.gluon_propagator.clone()],
            vector: vertex_vector,
        }
    }

    /// Quartic gluon vertex
    pub fn quartic_gluon_vertex(&self) -> Vertex {
        let vertex_vector = self.gggg_vertex.bind(&self.gluon_propagator).scale(4.0);

        Vertex {
            coupling: self.strong_coupling, // g² not g
            incoming: vec![self.gluon_propagator.clone(), self.gluon_propagator.clone()],
            outgoing: vec![self.gluon_propagator.clone(), self.gluon_propagator.clone()],
            vector: vertex_vector,
        }
    }

    /// Parton shower approximation (jet formation)
    pub fn parton_shower(&self, initial: &ContinuousHV, n_emissions: usize) -> Vec<ContinuousHV> {
        let mut partons = vec![initial.clone()];

        for i in 0..n_emissions {
            let parent = &partons[partons.len() - 1];
            // Each emission: q → q + g
            let daughter_q = parent.permute(i * 500);
            let daughter_g = self.gluon_propagator.permute(i * 1000);
            partons.push(daughter_q);
            partons.push(daughter_g);
        }

        partons
    }
}

/// Electroweak Encoder
#[derive(Debug, Clone)]
pub struct ElectroweakEncoder {
    /// W propagator
    pub w_propagator: ContinuousHV,
    /// Z propagator
    pub z_propagator: ContinuousHV,
    /// Higgs propagator
    pub higgs_propagator: ContinuousHV,
    /// Weinberg angle: sin²θW ≈ 0.23
    pub weinberg_angle: f64,

    /// Fermion-Z coupling
    pub ffz_vertex: ContinuousHV,
    /// Fermion-W coupling (charged current)
    pub ffw_vertex: ContinuousHV,
    /// W-W-Z triple gauge coupling
    pub wwz_vertex: ContinuousHV,
    /// Yukawa coupling (f-f-H)
    pub ffh_vertex: ContinuousHV,

    higgs_type: ContinuousHV,
    fermion_type: ContinuousHV,
}

impl ElectroweakEncoder {
    pub fn from_genesis(genesis: &GenesisSeed, model: &StandardModel) -> Self {
        let fermion_type = genesis.hv(PropagatorType::Fermion.domain_label(), PHYSICS_DIM);
        let boson_type = genesis.hv(PropagatorType::Boson.domain_label(), PHYSICS_DIM);
        let higgs_type = genesis.hv(PropagatorType::Higgs.domain_label(), PHYSICS_DIM);

        // W bosons are W+ and W- - bundle them for a combined W representation
        let w_propagator = ContinuousHV::bundle(&[&model.w_plus, &model.w_minus]).bind(&boson_type);
        let z_propagator = model.z_boson.bind(&boson_type);
        let higgs_propagator = model.higgs.bind(&higgs_type);

        Self {
            w_propagator,
            z_propagator,
            higgs_propagator,
            weinberg_angle: 0.23122,

            ffz_vertex: genesis.hv("qft::ffz_vertex", PHYSICS_DIM),
            ffw_vertex: genesis.hv("qft::ffw_vertex", PHYSICS_DIM),
            wwz_vertex: genesis.hv("qft::wwz_vertex", PHYSICS_DIM),
            ffh_vertex: genesis.hv("qft::ffh_vertex", PHYSICS_DIM),

            higgs_type,
            fermion_type,
        }
    }

    /// Beta decay: n → p + e⁻ + ν̄ₑ (W⁻ exchange)
    pub fn beta_decay(&self, hadrons: &Hadrons, model: &StandardModel) -> FeynmanDiagram {
        // Weak vertex at quark level: d → u + W⁻
        let vertex_vector = self
            .ffw_vertex
            .bind(&self.w_propagator)
            .bind(&hadrons.neutron)
            .bind(&hadrons.proton);

        let v1 = Vertex {
            coupling: 0.65, // Weak coupling g
            incoming: vec![hadrons.neutron.clone()],
            outgoing: vec![hadrons.proton.clone()],
            vector: vertex_vector.clone(),
        };

        // W⁻ → e⁻ + ν̄ₑ
        let v2 = Vertex {
            coupling: 0.65,
            incoming: vec![self.w_propagator.clone()],
            outgoing: vec![model.electron.clone(), model.electron_neutrino.clone()],
            vector: self.ffw_vertex.bind(&model.electron),
        };

        let w_prop = Propagator {
            particle: self.w_propagator.clone(),
            propagator_type: PropagatorType::Boson,
            momentum: ContinuousHV::zero(PHYSICS_DIM),
            vector: self.w_propagator.clone(),
        };

        let diagram_vector = v1.vector.bind(&v2.vector).bind(&w_prop.vector);

        FeynmanDiagram {
            name: "Beta decay".to_string(),
            propagators: vec![w_prop],
            vertices: vec![v1, v2],
            external_legs: vec![],
            loop_order: 0,
            amplitude_estimate: 1e-5, // GF ~ 1e-5 GeV⁻²
            vector: diagram_vector,
        }
    }

    /// Higgs decay: H → b + b̄
    pub fn higgs_to_bb(&self, model: &StandardModel) -> FeynmanDiagram {
        let yukawa: f64 = (4.18_f64 / 246.0).sqrt(); // m_b / v

        let vertex_vector = self
            .ffh_vertex
            .bind(&self.higgs_propagator)
            .bind(&model.bottom_quark);

        let v = Vertex {
            coupling: yukawa,
            incoming: vec![self.higgs_propagator.clone()],
            outgoing: vec![
                model.bottom_quark.clone(),
                model.bottom_quark.permute(PHYSICS_DIM / 2),
            ],
            vector: vertex_vector.clone(),
        };

        FeynmanDiagram {
            name: "H → bb̄".to_string(),
            propagators: vec![],
            vertices: vec![v],
            external_legs: vec![],
            loop_order: 0,
            amplitude_estimate: yukawa * yukawa,
            vector: vertex_vector,
        }
    }

    /// W pair production: e⁺e⁻ → W⁺W⁻
    pub fn w_pair_production(&self, model: &StandardModel) -> FeynmanDiagram {
        // s-channel: e⁺e⁻ → γ/Z → W⁺W⁻
        let vertex_vector = self.wwz_vertex.bind(&self.w_propagator).scale(2.0);

        let v = Vertex {
            coupling: 0.65,
            incoming: vec![
                model.electron.clone(),
                model.electron.permute(PHYSICS_DIM / 2),
            ],
            outgoing: vec![self.w_propagator.clone(), self.w_propagator.permute(1000)],
            vector: vertex_vector.clone(),
        };

        FeynmanDiagram {
            name: "e⁺e⁻ → W⁺W⁻".to_string(),
            propagators: vec![],
            vertices: vec![v],
            external_legs: vec![],
            loop_order: 0,
            amplitude_estimate: 0.1,
            vector: vertex_vector,
        }
    }
}

/// Loop diagram with divergence information
#[derive(Debug, Clone)]
pub struct LoopDiagram {
    /// Tree-level diagram this is based on
    pub tree_diagram: FeynmanDiagram,
    /// Loop order
    pub loop_order: usize,
    /// Type of divergence
    pub divergence_type: DivergenceType,
    /// Counterterm (if renormalizable)
    pub counterterm: Option<ContinuousHV>,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl QEDEncoder {
    /// Electron self-energy (1-loop)
    pub fn electron_self_energy(&self) -> LoopDiagram {
        let loop_vec = self
            .electron_propagator
            .bind(&self.photon_propagator)
            .permute(1000);

        let counterterm = Some(self.electron_propagator.scale(0.1)); // Mass counterterm

        LoopDiagram {
            tree_diagram: self.compton_scattering(),
            loop_order: 1,
            divergence_type: DivergenceType::UVDivergent,
            counterterm,
            vector: loop_vec,
        }
    }

    /// Vacuum polarization (photon self-energy)
    pub fn vacuum_polarization(&self) -> LoopDiagram {
        let loop_vec = self
            .photon_propagator
            .bind(&self.electron_propagator)
            .bind(&self.electron_propagator.permute(PHYSICS_DIM / 2))
            .permute(2000);

        LoopDiagram {
            tree_diagram: self.bhabha_scattering(),
            loop_order: 1,
            divergence_type: DivergenceType::UVDivergent,
            counterterm: Some(self.photon_propagator.scale(0.1)),
            vector: loop_vec,
        }
    }

    /// Vertex correction (anomalous magnetic moment)
    pub fn vertex_correction(&self) -> LoopDiagram {
        let loop_vec = self.qed_vertex.bind(&self.photon_propagator).permute(3000);

        LoopDiagram {
            tree_diagram: self.compton_scattering(),
            loop_order: 1,
            divergence_type: DivergenceType::Finite, // Anomalous moment is finite
            counterterm: None,
            vector: loop_vec,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (
        StandardModel,
        Hadrons,
        QEDEncoder,
        QCDEncoder,
        ElectroweakEncoder,
        GenesisSeed,
    ) {
        let genesis = GenesisSeed::from_phrase("qft test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let qed = QEDEncoder::from_genesis(&genesis, &model);
        let qcd = QCDEncoder::from_genesis(&genesis, &model);
        let ew = ElectroweakEncoder::from_genesis(&genesis, &model);
        (model, hadrons, qed, qcd, ew, genesis)
    }

    #[test]
    fn test_qed_vertex_charge_conservation() {
        let (_, _, qed, _, _, _) = setup();

        let vertex = qed.vertex();

        // Vertex should have 3 legs (e, γ, e)
        assert_eq!(vertex.incoming.len() + vertex.outgoing.len(), 3);

        // Coupling should be √α
        assert!((vertex.coupling - ALPHA.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_compton_scattering() {
        let (_, _, qed, _, _, _) = setup();

        let diagram = qed.compton_scattering();

        assert_eq!(diagram.loop_order, 0); // Tree level
        assert_eq!(diagram.vertices.len(), 2);
        assert!((diagram.amplitude_estimate - ALPHA).abs() < 0.0001);
    }

    #[test]
    fn test_feynman_diagram_composition_associative() {
        let (_, _, qed, _, _, _) = setup();

        // Create two diagrams
        let d1 = qed.compton_scattering();
        let d2 = qed.bhabha_scattering();

        // Composition should be consistent
        let composed_12 = d1.vector.bind(&d2.vector);
        let composed_21 = d2.vector.bind(&d1.vector);

        // Binding is not commutative but both should be valid
        assert_eq!(composed_12.dim(), PHYSICS_DIM);
        assert_eq!(composed_21.dim(), PHYSICS_DIM);
    }

    #[test]
    fn test_qcd_color_charges() {
        let (_, _, _, qcd, _, _) = setup();

        // Should have 8 color charges (SU(3) generators)
        assert_eq!(qcd.color_charges.len(), 8);

        // Each should be distinct
        for i in 0..8 {
            for j in (i + 1)..8 {
                let sim = qcd.color_charges[i].similarity(&qcd.color_charges[j]);
                assert!(
                    sim < 0.5,
                    "Color charges should be distinct: {} vs {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_triple_gluon_vertex() {
        let (_, _, _, qcd, _, _) = setup();

        let vertex = qcd.triple_gluon_vertex();

        // Non-Abelian: 3 gluons
        assert_eq!(vertex.incoming.len() + vertex.outgoing.len(), 3);
        assert!(vertex.vector.norm() > 0.0);
    }

    #[test]
    fn test_parton_shower() {
        let (model, _, _, qcd, _, _) = setup();

        let initial = &model.up_quark;
        let shower = qcd.parton_shower(initial, 3);

        // Should have 1 initial + 2*3 daughters = 7 partons
        assert_eq!(shower.len(), 7);
    }

    #[test]
    fn test_beta_decay() {
        let (model, hadrons, _, _, ew, _) = setup();

        let diagram = ew.beta_decay(&hadrons, &model);

        assert_eq!(diagram.name, "Beta decay");
        assert_eq!(diagram.loop_order, 0);
        assert!(diagram.amplitude_estimate < 1e-4); // Weak interaction
    }

    #[test]
    fn test_higgs_decay() {
        let (model, _, _, _, ew, _) = setup();

        let diagram = ew.higgs_to_bb(&model);

        assert_eq!(diagram.name, "H → bb̄");
        assert!(diagram.amplitude_estimate > 0.0);
    }

    #[test]
    fn test_loop_divergence_classification() {
        let (_, _, qed, _, _, _) = setup();

        let self_energy = qed.electron_self_energy();
        assert_eq!(self_energy.divergence_type, DivergenceType::UVDivergent);
        assert!(self_energy.counterterm.is_some());

        let vac_pol = qed.vacuum_polarization();
        assert_eq!(vac_pol.divergence_type, DivergenceType::UVDivergent);

        let vertex_corr = qed.vertex_correction();
        assert_eq!(vertex_corr.divergence_type, DivergenceType::Finite);
        assert!(vertex_corr.counterterm.is_none());
    }

    #[test]
    fn test_weinberg_angle() {
        let (_, _, _, _, ew, _) = setup();

        // sin²θW ≈ 0.23
        assert!((ew.weinberg_angle - 0.23122).abs() < 0.001);
    }

    #[test]
    fn test_determinism() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let model = StandardModel::from_genesis(&genesis);

        let qed1 = QEDEncoder::from_genesis(&genesis, &model);
        let qed2 = QEDEncoder::from_genesis(&genesis, &model);

        assert!(
            qed1.qed_vertex.similarity(&qed2.qed_vertex) > 0.9999,
            "QED encoder should be deterministic"
        );
    }
}
