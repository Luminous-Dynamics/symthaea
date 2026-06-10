// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::PrimitiveSystem;
use std::collections::HashMap;

use super::config::EntropyMethod;
use super::critical::TransitionOrder;
use super::free_energy::{EquilibriumStatus, FreeEnergyStatus};
use super::state::ConsciousnessPhase;

// ═══════════════════════════════════════════════════════════════════════════
// NSM PRIMITIVE GROUNDING FOR CONSCIOUSNESS THERMODYNAMICS
// ═══════════════════════════════════════════════════════════════════════════

/// NSM primitive grounding for entropy calculation methods.
///
/// Each entropy method is decomposed into Natural Semantic Metalanguage
/// primitives that capture its conceptual essence.
///
/// ## Entropy Method Semantics
///
/// - Shannon: probability-based information -> KNOW + NOT + MAYBE + MUCH
/// - VonNeumann: quantum state uncertainty -> KNOW + NOT + PART + SAME
/// - Renyi: generalized entropy -> KNOW + NOT + MAYBE + MORE
/// - KolmogorovSinai: dynamical unpredictability -> KNOW + NOT + AFTER + MOVE
#[derive(Debug, Clone)]
pub(crate) struct EntropyMethodPrimitiveGrounding {
    /// The entropy method being grounded
    pub method: EntropyMethod,

    /// NSM primitives composing this method's semantics
    pub nsm_primitives: Vec<String>,

    /// HDC encoding from bundled primitive vectors
    pub primitive_encoding: BinaryHV,

    /// Information theoretic emphasis: 0.0 (statistical) to 1.0 (dynamical)
    pub dynamical_emphasis: f32,

    /// Quantum emphasis: 0.0 (classical) to 1.0 (quantum)
    pub quantum_emphasis: f32,
}

impl EntropyMethodPrimitiveGrounding {
    /// Get NSM grounding for a specific entropy method
    pub(crate) fn for_method(method: EntropyMethod, primitive_system: &PrimitiveSystem) -> Self {
        let (primitives, dynamical_emphasis, quantum_emphasis) = match method {
            // Shannon: classical information theory
            EntropyMethod::Shannon => (
                vec!["NSM_KNOW", "NSM_NOT", "NSM_MAYBE", "NSM_MUCH"],
                0.0,
                0.0,
            ),

            // Von Neumann: quantum information
            EntropyMethod::VonNeumann => (
                vec!["NSM_KNOW", "NSM_NOT", "NSM_PART", "NSM_SAME"],
                0.2,
                1.0,
            ),

            // Renyi: generalized entropy family
            EntropyMethod::Renyi => (
                vec!["NSM_KNOW", "NSM_NOT", "NSM_MAYBE", "NSM_MORE"],
                0.3,
                0.3,
            ),

            // Kolmogorov-Sinai: dynamical systems chaos
            EntropyMethod::KolmogorovSinai => (
                vec!["NSM_KNOW", "NSM_NOT", "NSM_AFTER", "NSM_MOVE"],
                1.0,
                0.0,
            ),
        };

        let nsm_primitives: Vec<String> = primitives.iter().map(|s| s.to_string()).collect();

        let encodings: Vec<BinaryHV> = nsm_primitives
            .iter()
            .filter_map(|name| primitive_system.get(name).map(|p| p.encoding))
            .collect();

        let primitive_encoding = if encodings.is_empty() {
            BinaryHV::random(8000 + method as u64 * 100)
        } else {
            BinaryHV::bundle(&encodings)
        };

        Self {
            method,
            nsm_primitives,
            primitive_encoding,
            dynamical_emphasis,
            quantum_emphasis,
        }
    }

    /// Get all entropy method groundings
    pub(crate) fn all_groundings(
        primitive_system: &PrimitiveSystem,
    ) -> HashMap<EntropyMethod, Self> {
        [
            EntropyMethod::Shannon,
            EntropyMethod::VonNeumann,
            EntropyMethod::Renyi,
            EntropyMethod::KolmogorovSinai,
        ]
        .into_iter()
        .map(|m| (m, Self::for_method(m, primitive_system)))
        .collect()
    }

    /// Semantic formula representation
    pub(crate) fn semantic_formula(&self) -> String {
        self.nsm_primitives.join(" + ")
    }
}

/// NSM primitive grounding for consciousness phases.
///
/// Each phase of consciousness is mapped to NSM primitives that capture
/// its phenomenological character.
///
/// ## Consciousness Phase Semantics
///
/// - Frozen: rigid, stuck thinking -> NOT + MOVE + NOT + CHANGE + HARD
/// - Normal: everyday awareness -> THINK + KNOW + DO + NOW
/// - Critical: edge of chaos, creativity -> MAYBE + CHANGE + GOOD + BAD
/// - Chaotic: fragmented, overwhelmed -> VERY + MOVE + NOT + TOGETHER
/// - Flow: effortless action -> DO + GOOD + NOT + FEEL + BAD + MOVE
/// - Unified: deep integration -> ALL + TOGETHER + SAME + ONE + FEEL
#[derive(Debug, Clone)]
pub(crate) struct ConsciousnessPhasePrimitiveGrounding {
    /// The consciousness phase being grounded
    pub phase: ConsciousnessPhase,

    /// NSM primitives composing this phase's semantics
    pub nsm_primitives: Vec<String>,

    /// HDC encoding from bundled primitive vectors
    pub primitive_encoding: BinaryHV,

    /// Order level: 0.0 (chaotic) to 1.0 (frozen/rigid)
    pub order: f32,

    /// Creativity potential: 0.0 (low) to 1.0 (high)
    pub creativity: f32,

    /// Well-being: 0.0 (suffering) to 1.0 (flourishing)
    pub wellbeing: f32,
}

impl ConsciousnessPhasePrimitiveGrounding {
    /// Get NSM grounding for a specific consciousness phase
    pub(crate) fn for_phase(phase: ConsciousnessPhase, primitive_system: &PrimitiveSystem) -> Self {
        let (primitives, order, creativity, wellbeing) = match phase {
            // Frozen: rigid, stuck, over-ordered
            ConsciousnessPhase::Frozen => (
                vec!["NSM_NOT", "NSM_MOVE", "NSM_NOT", "NSM_CHANGE"],
                1.0,
                0.1,
                0.3,
            ),

            // Normal: everyday balanced consciousness
            ConsciousnessPhase::Normal => (
                vec!["NSM_THINK", "NSM_KNOW", "NSM_DO", "NSM_NOW"],
                0.5,
                0.4,
                0.6,
            ),

            // Critical: at the edge, maximum creativity
            ConsciousnessPhase::Critical => (
                vec!["NSM_MAYBE", "NSM_CHANGE", "NSM_GOOD", "NSM_BAD"],
                0.5,
                1.0,
                0.5,
            ),

            // Chaotic: fragmented, overwhelming
            ConsciousnessPhase::Chaotic => (
                vec!["NSM_VERY", "NSM_MOVE", "NSM_NOT", "NSM_TOGETHER"],
                0.0,
                0.3,
                0.2,
            ),

            // Flow: effortless doing
            ConsciousnessPhase::Flow => (
                vec!["NSM_DO", "NSM_GOOD", "NSM_MOVE", "NSM_NOT", "NSM_THINK"],
                0.4,
                0.8,
                0.95,
            ),

            // Unified: deep integration, oneness
            ConsciousnessPhase::Unified => (
                vec!["NSM_ALL", "NSM_TOGETHER", "NSM_SAME", "NSM_FEEL"],
                0.6,
                0.7,
                0.9,
            ),
        };

        let nsm_primitives: Vec<String> = primitives.iter().map(|s| s.to_string()).collect();

        let encodings: Vec<BinaryHV> = nsm_primitives
            .iter()
            .filter_map(|name| primitive_system.get(name).map(|p| p.encoding))
            .collect();

        let primitive_encoding = if encodings.is_empty() {
            BinaryHV::random(8100 + phase as u64 * 100)
        } else {
            BinaryHV::bundle(&encodings)
        };

        Self {
            phase,
            nsm_primitives,
            primitive_encoding,
            order,
            creativity,
            wellbeing,
        }
    }

    /// Get all consciousness phase groundings
    pub(crate) fn all_groundings(
        primitive_system: &PrimitiveSystem,
    ) -> HashMap<ConsciousnessPhase, Self> {
        [
            ConsciousnessPhase::Frozen,
            ConsciousnessPhase::Normal,
            ConsciousnessPhase::Critical,
            ConsciousnessPhase::Chaotic,
            ConsciousnessPhase::Flow,
            ConsciousnessPhase::Unified,
        ]
        .into_iter()
        .map(|p| (p, Self::for_phase(p, primitive_system)))
        .collect()
    }

    /// Semantic formula representation
    pub(crate) fn semantic_formula(&self) -> String {
        self.nsm_primitives.join(" + ")
    }

    /// Calculate similarity between two phases
    pub(crate) fn similarity(&self, other: &Self) -> f32 {
        self.primitive_encoding
            .similarity(&other.primitive_encoding)
    }
}

/// NSM primitive grounding for phase transition orders.
///
/// ## Transition Order Semantics
///
/// - FirstOrder: abrupt, discontinuous -> MOMENT + CHANGE + NOT + SAME
/// - SecondOrder: continuous, critical -> FOR_SOME_TIME + CHANGE + PART + SAME
/// - Crossover: smooth, gradual -> FOR_SOME_TIME + BECOME + SAME + MORE
#[derive(Debug, Clone)]
pub(crate) struct TransitionOrderPrimitiveGrounding {
    /// The transition order being grounded
    pub order: TransitionOrder,

    /// NSM primitives composing this order's semantics
    pub nsm_primitives: Vec<String>,

    /// HDC encoding from bundled primitive vectors
    pub primitive_encoding: BinaryHV,

    /// Discontinuity: 0.0 (smooth) to 1.0 (abrupt)
    pub discontinuity: f32,

    /// Critical behavior: 0.0 (none) to 1.0 (strongly critical)
    pub criticality: f32,
}

impl TransitionOrderPrimitiveGrounding {
    /// Get NSM grounding for a specific transition order
    pub(crate) fn for_order(order: TransitionOrder, primitive_system: &PrimitiveSystem) -> Self {
        let (primitives, discontinuity, criticality) = match order {
            // First order: sudden jump
            TransitionOrder::FirstOrder => (
                vec!["NSM_MOMENT", "NSM_CHANGE", "NSM_NOT", "NSM_SAME"],
                1.0,
                0.3,
            ),

            // Second order: continuous but singular
            TransitionOrder::SecondOrder => (
                vec!["NSM_FOR_SOME_TIME", "NSM_CHANGE", "NSM_PART", "NSM_SAME"],
                0.3,
                1.0,
            ),

            // Crossover: smooth transition
            TransitionOrder::Crossover => (
                vec!["NSM_FOR_SOME_TIME", "NSM_BECOME", "NSM_SAME", "NSM_MORE"],
                0.0,
                0.0,
            ),
        };

        let nsm_primitives: Vec<String> = primitives.iter().map(|s| s.to_string()).collect();

        let encodings: Vec<BinaryHV> = nsm_primitives
            .iter()
            .filter_map(|name| primitive_system.get(name).map(|p| p.encoding))
            .collect();

        let primitive_encoding = if encodings.is_empty() {
            BinaryHV::random(8200 + order as u64 * 100)
        } else {
            BinaryHV::bundle(&encodings)
        };

        Self {
            order,
            nsm_primitives,
            primitive_encoding,
            discontinuity,
            criticality,
        }
    }

    /// Get all transition order groundings
    pub(crate) fn all_groundings(
        primitive_system: &PrimitiveSystem,
    ) -> HashMap<TransitionOrder, Self> {
        [
            TransitionOrder::FirstOrder,
            TransitionOrder::SecondOrder,
            TransitionOrder::Crossover,
        ]
        .into_iter()
        .map(|o| (o, Self::for_order(o, primitive_system)))
        .collect()
    }

    /// Semantic formula representation
    pub(crate) fn semantic_formula(&self) -> String {
        self.nsm_primitives.join(" + ")
    }
}

/// NSM primitive grounding for free energy status.
///
/// ## Free Energy Status Semantics (Friston's Free Energy Principle)
///
/// - Minimizing: actively reducing -> DO + LESS + WANT + GOOD
/// - LocalMinimum: stuck local optimum -> SAME + NOT + MOVE + MAYBE + GOOD
/// - GlobalMinimum: best possible -> VERY + GOOD + SAME + NOT + CHANGE
/// - Increasing: losing coherence -> MORE + BAD + CHANGE + NOT + WANT
/// - Searching: exploring options -> MOVE + MAYBE + SEE + WANT + KNOW
#[derive(Debug, Clone)]
pub(crate) struct FreeEnergyStatusPrimitiveGrounding {
    /// The free energy status being grounded
    pub status: FreeEnergyStatus,

    /// NSM primitives composing this status's semantics
    pub nsm_primitives: Vec<String>,

    /// HDC encoding from bundled primitive vectors
    pub primitive_encoding: BinaryHV,

    /// Direction: -1.0 (minimizing) to 1.0 (increasing)
    pub direction: f32,

    /// Stability: 0.0 (dynamic) to 1.0 (stable)
    pub stability: f32,

    /// Optimality: 0.0 (poor) to 1.0 (optimal)
    pub optimality: f32,
}

impl FreeEnergyStatusPrimitiveGrounding {
    /// Get NSM grounding for a specific free energy status
    pub(crate) fn for_status(status: FreeEnergyStatus, primitive_system: &PrimitiveSystem) -> Self {
        let (primitives, direction, stability, optimality) = match status {
            // Minimizing: active reduction
            FreeEnergyStatus::Minimizing => (
                vec!["NSM_DO", "NSM_LESS", "NSM_WANT", "NSM_GOOD"],
                -1.0,
                0.3,
                0.5,
            ),

            // LocalMinimum: stuck at suboptimal
            FreeEnergyStatus::LocalMinimum => (
                vec!["NSM_SAME", "NSM_NOT", "NSM_MOVE", "NSM_MAYBE", "NSM_GOOD"],
                0.0,
                0.8,
                0.5,
            ),

            // GlobalMinimum: optimal state
            FreeEnergyStatus::GlobalMinimum => (
                vec!["NSM_VERY", "NSM_GOOD", "NSM_SAME", "NSM_NOT", "NSM_CHANGE"],
                0.0,
                1.0,
                1.0,
            ),

            // Increasing: deteriorating
            FreeEnergyStatus::Increasing => (
                vec!["NSM_MORE", "NSM_BAD", "NSM_CHANGE", "NSM_NOT", "NSM_WANT"],
                1.0,
                0.2,
                0.1,
            ),

            // Searching: exploring
            FreeEnergyStatus::Searching => (
                vec!["NSM_MOVE", "NSM_MAYBE", "NSM_SEE", "NSM_WANT", "NSM_KNOW"],
                0.0,
                0.1,
                0.3,
            ),
        };

        let nsm_primitives: Vec<String> = primitives.iter().map(|s| s.to_string()).collect();

        let encodings: Vec<BinaryHV> = nsm_primitives
            .iter()
            .filter_map(|name| primitive_system.get(name).map(|p| p.encoding))
            .collect();

        let primitive_encoding = if encodings.is_empty() {
            BinaryHV::random(8300 + status as u64 * 100)
        } else {
            BinaryHV::bundle(&encodings)
        };

        Self {
            status,
            nsm_primitives,
            primitive_encoding,
            direction,
            stability,
            optimality,
        }
    }

    /// Get all free energy status groundings
    pub(crate) fn all_groundings(
        primitive_system: &PrimitiveSystem,
    ) -> HashMap<FreeEnergyStatus, Self> {
        [
            FreeEnergyStatus::Minimizing,
            FreeEnergyStatus::LocalMinimum,
            FreeEnergyStatus::GlobalMinimum,
            FreeEnergyStatus::Increasing,
            FreeEnergyStatus::Searching,
        ]
        .into_iter()
        .map(|s| (s, Self::for_status(s, primitive_system)))
        .collect()
    }

    /// Semantic formula representation
    pub(crate) fn semantic_formula(&self) -> String {
        self.nsm_primitives.join(" + ")
    }
}

/// NSM primitive grounding for equilibrium status.
///
/// ## Equilibrium Status Semantics
///
/// - Equilibrium: balanced, stable -> SAME + NOT + CHANGE + NOW
/// - Equilibrating: approaching balance -> BECOME + SAME + FOR_SOME_TIME
/// - FarFromEquilibrium: active, living -> VERY + NOT + SAME + DO + LIVE
/// - Metastable: temporarily stable -> SAME + NOW + MAYBE + CHANGE + AFTER
#[derive(Debug, Clone)]
pub(crate) struct EquilibriumStatusPrimitiveGrounding {
    /// The equilibrium status being grounded
    pub status: EquilibriumStatus,

    /// NSM primitives composing this status's semantics
    pub nsm_primitives: Vec<String>,

    /// HDC encoding from bundled primitive vectors
    pub primitive_encoding: BinaryHV,

    /// Distance from equilibrium: 0.0 (at equilibrium) to 1.0 (far)
    pub distance: f32,

    /// Stability duration: 0.0 (transient) to 1.0 (permanent)
    pub permanence: f32,

    /// Activity level: 0.0 (quiescent) to 1.0 (highly active)
    pub activity: f32,
}

impl EquilibriumStatusPrimitiveGrounding {
    /// Get NSM grounding for a specific equilibrium status
    pub(crate) fn for_status(
        status: EquilibriumStatus,
        primitive_system: &PrimitiveSystem,
    ) -> Self {
        let (primitives, distance, permanence, activity) = match status {
            // Equilibrium: true balance
            EquilibriumStatus::Equilibrium => (
                vec!["NSM_SAME", "NSM_NOT", "NSM_CHANGE", "NSM_NOW"],
                0.0,
                1.0,
                0.0,
            ),

            // Equilibrating: approaching balance
            EquilibriumStatus::Equilibrating => (
                vec!["NSM_BECOME", "NSM_SAME", "NSM_FOR_SOME_TIME"],
                0.3,
                0.5,
                0.3,
            ),

            // FarFromEquilibrium: active living systems
            EquilibriumStatus::FarFromEquilibrium => (
                vec!["NSM_VERY", "NSM_NOT", "NSM_SAME", "NSM_DO", "NSM_LIVE"],
                1.0,
                0.2,
                1.0,
            ),

            // Metastable: temporary stability
            EquilibriumStatus::Metastable => (
                vec![
                    "NSM_SAME",
                    "NSM_NOW",
                    "NSM_MAYBE",
                    "NSM_CHANGE",
                    "NSM_AFTER",
                ],
                0.1,
                0.3,
                0.2,
            ),
        };

        let nsm_primitives: Vec<String> = primitives.iter().map(|s| s.to_string()).collect();

        let encodings: Vec<BinaryHV> = nsm_primitives
            .iter()
            .filter_map(|name| primitive_system.get(name).map(|p| p.encoding))
            .collect();

        let primitive_encoding = if encodings.is_empty() {
            BinaryHV::random(8400 + status as u64 * 100)
        } else {
            BinaryHV::bundle(&encodings)
        };

        Self {
            status,
            nsm_primitives,
            primitive_encoding,
            distance,
            permanence,
            activity,
        }
    }

    /// Get all equilibrium status groundings
    pub(crate) fn all_groundings(
        primitive_system: &PrimitiveSystem,
    ) -> HashMap<EquilibriumStatus, Self> {
        [
            EquilibriumStatus::Equilibrium,
            EquilibriumStatus::Equilibrating,
            EquilibriumStatus::FarFromEquilibrium,
            EquilibriumStatus::Metastable,
        ]
        .into_iter()
        .map(|s| (s, Self::for_status(s, primitive_system)))
        .collect()
    }

    /// Semantic formula representation
    pub(crate) fn semantic_formula(&self) -> String {
        self.nsm_primitives.join(" + ")
    }
}

/// Unified thermodynamics NSM grounding system.
///
/// Provides access to all thermodynamic concept groundings for
/// cross-domain semantic reasoning about consciousness states.
#[derive(Debug, Clone)]
pub(crate) struct ThermodynamicsNSMGrounding {
    /// Entropy method groundings
    pub entropy_methods: HashMap<EntropyMethod, EntropyMethodPrimitiveGrounding>,

    /// Consciousness phase groundings
    pub phases: HashMap<ConsciousnessPhase, ConsciousnessPhasePrimitiveGrounding>,

    /// Transition order groundings
    pub transition_orders: HashMap<TransitionOrder, TransitionOrderPrimitiveGrounding>,

    /// Free energy status groundings
    pub free_energy_statuses: HashMap<FreeEnergyStatus, FreeEnergyStatusPrimitiveGrounding>,

    /// Equilibrium status groundings
    pub equilibrium_statuses: HashMap<EquilibriumStatus, EquilibriumStatusPrimitiveGrounding>,
}

impl ThermodynamicsNSMGrounding {
    /// Create complete thermodynamics NSM grounding system
    pub(crate) fn new(primitive_system: &PrimitiveSystem) -> Self {
        Self {
            entropy_methods: EntropyMethodPrimitiveGrounding::all_groundings(primitive_system),
            phases: ConsciousnessPhasePrimitiveGrounding::all_groundings(primitive_system),
            transition_orders: TransitionOrderPrimitiveGrounding::all_groundings(primitive_system),
            free_energy_statuses: FreeEnergyStatusPrimitiveGrounding::all_groundings(
                primitive_system,
            ),
            equilibrium_statuses: EquilibriumStatusPrimitiveGrounding::all_groundings(
                primitive_system,
            ),
        }
    }

    /// Get total number of grounded concepts
    pub(crate) fn concept_count(&self) -> usize {
        self.entropy_methods.len()
            + self.phases.len()
            + self.transition_orders.len()
            + self.free_energy_statuses.len()
            + self.equilibrium_statuses.len()
    }

    /// Get semantic description of the thermodynamic state
    pub(crate) fn describe_state(
        &self,
        phase: ConsciousnessPhase,
        fe_status: FreeEnergyStatus,
        eq_status: EquilibriumStatus,
    ) -> String {
        let phase_formula = self
            .phases
            .get(&phase)
            .map(|g| g.semantic_formula())
            .unwrap_or_default();
        let fe_formula = self
            .free_energy_statuses
            .get(&fe_status)
            .map(|g| g.semantic_formula())
            .unwrap_or_default();
        let eq_formula = self
            .equilibrium_statuses
            .get(&eq_status)
            .map(|g| g.semantic_formula())
            .unwrap_or_default();

        format!("Phase[{phase_formula}] & Energy[{fe_formula}] & Balance[{eq_formula}]")
    }
}
