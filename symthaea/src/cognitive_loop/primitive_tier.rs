// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Primitive-tier consciousness subsystem manager.
//!
//! Groups all `Option<T>` subsystems gated by `enable_primitive_consciousness` into a
//! single struct. Reduces CognitiveLoopService field count by ~34 and moves
//! construction logic out of the monolithic constructor.

use crate::consciousness::adaptive_reasoning::AdaptiveReasoner;
use crate::consciousness::affective_consciousness::{
    AffectiveConfig, AffectiveConsciousnessAnalyzer,
};
use crate::consciousness::causal_explanation::CausalExplainer;
use crate::consciousness::code_primitives::CodePrimitiveRouter;
use crate::consciousness::compositionality::{CompositionalityConfig, CompositionalityEngine};
use crate::consciousness::consciousness_holography::{
    HolographicConfig, HolographicConsciousnessAnalyzer,
};
use crate::consciousness::context_aware_evolution::ContextAwareOptimizer;
use crate::consciousness::differentiable::DifferentiableConsciousness;
use crate::consciousness::dissipative_consciousness::DissipativeConsciousness;
use crate::consciousness::empathic_unification::EmpathicUnification;
use crate::consciousness::epistemic_conflict::{ConflictDetector, TheoryCalibrator};
use crate::consciousness::evolution_bridge::EvolutionCoordinator;
use crate::consciousness::gis_integration::EpistemicDecisionGate;
use crate::consciousness::harmonics::{HarmonicField, HarmonicResolver};
// HarmoniesIntegrator removed — now solely owned by EthicsEngine
use crate::consciousness::hierarchical_ltc::HierarchicalLTC;
use crate::consciousness::meta_reasoning::MetaCognitiveReasoner;
use crate::consciousness::multi_objective_evolution::MultiObjectiveEvolution;
use crate::consciousness::phi_validation::PhiValidationFramework;
use crate::consciousness::primitive_composition_rules::CompositionRuleEngine;
use crate::consciousness::primitive_consciousness::ConsciousnessPrimitiveProcessor;
use crate::consciousness::primitive_lattice::PrimitiveLattice;
use crate::consciousness::primitive_reasoning::{PrimitiveReasoner, ReasonerConfig};
use crate::consciousness::semantic_value_embedder::{EmbedderConfig, SemanticValueEmbedder};
use crate::consciousness::synthetic_states::SyntheticStatesNSMGrounding;
use crate::consciousness::temporal_primitives::ConsciousnessTemporalAnalyzer;
// UnifiedValueEvaluator removed — now solely owned by EthicsEngine
use crate::consciousness::value_feedback_loop::ValueFeedbackLoop;

use super::CognitiveLoopConfig;

/// Groups all primitive-consciousness-gated subsystems.
///
/// Every field is `Option<T>` (except `primitive_validation_result` and `value_feedback`)
/// and is `None` when `enable_primitive_consciousness` is false.
pub(crate) struct PrimitiveTierManager {
    /// Primitive consciousness decomposition for explainable consciousness.
    pub primitive_processor: Option<ConsciousnessPrimitiveProcessor>,

    /// Temporal consciousness analyzer using Allen's Interval Algebra.
    pub temporal_analyzer: Option<ConsciousnessTemporalAnalyzer>,

    /// Lattice structure over the 9-tier primitive system.
    pub primitive_lattice: Option<PrimitiveLattice>,

    /// Compositionality engine: algebraic composition of primitives.
    pub compositionality_engine: Option<CompositionalityEngine>,

    // NOTE: UnifiedValueEvaluator removed — now solely owned by EthicsEngine.
    /// Harmonic field: tracks strength of each of the Seven Fiduciary Harmonics.
    pub harmonic_field: Option<HarmonicField>,

    /// Harmonic resolver: resolves conflicts between harmonics.
    pub harmonic_resolver: Option<HarmonicResolver>,

    /// Primitive reasoner: HDC-based analogical reasoning with concept binding.
    pub primitive_reasoner: Option<PrimitiveReasoner>,

    /// Adaptive reasoner: Q-learning-guided primitive selection.
    pub adaptive_reasoner: Option<AdaptiveReasoner>,

    /// Phi validation framework: empirical validation of Phi against synthetic states.
    pub phi_validation: Option<PhiValidationFramework>,

    /// Causal self-explanation: builds causal model of primitive->Phi relationships.
    pub causal_explainer: Option<CausalExplainer>,

    /// Context-aware optimizer: dynamic weighting by reasoning context.
    pub context_optimizer: Option<ContextAwareOptimizer>,

    /// Evolution coordinator: co-evolves primitives and architecture via Thompson sampling.
    pub evolution_coordinator: Option<EvolutionCoordinator>,

    // NOTE: HarmoniesIntegrator removed — now solely owned by EthicsEngine.
    /// Semantic value embedder: value-aligned embeddings grounded in primitive tiers.
    pub semantic_value_embedder: Option<SemanticValueEmbedder>,

    /// Dissipative consciousness: Prigogine thermodynamic model.
    pub dissipative_consciousness: Option<DissipativeConsciousness>,

    /// Epistemic conflict detector: multi-theory conflict analysis.
    pub epistemic_conflict_detector: Option<ConflictDetector>,

    /// Theory calibrator for reliability weighting.
    pub theory_calibrator: Option<TheoryCalibrator>,

    // NOTE: ConsciousnessEquationV2 removed — now solely owned by ConsciousnessEngine.
    /// Hierarchical LTC: local circuits + global integrator.
    pub hierarchical_ltc: Option<HierarchicalLTC>,

    /// Holographic consciousness analyzer: interference-based binding.
    pub holographic_analyzer: Option<HolographicConsciousnessAnalyzer>,

    /// Differentiable consciousness: gradient-based optimization.
    pub differentiable_consciousness: Option<DifferentiableConsciousness>,

    /// Affective consciousness analyzer: valence-arousal-dominance tracking.
    pub affective_consciousness: Option<AffectiveConsciousnessAnalyzer>,

    // NOTE: UnifiedConsciousnessPipeline removed — now solely owned by ConsciousnessEngine.
    // NOTE: MultiModalIntegrator removed — now solely owned by ConsciousnessEngine.
    /// Primitive composition rules: domain-specific HDC binding operators.
    pub composition_rule_engine: Option<CompositionRuleEngine>,

    /// Synthetic states NSM grounding: maps BinaryHV to consciousness states.
    pub synthetic_grounding: Option<SyntheticStatesNSMGrounding>,

    /// Epistemic decision gate: evaluates through Graceful Ignorance System.
    pub epistemic_gate: Option<EpistemicDecisionGate>,

    /// Meta-cognitive reasoner: self-reflective reasoning about reasoning.
    pub meta_cognitive_reasoner: Option<MetaCognitiveReasoner>,

    /// Code primitive router: consciousness-aware code reasoning.
    pub code_primitive_router: Option<CodePrimitiveRouter>,

    /// Empathic unification: resonant empathy via user state inference.
    pub empathic_unification: Option<EmpathicUnification>,

    /// Multi-objective evolution: Pareto-frontier consciousness optimization.
    pub multi_objective_evolution: Option<MultiObjectiveEvolution>,

    /// Grid encoder for spatial reasoning (ARC-style).
    pub grid_encoder: Option<symthaea_core::hdc::grid_encoder::GridEncoder>,

    /// Cached primitive validation results (one-shot at cycle 500).
    pub primitive_validation_result: Option<(f64, f64)>,

    /// Value feedback loop for TD-learning on moral alignment.
    pub value_feedback: ValueFeedbackLoop,
}

impl PrimitiveTierManager {
    /// Construct from config. All subsystems are `None` when primitive consciousness is disabled.
    pub(crate) fn new(config: &CognitiveLoopConfig, cfc_input_dim: usize) -> Self {
        let has_primitive = config.enable_primitive_consciousness;

        let primitive_processor = if has_primitive {
            Some(ConsciousnessPrimitiveProcessor::new())
        } else {
            None
        };

        // Temporal analyzer + lattice (co-gated with primitive processor)
        let (temporal_analyzer, primitive_lattice) =
            if let Some(ref processor) = primitive_processor {
                let analyzer = ConsciousnessTemporalAnalyzer::new(0.3);
                let lattice = PrimitiveLattice::from_primitive_system(processor.primitive_system());
                (Some(analyzer), Some(lattice))
            } else {
                (None, None)
            };

        // Compositionality + harmonics + reasoning + validation
        // NOTE: UnifiedValueEvaluator removed — now solely owned by EthicsEngine
        let (
            compositionality_engine,
            harmonic_field,
            harmonic_resolver,
            primitive_reasoner,
            adaptive_reasoner,
            phi_validation,
        ) = if has_primitive {
            let comp_engine = {
                let ps = std::sync::Arc::new(
                    symthaea_core::hdc::primitive_system::PrimitiveSystem::new(),
                );
                CompositionalityEngine::new(ps, CompositionalityConfig::default())
            };
            let hf = HarmonicField::new();
            let hr = HarmonicResolver::new();
            let pr = PrimitiveReasoner::new(ReasonerConfig::default());
            let ar =
                AdaptiveReasoner::new(symthaea_core::hdc::primitive_system::PrimitiveTier::NSM);
            let pv = PhiValidationFramework::new();
            (
                Some(comp_engine),
                Some(hf),
                Some(hr),
                Some(pr),
                Some(ar),
                Some(pv),
            )
        } else {
            (None, None, None, None, None, None)
        };

        // Causal self-explainer
        let causal_explainer = if has_primitive {
            Some(CausalExplainer::new())
        } else {
            None
        };

        // Composition rule engine
        let composition_rule_engine = if has_primitive {
            Some(CompositionRuleEngine::new())
        } else {
            None
        };

        // NOTE: HarmoniesIntegrator removed — now solely owned by EthicsEngine

        // Context-aware optimizer
        let context_optimizer = if has_primitive {
            ContextAwareOptimizer::new(
                crate::consciousness::primitive_evolution::EvolutionConfig::default(),
            )
            .map_err(|e| tracing::warn!(err = %e, "ContextAwareOptimizer init failed"))
            .ok()
        } else {
            None
        };

        // Evolution coordinator
        let evolution_coordinator = if has_primitive {
            Some(EvolutionCoordinator::default())
        } else {
            None
        };

        // Semantic value embedder
        let semantic_value_embedder = if has_primitive {
            Some(SemanticValueEmbedder::new(EmbedderConfig {
                dimension: cfc_input_dim,
                ..Default::default()
            }))
        } else {
            None
        };

        // Dissipative + epistemic conflict + hierarchical LTC
        // NOTE: ConsciousnessEquationV2 removed — now solely owned by ConsciousnessEngine
        let (
            dissipative_consciousness,
            epistemic_conflict_detector,
            theory_calibrator,
            hierarchical_ltc,
        ) = if has_primitive {
            let dc = DissipativeConsciousness::new();
            let cd = ConflictDetector::new();
            let tc = TheoryCalibrator::new();
            let hltc = HierarchicalLTC::minimal_network()
                .map_err(|e| tracing::warn!(err = %e, "HierarchicalLTC init failed"))
                .ok();
            (Some(dc), Some(cd), Some(tc), hltc)
        } else {
            (None, None, None, None)
        };

        // Holographic + differentiable + affective
        // NOTE: UnifiedConsciousnessPipeline + MultiModalIntegrator removed —
        //       now solely owned by ConsciousnessEngine
        let (holographic_analyzer, differentiable_consciousness, affective_consciousness) =
            if has_primitive {
                let ha = HolographicConsciousnessAnalyzer::new(HolographicConfig::default());
                let dc = DifferentiableConsciousness::new();
                let ac = AffectiveConsciousnessAnalyzer::new(AffectiveConfig::default());
                (Some(ha), Some(dc), Some(ac))
            } else {
                (None, None, None)
            };

        // Synthetic states + epistemic gate
        let (synthetic_grounding, epistemic_gate) = if has_primitive {
            let sg = SyntheticStatesNSMGrounding::new(
                symthaea_core::hdc::primitive_system::PrimitiveSystem::global(),
            );
            let eg = EpistemicDecisionGate::new();
            (Some(sg), Some(eg))
        } else {
            (None, None)
        };

        // Meta-cognitive reasoner
        let meta_cognitive_reasoner = if has_primitive {
            MetaCognitiveReasoner::new(
                crate::consciousness::primitive_evolution::EvolutionConfig::default(),
                crate::consciousness::meta_reasoning::MetaReasoningConfig::default(),
            )
            .map_err(|e| tracing::warn!(err = %e, "MetaCognitiveReasoner init failed"))
            .ok()
        } else {
            None
        };

        // Code primitive router
        let code_primitive_router = if has_primitive {
            let mut router = CodePrimitiveRouter::new(cfc_input_dim);
            router.cache_primitives();
            Some(router)
        } else {
            None
        };

        // Empathic unification
        let empathic_unification = if has_primitive {
            Some(EmpathicUnification::new())
        } else {
            None
        };

        // Grid encoder
        let grid_encoder = if has_primitive {
            Some(symthaea_core::hdc::grid_encoder::GridEncoder::new(
                cfc_input_dim,
                30,
                30,
                10,
                0x9E3779B9_7F4A7C15,
            ))
        } else {
            None
        };

        // Multi-objective evolution
        let multi_objective_evolution = if has_primitive {
            MultiObjectiveEvolution::new(
                crate::consciousness::primitive_evolution::EvolutionConfig::default(),
            )
            .map_err(|e| tracing::warn!(err = %e, "MultiObjectiveEvolution init failed"))
            .ok()
        } else {
            None
        };

        Self {
            primitive_processor,
            temporal_analyzer,
            primitive_lattice,
            compositionality_engine,
            harmonic_field,
            harmonic_resolver,
            primitive_reasoner,
            adaptive_reasoner,
            phi_validation,
            causal_explainer,
            context_optimizer,
            evolution_coordinator,
            semantic_value_embedder,
            dissipative_consciousness,
            epistemic_conflict_detector,
            theory_calibrator,
            hierarchical_ltc,
            holographic_analyzer,
            differentiable_consciousness,
            affective_consciousness,
            composition_rule_engine,
            synthetic_grounding,
            epistemic_gate,
            meta_cognitive_reasoner,
            code_primitive_router,
            empathic_unification,
            multi_objective_evolution,
            grid_encoder,
            primitive_validation_result: None,
            value_feedback: ValueFeedbackLoop::default(),
        }
    }

    /// Reset all subsystems to their initial state.
    pub(crate) fn reset(&mut self) {
        if let Some(ref mut pp) = self.primitive_processor {
            *pp = ConsciousnessPrimitiveProcessor::new();
        }
        self.primitive_validation_result = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_primitive_tier() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = false;
        let tier = PrimitiveTierManager::new(&config, 64);
        assert!(tier.primitive_processor.is_none());
        assert!(tier.temporal_analyzer.is_none());
        assert!(tier.compositionality_engine.is_none());
    }

    #[test]
    fn test_enabled_primitive_tier() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = true;
        let tier = PrimitiveTierManager::new(&config, 64);
        assert!(tier.primitive_processor.is_some());
        assert!(tier.temporal_analyzer.is_some());
        assert!(tier.compositionality_engine.is_some());
        assert!(tier.harmonic_field.is_some());
    }

    #[test]
    fn test_reset() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = true;
        let mut tier = PrimitiveTierManager::new(&config, 64);
        tier.primitive_validation_result = Some((0.1, 0.05));
        tier.reset();
        assert!(tier.primitive_validation_result.is_none());
        assert!(tier.primitive_processor.is_some()); // Still present, just reset
    }
}
