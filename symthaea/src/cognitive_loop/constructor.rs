//! Constructor and backend selection for CognitiveLoopService.

use super::CycleCarryover;
use crate::brain::prefrontal::PrefrontalCortex;
use crate::causal::{CausalEnhancerConfig, CausalLoopEnhancer};
use crate::consciousness::attention_schema::AttentionSchema;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::autopoietic_consciousness::{
    AutopoieticConfig, AutopoieticConsciousness,
};
use crate::consciousness::consciousness_resonance::{ResonanceAnalyzer, ResonanceConfig};
use crate::consciousness::consciousness_unification::ConsciousnessUnificationEngine;
use crate::consciousness::embodied_cognition::{EmbodiedConfig, EmbodiedConsciousnessAnalyzer};
#[cfg(feature = "full_consciousness")]
use crate::consciousness::enactive_cognition::EnactiveCognition;
use crate::consciousness::fep_active_inference::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, EnhancedFEPBridge,
};
use crate::consciousness::gwt_integration::{UnifiedGWTConfig, UnifiedGlobalWorkspace};
use crate::consciousness::master_consciousness_equation::MasterConsciousnessEquation;
use crate::consciousness::narrative_gwt_integration::NarrativeGWTIntegration;
use crate::consciousness::narrative_self::{NarrativeSelfConfig, NarrativeSelfModel};
use crate::consciousness::predictive_self::{PredictiveSelfConfig, PredictiveSelfModel};
use crate::consciousness::primitive_belief_bridge::PrimitiveBeliefBridge;
use crate::consciousness::primitive_discovery::{
    DiscoveryServiceConfig, PrimitiveDiscoveryService,
};
use crate::consciousness::quantum_coherence::QuantumCoherenceAnalyzer;
use crate::consciousness::stability_regime::StabilityRegimeProcessor;
use crate::consciousness::temporal_consciousness::{
    TemporalConsciousnessAnalyzer, TemporalConsciousnessConfig,
};
#[cfg(feature = "full_consciousness")]
use crate::consciousness::unified_living_mind::UnifiedLivingMind;
use crate::dynamics::cfc::CfCNetwork;
use crate::dynamics::cfc_coherence::{CfCCoherenceBridge, CoherenceConfig};
use crate::dynamics::temporal_signatures::{SignatureConfig, TemporalSignatureEncoder};
use crate::exploration::SurpriseExplorationBridge;
use crate::hdc::moral_algebra::MoralAlgebra;
use crate::hdc::moral_parser::MoralParser;
use crate::hdc_ltc_bridge::HdcLtcBridge;
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::memory::memory_coordinator::{CoordinatorConfig, MemoryCoordinator};
use crate::memory::semantic_memory::SemanticMemory;
#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;
use crate::voice::voice_feedback::{VoiceFeedbackBridge, VoiceFeedbackConfig};
use crate::wisdom::meta_cognition::MetaCognitiveLayer;
use anyhow::Result;
use rand::Rng;
use std::collections::VecDeque;
use std::time::Instant;
use symthaea_core::hdc::predictive_encoder::PredictiveHdcEncoder;

use super::temporal_network::TemporalNetwork;
use super::training::AsyncTrainerHandle;
use super::{
    ActiveInferenceBridge, AdaptiveBehavior, ClosedLearningLoop, CognitiveDepth,
    CognitiveLoopConfig, CognitiveLoopService, CuriosityDrive, EmotionContagion,
    EpisodicMemoryBridge, FlowState, GoalSystemBridge, LoopStats, SelfReflection, TemporalBackend,
    ThalamicRouter, WorldModelBridge,
};

impl CognitiveLoopService {
    /// Create a new cognitive loop service
    pub fn new(config: CognitiveLoopConfig) -> Result<Self> {
        // Validate configuration and log any dependency warnings
        let warnings = config.validate();
        for w in &warnings {
            tracing::warn!(target: "cognitive_loop::config", "{w}");
        }

        let encoder = PredictiveHdcEncoder::new(config.encoder_config.clone())?;

        // Create temporal network based on selected backend
        let temporal_network = match config.temporal_backend {
            TemporalBackend::CfC => {
                // Create CfC network with input_dim and num_neurons
                let cfc = if let Some(ref phrase) = config.genesis_phrase {
                    let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(phrase);
                    let net_config = crate::dynamics::cfc::CfCNetworkConfig {
                        input_dim: config.cfc_config.input_dim,
                        hidden_dim: config.cfc_config.num_neurons,
                        ..Default::default()
                    };
                    CfCNetwork::from_genesis(net_config, &genesis, "cognitive_loop::cfc")
                } else {
                    CfCNetwork::new_with_input(
                        config.cfc_config.input_dim,
                        config.cfc_config.num_neurons,
                    )
                };
                TemporalNetwork::CfC(cfc)
            }
            TemporalBackend::HdcLtcUnified => {
                // Create HdcLtcBridge with appropriate config
                let mut bridge_config = config.hdc_ltc_config.clone();
                // Ensure dimensions match CfC config for compatibility
                bridge_config.input_dim = config.cfc_config.input_dim;
                bridge_config.output_dim = config.cfc_config.num_neurons;
                let bridge = if let Some(ref phrase) = config.genesis_phrase {
                    let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(phrase);
                    HdcLtcBridge::from_genesis(bridge_config, &genesis)
                } else {
                    HdcLtcBridge::new(bridge_config)
                };
                TemporalNetwork::HdcLtc(bridge)
            }
        };

        // Initialize coherence bridge with learning rate from config
        let coherence_config = CoherenceConfig {
            base_learning_rate: config.cfc_config.learning_rate,
            ..Default::default()
        };
        let coherence_bridge = CfCCoherenceBridge::new(coherence_config);

        // Initialize voice feedback bridge
        let voice_feedback_bridge = VoiceFeedbackBridge::new(VoiceFeedbackConfig::default());

        // Initialize temporal signature encoder for consciousness pattern detection
        let temporal_signature_encoder = TemporalSignatureEncoder::new(SignatureConfig::default());

        // Initialize adaptive behavior with defaults
        let adaptive_behavior = AdaptiveBehavior::default();

        // Initialize closed learning loop with genesis-seeded RNG when available
        let closed_learning_loop = if let Some(ref phrase) = config.genesis_phrase {
            let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(phrase);
            ClosedLearningLoop::with_rng(genesis.domain("cognitive_loop::exploration"))
        } else {
            ClosedLearningLoop::default()
        };

        // Spawn background training thread when async_training is enabled and backend is CfC
        let async_trainer = if config.async_training {
            match &temporal_network {
                TemporalNetwork::CfC(cfc) => Some(AsyncTrainerHandle::spawn(cfc.clone())),
                _ => None,
            }
        } else {
            None
        };

        // Build optional causal enhancer (needs config fields before move)
        let causal_enhancer = if config.causal_enhancement {
            let causal_config = CausalEnhancerConfig {
                discovery_interval: config.causal_discovery_interval,
                seed: config
                    .genesis_phrase
                    .as_ref()
                    .map(|p| {
                        symthaea_core::genesis::GenesisSeed::from_phrase(p)
                            .domain("causal_enhancer")
                            .gen::<u64>()
                    })
                    .unwrap_or(42),
                ..Default::default()
            };
            Some(CausalLoopEnhancer::with_config(causal_config))
        } else {
            None
        };

        // Build optional surprise exploration bridge
        let surprise_bridge = if config.enable_surprise_exploration {
            Some(SurpriseExplorationBridge::new())
        } else {
            None
        };

        // Build optional prefrontal cortex
        let prefrontal = if config.enable_prefrontal {
            Some(PrefrontalCortex::default())
        } else {
            None
        };

        // Build optional meta-cognitive layer
        let meta_cognition = if config.enable_meta_cognition {
            Some(MetaCognitiveLayer::new())
        } else {
            None
        };

        // Build optional narrative self-model
        let narrative_self = if config.enable_narrative_self {
            Some(NarrativeSelfModel::new(NarrativeSelfConfig::default()))
        } else {
            None
        };

        // Build optional virtual body adapter
        let virtual_body = if config.enable_virtual_body {
            Some(super::virtual_body::VirtualBody::new())
        } else {
            None
        };

        // Build optional predictive self-model
        let predictive_self = if config.enable_predictive_self {
            Some(PredictiveSelfModel::new(PredictiveSelfConfig::default()))
        } else {
            None
        };

        // Build optional attention schema
        let attention_schema = if config.enable_attention_schema {
            Some(AttentionSchema::new())
        } else {
            None
        };

        // Build optional GWT integration
        let gwt = if config.enable_gwt {
            Some(UnifiedGlobalWorkspace::new(UnifiedGWTConfig::default()))
        } else {
            None
        };

        // Build optional consciousness resonance monitor
        let consciousness_resonance = if config.enable_resonance {
            Some(ResonanceAnalyzer::new(ResonanceConfig::default()))
        } else {
            None
        };

        // Build optional quantum coherence observer
        let quantum_coherence = if config.enable_quantum_coherence {
            Some(QuantumCoherenceAnalyzer::new(
                crate::consciousness::quantum_coherence::CoherenceConfig::default(),
            ))
        } else {
            None
        };

        // Build optional temporal consciousness analyzer
        let temporal_consciousness = if config.enable_temporal_consciousness {
            Some(TemporalConsciousnessAnalyzer::new(
                TemporalConsciousnessConfig::default(),
            ))
        } else {
            None
        };

        // Build optional embodied cognition analyzer
        let embodied_cognition = if config.enable_embodied_cognition {
            Some(EmbodiedConsciousnessAnalyzer::new(EmbodiedConfig::default()))
        } else {
            None
        };

        // Build optional narrative-GWT integration (consciousness governance capstone)
        let narrative_gwt = if config.enable_narrative_gwt {
            Some(NarrativeGWTIntegration::default_config())
        } else {
            None
        };

        // Capture attestation buffer capacity before config is moved into struct
        let attestation_buf_cap = config.attestation_buffer_capacity.min(256);

        // Build optional dream engine for counterfactual learning
        let dream_engine = if config.enable_dream_replay {
            Some(crate::consciousness::dream::DreamEngine::with_defaults())
        } else {
            None
        };

        // Build optional predictive processing mind
        let predictive_mind = if config.enable_predictive_processing {
            Some(
                crate::consciousness::predictive_processing::PredictiveMind::new(
                    crate::consciousness::predictive_processing::PredictiveConfig::default(),
                ),
            )
        } else {
            None
        };

        // Build optional cross-modal binder
        let cross_modal_binder = if config.enable_cross_modal_binding {
            Some(
                crate::consciousness::cross_modal_binding::CrossModalBinder::new(
                    crate::consciousness::cross_modal_binding::BindingConfig::default(),
                ),
            )
        } else {
            None
        };

        // Build optional affective bridge
        let affective_bridge = if config.enable_affective_bridge {
            Some(crate::brain::affective_bridge::AffectiveBridge::default())
        } else {
            None
        };

        // Build optional consciousness thermodynamics analyzer
        let consciousness_thermodynamics = if config.enable_consciousness_thermodynamics {
            Some(
                crate::consciousness::consciousness_thermodynamics::ConsciousnessThermodynamicsAnalyzer::new(
                    crate::consciousness::consciousness_thermodynamics::ThermodynamicsConfig::default(),
                ),
            )
        } else {
            None
        };

        // Build optional phenomenal binding analyzer
        let phenomenal_binding = if config.enable_phenomenal_binding {
            Some(
                crate::consciousness::phenomenal_binding::TemporalSynchronizationAnalyzer::new(
                    crate::consciousness::phenomenal_binding::BindingConfig::default(),
                ),
            )
        } else {
            None
        };

        // Build optional hierarchical free energy engine
        let hierarchical_free_energy = if config.enable_hierarchical_free_energy {
            Some(
                crate::consciousness::hierarchical_free_energy::HierarchicalFreeEnergy::new(
                    crate::consciousness::hierarchical_free_energy::HierarchicalFEConfig::default(),
                ),
            )
        } else {
            None
        };

        // Build optional contextual weights
        let contextual_weights = if config.enable_contextual_weights {
            Some(crate::consciousness::contextual_weights::ContextualWeights::new())
        } else {
            None
        };

        // Build optional Phi-weighted attention
        let phi_attention = if config.enable_phi_attention {
            Some(crate::consciousness::phi_attention::AdaptiveThresholds::new(100))
        } else {
            None
        };

        // Build optional negation detector
        let negation_detector = if config.enable_negation_detection {
            Some(crate::consciousness::negation_detector::NegationDetector::new())
        } else {
            None
        };

        // Build optional metacognitive monitor
        let metacognitive_monitor = if config.enable_metacognitive_monitoring {
            Some(crate::consciousness::metacognitive_monitoring::MetacognitiveMonitor::new(0.001))
        } else {
            None
        };

        // Build optional safety gateway (pre-cognitive safety veto)
        let safety_gateway = if config.enable_safety_gateway {
            Some(crate::safety::SafetyGateway::new())
        } else {
            None
        };

        // Build optional primitive consciousness processor
        let primitive_processor = if config.enable_primitive_consciousness {
            Some(crate::consciousness::primitive_consciousness::ConsciousnessPrimitiveProcessor::new())
        } else {
            None
        };

        // Build optional temporal analyzer + lattice (co-gated with primitive consciousness)
        let (temporal_analyzer, primitive_lattice) = if let Some(ref processor) = primitive_processor {
            let analyzer = crate::consciousness::temporal_primitives::ConsciousnessTemporalAnalyzer::new(0.3);
            let lattice = crate::consciousness::primitive_lattice::PrimitiveLattice::from_primitive_system(
                processor.primitive_system(),
            );
            (Some(analyzer), Some(lattice))
        } else {
            (None, None)
        };

        // Build optional compositionality engine + value evaluator + harmonics + reasoning + validation
        // (all co-gated with primitive consciousness)
        let (compositionality_engine, value_evaluator, harmonic_field, harmonic_resolver,
             primitive_reasoner, adaptive_reasoner, phi_validation) =
            if primitive_processor.is_some() {
                let comp_engine = {
                    let ps = std::sync::Arc::new(
                        symthaea_core::hdc::primitive_system::PrimitiveSystem::new(),
                    );
                    crate::consciousness::compositionality::CompositionalityEngine::new(
                        ps,
                        crate::consciousness::compositionality::CompositionalityConfig::default(),
                    )
                };
                let val_eval = crate::consciousness::unified_value_evaluator::UnifiedValueEvaluator::new();
                let hf = crate::consciousness::harmonics::HarmonicField::new();
                let hr = crate::consciousness::harmonics::HarmonicResolver::new();
                let pr = crate::consciousness::primitive_reasoning::PrimitiveReasoner::new(
                    crate::consciousness::primitive_reasoning::ReasonerConfig::default(),
                );
                let ar = crate::consciousness::adaptive_reasoning::AdaptiveReasoner::new(
                    symthaea_core::hdc::primitive_system::PrimitiveTier::NSM,
                );
                let pv = crate::consciousness::phi_validation::PhiValidationFramework::new();
                (Some(comp_engine), Some(val_eval), Some(hf), Some(hr), Some(pr), Some(ar), Some(pv))
            } else {
                (None, None, None, None, None, None, None)
            };

        // Build optional causal self-explainer (co-gated with primitive consciousness)
        let causal_explainer = if primitive_processor.is_some() {
            Some(crate::consciousness::causal_explanation::CausalExplainer::new())
        } else {
            None
        };

        // Build optional composition rule engine (co-gated with primitive consciousness)
        let composition_rule_engine = if primitive_processor.is_some() {
            Some(crate::consciousness::primitive_composition_rules::CompositionRuleEngine::new())
        } else {
            None
        };

        // Build optional harmonies integrator (co-gated with primitive consciousness)
        let harmonies_integrator = if primitive_processor.is_some() {
            Some(crate::consciousness::harmonies_integration::HarmoniesIntegrator::new(
                crate::consciousness::harmonies_integration::HarmoniesIntegrationConfig {
                    dimension: config.cfc_config.input_dim,
                    ..Default::default()
                },
            ))
        } else {
            None
        };

        // Build optional context-aware optimizer (co-gated with primitive consciousness)
        let context_optimizer = if primitive_processor.is_some() {
            crate::consciousness::context_aware_evolution::ContextAwareOptimizer::new(
                crate::consciousness::primitive_evolution::EvolutionConfig::default(),
            )
            .ok()
        } else {
            None
        };

        // Build optional evolution coordinator (co-gated with primitive consciousness)
        let evolution_coordinator = if primitive_processor.is_some() {
            Some(crate::consciousness::evolution_bridge::EvolutionCoordinator::default())
        } else {
            None
        };

        // Build optional semantic value embedder at CfC input dimension (co-gated with primitive consciousness)
        let semantic_value_embedder = if primitive_processor.is_some() {
            Some(crate::consciousness::semantic_value_embedder::SemanticValueEmbedder::new(
                crate::consciousness::semantic_value_embedder::EmbedderConfig {
                    dimension: config.cfc_config.input_dim,
                    ..Default::default()
                },
            ))
        } else {
            None
        };

        // Build optional dissipative consciousness + epistemic conflict + equation v2 + hierarchical LTC
        // (all co-gated with primitive consciousness — NO feature gate)
        let (dissipative_consciousness, epistemic_conflict_detector, theory_calibrator,
             consciousness_equation_v2, hierarchical_ltc) =
            if primitive_processor.is_some() {
                let dc = crate::consciousness::dissipative_consciousness::DissipativeConsciousness::new();
                let cd = crate::consciousness::epistemic_conflict::ConflictDetector::new();
                let tc = crate::consciousness::epistemic_conflict::TheoryCalibrator::new();
                let eq = crate::consciousness::consciousness_equation_v2::ConsciousnessEquationV2::new();
                let hltc = crate::consciousness::hierarchical_ltc::HierarchicalLTC::minimal_network()
                    .ok();
                (Some(dc), Some(cd), Some(tc), Some(eq), hltc)
            } else {
                (None, None, None, None, None)
            };

        // Build optional holographic + differentiable + affective + pipeline + multi-modal
        // (all co-gated with primitive consciousness)
        let (holographic_analyzer, differentiable_consciousness, affective_consciousness,
             unified_consciousness_pipeline, multi_modal_integrator) =
            if primitive_processor.is_some() {
                let ha = crate::consciousness::consciousness_holography::HolographicConsciousnessAnalyzer::new(
                    crate::consciousness::consciousness_holography::HolographicConfig::default(),
                );
                let dc = crate::consciousness::differentiable::DifferentiableConsciousness::new();
                let ac = crate::consciousness::affective_consciousness::AffectiveConsciousnessAnalyzer::new(
                    crate::consciousness::affective_consciousness::AffectiveConfig::default(),
                );
                let ucp = crate::consciousness::unified_consciousness_pipeline::UnifiedConsciousnessPipeline::new(
                    crate::consciousness::unified_consciousness_pipeline::PipelineConfig::default(),
                ).ok(); // Returns Result — use .ok() for graceful degradation
                let mmi = crate::consciousness::multi_modal_integration::MultiModalIntegrator::new(
                    crate::consciousness::multi_modal_integration::IntegrationConfig::default(),
                );
                (Some(ha), Some(dc), Some(ac), ucp, Some(mmi))
            } else {
                (None, None, None, None, None)
            };

        // Build optional synthetic states NSM grounding + epistemic gate
        // (co-gated with primitive consciousness)
        let (synthetic_grounding, epistemic_gate) = if primitive_processor.is_some() {
            let sg = crate::consciousness::synthetic_states::SyntheticStatesNSMGrounding::new(
                symthaea_core::hdc::primitive_system::PrimitiveSystem::global(),
            );
            let eg = crate::consciousness::gis_integration::EpistemicDecisionGate::new();
            (Some(sg), Some(eg))
        } else {
            (None, None)
        };

        // Build optional meta-cognitive reasoner (co-gated with primitive consciousness)
        let meta_cognitive_reasoner = if primitive_processor.is_some() {
            crate::consciousness::meta_reasoning::MetaCognitiveReasoner::new(
                crate::consciousness::primitive_evolution::EvolutionConfig::default(),
                crate::consciousness::meta_reasoning::MetaReasoningConfig::default(),
            )
            .ok()
        } else {
            None
        };

        // Build optional code primitive router (co-gated with primitive consciousness)
        let code_primitive_router = if primitive_processor.is_some() {
            let mut router = crate::consciousness::code_primitives::CodePrimitiveRouter::new(
                config.cfc_config.input_dim,
            );
            router.cache_primitives();
            Some(router)
        } else {
            None
        };

        // Build optional empathic unification (co-gated with primitive consciousness)
        let empathic_unification = if primitive_processor.is_some() {
            Some(crate::consciousness::empathic_unification::EmpathicUnification::new())
        } else {
            None
        };

        // Build optional multi-objective evolution (co-gated with primitive consciousness)
        let multi_objective_evolution = if primitive_processor.is_some() {
            crate::consciousness::multi_objective_evolution::MultiObjectiveEvolution::new(
                crate::consciousness::primitive_evolution::EvolutionConfig::default(),
            )
            .ok()
        } else {
            None
        };

        // Build optional episodic replay (needs config fields before move)
        let phi_episodic_replay = if config.episodic_replay {
            Some(crate::memory::episodic_replay::EpisodicMemory::new(
                config.episodic_replay_config.clone(),
            ))
        } else {
            None
        };

        let enable_user_state = config.enable_user_state_inference;
        let enable_resonator_recall = config.enable_resonator_recall;
        let resonator_cfc_input_dim = config.cfc_config.input_dim;
        let resonator_genesis_phrase = config.genesis_phrase.clone();

        Ok(Self {
            config,
            encoder,
            temporal_network,
            buffer: VecDeque::with_capacity(1000),
            stats: LoopStats::default(),
            error_history: VecDeque::with_capacity(100),
            last_state: None,
            last_prediction: None,
            start_time: Instant::now(),
            is_consolidating: false,
            coherence_bridge,
            voice_feedback_bridge,
            temporal_signature_encoder,
            adaptive_behavior,
            prediction_confidence: 0.5, // Start neutral
            flow_state: FlowState::default(),
            emotion_contagion: EmotionContagion::default(),
            curiosity_drive: CuriosityDrive::default(),
            self_reflection: SelfReflection::default(),
            // Mega-unified architecture components
            thalamic_router: ThalamicRouter::default(),
            unification_engine: ConsciousnessUnificationEngine::new(),
            cognitive_depth: CognitiveDepth::default(),
            active_inference_bridge: ActiveInferenceBridge::with_defaults(),
            closed_learning_loop,
            // Memory system bridges
            episodic_memory: EpisodicMemoryBridge::default(),
            goal_system: GoalSystemBridge::new(),
            world_model: WorldModelBridge::default(),
            // FEP Active Inference Agent
            fep_agent: ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
                state_dim: 8,
                obs_dim: 4,
                num_actions: 4,
                enable_td_learning: true,
                ..Default::default()
            }),
            // Enhanced FEP Bridge with motor system (8 motor command types, 4D proprioceptive state)
            enhanced_fep_bridge: EnhancedFEPBridge::new(
                ActiveInferenceAgentConfig {
                    state_dim: 8,
                    obs_dim: 4,
                    num_actions: 8, // Matches MotorCommandType variants
                    enable_td_learning: true,
                    ..Default::default()
                },
                4, // Motor state dimension
            ),
            fep_learning_signal: 0.0,
            fep_lr_boost: 1.0,
            coherence_tracker: ConversationCoherenceTracker::new(0.3),
            stability_regime: StabilityRegimeProcessor::new(),
            discovery_service: PrimitiveDiscoveryService::new(DiscoveryServiceConfig::default()),
            // Semantic memory: HDC-based similarity lookup for CfC context
            // 1000 entries, 0.3 similarity threshold
            semantic_memory: SemanticMemory::with_threshold(1000, 0.3),
            // Memory coordinator: cross-tier signal broadcaster
            memory_coordinator: MemoryCoordinator::new(CoordinatorConfig::default()),
            // Resonator Memory: factorized episodic recall with 3 codebooks
            resonator_memory: if enable_resonator_recall {
                let dim = resonator_cfc_input_dim; // matches compressed_state
                let res_config = crate::dynamics::resonator::ResonatorConfig {
                    dim,
                    max_iters: 50,               // Real-time budget (default 100 too slow)
                    convergence_threshold: 0.995, // Slightly relaxed for speed
                    temperature: 0.1,
                    bipolar: true,
                };
                let mut mem = crate::dynamics::resonator::ResonatorMemory::new(res_config, 500);

                // Helper: generate deterministic random bipolar HV from seed
                let make_hv = |seed: u64, d: usize| -> Vec<f32> {
                    let mut state = seed ^ 0x9E3779B97F4A7C15; // xorshift64 seed-0 fix
                    (0..d).map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        if state % 2 == 0 { 1.0 } else { -1.0 }
                    }).collect()
                };

                let seed_base: u64 = resonator_genesis_phrase.as_ref()
                    .map(|p| {
                        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(p);
                        genesis.domain("resonator_memory").gen::<u64>()
                    })
                    .unwrap_or(0xBE50_0A70_0000_5EED);

                // Codebook 1: "semantic" — 8 proto-symbols, grows dynamically
                let mut semantic_cb = crate::dynamics::Codebook::new("semantic");
                for i in 0..8u64 {
                    semantic_cb.add(&format!("proto_{i}"), make_hv(seed_base.wrapping_add(i), dim));
                }
                mem.add_codebook(semantic_cb);

                // Codebook 2: "valence" — 3 fixed emotional poles
                let mut valence_cb = crate::dynamics::Codebook::new("valence");
                valence_cb.add("positive", make_hv(seed_base.wrapping_add(100), dim));
                valence_cb.add("neutral",  make_hv(seed_base.wrapping_add(101), dim));
                valence_cb.add("negative", make_hv(seed_base.wrapping_add(102), dim));
                mem.add_codebook(valence_cb);

                // Codebook 3: "phi_level" — 3 consciousness tiers
                let mut phi_cb = crate::dynamics::Codebook::new("phi_level");
                phi_cb.add("low",    make_hv(seed_base.wrapping_add(200), dim));
                phi_cb.add("medium", make_hv(seed_base.wrapping_add(201), dim));
                phi_cb.add("high",   make_hv(seed_base.wrapping_add(202), dim));
                mem.add_codebook(phi_cb);

                Some(mem)
            } else {
                None
            },
            #[cfg(feature = "neural-bridge")]
            neural_bridge: {
                let probe_path = std::path::Path::new("models/neural_bridge/probe_weights.npy");
                if probe_path.exists() {
                    match NeuralBridge::load(probe_path) {
                        Ok(nb) => {
                            tracing::info!(
                                input_dim = nb.input_dim(),
                                "Neural bridge loaded from {}",
                                probe_path.display()
                            );
                            Some(nb)
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load neural bridge: {e}");
                            None
                        }
                    }
                } else {
                    tracing::debug!(
                        "No probe weights at {}, neural bridge disabled",
                        probe_path.display()
                    );
                    None
                }
            },
            async_trainer,
            causal_enhancer,
            phi_episodic_replay,
            #[cfg(feature = "reasoning_engine")]
            reasoning_engine: Some(
                crate::consciousness::reasoning_engine::ConsciousReasoningEngine::new(),
            ),
            // MFDI Bridge for identity verification and signed outputs
            #[cfg(feature = "identity")]
            mfdi_bridge: crate::identity::MfdiBridge::new(
                crate::identity::MfdiBridgeConfig::default(),
            ),

            // Metacognitive monitoring for Phi trajectory anomaly detection
            metacognitive_monitor,
            // Safety gateway for pre-cognitive safety veto
            safety_gateway,
            // Moral Algebra for compositional ethical reasoning
            moral_algebra: MoralAlgebra::default_dim(),
            moral_parser: MoralParser::new(),
            last_moral_judgment: None,

            // Primitive-Belief Bridge for tier-level prediction error learning
            primitive_belief_bridge: PrimitiveBeliefBridge::new(),
            prev_primitive_state: None,
            dream_engine,
            predictive_mind,
            cross_modal_binder,
            affective_bridge,
            consciousness_thermodynamics,
            phenomenal_binding,
            hierarchical_free_energy,
            contextual_weights,
            phi_attention,
            negation_detector,
            primitive_processor,
            temporal_analyzer,
            primitive_lattice,
            compositionality_engine,
            value_evaluator,
            harmonic_field,
            harmonic_resolver,
            primitive_reasoner,
            adaptive_reasoner,
            phi_validation,
            causal_explainer,
            composition_rule_engine,
            harmonies_integrator,
            context_optimizer,
            evolution_coordinator,
            semantic_value_embedder,
            dissipative_consciousness,
            epistemic_conflict_detector,
            theory_calibrator,
            consciousness_equation_v2,
            hierarchical_ltc,
            synthetic_grounding,
            epistemic_gate,
            meta_cognitive_reasoner,
            code_primitive_router,
            empathic_unification,
            multi_objective_evolution,
            holographic_analyzer,
            differentiable_consciousness,
            affective_consciousness,
            unified_consciousness_pipeline,
            multi_modal_integrator,
            value_feedback: crate::consciousness::value_feedback_loop::ValueFeedbackLoop::default(),
            #[cfg(feature = "support")]
            support_predictive_engine: Some(symthaea_support::predictive::PredictiveEngine::new()),
            #[cfg(feature = "support")]
            support_knowledge_manager: Some(symthaea_support::knowledge::KnowledgeManager::new()),
            #[cfg(feature = "support")]
            support_triage_engine: Some(symthaea_support::triage::TriageEngine::new()),
            #[cfg(feature = "support")]
            support_privacy_manager: Some(symthaea_support::privacy::PrivacyManager::default()),
            #[cfg(feature = "support")]
            support_action_engine: Some(symthaea_support::actions::ActionEngine::new()),
            #[cfg(feature = "support")]
            support_cycle_counter: 0,
            carryover: CycleCarryover::default(),
            surprise_bridge,
            prefrontal,
            meta_cognition,
            narrative_self,
            predictive_self,
            attention_schema,
            gwt,
            consciousness_resonance,
            quantum_coherence,
            temporal_consciousness,
            embodied_cognition,
            narrative_gwt,
            spectral_mip_finder: symthaea_core::consciousness_metrics::SpectralMIPFinder::with_defaults(),
            soul: Some(crate::soul::Soul::new(crate::soul::SoulConfig {
                dimension: symthaea_core::hdc::unified_hv::HDC_DIMENSION,
                ..Default::default()
            })),
            attention_visualizer: Some(crate::visualization::AttentionVisualizer::new()),
            relational_psi: 0.0,
            external_reward: 0.0,
            social_trust: 0.5,
            social_cooperation_rate: 0.0,
            user_state: if enable_user_state {
                Some(crate::user_state_inference::UserStateInference::new())
            } else {
                None
            },
            virtual_body,
            psi_attestation_buffer: std::collections::VecDeque::with_capacity(attestation_buf_cap),
            policy_agreement_window: std::collections::VecDeque::with_capacity(20),
            master_equation: MasterConsciousnessEquation::default(),
            // Unified Living Mind: life-mind continuity (full_consciousness only)
            #[cfg(feature = "full_consciousness")]
            unified_living_mind: UnifiedLivingMind::new(),
            #[cfg(feature = "full_consciousness")]
            autopoietic: {
                let mut ap = AutopoieticConsciousness::new(AutopoieticConfig::default());
                ap.initialize(); // Bootstrap boundary + processing + memory + self-model components
                ap
            },
            #[cfg(feature = "full_consciousness")]
            enactive: EnactiveCognition::new(),
            biorhythm: crate::chronobiology::Biorhythm::current(),
            biorhythm_refresh_counter: 0,
            phi_attention_gate: Some(crate::attention::PhiAttentionGate::default_gate()),
            metrics_collector: Some(crate::infrastructure::MetricsCollector::new()),
            experience_bus: Some(crate::experience::ExperienceBus::with_defaults()),
        })
    }

    /// Get the current temporal backend type
    pub fn temporal_backend(&self) -> TemporalBackend {
        self.temporal_network.backend_type()
    }
}
