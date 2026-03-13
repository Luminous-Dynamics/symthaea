//! Constructor and backend selection for CognitiveLoopService.

use super::CycleCarryover;
use crate::brain::prefrontal::PrefrontalCortex;
use crate::causal::{CausalEnhancerConfig, CausalLoopEnhancer};
// AttentionSchema now owned by SelfModelTierManager
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
// NarrativeSelfModel + PredictiveSelfModel now owned by SelfModelTierManager
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
use crate::dynamics::cfc_coherence::CoherenceConfig;
use crate::exploration::SurpriseExplorationBridge;
use crate::hdc::moral_algebra::MoralAlgebra;
use crate::hdc::moral_parser::MoralParser;
use crate::hdc_ltc_bridge::HdcLtcBridge;
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::memory::memory_coordinator::{CoordinatorConfig, MemoryCoordinator};
use crate::memory::semantic_memory::SemanticMemory;
#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;
// VoiceFeedbackBridge now owned by VoiceCoherenceBridge
// MetaCognitiveLayer now owned by SelfModelTierManager
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
    EpisodicMemoryBridge, FlowState, GoalSystemBridge, LoopStats, TemporalBackend, ThalamicRouter,
    WorldModelBridge,
};

impl CognitiveLoopService {
    /// Create a new cognitive loop service
    pub fn new(mut config: CognitiveLoopConfig) -> Result<Self> {
        // Validate hard constraints (range errors)
        config.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
        // Validate threshold registry ordering invariants
        super::thresholds::validate();

        // Validate soft constraints (dependency warnings)
        let warnings = config.validate_dependencies();
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
            TemporalBackend::HierarchicalCfC => {
                // Create HierarchicalCfC with multi-scale temporal processing (PP-2)
                let mut hcfc_config = config.hierarchical_cfc_config.clone();
                // Ensure dimensions match CfC config for compatibility
                hcfc_config.input_dim = config.cfc_config.input_dim;
                hcfc_config.output_dim = config.cfc_config.num_neurons;
                let hcfc = crate::dynamics::hierarchical_cfc::HierarchicalCfC::new(hcfc_config);
                TemporalNetwork::HierarchicalCfC(hcfc)
            }
        };

        // Initialize voice-coherence bridge (CfC coherence + voice feedback + temporal signatures)
        let coherence_config = CoherenceConfig {
            base_learning_rate: config.cfc_config.learning_rate,
            ..Default::default()
        };
        let voice_coherence =
            super::voice_coherence_bridge::VoiceCoherenceBridge::new(coherence_config);

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

        // Build optional causal consciousness (reuses causal_enhancement flag)
        let causal_consciousness = if config.causal_enhancement {
            config.genesis_phrase.as_ref().map(|p| {
                crate::intelligence::CausalConsciousness::from_genesis(
                    &symthaea_core::genesis::GenesisSeed::from_phrase(p),
                    "causal_consciousness",
                    8,
                )
            })
        } else {
            None
        };

        // Extract timezone offset before config is moved
        let timezone_offset_hours = config.timezone_offset_hours;

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

        // Self-model subsystems now in SelfModelTierManager::new()

        // Build optional virtual body adapter
        let virtual_body = if config.enable_virtual_body {
            Some(super::virtual_body::VirtualBody::new())
        } else {
            None
        };

        // Predictive self + attention schema now in SelfModelTierManager::new()

        // Build optional GWT integration
        let mut gwt = if config.enable_gwt {
            Some(UnifiedGlobalWorkspace::new(UnifiedGWTConfig::default()))
        } else {
            None
        };

        // Register GWT handlers for memory and perception broadcast consumption.
        let gwt_memory_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let gwt_perception_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        if let Some(ref mut workspace) = gwt {
            let mf = gwt_memory_flag.clone();
            workspace.register_handler(
                "memory",
                Box::new(move |_| {
                    mf.store(true, std::sync::atomic::Ordering::Relaxed);
                }),
            );
            let pc = gwt_perception_count.clone();
            workspace.register_handler(
                "perception",
                Box::new(move |_| {
                    pc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }),
            );
        }

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

        // Clamp attestation buffer capacity to hard max of 256 to prevent unbounded growth.
        // This ensures both pre-allocation and eviction use the same bounded value.
        config.attestation_buffer_capacity = config.attestation_buffer_capacity.min(256);
        let attestation_buf_cap = config.attestation_buffer_capacity;

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
                    crate::consciousness::cross_modal_binding::CrossModalBindingConfig::default(),
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
                    crate::consciousness::phenomenal_binding::PhenomenalBindingConfig::default(),
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

        // Build all primitive-consciousness-gated subsystems as a single manager.
        let primitive_tier =
            super::primitive_tier::PrimitiveTierManager::new(&config, config.cfc_config.input_dim);

        // Build optional episodic replay (needs config fields before move)
        let phi_episodic_replay = if config.episodic_replay {
            Some(crate::memory::episodic_replay::EpisodicMemory::new(
                config.episodic_replay_config.clone(),
            ))
        } else {
            None
        };

        let enable_user_state = config.enable_user_state_inference;
        let enable_coherence_field = config.enable_coherence_field;
        #[cfg(feature = "nurture")]
        let enable_nurture_attachment = config.enable_nurture_attachment;
        let enable_resonator_recall = config.enable_resonator_recall;
        let resonator_cfc_input_dim = config.cfc_config.input_dim;
        let resonator_genesis_phrase = config.genesis_phrase.clone();
        let has_primitive_processor = primitive_tier.primitive_processor.is_some();

        // Somatic error bridge: infrastructure errors → felt interoceptive signals
        let (somatic_bridge_instance, pain_sender) =
            crate::infrastructure::somatic_error_bridge::SomaticErrorBridge::new();

        // SelfModelTierManager must be created before `config` is moved into the struct
        let self_model_tier = super::self_model_tier::SelfModelTierManager::new(&config);

        // Read config flags before `config` is moved into Self
        let enable_primitive_consciousness = config.enable_primitive_consciousness;

        let moral_anomaly_config = config.moral_anomaly_config.clone();

        #[cfg(feature = "semantic-encoder")]
        let enable_semantic_encoder = config.enable_semantic_encoder;

        #[cfg(feature = "vision-manifold")]
        let cross_manifold_predictor_init =
            if config.enable_cross_manifold_predictor && config.enable_vision_manifold {
                let cmp_seed = config
                    .genesis_phrase
                    .as_ref()
                    .map(|p| {
                        symthaea_core::genesis::GenesisSeed::from_phrase(p)
                            .domain("cognitive_loop::cross_manifold")
                            .gen::<u64>()
                    })
                    .unwrap_or(7_000_042);
                Some(symthaea_vision_manifold::CrossManifoldPredictor::new(
                    16_384, cmp_seed,
                ))
            } else {
                None
            };

        let substrate_manager = super::substrate_manager::SubstrateManager::new(&config);
        #[cfg(feature = "integrity")]
        let substrate_tau_for_integrity = substrate_manager.tau_factor;

        #[cfg(feature = "physics-bridge")]
        let physics_integration = if config.enable_physics_bridge {
            Some(super::physics_integration::PhysicsIntegration::new())
        } else {
            None
        };

        #[cfg(feature = "vision-manifold")]
        let vision_frame_width = config.vision_frame_width;
        #[cfg(feature = "vision-manifold")]
        let vision_frame_height = config.vision_frame_height;
        #[cfg(feature = "vision-manifold")]
        let vision_manifold_enabled = config.enable_vision_manifold;

        #[cfg(feature = "ssm_language")]
        let broca_checkpoint_path = config.broca_checkpoint_path.clone();
        #[cfg(feature = "ssm_language")]
        let broca_enabled = config.enable_broca_language;
        #[cfg(feature = "ssm_language")]
        let broca_genesis_phrase = config.genesis_phrase.clone();

        let enable_visualization = config.enable_visualization;
        let enable_soul_alignment = config.enable_soul_alignment;
        let enable_knowledge_engine = config.enable_knowledge_engine;
        let knowledge_graph_capacity = config.knowledge_graph_capacity;
        let knowledge_causal_capacity = config.knowledge_causal_capacity;
        let knowledge_search_top_k = config.knowledge_search_top_k;
        let knowledge_ontology_max = config.knowledge_ontology_max;

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
            voice_coherence,
            adaptive_behavior,
            prediction_confidence: 0.5_f64, // Start neutral
            flow_state: FlowState::default(),
            emotion_contagion: EmotionContagion::default(),
            curiosity_drive: CuriosityDrive::default(),
            // NOTE: self_reflection is now in self_model_tier
            // Mega-unified architecture components
            thalamic_router: ThalamicRouter::default(),
            unification_engine: ConsciousnessUnificationEngine::new(),
            cognitive_depth: CognitiveDepth::default(),
            fep: super::fep_module::FepModule {
                active_inference_bridge: ActiveInferenceBridge::with_defaults(),
                closed_learning_loop,
                episodic_memory: EpisodicMemoryBridge::default(),
                goal_system: GoalSystemBridge::new(),
                world_model: WorldModelBridge::default(),
                agent: ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
                    state_dim: 8,
                    obs_dim: 4,
                    num_actions: 4,
                    enable_td_learning: true,
                    ..Default::default()
                }),
                enhanced_bridge: EnhancedFEPBridge::new(
                    ActiveInferenceAgentConfig {
                        state_dim: 8,
                        obs_dim: 4,
                        num_actions: 8,
                        enable_td_learning: true,
                        ..Default::default()
                    },
                    4,
                ),
                learning_signal: 0.0,
                lr_boost: 1.0,
                surprise_bridge,
            },
            feedback_state: super::feedback_state::FeedbackState::new(),
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
                    max_iters: 50, // Real-time budget (default 100 too slow)
                    convergence_threshold: 0.995, // Slightly relaxed for speed
                    temperature: 0.1,
                    bipolar: true,
                };
                let mut mem = crate::dynamics::resonator::ResonatorMemory::new(res_config, 500);

                // Helper: generate deterministic random bipolar HV from seed
                let make_hv = |seed: u64, d: usize| -> Vec<f32> {
                    let mut state = seed ^ 0x9E3779B97F4A7C15; // xorshift64 seed-0 fix
                    (0..d)
                        .map(|_| {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            if state % 2 == 0 {
                                1.0
                            } else {
                                -1.0
                            }
                        })
                        .collect()
                };

                let seed_base: u64 = resonator_genesis_phrase
                    .as_ref()
                    .map(|p| {
                        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(p);
                        genesis.domain("resonator_memory").gen::<u64>()
                    })
                    .unwrap_or(0xBE50_0A70_0000_5EED);

                // Codebook 1: "semantic" — 8 proto-symbols, grows dynamically
                let mut semantic_cb = crate::dynamics::Codebook::new("semantic");
                for i in 0..8u64 {
                    semantic_cb.add(
                        &format!("proto_{i}"),
                        make_hv(seed_base.wrapping_add(i), dim),
                    );
                }
                mem.add_codebook(semantic_cb);

                // Codebook 2: "valence" — 3 fixed emotional poles
                let mut valence_cb = crate::dynamics::Codebook::new("valence");
                valence_cb.add("positive", make_hv(seed_base.wrapping_add(100), dim));
                valence_cb.add("neutral", make_hv(seed_base.wrapping_add(101), dim));
                valence_cb.add("negative", make_hv(seed_base.wrapping_add(102), dim));
                mem.add_codebook(valence_cb);

                // Codebook 3: "phi_level" — 3 consciousness tiers
                let mut phi_cb = crate::dynamics::Codebook::new("phi_level");
                phi_cb.add("low", make_hv(seed_base.wrapping_add(200), dim));
                phi_cb.add("medium", make_hv(seed_base.wrapping_add(201), dim));
                phi_cb.add("high", make_hv(seed_base.wrapping_add(202), dim));
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
            // Semantic encoder: background Qwen3 embedding channel + HdcBridge
            #[cfg(feature = "semantic-encoder")]
            semantic_embedding_channel: {
                if enable_semantic_encoder {
                    let qwen_config = symthaea_embeddings::Qwen3Config::simulated();
                    match symthaea_embeddings::channel::EmbeddingChannel::spawn(qwen_config) {
                        Ok(channel) => Some(channel),
                        Err(e) => {
                            tracing::warn!("Failed to spawn semantic encoder: {e}");
                            None
                        }
                    }
                } else {
                    None
                }
            },
            #[cfg(feature = "semantic-encoder")]
            semantic_hdc_bridge: {
                if enable_semantic_encoder {
                    Some(symthaea_embeddings::HdcBridge::for_qwen3())
                } else {
                    None
                }
            },
            #[cfg(feature = "semantic-encoder")]
            pending_semantic_rx: std::sync::Mutex::new(None),
            #[cfg(feature = "semantic-encoder")]
            last_semantic_continuous: None,
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
            // Moral parser + algebra now owned by EthicsEngine
            last_moral_judgment: None,

            // Primitive-Belief Bridge for tier-level prediction error learning
            primitive_belief_bridge: PrimitiveBeliefBridge::new(),
            prev_primitive_state: None,
            dream_engine,
            #[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
            dream_feedback_bridge:
                crate::consciousness::recursive_improvement::DreamFeedbackBridge::new(),
            consciousness_state: super::consciousness_state_manager::ConsciousnessStateManager {
                predictive_mind,
                cross_modal_binder,
                phi_attention_gate: Some(crate::attention::PhiAttentionGate::default_gate()),
                affective_bridge,
            },
            contextual_weights,
            phi_attention,
            negation_detector,
            primitive_tier,
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
            prefrontal,
            self_model_tier,
            gwt_mgr: super::gwt_manager::GwtManager::new(
                gwt,
                gwt_memory_flag,
                gwt_perception_count,
            ),
            consciousness_monitors: super::consciousness_monitor_tier::ConsciousnessMonitorTier {
                resonance: consciousness_resonance,
                quantum_coherence,
                temporal: temporal_consciousness,
                embodied: embodied_cognition,
                thermodynamics: consciousness_thermodynamics,
                phenomenal_binding,
                hierarchical_free_energy,
            },
            narrative_gwt,
            soul: if enable_soul_alignment {
                Some(crate::soul::Soul::new(crate::soul::SoulConfig {
                    dimension: symthaea_core::hdc::unified_hv::HDC_DIMENSION,
                    ..Default::default()
                }))
            } else {
                None
            },
            attention_visualizer: if enable_visualization {
                Some(crate::visualization::AttentionVisualizer::with_max_history(
                    500,
                ))
            } else {
                None
            },
            social_mgr: super::SocialManager::new(enable_primitive_consciousness),
            user_state: if enable_user_state {
                Some(crate::user_state_inference::UserStateInference::new())
            } else {
                None
            },
            coherence_field: if enable_coherence_field {
                Some(crate::physiology::CoherenceField::new())
            } else {
                None
            },
            virtual_body,
            #[cfg(feature = "nurture")]
            nurture_attachment: if enable_nurture_attachment {
                Some(super::nurture_bridge::NurtureAttachmentBridge::new())
            } else {
                None
            },
            #[cfg(feature = "vision-manifold")]
            vision_bridge: if vision_manifold_enabled {
                let vm_config = symthaea_vision_manifold::VisionConfig::default();
                Some(symthaea_vision_manifold::VisionBridge::new(
                    vm_config,
                    vision_frame_width,
                    vision_frame_height,
                ))
            } else {
                None
            },
            #[cfg(feature = "vision-manifold")]
            vision_frame_buffer: None,
            #[cfg(feature = "vision-manifold")]
            cross_manifold_predictor: cross_manifold_predictor_init,
            #[cfg(feature = "foveation")]
            foveation_manager: {
                let fov_config = symthaea_foveation::FoveationConfig::default();
                Some(std::sync::Mutex::new(
                    symthaea_foveation::FoveationManager::new(fov_config, 8),
                ))
            },
            #[cfg(feature = "ssm_language")]
            broca_manager: if broca_enabled {
                let genesis = broca_genesis_phrase
                    .as_deref()
                    .map(symthaea_core::genesis::GenesisSeed::from_phrase)
                    .unwrap_or_else(|| {
                        symthaea_core::genesis::GenesisSeed::from_phrase("symthaea-broca-default")
                    });
                Some(super::broca_bridge::BrocaManager::new(
                    &genesis,
                    symthaea_broca::BrocaConfig::default(),
                    broca_checkpoint_path.as_deref(),
                ))
            } else {
                None
            },
            #[cfg(feature = "ssm_language")]
            last_broca_text: None,
            #[cfg(feature = "canvas")]
            canvas_manager: Some(super::canvas_bridge::CanvasManager::new()),
            #[cfg(feature = "canvas")]
            last_canvas_svg: None,
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
            biorhythm_mgr: super::biorhythm_manager::BiorhythmManager::new(timezone_offset_hours),
            metrics_collector: Some(crate::infrastructure::MetricsCollector::new()),
            knowledge_manager: if enable_knowledge_engine {
                let km_config = crate::knowledge::manager::KnowledgeManagerConfig {
                    graph_capacity: knowledge_graph_capacity,
                    causal_capacity: knowledge_causal_capacity,
                    search_top_k: knowledge_search_top_k,
                    ontology_config: crate::knowledge::adaptive_ontology::AdaptiveOntologyConfig {
                        max_primitives: knowledge_ontology_max,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let mut km = crate::knowledge::KnowledgeManager::new(km_config);
                km.bootstrap_entities();
                Some(km)
            } else {
                None
            },
            experience_bus: Some(crate::experience::ExperienceBus::with_defaults()),
            #[cfg(feature = "school_learning")]
            school_bridge: None,
            causal_consciousness,
            thermodynamic_load: 0.0,
            mood_temperature: 1.0,
            neuromod: super::neuromod_manager::NeuromodManager::default(),
            somatic_bridge: somatic_bridge_instance,
            pain_tx: Some(pain_sender),
            subsystem_collector: super::subsystem_trait::OutputCollector::new(),
            last_snapshot: None,

            // ── Unified Engines (additive wiring — old fields remain) ────────
            consciousness_engine: {
                // SpectralMIPFinder is now solely owned by the engine (no top-level duplicate).
                let engine_smf =
                    symthaea_core::consciousness_metrics::SpectralMIPFinder::with_defaults();
                // Create fresh optional subsystems for the engine.
                let engine_mmi = if has_primitive_processor {
                    Some(
                        crate::consciousness::multi_modal_integration::MultiModalIntegrator::new(
                            crate::consciousness::multi_modal_integration::IntegrationConfig::default(),
                        ),
                    )
                } else {
                    None
                };
                let engine_eq = if has_primitive_processor {
                    Some(
                        crate::consciousness::consciousness_equation_v2::ConsciousnessEquationV2::new(),
                    )
                } else {
                    None
                };
                let engine_ucp = if has_primitive_processor {
                    crate::consciousness::unified_consciousness_pipeline::UnifiedConsciousnessPipeline::new(
                        crate::consciousness::unified_consciousness_pipeline::PipelineConfig::default(),
                    ).ok()
                } else {
                    None
                };
                super::consciousness_engine::ConsciousnessEngine::new(
                    engine_smf, engine_mmi, engine_eq, engine_ucp,
                )
            },
            substrate_manager,
            #[cfg(feature = "physics-bridge")]
            physics_integration,
            convergence_cycle: 0,
            ethics_engine: {
                let engine_mp = MoralParser::new();
                let engine_ma = MoralAlgebra::default_dim();
                let engine_ve = if has_primitive_processor {
                    Some(
                        crate::consciousness::unified_value_evaluator::UnifiedValueEvaluator::new(),
                    )
                } else {
                    None
                };
                let engine_hi = if has_primitive_processor {
                    Some(
                        crate::consciousness::harmonies_integration::HarmoniesIntegrator::new(
                            crate::consciousness::harmonies_integration::HarmoniesIntegrationConfig {
                                // Match moral algebra dim so HarmoniesIntegrator shares the
                                // HarmonyBasis with MoralTopology (dedup ~448KB of vectors)
                                // and evaluates proper text encodings, not compressed CfC state.
                                dimension: engine_ma.dim(),
                                ..Default::default()
                            },
                        ),
                    )
                } else {
                    None
                };
                // Dense HarmonyBasis: encode HARMONY_KEYWORDS via Qwen3 + HdcBridge
                // so basis vectors live in the same JL-projected semantic subspace as
                // scenario embeddings. Eliminates the n-gram ↔ contextual domain mismatch.
                #[cfg(feature = "semantic-encoder")]
                let dense_basis: Option<
                    std::sync::Arc<crate::hdc::harmony_basis::HarmonyBasis>,
                > = if enable_semantic_encoder {
                    Self::build_dense_harmony_basis(engine_ma.dim())
                } else {
                    None
                };
                #[cfg(not(feature = "semantic-encoder"))]
                let dense_basis: Option<
                    std::sync::Arc<crate::hdc::harmony_basis::HarmonyBasis>,
                > = None;

                super::ethics_engine::EthicsEngine::with_anomaly_config_and_basis(
                    engine_mp,
                    engine_ma,
                    engine_ve,
                    engine_hi,
                    moral_anomaly_config.clone(),
                    dense_basis,
                )
            },
            kosmic_song: crate::mycelix::KosmicSong::default(),
            drive_manager: super::managers::DriveManager::default(),
            memory_manager: super::managers::MemoryManager::default(),
            learning_manager: super::managers::LearningManager::default(),
            perception_manager: super::managers::PerceptionManager::default(),
            #[cfg(feature = "mycelix")]
            governance_mgr: super::managers::GovernanceManager::default(),
            cantor_broadcast_buffer: Vec::with_capacity(32),
            cantor_cleanup_engine: {
                use symthaea_core::hdc::cantor_resonator_cleanup::*;
                CantorCleanupEngine::with_codebook_capacity(
                    super::thresholds::CANTOR_CODEBOOK_MAX_ENTRIES,
                )
            },
            cantor_last_activation: 0.0,
            cantor_dream_surprise: 0.0,
            cantor_resonance_boost: 0.0,
            #[cfg(feature = "integrity")]
            integrity_manager: {
                let mut im = crate::integrity::IntegrityManager::new();
                // Register safety-critical thresholds for tamper detection (#5)
                im.register_safety_thresholds(&[
                    super::thresholds::MORAL_CONCERN_THRESHOLD,
                    super::thresholds::MORAL_BENEFIT_THRESHOLD,
                    super::thresholds::MORAL_CONCERN_EXPLORATION_DAMPEN,
                    super::thresholds::MORAL_CONCERN_PAUSE_BOOST,
                    super::thresholds::MORAL_BENEFIT_CONFIDENCE_BOOST,
                ]);
                im.register_consciousness_weights(&[
                    super::thresholds::DOMINANCE_FLOW_BASE,
                    super::thresholds::DOMINANCE_CONFIDENT,
                    super::thresholds::DOMINANCE_DEFAULT,
                    super::thresholds::POLICY_SOFT_THRESHOLD,
                ]);
                // Register baseline receptor sensitivities for tamper detection (#2)
                // Tolerance/withdrawal dynamics change these over time, but the baseline
                // (all 1.0 at startup) is the critical invariant — deviation at startup
                // means binary patching or memory corruption.
                im.register_receptor_sensitivities(&[
                    1.0, // dopamine
                    1.0, // noradrenaline
                    1.0, // serotonin
                    1.0, // acetylcholine
                    1.0, // GABA
                    1.0, // oxytocin
                    1.0, // glutamate
                    1.0, // adenosine
                    1.0, // endocannabinoid
                ]);
                // Register moral topology constants for tamper detection.
                // MORAL_DIM and harmony basis structure — distorting these
                // silently corrupts all moral evaluations.
                im.register_moral_topology_constants(&[
                    crate::hdc::moral_algebra::MORAL_DIM as f32,
                    symthaea_types::N_HARMONIES as f32,
                    // Harmony interaction constants (8 × weight 0.125 = normalized)
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                ]);
                // Register governance thresholds for integrity monitoring
                #[cfg(feature = "mycelix")]
                im.register_governance_thresholds();
                // Apply substrate tau factor for temporal consistency scaling (#3)
                im.set_substrate_tau_factor(substrate_tau_for_integrity);
                // Install panic hook for crash forensics — dumps integrity snapshot to disk
                crate::integrity::install_panic_hook();
                im
            },
            motor_output_bridge: None,
            pending_motor_request: None,
            last_motor_result: None,
            last_motor_phi: 0.0,
            math_service: super::math_service::MathService::new(),
            resonant_speech: crate::resonant_speech::ResonantSpeech::new(),
        })
    }

    /// Build a dense-encoded HarmonyBasis by encoding HARMONY_KEYWORDS through
    /// the Qwen3 embedder and projecting each via HdcBridge to ContinuousHV.
    ///
    /// One-time init cost (~8 embeddings). The result lives in the same
    /// JL-projected semantic subspace as runtime scenario embeddings.
    #[cfg(feature = "semantic-encoder")]
    fn build_dense_harmony_basis(
        dim: usize,
    ) -> Option<std::sync::Arc<crate::hdc::harmony_basis::HarmonyBasis>> {
        use crate::hdc::harmony_basis::{HarmonyBasis, HARMONY_KEYWORDS};
        use symthaea_core::hdc::ContinuousHV;
        use symthaea_types::N_HARMONIES;

        let qwen_config = symthaea_embeddings::Qwen3Config::simulated();
        let mut embedder = match symthaea_embeddings::Qwen3Embedder::new(qwen_config) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to create Qwen3 embedder for dense HarmonyBasis: {e}");
                return None;
            }
        };

        let bridge =
            symthaea_embeddings::HdcBridge::with_config(symthaea_embeddings::BridgeConfig {
                input_dim: symthaea_embeddings::QWEN3_DIMENSION,
                output_dim: dim,
                ..Default::default()
            });

        // Encode all 8 harmony keyword strings in batch
        let keyword_refs: Vec<&str> = HARMONY_KEYWORDS.to_vec();
        let batch_result = match embedder.embed_batch(&keyword_refs) {
            Ok(results) => results,
            Err(e) => {
                tracing::warn!("Failed to batch-encode harmony keywords: {e}");
                return None;
            }
        };

        if batch_result.len() != N_HARMONIES {
            tracing::warn!(
                "Expected {N_HARMONIES} embeddings, got {}; falling back to n-gram basis",
                batch_result.len()
            );
            return None;
        }

        // Project each dense embedding through HdcBridge → ContinuousHV
        let mut vectors: Vec<ContinuousHV> = Vec::with_capacity(N_HARMONIES);
        for result in &batch_result {
            let projected = bridge.project_continuous(&result.embedding);
            vectors.push(ContinuousHV::from_slice(&projected));
        }

        let arr: [ContinuousHV; N_HARMONIES] = vectors
            .try_into()
            .unwrap_or_else(|_| [(); N_HARMONIES].map(|_| ContinuousHV::zero(dim)));

        let basis = HarmonyBasis::with_dense_vectors(dim, arr);
        tracing::info!(
            "Dense HarmonyBasis built: {} keywords × {}D via Qwen3 + HdcBridge",
            N_HARMONIES,
            dim
        );

        Some(std::sync::Arc::new(basis))
    }

    /// Get the current temporal backend type
    pub fn temporal_backend(&self) -> TemporalBackend {
        self.temporal_network.backend_type()
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    // ── Default construction ──────────────────────────────────────────

    #[test]
    fn default_construction_succeeds() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default());
        assert!(
            service.is_ok(),
            "default config should construct successfully"
        );
    }

    #[test]
    fn default_prediction_confidence_is_neutral() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert!(
            (service.prediction_confidence() - 0.5).abs() < f32::EPSILON,
            "initial prediction_confidence should be 0.5, got {}",
            service.prediction_confidence()
        );
    }

    #[test]
    fn default_stats_are_zeroed() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        let stats = service.stats();
        assert_eq!(stats.total_cycles, 0);
        assert_eq!(stats.avg_prediction_error, 0.0);
    }

    #[test]
    fn default_cycle_count_is_zero() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert_eq!(service.stats().total_cycles, 0);
    }

    #[test]
    fn default_social_signals_are_defaults() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        // social_trust defaults to 0.5, cooperation_rate defaults to 0.0
        assert!((service.social_mgr.social.social_trust - 0.5).abs() < f32::EPSILON);
        assert!((service.social_mgr.social.social_cooperation_rate - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn default_external_reward_is_zero() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert!((service.social_mgr.social.external_reward - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn default_fep_learning_signal_is_zero() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert!((service.fep_learning_signal() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn default_relational_psi_is_zero() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert!((service.social_mgr.social.relational_psi - 0.0).abs() < f64::EPSILON);
    }

    // ── Backend selection ─────────────────────────────────────────────

    #[test]
    fn cfc_backend_selection() {
        let config = CognitiveLoopConfig::with_cfc();
        let service = CognitiveLoopService::new(config).unwrap();
        assert_eq!(service.temporal_backend(), TemporalBackend::CfC);
    }

    #[test]
    fn hdc_ltc_unified_backend_selection() {
        let config = CognitiveLoopConfig::with_hdc_ltc_unified();
        let service = CognitiveLoopService::new(config).unwrap();
        assert_eq!(service.temporal_backend(), TemporalBackend::HdcLtcUnified);
    }

    #[test]
    fn hdc_ltc_fast_backend_selection() {
        let config = CognitiveLoopConfig::with_hdc_ltc_fast();
        let service = CognitiveLoopService::new(config).unwrap();
        assert_eq!(service.temporal_backend(), TemporalBackend::HdcLtcUnified);
    }

    #[test]
    fn hdc_ltc_accurate_backend_selection() {
        let config = CognitiveLoopConfig::with_hdc_ltc_accurate();
        let service = CognitiveLoopService::new(config).unwrap();
        assert_eq!(service.temporal_backend(), TemporalBackend::HdcLtcUnified);
    }

    // ── Config validation passthrough ─────────────────────────────────

    #[test]
    fn invalid_config_rejected_zero_neurons() {
        let mut config = CognitiveLoopConfig::default();
        config.cfc_config.num_neurons = 0;
        let result = CognitiveLoopService::new(config);
        assert!(result.is_err(), "zero neurons should be rejected");
    }

    #[test]
    fn invalid_config_rejected_zero_buffer() {
        let mut config = CognitiveLoopConfig::default();
        config.buffer_size = 0;
        let result = CognitiveLoopService::new(config);
        assert!(result.is_err(), "zero buffer should be rejected");
    }

    #[test]
    fn invalid_config_rejected_negative_lr() {
        let mut config = CognitiveLoopConfig::default();
        config.cfc_config.learning_rate = -0.5;
        let result = CognitiveLoopService::new(config);
        assert!(result.is_err(), "negative learning rate should be rejected");
    }

    #[test]
    fn invalid_config_rejected_nan_threshold() {
        let mut config = CognitiveLoopConfig::default();
        config.learning_threshold = f32::NAN;
        let result = CognitiveLoopService::new(config);
        assert!(result.is_err(), "NaN learning threshold should be rejected");
    }

    // ── Genesis phrase ────────────────────────────────────────────────

    #[test]
    fn genesis_phrase_construction_succeeds() {
        let mut config = CognitiveLoopConfig::default();
        config.genesis_phrase = Some("We hold these truths".to_string());
        let service = CognitiveLoopService::new(config);
        assert!(service.is_ok(), "genesis phrase config should construct");
    }

    #[test]
    fn genesis_phrase_with_hdc_ltc_backend() {
        let mut config = CognitiveLoopConfig::with_hdc_ltc_unified();
        config.genesis_phrase = Some("deterministic seed phrase".to_string());
        let service = CognitiveLoopService::new(config);
        assert!(service.is_ok(), "genesis phrase + HdcLtc should construct");
    }

    // ── Optional subsystem gating ─────────────────────────────────────

    #[test]
    fn causal_enhancement_disabled_by_default() {
        let config = CognitiveLoopConfig::default();
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(!service.has_causal_structure());
        assert!(service.causal_graph().is_none());
    }

    #[test]
    fn causal_enhancement_enabled_creates_enhancer() {
        let mut config = CognitiveLoopConfig::default();
        config.causal_enhancement = true;
        let service = CognitiveLoopService::new(config).unwrap();
        // Causal enhancer exists but no structure discovered yet
        assert!(service.causal_stats().is_some());
        assert!(!service.has_causal_structure());
    }

    #[test]
    fn episodic_replay_disabled_by_default() {
        let config = CognitiveLoopConfig::default();
        let service = CognitiveLoopService::new(config).unwrap();
        assert_eq!(service.episodic_replay_count(), 0);
        assert!(service.episodic_replay_stats().is_none());
    }

    #[test]
    fn episodic_replay_enabled_creates_memory() {
        let mut config = CognitiveLoopConfig::default();
        config.episodic_replay = true;
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(service.episodic_replay_stats().is_some());
        assert_eq!(service.episodic_replay_count(), 0);
    }

    #[test]
    fn primitive_consciousness_disabled_means_no_subsystems() {
        let config = CognitiveLoopConfig::default();
        assert!(!config.enable_primitive_consciousness);
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(service.temporal_analyzer().is_none());
        assert!(service.primitive_lattice().is_none());
        assert!(service.compositionality_engine().is_none());
        assert!(service.value_evaluator().is_none());
        assert!(service.harmonic_field().is_none());
        assert!(service.primitive_reasoner().is_none());
        assert!(service.adaptive_reasoner().is_none());
        assert!(service.causal_explainer().is_none());
        assert!(service.epistemic_gate().is_none());
        assert!(service.meta_cognitive_reasoner().is_none());
        assert!(service.code_primitive_router().is_none());
    }

    #[test]
    fn primitive_consciousness_enabled_creates_subsystems() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = true;
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(service.temporal_analyzer().is_some());
        assert!(service.primitive_lattice().is_some());
        assert!(service.compositionality_engine().is_some());
        assert!(service.value_evaluator().is_some());
        assert!(service.harmonic_field().is_some());
        assert!(service.primitive_reasoner().is_some());
        assert!(service.epistemic_gate().is_some());
    }

    // ── Attestation buffer capacity ───────────────────────────────────

    #[test]
    fn attestation_buffer_clamped_to_256() {
        let mut config = CognitiveLoopConfig::default();
        config.attestation_buffer_capacity = 1000;
        let service = CognitiveLoopService::new(config).unwrap();
        // Constructor clamps to min(1000, 256) = 256
        assert!(service.config().attestation_buffer_capacity <= 256);
    }

    #[test]
    fn attestation_buffer_preserves_small_value() {
        let mut config = CognitiveLoopConfig::default();
        config.attestation_buffer_capacity = 32;
        let service = CognitiveLoopService::new(config).unwrap();
        assert_eq!(service.config().attestation_buffer_capacity, 32);
    }

    // ── Psi attestation ───────────────────────────────────────────────

    #[test]
    fn psi_attestation_initially_empty() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert_eq!(service.psi_attestation_count(), 0);
        assert!(service.latest_psi_attestation().is_none());
    }

    // ── Config preserved ──────────────────────────────────────────────

    #[test]
    fn config_accessor_returns_same_config() {
        let config = CognitiveLoopConfig::default();
        let service = CognitiveLoopService::new(config.clone()).unwrap();
        assert_eq!(
            service.config().learning_threshold,
            config.learning_threshold
        );
        assert_eq!(service.config().target_frequency, config.target_frequency);
        assert_eq!(service.config().temporal_backend, config.temporal_backend);
    }

    // ── ConsciousnessProfile-based construction ───────────────────────

    #[test]
    fn minimal_profile_construction() {
        let config =
            CognitiveLoopConfig::from_profile(super::super::config::ConsciousnessProfile::Minimal);
        let service = CognitiveLoopService::new(config);
        assert!(service.is_ok(), "Minimal profile should construct");
    }

    #[test]
    fn standard_profile_construction() {
        let config =
            CognitiveLoopConfig::from_profile(super::super::config::ConsciousnessProfile::Standard);
        let service = CognitiveLoopService::new(config);
        assert!(service.is_ok(), "Standard profile should construct");
    }

    #[test]
    fn full_profile_construction() {
        let config =
            CognitiveLoopConfig::from_profile(super::super::config::ConsciousnessProfile::Full);
        let service = CognitiveLoopService::new(config);
        assert!(service.is_ok(), "Full profile should construct");
    }

    #[test]
    fn research_profile_construction() {
        let config =
            CognitiveLoopConfig::from_profile(super::super::config::ConsciousnessProfile::Research);
        let service = CognitiveLoopService::new(config);
        assert!(service.is_ok(), "Research profile should construct");
    }

    // ── Phi-Dyad wiring (2A) ─────────────────────────────────────────

    #[test]
    fn phi_dyad_initialized_when_consciousness_enabled() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = true;
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(
            service.social_mgr.phi_dyad.is_some(),
            "phi_dyad should be Some when enable_primitive_consciousness=true"
        );
        assert!(
            service.social_mgr.partner_model.is_some(),
            "partner_model should be Some when enable_primitive_consciousness=true"
        );
    }

    #[test]
    fn phi_dyad_none_when_consciousness_disabled() {
        let config = CognitiveLoopConfig::default();
        assert!(!config.enable_primitive_consciousness);
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(
            service.social_mgr.phi_dyad.is_none(),
            "phi_dyad should be None when enable_primitive_consciousness=false"
        );
        assert!(
            service.social_mgr.partner_model.is_none(),
            "partner_model should be None when enable_primitive_consciousness=false"
        );
    }

    #[test]
    fn hv_ring_buffers_empty_at_construction() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = true;
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(service.social_mgr.recent_ai_hvs.is_empty());
        assert!(service.social_mgr.recent_input_hvs.is_empty());
    }
}
