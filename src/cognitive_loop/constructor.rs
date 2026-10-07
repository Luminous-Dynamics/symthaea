// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

// Bridge adapters removed — all platforms now implement EmbodimentBridge directly
// in their own crate (see each platform's embodiment.rs).

/// Optional consciousness monitors and ethics analyzers built from config flags.
///
/// All fields are `Option<T>` gated by the corresponding `config.enable_*` flag.
/// Extracted from the constructor to reduce its complexity (~165 LOC → 1 call).
struct OptionalMonitors {
    gwt: Option<UnifiedGlobalWorkspace>,
    gwt_memory_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    gwt_perception_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    consciousness_resonance: Option<ResonanceAnalyzer>,
    quantum_coherence: Option<QuantumCoherenceAnalyzer>,
    temporal_consciousness: Option<TemporalConsciousnessAnalyzer>,
    embodied_cognition: Option<EmbodiedConsciousnessAnalyzer>,
    narrative_gwt: Option<NarrativeGWTIntegration>,
    dream_engine: Option<crate::consciousness::dream::DreamEngine>,
    predictive_mind: Option<crate::consciousness::predictive_processing::PredictiveMind>,
    cross_modal_binder: Option<crate::consciousness::cross_modal_binding::CrossModalBinder>,
    affective_bridge: Option<crate::brain::affective_bridge::AffectiveBridge>,
    consciousness_thermodynamics: Option<
        crate::consciousness::consciousness_thermodynamics::ConsciousnessThermodynamicsAnalyzer,
    >,
    phenomenal_binding:
        Option<crate::consciousness::phenomenal_binding::TemporalSynchronizationAnalyzer>,
    hierarchical_free_energy:
        Option<crate::consciousness::hierarchical_free_energy::HierarchicalFreeEnergy>,
    contextual_weights: Option<crate::consciousness::contextual_weights::ContextualWeights>,
    phi_attention: Option<crate::consciousness::phi_attention::AdaptiveThresholds>,
    negation_detector: Option<crate::consciousness::negation_detector::NegationDetector>,
    metacognitive_monitor:
        Option<crate::consciousness::metacognitive_monitoring::MetacognitiveMonitor>,
    safety_gateway: Option<crate::safety::SafetyGateway>,
}

fn build_optional_monitors(config: &CognitiveLoopConfig) -> OptionalMonitors {
    // GWT workspace
    let mut gwt = if config.enable_gwt {
        Some(UnifiedGlobalWorkspace::new(UnifiedGWTConfig::default()))
    } else {
        None
    };
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

    // Consciousness monitors
    let consciousness_resonance = if config.enable_resonance {
        Some(ResonanceAnalyzer::new(ResonanceConfig::default()))
    } else {
        None
    };
    let quantum_coherence = if config.enable_quantum_coherence {
        Some(QuantumCoherenceAnalyzer::new(
            crate::consciousness::quantum_coherence::CoherenceConfig::default(),
        ))
    } else {
        None
    };
    let temporal_consciousness = if config.enable_temporal_consciousness {
        Some(TemporalConsciousnessAnalyzer::new(
            TemporalConsciousnessConfig::default(),
        ))
    } else {
        None
    };
    let embodied_cognition = if config.enable_embodied_cognition {
        Some(EmbodiedConsciousnessAnalyzer::new(EmbodiedConfig::default()))
    } else {
        None
    };
    let narrative_gwt = if config.enable_narrative_gwt {
        Some(NarrativeGWTIntegration::default_config())
    } else {
        None
    };
    let dream_engine = if config.enable_dream_replay {
        Some(crate::consciousness::dream::DreamEngine::with_defaults())
    } else {
        None
    };
    let predictive_mind = if config.enable_predictive_processing {
        Some(
            crate::consciousness::predictive_processing::PredictiveMind::new(
                crate::consciousness::predictive_processing::PredictiveConfig::default(),
            ),
        )
    } else {
        None
    };
    let cross_modal_binder = if config.enable_cross_modal_binding {
        Some(
            crate::consciousness::cross_modal_binding::CrossModalBinder::new(
                crate::consciousness::cross_modal_binding::CrossModalBindingConfig::default(),
            ),
        )
    } else {
        None
    };
    let affective_bridge = if config.enable_affective_bridge {
        Some(crate::brain::affective_bridge::AffectiveBridge::default())
    } else {
        None
    };
    let consciousness_thermodynamics = if config.enable_consciousness_thermodynamics {
        Some(crate::consciousness::consciousness_thermodynamics::ConsciousnessThermodynamicsAnalyzer::new(
            crate::consciousness::consciousness_thermodynamics::ThermodynamicsConfig::default(),
        ))
    } else {
        None
    };
    let phenomenal_binding = if config.enable_phenomenal_binding {
        Some(
            crate::consciousness::phenomenal_binding::TemporalSynchronizationAnalyzer::new(
                crate::consciousness::phenomenal_binding::PhenomenalBindingConfig::default(),
            ),
        )
    } else {
        None
    };
    let hierarchical_free_energy = if config.enable_hierarchical_free_energy {
        Some(
            crate::consciousness::hierarchical_free_energy::HierarchicalFreeEnergy::new(
                crate::consciousness::hierarchical_free_energy::HierarchicalFEConfig::default(),
            ),
        )
    } else {
        None
    };
    let contextual_weights = if config.enable_contextual_weights {
        Some(crate::consciousness::contextual_weights::ContextualWeights::new())
    } else {
        None
    };
    let phi_attention = if config.enable_phi_attention {
        Some(crate::consciousness::phi_attention::AdaptiveThresholds::new(100))
    } else {
        None
    };
    let negation_detector = if config.enable_negation_detection {
        Some(crate::consciousness::negation_detector::NegationDetector::new())
    } else {
        None
    };
    let metacognitive_monitor = if config.enable_metacognitive_monitoring {
        Some(crate::consciousness::metacognitive_monitoring::MetacognitiveMonitor::new(0.001))
    } else {
        None
    };
    let safety_gateway = if config.enable_safety_gateway {
        Some(crate::safety::SafetyGateway::new())
    } else {
        None
    };

    OptionalMonitors {
        gwt,
        gwt_memory_flag,
        gwt_perception_count,
        consciousness_resonance,
        quantum_coherence,
        temporal_consciousness,
        embodied_cognition,
        narrative_gwt,
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
        metacognitive_monitor,
        safety_gateway,
    }
}

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
                            .r#gen::<u64>()
                    })
                    .unwrap_or_else(|| {
                        tracing::debug!("CausalEnhancer: using default seed (no genesis phrase)");
                        super::thresholds::CAUSAL_ENHANCER_SEED_DEFAULT
                    }),
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

        // Build optional virtual body adapter
        let virtual_body = if config.enable_virtual_body {
            Some(super::virtual_body::VirtualBody::new())
        } else {
            None
        };

        // Build all optional consciousness monitors, GWT, and safety gateway from config flags.
        let monitors = build_optional_monitors(&config);
        let gwt = monitors.gwt;
        let gwt_memory_flag = monitors.gwt_memory_flag;
        let gwt_perception_count = monitors.gwt_perception_count;
        let consciousness_resonance = monitors.consciousness_resonance;
        let quantum_coherence = monitors.quantum_coherence;
        let temporal_consciousness = monitors.temporal_consciousness;
        let embodied_cognition = monitors.embodied_cognition;
        let narrative_gwt = monitors.narrative_gwt;

        // Clamp attestation buffer capacity to hard max of 256 to prevent unbounded growth.
        // This ensures both pre-allocation and eviction use the same bounded value.
        config.attestation_buffer_capacity = config.attestation_buffer_capacity.min(256);
        let attestation_buf_cap = config.attestation_buffer_capacity;

        let dream_engine = monitors.dream_engine;
        let predictive_mind = monitors.predictive_mind;
        let cross_modal_binder = monitors.cross_modal_binder;
        let affective_bridge = monitors.affective_bridge;
        let consciousness_thermodynamics = monitors.consciousness_thermodynamics;
        let phenomenal_binding = monitors.phenomenal_binding;
        let hierarchical_free_energy = monitors.hierarchical_free_energy;
        let contextual_weights = monitors.contextual_weights;
        let phi_attention = monitors.phi_attention;
        let negation_detector = monitors.negation_detector;
        let metacognitive_monitor = monitors.metacognitive_monitor;

        let safety_gateway = monitors.safety_gateway;

        // Build all primitive-consciousness-gated subsystems as a single manager.
        let primitive_tier =
            super::primitive_tier::PrimitiveTierManager::new(&config, config.cfc_config.input_dim);

        // Always create episodic replay
        let phi_episodic_replay = if config.memory_graduation || config.episodic_replay_training {
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
        let enable_visualization = config.enable_visualization;
        let resonator_cfc_input_dim = config.cfc_config.input_dim;
        let resonator_genesis_phrase = config.genesis_phrase.clone();
        let has_primitive_processor = primitive_tier.primitive_processor.is_some();

        // Somatic error bridge
        let (somatic_bridge_instance, pain_sender) =
            crate::infrastructure::somatic_error_bridge::SomaticErrorBridge::new();

        // Thermal bridge
        let (thermal_bridge_instance, thermal_sender) =
            crate::infrastructure::thermal_bridge::ThermalBridge::new();

        let self_model_tier = super::self_model_tier::SelfModelTierManager::new(&config);
        let enable_primitive_consciousness = config.enable_primitive_consciousness;
        let moral_anomaly_config = config.moral_anomaly_config.clone();

        // Spawn the semantic-encoder background channel. (Before 2026-07-09 the
        // flag was captured here and never used — the channel stayed None and
        // `enable_semantic_encoder` was decorative; found by the E1 causal-load
        // audit, fixed as part of OMI-1/2 open-model grounding.)
        #[cfg(feature = "semantic-encoder")]
        let (semantic_channel, semantic_bridge) = if config.enable_semantic_encoder {
            let qcfg = match &config.semantic_encoder_ollama {
                Some(model) => symthaea_embeddings::Qwen3Config::ollama(
                    model.clone(),
                    symthaea_embeddings::ollama::EMBEDDINGGEMMA_DIMENSION,
                ),
                None => symthaea_embeddings::Qwen3Config::simulated(),
            };
            let input_dim = qcfg.embedding_dim;
            match symthaea_embeddings::channel::EmbeddingChannel::spawn(qcfg) {
                Ok(ch) => {
                    let bridge = symthaea_embeddings::HdcBridge::with_config(
                        symthaea_embeddings::BridgeConfig {
                            input_dim,
                            ..Default::default()
                        },
                    );
                    (Some(ch), Some(bridge))
                }
                Err(e) => {
                    eprintln!("semantic-encoder: channel spawn failed ({e}); running without");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        #[cfg(feature = "vision-manifold")]
        let cross_manifold_predictor_init =
            if config.enable_cross_manifold_predictor && config.enable_vision_manifold {
                let cmp_seed = config
                    .genesis_phrase
                    .as_ref()
                    .map(|p| {
                        symthaea_core::genesis::GenesisSeed::from_phrase(p)
                            .domain("cognitive_loop::cross_manifold")
                            .r#gen::<u64>()
                    })
                    .unwrap_or_else(|| super::thresholds::CROSS_MANIFOLD_SEED_DEFAULT);
                Some(symthaea_vision_manifold::CrossManifoldPredictor::new(
                    16_384, cmp_seed,
                ))
            } else {
                None
            };

        let substrate_manager = super::substrate_manager::SubstrateManager::new(&config);

        #[cfg(feature = "physics-bridge")]
        let physics_integration = if config.enable_physics_bridge {
            Some(super::physics_integration::PhysicsIntegration::new())
        } else {
            None
        };

        #[cfg(feature = "analogy-engine")]
        let analogy_integration = if config.enable_analogy_engine {
            config.genesis_phrase.as_ref().map(|p| {
                let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(p);
                super::analogy_integration::AnalogyIntegration::new(&genesis)
            })
        } else {
            None
        };

        #[cfg(feature = "ucl-frames")]
        let ucl_frame_integration = if config.enable_ucl_frames {
            Some(super::ucl_frame_integration::UCLFrameIntegration::new())
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

        let broca_lite_seed: u64 = config
            .genesis_phrase
            .as_deref()
            .map(|p| {
                let mut h: u64 = 0xcbf29ce484222325;
                for b in p.as_bytes() {
                    h ^= *b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                h
            })
            .unwrap_or(0x5f3759df_u64);

        let enable_soul_alignment = config.enable_soul_alignment;
        let enable_knowledge_engine = config.enable_knowledge_engine;
        let knowledge_graph_capacity = config.knowledge_graph_capacity;
        let knowledge_causal_capacity = config.knowledge_causal_capacity;
        let knowledge_search_top_k = config.knowledge_search_top_k;
        let knowledge_ontology_max = config.knowledge_ontology_max;
        let knowledge_db_path = config.knowledge_db_path.clone();

        let (swarm_event_tx, swarm_event_rx) = std::sync::mpsc::channel();
        let (safety_alert_tx, safety_alert_rx) =
            std::sync::mpsc::sync_channel(super::safety_alert::SAFETY_ALERT_CHANNEL_CAPACITY);
        let (holon_inbound_tx, holon_inbound_rx) = std::sync::mpsc::channel();

        #[cfg(feature = "mesh")]
        let (mesh_outbound_tx, mesh_outbound_rx) = std::sync::mpsc::channel();

        let federation_handle = if config.federation_enabled {
            Some(
                super::managers::network_service_bridge::spawn_federated_coordinator(
                    crate::swarm::FederatedNetworkConfig::default(),
                    vec![0.0; 64],
                    std::time::Duration::from_millis(config.federation_round_interval_ms),
                    swarm_event_tx.clone(),
                ),
            )
        } else {
            None
        };

        let cfc_input_dim = config.cfc_config.input_dim;
        let enable_hierarchical_bundling = config.enable_hierarchical_bundling;
        let genesis_phrase_for_bundler = config.genesis_phrase.clone();
        let enable_mesh_time = config.enable_mesh_time;
        #[cfg(feature = "ssm_language")]
        let broca_nsm_semantic = config.enable_broca_nsm_semantic;
        #[cfg(feature = "ssm_language")]
        let broca_nsm_gate = config.enable_broca_nsm_gate;
        #[cfg(feature = "ssm_language")]
        let broca_multi_turn_depth = config.broca_multi_turn_depth;

        let trajectory_planning_enabled = config.enable_trajectory_planning;
        let trajectory_horizon_seconds = config.trajectory_horizon_seconds;
        let trajectory_planning_interval = config.trajectory_planning_interval;
        let enable_hodge_decomposition = config.enable_hodge_decomposition;

        // ── Build SensoriMotorExecution ──────────────────────────────
        let sensorimotor_built = {
            let vision_sensory = super::vision_sensory_manager::VisionAndSensoryManager {
                coherence_field: if enable_coherence_field {
                    Some(crate::physiology::CoherenceField::new())
                } else {
                    None
                },
                virtual_body,
                #[cfg(feature = "vision-manifold")]
                vision_bridge: if vision_manifold_enabled {
                    let mut vm_config = symthaea_vision_manifold::VisionConfig::default();
                    // Thread the loop-level dilation policy into the
                    // manifold's own FEP ExplorationTrigger — the manifold
                    // dilates ITSELF to Ultra otherwise, bypassing the
                    // loop-level gate entirely (observer-ΔΨ probe hunt,
                    // 2026-07-11).
                    vm_config.allow_auto_dilation = config.enable_vision_auto_dilation;
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
            };

            super::sensorimotor_execution::SensoriMotorExecution::new(
                vision_sensory,
                super::motor_rendering_manager::MotorRenderingManager::new_with_aesthetic_path(
                    config
                        .aesthetic_memory_path
                        .as_deref()
                        .map(std::path::PathBuf::from),
                ),
                somatic_bridge_instance,
                Some(pain_sender),
                thermal_bridge_instance,
                Some(thermal_sender),
                #[cfg(any(
                    feature = "humanoid",
                    feature = "helicopter",
                    feature = "flight",
                    feature = "vehicle",
                    feature = "auv",
                    feature = "manipulator",
                    feature = "exoskeleton",
                    feature = "surgical",
                    feature = "orbital",
                    feature = "quadruped",
                    feature = "subterranean",
                    feature = "infrastructure",
                    feature = "scavenger",
                    feature = "agribot",
                    feature = "biota",
                    feature = "clime",
                    feature = "phone"
                ))]
                None,
                #[cfg(any(
                    feature = "humanoid",
                    feature = "helicopter",
                    feature = "flight",
                    feature = "vehicle",
                    feature = "auv",
                    feature = "manipulator",
                    feature = "exoskeleton",
                    feature = "surgical",
                    feature = "orbital",
                    feature = "quadruped",
                    feature = "subterranean",
                    feature = "infrastructure",
                    feature = "scavenger",
                    feature = "agribot",
                    feature = "biota",
                    feature = "clime",
                    feature = "phone"
                ))]
                None,
                #[cfg(any(
                    feature = "humanoid",
                    feature = "helicopter",
                    feature = "flight",
                    feature = "vehicle",
                    feature = "auv",
                    feature = "manipulator",
                    feature = "exoskeleton",
                    feature = "surgical",
                    feature = "orbital",
                    feature = "quadruped",
                    feature = "subterranean",
                    feature = "infrastructure",
                    feature = "scavenger",
                    feature = "agribot",
                    feature = "biota",
                    feature = "clime",
                    feature = "phone"
                ))]
                Default::default(),
            )
        };

        #[cfg(feature = "jepa")]
        let jepa_input_dim = config.cfc_config.input_dim;

        let mut episodic_persistence =
            super::episodic_persistence_manager::EpisodicPersistenceManager::new(
                phi_episodic_replay,
            );
        if let Some(ref path) = config.memory_db_path {
            episodic_persistence
                .attach_sqlite_db(path)
                .map_err(|e| anyhow::anyhow!("failed to open memory_db_path {path:?}: {e}"))?;
        }

        let service = Self {
            config,
            encoder,
            temporal_network,
            train_history_snapshots: VecDeque::with_capacity(2),
            training_frozen: false,
            buffer: VecDeque::with_capacity(super::thresholds::EXPERIENCE_BUFFER_CAPACITY),
            stats: LoopStats::default(),
            error_history: VecDeque::with_capacity(super::thresholds::ERROR_HISTORY_CAPACITY),
            last_state: None,
            training_state_buf: None,
            last_prediction: None,
            start_time: Instant::now(),
            is_consolidating: false,
            language_comm: super::language_comm_manager::LanguageAndCommunicationManager {
                voice_coherence,
                #[cfg(feature = "ssm_language")]
                broca_manager: if broca_enabled {
                    let genesis = broca_genesis_phrase
                        .as_deref()
                        .map(symthaea_core::genesis::GenesisSeed::from_phrase)
                        .unwrap_or_else(|| {
                            symthaea_core::genesis::GenesisSeed::from_phrase(
                                "symthaea-broca-default",
                            )
                        });
                    let mut broca_config = symthaea_broca::BrocaConfig::default();
                    broca_config.enable_nsm_semantic = broca_nsm_semantic;
                    broca_config.enable_nsm_gate = broca_nsm_gate;
                    {
                        let mut mgr = super::broca_bridge::BrocaManager::new(
                            &genesis,
                            broca_config,
                            broca_checkpoint_path.as_deref(),
                        );
                        mgr.multi_turn_depth = broca_multi_turn_depth;
                        Some(mgr)
                    }
                } else {
                    None
                },
                broca_lite: super::broca_lite::BrocaLiteManager::new(broca_lite_seed),
                last_broca_text: None,
                last_language_source: None,
                last_qualified_language_output: None,
                user_state: if enable_user_state {
                    Some(crate::user_state_inference::UserStateInference::new())
                } else {
                    None
                },
                broca_code_channels: None,
            },
            voice_synthesis: None,
            voice_audio_buffer: VecDeque::new(),
            #[cfg(feature = "voice-stt")]
            pending_self_voice_hv: None,
            llm_language: None,
            behavior: super::behavioral_synthesis::BehavioralSynthesis::new(
                FlowState::default(),
                EmotionContagion::default(),
                CuriosityDrive::default(),
                adaptive_behavior,
                ThalamicRouter::default(),
                super::SocialManager::new(enable_primitive_consciousness),
            ),
            prediction_confidence: 0.5_f64,
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
                haptic_semantic_binder: symthaea_fep::HapticSemanticBinder::new(8, 64),
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
                last_action_idx: 0,
                lr_boost: 1.0,
                surprise_bridge,
                trajectory_config: super::fep_module::TrajectoryPlanningConfig {
                    enabled: trajectory_planning_enabled,
                    horizon_seconds: trajectory_horizon_seconds,
                    planning_interval: trajectory_planning_interval,
                    ..Default::default()
                },
                trajectory_telemetry: super::fep_module::TrajectoryTelemetry::default(),
                trajectory_history: VecDeque::new(),
                // CORRECT INITIALIZATION:
                ledger: symthaea_core::physics::thermodynamics::ThermodynamicLedger::new(1000.0),
            },
            viability: super::viability_fabric::ViabilityFabric::new(1024),
            feedback_state: super::feedback_state::FeedbackState::new(),
