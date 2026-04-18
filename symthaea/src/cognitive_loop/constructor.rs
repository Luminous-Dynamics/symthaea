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

/// Optional consciousness monitors built from config flags.
struct ConsciousnessMonitors {
    resonance: Option<ResonanceAnalyzer>,
    quantum_coherence: Option<crate::consciousness::quantum_coherence::QuantumCoherenceAnalyzer>,
    temporal: Option<TemporalConsciousnessAnalyzer>,
    embodied: Option<EmbodiedConsciousnessAnalyzer>,
    thermodynamics: Option<
        crate::consciousness::consciousness_thermodynamics::ConsciousnessThermodynamicsAnalyzer,
    >,
    phenomenal_binding:
        Option<crate::consciousness::phenomenal_binding::TemporalSynchronizationAnalyzer>,
    hierarchical_free_energy:
        Option<crate::consciousness::hierarchical_free_energy::HierarchicalFreeEnergy>,
}

/// Build optional consciousness monitors from config flags.
fn build_consciousness_monitors(config: &CognitiveLoopConfig) -> ConsciousnessMonitors {
    ConsciousnessMonitors {
        resonance: config.enable_resonance.then(|| {
            ResonanceAnalyzer::new(ResonanceConfig::default())
        }),
        quantum_coherence: config.enable_quantum_coherence.then(|| {
            crate::consciousness::quantum_coherence::QuantumCoherenceAnalyzer::new(
                crate::consciousness::quantum_coherence::CoherenceConfig::default(),
            )
        }),
        temporal: config.enable_temporal_consciousness.then(|| {
            TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default())
        }),
        embodied: config.enable_embodied_cognition.then(|| {
            EmbodiedConsciousnessAnalyzer::new(EmbodiedConfig::default())
        }),
        thermodynamics: config.enable_consciousness_thermodynamics.then(|| {
            crate::consciousness::consciousness_thermodynamics::ConsciousnessThermodynamicsAnalyzer::new(
                crate::consciousness::consciousness_thermodynamics::ThermodynamicsConfig::default(),
            )
        }),
        phenomenal_binding: config.enable_phenomenal_binding.then(|| {
            crate::consciousness::phenomenal_binding::TemporalSynchronizationAnalyzer::new(
                crate::consciousness::phenomenal_binding::PhenomenalBindingConfig::default(),
            )
        }),
        hierarchical_free_energy: config.enable_hierarchical_free_energy.then(|| {
            crate::consciousness::hierarchical_free_energy::HierarchicalFreeEnergy::new(
                crate::consciousness::hierarchical_free_energy::HierarchicalFEConfig::default(),
            )
        }),
    }
}

/// Optional higher-level consciousness subsystems built from config flags.
struct ConsciousnessSubsystems {
    narrative_gwt: Option<NarrativeGWTIntegration>,
    dream_engine: Option<crate::consciousness::dream::DreamEngine>,
    predictive_mind: Option<crate::consciousness::predictive_processing::PredictiveMind>,
    cross_modal_binder: Option<crate::consciousness::cross_modal_binding::CrossModalBinder>,
    affective_bridge: Option<crate::brain::affective_bridge::AffectiveBridge>,
}

/// Build optional higher-level consciousness subsystems from config flags.
fn build_consciousness_subsystems(config: &CognitiveLoopConfig) -> ConsciousnessSubsystems {
    ConsciousnessSubsystems {
        narrative_gwt: config
            .enable_narrative_gwt
            .then(|| NarrativeGWTIntegration::default_config()),
        dream_engine: config
            .enable_dream_replay
            .then(|| crate::consciousness::dream::DreamEngine::with_defaults()),
        predictive_mind: config.enable_predictive_processing.then(|| {
            crate::consciousness::predictive_processing::PredictiveMind::new(
                crate::consciousness::predictive_processing::PredictiveConfig::default(),
            )
        }),
        cross_modal_binder: config.enable_cross_modal_binding.then(|| {
            crate::consciousness::cross_modal_binding::CrossModalBinder::new(
                crate::consciousness::cross_modal_binding::CrossModalBindingConfig::default(),
            )
        }),
        affective_bridge: config
            .enable_affective_bridge
            .then(|| crate::brain::affective_bridge::AffectiveBridge::default()),
    }
}

/// Ethics subsystem optional components built from config flags.
struct EthicsComponents {
    contextual_weights: Option<crate::consciousness::contextual_weights::ContextualWeights>,
    phi_attention: Option<crate::consciousness::phi_attention::AdaptiveThresholds>,
    negation_detector: Option<crate::consciousness::negation_detector::NegationDetector>,
    metacognitive_monitor:
        Option<crate::consciousness::metacognitive_monitoring::MetacognitiveMonitor>,
    safety_gateway: Option<crate::safety::SafetyGateway>,
}

/// Build ethics and safety optional components from config flags.
fn build_ethics_components(config: &CognitiveLoopConfig) -> EthicsComponents {
    EthicsComponents {
        contextual_weights: config
            .enable_contextual_weights
            .then(|| crate::consciousness::contextual_weights::ContextualWeights::new()),
        phi_attention: config
            .enable_phi_attention
            .then(|| crate::consciousness::phi_attention::AdaptiveThresholds::new(100)),
        negation_detector: config
            .enable_negation_detection
            .then(|| crate::consciousness::negation_detector::NegationDetector::new()),
        metacognitive_monitor: config.enable_metacognitive_monitoring.then(|| {
            crate::consciousness::metacognitive_monitoring::MetacognitiveMonitor::new(0.001)
        }),
        safety_gateway: config
            .enable_safety_gateway
            .then(|| crate::safety::SafetyGateway::new()),
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
                            .gen::<u64>()
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

        // Self-model subsystems now in SelfModelTierManager::new()

        // Build optional virtual body adapter
        let virtual_body = if config.enable_virtual_body {
            Some(super::virtual_body::VirtualBody::new())
        } else {
            None
        };

        // Predictive self + attention schema now in SelfModelTierManager::new()

        // Build optional GWT integration
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

        // Always create episodic replay — it backs both graduation (always-on when
        // memory_graduation=true) and replay training (gated by episodic_replay_training).
        // Previously gated on config.episodic_replay, which blocked ALL graduation.
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
        let resonator_cfc_input_dim = config.cfc_config.input_dim;
        let resonator_genesis_phrase = config.genesis_phrase.clone();
        let has_primitive_processor = primitive_tier.primitive_processor.is_some();

        // Somatic error bridge: infrastructure errors → felt interoceptive signals
        let (somatic_bridge_instance, pain_sender) =
            crate::infrastructure::somatic_error_bridge::SomaticErrorBridge::new();

        // Thermal bridge: platform thermal state → CfC tau modulation
        // Science: Angilletta (2009) thermal performance curves
        let (thermal_bridge_instance, thermal_sender) =
            crate::infrastructure::thermal_bridge::ThermalBridge::new();

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
                    .unwrap_or_else(|| {
                        tracing::debug!(
                            "CrossManifoldPredictor: using default seed (no genesis phrase)"
                        );
                        super::thresholds::CROSS_MANIFOLD_SEED_DEFAULT
                    });
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

        // BrocaLite seed: hash the genesis phrase (or use default)
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

        let enable_visualization = config.enable_visualization;
        let enable_soul_alignment = config.enable_soul_alignment;
        let enable_knowledge_engine = config.enable_knowledge_engine;
        let knowledge_graph_capacity = config.knowledge_graph_capacity;
        let knowledge_causal_capacity = config.knowledge_causal_capacity;
        let knowledge_search_top_k = config.knowledge_search_top_k;
        let knowledge_ontology_max = config.knowledge_ontology_max;
        let knowledge_db_path = config.knowledge_db_path.clone();
        let enable_streaming_inference = config.enable_streaming_inference;
        #[cfg(feature = "therapeutic")]
        let therapeutic_crisis_threshold = config.therapeutic_crisis_threshold;

        // Create swarm event channel eagerly so the sender is always available.
        let (swarm_event_tx, swarm_event_rx) = std::sync::mpsc::channel();

        // Create safety alert channel — bounded (32) so the cognitive loop never blocks.
        // Host application drains via take_safety_alert_receiver().
        let (safety_alert_tx, safety_alert_rx) =
            std::sync::mpsc::sync_channel(super::safety_alert::SAFETY_ALERT_CHANNEL_CAPACITY);

        // Create Holon inbound channel eagerly so the sender is always available.
        // HTTP handlers (HolonHttpState) clone the tx to inject SomaMessages.
        let (holon_inbound_tx, holon_inbound_rx) = std::sync::mpsc::channel();

        // Create mesh outbound channel for sovereign beacon/name/content emission.
        #[cfg(feature = "mesh")]
        let (mesh_outbound_tx, mesh_outbound_rx) = std::sync::mpsc::channel();

        // Spawn federated coordinator if enabled.
        let federation_handle = if config.federation_enabled {
            Some(
                super::managers::network_service_bridge::spawn_federated_coordinator(
                    crate::swarm::FederatedNetworkConfig::default(),
                    vec![0.0; 64], // initial weights placeholder
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
        // Extract trajectory planning config before moving `config` into the struct
        let trajectory_planning_enabled = config.enable_trajectory_planning;
        let trajectory_horizon_seconds = config.trajectory_horizon_seconds;
        let trajectory_planning_interval = config.trajectory_planning_interval;
        let enable_hodge_decomposition = config.enable_hodge_decomposition;

        // ── Build SensoriMotorExecution before struct literal ─────────────
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
            };

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
                feature = "phone"
            ))]
            let embodiment_bridge_init = {
                use super::motor_bridge::EmbodimentPlatform;
                match config.embodiment_platform {
                    #[cfg(feature = "humanoid")]
                    EmbodimentPlatform::Humanoid => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p));
                        let bridge =
                            super::motor_bridge::MotorBridge::new(&genesis.unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            }));
                        Some(Box::new(bridge) as Box<dyn super::motor_bridge::EmbodimentBridge>)
                    }
                    #[cfg(feature = "helicopter")]
                    EmbodimentPlatform::Helicopter => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(
                            Box::new(crate::helicopter::embodiment::HelicopterEmbodiment::new(
                                &genesis,
                            ))
                                as Box<dyn super::motor_bridge::EmbodimentBridge>,
                        )
                    }
                    #[cfg(feature = "flight")]
                    EmbodimentPlatform::Quadrotor => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(
                            Box::new(crate::flight::embodiment::FlightEmbodiment::new(&genesis))
                                as Box<dyn super::motor_bridge::EmbodimentBridge>,
                        )
                    }
                    #[cfg(feature = "vehicle")]
                    EmbodimentPlatform::Vehicle => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(
                            Box::new(crate::vehicle::embodiment::VehicleEmbodiment::new(&genesis))
                                as Box<dyn super::motor_bridge::EmbodimentBridge>,
                        )
                    }
                    #[cfg(feature = "manipulator")]
                    EmbodimentPlatform::Manipulator => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(
                            Box::new(crate::manipulator::embodiment::ManipulatorEmbodiment::new(
                                &genesis,
                            ))
                                as Box<dyn super::motor_bridge::EmbodimentBridge>,
                        )
                    }
                    #[cfg(feature = "auv")]
                    EmbodimentPlatform::Auv => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(
                            Box::new(crate::auv::embodiment::AuvEmbodiment::new(&genesis))
                                as Box<dyn super::motor_bridge::EmbodimentBridge>,
                        )
                    }
                    #[cfg(feature = "exoskeleton")]
                    EmbodimentPlatform::Exoskeleton => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(Box::new(
                            symthaea_exoskeleton::embodiment::ExoskeletonEmbodiment::new(&genesis),
                        )
                            as Box<dyn super::motor_bridge::EmbodimentBridge>)
                    }
                    #[cfg(feature = "surgical")]
                    EmbodimentPlatform::Surgical => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(
                            Box::new(symthaea_surgical::embodiment::SurgicalEmbodiment::new(
                                &genesis,
                            ))
                                as Box<dyn super::motor_bridge::EmbodimentBridge>,
                        )
                    }
                    #[cfg(feature = "orbital")]
                    EmbodimentPlatform::Orbital => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(
                            Box::new(symthaea_orbital::embodiment::OrbitalEmbodiment::new(
                                &genesis,
                            ))
                                as Box<dyn super::motor_bridge::EmbodimentBridge>,
                        )
                    }
                    #[cfg(feature = "quadruped")]
                    EmbodimentPlatform::Quadruped => {
                        let genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(
                            Box::new(symthaea_quadruped::embodiment::QuadrupedEmbodiment::new(
                                &genesis,
                            ))
                                as Box<dyn super::motor_bridge::EmbodimentBridge>,
                        )
                    }
                    #[cfg(feature = "phone")]
                    EmbodimentPlatform::Phone => {
                        let _genesis = config
                            .genesis_phrase
                            .as_ref()
                            .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p))
                            .unwrap_or_else(|| {
                                symthaea_core::genesis::GenesisSeed::from_phrase("default")
                            });
                        Some(Box::new(symthaea_phone_embodiment::PhoneBridge::new(
                            "41201FDJG000UM",
                            1008,
                            2244,
                        ))
                            as Box<dyn super::motor_bridge::EmbodimentBridge>)
                    }
                    _ => None,
                }
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
                    feature = "phone"
                ))]
                embodiment_bridge_init,
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
                    feature = "phone"
                ))]
                super::motor_bridge::EmbodimentTelemetry::default(),
            )
        };

        #[cfg(feature = "jepa")]
        let jepa_input_dim = config.cfc_config.input_dim;
        #[cfg(feature = "mesh")]
        let mesh_domain_profile = config.domain_profile.clone();
        let mut service = Self {
            config,
            encoder,
            temporal_network,
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
                user_state: if enable_user_state {
                    Some(crate::user_state_inference::UserStateInference::new())
                } else {
                    None
                },
                broca_code_channels: None,
            },
            voice_synthesis: None, // Spawned on demand via enable_voice_synthesis()
            llm_language: None,    // Spawned on demand via enable_llm_language()
            behavior: super::behavioral_synthesis::BehavioralSynthesis::new(
                FlowState::default(),
                EmotionContagion::default(),
                CuriosityDrive::default(),
                adaptive_behavior,
                ThalamicRouter::default(),
                super::SocialManager::new(enable_primitive_consciousness),
            ),
            prediction_confidence: 0.5_f64, // Start neutral
            // NOTE: self_reflection is now in self_model_tier
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
            },
            feedback_state: super::feedback_state::FeedbackState::new(),
            coherence_tracker: ConversationCoherenceTracker::new(0.3),
            memory: super::memory_execution::MemoryExecution {
                memory_consol: super::memory_consolidation_manager::MemoryAndConsolidationManager {
                    stability_regime: StabilityRegimeProcessor::new(),
                    discovery_service: PrimitiveDiscoveryService::new(
                        DiscoveryServiceConfig::default(),
                    ),
                    semantic_memory: SemanticMemory::with_threshold(1000, 0.3),
                    memory_coordinator: MemoryCoordinator::new(CoordinatorConfig::default()),
                    resonator_memory: if enable_resonator_recall {
                        let dim = resonator_cfc_input_dim;
                        let res_config = crate::dynamics::resonator::ResonatorConfig {
                            dim,
                            max_iters: 50,
                            convergence_threshold: 0.995,
                            temperature: 0.1,
                            bipolar: true,
                        };
                        let mut mem =
                            crate::dynamics::resonator::ResonatorMemory::new(res_config, 500);
                        let make_hv = |seed: u64, d: usize| -> Vec<f32> {
                            let mut state = seed ^ 0x9E3779B97F4A7C15;
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
                            .unwrap_or_else(|| {
                                tracing::debug!(
                                    "ResonatorMemory: using default seed (no genesis phrase)"
                                );
                                super::thresholds::RESONATOR_MEMORY_SEED_DEFAULT
                            });
                        let mut semantic_cb = crate::dynamics::Codebook::new("semantic");
                        for i in 0..8u64 {
                            semantic_cb.add(
                                &format!("proto_{i}"),
                                make_hv(seed_base.wrapping_add(i), dim),
                            );
                        }
                        mem.add_codebook(semantic_cb);
                        let mut valence_cb = crate::dynamics::Codebook::new("valence");
                        valence_cb.add("positive", make_hv(seed_base.wrapping_add(100), dim));
                        valence_cb.add("neutral", make_hv(seed_base.wrapping_add(101), dim));
                        valence_cb.add("negative", make_hv(seed_base.wrapping_add(102), dim));
                        mem.add_codebook(valence_cb);
                        let mut phi_cb = crate::dynamics::Codebook::new("phi_level");
                        phi_cb.add("low", make_hv(seed_base.wrapping_add(200), dim));
                        phi_cb.add("medium", make_hv(seed_base.wrapping_add(201), dim));
                        phi_cb.add("high", make_hv(seed_base.wrapping_add(202), dim));
                        mem.add_codebook(phi_cb);
                        Some(mem)
                    } else {
                        None
                    },
                },
                causal_enhancer,
                episodic_persistence:
                    super::episodic_persistence_manager::EpisodicPersistenceManager::new(
                        phi_episodic_replay,
                    ),
                knowledge_manager: if enable_knowledge_engine {
                    let km_config = crate::knowledge::manager::KnowledgeManagerConfig {
                        graph_capacity: knowledge_graph_capacity,
                        causal_capacity: knowledge_causal_capacity,
                        search_top_k: knowledge_search_top_k,
                        ontology_config:
                            crate::knowledge::adaptive_ontology::AdaptiveOntologyConfig {
                                max_primitives: knowledge_ontology_max,
                                ..Default::default()
                            },
                        db_path: knowledge_db_path.clone(),
                        ..Default::default()
                    };
                    let mut km = crate::knowledge::KnowledgeManager::new(km_config);
                    km.bootstrap_entities();
                    Some(km)
                } else {
                    None
                },
            },
            feature_integ: super::feature_integration_manager::FeatureIntegrationManager {
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
                                tracing::warn!(err = %e, "Failed to load neural bridge");
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
                #[cfg(feature = "semantic-encoder")]
                semantic_embedding_channel: {
                    if enable_semantic_encoder {
                        let qwen_config = symthaea_embeddings::Qwen3Config::simulated();
                        match symthaea_embeddings::channel::EmbeddingChannel::spawn(qwen_config) {
                            Ok(channel) => Some(channel),
                            Err(e) => {
                                tracing::warn!(err = %e, "Failed to spawn semantic encoder");
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
                #[cfg(feature = "school_learning")]
                school_bridge: None,
                causal_consciousness,
                #[cfg(feature = "physics-bridge")]
                physics_integration,
                #[cfg(feature = "analogy-engine")]
                analogy_integration,
                #[cfg(feature = "ucl-frames")]
                ucl_frame_integration,
            },
            async_trainer,
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
            ethics_values: super::ethics_values_manager::EthicsAndValuesManager {
                last_moral_judgment: None,
                contextual_weights,
                phi_attention,
                negation_detector,
                soul: if enable_soul_alignment {
                    Some(crate::soul::Soul::new(crate::soul::SoulConfig {
                        dimension: symthaea_core::hdc::unified_hv::HDC_DIMENSION,
                        ..Default::default()
                    }))
                } else {
                    None
                },
            },

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
            primitive_tier,
            thermodynamic_mgr:
                super::managers::thermodynamic_manager::ThermodynamicManager::default(),
            #[cfg(feature = "support")]
            support: super::support_manager::SupportManager::new(),
            carryover: CycleCarryover::default(),
            prefrontal,
            // self_model_tier, gwt_mgr, consciousness_monitors moved into consciousness field below
            narrative_gwt,
            attention_visualizer: if enable_visualization {
                Some(crate::visualization::AttentionVisualizer::with_max_history(
                    500,
                ))
            } else {
                None
            },
            // social_mgr moved to behavior (BehavioralSynthesis)
            // vision_sensory, motor_rendering, somatic/thermal/embodiment moved to sensorimotor
            sensorimotor: sensorimotor_built,
            // STT capture: opt-in, started post-construction via start_stt_capture().
            #[cfg(feature = "voice-stt-live")]
            stt_capture: None,
            // IMU fusion: opt-in. Install a fusion via install_imu_fusion()
            // and push readings via inject_imu_reading().
            #[cfg(feature = "sensor-imu")]
            imu_fusion: None,
            #[cfg(feature = "sensor-imu")]
            latest_imu_reading: None,
            #[cfg(feature = "nurture")]
            nurture_attachment: if enable_nurture_attachment {
                Some(super::nurture_bridge::NurtureAttachmentBridge::new())
            } else {
                None
            },
            // vision/foveation/broca inits moved to vision_sensory and language_comm managers above
            // canvas_manager, last_canvas_svg moved to motor_rendering manager
            psi_attestation_buffer: std::collections::VecDeque::with_capacity(attestation_buf_cap),
            policy_agreement_window: std::collections::VecDeque::with_capacity(20),
            // master_equation moved into consciousness field below
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
            // last_reasoning_context moved to episodic_persistence manager
            experience_bus: Some(crate::experience::ExperienceBus::with_defaults()),
            // school_bridge + causal_consciousness moved to feature_integ manager
            thermodynamic_load: 0.0,
            mood_temperature: 1.0,
            neuromod: super::neuromod_manager::NeuromodManager::default(),
            // somatic_bridge, pain_tx, thermal_bridge, thermal_tx moved to sensorimotor_built
            subsystem_collector: super::subsystem_trait::OutputCollector::new(),
            subsystem_health: super::subsystem_trait::SubsystemHealthTracker::new(),
            last_snapshot: None,

            // ── Unified Engines (additive wiring — old fields remain) ────────
            consciousness: super::consciousness_execution::ConsciousnessExecution::new(
                {
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
                        ).map_err(|e| tracing::warn!(err = %e, "UnifiedConsciousnessPipeline init failed")).ok()
                    } else {
                        None
                    };
                    super::consciousness_engine::ConsciousnessEngine::new(
                        engine_smf, engine_mmi, engine_eq, engine_ucp,
                    )
                },
                super::consciousness_monitor_tier::ConsciousnessMonitorTier {
                    resonance: consciousness_resonance,
                    quantum_coherence,
                    temporal: temporal_consciousness,
                    embodied: embodied_cognition,
                    thermodynamics: consciousness_thermodynamics,
                    phenomenal_binding,
                    hierarchical_free_energy,
                },
                super::gwt_manager::GwtManager::new(gwt, gwt_memory_flag, gwt_perception_count),
                self_model_tier,
                MasterConsciousnessEquation::default(),
            ),
            substrate_manager,
            threshold_overrides: super::threshold_overrides::ThresholdOverrides::default(),
            #[cfg(feature = "jepa")]
            jepa_engine: {
                let seed: u64 = 42;
                Some(symthaea_jepa::JepaEngine::new(
                    symthaea_jepa::JepaConfig {
                        input_dim: jepa_input_dim,
                        ..Default::default()
                    },
                    seed,
                ))
            },
            #[cfg(feature = "neural_validation")]
            cortical_history: std::collections::VecDeque::with_capacity(1000),
            // physics_integration moved to feature_integ manager
            convergence_cycle: 0,
            governance_consciousness_lag: std::collections::VecDeque::with_capacity(
                super::thresholds::GOVERNANCE_CONSCIOUSNESS_LAG_SIZE,
            ),
            ethics_engine: {
                let engine_mp = MoralParser::new();
                let mut engine_ma = MoralAlgebra::default_dim();
                // Wire Spinozist as 5th ensemble signal (untrained at startup;
                // train via train_hybrid() when training data becomes available)
                engine_ma.set_spinozist(crate::hdc::spinozist_geometry::SpinozistClassifier::new());
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
                    enable_hodge_decomposition,
                )
            },
            last_ethics_verdict: super::ethics_engine::EthicalVerdict::Safe,
            ethics_verdict_override: None,
            kosmic_song: crate::mycelix::KosmicSong::default(),
            drive_manager: super::managers::DriveManager::default(),
            memory_manager: super::managers::MemoryManager::default(),
            learning_manager: super::managers::LearningManager::default(),
            perception_manager: super::managers::PerceptionManager::default(),
            soul_manager: if enable_soul_alignment {
                Some(super::managers::SoulManager::new())
            } else {
                None
            },
            #[cfg(feature = "mycelix")]
            governance_mgr: super::managers::GovernanceManager::default(),
            #[cfg(feature = "mycelix")]
            factcheck_bridge: super::broca_factcheck::BrocaFactcheckBridge::new(),
            #[cfg(feature = "epistemic")]
            known_unknowns: Some(crate::consciousness::sacred_stillness::KnownUnknowns::new()),
            swarm_manager: super::managers::SwarmManager::default(),
            #[cfg(feature = "muse")]
            muse_manager: {
                let mut mm = super::managers::MuseManager::new();
                // Restore motif memory from previous session if available
                let motif_path = config
                    .aesthetic_memory_path
                    .as_deref()
                    .map(|p| {
                        let mut pb = std::path::PathBuf::from(p);
                        pb.set_extension("motifs.json");
                        pb
                    })
                    .unwrap_or_else(|| std::path::PathBuf::from(".claude/motif_memory.json"));
                let snapshot = symthaea_muse::motif_memory::MotifSnapshot::load(&motif_path);
                if !snapshot.is_empty() {
                    log::info!(
                        "[muse] restored {} motif phrases from {:?}",
                        snapshot.phrases.len(),
                        motif_path,
                    );
                    mm.restore_motif(&snapshot);
                }
                mm.set_motif_save_path(motif_path);
                mm
            },
            #[cfg(feature = "muse")]
            music_publisher: super::managers::MusicPublisher::new(),
            holon_receiver: crate::consciousness::holon_receiver::HolonReceiver::new(),
            holon_inbound_rx: std::sync::Mutex::new(Some(holon_inbound_rx)),
            holon_inbound_tx,
            swarm_event_rx: std::sync::Mutex::new(Some(swarm_event_rx)),
            swarm_event_tx,
            safety_alert_tx,
            safety_alert_rx: std::sync::Mutex::new(Some(safety_alert_rx)),
            federation_handle,
            network_service: None,
            #[cfg(feature = "mesh")]
            spectrum_manager: super::managers::SpectrumManager::with_domain_profile(
                mesh_domain_profile,
            ),
            #[cfg(feature = "mesh")]
            consciousness_router:
                super::managers::radio_dispatcher::ConsciousnessAwareRouter::default(),
            #[cfg(feature = "mesh")]
            store_and_forward: super::managers::radio_dispatcher::StoreAndForward::default(),
            #[cfg(feature = "cpg")]
            cpg_manager: super::managers::CpgManager::new(
                super::managers::cpg_manager::CpgConfig::default(),
            ),
            #[cfg(feature = "spectral_state")]
            spectral_manager: super::managers::SpectralManager::new(
                super::managers::spectral_manager::SpectralManagerConfig::default(),
            ),
            #[cfg(feature = "therapeutic")]
            therapeutic_manager: {
                let mut tm = super::managers::TherapeuticManager::default();
                tm.crisis_detector
                    .set_threshold(therapeutic_crisis_threshold);
                tm
            },
            #[cfg(feature = "advanced-manufacturing")]
            fabrication_manager: super::managers::FabricationManager::default(),
            cantor_dream: super::cantor_dream_manager::CantorDreamManager::new(
                super::thresholds::CANTOR_CODEBOOK_MAX_ENTRIES,
            ),
            #[cfg(feature = "glyph_codex")]
            glyph_manager: super::managers::GlyphManager::with_dim(
                crate::hdc::moral_algebra::MORAL_DIM,
            ),
            #[cfg(feature = "mesh")]
            time_manager: super::managers::TimeManager::new(enable_mesh_time),
            #[cfg(feature = "mesh")]
            mesh_outbound_tx,
            #[cfg(feature = "mesh")]
            mesh_outbound_rx: std::sync::Mutex::new(Some(mesh_outbound_rx)),
            #[cfg(feature = "mesh-trust")]
            trust_manager: super::managers::TrustManager::new(
                format!("node_{:016x}", {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    std::time::SystemTime::now().hash(&mut h);
                    h.finish()
                }),
                true,
            ),
            #[cfg(feature = "social-fabric")]
            social_fabric_manager: super::managers::SocialFabricManager::new(true),
            #[cfg(feature = "survival")]
            survival_manager: super::managers::SurvivalManager::new(true),
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
            // motor_rendering moved to sensorimotor_built
            hierarchical_bundler: if enable_hierarchical_bundling {
                Some(
                    symthaea_core::hdc::hierarchical_bundle::HierarchicalBundler::new(
                        genesis_phrase_for_bundler.as_ref().map_or(42, |p| {
                            use std::hash::{Hash, Hasher};
                            let mut h = std::collections::hash_map::DefaultHasher::new();
                            p.hash(&mut h);
                            h.finish()
                        }),
                    ),
                )
            } else {
                None
            },
            cfc_input_buffer: ndarray::Array1::zeros(cfc_input_dim),
            #[cfg(feature = "mathematics")]
            math_service: super::math_service::MathService::new(),
            #[cfg(feature = "mathematics")]
            conjecture_engine: symthaea_core::hdc::conjecture_engine::ConjectureEngine::new(),
            #[cfg(feature = "epistemic_auditor")]
            epistemic_auditor: None, // initialized below after struct construction
            // Defense / Immune System
            #[cfg(feature = "safety-agents")]
            safety_agent: crate::safety::SafetyAgent::new(),
            #[cfg(feature = "safety-agents")]
            guardian_state: super::guardian::GuardianState::default(),
            #[cfg(feature = "sentinel")]
            sentinel_manager: super::managers::SentinelManager::default(),
            #[cfg(feature = "sentinel")]
            threat_memory: super::threat_memory::ThreatMemory::default(),
            #[cfg(feature = "sentinel")]
            collective_immune_state: super::collective_immunity::CollectiveImmuneState::default(),
            #[cfg(feature = "safety-agents")]
            defense_actions_proposed: 0,
            #[cfg(feature = "safety-agents")]
            defense_actions_approved: 0,
            #[cfg(feature = "safety-agents")]
            civic_crisis_detector: super::civic_crisis_detector::CivicCrisisDetector::new(),
            #[cfg(feature = "safety-agents")]
            pending_crisis_events: Vec::new(),
            #[cfg(feature = "neuroevolution")]
            neuroevolution_manager: super::managers::NeuroevolutionManager::default(),
            #[cfg(feature = "hypervisor")]
            hypervisor_manager: super::managers::HypervisorManager::new("agent".to_string()),
            #[cfg(feature = "reasoning_engine")]
            reasoning_manager: super::managers::ReasoningManager::default(),
            #[cfg(feature = "ssm_language")]
            language_manager: super::managers::LanguageManager::default(),
            #[cfg(feature = "vision-manifold")]
            vision_manager: super::managers::VisionManager::default(),
            security_telemetry: crate::swarm::SecurityTelemetry::default(),
            #[cfg(feature = "scientific_method")]
            scientific_method_engine: {
                let mut engine = crate::scientific_method::ScientificMethodEngine::new();
                // Seed with a standing "input is coherent" hypothesis (id 0).
                // Its posterior drifts each cycle based on prediction error.
                engine.hypothesize("Input representations are coherent and consistent", 0.5);
                engine
            },
            // embodiment_bridge, last_proprioceptive_hv, embodiment_telemetry moved to sensorimotor_built
            resonant_speech: crate::resonant_speech::ResonantSpeech::new(),
            streaming_inference: if enable_streaming_inference {
                // Cycle-aligned config: batch=1, max_latency=32ms (~31Hz loop)
                let si_config = crate::inference::StreamingConfig {
                    batch_accumulation: 1,
                    max_latency_ms: 32,
                    warmup_samples: 10,
                    ..crate::inference::StreamingConfig::low_latency()
                };
                Some(crate::inference::StreamingInference::with_default_network(
                    si_config,
                ))
            } else {
                None
            },
        };

        // Initialize Epistemic Auditor if configured
        #[cfg(feature = "epistemic_auditor")]
        {
            service.epistemic_auditor = service.config.epistemic_auditor_db_path.as_deref().map(|path| {
                match super::epistemic_auditor::EpistemicAuditor::new(Some(path)) {
                    Ok(auditor) => {
                        tracing::info!(path, "Epistemic Auditor initialized (DuckDB)");
                        Some(auditor)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Epistemic Auditor init failed — auditing disabled");
                        None
                    }
                }
            }).flatten();
        }

        // Initialize persistent memory database if configured
        if let Some(ref db_path) = service.config.memory_db_path {
            match crate::databases::SqliteMemory::new(db_path) {
                Ok(db) => {
                    tracing::info!(path = %db_path, "Memory persistence database initialized");
                    service.memory.episodic_persistence.db = Some(std::sync::Arc::new(db));
                }
                Err(e) => {
                    tracing::warn!(path = %db_path, error = %e, "Failed to open memory database — persistence disabled");
                }
            }
        }

        // Startup rehydration: load top-64 episodes from SQLite into episodic replay
        if let Some(ref db) = service.memory.episodic_persistence.db {
            if let Some(ref mut replay) = service.memory.episodic_persistence.replay {
                let records = db.load_top_by_psi_sync(64);
                let mut rehydrated = 0usize;
                for record in records {
                    // Convert BinaryHV to ContinuousHV via bipolar encoding (bit=1 → +1.0, bit=0 → -1.0)
                    let dim = record.encoding.0.len() * 8;
                    let mut values = Vec::with_capacity(dim);
                    for byte in &record.encoding.0 {
                        for bit in (0..8).rev() {
                            if (byte >> bit) & 1 == 1 {
                                values.push(1.0f32);
                            } else {
                                values.push(-1.0f32);
                            }
                        }
                    }
                    let input_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(values);
                    let output_hv = symthaea_core::hdc::unified_hv::ContinuousHV::zero(dim);
                    let episode = crate::memory::episodic_replay::Episode::new(
                        input_hv,
                        output_hv,
                        record.psi,
                        record.timestamp_ms / 20, // convert ms back to approximate cycle number
                    );
                    if replay.store_if_significant(episode) {
                        rehydrated += 1;
                    }
                }
                if rehydrated > 0 {
                    tracing::info!(
                        episodes = rehydrated,
                        "Startup rehydration: loaded episodes from SQLite"
                    );
                }
            }
        }

        // Honor enable_voice_synthesis config flag
        if service.config.enable_voice_synthesis {
            service.enable_voice_synthesis();
        }

        Ok(service)
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
                tracing::warn!(err = %e, "Failed to create Qwen3 embedder for dense HarmonyBasis");
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
                tracing::warn!(err = %e, "Failed to batch-encode harmony keywords");
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
    use crate::domain::DomainProfile;

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
        assert!((service.behavior.social_mgr.social.social_trust - 0.5).abs() < f32::EPSILON);
        assert!(
            (service.behavior.social_mgr.social.social_cooperation_rate - 0.0).abs() < f32::EPSILON
        );
    }

    #[test]
    fn default_external_reward_is_zero() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert!((service.behavior.social_mgr.social.external_reward - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn default_fep_learning_signal_is_zero() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert!((service.fep_learning_signal() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn default_relational_psi_is_zero() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert!((service.behavior.social_mgr.social.relational_psi - 0.0).abs() < f64::EPSILON);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn constructor_applies_domain_profile_to_spectrum_manager() {
        let config = CognitiveLoopConfig::for_domain(DomainProfile::underwater());
        let service = CognitiveLoopService::new(config).unwrap();
        assert_eq!(service.spectrum_manager.domain_profile().kind, "underwater");
    }

    #[test]
    fn constructor_starts_without_network_service() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert!(service.network_service().is_none());
        assert!(service
            .publish_local_navigation_estimate(
                &positioning::GaussianEstimate3D::from_diagonal_sigma([0.0, 0.0, 0.0], 5.0),
                Some(0.9),
            )
            .is_none());
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
    fn episodic_memory_created_by_default_for_graduation() {
        // memory_graduation defaults to true, so phi_episodic_replay is always created
        let config = CognitiveLoopConfig::default();
        let service = CognitiveLoopService::new(config).unwrap();
        assert_eq!(service.episodic_replay_count(), 0);
        assert!(service.episodic_replay_stats().is_some());
    }

    #[test]
    fn episodic_memory_absent_when_graduation_disabled() {
        let mut config = CognitiveLoopConfig::default();
        config.memory_graduation = false;
        config.episodic_replay_training = false;
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(service.episodic_replay_stats().is_none());
    }

    #[test]
    fn episodic_replay_enabled_creates_memory() {
        let mut config = CognitiveLoopConfig::default();
        config.episodic_replay_training = true;
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(service.episodic_replay_stats().is_some());
        assert_eq!(service.episodic_replay_count(), 0);
    }

    #[test]
    fn primitive_consciousness_disabled_means_no_subsystems() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = false;
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
            service.behavior.social_mgr.phi_dyad.is_some(),
            "phi_dyad should be Some when enable_primitive_consciousness=true"
        );
        assert!(
            service.behavior.social_mgr.partner_model.is_some(),
            "partner_model should be Some when enable_primitive_consciousness=true"
        );
    }

    #[test]
    fn phi_dyad_none_when_consciousness_disabled() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = false;
        assert!(!config.enable_primitive_consciousness);
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(
            service.behavior.social_mgr.phi_dyad.is_none(),
            "phi_dyad should be None when enable_primitive_consciousness=false"
        );
        assert!(
            service.behavior.social_mgr.partner_model.is_none(),
            "partner_model should be None when enable_primitive_consciousness=false"
        );
    }

    #[test]
    fn hv_ring_buffers_empty_at_construction() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_primitive_consciousness = true;
        let service = CognitiveLoopService::new(config).unwrap();
        assert!(service.behavior.social_mgr.recent_ai_hvs.is_empty());
        assert!(service.behavior.social_mgr.recent_input_hvs.is_empty());
    }
}
