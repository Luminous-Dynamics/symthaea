//! Constructor and backend selection for CognitiveLoopService.

use crate::causal::{CausalEnhancerConfig, CausalLoopEnhancer};
use crate::consciousness::consciousness_unification::ConsciousnessUnificationEngine;
use crate::consciousness::master_consciousness_equation::MasterConsciousnessEquation;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::autopoietic_consciousness::{AutopoieticConfig, AutopoieticConsciousness};
#[cfg(feature = "full_consciousness")]
use crate::consciousness::enactive_cognition::EnactiveCognition;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::unified_living_mind::UnifiedLivingMind;
use crate::brain::prefrontal::PrefrontalCortex;
use crate::consciousness::attention_schema::AttentionSchema;
use crate::consciousness::consciousness_resonance::{ResonanceAnalyzer, ResonanceConfig};
use crate::consciousness::gwt_integration::{UnifiedGWTConfig, UnifiedGlobalWorkspace};
use crate::consciousness::narrative_self::{NarrativeSelfConfig, NarrativeSelfModel};
use crate::consciousness::predictive_self::{PredictiveSelfConfig, PredictiveSelfModel};
use crate::consciousness::quantum_coherence::QuantumCoherenceAnalyzer;
use crate::consciousness::temporal_consciousness::{TemporalConsciousnessAnalyzer, TemporalConsciousnessConfig};
use crate::consciousness::embodied_cognition::{EmbodiedConfig, EmbodiedConsciousnessAnalyzer};
use crate::consciousness::narrative_gwt_integration::NarrativeGWTIntegration;
use crate::exploration::SurpriseExplorationBridge;
use crate::wisdom::meta_cognition::MetaCognitiveLayer;
use crate::consciousness::fep_active_inference::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, EnhancedFEPBridge,
};
use crate::consciousness::primitive_belief_bridge::PrimitiveBeliefBridge;
use crate::consciousness::primitive_discovery::{
    DiscoveryServiceConfig, PrimitiveDiscoveryService,
};
use crate::consciousness::stability_regime::StabilityRegimeProcessor;
use crate::dynamics::cfc::CfCNetwork;
use crate::dynamics::cfc_coherence::{CfCCoherenceBridge, CoherenceConfig};
use crate::dynamics::temporal_signatures::{SignatureConfig, TemporalSignatureEncoder};
use crate::hdc::moral_algebra::MoralAlgebra;
use crate::hdc::moral_parser::MoralParser;
use crate::hdc_ltc_bridge::HdcLtcBridge;
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::memory::memory_coordinator::{CoordinatorConfig, MemoryCoordinator};
use crate::memory::semantic_memory::SemanticMemory;
#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;
use crate::voice::voice_feedback::{VoiceFeedbackBridge, VoiceFeedbackConfig};
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
        let encoder = PredictiveHdcEncoder::new(config.encoder_config.clone());

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
            Some(TemporalConsciousnessAnalyzer::new(TemporalConsciousnessConfig::default()))
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

        // Build optional dream engine for counterfactual learning
        let dream_engine = if config.enable_dream_replay {
            Some(crate::consciousness::dream::DreamEngine::with_defaults())
        } else {
            None
        };

        // Build optional predictive processing mind
        let predictive_mind = if config.enable_predictive_processing {
            Some(crate::consciousness::predictive_processing::PredictiveMind::new(
                crate::consciousness::predictive_processing::PredictiveConfig::default(),
            ))
        } else {
            None
        };

        // Build optional cross-modal binder
        let cross_modal_binder = if config.enable_cross_modal_binding {
            Some(crate::consciousness::cross_modal_binding::CrossModalBinder::new(
                crate::consciousness::cross_modal_binding::BindingConfig::default(),
            ))
        } else {
            None
        };

        // Build optional affective bridge
        let affective_bridge = if config.enable_affective_bridge {
            Some(crate::brain::affective_bridge::AffectiveBridge::default())
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
            prev_predictive_phi_modulation: 1.0,
            prev_cross_modal_phi: 0.0,
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
            relational_phi: 0.0,
            external_reward: 0.0,
            virtual_body,
            prev_body_phi_modulation: 1.0,
            prev_embodied_phi_modulation: 1.0,
            consecutive_low_error: 0,
            prev_resonance_frequency: 0.0,
            prev_quantum_coherence: 0.0,
            mce_lr_boost: 0.0,
            narrative_veto_active: false,
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
        })
    }

    /// Get the current temporal backend type
    pub fn temporal_backend(&self) -> TemporalBackend {
        self.temporal_network.backend_type()
    }
}
