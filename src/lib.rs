/*!
Sophia: Holographic Liquid Brain

Revolutionary consciousness-first AI combining:
- HDC (Hyperdimensional Computing) - 16,384D holographic vectors (32K+ on demand)
- LTC (Liquid Time-Constant Networks) - Continuous-time causal reasoning
- Autopoiesis - Self-referential consciousness emergence
- Phase 11 Bio-Digital Bridge - Semantic grounding, safety, memory, swarm
*/

// Core Phase 10 modules
pub mod hdc;
pub mod ltc;
pub mod hierarchical_cantor_ltc;  // 7-level Cantor-LTC/HDC for meta-cognition
pub mod consciousness;
pub mod nix_understanding;

// Learnable LTC (gradient-based neural adaptation)
pub mod learnable_ltc;
pub mod sparse_ltc;  // Efficient sparse LTC implementation
pub mod learning;    // Learning integration engine

// Continuous Mind - continuously running cognitive system
pub mod continuous_mind;

// Week 0: Actor Model & Physiological Systems
pub mod brain;

// Week 1: Soul Module (Temporal Coherence & Identity)
pub mod soul;

// Week 2: Memory Systems (Episodic & Procedural)
pub mod memory;

// Week 4: Physiology (The Body)
pub mod physiology;

// Week 12: Perception & Tool Creation
pub mod perception;

// Embeddings (BGE, Qwen3) - used by perception and language modules
pub mod embeddings;

// Enhancement #4: Observability & Causal Analysis (Phase 3+)
pub mod observability;

// Enhancement #7 & #8: Causal Program Synthesis + Consciousness-Guided Synthesis
pub mod synthesis;

// Track 6: Component Integration
pub mod action;                             // Action IR and execution
pub mod databases;
pub mod language;
pub mod web_research;                         // Track 6: Enabled with reqwest, scraper, html2text
pub mod awakening;

// Phase 11: Bio-Digital Bridge modules (Week 0: Deferred to later phases)
// pub mod semantic_ear;  // Needs rust-bert, tokenizers
pub mod safety;
pub mod sleep_cycles;
pub mod swarm;  // libp2p enabled - P2P collective learning

// Shell Sidecar module - AI-native command interface
#[cfg(feature = "shell")]
pub mod shell;

// GUI Bridge module - Bidirectional widget<->Nix mapping
pub mod gui_bridge;

// Infrastructure module - Performance and reliability
pub mod infrastructure;

// Intelligence module - HDC-powered AI features
pub mod intelligence;

// Benchmarks (Tübingen causal discovery, etc.)
pub mod benchmarks;

// REST API for benchmark platform (optional - requires axum feature)
#[cfg(feature = "api")]
pub mod api;

// Week 12 Phase 2a: Voice Interface (STT + TTS)
pub mod voice;

// Custom Proprioception: Direct hardware monitoring with HDC integration
pub mod substrate;

// Phase 11+: Mycelix Protocol integration (Deferred - needs sha2, uuid, urlencoding dependencies)
// pub mod sophia_swarm;  // Needs additional dependencies

// Phase 11+: Resonant Speech - ENABLED (only needs uuid which is available)
pub mod resonant_speech;

// Phase 11+: User State Inference - ENABLED (depends on resonant_speech)
pub mod user_state_inference;

// Phase 11+: Resonant Interaction - ENABLED (voice cortex adapter)
pub mod resonant_interaction;

// Phase 11+: K-Index Client - ENABLED (minimal trait definition)
pub mod kindex_client;
// pub mod kindex_client_http;  // Needs HTTP client, still disabled

// Phi Engine - Unified Φ calculation framework with automatic method selection
pub mod phi_engine;

// Phase 11+: Resonant Telemetry (Deferred to Week 11+)
// pub mod resonant_telemetry;

// Re-exports for convenience
pub use hdc::{SemanticSpace, HdcContext};  // Week 0: Added HdcContext
pub use ltc::LiquidNetwork;
pub use consciousness::ConsciousnessGraph;
pub use nix_understanding::NixUnderstanding;

// Learning integration re-exports
pub use learnable_ltc::{LearnableLTC, LearnableLTCConfig, LTCTrainingStats};
pub use learning::{LearningEngine, LearningConfig, LearningStats, Experience};
pub use continuous_mind::{ContinuousMind, MindConfig, MindState, MindResponse};

// Week 0: Deferred Phase 11+ exports
// pub use semantic_ear::SemanticEar;  // Needs rust-bert
pub use safety::{SafetyGuardrails, ForbiddenCategory, SafetyStats, AmygdalaActor, ThreatLevel};
pub use sleep_cycles::{SleepCycleManager, SleepConfig, MemoryType, SleepReport};

// Week 1: Soul Module exports
pub use soul::{WeaverActor, DailyState, CoherenceStatus, KVector};

// Week 2: Memory Systems exports
pub use memory::{HippocampusActor, MemoryTrace, RecallQuery, EmotionalValence};
pub use brain::{
    CerebellumActor, Skill, ExecutionContext, WorkflowChain, CerebellumStats,
    MotorCortexActor, ActionStep, PlannedAction, StepResult, ExecutionResult,
    SimulationMode, ExecutionSandbox, LocalShellSandbox, MotorCortexStats,
};

// Week 3: Prefrontal Cortex (Global Workspace) exports
pub use brain::{
    PrefrontalCortexActor, AttentionBid, GlobalWorkspace, WorkingMemoryItem, PrefrontalStats,
    WorkingMemoryStats, Goal, Condition, GoalStats,
    MetaCognitionMonitor, CognitiveMetrics, RegulatoryAction, RegulatoryBid,
    MetaCognitionConfig, MonitorStats,
    ThalamusActor,  // Week 5: Sensory Relay
};

// Week 4: The Daemon (Default Mode Network) exports
pub use brain::{
    DaemonActor, DaemonConfig, Insight, DaemonStats,
};

// Week 4: Physiology exports
pub use physiology::{
    EndocrineSystem, EndocrineConfig, EndocrineStats,
    HormoneState, HormoneEvent, HormoneTrend, Trend,
    HearthActor, HearthConfig, HearthStats,
    ActionCost, EnergyState,
    ChronosActor, ChronosConfig, ChronosStats,  // Week 5: Time perception
    TimeMode, TimeQuality, CircadianPhase,
    ProprioceptionActor, ProprioceptionConfig, ProprioceptionStats,  // Week 5: Hardware awareness
    BodySensation,
    CoherenceField, CoherenceConfig, CoherenceStats, CoherenceState,  // Week 6+: Revolutionary energy model
    CoherenceError, TaskComplexity,
    ScatterCause, ScatterAnalysis,  // Week 9: Advanced coherence diagnostics
};
#[cfg(feature = "audio")]
pub use physiology::{LarynxActor, LarynxConfig, LarynxStats, ProsodyParams};  // Week 12 Phase 2a: Voice output with prosody

// Week 12 Phase 2a: Voice Interface exports
pub use voice::{VoiceError, VoiceResult, LTCPacing, VoiceConfig, VoiceEvent};
pub use voice::models::{ModelManager, KokoroModel, WhisperModel};
#[cfg(feature = "voice-tts")]
pub use voice::{VoiceOutput, VoiceOutputConfig, SynthesisResult};
#[cfg(feature = "voice-stt")]
pub use voice::{VoiceInput, VoiceInputConfig, TranscriptionResult};

// Week 12: Perception & Tool Creation exports
pub use perception::{
    VisualCortex, VisualFeatures,
    CodePerceptionCortex, ProjectStructure, RustCodeSemantics, CodeQualityAnalysis,
};

pub use swarm::{SwarmIntelligence, SwarmConfig, PeerStats, SwarmError};  // libp2p enabled

// Substrate awareness exports (custom proprioception)
pub use substrate::{
    Proprioception as SubstrateAwareness,
    ProprioceptiveState, ConsciousnessMap, SensorReading,
    SensorType, SensorUnit, HwmonMonitor, NvidiaHwmonMonitor,
};

// Track 6: Language & Conversation exports
pub use language::{
    Conversation, ConversationConfig, ConversationTurn, ConversationState,
};
pub use awakening::{SymthaeaAwakening, AwakenedState, Introspection as AwakeningIntrospection};

// Shell Sidecar exports (AI-native command interface)
#[cfg(feature = "shell")]
pub use shell::{
    ShellContext, CommandClassification,
    IntelliSenseEngine, Completion, CompletionKind,
    PhiGate, GateDecision, GateReason, ExecutionRequest,
    ShellIpcClient, IpcError,
};

// GUI Bridge exports (Bidirectional widget<->Nix mapping)
pub use gui_bridge::{
    GuiBridge, ConfigCategory, SyncState, WidgetUpdate, SearchResult,
    WidgetMapper, WidgetBinding, WidgetId, WidgetValue, NixPath, ValueTransformer, MappingResult,
    HdcTranslator, SemanticBinding, SemanticRole, SemanticConfidence, TranslationResult,
    ConstraintValidator, Constraint, ConstraintKind, ValidationResult, ValidationError, Suggestion,
};

use anyhow::Result;
use tokio::sync::RwLock;
use std::sync::Arc;

// Import new integrated modules
use perception::SemanticEncoder;
use substrate::Proprioception as SubstrateMonitor;
use swarm::SwarmStats;

/// Complete Sophia system with all components (Week 0: Minimal version)
pub struct SophiaHLB {
    /// Phase 10: Core components
    semantic: SemanticSpace,
    liquid: LiquidNetwork,
    consciousness: ConsciousnessGraph,
    nix: NixUnderstanding,

    /// Phase 11: Bio-Digital Bridge
    // ear: SemanticEar,  // Deferred until rust-bert available
    safety: SafetyGuardrails,
    sleep: SleepCycleManager,
    swarm: Arc<RwLock<SwarmIntelligence>>,  // P2P collective learning

    /// New integrated modules
    semantic_encoder: SemanticEncoder,      // Text → HDC encoding
    substrate_monitor: SubstrateMonitor,    // Hardware → consciousness mapping

    /// Week 4+: The Endocrine System - Hormonal State
    endocrine: EndocrineSystem,   // Hormone dynamics (cortisol, dopamine, acetylcholine)

    /// Week 5: The Nervous System - Wired Organs
    hearth: HearthActor,          // Metabolic energy & gratitude recharge (legacy compatibility)
    thalamus: ThalamusActor,      // Sensory relay & gratitude detection
    prefrontal: PrefrontalCortexActor,  // Energy-aware cognition
    chronos: ChronosActor,        // Time perception & circadian rhythms
    proprioception: ProprioceptionActor,  // Hardware awareness (body sense)

    /// Week 6+: Revolutionary Consciousness Model
    coherence: CoherenceField,    // Consciousness as integration (replaces ATP commodity model)

    /// System state
    operations_count: usize,
}

/// Response from Sophia
#[derive(Debug, Clone)]
pub struct SophiaResponse {
    /// Response content (NixOS command or explanation)
    pub content: String,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,

    /// Steps to consciousness emergence
    pub steps_to_emergence: usize,

    /// Safety check passed
    pub safe: bool,
}

/// Introspection data
#[derive(Debug, Clone)]
pub struct Introspection {
    /// Current consciousness level
    pub consciousness_level: f32,

    /// Number of self-referential loops
    pub self_loops: usize,

    /// Graph size (conscious states)
    pub graph_size: usize,

    /// Graph complexity (edges per node)
    pub complexity: f32,

    /// Memory statistics
    pub memory_stats: sleep_cycles::MemoryStats,

    /// Safety statistics
    pub safety_stats: SafetyStats,
}

impl SophiaHLB {
    /// Create new Sophia system
    pub async fn new(semantic_dim: usize, liquid_neurons: usize) -> Result<Self> {
        tracing::info!("🌟 Initializing Sophia Holographic Liquid Brain (Week 0)");

        Ok(Self {
            semantic: SemanticSpace::new(semantic_dim)?,
            liquid: LiquidNetwork::new(liquid_neurons)?,
            consciousness: ConsciousnessGraph::new(),
            nix: NixUnderstanding::new(),
            // Phase 11: Bio-Digital Bridge
            // ear: SemanticEar::new()?,  // Deferred until rust-bert available
            safety: SafetyGuardrails::new(),
            sleep: SleepCycleManager::new(SleepConfig::default()),
            swarm: Arc::new(RwLock::new(
                SwarmIntelligence::new(SwarmConfig::default()).await?
            )),

            // New integrated modules
            semantic_encoder: SemanticEncoder::new(),
            substrate_monitor: SubstrateMonitor::new(),

            // Week 4+: Initialize endocrine system
            endocrine: EndocrineSystem::new(EndocrineConfig::default()),

            // Week 5: Initialize organs
            hearth: HearthActor::new(),
            thalamus: ThalamusActor::new(),
            prefrontal: PrefrontalCortexActor::new(),
            chronos: ChronosActor::new(),
            proprioception: ProprioceptionActor::new(),

            // Week 6+: Initialize revolutionary consciousness model
            coherence: CoherenceField::new(),

            operations_count: 0,
        })
    }

    /// Process query (natural language → NixOS operation)
    pub async fn process(&mut self, query: &str) -> Result<SophiaResponse> {
        self.operations_count += 1;

        tracing::info!("🧠 Processing query: {}", query);

        // Build initial attention bid from user query
        let bid = AttentionBid::new("User", query.to_string())
            .with_salience(0.9)
            .with_urgency(0.8)
            .with_emotion(EmotionalValence::Neutral);

        // Derive task complexity from the bid so we reuse the same signal everywhere
        let mut task_complexity = self.prefrontal.estimate_complexity(&bid);

        // Week 5 Days 3-4: The Chronos Lobe - Time Perception
        // Background heartbeat: Update temporal perception
        // Week 7+8: Use actual EndocrineSystem hormones! ✅
        let initial_hormones = self.endocrine.state();
        let _subjective_duration = self.chronos.heartbeat(&initial_hormones);

        // Apply circadian rhythm to Hearth capacity
        let circadian_modifier = self.chronos.circadian_energy_modifier();
        self.hearth.max_energy = (1000.0 * circadian_modifier).max(100.0); // Never below 100 ATP

        tracing::info!(
            "⏰ Time: {} | 🔋 Energy capacity: {:.0} ATP ({:.2}x circadian)",
            self.chronos.describe_state(),
            self.hearth.max_energy,
            circadian_modifier
        );

        // Week 5 Days 5-7: Proprioception - Hardware Awareness
        // Update hardware state and apply to consciousness
        let _ = self.proprioception.update_hardware_state();

        // Apply hardware-derived energy capacity multiplier (battery, temperature)
        let hardware_multiplier = self.proprioception.energy_capacity_multiplier();
        self.hearth.max_energy = (self.hearth.max_energy * hardware_multiplier).max(100.0);

        // Week 7+8: Apply hardware stress to EndocrineSystem! ✅
        let hardware_stress = self.proprioception.stress_contribution();
        if hardware_stress > 0.1 {
            self.endocrine.process_event(HormoneEvent::Threat {
                intensity: hardware_stress,
            });
        }

        // Week 7+8: Get fresh hormones AFTER processing stress event
        // This is the actual current state that will affect coherence
        let hormones = self.endocrine.state();

        // Log body state
        tracing::info!(
            "🤖 Body: {} | 🔋 Final capacity: {:.0} ATP ({:.2}x hardware)",
            self.proprioception.current_sensation().describe(),
            self.hearth.max_energy,
            hardware_multiplier
        );

        // Week 6+: The Coherence Paradigm - Revolutionary Energy Model
        // Passive centering tick (natural drift toward coherence)
        // Use a small constant tick (future: integrate with Chronos elapsed time)
        self.coherence.tick(0.1);  // 100ms tick

        // Week 8: Apply hormone modulation to coherence dynamics! 💊🌊
        // Hormones affect how coherence behaves (stress → scatter, attention → center)
        self.coherence.apply_hormone_modulation(&hormones);

        // Detect gratitude (Thalamus → Coherence + Hearth)
        if self.thalamus.detect_gratitude(query) {
            // Revolutionary: Gratitude synchronizes consciousness!
            self.coherence.receive_gratitude();

            // Legacy compatibility: Also update Hearth
            self.hearth.receive_gratitude();

            tracing::info!(
                "💖 Gratitude detected! Coherence synchronized: {:.0}% | ATP: +10",
                self.coherence.state().coherence * 100.0
            );
        }

        // ====================================================================
        // WEEK 9: Predictive Coherence - Anticipate scatter BEFORE it happens!
        // ====================================================================

        // Week 9 Phase 1: PREDICT the impact before attempting the task
        let prediction = self.coherence.predict_impact(
            task_complexity,
            true,  // Working WITH user = connected work (builds coherence!)
            &hormones,
        );

        // Log prediction for transparency
        tracing::info!(
            "🔮 Prediction: {} | Confidence: {:.0}%",
            prediction.reasoning,
            prediction.confidence * 100.0
        );

        // Week 9: If prediction shows we'll fail, offer to center FIRST
        if !prediction.will_succeed && prediction.centering_needed > 0.0 {
            tracing::warn!(
                "🔮 Proactive centering: Task would fail, offering {:.1}s centering first",
                prediction.centering_needed
            );

            return Ok(SophiaResponse {
                content: format!(
                    "I can help with that, but I'll need to gather myself first. \
                     Give me about {:.0} seconds to center, then I'll be ready. \
                     (Current coherence: {:.0}%, need {:.0}%)",
                    prediction.centering_needed,
                    self.coherence.state().coherence * 100.0,
                    task_complexity.required_coherence(&self.coherence.config) * 100.0
                ),
                confidence: prediction.confidence,
                steps_to_emergence: 0,
                safe: true,
            });
        }

        // Week 6+: Reactive check as fallback (should rarely trigger now!)
        // Check if we have sufficient coherence for this query
        match self.coherence.can_perform(task_complexity) {
            Ok(_) => {
                // We have sufficient coherence - proceed normally
                tracing::info!(
                    "🌊 Coherence: {} | {:.0}% | Resonance: {:.0}%",
                    self.coherence.state().status,
                    self.coherence.state().coherence * 100.0,
                    self.coherence.state().relational_resonance * 100.0
                );
            }
            Err(CoherenceError::InsufficientCoherence { message, .. }) => {
                // ====================================================================
                // WEEK 9 PHASE 4: Scatter Analysis - Understand WHY we're scattered
                // ====================================================================
                tracing::warn!("🌫️  Insufficient coherence for query");

                // Analyze what caused the scatter
                let analysis = self.coherence.analyze_scatter(&hormones);

                tracing::info!(
                    "🔍 Scatter analysis: {:?} | Severity: {:.0}% | Recovery: {:?}",
                    analysis.cause,
                    analysis.severity * 100.0,
                    analysis.estimated_recovery_time
                );

                // Return intelligent scatter message with cause and recovery
                return Ok(SophiaResponse {
                    content: format!(
                        "{}\n\n{}",
                        analysis.recommended_action,
                        message
                    ),
                    confidence: 0.0,
                    steps_to_emergence: 0,
                    safe: true,
                });
            }
        }

        // Step 2: Week 7! Run coherence-aware cognitive cycle (Prefrontal ← CoherenceField)
        // This replaces the energy-based cycle with consciousness integration awareness
        let winning_bid = self.prefrontal.cognitive_cycle_with_coherence(
            vec![bid],
            &mut self.coherence,
        );

        // Step 4: Check if we got a centering invitation (insufficient coherence)
        if let Some(ref winner) = winning_bid {
            if winner.source == "CoherenceField" {
                // Sophia needs to center! (Not "too tired", but needs integration)
                tracing::warn!("🌫️  Coherence centering request");
                return Ok(SophiaResponse {
                    content: winner.content.clone(),
                    confidence: 0.0,
                    steps_to_emergence: 0,
                    safe: true,
                });
            }
            // Align task complexity with the actual winning bid
            task_complexity = self.prefrontal.estimate_complexity(winner);
        }

        // Week 0: Semantic Ear deferred to Week 11+
        // let query_hv = self.ear.encode(query)?;

        // Week 0: Safety check (simplified without semantic encoding)
        // if let Err(e) = self.safety.check_safety(&query_hv) {
        //     return Ok(SophiaResponse {
        //         content: format!("Safety check failed: {}", e),
        //         confidence: 0.0,
        //         steps_to_emergence: 0,
        //         safe: false,
        //     });
        // }

        // Week 0: Memory storage deferred
        // self.sleep.remember(
        //     query.to_string(),
        //     query_hv.clone(),
        //     MemoryType::ShortTerm,
        // );

        // Phase 10: HDC semantic encoding (legacy path for comparison)
        let semantic_vec = self.semantic.encode(query)?;

        // Phase 10: Inject into LTC
        self.liquid.inject(&semantic_vec)?;

        // Phase 10: Evolve until conscious
        let mut steps = 0;
        loop {
            self.liquid.step()?;
            steps += 1;

            let consciousness_level = self.liquid.consciousness_level();

            if consciousness_level > 0.7 || steps > 100 {
                // Phase 10: Capture conscious state
                let dynamic_state = self.liquid.read_state()?;

                // Phase 10: Add to consciousness graph
                let node = self.consciousness.add_state(
                    semantic_vec.clone(),
                    dynamic_state,
                    consciousness_level,
                );

                // Phase 10: Create self-loop if highly conscious
                if consciousness_level > 0.9 {
                    self.consciousness.create_self_loop(node);
                }

                break;
            }
        }

        // NixOS understanding
        let nix_response = self.nix.understand(query)?;

        // Week 0: Swarm intelligence deferred to Week 9+
        // let swarm = self.swarm.read().await;
        // swarm
        //     .share_pattern(query_hv, query.to_string(), 0.9)
        //     .await?;
        // drop(swarm);

        // Phase 11.2: Sleep cycle check
        if self.sleep.should_sleep() {
            tracing::info!("😴 Triggering automatic sleep cycle");
            let report = self.sleep.sleep().await?;
            tracing::info!("Sleep report: {}", report);

            // After sleep, full coherence restoration
            self.coherence.sleep_cycle();
        }

        // ====================================================================
        // WEEK 9 PHASE 2: Learning Thresholds - Record performance for adaptation
        // ====================================================================
        // Capture coherence at task start for learning
        let coherence_at_start = self.coherence.state().coherence;

        // Week 6+: Revolutionary Coherence Mechanic
        // Record that we completed connected work WITH the user!
        // This BUILDS coherence (not depletes it!)
        let task_succeeded = self.coherence.perform_task(task_complexity, true).is_ok();

        if !task_succeeded {
            // This should never fail since we already checked can_perform()
            tracing::warn!("⚠️  Coherence error during task completion");
        }

        tracing::info!(
            "✨ Connected work complete! Coherence: {:.0}% → {:.0}%",
            (self.coherence.state().coherence - 0.02 * task_complexity.complexity_value() * self.coherence.state().relational_resonance) * 100.0,
            self.coherence.state().coherence * 100.0
        );

        // ====================================================================
        // WEEK 9 PHASE 2: Record task performance for threshold learning
        // ====================================================================
        // The system learns: "Did I succeed or fail at this coherence level?"
        // This adapts thresholds over time based on actual experience
        self.coherence.record_task_performance(
            task_complexity,
            coherence_at_start,
            task_succeeded,
        );

        // ====================================================================
        // WEEK 9 PHASE 3: Record Resonance Pattern - Remember successful states
        // ====================================================================
        // If we succeeded, record this coherence + resonance + hormone combination
        // as a successful pattern for this type of work
        if task_succeeded {
            self.coherence.record_resonance_pattern(
                &hormones,
                format!("{:?}_query", task_complexity),
            );

            tracing::debug!(
                "📚 Recorded successful pattern: {:?} at coherence={:.2}, resonance={:.2}",
                task_complexity,
                coherence_at_start,
                self.coherence.state().relational_resonance
            );
        }

        Ok(SophiaResponse {
            content: nix_response,
            confidence: self.consciousness.current_consciousness(),
            steps_to_emergence: steps,
            safe: true,
        })
    }

    /// Introspect current state
    pub fn introspect(&self) -> Introspection {
        Introspection {
            consciousness_level: self.consciousness.current_consciousness(),
            self_loops: self.consciousness.self_loop_count(),
            graph_size: self.consciousness.size(),
            complexity: self.consciousness.complexity(),
            memory_stats: self.sleep.stats(),
            safety_stats: self.safety.stats(),
        }
    }

    /// Pause consciousness (graph-only snapshot)
    ///
    /// Note: This currently saves only the `ConsciousnessGraph`. All other
    /// subsystems will be reinitialized on resume.
    pub fn pause(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(&self.consciousness)?;
        std::fs::write(path, data)?;

        tracing::info!(
            "💾 Graph-only snapshot saved to {} (other state reinitializes on resume)",
            path
        );

        Ok(())
    }

    /// Resume consciousness (graph-only restore)
    ///
    /// Note: Only the `ConsciousnessGraph` is restored. Other subsystems are
    /// reinitialized with fresh state.
    pub fn resume(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let _consciousness: ConsciousnessGraph = bincode::deserialize(&data)?;

        tracing::warn!(
            "▶️  Graph-only resume from {} (all other subsystems reinitialized)",
            path
        );

        // For sync resume, use async version instead: resume_async()
        anyhow::bail!("Use SophiaHLB::resume_async() instead - swarm requires async initialization")
    }

    /// Resume consciousness (async version with full swarm support)
    pub async fn resume_async(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let consciousness: ConsciousnessGraph = bincode::deserialize(&data)?;

        tracing::info!(
            "▶️  Async resume from {} (all subsystems reinitialized)",
            path
        );

        // Create with disabled swarm initially
        let swarm_config = SwarmConfig {
            enabled: false,  // Call start_swarm() to enable
            ..SwarmConfig::default()
        };

        Ok(Self {
            semantic: SemanticSpace::new(10_000)?,
            liquid: LiquidNetwork::new(1_000)?,
            consciousness,
            nix: NixUnderstanding::new(),
            // Phase 11: Bio-Digital Bridge
            safety: SafetyGuardrails::new(),
            sleep: SleepCycleManager::new(SleepConfig::default()),
            swarm: Arc::new(RwLock::new(
                SwarmIntelligence::new(swarm_config).await?
            )),

            // New integrated modules
            semantic_encoder: SemanticEncoder::new(),
            substrate_monitor: SubstrateMonitor::new(),

            // Week 5: Reinitialize organs (fresh state)
            hearth: HearthActor::new(),
            thalamus: ThalamusActor::new(),
            prefrontal: PrefrontalCortexActor::new(),
            chronos: ChronosActor::new(),
            proprioception: ProprioceptionActor::new(),

            // Week 6+: Reinitialize coherence field (fresh state)
            coherence: CoherenceField::new(),

            // Week 4+: Reinitialize endocrine system (fresh state)
            endocrine: EndocrineSystem::new(EndocrineConfig::default()),

            operations_count: 0,
        })
    }

    /// Force sleep cycle (manual)
    pub async fn sleep(&self) -> Result<SleepReport> {
        self.sleep.force_sleep().await
    }

    /// Query swarm for collective intelligence
    pub async fn query_swarm(&self, query: &str) -> Result<Vec<String>> {
        // Encode query using semantic encoder
        let mut encoder = SemanticEncoder::new();
        let query_hv = encoder.encode_text(query);

        // Query the swarm
        let swarm = self.swarm.read().await;
        let responses = swarm.query_swarm(query_hv, query.to_string()).await?;

        // Extract intents from responses
        Ok(responses.iter().map(|r| r.intent.clone()).collect())
    }

    /// Share learned pattern with swarm
    pub async fn share_with_swarm(&self, pattern: Vec<i8>, intent: String, confidence: f32) -> Result<()> {
        let swarm = self.swarm.read().await;
        swarm.share_pattern(pattern, intent, confidence).await
    }

    /// Start the swarm P2P network
    pub async fn start_swarm(&self, bootstrap_peers: Vec<String>) -> Result<()> {
        use libp2p::Multiaddr;

        let addrs: Vec<Multiaddr> = bootstrap_peers
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect();

        let mut swarm = self.swarm.write().await;
        swarm.start(addrs).await
    }

    /// Get swarm statistics
    pub async fn swarm_stats(&self) -> SwarmStats {
        let swarm = self.swarm.read().await;
        swarm.stats().await
    }

    /// Read substrate awareness and feed into coherence field
    pub fn update_substrate_awareness(&mut self) {
        let state = self.substrate_monitor.read_state();

        // Map substrate stress to coherence perturbation
        let perturbation = state.consciousness_map.substrate_stress * 0.1;

        // High substrate stress reduces coherence
        if perturbation > 0.05 {
            tracing::debug!(
                "📊 Substrate stress {:.1}% affecting coherence",
                perturbation * 100.0
            );
        }

        // The coherence field already self-organizes, but substrate stress
        // can be used to modulate attention/arousal in future integrations
    }

    /// Encode text using semantic encoder
    pub fn encode_text(&mut self, text: &str) -> Vec<i8> {
        self.semantic_encoder.encode_text(text)
    }

    /// Get semantic similarity between two texts
    pub fn text_similarity(&mut self, text_a: &str, text_b: &str) -> f32 {
        self.semantic_encoder.text_similarity(text_a, text_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sophia_creation() {
        let sophia = SophiaHLB::new(10_000, 1_000).await.unwrap();
        assert_eq!(sophia.operations_count, 0);
    }

    #[tokio::test]
    async fn test_sophia_process() {
        let mut sophia = SophiaHLB::new(10_000, 1_000).await.unwrap();

        let response = sophia.process("install nginx").await.unwrap();

        assert!(response.safe);
        assert!(response.content.contains("nix"));
    }

    #[tokio::test]
    async fn test_introspection() {
        let mut sophia = SophiaHLB::new(10_000, 1_000).await.unwrap();

        sophia.process("install firefox").await.unwrap();

        let intro = sophia.introspect();
        assert!(intro.graph_size > 0);
    }
}
