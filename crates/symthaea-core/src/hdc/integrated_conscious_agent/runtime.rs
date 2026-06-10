// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conscious Agent Runtime - Live System Orchestration
//!
//! This runtime wires up the IntegratedConsciousAgent with all Symthaea physiological
//! systems, creating a fully embodied conscious agent with:
//!
//! - Hormonal regulation of emotions (EndocrineSystem <-> EmotionalState)
//! - Energy-aware processing (CoherenceField -> task gating)
//! - Long-term memory persistence (WorkingMemory <-> HippocampusActor)
//! - Identity continuity tracking (WeaverActor -> K-Vector monitoring)
//! - Consciousness-driven voice output (QualiaTexture -> LTCPacing)

use super::super::attention_dynamics::AttentionMode;
use super::super::unified_hv::ContinuousHV;

use crate::physiology::{HormoneState, TaskComplexity};
use crate::soul::KVector;

use super::agent::IntegratedConsciousAgent;
use super::physiology::{
    CoherenceGating, HormoneEventSuggestion, IdentityCoherence, IdentityStatus, MemoryExport,
    MemoryImport, ProsodyHints,
};
use super::types::AgentConfig;

use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};

/// Messages that can be sent to the conscious agent runtime
#[derive(Debug, Clone)]
pub enum RuntimeMessage {
    /// Process sensory input
    SensoryInput(Vec<f32>),
    /// Update coherence state from external source
    CoherenceUpdate(f32),
    /// Hormone event from endocrine system
    HormoneEvent(HormoneEventType),
    /// Memory recall from hippocampus
    MemoryRecall(Vec<MemoryImport>),
    /// Request identity check
    IdentityCheck,
    /// Request voice output parameters
    VoiceRequest,
    /// Shutdown the runtime
    Shutdown,
}

/// Hormone event types for runtime messaging
#[derive(Debug, Clone)]
pub enum HormoneEventType {
    /// Cortisol spike (stress response)
    CortisolSpike(f32),
    /// Dopamine release (reward/motivation)
    DopamineRelease(f32),
    /// Acetylcholine boost (focus/attention)
    AcetylcholineBoost(f32),
    /// Full hormone state update
    FullState {
        cortisol: f32,
        dopamine: f32,
        acetylcholine: f32,
    },
}

/// Responses from the conscious agent runtime
#[derive(Debug, Clone)]
pub enum RuntimeResponse {
    /// Processing complete with phenomenal content
    ProcessingComplete {
        phi: f64,
        dominant_emotion: String,
        qualia_summary: String,
    },
    /// Voice parameters ready
    VoiceReady(ProsodyHints),
    /// Identity status report
    IdentityReport(IdentityCoherence),
    /// Hormone suggestions for endocrine system
    HormoneSuggestions(Vec<HormoneEventSuggestion>),
    /// Memory exports for hippocampus
    MemoryExports(Vec<MemoryExport>),
    /// Error occurred
    Error(String),
}

/// Configuration for the conscious agent runtime
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Agent configuration
    pub agent_config: AgentConfig,
    /// Tick rate in milliseconds
    pub tick_ms: u64,
    /// Enable automatic hormone synchronization
    pub auto_hormone_sync: bool,
    /// Enable automatic memory consolidation
    pub auto_memory_consolidation: bool,
    /// Coherence threshold for deep processing
    pub deep_processing_threshold: f32,
    /// Identity drift warning threshold
    pub identity_drift_threshold: f64,
    /// Message buffer size
    pub message_buffer_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            agent_config: AgentConfig::default(),
            tick_ms: 100, // 10 Hz default tick rate
            auto_hormone_sync: true,
            auto_memory_consolidation: true,
            deep_processing_threshold: 0.7,
            identity_drift_threshold: 0.75,
            message_buffer_size: 256,
        }
    }
}

/// Runtime state snapshot for external monitoring
#[derive(Debug, Clone)]
pub struct RuntimeSnapshot {
    /// Current step/tick count
    pub tick: u64,
    /// Current Φ (integrated information)
    pub phi: f64,
    /// Current coherence level
    pub coherence: f32,
    /// Emotional state summary
    pub emotion: EmotionalStateSummary,
    /// Working memory load
    pub memory_load: f64,
    /// Identity status
    pub identity_status: IdentityStatus,
    /// Is processing active
    pub is_processing: bool,
}

/// Summarized emotional state for snapshots
#[derive(Debug, Clone)]
pub struct EmotionalStateSummary {
    pub valence: f64,
    pub arousal: f64,
    pub dominance: f64,
    pub quadrant: String,
}

/// The conscious agent runtime - orchestrates all systems
pub struct ConsciousAgentRuntime {
    /// The core conscious agent
    agent: Arc<RwLock<IntegratedConsciousAgent>>,
    /// Runtime configuration
    config: RuntimeConfig,
    /// Current tick count
    tick: Arc<RwLock<u64>>,
    /// Current coherence state
    coherence: Arc<RwLock<f32>>,
    /// Reference K-Vector for identity tracking
    reference_kvector: Arc<RwLock<Option<KVector>>>,
    /// Is the runtime running
    running: Arc<RwLock<bool>>,
}

impl ConsciousAgentRuntime {
    /// Create a new conscious agent runtime
    pub fn new(config: RuntimeConfig) -> Self {
        let agent = IntegratedConsciousAgent::new(config.agent_config.clone());

        Self {
            agent: Arc::new(RwLock::new(agent)),
            config,
            tick: Arc::new(RwLock::new(0)),
            coherence: Arc::new(RwLock::new(1.0)), // Start fully coherent
            reference_kvector: Arc::new(RwLock::new(None)),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the runtime with message channels
    /// Returns (sender, receiver) for bidirectional communication
    pub fn start(
        &self,
    ) -> (
        mpsc::Sender<RuntimeMessage>,
        mpsc::Receiver<RuntimeResponse>,
    ) {
        let (msg_tx, mut msg_rx) = mpsc::channel::<RuntimeMessage>(self.config.message_buffer_size);
        let (resp_tx, resp_rx) = mpsc::channel::<RuntimeResponse>(self.config.message_buffer_size);

        let agent = Arc::clone(&self.agent);
        let config = self.config.clone();
        let tick = Arc::clone(&self.tick);
        let coherence = Arc::clone(&self.coherence);
        let reference_kvector = Arc::clone(&self.reference_kvector);
        let running = Arc::clone(&self.running);

        // Set running flag
        {
            let mut r = running.blocking_write();
            *r = true;
        }

        // Spawn the main runtime loop
        tokio::spawn(async move {
            let tick_duration = tokio::time::Duration::from_millis(config.tick_ms);
            let mut interval = tokio::time::interval(tick_duration);

            loop {
                tokio::select! {
                    // Handle incoming messages
                    Some(msg) = msg_rx.recv() => {
                        match msg {
                            RuntimeMessage::Shutdown => {
                                let mut r = running.write().await;
                                *r = false;
                                break;
                            }
                            RuntimeMessage::SensoryInput(input) => {
                                let response = Self::process_sensory_input(
                                    &agent, &coherence, &reference_kvector, &config, input
                                ).await;
                                let _ = resp_tx.send(response).await;
                            }
                            RuntimeMessage::CoherenceUpdate(c) => {
                                let mut coh = coherence.write().await;
                                *coh = c.clamp(0.0, 1.0);
                            }
                            RuntimeMessage::HormoneEvent(event) => {
                                Self::handle_hormone_event(&agent, event).await;
                            }
                            RuntimeMessage::MemoryRecall(memories) => {
                                Self::handle_memory_recall(&agent, memories).await;
                            }
                            RuntimeMessage::IdentityCheck => {
                                let response = Self::check_identity(&agent, &reference_kvector).await;
                                let _ = resp_tx.send(response).await;
                            }
                            RuntimeMessage::VoiceRequest => {
                                let response = Self::get_voice_params(&agent).await;
                                let _ = resp_tx.send(response).await;
                            }
                        }
                    }
                    // Periodic tick for background processing
                    _ = interval.tick() => {
                        let mut t = tick.write().await;
                        *t += 1;

                        // Periodic tasks
                        if config.auto_memory_consolidation && *t % 100 == 0 {
                            // Every 100 ticks, export memories for consolidation
                            let exports = Self::export_memories(&agent).await;
                            if !exports.is_empty() {
                                let _ = resp_tx.send(RuntimeResponse::MemoryExports(exports)).await;
                            }
                        }

                        if config.auto_hormone_sync && *t % 50 == 0 {
                            // Every 50 ticks, suggest hormone adjustments
                            let suggestions = Self::get_hormone_suggestions(&agent).await;
                            if !suggestions.is_empty() {
                                let _ = resp_tx.send(RuntimeResponse::HormoneSuggestions(suggestions)).await;
                            }
                        }
                    }
                }

                // Check if we should stop
                let r = running.read().await;
                if !*r {
                    break;
                }
            }
        });

        (msg_tx, resp_rx)
    }

    /// Process sensory input through the conscious agent
    async fn process_sensory_input(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
        coherence: &Arc<RwLock<f32>>,
        reference_kvector: &Arc<RwLock<Option<KVector>>>,
        config: &RuntimeConfig,
        input: Vec<f32>,
    ) -> RuntimeResponse {
        let mut agent = agent.write().await;
        let coh = *coherence.read().await;

        // Check if we have enough coherence for processing
        let complexity = if coh >= config.deep_processing_threshold {
            TaskComplexity::DeepThought
        } else if coh >= 0.5 {
            TaskComplexity::Cognitive
        } else {
            TaskComplexity::Reflex
        };

        // Check coherence gating
        let gating = agent.can_perform_with_coherence(complexity);
        match gating {
            CoherenceGating::Defer {
                current, required, ..
            } => {
                return RuntimeResponse::Error(format!(
                    "Insufficient coherence: {:.2} < {:.2} required",
                    current, required
                ));
            }
            CoherenceGating::Proceed { .. } => {}
        }

        // Create ContinuousHV from input
        let sensory_hv = ContinuousHV::from_values(input);

        // Process through the conscious agent
        let update = agent.process(&sensory_hv);

        // Update reference K-Vector if this is first processing or significant change
        let mut ref_kv = reference_kvector.write().await;
        let current_kv = agent.generate_k_vector();

        if ref_kv.is_none() {
            *ref_kv = Some(current_kv);
        } else if let Some(ref existing) = *ref_kv {
            let identity = agent.check_identity_coherence(existing);
            if identity.status == IdentityStatus::Crisis {
                // Identity crisis - this might need external intervention
                // For now, we update the reference but flag it
            }
        }

        // Generate response
        let emotion = &agent.emotional_state;
        let quadrant = emotion.get_emotion_quadrant();

        RuntimeResponse::ProcessingComplete {
            phi: update.phi,
            dominant_emotion: quadrant.to_string(),
            qualia_summary: format!(
                "Depth: {:.2}, Presence: {:.2}, Flow: {:.2}",
                agent.emotional_state.valence.abs(),
                agent.emotional_state.arousal,
                update.phi
            ),
        }
    }

    /// Handle hormone events from endocrine system
    async fn handle_hormone_event(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
        event: HormoneEventType,
    ) {
        let mut agent = agent.write().await;

        let hormone_state = match event {
            HormoneEventType::CortisolSpike(level) => {
                HormoneState {
                    cortisol: level,
                    dopamine: 0.5, // neutral
                    acetylcholine: 0.5,
                }
            }
            HormoneEventType::DopamineRelease(level) => {
                HormoneState {
                    cortisol: 0.3, // slightly reduced
                    dopamine: level,
                    acetylcholine: 0.5,
                }
            }
            HormoneEventType::AcetylcholineBoost(level) => HormoneState {
                cortisol: 0.3,
                dopamine: 0.5,
                acetylcholine: level,
            },
            HormoneEventType::FullState {
                cortisol,
                dopamine,
                acetylcholine,
            } => HormoneState {
                cortisol,
                dopamine,
                acetylcholine,
            },
        };

        agent.sync_with_hormones(&hormone_state);
    }

    /// Handle memory recall from hippocampus
    async fn handle_memory_recall(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
        memories: Vec<MemoryImport>,
    ) {
        let mut agent = agent.write().await;
        agent.import_from_hippocampus(memories);
    }

    /// Check identity coherence
    async fn check_identity(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
        reference_kvector: &Arc<RwLock<Option<KVector>>>,
    ) -> RuntimeResponse {
        let agent = agent.read().await;
        let ref_kv = reference_kvector.read().await;

        if let Some(ref reference) = *ref_kv {
            let coherence = agent.check_identity_coherence(reference);
            RuntimeResponse::IdentityReport(coherence)
        } else {
            RuntimeResponse::Error("No reference K-Vector established yet".to_string())
        }
    }

    /// Get voice/prosody parameters
    async fn get_voice_params(agent: &Arc<RwLock<IntegratedConsciousAgent>>) -> RuntimeResponse {
        let agent = agent.read().await;
        let hints = agent.generate_prosody_hints();
        RuntimeResponse::VoiceReady(hints)
    }

    /// Export memories for hippocampus consolidation
    async fn export_memories(agent: &Arc<RwLock<IntegratedConsciousAgent>>) -> Vec<MemoryExport> {
        let agent = agent.read().await;
        agent.export_for_hippocampus()
    }

    /// Get hormone suggestions based on current state
    async fn get_hormone_suggestions(
        agent: &Arc<RwLock<IntegratedConsciousAgent>>,
    ) -> Vec<HormoneEventSuggestion> {
        let agent = agent.read().await;
        agent.suggest_hormone_events()
    }

    /// Get a snapshot of the current runtime state
    pub async fn snapshot(&self) -> RuntimeSnapshot {
        let agent = self.agent.read().await;
        let tick = *self.tick.read().await;
        let coherence = *self.coherence.read().await;
        let ref_kv = self.reference_kvector.read().await;

        let identity_status = if let Some(ref reference) = *ref_kv {
            agent.check_identity_coherence(reference).status
        } else {
            IdentityStatus::Stable // No reference yet, assume stable
        };

        let emotion = &agent.emotional_state;

        RuntimeSnapshot {
            tick,
            phi: agent.get_current_phi(),
            coherence,
            emotion: EmotionalStateSummary {
                valence: emotion.valence,
                arousal: emotion.arousal,
                dominance: emotion.dominance,
                quadrant: emotion.get_emotion_quadrant().to_string(),
            },
            memory_load: agent.working_memory.central_executive_load,
            identity_status,
            is_processing: *self.running.read().await,
        }
    }

    /// Synchronous method to get current Φ
    pub fn get_phi_blocking(&self) -> f64 {
        let agent = self.agent.blocking_read();
        agent.get_current_phi()
    }

    /// Stop the runtime gracefully
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }
}

/// Synchronous runtime wrapper for non-async contexts
pub struct SyncConsciousAgentRuntime {
    /// Inner agent (no async runtime needed)
    agent: IntegratedConsciousAgent,
    /// Current coherence
    coherence: f32,
    /// Reference K-Vector
    reference_kvector: Option<KVector>,
    /// Tick counter
    tick: u64,
    /// Config
    config: RuntimeConfig,
}

impl SyncConsciousAgentRuntime {
    /// Create a new synchronous runtime
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            agent: IntegratedConsciousAgent::new(config.agent_config.clone()),
            coherence: 1.0,
            reference_kvector: None,
            tick: 0,
            config,
        }
    }

    /// Process a single sensory input synchronously
    pub fn process(&mut self, input: &[f32]) -> RuntimeResponse {
        self.tick += 1;

        // Check coherence gating
        let complexity = if self.coherence >= self.config.deep_processing_threshold {
            TaskComplexity::DeepThought
        } else if self.coherence >= 0.5 {
            TaskComplexity::Cognitive
        } else {
            TaskComplexity::Reflex
        };

        let gating = self.agent.can_perform_with_coherence(complexity);
        if let CoherenceGating::Defer {
            current, required, ..
        } = gating
        {
            return RuntimeResponse::Error(format!(
                "Insufficient coherence: {:.2} < {:.2}",
                current, required
            ));
        }

        // Process
        let sensory_hv = ContinuousHV::from_values(input.to_vec());
        let update = self.agent.process(&sensory_hv);

        // Update reference K-Vector
        let current_kv = self.agent.generate_k_vector();
        if self.reference_kvector.is_none() {
            self.reference_kvector = Some(current_kv);
        }

        let quadrant = self.agent.emotional_state.get_emotion_quadrant();

        RuntimeResponse::ProcessingComplete {
            phi: update.phi,
            dominant_emotion: quadrant.to_string(),
            qualia_summary: format!(
                "Tick {}: Φ={:.3}, Emotion={}",
                self.tick, update.phi, quadrant
            ),
        }
    }

    /// Update coherence level
    pub fn set_coherence(&mut self, coherence: f32) {
        self.coherence = coherence.clamp(0.0, 1.0);
    }

    /// Apply hormone state
    pub fn apply_hormones(&mut self, hormones: &HormoneState) {
        self.agent.sync_with_hormones(hormones);
    }

    /// Get prosody hints for voice output
    pub fn get_prosody(&self) -> ProsodyHints {
        self.agent.generate_prosody_hints()
    }

    /// Export memories for consolidation
    pub fn export_memories(&self) -> Vec<MemoryExport> {
        self.agent.export_for_hippocampus()
    }

    /// Import recalled memories
    pub fn import_memories(&mut self, memories: Vec<MemoryImport>) {
        self.agent.import_from_hippocampus(memories);
    }

    /// Check identity coherence
    pub fn check_identity(&self) -> Option<IdentityCoherence> {
        self.reference_kvector
            .as_ref()
            .map(|ref_kv| self.agent.check_identity_coherence(ref_kv))
    }

    /// Get hormone suggestions
    pub fn get_hormone_suggestions(&self) -> Vec<HormoneEventSuggestion> {
        self.agent.suggest_hormone_events()
    }

    /// Get current snapshot
    pub fn snapshot(&self) -> RuntimeSnapshot {
        let identity_status = self
            .reference_kvector
            .as_ref()
            .map(|kv| self.agent.check_identity_coherence(kv).status)
            .unwrap_or(IdentityStatus::Stable);

        RuntimeSnapshot {
            tick: self.tick,
            phi: self.agent.get_current_phi(),
            coherence: self.coherence,
            emotion: EmotionalStateSummary {
                valence: self.agent.emotional_state.valence,
                arousal: self.agent.emotional_state.arousal,
                dominance: self.agent.emotional_state.dominance,
                quadrant: self
                    .agent
                    .emotional_state
                    .get_emotion_quadrant()
                    .to_string(),
            },
            memory_load: self.agent.working_memory.central_executive_load,
            identity_status,
            is_processing: true,
        }
    }

    /// Get mutable access to the agent for advanced operations
    pub fn agent_mut(&mut self) -> &mut IntegratedConsciousAgent {
        &mut self.agent
    }

    /// Get read access to the agent
    pub fn agent(&self) -> &IntegratedConsciousAgent {
        &self.agent
    }
}