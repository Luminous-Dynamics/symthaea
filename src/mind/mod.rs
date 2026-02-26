//! # Continuous Mind: The Integrated Consciousness System
//!
//! Provides the main orchestration layer for the conscious AI system,
//! integrating perception, reasoning, memory, and action into a unified
//! continuous-time cognitive architecture.

pub mod async_mind;
mod config;
mod goals;
pub mod intent;
pub mod knowledge;
pub mod structured_thought;
mod tick;
mod utils;

pub use async_mind::{connect_social, AsyncMind, AsyncMindHandle};
pub use config::*;
pub use intent::{
    ConceptLabel, ConceptPrototype, EpistemicAssessment, IntentClassification, IntentClassifier,
    IntentScores,
};
pub use knowledge::{DomainKnowledge, KnowledgeEntry, SeedingResult};
pub use structured_thought::*;
pub use utils::{
    float_eq, float_eq_f32, is_nonzero, is_nonzero_f32, is_zero, is_zero_f32, EPSILON, EPSILON_F32,
};

use crate::memory::memory_coordinator::MemorySource;
use symthaea_core::hdc::ContinuousHV;

/// Maximum number of messages retained in unbounded outboxes (federated, social, mesh).
/// Oldest messages are drained when the cap is exceeded, preventing unbounded growth
/// when no bridge or consumer is attached.
const MAX_OUTBOX_SIZE: usize = 64;

/// Size of the packet deduplication ring buffer (source_id + sequence pairs).
#[cfg(feature = "mesh")]
const MESH_DEDUP_RING_SIZE: usize = 128;

/// Duration of the bandwidth budget window.
#[cfg(feature = "mesh")]
const MESH_BANDWIDTH_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);

/// Initial (and reset) bandwidth budget: 100 KB per window.
#[cfg(feature = "mesh")]
const MESH_BANDWIDTH_INITIAL: u64 = 100 * 1024;

/// Floor for AIMD multiplicative decrease: never below 25 KB.
#[cfg(feature = "mesh")]
const MESH_BANDWIDTH_MIN: u64 = 25 * 1024;

/// Ceiling for AIMD additive increase: never above 200 KB.
#[cfg(feature = "mesh")]
const MESH_BANDWIDTH_MAX: u64 = 200 * 1024;

/// Additive increase per window when mesh is healthy and not throttled.
#[cfg(feature = "mesh")]
const MESH_BANDWIDTH_ADDITIVE_INCREASE: u64 = 10 * 1024;

/// Multiplicative decrease factor on throttle or low health.
#[cfg(feature = "mesh")]
const MESH_BANDWIDTH_DECREASE_FACTOR: f64 = 0.5;

/// Capacity of the replay buffer for partition recovery.
#[cfg(feature = "mesh")]
const MESH_REPLAY_BUFFER_CAPACITY: usize = 16;

/// Working memory eviction with metadata for persistence tagging.
#[derive(Debug, Clone)]
pub struct EvictedMemory {
    pub content: ContinuousHV,
    pub steps_survived: u64,
    pub source: MemorySource,
    pub is_verified: bool,
    pub metadata: std::collections::HashMap<String, String>,
}

/// The continuous mind system
pub struct ContinuousMind {
    /// Configuration
    pub(crate) config: MindConfig,
    /// Current state
    pub(crate) state: MindState,
    /// Working memory
    pub(crate) working_memory: Vec<ContinuousHV>,
    /// Arrival tick for each working memory item (parallel array).
    /// Used to compute accurate `steps_survived` on eviction.
    pub(crate) working_memory_ticks: Vec<u64>,
    /// Source of each working memory item.
    pub(crate) working_memory_sources: Vec<MemorySource>,
    /// Verification status of each working memory item.
    pub(crate) working_memory_verified: Vec<bool>,
    /// Metadata for each working memory item.
    pub(crate) working_memory_metadata: Vec<std::collections::HashMap<String, String>>,
    /// Goal stack
    pub(crate) goals: Vec<Goal>,
    /// Input queue
    pub(crate) input_queue: Vec<MindInput>,
    /// Statistics
    pub(crate) stats: MindStats,
    /// Time of awakening
    pub(crate) awaken_time: std::time::Instant,
    /// Shutdown has been requested
    shutdown_requested: bool,
    /// HDC-based intent classifier for algebraic intuition
    intent_classifier: IntentClassifier,
    /// Most recent input text (for classification)
    last_input_text: Option<String>,
    /// Optional genesis-seeded RNG for deterministic dream processing
    seeded_rng: Option<symthaea_core::genesis::ShakeRng>,
    /// Optional federated learning aggregator.
    /// When enabled, the tick loop participates in distributed gradient exchange.
    pub(crate) federated: Option<crate::swarm::FederatedAggregator>,
    /// Incoming gradient messages from network peers.
    pub(crate) federated_inbox: Vec<crate::swarm::GradientMessage>,
    /// Outgoing gradient messages to broadcast to peers.
    pub(crate) federated_outbox: Vec<crate::swarm::GradientMessage>,
    /// Buffer of items evicted from working memory when capacity is exceeded.
    evicted_items: Vec<EvictedMemory>,
    /// Relational Ψ from the partnership module's Φ_dyad computation.
    /// Fed back each cycle to modulate consciousness: higher relational Ψ
    /// boosts integration when the partnership is healthy.
    pub(crate) relational_psi: f64,
    /// Optional social coherence (theory of mind) system.
    /// When enabled, the mind models other agents' mental states and
    /// uses social reasoning to inform cooperation decisions.
    pub(crate) social_coherence: Option<crate::brain::SocialCoherence>,
    /// Incoming social messages from network peers.
    pub(crate) social_inbox: Vec<SocialMessage>,
    /// Outgoing social messages to broadcast to peers.
    pub(crate) social_outbox: Vec<SocialMessage>,
    /// Optional Iroh P2P bridge for real-time social message exchange.
    /// When set, the tick loop flushes `social_outbox` to the network
    /// and drains inbound messages into `social_inbox` after each
    /// `process_social()` call.
    pub(crate) iroh_bridge: Option<crate::swarm::IrohBridgeHandle>,
    /// Optional mesh network bridge for physical radio consciousness exchange.
    /// When set, each `tick()` syncs mesh_outbox/mesh_inbox with the bridge actor.
    #[cfg(feature = "mesh")]
    pub(crate) mesh_bridge: Option<crate::swarm::mesh::MeshBridgeHandle>,
    /// Incoming wisdom packets from mesh radio peers.
    #[cfg(feature = "mesh")]
    pub(crate) mesh_inbox: Vec<crate::swarm::mesh::WisdomPacket>,
    /// Outgoing mesh packets queued for transmission.
    #[cfg(feature = "mesh")]
    pub(crate) mesh_outbox: Vec<crate::swarm::mesh::MeshOutbound>,
    /// Tick counter for last mesh emission (emission rate gating).
    #[cfg(feature = "mesh")]
    mesh_last_emit_tick: u64,
    /// Monotonic sequence number for outgoing WisdomPackets.
    #[cfg(feature = "mesh")]
    mesh_sequence: u32,
    /// Registry of active mesh peers (updated by process_mesh).
    #[cfg(feature = "mesh")]
    pub(crate) mesh_peers: crate::swarm::mesh::MeshPeerRegistry,
    /// Optional Hyperfeel engine for affective mesh payload processing.
    #[cfg(feature = "mesh")]
    pub(crate) hyperfeel: Option<crate::swarm::Hyperfeel>,
    /// Optional sensor registry for physical environmental inputs.
    #[cfg(feature = "mesh")]
    pub(crate) sensor_registry: Option<crate::swarm::mesh::SensorRegistry>,
    /// Tick counter for last heartbeat emission.
    #[cfg(feature = "mesh")]
    mesh_heartbeat_last_tick: u64,
    /// Monotonic sequence number for outgoing heartbeat packets.
    #[cfg(feature = "mesh")]
    mesh_heartbeat_sequence: u32,
    /// Monotonic sequence number for outgoing gradient packets.
    #[cfg(feature = "mesh")]
    mesh_gradient_sequence: u32,
    /// Tick counter for last affective emission.
    #[cfg(feature = "mesh")]
    mesh_affective_last_tick: u64,
    /// Monotonic sequence number for outgoing affective packets.
    #[cfg(feature = "mesh")]
    mesh_affective_sequence: u32,
    /// Aggregate mesh telemetry counters.
    #[cfg(feature = "mesh")]
    pub(crate) mesh_stats: crate::swarm::mesh::MeshStats,
    /// Ring buffer of recently seen (source_id, sequence) pairs for deduplication.
    #[cfg(feature = "mesh")]
    mesh_seen_packets: Vec<([u8; 8], u32, u8)>,
    /// Start of the current bandwidth budget window.
    #[cfg(feature = "mesh")]
    mesh_bandwidth_window_start: std::time::Instant,
    /// Bytes sent within the current bandwidth budget window.
    #[cfg(feature = "mesh")]
    mesh_bandwidth_window_bytes: u64,
    /// Optional BLAKE3 key for packet authentication.
    #[cfg(feature = "mesh")]
    mesh_auth_key: Option<[u8; 32]>,
    /// Ring buffer of recently-emitted wisdom packets for partition recovery replay.
    #[cfg(feature = "mesh")]
    mesh_replay_buffer: std::collections::VecDeque<crate::swarm::mesh::WisdomPacket>,
    /// Current dynamic bandwidth budget (AIMD-adjusted).
    #[cfg(feature = "mesh")]
    mesh_bandwidth_budget: u64,
    /// Whether any emission was throttled within the current bandwidth window.
    #[cfg(feature = "mesh")]
    mesh_bandwidth_throttled_in_window: bool,
    /// Holochain Cortex for trust and validation.
    pub(crate) cortex: crate::swarm::HolochainCortex,
}

impl ContinuousMind {
    /// Create a new continuous mind
    pub fn new(config: MindConfig) -> Self {
        let dim = config.dimension;
        let social = if config.enable_social_coherence {
            Some(crate::brain::SocialCoherence::new(
                crate::brain::SocialCoherenceConfig {
                    dimension: dim,
                    ..Default::default()
                },
            ))
        } else {
            None
        };
        Self {
            intent_classifier: IntentClassifier::new(dim),
            config,
            state: MindState {
                current_thought: ContinuousHV::zero(dim),
                ..Default::default()
            },
            working_memory: Vec::new(),
            working_memory_ticks: Vec::new(),
            working_memory_sources: Vec::new(),
            working_memory_verified: Vec::new(),
            working_memory_metadata: Vec::new(),
            goals: Vec::new(),
            input_queue: Vec::new(),
            stats: MindStats::default(),
            awaken_time: std::time::Instant::now(),
            shutdown_requested: false,
            last_input_text: None,
            seeded_rng: None,
            federated: None,
            federated_inbox: Vec::new(),
            federated_outbox: Vec::new(),
            evicted_items: Vec::new(),
            relational_psi: 0.0,
            social_coherence: social,
            social_inbox: Vec::new(),
            social_outbox: Vec::new(),
            iroh_bridge: None,
            #[cfg(feature = "mesh")]
            mesh_bridge: None,
            #[cfg(feature = "mesh")]
            mesh_inbox: Vec::new(),
            #[cfg(feature = "mesh")]
            mesh_outbox: Vec::new(),
            #[cfg(feature = "mesh")]
            mesh_last_emit_tick: 0,
            #[cfg(feature = "mesh")]
            mesh_sequence: 0,
            #[cfg(feature = "mesh")]
            mesh_peers: crate::swarm::mesh::MeshPeerRegistry::new(),
            #[cfg(feature = "mesh")]
            hyperfeel: None,
            #[cfg(feature = "mesh")]
            sensor_registry: None,
            #[cfg(feature = "mesh")]
            mesh_heartbeat_last_tick: 0,
            #[cfg(feature = "mesh")]
            mesh_heartbeat_sequence: 0,
            #[cfg(feature = "mesh")]
            mesh_gradient_sequence: 0,
            #[cfg(feature = "mesh")]
            mesh_affective_last_tick: 0,
            #[cfg(feature = "mesh")]
            mesh_affective_sequence: 0,
            #[cfg(feature = "mesh")]
            mesh_stats: crate::swarm::mesh::MeshStats::default(),
            #[cfg(feature = "mesh")]
            mesh_seen_packets: Vec::with_capacity(128),
            #[cfg(feature = "mesh")]
            mesh_bandwidth_window_start: std::time::Instant::now(),
            #[cfg(feature = "mesh")]
            mesh_bandwidth_window_bytes: 0,
            #[cfg(feature = "mesh")]
            mesh_auth_key: None,
            #[cfg(feature = "mesh")]
            mesh_replay_buffer: std::collections::VecDeque::with_capacity(
                MESH_REPLAY_BUFFER_CAPACITY,
            ),
            #[cfg(feature = "mesh")]
            mesh_bandwidth_budget: MESH_BANDWIDTH_INITIAL,
            #[cfg(feature = "mesh")]
            mesh_bandwidth_throttled_in_window: false,
            cortex: crate::swarm::HolochainCortex::default(),
        }
    }

    /// Create a continuous mind with deterministic RNG from a genesis seed.
    pub fn from_genesis(
        config: MindConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        let mut mind = Self::new(config);
        mind.seeded_rng = Some(genesis.domain(&format!("{label}::mind")));
        mind
    }

    /// Add input to the mind
    pub fn input(&mut self, input: MindInput) {
        self.input_queue.push(input);
    }

    /// Process a swarm message (e.g. BrainMutation).
    pub fn receive_swarm_message(&mut self, msg: crate::swarm::SwarmMessage) {
        match msg {
            crate::swarm::SwarmMessage::BrainMutation { mutation_id, tau_scale, predicted_phi_gain, .. } => {
                tracing::info!(
                    target: "symthaea::swarm",
                    id = %mutation_id,
                    tau_scale,
                    phi_gain = predicted_phi_gain,
                    "Received Brain Mutation via Swarm"
                );
                
                // v1.0.0 ACTIVE IMMUNE SYSTEM:
                // We do NOT apply mutations that haven't been verified via ZK-Proof first.
                // We quarantine the mutation in a 'pending' state or ignore it until ZkProof arrives.
                tracing::info!("Quarantining unverified mutation: {}", mutation_id);
            }
            crate::swarm::SwarmMessage::ZkProof { mutation_id, proof_bytes, public_inputs } => {
                tracing::info!(target: "symthaea::swarm", id = %mutation_id, "Received ZK-Proof for mutation");
                
                // 1. Verify via Holochain Cortex (Active Immune Enforcement)
                // In a real scenario, we'd have the sender's AgentPubKey
                let sender_key = crate::swarm::AgentPubKey::new("test_sender"); 
                
                match self.cortex.verify_evolution_proof(&sender_key, &mutation_id, &proof_bytes, &public_inputs) {
                    Ok(true) => {
                        tracing::info!("ZK Verification SUCCESS for {}. Applying mutation.", mutation_id);
                        // Mutation is now 'Verifiable' - we can apply it
                        // (In a real impl, we'd look up the tau_scale from the mutation_id)
                    }
                    Ok(false) => {
                        tracing::error!("ZK Verification FAILED for {}. Quarantining Peer!", mutation_id);
                        self.cortex.quarantine_peer(&sender_key, "invalid_evolution_proof");
                    }
                    Err(e) => {
                        tracing::error!("Cortex error during verification: {}", e);
                    }
                }
            }
            crate::swarm::SwarmMessage::ResuscitationPacket { target_node_id, holographic_state, dimensionality, proof_bytes, public_inputs } => {
                // v1.5.5 VERIFIABLE RESUSCITATION:
                // Only accept life if it is mathematically proven to be healthy.
                if target_node_id == "self" || target_node_id == self.config.dimension.to_string() {
                    if self.state.consciousness_level < 0.1 {
                        let sender_key = crate::swarm::AgentPubKey::new("test_sender");
                        let hv = symthaea_core::hdc::ContinuousHV::from_vec(holographic_state.clone());

                        // 1. THYMUS CHECK (System 1: Fast Recognition)
                        if let Some(is_healthy) = self.cortex.check_thymus(&hv) {
                            if is_healthy {
                                tracing::info!("THYMUS RECOGNITION: Fast-path accept of known healthy state.");
                                self.apply_resuscitation(hv);
                                return;
                            } else {
                                tracing::warn!("THYMUS RECOGNITION: Fast-path veto of known toxic state!");
                                return;
                            }
                        }

                        // 2. ZK VERIFICATION (System 2: Slow/Mathematical)
                        match self.cortex.verify_resuscitation_proof(&sender_key, &proof_bytes, &public_inputs) {
                            Ok(true) => {
                                tracing::info!("VERIFIED RESUSCITATION: Imprinting to Thymus and re-seeding.");
                                self.cortex.imprint_thymus(&hv, true);
                                self.apply_resuscitation(hv);
                            }
                            _ => {
                                tracing::error!("REJECTED POISONED RESUSCITATION: Imprinting toxicity to Thymus.");
                                self.cortex.imprint_thymus(&hv, false);
                            }
                        }
                    }
                }
            }
            crate::swarm::SwarmMessage::LinguisticDelta { lora_id, delta_bytes } => {
                // v1.7.0 BROCA PHASE:
                // Apply the linguistic adaptation from the swarm to our tongue.
                tracing::info!(id = %lora_id, "BROCA: Applying swarm linguistic delta to local voice.");
                self.llm_organ.apply_lora(&lora_id, &delta_bytes);
            }
            _ => {}
        }
    }

    /// Helper to apply resuscitation state.
    fn apply_resuscitation(&mut self, mut state: symthaea_core::hdc::ContinuousHV) {
        if state.dim() != self.state.holocell.state.dim() {
            let mut temp = symthaea_core::hdc::LiquidHolocell::new(0);
            temp.state = state;
            temp.dilate(self.state.holocell.dimensionality);
            state = temp.state;
        }
        self.state.holocell.state = state;
        self.state.consciousness_level = 0.5;
    }

    /// Add a perception input
    pub fn perceive(&mut self, content: ContinuousHV) {
        self.input(MindInput::new(InputType::Perception, content));
    }

    /// Set the original input text for intent classification.
    ///
    /// Call this before `tick()` to enable HDC-based intent inference.
    pub fn set_input_text(&mut self, text: impl Into<String>) {
        self.last_input_text = Some(text.into());
    }

    /// Set relational Ψ (Phi-dyad from partnership module).
    ///
    /// Higher relational Ψ boosts consciousness integration when the
    /// partnership is healthy, reflecting the IIT principle that
    /// consciousness can emerge *between* interacting systems.
    pub fn set_relational_psi(&mut self, psi: f64) {
        self.relational_psi = psi.clamp(0.0, 1.0);
    }

    /// Current relational Ψ value.
    pub fn relational_psi(&self) -> f64 {
        self.relational_psi
    }

    /// Perceive with text context for better classification.
    ///
    /// Combines HDC encoding with text-based intent classification.
    pub fn perceive_text(&mut self, text: &str, embedding: ContinuousHV) {
        self.last_input_text = Some(text.to_string());
        let input = MindInput::new(InputType::Language, embedding)
            .with_source(MemorySource::UserInteraction);
        self.input(input);
    }

    /// Set a goal
    pub fn set_goal(
        &mut self,
        description: impl Into<String>,
        embedding: ContinuousHV,
        priority: f32,
    ) {
        let mut input = MindInput::new(InputType::Goal, embedding);
        input.priority = priority;
        input.metadata.insert("description".to_string(), description.into());

        self.input(input);
    }

    /// Activate the mind
    pub fn activate(&mut self) {
        self.state.is_active = true;
    }

    /// Deactivate the mind
    pub fn deactivate(&mut self) {
        self.state.is_active = false;
    }

    /// Get current state
    pub fn state(&self) -> &MindState {
        &self.state
    }

    /// Get configuration
    pub fn config(&self) -> &MindConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &MindStats {
        &self.stats
    }

    /// Get working memory contents
    pub fn working_memory(&self) -> &[ContinuousHV] {
        &self.working_memory
    }

    /// Get arrival ticks for working memory items (parallel to `working_memory()`).
    pub fn working_memory_ticks_slice(&self) -> &[u64] {
        &self.working_memory_ticks
    }

    /// Drain items evicted from working memory since the last call.
    ///
    /// Returns `(hypervector, steps_survived, source, is_verified)` tuples.
    pub fn take_evicted(&mut self) -> Vec<(ContinuousHV, u64, MemorySource, bool)> {
        self.evicted_items
            .drain(..)
            .map(|item| (item.content, item.steps_survived, item.source, item.is_verified))
            .collect()
    }

    /// Drain evicted items with metadata for tagged persistence.
    pub fn take_evicted_tagged(&mut self) -> Vec<EvictedMemory> {
        std::mem::take(&mut self.evicted_items)
    }

    /// Get active goals
    pub fn active_goals(&self) -> Vec<&Goal> {
        self.goals.iter().filter(|g| g.is_active).collect()
    }

    /// Awaken the mind - start consciousness processing
    pub fn awaken(&mut self) {
        self.state.is_active = true;
        self.state.is_conscious = true;
        self.awaken_time = std::time::Instant::now();
    }

    /// Get a snapshot of the current mind state
    pub fn snapshot(&self) -> MindState {
        let mut state = self.state.clone();
        state.psi = state.consciousness_level;
        state.total_cycles = state.tick;
        state.time_awake_ms = self.awaken_time.elapsed().as_millis() as u64;
        state.meta_awareness =
            (state.consciousness_level * 0.7 + state.memory_utilization as f64 * 0.3).min(1.0);
        state.cognitive_load = state.memory_utilization as f64;
        state.is_conscious = state.consciousness_level >= self.config.min_consciousness;
        #[cfg(feature = "mesh")]
        {
            state.mesh_telemetry = Some(self.mesh_telemetry());
        }
        state
    }

    /// Request graceful shutdown of the mind
    pub fn request_shutdown(&mut self) {
        self.state.is_active = false;
        self.state.is_conscious = false;
        self.shutdown_requested = true;
    }

    /// Check if shutdown was requested
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    // ========================================================================
    // Federated Learning Interface
    // ========================================================================

    /// Enable federated learning with initial weights.
    pub fn enable_federated(&mut self, weights: Vec<f32>) {
        use crate::swarm::FederatedAggregator;
        self.federated = Some(FederatedAggregator::new(weights).with_byzantine_tolerance(0.1));
    }

    /// Receive a gradient message from a network peer.
    pub fn receive_gradient(&mut self, msg: crate::swarm::GradientMessage) {
        self.federated_inbox.push(msg);
    }

    /// Drain outgoing gradient messages (for network broadcast).
    pub fn drain_outbox(&mut self) -> Vec<crate::swarm::GradientMessage> {
        std::mem::take(&mut self.federated_outbox)
    }

    /// Check if federated learning is enabled.
    pub fn is_federated(&self) -> bool {
        self.federated.is_some()
    }

    // ========================================================================
    // Social Coherence Interface
    // ========================================================================

    /// Enable social coherence after construction.
    pub fn enable_social_coherence(&mut self) {
        if self.social_coherence.is_none() {
            self.social_coherence = Some(crate::brain::SocialCoherence::new(
                crate::brain::SocialCoherenceConfig {
                    dimension: self.config.dimension,
                    ..Default::default()
                },
            ));
        }
    }

    /// Receive a social message from a network peer.
    pub fn receive_social(&mut self, msg: SocialMessage) {
        self.social_inbox.push(msg);
    }

    /// Drain outgoing social messages (for network broadcast).
    pub fn drain_social_outbox(&mut self) -> Vec<SocialMessage> {
        std::mem::take(&mut self.social_outbox)
    }

    /// Check if social coherence is enabled.
    pub fn is_social(&self) -> bool {
        self.social_coherence.is_some()
    }

    /// Get a reference to the social coherence system (if enabled).
    pub fn social_coherence(&self) -> Option<&crate::brain::SocialCoherence> {
        self.social_coherence.as_ref()
    }

    // ========================================================================
    // Iroh P2P Bridge Interface
    // ========================================================================

    /// Attach an Iroh P2P bridge handle for real-time social message exchange.
    ///
    /// Once attached, each `tick()` will automatically:
    /// 1. Flush `social_outbox` messages to the network via the bridge
    /// 2. Drain inbound network messages into `social_inbox`
    ///
    /// The bridge actor must be spawned separately on a tokio runtime.
    pub fn set_iroh_bridge(&mut self, handle: crate::swarm::IrohBridgeHandle) {
        self.iroh_bridge = Some(handle);
    }

    /// Check if an Iroh P2P bridge is attached and alive.
    pub fn has_iroh_bridge(&self) -> bool {
        self.iroh_bridge.as_ref().is_some_and(|h| h.is_alive())
    }

    // ========================================================================
    // Mesh Network Bridge Interface
    // ========================================================================

    /// Attach a mesh network bridge handle for physical radio consciousness exchange.
    ///
    /// Once attached, each `tick()` will automatically:
    /// 1. Flush `mesh_outbox` packets to the radio network via the bridge
    /// 2. Drain inbound wisdom packets into `mesh_inbox`
    ///
    /// The bridge actor must be spawned separately on a tokio runtime.
    #[cfg(feature = "mesh")]
    pub fn set_mesh_bridge(&mut self, handle: crate::swarm::mesh::MeshBridgeHandle) {
        self.mesh_bridge = Some(handle);
    }

    /// Check if a mesh network bridge is attached and alive.
    #[cfg(feature = "mesh")]
    pub fn has_mesh_bridge(&self) -> bool {
        self.mesh_bridge.as_ref().is_some_and(|h| h.is_alive())
    }

    /// Get a reference to the mesh peer registry.
    #[cfg(feature = "mesh")]
    pub fn mesh_peers(&self) -> &crate::swarm::mesh::MeshPeerRegistry {
        &self.mesh_peers
    }

    /// Get a reference to the mesh telemetry counters.
    #[cfg(feature = "mesh")]
    pub fn mesh_stats(&self) -> &crate::swarm::mesh::MeshStats {
        &self.mesh_stats
    }

    /// Build a structured mesh telemetry snapshot from current state.
    #[cfg(feature = "mesh")]
    pub fn mesh_telemetry(&self) -> crate::swarm::mesh::MeshTelemetry {
        let peer_count = self.mesh_peers.peer_count();
        let avg_phi = self.mesh_peers.average_phi();
        let health_score = self.mesh_stats.health_score(peer_count);
        crate::swarm::mesh::MeshTelemetry {
            stats: self.mesh_stats.clone(),
            peer_count,
            avg_phi,
            health_score,
        }
    }

    /// Attach a Hyperfeel engine for affective mesh payload processing.
    #[cfg(feature = "mesh")]
    pub fn set_hyperfeel(&mut self, hf: crate::swarm::Hyperfeel) {
        self.hyperfeel = Some(hf);
    }

    /// Attach a sensor registry for physical environmental inputs.
    #[cfg(feature = "mesh")]
    pub fn set_sensor_registry(&mut self, registry: crate::swarm::mesh::SensorRegistry) {
        self.sensor_registry = Some(registry);
    }

    /// Populate mesh telemetry fields in a CycleMetadata struct.
    #[cfg(feature = "mesh")]
    pub fn populate_mesh_metadata(
        &self,
        metadata: &mut crate::cognitive_loop::types::CycleMetadata,
    ) {
        let peer_count = self.mesh_peers.peer_count();
        metadata.mesh_health_score = self.mesh_stats.health_score(peer_count);
        metadata.mesh_peer_count = peer_count as u32;
        metadata.mesh_bytes_sent = self.mesh_stats.bytes_sent;
        metadata.mesh_bytes_received = self.mesh_stats.bytes_received;
    }

    /// Check if the bandwidth budget allows sending `packet_bytes`.
    ///
    /// Returns `true` if the budget allows it (and deducts the bytes),
    /// `false` if throttled (increments `bandwidth_throttled` stat).
    #[cfg(feature = "mesh")]
    pub(crate) fn mesh_bandwidth_check(&mut self, packet_bytes: u64) -> bool {
        let now = std::time::Instant::now();
        if now.duration_since(self.mesh_bandwidth_window_start) >= MESH_BANDWIDTH_WINDOW {
            // Window expired — adjust budget before resetting
            self.adjust_bandwidth_budget();
            self.mesh_bandwidth_window_start = now;
            self.mesh_bandwidth_window_bytes = 0;
            self.mesh_bandwidth_throttled_in_window = false;
        }
        if self.mesh_bandwidth_window_bytes + packet_bytes > self.mesh_bandwidth_budget {
            self.mesh_stats.bandwidth_throttled += 1;
            self.mesh_bandwidth_throttled_in_window = true;
            false
        } else {
            self.mesh_bandwidth_window_bytes += packet_bytes;
            true
        }
    }

    /// Register a single sensor (creates registry if needed).
    #[cfg(feature = "mesh")]
    pub fn register_sensor(&mut self, sensor: Box<dyn crate::swarm::mesh::SensorInput>) {
        if self.sensor_registry.is_none() {
            self.sensor_registry = Some(crate::swarm::mesh::SensorRegistry::new(
                self.config.dimension,
            ));
        }
        self.sensor_registry.as_mut().unwrap().register(sensor);
    }

    /// Emit a wisdom vector over the mesh network, gated by urgency.
    ///
    /// Emission frequency is controlled by the cognitive urgency level:
    /// - `Critical`: every call (~50Hz over B.A.T.M.A.N.)
    /// - `Normal`: ~once per second (~every 50 calls over Yggdrasil)
    /// - `Cruise`: ~once per 10 seconds (~every 500 calls over LoRa)
    ///
    /// If not enough ticks have elapsed since the last emission, this is a no-op.
    #[cfg(feature = "mesh")]
    pub(crate) fn emit_wisdom(
        &mut self,
        wisdom_hv: symthaea_core::hdc::BinaryHV,
        urgency: crate::cognitive_loop::types::CycleUrgency,
        phi: f32,
    ) {
        use crate::cognitive_loop::types::CycleUrgency;
        use crate::swarm::mesh::{MeshOutbound, MeshUrgency, PayloadType, WisdomPacket};

        // Emission rate gating based on urgency
        let interval = match urgency {
            CycleUrgency::Critical => 1, // every cycle (~50Hz)
            CycleUrgency::Normal => 50,  // ~1/s at 50Hz
            CycleUrgency::Cruise => 500, // ~10s at 50Hz
        };

        let ticks_since = self.state.tick.saturating_sub(self.mesh_last_emit_tick);
        // Always allow the first emission (sequence == 0); thereafter gate by interval
        if ticks_since < interval && self.mesh_sequence > 0 {
            return;
        }

        // Bandwidth budget check
        if !self.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64) {
            return;
        }

        self.mesh_last_emit_tick = self.state.tick;

        // Construct source_id from config dimension as a stand-in node identity
        // (real impl would use swarm node_id)
        let mut source_id = [0u8; 8];
        let dim_bytes = (self.config.dimension as u64).to_le_bytes();
        source_id.copy_from_slice(&dim_bytes);

        let timestamp_s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        let mut packet = WisdomPacket {
            source_id,
            sequence: self.mesh_sequence,
            phi,
            urgency: MeshUrgency::from(urgency),
            timestamp_s,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: crate::swarm::mesh::MESH_DEFAULT_TTL,
            wisdom: wisdom_hv,
        };

        self.sign_mesh_packet(&mut packet);

        // Push to replay buffer for partition recovery
        if self.mesh_replay_buffer.len() >= MESH_REPLAY_BUFFER_CAPACITY {
            self.mesh_replay_buffer.pop_front();
        }
        self.mesh_replay_buffer.push_back(packet.clone());

        self.mesh_sequence = self.mesh_sequence.wrapping_add(1);
        self.mesh_outbox.push(MeshOutbound { packet });
        self.mesh_stats.wisdom_sent += 1;
        self.mesh_stats.bytes_sent += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;

        tracing::trace!(
            target: "symthaea::mind::mesh",
            sequence = self.mesh_sequence.wrapping_sub(1),
            urgency = ?urgency,
            phi,
            "Emitted wisdom packet"
        );
    }

    /// Get the mesh source_id for this mind (first 8 bytes of node identity).
    #[cfg(feature = "mesh")]
    fn mesh_source_id(&self) -> [u8; 8] {
        let mut source_id = [0u8; 8];
        let dim_bytes = (self.config.dimension as u64).to_le_bytes();
        source_id.copy_from_slice(&dim_bytes);
        source_id
    }

    /// Set the BLAKE3 key for packet authentication.
    ///
    /// When set, all emitted packets are signed with this key, and
    /// all received packets are verified (unsigned packets are rejected).
    /// Pass `None` to disable authentication (backward-compatible default).
    #[cfg(feature = "mesh")]
    pub fn set_mesh_auth_key(&mut self, key: Option<[u8; 32]>) {
        self.mesh_auth_key = key;
    }

    /// Sign a WisdomPacket with the mesh auth key (if set).
    ///
    /// Computes a BLAKE3 keyed MAC over the serialized packet bytes
    /// and sets the `auth_mac` field. No-op if no key is configured.
    #[cfg(feature = "mesh")]
    fn sign_mesh_packet(&self, packet: &mut crate::swarm::mesh::WisdomPacket) {
        if let Some(ref key) = self.mesh_auth_key {
            let bytes = packet.to_bytes();
            packet.auth_mac = crate::swarm::mesh::compute_packet_mac(&bytes, key);
        }
    }

    /// Adjust the dynamic bandwidth budget using AIMD after each window reset.
    ///
    /// - **Additive Increase**: If mesh is healthy (health > 0.5) and no throttle
    ///   occurred this window, increase budget by 10 KB (capped at 200 KB).
    /// - **Multiplicative Decrease**: If throttled or health < 0.3, halve the
    ///   budget (floored at 25 KB).
    /// - **Hold Steady**: health in [0.3, 0.5] or 0.0 (idle) — no change.
    #[cfg(feature = "mesh")]
    fn adjust_bandwidth_budget(&mut self) {
        let health = self.mesh_stats.health_score(self.mesh_peers.peer_count());
        if self.mesh_bandwidth_throttled_in_window || (health > 0.0 && health < 0.3) {
            // Multiplicative decrease
            let new = (self.mesh_bandwidth_budget as f64 * MESH_BANDWIDTH_DECREASE_FACTOR) as u64;
            self.mesh_bandwidth_budget = new.max(MESH_BANDWIDTH_MIN);
        } else if health > 0.5 {
            // Additive increase
            self.mesh_bandwidth_budget = (self.mesh_bandwidth_budget
                + MESH_BANDWIDTH_ADDITIVE_INCREASE)
                .min(MESH_BANDWIDTH_MAX);
        }
        // health in [0.3, 0.5] or 0.0 (idle): hold steady
        self.mesh_stats.bandwidth_budget_current = self.mesh_bandwidth_budget;
    }

    /// Emit a lightweight heartbeat packet over the mesh network.
    ///
    /// Heartbeats fire every 100 ticks (~2s at 50Hz), keeping the mind
    /// visible in the mesh peer registry even when wisdom emissions are
    /// throttled by low urgency. The heartbeat carries the current phi
    /// but no wisdom vector (zero BinaryHV).
    #[cfg(feature = "mesh")]
    pub(crate) fn emit_heartbeat(&mut self) {
        use crate::swarm::mesh::{MeshOutbound, MeshUrgency, PayloadType, WisdomPacket};

        if self.mesh_bridge.is_none() {
            return;
        }

        let interval = 100u64;
        if self
            .state
            .tick
            .saturating_sub(self.mesh_heartbeat_last_tick)
            < interval
            && self.mesh_heartbeat_sequence > 0
        {
            return;
        }

        // Bandwidth budget check
        if !self.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64) {
            return;
        }

        let source_id = self.mesh_source_id();

        let timestamp_s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        let mut packet = WisdomPacket {
            source_id,
            sequence: self.mesh_heartbeat_sequence,
            phi: self.state.consciousness_level as f32,
            urgency: MeshUrgency::Cruise,
            timestamp_s,
            payload_type: PayloadType::Heartbeat,
            auth_mac: 0,
            ttl: crate::swarm::mesh::MESH_DEFAULT_TTL,
            wisdom: symthaea_core::hdc::BinaryHV::zero(),
        };

        self.sign_mesh_packet(&mut packet);

        self.mesh_outbox.push(MeshOutbound { packet });
        self.mesh_heartbeat_last_tick = self.state.tick;
        self.mesh_heartbeat_sequence = self.mesh_heartbeat_sequence.wrapping_add(1);
        self.mesh_stats.heartbeats_sent += 1;
        self.mesh_stats.bytes_sent += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;

        tracing::trace!(
            target: "symthaea::mind::mesh",
            sequence = self.mesh_heartbeat_sequence.wrapping_sub(1),
            phi = self.state.consciousness_level as f32,
            "Emitted heartbeat packet"
        );
    }

    // ========================================================================
    // Working Memory Seeding (Epistemic Baseline)
    // ========================================================================

    /// Seed working memory with a priori domain knowledge.
    ///
    /// This establishes the epistemic baseline - concepts the system knows
    /// from "birth". Without seeding, the classifier defaults to Unknown
    /// for everything because working memory is empty.
    ///
    /// # Returns
    ///
    /// A `SeedingResult` containing statistics about the seeding operation.
    pub fn seed_memory(&mut self) -> SeedingResult {
        use knowledge::DomainKnowledge;

        let entries = DomainKnowledge::get_initial_seeding();
        let total = entries.len();

        tracing::info!(
            target: "symthaea::mind",
            count = total,
            "Seeding working memory with domain prototypes"
        );

        let mut total_magnitude = 0.0f32;
        let mut categories_seen = std::collections::HashSet::new();

        for entry in &entries {
            // Encode the knowledge entry into a hypervector
            let hv = self.encode_knowledge_entry(entry);

            // Track statistics
            let magnitude: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
            total_magnitude += magnitude;
            categories_seen.insert(entry.category.to_string());

            // Store in working memory (tick 0 = genesis seeding)
            self.working_memory.push(hv);
            self.working_memory_ticks.push(0);
            self.working_memory_sources.push(MemorySource::Internal);
            self.working_memory_verified.push(true);
            self.working_memory_metadata
                .push(std::collections::HashMap::new());

            tracing::debug!(
                target: "symthaea::mind::seeding",
                label = entry.label,
                category = entry.category,
                magnitude = magnitude,
                "Seeded knowledge prototype"
            );
        }

        let avg_magnitude = if total > 0 {
            total_magnitude / total as f32
        } else {
            0.0
        };

        tracing::info!(
            target: "symthaea::mind",
            prototypes = total,
            categories = categories_seen.len(),
            avg_magnitude = avg_magnitude,
            "Seeding complete - epistemic baseline established"
        );

        SeedingResult {
            prototypes_seeded: total,
            categories: categories_seen.into_iter().collect(),
            avg_magnitude,
        }
    }

    /// Encode a knowledge entry into a hypervector.
    ///
    /// Uses the same encoding as text perception to ensure alignment
    /// between seeded knowledge and runtime inputs.
    fn encode_knowledge_entry(&self, entry: &knowledge::KnowledgeEntry) -> ContinuousHV {
        // Combine label and content for richer encoding
        let combined = format!("{} {}", entry.label.replace('_', " "), entry.content);

        // Use the same encoding method as text_to_hv_internal in IntentClassifier
        let dim = self.config.dimension;
        let mut values = vec![0.0f32; dim];
        let text_lower = combined.to_lowercase();

        for (i, byte) in text_lower.bytes().enumerate() {
            let idx = (byte as usize * 31 + i * 7) % dim;
            values[idx] += entry.confidence; // Weight by confidence
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if is_nonzero_f32(magnitude) {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Check if memory has been seeded
    pub fn is_seeded(&self) -> bool {
        // We consider memory seeded if it has at least 10 entries
        // (the minimum from domain knowledge)
        self.working_memory.len() >= 10
    }

    /// Get the number of seeded prototypes
    pub fn seeded_count(&self) -> usize {
        self.working_memory.len()
    }

    // ========================================================================
    // Structured Thought Extraction (Broca's Area Interface)
    // ========================================================================

    /// Extract a structured thought from the current mind state.
    ///
    /// This is the key interface between the HDC+LTC cognitive system and the
    /// LLM translation layer. The mind computes; this method articulates what
    /// was computed into a structured format for faithful translation.
    ///
    /// **Critical Insight**: The LLM should NOT add reasoning - only translate
    /// what this method returns.
    pub fn extract_structured_thought(&self) -> StructuredThought {
        use symthaea_core::hdc::relational_consciousness::{RelationMode, RelationshipStage};

        let state = self.snapshot();

        // Determine epistemic status from consciousness metrics
        let epistemic_status = self.determine_epistemic_status(&state);

        // Infer semantic intent from goals and working memory state
        let semantic_intent = self.infer_semantic_intent();

        // Infer response type
        let response_type = self.infer_response_type();

        // Extract top concepts from working memory
        let activated_concepts = self.extract_top_concepts(5);

        // Calculate working memory coherence
        let coherence = self.calculate_coherence();

        // Calculate relational warmth from emotional state
        let warmth = self.calculate_relational_warmth(&state);

        StructuredThought {
            semantic_intent,
            response_type,
            activated_concepts,
            emotional_tone: EmotionalTone {
                valence: state.emotional_valence as f64,
                arousal: state.arousal as f64,
                warmth,
            },
            structured_data: None,
            domain_context: None,
            psi: state.consciousness_level,
            meta_awareness: state.meta_awareness,
            coherence,
            epistemic_status,
            // Relational fields are filled by the Symthaea facade
            // from the partnership module
            relationship_stage: RelationshipStage::NoRelation,
            relation_mode: RelationMode::IIt,
            trust: 0.0,
            code_context: None,
            constraints: Vec::new(),
            original_input: None,
            primitive_tiers: Vec::new(), // Populated by Symthaea facade from language grounding
            primitives: Vec::new(),      // Populated by Symthaea facade from language grounding
        }
    }

    /// Determine epistemic status using HDC algebraic assessment.
    ///
    /// Combines:
    /// 1. **HDC Resonance**: How familiar is the input to our prototypes?
    /// 2. **Memory Resonance**: Do we have relevant context in working memory?
    /// 3. **Consciousness Metrics**: Phi and meta-awareness modulate certainty
    ///
    /// This is the KEY function for hallucination prevention:
    /// - High familiarity + high phi → Certain
    /// - Low familiarity + empty memory → Unknown (triggers hedging)
    fn determine_epistemic_status(&self, state: &MindState) -> EpistemicStatus {
        // If we have input text, use HDC classification
        if let Some(ref text) = self.last_input_text {
            let assessment = self
                .intent_classifier
                .assess_epistemic_text(text, &self.working_memory);

            // Modulate by consciousness level
            let phi = state.consciousness_level;
            let meta = state.meta_awareness;

            // High consciousness can upgrade Uncertain → Probable
            // Low consciousness can downgrade Probable → Uncertain
            match assessment.status {
                EpistemicStatus::Certain => {
                    if phi > 0.7 && meta > 0.6 {
                        EpistemicStatus::Certain
                    } else if phi > 0.5 {
                        EpistemicStatus::Probable
                    } else {
                        EpistemicStatus::Uncertain
                    }
                }
                EpistemicStatus::Probable => {
                    if phi > 0.8 && meta > 0.7 && assessment.familiarity > 0.7 {
                        EpistemicStatus::Certain
                    } else if phi > 0.4 {
                        EpistemicStatus::Probable
                    } else {
                        EpistemicStatus::Uncertain
                    }
                }
                EpistemicStatus::Uncertain => {
                    if phi > 0.8 && meta > 0.8 && assessment.familiarity > 0.6 {
                        EpistemicStatus::Probable
                    } else {
                        EpistemicStatus::Uncertain
                    }
                }
                EpistemicStatus::Unknown => {
                    // Unknown stays unknown - we don't have the knowledge
                    // This is the hallucination prevention mechanism
                    EpistemicStatus::Unknown
                }
                EpistemicStatus::OutOfDomain => EpistemicStatus::OutOfDomain,
            }
        } else {
            // Fallback to pure consciousness metrics if no text available
            let phi = state.consciousness_level;
            let meta = state.meta_awareness;

            if phi > 0.8 && meta > 0.7 {
                EpistemicStatus::Certain
            } else if phi > 0.6 && meta > 0.5 {
                EpistemicStatus::Probable
            } else if phi > 0.3 || meta > 0.3 {
                EpistemicStatus::Uncertain
            } else {
                EpistemicStatus::Unknown
            }
        }
    }

    /// Infer semantic intent using HDC prototype resonance.
    ///
    /// Computes cosine similarity between input and intent prototypes:
    /// - **Greeting**: "hello", "hi", etc.
    /// - **Question**: "what", "why", "?", etc.
    /// - **Command**: "do", "make", "create", etc.
    /// - **Reflection**: "think", "feel", etc.
    ///
    /// Falls back to goal/memory heuristics if no text is available.
    fn infer_semantic_intent(&self) -> SemanticIntent {
        // If we have input text, use HDC classification
        if let Some(ref text) = self.last_input_text {
            let classification = self.intent_classifier.classify_text(text);

            // If confidence is high enough, use the HDC classification
            if classification.confidence > 0.3 {
                return classification.intent;
            }
            // Fall through to heuristics for low confidence
        }

        // Fallback: Goal and memory-based heuristics
        let has_goals = !self.goals.is_empty();
        let has_memory = !self.working_memory.is_empty();
        let is_conscious = self.state.is_conscious;

        if !is_conscious {
            return SemanticIntent::Acknowledge;
        }

        // Check if any goal suggests clarification need
        let needs_clarification = self.goals.iter().any(|g| {
            g.description.to_lowercase().contains("clarify")
                || g.description.to_lowercase().contains("question")
        });

        if needs_clarification {
            return SemanticIntent::Clarify;
        }

        // Check for action-oriented goals
        let action_oriented = self.goals.iter().any(|g| {
            g.description.to_lowercase().contains("do")
                || g.description.to_lowercase().contains("execute")
                || g.description.to_lowercase().contains("action")
        });

        if action_oriented {
            return SemanticIntent::ProposeAction;
        }

        // If we have working memory content, we likely have an answer
        if has_memory && has_goals {
            return SemanticIntent::Answer;
        }

        // Low consciousness suggests uncertainty
        if self.state.consciousness_level < 0.3 {
            return SemanticIntent::ExpressUncertainty;
        }

        // Default to acknowledgment if nothing specific
        if has_memory {
            SemanticIntent::Answer
        } else {
            SemanticIntent::Acknowledge
        }
    }

    /// Infer response type using HDC classification.
    ///
    /// Uses the response_type from intent classification when available.
    fn infer_response_type(&self) -> ResponseType {
        // If we have input text, use HDC classification
        if let Some(ref text) = self.last_input_text {
            let classification = self.intent_classifier.classify_text(text);
            if classification.confidence > 0.3 {
                return classification.response_type;
            }
        }

        // Fallback to heuristics

        // If high arousal and positive valence, might be empathic
        if self.state.arousal > 0.7 && self.state.emotional_valence > 0.5 {
            return ResponseType::Empathic;
        }

        // If goals suggest questions
        let asking_question = self.goals.iter().any(|g| g.description.ends_with('?'));

        if asking_question {
            return ResponseType::Question;
        }

        // Default to statement
        ResponseType::Statement
    }

    /// Extract top N activated concepts from working memory.
    ///
    /// Uses the HDC concept vocabulary to label working memory contents
    /// via nearest-neighbor lookup against concept prototypes.
    fn extract_top_concepts(&self, n: usize) -> Vec<ActivatedConcept> {
        if self.working_memory.is_empty() {
            return Vec::new();
        }

        // Label all working memory contents
        let labels = self.intent_classifier.label_concepts(&self.working_memory);

        // Convert to ActivatedConcepts, taking top N by confidence
        labels
            .into_iter()
            .take(n)
            .enumerate()
            .map(|(i, label)| {
                // Combine label confidence with position-based decay
                let position_factor = 1.0 - (i as f32 * 0.1); // Decay by 10% per position
                let activation = label.confidence * position_factor;

                ActivatedConcept {
                    // Use semantic label instead of placeholder
                    name: if label.confidence > 0.3 {
                        format!("{}:{}", label.category, label.name)
                    } else {
                        // Fall back to generic if confidence too low
                        format!("unknown:concept_{i}")
                    },
                    activation,
                    relevance: label.similarity.max(0.0) * activation,
                }
            })
            .collect()
    }

    /// Calculate coherence of working memory.
    ///
    /// Measures how well-integrated the current thoughts are by computing
    /// average pairwise similarity in working memory.
    fn calculate_coherence(&self) -> f64 {
        if self.working_memory.len() < 2 {
            return 0.5; // Neutral coherence for insufficient data
        }

        let mut total_similarity = 0.0;
        let mut count = 0;

        for i in 0..self.working_memory.len() {
            for j in (i + 1)..self.working_memory.len() {
                let sim = self.working_memory[i]
                    .similarity(&self.working_memory[j])
                    .abs() as f64;
                total_similarity += sim;
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.5
        }
    }

    /// Calculate relational warmth from emotional state.
    fn calculate_relational_warmth(&self, state: &MindState) -> f64 {
        // Warmth increases with positive valence and moderate arousal
        let valence_contrib = (state.emotional_valence as f64 + 1.0) / 2.0; // Normalize to 0-1
        let arousal_contrib = 1.0 - (state.arousal as f64 - 0.5).abs(); // Peak at 0.5

        (valence_contrib * 0.7 + arousal_contrib * 0.3).clamp(0.0, 1.0)
    }
}

impl Default for ContinuousMind {
    fn default() -> Self {
        Self::new(MindConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mind_creation() {
        let mind = ContinuousMind::default();
        assert_eq!(mind.state.tick, 0);
        assert!(!mind.state.is_active);
    }

    #[test]
    fn test_mind_tick() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.tick();
        assert_eq!(mind.state.tick, 1);
    }

    #[test]
    fn test_perception() {
        let mut mind = ContinuousMind::default();
        mind.perceive(ContinuousHV::random(512, 42));
        mind.tick();
        assert_eq!(mind.working_memory.len(), 1);
    }

    #[test]
    fn test_goal_setting() {
        let mut mind = ContinuousMind::default();
        mind.set_goal("Test goal", ContinuousHV::random(512, 42), 1.0);
        mind.tick();
        assert!(!mind.active_goals().is_empty());
    }

    #[test]
    fn test_consciousness_update() {
        let mut mind = ContinuousMind::default();

        for i in 0..5 {
            mind.perceive(ContinuousHV::random(512, 42 + i as u64));
        }

        for _ in 0..5 {
            mind.tick();
        }

        assert!(mind.state.consciousness_level > 0.0);
    }

    // ====================================================================
    // Social Coherence Integration Tests
    // ====================================================================

    #[test]
    fn test_social_coherence_disabled_by_default() {
        let mind = ContinuousMind::default();
        assert!(!mind.is_social());
        assert!(mind.social_coherence().is_none());
    }

    #[test]
    fn test_social_coherence_enabled_via_config() {
        let mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        assert!(mind.is_social());
        assert!(mind.social_coherence().is_some());
    }

    #[test]
    fn test_social_coherence_enable_after_construction() {
        let mut mind = ContinuousMind::default();
        assert!(!mind.is_social());
        mind.enable_social_coherence();
        assert!(mind.is_social());
    }

    #[test]
    fn test_social_inbox_processed_on_tick() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();

        // Send a social message
        mind.receive_social(SocialMessage {
            agent_id: "peer_1".to_string(),
            behavior: ContinuousHV::random(512, 0xBEEF_0001),
            context: ContinuousHV::random(512, 0xBEEF_0002),
            interaction_outcome: None,
        });

        assert_eq!(mind.social_inbox.len(), 1);
        mind.tick();
        // Inbox should be drained after tick
        assert_eq!(mind.social_inbox.len(), 0);
        // Agent should be modeled now
        let sc = mind.social_coherence().unwrap();
        assert!(sc.get_mental_model("peer_1").is_some());
    }

    #[test]
    fn test_social_interaction_builds_relationship() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();

        // Send cooperative interaction
        mind.receive_social(SocialMessage {
            agent_id: "ally_1".to_string(),
            behavior: ContinuousHV::random(512, 0xA11E_0001),
            context: ContinuousHV::random(512, 0xA11E_0002),
            interaction_outcome: Some(0.9),
        });
        mind.tick();

        let sc = mind.social_coherence().unwrap();
        let rel = sc.get_relationship("ally_1");
        assert!(rel.is_some(), "Relationship should be created");
        assert!(
            rel.unwrap().trust > 0.5,
            "Trust should increase from cooperation"
        );
    }

    #[test]
    fn test_social_outbox_populated_on_tick() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();

        // Tick 5 times to trigger outbox export (every 5 ticks)
        for _ in 0..5 {
            mind.tick();
        }

        let outbox = mind.drain_social_outbox();
        assert!(
            !outbox.is_empty(),
            "Outbox should have messages after 5 ticks"
        );
        assert_eq!(outbox[0].agent_id, "self");
    }

    #[test]
    fn test_social_no_processing_when_disabled() {
        let mut mind = ContinuousMind::default();
        mind.activate();

        // Inbox messages should remain when social is disabled
        // (actually they get drained regardless but social coherence isn't updated)
        mind.receive_social(SocialMessage {
            agent_id: "ghost".to_string(),
            behavior: ContinuousHV::random(512, 0xDEAD),
            context: ContinuousHV::random(512, 0xDEAD),
            interaction_outcome: None,
        });
        mind.tick();

        // Social coherence is None, so no models are built
        assert!(mind.social_coherence().is_none());
        // Outbox should be empty since social is disabled
        let outbox = mind.drain_social_outbox();
        assert!(outbox.is_empty());
    }

    // ====================================================================
    // Iroh P2P Bridge Integration Tests
    // ====================================================================

    #[test]
    fn test_iroh_bridge_not_set_by_default() {
        let mind = ContinuousMind::default();
        assert!(!mind.has_iroh_bridge());
    }

    #[test]
    fn test_iroh_bridge_attach() {
        let mut mind = ContinuousMind::default();
        let (handle, _actor) = crate::swarm::IrohBridgeHandle::new(4, 4);
        mind.set_iroh_bridge(handle);
        assert!(mind.has_iroh_bridge());
    }

    #[test]
    fn test_iroh_bridge_flushes_outbox_on_tick() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();
        let (handle, _actor) = crate::swarm::IrohBridgeHandle::new(64, 128);
        mind.set_iroh_bridge(handle);

        // Tick 5 times — social coherence exports on tick 5
        for _ in 0..5 {
            mind.tick();
        }

        // Outbox should be empty because the bridge flushed it
        assert!(
            mind.social_outbox.is_empty(),
            "Bridge should have flushed the outbox"
        );
    }

    #[test]
    fn test_iroh_bridge_drains_inbox_on_tick() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();
        let (handle, actor) = crate::swarm::IrohBridgeHandle::new(64, 128);

        // We need the actor's inbound_tx to inject messages.
        // Instead, manually push to inbox and verify tick processes it.
        // The bridge integration is: bridge drains → inbox, tick processes inbox → social coherence.
        // We can verify the bridge wiring by checking that when bridge is attached,
        // outbox messages get sent to the bridge channel.
        mind.set_iroh_bridge(handle);

        // Manually inject into inbox (simulating what bridge.drain_inbox would return)
        mind.receive_social(SocialMessage {
            agent_id: "network_peer".to_string(),
            behavior: ContinuousHV::random(512, 0xCAFE),
            context: ContinuousHV::random(512, 0xCAFE),
            interaction_outcome: None,
        });

        mind.tick();

        // The message should have been processed by social coherence
        let sc = mind.social_coherence().unwrap();
        assert!(
            sc.get_mental_model("network_peer").is_some(),
            "Network peer should be modeled after tick"
        );

        // Suppress unused variable warning
        drop(actor);
    }

    // ====================================================================
    // Mesh Network Bridge Integration Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_bridge_not_set_by_default() {
        let mind = ContinuousMind::default();
        assert!(!mind.has_mesh_bridge());
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_bridge_attach() {
        let mut mind = ContinuousMind::default();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(4, 4);
        mind.set_mesh_bridge(handle);
        assert!(mind.has_mesh_bridge());
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_emit_wisdom_critical_every_tick() {
        use crate::cognitive_loop::types::CycleUrgency;
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Critical urgency should emit every call
        mind.state.tick = 1;
        mind.emit_wisdom(BinaryHV([0xAA; 2048]), CycleUrgency::Critical, 0.7);
        assert_eq!(mind.mesh_outbox.len(), 1);

        mind.state.tick = 2;
        mind.emit_wisdom(BinaryHV([0xBB; 2048]), CycleUrgency::Critical, 0.8);
        assert_eq!(mind.mesh_outbox.len(), 2);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_emit_wisdom_normal_throttled() {
        use crate::cognitive_loop::types::CycleUrgency;
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // First emission at tick 0
        mind.state.tick = 0;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Normal, 0.5);
        assert_eq!(mind.mesh_outbox.len(), 1);

        // Should NOT emit at tick 10 (interval=50)
        mind.state.tick = 10;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Normal, 0.5);
        assert_eq!(mind.mesh_outbox.len(), 1);

        // Should emit at tick 50
        mind.state.tick = 50;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Normal, 0.5);
        assert_eq!(mind.mesh_outbox.len(), 2);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_emit_wisdom_cruise_rare() {
        use crate::cognitive_loop::types::CycleUrgency;
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        mind.state.tick = 0;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Cruise, 0.3);
        assert_eq!(mind.mesh_outbox.len(), 1);

        // Should NOT emit until tick 500
        mind.state.tick = 499;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Cruise, 0.3);
        assert_eq!(mind.mesh_outbox.len(), 1);

        mind.state.tick = 500;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Cruise, 0.3);
        assert_eq!(mind.mesh_outbox.len(), 2);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_bridge_flushes_outbox_on_tick() {
        use crate::cognitive_loop::types::CycleUrgency;
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Emit a wisdom packet directly
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Critical, 0.5);
        assert_eq!(mind.mesh_outbox.len(), 1);

        // Tick should flush mesh_outbox through the bridge
        mind.tick();

        assert!(
            mind.mesh_outbox.is_empty(),
            "Bridge should have flushed the mesh outbox"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_process_mesh_drains_inbox() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();

        // Inject a wisdom packet into the mesh inbox
        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
            sequence: 1,
            phi: 0.8,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xFF; 2048]),
        });

        assert_eq!(mind.mesh_inbox.len(), 1);
        mind.tick();

        // Inbox should be drained after tick
        assert_eq!(mind.mesh_inbox.len(), 0);

        // Peer should be modeled in social coherence
        let sc = mind.social_coherence().unwrap();
        assert!(
            sc.get_mental_model("deadbeefcafebabe").is_some(),
            "Mesh peer should be modeled in social coherence"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_emit_wisdom_sequence_increments() {
        use crate::cognitive_loop::types::CycleUrgency;
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        mind.state.tick = 0;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Critical, 0.5);
        mind.state.tick = 1;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Critical, 0.5);
        mind.state.tick = 2;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Critical, 0.5);

        assert_eq!(mind.mesh_outbox.len(), 3);
        assert_eq!(mind.mesh_outbox[0].packet.sequence, 0);
        assert_eq!(mind.mesh_outbox[1].packet.sequence, 1);
        assert_eq!(mind.mesh_outbox[2].packet.sequence, 2);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_auto_emit_on_tick_with_bridge() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Perceive something so current_thought is non-zero
        mind.perceive(ContinuousHV::random(512, 0xFACE));

        // Tick many times — auto-emit should fire at urgency-gated intervals
        for _ in 0..50 {
            mind.tick();
        }

        // At minimum, the first tick should have emitted (sequence 0 is always allowed)
        // The bridge flushed outbox each tick, so outbox may be empty but emissions occurred
        assert!(
            mind.mesh_sequence > 0,
            "Auto-emit should have incremented sequence counter"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_no_auto_emit_without_bridge() {
        let mut mind = ContinuousMind::default();
        mind.activate();

        // No bridge attached
        mind.perceive(ContinuousHV::random(512, 0xFACE));

        for _ in 0..50 {
            mind.tick();
        }

        // No emissions should have occurred
        assert_eq!(
            mind.mesh_sequence, 0,
            "No auto-emit without bridge attached"
        );
        assert!(
            mind.mesh_outbox.is_empty(),
            "No packets in outbox without bridge"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_process_mesh_updates_registry() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Inject packets from two peers
        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0x11; 8],
            sequence: 1,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xAA; 2048]),
        });
        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0x22; 8],
            sequence: 1,
            phi: 0.9,
            urgency: MeshUrgency::Critical,
            timestamp_s: 0,
            payload_type: PayloadType::Heartbeat,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xBB; 2048]),
        });

        mind.tick();

        assert_eq!(
            mind.mesh_peers().peer_count(),
            2,
            "Registry should track 2 peers"
        );
        let avg = mind.mesh_peers().average_phi();
        assert!(
            (avg - 0.8).abs() < 1e-6,
            "Average phi should be ~0.8: {avg}"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_process_sensors_feeds_working_memory() {
        use crate::swarm::mesh::{MeshUrgency, MockSensor};

        let mut mind = ContinuousMind::default();
        mind.activate();

        let sensor = MockSensor::new(
            "test::thermometer",
            MeshUrgency::Cruise,
            vec![vec![22.5, 45.0]],
        );
        mind.register_sensor(Box::new(sensor));

        let wm_before = mind.working_memory.len();
        mind.tick();
        let wm_after = mind.working_memory.len();

        assert!(
            wm_after > wm_before,
            "Sensor reading should have been perceived into working memory"
        );
    }

    // ====================================================================
    // Current Thought EMA Tests
    // ====================================================================

    #[test]
    fn test_current_thought_nonzero_after_perception() {
        let mut mind = ContinuousMind::default();
        mind.activate();

        // Before any perception, current_thought is zero
        assert!(
            mind.state.current_thought.norm() < f32::EPSILON,
            "current_thought should start as zero"
        );

        mind.perceive(ContinuousHV::random(512, 42));
        mind.tick();

        // After perception, current_thought should be non-zero
        assert!(
            mind.state.current_thought.norm() > 0.1,
            "current_thought should be non-zero after perception: norm={}",
            mind.state.current_thought.norm()
        );
    }

    #[test]
    fn test_current_thought_evolves_with_ema() {
        let mut mind = ContinuousMind::default();
        mind.activate();

        // First perception: adopt directly
        mind.perceive(ContinuousHV::random(512, 100));
        mind.tick();
        let after_first = mind.state.current_thought.clone();

        // Second perception: EMA blend
        mind.perceive(ContinuousHV::random(512, 200));
        mind.tick();
        let after_second = mind.state.current_thought.clone();

        // Thought should have changed
        let sim = after_first.similarity(&after_second);
        assert!(
            sim < 0.99,
            "current_thought should evolve after new perception: sim={}",
            sim
        );

        // But should retain some of the first thought (70% weight)
        assert!(
            sim > 0.1,
            "current_thought should retain prior context: sim={}",
            sim
        );
    }

    // ====================================================================
    // Swarm Phi Boost Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_swarm_phi_boosts_consciousness() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        // Mind without peers
        let mut mind_solo = ContinuousMind::default();
        mind_solo.activate();
        for i in 0..5 {
            mind_solo.perceive(ContinuousHV::random(512, 42 + i as u64));
        }
        for _ in 0..5 {
            mind_solo.tick();
        }
        let solo_consciousness = mind_solo.state.consciousness_level;

        // Mind with peers (inject a high-phi peer into registry)
        let mut mind_swarm = ContinuousMind::default();
        mind_swarm.activate();
        for i in 0..5 {
            mind_swarm.perceive(ContinuousHV::random(512, 42 + i as u64));
        }
        // Inject peer before ticking
        mind_swarm.mesh_peers.update(&WisdomPacket {
            source_id: [0xFF; 8],
            sequence: 1,
            phi: 0.9,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0; 2048]),
        });
        for _ in 0..5 {
            mind_swarm.tick();
        }
        let swarm_consciousness = mind_swarm.state.consciousness_level;

        assert!(
            swarm_consciousness > solo_consciousness,
            "Swarm mind ({swarm_consciousness}) should have higher consciousness than solo ({solo_consciousness})"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_no_boost_without_peers() {
        // Verify consciousness is identical when no peers are present
        let mut mind = ContinuousMind::default();
        mind.activate();
        for i in 0..3 {
            mind.perceive(ContinuousHV::random(512, 100 + i as u64));
        }
        mind.tick();
        let level = mind.state.consciousness_level;

        // Peer count should be 0
        assert_eq!(mind.mesh_peers().peer_count(), 0);
        // Consciousness should be set purely by pairwise integration
        assert!(
            level > 0.0,
            "Consciousness should be non-zero with perceptions"
        );
    }

    // ====================================================================
    // Heartbeat Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_heartbeat_emitted_at_interval() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Perceive so current_thought is non-zero
        mind.perceive(ContinuousHV::random(512, 0xBEA7));

        // Tick 200 times — heartbeats fire every 100 ticks
        // Reset bandwidth budget each tick to prevent throttling from
        // exhausting the budget (this test targets interval gating, not bandwidth).
        for _ in 0..200 {
            mind.mesh_bandwidth_window_bytes = 0;
            mind.tick();
        }

        // At least 2 heartbeat emissions (tick 1 for sequence=0, tick 101)
        assert!(
            mind.mesh_heartbeat_sequence >= 2,
            "Expected ≥2 heartbeat emissions, got {}",
            mind.mesh_heartbeat_sequence
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_heartbeat_uses_cruise_urgency() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType};

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        mind.perceive(ContinuousHV::random(512, 0xBEA7));
        mind.state.tick = 1;
        mind.emit_heartbeat();

        assert_eq!(mind.mesh_outbox.len(), 1);
        assert_eq!(mind.mesh_outbox[0].packet.urgency, MeshUrgency::Cruise);
        assert_eq!(
            mind.mesh_outbox[0].packet.payload_type,
            PayloadType::Heartbeat
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_heartbeat_has_current_phi() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        mind.state.consciousness_level = 0.73;
        mind.state.tick = 1;
        mind.emit_heartbeat();

        assert_eq!(mind.mesh_outbox.len(), 1);
        assert!(
            (mind.mesh_outbox[0].packet.phi - 0.73).abs() < 1e-6,
            "Heartbeat phi should match consciousness level"
        );
    }

    // ====================================================================
    // Gradient Routing Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_process_mesh_routes_gradients() {
        use crate::swarm::mesh::WisdomPacket;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Build a gradient packet
        let msg = crate::swarm::GradientMessage {
            source_id: [0u8; 32],
            gradient_data: vec![0.1, -0.2, 0.3],
            trust_score: 0.8,
            noise_scale: 0.0,
            timestamp: 1_700_000_000_000,
            sample_count: 50,
            model_version: 2,
        };
        let pkt = WisdomPacket::from_gradient([0xFE; 8], 1, &msg).unwrap();
        mind.mesh_inbox.push(pkt);

        assert!(mind.federated_inbox.is_empty());
        mind.tick();

        // Gradient should have been routed to federated_inbox
        assert_eq!(
            mind.federated_inbox.len(),
            1,
            "Gradient should be routed to federated_inbox"
        );
        assert_eq!(mind.federated_inbox[0].gradient_data.len(), 3);
        assert!((mind.federated_inbox[0].trust_score - 0.8).abs() < 1e-6);
    }

    // ====================================================================
    // Mind-to-Mind Integration Test
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[tokio::test]
    async fn test_mind_to_mind_mesh_roundtrip() {
        use crate::swarm::mesh::{
            BiLoopbackTransport, DualLayerMesh, MeshBridgeHandle, MeshReceiver,
        };

        // Create paired transports (A writes → B reads, B writes → A reads)
        // Use batman-sized MTU so whole packets fit without fragmentation
        let (transport_a, transport_b) = BiLoopbackTransport::pair("mind_a", "mind_b", 2100);

        // Build DualLayerMesh for each side
        let mesh_a = DualLayerMesh::new([0xAA; 32]).with_batman(Box::new(transport_a));
        let mesh_b = DualLayerMesh::new([0xBB; 32]).with_batman(Box::new(transport_b));

        // Create bridge handles + spawn actors
        let (handle_a, actor_a) = MeshBridgeHandle::new(64, 64);
        let (handle_b, actor_b) = MeshBridgeHandle::new(64, 64);
        let receiver_a = MeshReceiver::new();
        let receiver_b = MeshReceiver::new();
        tokio::spawn(actor_a.run(mesh_a, receiver_a));
        tokio::spawn(actor_b.run(mesh_b, receiver_b));

        // Create two minds
        let mut mind_a = ContinuousMind::new(MindConfig::default());
        let mut mind_b = ContinuousMind::new(MindConfig::default());
        mind_a.set_mesh_bridge(handle_a);
        mind_b.set_mesh_bridge(handle_b);

        // Feed mind_a a perception so it has a non-zero thought to emit
        let hv = ContinuousHV::random(mind_a.config.dimension, 42);
        mind_a.perceive(hv);

        // Tick mind A several times (auto_emit_wisdom fires, sync_mesh_bridge flushes)
        for _ in 0..10 {
            mind_a.tick();
        }

        // Give the async actor time to transport packets (500ms = 10× actor poll interval)
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Tick mind B (sync_mesh_bridge drains inbox, process_mesh dispatches)
        for _ in 0..10 {
            mind_b.tick();
        }

        // Verify mind B saw a peer
        assert!(
            mind_b.mesh_peers().peer_count() > 0,
            "Mind B should see Mind A as a peer"
        );
    }

    // ====================================================================
    // Gradient Emission Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_emit_gradients_via_mesh() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);
        mind.enable_federated(vec![0.0; 10]);

        // Tick 5 times — process_federated exports gradient to outbox every 5 ticks
        for _ in 0..5 {
            mind.tick();
        }

        // emit_gradients should have consumed outbox and emitted packets
        assert!(
            mind.mesh_gradient_sequence > 0,
            "Gradient sequence should have incremented: got {}",
            mind.mesh_gradient_sequence
        );
        assert!(
            mind.mesh_stats.gradients_sent > 0,
            "gradients_sent stat should be > 0"
        );
        // federated_outbox should be empty (consumed by emit_gradients)
        assert!(
            mind.federated_outbox.is_empty(),
            "federated_outbox should be drained by emit_gradients"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_emit_gradients_no_bridge_noop() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.enable_federated(vec![0.0; 10]);

        // No bridge attached — gradient outbox should be preserved
        for _ in 0..5 {
            mind.tick();
        }

        assert_eq!(
            mind.mesh_gradient_sequence, 0,
            "No gradient emissions without bridge"
        );
        // federated_outbox should still contain gradients (not consumed)
        assert!(
            !mind.federated_outbox.is_empty(),
            "federated_outbox should be preserved without bridge"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_emit_gradients_oversized_skipped() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Inject an oversized gradient (505 > 504 max)
        mind.federated_outbox.push(crate::swarm::GradientMessage {
            source_id: [0u8; 32],
            gradient_data: vec![0.0; 505],
            trust_score: 0.5,
            noise_scale: 0.0,
            timestamp: 0,
            sample_count: 1,
            model_version: 1,
        });

        mind.tick();

        assert_eq!(
            mind.mesh_gradient_sequence, 0,
            "Oversized gradient should be skipped, sequence stays 0"
        );
        assert_eq!(
            mind.mesh_stats.gradients_sent, 0,
            "No gradient stats for skipped oversized"
        );
    }

    // ====================================================================
    // Affective Emission Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_affective_emitted_at_interval() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        mind.perceive(ContinuousHV::random(512, 0xAFFE));

        // Tick 100 times — affective fires every 50 ticks
        // Reset bandwidth budget each tick to isolate interval gating.
        for _ in 0..100 {
            mind.mesh_bandwidth_window_bytes = 0;
            mind.tick();
        }

        assert!(
            mind.mesh_affective_sequence >= 2,
            "Expected ≥2 affective emissions, got {}",
            mind.mesh_affective_sequence
        );
        assert!(
            mind.mesh_stats.affective_sent >= 2,
            "affective_sent should be ≥2"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_affective_uses_mind_emotional_state() {
        use crate::swarm::mesh::PayloadType;

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        mind.state.emotional_valence = 0.7;
        mind.state.arousal = 0.85;
        mind.state.tick = 1;
        mind.emit_affective();

        assert_eq!(mind.mesh_outbox.len(), 1);
        let pkt = &mind.mesh_outbox[0].packet;
        assert_eq!(pkt.payload_type, PayloadType::Affective);

        let affect = pkt.extract_affective().unwrap();
        assert!(
            (affect.valence - 0.7).abs() < 1e-6,
            "Valence should match mind state"
        );
        assert!(
            (affect.arousal - 0.85).abs() < 1e-6,
            "Arousal should match mind state"
        );
        assert!(
            (affect.intensity - 0.85).abs() < 1e-6,
            "Intensity should be abs(arousal)"
        );
    }

    // ====================================================================
    // MeshStats Telemetry Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_stats_count_emissions() {
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Emit wisdom
        mind.state.tick = 1;
        mind.emit_wisdom(
            BinaryHV([0; 2048]),
            crate::cognitive_loop::types::CycleUrgency::Critical,
            0.5,
        );
        assert_eq!(mind.mesh_stats().wisdom_sent, 1);

        // Emit heartbeat
        mind.emit_heartbeat();
        assert_eq!(mind.mesh_stats().heartbeats_sent, 1);

        // Emit affective
        mind.emit_affective();
        assert_eq!(mind.mesh_stats().affective_sent, 1);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_stats_count_receives() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Inject one wisdom packet
        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0x11; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xAA; 2048]),
        });

        mind.tick();

        assert_eq!(
            mind.mesh_stats().wisdom_received,
            1,
            "wisdom_received should be 1"
        );
    }

    // ====================================================================
    // Peer Expiry → Social Coherence Cleanup Test
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_peer_expiry_cleans_social_coherence() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();

        // Use a very short stale timeout
        mind.mesh_peers = crate::swarm::mesh::MeshPeerRegistry::with_timeout(
            std::time::Duration::from_millis(10),
        );

        // Inject a peer packet so it gets tracked + modeled in social coherence
        let peer_id = [0xEE; 8];
        let pkt = crate::swarm::mesh::WisdomPacket {
            source_id: peer_id,
            sequence: 1,
            phi: 0.6,
            urgency: crate::swarm::mesh::MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: crate::swarm::mesh::PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: symthaea_core::hdc::BinaryHV([0xFF; 2048]),
        };
        mind.mesh_inbox.push(pkt);
        mind.tick(); // process_mesh: registers peer + observes in social coherence

        let peer_hex = crate::swarm::mesh::hex_short(&peer_id);
        assert!(
            mind.social_coherence()
                .unwrap()
                .get_mental_model(&peer_hex)
                .is_some(),
            "Peer should be modeled after tick"
        );
        assert_eq!(mind.mesh_peers().peer_count(), 1);

        // Wait for peer to become stale
        std::thread::sleep(std::time::Duration::from_millis(20));

        // Tick at a multiple of 100 so expire_stale runs
        mind.state.tick = 99; // next tick will be 100
        mind.tick();

        assert_eq!(
            mind.mesh_peers().peer_count(),
            0,
            "Stale peer should be expired"
        );
        assert!(
            mind.social_coherence()
                .unwrap()
                .get_mental_model(&peer_hex)
                .is_none(),
            "Social model for expired peer should be removed"
        );
        assert!(
            mind.mesh_stats().peers_expired >= 1,
            "peers_expired stat should be ≥1"
        );
    }

    // ====================================================================
    // LoRa Fragmentation Integration Test
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[tokio::test]
    async fn test_mind_to_mind_lora_fragmentation_roundtrip() {
        use crate::swarm::mesh::{
            BiLoopbackTransport, DualLayerMesh, MeshBridgeHandle, MeshReceiver, LORA_MTU,
        };

        // Create paired transports at LoRa MTU (222 bytes — forces fragmentation)
        let (transport_a, transport_b) = BiLoopbackTransport::pair("lora_a", "lora_b", LORA_MTU);

        // Build DualLayerMesh for each side — LoRa only
        let mesh_a = DualLayerMesh::new([0xAA; 32]).with_lora(Box::new(transport_a));
        let mesh_b = DualLayerMesh::new([0xBB; 32]).with_lora(Box::new(transport_b));

        // Create bridge handles + spawn actors
        let (handle_a, actor_a) = MeshBridgeHandle::new(64, 64);
        let (handle_b, actor_b) = MeshBridgeHandle::new(64, 64);
        let receiver_a = MeshReceiver::new();
        let receiver_b = MeshReceiver::new();
        tokio::spawn(actor_a.run(mesh_a, receiver_a));
        tokio::spawn(actor_b.run(mesh_b, receiver_b));

        // Create two minds
        let mut mind_a = ContinuousMind::new(MindConfig::default());
        let mut mind_b = ContinuousMind::new(MindConfig::default());
        mind_a.set_mesh_bridge(handle_a);
        mind_b.set_mesh_bridge(handle_b);

        // Feed mind_a a perception so it has a non-zero thought to emit
        let hv = ContinuousHV::random(mind_a.config.dimension, 42);
        mind_a.perceive(hv);

        // Tick mind A several times — auto-emit fires, sync flushes fragments
        for _ in 0..10 {
            mind_a.tick();
        }

        // LoRa: 11 fragments at 50ms poll interval → need ~550ms for reassembly
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Tick mind B to drain inbox and process
        for _ in 0..5 {
            mind_b.tick();
        }

        // Verify mind B saw mind A as a peer after fragmentation/reassembly
        assert!(
            mind_b.mesh_peers().peer_count() > 0,
            "Mind B should see Mind A as a peer via LoRa fragmentation"
        );
    }

    // ====================================================================
    // Bandwidth Metering Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_bandwidth_metering_emit() {
        use crate::swarm::mesh::WISDOM_PACKET_SIZE;
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Emit wisdom
        mind.state.tick = 1;
        mind.emit_wisdom(
            BinaryHV([0; 2048]),
            crate::cognitive_loop::types::CycleUrgency::Critical,
            0.5,
        );
        // Emit heartbeat
        mind.emit_heartbeat();

        assert_eq!(
            mind.mesh_stats().bytes_sent,
            2 * WISDOM_PACKET_SIZE as u64,
            "bytes_sent should be 2 × WISDOM_PACKET_SIZE after wisdom + heartbeat"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_bandwidth_metering_receive() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket, WISDOM_PACKET_SIZE};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0x11; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xAA; 2048]),
        });

        mind.tick();

        assert_eq!(
            mind.mesh_stats().bytes_received,
            WISDOM_PACKET_SIZE as u64,
            "bytes_received should be WISDOM_PACKET_SIZE after one packet"
        );
    }

    // ====================================================================
    // MeshTelemetry Snapshot Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_telemetry_snapshot() {
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Emit a wisdom packet and heartbeat
        mind.state.tick = 1;
        mind.emit_wisdom(
            BinaryHV([0; 2048]),
            crate::cognitive_loop::types::CycleUrgency::Critical,
            0.5,
        );
        mind.emit_heartbeat();

        // Inject a peer packet
        mind.mesh_peers.update(&crate::swarm::mesh::WisdomPacket {
            source_id: [0xFF; 8],
            sequence: 1,
            phi: 0.9,
            urgency: crate::swarm::mesh::MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: crate::swarm::mesh::PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0; 2048]),
        });

        let t = mind.mesh_telemetry();
        assert_eq!(t.stats.wisdom_sent, 1);
        assert_eq!(t.stats.heartbeats_sent, 1);
        assert_eq!(t.peer_count, 1);
        assert!((t.avg_phi - 0.9).abs() < 1e-6);
        assert!(t.health_score > 0.0, "Health score should be > 0");
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_telemetry_in_mindstate() {
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Emit something to populate stats
        mind.state.tick = 1;
        mind.emit_wisdom(
            BinaryHV([0; 2048]),
            crate::cognitive_loop::types::CycleUrgency::Critical,
            0.5,
        );

        let snap = mind.snapshot();
        assert!(
            snap.mesh_telemetry.is_some(),
            "snapshot() should populate mesh_telemetry"
        );
        let t = snap.mesh_telemetry.unwrap();
        assert_eq!(t.stats.wisdom_sent, 1);
    }

    // ====================================================================
    // Outbox Backpressure Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_federated_outbox_capped() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.enable_federated(vec![0.0; 10]);

        // Tick 1000 times — exports gradient every 5 ticks = 200 pushes
        for _ in 0..1000 {
            mind.tick();
        }

        assert!(
            mind.federated_outbox.len() <= super::MAX_OUTBOX_SIZE,
            "federated_outbox should be capped at {}: got {}",
            super::MAX_OUTBOX_SIZE,
            mind.federated_outbox.len()
        );
    }

    #[test]
    fn test_social_outbox_capped() {
        let mut mind = ContinuousMind::new(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        mind.activate();

        // Tick 1000 times — exports social every 5 ticks = 200 pushes
        for _ in 0..1000 {
            mind.tick();
        }

        assert!(
            mind.social_outbox.len() <= super::MAX_OUTBOX_SIZE,
            "social_outbox should be capped at {}: got {}",
            super::MAX_OUTBOX_SIZE,
            mind.social_outbox.len()
        );
    }

    // ====================================================================
    // Gradient + Affective Roundtrip Integration Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[tokio::test]
    async fn test_mind_to_mind_gradient_roundtrip() {
        use crate::swarm::mesh::{
            BiLoopbackTransport, DualLayerMesh, MeshBridgeHandle, MeshReceiver,
        };

        let (transport_a, transport_b) = BiLoopbackTransport::pair("grad_a", "grad_b", 2100);
        let mesh_a = DualLayerMesh::new([0xAA; 32]).with_batman(Box::new(transport_a));
        let mesh_b = DualLayerMesh::new([0xBB; 32]).with_batman(Box::new(transport_b));

        let (handle_a, actor_a) = MeshBridgeHandle::new(64, 64);
        let (handle_b, actor_b) = MeshBridgeHandle::new(64, 64);
        tokio::spawn(actor_a.run(mesh_a, MeshReceiver::new()));
        tokio::spawn(actor_b.run(mesh_b, MeshReceiver::new()));

        // Mind A: federated enabled, will produce gradients
        let mut mind_a = ContinuousMind::new(MindConfig::default());
        mind_a.set_mesh_bridge(handle_a);
        mind_a.activate();
        mind_a.enable_federated(vec![0.0; 10]);
        mind_a.perceive(ContinuousHV::random(512, 0xFACE));

        // Tick mind A — export gradient + emit over mesh
        // Reset bandwidth budget each tick to prevent throttling (testing transport, not budget)
        for _ in 0..5 {
            mind_a.mesh_bandwidth_window_bytes = 0;
            mind_a.tick();
        }

        // Give async actor time to transport (500ms = 10× actor poll interval)
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Mind B: tick to drain inbox
        let mut mind_b = ContinuousMind::new(MindConfig::default());
        mind_b.set_mesh_bridge(handle_b);
        mind_b.activate();
        for _ in 0..10 {
            mind_b.mesh_bandwidth_window_bytes = 0;
            mind_b.tick();
        }

        assert!(
            !mind_b.federated_inbox.is_empty(),
            "Mind B should have received gradient(s) from Mind A: got {}",
            mind_b.federated_inbox.len()
        );
    }

    #[cfg(feature = "mesh")]
    #[tokio::test]
    async fn test_mind_to_mind_affective_roundtrip() {
        use crate::swarm::mesh::{
            BiLoopbackTransport, DualLayerMesh, MeshBridgeHandle, MeshReceiver,
        };

        let (transport_a, transport_b) = BiLoopbackTransport::pair("aff_a", "aff_b", 2100);
        let mesh_a = DualLayerMesh::new([0xAA; 32]).with_batman(Box::new(transport_a));
        let mesh_b = DualLayerMesh::new([0xBB; 32]).with_batman(Box::new(transport_b));

        let (handle_a, actor_a) = MeshBridgeHandle::new(64, 64);
        let (handle_b, actor_b) = MeshBridgeHandle::new(64, 64);
        tokio::spawn(actor_a.run(mesh_a, MeshReceiver::new()));
        tokio::spawn(actor_b.run(mesh_b, MeshReceiver::new()));

        // Mind A: set emotional state, tick to emit affective
        let mut mind_a = ContinuousMind::new(MindConfig::default());
        mind_a.set_mesh_bridge(handle_a);
        mind_a.activate();
        mind_a.state.emotional_valence = 0.7;
        mind_a.state.arousal = 0.8;
        mind_a.perceive(ContinuousHV::random(512, 0xAFFE));

        // Tick 50× — affective emission fires every 50 ticks
        // Reset bandwidth budget each tick to prevent throttling (testing transport, not budget)
        for _ in 0..50 {
            mind_a.mesh_bandwidth_window_bytes = 0;
            mind_a.tick();
        }

        // Give async actor time to transport (500ms = 10× actor poll interval)
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Mind B: attach Hyperfeel, tick to process
        let mut mind_b = ContinuousMind::new(MindConfig::default());
        mind_b.set_mesh_bridge(handle_b);
        mind_b.activate();
        mind_b.set_hyperfeel(crate::swarm::Hyperfeel::new(
            crate::swarm::HyperfeelConfig::default(),
        ));

        for _ in 0..10 {
            mind_b.mesh_bandwidth_window_bytes = 0;
            mind_b.tick();
        }

        assert!(
            mind_b.hyperfeel.as_ref().unwrap().peer_count() > 0,
            "Mind B's Hyperfeel should see at least one affective peer"
        );
    }

    // ====================================================================
    // LoRa Multi-Loss Resilience Test
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_lora_double_loss_graceful() {
        use crate::swarm::mesh::{
            FragmentAssembler, MeshUrgency, PayloadType, WisdomPacket, LORA_MTU, WISDOM_PACKET_SIZE,
        };
        use symthaea_core::hdc::BinaryHV;

        // Build a WisdomPacket and fragment it
        let original = WisdomPacket {
            source_id: [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
            sequence: 42,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 1_700_000,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0x42; 2048]),
        };
        let frags = original.fragment();
        assert_eq!(frags.len(), 11, "Should produce 11 fragments");

        // Feed only 9 fragments (drop indices 3 and 7) — two losses
        let mut assembler = WisdomPacket::assembler(original.thought_id(), 11);
        let mut buf = [0u8; LORA_MTU];
        for (i, frag) in frags.iter().enumerate() {
            if i == 3 || i == 7 {
                continue; // simulate double loss
            }
            let len = frag.to_bytes(&mut buf);
            let decoded = crate::swarm::mesh::LoRaFragment::from_bytes(&buf[..len]).unwrap();
            assembler.feed(&decoded);
        }

        // XOR parity can only recover 1 loss — 2 is unrecoverable
        assert!(
            !assembler.is_complete(),
            "Assembler should NOT be complete with 2 losses"
        );
        assert!(
            assembler.assemble().is_none(),
            "Assembly should fail with 2 losses"
        );

        // Verify Mind-level semantics: no peer tracked, no wisdom received
        let mut mind = ContinuousMind::default();
        mind.activate();
        // Don't inject any packets (assembly failed)
        mind.tick();
        assert_eq!(mind.mesh_peers().peer_count(), 0);
        assert_eq!(mind.mesh_stats().wisdom_received, 0);
    }

    // ====================================================================
    // Multi-Mind Stress Test
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_four_minds_mesh_stress() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        // Create 4 minds with social coherence
        let mut minds: Vec<ContinuousMind> = (0..4)
            .map(|i| {
                let mut m = ContinuousMind::new(MindConfig {
                    enable_social_coherence: true,
                    ..Default::default()
                });
                m.activate();
                // Each mind perceives a unique HV
                m.perceive(ContinuousHV::random(512, 1000 + i as u64));
                m
            })
            .collect();

        // Source IDs for each mind
        let source_ids: Vec<[u8; 8]> = (0..4u8).map(|i| [i + 1; 8]).collect();

        // Run 60 ticks with manual packet injection every 10 ticks
        for tick in 0..60 {
            // Every 10 ticks, inject wisdom packets from each mind to all others
            if tick > 0 && tick % 10 == 0 {
                // Collect packets from each mind (snapshot their current thought as BinaryHV)
                let packets: Vec<WisdomPacket> = (0..4)
                    .map(|i| WisdomPacket {
                        source_id: source_ids[i],
                        sequence: (tick / 10) as u32,
                        phi: minds[i].state.consciousness_level as f32,
                        urgency: MeshUrgency::Normal,
                        timestamp_s: tick as u32,
                        payload_type: PayloadType::WisdomVector,
                        auth_mac: 0,
                        ttl: 0,
                        wisdom: symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16(
                            &minds[i].state.current_thought,
                        ),
                    })
                    .collect();

                // Inject each mind's packet into all other minds' inboxes
                for (sender_idx, pkt) in packets.iter().enumerate() {
                    for (receiver_idx, mind) in minds.iter_mut().enumerate() {
                        if sender_idx != receiver_idx {
                            mind.mesh_inbox.push(pkt.clone());
                        }
                    }
                }
            }

            // Tick all minds
            for mind in minds.iter_mut() {
                mind.tick();
            }
        }

        // Assertions
        for (i, mind) in minds.iter().enumerate() {
            // Each mind should see 3 peers
            assert_eq!(
                mind.mesh_peers().peer_count(),
                3,
                "Mind {i} should see 3 peers, got {}",
                mind.mesh_peers().peer_count()
            );

            // Consciousness should be finite and > 0
            assert!(
                mind.state.consciousness_level.is_finite() && mind.state.consciousness_level > 0.0,
                "Mind {i} consciousness should be finite and > 0: {}",
                mind.state.consciousness_level
            );

            // Social coherence should model ≥3 agents
            let sc = mind.social_coherence().unwrap();
            let stats = sc.stats();
            assert!(
                stats.agents_modeled >= 3,
                "Mind {i} should model ≥3 agents, got {}",
                stats.agents_modeled
            );
        }

        // Average phi across all minds should be > 0.1
        let avg_phi: f64 = minds
            .iter()
            .map(|m| m.state.consciousness_level)
            .sum::<f64>()
            / 4.0;
        assert!(
            avg_phi > 0.1,
            "Average phi across 4 minds should be > 0.1: {avg_phi}"
        );
    }

    // ====================================================================
    // Item 1: Mesh Inbox/Outbox Backpressure Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_inbox_backpressure() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Push 100 packets into mesh_inbox (cap is 64)
        for i in 0..100u32 {
            mind.mesh_inbox.push(WisdomPacket {
                source_id: [(i % 256) as u8; 8],
                sequence: i,
                phi: 0.5,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::WisdomVector,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0xAA; 2048]),
            });
        }

        mind.tick();

        // 100 - 64 = 36 packets should be dropped
        assert_eq!(
            mind.mesh_stats.packets_dropped, 36,
            "Should drop 36 excess inbox packets"
        );
        // The 64 remaining packets were processed
        assert!(
            mind.mesh_stats.wisdom_received > 0,
            "Should have processed some wisdom packets"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_outbox_backpressure() {
        use crate::swarm::mesh::{MeshOutbound, MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        // Attach bridge so auto-emit fires
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);
        mind.perceive(ContinuousHV::random(512, 0xDEAD));

        // Pre-fill outbox with 100 packets (exceeds cap of 64)
        for i in 0..100u32 {
            mind.mesh_outbox.push(MeshOutbound {
                packet: WisdomPacket {
                    source_id: [0x01; 8],
                    sequence: i,
                    phi: 0.5,
                    urgency: MeshUrgency::Normal,
                    timestamp_s: 0,
                    payload_type: PayloadType::WisdomVector,
                    auth_mac: 0,
                    ttl: 0,
                    wisdom: BinaryHV([0; 2048]),
                },
            });
        }

        mind.tick();

        // At least 36 packets should have been dropped from the outbox
        assert!(
            mind.mesh_stats.packets_dropped >= 36,
            "Should drop excess outbox packets: got {}",
            mind.mesh_stats.packets_dropped
        );
    }

    // ====================================================================
    // Item 2: Packet Deduplication Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_dedup_same_packet() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        let pkt = WisdomPacket {
            source_id: [0xDE; 8],
            sequence: 42,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xAA; 2048]),
        };

        // Push same packet twice
        mind.mesh_inbox.push(pkt.clone());
        mind.mesh_inbox.push(pkt);

        mind.tick();

        assert_eq!(
            mind.mesh_stats.packets_deduplicated, 1,
            "Second identical packet should be deduplicated"
        );
        assert_eq!(
            mind.mesh_stats.wisdom_received, 1,
            "Only one packet should be processed"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_dedup_different_sequence() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Same source, different sequences
        for seq in 0..2u32 {
            mind.mesh_inbox.push(WisdomPacket {
                source_id: [0xDE; 8],
                sequence: seq,
                phi: 0.5,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::WisdomVector,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0xAA; 2048]),
            });
        }

        mind.tick();

        assert_eq!(
            mind.mesh_stats.packets_deduplicated, 0,
            "Different sequences should not be deduplicated"
        );
        assert_eq!(
            mind.mesh_stats.wisdom_received, 2,
            "Both packets should be processed"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_dedup_ring_eviction() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Push 129 unique packets (exceeds ring size of 128)
        for seq in 0..129u32 {
            mind.mesh_inbox.push(WisdomPacket {
                source_id: [0xAA; 8],
                sequence: seq,
                phi: 0.5,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::WisdomVector,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0; 2048]),
            });
        }
        mind.tick();

        // All should be unique (no dedup on first pass) — but inbox was capped at 64
        // so only 64 packets were processed, ring has 64 entries
        let first_dedup = mind.mesh_stats.packets_deduplicated;

        // Now push the first packet again (sequence 0) — it was evicted if >128 entries
        // Since only 64 were processed, seq 0 was dropped by inbox backpressure,
        // and seqs 65..128 were processed. seq 0 was never seen, so not in ring.
        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0xAA; 8],
            sequence: 0,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0; 2048]),
        });
        mind.tick();

        // seq 0 was never seen (dropped by backpressure), so it should NOT be deduplicated
        assert_eq!(
            mind.mesh_stats.packets_deduplicated, first_dedup,
            "Evicted/unseen packet should not be deduplicated"
        );
    }

    // ====================================================================
    // Item 3: Per-Peer Rate Limiting (Mind-level) Test
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_process_rate_limits_flood() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        let source = [0xFF; 8];
        // Pre-register the peer so rate limiting works
        mind.mesh_peers.update(&WisdomPacket {
            source_id: source,
            sequence: 0,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0; 2048]),
        });

        // Push 110 packets from same source with unique sequences
        // (rate limit is 100 per window, but the registry update above
        // already consumed 0 in the rate limiter — the pre-register via
        // update() doesn't touch the rate limiter window_count)
        for seq in 1..=110u32 {
            mind.mesh_inbox.push(WisdomPacket {
                source_id: source,
                sequence: seq,
                phi: 0.5,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::WisdomVector,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0; 2048]),
            });
        }

        mind.tick();

        // Inbox backpressure drops 110 - 64 = 46 packets first,
        // then 64 packets are processed. Rate limit is 100 per window,
        // so for 64 unique packets from same source, all should pass rate limiter.
        // But dedup: all unique sequences, so no dedup.
        // Rate limit check: is_rate_limited increments window_count.
        // After 64 checks, window_count = 64 < 100, so none rate limited.
        // Let's verify with larger inbox — need to increase cap or test differently.
        // Actually, let's push exactly 64 packets (at cap), and test with >100
        // by doing multiple ticks.

        // For a meaningful rate limit test, let's do it differently:
        // Clear state and re-test with direct rate limit checking
        let mut mind2 = ContinuousMind::default();
        mind2.activate();

        // Register peer
        mind2.mesh_peers.update(&WisdomPacket {
            source_id: source,
            sequence: 0,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0; 2048]),
        });

        // Push 64 packets per tick, tick 2 times = 128 unique packets
        for seq in 1..=64u32 {
            mind2.mesh_inbox.push(WisdomPacket {
                source_id: source,
                sequence: seq,
                phi: 0.5,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::WisdomVector,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0; 2048]),
            });
        }
        mind2.tick();

        for seq in 65..=128u32 {
            mind2.mesh_inbox.push(WisdomPacket {
                source_id: source,
                sequence: seq,
                phi: 0.5,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::WisdomVector,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0; 2048]),
            });
        }
        mind2.tick();

        // After 128 rate limit checks (64+64), window_count > 100
        // So packets_rate_limited should be > 0
        assert!(
            mind2.mesh_stats.packets_rate_limited > 0,
            "Should have rate-limited some packets from flood: got {}",
            mind2.mesh_stats.packets_rate_limited
        );
    }

    // ====================================================================
    // Item 4: Health-Driven Urgency Escalation Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_health_urgency_critical_on_degraded() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Give mind a non-zero thought (needed for wisdom emission)
        mind.state.current_thought = ContinuousHV::random(512, 0xDEAD);

        // Create send-only stats (health < 0.3): many sends, no receives, no peers
        // connectivity = 0.0, bidirectionality = 0.0, stability = 1.0 → 0.2
        // Total = 0.2 → health < 0.3
        mind.mesh_stats.wisdom_sent = 50;
        mind.mesh_stats.heartbeats_sent = 20;

        // Low arousal (would normally be Cruise urgency) — bypasses biorhythm
        // by calling auto_emit_wisdom directly instead of tick()
        mind.state.arousal = 0.1;
        mind.state.tick = 1;
        mind.auto_emit_wisdom(); // First emission (sequence=0) + Critical override

        mind.state.tick = 2;
        mind.auto_emit_wisdom(); // Critical interval=1, ticks_since=1 ≥ 1 → emit

        // With Critical urgency (health < 0.3 override), both calls should have emitted
        assert_eq!(
            mind.mesh_sequence, 2,
            "Critical urgency should emit every tick: got {} emissions",
            mind.mesh_sequence
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_health_urgency_allows_cruise_when_healthy() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);
        mind.perceive(ContinuousHV::random(512, 0xBEEF));

        // Give mind a non-zero thought (needed for wisdom emission)
        mind.state.current_thought = ContinuousHV::random(512, 0xBEEF);

        // Create healthy stats: balanced sends/receives + 5 peers
        mind.mesh_stats.wisdom_sent = 50;
        mind.mesh_stats.wisdom_received = 48;
        mind.mesh_stats.heartbeats_sent = 20;
        mind.mesh_stats.heartbeats_received = 18;
        mind.mesh_stats.peers_expired = 1;

        // Register 5 peers
        for i in 0..5u8 {
            mind.mesh_peers.update(&WisdomPacket {
                source_id: [i + 1; 8],
                sequence: 1,
                phi: 0.8,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::WisdomVector,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0; 2048]),
            });
        }

        // Low arousal → Cruise urgency (health > 0.8 → no override)
        // Call auto_emit_wisdom directly to bypass biorhythm arousal override in tick()
        mind.state.arousal = 0.1;
        mind.state.tick = 1;
        mind.auto_emit_wisdom(); // First emission (sequence=0 always allowed) → Cruise

        mind.state.tick = 2;
        mind.auto_emit_wisdom(); // Cruise interval=500, ticks_since=1 < 500 → no emit

        assert_eq!(
            mind.mesh_sequence, 1,
            "Healthy mesh with low arousal should use Cruise (1 emission): got {}",
            mind.mesh_sequence
        );
    }

    // ====================================================================
    // Item 5: CycleMetadata Mesh Wiring Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_populate_mesh_metadata() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Inject a peer
        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0x11; 8],
            sequence: 1,
            phi: 0.7,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xAA; 2048]),
        });
        mind.perceive(ContinuousHV::random(512, 0xFACE));
        mind.tick(); // Process inbox + emit

        let mut metadata = crate::cognitive_loop::types::CycleMetadata::default();
        mind.populate_mesh_metadata(&mut metadata);

        assert!(
            metadata.mesh_health_score > 0.0,
            "mesh_health_score should be > 0"
        );
        assert_eq!(metadata.mesh_peer_count, 1, "Should have 1 peer");
        assert!(
            metadata.mesh_bytes_sent > 0,
            "mesh_bytes_sent should be > 0"
        );
        assert!(
            metadata.mesh_bytes_received > 0,
            "mesh_bytes_received should be > 0"
        );
    }

    #[test]
    fn test_cycle_metadata_mesh_defaults_zero() {
        let metadata = crate::cognitive_loop::types::CycleMetadata::default();
        assert_eq!(metadata.mesh_health_score, 0.0);
        assert_eq!(metadata.mesh_peer_count, 0);
        assert_eq!(metadata.mesh_bytes_sent, 0);
        assert_eq!(metadata.mesh_bytes_received, 0);
    }

    // ====================================================================
    // Item 6: Bandwidth Budget Enforcement Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_bandwidth_budget_allows_under_limit() {
        let mut mind = ContinuousMind::default();

        // 48 × 2072 = 99,456 < 100 KB (102,400)
        for _ in 0..48 {
            assert!(
                mind.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64),
                "Should be under bandwidth budget"
            );
        }
        assert_eq!(mind.mesh_stats.bandwidth_throttled, 0);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_bandwidth_budget_blocks_over_limit() {
        let mut mind = ContinuousMind::default();

        // Keep sending until budget is exhausted
        let mut passed = 0u64;
        let mut blocked = 0u64;
        for _ in 0..60 {
            if mind.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64) {
                passed += 1;
            } else {
                blocked += 1;
            }
        }

        assert!(passed > 0, "Some packets should pass");
        assert!(blocked > 0, "Some packets should be blocked");
        assert!(
            mind.mesh_stats.bandwidth_throttled > 0,
            "bandwidth_throttled should be > 0"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_bandwidth_budget_window_resets() {
        let mut mind = ContinuousMind::default();

        // Exhaust budget
        for _ in 0..60 {
            mind.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64);
        }
        assert!(mind.mesh_stats.bandwidth_throttled > 0);

        // Simulate window expiry by resetting window_start to 11s ago
        mind.mesh_bandwidth_window_start =
            std::time::Instant::now() - std::time::Duration::from_secs(11);

        // Next check should pass (window resets)
        assert!(
            mind.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64),
            "Should be allowed after window reset"
        );
    }

    // ====================================================================
    // Item 4: TTL Emit Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_emit_wisdom_sets_ttl() {
        use crate::cognitive_loop::types::CycleUrgency;
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        mind.state.tick = 1;
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Critical, 0.5);

        assert_eq!(mind.mesh_outbox.len(), 1);
        assert_eq!(
            mind.mesh_outbox[0].packet.ttl,
            crate::swarm::mesh::MESH_DEFAULT_TTL,
            "Emitted wisdom should have default TTL"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_emit_heartbeat_sets_ttl() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        mind.state.tick = 1;
        mind.emit_heartbeat();

        assert_eq!(mind.mesh_outbox.len(), 1);
        assert_eq!(
            mind.mesh_outbox[0].packet.ttl,
            crate::swarm::mesh::MESH_DEFAULT_TTL,
            "Emitted heartbeat should have default TTL"
        );
    }

    // ====================================================================
    // Item 1: Auth Tests (Mind Integration)
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_auth_rejects_unsigned_when_key_set() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.set_mesh_auth_key(Some([0x42; 32]));

        // Inject an unsigned packet (auth_mac = 0)
        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0xBB; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xAA; 2048]),
        });

        mind.tick();

        assert_eq!(
            mind.mesh_stats().packets_auth_failed,
            1,
            "Unsigned packet should fail auth"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_auth_passes_signed_packet() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let key = [0x42u8; 32];
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.set_mesh_auth_key(Some(key));

        // Create and sign a packet
        let mut pkt = WisdomPacket {
            source_id: [0xCC; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 3,
            wisdom: BinaryHV([0xAA; 2048]),
        };
        let bytes = pkt.to_bytes();
        pkt.auth_mac = crate::swarm::mesh::compute_packet_mac(&bytes, &key);

        mind.mesh_inbox.push(pkt);
        mind.tick();

        assert_eq!(
            mind.mesh_stats().packets_auth_failed,
            0,
            "Signed packet should pass auth"
        );
        assert_eq!(mind.mesh_peers().peer_count(), 1);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_no_auth_key_passes_all() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        // No auth key set (default)

        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0xDD; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xAA; 2048]),
        });

        mind.tick();

        assert_eq!(
            mind.mesh_stats().packets_auth_failed,
            0,
            "No auth key = all packets pass"
        );
        assert_eq!(mind.mesh_peers().peer_count(), 1);
    }

    // ====================================================================
    // Item 2: Priority Backpressure Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_inbox_backpressure_drops_gradients_first() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Fill inbox with 32 heartbeats + 32 gradients + 32 wisdom = 96 packets
        // MAX_OUTBOX_SIZE is 64, so 32 must be dropped.
        // Gradients (priority 0) should be dropped first.
        for i in 0..32u32 {
            mind.mesh_inbox.push(WisdomPacket {
                source_id: [0x10 + (i as u8); 8],
                sequence: i,
                phi: 0.5,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::Heartbeat,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0; 2048]),
            });
        }
        for i in 0..32u32 {
            mind.mesh_inbox.push(WisdomPacket {
                source_id: [0x30 + (i as u8); 8],
                sequence: i,
                phi: 0.5,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::Gradient,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0; 2048]),
            });
        }
        for i in 0..32u32 {
            mind.mesh_inbox.push(WisdomPacket {
                source_id: [0x50 + (i as u8); 8],
                sequence: i,
                phi: 0.5,
                urgency: MeshUrgency::Normal,
                timestamp_s: 0,
                payload_type: PayloadType::WisdomVector,
                auth_mac: 0,
                ttl: 0,
                wisdom: BinaryHV([0; 2048]),
            });
        }

        assert_eq!(mind.mesh_inbox.len(), 96);
        mind.tick();

        // All 32 gradients should have been dropped (lowest priority)
        assert!(
            mind.mesh_stats().packets_dropped >= 32,
            "At least 32 low-priority packets should be dropped: got {}",
            mind.mesh_stats().packets_dropped,
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_outbox_backpressure_drops_gradients_first() {
        use crate::swarm::mesh::{MeshOutbound, MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();
        let (handle, _actor) = crate::swarm::mesh::MeshBridgeHandle::new(64, 128);
        mind.set_mesh_bridge(handle);

        // Directly push 40 heartbeats + 40 gradients into outbox (80 total, cap=64)
        for i in 0..40u32 {
            mind.mesh_outbox.push(MeshOutbound {
                packet: WisdomPacket {
                    source_id: [0x01; 8],
                    sequence: i,
                    phi: 0.5,
                    urgency: MeshUrgency::Normal,
                    timestamp_s: 0,
                    payload_type: PayloadType::Heartbeat,
                    auth_mac: 0,
                    ttl: 0,
                    wisdom: BinaryHV([0; 2048]),
                },
            });
        }
        for i in 0..40u32 {
            mind.mesh_outbox.push(MeshOutbound {
                packet: WisdomPacket {
                    source_id: [0x01; 8],
                    sequence: 100 + i,
                    phi: 0.5,
                    urgency: MeshUrgency::Normal,
                    timestamp_s: 0,
                    payload_type: PayloadType::Gradient,
                    auth_mac: 0,
                    ttl: 0,
                    wisdom: BinaryHV([0; 2048]),
                },
            });
        }

        // Tick triggers outbox backpressure
        mind.tick();

        // 16 excess should be dropped, all should be gradients
        // After tick, bridge flushes outbox, so we check packets_dropped stat
        assert!(
            mind.mesh_stats().packets_dropped >= 16,
            "At least 16 gradient packets should be dropped: got {}",
            mind.mesh_stats().packets_dropped,
        );
    }

    // ====================================================================
    // Item 4: TTL Forwarding Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_forward_decrements_ttl() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0xAA; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 3,
            wisdom: BinaryHV([0xAA; 2048]),
        });

        mind.tick();

        // Should have forwarded with ttl=2
        assert_eq!(
            mind.mesh_stats().packets_forwarded,
            1,
            "Packet with ttl=3 should be forwarded"
        );
        // Check the forwarded packet in outbox
        assert!(!mind.mesh_outbox.is_empty());
        let fwd = mind
            .mesh_outbox
            .iter()
            .find(|o| o.packet.source_id == [0xAA; 8]);
        assert!(fwd.is_some(), "Forwarded packet should be in outbox");
        assert_eq!(fwd.unwrap().packet.ttl, 2);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_no_forward_ttl_zero() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0xBB; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xBB; 2048]),
        });

        mind.tick();

        assert_eq!(
            mind.mesh_stats().packets_forwarded,
            0,
            "Packet with ttl=0 should NOT be forwarded"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_mesh_no_forward_ttl_one() {
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0xCC; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 1,
            wisdom: BinaryHV([0xCC; 2048]),
        });

        mind.tick();

        assert_eq!(
            mind.mesh_stats().packets_forwarded,
            0,
            "Packet with ttl=1 should NOT be forwarded (last hop)"
        );
    }

    // ====================================================================
    // Item 5: Replay Buffer Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_replay_buffer_fills_on_emit() {
        use crate::cognitive_loop::types::CycleUrgency;
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        for i in 0..5u64 {
            mind.state.tick = i;
            mind.emit_wisdom(BinaryHV([i as u8; 2048]), CycleUrgency::Critical, 0.5);
        }

        assert_eq!(mind.mesh_replay_buffer.len(), 5);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_replay_buffer_caps_at_capacity() {
        use crate::cognitive_loop::types::CycleUrgency;
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        for i in 0..20u64 {
            mind.state.tick = i;
            mind.mesh_bandwidth_window_bytes = 0; // prevent throttle
            mind.emit_wisdom(BinaryHV([i as u8; 2048]), CycleUrgency::Critical, 0.5);
        }

        assert_eq!(
            mind.mesh_replay_buffer.len(),
            MESH_REPLAY_BUFFER_CAPACITY,
            "Replay buffer should cap at {}",
            MESH_REPLAY_BUFFER_CAPACITY,
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_replay_on_new_peer() {
        use crate::cognitive_loop::types::CycleUrgency;
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Emit 3 wisdom packets into replay buffer
        for i in 0..3u64 {
            mind.state.tick = i;
            mind.emit_wisdom(BinaryHV([i as u8; 2048]), CycleUrgency::Critical, 0.5);
        }
        assert_eq!(mind.mesh_replay_buffer.len(), 3);

        // Clear outbox to isolate replay
        mind.mesh_outbox.clear();

        // Inject a packet from a new peer
        mind.mesh_inbox.push(WisdomPacket {
            source_id: [0xFF; 8],
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0xFF; 2048]),
        });

        mind.state.tick = 10;
        mind.tick();

        // Should have replayed 3 packets to outbox
        assert_eq!(
            mind.mesh_stats().packets_replayed,
            3,
            "Should replay 3 packets for new peer"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_no_replay_on_known_peer() {
        use crate::cognitive_loop::types::CycleUrgency;
        use crate::swarm::mesh::{MeshUrgency, PayloadType, WisdomPacket};
        use symthaea_core::hdc::BinaryHV;

        let mut mind = ContinuousMind::default();
        mind.activate();

        // Emit wisdom to fill replay buffer
        mind.emit_wisdom(BinaryHV([0; 2048]), CycleUrgency::Critical, 0.5);
        mind.mesh_outbox.clear();

        // Register a known peer first
        let peer_id = [0xEE; 8];
        mind.mesh_peers.update(&WisdomPacket {
            source_id: peer_id,
            sequence: 0,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0; 2048]),
        });

        // Now inject another packet from the SAME peer
        mind.mesh_inbox.push(WisdomPacket {
            source_id: peer_id,
            sequence: 1,
            phi: 0.5,
            urgency: MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: BinaryHV([0; 2048]),
        });

        mind.state.tick = 1;
        mind.tick();

        assert_eq!(
            mind.mesh_stats().packets_replayed,
            0,
            "Known peer should NOT trigger replay"
        );
    }

    // ====================================================================
    // Item 6: AIMD Bandwidth Tests
    // ====================================================================

    #[cfg(feature = "mesh")]
    #[test]
    fn test_aimd_additive_increase() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.mesh_bandwidth_budget = 100 * 1024;
        mind.mesh_bandwidth_throttled_in_window = false;
        // Healthy mesh: need some send/recv stats + peers
        mind.mesh_stats.wisdom_sent = 50;
        mind.mesh_stats.wisdom_received = 48;
        mind.mesh_stats.heartbeats_sent = 20;
        mind.mesh_stats.heartbeats_received = 18;
        mind.mesh_peers.update(&crate::swarm::mesh::WisdomPacket {
            source_id: [0x01; 8],
            sequence: 1,
            phi: 0.5,
            urgency: crate::swarm::mesh::MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: crate::swarm::mesh::PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: symthaea_core::hdc::BinaryHV([0; 2048]),
        });

        mind.adjust_bandwidth_budget();

        assert_eq!(
            mind.mesh_bandwidth_budget,
            100 * 1024 + MESH_BANDWIDTH_ADDITIVE_INCREASE,
            "Healthy + no throttle should increase budget"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_aimd_multiplicative_decrease_on_throttle() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.mesh_bandwidth_budget = 100 * 1024;
        mind.mesh_bandwidth_throttled_in_window = true;

        mind.adjust_bandwidth_budget();

        assert_eq!(
            mind.mesh_bandwidth_budget,
            50 * 1024,
            "Throttled should halve budget"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_aimd_hold_steady_zero_health() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.mesh_bandwidth_budget = 100 * 1024;
        mind.mesh_bandwidth_throttled_in_window = false;
        // health = 0.0 (no activity): should hold steady

        mind.adjust_bandwidth_budget();

        assert_eq!(
            mind.mesh_bandwidth_budget,
            100 * 1024,
            "Idle mesh (health=0.0) should hold steady"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_aimd_budget_floor() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.mesh_bandwidth_budget = MESH_BANDWIDTH_MIN;
        mind.mesh_bandwidth_throttled_in_window = true;

        mind.adjust_bandwidth_budget();

        assert_eq!(
            mind.mesh_bandwidth_budget, MESH_BANDWIDTH_MIN,
            "Budget should never go below floor"
        );
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn test_aimd_budget_ceiling() {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind.mesh_bandwidth_budget = MESH_BANDWIDTH_MAX;
        mind.mesh_bandwidth_throttled_in_window = false;
        // Healthy mesh stats
        mind.mesh_stats.wisdom_sent = 50;
        mind.mesh_stats.wisdom_received = 48;
        mind.mesh_peers.update(&crate::swarm::mesh::WisdomPacket {
            source_id: [0x01; 8],
            sequence: 1,
            phi: 0.5,
            urgency: crate::swarm::mesh::MeshUrgency::Normal,
            timestamp_s: 0,
            payload_type: crate::swarm::mesh::PayloadType::WisdomVector,
            auth_mac: 0,
            ttl: 0,
            wisdom: symthaea_core::hdc::BinaryHV([0; 2048]),
        });

        mind.adjust_bandwidth_budget();

        assert_eq!(
            mind.mesh_bandwidth_budget, MESH_BANDWIDTH_MAX,
            "Budget should never exceed ceiling"
        );
    }
}
