// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Continuous Mind: The Integrated Consciousness System
//!
//! Provides the main orchestration layer for the conscious AI system,
//! integrating perception, reasoning, memory, and action into a unified
//! continuous-time cognitive architecture.

pub mod async_mind;
mod config;
mod epistemic;
mod federated;
mod goals;
pub mod intent;
pub mod knowledge;
#[cfg(feature = "mesh")]
mod mesh;
#[cfg(feature = "provenance")]
pub mod provenance;
mod social;
pub mod structured_thought;
mod swarm;
mod tick;
mod utils;

pub use async_mind::{AsyncMind, AsyncMindHandle, connect_social};
pub use config::*;
pub use intent::{
    ConceptLabel, ConceptPrototype, EpistemicAssessment, IntentClassification, IntentClassifier,
    IntentScores,
};
pub use knowledge::{DomainKnowledge, KnowledgeEntry, SeedingResult};
pub use structured_thought::*;
pub use utils::{
    EPSILON, EPSILON_F32, float_eq, float_eq_f32, is_nonzero, is_nonzero_f32, is_zero, is_zero_f32,
};

use crate::memory::memory_coordinator::MemorySource;
use std::collections::VecDeque;
use symthaea_core::hdc::ContinuousHV;

/// Maximum number of messages retained in unbounded outboxes (federated, social, mesh).
/// Oldest messages are drained when the cap is exceeded, preventing unbounded growth
/// when no bridge or consumer is attached.
const MAX_OUTBOX_SIZE: usize = 64;

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
    /// Learned projection for social signals (perception dim → cognitive dim).
    /// Bridges the 512D→16384D gap for collective consciousness emergence.
    pub(crate) social_projection: Option<symthaea_core::hdc::projection::LearnedProjection>,
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
    /// Tick counter for last moral topology emission.
    #[cfg(feature = "mesh")]
    mesh_moral_topology_last_tick: u64,
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
    mesh_seen_packets: VecDeque<([u8; 8], u32, u8)>,
    /// Start of the current bandwidth budget window.
    #[cfg(feature = "mesh")]
    mesh_bandwidth_window_start: std::time::Instant,
    /// Bytes sent within the current bandwidth budget window.
    #[cfg(feature = "mesh")]
    mesh_bandwidth_window_bytes: u64,
    /// Optional BLAKE3 key for packet authentication.
    #[cfg(feature = "mesh")]
    mesh_auth_key: Option<[u8; 32]>,
    /// Optional ChaCha20-Poly1305 encryption key pair with rotation support.
    #[cfg(feature = "mesh-encryption")]
    pub(crate) mesh_encryption_key: Option<crate::swarm::mesh::RotatingKeyPair>,
    /// Random epoch byte for nonce construction — prevents restart nonce reuse.
    /// Generated once at Mind construction from thread_rng.
    #[cfg(feature = "mesh-encryption")]
    pub(crate) mesh_encryption_epoch: u8,
    /// Per-peer X25519 key store for Diffie-Hellman key agreement.
    #[cfg(feature = "mesh-key-exchange")]
    pub(crate) mesh_peer_keys: Option<crate::swarm::mesh::PeerKeyStore>,
    /// Automatic key rotation interval (ticks). 0 = disabled.
    #[cfg(feature = "mesh-encryption")]
    mesh_auto_rotate_interval: u64,
    /// Tick of last automatic key rotation.
    #[cfg(feature = "mesh-encryption")]
    mesh_last_rotation_tick: u64,
    /// Ring buffer of recently-emitted wisdom packets for partition recovery replay.
    #[cfg(feature = "mesh")]
    mesh_replay_buffer: std::collections::VecDeque<crate::swarm::mesh::WisdomPacket>,
    /// Current dynamic bandwidth budget (AIMD-adjusted).
    #[cfg(feature = "mesh")]
    mesh_bandwidth_budget: u64,
    /// Whether any emission was throttled within the current bandwidth window.
    #[cfg(feature = "mesh")]
    mesh_bandwidth_throttled_in_window: bool,
    /// Cached moral topology summary for mesh telemetry gossip.
    #[cfg(feature = "mesh")]
    cached_moral_topology: Option<crate::hdc::moral_topology::MoralTopologySummary>,
    /// Memory coordinator for cross-tier integration (graduation, pruning, retrieval tracking).
    pub(crate) memory_coordinator: crate::memory::memory_coordinator::MemoryCoordinator,
    /// Optional episodic memory for dream-state causal pruning.
    pub(crate) episodic_memory: Option<crate::memory::episodic_replay::EpisodicMemory>,
    /// Dream-level neuromodulator bath for sleep neurochemistry.
    /// Tracks adenosine clearance, allostatic recovery, and 5-HT1A up-regulation
    /// independently from the cognitive loop's bath (which doesn't run during dreams).
    /// Science: Xie et al. (2013), Piomelli (2003), Blier & de Montigny (1994).
    pub(crate) dream_bath: symthaea_neuromodulators::NeuromodulatorBath,
    /// Optional sender for forwarding swarm events (affective sync, federated rounds)
    /// to the CognitiveLoopService's SwarmManager via mpsc channel.
    /// Set via `set_swarm_channel()` after CLS creates the channel.
    pub(crate) swarm_event_tx: Option<std::sync::mpsc::Sender<crate::cognitive_loop::SwarmEvent>>,
    /// Receiver for mesh outbound packets from CLS (beacons, name responses, etc.).
    /// Set via `set_mesh_outbound_rx()`. Drained each tick in `sync_mesh_bridge()`.
    #[cfg(feature = "mesh")]
    pub(crate) mesh_outbound_rx:
        Option<std::sync::Mutex<std::sync::mpsc::Receiver<crate::swarm::mesh::MeshOutbound>>>,
    /// Holochain Cortex for trust and validation.
    pub(crate) cortex: crate::swarm::HolochainCortex,
    /// Optional LLM backend for swarm projection gradient exchange.
    /// When set, `emit_gradients()` also exports L-SSM projection weights
    /// and `process_federated()` applies aggregated peer weights.
    #[cfg(feature = "liquid-mamba")]
    pub(crate) llm_backend: Option<std::sync::Arc<dyn crate::language::llm_backend::LLMBackend>>,
    /// Genesis-derived identity for swarm gradient source_id (L-SSM).
    #[cfg(feature = "liquid-mamba")]
    pub(crate) genesis_identity: [u8; 32],
}

impl ContinuousMind {
    /// Create a new continuous mind with default (non-deterministic) initialization.
    ///
    /// For reproducible behavior across runs, use [`Self::from_genesis()`] instead.
    pub fn new(config: MindConfig) -> Self {
        let dim = config.dimension;
        let perception_dim = config.dimension;
        let social_projection_enabled = config.social_projection_enabled;
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
            social_projection: if social_projection_enabled {
                Some(symthaea_core::hdc::projection::LearnedProjection::new(
                    perception_dim,
                    16384,
                ))
            } else {
                None
            },
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
            mesh_moral_topology_last_tick: 0,
            #[cfg(feature = "mesh")]
            mesh_affective_last_tick: 0,
            #[cfg(feature = "mesh")]
            mesh_affective_sequence: 0,
            #[cfg(feature = "mesh")]
            mesh_stats: crate::swarm::mesh::MeshStats::default(),
            #[cfg(feature = "mesh")]
            mesh_seen_packets: VecDeque::with_capacity(128),
            #[cfg(feature = "mesh")]
            mesh_bandwidth_window_start: std::time::Instant::now(),
            #[cfg(feature = "mesh")]
            mesh_bandwidth_window_bytes: 0,
            #[cfg(feature = "mesh")]
            mesh_auth_key: None,
            #[cfg(feature = "mesh-encryption")]
            mesh_encryption_key: None,
            #[cfg(feature = "mesh-encryption")]
            mesh_encryption_epoch: rand::Rng::r#gen::<u8>(&mut rand::thread_rng()),
            #[cfg(feature = "mesh-key-exchange")]
            mesh_peer_keys: None,
            #[cfg(feature = "mesh-encryption")]
            mesh_auto_rotate_interval: 0,
            #[cfg(feature = "mesh-encryption")]
            mesh_last_rotation_tick: 0,
            #[cfg(feature = "mesh")]
            mesh_replay_buffer: std::collections::VecDeque::with_capacity(
                mesh::MESH_REPLAY_BUFFER_CAPACITY,
            ),
            #[cfg(feature = "mesh")]
            mesh_bandwidth_budget: mesh::MESH_BANDWIDTH_INITIAL,
            #[cfg(feature = "mesh")]
            mesh_bandwidth_throttled_in_window: false,
            #[cfg(feature = "mesh")]
            cached_moral_topology: None,
            memory_coordinator: crate::memory::memory_coordinator::MemoryCoordinator::default(),
            episodic_memory: None,
            swarm_event_tx: None,
            #[cfg(feature = "mesh")]
            mesh_outbound_rx: None,
            dream_bath: symthaea_neuromodulators::NeuromodulatorBath::default(),
            cortex: crate::swarm::HolochainCortex::default(),
            #[cfg(feature = "liquid-mamba")]
            llm_backend: None,
            #[cfg(feature = "liquid-mamba")]
            genesis_identity: [0u8; 32],
        }
    }

    /// Create a continuous mind with deterministic RNG from a genesis seed.
    ///
    /// All HDC vectors and network weights derive from `genesis` + `label`,
    /// making the system fully reproducible across runs.
    pub fn from_genesis(
        config: MindConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        let mut mind = Self::new(config);
        mind.seeded_rng = Some(genesis.domain(&format!("{label}::mind")));
        #[cfg(feature = "liquid-mamba")]
        {
            use rand::RngCore;
            let mut id_rng = genesis.domain("swarm::source_id");
            id_rng.fill_bytes(&mut mind.genesis_identity);
        }
        mind
    }

    /// Queue an input for processing on the next `tick()`.
    pub fn input(&mut self, input: MindInput) {
        self.input_queue.push(input);
    }

    /// Queue a raw perception (continuous HDC vector) for the next tick.
    pub fn perceive(&mut self, content: ContinuousHV) {
        self.input(MindInput::new(InputType::Perception, content));
    }

    /// Inject a surprise signal into the mind, boosting arousal and triggering
    /// heightened attention.
    ///
    /// `magnitude` is clamped to [0.0, 1.0]. The signal:
    /// - Boosts arousal toward alertness (Yerkes-Dodson, 1908)
    /// - Shifts emotional valence slightly negative (unexpected events)
    /// - Injects a high-entropy perception to capture attention
    ///
    /// Called by the facade when action outcomes deviate from expectations.
    pub fn inject_surprise(&mut self, magnitude: f32) {
        let m = magnitude.clamp(0.0, 1.0);

        // Boost arousal: blend toward full alertness by magnitude
        self.state.arousal = (self.state.arousal + m * (1.0 - self.state.arousal)).clamp(0.0, 1.0);

        // Slight negative valence shift (surprise ≠ positive)
        self.state.emotional_valence = (self.state.emotional_valence - m * 0.2).clamp(-1.0, 1.0);

        // Boost cognitive temperature → more exploratory sampling
        self.state.mood_temperature = (self.state.mood_temperature + m * 0.3).clamp(0.0, 2.0);

        tracing::debug!(
            target: "symthaea::mind::surprise",
            magnitude = m,
            arousal = self.state.arousal,
            valence = self.state.emotional_valence,
            temperature = self.state.mood_temperature,
            "Surprise injected"
        );
    }

    /// Set the original input text for intent classification.
    ///
    /// Call this before `tick()` to enable HDC-based intent inference.
    pub fn set_input_text(&mut self, text: impl Into<String>) {
        self.last_input_text = Some(text.into());
    }

    /// Install a swarm event sender for forwarding affective sync and
    /// federated round results to the CognitiveLoopService's SwarmManager.
    ///
    /// Call this after `CognitiveLoopService::create_swarm_event_channel()`.
    pub fn set_swarm_channel(
        &mut self,
        tx: std::sync::mpsc::Sender<crate::cognitive_loop::SwarmEvent>,
    ) {
        self.swarm_event_tx = Some(tx);
    }

    /// Set the mesh outbound receiver from CLS for sovereign beacon emission.
    #[cfg(feature = "mesh")]
    pub fn set_mesh_outbound_rx(
        &mut self,
        rx: std::sync::mpsc::Receiver<crate::swarm::mesh::MeshOutbound>,
    ) {
        self.mesh_outbound_rx = Some(std::sync::Mutex::new(rx));
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

    /// Set an active goal with a priority weight and HDC embedding.
    pub fn set_goal(
        &mut self,
        description: impl Into<String>,
        embedding: ContinuousHV,
        priority: f32,
    ) {
        let mut input = MindInput::new(InputType::Goal, embedding);
        input.priority = priority;
        input
            .metadata
            .insert("description".to_string(), description.into());

        self.input(input);
    }

    /// Activate the mind, enabling perception processing on subsequent ticks.
    pub fn activate(&mut self) {
        self.state.is_active = true;
    }

    /// Deactivate the mind, pausing perception processing.
    pub fn deactivate(&mut self) {
        self.state.is_active = false;
    }

    /// Get a read-only reference to the current mind state (arousal, valence, thought, etc.).
    pub fn state(&self) -> &MindState {
        &self.state
    }

    /// Get the configuration this mind was created with.
    pub fn config(&self) -> &MindConfig {
        &self.config
    }

    /// Get cumulative mind statistics (tick count, perception count, etc.).
    pub fn stats(&self) -> &MindStats {
        &self.stats
    }

    /// Get working memory contents (7±2 items, activation-decay managed).
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
            .map(|item| {
                (
                    item.content,
                    item.steps_survived,
                    item.source,
                    item.is_verified,
                )
            })
            .collect()
    }

    /// Drain evicted items with metadata for tagged persistence.
    pub fn take_evicted_tagged(&mut self) -> Vec<EvictedMemory> {
        std::mem::take(&mut self.evicted_items)
    }

    /// Get all currently active (non-completed) goals.
    pub fn active_goals(&self) -> Vec<&Goal> {
        self.goals.iter().filter(|g| g.is_active).collect()
    }

    /// Awaken the mind — activate, reset timers, and begin consciousness processing.
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

    /// Request graceful shutdown of the mind (checked each tick via `is_shutdown_requested()`).
    pub fn request_shutdown(&mut self) {
        self.state.is_active = false;
        self.state.is_conscious = false;
        self.shutdown_requested = true;
    }

    /// Returns `true` if `request_shutdown()` has been called.
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested
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

    /// Attach an LLM backend for swarm projection gradient exchange.
    ///
    /// When set, `emit_gradients()` exports L-SSM projection weights alongside
    /// HDC world-model gradients, and `process_federated()` applies incoming
    /// aggregated peer weights to the backend.
    #[cfg(feature = "liquid-mamba")]
    pub fn set_llm_backend(
        &mut self,
        backend: std::sync::Arc<dyn crate::language::llm_backend::LLMBackend>,
    ) {
        self.llm_backend = Some(backend);
    }
}

impl Default for ContinuousMind {
    fn default() -> Self {
        Self::new(MindConfig::default())
    }
}

#[cfg(test)]
mod tests;
