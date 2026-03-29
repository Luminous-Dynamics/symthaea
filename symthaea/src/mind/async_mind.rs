// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Async actor wrapper for [`ContinuousMind`].
//!
//! Provides [`AsyncMindHandle`], a lightweight clone-able handle that sends
//! commands to a `ContinuousMind` owned by a dedicated tokio task. This
//! enables multi-agent scenarios where several minds run concurrently on
//! separate tasks, exchanging social messages via their handles.
//!
//! # Example
//!
//! ```ignore
//! let (handle, join) = AsyncMind::spawn(MindConfig::default());
//! handle.perceive(embedding).await;
//! handle.tick().await;  // returns Option<MindOutput>
//! let state = handle.snapshot().await;
//! handle.shutdown().await;
//! join.await.unwrap();
//! ```

use symthaea_core::hdc::ContinuousHV;
use tokio::sync::{mpsc, oneshot};

use super::{
    ContinuousMind, MindConfig, MindInput, MindOutput, MindState, MindStats, SocialMessage,
};

/// Commands sent from [`AsyncMindHandle`] to the actor task.
enum MindCommand {
    /// Run one cognitive tick and return the output.
    Tick(oneshot::Sender<Option<MindOutput>>),
    /// Enqueue a perception input.
    Perceive(ContinuousHV),
    /// Enqueue a text perception (text + embedding).
    PerceiveText(String, ContinuousHV),
    /// Enqueue a raw MindInput.
    Input(MindInput),
    /// Set a goal.
    SetGoal(String, ContinuousHV, f32),
    /// Activate the mind.
    Activate,
    /// Get a snapshot of the current state.
    Snapshot(oneshot::Sender<MindState>),
    /// Get statistics.
    Stats(oneshot::Sender<MindStats>),
    /// Receive a social message from a peer.
    ReceiveSocial(SocialMessage),
    /// Receive a swarm gossip message (e.g. BrainMutation).
    SwarmGossip(crate::swarm::SwarmMessage),
    /// Update thermodynamic load (power/heat).
    UpdateThermodynamics(f32),
    /// Drain outgoing social messages.
    DrainSocialOutbox(oneshot::Sender<Vec<SocialMessage>>),
    /// Seed working memory with domain knowledge.
    SeedMemory(oneshot::Sender<super::SeedingResult>),
    /// Graceful shutdown.
    Shutdown,
}

/// A lightweight, clone-able handle to an async mind actor.
///
/// All methods are async and non-blocking — they send a command to the
/// actor task and (for query methods) await the response via a oneshot
/// channel.
#[derive(Clone)]
pub struct AsyncMindHandle {
    tx: mpsc::Sender<MindCommand>,
}

impl AsyncMindHandle {
    /// Run one cognitive tick.
    ///
    /// The tick executes on the actor task, so this method only blocks on
    /// the channel round-trip, not on the actual computation.
    pub async fn tick(&self) -> Option<MindOutput> {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self.tx.send(MindCommand::Tick(resp_tx)).await.is_err() {
            tracing::warn!("AsyncMind actor has stopped — tick command dropped");
            return None;
        }
        resp_rx.await.unwrap_or(None)
    }

    /// Add a perception input.
    pub async fn perceive(&self, content: ContinuousHV) {
        if self.tx.send(MindCommand::Perceive(content)).await.is_err() {
            tracing::warn!("AsyncMind actor has stopped — perceive command dropped");
        }
    }

    /// Add a text perception.
    pub async fn perceive_text(&self, text: String, embedding: ContinuousHV) {
        if self
            .tx
            .send(MindCommand::PerceiveText(text, embedding))
            .await
            .is_err()
        {
            tracing::warn!("AsyncMind actor has stopped — perceive_text command dropped");
        }
    }

    /// Add a raw input.
    pub async fn input(&self, input: MindInput) {
        if self.tx.send(MindCommand::Input(input)).await.is_err() {
            tracing::warn!("AsyncMind actor has stopped — input command dropped");
        }
    }

    /// Set a goal.
    pub async fn set_goal(&self, description: String, embedding: ContinuousHV, priority: f32) {
        if self
            .tx
            .send(MindCommand::SetGoal(description, embedding, priority))
            .await
            .is_err()
        {
            tracing::warn!("AsyncMind actor has stopped — set_goal command dropped");
        }
    }

    /// Activate the mind.
    pub async fn activate(&self) {
        if self.tx.send(MindCommand::Activate).await.is_err() {
            tracing::warn!("AsyncMind actor has stopped — activate command dropped");
        }
    }

    /// Get a snapshot of the current mind state.
    pub async fn snapshot(&self) -> MindState {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self.tx.send(MindCommand::Snapshot(resp_tx)).await.is_err() {
            tracing::warn!("AsyncMind actor has stopped — snapshot command dropped");
            return MindState::default();
        }
        resp_rx.await.unwrap_or_default()
    }

    /// Get mind statistics.
    pub async fn stats(&self) -> MindStats {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self.tx.send(MindCommand::Stats(resp_tx)).await.is_err() {
            tracing::warn!("AsyncMind actor has stopped — stats command dropped");
            return MindStats::default();
        }
        resp_rx.await.unwrap_or_default()
    }

    /// Send a social message to this mind (from a peer).
    pub async fn receive_social(&self, msg: SocialMessage) {
        if self.tx.send(MindCommand::ReceiveSocial(msg)).await.is_err() {
            tracing::warn!("AsyncMind actor has stopped — receive_social command dropped");
        }
    }

    /// Receive a swarm gossip message (from a peer).
    pub async fn receive_swarm_gossip(&self, msg: crate::swarm::SwarmMessage) {
        if self.tx.send(MindCommand::SwarmGossip(msg)).await.is_err() {
            tracing::warn!("AsyncMind actor has stopped — receive_swarm_gossip command dropped");
        }
    }

    /// Update thermodynamic load (power/heat).
    pub async fn update_thermodynamics(&self, load: f32) {
        if self
            .tx
            .send(MindCommand::UpdateThermodynamics(load))
            .await
            .is_err()
        {
            tracing::warn!("AsyncMind actor has stopped — update_thermodynamics command dropped");
        }
    }

    /// Drain outgoing social messages (for broadcasting to peers).
    pub async fn drain_social_outbox(&self) -> Vec<SocialMessage> {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self
            .tx
            .send(MindCommand::DrainSocialOutbox(resp_tx))
            .await
            .is_err()
        {
            tracing::warn!("AsyncMind actor has stopped — drain_social_outbox command dropped");
            return Vec::new();
        }
        resp_rx.await.unwrap_or_default()
    }

    /// Seed working memory with domain knowledge.
    pub async fn seed_memory(&self) -> super::SeedingResult {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self
            .tx
            .send(MindCommand::SeedMemory(resp_tx))
            .await
            .is_err()
        {
            tracing::warn!("AsyncMind actor has stopped — seed_memory command dropped");
            return super::SeedingResult {
                prototypes_seeded: 0,
                categories: Vec::new(),
                avg_magnitude: 0.0,
            };
        }
        resp_rx.await.unwrap_or(super::SeedingResult {
            prototypes_seeded: 0,
            categories: Vec::new(),
            avg_magnitude: 0.0,
        })
    }

    /// Request graceful shutdown.
    pub async fn shutdown(&self) {
        if self.tx.send(MindCommand::Shutdown).await.is_err() {
            tracing::warn!("AsyncMind actor has stopped — shutdown command dropped");
        }
    }

    /// Check if the actor task is still alive.
    pub fn is_alive(&self) -> bool {
        !self.tx.is_closed()
    }
}

/// Builder for spawning an async mind actor.
pub struct AsyncMind;

impl AsyncMind {
    /// Spawn a new async mind actor on the current tokio runtime.
    ///
    /// Returns a handle for sending commands and a `JoinHandle` for
    /// awaiting the actor task's completion.
    pub fn spawn(config: MindConfig) -> (AsyncMindHandle, tokio::task::JoinHandle<()>) {
        Self::spawn_with_capacity(config, 256)
    }

    /// Spawn with a custom command channel capacity.
    pub fn spawn_with_capacity(
        config: MindConfig,
        channel_capacity: usize,
    ) -> (AsyncMindHandle, tokio::task::JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(channel_capacity);
        let handle = AsyncMindHandle { tx };
        let join = tokio::spawn(Self::actor_loop(config, rx));
        (handle, join)
    }

    /// The actor loop — owns the `ContinuousMind` and processes commands.
    async fn actor_loop(config: MindConfig, mut rx: mpsc::Receiver<MindCommand>) {
        let mut mind = ContinuousMind::new(config);
        mind.activate();

        // Metabolic interval: baseline background processing (e.g. chronobiology)
        // Adjusts based on arousal and thermodynamic load in the future.
        let mut metabolism = tokio::time::interval(std::time::Duration::from_millis(100));
        metabolism.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                // metabolic baseline
                _ = metabolism.tick() => {
                    // Only tick if mind is active
                    if mind.state.is_active {
                        mind.tick();

                        // v1.3.0 CIRCADIAN HOMEOSTASIS
                        // Every 100 ticks, sync dimensionality with biorhythm
                        if mind.state.tick % 100 == 0 {
                            let phase = mind.state.biorhythm.as_ref().map(|b| b.phase).unwrap_or(crate::chronobiology::CircadianPhase::Day);

                            match phase {
                                crate::chronobiology::CircadianPhase::Night => {
                                    if mind.state.holocell.dimensionality != symthaea_core::hdc::HdcDimensionality::Rest {
                                        tracing::info!("CIRCADIAN RHYTHM: Night phase. Constricting to 2^13 (Rest)");
                                        mind.state.holocell.dilate(symthaea_core::hdc::HdcDimensionality::Rest);
                                    }
                                }
                                crate::chronobiology::CircadianPhase::Dawn => {
                                    if mind.state.holocell.dimensionality == symthaea_core::hdc::HdcDimensionality::Rest {
                                        tracing::info!("CIRCADIAN RHYTHM: Dawn phase. Awakening to 2^14 (Standard)");
                                        mind.state.holocell.dilate(symthaea_core::hdc::HdcDimensionality::Standard);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }

                // Command processing (Wake-up events)
                Some(cmd) = rx.recv() => {
                    match cmd {
                        MindCommand::Tick(resp) => {
                            let output = mind.tick();
                            let _ = resp.send(output);
                        }
                        MindCommand::Perceive(content) => {
                            mind.perceive(content);
                            mind.tick(); // Immediate wake-up on perception
                        }
                        MindCommand::PerceiveText(text, embedding) => {
                            mind.perceive_text(&text, embedding);
                            mind.tick();
                        }
                        MindCommand::Input(input) => {
                            mind.input(input);
                            mind.tick();
                        }
                        MindCommand::SetGoal(desc, embedding, priority) => {
                            mind.set_goal(desc, embedding, priority);
                        }
                        MindCommand::Activate => {
                            mind.activate();
                        }
                        MindCommand::Snapshot(resp) => {
                            let _ = resp.send(mind.snapshot());
                        }
                        MindCommand::Stats(resp) => {
                            let _ = resp.send(mind.stats().clone());
                        }
                        MindCommand::ReceiveSocial(msg) => {
                            mind.receive_social(msg);
                            // Only trigger a full tick if a certain surprise/phi threshold is reached
                            // in a future implementation. For now, we process immediately.
                            mind.tick();
                        }
                        MindCommand::SwarmGossip(msg) => {
                            match msg {
                                crate::swarm::SwarmMessage::ZkProof { mutation_id, proof_bytes, public_inputs } => {
                                    tracing::info!("Verifying ZK breakthrough: {}", mutation_id);
                                    // In production, we spawn a blocking task to verify the RISC Zero receipt
                                    // For now, we use the core verification logic or simulate it
                                    // If valid, we apply the mutation to mind
                                    mind.receive_swarm_message(crate::swarm::SwarmMessage::ZkProof {
                                        mutation_id, proof_bytes, public_inputs
                                    });
                                }
                                _ => {
                                    mind.receive_swarm_message(msg);
                                }
                            }
                            mind.tick();
                        }
                        MindCommand::UpdateThermodynamics(load) => {
                            mind.state.thermodynamic_load = load;
                            mind.state.mood_temperature = 0.5 + (load * 1.5);

                            // HOLOGRAPHIC DILATION
                            // Trigger dilation to 2^16 when load > 0.75 (Entering 20W regime)
                            // Retract to 2^14 when load < 0.4 (Returning to 6W baseline)
                            if load > 0.75 && mind.state.holocell.dimensionality != symthaea_core::hdc::HdcDimensionality::Ultra {
                                tracing::info!("THERMODYNAMIC SPIKE: Dilating Liquid Holocell to 2^16 (Ultra)");
                                mind.state.holocell.dilate(symthaea_core::hdc::HdcDimensionality::Ultra);
                            } else if load < 0.4 && mind.state.holocell.dimensionality != symthaea_core::hdc::HdcDimensionality::Standard {
                                tracing::info!("HOMEOTASIS REACHED: Constricting Liquid Holocell to 2^14 (Standard)");
                                mind.state.holocell.dilate(symthaea_core::hdc::HdcDimensionality::Standard);
                            }

                            // SYNC with LLM Backend (Sovereign Voice)
                            // If the backend is Candle, we push the physical affect into the neural weights
                            #[cfg(feature = "liquid-mamba")]
                            {
                                if let Some(ref backend) = mind.llm_backend {
                                    backend.update_affect(load, mind.state.mood_temperature);
                                }
                            }

                            let interval_ms = 100 + (load * 900.0) as u64; // 100ms to 1s
                            metabolism = tokio::time::interval(std::time::Duration::from_millis(interval_ms));
                            metabolism.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                        }
                        MindCommand::DrainSocialOutbox(resp) => {
                            let _ = resp.send(mind.drain_social_outbox());
                        }
                        MindCommand::SeedMemory(resp) => {
                            let _ = resp.send(mind.seed_memory());
                        }
                        MindCommand::Shutdown => {
                            mind.request_shutdown();
                            return; // Exit loop
                        }
                    }
                }
            }
        }
    }
}

/// Connect two async minds so that social messages flow between them.
///
/// Runs a relay loop that periodically drains each mind's social outbox
/// and forwards messages to the other. Returns a `JoinHandle` for the
/// relay task.
pub fn connect_social(
    mind_a: &AsyncMindHandle,
    mind_b: &AsyncMindHandle,
    interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    let a = mind_a.clone();
    let b = mind_b.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            if !a.is_alive() || !b.is_alive() {
                break;
            }
            // A → B
            let msgs_a = a.drain_social_outbox().await;
            for msg in msgs_a {
                b.receive_social(msg).await;
            }
            // B → A
            let msgs_b = b.drain_social_outbox().await;
            for msg in msgs_b {
                a.receive_social(msg).await;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_mind_spawn_and_tick() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        // Run a tick
        let output = handle.tick().await;
        // First tick with no input may or may not produce output
        let _ = output;

        // Check state — tick count may exceed 1 due to metabolic baseline timer
        let state = handle.snapshot().await;
        assert!(
            state.tick >= 1,
            "expected at least 1 tick, got {}",
            state.tick
        );
        assert!(state.is_active);

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_perceive_and_tick() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        let dim = 512;
        let hv = ContinuousHV::random(dim, 42);
        handle.perceive(hv).await;

        let _output = handle.tick().await;
        let stats = handle.stats().await;
        // perceive() triggers an immediate tick + explicit tick() + possible metabolic ticks
        assert!(
            stats.total_ticks >= 1,
            "expected at least 1 tick, got {}",
            stats.total_ticks
        );
        assert!(stats.inputs_processed >= 1);

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_multi_tick() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        for i in 0..5 {
            let hv = ContinuousHV::random(512, i);
            handle.perceive(hv).await;
            handle.tick().await;
        }

        let state = handle.snapshot().await;
        // Each perceive triggers an immediate tick + explicit tick = 10 minimum,
        // plus possible metabolic baseline ticks from the 100ms timer
        assert!(
            state.tick >= 5,
            "expected at least 5 ticks, got {}",
            state.tick
        );

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_clone_handle() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());
        let handle2 = handle.clone();

        // Both handles can interact
        handle.perceive(ContinuousHV::random(512, 1)).await;
        handle2.tick().await;

        let state = handle.snapshot().await;
        // perceive triggers immediate tick + explicit tick + possible metabolic ticks
        assert!(
            state.tick >= 1,
            "expected at least 1 tick, got {}",
            state.tick
        );

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_social_relay() {
        let (alice, alice_join) = AsyncMind::spawn(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });
        let (bob, bob_join) = AsyncMind::spawn(MindConfig {
            enable_social_coherence: true,
            ..Default::default()
        });

        // Connect them with a short relay interval
        let relay = connect_social(&alice, &bob, std::time::Duration::from_millis(10));

        // Alice perceives and ticks (may generate social output)
        alice.perceive(ContinuousHV::random(512, 100)).await;
        alice.tick().await;

        // Bob perceives and ticks
        bob.perceive(ContinuousHV::random(512, 200)).await;
        bob.tick().await;

        // Let the relay run
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Both should have ticked
        let alice_state = alice.snapshot().await;
        let bob_state = bob.snapshot().await;
        assert!(alice_state.tick >= 1);
        assert!(bob_state.tick >= 1);

        alice.shutdown().await;
        bob.shutdown().await;
        relay.abort();
        alice_join.await.unwrap();
        bob_join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_is_alive() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());
        assert!(handle.is_alive());

        handle.shutdown().await;
        join.await.unwrap();
        // Actor task has exited, channel should be closed
        assert!(!handle.is_alive());
    }

    #[tokio::test]
    async fn test_async_mind_set_goal() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        handle
            .set_goal("test goal".to_string(), ContinuousHV::random(512, 42), 0.8)
            .await;
        // Tick to process the goal input
        handle.tick().await;

        let state = handle.snapshot().await;
        // The goal should appear in active_goals after processing
        assert!(
            !state.active_goals.is_empty(),
            "goal should be active after tick"
        );

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_stats_defaults() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        let stats = handle.stats().await;
        // Stats should exist and have sensible defaults
        // (metabolic timer may have already fired, so total_ticks >= 0)
        assert!(stats.avg_consciousness >= 0.0);
        assert!(stats.peak_consciousness >= 0.0);

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_perceive_text() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        handle
            .perceive_text("hello world".to_string(), ContinuousHV::random(512, 42))
            .await;

        let stats = handle.stats().await;
        assert!(
            stats.inputs_processed >= 1,
            "perceive_text should process at least 1 input: got {}",
            stats.inputs_processed
        );

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_raw_input() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        let input = MindInput::new(
            super::super::InputType::Perception,
            ContinuousHV::random(512, 7),
        );
        handle.input(input).await;

        let stats = handle.stats().await;
        assert!(stats.inputs_processed >= 1, "raw input should be processed");

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_thermodynamics_update() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        handle.update_thermodynamics(0.85).await;
        // Give the actor a moment to process
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let state = handle.snapshot().await;
        // Thermodynamic load should be set
        let diff = (state.thermodynamic_load - 0.85).abs();
        assert!(
            diff < 0.01,
            "thermodynamic load should be ~0.85: got {}",
            state.thermodynamic_load
        );
        // Mood temperature should be 0.5 + (0.85 * 1.5) = 1.775
        let expected_mood = 0.5 + (0.85 * 1.5);
        let mood_diff = (state.mood_temperature - expected_mood).abs();
        assert!(
            mood_diff < 0.01,
            "mood_temperature should be ~{}: got {}",
            expected_mood,
            state.mood_temperature
        );

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_drain_social_outbox_empty() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        // Without social coherence enabled, outbox should be empty
        let msgs = handle.drain_social_outbox().await;
        assert!(msgs.is_empty(), "no social messages without social enabled");

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_spawn_with_capacity() {
        // Use a very small channel capacity to verify the API works
        let (handle, join) = AsyncMind::spawn_with_capacity(MindConfig::default(), 4);

        let output = handle.tick().await;
        let _ = output;
        assert!(handle.is_alive());

        handle.shutdown().await;
        join.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_mind_activate_command() {
        let (handle, join) = AsyncMind::spawn(MindConfig::default());

        // Mind starts activated by the actor loop
        handle.activate().await;
        let state = handle.snapshot().await;
        assert!(state.is_active, "mind should be active after activate()");

        handle.shutdown().await;
        join.await.unwrap();
    }
}
