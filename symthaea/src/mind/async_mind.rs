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
        if self.tx.send(MindCommand::SeedMemory(resp_tx)).await.is_err() {
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

        while let Some(cmd) = rx.recv().await {
            match cmd {
                MindCommand::Tick(resp) => {
                    let output = mind.tick();
                    let _ = resp.send(output);
                }
                MindCommand::Perceive(content) => {
                    mind.perceive(content);
                }
                MindCommand::PerceiveText(text, embedding) => {
                    mind.perceive_text(&text, embedding);
                }
                MindCommand::Input(input) => {
                    mind.input(input);
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
                }
                MindCommand::DrainSocialOutbox(resp) => {
                    let _ = resp.send(mind.drain_social_outbox());
                }
                MindCommand::SeedMemory(resp) => {
                    let _ = resp.send(mind.seed_memory());
                }
                MindCommand::Shutdown => {
                    mind.request_shutdown();
                    break;
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

        // Check state
        let state = handle.snapshot().await;
        assert_eq!(state.tick, 1);
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
        assert_eq!(stats.total_ticks, 1);
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
        assert_eq!(state.tick, 5);

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
        assert_eq!(state.tick, 1);

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
}
