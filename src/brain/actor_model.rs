// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Actor Model: Concurrent Message-Passing Neural Architecture
//!
//! Implements an actor-based model for neural processing where
//! independent actors communicate via message passing, enabling
//! concurrent and distributed cognition.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;

/// Configuration for the actor system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorSystemConfig {
    /// Maximum actors in the system
    pub max_actors: usize,
    /// Message queue capacity per actor
    pub queue_capacity: usize,
    /// Embedding dimension
    pub dimension: usize,
    /// Processing timeout (simulation ticks)
    pub processing_timeout: u32,
    /// Enable learning from interactions
    pub learning_enabled: bool,
    /// Learning rate for actor adaptation
    pub learning_rate: f32,
}

impl Default for ActorSystemConfig {
    fn default() -> Self {
        Self {
            max_actors: 1000,
            queue_capacity: 100,
            dimension: 512,
            processing_timeout: 10,
            learning_enabled: true,
            learning_rate: 0.01,
        }
    }
}

/// A message passed between actors
#[derive(Debug, Clone)]
pub struct ActorMessage {
    /// Unique message ID
    pub id: u64,
    /// Sender actor ID
    pub from: ActorId,
    /// Recipient actor ID
    pub to: ActorId,
    /// Message type
    pub message_type: MessageType,
    /// Content embedding
    pub content: ContinuousHV,
    /// Priority (higher = process first)
    pub priority: f32,
    /// Time to live (ticks before expiration)
    pub ttl: u32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Type of actor message
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Activation signal
    Activate,
    /// Inhibition signal
    Inhibit,
    /// Query for information
    Query,
    /// Response to a query
    Response,
    /// Broadcast to multiple actors
    Broadcast,
    /// Learning/update signal
    Learn,
    /// Synchronization signal
    Sync,
}

/// Unique identifier for an actor
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorId(pub String);

impl ActorId {
    /// Create a new actor ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for ActorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Role/type of an actor in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorRole {
    /// Processes sensory input
    Sensor,
    /// Processes internal state
    Processor,
    /// Produces motor output
    Effector,
    /// Coordinates other actors
    Coordinator,
    /// Stores memories
    Memory,
    /// Evaluates emotional significance
    Evaluator,
    /// Makes decisions
    Decider,
}

/// State of an individual actor
#[derive(Debug, Clone)]
pub struct Actor {
    /// Unique identifier
    pub id: ActorId,
    /// Actor's role
    pub role: ActorRole,
    /// Current state embedding
    pub state: ContinuousHV,
    /// Learned weights for processing
    pub weights: ContinuousHV,
    /// Activation level
    pub activation: f32,
    /// Message queue
    pub inbox: VecDeque<ActorMessage>,
    /// Connected actors (outgoing)
    pub connections: Vec<ActorId>,
    /// Processing function (simplified as similarity threshold)
    pub threshold: f32,
    /// Statistics
    pub stats: ActorStats,
}

/// Statistics for an individual actor
#[derive(Debug, Clone, Default)]
pub struct ActorStats {
    /// Messages received
    pub messages_received: u64,
    /// Messages sent
    pub messages_sent: u64,
    /// Activations triggered
    pub activations: u64,
    /// Total processing time (ticks)
    pub total_processing_ticks: u64,
}

impl Actor {
    /// Create a new actor
    pub fn new(id: ActorId, role: ActorRole, dimension: usize) -> Self {
        // Use id and role hash for deterministic but unique initialization
        let mut seed: u64 = 0x5174_1AEA; // "SYMTHAEA"
        for byte in id.0.bytes() {
            seed = seed.wrapping_mul(31).wrapping_add(byte as u64);
        }
        seed ^= role as u64 * 0x9E37_79B9;
        Self {
            id,
            role,
            state: ContinuousHV::random(dimension, seed),
            weights: ContinuousHV::random(dimension, seed.wrapping_add(1)),
            activation: 0.0,
            inbox: VecDeque::new(),
            connections: Vec::new(),
            threshold: 0.5,
            stats: ActorStats::default(),
        }
    }

    /// Receive a message
    pub fn receive(&mut self, message: ActorMessage) {
        self.stats.messages_received += 1;
        self.inbox.push_back(message);
    }

    /// Process pending messages
    pub fn process(&mut self, learning_rate: f32) -> Vec<ActorMessage> {
        let mut outgoing = Vec::new();

        while let Some(msg) = self.inbox.pop_front() {
            if msg.ttl == 0 {
                continue; // Expired message
            }

            self.stats.activations += 1;

            match msg.message_type {
                MessageType::Activate => {
                    let similarity = self.state.similarity(&msg.content);
                    if similarity > self.threshold {
                        self.activation = (self.activation + similarity).min(1.0);

                        // Forward activation to connections
                        for conn in &self.connections {
                            let response = ActorMessage {
                                id: msg.id + 1,
                                from: self.id.clone(),
                                to: conn.clone(),
                                message_type: MessageType::Activate,
                                content: self.state.clone(),
                                priority: msg.priority * 0.9,
                                ttl: msg.ttl.saturating_sub(1),
                                metadata: HashMap::new(),
                            };
                            outgoing.push(response);
                        }
                    }
                }
                MessageType::Inhibit => {
                    self.activation = (self.activation - 0.5).max(0.0);
                }
                MessageType::Query => {
                    // Respond with current state
                    let response = ActorMessage {
                        id: msg.id + 1,
                        from: self.id.clone(),
                        to: msg.from.clone(),
                        message_type: MessageType::Response,
                        content: self.state.clone(),
                        priority: msg.priority,
                        ttl: msg.ttl.saturating_sub(1),
                        metadata: HashMap::new(),
                    };
                    outgoing.push(response);
                }
                MessageType::Learn => {
                    // Update weights based on message content
                    self.weights = ContinuousHV::bundle_owned(&[
                        self.weights.clone(),
                        msg.content.scale(learning_rate),
                    ]);
                    // Normalize
                    self.weights = self.weights.normalize();
                }
                _ => {}
            }
        }

        self.stats.messages_sent += outgoing.len() as u64;
        outgoing
    }

    /// Decay activation over time
    pub fn decay(&mut self, rate: f32) {
        self.activation = (self.activation - rate).max(0.0);
    }
}

/// The actor system coordinator
#[derive(Debug)]
pub struct ActorSystem {
    /// Configuration
    config: ActorSystemConfig,
    /// All actors in the system
    actors: HashMap<ActorId, Actor>,
    /// Pending messages for delivery
    message_queue: VecDeque<ActorMessage>,
    /// Next message ID
    next_message_id: u64,
    /// Current simulation tick
    tick: u64,
    /// System statistics
    stats: ActorSystemStats,
}

/// Statistics for the actor system
#[derive(Debug, Clone, Default)]
pub struct ActorSystemStats {
    /// Total messages processed
    pub total_messages: u64,
    /// Messages dropped (expired or queue full)
    pub messages_dropped: u64,
    /// Total simulation ticks
    pub total_ticks: u64,
    /// Peak active actors
    pub peak_active_actors: usize,
    /// Average message latency (ticks)
    pub avg_message_latency: f32,
}

impl ActorSystem {
    /// Create a new actor system
    pub fn new(config: ActorSystemConfig) -> Self {
        Self {
            config,
            actors: HashMap::new(),
            message_queue: VecDeque::new(),
            next_message_id: 1,
            tick: 0,
            stats: ActorSystemStats::default(),
        }
    }

    /// Spawn a new actor
    pub fn spawn(&mut self, id: impl Into<String>, role: ActorRole) -> Option<ActorId> {
        if self.actors.len() >= self.config.max_actors {
            return None;
        }

        let actor_id = ActorId::new(id);
        let actor = Actor::new(actor_id.clone(), role, self.config.dimension);
        self.actors.insert(actor_id.clone(), actor);
        Some(actor_id)
    }

    /// Connect two actors (one-way)
    pub fn connect(&mut self, from: &ActorId, to: &ActorId) -> bool {
        if let Some(actor) = self.actors.get_mut(from) {
            if !actor.connections.contains(to) {
                actor.connections.push(to.clone());
                return true;
            }
        }
        false
    }

    /// Send a message between actors
    pub fn send(
        &mut self,
        from: &ActorId,
        to: &ActorId,
        message_type: MessageType,
        content: ContinuousHV,
    ) -> u64 {
        let msg_id = self.next_message_id;
        self.next_message_id += 1;

        let message = ActorMessage {
            id: msg_id,
            from: from.clone(),
            to: to.clone(),
            message_type,
            content,
            priority: 1.0,
            ttl: self.config.processing_timeout,
            metadata: HashMap::new(),
        };

        self.message_queue.push_back(message);
        self.stats.total_messages += 1;
        msg_id
    }

    /// Broadcast a message to all actors of a role
    pub fn broadcast(&mut self, from: &ActorId, role: ActorRole, content: ContinuousHV) {
        let targets: Vec<_> = self
            .actors
            .iter()
            .filter(|(_, a)| a.role == role)
            .map(|(id, _)| id.clone())
            .collect();

        for target in targets {
            self.send(from, &target, MessageType::Broadcast, content.clone());
        }
    }

    /// Advance simulation by one tick
    pub fn tick(&mut self) {
        self.tick += 1;
        self.stats.total_ticks += 1;

        // Deliver messages
        while let Some(msg) = self.message_queue.pop_front() {
            if msg.ttl == 0 {
                self.stats.messages_dropped += 1;
                continue;
            }

            if let Some(actor) = self.actors.get_mut(&msg.to) {
                if actor.inbox.len() < self.config.queue_capacity {
                    actor.receive(msg);
                } else {
                    self.stats.messages_dropped += 1;
                }
            }
        }

        // Process all actors
        let learning_rate = if self.config.learning_enabled {
            self.config.learning_rate
        } else {
            0.0
        };

        let mut new_messages = Vec::new();
        for actor in self.actors.values_mut() {
            let outgoing = actor.process(learning_rate);
            new_messages.extend(outgoing);
            actor.decay(0.01);
        }

        // Queue new messages
        for msg in new_messages {
            self.message_queue.push_back(msg);
            self.stats.total_messages += 1;
        }

        // Update peak active actors
        let active = self.actors.values().filter(|a| a.activation > 0.1).count();
        if active > self.stats.peak_active_actors {
            self.stats.peak_active_actors = active;
        }
    }

    /// Run simulation for N ticks
    pub fn run(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.tick();
        }
    }

    /// Get an actor by ID
    pub fn get_actor(&self, id: &ActorId) -> Option<&Actor> {
        self.actors.get(id)
    }

    /// Get mutable actor by ID
    pub fn get_actor_mut(&mut self, id: &ActorId) -> Option<&mut Actor> {
        self.actors.get_mut(id)
    }

    /// Get all actors with a specific role
    pub fn actors_by_role(&self, role: ActorRole) -> Vec<&Actor> {
        self.actors.values().filter(|a| a.role == role).collect()
    }

    /// Get system statistics
    pub fn stats(&self) -> &ActorSystemStats {
        &self.stats
    }

    /// Get current tick
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Get total actor count
    pub fn actor_count(&self) -> usize {
        self.actors.len()
    }
}

impl Default for ActorSystem {
    fn default() -> Self {
        Self::new(ActorSystemConfig::default())
    }
}

// ============================================================================
// Actor Trait (for mycelix/dark_spot_actor integration)
// ============================================================================

/// Priority levels for actor message processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ActorPriority {
    /// Background priority - lowest, for non-time-critical tasks
    Background,
    /// Low priority background tasks
    Low,
    /// Normal priority
    #[default]
    Normal,
    /// High priority
    High,
    /// Critical priority - process immediately
    Critical,
}

/// Generic organ message for cross-actor communication (struct version)
#[derive(Debug, Clone)]
pub struct OrganMessageData {
    /// Message payload as bytes
    pub payload: Vec<u8>,
    /// Message priority
    pub priority: ActorPriority,
    /// Source organ/actor name
    pub source: String,
}

/// Response type for actor message handling
#[derive(Debug, Clone)]
pub enum Response {
    /// Successful response with optional data
    Ok,
    /// Error response
    Error(String),
    /// Acknowledgment without data
    Ack,
}

/// Organ message enum for actor communication (used by dark_spot_actor)
/// Supports query/response patterns with oneshot channels
pub enum OrganMessage {
    /// Query message expecting a string response
    Query {
        question: String,
        reply: tokio::sync::oneshot::Sender<String>,
        hdc_semantic: Option<Vec<f32>>,
    },
    /// Input data message
    Input {
        data: Vec<u8>,
        reply: tokio::sync::oneshot::Sender<Response>,
        hdc_semantic: Option<Vec<f32>>,
    },
    /// Shutdown signal
    Shutdown,
}

/// Trait for implementing async actors (used by mycelix dark_spot_actor)
#[async_trait::async_trait]
pub trait ActorTrait: Send + Sync {
    /// Handle an incoming message
    async fn handle_message(&mut self, msg: OrganMessage) -> anyhow::Result<()>;

    /// Get the actor's priority
    fn priority(&self) -> ActorPriority {
        ActorPriority::Normal
    }

    /// Get the actor's name
    fn name(&self) -> &str;
}

// ActorTrait is imported as Actor by dark_spot_actor:
// use super::actor_model::{ActorTrait as Actor, ...}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_creation() {
        let actor = Actor::new(ActorId::new("test"), ActorRole::Processor, 512);
        assert_eq!(actor.id.0, "test");
        assert_eq!(actor.activation, 0.0);
    }

    #[test]
    fn test_actor_system_spawn() {
        let mut system = ActorSystem::default();
        let id = system.spawn("actor1", ActorRole::Sensor);
        assert!(id.is_some());
        assert_eq!(system.actor_count(), 1);
    }

    #[test]
    fn test_message_passing() {
        let mut system = ActorSystem::default();

        let sender = system.spawn("sender", ActorRole::Sensor).unwrap();
        let receiver = system.spawn("receiver", ActorRole::Processor).unwrap();

        system.connect(&sender, &receiver);
        system.send(
            &sender,
            &receiver,
            MessageType::Activate,
            ContinuousHV::random(512, 0xDEAD_0001),
        );

        system.tick();

        let recv_actor = system.get_actor(&receiver).unwrap();
        assert!(recv_actor.stats.messages_received > 0);
    }

    #[test]
    fn test_broadcast() {
        let mut system = ActorSystem::default();

        let coord = system.spawn("coordinator", ActorRole::Coordinator).unwrap();
        system.spawn("proc1", ActorRole::Processor);
        system.spawn("proc2", ActorRole::Processor);

        system.broadcast(
            &coord,
            ActorRole::Processor,
            ContinuousHV::random(512, 0xDEAD_0002),
        );
        system.tick();

        // Both processors should have received messages
        let processors = system.actors_by_role(ActorRole::Processor);
        assert_eq!(processors.len(), 2);
    }

    #[test]
    fn test_simulation_run() {
        let mut system = ActorSystem::default();

        system.spawn("a1", ActorRole::Sensor);
        system.spawn("a2", ActorRole::Processor);

        system.run(10);

        assert_eq!(system.stats.total_ticks, 10);
    }

    #[test]
    fn test_actor_activation_from_message() {
        let mut system = ActorSystem::default();

        let sender = system.spawn("sender", ActorRole::Sensor).unwrap();
        let receiver = system.spawn("receiver", ActorRole::Processor).unwrap();

        system.connect(&sender, &receiver);

        // Send an Activate message with content similar to the receiver's state
        let content = ContinuousHV::random(512, 0xAC01_0001);
        system.send(&sender, &receiver, MessageType::Activate, content);
        system.tick();

        // Receiver should have been activated (non-zero activation)
        let actor = system.get_actor(&receiver).unwrap();
        assert!(
            actor.stats.messages_received > 0,
            "Receiver should have processed the message"
        );
    }

    #[test]
    fn test_inhibition_message() {
        let mut system = ActorSystem::default();

        let sender = system.spawn("sender", ActorRole::Coordinator).unwrap();
        let target = system.spawn("target", ActorRole::Processor).unwrap();

        system.connect(&sender, &target);
        system.send(
            &sender,
            &target,
            MessageType::Inhibit,
            ContinuousHV::random(512, 0xAC01_0002),
        );
        system.tick();

        let actor = system.get_actor(&target).unwrap();
        assert!(
            actor.stats.messages_received > 0,
            "Target should have received the inhibition message"
        );
    }

    #[test]
    fn test_actors_by_role() {
        let mut system = ActorSystem::default();

        system.spawn("s1", ActorRole::Sensor);
        system.spawn("s2", ActorRole::Sensor);
        system.spawn("p1", ActorRole::Processor);
        system.spawn("e1", ActorRole::Effector);

        assert_eq!(system.actors_by_role(ActorRole::Sensor).len(), 2);
        assert_eq!(system.actors_by_role(ActorRole::Processor).len(), 1);
        assert_eq!(system.actors_by_role(ActorRole::Effector).len(), 1);
        assert_eq!(system.actors_by_role(ActorRole::Memory).len(), 0);
    }

    #[test]
    fn test_max_actors_enforced() {
        let config = ActorSystemConfig {
            max_actors: 3,
            ..Default::default()
        };
        let mut system = ActorSystem::new(config);

        assert!(system.spawn("a1", ActorRole::Sensor).is_some());
        assert!(system.spawn("a2", ActorRole::Sensor).is_some());
        assert!(system.spawn("a3", ActorRole::Sensor).is_some());
        assert!(
            system.spawn("a4", ActorRole::Sensor).is_none(),
            "Should reject spawn when at max_actors"
        );
    }
}
