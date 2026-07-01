// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Streaming: Real-Time State Broadcasting
//!
//! ## Revolutionary Features
//!
//! 1. **WebSocket Support**: Bidirectional real-time communication
//! 2. **SSE (Server-Sent Events)**: One-way streaming for dashboards
//! 3. **Pub/Sub Channels**: Internal message routing
//! 4. **Typed Events**: Strongly-typed consciousness event stream
//!
//! ## Architecture
//!
//! ```text
//! UnifiedConsciousBeing ─────────────────────────────────────────┐
//!        │                                                        │
//!        v                                                        v
//! ConsciousnessEventEmitter ──┬── WebSocket ──> External Clients
//!        │                    │
//!        │                    └── SSE ──────> Dashboard/Monitor
//!        v
//! Internal Subscribers (Persistence, Analytics, etc.)
//! ```
//!
//! ## Event Types
//!
//! - `PhiUpdate`: Integrated information changed
//! - `FlowStateChange`: Flow state transitioned
//! - `CognitiveModeTRansition`: Cognitive mode changed
//! - `DreamFragment`: New dream content generated
//! - `MemoryConsolidation`: Memory consolidation occurred
//! - `CausalInsight`: New causal relationship discovered

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use super::sleep_and_altered_states::DreamFragment;

// =============================================================================
// CONSCIOUSNESS EVENTS
// =============================================================================

/// A consciousness event that can be streamed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEvent {
    /// Unique event ID
    pub id: u64,
    /// Timestamp (Unix millis)
    pub timestamp: u64,
    /// Event type
    pub event_type: ConsciousnessEventType,
    /// Event payload (JSON-serializable data)
    pub payload: EventPayload,
}

/// Types of consciousness events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessEventType {
    /// Phi (integrated information) updated
    PhiUpdate,
    /// Flow state changed
    FlowStateChange,
    /// Cognitive mode transitioned
    CognitiveModeTransition,
    /// Dream fragment generated
    DreamFragment,
    /// Memory consolidated
    MemoryConsolidation,
    /// Causal insight discovered
    CausalInsight,
    /// Agent joined collective
    AgentJoined,
    /// Agent left collective
    AgentLeft,
    /// Sleep state changed
    SleepStateChange,
    /// Heartbeat (keepalive)
    Heartbeat,
}

/// Event payload variants
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum EventPayload {
    /// Phi update payload
    Phi {
        old_phi: f64,
        new_phi: f64,
        delta: f64,
    },
    /// Flow state payload
    Flow {
        old_state: f32,
        new_state: f32,
        trend: String, // "increasing", "decreasing", "stable"
    },
    /// Cognitive mode payload
    CognitiveMode {
        from_mode: String,
        to_mode: String,
        reason: String,
    },
    /// Dream fragment payload
    Dream {
        description: String,
        bizarreness: f64,
        valence: f64,
    },
    /// Memory consolidation payload
    Memory {
        consolidated_count: usize,
        pruned_count: usize,
    },
    /// Causal insight payload
    Causal {
        cause: String,
        effect: String,
        strength: f64,
    },
    /// Agent event payload
    Agent { agent_id: String, phi: f64 },
    /// Sleep state payload
    Sleep {
        state: String,
        consciousness_probability: f64,
    },
    /// Empty payload (for heartbeat)
    Empty,
}

impl ConsciousnessEvent {
    /// Create a new event with auto-generated ID and timestamp
    pub fn new(event_type: ConsciousnessEventType, payload: EventPayload) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            timestamp,
            event_type,
            payload,
        }
    }

    /// Create a phi update event
    pub fn phi_update(old_phi: f64, new_phi: f64) -> Self {
        Self::new(
            ConsciousnessEventType::PhiUpdate,
            EventPayload::Phi {
                old_phi,
                new_phi,
                delta: new_phi - old_phi,
            },
        )
    }

    /// Create a flow state change event
    pub fn flow_change(old_state: f32, new_state: f32) -> Self {
        let trend = if new_state > old_state + 0.05 {
            "increasing"
        } else if new_state < old_state - 0.05 {
            "decreasing"
        } else {
            "stable"
        };

        Self::new(
            ConsciousnessEventType::FlowStateChange,
            EventPayload::Flow {
                old_state,
                new_state,
                trend: trend.to_string(),
            },
        )
    }

    /// Create a cognitive mode transition event
    pub fn mode_transition(from: &str, to: &str, reason: &str) -> Self {
        Self::new(
            ConsciousnessEventType::CognitiveModeTransition,
            EventPayload::CognitiveMode {
                from_mode: from.to_string(),
                to_mode: to.to_string(),
                reason: reason.to_string(),
            },
        )
    }

    /// Create a dream fragment event
    pub fn dream_fragment(fragment: &DreamFragment) -> Self {
        Self::new(
            ConsciousnessEventType::DreamFragment,
            EventPayload::Dream {
                description: fragment.description.clone(),
                bizarreness: fragment.bizarreness,
                valence: fragment.valence,
            },
        )
    }

    /// Create a heartbeat event
    pub fn heartbeat() -> Self {
        Self::new(ConsciousnessEventType::Heartbeat, EventPayload::Empty)
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }

    /// Serialize for SSE (Server-Sent Events) format
    pub fn to_sse(&self) -> Result<String, String> {
        let json = self.to_json()?;
        Ok(format!(
            "event: {}\ndata: {}\nid: {}\n\n",
            self.event_type_str(),
            json,
            self.id
        ))
    }

    fn event_type_str(&self) -> &'static str {
        match self.event_type {
            ConsciousnessEventType::PhiUpdate => "phi",
            ConsciousnessEventType::FlowStateChange => "flow",
            ConsciousnessEventType::CognitiveModeTransition => "mode",
            ConsciousnessEventType::DreamFragment => "dream",
            ConsciousnessEventType::MemoryConsolidation => "memory",
            ConsciousnessEventType::CausalInsight => "causal",
            ConsciousnessEventType::AgentJoined => "agent_joined",
            ConsciousnessEventType::AgentLeft => "agent_left",
            ConsciousnessEventType::SleepStateChange => "sleep",
            ConsciousnessEventType::Heartbeat => "heartbeat",
        }
    }
}

// =============================================================================
// EVENT EMITTER (PUB/SUB)
// =============================================================================

/// Subscriber callback type
pub type SubscriberCallback = Box<dyn Fn(&ConsciousnessEvent) + Send + Sync>;

/// Subscription ID for managing subscriptions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

/// Event emitter for consciousness events
///
/// Provides pub/sub functionality for internal event routing
#[allow(clippy::type_complexity)]
pub struct ConsciousnessEventEmitter {
    /// Subscribers by event type
    subscribers:
        Arc<RwLock<HashMap<ConsciousnessEventType, Vec<(SubscriptionId, SubscriberCallback)>>>>,
    /// Global subscribers (receive all events)
    global_subscribers: Arc<RwLock<Vec<(SubscriptionId, SubscriberCallback)>>>,
    /// Next subscription ID
    next_sub_id: Arc<std::sync::atomic::AtomicU64>,
    /// Event history (ring buffer for replay)
    history: Arc<RwLock<std::collections::VecDeque<ConsciousnessEvent>>>,
    /// Max history size
    max_history: usize,
}

impl ConsciousnessEventEmitter {
    /// Create new emitter
    pub fn new(max_history: usize) -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            global_subscribers: Arc::new(RwLock::new(Vec::new())),
            next_sub_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            history: Arc::new(RwLock::new(std::collections::VecDeque::with_capacity(
                max_history,
            ))),
            max_history,
        }
    }

    /// Subscribe to specific event type
    pub fn subscribe<F>(&self, event_type: ConsciousnessEventType, callback: F) -> SubscriptionId
    where
        F: Fn(&ConsciousnessEvent) + Send + Sync + 'static,
    {
        let sub_id = SubscriptionId(
            self.next_sub_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        );

        let mut subs = self
            .subscribers
            .write()
            .expect("subscribers RwLock poisoned");
        subs.entry(event_type)
            .or_default()
            .push((sub_id, Box::new(callback)));

        sub_id
    }

    /// Subscribe to all events
    pub fn subscribe_all<F>(&self, callback: F) -> SubscriptionId
    where
        F: Fn(&ConsciousnessEvent) + Send + Sync + 'static,
    {
        let sub_id = SubscriptionId(
            self.next_sub_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        );

        let mut subs = self
            .global_subscribers
            .write()
            .expect("global_subscribers RwLock poisoned");
        subs.push((sub_id, Box::new(callback)));

        sub_id
    }

    /// Unsubscribe by ID
    pub fn unsubscribe(&self, sub_id: SubscriptionId) {
        // Remove from typed subscribers
        let mut subs = self
            .subscribers
            .write()
            .expect("subscribers RwLock poisoned");
        for (_, vec) in subs.iter_mut() {
            vec.retain(|(id, _)| *id != sub_id);
        }

        // Remove from global subscribers
        let mut global = self
            .global_subscribers
            .write()
            .expect("global_subscribers RwLock poisoned");
        global.retain(|(id, _)| *id != sub_id);
    }

    /// Emit an event
    pub fn emit(&self, event: ConsciousnessEvent) {
        // Add to history
        {
            let mut history = self.history.write().expect("history RwLock poisoned");
            history.push_back(event.clone());
            while history.len() > self.max_history {
                history.pop_front();
            }
        }

        // Notify typed subscribers
        {
            let subs = self
                .subscribers
                .read()
                .expect("subscribers RwLock poisoned");
            if let Some(callbacks) = subs.get(&event.event_type) {
                for (_, callback) in callbacks {
                    callback(&event);
                }
            }
        }

        // Notify global subscribers
        {
            let global = self
                .global_subscribers
                .read()
                .expect("global_subscribers RwLock poisoned");
            for (_, callback) in global.iter() {
                callback(&event);
            }
        }
    }

    /// Get recent events (for new subscriber catchup)
    pub fn get_history(&self, limit: usize) -> Vec<ConsciousnessEvent> {
        let history = self.history.read().expect("history RwLock poisoned");
        history.iter().rev().take(limit).cloned().collect()
    }
}

impl Default for ConsciousnessEventEmitter {
    fn default() -> Self {
        Self::new(1000)
    }
}

// =============================================================================
// STREAMING CHANNEL (for async/WebSocket)
// =============================================================================

/// A streaming channel for consciousness events
///
/// Use this to bridge to WebSocket/SSE handlers
pub struct ConsciousnessStream {
    /// Event receiver (for async consumption)
    receiver: std::sync::mpsc::Receiver<ConsciousnessEvent>,
    /// Sender (kept for cloning to emitter)
    sender: std::sync::mpsc::Sender<ConsciousnessEvent>,
    /// Subscription ID (for cleanup)
    sub_id: Option<SubscriptionId>,
}

impl ConsciousnessStream {
    /// Create a new stream connected to an emitter
    pub fn new(emitter: &ConsciousnessEventEmitter) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let sender_clone = sender.clone();

        let sub_id = emitter.subscribe_all(move |event| {
            let _ = sender_clone.send(event.clone());
        });

        Self {
            receiver,
            sender,
            sub_id: Some(sub_id),
        }
    }

    /// Try to receive next event (non-blocking)
    pub fn try_recv(&self) -> Option<ConsciousnessEvent> {
        self.receiver.try_recv().ok()
    }

    /// Receive next event (blocking)
    pub fn recv(&self) -> Option<ConsciousnessEvent> {
        self.receiver.recv().ok()
    }

    /// Receive with timeout
    pub fn recv_timeout(&self, timeout: std::time::Duration) -> Option<ConsciousnessEvent> {
        self.receiver.recv_timeout(timeout).ok()
    }

    /// Get all pending events (drains channel)
    pub fn drain(&self) -> Vec<ConsciousnessEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.receiver.try_recv() {
            events.push(event);
        }
        events
    }
}

// =============================================================================
// SSE FORMATTER
// =============================================================================

/// SSE (Server-Sent Events) stream formatter
pub struct SseFormatter {
    /// Last event ID sent
    last_id: u64,
    /// Retry interval (ms) - hint to client
    retry_ms: u32,
}

impl SseFormatter {
    /// Create new SSE formatter
    pub fn new(retry_ms: u32) -> Self {
        Self {
            last_id: 0,
            retry_ms,
        }
    }

    /// Format a single event for SSE
    pub fn format_event(&mut self, event: &ConsciousnessEvent) -> String {
        self.last_id = event.id;

        let json = serde_json::to_string(event).unwrap_or_default();

        format!(
            "event: {}\ndata: {}\nid: {}\n\n",
            event.event_type_str(),
            json,
            event.id
        )
    }

    /// Format the initial retry hint
    pub fn format_retry(&self) -> String {
        format!("retry: {}\n\n", self.retry_ms)
    }

    /// Format a comment (keepalive)
    pub fn format_keepalive(&self) -> String {
        ": keepalive\n\n".to_string()
    }
}

impl Default for SseFormatter {
    fn default() -> Self {
        Self::new(3000) // 3 second retry
    }
}

// =============================================================================
// WEBSOCKET MESSAGE TYPES
// =============================================================================

/// WebSocket message types for bidirectional communication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WebSocketMessage {
    /// Subscribe to specific event types
    Subscribe { event_types: Vec<String> },
    /// Unsubscribe from event types
    Unsubscribe { event_types: Vec<String> },
    /// Send input to consciousness system
    Input { text: String },
    /// Request consciousness state snapshot
    GetState,
    /// Consciousness event (server → client)
    Event(ConsciousnessEvent),
    /// State snapshot (server → client)
    State {
        phi: f64,
        flow_state: f32,
        cognitive_mode: String,
        memory_count: u64,
        causal_edges: usize,
    },
    /// Error message
    Error { message: String },
    /// Acknowledge message
    Ack { message_id: u64 },
}

impl WebSocketMessage {
    /// Parse from JSON
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = ConsciousnessEvent::phi_update(0.5, 0.6);

        assert_eq!(event.event_type, ConsciousnessEventType::PhiUpdate);
        assert!(event.id > 0);
        assert!(event.timestamp > 0);
    }

    #[test]
    fn test_sse_format() {
        let event = ConsciousnessEvent::heartbeat();
        let sse = event.to_sse().unwrap();

        assert!(sse.contains("event: heartbeat"));
        assert!(sse.contains("data: "));
        assert!(sse.contains("id: "));
    }

    #[test]
    fn test_emitter_subscribe() {
        let emitter = ConsciousnessEventEmitter::new(100);
        let received = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let received_clone = Arc::clone(&received);

        let _sub_id = emitter.subscribe(ConsciousnessEventType::PhiUpdate, move |_| {
            received_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        });

        emitter.emit(ConsciousnessEvent::phi_update(0.5, 0.6));

        assert!(received.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_websocket_message() {
        let msg = WebSocketMessage::Subscribe {
            event_types: vec!["phi".to_string(), "flow".to_string()],
        };

        let json = msg.to_json().unwrap();
        let parsed = WebSocketMessage::from_json(&json).unwrap();

        if let WebSocketMessage::Subscribe { event_types } = parsed {
            assert_eq!(event_types.len(), 2);
        } else {
            panic!("Wrong message type");
        }
    }
}
