//! Real-time WebSocket Updates for Proof Operations
//!
//! Provides live progress updates for long-running proof generation operations.
//!
//! ## Features
//!
//! - Live progress streaming for proof generation
//! - Event-based notifications for proof status changes
//! - Subscriber management for multiple clients
//! - Backpressure handling
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::realtime::{ProofEventBus, ProofEvent, Subscriber};
//!
//! // Create event bus
//! let bus = ProofEventBus::new(100); // 100 event buffer
//!
//! // Subscribe to events
//! let mut subscriber = bus.subscribe("client-1").await;
//!
//! // Listen for events
//! while let Some(event) = subscriber.recv().await {
//!     match event {
//!         ProofEvent::GenerationStarted { request_id, .. } => {
//!             println!("Started: {}", request_id);
//!         }
//!         ProofEvent::Progress { request_id, percent, .. } => {
//!             println!("Progress: {}%", percent);
//!         }
//!         ProofEvent::GenerationCompleted { request_id, .. } => {
//!             println!("Completed: {}", request_id);
//!         }
//!         _ => {}
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};

use crate::proofs::ProofType;

// ============================================================================
// Events
// ============================================================================

/// Proof lifecycle event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProofEvent {
    /// Generation started
    GenerationStarted {
        request_id: String,
        proof_type: ProofType,
        timestamp_ms: u64,
        estimated_time_ms: Option<f64>,
    },

    /// Progress update
    Progress {
        request_id: String,
        percent: f32,
        stage: String,
        message: Option<String>,
        timestamp_ms: u64,
    },

    /// Stage transition
    StageChanged {
        request_id: String,
        from_stage: String,
        to_stage: String,
        timestamp_ms: u64,
    },

    /// Generation completed successfully
    GenerationCompleted {
        request_id: String,
        proof_type: ProofType,
        size_bytes: usize,
        generation_time_ms: f64,
        timestamp_ms: u64,
    },

    /// Generation failed
    GenerationFailed {
        request_id: String,
        error: String,
        timestamp_ms: u64,
    },

    /// Verification started
    VerificationStarted {
        request_id: String,
        proof_type: ProofType,
        timestamp_ms: u64,
    },

    /// Verification completed
    VerificationCompleted {
        request_id: String,
        valid: bool,
        verification_time_ms: f64,
        timestamp_ms: u64,
    },

    /// Batch progress
    BatchProgress {
        batch_id: String,
        completed: usize,
        total: usize,
        failed: usize,
        timestamp_ms: u64,
    },

    /// Worker status change
    WorkerStatusChanged {
        worker_id: String,
        old_status: String,
        new_status: String,
        timestamp_ms: u64,
    },

    /// System metric update
    MetricUpdate {
        name: String,
        value: f64,
        labels: HashMap<String, String>,
        timestamp_ms: u64,
    },
}

impl ProofEvent {
    /// Get event timestamp
    pub fn timestamp_ms(&self) -> u64 {
        match self {
            ProofEvent::GenerationStarted { timestamp_ms, .. } => *timestamp_ms,
            ProofEvent::Progress { timestamp_ms, .. } => *timestamp_ms,
            ProofEvent::StageChanged { timestamp_ms, .. } => *timestamp_ms,
            ProofEvent::GenerationCompleted { timestamp_ms, .. } => *timestamp_ms,
            ProofEvent::GenerationFailed { timestamp_ms, .. } => *timestamp_ms,
            ProofEvent::VerificationStarted { timestamp_ms, .. } => *timestamp_ms,
            ProofEvent::VerificationCompleted { timestamp_ms, .. } => *timestamp_ms,
            ProofEvent::BatchProgress { timestamp_ms, .. } => *timestamp_ms,
            ProofEvent::WorkerStatusChanged { timestamp_ms, .. } => *timestamp_ms,
            ProofEvent::MetricUpdate { timestamp_ms, .. } => *timestamp_ms,
        }
    }

    /// Get request ID if applicable
    pub fn request_id(&self) -> Option<&str> {
        match self {
            ProofEvent::GenerationStarted { request_id, .. } => Some(request_id),
            ProofEvent::Progress { request_id, .. } => Some(request_id),
            ProofEvent::StageChanged { request_id, .. } => Some(request_id),
            ProofEvent::GenerationCompleted { request_id, .. } => Some(request_id),
            ProofEvent::GenerationFailed { request_id, .. } => Some(request_id),
            ProofEvent::VerificationStarted { request_id, .. } => Some(request_id),
            ProofEvent::VerificationCompleted { request_id, .. } => Some(request_id),
            ProofEvent::BatchProgress { batch_id, .. } => Some(batch_id),
            _ => None,
        }
    }

    /// Check if this is a terminal event
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ProofEvent::GenerationCompleted { .. }
                | ProofEvent::GenerationFailed { .. }
                | ProofEvent::VerificationCompleted { .. }
        )
    }
}

// ============================================================================
// Event Bus
// ============================================================================

/// Event bus configuration
#[derive(Debug, Clone)]
pub struct EventBusConfig {
    /// Channel buffer size
    pub buffer_size: usize,
    /// Maximum subscribers
    pub max_subscribers: usize,
    /// Event retention duration
    pub retention_secs: u64,
    /// Enable event persistence
    pub persist_events: bool,
}

impl Default for EventBusConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            max_subscribers: 100,
            retention_secs: 3600,
            persist_events: false,
        }
    }
}

/// Proof event bus for broadcasting events
pub struct ProofEventBus {
    config: EventBusConfig,
    sender: broadcast::Sender<ProofEvent>,
    subscribers: Arc<RwLock<HashMap<String, SubscriberInfo>>>,
    recent_events: Arc<RwLock<Vec<ProofEvent>>>,
}

/// Information about an event subscriber
struct SubscriberInfo {
    id: String,
    subscribed_at: Instant,
    filter: Option<EventFilter>,
    events_received: u64,
}

/// Event filter for subscribers
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// Filter by request ID
    pub request_ids: Option<Vec<String>>,
    /// Filter by proof type
    pub proof_types: Option<Vec<ProofType>>,
    /// Filter by event type
    pub event_types: Option<Vec<String>>,
    /// Include only terminal events
    pub terminal_only: bool,
}

impl EventFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn for_request(request_id: &str) -> Self {
        Self {
            request_ids: Some(vec![request_id.to_string()]),
            ..Default::default()
        }
    }

    pub fn for_proof_type(proof_type: ProofType) -> Self {
        Self {
            proof_types: Some(vec![proof_type]),
            ..Default::default()
        }
    }

    pub fn matches(&self, event: &ProofEvent) -> bool {
        // Check request ID filter
        if let Some(ref ids) = self.request_ids {
            if let Some(event_id) = event.request_id() {
                if !ids.iter().any(|id| id == event_id) {
                    return false;
                }
            }
        }

        // Check terminal only
        if self.terminal_only && !event.is_terminal() {
            return false;
        }

        true
    }
}

impl ProofEventBus {
    /// Create a new event bus
    pub fn new(buffer_size: usize) -> Self {
        let (sender, _) = broadcast::channel(buffer_size);
        Self {
            config: EventBusConfig {
                buffer_size,
                ..Default::default()
            },
            sender,
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            recent_events: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create with configuration
    pub fn with_config(config: EventBusConfig) -> Self {
        let (sender, _) = broadcast::channel(config.buffer_size);
        Self {
            config,
            sender,
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            recent_events: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Publish an event
    pub async fn publish(&self, event: ProofEvent) {
        // Store in recent events
        {
            let mut recent = self.recent_events.write().await;
            recent.push(event.clone());

            // Trim old events
            let cutoff = current_timestamp_ms() - (self.config.retention_secs * 1000);
            recent.retain(|e| e.timestamp_ms() >= cutoff);
        }

        // Broadcast to subscribers
        let _ = self.sender.send(event);
    }

    /// Subscribe to events
    pub async fn subscribe(&self, subscriber_id: &str) -> Subscriber {
        self.subscribe_with_filter(subscriber_id, None).await
    }

    /// Subscribe with filter
    pub async fn subscribe_with_filter(
        &self,
        subscriber_id: &str,
        filter: Option<EventFilter>,
    ) -> Subscriber {
        let receiver = self.sender.subscribe();

        let info = SubscriberInfo {
            id: subscriber_id.to_string(),
            subscribed_at: Instant::now(),
            filter: filter.clone(),
            events_received: 0,
        };

        self.subscribers
            .write()
            .await
            .insert(subscriber_id.to_string(), info);

        Subscriber {
            id: subscriber_id.to_string(),
            receiver,
            filter,
        }
    }

    /// Unsubscribe
    pub async fn unsubscribe(&self, subscriber_id: &str) {
        self.subscribers.write().await.remove(subscriber_id);
    }

    /// Get subscriber count
    pub async fn subscriber_count(&self) -> usize {
        self.subscribers.read().await.len()
    }

    /// Get recent events
    pub async fn recent_events(&self, limit: usize) -> Vec<ProofEvent> {
        let recent = self.recent_events.read().await;
        recent.iter().rev().take(limit).cloned().collect()
    }

    /// Get recent events for request
    pub async fn events_for_request(&self, request_id: &str) -> Vec<ProofEvent> {
        let recent = self.recent_events.read().await;
        recent
            .iter()
            .filter(|e| e.request_id() == Some(request_id))
            .cloned()
            .collect()
    }

    /// Get bus statistics
    pub async fn stats(&self) -> EventBusStats {
        let recent = self.recent_events.read().await;
        let subscribers = self.subscribers.read().await;

        EventBusStats {
            subscriber_count: subscribers.len(),
            recent_event_count: recent.len(),
            buffer_size: self.config.buffer_size,
        }
    }
}

/// Event bus statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBusStats {
    pub subscriber_count: usize,
    pub recent_event_count: usize,
    pub buffer_size: usize,
}

/// Event subscriber
pub struct Subscriber {
    id: String,
    receiver: broadcast::Receiver<ProofEvent>,
    filter: Option<EventFilter>,
}

impl Subscriber {
    /// Receive next event
    pub async fn recv(&mut self) -> Option<ProofEvent> {
        loop {
            match self.receiver.recv().await {
                Ok(event) => {
                    if let Some(ref filter) = self.filter {
                        if filter.matches(&event) {
                            return Some(event);
                        }
                    } else {
                        return Some(event);
                    }
                }
                Err(broadcast::error::RecvError::Closed) => return None,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
            }
        }
    }

    /// Try receive without blocking
    pub fn try_recv(&mut self) -> Option<ProofEvent> {
        loop {
            match self.receiver.try_recv() {
                Ok(event) => {
                    if let Some(ref filter) = self.filter {
                        if filter.matches(&event) {
                            return Some(event);
                        }
                    } else {
                        return Some(event);
                    }
                }
                Err(_) => return None,
            }
        }
    }

    /// Get subscriber ID
    pub fn id(&self) -> &str {
        &self.id
    }
}

// ============================================================================
// Progress Tracker
// ============================================================================

/// Progress tracker for a single proof operation
pub struct ProgressTracker {
    request_id: String,
    proof_type: ProofType,
    bus: Arc<ProofEventBus>,
    start_time: Instant,
    current_stage: String,
    current_percent: f32,
    stages: Vec<(String, f32)>, // (stage_name, weight)
    completed_stages: usize,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(request_id: &str, proof_type: ProofType, bus: Arc<ProofEventBus>) -> Self {
        Self {
            request_id: request_id.to_string(),
            proof_type,
            bus,
            start_time: Instant::now(),
            current_stage: "initializing".to_string(),
            current_percent: 0.0,
            stages: Vec::new(),
            completed_stages: 0,
        }
    }

    /// Define stages with weights
    pub fn with_stages(mut self, stages: Vec<(&str, f32)>) -> Self {
        self.stages = stages
            .into_iter()
            .map(|(name, weight)| (name.to_string(), weight))
            .collect();
        self
    }

    /// Start tracking
    pub async fn start(&self, estimated_time_ms: Option<f64>) {
        self.bus
            .publish(ProofEvent::GenerationStarted {
                request_id: self.request_id.clone(),
                proof_type: self.proof_type,
                timestamp_ms: current_timestamp_ms(),
                estimated_time_ms,
            })
            .await;
    }

    /// Update progress
    pub async fn update(&mut self, percent: f32, message: Option<&str>) {
        self.current_percent = percent;
        self.bus
            .publish(ProofEvent::Progress {
                request_id: self.request_id.clone(),
                percent,
                stage: self.current_stage.clone(),
                message: message.map(|s| s.to_string()),
                timestamp_ms: current_timestamp_ms(),
            })
            .await;
    }

    /// Transition to next stage
    pub async fn next_stage(&mut self, stage: &str) {
        let from_stage = self.current_stage.clone();
        self.current_stage = stage.to_string();
        self.completed_stages += 1;

        // Calculate percent based on stages
        if !self.stages.is_empty() {
            let total_weight: f32 = self.stages.iter().map(|(_, w)| w).sum();
            let completed_weight: f32 = self
                .stages
                .iter()
                .take(self.completed_stages)
                .map(|(_, w)| w)
                .sum();
            self.current_percent = (completed_weight / total_weight) * 100.0;
        }

        self.bus
            .publish(ProofEvent::StageChanged {
                request_id: self.request_id.clone(),
                from_stage,
                to_stage: stage.to_string(),
                timestamp_ms: current_timestamp_ms(),
            })
            .await;
    }

    /// Complete successfully
    pub async fn complete(self, proof_bytes: &[u8]) {
        let generation_time_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        self.bus
            .publish(ProofEvent::GenerationCompleted {
                request_id: self.request_id,
                proof_type: self.proof_type,
                size_bytes: proof_bytes.len(),
                generation_time_ms,
                timestamp_ms: current_timestamp_ms(),
            })
            .await;
    }

    /// Fail with error
    pub async fn fail(self, error: &str) {
        self.bus
            .publish(ProofEvent::GenerationFailed {
                request_id: self.request_id,
                error: error.to_string(),
                timestamp_ms: current_timestamp_ms(),
            })
            .await;
    }
}

// ============================================================================
// Batch Progress Tracker
// ============================================================================

/// Batch progress tracker
pub struct BatchProgressTracker {
    batch_id: String,
    bus: Arc<ProofEventBus>,
    total: usize,
    completed: usize,
    failed: usize,
}

impl BatchProgressTracker {
    /// Create new batch tracker
    pub fn new(batch_id: &str, total: usize, bus: Arc<ProofEventBus>) -> Self {
        Self {
            batch_id: batch_id.to_string(),
            bus,
            total,
            completed: 0,
            failed: 0,
        }
    }

    /// Record item completion
    pub async fn item_completed(&mut self, success: bool) {
        if success {
            self.completed += 1;
        } else {
            self.failed += 1;
        }

        self.bus
            .publish(ProofEvent::BatchProgress {
                batch_id: self.batch_id.clone(),
                completed: self.completed,
                total: self.total,
                failed: self.failed,
                timestamp_ms: current_timestamp_ms(),
            })
            .await;
    }

    /// Get progress percentage
    pub fn percent(&self) -> f32 {
        if self.total == 0 {
            return 100.0;
        }
        ((self.completed + self.failed) as f32 / self.total as f32) * 100.0
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        self.completed + self.failed >= self.total
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_event_filter_default() {
        let filter = EventFilter::default();
        assert!(!filter.terminal_only);
    }

    #[test]
    fn test_event_filter_for_request() {
        let filter = EventFilter::for_request("req-123");
        assert!(filter.request_ids.is_some());
    }

    #[test]
    fn test_event_is_terminal() {
        let completed = ProofEvent::GenerationCompleted {
            request_id: "req-1".to_string(),
            proof_type: ProofType::Range,
            size_bytes: 1000,
            generation_time_ms: 50.0,
            timestamp_ms: 0,
        };
        assert!(completed.is_terminal());

        let progress = ProofEvent::Progress {
            request_id: "req-1".to_string(),
            percent: 50.0,
            stage: "test".to_string(),
            message: None,
            timestamp_ms: 0,
        };
        assert!(!progress.is_terminal());
    }

    #[test]
    fn test_event_filter_matches() {
        let filter = EventFilter::for_request("req-123");

        let matching = ProofEvent::Progress {
            request_id: "req-123".to_string(),
            percent: 50.0,
            stage: "test".to_string(),
            message: None,
            timestamp_ms: 0,
        };
        assert!(filter.matches(&matching));

        let non_matching = ProofEvent::Progress {
            request_id: "req-456".to_string(),
            percent: 50.0,
            stage: "test".to_string(),
            message: None,
            timestamp_ms: 0,
        };
        assert!(!filter.matches(&non_matching));
    }

    #[tokio::test]
    async fn test_event_bus_publish_subscribe() {
        let bus = ProofEventBus::new(100);

        let mut subscriber = bus.subscribe("test-client").await;

        // Publish event
        bus.publish(ProofEvent::GenerationStarted {
            request_id: "req-1".to_string(),
            proof_type: ProofType::Range,
            timestamp_ms: current_timestamp_ms(),
            estimated_time_ms: Some(100.0),
        })
        .await;

        // Receive event
        let event = tokio::time::timeout(Duration::from_millis(100), subscriber.recv())
            .await
            .unwrap();

        assert!(event.is_some());
        assert_eq!(event.unwrap().request_id(), Some("req-1"));
    }

    #[tokio::test]
    async fn test_event_bus_recent_events() {
        let bus = ProofEventBus::new(100);

        bus.publish(ProofEvent::Progress {
            request_id: "req-1".to_string(),
            percent: 25.0,
            stage: "stage1".to_string(),
            message: None,
            timestamp_ms: current_timestamp_ms(),
        })
        .await;

        bus.publish(ProofEvent::Progress {
            request_id: "req-1".to_string(),
            percent: 50.0,
            stage: "stage2".to_string(),
            message: None,
            timestamp_ms: current_timestamp_ms(),
        })
        .await;

        let recent = bus.recent_events(10).await;
        assert_eq!(recent.len(), 2);
    }

    #[tokio::test]
    async fn test_event_bus_stats() {
        let bus = ProofEventBus::new(100);

        bus.subscribe("client-1").await;
        bus.subscribe("client-2").await;

        let stats = bus.stats().await;
        assert_eq!(stats.subscriber_count, 2);
    }

    #[tokio::test]
    async fn test_progress_tracker() {
        let bus = Arc::new(ProofEventBus::new(100));
        let mut subscriber = bus.subscribe("test").await;

        let mut tracker = ProgressTracker::new("req-1", ProofType::Range, bus.clone());
        tracker.start(Some(100.0)).await;
        tracker.update(50.0, Some("halfway")).await;

        // Receive events
        let event1 = tokio::time::timeout(Duration::from_millis(100), subscriber.recv())
            .await
            .unwrap()
            .unwrap();

        assert!(matches!(event1, ProofEvent::GenerationStarted { .. }));

        let event2 = tokio::time::timeout(Duration::from_millis(100), subscriber.recv())
            .await
            .unwrap()
            .unwrap();

        assert!(matches!(event2, ProofEvent::Progress { percent, .. } if percent == 50.0));
    }

    #[tokio::test]
    async fn test_batch_progress_tracker() {
        let bus = Arc::new(ProofEventBus::new(100));
        let mut tracker = BatchProgressTracker::new("batch-1", 10, bus);

        tracker.item_completed(true).await;
        tracker.item_completed(true).await;
        tracker.item_completed(false).await;

        assert!((tracker.percent() - 30.0).abs() < 0.001);
        assert!(!tracker.is_complete());
    }

    #[test]
    fn test_current_timestamp() {
        let ts = current_timestamp_ms();
        assert!(ts > 1700000000000); // After 2023
    }
}
