//! Holochain DHT signal listener.
//!
//! Handles real-time signals from the Holochain DHT for:
//! - New gradient submissions
//! - Reputation updates
//! - Round events
//! - Byzantine behavior detection

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use super::types::*;

/// Callback type for signal handlers.
pub type SignalHandler = Box<dyn Fn(HolochainSignal) + Send + Sync>;

/// Handler ID for unregistering.
pub type HandlerId = u64;

/// Signal listener that processes DHT signals.
pub struct HolochainListener {
    /// Signal receiver channel.
    signal_rx: mpsc::Receiver<HolochainSignal>,
    /// Registered handlers by type.
    handlers: Arc<RwLock<SignalHandlers>>,
    /// Running flag.
    running: Arc<RwLock<bool>>,
    /// Handler ID counter.
    next_handler_id: Arc<RwLock<u64>>,
}

/// Internal handler storage.
struct SignalHandlers {
    gradient_handlers: HashMap<HandlerId, SignalHandler>,
    reputation_handlers: HashMap<HandlerId, SignalHandler>,
    round_handlers: HashMap<HandlerId, SignalHandler>,
    byzantine_handlers: HashMap<HandlerId, SignalHandler>,
    ethereum_payment_handlers: HashMap<HandlerId, SignalHandler>,
    ethereum_anchor_handlers: HashMap<HandlerId, SignalHandler>,
    all_handlers: HashMap<HandlerId, SignalHandler>,
}

impl Default for SignalHandlers {
    fn default() -> Self {
        Self {
            gradient_handlers: HashMap::new(),
            reputation_handlers: HashMap::new(),
            round_handlers: HashMap::new(),
            byzantine_handlers: HashMap::new(),
            ethereum_payment_handlers: HashMap::new(),
            ethereum_anchor_handlers: HashMap::new(),
            all_handlers: HashMap::new(),
        }
    }
}

impl HolochainListener {
    /// Create a new signal listener.
    pub fn new(signal_rx: mpsc::Receiver<HolochainSignal>) -> Self {
        Self {
            signal_rx,
            handlers: Arc::new(RwLock::new(SignalHandlers::default())),
            running: Arc::new(RwLock::new(false)),
            next_handler_id: Arc::new(RwLock::new(1)),
        }
    }

    /// Create signal channel pair.
    /// Returns (sender for client, receiver for listener).
    pub fn channel(buffer: usize) -> (mpsc::Sender<HolochainSignal>, mpsc::Receiver<HolochainSignal>) {
        mpsc::channel(buffer)
    }

    /// Register handler for gradient submission signals.
    pub async fn on_gradient_submitted<F>(&self, handler: F) -> HandlerId
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        let id = self.next_id().await;
        self.handlers
            .write()
            .await
            .gradient_handlers
            .insert(id, Box::new(handler));
        id
    }

    /// Register handler for reputation update signals.
    pub async fn on_reputation_updated<F>(&self, handler: F) -> HandlerId
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        let id = self.next_id().await;
        self.handlers
            .write()
            .await
            .reputation_handlers
            .insert(id, Box::new(handler));
        id
    }

    /// Register handler for round event signals.
    pub async fn on_round_event<F>(&self, handler: F) -> HandlerId
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        let id = self.next_id().await;
        self.handlers
            .write()
            .await
            .round_handlers
            .insert(id, Box::new(handler));
        id
    }

    /// Register handler for Byzantine detection signals.
    pub async fn on_byzantine_detected<F>(&self, handler: F) -> HandlerId
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        let id = self.next_id().await;
        self.handlers
            .write()
            .await
            .byzantine_handlers
            .insert(id, Box::new(handler));
        id
    }

    /// Register handler for Ethereum payment request signals.
    pub async fn on_ethereum_payment_request<F>(&self, handler: F) -> HandlerId
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        let id = self.next_id().await;
        self.handlers
            .write()
            .await
            .ethereum_payment_handlers
            .insert(id, Box::new(handler));
        id
    }

    /// Register handler for Ethereum anchor request signals.
    pub async fn on_ethereum_anchor_request<F>(&self, handler: F) -> HandlerId
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        let id = self.next_id().await;
        self.handlers
            .write()
            .await
            .ethereum_anchor_handlers
            .insert(id, Box::new(handler));
        id
    }

    /// Register handler for all signals.
    pub async fn on_any_signal<F>(&self, handler: F) -> HandlerId
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        let id = self.next_id().await;
        self.handlers
            .write()
            .await
            .all_handlers
            .insert(id, Box::new(handler));
        id
    }

    /// Unregister a handler by ID.
    pub async fn unregister(&self, handler_id: HandlerId) -> bool {
        let mut handlers = self.handlers.write().await;

        handlers.gradient_handlers.remove(&handler_id).is_some()
            || handlers.reputation_handlers.remove(&handler_id).is_some()
            || handlers.round_handlers.remove(&handler_id).is_some()
            || handlers.byzantine_handlers.remove(&handler_id).is_some()
            || handlers.ethereum_payment_handlers.remove(&handler_id).is_some()
            || handlers.ethereum_anchor_handlers.remove(&handler_id).is_some()
            || handlers.all_handlers.remove(&handler_id).is_some()
    }

    /// Get next handler ID.
    async fn next_id(&self) -> HandlerId {
        let mut id = self.next_handler_id.write().await;
        let current = *id;
        *id += 1;
        current
    }

    /// Check if listener is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Stop the listener.
    pub async fn stop(&self) {
        *self.running.write().await = false;
    }

    /// Start listening for signals (blocking).
    /// Call this in a spawned task.
    pub async fn listen(mut self) {
        *self.running.write().await = true;
        info!("Starting Holochain signal listener");

        while *self.running.read().await {
            match self.signal_rx.recv().await {
                Some(signal) => {
                    debug!(?signal, "Received DHT signal");
                    self.dispatch_signal(signal).await;
                }
                None => {
                    warn!("Signal channel closed");
                    break;
                }
            }
        }

        info!("Holochain signal listener stopped");
    }

    /// Dispatch signal to appropriate handlers.
    ///
    /// This method is public to allow testing of signal handling
    /// without running the full listener loop.
    pub async fn dispatch_signal(&self, signal: HolochainSignal) {
        let handlers = self.handlers.read().await;

        // Call specific handlers
        let specific_handlers: &HashMap<HandlerId, SignalHandler> = match &signal {
            HolochainSignal::GradientSubmitted { .. } => &handlers.gradient_handlers,
            HolochainSignal::ReputationUpdated { .. } => &handlers.reputation_handlers,
            HolochainSignal::RoundEvent { .. } => &handlers.round_handlers,
            HolochainSignal::ByzantineDetected { .. } => &handlers.byzantine_handlers,
            HolochainSignal::EthereumPaymentRequest { .. } => &handlers.ethereum_payment_handlers,
            HolochainSignal::EthereumAnchorRequest { .. } => &handlers.ethereum_anchor_handlers,
        };

        for handler in specific_handlers.values() {
            handler(signal.clone());
        }

        // Call global handlers
        for handler in handlers.all_handlers.values() {
            handler(signal.clone());
        }
    }
}

/// Builder for creating listener with pre-configured handlers.
pub struct ListenerBuilder {
    signal_rx: mpsc::Receiver<HolochainSignal>,
    gradient_handlers: Vec<SignalHandler>,
    reputation_handlers: Vec<SignalHandler>,
    round_handlers: Vec<SignalHandler>,
    byzantine_handlers: Vec<SignalHandler>,
    ethereum_payment_handlers: Vec<SignalHandler>,
    ethereum_anchor_handlers: Vec<SignalHandler>,
    all_handlers: Vec<SignalHandler>,
}

impl ListenerBuilder {
    /// Create a new builder.
    pub fn new(signal_rx: mpsc::Receiver<HolochainSignal>) -> Self {
        Self {
            signal_rx,
            gradient_handlers: Vec::new(),
            reputation_handlers: Vec::new(),
            round_handlers: Vec::new(),
            byzantine_handlers: Vec::new(),
            ethereum_payment_handlers: Vec::new(),
            ethereum_anchor_handlers: Vec::new(),
            all_handlers: Vec::new(),
        }
    }

    /// Add gradient submission handler.
    pub fn on_gradient_submitted<F>(mut self, handler: F) -> Self
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        self.gradient_handlers.push(Box::new(handler));
        self
    }

    /// Add reputation update handler.
    pub fn on_reputation_updated<F>(mut self, handler: F) -> Self
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        self.reputation_handlers.push(Box::new(handler));
        self
    }

    /// Add round event handler.
    pub fn on_round_event<F>(mut self, handler: F) -> Self
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        self.round_handlers.push(Box::new(handler));
        self
    }

    /// Add Byzantine detection handler.
    pub fn on_byzantine_detected<F>(mut self, handler: F) -> Self
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        self.byzantine_handlers.push(Box::new(handler));
        self
    }

    /// Add Ethereum payment request handler.
    pub fn on_ethereum_payment_request<F>(mut self, handler: F) -> Self
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        self.ethereum_payment_handlers.push(Box::new(handler));
        self
    }

    /// Add Ethereum anchor request handler.
    pub fn on_ethereum_anchor_request<F>(mut self, handler: F) -> Self
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        self.ethereum_anchor_handlers.push(Box::new(handler));
        self
    }

    /// Add handler for all signals.
    pub fn on_any_signal<F>(mut self, handler: F) -> Self
    where
        F: Fn(HolochainSignal) + Send + Sync + 'static,
    {
        self.all_handlers.push(Box::new(handler));
        self
    }

    /// Build the listener.
    pub async fn build(self) -> HolochainListener {
        let listener = HolochainListener::new(self.signal_rx);

        // Register pre-configured handlers
        for handler in self.gradient_handlers {
            let id = listener.next_id().await;
            listener
                .handlers
                .write()
                .await
                .gradient_handlers
                .insert(id, handler);
        }

        for handler in self.reputation_handlers {
            let id = listener.next_id().await;
            listener
                .handlers
                .write()
                .await
                .reputation_handlers
                .insert(id, handler);
        }

        for handler in self.round_handlers {
            let id = listener.next_id().await;
            listener
                .handlers
                .write()
                .await
                .round_handlers
                .insert(id, handler);
        }

        for handler in self.byzantine_handlers {
            let id = listener.next_id().await;
            listener
                .handlers
                .write()
                .await
                .byzantine_handlers
                .insert(id, handler);
        }

        for handler in self.ethereum_payment_handlers {
            let id = listener.next_id().await;
            listener
                .handlers
                .write()
                .await
                .ethereum_payment_handlers
                .insert(id, handler);
        }

        for handler in self.ethereum_anchor_handlers {
            let id = listener.next_id().await;
            listener
                .handlers
                .write()
                .await
                .ethereum_anchor_handlers
                .insert(id, handler);
        }

        for handler in self.all_handlers {
            let id = listener.next_id().await;
            listener
                .handlers
                .write()
                .await
                .all_handlers
                .insert(id, handler);
        }

        listener
    }
}

/// Aggregated signal statistics.
#[derive(Clone, Debug, Default)]
pub struct SignalStats {
    pub gradients_received: u64,
    pub reputation_updates: u64,
    pub round_events: u64,
    pub byzantine_detections: u64,
    pub ethereum_payment_requests: u64,
    pub ethereum_anchor_requests: u64,
    pub total_signals: u64,
}

impl SignalStats {
    /// Update stats from a signal.
    pub fn record(&mut self, signal: &HolochainSignal) {
        self.total_signals += 1;
        match signal {
            HolochainSignal::GradientSubmitted { .. } => self.gradients_received += 1,
            HolochainSignal::ReputationUpdated { .. } => self.reputation_updates += 1,
            HolochainSignal::RoundEvent { .. } => self.round_events += 1,
            HolochainSignal::ByzantineDetected { .. } => self.byzantine_detections += 1,
            HolochainSignal::EthereumPaymentRequest { .. } => self.ethereum_payment_requests += 1,
            HolochainSignal::EthereumAnchorRequest { .. } => self.ethereum_anchor_requests += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[tokio::test]
    async fn test_channel_creation() {
        let (tx, rx) = HolochainListener::channel(16);
        drop(rx);

        // Send should fail after rx is dropped
        let signal = HolochainSignal::GradientSubmitted {
            node_id: "test".to_string(),
            round_num: 1,
            entry_hash: "hash".to_string(),
        };
        assert!(tx.send(signal).await.is_err());
    }

    #[tokio::test]
    async fn test_handler_registration() {
        let (_tx, rx) = HolochainListener::channel(16);
        let listener = HolochainListener::new(rx);

        let id1 = listener.on_gradient_submitted(|_| {}).await;
        let id2 = listener.on_reputation_updated(|_| {}).await;

        assert_ne!(id1, id2);
        assert!(listener.unregister(id1).await);
        assert!(!listener.unregister(id1).await); // Already removed
        assert!(listener.unregister(id2).await);
    }

    #[tokio::test]
    async fn test_signal_dispatch() {
        let (tx, rx) = HolochainListener::channel(16);
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        let listener = ListenerBuilder::new(rx)
            .on_gradient_submitted(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .build()
            .await;

        // Send signal
        let signal = HolochainSignal::GradientSubmitted {
            node_id: "test".to_string(),
            round_num: 1,
            entry_hash: "hash".to_string(),
        };
        tx.send(signal.clone()).await.unwrap();

        // Manually dispatch (since we're not running the listener loop)
        listener.dispatch_signal(signal).await;

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_signal_stats() {
        let mut stats = SignalStats::default();

        stats.record(&HolochainSignal::GradientSubmitted {
            node_id: "n1".to_string(),
            round_num: 1,
            entry_hash: "h1".to_string(),
        });
        stats.record(&HolochainSignal::ByzantineDetected {
            node_ids: vec!["n2".to_string()],
            detection_method: "test".to_string(),
            round_num: 1,
        });

        assert_eq!(stats.gradients_received, 1);
        assert_eq!(stats.byzantine_detections, 1);
        assert_eq!(stats.total_signals, 2);
    }
}
