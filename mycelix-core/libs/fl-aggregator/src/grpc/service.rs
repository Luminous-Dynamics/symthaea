//! gRPC service implementation for Federated Learning.
//!
//! This module provides the `FederatedLearningService` implementation that handles
//! gRPC requests for node registration, gradient submission, status queries, and
//! real-time round event streaming.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{broadcast, mpsc, RwLock};
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::{Request, Response, Status};

use crate::aggregator::AsyncAggregator;
use crate::payload::UnifiedPayload;

use super::conversions::{
    grpc_payload_to_unified, aggregator_status_to_grpc_response,
};
use super::{
    GetAggregatedRequest, GetAggregatedResponse, GetStatusRequest, GetStatusResponse,
    GrpcPayload, GrpcRoundEventType, GrpcRoundState, HeartbeatRequest, HeartbeatResponse,
    NodeCapabilities, NodeIdentity, RegisterNodeRequest, RegisterNodeResponse,
    RoundEvent, RoundEventDetails, RoundCompletedDetails,
    SubmissionReceivedDetails, SubmitRequest, SubmitResponse,
    UnregisterNodeRequest, UnregisterNodeResponse, WatchRoundsRequest,
    AggregationMetadata, GrpcPayloadType, GrpcPayloadData, GrpcDenseGradient,
};

// ============================================================================
// Node Session Management
// ============================================================================

/// Information about a connected node session.
#[derive(Clone, Debug)]
pub struct NodeSession {
    /// Node identity
    pub identity: NodeIdentity,

    /// Node capabilities
    pub capabilities: Option<NodeCapabilities>,

    /// Session token
    pub session_token: String,

    /// When the session was created
    pub created_at: Instant,

    /// Last heartbeat time
    pub last_heartbeat: Instant,

    /// Number of submissions from this node
    pub submission_count: u64,

    /// Total bytes submitted
    pub bytes_submitted: u64,
}

impl NodeSession {
    /// Create a new node session.
    pub fn new(identity: NodeIdentity, capabilities: Option<NodeCapabilities>) -> Self {
        let session_token = Self::generate_session_token(&identity.did);
        let now = Instant::now();

        Self {
            identity,
            capabilities,
            session_token,
            created_at: now,
            last_heartbeat: now,
            submission_count: 0,
            bytes_submitted: 0,
        }
    }

    /// Generate a session token from DID and timestamp.
    fn generate_session_token(did: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let mut hasher = DefaultHasher::new();
        did.hash(&mut hasher);
        timestamp.hash(&mut hasher);

        format!("sess_{:016x}", hasher.finish())
    }

    /// Update heartbeat timestamp.
    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
    }

    /// Check if session is expired.
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.last_heartbeat.elapsed() > timeout
    }

    /// Record a submission.
    pub fn record_submission(&mut self, bytes: u64) {
        self.submission_count += 1;
        self.bytes_submitted += bytes;
    }
}

// ============================================================================
// Service State
// ============================================================================

/// Shared state for the gRPC service.
///
/// This struct holds all the state needed by the service, including the
/// aggregator, node sessions, and event broadcast channels.
pub struct ServiceState {
    /// The underlying async aggregator
    pub aggregator: AsyncAggregator,

    /// Active node sessions (keyed by session token)
    pub node_sessions: RwLock<HashMap<String, NodeSession>>,

    /// Broadcast channel for round events
    pub round_events: broadcast::Sender<RoundEvent>,

    /// Service start time
    start_time: Instant,

    /// Service configuration
    pub config: GrpcServiceConfig,
}

impl ServiceState {
    /// Create a new service state.
    pub fn new(aggregator: AsyncAggregator) -> Self {
        Self::with_config(aggregator, GrpcServiceConfig::default())
    }

    /// Create a new service state with custom configuration.
    pub fn with_config(aggregator: AsyncAggregator, config: GrpcServiceConfig) -> Self {
        let (tx, _) = broadcast::channel(config.event_channel_capacity);

        Self {
            aggregator,
            node_sessions: RwLock::new(HashMap::new()),
            round_events: tx,
            start_time: Instant::now(),
            config,
        }
    }

    /// Get uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Get session by token.
    pub async fn get_session(&self, token: &str) -> Option<NodeSession> {
        let sessions = self.node_sessions.read().await;
        sessions.get(token).cloned()
    }

    /// Get session by DID.
    pub async fn get_session_by_did(&self, did: &str) -> Option<NodeSession> {
        let sessions = self.node_sessions.read().await;
        sessions.values().find(|s| s.identity.did == did).cloned()
    }

    /// Broadcast a round event.
    pub fn broadcast_event(&self, event: RoundEvent) {
        // Ignore send errors (no receivers is fine)
        let _ = self.round_events.send(event);
    }

    /// Get current unix timestamp.
    pub fn current_timestamp() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
    }
}

/// Configuration for the gRPC service.
#[derive(Clone, Debug)]
pub struct GrpcServiceConfig {
    /// Session timeout duration
    pub session_timeout: Duration,

    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,

    /// Event channel capacity
    pub event_channel_capacity: usize,

    /// Whether to require signatures
    pub require_signatures: bool,

    /// Whether to require proofs
    pub require_proofs: bool,

    /// Maximum payload size in bytes
    pub max_payload_bytes: usize,
}

impl Default for GrpcServiceConfig {
    fn default() -> Self {
        Self {
            session_timeout: Duration::from_secs(300), // 5 minutes
            max_concurrent_streams: 100,
            event_channel_capacity: 1000,
            require_signatures: false,
            require_proofs: false,
            max_payload_bytes: 100_000_000, // 100 MB
        }
    }
}

// ============================================================================
// Service Implementation
// ============================================================================

/// Type alias for streaming response
pub type RoundEventStream = Pin<Box<dyn Stream<Item = Result<RoundEvent, Status>> + Send>>;

/// FederatedLearning gRPC service implementation.
///
/// This service provides the core FL coordination functionality:
/// - Node registration and session management
/// - Gradient/payload submission
/// - Aggregation status and results
/// - Real-time round event streaming
pub struct FederatedLearningServer {
    state: Arc<ServiceState>,
}

impl FederatedLearningServer {
    /// Create a new FederatedLearning server.
    pub fn new(state: Arc<ServiceState>) -> Self {
        Self { state }
    }

    /// Create with new state from aggregator.
    pub fn from_aggregator(aggregator: AsyncAggregator, config: GrpcServiceConfig) -> Self {
        let state = Arc::new(ServiceState::with_config(aggregator, config));
        Self { state }
    }

    /// Get the underlying state.
    pub fn state(&self) -> Arc<ServiceState> {
        Arc::clone(&self.state)
    }
}

/// Trait for the FederatedLearning gRPC service.
///
/// This mirrors the proto service definition and allows for different
/// implementations (e.g., for testing).
#[tonic::async_trait]
pub trait FederatedLearningService: Send + Sync + 'static {
    /// Register a node with the aggregator.
    async fn register_node(
        &self,
        request: Request<RegisterNodeRequest>,
    ) -> Result<Response<RegisterNodeResponse>, Status>;

    /// Send a heartbeat to keep session alive.
    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status>;

    /// Unregister a node.
    async fn unregister_node(
        &self,
        request: Request<UnregisterNodeRequest>,
    ) -> Result<Response<UnregisterNodeResponse>, Status>;

    /// Submit a gradient/payload.
    async fn submit(
        &self,
        request: Request<SubmitRequest>,
    ) -> Result<Response<SubmitResponse>, Status>;

    /// Get current aggregator status.
    async fn get_status(
        &self,
        request: Request<GetStatusRequest>,
    ) -> Result<Response<GetStatusResponse>, Status>;

    /// Get aggregated result.
    async fn get_aggregated(
        &self,
        request: Request<GetAggregatedRequest>,
    ) -> Result<Response<GetAggregatedResponse>, Status>;

    /// Stream type for watch_rounds
    type WatchRoundsStream: Stream<Item = Result<RoundEvent, Status>> + Send + 'static;

    /// Watch for round events.
    async fn watch_rounds(
        &self,
        request: Request<WatchRoundsRequest>,
    ) -> Result<Response<Self::WatchRoundsStream>, Status>;
}

#[tonic::async_trait]
impl FederatedLearningService for FederatedLearningServer {
    async fn register_node(
        &self,
        request: Request<RegisterNodeRequest>,
    ) -> Result<Response<RegisterNodeResponse>, Status> {
        let req = request.into_inner();

        // Validate request
        let identity = req.node.ok_or_else(|| {
            Status::invalid_argument("Node identity is required")
        })?;

        if identity.did.is_empty() {
            return Err(Status::invalid_argument("Node DID cannot be empty"));
        }

        // Check if already registered
        if let Some(existing) = self.state.get_session_by_did(&identity.did).await {
            return Ok(Response::new(RegisterNodeResponse {
                accepted: true,
                error_message: String::new(),
                session_token: existing.session_token,
                current_round: self.state.aggregator.status().await.round,
            }));
        }

        // Create new session
        let session = NodeSession::new(identity.clone(), req.capabilities);
        let session_token = session.session_token.clone();

        // Register with aggregator
        if let Err(e) = self.state.aggregator.register_node(&identity.did).await {
            return Ok(Response::new(RegisterNodeResponse {
                accepted: false,
                error_message: e.to_string(),
                session_token: String::new(),
                current_round: 0,
            }));
        }

        // Store session
        {
            let mut sessions = self.state.node_sessions.write().await;
            sessions.insert(session_token.clone(), session);
        }

        let status = self.state.aggregator.status().await;

        tracing::info!(
            "Node registered: did={}, session={}",
            identity.did,
            session_token
        );

        Ok(Response::new(RegisterNodeResponse {
            accepted: true,
            error_message: String::new(),
            session_token,
            current_round: status.round,
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();

        if req.session_token.is_empty() {
            return Err(Status::invalid_argument("Session token is required"));
        }

        // Update session heartbeat
        let mut sessions = self.state.node_sessions.write().await;
        if let Some(session) = sessions.get_mut(&req.session_token) {
            session.update_heartbeat();

            let status = self.state.aggregator.status().await;
            let round_state = if status.is_complete {
                GrpcRoundState::Complete
            } else if status.submitted_nodes > 0 {
                GrpcRoundState::Collecting
            } else {
                GrpcRoundState::Pending
            };

            Ok(Response::new(HeartbeatResponse {
                ok: true,
                current_round: status.round,
                round_state: round_state as i32,
            }))
        } else {
            Err(Status::not_found("Session not found or expired"))
        }
    }

    async fn unregister_node(
        &self,
        request: Request<UnregisterNodeRequest>,
    ) -> Result<Response<UnregisterNodeResponse>, Status> {
        let req = request.into_inner();

        if req.session_token.is_empty() {
            return Err(Status::invalid_argument("Session token is required"));
        }

        // Remove session and unregister from aggregator
        let mut sessions = self.state.node_sessions.write().await;
        if let Some(session) = sessions.remove(&req.session_token) {
            let _ = self.state.aggregator.unregister_node(&session.identity.did).await;

            tracing::info!(
                "Node unregistered: did={}, submissions={}",
                session.identity.did,
                session.submission_count
            );

            Ok(Response::new(UnregisterNodeResponse { ok: true }))
        } else {
            Ok(Response::new(UnregisterNodeResponse { ok: false }))
        }
    }

    async fn submit(
        &self,
        request: Request<SubmitRequest>,
    ) -> Result<Response<SubmitResponse>, Status> {
        let req = request.into_inner();

        // Validate node identity
        let identity = req.node.ok_or_else(|| {
            Status::invalid_argument("Node identity is required")
        })?;

        // Validate payload
        let payload = req.payload.ok_or_else(|| {
            Status::invalid_argument("Payload is required")
        })?;

        // Check payload size
        let payload_size = estimate_payload_size(&payload);
        if payload_size > self.state.config.max_payload_bytes {
            return Err(Status::resource_exhausted(format!(
                "Payload size {} exceeds maximum {}",
                payload_size, self.state.config.max_payload_bytes
            )));
        }

        // Check signature if required
        if self.state.config.require_signatures && req.signature.is_none() {
            return Err(Status::unauthenticated("Signature is required"));
        }

        // Check proofs if required
        if self.state.config.require_proofs
            && req.pogq_proof.is_empty()
            && req.zkstark_proof.is_empty()
        {
            return Err(Status::failed_precondition("Proof is required"));
        }

        // Convert payload to internal format
        let unified = grpc_payload_to_unified(&payload).map_err(|e| {
            Status::invalid_argument(format!("Invalid payload: {}", e))
        })?;

        // Extract gradient for submission
        let gradient = match &unified {
            UnifiedPayload::Dense(g) => g.values.clone(),
            UnifiedPayload::Sparse(s) => s.to_dense().values,
            UnifiedPayload::Quantized(q) => q.to_dense().values,
            _ => {
                return Err(Status::unimplemented(
                    "HDC payload types not yet supported for submission"
                ));
            }
        };

        // Submit to aggregator
        match self.state.aggregator.submit(&identity.did, gradient).await {
            Ok(_) => {
                // Update session stats if exists
                if let Some(session) = self.state.get_session_by_did(&identity.did).await {
                    let mut sessions = self.state.node_sessions.write().await;
                    if let Some(s) = sessions.get_mut(&session.session_token) {
                        s.record_submission(payload_size as u64);
                    }
                }

                // Broadcast submission event
                let event = RoundEvent {
                    round: req.round,
                    event_type: GrpcRoundEventType::SubmissionReceived as i32,
                    timestamp: ServiceState::current_timestamp(),
                    details: Some(RoundEventDetails::Submission(SubmissionReceivedDetails {
                        node_did: identity.did.clone(),
                        payload_type: get_payload_type(&payload) as i32,
                        size_bytes: payload_size as u64,
                    })),
                };
                self.state.broadcast_event(event);

                let status = self.state.aggregator.status().await;

                // Check if round completed
                if status.is_complete {
                    let completed_event = RoundEvent {
                        round: status.round,
                        event_type: GrpcRoundEventType::RoundCompleted as i32,
                        timestamp: ServiceState::current_timestamp(),
                        details: Some(RoundEventDetails::Completed(RoundCompletedDetails {
                            participants: status.submitted_nodes as u32,
                            // Byzantine count: Currently 0 as the aggregator doesn't track excluded
                            // nodes separately. The Byzantine defense algorithms (Krum, TrimmedMean, etc.)
                            // filter internally but don't expose exclusion counts yet.
                            byzantine_count: 0,
                            aggregation_time_ms: 0.0, // Will be set when aggregation happens
                        })),
                    };
                    self.state.broadcast_event(completed_event);
                }

                Ok(Response::new(SubmitResponse {
                    accepted: true,
                    error_message: String::new(),
                    round: status.round,
                    submissions_received: status.submitted_nodes as u32,
                    submissions_expected: status.expected_nodes as u32,
                    round_complete: status.is_complete,
                }))
            }
            Err(e) => Ok(Response::new(SubmitResponse {
                accepted: false,
                error_message: e.to_string(),
                round: req.round,
                submissions_received: 0,
                submissions_expected: 0,
                round_complete: false,
            })),
        }
    }

    async fn get_status(
        &self,
        _request: Request<GetStatusRequest>,
    ) -> Result<Response<GetStatusResponse>, Status> {
        let status = self.state.aggregator.status().await;
        let response = aggregator_status_to_grpc_response(&status);
        Ok(Response::new(response))
    }

    async fn get_aggregated(
        &self,
        request: Request<GetAggregatedRequest>,
    ) -> Result<Response<GetAggregatedResponse>, Status> {
        let req = request.into_inner();

        // Get aggregated gradient for the requested round
        // Note: Currently returns latest completed round. Future enhancement:
        // implement historical round retrieval if req.round != 0
        let _requested_round = req.round;
        let _preferred_format = req.preferred_format;

        match self.state.aggregator.get_aggregated_gradient().await {
            Ok(Some(gradient)) => {
                let status = self.state.aggregator.status().await;

                // Convert to requested format
                let payload = GrpcPayload {
                    data: Some(GrpcPayloadData::Dense(GrpcDenseGradient {
                        values: gradient.to_vec(),
                        dimension: gradient.len() as u64,
                    })),
                };

                Ok(Response::new(GetAggregatedResponse {
                    round: status.round.saturating_sub(1),
                    aggregated: Some(payload),
                    metadata: Some(AggregationMetadata {
                        algorithm: status.algorithm.clone(),
                        participants: status.submitted_nodes as u32,
                        excluded: 0, // TODO: Track excluded count in aggregator when Byzantine detection occurs
                        byzantine_ratio: 0.0,
                        aggregation_time_ms: 0,
                        shapley_values: HashMap::new(),
                    }),
                    byzantine_nodes: Vec::new(),
                    healed_nodes: Vec::new(),
                    aggregation_proof: Vec::new(),
                }))
            }
            Ok(None) => {
                let status = self.state.aggregator.status().await;
                Err(Status::failed_precondition(format!(
                    "Round {} not complete: {}/{} submissions",
                    status.round, status.submitted_nodes, status.expected_nodes
                )))
            }
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    type WatchRoundsStream = ReceiverStream<Result<RoundEvent, Status>>;

    async fn watch_rounds(
        &self,
        request: Request<WatchRoundsRequest>,
    ) -> Result<Response<Self::WatchRoundsStream>, Status> {
        let req = request.into_inner();

        let (tx, rx) = mpsc::channel(32);
        let mut broadcast_rx = self.state.round_events.subscribe();

        // Filter settings
        let start_round = req.start_round;
        let event_types: Vec<String> = req.event_types;

        // Spawn task to forward events
        tokio::spawn(async move {
            while let Ok(event) = broadcast_rx.recv().await {
                // Apply filters
                if event.round < start_round {
                    continue;
                }

                if !event_types.is_empty() {
                    let event_type_str = match event.event_type {
                        x if x == GrpcRoundEventType::RoundStarted as i32 => "ROUND_STARTED",
                        x if x == GrpcRoundEventType::SubmissionReceived as i32 => "SUBMISSION_RECEIVED",
                        x if x == GrpcRoundEventType::RoundCompleted as i32 => "ROUND_COMPLETED",
                        x if x == GrpcRoundEventType::ByzantineDetected as i32 => "BYZANTINE_DETECTED",
                        _ => "UNSPECIFIED",
                    };

                    if !event_types.iter().any(|t| t == event_type_str) {
                        continue;
                    }
                }

                if tx.send(Ok(event)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Estimate payload size in bytes.
fn estimate_payload_size(payload: &GrpcPayload) -> usize {
    match &payload.data {
        Some(GrpcPayloadData::Dense(g)) => g.values.len() * 4,
        Some(GrpcPayloadData::Hyper(h)) => h.hypervector.len(),
        Some(GrpcPayloadData::Binary(b)) => b.data.len(),
        Some(GrpcPayloadData::Sparse(s)) => {
            s.indices.len() * 4 + s.values.len() * 4
        }
        Some(GrpcPayloadData::Quantized(q)) => q.data.len(),
        None => 0,
    }
}

/// Get payload type enum value.
fn get_payload_type(payload: &GrpcPayload) -> GrpcPayloadType {
    match &payload.data {
        Some(GrpcPayloadData::Dense(_)) => GrpcPayloadType::DenseGradient,
        Some(GrpcPayloadData::Hyper(_)) => GrpcPayloadType::HyperEncoded,
        Some(GrpcPayloadData::Binary(_)) => GrpcPayloadType::BinaryHypervector,
        Some(GrpcPayloadData::Sparse(_)) => GrpcPayloadType::SparseGradient,
        Some(GrpcPayloadData::Quantized(_)) => GrpcPayloadType::QuantizedGradient,
        None => GrpcPayloadType::Unspecified,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregator::AggregatorConfig;
    use crate::byzantine::Defense;

    fn create_test_state() -> Arc<ServiceState> {
        let config = AggregatorConfig::default()
            .with_expected_nodes(3)
            .with_defense(Defense::FedAvg);

        let aggregator = AsyncAggregator::new(config);
        Arc::new(ServiceState::new(aggregator))
    }

    #[test]
    fn test_node_session_creation() {
        let identity = NodeIdentity {
            did: "did:mycelix:test".to_string(),
            public_key: vec![1, 2, 3],
            assurance_level: 2,
        };

        let session = NodeSession::new(identity, None);

        assert!(session.session_token.starts_with("sess_"));
        assert_eq!(session.submission_count, 0);
        assert!(!session.is_expired(Duration::from_secs(10)));
    }

    #[test]
    fn test_session_token_generation() {
        let token1 = NodeSession::generate_session_token("did:mycelix:node1");
        let token2 = NodeSession::generate_session_token("did:mycelix:node2");

        assert!(token1.starts_with("sess_"));
        assert!(token2.starts_with("sess_"));
        // Tokens should be different for different DIDs
        // (though not guaranteed due to timing, in practice they will be)
        assert_eq!(token1.len(), token2.len());
    }

    #[test]
    fn test_session_heartbeat() {
        let identity = NodeIdentity {
            did: "did:mycelix:test".to_string(),
            ..Default::default()
        };

        let mut session = NodeSession::new(identity, None);
        let original_heartbeat = session.last_heartbeat;

        std::thread::sleep(std::time::Duration::from_millis(10));
        session.update_heartbeat();

        assert!(session.last_heartbeat > original_heartbeat);
    }

    #[test]
    fn test_session_submission_recording() {
        let identity = NodeIdentity {
            did: "did:mycelix:test".to_string(),
            ..Default::default()
        };

        let mut session = NodeSession::new(identity, None);
        assert_eq!(session.submission_count, 0);
        assert_eq!(session.bytes_submitted, 0);

        session.record_submission(1000);
        assert_eq!(session.submission_count, 1);
        assert_eq!(session.bytes_submitted, 1000);

        session.record_submission(2000);
        assert_eq!(session.submission_count, 2);
        assert_eq!(session.bytes_submitted, 3000);
    }

    #[tokio::test]
    async fn test_service_state_creation() {
        let state = create_test_state();

        assert!(state.uptime_secs() < 1);
        assert!(state.get_session("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_register_node() {
        let state = create_test_state();
        let server = FederatedLearningServer::new(state);

        let request = Request::new(RegisterNodeRequest {
            node: Some(NodeIdentity {
                did: "did:mycelix:test".to_string(),
                public_key: vec![1, 2, 3],
                assurance_level: 2,
            }),
            capabilities: Some(NodeCapabilities {
                supported_payloads: vec![GrpcPayloadType::DenseGradient as i32],
                supports_streaming: true,
                supports_proofs: false,
                max_gradient_dimension: 1_000_000,
                framework: "pytorch".to_string(),
            }),
        });

        let response = server.register_node(request).await.unwrap();
        let resp = response.into_inner();

        assert!(resp.accepted);
        assert!(!resp.session_token.is_empty());
        assert!(resp.session_token.starts_with("sess_"));
    }

    #[tokio::test]
    async fn test_register_node_missing_identity() {
        let state = create_test_state();
        let server = FederatedLearningServer::new(state);

        let request = Request::new(RegisterNodeRequest {
            node: None,
            capabilities: None,
        });

        let result = server.register_node(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let state = create_test_state();
        let server = FederatedLearningServer::new(state.clone());

        // First register a node
        let register_req = Request::new(RegisterNodeRequest {
            node: Some(NodeIdentity {
                did: "did:mycelix:test".to_string(),
                ..Default::default()
            }),
            capabilities: None,
        });

        let register_resp = server.register_node(register_req).await.unwrap();
        let session_token = register_resp.into_inner().session_token;

        // Send heartbeat
        let heartbeat_req = Request::new(HeartbeatRequest {
            session_token: session_token.clone(),
            timestamp: ServiceState::current_timestamp(),
        });

        let heartbeat_resp = server.heartbeat(heartbeat_req).await.unwrap();
        assert!(heartbeat_resp.into_inner().ok);
    }

    #[tokio::test]
    async fn test_get_status() {
        let state = create_test_state();
        let server = FederatedLearningServer::new(state);

        let request = Request::new(GetStatusRequest {});
        let response = server.get_status(request).await.unwrap();
        let resp = response.into_inner();

        assert_eq!(resp.current_round, 0);
        assert_eq!(resp.registered_nodes, 0);
        assert_eq!(resp.expected_this_round, 3); // From test config
    }

    #[tokio::test]
    async fn test_submit_gradient() {
        let state = create_test_state();
        let server = FederatedLearningServer::new(state.clone());

        // Register node first
        let register_req = Request::new(RegisterNodeRequest {
            node: Some(NodeIdentity {
                did: "did:mycelix:node1".to_string(),
                ..Default::default()
            }),
            capabilities: None,
        });
        server.register_node(register_req).await.unwrap();

        // Submit gradient
        let submit_req = Request::new(SubmitRequest {
            node: Some(NodeIdentity {
                did: "did:mycelix:node1".to_string(),
                ..Default::default()
            }),
            round: 0,
            payload: Some(GrpcPayload {
                data: Some(GrpcPayloadData::Dense(GrpcDenseGradient {
                    values: vec![1.0, 2.0, 3.0],
                    dimension: 3,
                })),
            }),
            ..Default::default()
        });

        let response = server.submit(submit_req).await.unwrap();
        let resp = response.into_inner();

        assert!(resp.accepted);
        assert_eq!(resp.submissions_received, 1);
        assert!(!resp.round_complete);
    }

    #[tokio::test]
    async fn test_unregister_node() {
        let state = create_test_state();
        let server = FederatedLearningServer::new(state);

        // Register
        let register_req = Request::new(RegisterNodeRequest {
            node: Some(NodeIdentity {
                did: "did:mycelix:test".to_string(),
                ..Default::default()
            }),
            capabilities: None,
        });

        let register_resp = server.register_node(register_req).await.unwrap();
        let session_token = register_resp.into_inner().session_token;

        // Unregister
        let unregister_req = Request::new(UnregisterNodeRequest { session_token });
        let response = server.unregister_node(unregister_req).await.unwrap();

        assert!(response.into_inner().ok);
    }

    #[test]
    fn test_estimate_payload_size() {
        let dense_payload = GrpcPayload {
            data: Some(GrpcPayloadData::Dense(GrpcDenseGradient {
                values: vec![1.0; 100],
                dimension: 100,
            })),
        };
        assert_eq!(estimate_payload_size(&dense_payload), 400); // 100 * 4 bytes

        let sparse_payload = GrpcPayload {
            data: Some(GrpcPayloadData::Sparse(super::super::GrpcSparseGradient {
                indices: vec![0, 1, 2],
                values: vec![1.0, 2.0, 3.0],
                full_dimension: 100,
                sparsity_ratio: 0.03,
            })),
        };
        assert_eq!(estimate_payload_size(&sparse_payload), 24); // 3*4 + 3*4
    }

    #[test]
    fn test_get_payload_type() {
        let dense = GrpcPayload {
            data: Some(GrpcPayloadData::Dense(GrpcDenseGradient::default())),
        };
        assert_eq!(get_payload_type(&dense), GrpcPayloadType::DenseGradient);

        let hyper = GrpcPayload {
            data: Some(GrpcPayloadData::Hyper(super::super::GrpcHyperEncodedGradient::default())),
        };
        assert_eq!(get_payload_type(&hyper), GrpcPayloadType::HyperEncoded);

        let empty = GrpcPayload { data: None };
        assert_eq!(get_payload_type(&empty), GrpcPayloadType::Unspecified);
    }

    #[test]
    fn test_grpc_service_config_default() {
        let config = GrpcServiceConfig::default();

        assert_eq!(config.session_timeout, Duration::from_secs(300));
        assert_eq!(config.max_concurrent_streams, 100);
        assert_eq!(config.event_channel_capacity, 1000);
        assert!(!config.require_signatures);
        assert!(!config.require_proofs);
    }
}
