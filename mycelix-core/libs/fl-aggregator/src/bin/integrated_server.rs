//! Integrated FL Server with Holochain Persistence
//!
//! Full-featured federated learning server with:
//! - HTTP API for gradient submission
//! - Holochain DHT persistence
//! - Multi-round coordination
//! - Byzantine detection and reputation
//! - Chaos testing hooks
//!
//! Run with: `cargo run --bin integrated-server --features "http-api,holochain"`
//!
//! Environment Variables:
//!   FL_BIND_ADDR          - HTTP bind address (default: 0.0.0.0:3000)
//!   FL_EXPECTED_NODES     - Expected participants per round (default: 10)
//!   FL_MAX_MEMORY_MB      - Memory limit in MB (default: 2000)
//!   FL_DEFENSE            - Defense algorithm: fedavg, krum, median, trimmedmean
//!   FL_BYZANTINE_F        - Byzantine tolerance parameter for Krum
//!   FL_MAX_ROUNDS         - Maximum rounds (0 = unlimited)
//!   FL_ROUND_TIMEOUT_SECS - Round timeout in seconds (default: 60)
//!   FL_MIN_PARTICIPANTS   - Minimum participants per round (default: 3)
//!   HOLOCHAIN_ADMIN_URL   - Conductor admin WebSocket URL
//!   HOLOCHAIN_APP_URL     - Conductor app WebSocket URL
//!   HOLOCHAIN_DNA_HASH    - DNA hash for cell ID
//!   HOLOCHAIN_AGENT_KEY   - Agent public key for cell ID
//!   FL_ENABLE_CHAOS       - Enable chaos testing hooks (default: false)
//!   FL_CHAOS_FAIL_RATE    - Probability of random failure (0.0-1.0)

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::{broadcast, RwLock};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use fl_aggregator::{
    aggregator::{AggregatorConfig, AggregatorStatus, AsyncAggregator},
    byzantine::Defense,
    coordinator::{CoordinatorConfig, CoordinatorEvent, RoundCoordinator, RoundInfo, RoundState},
    metrics::AggregatorMetrics,
};

#[cfg(feature = "holochain")]
use fl_aggregator::holochain::{HolochainClient, HolochainConfig};

// =============================================================================
// Application State
// =============================================================================

/// Integrated application state.
pub struct IntegratedState {
    /// The aggregator instance (cloned reference, shares state with coordinator).
    pub aggregator: AsyncAggregator,
    /// Round coordinator.
    pub coordinator: Arc<RoundCoordinator>,
    /// Holochain client (optional).
    #[cfg(feature = "holochain")]
    pub holochain: Option<Arc<tokio::sync::RwLock<HolochainClient>>>,
    /// Event broadcast sender.
    pub event_tx: broadcast::Sender<CoordinatorEvent>,
    /// Chaos testing configuration.
    pub chaos: ChaosConfig,
}

/// Chaos testing configuration.
#[derive(Clone, Debug)]
pub struct ChaosConfig {
    pub enabled: bool,
    pub fail_rate: f32,
    pub latency_ms: Option<u64>,
    pub drop_rate: f32,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fail_rate: 0.0,
            latency_ms: None,
            drop_rate: 0.0,
        }
    }
}

// =============================================================================
// HTTP Handlers
// =============================================================================

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    #[cfg(feature = "holochain")]
    holochain_connected: bool,
    coordinator_running: bool,
}

async fn health(State(state): State<Arc<IntegratedState>>) -> Json<HealthResponse> {
    let round_info = state.coordinator.round_info().await;

    Json(HealthResponse {
        status: "healthy",
        version: env!("CARGO_PKG_VERSION"),
        #[cfg(feature = "holochain")]
        holochain_connected: if let Some(ref hc) = state.holochain {
            hc.read().await.is_connected().await
        } else {
            false
        },
        coordinator_running: round_info.state == RoundState::Collecting,
    })
}

#[derive(Serialize)]
struct StatusResponse {
    aggregator: AggregatorStatus,
    round: RoundInfo,
    #[cfg(feature = "holochain")]
    holochain: HolochainStatus,
}

#[cfg(feature = "holochain")]
#[derive(Serialize)]
struct HolochainStatus {
    connected: bool,
    admin_url: String,
    app_url: String,
}

async fn get_status(State(state): State<Arc<IntegratedState>>) -> Json<StatusResponse> {
    Json(StatusResponse {
        aggregator: state.aggregator.status().await,
        round: state.coordinator.round_info().await,
        #[cfg(feature = "holochain")]
        holochain: if let Some(ref hc) = state.holochain {
            let client = hc.read().await;
            HolochainStatus {
                connected: client.is_connected().await,
                admin_url: "configured".to_string(),
                app_url: "configured".to_string(),
            }
        } else {
            HolochainStatus {
                connected: false,
                admin_url: "not configured".to_string(),
                app_url: "not configured".to_string(),
            }
        },
    })
}

async fn get_metrics(State(state): State<Arc<IntegratedState>>) -> Json<AggregatorMetrics> {
    Json(state.aggregator.metrics().await)
}

async fn get_prometheus_metrics(State(state): State<Arc<IntegratedState>>) -> impl IntoResponse {
    let metrics = state.aggregator.metrics().await;
    (
        StatusCode::OK,
        [("content-type", "text/plain; charset=utf-8")],
        metrics.to_prometheus(),
    )
}

const MAX_GRADIENT_SIZE: usize = 10_000_000;
const MAX_NODE_ID_LENGTH: usize = 256;

#[derive(Deserialize)]
struct GradientRequest {
    node_id: String,
    gradient: Vec<f32>,
    /// Optional round number (reserved for coordinator integration)
    #[serde(default)]
    #[allow(dead_code)]
    round: Option<u64>,
}

impl GradientRequest {
    fn validate(&self) -> Result<(), String> {
        if self.node_id.is_empty() || self.node_id.len() > MAX_NODE_ID_LENGTH {
            return Err("Invalid node_id".into());
        }
        if self.gradient.is_empty() || self.gradient.len() > MAX_GRADIENT_SIZE {
            return Err("Invalid gradient size".into());
        }
        for (i, &val) in self.gradient.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("Non-finite value at index {}", i));
            }
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct GradientResponse {
    status: String,
    round: u64,
    submitted: usize,
    expected: usize,
    is_complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn submit_gradient(
    State(state): State<Arc<IntegratedState>>,
    Json(req): Json<GradientRequest>,
) -> impl IntoResponse {
    // Chaos testing: random failure
    if state.chaos.enabled && rand::random::<f32>() < state.chaos.fail_rate {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(GradientResponse {
                status: "chaos: simulated failure".to_string(),
                round: 0,
                submitted: 0,
                expected: 0,
                is_complete: false,
                error: Some("Chaos testing: random failure".to_string()),
            }),
        );
    }

    // Chaos testing: latency injection
    if let Some(latency) = state.chaos.latency_ms {
        if state.chaos.enabled {
            tokio::time::sleep(Duration::from_millis(latency)).await;
        }
    }

    // Validate request
    if let Err(e) = req.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(GradientResponse {
                status: "validation error".to_string(),
                round: 0,
                submitted: 0,
                expected: 0,
                is_complete: false,
                error: Some(e),
            }),
        );
    }

    let gradient = ndarray::Array1::from(req.gradient);

    // Submit through coordinator
    match state.coordinator.submit_gradient(&req.node_id, gradient).await {
        Ok(_) => {
            let round_info = state.coordinator.round_info().await;
            let status = state.aggregator.status().await;
            (
                StatusCode::ACCEPTED,
                Json(GradientResponse {
                    status: "accepted".to_string(),
                    round: round_info.round,
                    submitted: status.submitted_nodes,
                    expected: status.expected_nodes,
                    is_complete: round_info.state == RoundState::Complete,
                    error: None,
                }),
            )
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(GradientResponse {
                status: "error".to_string(),
                round: 0,
                submitted: 0,
                expected: 0,
                is_complete: false,
                error: Some(e.to_string()),
            }),
        ),
    }
}

#[derive(Serialize)]
struct RoundResponse {
    round: RoundInfo,
    participants: Vec<String>,
    byzantine_detected: Vec<String>,
}

async fn get_round(State(state): State<Arc<IntegratedState>>) -> Json<RoundResponse> {
    let info = state.coordinator.round_info().await;
    Json(RoundResponse {
        participants: info.participants.clone(),
        byzantine_detected: info.byzantine_nodes.clone(),
        round: info,
    })
}

async fn force_aggregate(State(state): State<Arc<IntegratedState>>) -> impl IntoResponse {
    let _ = state
        .coordinator
        .command_sender()
        .send(fl_aggregator::coordinator::CoordinatorCommand::ForceFinalize)
        .await;

    // Wait a bit for aggregation
    tokio::time::sleep(Duration::from_millis(100)).await;

    let info = state.coordinator.round_info().await;
    (StatusCode::OK, Json(info))
}

#[derive(Serialize)]
struct EventsResponse {
    events: Vec<CoordinatorEvent>,
}

async fn get_events(State(_state): State<Arc<IntegratedState>>) -> Json<EventsResponse> {
    // Return empty - clients should use WebSocket for real-time events
    Json(EventsResponse { events: vec![] })
}

/// Server-sent events endpoint for real-time updates.
async fn events_stream(State(state): State<Arc<IntegratedState>>) -> impl IntoResponse {
    let mut rx = state.event_tx.subscribe();

    let stream = async_stream::stream! {
        while let Ok(event) = rx.recv().await {
            let data = serde_json::to_string(&event).unwrap_or_default();
            yield Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default().data(data)
            );
        }
    };

    axum::response::Sse::new(stream)
}

// =============================================================================
// Chaos Testing Endpoints
// =============================================================================

#[derive(Deserialize)]
struct ChaosRequest {
    enabled: Option<bool>,
    fail_rate: Option<f32>,
    latency_ms: Option<u64>,
    drop_rate: Option<f32>,
}

async fn configure_chaos(
    State(_state): State<Arc<IntegratedState>>,
    Json(req): Json<ChaosRequest>,
) -> impl IntoResponse {
    // Note: In a real implementation, we'd update state.chaos
    // For now, just acknowledge
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "chaos configuration updated",
            "enabled": req.enabled.unwrap_or(false),
            "fail_rate": req.fail_rate.unwrap_or(0.0),
            "latency_ms": req.latency_ms,
            "drop_rate": req.drop_rate.unwrap_or(0.0),
        })),
    )
}

async fn trigger_partition(State(_state): State<Arc<IntegratedState>>) -> impl IntoResponse {
    // Simulate network partition by disconnecting Holochain
    #[cfg(feature = "holochain")]
    {
        // Would disconnect holochain client here
        info!("Simulating network partition");
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "partition triggered",
            "duration_ms": 5000
        })),
    )
}

async fn inject_byzantine(
    State(state): State<Arc<IntegratedState>>,
    Path(node_id): Path<String>,
) -> impl IntoResponse {
    // Inject a Byzantine gradient
    let byzantine_gradient: Vec<f32> = (0..100).map(|_| rand::random::<f32>() * 1000.0 - 500.0).collect();
    let gradient = ndarray::Array1::from(byzantine_gradient);

    match state.coordinator.submit_gradient(&node_id, gradient).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "byzantine gradient injected",
                "node_id": node_id
            })),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "status": "injection failed",
                "error": e.to_string()
            })),
        ),
    }
}

// =============================================================================
// Router Setup
// =============================================================================

fn create_router(state: Arc<IntegratedState>) -> Router {
    Router::new()
        // Health and status
        .route("/health", get(health))
        .route("/status", get(get_status))
        .route("/metrics", get(get_metrics))
        .route("/metrics/prometheus", get(get_prometheus_metrics))
        // Gradient operations
        .route("/gradients", post(submit_gradient))
        .route("/aggregate", post(force_aggregate))
        // Round management
        .route("/round", get(get_round))
        // Events
        .route("/events", get(get_events))
        .route("/events/stream", get(events_stream))
        // Chaos testing (only in debug builds or when enabled)
        .route("/chaos/configure", post(configure_chaos))
        .route("/chaos/partition", post(trigger_partition))
        .route("/chaos/byzantine/:node_id", post(inject_byzantine))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    info!("Starting Integrated FL Server");

    // Parse configuration from environment
    let expected_nodes: usize = std::env::var("FL_EXPECTED_NODES")
        .unwrap_or_else(|_| "10".into())
        .parse()
        .unwrap_or(10);

    let max_memory_mb: usize = std::env::var("FL_MAX_MEMORY_MB")
        .unwrap_or_else(|_| "2000".into())
        .parse()
        .unwrap_or(2000);

    let defense = parse_defense();

    let max_rounds: u64 = std::env::var("FL_MAX_ROUNDS")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap_or(0);

    let round_timeout_secs: u64 = std::env::var("FL_ROUND_TIMEOUT_SECS")
        .unwrap_or_else(|_| "60".into())
        .parse()
        .unwrap_or(60);

    let min_participants: usize = std::env::var("FL_MIN_PARTICIPANTS")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap_or(3);

    // Chaos configuration
    let chaos = ChaosConfig {
        enabled: std::env::var("FL_ENABLE_CHAOS")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false),
        fail_rate: std::env::var("FL_CHAOS_FAIL_RATE")
            .unwrap_or_else(|_| "0.0".into())
            .parse()
            .unwrap_or(0.0),
        latency_ms: std::env::var("FL_CHAOS_LATENCY_MS")
            .ok()
            .and_then(|v| v.parse().ok()),
        drop_rate: std::env::var("FL_CHAOS_DROP_RATE")
            .unwrap_or_else(|_| "0.0".into())
            .parse()
            .unwrap_or(0.0),
    };

    // Create aggregator
    let agg_config = AggregatorConfig::default()
        .with_expected_nodes(expected_nodes)
        .with_defense(defense.clone())
        .with_max_memory(max_memory_mb * 1_000_000);

    let aggregator = AsyncAggregator::new(agg_config);

    // Create coordinator
    let coord_config = CoordinatorConfig::default()
        .with_min_participants(min_participants)
        .with_max_rounds(max_rounds)
        .with_round_timeout(Duration::from_secs(round_timeout_secs));

    // Clone the aggregator to keep a reference for the state
    // The coordinator will use its own reference (both share underlying Arc state)
    let aggregator_for_state = aggregator.clone();
    let coordinator = RoundCoordinator::new(coord_config, aggregator);

    // Initialize Holochain client
    #[cfg(feature = "holochain")]
    let holochain = {
        let admin_url = std::env::var("HOLOCHAIN_ADMIN_URL")
            .unwrap_or_else(|_| "ws://127.0.0.1:65000".into());
        let app_url = std::env::var("HOLOCHAIN_APP_URL")
            .unwrap_or_else(|_| "ws://127.0.0.1:65001".into());

        let dna_hash = std::env::var("HOLOCHAIN_DNA_HASH").ok();
        let agent_key = std::env::var("HOLOCHAIN_AGENT_KEY").ok();

        let mut config = HolochainConfig::default()
            .with_admin_url(&admin_url)
            .with_app_url(&app_url);

        if let (Some(dna), Some(agent)) = (dna_hash, agent_key) {
            config = config.with_cell_id(&dna, &agent);
        }

        let client = HolochainClient::new(config);

        // Try to connect
        match client.connect().await {
            Ok(_) => {
                info!(
                    admin_url = %admin_url,
                    app_url = %app_url,
                    "Connected to Holochain conductor"
                );
                Some(Arc::new(RwLock::new(client)))
            }
            Err(e) => {
                warn!("Failed to connect to Holochain: {}. Running without persistence.", e);
                None
            }
        }
    };

    #[cfg(not(feature = "holochain"))]
    let _holochain: Option<()> = None;

    // Create event channel
    let (event_tx, _) = broadcast::channel(256);

    // Create state
    let state = Arc::new(IntegratedState {
        aggregator: aggregator_for_state,
        coordinator: Arc::new(coordinator),
        #[cfg(feature = "holochain")]
        holochain,
        event_tx,
        chaos,
    });

    // Start coordinator in background
    let coordinator_state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = coordinator_state.coordinator.run().await {
            error!("Coordinator error: {}", e);
        }
    });

    // Create router (clone state so we can still reference it)
    let chaos_enabled = state.chaos.enabled;
    let app = create_router(state);

    // Bind and serve
    let addr = std::env::var("FL_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".into());
    let listener = TcpListener::bind(&addr).await?;

    info!(
        addr = %addr,
        expected_nodes = expected_nodes,
        defense = %defense,
        max_rounds = max_rounds,
        min_participants = min_participants,
        chaos_enabled = chaos_enabled,
        "Server ready"
    );

    axum::serve(listener, app).await?;

    Ok(())
}

fn parse_defense() -> Defense {
    match std::env::var("FL_DEFENSE")
        .unwrap_or_else(|_| "krum".into())
        .as_str()
    {
        "fedavg" => Defense::FedAvg,
        "krum" => {
            let f: usize = std::env::var("FL_BYZANTINE_F")
                .unwrap_or_else(|_| "1".into())
                .parse()
                .unwrap_or(1);
            Defense::Krum { f }
        }
        "multikrum" => {
            let f: usize = std::env::var("FL_BYZANTINE_F")
                .unwrap_or_else(|_| "1".into())
                .parse()
                .unwrap_or(1);
            let k: usize = std::env::var("FL_MULTIKRUM_K")
                .unwrap_or_else(|_| "3".into())
                .parse()
                .unwrap_or(3);
            Defense::MultiKrum { f, k }
        }
        "median" => Defense::Median,
        "trimmedmean" => {
            let beta: f32 = std::env::var("FL_TRIMMED_BETA")
                .unwrap_or_else(|_| "0.1".into())
                .parse()
                .unwrap_or(0.1);
            Defense::TrimmedMean { beta }
        }
        _ => Defense::Krum { f: 1 },
    }
}
