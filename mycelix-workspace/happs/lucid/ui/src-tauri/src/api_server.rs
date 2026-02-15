//! LUCID Local API Server
//!
//! Provides a local HTTP API for the browser extension to communicate with.
//! Listens on localhost:1420/api/* for POST requests.

use axum::{
    extract::State,
    http::{Method, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

// Note: SymthaeaState is managed by Tauri, not directly accessible here

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtPayload {
    pub content: String,
    pub thought_type: String,
    #[serde(default)]
    pub source_url: Option<String>,
    #[serde(default)]
    pub source_title: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ApiResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub online: bool,
    pub version: String,
    pub symthaea_ready: bool,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub thought_count: u64,
    pub today_captures: u64,
}

// ============================================================================
// Shared State
// ============================================================================

/// Callback type for emitting thought events to the frontend.
pub type ThoughtEventSender = Arc<dyn Fn(&ThoughtPayload) + Send + Sync>;

pub struct ApiState {
    pub thought_queue: Arc<Mutex<Vec<ThoughtPayload>>>,
    pub capture_count: Arc<Mutex<u64>>,
    pub symthaea_available: Arc<Mutex<bool>>,
    /// Optional callback to emit events when thoughts are captured.
    /// Set this to bridge API server events to Tauri/Holochain.
    pub event_sender: Option<ThoughtEventSender>,
}

impl ApiState {
    pub fn new() -> Self {
        Self {
            thought_queue: Arc::new(Mutex::new(Vec::new())),
            capture_count: Arc::new(Mutex::new(0)),
            symthaea_available: Arc::new(Mutex::new(false)),
            event_sender: None,
        }
    }

    /// Set the event sender for forwarding captured thoughts to the frontend.
    pub fn with_event_sender(mut self, sender: ThoughtEventSender) -> Self {
        self.event_sender = Some(sender);
        self
    }

    pub async fn set_symthaea_ready(&self, ready: bool) {
        let mut available = self.symthaea_available.lock().await;
        *available = ready;
    }
}

// ============================================================================
// Handlers
// ============================================================================

async fn status_handler(State(state): State<Arc<ApiState>>) -> Json<StatusResponse> {
    let symthaea_ready = *state.symthaea_available.lock().await;

    Json(StatusResponse {
        online: true,
        version: env!("CARGO_PKG_VERSION").to_string(),
        symthaea_ready,
    })
}

async fn create_thought_handler(
    State(state): State<Arc<ApiState>>,
    Json(payload): Json<ThoughtPayload>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    // Validate payload
    if payload.content.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                thought_id: None,
                error: Some("Content cannot be empty".to_string()),
            }),
        ));
    }

    // Queue the thought for processing
    let thought_id = uuid::Uuid::new_v4().to_string();

    {
        let mut queue = state.thought_queue.lock().await;
        queue.push(payload.clone());
    }

    {
        let mut count = state.capture_count.lock().await;
        *count += 1;
    }

    // Emit event to Tauri frontend for Holochain persistence
    if let Some(ref sender) = state.event_sender {
        sender(&payload);
    }

    tracing::info!(
        "Captured thought from browser extension: {} (type: {})",
        &payload.content[..payload.content.len().min(50)],
        payload.thought_type
    );

    Ok(Json(ApiResponse {
        success: true,
        thought_id: Some(thought_id),
        error: None,
    }))
}

async fn stats_handler(State(state): State<Arc<ApiState>>) -> Json<StatsResponse> {
    let count = {
        let queue = state.thought_queue.lock().await;
        queue.len() as u64
    };

    let today = {
        let count = state.capture_count.lock().await;
        *count
    };

    Json(StatsResponse {
        thought_count: count,
        today_captures: today,
    })
}

async fn get_pending_handler(
    State(state): State<Arc<ApiState>>,
) -> Json<Vec<ThoughtPayload>> {
    let queue = state.thought_queue.lock().await;
    Json(queue.clone())
}

async fn clear_pending_handler(
    State(state): State<Arc<ApiState>>,
) -> Json<ApiResponse> {
    let mut queue = state.thought_queue.lock().await;
    let count = queue.len();
    queue.clear();

    Json(ApiResponse {
        success: true,
        thought_id: None,
        error: Some(format!("Cleared {} pending thoughts", count)),
    })
}

// ============================================================================
// Server Setup
// ============================================================================

pub fn create_api_router(state: Arc<ApiState>) -> Router {
    // Configure CORS for browser extension
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    Router::new()
        .route("/api/status", get(status_handler))
        .route("/api/thoughts", post(create_thought_handler))
        .route("/api/thoughts/pending", get(get_pending_handler))
        .route("/api/thoughts/clear", post(clear_pending_handler))
        .route("/api/stats", get(stats_handler))
        .layer(cors)
        .with_state(state)
}

/// Start the API server in the background
pub async fn start_api_server(state: Arc<ApiState>) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_api_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:1420").await?;
    tracing::info!("LUCID API server listening on http://127.0.0.1:1420");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Start the API server in the background (non-blocking)
pub fn spawn_api_server() -> Arc<ApiState> {
    let state = Arc::new(ApiState::new());
    let state_clone = Arc::clone(&state);

    tokio::spawn(async move {
        if let Err(e) = start_api_server(state_clone).await {
            tracing::error!("API server error: {}", e);
        }
    });

    state
}
