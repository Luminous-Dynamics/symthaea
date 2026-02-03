//! Symthaea Bridge
//!
//! Connects Symthaea LLM inference to civic-happ DHT knowledge for
//! AI-augmented civic assistance.
//!
//! Architecture:
//! ```
//! SMS Gateway -> Symthaea Bridge -> Civic hApp (knowledge)
//!                                -> Symthaea Inference (LLM)
//!                                -> Civic hApp (reputation)
//! ```

use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

mod civic;
mod inference;
mod prompt;

use civic::CivicClient;
use inference::InferenceClient;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Clone, Debug)]
pub struct Config {
    /// Holochain conductor WebSocket URL
    pub conductor_url: String,
    /// Symthaea inference server URL
    pub inference_url: String,
    /// Server bind address
    pub bind_addr: String,
    /// Default model to use
    pub default_model: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            conductor_url: std::env::var("HOLOCHAIN_URL")
                .unwrap_or_else(|_| "ws://localhost:8888".to_string()),
            inference_url: std::env::var("INFERENCE_URL")
                .unwrap_or_else(|_| "http://localhost:3000".to_string()),
            bind_addr: std::env::var("BIND_ADDR")
                .unwrap_or_else(|_| "0.0.0.0:3030".to_string()),
            default_model: std::env::var("DEFAULT_MODEL")
                .unwrap_or_else(|_| "phi3-mini".to_string()),
        }
    }
}

// ============================================================================
// Application State
// ============================================================================

pub struct AppState {
    civic: Option<CivicClient>,
    inference: InferenceClient,
    config: Config,
}

impl AppState {
    async fn new(config: Config) -> Self {
        // Try to connect to Holochain
        let civic = match CivicClient::connect(&config.conductor_url).await {
            Ok(client) => {
                info!("Connected to Holochain conductor");
                Some(client)
            }
            Err(e) => {
                warn!("Failed to connect to Holochain: {}. Running in inference-only mode.", e);
                None
            }
        };

        // Create inference client
        let inference = InferenceClient::new(&config.inference_url);

        Self {
            civic,
            inference,
            config,
        }
    }
}

type SharedState = Arc<RwLock<AppState>>;

// ============================================================================
// API Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AskRequest {
    /// The citizen's question
    pub question: String,
    /// Optional location (zip code, city, state)
    pub location: Option<String>,
    /// Optional domain hint
    pub domain: Option<String>,
    /// Conversation ID for context
    pub conversation_id: Option<String>,
    /// Agent public key (for reputation tracking)
    pub agent_pubkey: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AskResponse {
    /// Generated response
    pub response: String,
    /// Sources used (from DHT)
    pub sources: Vec<SourceInfo>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Whether response used live DHT data
    pub used_dht: bool,
    /// Conversation ID
    pub conversation_id: String,
}

#[derive(Debug, Serialize)]
pub struct SourceInfo {
    pub title: String,
    pub domain: String,
    pub source: Option<String>,
    pub contact_phone: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct FeedbackRequest {
    pub conversation_id: String,
    pub agent_pubkey: String,
    pub feedback_type: FeedbackType,
    pub context: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackType {
    Helpful,
    NotHelpful,
    Accurate,
    Inaccurate,
}

#[derive(Debug, Serialize)]
pub struct FeedbackResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub holochain_connected: bool,
    pub inference_available: bool,
    pub version: String,
}

// ============================================================================
// Handlers
// ============================================================================

async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    let state = state.read().await;

    // Check inference availability
    let inference_available = state.inference.health_check().await.unwrap_or(false);

    Json(HealthResponse {
        status: "ok".to_string(),
        holochain_connected: state.civic.is_some(),
        inference_available,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn ask(
    State(state): State<SharedState>,
    Json(request): Json<AskRequest>,
) -> Result<Json<AskResponse>, (StatusCode, String)> {
    let state = state.read().await;
    let conversation_id = request.conversation_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Step 1: Search for relevant civic knowledge
    let mut sources = Vec::new();
    let mut knowledge_context = String::new();
    let mut used_dht = false;

    if let Some(ref civic) = state.civic {
        match civic.search_knowledge(&request.question, request.location.as_deref(), request.domain.as_deref()).await {
            Ok(results) => {
                used_dht = true;
                for result in results.iter().take(3) {
                    // Build context from knowledge
                    knowledge_context.push_str(&format!(
                        "- {}: {}\n",
                        result.title,
                        result.content.chars().take(200).collect::<String>()
                    ));

                    sources.push(SourceInfo {
                        title: result.title.clone(),
                        domain: result.domain.clone(),
                        source: result.source.clone(),
                        contact_phone: result.contact_phone.clone(),
                    });
                }
            }
            Err(e) => {
                warn!("Failed to search civic knowledge: {}", e);
            }
        }
    }

    // Step 2: Construct prompt with civic context
    let prompt = prompt::build_civic_prompt(
        &request.question,
        &knowledge_context,
        request.location.as_deref(),
    );

    // Step 3: Generate response
    let response = match state.inference.generate(&prompt, None).await {
        Ok(text) => text,
        Err(e) => {
            error!("Inference failed: {}", e);
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Inference service unavailable: {}", e),
            ));
        }
    };

    // Calculate confidence based on sources and response quality
    let confidence = if used_dht && !sources.is_empty() {
        0.85
    } else if used_dht {
        0.6
    } else {
        0.4
    };

    Ok(Json(AskResponse {
        response,
        sources,
        confidence,
        used_dht,
        conversation_id,
    }))
}

async fn submit_feedback(
    State(state): State<SharedState>,
    Json(request): Json<FeedbackRequest>,
) -> Result<Json<FeedbackResponse>, (StatusCode, String)> {
    let state = state.read().await;

    let Some(ref civic) = state.civic else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Holochain not connected".to_string(),
        ));
    };

    // Record the feedback event
    let event_type = match request.feedback_type {
        FeedbackType::Helpful => "helpful",
        FeedbackType::NotHelpful => "not_helpful",
        FeedbackType::Accurate => "accurate",
        FeedbackType::Inaccurate => "inaccurate",
    };

    match civic.record_reputation_event(
        &request.agent_pubkey,
        event_type,
        request.context.as_deref(),
        Some(&request.conversation_id),
    ).await {
        Ok(_) => Ok(Json(FeedbackResponse {
            success: true,
            message: "Feedback recorded".to_string(),
        })),
        Err(e) => {
            error!("Failed to record feedback: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to record feedback: {}", e),
            ))
        }
    }
}

async fn get_agent_trust(
    State(state): State<SharedState>,
    axum::extract::Path(agent_pubkey): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let state = state.read().await;

    let Some(ref civic) = state.civic else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Holochain not connected".to_string(),
        ));
    };

    match civic.get_trust_score(&agent_pubkey).await {
        Ok(Some(score)) => Ok(Json(serde_json::json!({
            "agent_pubkey": agent_pubkey,
            "quality": score.quality,
            "consistency": score.consistency,
            "reputation": score.reputation,
            "composite": score.composite,
            "confidence": score.confidence,
            "is_trustworthy": score.composite >= 0.55 && score.confidence >= 0.3,
        }))),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            "Agent not found".to_string(),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to get trust score: {}", e),
        )),
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("symthaea_bridge=info".parse()?)
        )
        .init();

    let config = Config::default();
    info!("Starting Symthaea Bridge");
    info!("  Holochain URL: {}", config.conductor_url);
    info!("  Inference URL: {}", config.inference_url);
    info!("  Bind address: {}", config.bind_addr);

    // Create application state
    let state = Arc::new(RwLock::new(AppState::new(config.clone()).await));

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/ask", post(ask))
        .route("/feedback", post(submit_feedback))
        .route("/trust/:agent_pubkey", get(get_agent_trust))
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    info!("Symthaea Bridge listening on {}", config.bind_addr);

    axum::serve(listener, app).await?;

    Ok(())
}
