// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HTTP coordinator wrapper with Prometheus metrics.

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use metrics::{counter, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use mycelix_core::{
    agent_registry::{AgentRegistry, ArchiveEntry as RegistryArchiveEntry},
    IpfsError, IpfsUploader, LossDeltaInput, ModelUpdate, ProofService,
};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
struct AppState {
    registry: Arc<AgentRegistry>,
    ipfs: Option<IpfsUploader>,
    token: Option<String>,
    metrics: metrics_exporter_prometheus::PrometheusHandle,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let recorder = PrometheusBuilder::new().install_recorder()?;

    let registry_path =
        std::env::var("MYCELIX_REGISTRY_PATH").unwrap_or_else(|_| "agent_registry.json".into());
    let registry = Arc::new(AgentRegistry::new(registry_path));
    let ipfs = std::env::var("MYCELIX_IPFS_API")
        .ok()
        .and_then(|endpoint| IpfsUploader::new(endpoint).ok());
    let token = std::env::var("MYCELIX_COORDINATOR_TOKEN").ok();

    let state = Arc::new(AppState {
        registry,
        ipfs,
        token,
        metrics: recorder,
    });

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/metrics", get(metrics_endpoint))
        .route("/api/agents/:id/archives", get(list_archives))
        .route("/api/validate_update", post(validate_update))
        .with_state(state);

    let addr: SocketAddr = std::env::var("MYCELIX_HTTP_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".into())
        .parse()
        .expect("invalid MYCELIX_HTTP_ADDR");
    tracing::info!("HTTP coordinator listening on http://{addr}");
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;
    Ok(())
}

async fn validate_update(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(update): Json<ModelUpdate>,
) -> impl IntoResponse {
    if let Err(resp) = authorize(&state.token, &headers) {
        return resp;
    }
    counter!("http_validate_requests_total", 1);
    let start = Instant::now();

    match perform_validation(&state, &update).await {
        Ok(()) => {
            histogram!(
                "http_validate_latency_seconds",
                start.elapsed().as_secs_f64(),
                "status" => "success"
            );
            (
                StatusCode::OK,
                Json(json!({
                    "success": true,
                    "valid": true,
                    "agent_id": update.agent_id,
                    "round_id": update.round_id
                })),
            )
        }
        Err(err) => {
            histogram!(
                "http_validate_latency_seconds",
                start.elapsed().as_secs_f64(),
                "status" => "error"
            );
            tracing::warn!("validation error: {}", err);
            (
                StatusCode::BAD_REQUEST,
                Json(json!({ "success": false, "error": err.to_string() })),
            )
        }
    }
}

async fn perform_validation(
    state: &AppState,
    update: &ModelUpdate,
) -> Result<(), ValidationError> {
    if update.schema_version != ModelUpdate::SCHEMA_VERSION {
        return Err(ValidationError::SchemaMismatch);
    }
    let record = state
        .registry
        .get(&update.agent_id)
        .await
        .ok_or(ValidationError::UnknownAgent)?;
    if record.public_key != update.public_key {
        return Err(ValidationError::KeyMismatch);
    }
    if record.schema_version != update.schema_version {
        return Err(ValidationError::SchemaMismatch);
    }
    if !update.verify() {
        return Err(ValidationError::InvalidSignature);
    }
    if !weights_within_threshold(update) {
        return Err(ValidationError::OutOfRange);
    }

    if let Some(client) = &state.ipfs {
        let registry = Arc::clone(&state.registry);
        let update_clone = update.clone();
        let uploader = client.clone();
        tokio::spawn(async move {
            if let Err(err) = archive_update_to_ipfs(uploader, registry, &update_clone).await {
                tracing::warn!("failed to archive update: {}", err);
            }
        });
    }

    Ok(())
}

async fn list_archives(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(agent_id): Path<String>,
) -> impl IntoResponse {
    if let Err(resp) = authorize(&state.token, &headers) {
        return resp;
    }
    match state.registry.get(&agent_id).await {
        Some(record) => (
            StatusCode::OK,
            Json(json!({ "agent_id": agent_id, "archives": record.archives })),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "agent not found" })),
        ),
    }
}

async fn health() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

async fn metrics_endpoint(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
}

fn authorize(
    expected: &Option<String>,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if let Some(token) = expected {
        if let Some(value) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
            if value == format!("Bearer {}", token) {
                return Ok(());
            }
        }
        Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "unauthorized" })),
        ))
    } else {
        Ok(())
    }
}

async fn archive_update_to_ipfs(
    client: IpfsUploader,
    registry: Arc<AgentRegistry>,
    update: &ModelUpdate,
) -> Result<(), ArchiveError> {
    let proof_input = loss_input_for_update(update);
    let proof = ProofService::prove_loss_delta(&proof_input)
        .map_err(|e| ArchiveError::Proof(e.to_string()))?;
    let payload = serde_json::to_vec(&json!({
        "update": update,
        "proof": proof,
    }))?;
    let filename = format!("update-{}-{}.json", update.agent_id, update.round_id);
    let cid = client.upload_bytes(payload, filename).await?;
    registry
        .record_archive(&update.agent_id, RegistryArchiveEntry::new(update.round_id, &cid))
        .await?;
    counter!("archive_success_total", 1);
    Ok(())
}

fn loss_input_for_update(update: &ModelUpdate) -> LossDeltaInput {
    let mut hasher = blake3::Hasher::new();
    for weight in &update.weights {
        hasher.update(&weight.to_le_bytes());
    }
    LossDeltaInput {
        round_id: update.round_id,
        model_hash: hasher.finalize().to_hex().to_string(),
        baseline_loss: update.weights.first().copied().unwrap_or(0.5).abs(),
        new_loss: update.weights.last().copied().unwrap_or(0.4).abs(),
        tolerance: 1.0,
    }
}

fn weights_within_threshold(update: &ModelUpdate) -> bool {
    const VALIDATION_THRESHOLD: f32 = 2.0;
    update
        .weights
        .iter()
        .all(|weight| weight.abs() <= VALIDATION_THRESHOLD)
}

#[derive(Debug)]
enum ValidationError {
    UnknownAgent,
    KeyMismatch,
    SchemaMismatch,
    InvalidSignature,
    OutOfRange,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::UnknownAgent => write!(f, "agent not registered"),
            ValidationError::KeyMismatch => write!(f, "public key mismatch"),
            ValidationError::SchemaMismatch => write!(f, "schema version mismatch"),
            ValidationError::InvalidSignature => write!(f, "invalid signature"),
            ValidationError::OutOfRange => write!(f, "weights outside threshold"),
        }
    }
}

impl std::error::Error for ValidationError {}

#[derive(Debug)]
enum ArchiveError {
    Serialize(serde_json::Error),
    Proof(String),
    Ipfs(IpfsError),
    Registry(mycelix_core::agent_registry::AgentRegistryError),
}

impl std::fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchiveError::Serialize(err) => write!(f, "serialization error: {}", err),
            ArchiveError::Proof(err) => write!(f, "proof error: {}", err),
            ArchiveError::Ipfs(err) => write!(f, "ipfs error: {}", err),
            ArchiveError::Registry(err) => write!(f, "registry error: {}", err),
        }
    }
}

impl std::error::Error for ArchiveError {}

impl From<serde_json::Error> for ArchiveError {
    fn from(value: serde_json::Error) -> Self {
        ArchiveError::Serialize(value)
    }
}

impl From<IpfsError> for ArchiveError {
    fn from(value: IpfsError) -> Self {
        ArchiveError::Ipfs(value)
    }
}

impl From<mycelix_core::agent_registry::AgentRegistryError> for ArchiveError {
    fn from(value: mycelix_core::agent_registry::AgentRegistryError) -> Self {
        ArchiveError::Registry(value)
    }
}
