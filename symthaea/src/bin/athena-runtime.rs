// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Athena Runtime — AI Triage Service

use axum::{Json, Router, extract::State, routing::post};
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use symthaea::intelligence::athena::{AthenaAgent, AthenaConfig, TriageTicket};
use tracing::{error, info};

struct AppState {
    athena: AthenaAgent,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    info!("Starting Athena L1 Runtime...");

    let sandbox_root =
        env::var("ATHENA_SANDBOX_ROOT").unwrap_or_else(|_| "/tmp/athena/sandbox".to_string());
    let ollama_model = env::var("ATHENA_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string());
    let port = env::var("ATHENA_PORT").unwrap_or_else(|_| "8430".to_string());

    let config = AthenaConfig {
        sandbox_root: sandbox_root.into(),
        ollama_model,
    };

    let athena = AthenaAgent::new(config);
    let state = Arc::new(AppState { athena });

    let app = Router::new()
        .route("/v1/triage", post(handle_triage))
        .with_state(state);

    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
    info!("Athena Ticket API listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_triage(
    State(state): State<Arc<AppState>>,
    Json(ticket): Json<TriageTicket>,
) -> Result<Json<TriageTicket>, axum::http::StatusCode> {
    info!(id = %ticket.id, "Received triage request");

    match state.athena.triage(ticket).await {
        Ok(triaged) => Ok(Json(triaged)),
        Err(err) => {
            error!(error = %err, "Triage failed");
            Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
