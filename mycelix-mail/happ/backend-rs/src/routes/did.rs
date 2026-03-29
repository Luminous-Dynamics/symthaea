// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DID routes
//!
//! All DID operations go through the Holochain DHT.
//! This replaces the Python DID Registry entirely.

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

use crate::error::AppResult;
use crate::middleware::AuthenticatedUser;
use crate::routes::AppState;
use crate::types::{DidResolution, RegisterDidInput};

/// Create DID routes
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/register", post(register_did))
        .route("/resolve/:did", get(resolve_did))
        .route("/whoami", get(whoami))
}

/// Register a new DID on the DHT
///
/// This is the ONLY way to register DIDs - no external database.
/// The DID is bound to the current agent's public key.
async fn register_did(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Json(input): Json<RegisterDidInput>,
) -> AppResult<Json<DidRegistrationResponse>> {
    tracing::info!("Registering DID {} for agent", input.did);

    // Validate DID format
    validate_did_format(&input.did)?;

    // Register on DHT
    let hash = state.holochain.register_did(&input.did).await?;

    Ok(Json(DidRegistrationResponse {
        did: input.did,
        agent_pub_key: user.agent_pub_key,
        action_hash: BASE64.encode(hash.get_raw_39()),
        registered_on: "holochain_dht".to_string(),
    }))
}

/// Resolve a DID to an AgentPubKey via DHT
///
/// This queries the Holochain DHT for the DidBinding entry.
/// No external registry is needed.
async fn resolve_did(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Path(did): Path<String>,
) -> AppResult<Json<DidResolution>> {
    tracing::debug!("Resolving DID: {}", did);

    // Decode URL-encoded DID (colons are encoded as %3A)
    let did_decoded = urlencoding::decode(&did)
        .map_err(|_| crate::error::AppError::ValidationError("Invalid DID encoding".to_string()))?
        .into_owned();

    let agent_pubkey = state.holochain.resolve_did(&did_decoded).await?;

    Ok(Json(DidResolution {
        did: did_decoded,
        agent_pub_key: BASE64.encode(agent_pubkey.get_raw_39()),
        is_local: true, // Resolved from local DHT
    }))
}

/// Get current user's DID info
async fn whoami(
    State(_state): State<AppState>,
    user: AuthenticatedUser,
) -> AppResult<Json<WhoamiResponse>> {
    Ok(Json(WhoamiResponse {
        did: user.did,
        agent_pub_key: user.agent_pub_key,
    }))
}

// Helper functions

fn validate_did_format(did: &str) -> AppResult<()> {
    // Basic DID validation
    // Format: did:<method>:<method-specific-id>

    if !did.starts_with("did:") {
        return Err(crate::error::AppError::ValidationError(
            "DID must start with 'did:'".to_string(),
        ));
    }

    let parts: Vec<&str> = did.split(':').collect();
    if parts.len() < 3 {
        return Err(crate::error::AppError::ValidationError(
            "DID must have format 'did:<method>:<identifier>'".to_string(),
        ));
    }

    let method = parts[1];
    if method.is_empty() {
        return Err(crate::error::AppError::ValidationError(
            "DID method cannot be empty".to_string(),
        ));
    }

    // Allow common methods
    let valid_methods = ["mycelix", "key", "web", "ion", "ethr", "pkh"];
    if !valid_methods.contains(&method) {
        tracing::warn!("Non-standard DID method: {}", method);
        // Still allow it, just warn
    }

    Ok(())
}

// Response types

#[derive(serde::Serialize)]
struct DidRegistrationResponse {
    did: String,
    agent_pub_key: String,
    action_hash: String,
    registered_on: String,
}

#[derive(serde::Serialize)]
struct WhoamiResponse {
    did: String,
    agent_pub_key: String,
}
