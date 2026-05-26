// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust API handlers

use axum::{
    Json,
    extract::{Path, State},
};
use std::sync::Arc;
use tracing::info;

use crate::{
    error::{ApiError, Result},
    models::*,
    state::AppState,
};

/// Get trust score for a participant
#[utoipa::path(
    get,
    path = "/api/v1/trust/{participant}",
    responses(
        (status = 200, description = "Trust score retrieved", body = TrustScoreResponse),
        (status = 404, description = "Participant not found", body = ApiError),
    ),
    params(
        ("participant" = String, Path, description = "Participant identifier")
    ),
    tag = "trust"
)]
pub async fn get_trust_score(
    State(state): State<Arc<AppState>>,
    Path(participant): Path<String>,
) -> Result<Json<TrustScoreResponse>> {
    info!(participant = %participant, "Retrieving trust score");

    let trust_manager = state.trust_manager.read().await;

    let trust_score = trust_manager.get_score(&participant);

    let response = TrustScoreResponse {
        participant: participant.clone(),
        score: trust_score.score,
        last_updated: chrono::Utc::now(), // In a real implementation, this would come from the trust manager
    };

    Ok(Json(response))
}

/// Update trust score for a participant
#[utoipa::path(
    put,
    path = "/api/v1/trust/{participant}",
    request_body = UpdateTrustScoreRequest,
    responses(
        (status = 200, description = "Trust score updated", body = TrustScoreResponse),
        (status = 400, description = "Invalid request", body = ApiError),
    ),
    params(
        ("participant" = String, Path, description = "Participant identifier")
    ),
    tag = "trust"
)]
pub async fn update_trust_score(
    State(state): State<Arc<AppState>>,
    Path(participant): Path<String>,
    Json(req): Json<UpdateTrustScoreRequest>,
) -> Result<Json<TrustScoreResponse>> {
    info!(
        participant = %participant,
        delta = req.delta,
        "Updating trust score"
    );

    // Validate delta is reasonable
    if req.delta.abs() > 1.0 {
        return Err(ApiError::invalid_request(
            "Trust score delta must be between -1.0 and 1.0",
        ));
    }

    let mut trust_manager = state.trust_manager.write().await;
    let positive = req.delta > 0.0;
    let weight = req.delta.abs();
    let updated_score = trust_manager.update_score(&participant, positive, weight)?;

    info!(
        participant = %participant,
        new_score = updated_score.score,
        "Trust score updated"
    );

    let response = TrustScoreResponse {
        participant: participant.clone(),
        score: updated_score.score,
        last_updated: chrono::Utc::now(),
    };

    Ok(Json(response))
}

/// Get trust network statistics
#[utoipa::path(
    get,
    path = "/api/v1/trust/stats",
    responses(
        (status = 200, description = "Trust statistics retrieved", body = TrustStatsResponse),
    ),
    tag = "trust"
)]
pub async fn get_trust_stats(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<TrustStatsResponse>> {
    info!("Retrieving trust statistics");

    // TODO: Implement trust statistics calculation
    // For now, return placeholder data
    let response = TrustStatsResponse {
        total_participants: 0,
        average_score: 0.5,
        median_score: 0.5,
        highest_score: 1.0,
        lowest_score: 0.0,
    };

    Ok(Json(response))
}
