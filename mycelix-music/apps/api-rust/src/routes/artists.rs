//! Artist Routes - Artist profiles and catalog

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;
use std::sync::Arc;

use crate::AppState;
use super::songs::Song;

#[derive(Debug, Serialize)]
pub struct ArtistProfile {
    pub address: String,
    pub total_songs: i64,
    pub total_plays: i64,
    pub total_earnings: f64,
    pub strategies_used: Vec<String>,
}

/// Get artist profile
pub async fn get_artist(
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<ArtistProfile>, StatusCode> {
    let stats = sqlx::query_as::<_, (i64, i64, f64)>(
        r#"
        SELECT
            COUNT(*) as total_songs,
            COALESCE(SUM(plays), 0) as total_plays,
            COALESCE(SUM(earnings), 0)::float8 as total_earnings
        FROM songs
        WHERE artist_address = $1
        "#,
    )
    .bind(&address)
    .fetch_one(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to get artist stats: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let strategies: Vec<String> = sqlx::query_scalar(
        r#"
        SELECT DISTINCT strategy_id FROM songs WHERE artist_address = $1
        "#,
    )
    .bind(&address)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to get artist strategies: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(ArtistProfile {
        address,
        total_songs: stats.0,
        total_plays: stats.1,
        total_earnings: stats.2,
        strategies_used: strategies,
    }))
}

/// Get songs by artist
pub async fn get_artist_songs(
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<Vec<Song>>, StatusCode> {
    let songs = sqlx::query_as::<_, Song>(
        r#"
        SELECT id, song_hash, title, artist_address, ipfs_hash,
               strategy_id, payment_model, plays, earnings::float8 as earnings, created_at
        FROM songs
        WHERE artist_address = $1
        ORDER BY created_at DESC
        "#,
    )
    .bind(&address)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to get artist songs: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(songs))
}
