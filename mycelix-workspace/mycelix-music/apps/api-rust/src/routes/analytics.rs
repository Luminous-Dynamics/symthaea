// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Analytics Routes - Earnings and performance data

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::AppState;

#[derive(Debug, Serialize)]
pub struct ArtistAnalytics {
    pub address: String,
    pub total_earnings: f64,
    pub total_plays: i64,
    pub avg_earnings_per_play: f64,
    pub top_songs: Vec<SongSummary>,
    pub earnings_by_strategy: Vec<StrategyEarnings>,
    pub recent_plays: Vec<RecentPlay>,
}

#[derive(Debug, Serialize)]
pub struct SongSummary {
    pub id: Uuid,
    pub title: String,
    pub plays: i64,
    pub earnings: f64,
}

#[derive(Debug, Serialize)]
pub struct StrategyEarnings {
    pub strategy_id: String,
    pub total_earnings: f64,
    pub song_count: i64,
}

#[derive(Debug, Serialize)]
pub struct RecentPlay {
    pub song_title: String,
    pub listener: String,
    pub amount: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct SongAnalytics {
    pub id: Uuid,
    pub title: String,
    pub total_plays: i64,
    pub total_earnings: f64,
    pub unique_listeners: i64,
    pub avg_tip: f64,
    pub plays_by_day: Vec<DailyPlays>,
}

#[derive(Debug, Serialize)]
pub struct DailyPlays {
    pub date: String,
    pub plays: i64,
    pub earnings: f64,
}

#[derive(Debug, Deserialize)]
pub struct TopSongsQuery {
    pub limit: Option<i64>,
    pub period: Option<String>, // "day", "week", "month", "all"
}

/// Get artist analytics
pub async fn artist_analytics(
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<ArtistAnalytics>, StatusCode> {
    // Get totals
    let totals = sqlx::query_as::<_, (f64, i64)>(
        r#"
        SELECT
            COALESCE(SUM(earnings), 0)::float8 as total_earnings,
            COALESCE(SUM(plays), 0) as total_plays
        FROM songs
        WHERE artist_address = $1
        "#,
    )
    .bind(&address)
    .fetch_one(&state.db_pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Get top songs
    let top_songs = sqlx::query_as::<_, (Uuid, String, i64, f64)>(
        r#"
        SELECT id, title, plays, earnings::float8
        FROM songs
        WHERE artist_address = $1
        ORDER BY earnings DESC
        LIMIT 5
        "#,
    )
    .bind(&address)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .into_iter()
    .map(|(id, title, plays, earnings)| SongSummary { id, title, plays, earnings })
    .collect();

    // Get earnings by strategy
    let earnings_by_strategy = sqlx::query_as::<_, (String, f64, i64)>(
        r#"
        SELECT strategy_id, SUM(earnings)::float8 as total, COUNT(*) as count
        FROM songs
        WHERE artist_address = $1
        GROUP BY strategy_id
        "#,
    )
    .bind(&address)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .into_iter()
    .map(|(strategy_id, total_earnings, song_count)| StrategyEarnings {
        strategy_id,
        total_earnings,
        song_count,
    })
    .collect();

    let avg = if totals.1 > 0 {
        totals.0 / totals.1 as f64
    } else {
        0.0
    };

    // Get recent plays from the plays table
    let recent_plays = sqlx::query_as::<_, (String, String, f64, chrono::DateTime<chrono::Utc>)>(
        r#"
        SELECT s.title, p.listener_address, p.amount::float8, p.timestamp
        FROM plays p
        JOIN songs s ON s.id = p.song_id
        WHERE s.artist_address = $1
        ORDER BY p.timestamp DESC
        LIMIT 20
        "#,
    )
    .bind(&address)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .into_iter()
    .map(|(song_title, listener, amount, timestamp)| RecentPlay {
        song_title,
        listener,
        amount,
        timestamp,
    })
    .collect();

    Ok(Json(ArtistAnalytics {
        address,
        total_earnings: totals.0,
        total_plays: totals.1,
        avg_earnings_per_play: avg,
        top_songs,
        earnings_by_strategy,
        recent_plays,
    }))
}

/// Get song analytics
pub async fn song_analytics(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SongAnalytics>, StatusCode> {
    let song = sqlx::query_as::<_, (String, i64, f64)>(
        r#"
        SELECT title, plays, earnings::float8
        FROM songs
        WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .ok_or(StatusCode::NOT_FOUND)?;

    // Get unique listeners
    let unique_listeners: i64 = sqlx::query_scalar(
        "SELECT COUNT(DISTINCT listener_address) FROM plays WHERE song_id = $1",
    )
    .bind(id)
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(0);

    // Get plays by day time series (last 30 days)
    let plays_by_day = sqlx::query_as::<_, (String, i64, f64)>(
        r#"
        SELECT
            TO_CHAR(DATE_TRUNC('day', timestamp), 'YYYY-MM-DD') as date,
            COUNT(*) as plays,
            COALESCE(SUM(amount), 0)::float8 as earnings
        FROM plays
        WHERE song_id = $1
            AND timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY DATE_TRUNC('day', timestamp)
        ORDER BY DATE_TRUNC('day', timestamp) DESC
        "#,
    )
    .bind(id)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .into_iter()
    .map(|(date, plays, earnings)| DailyPlays { date, plays, earnings })
    .collect();

    Ok(Json(SongAnalytics {
        id,
        title: song.0,
        total_plays: song.1,
        total_earnings: song.2,
        unique_listeners,
        avg_tip: if song.1 > 0 { song.2 / song.1 as f64 } else { 0.0 },
        plays_by_day,
    }))
}

/// Get top songs globally
pub async fn top_songs(
    State(state): State<Arc<AppState>>,
    Query(params): Query<TopSongsQuery>,
) -> Result<Json<Vec<SongSummary>>, StatusCode> {
    let limit = params.limit.unwrap_or(20).min(100);

    let songs = sqlx::query_as::<_, (Uuid, String, i64, f64)>(
        r#"
        SELECT id, title, plays, earnings::float8
        FROM songs
        ORDER BY plays DESC
        LIMIT $1
        "#,
    )
    .bind(limit)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .into_iter()
    .map(|(id, title, plays, earnings)| SongSummary { id, title, plays, earnings })
    .collect();

    Ok(Json(songs))
}
