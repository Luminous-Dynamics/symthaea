// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HTTP request handlers

use axum::{
    extract::{Multipart, Path, Query, State},
    response::Json,
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use mycelix_core::{codec, track};

use crate::{ApiError, ApiResponse, ApiResult, AppState, PaginationParams};

// === Track Handlers ===

/// Upload a new track
pub async fn upload_track(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> ApiResult<Json<ApiResponse<TrackResponse>>> {
    let mut file_data: Option<Vec<u8>> = None;
    let mut metadata = track::TrackMetadata::default();

    while let Some(field) = multipart.next_field().await.map_err(|e| ApiError::BadRequest(e.to_string()))? {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                file_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| ApiError::BadRequest(e.to_string()))?
                        .to_vec(),
                );
            }
            "title" => {
                metadata.title = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::BadRequest(e.to_string()))?,
                );
            }
            "artist" => {
                metadata.artist = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::BadRequest(e.to_string()))?,
                );
            }
            _ => {}
        }
    }

    let data = file_data.ok_or_else(|| ApiError::BadRequest("No file provided".to_string()))?;

    // Decode and create track
    let buffer = state.engine.decode(&data, "auto").await?;
    let track_id = Uuid::new_v4();
    let track = track::Track::with_metadata(track_id, buffer, metadata);

    Ok(Json(ApiResponse::success(TrackResponse {
        id: track.id,
        title: track.metadata.title.clone(),
        artist: track.metadata.artist.clone(),
        duration: track.duration(),
        sample_rate: track.sample_rate(),
        channels: track.channels(),
        analyzed: track.is_analyzed(),
    })))
}

/// Get track by ID
pub async fn get_track(
    State(_state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<TrackResponse>>> {
    // In production, would fetch from storage
    Err(ApiError::NotFound(format!("Track {} not found", id)))
}

/// Delete track
pub async fn delete_track(
    State(_state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<()>>> {
    // In production, would delete from storage
    Ok(Json(ApiResponse::success(())))
}

/// Analyze a track
pub async fn analyze_track(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<AnalysisResponse>>> {
    // In production, would fetch track and analyze
    // For now, return mock data
    Ok(Json(ApiResponse::success(AnalysisResponse {
        track_id: id,
        genre: "Electronic".to_string(),
        genre_confidence: 0.85,
        mood: "Energetic".to_string(),
        bpm: 128.0,
        key: "Am".to_string(),
        beats: vec![0.0, 0.468, 0.937, 1.406],
        segments: vec!["intro".to_string(), "verse".to_string()],
    })))
}

/// Stream track audio
pub async fn stream_track(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(params): Query<StreamParams>,
) -> ApiResult<Bytes> {
    // Would stream encoded audio
    Ok(Bytes::new())
}

#[derive(Debug, Deserialize)]
pub struct StreamParams {
    #[serde(default = "default_codec")]
    pub codec: String,
    #[serde(default = "default_quality")]
    pub quality: String,
}

fn default_codec() -> String {
    "opus".to_string()
}

fn default_quality() -> String {
    "high".to_string()
}

// === Search Handlers ===

/// Find similar tracks
pub async fn find_similar(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(params): Query<SimilarParams>,
) -> ApiResult<Json<ApiResponse<Vec<SimilarTrackResponse>>>> {
    let limit = params.limit.unwrap_or(10) as usize;

    // In production, would search index
    Ok(Json(ApiResponse::success(vec![])))
}

#[derive(Debug, Deserialize)]
pub struct SimilarParams {
    pub limit: Option<u32>,
}

/// Search tracks by query
pub async fn search_query(
    State(state): State<AppState>,
    Json(query): Json<SearchQuery>,
) -> ApiResult<Json<ApiResponse<Vec<SearchResultResponse>>>> {
    // In production, would search with filters
    Ok(Json(ApiResponse::success(vec![])))
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub text: Option<String>,
    pub genre: Option<String>,
    pub mood: Option<String>,
    pub bpm_min: Option<f32>,
    pub bpm_max: Option<f32>,
    pub key: Option<String>,
    pub limit: Option<u32>,
}

// === Session Handlers ===

/// Create a new session
pub async fn create_session(
    State(state): State<AppState>,
) -> ApiResult<Json<ApiResponse<SessionResponse>>> {
    let session_id = state.engine.create_session().await?;

    Ok(Json(ApiResponse::success(SessionResponse {
        id: session_id,
        status: "created".to_string(),
    })))
}

/// Get session info
pub async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<SessionDetailResponse>>> {
    let session = state.engine.get_session(id)?;
    let session_state = session.state();

    Ok(Json(ApiResponse::success(SessionDetailResponse {
        id,
        current_track: session_state.current_track,
        position: session_state.position,
        playback: format!("{:?}", session_state.playback),
        volume: session_state.volume,
        peers: session_state.peers,
    })))
}

/// Close a session
pub async fn close_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<()>>> {
    state.engine.close_session(id).await?;
    Ok(Json(ApiResponse::success(())))
}

/// Play in session
pub async fn session_play(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<()>>> {
    let session = state.engine.get_session(id)?;
    session.play();
    Ok(Json(ApiResponse::success(())))
}

/// Pause in session
pub async fn session_pause(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<()>>> {
    let session = state.engine.get_session(id)?;
    session.pause();
    Ok(Json(ApiResponse::success(())))
}

/// Seek in session
pub async fn session_seek(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(body): Json<SeekRequest>,
) -> ApiResult<Json<ApiResponse<()>>> {
    let session = state.engine.get_session(id)?;
    session.seek(body.position);
    Ok(Json(ApiResponse::success(())))
}

#[derive(Debug, Deserialize)]
pub struct SeekRequest {
    pub position: f32,
}

// === Analysis Handlers ===

/// Analyze uploaded audio
pub async fn analyze_audio(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> ApiResult<Json<ApiResponse<AnalysisResponse>>> {
    let mut file_data: Option<Vec<u8>> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| ApiError::BadRequest(e.to_string()))? {
        if field.name() == Some("file") {
            file_data = Some(
                field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::BadRequest(e.to_string()))?
                    .to_vec(),
            );
        }
    }

    let data = file_data.ok_or_else(|| ApiError::BadRequest("No file provided".to_string()))?;

    // Decode and analyze
    let buffer = state.engine.decode(&data, "auto").await?;
    let track_id = Uuid::new_v4();
    let mut track = track::Track::new(track_id, buffer);

    let analysis = state.engine.analyze_track(&mut track).await?;

    Ok(Json(ApiResponse::success(AnalysisResponse {
        track_id,
        genre: analysis.genre,
        genre_confidence: 0.85,
        mood: analysis.mood,
        bpm: analysis.bpm,
        key: analysis.key,
        beats: analysis.beats,
        segments: analysis.segments,
    })))
}

/// Batch analyze multiple files
pub async fn batch_analyze(
    State(_state): State<AppState>,
    Json(request): Json<BatchAnalyzeRequest>,
) -> ApiResult<Json<ApiResponse<Vec<AnalysisResponse>>>> {
    // Would process multiple tracks
    Ok(Json(ApiResponse::success(vec![])))
}

#[derive(Debug, Deserialize)]
pub struct BatchAnalyzeRequest {
    pub track_ids: Vec<Uuid>,
}

// === Codec Handlers ===

/// Transcode audio
pub async fn transcode(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> ApiResult<Bytes> {
    let mut file_data: Option<Vec<u8>> = None;
    let mut from_codec = "auto".to_string();
    let mut to_codec = "opus".to_string();
    let mut quality = "high".to_string();

    while let Some(field) = multipart.next_field().await.map_err(|e| ApiError::BadRequest(e.to_string()))? {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                file_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| ApiError::BadRequest(e.to_string()))?
                        .to_vec(),
                );
            }
            "from" => {
                from_codec = field.text().await.map_err(|e| ApiError::BadRequest(e.to_string()))?;
            }
            "to" => {
                to_codec = field.text().await.map_err(|e| ApiError::BadRequest(e.to_string()))?;
            }
            "quality" => {
                quality = field.text().await.map_err(|e| ApiError::BadRequest(e.to_string()))?;
            }
            _ => {}
        }
    }

    let data = file_data.ok_or_else(|| ApiError::BadRequest("No file provided".to_string()))?;

    let quality_preset = match quality.as_str() {
        "low" => codec::QualityPreset::Low,
        "medium" => codec::QualityPreset::Medium,
        "high" => codec::QualityPreset::High,
        "lossless" => codec::QualityPreset::Lossless,
        _ => codec::QualityPreset::High,
    };

    let transcoded = state
        .engine
        .transcode(&data, &from_codec, &to_codec, quality_preset)
        .await?;

    Ok(transcoded)
}

/// List supported formats
pub async fn list_formats() -> Json<ApiResponse<FormatsResponse>> {
    Json(ApiResponse::success(FormatsResponse {
        encoders: vec![
            FormatInfo {
                name: "opus".to_string(),
                description: "Opus audio codec".to_string(),
                extensions: vec!["opus".to_string(), "ogg".to_string()],
            },
            FormatInfo {
                name: "mp3".to_string(),
                description: "MP3 audio codec".to_string(),
                extensions: vec!["mp3".to_string()],
            },
            FormatInfo {
                name: "vorbis".to_string(),
                description: "Vorbis/OGG audio codec".to_string(),
                extensions: vec!["ogg".to_string()],
            },
        ],
        decoders: vec![
            FormatInfo {
                name: "opus".to_string(),
                description: "Opus audio codec".to_string(),
                extensions: vec!["opus".to_string()],
            },
            FormatInfo {
                name: "mp3".to_string(),
                description: "MP3 audio codec".to_string(),
                extensions: vec!["mp3".to_string()],
            },
            FormatInfo {
                name: "vorbis".to_string(),
                description: "Vorbis/OGG audio codec".to_string(),
                extensions: vec!["ogg".to_string()],
            },
            FormatInfo {
                name: "wav".to_string(),
                description: "WAV audio format".to_string(),
                extensions: vec!["wav".to_string()],
            },
            FormatInfo {
                name: "flac".to_string(),
                description: "FLAC lossless codec".to_string(),
                extensions: vec!["flac".to_string()],
            },
        ],
    }))
}

// === Response Types ===

#[derive(Debug, Serialize)]
pub struct TrackResponse {
    pub id: Uuid,
    pub title: Option<String>,
    pub artist: Option<String>,
    pub duration: f32,
    pub sample_rate: u32,
    pub channels: u8,
    pub analyzed: bool,
}

#[derive(Debug, Serialize)]
pub struct AnalysisResponse {
    pub track_id: Uuid,
    pub genre: String,
    pub genre_confidence: f32,
    pub mood: String,
    pub bpm: f32,
    pub key: String,
    pub beats: Vec<f32>,
    pub segments: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct SimilarTrackResponse {
    pub id: Uuid,
    pub title: Option<String>,
    pub artist: Option<String>,
    pub similarity: f32,
}

#[derive(Debug, Serialize)]
pub struct SearchResultResponse {
    pub id: Uuid,
    pub title: Option<String>,
    pub artist: Option<String>,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct SessionResponse {
    pub id: Uuid,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct SessionDetailResponse {
    pub id: Uuid,
    pub current_track: Option<Uuid>,
    pub position: f32,
    pub playback: String,
    pub volume: f32,
    pub peers: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct FormatsResponse {
    pub encoders: Vec<FormatInfo>,
    pub decoders: Vec<FormatInfo>,
}

#[derive(Debug, Serialize)]
pub struct FormatInfo {
    pub name: String,
    pub description: String,
    pub extensions: Vec<String>,
}
