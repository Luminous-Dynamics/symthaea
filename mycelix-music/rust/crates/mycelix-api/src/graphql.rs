// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GraphQL schema and resolvers

use std::sync::Arc;

use async_graphql::{
    Context, EmptySubscription, InputObject, Object, Schema, SimpleObject, ID,
};
use axum::{
    extract::Extension,
    response::{Html, IntoResponse},
    Json,
};
use uuid::Uuid;

use mycelix_core::MycelixEngine;

/// GraphQL schema type
pub type MycelixSchema = Schema<QueryRoot, MutationRoot, EmptySubscription>;

/// Create the GraphQL schema
pub fn create_schema(engine: Arc<MycelixEngine>) -> MycelixSchema {
    Schema::build(QueryRoot, MutationRoot, EmptySubscription)
        .data(engine)
        .finish()
}

/// GraphQL playground HTML
pub async fn graphql_playground() -> impl IntoResponse {
    Html(
        async_graphql::http::playground_source(async_graphql::http::GraphQLPlaygroundConfig::new("/graphql")),
    )
}

/// GraphQL handler
pub async fn graphql_handler(
    Extension(schema): Extension<MycelixSchema>,
    req: Json<async_graphql::Request>,
) -> Json<async_graphql::Response> {
    Json(schema.execute(req.0).await)
}

/// Query root
pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Get a track by ID
    async fn track(&self, ctx: &Context<'_>, id: ID) -> async_graphql::Result<Option<Track>> {
        let _engine = ctx.data::<Arc<MycelixEngine>>()?;
        let uuid = Uuid::parse_str(&id).map_err(|_| "Invalid UUID")?;

        // In production, would fetch from storage
        Ok(None)
    }

    /// List tracks with pagination
    async fn tracks(
        &self,
        ctx: &Context<'_>,
        #[graphql(default = 20)] limit: i32,
        #[graphql(default = 0)] offset: i32,
    ) -> async_graphql::Result<Vec<Track>> {
        let _engine = ctx.data::<Arc<MycelixEngine>>()?;

        // In production, would fetch from storage
        Ok(vec![])
    }

    /// Search for tracks
    async fn search_tracks(
        &self,
        ctx: &Context<'_>,
        query: SearchInput,
    ) -> async_graphql::Result<Vec<SearchResult>> {
        let _engine = ctx.data::<Arc<MycelixEngine>>()?;

        // In production, would search index
        Ok(vec![])
    }

    /// Find similar tracks
    async fn similar_tracks(
        &self,
        ctx: &Context<'_>,
        track_id: ID,
        #[graphql(default = 10)] limit: i32,
    ) -> async_graphql::Result<Vec<SimilarTrack>> {
        let _engine = ctx.data::<Arc<MycelixEngine>>()?;

        // In production, would search similarity index
        Ok(vec![])
    }

    /// Get a session by ID
    async fn session(&self, ctx: &Context<'_>, id: ID) -> async_graphql::Result<Option<Session>> {
        let engine = ctx.data::<Arc<MycelixEngine>>()?;
        let uuid = Uuid::parse_str(&id).map_err(|_| "Invalid UUID")?;

        match engine.get_session(uuid) {
            Ok(session) => {
                let state = session.state();
                Ok(Some(Session {
                    id: ID::from(uuid.to_string()),
                    current_track_id: state.current_track.map(|id| ID::from(id.to_string())),
                    position: state.position,
                    playback_state: format!("{:?}", state.playback),
                    volume: state.volume,
                    peer_count: state.peers.len() as i32,
                }))
            }
            Err(_) => Ok(None),
        }
    }

    /// List supported audio formats
    async fn formats(&self) -> Formats {
        Formats {
            encoders: vec![
                AudioFormat {
                    name: "opus".to_string(),
                    description: "Opus audio codec - excellent quality at low bitrates".to_string(),
                    mime_type: "audio/opus".to_string(),
                },
                AudioFormat {
                    name: "mp3".to_string(),
                    description: "MP3 audio codec - universal compatibility".to_string(),
                    mime_type: "audio/mpeg".to_string(),
                },
                AudioFormat {
                    name: "vorbis".to_string(),
                    description: "Vorbis/OGG audio codec".to_string(),
                    mime_type: "audio/ogg".to_string(),
                },
            ],
            decoders: vec![
                AudioFormat {
                    name: "opus".to_string(),
                    description: "Opus audio codec".to_string(),
                    mime_type: "audio/opus".to_string(),
                },
                AudioFormat {
                    name: "mp3".to_string(),
                    description: "MP3 audio codec".to_string(),
                    mime_type: "audio/mpeg".to_string(),
                },
                AudioFormat {
                    name: "vorbis".to_string(),
                    description: "Vorbis/OGG audio codec".to_string(),
                    mime_type: "audio/ogg".to_string(),
                },
                AudioFormat {
                    name: "wav".to_string(),
                    description: "WAV audio format".to_string(),
                    mime_type: "audio/wav".to_string(),
                },
                AudioFormat {
                    name: "flac".to_string(),
                    description: "FLAC lossless codec".to_string(),
                    mime_type: "audio/flac".to_string(),
                },
            ],
        }
    }

    /// Get system health status
    async fn health(&self) -> HealthStatus {
        HealthStatus {
            status: "healthy".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Mutation root
pub struct MutationRoot;

#[Object]
impl MutationRoot {
    /// Create a new session
    async fn create_session(&self, ctx: &Context<'_>) -> async_graphql::Result<Session> {
        let engine = ctx.data::<Arc<MycelixEngine>>()?;
        let session_id = engine.create_session().await?;

        Ok(Session {
            id: ID::from(session_id.to_string()),
            current_track_id: None,
            position: 0.0,
            playback_state: "Stopped".to_string(),
            volume: 1.0,
            peer_count: 0,
        })
    }

    /// Close a session
    async fn close_session(&self, ctx: &Context<'_>, id: ID) -> async_graphql::Result<bool> {
        let engine = ctx.data::<Arc<MycelixEngine>>()?;
        let uuid = Uuid::parse_str(&id).map_err(|_| "Invalid UUID")?;

        engine.close_session(uuid).await?;
        Ok(true)
    }

    /// Play in a session
    async fn session_play(&self, ctx: &Context<'_>, session_id: ID) -> async_graphql::Result<Session> {
        let engine = ctx.data::<Arc<MycelixEngine>>()?;
        let uuid = Uuid::parse_str(&session_id).map_err(|_| "Invalid UUID")?;

        let session = engine.get_session(uuid)?;
        session.play();

        let state = session.state();
        Ok(Session {
            id: session_id,
            current_track_id: state.current_track.map(|id| ID::from(id.to_string())),
            position: state.position,
            playback_state: "Playing".to_string(),
            volume: state.volume,
            peer_count: state.peers.len() as i32,
        })
    }

    /// Pause in a session
    async fn session_pause(&self, ctx: &Context<'_>, session_id: ID) -> async_graphql::Result<Session> {
        let engine = ctx.data::<Arc<MycelixEngine>>()?;
        let uuid = Uuid::parse_str(&session_id).map_err(|_| "Invalid UUID")?;

        let session = engine.get_session(uuid)?;
        session.pause();

        let state = session.state();
        Ok(Session {
            id: session_id,
            current_track_id: state.current_track.map(|id| ID::from(id.to_string())),
            position: state.position,
            playback_state: "Paused".to_string(),
            volume: state.volume,
            peer_count: state.peers.len() as i32,
        })
    }

    /// Seek to position in a session
    async fn session_seek(
        &self,
        ctx: &Context<'_>,
        session_id: ID,
        position: f32,
    ) -> async_graphql::Result<Session> {
        let engine = ctx.data::<Arc<MycelixEngine>>()?;
        let uuid = Uuid::parse_str(&session_id).map_err(|_| "Invalid UUID")?;

        let session = engine.get_session(uuid)?;
        session.seek(position);

        let state = session.state();
        Ok(Session {
            id: session_id,
            current_track_id: state.current_track.map(|id| ID::from(id.to_string())),
            position: state.position,
            playback_state: format!("{:?}", state.playback),
            volume: state.volume,
            peer_count: state.peers.len() as i32,
        })
    }

    /// Set volume in a session
    async fn session_set_volume(
        &self,
        ctx: &Context<'_>,
        session_id: ID,
        volume: f32,
    ) -> async_graphql::Result<Session> {
        let engine = ctx.data::<Arc<MycelixEngine>>()?;
        let uuid = Uuid::parse_str(&session_id).map_err(|_| "Invalid UUID")?;

        let session = engine.get_session(uuid)?;
        session.set_volume(volume);

        let state = session.state();
        Ok(Session {
            id: session_id,
            current_track_id: state.current_track.map(|id| ID::from(id.to_string())),
            position: state.position,
            playback_state: format!("{:?}", state.playback),
            volume: state.volume,
            peer_count: state.peers.len() as i32,
        })
    }

    /// Index a track for similarity search
    async fn index_track(
        &self,
        ctx: &Context<'_>,
        track_id: ID,
    ) -> async_graphql::Result<bool> {
        let _engine = ctx.data::<Arc<MycelixEngine>>()?;
        // In production, would index the track
        Ok(true)
    }
}

// === GraphQL Types ===

/// Track type
#[derive(SimpleObject)]
pub struct Track {
    pub id: ID,
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub duration: f32,
    pub sample_rate: u32,
    pub channels: i32,
    pub analysis: Option<TrackAnalysis>,
}

/// Track analysis results
#[derive(SimpleObject)]
pub struct TrackAnalysis {
    pub genre: String,
    pub genre_confidence: f32,
    pub mood: String,
    pub bpm: f32,
    pub key: String,
    pub beats: Vec<f32>,
    pub segments: Vec<String>,
}

/// Search input
#[derive(InputObject)]
pub struct SearchInput {
    pub text: Option<String>,
    pub genre: Option<String>,
    pub mood: Option<String>,
    pub bpm_min: Option<f32>,
    pub bpm_max: Option<f32>,
    pub key: Option<String>,
    #[graphql(default = 20)]
    pub limit: i32,
}

/// Search result
#[derive(SimpleObject)]
pub struct SearchResult {
    pub track: Track,
    pub score: f32,
}

/// Similar track result
#[derive(SimpleObject)]
pub struct SimilarTrack {
    pub track: Track,
    pub similarity: f32,
}

/// Session type
#[derive(SimpleObject)]
pub struct Session {
    pub id: ID,
    pub current_track_id: Option<ID>,
    pub position: f32,
    pub playback_state: String,
    pub volume: f32,
    pub peer_count: i32,
}

/// Audio format info
#[derive(SimpleObject)]
pub struct AudioFormat {
    pub name: String,
    pub description: String,
    pub mime_type: String,
}

/// Supported formats
#[derive(SimpleObject)]
pub struct Formats {
    pub encoders: Vec<AudioFormat>,
    pub decoders: Vec<AudioFormat>,
}

/// Health status
#[derive(SimpleObject)]
pub struct HealthStatus {
    pub status: String,
    pub version: String,
}
