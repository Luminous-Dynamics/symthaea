// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Core - Unified API Surface
//!
//! This crate provides a unified interface to all Mycelix components:
//! - Audio processing and analysis
//! - Codec encoding/decoding
//! - ML-based music analysis
//! - Music Information Retrieval
//! - Spatial audio rendering
//! - Real-time streaming
//! - Similarity search

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;
use uuid::Uuid;

// Re-export component crates
pub use mycelix_audio as audio;
pub use mycelix_codec as codec;
pub use mycelix_mir as mir;
pub use mycelix_ml as ml;
pub use mycelix_protocol as protocol;
pub use mycelix_realtime as realtime;
pub use mycelix_search as search;
pub use mycelix_spatial as spatial;
pub use mycelix_streaming as streaming;

pub mod engine;
pub mod pipeline;
pub mod session;
pub mod track;

/// Core error type
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Audio error: {0}")]
    Audio(#[from] mycelix_audio::AudioError),

    #[error("Codec error: {0}")]
    Codec(#[from] mycelix_codec::CodecError),

    #[error("ML error: {0}")]
    Ml(#[from] mycelix_ml::MlError),

    #[error("MIR error: {0}")]
    Mir(#[from] mycelix_mir::MirError),

    #[error("Protocol error: {0}")]
    Protocol(#[from] mycelix_protocol::ProtocolError),

    #[error("Search error: {0}")]
    Search(#[from] mycelix_search::SearchError),

    #[error("Spatial error: {0}")]
    Spatial(#[from] mycelix_spatial::SpatialError),

    #[error("Streaming error: {0}")]
    Streaming(#[from] mycelix_streaming::StreamingError),

    #[error("Realtime error: {0}")]
    Realtime(#[from] mycelix_realtime::RealtimeError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Session not found: {0}")]
    SessionNotFound(Uuid),

    #[error("Track not found: {0}")]
    TrackNotFound(Uuid),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;

/// Core configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    /// Audio configuration
    pub audio: AudioConfig,
    /// ML configuration
    pub ml: MlConfig,
    /// Network configuration
    pub network: NetworkConfig,
    /// Storage configuration
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u8,
    pub buffer_size: usize,
    pub default_codec: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlConfig {
    pub models_path: String,
    pub enable_gpu: bool,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub bind_address: String,
    pub max_connections: usize,
    pub enable_quic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_path: String,
    pub cache_size_mb: usize,
    pub index_path: String,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            audio: AudioConfig {
                sample_rate: 48000,
                channels: 2,
                buffer_size: 1024,
                default_codec: "opus".to_string(),
            },
            ml: MlConfig {
                models_path: "./models".to_string(),
                enable_gpu: false,
                batch_size: 32,
            },
            network: NetworkConfig {
                bind_address: "0.0.0.0:8443".to_string(),
                max_connections: 1000,
                enable_quic: true,
            },
            storage: StorageConfig {
                data_path: "./data".to_string(),
                cache_size_mb: 512,
                index_path: "./index".to_string(),
            },
        }
    }
}

/// Main Mycelix engine - the central orchestrator
pub struct MycelixEngine {
    config: CoreConfig,
    audio_engine: Arc<engine::AudioEngine>,
    ml_engine: Arc<engine::MlEngine>,
    search_engine: Arc<engine::SearchEngine>,
    stream_engine: Arc<engine::StreamEngine>,
    sessions: Arc<RwLock<std::collections::HashMap<Uuid, Arc<session::Session>>>>,
}

impl MycelixEngine {
    /// Create a new Mycelix engine with the given configuration
    pub async fn new(config: CoreConfig) -> Result<Self> {
        let audio_engine = Arc::new(engine::AudioEngine::new(&config.audio)?);
        let ml_engine = Arc::new(engine::MlEngine::new(&config.ml).await?);
        let search_engine = Arc::new(engine::SearchEngine::new(&config.storage)?);
        let stream_engine = Arc::new(engine::StreamEngine::new(&config.network).await?);

        Ok(Self {
            config,
            audio_engine,
            ml_engine,
            search_engine,
            stream_engine,
            sessions: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Create with default configuration
    pub async fn default_engine() -> Result<Self> {
        Self::new(CoreConfig::default()).await
    }

    /// Get the configuration
    pub fn config(&self) -> &CoreConfig {
        &self.config
    }

    // === Track Operations ===

    /// Load a track from a file path
    pub async fn load_track(&self, path: impl AsRef<Path>) -> Result<track::Track> {
        let audio_buffer = self.audio_engine.load_file(path.as_ref()).await?;
        let track_id = Uuid::new_v4();

        Ok(track::Track::new(track_id, audio_buffer))
    }

    /// Analyze a track (ML + MIR analysis)
    pub async fn analyze_track(&self, track: &mut track::Track) -> Result<track::TrackAnalysis> {
        // Run ML analysis
        let ml_analysis = self.ml_engine.analyze(&track.audio_buffer).await?;

        // Run MIR analysis
        let mir_analysis = self.audio_engine.mir_analyze(&track.audio_buffer)?;

        // Generate embedding for similarity search
        let embedding = self.search_engine.generate_embedding(track)?;

        let analysis = track::TrackAnalysis {
            genre: ml_analysis.genre,
            mood: ml_analysis.mood,
            bpm: mir_analysis.bpm,
            key: mir_analysis.key,
            beats: mir_analysis.beats,
            segments: mir_analysis.segments,
            embedding,
        };

        track.analysis = Some(analysis.clone());
        Ok(analysis)
    }

    /// Find similar tracks
    pub async fn find_similar(
        &self,
        track: &track::Track,
        limit: usize,
    ) -> Result<Vec<search::SearchResult>> {
        let embedding = track
            .analysis
            .as_ref()
            .map(|a| a.embedding.clone())
            .ok_or_else(|| CoreError::Pipeline("Track not analyzed".to_string()))?;

        self.search_engine.search(&embedding, limit).await
    }

    /// Index a track for similarity search
    pub async fn index_track(&self, track: &track::Track) -> Result<()> {
        let embedding = track
            .analysis
            .as_ref()
            .map(|a| a.embedding.clone())
            .ok_or_else(|| CoreError::Pipeline("Track not analyzed".to_string()))?;

        self.search_engine.index(track.id, embedding).await
    }

    // === Session Operations ===

    /// Create a new listening session
    pub async fn create_session(&self) -> Result<Uuid> {
        let session = session::Session::new(
            self.audio_engine.clone(),
            self.stream_engine.clone(),
        );
        let session_id = session.id();

        self.sessions.write().insert(session_id, Arc::new(session));
        Ok(session_id)
    }

    /// Get a session by ID
    pub fn get_session(&self, id: Uuid) -> Result<Arc<session::Session>> {
        self.sessions
            .read()
            .get(&id)
            .cloned()
            .ok_or(CoreError::SessionNotFound(id))
    }

    /// Close a session
    pub async fn close_session(&self, id: Uuid) -> Result<()> {
        let session = self.sessions.write().remove(&id);
        if let Some(s) = session {
            s.close().await?;
        }
        Ok(())
    }

    // === Streaming Operations ===

    /// Start streaming a track
    pub async fn stream_track(
        &self,
        session_id: Uuid,
        track: &track::Track,
    ) -> Result<mpsc::Receiver<Bytes>> {
        let session = self.get_session(session_id)?;
        session.stream_track(track).await
    }

    /// Connect to a remote peer for P2P streaming
    pub async fn connect_peer(&self, session_id: Uuid, peer_addr: &str) -> Result<()> {
        let session = self.get_session(session_id)?;
        session.connect_peer(peer_addr).await
    }

    // === Spatial Audio ===

    /// Enable spatial audio for a session
    pub async fn enable_spatial(&self, session_id: Uuid) -> Result<()> {
        let session = self.get_session(session_id)?;
        session.enable_spatial().await
    }

    /// Update listener position
    pub async fn update_listener_position(
        &self,
        session_id: Uuid,
        position: spatial::Position,
        orientation: spatial::Orientation,
    ) -> Result<()> {
        let session = self.get_session(session_id)?;
        session.update_listener(position, orientation).await
    }

    // === Codec Operations ===

    /// Encode audio to a specific format
    pub async fn encode(
        &self,
        audio: &audio::AudioBuffer,
        codec: &str,
        quality: codec::QualityPreset,
    ) -> Result<Bytes> {
        self.audio_engine.encode(audio, codec, quality).await
    }

    /// Decode audio from bytes
    pub async fn decode(&self, data: &[u8], codec: &str) -> Result<audio::AudioBuffer> {
        self.audio_engine.decode(data, codec).await
    }

    /// Transcode between formats
    pub async fn transcode(
        &self,
        data: &[u8],
        from_codec: &str,
        to_codec: &str,
        quality: codec::QualityPreset,
    ) -> Result<Bytes> {
        let audio = self.decode(data, from_codec).await?;
        self.encode(&audio, to_codec, quality).await
    }

    // === Engine Access ===

    /// Get direct access to audio engine
    pub fn audio(&self) -> &engine::AudioEngine {
        &self.audio_engine
    }

    /// Get direct access to ML engine
    pub fn ml(&self) -> &engine::MlEngine {
        &self.ml_engine
    }

    /// Get direct access to search engine
    pub fn search(&self) -> &engine::SearchEngine {
        &self.search_engine
    }

    /// Get direct access to stream engine
    pub fn stream(&self) -> &engine::StreamEngine {
        &self.stream_engine
    }
}

/// Event types emitted by the engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineEvent {
    TrackLoaded { track_id: Uuid },
    TrackAnalyzed { track_id: Uuid },
    SessionCreated { session_id: Uuid },
    SessionClosed { session_id: Uuid },
    StreamStarted { session_id: Uuid, track_id: Uuid },
    StreamEnded { session_id: Uuid, track_id: Uuid },
    PeerConnected { session_id: Uuid, peer_id: String },
    PeerDisconnected { session_id: Uuid, peer_id: String },
    Error { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CoreConfig::default();
        assert_eq!(config.audio.sample_rate, 48000);
        assert_eq!(config.audio.channels, 2);
    }
}
