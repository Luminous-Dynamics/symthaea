// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Session management for listening sessions

use std::sync::Arc;

use bytes::Bytes;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::{
    audio, codec,
    engine::{AudioEngine, StreamEngine, StreamHandle},
    spatial, track, CoreError, Result,
};

/// A listening session with playback state and connections
pub struct Session {
    id: Uuid,
    audio_engine: Arc<AudioEngine>,
    stream_engine: Arc<StreamEngine>,
    state: RwLock<SessionState>,
    spatial_enabled: RwLock<bool>,
    active_streams: RwLock<Vec<StreamHandle>>,
}

/// Session state
#[derive(Debug, Clone, Default)]
pub struct SessionState {
    /// Current track being played
    pub current_track: Option<Uuid>,
    /// Playback position in seconds
    pub position: f32,
    /// Playback state
    pub playback: PlaybackState,
    /// Volume (0.0 - 1.0)
    pub volume: f32,
    /// Connected peers
    pub peers: Vec<String>,
}

/// Playback state
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PlaybackState {
    #[default]
    Stopped,
    Playing,
    Paused,
    Buffering,
}

impl Session {
    /// Create a new session
    pub fn new(audio_engine: Arc<AudioEngine>, stream_engine: Arc<StreamEngine>) -> Self {
        Self {
            id: Uuid::new_v4(),
            audio_engine,
            stream_engine,
            state: RwLock::new(SessionState {
                volume: 1.0,
                ..Default::default()
            }),
            spatial_enabled: RwLock::new(false),
            active_streams: RwLock::new(Vec::new()),
        }
    }

    /// Get session ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get current state
    pub fn state(&self) -> SessionState {
        self.state.read().clone()
    }

    /// Start streaming a track
    pub async fn stream_track(&self, track: &track::Track) -> Result<mpsc::Receiver<Bytes>> {
        let codec = &self.audio_engine.config().default_codec;
        let mut handle = self.stream_engine.create_stream(codec, codec::QualityPreset::High).await?;

        // Update state
        {
            let mut state = self.state.write();
            state.current_track = Some(track.id);
            state.playback = PlaybackState::Playing;
            state.position = 0.0;
        }

        // Encode and stream in background
        let audio_engine = self.audio_engine.clone();
        let buffer = track.audio_buffer.clone();
        let sender = handle.sender.clone();
        let codec = codec.clone();

        tokio::spawn(async move {
            // Encode in chunks
            let chunk_size = 4096;

            for chunk in buffer.samples.chunks(chunk_size) {
                // Create mini buffer for encoding
                let mini_buffer = audio::AudioBuffer {
                    samples: chunk.to_vec(),
                    sample_rate: buffer.sample_rate,
                    channels: buffer.channels,
                    layout: buffer.layout.clone(),
                };

                let encoded = audio_engine
                    .encode(&mini_buffer, &codec, codec::QualityPreset::High)
                    .await;

                if let Ok(data) = encoded {
                    if sender.send(data).await.is_err() {
                        break;
                    }
                }
            }
        });

        let receiver = handle.take_receiver().ok_or_else(|| {
            CoreError::Pipeline("Failed to get stream receiver".to_string())
        })?;

        self.active_streams.write().push(handle);
        Ok(receiver)
    }

    /// Connect to a peer
    pub async fn connect_peer(&self, peer_addr: &str) -> Result<()> {
        self.stream_engine.connect_quic(peer_addr).await?;
        self.state.write().peers.push(peer_addr.to_string());
        Ok(())
    }

    /// Enable spatial audio
    pub async fn enable_spatial(&self) -> Result<()> {
        self.audio_engine.enable_spatial()?;
        *self.spatial_enabled.write() = true;
        Ok(())
    }

    /// Update listener position
    pub async fn update_listener(
        &self,
        _position: spatial::Position,
        _orientation: spatial::Orientation,
    ) -> Result<()> {
        // Listener position would be stored and used for spatial rendering
        Ok(())
    }

    /// Play
    pub fn play(&self) {
        self.state.write().playback = PlaybackState::Playing;
    }

    /// Pause
    pub fn pause(&self) {
        self.state.write().playback = PlaybackState::Paused;
    }

    /// Stop
    pub fn stop(&self) {
        let mut state = self.state.write();
        state.playback = PlaybackState::Stopped;
        state.position = 0.0;
    }

    /// Seek to position
    pub fn seek(&self, position: f32) {
        self.state.write().position = position;
    }

    /// Set volume
    pub fn set_volume(&self, volume: f32) {
        self.state.write().volume = volume.clamp(0.0, 1.0);
    }

    /// Close the session
    pub async fn close(&self) -> Result<()> {
        self.stop();
        self.active_streams.write().clear();
        Ok(())
    }
}

/// Session builder for configuration
pub struct SessionBuilder {
    audio_engine: Option<Arc<AudioEngine>>,
    stream_engine: Option<Arc<StreamEngine>>,
    spatial_enabled: bool,
    initial_volume: f32,
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionBuilder {
    pub fn new() -> Self {
        Self {
            audio_engine: None,
            stream_engine: None,
            spatial_enabled: false,
            initial_volume: 1.0,
        }
    }

    pub fn audio_engine(mut self, engine: Arc<AudioEngine>) -> Self {
        self.audio_engine = Some(engine);
        self
    }

    pub fn stream_engine(mut self, engine: Arc<StreamEngine>) -> Self {
        self.stream_engine = Some(engine);
        self
    }

    pub fn with_spatial(mut self) -> Self {
        self.spatial_enabled = true;
        self
    }

    pub fn volume(mut self, volume: f32) -> Self {
        self.initial_volume = volume;
        self
    }

    pub fn build(self) -> Result<Session> {
        let audio = self
            .audio_engine
            .ok_or_else(|| CoreError::Config("Audio engine required".to_string()))?;
        let stream = self
            .stream_engine
            .ok_or_else(|| CoreError::Config("Stream engine required".to_string()))?;

        let session = Session::new(audio, stream);
        session.set_volume(self.initial_volume);

        if self.spatial_enabled {
            session.audio_engine.enable_spatial()?;
            *session.spatial_enabled.write() = true;
        }

        Ok(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioConfig, NetworkConfig};

    #[tokio::test]
    async fn test_session_state() {
        let audio_config = AudioConfig {
            sample_rate: 48000,
            channels: 2,
            buffer_size: 1024,
            default_codec: "opus".to_string(),
        };
        let network_config = NetworkConfig {
            bind_address: "127.0.0.1:0".to_string(),
            max_connections: 10,
            enable_quic: false,
        };

        let audio_engine = Arc::new(AudioEngine::new(&audio_config).unwrap());
        let stream_engine = Arc::new(StreamEngine::new(&network_config).await.unwrap());

        let session = Session::new(audio_engine, stream_engine);

        assert_eq!(session.state().playback, PlaybackState::Stopped);
        assert_eq!(session.state().volume, 1.0);

        session.play();
        assert_eq!(session.state().playback, PlaybackState::Playing);

        session.set_volume(0.5);
        assert_eq!(session.state().volume, 0.5);
    }
}
