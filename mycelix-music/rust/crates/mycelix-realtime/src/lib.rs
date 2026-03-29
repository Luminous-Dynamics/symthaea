// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Real-Time Collaboration Engine
//!
//! High-performance real-time collaboration infrastructure providing:
//! - Live jam sessions with low-latency audio
//! - Synchronized listening parties
//! - Real-time playlist collaboration
//! - WebRTC signaling
//! - Presence and cursor tracking
//! - CRDT-based conflict resolution

pub mod session;
pub mod sync;
pub mod signaling;
pub mod presence;
pub mod crdt;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, RwLock};
use uuid::Uuid;

// ============================================================================
// ERROR TYPES
// ============================================================================

#[derive(Error, Debug)]
pub enum RealtimeError {
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("User not found: {0}")]
    UserNotFound(String),

    #[error("Peer not found: {0}")]
    PeerNotFound(Uuid),

    #[error("Room full")]
    RoomFull,

    #[error("Session full")]
    SessionFull,

    #[error("Session closed")]
    SessionClosed,

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Sync error: {0}")]
    SyncError(String),

    #[error("Timeout")]
    Timeout,
}

pub type RealtimeResult<T> = Result<T, RealtimeError>;

// ============================================================================
// CORE TYPES
// ============================================================================

/// Collaboration session type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionType {
    /// Live jam session with real-time audio
    LiveJam,
    /// Synchronized listening party
    ListeningParty,
    /// Collaborative playlist editing
    PlaylistCollab,
    /// DJ session with mixing
    DjSession,
}

/// Participant role in session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantRole {
    /// Session host with full control
    Host,
    /// Co-host with elevated permissions
    CoHost,
    /// Active participant (can contribute)
    Participant,
    /// Listener (view/listen only)
    Listener,
}

impl ParticipantRole {
    pub fn can_control_playback(&self) -> bool {
        matches!(self, Self::Host | Self::CoHost)
    }

    pub fn can_add_tracks(&self) -> bool {
        matches!(self, Self::Host | Self::CoHost | Self::Participant)
    }

    pub fn can_broadcast_audio(&self) -> bool {
        matches!(self, Self::Host | Self::CoHost | Self::Participant)
    }
}

/// Participant in a collaboration session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    pub id: Uuid,
    pub user_id: String,
    pub display_name: String,
    pub avatar_url: Option<String>,
    pub role: ParticipantRole,
    pub joined_at: DateTime<Utc>,
    pub is_muted: bool,
    pub is_deafened: bool,
    pub is_speaking: bool,
    pub connection_quality: ConnectionQuality,
}

/// Connection quality indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Disconnected,
}

/// Collaboration session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationSession {
    pub id: Uuid,
    pub session_type: SessionType,
    pub name: String,
    pub description: Option<String>,
    pub host_id: String,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub is_public: bool,
    pub max_participants: u32,
    pub settings: SessionSettings,
}

/// Session settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSettings {
    /// Enable audio broadcasting
    pub audio_enabled: bool,
    /// Enable video (for DJ sessions)
    pub video_enabled: bool,
    /// Enable chat
    pub chat_enabled: bool,
    /// Enable reactions
    pub reactions_enabled: bool,
    /// Latency target in milliseconds
    pub target_latency_ms: u32,
    /// Auto-sync tolerance in milliseconds
    pub sync_tolerance_ms: u32,
    /// Enable recording
    pub recording_enabled: bool,
}

impl Default for SessionSettings {
    fn default() -> Self {
        Self {
            audio_enabled: true,
            video_enabled: false,
            chat_enabled: true,
            reactions_enabled: true,
            target_latency_ms: 100,
            sync_tolerance_ms: 50,
            recording_enabled: false,
        }
    }
}

// ============================================================================
// PLAYBACK STATE (CRDT-based)
// ============================================================================

/// Playback state for synchronized listening
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybackState {
    /// Current track ID
    pub track_id: Option<String>,
    /// Current position in milliseconds
    pub position_ms: u64,
    /// Is playing
    pub is_playing: bool,
    /// Playback speed multiplier
    pub speed: f32,
    /// Volume (0.0 - 1.0)
    pub volume: f32,
    /// Queue of upcoming tracks
    pub queue: Vec<String>,
    /// Timestamp of last state change
    pub timestamp: DateTime<Utc>,
    /// Version for conflict resolution
    pub version: u64,
    /// Author of last change
    pub author: String,
}

impl PlaybackState {
    pub fn new() -> Self {
        Self {
            track_id: None,
            position_ms: 0,
            is_playing: false,
            speed: 1.0,
            volume: 1.0,
            queue: Vec::new(),
            timestamp: Utc::now(),
            version: 0,
            author: String::new(),
        }
    }

    /// Calculate expected position based on elapsed time
    pub fn expected_position(&self) -> u64 {
        if !self.is_playing {
            return self.position_ms;
        }

        let elapsed = Utc::now().signed_duration_since(self.timestamp);
        let elapsed_ms = elapsed.num_milliseconds().max(0) as u64;
        self.position_ms + (elapsed_ms as f32 * self.speed) as u64
    }

    /// Apply a state update (with version check)
    pub fn apply_update(&mut self, update: &PlaybackStateUpdate) -> bool {
        // Only apply if version is newer
        if update.version <= self.version {
            return false;
        }

        match update.action {
            PlaybackAction::Play => {
                self.is_playing = true;
                self.position_ms = update.position_ms.unwrap_or(self.position_ms);
            }
            PlaybackAction::Pause => {
                self.is_playing = false;
                self.position_ms = update.position_ms.unwrap_or(self.expected_position());
            }
            PlaybackAction::Seek => {
                self.position_ms = update.position_ms.unwrap_or(0);
            }
            PlaybackAction::ChangeTrack => {
                self.track_id = update.track_id.clone();
                self.position_ms = 0;
                self.is_playing = true;
            }
            PlaybackAction::AddToQueue => {
                if let Some(track_id) = &update.track_id {
                    self.queue.push(track_id.clone());
                }
            }
            PlaybackAction::RemoveFromQueue => {
                if let Some(index) = update.queue_index {
                    if index < self.queue.len() {
                        self.queue.remove(index);
                    }
                }
            }
            PlaybackAction::SetVolume => {
                self.volume = update.volume.unwrap_or(self.volume);
            }
            PlaybackAction::SetSpeed => {
                self.speed = update.speed.unwrap_or(self.speed);
            }
        }

        self.version = update.version;
        self.author = update.author.clone();
        self.timestamp = Utc::now();

        true
    }
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self::new()
    }
}

/// Playback action
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PlaybackAction {
    Play,
    Pause,
    Seek,
    ChangeTrack,
    AddToQueue,
    RemoveFromQueue,
    SetVolume,
    SetSpeed,
}

/// Playback state update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybackStateUpdate {
    pub action: PlaybackAction,
    pub track_id: Option<String>,
    pub position_ms: Option<u64>,
    pub volume: Option<f32>,
    pub speed: Option<f32>,
    pub queue_index: Option<usize>,
    pub version: u64,
    pub author: String,
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// MESSAGES
// ============================================================================

/// Real-time message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeMessage {
    // Connection
    #[serde(rename = "join")]
    Join { session_id: Uuid, user_id: String, display_name: String },

    #[serde(rename = "leave")]
    Leave { session_id: Uuid, user_id: String },

    #[serde(rename = "heartbeat")]
    Heartbeat { timestamp: DateTime<Utc> },

    // Presence
    #[serde(rename = "presence_update")]
    PresenceUpdate { user_id: String, status: PresenceStatus },

    // Playback
    #[serde(rename = "playback_update")]
    PlaybackUpdate { update: PlaybackStateUpdate },

    #[serde(rename = "playback_sync")]
    PlaybackSync { state: PlaybackState },

    // Audio
    #[serde(rename = "audio_frame")]
    AudioFrame { user_id: String, data: Vec<u8>, sequence: u64 },

    #[serde(rename = "mute_toggle")]
    MuteToggle { user_id: String, is_muted: bool },

    // Chat
    #[serde(rename = "chat")]
    Chat { user_id: String, message: String, timestamp: DateTime<Utc> },

    // Reactions
    #[serde(rename = "reaction")]
    Reaction { user_id: String, emoji: String, timestamp: DateTime<Utc> },

    // WebRTC Signaling
    #[serde(rename = "offer")]
    Offer { from: String, to: String, sdp: String },

    #[serde(rename = "answer")]
    Answer { from: String, to: String, sdp: String },

    #[serde(rename = "ice_candidate")]
    IceCandidate { from: String, to: String, candidate: String },
}

/// Presence status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceStatus {
    pub is_online: bool,
    pub is_speaking: bool,
    pub connection_quality: ConnectionQuality,
    pub last_active: DateTime<Utc>,
}

// ============================================================================
// ROOM MANAGER
// ============================================================================

/// Room state
struct RoomState {
    session: CollaborationSession,
    participants: HashMap<String, Participant>,
    playback_state: PlaybackState,
    message_tx: broadcast::Sender<RealtimeMessage>,
}

/// Room manager for handling collaboration sessions
pub struct RoomManager {
    rooms: Arc<RwLock<HashMap<Uuid, RoomState>>>,
}

impl RoomManager {
    pub fn new() -> Self {
        Self {
            rooms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new room
    pub async fn create_room(&self, session: CollaborationSession) -> RealtimeResult<Uuid> {
        let id = session.id;
        let (tx, _) = broadcast::channel(1000);

        let state = RoomState {
            session,
            participants: HashMap::new(),
            playback_state: PlaybackState::new(),
            message_tx: tx,
        };

        self.rooms.write().await.insert(id, state);
        Ok(id)
    }

    /// Join a room
    pub async fn join_room(
        &self,
        room_id: Uuid,
        user_id: String,
        display_name: String,
        avatar_url: Option<String>,
    ) -> RealtimeResult<(Participant, broadcast::Receiver<RealtimeMessage>)> {
        let mut rooms = self.rooms.write().await;
        let room = rooms.get_mut(&room_id)
            .ok_or_else(|| RealtimeError::SessionNotFound(room_id.to_string()))?;

        // Check if room is full
        if room.participants.len() >= room.session.max_participants as usize {
            return Err(RealtimeError::RoomFull);
        }

        // Determine role
        let role = if room.session.host_id == user_id {
            ParticipantRole::Host
        } else if room.participants.is_empty() {
            ParticipantRole::CoHost
        } else {
            ParticipantRole::Participant
        };

        let participant = Participant {
            id: Uuid::new_v4(),
            user_id: user_id.clone(),
            display_name: display_name.clone(),
            avatar_url,
            role,
            joined_at: Utc::now(),
            is_muted: false,
            is_deafened: false,
            is_speaking: false,
            connection_quality: ConnectionQuality::Good,
        };

        room.participants.insert(user_id.clone(), participant.clone());

        // Broadcast join message
        let _ = room.message_tx.send(RealtimeMessage::Join {
            session_id: room_id,
            user_id,
            display_name,
        });

        Ok((participant, room.message_tx.subscribe()))
    }

    /// Leave a room
    pub async fn leave_room(&self, room_id: Uuid, user_id: &str) -> RealtimeResult<()> {
        let mut rooms = self.rooms.write().await;
        let room = rooms.get_mut(&room_id)
            .ok_or_else(|| RealtimeError::SessionNotFound(room_id.to_string()))?;

        room.participants.remove(user_id);

        let _ = room.message_tx.send(RealtimeMessage::Leave {
            session_id: room_id,
            user_id: user_id.to_string(),
        });

        // Remove room if empty
        if room.participants.is_empty() {
            rooms.remove(&room_id);
        }

        Ok(())
    }

    /// Broadcast message to room
    pub async fn broadcast(&self, room_id: Uuid, message: RealtimeMessage) -> RealtimeResult<()> {
        let rooms = self.rooms.read().await;
        let room = rooms.get(&room_id)
            .ok_or_else(|| RealtimeError::SessionNotFound(room_id.to_string()))?;

        let _ = room.message_tx.send(message);
        Ok(())
    }

    /// Update playback state
    pub async fn update_playback(
        &self,
        room_id: Uuid,
        update: PlaybackStateUpdate,
    ) -> RealtimeResult<PlaybackState> {
        let mut rooms = self.rooms.write().await;
        let room = rooms.get_mut(&room_id)
            .ok_or_else(|| RealtimeError::SessionNotFound(room_id.to_string()))?;

        // Check if user has permission
        let participant = room.participants.get(&update.author)
            .ok_or_else(|| RealtimeError::UserNotFound(update.author.clone()))?;

        if !participant.role.can_control_playback() {
            return Err(RealtimeError::PermissionDenied(
                "Cannot control playback".to_string()
            ));
        }

        room.playback_state.apply_update(&update);

        // Broadcast update
        let _ = room.message_tx.send(RealtimeMessage::PlaybackUpdate { update });

        Ok(room.playback_state.clone())
    }

    /// Get room state
    pub async fn get_room_state(&self, room_id: Uuid) -> RealtimeResult<(CollaborationSession, Vec<Participant>, PlaybackState)> {
        let rooms = self.rooms.read().await;
        let room = rooms.get(&room_id)
            .ok_or_else(|| RealtimeError::SessionNotFound(room_id.to_string()))?;

        Ok((
            room.session.clone(),
            room.participants.values().cloned().collect(),
            room.playback_state.clone(),
        ))
    }

    /// Get participant count
    pub async fn get_participant_count(&self, room_id: Uuid) -> RealtimeResult<usize> {
        let rooms = self.rooms.read().await;
        let room = rooms.get(&room_id)
            .ok_or_else(|| RealtimeError::SessionNotFound(room_id.to_string()))?;

        Ok(room.participants.len())
    }

    /// List active rooms
    pub async fn list_rooms(&self) -> Vec<(Uuid, String, usize)> {
        let rooms = self.rooms.read().await;
        rooms.iter()
            .map(|(id, room)| (*id, room.session.name.clone(), room.participants.len()))
            .collect()
    }
}

impl Default for RoomManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SYNC MANAGER
// ============================================================================

/// Synchronization manager for ensuring consistent playback
pub struct SyncManager {
    /// Sync tolerance in milliseconds
    tolerance_ms: u32,
}

impl SyncManager {
    pub fn new(tolerance_ms: u32) -> Self {
        Self { tolerance_ms }
    }

    /// Calculate sync correction needed
    pub fn calculate_correction(
        &self,
        local_position_ms: u64,
        reference_state: &PlaybackState,
    ) -> SyncCorrection {
        let expected = reference_state.expected_position();
        let diff = local_position_ms as i64 - expected as i64;

        if diff.abs() <= self.tolerance_ms as i64 {
            SyncCorrection::InSync
        } else if diff.abs() <= 500 {
            // Small drift - adjust playback speed
            let speed_adjustment = if diff > 0 {
                0.95 // Slow down
            } else {
                1.05 // Speed up
            };
            SyncCorrection::AdjustSpeed(speed_adjustment)
        } else {
            // Large drift - hard seek
            SyncCorrection::Seek(expected)
        }
    }

    /// Estimate network latency from round-trip time
    pub fn estimate_latency(&self, rtt_samples: &[u32]) -> u32 {
        if rtt_samples.is_empty() {
            return 0;
        }

        // Use median RTT / 2 as one-way latency estimate
        let mut sorted = rtt_samples.to_vec();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        median / 2
    }
}

/// Sync correction action
#[derive(Debug, Clone, Copy)]
pub enum SyncCorrection {
    /// Playback is in sync
    InSync,
    /// Adjust playback speed temporarily
    AdjustSpeed(f32),
    /// Hard seek to position
    Seek(u64),
}

// ============================================================================
// WEBRTC SIGNALING
// ============================================================================

/// WebRTC signaling message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalingMessage {
    pub from: String,
    pub to: String,
    pub payload: SignalingPayload,
    pub timestamp: DateTime<Utc>,
}

/// Signaling payload
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SignalingPayload {
    Offer { sdp: String },
    Answer { sdp: String },
    IceCandidate { candidate: String, sdp_mid: Option<String>, sdp_mline_index: Option<u16> },
    Disconnect,
}

/// WebRTC signaling server
pub struct SignalingServer {
    /// Active connections
    connections: Arc<RwLock<HashMap<String, mpsc::Sender<SignalingMessage>>>>,
}

impl SignalingServer {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a connection
    pub async fn register(&self, user_id: String) -> mpsc::Receiver<SignalingMessage> {
        let (tx, rx) = mpsc::channel(100);
        self.connections.write().await.insert(user_id, tx);
        rx
    }

    /// Unregister a connection
    pub async fn unregister(&self, user_id: &str) {
        self.connections.write().await.remove(user_id);
    }

    /// Send signaling message
    pub async fn send(&self, message: SignalingMessage) -> RealtimeResult<()> {
        let connections = self.connections.read().await;

        if let Some(tx) = connections.get(&message.to) {
            tx.send(message).await
                .map_err(|_| RealtimeError::ConnectionError("Failed to send message".to_string()))?;
        }

        Ok(())
    }

    /// Broadcast to all except sender
    pub async fn broadcast_except(&self, sender: &str, message: SignalingMessage) -> RealtimeResult<()> {
        let connections = self.connections.read().await;

        for (user_id, tx) in connections.iter() {
            if user_id != sender {
                let mut msg = message.clone();
                msg.to = user_id.clone();
                let _ = tx.send(msg).await;
            }
        }

        Ok(())
    }
}

impl Default for SignalingServer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AUDIO MIXER (for live jam sessions)
// ============================================================================

/// Audio frame from a participant
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub user_id: String,
    pub data: Bytes,
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
    pub sample_rate: u32,
    pub channels: u8,
}

/// Audio mixer for combining multiple streams
pub struct AudioMixer {
    /// Active audio streams
    streams: HashMap<String, VecDeque<AudioFrame>>,
    /// Output sample rate
    sample_rate: u32,
    /// Buffer size in frames
    buffer_size: usize,
}

use std::collections::VecDeque;

impl AudioMixer {
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            streams: HashMap::new(),
            sample_rate,
            buffer_size,
        }
    }

    /// Add audio frame from participant
    pub fn add_frame(&mut self, frame: AudioFrame) {
        let buffer = self.streams.entry(frame.user_id.clone()).or_insert_with(VecDeque::new);

        buffer.push_back(frame);

        // Trim old frames
        while buffer.len() > self.buffer_size {
            buffer.pop_front();
        }
    }

    /// Mix all streams and return combined audio
    pub fn mix(&mut self) -> Option<Vec<f32>> {
        if self.streams.is_empty() {
            return None;
        }

        // Get the minimum available frames across all streams
        let min_frames = self.streams.values()
            .map(|b| b.len())
            .min()
            .unwrap_or(0);

        if min_frames == 0 {
            return None;
        }

        // Collect one frame from each stream
        let frames: Vec<AudioFrame> = self.streams.values_mut()
            .filter_map(|b| b.pop_front())
            .collect();

        // Decode and mix
        let mut output = vec![0.0f32; 1024]; // Assuming 1024 samples per frame

        for frame in frames {
            // Decode frame (assuming 16-bit PCM for simplicity)
            let samples: Vec<f32> = frame.data.chunks(2)
                .map(|chunk| {
                    if chunk.len() >= 2 {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    } else {
                        0.0
                    }
                })
                .collect();

            // Add to output
            for (i, sample) in samples.iter().enumerate() {
                if i < output.len() {
                    output[i] += sample;
                }
            }
        }

        // Normalize to prevent clipping
        let max_val = output.iter().map(|s| s.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(1.0);
        if max_val > 1.0 {
            for sample in &mut output {
                *sample /= max_val;
            }
        }

        Some(output)
    }

    /// Remove participant stream
    pub fn remove_stream(&mut self, user_id: &str) {
        self.streams.remove(user_id);
    }

    /// Get active stream count
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playback_state() {
        let mut state = PlaybackState::new();
        assert!(!state.is_playing);
        assert_eq!(state.position_ms, 0);

        let update = PlaybackStateUpdate {
            action: PlaybackAction::Play,
            track_id: Some("track123".to_string()),
            position_ms: Some(1000),
            volume: None,
            speed: None,
            queue_index: None,
            version: 1,
            author: "user1".to_string(),
            timestamp: Utc::now(),
        };

        state.apply_update(&update);
        assert!(state.is_playing);
        assert_eq!(state.position_ms, 1000);
    }

    #[test]
    fn test_sync_manager() {
        let sync = SyncManager::new(50);

        let state = PlaybackState {
            is_playing: true,
            position_ms: 10000,
            timestamp: Utc::now(),
            ..Default::default()
        };

        // In sync
        let correction = sync.calculate_correction(10025, &state);
        assert!(matches!(correction, SyncCorrection::InSync));

        // Needs speed adjustment
        let correction = sync.calculate_correction(10200, &state);
        assert!(matches!(correction, SyncCorrection::AdjustSpeed(_)));

        // Needs seek
        let correction = sync.calculate_correction(15000, &state);
        assert!(matches!(correction, SyncCorrection::Seek(_)));
    }

    #[tokio::test]
    async fn test_room_manager() {
        let manager = RoomManager::new();

        let session = CollaborationSession {
            id: Uuid::new_v4(),
            session_type: SessionType::ListeningParty,
            name: "Test Session".to_string(),
            description: None,
            host_id: "host123".to_string(),
            created_at: Utc::now(),
            started_at: None,
            ended_at: None,
            is_public: true,
            max_participants: 10,
            settings: SessionSettings::default(),
        };

        let room_id = manager.create_room(session).await.unwrap();

        // Join room
        let (participant, _rx) = manager.join_room(
            room_id,
            "host123".to_string(),
            "Host User".to_string(),
            None,
        ).await.unwrap();

        assert_eq!(participant.role, ParticipantRole::Host);

        // Check participant count
        let count = manager.get_participant_count(room_id).await.unwrap();
        assert_eq!(count, 1);

        // Leave room
        manager.leave_room(room_id, "host123").await.unwrap();

        // Room should be removed when empty
        assert!(manager.get_participant_count(room_id).await.is_err());
    }

    #[test]
    fn test_audio_mixer() {
        let mut mixer = AudioMixer::new(48000, 10);

        // Add frames from two participants
        let frame1 = AudioFrame {
            user_id: "user1".to_string(),
            data: Bytes::from(vec![0u8; 2048]),
            sequence: 0,
            timestamp: Utc::now(),
            sample_rate: 48000,
            channels: 2,
        };

        let frame2 = AudioFrame {
            user_id: "user2".to_string(),
            data: Bytes::from(vec![0u8; 2048]),
            sequence: 0,
            timestamp: Utc::now(),
            sample_rate: 48000,
            channels: 2,
        };

        mixer.add_frame(frame1);
        mixer.add_frame(frame2);

        assert_eq!(mixer.stream_count(), 2);

        let mixed = mixer.mix();
        assert!(mixed.is_some());
    }
}
