// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Streaming Engine
//!
//! High-performance streaming infrastructure providing:
//! - HLS (HTTP Live Streaming) segment generation
//! - DASH (Dynamic Adaptive Streaming over HTTP) support
//! - Adaptive bitrate streaming
//! - Live streaming with low latency
//! - DRM integration
//! - CDN-optimized delivery

pub mod hls;
pub mod dash;
pub mod segment;
pub mod manifest;
pub mod drm;
pub mod cdn;
pub mod live;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

// ============================================================================
// ERROR TYPES
// ============================================================================

#[derive(Error, Debug)]
pub enum StreamingError {
    #[error("Segment not found: {0}")]
    SegmentNotFound(String),

    #[error("Track not found: {0}")]
    TrackNotFound(String),

    #[error("Invalid bitrate: {0}")]
    InvalidBitrate(u32),

    #[error("Encoding error: {0}")]
    EncodingError(String),

    #[error("DRM error: {0}")]
    DrmError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Stream is not live")]
    StreamNotLive,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type StreamingResult<T> = Result<T, StreamingError>;

// ============================================================================
// CORE TYPES
// ============================================================================

/// Streaming quality variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamQuality {
    /// 64 kbps - Ultra low bandwidth
    UltraLow,
    /// 96 kbps - Low quality
    Low,
    /// 160 kbps - Medium quality
    Medium,
    /// 256 kbps - High quality
    High,
    /// 320 kbps - Very high quality
    VeryHigh,
    /// Lossless - FLAC/ALAC
    Lossless,
}

impl StreamQuality {
    pub fn bitrate(&self) -> u32 {
        match self {
            Self::UltraLow => 64_000,
            Self::Low => 96_000,
            Self::Medium => 160_000,
            Self::High => 256_000,
            Self::VeryHigh => 320_000,
            Self::Lossless => 1_411_000, // CD quality approximation
        }
    }

    pub fn codec(&self) -> &'static str {
        match self {
            Self::Lossless => "flac",
            _ => "aac",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::UltraLow => "Ultra Low",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::VeryHigh => "Very High",
            Self::Lossless => "Lossless",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::UltraLow => "ultra_low",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::VeryHigh => "very_high",
            Self::Lossless => "lossless",
        }
    }

    pub fn resolution(&self) -> &'static str {
        // Audio doesn't have resolution, but keeping for compatibility
        "AUDIO"
    }
}

/// Streaming format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamFormat {
    /// HTTP Live Streaming (Apple)
    Hls,
    /// Dynamic Adaptive Streaming over HTTP
    Dash,
    /// Progressive download
    Progressive,
}

/// Audio segment
#[derive(Debug, Clone)]
pub struct Segment {
    /// Segment identifier
    pub id: String,
    /// Segment sequence number
    pub sequence: u32,
    /// Duration in seconds
    pub duration: f32,
    /// Start time in seconds
    pub start_time: f32,
    /// Segment data
    pub data: Bytes,
    /// Quality variant
    pub quality: StreamQuality,
    /// Is this an initialization segment?
    pub is_init: bool,
    /// Encryption info if DRM-protected
    pub encryption: Option<SegmentEncryption>,
}

/// Segment encryption info
#[derive(Debug, Clone)]
pub struct SegmentEncryption {
    /// Encryption method
    pub method: EncryptionMethod,
    /// Key ID
    pub key_id: String,
    /// Initialization vector
    pub iv: Option<Vec<u8>>,
}

/// Encryption method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionMethod {
    /// AES-128
    Aes128,
    /// Sample AES
    SampleAes,
    /// CENC (Common Encryption)
    Cenc,
    /// CBCS (Common Encryption with CBC)
    Cbcs,
}

// ============================================================================
// STREAMING SESSION
// ============================================================================

/// Active streaming session
#[derive(Debug, Clone)]
pub struct StreamSession {
    /// Session identifier
    pub id: Uuid,
    /// User identifier
    pub user_id: String,
    /// Track being streamed
    pub track_id: String,
    /// Current quality
    pub quality: StreamQuality,
    /// Format
    pub format: StreamFormat,
    /// Session start time
    pub started_at: DateTime<Utc>,
    /// Last activity time
    pub last_activity: DateTime<Utc>,
    /// Current playback position in seconds
    pub position: f32,
    /// Device information
    pub device_info: DeviceInfo,
    /// DRM license if applicable
    pub drm_license: Option<DrmLicense>,
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub device_id: String,
    pub device_type: String,
    pub os: String,
    pub os_version: String,
    pub app_version: String,
    pub supports_drm: bool,
    pub max_quality: StreamQuality,
}

/// DRM license
#[derive(Debug, Clone)]
pub struct DrmLicense {
    pub license_id: String,
    pub key_id: String,
    pub expires_at: DateTime<Utc>,
    pub policy: DrmPolicy,
}

/// DRM policy
#[derive(Debug, Clone)]
pub struct DrmPolicy {
    pub allow_offline: bool,
    pub offline_duration_hours: u32,
    pub max_concurrent_streams: u32,
    pub hdcp_required: bool,
}

// ============================================================================
// SEGMENT GENERATOR
// ============================================================================

/// Configuration for segment generation
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Target segment duration in seconds
    pub segment_duration: f32,
    /// Qualities to generate
    pub qualities: Vec<StreamQuality>,
    /// Enable DRM encryption
    pub enable_drm: bool,
    /// DRM key ID
    pub drm_key_id: Option<String>,
    /// Generate HLS segments
    pub generate_hls: bool,
    /// Generate DASH segments
    pub generate_dash: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            segment_duration: 6.0,
            qualities: vec![
                StreamQuality::Low,
                StreamQuality::Medium,
                StreamQuality::High,
                StreamQuality::VeryHigh,
            ],
            enable_drm: false,
            drm_key_id: None,
            generate_hls: true,
            generate_dash: true,
        }
    }
}

/// Segment generation result
#[derive(Debug)]
pub struct SegmentationResult {
    /// Track ID
    pub track_id: String,
    /// Generated segments by quality
    pub segments: HashMap<StreamQuality, Vec<Segment>>,
    /// HLS master playlist
    pub hls_master_playlist: Option<String>,
    /// HLS variant playlists by quality
    pub hls_variant_playlists: HashMap<StreamQuality, String>,
    /// DASH MPD manifest
    pub dash_mpd: Option<String>,
    /// Total duration
    pub duration: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Audio segment generator
pub struct SegmentGenerator {
    config: SegmentConfig,
}

impl SegmentGenerator {
    pub fn new(config: SegmentConfig) -> Self {
        Self { config }
    }

    /// Generate segments from audio data
    pub async fn generate(
        &self,
        track_id: &str,
        audio_data: &[u8],
        source_format: mycelix_audio::AudioFormat,
    ) -> StreamingResult<SegmentationResult> {
        let start = std::time::Instant::now();

        // Decode audio
        let transcoder = mycelix_audio::Transcoder::new(mycelix_audio::TranscodeConfig {
            source_format,
            target_format: mycelix_audio::AudioFormat::Aac,
            target_quality: mycelix_audio::Quality::High,
            normalize: true,
            target_loudness: Some(-14.0),
            ..Default::default()
        });

        // Generate segments for each quality
        let mut segments: HashMap<StreamQuality, Vec<Segment>> = HashMap::new();
        let mut duration = 0.0f32;

        for quality in &self.config.qualities {
            let quality_segments = self.generate_quality_segments(
                track_id,
                audio_data,
                *quality,
            ).await?;

            if !quality_segments.is_empty() {
                duration = quality_segments.iter()
                    .map(|s| s.start_time + s.duration)
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
            }

            segments.insert(*quality, quality_segments);
        }

        // Generate manifests
        let hls_master_playlist = if self.config.generate_hls {
            Some(self.generate_hls_master(track_id, &segments))
        } else {
            None
        };

        let hls_variant_playlists = if self.config.generate_hls {
            self.generate_hls_variants(track_id, &segments)
        } else {
            HashMap::new()
        };

        let dash_mpd = if self.config.generate_dash {
            Some(self.generate_dash_mpd(track_id, &segments, duration))
        } else {
            None
        };

        let processing_time = start.elapsed().as_millis() as u64;

        Ok(SegmentationResult {
            track_id: track_id.to_string(),
            segments,
            hls_master_playlist,
            hls_variant_playlists,
            dash_mpd,
            duration,
            processing_time_ms: processing_time,
        })
    }

    async fn generate_quality_segments(
        &self,
        track_id: &str,
        _audio_data: &[u8],
        quality: StreamQuality,
    ) -> StreamingResult<Vec<Segment>> {
        // In production, this would:
        // 1. Transcode to target quality/bitrate
        // 2. Split into segments
        // 3. Apply DRM encryption if enabled
        // 4. Generate initialization segment

        let mut segments = Vec::new();
        let segment_duration = self.config.segment_duration;

        // Placeholder: Generate mock segments
        // Real implementation would use FFmpeg or similar
        let total_duration = 180.0; // 3 minutes placeholder
        let num_segments = (total_duration / segment_duration).ceil() as u32;

        // Add initialization segment
        segments.push(Segment {
            id: format!("{}_{}_{}_init", track_id, quality.label(), 0),
            sequence: 0,
            duration: 0.0,
            start_time: 0.0,
            data: Bytes::from(vec![0u8; 1024]), // Placeholder init segment
            quality,
            is_init: true,
            encryption: self.create_encryption_info(),
        });

        // Add media segments
        for i in 0..num_segments {
            let start_time = i as f32 * segment_duration;
            let duration = if i == num_segments - 1 {
                total_duration - start_time
            } else {
                segment_duration
            };

            segments.push(Segment {
                id: format!("{}_{}_{}", track_id, quality.label(), i + 1),
                sequence: i + 1,
                duration,
                start_time,
                data: Bytes::from(vec![0u8; (quality.bitrate() as f32 / 8.0 * duration) as usize]),
                quality,
                is_init: false,
                encryption: self.create_encryption_info(),
            });
        }

        Ok(segments)
    }

    fn create_encryption_info(&self) -> Option<SegmentEncryption> {
        if !self.config.enable_drm {
            return None;
        }

        Some(SegmentEncryption {
            method: EncryptionMethod::Cenc,
            key_id: self.config.drm_key_id.clone().unwrap_or_default(),
            iv: None, // Would be generated per-segment
        })
    }

    fn generate_hls_master(
        &self,
        track_id: &str,
        segments: &HashMap<StreamQuality, Vec<Segment>>,
    ) -> String {
        let mut playlist = String::from("#EXTM3U\n#EXT-X-VERSION:7\n\n");

        for quality in &self.config.qualities {
            if segments.contains_key(quality) {
                let bitrate = quality.bitrate();
                let codec = quality.codec();
                let variant_url = format!("{}/{}/playlist.m3u8", track_id, quality.label());

                playlist.push_str(&format!(
                    "#EXT-X-STREAM-INF:BANDWIDTH={},CODECS=\"mp4a.40.2\",AUDIO=\"audio\"\n{}\n",
                    bitrate, variant_url
                ));
            }
        }

        playlist
    }

    fn generate_hls_variants(
        &self,
        track_id: &str,
        segments: &HashMap<StreamQuality, Vec<Segment>>,
    ) -> HashMap<StreamQuality, String> {
        let mut playlists = HashMap::new();

        for (quality, segs) in segments {
            let mut playlist = String::from("#EXTM3U\n#EXT-X-VERSION:7\n");
            playlist.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", self.config.segment_duration.ceil() as u32));
            playlist.push_str("#EXT-X-MEDIA-SEQUENCE:0\n");
            playlist.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n\n");

            // Add map for initialization segment
            if let Some(init_seg) = segs.iter().find(|s| s.is_init) {
                playlist.push_str(&format!(
                    "#EXT-X-MAP:URI=\"{}/init.m4s\"\n\n",
                    track_id
                ));
            }

            // Add media segments
            for seg in segs.iter().filter(|s| !s.is_init) {
                playlist.push_str(&format!("#EXTINF:{:.3},\n", seg.duration));
                playlist.push_str(&format!("{}/{}.m4s\n", track_id, seg.sequence));
            }

            playlist.push_str("#EXT-X-ENDLIST\n");
            playlists.insert(*quality, playlist);
        }

        playlists
    }

    fn generate_dash_mpd(
        &self,
        track_id: &str,
        segments: &HashMap<StreamQuality, Vec<Segment>>,
        duration: f32,
    ) -> String {
        let duration_iso = format!("PT{:.3}S", duration);

        let mut mpd = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"
     xmlns:cenc="urn:mpeg:cenc:2013"
     type="static"
     mediaPresentationDuration="{}"
     minBufferTime="PT2S"
     profiles="urn:mpeg:dash:profile:isoff-on-demand:2011">
  <Period id="0" start="PT0S">
    <AdaptationSet mimeType="audio/mp4" lang="en" segmentAlignment="true">
"#,
            duration_iso
        );

        for quality in &self.config.qualities {
            if let Some(segs) = segments.get(quality) {
                let bitrate = quality.bitrate();
                let segment_count = segs.iter().filter(|s| !s.is_init).count();

                mpd.push_str(&format!(
                    r#"      <Representation id="{}" bandwidth="{}" codecs="mp4a.40.2">
        <SegmentTemplate media="{}/{}/$Number$.m4s"
                         initialization="{}/{}/init.m4s"
                         timescale="1000"
                         duration="{}"/>
      </Representation>
"#,
                    quality.label(),
                    bitrate,
                    track_id,
                    quality.label(),
                    track_id,
                    quality.label(),
                    (self.config.segment_duration * 1000.0) as u32
                ));
            }
        }

        mpd.push_str("    </AdaptationSet>\n  </Period>\n</MPD>");
        mpd
    }
}

// ============================================================================
// STREAMING SERVICE
// ============================================================================

/// Streaming service for managing active streams
pub struct StreamingService {
    /// Active sessions
    sessions: Arc<RwLock<HashMap<Uuid, StreamSession>>>,
    /// Segment cache
    segment_cache: Arc<RwLock<HashMap<String, Segment>>>,
    /// Event broadcaster
    events: broadcast::Sender<StreamEvent>,
    /// Segment generator
    generator: SegmentGenerator,
}

/// Streaming events
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Session started
    SessionStarted { session_id: Uuid, user_id: String, track_id: String },
    /// Session ended
    SessionEnded { session_id: Uuid, duration: f32 },
    /// Quality changed
    QualityChanged { session_id: Uuid, old_quality: StreamQuality, new_quality: StreamQuality },
    /// Playback position updated
    PositionUpdated { session_id: Uuid, position: f32 },
    /// Buffer underrun
    BufferUnderrun { session_id: Uuid },
}

impl StreamingService {
    pub fn new(config: SegmentConfig) -> Self {
        let (events, _) = broadcast::channel(1000);

        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            segment_cache: Arc::new(RwLock::new(HashMap::new())),
            events,
            generator: SegmentGenerator::new(config),
        }
    }

    /// Start a new streaming session
    pub async fn start_session(
        &self,
        user_id: String,
        track_id: String,
        quality: StreamQuality,
        format: StreamFormat,
        device_info: DeviceInfo,
    ) -> StreamingResult<StreamSession> {
        let session = StreamSession {
            id: Uuid::new_v4(),
            user_id: user_id.clone(),
            track_id: track_id.clone(),
            quality,
            format,
            started_at: Utc::now(),
            last_activity: Utc::now(),
            position: 0.0,
            device_info,
            drm_license: None,
        };

        self.sessions.write().await.insert(session.id, session.clone());

        let _ = self.events.send(StreamEvent::SessionStarted {
            session_id: session.id,
            user_id,
            track_id,
        });

        Ok(session)
    }

    /// Get a streaming session
    pub async fn get_session(&self, session_id: Uuid) -> Option<StreamSession> {
        self.sessions.read().await.get(&session_id).cloned()
    }

    /// Update session quality
    pub async fn update_quality(
        &self,
        session_id: Uuid,
        new_quality: StreamQuality,
    ) -> StreamingResult<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(&session_id) {
            let old_quality = session.quality;
            session.quality = new_quality;
            session.last_activity = Utc::now();

            let _ = self.events.send(StreamEvent::QualityChanged {
                session_id,
                old_quality,
                new_quality,
            });

            Ok(())
        } else {
            Err(StreamingError::SegmentNotFound(session_id.to_string()))
        }
    }

    /// Update playback position
    pub async fn update_position(
        &self,
        session_id: Uuid,
        position: f32,
    ) -> StreamingResult<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(&session_id) {
            session.position = position;
            session.last_activity = Utc::now();

            let _ = self.events.send(StreamEvent::PositionUpdated {
                session_id,
                position,
            });

            Ok(())
        } else {
            Err(StreamingError::SegmentNotFound(session_id.to_string()))
        }
    }

    /// End a streaming session
    pub async fn end_session(&self, session_id: Uuid) -> StreamingResult<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.remove(&session_id) {
            let duration = session.position;

            let _ = self.events.send(StreamEvent::SessionEnded {
                session_id,
                duration,
            });

            Ok(())
        } else {
            Err(StreamingError::SegmentNotFound(session_id.to_string()))
        }
    }

    /// Get a segment
    pub async fn get_segment(
        &self,
        track_id: &str,
        quality: StreamQuality,
        segment_id: u32,
    ) -> StreamingResult<Segment> {
        let cache_key = format!("{}_{}_{}", track_id, quality.label(), segment_id);

        // Check cache
        if let Some(segment) = self.segment_cache.read().await.get(&cache_key) {
            return Ok(segment.clone());
        }

        // Would fetch from storage in production
        Err(StreamingError::SegmentNotFound(cache_key))
    }

    /// Get manifest for track
    pub async fn get_manifest(
        &self,
        track_id: &str,
        format: StreamFormat,
    ) -> StreamingResult<String> {
        // Would fetch from storage in production
        match format {
            StreamFormat::Hls => {
                Ok(format!("#EXTM3U\n#EXT-X-VERSION:7\n# Placeholder for {}", track_id))
            }
            StreamFormat::Dash => {
                Ok(format!("<?xml version=\"1.0\"?>\n<!-- Placeholder for {} -->", track_id))
            }
            StreamFormat::Progressive => {
                Err(StreamingError::EncodingError("Progressive doesn't use manifests".to_string()))
            }
        }
    }

    /// Subscribe to streaming events
    pub fn subscribe(&self) -> broadcast::Receiver<StreamEvent> {
        self.events.subscribe()
    }

    /// Get active session count
    pub async fn active_session_count(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// Get sessions for user
    pub async fn get_user_sessions(&self, user_id: &str) -> Vec<StreamSession> {
        self.sessions
            .read()
            .await
            .values()
            .filter(|s| s.user_id == user_id)
            .cloned()
            .collect()
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self, max_idle_seconds: i64) {
        let now = Utc::now();
        let mut sessions = self.sessions.write().await;

        let expired: Vec<Uuid> = sessions
            .iter()
            .filter(|(_, s)| (now - s.last_activity).num_seconds() > max_idle_seconds)
            .map(|(id, _)| *id)
            .collect();

        for id in expired {
            if let Some(session) = sessions.remove(&id) {
                let _ = self.events.send(StreamEvent::SessionEnded {
                    session_id: id,
                    duration: session.position,
                });
            }
        }
    }
}

// ============================================================================
// ADAPTIVE BITRATE CONTROLLER
// ============================================================================

/// Bandwidth estimation for adaptive bitrate
#[derive(Debug, Clone)]
pub struct BandwidthEstimate {
    /// Estimated bandwidth in bits per second
    pub bandwidth_bps: u64,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,
    /// Latency in milliseconds
    pub latency_ms: u32,
    /// Packet loss rate
    pub packet_loss: f32,
}

/// Adaptive bitrate controller
pub struct AbrController {
    /// Bandwidth history
    bandwidth_history: Vec<(DateTime<Utc>, u64)>,
    /// Current estimate
    current_estimate: BandwidthEstimate,
    /// Buffer level in seconds
    buffer_level: f32,
    /// Min buffer before switching down
    min_buffer_switch_down: f32,
    /// Max buffer before switching up
    max_buffer_switch_up: f32,
}

impl AbrController {
    pub fn new() -> Self {
        Self {
            bandwidth_history: Vec::new(),
            current_estimate: BandwidthEstimate {
                bandwidth_bps: 1_000_000, // 1 Mbps default
                confidence: 0.0,
                latency_ms: 0,
                packet_loss: 0.0,
            },
            buffer_level: 0.0,
            min_buffer_switch_down: 5.0, // Switch down if buffer < 5s
            max_buffer_switch_up: 15.0, // Switch up if buffer > 15s
        }
    }

    /// Update bandwidth estimate
    pub fn update_bandwidth(&mut self, downloaded_bytes: u64, download_time_ms: u64) {
        let bandwidth_bps = if download_time_ms > 0 {
            downloaded_bytes * 8 * 1000 / download_time_ms
        } else {
            self.current_estimate.bandwidth_bps
        };

        self.bandwidth_history.push((Utc::now(), bandwidth_bps));

        // Keep last 20 samples
        if self.bandwidth_history.len() > 20 {
            self.bandwidth_history.remove(0);
        }

        // Calculate weighted average (more recent = more weight)
        let mut total_weight = 0.0f32;
        let mut weighted_sum = 0.0f64;

        for (i, (_, bw)) in self.bandwidth_history.iter().enumerate() {
            let weight = (i + 1) as f32;
            weighted_sum += *bw as f64 * weight as f64;
            total_weight += weight;
        }

        let avg_bandwidth = if total_weight > 0.0 {
            (weighted_sum / total_weight as f64) as u64
        } else {
            bandwidth_bps
        };

        self.current_estimate.bandwidth_bps = avg_bandwidth;
        self.current_estimate.confidence = (self.bandwidth_history.len() as f32 / 20.0).min(1.0);
    }

    /// Update buffer level
    pub fn update_buffer(&mut self, buffer_seconds: f32) {
        self.buffer_level = buffer_seconds;
    }

    /// Get recommended quality based on current conditions
    pub fn recommend_quality(&self, available_qualities: &[StreamQuality]) -> StreamQuality {
        // Conservative bandwidth usage (use 80% of estimated)
        let usable_bandwidth = (self.current_estimate.bandwidth_bps as f32 * 0.8) as u32;

        // Find best quality that fits bandwidth
        let mut best_quality = StreamQuality::Low;

        for quality in available_qualities {
            if quality.bitrate() <= usable_bandwidth {
                best_quality = *quality;
            }
        }

        // Adjust based on buffer level
        if self.buffer_level < self.min_buffer_switch_down {
            // Buffer is low, be conservative
            let current_idx = available_qualities.iter().position(|q| *q == best_quality).unwrap_or(0);
            if current_idx > 0 {
                best_quality = available_qualities[current_idx - 1];
            }
        } else if self.buffer_level > self.max_buffer_switch_up && self.current_estimate.confidence > 0.7 {
            // Buffer is healthy, can try higher quality
            let current_idx = available_qualities.iter().position(|q| *q == best_quality).unwrap_or(0);
            if current_idx < available_qualities.len() - 1 {
                best_quality = available_qualities[current_idx + 1];
            }
        }

        best_quality
    }

    /// Get current bandwidth estimate
    pub fn get_estimate(&self) -> &BandwidthEstimate {
        &self.current_estimate
    }
}

impl Default for AbrController {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_quality_bitrate() {
        assert_eq!(StreamQuality::Low.bitrate(), 96_000);
        assert_eq!(StreamQuality::High.bitrate(), 256_000);
        assert_eq!(StreamQuality::VeryHigh.bitrate(), 320_000);
    }

    #[test]
    fn test_abr_controller() {
        let mut abr = AbrController::new();

        // Simulate downloading 1MB in 1 second (8 Mbps)
        abr.update_bandwidth(1_000_000, 1000);

        let qualities = vec![
            StreamQuality::Low,
            StreamQuality::Medium,
            StreamQuality::High,
            StreamQuality::VeryHigh,
        ];

        abr.update_buffer(10.0); // Healthy buffer
        let recommended = abr.recommend_quality(&qualities);

        // Should recommend high quality given 8 Mbps bandwidth
        assert!(recommended.bitrate() >= StreamQuality::High.bitrate());
    }

    #[test]
    fn test_segment_generator() {
        let config = SegmentConfig::default();
        let generator = SegmentGenerator::new(config);

        // Test HLS master playlist generation
        let mut segments = HashMap::new();
        segments.insert(StreamQuality::Low, vec![]);
        segments.insert(StreamQuality::High, vec![]);

        let playlist = generator.generate_hls_master("track123", &segments);
        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("BANDWIDTH="));
    }

    #[tokio::test]
    async fn test_streaming_service() {
        let config = SegmentConfig::default();
        let service = StreamingService::new(config);

        let device = DeviceInfo {
            device_id: "test123".to_string(),
            device_type: "mobile".to_string(),
            os: "iOS".to_string(),
            os_version: "17.0".to_string(),
            app_version: "1.0.0".to_string(),
            supports_drm: true,
            max_quality: StreamQuality::VeryHigh,
        };

        let session = service
            .start_session(
                "user123".to_string(),
                "track456".to_string(),
                StreamQuality::High,
                StreamFormat::Hls,
                device,
            )
            .await
            .unwrap();

        assert_eq!(session.user_id, "user123");
        assert_eq!(session.track_id, "track456");
        assert_eq!(service.active_session_count().await, 1);

        service.end_session(session.id).await.unwrap();
        assert_eq!(service.active_session_count().await, 0);
    }
}
