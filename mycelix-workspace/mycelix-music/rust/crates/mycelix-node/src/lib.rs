// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Node.js Bindings
//!
//! Native Node.js module providing high-performance audio and streaming
//! capabilities to the TypeScript backend via napi-rs.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export core types
use mycelix_audio::{AudioBuffer, AudioFormat, AudioMetadata, Quality, TranscodeConfig};
use mycelix_streaming::{StreamQuality, StreamFormat, SegmentConfig};

// ============================================================================
// AUDIO PROCESSING
// ============================================================================

/// Audio analysis result returned to JavaScript
#[napi(object)]
pub struct JsAudioAnalysis {
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Bit depth (if available)
    pub bit_depth: Option<u8>,
    /// Integrated loudness in LUFS
    pub loudness_lufs: f64,
    /// True peak in dB
    pub true_peak_db: f64,
    /// Detected BPM
    pub bpm: f64,
    /// BPM confidence (0-1)
    pub bpm_confidence: f64,
    /// Detected key
    pub key: String,
    /// Key confidence (0-1)
    pub key_confidence: f64,
    /// RMS energy
    pub rms: f64,
    /// Spectral centroid
    pub spectral_centroid: f64,
}

/// Audio transcoding options
#[napi(object)]
pub struct JsTranscodeOptions {
    /// Target format (mp3, aac, flac, opus, wav)
    pub format: String,
    /// Target quality (low, medium, high, lossless)
    pub quality: String,
    /// Target sample rate (optional)
    pub sample_rate: Option<u32>,
    /// Target channels (optional, 1 for mono, 2 for stereo)
    pub channels: Option<u8>,
    /// Normalize audio
    pub normalize: bool,
    /// Target loudness in LUFS (optional)
    pub target_loudness: Option<f64>,
}

/// Audio transcoding result
#[napi(object)]
pub struct JsTranscodeResult {
    /// Transcoded audio data
    pub data: Buffer,
    /// Output format
    pub format: String,
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Output bitrate
    pub bitrate: u32,
    /// Processing time in milliseconds
    pub processing_time_ms: u32,
}

/// Waveform data for visualization
#[napi(object)]
pub struct JsWaveformData {
    /// Waveform samples (normalized 0-1)
    pub samples: Vec<f64>,
    /// Number of samples per second of audio
    pub samples_per_second: u32,
    /// Peak value
    pub peak: f64,
    /// RMS value
    pub rms: f64,
}

/// Audio processor class exposed to Node.js
#[napi]
pub struct AudioProcessor {
    sample_rate: u32,
}

#[napi]
impl AudioProcessor {
    /// Create a new audio processor
    #[napi(constructor)]
    pub fn new(sample_rate: Option<u32>) -> Self {
        Self {
            sample_rate: sample_rate.unwrap_or(44100),
        }
    }

    /// Analyze audio file and return comprehensive analysis
    #[napi]
    pub async fn analyze(&self, audio_data: Buffer) -> Result<JsAudioAnalysis> {
        let data = audio_data.to_vec();

        // Run analysis in blocking task to not block event loop
        let result = tokio::task::spawn_blocking(move || {
            // Detect format and decode
            let format = detect_format(&data)?;

            // Create transcoder to decode
            let transcoder = mycelix_audio::Transcoder::new(TranscodeConfig {
                source_format: format,
                target_format: AudioFormat::Wav,
                normalize: false,
                ..Default::default()
            });

            // Decode synchronously for analysis
            // In production, this would be properly async
            let buffer = decode_audio(&data, format)?;

            // Loudness analysis
            let mut loudness_analyzer = mycelix_audio::analysis::LoudnessAnalyzer::new(
                buffer.sample_rate,
                buffer.channels,
            );
            let loudness = loudness_analyzer.analyze(&buffer)
                .map_err(|e| Error::from_reason(e.to_string()))?;

            // BPM detection
            let bpm_detector = mycelix_audio::analysis::BpmDetector::new(buffer.sample_rate);
            let bpm_result = bpm_detector.detect(&buffer)
                .map_err(|e| Error::from_reason(e.to_string()))?;

            // Key detection
            let key_detector = mycelix_audio::analysis::KeyDetector::new(buffer.sample_rate);
            let key_result = key_detector.detect(&buffer)
                .map_err(|e| Error::from_reason(e.to_string()))?;

            // Spectral analysis
            let spectral_analyzer = mycelix_audio::analysis::SpectralAnalyzer::new(buffer.sample_rate);
            let spectral = spectral_analyzer.analyze(&buffer)
                .map_err(|e| Error::from_reason(e.to_string()))?;

            Ok::<_, Error>(JsAudioAnalysis {
                duration_ms: (buffer.duration() * 1000.0) as u32,
                sample_rate: buffer.sample_rate,
                channels: buffer.channels as u8,
                bit_depth: Some(16), // Assuming 16-bit for now
                loudness_lufs: loudness.integrated_lufs as f64,
                true_peak_db: loudness.true_peak_db as f64,
                bpm: bpm_result.bpm as f64,
                bpm_confidence: bpm_result.confidence as f64,
                key: key_result.key.name().to_string(),
                key_confidence: key_result.confidence as f64,
                rms: spectral.rms as f64,
                spectral_centroid: spectral.centroid as f64,
            })
        }).await
            .map_err(|e| Error::from_reason(e.to_string()))??;

        Ok(result)
    }

    /// Transcode audio to a different format
    #[napi]
    pub async fn transcode(
        &self,
        audio_data: Buffer,
        options: JsTranscodeOptions,
    ) -> Result<JsTranscodeResult> {
        let data = audio_data.to_vec();

        let result = tokio::task::spawn_blocking(move || {
            let source_format = detect_format(&data)?;
            let target_format = parse_format(&options.format)?;
            let target_quality = parse_quality(&options.quality)?;

            let config = TranscodeConfig {
                source_format,
                target_format,
                target_quality,
                target_sample_rate: options.sample_rate,
                target_channels: options.channels.map(|c| c as usize),
                normalize: options.normalize,
                target_loudness: options.target_loudness.map(|l| l as f32),
                ..Default::default()
            };

            let transcoder = mycelix_audio::Transcoder::new(config);

            // Synchronous transcode for now
            // In production, this would use the async version
            let start = std::time::Instant::now();
            let buffer = decode_audio(&data, source_format)?;
            let output = encode_audio(&buffer, target_format, target_quality)?;
            let processing_time = start.elapsed().as_millis() as u32;

            Ok::<_, Error>(JsTranscodeResult {
                data: Buffer::from(output),
                format: options.format,
                duration_ms: (buffer.duration() * 1000.0) as u32,
                bitrate: target_quality.bitrate(),
                processing_time_ms: processing_time,
            })
        }).await
            .map_err(|e| Error::from_reason(e.to_string()))??;

        Ok(result)
    }

    /// Generate waveform data for visualization
    #[napi]
    pub async fn generate_waveform(
        &self,
        audio_data: Buffer,
        samples_per_second: Option<u32>,
    ) -> Result<JsWaveformData> {
        let data = audio_data.to_vec();
        let sps = samples_per_second.unwrap_or(100);

        let result = tokio::task::spawn_blocking(move || {
            let format = detect_format(&data)?;
            let buffer = decode_audio(&data, format)?;

            // Convert to mono for waveform
            let mono = buffer.to_mono();
            let samples = &mono.samples;

            // Calculate samples per waveform point
            let samples_per_point = (mono.sample_rate / sps) as usize;
            let num_points = samples.len() / samples_per_point;

            let mut waveform = Vec::with_capacity(num_points);
            let mut peak = 0.0f32;
            let mut rms_sum = 0.0f32;

            for i in 0..num_points {
                let start = i * samples_per_point;
                let end = (start + samples_per_point).min(samples.len());

                // Calculate RMS for this segment
                let segment = &samples[start..end];
                let rms: f32 = (segment.iter().map(|s| s * s).sum::<f32>() / segment.len() as f32).sqrt();

                waveform.push(rms as f64);
                peak = peak.max(rms);
                rms_sum += rms * rms;
            }

            let overall_rms = (rms_sum / num_points as f32).sqrt();

            Ok::<_, Error>(JsWaveformData {
                samples: waveform,
                samples_per_second: sps,
                peak: peak as f64,
                rms: overall_rms as f64,
            })
        }).await
            .map_err(|e| Error::from_reason(e.to_string()))??;

        Ok(result)
    }

    /// Normalize audio to target loudness
    #[napi]
    pub async fn normalize(
        &self,
        audio_data: Buffer,
        target_lufs: Option<f64>,
    ) -> Result<Buffer> {
        let data = audio_data.to_vec();
        let target = target_lufs.unwrap_or(-14.0);

        let result = tokio::task::spawn_blocking(move || {
            let format = detect_format(&data)?;
            let mut buffer = decode_audio(&data, format)?;

            // Analyze current loudness
            let mut analyzer = mycelix_audio::analysis::LoudnessAnalyzer::new(
                buffer.sample_rate,
                buffer.channels,
            );
            let loudness = analyzer.analyze(&buffer)
                .map_err(|e| Error::from_reason(e.to_string()))?;

            // Calculate gain needed
            let current_lufs = loudness.integrated_lufs;
            let gain_db = target as f32 - current_lufs;
            let gain_linear = 10.0f32.powf(gain_db / 20.0);

            // Apply gain
            for sample in &mut buffer.samples {
                *sample *= gain_linear;

                // Soft clip to prevent clipping
                *sample = (*sample).tanh();
            }

            // Re-encode
            let output = encode_audio(&buffer, format, Quality::High)?;
            Ok::<_, Error>(Buffer::from(output))
        }).await
            .map_err(|e| Error::from_reason(e.to_string()))??;

        Ok(result)
    }
}

// ============================================================================
// STREAMING
// ============================================================================

/// Segment generation options
#[napi(object)]
pub struct JsSegmentOptions {
    /// Segment duration in seconds
    pub segment_duration: f64,
    /// Quality variants to generate
    pub qualities: Vec<String>,
    /// Enable DRM encryption
    pub enable_drm: bool,
    /// Generate HLS segments
    pub generate_hls: bool,
    /// Generate DASH segments
    pub generate_dash: bool,
}

/// Generated segment info
#[napi(object)]
pub struct JsSegmentInfo {
    pub id: String,
    pub sequence: u32,
    pub duration: f64,
    pub start_time: f64,
    pub quality: String,
    pub is_init: bool,
    pub size_bytes: u32,
}

/// Segmentation result
#[napi(object)]
pub struct JsSegmentationResult {
    pub track_id: String,
    pub segments: Vec<JsSegmentInfo>,
    pub hls_master_playlist: Option<String>,
    pub dash_mpd: Option<String>,
    pub duration: f64,
    pub processing_time_ms: u32,
}

/// Streaming engine exposed to Node.js
#[napi]
pub struct StreamingEngine {
    config: SegmentConfig,
}

#[napi]
impl StreamingEngine {
    /// Create a new streaming engine
    #[napi(constructor)]
    pub fn new(options: Option<JsSegmentOptions>) -> Self {
        let config = match options {
            Some(opts) => SegmentConfig {
                segment_duration: opts.segment_duration as f32,
                qualities: opts.qualities.iter()
                    .filter_map(|q| parse_stream_quality(q).ok())
                    .collect(),
                enable_drm: opts.enable_drm,
                generate_hls: opts.generate_hls,
                generate_dash: opts.generate_dash,
                ..Default::default()
            },
            None => SegmentConfig::default(),
        };

        Self { config }
    }

    /// Generate segments from audio data
    #[napi]
    pub async fn generate_segments(
        &self,
        track_id: String,
        audio_data: Buffer,
        source_format: String,
    ) -> Result<JsSegmentationResult> {
        let data = audio_data.to_vec();
        let format = parse_format(&source_format)?;
        let config = self.config.clone();

        let result = tokio::task::spawn_blocking(move || {
            let start = std::time::Instant::now();
            let generator = mycelix_streaming::SegmentGenerator::new(config);

            // Generate segments (simplified for sync context)
            let buffer = decode_audio(&data, format)?;
            let duration = buffer.duration() as f64;

            // Generate mock segments for each quality
            let mut segments: Vec<JsSegmentInfo> = Vec::new();
            let segment_duration = 6.0;
            let num_segments = (duration / segment_duration).ceil() as u32;

            for quality in &["Low", "Medium", "High", "VeryHigh"] {
                for i in 0..num_segments {
                    segments.push(JsSegmentInfo {
                        id: format!("{}_{}_seg_{}", track_id, quality, i),
                        sequence: i,
                        duration: if i == num_segments - 1 {
                            duration - (i as f64 * segment_duration)
                        } else {
                            segment_duration
                        },
                        start_time: i as f64 * segment_duration,
                        quality: quality.to_string(),
                        is_init: i == 0,
                        size_bytes: 50000, // Placeholder
                    });
                }
            }

            let processing_time = start.elapsed().as_millis() as u32;

            Ok::<_, Error>(JsSegmentationResult {
                track_id,
                segments,
                hls_master_playlist: Some("#EXTM3U\n#EXT-X-VERSION:7".to_string()),
                dash_mpd: Some("<?xml version=\"1.0\"?>".to_string()),
                duration,
                processing_time_ms: processing_time,
            })
        }).await
            .map_err(|e| Error::from_reason(e.to_string()))??;

        Ok(result)
    }

    /// Get available quality variants
    #[napi]
    pub fn get_quality_variants(&self) -> Vec<String> {
        vec![
            "UltraLow".to_string(),
            "Low".to_string(),
            "Medium".to_string(),
            "High".to_string(),
            "VeryHigh".to_string(),
            "Lossless".to_string(),
        ]
    }
}

// ============================================================================
// REAL-TIME COLLABORATION
// ============================================================================

/// Collaboration session info
#[napi(object)]
pub struct JsSessionInfo {
    pub id: String,
    pub name: String,
    pub session_type: String,
    pub host_id: String,
    pub participant_count: u32,
    pub is_public: bool,
    pub created_at: String,
}

/// Participant info
#[napi(object)]
pub struct JsParticipant {
    pub id: String,
    pub user_id: String,
    pub display_name: String,
    pub role: String,
    pub is_muted: bool,
    pub is_speaking: bool,
    pub connection_quality: String,
}

/// Playback state
#[napi(object)]
pub struct JsPlaybackState {
    pub track_id: Option<String>,
    pub position_ms: u32,
    pub is_playing: bool,
    pub speed: f64,
    pub volume: f64,
    pub version: u32,
}

/// Real-time collaboration manager
#[napi]
pub struct CollaborationManager {
    room_manager: Arc<RwLock<mycelix_realtime::RoomManager>>,
}

#[napi]
impl CollaborationManager {
    /// Create a new collaboration manager
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            room_manager: Arc::new(RwLock::new(mycelix_realtime::RoomManager::new())),
        }
    }

    /// Create a new collaboration session
    #[napi]
    pub async fn create_session(
        &self,
        name: String,
        session_type: String,
        host_id: String,
        is_public: bool,
        max_participants: u32,
    ) -> Result<String> {
        let session = mycelix_realtime::CollaborationSession {
            id: uuid::Uuid::new_v4(),
            session_type: parse_session_type(&session_type)?,
            name,
            description: None,
            host_id,
            created_at: chrono::Utc::now(),
            started_at: None,
            ended_at: None,
            is_public,
            max_participants,
            settings: mycelix_realtime::SessionSettings::default(),
        };

        let id = session.id;
        self.room_manager.write().await.create_room(session).await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(id.to_string())
    }

    /// Join a session
    #[napi]
    pub async fn join_session(
        &self,
        session_id: String,
        user_id: String,
        display_name: String,
    ) -> Result<JsParticipant> {
        let id = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        let (participant, _rx) = self.room_manager.write().await
            .join_room(id, user_id, display_name, None).await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(JsParticipant {
            id: participant.id.to_string(),
            user_id: participant.user_id,
            display_name: participant.display_name,
            role: format!("{:?}", participant.role),
            is_muted: participant.is_muted,
            is_speaking: participant.is_speaking,
            connection_quality: format!("{:?}", participant.connection_quality),
        })
    }

    /// Leave a session
    #[napi]
    pub async fn leave_session(&self, session_id: String, user_id: String) -> Result<()> {
        let id = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        self.room_manager.write().await
            .leave_room(id, &user_id).await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(())
    }

    /// Get session info
    #[napi]
    pub async fn get_session_info(&self, session_id: String) -> Result<JsSessionInfo> {
        let id = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        let (session, participants, _) = self.room_manager.read().await
            .get_room_state(id).await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(JsSessionInfo {
            id: session.id.to_string(),
            name: session.name,
            session_type: format!("{:?}", session.session_type),
            host_id: session.host_id,
            participant_count: participants.len() as u32,
            is_public: session.is_public,
            created_at: session.created_at.to_rfc3339(),
        })
    }

    /// Get playback state
    #[napi]
    pub async fn get_playback_state(&self, session_id: String) -> Result<JsPlaybackState> {
        let id = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        let (_, _, state) = self.room_manager.read().await
            .get_room_state(id).await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(JsPlaybackState {
            track_id: state.track_id,
            position_ms: state.position_ms as u32,
            is_playing: state.is_playing,
            speed: state.speed as f64,
            volume: state.volume as f64,
            version: state.version as u32,
        })
    }

    /// List active sessions
    #[napi]
    pub async fn list_sessions(&self) -> Vec<JsSessionInfo> {
        let rooms = self.room_manager.read().await.list_rooms().await;

        rooms.into_iter()
            .map(|(id, name, count)| JsSessionInfo {
                id: id.to_string(),
                name,
                session_type: "Unknown".to_string(),
                host_id: String::new(),
                participant_count: count as u32,
                is_public: true,
                created_at: String::new(),
            })
            .collect()
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn detect_format(data: &[u8]) -> Result<AudioFormat> {
    // Check magic bytes
    if data.len() >= 4 {
        if &data[0..4] == b"fLaC" {
            return Ok(AudioFormat::Flac);
        }
        if &data[0..4] == b"RIFF" {
            return Ok(AudioFormat::Wav);
        }
        if &data[0..4] == b"OggS" {
            return Ok(AudioFormat::Ogg);
        }
        if data[0..3] == [0xFF, 0xFB, 0x90] || data[0..2] == [0xFF, 0xFB] {
            return Ok(AudioFormat::Mp3);
        }
        if &data[0..4] == b"ID3\x04" || &data[0..3] == b"ID3" {
            return Ok(AudioFormat::Mp3);
        }
    }

    // Default to MP3
    Ok(AudioFormat::Mp3)
}

fn parse_format(format: &str) -> Result<AudioFormat> {
    match format.to_lowercase().as_str() {
        "mp3" => Ok(AudioFormat::Mp3),
        "aac" | "m4a" => Ok(AudioFormat::Aac),
        "flac" => Ok(AudioFormat::Flac),
        "opus" => Ok(AudioFormat::Opus),
        "wav" => Ok(AudioFormat::Wav),
        "ogg" => Ok(AudioFormat::Ogg),
        _ => Err(Error::from_reason(format!("Unsupported format: {}", format))),
    }
}

fn parse_quality(quality: &str) -> Result<Quality> {
    match quality.to_lowercase().as_str() {
        "low" => Ok(Quality::Low),
        "medium" => Ok(Quality::Medium),
        "high" => Ok(Quality::High),
        "lossless" => Ok(Quality::Lossless),
        _ => Err(Error::from_reason(format!("Unsupported quality: {}", quality))),
    }
}

fn parse_stream_quality(quality: &str) -> Result<StreamQuality> {
    match quality.to_lowercase().as_str() {
        "ultralow" => Ok(StreamQuality::UltraLow),
        "low" => Ok(StreamQuality::Low),
        "medium" => Ok(StreamQuality::Medium),
        "high" => Ok(StreamQuality::High),
        "veryhigh" => Ok(StreamQuality::VeryHigh),
        "lossless" => Ok(StreamQuality::Lossless),
        _ => Err(Error::from_reason(format!("Unsupported stream quality: {}", quality))),
    }
}

fn parse_session_type(session_type: &str) -> Result<mycelix_realtime::SessionType> {
    match session_type.to_lowercase().as_str() {
        "livejam" | "live_jam" => Ok(mycelix_realtime::SessionType::LiveJam),
        "listeningparty" | "listening_party" => Ok(mycelix_realtime::SessionType::ListeningParty),
        "playlistcollab" | "playlist_collab" => Ok(mycelix_realtime::SessionType::PlaylistCollab),
        "djsession" | "dj_session" => Ok(mycelix_realtime::SessionType::DjSession),
        _ => Err(Error::from_reason(format!("Unsupported session type: {}", session_type))),
    }
}

fn decode_audio(data: &[u8], _format: AudioFormat) -> Result<AudioBuffer> {
    // Simplified decode - in production would use symphonia
    Ok(AudioBuffer::new(44100, 2, mycelix_audio::ChannelLayout::Stereo))
}

fn encode_audio(buffer: &AudioBuffer, _format: AudioFormat, _quality: Quality) -> Result<Vec<u8>> {
    // Simplified encode - in production would use proper encoders
    Ok(Vec::new())
}

// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

#[napi]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[napi]
pub fn get_supported_formats() -> Vec<String> {
    vec![
        "mp3".to_string(),
        "aac".to_string(),
        "flac".to_string(),
        "opus".to_string(),
        "wav".to_string(),
        "ogg".to_string(),
    ]
}

#[napi]
pub fn get_supported_qualities() -> Vec<String> {
    vec![
        "low".to_string(),
        "medium".to_string(),
        "high".to_string(),
        "lossless".to_string(),
    ]
}
