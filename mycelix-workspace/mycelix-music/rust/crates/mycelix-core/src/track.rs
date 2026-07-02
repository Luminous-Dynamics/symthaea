// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Track representation and analysis

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::audio::AudioBuffer;

/// A music track with associated metadata and analysis
#[derive(Debug, Clone)]
pub struct Track {
    /// Unique track identifier
    pub id: Uuid,
    /// Audio data
    pub audio_buffer: AudioBuffer,
    /// Metadata
    pub metadata: TrackMetadata,
    /// Analysis results (populated after analysis)
    pub analysis: Option<TrackAnalysis>,
}

impl Track {
    /// Create a new track from an audio buffer
    pub fn new(id: Uuid, audio_buffer: AudioBuffer) -> Self {
        Self {
            id,
            audio_buffer,
            metadata: TrackMetadata::default(),
            analysis: None,
        }
    }

    /// Create with metadata
    pub fn with_metadata(id: Uuid, audio_buffer: AudioBuffer, metadata: TrackMetadata) -> Self {
        Self {
            id,
            audio_buffer,
            metadata,
            analysis: None,
        }
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.audio_buffer.duration() as f32
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.audio_buffer.sample_rate
    }

    /// Get number of channels
    pub fn channels(&self) -> u8 {
        self.audio_buffer.channels as u8
    }

    /// Check if track has been analyzed
    pub fn is_analyzed(&self) -> bool {
        self.analysis.is_some()
    }
}

/// Track metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrackMetadata {
    /// Track title
    pub title: Option<String>,
    /// Artist name
    pub artist: Option<String>,
    /// Album name
    pub album: Option<String>,
    /// Track number
    pub track_number: Option<u32>,
    /// Year released
    pub year: Option<u32>,
    /// User-defined tags
    pub tags: Vec<String>,
    /// Source file path
    pub source_path: Option<String>,
    /// File format
    pub format: Option<String>,
    /// Original bitrate (if encoded)
    pub bitrate: Option<u32>,
}

/// Analysis results for a track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackAnalysis {
    /// Detected genre
    pub genre: String,
    /// Detected mood
    pub mood: String,
    /// Tempo in BPM
    pub bpm: f32,
    /// Musical key
    pub key: String,
    /// Beat timestamps in seconds
    pub beats: Vec<f32>,
    /// Structural segments
    pub segments: Vec<String>,
    /// Audio embedding for similarity search
    pub embedding: Vec<f32>,
}

/// Detailed timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingInfo {
    /// Beat positions in seconds
    pub beats: Vec<f32>,
    /// Downbeat positions (first beat of each bar)
    pub downbeats: Vec<f32>,
    /// Tempo curve (time, bpm) pairs
    pub tempo_curve: Vec<(f32, f32)>,
    /// Time signature
    pub time_signature: (u8, u8),
}

/// Harmonic analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicInfo {
    /// Overall key
    pub key: String,
    /// Key confidence (0-1)
    pub key_confidence: f32,
    /// Chord progression
    pub chords: Vec<ChordEvent>,
    /// Key changes
    pub key_changes: Vec<KeyChange>,
}

/// A chord at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChordEvent {
    /// Start time in seconds
    pub start: f32,
    /// Duration in seconds
    pub duration: f32,
    /// Chord name (e.g., "Am", "G7")
    pub chord: String,
    /// Confidence (0-1)
    pub confidence: f32,
}

/// A key change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyChange {
    /// Time of key change in seconds
    pub time: f32,
    /// New key
    pub key: String,
}

/// Structural segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Segment label (e.g., "intro", "verse", "chorus")
    pub label: String,
    /// Confidence (0-1)
    pub confidence: f32,
}

/// Energy and dynamics information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsInfo {
    /// Overall loudness in LUFS
    pub loudness_lufs: f32,
    /// Peak level in dB
    pub peak_db: f32,
    /// Dynamic range in dB
    pub dynamic_range: f32,
    /// Energy curve (time, energy) pairs
    pub energy_curve: Vec<(f32, f32)>,
}

/// Complete analysis bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullAnalysis {
    /// Basic analysis
    pub basic: TrackAnalysis,
    /// Detailed timing
    pub timing: TimingInfo,
    /// Harmonic analysis
    pub harmonic: HarmonicInfo,
    /// Structural segments
    pub structure: Vec<Segment>,
    /// Dynamics
    pub dynamics: DynamicsInfo,
}

/// Track collection/playlist
#[derive(Debug, Clone, Default)]
pub struct TrackCollection {
    pub id: Uuid,
    pub name: String,
    pub tracks: Vec<Uuid>,
}

impl TrackCollection {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            tracks: Vec::new(),
        }
    }

    pub fn add(&mut self, track_id: Uuid) {
        if !self.tracks.contains(&track_id) {
            self.tracks.push(track_id);
        }
    }

    pub fn remove(&mut self, track_id: Uuid) {
        self.tracks.retain(|&id| id != track_id);
    }

    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::ChannelLayout;

    #[test]
    fn test_track_creation() {
        let buffer = AudioBuffer {
            samples: vec![0.0; 48000 * 2],
            sample_rate: 48000,
            channels: 2,
            layout: ChannelLayout::Stereo,
        };
        let track = Track::new(Uuid::new_v4(), buffer);

        assert_eq!(track.duration(), 1.0);
        assert_eq!(track.channels(), 2);
        assert!(!track.is_analyzed());
    }

    #[test]
    fn test_track_collection() {
        let mut collection = TrackCollection::new("My Playlist");
        let track_id = Uuid::new_v4();

        collection.add(track_id);
        assert_eq!(collection.len(), 1);

        collection.add(track_id); // Duplicate
        assert_eq!(collection.len(), 1);

        collection.remove(track_id);
        assert!(collection.is_empty());
    }
}
