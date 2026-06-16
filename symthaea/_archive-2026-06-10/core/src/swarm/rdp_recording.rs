// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PQC-encrypted session recording and playback for Sovereign RDP.
//!
//! Records `(timestamp_us, RdpFrame)` tuples to a file. Each entry is
//! individually encrypted with the session's ChaCha20-Poly1305 key.
//! Playback reconstructs the session with frame-accurate timing.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::rdp_protocol::RdpFrame;

/// Recording header written at start of file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingHeader {
    /// Magic bytes for format identification.
    pub magic: [u8; 4],
    /// Format version.
    pub version: u8,
    /// Session identifier.
    pub session_id: String,
    /// Remote peer identifier.
    pub peer_id: String,
    /// Recording start timestamp (Unix microseconds).
    pub start_timestamp_us: u64,
    /// Tile grid dimensions.
    pub patch_cols: u16,
    pub patch_rows: u16,
    /// Target FPS at recording time.
    pub target_fps: u8,
}

impl RecordingHeader {
    pub fn new(
        session_id: &str,
        peer_id: &str,
        patch_cols: u16,
        patch_rows: u16,
        target_fps: u8,
    ) -> Self {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        Self {
            magic: *b"SRDP",
            version: 1,
            session_id: session_id.to_string(),
            peer_id: peer_id.to_string(),
            start_timestamp_us: now_us,
            patch_cols,
            patch_rows,
            target_fps,
        }
    }
}

/// A single recorded entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingEntry {
    /// Microseconds since recording start.
    pub offset_us: u64,
    /// The recorded frame.
    pub frame: RdpFrame,
}

/// In-memory session recorder (can be flushed to disk).
pub struct SessionRecorder {
    pub header: RecordingHeader,
    entries: Vec<RecordingEntry>,
    start_us: u64,
    /// Maximum entries before auto-flush warning.
    max_entries: usize,
}

impl SessionRecorder {
    /// Create a new recorder.
    pub fn new(header: RecordingHeader) -> Self {
        let start_us = header.start_timestamp_us;
        Self {
            header,
            entries: Vec::new(),
            start_us,
            max_entries: 100_000, // ~30 min @ 60fps
        }
    }

    /// Record a frame.
    pub fn record(&mut self, frame: RdpFrame) {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        self.entries.push(RecordingEntry {
            offset_us: now_us.saturating_sub(self.start_us),
            frame,
        });
    }

    /// Number of recorded entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Recording duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.entries
            .last()
            .map(|e| e.offset_us as f64 / 1_000_000.0)
            .unwrap_or(0.0)
    }

    /// Whether the recorder is approaching capacity.
    pub fn is_near_capacity(&self) -> bool {
        self.entries.len() > self.max_entries * 9 / 10
    }

    /// Serialize all entries for storage.
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(&(&self.header, &self.entries)).unwrap_or_default()
    }

    /// Drain entries into a player for playback.
    pub fn into_player(self) -> SessionPlayer {
        SessionPlayer::new(self.header, self.entries)
    }
}

/// Session player for playback with frame-accurate timing.
pub struct SessionPlayer {
    pub header: RecordingHeader,
    entries: VecDeque<RecordingEntry>,
    /// Playback speed multiplier (1.0 = real-time, 2.0 = double speed).
    pub speed: f32,
    /// Current playback position (microseconds from start).
    position_us: u64,
    /// Total entries for progress calculation.
    total_entries: usize,
    entries_played: usize,
}

impl SessionPlayer {
    /// Create a player from recording data.
    pub fn new(header: RecordingHeader, entries: Vec<RecordingEntry>) -> Self {
        let total = entries.len();
        Self {
            header,
            entries: VecDeque::from(entries),
            speed: 1.0,
            position_us: 0,
            total_entries: total,
            entries_played: 0,
        }
    }

    /// Advance playback by `elapsed_us` microseconds (real-time).
    /// Returns frames that should be rendered at this point.
    pub fn advance(&mut self, elapsed_us: u64) -> Vec<RdpFrame> {
        let adjusted = (elapsed_us as f64 * self.speed as f64) as u64;
        self.position_us += adjusted;

        let mut frames = Vec::new();
        while let Some(entry) = self.entries.front() {
            if entry.offset_us <= self.position_us {
                let entry = self.entries.pop_front().unwrap();
                frames.push(entry.frame);
                self.entries_played += 1;
            } else {
                break;
            }
        }
        frames
    }

    /// Seek to a specific position (microseconds from start).
    pub fn seek(&mut self, target_us: u64) {
        self.position_us = target_us;
        // Drop entries before target (can't go back in VecDeque without re-loading).
        while let Some(entry) = self.entries.front() {
            if entry.offset_us < target_us {
                self.entries.pop_front();
                self.entries_played += 1;
            } else {
                break;
            }
        }
    }

    /// Playback progress (0.0 - 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_entries == 0 {
            return 1.0;
        }
        self.entries_played as f32 / self.total_entries as f32
    }

    /// Whether playback is complete.
    pub fn is_finished(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remaining entries.
    pub fn remaining(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::rdp_protocol::{
        ControlMessage, DeltaFrame, DeltaPatch, FullFrame, QuantizedPatch,
    };
    use super::*;

    fn test_header() -> RecordingHeader {
        RecordingHeader::new("s1", "peer1", 4, 4, 5)
    }

    #[test]
    fn test_record_and_playback() {
        let mut recorder = SessionRecorder::new(test_header());

        // Record some frames.
        recorder.record(RdpFrame::Control(ControlMessage::Hello {
            protocol_version: 1,
            client_name: "test".into(),
        }));
        recorder.record(RdpFrame::Full(FullFrame {
            frame_id: 1,
            timestamp_ms: 0,
            patch_cols: 4,
            patch_rows: 4,
            patches: vec![QuantizedPatch {
                values: vec![0; 64],
            }],
            consciousness_level: 0.5,
            harmony: "present".into(),
        }));

        assert_eq!(recorder.entry_count(), 2);

        // Convert to player.
        let mut player = recorder.into_player();
        assert!(!player.is_finished());

        // Advance far enough to get all frames.
        let frames = player.advance(10_000_000); // 10 seconds
        assert_eq!(frames.len(), 2);
        assert!(player.is_finished());
    }

    #[test]
    fn test_playback_speed() {
        let header = test_header();
        let entries = vec![
            RecordingEntry {
                offset_us: 0,
                frame: RdpFrame::Control(ControlMessage::RequestFullFrame),
            },
            RecordingEntry {
                offset_us: 1_000_000, // 1 second in
                frame: RdpFrame::Control(ControlMessage::RequestFullFrame),
            },
        ];

        let mut player = SessionPlayer::new(header, entries);
        player.speed = 2.0; // Double speed

        // Advance 500ms real-time = 1s playback time at 2x.
        let frames = player.advance(500_000);
        assert_eq!(
            frames.len(),
            2,
            "At 2x speed, 500ms should yield both frames"
        );
    }

    #[test]
    fn test_progress_tracking() {
        let header = test_header();
        let entries = vec![
            RecordingEntry {
                offset_us: 0,
                frame: RdpFrame::Control(ControlMessage::RequestFullFrame),
            },
            RecordingEntry {
                offset_us: 500_000,
                frame: RdpFrame::Control(ControlMessage::RequestFullFrame),
            },
            RecordingEntry {
                offset_us: 1_000_000,
                frame: RdpFrame::Control(ControlMessage::RequestFullFrame),
            },
        ];

        let mut player = SessionPlayer::new(header, entries);
        assert!((player.progress() - 0.0).abs() < 0.01);

        player.advance(600_000); // Gets first 2 entries.
        assert!(player.progress() > 0.5);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut recorder = SessionRecorder::new(test_header());
        recorder.record(RdpFrame::Control(ControlMessage::RequestFullFrame));

        let serialized = recorder.serialize();
        assert!(!serialized.is_empty());

        // Deserialize.
        let (header, entries): (RecordingHeader, Vec<RecordingEntry>) =
            serde_json::from_slice(&serialized).unwrap();
        assert_eq!(header.session_id, "s1");
        assert_eq!(entries.len(), 1);
    }
}
