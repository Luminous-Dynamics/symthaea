// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RDP Protocol v3 Extensions: Chat, RemoteExec, Audio, Clipboard, Support.
//!
//! These types extend the base `rdp_protocol.rs` without modifying it,
//! preventing conflicts with concurrent sessions editing that file.

use serde::{Deserialize, Serialize};

// Re-export base protocol types.
pub use super::rdp_protocol::*;

/// Maximum audio payload per frame (16 KB).
pub const RDP_AUDIO_MAX_CHUNK_BYTES: usize = 16384;

/// Audio encoding quality, consciousness-gated.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AudioEncoding {
    /// Raw PCM f32 stereo (highest fidelity, ~352 kbps at 44.1kHz).
    RawF32Stereo,
    /// Raw PCM i16 stereo (~176 kbps at 44.1kHz).
    RawI16Stereo,
    /// Mu-law compressed mono (~88 kbps at 44.1kHz).
    MuLawMono,
}

impl AudioEncoding {
    /// Select encoding based on peer consciousness level.
    pub fn from_phi(phi: f32) -> Self {
        if phi > 0.7 {
            Self::RawF32Stereo
        } else if phi > 0.4 {
            Self::RawI16Stereo
        } else {
            Self::MuLawMono
        }
    }
}

/// Audio frame with consciousness metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFrame {
    pub audio_seq: u64,
    pub timestamp_ms: u64,
    pub sample_rate: u32,
    pub num_samples: u16,
    pub data: Vec<u8>,
    pub encoding: AudioEncoding,
    pub consciousness_level: f32,
    pub allostatic_load: f32,
    pub dominant_harmony: u8,
}

/// Chat message between support agent and user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub sender: String,
    pub text: String,
    pub timestamp_ms: u64,
}

/// Remote command execution message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemoteExecMessage {
    Request {
        exec_id: u64,
        command: String,
        working_dir: Option<String>,
        timeout_ms: u64,
    },
    Output {
        exec_id: u64,
        stdout: String,
        stderr: String,
        exit_code: Option<i32>,
    },
    Cancel {
        exec_id: u64,
    },
}

/// Clipboard content format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClipboardFormat {
    PlainText,
    RichText,
    Image,
    Files,
}

/// Monitor information for multi-monitor support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorInfo {
    pub index: u8,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub is_primary: bool,
}
