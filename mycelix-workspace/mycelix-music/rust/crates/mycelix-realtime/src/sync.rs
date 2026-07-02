// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Playback synchronization

use crate::RealtimeResult;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Synchronized playback state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybackState {
    pub track_id: Option<Uuid>,
    pub position_ms: u64,
    pub is_playing: bool,
    pub timestamp: u64,
    pub version: u64,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            track_id: None,
            position_ms: 0,
            is_playing: false,
            timestamp: 0,
            version: 0,
        }
    }
}

/// Sync manager for coordinating playback
pub struct SyncManager {
    state: std::sync::RwLock<PlaybackState>,
    version: AtomicU64,
    tolerance_ms: u64,
}

impl SyncManager {
    pub fn new(tolerance_ms: u64) -> Self {
        Self {
            state: std::sync::RwLock::new(PlaybackState::default()),
            version: AtomicU64::new(0),
            tolerance_ms,
        }
    }

    pub fn update_state(&self, mut new_state: PlaybackState) -> bool {
        let current_version = self.version.load(Ordering::SeqCst);
        if new_state.version <= current_version {
            return false;
        }

        let new_version = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        new_state.version = new_version;

        let mut state = self.state.write().unwrap();
        *state = new_state;
        true
    }

    pub fn get_state(&self) -> PlaybackState {
        self.state.read().unwrap().clone()
    }

    pub fn check_sync(&self, client_position_ms: u64) -> SyncResult {
        let state = self.state.read().unwrap();

        if !state.is_playing {
            return SyncResult::Paused;
        }

        let diff = if client_position_ms > state.position_ms {
            client_position_ms - state.position_ms
        } else {
            state.position_ms - client_position_ms
        };

        if diff <= self.tolerance_ms {
            SyncResult::InSync
        } else if client_position_ms > state.position_ms {
            SyncResult::Ahead { diff_ms: diff }
        } else {
            SyncResult::Behind { diff_ms: diff }
        }
    }

    pub fn play(&self, track_id: Uuid, position_ms: u64) {
        let version = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        let mut state = self.state.write().unwrap();
        state.track_id = Some(track_id);
        state.position_ms = position_ms;
        state.is_playing = true;
        state.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        state.version = version;
    }

    pub fn pause(&self) {
        let version = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        let mut state = self.state.write().unwrap();
        state.is_playing = false;
        state.version = version;
    }

    pub fn seek(&self, position_ms: u64) {
        let version = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        let mut state = self.state.write().unwrap();
        state.position_ms = position_ms;
        state.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        state.version = version;
    }
}

impl Default for SyncManager {
    fn default() -> Self {
        Self::new(500) // 500ms tolerance
    }
}

/// Result of sync check
#[derive(Debug, Clone)]
pub enum SyncResult {
    InSync,
    Ahead { diff_ms: u64 },
    Behind { diff_ms: u64 },
    Paused,
}

/// Clock synchronization using NTP-like algorithm
pub struct ClockSync {
    offset_ms: std::sync::RwLock<i64>,
    samples: std::sync::RwLock<Vec<i64>>,
    max_samples: usize,
}

impl ClockSync {
    pub fn new(max_samples: usize) -> Self {
        Self {
            offset_ms: std::sync::RwLock::new(0),
            samples: std::sync::RwLock::new(Vec::new()),
            max_samples,
        }
    }

    pub fn add_sample(&self, local_send: u64, remote_recv: u64, remote_send: u64, local_recv: u64) {
        // Calculate round-trip time and offset
        let rtt = (local_recv - local_send) as i64 - (remote_send - remote_recv) as i64;
        let offset = ((remote_recv as i64 - local_send as i64) + (remote_send as i64 - local_recv as i64)) / 2;

        let mut samples = self.samples.write().unwrap();
        samples.push(offset);

        if samples.len() > self.max_samples {
            samples.remove(0);
        }

        // Update offset using median
        if !samples.is_empty() {
            let mut sorted = samples.clone();
            sorted.sort();
            let median = sorted[sorted.len() / 2];
            *self.offset_ms.write().unwrap() = median;
        }
    }

    pub fn get_offset(&self) -> i64 {
        *self.offset_ms.read().unwrap()
    }

    pub fn adjust_timestamp(&self, local_timestamp: u64) -> u64 {
        let offset = *self.offset_ms.read().unwrap();
        if offset >= 0 {
            local_timestamp + offset as u64
        } else {
            local_timestamp.saturating_sub((-offset) as u64)
        }
    }
}

impl Default for ClockSync {
    fn default() -> Self {
        Self::new(10)
    }
}
