// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Presence and cursor tracking

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

/// User presence state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresenceState {
    Online,
    Away,
    Busy,
    Offline,
}

/// User presence info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Presence {
    pub user_id: Uuid,
    pub display_name: String,
    pub state: PresenceState,
    pub last_seen: u64,
    pub cursor_position: Option<CursorPosition>,
    pub current_activity: Option<String>,
}

/// Cursor position for collaborative editing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    pub x: f32,
    pub y: f32,
    pub context: CursorContext,
}

/// Where the cursor is located
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CursorContext {
    Playlist { playlist_id: Uuid, index: usize },
    Waveform { track_id: Uuid, time_ms: u64 },
    Tracklist { index: usize },
    Custom { name: String },
}

/// Presence manager
pub struct PresenceManager {
    presence: RwLock<HashMap<Uuid, Presence>>,
    timeout: Duration,
}

impl PresenceManager {
    pub fn new(timeout: Duration) -> Self {
        Self {
            presence: RwLock::new(HashMap::new()),
            timeout,
        }
    }

    pub async fn update_presence(&self, presence: Presence) {
        let mut map = self.presence.write().await;
        map.insert(presence.user_id, presence);
    }

    pub async fn set_state(&self, user_id: Uuid, state: PresenceState) {
        let mut map = self.presence.write().await;
        if let Some(presence) = map.get_mut(&user_id) {
            presence.state = state;
            presence.last_seen = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
    }

    pub async fn update_cursor(&self, user_id: Uuid, cursor: CursorPosition) {
        let mut map = self.presence.write().await;
        if let Some(presence) = map.get_mut(&user_id) {
            presence.cursor_position = Some(cursor);
            presence.last_seen = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
    }

    pub async fn heartbeat(&self, user_id: Uuid) {
        let mut map = self.presence.write().await;
        if let Some(presence) = map.get_mut(&user_id) {
            presence.last_seen = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            if presence.state == PresenceState::Away {
                presence.state = PresenceState::Online;
            }
        }
    }

    pub async fn get_presence(&self, user_id: Uuid) -> Option<Presence> {
        let map = self.presence.read().await;
        map.get(&user_id).cloned()
    }

    pub async fn get_all_presence(&self) -> Vec<Presence> {
        let map = self.presence.read().await;
        map.values().cloned().collect()
    }

    pub async fn get_online_users(&self) -> Vec<Presence> {
        let map = self.presence.read().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        map.values()
            .filter(|p| p.state != PresenceState::Offline && now - p.last_seen < self.timeout.as_secs())
            .cloned()
            .collect()
    }

    pub async fn remove_user(&self, user_id: Uuid) {
        let mut map = self.presence.write().await;
        map.remove(&user_id);
    }

    pub async fn cleanup_stale(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let timeout_secs = self.timeout.as_secs();

        let mut map = self.presence.write().await;
        map.retain(|_, p| now - p.last_seen < timeout_secs * 2);

        // Mark as away if past timeout but not double
        for presence in map.values_mut() {
            if now - presence.last_seen > timeout_secs && presence.state == PresenceState::Online {
                presence.state = PresenceState::Away;
            }
        }
    }
}

impl Default for PresenceManager {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}

/// Cursor broadcast for real-time cursor sharing
pub struct CursorBroadcast {
    cursors: RwLock<HashMap<Uuid, CursorPosition>>,
    max_update_rate: Duration,
    last_updates: RwLock<HashMap<Uuid, Instant>>,
}

impl CursorBroadcast {
    pub fn new(max_update_rate: Duration) -> Self {
        Self {
            cursors: RwLock::new(HashMap::new()),
            max_update_rate,
            last_updates: RwLock::new(HashMap::new()),
        }
    }

    pub async fn update_cursor(&self, user_id: Uuid, cursor: CursorPosition) -> bool {
        let now = Instant::now();
        let mut last_updates = self.last_updates.write().await;

        if let Some(last) = last_updates.get(&user_id) {
            if now.duration_since(*last) < self.max_update_rate {
                return false; // Rate limited
            }
        }

        last_updates.insert(user_id, now);
        drop(last_updates);

        let mut cursors = self.cursors.write().await;
        cursors.insert(user_id, cursor);
        true
    }

    pub async fn get_cursors(&self) -> HashMap<Uuid, CursorPosition> {
        let cursors = self.cursors.read().await;
        cursors.clone()
    }

    pub async fn remove_cursor(&self, user_id: Uuid) {
        let mut cursors = self.cursors.write().await;
        cursors.remove(&user_id);
    }
}

impl Default for CursorBroadcast {
    fn default() -> Self {
        Self::new(Duration::from_millis(50)) // 20fps max
    }
}
