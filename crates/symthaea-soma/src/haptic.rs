// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Haptic awareness: consciousness-gated tactile events.
//!
//! Events are queued when consciousness state changes significantly,
//! then drained by the platform layer for haptic feedback rendering.
//! Gating prevents spam — only meaningful shifts produce haptics.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const MAX_PENDING_EVENTS: usize = 32;
const DEFAULT_CONSCIOUSNESS_DELTA_THRESHOLD: f32 = 0.10;
const DEFAULT_SURPRISE_THRESHOLD: f32 = 0.50;
/// Minimum cycles between haptic events (cooldown to prevent buzzing).
const HAPTIC_COOLDOWN_CYCLES: u32 = 50; // ~5 seconds at 10Hz

/// Haptic event types for platform rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HapticEvent {
    /// Consciousness level shifted by more than threshold.
    ConsciousnessShift { delta: f32 },
    /// Dream consolidation produced wisdom.
    DreamWisdom,
    /// New BLE peer discovered.
    PeerDiscovered,
    /// High prediction error (surprise).
    HighSurprise { surprise: f32 },
    /// Harmony milestone reached.
    HarmonyMilestone { harmony: String },
}

/// Haptic awareness manager.
pub struct HapticManager {
    enabled: bool,
    queue: VecDeque<HapticEvent>,
    last_consciousness: f32,
    consciousness_delta_threshold: f32,
    surprise_threshold: f32,
    /// Cycles since last haptic event (cooldown counter).
    cycles_since_last: u32,
}

impl HapticManager {
    pub fn new() -> Self {
        Self {
            enabled: true,
            queue: VecDeque::with_capacity(8),
            last_consciousness: 0.0,
            consciousness_delta_threshold: DEFAULT_CONSCIOUSNESS_DELTA_THRESHOLD,
            surprise_threshold: DEFAULT_SURPRISE_THRESHOLD,
            cycles_since_last: HAPTIC_COOLDOWN_CYCLES, // ready to fire immediately
        }
    }

    /// Call once per engine cycle to advance the cooldown timer.
    pub fn tick(&mut self) {
        self.cycles_since_last = self.cycles_since_last.saturating_add(1);
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.queue.clear();
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check consciousness level and enqueue shift event if delta exceeds threshold.
    pub fn check_consciousness(&mut self, level: f32) {
        if !self.enabled {
            return;
        }
        let delta = level - self.last_consciousness;
        if delta.abs() >= self.consciousness_delta_threshold {
            self.push(HapticEvent::ConsciousnessShift { delta });
        }
        self.last_consciousness = level;
    }

    /// Check surprise level and enqueue if above threshold.
    pub fn check_surprise(&mut self, prediction_error: f32) {
        if !self.enabled {
            return;
        }
        if prediction_error > self.surprise_threshold {
            self.push(HapticEvent::HighSurprise {
                surprise: prediction_error,
            });
        }
    }

    /// Signal dream wisdom was produced.
    pub fn notify_dream_wisdom(&mut self) {
        if !self.enabled {
            return;
        }
        self.push(HapticEvent::DreamWisdom);
    }

    /// Signal a new BLE peer was discovered.
    pub fn notify_peer_discovered(&mut self) {
        if !self.enabled {
            return;
        }
        self.push(HapticEvent::PeerDiscovered);
    }

    /// Signal a harmony milestone.
    pub fn notify_harmony_milestone(&mut self, harmony: &str) {
        if !self.enabled {
            return;
        }
        self.push(HapticEvent::HarmonyMilestone {
            harmony: harmony.to_string(),
        });
    }

    /// Number of pending events.
    pub fn pending_count(&self) -> u32 {
        self.queue.len() as u32
    }

    /// Drain all pending events as JSON.
    pub fn drain_json(&mut self) -> String {
        let events: Vec<HapticEvent> = self.queue.drain(..).collect();
        serde_json::to_string(&events).unwrap_or_else(|_| "[]".to_string())
    }

    /// Drain all pending events.
    pub fn drain(&mut self) -> Vec<HapticEvent> {
        self.queue.drain(..).collect()
    }

    fn push(&mut self, event: HapticEvent) {
        if self.cycles_since_last < HAPTIC_COOLDOWN_CYCLES {
            return; // cooldown active — skip this event
        }
        if self.queue.len() >= MAX_PENDING_EVENTS {
            self.queue.pop_front();
        }
        self.queue.push_back(event);
        self.cycles_since_last = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_shift_triggers() {
        let mut h = HapticManager::new();
        h.last_consciousness = 0.5;
        h.check_consciousness(0.7);
        assert_eq!(h.pending_count(), 1);
    }

    #[test]
    fn test_small_shift_no_trigger() {
        let mut h = HapticManager::new();
        h.last_consciousness = 0.5;
        h.check_consciousness(0.55); // delta 0.05 < threshold 0.10
        assert_eq!(h.pending_count(), 0);
    }

    #[test]
    fn test_surprise_triggers() {
        let mut h = HapticManager::new();
        h.check_surprise(0.8);
        assert_eq!(h.pending_count(), 1);
    }

    #[test]
    fn test_surprise_below_threshold() {
        let mut h = HapticManager::new();
        h.check_surprise(0.3); // below threshold of 0.50
        assert_eq!(h.pending_count(), 0);
    }

    #[test]
    fn test_disabled_no_events() {
        let mut h = HapticManager::new();
        h.set_enabled(false);
        h.check_consciousness(1.0);
        h.check_surprise(1.0);
        h.notify_dream_wisdom();
        h.notify_peer_discovered();
        assert_eq!(h.pending_count(), 0);
    }

    #[test]
    fn test_drain_clears_queue() {
        let mut h = HapticManager::new();
        h.notify_dream_wisdom();
        // Advance past cooldown so second event can fire
        for _ in 0..HAPTIC_COOLDOWN_CYCLES {
            h.tick();
        }
        h.notify_peer_discovered();
        assert_eq!(h.pending_count(), 2);
        let events = h.drain();
        assert_eq!(events.len(), 2);
        assert_eq!(h.pending_count(), 0);
    }

    #[test]
    fn test_drain_json() {
        let mut h = HapticManager::new();
        h.notify_dream_wisdom();
        let json = h.drain_json();
        assert!(json.contains("DreamWisdom"));
        assert_eq!(h.pending_count(), 0);
    }

    #[test]
    fn test_max_pending_cap() {
        let mut h = HapticManager::new();
        for _ in 0..40 {
            h.notify_dream_wisdom();
            // Advance past cooldown so next event can fire
            for _ in 0..HAPTIC_COOLDOWN_CYCLES {
                h.tick();
            }
        }
        assert_eq!(h.pending_count(), MAX_PENDING_EVENTS as u32);
    }

    #[test]
    fn test_harmony_milestone() {
        let mut h = HapticManager::new();
        h.notify_harmony_milestone("SacredStillness");
        assert_eq!(h.pending_count(), 1);
        let json = h.drain_json();
        assert!(json.contains("SacredStillness"));
    }
}
