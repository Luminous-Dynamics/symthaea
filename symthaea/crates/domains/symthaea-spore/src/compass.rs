// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Compass: JSON-serializable snapshot for the Android widget.
//!
//! Aggregates consciousness state into a compact struct suitable for
//! rendering in a homescreen widget or notification.

use serde::{Deserialize, Serialize};

/// Compact consciousness snapshot for platform consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompassSnapshot {
    pub consciousness_level: f32,
    pub dominant_harmony: String,
    /// [dopamine, norepinephrine, serotonin, oxytocin] — normalized 0-1
    pub neuromod_balance: [f32; 4],
    pub wake_state: u8,
    pub motion_state: u8,
    pub privacy_mode: bool,
    pub valence: f32,
    pub arousal: f32,
    pub dream_count: u32,
    pub wisdom_count: u32,
    pub cycle: u64,
    pub timestamp_epoch_ms: u64,
}

impl CompassSnapshot {
    /// Build a compass snapshot from engine state.
    ///
    /// `neuromods`: [DA, NE, 5-HT, OT]
    /// Valence derived from DA + OT, arousal from NE.
    pub fn build(
        consciousness_level: f32,
        dominant_harmony: &str,
        neuromods: [f32; 4],
        wake_state: u8,
        motion_state: u8,
        privacy_mode: bool,
        dream_count: u32,
        wisdom_count: u32,
        cycle: u64,
    ) -> Self {
        let [da, ne, _sht, ot] = neuromods;
        let valence = ((da - 0.5) + (ot - 0.5)).clamp(-1.0, 1.0);
        let arousal = ne.clamp(0.0, 1.0);

        Self {
            consciousness_level,
            dominant_harmony: dominant_harmony.to_string(),
            neuromod_balance: neuromods,
            wake_state,
            motion_state,
            privacy_mode,
            valence,
            arousal,
            dream_count,
            wisdom_count,
            cycle,
            timestamp_epoch_ms: 0, // Platform sets real timestamp
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compass_build() {
        let snap = CompassSnapshot::build(
            0.7,
            "Resonance",
            [0.6, 0.5, 0.4, 0.3],
            2,
            1,
            false,
            5,
            3,
            100,
        );
        assert_eq!(snap.consciousness_level, 0.7);
        assert_eq!(snap.dominant_harmony, "Resonance");
        assert_eq!(snap.wake_state, 2);
        assert_eq!(snap.motion_state, 1);
        assert!(!snap.privacy_mode);
        assert_eq!(snap.dream_count, 5);
        assert_eq!(snap.wisdom_count, 3);
        assert_eq!(snap.cycle, 100);
    }

    #[test]
    fn test_compass_valence_arousal() {
        let snap =
            CompassSnapshot::build(0.5, "Harmony", [0.8, 0.9, 0.5, 0.7], 2, 0, false, 0, 0, 1);
        // valence = (0.8-0.5) + (0.7-0.5) = 0.5
        assert!((snap.valence - 0.5).abs() < 0.01);
        // arousal = NE = 0.9
        assert!((snap.arousal - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_compass_json_roundtrip() {
        let snap = CompassSnapshot::build(0.5, "Peace", [0.5, 0.5, 0.5, 0.5], 0, 0, true, 0, 0, 0);
        let json = snap.to_json();
        assert!(json.contains("consciousness_level"));
        assert!(json.contains("Peace"));
        let parsed: CompassSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.consciousness_level, 0.5);
        assert!(parsed.privacy_mode);
    }

    #[test]
    fn test_compass_valence_clamp() {
        let snap = CompassSnapshot::build(0.5, "H", [1.0, 0.5, 0.5, 1.0], 0, 0, false, 0, 0, 0);
        // valence = (1.0-0.5) + (1.0-0.5) = 1.0 (clamped)
        assert!(snap.valence <= 1.0);
        assert!(snap.valence >= -1.0);
    }
}
