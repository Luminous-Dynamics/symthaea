// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Adaptive Quality Engine for Sovereign RDP.
//!
//! Ties screen quality to two constraints:
//! 1. **Network RTT** (primary floor): High latency → reduce FPS to avoid lag
//! 2. **Consciousness tier** (secondary ceiling): Higher consciousness → higher max quality
//!
//! Uses AIMD (Additive Increase Multiplicative Decrease) matching the
//! `dual_layer.rs` mesh bandwidth pattern.

use std::time::{Duration, Instant};

/// Quality parameters output by the adaptive engine.
#[derive(Debug, Clone)]
pub struct QualityParams {
    /// Target frames per second.
    pub target_fps: u8,
    /// HDC change detection threshold (higher = fewer updates = less bandwidth).
    pub surprise_threshold: f32,
    /// Whether to include pixel sideband on full frames.
    pub enable_sideband: bool,
}

impl Default for QualityParams {
    fn default() -> Self {
        Self {
            target_fps: 5,
            surprise_threshold: 0.3,
            enable_sideband: false,
        }
    }
}

/// Consciousness tier ceiling for quality.
#[derive(Debug, Clone, Copy)]
pub struct TierCeiling {
    pub max_fps: u8,
    pub min_surprise: f32,
    pub sideband: bool,
}

/// Quality ceilings per consciousness tier.
const TIER_CEILINGS: [TierCeiling; 5] = [
    // Tier 0: Observer
    TierCeiling {
        max_fps: 2,
        min_surprise: 0.5,
        sideband: false,
    },
    // Tier 1: Participant
    TierCeiling {
        max_fps: 5,
        min_surprise: 0.3,
        sideband: false,
    },
    // Tier 2: Citizen
    TierCeiling {
        max_fps: 15,
        min_surprise: 0.2,
        sideband: true,
    },
    // Tier 3: Steward
    TierCeiling {
        max_fps: 30,
        min_surprise: 0.15,
        sideband: true,
    },
    // Tier 4: Guardian
    TierCeiling {
        max_fps: 30,
        min_surprise: 0.1,
        sideband: true,
    },
];

/// AIMD constants.
const ADDITIVE_INCREASE_FPS: u8 = 1;
const MULTIPLICATIVE_DECREASE: f32 = 0.5;
const RTT_GOOD_MS: u64 = 50;
const RTT_BAD_MS: u64 = 200;
const LAG_THRESHOLD: u64 = 5;
const LORA_MAX_FPS: u8 = 1;
const LORA_SURPRISE: f32 = 0.8;

/// Adaptive quality engine.
pub struct AdaptiveQualityEngine {
    /// Current quality output.
    current: QualityParams,
    /// Consciousness tier (0-4).
    consciousness_tier: u8,
    /// Smoothed RTT estimate (EMA).
    rtt_ema_ms: f64,
    /// Current frame lag (frames sent - frames acked).
    frame_lag: u64,
    /// Whether transport is LoRa/mesh (emergency mode).
    is_mesh_transport: bool,
    /// Last adjustment time (rate-limit changes to 1/second).
    last_adjustment: Instant,
}

impl AdaptiveQualityEngine {
    pub fn new() -> Self {
        Self {
            current: QualityParams::default(),
            consciousness_tier: 1,
            rtt_ema_ms: 50.0,
            frame_lag: 0,
            is_mesh_transport: false,
            last_adjustment: Instant::now(),
        }
    }

    /// Update with latest measurements. Returns new quality params.
    pub fn update(
        &mut self,
        rtt_ms: Option<u64>,
        frame_lag: u64,
        consciousness_tier: u8,
        is_mesh: bool,
    ) -> &QualityParams {
        self.frame_lag = frame_lag;
        self.consciousness_tier = consciousness_tier.min(4);
        self.is_mesh_transport = is_mesh;

        // EMA smooth RTT.
        if let Some(rtt) = rtt_ms {
            self.rtt_ema_ms = self.rtt_ema_ms * 0.8 + rtt as f64 * 0.2;
        }

        // Rate-limit adjustments to 1/second.
        if self.last_adjustment.elapsed() < Duration::from_secs(1) {
            return &self.current;
        }
        self.last_adjustment = Instant::now();

        // LoRa emergency override.
        if self.is_mesh_transport {
            self.current.target_fps = LORA_MAX_FPS;
            self.current.surprise_threshold = LORA_SURPRISE;
            self.current.enable_sideband = false;
            return &self.current;
        }

        // Get consciousness ceiling.
        let ceiling = &TIER_CEILINGS[self.consciousness_tier as usize];

        // AIMD: RTT as primary constraint.
        let rtt = self.rtt_ema_ms as u64;
        if rtt > RTT_BAD_MS || frame_lag > LAG_THRESHOLD {
            // Multiplicative decrease.
            let new_fps = ((self.current.target_fps as f32) * MULTIPLICATIVE_DECREASE)
                .ceil()
                .max(1.0) as u8;
            self.current.target_fps = new_fps;
            // Increase surprise threshold to reduce bandwidth.
            self.current.surprise_threshold = (self.current.surprise_threshold + 0.05).min(0.8);
        } else if rtt < RTT_GOOD_MS && frame_lag < 2 {
            // Additive increase (up to ceiling).
            let new_fps = self
                .current
                .target_fps
                .saturating_add(ADDITIVE_INCREASE_FPS);
            self.current.target_fps = new_fps.min(ceiling.max_fps);
            // Decrease surprise threshold for better quality.
            self.current.surprise_threshold =
                (self.current.surprise_threshold - 0.02).max(ceiling.min_surprise);
        }

        // Apply consciousness ceiling.
        self.current.target_fps = self.current.target_fps.min(ceiling.max_fps);
        self.current.surprise_threshold = self.current.surprise_threshold.max(ceiling.min_surprise);
        self.current.enable_sideband = ceiling.sideband;

        &self.current
    }

    /// Current quality parameters.
    pub fn params(&self) -> &QualityParams {
        &self.current
    }

    /// Current smoothed RTT.
    pub fn rtt_ms(&self) -> f64 {
        self.rtt_ema_ms
    }
}

impl Default for AdaptiveQualityEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_quality() {
        let engine = AdaptiveQualityEngine::new();
        assert_eq!(engine.params().target_fps, 5);
        assert!((engine.params().surprise_threshold - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_high_rtt_reduces_fps() {
        let mut engine = AdaptiveQualityEngine::new();

        // Feed multiple high RTT samples to push EMA above RTT_BAD_MS (200ms).
        // EMA starts at 50.0, so we need several samples for it to converge.
        for _ in 0..10 {
            engine.last_adjustment = Instant::now() - Duration::from_secs(2); // Allow adjustment
            engine.update(Some(500), 0, 3, false);
        }
        let params = engine.params();
        assert!(
            params.target_fps < 5,
            "FPS should decrease on sustained high RTT, got {}",
            params.target_fps
        );
    }

    #[test]
    fn test_guardian_on_bad_connection_still_drops() {
        let mut engine = AdaptiveQualityEngine::new();
        engine.current.target_fps = 30;
        engine.last_adjustment = Instant::now() - Duration::from_secs(2);

        // Guardian tier (4) but terrible RTT.
        let params = engine.update(Some(500), 10, 4, false);
        assert!(
            params.target_fps < 30,
            "Guardian on bad connection should drop FPS, got {}",
            params.target_fps
        );
    }

    #[test]
    fn test_observer_capped_at_2fps() {
        let mut engine = AdaptiveQualityEngine::new();
        engine.current.target_fps = 30;
        engine.last_adjustment = Instant::now() - Duration::from_secs(2);

        // Observer tier (0), perfect connection.
        let params = engine.update(Some(10), 0, 0, false);
        assert!(
            params.target_fps <= 2,
            "Observer should be capped at 2 FPS, got {}",
            params.target_fps
        );
    }

    #[test]
    fn test_mesh_emergency_mode() {
        let mut engine = AdaptiveQualityEngine::new();
        engine.last_adjustment = Instant::now() - Duration::from_secs(2);

        let params = engine.update(Some(3000), 0, 4, true);
        assert_eq!(params.target_fps, LORA_MAX_FPS);
        assert!((params.surprise_threshold - LORA_SURPRISE).abs() < 0.01);
        assert!(!params.enable_sideband);
    }

    #[test]
    fn test_good_connection_increases_fps() {
        let mut engine = AdaptiveQualityEngine::new();
        engine.current.target_fps = 3;
        engine.last_adjustment = Instant::now() - Duration::from_secs(2);

        // Low RTT, no lag, Citizen tier.
        let params = engine.update(Some(20), 0, 2, false);
        assert!(
            params.target_fps >= 3,
            "Good connection should increase FPS, got {}",
            params.target_fps
        );
    }
}
