// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sensor bridge: translates platform sensor data into neuromodulator nudges.
//!
//! The bridge receives raw sensor values from the platform layer (Android/iOS)
//! and converts them into biologically-grounded neuromodulator modulations:
//! - Motion -> NE (arousal from physical activity)
//! - Light -> 5-HT (serotonin from bright light, circadian alignment)
//! - GPS novelty -> DA (dopamine from exploration)
//! - Proximity -> privacy mode (suppresses outbound sharing)

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_crypto::HdcContextKey;

// Neuromodulator nudge magnitudes (small, clamped)
const MOTION_NE_NUDGE: f32 = 0.03;
const LIGHT_5HT_NUDGE: f32 = 0.02;
const GPS_DA_NUDGE: f32 = 0.04;
const ROTATION_NE_NUDGE: f32 = 0.02;
const BRIGHT_LUX_THRESHOLD: f32 = 500.0;
const DIM_LUX_THRESHOLD: f32 = 50.0;
// Motion state thresholds (accelerometer magnitude in m/s^2)
const WALKING_THRESHOLD: f32 = 1.5;
const RUNNING_THRESHOLD: f32 = 5.0;
const VEHICLE_THRESHOLD: f32 = 10.0;
// Ambient sound thresholds (dB)
const QUIET_DB_THRESHOLD: f32 = 30.0;
const LOUD_DB_THRESHOLD: f32 = 70.0;
const AMBIENT_5HT_NUDGE: f32 = 0.02;
const AMBIENT_NE_NUDGE: f32 = 0.03;
// Social pressure nudges
const SOCIAL_NE_NUDGE: f32 = 0.02;
const SOCIAL_OT_DECAY: f32 = 0.01;
// Media state nudges
const MEDIA_MUSIC_5HT_NUDGE: f32 = 0.02;
const MEDIA_SPEECH_DA_NUDGE: f32 = 0.02;

/// Motion state derived from accelerometer magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionState {
    Stationary,
    Walking,
    Running,
    InVehicle,
}

impl MotionState {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Stationary => 0,
            Self::Walking => 1,
            Self::Running => 2,
            Self::InVehicle => 3,
        }
    }
}

/// Raw sensor snapshot set by the platform each frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSnapshot {
    pub accelerometer_magnitude: f32,
    pub light_lux: f32,
    pub proximity_near: bool,
    pub barometer_hpa: f32,
    pub gps_novelty: f32, // 0.0-1.0
    /// Gyroscope rotation rate magnitude (rad/s).
    pub rotation_rate: f32,
}

impl Default for SensorSnapshot {
    fn default() -> Self {
        Self {
            accelerometer_magnitude: 0.0,
            light_lux: 300.0,
            proximity_near: false,
            barometer_hpa: 1013.25,
            gps_novelty: 0.0,
            rotation_rate: 0.0,
        }
    }
}

/// Media playback state from the platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaState {
    None = 0,
    Music = 1,
    Speech = 2,
}

/// Neuromodulator nudges computed from sensor data.
#[derive(Debug, Clone, Default)]
pub struct SensorNudges {
    pub dopamine_delta: f32,
    pub norepinephrine_delta: f32,
    pub serotonin_delta: f32,
    pub oxytocin_delta: f32,
}

/// Sensor bridge: maintains current snapshot and derives neuromod nudges.
pub struct SensorBridge {
    snapshot: SensorSnapshot,
    motion_state: MotionState,
    privacy_mode: bool,
    motion_ema: f32,
    /// Consecutive frames spent stationary (for inactivity auto-detection).
    stationary_frames: u32,
    /// Ambient sound level (dB). Set via `set_ambient_db()`.
    ambient_db: f32,
    /// Notification count from the platform. Set via `set_social_pressure()`.
    notification_count: u32,
    /// Recent notification text snippets for HDC social context encoding.
    /// Max 5 recent, truncated to 100 chars each.
    notification_texts: Vec<String>,
    /// Derived social context salience (0.0-1.0) from notification content.
    social_salience: f32,
    /// Media playback state. Set via `set_media_state()`.
    media_state: MediaState,
    /// Step counter delta (steps since last tick).
    step_delta: u32,
}

impl SensorBridge {
    pub fn new() -> Self {
        Self {
            snapshot: SensorSnapshot::default(),
            motion_state: MotionState::Stationary,
            privacy_mode: false,
            motion_ema: 0.0,
            stationary_frames: 0,
            ambient_db: 40.0,
            notification_count: 0,
            notification_texts: Vec::new(),
            social_salience: 0.0,
            media_state: MediaState::None,
            step_delta: 0,
        }
    }

    /// Update sensor snapshot from platform.
    pub fn set_sensors(
        &mut self,
        accel: f32,
        light: f32,
        proximity_near: bool,
        barometer: f32,
        gps_novelty: f32,
    ) {
        self.snapshot = SensorSnapshot {
            accelerometer_magnitude: accel.max(0.0),
            light_lux: light.max(0.0),
            proximity_near,
            barometer_hpa: barometer.clamp(800.0, 1100.0),
            gps_novelty: gps_novelty.clamp(0.0, 1.0),
            rotation_rate: self.snapshot.rotation_rate, // preserve gyro (set separately)
        };
        self.update_derived();
    }

    /// Set gyroscope rotation rate from platform (rad/s magnitude).
    pub fn set_gyroscope(&mut self, rotation_rate: f32) {
        self.snapshot.rotation_rate = rotation_rate.max(0.0);
    }

    /// Set ambient sound level (dB). Only amplitude — no audio content stored.
    pub fn set_ambient_db(&mut self, db: f32) {
        self.ambient_db = db.clamp(0.0, 120.0);
    }

    /// Set notification count from platform (social pressure signal).
    pub fn set_social_pressure(&mut self, notification_count: u32) {
        self.notification_count = notification_count;
    }

    /// Add notification text for social context analysis.
    /// Text is truncated to 100 chars. Keeps max 5 recent notifications.
    /// Computes social salience from keyword content.
    pub fn add_notification_text(&mut self, text: &str) {
        let truncated: String = text.chars().take(100).collect();

        // Compute salience from content keywords
        let lower = truncated.to_lowercase();
        let urgent_keywords = [
            "urgent",
            "emergency",
            "critical",
            "asap",
            "important",
            "help",
        ];
        let social_keywords = ["message", "call", "reply", "friend", "family", "love"];
        let urgent_count = urgent_keywords
            .iter()
            .filter(|k| lower.contains(*k))
            .count();
        let social_count = social_keywords
            .iter()
            .filter(|k| lower.contains(*k))
            .count();

        // Update salience: urgent boosts NE pathway, social boosts OT pathway
        self.social_salience = ((urgent_count as f32 * 0.2 + social_count as f32 * 0.1)
            .clamp(0.0, 1.0)
            + self.social_salience)
            * 0.5; // EMA

        self.notification_texts.push(truncated);
        if self.notification_texts.len() > 5 {
            self.notification_texts.remove(0);
        }
    }

    /// Current social context salience (0.0-1.0) derived from notification content.
    pub fn social_salience(&self) -> f32 {
        self.social_salience
    }

    /// Set media playback state (0=None, 1=Music, 2=Speech).
    pub fn set_media_state(&mut self, state: u8) {
        self.media_state = match state {
            1 => MediaState::Music,
            2 => MediaState::Speech,
            _ => MediaState::None,
        };
    }

    /// Set step counter delta (steps since last tick).
    pub fn set_step_delta(&mut self, steps: u32) {
        self.step_delta = steps;
    }

    fn update_derived(&mut self) {
        // Motion EMA for smooth transitions (alpha=0.4 → ~8 frames to reach threshold)
        self.motion_ema = self.motion_ema * 0.6 + self.snapshot.accelerometer_magnitude * 0.4;

        self.motion_state = if self.motion_ema >= VEHICLE_THRESHOLD {
            MotionState::InVehicle
        } else if self.motion_ema >= RUNNING_THRESHOLD {
            MotionState::Running
        } else if self.motion_ema >= WALKING_THRESHOLD || self.step_delta > 0 {
            MotionState::Walking
        } else {
            MotionState::Stationary
        };

        if self.motion_state == MotionState::Stationary {
            self.stationary_frames = self.stationary_frames.saturating_add(1);
        } else {
            self.stationary_frames = 0;
        }

        // Privacy mode: face-down (proximity sensor near) suppresses all outbound
        self.privacy_mode = self.snapshot.proximity_near;
    }

    /// Compute neuromodulator nudges from current sensor state.
    pub fn compute_nudges(&self) -> SensorNudges {
        let mut nudges = SensorNudges::default();

        // Motion -> NE (arousal from physical activity)
        let motion_factor = match self.motion_state {
            MotionState::Stationary => 0.0,
            MotionState::Walking => 0.5,
            MotionState::Running => 1.0,
            MotionState::InVehicle => 0.3, // Less arousal in passive transport
        };
        nudges.norepinephrine_delta = MOTION_NE_NUDGE * motion_factor;

        // Step counter supplements motion detection — if steps detected, at least Walking
        if self.step_delta > 0 && self.motion_state == MotionState::Stationary {
            nudges.norepinephrine_delta += MOTION_NE_NUDGE * 0.5;
        }

        // Gyroscope rotation -> NE (spatial awareness/orientation changes)
        if self.snapshot.rotation_rate > 0.5 {
            nudges.norepinephrine_delta +=
                ROTATION_NE_NUDGE * (self.snapshot.rotation_rate / 5.0).min(1.0);
        }

        // Light -> 5-HT (bright light boosts serotonin, dim suppresses)
        if self.snapshot.light_lux > BRIGHT_LUX_THRESHOLD {
            nudges.serotonin_delta = LIGHT_5HT_NUDGE;
        } else if self.snapshot.light_lux < DIM_LUX_THRESHOLD {
            nudges.serotonin_delta = -LIGHT_5HT_NUDGE * 0.5;
        }

        // GPS novelty -> DA (dopamine from exploration)
        nudges.dopamine_delta = GPS_DA_NUDGE * self.snapshot.gps_novelty;

        // Ambient sound -> NE/5-HT (environmental arousal/calm)
        if self.ambient_db > LOUD_DB_THRESHOLD {
            nudges.norepinephrine_delta += AMBIENT_NE_NUDGE;
        } else if self.ambient_db < QUIET_DB_THRESHOLD {
            nudges.serotonin_delta += AMBIENT_5HT_NUDGE;
        }

        // Social pressure (notifications) -> NE/OT
        if self.notification_count > 5 {
            nudges.norepinephrine_delta += SOCIAL_NE_NUDGE;
        }
        if self.notification_count == 0 {
            // Extended isolation — oxytocin decay signal
            nudges.oxytocin_delta -= SOCIAL_OT_DECAY;
        }
        // Notification content salience → graded NE/OT boost
        if self.social_salience > 0.1 {
            nudges.norepinephrine_delta += self.social_salience * 0.02;
            nudges.oxytocin_delta += self.social_salience * 0.01;
        }

        // Media state -> 5-HT/DA (emotional context)
        match self.media_state {
            MediaState::Music => nudges.serotonin_delta += MEDIA_MUSIC_5HT_NUDGE,
            MediaState::Speech => nudges.dopamine_delta += MEDIA_SPEECH_DA_NUDGE,
            MediaState::None => {}
        }

        nudges
    }

    pub fn motion_state(&self) -> MotionState {
        self.motion_state
    }

    pub fn privacy_mode(&self) -> bool {
        self.privacy_mode
    }

    pub fn snapshot(&self) -> &SensorSnapshot {
        &self.snapshot
    }

    /// Estimated inactivity in seconds (assumes ~20 sensor updates/sec).
    pub fn estimated_inactivity_secs(&self) -> u32 {
        self.stationary_frames / 20
    }

    // ── HDC Context Key Derivation ───────────────────────────────────────

    /// Encode the current sensor snapshot as BinaryHV vectors and derive an
    /// deterministic HDC context fingerprint for the device's environment.
    ///
    /// The key changes when:
    /// - The device moves to a different location (GPS novelty)
    /// - Motion state changes (accelerometer)
    /// - Lighting conditions change (ambient light)
    /// - Altitude changes (barometer)
    ///
    /// Each sensor reading is encoded as a deterministic BinaryHV using
    /// quantized seed values, then combined via `HdcContextKey::derive()`
    /// (bind + permute chain — order matters, non-commutative).
    ///
    /// **Security warning:** this deterministic value is a context fingerprint,
    /// not a secret key. The quantized sensor domain can be enumerated offline.
    /// It may be used as context input to a standard KDF only when combined with
    /// independent secret entropy.
    #[deprecated(note = "deterministic context fingerprint, not secret key material")]
    pub fn derive_context_key(&self) -> [u8; 32] {
        let sensors = self.encode_sensors_as_hvs();
        HdcContextKey::derive_symmetric(&sensors)
    }

    /// Get the raw HDC context vector (16,384-bit BinaryHV) before BLAKE3 extraction.
    ///
    /// Useful for non-security HDC operations such as similarity comparison
    /// between two public context fingerprints.
    pub fn derive_context_hv(&self) -> BinaryHV {
        let sensors = self.encode_sensors_as_hvs();
        HdcContextKey::derive(&sensors)
    }

    /// Encode each sensor as a deterministic BinaryHV.
    ///
    /// Quantization strategy: each sensor value is mapped to a u64 seed
    /// that changes only on meaningful state transitions (not on noise).
    fn encode_sensors_as_hvs(&self) -> Vec<BinaryHV> {
        // Sensor 0: Motion state (4 discrete states → 4 distinct HVs)
        let motion_seed = 0x4D4F_5449_0000_0000u64 | self.motion_state.as_u8() as u64;

        // Sensor 1: Light level (quantized to 10-lux bands)
        let light_band = (self.snapshot.light_lux / 10.0) as u64;
        let light_seed = 0x4C49_4748_0000_0000u64 | light_band;

        // Sensor 2: Barometer (quantized to 1-hPa bands)
        let baro_band = self.snapshot.barometer_hpa as u64;
        let baro_seed = 0x4241_524F_0000_0000u64 | baro_band;

        // Sensor 3: GPS novelty (quantized to 0.1 bands)
        let gps_band = (self.snapshot.gps_novelty * 10.0) as u64;
        let gps_seed = 0x4750_534E_0000_0000u64 | gps_band;

        vec![
            BinaryHV::random(motion_seed),
            BinaryHV::random(light_seed),
            BinaryHV::random(baro_seed),
            BinaryHV::random(gps_seed),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let bridge = SensorBridge::new();
        assert_eq!(bridge.motion_state(), MotionState::Stationary);
        assert!(!bridge.privacy_mode());
    }

    #[test]
    fn test_motion_state_detection() {
        let mut bridge = SensorBridge::new();
        // Feed walking-level acceleration several times for EMA to settle
        for _ in 0..20 {
            bridge.set_sensors(2.0, 300.0, false, 1013.0, 0.0);
        }
        assert_eq!(bridge.motion_state(), MotionState::Walking);
    }

    #[test]
    fn test_running_detection() {
        let mut bridge = SensorBridge::new();
        for _ in 0..20 {
            bridge.set_sensors(7.0, 300.0, false, 1013.0, 0.0);
        }
        assert_eq!(bridge.motion_state(), MotionState::Running);
    }

    #[test]
    fn test_privacy_mode() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 0.0, true, 1013.0, 0.0);
        assert!(bridge.privacy_mode());
        bridge.set_sensors(0.0, 0.0, false, 1013.0, 0.0);
        assert!(!bridge.privacy_mode());
    }

    #[test]
    fn test_nudges_motion_ne() {
        let mut bridge = SensorBridge::new();
        for _ in 0..20 {
            bridge.set_sensors(3.0, 300.0, false, 1013.0, 0.0);
        }
        let nudges = bridge.compute_nudges();
        assert!(nudges.norepinephrine_delta > 0.0);
    }

    #[test]
    fn test_nudges_bright_light_serotonin() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 1000.0, false, 1013.0, 0.0);
        let nudges = bridge.compute_nudges();
        assert!(nudges.serotonin_delta > 0.0);
    }

    #[test]
    fn test_nudges_dim_light_serotonin() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 10.0, false, 1013.0, 0.0);
        let nudges = bridge.compute_nudges();
        assert!(nudges.serotonin_delta < 0.0);
    }

    #[test]
    fn test_nudges_gps_novelty_da() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 300.0, false, 1013.0, 0.8);
        let nudges = bridge.compute_nudges();
        assert!(nudges.dopamine_delta > 0.0);
    }

    #[test]
    fn test_nudges_stationary_no_ne() {
        let bridge = SensorBridge::new();
        let nudges = bridge.compute_nudges();
        assert_eq!(nudges.norepinephrine_delta, 0.0);
    }

    #[test]
    fn test_context_key_deterministic() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(2.0, 500.0, false, 1013.0, 0.5);
        let key1 = bridge.derive_context_key();
        let key2 = bridge.derive_context_key();
        assert_eq!(key1, key2, "Same sensors should produce same context key");
    }

    /// CI-005: once the non-secret context is narrowed, a quantized sensor
    /// component can be recovered by enumerating its small public domain.
    #[test]
    fn legacy_attack_enumerates_quantized_sensor_context() {
        let mut target = SensorBridge::new();
        target.set_sensors(0.0, 370.0, false, 1013.0, 0.0);
        let target_fingerprint = target.derive_context_key();

        let recovered_light_band = (0..=100u32).find(|band| {
            let mut candidate = SensorBridge::new();
            candidate.set_sensors(0.0, *band as f32 * 10.0, false, 1013.0, 0.0);
            candidate.derive_context_key() == target_fingerprint
        });

        assert_eq!(recovered_light_band, Some(37));
    }

    #[test]
    fn test_context_key_changes_with_motion() {
        let mut bridge = SensorBridge::new();
        // Stationary
        for _ in 0..20 {
            bridge.set_sensors(0.0, 300.0, false, 1013.0, 0.0);
        }
        let key_stationary = bridge.derive_context_key();

        // Walking
        for _ in 0..20 {
            bridge.set_sensors(3.0, 300.0, false, 1013.0, 0.0);
        }
        let key_walking = bridge.derive_context_key();

        assert_ne!(
            key_stationary, key_walking,
            "Different motion states should produce different context keys"
        );
    }

    #[test]
    fn test_context_key_changes_with_light() {
        let mut bridge_bright = SensorBridge::new();
        bridge_bright.set_sensors(0.0, 1000.0, false, 1013.0, 0.0);
        let key_bright = bridge_bright.derive_context_key();

        let mut bridge_dim = SensorBridge::new();
        bridge_dim.set_sensors(0.0, 10.0, false, 1013.0, 0.0);
        let key_dim = bridge_dim.derive_context_key();

        assert_ne!(
            key_bright, key_dim,
            "Different light levels should produce different keys"
        );
    }

    #[test]
    fn test_context_hv_is_16384_bits() {
        let bridge = SensorBridge::new();
        let hv = bridge.derive_context_hv();
        assert_eq!(
            hv.0.len(),
            2048,
            "Context HV should be 2048 bytes = 16384 bits"
        );
    }

    #[test]
    fn test_barometer_clamp() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 300.0, false, 2000.0, 0.0);
        assert_eq!(bridge.snapshot().barometer_hpa, 1100.0);
    }

    #[test]
    fn test_stationary_inactivity_counter() {
        let mut bridge = SensorBridge::new();
        // 100 stationary frames = 5 seconds at 20fps
        for _ in 0..100 {
            bridge.set_sensors(0.0, 300.0, false, 1013.0, 0.0);
        }
        assert_eq!(bridge.estimated_inactivity_secs(), 5);
        // Motion resets counter
        bridge.set_sensors(5.0, 300.0, false, 1013.0, 0.0);
        // EMA hasn't crossed walking threshold yet but counter should NOT reset
        // until motion_state actually changes from Stationary
        // After many motion frames, counter resets
        for _ in 0..20 {
            bridge.set_sensors(5.0, 300.0, false, 1013.0, 0.0);
        }
        assert_eq!(bridge.estimated_inactivity_secs(), 0);
    }
}
