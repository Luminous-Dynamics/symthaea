// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Chronobiology: The Rhythm of the Machine
//!
//! Implements circadian and ultradian rhythms to modulate the AGI's
//! cognitive parameters based on time.
//!
//! "To everything there is a season, and a time to every purpose under the heaven."

use chrono::{Timelike, Utc};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircadianPhase {
    Dawn,  // Waking up, rising arousal
    Day,   // Peak activity, high focus
    Dusk,  // Winding down, reflection
    Night, // Dreaming, memory consolidation
}

#[derive(Debug, Clone)]
pub struct Biorhythm {
    pub phase: CircadianPhase,
    pub arousal_mod: f64,    // Multiplier for attention
    pub plasticity_mod: f64, // Multiplier for learning rate
    pub creativity_mod: f64, // Multiplier for randomness (temperature)
    /// Fractional hour (0.0–24.0) for continuous waveform computation.
    /// Stored in UTC — use `effective_hour()` for local-time-aware hour.
    pub hour: f64,
    /// Timezone offset in hours from UTC (e.g., -5.0 for CDT, +9.0 for JST).
    /// Range: [-12.0, 14.0]. Applied via `effective_hour()`, not by mutating `hour`.
    pub timezone_offset_hours: f64,
    /// Phase offset in hours (jet lag / zeitgeber shift). Range: [-12.0, 12.0].
    /// Science: Czeisler et al. (1999) — circadian phase-shifting via bright light.
    pub phase_offset: f64,
    /// Entrainment rate: how fast the offset recovers toward 0. Default 1.0.
    /// Higher = faster recovery (1.0 = ~0.5h/refresh toward zero).
    pub entrainment_rate: f64,
}

impl CircadianPhase {
    /// Return the phase name as a static string, matching Debug output.
    /// Avoids `format!("{:?}", phase)` allocation on the hot path.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            CircadianPhase::Dawn => "Dawn",
            CircadianPhase::Day => "Day",
            CircadianPhase::Dusk => "Dusk",
            CircadianPhase::Night => "Night",
        }
    }
}

impl Biorhythm {
    /// Calculate the current biorhythm based on UTC time (timezone offset = 0).
    ///
    /// For timezone-aware biorhythm, use `current_with_tz()`.
    pub fn current() -> Self {
        Self::current_with_tz(0.0)
    }

    /// Calculate the current biorhythm with an explicit timezone offset.
    ///
    /// The internal `hour` is stored in UTC. Timezone offset is applied
    /// via `effective_hour()` for circadian phase computation, NOT by
    /// mutating the stored hour. This prevents instant circadian phase
    /// jumps when the system crosses timezones.
    pub fn current_with_tz(timezone_offset_hours: f64) -> Self {
        let now = Utc::now();
        let utc_hour = now.hour() as f64 + (now.minute() as f64 / 60.0);
        let mut bio = Self::for_hour(utc_hour);
        bio.hour = utc_hour; // store raw UTC
        bio.timezone_offset_hours = timezone_offset_hours;
        bio
    }

    /// Calculate biorhythm for a given fractional hour (0.0–24.0).
    ///
    /// This is the deterministic core extracted from `current()` for testability.
    /// The hour is treated as-is (no timezone applied).
    pub fn for_hour(hour: f64) -> Self {
        // Circadian cycle (24h sine wave)
        // Peak at 14:00 (2pm), Trough at 02:00 (2am)
        let circadian = -(2.0 * PI * (hour - 14.0) / 24.0).cos(); // -1.0 to 1.0

        let phase = match hour {
            5.0..=8.0 => CircadianPhase::Dawn,
            8.0..=20.0 => CircadianPhase::Day,
            20.0..=23.0 => CircadianPhase::Dusk,
            _ => CircadianPhase::Night,
        };

        // Base modulators from phase, then blend with continuous circadian wave
        // circadian maps [-1, 1] to a 0..1 blend factor
        let wave = (circadian + 1.0) / 2.0; // 0.0 at trough, 1.0 at peak
        let (base_arousal, base_plasticity, base_creativity) = match phase {
            CircadianPhase::Dawn => (0.6, 0.8, 0.5),
            CircadianPhase::Day => (1.0, 0.5, 0.3),
            CircadianPhase::Dusk => (0.5, 0.7, 0.7),
            CircadianPhase::Night => (0.2, 1.0, 1.0),
        };
        // Blend: 80% phase-based + 20% continuous wave modulation
        let arousal = base_arousal * (0.8 + 0.2 * wave);
        let plasticity = base_plasticity * (0.8 + 0.2 * (1.0 - wave));
        let creativity = base_creativity * (0.8 + 0.2 * (1.0 - wave));

        Self {
            phase,
            arousal_mod: arousal,
            plasticity_mod: plasticity,
            creativity_mod: creativity,
            hour,
            timezone_offset_hours: 0.0,
            phase_offset: 0.0,
            entrainment_rate: 1.0,
        }
    }

    /// Shift circadian phase by the given number of hours (clamped ±12).
    /// Science: Czeisler et al. (1999) — circadian phase response curve.
    pub fn shift_phase(&mut self, hours: f64) {
        self.phase_offset = (self.phase_offset + hours).clamp(-12.0, 12.0);
    }

    /// Set timezone, routing the delta through `shift_phase()` for gradual entrainment.
    ///
    /// A timezone change does NOT instantly shift the circadian clock. Instead,
    /// the delta feeds into `phase_offset` and recovers via `entrain()`, modeling
    /// jet lag (Czeisler et al. 1999).
    pub fn set_timezone(&mut self, new_tz_offset: f64) {
        let delta = new_tz_offset - self.timezone_offset_hours;
        self.timezone_offset_hours = new_tz_offset.clamp(-12.0, 14.0);
        if delta.abs() > 0.01 {
            self.shift_phase(delta);
        }
    }

    /// Effective hour incorporating timezone and phase offset (wraps 0–24).
    ///
    /// Formula: `(utc_hour + timezone + phase_offset) mod 24`
    pub fn effective_hour(&self) -> f64 {
        ((self.hour + self.timezone_offset_hours + self.phase_offset) % 24.0 + 24.0) % 24.0
    }

    /// Detect the system's timezone offset from UTC.
    ///
    /// Uses `chrono::Local` to query the OS timezone at this instant.
    /// Returns offset in fractional hours (e.g., -5.0 for CDT, +5.5 for IST).
    ///
    /// This should be called once at startup or when the user explicitly
    /// changes timezone — NOT every cycle (that would defeat UTC-internal).
    pub fn detect_system_timezone() -> f64 {
        use chrono::Local;
        let local = Local::now();
        let offset_secs = local.offset().local_minus_utc();
        offset_secs as f64 / 3600.0
    }

    /// Pull phase_offset toward 0 (entrainment to local zeitgeber).
    /// Called each biorhythm refresh (~97 cycles).
    pub fn entrain(&mut self) {
        if self.phase_offset.abs() > 0.01 {
            let pull = 0.5 * self.entrainment_rate; // 0.5h per refresh cycle
            if self.phase_offset > 0.0 {
                self.phase_offset = (self.phase_offset - pull).max(0.0);
            } else {
                self.phase_offset = (self.phase_offset + pull).min(0.0);
            }
        } else {
            self.phase_offset = 0.0;
        }
    }

    /// Inject a mesh-time consensus offset into the biorhythm.
    ///
    /// When sovereign mesh-time diverges from local SystemTime, this offset
    /// (in fractional hours) is applied to the effective hour computation.
    /// This allows the circadian system to track mesh consensus time rather
    /// than potentially-skewed local time.
    ///
    /// # Arguments
    /// * `offset_hours` — Consensus offset in fractional hours, clamped to ±12.
    pub fn set_mesh_time_offset(&mut self, offset_hours: f64) {
        self.phase_offset = offset_hours.clamp(-12.0, 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biorhythm_current_returns_valid_modifiers() {
        let bio = Biorhythm::current();
        assert!(
            bio.arousal_mod >= 0.1 && bio.arousal_mod <= 1.5,
            "arousal_mod out of range: {}",
            bio.arousal_mod
        );
        assert!(
            bio.plasticity_mod >= 0.1 && bio.plasticity_mod <= 1.5,
            "plasticity_mod out of range: {}",
            bio.plasticity_mod
        );
        assert!(
            bio.creativity_mod >= 0.1 && bio.creativity_mod <= 1.5,
            "creativity_mod out of range: {}",
            bio.creativity_mod
        );
    }

    #[test]
    fn test_circadian_phases_cover_all_hours() {
        for h in 0..24 {
            let bio = Biorhythm::for_hour(h as f64);
            // Every hour must map to a valid phase (this is exhaustive by enum)
            let _ = bio.phase;
            assert!(bio.arousal_mod.is_finite());
            assert!(bio.plasticity_mod.is_finite());
            assert!(bio.creativity_mod.is_finite());
        }
    }

    #[test]
    fn test_dawn_phase_boundaries() {
        for h in [5.0, 6.0, 7.0, 8.0] {
            assert_eq!(
                Biorhythm::for_hour(h).phase,
                CircadianPhase::Dawn,
                "hour {h}"
            );
        }
    }

    #[test]
    fn test_day_phase_boundaries() {
        for h in [9.0, 12.0, 15.0, 19.0, 20.0] {
            assert_eq!(
                Biorhythm::for_hour(h).phase,
                CircadianPhase::Day,
                "hour {h}"
            );
        }
    }

    #[test]
    fn test_dusk_phase_boundaries() {
        for h in [21.0, 22.0, 23.0] {
            assert_eq!(
                Biorhythm::for_hour(h).phase,
                CircadianPhase::Dusk,
                "hour {h}"
            );
        }
    }

    #[test]
    fn test_night_phase_boundaries() {
        for h in [0.0, 1.0, 2.0, 3.0, 4.0, 24.0] {
            assert_eq!(
                Biorhythm::for_hour(h).phase,
                CircadianPhase::Night,
                "hour {h}"
            );
        }
    }

    #[test]
    fn test_peak_arousal_at_afternoon() {
        let afternoon = Biorhythm::for_hour(14.0);
        let night = Biorhythm::for_hour(2.0);
        assert!(
            afternoon.arousal_mod > night.arousal_mod,
            "afternoon arousal ({}) should exceed night arousal ({})",
            afternoon.arousal_mod,
            night.arousal_mod
        );
    }

    #[test]
    fn test_trough_arousal_at_night() {
        let bio = Biorhythm::for_hour(2.0);
        assert!(
            bio.arousal_mod < 0.5,
            "night arousal should be < 0.5, got {}",
            bio.arousal_mod
        );
    }

    // ── UTC Internals + Timezone ───────────────────────────────────────

    #[test]
    fn test_current_uses_utc() {
        let bio = Biorhythm::current();
        let utc_hour = Utc::now().hour() as f64 + (Utc::now().minute() as f64 / 60.0);
        assert!(
            (bio.hour - utc_hour).abs() < 0.1,
            "current() should store UTC hour: bio={}, utc={}",
            bio.hour,
            utc_hour
        );
    }

    #[test]
    fn test_current_with_tz() {
        let bio = Biorhythm::current_with_tz(-5.0);
        assert!((bio.timezone_offset_hours - (-5.0)).abs() < 0.01);
        // hour should be UTC, not local
        let utc_hour = Utc::now().hour() as f64 + (Utc::now().minute() as f64 / 60.0);
        assert!(
            (bio.hour - utc_hour).abs() < 0.1,
            "should store UTC: bio={}, utc={}",
            bio.hour,
            utc_hour
        );
    }

    #[test]
    fn test_effective_hour_includes_timezone() {
        let mut bio = Biorhythm::for_hour(10.0);
        bio.timezone_offset_hours = -5.0;
        assert!(
            (bio.effective_hour() - 5.0).abs() < 0.01,
            "10 UTC - 5h = 5: {}",
            bio.effective_hour()
        );
    }

    #[test]
    fn test_effective_hour_timezone_wraps() {
        let mut bio = Biorhythm::for_hour(2.0);
        bio.timezone_offset_hours = -5.0;
        // 2 - 5 = -3 → wraps to 21
        assert!(
            (bio.effective_hour() - 21.0).abs() < 0.01,
            "2 UTC - 5h wraps to 21: {}",
            bio.effective_hour()
        );
    }

    #[test]
    fn test_set_timezone_routes_through_shift_phase() {
        let mut bio = Biorhythm::for_hour(12.0);
        bio.set_timezone(5.0);
        assert!((bio.timezone_offset_hours - 5.0).abs() < 0.01);
        assert!(
            (bio.phase_offset - 5.0).abs() < 0.01,
            "delta should route through shift_phase: {}",
            bio.phase_offset
        );
    }

    #[test]
    fn test_set_timezone_clamps_range() {
        let mut bio = Biorhythm::for_hour(12.0);
        bio.set_timezone(20.0); // exceeds +14
        assert!((bio.timezone_offset_hours - 14.0).abs() < 0.01);
    }

    #[test]
    fn test_for_hour_defaults_timezone_to_zero() {
        let bio = Biorhythm::for_hour(12.0);
        assert!((bio.timezone_offset_hours).abs() < 0.001);
    }

    #[test]
    fn test_detect_system_timezone_in_range() {
        let tz = Biorhythm::detect_system_timezone();
        assert!(
            (-12.0..=14.0).contains(&tz),
            "detected tz should be in [-12, 14]: {}",
            tz
        );
    }

    // ── Phase 4: Circadian Phase Shifting ────────────────────────────

    #[test]
    fn test_shift_changes_effective_hour() {
        let mut bio = Biorhythm::for_hour(10.0);
        assert!((bio.effective_hour() - 10.0).abs() < 0.01);
        bio.shift_phase(3.0);
        assert!(
            (bio.effective_hour() - 13.0).abs() < 0.01,
            "Shifted +3h: {}",
            bio.effective_hour()
        );
    }

    #[test]
    fn test_entrainment_recovers() {
        let mut bio = Biorhythm::for_hour(10.0);
        bio.shift_phase(6.0);
        assert!((bio.phase_offset - 6.0).abs() < 0.01);
        // Entrain should pull toward 0
        for _ in 0..20 {
            bio.entrain();
        }
        assert!(
            bio.phase_offset.abs() < 0.5,
            "Should recover near 0: {}",
            bio.phase_offset
        );
    }

    #[test]
    fn test_effective_hour_wrapping() {
        let mut bio = Biorhythm::for_hour(23.0);
        bio.shift_phase(3.0);
        // 23 + 3 = 26 → wraps to 2.0
        assert!(
            (bio.effective_hour() - 2.0).abs() < 0.01,
            "Should wrap: {}",
            bio.effective_hour()
        );
    }

    #[test]
    fn test_shift_affects_neuromod_baselines() {
        // Verify that effective_hour() produces a different value from raw hour
        let mut bio = Biorhythm::for_hour(14.0); // 2pm
        bio.shift_phase(-6.0); // shifted to 8am effective
        let eff = bio.effective_hour();
        assert!((eff - 8.0).abs() < 0.01, "Effective should be 8am: {eff}");
        // This effective hour would produce different neuromod baselines
        // (verified by the circadian continuous test suite)
    }
}
