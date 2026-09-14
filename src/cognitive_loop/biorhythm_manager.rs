// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Biorhythm Manager — groups chronobiology rhythm state.
//!
//! Consolidates biorhythm + refresh counter from CognitiveLoopService
//! into a single coherent module.

use crate::chronobiology::Biorhythm;

/// Consolidated biorhythm manager.
///
/// Groups the circadian/ultradian rhythm modulator and its refresh counter.
/// Refreshes every 100 cycles to avoid unnecessary chrono calls.
pub(crate) struct BiorhythmManager {
    /// Circadian/ultradian rhythm modulation.
    /// Modulates learning rate (plasticity) and exploration (creativity) based on local time.
    pub rhythm: Biorhythm,

    /// Cycle counter for biorhythm refresh (refreshes every 100 cycles).
    pub refresh_counter: usize,

    /// Timezone offset in hours from UTC (mirrors CognitiveLoopConfig).
    pub timezone_offset_hours: f64,

    /// Scientific-method-only clock authority. When set, every biorhythm
    /// reconstruction uses this UTC fractional hour instead of `Utc::now()`.
    ///
    /// This deliberately freezes only the clock input. It preserves the current
    /// production semantics for timezone, phase offset, entrainment, and phase
    /// classification rather than repairing/reinterpreting them inside an
    /// experiment lineage.
    #[cfg(any(test, feature = "scientific_method"))]
    experimental_fixed_utc_hour: Option<f64>,
}

impl BiorhythmManager {
    /// Create a new BiorhythmManager with the given timezone offset.
    pub fn new(timezone_offset_hours: f64) -> Self {
        Self {
            rhythm: Biorhythm::current_with_tz(timezone_offset_hours),
            refresh_counter: 0,
            timezone_offset_hours,
            #[cfg(any(test, feature = "scientific_method"))]
            experimental_fixed_utc_hour: None,
        }
    }

    /// Build the next rhythm from the currently authorized clock source.
    fn rhythm_from_authorized_clock(&self) -> Biorhythm {
        #[cfg(any(test, feature = "scientific_method"))]
        if let Some(hour) = self.experimental_fixed_utc_hour {
            // Reproduce `Biorhythm::current_with_tz()` semantics exactly, except
            // the UTC hour is supplied by the experiment instead of wall clock.
            let mut rhythm = Biorhythm::for_hour(hour);
            rhythm.hour = hour;
            rhythm.timezone_offset_hours = self.timezone_offset_hours;
            return rhythm;
        }

        Biorhythm::current_with_tz(self.timezone_offset_hours)
    }

    /// Refresh the biorhythm from the authorized UTC time, preserving phase state.
    pub fn refresh(&mut self) {
        let old_phase_offset = self.rhythm.phase_offset;
        let old_entrainment_rate = self.rhythm.entrainment_rate;
        self.rhythm = self.rhythm_from_authorized_clock();
        self.rhythm.phase_offset = old_phase_offset;
        self.rhythm.entrainment_rate = old_entrainment_rate;
        self.refresh_counter = 0;
    }

    /// Update timezone, routing delta through `shift_phase()` for gradual entrainment.
    pub fn set_timezone(&mut self, new_tz: f64) {
        self.rhythm.set_timezone(new_tz);
        self.timezone_offset_hours = self.rhythm.timezone_offset_hours;
    }

    /// Install a fixed UTC fractional hour for scientific/reproducibility runs.
    ///
    /// Installation immediately rebuilds the current rhythm, so cycles before
    /// the normal refresh interval cannot retain ambient construction time.
    #[cfg(any(test, feature = "scientific_method"))]
    pub(crate) fn install_experimental_fixed_utc_hour(
        &mut self,
        hour: f64,
    ) -> Result<(), &'static str> {
        if !hour.is_finite() {
            return Err("experimental UTC hour must be finite");
        }
        if !(0.0..24.0).contains(&hour) {
            return Err("experimental UTC hour must be in [0, 24)");
        }

        self.experimental_fixed_utc_hour = Some(hour);
        self.refresh();
        Ok(())
    }

    /// Return the installed fixed UTC hour, if scientific clock override is active.
    #[cfg(any(test, feature = "scientific_method"))]
    pub(crate) fn experimental_fixed_utc_hour(&self) -> Option<f64> {
        self.experimental_fixed_utc_hour
    }

    /// Clear the scientific clock override and return to ambient UTC time.
    #[cfg(any(test, feature = "scientific_method"))]
    pub(crate) fn clear_experimental_fixed_utc_hour(&mut self) {
        self.experimental_fixed_utc_hour = None;
        self.refresh();
    }
}

/// Narrow scientific-method boundary for deterministic clock control.
///
/// The normal production API does not expose these methods. They are compiled
/// only when the existing `scientific_method` feature is enabled, and they do
/// not expose the underlying manager or biorhythm for arbitrary mutation.
#[cfg(feature = "scientific_method")]
impl super::CognitiveLoopService {
    /// Freeze the biorhythm clock source to one UTC fractional hour in `[0, 24)`.
    pub fn install_experimental_fixed_utc_hour(
        &mut self,
        hour: f64,
    ) -> Result<(), &'static str> {
        self.biorhythm_mgr
            .install_experimental_fixed_utc_hour(hour)
    }

    /// Return the currently installed fixed UTC hour, if any.
    pub fn experimental_fixed_utc_hour(&self) -> Option<f64> {
        self.biorhythm_mgr.experimental_fixed_utc_hour()
    }

    /// Return biorhythm reconstruction to the ambient UTC clock.
    pub fn clear_experimental_fixed_utc_hour(&mut self) {
        self.biorhythm_mgr.clear_experimental_fixed_utc_hour();
    }
}

impl Default for BiorhythmManager {
    fn default() -> Self {
        Self::new(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_utc_hour_applies_immediately_and_survives_refresh() {
        let mut manager = BiorhythmManager::new(2.0);
        manager
            .install_experimental_fixed_utc_hour(13.25)
            .expect("valid fixed hour");

        assert_eq!(manager.experimental_fixed_utc_hour(), Some(13.25));
        assert!((manager.rhythm.hour - 13.25).abs() < f64::EPSILON);
        assert!((manager.timezone_offset_hours - 2.0).abs() < f64::EPSILON);
        assert!((manager.rhythm.timezone_offset_hours - 2.0).abs() < f64::EPSILON);

        manager.refresh_counter = 96;
        manager.refresh();

        assert!((manager.rhythm.hour - 13.25).abs() < f64::EPSILON);
        assert_eq!(manager.refresh_counter, 0);
    }

    #[test]
    fn fixed_refresh_preserves_phase_offset_and_entrainment_semantics() {
        let mut manager = BiorhythmManager::new(-5.0);
        manager.rhythm.phase_offset = 3.5;
        manager.rhythm.entrainment_rate = 0.75;
        manager
            .install_experimental_fixed_utc_hour(7.5)
            .expect("valid fixed hour");

        assert!((manager.rhythm.phase_offset - 3.5).abs() < f64::EPSILON);
        assert!((manager.rhythm.entrainment_rate - 0.75).abs() < f64::EPSILON);
        assert!((manager.rhythm.timezone_offset_hours + 5.0).abs() < f64::EPSILON);

        manager.refresh();
        assert!((manager.rhythm.hour - 7.5).abs() < f64::EPSILON);
        assert!((manager.rhythm.phase_offset - 3.5).abs() < f64::EPSILON);
        assert!((manager.rhythm.entrainment_rate - 0.75).abs() < f64::EPSILON);
        assert!((manager.rhythm.timezone_offset_hours + 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn invalid_fixed_hours_fail_closed_without_installing() {
        let mut manager = BiorhythmManager::default();
        for invalid in [f64::NAN, f64::INFINITY, -0.001, 24.0] {
            assert!(manager.install_experimental_fixed_utc_hour(invalid).is_err());
            assert_eq!(manager.experimental_fixed_utc_hour(), None);
        }
    }

    #[test]
    fn clear_returns_manager_to_ambient_clock_mode() {
        let mut manager = BiorhythmManager::new(9.0);
        manager
            .install_experimental_fixed_utc_hour(4.5)
            .expect("valid fixed hour");
        assert_eq!(manager.experimental_fixed_utc_hour(), Some(4.5));

        manager.clear_experimental_fixed_utc_hour();
        assert_eq!(manager.experimental_fixed_utc_hour(), None);
        assert!((manager.timezone_offset_hours - 9.0).abs() < f64::EPSILON);
        assert!((manager.rhythm.timezone_offset_hours - 9.0).abs() < f64::EPSILON);
    }
}
