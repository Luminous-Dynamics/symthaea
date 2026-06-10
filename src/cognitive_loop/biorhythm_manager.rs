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
}

impl BiorhythmManager {
    /// Create a new BiorhythmManager with the given timezone offset.
    pub fn new(timezone_offset_hours: f64) -> Self {
        Self {
            rhythm: Biorhythm::current_with_tz(timezone_offset_hours),
            refresh_counter: 0,
            timezone_offset_hours,
        }
    }

    /// Refresh the biorhythm from the current UTC time, preserving phase state.
    pub fn refresh(&mut self) {
        let old_phase_offset = self.rhythm.phase_offset;
        let old_entrainment_rate = self.rhythm.entrainment_rate;
        self.rhythm = Biorhythm::current_with_tz(self.timezone_offset_hours);
        self.rhythm.phase_offset = old_phase_offset;
        self.rhythm.entrainment_rate = old_entrainment_rate;
        self.refresh_counter = 0;
    }

    /// Update timezone, routing delta through `shift_phase()` for gradual entrainment.
    pub fn set_timezone(&mut self, new_tz: f64) {
        self.rhythm.set_timezone(new_tz);
        self.timezone_offset_hours = self.rhythm.timezone_offset_hours;
    }
}

impl Default for BiorhythmManager {
    fn default() -> Self {
        Self::new(0.0)
    }
}
