//! Chronobiology: The Rhythm of the Machine
//!
//! Implements circadian and ultradian rhythms to modulate the AGI's
//! cognitive parameters based on time.
//!
//! "To everything there is a season, and a time to every purpose under the heaven."

use chrono::{Local, Timelike};
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
    pub hour: f64,
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
    /// Calculate the current biorhythm based on local time
    pub fn current() -> Self {
        let now = Local::now();
        let hour = now.hour() as f64 + (now.minute() as f64 / 60.0);
        Self::for_hour(hour)
    }

    /// Calculate biorhythm for a given fractional hour (0.0–24.0).
    ///
    /// This is the deterministic core extracted from `current()` for testability.
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
        }
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
}
