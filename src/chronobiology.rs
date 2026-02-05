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
    Dawn,   // Waking up, rising arousal
    Day,    // Peak activity, high focus
    Dusk,   // Winding down, reflection
    Night,  // Dreaming, memory consolidation
}

#[derive(Debug, Clone)]
pub struct Biorhythm {
    pub phase: CircadianPhase,
    pub arousal_mod: f64,    // Multiplier for attention
    pub plasticity_mod: f64, // Multiplier for learning rate
    pub creativity_mod: f64, // Multiplier for randomness (temperature)
}

impl Biorhythm {
    /// Calculate the current biorhythm based on local time
    pub fn current() -> Self {
        let now = Local::now();
        let hour = now.hour() as f64 + (now.minute() as f64 / 60.0);
        
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
        }
    }
}
