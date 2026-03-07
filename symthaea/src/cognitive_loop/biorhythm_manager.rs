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
}

impl Default for BiorhythmManager {
    fn default() -> Self {
        Self {
            rhythm: Biorhythm::current(),
            refresh_counter: 0,
        }
    }
}
