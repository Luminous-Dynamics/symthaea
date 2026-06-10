// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Density regulator: ensures note spacing matches the section's emotional needs.
//!
//! Exposition should breathe (sparse). Climax should be dense. Coda should
//! wind down. Without this, notes are uniformly distributed regardless of
//! section — the music doesn't breathe.

use crate::structure::SectionType;
use std::collections::VecDeque;

/// Section-aware density targets (inter-onset interval in milliseconds).
pub struct DensityTarget {
    pub min_ioi_ms: f32,
    pub max_ioi_ms: f32,
    pub target_ioi_ms: f32,
}

impl DensityTarget {
    fn for_section(section: SectionType) -> Self {
        match section {
            SectionType::Ambient => Self {
                min_ioi_ms: 500.0,
                max_ioi_ms: 1200.0,
                target_ioi_ms: 700.0,
            },
            SectionType::Exploratory => Self {
                min_ioi_ms: 250.0,
                max_ioi_ms: 600.0,
                target_ioi_ms: 400.0,
            },
            SectionType::Developmental => Self {
                min_ioi_ms: 300.0,
                max_ioi_ms: 700.0,
                target_ioi_ms: 500.0,
            },
            SectionType::Climactic => Self {
                min_ioi_ms: 150.0,
                max_ioi_ms: 400.0,
                target_ioi_ms: 250.0,
            },
        }
    }
}

/// Action recommended by the density regulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DensityAction {
    NoAction,
    InsertRest,
    ReduceCadence(u32),
    IncreaseCadence(u32),
}

/// Tracks note onsets and regulates density per section.
pub struct DensityRegulator {
    recent_onsets: VecDeque<f32>, // timestamps in seconds
    max_history: usize,
    current_target: DensityTarget,
}

impl DensityRegulator {
    pub fn new() -> Self {
        Self {
            recent_onsets: VecDeque::with_capacity(64),
            max_history: 64,
            current_target: DensityTarget::for_section(SectionType::Ambient),
        }
    }

    pub fn set_section(&mut self, section: SectionType) {
        self.current_target = DensityTarget::for_section(section);
    }

    pub fn record_onset(&mut self, time_secs: f32) {
        self.recent_onsets.push_back(time_secs);
        if self.recent_onsets.len() > self.max_history {
            self.recent_onsets.pop_front();
        }
    }

    /// Mean inter-onset interval in milliseconds.
    pub fn mean_ioi_ms(&self) -> f32 {
        if self.recent_onsets.len() < 2 {
            return self.current_target.target_ioi_ms;
        }
        let iois: Vec<f32> = self
            .recent_onsets
            .iter()
            .zip(self.recent_onsets.iter().skip(1))
            .map(|(a, b)| (b - a) * 1000.0)
            .filter(|&ioi| ioi > 0.0)
            .collect();
        if iois.is_empty() {
            return self.current_target.target_ioi_ms;
        }
        iois.iter().sum::<f32>() / iois.len() as f32
    }

    /// Recommend an action based on current density vs target.
    pub fn recommend(&self) -> DensityAction {
        let mean = self.mean_ioi_ms();
        let target = &self.current_target;

        if mean < target.min_ioi_ms {
            // Way too dense — skip notes
            DensityAction::InsertRest
        } else if mean < target.target_ioi_ms * 0.8 {
            // Somewhat too dense — slow down
            DensityAction::IncreaseCadence(2)
        } else if mean > target.max_ioi_ms {
            // Way too sparse — generate more
            DensityAction::ReduceCadence(1)
        } else if mean > target.target_ioi_ms * 1.3 {
            // Somewhat too sparse
            DensityAction::ReduceCadence(1)
        } else {
            DensityAction::NoAction
        }
    }

    /// Adjust note_gen_cadence based on current density.
    pub fn adjust_cadence(&self, current: u32) -> u32 {
        // Respect the caller's cadence range — don't cap to 16.
        // Low-arousal states legitimately need cadences of 100+ chunks.
        match self.recommend() {
            DensityAction::IncreaseCadence(n) => current + n,
            DensityAction::ReduceCadence(n) => current.saturating_sub(n).max(1),
            DensityAction::InsertRest => current + 5,
            DensityAction::NoAction => current,
        }
    }

    pub fn reset(&mut self) {
        self.recent_onsets.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_onset_recommends_reduce() {
        let mut reg = DensityRegulator::new();
        reg.set_section(SectionType::Climactic); // wants dense (250ms)
        // Record onsets 2 seconds apart (very sparse)
        reg.record_onset(0.0);
        reg.record_onset(2.0);
        reg.record_onset(4.0);
        assert!(reg.mean_ioi_ms() > 1000.0);
        assert!(matches!(reg.recommend(), DensityAction::ReduceCadence(_)));
    }

    #[test]
    fn dense_onset_recommends_rest() {
        let mut reg = DensityRegulator::new();
        reg.set_section(SectionType::Ambient); // wants sparse (700ms)
        // Record onsets 50ms apart (very dense)
        for i in 0..20 {
            reg.record_onset(i as f32 * 0.05);
        }
        assert!(reg.mean_ioi_ms() < 100.0);
        assert_eq!(reg.recommend(), DensityAction::InsertRest);
    }

    #[test]
    fn good_density_no_action() {
        let mut reg = DensityRegulator::new();
        reg.set_section(SectionType::Developmental); // target 500ms
        for i in 0..10 {
            reg.record_onset(i as f32 * 0.5); // 500ms apart
        }
        let mean = reg.mean_ioi_ms();
        assert!((mean - 500.0).abs() < 50.0);
        assert_eq!(reg.recommend(), DensityAction::NoAction);
    }

    #[test]
    fn cadence_adjustment_bounded() {
        let mut reg = DensityRegulator::new();
        // With proper density data, cadence should stay bounded
        for i in 0..10 {
            reg.record_onset(i as f32 * 0.5); // 500ms apart (normal)
        }
        let adjusted = reg.adjust_cadence(4);
        assert!(
            adjusted >= 1 && adjusted <= 16,
            "cadence should be bounded: {adjusted}"
        );
    }
}
