// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dream journal: stores narrative fragments from overnight consolidation.
//!
//! When the Spore enters Sleep + Charging + NightMode, dream consolidation
//! runs automatically. Wisdom entries feed into ThoughtChannels, then BrocaLite
//! generates short dream-like text fragments stored in a ring buffer.

use crate::broca::{BrocaLite, ThoughtChannels};
use crate::dream::Wisdom;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const MAX_DREAM_FRAGMENTS: usize = 32;
const DREAM_MAX_TOKENS: usize = 32;

/// A single dream narrative fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamFragment {
    pub narrative: String,
    pub phi_improvement: f32,
    pub dominant_emotion: String,
    pub cycle_recorded: u64,
}

/// Dream journal: ring buffer of narrative fragments.
pub struct DreamJournal {
    fragments: VecDeque<DreamFragment>,
}

impl DreamJournal {
    pub fn new() -> Self {
        Self {
            fragments: VecDeque::with_capacity(MAX_DREAM_FRAGMENTS),
        }
    }

    /// Generate a dream fragment from wisdom entries using BrocaLite.
    ///
    /// Returns true if a fragment was generated.
    pub fn consolidate(
        &mut self,
        wisdoms: &[Wisdom],
        broca: &mut BrocaLite,
        consciousness_level: f32,
        neuromods: [f32; 4],
        cycle: u64,
    ) -> bool {
        if wisdoms.is_empty() {
            return false;
        }

        // Aggregate wisdom into ThoughtChannels
        let total_phi: f32 = wisdoms.iter().map(|w| w.phi_improvement).sum();
        let avg_confidence: f32 =
            wisdoms.iter().map(|w| w.confidence).sum::<f32>() / wisdoms.len() as f32;

        let mut channels = ThoughtChannels::from_cycle(
            consciousness_level.max(0.3), // Dreams have at least some consciousness
            total_phi.clamp(0.0, 1.0),    // PE from phi improvement
            avg_confidence,               // Harmony from confidence
            neuromods,
        );
        // Mark as dreaming: set epistemic to high uncertainty (dream state)
        channels.set_epistemic(0.8);

        let result = broca.generate(&channels, DREAM_MAX_TOKENS);

        if result.text.is_empty() {
            return false;
        }

        // Determine dominant emotion from neuromods
        let dominant_emotion = dominant_emotion_from_neuromods(neuromods);

        let fragment = DreamFragment {
            narrative: result.text,
            phi_improvement: total_phi,
            dominant_emotion,
            cycle_recorded: cycle,
        };

        if self.fragments.len() >= MAX_DREAM_FRAGMENTS {
            self.fragments.pop_front();
        }
        self.fragments.push_back(fragment);
        true
    }

    /// Get the most recent fragment.
    pub fn latest(&self) -> Option<&DreamFragment> {
        self.fragments.back()
    }

    /// Get all fragments as a JSON string.
    pub fn all_json(&self) -> String {
        let v: Vec<&DreamFragment> = self.fragments.iter().collect();
        serde_json::to_string(&v).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get the latest fragment as JSON.
    pub fn latest_json(&self) -> String {
        match self.latest() {
            Some(f) => serde_json::to_string(f).unwrap_or_else(|_| "null".to_string()),
            None => "null".to_string(),
        }
    }

    /// Number of stored fragments.
    pub fn count(&self) -> u32 {
        self.fragments.len() as u32
    }

    /// Get all fragments.
    pub fn fragments(&self) -> &VecDeque<DreamFragment> {
        &self.fragments
    }
}

fn dominant_emotion_from_neuromods(neuromods: [f32; 4]) -> String {
    let [da, ne, sht, ot] = neuromods;
    // Simple heuristic: highest neuromod determines emotional tone
    let max = da.max(ne).max(sht).max(ot);
    if (max - da).abs() < 0.01 {
        "curiosity".to_string()
    } else if (max - ne).abs() < 0.01 {
        "alertness".to_string()
    } else if (max - sht).abs() < 0.01 {
        "serenity".to_string()
    } else {
        "warmth".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wisdoms(count: usize) -> Vec<Wisdom> {
        (0..count)
            .map(|i| Wisdom {
                context_state: vec![i as f32 * 0.1],
                better_action: vec![i as f32 * 0.2],
                phi_improvement: 0.05 + i as f32 * 0.01,
                confidence: 0.7,
            })
            .collect()
    }

    #[test]
    fn test_empty_wisdom_no_fragment() {
        let mut journal = DreamJournal::new();
        let mut broca = BrocaLite::new(42);
        let result = journal.consolidate(&[], &mut broca, 0.5, [0.5; 4], 1);
        assert!(!result);
        assert_eq!(journal.count(), 0);
    }

    #[test]
    fn test_consolidation_produces_fragment() {
        let mut journal = DreamJournal::new();
        let mut broca = BrocaLite::new(42);
        let wisdoms = make_wisdoms(3);
        let result = journal.consolidate(&wisdoms, &mut broca, 0.6, [0.5, 0.4, 0.6, 0.3], 100);
        assert!(result);
        assert_eq!(journal.count(), 1);
        let frag = journal.latest().unwrap();
        assert!(!frag.narrative.is_empty());
        assert_eq!(frag.cycle_recorded, 100);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut journal = DreamJournal::new();
        let mut broca = BrocaLite::new(42);
        let wisdoms = make_wisdoms(2);
        for i in 0..MAX_DREAM_FRAGMENTS + 5 {
            journal.consolidate(&wisdoms, &mut broca, 0.5, [0.5; 4], i as u64);
        }
        assert_eq!(journal.count(), MAX_DREAM_FRAGMENTS as u32);
    }

    #[test]
    fn test_latest_json() {
        let mut journal = DreamJournal::new();
        let mut broca = BrocaLite::new(42);
        let wisdoms = make_wisdoms(1);
        journal.consolidate(&wisdoms, &mut broca, 0.5, [0.5; 4], 50);
        let json = journal.latest_json();
        assert!(json.contains("narrative"));
        assert!(json.contains("phi_improvement"));
    }

    #[test]
    fn test_all_json() {
        let mut journal = DreamJournal::new();
        let mut broca = BrocaLite::new(42);
        let wisdoms = make_wisdoms(2);
        journal.consolidate(&wisdoms, &mut broca, 0.5, [0.5; 4], 1);
        journal.consolidate(&wisdoms, &mut broca, 0.6, [0.6, 0.4, 0.5, 0.3], 2);
        let json = journal.all_json();
        assert!(json.starts_with('['));
    }

    #[test]
    fn test_empty_latest_json() {
        let journal = DreamJournal::new();
        assert_eq!(journal.latest_json(), "null");
    }

    #[test]
    fn test_dominant_emotion() {
        assert_eq!(
            dominant_emotion_from_neuromods([0.9, 0.1, 0.1, 0.1]),
            "curiosity"
        );
        assert_eq!(
            dominant_emotion_from_neuromods([0.1, 0.9, 0.1, 0.1]),
            "alertness"
        );
        assert_eq!(
            dominant_emotion_from_neuromods([0.1, 0.1, 0.9, 0.1]),
            "serenity"
        );
        assert_eq!(
            dominant_emotion_from_neuromods([0.1, 0.1, 0.1, 0.9]),
            "warmth"
        );
    }
}
