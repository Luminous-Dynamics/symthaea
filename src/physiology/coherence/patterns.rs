// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Week 9 Phase 3: Resonance Pattern Recognition
//!
//! This module implements pattern recognition for successful states:
//! - `ResonancePattern` - A recognized successful state combination
//! - `PatternLibrary` - Collection of discovered successful patterns
//!
//! ## Key Insight
//!
//! Some states just WORK. When we find them, remember them!
//! The system learns "I do well when I'm in THIS state for THIS kind of work."

use std::cmp::Ordering;
use std::time::Instant;

use super::super::endocrine::HormoneState;

/// **Week 9 Phase 3: Resonance Pattern**
///
/// A recognized combination of coherence, resonance, and hormones that
/// consistently leads to successful task performance.
///
/// **Key Insight**: Some states just WORK. When we find them, remember them!
#[derive(Debug, Clone)]
pub struct ResonancePattern {
    /// Coherence level during this successful state
    pub coherence: f32,

    /// Relational resonance during this successful state
    pub resonance: f32,

    /// Hormone state during this successful state
    pub hormones: HormoneState,

    /// What context/task made this successful
    pub context: String,

    /// How reliably does this pattern lead to success? (0.0-1.0)
    pub success_rate: f32,

    /// When was this pattern last observed
    pub last_seen: Instant,

    /// How many times has this pattern been observed
    pub observation_count: u32,
}

/// **Week 9 Phase 3: Pattern Library**
///
/// Maintains a collection of successful resonance patterns.
/// The system learns "I do well when I'm in THIS state for THIS kind of work."
#[derive(Debug, Clone)]
pub struct PatternLibrary {
    /// Collection of discovered successful patterns
    patterns: Vec<ResonancePattern>,

    /// Maximum number of patterns to remember
    capacity: usize,
}

impl PatternLibrary {
    /// Create new pattern library
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            capacity: 50, // Remember top 50 patterns
        }
    }

    /// Try to recognize if current state matches a known successful pattern
    pub fn recognize_pattern(
        &self,
        coherence: f32,
        resonance: f32,
        hormones: &HormoneState,
    ) -> Option<&ResonancePattern> {
        // Find a pattern that matches current state
        self.patterns.iter().find(|p| {
            // Allow 10% tolerance on coherence and resonance
            let coherence_match = (p.coherence - coherence).abs() < 0.1;
            let resonance_match = (p.resonance - resonance).abs() < 0.1;

            // Hormones should be "similar" (all within 0.2)
            let hormone_match = (p.hormones.dopamine - hormones.dopamine).abs() < 0.2
                && (p.hormones.acetylcholine - hormones.acetylcholine).abs() < 0.2
                && (p.hormones.cortisol - hormones.cortisol).abs() < 0.2;

            coherence_match && resonance_match && hormone_match
        })
    }

    /// Record a successful state as a pattern
    pub fn record_success(
        &mut self,
        coherence: f32,
        resonance: f32,
        hormones: HormoneState,
        context: String,
    ) {
        // Check if we already have a pattern for this context
        if let Some(existing) = self.patterns.iter_mut().find(|p| p.context == context) {
            // Update existing pattern (exponential moving average)
            existing.coherence = (existing.coherence * 0.7) + (coherence * 0.3);
            existing.resonance = (existing.resonance * 0.7) + (resonance * 0.3);
            existing.hormones = hormones.clone();
            existing.success_rate = (existing.success_rate * 0.9) + 0.1;
            existing.last_seen = Instant::now();
            existing.observation_count += 1;

            tracing::debug!(
                "Updated pattern '{}': coherence={:.2}, resonance={:.2}, success_rate={:.2}, count={}",
                context,
                existing.coherence,
                existing.resonance,
                existing.success_rate,
                existing.observation_count
            );
        } else {
            // Create new pattern
            let pattern = ResonancePattern {
                coherence,
                resonance,
                hormones: hormones.clone(),
                context: context.clone(),
                success_rate: 1.0,
                last_seen: Instant::now(),
                observation_count: 1,
            };

            self.patterns.push(pattern);

            tracing::info!(
                "Discovered new pattern '{}': coherence={:.2}, resonance={:.2}",
                context,
                coherence,
                resonance
            );

            // Enforce capacity limit (remove oldest/worst patterns)
            if self.patterns.len() > self.capacity {
                self.prune_patterns();
            }
        }
    }

    /// Suggest optimal coherence + resonance for a given context
    pub fn suggest_state(&self, context: &str) -> Option<(f32, f32)> {
        // Find the best pattern for this context
        self.patterns
            .iter()
            .filter(|p| p.context.contains(context))
            .max_by(|a, b| {
                // Sort by success_rate, then by observation_count
                a.success_rate
                    .partial_cmp(&b.success_rate)
                    .unwrap_or(Ordering::Equal)
                    .then(a.observation_count.cmp(&b.observation_count))
            })
            .map(|p| {
                tracing::info!(
                    "Suggested state for '{}': coherence={:.2}, resonance={:.2} (success_rate={:.0}%, count={})",
                    context,
                    p.coherence,
                    p.resonance,
                    p.success_rate * 100.0,
                    p.observation_count
                );
                (p.coherence, p.resonance)
            })
    }

    /// Remove least useful patterns when at capacity
    fn prune_patterns(&mut self) {
        // Sort by usefulness (success_rate * observation_count)
        self.patterns.sort_by(|a, b| {
            let usefulness_a = a.success_rate * (a.observation_count as f32);
            let usefulness_b = b.success_rate * (b.observation_count as f32);
            usefulness_b
                .partial_cmp(&usefulness_a)
                .unwrap_or(Ordering::Equal)
        });

        // Keep only the top capacity
        self.patterns.truncate(self.capacity);
    }

    /// Get number of patterns discovered
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

impl Default for PatternLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_library_records_and_recognizes_patterns() {
        let mut library = PatternLibrary::new();

        let hormones = HormoneState {
            dopamine: 0.7,
            cortisol: 0.2,
            acetylcholine: 0.8,
            oxytocin: 0.5,
            norepinephrine: 0.5,
            serotonin: 0.5,
        };

        // Record a successful pattern
        library.record_success(0.8, 0.9, hormones.clone(), "deep_analysis".to_string());

        assert_eq!(library.pattern_count(), 1, "Should have recorded 1 pattern");

        // Try to recognize the same pattern
        let recognized = library.recognize_pattern(0.8, 0.9, &hormones);
        assert!(
            recognized.is_some(),
            "Should recognize the pattern we just recorded"
        );

        let pattern = recognized.unwrap();
        assert_eq!(pattern.context, "deep_analysis");
        assert!((pattern.coherence - 0.8).abs() < 0.001);
        assert!((pattern.resonance - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_pattern_library_updates_existing_patterns() {
        let mut library = PatternLibrary::new();
        let hormones = HormoneState::neutral();

        // Record same context multiple times with different coherence
        library.record_success(0.7, 0.8, hormones.clone(), "creative_work".to_string());
        library.record_success(0.9, 0.85, hormones.clone(), "creative_work".to_string());

        // Should still have only 1 pattern (updated, not duplicated)
        assert_eq!(
            library.pattern_count(),
            1,
            "Should update existing pattern, not create duplicate"
        );

        // The pattern should reflect the exponential moving average
        if let Some(suggested) = library.suggest_state("creative_work") {
            let (coh, res) = suggested;
            // Should be weighted toward the second recording (0.9, 0.85)
            assert!(
                coh > 0.7 && coh < 0.9,
                "Coherence should be averaged: {}",
                coh
            );
            assert!(
                res > 0.8 && res < 0.85,
                "Resonance should be averaged: {}",
                res
            );
        } else {
            panic!("Should have a pattern for creative_work");
        }
    }

    #[test]
    fn test_pattern_library_suggests_optimal_states() {
        let mut library = PatternLibrary::new();
        let hormones = HormoneState::neutral();

        // Record several successful patterns for different contexts
        library.record_success(0.8, 0.9, hormones.clone(), "deep_analysis".to_string());
        library.record_success(0.6, 0.7, hormones.clone(), "routine_work".to_string());
        library.record_success(0.9, 0.95, hormones.clone(), "creative_flow".to_string());

        assert_eq!(
            library.pattern_count(),
            3,
            "Should have 3 distinct patterns"
        );

        // Get suggestions for each context
        let deep = library.suggest_state("deep_analysis");
        let routine = library.suggest_state("routine_work");
        let creative = library.suggest_state("creative_flow");

        assert!(deep.is_some(), "Should suggest state for deep_analysis");
        assert!(routine.is_some(), "Should suggest state for routine_work");
        assert!(creative.is_some(), "Should suggest state for creative_flow");

        // Suggestions should match what we recorded
        let (deep_coh, deep_res) = deep.unwrap();
        assert!(
            (deep_coh - 0.8).abs() < 0.1,
            "Deep analysis coherence: {}",
            deep_coh
        );
        assert!(
            (deep_res - 0.9).abs() < 0.1,
            "Deep analysis resonance: {}",
            deep_res
        );

        let (creative_coh, creative_res) = creative.unwrap();
        assert!(
            creative_coh > deep_coh,
            "Creative should need higher coherence"
        );
        assert!(
            creative_res > deep_res,
            "Creative should need higher resonance"
        );
    }
}
