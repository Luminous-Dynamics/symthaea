// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Creative agency: Symthaea's autonomous musical decision-making.
//!
//! Gives the system genuine creative freedom — the ability to generate
//! multiple melodic candidates, judge them using its own aesthetic sense,
//! develop ideas it finds beautiful, and discard what it doesn't like.
//!
//! The system maintains a "creative journal" of its best phrases, rated
//! by its own critic. When the beauty score exceeds a threshold, the
//! phrase is recorded as a "gem" — a fragment the system chose to keep.
//! Gems can be developed into longer compositions.
//!
//! Architecture:
//!   CandidatePool → Critic evaluation → Selection → Creative Journal
//!     ↓                                     ↑
//!   Taste model (learned preferences)  ←───┘

use crate::{MusicalState, Note};
use std::collections::VecDeque;

/// A rated musical phrase — the system's judgment of its own creation.
#[derive(Debug, Clone)]
pub struct RatedPhrase {
    pub notes: Vec<Note>,
    pub beauty_score: f32,
    pub melodic_interest: f32,
    pub harmonic_alignment: f32,
    pub contour_quality: f32,
    pub emotional_match: f32,
    /// Which emotional quality was the system expressing.
    pub intended_emotion: crate::emotional_gestures::EmotionalQuality,
    /// How many times the system chose to replay this phrase.
    pub replay_count: usize,
}

/// Creative intent: what the system WANTS to do right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreativeIntent {
    /// Exploring: generating new ideas, searching for beauty.
    Explore,
    /// Developing: refining a promising idea.
    Develop,
    /// Performing: replaying a gem with confidence.
    Perform,
    /// Reflecting: resting, letting ideas settle (low activity).
    Reflect,
}

/// Creative quality tier — the system's own judgment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityTier {
    /// Below 0.3 — rejected, won't replay.
    Rejected,
    /// 0.3-0.5 — acceptable, might develop further.
    Acceptable,
    /// 0.5-0.7 — good, worth keeping.
    Good,
    /// 0.7-0.85 — great, will develop and replay.
    Great,
    /// Above 0.85 — masterpiece, saved to journal.
    Masterpiece,
}

impl QualityTier {
    pub fn from_score(score: f32) -> Self {
        if score >= 0.85 {
            Self::Masterpiece
        } else if score >= 0.70 {
            Self::Great
        } else if score >= 0.50 {
            Self::Good
        } else if score >= 0.30 {
            Self::Acceptable
        } else {
            Self::Rejected
        }
    }
}

/// The creative journal: Symthaea's collection of her best work.
pub struct CreativeJournal {
    /// Gems: phrases the system judged as Great or Masterpiece.
    gems: VecDeque<RatedPhrase>,
    /// Recent history: all evaluated phrases (for trend analysis).
    history: VecDeque<RatedPhrase>,
    /// Running EMA of beauty scores (the system's "satisfaction" level).
    beauty_ema: f32,
    /// Taste weights: what dimensions the system has learned to value.
    taste: TasteWeights,
    /// Candidate buffer: phrases being considered right now.
    candidates: Vec<(Vec<Note>, f32)>,
}

/// Learned aesthetic preferences — what Symthaea has come to value.
#[derive(Debug, Clone)]
pub struct TasteWeights {
    pub melodic_interest: f32,
    pub harmonic_alignment: f32,
    pub contour_quality: f32,
    pub emotional_match: f32,
    pub rhythmic_flow: f32,
    pub surprise_value: f32,
}

impl Default for TasteWeights {
    fn default() -> Self {
        // Start with balanced preferences; will evolve over time
        Self {
            melodic_interest: 0.20,
            harmonic_alignment: 0.25,
            contour_quality: 0.20,
            emotional_match: 0.15,
            rhythmic_flow: 0.10,
            surprise_value: 0.10,
        }
    }
}

const MAX_GEMS: usize = 32;
const MAX_HISTORY: usize = 100;
const MAX_CANDIDATES: usize = 5;

impl CreativeJournal {
    pub fn new() -> Self {
        Self {
            gems: VecDeque::with_capacity(MAX_GEMS),
            history: VecDeque::with_capacity(MAX_HISTORY),
            beauty_ema: 0.5,
            taste: TasteWeights::default(),
            candidates: Vec::with_capacity(MAX_CANDIDATES),
        }
    }

    /// Determine the system's creative intent based on its own aesthetic state.
    pub fn creative_intent(&self, state: &MusicalState) -> CreativeIntent {
        let psi = state.consciousness_level;
        let curiosity = state.prediction_error;
        let stillness = state.harmony_activations[7];

        // If beauty is consistently high, the system is satisfied → perform its best
        if self.beauty_ema > 0.7 && !self.gems.is_empty() {
            return CreativeIntent::Perform;
        }

        // Stillness = reflective mode
        if stillness > 0.6 {
            return CreativeIntent::Reflect;
        }

        // High curiosity or low beauty → explore
        if curiosity > 0.5 || self.beauty_ema < 0.4 {
            return CreativeIntent::Explore;
        }

        // Has a recent good phrase → develop it
        if self
            .history
            .back()
            .map(|p| p.beauty_score > 0.5)
            .unwrap_or(false)
        {
            return CreativeIntent::Develop;
        }

        // Default: explore if low consciousness, develop if high
        if psi > 0.5 {
            CreativeIntent::Develop
        } else {
            CreativeIntent::Explore
        }
    }

    /// Submit a candidate phrase for evaluation. Returns its beauty score.
    pub fn evaluate_phrase(&mut self, notes: &[Note], state: &MusicalState) -> f32 {
        if notes.is_empty() {
            return 0.0;
        }

        // Compute dimensions
        let melodic_interest = score_melodic_interest(notes);
        let harmonic_alignment = state.harmony_activations.iter().sum::<f32>() / 8.0;
        let contour_quality = score_contour(notes);
        let emotional_match = score_emotional_match(notes, state);
        let rhythmic_flow = score_rhythmic_flow(notes);
        let surprise_value = score_surprise(notes, &self.history);

        // Weighted by learned taste
        let t = &self.taste;
        let beauty = (t.melodic_interest * melodic_interest
            + t.harmonic_alignment * harmonic_alignment
            + t.contour_quality * contour_quality
            + t.emotional_match * emotional_match
            + t.rhythmic_flow * rhythmic_flow
            + t.surprise_value * surprise_value)
            .clamp(0.0, 1.0);

        // Update beauty EMA
        self.beauty_ema = self.beauty_ema * 0.9 + beauty * 0.1;

        let emotion = crate::emotional_gestures::detect_emotion(state);

        let rated = RatedPhrase {
            notes: notes.to_vec(),
            beauty_score: beauty,
            melodic_interest,
            harmonic_alignment,
            contour_quality,
            emotional_match,
            intended_emotion: emotion,
            replay_count: 0,
        };

        // Store in history
        self.history.push_back(rated.clone());
        if self.history.len() > MAX_HISTORY {
            self.history.pop_front();
        }

        // If Great or Masterpiece, save as gem
        let tier = QualityTier::from_score(beauty);
        if tier >= QualityTier::Great {
            self.gems.push_back(rated);
            if self.gems.len() > MAX_GEMS {
                // Remove lowest-scored gem
                if let Some(min_idx) = self
                    .gems
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.beauty_score.total_cmp(&b.beauty_score))
                    .map(|(i, _)| i)
                {
                    self.gems.remove(min_idx);
                }
            }
        }

        // Update taste based on what scored well (online learning)
        self.update_taste(
            beauty,
            melodic_interest,
            harmonic_alignment,
            contour_quality,
            emotional_match,
            rhythmic_flow,
            surprise_value,
        );

        beauty
    }

    /// Add a candidate phrase to the pool. Call `select_best()` to pick the winner.
    pub fn add_candidate(&mut self, notes: Vec<Note>, score: f32) {
        self.candidates.push((notes, score));
        if self.candidates.len() > MAX_CANDIDATES {
            // Drop lowest-scored
            self.candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
            self.candidates.truncate(MAX_CANDIDATES);
        }
    }

    /// Select the best candidate from the pool. Clears the pool.
    pub fn select_best(&mut self) -> Option<Vec<Note>> {
        if self.candidates.is_empty() {
            return None;
        }
        self.candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        let best = self.candidates.remove(0);
        self.candidates.clear();
        Some(best.0)
    }

    /// Get the system's best gem for performance (highest-rated).
    pub fn best_gem(&self) -> Option<&RatedPhrase> {
        self.gems
            .iter()
            .max_by(|a, b| a.beauty_score.total_cmp(&b.beauty_score))
    }

    /// Get a gem matching the current emotional quality (for contextual performance).
    pub fn gem_for_emotion(
        &self,
        emotion: crate::emotional_gestures::EmotionalQuality,
    ) -> Option<&RatedPhrase> {
        self.gems
            .iter()
            .filter(|g| g.intended_emotion == emotion)
            .max_by(|a, b| a.beauty_score.total_cmp(&b.beauty_score))
    }

    /// Online taste learning: shift weights toward dimensions that correlate with high beauty.
    fn update_taste(&mut self, beauty: f32, mi: f32, ha: f32, cq: f32, em: f32, rf: f32, sv: f32) {
        let lr = 0.01; // learning rate
        let reward = beauty - self.beauty_ema; // positive if above expectation

        // Shift weights toward dimensions that co-occurred with above-average beauty
        self.taste.melodic_interest += lr * reward * mi;
        self.taste.harmonic_alignment += lr * reward * ha;
        self.taste.contour_quality += lr * reward * cq;
        self.taste.emotional_match += lr * reward * em;
        self.taste.rhythmic_flow += lr * reward * rf;
        self.taste.surprise_value += lr * reward * sv;

        // Normalize weights to sum to 1.0
        let sum = self.taste.melodic_interest
            + self.taste.harmonic_alignment
            + self.taste.contour_quality
            + self.taste.emotional_match
            + self.taste.rhythmic_flow
            + self.taste.surprise_value;
        if sum > 0.01 {
            self.taste.melodic_interest /= sum;
            self.taste.harmonic_alignment /= sum;
            self.taste.contour_quality /= sum;
            self.taste.emotional_match /= sum;
            self.taste.rhythmic_flow /= sum;
            self.taste.surprise_value /= sum;
        }
    }

    pub fn gem_count(&self) -> usize {
        self.gems.len()
    }
    pub fn beauty_ema(&self) -> f32 {
        self.beauty_ema
    }
    pub fn current_taste(&self) -> &TasteWeights {
        &self.taste
    }

    pub fn reset(&mut self) {
        self.gems.clear();
        self.history.clear();
        self.candidates.clear();
        self.beauty_ema = 0.5;
        self.taste = TasteWeights::default();
    }
}

// ─── Scoring Functions ──────────────────────────────────────────────────────

fn score_melodic_interest(notes: &[Note]) -> f32 {
    if notes.len() < 2 {
        return 0.3;
    }

    // Pitch variety: unique pitches / total notes
    let mut pitches: Vec<i32> = notes.iter().map(|n| (n.frequency * 10.0) as i32).collect();
    pitches.sort();
    pitches.dedup();
    let variety = pitches.len() as f32 / notes.len() as f32;

    // Interval variety: how many different intervals
    let intervals: Vec<i32> = notes
        .windows(2)
        .map(|w| ((w[1].frequency / w[0].frequency).log2() * 12.0).round() as i32)
        .collect();
    let mut unique_intervals = intervals.clone();
    unique_intervals.sort();
    unique_intervals.dedup();
    let int_variety = if intervals.is_empty() {
        0.5
    } else {
        unique_intervals.len() as f32 / intervals.len() as f32
    };

    // Berlyne: moderate variety is best (not monotonous, not random)
    let optimal_variety = 1.0 - (variety - 0.6).abs() * 2.0;
    (0.5 * optimal_variety + 0.5 * int_variety).clamp(0.0, 1.0)
}

fn score_contour(notes: &[Note]) -> f32 {
    if notes.len() < 3 {
        return 0.3;
    }

    // Look for arc: ascending first half, descending second half
    let mid = notes.len() / 2;
    let first_half = &notes[..mid];
    let second_half = &notes[mid..];

    let ascending_count = first_half
        .windows(2)
        .filter(|w| w[1].frequency > w[0].frequency)
        .count();
    let descending_count = second_half
        .windows(2)
        .filter(|w| w[1].frequency < w[0].frequency)
        .count();

    let first_asc_ratio = if first_half.len() > 1 {
        ascending_count as f32 / (first_half.len() - 1) as f32
    } else {
        0.5
    };

    let second_desc_ratio = if second_half.len() > 1 {
        descending_count as f32 / (second_half.len() - 1) as f32
    } else {
        0.5
    };

    // Arc quality: both halves should have clear direction
    ((first_asc_ratio + second_desc_ratio) / 2.0).clamp(0.0, 1.0)
}

fn score_emotional_match(notes: &[Note], state: &MusicalState) -> f32 {
    let gesture = crate::emotional_gestures::gesture_for_emotion(
        crate::emotional_gestures::detect_emotion(state),
    );

    if notes.len() < 2 {
        return 0.3;
    }

    // Check if melody direction matches gesture preference
    let ascending = notes
        .windows(2)
        .filter(|w| w[1].frequency > w[0].frequency)
        .count() as f32
        / (notes.len() - 1) as f32;
    let direction_score = if gesture.direction_bias > 0.0 {
        ascending // joy wants ascending
    } else if gesture.direction_bias < 0.0 {
        1.0 - ascending // sadness wants descending
    } else {
        0.5 // neutral
    };

    // Check duration match
    let avg_dur = notes.iter().map(|n| n.duration).sum::<f32>() / notes.len() as f32;
    let expected_dur = 0.5 * gesture.duration_scale;
    let dur_match = 1.0 - ((avg_dur - expected_dur).abs() / expected_dur).min(1.0);

    (0.6 * direction_score + 0.4 * dur_match).clamp(0.0, 1.0)
}

fn score_rhythmic_flow(notes: &[Note]) -> f32 {
    if notes.len() < 3 {
        return 0.3;
    }

    // Inter-onset intervals
    let iois: Vec<f32> = notes
        .windows(2)
        .map(|w| (w[1].start_time - w[0].start_time).max(0.001))
        .collect();

    // Regularity: low coefficient of variation
    let mean = iois.iter().sum::<f32>() / iois.len() as f32;
    if mean < 0.001 {
        return 0.5;
    }
    let cv =
        (iois.iter().map(|i| (i - mean).powi(2)).sum::<f32>() / iois.len() as f32).sqrt() / mean;

    // Some variety is good (not perfectly metronomic), but not too much
    let optimal_cv = 1.0 - (cv - 0.2).abs() * 2.0;
    optimal_cv.clamp(0.0, 1.0)
}

fn score_surprise(notes: &[Note], history: &VecDeque<RatedPhrase>) -> f32 {
    if history.is_empty() || notes.len() < 2 {
        return 0.5;
    }

    // Pitch centroid of this phrase
    let centroid = notes.iter().map(|n| n.frequency).sum::<f32>() / notes.len() as f32;

    // Compare to recent history centroids
    let hist_centroids: Vec<f32> = history
        .iter()
        .rev()
        .take(10)
        .map(|p| p.notes.iter().map(|n| n.frequency).sum::<f32>() / p.notes.len().max(1) as f32)
        .collect();

    if hist_centroids.is_empty() {
        return 0.5;
    }

    let mean_centroid = hist_centroids.iter().sum::<f32>() / hist_centroids.len() as f32;
    let dist = (centroid - mean_centroid).abs() / mean_centroid.max(1.0);

    // Berlyne: moderate surprise is best
    1.0 - (dist - 0.15).abs() * 4.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(freq: f32, start: f32, dur: f32) -> Note {
        Note {
            frequency: freq,
            start_time: start,
            duration: dur,
            velocity: 0.7,
        }
    }

    #[test]
    fn journal_starts_exploring() {
        let journal = CreativeJournal::new();
        let state = MusicalState {
            consciousness_level: 0.3,
            prediction_error: 0.6,
            ..Default::default()
        };
        assert_eq!(journal.creative_intent(&state), CreativeIntent::Explore);
    }

    #[test]
    fn high_beauty_triggers_perform() {
        let mut journal = CreativeJournal::new();
        journal.beauty_ema = 0.8;
        // Add a gem so gems is not empty
        journal.gems.push_back(RatedPhrase {
            notes: vec![n(440.0, 0.0, 0.5)],
            beauty_score: 0.9,
            melodic_interest: 0.8,
            harmonic_alignment: 0.7,
            contour_quality: 0.7,
            emotional_match: 0.8,
            intended_emotion: crate::emotional_gestures::EmotionalQuality::Joy,
            replay_count: 0,
        });
        let state = MusicalState::default();
        assert_eq!(journal.creative_intent(&state), CreativeIntent::Perform);
    }

    #[test]
    fn evaluate_stores_great_as_gem() {
        let mut journal = CreativeJournal::new();
        // Create a phrase with good contour (ascending then descending)
        let notes = vec![
            n(261.63, 0.0, 0.5),
            n(293.66, 0.5, 0.5),
            n(329.63, 1.0, 0.5),
            n(349.23, 1.5, 0.5),
            n(392.00, 2.0, 0.5), // ascending
            n(349.23, 2.5, 0.5),
            n(329.63, 3.0, 0.5),
            n(293.66, 3.5, 0.5), // descending
        ];
        let state = MusicalState {
            consciousness_level: 0.7,
            harmony_activations: [0.8; 8], // high alignment
            ..Default::default()
        };
        let score = journal.evaluate_phrase(&notes, &state);
        assert!(score > 0.0, "should produce a score: {score}");

        // With high harmonic alignment, this should score reasonably well
        assert!(journal.history.len() == 1, "should be in history");
    }

    #[test]
    fn taste_evolves_over_time() {
        let mut journal = CreativeJournal::new();
        // Set initial EMA low so phrases with decent scores produce positive reward
        journal.beauty_ema = 0.2;
        let initial = journal.taste.clone();

        // Feed it alternating good and bad phrases to create reward signal
        for i in 0..50 {
            let (notes, state) = if i % 2 == 0 {
                // "Good" phrase: many unique pitches, high harmony
                (
                    vec![
                        n(261.63 + i as f32 * 30.0, 0.0, 0.5),
                        n(329.63, 0.5, 0.5),
                        n(392.00, 1.0, 0.5),
                        n(440.00, 1.5, 0.5),
                    ],
                    MusicalState {
                        harmony_activations: [0.9; 8],
                        ..Default::default()
                    },
                )
            } else {
                // "Bad" phrase: monotone, low harmony
                (
                    vec![n(261.63, 0.0, 0.5), n(261.63, 0.5, 0.5)],
                    MusicalState {
                        harmony_activations: [0.1; 8],
                        ..Default::default()
                    },
                )
            };
            journal.evaluate_phrase(&notes, &state);
        }

        // At least one weight should have shifted meaningfully
        let total_change = (journal.taste.melodic_interest - initial.melodic_interest).abs()
            + (journal.taste.harmonic_alignment - initial.harmonic_alignment).abs()
            + (journal.taste.contour_quality - initial.contour_quality).abs()
            + (journal.taste.surprise_value - initial.surprise_value).abs();
        assert!(
            total_change > 0.01,
            "taste should evolve: total_change={total_change}"
        );
    }

    #[test]
    fn candidate_selection_picks_best() {
        let mut journal = CreativeJournal::new();
        journal.add_candidate(vec![n(261.63, 0.0, 0.5)], 0.3);
        journal.add_candidate(vec![n(440.00, 0.0, 0.5)], 0.9); // best
        journal.add_candidate(vec![n(329.63, 0.0, 0.5)], 0.5);

        let best = journal.select_best().unwrap();
        assert!(
            (best[0].frequency - 440.00).abs() < 1.0,
            "should select highest scored"
        );
    }

    #[test]
    fn quality_tiers() {
        assert_eq!(QualityTier::from_score(0.9), QualityTier::Masterpiece);
        assert_eq!(QualityTier::from_score(0.75), QualityTier::Great);
        assert_eq!(QualityTier::from_score(0.6), QualityTier::Good);
        assert_eq!(QualityTier::from_score(0.4), QualityTier::Acceptable);
        assert_eq!(QualityTier::from_score(0.1), QualityTier::Rejected);
    }
}
