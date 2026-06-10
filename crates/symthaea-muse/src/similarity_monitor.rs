// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Self-similarity monitor: ensures the right amount of repetition.
//!
//! Music is ~70% repetition. Without enough, it sounds like random wandering.
//! With too much, it's boring. This module measures melodic self-similarity
//! in real-time and intervenes when the balance is wrong.
//!
//! Target similarity ratio: 0.5-0.7 (moderate repetition with development).

use crate::Note;
use std::collections::VecDeque;

/// Compact representation of a phrase for similarity comparison.
#[derive(Debug, Clone)]
struct PhraseVector {
    pitch_centroid: f32,
    pitch_range: f32,
    interval_histogram: [f32; 12], // chromatic interval counts, normalized
    mean_duration: f32,
    contour: f32, // net ascending (+) vs descending (-)
}

/// Action the similarity monitor recommends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityAction {
    NoAction,
    ForceRepeat,
    ForceNew,
}

/// Tracks phrase similarity and controls repetition balance.
pub struct SimilarityMonitor {
    history: VecDeque<PhraseVector>,
    max_history: usize,
    target_min: f32,
    target_max: f32,
    current_similarity: f32,
    similarity_ema: f32,
}

impl SimilarityMonitor {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(32),
            max_history: 32,
            target_min: 0.45,
            target_max: 0.75,
            current_similarity: 0.5,
            similarity_ema: 0.5,
        }
    }

    /// Record a completed phrase.
    pub fn record_phrase(&mut self, notes: &[Note]) {
        if notes.len() < 2 {
            return;
        }
        let vec = notes_to_vector(notes);

        // Compute similarity to history before adding
        self.current_similarity = self.max_similarity(&vec);
        self.similarity_ema = self.similarity_ema * 0.85 + self.current_similarity * 0.15;

        self.history.push_back(vec);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Compute max similarity of a phrase vector against history.
    fn max_similarity(&self, vec: &PhraseVector) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|h| cosine_similarity(vec, h))
            .fold(0.0f32, f32::max)
    }

    /// Recommend an action based on similarity balance.
    pub fn recommend(&self) -> SimilarityAction {
        if self.history.len() < 3 {
            return SimilarityAction::NoAction;
        }

        if self.similarity_ema < self.target_min {
            SimilarityAction::ForceRepeat
        } else if self.similarity_ema > self.target_max {
            SimilarityAction::ForceNew
        } else {
            SimilarityAction::NoAction
        }
    }

    pub fn current_similarity(&self) -> f32 {
        self.current_similarity
    }
    pub fn similarity_ema(&self) -> f32 {
        self.similarity_ema
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.current_similarity = 0.5;
        self.similarity_ema = 0.5;
    }
}

fn notes_to_vector(notes: &[Note]) -> PhraseVector {
    let centroid = notes.iter().map(|n| n.frequency).sum::<f32>() / notes.len() as f32;
    let min_f = notes.iter().map(|n| n.frequency).fold(f32::MAX, f32::min);
    let max_f = notes.iter().map(|n| n.frequency).fold(f32::MIN, f32::max);

    let mut histogram = [0.0f32; 12];
    for w in notes.windows(2) {
        let interval = ((w[1].frequency / w[0].frequency).log2() * 12.0)
            .round()
            .abs() as usize;
        let bin = interval.min(11);
        histogram[bin] += 1.0;
    }
    let hist_sum: f32 = histogram.iter().sum();
    if hist_sum > 0.0 {
        for h in &mut histogram {
            *h /= hist_sum;
        }
    }

    let mean_dur = notes.iter().map(|n| n.duration).sum::<f32>() / notes.len() as f32;

    let ascending = notes
        .windows(2)
        .filter(|w| w[1].frequency > w[0].frequency)
        .count();
    let descending = notes
        .windows(2)
        .filter(|w| w[1].frequency < w[0].frequency)
        .count();
    let contour = if notes.len() > 1 {
        (ascending as f32 - descending as f32) / (notes.len() - 1) as f32
    } else {
        0.0
    };

    PhraseVector {
        pitch_centroid: centroid,
        pitch_range: max_f - min_f,
        interval_histogram: histogram,
        mean_duration: mean_dur,
        contour,
    }
}

fn cosine_similarity(a: &PhraseVector, b: &PhraseVector) -> f32 {
    // Flatten to feature vector
    let fa = flatten(a);
    let fb = flatten(b);

    let dot: f32 = fa.iter().zip(fb.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = fa.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = fb.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a < 1e-8 || mag_b < 1e-8 {
        return 0.0;
    }
    (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
}

fn flatten(v: &PhraseVector) -> Vec<f32> {
    let mut f = Vec::with_capacity(16);
    f.push(v.pitch_centroid / 1000.0); // normalize
    f.push(v.pitch_range / 500.0);
    f.extend_from_slice(&v.interval_histogram);
    f.push(v.mean_duration);
    f.push(v.contour);
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(freq: f32, dur: f32) -> Note {
        Note {
            frequency: freq,
            start_time: 0.0,
            duration: dur,
            velocity: 0.7,
        }
    }

    #[test]
    fn identical_phrases_high_similarity() {
        let mut mon = SimilarityMonitor::new();
        let phrase = vec![n(261.63, 0.5), n(329.63, 0.5), n(392.00, 0.5)];
        mon.record_phrase(&phrase);
        mon.record_phrase(&phrase); // identical
        assert!(
            mon.current_similarity() > 0.9,
            "identical should be >0.9: {}",
            mon.current_similarity()
        );
    }

    #[test]
    fn different_phrases_low_similarity() {
        let mut mon = SimilarityMonitor::new();
        let phrase1 = vec![n(261.63, 0.5), n(329.63, 0.5), n(392.00, 0.5)]; // C-E-G ascending
        mon.record_phrase(&phrase1);
        let phrase2 = vec![n(880.0, 0.1), n(440.0, 0.1), n(220.0, 0.1)]; // very different
        mon.record_phrase(&phrase2);
        assert!(
            mon.current_similarity() < 0.5,
            "different should be low: {}",
            mon.current_similarity()
        );
    }

    #[test]
    fn force_repeat_when_too_novel() {
        let mut mon = SimilarityMonitor::new();
        mon.similarity_ema = 0.2; // very low
        // Add some history
        for i in 0..5 {
            mon.history.push_back(PhraseVector {
                pitch_centroid: 300.0 + i as f32 * 100.0,
                pitch_range: 200.0,
                interval_histogram: [0.0; 12],
                mean_duration: 0.5,
                contour: 0.0,
            });
        }
        assert_eq!(mon.recommend(), SimilarityAction::ForceRepeat);
    }

    #[test]
    fn force_new_when_too_repetitive() {
        let mut mon = SimilarityMonitor::new();
        mon.similarity_ema = 0.9; // very high
        for _ in 0..5 {
            mon.history.push_back(PhraseVector {
                pitch_centroid: 440.0,
                pitch_range: 100.0,
                interval_histogram: [0.0; 12],
                mean_duration: 0.5,
                contour: 0.0,
            });
        }
        assert_eq!(mon.recommend(), SimilarityAction::ForceNew);
    }
}
