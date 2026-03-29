// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Beat tracking and tempo estimation

use crate::{Beat, Result};
use std::f32::consts::PI;

/// Beat tracker using onset detection and autocorrelation
pub struct BeatTracker {
    sample_rate: u32,
    hop_size: usize,
    frame_size: usize,
    min_tempo: f32,
    max_tempo: f32,
}

impl BeatTracker {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            hop_size: 512,
            frame_size: 2048,
            min_tempo: 60.0,
            max_tempo: 200.0,
        }
    }

    pub fn with_tempo_range(mut self, min: f32, max: f32) -> Self {
        self.min_tempo = min;
        self.max_tempo = max;
        self
    }

    /// Estimate tempo in BPM
    pub fn estimate_tempo(&self, audio: &[f32]) -> f32 {
        // Compute onset detection function
        let onsets = self.compute_onset_function(audio);

        // Autocorrelation for periodicity detection
        let tempo = self.autocorrelation_tempo(&onsets);

        tempo.clamp(self.min_tempo, self.max_tempo)
    }

    /// Detect beat positions
    pub fn detect_beats(&self, audio: &[f32]) -> Vec<Beat> {
        let onsets = self.compute_onset_function(audio);
        let tempo = self.autocorrelation_tempo(&onsets);
        let beat_period = 60.0 / tempo * self.sample_rate as f32 / self.hop_size as f32;

        // Find beat positions using dynamic programming
        let beats = self.track_beats(&onsets, beat_period);

        // Convert to Beat structs
        let mut result = Vec::new();
        let mut beat_in_bar = 0u8;

        for (i, &onset_idx) in beats.iter().enumerate() {
            let time = onset_idx as f32 * self.hop_size as f32 / self.sample_rate as f32;
            let strength = if onset_idx < onsets.len() {
                onsets[onset_idx]
            } else {
                0.5
            };

            beat_in_bar = (beat_in_bar % 4) + 1;

            result.push(Beat {
                time,
                strength: strength.clamp(0.0, 1.0),
                position: beat_in_bar,
            });
        }

        result
    }

    fn compute_onset_function(&self, audio: &[f32]) -> Vec<f32> {
        let num_frames = (audio.len() - self.frame_size) / self.hop_size + 1;
        let mut onsets = vec![0.0f32; num_frames];
        let mut prev_spectrum = vec![0.0f32; self.frame_size / 2];

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let end = start + self.frame_size;

            if end > audio.len() {
                break;
            }

            let frame = &audio[start..end];

            // Compute magnitude spectrum
            let spectrum = self.compute_magnitude_spectrum(frame);

            // Spectral flux (sum of positive differences)
            let mut flux = 0.0;
            for i in 0..spectrum.len().min(prev_spectrum.len()) {
                let diff = spectrum[i] - prev_spectrum[i];
                if diff > 0.0 {
                    flux += diff;
                }
            }

            onsets[frame_idx] = flux;
            prev_spectrum = spectrum;
        }

        // Normalize
        let max_onset = onsets.iter().cloned().fold(0.0f32, f32::max);
        if max_onset > 0.0 {
            for onset in &mut onsets {
                *onset /= max_onset;
            }
        }

        // Apply adaptive threshold
        self.apply_adaptive_threshold(&mut onsets);

        onsets
    }

    fn compute_magnitude_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        let mut spectrum = vec![0.0f32; n / 2];

        // Simple DFT (would use FFT in production)
        for k in 0..n / 2 {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in frame.iter().enumerate() {
                let angle = 2.0 * PI * k as f32 * i as f32 / n as f32;
                let window = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos(); // Hann
                let windowed = sample * window;
                real += windowed * angle.cos();
                imag -= windowed * angle.sin();
            }

            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        spectrum
    }

    fn apply_adaptive_threshold(&self, onsets: &mut [f32]) {
        let window_size = 10;

        // Compute local mean
        let mut thresholds = vec![0.0f32; onsets.len()];
        for i in 0..onsets.len() {
            let start = i.saturating_sub(window_size);
            let end = (i + window_size + 1).min(onsets.len());
            let mean: f32 = onsets[start..end].iter().sum::<f32>() / (end - start) as f32;
            thresholds[i] = mean + 0.1;
        }

        // Apply threshold
        for (i, onset) in onsets.iter_mut().enumerate() {
            if *onset < thresholds[i] {
                *onset = 0.0;
            }
        }
    }

    fn autocorrelation_tempo(&self, onsets: &[f32]) -> f32 {
        let min_lag = (60.0 / self.max_tempo * self.sample_rate as f32 / self.hop_size as f32) as usize;
        let max_lag = (60.0 / self.min_tempo * self.sample_rate as f32 / self.hop_size as f32) as usize;

        let mut best_lag = min_lag;
        let mut best_correlation = 0.0f32;

        // Compute autocorrelation at tempo-relevant lags
        for lag in min_lag..=max_lag.min(onsets.len() / 2) {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..onsets.len() - lag {
                correlation += onsets[i] * onsets[i + lag];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f32;

                // Weight towards common tempos
                let tempo = 60.0 / (lag as f32 * self.hop_size as f32 / self.sample_rate as f32);
                let weight = tempo_prior(tempo);
                correlation *= weight;

                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_lag = lag;
                }
            }
        }

        // Convert lag to tempo
        60.0 / (best_lag as f32 * self.hop_size as f32 / self.sample_rate as f32)
    }

    fn track_beats(&self, onsets: &[f32], beat_period: f32) -> Vec<usize> {
        if onsets.is_empty() {
            return vec![];
        }

        // Dynamic programming beat tracker
        let period = beat_period.round() as usize;
        let tolerance = (period as f32 * 0.2) as usize;

        // Cost function: penalize deviation from expected beat period
        let mut beats = Vec::new();
        let mut last_beat = 0usize;

        // Find strong onset near start
        for (i, &onset) in onsets.iter().take(period * 2).enumerate() {
            if onset > 0.3 {
                beats.push(i);
                last_beat = i;
                break;
            }
        }

        if beats.is_empty() {
            beats.push(0);
        }

        // Track subsequent beats
        while last_beat + period < onsets.len() {
            let expected = last_beat + period;
            let search_start = expected.saturating_sub(tolerance);
            let search_end = (expected + tolerance).min(onsets.len() - 1);

            // Find best beat in search window
            let mut best_idx = expected;
            let mut best_score = -1.0f32;

            for i in search_start..=search_end {
                let onset_score = onsets[i];
                let timing_score = 1.0 - (i as f32 - expected as f32).abs() / tolerance as f32;
                let score = onset_score * 0.7 + timing_score * 0.3;

                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }

            beats.push(best_idx);
            last_beat = best_idx;
        }

        beats
    }
}

/// Prior probability for tempo (favor common tempos)
fn tempo_prior(tempo: f32) -> f32 {
    // Most music is between 80-160 BPM
    let center = 120.0;
    let spread = 40.0;
    let x = (tempo - center) / spread;
    (-0.5 * x * x).exp()
}

/// Downbeat tracker (find bar boundaries)
pub struct DownbeatTracker {
    sample_rate: u32,
    beats_per_bar: u8,
}

impl DownbeatTracker {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            beats_per_bar: 4,
        }
    }

    pub fn with_time_signature(mut self, beats_per_bar: u8) -> Self {
        self.beats_per_bar = beats_per_bar;
        self
    }

    /// Find downbeat positions from beat list
    pub fn find_downbeats(&self, beats: &[Beat]) -> Vec<usize> {
        if beats.is_empty() {
            return vec![];
        }

        let mut downbeats = Vec::new();

        // Analyze beat strengths to find periodic pattern
        let strengths: Vec<f32> = beats.iter().map(|b| b.strength).collect();

        // Find strongest beat in each potential bar
        for start in (0..beats.len()).step_by(self.beats_per_bar as usize) {
            let end = (start + self.beats_per_bar as usize).min(beats.len());
            let window = &strengths[start..end];

            let max_idx = window
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            downbeats.push(start + max_idx);
        }

        downbeats
    }
}

/// Time signature detector
pub struct TimeSignatureDetector {
    sample_rate: u32,
}

impl TimeSignatureDetector {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Detect likely time signature
    pub fn detect(&self, beats: &[Beat]) -> (u8, u8) {
        if beats.len() < 8 {
            return (4, 4); // Default
        }

        // Analyze beat strength patterns
        let strengths: Vec<f32> = beats.iter().map(|b| b.strength).collect();

        // Test different groupings
        let mut best_grouping = 4;
        let mut best_score = 0.0f32;

        for grouping in [3, 4, 5, 6, 7] {
            let score = self.test_grouping(&strengths, grouping);
            if score > best_score {
                best_score = score;
                best_grouping = grouping;
            }
        }

        // Beat unit is typically quarter note
        (best_grouping, 4)
    }

    fn test_grouping(&self, strengths: &[f32], grouping: u8) -> f32 {
        let g = grouping as usize;
        if strengths.len() < g * 2 {
            return 0.0;
        }

        // Compute average strength at each position in group
        let mut position_strengths = vec![0.0f32; g];
        let mut counts = vec![0usize; g];

        for (i, &s) in strengths.iter().enumerate() {
            let pos = i % g;
            position_strengths[pos] += s;
            counts[pos] += 1;
        }

        for (i, ps) in position_strengths.iter_mut().enumerate() {
            if counts[i] > 0 {
                *ps /= counts[i] as f32;
            }
        }

        // Score based on first beat being strongest
        let first_strength = position_strengths[0];
        let avg_other: f32 = position_strengths[1..].iter().sum::<f32>() / (g - 1) as f32;

        if avg_other > 0.0 {
            first_strength / avg_other
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tempo_prior() {
        // 120 BPM should have highest prior
        let p120 = tempo_prior(120.0);
        let p180 = tempo_prior(180.0);
        let p60 = tempo_prior(60.0);
        assert!(p120 > p180);
        assert!(p120 > p60);
    }

    #[test]
    fn test_beat_tracker() {
        let tracker = BeatTracker::new(44100);
        // Generate simple click track at 120 BPM
        let sample_rate = 44100;
        let duration = 5.0;
        let beat_interval = sample_rate as f32 * 60.0 / 120.0;

        let mut audio = vec![0.0f32; (sample_rate as f32 * duration) as usize];
        let mut t = 0.0;
        while t < audio.len() as f32 {
            let idx = t as usize;
            if idx < audio.len() {
                // Simple click
                for i in 0..100 {
                    if idx + i < audio.len() {
                        audio[idx + i] = 0.5 * (-0.1 * i as f32).exp();
                    }
                }
            }
            t += beat_interval;
        }

        let tempo = tracker.estimate_tempo(&audio);
        // Should be close to 120 BPM
        assert!((tempo - 120.0).abs() < 10.0);
    }
}
