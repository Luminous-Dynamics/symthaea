// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio fingerprinting for music identification

use crate::{AudioBuffer, AudioError, AudioResult};
use std::collections::HashMap;

/// Audio fingerprint
#[derive(Debug, Clone)]
pub struct Fingerprint {
    /// Unique hash of the audio
    pub hash: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Spectral landmarks
    pub landmarks: Vec<Landmark>,
}

/// Spectral landmark for fingerprinting
#[derive(Debug, Clone, Copy)]
pub struct Landmark {
    /// Time offset in samples
    pub time: u32,
    /// Frequency bin 1
    pub freq1: u16,
    /// Frequency bin 2
    pub freq2: u16,
    /// Time delta between peaks
    pub delta_time: u16,
}

impl Landmark {
    /// Generate hash from landmark
    pub fn hash(&self) -> u32 {
        (self.freq1 as u32) << 20 | (self.freq2 as u32) << 10 | self.delta_time as u32
    }
}

/// Fingerprint generator configuration
#[derive(Debug, Clone)]
pub struct FingerprintConfig {
    /// FFT size
    pub fft_size: usize,
    /// Hop size between FFT frames
    pub hop_size: usize,
    /// Number of peaks to extract per frame
    pub peaks_per_frame: usize,
    /// Fan-out factor for landmark pairs
    pub fan_out: usize,
    /// Maximum time delta for landmark pairs
    pub max_delta_time: usize,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            fft_size: 4096,
            hop_size: 2048,
            peaks_per_frame: 5,
            fan_out: 15,
            max_delta_time: 200,
        }
    }
}

/// Fingerprint generator
pub struct FingerprintGenerator {
    config: FingerprintConfig,
}

impl FingerprintGenerator {
    pub fn new(config: FingerprintConfig) -> Self {
        Self { config }
    }

    /// Generate fingerprint from audio buffer
    pub fn generate(&self, buffer: &AudioBuffer) -> AudioResult<Fingerprint> {
        // Convert to mono if needed
        let mono = if buffer.channels > 1 {
            buffer.to_mono()
        } else {
            buffer.clone()
        };

        // Extract spectral peaks
        let peaks = self.extract_peaks(&mono.samples, mono.sample_rate)?;

        // Generate landmarks from peak pairs
        let landmarks = self.generate_landmarks(&peaks)?;

        // Generate hash
        let hash = self.generate_hash(&landmarks);

        let duration_ms = (mono.samples.len() as f64 / mono.sample_rate as f64 * 1000.0) as u64;

        Ok(Fingerprint {
            hash,
            duration_ms,
            landmarks,
        })
    }

    fn extract_peaks(&self, samples: &[f32], sample_rate: u32) -> AudioResult<Vec<(u32, u16, f32)>> {
        let mut peaks = Vec::new();
        let num_frames = (samples.len() - self.config.fft_size) / self.config.hop_size;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.config.hop_size;
            let frame = &samples[start..start + self.config.fft_size];

            // Simple magnitude spectrum (would use FFT in production)
            let mut magnitudes: Vec<f32> = Vec::with_capacity(self.config.fft_size / 2);
            for k in 0..self.config.fft_size / 2 {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;
                for (n, &sample) in frame.iter().enumerate() {
                    let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / self.config.fft_size as f32;
                    real += sample * angle.cos();
                    imag += sample * angle.sin();
                }
                magnitudes.push((real * real + imag * imag).sqrt());
            }

            // Find peaks in this frame
            let mut frame_peaks: Vec<(u16, f32)> = Vec::new();
            for (i, &mag) in magnitudes.iter().enumerate().skip(1) {
                if i >= magnitudes.len() - 1 {
                    break;
                }
                if mag > magnitudes[i - 1] && mag > magnitudes[i + 1] && mag > 0.01 {
                    frame_peaks.push((i as u16, mag));
                }
            }

            // Sort by magnitude and take top N
            frame_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            frame_peaks.truncate(self.config.peaks_per_frame);

            let time = (frame_idx * self.config.hop_size) as u32;
            for (freq, mag) in frame_peaks {
                peaks.push((time, freq, mag));
            }
        }

        Ok(peaks)
    }

    fn generate_landmarks(&self, peaks: &[(u32, u16, f32)]) -> AudioResult<Vec<Landmark>> {
        let mut landmarks = Vec::new();

        for (i, &(time1, freq1, _)) in peaks.iter().enumerate() {
            let mut pairs = 0;
            for &(time2, freq2, _) in peaks.iter().skip(i + 1) {
                if pairs >= self.config.fan_out {
                    break;
                }

                let delta_time = time2.saturating_sub(time1) / self.config.hop_size as u32;
                if delta_time == 0 || delta_time > self.config.max_delta_time as u32 {
                    continue;
                }

                landmarks.push(Landmark {
                    time: time1,
                    freq1,
                    freq2,
                    delta_time: delta_time as u16,
                });

                pairs += 1;
            }
        }

        Ok(landmarks)
    }

    fn generate_hash(&self, landmarks: &[Landmark]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for landmark in landmarks {
            landmark.hash().hash(&mut hasher);
        }
        format!("{:016x}", hasher.finish())
    }
}

impl Default for FingerprintGenerator {
    fn default() -> Self {
        Self::new(FingerprintConfig::default())
    }
}

/// Fingerprint matcher for identification
pub struct FingerprintMatcher {
    /// Database of fingerprints (hash -> (track_id, time_offset))
    database: HashMap<u32, Vec<(String, u32)>>,
}

impl FingerprintMatcher {
    pub fn new() -> Self {
        Self {
            database: HashMap::new(),
        }
    }

    /// Add a fingerprint to the database
    pub fn add(&mut self, track_id: &str, fingerprint: &Fingerprint) {
        for landmark in &fingerprint.landmarks {
            let hash = landmark.hash();
            self.database
                .entry(hash)
                .or_default()
                .push((track_id.to_string(), landmark.time));
        }
    }

    /// Match a query fingerprint against the database
    pub fn match_fingerprint(&self, query: &Fingerprint) -> Vec<MatchResult> {
        let mut time_deltas: HashMap<String, HashMap<i32, u32>> = HashMap::new();

        for landmark in &query.landmarks {
            let hash = landmark.hash();
            if let Some(matches) = self.database.get(&hash) {
                for (track_id, db_time) in matches {
                    let delta = *db_time as i32 - landmark.time as i32;
                    *time_deltas
                        .entry(track_id.clone())
                        .or_default()
                        .entry(delta)
                        .or_insert(0) += 1;
                }
            }
        }

        let mut results: Vec<MatchResult> = time_deltas
            .into_iter()
            .map(|(track_id, deltas)| {
                let (best_delta, count) = deltas
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .unwrap_or((0, 0));

                MatchResult {
                    track_id,
                    confidence: count as f32 / query.landmarks.len() as f32,
                    time_offset: best_delta,
                    match_count: count,
                }
            })
            .collect();

        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        results
    }

    /// Clear the database
    pub fn clear(&mut self) {
        self.database.clear();
    }

    /// Get database size (number of hashes)
    pub fn size(&self) -> usize {
        self.database.len()
    }
}

impl Default for FingerprintMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a fingerprint match
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Matched track ID
    pub track_id: String,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Time offset in samples
    pub time_offset: i32,
    /// Number of matching landmarks
    pub match_count: u32,
}

/// Compare two fingerprints directly
pub fn compare_fingerprints(fp1: &Fingerprint, fp2: &Fingerprint) -> f32 {
    let hashes1: std::collections::HashSet<u32> = fp1.landmarks.iter().map(|l| l.hash()).collect();
    let hashes2: std::collections::HashSet<u32> = fp2.landmarks.iter().map(|l| l.hash()).collect();

    let intersection = hashes1.intersection(&hashes2).count();
    let union = hashes1.union(&hashes2).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChannelLayout;

    #[test]
    fn test_fingerprint_generation() {
        let samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 / 44100.0 * 440.0 * 2.0 * std::f32::consts::PI).sin())
            .collect();

        let buffer = AudioBuffer::from_samples(samples, 44100, 1, ChannelLayout::Mono);
        let generator = FingerprintGenerator::default();
        let fp = generator.generate(&buffer).unwrap();

        assert!(!fp.hash.is_empty());
        assert!(fp.duration_ms > 0);
    }

    #[test]
    fn test_landmark_hash() {
        let landmark = Landmark {
            time: 1000,
            freq1: 100,
            freq2: 200,
            delta_time: 50,
        };
        let hash = landmark.hash();
        assert!(hash > 0);
    }
}
