// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Melody projection training via Evolution Strategy.
//!
//! Trains NeuralMelody's projection vectors on real MIDI melodies.
//! The CfC network architecture stays the same — only the input/output
//! mappings learn what "good melody" sounds like from actual music.
//!
//! # Training Loop
//!
//! ```text
//! For each epoch:
//!   1. Sample a target melody from the dataset
//!   2. Generate N perturbations of the projection vectors
//!   3. For each perturbation: generate melody → compute fitness vs target
//!   4. Update projections in the direction of highest fitness
//!   5. Repeat
//! ```
//!
//! # Fitness Function
//!
//! NOT pure similarity (that would produce copies). Instead:
//! - Pitch contour correlation (shape similarity, not exact pitch)
//! - Interval distribution match (same kinds of jumps between notes)
//! - Rhythm coherence (timing regularity relative to target)
//! - Penalize exact copies (novelty bonus for creative interpolation)

use crate::{
    MuseConfig, MusicalState, Note, midi_loader, neural_melody::NeuralMelody, pitch, rhythm,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of projection perturbations per epoch.
    pub population_size: usize,
    /// Gaussian noise standard deviation for perturbations.
    pub noise_sigma: f32,
    /// Learning rate for ES update.
    pub learning_rate: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Maximum melodies to load from dataset (0 = all).
    pub max_melodies: usize,
    /// Seed for reproducibility.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            population_size: 8,
            noise_sigma: 0.02,
            learning_rate: 0.005,
            epochs: 50,
            max_melodies: 200,
            seed: 42,
        }
    }
}

/// Result of a training run.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Fitness trajectory (mean fitness per epoch).
    pub fitness_trajectory: Vec<f32>,
    /// Best fitness achieved.
    pub best_fitness: f32,
    /// Trained projection vectors.
    pub projections: Vec<Vec<f32>>,
    /// Number of melodies used.
    pub melodies_used: usize,
    /// Epochs completed.
    pub epochs_completed: usize,
}

/// Train melody projections on MIDI data using Evolution Strategy.
///
/// Loads melodies from `midi_dir`, trains NeuralMelody projections
/// to produce melodies that resemble real music in contour and rhythm
/// (but not exact copies).
pub fn train_projections(
    midi_dir: &Path,
    config: &TrainingConfig,
    muse_config: &MuseConfig,
) -> TrainingResult {
    // Load melodies
    let raw_melodies = midi_loader::load_melodies(midi_dir);

    if raw_melodies.is_empty() {
        return TrainingResult {
            fitness_trajectory: vec![],
            best_fitness: 0.0,
            projections: vec![],
            melodies_used: 0,
            epochs_completed: 0,
        };
    }

    // Extract short phrases (8-24 notes) from long performances.
    // Training on 3000-note piano pieces vs 16-note output is a mismatch.
    // Phrase segmentation: split on large time gaps or every N notes.
    let phrase_len = muse_config.max_notes.max(8);
    let mut melodies: Vec<Vec<Note>> = Vec::new();
    for melody in &raw_melodies {
        if melody.len() < 4 {
            continue;
        }
        // Split into overlapping phrases
        for chunk in melody.windows(phrase_len).step_by(phrase_len / 2) {
            if chunk.len() >= 4 {
                // Re-zero start times relative to phrase start
                let offset = chunk[0].start_time;
                let phrase: Vec<Note> = chunk
                    .iter()
                    .map(|n| Note {
                        frequency: n.frequency,
                        start_time: n.start_time - offset,
                        duration: n.duration,
                        velocity: n.velocity,
                    })
                    .collect();
                melodies.push(phrase);
            }
        }
    }

    if config.max_melodies > 0 && melodies.len() > config.max_melodies {
        // Shuffle deterministically then truncate
        let mut rng_shuffle = StdRng::seed_from_u64(config.seed + 1);
        use rand::seq::SliceRandom;
        melodies.shuffle(&mut rng_shuffle);
        melodies.truncate(config.max_melodies);
    }
    let melodies_used = melodies.len();

    // Initialize projections from genesis seed
    let genesis = GenesisSeed::from_phrase(&format!("melody-training-{}", config.seed));
    let num_projections = 2; // [decode_weights(3), decode_biases(3)]

    // Initialize: weights = [3.0, 2.0, 2.0], biases = [0.0, 0.0, 0.0]
    let mut projections: Vec<ContinuousHV> = vec![
        ContinuousHV::from_values(vec![3.0, 2.0, 2.0]), // weights
        ContinuousHV::from_values(vec![0.0, 0.0, 0.0]), // biases
    ];

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut fitness_trajectory = Vec::with_capacity(config.epochs);
    let mut best_fitness = 0.0f32;

    let state = MusicalState::default();
    let scale = pitch::build_scale(&state);
    let tempo = rhythm::compute_tempo(muse_config, &state);
    let beat_duration = 60.0 / tempo;

    for epoch in 0..config.epochs {
        // Sample a random target melody
        let target_idx = rng.gen_range(0..melodies.len());
        let target = &melodies[target_idx];

        // Generate perturbations and evaluate fitness
        let mut fitnesses = Vec::with_capacity(config.population_size);
        let mut perturbations: Vec<Vec<ContinuousHV>> = Vec::with_capacity(config.population_size);

        for _ in 0..config.population_size {
            // Perturb each projection vector
            let perturbed: Vec<ContinuousHV> = projections
                .iter()
                .map(|proj| {
                    let noise: Vec<f32> = (0..proj.values.len())
                        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
                        .map(|n| n * config.noise_sigma)
                        .collect();
                    let mut p = proj.clone();
                    for (v, n) in p.values.iter_mut().zip(noise.iter()) {
                        *v += n;
                    }
                    p
                })
                .collect();

            // Generate melody with perturbed projections
            let mut neural = NeuralMelody::new(&genesis, muse_config);
            neural.set_projections(perturbed.clone());
            let generated = neural.generate(
                &MuseConfig {
                    max_notes: target.len().min(muse_config.max_notes),
                    duration_secs: target
                        .last()
                        .map(|n| n.start_time + n.duration + 1.0)
                        .unwrap_or(4.0),
                    ..muse_config.clone()
                },
                &state,
                &scale,
                beat_duration,
            );

            let fitness = compute_fitness(&generated, target);
            fitnesses.push(fitness);
            perturbations.push(perturbed);
        }

        // ES update: weighted average of perturbations by fitness
        let mean_fitness: f32 = fitnesses.iter().sum::<f32>() / fitnesses.len() as f32;
        let std_fitness: f32 = {
            let var: f32 = fitnesses
                .iter()
                .map(|f| (f - mean_fitness).powi(2))
                .sum::<f32>()
                / fitnesses.len() as f32;
            var.sqrt().max(0.001)
        };

        // Normalize fitnesses to advantages
        let advantages: Vec<f32> = fitnesses
            .iter()
            .map(|f| (f - mean_fitness) / std_fitness)
            .collect();

        // Update projections toward high-fitness perturbations
        for proj_idx in 0..num_projections {
            for dim_idx in 0..projections[proj_idx].values.len() {
                let mut gradient = 0.0f32;
                for (pop_idx, advantage) in advantages.iter().enumerate() {
                    let delta = perturbations[pop_idx][proj_idx].values[dim_idx]
                        - projections[proj_idx].values[dim_idx];
                    gradient += advantage * delta;
                }
                gradient /= (config.population_size as f32) * config.noise_sigma;
                projections[proj_idx].values[dim_idx] += config.learning_rate * gradient;
            }
        }

        fitness_trajectory.push(mean_fitness);
        if mean_fitness > best_fitness {
            best_fitness = mean_fitness;
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            eprintln!(
                "  Epoch {:3}/{}: fitness={:.4} best={:.4} (target: {} notes)",
                epoch + 1,
                config.epochs,
                mean_fitness,
                best_fitness,
                target.len()
            );
        }
    }

    // Extract projection values for serialization
    let proj_values: Vec<Vec<f32>> = projections.iter().map(|p| p.values.clone()).collect();

    TrainingResult {
        fitness_trajectory,
        best_fitness,
        projections: proj_values,
        melodies_used,
        epochs_completed: config.epochs,
    }
}

/// Compute fitness of a generated melody against a target.
///
/// Measures similarity in shape (contour), interval patterns, and rhythm
/// while penalizing exact copies.
fn compute_fitness(generated: &[Note], target: &[Note]) -> f32 {
    if generated.is_empty() || target.is_empty() {
        return 0.0;
    }

    // 1. Pitch contour correlation (shape, not exact pitch)
    let contour_score = pitch_contour_correlation(generated, target);

    // 2. Interval distribution similarity
    let interval_score = interval_distribution_similarity(generated, target);

    // 3. Rhythm coherence
    let rhythm_score = rhythm_coherence(generated, target);

    // 4. Novelty bonus: penalize exact pitch copies
    let novelty = novelty_bonus(generated, target);

    // Weighted composite
    (0.35 * contour_score + 0.25 * interval_score + 0.25 * rhythm_score + 0.15 * novelty)
        .clamp(0.0, 1.0)
}

/// Pitch contour correlation: do the melodies rise and fall similarly?
fn pitch_contour_correlation(a: &[Note], b: &[Note]) -> f32 {
    let len = a.len().min(b.len());
    if len < 2 {
        return 0.5;
    }

    // Extract contour direction sequences
    let contour_a: Vec<f32> = a
        .windows(2)
        .map(|w| {
            if w[1].frequency > w[0].frequency {
                1.0
            } else if w[1].frequency < w[0].frequency {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    let contour_b: Vec<f32> = b
        .windows(2)
        .map(|w| {
            if w[1].frequency > w[0].frequency {
                1.0
            } else if w[1].frequency < w[0].frequency {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    let min_len = contour_a.len().min(contour_b.len());
    if min_len == 0 {
        return 0.5;
    }

    let matches: f32 = contour_a[..min_len]
        .iter()
        .zip(contour_b[..min_len].iter())
        .map(|(a, b)| if (a - b).abs() < 0.5 { 1.0 } else { 0.0 })
        .sum();

    matches / min_len as f32
}

/// Interval distribution similarity: same kinds of jumps between notes?
fn interval_distribution_similarity(a: &[Note], b: &[Note]) -> f32 {
    if a.len() < 2 || b.len() < 2 {
        return 0.5;
    }

    // Quantize intervals to semitone buckets
    let intervals_a = extract_intervals(a);
    let intervals_b = extract_intervals(b);

    // Compare distributions via histogram overlap
    histogram_overlap(&intervals_a, &intervals_b)
}

fn extract_intervals(notes: &[Note]) -> Vec<i32> {
    notes
        .windows(2)
        .map(|w| {
            let semitones = 12.0 * (w[1].frequency / w[0].frequency.max(0.01)).log2();
            semitones.round() as i32
        })
        .collect()
}

fn histogram_overlap(a: &[i32], b: &[i32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Build frequency histograms (buckets: -12 to +12 semitones)
    let mut hist_a = [0u32; 25]; // index 0 = -12, index 12 = 0, index 24 = +12
    let mut hist_b = [0u32; 25];

    for &interval in a {
        let idx = (interval + 12).clamp(0, 24) as usize;
        hist_a[idx] += 1;
    }
    for &interval in b {
        let idx = (interval + 12).clamp(0, 24) as usize;
        hist_b[idx] += 1;
    }

    // Normalize
    let sum_a: f32 = hist_a.iter().sum::<u32>() as f32;
    let sum_b: f32 = hist_b.iter().sum::<u32>() as f32;
    if sum_a < 1.0 || sum_b < 1.0 {
        return 0.0;
    }

    // Overlap coefficient
    let overlap: f32 = hist_a
        .iter()
        .zip(hist_b.iter())
        .map(|(&a, &b)| (a as f32 / sum_a).min(b as f32 / sum_b))
        .sum();

    overlap.clamp(0.0, 1.0)
}

/// Rhythm coherence: similar timing patterns?
fn rhythm_coherence(a: &[Note], b: &[Note]) -> f32 {
    if a.len() < 2 || b.len() < 2 {
        return 0.5;
    }

    // Compare inter-onset interval distributions
    let ioi_a: Vec<f32> = a
        .windows(2)
        .map(|w| w[1].start_time - w[0].start_time)
        .collect();
    let ioi_b: Vec<f32> = b
        .windows(2)
        .map(|w| w[1].start_time - w[0].start_time)
        .collect();

    // Quantize to beat fractions and compare
    let mean_a = ioi_a.iter().sum::<f32>() / ioi_a.len() as f32;
    let mean_b = ioi_b.iter().sum::<f32>() / ioi_b.len() as f32;

    if mean_a < 0.001 || mean_b < 0.001 {
        return 0.5;
    }

    // Normalize by mean (relative rhythm)
    let rel_a: Vec<f32> = ioi_a.iter().map(|t| t / mean_a).collect();
    let rel_b: Vec<f32> = ioi_b.iter().map(|t| t / mean_b).collect();

    // Compare: correlation of normalized rhythms
    let min_len = rel_a.len().min(rel_b.len());
    if min_len == 0 {
        return 0.5;
    }

    let diff: f32 = rel_a[..min_len]
        .iter()
        .zip(rel_b[..min_len].iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / min_len as f32;

    (1.0 - diff).clamp(0.0, 1.0)
}

/// Novelty bonus: reward creative interpolation, penalize exact copies.
fn novelty_bonus(generated: &[Note], target: &[Note]) -> f32 {
    if generated.is_empty() || target.is_empty() {
        return 0.5;
    }

    let len = generated.len().min(target.len());
    let exact_matches: usize = generated[..len]
        .iter()
        .zip(target[..len].iter())
        .filter(|(g, t)| (g.frequency - t.frequency).abs() < 1.0)
        .count();

    let copy_ratio = exact_matches as f32 / len as f32;

    // Sweet spot: some similarity (0.2-0.5) is good, too much (>0.8) is copying
    if copy_ratio < 0.2 {
        0.3 // Too different — not learning from the data
    } else if copy_ratio > 0.8 {
        0.2 // Too similar — just copying
    } else {
        // Sweet spot: creative interpolation
        1.0 - (copy_ratio - 0.35).abs() * 2.0
    }
}

/// Save trained projections to a JSON file.
pub fn save_projections(projections: &[Vec<f32>], path: &Path) -> std::io::Result<()> {
    // Simple JSON format: array of arrays
    let mut json = String::from("[");
    for (i, proj) in projections.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push('[');
        for (j, &v) in proj.iter().enumerate() {
            if j > 0 {
                json.push(',');
            }
            json.push_str(&format!("{v}"));
        }
        json.push(']');
    }
    json.push(']');
    std::fs::write(path, json)
}

/// Load trained projections from a JSON file.
pub fn load_projections(path: &Path) -> std::io::Result<Vec<Vec<f32>>> {
    let data = std::fs::read_to_string(path)?;
    // Simple parser: strip brackets, split by ],[
    let inner = data.trim().trim_start_matches('[').trim_end_matches(']');
    let mut result = Vec::new();
    for chunk in inner.split("],[") {
        let clean = chunk.trim_matches('[').trim_matches(']');
        let values: Vec<f32> = clean
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if !values.is_empty() {
            result.push(values);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_notes(freqs: &[f32]) -> Vec<Note> {
        freqs
            .iter()
            .enumerate()
            .map(|(i, &f)| Note {
                frequency: f,
                start_time: i as f32 * 0.5,
                duration: 0.4,
                velocity: 0.8,
            })
            .collect()
    }

    #[test]
    fn contour_identical_is_perfect() {
        let a = make_notes(&[261.0, 330.0, 392.0, 330.0, 261.0]);
        let score = pitch_contour_correlation(&a, &a);
        assert!((score - 1.0).abs() < 0.01, "identical contour: {score}");
    }

    #[test]
    fn contour_opposite_is_low() {
        let rising = make_notes(&[200.0, 300.0, 400.0, 500.0]);
        let falling = make_notes(&[500.0, 400.0, 300.0, 200.0]);
        let score = pitch_contour_correlation(&rising, &falling);
        assert!(score < 0.3, "opposite contour should be low: {score}");
    }

    #[test]
    fn interval_same_distribution() {
        let a = make_notes(&[261.0, 293.0, 329.0, 349.0, 392.0]);
        let score = interval_distribution_similarity(&a, &a);
        assert!(score > 0.9, "same intervals: {score}");
    }

    #[test]
    fn rhythm_identical_is_perfect() {
        let a = make_notes(&[261.0, 330.0, 392.0, 440.0]);
        let score = rhythm_coherence(&a, &a);
        assert!((score - 1.0).abs() < 0.01, "identical rhythm: {score}");
    }

    #[test]
    fn novelty_penalizes_copies() {
        let a = make_notes(&[261.0, 330.0, 392.0, 440.0]);
        let exact_copy = novelty_bonus(&a, &a);
        let different = make_notes(&[500.0, 600.0, 700.0, 800.0]);
        let novel = novelty_bonus(&different, &a);
        assert!(
            exact_copy < 0.5,
            "exact copy should be penalized: {exact_copy}"
        );
        // The relative property is the one this test is named for: an exact
        // copy must score strictly lower than genuinely different material.
        // The absolute bound above cannot catch a novelty_bonus that
        // penalizes everything equally.
        assert!(
            exact_copy < novel,
            "exact copy should score below novel material: \
             copy={exact_copy}, novel={novel}"
        );
    }

    #[test]
    fn fitness_bounded() {
        let a = make_notes(&[261.0, 330.0, 392.0, 440.0, 330.0]);
        let b = make_notes(&[293.0, 349.0, 440.0, 392.0, 293.0]);
        let f = compute_fitness(&a, &b);
        assert!(f >= 0.0 && f <= 1.0, "fitness: {f}");
    }

    #[test]
    fn fitness_empty_is_zero() {
        assert_eq!(compute_fitness(&[], &[]), 0.0);
    }

    #[test]
    fn train_empty_dir() {
        let config = TrainingConfig {
            epochs: 1,
            max_melodies: 10,
            ..Default::default()
        };
        let muse_config = MuseConfig::default();
        let result = train_projections(Path::new("/nonexistent"), &config, &muse_config);
        assert_eq!(result.melodies_used, 0);
    }

    #[test]
    fn histogram_overlap_identical() {
        let a = vec![0, 2, 4, -1, 3];
        let overlap = histogram_overlap(&a, &a);
        assert!((overlap - 1.0).abs() < 0.01, "identical: {overlap}");
    }

    #[test]
    fn save_load_projections_roundtrip() {
        let projections = vec![vec![0.1f32, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let dir = std::env::temp_dir().join("symthaea_proj_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test_proj.bin");

        save_projections(&projections, &path).expect("save");
        let loaded = load_projections(&path).expect("load");

        assert_eq!(projections.len(), loaded.len());
        for (orig, load) in projections.iter().zip(loaded.iter()) {
            for (a, b) in orig.iter().zip(load.iter()) {
                assert!((a - b).abs() < 0.001);
            }
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
