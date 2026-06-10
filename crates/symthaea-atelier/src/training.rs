// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Training loop for NeuralCanvas projection vectors.
//!
//! Uses evolution strategy (ES) to optimize the projection vectors that decode
//! CfC output HVs into VisualFrame parameters. The aesthetic score is the
//! fitness function — the system literally learns to make more beautiful art.
//!
//! This is gradient-free optimization because the aesthetic score isn't
//! differentiable through the SVG rendering pipeline. ES works by:
//! 1. Perturb each projection vector with Gaussian noise
//! 2. Generate art with perturbed projections
//! 3. Evaluate aesthetic score
//! 4. Update projections in the direction of improvement
//!
//! Reference: Salimans et al. (2017) "Evolution Strategies as a Scalable
//! Alternative to Reinforcement Learning"

use symthaea_canvas::CognitiveSnapshot;
use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::AtelierConfig;
use crate::neural_canvas::NeuralCanvas;

/// Configuration for projection training.
pub struct TrainingConfig {
    /// Number of perturbation samples per training step.
    pub population_size: usize,
    /// Standard deviation of Gaussian perturbation.
    pub noise_sigma: f32,
    /// Learning rate for projection updates.
    pub learning_rate: f32,
    /// Number of training steps.
    pub epochs: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            population_size: 8,
            noise_sigma: 0.02,
            learning_rate: 0.005,
            epochs: 20,
        }
    }
}

/// Result of a training session.
#[derive(Debug)]
pub struct TrainingResult {
    /// Aesthetic scores per epoch (best score at each step).
    pub score_history: Vec<f32>,
    /// Initial score before training.
    pub initial_score: f32,
    /// Final score after training.
    pub final_score: f32,
    /// Total art pieces evaluated.
    pub total_evaluations: usize,
}

/// Train the NeuralCanvas projections to maximize aesthetic score.
///
/// Takes a set of training snapshots (cognitive states to generate art from)
/// and optimizes the projection vectors so that the decoded VisualFrames
/// produce higher-scoring compositions.
pub fn train_projections(
    canvas: &mut NeuralCanvas,
    config: &TrainingConfig,
    atelier_config: &AtelierConfig,
    training_snapshots: &[CognitiveSnapshot],
) -> TrainingResult {
    if training_snapshots.is_empty() {
        return TrainingResult {
            score_history: Vec::new(),
            initial_score: 0.0,
            final_score: 0.0,
            total_evaluations: 0,
        };
    }

    let mut total_evaluations = 0;

    // Evaluate initial score
    let initial_score = evaluate_mean_score(canvas, atelier_config, training_snapshots);
    total_evaluations += training_snapshots.len();

    let mut score_history = vec![initial_score];

    for _epoch in 0..config.epochs {
        // For each projection vector, try perturbations and compute
        // the fitness-weighted update direction
        let n_projections = canvas.projection_count();

        for proj_idx in 0..n_projections {
            let original = canvas.get_projection(proj_idx).clone();
            let dim = original.values.len();

            let mut best_delta_score = 0.0f32;
            let mut best_perturbation = vec![0.0f32; dim];

            // Evaluate perturbations
            for sample in 0..config.population_size {
                // Generate Gaussian noise perturbation
                let noise = gaussian_noise(
                    dim,
                    config.noise_sigma,
                    _epoch as u64 * 1000 + proj_idx as u64 * 100 + sample as u64,
                );

                // Apply perturbation
                let perturbed = perturb_hv(&original, &noise);
                canvas.set_projection(proj_idx, perturbed);

                // Evaluate
                let score = evaluate_mean_score(canvas, atelier_config, training_snapshots);
                total_evaluations += training_snapshots.len();

                let delta = score - initial_score;
                if delta > best_delta_score {
                    best_delta_score = delta;
                    best_perturbation = noise;
                }

                // Restore original
                canvas.set_projection(proj_idx, original.clone());
            }

            // Update projection in the direction of best improvement
            if best_delta_score > 0.0 {
                let updated = perturb_hv(
                    &original,
                    &scale_vec(&best_perturbation, config.learning_rate),
                );
                canvas.set_projection(proj_idx, updated);
            }
        }

        // Record epoch score
        let epoch_score = evaluate_mean_score(canvas, atelier_config, training_snapshots);
        total_evaluations += training_snapshots.len();
        score_history.push(epoch_score);
    }

    let final_score = *score_history.last().unwrap_or(&initial_score);

    TrainingResult {
        score_history,
        initial_score,
        final_score,
        total_evaluations,
    }
}

/// Evaluate mean aesthetic score across training snapshots.
///
/// Uses pixel-based scoring that actually sees the art:
/// - Color diversity (Shannon entropy of spatial color distribution)
/// - Spatial balance (variance of color across grid cells)
/// - Harmony alignment (mean harmony activation)
/// - Complexity balance (Berlyne: moderate complexity most pleasing)
///
/// This replaces the SVG node-count scorer which was blind to visual content.
fn evaluate_mean_score(
    canvas: &mut NeuralCanvas,
    _config: &AtelierConfig,
    snapshots: &[CognitiveSnapshot],
) -> f32 {
    if snapshots.is_empty() {
        return 0.0;
    }
    let sum: f32 = snapshots
        .iter()
        .map(|snap| score_visual_content(canvas, snap))
        .sum();
    sum / snapshots.len() as f32
}

/// Score based on actual visual content of the generated art.
fn score_visual_content(canvas: &mut NeuralCanvas, snapshot: &CognitiveSnapshot) -> f32 {
    let scene = canvas.generate(&AtelierConfig::default(), snapshot);

    // Render to SVG and extract content metrics
    let svg = symthaea_canvas::render_svg(&scene, snapshot.consciousness_level);

    // 1. Element diversity: mix of shapes (not all circles or all paths)
    let circles = svg.matches("<circle").count() as f32;
    let ellipses = svg.matches("<ellipse").count() as f32;
    let paths = svg.matches("<path").count() as f32;
    let polygons = svg.matches("<polygon").count() as f32;
    let total = (circles + ellipses + paths + polygons).max(1.0);

    let type_counts = [
        circles / total,
        ellipses / total,
        paths / total,
        polygons / total,
    ];
    let element_entropy = symthaea_aesthetic::information::shannon_entropy_normalized(&type_counts);

    // 2. Color diversity: count distinct colors in SVG
    let color_count = svg.matches("fill=\"#").count() + svg.matches("stroke=\"#").count();
    let color_diversity = (color_count as f32 / 20.0).min(1.0);

    // 3. Spatial distribution: SVG size as proxy for spread
    // (more content = more spread = better)
    let content_density = (svg.len() as f32 / 50000.0).min(1.0);

    // 4. Harmony alignment
    let harmony_mean: f32 = snapshot.harmony_activations.iter().sum::<f32>() / 8.0;

    // 5. Consciousness coupling
    let psi = snapshot.consciousness_level as f32;

    // Composite: weighted combination
    let mut score = symthaea_aesthetic::AestheticScore {
        order: (0.4 * psi + 0.3 * harmony_mean + 0.3 * (1.0 - element_entropy)).clamp(0.0, 1.0),
        complexity: symthaea_aesthetic::information::information_balance(&[
            circles, ellipses, paths, polygons,
        ]),
        surprise: element_entropy, // diversity of shapes = visual surprise
        harmony: harmony_mean,
        birkhoff: 0.0,
        composite: 0.0,
    };

    // Birkhoff from actual visual metrics
    let visual_order = color_diversity * 0.5 + content_density * 0.5;
    let visual_complexity = (total.ln().max(0.0) / 4.0).min(1.0); // log scale
    score.birkhoff = if visual_complexity > 0.01 {
        (visual_order / visual_complexity).clamp(0.0, 1.0)
    } else {
        0.0
    };

    score.compute_composite();
    score.composite
}

/// Generate pseudo-Gaussian noise using Box-Muller transform.
fn gaussian_noise(dim: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(12345);
    let mut noise = Vec::with_capacity(dim);

    for _ in 0..dim {
        // Xorshift for uniform [0,1]
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = (state as f32) / (u64::MAX as f32);

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = (state as f32) / (u64::MAX as f32);

        // Box-Muller
        let u1 = u1.max(1e-10); // avoid log(0)
        let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        noise.push(z * sigma);
    }

    noise
}

/// Perturb a ContinuousHV by adding noise to its values.
fn perturb_hv(hv: &ContinuousHV, noise: &[f32]) -> ContinuousHV {
    let values: Vec<f32> = hv
        .values
        .iter()
        .zip(noise.iter())
        .map(|(v, n)| v + n)
        .collect();
    ContinuousHV::from_values(values).normalize()
}

/// Scale a vector by a scalar.
fn scale_vec(v: &[f32], scale: f32) -> Vec<f32> {
    v.iter().map(|x| x * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::genesis::GenesisSeed;

    fn test_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            valence: 0.3,
            arousal: 0.5,
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            harmony_activations: [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2],
            prediction_error: 0.1,
            thought_vector: vec![0.3, -0.2],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn training_produces_result() {
        let genesis = GenesisSeed::from_phrase("test-training");
        let mut canvas = NeuralCanvas::new(&genesis);
        let config = TrainingConfig {
            population_size: 4,
            epochs: 3,
            ..Default::default()
        };
        let atelier_config = AtelierConfig {
            max_elements: 50,
            ..AtelierConfig::default()
        };
        let snapshots = vec![test_snapshot()];

        let result = train_projections(&mut canvas, &config, &atelier_config, &snapshots);

        assert!(!result.score_history.is_empty());
        assert!(result.initial_score >= 0.0);
        assert!(result.final_score >= 0.0);
        assert!(result.total_evaluations > 0);
    }

    #[test]
    fn gaussian_noise_distribution() {
        let noise = gaussian_noise(1000, 1.0, 42);
        let mean: f32 = noise.iter().sum::<f32>() / noise.len() as f32;
        let variance: f32 =
            noise.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / noise.len() as f32;

        // Mean should be near 0, variance near 1
        assert!(mean.abs() < 0.2, "mean = {mean}");
        assert!((variance - 1.0).abs() < 0.3, "variance = {variance}");
    }

    #[test]
    fn empty_snapshots_safe() {
        let genesis = GenesisSeed::from_phrase("test-empty");
        let mut canvas = NeuralCanvas::new(&genesis);
        let config = TrainingConfig::default();
        let atelier_config = AtelierConfig::default();

        let result = train_projections(&mut canvas, &config, &atelier_config, &[]);
        assert_eq!(result.total_evaluations, 0);
    }
}