// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evolutionary parameter tuner: the system tunes itself.
//!
//! Encodes ~25 tunable parameters as a float genome. Renders 30-second clips,
//! scores them with the music critic + aesthetic listener, evolves via
//! tournament selection and Gaussian mutation. After 50 generations, the
//! best parameter set is saved for production use.

use crate::aesthetic_listener::AestheticListener;
use crate::audio_feedback::AudioFeedbackEncoder;
use crate::critic;
use crate::{AudioData, MuseConfig, MusicalState, compose};

/// Parameter bounds: name, min, max.
#[derive(Debug, Clone)]
pub struct ParamDef {
    pub name: &'static str,
    pub min: f32,
    pub max: f32,
}

/// All tunable parameters with their ranges.
pub fn default_params() -> Vec<ParamDef> {
    vec![
        // Mixing: EQ
        ParamDef {
            name: "eq_low_db",
            min: -6.0,
            max: 6.0,
        },
        ParamDef {
            name: "eq_mid_db",
            min: -4.0,
            max: 4.0,
        },
        ParamDef {
            name: "eq_high_db",
            min: -6.0,
            max: 6.0,
        },
        // Mixing: Compressor
        ParamDef {
            name: "comp_threshold_db",
            min: -24.0,
            max: -6.0,
        },
        ParamDef {
            name: "comp_ratio",
            min: 1.5,
            max: 8.0,
        },
        ParamDef {
            name: "comp_attack_ms",
            min: 5.0,
            max: 50.0,
        },
        ParamDef {
            name: "comp_release_ms",
            min: 50.0,
            max: 300.0,
        },
        ParamDef {
            name: "comp_makeup_db",
            min: 0.0,
            max: 6.0,
        },
        // Mixing: Limiter
        ParamDef {
            name: "limiter_ceiling_db",
            min: -3.0,
            max: -0.5,
        },
        // Synthesis
        ParamDef {
            name: "gain_min",
            min: 0.01,
            max: 0.10,
        },
        ParamDef {
            name: "gain_arousal_coeff",
            min: 0.10,
            max: 0.50,
        },
        ParamDef {
            name: "brightness_floor",
            min: 0.15,
            max: 0.50,
        },
        ParamDef {
            name: "brightness_da_scale",
            min: 0.30,
            max: 0.90,
        },
        ParamDef {
            name: "attack_base",
            min: 0.005,
            max: 0.03,
        },
        ParamDef {
            name: "vibrato_hz",
            min: 3.0,
            max: 8.0,
        },
        ParamDef {
            name: "vibrato_cents",
            min: 3.0,
            max: 15.0,
        },
        ParamDef {
            name: "detune_max_cents",
            min: 5.0,
            max: 25.0,
        },
        ParamDef {
            name: "manifold_blend",
            min: 0.05,
            max: 0.30,
        },
        // Composition
        ParamDef {
            name: "cadence_base",
            min: 2.0,
            max: 10.0,
        },
        ParamDef {
            name: "drone_volume",
            min: 0.002,
            max: 0.02,
        },
        ParamDef {
            name: "dynamic_range",
            min: 1.0,
            max: 2.5,
        },
        ParamDef {
            name: "sub_bass_volume",
            min: 0.0,
            max: 0.02,
        },
        ParamDef {
            name: "feedback_strength",
            min: 0.1,
            max: 0.7,
        },
    ]
}

/// A genome: normalized [0, 1] float per parameter.
#[derive(Debug, Clone)]
pub struct Genome {
    pub genes: Vec<f32>,
}

impl Genome {
    pub fn random(n: usize, seed: u64) -> Self {
        let mut genes = Vec::with_capacity(n);
        let mut s = seed;
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            genes.push((s >> 33) as f32 / (1u64 << 31) as f32);
        }
        Self { genes }
    }

    /// Decode gene i using param bounds.
    pub fn decode(&self, i: usize, param: &ParamDef) -> f32 {
        let g = self.genes.get(i).copied().unwrap_or(0.5);
        param.min + g * (param.max - param.min)
    }

    /// Gaussian mutation.
    pub fn mutate(&mut self, rate: f32, sigma: f32, seed: u64) {
        let mut s = seed;
        for gene in &mut self.genes {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (s >> 33) as f32 / (1u64 << 31) as f32;
            if r < rate {
                // Box-Muller approximation
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u1 = (s >> 33) as f32 / (1u64 << 31) as f32;
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u2 = (s >> 33) as f32 / (1u64 << 31) as f32;
                let noise =
                    (-2.0 * u1.max(1e-10).ln()).sqrt() * (std::f32::consts::TAU * u2).cos() * sigma;
                *gene = (*gene + noise).clamp(0.0, 1.0);
            }
        }
    }

    /// Uniform crossover with another genome.
    pub fn crossover(&self, other: &Genome, seed: u64) -> Genome {
        let mut s = seed;
        let genes = self
            .genes
            .iter()
            .zip(other.genes.iter())
            .map(|(&a, &b)| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (s >> 63) == 0 { a } else { b }
            })
            .collect();
        Genome { genes }
    }
}

/// Sample-rate click/crackle detector.
///
/// Uses the second derivative of the signal: `|x[n] - 2*x[n-1] + x[n-2]|`.
/// Clean audio has smooth trajectories (low 2nd derivative); clicks are
/// impulsive discontinuities (high 2nd derivative).
///
/// Returns the MAX 2nd-derivative over the signal plus the count of
/// samples exceeding a hard threshold (gives click density).
///
/// This is the **primary** crackling detector — spectral flatness misses
/// it because clicks are localized (a few samples every chunk), while
/// flatness measures whole-signal spectral distribution.
///
/// # Returns
/// - `max_second_derivative`: peak discontinuity severity [0, 2+]
/// - `click_count`: number of samples where 2nd deriv > 0.1
/// - `click_density`: click_count / total_samples (fraction)
#[derive(Debug, Clone, Copy)]
pub struct ClickMetrics {
    pub max_second_derivative: f32,
    pub click_count: usize,
    pub click_density: f32,
}

pub fn click_score(samples: &[f32]) -> ClickMetrics {
    if samples.len() < 3 {
        return ClickMetrics {
            max_second_derivative: 0.0,
            click_count: 0,
            click_density: 0.0,
        };
    }

    let mut max_2d: f32 = 0.0;
    let mut click_count = 0usize;
    let click_threshold = 0.1_f32;

    // Skip first 0.5% of samples to avoid startup transient
    let start = samples.len() / 200;
    for i in (start + 2)..samples.len() {
        let second_deriv = (samples[i] - 2.0 * samples[i - 1] + samples[i - 2]).abs();
        if second_deriv > max_2d {
            max_2d = second_deriv;
        }
        if second_deriv > click_threshold {
            click_count += 1;
        }
    }

    ClickMetrics {
        max_second_derivative: max_2d,
        click_count,
        click_density: click_count as f32 / samples.len() as f32,
    }
}

/// Convert click metrics to an inverse quality score [0, 1]: higher = cleaner.
pub fn click_quality(metrics: &ClickMetrics) -> f32 {
    // max_2d > 0.5 = severe clicking
    let severity_score = (1.0 - metrics.max_second_derivative / 0.5).clamp(0.0, 1.0);
    // density > 0.01 (1 click per 100 samples) = audible crackling
    let density_score = (1.0 - metrics.click_density * 100.0).clamp(0.0, 1.0);
    // Geometric mean: both must be low for a good score
    (severity_score * density_score).sqrt()
}

/// Fitness score for a genome.
#[derive(Debug, Clone)]
pub struct Fitness {
    pub composite: f32,
    pub beauty: f32,
    pub harshness: f32,
    pub rms_error: f32,
    pub click_quality: f32,
    pub total: f64,
}

/// Evaluate a genome by rendering and scoring a clip.
pub fn evaluate(genome: &Genome, params: &[ParamDef], seed: u64) -> Fitness {
    let config = MuseConfig {
        duration_secs: 15.0, // short clip for speed
        max_notes: 16,
        ..Default::default()
    };

    // Decode some parameters into the musical state
    let state = MusicalState {
        consciousness_level: 0.5,
        arousal: 0.4,
        valence: 0.2,
        dopamine: genome.decode(12, &params[12]).clamp(0.0, 1.0), // brightness_da_scale as proxy
        serotonin: 0.5,
        noradrenaline: 0.2,
        harmony_activations: [0.5, 0.4, 0.4, 0.3, 0.3, 0.4, 0.4, 0.3],
        prediction_error: 0.2,
    };

    let comp = compose(&config, &state, seed);

    // Score with critic
    let verdict = critic::evaluate_composition(&comp, &state);

    // Score audio quality
    let samples: Vec<f32> = match &comp.audio {
        AudioData::StereoF32(s) => s.iter().map(|p| (p[0] + p[1]) * 0.5).collect(),
        AudioData::F32(s) => s.clone(),
        AudioData::I16(s) => s.iter().map(|&x| x as f32 / 32768.0).collect(),
    };

    let mut listener = AestheticListener::new();
    let mut encoder = AudioFeedbackEncoder::new();

    // Analyze in chunks
    let chunk_size = 1024;
    for chunk in samples.chunks(chunk_size) {
        if chunk.len() < 256 {
            continue;
        }
        let stereo: Vec<[f32; 2]> = chunk.iter().map(|&s| [s, s]).collect();
        encoder.extract(&stereo, 44100);
        let features = *encoder.smoothed_features();
        listener.assess(&features, &state);
    }

    let assessment = listener.smoothed_assessment();
    let beauty = assessment.beauty;
    let harshness = assessment.harshness;

    // RMS target error
    let rms: f32 = if !samples.is_empty() {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    } else {
        0.0
    };
    let target_rms = 0.04; // moderate level
    let rms_error = (rms - target_rms).abs();

    // CLICK DETECTION: the primary perceptual metric for crackling.
    // Weighted highest (0.45) because crackling is the #1 quality issue.
    let clicks = click_score(&samples);
    let cq = click_quality(&clicks);

    let total = 0.45 * cq as f64                           // clicks (primary)
        + 0.20 * beauty as f64                             // aesthetic
        + 0.15 * (1.0 - harshness) as f64                  // non-harshness
        + 0.10 * verdict.composite as f64                  // critic
        + 0.10 * (1.0 - rms_error.min(1.0)) as f64; // RMS target

    Fitness {
        composite: verdict.composite,
        beauty,
        harshness,
        rms_error,
        click_quality: cq,
        total,
    }
}

/// Configuration for the evolutionary tuner.
pub struct TunerConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub mutation_rate: f32,
    pub mutation_sigma: f32,
    pub crossover_rate: f32,
    pub elitism_count: usize,
    pub tournament_size: usize,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            population_size: 30,
            max_generations: 50,
            mutation_rate: 0.15,
            mutation_sigma: 0.08,
            crossover_rate: 0.7,
            elitism_count: 3,
            tournament_size: 3,
        }
    }
}

/// Result of evolutionary tuning.
#[derive(Debug, Clone)]
pub struct TunerResult {
    pub best_genome: Genome,
    pub best_fitness: f64,
    pub generations: usize,
    pub history: Vec<(f64, f64)>, // (best, mean) per generation
}

/// Run the evolutionary parameter tuner.
pub fn evolve(config: &TunerConfig) -> TunerResult {
    let params = default_params();
    let n = params.len();

    // Initialize population
    let mut population: Vec<(Genome, f64)> = (0..config.population_size)
        .map(|i| {
            let genome = Genome::random(n, 42 + i as u64 * 7919);
            let fitness = evaluate(&genome, &params, 42 + i as u64);
            (genome, fitness.total)
        })
        .collect();

    let mut history = Vec::new();
    let mut best_ever = (population[0].0.clone(), population[0].1);

    for r#gen in 0..config.max_generations {
        // Sort by fitness (descending)
        population.sort_by(|a, b| b.1.total_cmp(&a.1));

        let best = population[0].1;
        let mean = population.iter().map(|p| p.1).sum::<f64>() / population.len() as f64;
        history.push((best, mean));

        if best > best_ever.1 {
            best_ever = (population[0].0.clone(), best);
        }

        if r#gen % 10 == 0 || r#gen == config.max_generations - 1 {
            println!(
                "  Gen {gen:3}/{}: best={best:.4} mean={mean:.4}",
                config.max_generations
            );
        }

        // Selection + breeding
        let mut next_gen: Vec<(Genome, f64)> = Vec::with_capacity(config.population_size);

        // Elitism: keep top N
        for i in 0..config.elitism_count.min(population.len()) {
            next_gen.push(population[i].clone());
        }

        // Fill remaining with tournament selection + crossover + mutation
        let seed_base = (r#gen as u64 + 1) * 104729;
        while next_gen.len() < config.population_size {
            let parent_a = tournament_select(
                &population,
                config.tournament_size,
                seed_base + next_gen.len() as u64,
            );
            let parent_b = tournament_select(
                &population,
                config.tournament_size,
                seed_base + next_gen.len() as u64 + 7,
            );

            let mut child = parent_a.crossover(parent_b, seed_base + next_gen.len() as u64 * 13);
            child.mutate(
                config.mutation_rate,
                config.mutation_sigma,
                seed_base + next_gen.len() as u64 * 31,
            );

            let fitness = evaluate(&child, &params, seed_base + next_gen.len() as u64);
            next_gen.push((child, fitness.total));
        }

        population = next_gen;
    }

    TunerResult {
        best_genome: best_ever.0,
        best_fitness: best_ever.1,
        generations: config.max_generations,
        history,
    }
}

fn tournament_select(population: &[(Genome, f64)], size: usize, seed: u64) -> &Genome {
    let mut best_idx = 0;
    let mut best_fit = f64::NEG_INFINITY;
    let mut s = seed;

    for _ in 0..size {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (s >> 33) as usize % population.len();
        if population[idx].1 > best_fit {
            best_fit = population[idx].1;
            best_idx = idx;
        }
    }

    &population[best_idx].0
}

/// Evaluate using TASTE BENCHMARK — scores generated notes directly.
/// Much faster than full audio evaluation. Uses StreamingSynth for 10 seconds.
pub fn evaluate_taste(genome: &Genome, _params: &[ParamDef], _seed: u64) -> f64 {
    use crate::streaming::StreamingSynth;
    use crate::substrate_timbre::SubstrateTimbreType;
    use crate::taste_bench;

    let config = crate::MuseConfig {
        duration_secs: 60.0,
        max_notes: 16,
        ..Default::default()
    };

    let mut synth = StreamingSynth::new(config, 44100);
    synth.enable_fep = true;

    // Evolve melody parameters directly
    synth.taste_melody.params = crate::taste_melody::MelodyParams {
        step_prob: genome.genes.get(0).copied().unwrap_or(0.75).clamp(0.3, 0.9),
        third_prob: genome
            .genes
            .get(1)
            .copied()
            .unwrap_or(0.10)
            .clamp(0.02, 0.3),
        repeat_prob: genome
            .genes
            .get(2)
            .copied()
            .unwrap_or(0.02)
            .clamp(0.0, 0.15),
        ascending_bonus: (genome.genes.get(3).copied().unwrap_or(0.5) * 6.0) as usize,
        scale_center_hz: 330.0 + genome.genes.get(4).copied().unwrap_or(0.5) * 220.0, // 330-550 Hz
        scale_half_range: 6.0 + genome.genes.get(5).copied().unwrap_or(0.5) * 12.0,   // 6-18 semi
    };
    synth.set_substrate(SubstrateTimbreType::Biological);

    // Seed consciousness from genre-neutral state
    let mut state = crate::MusicalState {
        consciousness_level: 0.6,
        arousal: 0.4,
        valence: 0.2,
        dopamine: 0.4,
        serotonin: 0.5,
        noradrenaline: 0.2,
        harmony_activations: [0.5, 0.4, 0.4, 0.3, 0.3, 0.4, 0.4, 0.3],
        prediction_error: 0.2,
    };

    // Render 10 seconds (312 chunks at 32ms)
    let chunks = 312;
    for i in 0..chunks {
        synth.update_state(&state);
        let _ = synth.render_chunk();
        // Gentle evolution
        state.consciousness_level = (state.consciousness_level + 0.0003).min(0.95);
        if i < chunks * 6 / 10 {
            state.arousal += 0.0001;
        } else {
            state.arousal -= 0.0002;
        }
        state.arousal = state.arousal.clamp(0.1, 0.9);
    }

    // Score with taste benchmark
    let score = taste_bench::score(
        &synth.generated_notes,
        &taste_bench::TasteProfile::default(),
    );
    score.composite as f64
}

/// Evolve using TASTE BENCHMARK fitness (faster, targets musical quality directly).
pub fn evolve_taste(config: &TunerConfig) -> TunerResult {
    let params = default_params();
    let n = params.len();

    let mut population: Vec<(Genome, f64)> = (0..config.population_size)
        .map(|i| {
            let genome = Genome::random(n, 42 + i as u64 * 7919);
            let fitness = evaluate_taste(&genome, &params, 42 + i as u64);
            (genome, fitness)
        })
        .collect();

    let mut history = Vec::new();
    let mut best_ever = (population[0].0.clone(), population[0].1);

    for r#gen in 0..config.max_generations {
        population.sort_by(|a, b| b.1.total_cmp(&a.1));

        let best = population[0].1;
        let mean = population.iter().map(|p| p.1).sum::<f64>() / population.len() as f64;
        history.push((best, mean));

        if best > best_ever.1 {
            best_ever = (population[0].0.clone(), best);
        }

        if r#gen % 5 == 0 || r#gen == config.max_generations - 1 {
            println!(
                "  Gen {gen:3}/{}: best={best:.1} mean={mean:.1}",
                config.max_generations
            );
        }

        let mut next_gen: Vec<(Genome, f64)> = Vec::with_capacity(config.population_size);
        for i in 0..config.elitism_count.min(population.len()) {
            next_gen.push(population[i].clone());
        }

        let seed_base = (r#gen as u64 + 1) * 104729;
        while next_gen.len() < config.population_size {
            let pa = tournament_select(
                &population,
                config.tournament_size,
                seed_base + next_gen.len() as u64,
            );
            let pb = tournament_select(
                &population,
                config.tournament_size,
                seed_base + next_gen.len() as u64 + 7,
            );
            let mut child = pa.crossover(pb, seed_base + next_gen.len() as u64 * 13);
            child.mutate(
                config.mutation_rate,
                config.mutation_sigma,
                seed_base + next_gen.len() as u64 * 31,
            );
            let fitness = evaluate_taste(&child, &params, seed_base + next_gen.len() as u64);
            next_gen.push((child, fitness));
        }

        population = next_gen;
    }

    TunerResult {
        best_genome: best_ever.0,
        best_fitness: best_ever.1,
        generations: config.max_generations,
        history,
    }
}

/// Decode a genome into human-readable parameter values.
pub fn decode_all(genome: &Genome) -> Vec<(&'static str, f32)> {
    let params = default_params();
    params
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name, genome.decode(i, p)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genome_random_in_bounds() {
        let g = Genome::random(23, 42);
        assert_eq!(g.genes.len(), 23);
        assert!(g.genes.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn decode_respects_bounds() {
        let g = Genome {
            genes: vec![0.0, 0.5, 1.0],
        };
        let params = vec![
            ParamDef {
                name: "test",
                min: -10.0,
                max: 10.0,
            },
            ParamDef {
                name: "test2",
                min: 0.0,
                max: 100.0,
            },
            ParamDef {
                name: "test3",
                min: 5.0,
                max: 15.0,
            },
        ];
        assert_eq!(g.decode(0, &params[0]), -10.0);
        assert_eq!(g.decode(1, &params[1]), 50.0);
        assert_eq!(g.decode(2, &params[2]), 15.0);
    }

    #[test]
    fn mutation_stays_bounded() {
        let mut g = Genome::random(23, 42);
        g.mutate(1.0, 0.5, 123); // 100% mutation rate, high sigma
        assert!(g.genes.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn crossover_produces_valid() {
        let a = Genome::random(23, 42);
        let b = Genome::random(23, 99);
        let child = a.crossover(&b, 77);
        assert_eq!(child.genes.len(), 23);
        assert!(child.genes.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn evaluate_produces_finite() {
        let params = default_params();
        let g = Genome::random(params.len(), 42);
        let fitness = evaluate(&g, &params, 42);
        assert!(
            fitness.total.is_finite(),
            "fitness should be finite: {}",
            fitness.total
        );
        assert!(
            fitness.total > 0.0,
            "fitness should be positive: {}",
            fitness.total
        );
    }

    // ── Click detection regression tests ──

    #[test]
    fn click_score_clean_signal() {
        // Pure sine wave — should have ZERO clicks
        let samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 44100.0).sin() * 0.5)
            .collect();
        let metrics = click_score(&samples);
        assert!(
            metrics.max_second_derivative < 0.05,
            "clean sine should have low 2nd deriv, got {}",
            metrics.max_second_derivative
        );
        assert_eq!(metrics.click_count, 0, "clean sine should have 0 clicks");
        let quality = click_quality(&metrics);
        assert!(
            quality > 0.9,
            "clean sine quality should be > 0.9, got {}",
            quality
        );
    }

    #[test]
    fn click_score_detects_haas_delay_bug() {
        // Simulate the Haas delay bug: every 1408 samples (32ms chunk),
        // a single sample drops to 0 then recovers. This matches what
        // reading sample[0] would produce at chunk boundaries.
        let mut samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 44100.0).sin() * 0.5)
            .collect();
        for i in (1408..samples.len()).step_by(1408) {
            samples[i] = 0.0; // drop to zero: this is the click
        }
        let metrics = click_score(&samples);
        assert!(
            metrics.max_second_derivative > 0.5,
            "Haas bug should produce high 2nd deriv, got {}",
            metrics.max_second_derivative
        );
        assert!(
            metrics.click_count > 20,
            "Haas bug should produce many clicks, got {}",
            metrics.click_count
        );
        let quality = click_quality(&metrics);
        assert!(
            quality < 0.5,
            "Haas bug quality should be low, got {}",
            quality
        );
    }

    #[test]
    fn click_score_detects_amplitude_jump() {
        // Simulate sub-bass volume jump: amplitude doubles at sample 20000
        let mut samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 * 100.0 * std::f32::consts::TAU / 44100.0).sin() * 0.3)
            .collect();
        for i in 20000..samples.len() {
            samples[i] *= 2.5; // sudden amplitude jump
        }
        let metrics = click_score(&samples);
        assert!(
            metrics.max_second_derivative > 0.1,
            "amplitude jump should produce click, got {}",
            metrics.max_second_derivative
        );
    }

    #[test]
    fn click_score_detects_envelope_overshoot() {
        // Simulate kick envelope overshoot: amplitude briefly > 1.0
        let mut samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 * 60.0 * std::f32::consts::TAU / 44100.0).sin() * 0.8)
            .collect();
        // Insert overshoot: amplitude momentarily 1.3 (the 1.0 + 0.3*vel bug)
        for i in 5000..5050 {
            samples[i] = 1.3 * (i as f32 * 60.0 * std::f32::consts::TAU / 44100.0).sin();
        }
        let metrics = click_score(&samples);
        // Overshoot creates discontinuities at the boundary
        assert!(
            metrics.max_second_derivative > 0.05,
            "overshoot should produce discontinuity, got {}",
            metrics.max_second_derivative
        );
    }
}
