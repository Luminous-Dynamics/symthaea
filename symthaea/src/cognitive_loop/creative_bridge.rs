// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Creative bridge: CognitiveLoop → Atelier/Muse → Art output.
//!
//! Parallel to canvas_bridge.rs — stateful manager that produces generative art
//! (visual SVG, musical PCM, poetry text) driven by consciousness state and
//! the Wallas creative cycle. Feeds aesthetic scores back into the neuromodulator
//! bath to close the creative feedback loop.

#[cfg(feature = "creative")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "creative")]
use symthaea_aesthetic::{AestheticConfig, AestheticFeedback, AestheticScore, AestheticTracker};
#[cfg(feature = "creative")]
use symthaea_atelier::{AtelierConfig, AtelierStyle, Artwork};
#[cfg(feature = "creative")]
use symthaea_canvas::CognitiveSnapshot;
#[cfg(feature = "creative")]
use symthaea_muse::{Composition, MuseConfig, MusicalState};

/// Telemetry from the creative pipeline, stored in CycleMetadata.
#[cfg(feature = "creative")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreativeTelemetry {
    /// Whether any art was generated this cycle.
    pub generated: bool,
    /// Whether generation was gated by low consciousness.
    pub consciousness_gated: bool,
    /// Aesthetic score of the most recent artwork.
    pub aesthetic_score: f32,
    /// Aesthetic EMA (the system's aesthetic expectation).
    pub aesthetic_ema: f32,
    /// Dopamine delta from aesthetic feedback.
    pub dopamine_delta: f32,
    /// Serotonin delta from aesthetic feedback.
    pub serotonin_delta: f32,
    /// Surprise signal for exploration.
    pub surprise_signal: f32,
    /// Which modality was generated.
    pub modality: String,
    /// Number of generate-evaluate iterations used.
    pub iteration_count: usize,
    /// Generation time in microseconds.
    pub generation_time_us: u64,
    /// Total artworks created since startup.
    pub total_artworks: u64,
}

/// Creative output from a single cycle.
#[cfg(feature = "creative")]
#[derive(Debug, Clone)]
pub struct CreativeOutput {
    /// Generated SVG artwork, if any.
    pub artwork_svg: Option<String>,
    /// Generated PCM audio samples, if any.
    pub music_samples: Option<Vec<i16>>,
    /// Generated poem/creative text, if any.
    pub poem: Option<String>,
    /// Aesthetic feedback to inject into neuromodulator bath.
    pub feedback: AestheticFeedback,
}

#[cfg(feature = "creative")]
impl Default for CreativeOutput {
    fn default() -> Self {
        Self {
            artwork_svg: None,
            music_samples: None,
            poem: None,
            feedback: AestheticFeedback::neutral(),
        }
    }
}

/// Stateful creative manager — holds atelier/muse configs, aesthetic tracker,
/// and generation state.
#[cfg(feature = "creative")]
pub(crate) struct CreativeManager {
    /// Visual art configuration.
    atelier_config: AtelierConfig,
    /// Music generation configuration.
    muse_config: MuseConfig,
    /// Aesthetic feedback tracker (EMA + reward computation).
    tracker: AestheticTracker,
    /// Consciousness threshold for creative generation.
    consciousness_threshold: f32,
    /// Generate art every N cycles.
    generation_interval: u32,
    /// Cycle counter for interval gating.
    cycles_since_generation: u32,
    /// Monotonic seed for deterministic-per-cycle generation.
    seed_counter: u64,
    /// Total artworks produced.
    total_artworks: u64,
    /// Most recent telemetry.
    last_telemetry: CreativeTelemetry,
    /// Which modality to produce next (rotates: visual → music → visual → ...)
    next_modality: CreativeModality,
}

#[cfg(feature = "creative")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CreativeModality {
    Visual,
    Music,
}

#[cfg(feature = "creative")]
impl CreativeManager {
    pub fn new() -> Self {
        Self {
            atelier_config: AtelierConfig {
                style: AtelierStyle::Composite,
                iteration_budget: 5,
                ..AtelierConfig::default()
            },
            muse_config: MuseConfig {
                duration_secs: 4.0,
                max_notes: 16,
                ..MuseConfig::default()
            },
            tracker: AestheticTracker::new(AestheticConfig::default()),
            consciousness_threshold: 0.3,
            generation_interval: 10,
            cycles_since_generation: 0,
            seed_counter: 0,
            total_artworks: 0,
            last_telemetry: CreativeTelemetry::default(),
            next_modality: CreativeModality::Visual,
        }
    }

    /// Tick the creative pipeline. Returns creative output when art is generated.
    pub fn tick(&mut self, snap: &CognitiveSnapshot) -> Option<CreativeOutput> {
        self.cycles_since_generation += 1;

        // Interval gating
        if self.cycles_since_generation < self.generation_interval {
            self.last_telemetry = CreativeTelemetry {
                generated: false,
                aesthetic_ema: self.tracker.expectation(),
                total_artworks: self.total_artworks,
                ..CreativeTelemetry::default()
            };
            return None;
        }
        self.cycles_since_generation = 0;

        // Consciousness gating
        if (snap.consciousness_level as f32) < self.consciousness_threshold {
            self.last_telemetry = CreativeTelemetry {
                generated: false,
                consciousness_gated: true,
                aesthetic_ema: self.tracker.expectation(),
                total_artworks: self.total_artworks,
                ..CreativeTelemetry::default()
            };
            return None;
        }

        let start = std::time::Instant::now();
        self.seed_counter += 1;

        let mut output = CreativeOutput::default();
        let modality_name;

        match self.next_modality {
            CreativeModality::Visual => {
                let artwork = symthaea_atelier::create_artwork_iterative(
                    &self.atelier_config,
                    snap,
                    self.seed_counter,
                );
                let score = artwork.aesthetic_score;
                let feedback =
                    self.tracker.process(&score, &snap.harmony_activations);
                output.artwork_svg = Some(artwork.svg);
                output.feedback = feedback;
                modality_name = "visual";

                self.record_telemetry(
                    &score,
                    &feedback,
                    modality_name,
                    artwork.generation_cycles,
                    start.elapsed(),
                );

                self.next_modality = CreativeModality::Music;
            }
            CreativeModality::Music => {
                let musical_state = snapshot_to_musical_state(snap);
                let composition =
                    symthaea_muse::compose(&self.muse_config, &musical_state, self.seed_counter);

                // Score music using harmony alignment as a proxy
                let music_score = score_composition(&composition, snap);
                let feedback =
                    self.tracker.process(&music_score, &snap.harmony_activations);
                output.music_samples = Some(composition.samples);
                output.feedback = feedback;
                modality_name = "music";

                self.record_telemetry(
                    &music_score,
                    &feedback,
                    modality_name,
                    1,
                    start.elapsed(),
                );

                self.next_modality = CreativeModality::Visual;
            }
        }

        self.total_artworks += 1;
        Some(output)
    }

    fn record_telemetry(
        &mut self,
        score: &AestheticScore,
        feedback: &AestheticFeedback,
        modality: &str,
        iterations: usize,
        elapsed: std::time::Duration,
    ) {
        self.last_telemetry = CreativeTelemetry {
            generated: true,
            consciousness_gated: false,
            aesthetic_score: score.composite,
            aesthetic_ema: self.tracker.expectation(),
            dopamine_delta: feedback.dopamine_delta,
            serotonin_delta: feedback.serotonin_delta,
            surprise_signal: feedback.surprise_signal,
            modality: modality.to_string(),
            iteration_count: iterations,
            generation_time_us: elapsed.as_micros() as u64,
            total_artworks: self.total_artworks + 1,
        };
    }

    /// Most recent telemetry.
    pub fn last_telemetry(&self) -> &CreativeTelemetry {
        &self.last_telemetry
    }

    /// Current aesthetic expectation (EMA).
    pub fn aesthetic_expectation(&self) -> f32 {
        self.tracker.expectation()
    }

    /// Total artworks produced.
    pub fn total_artworks(&self) -> u64 {
        self.total_artworks
    }
}

#[cfg(feature = "creative")]
impl Default for CreativeManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a CognitiveSnapshot to MusicalState.
#[cfg(feature = "creative")]
fn snapshot_to_musical_state(snap: &CognitiveSnapshot) -> MusicalState {
    MusicalState {
        harmony_activations: snap.harmony_activations,
        dopamine: snap.dopamine,
        serotonin: snap.serotonin,
        noradrenaline: snap.noradrenaline,
        arousal: snap.arousal,
        valence: snap.valence,
        consciousness_level: snap.consciousness_level as f32,
        prediction_error: snap.prediction_error,
    }
}

/// Score a musical composition aesthetically.
///
/// Uses harmony alignment and note diversity as proxies for musical beauty.
#[cfg(feature = "creative")]
fn score_composition(comp: &Composition, snap: &CognitiveSnapshot) -> AestheticScore {
    // Order: rhythmic regularity (notes evenly spaced)
    let order = if comp.notes.len() >= 2 {
        let intervals: Vec<f32> = comp
            .notes
            .windows(2)
            .map(|w| w[1].start_time - w[0].start_time)
            .collect();
        let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
        if mean_interval > 0.0 {
            let variance = intervals
                .iter()
                .map(|&i| (i - mean_interval).powi(2))
                .sum::<f32>()
                / intervals.len() as f32;
            let cv = variance.sqrt() / mean_interval;
            (1.0 - cv).clamp(0.0, 1.0)
        } else {
            0.5
        }
    } else {
        0.3
    };

    // Complexity: pitch diversity
    let mut unique_pitches: Vec<f32> = comp.notes.iter().map(|n| (n.frequency * 10.0).round()).collect();
    unique_pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_pitches.dedup();
    let complexity = if comp.notes.is_empty() {
        0.1
    } else {
        (unique_pitches.len() as f32 / comp.notes.len() as f32).clamp(0.0, 1.0)
    };

    // Harmony: mean harmony activation
    let harmony = snap.harmony_activations.iter().sum::<f32>() / 8.0;

    let mut score = AestheticScore {
        order,
        complexity,
        surprise: 0.0,
        harmony,
        birkhoff: if complexity > 0.01 {
            (order / complexity).clamp(0.0, 1.0)
        } else {
            0.0
        },
        composite: 0.0,
    };
    score.compute_composite();
    score
}

#[cfg(test)]
#[cfg(feature = "creative")]
mod tests {
    use super::*;

    fn test_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2],
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            arousal: 0.5,
            valence: 0.2,
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8]],
            persistence_cycles: vec![[0.2, 0.6]],
            thought_vector: vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, -0.1],
            betti_0: 3,
            betti_1: 1,
            betti_2: 0,
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn manager_respects_interval() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 3;
        let snap = test_snapshot();

        // First two ticks should be gated
        assert!(manager.tick(&snap).is_none());
        assert!(manager.tick(&snap).is_none());
        // Third tick should produce output
        assert!(manager.tick(&snap).is_some());
    }

    #[test]
    fn manager_gates_on_low_consciousness() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1; // generate every cycle
        let snap = CognitiveSnapshot {
            consciousness_level: 0.1, // below threshold
            ..CognitiveSnapshot::dormant()
        };
        assert!(manager.tick(&snap).is_none());
        assert!(manager.last_telemetry().consciousness_gated);
    }

    #[test]
    fn manager_alternates_modalities() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        let snap = test_snapshot();

        let first = manager.tick(&snap).unwrap();
        assert!(first.artwork_svg.is_some()); // visual first
        assert!(first.music_samples.is_none());

        let second = manager.tick(&snap).unwrap();
        assert!(second.music_samples.is_some()); // music second
        assert!(second.artwork_svg.is_none());

        let third = manager.tick(&snap).unwrap();
        assert!(third.artwork_svg.is_some()); // back to visual
    }

    #[test]
    fn feedback_has_dopamine_signal() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        let snap = test_snapshot();

        let output = manager.tick(&snap).unwrap();
        // Feedback should be non-neutral (either positive or negative delta)
        let f = &output.feedback;
        // At minimum, serotonin_delta should be positive (harmony > 0)
        assert!(f.serotonin_delta > 0.0);
    }

    #[test]
    fn telemetry_tracks_artworks() {
        let mut manager = CreativeManager::new();
        manager.generation_interval = 1;
        let snap = test_snapshot();

        manager.tick(&snap);
        assert_eq!(manager.total_artworks(), 1);

        manager.tick(&snap);
        assert_eq!(manager.total_artworks(), 2);
    }

    #[test]
    fn snapshot_to_musical_state_maps_correctly() {
        let snap = test_snapshot();
        let ms = snapshot_to_musical_state(&snap);
        assert_eq!(ms.dopamine, snap.dopamine);
        assert_eq!(ms.valence, snap.valence);
        assert_eq!(ms.harmony_activations, snap.harmony_activations);
    }

    #[test]
    fn score_composition_bounded() {
        let comp = symthaea_muse::compose(
            &MuseConfig {
                duration_secs: 1.0,
                max_notes: 4,
                ..Default::default()
            },
            &MusicalState::default(),
            42,
        );
        let snap = test_snapshot();
        let score = score_composition(&comp, &snap);
        assert!(score.composite >= 0.0 && score.composite <= 1.0);
        assert!(score.order >= 0.0 && score.order <= 1.0);
        assert!(score.complexity >= 0.0 && score.complexity <= 1.0);
    }
}
