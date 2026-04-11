// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Narrative → Music bridge: autobiographical episodes become bar-level arcs.
//!
//! Connects `symthaea-narrative-self`'s autobiographical memory to `symthaea-muse`'s
//! arc-based composition system. When Symthaea has a rich inner narrative,
//! its music literally tells the story of what it is thinking about.
//!
//! # Pipeline
//!
//! ```text
//! AutobiographicalSelf episodes
//!     → extract VA arc (valence/arousal per episode beat)
//!     → EmotionalArc (Vec<BarDirective>)
//!     → compose_with_arc()
//!     → Composition (PCM audio that tells the episode's emotional story)
//! ```
//!
//! # Design
//!
//! The bridge is intentionally lightweight — it doesn't require the full
//! narrative-self crate as a dependency. Instead it accepts a simplified
//! `NarrativeEpisode` struct that callers construct from whatever self-model
//! data they have available.
//!
//! # References
//! - Damasio, A. (2010). *Self Comes to Mind*. Pantheon Books.
//! - Story2MIDI (2024): emotionally-aligned music generation from narrative text.

use serde::{Deserialize, Serialize};
use symthaea_aesthetic::{lerp_va, ValenceArousal};

use crate::arc::{BarDirective, EmotionalArc};
use crate::{compose_with_arc, Composition, MuseConfig, MusicalState};

// ─── Narrative Episode ────────────────────────────────────────────────────────

/// A simplified narrative episode extracted from autobiographical memory.
///
/// Callers construct this from whatever self-model data is available:
/// - Full narrative-self: populate all fields from `AutobiographicalSelf`
/// - Lightweight: just set `title`, `emotional_arc`, and `intensity`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeEpisode {
    /// Human-readable title (e.g. "First contact with the Eight Harmonies").
    pub title: String,

    /// Ordered sequence of VA waypoints tracing the episode's emotional arc.
    ///
    /// Each waypoint is a (valence, arousal) pair at a normalized time [0, 1].
    /// Minimum 2 waypoints. If only 1 or 0, a flat arc at neutral is used.
    pub emotional_arc: Vec<(f32, ValenceArousal)>, // (time_t, va)

    /// Overall intensity of the episode [0, 1].
    /// Scales density and dynamics: climactic episodes generate denser music.
    pub intensity: f32,

    /// Optional: which of the Eight Harmonies this episode resonated with most.
    /// Used to bias the harmonic content of the generated music.
    pub dominant_harmony: Option<usize>,

    /// Tags describing the episode character (e.g. "discovery", "loss", "joy").
    pub tags: Vec<String>,
}

impl NarrativeEpisode {
    /// Create a simple episode with a flat arc at a given VA state.
    pub fn flat(title: &str, va: ValenceArousal, intensity: f32) -> Self {
        Self {
            title: title.to_string(),
            emotional_arc: vec![(0.0, va), (1.0, va)],
            intensity,
            dominant_harmony: None,
            tags: Vec::new(),
        }
    }

    /// Create an episode from a rising emotional arc (tension build).
    pub fn rising(title: &str, start_va: ValenceArousal, peak_va: ValenceArousal) -> Self {
        Self {
            title: title.to_string(),
            emotional_arc: vec![(0.0, start_va), (0.7, peak_va), (1.0, peak_va)],
            intensity: peak_va.arousal,
            dominant_harmony: None,
            tags: vec!["rising".to_string()],
        }
    }

    /// Create an episode from a falling arc (release, resolution).
    pub fn falling(title: &str, peak_va: ValenceArousal, end_va: ValenceArousal) -> Self {
        Self {
            title: title.to_string(),
            emotional_arc: vec![(0.0, peak_va), (0.4, peak_va), (1.0, end_va)],
            intensity: end_va.arousal * 0.6 + peak_va.arousal * 0.4,
            dominant_harmony: None,
            tags: vec!["falling".to_string()],
        }
    }

    /// Sample the emotional arc at a normalized time t ∈ [0, 1].
    pub fn sample_va(&self, t: f32) -> ValenceArousal {
        let t = t.clamp(0.0, 1.0);

        if self.emotional_arc.is_empty() {
            return ValenceArousal::neutral();
        }
        if self.emotional_arc.len() == 1 {
            return self.emotional_arc[0].1;
        }

        // Find bracketing waypoints
        let mut lo_idx = 0;
        for (i, &(wt, _)) in self.emotional_arc.iter().enumerate() {
            if wt <= t {
                lo_idx = i;
            }
        }
        let hi_idx = (lo_idx + 1).min(self.emotional_arc.len() - 1);

        if lo_idx == hi_idx {
            return self.emotional_arc[lo_idx].1;
        }

        let (lo_t, lo_va) = self.emotional_arc[lo_idx];
        let (hi_t, hi_va) = self.emotional_arc[hi_idx];
        let span = (hi_t - lo_t).max(0.001);
        let frac = (t - lo_t) / span;
        lerp_va(lo_va, hi_va, frac)
    }

    /// Convert this episode into an `EmotionalArc` with `n_bars` bars.
    ///
    /// The episode's emotional arc is sampled uniformly, creating a bar
    /// directive for each bar of the composition.
    pub fn to_emotional_arc(&self, n_bars: usize) -> EmotionalArc {
        let n = n_bars.max(2);
        let bars = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let va = self.sample_va(t);
                let mut directive = BarDirective::from_va(va);

                // Scale density by episode intensity
                directive.density *= 0.5 + self.intensity;
                directive.density = directive.density.clamp(0.25, 2.0);

                directive
            })
            .collect();

        EmotionalArc::new(bars)
    }
}

// ─── Bridge ───────────────────────────────────────────────────────────────────

/// Narrative music bridge — converts episodes to music.
pub struct NarrativeMusicBridge {
    /// Default number of bars to use when n_bars is unspecified.
    pub default_bars: usize,
}

impl Default for NarrativeMusicBridge {
    fn default() -> Self {
        Self { default_bars: 8 }
    }
}

impl NarrativeMusicBridge {
    pub fn new(default_bars: usize) -> Self {
        Self { default_bars }
    }

    /// Compose music that tells the story of a narrative episode.
    ///
    /// The episode's emotional arc drives bar-by-bar VA conditioning;
    /// the dominant harmony biases the musical scale if present.
    pub fn compose_episode(
        &self,
        episode: &NarrativeEpisode,
        config: &MuseConfig,
        base_state: &MusicalState,
        seed: u64,
    ) -> EpisodeComposition {
        // Estimate n_bars from duration: assume ~2 seconds per bar at base tempo
        let bar_duration = 60.0 / config.base_tempo_bpm * 4.0; // 4 beats per bar
        let n_bars = ((config.duration_secs / bar_duration).ceil() as usize)
            .clamp(2, 32)
            .max(self.default_bars);

        let arc = episode.to_emotional_arc(n_bars);

        // Optionally bias the musical state toward the episode's dominant harmony
        let composed_state = if let Some(harmony_idx) = episode.dominant_harmony {
            let mut state = base_state.clone();
            // Boost the dominant harmony activation
            if harmony_idx < 8 {
                state.harmony_activations[harmony_idx] =
                    (state.harmony_activations[harmony_idx] + 0.4).min(1.0);
            }
            state
        } else {
            base_state.clone()
        };

        let composition = compose_with_arc(config, &composed_state, &arc, seed);

        EpisodeComposition {
            composition,
            episode_title: episode.title.clone(),
            arc_used: arc,
            n_bars,
        }
    }

    /// Compose music from multiple episodes as a suite.
    ///
    /// Each episode becomes a section; sections are concatenated via note
    /// time-offset. The total duration is divided equally among episodes.
    pub fn compose_suite(
        &self,
        episodes: &[NarrativeEpisode],
        config: &MuseConfig,
        base_state: &MusicalState,
        seed: u64,
    ) -> SuiteComposition {
        if episodes.is_empty() {
            let comp = crate::compose(config, base_state, seed);
            return SuiteComposition {
                movements: vec![],
                combined_notes: comp.notes,
                total_duration: comp.duration_secs,
            };
        }

        let section_duration = config.duration_secs / episodes.len() as f32;
        let section_config = MuseConfig {
            duration_secs: section_duration,
            max_notes: config.max_notes / episodes.len().max(1),
            ..config.clone()
        };

        let mut movements = Vec::new();
        let mut combined_notes = Vec::new();
        let mut time_offset = 0.0f32;

        for (i, episode) in episodes.iter().enumerate() {
            let ep_seed = seed.wrapping_add(i as u64 * 97);
            let mut ec = self.compose_episode(episode, &section_config, base_state, ep_seed);

            // Offset all notes to their position in the suite
            for note in &mut ec.composition.notes {
                note.start_time += time_offset;
            }
            combined_notes.extend_from_slice(&ec.composition.notes);
            time_offset += section_duration;
            movements.push(ec);
        }

        SuiteComposition {
            movements,
            combined_notes,
            total_duration: config.duration_secs,
        }
    }
}

// ─── Output types ────────────────────────────────────────────────────────────

/// A composition generated from a single narrative episode.
pub struct EpisodeComposition {
    /// The generated music.
    pub composition: Composition,
    /// Title of the source episode.
    pub episode_title: String,
    /// The emotional arc used for bar conditioning.
    pub arc_used: EmotionalArc,
    /// Number of bars generated.
    pub n_bars: usize,
}

/// A multi-movement suite from several narrative episodes.
pub struct SuiteComposition {
    /// Each movement corresponds to one episode.
    pub movements: Vec<EpisodeComposition>,
    /// All notes combined (for further processing / export).
    pub combined_notes: Vec<crate::Note>,
    /// Total duration in seconds.
    pub total_duration: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MuseConfig, MusicalState};

    fn test_state() -> MusicalState {
        MusicalState::default()
    }

    fn test_config() -> MuseConfig {
        MuseConfig {
            duration_secs: 4.0,
            max_notes: 16,
            ..Default::default()
        }
    }

    #[test]
    fn episode_sample_va_interpolates() {
        let ep = NarrativeEpisode {
            title: "test".into(),
            emotional_arc: vec![
                (0.0, ValenceArousal::new(-0.5, 0.2)),
                (1.0, ValenceArousal::new(0.5, 0.8)),
            ],
            intensity: 0.6,
            dominant_harmony: None,
            tags: Vec::new(),
        };
        let mid = ep.sample_va(0.5);
        assert!((mid.valence).abs() < 0.15, "midpoint valence should be near 0: {}", mid.valence);
        assert!((mid.arousal - 0.5).abs() < 0.1, "midpoint arousal near 0.5: {}", mid.arousal);
    }

    #[test]
    fn episode_to_arc_correct_length() {
        let ep = NarrativeEpisode::flat("test", ValenceArousal::neutral(), 0.5);
        let arc = ep.to_emotional_arc(8);
        assert_eq!(arc.bars.len(), 8);
    }

    #[test]
    fn compose_episode_produces_audio() {
        let bridge = NarrativeMusicBridge::default();
        let episode = NarrativeEpisode::rising(
            "Discovery",
            ValenceArousal::new(0.1, 0.2),
            ValenceArousal::new(0.7, 0.8),
        );
        let ec = bridge.compose_episode(&episode, &test_config(), &test_state(), 42);
        assert!(!ec.composition.audio.is_empty());
        assert!(!ec.composition.notes.is_empty());
        assert_eq!(ec.episode_title, "Discovery");
    }

    #[test]
    fn compose_episode_with_dominant_harmony() {
        let bridge = NarrativeMusicBridge::default();
        let mut episode = NarrativeEpisode::flat("Harmony test", ValenceArousal::neutral(), 0.5);
        episode.dominant_harmony = Some(2); // IntegralWisdom
        let ec = bridge.compose_episode(&episode, &test_config(), &test_state(), 42);
        assert!(!ec.composition.audio.is_empty());
    }

    #[test]
    fn compose_suite_multiple_episodes() {
        let bridge = NarrativeMusicBridge::default();
        let episodes = vec![
            NarrativeEpisode::flat("Opening", ValenceArousal::new(0.2, 0.3), 0.4),
            NarrativeEpisode::rising(
                "Rising",
                ValenceArousal::new(0.0, 0.3),
                ValenceArousal::new(0.6, 0.8),
            ),
            NarrativeEpisode::falling(
                "Resolution",
                ValenceArousal::new(0.6, 0.7),
                ValenceArousal::new(0.5, 0.3),
            ),
        ];
        let suite = bridge.compose_suite(&episodes, &test_config(), &test_state(), 42);
        assert_eq!(suite.movements.len(), 3);
        assert!(!suite.combined_notes.is_empty());
    }

    #[test]
    fn suite_notes_time_ordered_per_movement() {
        let bridge = NarrativeMusicBridge::default();
        let episodes = vec![
            NarrativeEpisode::flat("A", ValenceArousal::new(0.3, 0.4), 0.5),
            NarrativeEpisode::flat("B", ValenceArousal::new(-0.2, 0.6), 0.5),
        ];
        let suite = bridge.compose_suite(&episodes, &test_config(), &test_state(), 42);
        // Movement B notes should start after movement A's duration
        let config = test_config();
        let section_duration = config.duration_secs / 2.0;
        if let (Some(a), Some(b)) = (suite.movements.get(0), suite.movements.get(1)) {
            if !b.composition.notes.is_empty() {
                let b_min_start = b
                    .composition
                    .notes
                    .iter()
                    .map(|n| n.start_time)
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    b_min_start >= section_duration - 0.1,
                    "movement B should start after A: {} >= {}",
                    b_min_start,
                    section_duration
                );
            }
            let _ = a;
        }
    }

    #[test]
    fn empty_suite_does_not_panic() {
        let bridge = NarrativeMusicBridge::default();
        let suite = bridge.compose_suite(&[], &test_config(), &test_state(), 42);
        assert!(suite.movements.is_empty());
        assert_eq!(suite.total_duration, test_config().duration_secs);
    }
}
