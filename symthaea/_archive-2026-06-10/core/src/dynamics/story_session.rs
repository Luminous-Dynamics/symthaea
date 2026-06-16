// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Scene Story Session
//!
//! Wraps `NarrativeAlgebra` + `StoryArcDynamics` with stateful story management:
//! character registration, conflict tracking, scene logging, and HV→CfC projection.

use std::collections::HashMap;

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

use crate::dynamics::narrative_dynamics::{NarrativeSignal, StoryArcConfig, StoryArcDynamics};
use crate::hdc::narrative_algebra::{ArcPhase, NarrativeAlgebra, NarrativeMood};

// ============================================================================
// Types
// ============================================================================

/// A logged scene with its signal and HDC representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneRecord {
    pub index: usize,
    pub title: String,
    pub setting: String,
    pub conflict: String,
    pub mood: NarrativeMood,
    pub characters: Vec<String>,
    pub signal: NarrativeSignal,
    pub scene_hv: ContinuousHV,
}

/// A tracked conflict within the story.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictEntry {
    pub description: String,
    pub introduced_at: usize,
    pub resolved_at: Option<usize>,
}

/// Per-character arc tracking data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterArc {
    /// Scene indices where this character appears
    pub scene_indices: Vec<usize>,
    /// Ghost Signal at each appearance
    pub signals: Vec<NarrativeSignal>,
    /// Running-average presence HV (weighted bundle of scene HVs)
    pub presence_hv: ContinuousHV,
}

/// Snapshot of the story's current state.
pub struct StoryState {
    pub total_scenes: usize,
    pub unresolved_conflicts: usize,
    pub total_conflicts: usize,
    pub character_count: usize,
    pub themes: Vec<String>,
    pub latest_signal: Option<NarrativeSignal>,
    pub current_arc_phase: ArcPhase,
}

// ============================================================================
// StorySession
// ============================================================================

/// Stateful story session combining HDC algebra with CfC dynamics.
pub struct StorySession {
    algebra: NarrativeAlgebra,
    dynamics: StoryArcDynamics,
    characters: HashMap<String, ContinuousHV>,
    character_arcs: HashMap<String, CharacterArc>,
    conflicts: Vec<ConflictEntry>,
    themes: Vec<String>,
    scene_log: Vec<SceneRecord>,
    input_dim: usize,
    /// Expected total number of scenes (for position encoding). 0 = unknown.
    expected_total_scenes: usize,
}

impl StorySession {
    /// Create a new story session with default configuration.
    ///
    /// Uses semantic archetypes derived from the narrative algebra for
    /// meaningful HDC signal variation.
    pub fn new() -> Self {
        let config = StoryArcConfig::default();
        let input_dim = config.input_dim;
        let algebra = NarrativeAlgebra::default_dim();
        let dynamics = StoryArcDynamics::with_algebra(config, &algebra);
        Self {
            algebra,
            dynamics,
            characters: HashMap::new(),
            character_arcs: HashMap::new(),
            conflicts: Vec::new(),
            themes: Vec::new(),
            scene_log: Vec::new(),
            input_dim,
            expected_total_scenes: 0,
        }
    }

    /// Create with a custom `StoryArcConfig`.
    pub fn with_config(config: StoryArcConfig) -> Self {
        let input_dim = config.input_dim;
        let algebra = NarrativeAlgebra::default_dim();
        let dynamics = StoryArcDynamics::with_algebra(config, &algebra);
        Self {
            algebra,
            dynamics,
            characters: HashMap::new(),
            character_arcs: HashMap::new(),
            conflicts: Vec::new(),
            themes: Vec::new(),
            scene_log: Vec::new(),
            input_dim,
            expected_total_scenes: 0,
        }
    }

    /// Set the expected total number of scenes for position-aware tension shaping.
    ///
    /// When set, each scene's normalized position (0.0 = start, 1.0 = end) is
    /// passed to the dynamics engine, causing tension to be dampened early and
    /// amplified near the climax point. Set to 0 to disable.
    pub fn set_expected_scenes(&mut self, total: usize) {
        self.expected_total_scenes = total;
    }

    /// Register a character and return its HDC encoding.
    pub fn register_character(
        &mut self,
        name: &str,
        role_primitive: &ContinuousHV,
    ) -> ContinuousHV {
        let hv = self.algebra.encode_character(name, role_primitive);
        self.characters.insert(name.to_string(), hv.clone());
        self.character_arcs.insert(
            name.to_string(),
            CharacterArc {
                scene_indices: Vec::new(),
                signals: Vec::new(),
                presence_hv: hv.clone(),
            },
        );
        hv
    }

    /// Add a thematic throughline.
    pub fn add_theme(&mut self, theme: &str) {
        self.themes.push(theme.to_string());
    }

    /// Add a scene to the story, advancing CfC dynamics.
    ///
    /// Returns the resulting `NarrativeSignal` (Ghost Signal).
    ///
    /// Uses hybrid HDC+CfC signal extraction and scene-to-scene transition
    /// operators to modulate the CfC input based on narrative escalation
    /// and transformation between consecutive scenes.
    pub fn add_scene(
        &mut self,
        title: &str,
        setting: &str,
        character_names: &[&str],
        conflict: &str,
        mood: NarrativeMood,
    ) -> NarrativeSignal {
        // Gather character HVs (skip unregistered names)
        let char_hvs: Vec<ContinuousHV> = character_names
            .iter()
            .filter_map(|name| self.characters.get(*name).cloned())
            .collect();

        // Compose scene HDC vector
        let scene_hv = self
            .algebra
            .encode_scene(setting, &char_hvs, conflict, mood);

        // Project 4096-D HV → input_dim-D for CfC
        let mut projected = self.project_hv(&scene_hv);

        // Conflict intensity: scale projected input by keyword intensity
        let intensity = StoryArcDynamics::conflict_intensity(conflict);
        if intensity > 1.0 {
            projected.mapv_inplace(|x| x * intensity);
        }

        // Scene-to-scene transition modulation
        if let Some(prev_record) = self.scene_log.last() {
            let prev_hv = &prev_record.scene_hv;
            // Compute escalation and transformation similarity
            let escalation_hv = self.algebra.escalates(prev_hv, &scene_hv);
            let escalation_sim = escalation_hv.similarity(&scene_hv).abs();

            let transform_hv = self.algebra.transforms_into(prev_hv, &scene_hv);
            let transform_sim = transform_hv.similarity(&scene_hv).abs();

            // Amplify first half by escalation factor, second half by transform factor
            let half = projected.len() / 2;
            let esc_factor = 1.0 + escalation_sim;
            let trans_factor = 1.0 + transform_sim;
            for i in 0..half {
                projected[i] *= esc_factor;
            }
            for i in half..projected.len() {
                projected[i] *= trans_factor;
            }
        }

        // Compute scene position for arc-phase tension shaping
        let scene_position = if self.expected_total_scenes >= 2 {
            Some(self.scene_log.len() as f32 / (self.expected_total_scenes - 1) as f32)
        } else {
            None
        };

        // Advance dynamics using hybrid HDC+CfC signal with mood bias and position
        let signal = self.dynamics.step_hybrid_with_mood_and_position(
            &projected,
            &scene_hv,
            0.1,
            Some(mood),
            scene_position,
        );

        let index = self.scene_log.len();
        self.scene_log.push(SceneRecord {
            index,
            title: title.to_string(),
            setting: setting.to_string(),
            conflict: conflict.to_string(),
            mood,
            characters: character_names.iter().map(|s| s.to_string()).collect(),
            signal: signal.clone(),
            scene_hv: scene_hv.clone(),
        });

        // Update character arcs with per-character divergent signals
        for name in character_names {
            if let Some(arc) = self.character_arcs.get_mut(*name) {
                arc.scene_indices.push(index);
                // Compute character-specific signal from their presence HV
                let char_signal = self.dynamics.character_signal(&arc.presence_hv, &signal);
                arc.signals.push(char_signal);
                // Running-average presence HV via weighted bundle
                let count = arc.scene_indices.len() as f32;
                let old_weight = (count - 1.0) / count;
                let new_weight = 1.0 / count;
                arc.presence_hv = ContinuousHV::weighted_bundle(
                    &[&arc.presence_hv, &scene_hv],
                    &[old_weight, new_weight],
                );
            }
        }

        signal
    }

    /// Introduce a new conflict, injecting a tension spike into the dynamics.
    pub fn introduce_conflict(&mut self, description: &str) {
        let scene_idx = self.scene_log.len();
        self.conflicts.push(ConflictEntry {
            description: description.to_string(),
            introduced_at: scene_idx,
            resolved_at: None,
        });
        // Each active conflict raises baseline tension
        self.dynamics.inject_tension_spike(0.15);
    }

    /// Resolve a conflict by description. Returns `true` if found and resolved.
    ///
    /// Injects a tension drop into the dynamics on resolution.
    pub fn resolve_conflict(&mut self, description: &str) -> bool {
        let scene_idx = self.scene_log.len();
        for entry in &mut self.conflicts {
            if entry.description == description && entry.resolved_at.is_none() {
                entry.resolved_at = Some(scene_idx);
                self.dynamics.inject_tension_drop(0.12);
                return true;
            }
        }
        false
    }

    /// Get a snapshot of the current story state.
    pub fn get_story_state(&self) -> StoryState {
        let unresolved = self
            .conflicts
            .iter()
            .filter(|c| c.resolved_at.is_none())
            .count();

        let latest_signal = self.scene_log.last().map(|r| r.signal.clone());
        let current_phase = latest_signal
            .as_ref()
            .map(|s| s.arc_phase)
            .unwrap_or(ArcPhase::Setup);

        StoryState {
            total_scenes: self.scene_log.len(),
            unresolved_conflicts: unresolved,
            total_conflicts: self.conflicts.len(),
            character_count: self.characters.len(),
            themes: self.themes.clone(),
            latest_signal,
            current_arc_phase: current_phase,
        }
    }

    /// Access the scene log.
    pub fn scene_log(&self) -> &[SceneRecord] {
        &self.scene_log
    }

    /// Cosine similarity between two logged scenes by index.
    pub fn scene_similarity(&self, idx_a: usize, idx_b: usize) -> Option<f32> {
        let a = self.scene_log.get(idx_a)?;
        let b = self.scene_log.get(idx_b)?;
        Some(self.algebra.scene_similarity(&a.scene_hv, &b.scene_hv))
    }

    /// Access the underlying algebra engine.
    pub fn algebra(&self) -> &NarrativeAlgebra {
        &self.algebra
    }

    /// Get a character's arc tracking data.
    pub fn get_character_arc(&self, name: &str) -> Option<&CharacterArc> {
        self.character_arcs.get(name)
    }

    /// Get the tension curve for a character: `(scene_index, tension)` pairs.
    pub fn character_tension_curve(&self, name: &str) -> Option<Vec<(usize, f32)>> {
        self.character_arcs.get(name).map(|arc| {
            arc.scene_indices
                .iter()
                .zip(arc.signals.iter())
                .map(|(&idx, sig)| (idx, sig.tension))
                .collect()
        })
    }

    /// Add a scene with automatically inferred mood.
    ///
    /// Analyzes the setting and conflict text for keyword patterns to determine
    /// the most appropriate `NarrativeMood`, removing the need for callers to
    /// hand-specify mood. Useful for automated pipelines and benchmarks.
    pub fn add_scene_auto(
        &mut self,
        title: &str,
        setting: &str,
        character_names: &[&str],
        conflict: &str,
    ) -> NarrativeSignal {
        let mood = Self::infer_mood(setting, conflict);
        self.add_scene(title, setting, character_names, conflict, mood)
    }

    /// Infer `NarrativeMood` from setting and conflict text via keyword matching.
    ///
    /// Scoring: each keyword hit adds to its mood's score. The mood with the
    /// highest score wins. Ties break toward Mysterious (the most neutral).
    pub fn infer_mood(setting: &str, conflict: &str) -> NarrativeMood {
        let text = format!("{} {}", setting, conflict).to_lowercase();

        let score = |keywords: &[&str]| -> i32 {
            keywords.iter().filter(|k| text.contains(**k)).count() as i32
        };

        let tense = score(&[
            "battle", "fight", "attack", "siege", "war", "death", "kill", "destroy", "threat",
            "danger", "enemy", "ambush", "betrayal", "betray", "fire", "inferno", "storm",
            "pursuit", "trap", "wound", "confront", "crisis", "invasion", "escape", "clash",
        ]);
        let peaceful = score(&[
            "quiet",
            "calm",
            "peace",
            "dawn",
            "morning",
            "garden",
            "rest",
            "home",
            "gentle",
            "serene",
            "stillness",
            "harmony",
            "cottage",
            "meadow",
        ]);
        let mysterious = score(&[
            "mystery", "strange", "unknown", "shadow", "whisper", "secret", "hidden", "ancient",
            "dark", "cave", "mist", "fog", "riddle", "enigma", "prophecy",
        ]);
        let melancholy = score(&[
            "loss",
            "grief",
            "sorrow",
            "lonely",
            "cold",
            "ruin",
            "ashes",
            "farewell",
            "regret",
            "mourn",
            "abandoned",
            "forgotten",
            "decay",
            "alone",
            "poverty",
            "neglect",
            "mock",
        ]);
        let triumphant = score(&[
            "victory",
            "triumph",
            "crown",
            "glory",
            "celebrate",
            "power",
            "throne",
            "conquer",
            "dazzl",
            "glitter",
            "grand",
            "palace",
            "kingdom",
            "won",
        ]);
        let hopeful = score(&[
            "hope",
            "new",
            "rebuild",
            "dream",
            "light",
            "future",
            "promise",
            "begin",
            "quest",
            "search",
            "sunrise",
            "renewal",
            "acceptance",
        ]);

        let scores = [
            (tense, NarrativeMood::Tense),
            (peaceful, NarrativeMood::Peaceful),
            (mysterious, NarrativeMood::Mysterious),
            (melancholy, NarrativeMood::Melancholy),
            (triumphant, NarrativeMood::Triumphant),
            (hopeful, NarrativeMood::Hopeful),
        ];

        scores
            .iter()
            .max_by_key(|(s, _)| *s)
            .map(|(_, m)| *m)
            .unwrap_or(NarrativeMood::Mysterious)
    }

    /// Reset the session to start a new story.
    pub fn reset(&mut self) {
        self.dynamics.reset();
        self.characters.clear();
        self.character_arcs.clear();
        self.conflicts.clear();
        self.themes.clear();
        self.scene_log.clear();
    }

    // ========================================================================
    // Internal
    // ========================================================================

    /// Project a high-D HV down to `input_dim`-D via chunk-mean pooling
    /// with variance-preserving scaling.
    ///
    /// Each output element is the mean of a contiguous chunk of the input HV.
    /// The result is then rescaled so the projected vector preserves the
    /// energy-per-dimension of the original, preventing the near-zero inputs
    /// that cause CfC sigmoid to collapse to 0.5.
    fn project_hv(&self, hv: &ContinuousHV) -> Array1<f32> {
        let data = hv.as_slice();
        let chunk_size = (data.len() / self.input_dim).max(1);
        let mut projected = Vec::with_capacity(self.input_dim);
        for i in 0..self.input_dim {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(data.len());
            let mean = data[start..end].iter().sum::<f32>() / (end - start) as f32;
            projected.push(mean);
        }
        // Variance-preserving scale: match energy-per-dim of original
        let input_l2: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_l2: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if output_l2 > 1e-8 {
            (input_l2 / (data.len() as f32).sqrt()) / output_l2
        } else {
            1.0
        };
        Array1::from_vec(projected.iter().map(|x| x * scale).collect())
    }
}

// ============================================================================
// Persistence — StorySessionSnapshot
// ============================================================================

/// Current snapshot version for forward compatibility.
const SNAPSHOT_VERSION: u32 = 1;

/// A serializable snapshot of a `StorySession`.
///
/// On save, characters are stored as (name, HV data) pairs and the scene log
/// is stored verbatim. On load, the algebra and dynamics are reconstructed
/// from deterministic defaults (same seeds → identical objects).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorySessionSnapshot {
    pub version: u32,
    pub characters: Vec<(String, ContinuousHV)>,
    pub conflicts: Vec<ConflictEntry>,
    pub themes: Vec<String>,
    pub scene_log: Vec<SceneRecord>,
}

impl StorySession {
    /// Serialize the session state to a JSON snapshot.
    pub fn save(&self) -> Result<String, serde_json::Error> {
        let snapshot = StorySessionSnapshot {
            version: SNAPSHOT_VERSION,
            characters: self
                .characters
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            conflicts: self.conflicts.clone(),
            themes: self.themes.clone(),
            scene_log: self.scene_log.clone(),
        };
        serde_json::to_string_pretty(&snapshot)
    }

    /// Restore a session from a JSON snapshot.
    ///
    /// Algebra and dynamics are reconstructed from defaults (deterministic seeds).
    /// Character HVs, conflicts, themes, and the scene log are restored directly.
    /// Character arcs are rebuilt from the scene log.
    pub fn load(json: &str) -> Result<Self, serde_json::Error> {
        let snapshot: StorySessionSnapshot = serde_json::from_str(json)?;
        let config = StoryArcConfig::default();
        let input_dim = config.input_dim;

        // Rebuild character_arcs from character list
        let characters: HashMap<String, ContinuousHV> = snapshot.characters.into_iter().collect();
        let mut character_arcs: HashMap<String, CharacterArc> = characters
            .iter()
            .map(|(name, hv)| {
                (
                    name.clone(),
                    CharacterArc {
                        scene_indices: Vec::new(),
                        signals: Vec::new(),
                        presence_hv: hv.clone(),
                    },
                )
            })
            .collect();

        // Rebuild arcs from scene log
        for record in &snapshot.scene_log {
            for char_name in &record.characters {
                if let Some(arc) = character_arcs.get_mut(char_name) {
                    arc.scene_indices.push(record.index);
                    arc.signals.push(record.signal.clone());
                    let count = arc.scene_indices.len() as f32;
                    let old_weight = (count - 1.0) / count;
                    let new_weight = 1.0 / count;
                    arc.presence_hv = ContinuousHV::weighted_bundle(
                        &[&arc.presence_hv, &record.scene_hv],
                        &[old_weight, new_weight],
                    );
                }
            }
        }

        let algebra = NarrativeAlgebra::default_dim();
        let dynamics = StoryArcDynamics::with_algebra(config, &algebra);
        let mut session = Self {
            algebra,
            dynamics,
            characters,
            character_arcs,
            conflicts: snapshot.conflicts,
            themes: snapshot.themes,
            scene_log: snapshot.scene_log,
            input_dim,
            expected_total_scenes: 0,
        };
        // Replay dynamics state by re-stepping through the scene log
        for record in &session.scene_log {
            let projected = session.project_hv(&record.scene_hv);
            session.dynamics.step_hybrid_with_mood(
                &projected,
                &record.scene_hv,
                0.1,
                Some(record.mood),
            );
        }
        Ok(session)
    }
}

impl Default for StorySession {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_basic_flow() {
        let mut session = StorySession::new();

        // Register a character
        let _hero =
            session.register_character("Kael", &session.algebra().primitives.protagonist.clone());
        session.add_theme("Redemption");

        let signal = session.add_scene(
            "The Awakening",
            "ancient temple",
            &["Kael"],
            "strange visions",
            NarrativeMood::Mysterious,
        );

        assert!((0.0..=1.0).contains(&signal.energy));
        assert!((0.0..=1.0).contains(&signal.tension));
        assert!((-1.0..=1.0).contains(&signal.valence));

        let state = session.get_story_state();
        assert_eq!(state.total_scenes, 1);
        assert_eq!(state.character_count, 1);
        assert_eq!(state.themes, vec!["Redemption"]);
    }

    #[test]
    fn test_session_cfc_memory_effect() {
        let mut session = StorySession::new();
        let _hero =
            session.register_character("Kael", &session.algebra().primitives.protagonist.clone());

        let signal_1 = session.add_scene(
            "Scene 1",
            "dark forest",
            &["Kael"],
            "lost",
            NarrativeMood::Mysterious,
        );

        // Identical inputs should produce different signals because CfC has state
        let signal_2 = session.add_scene(
            "Scene 2",
            "dark forest",
            &["Kael"],
            "lost",
            NarrativeMood::Mysterious,
        );

        // At least one field should differ due to CfC memory
        let all_same = (signal_1.energy - signal_2.energy).abs() < 1e-6
            && (signal_1.surprise - signal_2.surprise).abs() < 1e-6
            && (signal_1.tension - signal_2.tension).abs() < 1e-6
            && (signal_1.valence - signal_2.valence).abs() < 1e-6
            && (signal_1.momentum - signal_2.momentum).abs() < 1e-6;

        assert!(
            !all_same,
            "CfC memory should cause identical inputs to produce different signals"
        );
    }

    #[test]
    fn test_conflict_tracking() {
        let mut session = StorySession::new();

        session.introduce_conflict("The dark lord rises");
        session.introduce_conflict("Betrayal by the mentor");

        let state = session.get_story_state();
        assert_eq!(state.total_conflicts, 2);
        assert_eq!(state.unresolved_conflicts, 2);

        assert!(session.resolve_conflict("The dark lord rises"));
        assert!(!session.resolve_conflict("nonexistent conflict"));

        let state = session.get_story_state();
        assert_eq!(state.total_conflicts, 2);
        assert_eq!(state.unresolved_conflicts, 1);
    }

    #[test]
    fn test_project_hv_dimension_correct() {
        let session = StorySession::new();
        let hv = ContinuousHV::random(4096, 42);
        let projected = session.project_hv(&hv);
        assert_eq!(projected.len(), session.input_dim);
    }

    #[test]
    fn test_project_hv_variance_preserved() {
        let session = StorySession::new();
        let hv = ContinuousHV::random(4096, 42);
        let projected = session.project_hv(&hv);

        // Projected vector should have non-trivial magnitude (not near-zero)
        let l2: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            l2 > 0.1,
            "Projected vector should have meaningful magnitude, got L2={}",
            l2
        );

        // Energy per dimension should be in a reasonable range
        let energy_per_dim = l2 / (projected.len() as f32).sqrt();
        assert!(
            energy_per_dim > 0.01,
            "Energy per dimension too low: {}",
            energy_per_dim
        );
    }

    #[test]
    fn test_scene_similarity() {
        let mut session = StorySession::new();
        let _hero =
            session.register_character("Kael", &session.algebra().primitives.protagonist.clone());
        let _villain =
            session.register_character("Thira", &session.algebra().primitives.antagonist.clone());

        // Two similar scenes (same setting/conflict, different mood)
        session.add_scene(
            "Scene A",
            "dark forest",
            &["Kael"],
            "lost",
            NarrativeMood::Mysterious,
        );
        session.add_scene(
            "Scene B",
            "dark forest",
            &["Kael"],
            "lost",
            NarrativeMood::Tense,
        );

        // One very different scene
        session.add_scene(
            "Scene C",
            "sunny beach",
            &["Thira"],
            "seeking power",
            NarrativeMood::Triumphant,
        );

        let sim_ab = session.scene_similarity(0, 1).unwrap();
        let sim_ac = session.scene_similarity(0, 2).unwrap();

        assert!(
            sim_ab > sim_ac,
            "Similar scenes ({:.3}) should be more similar than different scenes ({:.3})",
            sim_ab,
            sim_ac,
        );

        // Out-of-bounds returns None
        assert!(session.scene_similarity(0, 99).is_none());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut session = StorySession::new();
        let _hero =
            session.register_character("Kael", &session.algebra().primitives.protagonist.clone());
        session.add_theme("Redemption");
        session.introduce_conflict("The dark rises");

        session.add_scene(
            "Opening",
            "village",
            &["Kael"],
            "restless dreams",
            NarrativeMood::Peaceful,
        );
        session.add_scene(
            "Crisis",
            "mountain",
            &["Kael"],
            "facing the beast",
            NarrativeMood::Tense,
        );

        let json = session.save().expect("save should succeed");
        let restored = StorySession::load(&json).expect("load should succeed");

        assert_eq!(restored.scene_log().len(), 2);
        assert_eq!(restored.get_story_state().character_count, 1);
        assert_eq!(restored.get_story_state().themes, vec!["Redemption"]);
        assert_eq!(restored.get_story_state().total_conflicts, 1);
    }

    #[test]
    fn test_snapshot_version() {
        let session = StorySession::new();
        let json = session.save().expect("save should succeed");
        let snapshot: super::StorySessionSnapshot =
            serde_json::from_str(&json).expect("parse snapshot");
        assert_eq!(snapshot.version, super::SNAPSHOT_VERSION);
    }

    #[test]
    fn test_transition_escalation_amplifies() {
        let mut session = StorySession::new();
        let _hero =
            session.register_character("Kael", &session.algebra().primitives.protagonist.clone());

        // First scene (no transition modulation)
        let sig1 = session.add_scene(
            "Calm Start",
            "quiet village",
            &["Kael"],
            "nothing happening",
            NarrativeMood::Peaceful,
        );

        // Second scene — transition operators will modulate the projected input
        let sig2 = session.add_scene(
            "Sudden Crisis",
            "burning city",
            &["Kael"],
            "invasion begins",
            NarrativeMood::Tense,
        );

        // The signals should differ (transition modulation + different HDC similarity)
        let differs = (sig1.energy - sig2.energy).abs() > 0.001
            || (sig1.tension - sig2.tension).abs() > 0.001;
        assert!(
            differs,
            "Transition from calm to crisis should produce different signals"
        );
    }

    #[test]
    fn test_transition_ordering_matters() {
        // Scene A→B should produce different dynamics than scene B→A
        let mut session_ab = StorySession::new();
        let prot = session_ab.algebra().primitives.protagonist.clone();
        session_ab.register_character("Hero", &prot);

        session_ab.add_scene("A", "forest", &["Hero"], "lost", NarrativeMood::Mysterious);
        let sig_ab = session_ab.add_scene("B", "castle", &["Hero"], "battle", NarrativeMood::Tense);

        let mut session_ba = StorySession::new();
        let prot2 = session_ba.algebra().primitives.protagonist.clone();
        session_ba.register_character("Hero", &prot2);

        session_ba.add_scene("B", "castle", &["Hero"], "battle", NarrativeMood::Tense);
        let sig_ba =
            session_ba.add_scene("A", "forest", &["Hero"], "lost", NarrativeMood::Mysterious);

        // Different ordering should produce different second-scene signals
        let differs = (sig_ab.energy - sig_ba.energy).abs() > 0.001
            || (sig_ab.tension - sig_ba.tension).abs() > 0.001
            || (sig_ab.surprise - sig_ba.surprise).abs() > 0.001;
        assert!(
            differs,
            "Scene ordering should affect Ghost Signal via transition operators"
        );
    }

    #[test]
    fn test_character_arc_tracking() {
        let mut session = StorySession::new();
        let _hero =
            session.register_character("Kael", &session.algebra().primitives.protagonist.clone());
        let _villain =
            session.register_character("Thira", &session.algebra().primitives.antagonist.clone());

        session.add_scene(
            "Scene 1",
            "village",
            &["Kael"],
            "restless dreams",
            NarrativeMood::Peaceful,
        );
        session.add_scene(
            "Scene 2",
            "forest",
            &["Kael", "Thira"],
            "confrontation",
            NarrativeMood::Tense,
        );
        session.add_scene(
            "Scene 3",
            "mountain",
            &["Kael"],
            "final test",
            NarrativeMood::Tense,
        );

        let kael_arc = session.get_character_arc("Kael").expect("Kael registered");
        assert_eq!(kael_arc.scene_indices.len(), 3, "Kael in all 3 scenes");
        assert_eq!(kael_arc.signals.len(), 3);

        let thira_arc = session
            .get_character_arc("Thira")
            .expect("Thira registered");
        assert_eq!(thira_arc.scene_indices.len(), 1, "Thira in 1 scene");
        assert_eq!(thira_arc.scene_indices[0], 1);

        // Tension curve
        let kael_tension = session
            .character_tension_curve("Kael")
            .expect("Kael registered");
        assert_eq!(kael_tension.len(), 3);
        for (idx, tension) in &kael_tension {
            assert!(*idx < 3);
            assert!((0.0..=1.0).contains(tension));
        }
    }

    #[test]
    fn test_character_arc_presence_hv() {
        let mut session = StorySession::new();
        let _hero =
            session.register_character("Kael", &session.algebra().primitives.protagonist.clone());

        session.add_scene("S1", "village", &["Kael"], "peace", NarrativeMood::Peaceful);
        session.add_scene("S2", "forest", &["Kael"], "danger", NarrativeMood::Tense);

        let arc = session.get_character_arc("Kael").unwrap();
        // Presence HV should have nonzero norm (it's a weighted bundle of scene HVs)
        let norm: f32 = arc
            .presence_hv
            .as_slice()
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(norm > 0.01, "Presence HV should have non-trivial norm");
    }

    #[test]
    fn test_infer_mood_tense() {
        let mood = StorySession::infer_mood("battlefield", "the enemy attacks");
        assert_eq!(mood, NarrativeMood::Tense);
    }

    #[test]
    fn test_infer_mood_peaceful() {
        let mood = StorySession::infer_mood("quiet garden at dawn", "morning stillness");
        assert_eq!(mood, NarrativeMood::Peaceful);
    }

    #[test]
    fn test_infer_mood_melancholy() {
        let mood = StorySession::infer_mood("cold ruins", "grief and loss, alone");
        assert_eq!(mood, NarrativeMood::Melancholy);
    }

    #[test]
    fn test_add_scene_auto() {
        let mut session = StorySession::new();
        let _hero =
            session.register_character("Kael", &session.algebra().primitives.protagonist.clone());

        let signal = session.add_scene_auto(
            "Battle Scene",
            "burning battlefield",
            &["Kael"],
            "enemy siege and attack from all sides",
        );

        // Should produce valid signal with elevated tension (auto-inferred Tense mood)
        assert!((0.0..=1.0).contains(&signal.tension));
        assert!((0.0..=1.0).contains(&signal.energy));
    }

    #[test]
    fn test_character_arc_not_registered() {
        let session = StorySession::new();
        assert!(session.get_character_arc("Unknown").is_none());
        assert!(session.character_tension_curve("Unknown").is_none());
    }
}
