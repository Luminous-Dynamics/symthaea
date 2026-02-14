//! Multi-Scene Story Session
//!
//! Wraps `NarrativeAlgebra` + `StoryArcDynamics` with stateful story management:
//! character registration, conflict tracking, scene logging, and HV→CfC projection.

use std::collections::HashMap;

use ndarray::Array1;
use symthaea_core::hdc::ContinuousHV;

use crate::dynamics::narrative_dynamics::{NarrativeSignal, StoryArcConfig, StoryArcDynamics};
use crate::hdc::narrative_algebra::{ArcPhase, NarrativeAlgebra, NarrativeMood};

// ============================================================================
// Types
// ============================================================================

/// A logged scene with its signal and HDC representation.
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
pub struct ConflictEntry {
    pub description: String,
    pub introduced_at: usize,
    pub resolved_at: Option<usize>,
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
    conflicts: Vec<ConflictEntry>,
    themes: Vec<String>,
    scene_log: Vec<SceneRecord>,
    input_dim: usize,
}

impl StorySession {
    /// Create a new story session with default configuration.
    pub fn new() -> Self {
        let config = StoryArcConfig::default();
        let input_dim = config.input_dim;
        Self {
            algebra: NarrativeAlgebra::default_dim(),
            dynamics: StoryArcDynamics::new(config),
            characters: HashMap::new(),
            conflicts: Vec::new(),
            themes: Vec::new(),
            scene_log: Vec::new(),
            input_dim,
        }
    }

    /// Create with a custom `StoryArcConfig`.
    pub fn with_config(config: StoryArcConfig) -> Self {
        let input_dim = config.input_dim;
        Self {
            algebra: NarrativeAlgebra::default_dim(),
            dynamics: StoryArcDynamics::new(config),
            characters: HashMap::new(),
            conflicts: Vec::new(),
            themes: Vec::new(),
            scene_log: Vec::new(),
            input_dim,
        }
    }

    /// Register a character and return its HDC encoding.
    pub fn register_character(
        &mut self,
        name: &str,
        role_primitive: &ContinuousHV,
    ) -> ContinuousHV {
        let hv = self.algebra.encode_character(name, role_primitive);
        self.characters.insert(name.to_string(), hv.clone());
        hv
    }

    /// Add a thematic throughline.
    pub fn add_theme(&mut self, theme: &str) {
        self.themes.push(theme.to_string());
    }

    /// Add a scene to the story, advancing CfC dynamics.
    ///
    /// Returns the resulting `NarrativeSignal` (Ghost Signal).
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
        let projected = self.project_hv(&scene_hv);

        // Advance dynamics
        let signal = self.dynamics.step(&projected, 0.1);

        let index = self.scene_log.len();
        self.scene_log.push(SceneRecord {
            index,
            title: title.to_string(),
            setting: setting.to_string(),
            conflict: conflict.to_string(),
            mood,
            characters: character_names.iter().map(|s| s.to_string()).collect(),
            signal: signal.clone(),
            scene_hv,
        });

        signal
    }

    /// Introduce a new conflict.
    pub fn introduce_conflict(&mut self, description: &str) {
        let scene_idx = self.scene_log.len();
        self.conflicts.push(ConflictEntry {
            description: description.to_string(),
            introduced_at: scene_idx,
            resolved_at: None,
        });
    }

    /// Resolve a conflict by description. Returns `true` if found and resolved.
    pub fn resolve_conflict(&mut self, description: &str) -> bool {
        let scene_idx = self.scene_log.len();
        for entry in &mut self.conflicts {
            if entry.description == description && entry.resolved_at.is_none() {
                entry.resolved_at = Some(scene_idx);
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

    /// Reset the session to start a new story.
    pub fn reset(&mut self) {
        self.dynamics.reset();
        self.characters.clear();
        self.conflicts.clear();
        self.themes.clear();
        self.scene_log.clear();
    }

    // ========================================================================
    // Internal
    // ========================================================================

    /// Project a 4096-D HV down to `input_dim`-D by stride sampling.
    ///
    /// Takes every `(dim / input_dim)`-th element to produce a compact
    /// representation suitable for the CfC input layer.
    fn project_hv(&self, hv: &ContinuousHV) -> Array1<f32> {
        let data = hv.as_slice();
        let stride = data.len() / self.input_dim;
        let stride = stride.max(1);
        let projected: Vec<f32> = (0..self.input_dim)
            .map(|i| data.get(i * stride).copied().unwrap_or(0.0))
            .collect();
        Array1::from_vec(projected)
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
}
