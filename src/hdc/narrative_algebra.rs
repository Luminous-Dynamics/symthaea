// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Narrative Algebra in Hyperdimensional Space
//!
//! Compositional narrative reasoning using HDC primitives, mirroring the
//! architecture of `moral_algebra.rs`. Instead of moral roles and rules, we
//! define **narrative primitives** (protagonist, antagonist, setting, ...) and
//! **narrative operators** (causes, transforms_into, escalates, ...) that
//! compose into structured scene representations.
//!
//! # Design Philosophy
//!
//! Traditional story-generation systems treat narrative structure as discrete
//! tokens or templates. This module encodes narrative elements as continuous
//! hypervectors so that:
//! - Scene similarity is a cosine distance (coherence checking)
//! - Operators are non-commutative bindings (A causes B ≠ B causes A)
//! - Arc phases, tension levels, and moods are first-class algebraic objects
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
//! │  SCENE DATA  │───▶│  HDC NARR.   │───▶│  DYNAMICS    │
//! │  (elements)  │    │   ALGEBRA    │    │  (CfC arc)   │
//! └──────────────┘    └──────────────┘    └──────────────┘
//!       │                   │                   │
//!       ▼                   ▼                   ▼
//!  Characters,         Bound HVs            Ghost Signal
//!  setting, etc.
//! ```

use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

/// Default dimension for narrative hypervectors (matches MORAL_DIM)
pub const NARRATIVE_DIM: usize = 4096;

// ============================================================================
// Narrative Primitives
// ============================================================================

/// The seven semantic role primitives for narrative reasoning.
///
/// These are the "nouns" of our narrative algebra — they represent the
/// structural roles that elements play in a scene.
#[derive(Debug, Clone)]
pub struct NarrativePrimitives {
    /// Dimension of all hypervectors
    pub dim: usize,

    /// PROTAGONIST — the central character driving the scene
    pub protagonist: ContinuousHV,

    /// ANTAGONIST — the opposing force
    pub antagonist: ContinuousHV,

    /// SETTING — where/when the scene takes place
    pub setting: ContinuousHV,

    /// CONFLICT — the central tension or problem
    pub conflict: ContinuousHV,

    /// THEME — the abstract idea explored
    pub theme: ContinuousHV,

    /// MOOD — the emotional atmosphere
    pub mood: ContinuousHV,

    /// STAKES — what is at risk
    pub stakes: ContinuousHV,
}

impl NarrativePrimitives {
    /// Create a new set of narrative primitives with deterministic seeds.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            protagonist: ContinuousHV::random(dim, 8_000_003),
            antagonist: ContinuousHV::random(dim, 8_000_017),
            setting: ContinuousHV::random(dim, 8_000_029),
            conflict: ContinuousHV::random(dim, 8_000_039),
            theme: ContinuousHV::random(dim, 8_000_053),
            mood: ContinuousHV::random(dim, 8_000_063),
            stakes: ContinuousHV::random(dim, 8_000_077),
        }
    }

    /// Create with default dimension (4096)
    pub fn default_dim() -> Self {
        Self::new(NARRATIVE_DIM)
    }

    /// Verify that primitives are approximately orthogonal.
    /// Returns the maximum pairwise similarity.
    pub fn verify_orthogonality(&self) -> f32 {
        let primitives = [
            &self.protagonist,
            &self.antagonist,
            &self.setting,
            &self.conflict,
            &self.theme,
            &self.mood,
            &self.stakes,
        ];

        let mut max_similarity = 0.0f32;
        for (i, a) in primitives.iter().enumerate() {
            for b in primitives.iter().skip(i + 1) {
                let sim = a.similarity(b).abs();
                if sim > max_similarity {
                    max_similarity = sim;
                }
            }
        }
        max_similarity
    }
}

// ============================================================================
// Narrative Operators
// ============================================================================

/// The five compositional operators for narrative reasoning.
///
/// These are the "verbs" of our narrative algebra — they define how
/// narrative elements combine to form story structures.
#[derive(Debug, Clone)]
pub struct NarrativeOperators {
    /// Dimension of all hypervectors
    pub dim: usize,

    /// CAUSES — causal relationship (A causes B)
    pub causes: ContinuousHV,

    /// TRANSFORMS_INTO — character or situation change (A becomes B)
    pub transforms_into: ContinuousHV,

    /// CONTRASTS_WITH — thematic opposition (A vs B)
    pub contrasts_with: ContinuousHV,

    /// ESCALATES — raising tension (event + stakes)
    pub escalates: ContinuousHV,

    /// RESOLVES — releasing tension (conflict + resolution)
    pub resolves: ContinuousHV,
}

impl NarrativeOperators {
    /// Create a new set of narrative operators with deterministic seeds.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            causes: ContinuousHV::random(dim, 9_000_003),
            transforms_into: ContinuousHV::random(dim, 9_000_017),
            contrasts_with: ContinuousHV::random(dim, 9_000_029),
            escalates: ContinuousHV::random(dim, 9_000_039),
            resolves: ContinuousHV::random(dim, 9_000_053),
        }
    }

    /// Create with default dimension
    pub fn default_dim() -> Self {
        Self::new(NARRATIVE_DIM)
    }
}

// ============================================================================
// Enums
// ============================================================================

/// Phase of a narrative arc (five-act structure)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ArcPhase {
    /// Exposition / world-building
    Setup,
    /// Complications and escalation
    RisingAction,
    /// The peak moment of conflict
    Climax,
    /// Consequences unfold
    FallingAction,
    /// New equilibrium
    Resolution,
}

/// Tension level within a scene or beat
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TensionLevel {
    /// Peaceful, no conflict
    Calm,
    /// Something is off, foreshadowing
    Building,
    /// Active conflict
    High,
    /// Maximum intensity
    Peak,
    /// Catharsis, aftermath
    Release,
}

/// Emotional mood / atmosphere of a scene
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum NarrativeMood {
    /// Optimistic, light
    Hopeful,
    /// Suspenseful, anxious
    Tense,
    /// Sad, wistful
    Melancholy,
    /// Victorious, uplifting
    Triumphant,
    /// Enigmatic, uncanny
    Mysterious,
    /// Serene, still
    Peaceful,
}

// ============================================================================
// Narrative Algebra Engine
// ============================================================================

/// The main narrative algebra engine.
///
/// Provides methods to compose narrative scenes from primitives and operators,
/// then measure structural similarity between scenes.
#[derive(Debug, Clone)]
pub struct NarrativeAlgebra {
    /// Semantic role primitives
    pub primitives: NarrativePrimitives,

    /// Compositional operators
    pub operators: NarrativeOperators,

    /// Arc-phase hypervectors
    phase_hvs: HashMap<ArcPhase, ContinuousHV>,

    /// Tension-level hypervectors
    tension_hvs: HashMap<TensionLevel, ContinuousHV>,

    /// Mood hypervectors
    mood_hvs: HashMap<NarrativeMood, ContinuousHV>,

    /// Dimension
    dim: usize,
}

impl NarrativeAlgebra {
    /// Create a new narrative algebra engine.
    pub fn new(dim: usize) -> Self {
        let primitives = NarrativePrimitives::new(dim);
        let operators = NarrativeOperators::new(dim);

        // Arc phase HVs (seed range 10_000_000+)
        let mut phase_hvs = HashMap::new();
        phase_hvs.insert(ArcPhase::Setup, ContinuousHV::random(dim, 10_000_003));
        phase_hvs.insert(
            ArcPhase::RisingAction,
            ContinuousHV::random(dim, 10_000_017),
        );
        phase_hvs.insert(ArcPhase::Climax, ContinuousHV::random(dim, 10_000_029));
        phase_hvs.insert(
            ArcPhase::FallingAction,
            ContinuousHV::random(dim, 10_000_039),
        );
        phase_hvs.insert(ArcPhase::Resolution, ContinuousHV::random(dim, 10_000_053));

        // Tension level HVs (seed range 11_000_000+)
        let mut tension_hvs = HashMap::new();
        tension_hvs.insert(TensionLevel::Calm, ContinuousHV::random(dim, 11_000_003));
        tension_hvs.insert(
            TensionLevel::Building,
            ContinuousHV::random(dim, 11_000_017),
        );
        tension_hvs.insert(TensionLevel::High, ContinuousHV::random(dim, 11_000_029));
        tension_hvs.insert(TensionLevel::Peak, ContinuousHV::random(dim, 11_000_039));
        tension_hvs.insert(TensionLevel::Release, ContinuousHV::random(dim, 11_000_053));

        // Mood HVs (seed range 12_000_000+)
        let mut mood_hvs = HashMap::new();
        mood_hvs.insert(
            NarrativeMood::Hopeful,
            ContinuousHV::random(dim, 12_000_003),
        );
        mood_hvs.insert(NarrativeMood::Tense, ContinuousHV::random(dim, 12_000_017));
        mood_hvs.insert(
            NarrativeMood::Melancholy,
            ContinuousHV::random(dim, 12_000_029),
        );
        mood_hvs.insert(
            NarrativeMood::Triumphant,
            ContinuousHV::random(dim, 12_000_039),
        );
        mood_hvs.insert(
            NarrativeMood::Mysterious,
            ContinuousHV::random(dim, 12_000_053),
        );
        mood_hvs.insert(
            NarrativeMood::Peaceful,
            ContinuousHV::random(dim, 12_000_067),
        );

        Self {
            primitives,
            operators,
            phase_hvs,
            tension_hvs,
            mood_hvs,
            dim,
        }
    }

    /// Create with default dimension (4096)
    pub fn default_dim() -> Self {
        Self::new(NARRATIVE_DIM)
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    // ========================================================================
    // Primitive Encoding
    // ========================================================================

    /// Encode a character by name, bound to a role primitive.
    ///
    /// `encode_character("Hamlet", protagonist)` = PROTAGONIST ⊗ hash("Hamlet")
    pub fn encode_character(&self, name: &str, role_primitive: &ContinuousHV) -> ContinuousHV {
        let name_hv = self.hash_string(name);
        role_primitive.bind(&name_hv)
    }

    /// Encode a scene from its constituent elements.
    ///
    /// `scene = SETTING(place) ⊕ characters ⊕ CONFLICT(problem) ⊕ MOOD(atmosphere)`
    ///
    /// Uses bundling (superposition) so all elements are recoverable.
    pub fn encode_scene(
        &self,
        setting_desc: &str,
        characters: &[ContinuousHV],
        conflict_desc: &str,
        mood: NarrativeMood,
    ) -> ContinuousHV {
        let setting_hv = self
            .primitives
            .setting
            .bind(&self.hash_string(setting_desc));
        let conflict_hv = self
            .primitives
            .conflict
            .bind(&self.hash_string(conflict_desc));
        let mood_hv = self
            .primitives
            .mood
            .bind(self.mood_hvs.get(&mood).expect("map covers all variants"));

        // Bundle all elements
        let mut components = vec![setting_hv, conflict_hv, mood_hv];
        components.extend(characters.iter().cloned());

        let refs: Vec<&ContinuousHV> = components.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Encode an arc phase as a hypervector.
    pub fn encode_arc_phase(&self, phase: ArcPhase) -> ContinuousHV {
        self.phase_hvs
            .get(&phase)
            .expect("map covers all variants")
            .clone()
    }

    /// Encode a tension level as a hypervector.
    pub fn encode_tension(&self, level: TensionLevel) -> ContinuousHV {
        self.tension_hvs
            .get(&level)
            .expect("map covers all variants")
            .clone()
    }

    /// Encode a mood as a hypervector.
    pub fn encode_mood(&self, mood: NarrativeMood) -> ContinuousHV {
        self.mood_hvs
            .get(&mood)
            .expect("map covers all variants")
            .clone()
    }

    // ========================================================================
    // Operator Composition
    // ========================================================================

    /// Compose: A CAUSES B (causal chain)
    ///
    /// Non-commutative: A causes B ≠ B causes A.
    /// The effect argument is permuted (circular shift) before binding,
    /// which is the standard HDC trick for encoding argument position.
    pub fn causes(&self, cause: &ContinuousHV, effect: &ContinuousHV) -> ContinuousHV {
        cause.bind(&self.operators.causes).bind(&effect.permute(1))
    }

    /// Compose: A TRANSFORMS_INTO B (character/situation arc)
    ///
    /// Non-commutative: encodes before-state → after-state directionality.
    pub fn transforms_into(&self, before: &ContinuousHV, after: &ContinuousHV) -> ContinuousHV {
        before
            .bind(&self.operators.transforms_into)
            .bind(&after.permute(1))
    }

    /// Compose: A CONTRASTS_WITH B (thematic opposition)
    ///
    /// Non-commutative for consistent representation.
    pub fn contrasts_with(&self, a: &ContinuousHV, b: &ContinuousHV) -> ContinuousHV {
        a.bind(&self.operators.contrasts_with).bind(&b.permute(1))
    }

    /// Compose: EVENT ESCALATES STAKES (raise tension)
    ///
    /// Non-commutative: event position is distinct from stakes position.
    pub fn escalates(&self, event: &ContinuousHV, stakes: &ContinuousHV) -> ContinuousHV {
        event
            .bind(&self.operators.escalates)
            .bind(&stakes.permute(1))
    }

    /// Compose: CONFLICT RESOLVES RESOLUTION (release tension)
    ///
    /// Non-commutative: conflict position is distinct from resolution position.
    pub fn resolves(&self, conflict: &ContinuousHV, resolution: &ContinuousHV) -> ContinuousHV {
        conflict
            .bind(&self.operators.resolves)
            .bind(&resolution.permute(1))
    }

    // ========================================================================
    // Similarity / Coherence
    // ========================================================================

    /// Compute cosine similarity between two scene encodings.
    ///
    /// Useful for coherence checking: scenes in the same story should have
    /// moderate similarity (shared elements) but not identical (progression).
    pub fn scene_similarity(&self, scene_a: &ContinuousHV, scene_b: &ContinuousHV) -> f32 {
        scene_a.similarity(scene_b)
    }

    // ========================================================================
    // Internal
    // ========================================================================

    /// Deterministic string → hypervector mapping.
    fn hash_string(&self, s: &str) -> ContinuousHV {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        let seed = hasher.finish();

        ContinuousHV::random(self.dim, seed)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitives_orthogonal() {
        let primitives = NarrativePrimitives::default_dim();
        let max_sim = primitives.verify_orthogonality();
        assert!(
            max_sim < 0.15,
            "Primitives not orthogonal: max_sim = {}",
            max_sim
        );
    }

    #[test]
    fn test_operators_orthogonal() {
        let ops = NarrativeOperators::default_dim();
        let prims = NarrativePrimitives::default_dim();

        let all: Vec<&ContinuousHV> = vec![
            &ops.causes,
            &ops.transforms_into,
            &ops.contrasts_with,
            &ops.escalates,
            &ops.resolves,
            // Also check operators are orthogonal to primitives
            &prims.protagonist,
            &prims.antagonist,
            &prims.setting,
            &prims.conflict,
            &prims.theme,
            &prims.mood,
            &prims.stakes,
        ];

        let mut max_sim = 0.0f32;
        for (i, a) in all.iter().enumerate() {
            for b in all.iter().skip(i + 1) {
                let sim = a.similarity(b).abs();
                if sim > max_sim {
                    max_sim = sim;
                }
            }
        }
        assert!(
            max_sim < 0.15,
            "Operators/primitives not orthogonal: max_sim = {}",
            max_sim
        );
    }

    #[test]
    fn test_character_encoding() {
        let algebra = NarrativeAlgebra::default_dim();

        let hamlet = algebra.encode_character("Hamlet", &algebra.primitives.protagonist);
        let macbeth = algebra.encode_character("Macbeth", &algebra.primitives.protagonist);

        // Same role, different names → different vectors
        let sim = hamlet.similarity(&macbeth);
        assert!(
            sim < 0.5,
            "Different characters in same role should be distinguishable: sim = {}",
            sim
        );

        // Same character encoded twice → identical
        let hamlet2 = algebra.encode_character("Hamlet", &algebra.primitives.protagonist);
        let self_sim = hamlet.similarity(&hamlet2);
        assert!(
            self_sim > 0.99,
            "Same character should reproduce: sim = {}",
            self_sim
        );
    }

    #[test]
    fn test_scene_composition() {
        let algebra = NarrativeAlgebra::default_dim();

        let hero = algebra.encode_character("Hero", &algebra.primitives.protagonist);

        let scene_a = algebra.encode_scene(
            "dark forest",
            &[hero.clone()],
            "lost and alone",
            NarrativeMood::Mysterious,
        );
        let scene_b = algebra.encode_scene(
            "dark forest",
            &[hero],
            "lost and alone",
            NarrativeMood::Mysterious,
        );

        let sim = scene_a.similarity(&scene_b);
        assert!(
            sim > 0.99,
            "Same scene composed twice should be near-identical: sim = {}",
            sim
        );
    }

    #[test]
    fn test_different_scenes_dissimilar() {
        let algebra = NarrativeAlgebra::default_dim();

        let hero = algebra.encode_character("Hero", &algebra.primitives.protagonist);
        let villain = algebra.encode_character("Villain", &algebra.primitives.antagonist);

        let scene_a = algebra.encode_scene(
            "dark forest",
            &[hero],
            "lost and alone",
            NarrativeMood::Mysterious,
        );
        let scene_b = algebra.encode_scene(
            "sunny beach",
            &[villain],
            "seeking revenge",
            NarrativeMood::Tense,
        );

        let sim = algebra.scene_similarity(&scene_a, &scene_b);
        assert!(
            sim < 0.5,
            "Structurally different scenes should have low similarity: sim = {}",
            sim
        );
    }

    #[test]
    fn test_causes_operator() {
        let algebra = NarrativeAlgebra::default_dim();

        let event_a = algebra.hash_string("betrayal");
        let event_b = algebra.hash_string("war");

        let a_causes_b = algebra.causes(&event_a, &event_b);
        let b_causes_a = algebra.causes(&event_b, &event_a);

        // Causal direction matters: A→B ≠ B→A
        let sim = a_causes_b.similarity(&b_causes_a);
        assert!(
            sim < 0.5,
            "causes() should be non-commutative: sim = {}",
            sim
        );
    }

    #[test]
    fn test_arc_phase_distinct() {
        let algebra = NarrativeAlgebra::default_dim();

        let phases = [
            ArcPhase::Setup,
            ArcPhase::RisingAction,
            ArcPhase::Climax,
            ArcPhase::FallingAction,
            ArcPhase::Resolution,
        ];

        for (i, &phase_a) in phases.iter().enumerate() {
            for &phase_b in phases.iter().skip(i + 1) {
                let hv_a = algebra.encode_arc_phase(phase_a);
                let hv_b = algebra.encode_arc_phase(phase_b);
                let sim = hv_a.similarity(&hv_b).abs();
                assert!(
                    sim < 0.15,
                    "Arc phases {:?} and {:?} not distinct: sim = {}",
                    phase_a,
                    phase_b,
                    sim
                );
            }
        }
    }
}
