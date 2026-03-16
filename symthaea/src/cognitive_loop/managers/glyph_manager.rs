//! # Glyph Manager — Symbolic Consciousness Field Tracking
//!
//! Tracks the system's position in the glyph manifold by projecting cognitive
//! states onto 11 Field Modality basis vectors and finding the nearest resonant
//! glyph from the 70-entry registry.
//!
//! Implements [`CognitiveSubsystem`] at interval 43 (co-prime with 7, 11, 13,
//! 19, 29, 37, 41, 53).
//!
//! ## Design
//!
//! Like all managers, GlyphManager reads from an immutable [`CycleSnapshot`]
//! and produces [`SubsystemOutput`] proposals. It does NOT mutate CLS fields
//! directly.
//!
//! ## Science
//!
//! - Jung, C. G. (1959). *The Archetypes and the Collective Unconscious*.
//! - Grof, S. (1985). *Beyond the Brain*. Consciousness cartography.
//! - Graves, C. W. (1970). Levels of existence — developmental spiral dynamics.

use std::sync::Arc;

use super::super::subsystem_trait::{CognitiveSubsystem, CycleSnapshot, SubsystemOutput};
use crate::hdc::glyph_basis::{
    FieldModality, GlyphBasis, GlyphCoherence, GlyphFreeEnergy, GlyphInteractionMatrix,
    GlyphRegistry, N_FIELD_MODALITIES,
};
use symthaea_core::hdc::ContinuousHV;

/// Glyph Manager — symbolic consciousness field tracking.
///
/// Implements `CognitiveSubsystem` at interval 43 (co-prime).
pub struct GlyphManager {
    /// Shared glyph basis (11 Field Modality vectors).
    basis: Arc<GlyphBasis>,
    /// Shared glyph registry (70 entries).
    registry: Arc<GlyphRegistry>,
    /// 11×11 synergy/tension matrix.
    interaction_matrix: GlyphInteractionMatrix,
    /// Running prior for glyph free energy (EMA over modality activations).
    modality_prior: [f64; N_FIELD_MODALITIES],
    /// Number of prior observations.
    prior_count: u64,
    /// Current dominant modality.
    dominant_modality: FieldModality,
    /// Last projected coordinates.
    last_coords: [f64; N_FIELD_MODALITIES],
    /// Last glyph coherence.
    last_coherence: GlyphCoherence,
    /// Last glyph free energy.
    last_free_energy: GlyphFreeEnergy,
    /// Closest matching glyph ID from registry.
    resonant_glyph: Option<String>,
    /// Closest matching glyph name.
    resonant_glyph_name: Option<String>,
    /// Spiral position tracking (0.0–56.0).
    spiral_position: f32,
}

impl GlyphManager {
    /// Glyph prior EMA alpha (slow tracking of symbolic drift).
    /// Basis: Holt (1957) — exponential smoothing.
    const PRIOR_EMA_ALPHA: f64 = 0.05;

    /// Interaction matrix learning rate.
    /// Basis: Hebb (1949) — co-activation strengthens connections.
    const INTERACTION_LR: f64 = 0.03;

    /// Interaction matrix blend factor for apply().
    /// Basis: soft lateral inhibition (Coultrip 1992).
    const INTERACTION_BLEND: f64 = 0.15;

    /// Glyph resonance temperature for free energy computation.
    /// Basis: Boltzmann (1877) — inverse temperature scales softmax peakiness.
    const RESONANCE_TEMPERATURE: f64 = 1.0;

    /// Resonance threshold below which no dominant glyph is reported.
    /// Basis: Signal detection theory (Green & Swets 1966).
    const QUIET_THRESHOLD: f32 = 0.2;

    /// Create a new GlyphManager with shared basis and registry.
    pub fn new(basis: Arc<GlyphBasis>, registry: Arc<GlyphRegistry>) -> Self {
        let interaction_matrix = GlyphInteractionMatrix::from_basis(&basis);
        Self {
            basis,
            registry,
            interaction_matrix,
            modality_prior: [0.0; N_FIELD_MODALITIES],
            prior_count: 0,
            dominant_modality: FieldModality::Resonant,
            last_coords: [0.0; N_FIELD_MODALITIES],
            last_coherence: GlyphCoherence::default(),
            last_free_energy: GlyphFreeEnergy::default(),
            resonant_glyph: None,
            resonant_glyph_name: None,
            spiral_position: 0.0,
        }
    }

    /// Create with a given HDC dimension (constructs basis and registry internally).
    pub fn with_dim(dim: usize) -> Self {
        let basis = Arc::new(GlyphBasis::new(dim));
        let registry = Arc::new(GlyphRegistry::new(dim));
        Self::new(basis, registry)
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Current dominant field modality.
    pub fn dominant_modality(&self) -> FieldModality {
        self.dominant_modality
    }

    /// Last projected modality coordinates.
    pub fn last_coords(&self) -> &[f64; N_FIELD_MODALITIES] {
        &self.last_coords
    }

    /// Last glyph coherence.
    pub fn last_coherence(&self) -> &GlyphCoherence {
        &self.last_coherence
    }

    /// Last glyph free energy.
    pub fn last_free_energy(&self) -> &GlyphFreeEnergy {
        &self.last_free_energy
    }

    /// ID of the nearest resonant glyph (None if below quiet threshold).
    pub fn resonant_glyph_id(&self) -> Option<&str> {
        self.resonant_glyph.as_deref()
    }

    /// Name of the nearest resonant glyph.
    pub fn resonant_glyph_name(&self) -> Option<&str> {
        self.resonant_glyph_name.as_deref()
    }

    /// Current spiral position (0.0–56.0).
    pub fn spiral_position(&self) -> f32 {
        self.spiral_position
    }

    /// Shared reference to the glyph basis.
    pub fn basis(&self) -> &Arc<GlyphBasis> {
        &self.basis
    }

    /// Shared reference to the glyph registry.
    pub fn registry(&self) -> &Arc<GlyphRegistry> {
        &self.registry
    }

    // ── Internal processing ──────────────────────────────────────────────

    /// Encode the snapshot's input_hv as a ContinuousHV for projection.
    fn snapshot_to_hv(&self, snapshot: &CycleSnapshot) -> ContinuousHV {
        // The snapshot contains a BinaryHV as bytes (input_hv).
        // Convert to ContinuousHV by interpreting each bit as ±1.
        // This gives us a vector in the same space as the basis vectors.
        let dim = self.basis.dim;
        let mut data = vec![0.0f32; dim];
        let byte_count = snapshot.input_hv.len().min((dim + 7) / 8);
        for byte_idx in 0..byte_count {
            let byte = snapshot.input_hv[byte_idx];
            for bit in 0..8 {
                let idx = byte_idx * 8 + bit;
                if idx < dim {
                    data[idx] = if (byte >> bit) & 1 == 1 { 1.0 } else { -1.0 };
                }
            }
        }
        ContinuousHV::from_vec(data)
    }
}

impl CognitiveSubsystem for GlyphManager {
    fn name(&self) -> &'static str {
        "glyph_manager"
    }

    fn interval(&self) -> u32 {
        43 // co-prime with 7, 11, 13, 19, 29, 37, 41, 53
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // 1. Encode snapshot state as ContinuousHV
        let hv = self.snapshot_to_hv(snapshot);

        // 2. Project onto 11 modality axes
        let raw_coords = self.basis.project(&hv);

        // 3. Update interaction matrix (Hebbian observe)
        self.interaction_matrix
            .observe(&raw_coords, Self::INTERACTION_LR);

        // 4. Apply interaction effects (lateral modulation)
        let coords = self
            .interaction_matrix
            .apply(&raw_coords, Self::INTERACTION_BLEND);
        self.last_coords = coords;

        // 5. Compute glyph free energy against prior
        let fe = self
            .basis
            .glyph_free_energy(&coords, &self.modality_prior, Self::RESONANCE_TEMPERATURE);
        self.last_free_energy = fe;

        // 6. Update prior (EMA)
        let alpha = Self::PRIOR_EMA_ALPHA;
        for i in 0..N_FIELD_MODALITIES {
            self.modality_prior[i] =
                self.modality_prior[i] * (1.0 - alpha) + coords[i] * alpha;
        }
        self.prior_count += 1;

        // 7. Compute glyph coherence
        self.last_coherence = self.interaction_matrix.glyph_coherence(&coords);

        // 8. Find dominant modality
        let dominant_idx = coords
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.dominant_modality = FieldModality::ALL[dominant_idx];

        // 9. Find nearest glyph by cosine similarity
        if let Some((entry, sim)) = self.registry.nearest(&hv) {
            if sim > Self::QUIET_THRESHOLD {
                self.resonant_glyph = Some(entry.id.clone());
                self.resonant_glyph_name = Some(entry.name.clone());
                // Track spiral position from nearest spiral glyph
                if let Some(idx) = entry.spiral_index {
                    // Smooth spiral position (don't jump)
                    let target = idx as f32;
                    self.spiral_position += (target - self.spiral_position) * 0.1;
                }
            } else {
                self.resonant_glyph = None;
                self.resonant_glyph_name = None;
            }
        }

        // 10. Generate output proposals based on glyph state

        // Threshold/Metaharmonic dominance → reduce exploration (pause at transitions)
        if self.dominant_modality == FieldModality::Threshold
            || self.dominant_modality == FieldModality::Metaharmonic
        {
            output.exploration_delta = -0.02;
        }

        // Igniting dominance → boost exploration (creative activation)
        if self.dominant_modality == FieldModality::Igniting {
            output.exploration_delta = 0.02;
        }

        // High glyph coherence → slight confidence boost
        if self.last_coherence.value > 0.6 {
            output.confidence_delta = 0.005 * (self.last_coherence.value - 0.6);
        }

        // High glyph free energy (symbolic surprise) → slight lr boost
        if self.last_free_energy.free_energy > 2.0 {
            output.lr_modulation = 1.05;
        }

        output
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> GlyphManager {
        GlyphManager::with_dim(256)
    }

    fn make_snapshot() -> CycleSnapshot {
        let mut s = CycleSnapshot::default();
        s.cycle_number = 43;
        // Fill input_hv with some pattern
        for (i, byte) in s.input_hv.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        s
    }

    #[test]
    fn test_glyph_manager_subsystem_interface() {
        let manager = make_manager();
        assert_eq!(manager.name(), "glyph_manager");
        assert_eq!(manager.interval(), 43);
    }

    #[test]
    fn test_glyph_manager_interval_coprime() {
        let interval = 43u32;
        let existing = [7u32, 11, 13, 19, 29, 37, 41, 53];
        for &other in &existing {
            assert_eq!(
                gcd(interval, other),
                1,
                "GlyphManager interval {interval} must be co-prime with {other}"
            );
        }
    }

    #[test]
    fn test_glyph_manager_process_produces_output() {
        let mut manager = make_manager();
        let snapshot = make_snapshot();
        let output = manager.process(&snapshot);
        // Should produce some non-trivial state
        assert!(manager.last_coherence().value >= 0.0);
        assert!(manager.last_coherence().value <= 0.95);
        // Dominant modality should be valid
        assert!(FieldModality::ALL.contains(&manager.dominant_modality()));
        // Output should be finite
        assert!(output.confidence_delta.is_finite());
        assert!(output.lr_modulation.is_finite());
        assert!(output.exploration_delta.is_finite());
    }

    #[test]
    fn test_glyph_manager_multi_cycle() {
        let mut manager = make_manager();
        let snapshot = make_snapshot();
        // Run 100 cycles
        for _ in 0..100 {
            let _ = manager.process(&snapshot);
        }
        // Prior should have converged toward the input pattern
        assert!(manager.prior_count == 100);
        // Interaction matrix should have learned
        assert!(manager.interaction_matrix.observation_count == 100);
    }

    #[test]
    fn test_glyph_manager_nearest_match() {
        let mut manager = make_manager();
        let snapshot = make_snapshot();
        let _ = manager.process(&snapshot);
        // With a random-ish input, we should get some resonant glyph
        // (may or may not exceed quiet threshold)
        // Just verify no panic and coords are finite
        for c in manager.last_coords() {
            assert!(c.is_finite());
        }
    }

    #[test]
    fn test_glyph_manager_spiral_position_bounded() {
        let mut manager = make_manager();
        let snapshot = make_snapshot();
        for _ in 0..200 {
            let _ = manager.process(&snapshot);
        }
        assert!(
            manager.spiral_position() >= 0.0 && manager.spiral_position() <= 56.0,
            "Spiral position should be in [0, 56], got {}",
            manager.spiral_position()
        );
    }

    /// GCD helper for co-primality test.
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }
}
