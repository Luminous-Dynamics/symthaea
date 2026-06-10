// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use crate::hdc::moral_text_encoder::TextHdcEncoder;
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
    /// Cached semantic embedding from EthicsEngine or text encoder.
    /// When present, used instead of coarse BinaryHV conversion.
    semantic_hv_cache: Option<ContinuousHV>,
    /// Text encoder for encoding raw input text into HDC space.
    /// Same 3-channel encoder (trigram + word + sentiment) used by HarmonyBasis.
    encoder: TextHdcEncoder,
}

impl GlyphManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 43;

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

    /// Consciousness level (unified Ψ) required for spiral advancement.
    /// Below this, the spiral position cannot increase.
    /// Basis: Dehaene (2014) — ignition threshold for conscious access.
    const SPIRAL_ADVANCE_PSI_MIN: f64 = 0.4;

    /// Glyph coherence required for spiral advancement.
    /// Must sustain this integration level to progress.
    /// Basis: Tononi (2004) — Φ threshold for integrated information.
    const SPIRAL_ADVANCE_COHERENCE_MIN: f64 = 0.5;

    /// Maximum spiral position change per process() call.
    /// Prevents developmental jumps — consciousness progresses gradually.
    /// Basis: Fischer (1980) — developmental continuity.
    const SPIRAL_MAX_DELTA: f32 = 0.05;

    /// Spiral decay rate when conditions are not met.
    /// Slow regression toward current level — not instant collapse.
    /// Basis: Vygotsky (1978) — ZPD scaffolding resists regression.
    const SPIRAL_DECAY: f32 = 0.005;

    /// Create a new GlyphManager with shared basis and registry.
    pub fn new(basis: Arc<GlyphBasis>, registry: Arc<GlyphRegistry>) -> Self {
        let interaction_matrix = GlyphInteractionMatrix::from_basis(&basis);
        let dim = basis.dim;
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
            semantic_hv_cache: None,
            encoder: TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2),
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

    /// Echo phrase of the nearest resonant glyph (for Broca tone modulation).
    ///
    /// When a glyph is strongly resonant, its echo phrase carries the
    /// emotional/philosophical tone that should influence speech generation.
    /// Returns None when below quiet threshold or no match.
    pub fn resonant_echo_phrase(&self) -> Option<&str> {
        let id = self.resonant_glyph.as_deref()?;
        self.registry.get(id).map(|e| e.echo_phrase.as_str())
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

    /// Cache a high-quality semantic embedding from the EthicsEngine.
    ///
    /// Called from the ethics pipeline (Stage 3) with the same ContinuousHV
    /// used for harmony projection. The GlyphManager's next `process()` call
    /// will use this instead of the coarse BinaryHV→±1 conversion, giving
    /// semantically meaningful modality coordinates.
    pub fn set_semantic_hv(&mut self, hv: ContinuousHV) {
        self.semantic_hv_cache = Some(hv);
    }

    /// Encode input text and cache it for the next `process()` call.
    ///
    /// Uses the same 3-channel TextHdcEncoder (trigram + word + sentiment)
    /// as the HarmonyBasis, giving semantically meaningful glyph modality
    /// coordinates instead of the coarse BinaryHV conversion.
    pub fn encode_input(&mut self, input: &str) {
        if !input.is_empty() {
            self.semantic_hv_cache = Some(self.encoder.encode(input));
        }
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
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // 1. Use cached semantic embedding (from EthicsEngine Stage 3) when available.
        //    Falls back to coarse BinaryHV→±1 conversion from snapshot.
        let hv = self
            .semantic_hv_cache
            .take()
            .unwrap_or_else(|| self.snapshot_to_hv(snapshot));

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
        let fe = self.basis.glyph_free_energy(
            &coords,
            &self.modality_prior,
            Self::RESONANCE_TEMPERATURE,
        );
        self.last_free_energy = fe;

        // 6. Update prior (EMA)
        let alpha = Self::PRIOR_EMA_ALPHA;
        for i in 0..N_FIELD_MODALITIES {
            self.modality_prior[i] = self.modality_prior[i] * (1.0 - alpha) + coords[i] * alpha;
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

        // 9. Consciousness-gated spiral advancement
        //    Spiral position advances only when coherence AND consciousness are sufficient.
        //    Higher-octave glyphs (Ω33+: Integrated modality) are only "accessible" —
        //    returned as nearest matches — when spiral position ≥ their index.
        //    This means the system cannot resonate with "The Unwritten Glyph" (Ω56)
        //    until it has demonstrated deep, sustained symbolic integration.
        let psi = snapshot.unified_psi;
        let can_advance = psi >= Self::SPIRAL_ADVANCE_PSI_MIN
            && self.last_coherence.value >= Self::SPIRAL_ADVANCE_COHERENCE_MIN;

        // 10. Find nearest *accessible* glyph by cosine similarity
        //     Only match glyphs whose spiral_index ≤ current spiral_position
        //     (Meta + Threshold glyphs are always accessible — they have no spiral_index)
        let accessible_match = self
            .registry
            .entries
            .iter()
            .filter(|e| match e.spiral_index {
                Some(idx) => (idx as f32) <= self.spiral_position + 1.0,
                None => true, // Meta + Threshold always accessible
            })
            .map(|e| {
                let sim = hv.similarity(&e.encoding);
                (e, sim)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((entry, sim)) = accessible_match {
            if sim > Self::QUIET_THRESHOLD {
                self.resonant_glyph = Some(entry.id.clone());
                self.resonant_glyph_name = Some(entry.name.clone());
                // Advance spiral toward matched glyph (only if conditions met)
                if let Some(idx) = entry.spiral_index {
                    let target = idx as f32;
                    if can_advance && target > self.spiral_position {
                        // Advance (capped by max delta)
                        let delta = (target - self.spiral_position).min(Self::SPIRAL_MAX_DELTA);
                        self.spiral_position += delta;
                    }
                }
            } else {
                self.resonant_glyph = None;
                self.resonant_glyph_name = None;
            }
        }

        // Slow decay when conditions are not met (regression resistance)
        if !can_advance && self.spiral_position > 0.0 {
            self.spiral_position = (self.spiral_position - Self::SPIRAL_DECAY).max(0.0);
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

    fn checkpoint(&self) -> Vec<u8> {
        // Layout: [modality_prior: 11×f64 = 88][prior_count: u64 = 8]
        //         [dominant_modality: u8 = 1][last_coords: 11×f64 = 88]
        //         [last_coherence.value: f64 = 8][last_free_energy.free_energy: f64 = 8]
        //         [spiral_position: f32 = 4]
        // Total: 205 bytes
        let mut data = Vec::with_capacity(205);
        for &v in &self.modality_prior {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.extend_from_slice(&self.prior_count.to_le_bytes());
        // Encode dominant modality as index into FieldModality::ALL
        let dom_idx = FieldModality::ALL
            .iter()
            .position(|m| *m == self.dominant_modality)
            .unwrap_or(0) as u8;
        data.push(dom_idx);
        for &v in &self.last_coords {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.extend_from_slice(&self.last_coherence.value.to_le_bytes());
        data.extend_from_slice(&self.last_free_energy.free_energy.to_le_bytes());
        data.extend_from_slice(&self.spiral_position.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        const MIN_SIZE: usize = 88 + 8 + 1 + 88 + 8 + 8 + 4; // 205
        if data.len() < MIN_SIZE {
            return Err(format!(
                "GlyphManager checkpoint too short: {} < {}",
                data.len(),
                MIN_SIZE
            ));
        }
        let mut offset = 0;
        for i in 0..N_FIELD_MODALITIES {
            self.modality_prior[i] = f64::from_le_bytes(
                data[offset..offset + 8]
                    .try_into()
                    .map_err(|_| "GlyphManager: corrupt modality_prior bytes")?,
            );
            offset += 8;
        }
        self.prior_count = u64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| "GlyphManager: corrupt prior_count bytes")?,
        );
        offset += 8;
        let dom_idx = data[offset] as usize;
        self.dominant_modality = if dom_idx < N_FIELD_MODALITIES {
            FieldModality::ALL[dom_idx]
        } else {
            FieldModality::Resonant
        };
        offset += 1;
        for i in 0..N_FIELD_MODALITIES {
            self.last_coords[i] = f64::from_le_bytes(
                data[offset..offset + 8]
                    .try_into()
                    .map_err(|_| "GlyphManager: corrupt last_coords bytes")?,
            );
            offset += 8;
        }
        self.last_coherence.value = f64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| "GlyphManager: corrupt coherence bytes")?,
        );
        offset += 8;
        self.last_free_energy.free_energy = f64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| "GlyphManager: corrupt free_energy bytes")?,
        );
        offset += 8;
        self.spiral_position = f32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| "GlyphManager: corrupt spiral_position bytes")?,
        );
        Ok(())
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
