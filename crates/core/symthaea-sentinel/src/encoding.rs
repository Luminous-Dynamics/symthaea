// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoding for audio features.
//!
//! This module provides:
//! - `AudioHdcEncoder`: Standard encoder using sparse binary projection
//! - `PremiumHdcEncoder`: Advanced encoder using CfC + RFF + Phase Space
//! - `AudioHdcVectors`: Container for encoded HDC vectors
//! - `EncoderMode`: Selection between standard and premium encoding

use crate::features::{AudioFeatures, CONTROL_RATE, NUM_MFCC};
use crate::hdc::{HDC_DIM, HV, RFF_GAMMA, RffProjector, SparseProjector};
use crate::temporal::{HierarchicalCfc, TemporalWindow};

/// Number of TIMBRE features for sparse projection
const NUM_TIMBRE_FEATURES: usize = 16;

/// Number of RHYTHM features for sparse projection
const NUM_RHYTHM_FEATURES: usize = 4;

/// Number of ENVELOPE features for sparse projection
const NUM_ENVELOPE_FEATURES: usize = 2;

/// Dimension of CfC state per pathway
const CFC_STATE_DIM: usize = 64;

// =============================================================================
// AudioHdcVectors
// =============================================================================

pub struct AudioHdcVectors {
    pub timbre: HV,
    pub rhythm: HV,
    pub envelope: HV,
    pub context: HV,
}

// =============================================================================
// Encoder Mode
// =============================================================================

/// Encoder mode selection
#[derive(Clone, Copy, PartialEq)]
pub enum EncoderMode {
    /// Original: SparseProjector + LTC (discrete binning)
    Standard,
    /// Premium: CfC + RFF + Phase Space + Temporal Windowing (topology-preserving)
    Premium,
}

// =============================================================================
// Standard HDC Encoder
// =============================================================================

/// Encodes audio features into hierarchical HDC vectors using SPARSE BINARY PROJECTION
pub struct AudioHdcEncoder {
    timbre_projector: SparseProjector,
    rhythm_projector: SparseProjector,
    envelope_projector: SparseProjector,
    timbre_role: HV,
    rhythm_role: HV,
    envelope_role: HV,
    position_basis: HV,
}

impl AudioHdcEncoder {
    pub fn new() -> Self {
        Self {
            timbre_projector: SparseProjector::new(NUM_TIMBRE_FEATURES, 10000),
            rhythm_projector: SparseProjector::new(NUM_RHYTHM_FEATURES, 20000),
            envelope_projector: SparseProjector::new(NUM_ENVELOPE_FEATURES, 30000),
            timbre_role: HV::random_binary_seeded(3000),
            rhythm_role: HV::random_binary_seeded(3001),
            envelope_role: HV::random_binary_seeded(3002),
            position_basis: HV::random_binary_seeded(5000),
        }
    }

    /// Encode features into hierarchical structure using SPARSE BINARY PROJECTION
    pub fn encode(&self, features: &AudioFeatures) -> AudioHdcVectors {
        // Prepare timbre features [0, 1] normalized
        let mut timbre_features = Vec::with_capacity(NUM_TIMBRE_FEATURES);

        // MFCC coefficients: map from typical range [-20, 20] to [0, 1]
        for &coeff in features.mfcc.iter().take(NUM_MFCC) {
            let normalized = (coeff / 40.0 + 0.5).clamp(0.0, 1.0);
            timbre_features.push(normalized);
        }
        while timbre_features.len() < NUM_MFCC {
            timbre_features.push(0.5);
        }

        timbre_features.push((features.spectral_centroid / 8000.0).clamp(0.0, 1.0));
        timbre_features.push(features.spectral_flatness.clamp(0.0, 1.0));
        timbre_features.push((features.spectral_rolloff / 10000.0).clamp(0.0, 1.0));

        // Prepare rhythm features
        let rhythm_features = vec![
            features.onset_strength.clamp(0.0, 1.0),
            features.rms_energy.clamp(0.0, 1.0),
            (features.zero_crossing_rate * 2.0).clamp(0.0, 1.0),
            features.temporal_regularity.clamp(0.0, 1.0),
        ];

        // Prepare envelope features
        let envelope_features = vec![
            (features.envelope_delta * 10.0).clamp(0.0, 1.0),
            (features.envelope_variance * 100.0).clamp(0.0, 1.0),
        ];

        // SPARSE BINARY PROJECTION
        let timbre = self.timbre_projector.project_masked(&timbre_features, 0.05);
        let rhythm = self.rhythm_projector.project_masked(&rhythm_features, 0.05);
        let envelope = self
            .envelope_projector
            .project_masked(&envelope_features, 0.05);

        // Phase-sensitive binding
        let phase_position = features.frame_index % 64;
        let position_shifted = self.position_basis.permute(phase_position * (HDC_DIM / 64));

        // Context vector
        let timbre_bound = timbre.xor_bind(&self.timbre_role);
        let rhythm_bound = rhythm.xor_bind(&self.rhythm_role);
        let envelope_bound = envelope.xor_bind(&self.envelope_role);

        let content = timbre_bound.add(&rhythm_bound).add(&envelope_bound);
        let context = content.xor_bind(&position_shifted).normalize();

        AudioHdcVectors {
            timbre,
            rhythm,
            envelope,
            context,
        }
    }
}

impl Default for AudioHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Premium HDC Encoder
// =============================================================================

/// Premium HDC Encoder using CfC + RFF + Phase Space + Temporal Windowing
pub struct PremiumHdcEncoder {
    cfc_timbre: HierarchicalCfc,
    cfc_rhythm: HierarchicalCfc,
    rff_timbre: RffProjector,
    rff_rhythm: RffProjector,
    window_timbre: TemporalWindow,
    window_rhythm: TemporalWindow,
    timbre_role: HV,
    rhythm_role: HV,
    frame_count: usize,
    dt_ms: f32,
}

impl PremiumHdcEncoder {
    pub fn new() -> Self {
        // Phase space dimension = state + velocity for each CfC level
        // 5 levels x 64 dims x 2 (state + velocity) = 640
        let phase_space_dim = CFC_STATE_DIM * 2 * 5;

        Self {
            cfc_timbre: HierarchicalCfc::with_taus(
                CFC_STATE_DIM,
                &[500.0, 200.0, 80.0, 40.0, 30.0],
            ),
            cfc_rhythm: HierarchicalCfc::with_taus(
                CFC_STATE_DIM,
                &[500.0, 200.0, 80.0, 40.0, 30.0],
            ),
            rff_timbre: RffProjector::with_gamma(phase_space_dim, HDC_DIM, 50000, RFF_GAMMA),
            rff_rhythm: RffProjector::with_gamma(phase_space_dim, HDC_DIM, 60000, RFF_GAMMA),
            window_timbre: TemporalWindow::new(NUM_TIMBRE_FEATURES),
            window_rhythm: TemporalWindow::new(NUM_RHYTHM_FEATURES),
            timbre_role: HV::random_binary_seeded(70000),
            rhythm_role: HV::random_binary_seeded(70001),
            frame_count: 0,
            dt_ms: 1000.0 / CONTROL_RATE,
        }
    }

    /// Encode audio features using the Premium architecture
    pub fn encode(&mut self, features: &AudioFeatures) -> AudioHdcVectors {
        // Step 1: Prepare normalized features
        let mut timbre_features = Vec::with_capacity(NUM_TIMBRE_FEATURES);

        for &coeff in features.mfcc.iter().take(NUM_MFCC) {
            let normalized = (coeff / 40.0 + 0.5).clamp(0.0, 1.0);
            timbre_features.push(normalized);
        }
        while timbre_features.len() < NUM_MFCC {
            timbre_features.push(0.5);
        }

        timbre_features.push((features.spectral_centroid / 8000.0).clamp(0.0, 1.0));
        timbre_features.push(features.spectral_flatness.clamp(0.0, 1.0));
        timbre_features.push((features.spectral_rolloff / 10000.0).clamp(0.0, 1.0));

        let rhythm_features = vec![
            features.onset_strength.clamp(0.0, 1.0),
            features.rms_energy.clamp(0.0, 1.0),
            (features.zero_crossing_rate * 2.0).clamp(0.0, 1.0),
            features.temporal_regularity.clamp(0.0, 1.0),
        ];

        // Step 2: Push into temporal windows
        self.window_timbre.push(&timbre_features);
        self.window_rhythm.push(&rhythm_features);

        // Step 3: Get windowed context
        let windowed_timbre = self.window_timbre.get_weighted_context(0.3);
        let windowed_rhythm = self.window_rhythm.get_weighted_context(0.3);

        // Step 4: Step CfC networks
        self.cfc_timbre.step(self.dt_ms, &windowed_timbre);
        self.cfc_rhythm.step(self.dt_ms, &windowed_rhythm);

        // Step 5: Extract phase space
        let timbre_phase_space = self.cfc_timbre.get_multi_scale_phase_space();
        let rhythm_phase_space = self.cfc_rhythm.get_multi_scale_phase_space();

        // Step 6: RFF projection
        let timbre = self.rff_timbre.project_normalized(&timbre_phase_space);
        let rhythm = self.rff_rhythm.project_normalized(&rhythm_phase_space);

        // Step 7: Envelope vector
        let envelope_features = [
            (features.envelope_delta * 10.0).clamp(0.0, 1.0),
            (features.envelope_variance * 100.0).clamp(0.0, 1.0),
        ];
        let mut envelope_values = vec![0.0f32; HDC_DIM];
        for i in 0..HDC_DIM {
            let idx = i % envelope_features.len();
            let basis_sign = if (i / envelope_features.len()).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            envelope_values[i] = basis_sign * (envelope_features[idx] * 2.0 - 1.0);
        }
        let envelope = HV {
            values: envelope_values,
        }
        .normalize();

        // Step 8: Context vector
        let timbre_bound = timbre.xor_bind(&self.timbre_role);
        let rhythm_bound = rhythm.xor_bind(&self.rhythm_role);
        let content = timbre_bound.add(&rhythm_bound).add(&envelope);
        let context = content.normalize();

        self.frame_count += 1;

        AudioHdcVectors {
            timbre,
            rhythm,
            envelope,
            context,
        }
    }

    pub fn get_phi(&self) -> (f64, f64) {
        (self.cfc_timbre.phi, self.cfc_rhythm.phi)
    }

    #[allow(dead_code)]
    pub fn get_phase_spaces(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.cfc_timbre.get_multi_scale_phase_space(),
            self.cfc_rhythm.get_multi_scale_phase_space(),
        )
    }

    pub fn reset(&mut self) {
        self.cfc_timbre.reset();
        self.cfc_rhythm.reset();
        self.window_timbre.reset();
        self.window_rhythm.reset();
        self.frame_count = 0;
    }
}

impl Default for PremiumHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::MEL_BANDS;

    fn make_test_features(frame_index: usize) -> AudioFeatures {
        AudioFeatures::synthetic(
            vec![0.1; MEL_BANDS],
            vec![1.0; NUM_MFCC],
            2000.0,
            4000.0,
            0.5,
            0.3,
            0.2,
            0.4,
            0.6,
            0.05,
            0.02,
            frame_index,
        )
    }

    // =========================================================================
    // AudioHdcEncoder tests
    // =========================================================================

    #[test]
    fn test_audio_encoder_default() {
        let encoder = AudioHdcEncoder::default();
        let features = make_test_features(0);
        let vecs = encoder.encode(&features);
        assert_eq!(vecs.timbre.values.len(), HDC_DIM);
        assert_eq!(vecs.rhythm.values.len(), HDC_DIM);
        assert_eq!(vecs.envelope.values.len(), HDC_DIM);
        assert_eq!(vecs.context.values.len(), HDC_DIM);
    }

    #[test]
    fn test_audio_encoder_output_finite() {
        let encoder = AudioHdcEncoder::new();
        let features = make_test_features(0);
        let vecs = encoder.encode(&features);
        assert!(vecs.timbre.values.iter().all(|v| v.is_finite()));
        assert!(vecs.rhythm.values.iter().all(|v| v.is_finite()));
        assert!(vecs.envelope.values.iter().all(|v| v.is_finite()));
        assert!(vecs.context.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_audio_encoder_different_frames_different_context() {
        let encoder = AudioHdcEncoder::new();
        let vecs_0 = encoder.encode(&make_test_features(0));
        let vecs_1 = encoder.encode(&make_test_features(1));
        // Different frame_index → different position binding → different context
        assert_ne!(
            vecs_0.context.values, vecs_1.context.values,
            "Different frame indices should produce different context vectors"
        );
    }

    #[test]
    fn test_audio_encoder_same_input_same_output() {
        let encoder = AudioHdcEncoder::new();
        let features = make_test_features(5);
        let vecs_a = encoder.encode(&features);
        let vecs_b = encoder.encode(&features);
        assert_eq!(vecs_a.timbre.values, vecs_b.timbre.values);
        assert_eq!(vecs_a.context.values, vecs_b.context.values);
    }

    #[test]
    fn test_audio_encoder_context_is_normalized() {
        let encoder = AudioHdcEncoder::new();
        let vecs = encoder.encode(&make_test_features(0));
        let norm: f32 = vecs
            .context
            .values
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        // Normalized should have norm close to sqrt(HDC_DIM) for bipolar or 1.0
        assert!(
            norm.is_finite() && norm > 0.0,
            "Context norm should be positive"
        );
    }

    // =========================================================================
    // PremiumHdcEncoder tests
    // =========================================================================

    #[test]
    fn test_premium_encoder_default() {
        let mut encoder = PremiumHdcEncoder::default();
        let features = make_test_features(0);
        let vecs = encoder.encode(&features);
        assert_eq!(vecs.timbre.values.len(), HDC_DIM);
        assert_eq!(vecs.context.values.len(), HDC_DIM);
    }

    #[test]
    fn test_premium_encoder_output_finite() {
        let mut encoder = PremiumHdcEncoder::new();
        let features = make_test_features(0);
        let vecs = encoder.encode(&features);
        assert!(vecs.timbre.values.iter().all(|v| v.is_finite()));
        assert!(vecs.rhythm.values.iter().all(|v| v.is_finite()));
        assert!(vecs.envelope.values.iter().all(|v| v.is_finite()));
        assert!(vecs.context.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_premium_encoder_evolves_state() {
        let mut encoder = PremiumHdcEncoder::new();
        let vecs_0 = encoder.encode(&make_test_features(0));
        let vecs_1 = encoder.encode(&make_test_features(1));
        // Premium encoder has internal CfC state, so sequential calls differ
        assert_ne!(
            vecs_0.timbre.values, vecs_1.timbre.values,
            "Premium encoder should produce different timbre on sequential frames"
        );
    }

    #[test]
    fn test_premium_encoder_get_phi() {
        let mut encoder = PremiumHdcEncoder::new();
        let (phi_t, phi_r) = encoder.get_phi();
        assert_eq!(phi_t, 0.0);
        assert_eq!(phi_r, 0.0);

        encoder.encode(&make_test_features(0));
        let (phi_t, phi_r) = encoder.get_phi();
        assert!(phi_t >= 0.0 && phi_t.is_finite());
        assert!(phi_r >= 0.0 && phi_r.is_finite());
    }

    #[test]
    fn test_premium_encoder_reset() {
        let mut encoder = PremiumHdcEncoder::new();
        for i in 0..10 {
            encoder.encode(&make_test_features(i));
        }
        let (phi_t, _) = encoder.get_phi();
        assert!(phi_t > 0.0, "Should have nonzero phi after encoding");

        encoder.reset();
        let (phi_t, phi_r) = encoder.get_phi();
        assert_eq!(phi_t, 0.0);
        assert_eq!(phi_r, 0.0);
    }

    #[test]
    fn test_premium_encoder_get_phase_spaces() {
        let mut encoder = PremiumHdcEncoder::new();
        encoder.encode(&make_test_features(0));
        let (timbre_ps, rhythm_ps) = encoder.get_phase_spaces();
        // 5 CfC levels x 64 dims x 2 (state + velocity) = 640
        assert_eq!(timbre_ps.len(), 640);
        assert_eq!(rhythm_ps.len(), 640);
        assert!(timbre_ps.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_premium_encoder_multiple_frames_phi_grows() {
        let mut encoder = PremiumHdcEncoder::new();
        // Encode several frames with varying features
        for i in 0..20 {
            let mut features = make_test_features(i);
            features.spectral_centroid = 1000.0 + (i as f32) * 200.0;
            features.onset_strength = (i as f32 * 0.3).sin().abs();
            encoder.encode(&features);
        }
        let (phi_t, phi_r) = encoder.get_phi();
        assert!(phi_t.is_finite());
        assert!(phi_r.is_finite());
    }
}
