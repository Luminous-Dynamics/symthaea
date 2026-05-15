// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multimodal Manager — MCE Gating for External Gen Models
//!
//! Consolidates multimodal state into a single [`CognitiveSubsystem`] that reads
//! from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Research Directions 2026-2027 Implementation
//!
//! 1. **Consciousness-Gated Direction**: Uses Unified Psi (Ψ) as an epistemic gate.
//! 2. **Moral Action Gating**: Uses Moral Algebra score to veto harmful multimodal requests.
//! 3. **Thermodynamic Regulation**: Dampens request frequency under high metabolic load.

use super::super::subsystem_trait::{
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};

/// Multimodal Manager — gates external generative models based on consciousness and ethics.
pub struct MultimodalManager {
    /// Count of multimodal requests processed this session
    total_requests: u64,
    /// Count of requests vetoed by the manager
    vetoed_requests: u64,
    /// EMA of moral alignment for multimodal actions
    moral_alignment_ema: f32,
}

impl Default for MultimodalManager {
    fn default() -> Self {
        Self {
            total_requests: 0,
            vetoed_requests: 0,
            moral_alignment_ema: 0.0,
        }
    }
}

impl MultimodalManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 31;

    /// Minimum consciousness (Ψ) required for complex multimodal generation (e.g. video).
    /// Science: Baars (1988) — complex coordination requires global broadcast.
    const MIN_PSI_FOR_GEN: f64 = 0.35;

    /// Moral score threshold below which generation is vetoed.
    /// Science: Zak (2012) — pro-social sentiment gates complex cooperation.
    const MORAL_VETO_THRESHOLD: f32 = -0.15;

    /// Thermodynamic load above which generation is throttled (LR dampening).
    const THERMO_THROTTLE_THRESHOLD: f32 = 0.80;

    /// EMA smoothing for moral alignment.
    const MORAL_EMA_ALPHA: f32 = 0.15;

    /// Return telemetry for the manager.
    pub fn telemetry(&self) -> MultimodalTelemetry {
        MultimodalTelemetry {
            total_requests: self.total_requests,
            vetoed_requests: self.vetoed_requests,
            moral_alignment_ema: self.moral_alignment_ema,
        }
    }
}

/// Telemetry for the Multimodal Manager.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct MultimodalTelemetry {
    pub total_requests: u64,
    pub vetoed_requests: u64,
    pub moral_alignment_ema: f32,
}

impl CognitiveSubsystem for MultimodalManager {
    fn name(&self) -> &'static str {
        "multimodal_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // Update moral EMA
        self.moral_alignment_ema = self.moral_alignment_ema * (1.0 - Self::MORAL_EMA_ALPHA)
            + snapshot.moral_score * Self::MORAL_EMA_ALPHA;

        // ── 0. High-Resolution Gating ─────────────────────────────────────
        // If in Ultra mode (64K), we require higher consciousness and moral clarity
        // because the "directorial" impact is higher.
        if snapshot.vision_hdc_dim > 16384 {
            if snapshot.unified_psi < Self::MIN_PSI_FOR_GEN + 0.1 {
                output.flags |= output_flags::VETO_ACTION;
                tracing::warn!(
                    psi = snapshot.unified_psi,
                    "Multimodal VETO: Ultra-res vision requires higher Ψ ({:.2} < {:.2})",
                    snapshot.unified_psi,
                    Self::MIN_PSI_FOR_GEN + 0.1
                );
            }

            if snapshot.moral_score < 0.0 {
                output.flags |= output_flags::VETO_ACTION;
                tracing::warn!(
                    moral = snapshot.moral_score,
                    "Multimodal VETO: Ultra-res vision requires positive moral alignment"
                );
            }
        }

        // Only act if there are active external requests
        if snapshot.active_external_requests != 0 {
            self.total_requests += 1;

            // ── 1. Consciousness Gating ──────────────────────────────────────
            // If Ψ is too low, the system lacks the integration to "direct" a gen model.
            if snapshot.unified_psi < Self::MIN_PSI_FOR_GEN {
                output.flags |= output_flags::VETO_ACTION;
                output.flags |= output_flags::ANOMALY_DETECTED;
                output.confidence_delta -= 0.05;
                self.vetoed_requests += 1;
                tracing::warn!(
                    psi = snapshot.unified_psi,
                    "Multimodal VETO: Insufficient consciousness (Ψ < {})",
                    Self::MIN_PSI_FOR_GEN
                );
            }

            // ── 2. Moral Gating ──────────────────────────────────────────────
            // If the moral score is too low, block the action at the source.
            if snapshot.moral_score < Self::MORAL_VETO_THRESHOLD {
                output.flags |= output_flags::VETO_ACTION;
                output.flags |= output_flags::ANOMALY_DETECTED;
                output.valence_delta -= 0.1; // Negative affect for moral violation
                if (output.flags & output_flags::VETO_ACTION) == 0 {
                    self.vetoed_requests += 1;
                }
                tracing::warn!(
                    moral_score = snapshot.moral_score,
                    "Multimodal VETO: Moral alignment violation (score < {})",
                    Self::MORAL_VETO_THRESHOLD
                );
            }

            // ── 3. Thermodynamic Throttling ──────────────────────────────────
            // High load → dampen learning/excitement for heavy multimodal tasks.
            if snapshot.thermodynamic_load > Self::THERMO_THROTTLE_THRESHOLD {
                output.lr_modulation *= 0.85; // Slow down learning during stress
                output.arousal_delta -= 0.05; // Try to calm down
            }
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(20);
        data.extend_from_slice(&self.total_requests.to_le_bytes());
        data.extend_from_slice(&self.vetoed_requests.to_le_bytes());
        data.extend_from_slice(&self.moral_alignment_ema.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 20 {
            return Err(format!(
                "MultimodalManager checkpoint too short: {} < 20",
                data.len()
            ));
        }
        self.total_requests = u64::from_le_bytes(data[0..8].try_into().unwrap());
        self.vetoed_requests = u64::from_le_bytes(data[8..16].try_into().unwrap());
        self.moral_alignment_ema = f32::from_le_bytes(data[16..20].try_into().unwrap());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    #[test]
    fn test_multimodal_veto_low_psi() {
        let mut mgr = MultimodalManager::default();
        let mut snap = CycleSnapshot::default();
        snap.active_external_requests = 1;
        snap.unified_psi = 0.2; // Below 0.35 threshold

        let output = mgr.process(&snap);
        assert!(output.has_flag(output_flags::VETO_ACTION));
        assert_eq!(mgr.vetoed_requests, 1);
    }

    #[test]
    fn test_multimodal_veto_bad_moral() {
        let mut mgr = MultimodalManager::default();
        let mut snap = CycleSnapshot::default();
        snap.active_external_requests = 1;
        snap.unified_psi = 0.8;
        snap.moral_score = -0.5; // Below -0.15 threshold

        let output = mgr.process(&snap);
        assert!(output.has_flag(output_flags::VETO_ACTION));
    }

    #[test]
    fn test_multimodal_pass() {
        let mut mgr = MultimodalManager::default();
        let mut snap = CycleSnapshot::default();
        snap.active_external_requests = 1;
        snap.unified_psi = 0.8;
        snap.moral_score = 0.5;

        let output = mgr.process(&snap);
        assert!(!output.has_flag(output_flags::VETO_ACTION));
        assert_eq!(mgr.vetoed_requests, 0);
        assert_eq!(mgr.total_requests, 1);
    }

    #[test]
    fn test_ultra_res_gating() {
        let mut mgr = MultimodalManager::default();
        let mut snap = CycleSnapshot::default();
        snap.vision_hdc_dim = 65536; // Ultra mode

        // 1. Low Ψ for Ultra (0.4 is > MIN_PSI_FOR_GEN but < MIN_PSI_FOR_GEN + 0.1)
        snap.unified_psi = 0.4;
        snap.moral_score = 0.5;
        let output = mgr.process(&snap);
        assert!(output.has_flag(output_flags::VETO_ACTION));

        // 2. Bad moral score for Ultra
        snap.unified_psi = 0.8;
        snap.moral_score = -0.05;
        let output = mgr.process(&snap);
        assert!(output.has_flag(output_flags::VETO_ACTION));

        // 3. High Ψ and good moral score -> PASS
        snap.moral_score = 0.1;
        let output = mgr.process(&snap);
        assert!(!output.has_flag(output_flags::VETO_ACTION));
    }
}
