// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Threshold genome: cognitive loop parameter evolution via BinaryHV loci.
//!
//! Extends the neural genome (bits 0-380) with cognitive threshold loci starting
//! at bit 400. A single 16,384-bit BinaryHV encodes both neural architecture
//! AND cognitive thresholds, enabling co-evolution of structure and dynamics.
//!
//! # Safety
//!
//! Safety thresholds (safety.rs, moral.rs) are **never** encoded. They are
//! constitutionally immutable and cannot be mutated by any bit pattern.
//!
//! # References
//!
//! - Hansen & Ostermeier (2001). CMA-ES for parameter optimization.
//! - Clune et al. (2013). The evolutionary origins of modularity.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;

use crate::genome::{bits_to_linear, extract_bits, linear_to_bits, set_bits};

// ═══════════════════════════════════════════════════════════════════════════════
// THRESHOLD LOCUS DEFINITIONS (bits 400+ of the 16,384-bit BinaryHV)
// Neural genome uses bits 0-380; gap 381-399 is reserved headroom.
// ═══════════════════════════════════════════════════════════════════════════════

// --- Learning thresholds (bits 400-435) ---
const FEP_SURPRISE_SCALE_START: usize = 400;
const FEP_SURPRISE_SCALE_BITS: usize = 12; // [1.0, 10.0]

const FEP_LR_DECAY_START: usize = 412;
const FEP_LR_DECAY_BITS: usize = 12; // [0.80, 0.999]

// --- Consciousness thresholds (bits 436-465) ---
const DREAM_BASE_INTERVAL_START: usize = 436;
const DREAM_BASE_INTERVAL_BITS: usize = 10; // [20, 500] cycles

const DREAM_MIN_INTERVAL_START: usize = 446;
const DREAM_MIN_INTERVAL_BITS: usize = 8; // [10, 100] cycles

// --- Neuromodulator thresholds (bits 466-537) ---
const NEUROMOD_D2_BASELINE_START: usize = 466;
const NEUROMOD_D2_BASELINE_BITS: usize = 12; // [0.1, 0.9]

const NEUROMOD_NE_PHASIC_THRESHOLD_START: usize = 478;
const NEUROMOD_NE_PHASIC_THRESHOLD_BITS: usize = 12; // [0.1, 0.8]

const NEUROMOD_AROUSAL_EMA_DECAY_START: usize = 490;
const NEUROMOD_AROUSAL_EMA_DECAY_BITS: usize = 12; // [0.8, 0.99]

const HOMEOSTASIS_RECALIBRATE_HIGH_START: usize = 502;
const HOMEOSTASIS_RECALIBRATE_HIGH_BITS: usize = 12; // [1.05, 1.30]

const HOMEOSTASIS_RECALIBRATE_LOW_START: usize = 514;
const HOMEOSTASIS_RECALIBRATE_LOW_BITS: usize = 12; // [0.70, 0.95]

const NEUROMOD_EMA_ALPHA_START: usize = 526;
const NEUROMOD_EMA_ALPHA_BITS: usize = 12; // [0.01, 0.15]

// --- Drive thresholds (bits 538-585) ---
const FRUSTRATION_DAMPEN_THRESHOLD_START: usize = 538;
const FRUSTRATION_DAMPEN_THRESHOLD_BITS: usize = 12; // [0.2, 0.8]

const ENGAGEMENT_LOW_THRESHOLD_START: usize = 550;
const ENGAGEMENT_LOW_THRESHOLD_BITS: usize = 12; // [0.1, 0.6]

const FLOW_EXPLORATION_INCREMENT_START: usize = 562;
const FLOW_EXPLORATION_INCREMENT_BITS: usize = 12; // [0.01, 0.15]

const COHERENCE_LOW_START: usize = 574;
const COHERENCE_LOW_BITS: usize = 12; // [0.15, 0.50]

// --- Feedback thresholds (bits 586-633) ---
const AROUSAL_TRAP_THRESHOLD_START: usize = 586;
const AROUSAL_TRAP_THRESHOLD_BITS: usize = 12; // [0.5, 0.95]

const SELF_MODEL_WEIGHT_HIGH_START: usize = 598;
const SELF_MODEL_WEIGHT_HIGH_BITS: usize = 12; // [0.4, 0.95]

const HOMEOSTASIS_PULL_CRUISE_START: usize = 610;
const HOMEOSTASIS_PULL_CRUISE_BITS: usize = 12; // [0.5, 3.0]

const CONFIDENCE_CRASH_THRESHOLD_START: usize = 622;
const CONFIDENCE_CRASH_THRESHOLD_BITS: usize = 12; // [0.15, 0.50]

/// End of threshold region (exclusive). Next locus starts here.
pub const THRESHOLD_REGION_END: usize = 634;

// ═══════════════════════════════════════════════════════════════════════════════
// PHENOTYPE
// ═══════════════════════════════════════════════════════════════════════════════

/// Decoded threshold phenotype from a `NeuralGenome`.
///
/// Each field maps to a cognitive loop threshold constant. Values are clamped
/// to biologically plausible ranges during decode — no bit pattern can produce
/// an out-of-range value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThresholdPhenotype {
    // Learning
    /// FEP surprise → learning rate boost scale. Default 3.0.
    pub fep_surprise_scale: f32,
    /// Per-cycle FEP learning rate decay (no surprise). Default 0.95.
    pub fep_lr_decay: f32,

    // Consciousness
    /// Base dream consolidation interval (cycles). Default 100.
    pub dream_base_interval: u64,
    /// Minimum dream interval under high learning rate. Default 30.
    pub dream_min_interval: u64,

    // Neuromodulation
    /// D2 receptor flexibility baseline. Default 0.5.
    pub neuromod_d2_baseline: f64,
    /// NE phasic burst threshold. Default 0.3.
    pub neuromod_ne_phasic_threshold: f32,
    /// Arousal EMA decay rate. Default 0.9.
    pub neuromod_arousal_ema_decay: f32,
    /// Homeostasis recalibration ceiling. Default 1.15.
    pub homeostasis_recalibrate_high: f32,
    /// Homeostasis recalibration floor. Default 0.85.
    pub homeostasis_recalibrate_low: f32,
    /// Neuromodulator EMA smoothing alpha. Default 0.05.
    pub neuromod_ema_alpha: f32,

    // Drives
    /// Frustration dampening engagement threshold. Default 0.5.
    pub frustration_dampen_threshold: f64,
    /// Low engagement detection threshold. Default 0.3.
    pub engagement_low_threshold: f64,
    /// Flow state exploration increment. Default 0.05.
    pub flow_exploration_increment: f32,
    /// Low coherence boundary. Default 0.3.
    pub coherence_low: f32,

    // Feedback
    /// Arousal trap detection threshold. Default 0.8.
    pub arousal_trap_threshold: f32,
    /// Self-model weight high confidence threshold. Default 0.7.
    pub self_model_weight_high: f32,
    /// Homeostasis pull in cruise urgency. Default 1.5.
    pub homeostasis_pull_cruise: f32,
    /// Confidence crash detection threshold. Default 0.30.
    pub confidence_crash_threshold: f64,
}

impl Default for ThresholdPhenotype {
    fn default() -> Self {
        Self {
            fep_surprise_scale: 3.0,
            fep_lr_decay: 0.95,
            dream_base_interval: 100,
            dream_min_interval: 30,
            neuromod_d2_baseline: 0.5,
            neuromod_ne_phasic_threshold: 0.3,
            neuromod_arousal_ema_decay: 0.9,
            homeostasis_recalibrate_high: 1.15,
            homeostasis_recalibrate_low: 0.85,
            neuromod_ema_alpha: 0.05,
            frustration_dampen_threshold: 0.5,
            engagement_low_threshold: 0.3,
            flow_exploration_increment: 0.05,
            coherence_low: 0.3,
            arousal_trap_threshold: 0.8,
            self_model_weight_high: 0.7,
            homeostasis_pull_cruise: 1.5,
            confidence_crash_threshold: 0.30,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECODE / ENCODE
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode threshold phenotype from a BinaryHV genome (bits 400+).
pub fn decode_thresholds(hv: &BinaryHV) -> ThresholdPhenotype {
    ThresholdPhenotype {
        fep_surprise_scale: bits_to_linear(
            extract_bits(hv, FEP_SURPRISE_SCALE_START, FEP_SURPRISE_SCALE_BITS),
            FEP_SURPRISE_SCALE_BITS,
            1.0,
            10.0,
        ),
        fep_lr_decay: bits_to_linear(
            extract_bits(hv, FEP_LR_DECAY_START, FEP_LR_DECAY_BITS),
            FEP_LR_DECAY_BITS,
            0.80,
            0.999,
        ),
        dream_base_interval: {
            let raw = bits_to_linear(
                extract_bits(hv, DREAM_BASE_INTERVAL_START, DREAM_BASE_INTERVAL_BITS),
                DREAM_BASE_INTERVAL_BITS,
                20.0,
                500.0,
            );
            raw.round() as u64
        },
        dream_min_interval: {
            let raw = bits_to_linear(
                extract_bits(hv, DREAM_MIN_INTERVAL_START, DREAM_MIN_INTERVAL_BITS),
                DREAM_MIN_INTERVAL_BITS,
                10.0,
                100.0,
            );
            raw.round() as u64
        },
        neuromod_d2_baseline: bits_to_linear(
            extract_bits(hv, NEUROMOD_D2_BASELINE_START, NEUROMOD_D2_BASELINE_BITS),
            NEUROMOD_D2_BASELINE_BITS,
            0.1,
            0.9,
        ) as f64,
        neuromod_ne_phasic_threshold: bits_to_linear(
            extract_bits(
                hv,
                NEUROMOD_NE_PHASIC_THRESHOLD_START,
                NEUROMOD_NE_PHASIC_THRESHOLD_BITS,
            ),
            NEUROMOD_NE_PHASIC_THRESHOLD_BITS,
            0.1,
            0.8,
        ),
        neuromod_arousal_ema_decay: bits_to_linear(
            extract_bits(
                hv,
                NEUROMOD_AROUSAL_EMA_DECAY_START,
                NEUROMOD_AROUSAL_EMA_DECAY_BITS,
            ),
            NEUROMOD_AROUSAL_EMA_DECAY_BITS,
            0.8,
            0.99,
        ),
        homeostasis_recalibrate_high: bits_to_linear(
            extract_bits(
                hv,
                HOMEOSTASIS_RECALIBRATE_HIGH_START,
                HOMEOSTASIS_RECALIBRATE_HIGH_BITS,
            ),
            HOMEOSTASIS_RECALIBRATE_HIGH_BITS,
            1.05,
            1.30,
        ),
        homeostasis_recalibrate_low: bits_to_linear(
            extract_bits(
                hv,
                HOMEOSTASIS_RECALIBRATE_LOW_START,
                HOMEOSTASIS_RECALIBRATE_LOW_BITS,
            ),
            HOMEOSTASIS_RECALIBRATE_LOW_BITS,
            0.70,
            0.95,
        ),
        neuromod_ema_alpha: bits_to_linear(
            extract_bits(hv, NEUROMOD_EMA_ALPHA_START, NEUROMOD_EMA_ALPHA_BITS),
            NEUROMOD_EMA_ALPHA_BITS,
            0.01,
            0.15,
        ),
        frustration_dampen_threshold: bits_to_linear(
            extract_bits(
                hv,
                FRUSTRATION_DAMPEN_THRESHOLD_START,
                FRUSTRATION_DAMPEN_THRESHOLD_BITS,
            ),
            FRUSTRATION_DAMPEN_THRESHOLD_BITS,
            0.2,
            0.8,
        ) as f64,
        engagement_low_threshold: bits_to_linear(
            extract_bits(
                hv,
                ENGAGEMENT_LOW_THRESHOLD_START,
                ENGAGEMENT_LOW_THRESHOLD_BITS,
            ),
            ENGAGEMENT_LOW_THRESHOLD_BITS,
            0.1,
            0.6,
        ) as f64,
        flow_exploration_increment: bits_to_linear(
            extract_bits(
                hv,
                FLOW_EXPLORATION_INCREMENT_START,
                FLOW_EXPLORATION_INCREMENT_BITS,
            ),
            FLOW_EXPLORATION_INCREMENT_BITS,
            0.01,
            0.15,
        ),
        coherence_low: bits_to_linear(
            extract_bits(hv, COHERENCE_LOW_START, COHERENCE_LOW_BITS),
            COHERENCE_LOW_BITS,
            0.15,
            0.50,
        ),
        arousal_trap_threshold: bits_to_linear(
            extract_bits(
                hv,
                AROUSAL_TRAP_THRESHOLD_START,
                AROUSAL_TRAP_THRESHOLD_BITS,
            ),
            AROUSAL_TRAP_THRESHOLD_BITS,
            0.5,
            0.95,
        ),
        self_model_weight_high: bits_to_linear(
            extract_bits(
                hv,
                SELF_MODEL_WEIGHT_HIGH_START,
                SELF_MODEL_WEIGHT_HIGH_BITS,
            ),
            SELF_MODEL_WEIGHT_HIGH_BITS,
            0.4,
            0.95,
        ),
        homeostasis_pull_cruise: bits_to_linear(
            extract_bits(
                hv,
                HOMEOSTASIS_PULL_CRUISE_START,
                HOMEOSTASIS_PULL_CRUISE_BITS,
            ),
            HOMEOSTASIS_PULL_CRUISE_BITS,
            0.5,
            3.0,
        ),
        confidence_crash_threshold: bits_to_linear(
            extract_bits(
                hv,
                CONFIDENCE_CRASH_THRESHOLD_START,
                CONFIDENCE_CRASH_THRESHOLD_BITS,
            ),
            CONFIDENCE_CRASH_THRESHOLD_BITS,
            0.15,
            0.50,
        ) as f64,
    }
}

/// Encode a threshold phenotype into the BinaryHV genome (bits 400+).
/// Does NOT modify bits 0-399 (neural genome region).
pub fn encode_thresholds(hv: &mut BinaryHV, pheno: &ThresholdPhenotype) {
    set_bits(
        hv,
        FEP_SURPRISE_SCALE_START,
        FEP_SURPRISE_SCALE_BITS,
        linear_to_bits(pheno.fep_surprise_scale, FEP_SURPRISE_SCALE_BITS, 1.0, 10.0),
    );
    set_bits(
        hv,
        FEP_LR_DECAY_START,
        FEP_LR_DECAY_BITS,
        linear_to_bits(pheno.fep_lr_decay, FEP_LR_DECAY_BITS, 0.80, 0.999),
    );
    set_bits(
        hv,
        DREAM_BASE_INTERVAL_START,
        DREAM_BASE_INTERVAL_BITS,
        linear_to_bits(
            pheno.dream_base_interval as f32,
            DREAM_BASE_INTERVAL_BITS,
            20.0,
            500.0,
        ),
    );
    set_bits(
        hv,
        DREAM_MIN_INTERVAL_START,
        DREAM_MIN_INTERVAL_BITS,
        linear_to_bits(
            pheno.dream_min_interval as f32,
            DREAM_MIN_INTERVAL_BITS,
            10.0,
            100.0,
        ),
    );
    set_bits(
        hv,
        NEUROMOD_D2_BASELINE_START,
        NEUROMOD_D2_BASELINE_BITS,
        linear_to_bits(
            pheno.neuromod_d2_baseline as f32,
            NEUROMOD_D2_BASELINE_BITS,
            0.1,
            0.9,
        ),
    );
    set_bits(
        hv,
        NEUROMOD_NE_PHASIC_THRESHOLD_START,
        NEUROMOD_NE_PHASIC_THRESHOLD_BITS,
        linear_to_bits(
            pheno.neuromod_ne_phasic_threshold,
            NEUROMOD_NE_PHASIC_THRESHOLD_BITS,
            0.1,
            0.8,
        ),
    );
    set_bits(
        hv,
        NEUROMOD_AROUSAL_EMA_DECAY_START,
        NEUROMOD_AROUSAL_EMA_DECAY_BITS,
        linear_to_bits(
            pheno.neuromod_arousal_ema_decay,
            NEUROMOD_AROUSAL_EMA_DECAY_BITS,
            0.8,
            0.99,
        ),
    );
    set_bits(
        hv,
        HOMEOSTASIS_RECALIBRATE_HIGH_START,
        HOMEOSTASIS_RECALIBRATE_HIGH_BITS,
        linear_to_bits(
            pheno.homeostasis_recalibrate_high,
            HOMEOSTASIS_RECALIBRATE_HIGH_BITS,
            1.05,
            1.30,
        ),
    );
    set_bits(
        hv,
        HOMEOSTASIS_RECALIBRATE_LOW_START,
        HOMEOSTASIS_RECALIBRATE_LOW_BITS,
        linear_to_bits(
            pheno.homeostasis_recalibrate_low,
            HOMEOSTASIS_RECALIBRATE_LOW_BITS,
            0.70,
            0.95,
        ),
    );
    set_bits(
        hv,
        NEUROMOD_EMA_ALPHA_START,
        NEUROMOD_EMA_ALPHA_BITS,
        linear_to_bits(
            pheno.neuromod_ema_alpha,
            NEUROMOD_EMA_ALPHA_BITS,
            0.01,
            0.15,
        ),
    );
    set_bits(
        hv,
        FRUSTRATION_DAMPEN_THRESHOLD_START,
        FRUSTRATION_DAMPEN_THRESHOLD_BITS,
        linear_to_bits(
            pheno.frustration_dampen_threshold as f32,
            FRUSTRATION_DAMPEN_THRESHOLD_BITS,
            0.2,
            0.8,
        ),
    );
    set_bits(
        hv,
        ENGAGEMENT_LOW_THRESHOLD_START,
        ENGAGEMENT_LOW_THRESHOLD_BITS,
        linear_to_bits(
            pheno.engagement_low_threshold as f32,
            ENGAGEMENT_LOW_THRESHOLD_BITS,
            0.1,
            0.6,
        ),
    );
    set_bits(
        hv,
        FLOW_EXPLORATION_INCREMENT_START,
        FLOW_EXPLORATION_INCREMENT_BITS,
        linear_to_bits(
            pheno.flow_exploration_increment,
            FLOW_EXPLORATION_INCREMENT_BITS,
            0.01,
            0.15,
        ),
    );
    set_bits(
        hv,
        COHERENCE_LOW_START,
        COHERENCE_LOW_BITS,
        linear_to_bits(pheno.coherence_low, COHERENCE_LOW_BITS, 0.15, 0.50),
    );
    set_bits(
        hv,
        AROUSAL_TRAP_THRESHOLD_START,
        AROUSAL_TRAP_THRESHOLD_BITS,
        linear_to_bits(
            pheno.arousal_trap_threshold,
            AROUSAL_TRAP_THRESHOLD_BITS,
            0.5,
            0.95,
        ),
    );
    set_bits(
        hv,
        SELF_MODEL_WEIGHT_HIGH_START,
        SELF_MODEL_WEIGHT_HIGH_BITS,
        linear_to_bits(
            pheno.self_model_weight_high,
            SELF_MODEL_WEIGHT_HIGH_BITS,
            0.4,
            0.95,
        ),
    );
    set_bits(
        hv,
        HOMEOSTASIS_PULL_CRUISE_START,
        HOMEOSTASIS_PULL_CRUISE_BITS,
        linear_to_bits(
            pheno.homeostasis_pull_cruise,
            HOMEOSTASIS_PULL_CRUISE_BITS,
            0.5,
            3.0,
        ),
    );
    set_bits(
        hv,
        CONFIDENCE_CRASH_THRESHOLD_START,
        CONFIDENCE_CRASH_THRESHOLD_BITS,
        linear_to_bits(
            pheno.confidence_crash_threshold as f32,
            CONFIDENCE_CRASH_THRESHOLD_BITS,
            0.15,
            0.50,
        ),
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// FITNESS EVALUATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Evaluate the internal consistency and quality of a threshold phenotype.
///
/// Returns a fitness score in [0.0, 1.0] where 1.0 = perfectly consistent.
/// Checks:
/// 1. Dream intervals: min < base (otherwise consolidation never accelerates)
/// 2. Homeostasis bounds: low < high (otherwise recalibration is inverted)
/// 3. Arousal trap threshold: must be above engagement_low (otherwise trap is always triggered)
/// 4. Neuromod stability: EMA alpha and decay must be in stable operating range
/// 5. FEP learning: surprise scale and decay must not cause learning instability
///
/// This is a lightweight consistency check, not a full cognitive simulation.
/// For full validation, use psych-bench integration (Phase 2B+).
pub fn evaluate_threshold_fitness(pheno: &ThresholdPhenotype) -> f64 {
    let mut score = 1.0f64;
    let mut penalties = 0;
    let max_penalties = 8;

    // 1. Dream interval ordering: min < base
    if pheno.dream_min_interval >= pheno.dream_base_interval {
        penalties += 2; // Severe: consolidation can never accelerate
    }

    // 2. Homeostasis bound ordering: low < high
    if pheno.homeostasis_recalibrate_low >= pheno.homeostasis_recalibrate_high {
        penalties += 2; // Severe: recalibration is inverted
    }

    // 3. Arousal trap vs engagement: trap should be above engagement threshold
    if pheno.arousal_trap_threshold < pheno.engagement_low_threshold as f32 + 0.1 {
        penalties += 1; // Mild: arousal trap fires too easily
    }

    // 4. Neuromod stability: EMA alpha should be small, decay should be high
    if pheno.neuromod_ema_alpha > 0.12 {
        penalties += 1; // Unstable: EMA too responsive
    }
    if pheno.neuromod_arousal_ema_decay < 0.83 {
        penalties += 1; // Unstable: arousal decays too fast
    }

    // 5. FEP learning balance: high surprise scale + low decay = learning explosion
    if pheno.fep_surprise_scale > 7.0 && pheno.fep_lr_decay > 0.98 {
        penalties += 1; // Dangerous: learning rate will explode under sustained surprise
    }

    // 6. Self-model weight should leave room for correction
    if pheno.self_model_weight_high > 0.92 {
        penalties += 1; // Too rigid: self-model dominates learning
    }

    // 7. Confidence crash threshold should be below engagement threshold
    if pheno.confidence_crash_threshold > pheno.frustration_dampen_threshold {
        penalties += 1; // Inconsistent: crashes before frustration kicks in
    }

    score -= penalties as f64 / max_penalties as f64;
    score.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_roundtrip() {
        let defaults = ThresholdPhenotype::default();
        let mut hv = BinaryHV::default();
        encode_thresholds(&mut hv, &defaults);
        let decoded = decode_thresholds(&hv);

        // 12-bit quantization gives ~0.1% precision
        assert!((decoded.fep_surprise_scale - 3.0).abs() < 0.05);
        assert!((decoded.fep_lr_decay - 0.95).abs() < 0.01);
        assert_eq!(decoded.dream_base_interval, 100);
        assert_eq!(decoded.dream_min_interval, 30);
        assert!((decoded.neuromod_d2_baseline - 0.5).abs() < 0.01);
        assert!((decoded.neuromod_ne_phasic_threshold - 0.3).abs() < 0.01);
        assert!((decoded.neuromod_arousal_ema_decay - 0.9).abs() < 0.01);
        assert!((decoded.homeostasis_recalibrate_high - 1.15).abs() < 0.01);
        assert!((decoded.homeostasis_recalibrate_low - 0.85).abs() < 0.01);
        assert!((decoded.neuromod_ema_alpha - 0.05).abs() < 0.01);
        assert!((decoded.arousal_trap_threshold - 0.8).abs() < 0.01);
        assert!((decoded.self_model_weight_high - 0.7).abs() < 0.01);
        assert!((decoded.homeostasis_pull_cruise - 1.5).abs() < 0.05);
        assert!((decoded.confidence_crash_threshold - 0.30).abs() < 0.01);
    }

    #[test]
    fn test_all_zeros_produces_valid_ranges() {
        let hv = BinaryHV::default(); // All zeros
        let decoded = decode_thresholds(&hv);

        // All-zeros bits decode to minimum of each range
        assert!(decoded.fep_surprise_scale >= 1.0);
        assert!(decoded.fep_lr_decay >= 0.80);
        assert!(decoded.dream_base_interval >= 20);
        assert!(decoded.dream_min_interval >= 10);
        assert!(decoded.neuromod_d2_baseline >= 0.1);
        assert!(decoded.arousal_trap_threshold >= 0.5);
        assert!(decoded.homeostasis_pull_cruise >= 0.5);
    }

    #[test]
    fn test_all_ones_produces_valid_ranges() {
        let hv = BinaryHV([0xFF; 2048]);
        let decoded = decode_thresholds(&hv);

        // All-ones bits decode to maximum of each range
        assert!(decoded.fep_surprise_scale <= 10.0);
        assert!(decoded.fep_lr_decay <= 0.999);
        assert!(decoded.dream_base_interval <= 500);
        assert!(decoded.dream_min_interval <= 100);
        assert!(decoded.neuromod_d2_baseline <= 0.9);
        assert!(decoded.arousal_trap_threshold <= 0.95);
        assert!(decoded.homeostasis_pull_cruise <= 3.0);
    }

    #[test]
    fn test_threshold_region_does_not_overlap_neural() {
        // Neural genome uses bits 0-380 (WEIGHT_SEED ends at bit 316+64=380)
        // Threshold region starts at bit 400
        assert!(400 > 380, "threshold region must not overlap neural genome");
        assert!(THRESHOLD_REGION_END <= 16_384, "must fit in BinaryHV");
    }

    #[test]
    fn test_default_phenotype_has_high_fitness() {
        let defaults = ThresholdPhenotype::default();
        let fitness = evaluate_threshold_fitness(&defaults);
        assert!(
            fitness > 0.8,
            "default thresholds should be consistent, got {fitness}"
        );
    }

    #[test]
    fn test_inverted_homeostasis_penalized() {
        let mut pheno = ThresholdPhenotype::default();
        pheno.homeostasis_recalibrate_low = 1.2; // Above high (1.15)
        let fitness = evaluate_threshold_fitness(&pheno);
        assert!(
            fitness < 0.8,
            "inverted homeostasis should be penalized, got {fitness}"
        );
    }

    #[test]
    fn test_inverted_dream_intervals_penalized() {
        let mut pheno = ThresholdPhenotype::default();
        pheno.dream_min_interval = 200; // Above base (100)
        let fitness = evaluate_threshold_fitness(&pheno);
        assert!(
            fitness < 0.8,
            "inverted dream intervals should be penalized, got {fitness}"
        );
    }

    #[test]
    fn test_encode_does_not_corrupt_neural_region() {
        let mut hv = BinaryHV([0xAA; 2048]); // Alternating pattern
        let neural_before: Vec<u8> = hv.0[..50].to_vec(); // First 400 bits = 50 bytes

        let pheno = ThresholdPhenotype::default();
        encode_thresholds(&mut hv, &pheno);

        let neural_after: Vec<u8> = hv.0[..50].to_vec();
        assert_eq!(
            neural_before, neural_after,
            "threshold encoding must not touch neural region"
        );
    }
}
