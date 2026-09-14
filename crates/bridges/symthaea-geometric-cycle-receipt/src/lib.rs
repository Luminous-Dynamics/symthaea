// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical deterministic per-cycle scientific-state receipts for GEOM D0.
//!
//! The receipt intentionally excludes wall-clock/performance/presentation fields.
//! It stores floating-point values as raw IEEE bit patterns so the sham gate is
//! exact rather than tolerance-based.

use serde::{Deserialize, Serialize};
use symthaea::cognitive_loop::CycleResult;

const RECEIPT_DOMAIN: &[u8] = b"SYMTHAEA_GEOM_SCIENTIFIC_CYCLE_RECEIPT_V1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificCycleReceipt {
    // Native cycle result.
    pub output_bits: Vec<u32>,
    pub prediction_error_bits: u32,
    pub peak_attention_bits: u32,
    pub detected_primitives: Vec<String>,
    pub learning_occurred: bool,
    pub training_loss_bits: Option<u32>,
    pub bits_saved_persist_bits: Option<u32>,
    pub bits_saved_zero_bits: Option<u32>,
    pub bits_kappa_bits: Option<u32>,
    pub recall_fired: bool,
    pub recall_similarity_bits: Option<u32>,
    pub recall_matched_timestamp: Option<u64>,
    pub thought_vector_bits: Vec<u32>,
    pub wisdom_hv_blake3: [u8; 32],

    // Strategy/control state.
    pub selected_strategy: String,
    pub reasoning_confidence_bits: u32,
    pub actual_effective_lr_bits: u32,

    // GWT state and built-in delivery side effects.
    pub gwt_broadcast: bool,
    pub gwt_coalition_size: u32,
    pub attention_budget_exceeded: bool,
    pub gwt_memory_consolidation_requested: bool,
    pub gwt_perception_broadcasts: u32,

    // Core cognitive dynamics used by existing longitudinal instrumentation.
    pub consciousness_level_bits: u64,
    pub equation_v2_consciousness_bits: u64,
    pub fep_action: u64,
    pub fep_surprise_bits: u64,
    pub fep_td_error_bits: u64,
    pub exploration_mod_bits: u32,
    pub coherence_velocity_bits: u32,
    pub meta_cognitive_accuracy_bits: u32,

    // Clock/neuromod propagation checks (logical state, not wall time).
    pub circadian_phase: String,
    pub circadian_plasticity_bits: u32,
    pub circadian_hour_bits: u32,
    pub circadian_effective_hour_bits: u32,
    pub circadian_timezone_offset_bits: u32,
    pub dopamine_effective_bits: u32,
    pub noradrenaline_effective_bits: u32,
    pub serotonin_effective_bits: u32,
    pub acetylcholine_effective_bits: u32,
}

impl ScientificCycleReceipt {
    /// Project one real cycle into the frozen deterministic receipt surface.
    ///
    /// Deliberately NOT read here: cycle_time_us, metadata.cycle_duration_us,
    /// module_timings_us, language/canvas/render output, signatures, or network
    /// transport metadata.
    pub fn from_cycle_result(result: &CycleResult) -> Self {
        let m = &result.metadata;
        let wisdom_hv_blake3 = *blake3::hash(&result.wisdom_hv.0).as_bytes();

        Self {
            output_bits: result.output.iter().map(|v| v.to_bits()).collect(),
            prediction_error_bits: result.prediction_error.to_bits(),
            peak_attention_bits: result.peak_attention.to_bits(),
            detected_primitives: result.detected_primitives.clone(),
            learning_occurred: result.learning_occurred,
            training_loss_bits: result.training_loss.map(f32::to_bits),
            bits_saved_persist_bits: result.bits_saved_persist.map(f32::to_bits),
            bits_saved_zero_bits: result.bits_saved_zero.map(f32::to_bits),
            bits_kappa_bits: result.bits_kappa.map(f32::to_bits),
            recall_fired: result.recall_fired,
            recall_similarity_bits: result.recall_similarity.map(f32::to_bits),
            recall_matched_timestamp: result.recall_matched_timestamp,
            thought_vector_bits: result.thought_vector.iter().map(|v| v.to_bits()).collect(),
            wisdom_hv_blake3,

            selected_strategy: m.selected_strategy.clone(),
            reasoning_confidence_bits: m.reasoning_confidence.to_bits(),
            actual_effective_lr_bits: m.actual_effective_lr.to_bits(),

            gwt_broadcast: m.attention.gwt_broadcast,
            gwt_coalition_size: m.attention.gwt_coalition_size,
            attention_budget_exceeded: m.attention.attention_budget_exceeded,
            gwt_memory_consolidation_requested: m.gwt_memory_consolidation_requested,
            gwt_perception_broadcasts: m.gwt_perception_broadcasts,

            consciousness_level_bits: m.consciousness.consciousness_level.to_bits(),
            equation_v2_consciousness_bits: m.quality.equation_v2_consciousness.to_bits(),
            fep_action: m.fep.fep_action as u64,
            fep_surprise_bits: m.fep.fep_surprise.to_bits(),
            fep_td_error_bits: m.fep.fep_td_error.to_bits(),
            exploration_mod_bits: m.neuromod.neuromod_mcts_exploration_mod.to_bits(),
            coherence_velocity_bits: m.quality.coherence_velocity.to_bits(),
            meta_cognitive_accuracy_bits: m.quality.meta_cognitive_accuracy.to_bits(),

            circadian_phase: m.circadian_phase.clone(),
            circadian_plasticity_bits: m.circadian_plasticity.to_bits(),
            circadian_hour_bits: m.neuromod.circadian_hour.to_bits(),
            circadian_effective_hour_bits: m.neuromod.circadian_effective_hour.to_bits(),
            circadian_timezone_offset_bits: m.neuromod.circadian_timezone_offset.to_bits(),
            dopamine_effective_bits: m.neuromod.dopamine_effective.to_bits(),
            noradrenaline_effective_bits: m.neuromod.noradrenaline_effective.to_bits(),
            serotonin_effective_bits: m.neuromod.serotonin_effective.to_bits(),
            acetylcholine_effective_bits: m.neuromod.acetylcholine_effective.to_bits(),
        }
    }

    /// Domain-separated canonical digest of the typed receipt.
    pub fn digest(&self) -> Result<[u8; 32], bincode::Error> {
        let encoded = bincode::serialize(self)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(RECEIPT_DOMAIN);
        hasher.update(&encoded);
        Ok(*hasher.finalize().as_bytes())
    }

    pub fn digest_hex(&self) -> Result<String, bincode::Error> {
        Ok(blake3::Hash::from_bytes(self.digest()?).to_hex().to_string())
    }

    /// Return exact typed field names that differ between two receipts.
    pub fn differing_fields(&self, other: &Self) -> Vec<&'static str> {
        let mut fields = Vec::new();
        macro_rules! diff {
            ($field:ident) => {
                if self.$field != other.$field {
                    fields.push(stringify!($field));
                }
            };
        }

        diff!(output_bits);
        diff!(prediction_error_bits);
        diff!(peak_attention_bits);
        diff!(detected_primitives);
        diff!(learning_occurred);
        diff!(training_loss_bits);
        diff!(bits_saved_persist_bits);
        diff!(bits_saved_zero_bits);
        diff!(bits_kappa_bits);
        diff!(recall_fired);
        diff!(recall_similarity_bits);
        diff!(recall_matched_timestamp);
        diff!(thought_vector_bits);
        diff!(wisdom_hv_blake3);
        diff!(selected_strategy);
        diff!(reasoning_confidence_bits);
        diff!(actual_effective_lr_bits);
        diff!(gwt_broadcast);
        diff!(gwt_coalition_size);
        diff!(attention_budget_exceeded);
        diff!(gwt_memory_consolidation_requested);
        diff!(gwt_perception_broadcasts);
        diff!(consciousness_level_bits);
        diff!(equation_v2_consciousness_bits);
        diff!(fep_action);
        diff!(fep_surprise_bits);
        diff!(fep_td_error_bits);
        diff!(exploration_mod_bits);
        diff!(coherence_velocity_bits);
        diff!(meta_cognitive_accuracy_bits);
        diff!(circadian_phase);
        diff!(circadian_plasticity_bits);
        diff!(circadian_hour_bits);
        diff!(circadian_effective_hour_bits);
        diff!(circadian_timezone_offset_bits);
        diff!(dopamine_effective_bits);
        diff!(noradrenaline_effective_bits);
        diff!(serotonin_effective_bits);
        diff!(acetylcholine_effective_bits);
        fields
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_receipt() -> ScientificCycleReceipt {
        ScientificCycleReceipt {
            output_bits: vec![0.1f32.to_bits(), (-0.2f32).to_bits()],
            prediction_error_bits: 0.3f32.to_bits(),
            peak_attention_bits: 0.4f32.to_bits(),
            detected_primitives: vec!["CAUSE".into(), "TIME".into()],
            learning_occurred: false,
            training_loss_bits: None,
            bits_saved_persist_bits: Some(0.2f32.to_bits()),
            bits_saved_zero_bits: Some(0.1f32.to_bits()),
            bits_kappa_bits: Some(2.0f32.to_bits()),
            recall_fired: false,
            recall_similarity_bits: None,
            recall_matched_timestamp: None,
            thought_vector_bits: vec![0.25f32.to_bits(); 32],
            wisdom_hv_blake3: *blake3::hash(b"wisdom-a").as_bytes(),
            selected_strategy: "Exploratory".into(),
            reasoning_confidence_bits: 0.6f32.to_bits(),
            actual_effective_lr_bits: 0.0f32.to_bits(),
            gwt_broadcast: true,
            gwt_coalition_size: 3,
            attention_budget_exceeded: false,
            gwt_memory_consolidation_requested: true,
            gwt_perception_broadcasts: 1,
            consciousness_level_bits: 0.7f64.to_bits(),
            equation_v2_consciousness_bits: 0.65f64.to_bits(),
            fep_action: 2,
            fep_surprise_bits: 0.12f64.to_bits(),
            fep_td_error_bits: (-0.03f64).to_bits(),
            exploration_mod_bits: 1.1f32.to_bits(),
            coherence_velocity_bits: 0.02f32.to_bits(),
            meta_cognitive_accuracy_bits: 0.8f32.to_bits(),
            circadian_phase: "Day".into(),
            circadian_plasticity_bits: 0.9f32.to_bits(),
            circadian_hour_bits: 12.0f32.to_bits(),
            circadian_effective_hour_bits: 14.0f32.to_bits(),
            circadian_timezone_offset_bits: 2.0f32.to_bits(),
            dopamine_effective_bits: 1.0f32.to_bits(),
            noradrenaline_effective_bits: 1.0f32.to_bits(),
            serotonin_effective_bits: 1.0f32.to_bits(),
            acetylcholine_effective_bits: 1.0f32.to_bits(),
        }
    }

    #[test]
    fn identical_receipts_have_identical_domain_separated_digest() {
        let a = sample_receipt();
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(a.digest().unwrap(), b.digest().unwrap());
        assert!(a.differing_fields(&b).is_empty());
    }

    #[test]
    fn one_bit_float_difference_is_visible() {
        let a = sample_receipt();
        let mut b = a.clone();
        b.prediction_error_bits ^= 1;
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
        assert_eq!(a.differing_fields(&b), vec!["prediction_error_bits"]);
    }

    #[test]
    fn wisdom_hv_commitment_difference_is_visible() {
        let a = sample_receipt();
        let mut b = a.clone();
        b.wisdom_hv_blake3 = *blake3::hash(b"wisdom-b").as_bytes();
        assert_eq!(a.differing_fields(&b), vec!["wisdom_hv_blake3"]);
    }

    #[test]
    fn gwt_and_control_families_participate_in_exact_equality() {
        let a = sample_receipt();
        let mut b = a.clone();
        b.selected_strategy = "Supportive".into();
        b.gwt_broadcast = false;
        b.gwt_coalition_size = 0;
        b.gwt_memory_consolidation_requested = false;
        b.reasoning_confidence_bits ^= 1;
        assert_eq!(
            a.differing_fields(&b),
            vec![
                "selected_strategy",
                "reasoning_confidence_bits",
                "gwt_broadcast",
                "gwt_coalition_size",
                "gwt_memory_consolidation_requested",
            ]
        );
    }

    #[test]
    fn neuromod_and_circadian_families_participate_in_exact_equality() {
        let a = sample_receipt();
        let mut b = a.clone();
        b.circadian_effective_hour_bits ^= 1;
        b.dopamine_effective_bits ^= 1;
        assert_eq!(
            a.differing_fields(&b),
            vec!["circadian_effective_hour_bits", "dopamine_effective_bits"]
        );
    }
}
