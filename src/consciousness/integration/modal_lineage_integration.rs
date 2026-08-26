// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Non-breaking lineage sidecar for root multimodal integration.
//!
//! [`ModalInput`] predates typed content lineage and is a public Rust struct.
//! Adding a new public field would break downstream struct literals even if the
//! field were optional. This module therefore keeps the legacy input and result
//! types unchanged and provides an additive envelope around them.
//!
//! The existing [`MultiModalIntegrator::integrate`] implementation remains the
//! sole owner of fusion math, confidence sanitization, current-cycle truncation,
//! channel availability, and last-observation-wins behavior. This adapter calls
//! that implementation and reconstructs only the lineage disposition from the
//! observable processing contract; it does not duplicate or fork fusion logic.

use std::collections::HashMap;

use super::cross_modal_binding::Modality;
use super::modal_lineage::ModalLineageReceipt;
use super::multi_modal_integration::{IntegrationResult, ModalInput, MultiModalIntegrator};

/// A legacy [`ModalInput`] paired with exact content lineage.
#[derive(Debug, Clone)]
pub struct LineagedModalInput {
    input: ModalInput,
    lineage: ModalLineageReceipt,
}

impl LineagedModalInput {
    pub fn new(input: ModalInput, lineage: ModalLineageReceipt) -> Self {
        Self { input, lineage }
    }

    pub fn input(&self) -> &ModalInput {
        &self.input
    }

    pub fn lineage(&self) -> &ModalLineageReceipt {
        &self.lineage
    }

    pub fn into_parts(self) -> (ModalInput, ModalLineageReceipt) {
        (self.input, self.lineage)
    }
}

/// Result of running the unchanged root integrator with a typed lineage sidecar.
///
/// `processed_lineage` records the lineage of the final processed observation for
/// each modality, matching the root integrator's last-observation-wins policy.
/// It includes zero-confidence observations because those were still observed and
/// entered current-cycle bookkeeping.
///
/// `fused_lineage` is narrower: it includes only processed modalities whose
/// sanitized current-cycle confidence is positive and for which the root result
/// reports a configured channel. It therefore describes lineage that actually
/// reached the root fusion input set, not merely evidence that was presented.
#[derive(Debug, Clone)]
pub struct LineagedIntegrationResult {
    pub integration: IntegrationResult,
    pub processed_lineage: HashMap<Modality, ModalLineageReceipt>,
    pub fused_lineage: HashMap<Modality, ModalLineageReceipt>,
    pub processed_input_count: usize,
}

impl LineagedIntegrationResult {
    pub fn processed_lineage_for(&self, modality: Modality) -> Option<&ModalLineageReceipt> {
        self.processed_lineage.get(&modality)
    }

    pub fn fused_lineage_for(&self, modality: Modality) -> Option<&ModalLineageReceipt> {
        self.fused_lineage.get(&modality)
    }
}

/// Run the existing root multimodal integrator while preserving typed content
/// lineage as a sidecar.
///
/// The function deliberately derives the number of inputs actually processed
/// from the integrator's public `total_inputs` counter. This keeps lineage
/// truncation aligned with `max_channels_per_cycle` without reaching into private
/// integrator configuration or reimplementing its policy.
pub fn integrate_with_lineage(
    integrator: &mut MultiModalIntegrator,
    inputs: &[LineagedModalInput],
) -> LineagedIntegrationResult {
    let before_total_inputs = integrator.stats().total_inputs;
    let modal_inputs: Vec<ModalInput> = inputs.iter().map(|entry| entry.input.clone()).collect();

    let integration = integrator.integrate(&modal_inputs);

    let processed_delta = integrator
        .stats()
        .total_inputs
        .saturating_sub(before_total_inputs);
    let processed_input_count = usize::try_from(processed_delta)
        .unwrap_or(usize::MAX)
        .min(inputs.len());

    // Last processed observation for a modality wins, exactly matching the
    // feature/confidence update policy in MultiModalIntegrator::integrate().
    let mut processed_lineage = HashMap::new();
    for entry in inputs.iter().take(processed_input_count) {
        processed_lineage.insert(entry.input.modality, entry.lineage.clone());
    }

    // `attention_weights` contains an entry only when the modality corresponds
    // to a configured root channel. Positive sanitized confidence is the other
    // condition for inclusion in `channel_inputs` inside the root integrator.
    let mut fused_lineage = HashMap::new();
    for (modality, lineage) in &processed_lineage {
        let positive_confidence = integrator
            .current_confidence(*modality)
            .is_some_and(|confidence| confidence > 0.0);
        let configured_channel_reported = integration
            .attention_weights
            .contains_key(&format!("{modality:?}"));

        if positive_confidence && configured_channel_reported {
            fused_lineage.insert(*modality, lineage.clone());
        }
    }

    LineagedIntegrationResult {
        integration,
        processed_lineage,
        fused_lineage,
        processed_input_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::multi_modal_integration::{
        IntegrationConfig, auditory_input, visual_input,
    };
    use symthaea_core::hdc::binary_hv::BinaryHV;
    use symthaea_evidence_plane::ContentAddress32;

    fn lineage(byte: u8) -> ModalLineageReceipt {
        ModalLineageReceipt::from_single_evidence(
            ContentAddress32::new("blake3-256", "symthaea-lineage-test-v1", [byte; 32])
                .unwrap(),
        )
    }

    fn assert_same_semantic_result(left: &IntegrationResult, right: &IntegrationResult) {
        assert_eq!(left.unified_representation, right.unified_representation);
        assert_eq!(left.integrated_phi, right.integrated_phi);
        assert_eq!(left.binding_coherence, right.binding_coherence);
        assert_eq!(left.attention_weights, right.attention_weights);
        assert_eq!(left.dominant_modality, right.dominant_modality);
        assert_eq!(left.active_zones.len(), right.active_zones.len());

        for (left_zone, right_zone) in left.active_zones.iter().zip(&right.active_zones) {
            assert_eq!(left_zone.level, right_zone.level);
            assert_eq!(left_zone.sources, right_zone.sources);
            assert_eq!(left_zone.binding_strength, right_zone.binding_strength);
            assert_eq!(left_zone.activation, right_zone.activation);
        }
    }

    #[test]
    fn lineaged_path_preserves_legacy_fusion_behavior() {
        let visual = visual_input(BinaryHV::random(11), 0.8);
        let auditory = auditory_input(BinaryHV::random(12), 0.7);

        let mut legacy = MultiModalIntegrator::new(IntegrationConfig::default());
        let legacy_result = legacy.integrate(&[visual.clone(), auditory.clone()]);

        let mut lineaged = MultiModalIntegrator::new(IntegrationConfig::default());
        let lineaged_result = integrate_with_lineage(
            &mut lineaged,
            &[
                LineagedModalInput::new(visual, lineage(1)),
                LineagedModalInput::new(auditory, lineage(2)),
            ],
        );

        assert_same_semantic_result(&legacy_result, &lineaged_result.integration);
        assert_eq!(lineaged_result.processed_input_count, 2);
        assert_eq!(lineaged_result.processed_lineage.len(), 2);
        assert_eq!(lineaged_result.fused_lineage.len(), 2);
    }

    #[test]
    fn final_processed_observation_lineage_wins_for_duplicate_modality() {
        let first = LineagedModalInput::new(
            visual_input(BinaryHV::random(21), 1.0),
            lineage(1),
        );
        let second_lineage = lineage(2);
        let second = LineagedModalInput::new(
            visual_input(BinaryHV::random(22), 1.0),
            second_lineage.clone(),
        );

        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let result = integrate_with_lineage(&mut integrator, &[first, second]);

        assert_eq!(
            result.processed_lineage_for(Modality::Visual),
            Some(&second_lineage)
        );
        assert_eq!(
            result.fused_lineage_for(Modality::Visual),
            Some(&second_lineage)
        );
    }

    #[test]
    fn lineage_respects_root_max_input_truncation() {
        let config = IntegrationConfig {
            max_channels_per_cycle: 1,
            ..IntegrationConfig::default()
        };
        let mut integrator = MultiModalIntegrator::new(config);

        let result = integrate_with_lineage(
            &mut integrator,
            &[
                LineagedModalInput::new(
                    visual_input(BinaryHV::random(31), 1.0),
                    lineage(1),
                ),
                LineagedModalInput::new(
                    auditory_input(BinaryHV::random(32), 1.0),
                    lineage(2),
                ),
            ],
        );

        assert_eq!(result.processed_input_count, 1);
        assert!(result.processed_lineage.contains_key(&Modality::Visual));
        assert!(!result.processed_lineage.contains_key(&Modality::Auditory));
        assert!(!result.fused_lineage.contains_key(&Modality::Auditory));
    }

    #[test]
    fn zero_confidence_lineage_is_processed_but_not_fused() {
        let expected = lineage(7);
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let result = integrate_with_lineage(
            &mut integrator,
            &[LineagedModalInput::new(
                visual_input(BinaryHV::random(41), 0.0),
                expected.clone(),
            )],
        );

        assert_eq!(
            result.processed_lineage_for(Modality::Visual),
            Some(&expected)
        );
        assert!(result.fused_lineage_for(Modality::Visual).is_none());
    }

    #[test]
    fn unconfigured_modality_lineage_is_processed_but_not_fused() {
        // Abstract is a defined modality identity but is not part of the legacy
        // root channel set returned by Modality::all(). This test protects the
        // distinction established by the stable-modality contract.
        let expected = lineage(9);
        let input = ModalInput::new(Modality::Abstract, BinaryHV::random(51), 1.0);
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let result = integrate_with_lineage(
            &mut integrator,
            &[LineagedModalInput::new(input, expected.clone())],
        );

        assert_eq!(
            result.processed_lineage_for(Modality::Abstract),
            Some(&expected)
        );
        assert!(result.fused_lineage_for(Modality::Abstract).is_none());
    }
}
