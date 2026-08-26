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
//!
//! ## Scope of the lineage sidecar
//!
//! Current-state fusion and temporal processing are deliberately distinguished.
//! The root integrator pushes every accepted input into a modality's temporal
//! buffer, including same-cycle duplicate modalities, before the final
//! observation becomes the current feature/confidence for that modality.
//! `processed_sequence` therefore preserves every accepted lineage in exact
//! processing order, while `processed_lineage` and `fused_lineage` describe the
//! final current-cycle state per modality.
//!
//! This sidecar is not yet a complete historical causal receipt for metrics such
//! as temporal binding coherence, because those metrics may also depend on
//! modality-buffer evidence retained from earlier integration cycles. It never
//! claims otherwise.

use std::collections::HashMap;

use super::cross_modal_binding::Modality;
use super::modal_lineage::ModalLineageReceipt;
use super::modality_identity::LEGACY_ROOT_MODALITIES;
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

/// One accepted current-cycle input in the exact order processed by the root
/// integrator.
///
/// The ordered sequence is semantically important because duplicate inputs for
/// one modality all enter that modality's temporal buffer even though only the
/// final observation becomes the current feature/confidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessedModalLineage {
    pub modality: Modality,
    pub lineage: ModalLineageReceipt,
}

/// Result of running the unchanged root integrator with a typed lineage sidecar.
///
/// `processed_sequence` records every accepted input lineage in processing order.
/// It is the current-cycle lineage surface that preserves duplicate-modality
/// temporal effects.
///
/// `processed_lineage` records the lineage of the final processed observation for
/// each modality, matching the root integrator's current-state
/// last-observation-wins policy. It includes zero-confidence observations because
/// those were still observed and entered current-cycle bookkeeping.
///
/// `fused_lineage` is narrower still: it includes only final processed modalities
/// whose sanitized current-cycle confidence is positive and which belong to the
/// explicitly pinned root channel topology. It therefore describes lineage
/// behind the current-cycle feature vectors that reached the fusion input set.
///
/// None of these fields claim complete historical lineage for temporal metrics
/// that also depend on observations retained from earlier cycles.
#[derive(Debug, Clone)]
pub struct LineagedIntegrationResult {
    pub integration: IntegrationResult,
    pub processed_sequence: Vec<ProcessedModalLineage>,
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

    pub fn processed_sequence_for(
        &self,
        modality: Modality,
    ) -> impl Iterator<Item = &ProcessedModalLineage> {
        self.processed_sequence
            .iter()
            .filter(move |entry| entry.modality == modality)
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

    let processed_entries = inputs.iter().take(processed_input_count);
    let mut processed_sequence = Vec::with_capacity(processed_input_count);
    let mut processed_lineage = HashMap::new();

    // Preserve every accepted input lineage in order because each accepted
    // duplicate enters the modality temporal buffer. The map separately records
    // only the final observation for current-state semantics.
    for entry in processed_entries {
        processed_sequence.push(ProcessedModalLineage {
            modality: entry.input.modality,
            lineage: entry.lineage.clone(),
        });
        processed_lineage.insert(entry.input.modality, entry.lineage.clone());
    }

    // The stable identity module explicitly pins the set instantiated by the
    // root integrator. Use that typed topology contract instead of inferring
    // channel configuration from debug-formatted telemetry keys.
    let mut fused_lineage = HashMap::new();
    for (modality, lineage) in &processed_lineage {
        let positive_confidence = integrator
            .current_confidence(*modality)
            .is_some_and(|confidence| confidence > 0.0);
        let configured_channel = LEGACY_ROOT_MODALITIES.contains(modality);

        if positive_confidence && configured_channel {
            fused_lineage.insert(*modality, lineage.clone());
        }
    }

    LineagedIntegrationResult {
        integration,
        processed_sequence,
        processed_lineage,
        fused_lineage,
        processed_input_count,
    }
}

#[cfg(test)]
mod tests {
    use super::super::multi_modal_integration::{
        IntegrationConfig, auditory_input, visual_input,
    };
    use super::*;
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
        assert_eq!(lineaged_result.processed_sequence.len(), 2);
        assert_eq!(lineaged_result.processed_lineage.len(), 2);
        assert_eq!(lineaged_result.fused_lineage.len(), 2);
    }

    #[test]
    fn duplicate_modalities_preserve_ordered_lineage_and_final_state_lineage() {
        let first_lineage = lineage(1);
        let first = LineagedModalInput::new(
            visual_input(BinaryHV::random(21), 1.0),
            first_lineage.clone(),
        );
        let second_lineage = lineage(2);
        let second = LineagedModalInput::new(
            visual_input(BinaryHV::random(22), 1.0),
            second_lineage.clone(),
        );

        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let result = integrate_with_lineage(&mut integrator, &[first, second]);

        let visual_sequence: Vec<_> = result.processed_sequence_for(Modality::Visual).collect();
        assert_eq!(visual_sequence.len(), 2);
        assert_eq!(visual_sequence[0].lineage, first_lineage);
        assert_eq!(visual_sequence[1].lineage, second_lineage);
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
        assert_eq!(result.processed_sequence.len(), 1);
        assert_eq!(result.processed_sequence[0].modality, Modality::Visual);
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

        assert_eq!(result.processed_sequence.len(), 1);
        assert_eq!(result.processed_sequence[0].lineage, expected);
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

        assert_eq!(result.processed_sequence.len(), 1);
        assert_eq!(
            result.processed_lineage_for(Modality::Abstract),
            Some(&expected)
        );
        assert!(result.fused_lineage_for(Modality::Abstract).is_none());
    }
}
