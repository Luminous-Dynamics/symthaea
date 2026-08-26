// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Stateful lineage history for root multimodal integration.
//!
//! [`integrate_with_lineage`](super::modal_lineage_integration::integrate_with_lineage)
//! preserves exact lineage for one current cycle. Temporal coherence, however,
//! can also depend on observations retained in modality buffers from earlier
//! cycles. This wrapper owns both the root integrator and a parallel bounded
//! lineage history so those historical dependencies remain inspectable.
//!
//! It does not alter fusion math, attention, confidence, or root topology.

use std::collections::{HashMap, HashSet, VecDeque};

use super::cross_modal_binding::Modality;
use super::modal_lineage::ModalLineageReceipt;
use super::modal_lineage_integration::{
    LineagedIntegrationResult, LineagedModalInput, integrate_with_lineage,
};
use super::modality_identity::LEGACY_ROOT_MODALITIES;
use super::multi_modal_integration::{IntegrationConfig, MultiModalIntegrator};

/// Capacity of the current root `ModalityChannel` temporal buffer.
///
/// This is a structural mirror, not a scientific threshold. The regression test
/// below deliberately checks it against the real channel behavior so any future
/// root-capacity change fails loudly instead of silently desynchronizing lineage.
pub const ROOT_MODAL_TEMPORAL_LINEAGE_CAPACITY: usize = 8;

/// One lineaged integration result plus the exact retained lineage histories for
/// modalities whose current temporal coherence participated in this cycle.
#[derive(Debug, Clone)]
pub struct HistoricalLineagedIntegrationResult {
    pub current: LineagedIntegrationResult,
    pub temporal_context_lineage: HashMap<Modality, Vec<ModalLineageReceipt>>,
}

impl HistoricalLineagedIntegrationResult {
    pub fn temporal_lineage_for(
        &self,
        modality: Modality,
    ) -> Option<&[ModalLineageReceipt]> {
        self.temporal_context_lineage
            .get(&modality)
            .map(Vec::as_slice)
    }
}

/// Root multimodal integrator with synchronized evidence-lineage history.
///
/// The inner integrator is intentionally exposed read-only. Mutable access would
/// allow callers to process or reset sensory state without updating the lineage
/// mirror, destroying the invariant this wrapper exists to provide.
pub struct LineagedMultiModalIntegrator {
    inner: MultiModalIntegrator,
    temporal_lineage: HashMap<Modality, VecDeque<ModalLineageReceipt>>,
}

impl LineagedMultiModalIntegrator {
    pub fn new(config: IntegrationConfig) -> Self {
        Self {
            inner: MultiModalIntegrator::new(config),
            temporal_lineage: HashMap::new(),
        }
    }

    pub fn integrator(&self) -> &MultiModalIntegrator {
        &self.inner
    }

    pub fn temporal_history(
        &self,
        modality: Modality,
    ) -> Option<&VecDeque<ModalLineageReceipt>> {
        self.temporal_lineage.get(&modality)
    }

    /// Process one current cycle while keeping the lineage mirror synchronized
    /// with the root modality temporal buffers.
    pub fn integrate(
        &mut self,
        inputs: &[LineagedModalInput],
    ) -> HistoricalLineagedIntegrationResult {
        let current = integrate_with_lineage(&mut self.inner, inputs);

        // The stable identity module pins the exact set instantiated by the root
        // multimodal integrator. Use that typed topology contract rather than
        // deriving channel configuration from debug-formatted telemetry keys.
        let configured_modalities: HashSet<Modality> = current
            .processed_lineage
            .keys()
            .copied()
            .filter(|modality| LEGACY_ROOT_MODALITIES.contains(modality))
            .collect();

        for processed in &current.processed_sequence {
            if !configured_modalities.contains(&processed.modality) {
                continue;
            }

            let history = self.temporal_lineage.entry(processed.modality).or_default();
            if history.len() >= ROOT_MODAL_TEMPORAL_LINEAGE_CAPACITY {
                history.pop_front();
            }
            history.push_back(processed.lineage.clone());
        }

        // `compute_binding_coherence()` currently evaluates only configured
        // modalities present this cycle with positive sanitized confidence. Its
        // temporal buffer can nevertheless contain earlier zero-confidence
        // observations, so the complete retained history is surfaced here.
        let mut temporal_context_lineage = HashMap::new();
        for modality in configured_modalities {
            let participates_in_current_temporal_metric = self
                .inner
                .current_confidence(modality)
                .is_some_and(|confidence| confidence > 0.0);

            if !participates_in_current_temporal_metric {
                continue;
            }

            if let Some(history) = self.temporal_lineage.get(&modality) {
                temporal_context_lineage.insert(
                    modality,
                    history.iter().cloned().collect::<Vec<_>>(),
                );
            }
        }

        HistoricalLineagedIntegrationResult {
            current,
            temporal_context_lineage,
        }
    }

    /// Reset sensory state and lineage history atomically.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.temporal_lineage.clear();
    }

    /// Consume the wrapper and recover the legacy integrator. Historical lineage
    /// is intentionally discarded by this operation.
    pub fn into_inner(self) -> MultiModalIntegrator {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::super::cross_modal_binding::ModalityChannel;
    use super::super::multi_modal_integration::{IntegrationConfig, visual_input};
    use super::*;
    use symthaea_core::hdc::binary_hv::BinaryHV;
    use symthaea_evidence_plane::ContentAddress32;

    fn lineage(byte: u8) -> ModalLineageReceipt {
        ModalLineageReceipt::from_single_evidence(
            ContentAddress32::new(
                "blake3-256",
                "symthaea-lineage-history-test-v1",
                [byte; 32],
            )
            .unwrap(),
        )
    }

    #[test]
    fn mirrored_capacity_matches_root_modality_channel_behavior() {
        let mut channel = ModalityChannel::new(Modality::Visual);
        for seed in 0..(ROOT_MODAL_TEMPORAL_LINEAGE_CAPACITY + 3) {
            channel.update(BinaryHV::random(seed as u64));
        }
        assert_eq!(
            channel.temporal_buffer.len(),
            ROOT_MODAL_TEMPORAL_LINEAGE_CAPACITY
        );
    }

    #[test]
    fn historical_lineage_survives_across_cycles_in_root_buffer_order() {
        let mut integrator = LineagedMultiModalIntegrator::new(IntegrationConfig::default());
        let first = lineage(1);
        let second = lineage(2);

        let _ = integrator.integrate(&[LineagedModalInput::new(
            visual_input(BinaryHV::random(11), 1.0),
            first.clone(),
        )]);
        let result = integrator.integrate(&[LineagedModalInput::new(
            visual_input(BinaryHV::random(12), 1.0),
            second.clone(),
        )]);

        assert_eq!(
            result.temporal_lineage_for(Modality::Visual),
            Some([first, second].as_slice())
        );
    }

    #[test]
    fn zero_confidence_history_is_retained_for_later_temporal_context() {
        let mut integrator = LineagedMultiModalIntegrator::new(IntegrationConfig::default());
        let untrusted = lineage(3);
        let trusted = lineage(4);

        let first = integrator.integrate(&[LineagedModalInput::new(
            visual_input(BinaryHV::random(21), 0.0),
            untrusted.clone(),
        )]);
        assert!(first.temporal_lineage_for(Modality::Visual).is_none());

        let second = integrator.integrate(&[LineagedModalInput::new(
            visual_input(BinaryHV::random(22), 1.0),
            trusted.clone(),
        )]);

        assert_eq!(
            second.temporal_lineage_for(Modality::Visual),
            Some([untrusted, trusted].as_slice())
        );
    }

    #[test]
    fn same_cycle_duplicates_are_both_retained_in_temporal_history() {
        let mut integrator = LineagedMultiModalIntegrator::new(IntegrationConfig::default());
        let first = lineage(5);
        let second = lineage(6);

        let result = integrator.integrate(&[
            LineagedModalInput::new(
                visual_input(BinaryHV::random(31), 1.0),
                first.clone(),
            ),
            LineagedModalInput::new(
                visual_input(BinaryHV::random(32), 1.0),
                second.clone(),
            ),
        ]);

        assert_eq!(
            result.temporal_lineage_for(Modality::Visual),
            Some([first, second].as_slice())
        );
    }

    #[test]
    fn lineage_history_is_bounded_with_oldest_eviction() {
        let mut integrator = LineagedMultiModalIntegrator::new(IntegrationConfig::default());

        for byte in 0..(ROOT_MODAL_TEMPORAL_LINEAGE_CAPACITY as u8 + 2) {
            let _ = integrator.integrate(&[LineagedModalInput::new(
                visual_input(BinaryHV::random(byte as u64 + 100), 1.0),
                lineage(byte),
            )]);
        }

        let history = integrator.temporal_history(Modality::Visual).unwrap();
        assert_eq!(history.len(), ROOT_MODAL_TEMPORAL_LINEAGE_CAPACITY);
        assert_eq!(history.front(), Some(&lineage(2)));
    }

    #[test]
    fn reset_clears_root_and_lineage_history_together() {
        let mut integrator = LineagedMultiModalIntegrator::new(IntegrationConfig::default());
        let _ = integrator.integrate(&[LineagedModalInput::new(
            visual_input(BinaryHV::random(41), 1.0),
            lineage(9),
        )]);
        assert!(integrator.temporal_history(Modality::Visual).is_some());

        integrator.reset();

        assert!(integrator.temporal_history(Modality::Visual).is_none());
        assert!(integrator.integrator().active_modalities().is_empty());
    }
}
