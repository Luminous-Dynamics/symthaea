// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-preserving bridge from chemical perception into Symthaea's root
//! multimodal input contract.
//!
//! This crate deliberately owns the dependency seam between the standalone
//! `symthaea-chemosensation` domain and the root `symthaea` cognition package.
//! Neither side needs to depend on the other directly.
//!
//! The bridge does **not** enable olfaction or gustation in the default root
//! topology. It only converts a validated chemical aggregate into a typed,
//! lineaged root input. Root activation remains a separate explicit decision.

use std::fmt;
use std::time::Duration;

use symthaea::consciousness::integration::cross_modal_binding::Modality;
use symthaea::consciousness::integration::modal_lineage::{
    ModalLineageError, ModalLineageReceipt,
};
use symthaea::consciousness::integration::modal_lineage_integration::LineagedModalInput;
use symthaea::consciousness::integration::modality_identity::stable_modality_id;
use symthaea::consciousness::integration::multi_modal_integration::ModalInput;
use symthaea_chemosensation::{
    ChemicalBridgeTarget, ChemicalContentAddressError, ChemicalModalBridgeInput,
    ChemicalRootContentLineage, ChemicalRootProjection, ChemicalRootProjectionError,
    ChemicalRootProjector,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalRootBridgeError {
    Projection(ChemicalRootProjectionError),
    ContentAddress(ChemicalContentAddressError),
    ModalLineage(ModalLineageError),
    StableIdentityMismatch {
        chemical_target_id: u16,
        root_modality_id: u16,
    },
}

impl fmt::Display for ChemicalRootBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Projection(error) => write!(f, "chemical projection failed: {error:?}"),
            Self::ContentAddress(error) => write!(f, "chemical content lineage failed: {error}"),
            Self::ModalLineage(error) => write!(f, "root modal lineage failed: {error}"),
            Self::StableIdentityMismatch {
                chemical_target_id,
                root_modality_id,
            } => write!(
                f,
                "chemical target stable ID {chemical_target_id} does not match root modality stable ID {root_modality_id}"
            ),
        }
    }
}

impl std::error::Error for ChemicalRootBridgeError {}

impl From<ChemicalRootProjectionError> for ChemicalRootBridgeError {
    fn from(value: ChemicalRootProjectionError) -> Self {
        Self::Projection(value)
    }
}

impl From<ChemicalContentAddressError> for ChemicalRootBridgeError {
    fn from(value: ChemicalContentAddressError) -> Self {
        Self::ContentAddress(value)
    }
}

impl From<ModalLineageError> for ChemicalRootBridgeError {
    fn from(value: ModalLineageError) -> Self {
        Self::ModalLineage(value)
    }
}

/// Validated chemical projection paired with the exact root input produced from
/// it. Keeping the projection available preserves agreement/quantization quality
/// diagnostics that do not fit inside the generic root `ModalInput` contract.
pub struct ChemicalRootHandoff {
    projection: ChemicalRootProjection,
    lineaged_input: LineagedModalInput,
}

impl ChemicalRootHandoff {
    pub fn projection(&self) -> &ChemicalRootProjection {
        &self.projection
    }

    pub fn lineaged_input(&self) -> &LineagedModalInput {
        &self.lineaged_input
    }

    pub fn into_parts(self) -> (ChemicalRootProjection, LineagedModalInput) {
        (self.projection, self.lineaged_input)
    }

    pub fn into_lineaged_input(self) -> LineagedModalInput {
        self.lineaged_input
    }
}

/// Map the chemical domain target to the canonical root modality while checking
/// that both subsystems still agree on the explicit stable identity contract.
pub fn root_modality_for_target(
    target: ChemicalBridgeTarget,
) -> Result<Modality, ChemicalRootBridgeError> {
    let modality = match target {
        ChemicalBridgeTarget::Olfactory => Modality::Olfactory,
        ChemicalBridgeTarget::Gustatory => Modality::Gustatory,
    };

    let chemical_target_id = target.stable_id();
    let root_modality_id = stable_modality_id(modality);
    if chemical_target_id != root_modality_id {
        return Err(ChemicalRootBridgeError::StableIdentityMismatch {
            chemical_target_id,
            root_modality_id,
        });
    }

    Ok(modality)
}

/// Convert a validated chemical aggregate into a typed root input.
///
/// Validation is deliberately owned by [`ChemicalRootProjector::project`]. The
/// bridge accepts the pre-projection aggregate rather than trusting a publicly
/// constructible `ChemicalRootProjection`, so forged evidence/space/timestamp
/// fields are rejected by the existing chemical validation path before any root
/// input is created.
pub fn project_to_lineaged_root_input(
    projector: &ChemicalRootProjector,
    chemical_input: &ChemicalModalBridgeInput,
) -> Result<ChemicalRootHandoff, ChemicalRootBridgeError> {
    let projection = projector.project(chemical_input)?;
    let modality = root_modality_for_target(projection.target)?;
    let content_lineage = ChemicalRootContentLineage::from_projection(&projection)?;

    let modal_lineage = ModalLineageReceipt::from_single_evidence(content_lineage.evidence_bundle)
        .with_input_space(content_lineage.input_space)
        .with_transform(content_lineage.projection_policy)?
        .with_output_space(content_lineage.output_space);

    // The chemical bridge's aggregate confidence already includes same-modality
    // sensor disagreement. Root cognition receives that conservative confidence;
    // agreement is preserved separately on the returned projection diagnostics.
    let mut modal_input = ModalInput::new(
        modality,
        projection.binary_vector,
        f64::from(projection.confidence),
    )
    .with_source(source_label(projection.target));

    // Chemical timestamps are microseconds since the acquisition epoch used by
    // the chemical evidence path. Root `ModalInput` stores a Duration, so retain
    // the newest component timestamp exactly at microsecond resolution.
    modal_input.timestamp = Duration::from_micros(projection.latest_timestamp_us);

    Ok(ChemicalRootHandoff {
        projection,
        lineaged_input: LineagedModalInput::new(modal_input, modal_lineage),
    })
}

fn source_label(target: ChemicalBridgeTarget) -> &'static str {
    match target {
        ChemicalBridgeTarget::Olfactory => "symthaea-chemosensation/olfactory",
        ChemicalBridgeTarget::Gustatory => "symthaea-chemosensation/gustatory",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea::consciousness::integration::modal_lineage_history::LineagedMultiModalIntegrator;
    use symthaea::consciousness::integration::modality_identity::{
        GUSTATORY_STABLE_ID, LEGACY_ROOT_MODALITIES, OLFACTORY_STABLE_ID,
    };
    use symthaea::consciousness::integration::multi_modal_integration::{
        IntegrationConfig, IntegrationResult, MultiModalIntegrator, ModalInput,
    };
    use symthaea_chemosensation::{
        CalibrationState, ChannelEncodingSpec, ChemicalChannel, ChemicalFingerprintEncoder,
        ChemicalModalBridge, ChemicalModality, ChemicalObservation, ChemicalPercept,
        MeasurementUnit, SensorHealth,
    };

    fn encoder() -> ChemicalFingerprintEncoder {
        ChemicalFingerprintEncoder::new(vec![ChannelEncodingSpec::new(
            "chemical-signal",
            MeasurementUnit::PartsPerMillion,
            0.0,
            100.0,
            11,
            11,
            101,
        )])
        .unwrap()
    }

    fn percept(
        modality: ChemicalModality,
        timestamp_us: u64,
        value: f32,
        source: &str,
    ) -> ChemicalPercept {
        let encoder = encoder();
        let observation = ChemicalObservation::new(
            timestamp_us,
            modality,
            source,
            vec![ChemicalChannel {
                name: "chemical-signal".into(),
                raw_value: value,
                unit: MeasurementUnit::PartsPerMillion,
                calibration: CalibrationState::identity("cal-v1"),
                health: SensorHealth::default(),
            }],
        );
        let fingerprint = encoder.encode(&observation).unwrap().unwrap();
        ChemicalPercept {
            evidence: observation,
            fingerprint,
        }
    }

    fn chemical_input(modality: ChemicalModality, timestamp_us: u64, value: f32) -> ChemicalModalBridgeInput {
        ChemicalModalBridge::default()
            .aggregate(&[percept(modality, timestamp_us, value, "fixture")])
            .unwrap()
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
    fn chemical_targets_match_reserved_root_ids_without_activation() {
        assert_eq!(
            stable_modality_id(root_modality_for_target(ChemicalBridgeTarget::Olfactory).unwrap()),
            OLFACTORY_STABLE_ID
        );
        assert_eq!(
            stable_modality_id(root_modality_for_target(ChemicalBridgeTarget::Gustatory).unwrap()),
            GUSTATORY_STABLE_ID
        );
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Olfactory));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Gustatory));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Chemesthetic));
    }

    #[test]
    fn validated_projection_preserves_lineage_timestamp_and_confidence() {
        let chemical = chemical_input(ChemicalModality::Olfactory, 123_456, 50.0);
        let handoff = project_to_lineaged_root_input(&ChemicalRootProjector::default(), &chemical)
            .unwrap();

        assert_eq!(handoff.projection().target, ChemicalBridgeTarget::Olfactory);
        assert_eq!(handoff.lineaged_input().input().modality, Modality::Olfactory);
        assert_eq!(
            handoff.lineaged_input().input().timestamp,
            Duration::from_micros(123_456)
        );
        assert_eq!(
            handoff.lineaged_input().input().confidence,
            f64::from(handoff.projection().confidence)
        );
        assert_eq!(
            handoff.lineaged_input().input().features,
            handoff.projection().binary_vector
        );

        let expected = ChemicalRootContentLineage::from_projection(handoff.projection()).unwrap();
        let lineage = handoff.lineaged_input().lineage();
        assert_eq!(lineage.evidence(), std::slice::from_ref(&expected.evidence_bundle));
        assert_eq!(lineage.input_space(), Some(&expected.input_space));
        assert_eq!(
            lineage.transforms(),
            std::slice::from_ref(&expected.projection_policy)
        );
        assert_eq!(lineage.output_space(), Some(&expected.output_space));
    }

    #[test]
    fn defined_but_unconfigured_olfaction_is_observed_without_fusion() {
        let chemical = chemical_input(ChemicalModality::Olfactory, 10, 40.0);
        let handoff = project_to_lineaged_root_input(&ChemicalRootProjector::default(), &chemical)
            .unwrap();
        let mut root = LineagedMultiModalIntegrator::new(IntegrationConfig::default());
        let result = root.integrate(&[handoff.into_lineaged_input()]);

        assert!(root.integrator().active_modalities().contains(&Modality::Olfactory));
        assert!(result.current.processed_lineage_for(Modality::Olfactory).is_some());
        assert!(result.current.fused_lineage_for(Modality::Olfactory).is_none());
        assert!(result.current.integration.active_zones.is_empty());
        assert_eq!(result.current.integration.integrated_phi, 0.0);
        assert!(root.temporal_history(Modality::Olfactory).is_none());
    }

    #[test]
    fn unconfigured_chemical_input_cannot_change_legacy_visual_fusion() {
        let chemical = chemical_input(ChemicalModality::Olfactory, 10, 65.0);
        let handoff = project_to_lineaged_root_input(&ChemicalRootProjector::default(), &chemical)
            .unwrap();
        let visual = ModalInput::new(
            Modality::Visual,
            handoff.projection().binary_vector,
            0.8,
        );

        let mut control = MultiModalIntegrator::new(IntegrationConfig::default());
        let control_result = control.integrate(std::slice::from_ref(&visual));

        let mut treatment = MultiModalIntegrator::new(IntegrationConfig::default());
        let treatment_result = treatment.integrate(&[
            visual,
            handoff.lineaged_input().input().clone(),
        ]);

        assert!(treatment.active_modalities().contains(&Modality::Olfactory));
        assert_same_semantic_result(&control_result, &treatment_result);
    }

    #[test]
    fn gustation_maps_to_gustatory_without_flavor_or_chemesthetic_aliasing() {
        let chemical = chemical_input(ChemicalModality::Gustatory, 77, 25.0);
        let handoff = project_to_lineaged_root_input(&ChemicalRootProjector::default(), &chemical)
            .unwrap();

        assert_eq!(handoff.projection().target, ChemicalBridgeTarget::Gustatory);
        assert_eq!(handoff.lineaged_input().input().modality, Modality::Gustatory);
        assert_ne!(handoff.lineaged_input().input().modality, Modality::Chemesthetic);
    }
}
