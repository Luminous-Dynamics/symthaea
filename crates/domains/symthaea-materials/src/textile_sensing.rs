// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence admission contracts for textile visual and near-infrared sensing.
//!
//! This module deliberately separates visual garment-family observations from
//! spectral fibre-composition observations. It validates source/instrument,
//! calibration, wavelength, specimen and composition contracts; it does not
//! implement a spectral classifier or authorize physical sorting/recycling.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Integer scale for fibre mass fractions: one million parts represent 100%.
pub const FRACTION_SCALE_PPM: u64 = 1_000_000;

/// Fibre families that a qualified composition method may report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FibreKind {
    /// Cotton fibre.
    Cotton,
    /// Polyester fibre.
    Polyester,
    /// Wool fibre.
    Wool,
    /// Nylon or polyamide fibre.
    Nylon,
    /// Viscose or rayon fibre.
    Viscose,
    /// Lyocell fibre.
    Lyocell,
    /// Acrylic fibre.
    Acrylic,
    /// Elastane or spandex fibre.
    Elastane,
    /// Another declared fibre not represented by this vocabulary.
    Other,
    /// An unresolved fraction retained explicitly rather than guessed.
    Unknown,
}

/// One bounded fibre fraction in a composition estimate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FibreFraction {
    /// Declared fibre identity.
    pub kind: FibreKind,
    /// Point estimate in parts per million of total fibre mass.
    pub fraction_ppm: u32,
    /// Lower uncertainty bound in parts per million.
    pub lower_bound_ppm: u32,
    /// Upper uncertainty bound in parts per million.
    pub upper_bound_ppm: u32,
}

/// A specimen-level fibre composition estimate with explicit uncertainty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionEstimate {
    /// Method or model profile that generated this composition estimate.
    pub method_ref: String,
    /// Fractions whose point estimates must sum to exactly one million ppm.
    pub components: Vec<FibreFraction>,
}

impl CompositionEstimate {
    /// Validate mass-fraction closure, unique fibres and interval consistency.
    pub fn validate(&self) -> Result<(), CompositionError> {
        if self.method_ref.trim().is_empty() {
            return Err(CompositionError::MissingMethodReference);
        }
        if self.components.is_empty() {
            return Err(CompositionError::EmptyComposition);
        }

        let mut seen = BTreeSet::new();
        let mut total = 0_u64;
        let mut lower_total = 0_u64;
        let mut upper_total = 0_u64;

        for component in &self.components {
            if !seen.insert(component.kind) {
                return Err(CompositionError::DuplicateFibreKind(component.kind));
            }
            if component.lower_bound_ppm > component.fraction_ppm
                || component.fraction_ppm > component.upper_bound_ppm
                || component.upper_bound_ppm as u64 > FRACTION_SCALE_PPM
            {
                return Err(CompositionError::InvalidFractionBounds(component.kind));
            }

            total = total
                .checked_add(component.fraction_ppm as u64)
                .ok_or(CompositionError::FractionArithmeticOverflow)?;
            lower_total = lower_total
                .checked_add(component.lower_bound_ppm as u64)
                .ok_or(CompositionError::FractionArithmeticOverflow)?;
            upper_total = upper_total
                .checked_add(component.upper_bound_ppm as u64)
                .ok_or(CompositionError::FractionArithmeticOverflow)?;
        }

        if total != FRACTION_SCALE_PPM {
            return Err(CompositionError::FractionTotalMismatch { actual_ppm: total });
        }
        if lower_total > FRACTION_SCALE_PPM || upper_total < FRACTION_SCALE_PPM {
            return Err(CompositionError::UncertaintyBoundsDoNotCoverTotal);
        }

        Ok(())
    }

    /// Whether any non-zero fraction remains explicitly unidentified.
    pub fn contains_unknown_fraction(&self) -> bool {
        self.components
            .iter()
            .any(|component| component.kind == FibreKind::Unknown && component.upper_bound_ppm > 0)
    }
}

/// Structural failures in a declared composition estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionError {
    /// No method identity was supplied.
    MissingMethodReference,
    /// No fibre components were supplied.
    EmptyComposition,
    /// The same fibre identity occurs more than once.
    DuplicateFibreKind(FibreKind),
    /// A lower/point/upper interval is incoherent or exceeds 100%.
    InvalidFractionBounds(FibreKind),
    /// Nominal fractions do not close to exactly 100%.
    FractionTotalMismatch {
        /// Observed total in parts per million.
        actual_ppm: u64,
    },
    /// Uncertainty intervals cannot contain a 100% composition.
    UncertaintyBoundsDoNotCoverTotal,
    /// Checked integer summation overflowed.
    FractionArithmeticOverflow,
}

/// Result state for one sensor observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationDisposition {
    /// Required identity, calibration, domain and evidence checks passed.
    Admitted,
    /// Evidence is incomplete or outside the qualified envelope; abstain.
    Unknown,
    /// A declared contract is malformed or contradictory.
    Rejected,
}

/// Quality state reported by the NIR instrument or upstream acquisition layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectralQuality {
    /// Acquisition passed the instrument's declared signal-quality checks.
    Valid,
    /// Signal was too weak for the declared profile.
    LowSignal,
    /// One or more channels saturated.
    Saturated,
    /// Observation is outside the sensor/model's qualified input domain.
    OutOfDomain,
    /// Instrument or acquisition reported a fault.
    SensorFault,
}

/// Declared operating range and calibration-profile identity for one NIR instrument.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NirInstrumentProfile {
    /// Stable instrument identity.
    pub instrument_id: String,
    /// Calibration profile required for this instrument configuration.
    pub calibration_profile_id: String,
    /// Lower wavelength boundary of the qualified profile, in nanometres.
    pub minimum_wavelength_nm: u32,
    /// Upper wavelength boundary of the qualified profile, in nanometres.
    pub maximum_wavelength_nm: u32,
}

/// A calibration record whose validity is evaluated at acquisition time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NirCalibration {
    /// Stable calibration-profile identity.
    pub calibration_profile_id: String,
    /// Instrument identity to which the calibration applies.
    pub instrument_id: String,
    /// Inclusive start of calibration validity, in nanoseconds from the declared epoch.
    pub valid_from_ns: u64,
    /// Inclusive end of calibration validity, or no scheduled end.
    pub valid_until_ns: Option<u64>,
    /// Whether the calibration has been explicitly revoked.
    pub revoked: bool,
}

/// One specimen-level NIR observation; replicate records share specimen identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpectralObservation {
    /// Unique observation identity.
    pub observation_id: String,
    /// Physical item identity shared with any paired visual observation.
    pub subject_id: String,
    /// Physical specimen identity used for leakage-safe dataset splits.
    pub specimen_id: String,
    /// Replicate group identity for repeated acquisitions of the same specimen.
    pub replicate_group_id: String,
    /// Instrument identity recorded at acquisition.
    pub instrument_id: String,
    /// Calibration profile identity recorded at acquisition.
    pub calibration_profile_id: String,
    /// Acquisition time in nanoseconds from the declared epoch.
    pub acquired_at_ns: u64,
    /// Lowest wavelength represented in the observation, in nanometres.
    pub wavelength_start_nm: u32,
    /// Highest wavelength represented in the observation, in nanometres.
    pub wavelength_end_nm: u32,
    /// Instrument-reported acquisition quality.
    pub quality: SpectralQuality,
    /// Optional composition estimate; absence remains unknown.
    pub composition: Option<CompositionEstimate>,
    /// Resolvable source evidence/artifact reference.
    pub evidence_ref: String,
}

/// Stable reasons for admitting, abstaining from or rejecting NIR evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectralAssessmentReason {
    /// Required specimen, observation or subject identity is absent.
    MissingIdentity,
    /// No source evidence reference was supplied.
    MissingEvidenceReference,
    /// No instrument profile was supplied.
    MissingInstrumentProfile,
    /// No matching calibration record was supplied.
    MissingCalibration,
    /// Instrument identity differs between observation and profile.
    InstrumentProfileMismatch,
    /// Calibration identity differs from the declared instrument profile.
    CalibrationProfileMismatch,
    /// Calibration was explicitly revoked.
    CalibrationRevoked,
    /// Acquisition time falls outside calibration validity.
    CalibrationNotValidAtAcquisition,
    /// Wavelength interval is reversed or empty.
    InvalidWavelengthRange,
    /// Wavelength interval is outside the qualified instrument profile.
    WavelengthOutsideQualifiedProfile,
    /// Instrument reported insufficient signal.
    LowSignal,
    /// Instrument reported saturation.
    Saturated,
    /// Acquisition is outside the qualified sensor/model domain.
    OutOfDomain,
    /// Instrument/acquisition reported a fault.
    SensorFault,
    /// No composition estimate was produced.
    CompositionMissing,
    /// Composition estimate violates fraction/uncertainty semantics.
    CompositionInvalid(CompositionError),
    /// A portion of the material remains explicitly unidentified.
    UnknownFibreFraction,
}

/// Assessment of one spectral observation; no recycling or grade eligibility is inferred.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpectralAssessment {
    /// Admission disposition.
    pub disposition: ObservationDisposition,
    /// Stable explanation when the disposition is not fully admitted.
    pub reason: Option<SpectralAssessmentReason>,
    /// Validated/partial composition estimate, retained even when it contains unknown fraction.
    pub composition: Option<CompositionEstimate>,
}

/// Validate an NIR observation against its exact instrument and calibration profiles.
///
/// This function does not perform spectral classification. It only admits a
/// supplied estimate when its evidence, acquisition and composition contracts
/// are coherent. Unknown or missing evidence produces abstention.
pub fn assess_spectral_observation(
    observation: &SpectralObservation,
    profile: Option<&NirInstrumentProfile>,
    calibration: Option<&NirCalibration>,
) -> SpectralAssessment {
    let unknown = |reason| SpectralAssessment {
        disposition: ObservationDisposition::Unknown,
        reason: Some(reason),
        composition: None,
    };
    let rejected = |reason| SpectralAssessment {
        disposition: ObservationDisposition::Rejected,
        reason: Some(reason),
        composition: None,
    };

    if observation.observation_id.trim().is_empty()
        || observation.subject_id.trim().is_empty()
        || observation.specimen_id.trim().is_empty()
        || observation.replicate_group_id.trim().is_empty()
        || observation.instrument_id.trim().is_empty()
    {
        return rejected(SpectralAssessmentReason::MissingIdentity);
    }
    if observation.evidence_ref.trim().is_empty() {
        return rejected(SpectralAssessmentReason::MissingEvidenceReference);
    }

    let Some(profile) = profile else {
        return unknown(SpectralAssessmentReason::MissingInstrumentProfile);
    };
    if profile.instrument_id.trim().is_empty()
        || profile.calibration_profile_id.trim().is_empty()
        || profile.minimum_wavelength_nm >= profile.maximum_wavelength_nm
    {
        return rejected(SpectralAssessmentReason::InstrumentProfileMismatch);
    }
    if observation.instrument_id != profile.instrument_id {
        return rejected(SpectralAssessmentReason::InstrumentProfileMismatch);
    }

    let Some(calibration) = calibration else {
        return unknown(SpectralAssessmentReason::MissingCalibration);
    };
    if calibration.instrument_id != profile.instrument_id
        || calibration.calibration_profile_id != profile.calibration_profile_id
        || observation.calibration_profile_id != profile.calibration_profile_id
    {
        return rejected(SpectralAssessmentReason::CalibrationProfileMismatch);
    }
    if calibration.revoked {
        return rejected(SpectralAssessmentReason::CalibrationRevoked);
    }
    if observation.acquired_at_ns < calibration.valid_from_ns
        || calibration
            .valid_until_ns
            .is_some_and(|until| observation.acquired_at_ns > until)
    {
        return unknown(SpectralAssessmentReason::CalibrationNotValidAtAcquisition);
    }

    if observation.wavelength_start_nm >= observation.wavelength_end_nm {
        return rejected(SpectralAssessmentReason::InvalidWavelengthRange);
    }
    if observation.wavelength_start_nm < profile.minimum_wavelength_nm
        || observation.wavelength_end_nm > profile.maximum_wavelength_nm
    {
        return unknown(SpectralAssessmentReason::WavelengthOutsideQualifiedProfile);
    }

    let quality_reason = match observation.quality {
        SpectralQuality::Valid => None,
        SpectralQuality::LowSignal => Some(SpectralAssessmentReason::LowSignal),
        SpectralQuality::Saturated => Some(SpectralAssessmentReason::Saturated),
        SpectralQuality::OutOfDomain => Some(SpectralAssessmentReason::OutOfDomain),
        SpectralQuality::SensorFault => Some(SpectralAssessmentReason::SensorFault),
    };
    if let Some(reason) = quality_reason {
        return unknown(reason);
    }

    let Some(composition) = observation.composition.clone() else {
        return unknown(SpectralAssessmentReason::CompositionMissing);
    };
    if let Err(error) = composition.validate() {
        return rejected(SpectralAssessmentReason::CompositionInvalid(error));
    }
    if composition.contains_unknown_fraction() {
        return SpectralAssessment {
            disposition: ObservationDisposition::Unknown,
            reason: Some(SpectralAssessmentReason::UnknownFibreFraction),
            composition: Some(composition),
        };
    }

    SpectralAssessment {
        disposition: ObservationDisposition::Admitted,
        reason: None,
        composition: Some(composition),
    }
}

/// Visual garment families; this vocabulary intentionally contains no fibre identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualGarmentCategory {
    /// Shirt or blouse.
    Shirt,
    /// Jacket or coat.
    Jacket,
    /// Trousers or shorts.
    Trousers,
    /// Dress or skirt.
    Dress,
    /// Knitwear or sweater.
    Knitwear,
    /// Unrecognized or out-of-taxonomy garment.
    Unknown,
}

/// Presentation state for a visually observed garment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualPresentationState {
    /// Laid out and substantially visible.
    Flat,
    /// Wrinkled or crumpled.
    Wrinkled,
    /// Folded or bundled.
    Folded,
    /// Multiple items overlap.
    Overlapped,
    /// Partially occluded.
    Occluded,
    /// Wet or visibly contaminated.
    WetOrContaminated,
    /// Presentation state could not be determined.
    Unknown,
}

/// Visual observation for garment family/presentation only, never fibre composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualObservation {
    /// Unique visual observation identity.
    pub observation_id: String,
    /// Physical item identity for optional pairing with NIR observations.
    pub subject_id: String,
    /// Garment identity used for leakage-safe dataset grouping.
    pub garment_id: String,
    /// Visual model/version identity.
    pub model_ref: String,
    /// Predicted garment family, not a fibre claim.
    pub category: VisualGarmentCategory,
    /// Observed presentation state.
    pub presentation: VisualPresentationState,
    /// Model confidence in parts per million.
    pub confidence_ppm: u32,
    /// Resolvable source image/annotation reference.
    pub evidence_ref: String,
}

/// Reasons a visual observation is not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualAssessmentReason {
    /// Required observation/subject/garment/model identity is absent.
    MissingIdentity,
    /// Source image or annotation reference is absent.
    MissingEvidenceReference,
    /// Confidence exceeds the defined parts-per-million range.
    ConfidenceOutOfRange,
    /// Confidence is below the caller-declared policy threshold.
    BelowDeclaredConfidenceThreshold,
    /// Category is explicitly unknown.
    CategoryUnknown,
}

/// Assessment of visual garment-family evidence only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualAssessment {
    /// Admission disposition.
    pub disposition: ObservationDisposition,
    /// Stable reason when not admitted.
    pub reason: Option<VisualAssessmentReason>,
    /// Predicted visual category, if one was supplied.
    pub category: Option<VisualGarmentCategory>,
    /// Presentation state, if one was supplied.
    pub presentation: Option<VisualPresentationState>,
}

/// Validate a visual garment observation against a caller-declared confidence floor.
pub fn assess_visual_observation(
    observation: &VisualObservation,
    minimum_confidence_ppm: u32,
) -> VisualAssessment {
    let unknown = |reason| VisualAssessment {
        disposition: ObservationDisposition::Unknown,
        reason: Some(reason),
        category: Some(observation.category),
        presentation: Some(observation.presentation),
    };
    let rejected = |reason| VisualAssessment {
        disposition: ObservationDisposition::Rejected,
        reason: Some(reason),
        category: Some(observation.category),
        presentation: Some(observation.presentation),
    };

    if observation.observation_id.trim().is_empty()
        || observation.subject_id.trim().is_empty()
        || observation.garment_id.trim().is_empty()
        || observation.model_ref.trim().is_empty()
    {
        return rejected(VisualAssessmentReason::MissingIdentity);
    }
    if observation.evidence_ref.trim().is_empty() {
        return rejected(VisualAssessmentReason::MissingEvidenceReference);
    }
    if observation.confidence_ppm > FRACTION_SCALE_PPM as u32
        || minimum_confidence_ppm > FRACTION_SCALE_PPM as u32
    {
        return rejected(VisualAssessmentReason::ConfidenceOutOfRange);
    }
    if observation.category == VisualGarmentCategory::Unknown
        || observation.presentation == VisualPresentationState::Unknown
    {
        return unknown(VisualAssessmentReason::CategoryUnknown);
    }
    if observation.confidence_ppm < minimum_confidence_ppm {
        return unknown(VisualAssessmentReason::BelowDeclaredConfidenceThreshold);
    }

    VisualAssessment {
        disposition: ObservationDisposition::Admitted,
        reason: None,
        category: Some(observation.category),
        presentation: Some(observation.presentation),
    }
}

/// Whether separately observed modalities refer to the same physical item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairBinding {
    /// Visual and spectral records name the same subject.
    Bound,
    /// The two records identify different subjects.
    SubjectMismatch,
    /// A visual record is absent.
    MissingVisual,
    /// A spectral record is absent.
    MissingSpectral,
    /// Both modality records are absent.
    NoObservations,
}

/// Separate assessments for paired visual and spectral observations.
///
/// This type deliberately exposes two independent results. It contains no
/// function that derives fibre composition from a visual garment category or
/// derives recovery eligibility from either sensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairedTextileAssessment {
    /// Identity binding between the two observation sources.
    pub pair_binding: PairBinding,
    /// Visual garment-family assessment, when supplied.
    pub visual: Option<VisualAssessment>,
    /// NIR fibre-composition assessment, when supplied.
    pub spectral: Option<SpectralAssessment>,
}

/// Assess optional modalities while preserving their independent semantics.
pub fn assess_paired_textile_observations(
    visual: Option<&VisualObservation>,
    spectral: Option<&SpectralObservation>,
    profile: Option<&NirInstrumentProfile>,
    calibration: Option<&NirCalibration>,
    minimum_visual_confidence_ppm: u32,
) -> PairedTextileAssessment {
    let pair_binding = match (visual, spectral) {
        (Some(visual), Some(spectral)) if visual.subject_id == spectral.subject_id => {
            PairBinding::Bound
        }
        (Some(_), Some(_)) => PairBinding::SubjectMismatch,
        (Some(_), None) => PairBinding::MissingSpectral,
        (None, Some(_)) => PairBinding::MissingVisual,
        (None, None) => PairBinding::NoObservations,
    };

    PairedTextileAssessment {
        pair_binding,
        visual: visual.map(|value| assess_visual_observation(value, minimum_visual_confidence_ppm)),
        spectral: spectral.map(|value| assess_spectral_observation(value, profile, calibration)),
    }
}

/// Kind of identity that must remain in exactly one dataset partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SplitGroupKind {
    /// TexSortML-style frames grouped by physical garment identity.
    GarmentIdentity,
    /// NIR-SORT-style replicate spectra grouped by physical specimen identity.
    PhysicalSpecimen,
}

/// Dataset partition for a row or frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DataPartition {
    /// Model fitting/training data.
    Train,
    /// Hyperparameter/model selection data.
    Validation,
    /// Held-out evaluation data.
    Test,
    /// One fold in a grouped cross-validation protocol.
    Fold(u8),
}

/// One row assignment used to audit leakage-safe grouped dataset splits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitAssignment {
    /// Stable row, frame or replicate identity.
    pub row_id: String,
    /// Stable garment or physical-specimen group identity.
    pub group_id: String,
    /// Group identity semantics.
    pub group_kind: SplitGroupKind,
    /// Assigned data partition.
    pub partition: DataPartition,
}

/// Split-validation failures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitError {
    /// A row or group identity is missing.
    MissingIdentity,
    /// The same row identity occurs more than once for one grouping kind.
    DuplicateRow {
        /// Duplicate row identity.
        row_id: String,
        /// Identity grouping semantics.
        group_kind: SplitGroupKind,
    },
    /// One physical group appears in more than one partition.
    GroupLeakage {
        /// Leaking group identity.
        group_id: String,
        /// Identity grouping semantics.
        group_kind: SplitGroupKind,
    },
}

/// Reject row duplication and garment/specimen leakage across partitions.
pub fn validate_grouped_splits(rows: &[SplitAssignment]) -> Result<(), SplitError> {
    let mut row_ids = BTreeSet::new();
    let mut group_partitions = BTreeMap::new();

    for row in rows {
        if row.row_id.trim().is_empty() || row.group_id.trim().is_empty() {
            return Err(SplitError::MissingIdentity);
        }
        if !row_ids.insert((row.group_kind, row.row_id.clone())) {
            return Err(SplitError::DuplicateRow {
                row_id: row.row_id.clone(),
                group_kind: row.group_kind,
            });
        }

        let key = (row.group_kind, row.group_id.clone());
        if let Some(existing) = group_partitions.get(&key) {
            if *existing != row.partition {
                return Err(SplitError::GroupLeakage {
                    group_id: row.group_id.clone(),
                    group_kind: row.group_kind,
                });
            }
        } else {
            group_partitions.insert(key, row.partition);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_profile() -> NirInstrumentProfile {
        NirInstrumentProfile {
            instrument_id: "NIR-BENCHTOP-01".into(),
            calibration_profile_id: "CAL-NIR-01".into(),
            minimum_wavelength_nm: 833,
            maximum_wavelength_nm: 2500,
        }
    }

    fn valid_calibration() -> NirCalibration {
        NirCalibration {
            calibration_profile_id: "CAL-NIR-01".into(),
            instrument_id: "NIR-BENCHTOP-01".into(),
            valid_from_ns: 100,
            valid_until_ns: Some(1_000),
            revoked: false,
        }
    }

    fn valid_composition() -> CompositionEstimate {
        CompositionEstimate {
            method_ref: "NIR-METHOD-01".into(),
            components: vec![
                FibreFraction {
                    kind: FibreKind::Cotton,
                    fraction_ppm: 700_000,
                    lower_bound_ppm: 650_000,
                    upper_bound_ppm: 750_000,
                },
                FibreFraction {
                    kind: FibreKind::Polyester,
                    fraction_ppm: 300_000,
                    lower_bound_ppm: 250_000,
                    upper_bound_ppm: 350_000,
                },
            ],
        }
    }

    fn valid_spectral() -> SpectralObservation {
        SpectralObservation {
            observation_id: "OBS-NIR-01".into(),
            subject_id: "ITEM-01".into(),
            specimen_id: "SPECIMEN-01".into(),
            replicate_group_id: "REPLICATE-GROUP-01".into(),
            instrument_id: "NIR-BENCHTOP-01".into(),
            calibration_profile_id: "CAL-NIR-01".into(),
            acquired_at_ns: 500,
            wavelength_start_nm: 850,
            wavelength_end_nm: 2400,
            quality: SpectralQuality::Valid,
            composition: Some(valid_composition()),
            evidence_ref: "EVIDENCE-NIR-01".into(),
        }
    }

    fn valid_visual() -> VisualObservation {
        VisualObservation {
            observation_id: "OBS-VIS-01".into(),
            subject_id: "ITEM-01".into(),
            garment_id: "GARMENT-01".into(),
            model_ref: "TEXSORTML-BASELINE-01".into(),
            category: VisualGarmentCategory::Jacket,
            presentation: VisualPresentationState::Wrinkled,
            confidence_ppm: 920_000,
            evidence_ref: "EVIDENCE-VIS-01".into(),
        }
    }

    #[test]
    fn current_calibrated_spectrum_with_closed_composition_is_admitted() {
        let result = assess_spectral_observation(
            &valid_spectral(),
            Some(&valid_profile()),
            Some(&valid_calibration()),
        );
        assert_eq!(result.disposition, ObservationDisposition::Admitted);
        assert_eq!(result.reason, None);
        assert!(result.composition.is_some());
    }

    #[test]
    fn expired_calibration_at_acquisition_time_abstains() {
        let mut observation = valid_spectral();
        observation.acquired_at_ns = 1_001;
        let result = assess_spectral_observation(
            &observation,
            Some(&valid_profile()),
            Some(&valid_calibration()),
        );
        assert_eq!(result.disposition, ObservationDisposition::Unknown);
        assert_eq!(
            result.reason,
            Some(SpectralAssessmentReason::CalibrationNotValidAtAcquisition)
        );
    }

    #[test]
    fn revoked_calibration_is_rejected() {
        let mut calibration = valid_calibration();
        calibration.revoked = true;
        let result = assess_spectral_observation(
            &valid_spectral(),
            Some(&valid_profile()),
            Some(&calibration),
        );
        assert_eq!(result.disposition, ObservationDisposition::Rejected);
        assert_eq!(
            result.reason,
            Some(SpectralAssessmentReason::CalibrationRevoked)
        );
    }

    #[test]
    fn mismatched_instrument_is_rejected() {
        let mut observation = valid_spectral();
        observation.instrument_id = "NIR-HANDHELD-02".into();
        let result = assess_spectral_observation(
            &observation,
            Some(&valid_profile()),
            Some(&valid_calibration()),
        );
        assert_eq!(result.disposition, ObservationDisposition::Rejected);
        assert_eq!(
            result.reason,
            Some(SpectralAssessmentReason::InstrumentProfileMismatch)
        );
    }

    #[test]
    fn wavelength_range_outside_profile_abstains() {
        let mut observation = valid_spectral();
        observation.wavelength_start_nm = 700;
        let result = assess_spectral_observation(
            &observation,
            Some(&valid_profile()),
            Some(&valid_calibration()),
        );
        assert_eq!(result.disposition, ObservationDisposition::Unknown);
        assert_eq!(
            result.reason,
            Some(SpectralAssessmentReason::WavelengthOutsideQualifiedProfile)
        );
    }

    #[test]
    fn saturated_spectral_observation_abstains() {
        let mut observation = valid_spectral();
        observation.quality = SpectralQuality::Saturated;
        let result = assess_spectral_observation(
            &observation,
            Some(&valid_profile()),
            Some(&valid_calibration()),
        );
        assert_eq!(result.disposition, ObservationDisposition::Unknown);
        assert_eq!(result.reason, Some(SpectralAssessmentReason::Saturated));
    }

    #[test]
    fn missing_composition_estimate_abstains() {
        let mut observation = valid_spectral();
        observation.composition = None;
        let result = assess_spectral_observation(
            &observation,
            Some(&valid_profile()),
            Some(&valid_calibration()),
        );
        assert_eq!(result.disposition, ObservationDisposition::Unknown);
        assert_eq!(
            result.reason,
            Some(SpectralAssessmentReason::CompositionMissing)
        );
    }

    #[test]
    fn invalid_composition_total_is_rejected() {
        let mut observation = valid_spectral();
        let mut composition = valid_composition();
        composition.components[0].fraction_ppm = 600_000;
        composition.components[0].lower_bound_ppm = 550_000;
        composition.components[0].upper_bound_ppm = 650_000;
        observation.composition = Some(composition);
        let result = assess_spectral_observation(
            &observation,
            Some(&valid_profile()),
            Some(&valid_calibration()),
        );
        assert_eq!(result.disposition, ObservationDisposition::Rejected);
        assert!(matches!(
            result.reason,
            Some(SpectralAssessmentReason::CompositionInvalid(
                CompositionError::FractionTotalMismatch { .. }
            ))
        ));
    }

    #[test]
    fn unidentified_fraction_is_preserved_and_causes_abstention() {
        let mut observation = valid_spectral();
        let mut composition = valid_composition();
        composition.components.push(FibreFraction {
            kind: FibreKind::Unknown,
            fraction_ppm: 10_000,
            lower_bound_ppm: 0,
            upper_bound_ppm: 50_000,
        });
        composition.components[0].fraction_ppm = 690_000;
        composition.components[0].lower_bound_ppm = 640_000;
        composition.components[0].upper_bound_ppm = 740_000;
        observation.composition = Some(composition);
        let result = assess_spectral_observation(
            &observation,
            Some(&valid_profile()),
            Some(&valid_calibration()),
        );
        assert_eq!(result.disposition, ObservationDisposition::Unknown);
        assert_eq!(
            result.reason,
            Some(SpectralAssessmentReason::UnknownFibreFraction)
        );
        assert!(result.composition.is_some());
    }

    #[test]
    fn uncertain_unknown_fraction_abstains_even_when_point_estimate_is_zero() {
        let mut observation = valid_spectral();
        let mut composition = valid_composition();
        composition.components.push(FibreFraction {
            kind: FibreKind::Unknown,
            fraction_ppm: 0,
            lower_bound_ppm: 0,
            upper_bound_ppm: 50_000,
        });
        observation.composition = Some(composition);
        let result = assess_spectral_observation(
            &observation,
            Some(&valid_profile()),
            Some(&valid_calibration()),
        );
        assert_eq!(result.disposition, ObservationDisposition::Unknown);
        assert_eq!(
            result.reason,
            Some(SpectralAssessmentReason::UnknownFibreFraction)
        );
    }

    #[test]
    fn visual_only_observation_does_not_create_fibre_composition_evidence() {
        let visual = valid_visual();
        let result = assess_paired_textile_observations(Some(&visual), None, None, None, 700_000);
        assert_eq!(result.pair_binding, PairBinding::MissingSpectral);
        assert_eq!(
            result.visual.as_ref().map(|value| value.disposition),
            Some(ObservationDisposition::Admitted)
        );
        assert!(result.spectral.is_none());
    }

    #[test]
    fn visual_and_spectral_observations_must_bind_to_the_same_subject() {
        let visual = valid_visual();
        let mut spectral = valid_spectral();
        spectral.subject_id = "ITEM-OTHER".into();
        let result = assess_paired_textile_observations(
            Some(&visual),
            Some(&spectral),
            Some(&valid_profile()),
            Some(&valid_calibration()),
            700_000,
        );
        assert_eq!(result.pair_binding, PairBinding::SubjectMismatch);
        assert_eq!(
            result.spectral.as_ref().map(|value| value.disposition),
            Some(ObservationDisposition::Admitted)
        );
    }

    #[test]
    fn visual_unknown_category_or_presentation_abstains() {
        let mut visual = valid_visual();
        visual.category = VisualGarmentCategory::Unknown;
        let result = assess_visual_observation(&visual, 700_000);
        assert_eq!(result.disposition, ObservationDisposition::Unknown);
        assert_eq!(result.reason, Some(VisualAssessmentReason::CategoryUnknown));

        let mut visual = valid_visual();
        visual.presentation = VisualPresentationState::Unknown;
        let result = assess_visual_observation(&visual, 700_000);
        assert_eq!(result.disposition, ObservationDisposition::Unknown);
        assert_eq!(result.reason, Some(VisualAssessmentReason::CategoryUnknown));
    }

    #[test]
    fn visual_observation_below_declared_confidence_floor_abstains() {
        let mut visual = valid_visual();
        visual.confidence_ppm = 500_000;
        let result = assess_visual_observation(&visual, 700_000);
        assert_eq!(result.disposition, ObservationDisposition::Unknown);
        assert_eq!(
            result.reason,
            Some(VisualAssessmentReason::BelowDeclaredConfidenceThreshold)
        );
    }

    #[test]
    fn garment_frames_from_one_identity_cannot_cross_partitions() {
        let rows = vec![
            SplitAssignment {
                row_id: "FRAME-1".into(),
                group_id: "GARMENT-1".into(),
                group_kind: SplitGroupKind::GarmentIdentity,
                partition: DataPartition::Train,
            },
            SplitAssignment {
                row_id: "FRAME-2".into(),
                group_id: "GARMENT-1".into(),
                group_kind: SplitGroupKind::GarmentIdentity,
                partition: DataPartition::Test,
            },
        ];
        assert!(matches!(
            validate_grouped_splits(&rows),
            Err(SplitError::GroupLeakage {
                group_kind: SplitGroupKind::GarmentIdentity,
                ..
            })
        ));
    }

    #[test]
    fn nir_replicates_from_one_specimen_cannot_cross_partitions() {
        let rows = vec![
            SplitAssignment {
                row_id: "SPECTRUM-1".into(),
                group_id: "SPECIMEN-1".into(),
                group_kind: SplitGroupKind::PhysicalSpecimen,
                partition: DataPartition::Train,
            },
            SplitAssignment {
                row_id: "SPECTRUM-2".into(),
                group_id: "SPECIMEN-1".into(),
                group_kind: SplitGroupKind::PhysicalSpecimen,
                partition: DataPartition::Validation,
            },
        ];
        assert!(matches!(
            validate_grouped_splits(&rows),
            Err(SplitError::GroupLeakage {
                group_kind: SplitGroupKind::PhysicalSpecimen,
                ..
            })
        ));
    }

    #[test]
    fn grouped_split_accepts_multiple_frames_or_replicates_in_one_partition() {
        let rows = vec![
            SplitAssignment {
                row_id: "SPECTRUM-1".into(),
                group_id: "SPECIMEN-1".into(),
                group_kind: SplitGroupKind::PhysicalSpecimen,
                partition: DataPartition::Fold(2),
            },
            SplitAssignment {
                row_id: "SPECTRUM-2".into(),
                group_id: "SPECIMEN-1".into(),
                group_kind: SplitGroupKind::PhysicalSpecimen,
                partition: DataPartition::Fold(2),
            },
        ];
        assert_eq!(validate_grouped_splits(&rows), Ok(()));
    }
    fn exercise_corpus_case(case_id: &str) -> (&'static str, Option<&'static str>) {
        match case_id {
            "SORT-001-CURRENT-NIR" => {
                current_calibrated_spectrum_with_closed_composition_is_admitted();
                ("admitted", None)
            }
            "SORT-002-EXPIRED-CALIBRATION" => {
                expired_calibration_at_acquisition_time_abstains();
                ("unknown", Some("CalibrationNotValidAtAcquisition"))
            }
            "SORT-003-REVOKED-CALIBRATION" => {
                revoked_calibration_is_rejected();
                ("rejected", Some("CalibrationRevoked"))
            }
            "SORT-004-SATURATED-SIGNAL" => {
                saturated_spectral_observation_abstains();
                ("unknown", Some("Saturated"))
            }
            "SORT-005-OUTSIDE-WAVELENGTH-PROFILE" => {
                wavelength_range_outside_profile_abstains();
                ("unknown", Some("WavelengthOutsideQualifiedProfile"))
            }
            "SORT-006-UNKNOWN-FIBRE-FRACTION" => {
                unidentified_fraction_is_preserved_and_causes_abstention();
                ("unknown", Some("UnknownFibreFraction"))
            }
            "SORT-007-VISUAL-ONLY" => {
                visual_only_observation_does_not_create_fibre_composition_evidence();
                ("unknown", Some("MissingSpectral"))
            }
            "SORT-008-SUBJECT-MISMATCH" => {
                visual_and_spectral_observations_must_bind_to_the_same_subject();
                ("rejected", Some("SubjectMismatch"))
            }
            "SORT-009-VISUAL-LOW-CONFIDENCE" => {
                visual_observation_below_declared_confidence_floor_abstains();
                ("unknown", Some("BelowDeclaredConfidenceThreshold"))
            }
            "SORT-010-VISUAL-UNKNOWN-CATEGORY" => {
                visual_unknown_category_or_presentation_abstains();
                ("unknown", Some("CategoryUnknown"))
            }
            "SORT-011-GARMENT-FRAME-LEAKAGE" => {
                garment_frames_from_one_identity_cannot_cross_partitions();
                ("group_leakage", Some("GroupLeakage:GarmentIdentity"))
            }
            "SORT-012-NIR-REPLICATE-LEAKAGE" => {
                nir_replicates_from_one_specimen_cannot_cross_partitions();
                ("group_leakage", Some("GroupLeakage:PhysicalSpecimen"))
            }
            "SORT-013-COMPOSITION-FRACTION-NOT-CLOSED" => {
                invalid_composition_total_is_rejected();
                ("rejected", Some("CompositionInvalid:FractionTotalMismatch"))
            }
            "SORT-014-INSTRUMENT-MISMATCH" => {
                mismatched_instrument_is_rejected();
                ("rejected", Some("InstrumentProfileMismatch"))
            }
            "SORT-015-GROUPED-SPLIT-VALID" => {
                grouped_split_accepts_multiple_frames_or_replicates_in_one_partition();
                ("valid_split", None)
            }
            other => panic!("unknown textile sensing corpus case: {other}"),
        }
    }

    #[test]
    fn synthetic_corpus_drives_all_declared_contract_regression_cases() {
        const CORPUS: &str = include_str!(
            "../../../../docs/research/fixtures/textile_sort_fusion_v0_1/sensor_cases.example.json"
        );
        const SCHEMA: &str = include_str!(
            "../../../../docs/research/fixtures/textile_sort_fusion_v0_1/sensor_cases.schema.json"
        );

        let corpus: serde_json::Value =
            serde_json::from_str(CORPUS).expect("synthetic textile corpus must parse as JSON");
        let schema: serde_json::Value =
            serde_json::from_str(SCHEMA).expect("textile corpus schema must parse as JSON");

        assert_eq!(corpus["schema_version"], "0.1.0");
        assert_eq!(corpus["corpus_id"], "TEXTILE-SORT-FUSION-CASES-001");
        assert_eq!(corpus["fixture_class"], "synthetic");
        assert_eq!(schema["properties"]["fixture_class"]["const"], "synthetic");
        assert_eq!(schema["properties"]["cases"]["minItems"], 10);

        let cases = corpus["cases"]
            .as_array()
            .expect("corpus cases must be an array");
        assert_eq!(
            cases.len(),
            15,
            "corpus case count changed; review mappings"
        );

        let mut seen = BTreeSet::new();
        for case in cases {
            let case_id = case["case_id"]
                .as_str()
                .expect("every corpus case needs a string case_id");
            assert!(seen.insert(case_id), "duplicate corpus case {case_id}");
            assert!(
                case["mutation"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()),
                "case {case_id} must declare its mutation"
            );
            assert!(
                case["invariant"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()),
                "case {case_id} must declare its invariant"
            );
            let expected = case["expected"]
                .as_object()
                .expect("every corpus case needs expected outcome");
            assert!(
                expected
                    .get("disposition")
                    .and_then(|v| v.as_str())
                    .is_some()
            );
            assert!(expected.contains_key("reason"));
            let (disposition, reason) = exercise_corpus_case(case_id);
            assert_eq!(
                expected["disposition"].as_str(),
                Some(disposition),
                "case {case_id} disposition changed"
            );
            assert_eq!(
                expected["reason"].as_str(),
                reason,
                "case {case_id} reason changed"
            );
        }

        assert_eq!(seen.len(), 15);
        assert!(
            corpus["claim_ceiling"]
                .as_str()
                .is_some_and(|value| value.contains("Synthetic"))
        );
    }
}
