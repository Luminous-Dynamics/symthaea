// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Method-aware, uncertainty-aware fibre-length metrology.
//!
//! The contract does not assume that two methods are interchangeable because
//! both report micrometres. Scope, sampling, weighting, calibration, tails,
//! uncertainty and output-lot lineage remain explicit.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const FRACTION_SCALE_PPM: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialClass {
    Cotton,
    Polyester,
    RegeneratedCellulose,
    Wool,
    Polyamide,
    Elastane,
    Mixed,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessState {
    PreProcessReferenceFibre,
    LooseVirginFibre,
    LooseRecoveredFibre,
    RecoveredAfterOpening,
    YarnExtractedFibre,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightingBasis {
    MassWeighted,
    NumberWeighted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodDefinition {
    pub method_id: String,
    pub version: String,
    pub citation: String,
    pub allowed_materials: Vec<MaterialClass>,
    pub allowed_process_states: Vec<ProcessState>,
    pub supported_weighting_bases: Vec<WeightingBasis>,
    pub minimum_independent_subsamples: u32,
    pub minimum_acquisitions: u32,
    pub resolution_um: u32,
    pub calibration_required: bool,
    /// States not explicitly covered by the formal method scope require a
    /// local, source-bound validation record before the method can be used.
    pub local_validation_required_states: Vec<ProcessState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRef {
    pub evidence_id: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationRecord {
    pub calibration_id: String,
    pub sha256: String,
    pub instrument_id: String,
    pub method_id: String,
    pub method_version: String,
    pub valid_from_ns: u64,
    pub valid_until_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplePlan {
    /// Independent physical subsamples, not repeat scans of one subsample.
    pub independent_subsample_ids: Vec<String>,
    /// Acquisition IDs; multiple acquisitions do not increase independent n.
    pub acquisition_ids: Vec<String>,
    pub sampling_frame: String,
    pub conditioning_ref: String,
    pub sampled_mass_mg: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FibreLengthBin {
    /// Inclusive lower bound in micrometres.
    pub lower_bound_um: u32,
    /// Exclusive upper bound in micrometres.
    pub upper_bound_um: u32,
    /// Fraction on the explicitly declared weighting basis, in ppm.
    pub fraction_ppm: u32,
    /// Expanded absolute uncertainty for the bin fraction, in ppm.
    pub expanded_uncertainty_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FibreTail {
    /// Underflow is all lengths below this boundary; overflow is all lengths
    /// at or above this boundary.
    pub boundary_um: u32,
    pub fraction_ppm: u32,
    pub expanded_uncertainty_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalScopeValidation {
    pub method_id: String,
    pub method_version: String,
    pub material_class: MaterialClass,
    pub process_state: ProcessState,
    pub evidence: EvidenceRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FibreLengthDistribution {
    pub measurement_id: String,
    pub measured_lot_id: String,
    pub parent_lot_id: String,
    pub physical_subject_id: String,
    /// Present for output distributions; must identify the process that
    /// produced this exact lot.
    pub source_process_trial_id: Option<String>,
    pub method_id: String,
    pub method_version: String,
    pub material_class: MaterialClass,
    pub process_state: ProcessState,
    pub weighting_basis: WeightingBasis,
    pub observed_at_ns: u64,
    pub instrument_id: String,
    pub calibration: Option<CalibrationRecord>,
    pub local_scope_validation: Option<LocalScopeValidation>,
    pub sample_plan: SamplePlan,
    pub underflow: FibreTail,
    pub bins: Vec<FibreLengthBin>,
    pub overflow: FibreTail,
    pub unclassified_fraction_ppm: u32,
    /// Combined expanded uncertainty, in ppm, for the reported distribution.
    pub total_expanded_uncertainty_ppm: u32,
    pub measurement_evidence: EvidenceRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiverCriterion {
    pub specification_id: String,
    pub specification_version: String,
    pub minimum_fibre_length_um: u32,
    pub minimum_qualifying_fraction_ppm: u32,
    pub maximum_expanded_uncertainty_ppm: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Qualified,
    InsufficientEvidence,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Finding {
    InvalidMethodDefinition,
    MethodIdentityMismatch,
    MethodScopeExceeded,
    LocalScopeValidationMissing,
    LocalScopeValidationMismatch,
    UnsupportedWeightingBasis,
    MissingCalibration,
    InvalidCalibration,
    CalibrationMethodMismatch,
    CalibrationInstrumentMismatch,
    MeasurementEvidenceInvalid,
    CalibrationEvidenceInvalid,
    SamplePlanInvalid,
    InsufficientIndependentSubsamples,
    InsufficientAcquisitions,
    InvalidReceiverCriterion,
    EmptyDistribution,
    InvalidBinBounds,
    BinGap,
    BinOverlap,
    TailBoundaryMismatch,
    FractionOutsideScale,
    FractionTotalNotClosed,
    ExpandedUncertaintyAbovePolicy,
    ReceiverLengthFractionBelowPolicy,
    ProcessLotIdentityMismatch,
    ProcessTrialIdentityMismatch,
    MethodNotDirectlyComparable,
    BridgeIdentityMismatch,
    BridgeInsufficientPairs,
    BridgeDuplicateSpecimen,
    BridgeEvidenceInvalid,
    BridgeFractionOutsideScale,
    BridgeEquivalenceMarginExceeded,
    InvalidProcessMasses,
    InputDistributionInsufficient,
    OutputDistributionInsufficient,
    MaterialClassMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionAssessment {
    pub decision: Decision,
    pub findings: Vec<Finding>,
    /// Conservative lower bound: bins wholly above the threshold, less
    /// expanded uncertainty. Straddling bins receive no qualifying credit.
    pub qualifying_fraction_lower_ppm: u32,
    /// Fraction in bins/tails whose interval crosses the receiver threshold.
    pub threshold_ambiguous_fraction_ppm: u32,
    pub total_expanded_uncertainty_ppm: u32,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn valid_evidence(evidence: &EvidenceRef) -> bool {
    nonempty(&evidence.evidence_id) && valid_sha256(&evidence.sha256)
}

fn push_unique(findings: &mut Vec<Finding>, finding: Finding) {
    if !findings.contains(&finding) {
        findings.push(finding);
    }
}

fn validate_method_definition(method: &MethodDefinition) -> bool {
    nonempty(&method.method_id)
        && nonempty(&method.version)
        && nonempty(&method.citation)
        && !method.allowed_materials.is_empty()
        && !method.allowed_process_states.is_empty()
        && !method.supported_weighting_bases.is_empty()
        && method.minimum_independent_subsamples > 0
        && method.minimum_acquisitions > 0
        && method.resolution_um > 0
}

pub fn assess_distribution(
    distribution: &FibreLengthDistribution,
    method: &MethodDefinition,
    criterion: &ReceiverCriterion,
) -> DistributionAssessment {
    let mut findings = Vec::new();
    let mut lower = 0_u64;
    let mut ambiguous = 0_u64;
    let mut calculated_uncertainty = 0_u64;

    if !validate_method_definition(method) {
        push_unique(&mut findings, Finding::InvalidMethodDefinition);
    }
    if !nonempty(&criterion.specification_id)
        || !nonempty(&criterion.specification_version)
        || criterion.minimum_fibre_length_um == 0
        || criterion.minimum_qualifying_fraction_ppm > FRACTION_SCALE_PPM
        || criterion.maximum_expanded_uncertainty_ppm > FRACTION_SCALE_PPM
    {
        push_unique(&mut findings, Finding::InvalidReceiverCriterion);
    }
    if distribution.method_id != method.method_id
        || distribution.method_version != method.version
    {
        push_unique(&mut findings, Finding::MethodIdentityMismatch);
    }
    if !method.allowed_materials.contains(&distribution.material_class)
        || !method.allowed_process_states.contains(&distribution.process_state)
    {
        push_unique(&mut findings, Finding::MethodScopeExceeded);
    }
    if method.local_validation_required_states.contains(&distribution.process_state) {
        match distribution.local_scope_validation.as_ref() {
            None => push_unique(&mut findings, Finding::LocalScopeValidationMissing),
            Some(validation) => {
                if validation.method_id != method.method_id
                    || validation.method_version != method.version
                    || validation.material_class != distribution.material_class
                    || validation.process_state != distribution.process_state
                {
                    push_unique(&mut findings, Finding::LocalScopeValidationMismatch);
                }
                if !valid_evidence(&validation.evidence) {
                    push_unique(&mut findings, Finding::LocalScopeValidationMissing);
                }
            }
        }
    }
    if !method.supported_weighting_bases.contains(&distribution.weighting_basis)
        || distribution.weighting_basis != WeightingBasis::MassWeighted
    {
        push_unique(&mut findings, Finding::UnsupportedWeightingBasis);
    }
    if !valid_evidence(&distribution.measurement_evidence) {
        push_unique(&mut findings, Finding::MeasurementEvidenceInvalid);
    }

    match (&distribution.calibration, method.calibration_required) {
        (None, true) => push_unique(&mut findings, Finding::MissingCalibration),
        (Some(calibration), _) => {
            if !nonempty(&calibration.calibration_id)
                || !valid_sha256(&calibration.sha256)
                || calibration.valid_from_ns >= calibration.valid_until_ns
                || distribution.observed_at_ns < calibration.valid_from_ns
                || distribution.observed_at_ns > calibration.valid_until_ns
            {
                push_unique(&mut findings, Finding::InvalidCalibration);
            }
            if calibration.method_id != method.method_id
                || calibration.method_version != method.version
            {
                push_unique(&mut findings, Finding::CalibrationMethodMismatch);
            }
            if calibration.instrument_id != distribution.instrument_id {
                push_unique(&mut findings, Finding::CalibrationInstrumentMismatch);
            }
            if !valid_sha256(&calibration.sha256) {
                push_unique(&mut findings, Finding::CalibrationEvidenceInvalid);
            }
        }
        (None, false) => {}
    }
    if !nonempty(&distribution.instrument_id)
        || !nonempty(&distribution.measurement_id)
        || !nonempty(&distribution.measured_lot_id)
        || !nonempty(&distribution.parent_lot_id)
        || !nonempty(&distribution.physical_subject_id)
        || distribution.source_process_trial_id.as_deref().is_some_and(|v| !nonempty(v))
    {
        push_unique(&mut findings, Finding::MeasurementEvidenceInvalid);
    }

    let plan = &distribution.sample_plan;
    let unique_subsamples: BTreeSet<&str> = plan
        .independent_subsample_ids
        .iter()
        .map(String::as_str)
        .collect();
    let unique_acquisitions: BTreeSet<&str> = plan
        .acquisition_ids
        .iter()
        .map(String::as_str)
        .collect();
    if !nonempty(&plan.sampling_frame)
        || !nonempty(&plan.conditioning_ref)
        || plan.sampled_mass_mg == 0
        || unique_subsamples.len() != plan.independent_subsample_ids.len()
        || unique_acquisitions.len() != plan.acquisition_ids.len()
        || plan.independent_subsample_ids.iter().any(|value| !nonempty(value))
        || plan.acquisition_ids.iter().any(|value| !nonempty(value))
    {
        push_unique(&mut findings, Finding::SamplePlanInvalid);
    }
    if plan.independent_subsample_ids.len() < method.minimum_independent_subsamples as usize {
        push_unique(&mut findings, Finding::InsufficientIndependentSubsamples);
    }
    if plan.acquisition_ids.len() < method.minimum_acquisitions as usize {
        push_unique(&mut findings, Finding::InsufficientAcquisitions);
    }

    if distribution.bins.is_empty() {
        push_unique(&mut findings, Finding::EmptyDistribution);
    }
    if distribution.underflow.fraction_ppm > FRACTION_SCALE_PPM
        || distribution.overflow.fraction_ppm > FRACTION_SCALE_PPM
        || distribution.unclassified_fraction_ppm > FRACTION_SCALE_PPM
        || distribution.underflow.expanded_uncertainty_ppm > FRACTION_SCALE_PPM
        || distribution.overflow.expanded_uncertainty_ppm > FRACTION_SCALE_PPM
        || distribution.total_expanded_uncertainty_ppm > FRACTION_SCALE_PPM
    {
        push_unique(&mut findings, Finding::FractionOutsideScale);
    }

    let mut total_fraction = u64::from(distribution.underflow.fraction_ppm)
        + u64::from(distribution.overflow.fraction_ppm)
        + u64::from(distribution.unclassified_fraction_ppm);
    calculated_uncertainty += u64::from(distribution.underflow.expanded_uncertainty_ppm)
        + u64::from(distribution.overflow.expanded_uncertainty_ppm);
    ambiguous += u64::from(distribution.unclassified_fraction_ppm);

    let first_lower = distribution.bins.first().map(|bin| bin.lower_bound_um);
    let last_upper = distribution.bins.last().map(|bin| bin.upper_bound_um);
    if first_lower != Some(distribution.underflow.boundary_um)
        || last_upper != Some(distribution.overflow.boundary_um)
    {
        push_unique(&mut findings, Finding::TailBoundaryMismatch);
    }

    let threshold = criterion.minimum_fibre_length_um;
    if distribution.underflow.boundary_um > threshold {
        ambiguous += u64::from(distribution.underflow.fraction_ppm);
    }

    let mut previous_upper = None;
    for bin in &distribution.bins {
        if bin.lower_bound_um >= bin.upper_bound_um
            || bin.fraction_ppm > FRACTION_SCALE_PPM
            || bin.expanded_uncertainty_ppm > FRACTION_SCALE_PPM
        {
            push_unique(&mut findings, Finding::InvalidBinBounds);
        }
        if let Some(previous) = previous_upper {
            if bin.lower_bound_um < previous {
                push_unique(&mut findings, Finding::BinOverlap);
            } else if bin.lower_bound_um > previous {
                push_unique(&mut findings, Finding::BinGap);
            }
        }
        previous_upper = Some(bin.upper_bound_um);
        total_fraction += u64::from(bin.fraction_ppm);
        calculated_uncertainty += u64::from(bin.expanded_uncertainty_ppm);
        if bin.lower_bound_um >= threshold {
            lower += u64::from(
                bin.fraction_ppm
                    .saturating_sub(bin.expanded_uncertainty_ppm),
            );
        } else if bin.upper_bound_um > threshold {
            ambiguous += u64::from(bin.fraction_ppm);
        }
    }

    if distribution.overflow.boundary_um < threshold {
        ambiguous += u64::from(distribution.overflow.fraction_ppm);
    } else {
        lower += u64::from(
            distribution
                .overflow
                .fraction_ppm
                .saturating_sub(distribution.overflow.expanded_uncertainty_ppm),
        );
    }

    calculated_uncertainty = calculated_uncertainty
        .max(u64::from(distribution.total_expanded_uncertainty_ppm));

    if total_fraction != u64::from(FRACTION_SCALE_PPM) {
        push_unique(&mut findings, Finding::FractionTotalNotClosed);
    }
    if calculated_uncertainty > u64::from(criterion.maximum_expanded_uncertainty_ppm) {
        push_unique(&mut findings, Finding::ExpandedUncertaintyAbovePolicy);
    }
    if lower < u64::from(criterion.minimum_qualifying_fraction_ppm) {
        push_unique(&mut findings, Finding::ReceiverLengthFractionBelowPolicy);
    }

    let reject_findings = [
        Finding::InvalidMethodDefinition,
        Finding::InvalidReceiverCriterion,
        Finding::MethodIdentityMismatch,
        Finding::MethodScopeExceeded,
        Finding::UnsupportedWeightingBasis,
        Finding::MeasurementEvidenceInvalid,
        Finding::CalibrationMethodMismatch,
        Finding::CalibrationInstrumentMismatch,
        Finding::CalibrationEvidenceInvalid,
        Finding::SamplePlanInvalid,
        Finding::EmptyDistribution,
        Finding::InvalidBinBounds,
        Finding::BinGap,
        Finding::BinOverlap,
        Finding::TailBoundaryMismatch,
        Finding::FractionOutsideScale,
        Finding::FractionTotalNotClosed,
    ];
    let rejected = findings.iter().any(|item| reject_findings.contains(item));
    let insufficient = !findings.is_empty();
    let decision = if rejected {
        Decision::Rejected
    } else if insufficient {
        Decision::InsufficientEvidence
    } else {
        Decision::Qualified
    };

    DistributionAssessment {
        decision,
        findings,
        qualifying_fraction_lower_ppm: lower.min(u64::from(u32::MAX)) as u32,
        threshold_ambiguous_fraction_ppm: ambiguous.min(u64::from(u32::MAX)) as u32,
        total_expanded_uncertainty_ppm: calculated_uncertainty.min(u64::from(u32::MAX)) as u32,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodBridgePair {
    pub physical_specimen_id: String,
    pub method_a_fraction_ppm: u32,
    pub method_b_fraction_ppm: u32,
    pub method_a_measurement: EvidenceRef,
    pub method_b_measurement: EvidenceRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodBridge {
    pub bridge_id: String,
    pub method_a_id: String,
    pub method_a_version: String,
    pub method_b_id: String,
    pub method_b_version: String,
    pub weighting_basis: WeightingBasis,
    pub receiver_threshold_um: u32,
    pub minimum_paired_specimens: u32,
    pub maximum_absolute_difference_ppm: u32,
    pub paired_specimens: Vec<MethodBridgePair>,
    pub bridge_evidence: EvidenceRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeAssessment {
    pub equivalent_for_declared_metric: bool,
    pub maximum_observed_absolute_difference_ppm: u32,
    pub mean_signed_bias_ppm: i64,
    pub paired_specimen_count: u32,
    pub findings: Vec<Finding>,
}

pub fn assess_method_bridge(
    bridge: &MethodBridge,
    method_a: &MethodDefinition,
    method_b: &MethodDefinition,
) -> BridgeAssessment {
    let mut findings = Vec::new();
    let mut ids = BTreeSet::new();
    let mut maximum = 0_u32;
    let mut signed_sum = 0_i128;

    if !nonempty(&bridge.bridge_id)
        || bridge.method_a_id != method_a.method_id
        || bridge.method_a_version != method_a.version
        || bridge.method_b_id != method_b.method_id
        || bridge.method_b_version != method_b.version
        || bridge.method_a_id == bridge.method_b_id && bridge.method_a_version == bridge.method_b_version
    {
        push_unique(&mut findings, Finding::BridgeIdentityMismatch);
    }
    if bridge.weighting_basis != WeightingBasis::MassWeighted
        || bridge.receiver_threshold_um == 0
        || bridge.minimum_paired_specimens == 0
        || bridge.maximum_absolute_difference_ppm > FRACTION_SCALE_PPM
    {
        push_unique(&mut findings, Finding::BridgeIdentityMismatch);
    }
    if !valid_evidence(&bridge.bridge_evidence) {
        push_unique(&mut findings, Finding::BridgeEvidenceInvalid);
    }
    if bridge.paired_specimens.len() < bridge.minimum_paired_specimens as usize {
        push_unique(&mut findings, Finding::BridgeInsufficientPairs);
    }

    for pair in &bridge.paired_specimens {
        if !nonempty(&pair.physical_specimen_id) || !ids.insert(pair.physical_specimen_id.as_str()) {
            push_unique(&mut findings, Finding::BridgeDuplicateSpecimen);
        }
        if pair.method_a_fraction_ppm > FRACTION_SCALE_PPM
            || pair.method_b_fraction_ppm > FRACTION_SCALE_PPM
        {
            push_unique(&mut findings, Finding::BridgeFractionOutsideScale);
            continue;
        }
        if !valid_evidence(&pair.method_a_measurement)
            || !valid_evidence(&pair.method_b_measurement)
        {
            push_unique(&mut findings, Finding::BridgeEvidenceInvalid);
        }
        let difference = pair.method_a_fraction_ppm.abs_diff(pair.method_b_fraction_ppm);
        maximum = maximum.max(difference);
        signed_sum += i128::from(pair.method_a_fraction_ppm)
            - i128::from(pair.method_b_fraction_ppm);
    }

    if maximum > bridge.maximum_absolute_difference_ppm {
        push_unique(&mut findings, Finding::BridgeEquivalenceMarginExceeded);
    }
    let count = bridge.paired_specimens.len().max(1) as i128;
    let mean_bias = (signed_sum / count).clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
    BridgeAssessment {
        equivalent_for_declared_metric: findings.is_empty(),
        maximum_observed_absolute_difference_ppm: maximum,
        mean_signed_bias_ppm: mean_bias,
        paired_specimen_count: bridge.paired_specimens.len().min(u32::MAX as usize) as u32,
        findings,
    }
}

pub fn methods_directly_comparable(
    method_a: &MethodDefinition,
    method_b: &MethodDefinition,
    basis_a: WeightingBasis,
    basis_b: WeightingBasis,
) -> bool {
    method_a.method_id == method_b.method_id
        && method_a.version == method_b.version
        && basis_a == basis_b
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessComparison {
    pub incoming_lot_id: String,
    pub output_lot_id: String,
    pub process_trial_evidence_id: String,
    pub incoming_mass_mg: u64,
    pub output_mass_mg: u64,
    pub incoming_distribution: FibreLengthDistribution,
    pub output_distribution: FibreLengthDistribution,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessComparisonAssessment {
    pub decision: Decision,
    pub findings: Vec<Finding>,
    pub incoming_qualifying_fraction_ppm: u32,
    pub output_qualifying_fraction_ppm: u32,
    /// Conservative fibre mass meeting the declared length threshold, as a
    /// fraction of incoming lot mass. This is not receiver acceptance.
    pub length_qualified_output_per_input_ppm: u32,
    pub qualifying_fraction_change_ppm: i64,
}

pub fn assess_process_comparison(
    comparison: &ProcessComparison,
    method: &MethodDefinition,
    criterion: &ReceiverCriterion,
) -> ProcessComparisonAssessment {
    let mut findings = Vec::new();
    if !nonempty(&comparison.incoming_lot_id)
        || !nonempty(&comparison.output_lot_id)
        || !nonempty(&comparison.process_trial_evidence_id)
        || comparison.incoming_mass_mg == 0
        || comparison.output_mass_mg == 0
        || comparison.output_mass_mg > comparison.incoming_mass_mg
    {
        push_unique(&mut findings, Finding::InvalidProcessMasses);
    }
    let input = &comparison.incoming_distribution;
    let output = &comparison.output_distribution;
    if input.measured_lot_id != comparison.incoming_lot_id
        || output.measured_lot_id != comparison.output_lot_id
        || output.parent_lot_id != comparison.incoming_lot_id
    {
        push_unique(&mut findings, Finding::ProcessLotIdentityMismatch);
    }
    if output.source_process_trial_id.as_deref()
        != Some(comparison.process_trial_evidence_id.as_str())
    {
        push_unique(&mut findings, Finding::ProcessTrialIdentityMismatch);
    }
    if input.method_id != output.method_id
        || input.method_version != output.method_version
        || input.weighting_basis != output.weighting_basis
        || input.material_class != output.material_class
        || !methods_directly_comparable(method, method, input.weighting_basis, output.weighting_basis)
    {
        push_unique(&mut findings, Finding::MethodNotDirectlyComparable);
    }

    let input_assessment = assess_distribution(input, method, criterion);
    let output_assessment = assess_distribution(output, method, criterion);
    if input_assessment.decision != Decision::Qualified {
        push_unique(&mut findings, Finding::InputDistributionInsufficient);
    }
    if output_assessment.decision != Decision::Qualified {
        push_unique(&mut findings, Finding::OutputDistributionInsufficient);
    }

    let output_fraction = output_assessment.qualifying_fraction_lower_ppm;
    let output_per_input = if comparison.incoming_mass_mg > 0 {
        (u128::from(comparison.output_mass_mg) * u128::from(output_fraction)
            / u128::from(comparison.incoming_mass_mg))
            .min(u128::from(u32::MAX)) as u32
    } else {
        0
    };
    let decision = if !findings.is_empty() {
        Decision::InsufficientEvidence
    } else {
        Decision::Qualified
    };
    ProcessComparisonAssessment {
        decision,
        findings,
        incoming_qualifying_fraction_ppm: input_assessment.qualifying_fraction_lower_ppm,
        output_qualifying_fraction_ppm: output_fraction,
        length_qualified_output_per_input_ppm: output_per_input,
        qualifying_fraction_change_ppm: i64::from(output_fraction)
            - i64::from(input_assessment.qualifying_fraction_lower_ppm),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(id: &str, digest: char) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.to_owned(),
            sha256: digest.to_string().repeat(64),
        }
    }

    fn method() -> MethodDefinition {
        MethodDefinition {
            method_id: "ASTM-D1440-07-R2024".to_owned(),
            version: "2024".to_owned(),
            citation: "ASTM D1440-07(2024)".to_owned(),
            allowed_materials: vec![MaterialClass::Cotton],
            allowed_process_states: vec![
                ProcessState::PreProcessReferenceFibre,
                ProcessState::LooseRecoveredFibre,
                ProcessState::RecoveredAfterOpening,
            ],
            supported_weighting_bases: vec![WeightingBasis::MassWeighted],
            minimum_independent_subsamples: 3,
            minimum_acquisitions: 2,
            resolution_um: 100,
            calibration_required: true,
            local_validation_required_states: vec![ProcessState::RecoveredAfterOpening],
        }
    }

    fn criterion() -> ReceiverCriterion {
        ReceiverCriterion {
            specification_id: "ring-spinning-cotton-v1".to_owned(),
            specification_version: "1.0".to_owned(),
            minimum_fibre_length_um: 12_700,
            minimum_qualifying_fraction_ppm: 600_000,
            maximum_expanded_uncertainty_ppm: 5_000,
        }
    }

    fn calibration(method: &MethodDefinition) -> CalibrationRecord {
        CalibrationRecord {
            calibration_id: "cal-001".to_owned(),
            sha256: "c".repeat(64),
            instrument_id: "array-apparatus-01".to_owned(),
            method_id: method.method_id.clone(),
            method_version: method.version.clone(),
            valid_from_ns: 10,
            valid_until_ns: 1_000,
        }
    }

    fn local_scope_validation(method: &MethodDefinition) -> LocalScopeValidation {
        LocalScopeValidation {
            method_id: method.method_id.clone(),
            method_version: method.version.clone(),
            material_class: MaterialClass::Cotton,
            process_state: ProcessState::RecoveredAfterOpening,
            evidence: evidence("local-scope", 'e'),
        }
    }

    fn distribution(method: &MethodDefinition, lot: &str, state: ProcessState) -> FibreLengthDistribution {
        FibreLengthDistribution {
            measurement_id: format!("measurement-{lot}"),
            measured_lot_id: lot.to_owned(),
            parent_lot_id: lot.to_owned(),
            physical_subject_id: "specimen-001".to_owned(),
            source_process_trial_id: None,
            method_id: method.method_id.clone(),
            method_version: method.version.clone(),
            material_class: MaterialClass::Cotton,
            process_state: state,
            weighting_basis: WeightingBasis::MassWeighted,
            observed_at_ns: 100,
            instrument_id: "array-apparatus-01".to_owned(),
            calibration: Some(calibration(method)),
            local_scope_validation: None,
            sample_plan: SamplePlan {
                independent_subsample_ids: vec![
                    "subsample-1".to_owned(),
                    "subsample-2".to_owned(),
                    "subsample-3".to_owned(),
                ],
                acquisition_ids: vec!["acquisition-1".to_owned(), "acquisition-2".to_owned()],
                sampling_frame: "randomized-panel-subsamples-v1".to_owned(),
                conditioning_ref: "conditioning-protocol-v1".to_owned(),
                sampled_mass_mg: 2_000,
            },
            underflow: FibreTail {
                boundary_um: 0,
                fraction_ppm: 0,
                expanded_uncertainty_ppm: 0,
            },
            bins: vec![
                FibreLengthBin {
                    lower_bound_um: 0,
                    upper_bound_um: 12_700,
                    fraction_ppm: 300_000,
                    expanded_uncertainty_ppm: 1_000,
                },
                FibreLengthBin {
                    lower_bound_um: 12_700,
                    upper_bound_um: 25_000,
                    fraction_ppm: 500_000,
                    expanded_uncertainty_ppm: 1_000,
                },
                FibreLengthBin {
                    lower_bound_um: 25_000,
                    upper_bound_um: 40_000,
                    fraction_ppm: 200_000,
                    expanded_uncertainty_ppm: 1_000,
                },
            ],
            overflow: FibreTail {
                boundary_um: 40_000,
                fraction_ppm: 0,
                expanded_uncertainty_ppm: 0,
            },
            unclassified_fraction_ppm: 0,
            total_expanded_uncertainty_ppm: 3_000,
            measurement_evidence: evidence("measure-001", 'a'),
        }
    }

    #[test]
    fn mass_weighted_distribution_qualifies_conservatively() {
        let method = method();
        let result = assess_distribution(
            &distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre),
            &method,
            &criterion(),
        );
        assert_eq!(result.decision, Decision::Qualified, "{:?}", result.findings);
        assert_eq!(result.qualifying_fraction_lower_ppm, 698_000);
    }

    #[test]
    fn number_weighted_distribution_cannot_qualify_mass_yield() {
        let method = method();
        let mut specimen = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        specimen.weighting_basis = WeightingBasis::NumberWeighted;
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::Rejected);
        assert!(result.findings.contains(&Finding::UnsupportedWeightingBasis));
    }

    #[test]
    fn method_outside_material_scope_is_rejected() {
        let method = method();
        let mut specimen = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        specimen.material_class = MaterialClass::Polyester;
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::Rejected);
        assert!(result.findings.contains(&Finding::MethodScopeExceeded));
    }

    #[test]
    fn recovered_state_requires_local_scope_validation() {
        let method = method();
        let specimen = distribution(&method, "output-001", ProcessState::RecoveredAfterOpening);
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::InsufficientEvidence);
        assert!(result.findings.contains(&Finding::LocalScopeValidationMissing));
    }

    #[test]
    fn local_scope_validation_is_bound_to_exact_method_material_and_process_state() {
        let method = method();
        let mut specimen = distribution(
            &method,
            "output-001",
            ProcessState::RecoveredAfterOpening,
        );
        specimen.local_scope_validation = Some(local_scope_validation(&method));
        assert_eq!(
            assess_distribution(&specimen, &method, &criterion()).decision,
            Decision::Qualified
        );

        let mut wrong = local_scope_validation(&method);
        wrong.method_version = "other-version".to_owned();
        specimen.local_scope_validation = Some(wrong);
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::InsufficientEvidence);
        assert!(result.findings.contains(&Finding::LocalScopeValidationMismatch));
    }

    #[test]
    fn photoelectric_method_profile_cannot_be_assumed_valid_for_recovered_fabric_fibres() {
        let mut d1447 = method();
        d1447.method_id = "ASTM-D1447-26".to_owned();
        d1447.version = "2026".to_owned();
        d1447.allowed_process_states = vec![ProcessState::PreProcessReferenceFibre];
        d1447.local_validation_required_states.clear();
        let specimen = distribution(
            &d1447,
            "recovered-output-001",
            ProcessState::RecoveredAfterOpening,
        );
        let result = assess_distribution(&specimen, &d1447, &criterion());
        assert_eq!(result.decision, Decision::Rejected);
        assert!(result.findings.contains(&Finding::MethodScopeExceeded));
    }

    #[test]
    fn expired_calibration_abstains() {
        let method = method();
        let mut specimen = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        specimen.calibration.as_mut().unwrap().valid_until_ns = 99;
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::InsufficientEvidence);
        assert!(result.findings.contains(&Finding::InvalidCalibration));
    }

    #[test]
    fn calibration_from_another_instrument_is_rejected() {
        let method = method();
        let mut specimen = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        specimen.calibration.as_mut().unwrap().instrument_id = "other-instrument".to_owned();
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::Rejected);
        assert!(result.findings.contains(&Finding::CalibrationInstrumentMismatch));
    }

    #[test]
    fn missing_calibration_abstains_when_method_requires_it() {
        let method = method();
        let mut specimen = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        specimen.calibration = None;
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::InsufficientEvidence);
        assert!(result.findings.contains(&Finding::MissingCalibration));
    }

    #[test]
    fn bin_gap_and_overlap_are_rejected() {
        let method = method();
        let mut gap = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        gap.bins[1].lower_bound_um += 100;
        assert_eq!(
            assess_distribution(&gap, &method, &criterion()).decision,
            Decision::Rejected
        );
        assert!(assess_distribution(&gap, &method, &criterion()).findings.contains(&Finding::BinGap));

        let mut overlap = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        overlap.bins[1].lower_bound_um -= 100;
        let result = assess_distribution(&overlap, &method, &criterion());
        assert_eq!(result.decision, Decision::Rejected);
        assert!(result.findings.contains(&Finding::BinOverlap));
    }

    #[test]
    fn non_closing_fractions_are_rejected() {
        let method = method();
        let mut specimen = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        specimen.unclassified_fraction_ppm = 10_000;
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::Rejected);
        assert!(result.findings.contains(&Finding::FractionTotalNotClosed));
    }

    #[test]
    fn duplicate_subsample_identity_is_rejected_as_pseudoreplication() {
        let method = method();
        let mut specimen = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        specimen.sample_plan.independent_subsample_ids[2] = "subsample-1".to_owned();
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::Rejected);
        assert!(result.findings.contains(&Finding::SamplePlanInvalid));
    }

    #[test]
    fn straddling_threshold_bin_receives_no_conservative_qualification_credit() {
        let method = method();
        let mut specimen = distribution(&method, "incoming-001", ProcessState::PreProcessReferenceFibre);
        specimen.bins[0].upper_bound_um = 10_000;
        specimen.bins[1].lower_bound_um = 10_000;
        specimen.bins[1].upper_bound_um = 15_000;
        specimen.bins[2].lower_bound_um = 15_000;
        let result = assess_distribution(&specimen, &method, &criterion());
        assert_eq!(result.decision, Decision::InsufficientEvidence);
        assert_eq!(result.qualifying_fraction_lower_ppm, 199_000);
        assert_eq!(result.threshold_ambiguous_fraction_ppm, 500_000);
    }

    fn bridge(method_a: &MethodDefinition, method_b: &MethodDefinition) -> MethodBridge {
        MethodBridge {
            bridge_id: "bridge-array-to-photoelectric-v1".to_owned(),
            method_a_id: method_a.method_id.clone(),
            method_a_version: method_a.version.clone(),
            method_b_id: method_b.method_id.clone(),
            method_b_version: method_b.version.clone(),
            weighting_basis: WeightingBasis::MassWeighted,
            receiver_threshold_um: 12_700,
            minimum_paired_specimens: 3,
            maximum_absolute_difference_ppm: 25_000,
            paired_specimens: vec![
                MethodBridgePair {
                    physical_specimen_id: "bridge-specimen-1".to_owned(),
                    method_a_fraction_ppm: 700_000,
                    method_b_fraction_ppm: 710_000,
                    method_a_measurement: evidence("a-1", 'a'),
                    method_b_measurement: evidence("b-1", 'b'),
                },
                MethodBridgePair {
                    physical_specimen_id: "bridge-specimen-2".to_owned(),
                    method_a_fraction_ppm: 650_000,
                    method_b_fraction_ppm: 660_000,
                    method_a_measurement: evidence("a-2", 'a'),
                    method_b_measurement: evidence("b-2", 'b'),
                },
                MethodBridgePair {
                    physical_specimen_id: "bridge-specimen-3".to_owned(),
                    method_a_fraction_ppm: 600_000,
                    method_b_fraction_ppm: 615_000,
                    method_a_measurement: evidence("a-3", 'a'),
                    method_b_measurement: evidence("b-3", 'b'),
                },
            ],
            bridge_evidence: evidence("bridge-evidence", 'd'),
        }
    }

    #[test]
    fn methods_require_a_paired_bridge_before_cross_method_comparison() {
        let a = method();
        let mut b = method();
        b.method_id = "PHOTOELECTRIC-X".to_owned();
        assert!(!methods_directly_comparable(
            &a,
            &b,
            WeightingBasis::MassWeighted,
            WeightingBasis::MassWeighted
        ));
        let result = assess_method_bridge(&bridge(&a, &b), &a, &b);
        assert!(result.equivalent_for_declared_metric, "{:?}", result.findings);
        assert_eq!(result.paired_specimen_count, 3);
    }

    #[test]
    fn bridge_with_too_few_pairs_is_not_equivalent() {
        let a = method();
        let mut b = method();
        b.method_id = "PHOTOELECTRIC-X".to_owned();
        let mut bridge = bridge(&a, &b);
        bridge.paired_specimens.pop();
        let result = assess_method_bridge(&bridge, &a, &b);
        assert!(!result.equivalent_for_declared_metric);
        assert!(result.findings.contains(&Finding::BridgeInsufficientPairs));
    }

    #[test]
    fn bridge_outside_preregistered_equivalence_margin_fails() {
        let a = method();
        let mut b = method();
        b.method_id = "PHOTOELECTRIC-X".to_owned();
        let mut bridge = bridge(&a, &b);
        bridge.paired_specimens[0].method_b_fraction_ppm = 760_000;
        let result = assess_method_bridge(&bridge, &a, &b);
        assert!(!result.equivalent_for_declared_metric);
        assert!(result.findings.contains(&Finding::BridgeEquivalenceMarginExceeded));
    }

    #[test]
    fn process_comparison_binds_exact_parent_trial_and_output_lot() {
        let method = method();
        let input = distribution(&method, "incoming-lot", ProcessState::PreProcessReferenceFibre);
        let mut output = distribution(&method, "output-lot", ProcessState::RecoveredAfterOpening);
        output.parent_lot_id = "incoming-lot".to_owned();
        output.source_process_trial_id = Some("trial-001".to_owned());
        output.local_scope_validation = Some(local_scope_validation(&method));
        let comparison = ProcessComparison {
            incoming_lot_id: "incoming-lot".to_owned(),
            output_lot_id: "output-lot".to_owned(),
            process_trial_evidence_id: "trial-001".to_owned(),
            incoming_mass_mg: 100_000,
            output_mass_mg: 60_000,
            incoming_distribution: input,
            output_distribution: output,
        };
        let result = assess_process_comparison(&comparison, &method, &criterion());
        assert_eq!(result.decision, Decision::Qualified, "{:?}", result.findings);
        assert_eq!(result.length_qualified_output_per_input_ppm, 418_800);
    }

    fn exercise_distribution_case(case_id: &str) -> Decision {
        let method = method();
        let mut specimen = distribution(
            &method,
            "incoming-001",
            ProcessState::PreProcessReferenceFibre,
        );
        match case_id {
            "FM-001-VALID-MASS-WEIGHTED" => {}
            "FM-002-NUMBER-WEIGHTED" => {
                specimen.weighting_basis = WeightingBasis::NumberWeighted;
            }
            "FM-003-MATERIAL-OUT-OF-SCOPE" => {
                specimen.material_class = MaterialClass::Polyester;
            }
            "FM-004-LOCAL-VALIDATION-MISSING" => {
                specimen.process_state = ProcessState::RecoveredAfterOpening;
            }
            "FM-005-CALIBRATION-EXPIRED" => {
                specimen.calibration.as_mut().unwrap().valid_until_ns = 99;
            }
            "FM-006-CALIBRATION-INSTRUMENT-MISMATCH" => {
                specimen.calibration.as_mut().unwrap().instrument_id =
                    "another-instrument".to_owned();
            }
            "FM-007-BIN-GAP" => {
                specimen.bins[1].lower_bound_um += 100;
            }
            "FM-008-BIN-OVERLAP" => {
                specimen.bins[1].lower_bound_um -= 100;
            }
            "FM-009-FRACTION-NOT-CLOSED" => {
                specimen.unclassified_fraction_ppm = 10_000;
            }
            "FM-010-PSEUDOREPLICATED-SUBSAMPLES" => {
                let reused_id = specimen.sample_plan.independent_subsample_ids[0].clone();
                specimen.sample_plan.independent_subsample_ids[2] = reused_id;
            }
            "FM-011-THRESHOLD-STRADDLE" => {
                specimen.bins[0].upper_bound_um = 10_000;
                specimen.bins[1].lower_bound_um = 10_000;
                specimen.bins[1].upper_bound_um = 15_000;
                specimen.bins[2].lower_bound_um = 15_000;
            }
            "FM-012-INSUFFICIENT-ACQUISITIONS" => {
                specimen.sample_plan.acquisition_ids.pop();
            }
            other => panic!("unknown distribution corpus case {other}"),
        }
        assess_distribution(&specimen, &method, &criterion()).decision
    }

    fn exercise_bridge_case(case_id: &str) -> bool {
        let a = method();
        let mut b = method();
        b.method_id = "PHOTOELECTRIC-X".to_owned();
        match case_id {
            "FM-013-CROSS-METHOD-PAIRED-BRIDGE" => {
                assess_method_bridge(&bridge(&a, &b), &a, &b)
                    .equivalent_for_declared_metric
            }
            "FM-014-BRIDGE-TOO-FEW-PAIRS" => {
                let mut candidate = bridge(&a, &b);
                candidate.paired_specimens.pop();
                assess_method_bridge(&candidate, &a, &b)
                    .equivalent_for_declared_metric
            }
            "FM-015-BRIDGE-MARGIN-EXCEEDED" => {
                let mut candidate = bridge(&a, &b);
                candidate.paired_specimens[0].method_b_fraction_ppm = 760_000;
                assess_method_bridge(&candidate, &a, &b)
                    .equivalent_for_declared_metric
            }
            other => panic!("unknown bridge corpus case {other}"),
        }
    }

    fn exercise_process_case(case_id: &str) -> Decision {
        let method = method();
        let input = distribution(
            &method,
            "incoming-lot",
            ProcessState::PreProcessReferenceFibre,
        );
        let mut output = distribution(
            &method,
            "output-lot",
            ProcessState::RecoveredAfterOpening,
        );
        output.parent_lot_id = "incoming-lot".to_owned();
        output.source_process_trial_id = Some("trial-001".to_owned());
        output.local_scope_validation = Some(local_scope_validation(&method));
        match case_id {
            "FM-016-OUTPUT-LOT-TRIAL-MISMATCH" => {
                output.source_process_trial_id = Some("different-trial".to_owned());
            }
            other => panic!("unknown process corpus case {other}"),
        }
        let comparison = ProcessComparison {
            incoming_lot_id: "incoming-lot".to_owned(),
            output_lot_id: "output-lot".to_owned(),
            process_trial_evidence_id: "trial-001".to_owned(),
            incoming_mass_mg: 100_000,
            output_mass_mg: 60_000,
            incoming_distribution: input,
            output_distribution: output,
        };
        assess_process_comparison(&comparison, &method, &criterion()).decision
    }

    #[test]
    fn synthetic_corpus_drives_all_method_scope_and_lineage_regressions() {
        const CORPUS: &str = include_str!("../fixtures/metrology-cases-v0.1.json");
        const SCHEMA: &str = include_str!("../bundle.schema.json");
        const PROFILES: &str = include_str!("../fixtures/method-profiles.example.json");
        let corpus: serde_json::Value = serde_json::from_str(CORPUS).unwrap();
        let schema: serde_json::Value = serde_json::from_str(SCHEMA).unwrap();
        let profiles: serde_json::Value = serde_json::from_str(PROFILES).unwrap();
        assert_eq!(corpus["schema_version"], "0.1.0");
        assert_eq!(corpus["corpus_id"], "TEXTILE-FIBRE-METROLOGY-CASES-001");
        assert_eq!(corpus["fixture_class"], "synthetic");
        assert_eq!(schema["$schema"], "https://json-schema.org/draft/2020-12/schema");
        assert_eq!(corpus["cases"].as_array().unwrap().len(), 16);
        assert_eq!(profiles["schema_version"], "0.1.0");
        let method_profiles = profiles["profiles"].as_array().unwrap();
        assert_eq!(method_profiles.len(), 3);
        let photoelectric = method_profiles
            .iter()
            .find(|profile| profile["method_id"] == "ASTM-D1447-26")
            .unwrap();
        assert!(
            !photoelectric["allowed_process_states"]
                .as_array()
                .unwrap()
                .iter()
                .any(|state| state == "recovered_after_opening")
        );
        let single_fibre = method_profiles
            .iter()
            .find(|profile| profile["method_id"] == "ISO-6989-1981")
            .unwrap();
        assert_eq!(
            single_fibre["supported_weighting_bases"][0],
            "number_weighted"
        );

        let mut seen = BTreeSet::new();
        for case in corpus["cases"].as_array().unwrap() {
            let case_id = case["case_id"].as_str().unwrap();
            assert!(seen.insert(case_id), "duplicate corpus case {case_id}");
            let operation = case["operation"].as_str().unwrap();
            let expected = case["expected"].as_str().unwrap();
            let actual = match operation {
                "assess_distribution" => {
                    exercise_distribution_case(case_id) == match expected {
                        "qualified" => Decision::Qualified,
                        "insufficient_evidence" => Decision::InsufficientEvidence,
                        "rejected" => Decision::Rejected,
                        other => panic!("unknown distribution expectation {other}"),
                    }
                }
                "assess_method_bridge" => {
                    exercise_bridge_case(case_id) == (expected == "equivalent")
                }
                "assess_process_comparison" => {
                    exercise_process_case(case_id) == Decision::InsufficientEvidence
                        && expected == "insufficient_evidence"
                }
                other => panic!("unknown corpus operation {other}"),
            };
            assert!(actual, "corpus expectation failed for {case_id}");
        }
        assert_eq!(seen.len(), 16);
    }

    #[test]
    fn process_comparison_rejects_borrowed_output_lot_or_trial() {
        let method = method();
        let input = distribution(&method, "incoming-lot", ProcessState::PreProcessReferenceFibre);
        let mut output = distribution(&method, "output-lot", ProcessState::RecoveredAfterOpening);
        output.parent_lot_id = "different-incoming-lot".to_owned();
        output.source_process_trial_id = Some("other-trial".to_owned());
        output.local_scope_validation = Some(local_scope_validation(&method));
        let comparison = ProcessComparison {
            incoming_lot_id: "incoming-lot".to_owned(),
            output_lot_id: "output-lot".to_owned(),
            process_trial_evidence_id: "trial-001".to_owned(),
            incoming_mass_mg: 100_000,
            output_mass_mg: 60_000,
            incoming_distribution: input,
            output_distribution: output,
        };
        let result = assess_process_comparison(&comparison, &method, &criterion());
        assert_eq!(result.decision, Decision::InsufficientEvidence);
        assert!(result.findings.contains(&Finding::ProcessLotIdentityMismatch));
        assert!(result.findings.contains(&Finding::ProcessTrialIdentityMismatch));
    }
}
