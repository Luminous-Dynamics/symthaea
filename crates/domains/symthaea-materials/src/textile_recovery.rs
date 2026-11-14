// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-qualified textile recovery-route contracts.
//!
//! This module prevents a fibre prediction from becoming a recyclability claim.
//! It qualifies a *candidate route* only when route-specific physical evidence,
//! mass accounting and independent receiving-process acceptance are present.
//! It does not select a route from model scores, authenticate evidence issuers,
//! or prove legal, safety, chemical or environmental compliance.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Integer scale for mass fractions: one million parts represent 100%.
pub const FRACTION_SCALE_PPM: u64 = 1_000_000;

/// Candidate destination for a textile item or material lot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryRoute {
    /// Continue use as a garment or textile product without repair.
    ReuseAsIs,
    /// Repair the garment/product before returning it to use.
    Repair,
    /// Recover a reusable component (for example, a fastener or trim).
    ComponentRecovery,
    /// Mechanically open textile material into fibres.
    MechanicalFibreRecycling,
    /// Chemically process known polymer material into a recovered polymer stream.
    ChemicalFibreRecycling,
    /// Convert material into a lower-grade but explicitly accepted product stream.
    Downcycling,
}

/// Semantic role of an evidence artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    /// Physical item/lot inventory and measured incoming mass.
    PhysicalInventory,
    /// Human inspection of product condition and safe-use attributes.
    ConditionInspection,
    /// Calibrated measurement supporting fibre composition.
    CompositionMeasurement,
    /// Physical sorting/routing trial that creates identified material streams.
    SortingTrial,
    /// Instrument calibration record in force at acquisition.
    CalibrationRecord,
    /// Physical repair operation and post-repair testing.
    RepairTrial,
    /// Physical component harvesting/disassembly trial.
    ComponentRecoveryTrial,
    /// Physical mechanical-opening/recycling trial.
    MechanicalRecyclingTrial,
    /// Versioned and human-reviewed sampling protocol used for fibre-length measurement.
    FibreLengthSamplingPlan,
    /// Measured fibre-length distribution for an actual recovered output.
    FibreLengthMeasurement,
    /// Physical test of a chemical route's material/process compatibility.
    ChemicalCompatibilityTrial,
    /// Physical chemical-recycling process trial.
    ChemicalRecyclingTrial,
    /// Physical downcycling process trial.
    DowncyclingTrial,
    /// Physical inspection binding a specimen to a versioned design dossier.
    DesignRevisionConformity,
    /// Acceptance/rejection decision by the downstream receiving process.
    DownstreamAcceptance,
}

/// Evidence maturity. Higher levels do not imply cryptographic authenticity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceTier {
    /// Generated fixture with no physical specimen.
    Synthetic,
    /// Simulation-only result.
    Simulation,
    /// Model output or inference without independent physical measurement.
    ModelInference,
    /// Instrument measurement; calibration and source still require checking.
    SensorMeasured,
    /// Recorded human inspection.
    HumanInspected,
    /// Physical process trial on identified material.
    PhysicalTrial,
    /// Independent receiving-process acceptance with a distinct verifier.
    IndependentAcceptance,
}

/// Source-bound reference to a supporting evidence artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// Stable evidence identifier.
    pub evidence_id: String,
    /// SHA-256 of the exact artifact bytes, lowercase hexadecimal.
    pub sha256: String,
    /// Evidence role.
    pub kind: EvidenceKind,
    /// Declared evidence maturity.
    pub tier: EvidenceTier,
    /// Measurement/inspection/process method identifier and version.
    pub method_ref: String,
    /// Organization/person responsible for creating the evidence.
    pub issuer_id: String,
    /// Independent verifier; required for IndependentAcceptance.
    pub verifier_id: Option<String>,
    /// Observation/record time in nanoseconds from a declared epoch.
    pub observed_at_ns: u64,
}

/// Physical condition of the garment/product at inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductCondition {
    /// Fit for intended use without repair, according to the inspected criteria.
    Wearable,
    /// Not fit for current use but has a declared repair pathway.
    Repairable,
    /// Not fit for product reuse under the inspected criteria.
    NonWearable,
    /// Condition was not determined.
    Unknown,
}

/// Result of a safety/hygiene inspection relevant to the declared reuse route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UseSafetyStatus {
    /// The declared inspection protocol passed.
    Passed,
    /// The declared inspection protocol failed.
    Failed,
    /// Safety/hygiene was not determined.
    Unknown,
}

/// Physical condition evidence, not a computer-vision category prediction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionEvidence {
    /// Inspected state.
    pub condition: ProductCondition,
    /// Safety/hygiene result for the declared intended use.
    pub safety: UseSafetyStatus,
    /// Human inspection evidence.
    pub evidence: EvidenceRef,
}

/// Disposition of a composition estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionDisposition {
    /// Measurement passed its declared instrument/calibration admission contract.
    Admitted,
    /// Measurement is incomplete, out of domain or otherwise abstained.
    Unknown,
    /// Measurement has contradictory/malformed evidence.
    Rejected,
}

/// Composition evidence must identify both the measurement and calibration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionEvidence {
    /// Admission disposition from the sensing contract.
    pub disposition: CompositionDisposition,
    /// Upper bound on fibre mass fraction that remains unidentified.
    pub unknown_fraction_upper_ppm: u32,
    /// Composition measurement artifact.
    pub measurement: EvidenceRef,
    /// Calibration record artifact.
    pub calibration: EvidenceRef,
}

/// Result of a process trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialOutcome {
    /// Declared process and output tests passed.
    Passed,
    /// A required process/output criterion failed.
    Failed,
    /// Outcome was not determined.
    Unknown,
}

/// Mass categories in a physical recovery trial.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MassBalance {
    /// Incoming mass measured for the trial.
    pub input_mass_mg: u64,
    /// Exact receiver-bound identity of the accepted output lot measured below.
    pub accepted_output_lot_id: String,
    /// Mass accepted into the target output stream.
    pub accepted_output_mass_mg: u64,
    /// Recovered reusable components.
    pub recovered_component_mass_mg: u64,
    /// Recovered trim or non-textile attachment mass.
    pub recovered_trim_mass_mg: u64,
    /// Rejected material mass.
    pub rejected_mass_mg: u64,
    /// Measured waste/residual mass.
    pub waste_mass_mg: u64,
    /// Combined expanded measurement uncertainty for the mass categories.
    pub uncertainty_mass_mg: u64,
}

impl MassBalance {
    /// Returns the absolute unexplained mass difference in milligrams.
    pub fn unexplained_mass_mg(&self) -> u64 {
        let accounted = u128::from(self.accepted_output_mass_mg)
            + u128::from(self.recovered_component_mass_mg)
            + u128::from(self.recovered_trim_mass_mg)
            + u128::from(self.rejected_mass_mg)
            + u128::from(self.waste_mass_mg);
        let input = u128::from(self.input_mass_mg);
        input.abs_diff(accounted).min(u128::from(u64::MAX)) as u64
    }

    /// Checks that mass closure is within declared measurement uncertainty and policy ceiling.
    pub fn is_closed(&self, max_uncertainty_ppm: u32) -> bool {
        if self.input_mass_mg == 0 {
            return false;
        }
        let uncertainty_ppm = u128::from(self.uncertainty_mass_mg)
            .saturating_mul(u128::from(FRACTION_SCALE_PPM))
            / u128::from(self.input_mass_mg);
        uncertainty_ppm <= u128::from(max_uncertainty_ppm)
            && self.unexplained_mass_mg() <= self.uncertainty_mass_mg
    }
}

/// Exact design revision and specimen binding for a physical process trial.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesignRevisionBinding {
    /// Product ID from the static textile design dossier.
    pub product_id: String,
    /// Exact design revision tested.
    pub design_revision: String,
    /// SHA-256 digest of the exact design dossier bytes.
    pub dossier_sha256: String,
    /// Physical specimen/item identity used in this process trial.
    pub specimen_id: String,
    /// Inspection evidence that this specimen conforms to the declared design revision.
    pub conformity_evidence: EvidenceRef,
}

/// Evidence that a route was physically trialled on identified material.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessTrialEvidence {
    /// Trial disposition.
    pub outcome: TrialOutcome,
    /// Physical sample/specimen identity tested in this trial.
    pub physical_sample_id: String,
    /// Physical process-trial artifact.
    pub evidence: EvidenceRef,
    /// Optional exact design revision binding for a trial on a designed product.
    pub design_binding: Option<DesignRevisionBinding>,
    /// Required mass accounting for destructive routes.
    pub mass_balance: Option<MassBalance>,
}

/// One measured fibre-length interval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FibreLengthBin {
    /// Inclusive lower bound of the bin, in micrometres.
    pub lower_bound_um: u32,
    /// Exclusive upper bound of the bin, in micrometres.
    pub upper_bound_um: u32,
    /// Share of recovered fibre mass in the bin, parts per million.
    pub mass_fraction_ppm: u32,
}

/// Fibre-length distribution measured from an actual recovered material stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FibreLengthEvidence {
    /// Method/standard and version used to measure fibre length.
    pub method_ref: String,
    /// Human-reviewed, content-addressed sampling protocol for the exact output-lot sample.
    pub sampling_plan_evidence: EvidenceRef,
    /// Measured sample mass in milligrams taken from the accepted output lot.
    pub sample_mass_mg: u64,
    /// Number of independently measured subsamples/replicates represented by this distribution.
    pub replicate_count: u32,
    /// Declared relative expanded measurement uncertainty, parts per million.
    pub measurement_uncertainty_ppm: u32,
    /// Coverage probability for the uncertainty declaration, parts per million (for example, 950_000 = 95%).
    pub measurement_coverage_probability_ppm: u32,
    /// Exact accepted output lot physically sampled for this distribution.
    pub accepted_output_lot_id: String,
    /// Exact process-trial evidence ID which generated this accepted output lot.
    pub source_trial_evidence_id: String,
    /// Instrument/measurement evidence.
    pub evidence: EvidenceRef,
    /// Contiguous fibre-length partition of [0, u32::MAX) micrometres.
    ///
    /// The final bin must end at u32::MAX, which is the open-ended upper-tail
    /// sentinel. This prevents a closed mass sum from silently omitting an
    /// unreported length interval.
    pub bins: Vec<FibreLengthBin>,
}

/// Required downstream output criteria, stated by the receiving process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryPolicy {
    /// Maximum unresolved fibre mass fraction permitted for a candidate route.
    pub max_unknown_fraction_ppm: u32,
    /// Maximum mass-balance expanded uncertainty as a share of input mass.
    pub max_mass_balance_uncertainty_ppm: u32,
    /// Maximum relative expanded uncertainty permitted for the fibre-length measurement.
    pub max_fibre_length_measurement_uncertainty_ppm: u32,
    /// Minimum mass of the accepted output lot that must be sampled.
    pub minimum_fibre_length_sample_mass_mg: u64,
    /// Minimum number of independent fibre-length subsamples/replicates.
    pub minimum_fibre_length_replicates: u32,
    /// Minimum expanded-uncertainty coverage probability required by the receiver.
    pub minimum_fibre_length_measurement_coverage_probability_ppm: u32,
    /// Minimum fibre length required by the declared downstream spinning route.
    pub minimum_fibre_length_um: u32,
    /// Minimum recovered fibre mass fraction in bins meeting that length threshold.
    pub minimum_mass_fraction_at_or_above_length_ppm: u32,
}

impl RecoveryPolicy {
    /// Validates policy scales before they are used in route qualification.
    pub fn validate(&self) -> bool {
        self.max_unknown_fraction_ppm <= FRACTION_SCALE_PPM as u32
            && self.max_mass_balance_uncertainty_ppm <= FRACTION_SCALE_PPM as u32
            && self.max_fibre_length_measurement_uncertainty_ppm <= FRACTION_SCALE_PPM as u32
            && self.minimum_fibre_length_sample_mass_mg > 0
            && self.minimum_fibre_length_replicates > 0
            && self.minimum_fibre_length_measurement_coverage_probability_ppm > 0
            && self.minimum_fibre_length_measurement_coverage_probability_ppm
                <= FRACTION_SCALE_PPM as u32
            && self.minimum_fibre_length_um > 0
            && self.minimum_mass_fraction_at_or_above_length_ppm <= FRACTION_SCALE_PPM as u32
    }
}

/// Outcome of a downstream receiving-process evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceptanceOutcome {
    /// Material passed the declared receiving specification.
    Accepted,
    /// Material failed the declared receiving specification.
    Rejected,
    /// Material may be accepted only after further work or a waiver.
    Conditional,
    /// No final decision was recorded.
    Unknown,
}

/// Versioned downstream specification and its independent acceptance record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownstreamAcceptance {
    /// Exact receiving specification identifier/version.
    pub specification_id: String,
    /// Exact physical item/output lot accepted by the receiver.
    pub accepted_output_lot_id: String,
    /// Exact evidence ID for the inspection or process trial which produced the accepted subject.
    pub basis_evidence_id: String,
    /// Recorded acceptance disposition.
    pub outcome: AcceptanceOutcome,
    /// Independent receiving-process evidence.
    pub evidence: EvidenceRef,
}

/// Chemical compatibility assessment backed by a physical test.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChemicalCompatibilityEvidence {
    /// Compatibility test disposition.
    pub outcome: TrialOutcome,
    /// Physical chemistry/process test evidence.
    pub evidence: EvidenceRef,
}

/// All evidence supplied for one physical item/lot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryEvidenceBundle {
    /// Physical item/lot identifier.
    pub lot_id: String,
    /// Independently measured incoming mass; zero is not a valid lot.
    pub input_mass_mg: u64,
    /// Physical inventory evidence.
    pub inventory: EvidenceRef,
    /// Optional human condition/safety inspection.
    pub condition: Option<ConditionEvidence>,
    /// Optional calibrated material-composition evidence.
    pub composition: Option<CompositionEvidence>,
    /// Optional route process trial.
    pub process_trial: Option<ProcessTrialEvidence>,
    /// Optional measured fibre-length distribution from the actual trial output.
    pub fibre_length: Option<FibreLengthEvidence>,
    /// Optional chemical compatibility trial for the declared route.
    pub chemical_compatibility: Option<ChemicalCompatibilityEvidence>,
    /// Optional downstream acceptance record.
    pub downstream_acceptance: Option<DownstreamAcceptance>,
}

/// Route qualification disposition; this is not a legal or environmental certification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteDisposition {
    /// Required evidence is present and all declared gates pass.
    Eligible,
    /// Evidence is missing, unknown or insufficient to authorize the claim.
    InsufficientEvidence,
    /// A declared hard criterion failed or evidence is structurally invalid.
    Ineligible,
}

/// Stable explanation for a route assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteReason {
    /// Empty lot identity or zero incoming mass.
    InvalidPhysicalInventory,
    /// Inventory record has the wrong role, tier or malformed digest.
    InvalidInventoryEvidence,
    /// No condition inspection was supplied.
    MissingConditionInspection,
    /// Condition is incompatible with reuse as-is.
    ProductNotWearable,
    /// Product condition was not determined by the inspection.
    ProductConditionUnknown,
    /// Use safety/hygiene inspection failed.
    UseSafetyNotPassed,
    /// Use safety/hygiene status was not determined.
    UseSafetyUnknown,
    /// Required physical process trial is absent.
    MissingProcessTrial,
    /// Process trial failed.
    ProcessTrialNotPassed,
    /// Process trial outcome is unknown.
    ProcessTrialUnknown,
    /// Process trial evidence has the wrong role or tier.
    InvalidProcessTrialEvidence,
    /// Physical trial has no specimen identity.
    InvalidPhysicalTrialIdentity,
    /// Design dossier/specimen/revision binding is malformed or lacks conformity evidence.
    InvalidDesignRevisionBinding,
    /// Destructive route has no measured mass balance.
    MissingMassBalance,
    /// Mass balance does not close within measurement uncertainty/policy.
    MassBalanceNotClosed,
    /// Trial input mass differs from the physical lot inventory mass.
    MassBalanceInputMismatch,
    /// Composition measurement/calibration evidence is absent.
    MissingCompositionEvidence,
    /// Composition record is unknown or rejected.
    CompositionNotAdmitted,
    /// Unknown fibre fraction exceeds the route policy.
    UnknownFibreFractionAbovePolicy,
    /// Composition/calibration references are malformed or mis-typed.
    InvalidCompositionEvidence,
    /// Fibre-length distribution is absent.
    MissingFibreLengthEvidence,
    /// Fibre-length distribution is malformed or does not close to 100%.
    InvalidFibreLengthDistribution,
    /// Fibre-length measurements are not bound to the exact trial and accepted output lot.
    FibreLengthTrialBindingMismatch,
    /// Fibre-length sample mass exceeds the accepted output lot mass.
    FibreLengthSampleExceedsOutput,
    /// Fibre-length sampling plan, replicate count or uncertainty declaration is invalid.
    InvalidFibreLengthMeasurementProtocol,
    /// Fibre-length measurement uncertainty exceeds the receiving route policy.
    FibreLengthMeasurementUncertaintyAbovePolicy,
    /// Fibre-length sample mass or independent replicate count is below the receiver's protocol.
    FibreLengthSamplingBelowPolicy,
    /// Expanded-uncertainty coverage probability is below the receiver's protocol.
    FibreLengthCoverageBelowPolicy,
    /// Measured fibre-length distribution fails the receiving route policy.
    FibreLengthBelowPolicy,
    /// Chemical compatibility trial is absent.
    MissingChemicalCompatibilityEvidence,
    /// Chemical compatibility evidence is malformed or wrong-tier.
    InvalidChemicalCompatibilityEvidence,
    /// Chemical compatibility trial failed.
    ChemicalCompatibilityFailed,
    /// Chemical compatibility result is unknown.
    ChemicalCompatibilityUnknown,
    /// Evidence identifiers are duplicated within the bundle.
    DuplicateEvidenceIdentity,
    /// No independent downstream receiving-process acceptance is present.
    MissingDownstreamAcceptance,
    /// Receiving process explicitly rejected the output.
    DownstreamAcceptanceRejected,
    /// Receiving process acceptance remains conditional or unknown.
    DownstreamAcceptanceUnresolved,
    /// Downstream acceptance is not independent or is malformed.
    InvalidDownstreamAcceptance,
    /// Downstream acceptance is not bound to this exact item/output lot and its producing evidence.
    DownstreamAcceptanceBindingMismatch,
    /// Receiving issuer or verifier is not independent of the originating process/inspection issuer.
    DownstreamAcceptanceNotIndependentOfBasis,
    /// Policy itself has impossible/out-of-range values.
    InvalidPolicy,
}

/// Result of qualifying one candidate recovery route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteAssessment {
    /// Candidate route evaluated.
    pub route: RecoveryRoute,
    /// Qualification disposition.
    pub disposition: RouteDisposition,
    /// Stable explanation codes; empty only for Eligible.
    pub reasons: Vec<RouteReason>,
    /// Evidence identifiers considered by this assessment.
    pub evidence_ids: Vec<String>,
}

/// Checks whether an identifier is non-empty after trimming.
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

fn valid_evidence(
    evidence: &EvidenceRef,
    expected_kind: EvidenceKind,
    minimum_tier: EvidenceTier,
) -> bool {
    if evidence.kind != expected_kind
        || evidence.tier != minimum_tier
        || evidence.evidence_id.trim().is_empty()
        || evidence.method_ref.trim().is_empty()
        || evidence.issuer_id.trim().is_empty()
        || !valid_sha256(&evidence.sha256)
    {
        return false;
    }
    if evidence.tier == EvidenceTier::IndependentAcceptance {
        evidence
            .verifier_id
            .as_deref()
            .is_some_and(|verifier| !verifier.trim().is_empty() && verifier != evidence.issuer_id)
    } else {
        true
    }
}

fn require_downstream_acceptance(
    acceptance: Option<&DownstreamAcceptance>,
    expected_output_lot_id: &str,
    expected_basis_evidence_id: &str,
    expected_basis_issuer_id: &str,
    reasons: &mut Vec<RouteReason>,
    evidence_ids: &mut Vec<String>,
) {
    let Some(acceptance) = acceptance else {
        reasons.push(RouteReason::MissingDownstreamAcceptance);
        return;
    };
    evidence_ids.push(acceptance.evidence.evidence_id.clone());
    if acceptance.specification_id.trim().is_empty()
        || !valid_evidence(
            &acceptance.evidence,
            EvidenceKind::DownstreamAcceptance,
            EvidenceTier::IndependentAcceptance,
        )
    {
        reasons.push(RouteReason::InvalidDownstreamAcceptance);
        return;
    }
    if !nonempty(expected_output_lot_id)
        || !nonempty(expected_basis_evidence_id)
        || acceptance.accepted_output_lot_id != expected_output_lot_id
        || acceptance.basis_evidence_id != expected_basis_evidence_id
    {
        reasons.push(RouteReason::DownstreamAcceptanceBindingMismatch);
        return;
    }
    let receiver_issuer = acceptance.evidence.issuer_id.as_str();
    let verifier = acceptance.evidence.verifier_id.as_deref().unwrap_or("");
    if !nonempty(expected_basis_issuer_id)
        || receiver_issuer == expected_basis_issuer_id
        || verifier == expected_basis_issuer_id
    {
        reasons.push(RouteReason::DownstreamAcceptanceNotIndependentOfBasis);
        return;
    }
    match acceptance.outcome {
        AcceptanceOutcome::Accepted => {}
        AcceptanceOutcome::Rejected => reasons.push(RouteReason::DownstreamAcceptanceRejected),
        AcceptanceOutcome::Conditional | AcceptanceOutcome::Unknown => {
            reasons.push(RouteReason::DownstreamAcceptanceUnresolved)
        }
    }
}

fn trial_acceptance_binding(bundle: &RecoveryEvidenceBundle) -> (&str, &str, &str) {
    let Some(trial) = bundle.process_trial.as_ref() else {
        return ("", "", "");
    };
    let output_lot_id = trial
        .mass_balance
        .as_ref()
        .map(|balance| balance.accepted_output_lot_id.as_str())
        .unwrap_or("");
    (
        output_lot_id,
        trial.evidence.evidence_id.as_str(),
        trial.evidence.issuer_id.as_str(),
    )
}

fn require_chemical_compatibility(
    compatibility: Option<&ChemicalCompatibilityEvidence>,
    reasons: &mut Vec<RouteReason>,
    evidence_ids: &mut Vec<String>,
) {
    let Some(compatibility) = compatibility else {
        reasons.push(RouteReason::MissingChemicalCompatibilityEvidence);
        return;
    };
    evidence_ids.push(compatibility.evidence.evidence_id.clone());
    if !valid_evidence(
        &compatibility.evidence,
        EvidenceKind::ChemicalCompatibilityTrial,
        EvidenceTier::PhysicalTrial,
    ) {
        reasons.push(RouteReason::InvalidChemicalCompatibilityEvidence);
    }
    match compatibility.outcome {
        TrialOutcome::Passed => {}
        TrialOutcome::Failed => reasons.push(RouteReason::ChemicalCompatibilityFailed),
        TrialOutcome::Unknown => reasons.push(RouteReason::ChemicalCompatibilityUnknown),
    }
}

fn add_trial_checks(
    trial: Option<&ProcessTrialEvidence>,
    expected_kind: EvidenceKind,
    require_mass_balance: bool,
    expected_input_mass_mg: u64,
    policy: &RecoveryPolicy,
    reasons: &mut Vec<RouteReason>,
    evidence_ids: &mut Vec<String>,
) {
    let Some(trial) = trial else {
        reasons.push(RouteReason::MissingProcessTrial);
        return;
    };
    evidence_ids.push(trial.evidence.evidence_id.clone());
    if !valid_evidence(&trial.evidence, expected_kind, EvidenceTier::PhysicalTrial) {
        reasons.push(RouteReason::InvalidProcessTrialEvidence);
    }
    if !nonempty(&trial.physical_sample_id) {
        reasons.push(RouteReason::InvalidPhysicalTrialIdentity);
    }
    if let Some(binding) = &trial.design_binding {
        evidence_ids.push(binding.conformity_evidence.evidence_id.clone());
        if !nonempty(&binding.product_id)
            || !nonempty(&binding.design_revision)
            || !nonempty(&binding.specimen_id)
            || binding.specimen_id != trial.physical_sample_id
            || !valid_sha256(&binding.dossier_sha256)
            || !valid_evidence(
                &binding.conformity_evidence,
                EvidenceKind::DesignRevisionConformity,
                EvidenceTier::PhysicalTrial,
            )
        {
            reasons.push(RouteReason::InvalidDesignRevisionBinding);
        }
    }
    match trial.outcome {
        TrialOutcome::Passed => {}
        TrialOutcome::Failed => reasons.push(RouteReason::ProcessTrialNotPassed),
        TrialOutcome::Unknown => reasons.push(RouteReason::ProcessTrialUnknown),
    }
    if require_mass_balance {
        match &trial.mass_balance {
            None => reasons.push(RouteReason::MissingMassBalance),
            Some(balance) => {
                if balance.input_mass_mg != expected_input_mass_mg {
                    reasons.push(RouteReason::MassBalanceInputMismatch);
                }
                if !nonempty(&balance.accepted_output_lot_id) {
                    reasons.push(RouteReason::FibreLengthTrialBindingMismatch);
                }
                if !balance.is_closed(policy.max_mass_balance_uncertainty_ppm) {
                    reasons.push(RouteReason::MassBalanceNotClosed);
                }
            }
        }
    }
}

fn fibre_length_binds_to_trial(
    evidence: &FibreLengthEvidence,
    trial: &ProcessTrialEvidence,
) -> bool {
    if !nonempty(&evidence.accepted_output_lot_id)
        || !nonempty(&evidence.source_trial_evidence_id)
        || evidence.source_trial_evidence_id != trial.evidence.evidence_id
    {
        return false;
    }
    trial.mass_balance.as_ref().is_some_and(|balance| {
        nonempty(&balance.accepted_output_lot_id)
            && balance.accepted_output_lot_id == evidence.accepted_output_lot_id
            && balance.accepted_output_mass_mg > 0
            && evidence.sample_mass_mg > 0
            && evidence.sample_mass_mg <= balance.accepted_output_mass_mg
    })
}

fn fibre_length_protocol_valid(evidence: &FibreLengthEvidence) -> bool {
    nonempty(&evidence.method_ref)
        && valid_evidence(
            &evidence.sampling_plan_evidence,
            EvidenceKind::FibreLengthSamplingPlan,
            EvidenceTier::HumanInspected,
        )
        && evidence.sample_mass_mg > 0
        && evidence.replicate_count > 0
        && evidence.measurement_uncertainty_ppm <= FRACTION_SCALE_PPM as u32
        && evidence.measurement_coverage_probability_ppm > 0
        && evidence.measurement_coverage_probability_ppm <= FRACTION_SCALE_PPM as u32
        && evidence.evidence.method_ref == evidence.method_ref
}

fn fibre_length_sampling_meets_policy(
    evidence: &FibreLengthEvidence,
    policy: &RecoveryPolicy,
) -> bool {
    evidence.sample_mass_mg >= policy.minimum_fibre_length_sample_mass_mg
        && evidence.replicate_count >= policy.minimum_fibre_length_replicates
}

fn fibre_length_coverage_meets_policy(
    evidence: &FibreLengthEvidence,
    policy: &RecoveryPolicy,
) -> bool {
    evidence.measurement_coverage_probability_ppm
        >= policy.minimum_fibre_length_measurement_coverage_probability_ppm
}

/// Converts the receiver's nominal length floor into a conservative measured
/// threshold using the declared relative expanded uncertainty.
///
/// For a measured length x with relative expanded uncertainty U, the lower
/// end of the symmetric relative interval is x/(1+U). Therefore a bin is
/// counted only when its lower edge is at least nominal_floor * (1+U).
fn conservative_measured_length_floor_um(evidence: &FibreLengthEvidence, nominal_floor_um: u32) -> Option<u32> {
    let scale = u64::from(FRACTION_SCALE_PPM as u32);
    let factor = scale.checked_add(u64::from(evidence.measurement_uncertainty_ppm))?;
    let numerator = u64::from(nominal_floor_um).checked_mul(factor)?;
    let rounded_up = numerator.checked_add(scale.checked_sub(1)?)? / scale;
    u32::try_from(rounded_up).ok()
}

fn fibre_length_fraction_at_or_above(
    evidence: &FibreLengthEvidence,
    minimum_length_um: u32,
) -> Option<u32> {
    if !fibre_length_protocol_valid(evidence) || evidence.bins.is_empty() {
        return None;
    }
    let measured_floor =
        conservative_measured_length_floor_um(evidence, minimum_length_um)?;
    let mut total = 0_u64;
    let mut qualifying = 0_u64;
    let mut previous_upper = None;

    for (index, bin) in evidence.bins.iter().enumerate() {
        if bin.lower_bound_um >= bin.upper_bound_um
            || bin.mass_fraction_ppm > FRACTION_SCALE_PPM as u32
        {
            return None;
        }
        if index == 0 {
            if bin.lower_bound_um != 0 {
                return None;
            }
        } else if previous_upper != Some(bin.lower_bound_um) {
            // Require exact contiguity: gaps and overlaps both invalidate the
            // distribution, even when the listed mass fractions sum to 100%.
            return None;
        }

        total = total.checked_add(u64::from(bin.mass_fraction_ppm))?;
        if bin.lower_bound_um >= measured_floor {
            qualifying = qualifying.checked_add(u64::from(bin.mass_fraction_ppm))?;
        }
        previous_upper = Some(bin.upper_bound_um);
    }

    if total != FRACTION_SCALE_PPM || previous_upper != Some(u32::MAX) {
        return None;
    }
    u32::try_from(qualifying).ok()
}

fn add_composition_checks(
    composition: Option<&CompositionEvidence>,
    policy: &RecoveryPolicy,
    reasons: &mut Vec<RouteReason>,
    evidence_ids: &mut Vec<String>,
) {
    let Some(composition) = composition else {
        reasons.push(RouteReason::MissingCompositionEvidence);
        return;
    };
    evidence_ids.push(composition.measurement.evidence_id.clone());
    evidence_ids.push(composition.calibration.evidence_id.clone());
    if !valid_evidence(
        &composition.measurement,
        EvidenceKind::CompositionMeasurement,
        EvidenceTier::SensorMeasured,
    ) || !valid_evidence(
        &composition.calibration,
        EvidenceKind::CalibrationRecord,
        EvidenceTier::SensorMeasured,
    ) {
        reasons.push(RouteReason::InvalidCompositionEvidence);
    }
    match composition.disposition {
        CompositionDisposition::Admitted => {}
        CompositionDisposition::Unknown | CompositionDisposition::Rejected => {
            reasons.push(RouteReason::CompositionNotAdmitted)
        }
    }
    if composition.unknown_fraction_upper_ppm > policy.max_unknown_fraction_ppm {
        reasons.push(RouteReason::UnknownFibreFractionAbovePolicy);
    }
}

/// Qualifies one candidate route from source-bound evidence and a declared receiving policy.
///
/// Missing/unknown data never promote a route. The function is deterministic and
/// does not fetch data, invoke models or establish that any evidence issuer is truthful.
pub fn qualify_recovery_route(
    route: RecoveryRoute,
    bundle: &RecoveryEvidenceBundle,
    policy: &RecoveryPolicy,
) -> RouteAssessment {
    let mut reasons = Vec::new();
    let mut evidence_ids = Vec::new();

    if !policy.validate() {
        reasons.push(RouteReason::InvalidPolicy);
    }
    if bundle.lot_id.trim().is_empty() || bundle.input_mass_mg == 0 {
        reasons.push(RouteReason::InvalidPhysicalInventory);
    }
    evidence_ids.push(bundle.inventory.evidence_id.clone());
    if !valid_evidence(
        &bundle.inventory,
        EvidenceKind::PhysicalInventory,
        EvidenceTier::SensorMeasured,
    ) {
        reasons.push(RouteReason::InvalidInventoryEvidence);
    }

    match route {
        RecoveryRoute::ReuseAsIs => {
            let Some(condition) = &bundle.condition else {
                reasons.push(RouteReason::MissingConditionInspection);
                return assessment(route, reasons, evidence_ids);
            };
            evidence_ids.push(condition.evidence.evidence_id.clone());
            if !valid_evidence(
                &condition.evidence,
                EvidenceKind::ConditionInspection,
                EvidenceTier::HumanInspected,
            ) {
                reasons.push(RouteReason::MissingConditionInspection);
            }
            match condition.condition {
                ProductCondition::Wearable => {}
                ProductCondition::Unknown => reasons.push(RouteReason::ProductConditionUnknown),
                ProductCondition::Repairable | ProductCondition::NonWearable => {
                    reasons.push(RouteReason::ProductNotWearable)
                }
            }
            match condition.safety {
                UseSafetyStatus::Passed => {}
                UseSafetyStatus::Failed => reasons.push(RouteReason::UseSafetyNotPassed),
                UseSafetyStatus::Unknown => reasons.push(RouteReason::UseSafetyUnknown),
            }
            let (basis_evidence_id, basis_issuer_id) = bundle
                .condition
                .as_ref()
                .map(|condition| {
                    (
                        condition.evidence.evidence_id.as_str(),
                        condition.evidence.issuer_id.as_str(),
                    )
                })
                .unwrap_or(("", ""));
            require_downstream_acceptance(
                bundle.downstream_acceptance.as_ref(),
                &bundle.lot_id,
                basis_evidence_id,
                basis_issuer_id,
                &mut reasons,
                &mut evidence_ids,
            );
        }
        RecoveryRoute::Repair => {
            let Some(condition) = &bundle.condition else {
                reasons.push(RouteReason::MissingConditionInspection);
                return assessment(route, reasons, evidence_ids);
            };
            evidence_ids.push(condition.evidence.evidence_id.clone());
            if !valid_evidence(
                &condition.evidence,
                EvidenceKind::ConditionInspection,
                EvidenceTier::HumanInspected,
            ) {
                reasons.push(RouteReason::MissingConditionInspection);
            }
            match condition.condition {
                ProductCondition::Repairable => {}
                ProductCondition::Unknown => reasons.push(RouteReason::ProductConditionUnknown),
                ProductCondition::Wearable | ProductCondition::NonWearable => {
                    reasons.push(RouteReason::ProductNotWearable)
                }
            }
            add_trial_checks(
                bundle.process_trial.as_ref(),
                EvidenceKind::RepairTrial,
                false,
                bundle.input_mass_mg,
                policy,
                &mut reasons,
                &mut evidence_ids,
            );
            let (basis_evidence_id, basis_issuer_id) = bundle
                .process_trial
                .as_ref()
                .map(|trial| {
                    (
                        trial.evidence.evidence_id.as_str(),
                        trial.evidence.issuer_id.as_str(),
                    )
                })
                .unwrap_or(("", ""));
            require_downstream_acceptance(
                bundle.downstream_acceptance.as_ref(),
                &bundle.lot_id,
                basis_evidence_id,
                basis_issuer_id,
                &mut reasons,
                &mut evidence_ids,
            );
        }
        RecoveryRoute::ComponentRecovery => {
            add_trial_checks(
                bundle.process_trial.as_ref(),
                EvidenceKind::ComponentRecoveryTrial,
                true,
                bundle.input_mass_mg,
                policy,
                &mut reasons,
                &mut evidence_ids,
            );
            let (output_lot_id, trial_evidence_id, trial_issuer_id) =
                trial_acceptance_binding(bundle);
            require_downstream_acceptance(
                bundle.downstream_acceptance.as_ref(),
                output_lot_id,
                trial_evidence_id,
                trial_issuer_id,
                &mut reasons,
                &mut evidence_ids,
            );
        }
        RecoveryRoute::MechanicalFibreRecycling => {
            add_composition_checks(
                bundle.composition.as_ref(),
                policy,
                &mut reasons,
                &mut evidence_ids,
            );
            add_trial_checks(
                bundle.process_trial.as_ref(),
                EvidenceKind::MechanicalRecyclingTrial,
                true,
                bundle.input_mass_mg,
                policy,
                &mut reasons,
                &mut evidence_ids,
            );
            match &bundle.fibre_length {
                None => reasons.push(RouteReason::MissingFibreLengthEvidence),
                Some(length) => {
                    evidence_ids.push(length.evidence.evidence_id.clone());
                    evidence_ids.push(length.sampling_plan_evidence.evidence_id.clone());
                    if !valid_evidence(
                        &length.evidence,
                        EvidenceKind::FibreLengthMeasurement,
                        EvidenceTier::SensorMeasured,
                    ) {
                        reasons.push(RouteReason::InvalidFibreLengthDistribution);
                    }
                    let protocol_valid = fibre_length_protocol_valid(length);
                    if !protocol_valid {
                        reasons.push(RouteReason::InvalidFibreLengthMeasurementProtocol);
                    }
                    if length.measurement_uncertainty_ppm
                        > policy.max_fibre_length_measurement_uncertainty_ppm
                    {
                        reasons.push(RouteReason::FibreLengthMeasurementUncertaintyAbovePolicy);
                    }
                    if !fibre_length_sampling_meets_policy(length, policy) {
                        reasons.push(RouteReason::FibreLengthSamplingBelowPolicy);
                    }
                    if !fibre_length_coverage_meets_policy(length, policy) {
                        reasons.push(RouteReason::FibreLengthCoverageBelowPolicy);
                    }
                    let sample_exceeds_output = bundle
                        .process_trial
                        .as_ref()
                        .and_then(|trial| trial.mass_balance.as_ref())
                        .is_some_and(|balance| {
                            length.sample_mass_mg > balance.accepted_output_mass_mg
                        });
                    if sample_exceeds_output {
                        reasons.push(RouteReason::FibreLengthSampleExceedsOutput);
                    }
                    let trial_binding_matches = bundle
                        .process_trial
                        .as_ref()
                        .is_some_and(|trial| fibre_length_binds_to_trial(length, trial));
                    if !trial_binding_matches && !sample_exceeds_output {
                        reasons.push(RouteReason::FibreLengthTrialBindingMismatch);
                    }
                    if protocol_valid {
                        match fibre_length_fraction_at_or_above(
                            length,
                            policy.minimum_fibre_length_um,
                        ) {
                            None => reasons.push(RouteReason::InvalidFibreLengthDistribution),
                            Some(fraction)
                                if fraction
                                    < policy.minimum_mass_fraction_at_or_above_length_ppm =>
                            {
                                reasons.push(RouteReason::FibreLengthBelowPolicy)
                            }
                            Some(_) => {}
                        }
                    }
                }
            }
            let (output_lot_id, trial_evidence_id, trial_issuer_id) =
                trial_acceptance_binding(bundle);
            require_downstream_acceptance(
                bundle.downstream_acceptance.as_ref(),
                output_lot_id,
                trial_evidence_id,
                trial_issuer_id,
                &mut reasons,
                &mut evidence_ids,
            );
        }
        RecoveryRoute::ChemicalFibreRecycling => {
            add_composition_checks(
                bundle.composition.as_ref(),
                policy,
                &mut reasons,
                &mut evidence_ids,
            );
            require_chemical_compatibility(
                bundle.chemical_compatibility.as_ref(),
                &mut reasons,
                &mut evidence_ids,
            );
            add_trial_checks(
                bundle.process_trial.as_ref(),
                EvidenceKind::ChemicalRecyclingTrial,
                true,
                bundle.input_mass_mg,
                policy,
                &mut reasons,
                &mut evidence_ids,
            );
            let (output_lot_id, trial_evidence_id, trial_issuer_id) =
                trial_acceptance_binding(bundle);
            require_downstream_acceptance(
                bundle.downstream_acceptance.as_ref(),
                output_lot_id,
                trial_evidence_id,
                trial_issuer_id,
                &mut reasons,
                &mut evidence_ids,
            );
        }
        RecoveryRoute::Downcycling => {
            add_trial_checks(
                bundle.process_trial.as_ref(),
                EvidenceKind::DowncyclingTrial,
                true,
                bundle.input_mass_mg,
                policy,
                &mut reasons,
                &mut evidence_ids,
            );
            let (output_lot_id, trial_evidence_id, trial_issuer_id) =
                trial_acceptance_binding(bundle);
            require_downstream_acceptance(
                bundle.downstream_acceptance.as_ref(),
                output_lot_id,
                trial_evidence_id,
                trial_issuer_id,
                &mut reasons,
                &mut evidence_ids,
            );
        }
    }

    assessment(route, reasons, evidence_ids)
}

fn assessment(
    route: RecoveryRoute,
    mut reasons: Vec<RouteReason>,
    evidence_ids: Vec<String>,
) -> RouteAssessment {
    let mut unique = BTreeSet::new();
    if evidence_ids
        .iter()
        .any(|evidence_id| !unique.insert(evidence_id.clone()))
    {
        reasons.push(RouteReason::DuplicateEvidenceIdentity);
    }
    let disposition = if reasons.is_empty() {
        RouteDisposition::Eligible
    } else if reasons.iter().any(|reason| {
        matches!(
            reason,
            RouteReason::ProductNotWearable
                | RouteReason::UseSafetyNotPassed
                | RouteReason::ProcessTrialNotPassed
                | RouteReason::MassBalanceNotClosed
                | RouteReason::MassBalanceInputMismatch
                | RouteReason::ChemicalCompatibilityFailed
                | RouteReason::DownstreamAcceptanceRejected
                | RouteReason::FibreLengthBelowPolicy
                | RouteReason::InvalidInventoryEvidence
                | RouteReason::InvalidProcessTrialEvidence
                | RouteReason::InvalidPhysicalTrialIdentity
                | RouteReason::InvalidDesignRevisionBinding
                | RouteReason::InvalidCompositionEvidence
                | RouteReason::InvalidFibreLengthDistribution
                | RouteReason::FibreLengthTrialBindingMismatch
                | RouteReason::FibreLengthSampleExceedsOutput
                | RouteReason::InvalidFibreLengthMeasurementProtocol
                | RouteReason::FibreLengthMeasurementUncertaintyAbovePolicy
                | RouteReason::InvalidChemicalCompatibilityEvidence
                | RouteReason::InvalidDownstreamAcceptance
                | RouteReason::DownstreamAcceptanceBindingMismatch
                | RouteReason::DownstreamAcceptanceNotIndependentOfBasis
                | RouteReason::InvalidPolicy
                | RouteReason::DuplicateEvidenceIdentity
                | RouteReason::InvalidPhysicalInventory
        )
    }) {
        RouteDisposition::Ineligible
    } else {
        RouteDisposition::InsufficientEvidence
    };

    RouteAssessment {
        route,
        disposition,
        reasons,
        evidence_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(
        id: &str,
        kind: EvidenceKind,
        tier: EvidenceTier,
        issuer: &str,
        verifier: Option<&str>,
    ) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.to_owned(),
            sha256: "a".repeat(64),
            kind,
            tier,
            method_ref: "METHOD-V1".to_owned(),
            issuer_id: issuer.to_owned(),
            verifier_id: verifier.map(str::to_owned),
            observed_at_ns: 100,
        }
    }

    fn policy() -> RecoveryPolicy {
        RecoveryPolicy {
            max_unknown_fraction_ppm: 20_000,
            max_mass_balance_uncertainty_ppm: 10_000,
            max_fibre_length_measurement_uncertainty_ppm: 50_000,
            minimum_fibre_length_sample_mass_mg: 20_000,
            minimum_fibre_length_replicates: 3,
            minimum_fibre_length_measurement_coverage_probability_ppm: 950_000,
            minimum_fibre_length_um: 12_700,
            minimum_mass_fraction_at_or_above_length_ppm: 500_000,
        }
    }

    fn inventory() -> EvidenceRef {
        evidence(
            "INV-01",
            EvidenceKind::PhysicalInventory,
            EvidenceTier::SensorMeasured,
            "scale-lab",
            None,
        )
    }

    fn acceptance_for(output_lot_id: &str, basis_evidence_id: &str) -> DownstreamAcceptance {
        DownstreamAcceptance {
            specification_id: "RECEIVING-SPEC-V1".to_owned(),
            accepted_output_lot_id: output_lot_id.to_owned(),
            basis_evidence_id: basis_evidence_id.to_owned(),
            outcome: AcceptanceOutcome::Accepted,
            evidence: evidence(
                "ACCEPT-01",
                EvidenceKind::DownstreamAcceptance,
                EvidenceTier::IndependentAcceptance,
                "receiving-mill",
                Some("independent-auditor"),
            ),
        }
    }

    fn acceptance() -> DownstreamAcceptance {
        acceptance_for("ACCEPTED-FIBRE-LOT-01", "TRIAL-01")
    }

    fn item_acceptance() -> DownstreamAcceptance {
        acceptance_for("LOT-01", "COND-01")
    }

    fn repair_acceptance() -> DownstreamAcceptance {
        acceptance_for("LOT-01", "TRIAL-01")
    }

    fn condition(state: ProductCondition, safety: UseSafetyStatus) -> ConditionEvidence {
        ConditionEvidence {
            condition: state,
            safety,
            evidence: evidence(
                "COND-01",
                EvidenceKind::ConditionInspection,
                EvidenceTier::HumanInspected,
                "inspector-01",
                None,
            ),
        }
    }

    fn composition() -> CompositionEvidence {
        CompositionEvidence {
            disposition: CompositionDisposition::Admitted,
            unknown_fraction_upper_ppm: 10_000,
            measurement: evidence(
                "COMP-01",
                EvidenceKind::CompositionMeasurement,
                EvidenceTier::SensorMeasured,
                "nir-lab",
                None,
            ),
            calibration: evidence(
                "CAL-01",
                EvidenceKind::CalibrationRecord,
                EvidenceTier::SensorMeasured,
                "metrology-lab",
                None,
            ),
        }
    }

    fn balance() -> MassBalance {
        MassBalance {
            input_mass_mg: 1_000_000,
            accepted_output_lot_id: "ACCEPTED-FIBRE-LOT-01".to_owned(),
            accepted_output_mass_mg: 800_000,
            recovered_component_mass_mg: 50_000,
            recovered_trim_mass_mg: 50_000,
            rejected_mass_mg: 50_000,
            waste_mass_mg: 50_000,
            uncertainty_mass_mg: 2_000,
        }
    }

    fn process_trial(kind: EvidenceKind, with_balance: bool) -> ProcessTrialEvidence {
        ProcessTrialEvidence {
            outcome: TrialOutcome::Passed,
            physical_sample_id: "PHYSICAL-ITEM-01".to_owned(),
            evidence: evidence(
                "TRIAL-01",
                kind,
                EvidenceTier::PhysicalTrial,
                "process-lab",
                None,
            ),
            design_binding: None,
            mass_balance: with_balance.then(balance),
        }
    }

    fn length() -> FibreLengthEvidence {
        let method_ref = "FIBRE-ARRAY-ASTM-D1440-V1".to_owned();
        let mut measurement = evidence(
            "LENGTH-01",
            EvidenceKind::FibreLengthMeasurement,
            EvidenceTier::SensorMeasured,
            "fibre-lab",
            None,
        );
        measurement.method_ref = method_ref.clone();
        FibreLengthEvidence {
            method_ref,
            sampling_plan_evidence: evidence(
                "FIBRE-SAMPLING-PLAN-01",
                EvidenceKind::FibreLengthSamplingPlan,
                EvidenceTier::HumanInspected,
                "fibre-methods-lab",
                None,
            ),
            sample_mass_mg: 25_000,
            replicate_count: 3,
            measurement_uncertainty_ppm: 20_000,
            measurement_coverage_probability_ppm: 950_000,
            accepted_output_lot_id: "ACCEPTED-FIBRE-LOT-01".to_owned(),
            source_trial_evidence_id: "TRIAL-01".to_owned(),
            evidence: measurement,
            bins: vec![
                FibreLengthBin {
                    lower_bound_um: 0,
                    upper_bound_um: 13_000,
                    mass_fraction_ppm: 300_000,
                },
                FibreLengthBin {
                    lower_bound_um: 13_000,
                    upper_bound_um: 25_400,
                    mass_fraction_ppm: 500_000,
                },
                FibreLengthBin {
                    lower_bound_um: 25_400,
                    upper_bound_um: u32::MAX,
                    mass_fraction_ppm: 200_000,
                },
            ],
        }
    }

    fn empty_bundle() -> RecoveryEvidenceBundle {
        RecoveryEvidenceBundle {
            lot_id: "LOT-01".to_owned(),
            input_mass_mg: 1_000_000,
            inventory: inventory(),
            condition: None,
            composition: None,
            process_trial: None,
            fibre_length: None,
            chemical_compatibility: None,
            downstream_acceptance: None,
        }
    }

    #[test]
    fn reuse_requires_physical_condition_safety_and_independent_item_acceptance() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Wearable,
            UseSafetyStatus::Passed,
        ));
        bundle.downstream_acceptance = Some(item_acceptance());
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Eligible);
    }

    #[test]
    fn visual_inference_cannot_substitute_for_human_condition_inspection() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(ConditionEvidence {
            condition: ProductCondition::Wearable,
            safety: UseSafetyStatus::Passed,
            evidence: evidence(
                "VIS-01",
                EvidenceKind::ConditionInspection,
                EvidenceTier::ModelInference,
                "vision-model",
                None,
            ),
        });
        bundle.downstream_acceptance = Some(acceptance_for("LOT-01", "VIS-01"));
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::InsufficientEvidence);
        assert!(
            result
                .reasons
                .contains(&RouteReason::MissingConditionInspection)
        );
    }

    #[test]
    fn failed_use_safety_blocks_reuse() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Wearable,
            UseSafetyStatus::Failed,
        ));
        bundle.downstream_acceptance = Some(item_acceptance());
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(result.reasons.contains(&RouteReason::UseSafetyNotPassed));
    }

    #[test]
    fn repair_requires_repairable_condition_physical_trial_and_repair_acceptance() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Repairable,
            UseSafetyStatus::Passed,
        ));
        bundle.process_trial = Some(process_trial(EvidenceKind::RepairTrial, false));
        bundle.downstream_acceptance = Some(repair_acceptance());
        assert_eq!(
            qualify_recovery_route(RecoveryRoute::Repair, &bundle, &policy()).disposition,
            RouteDisposition::Eligible
        );
    }

    #[test]
    fn receiving_acceptance_issuer_must_differ_from_process_issuer() {
        let mut bundle = empty_bundle();
        bundle.process_trial = Some(process_trial(EvidenceKind::ComponentRecoveryTrial, true));
        let mut accepted = acceptance();
        accepted.evidence.issuer_id = "process-lab".to_owned();
        bundle.downstream_acceptance = Some(accepted);

        let result = qualify_recovery_route(RecoveryRoute::ComponentRecovery, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::DownstreamAcceptanceNotIndependentOfBasis)
        );
    }

    #[test]
    fn receiving_verifier_must_differ_from_process_issuer_too() {
        let mut bundle = empty_bundle();
        bundle.process_trial = Some(process_trial(EvidenceKind::ComponentRecoveryTrial, true));
        let mut accepted = acceptance();
        accepted.evidence.verifier_id = Some("process-lab".to_owned());
        bundle.downstream_acceptance = Some(accepted);

        let result = qualify_recovery_route(RecoveryRoute::ComponentRecovery, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::DownstreamAcceptanceNotIndependentOfBasis)
        );
    }

    #[test]
    fn receiving_acceptance_cannot_be_borrowed_from_another_output_lot() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        bundle.fibre_length = Some(length());
        let mut accepted = acceptance();
        accepted.accepted_output_lot_id = "UNRELATED-OUTPUT-LOT".to_owned();
        bundle.downstream_acceptance = Some(accepted);

        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::DownstreamAcceptanceBindingMismatch)
        );
    }

    #[test]
    fn receiving_acceptance_must_reference_the_trial_that_produced_the_output() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        bundle.fibre_length = Some(length());
        let mut accepted = acceptance();
        accepted.basis_evidence_id = "OTHER-TRIAL".to_owned();
        bundle.downstream_acceptance = Some(accepted);

        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::DownstreamAcceptanceBindingMismatch)
        );
    }

    #[test]
    fn component_recovery_requires_mass_closure_and_acceptance() {
        let mut bundle = empty_bundle();
        bundle.process_trial = Some(process_trial(EvidenceKind::ComponentRecoveryTrial, true));
        bundle.downstream_acceptance = Some(acceptance());
        assert_eq!(
            qualify_recovery_route(RecoveryRoute::ComponentRecovery, &bundle, &policy())
                .disposition,
            RouteDisposition::Eligible
        );
    }

    #[test]
    fn mechanical_recycling_requires_composition_fibre_length_trial_and_acceptance() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        bundle.fibre_length = Some(length());
        bundle.downstream_acceptance = Some(acceptance());
        assert_eq!(
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy())
                .disposition,
            RouteDisposition::Eligible
        );
    }

    #[test]
    fn fibre_length_evidence_must_bind_to_the_exact_physical_trial() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        bundle.fibre_length = Some(length());
        bundle.downstream_acceptance = Some(acceptance());

        bundle
            .fibre_length
            .as_mut()
            .unwrap()
            .source_trial_evidence_id = "TRIAL-OTHER".to_owned();

        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::FibreLengthTrialBindingMismatch)
        );
    }

    #[test]
    fn fibre_length_evidence_must_bind_to_the_mass_balanced_accepted_output_lot() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        bundle.fibre_length = Some(length());
        bundle.downstream_acceptance = Some(acceptance());

        bundle.fibre_length.as_mut().unwrap().accepted_output_lot_id =
            "UNRELATED-FIBRE-LOT".to_owned();

        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::FibreLengthTrialBindingMismatch)
        );
    }

    #[test]
    fn fibre_length_protocol_requires_sampling_plan_sample_mass_and_replicates() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        let mut distribution = length();
        distribution.replicate_count = 0;
        distribution.sampling_plan_evidence.evidence_id.clear();
        bundle.fibre_length = Some(distribution);
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidFibreLengthMeasurementProtocol)
        );
    }

    #[test]
    fn fibre_sampling_plan_and_measurement_cannot_reuse_evidence_identity() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        let mut distribution = length();
        distribution.sampling_plan_evidence.evidence_id =
            distribution.evidence.evidence_id.clone();
        bundle.fibre_length = Some(distribution);
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(result.reasons.contains(&RouteReason::DuplicateEvidenceIdentity));
    }

    #[test]
    fn fibre_measurement_uncertainty_must_declare_coverage_probability() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        let mut distribution = length();
        distribution.measurement_coverage_probability_ppm = 0;
        bundle.fibre_length = Some(distribution);
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidFibreLengthMeasurementProtocol)
        );
    }

    #[test]
    fn fibre_sampling_plan_must_bind_a_valid_content_digest() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        let mut distribution = length();
        distribution.sampling_plan_evidence.sha256 = "not-a-digest".to_owned();
        bundle.fibre_length = Some(distribution);
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidFibreLengthMeasurementProtocol)
        );
    }

    #[test]
    fn fibre_length_sample_mass_cannot_exceed_accepted_output_lot() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        let mut distribution = length();
        distribution.sample_mass_mg = 900_000;
        bundle.fibre_length = Some(distribution);
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::FibreLengthSampleExceedsOutput)
        );
    }

    #[test]
    fn fibre_length_measurement_uncertainty_must_meet_receiving_policy() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        let mut distribution = length();
        distribution.measurement_uncertainty_ppm = 60_000;
        bundle.fibre_length = Some(distribution);
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::FibreLengthMeasurementUncertaintyAbovePolicy)
        );
    }

    #[test]
    fn composition_prediction_without_sensor_evidence_cannot_authorize_recycling() {
        let mut bundle = empty_bundle();
        let mut composition = composition();
        composition.measurement.tier = EvidenceTier::ModelInference;
        bundle.composition = Some(composition);
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        bundle.fibre_length = Some(length());
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidCompositionEvidence)
        );
    }

    #[test]
    fn unresolved_composition_above_policy_abstains() {
        let mut bundle = empty_bundle();
        let mut composition = composition();
        composition.unknown_fraction_upper_ppm = 50_000;
        bundle.composition = Some(composition);
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        bundle.fibre_length = Some(length());
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::InsufficientEvidence);
        assert!(
            result
                .reasons
                .contains(&RouteReason::UnknownFibreFractionAbovePolicy)
        );
    }

    #[test]
    fn missing_fibre_length_blocks_mechanical_route() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert!(
            result
                .reasons
                .contains(&RouteReason::MissingFibreLengthEvidence)
        );
    }

    #[test]
    fn short_fibre_distribution_fails_receiving_policy() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        let mut distribution = length();
        distribution.bins = vec![
            FibreLengthBin {
                lower_bound_um: 0,
                upper_bound_um: 12_700,
                mass_fraction_ppm: 700_000,
            },
            FibreLengthBin {
                lower_bound_um: 12_700,
                upper_bound_um: 25_400,
                mass_fraction_ppm: 300_000,
            },
        ];
        bundle.fibre_length = Some(distribution);
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::FibreLengthBelowPolicy)
        );
    }

    #[test]
    fn malformed_fibre_bins_do_not_close() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::MechanicalRecyclingTrial, true));
        let mut distribution = length();
        distribution.bins[1].mass_fraction_ppm = 400_000;
        bundle.fibre_length = Some(distribution);
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidFibreLengthDistribution)
        );
    }

    #[test]
    fn chemical_route_requires_compatibility_trial_and_downstream_acceptance() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.chemical_compatibility = Some(ChemicalCompatibilityEvidence {
            outcome: TrialOutcome::Passed,
            evidence: evidence(
                "CHEM-COMPAT-01",
                EvidenceKind::ChemicalCompatibilityTrial,
                EvidenceTier::PhysicalTrial,
                "chemical-lab",
                None,
            ),
        });
        bundle.process_trial = Some(process_trial(EvidenceKind::ChemicalRecyclingTrial, true));
        bundle.downstream_acceptance = Some(acceptance());
        assert_eq!(
            qualify_recovery_route(RecoveryRoute::ChemicalFibreRecycling, &bundle, &policy())
                .disposition,
            RouteDisposition::Eligible
        );
    }

    #[test]
    fn missing_chemical_compatibility_is_not_inferred_from_fibre_identity() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.process_trial = Some(process_trial(EvidenceKind::ChemicalRecyclingTrial, true));
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::ChemicalFibreRecycling, &bundle, &policy());
        assert!(
            result
                .reasons
                .contains(&RouteReason::MissingChemicalCompatibilityEvidence)
        );
    }

    #[test]
    fn unclosed_mass_balance_blocks_destructive_route() {
        let mut bundle = empty_bundle();
        let mut trial = process_trial(EvidenceKind::MechanicalRecyclingTrial, true);
        trial.mass_balance.as_mut().unwrap().waste_mass_mg = 20_000;
        bundle.composition = Some(composition());
        bundle.process_trial = Some(trial);
        bundle.fibre_length = Some(length());
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(result.reasons.contains(&RouteReason::MassBalanceNotClosed));
    }

    #[test]
    fn acceptance_from_same_issuer_is_not_independent() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Wearable,
            UseSafetyStatus::Passed,
        ));
        let mut accepted = item_acceptance();
        accepted.evidence.evidence_id = "ACCEPT-SELF".to_owned();
        accepted.evidence.verifier_id = Some(accepted.evidence.issuer_id.clone());
        bundle.downstream_acceptance = Some(accepted);
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidDownstreamAcceptance)
        );
    }

    #[test]
    fn synthetic_trial_cannot_be_promoted_to_physical_trial() {
        let mut bundle = empty_bundle();
        let mut trial = process_trial(EvidenceKind::MechanicalRecyclingTrial, true);
        trial.evidence.tier = EvidenceTier::Synthetic;
        bundle.composition = Some(composition());
        bundle.process_trial = Some(trial);
        bundle.fibre_length = Some(length());
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidProcessTrialEvidence)
        );
    }

    #[test]
    fn corrupted_evidence_digest_is_rejected() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Wearable,
            UseSafetyStatus::Passed,
        ));
        bundle.downstream_acceptance = Some(item_acceptance());
        bundle
            .downstream_acceptance
            .as_mut()
            .unwrap()
            .evidence
            .sha256 = "not-a-digest".to_owned();
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidDownstreamAcceptance)
        );
    }

    #[test]
    fn rejected_downstream_output_is_ineligible() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Wearable,
            UseSafetyStatus::Passed,
        ));
        let mut accepted = item_acceptance();
        accepted.outcome = AcceptanceOutcome::Rejected;
        bundle.downstream_acceptance = Some(accepted);
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::DownstreamAcceptanceRejected)
        );
    }

    #[test]
    fn conditional_or_unknown_acceptance_remains_insufficient() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Wearable,
            UseSafetyStatus::Passed,
        ));
        let mut accepted = item_acceptance();
        accepted.outcome = AcceptanceOutcome::Conditional;
        bundle.downstream_acceptance = Some(accepted);
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::InsufficientEvidence);
        assert!(
            result
                .reasons
                .contains(&RouteReason::DownstreamAcceptanceUnresolved)
        );
    }

    #[test]
    fn unknown_process_trial_is_insufficient_not_a_false_failure() {
        let mut bundle = empty_bundle();
        let mut trial = process_trial(EvidenceKind::MechanicalRecyclingTrial, true);
        trial.outcome = TrialOutcome::Unknown;
        bundle.composition = Some(composition());
        bundle.process_trial = Some(trial);
        bundle.fibre_length = Some(length());
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::InsufficientEvidence);
        assert!(result.reasons.contains(&RouteReason::ProcessTrialUnknown));
    }

    #[test]
    fn process_trial_mass_must_match_physical_lot_mass() {
        let mut bundle = empty_bundle();
        let mut trial = process_trial(EvidenceKind::MechanicalRecyclingTrial, true);
        trial.mass_balance.as_mut().unwrap().input_mass_mg = 900_000;
        bundle.composition = Some(composition());
        bundle.process_trial = Some(trial);
        bundle.fibre_length = Some(length());
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::MechanicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::MassBalanceInputMismatch)
        );
    }

    #[test]
    fn chemical_compatibility_failure_blocks_chemical_route() {
        let mut bundle = empty_bundle();
        bundle.composition = Some(composition());
        bundle.chemical_compatibility = Some(ChemicalCompatibilityEvidence {
            outcome: TrialOutcome::Failed,
            evidence: evidence(
                "CHEM-COMPAT-FAIL",
                EvidenceKind::ChemicalCompatibilityTrial,
                EvidenceTier::PhysicalTrial,
                "chemical-lab",
                None,
            ),
        });
        bundle.process_trial = Some(process_trial(EvidenceKind::ChemicalRecyclingTrial, true));
        bundle.downstream_acceptance = Some(acceptance());
        let result =
            qualify_recovery_route(RecoveryRoute::ChemicalFibreRecycling, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::ChemicalCompatibilityFailed)
        );
    }

    #[test]
    fn evidence_ids_cannot_be_reused_across_independent_claims() {
        let mut bundle = empty_bundle();
        let mut accepted = item_acceptance();
        accepted.evidence.evidence_id = "INV-01".to_owned();
        bundle.condition = Some(condition(
            ProductCondition::Wearable,
            UseSafetyStatus::Passed,
        ));
        bundle.downstream_acceptance = Some(accepted);
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::DuplicateEvidenceIdentity)
        );
    }

    fn exercise_route_case(case_id: &str) -> (&'static str, Option<&'static str>) {
        match case_id {
            "ROUTE-001-REUSE-ACCEPTED" => {
                reuse_requires_physical_condition_safety_and_independent_item_acceptance();
                ("eligible", None)
            }
            "ROUTE-002-VISUAL-INFERENCE-ONLY" => {
                visual_inference_cannot_substitute_for_human_condition_inspection();
                ("insufficient_evidence", Some("MissingConditionInspection"))
            }
            "ROUTE-003-REUSE-SAFETY-FAIL" => {
                failed_use_safety_blocks_reuse();
                ("ineligible", Some("UseSafetyNotPassed"))
            }
            "ROUTE-004-REPAIR-ACCEPTED" => {
                repair_requires_repairable_condition_physical_trial_and_repair_acceptance();
                ("eligible", None)
            }
            "ROUTE-005-COMPONENT-RECOVERY" => {
                component_recovery_requires_mass_closure_and_acceptance();
                ("eligible", None)
            }
            "ROUTE-006-MECHANICAL-ACCEPTED" => {
                mechanical_recycling_requires_composition_fibre_length_trial_and_acceptance();
                ("eligible", None)
            }
            "ROUTE-007-MODEL-COMPOSITION" => {
                composition_prediction_without_sensor_evidence_cannot_authorize_recycling();
                ("ineligible", Some("InvalidCompositionEvidence"))
            }
            "ROUTE-008-UNKNOWN-FIBRE-FRACTION" => {
                unresolved_composition_above_policy_abstains();
                (
                    "insufficient_evidence",
                    Some("UnknownFibreFractionAbovePolicy"),
                )
            }
            "ROUTE-009-MISSING-FIBRE-LENGTH" => {
                missing_fibre_length_blocks_mechanical_route();
                ("insufficient_evidence", Some("MissingFibreLengthEvidence"))
            }
            "ROUTE-010-SHORT-FIBRE-OUTPUT" => {
                short_fibre_distribution_fails_receiving_policy();
                ("ineligible", Some("FibreLengthBelowPolicy"))
            }
            "ROUTE-011-MALFORMED-FIBRE-BINS" => {
                malformed_fibre_bins_do_not_close();
                ("ineligible", Some("InvalidFibreLengthDistribution"))
            }
            "ROUTE-012-CHEMICAL-ACCEPTED" => {
                chemical_route_requires_compatibility_trial_and_downstream_acceptance();
                ("eligible", None)
            }
            "ROUTE-013-CHEMICAL-COMPATIBILITY-MISSING" => {
                missing_chemical_compatibility_is_not_inferred_from_fibre_identity();
                (
                    "insufficient_evidence",
                    Some("MissingChemicalCompatibilityEvidence"),
                )
            }
            "ROUTE-014-MASS-NOT-CLOSED" => {
                unclosed_mass_balance_blocks_destructive_route();
                ("ineligible", Some("MassBalanceNotClosed"))
            }
            "ROUTE-015-SELF-ISSUED-ACCEPTANCE" => {
                acceptance_from_same_issuer_is_not_independent();
                ("ineligible", Some("InvalidDownstreamAcceptance"))
            }
            "ROUTE-016-SYNTHETIC-PHYSICAL-TRIAL" => {
                synthetic_trial_cannot_be_promoted_to_physical_trial();
                ("ineligible", Some("InvalidProcessTrialEvidence"))
            }
            "ROUTE-017-CORRUPTED-DIGEST" => {
                corrupted_evidence_digest_is_rejected();
                ("ineligible", Some("InvalidDownstreamAcceptance"))
            }
            "ROUTE-018-RECEIVER-REJECTED" => {
                rejected_downstream_output_is_ineligible();
                ("ineligible", Some("DownstreamAcceptanceRejected"))
            }
            "ROUTE-019-RECEIVER-CONDITIONAL" => {
                conditional_or_unknown_acceptance_remains_insufficient();
                (
                    "insufficient_evidence",
                    Some("DownstreamAcceptanceUnresolved"),
                )
            }
            "ROUTE-020-TRIAL-UNKNOWN" => {
                unknown_process_trial_is_insufficient_not_a_false_failure();
                ("insufficient_evidence", Some("ProcessTrialUnknown"))
            }
            "ROUTE-021-TRIAL-INPUT-MISMATCH" => {
                process_trial_mass_must_match_physical_lot_mass();
                ("ineligible", Some("MassBalanceInputMismatch"))
            }
            "ROUTE-022-CHEMICAL-COMPATIBILITY-FAIL" => {
                chemical_compatibility_failure_blocks_chemical_route();
                ("ineligible", Some("ChemicalCompatibilityFailed"))
            }
            "ROUTE-023-EVIDENCE-ID-DUPLICATE" => {
                evidence_ids_cannot_be_reused_across_independent_claims();
                ("ineligible", Some("DuplicateEvidenceIdentity"))
            }
            "ROUTE-024-POLICY-INVALID" => {
                invalid_policy_cannot_promote_route();
                ("ineligible", Some("InvalidPolicy"))
            }
            "ROUTE-025-CONDITION-UNKNOWN" => {
                unknown_product_condition_remains_insufficient();
                ("insufficient_evidence", Some("ProductConditionUnknown"))
            }
            "ROUTE-026-USE-SAFETY-UNKNOWN" => {
                unknown_use_safety_remains_insufficient();
                ("insufficient_evidence", Some("UseSafetyUnknown"))
            }
            "ROUTE-027-DOWNCYCLING-ACCEPTED" => {
                downcycling_requires_physical_trial_mass_balance_and_receiver_acceptance();
                ("eligible", None)
            }
            "ROUTE-028-DESIGN-REVISION-BOUND" => {
                design_linked_component_trial_binds_exact_revision_and_specimen();
                ("eligible", None)
            }
            "ROUTE-029-DESIGN-DOSSIER-DIGEST-INVALID" => {
                malformed_design_revision_binding_blocks_component_claim();
                ("ineligible", Some("InvalidDesignRevisionBinding"))
            }
            "ROUTE-030-DESIGN-SPECIMEN-MISMATCH" => {
                design_conformity_must_bind_the_same_physical_sample_as_the_trial();
                ("ineligible", Some("InvalidDesignRevisionBinding"))
            }
            "ROUTE-031-FIBRE-LENGTH-TRIAL-MISMATCH" => {
                fibre_length_evidence_must_bind_to_the_exact_physical_trial();
                ("ineligible", Some("FibreLengthTrialBindingMismatch"))
            }
            "ROUTE-032-FIBRE-LENGTH-OUTPUT-LOT-MISMATCH" => {
                fibre_length_evidence_must_bind_to_the_mass_balanced_accepted_output_lot();
                ("ineligible", Some("FibreLengthTrialBindingMismatch"))
            }
            "ROUTE-033-RECEIVER-OUTPUT-LOT-MISMATCH" => {
                receiving_acceptance_cannot_be_borrowed_from_another_output_lot();
                ("ineligible", Some("DownstreamAcceptanceBindingMismatch"))
            }
            "ROUTE-034-RECEIVER-BASIS-TRIAL-MISMATCH" => {
                receiving_acceptance_must_reference_the_trial_that_produced_the_output();
                ("ineligible", Some("DownstreamAcceptanceBindingMismatch"))
            }
            "ROUTE-035-RECEIVER-ISSUER-IS-PROCESS-ISSUER" => {
                receiving_acceptance_issuer_must_differ_from_process_issuer();
                (
                    "ineligible",
                    Some("DownstreamAcceptanceNotIndependentOfBasis"),
                )
            }
            "ROUTE-036-RECEIVER-VERIFIER-IS-PROCESS-ISSUER" => {
                receiving_verifier_must_differ_from_process_issuer_too();
                (
                    "ineligible",
                    Some("DownstreamAcceptanceNotIndependentOfBasis"),
                )
            }
            "ROUTE-037-FIBRE-LENGTH-PROTOCOL-INCOMPLETE" => {
                fibre_length_protocol_requires_sampling_plan_sample_mass_and_replicates();
                ("ineligible", Some("InvalidFibreLengthMeasurementProtocol"))
            }
            "ROUTE-038-FIBRE-LENGTH-SAMPLE-EXCEEDS-OUTPUT" => {
                fibre_length_sample_mass_cannot_exceed_accepted_output_lot();
                ("ineligible", Some("FibreLengthSampleExceedsOutput"))
            }
            "ROUTE-039-FIBRE-LENGTH-UNCERTAINTY-ABOVE-POLICY" => {
                fibre_length_measurement_uncertainty_must_meet_receiving_policy();
                ("ineligible", Some("FibreLengthMeasurementUncertaintyAbovePolicy"))
            }
            "ROUTE-040-FIBRE-SAMPLING-PLAN-DIGEST-INVALID" => {
                fibre_sampling_plan_must_bind_a_valid_content_digest();
                ("ineligible", Some("InvalidFibreLengthMeasurementProtocol"))
            }
            "ROUTE-041-FIBRE-UNCERTAINTY-COVERAGE-MISSING" => {
                fibre_measurement_uncertainty_must_declare_coverage_probability();
                ("ineligible", Some("InvalidFibreLengthMeasurementProtocol"))
            }
            "ROUTE-042-SAMPLING-PLAN-MEASUREMENT-EVIDENCE-REUSED" => {
                fibre_sampling_plan_and_measurement_cannot_reuse_evidence_identity();
                ("ineligible", Some("DuplicateEvidenceIdentity"))
            }
            other => panic!("unknown textile recovery corpus case: {other}"),
        }
    }

    #[test]
    fn synthetic_recovery_corpus_executes_all_route_regressions() {
        const CORPUS: &str = include_str!(
            "../../../../docs/research/fixtures/textile_sort_fusion_v0_1/recovery_cases.example.json"
        );
        const SCHEMA: &str = include_str!(
            "../../../../docs/research/fixtures/textile_sort_fusion_v0_1/recovery_cases.schema.json"
        );

        let corpus: serde_json::Value =
            serde_json::from_str(CORPUS).expect("recovery corpus must parse as JSON");
        let schema: serde_json::Value =
            serde_json::from_str(SCHEMA).expect("recovery schema must parse as JSON");
        assert_eq!(schema["properties"]["schema_version"]["const"], "0.1.0");
        assert_eq!(corpus["schema_version"], "0.1.0");
        assert_eq!(corpus["corpus_id"], "TEXTILE-ROUTE-CASES-001");
        assert_eq!(corpus["fixture_class"], "synthetic");
        assert_eq!(schema["properties"]["cases"]["minItems"].as_u64(), Some(42));

        let cases = corpus["cases"]
            .as_array()
            .expect("recovery cases must be an array");
        assert_eq!(cases.len(), 42, "review route corpus case count changes");
        let mut seen = BTreeSet::new();
        for case in cases {
            let case_id = case["case_id"].as_str().expect("case_id must be a string");
            assert!(seen.insert(case_id), "duplicate recovery case {case_id}");
            assert!(
                case["mutation"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()),
                "case {case_id} must describe its mutation"
            );
            let expected = &case["expected"];
            let expected_disposition = expected["disposition"]
                .as_str()
                .expect("expected disposition must be a string");
            let expected_reason = expected["reason"].as_str();
            let (disposition, reason) = exercise_route_case(case_id);
            assert_eq!(
                disposition, expected_disposition,
                "case {case_id} disposition"
            );
            assert_eq!(reason, expected_reason, "case {case_id} reason");
        }
        assert_eq!(seen.len(), 42);
    }

    #[test]
    fn unknown_product_condition_remains_insufficient() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Unknown,
            UseSafetyStatus::Passed,
        ));
        bundle.downstream_acceptance = Some(item_acceptance());
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::InsufficientEvidence);
        assert!(
            result
                .reasons
                .contains(&RouteReason::ProductConditionUnknown)
        );
    }

    #[test]
    fn unknown_use_safety_remains_insufficient() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Wearable,
            UseSafetyStatus::Unknown,
        ));
        bundle.downstream_acceptance = Some(item_acceptance());
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::InsufficientEvidence);
        assert!(result.reasons.contains(&RouteReason::UseSafetyUnknown));
    }

    #[test]
    fn downcycling_requires_physical_trial_mass_balance_and_receiver_acceptance() {
        let mut bundle = empty_bundle();
        bundle.process_trial = Some(process_trial(EvidenceKind::DowncyclingTrial, true));
        bundle.downstream_acceptance = Some(acceptance());
        assert_eq!(
            qualify_recovery_route(RecoveryRoute::Downcycling, &bundle, &policy()).disposition,
            RouteDisposition::Eligible
        );
    }

    #[test]
    fn design_linked_component_trial_binds_exact_revision_and_specimen() {
        let mut bundle = empty_bundle();
        let mut trial = process_trial(EvidenceKind::ComponentRecoveryTrial, true);
        trial.design_binding = Some(DesignRevisionBinding {
            product_id: "OVERSHIRT-01".to_owned(),
            design_revision: "REV-A".to_owned(),
            dossier_sha256: "c".repeat(64),
            specimen_id: "PHYSICAL-ITEM-01".to_owned(),
            conformity_evidence: evidence(
                "DESIGN-CONFORMITY-01",
                EvidenceKind::DesignRevisionConformity,
                EvidenceTier::PhysicalTrial,
                "inspection-lab",
                None,
            ),
        });
        bundle.process_trial = Some(trial);
        bundle.downstream_acceptance = Some(acceptance());
        assert_eq!(
            qualify_recovery_route(RecoveryRoute::ComponentRecovery, &bundle, &policy())
                .disposition,
            RouteDisposition::Eligible
        );
    }

    #[test]
    fn malformed_design_revision_binding_blocks_component_claim() {
        let mut bundle = empty_bundle();
        let mut trial = process_trial(EvidenceKind::ComponentRecoveryTrial, true);
        trial.design_binding = Some(DesignRevisionBinding {
            product_id: "OVERSHIRT-01".to_owned(),
            design_revision: "REV-A".to_owned(),
            dossier_sha256: "not-a-digest".to_owned(),
            specimen_id: "PHYSICAL-ITEM-01".to_owned(),
            conformity_evidence: evidence(
                "DESIGN-CONFORMITY-02",
                EvidenceKind::DesignRevisionConformity,
                EvidenceTier::PhysicalTrial,
                "inspection-lab",
                None,
            ),
        });
        bundle.process_trial = Some(trial);
        bundle.downstream_acceptance = Some(acceptance());
        let result = qualify_recovery_route(RecoveryRoute::ComponentRecovery, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidDesignRevisionBinding)
        );
    }

    #[test]
    fn design_conformity_must_bind_the_same_physical_sample_as_the_trial() {
        let mut bundle = empty_bundle();
        let mut trial = process_trial(EvidenceKind::ComponentRecoveryTrial, true);
        trial.design_binding = Some(DesignRevisionBinding {
            product_id: "OVERSHIRT-01".to_owned(),
            design_revision: "REV-A".to_owned(),
            dossier_sha256: "c".repeat(64),
            specimen_id: "DIFFERENT-PHYSICAL-ITEM".to_owned(),
            conformity_evidence: evidence(
                "DESIGN-CONFORMITY-03",
                EvidenceKind::DesignRevisionConformity,
                EvidenceTier::PhysicalTrial,
                "inspection-lab",
                None,
            ),
        });
        bundle.process_trial = Some(trial);
        bundle.downstream_acceptance = Some(acceptance());
        let result = qualify_recovery_route(RecoveryRoute::ComponentRecovery, &bundle, &policy());
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(
            result
                .reasons
                .contains(&RouteReason::InvalidDesignRevisionBinding)
        );
    }

    #[test]
    fn invalid_policy_cannot_promote_route() {
        let mut bundle = empty_bundle();
        bundle.condition = Some(condition(
            ProductCondition::Wearable,
            UseSafetyStatus::Passed,
        ));
        bundle.downstream_acceptance = Some(item_acceptance());
        let mut invalid = policy();
        invalid.max_unknown_fraction_ppm = 2_000_000;
        let result = qualify_recovery_route(RecoveryRoute::ReuseAsIs, &bundle, &invalid);
        assert_eq!(result.disposition, RouteDisposition::Ineligible);
        assert!(result.reasons.contains(&RouteReason::InvalidPolicy));
    }
}
