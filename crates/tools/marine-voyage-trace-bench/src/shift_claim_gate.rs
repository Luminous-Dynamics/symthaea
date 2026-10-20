//! Fail-closed admissibility gate for marine prediction claims under distribution shift.
//!
//! This module does not estimate a shift correction or recompute calibration metrics. It consumes
//! the separate held-out interval-calibration report and checks whether the assumptions needed
//! for a *declared* shift-specific claim have evidence. A passing result is not a universal
//! guarantee, a physical validation, or an operational release authorization.

use crate::prediction_interval_calibration::{
    CalibrationDisposition, CalibrationReport,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShiftFamily {
    NoShiftControl,
    CovariateShift,
    Heteroscedasticity,
    LocalUndercoverage,
    JointCovariateOutcomeShift,
    SupportGap,
    TemporalBiasOrDrift,
    DependentSamples,
    SparseOrMissing,
}

impl ShiftFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::NoShiftControl => "no_shift_control",
            Self::CovariateShift => "covariate_shift",
            Self::Heteroscedasticity => "heteroscedasticity",
            Self::LocalUndercoverage => "local_undercoverage",
            Self::JointCovariateOutcomeShift => "joint_covariate_outcome_shift",
            Self::SupportGap => "support_gap",
            Self::TemporalBiasOrDrift => "temporal_bias_or_drift",
            Self::DependentSamples => "dependent_samples",
            Self::SparseOrMissing => "sparse_or_missing",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShiftCorrection {
    /// Unmodified reference arm. Required for the no-shift control.
    UnmodifiedBaseline,
    /// A covariate-shift correction; admissible only with reviewed assumptions and support checks.
    ImportanceWeighted,
    /// Separate calibration by a prespecified operating group.
    GroupConditional,
    /// Target-label-driven recalibration on a disjoint labelled set.
    TargetRecalibrated,
    /// Explicitly abstains on groups outside the declared support.
    SupportAwareSelective,
}

impl ShiftCorrection {
    fn as_str(self) -> &'static str {
        match self {
            Self::UnmodifiedBaseline => "unmodified_baseline",
            Self::ImportanceWeighted => "importance_weighted",
            Self::GroupConditional => "group_conditional",
            Self::TargetRecalibrated => "target_recalibrated",
            Self::SupportAwareSelective => "support_aware_selective",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftGateDisposition {
    /// Calibration passed and the declared shift family's reviewable assumptions were met.
    /// This is scoped to the declared evidence and does not imply a universal guarantee.
    EligibleForDeclaredClaim,
    /// Evidence supports a claim only over the covered/accepted subset; abstained cases remain
    /// outside the coverage claim.
    SelectiveClaimOnly,
    /// Source-only evidence cannot identify the requested guarantee.
    IndeterminateNeedsTargetLabels,
    /// Overlap or importance-weight diagnostics are insufficient for the declared correction.
    IndeterminateInsufficientSupport,
    /// The sampling/dependence treatment needs domain review before interpreting the interval.
    IndeterminateDependenceReview,
    /// Evidence/counts do not support a shift claim.
    IndeterminateInsufficientEvidence,
    /// Contradictory, stale, invalid, or explicitly violated evidence must not be used to pass.
    FailClosed,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShiftGateIssue {
    EmptyProfileId,
    InvalidDigest { field: &'static str },
    ProfileManifestDigestMismatch,
    CalibrationProfileMismatch,
    CalibrationThresholdsNotMet,
    CalibrationIncomplete,
    CalibrationDomainNotEstablished,
    CalibrationInvalid,
    EmptyGroupIdentifier { field: &'static str },
    DuplicateGroupIdentifier { field: &'static str, group_id: String },
    InvalidNumericEvidence { field: &'static str },
    AssumptionsNotReviewed,
    TargetLabelsNotIndependentOfCalibrationAndTuning,
    TargetLabelsInsufficient { minimum: usize, observed: usize },
    SupportOverlapBelowMinimum { minimum_ppm: u32, observed_ppm: u32 },
    EffectiveSampleSizeBelowMinimum { minimum_milli: u64, observed_milli: u64 },
    ImportanceWeightAboveMaximum { maximum_milli: u64, observed_milli: u64 },
    MethodIncompatibleWithShift {
        family: ShiftFamily,
        method: ShiftCorrection,
    },
    NoShiftAdaptationNeedsPairedReplay,
    UnsupportedGroupNotAbstained { group_id: String },
    UnexpectedAbstention { group_id: String },
    NoUnsupportedGroupsDeclared,
    DependenceTreatmentNotReviewed,
    SparseRegimeCannotEstablishClaim,
}

/// Frozen contract for interpreting one candidate's held-out calibration report under a
/// predeclared distribution-shift family.
///
/// Fractions are stored as floating point numbers in [0, 1]. Evidence such as overlap, label
/// roster, and review flags is caller-attested; the evaluator cannot authenticate data provenance
/// or prove that an assessor actually performed the stated review.
#[derive(Debug, Clone, PartialEq)]
pub struct ShiftClaimProfile {
    pub profile_id: String,
    /// Must exactly match the profile digest used by the consumed CalibrationReport.
    pub calibration_profile_manifest_sha256: String,
    /// Canonical digest of all acceptance-relevant fields below, excluding this field itself.
    pub profile_manifest_sha256: String,
    pub shift_family: ShiftFamily,
    pub correction: ShiftCorrection,
    /// A qualified reviewer has assessed whether the method's shift assumptions fit this scenario.
    pub shift_assumptions_reviewed: bool,
    /// Independent target-label groups used for evaluation / recalibration evidence. These IDs
    /// must not be reused for calibration or method/threshold tuning.
    pub target_label_group_ids: Vec<String>,
    pub minimum_target_label_groups: usize,
    pub target_labels_disjoint_from_calibration_and_tuning: bool,
    /// Reviewed fraction of the target mass judged to lie within source support, in [0, 1].
    pub support_overlap_fraction: f64,
    pub minimum_support_overlap_fraction: f64,
    /// Effective sample size from the declared importance weights, in independent-group units.
    pub importance_weight_effective_sample_size: f64,
    pub minimum_effective_sample_size: f64,
    /// Maximum normalized importance weight (maximum weight divided by mean weight).
    pub maximum_normalized_importance_weight: f64,
    pub allowed_maximum_normalized_importance_weight: f64,
    /// Groups identified as outside declared source support and groups explicitly abstained on.
    pub unsupported_group_ids: Vec<String>,
    pub abstained_group_ids: Vec<String>,
    /// True only after the dependence/aggregation treatment has been independently reviewed.
    pub dependence_treatment_reviewed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShiftGateReport {
    pub profile_id: String,
    pub profile_manifest_sha256: String,
    pub calibration_profile_manifest_sha256: String,
    pub disposition: ShiftGateDisposition,
    pub target_label_group_count: usize,
    pub unsupported_group_count: usize,
    pub abstained_group_count: usize,
    pub issues: Vec<ShiftGateIssue>,
}

fn push_field(buffer: &mut Vec<u8>, name: &str, value: &[u8]) {
    buffer.extend_from_slice(&(name.len() as u64).to_be_bytes());
    buffer.extend_from_slice(name.as_bytes());
    buffer.extend_from_slice(&(value.len() as u64).to_be_bytes());
    buffer.extend_from_slice(value);
}

fn push_text(buffer: &mut Vec<u8>, name: &str, value: &str) {
    push_field(buffer, name, value.as_bytes());
}

fn push_bool(buffer: &mut Vec<u8>, name: &str, value: bool) {
    push_field(buffer, name, &[u8::from(value)]);
}

fn push_usize(buffer: &mut Vec<u8>, name: &str, value: usize) {
    push_field(buffer, name, &(value as u64).to_be_bytes());
}

fn push_f64(buffer: &mut Vec<u8>, name: &str, value: f64) {
    push_field(buffer, name, &value.to_bits().to_be_bytes());
}

fn push_sorted_strings(buffer: &mut Vec<u8>, name: &str, values: &[String]) {
    let mut sorted: Vec<&str> = values.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    push_usize(buffer, &format!("{name}.count"), sorted.len());
    for value in sorted {
        push_text(buffer, name, value);
    }
}

/// Returns the canonical SHA-256 for the shift-gate profile. Roster ordering does not affect it;
/// duplicate roster entries remain represented and are separately rejected by assessment.
pub fn compute_shift_profile_manifest_sha256(profile: &ShiftClaimProfile) -> String {
    let mut bytes = Vec::new();
    push_text(&mut bytes, "domain", "marine-shift-claim-gate-v0");
    push_text(&mut bytes, "profile_id", &profile.profile_id);
    push_text(
        &mut bytes,
        "calibration_profile_manifest_sha256",
        &profile.calibration_profile_manifest_sha256,
    );
    push_text(&mut bytes, "shift_family", profile.shift_family.as_str());
    push_text(&mut bytes, "correction", profile.correction.as_str());
    push_bool(
        &mut bytes,
        "shift_assumptions_reviewed",
        profile.shift_assumptions_reviewed,
    );
    push_sorted_strings(
        &mut bytes,
        "target_label_group_ids",
        &profile.target_label_group_ids,
    );
    push_usize(
        &mut bytes,
        "minimum_target_label_groups",
        profile.minimum_target_label_groups,
    );
    push_bool(
        &mut bytes,
        "target_labels_disjoint_from_calibration_and_tuning",
        profile.target_labels_disjoint_from_calibration_and_tuning,
    );
    push_f64(
        &mut bytes,
        "support_overlap_fraction",
        profile.support_overlap_fraction,
    );
    push_f64(
        &mut bytes,
        "minimum_support_overlap_fraction",
        profile.minimum_support_overlap_fraction,
    );
    push_f64(
        &mut bytes,
        "importance_weight_effective_sample_size",
        profile.importance_weight_effective_sample_size,
    );
    push_f64(
        &mut bytes,
        "minimum_effective_sample_size",
        profile.minimum_effective_sample_size,
    );
    push_f64(
        &mut bytes,
        "maximum_normalized_importance_weight",
        profile.maximum_normalized_importance_weight,
    );
    push_f64(
        &mut bytes,
        "allowed_maximum_normalized_importance_weight",
        profile.allowed_maximum_normalized_importance_weight,
    );
    push_sorted_strings(
        &mut bytes,
        "unsupported_group_ids",
        &profile.unsupported_group_ids,
    );
    push_sorted_strings(
        &mut bytes,
        "abstained_group_ids",
        &profile.abstained_group_ids,
    );
    push_bool(
        &mut bytes,
        "dependence_treatment_reviewed",
        profile.dependence_treatment_reviewed,
    );

    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn duplicate_or_empty_groups(
    groups: &[String],
    field: &'static str,
    issues: &mut Vec<ShiftGateIssue>,
) {
    let mut seen = BTreeSet::new();
    for group_id in groups {
        if group_id.trim().is_empty() {
            issues.push(ShiftGateIssue::EmptyGroupIdentifier { field });
        } else if !seen.insert(group_id.as_str()) {
            issues.push(ShiftGateIssue::DuplicateGroupIdentifier {
                field,
                group_id: group_id.clone(),
            });
        }
    }
}

fn calibration_is_usable(report: &CalibrationReport) -> bool {
    report.disposition == CalibrationDisposition::PassDeclaredCalibrationThresholds
        && !report.slices.is_empty()
        && report.issues.is_empty()
        && report.slices.iter().all(|slice| {
            slice.applicability_gate_met
                && slice.coverage_gate_met == Some(true)
                && slice.interval_score_gate_met == Some(true)
                && slice.thresholds_met == Some(true)
        })
}

fn assessment_is_structurally_valid(profile: &ShiftClaimProfile) -> (bool, Vec<ShiftGateIssue>) {
    let mut issues = Vec::new();
    if profile.profile_id.trim().is_empty() {
        issues.push(ShiftGateIssue::EmptyProfileId);
    }
    if !valid_sha256_hex(&profile.calibration_profile_manifest_sha256) {
        issues.push(ShiftGateIssue::InvalidDigest {
            field: "calibration_profile_manifest_sha256",
        });
    }
    if !valid_sha256_hex(&profile.profile_manifest_sha256) {
        issues.push(ShiftGateIssue::InvalidDigest {
            field: "profile_manifest_sha256",
        });
    }
    if profile.profile_manifest_sha256 != compute_shift_profile_manifest_sha256(profile) {
        issues.push(ShiftGateIssue::ProfileManifestDigestMismatch);
    }

    if !profile.support_overlap_fraction.is_finite()
        || !(0.0..=1.0).contains(&profile.support_overlap_fraction)
    {
        issues.push(ShiftGateIssue::InvalidNumericEvidence {
            field: "support_overlap_fraction",
        });
    }
    if !profile.minimum_support_overlap_fraction.is_finite()
        || !(0.0..=1.0).contains(&profile.minimum_support_overlap_fraction)
    {
        issues.push(ShiftGateIssue::InvalidNumericEvidence {
            field: "minimum_support_overlap_fraction",
        });
    }
    if !profile.importance_weight_effective_sample_size.is_finite()
        || profile.importance_weight_effective_sample_size < 0.0
    {
        issues.push(ShiftGateIssue::InvalidNumericEvidence {
            field: "importance_weight_effective_sample_size",
        });
    }
    if !profile.minimum_effective_sample_size.is_finite()
        || profile.minimum_effective_sample_size <= 0.0
    {
        issues.push(ShiftGateIssue::InvalidNumericEvidence {
            field: "minimum_effective_sample_size",
        });
    }
    if !profile.maximum_normalized_importance_weight.is_finite()
        || profile.maximum_normalized_importance_weight < 0.0
    {
        issues.push(ShiftGateIssue::InvalidNumericEvidence {
            field: "maximum_normalized_importance_weight",
        });
    }
    if !profile.allowed_maximum_normalized_importance_weight.is_finite()
        || profile.allowed_maximum_normalized_importance_weight <= 0.0
    {
        issues.push(ShiftGateIssue::InvalidNumericEvidence {
            field: "allowed_maximum_normalized_importance_weight",
        });
    }

    duplicate_or_empty_groups(
        &profile.target_label_group_ids,
        "target_label_group_ids",
        &mut issues,
    );
    duplicate_or_empty_groups(
        &profile.unsupported_group_ids,
        "unsupported_group_ids",
        &mut issues,
    );
    duplicate_or_empty_groups(
        &profile.abstained_group_ids,
        "abstained_group_ids",
        &mut issues,
    );

    let unsupported: BTreeSet<&str> = profile
        .unsupported_group_ids
        .iter()
        .map(String::as_str)
        .collect();
    for group_id in &profile.abstained_group_ids {
        if !unsupported.contains(group_id.as_str()) {
            issues.push(ShiftGateIssue::UnexpectedAbstention {
                group_id: group_id.clone(),
            });
        }
    }

    issues.sort();
    issues.dedup();
    (issues.is_empty(), issues)
}

fn target_labels_sufficient(profile: &ShiftClaimProfile, issues: &mut Vec<ShiftGateIssue>) -> bool {
    let count = profile.target_label_group_ids.len();
    let sufficient = count >= profile.minimum_target_label_groups
        && profile.minimum_target_label_groups > 0;
    if !sufficient {
        issues.push(ShiftGateIssue::TargetLabelsInsufficient {
            minimum: profile.minimum_target_label_groups.max(1),
            observed: count,
        });
    }
    if !profile.target_labels_disjoint_from_calibration_and_tuning {
        issues.push(
            ShiftGateIssue::TargetLabelsNotIndependentOfCalibrationAndTuning,
        );
    }
    sufficient && profile.target_labels_disjoint_from_calibration_and_tuning
}

/// Assess whether the declared shift-specific prediction claim may proceed.
///
/// This is an admission/policy check, not the shift correction itself. In particular, covariate
/// shift is not inferred from unlabeled covariates here; it must be reviewed and declared by the
/// caller. Joint covariate/outcome shift without independent target labels is indeterminate even
/// when the source-domain calibration report passes.
pub fn assess_shift_claim(
    profile: &ShiftClaimProfile,
    calibration: &CalibrationReport,
) -> ShiftGateReport {
    let (structurally_valid, mut issues) = assessment_is_structurally_valid(profile);
    if calibration.profile_manifest_sha256 != profile.calibration_profile_manifest_sha256 {
        issues.push(ShiftGateIssue::CalibrationProfileMismatch);
    }

    let calibration_state = match calibration.disposition {
        CalibrationDisposition::PassDeclaredCalibrationThresholds => {
            if !calibration_is_usable(calibration) {
                issues.push(ShiftGateIssue::CalibrationThresholdsNotMet);
                0_u8
            } else {
                1_u8
            }
        }
        CalibrationDisposition::IncompleteCoverage => {
            issues.push(ShiftGateIssue::CalibrationIncomplete);
            2_u8
        }
        CalibrationDisposition::DomainNotEstablished => {
            issues.push(ShiftGateIssue::CalibrationDomainNotEstablished);
            2_u8
        }
        CalibrationDisposition::InvalidProfile
        | CalibrationDisposition::InvalidObservations => {
            issues.push(ShiftGateIssue::CalibrationInvalid);
            0_u8
        }
        CalibrationDisposition::FailDeclaredCalibrationThresholds => {
            issues.push(ShiftGateIssue::CalibrationThresholdsNotMet);
            0_u8
        }
    };

    let mut disposition = if !structurally_valid
        || issues.iter().any(|issue| {
            matches!(
                issue,
                ShiftGateIssue::CalibrationProfileMismatch
                    | ShiftGateIssue::CalibrationThresholdsNotMet
                    | ShiftGateIssue::CalibrationInvalid
            )
        })
    {
        ShiftGateDisposition::FailClosed
    } else if calibration_state == 2 {
        ShiftGateDisposition::IndeterminateInsufficientEvidence
    } else if calibration_state == 0 {
        ShiftGateDisposition::FailClosed
    } else {
        match profile.shift_family {
            ShiftFamily::NoShiftControl => {
                if profile.correction == ShiftCorrection::UnmodifiedBaseline {
                    ShiftGateDisposition::EligibleForDeclaredClaim
                } else {
                    issues.push(ShiftGateIssue::NoShiftAdaptationNeedsPairedReplay);
                    ShiftGateDisposition::FailClosed
                }
            }
            ShiftFamily::CovariateShift => match profile.correction {
                ShiftCorrection::ImportanceWeighted => {
                    if !profile.shift_assumptions_reviewed {
                        issues.push(ShiftGateIssue::AssumptionsNotReviewed);
                        ShiftGateDisposition::IndeterminateInsufficientEvidence
                    } else if profile.support_overlap_fraction
                        < profile.minimum_support_overlap_fraction
                    {
                        issues.push(ShiftGateIssue::SupportOverlapBelowMinimum {
                            minimum_ppm: (profile.minimum_support_overlap_fraction * 1_000_000.0)
                                .round() as u32,
                            observed_ppm: (profile.support_overlap_fraction * 1_000_000.0)
                                .round() as u32,
                        });
                        ShiftGateDisposition::IndeterminateInsufficientSupport
                    } else if profile.importance_weight_effective_sample_size
                        < profile.minimum_effective_sample_size
                    {
                        issues.push(ShiftGateIssue::EffectiveSampleSizeBelowMinimum {
                            minimum_milli: (profile.minimum_effective_sample_size * 1_000.0)
                                .round() as u64,
                            observed_milli: (profile.importance_weight_effective_sample_size
                                * 1_000.0)
                                .round() as u64,
                        });
                        ShiftGateDisposition::IndeterminateInsufficientSupport
                    } else if profile.maximum_normalized_importance_weight
                        > profile.allowed_maximum_normalized_importance_weight
                    {
                        issues.push(ShiftGateIssue::ImportanceWeightAboveMaximum {
                            maximum_milli: (profile.allowed_maximum_normalized_importance_weight
                                * 1_000.0)
                                .round() as u64,
                            observed_milli: (profile.maximum_normalized_importance_weight
                                * 1_000.0)
                                .round() as u64,
                        });
                        ShiftGateDisposition::IndeterminateInsufficientSupport
                    } else {
                        ShiftGateDisposition::EligibleForDeclaredClaim
                    }
                }
                ShiftCorrection::SupportAwareSelective => {
                    assess_support_gap(profile, &mut issues)
                }
                _ => {
                    issues.push(ShiftGateIssue::MethodIncompatibleWithShift {
                        family: profile.shift_family,
                        method: profile.correction,
                    });
                    ShiftGateDisposition::FailClosed
                }
            },
            ShiftFamily::Heteroscedasticity | ShiftFamily::LocalUndercoverage => {
                if !target_labels_sufficient(profile, &mut issues) {
                    ShiftGateDisposition::IndeterminateNeedsTargetLabels
                } else if !profile.shift_assumptions_reviewed {
                    issues.push(ShiftGateIssue::AssumptionsNotReviewed);
                    ShiftGateDisposition::IndeterminateInsufficientEvidence
                } else if matches!(
                    profile.correction,
                    ShiftCorrection::GroupConditional | ShiftCorrection::TargetRecalibrated
                ) {
                    ShiftGateDisposition::EligibleForDeclaredClaim
                } else {
                    issues.push(ShiftGateIssue::MethodIncompatibleWithShift {
                        family: profile.shift_family,
                        method: profile.correction,
                    });
                    ShiftGateDisposition::FailClosed
                }
            }
            ShiftFamily::JointCovariateOutcomeShift | ShiftFamily::TemporalBiasOrDrift => {
                if !target_labels_sufficient(profile, &mut issues) {
                    ShiftGateDisposition::IndeterminateNeedsTargetLabels
                } else if !profile.shift_assumptions_reviewed {
                    issues.push(ShiftGateIssue::AssumptionsNotReviewed);
                    ShiftGateDisposition::IndeterminateInsufficientEvidence
                } else if profile.correction == ShiftCorrection::TargetRecalibrated {
                    ShiftGateDisposition::EligibleForDeclaredClaim
                } else {
                    issues.push(ShiftGateIssue::MethodIncompatibleWithShift {
                        family: profile.shift_family,
                        method: profile.correction,
                    });
                    ShiftGateDisposition::FailClosed
                }
            }
            ShiftFamily::SupportGap => assess_support_gap(profile, &mut issues),
            ShiftFamily::DependentSamples => {
                if profile.dependence_treatment_reviewed {
                    ShiftGateDisposition::EligibleForDeclaredClaim
                } else {
                    issues.push(ShiftGateIssue::DependenceTreatmentNotReviewed);
                    ShiftGateDisposition::IndeterminateDependenceReview
                }
            }
            ShiftFamily::SparseOrMissing => {
                issues.push(ShiftGateIssue::SparseRegimeCannotEstablishClaim);
                ShiftGateDisposition::IndeterminateInsufficientEvidence
            }
        }
    };

    issues.sort();
    issues.dedup();
    // Invalid/stale input evidence overrides every potentially passing branch.
    if !structurally_valid
        || issues.iter().any(|issue| {
            matches!(
                issue,
                ShiftGateIssue::CalibrationProfileMismatch
                    | ShiftGateIssue::CalibrationThresholdsNotMet
                    | ShiftGateIssue::CalibrationInvalid
                    | ShiftGateIssue::UnexpectedAbstention { .. }
                    | ShiftGateIssue::DuplicateGroupIdentifier { .. }
                    | ShiftGateIssue::EmptyGroupIdentifier { .. }
                    | ShiftGateIssue::InvalidDigest { .. }
                    | ShiftGateIssue::InvalidNumericEvidence { .. }
                    | ShiftGateIssue::ProfileManifestDigestMismatch
                    | ShiftGateIssue::EmptyProfileId
            )
        })
    {
        disposition = ShiftGateDisposition::FailClosed;
    }

    ShiftGateReport {
        profile_id: profile.profile_id.clone(),
        profile_manifest_sha256: profile.profile_manifest_sha256.clone(),
        calibration_profile_manifest_sha256: profile
            .calibration_profile_manifest_sha256
            .clone(),
        disposition,
        target_label_group_count: profile.target_label_group_ids.len(),
        unsupported_group_count: profile.unsupported_group_ids.len(),
        abstained_group_count: profile.abstained_group_ids.len(),
        issues,
    }
}

fn assess_support_gap(
    profile: &ShiftClaimProfile,
    issues: &mut Vec<ShiftGateIssue>,
) -> ShiftGateDisposition {
    if profile.correction != ShiftCorrection::SupportAwareSelective {
        issues.push(ShiftGateIssue::MethodIncompatibleWithShift {
            family: profile.shift_family,
            method: profile.correction,
        });
        return ShiftGateDisposition::FailClosed;
    }
    if profile.unsupported_group_ids.is_empty() {
        issues.push(ShiftGateIssue::NoUnsupportedGroupsDeclared);
        return ShiftGateDisposition::IndeterminateInsufficientEvidence;
    }

    let abstained: BTreeSet<&str> = profile
        .abstained_group_ids
        .iter()
        .map(String::as_str)
        .collect();
    for group_id in &profile.unsupported_group_ids {
        if !abstained.contains(group_id.as_str()) {
            issues.push(ShiftGateIssue::UnsupportedGroupNotAbstained {
                group_id: group_id.clone(),
            });
        }
    }
    if issues
        .iter()
        .any(|issue| matches!(issue, ShiftGateIssue::UnsupportedGroupNotAbstained { .. }))
    {
        ShiftGateDisposition::FailClosed
    } else {
        ShiftGateDisposition::SelectiveClaimOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prediction_interval_calibration::{
        CalibrationSliceScore,
    };

    const CALIBRATION_DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn passing_calibration() -> CalibrationReport {
        CalibrationReport {
            profile_id: "calibration-profile".to_string(),
            profile_manifest_sha256: CALIBRATION_DIGEST.to_string(),
            disposition: CalibrationDisposition::PassDeclaredCalibrationThresholds,
            input_observation_count: 100,
            unique_observation_count: 100,
            duplicated_observation_record_count: 0,
            duplicated_evaluation_group_record_count: 0,
            slices: vec![CalibrationSliceScore {
                operating_case_id: "shallow-water".to_string(),
                observable_id: "resistance".to_string(),
                unit_id: "N".to_string(),
                nominal_coverage_bp: 9_500,
                observation_count: 100,
                required_observation_count: 100,
                covered_count: 95,
                empirical_coverage: Some(0.95),
                wilson_95_lower: Some(0.88),
                wilson_95_upper: Some(0.98),
                mean_interval_width: Some(5.0),
                normalized_mean_interval_width: Some(0.5),
                mean_interval_score: Some(6.0),
                normalized_mean_interval_score: Some(0.6),
                within_declared_envelope_count: 100,
                borderline_count: 0,
                out_of_domain_count: 0,
                unknown_applicability_count: 0,
                applicability_gate_met: true,
                coverage_gate_met: Some(true),
                interval_score_gate_met: Some(true),
                thresholds_met: Some(true),
            }],
            issues: vec![],
        }
    }

    fn profile(family: ShiftFamily, correction: ShiftCorrection) -> ShiftClaimProfile {
        let mut value = ShiftClaimProfile {
            profile_id: "shift-profile".to_string(),
            calibration_profile_manifest_sha256: CALIBRATION_DIGEST.to_string(),
            profile_manifest_sha256: String::new(),
            shift_family: family,
            correction,
            shift_assumptions_reviewed: true,
            target_label_group_ids: (0..40).map(|i| format!("target-{i:03}")).collect(),
            minimum_target_label_groups: 30,
            target_labels_disjoint_from_calibration_and_tuning: true,
            support_overlap_fraction: 0.95,
            minimum_support_overlap_fraction: 0.80,
            importance_weight_effective_sample_size: 100.0,
            minimum_effective_sample_size: 30.0,
            maximum_normalized_importance_weight: 8.0,
            allowed_maximum_normalized_importance_weight: 20.0,
            unsupported_group_ids: vec![],
            abstained_group_ids: vec![],
            dependence_treatment_reviewed: true,
        };
        value.profile_manifest_sha256 = compute_shift_profile_manifest_sha256(&value);
        value
    }

    fn refresh_digest(profile: &mut ShiftClaimProfile) {
        profile.profile_manifest_sha256 = compute_shift_profile_manifest_sha256(profile);
    }

    #[test]
    fn valid_covariate_shift_requires_and_accepts_overlap_and_weight_diagnostics() {
        let profile = profile(ShiftFamily::CovariateShift, ShiftCorrection::ImportanceWeighted);
        let report = assess_shift_claim(&profile, &passing_calibration());
        assert_eq!(report.disposition, ShiftGateDisposition::EligibleForDeclaredClaim);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn low_support_overlap_makes_importance_weighting_indeterminate() {
        let mut profile = profile(ShiftFamily::CovariateShift, ShiftCorrection::ImportanceWeighted);
        profile.support_overlap_fraction = 0.4;
        refresh_digest(&mut profile);
        let report = assess_shift_claim(&profile, &passing_calibration());
        assert_eq!(report.disposition, ShiftGateDisposition::IndeterminateInsufficientSupport);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            ShiftGateIssue::SupportOverlapBelowMinimum { .. }
        )));
    }

    #[test]
    fn low_effective_sample_size_makes_importance_weighting_indeterminate() {
        let mut profile = profile(ShiftFamily::CovariateShift, ShiftCorrection::ImportanceWeighted);
        profile.importance_weight_effective_sample_size = 4.0;
        refresh_digest(&mut profile);
        let report = assess_shift_claim(&profile, &passing_calibration());
        assert_eq!(report.disposition, ShiftGateDisposition::IndeterminateInsufficientSupport);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            ShiftGateIssue::EffectiveSampleSizeBelowMinimum { .. }
        )));
    }

    #[test]
    fn joint_shift_without_disjoint_target_labels_is_indeterminate_even_if_source_calibration_passes() {
        let mut profile = profile(
            ShiftFamily::JointCovariateOutcomeShift,
            ShiftCorrection::TargetRecalibrated,
        );
        profile.target_label_group_ids.clear();
        profile.target_labels_disjoint_from_calibration_and_tuning = false;
        refresh_digest(&mut profile);
        let report = assess_shift_claim(&profile, &passing_calibration());
        assert_eq!(report.disposition, ShiftGateDisposition::IndeterminateNeedsTargetLabels);
    }

    #[test]
    fn target_labels_reused_for_tuning_cannot_authorize_joint_shift_claim() {
        let mut profile = profile(
            ShiftFamily::JointCovariateOutcomeShift,
            ShiftCorrection::TargetRecalibrated,
        );
        profile.target_labels_disjoint_from_calibration_and_tuning = false;
        refresh_digest(&mut profile);
        let report = assess_shift_claim(&profile, &passing_calibration());
        assert_eq!(report.disposition, ShiftGateDisposition::IndeterminateNeedsTargetLabels);
        assert!(report.issues.contains(
            &ShiftGateIssue::TargetLabelsNotIndependentOfCalibrationAndTuning
        ));
    }

    #[test]
    fn support_gap_passes_only_as_selective_when_every_unsupported_group_is_abstained() {
        let mut profile = profile(
            ShiftFamily::SupportGap,
            ShiftCorrection::SupportAwareSelective,
        );
        profile.unsupported_group_ids = vec!["regime-new-hull".to_string()];
        profile.abstained_group_ids = vec!["regime-new-hull".to_string()];
        refresh_digest(&mut profile);
        let report = assess_shift_claim(&profile, &passing_calibration());
        assert_eq!(report.disposition, ShiftGateDisposition::SelectiveClaimOnly);
    }

    #[test]
    fn one_unsupported_nonabstained_group_fails_closed() {
        let mut profile = profile(
            ShiftFamily::SupportGap,
            ShiftCorrection::SupportAwareSelective,
        );
        profile.unsupported_group_ids = vec!["regime-new-hull".to_string()];
        refresh_digest(&mut profile);
        let report = assess_shift_claim(&profile, &passing_calibration());
        assert_eq!(report.disposition, ShiftGateDisposition::FailClosed);
        assert!(report.issues.contains(
            &ShiftGateIssue::UnsupportedGroupNotAbstained {
                group_id: "regime-new-hull".to_string()
            }
        ));
    }

    #[test]
    fn adaptive_method_must_not_skip_the_no_shift_paired_replay() {
        let profile = profile(ShiftFamily::NoShiftControl, ShiftCorrection::ImportanceWeighted);
        let report = assess_shift_claim(&profile, &passing_calibration());
        assert_eq!(report.disposition, ShiftGateDisposition::FailClosed);
        assert!(report.issues.contains(&ShiftGateIssue::NoShiftAdaptationNeedsPairedReplay));
    }

    #[test]
    fn stale_calibration_profile_binding_fails_closed() {
        let mut profile = profile(ShiftFamily::CovariateShift, ShiftCorrection::ImportanceWeighted);
        profile.calibration_profile_manifest_sha256 =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string();
        // Deliberately leave the shift-profile digest stale.
        let report = assess_shift_claim(&profile, &passing_calibration());
        assert_eq!(report.disposition, ShiftGateDisposition::FailClosed);
    }

    #[test]
    fn digest_is_independent_of_group_roster_order() {
        let mut a = profile(ShiftFamily::CovariateShift, ShiftCorrection::ImportanceWeighted);
        let mut b = a.clone();
        b.target_label_group_ids.reverse();
        assert_eq!(
            compute_shift_profile_manifest_sha256(&a),
            compute_shift_profile_manifest_sha256(&b)
        );
        a.unsupported_group_ids = vec!["b".to_string(), "a".to_string()];
        b.unsupported_group_ids = vec!["a".to_string(), "b".to_string()];
        assert_eq!(
            compute_shift_profile_manifest_sha256(&a),
            compute_shift_profile_manifest_sha256(&b)
        );
    }

    #[test]
    fn incomplete_or_out_of_domain_calibration_cannot_pass_shift_gate() {
        let profile = profile(ShiftFamily::CovariateShift, ShiftCorrection::ImportanceWeighted);
        let mut report = passing_calibration();
        report.disposition = CalibrationDisposition::DomainNotEstablished;
        report.slices[0].applicability_gate_met = false;
        let gate = assess_shift_claim(&profile, &report);
        assert_eq!(
            gate.disposition,
            ShiftGateDisposition::IndeterminateInsufficientEvidence
        );
    }
}
