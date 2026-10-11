// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Approximate planning check for main-effect power in blocked 2-level full factorials.
//!
//! This is intentionally not an exact noncentral-t calculation. It uses a normal
//! approximation and a supplied residual SD as if known. The result is a preliminary
//! design-risk signal, not proof that the realized experiment is adequately powered.

use serde::{Deserialize, Serialize};

use symthaea_agribot::soil_process::{EvidenceKind, EvidenceRef};

use crate::screening_design::{
    generate_screening_design, ScreeningDesignError, ScreeningDesignRequest,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreeningPowerError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl ScreeningPowerError {
    fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

impl std::fmt::Display for ScreeningPowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.reason)
    }
}

impl std::error::Error for ScreeningPowerError {}

impl From<ScreeningDesignError> for ScreeningPowerError {
    fn from(error: ScreeningDesignError) -> Self {
        Self {
            field: error.field,
            reason: error.reason,
        }
    }
}

/// Evidence-backed lower/upper bounds for residual variation in the preregistered
/// primary endpoint, expressed in exactly the same outcome unit. The context identifies
/// the pilot, study, or publication and why it applies. The interval is not assumed to be
/// a confidence interval unless the cited method explicitly supports that interpretation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualVariationEstimate {
    pub residual_standard_deviation_lower_bound: f64,
    pub residual_standard_deviation_upper_bound: f64,
    pub unit: String,
    pub source_context_id: String,
    pub estimation_method_id: String,
    pub evidence: EvidenceRef,
}

impl ResidualVariationEstimate {
    fn validate(&self, endpoint_unit: &str) -> Result<(), ScreeningPowerError> {
        if !self.residual_standard_deviation_lower_bound.is_finite()
            || self.residual_standard_deviation_lower_bound <= 0.0
        {
            return Err(ScreeningPowerError::new(
                "residual_standard_deviation_lower_bound",
                "must be finite and greater than zero",
            ));
        }
        if !self.residual_standard_deviation_upper_bound.is_finite()
            || self.residual_standard_deviation_upper_bound
                < self.residual_standard_deviation_lower_bound
        {
            return Err(ScreeningPowerError::new(
                "residual_standard_deviation_upper_bound",
                "must be finite and greater than or equal to the lower bound",
            ));
        }
        if self.unit.trim().is_empty() || self.unit != endpoint_unit {
            return Err(ScreeningPowerError::new(
                "residual_variation.unit",
                "must exactly match the primary endpoint unit; implicit unit conversion is not supported",
            ));
        }
        if self.source_context_id.trim().is_empty() {
            return Err(ScreeningPowerError::new(
                "residual_variation.source_context_id",
                "must identify the pilot/study/publication and applicability context",
            ));
        }
        if self.estimation_method_id.trim().is_empty() {
            return Err(ScreeningPowerError::new(
                "residual_variation.estimation_method_id",
                "must identify the variance estimation method",
            ));
        }
        if self.evidence.evidence_id.trim().is_empty() {
            return Err(ScreeningPowerError::new(
                "residual_variation.evidence",
                "evidence ID cannot be empty",
            ));
        }
        if self.evidence.kind == EvidenceKind::Scenario {
            return Err(ScreeningPowerError::new(
                "residual_variation.evidence.kind",
                "scenario-only variance cannot support a power-planning assessment",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreeningPowerStatus {
    /// Normal-approximation power meets the declared target under stated assumptions.
    ProjectedTargetMet,
    /// Normal-approximation power falls below the declared target.
    ProjectedTargetNotMet,
    /// The calculated replication target exceeds the schedule generator's current bounds.
    RequiredReplicationExceedsPlannerBounds,
}

/// A block/replication-count combination supported by the current schedule generator.
///
/// These are capacity-envelope options, not newly approved plans: any option that changes
/// the current block count or within-block replication requires a changed design fingerprint
/// and fresh bench-scale review. Block identities and evidence must be supplied in the new
/// request; this type does not synthesize them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeasibleReplicationOption {
    pub block_count: u32,
    pub replicates_per_setting_per_block: u32,
    /// Complete factorial replications = block_count × within-block replication.
    pub complete_factorial_replicates: u32,
    pub treatment_run_count: u32,
    pub center_point_run_count: u32,
    pub total_run_count: u32,
    /// Upper-bound standard error for the main effect at this feasible replication count.
    pub standard_error_main_effect_upper_bound: f64,
    /// Projected two-sided power using the same normal approximation as the current design.
    pub projected_power_normal_approx: f64,
    /// True when this configuration's calculated projected power meets target_power.
    pub meets_target_power_normal_approx: bool,
    /// True when this count reaches the rounded normal-approximation replication lower bound.
    pub meets_approximate_replication_target: bool,
    /// True only when the block and replication counts equal those in the current request.
    pub matches_current_request_counts: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningPowerAssessment {
    pub experiment_id: String,
    pub input_snapshot_id: String,
    /// SHA-256 of the versioned design-input payload used by the reviewed schedule.
    pub request_sha256: String,
    /// SHA-256 of the exact generated run-order artifact used for this assessment.
    pub schedule_sha256: String,
    pub primary_endpoint_id: String,
    pub endpoint_unit: String,
    pub effect_family: String,
    pub factor_count: u32,
    pub factorial_combinations: u32,
    /// Number of complete factorial replicates represented by blocks × within-cell replicates.
    pub current_full_factorial_replicates: u32,
    pub maximum_full_factorial_replicates_supported: u32,
    pub required_full_factorial_replicates_approx: u32,
    /// Lower-bound arithmetic difference only; this may understate the increment needed by
    /// a realizable block/replication configuration. Consult feasible_replication_options.
    pub additional_full_factorial_replicates_approx: u32,
    /// All 14 currently supported block × within-block replication count combinations,
    /// with run-budget consequences. Not every integer from 2 through 16 is realizable.
    pub feasible_replication_options: Vec<FeasibleReplicationOption>,
    pub familywise_alpha: f64,
    pub bonferroni_alpha_per_main_effect: f64,
    pub target_power: f64,
    pub projected_power_normal_approx: f64,
    pub residual_standard_deviation_lower_bound: f64,
    /// Conservative upper SD bound used for standard error, projected power and replication.
    pub residual_standard_deviation_upper_bound_used: f64,
    pub residual_variation_source_context_id: String,
    pub residual_variation_evidence: EvidenceRef,
    pub standard_error_main_effect_upper_bound: f64,
    pub standardized_main_effect_lower_bound_at_minimum_meaningful_difference: f64,
    pub status: ScreeningPowerStatus,
    pub assumptions: Vec<String>,
    pub scope_note: String,
}

/// Assess approximate power for the k main effects of a blocked two-level full factorial.
///
/// Uses Var(main-effect contrast) = sigma^2 / (n * 2^(k-2)), where n is the number of
/// independent complete-factorial replicates (blocks × within-setting replicates). The
/// familywise alpha is Bonferroni-adjusted across the k main effects. Required replication
/// is estimated with a two-sided normal approximation at the requested target power.
///
/// This does NOT calculate exact finite-sample t power, account for uncertainty in the
/// variance estimate, correct for interaction tests, assess block-by-treatment interactions,
/// model non-normal/heteroskedastic outcomes, or account for center points in the contrast.
pub fn assess_screening_power(
    request: &ScreeningDesignRequest,
    residual_variation: &ResidualVariationEstimate,
    familywise_alpha: f64,
    target_power: f64,
) -> Result<ScreeningPowerAssessment, ScreeningPowerError> {
    // Re-use full input validation so a power assessment cannot be detached from an
    // invalid or unreviewed design request. This generates a bounded in-memory schedule only.
    let validated_schedule = generate_screening_design(request)?;

    if !familywise_alpha.is_finite() || !(0.0..0.20).contains(&familywise_alpha) || familywise_alpha == 0.0 {
        return Err(ScreeningPowerError::new(
            "familywise_alpha",
            "must be finite and strictly between 0 and 0.20",
        ));
    }
    if !target_power.is_finite() || !(0.50..=0.99).contains(&target_power) {
        return Err(ScreeningPowerError::new(
            "target_power",
            "must be finite and between 0.50 and 0.99",
        ));
    }

    let endpoint = &request.primary_endpoint;
    residual_variation.validate(&endpoint.unit)?;

    let factor_count = request.factors.len();
    let combinations = 1_usize << factor_count;
    let block_count = request.blocks.len();
    let within_block_replicates = request.replicates_per_setting_per_block as usize;
    let current_full_factorial_replicates = block_count * within_block_replicates;
    // Matches the screening generator bounds: at most 8 complete blocks and 2 replicates
    // per setting in each block. This is a software bound, not a scientific optimum.
    const MAX_FULL_FACTORIAL_REPLICATES: usize = 8 * 2;

    let sd = residual_variation.residual_standard_deviation_upper_bound;
    let delta = endpoint.minimum_practically_meaningful_difference;
    let alpha_per_main_effect = familywise_alpha / factor_count as f64;
    let critical_z = inverse_normal_cdf(1.0 - alpha_per_main_effect / 2.0);
    let target_z = inverse_normal_cdf(target_power);
    let contrast_scale = (current_full_factorial_replicates as f64
        * 2_f64.powi(factor_count as i32 - 2))
        .sqrt();
    let standard_error = sd / contrast_scale;
    let noncentrality = delta / standard_error;
    let projected_power =
        (normal_cdf(-critical_z - noncentrality) + 1.0 - normal_cdf(critical_z - noncentrality))
            .clamp(0.0, 1.0);

    let required_replicates_f = (((critical_z + target_z) * sd / delta).powi(2)
        / 2_f64.powi(factor_count as i32 - 2))
        .ceil()
        .max(1.0);
    if !required_replicates_f.is_finite() || required_replicates_f > u32::MAX as f64 {
        return Err(ScreeningPowerError::new(
            "required_full_factorial_replicates_approx",
            "calculated replication requirement overflows supported output",
        ));
    }
    let required_replicates = required_replicates_f as usize;
    let additional_replicates = required_replicates.saturating_sub(current_full_factorial_replicates);

    // The analytical replication lower bound is an integer, but the generator supports
    // only B × r where B ∈ [2, 8] complete blocks and r ∈ [1, 2] runs per treatment cell
    // within each block. Therefore counts such as 9, 11, 13, and 15 cannot be instantiated
    // without rounding up to a feasible configuration. Expose every feasible budget rather
    // than implying that an unattainable integer is an executable schedule.
    let center_points_per_block = request.center_point_runs_per_block as usize;
    let current_total_run_count = validated_schedule.total_planned_run_count as usize;
    let mut feasible_replication_options = Vec::with_capacity(
        (MAX_BLOCKS - MIN_BLOCKS + 1) * MAX_REPLICATES_PER_SETTING_PER_BLOCK as usize,
    );
    for candidate_block_count in MIN_BLOCKS..=MAX_BLOCKS {
        for candidate_replicates_per_setting in 1..=MAX_REPLICATES_PER_SETTING_PER_BLOCK {
            let candidate_full_factorial_replicates =
                candidate_block_count * candidate_replicates_per_setting as usize;
            let treatment_run_count = combinations * candidate_full_factorial_replicates;
            let center_point_run_count = center_points_per_block * candidate_block_count;
            let total_run_count = treatment_run_count + center_point_run_count;
            let option_contrast_scale = (candidate_full_factorial_replicates as f64
                * 2_f64.powi(factor_count as i32 - 2))
                .sqrt();
            let option_standard_error = sd / option_contrast_scale;
            let option_noncentrality = delta / option_standard_error;
            let option_projected_power = (normal_cdf(-critical_z - option_noncentrality)
                + 1.0
                - normal_cdf(critical_z - option_noncentrality))
            .clamp(0.0, 1.0);
            let matches_current_request_counts = candidate_block_count == block_count
                && candidate_replicates_per_setting as usize == within_block_replicates;
            feasible_replication_options.push(FeasibleReplicationOption {
                block_count: candidate_block_count as u32,
                replicates_per_setting_per_block: candidate_replicates_per_setting as u32,
                complete_factorial_replicates: candidate_full_factorial_replicates as u32,
                treatment_run_count: treatment_run_count as u32,
                center_point_run_count: center_point_run_count as u32,
                total_run_count: total_run_count as u32,
                standard_error_main_effect_upper_bound: option_standard_error,
                projected_power_normal_approx: option_projected_power,
                meets_target_power_normal_approx: option_projected_power >= target_power,
                meets_approximate_replication_target: candidate_full_factorial_replicates
                    >= required_replicates,
                matches_current_request_counts,
            });
        }
    }
    feasible_replication_options.sort_by(|left, right| {
        left.complete_factorial_replicates
            .cmp(&right.complete_factorial_replicates)
            .then_with(|| right.block_count.cmp(&left.block_count))
            .then_with(|| {
                left.replicates_per_setting_per_block
                    .cmp(&right.replicates_per_setting_per_block)
            })
    });

    debug_assert_eq!(
        feasible_replication_options
            .iter()
            .filter(|option| option.matches_current_request_counts)
            .map(|option| option.total_run_count as usize)
            .next(),
        Some(current_total_run_count)
    );

    let status = if required_replicates > MAX_FULL_FACTORIAL_REPLICATES {
        ScreeningPowerStatus::RequiredReplicationExceedsPlannerBounds
    } else if projected_power >= target_power {
        ScreeningPowerStatus::ProjectedTargetMet
    } else {
        ScreeningPowerStatus::ProjectedTargetNotMet
    };

    Ok(ScreeningPowerAssessment {
        experiment_id: request.experiment_id.clone(),
        input_snapshot_id: request.input_snapshot_id.clone(),
        request_sha256: validated_schedule.request_sha256.clone(),
        schedule_sha256: validated_schedule.schedule_sha256.clone(),
        primary_endpoint_id: endpoint.endpoint_id.clone(),
        endpoint_unit: endpoint.unit.clone(),
        effect_family: "two-sided main effects only; Bonferroni familywise adjustment across all declared factors".into(),
        factor_count: factor_count as u32,
        factorial_combinations: combinations as u32,
        current_full_factorial_replicates: current_full_factorial_replicates as u32,
        maximum_full_factorial_replicates_supported: MAX_FULL_FACTORIAL_REPLICATES as u32,
        required_full_factorial_replicates_approx: required_replicates as u32,
        additional_full_factorial_replicates_approx: additional_replicates.min(u32::MAX as usize) as u32,
        feasible_replication_options,
        familywise_alpha,
        bonferroni_alpha_per_main_effect: alpha_per_main_effect,
        target_power,
        projected_power_normal_approx: projected_power,
        residual_standard_deviation_lower_bound: residual_variation
            .residual_standard_deviation_lower_bound,
        residual_standard_deviation_upper_bound_used: sd,
        residual_variation_source_context_id: residual_variation.source_context_id.clone(),
        residual_variation_evidence: residual_variation.evidence.clone(),
        standard_error_main_effect_upper_bound: standard_error,
        standardized_main_effect_lower_bound_at_minimum_meaningful_difference: noncentrality,
        status,
        assumptions: vec![
            "Complete two-level factorial treatment combinations with independent, balanced replicates.".into(),
            "Independent residual errors with common variance across factors and treatment settings.".into(),
            "Additive block effects; no block-by-treatment interaction affecting the target contrast.".into(),
            "Residual standard deviation treated as known for this normal approximation.".into(),
            "Bonferroni adjustment covers the k main effects only, not interactions or secondary endpoints.".into(),
            "Center-point runs are not used to estimate the factorial main-effect contrast or residual variance.".into(),
        ],
        scope_note: "preliminary normal-approximation planning screen only; not exact finite-sample power, proof of adequate power, a causal conclusion, an agronomic validation, or permission to execute/apply material. Feasible block/replication count options expose run-budget consequences; changing counts requires a new preregistration fingerprint and review.".into(),
    })
}

/// Standard normal CDF approximation (Abramowitz–Stegun 7.1.26; absolute error on the
/// order of 1e-7). Kept local to avoid silently adding a numerical dependency.
fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * z);
    let density = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let tail = density
        * t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    if x >= 0.0 {
        (1.0 - tail).clamp(0.0, 1.0)
    } else {
        tail.clamp(0.0, 1.0)
    }
}

fn inverse_normal_cdf(probability: f64) -> f64 {
    // Input range is strictly inside (0, 1), validated by caller bounds.
    let mut lower = -8.0;
    let mut upper = 8.0;
    for _ in 0..100 {
        let midpoint = lower + (upper - lower) / 2.0;
        if normal_cdf(midpoint) < probability {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    lower + (upper - lower) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening_design::{
        BenchScaleReview, BenchScaleReviewStatus, ExperimentBlock, FactorRandomizationClass,
        PrimaryEndpoint, ScreeningFactor,
    };

    fn evidence(id: &str, kind: EvidenceKind) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            kind,
        }
    }

    fn request() -> ScreeningDesignRequest {
        let mut request = ScreeningDesignRequest {
            experiment_id: "power-screen-001".into(),
            preregistration_id: "prereg-power-v1".into(),
            input_snapshot_id: "power-inputs-v1".into(),
            protocol_id: "bench-power-v1".into(),
            objective: "Estimate main effects on dry-basis char yield".into(),
            primary_hypothesis: "At least one main effect reaches the predeclared difference".into(),
            analysis_plan_id: "power-analysis-v1".into(),
            randomization_seed: 42,
            factors: vec![
                ScreeningFactor {
                    factor_id: "temperature".into(),
                    label: "Peak temperature".into(),
                    unit: "degree_C".into(),
                    low_value: 400.0,
                    high_value: 500.0,
                    max_absolute_deviation: 2.0,
                    deviation_tolerance_evidence: evidence("temp-tolerance", EvidenceKind::Literature),
                    randomization_class: FactorRandomizationClass::RandomizablePerRun,
                    low_level_evidence: evidence("temp-low", EvidenceKind::Literature),
                    high_level_evidence: evidence("temp-high", EvidenceKind::Literature),
                },
                ScreeningFactor {
                    factor_id: "residence".into(),
                    label: "Residence time".into(),
                    unit: "minute".into(),
                    low_value: 10.0,
                    high_value: 30.0,
                    max_absolute_deviation: 1.0,
                    deviation_tolerance_evidence: evidence("time-tolerance", EvidenceKind.Literature),
                    randomization_class: FactorRandomizationClass::RandomizablePerRun,
                    low_level_evidence: evidence("time-low", EvidenceKind::Literature),
                    high_level_evidence: evidence("time-high", EvidenceKind::Literature),
                },
            ],
            blocks: vec![
                ExperimentBlock {
                    block_id: "day-1".into(),
                    description: "Bench day one".into(),
                    evidence: evidence("day1", EvidenceKind::Literature),
                },
                ExperimentBlock {
                    block_id: "day-2".into(),
                    description: "Bench day two".into(),
                    evidence: evidence("day2", EvidenceKind::Literature),
                },
            ],
            replicates_per_setting_per_block: 1,
            center_point_runs_per_block: 0,
            primary_endpoint: PrimaryEndpoint {
                endpoint_id: "dry-char-yield".into(),
                description: "Dry char mass divided by dry feedstock mass".into(),
                unit: "kg_per_kg_dry_feedstock".into(),
                measurement_method_id: "gravimetric-v1".into(),
                minimum_practically_meaningful_difference: 0.03,
                difference_rationale_evidence: evidence("mpmd-rationale", EvidenceKind::Literature),
            },
            bench_scale_review: BenchScaleReview {
                status: BenchScaleReviewStatus::ApprovedForThisBenchScaleProtocol,
                protocol_id: "bench-power-v1".into(),
                reviewed_input_snapshot_id: "power-inputs-v1".into(),
                reviewed_design_sha256: String::new(),
                review_id: "review-power-v1".into(),
                reviewer_role: "qualified bench-scale reviewer".into(),
                evidence: evidence("review-record", EvidenceKind::Measured),
            },
        };
        let design_digest =
            crate::screening_design::screening_design_sha256(&request).unwrap();
        request.bench_scale_review.reviewed_design_sha256 = design_digest;
        request
    }

    fn variation(
        lower: f64,
        upper: f64,
        kind: EvidenceKind,
        unit: &str,
    ) -> ResidualVariationEstimate {
        ResidualVariationEstimate {
            residual_standard_deviation_lower_bound: lower,
            residual_standard_deviation_upper_bound: upper,
            unit: unit.into(),
            source_context_id: "pilot-batch-series-v1".into(),
            estimation_method_id: "replicated-pilot-residual-ms-v1".into(),
            evidence: evidence("pilot-residual-sd-v1", kind),
        }
    }

    #[test]
    fn projected_power_and_required_replication_are_explicit() {
        let input = request();
        let result = assess_screening_power(
            &input,
            &variation(0.008, 0.01, EvidenceKind::Measured, &input.primary_endpoint.unit),
            0.05,
            0.80,
        )
        .unwrap();
        assert_eq!(result.current_full_factorial_replicates, 2);
        assert_eq!(result.factorial_combinations, 4);
        assert_eq!(
            result.request_sha256,
            crate::screening_design::screening_design_sha256(&input).unwrap()
        );
        let planned = generate_screening_design(&input).unwrap();
        assert_eq!(result.schedule_sha256, planned.schedule_sha256);
        assert!(result.projected_power_normal_approx >= 0.80);
        assert_eq!(result.status, ScreeningPowerStatus::ProjectedTargetMet);
        assert_eq!(result.bonferroni_alpha_per_main_effect, 0.025);
        assert_eq!(result.residual_standard_deviation_lower_bound, 0.008);
        assert_eq!(result.residual_standard_deviation_upper_bound_used, 0.01);
        assert!(result
            .scope_note
            .contains("not exact finite-sample power"));
    }

    #[test]
    fn larger_variance_reduces_projected_power_and_increases_replication_need() {
        let input = request();
        let low = assess_screening_power(
            &input,
            &variation(0.008, 0.01, EvidenceKind::Measured, &input.primary_endpoint.unit),
            0.05,
            0.80,
        )
        .unwrap();
        let high = assess_screening_power(
            &input,
            &variation(0.10, 0.20, EvidenceKind::Measured, &input.primary_endpoint.unit),
            0.05,
            0.80,
        )
        .unwrap();
        assert!(high.projected_power_normal_approx < low.projected_power_normal_approx);
        assert!(
            high.required_full_factorial_replicates_approx
                > low.required_full_factorial_replicates_approx
        );
        assert_eq!(
            high.status,
            ScreeningPowerStatus::RequiredReplicationExceedsPlannerBounds
        );
        assert!(high
            .feasible_replication_options
            .iter()
            .all(|option| !option.meets_approximate_replication_target));
    }

    #[test]
    fn feasible_replication_options_round_up_unrealizable_counts_and_include_center_point_cost() {
        let mut input = request();
        input.center_point_runs_per_block = 3;
        let reviewed_design_sha256 =
            crate::screening_design::screening_design_sha256(&input).unwrap();
        input.bench_scale_review.reviewed_design_sha256 = reviewed_design_sha256;

        // For these values, the normal approximation produces a lower-bound requirement
        // of 9 complete factorial replicates. The generator cannot express 9: B × r must
        // use B ∈ [2, 8] and r ∈ [1, 2]. Its first feasible count above 9 is 10 (5 × 2).
        let result = assess_screening_power(
            &input,
            &variation(0.008, 0.028, EvidenceKind::Measured, &input.primary_endpoint.unit),
            0.05,
            0.80,
        )
        .unwrap();

        assert_eq!(result.required_full_factorial_replicates_approx, 9);
        assert_eq!(result.feasible_replication_options.len(), 14);
        assert!(!result.feasible_replication_options.iter().any(|option| {
            option.complete_factorial_replicates == 9
        }));
        let min_feasible_target = result
            .feasible_replication_options
            .iter()
            .filter(|option| option.meets_approximate_replication_target)
            .map(|option| option.complete_factorial_replicates)
            .min();
        assert_eq!(min_feasible_target, Some(10));

        let ten_replicate_option = result
            .feasible_replication_options
            .iter()
            .find(|option| option.complete_factorial_replicates == 10)
            .unwrap();
        assert_eq!(ten_replicate_option.block_count, 5);
        assert_eq!(ten_replicate_option.replicates_per_setting_per_block, 2);
        assert_eq!(ten_replicate_option.treatment_run_count, 40);
        assert_eq!(ten_replicate_option.center_point_run_count, 15);
        assert_eq!(ten_replicate_option.total_run_count, 55);
        assert!(ten_replicate_option.standard_error_main_effect_upper_bound.is_finite());
        assert!(ten_replicate_option.projected_power_normal_approx >= 0.80);
        assert!(ten_replicate_option.meets_target_power_normal_approx);
        assert!(!ten_replicate_option.matches_current_request_counts);

        let mut previous_projected_power = 0.0_f64;
        for option in &result.feasible_replication_options {
            // Options are ordered by complete-factorial replication count; with fixed
            // factor/variance inputs, estimated power must not decrease as replication rises.
            assert!(
                option.projected_power_normal_approx + 1e-12 >= previous_projected_power
            );
            previous_projected_power = option.projected_power_normal_approx;

            assert!((2..=8).contains(&option.block_count));
            assert!((1..=2).contains(&option.replicates_per_setting_per_block));
            assert_eq!(
                option.complete_factorial_replicates,
                option.block_count * option.replicates_per_setting_per_block
            );
            assert_eq!(
                option.total_run_count,
                option.treatment_run_count + option.center_point_run_count
            );
            assert_eq!(
                option.center_point_run_count,
                option.block_count * input.center_point_runs_per_block as u32
            );
            assert!(option.total_run_count <= 1_100);
            assert!(option.projected_power_normal_approx.is_finite());
            assert!((0.0..=1.0).contains(&option.projected_power_normal_approx));
            assert_eq!(
                option.meets_target_power_normal_approx,
                option.projected_power_normal_approx >= result.target_power
            );
            if option.meets_approximate_replication_target {
                assert!(option.complete_factorial_replicates >= 9);
            }
        }
    }

    #[test]
    fn scenario_variance_and_unit_mismatch_are_rejected() {
        let input = request();
        assert!(assess_screening_power(
            &input,
            &variation(0.008, 0.01, EvidenceKind::Scenario, &input.primary_endpoint.unit),
            0.05,
            0.80,
        )
        .is_err());
        assert!(assess_screening_power(
            &input,
            &variation(0.008, 0.01, EvidenceKind::Measured, "percent"),
            0.05,
            0.80,
        )
        .is_err());
    }

    #[test]
    fn alpha_power_and_variance_inputs_fail_closed() {
        let input = request();
        let sd = variation(0.008, 0.01, EvidenceKind::Measured, &input.primary_endpoint.unit);
        assert!(assess_screening_power(&input, &sd, 0.0, 0.80).is_err());
        assert!(assess_screening_power(&input, &sd, 0.05, 1.0).is_err());
        assert!(assess_screening_power(
            &input,
            &variation(f64::NAN, 0.01, EvidenceKind::Measured, &input.primary_endpoint.unit),
            0.05,
            0.80
        )
        .is_err());
        assert!(assess_screening_power(
            &input,
            &variation(0.02, 0.01, EvidenceKind::Measured, &input.primary_endpoint.unit),
            0.05,
            0.80
        )
        .is_err());
    }

    #[test]
    fn known_normal_quantiles_and_tail_behavior_are_sane() {
        assert!((inverse_normal_cdf(0.975) - 1.96).abs() < 0.01);
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!(normal_cdf(3.0) > 0.998);
        assert!(normal_cdf(-3.0) < 0.002);
    }
}
