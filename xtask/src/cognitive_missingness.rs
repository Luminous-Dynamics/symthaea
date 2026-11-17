//! Partial-identification scoring for bounded task loss with delayed or missing outcomes.
//!
//! This module computes observed-case summaries, worst-case identification bounds, and
//! preregistered MNAR sensitivity surfaces. It does not establish measurement authenticity,
//! causal identification, or benchmark superiority.

use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MissingnessInput {
    pub schema_version: u32,
    pub dataset_id: String,
    pub scenario_id: String,
    pub comparison_track: String,
    pub evaluation_horizon_tick: u64,
    pub candidate: MissingnessArmInput,
    pub reference: MissingnessArmInput,
    pub sensitivity_grid: SensitivityGrid,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MissingnessArmInput {
    pub name: String,
    /// Number of eligible episodes assigned to this arm at the registered horizon.
    pub eligible_count: usize,
    /// Only outcomes actually revealed by the evaluation horizon; each must be in [0, 1].
    pub observed_losses: Vec<f64>,
    pub delayed_pending_count: usize,
    pub right_censored_count: usize,
    pub declared_missing_count: usize,
    pub unknown_count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SensitivityGrid {
    /// Delta = mean loss among unobserved outcomes minus observed-case mean for candidate.
    pub candidate_deltas: Vec<f64>,
    /// Same delta definition for the reference arm.
    pub reference_deltas: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ArmMissingnessSummary {
    pub name: String,
    pub eligible_count: usize,
    pub observed_count: usize,
    pub observed_outcome_fraction: f64,
    pub observed_mean_loss: Option<f64>,
    pub delayed_pending_count: usize,
    pub right_censored_count: usize,
    pub declared_missing_count: usize,
    pub unknown_count: usize,
    /// Worst-case full-arm mean-loss identification bounds, not a confidence interval.
    pub full_mean_loss_lower_bound: f64,
    pub full_mean_loss_upper_bound: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SensitivityPoint {
    pub candidate_delta: f64,
    pub reference_delta: f64,
    pub candidate_mean_loss: f64,
    pub reference_mean_loss: f64,
    /// Negative favors candidate because the outcome is loss.
    pub candidate_minus_reference_loss: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SensitivityTippingPoint {
    pub candidate_delta: f64,
    pub reference_delta: f64,
    pub candidate_minus_reference_loss: f64,
    /// L-infinity departure from the no-selection-difference point (delta_candidate, delta_reference)=(0,0).
    pub max_abs_sensitivity_delta: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SensitivityGridSummary {
    pub grid_point_count: usize,
    pub candidate_favoring_point_count: usize,
    pub reference_favoring_point_count: usize,
    pub tie_point_count: usize,
    pub minimum_candidate_minus_reference_loss: Option<f64>,
    pub maximum_candidate_minus_reference_loss: Option<f64>,
    /// True only when every registered point favors candidate; not a statistical claim.
    pub candidate_favoring_at_every_registered_point: bool,
    /// True when the registered grid contains points favoring both arms.
    pub ranking_reverses_within_registered_grid: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MissingnessScore {
    pub metric_version: String,
    pub source_input_sha256: String,
    /// SHA-256 of serialized output with this field set to null.
    pub result_sha256: Option<String>,
    pub dataset_id: String,
    pub scenario_id: String,
    pub comparison_track: String,
    pub evaluation_horizon_tick: u64,
    pub candidate: ArmMissingnessSummary,
    pub reference: ArmMissingnessSummary,
    /// Conservative difference interval from separate arm bounds; not a confidence interval.
    pub candidate_minus_reference_loss_lower_bound: f64,
    pub candidate_minus_reference_loss_upper_bound: f64,
    /// True only when the identification interval alone lies wholly in the candidate-favoring direction.
    pub identification_interval_excludes_zero_in_candidate_favor: bool,
    /// Observed-case candidate-minus-reference loss; selected-data statistic, not full-data value.
    pub observed_case_candidate_minus_reference_loss: Option<f64>,
    pub sensitivity_surface: Vec<SensitivityPoint>,
    pub sensitivity_grid_summary: SensitivityGridSummary,
    /// Nearest registered sensitivity-grid point that ties or reverses the observed-case sign.
    pub sensitivity_tipping_point: Option<SensitivityTippingPoint>,
    pub sensitivity_status: String,
    pub interpretation_guard: String,
}

pub fn evaluate_file(path: &Path) -> anyhow::Result<MissingnessScore> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading missingness input {}", path.display()))?;
    let input: MissingnessInput = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing missingness input {}", path.display()))?;
    let mut score = score(input, &bytes)?;
    score.result_sha256 = None;
    let canonical = serde_json::to_vec(&score)?;
    score.result_sha256 = Some(sha256_hex(&canonical));
    Ok(score)
}

/// Score one candidate/reference pair using only observed outcomes and explicit availability counts.
pub fn score(input: MissingnessInput, source_bytes: &[u8]) -> anyhow::Result<MissingnessScore> {
    ensure!(input.schema_version == 1, "unsupported missingness schema_version {}", input.schema_version);
    ensure!(!input.dataset_id.trim().is_empty(), "dataset_id must not be empty");
    ensure!(!input.scenario_id.trim().is_empty(), "scenario_id must not be empty");
    ensure!(!input.comparison_track.trim().is_empty(), "comparison_track must not be empty");
    ensure!(input.candidate.name != input.reference.name, "candidate and reference arm names must differ");

    validate_arm(&input.candidate)?;
    validate_arm(&input.reference)?;

    ensure!(!input.sensitivity_grid.candidate_deltas.is_empty(), "candidate sensitivity grid must not be empty");
    ensure!(!input.sensitivity_grid.reference_deltas.is_empty(), "reference sensitivity grid must not be empty");
    ensure!(
        input.sensitivity_grid.candidate_deltas.iter().any(|delta| *delta == 0.0),
        "candidate sensitivity grid must include delta=0"
    );
    ensure!(
        input.sensitivity_grid.reference_deltas.iter().any(|delta| *delta == 0.0),
        "reference sensitivity grid must include delta=0"
    );
    ensure!(
        input.sensitivity_grid.candidate_deltas.iter().all(|delta| delta.is_finite()),
        "candidate sensitivity grid values must be finite"
    );
    ensure!(
        input.sensitivity_grid.reference_deltas.iter().all(|delta| delta.is_finite()),
        "reference sensitivity grid values must be finite"
    );

    let candidate = summarize_arm(&input.candidate)?;
    let reference = summarize_arm(&input.reference)?;
    let delta_lower = candidate.full_mean_loss_lower_bound - reference.full_mean_loss_upper_bound;
    let delta_upper = candidate.full_mean_loss_upper_bound - reference.full_mean_loss_lower_bound;

    let mut sensitivity_surface = Vec::new();
    let sensitivity_status;
    if let (Some(candidate_observed_mean), Some(reference_observed_mean)) =
        (candidate.observed_mean_loss, reference.observed_mean_loss)
    {
        validate_delta_grid(
            "candidate",
            &input.sensitivity_grid.candidate_deltas,
            candidate_observed_mean,
        )?;
        validate_delta_grid(
            "reference",
            &input.sensitivity_grid.reference_deltas,
            reference_observed_mean,
        )?;
        let candidate_missing_fraction =
            1.0 - candidate.observed_outcome_fraction;
        let reference_missing_fraction =
            1.0 - reference.observed_outcome_fraction;
        for candidate_delta in &input.sensitivity_grid.candidate_deltas {
            let candidate_mean = candidate_observed_mean
                + candidate_missing_fraction * candidate_delta;
            for reference_delta in &input.sensitivity_grid.reference_deltas {
                let reference_mean = reference_observed_mean
                    + reference_missing_fraction * reference_delta;
                ensure!(
                    candidate_mean >= -1e-12 && candidate_mean <= 1.0 + 1e-12,
                    "candidate sensitivity mean falls outside [0, 1]"
                );
                ensure!(
                    reference_mean >= -1e-12 && reference_mean <= 1.0 + 1e-12,
                    "reference sensitivity mean falls outside [0, 1]"
                );
                sensitivity_surface.push(SensitivityPoint {
                    candidate_delta: *candidate_delta,
                    reference_delta: *reference_delta,
                    candidate_mean_loss: candidate_mean.clamp(0.0, 1.0),
                    reference_mean_loss: reference_mean.clamp(0.0, 1.0),
                    candidate_minus_reference_loss: candidate_mean - reference_mean,
                });
            }
        }
        sensitivity_status = "computed-from-preregistered-feasible-delta-grid".to_string();
    } else {
        // No observed cases means the selected-case mean, and therefore this delta parameterization,
        // is undefined for at least one arm. Keep the assumption-light bounds and say why.
        sensitivity_status = "unavailable-no-observed-outcomes-in-at-least-one-arm".to_string();
    }

    let sensitivity_grid_summary = summarize_sensitivity_grid(&sensitivity_surface);
    let observed_case_difference = candidate
        .observed_mean_loss
        .zip(reference.observed_mean_loss)
        .map(|(candidate_mean, reference_mean)| candidate_mean - reference_mean);
    let sensitivity_tipping_point =
        find_sensitivity_tipping_point(&sensitivity_surface, observed_case_difference);
    let mut result = MissingnessScore {
        metric_version: "informative-missingness-partial-identification-v1".to_string(),
        source_input_sha256: sha256_hex(source_bytes),
        result_sha256: None,
        dataset_id: input.dataset_id,
        scenario_id: input.scenario_id,
        comparison_track: input.comparison_track,
        evaluation_horizon_tick: input.evaluation_horizon_tick,
        candidate,
        reference,
        candidate_minus_reference_loss_lower_bound: delta_lower,
        candidate_minus_reference_loss_upper_bound: delta_upper,
        identification_interval_excludes_zero_in_candidate_favor: delta_upper < 0.0,
        observed_case_candidate_minus_reference_loss: observed_case_difference,
        sensitivity_surface,
        sensitivity_grid_summary,
        sensitivity_tipping_point,
        sensitivity_status,
        interpretation_guard: "Partial-identification statistic only; not a confidence interval, proof of measurement authenticity, causal identification, or SOTA claim.".to_string(),
    };
    // Digest of the canonical score with result_sha256=null, so its own digest is not circular.
    result.result_sha256 = None;
    let canonical = serde_json::to_vec(&result)?;
    result.result_sha256 = Some(sha256_hex(&canonical));
    Ok(result)
}

fn find_sensitivity_tipping_point(
    points: &[SensitivityPoint],
    observed_case_difference: Option<f64>,
) -> Option<SensitivityTippingPoint> {
    let baseline = observed_case_difference?;
    // A baseline tie has no observed ranking whose sign can be tipped.
    if baseline.abs() <= 1e-12 {
        return None;
    }
    points
        .iter()
        .filter(|point| {
            point.candidate_minus_reference_loss.abs() <= 1e-12
                || point.candidate_minus_reference_loss.is_sign_negative() != baseline.is_sign_negative()
        })
        .min_by(|left, right| {
            let left_radius = left.candidate_delta.abs().max(left.reference_delta.abs());
            let right_radius = right.candidate_delta.abs().max(right.reference_delta.abs());
            left_radius
                .total_cmp(&right_radius)
                .then_with(|| left.candidate_delta.total_cmp(&right.candidate_delta))
                .then_with(|| left.reference_delta.total_cmp(&right.reference_delta))
        })
        .map(|point| SensitivityTippingPoint {
            candidate_delta: point.candidate_delta,
            reference_delta: point.reference_delta,
            candidate_minus_reference_loss: point.candidate_minus_reference_loss,
            max_abs_sensitivity_delta: point.candidate_delta.abs().max(point.reference_delta.abs()),
        })
}

fn summarize_sensitivity_grid(points: &[SensitivityPoint]) -> SensitivityGridSummary {
    let candidate_favoring_point_count = points
        .iter()
        .filter(|point| point.candidate_minus_reference_loss < -1e-12)
        .count();
    let reference_favoring_point_count = points
        .iter()
        .filter(|point| point.candidate_minus_reference_loss > 1e-12)
        .count();
    let tie_point_count = points.len()
        .saturating_sub(candidate_favoring_point_count)
        .saturating_sub(reference_favoring_point_count);
    SensitivityGridSummary {
        grid_point_count: points.len(),
        candidate_favoring_point_count,
        reference_favoring_point_count,
        tie_point_count,
        minimum_candidate_minus_reference_loss: points
            .iter()
            .map(|point| point.candidate_minus_reference_loss)
            .min_by(f64::total_cmp),
        maximum_candidate_minus_reference_loss: points
            .iter()
            .map(|point| point.candidate_minus_reference_loss)
            .max_by(f64::total_cmp),
        candidate_favoring_at_every_registered_point: !points.is_empty()
            && candidate_favoring_point_count == points.len(),
        ranking_reverses_within_registered_grid: candidate_favoring_point_count > 0
            && reference_favoring_point_count > 0,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

fn validate_arm(arm: &MissingnessArmInput) -> anyhow::Result<()> {
    ensure!(!arm.name.trim().is_empty(), "arm name must not be empty");
    ensure!(arm.eligible_count > 0, "arm {} must have at least one eligible episode", arm.name);
    ensure!(
        arm.observed_losses.iter().all(|loss| loss.is_finite() && (0.0..=1.0).contains(loss)),
        "arm {} has a non-finite or out-of-range observed loss; expected [0, 1]",
        arm.name
    );
    let state_total = arm.observed_losses.len()
        .checked_add(arm.delayed_pending_count)
        .and_then(|n| n.checked_add(arm.right_censored_count))
        .and_then(|n| n.checked_add(arm.declared_missing_count))
        .and_then(|n| n.checked_add(arm.unknown_count))
        .context("outcome-state count overflow")?;
    ensure!(
        state_total == arm.eligible_count,
        "arm {} outcome-state counts total {} but eligible_count is {}",
        arm.name, state_total, arm.eligible_count
    );
    Ok(())
}

fn summarize_arm(arm: &MissingnessArmInput) -> anyhow::Result<ArmMissingnessSummary> {
    validate_arm(arm)?;
    let observed_count = arm.observed_losses.len();
    let observed_sum: f64 = arm.observed_losses.iter().sum();
    let eligible = arm.eligible_count as f64;
    let observed_outcome_fraction = observed_count as f64 / eligible;
    let observed_mean_loss = if observed_count == 0 {
        None
    } else {
        Some(observed_sum / observed_count as f64)
    };
    // For loss in [0,1], unobserved losses can contribute anywhere from 0 to 1 each.
    let full_mean_loss_lower_bound = observed_sum / eligible;
    let full_mean_loss_upper_bound =
        (observed_sum + (arm.eligible_count - observed_count) as f64) / eligible;

    Ok(ArmMissingnessSummary {
        name: arm.name.clone(),
        eligible_count: arm.eligible_count,
        observed_count,
        observed_outcome_fraction,
        observed_mean_loss,
        delayed_pending_count: arm.delayed_pending_count,
        right_censored_count: arm.right_censored_count,
        declared_missing_count: arm.declared_missing_count,
        unknown_count: arm.unknown_count,
        full_mean_loss_lower_bound,
        full_mean_loss_upper_bound,
    })
}

fn validate_delta_grid(name: &str, deltas: &[f64], observed_mean: f64) -> anyhow::Result<()> {
    let mut unique = BTreeSet::new();
    for delta in deltas {
        ensure!(delta.is_finite(), "{} sensitivity delta must be finite", name);
        ensure!(
            *delta >= -observed_mean - 1e-12 && *delta <= 1.0 - observed_mean + 1e-12,
            "{} sensitivity delta {} is infeasible for observed mean {}",
            name, delta, observed_mean
        );
        ensure!(unique.insert(delta.to_bits()), "{} sensitivity grid contains a duplicate value", name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(name: &str, eligible: usize, losses: Vec<f64>, pending: usize, censored: usize, missing: usize, unknown: usize) -> MissingnessArmInput {
        MissingnessArmInput {
            name: name.to_string(),
            eligible_count: eligible,
            observed_losses: losses,
            delayed_pending_count: pending,
            right_censored_count: censored,
            declared_missing_count: missing,
            unknown_count: unknown,
        }
    }

    fn input(candidate: MissingnessArmInput, reference: MissingnessArmInput) -> MissingnessInput {
        MissingnessInput {
            schema_version: 1,
            dataset_id: "fixture-dataset".to_string(),
            scenario_id: "fixture-scenario".to_string(),
            comparison_track: "delayed-observed-outcome".to_string(),
            evaluation_horizon_tick: 64,
            candidate,
            reference,
            sensitivity_grid: SensitivityGrid {
                candidate_deltas: vec![-0.1, 0.0, 0.25, 0.75],
                reference_deltas: vec![-0.2, 0.0, 0.2, 0.6],
            },
        }
    }

    #[test]
    fn fully_observed_bounds_collapse_to_the_observed_mean() {
        let candidate = arm("candidate", 4, vec![0.1, 0.2, 0.3, 0.4], 0, 0, 0, 0);
        let reference = arm("reference", 4, vec![0.2, 0.2, 0.2, 0.2], 0, 0, 0, 0);
        let scored = score(input(candidate, reference), b"fully-observed").unwrap();
        assert!((scored.candidate.full_mean_loss_lower_bound - 0.25).abs() < 1e-12);
        assert!((scored.candidate.full_mean_loss_upper_bound - 0.25).abs() < 1e-12);
        assert!((scored.candidate_minus_reference_loss_lower_bound - 0.05).abs() < 1e-12);
        assert!((scored.candidate_minus_reference_loss_upper_bound - 0.05).abs() < 1e-12);
        assert!(!scored.identification_interval_excludes_zero_in_candidate_favor);
    }

    #[test]
    fn ranking_reversal_fixture_does_not_turn_observed_case_win_into_superiority() {
        // Observed cases favor the candidate (0.1 vs 0.3), but 8/10 candidate outcomes are missing.
        let candidate = arm("candidate", 10, vec![0.1, 0.1], 0, 0, 8, 0);
        let reference = arm("reference", 10, vec![0.3; 10], 0, 0, 0, 0);
        let scored = score(input(candidate, reference), b"ranking-reversal").unwrap();
        assert!(scored.candidate.observed_mean_loss.unwrap() < scored.reference.observed_mean_loss.unwrap());
        // Evaluator-only truth assumes the eight unobserved candidate losses are 0.9.
        // This hidden truth is not a policy input; it demonstrates that the observed-case
        // ranking can reverse while the identification interval remains honest.
        let candidate_full_truth = (0.1 + 0.1 + 8.0 * 0.9) / 10.0;
        assert!(candidate_full_truth > scored.reference.observed_mean_loss.unwrap());
        assert!(scored.candidate.full_mean_loss_lower_bound <= candidate_full_truth);
        assert!(candidate_full_truth <= scored.candidate.full_mean_loss_upper_bound);
        assert!(scored.candidate_minus_reference_loss_lower_bound < 0.0);
        assert!(scored.candidate_minus_reference_loss_upper_bound > 0.0);
        assert!(!scored.identification_interval_excludes_zero_in_candidate_favor);
        assert_eq!(scored.sensitivity_surface.len(), 16);
        assert!(scored.sensitivity_grid_summary.ranking_reverses_within_registered_grid);
        assert!(!scored.sensitivity_grid_summary.candidate_favoring_at_every_registered_point);
        assert!(scored.sensitivity_grid_summary.candidate_favoring_point_count > 0);
        assert!(scored.sensitivity_grid_summary.reference_favoring_point_count > 0);
        assert!(scored.sensitivity_tipping_point.is_some());
        let tipping = scored.sensitivity_tipping_point.as_ref().unwrap();
        assert!((tipping.max_abs_sensitivity_delta - 0.25).abs() < 1e-12);
        assert!(tipping.candidate_minus_reference_loss.abs() < 1e-12);
        assert!(scored.observed_case_candidate_minus_reference_loss.unwrap() < 0.0);
    }

    #[test]
    fn fully_observed_candidate_can_favor_entire_registered_grid() {
        let candidate = arm("candidate", 5, vec![0.1; 5], 0, 0, 0, 0);
        let reference = arm("reference", 5, vec![0.3; 5], 0, 0, 0, 0);
        let scored = score(input(candidate, reference), b"grid-stable-win").unwrap();
        assert!(scored.sensitivity_grid_summary.candidate_favoring_at_every_registered_point);
        assert!(!scored.sensitivity_grid_summary.ranking_reverses_within_registered_grid);
        assert!(scored.sensitivity_tipping_point.is_none());
        assert_eq!(scored.sensitivity_grid_summary.grid_point_count, 16);
        assert_eq!(scored.sensitivity_grid_summary.candidate_favoring_point_count, 16);
    }

    #[test]
    fn no_observed_outcomes_produce_full_bounds_and_no_fake_observed_mean() {
        let candidate = arm("candidate", 5, vec![], 1, 1, 2, 1);
        let reference = arm("reference", 5, vec![0.4; 5], 0, 0, 0, 0);
        let scored = score(input(candidate, reference), b"all-missing-candidate").unwrap();
        assert_eq!(scored.candidate.observed_mean_loss, None);
        assert_eq!(scored.candidate.full_mean_loss_lower_bound, 0.0);
        assert_eq!(scored.candidate.full_mean_loss_upper_bound, 1.0);
        assert!(scored.sensitivity_surface.is_empty());
        assert!(scored.sensitivity_status.contains("no-observed-outcomes"));
    }

    #[test]
    fn worst_case_bounds_contain_all_binary_hidden_outcome_completions() {
        let partially_observed = arm("candidate", 6, vec![0.2, 0.8], 0, 0, 4, 0);
        let summary = summarize_arm(&partially_observed).unwrap();
        // Four missing outcomes have 2^4 binary completions. Since each hidden loss
        // lies in [0,1], every completion must lie inside the registered mean bounds.
        for mask in 0_u32..(1_u32 << 4) {
            let hidden_sum: f64 = (0..4)
                .map(|bit| if mask & (1 << bit) == 0 { 0.0 } else { 1.0 })
                .sum();
            let full_mean = (1.0 + hidden_sum) / 6.0;
            assert!(
                summary.full_mean_loss_lower_bound <= full_mean + 1e-12
                    && full_mean <= summary.full_mean_loss_upper_bound + 1e-12,
                "bounds do not contain hidden-outcome completion {mask}: {full_mean}"
            );
        }
    }

    #[test]
    fn rejects_outcome_state_count_mismatch() {
        let candidate = arm("candidate", 3, vec![0.1, 0.2], 0, 0, 0, 0);
        let reference = arm("reference", 2, vec![0.2, 0.2], 0, 0, 0, 0);
        let err = score(input(candidate, reference), b"bad-counts").unwrap_err();
        assert!(err.to_string().contains("outcome-state counts total"));
    }

    #[test]
    fn rejects_non_finite_and_out_of_range_losses() {
        for loss in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            let candidate = arm("candidate", 1, vec![loss], 0, 0, 0, 0);
            let reference = arm("reference", 1, vec![0.2], 0, 0, 0, 0);
            assert!(score(input(candidate, reference), b"bad-loss").is_err());
        }
    }

    #[test]
    fn rejects_infeasible_sensitivity_values() {
        let candidate = arm("candidate", 2, vec![0.95, 0.95], 0, 0, 0, 0);
        let reference = arm("reference", 2, vec![0.2, 0.2], 0, 0, 0, 0);
        let mut bad = input(candidate, reference);
        bad.sensitivity_grid.candidate_deltas = vec![-0.95, 0.0, 0.25];
        assert!(score(bad, b"bad-delta").unwrap_err().to_string().contains("infeasible"));
    }
}
