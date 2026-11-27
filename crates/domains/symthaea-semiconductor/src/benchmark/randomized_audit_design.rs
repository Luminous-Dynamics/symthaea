// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed-size randomized audit sampling combined with risk-targeted Bernoulli
//! metrology selection.
//!
//! The audit sample is SRSWOR using externally supplied continuous priority
//! draws. Targeted Bernoulli selection is an independent channel. First- and
//! second-order union inclusion probabilities are exact under those declared
//! random-design assumptions. Pairwise probabilities are computed on demand
//! to avoid materializing an O(N^2) matrix for large wafer populations.
//!
//! This is a sampling design and finite-population HT estimator. It is not
//! SOWRC's martingale-mixture risk-control theorem. K1 qualification failures
//! remain a separate observation process and suppress this clean-label HT
//! helper rather than being "fixed" by the audit probability.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const RANDOMIZED_AUDIT_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.randomized-audit-input.v1";
pub const RANDOMIZED_AUDIT_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.randomized-audit-report.v1";
pub const RANDOMIZED_AUDIT_HT_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.randomized-audit-ht-input.v1";
pub const RANDOMIZED_AUDIT_HT_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.randomized-audit-ht-report.v1";
pub const RANDOMIZED_AUDIT_PAIRWISE_SCHEMA_V1: &str =
    "luminous.semi-bench.randomized-audit-pairwise-srswor-bernoulli-union.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomizedAuditPopulationUnitV1 {
    pub sample_id: String,
    pub measurement_event_id: String,
    pub context_key: Option<String>,
    /// Inclusion probability under the separate risk-targeted Bernoulli channel.
    pub target_selection_probability: f64,
    /// Exogenous independent uniform draw for the targeted channel, in [0,1).
    pub target_uniform_draw: f64,
    /// Exogenous continuous uniform priority for SRSWOR audit selection, in [0,1).
    pub audit_priority: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomizedAuditDesignInputV1 {
    pub schema: String,
    pub design_id: String,
    pub population_size: u64,
    /// Fixed number of units selected by the audit channel, 1 <= K <= N.
    pub audit_capacity: u64,
    pub population_frame_sha256: String,
    pub target_policy_sha256: String,
    pub random_draw_manifest_sha256: String,
    pub random_source_receipt_sha256: String,
    pub target_policy_receipt_sha256: String,
    pub units: Vec<RandomizedAuditPopulationUnitV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomizedAuditUnitSelectionV1 {
    pub sample_id: String,
    pub measurement_event_id: String,
    pub context_key: Option<String>,
    pub target_selection_probability: f64,
    pub audit_selection_probability: f64,
    pub union_selection_probability: f64,
    pub selected_by_audit: bool,
    pub selected_by_target: bool,
    pub selected_union: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomizedAuditDesignReportV1 {
    pub schema: String,
    pub design_id: String,
    pub population_size: u64,
    pub audit_capacity: u64,
    pub audit_selection_probability: f64,
    pub pairwise_schema: String,
    pub population_frame_sha256: String,
    pub target_policy_sha256: String,
    pub random_draw_manifest_sha256: String,
    pub random_source_receipt_sha256: String,
    pub target_policy_receipt_sha256: String,
    pub audit_sample_ids: Vec<String>,
    pub targeted_sample_ids: Vec<String>,
    pub selected_sample_ids: Vec<String>,
    pub units: Vec<RandomizedAuditUnitSelectionV1>,
    pub expected_union_sample_size: f64,
    pub realized_union_sample_size: usize,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomizedAuditObservedLossV1 {
    pub sample_id: String,
    /// This helper accepts only fully qualified bounded losses.
    pub qualified_loss: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomizedAuditHtInputV1 {
    pub schema: String,
    pub design_report: RandomizedAuditDesignReportV1,
    pub observed_losses: Vec<RandomizedAuditObservedLossV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomizedAuditHtReportV1 {
    pub schema: String,
    pub design_id: String,
    pub design_receipt_sha256: String,
    pub population_size: u64,
    pub observed_selected_units: usize,
    pub horvitz_thompson_mean: f64,
    /// Unbiased design-based variance estimator. It is deliberately not
    /// clamped to zero; an unbiased variance estimator can be negative.
    pub horvitz_thompson_variance_raw: f64,
    pub effective_sample_size: f64,
    pub maximum_inverse_weight: f64,
    pub minimum_first_order_inclusion_probability: f64,
    pub qualification_missingness_adjusted: bool,
    pub theorem_or_guarantee_claimed: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct RandomizedAuditErrorV1(pub String);

#[derive(Serialize)]
struct FrameRow<'a> {
    sample_id: &'a str,
    measurement_event_id: &'a str,
    context_key: &'a Option<String>,
}

#[derive(Serialize)]
struct TargetPolicyRow<'a> {
    sample_id: &'a str,
    target_selection_probability: f64,
}

#[derive(Serialize)]
struct RandomDrawRow<'a> {
    sample_id: &'a str,
    target_uniform_draw: f64,
    audit_priority: f64,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), RandomizedAuditErrorV1> {
    if value.trim().is_empty() {
        return Err(RandomizedAuditErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, RandomizedAuditErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        RandomizedAuditErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn sorted_units(
    input: &RandomizedAuditDesignInputV1,
) -> Result<Vec<RandomizedAuditPopulationUnitV1>, RandomizedAuditErrorV1> {
    let mut units = input.units.clone();
    units.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let mut sample_ids = BTreeSet::new();
    let mut event_ids = BTreeSet::new();
    let mut priorities = BTreeSet::new();
    for unit in &units {
        require_text(&unit.sample_id, "unit.sample_id")?;
        require_text(&unit.measurement_event_id, "unit.measurement_event_id")?;
        if unit
            .context_key
            .as_deref()
            .is_some_and(|context| context.trim().is_empty())
        {
            return Err(RandomizedAuditErrorV1(
                "context key cannot be empty when supplied".into(),
            ));
        }
        if !sample_ids.insert(unit.sample_id.as_str())
            || !event_ids.insert(unit.measurement_event_id.as_str())
        {
            return Err(RandomizedAuditErrorV1(
                "duplicate sample or measurement-event identity in audit frame".into(),
            ));
        }
        if !unit.target_selection_probability.is_finite()
            || !(0.0..=1.0).contains(&unit.target_selection_probability)
            || !unit.target_uniform_draw.is_finite()
            || !(0.0..1.0).contains(&unit.target_uniform_draw)
            || !unit.audit_priority.is_finite()
            || !(0.0..1.0).contains(&unit.audit_priority)
        {
            return Err(RandomizedAuditErrorV1(format!(
                "unit {} has invalid propensity or random draw",
                unit.sample_id
            )));
        }
        if !priorities.insert(unit.audit_priority.to_bits()) {
            return Err(RandomizedAuditErrorV1(
                "audit priority ties are rejected to preserve deterministic SRSWOR ordering"
                    .into(),
            ));
        }
    }
    Ok(units)
}

pub fn randomized_audit_population_frame_digest_v1(
    units: &[RandomizedAuditPopulationUnitV1],
) -> Result<String, RandomizedAuditErrorV1> {
    let mut rows: Vec<FrameRow<'_>> = units
        .iter()
        .map(|unit| FrameRow {
            sample_id: &unit.sample_id,
            measurement_event_id: &unit.measurement_event_id,
            context_key: &unit.context_key,
        })
        .collect();
    rows.sort_by(|left, right| left.sample_id.cmp(right.sample_id));
    json_sha256(&rows)
}

pub fn randomized_audit_target_policy_digest_v1(
    units: &[RandomizedAuditPopulationUnitV1],
) -> Result<String, RandomizedAuditErrorV1> {
    let mut rows: Vec<TargetPolicyRow<'_>> = units
        .iter()
        .map(|unit| TargetPolicyRow {
            sample_id: &unit.sample_id,
            target_selection_probability: unit.target_selection_probability,
        })
        .collect();
    rows.sort_by(|left, right| left.sample_id.cmp(right.sample_id));
    json_sha256(&rows)
}

pub fn randomized_audit_draw_manifest_digest_v1(
    units: &[RandomizedAuditPopulationUnitV1],
) -> Result<String, RandomizedAuditErrorV1> {
    let mut rows: Vec<RandomDrawRow<'_>> = units
        .iter()
        .map(|unit| RandomDrawRow {
            sample_id: &unit.sample_id,
            target_uniform_draw: unit.target_uniform_draw,
            audit_priority: unit.audit_priority,
        })
        .collect();
    rows.sort_by(|left, right| left.sample_id.cmp(right.sample_id));
    json_sha256(&rows)
}

fn audit_union_probability(
    population_size: u64,
    audit_capacity: u64,
    target_probability: f64,
) -> f64 {
    let audit = audit_capacity as f64 / population_size as f64;
    1.0 - (1.0 - audit) * (1.0 - target_probability)
}

/// Computes the exact second-order inclusion probability for two distinct
/// units under SRSWOR audit selection plus independent targeted Bernoulli draws.
fn audit_union_pair_probability(
    population_size: u64,
    audit_capacity: u64,
    left_target_probability: f64,
    right_target_probability: f64,
) -> f64 {
    let n = population_size as f64;
    let k = audit_capacity as f64;
    let no_audit_left = 1.0 - k / n;
    let no_audit_right = no_audit_left;
    let remaining_after_audit = population_size - audit_capacity;
    let no_audit_pair = if remaining_after_audit < 2 {
        0.0
    } else {
        (remaining_after_audit * (remaining_after_audit - 1)) as f64 / (n * (n - 1.0))
    };
    let no_left = no_audit_left * (1.0 - left_target_probability);
    let no_right = no_audit_right * (1.0 - right_target_probability);
    let no_either = no_audit_pair
        * (1.0 - left_target_probability)
        * (1.0 - right_target_probability);
    (1.0 - no_left - no_right + no_either).clamp(0.0, 1.0)
}

fn design_report_digest(
    report: &RandomizedAuditDesignReportV1,
) -> Result<String, RandomizedAuditErrorV1> {
    let mut canonical = report.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn validate_design_report(
    report: &RandomizedAuditDesignReportV1,
) -> Result<(), RandomizedAuditErrorV1> {
    if report.schema != RANDOMIZED_AUDIT_REPORT_SCHEMA_V1
        || report.pairwise_schema != RANDOMIZED_AUDIT_PAIRWISE_SCHEMA_V1
    {
        return Err(RandomizedAuditErrorV1(
            "randomized audit report schema is unsupported".into(),
        ));
    }
    if !valid_sha256(&report.population_frame_sha256)
        || !valid_sha256(&report.target_policy_sha256)
        || !valid_sha256(&report.random_draw_manifest_sha256)
        || !valid_sha256(&report.random_source_receipt_sha256)
        || !valid_sha256(&report.target_policy_receipt_sha256)
        || design_report_digest(report)? != report.receipt_sha256
    {
        return Err(RandomizedAuditErrorV1(
            "randomized audit report digest or evidence reference is invalid".into(),
        ));
    }
    Ok(())
}

/// Builds the fixed-size audit sample plus risk-targeted Bernoulli union.
///
/// The audit priority order is a uniform random permutation only when the
/// supplied priorities are independently generated continuous uniforms. This
/// function records and validates the draws; the caller remains responsible
/// for the random-source receipt and its provenance.
pub fn create_randomized_audit_design_v1(
    input: &RandomizedAuditDesignInputV1,
) -> Result<RandomizedAuditDesignReportV1, RandomizedAuditErrorV1> {
    if input.schema != RANDOMIZED_AUDIT_INPUT_SCHEMA_V1 {
        return Err(RandomizedAuditErrorV1(
            "randomized audit input schema is unsupported".into(),
        ));
    }
    require_text(&input.design_id, "input.design_id")?;
    if input.population_size == 0
        || input.units.len() as u64 != input.population_size
        || input.audit_capacity == 0
        || input.audit_capacity > input.population_size
    {
        return Err(RandomizedAuditErrorV1(
            "population size must match units and audit capacity must be in 1..=N".into(),
        ));
    }
    for (field, value) in [
        ("population_frame_sha256", &input.population_frame_sha256),
        ("target_policy_sha256", &input.target_policy_sha256),
        ("random_draw_manifest_sha256", &input.random_draw_manifest_sha256),
        ("random_source_receipt_sha256", &input.random_source_receipt_sha256),
        ("target_policy_receipt_sha256", &input.target_policy_receipt_sha256),
    ] {
        if !valid_sha256(value) {
            return Err(RandomizedAuditErrorV1(format!(
                "{field}: expected lowercase SHA-256"
            )));
        }
    }

    let units = sorted_units(input)?;
    if randomized_audit_population_frame_digest_v1(&units)? != input.population_frame_sha256
        || randomized_audit_target_policy_digest_v1(&units)? != input.target_policy_sha256
        || randomized_audit_draw_manifest_digest_v1(&units)? != input.random_draw_manifest_sha256
    {
        return Err(RandomizedAuditErrorV1(
            "frozen frame, target policy or random draw manifest digest mismatch".into(),
        ));
    }

    let mut audit_order = units.clone();
    audit_order.sort_by(|left, right| {
        left.audit_priority
            .total_cmp(&right.audit_priority)
            .then_with(|| left.sample_id.cmp(&right.sample_id))
    });
    let audit_selected: BTreeSet<String> = audit_order
        .iter()
        .take(input.audit_capacity as usize)
        .map(|unit| unit.sample_id.clone())
        .collect();

    let audit_probability = input.audit_capacity as f64 / input.population_size as f64;
    let mut selections = Vec::with_capacity(units.len());
    let mut audit_sample_ids = Vec::new();
    let mut targeted_sample_ids = Vec::new();
    let mut selected_sample_ids = Vec::new();
    let mut expected_union_sample_size = 0.0;

    for unit in &units {
        let audit = audit_selected.contains(&unit.sample_id);
        let target = unit.target_uniform_draw < unit.target_selection_probability;
        let selected = audit || target;
        let union_probability = audit_union_probability(
            input.population_size,
            input.audit_capacity,
            unit.target_selection_probability,
        );
        expected_union_sample_size += union_probability;
        if audit {
            audit_sample_ids.push(unit.sample_id.clone());
        }
        if target {
            targeted_sample_ids.push(unit.sample_id.clone());
        }
        if selected {
            selected_sample_ids.push(unit.sample_id.clone());
        }
        selections.push(RandomizedAuditUnitSelectionV1 {
            sample_id: unit.sample_id.clone(),
            measurement_event_id: unit.measurement_event_id.clone(),
            context_key: unit.context_key.clone(),
            target_selection_probability: unit.target_selection_probability,
            audit_selection_probability: audit_probability,
            union_selection_probability: union_probability,
            selected_by_audit: audit,
            selected_by_target: target,
            selected_union: selected,
        });
    }

    let mut report = RandomizedAuditDesignReportV1 {
        schema: RANDOMIZED_AUDIT_REPORT_SCHEMA_V1.into(),
        design_id: input.design_id.clone(),
        population_size: input.population_size,
        audit_capacity: input.audit_capacity,
        audit_selection_probability: audit_probability,
        pairwise_schema: RANDOMIZED_AUDIT_PAIRWISE_SCHEMA_V1.into(),
        population_frame_sha256: input.population_frame_sha256.clone(),
        target_policy_sha256: input.target_policy_sha256.clone(),
        random_draw_manifest_sha256: input.random_draw_manifest_sha256.clone(),
        random_source_receipt_sha256: input.random_source_receipt_sha256.clone(),
        target_policy_receipt_sha256: input.target_policy_receipt_sha256.clone(),
        audit_sample_ids,
        targeted_sample_ids,
        selected_sample_ids: selected_sample_ids.clone(),
        units: selections,
        expected_union_sample_size,
        realized_union_sample_size: selected_sample_ids.len(),
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = design_report_digest(&report)?;
    Ok(report)
}

/// Returns pi_ij for two distinct population units. For i==j, this returns pi_i.
/// Pairwise probabilities are calculated on demand rather than storing O(N^2)
/// entries in the design receipt.
pub fn randomized_audit_pairwise_inclusion_probability_v1(
    report: &RandomizedAuditDesignReportV1,
    left_sample_id: &str,
    right_sample_id: &str,
) -> Result<f64, RandomizedAuditErrorV1> {
    validate_design_report(report)?;
    let left = report
        .units
        .iter()
        .find(|unit| unit.sample_id == left_sample_id)
        .ok_or_else(|| RandomizedAuditErrorV1("left sample is not in frozen frame".into()))?;
    let right = report
        .units
        .iter()
        .find(|unit| unit.sample_id == right_sample_id)
        .ok_or_else(|| RandomizedAuditErrorV1("right sample is not in frozen frame".into()))?;
    if left_sample_id == right_sample_id {
        return Ok(left.union_selection_probability);
    }
    Ok(audit_union_pair_probability(
        report.population_size,
        report.audit_capacity,
        left.target_selection_probability,
        right.target_selection_probability,
    ))
}

fn ht_report_digest(report: &RandomizedAuditHtReportV1) -> Result<String, RandomizedAuditErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

/// Computes the finite-population Horvitz-Thompson mean and unbiased
/// design-based variance estimator for a complete set of qualified losses
/// on the realized union sample. Qualification nonresponse is deliberately
/// not adjusted here; use the separate population-risk bounds/IPW contract.
pub fn compute_randomized_audit_ht_v1(
    input: &RandomizedAuditHtInputV1,
) -> Result<RandomizedAuditHtReportV1, RandomizedAuditErrorV1> {
    if input.schema != RANDOMIZED_AUDIT_HT_INPUT_SCHEMA_V1 {
        return Err(RandomizedAuditErrorV1(
            "randomized audit HT input schema is unsupported".into(),
        ));
    }
    validate_design_report(&input.design_report)?;
    let selected_ids: BTreeSet<&str> = input
        .design_report
        .selected_sample_ids
        .iter()
        .map(String::as_str)
        .collect();
    let mut losses = BTreeMap::new();
    for outcome in &input.observed_losses {
        require_text(&outcome.sample_id, "observed_loss.sample_id")?;
        if !selected_ids.contains(outcome.sample_id.as_str()) {
            return Err(RandomizedAuditErrorV1(
                "loss supplied for a sample outside the realized selected union".into(),
            ));
        }
        if !outcome.qualified_loss.is_finite() || !(0.0..=1.0).contains(&outcome.qualified_loss) {
            return Err(RandomizedAuditErrorV1(
                "qualified loss must be finite and in [0,1]".into(),
            ));
        }
        if losses
            .insert(outcome.sample_id.as_str(), outcome.qualified_loss)
            .is_some()
        {
            return Err(RandomizedAuditErrorV1(
                "duplicate qualified loss for selected sample".into(),
            ));
        }
    }
    if losses.len() != selected_ids.len() || selected_ids.is_empty() {
        return Err(RandomizedAuditErrorV1(
            "HT estimate requires one qualified loss for every selected union unit".into(),
        ));
    }

    let unit_by_id: BTreeMap<&str, &RandomizedAuditUnitSelectionV1> = input
        .design_report
        .units
        .iter()
        .map(|unit| (unit.sample_id.as_str(), unit))
        .collect();
    let selected: Vec<&str> = input
        .design_report
        .selected_sample_ids
        .iter()
        .map(String::as_str)
        .collect();
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut squared_weight_sum = 0.0;
    let mut maximum_inverse_weight = 0.0_f64;
    let mut minimum_probability = 1.0_f64;

    for sample_id in &selected {
        let unit = unit_by_id[sample_id];
        let probability = unit.union_selection_probability;
        if probability <= 0.0 {
            return Err(RandomizedAuditErrorV1(
                "selected unit has zero first-order inclusion probability".into(),
            ));
        }
        let inverse = 1.0 / probability;
        let loss = losses[sample_id];
        weighted_sum += inverse * loss;
        weight_sum += inverse;
        squared_weight_sum += inverse * inverse;
        maximum_inverse_weight = maximum_inverse_weight.max(inverse);
        minimum_probability = minimum_probability.min(probability);
    }

    let mut variance_sum = 0.0;
    for left_id in &selected {
        let left = unit_by_id[left_id];
        let left_probability = left.union_selection_probability;
        let left_term = losses[left_id] / left_probability;
        for right_id in &selected {
            let right = unit_by_id[right_id];
            let right_probability = right.union_selection_probability;
            let pair_probability = randomized_audit_pairwise_inclusion_probability_v1(
                &input.design_report,
                left_id,
                right_id,
            )?;
            if pair_probability <= 0.0 {
                return Err(RandomizedAuditErrorV1(
                    "realized selected pair has zero second-order inclusion probability".into(),
                ));
            }
            let right_term = losses[right_id] / right_probability;
            variance_sum += (pair_probability - left_probability * right_probability)
                / pair_probability
                * left_term
                * right_term;
        }
    }

    let population = input.design_report.population_size as f64;
    let mut report = RandomizedAuditHtReportV1 {
        schema: RANDOMIZED_AUDIT_HT_REPORT_SCHEMA_V1.into(),
        design_id: input.design_report.design_id.clone(),
        design_receipt_sha256: input.design_report.receipt_sha256.clone(),
        population_size: input.design_report.population_size,
        observed_selected_units: selected.len(),
        horvitz_thompson_mean: weighted_sum / population,
        horvitz_thompson_variance_raw: variance_sum / (population * population),
        effective_sample_size: weight_sum * weight_sum / squared_weight_sum,
        maximum_inverse_weight,
        minimum_first_order_inclusion_probability: minimum_probability,
        qualification_missingness_adjusted: false,
        theorem_or_guarantee_claimed: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = ht_report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn units(
        target_probabilities: &[f64],
        audit_priorities: &[f64],
        target_draws: &[f64],
    ) -> Vec<RandomizedAuditPopulationUnitV1> {
        target_probabilities
            .iter()
            .enumerate()
            .map(|(index, probability)| RandomizedAuditPopulationUnitV1 {
                sample_id: format!("sample-{index:02}"),
                measurement_event_id: format!("event-{index:02}"),
                context_key: Some(if index % 2 == 0 { "eqp-a/ch-1" } else { "eqp-b/ch-2" }.into()),
                target_selection_probability: *probability,
                target_uniform_draw: target_draws[index],
                audit_priority: audit_priorities[index],
            })
            .collect()
    }

    fn input(
        units: Vec<RandomizedAuditPopulationUnitV1>,
        audit_capacity: u64,
    ) -> RandomizedAuditDesignInputV1 {
        RandomizedAuditDesignInputV1 {
            schema: RANDOMIZED_AUDIT_INPUT_SCHEMA_V1.into(),
            design_id: "audit-design-v1".into(),
            population_size: units.len() as u64,
            audit_capacity,
            population_frame_sha256: randomized_audit_population_frame_digest_v1(&units).unwrap(),
            target_policy_sha256: randomized_audit_target_policy_digest_v1(&units).unwrap(),
            random_draw_manifest_sha256: randomized_audit_draw_manifest_digest_v1(&units).unwrap(),
            random_source_receipt_sha256: digest('a'),
            target_policy_receipt_sha256: digest('b'),
            units,
        }
    }

    #[test]
    fn fixed_audit_capacity_selects_exactly_the_k_lowest_priorities() {
        let population = units(
            &[0.0, 0.0, 0.0, 0.0],
            &[0.7, 0.1, 0.9, 0.3],
            &[0.2, 0.2, 0.2, 0.2],
        );
        let report = create_randomized_audit_design_v1(&input(population, 1)).unwrap();
        assert_eq!(report.audit_sample_ids, vec!["sample-01"]);
        assert_eq!(report.selected_sample_ids, vec!["sample-01"]);
        assert_eq!(report.realized_union_sample_size, 1);
    }

    #[test]
    fn audit_census_has_unit_first_and_pairwise_inclusion_probabilities() {
        let population = units(&[0.0, 0.3, 1.0], &[0.7, 0.1, 0.9], &[0.2, 0.2, 0.2]);
        let report = create_randomized_audit_design_v1(&input(population, 3)).unwrap();
        assert_eq!(report.audit_sample_ids.len(), 3);
        for unit in &report.units {
            assert_eq!(unit.union_selection_probability, 1.0);
        }
        assert_eq!(
            randomized_audit_pairwise_inclusion_probability_v1(
                &report,
                "sample-00",
                "sample-02"
            )
            .unwrap(),
            1.0
        );
    }

    #[test]
    fn zero_capacity_and_capacity_above_population_are_rejected() {
        let population = units(&[0.0, 0.0], &[0.2, 0.8], &[0.1, 0.9]);
        assert!(create_randomized_audit_design_v1(&input(population.clone(), 0)).is_err());
        assert!(create_randomized_audit_design_v1(&input(population, 3)).is_err());
    }

    #[test]
    fn input_order_does_not_change_frame_policy_draw_or_design_receipts() {
        let original = units(
            &[0.1, 0.5, 0.0, 0.9],
            &[0.7, 0.1, 0.9, 0.3],
            &[0.2, 0.4, 0.8, 0.1],
        );
        let first = create_randomized_audit_design_v1(&input(original.clone(), 2)).unwrap();
        let mut permuted = original;
        permuted.reverse();
        let second = create_randomized_audit_design_v1(&input(permuted, 2)).unwrap();
        assert_eq!(first.population_frame_sha256, second.population_frame_sha256);
        assert_eq!(first.target_policy_sha256, second.target_policy_sha256);
        assert_eq!(first.random_draw_manifest_sha256, second.random_draw_manifest_sha256);
        assert_eq!(first.receipt_sha256, second.receipt_sha256);
    }

    #[test]
    fn targeted_audit_and_union_selection_channels_are_distinct() {
        let population = units(
            &[1.0, 0.0, 0.5, 0.0],
            &[0.9, 0.1, 0.8, 0.2],
            &[0.4, 0.4, 0.7, 0.4],
        );
        let report = create_randomized_audit_design_v1(&input(population, 1)).unwrap();
        let first = report.units.iter().find(|unit| unit.sample_id == "sample-00").unwrap();
        assert!(first.selected_by_audit);
        assert!(first.selected_by_target);
        assert!(first.selected_union);
        let third = report.units.iter().find(|unit| unit.sample_id == "sample-02").unwrap();
        assert!(!third.selected_by_audit);
        assert!(!third.selected_by_target);
        assert!(!third.selected_union);
    }

    #[test]
    fn first_order_union_probability_matches_scalar_oracle() {
        let population = units(
            &[0.0, 0.25, 0.5, 1.0],
            &[0.7, 0.1, 0.9, 0.3],
            &[0.2, 0.4, 0.8, 0.1],
        );
        let report = create_randomized_audit_design_v1(&input(population, 2)).unwrap();
        for unit in &report.units {
            let expected = 1.0 - (1.0 - 0.5) * (1.0 - unit.target_selection_probability);
            assert!((unit.union_selection_probability - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn pairwise_union_probability_matches_independent_scalar_formula() {
        let population = units(
            &[0.0, 0.25, 0.5, 1.0],
            &[0.7, 0.1, 0.9, 0.3],
            &[0.2, 0.4, 0.8, 0.1],
        );
        let report = create_randomized_audit_design_v1(&input(population, 1)).unwrap();
        let left = 0.25_f64;
        let right = 0.50_f64;
        let no_audit_single = 0.75_f64;
        let no_audit_pair = 0.5_f64;
        let expected = 1.0
            - no_audit_single * (1.0 - left)
            - no_audit_single * (1.0 - right)
            + no_audit_pair * (1.0 - left) * (1.0 - right);
        let actual = randomized_audit_pairwise_inclusion_probability_v1(
            &report,
            "sample-01",
            "sample-02",
        )
        .unwrap();
        assert!((actual - expected).abs() < 1.0e-12);
    }

    #[test]
    fn single_unit_population_has_no_pairwise_distinct_record() {
        let population = units(&[0.25], &[0.6], &[0.7]);
        let report = create_randomized_audit_design_v1(&input(population, 1)).unwrap();
        assert_eq!(
            randomized_audit_pairwise_inclusion_probability_v1(
                &report,
                "sample-00",
                "sample-00"
            )
            .unwrap(),
            1.0
        );
        assert!(
            randomized_audit_pairwise_inclusion_probability_v1(
                &report,
                "sample-00",
                "missing"
            )
            .is_err()
        );
    }

    #[test]
    fn zero_and_one_target_probabilities_have_exact_union_support() {
        let population = units(&[0.0, 1.0], &[0.8, 0.2], &[0.9, 0.9]);
        let report = create_randomized_audit_design_v1(&input(population, 1)).unwrap();
        let zero = report.units.iter().find(|unit| unit.sample_id == "sample-00").unwrap();
        let one = report.units.iter().find(|unit| unit.sample_id == "sample-01").unwrap();
        assert_eq!(zero.union_selection_probability, 0.5);
        assert_eq!(one.union_selection_probability, 1.0);
        assert!(one.selected_by_target);
    }

    #[test]
    fn duplicate_ids_tied_priorities_and_invalid_draws_are_rejected() {
        let mut duplicate = units(&[0.0, 0.0], &[0.2, 0.8], &[0.1, 0.9]);
        duplicate[1].sample_id = duplicate[0].sample_id.clone();
        assert!(create_randomized_audit_design_v1(&input(duplicate, 1)).is_err());

        let mut tied = units(&[0.0, 0.0], &[0.5, 0.5], &[0.1, 0.9]);
        assert!(create_randomized_audit_design_v1(&input(tied.clone(), 1)).is_err());

        tied[1].audit_priority = 0.7;
        tied[0].target_uniform_draw = 1.0;
        assert!(create_randomized_audit_design_v1(&input(tied, 1)).is_err());
    }

    #[test]
    fn horvitz_thompson_mean_and_variance_match_known_srswor_answer() {
        let population = units(&[0.0, 0.0, 0.0, 0.0], &[0.1, 0.2, 0.8, 0.9], &[0.5; 4]);
        let design = create_randomized_audit_design_v1(&input(population, 2)).unwrap();
        assert_eq!(design.audit_sample_ids, vec!["sample-00", "sample-01"]);
        let report = compute_randomized_audit_ht_v1(&RandomizedAuditHtInputV1 {
            schema: RANDOMIZED_AUDIT_HT_INPUT_SCHEMA_V1.into(),
            design_report: design,
            observed_losses: vec![
                RandomizedAuditObservedLossV1 { sample_id: "sample-00".into(), qualified_loss: 0.0 },
                RandomizedAuditObservedLossV1 { sample_id: "sample-01".into(), qualified_loss: 1.0 },
            ],
        }).unwrap();
        assert!((report.horvitz_thompson_mean - 0.5).abs() < 1.0e-12);
        assert!((report.horvitz_thompson_variance_raw - 0.125).abs() < 1.0e-12);
        assert_eq!(report.effective_sample_size, 2.0);
        assert!(!report.qualification_missingness_adjusted);
        assert!(!report.theorem_or_guarantee_claimed);
    }

    #[test]
    fn ht_refuses_unqualified_or_incomplete_selected_losses() {
        let population = units(&[0.0, 0.0, 0.0], &[0.1, 0.2, 0.8], &[0.5; 3]);
        let design = create_randomized_audit_design_v1(&input(population, 2)).unwrap();
        let first_id = design.selected_sample_ids[0].clone();
        assert!(compute_randomized_audit_ht_v1(&RandomizedAuditHtInputV1 {
            schema: RANDOMIZED_AUDIT_HT_INPUT_SCHEMA_V1.into(),
            design_report: design,
            observed_losses: vec![RandomizedAuditObservedLossV1 {
                sample_id: first_id,
                qualified_loss: 0.4,
            }],
        }).is_err());
    }

    #[test]
    fn tampered_frame_draw_or_target_policy_receipt_is_rejected() {
        let population = units(&[0.2, 0.3, 0.4], &[0.2, 0.5, 0.8], &[0.1, 0.5, 0.9]);
        let mut design_input = input(population, 1);
        design_input.population_frame_sha256 = digest('f');
        assert!(create_randomized_audit_design_v1(&design_input).is_err());
        let mut design_input = input(
            units(&[0.2, 0.3, 0.4], &[0.2, 0.5, 0.8], &[0.1, 0.5, 0.9]),
            1,
        );
        design_input.target_policy_receipt_sha256 = "invalid".into();
        assert!(create_randomized_audit_design_v1(&design_input).is_err());
    }
}
