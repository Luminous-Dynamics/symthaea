//! Design-based audit of candidate-generation recall beyond the frozen proposal pool.
//!
//! This evaluator independently samples complete primary sampling units (PSUs) from
//! a declared spatial/neuron search frame. It estimates errors covered and missed by
//! the frozen candidate generator; it never feeds audit outcomes back to ranking.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-candidate-coverage-audit",
    version,
    about = "Estimate candidate-generation recall with an independent probability sample of the search frame"
)]
struct Args {
    /// Exact frozen proposal input previously passed to connectome-query-impact.
    #[arg(long)]
    proposals: PathBuf,
    /// Exact frozen ranking report for the same proposal input.
    #[arg(long)]
    ranking_report: PathBuf,
    /// Exact sampling protocol artifact committed before outcome collection.
    #[arg(long)]
    sampling_protocol: PathBuf,
    /// Independent exhaustive-search outcomes from the probability sample.
    #[arg(long)]
    audit: PathBuf,
    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ProposalInput {
    schema_version: u32,
    candidates: Vec<ProposalCandidate>,
}

#[derive(Debug, Deserialize)]
struct ProposalCandidate {
    candidate_id: String,
}

#[derive(Debug, Deserialize)]
struct RankingInput {
    schema_version: u32,
    input_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AuditInput {
    schema_version: u32,
    source_input_sha256: String,
    source_ranking_report_sha256: String,
    sampling_protocol_sha256: String,
    sample_seed: String,
    sample_seed_commitment_sha256: String,
    error_families: Vec<String>,
    strata: Vec<StratumPlan>,
    population_frame: Vec<PopulationPsu>,
    sampled_units: Vec<SampledPsuOutcome>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StratumPlan {
    stratum_id: String,
    sampled_psu_count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PopulationPsu {
    psu_id: String,
    stratum_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SampledPsuOutcome {
    psu_id: String,
    exhaustive_search_completed: bool,
    audit_panel_person_minutes: f64,
    errors: Vec<VerifiedError>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VerifiedError {
    verified_error_id: String,
    error_family: String,
    impact_weight: f64,
    proposal_candidate_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Estimate {
    total: f64,
    design_variance: f64,
    standard_error: f64,
    normal_approx_95_interval: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize)]
struct RatioEstimate {
    numerator: Estimate,
    denominator: Estimate,
    ratio: Option<f64>,
    delta_method_design_variance: Option<f64>,
    standard_error: Option<f64>,
    normal_approx_95_interval: Option<[f64; 2]>,
    ratio_scope: &'static str,
}

#[derive(Debug, Serialize)]
struct StratumEstimate {
    stratum_id: String,
    population_psu_count: usize,
    sampled_psu_count: usize,
    psu_inclusion_probability: f64,
    sampled_error_count: usize,
    sampled_covered_error_count: usize,
    sampled_uncovered_error_count: usize,
    estimated_total_errors: Estimate,
    estimated_covered_errors: Estimate,
    estimated_uncovered_errors: Estimate,
    estimated_total_impact_weight: Estimate,
    estimated_covered_impact_weight: Estimate,
    estimated_uncovered_impact_weight: Estimate,
    estimated_audit_panel_person_minutes: Estimate,
}

#[derive(Debug, Serialize)]
struct FamilyCoverage {
    estimated_total_errors: Estimate,
    estimated_covered_errors: Estimate,
    estimated_uncovered_errors: Estimate,
    candidate_generation_recall: RatioEstimate,
    estimated_total_impact_weight: Estimate,
    estimated_covered_impact_weight: Estimate,
    estimated_uncovered_impact_weight: Estimate,
    impact_weighted_candidate_generation_recall: RatioEstimate,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u32,
    tool: &'static str,
    proposal_input_sha256: String,
    ranking_report_sha256: String,
    sampling_protocol_sha256: String,
    sampling_frame_sha256: String,
    audit_input_sha256: String,
    sampled_outcomes_sha256: String,
    sample_seed_commitment_sha256: String,
    candidate_pool_size: usize,
    population_psu_count: usize,
    sampled_psu_count: usize,
    sampled_error_count: usize,
    sampled_covered_error_count: usize,
    sampled_uncovered_error_count: usize,
    estimated_total_errors: Estimate,
    estimated_covered_errors: Estimate,
    estimated_uncovered_errors: Estimate,
    estimated_total_impact_weight: Estimate,
    estimated_covered_impact_weight: Estimate,
    estimated_uncovered_impact_weight: Estimate,
    estimated_audit_panel_person_minutes: Estimate,
    candidate_generation_recall: RatioEstimate,
    impact_weighted_candidate_generation_recall: RatioEstimate,
    error_family_coverage: BTreeMap<String, FamilyCoverage>,
    strata: Vec<StratumEstimate>,
    warnings: Vec<String>,
    status: &'static str,
    scope_note: &'static str,
}

#[derive(Debug, Clone, Default)]
struct PsuTotals {
    errors: f64,
    covered_errors: f64,
    uncovered_errors: f64,
    impact: f64,
    covered_impact: f64,
    uncovered_impact: f64,
    panel_minutes: f64,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn frame_sha256(frame: &[PopulationPsu]) -> String {
    let mut rows: Vec<(String, String)> = frame
        .iter()
        .map(|row| (row.psu_id.clone(), row.stratum_id.clone()))
        .collect();
    rows.sort();
    let mut hasher = Sha256::new();
    for (psu_id, stratum_id) in rows {
        hasher.update((psu_id.len() as u64).to_be_bytes());
        hasher.update(psu_id.as_bytes());
        hasher.update((stratum_id.len() as u64).to_be_bytes());
        hasher.update(stratum_id.as_bytes());
    }
    hex::encode(hasher.finalize())
}

fn sampled_outcomes_sha256(outcomes: &[SampledPsuOutcome]) -> Result<String, String> {
    let mut canonical = outcomes.to_vec();
    canonical.sort_by(|left, right| left.psu_id.cmp(&right.psu_id));
    for unit in &mut canonical {
        unit.errors.sort_by(|left, right| left.verified_error_id.cmp(&right.verified_error_id));
        for error in &mut unit.errors {
            error.proposal_candidate_ids.sort();
        }
    }
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| format!("could not serialize canonical sampled outcomes: {error}"))?;
    Ok(sha256(&bytes))
}

fn psu_rank(seed: &str, stratum_id: &str, psu_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-connectome-candidate-coverage-v1\0");
    hasher.update(seed.as_bytes());
    hasher.update(b"\0");
    hasher.update(stratum_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(psu_id.as_bytes());
    hex::encode(hasher.finalize())
}

fn expected_sample(
    seed: &str,
    frame: &BTreeMap<String, Vec<String>>,
    plans: &[StratumPlan],
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let mut selected = BTreeMap::new();
    for plan in plans {
        let psus = frame
            .get(&plan.stratum_id)
            .ok_or_else(|| format!("sampling plan references absent stratum {}", plan.stratum_id))?;
        let population_count = psus.len();
        let sample_count = plan.sampled_psu_count;
        if sample_count == 0 || sample_count > population_count {
            return Err(format!(
                "stratum {} sampled_psu_count must be in 1..={}",
                plan.stratum_id, population_count
            ));
        }
        if sample_count < population_count && sample_count < 2 {
            return Err(format!(
                "stratum {} needs at least two sampled PSUs for a non-census variance estimate",
                plan.stratum_id
            ));
        }

        let mut ranked: Vec<(String, String)> = psus
            .iter()
            .map(|psu_id| (psu_rank(seed, &plan.stratum_id, psu_id), psu_id.clone()))
            .collect();
        ranked.sort();
        selected.insert(
            plan.stratum_id.clone(),
            ranked
                .into_iter()
                .take(sample_count)
                .map(|(_, psu_id)| psu_id)
                .collect(),
        );
    }
    Ok(selected)
}

fn validate(
    proposals: &ProposalInput,
    ranking: &RankingInput,
    audit: &AuditInput,
    proposal_hash: &str,
    ranking_hash: &str,
    protocol_hash: &str,
) -> Result<
    (
        BTreeMap<String, Vec<String>>,
        BTreeMap<String, SampledPsuOutcome>,
        BTreeMap<String, BTreeSet<String>>,
    ),
    String,
> {
    if proposals.schema_version != 1 || ranking.schema_version != 1 || audit.schema_version != 1 {
        return Err("proposal, ranking, and coverage-audit schema_version values must all be 1".into());
    }
    if ranking.input_sha256 != proposal_hash || audit.source_input_sha256 != proposal_hash {
        return Err("ranking/audit proposal digest does not match exact proposal bytes".into());
    }
    if audit.source_ranking_report_sha256 != ranking_hash {
        return Err("audit ranking digest does not match exact frozen ranking-report bytes".into());
    }
    if audit.sampling_protocol_sha256 != protocol_hash {
        return Err("audit protocol digest does not match exact sampling-protocol bytes".into());
    }
    if !is_lower_hex(proposal_hash, 64)
        || !is_lower_hex(ranking_hash, 64)
        || !is_lower_hex(protocol_hash, 64)
        || !is_lower_hex(&audit.sample_seed_commitment_sha256, 64)
    {
        return Err("proposal/ranking/protocol/seed SHA-256 value is malformed".into());
    }
    if audit.sample_seed.len() < 16 {
        return Err("sample_seed must contain at least 16 bytes".into());
    }
    if sha256(audit.sample_seed.as_bytes()) != audit.sample_seed_commitment_sha256 {
        return Err("sample_seed does not match its preregistered SHA-256 commitment".into());
    }

    let mut candidate_ids = BTreeSet::new();
    for candidate in &proposals.candidates {
        if candidate.candidate_id.trim().is_empty()
            || !candidate_ids.insert(candidate.candidate_id.clone())
        {
            return Err("frozen proposal candidate IDs must be non-empty and unique".into());
        }
    }
    if candidate_ids.is_empty() {
        return Err("frozen proposal candidate pool must be non-empty".into());
    }

    let declared_families: BTreeSet<String> = audit.error_families.iter().cloned().collect();
    if audit.error_families.is_empty()
        || declared_families.len() != audit.error_families.len()
        || audit.error_families.iter().any(|family| family.trim().is_empty())
    {
        return Err("error_families must be non-empty, non-empty-string, unique, and preregistered".into());
    }

    let mut frame = BTreeMap::<String, Vec<String>>::new();
    let mut frame_ids = BTreeSet::new();
    for psu in &audit.population_frame {
        if psu.psu_id.trim().is_empty() || psu.stratum_id.trim().is_empty() {
            return Err("population-frame PSU and stratum IDs must be non-empty".into());
        }
        if !frame_ids.insert(psu.psu_id.clone()) {
            return Err(format!("duplicate PSU in population frame: {}", psu.psu_id));
        }
        frame.entry(psu.stratum_id.clone()).or_default().push(psu.psu_id.clone());
    }
    if frame_ids.is_empty() {
        return Err("population frame must contain at least one PSU".into());
    }

    let mut plans = audit.strata.clone();
    plans.sort_by(|left, right| left.stratum_id.cmp(&right.stratum_id));
    let mut plan_ids = BTreeSet::new();
    for plan in &plans {
        if plan.stratum_id.trim().is_empty() || !plan_ids.insert(plan.stratum_id.clone()) {
            return Err("sampling strata must have non-empty, unique IDs".into());
        }
        if !frame.contains_key(&plan.stratum_id) {
            return Err(format!("sampling plan references absent stratum {}", plan.stratum_id));
        }
    }
    let frame_strata: BTreeSet<String> = frame.keys().cloned().collect();
    if frame_strata != plan_ids {
        return Err("sampling plan strata must exactly match the complete population-frame strata".into());
    }

    let selected = expected_sample(&audit.sample_seed, &frame, &plans)?;
    let expected_ids: BTreeSet<String> = selected.values().flat_map(|ids| ids.iter().cloned()).collect();

    let mut outcomes = BTreeMap::new();
    let mut error_ids = BTreeSet::new();
    for unit in &audit.sampled_units {
        if unit.psu_id.trim().is_empty() || !expected_ids.contains(&unit.psu_id) {
            return Err(format!(
                "sampled outcome references an empty or unselected PSU: {}",
                unit.psu_id
            ));
        }
        if !unit.exhaustive_search_completed {
            return Err(format!(
                "PSU {} is not attested as exhaustively searched",
                unit.psu_id
            ));
        }
        if !unit.audit_panel_person_minutes.is_finite() || unit.audit_panel_person_minutes <= 0.0 {
            return Err(format!("PSU {} has invalid audit_panel_person_minutes", unit.psu_id));
        }
        if outcomes.insert(unit.psu_id.clone(), unit.clone()).is_some() {
            return Err(format!("duplicate sampled outcome for PSU {}", unit.psu_id));
        }
        for error in &unit.errors {
            if error.verified_error_id.trim().is_empty()
                || !error_ids.insert(error.verified_error_id.clone())
            {
                return Err(format!(
                    "verified error IDs must be non-empty and globally unique; duplicate/empty ID {}",
                    error.verified_error_id
                ));
            }
            if error.error_family.trim().is_empty()
                || !declared_families.contains(&error.error_family)
                || !error.impact_weight.is_finite()
                || error.impact_weight < 0.0
            {
                return Err(format!("verified error {} has invalid family or impact weight", error.verified_error_id));
            }
            let mut mapped_candidates = BTreeSet::new();
            for candidate_id in &error.proposal_candidate_ids {
                if candidate_id.trim().is_empty()
                    || !mapped_candidates.insert(candidate_id.clone())
                {
                    return Err(format!(
                        "verified error {} has an empty or duplicate proposal candidate reference",
                        error.verified_error_id
                    ));
                }
                if !candidate_ids.contains(candidate_id) {
                    return Err(format!(
                        "verified error {} references proposal candidate {} absent from the frozen pool",
                        error.verified_error_id, candidate_id
                    ));
                }
            }
        }
    }

    let observed_ids: BTreeSet<String> = outcomes.keys().cloned().collect();
    if observed_ids != expected_ids {
        let missing: Vec<String> = expected_ids.difference(&observed_ids).cloned().collect();
        let extra: Vec<String> = observed_ids.difference(&expected_ids).cloned().collect();
        return Err(format!(
            "sampled outcomes must exactly cover selected PSUs; missing=[{}], extra=[{}]",
            missing.join(","), extra.join(",")
        ));
    }

    Ok((frame, outcomes, selected))
}

fn psu_totals(outcome: &SampledPsuOutcome) -> PsuTotals {
    let mut totals = PsuTotals {
        panel_minutes: outcome.audit_panel_person_minutes,
        ..PsuTotals::default()
    };
    for error in &outcome.errors {
        totals.errors += 1.0;
        totals.impact += error.impact_weight;
        if error.proposal_candidate_ids.is_empty() {
            totals.uncovered_errors += 1.0;
            totals.uncovered_impact += error.impact_weight;
        } else {
            totals.covered_errors += 1.0;
            totals.covered_impact += error.impact_weight;
        }
    }
    totals
}

fn sample_variance(values: &[f64]) -> Result<f64, String> {
    if values.len() < 2 {
        return Err("sample variance requires at least two sampled PSUs".into());
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    if !variance.is_finite() {
        return Err("sample variance overflowed or became non-finite".into());
    }
    Ok(variance)
}

fn estimate(values: &[f64], population_psus: usize) -> Result<Estimate, String> {
    let sample_psus = values.len();
    if sample_psus == 0 || sample_psus > population_psus {
        return Err("invalid sample/population PSU counts for total estimator".into());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("total estimator received a non-finite PSU value".into());
    }
    let scale = population_psus as f64 / sample_psus as f64;
    let total = scale * values.iter().sum::<f64>();
    let variance = if sample_psus == population_psus {
        0.0
    } else {
        let sampling_fraction = sample_psus as f64 / population_psus as f64;
        (population_psus as f64).powi(2)
            * (1.0 - sampling_fraction)
            * sample_variance(values)?
            / sample_psus as f64
    };
    if !total.is_finite() || !variance.is_finite() || variance < 0.0 {
        return Err("Horvitz-Thompson estimate or design variance is non-finite".into());
    }
    let standard_error = variance.sqrt();
    let normal_approx_95_interval = if sample_psus == population_psus {
        Some([total, total])
    } else if sample_psus >= 30 {
        Some([total - 1.96 * standard_error, total + 1.96 * standard_error])
    } else {
        None
    };
    Ok(Estimate {
        total,
        design_variance: variance,
        standard_error,
        normal_approx_95_interval,
    })
}

fn combine_estimates(estimates: &[Estimate]) -> Estimate {
    let total = estimates.iter().map(|value| value.total).sum::<f64>();
    let variance = estimates.iter().map(|value| value.design_variance).sum::<f64>();
    let standard_error = variance.sqrt();
    let interval = estimates
        .iter()
        .all(|value| value.normal_approx_95_interval.is_some())
        .then_some([total - 1.96 * standard_error, total + 1.96 * standard_error]);
    Estimate {
        total,
        design_variance: variance,
        standard_error,
        normal_approx_95_interval: interval,
    }
}

fn estimate_metric<F>(
    plans: &[StratumPlan],
    frame: &BTreeMap<String, Vec<String>>,
    outcomes: &BTreeMap<String, SampledPsuOutcome>,
    selected: &BTreeMap<String, BTreeSet<String>>,
    metric: F,
) -> Result<(Estimate, Vec<Estimate>), String>
where
    F: Fn(&SampledPsuOutcome) -> f64,
{
    let mut by_stratum = Vec::new();
    for plan in plans {
        let values: Vec<f64> = selected[&plan.stratum_id]
            .iter()
            .map(|psu_id| metric(&outcomes[psu_id]))
            .collect();
        by_stratum.push(estimate(&values, frame[&plan.stratum_id].len())?);
    }
    Ok((combine_estimates(&by_stratum), by_stratum))
}

fn ratio_estimate<N, D>(
    plans: &[StratumPlan],
    frame: &BTreeMap<String, Vec<String>>,
    outcomes: &BTreeMap<String, SampledPsuOutcome>,
    selected: &BTreeMap<String, BTreeSet<String>>,
    numerator_metric: N,
    denominator_metric: D,
) -> Result<RatioEstimate, String>
where
    N: Fn(&SampledPsuOutcome) -> f64,
    D: Fn(&SampledPsuOutcome) -> f64,
{
    let (numerator, _) = estimate_metric(plans, frame, outcomes, selected, &numerator_metric)?;
    let (denominator, _) = estimate_metric(plans, frame, outcomes, selected, &denominator_metric)?;
    let Some(ratio) = (denominator.total > 0.0).then_some(numerator.total / denominator.total) else {
        return Ok(RatioEstimate {
            numerator,
            denominator,
            ratio: None,
            delta_method_design_variance: None,
            standard_error: None,
            normal_approx_95_interval: None,
            ratio_scope: "finite declared sampling frame; null when estimated denominator is zero",
        });
    };
    if !ratio.is_finite() {
        return Err("ratio estimate is non-finite".into());
    }

    let mut ratio_variance_numerator = 0.0;
    let mut interval_available = true;
    for plan in plans {
        let population_psus = frame[&plan.stratum_id].len();
        let sample_ids = &selected[&plan.stratum_id];
        let sample_psus = sample_ids.len();
        if sample_psus < population_psus && sample_psus < 30 {
            interval_available = false;
        }
        if sample_psus == population_psus {
            continue;
        }
        let residuals: Vec<f64> = sample_ids
            .iter()
            .map(|psu_id| {
                numerator_metric(&outcomes[psu_id]) - ratio * denominator_metric(&outcomes[psu_id])
            })
            .collect();
        let sampling_fraction = sample_psus as f64 / population_psus as f64;
        ratio_variance_numerator += (population_psus as f64).powi(2)
            * (1.0 - sampling_fraction)
            * sample_variance(&residuals)?
            / sample_psus as f64;
    }

    let variance = ratio_variance_numerator / denominator.total.powi(2);
    if !variance.is_finite() || variance < 0.0 {
        return Err("ratio delta-method design variance is non-finite".into());
    }
    let standard_error = variance.sqrt();
    let interval = interval_available.then_some([
        (ratio - 1.96 * standard_error).max(0.0),
        (ratio + 1.96 * standard_error).min(1.0),
    ]);

    Ok(RatioEstimate {
        numerator,
        denominator,
        ratio: Some(ratio),
        delta_method_design_variance: Some(variance),
        standard_error: Some(standard_error),
        normal_approx_95_interval: interval,
        ratio_scope: "finite declared sampling frame; first-order Taylor linearization",
    })
}

fn evaluate(
    proposals: &ProposalInput,
    audit: &AuditInput,
    frame: &BTreeMap<String, Vec<String>>,
    outcomes: &BTreeMap<String, SampledPsuOutcome>,
    selected: &BTreeMap<String, BTreeSet<String>>,
    proposal_hash: String,
    ranking_hash: String,
    protocol_hash: String,
    audit_hash: String,
) -> Result<AuditReport, String> {
    let mut plans = audit.strata.clone();
    plans.sort_by(|left, right| left.stratum_id.cmp(&right.stratum_id));

    let error_count = |outcome: &SampledPsuOutcome| psu_totals(outcome).errors;
    let covered_count = |outcome: &SampledPsuOutcome| psu_totals(outcome).covered_errors;
    let uncovered_count = |outcome: &SampledPsuOutcome| psu_totals(outcome).uncovered_errors;
    let impact = |outcome: &SampledPsuOutcome| psu_totals(outcome).impact;
    let covered_impact = |outcome: &SampledPsuOutcome| psu_totals(outcome).covered_impact;
    let uncovered_impact = |outcome: &SampledPsuOutcome| psu_totals(outcome).uncovered_impact;
    let panel_minutes = |outcome: &SampledPsuOutcome| psu_totals(outcome).panel_minutes;

    let (estimated_total_errors, _) = estimate_metric(&plans, frame, outcomes, selected, error_count)?;
    let (estimated_covered_errors, _) = estimate_metric(&plans, frame, outcomes, selected, covered_count)?;
    let (estimated_uncovered_errors, _) = estimate_metric(&plans, frame, outcomes, selected, uncovered_count)?;
    let (estimated_total_impact_weight, _) = estimate_metric(&plans, frame, outcomes, selected, impact)?;
    let (estimated_covered_impact_weight, _) = estimate_metric(&plans, frame, outcomes, selected, covered_impact)?;
    let (estimated_uncovered_impact_weight, _) = estimate_metric(&plans, frame, outcomes, selected, uncovered_impact)?;
    let (estimated_audit_panel_person_minutes, _) = estimate_metric(&plans, frame, outcomes, selected, panel_minutes)?;

    let candidate_generation_recall = ratio_estimate(
        &plans, frame, outcomes, selected, covered_count, error_count,
    )?;
    let impact_weighted_candidate_generation_recall = ratio_estimate(
        &plans, frame, outcomes, selected, covered_impact, impact,
    )?;

    let mut sampled_error_count = 0usize;
    let mut sampled_covered_error_count = 0usize;
    let mut sampled_uncovered_error_count = 0usize;
    let mut sampled_psu_count = 0usize;
    let mut strata = Vec::new();

    for plan in &plans {
        let population_psus = frame[&plan.stratum_id].len();
        let selected_psus = &selected[&plan.stratum_id];
        let sample_psu_count = selected_psus.len();
        let inclusion_probability = sample_psu_count as f64 / population_psus as f64;
        let mut totals = Vec::new();
        let mut stratum_errors = 0usize;
        let mut stratum_covered = 0usize;
        let mut stratum_uncovered = 0usize;

        for psu_id in selected_psus {
            let outcome = &outcomes[psu_id];
            let psu = psu_totals(outcome);
            stratum_errors += outcome.errors.len();
            stratum_covered += psu.covered_errors as usize;
            stratum_uncovered += psu.uncovered_errors as usize;
            totals.push(psu);
        }

        sampled_error_count += stratum_errors;
        sampled_covered_error_count += stratum_covered;
        sampled_uncovered_error_count += stratum_uncovered;
        sampled_psu_count += sample_psu_count;

        let collect = |metric: fn(&PsuTotals) -> f64| totals.iter().map(metric).collect::<Vec<_>>();
        let total_errors = estimate(&collect(|x| x.errors), population_psus)?;
        let covered_errors = estimate(&collect(|x| x.covered_errors), population_psus)?;
        let uncovered_errors = estimate(&collect(|x| x.uncovered_errors), population_psus)?;
        let total_impact = estimate(&collect(|x| x.impact), population_psus)?;
        let covered_impact = estimate(&collect(|x| x.covered_impact), population_psus)?;
        let uncovered_impact = estimate(&collect(|x| x.uncovered_impact), population_psus)?;
        let audit_minutes = estimate(&collect(|x| x.panel_minutes), population_psus)?;

        strata.push(StratumEstimate {
            stratum_id: plan.stratum_id.clone(),
            population_psu_count: population_psus,
            sampled_psu_count: sample_psu_count,
            psu_inclusion_probability: inclusion_probability,
            sampled_error_count: stratum_errors,
            sampled_covered_error_count: stratum_covered,
            sampled_uncovered_error_count: stratum_uncovered,
            estimated_total_errors: total_errors,
            estimated_covered_errors: covered_errors,
            estimated_uncovered_errors: uncovered_errors,
            estimated_total_impact_weight: total_impact,
            estimated_covered_impact_weight: covered_impact,
            estimated_uncovered_impact_weight: uncovered_impact,
            estimated_audit_panel_person_minutes: audit_minutes,
        });
    }

    let families: BTreeSet<String> = audit.error_families.iter().cloned().collect();
    let mut error_family_coverage = BTreeMap::new();
    for family in families {
        let family_count = |outcome: &SampledPsuOutcome| {
            outcome.errors.iter().filter(|error| error.error_family == family).count() as f64
        };
        let family_covered = |outcome: &SampledPsuOutcome| {
            outcome.errors.iter().filter(|error| {
                error.error_family == family && !error.proposal_candidate_ids.is_empty()
            }).count() as f64
        };
        let family_uncovered = |outcome: &SampledPsuOutcome| {
            outcome.errors.iter().filter(|error| {
                error.error_family == family && error.proposal_candidate_ids.is_empty()
            }).count() as f64
        };
        let family_impact = |outcome: &SampledPsuOutcome| {
            outcome.errors.iter().filter(|error| error.error_family == family)
                .map(|error| error.impact_weight).sum::<f64>()
        };
        let family_covered_impact = |outcome: &SampledPsuOutcome| {
            outcome.errors.iter().filter(|error| {
                error.error_family == family && !error.proposal_candidate_ids.is_empty()
            }).map(|error| error.impact_weight).sum::<f64>()
        };
        let family_uncovered_impact = |outcome: &SampledPsuOutcome| {
            outcome.errors.iter().filter(|error| {
                error.error_family == family && error.proposal_candidate_ids.is_empty()
            }).map(|error| error.impact_weight).sum::<f64>()
        };

        let (family_total, _) = estimate_metric(&plans, frame, outcomes, selected, family_count)?;
        let (family_covered_total, _) = estimate_metric(&plans, frame, outcomes, selected, family_covered)?;
        let (family_uncovered_total, _) = estimate_metric(&plans, frame, outcomes, selected, family_uncovered)?;
        let (family_total_impact, _) = estimate_metric(&plans, frame, outcomes, selected, family_impact)?;
        let (family_covered_impact, _) = estimate_metric(&plans, frame, outcomes, selected, family_covered_impact)?;
        let (family_uncovered_impact, _) = estimate_metric(&plans, frame, outcomes, selected, family_uncovered_impact)?;
        let family_recall = ratio_estimate(
            &plans, frame, outcomes, selected, family_covered, family_count,
        )?;
        let family_impact_recall = ratio_estimate(
            &plans, frame, outcomes, selected, family_covered_impact, family_impact,
        )?;

        error_family_coverage.insert(family.clone(), FamilyCoverage {
            estimated_total_errors: family_total,
            estimated_covered_errors: family_covered_total,
            estimated_uncovered_errors: family_uncovered_total,
            candidate_generation_recall: family_recall,
            estimated_total_impact_weight: family_total_impact,
            estimated_covered_impact_weight: family_covered_impact,
            estimated_uncovered_impact_weight: family_uncovered_impact,
            impact_weighted_candidate_generation_recall: family_impact_recall,
        });
    }

    let sampled_outcomes_digest = sampled_outcomes_sha256(&audit.sampled_units)?;
    let warnings = vec![
        "Estimates cover only the complete, declared finite search frame; they are not whole-connectome recall.".to_string(),
        "Exhaustive-search attestations and proposal-to-error coverage mappings require independent human protocol enforcement.".to_string(),
        "Audit-panel person-minutes are separate from normal review-queue effort and must be included in all-in cost.".to_string(),
        "Ratio confidence intervals use first-order Taylor linearization and are withheld when any non-census stratum has fewer than 30 sampled PSUs.".to_string(),
        "Audit outcomes, sampled locations, uncovered-error identities, and impact weights must remain unavailable to proposal generation and ranking.".to_string(),
    ];

    Ok(AuditReport {
        schema_version: 1,
        tool: "connectome-candidate-coverage-audit",
        proposal_input_sha256: proposal_hash,
        ranking_report_sha256: ranking_hash,
        sampling_protocol_sha256: protocol_hash,
        sampling_frame_sha256: frame_sha256(&audit.population_frame),
        audit_input_sha256: audit_hash,
        sampled_outcomes_sha256: sampled_outcomes_digest,
        sample_seed_commitment_sha256: audit.sample_seed_commitment_sha256.clone(),
        candidate_pool_size: proposals.candidates.len(),
        population_psu_count: audit.population_frame.len(),
        sampled_psu_count,
        sampled_error_count,
        sampled_covered_error_count,
        sampled_uncovered_error_count,
        estimated_total_errors,
        estimated_covered_errors,
        estimated_uncovered_errors,
        estimated_total_impact_weight,
        estimated_covered_impact_weight,
        estimated_uncovered_impact_weight,
        estimated_audit_panel_person_minutes,
        candidate_generation_recall,
        impact_weighted_candidate_generation_recall,
        error_family_coverage,
        strata,
        warnings,
        status: "POST_HOC_CANDIDATE_COVERAGE_EVALUATED",
        scope_note: "Finite declared PSU search frame only; candidate-pool recall and whole-connectome recall are distinct estimands.",
    })
}

fn read_file(path: &PathBuf, label: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("could not read {label} {}: {error}", path.display()))
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let proposal_bytes = read_file(&args.proposals, "proposal input")?;
    let ranking_bytes = read_file(&args.ranking_report, "ranking report")?;
    let protocol_bytes = read_file(&args.sampling_protocol, "sampling protocol")?;
    let audit_bytes = read_file(&args.audit, "coverage audit")?;

    let proposal_hash = sha256(&proposal_bytes);
    let ranking_hash = sha256(&ranking_bytes);
    let protocol_hash = sha256(&protocol_bytes);
    let audit_hash = sha256(&audit_bytes);

    let proposals: ProposalInput = serde_json::from_slice(&proposal_bytes)
        .map_err(|error| format!("invalid proposal JSON: {error}"))?;
    let ranking: RankingInput = serde_json::from_slice(&ranking_bytes)
        .map_err(|error| format!("invalid ranking-report JSON: {error}"))?;
    let audit: AuditInput = serde_json::from_slice(&audit_bytes)
        .map_err(|error| format!("invalid coverage-audit JSON: {error}"))?;

    let (frame, outcomes, selected) = validate(
        &proposals, &ranking, &audit, &proposal_hash, &ranking_hash, &protocol_hash,
    )?;
    let report = evaluate(
        &proposals,
        &audit,
        &frame,
        &outcomes,
        &selected,
        proposal_hash,
        ranking_hash,
        protocol_hash,
        audit_hash,
    )?;
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("could not serialize candidate coverage report: {error}"))?;

    if let Some(path) = args.output {
        fs::write(&path, rendered)
            .map_err(|error| format!("could not write report {}: {error}", path.display()))?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("connectome-candidate-coverage-audit: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (ProposalInput, RankingInput, AuditInput, String, String, String) {
        let proposal_bytes = b"frozen proposal bytes".to_vec();
        let ranking_bytes = b"frozen ranking bytes".to_vec();
        let protocol_bytes = b"frozen coverage protocol".to_vec();
        let proposal_hash = sha256(&proposal_bytes);
        let ranking_hash = sha256(&ranking_bytes);
        let protocol_hash = sha256(&protocol_bytes);
        let seed = "test-candidate-coverage-seed-2026".to_string();

        let proposals = ProposalInput {
            schema_version: 1,
            candidates: vec![
                ProposalCandidate { candidate_id: "candidate-1".into() },
                ProposalCandidate { candidate_id: "candidate-2".into() },
            ],
        };
        let ranking = RankingInput { schema_version: 1, input_sha256: proposal_hash.clone() };
        let audit = AuditInput {
            schema_version: 1,
            source_input_sha256: proposal_hash.clone(),
            source_ranking_report_sha256: ranking_hash.clone(),
            sampling_protocol_sha256: protocol_hash.clone(),
            sample_seed_commitment_sha256: sha256(seed.as_bytes()),
            error_families: vec!["merge".into(), "synapse_missing".into(), "split".into(), "axon_fragment".into()],
            sample_seed: seed,
            strata: vec![
                StratumPlan { stratum_id: "region-a".into(), sampled_psu_count: 2 },
                StratumPlan { stratum_id: "region-b".into(), sampled_psu_count: 2 },
            ],
            population_frame: vec![
                PopulationPsu { psu_id: "psu-a1".into(), stratum_id: "region-a".into() },
                PopulationPsu { psu_id: "psu-a2".into(), stratum_id: "region-a".into() },
                PopulationPsu { psu_id: "psu-b1".into(), stratum_id: "region-b".into() },
                PopulationPsu { psu_id: "psu-b2".into(), stratum_id: "region-b".into() },
            ],
            sampled_units: vec![
                SampledPsuOutcome {
                    psu_id: "psu-a1".into(),
                    exhaustive_search_completed: true,
                    audit_panel_person_minutes: 10.0,
                    errors: vec![
                        VerifiedError {
                            verified_error_id: "error-1".into(),
                            error_family: "merge".into(),
                            impact_weight: 2.0,
                            proposal_candidate_ids: vec!["candidate-1".into()],
                        },
                        VerifiedError {
                            verified_error_id: "error-2".into(),
                            error_family: "synapse_missing".into(),
                            impact_weight: 1.0,
                            proposal_candidate_ids: vec![],
                        },
                    ],
                },
                SampledPsuOutcome {
                    psu_id: "psu-a2".into(),
                    exhaustive_search_completed: true,
                    audit_panel_person_minutes: 9.0,
                    errors: vec![],
                },
                SampledPsuOutcome {
                    psu_id: "psu-b1".into(),
                    exhaustive_search_completed: true,
                    audit_panel_person_minutes: 11.0,
                    errors: vec![VerifiedError {
                        verified_error_id: "error-3".into(),
                        error_family: "split".into(),
                        impact_weight: 3.0,
                        proposal_candidate_ids: vec!["candidate-2".into()],
                    }],
                },
                SampledPsuOutcome {
                    psu_id: "psu-b2".into(),
                    exhaustive_search_completed: true,
                    audit_panel_person_minutes: 8.0,
                    errors: vec![VerifiedError {
                        verified_error_id: "error-4".into(),
                        error_family: "merge".into(),
                        impact_weight: 8.0,
                        proposal_candidate_ids: vec![],
                    }],
                },
            ],
        };
        (proposals, ranking, audit, proposal_hash, ranking_hash, protocol_hash)
    }

    fn validated_fixture() -> (
        ProposalInput,
        AuditInput,
        BTreeMap<String, Vec<String>>,
        BTreeMap<String, SampledPsuOutcome>,
        BTreeMap<String, BTreeSet<String>>,
        String,
        String,
        String,
    ) {
        let (proposals, ranking, audit, proposal_hash, ranking_hash, protocol_hash) = fixture();
        let (frame, outcomes, selected) = validate(
            &proposals, &ranking, &audit, &proposal_hash, &ranking_hash, &protocol_hash,
        ).unwrap();
        (
            proposals, audit, frame, outcomes, selected,
            proposal_hash, ranking_hash, protocol_hash,
        )
    }

    #[test]
    fn census_recovers_exact_error_totals_and_candidate_recall() {
        let (proposals, audit, frame, outcomes, selected, proposal, ranking, protocol) = validated_fixture();
        let report = evaluate(
            &proposals, &audit, &frame, &outcomes, &selected,
            proposal, ranking, protocol, "audit".into(),
        ).unwrap();
        assert_eq!(report.estimated_total_errors.total, 4.0);
        assert_eq!(report.estimated_covered_errors.total, 2.0);
        assert_eq!(report.estimated_uncovered_errors.total, 2.0);
        assert!((report.candidate_generation_recall.ratio.unwrap() - 0.5).abs() < 1e-12);
        assert!((report.impact_weighted_candidate_generation_recall.ratio.unwrap() - (5.0 / 10.0)).abs() < 1e-12);
        assert_eq!(report.candidate_generation_recall.normal_approx_95_interval, Some([0.5, 0.5]));
        assert_eq!(report.sampled_psu_count, 4);
        assert_eq!(report.sampled_error_count, 4);
    }

    #[test]
    fn candidate_recall_counts_a_verified_error_once_even_with_multiple_proposals() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.sampled_units[0].errors[0].proposal_candidate_ids = vec![
            "candidate-1".into(), "candidate-2".into(),
        ];
        let (frame, outcomes, selected) = validate(
            &proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol,
        ).unwrap();
        let report = evaluate(
            &proposals, &audit, &frame, &outcomes, &selected,
            proposal, ranking_hash, protocol, "audit".into(),
        ).unwrap();
        assert_eq!(report.sampled_covered_error_count, 2);
        assert_eq!(report.estimated_covered_errors.total, 2.0);
    }

    #[test]
    fn empty_psu_is_retained_in_the_census_denominator() {
        let report = {
            let (proposals, audit, frame, outcomes, selected, proposal, ranking, protocol) = validated_fixture();
            evaluate(&proposals, &audit, &frame, &outcomes, &selected, proposal, ranking, protocol, "audit".into()).unwrap()
        };
        assert_eq!(report.population_psu_count, 4);
        assert_eq!(report.sampled_psu_count, 4);
        assert_eq!(report.estimated_total_errors.total, 4.0);
    }

    #[test]
    fn no_verified_errors_yields_undefined_recall_not_zero() {
        let (proposals, _, mut audit, proposal, ranking_hash, protocol) = fixture();
        for unit in &mut audit.sampled_units { unit.errors.clear(); }
        let ranking = RankingInput { schema_version: 1, input_sha256: proposal.clone() };
        let (frame, outcomes, selected) = validate(
            &proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol,
        ).unwrap();
        let report = evaluate(
            &proposals, &audit, &frame, &outcomes, &selected,
            proposal, ranking_hash, protocol, "audit".into(),
        ).unwrap();
        assert_eq!(report.candidate_generation_recall.ratio, None);
        assert_eq!(report.impact_weighted_candidate_generation_recall.ratio, None);
    }

    #[test]
    fn non_census_small_sample_withholds_normal_interval() {
        let values = vec![1.0, 2.0];
        let estimate = estimate(&values, 5).unwrap();
        assert!(estimate.design_variance > 0.0);
        assert_eq!(estimate.normal_approx_95_interval, None);
    }

    #[test]
    fn horvitz_thompson_total_is_exactly_unbiased_over_all_two_of_three_samples() {
        let population_total = 1.0 + 2.0 + 3.0;
        let samples = [vec![1.0, 2.0], vec![1.0, 3.0], vec![2.0, 3.0]];
        let mean_estimate = samples
            .iter()
            .map(|sample| estimate(sample, 3).unwrap().total)
            .sum::<f64>()
            / samples.len() as f64;
        assert!((mean_estimate - population_total).abs() < 1e-12);
    }

    #[test]
    fn horvitz_thompson_variance_uses_the_finite_population_correction() {
        let estimate = estimate(&[1.0, 3.0], 4).unwrap();
        // N^2 * (1 - n/N) * s^2 / n = 16 * 0.5 * 2 / 2 = 8.
        assert_eq!(estimate.total, 8.0);
        assert!((estimate.design_variance - 8.0).abs() < 1e-12);
    }

    #[test]
    fn census_total_has_zero_design_variance() {
        let estimate = estimate(&[1.0, 0.0, 3.0], 3).unwrap();
        assert_eq!(estimate.total, 4.0);
        assert_eq!(estimate.design_variance, 0.0);
        assert_eq!(estimate.normal_approx_95_interval, Some([4.0, 4.0]));
    }

    #[test]
    fn seed_commitment_tampering_is_rejected() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.sample_seed_commitment_sha256 = "0".repeat(64);
        assert!(validate(&proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol)
            .unwrap_err().contains("commitment"));
    }

    #[test]
    fn changed_ranking_report_is_rejected() {
        let (proposals, ranking, audit, proposal, ranking_hash, protocol) = fixture();
        assert!(validate(&proposals, &ranking, &audit, &proposal, &"f".repeat(64), &protocol)
            .unwrap_err().contains("ranking digest"));
        assert_ne!(ranking_hash, "f".repeat(64));
    }

    #[test]
    fn unknown_candidate_reference_is_rejected() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.sampled_units[0].errors[0].proposal_candidate_ids = vec!["not-in-pool".into()];
        assert!(validate(&proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol)
            .unwrap_err().contains("absent from the frozen pool"));
    }

    #[test]
    fn duplicate_verified_error_id_is_rejected() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.sampled_units[1].errors.push(audit.sampled_units[0].errors[0].clone());
        assert!(validate(&proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol)
            .unwrap_err().contains("globally unique"));
    }

    #[test]
    fn non_exhaustive_sampled_psu_is_rejected() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.sampled_units[0].exhaustive_search_completed = false;
        assert!(validate(&proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol)
            .unwrap_err().contains("not attested as exhaustively searched"));
    }

    #[test]
    fn missing_sampled_psu_is_rejected() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.sampled_units.pop();
        assert!(validate(&proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol)
            .unwrap_err().contains("exactly cover selected PSUs"));
    }

    #[test]
    fn outcome_for_unselected_psu_is_rejected() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.population_frame.push(PopulationPsu {
            psu_id: "psu-a3".into(),
            stratum_id: "region-a".into(),
        });
        audit.strata[0].sampled_psu_count = 2;

        let mut frame = BTreeMap::<String, Vec<String>>::new();
        for psu in &audit.population_frame {
            frame.entry(psu.stratum_id.clone()).or_default().push(psu.psu_id.clone());
        }
        let selected = expected_sample(&audit.sample_seed, &frame, &audit.strata).unwrap();
        audit.sampled_units.clear();
        for psu_id in selected.values().flat_map(|ids| ids.iter()) {
            audit.sampled_units.push(SampledPsuOutcome {
                psu_id: psu_id.clone(),
                exhaustive_search_completed: true,
                audit_panel_person_minutes: 1.0,
                errors: vec![],
            });
        }
        let selected_ids: BTreeSet<String> =
            selected.values().flat_map(|ids| ids.iter().cloned()).collect();
        let unselected_psu = frame["region-a"]
            .iter()
            .find(|psu_id| !selected_ids.contains(*psu_id))
            .unwrap()
            .clone();
        audit.sampled_units.push(SampledPsuOutcome {
            psu_id: unselected_psu,
            exhaustive_search_completed: true,
            audit_panel_person_minutes: 1.0,
            errors: vec![],
        });

        assert!(validate(&proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol)
            .unwrap_err().contains("unselected PSU"));
    }

    #[test]
    fn duplicate_population_psu_is_rejected() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.population_frame.push(audit.population_frame[0].clone());
        assert!(validate(&proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol)
            .unwrap_err().contains("duplicate PSU"));
    }

    #[test]
    fn sampling_plan_must_cover_exact_frame_strata() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.strata.pop();
        assert!(validate(&proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol)
            .unwrap_err().contains("exactly match"));
    }

    #[test]
    fn preregistered_error_family_absent_from_sample_is_retained_with_undefined_recall() {
        let (proposals, ranking, audit, proposal, ranking_hash, protocol) = fixture();
        let (frame, outcomes, selected) = validate(
            &proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol,
        ).unwrap();
        let report = evaluate(
            &proposals, &audit, &frame, &outcomes, &selected,
            proposal, ranking_hash, protocol, "audit".into(),
        ).unwrap();
        let rare = &report.error_family_coverage["axon_fragment"];
        assert_eq!(rare.estimated_total_errors.total, 0.0);
        assert_eq!(rare.candidate_generation_recall.ratio, None);
    }

    #[test]
    fn undeclared_error_family_is_rejected() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        audit.sampled_units[0].errors[0].error_family = "unregistered_family".into();
        assert!(validate(&proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol)
            .unwrap_err().contains("invalid family"));
    }

    #[test]
    fn zero_impact_denominator_yields_undefined_weighted_recall() {
        let (proposals, ranking, mut audit, proposal, ranking_hash, protocol) = fixture();
        for unit in &mut audit.sampled_units {
            for error in &mut unit.errors { error.impact_weight = 0.0; }
        }
        let (frame, outcomes, selected) = validate(
            &proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol,
        ).unwrap();
        let report = evaluate(
            &proposals, &audit, &frame, &outcomes, &selected,
            proposal, ranking_hash, protocol, "audit".into(),
        ).unwrap();
        assert_eq!(report.impact_weighted_candidate_generation_recall.ratio, None);
    }

    #[test]
    fn impact_weighted_recall_penalizes_uncovered_high_impact_errors() {
        let (proposals, ranking, audit, proposal, ranking_hash, protocol) = fixture();
        let (frame, outcomes, selected) = validate(
            &proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol,
        ).unwrap();
        let report = evaluate(
            &proposals, &audit, &frame, &outcomes, &selected,
            proposal, ranking_hash, protocol, "audit".into(),
        ).unwrap();
        assert!((report.candidate_generation_recall.ratio.unwrap() - 0.5).abs() < 1e-12);
        assert!((report.impact_weighted_candidate_generation_recall.ratio.unwrap() - (5.0 / 14.0)).abs() < 1e-12);
        assert_eq!(report.error_family_coverage["merge"].estimated_total_errors.total, 2.0);
        assert!((report.error_family_coverage["merge"].candidate_generation_recall.ratio.unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn deterministic_selection_uses_all_declared_psus_at_census() {
        let (proposals, ranking, audit, proposal, ranking_hash, protocol) = fixture();
        let (frame, _outcomes, selected) = validate(
            &proposals, &ranking, &audit, &proposal, &ranking_hash, &protocol,
        ).unwrap();
        assert_eq!(selected["region-a"], BTreeSet::from(["psu-a1".into(), "psu-a2".into()]));
        assert_eq!(selected["region-b"], BTreeSet::from(["psu-b1".into(), "psu-b2".into()]));
        assert_eq!(frame["region-a"].len(), 2);
    }
}
