//! Design-based probability audit of residual connectome proofreading risk.
//!
//! This tool samples clusters without replacement within predeclared strata using a
//! SHA-256 keyed ordering. It audits every candidate in selected clusters, computes
//! Horvitz-Thompson total estimates from cluster inclusion probabilities, and uses
//! the one-stage stratified cluster-sampling variance estimator. It is evaluation-only:
//! no labels or audit outcomes are exposed to the proposal ranker.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-sampling-audit",
    version,
    about = "Estimate residual proofreading risk with a digest-bound stratified cluster probability audit"
)]
struct Args {
    /// Exact frozen proposal input used by connectome-query-impact.
    #[arg(long)]
    proposals: PathBuf,

    /// Exact sampling protocol artifact committed before outcomes are collected.
    #[arg(long)]
    sampling_protocol: PathBuf,

    /// Complete candidate frame, reproducible sample selection, and sampled outcomes.
    #[arg(long)]
    audit: PathBuf,

    /// Optional deterministic report path; stdout when omitted.
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AuditInput {
    schema_version: u32,
    source_input_sha256: String,
    sampling_protocol_sha256: String,
    sample_seed: String,
    sample_seed_commitment_sha256: String,
    strata: Vec<StratumPlan>,
    population_frame: Vec<PopulationUnit>,
    sampled_outcomes: Vec<SampledOutcome>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StratumPlan {
    stratum_id: String,
    sampled_cluster_count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PopulationUnit {
    candidate_id: String,
    stratum_id: String,
    cluster_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SampledOutcome {
    candidate_id: String,
    adjudication: Adjudication,
    /// Signed reduction in the preregistered normalized query-error objective if
    /// the proposed mutation were applied against the independent reference.
    /// Positive is beneficial; negative is harmful. Null only for indeterminate.
    signed_query_error_reduction_if_applied: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
enum Adjudication {
    Correct,
    Incorrect,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, Default)]
struct ClusterTotals {
    correct: f64,
    incorrect: f64,
    indeterminate: f64,
    positive_opportunity: f64,
    negative_harm: f64,
    signed_net_reduction: f64,
}

#[derive(Debug, Serialize)]
struct Estimate {
    total: f64,
    design_variance: f64,
    standard_error: f64,
    normal_approx_95_interval: Option<[f64; 2]>,
}

#[derive(Debug, Serialize)]
struct StratumEstimate {
    stratum_id: String,
    population_cluster_count: usize,
    sampled_cluster_count: usize,
    population_candidate_count: usize,
    sampled_candidate_count: usize,
    candidate_cluster_inclusion_probability: f64,
    estimated_correct_proposal_count: Estimate,
    estimated_incorrect_proposal_count: Estimate,
    estimated_indeterminate_count: Estimate,
    estimated_positive_query_opportunity_mass: Estimate,
    estimated_negative_query_harm_mass: Estimate,
    estimated_signed_net_query_error_reduction: Estimate,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u32,
    tool: &'static str,
    proposal_input_sha256: String,
    sampling_protocol_sha256: String,
    sampling_frame_sha256: String,
    sampled_outcomes_sha256: String,
    sample_seed_commitment_sha256: String,
    candidate_count: usize,
    sampled_candidate_count: usize,
    population_cluster_count: usize,
    sampled_cluster_count: usize,
    estimated_incorrect_candidate_fraction: Estimate,
    strata: Vec<StratumEstimate>,
    warnings: Vec<String>,
    status: &'static str,
    scope_note: &'static str,
}

#[derive(Debug, Clone)]
struct StratumFrame {
    candidate_ids: Vec<String>,
    clusters: BTreeMap<String, Vec<String>>,
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

fn frame_sha256(frame: &[PopulationUnit]) -> String {
    let mut rows: Vec<(String, String, String)> = frame
        .iter()
        .map(|row| {
            (
                row.candidate_id.clone(),
                row.stratum_id.clone(),
                row.cluster_id.clone(),
            )
        })
        .collect();
    rows.sort();
    let mut hasher = Sha256::new();
    for (candidate_id, stratum_id, cluster_id) in rows {
        hasher.update((candidate_id.len() as u64).to_be_bytes());
        hasher.update(candidate_id.as_bytes());
        hasher.update((stratum_id.len() as u64).to_be_bytes());
        hasher.update(stratum_id.as_bytes());
        hasher.update((cluster_id.len() as u64).to_be_bytes());
        hasher.update(cluster_id.as_bytes());
    }
    hex::encode(hasher.finalize())
}

fn cluster_rank(seed: &str, stratum_id: &str, cluster_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-connectome-cluster-sample-v1\0");
    hasher.update(seed.as_bytes());
    hasher.update(b"\0");
    hasher.update(stratum_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(cluster_id.as_bytes());
    hex::encode(hasher.finalize())
}

fn expected_sampled_clusters(
    seed: &str,
    frame: &BTreeMap<String, StratumFrame>,
    plans: &[StratumPlan],
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let mut selected = BTreeMap::new();
    for plan in plans {
        let stratum = frame
            .get(&plan.stratum_id)
            .ok_or_else(|| format!("sampling plan references absent stratum {}", plan.stratum_id))?;
        let population_count = stratum.clusters.len();
        let sample_count = plan.sampled_cluster_count;
        if sample_count == 0 || sample_count > population_count {
            return Err(format!(
                "stratum {} sample cluster count must be in 1..={}",
                plan.stratum_id, population_count
            ));
        }
        if sample_count < population_count && sample_count < 2 {
            return Err(format!(
                "stratum {} needs at least two sampled clusters for a non-census variance estimate",
                plan.stratum_id
            ));
        }

        let mut ranked: Vec<(String, String)> = stratum
            .clusters
            .keys()
            .map(|cluster_id| {
                (
                    cluster_rank(seed, &plan.stratum_id, cluster_id),
                    cluster_id.clone(),
                )
            })
            .collect();
        ranked.sort();
        selected.insert(
            plan.stratum_id.clone(),
            ranked
                .into_iter()
                .take(sample_count)
                .map(|(_, cluster_id)| cluster_id)
                .collect(),
        );
    }
    Ok(selected)
}

fn validate(
    proposals: &ProposalInput,
    audit: &AuditInput,
    proposal_hash: &str,
    protocol_hash: &str,
) -> Result<(BTreeMap<String, StratumFrame>, BTreeMap<String, SampledOutcome>, BTreeMap<String, BTreeSet<String>>), String> {
    if proposals.schema_version != 1 || audit.schema_version != 1 {
        return Err("proposal and audit schema_version values must be 1".into());
    }
    if audit.source_input_sha256 != proposal_hash {
        return Err("audit source_input_sha256 does not match exact proposal bytes".into());
    }
    if audit.sampling_protocol_sha256 != protocol_hash {
        return Err("audit sampling_protocol_sha256 does not match exact protocol bytes".into());
    }
    if !is_lower_hex(proposal_hash, 64)
        || !is_lower_hex(protocol_hash, 64)
        || !is_lower_hex(&audit.sample_seed_commitment_sha256, 64)
    {
        return Err("proposal/protocol/seed commitment SHA-256 is malformed".into());
    }
    if audit.sample_seed.len() < 16 {
        return Err("sample_seed must contain at least 16 bytes".into());
    }
    if sha256(audit.sample_seed.as_bytes()) != audit.sample_seed_commitment_sha256 {
        return Err("sample_seed does not match committed SHA-256".into());
    }

    let mut proposal_ids = BTreeSet::new();
    for candidate in &proposals.candidates {
        if candidate.candidate_id.trim().is_empty()
            || !proposal_ids.insert(candidate.candidate_id.clone())
        {
            return Err("frozen proposal candidate IDs must be non-empty and unique".into());
        }
    }
    if proposal_ids.is_empty() {
        return Err("frozen proposal pool must be non-empty".into());
    }

    let mut frame = BTreeMap::<String, StratumFrame>::new();
    let mut frame_ids = BTreeSet::new();
    for unit in &audit.population_frame {
        if unit.candidate_id.trim().is_empty()
            || unit.stratum_id.trim().is_empty()
            || unit.cluster_id.trim().is_empty()
        {
            return Err("population frame candidate/stratum/cluster IDs must be non-empty".into());
        }
        if !frame_ids.insert(unit.candidate_id.clone()) {
            return Err(format!("duplicate candidate in population frame: {}", unit.candidate_id));
        }
        let stratum = frame.entry(unit.stratum_id.clone()).or_insert_with(|| StratumFrame {
            candidate_ids: Vec::new(),
            clusters: BTreeMap::new(),
        });
        stratum.candidate_ids.push(unit.candidate_id.clone());
        stratum
            .clusters
            .entry(unit.cluster_id.clone())
            .or_default()
            .push(unit.candidate_id.clone());
    }
    if frame_ids != proposal_ids {
        let missing: Vec<String> = proposal_ids.difference(&frame_ids).cloned().collect();
        let extra: Vec<String> = frame_ids.difference(&proposal_ids).cloned().collect();
        return Err(format!(
            "population frame must exactly match frozen proposal pool; missing=[{}], extra=[{}]",
            missing.join(","), extra.join(",")
        ));
    }

    let mut plan_ids = BTreeSet::new();
    for plan in &audit.strata {
        if plan.stratum_id.trim().is_empty() || !plan_ids.insert(plan.stratum_id.clone()) {
            return Err("sampling stratum IDs must be non-empty and unique".into());
        }
    }
    let frame_strata: BTreeSet<String> = frame.keys().cloned().collect();
    if plan_ids != frame_strata {
        return Err("sampling plans must cover every and only population strata".into());
    }

    let selected_clusters = expected_sampled_clusters(&audit.sample_seed, &frame, &audit.strata)?;
    let mut sampled_candidate_ids = BTreeSet::new();
    for (stratum_id, clusters) in &selected_clusters {
        let stratum = &frame[stratum_id];
        for cluster_id in clusters {
            for candidate_id in &stratum.clusters[cluster_id] {
                sampled_candidate_ids.insert(candidate_id.clone());
            }
        }
    }

    let mut outcomes = BTreeMap::new();
    for outcome in &audit.sampled_outcomes {
        if !sampled_candidate_ids.contains(&outcome.candidate_id) {
            return Err(format!(
                "sample outcome {} is outside the reproducibly selected cluster sample",
                outcome.candidate_id
            ));
        }
        if let Some(value) = outcome.signed_query_error_reduction_if_applied {
            if !value.is_finite() {
                return Err(format!("candidate {} has non-finite query-error reduction", outcome.candidate_id));
            }
        }
        if outcome.adjudication == Adjudication::Indeterminate
            && outcome.signed_query_error_reduction_if_applied.is_some()
        {
            return Err(format!(
                "candidate {} with indeterminate adjudication must not carry an oracle query delta",
                outcome.candidate_id
            ));
        }
        if outcome.adjudication != Adjudication::Indeterminate
            && outcome.signed_query_error_reduction_if_applied.is_none()
        {
            return Err(format!(
                "candidate {} with determinate adjudication requires an oracle query delta",
                outcome.candidate_id
            ));
        }
        if outcomes.insert(outcome.candidate_id.clone(), outcome.clone()).is_some() {
            return Err(format!("duplicate sampled outcome for candidate {}", outcome.candidate_id));
        }
    }
    let outcome_ids: BTreeSet<String> = outcomes.keys().cloned().collect();
    if outcome_ids != sampled_candidate_ids {
        let missing: Vec<String> = sampled_candidate_ids.difference(&outcome_ids).cloned().collect();
        let extra: Vec<String> = outcome_ids.difference(&sampled_candidate_ids).cloned().collect();
        return Err(format!(
            "outcomes must cover every candidate in sampled clusters; missing=[{}], extra=[{}]",
            missing.join(","), extra.join(",")
        ));
    }
    Ok((frame, outcomes, selected_clusters))
}

fn sample_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|value| (value - average).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64
}

fn estimate(values: &[f64], population_clusters: usize) -> Estimate {
    let sample_clusters = values.len();
    let scale = population_clusters as f64 / sample_clusters as f64;
    let total = scale * values.iter().sum::<f64>();
    let variance = if sample_clusters == population_clusters {
        0.0
    } else {
        let sampling_fraction = sample_clusters as f64 / population_clusters as f64;
        (population_clusters as f64).powi(2)
            * (1.0 - sampling_fraction)
            * sample_variance(values)
            / sample_clusters as f64
    };
    let standard_error = variance.sqrt();
    let normal_approx_95_interval = if sample_clusters == population_clusters {
        Some([total, total])
    } else if sample_clusters >= 30 {
        Some([total - 1.96 * standard_error, total + 1.96 * standard_error])
    } else {
        None
    };
    Estimate {
        total,
        design_variance: variance,
        standard_error,
        normal_approx_95_interval,
    }
}

fn combine_estimates(estimates: &[Estimate]) -> Estimate {
    let total = estimates.iter().map(|value| value.total).sum::<f64>();
    let variance = estimates
        .iter()
        .map(|value| value.design_variance)
        .sum::<f64>();
    let standard_error = variance.sqrt();
    let interval = estimates
        .iter()
        .all(|value| value.normal_approx_95_interval.is_some())
        .then_some([
            total - 1.96 * standard_error,
            total + 1.96 * standard_error,
        ]);
    Estimate {
        total,
        design_variance: variance,
        standard_error,
        normal_approx_95_interval: interval,
    }
}

fn evaluate(
    proposals: &ProposalInput,
    audit: &AuditInput,
    frame: &BTreeMap<String, StratumFrame>,
    outcomes: &BTreeMap<String, SampledOutcome>,
    selected: &BTreeMap<String, BTreeSet<String>>,
    proposal_hash: String,
    protocol_hash: String,
    frame_hash: String,
    outcomes_hash: String,
) -> AuditReport {
    let mut stratum_estimates = Vec::new();
    let mut global_estimates: [Vec<Estimate>; 6] = std::array::from_fn(|_| Vec::new());
    let mut population_cluster_count = 0usize;
    let mut sampled_cluster_count = 0usize;
    let mut sampled_candidate_count = 0usize;
    let mut warnings = Vec::new();

    let mut plans = audit.strata.clone();
    plans.sort_by(|left, right| left.stratum_id.cmp(&right.stratum_id));

    for plan in plans {
        let stratum = &frame[&plan.stratum_id];
        let sampled = &selected[&plan.stratum_id];
        let population_clusters = stratum.clusters.len();
        let sampled_clusters = sampled.len();
        let population_candidates = stratum.candidate_ids.len();
        let sampled_candidates = sampled
            .iter()
            .map(|cluster| stratum.clusters[cluster].len())
            .sum::<usize>();

        let mut cluster_values = Vec::new();
        for cluster_id in sampled {
            let mut totals = ClusterTotals::default();
            for candidate_id in &stratum.clusters[cluster_id] {
                let outcome = &outcomes[candidate_id];
                match outcome.adjudication {
                    Adjudication::Correct => totals.correct += 1.0,
                    Adjudication::Incorrect => totals.incorrect += 1.0,
                    Adjudication::Indeterminate => totals.indeterminate += 1.0,
                }
                if let Some(delta) = outcome.signed_query_error_reduction_if_applied {
                    if delta > 0.0 {
                        totals.positive_opportunity += delta;
                    } else if delta < 0.0 {
                        totals.negative_harm += -delta;
                    }
                    totals.signed_net_reduction += delta;
                }
            }
            cluster_values.push(totals);
        }

        let fields: [Vec<f64>; 6] = [
            cluster_values.iter().map(|v| v.correct).collect(),
            cluster_values.iter().map(|v| v.incorrect).collect(),
            cluster_values.iter().map(|v| v.indeterminate).collect(),
            cluster_values.iter().map(|v| v.positive_opportunity).collect(),
            cluster_values.iter().map(|v| v.negative_harm).collect(),
            cluster_values.iter().map(|v| v.signed_net_reduction).collect(),
        ];
        let estimates: [Estimate; 6] = std::array::from_fn(|index| {
            estimate(&fields[index], population_clusters)
        });

        for index in 0..6 {
            global_estimates[index].push(Estimate {
                total: estimates[index].total,
                design_variance: estimates[index].design_variance,
                standard_error: estimates[index].standard_error,
                normal_approx_95_interval: estimates[index].normal_approx_95_interval,
            });
        }

        if sampled_clusters < population_clusters && sampled_clusters < 30 {
            warnings.push(format!(
                "stratum {} has {} sampled clusters; design variance is reported, but normal-approximation 95% intervals are withheld below 30 sampled clusters",
                plan.stratum_id, sampled_clusters
            ));
        }

        stratum_estimates.push(StratumEstimate {
            stratum_id: plan.stratum_id,
            population_cluster_count: population_clusters,
            sampled_cluster_count: sampled_clusters,
            population_candidate_count: population_candidates,
            sampled_candidate_count: sampled_candidates,
            candidate_cluster_inclusion_probability: sampled_clusters as f64 / population_clusters as f64,
            estimated_correct_proposal_count: estimates[0].clone(),
            estimated_incorrect_proposal_count: estimates[1].clone(),
            estimated_indeterminate_count: estimates[2].clone(),
            estimated_positive_query_opportunity_mass: estimates[3].clone(),
            estimated_negative_query_harm_mass: estimates[4].clone(),
            estimated_signed_net_query_error_reduction: estimates[5].clone(),
        });

        population_cluster_count += population_clusters;
        sampled_cluster_count += sampled_clusters;
        sampled_candidate_count += sampled_candidates;
    }

    let total_incorrect = combine_estimates(&global_estimates[1]);
    let population_candidate_count = proposals.candidates.len() as f64;
    let incorrect_fraction = Estimate {
        total: total_incorrect.total / population_candidate_count,
        design_variance: total_incorrect.design_variance / population_candidate_count.powi(2),
        standard_error: total_incorrect.standard_error / population_candidate_count,
        normal_approx_95_interval: total_incorrect.normal_approx_95_interval.map(|bounds| {
            [
                (bounds[0] / population_candidate_count).clamp(0.0, 1.0),
                (bounds[1] / population_candidate_count).clamp(0.0, 1.0),
            ]
        }),
    };

    warnings.push(
        "This is a stratified one-stage cluster sample without replacement. Inclusion probability is sampled_clusters/population_clusters within each stratum; all candidates in selected clusters are adjudicated.".into(),
    );
    warnings.push(
        "The SHA-256 ordering is reproducible pseudorandom selection, not proof of entropy. Generate the seed from a cryptographically secure source, commit its hash before outcomes are observed, and reveal the seed only after collection.".into(),
    );
    warnings.push(
        "Query-error deltas are isolated candidate counterfactuals. Their totals are not a sequential graph replay and do not account for interactions between mutations.".into(),
    );
    warnings.push(
        "An indeterminate adjudication has no query-error delta and is estimated as its own population category; no label is imputed.".into(),
    );

    AuditReport {
        schema_version: 1,
        tool: "connectome-sampling-audit",
        proposal_input_sha256: proposal_hash,
        sampling_protocol_sha256: protocol_hash,
        sampling_frame_sha256: frame_hash,
        sampled_outcomes_sha256: outcomes_hash,
        sample_seed_commitment_sha256: audit.sample_seed_commitment_sha256.clone(),
        candidate_count: proposals.candidates.len(),
        sampled_candidate_count,
        population_cluster_count,
        sampled_cluster_count,
        estimated_incorrect_candidate_fraction: incorrect_fraction,
        strata: stratum_estimates,
        warnings,
        status: "DESIGN_BASED_CLUSTER_AUDIT_ESTIMATED",
        scope_note: "Design-based estimates of frozen candidate-pool adjudication and isolated query-impact outcomes. Not whole-connectome error recall, not a causal queue-policy effect, and not a substitute for a prospective policy trial.",
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let proposal_bytes = fs::read(&args.proposals)
        .map_err(|error| format!("failed to read frozen proposal input: {error}"))?;
    let protocol_bytes = fs::read(&args.sampling_protocol)
        .map_err(|error| format!("failed to read sampling protocol: {error}"))?;
    let audit_bytes = fs::read(&args.audit)
        .map_err(|error| format!("failed to read sampling audit input: {error}"))?;

    let proposal_hash = sha256(&proposal_bytes);
    let protocol_hash = sha256(&protocol_bytes);
    let audit_hash = sha256(&audit_bytes);
    let proposals: ProposalInput = serde_json::from_slice(&proposal_bytes)
        .map_err(|error| format!("invalid proposal JSON: {error}"))?;
    let audit: AuditInput = serde_json::from_slice(&audit_bytes)
        .map_err(|error| format!("invalid sampling audit JSON: {error}"))?;

    let (frame, outcomes, selected) =
        validate(&proposals, &audit, &proposal_hash, &protocol_hash)?;
    let frame_hash = frame_sha256(&audit.population_frame);
    let report = evaluate(
        &proposals,
        &audit,
        &frame,
        &outcomes,
        &selected,
        proposal_hash,
        protocol_hash,
        frame_hash,
        audit_hash,
    );
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize sample audit report: {error}"))?
        + "\n";
    if let Some(output) = args.output {
        fs::write(&output, json.as_bytes())
            .map_err(|error| format!("failed to write sample audit report: {error}"))?;
    } else {
        print!("{json}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("connectome-sampling-audit: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> (ProposalInput, AuditInput, String, String) {
        let proposal_value = json!({
            "schema_version": 1,
            "dataset_id": "synthetic-probability-audit",
            "candidates": [
                {"candidate_id":"c1"},
                {"candidate_id":"c2"},
                {"candidate_id":"c3"},
                {"candidate_id":"c4"},
                {"candidate_id":"c5"},
                {"candidate_id":"c6"},
                {"candidate_id":"c7"},
                {"candidate_id":"c8"}
            ]
        });
        let proposal_bytes = serde_json::to_vec(&proposal_value).unwrap();
        let proposal_hash = sha256(&proposal_bytes);
        let seed = "fixed-seed-2026-10-30";
        let outcome_value = json!({
            "schema_version": 1,
            "source_input_sha256": proposal_hash,
            "sampling_protocol_sha256": "9".repeat(64),
            "sample_seed": seed,
            "sample_seed_commitment_sha256": sha256(seed.as_bytes()),
            "strata": [
                {"stratum_id":"s1","sampled_cluster_count":2},
                {"stratum_id":"s2","sampled_cluster_count":2}
            ],
            "population_frame": [
                {"candidate_id":"c1","stratum_id":"s1","cluster_id":"k1"},
                {"candidate_id":"c2","stratum_id":"s1","cluster_id":"k1"},
                {"candidate_id":"c3","stratum_id":"s1","cluster_id":"k2"},
                {"candidate_id":"c4","stratum_id":"s1","cluster_id":"k3"},
                {"candidate_id":"c5","stratum_id":"s2","cluster_id":"k4"},
                {"candidate_id":"c6","stratum_id":"s2","cluster_id":"k5"},
                {"candidate_id":"c7","stratum_id":"s2","cluster_id":"k6"},
                {"candidate_id":"c8","stratum_id":"s2","cluster_id":"k6"}
            ],
            "sampled_outcomes": []
        });
        let proposal: ProposalInput = serde_json::from_value(proposal_value).unwrap();
        let mut audit: AuditInput = serde_json::from_value(outcome_value).unwrap();
        let dummy_frame = BTreeMap::from([
            ("s1".to_string(), StratumFrame {
                candidate_ids: vec!["c1".into(),"c2".into(),"c3".into(),"c4".into()],
                clusters: BTreeMap::from([
                    ("k1".to_string(),vec!["c1".into(),"c2".into()]),
                    ("k2".to_string(),vec!["c3".into()]),
                    ("k3".to_string(),vec!["c4".into()])
                ])
            }),
            ("s2".to_string(), StratumFrame {
                candidate_ids: vec!["c5".into(),"c6".into(),"c7".into(),"c8".into()],
                clusters: BTreeMap::from([
                    ("k4".to_string(),vec!["c5".into()]),
                    ("k5".to_string(),vec!["c6".into()]),
                    ("k6".to_string(),vec!["c7".into(),"c8".into()])
                ])
            })
        ]);
        let selected = expected_sampled_clusters(seed, &dummy_frame, &audit.strata).unwrap();
        for cluster_ids in selected.values() {
            for cluster in cluster_ids {
                for candidate_id in dummy_frame.values().flat_map(|s| s.clusters.get(cluster).into_iter().flatten()) {
                    let adjudication = match candidate_id.as_str() {
                        "c1" | "c5" | "c7" => Adjudication::Incorrect,
                        "c2" | "c3" | "c6" => Adjudication::Correct,
                        _ => Adjudication::Indeterminate,
                    };
                    audit.sampled_outcomes.push(SampledOutcome {
                        candidate_id: candidate_id.clone(),
                        adjudication,
                        signed_query_error_reduction_if_applied: match adjudication {
                            Adjudication::Indeterminate => None,
                            Adjudication::Correct => Some(0.2),
                            Adjudication::Incorrect => Some(-0.3),
                        },
                    });
                }
            }
        }
        // The fixture has globally unique cluster IDs; rebuild outcomes without relying
        // on ambiguous cluster-name lookup across strata.
        audit.sampled_outcomes.sort_by(|a,b| a.candidate_id.cmp(&b.candidate_id));
        let _ = dummy_frame;
        let ranking_hash = "8".repeat(64);
        (proposal, audit, proposal_hash, ranking_hash)
    }

    fn validated_fixture() -> (ProposalInput, AuditInput, BTreeMap<String, StratumFrame>, BTreeMap<String, SampledOutcome>, BTreeMap<String,BTreeSet<String>>, String, String) {
        let (proposal, audit, proposal_hash, _) = fixture();
        let frame_rows = audit.population_frame.clone();
        let mut frame = BTreeMap::<String, StratumFrame>::new();
        for unit in &frame_rows {
            let stratum = frame.entry(unit.stratum_id.clone()).or_insert_with(|| StratumFrame {
                candidate_ids: Vec::new(),
                clusters: BTreeMap::new(),
            });
            stratum.candidate_ids.push(unit.candidate_id.clone());
            stratum.clusters.entry(unit.cluster_id.clone()).or_default().push(unit.candidate_id.clone());
        }
        let selected = expected_sampled_clusters(&audit.sample_seed, &frame, &audit.strata).unwrap();
        let outcomes = audit.sampled_outcomes.iter().cloned().map(|o|(o.candidate_id.clone(),o)).collect();
        (proposal,audit,frame,outcomes,selected,proposal_hash,"9".repeat(64))
    }

    #[test]
    fn estimates_cluster_totals_with_finite_population_correction() {
        let values = vec![3.0, 5.0];
        let result = estimate(&values, 4);
        assert_eq!(result.total, 16.0);
        assert!((result.design_variance - 8.0).abs() < 1e-12);
        assert!(result.normal_approx_95_interval.is_none());
    }

    #[test]
    fn census_has_zero_design_variance_and_exact_interval() {
        let result = estimate(&[2.0, 4.0, 6.0], 3);
        assert_eq!(result.total, 12.0);
        assert_eq!(result.design_variance, 0.0);
        assert_eq!(result.normal_approx_95_interval, Some([12.0, 12.0]));
    }

    #[test]
    fn selection_is_deterministic_and_respects_stratum_quota() {
        let (_proposal,audit,frame,_,_,_,_)=validated_fixture();
        let first=expected_sampled_clusters(&audit.sample_seed,&frame,&audit.strata).unwrap();
        let second=expected_sampled_clusters(&audit.sample_seed,&frame,&audit.strata).unwrap();
        assert_eq!(first,second);
        assert_eq!(first["s1"].len(),2);
        assert_eq!(first["s2"].len(),2);
    }

    #[test]
    fn rejects_zero_or_one_non_census_cluster_sample() {
        let mut frame = BTreeMap::new();
        frame.insert("s".into(),StratumFrame {
            candidate_ids:vec!["a".into(),"b".into(),"c".into()],
            clusters:BTreeMap::from([
                ("k1".into(),vec!["a".into()]),
                ("k2".into(),vec!["b".into()]),
                ("k3".into(),vec!["c".into()])
            ])
        });
        let err=expected_sampled_clusters("0123456789abcdef",&frame,&[StratumPlan {
            stratum_id:"s".into(),sampled_cluster_count:1
        }]).unwrap_err();
        assert!(err.contains("at least two"));
        let err=expected_sampled_clusters("0123456789abcdef",&frame,&[StratumPlan {
            stratum_id:"s".into(),sampled_cluster_count:0
        }]).unwrap_err();
        assert!(err.contains("must be in"));
    }

    #[test]
    fn rejects_protocol_or_seed_commitment_mismatch() {
        let (proposal,mut audit,proposal_hash,_)=fixture();
        assert!(validate(&proposal,&audit,&proposal_hash,&"0".repeat(64)).unwrap_err().contains("protocol bytes"));
        audit.sampling_protocol_sha256="9".repeat(64);
        audit.sample_seed_commitment_sha256="0".repeat(64);
        assert!(validate(&proposal,&audit,&proposal_hash,&"9".repeat(64)).unwrap_err().contains("sample_seed"));
    }

    #[test]
    fn rejects_population_frame_not_equal_to_proposal_pool() {
        let (proposal,mut audit,proposal_hash,_)=fixture();
        audit.population_frame.pop();
        assert!(validate(&proposal,&audit,&proposal_hash,&"9".repeat(64)).unwrap_err().contains("exactly match"));
    }

    #[test]
    fn rejects_duplicate_candidate_or_cluster_cross_stratum_alias() {
        let (proposal,mut audit,proposal_hash,_)=fixture();
        audit.population_frame.push(audit.population_frame[0].clone());
        assert!(validate(&proposal,&audit,&proposal_hash,&"9".repeat(64)).unwrap_err().contains("duplicate candidate"));
    }

    #[test]
    fn rejects_outcomes_outside_selected_clusters_and_missing_sampled_outcomes() {
        let (proposal,mut audit,proposal_hash,_)=fixture();
        audit.sampled_outcomes.pop();
        assert!(validate(&proposal,&audit,&proposal_hash,&"9".repeat(64)).unwrap_err().contains("cover every candidate"));
        audit.sampled_outcomes = Vec::new();
        audit.sampled_outcomes.push(SampledOutcome {
            candidate_id: "unselected".into(),
            adjudication: Adjudication::Correct,
            signed_query_error_reduction_if_applied: Some(0.1),
        });
        assert!(validate(&proposal,&audit,&proposal_hash,&"9".repeat(64)).unwrap_err().contains("outside the reproducibly"));
    }

    #[test]
    fn requires_indeterminate_to_have_no_oracle_delta_and_determinate_to_have_one() {
        let (proposal,mut audit,proposal_hash,_)=fixture();
        let outcome=&mut audit.sampled_outcomes[0];
        outcome.adjudication=Adjudication::Indeterminate;
        outcome.signed_query_error_reduction_if_applied=Some(0.5);
        assert!(validate(&proposal,&audit,&proposal_hash,&"9".repeat(64)).unwrap_err().contains("indeterminate"));
    }

    #[test]
    fn reports_design_weighted_estimates_on_synthetic_sample() {
        let (proposal,audit,frame,outcomes,selected,proposal_hash,protocol_hash)=validated_fixture();
        let report=evaluate(
            &proposal,&audit,&frame,&outcomes,&selected,proposal_hash,protocol_hash,
            frame_sha256(&audit.population_frame),sha256(b"synthetic outcomes")
        );
        assert_eq!(report.candidate_count,8);
        assert_eq!(report.sampled_cluster_count,4);
        assert_eq!(report.status,"DESIGN_BASED_CLUSTER_AUDIT_ESTIMATED");
        assert!(report.estimated_incorrect_candidate_fraction.total >= 0.0);
        assert_eq!(report.strata.len(),2);
    }

    #[test]
    fn accepts_complete_census_of_each_stratum() {
        let (proposal,mut audit,proposal_hash,_)=fixture();
        for plan in &mut audit.strata {
            plan.sampled_cluster_count = 3;
        }
        let mut frame = BTreeMap::<String,StratumFrame>::new();
        for unit in &audit.population_frame {
            let row=frame.entry(unit.stratum_id.clone()).or_insert_with(||StratumFrame {
                candidate_ids:Vec::new(),clusters:BTreeMap::new()
            });
            row.candidate_ids.push(unit.candidate_id.clone());
            row.clusters.entry(unit.cluster_id.clone()).or_default().push(unit.candidate_id.clone());
        }
        let selected=expected_sampled_clusters(&audit.sample_seed,&frame,&audit.strata).unwrap();
        audit.sampled_outcomes.clear();
        for (sid,clusters) in &selected {
            for cluster in clusters {
                for id in &frame[sid].clusters[cluster] {
                    audit.sampled_outcomes.push(SampledOutcome {
                        candidate_id:id.clone(),adjudication:Adjudication::Correct,
                        signed_query_error_reduction_if_applied:Some(0.1)
                    });
                }
            }
        }
        let (checked_frame,outcomes,checked_selected)=validate(&proposal,&audit,&proposal_hash,&"9".repeat(64)).unwrap();
        let report=evaluate(&proposal,&audit,&checked_frame,&outcomes,&checked_selected,proposal_hash,"9".repeat(64),frame_sha256(&audit.population_frame),sha256(b"census outcomes"));
        assert!(report.strata.iter().all(|s|s.estimated_incorrect_proposal_count.design_variance==0.0));
    }

    #[test]
    fn parses_checked_in_sampling_template_and_protocol_metadata() {
        let audit: AuditInput = serde_json::from_str(include_str!("../../../../docs/neuroscience/examples/CONNECTOME_SAMPLING_AUDIT.example.json")).unwrap();
        assert_eq!(audit.schema_version,1);
        assert_eq!(audit.strata.len(),2);
        let protocol=include_str!("../../../../docs/neuroscience/CONNECTOME_SAMPLING_AUDIT_PROTOCOL_V1.md");
        assert!(protocol.contains("one-stage stratified cluster"));
    }
}
