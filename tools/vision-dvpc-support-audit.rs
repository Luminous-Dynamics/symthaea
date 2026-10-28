//! VIS-DVPC-027: structural support audit for selective-feedback policy evaluation.
//! This is an identification/overlap diagnostic, not a policy-value certification.
#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;

const EPS: f64 = 1e-9;
const MIN_EFFECTIVE_SAMPLE_SIZE: f64 = 5.0;
const MAX_SUPPORTED_WEIGHT: f64 = 10.0;
const SUPPORTED_FIXTURE: &str = include_str!("../tests/fixtures/vision-dvpc-027-supported.tsv");
const ZERO_SUPPORT_FIXTURE: &str = include_str!("../tests/fixtures/vision-dvpc-027-zero-support.tsv");

#[derive(Clone, Debug)]
struct Row {
    context: String,
    action: String,
    target_p: f64,
    logging_p: f64,
    selected: bool,
    reward: Option<f64>,
    cluster: String,
}

#[derive(Clone, Debug)]
struct Report {
    status: &'static str,
    context_count: usize,
    selected_count: usize,
    unsupported_pairs: usize,
    mean_target_mass_outside_support: f64,
    max_observed_weight: f64,
    max_supported_weight: f64,
    effective_sample_size: f64,
    ips_estimate: Option<f64>,
    declared_cluster_count: usize,
    max_contexts_per_cluster: usize,
    contexts_in_repeated_clusters: usize,
}

fn fail(message: impl Into<String>) -> String { message.into() }

fn parse_trace(input: &str) -> Result<Vec<Row>, String> {
    let mut lines = input.lines();
    let header = lines.next().ok_or_else(|| fail("empty trace"))?;
    if header != "context_id\taction_id\ttarget_probability\tlogging_probability\tselected\treward\tcluster_id" {
        return Err(fail("unexpected trace header"));
    }

    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 7 {
            return Err(fail(format!("line {}: expected seven TSV fields", line_index + 2)));
        }
        if fields[0].is_empty() || fields[1].is_empty() || fields[6].is_empty() {
            return Err(fail(format!("line {}: context, action, and cluster IDs are required", line_index + 2)));
        }
        if !seen.insert((fields[0].to_owned(), fields[1].to_owned())) {
            return Err(fail(format!("line {}: duplicate context/action pair", line_index + 2)));
        }
        let target_p = fields[2].parse::<f64>().map_err(|_| fail(format!("line {}: invalid target probability", line_index + 2)))?;
        let logging_p = fields[3].parse::<f64>().map_err(|_| fail(format!("line {}: invalid logging probability", line_index + 2)))?;
        if !target_p.is_finite() || !logging_p.is_finite() || !(-EPS..=1.0 + EPS).contains(&target_p) || !(-EPS..=1.0 + EPS).contains(&logging_p) {
            return Err(fail(format!("line {}: probabilities must be finite values in [0,1]", line_index + 2)));
        }
        let selected = match fields[4] {
            "yes" => true,
            "no" => false,
            _ => return Err(fail(format!("line {}: selected must be yes or no", line_index + 2))),
        };
        let reward = if fields[5] == "NA" {
            None
        } else {
            let value = fields[5].parse::<f64>().map_err(|_| fail(format!("line {}: reward must be in [0,1] or NA", line_index + 2)))?;
            if !value.is_finite() || !(-EPS..=1.0 + EPS).contains(&value) {
                return Err(fail(format!("line {}: reward must be finite and in [0,1]", line_index + 2)));
            }
            Some(value.clamp(0.0, 1.0))
        };
        if selected && reward.is_none() {
            return Err(fail(format!("line {}: selected action needs an observed reward", line_index + 2)));
        }
        if !selected && reward.is_some() {
            return Err(fail(format!("line {}: unselected action must not reveal a counterfactual reward", line_index + 2)));
        }
        rows.push(Row {
            context: fields[0].to_owned(),
            action: fields[1].to_owned(),
            target_p: target_p.clamp(0.0, 1.0),
            logging_p: logging_p.clamp(0.0, 1.0),
            selected,
            reward,
            cluster: fields[6].to_owned(),
        });
    }
    if rows.is_empty() { return Err(fail("trace has no decision rows")); }
    Ok(rows)
}

fn audit(rows: &[Row]) -> Result<Report, String> {
    let mut by_context: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
    let mut contexts_by_cluster: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for row in rows {
        by_context.entry(&row.context).or_default().push(row);
        contexts_by_cluster.entry(&row.cluster).or_default().insert(&row.context);
    }

    let mut unsupported_pairs = 0usize;
    let mut total_unsupported_mass = 0.0;
    let mut selected_count = 0usize;
    let mut weight_sum = 0.0;
    let mut weight_sq_sum = 0.0;
    let mut weighted_reward_sum = 0.0;
    let mut max_observed_weight: f64 = 0.0;
    let mut max_supported_weight: f64 = 0.0;

    for (context, actions) in &by_context {
        let target_sum: f64 = actions.iter().map(|row| row.target_p).sum();
        let logging_sum: f64 = actions.iter().map(|row| row.logging_p).sum();
        if (target_sum - 1.0).abs() > EPS {
            return Err(fail(format!("context {context}: target probabilities sum to {target_sum}, not 1")));
        }
        if (logging_sum - 1.0).abs() > EPS {
            return Err(fail(format!("context {context}: logging probabilities sum to {logging_sum}, not 1")));
        }
        if actions.iter().any(|row| row.cluster != actions[0].cluster) {
            return Err(fail(format!("context {context}: action rows disagree on cluster_id")));
        }
        let selected: Vec<&Row> = actions.iter().copied().filter(|row| row.selected).collect();
        if selected.len() != 1 {
            return Err(fail(format!("context {context}: exactly one logged action must be selected")));
        }
        let chosen = selected[0];
        if chosen.logging_p <= 0.0 {
            return Err(fail(format!("context {context}: selected action has zero logging propensity")));
        }
        selected_count += 1;
        let observed_weight = chosen.target_p / chosen.logging_p;
        if !observed_weight.is_finite() {
            return Err(fail(format!("context {context}: non-finite importance weight")));
        }
        weight_sum += observed_weight;
        weight_sq_sum += observed_weight * observed_weight;
        weighted_reward_sum += observed_weight * chosen.reward.unwrap_or(0.0);
        max_observed_weight = max_observed_weight.max(observed_weight);

        let mut context_unsupported_mass = 0.0;
        for row in actions {
            if row.target_p > 0.0 && row.logging_p <= 0.0 {
                unsupported_pairs += 1;
                context_unsupported_mass += row.target_p;
            } else if row.target_p > 0.0 {
                max_supported_weight = max_supported_weight.max(row.target_p / row.logging_p);
            }
        }
        total_unsupported_mass += context_unsupported_mass;
    }

    let context_count = by_context.len();
    let effective_sample_size = if weight_sq_sum > 0.0 { weight_sum * weight_sum / weight_sq_sum } else { 0.0 };
    let mean_target_mass_outside_support = total_unsupported_mass / context_count as f64;
    let (status, ips_estimate) = if unsupported_pairs > 0 {
        ("NOT_IDENTIFIED", None)
    } else if effective_sample_size < MIN_EFFECTIVE_SAMPLE_SIZE || max_supported_weight > MAX_SUPPORTED_WEIGHT {
        ("SUPPORTED_BUT_WEAK_OVERLAP", Some(weighted_reward_sum / context_count as f64))
    } else {
        ("SUPPORTED_DIAGNOSTIC_ONLY", Some(weighted_reward_sum / context_count as f64))
    };

    let declared_cluster_count = contexts_by_cluster.len();
    let max_contexts_per_cluster = contexts_by_cluster.values().map(BTreeSet::len).max().unwrap_or(0);
    let contexts_in_repeated_clusters = contexts_by_cluster.values().filter(|contexts| contexts.len() > 1).map(BTreeSet::len).sum();

    Ok(Report {
        status, context_count, selected_count, unsupported_pairs, mean_target_mass_outside_support,
        max_observed_weight, max_supported_weight, effective_sample_size, ips_estimate,
        declared_cluster_count, max_contexts_per_cluster, contexts_in_repeated_clusters,
    })
}

fn render(report: &Report) -> String {
    let mut output = String::new();
    output.push_str("VIS_DVPC_027_AUDIT\tformat\tv1\n");
    output.push_str(&format!("SUMMARY\tstatus\t{}\n", report.status));
    output.push_str(&format!("SUMMARY\tcontext_count\t{}\n", report.context_count));
    output.push_str(&format!("SUMMARY\tselected_count\t{}\n", report.selected_count));
    output.push_str(&format!("SUMMARY\tunsupported_context_action_pairs\t{}\n", report.unsupported_pairs));
    output.push_str(&format!("SUMMARY\tmean_target_mass_outside_support\t{:.12}\n", report.mean_target_mass_outside_support));
    output.push_str(&format!("SUMMARY\tmax_observed_importance_weight\t{:.12}\n", report.max_observed_weight));
    output.push_str(&format!("SUMMARY\tmax_supported_policy_weight\t{:.12}\n", report.max_supported_weight));
    output.push_str(&format!("SUMMARY\teffective_sample_size_diagnostic\t{:.12}\n", report.effective_sample_size));
    output.push_str(&format!("SUMMARY\tdeclared_cluster_count\t{}\n", report.declared_cluster_count));
    output.push_str(&format!("SUMMARY\tmax_contexts_per_declared_cluster\t{}\n", report.max_contexts_per_cluster));
    output.push_str(&format!("SUMMARY\tcontexts_in_repeated_clusters\t{}\n", report.contexts_in_repeated_clusters));
    match report.ips_estimate {
        Some(value) => output.push_str(&format!("SUMMARY\tips_point_estimate_diagnostic_only\t{value:.12}\n")),
        None => output.push_str("SUMMARY\tips_point_estimate_diagnostic_only\tNA_NOT_IDENTIFIED\n"),
    }
    output.push_str("LIMIT\tno confidence interval or authorization claim is computed\n");
    output.push_str("LIMIT\tdeclared propensities are not independently verified\n");
    output.push_str("LIMIT\tcluster counts are descriptive, not cluster-robust inference\n");
    output
}

fn run(args: &[String]) -> Result<String, String> {
    let input = match args {
        [_program, flag, path] if flag == "--audit" => fs::read_to_string(path).map_err(|error| fail(format!("cannot read {path}: {error}")))?,
        [_program, flag] if flag == "--self-check-supported" => SUPPORTED_FIXTURE.to_owned(),
        [_program, flag] if flag == "--self-check-zero-support" => ZERO_SUPPORT_FIXTURE.to_owned(),
        _ => return Err("usage: vision-dvpc-support-audit --audit TRACE.tsv | --self-check-supported | --self-check-zero-support".into()),
    };
    Ok(render(&audit(&parse_trace(&input)?)?))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-027 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_fixture_is_diagnostic_not_authorization() {
        let report = audit(&parse_trace(SUPPORTED_FIXTURE).unwrap()).unwrap();
        assert_eq!(report.status, "SUPPORTED_DIAGNOSTIC_ONLY");
        assert_eq!(report.context_count, 12);
        assert_eq!(report.selected_count, 12);
        assert_eq!(report.unsupported_pairs, 0);
        assert!(report.effective_sample_size >= MIN_EFFECTIVE_SAMPLE_SIZE);
        assert!(report.max_supported_weight <= MAX_SUPPORTED_WEIGHT);
        assert!(report.ips_estimate.is_some());
    }

    #[test]
    fn target_mass_with_zero_logging_support_is_not_identified() {
        let report = audit(&parse_trace(ZERO_SUPPORT_FIXTURE).unwrap()).unwrap();
        assert_eq!(report.status, "NOT_IDENTIFIED");
        assert_eq!(report.unsupported_pairs, 4);
        assert!((report.mean_target_mass_outside_support - 0.3).abs() < EPS);
        assert!(report.ips_estimate.is_none());
    }

    #[test]
    fn weak_overlap_is_flagged_even_without_exact_zero_support() {
        let trace = "context_id\taction_id\ttarget_probability\tlogging_probability\tselected\treward\tcluster_id\nc1\ta\t0.5\t0.99\tyes\t0.1\tg1\nc1\tb\t0.5\t0.01\tno\tNA\tg1\nc2\ta\t0.5\t0.99\tyes\t0.1\tg2\nc2\tb\t0.5\t0.01\tno\tNA\tg2\nc3\ta\t0.5\t0.99\tyes\t0.1\tg3\nc3\tb\t0.5\t0.01\tno\tNA\tg3\nc4\ta\t0.5\t0.99\tyes\t0.1\tg4\nc4\tb\t0.5\t0.01\tno\tNA\tg4\nc5\ta\t0.5\t0.99\tyes\t0.1\tg5\nc5\tb\t0.5\t0.01\tno\tNA\tg5\nc6\ta\t0.5\t0.99\tyes\t0.1\tg6\nc6\tb\t0.5\t0.01\tno\tNA\tg6\n";
        let report = audit(&parse_trace(trace).unwrap()).unwrap();
        assert_eq!(report.status, "SUPPORTED_BUT_WEAK_OVERLAP");
        assert!(report.max_supported_weight > MAX_SUPPORTED_WEIGHT);
    }

    #[test]
    fn selected_action_with_zero_logging_probability_is_invalid() {
        let trace = "context_id\taction_id\ttarget_probability\tlogging_probability\tselected\treward\tcluster_id\nc1\ta\t1\t0\tyes\t0.5\tg1\nc1\tb\t0\t1\tno\tNA\tg1\n";
        assert!(audit(&parse_trace(trace).unwrap()).is_err());
    }

    #[test]
    fn unselected_counterfactual_rewards_are_rejected() {
        let trace = "context_id\taction_id\ttarget_probability\tlogging_probability\tselected\treward\tcluster_id\nc1\ta\t0.5\t0.5\tyes\t0.5\tg1\nc1\tb\t0.5\t0.5\tno\t0.9\tg1\n";
        assert!(parse_trace(trace).is_err());
    }

    #[test]
    fn probability_sums_must_equal_one() {
        let trace = "context_id\taction_id\ttarget_probability\tlogging_probability\tselected\treward\tcluster_id\nc1\ta\t0.6\t0.5\tyes\t0.2\tg1\nc1\tb\t0.5\t0.5\tno\tNA\tg1\n";
        assert!(audit(&parse_trace(trace).unwrap()).is_err());
    }

    #[test]
    fn repeated_clusters_are_reported_without_claiming_cluster_validity() {
        let trace = "context_id\taction_id\ttarget_probability\tlogging_probability\tselected\treward\tcluster_id\nc1\ta\t1\t1\tyes\t0.5\tg1\nc2\ta\t1\t1\tyes\t0.6\tg1\nc3\ta\t1\t1\tyes\t0.7\tg2\n";
        let report = audit(&parse_trace(trace).unwrap()).unwrap();
        assert_eq!(report.declared_cluster_count, 2);
        assert_eq!(report.max_contexts_per_cluster, 2);
        assert_eq!(report.contexts_in_repeated_clusters, 2);
        assert!(render(&report).contains("cluster counts are descriptive"));
    }

    #[test]
    fn malformed_and_duplicate_rows_fail_closed() {
        assert!(parse_trace("bad header\n").is_err());
        let duplicate = "context_id\taction_id\ttarget_probability\tlogging_probability\tselected\treward\tcluster_id\nc1\ta\t0.5\t0.5\tyes\t0.5\tg1\nc1\ta\t0.5\t0.5\tno\tNA\tg1\n";
        assert!(parse_trace(duplicate).is_err());
    }
}
