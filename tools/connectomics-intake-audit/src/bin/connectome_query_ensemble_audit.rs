//! Evaluate scientific-query decision stability across explicit connectome scenarios.
//!
//! V1 accepts a finite, unweighted set of provenance-bound graph reconstructions.
//! It intentionally makes no probability/posterior claim. Agreement means only
//! robustness within the exact submitted scenario set and declared scope.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-query-ensemble-audit",
    version,
    about = "Audit scientific-query decision stability across finite connectome scenarios"
)]
struct Args {
    /// Finite scenario-set input JSON.
    #[arg(long)]
    input: PathBuf,
    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    analysis_id: String,
    dataset_revision_commit_sha: String,
    scope_coverage: String,
    scenarios: Vec<Scenario>,
    queries: Vec<Query>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    graph_sha256: String,
    lineage_groups: Vec<String>,
    edges: Vec<Edge>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    source: u64,
    target: u64,
    weight: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: String,
    metric: QueryMetric,
    decision: DecisionRule,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum QueryMetric {
    EdgeWeight { source: u64, target: u64 },
    InDegree { node: u64 },
    OutDegree { node: u64 },
    PathExists { source: u64, target: u64 },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum DecisionRule {
    BooleanIs { value: bool },
    GreaterThan { threshold: f64 },
    GreaterOrEqual { threshold: f64 },
    LessThan { threshold: f64 },
    LessOrEqual { threshold: f64 },
    BetweenInclusive { lower: f64, upper: f64 },
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    analysis_id: String,
    input_sha256: String,
    dataset_revision_commit_sha: String,
    scope_coverage: String,
    scenario_count: usize,
    scenario_set_sha256: String,
    unique_lineage_group_count: usize,
    queries: Vec<QueryReport>,
    status: &'static str,
    probability_claim: &'static str,
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct QueryReport {
    query_id: String,
    metric: &'static str,
    decision_rule: String,
    minimum_value: f64,
    maximum_value: f64,
    range_width: f64,
    scenario_count: usize,
    scenarios_meeting_decision: usize,
    scenarios_not_meeting_decision: usize,
    decision_status: &'static str,
    scenario_results: Vec<ScenarioResult>,
}

#[derive(Debug, Serialize)]
struct ScenarioResult {
    scenario_id: String,
    graph_sha256: String,
    value: f64,
    decision_met: bool,
}

type Graph = BTreeMap<(u64, u64), f64>;

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_graph_sha256(graph: &Graph) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-query-scenario-graph-v1");
    for ((source, target), weight) in graph {
        hasher.update(source.to_be_bytes());
        hasher.update(target.to_be_bytes());
        hasher.update(weight.to_bits().to_be_bytes());
    }
    hex::encode(hasher.finalize())
}

fn graph_for(scenario: &Scenario) -> Result<Graph, String> {
    let mut graph = BTreeMap::new();
    for edge in &scenario.edges {
        if !edge.weight.is_finite() || edge.weight <= 0.0 {
            return Err(format!(
                "scenario {} contains non-finite or non-positive edge weight",
                scenario.scenario_id
            ));
        }
        if graph
            .insert((edge.source, edge.target), edge.weight)
            .is_some()
        {
            return Err(format!(
                "scenario {} contains duplicate directed edge {} -> {}",
                scenario.scenario_id, edge.source, edge.target
            ));
        }
    }
    let actual = canonical_graph_sha256(&graph);
    if actual != scenario.graph_sha256 {
        return Err(format!(
            "scenario {} graph digest mismatch: declared {}, computed {}",
            scenario.scenario_id, scenario.graph_sha256, actual
        ));
    }
    Ok(graph)
}

fn validate(input: &Input) -> Result<(), String> {
    if input.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if input.analysis_id.trim().is_empty() {
        return Err("analysis_id must be non-empty".into());
    }
    if !valid_hex(&input.dataset_revision_commit_sha, 40) {
        return Err("dataset_revision_commit_sha must be 40 lowercase hexadecimal characters".into());
    }
    if !matches!(
        input.scope_coverage.as_str(),
        "exhaustive_declared_scenario_space" | "sampled_sensitivity_set"
    ) {
        return Err(
            "scope_coverage must be exhaustive_declared_scenario_space or sampled_sensitivity_set"
                .into(),
        );
    }
    if input.scenarios.len() < 2 {
        return Err("at least two explicit scenarios are required".into());
    }
    if input.queries.is_empty() {
        return Err("at least one scientific query is required".into());
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in &input.scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.clone())
        {
            return Err("scenario IDs must be non-empty and unique".into());
        }
        if !valid_hex(&scenario.graph_sha256, 64) {
            return Err(format!(
                "scenario {} graph_sha256 must be lowercase SHA-256",
                scenario.scenario_id
            ));
        }
        if scenario.lineage_groups.is_empty() {
            return Err(format!(
                "scenario {} must declare at least one lineage group; use an explicit group for shared ancestry",
                scenario.scenario_id
            ));
        }
        let mut groups = BTreeSet::new();
        for group in &scenario.lineage_groups {
            if group.trim().is_empty() || !groups.insert(group) {
                return Err(format!(
                    "scenario {} lineage_groups must be non-empty and unique",
                    scenario.scenario_id
                ));
            }
        }
        graph_for(scenario)?;
    }

    let mut query_ids = BTreeSet::new();
    for query in &input.queries {
        if query.query_id.trim().is_empty() || !query_ids.insert(query.query_id.clone()) {
            return Err("query IDs must be non-empty and unique".into());
        }
        match (&query.metric, &query.decision) {
            (QueryMetric::PathExists { .. }, DecisionRule::BooleanIs { .. }) => {}
            (QueryMetric::PathExists { .. }, _) => {
                return Err(format!(
                    "query {} uses path_exists and requires boolean_is decision",
                    query.query_id
                ));
            }
            (_, DecisionRule::BooleanIs { .. }) => {
                return Err(format!(
                    "query {} uses a numeric metric and cannot use boolean_is",
                    query.query_id
                ));
            }
            (_, DecisionRule::GreaterThan { threshold })
            | (_, DecisionRule::GreaterOrEqual { threshold })
            | (_, DecisionRule::LessThan { threshold })
            | (_, DecisionRule::LessOrEqual { threshold })
                if !threshold.is_finite() =>
            {
                return Err(format!("query {} threshold must be finite", query.query_id));
            }
            (_, DecisionRule::BetweenInclusive { lower, upper })
                if !lower.is_finite() || !upper.is_finite() || lower > upper =>
            {
                return Err(format!(
                    "query {} between_inclusive requires finite lower <= upper",
                    query.query_id
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn path_exists(graph: &Graph, source: u64, target: u64) -> bool {
    if source == target {
        return true;
    }
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::from([source]);
    visited.insert(source);
    while let Some(node) = queue.pop_front() {
        for ((edge_source, edge_target), weight) in graph {
            if *edge_source != node || *weight <= 0.0 {
                continue;
            }
            if *edge_target == target {
                return true;
            }
            if visited.insert(*edge_target) {
                queue.push_back(*edge_target);
            }
        }
    }
    false
}

fn query_value(graph: &Graph, metric: &QueryMetric) -> f64 {
    match metric {
        QueryMetric::EdgeWeight { source, target } => {
            graph.get(&(*source, *target)).copied().unwrap_or(0.0)
        }
        QueryMetric::InDegree { node } => graph
            .iter()
            .filter(|((_, target), _)| target == node)
            .map(|(_, weight)| *weight)
            .sum(),
        QueryMetric::OutDegree { node } => graph
            .iter()
            .filter(|((source, _), _)| source == node)
            .map(|(_, weight)| *weight)
            .sum(),
        QueryMetric::PathExists { source, target } => {
            if path_exists(graph, *source, *target) {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn decision_met(value: f64, rule: &DecisionRule) -> bool {
    match rule {
        DecisionRule::BooleanIs { value: expected } => (value == 1.0) == *expected,
        DecisionRule::GreaterThan { threshold } => value > *threshold,
        DecisionRule::GreaterOrEqual { threshold } => value >= *threshold,
        DecisionRule::LessThan { threshold } => value < *threshold,
        DecisionRule::LessOrEqual { threshold } => value <= *threshold,
        DecisionRule::BetweenInclusive { lower, upper } => value >= *lower && value <= *upper,
    }
}

fn metric_name(metric: &QueryMetric) -> &'static str {
    match metric {
        QueryMetric::EdgeWeight { .. } => "edge_weight",
        QueryMetric::InDegree { .. } => "in_degree",
        QueryMetric::OutDegree { .. } => "out_degree",
        QueryMetric::PathExists { .. } => "path_exists",
    }
}

fn decision_description(rule: &DecisionRule) -> String {
    match rule {
        DecisionRule::BooleanIs { value } => format!("boolean_is:{value}"),
        DecisionRule::GreaterThan { threshold } => format!("greater_than:{threshold}"),
        DecisionRule::GreaterOrEqual { threshold } => format!("greater_or_equal:{threshold}"),
        DecisionRule::LessThan { threshold } => format!("less_than:{threshold}"),
        DecisionRule::LessOrEqual { threshold } => format!("less_or_equal:{threshold}"),
        DecisionRule::BetweenInclusive { lower, upper } => {
            format!("between_inclusive:{lower}:{upper}")
        }
    }
}

fn scenario_set_sha256(scenarios: &[Scenario]) -> Result<(String, usize), String> {
    let mut rows = scenarios
        .iter()
        .map(|scenario| {
            let mut groups = scenario.lineage_groups.clone();
            groups.sort();
            (scenario.scenario_id.clone(), scenario.graph_sha256.clone(), groups)
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-query-finite-scenario-set-v1");
    let mut all_groups = BTreeSet::new();
    for (scenario_id, graph_sha, groups) in rows {
        hasher.update((scenario_id.len() as u64).to_be_bytes());
        hasher.update(scenario_id.as_bytes());
        hasher.update(graph_sha.as_bytes());
        for group in groups {
            hasher.update((group.len() as u64).to_be_bytes());
            hasher.update(group.as_bytes());
            all_groups.insert(group);
        }
    }
    Ok((hex::encode(hasher.finalize()), all_groups.len()))
}

fn run(input: &Input, input_bytes: &[u8]) -> Result<Report, String> {
    validate(input)?;
    let (scenario_set_sha, lineage_group_count) = scenario_set_sha256(&input.scenarios)?;
    let mut scenarios = input.scenarios.iter().collect::<Vec<_>>();
    scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let mut query_reports = Vec::with_capacity(input.queries.len());
    for query in &input.queries {
        let mut results = Vec::with_capacity(scenarios.len());
        for scenario in &scenarios {
            let graph = graph_for(scenario)?;
            let value = query_value(&graph, &query.metric);
            if !value.is_finite() {
                return Err(format!(
                    "query {} produced a non-finite value in scenario {}",
                    query.query_id, scenario.scenario_id
                ));
            }
            results.push(ScenarioResult {
                scenario_id: scenario.scenario_id.clone(),
                graph_sha256: scenario.graph_sha256.clone(),
                value,
                decision_met: decision_met(value, &query.decision),
            });
        }
        let minimum_value = results
            .iter()
            .map(|result| result.value)
            .min_by(f64::total_cmp)
            .ok_or_else(|| format!("query {} has no scenario results", query.query_id))?;
        let maximum_value = results
            .iter()
            .map(|result| result.value)
            .max_by(f64::total_cmp)
            .ok_or_else(|| format!("query {} has no scenario results", query.query_id))?;
        let scenarios_meeting_decision = results.iter().filter(|result| result.decision_met).count();
        let scenarios_not_meeting_decision = results.len() - scenarios_meeting_decision;
        let decision_status = if scenarios_meeting_decision == results.len() {
            "ROBUST_DECISION_MET"
        } else if scenarios_not_meeting_decision == results.len() {
            "ROBUST_DECISION_NOT_MET"
        } else {
            "SENSITIVE_TO_RECONSTRUCTION"
        };
        query_reports.push(QueryReport {
            query_id: query.query_id.clone(),
            metric: metric_name(&query.metric),
            decision_rule: decision_description(&query.decision),
            minimum_value,
            maximum_value,
            range_width: maximum_value - minimum_value,
            scenario_count: results.len(),
            scenarios_meeting_decision,
            scenarios_not_meeting_decision,
            decision_status,
            scenario_results: results,
        });
    }

    Ok(Report {
        schema_version: 1,
        tool: "connectome-query-ensemble-audit",
        analysis_id: input.analysis_id.clone(),
        input_sha256: sha256(input_bytes),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        scope_coverage: input.scope_coverage.clone(),
        scenario_count: scenarios.len(),
        scenario_set_sha256: scenario_set_sha,
        unique_lineage_group_count: lineage_group_count,
        queries: query_reports,
        status: "EVALUATED_FINITE_SCENARIO_SET",
        probability_claim: "NONE_FINITE_SCENARIO_SET_ONLY",
        scope_note: if input.scope_coverage == "exhaustive_declared_scenario_space" {
            "Decision stability is established only across the exact exhaustive scenario space declared by the input; this does not prove that the scenario space contains every biologically plausible reconstruction."
        } else {
            "Decision stability is established only across the submitted sensitivity scenarios; sampled scenario agreement is not evidence of robustness over all plausible reconstructions."
        },
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let input_bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("could not read {}: {error}", args.input.display());
            return ExitCode::from(2);
        }
    };
    let input = match serde_json::from_slice::<Input>(&input_bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid scenario-set JSON: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match run(&input, &input_bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("connectome-query-ensemble-audit failed: {error}");
            return ExitCode::from(1);
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("could not serialize report: {error}");
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            return ExitCode::from(2);
        }
    } else {
        println!("{rendered}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_sha(edges: &[Edge]) -> String {
        let graph = edges
            .iter()
            .map(|edge| ((edge.source, edge.target), edge.weight))
            .collect::<Graph>();
        canonical_graph_sha256(&graph)
    }

    fn scenario(id: &str, edges: Vec<Edge>) -> Scenario {
        let digest = graph_sha(&edges);
        Scenario {
            scenario_id: id.into(),
            graph_sha256: digest,
            lineage_groups: vec!["segmentation-family-A".into()],
            edges,
        }
    }

    fn edge(source: u64, target: u64, weight: f64) -> Edge {
        Edge { source, target, weight }
    }

    fn fixture() -> Input {
        Input {
            schema_version: 1,
            analysis_id: "synthetic-query-reconstruction-fixture-v1".into(),
            dataset_revision_commit_sha: "a".repeat(40),
            scope_coverage: "exhaustive_declared_scenario_space".into(),
            scenarios: vec![
                scenario(
                    "path-present",
                    vec![
                        edge(1, 2, 1.0),
                        edge(2, 3, 1.0),
                        edge(3, 4, 1.0),
                        edge(10, 11, 1.0),
                    ],
                ),
                scenario(
                    "path-absent",
                    vec![
                        edge(1, 2, 1.0),
                        edge(3, 4, 1.0),
                        edge(10, 11, 0.25),
                    ],
                ),
                scenario(
                    "path-present-alternative",
                    vec![
                        edge(1, 2, 1.0),
                        edge(2, 3, 1.0),
                        edge(3, 4, 1.0),
                    ],
                ),
            ],
            queries: vec![
                Query {
                    query_id: "critical-path-1-to-4".into(),
                    metric: QueryMetric::PathExists { source: 1, target: 4 },
                    decision: DecisionRule::BooleanIs { value: true },
                },
                Query {
                    query_id: "spurious-edge-below-threshold".into(),
                    metric: QueryMetric::EdgeWeight { source: 10, target: 11 },
                    decision: DecisionRule::LessThan { threshold: 0.5 },
                },
            ],
        }
    }

    #[test]
    fn fixture_exposes_query_sensitive_to_reconstruction_and_robust_control_query() {
        let input = fixture();
        let bytes = b"synthetic-input";
        let report = run(&input, &bytes).expect("finite scenario fixture should evaluate");
        assert_eq!(report.probability_claim, "NONE_FINITE_SCENARIO_SET_ONLY");
        let path = report
            .queries
            .iter()
            .find(|query| query.query_id == "critical-path-1-to-4")
            .unwrap();
        assert_eq!(path.decision_status, "SENSITIVE_TO_RECONSTRUCTION");
        assert_eq!(path.scenarios_meeting_decision, 2);
        assert_eq!(path.scenarios_not_meeting_decision, 1);
        let edge_control = report
            .queries
            .iter()
            .find(|query| query.query_id == "spurious-edge-below-threshold")
            .unwrap();
        assert_eq!(edge_control.decision_status, "ROBUST_DECISION_MET");
        assert_eq!(edge_control.scenarios_meeting_decision, 3);
    }

    #[test]
    fn scenario_order_does_not_change_canonical_scenario_set_digest() {
        let mut input = fixture();
        let before = scenario_set_sha256(&input.scenarios).unwrap().0;
        input.scenarios.reverse();
        let after = scenario_set_sha256(&input.scenarios).unwrap().0;
        assert_eq!(before, after);
    }

    #[test]
    fn scenario_result_order_is_canonical() {
        let mut input = fixture();
        input.scenarios.reverse();
        let bytes = b"synthetic-input";
        let report = run(&input, &bytes).unwrap();
        let ids = report.queries[0]
            .scenario_results
            .iter()
            .map(|result| result.scenario_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["path-absent", "path-present", "path-present-alternative"]);
    }

    #[test]
    fn graph_digest_mismatch_fails_closed() {
        let mut input = fixture();
        input.scenarios[0].graph_sha256 = "0".repeat(64);
        assert!(validate(&input).unwrap_err().contains("graph digest mismatch"));
    }

    #[test]
    fn duplicate_scenario_id_fails_closed() {
        let mut input = fixture();
        let duplicate_id = input.scenarios[0].scenario_id.clone();
        input.scenarios[1].scenario_id = duplicate_id;
        assert!(validate(&input).unwrap_err().contains("scenario IDs"));
    }

    #[test]
    fn duplicate_directed_edge_fails_closed() {
        let mut input = fixture();
        input.scenarios[0].edges.push(edge(1, 2, 0.5));
        input.scenarios[0].graph_sha256 = graph_sha(&input.scenarios[0].edges);
        assert!(validate(&input).unwrap_err().contains("duplicate directed edge"));
    }

    #[test]
    fn boolean_path_query_rejects_numeric_decision_rule() {
        let mut input = fixture();
        input.queries[0].decision = DecisionRule::GreaterThan { threshold: 0.5 };
        assert!(validate(&input).unwrap_err().contains("requires boolean_is"));
    }

    #[test]
    fn path_search_terminates_with_cycles() {
        let graph = vec![
            edge(1, 2, 1.0),
            edge(2, 1, 1.0),
            edge(2, 3, 1.0),
        ]
        .into_iter()
        .map(|edge| ((edge.source, edge.target), edge.weight))
        .collect::<Graph>();
        assert!(path_exists(&graph, 1, 3));
        assert!(!path_exists(&graph, 3, 1));
    }

    #[test]
    fn sampled_sensitivity_set_never_claims_exhaustive_space() {
        let mut input = fixture();
        input.scope_coverage = "sampled_sensitivity_set".into();
        let bytes = b"synthetic-input";
        let report = run(&input, bytes).unwrap();
        assert!(report.scope_note.contains("not evidence of robustness over all plausible"));
    }
}
