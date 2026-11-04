//! Exact, bounded search for minimum-cost admissible edit sets that flip a frozen
//! connectome query decision. This is a counterexample finder, not a certificate
//! of biological truth or of completeness of the declared edit universe.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const MAX_EDIT_ATOMS: usize = 20;
const MAX_RETURNED_MINIMAL_WITNESSES: usize = 128;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-query-flipping-witness-audit",
    version,
    about = "Find exact minimum-cost admissible edit witnesses that flip a scientific query"
)]
struct Args {
    /// Frozen baseline graph, query, edit universe and effort budget.
    #[arg(long)]
    input: PathBuf,
    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    analysis_id: String,
    dataset_revision_commit_sha: String,
    edit_universe_scope: String,
    baseline_graph_sha256: String,
    baseline_graph: Vec<Edge>,
    query: Query,
    effort_unit: String,
    max_total_effort_units: u64,
    edits: Vec<EditAtom>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    source: u64,
    target: u64,
    weight: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: String,
    metric: QueryMetric,
    decision: DecisionRule,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum QueryMetric {
    EdgeWeight { source: u64, target: u64 },
    InDegree { node: u64 },
    OutDegree { node: u64 },
    PathExists { source: u64, target: u64 },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum DecisionRule {
    BooleanIs { value: bool },
    GreaterThan { threshold: f64 },
    GreaterOrEqual { threshold: f64 },
    LessThan { threshold: f64 },
    LessOrEqual { threshold: f64 },
    BetweenInclusive { lower: f64, upper: f64 },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum EditKind {
    GraphEdit,
    EvidenceGate,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EdgeChange {
    source: u64,
    target: u64,
    /// Zero represents an absent edge. Positive values represent exact weights.
    expected_weight: f64,
    /// Zero deletes the edge. Positive values set its exact replacement weight.
    replacement_weight: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EditAtom {
    edit_id: String,
    kind: EditKind,
    effort_units: u64,
    evidence_sha256: String,
    requires: Vec<String>,
    conflicts_with: Vec<String>,
    changes: Vec<EdgeChange>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    analysis_id: String,
    input_sha256: String,
    dataset_revision_commit_sha: String,
    edit_universe_scope: String,
    baseline_graph_sha256: String,
    query_id: String,
    query_spec_sha256: String,
    effort_unit: String,
    edit_universe_sha256: String,
    baseline_query_value: f64,
    baseline_decision_met: bool,
    edit_atom_count: usize,
    max_total_effort_units: u64,
    subset_count_evaluated: u64,
    budget_rejected_subset_count: u64,
    constraint_rejected_subset_count: u64,
    admissible_subset_count_within_budget: u64,
    decision_flipping_subset_count_within_budget: u64,
    minimal_witness_effort_units: Option<u64>,
    minimal_witness_count: u64,
    witnesses_returned: usize,
    minimal_witnesses_truncated: bool,
    witnesses: Vec<Witness>,
    status: &'static str,
    search_complete: bool,
    probability_claim: &'static str,
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct Witness {
    edit_ids: Vec<String>,
    effort_units: u64,
    resulting_graph_sha256: String,
    resulting_query_value: f64,
    resulting_decision_met: bool,
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

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn graph_from_edges(edges: &[Edge]) -> Result<Graph, String> {
    let mut graph = BTreeMap::new();
    for edge in edges {
        if edge.source == edge.target {
            return Err(format!(
                "self-loop {} -> {} is not admitted",
                edge.source, edge.target
            ));
        }
        if !edge.weight.is_finite() || edge.weight <= 0.0 {
            return Err("baseline graph weights must be finite and strictly positive".into());
        }
        if graph
            .insert((edge.source, edge.target), edge.weight)
            .is_some()
        {
            return Err(format!(
                "duplicate directed edge {} -> {}",
                edge.source, edge.target
            ));
        }
    }
    Ok(graph)
}

fn graph_sha256(graph: &Graph) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-query-scenario-graph-v1");
    for ((source, target), weight) in graph {
        hasher.update(source.to_be_bytes());
        hasher.update(target.to_be_bytes());
        hasher.update(weight.to_bits().to_be_bytes());
    }
    hex::encode(hasher.finalize())
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

fn path_exists(graph: &Graph, source: u64, target: u64) -> bool {
    if source == target {
        return true;
    }
    let mut adjacency = BTreeMap::<u64, Vec<u64>>::new();
    for ((from, to), weight) in graph {
        if *weight > 0.0 {
            adjacency.entry(*from).or_default().push(*to);
        }
    }
    let mut visited = BTreeSet::from([source]);
    let mut queue = VecDeque::from([source]);
    while let Some(node) = queue.pop_front() {
        for next in adjacency.get(&node).into_iter().flatten() {
            if *next == target {
                return true;
            }
            if visited.insert(*next) {
                queue.push_back(*next);
            }
        }
    }
    false
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

fn validate_query(query: &Query) -> Result<(), String> {
    if !nonempty(&query.query_id) {
        return Err("query_id must be non-empty".into());
    }
    match (&query.metric, &query.decision) {
        (QueryMetric::PathExists { .. }, DecisionRule::BooleanIs { .. }) => {}
        (QueryMetric::PathExists { .. }, _) => {
            return Err("path_exists requires a boolean_is decision".into());
        }
        (_, DecisionRule::BooleanIs { .. }) => {
            return Err("numeric graph metrics cannot use boolean_is".into());
        }
        (_, DecisionRule::GreaterThan { threshold })
        | (_, DecisionRule::GreaterOrEqual { threshold })
        | (_, DecisionRule::LessThan { threshold })
        | (_, DecisionRule::LessOrEqual { threshold })
            if !threshold.is_finite() =>
        {
            return Err("query threshold must be finite".into());
        }
        (_, DecisionRule::BetweenInclusive { lower, upper })
            if !lower.is_finite() || !upper.is_finite() || lower > upper =>
        {
            return Err("between_inclusive requires finite lower <= upper".into());
        }
        _ => {}
    }
    Ok(())
}

fn validate_requirement_dag(
    edits: &[EditAtom],
    indices: &BTreeMap<String, usize>,
) -> Result<(), String> {
    fn visit(
        index: usize,
        edits: &[EditAtom],
        indices: &BTreeMap<String, usize>,
        colors: &mut [u8],
    ) -> Result<(), String> {
        match colors[index] {
            1 => return Err("edit requirement graph contains a cycle".into()),
            2 => return Ok(()),
            _ => {}
        }
        colors[index] = 1;
        for required in &edits[index].requires {
            let child = *indices.get(required).ok_or_else(|| {
                format!(
                    "edit {} requires unknown edit {}",
                    edits[index].edit_id, required
                )
            })?;
            visit(child, edits, indices, colors)?;
        }
        colors[index] = 2;
        Ok(())
    }

    let mut colors = vec![0_u8; edits.len()];
    for index in 0..edits.len() {
        visit(index, edits, indices, &mut colors)?;
    }
    Ok(())
}

fn validate(input: &Input) -> Result<Graph, String> {
    if input.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if !nonempty(&input.analysis_id) {
        return Err("analysis_id must be non-empty".into());
    }
    if !valid_hex(&input.dataset_revision_commit_sha, 40) {
        return Err(
            "dataset_revision_commit_sha must be 40 lowercase hexadecimal characters".into(),
        );
    }
    if !matches!(
        input.edit_universe_scope.as_str(),
        "exhaustive_declared_edit_universe" | "bounded_candidate_subset"
    ) {
        return Err("edit_universe_scope must be exhaustive_declared_edit_universe or bounded_candidate_subset".into());
    }
    if !valid_hex(&input.baseline_graph_sha256, 64) {
        return Err("baseline_graph_sha256 must be lowercase SHA-256".into());
    }
    if !nonempty(&input.effort_unit) {
        return Err("effort_unit must be non-empty and explicit".into());
    }
    if input.max_total_effort_units == 0 {
        return Err("max_total_effort_units must be positive".into());
    }
    let graph = graph_from_edges(&input.baseline_graph)?;
    let actual_graph_hash = graph_sha256(&graph);
    if actual_graph_hash != input.baseline_graph_sha256 {
        return Err(format!(
            "baseline graph digest mismatch: declared {}, computed {}",
            input.baseline_graph_sha256, actual_graph_hash
        ));
    }
    validate_query(&input.query)?;
    if input.edits.is_empty() || input.edits.len() > MAX_EDIT_ATOMS {
        return Err(format!(
            "edits must contain between 1 and {MAX_EDIT_ATOMS} atoms for exact enumeration"
        ));
    }

    let mut indices = BTreeMap::<String, usize>::new();
    let mut total_effort = 0_u64;
    for (index, edit) in input.edits.iter().enumerate() {
        if !nonempty(&edit.edit_id) || indices.insert(edit.edit_id.clone(), index).is_some() {
            return Err("edit IDs must be non-empty and unique".into());
        }
        if edit.effort_units == 0 {
            return Err(format!(
                "edit {} effort_units must be positive",
                edit.edit_id
            ));
        }
        total_effort = total_effort
            .checked_add(edit.effort_units)
            .ok_or_else(|| "sum of edit effort_units overflows u64".to_string())?;
        if !valid_hex(&edit.evidence_sha256, 64) {
            return Err(format!(
                "edit {} evidence_sha256 must be lowercase SHA-256",
                edit.edit_id
            ));
        }
        match edit.kind {
            EditKind::GraphEdit if edit.changes.is_empty() => {
                return Err(format!(
                    "graph_edit {} must contain at least one edge change",
                    edit.edit_id
                ));
            }
            EditKind::EvidenceGate if !edit.changes.is_empty() => {
                return Err(format!(
                    "evidence_gate {} must not mutate the graph",
                    edit.edit_id
                ));
            }
            _ => {}
        }

        let mut required_ids = BTreeSet::new();
        for required in &edit.requires {
            if required == &edit.edit_id || !required_ids.insert(required) {
                return Err(format!(
                    "edit {} has a self or duplicate requirement",
                    edit.edit_id
                ));
            }
        }
        let mut conflict_ids = BTreeSet::new();
        for conflict in &edit.conflicts_with {
            if conflict == &edit.edit_id || !conflict_ids.insert(conflict) {
                return Err(format!(
                    "edit {} has a self or duplicate conflict",
                    edit.edit_id
                ));
            }
        }

        let mut touched_edges = BTreeSet::new();
        for change in &edit.changes {
            if change.source == change.target {
                return Err(format!(
                    "edit {} contains a self-loop mutation",
                    edit.edit_id
                ));
            }
            if !change.expected_weight.is_finite()
                || change.expected_weight < 0.0
                || !change.replacement_weight.is_finite()
                || change.replacement_weight < 0.0
            {
                return Err(format!(
                    "edit {} contains a non-finite or negative edge weight",
                    edit.edit_id
                ));
            }
            if !touched_edges.insert((change.source, change.target)) {
                return Err(format!(
                    "edit {} mutates one directed edge more than once",
                    edit.edit_id
                ));
            }
            let current = graph
                .get(&(change.source, change.target))
                .copied()
                .unwrap_or(0.0);
            if current.to_bits() != change.expected_weight.to_bits() {
                return Err(format!(
                    "edit {} precondition mismatch for {} -> {}: expected {}, baseline has {}",
                    edit.edit_id, change.source, change.target, change.expected_weight, current
                ));
            }
        }
    }
    if total_effort == 0 {
        return Err("edit universe must have positive total effort".into());
    }

    for edit in &input.edits {
        for required in &edit.requires {
            if !indices.contains_key(required) {
                return Err(format!(
                    "edit {} requires unknown edit {}",
                    edit.edit_id, required
                ));
            }
        }
        for conflict in &edit.conflicts_with {
            let other_index = *indices.get(conflict).ok_or_else(|| {
                format!(
                    "edit {} conflicts with unknown edit {}",
                    edit.edit_id, conflict
                )
            })?;
            if !input.edits[other_index]
                .conflicts_with
                .contains(&edit.edit_id)
            {
                return Err(format!(
                    "conflict {} <-> {} must be declared symmetrically",
                    edit.edit_id, conflict
                ));
            }
        }
    }
    validate_requirement_dag(&input.edits, &indices)?;

    let mut edge_owners = BTreeMap::<(u64, u64), Vec<usize>>::new();
    for (index, edit) in input.edits.iter().enumerate() {
        for change in &edit.changes {
            edge_owners
                .entry((change.source, change.target))
                .or_default()
                .push(index);
        }
    }
    for owners in edge_owners.values() {
        for left in 0..owners.len() {
            for right in (left + 1)..owners.len() {
                let a = &input.edits[owners[left]];
                let b = &input.edits[owners[right]];
                if !a.conflicts_with.contains(&b.edit_id) || !b.conflicts_with.contains(&a.edit_id)
                {
                    return Err(format!(
                        "edits {} and {} mutate the same edge and must explicitly conflict",
                        a.edit_id, b.edit_id
                    ));
                }
            }
        }
    }
    Ok(graph)
}

fn canonical_query_sha256(query: &Query) -> Result<String, String> {
    let bytes = serde_json::to_vec(query)
        .map_err(|error| format!("could not serialize query specification: {error}"))?;
    Ok(sha256(&bytes))
}

fn canonical_edit_universe_sha256(edits: &[EditAtom]) -> Result<String, String> {
    let mut rows = edits.to_vec();
    for edit in &mut rows {
        edit.requires.sort();
        edit.conflicts_with.sort();
        edit.changes
            .sort_by_key(|change| (change.source, change.target));
    }
    rows.sort_by(|left, right| left.edit_id.cmp(&right.edit_id));
    let bytes = serde_json::to_vec(&rows)
        .map_err(|error| format!("could not serialize canonical edit universe: {error}"))?;
    Ok(sha256(&bytes))
}

fn requirement_and_conflict_masks(
    edits: &[EditAtom],
    indices: &BTreeMap<String, usize>,
) -> (Vec<u64>, Vec<u64>) {
    let mut requirements = vec![0_u64; edits.len()];
    let mut conflicts = vec![0_u64; edits.len()];
    for (index, edit) in edits.iter().enumerate() {
        for id in &edit.requires {
            requirements[index] |= 1_u64 << indices[id];
        }
        for id in &edit.conflicts_with {
            conflicts[index] |= 1_u64 << indices[id];
        }
    }
    (requirements, conflicts)
}

fn subset_is_admissible(mask: u64, requirements: &[u64], conflicts: &[u64]) -> bool {
    for index in 0..requirements.len() {
        if mask & (1_u64 << index) == 0 {
            continue;
        }
        if mask & requirements[index] != requirements[index] {
            return false;
        }
        if mask & conflicts[index] != 0 {
            return false;
        }
    }
    true
}

fn apply_subset(graph: &Graph, mask: u64, edits: &[EditAtom]) -> Graph {
    let mut result = graph.clone();
    for (index, edit) in edits.iter().enumerate() {
        if mask & (1_u64 << index) == 0 || edit.kind == EditKind::EvidenceGate {
            continue;
        }
        for change in &edit.changes {
            if change.replacement_weight == 0.0 {
                result.remove(&(change.source, change.target));
            } else {
                result.insert((change.source, change.target), change.replacement_weight);
            }
        }
    }
    result
}

fn selected_edit_ids(mask: u64, edits: &[EditAtom]) -> Vec<String> {
    let mut ids = edits
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1_u64 << index) != 0)
        .map(|(_, edit)| edit.edit_id.clone())
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn evaluate(input: &Input, input_bytes: &[u8]) -> Result<Report, String> {
    let baseline = validate(input)?;
    let query_sha = canonical_query_sha256(&input.query)?;
    let universe_sha = canonical_edit_universe_sha256(&input.edits)?;
    let baseline_value = query_value(&baseline, &input.query.metric);
    if !baseline_value.is_finite() {
        return Err("baseline query produced a non-finite value".into());
    }
    let baseline_met = decision_met(baseline_value, &input.query.decision);
    let indices = input
        .edits
        .iter()
        .enumerate()
        .map(|(index, edit)| (edit.edit_id.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let (requirements, conflicts) = requirement_and_conflict_masks(&input.edits, &indices);
    let total_subsets = (1_u64 << input.edits.len()) - 1;

    let mut budget_rejected = 0_u64;
    let mut constraint_rejected = 0_u64;
    let mut admissible = 0_u64;
    let mut flipping = 0_u64;
    let mut best_cost: Option<u64> = None;
    let mut minimal_count = 0_u64;
    let mut witnesses = Vec::new();

    for mask in 1..=total_subsets {
        let mut cost = 0_u64;
        for (index, edit) in input.edits.iter().enumerate() {
            if mask & (1_u64 << index) != 0 {
                cost = cost
                    .checked_add(edit.effort_units)
                    .ok_or_else(|| "subset effort sum overflows u64".to_string())?;
            }
        }
        if cost > input.max_total_effort_units {
            budget_rejected += 1;
            continue;
        }
        if !subset_is_admissible(mask, &requirements, &conflicts) {
            constraint_rejected += 1;
            continue;
        }
        admissible += 1;

        let graph = apply_subset(&baseline, mask, &input.edits);
        let value = query_value(&graph, &input.query.metric);
        if !value.is_finite() {
            return Err("an admissible edit subset produced a non-finite query value".into());
        }
        let met = decision_met(value, &input.query.decision);
        if met == baseline_met {
            continue;
        }
        flipping += 1;

        let replace_best = best_cost.is_none_or(|best| cost < best);
        if replace_best {
            best_cost = Some(cost);
            minimal_count = 0;
            witnesses.clear();
        }
        if best_cost == Some(cost) {
            minimal_count += 1;
            if witnesses.len() < MAX_RETURNED_MINIMAL_WITNESSES {
                witnesses.push(Witness {
                    edit_ids: selected_edit_ids(mask, &input.edits),
                    effort_units: cost,
                    resulting_graph_sha256: graph_sha256(&graph),
                    resulting_query_value: value,
                    resulting_decision_met: met,
                });
            }
        }
    }
    witnesses.sort_by(|left, right| left.edit_ids.cmp(&right.edit_ids));
    let status = if best_cost.is_some() {
        "COUNTEREXAMPLE_FOUND"
    } else {
        "NO_WITNESS_WITHIN_DECLARED_EDIT_UNIVERSE_AND_BUDGET"
    };

    Ok(Report {
        schema_version: 1,
        tool: "connectome-query-flipping-witness-audit",
        analysis_id: input.analysis_id.clone(),
        input_sha256: sha256(input_bytes),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        edit_universe_scope: input.edit_universe_scope.clone(),
        baseline_graph_sha256: input.baseline_graph_sha256.clone(),
        query_id: input.query.query_id.clone(),
        query_spec_sha256: query_sha,
        effort_unit: input.effort_unit.clone(),
        edit_universe_sha256: universe_sha,
        baseline_query_value: baseline_value,
        baseline_decision_met: baseline_met,
        edit_atom_count: input.edits.len(),
        max_total_effort_units: input.max_total_effort_units,
        subset_count_evaluated: total_subsets,
        budget_rejected_subset_count: budget_rejected,
        constraint_rejected_subset_count: constraint_rejected,
        admissible_subset_count_within_budget: admissible,
        decision_flipping_subset_count_within_budget: flipping,
        minimal_witness_effort_units: best_cost,
        minimal_witness_count: minimal_count,
        witnesses_returned: witnesses.len(),
        minimal_witnesses_truncated: minimal_count > MAX_RETURNED_MINIMAL_WITNESSES as u64,
        witnesses,
        status,
        search_complete: true,
        probability_claim: "NONE_DETERMINISTIC_FINITE_EDIT_UNIVERSE_ONLY",
        scope_note: "An exact search over all subsets of the declared edit atoms and effort budget. A no-witness result is bounded by this exact edit universe, its preconditions, dependency/conflict declarations, and budget; it does not establish that no biological counterexample exists outside the declared universe. This tool does not verify route observability, evidence-frame closure, provenance completeness, evidence independence, or biological correctness.",
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let input_bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("connectome-query-flipping-witness-audit: could not read input: {error}");
            return ExitCode::from(2);
        }
    };
    let input: Input = match serde_json::from_slice(&input_bytes) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("connectome-query-flipping-witness-audit: invalid JSON input: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match evaluate(&input, &input_bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("connectome-query-flipping-witness-audit: {error}");
            return ExitCode::from(2);
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(json) => json + "\n",
        Err(error) => {
            eprintln!(
                "connectome-query-flipping-witness-audit: could not serialize report: {error}"
            );
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(path, rendered.as_bytes()) {
            eprintln!("connectome-query-flipping-witness-audit: could not write report: {error}");
            return ExitCode::from(2);
        }
    } else {
        print!("{rendered}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Input {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_QUERY_FLIPPING_WITNESS.example.json"
        ))
        .expect("checked-in witness fixture must parse")
    }

    fn fixture_bytes() -> Vec<u8> {
        include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_QUERY_FLIPPING_WITNESS.example.json"
        )
        .as_bytes()
        .to_vec()
    }

    #[test]
    fn exact_search_finds_minimum_two_edit_synergy_with_evidence_gate() {
        let input = fixture();
        let report = evaluate(&input, &fixture_bytes()).unwrap();
        assert!((report.baseline_query_value - 1.8).abs() < 1.0e-12);
        assert!(report.baseline_decision_met);
        assert_eq!(report.minimal_witness_effort_units, Some(5));
        assert_eq!(report.minimal_witness_count, 5);
        assert_eq!(report.status, "COUNTEREXAMPLE_FOUND");
        assert_eq!(report.subset_count_evaluated, 127);
        assert_eq!(report.minimal_witnesses_truncated, false);
        assert!(report.witnesses.iter().all(|witness| {
            witness.effort_units == 5
                && !witness.resulting_decision_met
                && witness.edit_ids.contains(&"adjudication-gate".to_string())
        }));
        assert!(report.witnesses.iter().all(|witness| {
            !(witness.edit_ids.contains(&"remove-13-a".to_string())
                && witness.edit_ids.contains(&"remove-13-b".to_string()))
        }));
    }

    #[test]
    fn three_edit_synergy_is_found_when_no_singleton_or_pair_flips() {
        let baseline_graph = vec![
            Edge {
                source: 1,
                target: 2,
                weight: 0.5,
            },
            Edge {
                source: 1,
                target: 3,
                weight: 0.5,
            },
            Edge {
                source: 1,
                target: 4,
                weight: 0.5,
            },
            Edge {
                source: 1,
                target: 5,
                weight: 0.5,
            },
        ];
        let graph = graph_from_edges(&baseline_graph).unwrap();
        let targets = [2_u64, 3, 4, 5];
        let edits = targets
            .iter()
            .enumerate()
            .map(|(index, target)| EditAtom {
                edit_id: format!("remove-{target}"),
                kind: EditKind::GraphEdit,
                effort_units: 1,
                evidence_sha256: format!("{:064x}", index + 1),
                requires: Vec::new(),
                conflicts_with: Vec::new(),
                changes: vec![EdgeChange {
                    source: 1,
                    target: *target,
                    expected_weight: 0.5,
                    replacement_weight: 0.0,
                }],
            })
            .collect::<Vec<_>>();
        let input = Input {
            schema_version: 1,
            analysis_id: "three-way-synergy-unit-test".into(),
            dataset_revision_commit_sha: "a".repeat(40),
            edit_universe_scope: "exhaustive_declared_edit_universe".into(),
            baseline_graph_sha256: graph_sha256(&graph),
            baseline_graph,
            query: Query {
                query_id: "out-degree-at-least-0-75".into(),
                metric: QueryMetric::OutDegree { node: 1 },
                decision: DecisionRule::GreaterOrEqual { threshold: 0.75 },
            },
            effort_unit: "synthetic_effort_units".into(),
            max_total_effort_units: 3,
            edits,
        };
        let baseline = validate(&input).unwrap();
        for mask in 1_u64..(1_u64 << input.edits.len()) {
            if mask.count_ones() <= 2 {
                let graph = apply_subset(&baseline, mask, &input.edits);
                assert!(decision_met(
                    query_value(&graph, &input.query.metric),
                    &input.query.decision
                ));
            }
        }
        let bytes = serde_json::to_vec(&input).unwrap();
        let report = evaluate(&input, &bytes).unwrap();
        assert_eq!(report.minimal_witness_effort_units, Some(3));
        assert_eq!(report.minimal_witness_count, 4);
        assert_eq!(report.decision_flipping_subset_count_within_budget, 5);
        assert!(
            report
                .witnesses
                .iter()
                .all(|witness| { witness.edit_ids.len() == 3 && !witness.resulting_decision_met })
        );
    }

    #[test]
    fn no_witness_is_reported_when_budget_is_below_the_minimum() {
        let mut input = fixture();
        input.max_total_effort_units = 4;
        let bytes = serde_json::to_vec(&input).unwrap();
        let report = evaluate(&input, &bytes).unwrap();
        assert_eq!(report.minimal_witness_effort_units, None);
        assert_eq!(report.minimal_witness_count, 0);
        assert_eq!(
            report.status,
            "NO_WITNESS_WITHIN_DECLARED_EDIT_UNIVERSE_AND_BUDGET"
        );
        assert!(report.search_complete);
        assert_eq!(
            report.probability_claim,
            "NONE_DETERMINISTIC_FINITE_EDIT_UNIVERSE_ONLY"
        );
    }

    #[test]
    fn every_single_graph_edit_fails_to_flip_the_query() {
        let input = fixture();
        let baseline = validate(&input).unwrap();
        let baseline_value = query_value(&baseline, &input.query.metric);
        for edit in &input.edits {
            if edit.kind != EditKind::GraphEdit {
                continue;
            }
            let index = input
                .edits
                .iter()
                .position(|candidate| candidate.edit_id == edit.edit_id)
                .unwrap();
            let graph = apply_subset(&baseline, 1_u64 << index, &input.edits);
            assert_eq!(
                decision_met(
                    query_value(&graph, &input.query.metric),
                    &input.query.decision
                ),
                decision_met(baseline_value, &input.query.decision),
                "single edit {} unexpectedly flips the query",
                edit.edit_id
            );
        }
    }

    #[test]
    fn same_edge_alternatives_are_never_combined() {
        let input = fixture();
        let report = evaluate(&input, &fixture_bytes()).unwrap();
        for witness in report.witnesses {
            assert!(
                !(witness.edit_ids.contains(&"remove-13-a".to_string())
                    && witness.edit_ids.contains(&"remove-13-b".to_string()))
            );
        }
    }

    #[test]
    fn stale_edge_preconditions_fail_closed() {
        let mut input = fixture();
        let edit = input
            .edits
            .iter_mut()
            .find(|edit| edit.edit_id == "remove-12")
            .unwrap();
        edit.changes[0].expected_weight = 0.5;
        assert!(
            validate(&input)
                .unwrap_err()
                .contains("precondition mismatch")
        );
    }

    #[test]
    fn dependency_cycles_fail_closed() {
        let mut input = fixture();
        let gate = input
            .edits
            .iter_mut()
            .find(|edit| edit.edit_id == "adjudication-gate")
            .unwrap();
        gate.requires.push("remove-12".into());
        assert!(validate(&input).unwrap_err().contains("cycle"));
    }

    #[test]
    fn conflict_declarations_must_be_symmetric() {
        let mut input = fixture();
        let edit = input
            .edits
            .iter_mut()
            .find(|edit| edit.edit_id == "remove-13-b")
            .unwrap();
        edit.conflicts_with.clear();
        assert!(validate(&input).unwrap_err().contains("symmetrically"));
    }

    #[test]
    fn same_edge_mutations_require_explicit_conflicts() {
        let mut input = fixture();
        let edit = input
            .edits
            .iter_mut()
            .find(|edit| edit.edit_id == "replace-12-small")
            .unwrap();
        edit.conflicts_with.clear();
        assert!(validate(&input).unwrap_err().contains("symmetrically"));
    }

    #[test]
    fn edit_universe_digest_is_independent_of_edit_and_relation_order() {
        let mut first = fixture();
        let mut second = fixture();
        let before = canonical_edit_universe_sha256(&first.edits).unwrap();
        second.edits.reverse();
        for edit in &mut second.edits {
            edit.requires.reverse();
            edit.conflicts_with.reverse();
            edit.changes.reverse();
        }
        let after = canonical_edit_universe_sha256(&second.edits).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn graph_digest_mismatch_is_rejected() {
        let mut input = fixture();
        input.baseline_graph_sha256 = "0".repeat(64);
        assert!(validate(&input).unwrap_err().contains("digest mismatch"));
    }

    #[test]
    fn exact_enumeration_limit_is_enforced() {
        let mut input = fixture();
        while input.edits.len() <= MAX_EDIT_ATOMS {
            let mut decoy = input.edits.last().unwrap().clone();
            decoy.edit_id = format!("extra-{}", input.edits.len());
            decoy.effort_units = 1;
            decoy.requires.clear();
            decoy.conflicts_with.clear();
            decoy.evidence_sha256 = "f".repeat(64);
            decoy.changes[0].source = 100 + input.edits.len() as u64;
            decoy.changes[0].target = 200 + input.edits.len() as u64;
            decoy.changes[0].expected_weight = 0.0;
            decoy.changes[0].replacement_weight = 0.1;
            input.edits.push(decoy);
        }
        assert!(validate(&input).unwrap_err().contains("exact enumeration"));
    }
}
