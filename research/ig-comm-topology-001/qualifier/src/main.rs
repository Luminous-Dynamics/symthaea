use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env, fs,
    process::ExitCode,
};

const CONTRACT_ID: &str = "IG-COMM-TOPOLOGY-001";
const EXPECTED_PROFILES: usize = 8;
const EXPECTED_FIXTURES: usize = 30;

#[derive(Debug, PartialEq, Eq)]
struct Report {
    profiles: usize,
    fixtures: usize,
    topology_families: usize,
}

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|s| !s.trim().is_empty())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn add(errors: &mut Vec<String>, context: &str, message: impl AsRef<str>) {
    errors.push(format!("{context}: {}", message.as_ref()));
}

fn strings(value: &Value) -> Vec<String> {
    value.as_array().into_iter().flatten()
        .filter_map(Value::as_str).map(ToOwned::to_owned).collect()
}

fn string_set(value: &Value) -> BTreeSet<String> {
    strings(value).into_iter().collect()
}

fn validate_schema_header(schema: &Value, expected_id: &str, errors: &mut Vec<String>) {
    if schema["$schema"].as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
        add(errors, "schema", "unexpected JSON Schema dialect");
    }
    if schema["$id"].as_str() != Some(expected_id) {
        add(errors, "schema", "unexpected schema identifier");
    }
    if schema["type"].as_str() != Some("object")
        || schema["additionalProperties"].as_bool() != Some(false)
    {
        add(errors, "schema", "top-level schema must be a closed object");
    }
    let required = string_set(&schema["required"]);
    for name in [
        "schema_version", "contract_id", "status", "profile_count",
        "fixture_count", "profiles", "fixtures", "claim_ceiling",
    ] {
        if !required.contains(name) {
            add(errors, "schema", format!("required key {name} is missing"));
        }
    }
}

fn graph_payload(profile: &Value) -> Value {
    json!({
        "profile_id": profile["profile_id"],
        "profile_version": profile["profile_version"],
        "topology_family": profile["topology_family"],
        "node_identity": profile["node_identity"],
        "edge_semantics": profile["edge_semantics"],
        "rng_algorithm": profile["rng_algorithm"],
        "seed": profile["seed"],
        "nodes": profile["nodes"],
        "edges": profile["edges"],
        "graph_state": profile["graph_state"],
        "snapshot_round": profile["snapshot_round"],
        "temporal_events": profile["temporal_events"],
        "self_loops": profile["self_loops"],
        "extra": profile["extra"]
    })
}

fn graph_sha256(profile: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(&graph_payload(profile))
        .map_err(|error| format!("canonical graph serialization failed: {error}"))?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{digest:x}"))
}

fn edge_adjacency(profile: &Value) -> Result<Vec<Vec<usize>>, String> {
    let nodes = profile["nodes"].as_array()
        .ok_or_else(|| "nodes must be an array".to_string())?;
    let mut indexes = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        let id = field(node, "agent_instance_id")
            .ok_or_else(|| "node AgentInstanceId must be non-empty".to_string())?;
        if indexes.insert(id.to_string(), index).is_some() {
            return Err(format!("duplicate AgentInstanceId {id}"));
        }
    }

    let edges = profile["edges"].as_array()
        .ok_or_else(|| "edges must be an array".to_string())?;
    let mut adjacency = vec![Vec::new(); nodes.len()];
    for edge in edges {
        let from = field(edge, "sender")
            .ok_or_else(|| "edge sender must be non-empty".to_string())?;
        let to = field(edge, "recipient")
            .ok_or_else(|| "edge recipient must be non-empty".to_string())?;
        let from_index = *indexes.get(from)
            .ok_or_else(|| format!("unknown sender AgentInstanceId {from}"))?;
        let to_index = *indexes.get(to)
            .ok_or_else(|| format!("unknown recipient AgentInstanceId {to}"))?;
        if from_index == to_index {
            return Err("self-loop is not permitted".to_string());
        }
        adjacency[from_index].push(to_index);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    Ok(adjacency)
}

fn is_strongly_connected(adjacency: &[Vec<usize>]) -> bool {
    let n = adjacency.len();
    if n == 0 {
        return false;
    }
    for start in 0..n {
        let mut visited = vec![false; n];
        let mut queue = VecDeque::from([start]);
        visited[start] = true;
        while let Some(node) = queue.pop_front() {
            for next in &adjacency[node] {
                if !visited[*next] {
                    visited[*next] = true;
                    queue.push_back(*next);
                }
            }
        }
        if visited.iter().any(|seen| !seen) {
            return false;
        }
    }
    true
}

fn normalized_betweenness(adjacency: &[Vec<usize>]) -> Vec<f64> {
    let n = adjacency.len();
    let mut centrality = vec![0.0_f64; n];

    for source in 0..n {
        let mut stack = Vec::with_capacity(n);
        let mut predecessors = vec![Vec::<usize>::new(); n];
        let mut sigma = vec![0.0_f64; n];
        let mut distance = vec![usize::MAX; n];
        let mut queue = VecDeque::new();

        sigma[source] = 1.0;
        distance[source] = 0;
        queue.push_back(source);

        while let Some(vertex) = queue.pop_front() {
            stack.push(vertex);
            for next in &adjacency[vertex] {
                if distance[*next] == usize::MAX {
                    distance[*next] = distance[vertex] + 1;
                    queue.push_back(*next);
                }
                if distance[*next] == distance[vertex] + 1 {
                    sigma[*next] += sigma[vertex];
                    predecessors[*next].push(vertex);
                }
            }
        }

        let mut dependency = vec![0.0_f64; n];
        while let Some(vertex) = stack.pop() {
            for predecessor in &predecessors[vertex] {
                if sigma[vertex] > 0.0 {
                    dependency[*predecessor] +=
                        (sigma[*predecessor] / sigma[vertex]) * (1.0 + dependency[vertex]);
                }
            }
            if vertex != source {
                centrality[vertex] += dependency[vertex];
            }
        }
    }

    let denominator = ((n.saturating_sub(1)) * (n.saturating_sub(2))) as f64;
    if denominator > 0.0 {
        for value in &mut centrality {
            *value /= denominator;
        }
    }
    centrality
}

fn centrality_ties(
    ids: &[String],
    values: &[f64],
) -> (f64, f64, Vec<String>, Vec<String>) {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut min_ids = Vec::new();
    let mut max_ids = Vec::new();
    for (id, value) in ids.iter().zip(values.iter()) {
        if (value - minimum).abs() < 1e-8 {
            min_ids.push(id.clone());
        }
        if (value - maximum).abs() < 1e-8 {
            max_ids.push(id.clone());
        }
    }
    min_ids.sort();
    max_ids.sort();
    (minimum, maximum, min_ids, max_ids)
}

fn validate_centrality(profile: &Value, errors: &mut Vec<String>) {
    let context = field(profile, "profile_id").unwrap_or("profile");
    let Some(nodes) = profile["nodes"].as_array() else {
        add(errors, context, "centrality requires a node array");
        return;
    };
    let ids: Vec<String> = nodes.iter()
        .filter_map(|node| field(node, "agent_instance_id").map(ToOwned::to_owned))
        .collect();
    let adjacency = match edge_adjacency(profile) {
        Ok(value) => value,
        Err(error) => {
            add(errors, context, error);
            return;
        }
    };

    let reference = &profile["centrality_reference"];
    match field(reference, "status") {
        Some("Pass") => {
            if !is_strongly_connected(&adjacency) {
                add(errors, context, "centrality PASS requires a strongly connected graph");
                return;
            }
            if field(reference, "algorithm_id") != Some("normalized_betweenness_v1")
                || field(reference, "direction") != Some("directed")
                || field(reference, "weights") != Some("unweighted")
                || field(reference, "distance") != Some("hop_count")
            {
                add(errors, context, "centrality reference algorithm profile mismatch");
            }

            let actual = normalized_betweenness(&adjacency);
            let declared = reference["values"].as_array().cloned().unwrap_or_default();
            if declared.len() != ids.len() {
                add(errors, context, "centrality value count differs from node count");
                return;
            }
            for (index, item) in declared.iter().enumerate() {
                let declared_id = field(item, "agent_instance_id");
                let declared_score = item["score"].as_f64();
                if declared_id != Some(ids[index].as_str())
                    || declared_score.is_none_or(|score| (score - actual[index]).abs() > 1e-8)
                {
                    add(errors, context, "centrality score or node order differs from recomputation");
                    break;
                }
            }

            let (minimum, maximum, min_ids, max_ids) = centrality_ties(&ids, &actual);
            if (reference["minimum_score"].as_f64().unwrap_or(f64::NAN) - minimum).abs() > 1e-8
                || (reference["maximum_score"].as_f64().unwrap_or(f64::NAN) - maximum).abs() > 1e-8
                || strings(&reference["minimum_tie_set"]) != min_ids
                || strings(&reference["maximum_tie_set"]) != max_ids
            {
                add(errors, context, "centrality extrema or tie sets differ from recomputation");
            }
            if field(reference, "minimum_selected_agent_instance_id")
                != min_ids.first().map(String::as_str)
                || field(reference, "maximum_selected_agent_instance_id")
                    != max_ids.first().map(String::as_str)
            {
                add(errors, context, "centrality selections do not follow lexical tie-break");
            }
            if min_ids == max_ids {
                add(errors, context, "centrality PASS cannot use identical min/max tie sets");
            }
        }
        Some("Unsupported") => {
            let min_ids = strings(&reference["minimum_tie_set"]);
            let max_ids = strings(&reference["maximum_tie_set"]);
            let all_nodes_tied = min_ids.len() == ids.len()
                && max_ids.len() == ids.len()
                && min_ids == max_ids;
            if is_strongly_connected(&adjacency) && !all_nodes_tied {
                add(errors, context, "Unsupported centrality requires disconnection or no distinct extrema");
            }
        }
        _ => add(errors, context, "centrality status must be Pass or Unsupported"),
    }
}

fn validate_profile(profile: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let context = field(profile, "profile_id").unwrap_or("<missing profile_id>");
    if !context.starts_with("IG-COMM-TOPOLOGY-001/")
        || field(profile, "profile_version") != Some("0.1")
        || field(profile, "node_identity") != Some("AgentInstanceId")
        || field(profile, "edge_semantics") != Some("directed_message_channel")
        || profile["self_loops"].as_bool() != Some(false)
    {
        add(&mut errors, context, "CommunicationGraph type/version invariants failed");
    }

    let mut node_ids = BTreeSet::new();
    for node in profile["nodes"].as_array().into_iter().flatten() {
        let id = field(node, "agent_instance_id").unwrap_or_default();
        if id.is_empty() || !node_ids.insert(id.to_string()) {
            add(&mut errors, context, "duplicate or empty AgentInstanceId");
        }
    }

    let mut channel_ids = BTreeSet::new();
    for edge in profile["edges"].as_array().into_iter().flatten() {
        let id = field(edge, "channel_id").unwrap_or_default();
        if id.is_empty() || !channel_ids.insert(id.to_string()) {
            add(&mut errors, context, "duplicate or empty channel_id");
        }
        if !node_ids.contains(field(edge, "sender").unwrap_or_default())
            || !node_ids.contains(field(edge, "recipient").unwrap_or_default())
        {
            add(&mut errors, context, "unknown AgentInstanceId endpoint");
        }
        if field(edge, "sender") == field(edge, "recipient") {
            add(&mut errors, context, "self_loop_not_permitted");
        }
        if edge["capacity_per_round"].as_u64().unwrap_or(0) == 0 {
            add(&mut errors, context, "channel capacity must be positive");
        }
    }
    if let Err(error) = edge_adjacency(profile) {
        add(&mut errors, context, error);
    }

    match graph_sha256(profile) {
        Ok(actual) if field(profile, "graph_sha256") == Some(actual.as_str()) => {}
        Ok(_) => add(&mut errors, context, "graph_sha256 does not match canonical graph payload"),
        Err(error) => add(&mut errors, context, error),
    }

    let events = profile["temporal_events"].as_array().cloned().unwrap_or_default();
    let mut previous_round = None;
    let mut event_ids = BTreeSet::new();
    for event in &events {
        let id = field(event, "event_id").unwrap_or_default();
        if id.is_empty() || !event_ids.insert(id.to_string()) {
            add(&mut errors, context, "temporal event IDs must be unique and non-empty");
        }
        let round = event["round"].as_u64();
        if round.is_none() || previous_round.is_some_and(|prev| round.unwrap_or(0) <= prev) {
            add(&mut errors, context, "temporal_event_order_invalid");
        }
        previous_round = round;
        if field(event, "kind") == Some("restore_channel") {
            let old_ids = string_set(&event["channel_ids"]);
            let new_ids = string_set(&event["new_channel_ids"]);
            if new_ids.is_empty() || !new_ids.is_disjoint(&old_ids) {
                add(&mut errors, context, "restore must use new channel identities");
            }
        }
    }

    if matches!(profile["graph_state"].as_str(), Some("static_snapshot" | "event_sourced")) {
        validate_centrality(profile, &mut errors);
    } else {
        add(&mut errors, context, "unknown graph_state");
    }

    errors.sort();
    errors.dedup();
    errors
}

fn validate_manifest(manifest: &Value, schema: &Value) -> Result<Report, Vec<String>> {
    let mut errors = Vec::new();
    validate_schema_header(
        schema,
        "https://github.com/Luminous-Dynamics/symthaea/blob/main/research/ig-comm-topology-001/COMMUNICATION_GRAPH_FIXTURES_V0_1.schema.json",
        &mut errors,
    );
    if field(manifest, "schema_version") != Some("communication-graph-fixtures-v0.1")
        || field(manifest, "contract_id") != Some(CONTRACT_ID)
        || field(manifest, "status") != Some("synthetic_graph_profiles_unexecuted")
    {
        add(&mut errors, "manifest", "contract identity or claim status mismatch");
    }
    if manifest["owner_issue"].as_str()
        != Some("https://github.com/Luminous-Dynamics/symthaea/issues/11337")
        || manifest["delegation_graph_issue"].as_str()
            != Some("https://github.com/Luminous-Dynamics/symthaea/issues/11248")
    {
        add(&mut errors, "manifest", "communication and delegation graph owners must remain distinct");
    }

    let Some(profiles) = manifest["profiles"].as_array() else {
        add(&mut errors, "manifest", "profiles must be an array");
        return Err(errors);
    };
    if profiles.len() != EXPECTED_PROFILES
        || manifest["profile_count"].as_u64() != Some(profiles.len() as u64)
        || manifest["topology_count"].as_u64() != Some(EXPECTED_PROFILES as u64)
    {
        add(&mut errors, "manifest", "expected eight topology profiles and matching counts");
    }

    let mut profile_ids = BTreeSet::new();
    let mut topologies = BTreeSet::new();
    for (index, profile) in profiles.iter().enumerate() {
        let id = field(profile, "profile_id").unwrap_or_default();
        if id.is_empty() || !profile_ids.insert(id.to_string()) {
            add(&mut errors, "profiles", "profile IDs must be unique and non-empty");
        }
        if let Some(family) = field(profile, "topology_family") {
            topologies.insert(family.to_string());
        }
        for error in validate_profile(profile) {
            add(&mut errors, &format!("profiles[{index}]"), error);
        }
    }
    if topologies.len() != EXPECTED_PROFILES {
        add(&mut errors, "profiles", "all eight topology families must be represented");
    }

    let Some(fixtures) = manifest["fixtures"].as_array() else {
        add(&mut errors, "manifest", "fixtures must be an array");
        return Err(errors);
    };
    if fixtures.len() != EXPECTED_FIXTURES
        || manifest["fixture_count"].as_u64() != Some(fixtures.len() as u64)
    {
        add(&mut errors, "manifest", "expected exactly 30 fixture descriptors");
    }

    let mut fixture_ids = BTreeSet::new();
    let mut centrality_families = BTreeSet::new();
    for (index, fixture) in fixtures.iter().enumerate() {
        let id = field(fixture, "fixture_id").unwrap_or_default();
        if id.is_empty() || !fixture_ids.insert(id.to_string()) {
            add(&mut errors, "fixtures", "fixture IDs must be unique and non-empty");
        }
        let profile_ref = field(fixture, "profile_ref").unwrap_or_default();
        if !profile_ids.contains(profile_ref) {
            add(&mut errors, &format!("fixtures[{index}]"), "profile_ref does not resolve");
        }
        if field(fixture, "kind") == Some("centrality_known_answer") {
            if let Some(profile) = profiles.iter().find(|p| field(p, "profile_id") == Some(profile_ref)) {
                if field(&profile["centrality_reference"], "status") != Some("Pass") {
                    add(&mut errors, &format!("fixtures[{index}]"), "centrality fixture must reference supported profile");
                }
                if let Some(family) = field(profile, "topology_family") {
                    centrality_families.insert(family.to_string());
                }
            }
        }
        if !matches!(field(fixture, "expected_status"), Some("Pass" | "Fail" | "Unsupported" | "Indeterminate")) {
            add(&mut errors, &format!("fixtures[{index}]"), "invalid expected_status");
        }
    }
    if centrality_families.len() != 5 {
        add(&mut errors, "fixtures", "five distinct centrality topology families are required");
    }

    if errors.is_empty() {
        Ok(Report {
            profiles: profiles.len(),
            fixtures: fixtures.len(),
            topology_families: topologies.len(),
        })
    } else {
        Err(errors)
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let content = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&content).map_err(|error| format!("{path}: invalid JSON: {error}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: ig-comm-topology-qualifier <fixtures.json> <fixtures.schema.json>");
        return ExitCode::from(2);
    }
    let manifest = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let schema = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };

    match validate_manifest(&manifest, &schema) {
        Ok(report) => {
            println!(
                "QUALIFIED COMMUNICATION GRAPH FIXTURES: profiles={} topologies={} fixtures={}",
                report.profiles, report.topology_families, report.fixtures
            );
            println!("NOT EXECUTED: channel reachability is not message delivery; no agent interaction was run.");
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!("COMMUNICATION GRAPH QUALIFICATION FAILED ({} findings):", errors.len());
            for error in errors {
                eprintln!(" - {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str =
        include_str!("../../COMMUNICATION_GRAPH_FIXTURES_V0_1.json");
    const SCHEMA: &str =
        include_str!("../../COMMUNICATION_GRAPH_FIXTURES_V0_1.schema.json");

    fn parsed(text: &str) -> Value {
        serde_json::from_str(text).expect("checked-in JSON must parse")
    }

    fn profile(family: &str) -> Value {
        parsed(MANIFEST)["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .find(|profile| profile["topology_family"] == family)
            .unwrap()
            .clone()
    }

    #[test]
    fn eight_profiles_and_thirty_fixture_descriptors_validate() {
        let report = validate_manifest(&parsed(MANIFEST), &parsed(SCHEMA))
            .expect("checked-in communication graph corpus must validate");
        assert_eq!(report.profiles, 8);
        assert_eq!(report.topology_families, 8);
        assert_eq!(report.fixtures, 30);
    }

    #[test]
    fn chain_centrality_reference_recomputes() {
        let profile = profile("chain");
        let adjacency = edge_adjacency(&profile).unwrap();
        let values = normalized_betweenness(&adjacency);
        assert!((values[3] - 0.57142857).abs() < 1e-8);
        assert_eq!(
            field(&profile["centrality_reference"], "minimum_selected_agent_instance_id"),
            Some("AGENT-00")
        );
        assert_eq!(
            field(&profile["centrality_reference"], "maximum_selected_agent_instance_id"),
            Some("AGENT-03")
        );
    }

    #[test]
    fn star_center_is_maximum_betweenness() {
        let profile = profile("star");
        let adjacency = edge_adjacency(&profile).unwrap();
        let values = normalized_betweenness(&adjacency);
        assert!((values[0] - 1.0).abs() < 1e-8);
        assert!(values[1..].iter().all(|value| value.abs() < 1e-8));
    }

    #[test]
    fn complete_graph_has_no_distinguished_centrality_node() {
        let profile = profile("complete");
        let adjacency = edge_adjacency(&profile).unwrap();
        let values = normalized_betweenness(&adjacency);
        assert!(values.iter().all(|value| value.abs() < 1e-8));
        assert_eq!(field(&profile["centrality_reference"], "status"), Some("Unsupported"));
    }

    #[test]
    fn sparse_random_profile_retains_all_node_tie_set() {
        let profile = profile("sparse_random");
        let adjacency = edge_adjacency(&profile).unwrap();
        let values = normalized_betweenness(&adjacency);
        let ids: Vec<String> = profile["nodes"].as_array().unwrap()
            .iter()
            .map(|node| field(node, "agent_instance_id").unwrap().to_string())
            .collect();
        let (_, _, min_ids, max_ids) = centrality_ties(&ids, &values);
        assert_eq!(min_ids.len(), 8);
        assert_eq!(max_ids.len(), 8);
    }

    #[test]
    fn reciprocal_channels_are_explicit_directed_edges() {
        let mut profile = profile("star");
        let edges = profile["edges"].as_array_mut().unwrap();
        edges.retain(|edge| !(edge["sender"] == "AGENT-01" && edge["recipient"] == "AGENT-00"));
        let adjacency = edge_adjacency(&profile).unwrap();
        assert_eq!(adjacency[1].len(), 0);
        assert!(adjacency[0].contains(&1));
    }

    #[test]
    fn self_loop_is_rejected() {
        let mut profile = profile("chain");
        profile["edges"].as_array_mut().unwrap().push(json!({
            "channel_id":"SELF-LOOP",
            "sender":"AGENT-00",
            "recipient":"AGENT-00"
        }));
        assert!(edge_adjacency(&profile).is_err());
    }

    #[test]
    fn duplicate_channel_id_is_rejected() {
        let mut profile = profile("chain");
        let edge = profile["edges"][0].clone();
        profile["edges"].as_array_mut().unwrap().push(edge);
        let errors = validate_profile(&profile);
        assert!(errors.iter().any(|error| error.contains("duplicate or empty channel_id")));
    }

    #[test]
    fn unknown_agent_endpoint_is_rejected() {
        let mut profile = profile("chain");
        profile["edges"][0]["recipient"] = Value::String("AGENT-UNKNOWN".into());
        assert!(edge_adjacency(&profile).is_err());
    }

    #[test]
    fn principal_multiplicity_does_not_merge_agent_instances() {
        let profile = profile("star");
        let ids: BTreeSet<String> = profile["nodes"].as_array().unwrap()
            .iter()
            .map(|node| field(node, "agent_instance_id").unwrap().to_string())
            .collect();
        assert_eq!(ids.len(), 8);
        assert_eq!(profile["nodes"][3]["principal_ref"], profile["nodes"][7]["principal_ref"]);
    }

    #[test]
    fn graph_identity_changes_when_seed_changes() {
        let mut profile = profile("chain");
        let before = graph_sha256(&profile).unwrap();
        profile["seed"] = Value::from(9999_u64);
        let after = graph_sha256(&profile).unwrap();
        assert_ne!(before, after);
    }

    #[test]
    fn out_of_order_temporal_event_is_rejected() {
        let mut profile = profile("temporal_churn");
        profile["temporal_events"][1]["round"] = Value::from(1_u64);
        let errors = validate_profile(&profile);
        assert!(errors.iter().any(|error| error.contains("temporal_event_order_invalid")));
    }

    #[test]
    fn restore_uses_new_channel_identity() {
        let mut profile = profile("temporal_churn");
        let old_ids = profile["temporal_events"][1]["channel_ids"].clone();
        profile["temporal_events"][1]["new_channel_ids"] = old_ids;
        let errors = validate_profile(&profile);
        assert!(errors.iter().any(|error| error.contains("restore must use new channel identities")));
    }

    #[test]
    fn graph_hash_detects_edge_tampering() {
        let mut profile = profile("chain");
        let before = graph_sha256(&profile).unwrap();
        profile["edges"][0]["capacity_per_round"] = Value::from(999_u64);
        let after = graph_sha256(&profile).unwrap();
        assert_ne!(before, after);
    }

    #[test]
    fn disconnected_graph_is_not_assigned_global_centrality() {
        let mut profile = profile("stochastic_block");
        profile["edges"].as_array_mut().unwrap().retain(|edge| {
            !((edge["sender"] == "AGENT-03" && edge["recipient"] == "AGENT-04")
                || (edge["sender"] == "AGENT-04" && edge["recipient"] == "AGENT-03"))
        });
        let adjacency = edge_adjacency(&profile).unwrap();
        assert!(!is_strongly_connected(&adjacency));
    }
}
