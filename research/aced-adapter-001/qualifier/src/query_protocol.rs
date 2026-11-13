use serde_json::Value;
use std::collections::BTreeSet;

fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|item| !item.trim().is_empty())
}

fn exact_fields(value: &Value, expected: &[&str]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.len() == expected.len() && expected.iter().all(|key| object.contains_key(*key))
}

fn string_array(value: &Value) -> Option<BTreeSet<String>> {
    value.as_array().map(|rows| {
        rows.iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()
    })
}

fn closed_schema_object(schema: &Value, required: &[&str]) -> bool {
    let properties = schema["properties"].as_object();
    let declared = schema["required"].as_array().and_then(|rows| {
        rows.iter()
            .map(Value::as_str)
            .collect::<Option<Vec<_>>>()
    });
    schema["type"] == "object"
        && schema["additionalProperties"] == false
        && properties.is_some_and(|object| {
            object.len() == required.len() && required.iter().all(|key| object.contains_key(*key))
        })
        && declared.is_some_and(|rows| {
            rows.len() == required.len()
                && rows.iter().copied().collect::<BTreeSet<_>>()
                    == required.iter().copied().collect::<BTreeSet<_>>()
        })
}

fn schema_has_expected_contract(schema: &Value) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let top = [
        "protocol_id",
        "version",
        "status",
        "upstream",
        "source_artifacts",
        "track_to_question_projection",
        "official_resource_regimes",
        "agent_arms",
        "published_comparators",
        "evaluator_reproduction",
        "primary_metrics",
        "statistical_design",
        "proposed_targets",
        "security_boundary",
        "claim_ceiling",
    ];
    if text(schema, "$schema") != Some("https://json-schema.org/draft/2020-12/schema")
        || text(schema, "$id")
            != Some("https://github.com/Luminous-Dynamics/symthaea/blob/main/research/aced-adapter-001/ACED_QUERY_COMPARISON_PROTOCOL_V0_1.schema.json")
        || !closed_schema_object(schema, &top)
    {
        errors.push("Query comparison schema top-level identity or closed contract mismatch".into());
    }

    let closed_objects = [
        (
            "upstream",
            &[
                "repository",
                "commit",
                "paper",
                "paper_revision",
                "paper_revision_date",
                "license",
            ][..],
        ),
        (
            "source_artifacts",
            &[
                "metadata_manifest",
                "metadata_inventory",
                "metadata_inventory_sha256",
                "upstream_sha256sums_sha256",
                "metadata_only",
            ][..],
        ),
        (
            "track_to_question_projection",
            &[
                "world_inventory_counts",
                "active_agent_questions",
                "guess_only_questions",
                "derivation",
                "interpretation",
            ][..],
        ),
        (
            "published_comparators",
            &[
                "best_reported_active_agent",
                "question_targeted_active_learning",
                "sample_budget_ablation",
            ][..],
        ),
        (
            "evaluator_reproduction",
            &[
                "preserve_official_structural_and_decision_evaluators",
                "request_parser_model",
                "parser_identity_source",
                "pin_parser_model_prompt_temperature_output_limit_and_revision",
                "preserve_candidate_raw_response",
                "preserve_extracted_structured_answer",
                "preserve_official_tolerance_and_exact_match_rules",
                "report_parse_failures_separately",
                "diagnostic_parser_arm",
                "independent_adjudication",
            ][..],
        ),
        (
            "primary_metrics",
            &["core", "qual", "trajectory", "measurement_validity"][..],
        ),
        (
            "statistical_design",
            &[
                "unit",
                "pairing",
                "overlap_control",
                "parser_control",
                "model_control",
                "cost_control",
                "confidence_intervals",
                "multiplicity",
            ][..],
        ),
        (
            "proposed_targets",
            &[
                "baseline_reproduction",
                "primary_stretch",
                "budget_normalized",
                "safety",
            ][..],
        ),
    ];
    for (name, fields) in closed_objects {
        if !closed_schema_object(&schema["properties"][name], fields) {
            errors.push(format!(
                "Query comparison schema nested object {name} must be closed over its exact declared properties"
            ));
        }
    }

    let track_projection = &schema["properties"]["track_to_question_projection"]["properties"];
    for (name, fields) in [
        (
            "world_inventory_counts",
            &["core_struct", "core_decision", "guess_struct", "guess_decision"][..],
        ),
        (
            "active_agent_questions",
            &[
                "core_list_queries",
                "core_decision_tasks",
                "core_total",
                "qual_questions",
            ][..],
        ),
        (
            "guess_only_questions",
            &[
                "core_list_queries",
                "core_decision_tasks",
                "core_total",
                "qual_questions",
            ][..],
        ),
        (
            "derivation",
            &["core_struct", "core_decision", "guess_struct", "guess_decision"][..],
        ),
    ] {
        if !closed_schema_object(&track_projection[name], fields) {
            errors.push(format!(
                "Query comparison schema projection object {name} is not closed or has an incorrect field set"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

pub fn qualify_query_comparison_protocol(
    protocol: &Value,
    schema: &Value,
    source_manifest: &Value,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let top = [
        "protocol_id",
        "version",
        "status",
        "upstream",
        "source_artifacts",
        "track_to_question_projection",
        "official_resource_regimes",
        "agent_arms",
        "published_comparators",
        "evaluator_reproduction",
        "primary_metrics",
        "statistical_design",
        "proposed_targets",
        "security_boundary",
        "claim_ceiling",
    ];
    if !exact_fields(protocol, &top)
        || text(protocol, "protocol_id") != Some("ACED-QUERY-COMPARISON-001")
        || text(protocol, "version") != Some("0.1.0")
        || text(protocol, "status") != Some("preregistered_comparison_design_not_model_results")
    {
        errors.push("ACED-Query comparison protocol identity or top-level closed contract mismatch".into());
    }

    let upstream = &protocol["upstream"];
    if text(upstream, "repository") != text(&source_manifest["upstream"], "repository")
        || text(upstream, "commit") != text(&source_manifest["upstream"], "commit")
        || text(upstream, "license") != Some("MIT")
        || text(upstream, "paper") != text(&source_manifest["upstream"], "paper")
        || text(upstream, "paper_revision") != text(&source_manifest["upstream"], "paper_revision")
        || text(upstream, "paper_revision_date")
            != text(&source_manifest["upstream"], "paper_revision_date")
    {
        errors.push("ACED-Query comparison protocol and pinned metadata manifest disagree on upstream source identity".into());
    }

    let artifacts = &protocol["source_artifacts"];
    if text(artifacts, "metadata_manifest") != Some("ACED_QUERY_SOURCE_MANIFEST_V0_1.json")
        || text(artifacts, "metadata_inventory")
            != Some("upstream/aced_query_world_inventory.json")
        || text(artifacts, "metadata_inventory_sha256")
            != text(&source_manifest["artifacts"], "inventory_sha256")
        || text(artifacts, "upstream_sha256sums_sha256")
            != text(&source_manifest["artifacts"], "upstream_checksums_sha256")
        || artifacts["metadata_only"] != true
    {
        errors.push("ACED-Query comparison protocol does not bind the exact metadata manifest/inventory/checksum artifacts".into());
    }

    let projection = &protocol["track_to_question_projection"];
    for (track, count) in [
        ("core_struct", 10_u64),
        ("core_decision", 60),
        ("guess_struct", 30),
        ("guess_decision", 48),
    ] {
        if projection["world_inventory_counts"][track].as_u64() != Some(count)
            || source_manifest["track_counts"][track].as_u64() != Some(count)
        {
            errors.push(format!("ACED-Query world count mismatch for {track}"));
        }
    }
    for (key, expected) in [
        ("core_list_queries", 10_u64),
        ("core_decision_tasks", 60),
        ("core_total", 70),
        ("qual_questions", 50),
    ] {
        if projection["active_agent_questions"][key].as_u64() != Some(expected) {
            errors.push(format!(
                "ACED-Query active task projection mismatch for {key}"
            ));
        }
    }
    for (key, expected) in [
        ("core_list_queries", 30_u64),
        ("core_decision_tasks", 48),
        ("core_total", 78),
        ("qual_questions", 150),
    ] {
        if projection["guess_only_questions"][key].as_u64() != Some(expected) {
            errors.push(format!(
                "ACED-Query guess-only task projection mismatch for {key}"
            ));
        }
    }

    let regimes = protocol["official_resource_regimes"].as_array();
    match regimes {
        Some(regimes) if regimes.len() == 2 => {
            let four_turn = regimes.iter().find(|regime| text(regime, "id") == Some("four_turn"));
            let sample = regimes
                .iter()
                .find(|regime| text(regime, "id") == Some("sample_budget"));
            if four_turn.is_none_or(|regime| {
                regime["max_outer_turns"].as_u64() != Some(4)
                    || !regime["arms"].as_array().is_some_and(|arms| {
                        ["active", "active_code", "modular", "guess_only"]
                            .iter()
                            .all(|arm| arms.iter().any(|value| value.as_str() == Some(arm)))
                    })
            }) {
                errors.push("ACED-Query official four-turn regime must preserve four turns and four official agent arms".into());
            }
            if sample.is_none_or(|regime| {
                !regime["max_outer_turns"].is_null()
                    || regime["max_sampled_rows_per_task"].as_u64() != Some(5000)
            }) {
                errors.push("ACED-Query Samp5k regime must remain a separate 5,000-row, non-turn-limited arm".into());
            }
        }
        _ => errors.push("ACED-Query comparison must declare exactly two distinct resource regimes".into()),
    }

    let arm_ids = protocol["agent_arms"]
        .as_array()
        .map(|arms| {
            arms.iter()
                .filter_map(|arm| text(arm, "id").map(str::to_owned))
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let expected_arms = [
        "active",
        "active_code",
        "modular",
        "guess_only",
        "question_targeted_active_learning",
        "symthaea_evidence_gated",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    if arm_ids != expected_arms {
        errors.push("ACED-Query comparison arm set must include all three official agents, Guess-only, classical Q-TAL and Symthaea".into());
    }

    let comparators = &protocol["published_comparators"];
    if comparators["best_reported_active_agent"]["core_accuracy"].as_f64() != Some(0.94)
        || comparators["best_reported_active_agent"]["qual_accuracy"].as_f64() != Some(0.98)
        || comparators["question_targeted_active_learning"]["core_accuracy"].as_f64()
            != Some(0.96)
        || comparators["question_targeted_active_learning"]["qual_accuracy"].as_f64()
            != Some(1.0)
        || !text(&comparators["best_reported_active_agent"], "source")
            .is_some_and(|source| source == "https://arxiv.org/abs/2610.14381")
        || !text(&comparators["question_targeted_active_learning"], "source")
            .is_some_and(|source| source == "https://arxiv.org/abs/2610.14381")
    {
        errors.push("ACED-Query published comparator identities or baseline values do not match the pinned paper contract".into());
    }

    let evaluator = &protocol["evaluator_reproduction"];
    if text(evaluator, "request_parser_model") != Some("us.anthropic.claude-opus-4-6-v1")
        || evaluator["preserve_official_structural_and_decision_evaluators"] != true
        || evaluator["preserve_candidate_raw_response"] != true
        || evaluator["preserve_extracted_structured_answer"] != true
        || evaluator["preserve_official_tolerance_and_exact_match_rules"] != true
        || evaluator["report_parse_failures_separately"] != true
    {
        errors.push("ACED-Query evaluator reproduction contract must preserve the pinned parser and official scoring".into());
    }

    let hidden_boundary = [
        "Hidden PGM",
        "CPDs",
        "graph edges",
        "generator seed",
        "oracle answer",
        "scoring roles",
        "evaluator state",
    ];
    let security = protocol["security_boundary"].as_array();
    if security.is_none_or(|rows| {
        rows.len() < 4
            || hidden_boundary
                .iter()
                .any(|term| !rows.iter().any(|row| row.as_str().is_some_and(|value| value.contains(term))))
    }) {
        errors.push("ACED-Query candidate security boundary must explicitly protect hidden PGM, CPDs, graph, seed, oracle, scoring roles and evaluator state".into());
    }

    if let Err(found) = schema_has_expected_contract(schema) {
        errors.extend(found);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROTOCOL: &str = include_str!("../../ACED_QUERY_COMPARISON_PROTOCOL_V0_1.json");
    const SCHEMA: &str = include_str!("../../ACED_QUERY_COMPARISON_PROTOCOL_V0_1.schema.json");
    const SOURCE_MANIFEST: &str = include_str!("../../ACED_QUERY_SOURCE_MANIFEST_V0_1.json");

    fn parsed(source: &str) -> Value {
        serde_json::from_str(source).expect("checked-in comparison artifact must parse")
    }

    #[test]
    fn checked_in_query_comparison_protocol_and_closed_schema_qualify() {
        qualify_query_comparison_protocol(
            &parsed(PROTOCOL),
            &parsed(SCHEMA),
            &parsed(SOURCE_MANIFEST),
        )
        .expect("pinned query comparison protocol qualifies");
    }

    #[test]
    fn active_question_denominator_drift_fails_closed() {
        let mut protocol = parsed(PROTOCOL);
        protocol["track_to_question_projection"]["active_agent_questions"]["core_total"] =
            Value::from(71);
        assert!(qualify_query_comparison_protocol(
            &protocol,
            &parsed(SCHEMA),
            &parsed(SOURCE_MANIFEST),
        )
        .is_err());
    }

    #[test]
    fn guess_only_question_denominator_drift_fails_closed() {
        let mut protocol = parsed(PROTOCOL);
        protocol["track_to_question_projection"]["guess_only_questions"]["qual_questions"] =
            Value::from(149);
        assert!(qualify_query_comparison_protocol(
            &protocol,
            &parsed(SCHEMA),
            &parsed(SOURCE_MANIFEST),
        )
        .is_err());
    }

    #[test]
    fn parser_identity_drift_fails_closed() {
        let mut protocol = parsed(PROTOCOL);
        protocol["evaluator_reproduction"]["request_parser_model"] =
            Value::String("some-other-parser".into());
        assert!(qualify_query_comparison_protocol(
            &protocol,
            &parsed(SCHEMA),
            &parsed(SOURCE_MANIFEST),
        )
        .is_err());
    }

    #[test]
    fn published_baseline_drift_fails_closed() {
        let mut protocol = parsed(PROTOCOL);
        protocol["published_comparators"]["question_targeted_active_learning"]["core_accuracy"] =
            Value::from(0.91);
        assert!(qualify_query_comparison_protocol(
            &protocol,
            &parsed(SCHEMA),
            &parsed(SOURCE_MANIFEST),
        )
        .is_err());
    }

    #[test]
    fn open_nested_schema_object_fails_closed() {
        let mut schema = parsed(SCHEMA);
        schema["properties"]["statistical_design"]["additionalProperties"] = Value::Bool(true);
        assert!(schema_has_expected_contract(&schema).is_err());
    }

    #[test]
    fn mismatched_inventory_source_digest_fails_closed() {
        let mut protocol = parsed(PROTOCOL);
        protocol["source_artifacts"]["metadata_inventory_sha256"] =
            Value::String("f".repeat(64));
        assert!(qualify_query_comparison_protocol(
            &protocol,
            &parsed(SCHEMA),
            &parsed(SOURCE_MANIFEST),
        )
        .is_err());
    }
}
