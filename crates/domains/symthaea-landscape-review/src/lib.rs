// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict structural/policy checks for evidence-bound landscape review artifacts.
//!
//! This contract crate does not generate a review, run a physical model, or
//! prove environmental/social impact. The JSON Schema remains the exchange
//! contract; these Rust checks enforce a small set of safety-critical
//! cross-field invariants and deterministic CLI validation.

use serde_json::Value;
use std::collections::HashSet;

pub const REVIEW_SCHEMA: &str = "symthaea.landscape-alternative-review.v0";
pub const REVIEW_SCHEMA_URI: &str =
    "https://luminousdynamics.io/schemas/landscape-alternative-review-v0.schema.json";
pub const SOURCE_CATALOG_STATUS_METADATA_ONLY: &str = "metadata-only-not-ingested";
pub const LOCAL_DRAWING_FRAME: &str = "LOCAL_CARTESIAN_DRAWING_UNITS";

const VALID_SCOPES: &[&str] = &["site-parcel", "district", "watershed", "bioregion"];
const VALID_GEOMETRY_STATUS: &[&str] = &["schematic-design-intent", "synthetic-benchmark"];
const VALID_METRIC_STATUS: &[&str] = &["computed", "blocked", "conflicted", "not-applicable"];
const VALID_RANKING_STATUS: &[&str] = &[
    "no-defensible-ranking",
    "partial-comparison",
    "qualified-comparison",
];
const VALID_RECOMMENDATION_STATUS: &[&str] =
    &["no-ranking", "candidate-next-action", "prefer-alternative"];

/// Validate high-consequence structural and evidence-policy invariants.
/// Call a full JSON Schema validator as a separate gate before accepting input.
pub fn validate_review(review: &Value) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if !review.is_object() {
        return Err(vec!["$: expected a JSON object".to_string()]);
    }

    require_const(review, "schema", REVIEW_SCHEMA, "$", &mut errors);
    require_const(review, "schema_uri", REVIEW_SCHEMA_URI, "$", &mut errors);
    for field in ["review_id", "created_at"] {
        require_nonblank(review, field, "$", &mut errors);
    }
    require_const(
        review,
        "analysis_profile",
        "LAND-REVIEW-001",
        "$",
        &mut errors,
    );

    let authority = object_field(review, "authority", "$", &mut errors);
    if let Some(authority) = authority {
        require_bool(authority, "read_only", true, "$.authority", &mut errors);
        require_bool(
            authority,
            "mutates_project",
            false,
            "$.authority",
            &mut errors,
        );
        require_bool(
            authority,
            "actuation_authority",
            false,
            "$.authority",
            &mut errors,
        );
    }

    let inputs = object_field(review, "inputs", "$", &mut errors);
    let mut baseline_geometry_status = "";
    let mut source_catalog_status = "";
    if let Some(inputs) = inputs {
        let baseline = object_field(inputs, "baseline_design", "$.inputs", &mut errors);
        if let Some(baseline) = baseline {
            require_const(
                baseline,
                "system",
                "sol-atlas",
                "$.inputs.baseline_design",
                &mut errors,
            );
            require_const(
                baseline,
                "schema",
                "luminous.landscape-design.v0",
                "$.inputs.baseline_design",
                &mut errors,
            );
            require_const(
                baseline,
                "schema_uri",
                "https://luminousdynamics.io/schemas/landscape-design-v0.schema.json",
                "$.inputs.baseline_design",
                &mut errors,
            );
            require_sha256(baseline, "sha256", "$.inputs.baseline_design", &mut errors);
            let scope = string_value(baseline.get("scope"));
            if !VALID_SCOPES.contains(&scope) {
                errors.push("$.inputs.baseline_design.scope: unsupported scope".to_string());
            }
            if string_value(baseline.get("coordinate_frame_kind")) != LOCAL_DRAWING_FRAME {
                errors.push(format!(
                    "$.inputs.baseline_design.coordinate_frame_kind: expected {LOCAL_DRAWING_FRAME}"
                ));
            }
            require_bool(
                baseline,
                "georeferenced",
                false,
                "$.inputs.baseline_design",
                &mut errors,
            );
            baseline_geometry_status = string_value(baseline.get("geometry_status"));
            if !VALID_GEOMETRY_STATUS.contains(&baseline_geometry_status) {
                errors.push(
                    "$.inputs.baseline_design.geometry_status: unsupported v0 geometry status"
                        .to_string(),
                );
            }
        }
        require_sha256(inputs, "metric_registry_sha256", "$.inputs", &mut errors);
        require_sha256(inputs, "source_catalog_sha256", "$.inputs", &mut errors);
        source_catalog_status = string_value(inputs.get("source_catalog_status"));
        if !matches!(
            source_catalog_status,
            "metadata-only-not-ingested"
                | "partially-ingested-unqualified"
                | "ingested-unqualified"
                | "qualified"
        ) {
            errors.push("$.inputs.source_catalog_status: unsupported catalog status".to_string());
        }
        validate_scenario_runs(inputs.get("scenario_runs"), &mut errors);
    }

    let alternatives = validate_alternatives(
        review.get("alternatives"),
        baseline_geometry_status,
        &mut errors,
    );
    if let Some(inputs) = inputs {
        if let Some(baseline) = inputs.get("baseline_design") {
            let baseline_sha = string_value(baseline.get("sha256"));
            let alternatives_contain_baseline = review
                .get("alternatives")
                .and_then(Value::as_array)
                .is_some_and(|items| {
                    items.iter().any(|alternative| {
                        string_value(alternative.get("snapshot_sha256")) == baseline_sha
                    })
                });
            if !baseline_sha.is_empty() && !alternatives_contain_baseline {
                errors.push(
                    "$.alternatives: at least one alternative must bind the exact baseline design snapshot"
                        .to_string(),
                );
            }
        }
    }
    let metric_assessments = review.get("metric_assessments").and_then(Value::as_array);
    if let Some(metrics) = metric_assessments {
        for (index, metric) in metrics.iter().enumerate() {
            validate_metric(
                metric,
                index,
                baseline_geometry_status,
                source_catalog_status,
                &mut errors,
            );
        }
    } else {
        errors.push("$.metric_assessments: expected an array".to_string());
    }

    validate_candidate_sources(review.get("candidate_sources"), &mut errors);
    let comparison = object_field(review, "comparison", "$", &mut errors);
    let mut ranking_status = "";
    if let Some(comparison) = comparison {
        ranking_status = string_value(comparison.get("ranking_status"));
        if !VALID_RANKING_STATUS.contains(&ranking_status) {
            errors.push("$.comparison.ranking_status: unsupported ranking status".to_string());
        }
        if ranking_status == "no-defensible-ranking"
            && !nonempty_string_array(comparison.get("ranking_reasons"))
        {
            errors.push(
                "$.comparison.ranking_reasons: no-ranking result requires explicit reasons"
                    .to_string(),
            );
        }
        validate_pairwise_tradeoffs(
            comparison.get("pairwise_tradeoffs"),
            &alternatives,
            &mut errors,
        );
    }

    let scenario_runs_value = inputs.and_then(|value| value.get("scenario_runs"));
    let baseline_scope = inputs
        .and_then(|value| value.get("baseline_design"))
        .map(|value| string_value(value.get("scope")))
        .unwrap_or("");
    let expected_metric_registry_sha256 = inputs
        .map(|value| string_value(value.get("metric_registry_sha256")))
        .unwrap_or("");
    validate_scenario_pairing(
        scenario_runs_value,
        review.get("alternatives"),
        baseline_scope,
        expected_metric_registry_sha256,
        ranking_status,
        &mut errors,
    );

    let recommendation = object_field(review, "recommendation", "$", &mut errors);
    if let Some(recommendation) = recommendation {
        let recommendation_status = string_value(recommendation.get("status"));
        if !VALID_RECOMMENDATION_STATUS.contains(&recommendation_status) {
            errors.push("$.recommendation.status: unsupported recommendation status".to_string());
        }
        require_nonblank(recommendation, "rationale", "$.recommendation", &mut errors);
        require_nonblank(recommendation, "next_step", "$.recommendation", &mut errors);

        if ranking_status == "no-defensible-ranking" && recommendation_status != "no-ranking" {
            errors.push(
                "$.recommendation.status: no-defensible-ranking requires no-ranking recommendation"
                    .to_string(),
            );
        }
        if recommendation_status == "prefer-alternative" {
            if ranking_status != "qualified-comparison" {
                errors.push(
                    "$.recommendation.status: prefer-alternative requires qualified-comparison"
                        .to_string(),
                );
            }
            let selected = string_value(recommendation.get("selected_alternative_id"));
            if selected.is_empty() || !alternatives.contains(selected) {
                errors.push(
                    "$.recommendation.selected_alternative_id: selected alternative must exist in alternatives"
                        .to_string(),
                );
            }
            if !nonempty_string_array(recommendation.get("tradeoffs")) {
                errors.push(
                    "$.recommendation.tradeoffs: preferred alternative requires trade-offs"
                        .to_string(),
                );
            }
            if !matches!(
                string_value(recommendation.get("confidence")),
                "high" | "medium" | "low" | "not-calibrated"
            ) {
                errors.push("$.recommendation.confidence: confidence must be declared".to_string());
            }
        } else if recommendation.get("selected_alternative_id").is_some() {
            errors.push(
                "$.recommendation.selected_alternative_id: only prefer-alternative may select a winner"
                    .to_string(),
            );
        }
    }

    if baseline_geometry_status == "synthetic-benchmark"
        && metric_assessments.is_some_and(|metrics| {
            metrics
                .iter()
                .any(|metric| string_value(metric.get("status")) == "computed")
        })
    {
        errors.push(
            "$.metric_assessments: synthetic benchmark geometry cannot qualify environmental/social outcome metrics"
                .to_string(),
        );
    }
    if metric_assessments.is_some_and(|metrics| {
        metrics
            .iter()
            .any(|metric| string_value(metric.get("status")) == "computed")
    }) {
        errors.push(
            "$.metric_assessments: the v0 LOCAL_CARTESIAN_DRAWING_UNITS frame cannot support physical or spatial outcome values"
                .to_string(),
        );
    }

    if !nonempty_string_array(review.get("falsification_questions")) {
        errors.push(
            "$.falsification_questions: at least one falsification question is required"
                .to_string(),
        );
    }
    if !nonempty_string_array(review.get("limitations")) {
        errors.push("$.limitations: at least one limitation is required".to_string());
    }
    validate_replay(review.get("replay"), &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_scenario_runs(value: Option<&Value>, errors: &mut Vec<String>) {
    let Some(runs) = value.and_then(Value::as_array) else {
        errors.push("$.inputs.scenario_runs: expected an array".to_string());
        return;
    };
    if runs.len() > 100 {
        errors.push("$.inputs.scenario_runs: exceeds v0 limit of 100".to_string());
    }
    for (index, run) in runs.iter().enumerate() {
        let path = format!("$.inputs.scenario_runs[{index}]");
        require_const(run, "system", "symtropy", &path, errors);
        require_sha256(run, "manifest_sha256", &path, errors);
        let purpose = string_value(run.get("purpose"));
        let evidence_class = string_value(run.get("evidence_class"));
        let status = string_value(run.get("status"));
        if !matches!(
            purpose,
            "schema-qualification" | "synthetic-benchmark" | "analytical-simulation"
        ) {
            errors.push(format!("{path}.purpose: unsupported purpose"));
        }
        if !matches!(
            evidence_class,
            "observed" | "curated" | "synthetic" | "modelled" | "assumed" | "missing"
        ) {
            errors.push(format!("{path}.evidence_class: unsupported evidence class"));
        }
        if !matches!(
            status,
            "qualified" | "blocked" | "unsupported" | "conflicted"
        ) {
            errors.push(format!("{path}.status: unsupported status"));
        }
        require_nonblank(run, "model_id", &path, errors);
        require_nonblank(run, "model_version", &path, errors);
        let context_path = format!("{path}.pairing_context");
        let context = object_field(run, "pairing_context", &path, errors);
        if let Some(context) = context {
            require_nonblank(context, "alternative_id", &context_path, errors);
            for field in [
                "design_snapshot_sha256",
                "boundary_conditions_sha256",
                "evaluation_plan_sha256",
                "metric_registry_sha256",
            ] {
                require_sha256(context, field, &context_path, errors);
            }
            if context.get("seed").and_then(Value::as_u64).is_none() {
                errors.push(format!(
                    "{context_path}.seed: expected a non-negative integer"
                ));
            }
            let scope = string_value(context.get("scope"));
            if !VALID_SCOPES.contains(&scope) {
                errors.push(format!("{context_path}.scope: unsupported scope"));
            }
        }
        if status == "qualified" {
            require_sha256(run, "result_sha256", &path, errors);
        }
        if purpose == "synthetic-benchmark" && evidence_class != "synthetic" {
            errors.push(format!(
                "{path}.evidence_class: synthetic benchmark must remain synthetic"
            ));
        }
        if purpose == "analytical-simulation" && status == "qualified" {
            errors.push(format!(
                "{path}.status: analytical simulation cannot be qualified for the current schematic frame"
            ));
        }
    }
}

fn validate_scenario_pairing(
    value: Option<&Value>,
    alternatives_value: Option<&Value>,
    baseline_scope: &str,
    expected_metric_registry_sha256: &str,
    ranking_status: &str,
    errors: &mut Vec<String>,
) {
    let Some(runs) = value.and_then(Value::as_array) else {
        return;
    };
    let Some(alternatives) = alternatives_value.and_then(Value::as_array) else {
        if !runs.is_empty() {
            errors.push(
                "$.alternatives: paired scenarios require alternative definitions".to_string(),
            );
        }
        return;
    };

    let mut seen_alternatives = HashSet::new();
    let mut design_digests = HashSet::new();
    let mut manifest_digests = HashSet::new();
    let mut common_context: Option<(String, String, String, u64, String, String, String)> = None;

    for (index, run) in runs.iter().enumerate() {
        let path = format!("$.inputs.scenario_runs[{index}]");
        let manifest_sha = string_value(run.get("manifest_sha256"));
        if !manifest_digests.insert(manifest_sha.to_string()) {
            errors.push(format!(
                "{path}.manifest_sha256: duplicate scenario manifest"
            ));
        }
        let Some(context) = run.get("pairing_context").filter(|value| value.is_object()) else {
            continue;
        };
        let alternative_id = string_value(context.get("alternative_id"));
        if alternative_id.is_empty() || !seen_alternatives.insert(alternative_id.to_string()) {
            errors.push(format!(
                "{path}.pairing_context.alternative_id: missing or duplicate alternative ID"
            ));
        }

        let design_sha = string_value(context.get("design_snapshot_sha256"));
        let boundary_sha = string_value(context.get("boundary_conditions_sha256"));
        let evaluation_sha = string_value(context.get("evaluation_plan_sha256"));
        let registry_sha = string_value(context.get("metric_registry_sha256"));
        let scope = string_value(context.get("scope"));
        let seed = context
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or(u64::MAX);
        let model_id = string_value(run.get("model_id"));
        let model_version = string_value(run.get("model_version"));

        design_digests.insert(design_sha.to_string());
        if registry_sha != expected_metric_registry_sha256 {
            errors.push(format!("{path}.pairing_context.metric_registry_sha256: must match the review's pinned metric registry"));
        }
        if scope != baseline_scope {
            errors.push(format!(
                "{path}.pairing_context.scope: must match the baseline design scope"
            ));
        }

        match alternatives
            .iter()
            .find(|alternative| string_value(alternative.get("alternative_id")) == alternative_id)
        {
            Some(alternative) => {
                if string_value(alternative.get("snapshot_sha256")) != design_sha {
                    errors.push(format!("{path}.pairing_context.design_snapshot_sha256: does not match the exact alternative snapshot"));
                }
                if string_value(alternative.get("scope")) != scope {
                    errors.push(format!("{path}.pairing_context.scope: does not match the referenced alternative scope"));
                }
            }
            None => errors.push(format!(
                "{path}.pairing_context.alternative_id: unknown alternative"
            )),
        }

        let context_tuple = (
            boundary_sha.to_string(),
            evaluation_sha.to_string(),
            registry_sha.to_string(),
            seed,
            scope.to_string(),
            model_id.to_string(),
            model_version.to_string(),
        );
        if let Some(reference) = &common_context {
            if reference != &context_tuple {
                errors.push(format!(
                    "{path}.pairing_context: model/version, boundary conditions, evaluation plan, metric registry, seed and scope must match across paired alternatives"
                ));
            }
        } else {
            common_context = Some(context_tuple);
        }
    }

    if runs.len() >= 2 && design_digests.len() != runs.len() {
        errors.push(
            "$.inputs.scenario_runs: paired alternatives must use distinct exact design snapshots"
                .to_string(),
        );
    }
    if runs.len() >= 2 && manifest_digests.len() != runs.len() {
        errors.push(
            "$.inputs.scenario_runs: each paired scenario must have a distinct manifest digest"
                .to_string(),
        );
    }

    if ranking_status == "qualified-comparison" {
        let expected_ids: HashSet<&str> = alternatives
            .iter()
            .map(|alternative| string_value(alternative.get("alternative_id")))
            .collect();
        if runs.len() != alternatives.len()
            || seen_alternatives.len() != alternatives.len()
            || !expected_ids
                .iter()
                .all(|id| seen_alternatives.contains(*id))
        {
            errors.push(
                "$.inputs.scenario_runs: qualified comparison requires one qualified run per exact alternative"
                    .to_string(),
            );
        }
        if runs
            .iter()
            .any(|run| string_value(run.get("status")) != "qualified")
        {
            errors.push(
                "$.inputs.scenario_runs: qualified comparison cannot include blocked or conflicted runs"
                    .to_string(),
            );
        }
    }
}

fn validate_alternatives(
    value: Option<&Value>,
    baseline_geometry_status: &str,
    errors: &mut Vec<String>,
) -> HashSet<String> {
    let Some(alternatives) = value.and_then(Value::as_array) else {
        errors.push("$.alternatives: expected an array".to_string());
        return HashSet::new();
    };
    if !(2..=100).contains(&alternatives.len()) {
        errors.push("$.alternatives: v0 requires 2..=100 alternatives".to_string());
    }
    let mut ids = HashSet::with_capacity(alternatives.len());
    for (index, alternative) in alternatives.iter().enumerate() {
        let path = format!("$.alternatives[{index}]");
        let id = string_value(alternative.get("alternative_id"));
        if id.is_empty() || !ids.insert(id.to_string()) {
            errors.push(format!(
                "{path}.alternative_id: missing or duplicate alternative ID"
            ));
        }
        require_sha256(alternative, "snapshot_sha256", &path, errors);
        require_nonblank(alternative, "change_summary", &path, errors);
        let scope = string_value(alternative.get("scope"));
        if !VALID_SCOPES.contains(&scope) {
            errors.push(format!("{path}.scope: unsupported scope"));
        }
        if string_value(alternative.get("coordinate_frame_kind")) != LOCAL_DRAWING_FRAME {
            errors.push(format!("{path}.coordinate_frame_kind: unsupported frame"));
        }
        let geometry_status = string_value(alternative.get("geometry_status"));
        if !VALID_GEOMETRY_STATUS.contains(&geometry_status) {
            errors.push(format!(
                "{path}.geometry_status: unsupported geometry status"
            ));
        }
        if baseline_geometry_status == "synthetic-benchmark"
            && geometry_status != "synthetic-benchmark"
        {
            errors.push(format!(
                "{path}.geometry_status: cannot promote alternatives from a synthetic benchmark baseline"
            ));
        }
    }
    ids
}

fn validate_candidate_sources(value: Option<&Value>, errors: &mut Vec<String>) {
    let Some(sources) = value.and_then(Value::as_array) else {
        errors.push("$.candidate_sources: expected an array".to_string());
        return;
    };
    if sources.len() > 200 {
        errors.push("$.candidate_sources: exceeds v0 limit of 200".to_string());
    }
    let mut ids = HashSet::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        let path = format!("$.candidate_sources[{index}]");
        let id = string_value(source.get("source_id"));
        if id.is_empty() || !ids.insert(id.to_string()) {
            errors.push(format!("{path}.source_id: missing or duplicate source ID"));
        }
        for field in ["role", "limitation"] {
            require_nonblank(source, field, &path, errors);
        }
        let state = string_value(source.get("catalog_state"));
        if !matches!(
            state,
            "not-ingested"
                | "ingested-unqualified"
                | "qualified"
                | "unavailable"
                | "license-review-required"
        ) {
            errors.push(format!("{path}.catalog_state: unsupported catalog state"));
        }
        if state == "not-ingested" && source.get("source_sha256").is_some() {
            errors.push(format!("{path}.source_sha256: a not-ingested source cannot declare an ingested source digest"));
        }
    }
}

fn validate_metric(
    metric: &Value,
    index: usize,
    baseline_geometry_status: &str,
    source_catalog_status: &str,
    errors: &mut Vec<String>,
) {
    let path = format!("$.metric_assessments[{index}]");
    for field in ["metric_id", "family"] {
        require_nonblank(metric, field, &path, errors);
    }
    let status = string_value(metric.get("status"));
    if !VALID_METRIC_STATUS.contains(&status) {
        errors.push(format!("{path}.status: unsupported metric status"));
        return;
    }
    if status == "computed" {
        if baseline_geometry_status == "synthetic-benchmark" {
            errors.push(format!(
                "{path}: synthetic baseline cannot produce a computed impact metric"
            ));
        }
        if source_catalog_status == SOURCE_CATALOG_STATUS_METADATA_ONLY {
            errors.push(format!(
                "{path}: metadata-only source catalog cannot support computed metric results"
            ));
        }
        if !metric.get("value").is_some_and(Value::is_number) {
            errors.push(format!(
                "{path}.value: computed metric requires a numeric value"
            ));
        }
        for field in ["unit", "evidence_class", "method_ref"] {
            require_nonblank(metric, field, &path, errors);
        }
        if !metric
            .get("input_sha256")
            .and_then(Value::as_array)
            .is_some_and(|values| {
                !values.is_empty()
                    && values
                        .iter()
                        .all(|value| value.as_str().is_some_and(is_sha256))
            })
        {
            errors.push(format!(
                "{path}.input_sha256: computed result requires 64-hex input digests"
            ));
        }
        let uncertainty = metric.get("uncertainty");
        if !uncertainty.is_some_and(Value::is_object)
            || !uncertainty
                .and_then(|item| item.get("description"))
                .and_then(Value::as_str)
                .is_some_and(|text| !text.trim().is_empty())
        {
            errors.push(format!(
                "{path}.uncertainty: computed metric requires an uncertainty declaration"
            ));
        }
        if metric.get("reason").is_some()
            || metric.get("missing_inputs").is_some()
            || metric.get("conflicting_evidence_refs").is_some()
        {
            errors.push(format!(
                "{path}: computed status cannot carry blocked/conflict fields"
            ));
        }
    } else {
        if metric.get("value").is_some() {
            errors.push(format!(
                "{path}.value: blocked/conflicted/not-applicable metric must not carry a value"
            ));
        }
        match status {
            "blocked" => {
                require_nonblank(metric, "reason", &path, errors);
                if !nonempty_string_array(metric.get("missing_inputs")) {
                    errors.push(format!(
                        "{path}.missing_inputs: blocked metric requires missing inputs"
                    ));
                }
            }
            "conflicted" => {
                require_nonblank(metric, "reason", &path, errors);
                if !metric
                    .get("conflicting_evidence_refs")
                    .and_then(Value::as_array)
                    .is_some_and(|items| items.len() >= 2)
                {
                    errors.push(format!("{path}.conflicting_evidence_refs: conflicted metric requires at least two evidence refs"));
                }
            }
            "not-applicable" => require_nonblank(metric, "reason", &path, errors),
            _ => {}
        }
    }
}

fn validate_pairwise_tradeoffs(
    value: Option<&Value>,
    alternatives: &HashSet<String>,
    errors: &mut Vec<String>,
) {
    let Some(tradeoffs) = value.and_then(Value::as_array) else {
        errors.push("$.comparison.pairwise_tradeoffs: expected an array".to_string());
        return;
    };
    if tradeoffs.len() > 500 {
        errors.push("$.comparison.pairwise_tradeoffs: exceeds v0 limit of 500".to_string());
    }
    for (index, tradeoff) in tradeoffs.iter().enumerate() {
        let path = format!("$.comparison.pairwise_tradeoffs[{index}]");
        let left = string_value(tradeoff.get("left_alternative_id"));
        let right = string_value(tradeoff.get("right_alternative_id"));
        if !alternatives.contains(left) || !alternatives.contains(right) || left == right {
            errors.push(format!(
                "{path}: both referenced alternatives must exist and differ"
            ));
        }
        for field in ["dimension", "summary"] {
            require_nonblank(tradeoff, field, &path, errors);
        }
    }
}

fn validate_replay(value: Option<&Value>, errors: &mut Vec<String>) {
    let Some(replay) = value.filter(|item| item.is_object()) else {
        errors.push("$.replay: expected an object".to_string());
        return;
    };
    for field in ["analysis_method_id", "analysis_method_version"] {
        require_nonblank(replay, field, "$.replay", errors);
    }
    require_sha256(replay, "ruleset_sha256", "$.replay", errors);
    require_sha256(replay, "normalized_inputs_sha256", "$.replay", errors);
    let replay_class = string_value(replay.get("replay_class"));
    if !matches!(
        replay_class,
        "deterministic-structural-review" | "deterministic-model-replay" | "not-established"
    ) {
        errors.push("$.replay.replay_class: unsupported replay classification".to_string());
    }
}

fn object_field<'a>(
    object: &'a Value,
    key: &str,
    path: &str,
    errors: &mut Vec<String>,
) -> Option<&'a Value> {
    match object.get(key).filter(|value| value.is_object()) {
        Some(value) => Some(value),
        None => {
            errors.push(format!("{path}.{key}: expected an object"));
            None
        }
    }
}

fn require_const(object: &Value, key: &str, expected: &str, path: &str, errors: &mut Vec<String>) {
    if object.get(key).and_then(Value::as_str) != Some(expected) {
        errors.push(format!("{path}.{key}: expected {expected}"));
    }
}

fn require_bool(object: &Value, key: &str, expected: bool, path: &str, errors: &mut Vec<String>) {
    if object.get(key).and_then(Value::as_bool) != Some(expected) {
        errors.push(format!("{path}.{key}: expected {expected}"));
    }
}

fn require_nonblank(object: &Value, key: &str, path: &str, errors: &mut Vec<String>) {
    if !object
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        errors.push(format!("{path}.{key}: required non-empty string"));
    }
}

fn require_sha256(object: &Value, key: &str, path: &str, errors: &mut Vec<String>) -> String {
    let value = object.get(key).and_then(Value::as_str).unwrap_or("");
    if !is_sha256(value) {
        errors.push(format!(
            "{path}.{key}: expected 64 lowercase hexadecimal characters"
        ));
    }
    value.to_string()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn string_value(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or("")
}

fn nonempty_string_array(value: Option<&Value>) -> bool {
    value.and_then(Value::as_array).is_some_and(|values| {
        !values.is_empty()
            && values
                .iter()
                .all(|item| item.as_str().is_some_and(|text| !text.trim().is_empty()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../fixtures/landscape/review-unranked-v0.json"
        ))
        .expect("canonical review fixture must be valid JSON")
    }

    #[test]
    fn canonical_unranked_fixture_is_valid() {
        assert!(validate_review(&fixture()).is_ok());
    }

    #[test]
    fn a_no_defensible_ranking_never_selects_a_winner() {
        let mut review = fixture();
        review["recommendation"]["status"] = json!("prefer-alternative");
        review["recommendation"]["selected_alternative_id"] = json!("baseline-canopy-water");
        review["recommendation"]["tradeoffs"] = json!(["tradeoff"]);
        review["recommendation"]["confidence"] = json!("low");
        let errors = validate_review(&review).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("requires no-ranking recommendation"))
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("requires qualified-comparison"))
        );
    }

    #[test]
    fn blocked_metric_cannot_smuggle_a_numeric_value() {
        let mut review = fixture();
        review["metric_assessments"][0]["value"] = json!(0.0);
        let errors = validate_review(&review).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("must not carry a value"))
        );
    }

    #[test]
    fn metadata_only_sources_block_computed_outcomes() {
        let mut review = fixture();
        review["metric_assessments"][0] = json!({
            "metric_id":"heat.shade_fraction_at_declared_time",
            "family":"heat-and-comfort",
            "status":"computed",
            "value":23.0,
            "unit":"percent_of_area",
            "evidence_class":"modelled",
            "method_ref":"shadow-model-v0",
            "input_sha256":["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
            "uncertainty":{"kind":"qualitative","description":"Uncertainty not yet quantified."}
        });
        review["comparison"]["ranking_status"] = json!("partial-comparison");
        review["comparison"]["ranking_reasons"] = json!(["Some evidence is still missing."]);
        review["recommendation"]["status"] = json!("candidate-next-action");
        let errors = validate_review(&review).unwrap_err();
        assert!(errors.iter().any(|error| {
            error.contains("metadata-only source catalog cannot support computed metric")
        }));
    }

    #[test]
    fn local_drawing_frame_cannot_support_computed_metric_values() {
        let mut review = fixture();
        review["inputs"]["source_catalog_status"] = json!("qualified");
        review["comparison"]["ranking_status"] = json!("qualified-comparison");
        review["comparison"]["ranking_reasons"] =
            json!(["All declared comparison requirements were structurally qualified."]);
        review["metric_assessments"][0] = json!({
            "metric_id":"heat.shade_fraction_at_declared_time",
            "family":"heat-and-comfort",
            "status":"computed",
            "value":23.0,
            "unit":"percent_of_area",
            "evidence_class":"modelled",
            "method_ref":"shadow-model-v1",
            "input_sha256":["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
            "uncertainty":{"kind":"qualitative","description":"Scenario range requires calibration."}
        });
        let errors = validate_review(&review).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("LOCAL_CARTESIAN_DRAWING_UNITS frame cannot support"))
        );
    }

    #[test]
    fn analytical_scenario_cannot_be_claimed_qualified_for_schematic_v0() {
        let mut review = fixture();
        review["inputs"]["scenario_runs"] = json!([{
            "system":"symtropy",
            "manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "result_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "purpose":"analytical-simulation",
            "evidence_class":"modelled",
            "status":"qualified",
            "model_id":"runoff-model",
            "model_version":"1.0"
        }]);
        let errors = validate_review(&review).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("analytical simulation cannot be qualified"))
        );
    }

    #[test]
    fn synthetic_benchmark_geometry_cannot_produce_computed_metrics() {
        let mut review = fixture();
        review["inputs"]["baseline_design"]["geometry_status"] = json!("synthetic-benchmark");
        review["alternatives"][0]["geometry_status"] = json!("synthetic-benchmark");
        review["alternatives"][1]["geometry_status"] = json!("synthetic-benchmark");
        review["metric_assessments"][0]["status"] = json!("computed");
        review["metric_assessments"][0]["value"] = json!(1.0);
        let errors = validate_review(&review).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("synthetic benchmark geometry cannot qualify"))
        );
    }

    #[test]
    fn unknown_tradeoff_alternative_fails_closed() {
        let mut review = fixture();
        review["comparison"]["pairwise_tradeoffs"][0]["left_alternative_id"] = json!("not-present");
        assert!(
            validate_review(&review)
                .unwrap_err()
                .iter()
                .any(|error| error.contains("both referenced alternatives must exist"))
        );
    }

    fn paired_scenario_review_fixture() -> Value {
        let mut review = fixture();
        let baseline_id = string_value(review["alternatives"][0].get("alternative_id")).to_owned();
        let alternative_id =
            string_value(review["alternatives"][1].get("alternative_id")).to_owned();
        let baseline_design_sha =
            string_value(review["alternatives"][0].get("snapshot_sha256")).to_owned();
        let alternative_design_sha =
            string_value(review["alternatives"][1].get("snapshot_sha256")).to_owned();
        let registry_sha = string_value(review["inputs"].get("metric_registry_sha256")).to_owned();
        let boundary_sha = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
        let evaluation_sha = "1111111111111111111111111111111111111111111111111111111111111111";

        review["inputs"]["scenario_runs"] = json!([
            {
                "system": "symtropy",
                "manifest_sha256": "8888888888888888888888888888888888888888888888888888888888888888",
                "purpose": "synthetic-benchmark",
                "evidence_class": "synthetic",
                "status": "blocked",
                "model_id": "landscape-model-v0",
                "model_version": "0.1.0",
                "pairing_context": {
                    "alternative_id": baseline_id,
                    "design_snapshot_sha256": baseline_design_sha,
                    "boundary_conditions_sha256": boundary_sha,
                    "evaluation_plan_sha256": evaluation_sha,
                    "metric_registry_sha256": registry_sha,
                    "seed": 7,
                    "scope": "site-parcel"
                }
            },
            {
                "system": "symtropy",
                "manifest_sha256": "9999999999999999999999999999999999999999999999999999999999999999",
                "purpose": "synthetic-benchmark",
                "evidence_class": "synthetic",
                "status": "blocked",
                "model_id": "landscape-model-v0",
                "model_version": "0.1.0",
                "pairing_context": {
                    "alternative_id": alternative_id,
                    "design_snapshot_sha256": alternative_design_sha,
                    "boundary_conditions_sha256": boundary_sha,
                    "evaluation_plan_sha256": evaluation_sha,
                    "metric_registry_sha256": registry_sha,
                    "seed": 7,
                    "scope": "site-parcel"
                }
            }
        ]);
        review
    }

    #[test]
    fn review_alternatives_must_include_exact_baseline_design_snapshot() {
        let mut review = fixture();
        review["alternatives"][0]["snapshot_sha256"] =
            json!("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
        let errors = validate_review(&review).unwrap_err();
        assert!(errors.iter().any(|error| {
            error.contains("at least one alternative must bind the exact baseline design snapshot")
        }));
    }

    #[test]
    fn paired_scenario_runs_with_shared_context_are_accepted_as_unranked() {
        let review = paired_scenario_review_fixture();
        assert!(validate_review(&review).is_ok());
    }

    #[test]
    fn paired_scenario_runs_reject_boundary_condition_mismatch() {
        let mut review = paired_scenario_review_fixture();
        review["inputs"]["scenario_runs"][1]["pairing_context"]["boundary_conditions_sha256"] =
            json!("0000000000000000000000000000000000000000000000000000000000000000");
        let errors = validate_review(&review).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| { error.contains("must match across paired alternatives") })
        );
    }

    #[test]
    fn paired_scenario_run_must_bind_exact_alternative_snapshot() {
        let mut review = paired_scenario_review_fixture();
        review["inputs"]["scenario_runs"][1]["pairing_context"]["design_snapshot_sha256"] =
            json!("0000000000000000000000000000000000000000000000000000000000000000");
        let errors = validate_review(&review).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| { error.contains("does not match the exact alternative snapshot") })
        );
    }

    #[test]
    fn qualified_comparison_cannot_use_blocked_scenario_runs() {
        let mut review = paired_scenario_review_fixture();
        review["comparison"]["ranking_status"] = json!("qualified-comparison");
        let errors = validate_review(&review).unwrap_err();
        assert!(errors.iter().any(|error| {
            error.contains("qualified comparison cannot include blocked or conflicted runs")
        }));
    }

    #[test]
    fn invalid_input_digest_is_rejected() {
        let mut review = fixture();
        review["inputs"]["source_catalog_sha256"] = json!("not-a-digest");
        assert!(
            validate_review(&review)
                .unwrap_err()
                .iter()
                .any(|error| error.contains("expected 64 lowercase hexadecimal"))
        );
    }
}


#[cfg(test)]
mod schema_tests {
    use super::*;

    #[test]
    fn review_schema_uri_matches_fixture_contract() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../../../schemas/landscape-alternative-review-v0.schema.json"
        ))
        .expect("landscape review JSON Schema must be valid JSON");
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../fixtures/landscape/review-unranked-v0.json"
        ))
        .expect("canonical review fixture must be valid JSON");

        assert_eq!(schema["$id"], REVIEW_SCHEMA_URI);
        assert_eq!(schema["properties"]["schema"]["const"], REVIEW_SCHEMA);
        assert_eq!(
            schema["properties"]["schema_uri"]["const"],
            REVIEW_SCHEMA_URI
        );
        assert!(validate_review(&fixture).is_ok());
    }
}
