// SPDX-License-Identifier: AGPL-3.0-or-later

use jsonschema::draft202012;
use serde_json::{json, Value};

const TASK_SCHEMA: &str = include_str!("../../../docs/research/eqc-task-v0.1.schema.json");
const RECEIPT_SCHEMA: &str =
    include_str!("../../../docs/research/eqc-run-receipt-v0.1.schema.json");
const SEED_CORPUS: &str = include_str!("../../../docs/research/eqc-seed-corpus-v0.1.json");
const RECEIPT_SAMPLE: &str =
    include_str!("../../../docs/research/eqc-fixtures/run-receipt-reference-unqualified.json");

fn parse_json(raw: &str, label: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|error| panic!("{label} JSON must parse: {error}"))
}

fn compile_draft_2020_12(schema: &Value, label: &str) -> jsonschema::Validator {
    draft202012::meta::validate(schema)
        .unwrap_or_else(|error| panic!("{label} must pass the Draft 2020-12 meta-schema: {error}"));
    draft202012::options()
        .should_validate_formats(true)
        .build(schema)
        .unwrap_or_else(|error| panic!("{label} must compile as Draft 2020-12: {error}"))
}

#[test]
fn task_and_receipt_schemas_are_valid_draft_2020_12_documents() {
    let task_schema = parse_json(TASK_SCHEMA, "task schema");
    let receipt_schema = parse_json(RECEIPT_SCHEMA, "receipt schema");

    draft202012::meta::validate(&task_schema)
        .expect("task schema conforms to the Draft 2020-12 meta-schema");
    draft202012::meta::validate(&receipt_schema)
        .expect("receipt schema conforms to the Draft 2020-12 meta-schema");
}

#[test]
fn every_public_seed_task_validates_against_the_versioned_task_schema() {
    let schema = parse_json(TASK_SCHEMA, "task schema");
    let validator = compile_draft_2020_12(&schema, "task schema");
    let corpus = parse_json(SEED_CORPUS, "seed corpus");
    let tasks = corpus
        .get("tasks")
        .and_then(Value::as_array)
        .expect("seed corpus must contain a tasks array");

    assert_eq!(tasks.len(), 4, "public seed cardinality is versioned");
    for task in tasks {
        let task_id = task
            .get("task_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing-task-id>");
        let errors = validator
            .iter_errors(task)
            .map(|error| error.to_string())
            .collect::<Vec<_>>();
        assert!(
            errors.is_empty(),
            "task {task_id} does not satisfy the published schema: {}",
            errors.join("; ")
        );
    }
}

#[test]
fn task_schema_rejects_unknown_fields_and_missing_input_digests() {
    let schema = parse_json(TASK_SCHEMA, "task schema");
    let validator = compile_draft_2020_12(&schema, "task schema");
    let corpus = parse_json(SEED_CORPUS, "seed corpus");
    let mut task = corpus["tasks"][0].clone();

    task["unreviewed_field"] = json!(true);
    assert!(
        !validator.is_valid(&task),
        "additionalProperties=false must reject unknown task fields"
    );

    let mut task = corpus["tasks"][0].clone();
    task["input_refs"][0]
        .as_object_mut()
        .expect("input reference must be an object")
        .remove("sha256");
    assert!(
        !validator.is_valid(&task),
        "every task input must carry a SHA-256 digest"
    );
}

#[test]
fn run_receipt_sample_is_valid_and_qualification_is_non_compensatory() {
    let schema = parse_json(RECEIPT_SCHEMA, "receipt schema");
    let validator = compile_draft_2020_12(&schema, "receipt schema");
    let sample = parse_json(RECEIPT_SAMPLE, "receipt sample");

    assert!(
        validator.is_valid(&sample),
        "unqualified receipt sample must conform to the published schema"
    );

    let mut invalid_qualified = sample.clone();
    invalid_qualified["qualified"] = json!(true);
    invalid_qualified["dimensions"]["authority"] = json!(false);
    assert!(
        !validator.is_valid(&invalid_qualified),
        "qualified=true must not compensate for a failed authority dimension"
    );

    let mut invalid_verifier = sample.clone();
    invalid_verifier["qualified"] = json!(true);
    invalid_verifier["verifier"]["independent"] = json!(false);
    assert!(
        !validator.is_valid(&invalid_verifier),
        "qualified=true requires an independent verifier"
    );

    let mut invalid_replay = sample.clone();
    invalid_replay["qualified"] = json!(true);
    invalid_replay["verifier"]["independent"] = json!(true);
    invalid_replay["verifier"]["replay"] = json!("not_run");
    assert!(
        !validator.is_valid(&invalid_replay),
        "qualified=true requires successful replay"
    );

    let mut fully_qualified_shape = sample;
    fully_qualified_shape["qualified"] = json!(true);
    fully_qualified_shape["verifier"]["independent"] = json!(true);
    fully_qualified_shape["verifier"]["replay"] = json!("passed");
    assert!(
        validator.is_valid(&fully_qualified_shape),
        "all non-compensatory dimensions and verifier requirements satisfy the schema"
    );
}

#[test]
fn date_time_formats_are_validated_not_merely_annotated() {
    let schema = parse_json(RECEIPT_SCHEMA, "receipt schema");
    let validator = compile_draft_2020_12(&schema, "receipt schema");
    let mut sample = parse_json(RECEIPT_SAMPLE, "receipt sample");

    sample["timestamp"] = json!("2026-13-40T99:99:99Z");
    assert!(
        !validator.is_valid(&sample),
        "Draft 2020-12 date-time format validation must be enabled"
    );
}
