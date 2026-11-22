use jsonschema::validator_for;
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Component, Path};
use std::process::ExitCode;
use sha2::{Digest, Sha256};

const ANCHOR_MARKER: &str = "data-sfeb-id=\"";

const DEFAULT_SCHEMA: &str = "benchmarks/frontend/interface-contract-v1.schema.json";
const DEFAULT_CONTRACT: &str = "benchmarks/frontend/contracts/symthaea-ui-v0.json";
const DEFAULT_SOURCE: &str = "crates/bridges/symthaea-ui/src/app.rs";

fn read_json(path: &str) -> Result<Value, String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {path}: {error}"))?;
    serde_json::from_str(&source)
        .map_err(|error| format!("failed to parse JSON in {path}: {error}"))
}

/// Extract the explicit SFEB source-region anchors from the target component
/// source. Duplicate IDs are ambiguous and therefore fail closed.
fn extract_source_anchors(source: &str) -> Result<BTreeSet<String>, String> {
    let mut anchors = BTreeSet::new();
    let mut remaining = source;

    while let Some(marker_index) = remaining.find(ANCHOR_MARKER) {
        remaining = &remaining[marker_index + ANCHOR_MARKER.len()..];
        let Some(end) = remaining.find('"') else {
            return Err("unterminated data-sfeb-id source anchor".to_string());
        };
        let anchor = &remaining[..end];

        if anchor.is_empty()
            || anchor.len() > 128
            || !anchor.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')
            })
        {
            return Err(format!("invalid data-sfeb-id source anchor: {anchor:?}"));
        }
        if !anchors.insert(anchor.to_string()) {
            return Err(format!("duplicate data-sfeb-id source anchor: {anchor}"));
        }

        remaining = &remaining[end + 1..];
    }

    if anchors.is_empty() {
        return Err("no data-sfeb-id source anchors were found".to_string());
    }
    Ok(anchors)
}

fn validate_contract(
    schema: &Value,
    contract: &Value,
    source_text: &str,
) -> Result<(usize, usize), String> {
    let validator = validator_for(schema)
        .map_err(|error| format!("invalid JSON Schema: {error}"))?;

    let errors: Vec<String> = validator
        .iter_errors(contract)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect();

    if !errors.is_empty() {
        return Err(format!(
            "contract does not satisfy its schema:\n{}",
            errors.join("\n")
        ));
    }

    let source_anchors = extract_source_anchors(source_text)?;
    let regions = contract
        .get("regions")
        .and_then(Value::as_array)
        .ok_or_else(|| "contract.regions must be an array".to_string())?;

    let mut contract_anchors = BTreeSet::new();
    for region in regions {
        let anchor = region
            .get("source_anchor")
            .and_then(Value::as_str)
            .ok_or_else(|| "every contract region needs a source_anchor".to_string())?;

        if !contract_anchors.insert(anchor.to_string()) {
            return Err(format!(
                "duplicate source_anchor in interface contract: {anchor}"
            ));
        }
        if !source_anchors.contains(anchor) {
            return Err(format!(
                "contract source_anchor is absent from Leptos source: {anchor}"
            ));
        }
    }

    let uncontracted: Vec<_> = source_anchors
        .difference(&contract_anchors)
        .cloned()
        .collect();
    if !uncontracted.is_empty() {
        return Err(format!(
            "Leptos source contains anchors missing from the interface contract: {}",
            uncontracted.join(", ")
        ));
    }

    let journeys = contract
        .get("acceptance")
        .and_then(|acceptance| acceptance.get("journeys"))
        .and_then(Value::as_array)
        .ok_or_else(|| "contract.acceptance.journeys must be an array".to_string())?;

    if journeys.is_empty() {
        return Err("interface contract must include at least one acceptance journey".to_string());
    }

    Ok((regions.len(), journeys.len()))
}


const DEFAULT_TASK_SCHEMA: &str = "benchmarks/frontend/task-oracle-v1.schema.json";
const DEFAULT_TASK: &str = "benchmarks/frontend/tasks/sfeb-l1-service-dashboard-v1.json";

fn fixture_sha256(html: &[u8], css: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(html);
    digest.update([0]);
    digest.update(css);
    hex::encode(digest.finalize())
}

fn validate_task_instance(
    schema: &Value,
    task: &Value,
    html: &[u8],
    css: &[u8],
) -> Result<(usize, usize), String> {
    let validator = validator_for(schema)
        .map_err(|error| format!("invalid SFEB task schema: {error}"))?;
    let errors: Vec<String> = validator
        .iter_errors(task)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect();
    if !errors.is_empty() {
        return Err(format!(
            "task does not satisfy its schema:\n{}",
            errors.join("\n")
        ));
    }

    let html_text = std::str::from_utf8(html)
        .map_err(|error| format!("reference HTML is not UTF-8: {error}"))?;
    let css_text = std::str::from_utf8(css)
        .map_err(|error| format!("reference CSS is not UTF-8: {error}"))?;

    if html_text.contains("https://")
        || html_text.contains("http://")
        || css_text.contains("https://")
        || css_text.contains("http://")
        || html_text.contains("<script")
    {
        return Err("reference fixture must not load remote resources or scripts".to_string());
    }

    let expected_digest = task
        .get("reference_fixture")
        .and_then(|fixture| fixture.get("sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| "task.reference_fixture.sha256 is required".to_string())?;
    let actual_digest = fixture_sha256(html, css);
    if actual_digest != expected_digest {
        return Err(format!(
            "reference fixture digest mismatch: expected {expected_digest}, got {actual_digest}"
        ));
    }

    let anchors = extract_source_anchors(html_text)?;
    let regions = task
        .get("required_regions")
        .and_then(Value::as_array)
        .ok_or_else(|| "task.required_regions must be an array".to_string())?;

    let mut region_ids = BTreeSet::new();
    for region in regions {
        let id = region
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "every required region needs an id".to_string())?;
        if !region_ids.insert(id.to_string()) {
            return Err(format!("duplicate required region id: {id}"));
        }
        if !anchors.contains(id) {
            return Err(format!(
                "required region id is missing from reference fixture: {id}"
            ));
        }
    }

    let assertions = task
        .get("assertions")
        .and_then(Value::as_array)
        .ok_or_else(|| "task.assertions must be an array".to_string())?;
    let mut assertion_ids = BTreeSet::new();
    for assertion in assertions {
        let id = assertion
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "every assertion needs an id".to_string())?;
        if !assertion_ids.insert(id.to_string()) {
            return Err(format!("duplicate assertion id: {id}"));
        }
    }

    Ok((regions.len(), assertions.len()))
}

fn run_task(schema_path: &str, task_path: &str) -> Result<(), String> {
    let schema = read_json(schema_path)?;
    let task = read_json(task_path)?;
    let fixture_dir = task
        .get("reference_fixture")
        .and_then(|fixture| fixture.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| "task.reference_fixture.path is required".to_string())?;
    let fixture_dir = Path::new(fixture_dir);
    if fixture_dir.is_absolute()
        || fixture_dir.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err("reference fixture path must remain inside the repository".to_string());
    }

    let html_path = fixture_dir.join("reference.html");
    let css_path = fixture_dir.join("style.css");
    let html = fs::read(&html_path)
        .map_err(|error| format!("failed to read {}: {error}", html_path.display()))?;
    let css = fs::read(&css_path)
        .map_err(|error| format!("failed to read {}: {error}", css_path.display()))?;

    let (regions, assertions) = validate_task_instance(&schema, &task, &html, &css)?;
    let task_id = task
        .get("task_id")
        .and_then(Value::as_str)
        .unwrap_or("<unnamed>");
    let status = task
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");

    println!(
        "SFEB task valid: {task_id} (status={status}, required_regions={regions}, assertions={assertions})"
    );
    println!("Reference fixture digest: verified; local-only asset policy: verified.");
    println!("Note: schema and fixture validity do not imply browser or agent verification.");
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "Usage:\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- [CONTRACT_SCHEMA [CONTRACT_JSON [APP_SOURCE.rs]]]\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --task [TASK_SCHEMA [TASK_JSON]]\n\nThe default mode validates the Leptos interface contract and exact source anchors. --task validates the task schema, required fixture regions and canonical fixture digest."
        );
        return Ok(());
    }
    if args.first().is_some_and(|arg| arg == "--task") {
        if args.len() > 3 {
            return Err("--task accepts at most a schema path and task path".to_string());
        }
        return run_task(
            args.get(1).map(String::as_str).unwrap_or(DEFAULT_TASK_SCHEMA),
            args.get(2).map(String::as_str).unwrap_or(DEFAULT_TASK),
        );
    }
    if args.len() > 3 {
        return Err("expected at most three paths; use --help".to_string());
    }

    let schema_path = args.first().map(String::as_str).unwrap_or(DEFAULT_SCHEMA);
    let contract_path = args.get(1).map(String::as_str).unwrap_or(DEFAULT_CONTRACT);
    let source_path = args.get(2).map(String::as_str).unwrap_or(DEFAULT_SOURCE);

    let schema = read_json(schema_path)?;
    let contract = read_json(contract_path)?;
    let source_text = fs::read_to_string(source_path)
        .map_err(|error| format!("failed to read Leptos source {source_path}: {error}"))?;

    let (regions, journeys) = validate_contract(&schema, &contract, &source_text)?;

    let contract_id = contract
        .get("contract_id")
        .and_then(Value::as_str)
        .unwrap_or("<unnamed>");
    let status = contract
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");

    println!(
        "SFEB contract valid: {contract_id} (status={status}, regions={regions}, journeys={journeys})"
    );
    println!("Source anchor mapping: exact match; no duplicates or orphans.");
    println!("Note: schema validity and source mapping do not imply browser verification.");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sfeb-contract-check: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_unique_source_anchors() {
        let source = r#"
            <main data-sfeb-id="page.main">
            <button data-sfeb-id="button.submit">
        "#;
        let anchors = extract_source_anchors(source).expect("valid anchors");
        assert_eq!(anchors.len(), 2);
        assert!(anchors.contains("page.main"));
        assert!(anchors.contains("button.submit"));
    }

    #[test]
    fn rejects_duplicate_source_anchors() {
        let source = r#"
            <main data-sfeb-id="page.main">
            <section data-sfeb-id="page.main">
        "#;
        assert!(extract_source_anchors(source).is_err());
    }

    #[test]
    fn rejects_unsafe_source_anchors() {
        let source = r#"<main data-sfeb-id="../private">"#;
        assert!(extract_source_anchors(source).is_err());
    }

    #[test]
    fn rejects_missing_or_unterminated_source_anchors() {
        assert!(extract_source_anchors("<main></main>").is_err());
        assert!(extract_source_anchors(r#"<main data-sfeb-id="unfinished>"#).is_err());
    }

    #[test]
    fn checked_in_symthaea_ui_contract_matches_schema_and_source() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../interface-contract-v1.schema.json"
        ))
        .expect("checked-in interface schema is valid JSON");
        let contract: Value = serde_json::from_str(include_str!(
            "../../contracts/symthaea-ui-v0.json"
        ))
        .expect("checked-in interface contract is valid JSON");
        let source = include_str!("../../../../crates/bridges/symthaea-ui/src/app.rs");

        let (regions, journeys) = validate_contract(&schema, &contract, source)
            .expect("checked-in contract validates against its schema and source anchors");
        assert_eq!(regions, 16);
        assert_eq!(journeys, 6);
    }

    #[test]
    fn checked_in_sfeb_task_matches_schema_and_fixture() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../task-oracle-v1.schema.json"
        ))
        .expect("checked-in task schema is valid JSON");
        let task: Value = serde_json::from_str(include_str!(
            "../../tasks/sfeb-l1-service-dashboard-v1.json"
        ))
        .expect("checked-in task is valid JSON");
        let html = include_bytes!("../../fixtures/service-dashboard/reference.html");
        let css = include_bytes!("../../fixtures/service-dashboard/style.css");

        let (regions, assertions) = validate_task_instance(&schema, &task, html, css)
            .expect("task schema, canonical fixture digest and region anchors must validate");
        assert_eq!(regions, 8);
        assert_eq!(assertions, 11);
    }
}
