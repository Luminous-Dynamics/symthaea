use jsonschema::validator_for;
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::process::ExitCode;

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

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "Usage: cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- [SCHEMA.json [CONTRACT.json [APP_SOURCE.rs]]]\n\nDefaults validate the checked-in symthaea-ui source-observed contract."
        );
        return Ok(());
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
}
