use jsonschema::validator_for;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Component, Path};
use sha1::Sha1;
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


const DEFAULT_PATTERN_SCHEMA: &str =
    "benchmarks/frontend/knowledge/leptos-0.8/pattern-v1.schema.json";
const DEFAULT_PATTERN_CORPUS: &str =
    "benchmarks/frontend/knowledge/leptos-0.8/patterns-v0.json";

/// Calculate the Git blob object SHA-1 (header + content), not the raw-file SHA-1.
fn git_blob_sha1(bytes: &[u8]) -> String {
    let header = format!("blob {}\0", bytes.len());
    let mut digest = Sha1::new();
    digest.update(header.as_bytes());
    digest.update(bytes);
    hex::encode(digest.finalize())
}

fn validate_relative_repo_path(raw: &str) -> Result<(), String> {
    let path = Path::new(raw);
    if raw.is_empty()
        || raw.contains('\\')
        || raw.split('/').any(|part| part.is_empty() || part == "." || part == "..")
        || path.is_absolute()
        || path.components().any(|component| {
            !matches!(component, Component::Normal(_))
        })
    {
        return Err(format!(
            "source path must be a normalized repository-relative path: {raw}"
        ));
    }
    Ok(())
}

fn verify_pattern_source_blob(
    repository_root: &Path,
    raw_path: &str,
    expected_sha: &str,
    pattern_id: &str,
    source_digests: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    validate_relative_repo_path(raw_path)?;
    let expected_sha = expected_sha.to_ascii_lowercase();

    if let Some(previous) = source_digests.get(raw_path) {
        if previous != &expected_sha {
            return Err(format!(
                "patterns referencing {raw_path} disagree on the source blob SHA"
            ));
        }
        return Ok(());
    }

    let canonical_root = repository_root
        .canonicalize()
        .map_err(|error| format!(
            "failed to resolve repository root {}: {error}",
            repository_root.display()
        ))?;
    let joined_path = canonical_root.join(raw_path);
    let full_path = joined_path
        .canonicalize()
        .map_err(|error| format!(
            "failed to resolve pattern source {}: {error}",
            joined_path.display()
        ))?;
    if !full_path.starts_with(&canonical_root) {
        return Err(format!(
            "pattern source resolves outside the repository root: {raw_path}"
        ));
    }
    let bytes = std::fs::read(&full_path)
        .map_err(|error| format!(
            "failed to read pattern source {}: {error}",
            full_path.display()
        ))?;
    let actual_sha = git_blob_sha1(&bytes);
    if actual_sha != expected_sha {
        return Err(format!(
            "stale source provenance for {pattern_id}: {raw_path} expected {expected_sha}, got {actual_sha}"
        ));
    }
    source_digests.insert(raw_path.to_string(), actual_sha);
    Ok(())
}

fn validate_pattern_corpus(
    schema: &Value,
    corpus: &Value,
    repository_root: &Path,
) -> Result<(usize, usize), String> {
    let validator = validator_for(schema)
        .map_err(|error| format!("invalid Leptos pattern schema: {error}"))?;
    let errors: Vec<String> = validator
        .iter_errors(corpus)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect();
    if !errors.is_empty() {
        return Err(format!(
            "Leptos pattern corpus does not satisfy its schema:\\n{}",
            errors.join("\\n")
        ));
    }

    let patterns = corpus
        .get("patterns")
        .and_then(Value::as_array)
        .ok_or_else(|| "corpus.patterns must be an array".to_string())?;
    let mut pattern_ids = BTreeSet::new();
    let mut source_digests = BTreeMap::<String, String>::new();

    for pattern in patterns {
        let pattern_id = pattern
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "every Leptos pattern requires an id".to_string())?;
        if !pattern_ids.insert(pattern_id.to_string()) {
            return Err(format!("duplicate Leptos pattern id: {pattern_id}"));
        }

        let source = pattern
            .get("source")
            .ok_or_else(|| format!("pattern {pattern_id} is missing source provenance"))?;
        let raw_path = source
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("pattern {pattern_id} is missing source.path"))?;
        let expected_sha = source
            .get("blob_sha")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("pattern {pattern_id} is missing source.blob_sha"))?;
        verify_pattern_source_blob(
            repository_root,
            raw_path,
            expected_sha,
            pattern_id,
            &mut source_digests,
        )?;

        if let Some(related) = source.get("related_sources").and_then(Value::as_array) {
            for related_source in related {
                let related_path = related_source
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!(
                        "pattern {pattern_id} has a related source without a path"
                    ))?;
                let related_sha = related_source
                    .get("blob_sha")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!(
                        "pattern {pattern_id} has a related source without a blob SHA"
                    ))?;
                verify_pattern_source_blob(
                    repository_root,
                    related_path,
                    related_sha,
                    pattern_id,
                    &mut source_digests,
                )?;
            }
        }
    }

    Ok((patterns.len(), source_digests.len()))
}

fn run_patterns(schema_path: &str, corpus_path: &str, repository_root: &str) -> Result<(), String> {
    let schema = read_json(schema_path)?;
    let corpus = read_json(corpus_path)?;
    let root = Path::new(repository_root);
    let (patterns, sources) = validate_pattern_corpus(&schema, &corpus, root)?;

    let corpus_id = corpus
        .get("corpus_id")
        .and_then(Value::as_str)
        .unwrap_or("<unnamed>");
    let framework = corpus
        .get("framework")
        .and_then(|value| value.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");
    let api_family = corpus
        .get("framework")
        .and_then(|value| value.get("api_family"))
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");

    println!(
        "Leptos knowledge corpus valid: {corpus_id} ({framework} {api_family}, patterns={patterns}, source_files={sources})"
    );
    println!("Git blob provenance: exact; no stale, duplicate or unsafe source paths.");
    println!("Note: source provenance and schema validity do not imply compilation or browser verification.");
    Ok(())
}


const DEFAULT_RETRIEVAL_SCHEMA: &str =
    "benchmarks/frontend/knowledge/leptos-0.8/retrieval-v1.schema.json";
const DEFAULT_RETRIEVAL_SET: &str =
    "benchmarks/frontend/knowledge/leptos-0.8/retrieval-queries-v0.json";
const DEFAULT_RETRIEVAL_CORPUS: &str =
    "benchmarks/frontend/knowledge/leptos-0.8/patterns-v0.json";

fn validate_retrieval_set(
    schema: &Value,
    retrieval: &Value,
    pattern_corpus: &Value,
) -> Result<(usize, usize), String> {
    let validator = validator_for(schema)
        .map_err(|error| format!("invalid retrieval schema: {error}"))?;
    let errors: Vec<String> = validator
        .iter_errors(retrieval)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect();
    if !errors.is_empty() {
        return Err(format!(
            "retrieval set does not satisfy its schema:\n{}",
            errors.join("\n")
        ));
    }

    let known_ids: BTreeSet<String> = pattern_corpus
        .get("patterns")
        .and_then(Value::as_array)
        .ok_or_else(|| "pattern corpus must contain a patterns array".to_string())?
        .iter()
        .filter_map(|pattern| pattern.get("id").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    if known_ids.is_empty() {
        return Err("pattern corpus contains no pattern identifiers".to_string());
    }

    let queries = retrieval
        .get("queries")
        .and_then(Value::as_array)
        .ok_or_else(|| "retrieval.queries must be an array".to_string())?;
    let mut query_ids = BTreeSet::new();
    let mut positive = 0;
    let mut negatives = 0;

    for query in queries {
        let id = query
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "every retrieval query requires an id".to_string())?;
        if !query_ids.insert(id.to_string()) {
            return Err(format!("duplicate retrieval query id: {id}"));
        }

        let kind = query
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("retrieval query {id} requires kind"))?;
        let expected = query
            .get("expected_pattern_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("retrieval query {id} requires expected_pattern_ids"))?;

        match kind {
            "positive" => {
                if expected.is_empty() {
                    return Err(format!("positive query {id} has no expected pattern IDs"));
                }
                positive += 1;
            }
            "hard_negative" => {
                if !expected.is_empty() {
                    return Err(format!("hard-negative query {id} must have an empty expected set"));
                }
                negatives += 1;
            }
            _ => return Err(format!("retrieval query {id} has unsupported kind {kind}")),
        }

        for expected_id in expected {
            let expected_id = expected_id
                .as_str()
                .ok_or_else(|| format!("query {id} contains a non-string pattern ID"))?;
            if !known_ids.contains(expected_id) {
                return Err(format!(
                    "query {id} references unknown Leptos pattern ID: {expected_id}"
                ));
            }
        }
    }

    if positive == 0 || negatives == 0 {
        return Err("retrieval set must include both positive queries and hard negatives".to_string());
    }
    Ok((positive, negatives))
}

fn run_retrieval(
    schema_path: &str,
    retrieval_path: &str,
    corpus_path: &str,
) -> Result<(), String> {
    let schema = read_json(schema_path)?;
    let retrieval = read_json(retrieval_path)?;
    let corpus = read_json(corpus_path)?;
    let (positive, negatives) = validate_retrieval_set(&schema, &retrieval, &corpus)?;
    let total = positive + negatives;
    let exposure = retrieval
        .get("exposure")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");
    println!(
        "Leptos retrieval set valid: queries={total}, positive={positive}, hard_negatives={negatives}, exposure={exposure}"
    );
    println!("Gold pattern references: all resolve to the versioned corpus.");
    println!("Note: this validates the evaluation set, not retrieval quality.");
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "Usage:\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- [CONTRACT_SCHEMA [CONTRACT_JSON [APP_SOURCE.rs]]]\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --task [TASK_SCHEMA [TASK_JSON]]\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --patterns [PATTERN_SCHEMA [CORPUS_JSON [REPOSITORY_ROOT]]]\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --retrieval [RETRIEVAL_SCHEMA [QUERY_SET_JSON [PATTERN_CORPUS_JSON]]]\n\nThe default mode validates the Leptos interface contract and exact source anchors. --task validates the task schema, required fixture regions and canonical fixture digest. --patterns validates the Leptos knowledge schema and exact Git blob provenance for every source example. --retrieval validates positive/hard-negative queries and their gold IDs against the versioned pattern corpus."
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
    if args.first().is_some_and(|arg| arg == "--patterns") {
        if args.len() > 4 {
            return Err("--patterns accepts at most schema, corpus and repository-root paths".to_string());
        }
        return run_patterns(
            args.get(1).map(String::as_str).unwrap_or(DEFAULT_PATTERN_SCHEMA),
            args.get(2).map(String::as_str).unwrap_or(DEFAULT_PATTERN_CORPUS),
            args.get(3).map(String::as_str).unwrap_or("."),
        );
    }
    if args.first().is_some_and(|arg| arg == "--retrieval") {
        if args.len() > 4 {
            return Err("--retrieval accepts at most schema, query-set and pattern-corpus paths".to_string());
        }
        return run_retrieval(
            args.get(1).map(String::as_str).unwrap_or(DEFAULT_RETRIEVAL_SCHEMA),
            args.get(2).map(String::as_str).unwrap_or(DEFAULT_RETRIEVAL_SET),
            args.get(3).map(String::as_str).unwrap_or(DEFAULT_RETRIEVAL_CORPUS),
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
    fn checked_in_retrieval_set_resolves_to_pattern_corpus() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/retrieval-v1.schema.json"
        ))
        .expect("checked-in retrieval schema is valid JSON");
        let retrieval: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/retrieval-queries-v0.json"
        ))
        .expect("checked-in retrieval set is valid JSON");
        let corpus: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/patterns-v0.json"
        ))
        .expect("checked-in pattern corpus is valid JSON");

        let (positive, negatives) = validate_retrieval_set(&schema, &retrieval, &corpus)
            .expect("retrieval set must match schema and corpus gold IDs");
        assert_eq!(positive, 16);
        assert_eq!(negatives, 8);
    }

    #[test]
    fn retrieval_validator_rejects_unknown_pattern_ids() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/retrieval-v1.schema.json"
        ))
        .expect("checked-in retrieval schema is valid JSON");
        let mut retrieval: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/retrieval-queries-v0.json"
        ))
        .expect("checked-in retrieval set is valid JSON");
        let corpus: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/patterns-v0.json"
        ))
        .expect("checked-in pattern corpus is valid JSON");
        retrieval["queries"][0]["expected_pattern_ids"][0] =
            Value::String("UNKNOWN-PATTERN".into());

        assert!(validate_retrieval_set(&schema, &retrieval, &corpus).is_err());
    }

    #[test]
    fn git_blob_sha1_matches_known_git_object_vector() {
        assert_eq!(
            git_blob_sha1(b"test\n"),
            "9daeafb9864cf43055ae93beb0afd6c7d144bfa4"
        );
    }

    #[test]
    fn checked_in_leptos_pattern_corpus_matches_schema_and_git_blobs() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/pattern-v1.schema.json"
        ))
        .expect("checked-in Leptos pattern schema is valid JSON");
        let corpus: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/patterns-v0.json"
        ))
        .expect("checked-in Leptos pattern corpus is valid JSON");
        let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");

        let (patterns, sources) = validate_pattern_corpus(&schema, &corpus, &repository_root)
            .expect("pattern corpus schema and Git blob provenance must match");
        assert_eq!(patterns, 8);
        assert_eq!(sources, 2);
    }

    #[test]
    fn pattern_validator_rejects_unsafe_repository_paths() {
        assert!(validate_relative_repo_path("../outside.rs").is_err());
        assert!(validate_relative_repo_path("/etc/passwd").is_err());
        assert!(validate_relative_repo_path("crates/../secret.rs").is_err());
        assert!(validate_relative_repo_path("crates/./secret.rs").is_err());
        assert!(validate_relative_repo_path(r"crates\\secret.rs").is_err());
        assert!(validate_relative_repo_path("crates//secret.rs").is_err());
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
        assert_eq!(regions, 17);
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

    #[test]
    fn task_validator_rejects_fixture_digest_tampering() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../task-oracle-v1.schema.json"
        ))
        .expect("checked-in task schema is valid JSON");
        let task: Value = serde_json::from_str(include_str!(
            "../../tasks/sfeb-l1-service-dashboard-v1.json"
        ))
        .expect("checked-in task is valid JSON");
        let html = include_bytes!("../../fixtures/service-dashboard/reference.html");
        let mut css = include_bytes!("../../fixtures/service-dashboard/style.css").to_vec();
        css.push(b' ');

        assert!(validate_task_instance(&schema, &task, html, &css).is_err());
    }

    #[test]
    fn task_validator_rejects_uncontracted_regions() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../task-oracle-v1.schema.json"
        ))
        .expect("checked-in task schema is valid JSON");
        let mut task: Value = serde_json::from_str(include_str!(
            "../../tasks/sfeb-l1-service-dashboard-v1.json"
        ))
        .expect("checked-in task is valid JSON");
        let html = include_bytes!("../../fixtures/service-dashboard/reference.html");
        let css = include_bytes!("../../fixtures/service-dashboard/style.css");
        task["required_regions"][0]["id"] = Value::String("missing.region".into());

        assert!(validate_task_instance(&schema, &task, html, css).is_err());
    }
}
