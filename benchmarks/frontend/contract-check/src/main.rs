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

const DEFAULT_MUTATION_SCHEMA: &str =
    "benchmarks/frontend/qualification/mutation-campaign-v1.schema.json";
const DEFAULT_MUTATION_CAMPAIGN: &str =
    "benchmarks/frontend/qualification/mutation-campaign-v1.json";

/// Validate the qualification plan itself. This does not execute mutants.
fn validate_mutation_campaign(
    schema: &Value,
    campaign: &Value,
) -> Result<(usize, usize, usize), String> {
    let validator = validator_for(schema)
        .map_err(|error| format!("invalid mutation campaign schema: {error}"))?;
    let errors: Vec<String> = validator
        .iter_errors(campaign)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect();
    if !errors.is_empty() {
        return Err(format!(
            "mutation campaign does not satisfy its schema:\n{}",
            errors.join("\n")
        ));
    }

    let mutations = campaign
        .get("mutations")
        .and_then(Value::as_array)
        .ok_or_else(|| "campaign.mutations must be an array".to_string())?;
    let positives = campaign
        .get("positive_controls")
        .and_then(Value::as_array)
        .ok_or_else(|| "campaign.positive_controls must be an array".to_string())?;
    let relations = campaign
        .get("metamorphic_relations")
        .and_then(Value::as_array)
        .ok_or_else(|| "campaign.metamorphic_relations must be an array".to_string())?;

    if mutations.len() < 20 || positives.len() < 5 || relations.len() < 5 {
        return Err(format!(
            "qualification campaign is incomplete: mutations={}, positive_controls={}, metamorphic_relations={}",
            mutations.len(),
            positives.len(),
            relations.len()
        ));
    }

    let mut mutation_ids = BTreeSet::new();
    let mut critical_mutations = 0usize;
    let mut critical_security_mutations = 0usize;
    let mut critical_evidence_mutations = 0usize;
    let mut security_mutations = 0usize;
    let mut evidence_mutations = 0usize;
    let mut high_mutations = 0usize;
    for mutation in mutations {
        let id = mutation
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "every mutation requires an id".to_string())?;
        if !mutation_ids.insert(id.to_string()) {
            return Err(format!("duplicate mutation id: {id}"));
        }
        if mutation.get("expected_disposition").and_then(Value::as_str) != Some("reject") {
            return Err(format!("mutation {id} must expect rejection"));
        }
        let category = mutation
            .get("category")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("mutation {id} is missing category"))?;
        let severity = mutation
            .get("severity")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("mutation {id} is missing severity"))?;
        if severity == "critical" {
            critical_mutations += 1;
            if category == "security" {
                critical_security_mutations += 1;
            }
            if category == "evidence_integrity" {
                critical_evidence_mutations += 1;
            }
        }
        if severity == "high" {
            high_mutations += 1;
        }
        if category == "security" {
            security_mutations += 1;
        }
        if category == "evidence_integrity" {
            evidence_mutations += 1;
        }
    }

    let mut positive_ids = BTreeSet::new();
    for control in positives {
        let id = control
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "every positive control requires an id".to_string())?;
        if !positive_ids.insert(id.to_string()) {
            return Err(format!("duplicate positive-control id: {id}"));
        }
        if control.get("expected_disposition").and_then(Value::as_str) != Some("accept") {
            return Err(format!("positive control {id} must expect acceptance"));
        }
    }

    let mut relation_ids = BTreeSet::new();
    for relation in relations {
        let id = relation
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "every metamorphic relation requires an id".to_string())?;
        if !relation_ids.insert(id.to_string()) {
            return Err(format!("duplicate metamorphic-relation id: {id}"));
        }
    }

    let targets = campaign
        .get("acceptance_targets")
        .ok_or_else(|| "campaign.acceptance_targets is required".to_string())?;
    for key in [
        "critical_mutation_kill_rate_min",
        "security_mutation_kill_rate_min",
        "evidence_integrity_mutation_kill_rate_min",
    ] {
        let value = targets
            .get(key)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("acceptance target {key} must be numeric"))?;
        if (value - 1.0).abs() > f64::EPSILON {
            return Err(format!("{key} must remain a hard 100% gate"));
        }
    }
    for (key, minimum) in [
        ("high_severity_mutation_kill_rate_min", 0.95),
        ("overall_mutation_kill_rate_min", 0.90),
    ] {
        let value = targets
            .get(key)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("acceptance target {key} must be numeric"))?;
        if value + f64::EPSILON < minimum {
            return Err(format!("{key} must be at least {minimum:.2}"));
        }
    }
    let false_reject_limit = targets
        .get("positive_control_false_reject_rate_max")
        .and_then(Value::as_f64)
        .ok_or_else(|| "positive-control false-reject target must be numeric".to_string())?;
    if false_reject_limit.abs() > f64::EPSILON {
        return Err("positive-control false-reject rate must remain zero".to_string());
    }

    if critical_mutations == 0
        || critical_security_mutations == 0
        || critical_evidence_mutations == 0
        || security_mutations == 0
        || evidence_mutations == 0
        || high_mutations == 0
    {
        return Err(
            "qualification campaign must include critical, high-severity, security and critical evidence-integrity mutants"
                .to_string(),
        );
    }

    Ok((mutations.len(), positives.len(), relations.len()))
}

fn run_mutations(schema_path: &str, campaign_path: &str) -> Result<(), String> {
    let schema = read_json(schema_path)?;
    let campaign = read_json(campaign_path)?;
    let (mutations, positives, relations) =
        validate_mutation_campaign(&schema, &campaign)?;
    let campaign_id = campaign
        .get("campaign_id")
        .and_then(Value::as_str)
        .unwrap_or("<unnamed>");
    let status = campaign
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");

    println!(
        "SFEB qualification plan valid: {campaign_id} (mutations={mutations}, positive_controls={positives}, metamorphic_relations={relations}, status={status})"
    );
    println!("Critical, security and evidence-integrity kill-rate targets remain hard 100% gates.");
    println!("Note: this validates the campaign definition only; it does not execute mutants or qualify the browser oracles.");
    Ok(())
}


const DEFAULT_QUALIFICATION_REPORT_SCHEMA: &str =
    "benchmarks/frontend/qualification/qualification-report-v1.schema.json";

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_hex_digest(value: &str, lengths: &[usize]) -> bool {
    lengths.contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn wilson_interval(successes: usize, trials: usize) -> Value {
    if trials == 0 {
        return Value::Null;
    }
    let z = 1.959_963_984_540_054_f64;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    serde_json::json!({
        "lower_95": (center - margin).max(0.0),
        "upper_95": (center + margin).min(1.0)
    })
}

fn rate_summary(successes: usize, trials: usize) -> Value {
    serde_json::json!({
        "successes": successes,
        "trials": trials,
        "rate": if trials == 0 { Value::Null } else { serde_json::json!(successes as f64 / trials as f64) },
        "wilson_95": wilson_interval(successes, trials)
    })
}

fn string_set(value: &Value, key: &str) -> Result<BTreeSet<String>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))?;
    let mut set = BTreeSet::new();
    for entry in values {
        let item = entry
            .as_str()
            .ok_or_else(|| format!("{key} entries must be strings"))?;
        if !set.insert(item.to_string()) {
            return Err(format!("{key} contains a duplicate value: {item}"));
        }
    }
    Ok(set)
}

fn contains_all(required: &[Value], actual: &BTreeSet<String>) -> bool {
    required
        .iter()
        .filter_map(Value::as_str)
        .all(|value| actual.contains(value))
}

fn expected_oracle_fired(expected: &[Value], observed: &BTreeSet<String>) -> bool {
    expected
        .iter()
        .filter_map(Value::as_str)
        .any(|value| observed.contains(value))
}

fn read_beneath(root: &Path, relative_path: &str) -> Result<Vec<u8>, String> {
    validate_relative_repo_path(relative_path)?;
    let candidate = root.join(relative_path);
    let canonical = candidate.canonicalize().map_err(|error| {
        format!("failed to resolve evidence artifact {}: {error}", candidate.display())
    })?;
    if !canonical.starts_with(root) {
        return Err(format!(
            "evidence artifact escapes its artifact root: {relative_path}"
        ));
    }
    fs::read(&canonical).map_err(|error| {
        format!("failed to read evidence artifact {}: {error}", canonical.display())
    })
}

fn validate_evidence_run(
    run: &Value,
    artifact_root: &Path,
    case_id: &str,
    phase: &str,
    task_id: &str,
    fixture_id: &str,
    expected_environment_sha: &str,
) -> Result<(), String> {
    let run_id = run
        .get("run_id")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{case_id} {phase}: run_id is required"))?;
    let source_commit = run
        .get("source_commit")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{case_id} {phase}: source_commit is required"))?;
    let environment_sha = run
        .get("environment_manifest_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{case_id} {phase}: environment digest is required"))?;
    let expected_receipt_sha = run
        .get("evidence_receipt_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{case_id} {phase}: receipt digest is required"))?;
    let receipt_path = run
        .get("evidence_receipt_path")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{case_id} {phase}: receipt path is required"))?;

    if !is_hex_digest(source_commit, &[40, 64])
        || !is_hex_digest(environment_sha, &[64])
        || !is_hex_digest(expected_receipt_sha, &[64])
    {
        return Err(format!("{case_id} {phase}: malformed source or evidence digest"));
    }
    if environment_sha != expected_environment_sha {
        return Err(format!("{case_id} {phase}: environment manifest does not match report"));
    }

    let receipt_bytes = read_beneath(artifact_root, receipt_path)?;
    let actual_receipt_sha = sha256_bytes(&receipt_bytes);
    if actual_receipt_sha != expected_receipt_sha.to_ascii_lowercase() {
        return Err(format!("{case_id} {phase}: evidence receipt SHA-256 mismatch"));
    }
    let receipt: Value = serde_json::from_slice(&receipt_bytes)
        .map_err(|error| format!("{case_id} {phase}: receipt JSON is invalid: {error}"))?;

    for (field, expected) in [
        ("schema", "sfeb-frontend-evidence/v1"),
        ("run_id", run_id),
        ("task_id", task_id),
        ("fixture_id", fixture_id),
        ("source_commit", source_commit),
        ("environment_manifest_sha256", environment_sha),
    ] {
        let actual = receipt.get(field).and_then(Value::as_str).unwrap_or_default();
        if actual != expected {
            return Err(format!(
                "{case_id} {phase}: receipt field {field} does not match the report"
            ));
        }
    }

    let receipt_artifacts = receipt
        .get("artifacts")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{case_id} {phase}: receipt.artifacts must be an array"))?;
    let reported_artifacts = run
        .get("artifacts")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{case_id} {phase}: run.artifacts must be an array"))?;
    if receipt_artifacts.len() != 3 || reported_artifacts.len() != 3 {
        return Err(format!("{case_id} {phase}: exactly three evidence artifacts are required"));
    }

    let mut seen = BTreeSet::new();
    for expected_kind in ["screenshot", "accessibility", "layout"] {
        let reported = reported_artifacts
            .iter()
            .find(|item| item.get("kind").and_then(Value::as_str) == Some(expected_kind))
            .ok_or_else(|| format!("{case_id} {phase}: missing {expected_kind} artifact path"))?;
        let receipt_item = receipt_artifacts
            .iter()
            .find(|item| item.get("kind").and_then(Value::as_str) == Some(expected_kind))
            .ok_or_else(|| format!("{case_id} {phase}: receipt missing {expected_kind} digest"))?;

        if !seen.insert(expected_kind) {
            return Err(format!("{case_id} {phase}: duplicate {expected_kind} artifact"));
        }
        let relative_path = reported
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{case_id} {phase}: {expected_kind} path is missing"))?;
        let expected_sha = reported
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{case_id} {phase}: {expected_kind} SHA is missing"))?;
        let expected_length = reported
            .get("byte_length")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{case_id} {phase}: {expected_kind} byte_length is missing"))?;
        if !is_hex_digest(expected_sha, &[64]) {
            return Err(format!("{case_id} {phase}: malformed {expected_kind} digest"));
        }

        let bytes = read_beneath(artifact_root, relative_path)?;
        if bytes.len() as u64 != expected_length || sha256_bytes(&bytes) != expected_sha.to_ascii_lowercase() {
            return Err(format!("{case_id} {phase}: {expected_kind} artifact bytes do not match report digest"));
        }
        let receipt_sha = receipt_item
            .get("sha256")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let receipt_length = receipt_item
            .get("byte_length")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        if receipt_sha != expected_sha.to_ascii_lowercase() || receipt_length != expected_length {
            return Err(format!("{case_id} {phase}: {expected_kind} does not match receipt metadata"));
        }
        let media_type = receipt_item
            .get("media_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let required_media = match expected_kind {
            "screenshot" => "image/png",
            _ => "application/json",
        };
        if media_type != required_media {
            return Err(format!("{case_id} {phase}: unexpected media type for {expected_kind}"));
        }

        match expected_kind {
            "screenshot" => {
                if bytes.len() < 24
                    || !bytes.starts_with(b"\x89PNG\r\n\x1a\n")
                    || bytes.get(12..16) != Some(&b"IHDR"[..])
                {
                    return Err(format!("{case_id} {phase}: screenshot is not a valid PNG header"));
                }
                let png_width = u32::from_be_bytes(bytes[16..20].try_into().expect("fixed PNG width slice"));
                let png_height = u32::from_be_bytes(bytes[20..24].try_into().expect("fixed PNG height slice"));
                let viewport = receipt.get("viewport").ok_or_else(|| {
                    format!("{case_id} {phase}: receipt viewport is missing")
                })?;
                let css_width = viewport.get("width").and_then(Value::as_u64).unwrap_or_default() as f64;
                let css_height = viewport.get("height").and_then(Value::as_u64).unwrap_or_default() as f64;
                let scale = viewport
                    .get("device_scale_factor")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| format!("{case_id} {phase}: invalid viewport scale factor"))?;
                if !scale.is_finite()
                    || !(0.5..=4.0).contains(&scale)
                    || css_width <= 0.0
                    || css_height <= 0.0
                    || png_width == 0
                    || png_height == 0
                    || png_width > 4_096
                    || png_height > 4_096
                    || (png_width as f64 - (css_width * scale).round()).abs() > f64::EPSILON
                    || (png_height as f64 - (css_height * scale).round()).abs() > f64::EPSILON
                {
                    return Err(format!("{case_id} {phase}: PNG dimensions do not match receipt viewport"));
                }
            }
            "accessibility" => {
                serde_json::from_slice::<Value>(&bytes).map_err(|error| {
                    format!("{case_id} {phase}: accessibility artifact JSON is invalid: {error}")
                })?;
            }
            "layout" => {
                let layout: Value = serde_json::from_slice(&bytes).map_err(|error| {
                    format!("{case_id} {phase}: layout artifact JSON is invalid: {error}")
                })?;
                let observed_viewport = layout.get("viewport").ok_or_else(|| {
                    format!("{case_id} {phase}: layout viewport is missing")
                })?;
                let receipt_viewport = receipt.get("viewport").ok_or_else(|| {
                    format!("{case_id} {phase}: receipt viewport is missing")
                })?;
                for key in ["width", "height", "device_scale_factor"] {
                    if observed_viewport.get(key) != receipt_viewport.get(key) {
                        return Err(format!(
                            "{case_id} {phase}: layout and receipt viewport disagree on {key}"
                        ));
                    }
                }
            }
            _ => unreachable!("artifact kind is restricted above"),
        }
    }

    Ok(())
}

fn mutation_killed(candidate: &Value, expected_oracles: &[Value]) -> bool {
    candidate.get("disposition").and_then(Value::as_str) == Some("rejected")
        && string_set(candidate, "observed_oracle_ids")
            .is_ok_and(|observed| expected_oracle_fired(expected_oracles, &observed))
}

fn equivalence_passes(baseline: &Value, candidate: &Value, required: &[Value]) -> bool {
    let baseline_accepted = baseline.get("disposition").and_then(Value::as_str) == Some("accepted");
    let candidate_accepted = candidate.get("disposition").and_then(Value::as_str) == Some("accepted");
    let same_semantic_outcome = baseline.get("semantic_outcome_sha256")
        == candidate.get("semantic_outcome_sha256");
    let verified = string_set(candidate, "verified_invariants").unwrap_or_default();
    baseline_accepted && candidate_accepted && same_semantic_outcome && contains_all(required, &verified)
}

fn validate_qualification_report(
    campaign: &Value,
    report: &Value,
    campaign_sha: &str,
    report_path: &Path,
) -> Result<Value, String> {
    let campaign_id = campaign.get("campaign_id").and_then(Value::as_str)
        .ok_or_else(|| "campaign_id is missing from campaign".to_string())?;
    if report.get("campaign_id").and_then(Value::as_str) != Some(campaign_id) {
        return Err("qualification report campaign_id does not match campaign".to_string());
    }
    if report.get("campaign_sha256").and_then(Value::as_str)
        != Some(campaign_sha)
    {
        return Err("qualification report does not bind the exact campaign file SHA-256".to_string());
    }

    let baseline_commit = report.get("baseline_source_commit").and_then(Value::as_str)
        .ok_or_else(|| "baseline_source_commit is required".to_string())?;
    let environment_sha = report.get("environment_manifest_sha256").and_then(Value::as_str)
        .ok_or_else(|| "environment_manifest_sha256 is required".to_string())?;
    if !is_hex_digest(baseline_commit, &[40, 64]) || !is_hex_digest(environment_sha, &[64]) {
        return Err("report has malformed baseline source or environment digest".to_string());
    }

    let report_parent = report_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let report_dir = report_parent
        .canonicalize()
        .map_err(|error| format!("cannot resolve report directory: {error}"))?;
    let artifact_root_relative = report.get("artifact_root").and_then(Value::as_str)
        .ok_or_else(|| "artifact_root is required".to_string())?;
    validate_relative_repo_path(artifact_root_relative)?;
    let artifact_root_candidate = report_dir.join(artifact_root_relative);
    let artifact_root = artifact_root_candidate.canonicalize()
        .map_err(|error| format!("cannot resolve artifact_root: {error}"))?;
    if !artifact_root.starts_with(&report_dir) {
        return Err("artifact_root escapes the report directory".to_string());
    }

    let cases = report.get("case_results").and_then(Value::as_array)
        .ok_or_else(|| "case_results must be an array".to_string())?;
    let mut case_map = BTreeMap::<String, &Value>::new();
    for case in cases {
        let id = case.get("case_id").and_then(Value::as_str)
            .ok_or_else(|| "every case result needs case_id".to_string())?;
        if case_map.insert(id.to_string(), case).is_some() {
            return Err(format!("duplicate qualification case result: {id}"));
        }
    }

    let mut mutation_total = 0usize;
    let mut mutation_kills = 0usize;
    let mut critical_total = 0usize;
    let mut critical_kills = 0usize;
    let mut high_total = 0usize;
    let mut high_kills = 0usize;
    let mut category_counts = BTreeMap::<String, (usize, usize)>::new();
    let mut security_total = 0usize;
    let mut security_kills = 0usize;
    let mut evidence_total = 0usize;
    let mut evidence_kills = 0usize;
    let mut positive_total = 0usize;
    let mut positive_false_rejects = 0usize;
    let mut positive_passes = 0usize;
    let mut metamorphic_total = 0usize;
    let mut metamorphic_passes = 0usize;
    let mut adjudicated_cases = Vec::new();
    let mut run_ids = BTreeSet::new();

    let mutations = campaign.get("mutations").and_then(Value::as_array)
        .ok_or_else(|| "campaign.mutations is missing".to_string())?;
    for definition in mutations {
        let id = definition.get("id").and_then(Value::as_str).unwrap_or_default();
        let case = case_map.remove(id).ok_or_else(|| format!("missing mutation result {id}"))?;
        if case.get("case_kind").and_then(Value::as_str) != Some("mutation") {
            return Err(format!("case {id} has the wrong case_kind"));
        }
        let task_id = case.get("task_id").and_then(Value::as_str).unwrap_or_default();
        let fixture_id = case.get("fixture_id").and_then(Value::as_str).unwrap_or_default();
        let baseline = case.get("baseline").ok_or_else(|| format!("{id}: baseline run missing"))?;
        let candidate = case.get("candidate").ok_or_else(|| format!("{id}: candidate run missing"))?;
        let baseline_run_id = baseline.get("run_id").and_then(Value::as_str).unwrap_or_default();
        let candidate_run_id = candidate.get("run_id").and_then(Value::as_str).unwrap_or_default();
        if baseline_run_id == candidate_run_id || !run_ids.insert(baseline_run_id.to_string())
            || !run_ids.insert(candidate_run_id.to_string())
        {
            return Err(format!("{id}: run IDs must be globally unique and baseline/candidate distinct"));
        }
        if baseline.get("disposition").and_then(Value::as_str) != Some("accepted") {
            return Err(format!("{id}: baseline control must be accepted before mutation"));
        }
        for (run, phase) in [(baseline, "baseline"), (candidate, "candidate")] {
            if run.get("fixture_sha256") != baseline.get("fixture_sha256")
                || run.get("environment_manifest_sha256") != baseline.get("environment_manifest_sha256")
                || run.get("environment_manifest_sha256").and_then(Value::as_str) != Some(environment_sha)
            {
                return Err(format!("{id}: paired runs must use identical fixture and environment digests"));
            }
            if run.get("fixture_sha256").and_then(Value::as_str)
                .is_none_or(|value| !is_hex_digest(value, &[64]))
            {
                return Err(format!("{id}: malformed fixture digest"));
            }
            validate_evidence_run(
                run,
                &artifact_root,
                id,
                phase,
                task_id,
                fixture_id,
                environment_sha,
            )?;
        }
        if baseline.get("source_commit").and_then(Value::as_str) != Some(baseline_commit) {
            return Err(format!("{id}: baseline source commit does not match report"));
        }

        let expected = definition.get("expected_oracles").and_then(Value::as_array)
            .ok_or_else(|| format!("{id}: expected_oracles missing"))?;
        let killed = mutation_killed(candidate, expected);
        mutation_total += 1;
        if killed { mutation_kills += 1; }
        let category = definition.get("category").and_then(Value::as_str).unwrap_or("unknown");
        let severity = definition.get("severity").and_then(Value::as_str).unwrap_or("unknown");
        let category_count = category_counts.entry(category.to_string()).or_insert((0, 0));
        category_count.1 += 1;
        if killed { category_count.0 += 1; }
        if severity == "critical" {
            critical_total += 1;
            if killed { critical_kills += 1; }
        }
        if severity == "high" {
            high_total += 1;
            if killed { high_kills += 1; }
        }
        if category == "security" {
            security_total += 1;
            if killed { security_kills += 1; }
        }
        if category == "evidence_integrity" {
            evidence_total += 1;
            if killed { evidence_kills += 1; }
        }
        adjudicated_cases.push(serde_json::json!({
            "case_id": id, "kind": "mutation", "expected": "reject",
            "observed": candidate.get("disposition"), "killed": killed
        }));
    }

    let positives = campaign.get("positive_controls").and_then(Value::as_array)
        .ok_or_else(|| "campaign.positive_controls is missing".to_string())?;
    for definition in positives {
        let id = definition.get("id").and_then(Value::as_str).unwrap_or_default();
        let case = case_map.remove(id).ok_or_else(|| format!("missing positive-control result {id}"))?;
        if case.get("case_kind").and_then(Value::as_str) != Some("positive_control") {
            return Err(format!("case {id} has the wrong case_kind"));
        }
        let task_id = case.get("task_id").and_then(Value::as_str).unwrap_or_default();
        let fixture_id = case.get("fixture_id").and_then(Value::as_str).unwrap_or_default();
        let baseline = case.get("baseline").ok_or_else(|| format!("{id}: baseline run missing"))?;
        let candidate = case.get("candidate").ok_or_else(|| format!("{id}: candidate run missing"))?;
        if baseline.get("source_commit").and_then(Value::as_str) != Some(baseline_commit) {
            return Err(format!("{id}: baseline source commit does not match report"));
        }
        for run in [baseline, candidate] {
            let run_id = run.get("run_id").and_then(Value::as_str).unwrap_or_default();
            if !run_ids.insert(run_id.to_string()) {
                return Err(format!("{id}: duplicate run ID"));
            }
            if run.get("fixture_sha256") != baseline.get("fixture_sha256")
                || run.get("environment_manifest_sha256") != baseline.get("environment_manifest_sha256")
                || run.get("environment_manifest_sha256").and_then(Value::as_str) != Some(environment_sha)
            {
                return Err(format!("{id}: paired runs must use identical fixture and environment digests"));
            }
            validate_evidence_run(run, &artifact_root, id, "positive",
                task_id, fixture_id, environment_sha)?;
        }
        if baseline.get("disposition").and_then(Value::as_str) != Some("accepted") {
            return Err(format!("{id}: positive-control baseline must be accepted"));
        }
        positive_total += 1;
        let candidate_disposition = candidate.get("disposition").and_then(Value::as_str);
        if candidate_disposition == Some("rejected") {
            positive_false_rejects += 1;
        }
        let invariants = definition.get("invariants").and_then(Value::as_array)
            .ok_or_else(|| format!("{id}: positive-control invariants missing"))?;
        let passed = equivalence_passes(baseline, candidate, invariants);
        if passed { positive_passes += 1; }
        adjudicated_cases.push(serde_json::json!({
            "case_id": id, "kind": "positive_control", "expected": "accept",
            "observed": candidate.get("disposition"), "passed": passed
        }));
    }

    let relations = campaign.get("metamorphic_relations").and_then(Value::as_array)
        .ok_or_else(|| "campaign.metamorphic_relations is missing".to_string())?;
    for definition in relations {
        let id = definition.get("id").and_then(Value::as_str).unwrap_or_default();
        let case = case_map.remove(id).ok_or_else(|| format!("missing metamorphic result {id}"))?;
        if case.get("case_kind").and_then(Value::as_str) != Some("metamorphic") {
            return Err(format!("case {id} has the wrong case_kind"));
        }
        let task_id = case.get("task_id").and_then(Value::as_str).unwrap_or_default();
        let fixture_id = case.get("fixture_id").and_then(Value::as_str).unwrap_or_default();
        let baseline = case.get("baseline").ok_or_else(|| format!("{id}: baseline run missing"))?;
        let candidate = case.get("candidate").ok_or_else(|| format!("{id}: candidate run missing"))?;
        if baseline.get("source_commit").and_then(Value::as_str) != Some(baseline_commit) {
            return Err(format!("{id}: baseline source commit does not match report"));
        }
        for run in [baseline, candidate] {
            let run_id = run.get("run_id").and_then(Value::as_str).unwrap_or_default();
            if !run_ids.insert(run_id.to_string()) {
                return Err(format!("{id}: duplicate run ID"));
            }
            if run.get("fixture_sha256") != baseline.get("fixture_sha256")
                || run.get("environment_manifest_sha256") != baseline.get("environment_manifest_sha256")
                || run.get("environment_manifest_sha256").and_then(Value::as_str) != Some(environment_sha)
            {
                return Err(format!("{id}: paired runs must use identical fixture and environment digests"));
            }
            validate_evidence_run(run, &artifact_root, id, "metamorphic",
                task_id, fixture_id, environment_sha)?;
        }
        let required_oracle = definition.get("oracle").and_then(Value::as_str)
            .ok_or_else(|| format!("{id}: metamorphic oracle is missing"))?;
        let required = vec![Value::String(required_oracle.to_string())];
        let passed = equivalence_passes(baseline, candidate, &required);
        metamorphic_total += 1;
        if passed { metamorphic_passes += 1; }
        adjudicated_cases.push(serde_json::json!({
            "case_id": id, "kind": "metamorphic", "expected": "equivalent",
            "observed": candidate.get("disposition"), "passed": passed
        }));
    }

    if !case_map.is_empty() {
        return Err(format!("qualification report contains unknown cases: {}", case_map.keys().cloned().collect::<Vec<_>>().join(", ")));
    }
    let expected_total = mutations.len() + positives.len() + relations.len();
    if run_ids.len() != expected_total * 2 {
        return Err("qualification report does not contain two unique runs for every case".to_string());
    }

    let mut categories = serde_json::Map::new();
    for (name, (successes, trials)) in category_counts {
        categories.insert(name, rate_summary(successes, trials));
    }
    let overall_rate = mutation_kills as f64 / mutation_total.max(1) as f64;
    let high_rate = high_kills as f64 / high_total.max(1) as f64;
    let critical_rate = critical_kills as f64 / critical_total.max(1) as f64;
    let security_rate = security_kills as f64 / security_total.max(1) as f64;
    let evidence_rate = evidence_kills as f64 / evidence_total.max(1) as f64;
    let false_reject_rate = positive_false_rejects as f64 / positive_total.max(1) as f64;
    let qualified = critical_total > 0
        && security_total > 0
        && evidence_total > 0
        && critical_rate >= 1.0
        && security_rate >= 1.0
        && evidence_rate >= 1.0
        && high_rate >= 0.95
        && overall_rate >= 0.90
        && positive_false_rejects == 0
        && positive_passes == positive_total
        && metamorphic_passes == metamorphic_total;

    Ok(serde_json::json!({
        "schema": "sfeb-qualification-summary-v1",
        "report_id": report.get("report_id"),
        "campaign_id": campaign_id,
        "campaign_sha256": campaign_sha,
        "status": if qualified { "qualified" } else { "not_qualified" },
        "mutation_kill_rates": {
            "overall": rate_summary(mutation_kills, mutation_total),
            "critical": rate_summary(critical_kills, critical_total),
            "high_severity": rate_summary(high_kills, high_total),
            "security": rate_summary(security_kills, security_total),
            "evidence_integrity": rate_summary(evidence_kills, evidence_total),
            "by_category": Value::Object(categories)
        },
        "positive_controls": {
            "false_rejections": positive_false_rejects,
            "false_reject_rate": rate_summary(positive_false_rejects, positive_total),
            "accepted_with_all_invariants": rate_summary(positive_passes, positive_total)
        },
        "metamorphic_relations": rate_summary(metamorphic_passes, metamorphic_total),
        "case_results": adjudicated_cases,
        "important_limitations": [
            "Wilson intervals quantify binomial uncertainty for this finite campaign; they do not estimate all real-world UI defects.",
            "This adjudicator verifies report, receipt and artifact integrity; it does not execute the browser, mutate source, or independently establish that each oracle is semantically adequate.",
            "A qualified result applies only to the exact campaign SHA-256, baseline revision, runner revision and environment manifest."
        ]
    }))
}

fn run_qualification_report(
    report_schema_path: &str,
    campaign_path: &str,
    report_path: &str,
) -> Result<(), String> {
    let campaign_schema = read_json(DEFAULT_MUTATION_SCHEMA)?;
    let campaign_bytes = fs::read(campaign_path)
        .map_err(|error| format!("failed to read campaign {campaign_path}: {error}"))?;
    let campaign: Value = serde_json::from_slice(&campaign_bytes)
        .map_err(|error| format!("failed to parse campaign {campaign_path}: {error}"))?;
    validate_mutation_campaign(&campaign_schema, &campaign)?;
    let campaign_sha = sha256_bytes(&campaign_bytes);

    let report_schema = read_json(report_schema_path)?;
    let report = read_json(report_path)?;
    let validator = validator_for(&report_schema)
        .map_err(|error| format!("invalid qualification report schema: {error}"))?;
    let errors: Vec<String> = validator
        .iter_errors(&report)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect();
    if !errors.is_empty() {
        return Err(format!(
            "qualification report does not satisfy its schema:\n{}",
            errors.join("\n")
        ));
    }

    let summary = validate_qualification_report(
        &campaign,
        &report,
        &campaign_sha,
        Path::new(report_path),
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)
        .map_err(|error| format!("failed to serialize qualification summary: {error}"))?);
    if summary.get("status").and_then(Value::as_str) != Some("qualified") {
        return Err("qualification report is valid but the campaign did not meet its quality gates".to_string());
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "Usage:\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- [CONTRACT_SCHEMA [CONTRACT_JSON [APP_SOURCE.rs]]]\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --task [TASK_SCHEMA [TASK_JSON]]\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --patterns [PATTERN_SCHEMA [CORPUS_JSON [REPOSITORY_ROOT]]]\n  cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --retrieval [RETRIEVAL_SCHEMA [QUERY_SET_JSON [PATTERN_CORPUS_JSON]]]\n\nThe default mode validates the Leptos interface contract and exact source anchors. --task validates the task schema, required fixture regions and canonical fixture digest. --patterns validates the Leptos knowledge schema and exact Git blob provenance for every source example. --retrieval validates positive/hard-negative queries and their gold IDs against the versioned pattern corpus. --mutations validates the oracle-qualification campaign definition only; it does not execute the campaign. --qualification-report validates the complete paired execution report, receipts and artifact bytes, then computes Wilson intervals and hard-gate qualification."
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
    if args.first().is_some_and(|arg| arg == "--mutations") {
        if args.len() > 3 {
            return Err("--mutations accepts at most a schema path and campaign path".to_string());
        }
        return run_mutations(
            args.get(1).map(String::as_str).unwrap_or(DEFAULT_MUTATION_SCHEMA),
            args.get(2).map(String::as_str).unwrap_or(DEFAULT_MUTATION_CAMPAIGN),
        );
    }
    if args.first().is_some_and(|arg| arg == "--qualification-report") {
        if args.len() > 4 {
            return Err("--qualification-report accepts report schema, campaign JSON and report JSON paths".to_string());
        }
        return run_qualification_report(
            args.get(1).map(String::as_str).unwrap_or(DEFAULT_QUALIFICATION_REPORT_SCHEMA),
            args.get(2).map(String::as_str).unwrap_or(DEFAULT_MUTATION_CAMPAIGN),
            args.get(3).map(String::as_str).unwrap_or("qualification-report.json"),
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
        assert_eq!(positive, 34);
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
        assert_eq!(patterns, 17);
        assert_eq!(sources, 7);
    }

    #[test]
    fn pattern_validator_rejects_stale_related_source_blob() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/pattern-v1.schema.json"
        ))
        .expect("checked-in Leptos pattern schema is valid JSON");
        let mut corpus: Value = serde_json::from_str(include_str!(
            "../../knowledge/leptos-0.8/patterns-v0.json"
        ))
        .expect("checked-in Leptos pattern corpus is valid JSON");
        let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");

        corpus["patterns"][9]["source"]["related_sources"][0]["blob_sha"] =
            Value::String("0".repeat(40));
        assert!(validate_pattern_corpus(&schema, &corpus, &repository_root).is_err());
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
    #[test]
    fn checked_in_mutation_campaign_matches_schema_and_qualification_gates() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../qualification/mutation-campaign-v1.schema.json"
        ))
        .expect("checked-in mutation campaign schema is valid JSON");
        let campaign: Value = serde_json::from_str(include_str!(
            "../../qualification/mutation-campaign-v1.json"
        ))
        .expect("checked-in mutation campaign is valid JSON");

        let (mutations, positives, relations) =
            validate_mutation_campaign(&schema, &campaign)
                .expect("mutation campaign must satisfy its schema and fail-closed gates");
        assert_eq!(mutations, 20);
        assert_eq!(positives, 5);
        assert_eq!(relations, 5);
    }

    #[test]
    fn mutation_campaign_rejects_duplicate_mutation_ids() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../qualification/mutation-campaign-v1.schema.json"
        ))
        .expect("checked-in mutation campaign schema is valid JSON");
        let mut campaign: Value = serde_json::from_str(include_str!(
            "../../qualification/mutation-campaign-v1.json"
        ))
        .expect("checked-in mutation campaign is valid JSON");
        let duplicate_id = campaign["mutations"][0]["id"].clone();
        campaign["mutations"][1]["id"] = duplicate_id;

        assert!(validate_mutation_campaign(&schema, &campaign).is_err());
    }

    #[test]
    fn mutation_campaign_cannot_lower_aggregate_or_positive_control_gates() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../qualification/mutation-campaign-v1.schema.json"
        ))
        .expect("checked-in mutation campaign schema is valid JSON");
        let mut campaign: Value = serde_json::from_str(include_str!(
            "../../qualification/mutation-campaign-v1.json"
        ))
        .expect("checked-in mutation campaign is valid JSON");

        campaign["acceptance_targets"]["overall_mutation_kill_rate_min"] =
            Value::from(0.89);
        assert!(validate_mutation_campaign(&schema, &campaign).is_err());

        campaign["acceptance_targets"]["overall_mutation_kill_rate_min"] =
            Value::from(0.90);
        campaign["acceptance_targets"]["positive_control_false_reject_rate_max"] =
            Value::from(0.01);
        assert!(validate_mutation_campaign(&schema, &campaign).is_err());
    }

    #[test]
    fn qualification_report_schema_is_valid_and_has_paired_case_contract() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../qualification/qualification-report-v1.schema.json"
        ))
        .expect("qualification report schema must be valid JSON");
        validator_for(&schema).expect("qualification report schema must compile");

        let case = &schema["$defs"]["case"];
        assert!(case["required"].as_array().unwrap().iter().any(|value| value == "baseline"));
        assert!(case["required"].as_array().unwrap().iter().any(|value| value == "candidate"));
        assert_eq!(schema["properties"]["case_results"]["minItems"], 30);
    }

    #[test]
    fn qualification_report_rejects_campaign_hash_mismatch_before_artifact_access() {
        let campaign: Value = serde_json::from_str(include_str!(
            "../../qualification/mutation-campaign-v1.json"
        ))
        .expect("campaign must be valid JSON");
        let report = serde_json::json!({
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": "0".repeat(64)
        });

        let error = validate_qualification_report(
            &campaign,
            &report,
            &"1".repeat(64),
            Path::new("report.json"),
        )
        .expect_err("report with wrong campaign digest must fail closed");
        assert!(error.contains("exact campaign file SHA-256"));
    }

    #[test]
    fn wilson_interval_is_bounded_and_nonzero_for_small_perfect_samples() {
        let interval = wilson_interval(5, 5);
        assert!(interval["lower_95"].as_f64().unwrap() > 0.0);
        assert_eq!(interval["upper_95"], 1.0);
        assert_eq!(wilson_interval(0, 0), Value::Null);
    }

    #[test]
    fn mutation_is_killed_only_by_an_expected_oracle() {
        let candidate = serde_json::json!({
            "disposition": "rejected",
            "observed_oracle_ids": ["layout.overflow"]
        });
        assert!(mutation_killed(
            &candidate,
            &[Value::String("layout.overflow".into()), Value::String("keyboard.route".into())]
        ));
        assert!(!mutation_killed(
            &candidate,
            &[Value::String("receipt.integrity".into())]
        ));
        let accepted = serde_json::json!({
            "disposition": "accepted",
            "observed_oracle_ids": ["layout.overflow"]
        });
        assert!(!mutation_killed(
            &accepted,
            &[Value::String("layout.overflow".into())]
        ));
    }

    #[test]
    fn positive_and_metamorphic_controls_require_semantic_equivalence() {
        let baseline = serde_json::json!({
            "disposition": "accepted",
            "semantic_outcome_sha256": "a".repeat(64)
        });
        let candidate = serde_json::json!({
            "disposition": "accepted",
            "semantic_outcome_sha256": "a".repeat(64),
            "verified_invariants": ["keyboard.journey", "accessible.name"]
        });
        assert!(equivalence_passes(
            &baseline,
            &candidate,
            &[Value::String("keyboard.journey".into())]
        ));
        assert!(!equivalence_passes(
            &baseline,
            &candidate,
            &[Value::String("missing.invariant".into())]
        ));
        let changed = serde_json::json!({
            "disposition": "accepted",
            "semantic_outcome_sha256": "b".repeat(64),
            "verified_invariants": ["keyboard.journey"]
        });
        assert!(!equivalence_passes(
            &baseline,
            &changed,
            &[Value::String("keyboard.journey".into())]
        ));
    }

    #[test]
    fn evidence_path_validation_rejects_traversal_and_absolute_paths() {
        assert!(validate_relative_repo_path("../outside.json").is_err());
        assert!(validate_relative_repo_path("/tmp/outside.json").is_err());
        assert!(validate_relative_repo_path("artifacts/../outside.json").is_err());
        assert!(validate_relative_repo_path("artifacts/receipt.json").is_ok());
    }

    #[test]
    fn mutation_campaign_cannot_lower_critical_hard_gates() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../qualification/mutation-campaign-v1.schema.json"
        ))
        .expect("checked-in mutation campaign schema is valid JSON");
        let mut campaign: Value = serde_json::from_str(include_str!(
            "../../qualification/mutation-campaign-v1.json"
        ))
        .expect("checked-in mutation campaign is valid JSON");
        campaign["acceptance_targets"]["security_mutation_kill_rate_min"] =
            Value::from(0.99);

        assert!(validate_mutation_campaign(&schema, &campaign).is_err());
    }

}
