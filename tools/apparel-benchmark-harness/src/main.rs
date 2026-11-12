use jsonschema::draft202012;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
const SCHEMA_PATH: &str =
    "docs/research/fixtures/apparel_engineering_thread_v0_1/apparel_thread.schema.json";
const INSTANCE_PATH: &str =
    "docs/research/fixtures/apparel_engineering_thread_v0_1/apparel_thread.example.json";
const SEMANTIC_PATH: &str =
    "docs/research/fixtures/apparel_engineering_thread_v0_1/semantic_cases.example.json";
const SOURCES_PATH: &str =
    "docs/research/fixtures/apparel_engineering_thread_v0_1/benchmark_sources_2026-11-12.json";

#[derive(Debug)]
struct Inputs {
    schema: PathBuf,
    instance: PathBuf,
    semantic_cases: PathBuf,
    source_manifest: PathBuf,
    expected_head: String,
    output: PathBuf,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_json(path: &Path) -> Result<(Vec<u8>, Value), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let value: Value = serde_json::from_slice(&bytes)?;
    Ok((bytes, value))
}

fn get_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn array<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn unique_ids(
    items: &[Value],
    field: &str,
    path: &str,
    errors: &mut Vec<String>,
) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        match get_str(item, field) {
            Some(value) if !value.trim().is_empty() => {
                if !ids.insert(value.to_owned()) {
                    errors.push(format!("{path}[{index}]: duplicate {field} {value}"));
                }
            }
            _ => errors.push(format!("{path}[{index}]: missing or empty {field}")),
        }
    }
    ids
}

fn reject_external_schema_references(value: &Value, path: &str) -> Result<(), String> {
    match value {
        Value::Object(object) => {
            for key in ["$ref", "$dynamicRef", "$recursiveRef"] {
                if object.contains_key(key) {
                    return Err(format!(
                        "{path}: {key} is disallowed in the self-contained apparel schema"
                    ));
                }
            }
            for (key, child) in object {
                reject_external_schema_references(child, &format!("{path}/{key}"))?;
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                reject_external_schema_references(child, &format!("{path}/{index}"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_schema_document(schema: &Value) -> Result<(), String> {
    if get_str(schema, "$schema") != Some(DIALECT) {
        return Err(format!(
            "schema must declare Draft 2020-12 dialect {DIALECT}"
        ));
    }
    reject_external_schema_references(schema, "$")?;
    if !draft202012::meta::is_valid(schema) {
        return Err(
            "schema document does not validate against the Draft 2020-12 meta-schema".into(),
        );
    }
    draft202012::new(schema)
        .map(|_| ())
        .map_err(|error| format!("Draft 2020-12 validator construction failed: {error}"))
}

fn validate_instance(schema: &Value, instance: &Value) -> Result<(), Vec<String>> {
    let validator = match draft202012::new(schema) {
        Ok(validator) => validator,
        Err(error) => return Err(vec![format!("validator construction failed: {error}")]),
    };
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| format!("{}: {}", error.instance_path(), error))
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_semantics(instance: &Value) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    let materials = array(instance, "materials");
    let material_ids = unique_ids(materials, "material_id", "materials", &mut errors);
    let pattern = instance.get("pattern").unwrap_or(&Value::Null);
    let pieces = array(pattern, "pieces");
    let _piece_ids = unique_ids(pieces, "piece_id", "pattern.pieces", &mut errors);
    let operations = array(instance, "operations");
    let _operation_ids = unique_ids(operations, "operation_id", "operations", &mut errors);
    let projections = array(instance, "geometry_projections");
    let projection_ids = unique_ids(
        projections,
        "projection_id",
        "geometry_projections",
        &mut errors,
    );
    let physical_tests = array(instance, "physical_tests");
    let test_ids = unique_ids(physical_tests, "test_id", "physical_tests", &mut errors);
    let claims = array(instance, "claims");
    let _claim_ids = unique_ids(claims, "claim_id", "claims", &mut errors);

    let design_revision = instance.pointer("/design/revision").and_then(Value::as_str);
    let pattern_revision = get_str(pattern, "revision");
    if design_revision.is_some() && pattern_revision != design_revision {
        errors.push("pattern revision must equal the current design revision".into());
    }

    for (index, material) in materials.iter().enumerate() {
        let properties = array(material, "properties");
        let composition_is_measured = get_str(material, "composition_state") == Some("measured");
        let measured_composition_has_evidence = properties.iter().any(|property| {
            get_str(property, "property_id") == Some("composition")
                && get_str(property, "status") == Some("measured")
                && property.get("value").is_some_and(|value| !value.is_null())
                && get_str(property, "unit").is_some_and(|value| !value.trim().is_empty())
                && get_str(property, "evidence_ref").is_some_and(|value| !value.trim().is_empty())
        });
        if composition_is_measured && !measured_composition_has_evidence {
            errors.push(format!(
                "materials[{index}]: measured composition requires a measured composition property with value, unit and evidence_ref"
            ));
        }
        for (property_index, property) in properties.iter().enumerate() {
            if get_str(property, "status") == Some("measured")
                && (property.get("value").is_none_or(Value::is_null)
                    || get_str(property, "unit").is_none_or(|value| value.trim().is_empty())
                    || get_str(property, "evidence_ref").is_none_or(|value| value.trim().is_empty()))
            {
                errors.push(format!(
                    "materials[{index}].properties[{property_index}]: measured property requires value, unit and evidence_ref"
                ));
            }
        }
    }

    let mut operation_sequences = BTreeSet::new();
    for (index, operation) in operations.iter().enumerate() {
        if let Some(sequence) = operation.get("sequence").and_then(Value::as_i64)
            && !operation_sequences.insert(sequence)
        {
            errors.push(format!(
                "operations[{index}]: duplicate operation sequence {sequence}"
            ));
        }
    }

    for (index, item) in array(instance, "bom").iter().enumerate() {
        let material_id = get_str(item, "material_id").unwrap_or_default();
        if !material_ids.contains(material_id) {
            errors.push(format!(
                "bom[{index}]: unresolved material_id {material_id}"
            ));
        }
        if get_str(item, "lot_binding") == Some("bound")
            && item.get("quantity").is_none_or(Value::is_null)
        {
            errors.push(format!(
                "bom[{index}]: a bound lot requires a declared quantity"
            ));
        }
    }

    for (index, projection) in projections.iter().enumerate() {
        let projection_revision = get_str(projection, "source_revision");
        let status = get_str(projection, "status");
        if design_revision.is_some()
            && projection_revision != design_revision
            && matches!(status, Some("available" | "incomplete"))
        {
            errors.push(format!(
                "geometry_projections[{index}]: projection revision differs from current design revision but is not marked stale/failed/not_performed"
            ));
        }
    }

    for (index, test) in physical_tests.iter().enumerate() {
        if get_str(test, "status") == Some("measured") {
            if test.get("result").is_none_or(Value::is_null) {
                errors.push(format!(
                    "physical_tests[{index}]: measured result cannot be null"
                ));
            }
            if get_str(test, "evidence_ref").is_none_or(str::is_empty) {
                errors.push(format!(
                    "physical_tests[{index}]: measured result requires evidence_ref"
                ));
            }
        }
    }

    let mut known_evidence = projection_ids;
    known_evidence.extend(test_ids);
    for material in materials {
        for property in array(material, "properties") {
            if let Some(reference) = get_str(property, "evidence_ref")
                && !reference.trim().is_empty()
            {
                known_evidence.insert(reference.to_owned());
            }
        }
    }
    for test in physical_tests {
        if let Some(reference) = get_str(test, "evidence_ref")
            && !reference.trim().is_empty()
        {
            known_evidence.insert(reference.to_owned());
        }
    }
    for operation in operations {
        if let Some(reference) = get_str(operation, "actual_event_ref")
            && !reference.trim().is_empty()
        {
            known_evidence.insert(reference.to_owned());
        }
    }
    known_evidence.insert("FIXTURE-SELF-CHECK-001".into());

    for (index, claim) in claims.iter().enumerate() {
        let references = claim
            .get("evidence_refs")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for reference in references {
            if let Some(reference) = reference.as_str()
                && !known_evidence.contains(reference)
            {
                errors.push(format!(
                    "claims[{index}]: unresolved evidence reference {reference}"
                ));
            }
        }

        if get_str(claim, "claim_class") == Some("physical_test")
            && get_str(claim, "disposition") == Some("supported_within_scope")
        {
            let supported_by_measured_test = references.iter().any(|reference| {
                let Some(reference) = reference.as_str() else {
                    return false;
                };
                physical_tests.iter().any(|test| {
                    (get_str(test, "test_id") == Some(reference)
                        || get_str(test, "evidence_ref") == Some(reference))
                        && get_str(test, "status") == Some("measured")
                        && get_str(test, "evidence_ref")
                            .is_some_and(|value| !value.trim().is_empty())
                        && test.get("result").is_some_and(|value| !value.is_null())
                })
            });
            if !supported_by_measured_test {
                errors.push(format!(
                    "claims[{index}]: physical_test claim cannot be supported without a measured, evidence-bound test"
                ));
            }
        }
    }

    if get_str(instance, "fixture_class") == Some("synthetic") {
        let physical_qualification = get_str(
            instance
                .pointer("/expected_disposition")
                .unwrap_or(&Value::Null),
            "physical_qualification",
        );
        if physical_qualification != Some("not_qualified") {
            errors.push("synthetic fixture must remain physically unqualified".into());
        }
    }

    if get_str(
        instance.pointer("/product").unwrap_or(&Value::Null),
        "article_scope",
    ) == Some("single_prototype")
    {
        for (index, claim) in claims.iter().enumerate() {
            let statement = get_str(claim, "statement")
                .unwrap_or_default()
                .to_ascii_lowercase();
            if get_str(claim, "disposition") == Some("supported_within_scope")
                && (statement.contains("all sizes")
                    || statement.contains("production batches")
                    || statement.contains("production process"))
            {
                errors.push(format!(
                    "claims[{index}]: production-wide capability exceeds single_prototype scope"
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_semantic_corpus(corpus: &Value) -> Result<(usize, usize), Vec<String>> {
    let mut errors = Vec::new();
    let cases = array(corpus, "cases");
    let mut ids = BTreeSet::new();
    let mut negatives = 0usize;
    let mut positives = 0usize;

    for (index, case) in cases.iter().enumerate() {
        let id = get_str(case, "case_id").unwrap_or_default();
        if id.trim().is_empty() || !ids.insert(id.to_owned()) {
            errors.push(format!(
                "semantic_cases[{index}]: missing or duplicate case_id"
            ));
        }
        if get_str(case, "mutation").is_none_or(str::is_empty)
            || get_str(case, "invariant").is_none_or(str::is_empty)
        {
            errors.push(format!(
                "semantic_cases[{index}]: mutation and invariant are required"
            ));
        }
        match get_str(case, "case_type") {
            Some("negative") => negatives += 1,
            Some("positive") => positives += 1,
            other => errors.push(format!(
                "semantic_cases[{index}]: unsupported case_type {other:?}"
            )),
        }
        if case
            .get("expected_reason_codes")
            .and_then(Value::as_array)
            .is_none()
        {
            errors.push(format!(
                "semantic_cases[{index}]: expected_reason_codes must be an array"
            ));
        }
    }

    if get_str(corpus, "fixture_class") != Some("synthetic") {
        errors.push("semantic corpus must remain explicitly synthetic".into());
    }

    let coverage = corpus.get("execution_coverage").unwrap_or(&Value::Null);
    let mut declared_coverage = BTreeSet::new();
    for field in ["executable_case_ids", "not_yet_executed_case_ids"] {
        let Some(items) = coverage.get(field).and_then(Value::as_array) else {
            errors.push(format!("execution_coverage.{field} must be an array"));
            continue;
        };
        for item in items {
            let Some(id) = item.as_str() else {
                errors.push(format!(
                    "execution_coverage.{field} entries must be strings"
                ));
                continue;
            };
            if !ids.contains(id) {
                errors.push(format!(
                    "execution_coverage.{field} contains unknown case {id}"
                ));
            }
            if !declared_coverage.insert(id.to_owned()) {
                errors.push(format!("execution_coverage duplicates case {id}"));
            }
        }
    }
    if declared_coverage.len() != ids.len() {
        errors.push("execution_coverage must classify every semantic case exactly once".into());
    }

    if errors.is_empty() {
        Ok((negatives, positives))
    } else {
        Err(errors)
    }
}

fn validate_source_manifest(manifest: &Value) -> Result<usize, Vec<String>> {
    let mut errors = Vec::new();
    let sources = array(manifest, "sources");
    let ids = unique_ids(sources, "source_id", "sources", &mut errors);
    if ids.len() != sources.len() {
        errors.push("benchmark source IDs must be unique".into());
    }
    if get_str(manifest, "revision").is_none_or(str::is_empty) {
        errors.push("benchmark manifest revision is required".into());
    }

    let nir = sources
        .iter()
        .find(|source| get_str(source, "source_id") == Some("NIST-NIR-SORT"));
    if let Some(nir) = nir {
        let total = nir
            .pointer("/public_scope/specimens_total")
            .and_then(Value::as_u64);
        let group_sum = nir
            .pointer("/public_scope/groups")
            .and_then(Value::as_array)
            .map(|groups| {
                groups
                    .iter()
                    .filter_map(|group| group.get("count").and_then(Value::as_u64))
                    .sum::<u64>()
            });
        if total.is_some() && group_sum.is_some() && total != group_sum {
            errors.push("NIST NIR-SORT group counts do not sum to specimens_total".into());
        }
    }

    if let Some(texsort) = sources
        .iter()
        .find(|source| get_str(source, "source_id") == Some("TEXSORTML-2026"))
    {
        let garments = texsort
            .pointer("/public_scope/garments")
            .and_then(Value::as_u64);
        let stages = texsort
            .pointer("/public_scope/presentation_stages")
            .and_then(Value::as_u64);
        let frames = texsort
            .pointer("/public_scope/annotated_frames")
            .and_then(Value::as_u64);
        if let (Some(garments), Some(stages), Some(frames)) = (garments, stages, frames)
            && garments * stages != frames
        {
            errors.push("TexSortML garment × stage count does not equal annotated_frames".into());
        }
    }

    if errors.is_empty() {
        Ok(sources.len())
    } else {
        Err(errors)
    }
}

fn current_git_head() -> Result<String, Box<dyn Error>> {
    let output = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    if !output.status.success() {
        return Err("git rev-parse HEAD failed".into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn parse_args() -> Result<Inputs, Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let allowed = [
        "--schema",
        "--instance",
        "--semantic-cases",
        "--source-manifest",
        "--expected-head",
        "--out",
    ];
    let mut options = std::collections::BTreeMap::new();
    let mut index = 0usize;
    while index < args.len() {
        let key = args[index].as_str();
        if !allowed.contains(&key) {
            return Err(format!("unknown argument {key}").into());
        }
        if index + 1 >= args.len() || args[index + 1].starts_with("--") {
            return Err(format!("missing value for {key}").into());
        }
        if options
            .insert(key.to_owned(), args[index + 1].clone())
            .is_some()
        {
            return Err(format!("duplicate argument {key}").into());
        }
        index += 2;
    }

    let resolve = |key: &str, default: &str| {
        options
            .get(key)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(default))
    };
    let expected_head = options
        .get("--expected-head")
        .cloned()
        .ok_or("--expected-head is required")?;
    if expected_head.len() != 40 || !expected_head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("--expected-head must be a 40-character commit SHA".into());
    }
    Ok(Inputs {
        schema: resolve("--schema", SCHEMA_PATH),
        instance: resolve("--instance", INSTANCE_PATH),
        semantic_cases: resolve("--semantic-cases", SEMANTIC_PATH),
        source_manifest: resolve("--source-manifest", SOURCES_PATH),
        expected_head,
        output: resolve("--out", "/tmp/apparel-engineering-receipt.json"),
    })
}

fn run(inputs: Inputs) -> Result<Value, Box<dyn Error>> {
    let actual_head = current_git_head()?;
    if actual_head != inputs.expected_head {
        return Err(format!(
            "exact-head mismatch: expected {}, checked out {}",
            inputs.expected_head, actual_head
        )
        .into());
    }

    let (schema_bytes, schema) = read_json(&inputs.schema)?;
    let (instance_bytes, instance) = read_json(&inputs.instance)?;
    let (semantic_bytes, semantic) = read_json(&inputs.semantic_cases)?;
    let (source_bytes, sources) = read_json(&inputs.source_manifest)?;

    validate_schema_document(&schema).map_err(|error| format!("schema: {error}"))?;
    validate_instance(&schema, &instance)
        .map_err(|errors| format!("instance schema errors: {}", errors.join("; ")))?;
    validate_semantics(&instance)
        .map_err(|errors| format!("semantic errors: {}", errors.join("; ")))?;
    let (negative_cases, positive_cases) = validate_semantic_corpus(&semantic)
        .map_err(|errors| format!("semantic corpus errors: {}", errors.join("; ")))?;
    let source_count = validate_source_manifest(&sources)
        .map_err(|errors| format!("source manifest errors: {}", errors.join("; ")))?;

    let receipt = json!({
        "receipt_version": "0.1.0",
        "status": "PASS",
        "commit": actual_head,
        "fixture_id": get_str(&instance, "fixture_id"),
        "fixture_class": get_str(&instance, "fixture_class"),
        "physical_qualification": get_str(
            instance.pointer("/expected_disposition").unwrap_or(&Value::Null),
            "physical_qualification"
        ),
        "schema": {
            "path": inputs.schema.display().to_string(),
            "sha256": sha256(&schema_bytes),
            "dialect": DIALECT,
            "meta_schema_validated": true,
            "external_refs": "rejected"
        },
        "instance": {
            "path": inputs.instance.display().to_string(),
            "sha256": sha256(&instance_bytes),
            "draft_2020_12_valid": true,
            "cross_field_semantics_valid": true
        },
        "semantic_corpus": {
            "path": inputs.semantic_cases.display().to_string(),
            "sha256": sha256(&semantic_bytes),
            "case_count": negative_cases + positive_cases,
            "negative_intentions": negative_cases,
            "positive_intentions": positive_cases,
            "execution_status": "intent-corpus-validated-not-executed"
        },
        "benchmark_sources": {
            "path": inputs.source_manifest.display().to_string(),
            "sha256": sha256(&source_bytes),
            "source_count": source_count,
            "ingestion_or_reproduction_claim": "none"
        },
        "claim_ceiling": "Synthetic contract validation only; no physical product, manufacturing, recycling, safety, environmental, or regulatory qualification."
    });

    if let Some(parent) = inputs.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&inputs.output, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(receipt) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&receipt).expect("receipt serializes")
            );
        }
        Err(error) => {
            eprintln!("apparel thread audit failed: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_inputs() -> (Value, Value, Value, Value) {
        let schema: Value = serde_json::from_str(include_str!(
            "../../../docs/research/fixtures/apparel_engineering_thread_v0_1/apparel_thread.schema.json"
        ))
        .expect("schema parses");
        let instance: Value = serde_json::from_str(include_str!(
            "../../../docs/research/fixtures/apparel_engineering_thread_v0_1/apparel_thread.example.json"
        ))
        .expect("instance parses");
        let semantics: Value = serde_json::from_str(include_str!(
            "../../../docs/research/fixtures/apparel_engineering_thread_v0_1/semantic_cases.example.json"
        ))
        .expect("semantic corpus parses");
        let sources: Value = serde_json::from_str(include_str!(
            "../../../docs/research/fixtures/apparel_engineering_thread_v0_1/benchmark_sources_2026-11-12.json"
        ))
        .expect("source manifest parses");
        (schema, instance, semantics, sources)
    }

    #[test]
    fn schema_meta_schema_and_seed_instance_validate() {
        let (schema, instance, semantics, sources) = example_inputs();
        validate_schema_document(&schema).expect("schema is valid Draft 2020-12");
        validate_instance(&schema, &instance).expect("seed validates against schema");
        validate_semantics(&instance).expect("seed cross-field contract passes");
        let (negative, positive) =
            validate_semantic_corpus(&semantics).expect("semantic corpus is consistent");
        assert_eq!((negative, positive), (26, 4));
        let source_count = validate_source_manifest(&sources).unwrap();
        assert_eq!(source_count, array(&sources, "sources").len());
    }

    #[test]
    fn schema_rejects_unknown_root_property() {
        let (schema, mut instance, _, _) = example_inputs();
        instance
            .as_object_mut()
            .unwrap()
            .insert("unexpected".into(), json!("blocked"));
        assert!(validate_instance(&schema, &instance).is_err());
    }

    #[test]
    fn schema_rejects_missing_pattern_piece_identity() {
        let (schema, mut instance, _, _) = example_inputs();
        instance["pattern"]["pieces"][0]
            .as_object_mut()
            .unwrap()
            .remove("piece_id");
        assert!(validate_instance(&schema, &instance).is_err());
    }

    #[test]
    fn semantic_validator_rejects_unknown_bom_material() {
        let (_, mut instance, _, _) = example_inputs();
        instance["bom"][0]["material_id"] = json!("UNKNOWN-MATERIAL");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unresolved material_id"))
        );
    }

    #[test]
    fn semantic_validator_rejects_modelled_drape_as_physical_test() {
        let (_, mut instance, _, _) = example_inputs();
        instance["claims"][1]["disposition"] = json!("supported_within_scope");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("physical_test claim"))
        );
    }

    #[test]
    fn semantic_validator_rejects_measured_test_without_artifact() {
        let (_, mut instance, _, _) = example_inputs();
        instance["physical_tests"][0]["status"] = json!("measured");
        instance["physical_tests"][0]["result"] = json!(7);
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("requires evidence_ref"))
        );
    }

    #[test]
    fn semantic_validator_rejects_stale_geometry_claim_as_current() {
        let (_, mut instance, _, _) = example_inputs();
        instance["design"]["revision"] = json!("B");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("projection revision differs"))
        );
    }

    #[test]
    fn semantic_validator_rejects_synthetic_physical_qualification() {
        let (_, mut instance, _, _) = example_inputs();
        instance["expected_disposition"]["physical_qualification"] =
            json!("qualified_for_declared_scope");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("synthetic fixture"))
        );
    }

    #[test]
    fn semantic_validator_rejects_duplicate_pattern_piece_ids() {
        let (_, mut instance, _, _) = example_inputs();
        instance["pattern"]["pieces"][1]["piece_id"] = json!("FRONT-L");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate piece_id"))
        );
    }

    #[test]
    fn semantic_validator_rejects_duplicate_operation_sequences() {
        let (_, mut instance, _, _) = example_inputs();
        instance["operations"][1]["sequence"] = json!(10);
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate operation sequence"))
        );
    }

    #[test]
    fn semantic_validator_rejects_design_pattern_revision_mismatch() {
        let (_, mut instance, _, _) = example_inputs();
        instance["pattern"]["revision"] = json!("B");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("pattern revision"))
        );
    }

    #[test]
    fn semantic_validator_rejects_measured_material_property_without_evidence() {
        let (_, mut instance, _, _) = example_inputs();
        instance["materials"][0]["properties"][0]["status"] = json!("measured");
        instance["materials"][0]["properties"][0]["value"] = json!(180.0);
        instance["materials"][0]["properties"][0]["unit"] = json!("g/m2");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("measured property requires"))
        );
    }

    #[test]
    fn semantic_validator_rejects_measured_composition_without_composition_evidence() {
        let (_, mut instance, _, _) = example_inputs();
        instance["materials"][0]["composition_state"] = json!("measured");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("measured composition requires"))
        );
    }

    #[test]
    fn semantic_validator_rejects_unresolved_claim_evidence() {
        let (_, mut instance, _, _) = example_inputs();
        instance["claims"][0]["evidence_refs"] = json!(["DOES-NOT-EXIST"]);
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unresolved evidence reference"))
        );
    }

    #[test]
    fn semantic_validator_rejects_bound_material_lot_without_quantity() {
        let (_, mut instance, _, _) = example_inputs();
        instance["bom"][0]["lot_binding"] = json!("bound");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("bound lot requires"))
        );
    }

    #[test]
    fn semantic_validator_accepts_physical_claim_bound_to_measured_test() {
        let (_, mut instance, _, _) = example_inputs();
        instance["physical_tests"][0]["status"] = json!("measured");
        instance["physical_tests"][0]["result"] = json!(24.0);
        instance["physical_tests"][0]["evidence_ref"] = json!("EVIDENCE-TEST-001");
        instance["claims"][1]["evidence_refs"] = json!(["EVIDENCE-TEST-001"]);
        instance["claims"][1]["disposition"] = json!("supported_within_scope");
        assert!(validate_semantics(&instance).is_ok());
    }

    #[test]
    fn semantic_validator_rejects_physical_claim_bound_only_to_model_projection() {
        let (_, mut instance, _, _) = example_inputs();
        instance["claims"][1]["evidence_refs"] = json!(["PROJ-DRAPE-001"]);
        instance["claims"][1]["disposition"] = json!("supported_within_scope");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("physical_test claim"))
        );
    }

    #[test]
    fn semantic_validator_rejects_production_scope_inflation() {
        let (_, mut instance, _, _) = example_inputs();
        instance["claims"][0]["statement"] =
            json!("This process is qualified for all sizes and production batches.");
        instance["claims"][0]["disposition"] = json!("supported_within_scope");
        let errors = validate_semantics(&instance).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("production-wide capability"))
        );
    }

    #[test]
    fn schema_references_are_self_contained() {
        let (schema, _, _, _) = example_inputs();
        reject_external_schema_references(&schema, "$").unwrap();
    }
}
