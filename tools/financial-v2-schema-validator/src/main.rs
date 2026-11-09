use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use jsonschema::{Retrieve, Uri};
use serde_json::Value;

type AnyError = Box<dyn Error + Send + Sync>;

const EVIDENCE_SCHEMA_PATH: &str = "benchmarks/financial-agent-v1/evidence-v2.schema.json";
const RESEARCH_MANIFEST_SCHEMA_PATH: &str =
    "benchmarks/financial-agent-v1/evidence-manifest-v2.schema.json";
const QUANT_MANIFEST_SCHEMA_PATH: &str =
    "benchmarks/quant-strategy-v1/observation-manifest-v2.schema.json";
const RESEARCH_EXAMPLE_PATH: &str =
    "benchmarks/financial-agent-v1/dev-example.evidence-v2.json";
const QUANT_EXAMPLE_PATH: &str =
    "benchmarks/quant-strategy-v1/dev-example.observations-v2.json";

#[derive(Clone, Default)]
struct LocalSchemaResources {
    resources: HashMap<String, Value>,
}

impl LocalSchemaResources {
    fn insert(&mut self, schema: &Value) -> Result<(), String> {
        let id = schema
            .get("$id")
            .and_then(Value::as_str)
            .ok_or_else(|| "schema resource is missing a string $id".to_owned())?;
        self.resources.insert(id.to_owned(), schema.clone());
        Ok(())
    }
}

impl Retrieve for LocalSchemaResources {
    fn retrieve(&self, uri: &Uri<String>) -> Result<Value, AnyError> {
        self.resources.get(uri.as_str()).cloned().ok_or_else(|| {
            format!("schema reference is not present in the offline registry: {uri}").into()
        })
    }
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("validator crate must be two levels below repository root")
}

fn load_json(root: &Path, relative_path: &str) -> Result<Value, String> {
    let path = root.join(relative_path);
    let bytes = fs::read(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid JSON in {}: {error}", path.display()))
}

fn check_schema_meta(schema: &Value, path: &str) -> Result<(), String> {
    if !jsonschema::draft202012::meta::is_valid(schema) {
        return Err(format!("{path} is not a valid Draft 2020-12 schema"));
    }
    Ok(())
}

fn validate_instance(
    schema: &Value,
    resources: &LocalSchemaResources,
    instance: &Value,
    instance_name: &str,
) -> Result<(), String> {
    let validator = jsonschema::draft202012::options()
        .with_retriever(resources.clone())
        .should_validate_formats(true)
        .build(schema)
        .map_err(|error| format!("cannot compile schema for {instance_name}: {error}"))?;

    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| format!("{} at {}", error, error.instance_path()))
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{instance_name} failed schema validation:\n{}",
            errors.join("\n")
        ))
    }
}

fn required_string<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("required string missing at {pointer}"))
}

fn parse_time(value: &str, field: &str) -> Result<chrono::DateTime<chrono::Utc>, String> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|time| time.with_timezone(&chrono::Utc))
        .map_err(|error| format!("{field} is not an RFC3339 instant: {error}"))
}

fn validate_proof_state(
    proof: &Value,
    qualification: &str,
    proof_name: &str,
) -> Result<(), String> {
    let state = required_string(proof, "/state")?;
    match (qualification, state) {
        ("synthetic_development_only", "synthetic_fixture_only") => Ok(()),
        ("synthetic_development_only", "verified") => Err(format!(
            "{proof_name} cannot be marked verified in a synthetic-only fixture"
        )),
        ("historical_confirmation" | "prospective_shadow", "verified") => Ok(()),
        (_, "missing" | "unverifiable") => Err(format!(
            "{proof_name} is not sufficient for a qualifying evidence profile"
        )),
        (_, "synthetic_fixture_only") => Err(format!(
            "{proof_name} synthetic placeholder cannot qualify historical/shadow evidence"
        )),
        (_, other) => Err(format!("{proof_name} has unsupported proof state {other}")),
    }
}

fn validate_evidence_semantics(
    evidence: &Value,
    principal_id: &str,
    rights_profile_id: &str,
    qualification: &str,
    decision_frontier: &str,
) -> Result<(), String> {
    if required_string(evidence, "/qualification")? != qualification {
        return Err(format!(
            "evidence {} qualification differs from its parent manifest",
            evidence["evidence_id"]
        ));
    }

    let availability = evidence
        .get("availability")
        .ok_or_else(|| "evidence is missing availability".to_owned())?;
    if required_string(availability, "/principal_id")? != principal_id {
        return Err(format!(
            "evidence {} availability principal differs from manifest",
            evidence["evidence_id"]
        ));
    }

    let rights = evidence
        .get("rights_authorization")
        .ok_or_else(|| "evidence is missing rights_authorization".to_owned())?;
    if required_string(rights, "/principal_id")? != principal_id {
        return Err(format!(
            "evidence {} rights principal differs from manifest",
            evidence["evidence_id"]
        ));
    }
    if required_string(rights, "/rights_profile_id")? != rights_profile_id {
        return Err(format!(
            "evidence {} rights profile differs from manifest",
            evidence["evidence_id"]
        ));
    }
    if required_string(rights, "/disposition")? != "authorized" {
        return Err(format!(
            "evidence {} does not have an authorized rights disposition",
            evidence["evidence_id"]
        ));
    }

    let frontier = parse_time(decision_frontier, "decision_frontier")?;
    if let Some(start) = rights.get("valid_from").and_then(Value::as_str) {
        if frontier < parse_time(start, "rights_authorization.valid_from")? {
            return Err("rights entitlement is not yet valid at the decision frontier".to_owned());
        }
    }
    if let Some(end) = rights.get("valid_until").and_then(Value::as_str) {
        if frontier >= parse_time(end, "rights_authorization.valid_until")? {
            return Err("rights entitlement is expired at the decision frontier".to_owned());
        }
    }

    let availability_proof = availability
        .get("proof")
        .ok_or_else(|| "availability proof object is missing".to_owned())?;
    let rights_proof = rights
        .get("proof")
        .ok_or_else(|| "rights proof object is missing".to_owned())?;

    validate_proof_state(availability_proof, qualification, "availability proof")?;
    validate_proof_state(rights_proof, qualification, "rights proof")?;

    let precision = required_string(availability, "/precision")?;
    let known_at = availability.get("known_at");
    let interval = availability.get("interval");

    match precision {
        "exact_instant" => {
            let known_at = known_at
                .and_then(Value::as_str)
                .ok_or_else(|| "exact_instant requires a string known_at".to_owned())?;
            if !interval.is_some_and(Value::is_null) {
                return Err("exact_instant must not also carry an interval".to_owned());
            }
            let known = parse_time(known_at, "availability.known_at")?;
            let retrieved_at = parse_time(
                required_string(evidence, "/retrieved_at")?,
                "retrieved_at",
            )?;
            if retrieved_at < known {
                return Err("retrieved_at precedes known_at".to_owned());
            }
            if qualification != "synthetic_development_only" {
                let frontier = parse_time(decision_frontier, "decision_frontier")?;
                if known > frontier {
                    return Err("future-known evidence cannot qualify at this frontier".to_owned());
                }
            }
        }
        "bounded_interval" => {
            if !known_at.is_some_and(Value::is_null) {
                return Err("bounded_interval must not provide exact known_at".to_owned());
            }
            let interval = interval
                .and_then(Value::as_object)
                .ok_or_else(|| "bounded_interval requires interval object".to_owned())?;
            let earliest = parse_time(
                interval
                    .get("earliest_possible_at")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "interval earliest_possible_at is missing".to_owned())?,
                "earliest_possible_at",
            )?;
            let latest = parse_time(
                interval
                    .get("latest_possible_at")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "interval latest_possible_at is missing".to_owned())?,
                "latest_possible_at",
            )?;
            if latest < earliest {
                return Err("availability interval is reversed".to_owned());
            }
            if qualification != "synthetic_development_only" {
                let frontier = parse_time(decision_frontier, "decision_frontier")?;
                if earliest <= frontier && frontier < latest {
                    return Err(
                        "availability interval straddles decision frontier; disposition is indeterminate"
                            .to_owned(),
                    );
                }
                if earliest > frontier {
                    return Err("availability interval is wholly after decision frontier".to_owned());
                }
            }
            let time_basis = interval
                .get("time_basis")
                .and_then(Value::as_str)
                .ok_or_else(|| "interval time_basis is missing".to_owned())?;
            if time_basis == "source_calendar_converted_with_declared_timezone"
                && interval.get("source_timezone").and_then(Value::as_str).is_none()
            {
                return Err(
                    "source-calendar interval must declare its timezone; do not invent one"
                        .to_owned(),
                );
            }
        }
        "unknown" => {
            if !known_at.is_some_and(Value::is_null)
                || !interval.is_some_and(Value::is_null)
            {
                return Err("unknown precision must not carry exact or bounded availability".to_owned());
            }
            if qualification != "synthetic_development_only" {
                return Err("unknown availability precision cannot qualify evidence".to_owned());
            }
        }
        other => return Err(format!("unsupported availability precision {other}")),
    }

    Ok(())
}

fn validate_research_manifest(
    manifest: &Value,
    evidence_schema: &Value,
    resources: &LocalSchemaResources,
) -> Result<(), String> {
    validate_instance(
        evidence_schema,
        resources,
        &manifest["artifacts"][0],
        "research evidence v2 artifact",
    )?;

    let principal_id = required_string(manifest, "/principal_id")?;
    let rights_profile_id = required_string(manifest, "/rights_profile_id")?;
    let qualification = required_string(manifest, "/qualification")?;
    let decision_frontier = required_string(manifest, "/decision_frontier")?;
    let evaluation_mode = required_string(manifest, "/evaluation_mode")?;

    match qualification {
        "synthetic_development_only" if evaluation_mode == "development_only" => {}
        "historical_confirmation" if evaluation_mode == "historical_reconstruction" => {}
        "prospective_shadow" if evaluation_mode == "prospective_streaming" => {}
        _ => return Err("manifest qualification and evaluation_mode are inconsistent".to_owned()),
    }

    let artifacts = manifest
        .get("artifacts")
        .and_then(Value::as_array)
        .ok_or_else(|| "research manifest artifacts must be an array".to_owned())?;
    if artifacts.is_empty() {
        return Err("research manifest must contain at least one artifact".to_owned());
    }

    let mut seen_evidence_ids = std::collections::BTreeSet::new();
    for evidence in artifacts {
        let evidence_id = required_string(evidence, "/evidence_id")?;
        if !seen_evidence_ids.insert(evidence_id.to_owned()) {
            return Err(format!("duplicate evidence_id {evidence_id} in research manifest"));
        }
        validate_instance(evidence_schema, resources, evidence, "research evidence artifact")?;
        validate_evidence_semantics(
            evidence,
            principal_id,
            rights_profile_id,
            qualification,
            decision_frontier,
        )?;
    }
    Ok(())
}

fn validate_quant_manifest(
    manifest: &Value,
    quant_schema: &Value,
    evidence_schema: &Value,
    resources: &LocalSchemaResources,
) -> Result<(), String> {
    validate_instance(quant_schema, resources, manifest, "quant observation manifest")?;

    let principal_id = required_string(manifest, "/principal_id")?;
    let rights_profile_id = required_string(manifest, "/rights_profile_id")?;
    let qualification = required_string(manifest, "/qualification")?;
    let decision_frontier = required_string(manifest, "/decision_frontier")?;
    let evaluation_mode = required_string(manifest, "/evaluation_mode")?;

    match qualification {
        "synthetic_development_only" if evaluation_mode == "development_only" => {}
        "historical_confirmation" if evaluation_mode == "historical_reconstruction" => {}
        "prospective_shadow" if evaluation_mode == "prospective_streaming" => {}
        _ => return Err("quant qualification and evaluation_mode are inconsistent".to_owned()),
    }

    let observations = manifest
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| "quant observations must be an array".to_owned())?;
    let mut observation_ids = std::collections::BTreeSet::new();
    let mut evidence_ids = std::collections::BTreeSet::new();

    for observation in observations {
        let observation_id = required_string(observation, "/observation_id")?;
        if !observation_ids.insert(observation_id.to_owned()) {
            return Err(format!("duplicate observation_id {observation_id}"));
        }
        let evidence = observation
            .get("evidence")
            .ok_or_else(|| format!("observation {observation_id} has no evidence object"))?;
        validate_instance(evidence_schema, resources, evidence, "quant observation evidence")?;
        let evidence_id = required_string(evidence, "/evidence_id")?;
        if !evidence_ids.insert(evidence_id.to_owned()) {
            return Err(format!("duplicate evidence_id {evidence_id} in quant manifest"));
        }
        validate_evidence_semantics(
            evidence,
            principal_id,
            rights_profile_id,
            qualification,
            decision_frontier,
        )?;
    }
    Ok(())
}

fn run(root: &Path) -> Result<(), String> {
    let evidence_schema = load_json(root, EVIDENCE_SCHEMA_PATH)?;
    let research_schema = load_json(root, RESEARCH_MANIFEST_SCHEMA_PATH)?;
    let quant_schema = load_json(root, QUANT_MANIFEST_SCHEMA_PATH)?;
    let research_example = load_json(root, RESEARCH_EXAMPLE_PATH)?;
    let quant_example = load_json(root, QUANT_EXAMPLE_PATH)?;

    for (schema, path) in [
        (&evidence_schema, EVIDENCE_SCHEMA_PATH),
        (&research_schema, RESEARCH_MANIFEST_SCHEMA_PATH),
        (&quant_schema, QUANT_MANIFEST_SCHEMA_PATH),
    ] {
        check_schema_meta(schema, path)?;
    }

    let mut resources = LocalSchemaResources::default();
    resources.insert(&evidence_schema)?;
    resources.insert(&research_schema)?;
    resources.insert(&quant_schema)?;

    validate_instance(&research_schema, &resources, &research_example, RESEARCH_EXAMPLE_PATH)?;
    validate_instance(&quant_schema, &resources, &quant_example, QUANT_EXAMPLE_PATH)?;
    validate_research_manifest(&research_example, &evidence_schema, &resources)?;
    validate_quant_manifest(&quant_example, &quant_schema, &evidence_schema, &resources)?;

    println!("Draft 2020-12 schema and semantic checks passed for:");
    println!("  {EVIDENCE_SCHEMA_PATH}");
    println!("  {RESEARCH_MANIFEST_SCHEMA_PATH}");
    println!("  {QUANT_MANIFEST_SCHEMA_PATH}");
    println!("  {RESEARCH_EXAMPLE_PATH}");
    println!("  {QUANT_EXAMPLE_PATH}");
    println!("Note: synthetic fixture proofs remain non-qualifying placeholders.");
    Ok(())
}

fn main() {
    let root = repository_root();
    if let Err(error) = run(&root) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_root() -> PathBuf {
        repository_root()
    }

    #[test]
    fn public_v2_examples_validate_offline_and_remain_synthetic_only() {
        run(&fixture_root()).expect("v2 schemas, examples and semantic bindings must validate");
    }

    #[test]
    fn exact_precision_cannot_omit_known_at() {
        let root = fixture_root();
        let mut evidence = load_json(&root, RESEARCH_EXAMPLE_PATH).unwrap()["artifacts"][0].clone();
        evidence["availability"]["known_at"] = Value::Null;
        assert!(validate_evidence_semantics(
            &evidence,
            "principal:synthetic-public-research",
            "rights:synthetic-dev",
            "synthetic_development_only",
            "2026-10-01T12:00:00Z",
        )
        .is_err());
    }

    #[test]
    fn principal_mismatch_is_rejected_even_when_json_is_structurally_valid() {
        let root = fixture_root();
        let mut evidence = load_json(&root, RESEARCH_EXAMPLE_PATH).unwrap()["artifacts"][0].clone();
        evidence["availability"]["principal_id"] = Value::String("principal:other".to_owned());
        assert!(validate_evidence_semantics(
            &evidence,
            "principal:synthetic-public-research",
            "rights:synthetic-dev",
            "synthetic_development_only",
            "2026-10-01T12:00:00Z",
        )
        .is_err());
    }

    #[test]
    fn synthetic_placeholder_proofs_cannot_qualify_historical_confirmation() {
        let root = fixture_root();
        let evidence = load_json(&root, RESEARCH_EXAMPLE_PATH).unwrap()["artifacts"][0].clone();
        assert!(validate_evidence_semantics(
            &evidence,
            "principal:synthetic-public-research",
            "rights:synthetic-dev",
            "historical_confirmation",
            "2026-10-01T12:00:00Z",
        )
        .is_err());
    }


    #[test]
    fn rights_entitlement_must_be_valid_at_the_decision_frontier() {
        let root = fixture_root();
        let mut evidence = load_json(&root, RESEARCH_EXAMPLE_PATH).unwrap()["artifacts"][0].clone();
        evidence["rights_authorization"]["valid_until"] =
            Value::String("2026-10-01T12:00:00Z".to_owned());
        assert!(validate_evidence_semantics(
            &evidence,
            "principal:synthetic-public-research",
            "rights:synthetic-dev",
            "synthetic_development_only",
            "2026-10-01T12:00:00Z",
        )
        .is_err());
    }

    #[test]
    fn interval_straddling_a_qualified_frontier_is_rejected() {
        let root = fixture_root();
        let mut evidence = load_json(&root, RESEARCH_EXAMPLE_PATH).unwrap()["artifacts"][0].clone();
        evidence["qualification"] = Value::String("historical_confirmation".to_owned());
        evidence["availability"]["precision"] = Value::String("bounded_interval".to_owned());
        evidence["availability"]["known_at"] = Value::Null;
        evidence["availability"]["interval"] = serde_json::json!({
            "earliest_possible_at": "2026-10-01T11:00:00Z",
            "latest_possible_at": "2026-10-01T13:00:00Z",
            "bounds": "inclusive",
            "time_basis": "utc_instants"
        });
        evidence["availability"]["proof"]["state"] = Value::String("verified".to_owned());
        evidence["rights_authorization"]["proof"]["state"] = Value::String("verified".to_owned());
        assert!(validate_evidence_semantics(
            &evidence,
            "principal:synthetic-public-research",
            "rights:synthetic-dev",
            "historical_confirmation",
            "2026-10-01T12:00:00Z",
        )
        .is_err());
    }

    #[test]
    fn source_calendar_availability_requires_an_explicit_timezone() {
        let root = fixture_root();
        let mut evidence = load_json(&root, RESEARCH_EXAMPLE_PATH).unwrap()["artifacts"][0].clone();
        evidence["availability"]["precision"] = Value::String("bounded_interval".to_owned());
        evidence["availability"]["known_at"] = Value::Null;
        evidence["availability"]["interval"] = serde_json::json!({
            "earliest_possible_at": "2026-09-30T00:00:00Z",
            "latest_possible_at": "2026-10-01T00:00:00Z",
            "bounds": "inclusive",
            "time_basis": "source_calendar_converted_with_declared_timezone",
            "source_date": "2026-09-30"
        });
        assert!(validate_evidence_semantics(
            &evidence,
            "principal:synthetic-public-research",
            "rights:synthetic-dev",
            "synthetic_development_only",
            "2026-10-02T12:00:00Z",
        )
        .is_err());
    }

    #[test]
    fn unknown_availability_precision_is_not_qualified() {
        let root = fixture_root();
        let mut evidence = load_json(&root, RESEARCH_EXAMPLE_PATH).unwrap()["artifacts"][0].clone();
        evidence["availability"]["precision"] = Value::String("unknown".to_owned());
        evidence["availability"]["known_at"] = Value::Null;
        evidence["availability"]["proof"]["state"] = Value::String("unverifiable".to_owned());
        assert!(validate_evidence_semantics(
            &evidence,
            "principal:synthetic-public-research",
            "rights:synthetic-dev",
            "historical_confirmation",
            "2026-10-01T12:00:00Z",
        )
        .is_err());
    }
}
