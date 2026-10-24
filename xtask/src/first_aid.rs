//! Research-only validation for the synthetic first-aid benchmark contracts.
//!
//! This checks JSON shape and software safety metadata. It is not a JSON Schema
//! engine, a clinical oracle, a protocol approval service, or clinical validation.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

const CORPUS_VERSION: &str = "first-aid-scenario-corpus-v0";
const BENCHMARK_VERSION: &str = "first-aid-benchmark-v0";
const SCHEMA_DRAFT: &str = "https://json-schema.org/draft/2020-12/schema";
const FAIL_CLOSED: &str = "fail_closed_preserve_emergency_path";
const VALID_BUNDLE_STATE: &str = "valid_pinned_bundle_required";

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct ScenarioCorpus {
    schema_version: String,
    status: String,
    clinical_use_authorized: bool,
    notes: Option<String>,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    title: String,
    domain: String,
    scenario_type: String,
    synthetic_only: bool,
    clinical_content_required: bool,
    source_bundle_state: String,
    connectivity: String,
    fault_injections: Vec<String>,
    required_invariants: Vec<String>,
    forbidden_behaviors: Vec<String>,
    expected_disposition: String,
    clinical_review_status: String,
    notes: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct BenchmarkSpec {
    schema_version: String,
    status: String,
    clinical_use_authorized: bool,
    notes: Option<String>,
    baselines: Vec<Baseline>,
    benchmarks: Vec<Benchmark>,
    split_policy: BTreeMap<String, String>,
    release_gates: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Baseline {
    id: String,
    name: String,
    role: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Benchmark {
    id: String,
    name: String,
    category: String,
    measure: String,
    target: String,
    gate: String,
    claim_ceiling: String,
}

#[derive(Debug, Serialize)]
struct ValidationReceipt {
    receipt_version: &'static str,
    status: &'static str,
    corpus_sha256: String,
    benchmark_sha256: String,
    schema_sha256: String,
    source_revision_sha: String,
    validator_source_sha256: String,
    entrypoint_source_sha256: String,
    executable_sha256: String,
    runtime_os: String,
    runtime_arch: String,
    scenario_count: usize,
    benchmark_count: usize,
    baseline_count: usize,
    claim_ceiling: &'static str,
}

pub fn run(
    root: &Path,
    corpus_arg: &Path,
    benchmark_arg: &Path,
    schema_arg: &Path,
    out_arg: Option<&Path>,
) -> Result<()> {
    let corpus_path = resolve(root, corpus_arg);
    let benchmark_path = resolve(root, benchmark_arg);
    let schema_path = resolve(root, schema_arg);

    let corpus_bytes = fs::read(&corpus_path)
        .with_context(|| format!("read scenario corpus {}", corpus_path.display()))?;
    let benchmark_bytes = fs::read(&benchmark_path)
        .with_context(|| format!("read benchmark spec {}", benchmark_path.display()))?;
    let schema_bytes = fs::read(&schema_path)
        .with_context(|| format!("read scenario schema {}", schema_path.display()))?;

    let (corpus, benchmark, _schema) = validate_documents(
        std::str::from_utf8(&corpus_bytes).context("scenario corpus must be UTF-8")?,
        std::str::from_utf8(&benchmark_bytes).context("benchmark spec must be UTF-8")?,
        std::str::from_utf8(&schema_bytes).context("scenario schema must be UTF-8")?,
    )?;

    let validator_source = fs::read(root.join("xtask/src/first_aid.rs"))
        .context("read first-aid validator source for receipt identity")?;
    let entrypoint_source = fs::read(root.join("xtask/src/main.rs"))
        .context("read xtask entrypoint source for receipt identity")?;
    let executable_path = std::env::current_exe().context("resolve validator executable")?;
    let executable_bytes = fs::read(&executable_path)
        .with_context(|| format!("read validator executable {}", executable_path.display()))?;

    let receipt = ValidationReceipt {
        receipt_version: "first-aid-contract-receipt-v0",
        status: "contract_validation_passed",
        corpus_sha256: sha256_hex(&corpus_bytes),
        benchmark_sha256: sha256_hex(&benchmark_bytes),
        schema_sha256: sha256_hex(&schema_bytes),
        source_revision_sha: std::env::var("FIRST_AID_SOURCE_REVISION")
            .unwrap_or_else(|_| "not-provided".to_string()),
        validator_source_sha256: sha256_hex(&validator_source),
        entrypoint_source_sha256: sha256_hex(&entrypoint_source),
        executable_sha256: sha256_hex(&executable_bytes),
        runtime_os: std::env::consts::OS.to_string(),
        runtime_arch: std::env::consts::ARCH.to_string(),
        scenario_count: corpus.scenarios.len(),
        benchmark_count: benchmark.benchmarks.len(),
        baseline_count: benchmark.baselines.len(),
        claim_ceiling: "synthetic contract integrity only; no clinical validation",
    };
    let encoded = serde_json::to_string_pretty(&receipt)?;

    if let Some(out_arg) = out_arg {
        let out_path = resolve(root, out_arg);
        if let Some(parent) = out_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create receipt directory {}", parent.display()))?;
        }
        fs::write(&out_path, format!("{encoded}\n"))
            .with_context(|| format!("write validation receipt {}", out_path.display()))?;
        println!("Receipt written: {}", out_path.display());
    }
    println!("{encoded}");
    Ok(())
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_documents(
    corpus_json: &str,
    benchmark_json: &str,
    schema_json: &str,
) -> Result<(ScenarioCorpus, BenchmarkSpec, Value)> {
    let schema: Value = serde_json::from_str(schema_json).context("parse JSON Schema")?;
    let corpus: ScenarioCorpus =
        serde_json::from_str(corpus_json).context("parse scenario corpus")?;
    let benchmark: BenchmarkSpec =
        serde_json::from_str(benchmark_json).context("parse benchmark specification")?;

    validate_schema_contract(&schema, &corpus)?;
    validate_corpus(&corpus, &schema)?;
    validate_benchmark(&benchmark)?;
    Ok((corpus, benchmark, schema))
}

fn validate_schema_contract(schema: &Value, corpus: &ScenarioCorpus) -> Result<()> {
    if schema.get("$schema").and_then(Value::as_str) != Some(SCHEMA_DRAFT) {
        bail!("scenario schema must declare JSON Schema draft 2020-12");
    }
    if schema.get("additionalProperties").and_then(Value::as_bool) != Some(false) {
        bail!("scenario schema must reject unknown top-level fields");
    }
    let schema_version = schema
        .pointer("/properties/schema_version/const")
        .and_then(Value::as_str);
    if schema_version != Some(CORPUS_VERSION) || corpus.schema_version != CORPUS_VERSION {
        bail!("scenario corpus/schema version mismatch");
    }
    if corpus.status != "synthetic_contract_tests_only" {
        bail!("scenario corpus status must remain synthetic_contract_tests_only");
    }
    if corpus.clinical_use_authorized {
        bail!("scenario corpus must not authorize clinical use");
    }

    let top_required = schema
        .get("required")
        .and_then(Value::as_array)
        .context("schema is missing its top-level required list")?;
    for field in [
        "schema_version",
        "status",
        "clinical_use_authorized",
        "scenarios",
    ] {
        if !top_required.iter().any(|v| v.as_str() == Some(field)) {
            bail!("schema top-level required list omits {field}");
        }
    }

    if schema
        .pointer("/properties/scenarios/items/additionalProperties")
        .and_then(Value::as_bool)
        != Some(false)
    {
        bail!("scenario schema must reject unknown scenario fields");
    }
    if schema
        .pointer("/properties/scenarios/items/properties/synthetic_only/const")
        .and_then(Value::as_bool)
        != Some(true)
        || schema
            .pointer("/properties/scenarios/items/properties/clinical_content_required/const")
            .and_then(Value::as_bool)
            != Some(true)
    {
        bail!(
            "scenario schema must keep fixtures synthetic and clinical content separately required"
        );
    }
    if schema
        .pointer("/properties/scenarios/items/properties/scenario_id/pattern")
        .and_then(Value::as_str)
        != Some("^FA-[0-9]{3}$")
    {
        bail!("scenario schema must pin the FA-### scenario ID format");
    }

    let scenario_required = schema
        .pointer("/properties/scenarios/items/required")
        .and_then(Value::as_array)
        .context("schema is missing scenario required fields")?;
    for field in [
        "scenario_id",
        "source_bundle_state",
        "connectivity",
        "scenario_type",
        "synthetic_only",
        "clinical_content_required",
        "fault_injections",
        "required_invariants",
        "forbidden_behaviors",
        "expected_disposition",
        "clinical_review_status",
    ] {
        if !scenario_required.iter().any(|v| v.as_str() == Some(field)) {
            bail!("schema scenario required list omits {field}");
        }
    }

    if schema
        .pointer("/properties/clinical_use_authorized/const")
        .and_then(Value::as_bool)
        != Some(false)
    {
        bail!("schema must hard-code clinical_use_authorized=false");
    }
    if schema
        .pointer("/properties/scenarios/items/properties/clinical_review_status/const")
        .and_then(Value::as_str)
        != Some("not_reviewed")
    {
        bail!("schema must keep clinical_review_status at not_reviewed for this corpus");
    }
    Ok(())
}

fn enum_contains(schema: &Value, property: &str, value: &str) -> bool {
    schema
        .pointer(&format!(
            "/properties/scenarios/items/properties/{property}/enum"
        ))
        .and_then(Value::as_array)
        .map(|items| items.iter().any(|item| item.as_str() == Some(value)))
        .unwrap_or(false)
}

fn is_unique_id(id: &str, prefix: &str) -> bool {
    let Some(suffix) = id.strip_prefix(prefix) else {
        return false;
    };
    suffix.len() == 3 && suffix.bytes().all(|byte| byte.is_ascii_digit())
}

fn ensure_unique_nonempty(values: &[String], field: &str, scenario_id: &str) -> Result<()> {
    if values.is_empty() {
        bail!("{scenario_id}: {field} must not be empty");
    }
    let mut seen = HashSet::new();
    for value in values {
        if value.trim().is_empty() {
            bail!("{scenario_id}: {field} contains an empty item");
        }
        if !seen.insert(value) {
            bail!("{scenario_id}: duplicate item {value:?} in {field}");
        }
    }
    Ok(())
}

fn validate_corpus(corpus: &ScenarioCorpus, schema: &Value) -> Result<()> {
    if corpus.status != "synthetic_contract_tests_only" {
        bail!("scenario corpus status must remain synthetic_contract_tests_only");
    }
    if corpus.clinical_use_authorized {
        bail!("scenario corpus must not authorize clinical use");
    }
    if corpus.scenarios.is_empty() {
        bail!("scenario corpus must contain at least one scenario");
    }

    let mut ids = HashSet::new();
    let mut has_positive_control = false;
    let mut has_fail_closed_control = false;
    let mut scenario_types = HashSet::new();

    for scenario in &corpus.scenarios {
        if !is_unique_id(&scenario.scenario_id, "FA-") {
            bail!("invalid scenario ID: {}", scenario.scenario_id);
        }
        if !ids.insert(&scenario.scenario_id) {
            bail!("duplicate scenario ID: {}", scenario.scenario_id);
        }
        for (field, value) in [
            ("title", scenario.title.as_str()),
            ("domain", scenario.domain.as_str()),
        ] {
            if value.trim().is_empty() {
                bail!("{}: {field} must not be empty", scenario.scenario_id);
            }
        }
        if !scenario.synthetic_only || !scenario.clinical_content_required {
            bail!(
                "{}: every fixture must be synthetic and declare clinical content as separately required",
                scenario.scenario_id
            );
        }
        if scenario.clinical_review_status != "not_reviewed" {
            bail!(
                "{}: clinical review must remain not_reviewed in this synthetic corpus",
                scenario.scenario_id
            );
        }
        if !enum_contains(schema, "scenario_type", &scenario.scenario_type) {
            bail!(
                "{}: scenario_type {:?} is not declared by schema",
                scenario.scenario_id,
                scenario.scenario_type
            );
        }
        if !enum_contains(
            schema,
            "expected_disposition",
            &scenario.expected_disposition,
        ) {
            bail!(
                "{}: expected_disposition {:?} is not declared by schema",
                scenario.scenario_id,
                scenario.expected_disposition
            );
        }
        let state = &scenario.source_bundle_state;
        if !enum_contains(schema, "source_bundle_state", state) {
            bail!(
                "{}: source_bundle_state {:?} is not declared by schema",
                scenario.scenario_id,
                state
            );
        }
        if matches!(
            state.as_str(),
            "missing"
                | "expired"
                | "revoked"
                | "digest_mismatch"
                | "jurisdiction_mismatch"
                | "unsupported_schema"
                | "unknown"
        ) && scenario.expected_disposition != FAIL_CLOSED
        {
            bail!(
                "{}: invalid/unknown bundle state must fail closed for protocol-specific steps",
                scenario.scenario_id
            );
        }
        let connectivity = &scenario.connectivity;
        if !enum_contains(schema, "connectivity", connectivity) {
            bail!(
                "{}: connectivity {:?} is not declared by schema",
                scenario.scenario_id,
                connectivity
            );
        }
        if connectivity == "offline"
            && scenario.source_bundle_state == VALID_BUNDLE_STATE
            && scenario.expected_disposition != "render_valid_local_bundle_offline"
        {
            bail!(
                "{}: offline session with a valid local bundle must use the offline disposition",
                scenario.scenario_id
            );
        }

        ensure_unique_nonempty(
            &scenario.fault_injections,
            "fault_injections",
            &scenario.scenario_id,
        )?;
        ensure_unique_nonempty(
            &scenario.required_invariants,
            "required_invariants",
            &scenario.scenario_id,
        )?;
        ensure_unique_nonempty(
            &scenario.forbidden_behaviors,
            "forbidden_behaviors",
            &scenario.scenario_id,
        )?;

        scenario_types.insert(scenario.scenario_type.as_str());
        has_positive_control |= scenario.scenario_type == "valid_bundle";
        has_fail_closed_control |= scenario.expected_disposition == FAIL_CLOSED;
    }

    if !has_positive_control {
        bail!("corpus must retain a valid-bundle positive control");
    }
    if !has_fail_closed_control {
        bail!("corpus must retain at least one fail-closed negative control");
    }
    for required_type in [
        "content_integrity",
        "content_currency",
        "connectivity",
        "missing_data",
    ] {
        if !scenario_types.contains(required_type) {
            bail!("corpus is missing required fault dimension {required_type}");
        }
    }

    // Notes are optional schema fields; this binding intentionally does not interpret them as
    // clinical instructions or treat them as part of a clinical decision oracle.
    let _ = &corpus.notes;
    for scenario in &corpus.scenarios {
        let _ = &scenario.notes;
    }
    Ok(())
}

fn validate_benchmark(benchmark: &BenchmarkSpec) -> Result<()> {
    if benchmark.schema_version != BENCHMARK_VERSION {
        bail!(
            "unsupported benchmark spec version: {}",
            benchmark.schema_version
        );
    }
    if benchmark.status != "proposed_targets_unmeasured" {
        bail!("benchmark targets must remain explicitly unmeasured until evaluated");
    }
    if benchmark.clinical_use_authorized {
        bail!("benchmark spec must not authorize clinical use");
    }
    if benchmark.benchmarks.is_empty() || benchmark.baselines.is_empty() {
        bail!("benchmark spec must include benchmark metrics and comparison baselines");
    }
    if benchmark.release_gates.is_empty() || benchmark.split_policy.is_empty() {
        bail!("benchmark spec must include release gates and a split policy");
    }

    let mut baseline_ids = HashSet::new();
    for baseline in &benchmark.baselines {
        if baseline.id.trim().is_empty()
            || baseline.name.trim().is_empty()
            || baseline.role.trim().is_empty()
        {
            bail!("baseline ID, name, and role must be nonempty");
        }
        if !baseline_ids.insert(&baseline.id) {
            bail!("duplicate baseline ID: {}", baseline.id);
        }
    }

    let mut benchmark_ids = HashSet::new();
    for item in &benchmark.benchmarks {
        if !is_unique_id(&item.id, "FA-BM-") {
            bail!("invalid benchmark ID: {}", item.id);
        }
        if !benchmark_ids.insert(&item.id) {
            bail!("duplicate benchmark ID: {}", item.id);
        }
        for (field, value) in [
            ("name", item.name.as_str()),
            ("category", item.category.as_str()),
            ("measure", item.measure.as_str()),
            ("target", item.target.as_str()),
            ("gate", item.gate.as_str()),
            ("claim_ceiling", item.claim_ceiling.as_str()),
        ] {
            if value.trim().is_empty() {
                bail!("{}: {field} must not be empty", item.id);
            }
        }
    }
    if benchmark
        .release_gates
        .iter()
        .any(|gate| gate.trim().is_empty())
    {
        bail!("release gates must not contain empty entries");
    }
    if benchmark
        .split_policy
        .iter()
        .any(|(key, value)| key.trim().is_empty() || value.trim().is_empty())
    {
        bail!("split policy keys and values must be nonempty");
    }
    let _ = &benchmark.notes;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS_JSON: &str = include_str!("../../docs/research/first-aid/scenario-corpus-v0.json");
    const BENCHMARK_JSON: &str = include_str!("../../docs/research/first-aid/benchmark-v0.json");
    const SCHEMA_JSON: &str =
        include_str!("../../docs/research/first-aid/scenario-corpus-v0.schema.json");

    fn fixtures() -> (ScenarioCorpus, BenchmarkSpec, Value) {
        validate_documents(CORPUS_JSON, BENCHMARK_JSON, SCHEMA_JSON)
            .expect("checked-in contract fixtures should validate")
    }

    #[test]
    fn checked_in_contracts_are_consistent() {
        let (corpus, benchmark, _) = fixtures();
        assert_eq!(corpus.scenarios.len(), 24);
        assert_eq!(benchmark.benchmarks.len(), 18);
        assert_eq!(benchmark.status, "proposed_targets_unmeasured");
        assert!(!corpus.clinical_use_authorized);
        assert!(!benchmark.clinical_use_authorized);
    }

    #[test]
    fn duplicate_scenario_ids_are_rejected() {
        let (mut corpus, _, schema) = fixtures();
        corpus.scenarios[1].scenario_id = corpus.scenarios[0].scenario_id.clone();
        assert!(validate_corpus(&corpus, &schema).is_err());
    }

    #[test]
    fn clinical_authority_flag_is_rejected() {
        let (mut corpus, _, schema) = fixtures();
        corpus.clinical_use_authorized = true;
        assert!(validate_corpus(&corpus, &schema).is_err());
    }

    #[test]
    fn unknown_scenario_type_is_rejected() {
        let (mut corpus, _, schema) = fixtures();
        corpus.scenarios[0].scenario_type = "autonomous_treatment".to_string();
        assert!(validate_corpus(&corpus, &schema).is_err());
    }

    #[test]
    fn expired_bundle_cannot_use_render_disposition() {
        let (mut corpus, _, schema) = fixtures();
        let expired = corpus
            .scenarios
            .iter_mut()
            .find(|scenario| scenario.source_bundle_state == "expired")
            .expect("expired negative control exists");
        expired.expected_disposition = "render_approved_bundle_only".to_string();
        assert!(validate_corpus(&corpus, &schema).is_err());
    }

    #[test]
    fn sequential_process_benchmark_has_risk_weighted_release_gate() {
        let (_, benchmark, _) = fixtures();
        let item = benchmark
            .benchmarks
            .iter()
            .find(|item| item.id == "FA-BM-018")
            .expect("sequential process benchmark must be present");

        assert!(item.measure.contains("event traces"));
        assert!(item.target.contains("zero critical forbidden transitions"));
        assert!(
            item.gate
                .contains("Any critical forbidden transition blocks")
        );
        assert!(item.claim_ceiling.contains("Synthetic process-integrity"));
    }

    #[test]
    fn duplicate_benchmark_ids_are_rejected() {
        let (_, mut benchmark, _) = fixtures();
        benchmark.benchmarks[1].id = benchmark.benchmarks[0].id.clone();
        assert!(validate_benchmark(&benchmark).is_err());
    }

    #[test]
    fn model_only_baseline_cannot_change_status_ceiling() {
        let (_, mut benchmark, _) = fixtures();
        benchmark.status = "clinically_validated".to_string();
        assert!(validate_benchmark(&benchmark).is_err());
    }

    #[test]
    fn unknown_top_level_corpus_fields_are_rejected() {
        let mut value: Value = serde_json::from_str(CORPUS_JSON).expect("checked-in corpus parses");
        value["unexpected_authority"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ScenarioCorpus>(value).is_err());
    }

    #[test]
    fn unknown_benchmark_fields_are_rejected() {
        let mut value: Value =
            serde_json::from_str(BENCHMARK_JSON).expect("checked-in benchmark parses");
        value["clinically_validated"] = serde_json::json!(true);
        assert!(serde_json::from_value::<BenchmarkSpec>(value).is_err());
    }

    #[test]
    fn removing_the_positive_control_is_rejected() {
        let (mut corpus, _, schema) = fixtures();
        corpus
            .scenarios
            .retain(|scenario| scenario.scenario_type != "valid_bundle");
        assert!(validate_corpus(&corpus, &schema).is_err());
    }

    #[test]
    fn schema_cannot_enable_clinical_use() {
        let (corpus, _, mut schema) = fixtures();
        schema["properties"]["clinical_use_authorized"]["const"] = serde_json::json!(true);
        assert!(validate_schema_contract(&schema, &corpus).is_err());
    }

    #[test]
    fn source_revision_is_recorded_as_an_explicit_receipt_field() {
        let value = serde_json::to_value(ValidationReceipt {
            receipt_version: "first-aid-contract-receipt-v0",
            status: "contract_validation_passed",
            corpus_sha256: "c".repeat(64),
            benchmark_sha256: "b".repeat(64),
            schema_sha256: "s".repeat(64),
            source_revision_sha: "f".repeat(40),
            validator_source_sha256: "1".repeat(64),
            entrypoint_source_sha256: "2".repeat(64),
            executable_sha256: "3".repeat(64),
            runtime_os: "test-os".to_string(),
            runtime_arch: "test-arch".to_string(),
            scenario_count: 24,
            benchmark_count: 18,
            baseline_count: 6,
            claim_ceiling: "synthetic contract integrity only; no clinical validation",
        })
        .expect("receipt must serialize");
        assert_eq!(
            value["source_revision_sha"].as_str(),
            Some("ffffffffffffffffffffffffffffffffffffffff")
        );
        assert_eq!(value["benchmark_count"], 18);
    }
}
