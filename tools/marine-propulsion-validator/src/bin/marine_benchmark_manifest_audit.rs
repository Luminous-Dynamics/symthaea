use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Serialize)]
struct Diagnostic {
    path: String,
    code: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    benchmark_id: Option<String>,
    task_class: Option<String>,
    claim_stage: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    dataset_count: usize,
    baseline_count: usize,
    metric_count: usize,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Default)]
struct Findings {
    diagnostics: Vec<Diagnostic>,
    invalid: bool,
    incomplete: bool,
}

impl Findings {
    fn push(&mut self, path: impl Into<String>, code: &'static str, message: impl Into<String>) {
        match code {
            "INVALID" => self.invalid = true,
            "INCOMPLETE" => self.incomplete = true,
            _ => {}
        }
        self.diagnostics.push(Diagnostic {
            path: path.into(),
            code: code.to_owned(),
            message: message.into(),
        });
    }

    fn disposition(&self) -> &'static str {
        if self.invalid {
            "INVALID"
        } else if self.incomplete {
            "INCOMPLETE"
        } else {
            "MANIFEST_CONTRACT_VALID"
        }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
}

fn validate_schema(schema: &Value, manifest: &Value, findings: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        findings.push("/$schema", "INVALID", "only JSON Schema Draft 2020-12 is supported");
        return;
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        findings.push("/", "INVALID", "benchmark manifest schema does not validate against Draft 2020-12 meta-schema");
        return;
    }
    let validator = match jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(schema)
    {
        Ok(validator) => validator,
        Err(error) => {
            findings.push("/", "INVALID", format!("cannot compile benchmark manifest schema: {error}"));
            return;
        }
    };
    for error in validator.iter_errors(manifest) {
        findings.push(
            error.instance_path().to_string(),
            "INVALID",
            format!("manifest schema violation: {error}"),
        );
    }
}

fn collect_ids(
    array: Option<&[Value]>,
    id_field: &str,
    path: &str,
    findings: &mut Findings,
) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    let Some(items) = array else { return ids };
    for (index, item) in items.iter().enumerate() {
        match item.get(id_field).and_then(Value::as_str) {
            Some(id) if !ids.insert(id.to_owned()) => findings.push(
                format!("{path}/{index}/{id_field}"),
                "INVALID",
                format!("duplicate {id_field} '{id}'"),
            ),
            Some(_) => {}
            None => findings.push(
                format!("{path}/{index}/{id_field}"),
                "INVALID",
                format!("{id_field} is missing"),
            ),
        }
    }
    ids
}

fn check_semantics(manifest: &Value, findings: &mut Findings) {
    if manifest.get("schema_version").and_then(Value::as_str) != Some("0.1.0") {
        findings.push("/schema_version", "INVALID", "unsupported benchmark manifest version");
    }

    let source_registry = manifest.get("source_registry").and_then(Value::as_array);
    let datasets = manifest.get("datasets").and_then(Value::as_array);
    let baselines = manifest.get("baselines").and_then(Value::as_array);
    let metrics = manifest.get("metric_contracts").and_then(Value::as_array);
    let gates = manifest.get("hard_gates").and_then(Value::as_array);
    let protocol = manifest.get("evaluation_protocol").unwrap_or(&Value::Null);
    let task_class = manifest.get("task_class").and_then(Value::as_str).unwrap_or("");
    let claim_stage = manifest.get("claim_stage").and_then(Value::as_str).unwrap_or("");

    let source_ids = collect_ids(source_registry, "source_id", "/source_registry", findings);
    let _dataset_ids = collect_ids(datasets.map(Vec::as_slice), "dataset_id", "/datasets", findings);
    let _baseline_ids = collect_ids(baselines.map(Vec::as_slice), "baseline_id", "/baselines", findings);
    let _metric_ids = collect_ids(metrics.map(Vec::as_slice), "metric_id", "/metric_contracts", findings);
    let _gate_ids = collect_ids(gates.map(Vec::as_slice), "gate_id", "/hard_gates", findings);

    if datasets.is_none_or(Vec::is_empty) {
        findings.push("/datasets", "INVALID", "at least one dataset record is required");
    }
    if baselines.is_none_or(Vec::is_empty) {
        findings.push("/baselines", "INVALID", "at least one baseline is required");
    }
    if metrics.is_none_or(Vec::is_empty) {
        findings.push("/metric_contracts", "INVALID", "at least one metric contract is required");
    }
    if gates.is_none_or(Vec::is_empty) {
        findings.push("/hard_gates", "INVALID", "at least one non-compensable hard gate is required");
    }

    let mut any_synthetic = false;
    if let Some(datasets) = datasets {
        for (index, dataset) in datasets.iter().enumerate() {
            let path = format!("/datasets/{index}");
            if dataset.get("evidence_class").and_then(Value::as_str) == Some("synthetic") {
                any_synthetic = true;
                if claim_stage != "synthetic_smoke" {
                    findings.push(
                        format!("{path}/evidence_class"),
                        "INVALID",
                        "synthetic dataset cannot support a claim stage beyond synthetic_smoke",
                    );
                }
            }

            match dataset.get("rights_status").and_then(Value::as_str) {
                Some("unknown") => findings.push(
                    format!("{path}/rights_status"),
                    "INCOMPLETE",
                    "dataset rights/access status is unknown",
                ),
                Some(_) => {}
                None => findings.push(
                    format!("{path}/rights_status"),
                    "INVALID",
                    "dataset rights/access status must be explicit",
                ),
            }

            if let Some(refs) = dataset.get("source_ids").and_then(Value::as_array) {
                for (source_index, source) in refs.iter().enumerate() {
                    if let Some(source_id) = source.as_str() {
                        if !source_ids.contains(source_id) {
                            findings.push(
                                format!("{path}/source_ids/{source_index}"),
                                "INVALID",
                                format!("dataset source id '{source_id}' does not resolve in source_registry"),
                            );
                        }
                    }
                }
            }
        }
    }

    if any_synthetic && claim_stage != "synthetic_smoke" {
        findings.push(
            "/claim_stage",
            "INVALID",
            "a benchmark containing synthetic evidence cannot claim a physical or published-data stage",
        );
    }

    let routing_task = matches!(task_class, "voyage_routing" | "energy_management");
    if routing_task {
        if protocol.get("forecast_vintage_required").and_then(Value::as_bool) != Some(true) {
            findings.push(
                "/evaluation_protocol/forecast_vintage_required",
                "INVALID",
                "routing/energy-management benchmarks must require forecast vintage metadata",
            );
        }
        if protocol.get("operational_inputs_only").and_then(Value::as_bool) != Some(true) {
            findings.push(
                "/evaluation_protocol/operational_inputs_only",
                "INVALID",
                "deployable policies must receive only operationally available inputs",
            );
        }
        if protocol.get("future_observations_prohibited").and_then(Value::as_bool) != Some(true) {
            findings.push(
                "/evaluation_protocol/future_observations_prohibited",
                "INVALID",
                "future observation/reanalysis leakage must be explicitly prohibited",
            );
        }
        let split = protocol.get("split_strategy").and_then(Value::as_str);
        if !matches!(split, Some("grouped_voyage" | "grouped_route_season" | "leave_one_vessel_out" | "time_block_purged")) {
            findings.push(
                "/evaluation_protocol/split_strategy",
                "INVALID",
                "routing/energy-management benchmark must use grouped voyage/route/season/vessel or purged time-block holdouts",
            );
        }
    }

    if protocol.get("same_service_fingerprint_required").and_then(Value::as_bool) != Some(true) {
        findings.push(
            "/evaluation_protocol/same_service_fingerprint_required",
            "INVALID",
            "cross-policy comparisons must preserve the declared service fingerprint",
        );
    }

    if let Some(audits) = protocol.get("leakage_audits").and_then(Value::as_array) {
        for (index, audit) in audits.iter().enumerate() {
            let path = format!("/evaluation_protocol/leakage_audits/{index}");
            match audit.get("result").and_then(Value::as_str) {
                Some("pass") | Some("not_applicable") => {}
                Some("not_run") => findings.push(
                    format!("{path}/result"),
                    "INCOMPLETE",
                    "leakage audit has not been run; the manifest is not evidence of a leakage-free result",
                ),
                Some("fail") => findings.push(
                    format!("{path}/result"),
                    "INVALID",
                    "a declared leakage audit failed",
                ),
                _ => findings.push(
                    format!("{path}/result"),
                    "INVALID",
                    "leakage audit result is missing or unsupported",
                ),
            }
        }
    } else {
        findings.push(
            "/evaluation_protocol/leakage_audits",
            "INVALID",
            "explicit leakage audit records are required",
        );
    }

    if let Some(baselines) = baselines {
        let mut oracle_count = 0usize;
        for (index, baseline) in baselines.iter().enumerate() {
            let path = format!("/baselines/{index}");
            let oracle = baseline.get("oracle_only").and_then(Value::as_bool) == Some(true);
            let deployable = baseline.get("deployable").and_then(Value::as_bool) == Some(true);
            if oracle {
                oracle_count += 1;
                if deployable {
                    findings.push(
                        format!("{path}/deployable"),
                        "INVALID",
                        "oracle-only baseline cannot be marked deployable",
                    );
                }
            }
        }
        if routing_task && oracle_count == 0 {
            findings.push(
                "/baselines",
                "INCOMPLETE",
                "routing/energy-management benchmark should declare a clairvoyant or other explicitly bounded reference where one is applicable",
            );
        }
    }

    if let Some(gates) = gates {
        for (index, gate) in gates.iter().enumerate() {
            if gate.get("compensable").and_then(Value::as_bool) != Some(false) {
                findings.push(
                    format!("/hard_gates/{index}/compensable"),
                    "INVALID",
                    "hard gates must be non-compensable; a good energy score cannot offset a hard-gate failure",
                );
            }
        }
    }

    if let Some(sources) = source_registry {
        for (index, source) in sources.iter().enumerate() {
            if source.get("limitations").and_then(Value::as_array).is_none_or(Vec::is_empty) {
                findings.push(
                    format!("/source_registry/{index}/limitations"),
                    "INVALID",
                    "each source reference must retain at least one limitation/scope statement",
                );
            }
        }
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-benchmark-manifest-audit <schema.json> <manifest.json>".to_owned());
    }

    let schema = read_json(&args[1])?;
    let manifest = read_json(&args[2])?;
    let mut findings = Findings::default();
    validate_schema(&schema, &manifest, &mut findings);
    check_semantics(&manifest, &mut findings);

    let report = AuditReport {
        auditor: "marine-benchmark-manifest-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        benchmark_id: manifest.get("benchmark_id").and_then(Value::as_str).map(str::to_owned),
        task_class: manifest.get("task_class").and_then(Value::as_str).map(str::to_owned),
        claim_stage: manifest.get("claim_stage").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "Manifest schema, provenance/reference and protocol-integrity checks only. This report is not a benchmark result, independent reproduction, physical validation or regulatory approval.",
        dataset_count: manifest.get("datasets").and_then(Value::as_array).map_or(0, Vec::len),
        baseline_count: manifest.get("baselines").and_then(Value::as_array).map_or(0, Vec::len),
        metric_count: manifest.get("metric_contracts").and_then(Value::as_array).map_or(0, Vec::len),
        diagnostics: findings.diagnostics,
    };

    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.disposition == "MANIFEST_CONTRACT_VALID" { 0 } else { 2 })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BENCHMARK_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-benchmark-manifest-v0.1.0.schema.json");
    const SYNTHETIC_MANIFEST: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-benchmark-manifest-v0.1.0.json");

    fn schema_and_fixture() -> (Value, Value) {
        (
            serde_json::from_str(BENCHMARK_SCHEMA).expect("benchmark schema JSON"),
            serde_json::from_str(SYNTHETIC_MANIFEST).expect("benchmark manifest JSON"),
        )
    }

    #[test]
    fn benchmark_manifest_fixture_matches_draft_2020_12_schema() {
        let (schema, manifest) = schema_and_fixture();
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("benchmark schema compiles");
        assert!(validator.is_valid(&manifest));
    }

    #[test]
    fn synthetic_manifest_is_incomplete_not_a_physical_result() {
        let (_, manifest) = schema_and_fixture();
        let mut findings = Findings::default();
        check_semantics(&manifest, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
        assert!(findings.incomplete);
        assert_eq!(findings.disposition(), "INCOMPLETE");
    }

    #[test]
    fn dataset_source_references_must_resolve() {
        let (_, mut manifest) = schema_and_fixture();
        manifest["datasets"][0]["source_ids"][0] = serde_json::json!("MISSING-SOURCE");
        let mut findings = Findings::default();
        check_semantics(&manifest, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn synthetic_dataset_cannot_claim_sea_trial_stage() {
        let (_, mut manifest) = schema_and_fixture();
        manifest["claim_stage"] = serde_json::json!("sea_trial");
        let mut findings = Findings::default();
        check_semantics(&manifest, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn routing_benchmark_cannot_use_future_observations() {
        let (_, mut manifest) = schema_and_fixture();
        manifest["evaluation_protocol"]["future_observations_prohibited"] = serde_json::json!(false);
        let mut findings = Findings::default();
        check_semantics(&manifest, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn clairvoyant_oracle_cannot_be_called_deployable() {
        let (_, mut manifest) = schema_and_fixture();
        manifest["baselines"][3]["deployable"] = serde_json::json!(true);
        let mut findings = Findings::default();
        check_semantics(&manifest, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn hard_gates_are_non_compensable() {
        let (_, mut manifest) = schema_and_fixture();
        manifest["hard_gates"][0]["compensable"] = serde_json::json!(true);
        let mut findings = Findings::default();
        check_semantics(&manifest, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn duplicate_baseline_ids_are_rejected() {
        let (_, mut manifest) = schema_and_fixture();
        manifest["baselines"][1]["baseline_id"] = serde_json::json!("B0");
        let mut findings = Findings::default();
        check_semantics(&manifest, &mut findings);
        assert!(findings.invalid);
    }
}
