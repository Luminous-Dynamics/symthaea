use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-field-evidence-registry-entry/0.1";
const CORPUS_VERSION: &str = "ev-field-evidence-corpus/0.1";
const PROTOCOL_VERSION: &str = "ev-field-evidence-registry-entry/0.1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    corpus_version: String,
    schema_id: String,
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusCase {
    case_id: String,
    expectation: String,
    manifest: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    protocol_version: String,
    registry_snapshot: RegistrySnapshot,
    source: Source,
    rights: Rights,
    intended_use: IntendedUse,
    dataset: Dataset,
    adapter: Adapter,
    tasks: Vec<TaskEligibility>,
    artifact_policy: ArtifactPolicy,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistrySnapshot {
    registry_ref: String,
    snapshot_version: String,
    snapshot_digest: String,
    snapshot_at_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    dataset_id: String,
    title: String,
    release_version: String,
    publisher: String,
    source_uri: String,
    publication_uri: Option<String>,
    accessed_at_ms: u64,
    source_digest: String,
    license_class: String,
    license_ref: String,
    license_digest: String,
    access_mode: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Permission {
    status: String,
    evidence_ref: String,
    evidence_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Rights {
    access: Permission,
    local_processing: Permission,
    derived_data_publish: Permission,
    model_training: Permission,
    model_weights_release: Permission,
    commercial_use: Permission,
    raw_data_redistribution: Permission,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntendedUse {
    local_processing: bool,
    derived_data_publish: bool,
    model_training: bool,
    model_weights_release: bool,
    commercial_use: bool,
    raw_data_redistribution: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    application_types: Vec<String>,
    chemistries: Vec<String>,
    observation_start_ms: u64,
    observation_end_ms: u64,
    system_count: usize,
    unit_count: usize,
    reported_fault_count: usize,
    hierarchy_levels: Vec<String>,
    measurement_time_available: bool,
    receive_time_available: bool,
    timezone_basis: String,
    timestamp_mapping_digest: String,
    unit_mapping_digest: String,
    missingness_fraction: f64,
    missingness_mapping_digest: String,
    sentinel_handling: String,
    unlabeled_periods_are_negative: bool,
    censoring_supported: bool,
    topology_descriptor_digest: Option<String>,
    topology_qualification_digest: Option<String>,
    service_event_completion_available: bool,
    post_intervention_followup_available: bool,
    matched_comparison_available: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Adapter {
    adapter_ref: String,
    adapter_version: String,
    source_schema_digest: String,
    mapping_digest: String,
    validation_report_digest: String,
    validation_status: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskEligibility {
    task: String,
    status: String,
    evidence_class: String,
    label_source_ref: Option<String>,
    target_definition_digest: Option<String>,
    eligibility_evidence_digest: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactPolicy {
    registry_hosts_raw_data: bool,
    raw_data_copied_into_registry: bool,
    rights_lineage_required_for_derivatives: bool,
}

#[derive(Debug, Serialize)]
struct CaseAudit {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    supported_tasks: usize,
    partial_tasks: usize,
    not_supported_tasks: usize,
    permitted_intended_operations: usize,
    adapter_status: String,
}

#[derive(Debug, Serialize)]
struct ReportPayload {
    corpus_version: String,
    schema_id: String,
    schema_sha256: String,
    canonical_corpus_sha256: String,
    case_count: usize,
    accepted_count: usize,
    rejected_count: usize,
    metadata_only_count: usize,
    noncommercial_only_count: usize,
    cases: Vec<CaseAudit>,
}

#[derive(Debug, Serialize)]
pub struct CorpusReport {
    #[serde(flatten)]
    payload: ReportPayload,
    report_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn schema_validator(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema identity does not match field evidence registry v0.1".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn validate_permission(permission: &Permission, label: &str, errors: &mut Vec<String>) {
    if !matches!(
        permission.status.as_str(),
        "permitted" | "prohibited" | "conditional" | "unknown"
    ) || !nonempty(&permission.evidence_ref)
        || !is_digest(&permission.evidence_digest)
    {
        errors.push(format!("{label} permission lacks an explicit, evidence-bound status"));
    }
}

fn require_permission(
    permission: &Permission,
    operation: &str,
    intended: bool,
    errors: &mut Vec<String>,
) {
    if intended && permission.status != "permitted" {
        errors.push(format!(
            "planned {operation} requires explicit permission; current status is {}",
            permission.status
        ));
    }
}

fn validate_manifest(manifest: &Manifest) -> (Vec<String>, usize, usize, usize, usize) {
    let mut errors = Vec::new();
    if manifest.protocol_version != PROTOCOL_VERSION {
        errors.push("protocol version is invalid".to_owned());
    }

    let snapshot = &manifest.registry_snapshot;
    if !nonempty(&snapshot.registry_ref)
        || !nonempty(&snapshot.snapshot_version)
        || !is_digest(&snapshot.snapshot_digest)
        || snapshot.snapshot_at_ms == 0
    {
        errors.push("registry snapshot identity or timestamp is invalid".to_owned());
    }

    let source = &manifest.source;
    if !nonempty(&source.dataset_id)
        || !nonempty(&source.title)
        || !nonempty(&source.release_version)
        || !nonempty(&source.publisher)
        || !nonempty(&source.source_uri)
        || source.accessed_at_ms == 0
        || !is_digest(&source.source_digest)
        || !nonempty(&source.license_ref)
        || !is_digest(&source.license_digest)
    {
        errors.push("source identity, retrieval or licence provenance is incomplete".to_owned());
    }
    if !matches!(
        source.license_class.as_str(),
        "open_license" | "terms_of_use" | "restricted" | "no_explicit_license" | "unknown"
    ) {
        errors.push("unknown source licence classification".to_owned());
    }
    if !matches!(
        source.access_mode.as_str(),
        "public_download" | "controlled_access" | "metadata_only" | "licensed_delivery"
    ) {
        errors.push("unknown source access mode".to_owned());
    }

    let rights = &manifest.rights;
    for (label, permission) in [
        ("access", &rights.access),
        ("local_processing", &rights.local_processing),
        ("derived_data_publish", &rights.derived_data_publish),
        ("model_training", &rights.model_training),
        ("model_weights_release", &rights.model_weights_release),
        ("commercial_use", &rights.commercial_use),
        ("raw_data_redistribution", &rights.raw_data_redistribution),
    ] {
        validate_permission(permission, label, &mut errors);
    }

    let intended = &manifest.intended_use;
    if source.access_mode == "metadata_only" {
        if intended.local_processing
            || intended.derived_data_publish
            || intended.model_training
            || intended.model_weights_release
            || intended.commercial_use
            || intended.raw_data_redistribution
        {
            errors.push("metadata-only registry records cannot authorize raw-data processing or modelling".to_owned());
        }
    } else {
        require_permission(&rights.access, "source access", true, &mut errors);
    }
    require_permission(
        &rights.local_processing,
        "local processing",
        intended.local_processing,
        &mut errors,
    );
    require_permission(
        &rights.derived_data_publish,
        "derived-data publication",
        intended.derived_data_publish,
        &mut errors,
    );
    require_permission(
        &rights.model_training,
        "model training",
        intended.model_training,
        &mut errors,
    );
    require_permission(
        &rights.model_weights_release,
        "model-weight release",
        intended.model_weights_release,
        &mut errors,
    );
    require_permission(
        &rights.commercial_use,
        "commercial use",
        intended.commercial_use,
        &mut errors,
    );
    require_permission(
        &rights.raw_data_redistribution,
        "raw-data redistribution",
        intended.raw_data_redistribution,
        &mut errors,
    );

    if intended.model_weights_release && !intended.model_training {
        errors.push("model weights cannot be released when model training is not an intended use".to_owned());
    }
    if intended.commercial_use && !intended.local_processing {
        errors.push("commercial use requires local processing to be declared".to_owned());
    }
    if intended.raw_data_redistribution && !intended.local_processing {
        errors.push("raw-data redistribution requires local processing to be declared".to_owned());
    }

    let dataset = &manifest.dataset;
    if dataset.observation_start_ms == 0
        || dataset.observation_end_ms < dataset.observation_start_ms
        || dataset.application_types.is_empty()
        || dataset.chemistries.is_empty()
        || dataset.hierarchy_levels.is_empty()
        || !matches!(
            dataset.timezone_basis.as_str(),
            "UTC" | "source_offset_preserved" | "unknown"
        )
        || !is_digest(&dataset.timestamp_mapping_digest)
        || !is_digest(&dataset.unit_mapping_digest)
        || !is_digest(&dataset.missingness_mapping_digest)
        || !dataset.missingness_fraction.is_finite()
        || !(0.0..=1.0).contains(&dataset.missingness_fraction)
    {
        errors.push("dataset observation, hierarchy, timestamp, unit or missingness metadata is invalid".to_owned());
    }
    if dataset.unlabeled_periods_are_negative {
        errors.push("unlabeled periods must never be converted into negative fault labels".to_owned());
    }
    if dataset.sentinel_handling == "collapsed_to_zero" {
        errors.push("source sentinel values cannot be collapsed into valid numeric zero".to_owned());
    }

    let adapter = &manifest.adapter;
    if !nonempty(&adapter.adapter_ref)
        || !nonempty(&adapter.adapter_version)
        || !is_digest(&adapter.source_schema_digest)
        || !is_digest(&adapter.mapping_digest)
        || !is_digest(&adapter.validation_report_digest)
        || !matches!(
            adapter.validation_status.as_str(),
            "validated" | "partial" | "unvalidated"
        )
    {
        errors.push("adapter identity, mapping or validation receipt is invalid".to_owned());
    }

    let expected_tasks = [
        "sensor_data_integrity",
        "operational_fault_detection",
        "soh_estimation",
        "remaining_life",
        "intervention_outcomes",
        "pack_capability",
    ];
    let mut task_ids = BTreeSet::new();
    let mut supported_tasks = 0usize;
    let mut partial_tasks = 0usize;
    let mut not_supported_tasks = 0usize;
    for task in &manifest.tasks {
        if !task_ids.insert(task.task.clone())
            || !nonempty(&task.reason)
            || !is_digest(&task.eligibility_evidence_digest)
        {
            errors.push("task eligibility has a duplicate identity or incomplete evidence".to_owned());
        }
        match task.status.as_str() {
            "SUPPORTED" => {
                supported_tasks += 1;
                if !matches!(
                    task.evidence_class.as_str(),
                    "physical_observation" | "derived_from_physical" | "simulation" | "vendor_claim"
                ) || task.label_source_ref.as_deref().is_none_or(|value| !nonempty(value))
                    || task.target_definition_digest.as_deref().is_none_or(|value| !is_digest(value))
                {
                    errors.push(format!(
                        "supported task {} requires a label source, target definition and non-synthetic evidence class",
                        task.task
                    ));
                }
                if adapter.validation_status != "validated" {
                    errors.push(format!(
                        "task {} cannot be fully supported by a partial or unvalidated adapter",
                        task.task
                    ));
                }
                match task.task.as_str() {
                    "sensor_data_integrity"
                        if !dataset.measurement_time_available
                            || !dataset.receive_time_available
                            || dataset.timezone_basis == "unknown"
                            || dataset.sentinel_handling == "unmapped" =>
                    {
                        errors.push("sensor integrity support requires mapped timestamps, timezone semantics and sentinel handling".to_owned());
                    }
                    "operational_fault_detection"
                        if dataset.reported_fault_count == 0
                            || !dataset.measurement_time_available
                            || !dataset.receive_time_available
                            || dataset.timezone_basis == "unknown"
                            || dataset.sentinel_handling == "unmapped" =>
                    {
                        errors.push("fault detection support requires labelled fault events, mapped event/receive times and sentinel semantics".to_owned());
                    }
                    "soh_estimation" => {
                        if !dataset.hierarchy_levels.iter().any(|level| level == "cell")
                            || dataset.unit_count == 0
                        {
                            errors.push("SOH support requires non-empty cell-level identity".to_owned());
                        }
                    }
                    "remaining_life" if !dataset.censoring_supported || dataset.unit_count == 0 => {
                        errors.push("remaining-life support requires a censoring-aware observation contract and non-empty unit population".to_owned());
                    }
                    "intervention_outcomes"
                        if !dataset.service_event_completion_available
                            || !dataset.post_intervention_followup_available
                            || !dataset.matched_comparison_available =>
                    {
                        errors.push("intervention outcomes require verified service completion, post-service follow-up and a matched comparison design".to_owned());
                    }
                    "pack_capability" => {
                        let levels = &dataset.hierarchy_levels;
                        if !levels.iter().any(|level| level == "cell")
                            || !levels.iter().any(|level| level == "module")
                            || !levels.iter().any(|level| level == "pack")
                            || dataset.topology_descriptor_digest.as_deref().is_none_or(|digest| !is_digest(digest))
                            || dataset.topology_qualification_digest.as_deref().is_none_or(|digest| !is_digest(digest))
                        {
                            errors.push("pack capability support requires cell/module/pack hierarchy plus qualified topology evidence".to_owned());
                        }
                    }
                    _ => {}
                }
            }
            "PARTIAL" => {
                partial_tasks += 1;
                if !matches!(
                    task.evidence_class.as_str(),
                    "physical_observation" | "derived_from_physical" | "simulation" | "vendor_claim" | "none"
                ) {
                    errors.push(format!("partial task {} has an invalid evidence class", task.task));
                }
            }
            "NOT_SUPPORTED" => {
                not_supported_tasks += 1;
                if task.evidence_class != "none"
                    || task.label_source_ref.is_some()
                    || task.target_definition_digest.is_some()
                {
                    errors.push(format!(
                        "not-supported task {} must not carry a target or label claim",
                        task.task
                    ));
                }
            }
            _ => errors.push(format!("task {} has an unknown support status", task.task)),
        }
    }
    let expected: BTreeSet<String> = expected_tasks.iter().map(|task| (*task).to_owned()).collect();
    if task_ids != expected {
        errors.push("task matrix must contain exactly one entry for each of the six canonical task families".to_owned());
    }

    if manifest.artifact_policy.registry_hosts_raw_data
        || manifest.artifact_policy.raw_data_copied_into_registry
        || !manifest.artifact_policy.rights_lineage_required_for_derivatives
    {
        errors.push("registry must remain metadata-only and require rights lineage for derived artifacts".to_owned());
    }

    let permitted_intended_operations = [
        intended.local_processing,
        intended.derived_data_publish,
        intended.model_training,
        intended.model_weights_release,
        intended.commercial_use,
        intended.raw_data_redistribution,
    ]
    .into_iter()
    .filter(|value| *value)
    .count();

    (
        errors,
        supported_tasks,
        partial_tasks,
        not_supported_tasks,
        permitted_intended_operations,
    )
}

fn canonicalize_corpus(value: &mut Value) {
    let Some(cases) = value.get_mut("cases").and_then(Value::as_array_mut) else {
        return;
    };
    cases.sort_by(|left, right| {
        left.get("case_id")
            .and_then(Value::as_str)
            .cmp(&right.get("case_id").and_then(Value::as_str))
    });
    for case in cases {
        if let Some(manifest) = case.get_mut("manifest") {
            if let Some(tasks) = manifest.get_mut("tasks").and_then(Value::as_array_mut) {
                tasks.sort_by(|left, right| {
                    left.get("task")
                        .and_then(Value::as_str)
                        .cmp(&right.get("task").and_then(Value::as_str))
                });
            }
            for field in ["application_types", "chemistries", "hierarchy_levels"] {
                if let Some(rows) = manifest
                    .get_mut("dataset")
                    .and_then(|dataset| dataset.get_mut(field))
                    .and_then(Value::as_array_mut)
                {
                    rows.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
                }
            }
        }
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<CorpusReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let mut corpus_value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    canonicalize_corpus(&mut corpus_value);
    let canonical = serde_json::to_vec(&corpus_value)
        .map_err(|error| format!("canonical corpus serialization: {error}"))?;
    let corpus: Corpus =
        serde_json::from_slice(&canonical).map_err(|error| format!("typed corpus: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.schema_id != SCHEMA_ID {
        return Err("corpus version or schema identity mismatch".to_owned());
    }

    let mut seen = BTreeSet::new();
    let mut cases = Vec::new();
    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;
    let mut metadata_only_count = 0usize;
    let mut noncommercial_only_count = 0usize;
    for case in corpus.cases {
        if !nonempty(&case.case_id) || !seen.insert(case.case_id.clone()) {
            return Err(format!("case identity is empty or duplicated: {}", case.case_id));
        }
        if !matches!(case.expectation.as_str(), "accept" | "reject") {
            return Err(format!("case {} has unknown expectation", case.case_id));
        }
        let schema_errors: Vec<String> = validator
            .iter_errors(&case.manifest)
            .map(|error| error.to_string())
            .collect();
        let outcome = if schema_errors.is_empty() {
            match serde_json::from_value::<Manifest>(case.manifest) {
                Ok(manifest) => {
                    let (errors, supported, partial, not_supported, permitted) =
                        validate_manifest(&manifest);
                    if errors.is_empty() {
                        Ok((
                            supported,
                            partial,
                            not_supported,
                            permitted,
                            manifest.source.access_mode.clone(),
                            manifest.intended_use.commercial_use,
                        ))
                    } else {
                        Err(errors)
                    }
                }
                Err(error) => Err(vec![format!("typed manifest parse: {error}")]),
            }
        } else {
            Err(schema_errors)
        };
        let accepted = outcome.is_ok();
        if accepted != (case.expectation == "accept") {
            let detail = match &outcome {
                Ok(_) => "unexpected acceptance".to_owned(),
                Err(errors) => errors.join("; "),
            };
            return Err(format!("case {} expectation mismatch: {}", case.case_id, detail));
        }
        let (supported, partial, not_supported, permitted, access_mode, commercial_use, errors) =
            match outcome {
                Ok((supported, partial, not_supported, permitted, access_mode, commercial_use)) => {
                    if access_mode == "metadata_only" {
                        metadata_only_count += 1;
                    }
                    if !commercial_use {
                        noncommercial_only_count += 1;
                    }
                    accepted_count += 1;
                    (
                        supported,
                        partial,
                        not_supported,
                        permitted,
                        access_mode,
                        commercial_use,
                        Vec::new(),
                    )
                }
                Err(errors) => {
                    rejected_count += 1;
                    (0, 0, 0, 0, "rejected".to_owned(), false, errors)
                }
            };
        cases.push(CaseAudit {
            case_id: case.case_id,
            expectation: case.expectation,
            accepted,
            errors,
            supported_tasks: supported,
            partial_tasks: partial,
            not_supported_tasks: not_supported,
            permitted_intended_operations: permitted,
            adapter_status: access_mode,
        });
    }

    let payload = ReportPayload {
        corpus_version: CORPUS_VERSION.to_owned(),
        schema_id: SCHEMA_ID.to_owned(),
        schema_sha256: sha256(schema_bytes),
        canonical_corpus_sha256: sha256(&canonical),
        case_count: cases.len(),
        accepted_count,
        rejected_count,
        metadata_only_count,
        noncommercial_only_count,
        cases,
    };
    let bytes =
        serde_json::to_vec(&payload).map_err(|error| format!("report serialization: {error}"))?;
    Ok(CorpusReport {
        payload,
        report_sha256: sha256(&bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_field_evidence_registry_entry_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/field_evidence_registry_scenarios_v0_1.json");

    fn report() -> CorpusReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen field evidence registry corpus")
    }

    #[test]
    fn frozen_corpus_accepts_five_and_rejects_eight_controls() {
        let report = report();
        assert_eq!(report.payload.case_count, 13);
        assert_eq!(report.payload.accepted_count, 5);
        assert_eq!(report.payload.rejected_count, 8);
    }

    #[test]
    fn metadata_only_registry_entry_does_not_claim_data_or_task_access() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "metadata-only-no-data-rights")
            .expect("metadata-only case");
        assert!(case.accepted);
        assert_eq!(case.supported_tasks, 0);
        assert_eq!(case.not_supported_tasks, 6);
        assert_eq!(case.adapter_status, "metadata_only");
    }

    #[test]
    fn noncommercial_terms_allow_research_processing_but_not_commercial_use() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "noncommercial-controlled")
            .expect("noncommercial rights case");
        assert!(case.accepted);
        assert_eq!(case.permitted_intended_operations, 2);
    }

    #[test]
    fn licence_permission_is_checked_per_operation() {
        let report = report();
        for case_id in [
            "unknown-rights-training-rejected",
            "commercial-not-permitted-rejected",
            "raw-redistribution-prohibited-rejected",
            "conditional-rights-used-without-clearance-rejected",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("rights negative control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn unlabeled_fault_periods_and_sentinel_values_are_never_silently_rewritten() {
        let report = report();
        for case_id in [
            "unlabeled-periods-as-negative-rejected",
            "sentinel-collapsed-to-zero-rejected",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("field data semantics negative control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn task_eligibility_requires_reference_and_application_evidence() {
        let report = report();
        for case_id in [
            "soh-supported-without-reference-label-rejected",
            "fault-detection-without-event-time-rejected",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("task eligibility negative control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn partial_adapters_must_not_overclaim_supported_tasks() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "partial-adapter-declared-partial")
            .expect("partial adapter case");
        assert!(case.accepted);
        assert_eq!(case.supported_tasks, 2);
        assert_eq!(case.partial_tasks, 1);
    }

    #[test]
    fn pack_capability_and_intervention_outcomes_require_distinct_physical_evidence() {
        let report = report();
        let qualified_pack = report.payload.cases.iter()
            .find(|case| case.case_id == "qualified-pack-capability-task-supported")
            .expect("qualified topology case");
        assert!(qualified_pack.accepted);
        for case_id in [
            "pack-capability-supported-without-topology-rejected",
            "intervention-supported-without-followup-rejected",
            "unmapped-sentinel-with-supported-task-rejected",
            "unknown-timezone-with-fault-support-rejected",
        ] {
            let case = report.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("task prerequisite negative control");
            assert!(!case.accepted);
        }
    }

    #[test]
    fn partial_adapter_does_not_claim_fully_supported_tasks() {
        let report = report();
        let case = report.payload.cases.iter()
            .find(|case| case.case_id == "partial-adapter-declared-partial")
            .expect("partial adapter case");
        assert!(case.accepted);
        assert_eq!(case.supported_tasks, 0);
        assert_eq!(case.partial_tasks, 3);
        assert_eq!(case.not_supported_tasks, 3);
    }

    #[test]
    fn canonical_report_is_invariant_to_case_task_and_hierarchy_order() {
        let original = report();
        let mut value: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        value["cases"].as_array_mut().unwrap().reverse();
        for case in value["cases"].as_array_mut().unwrap() {
            case["manifest"]["tasks"].as_array_mut().unwrap().reverse();
            for field in ["application_types", "chemistries", "hierarchy_levels"] {
                case["manifest"]["dataset"][field].as_array_mut().unwrap().reverse();
            }
        }
        let bytes = serde_json::to_vec(&value).expect("permuted field corpus");
        let replay = audit_corpus(SCHEMA, &bytes).expect("permuted field evidence audit");
        assert_eq!(original.report_sha256, replay.report_sha256);
        assert_eq!(
            original.payload.canonical_corpus_sha256,
            replay.payload.canonical_corpus_sha256
        );
    }
}
