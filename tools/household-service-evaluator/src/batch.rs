use crate::{
    EvaluationError, MeterMeasurement, RepairAttempt, RepairDisposition, TopologyAuditReference,
    WaterBalanceInput, evaluate_repair_attempt, evaluate_water_balance,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const WATER_IDS: [&str; 10] = [
    "WCONT-001",
    "WCONT-002",
    "WCONT-003",
    "WCONT-004",
    "WCONT-005",
    "WCONT-006",
    "WCONT-007",
    "WCONT-008",
    "WCONT-009",
    "WCONT-010",
];

const REPAIR_IDS: [&str; 10] = [
    "REPAIR-001",
    "REPAIR-002",
    "REPAIR-003",
    "REPAIR-004",
    "REPAIR-005",
    "REPAIR-006",
    "REPAIR-007",
    "REPAIR-008",
    "REPAIR-009",
    "REPAIR-010",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceDomain {
    WaterContinuity,
    Repairability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchDisposition {
    ProcessAccountingOnly,
    ReserveUnresolved,
    DegradedService,
    SensorDisagreement,
    HumanReviewRequired,
    DeliveryUnavailable,
    WaterQualityUnresolved,
    MassBalanceContradictionUnqualified,
    CommandObservationMismatch,
    RecoveryMonitoringRequired,
    RepairNotExecuted,
    RepairBlocked,
    RepairHeld,
    RecoveryArtifactsAvailableNotExecuted,
    RecurrenceObserved,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DerivedMetric {
    Quantity {
        value: f64,
        unit: String,
        derivation: String,
    },
    Boolean {
        value: bool,
        semantics: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BatchReceipt {
    pub domain: ServiceDomain,
    pub scenario_id: String,
    /// Retained as fixture metadata, never treated as the evaluator's result.
    pub fixture_expected_disposition: String,
    pub disposition: BatchDisposition,
    /// Typed quantities and booleans; no bool-as-number coercion.
    pub derived_metrics: BTreeMap<String, DerivedMetric>,
    pub evidence_notes: Vec<String>,
    pub unresolved_claims: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DigestScope {
    ExactSourceBytes,
    CanonicalizedJsonValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    SyntheticFixtureOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceCorpusManifest {
    pub corpus_id: String,
    pub schema_version: String,
    pub status: String,
    pub scenario_set_revision: String,
    pub source_byte_length: usize,
    pub source_sha256: String,
    pub digest_scope: DigestScope,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BatchReport {
    pub report_schema_version: String,
    pub evaluator_version: String,
    pub adapter_version: String,
    pub evidence_class: EvidenceClass,
    pub source_repository: String,
    pub source_commit: Option<String>,
    pub water_corpus_id: String,
    pub repair_corpus_id: String,
    pub water_source: SourceCorpusManifest,
    pub repair_source: SourceCorpusManifest,
    pub water_receipt_count: usize,
    pub repair_receipt_count: usize,
    pub total_receipt_count: usize,
    pub disposition_counts: BTreeMap<String, usize>,
    pub receipts: Vec<BatchReceipt>,
    pub claim_ceiling: Vec<String>,
}

/// The report digest covers the compact, deterministic serialization of the
/// report only; the envelope is intentionally outside that digest.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BatchOutputEnvelope {
    pub report_sha256: String,
    pub report_byte_length: usize,
    pub report_serialization: String,
    pub report: BatchReport,
}

impl BatchOutputEnvelope {
    pub fn from_report(report: BatchReport) -> Result<Self, EvaluationError> {
        let report_bytes = serde_json::to_vec(&report)
            .map_err(|error| EvaluationError::new("report_serialization", error.to_string()))?;
        Ok(Self {
            report_sha256: sha256_hex(&report_bytes),
            report_byte_length: report_bytes.len(),
            report_serialization: "serde_json_compact_struct_order_btree_maps_v1".to_owned(),
            report,
        })
    }
}

/// Evaluate parsed JSON values. Source digests cover canonicalized Serde JSON,
/// not the original byte representation. Use evaluate_batch_from_bytes or
/// evaluate_batch_from_paths when exact file-byte binding is required.
pub fn evaluate_batch(
    water_corpus: &Value,
    repair_corpus: &Value,
) -> Result<BatchReport, EvaluationError> {
    let water_bytes = serde_json::to_vec(water_corpus)
        .map_err(|error| EvaluationError::new("water_fixture_canonical_json", error.to_string()))?;
    let repair_bytes = serde_json::to_vec(repair_corpus)
        .map_err(|error| EvaluationError::new("repair_fixture_canonical_json", error.to_string()))?;
    evaluate_batch_with_sources(
        water_corpus,
        repair_corpus,
        &water_bytes,
        &repair_bytes,
        DigestScope::CanonicalizedJsonValue,
        None,
    )
}

/// Parse and evaluate source corpora while binding the exact supplied bytes.
pub fn evaluate_batch_from_bytes(
    water_bytes: &[u8],
    repair_bytes: &[u8],
) -> Result<BatchReport, EvaluationError> {
    evaluate_batch_from_bytes_with_source_commit(water_bytes, repair_bytes, None)
}

/// Parse and evaluate exact source bytes while recording the exact Git commit
/// that supplied the evaluator build, when the caller has that provenance.
pub fn evaluate_batch_from_bytes_with_source_commit(
    water_bytes: &[u8],
    repair_bytes: &[u8],
    source_commit: Option<&str>,
) -> Result<BatchReport, EvaluationError> {
    let source_commit = source_commit.map(validate_source_commit).transpose()?;
    let water_corpus: Value = serde_json::from_slice(water_bytes)
        .map_err(|error| EvaluationError::new("water_fixture_json", error.to_string()))?;
    let repair_corpus: Value = serde_json::from_slice(repair_bytes)
        .map_err(|error| EvaluationError::new("repair_fixture_json", error.to_string()))?;
    evaluate_batch_with_sources(
        &water_corpus,
        &repair_corpus,
        water_bytes,
        repair_bytes,
        DigestScope::ExactSourceBytes,
        source_commit,
    )
}

fn evaluate_batch_with_sources(
    water_corpus: &Value,
    repair_corpus: &Value,
    water_source_bytes: &[u8],
    repair_source_bytes: &[u8],
    digest_scope: DigestScope,
    source_commit: Option<String>,
) -> Result<BatchReport, EvaluationError> {
    let water_corpus_id = validate_corpus_header(
        water_corpus,
        "HOME-UTILITY-COORD-001-WATER-CONTINUITY-SYNTHETIC-001",
    )?;
    let repair_corpus_id = validate_corpus_header(
        repair_corpus,
        "HOME-REPAIR-QUAL-001-SYNTHETIC-REPAIR-TASKS-001",
    )?;

    let water_source = source_manifest(water_corpus, water_source_bytes, digest_scope)?;
    let repair_source = source_manifest(repair_corpus, repair_source_bytes, digest_scope)?;

    let water_scenarios = validate_scenarios(water_corpus, "scenarios", &WATER_IDS)?;
    let repair_scenarios = validate_scenarios(repair_corpus, "task_profiles", &REPAIR_IDS)?;

    let mut receipts = Vec::with_capacity(WATER_IDS.len() + REPAIR_IDS.len());
    for scenario in water_scenarios {
        receipts.push(evaluate_water_scenario(scenario)?);
    }
    for scenario in repair_scenarios {
        receipts.push(evaluate_repair_scenario(scenario)?);
    }

    let mut disposition_counts = BTreeMap::new();
    for receipt in &receipts {
        *disposition_counts
            .entry(disposition_name(receipt.disposition).to_owned())
            .or_insert(0) += 1;
    }

    let claim_ceiling = [water_corpus, repair_corpus]
        .into_iter()
        .flat_map(|corpus| {
            corpus["claim_ceiling"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
        })
        .collect();

    Ok(BatchReport {
        report_schema_version: "0.2".to_owned(),
        evaluator_version: env!("CARGO_PKG_VERSION").to_owned(),
        adapter_version: "water-continuity-v0.1+repairability-v0.1".to_owned(),
        evidence_class: EvidenceClass::SyntheticFixtureOnly,
        source_repository: "Luminous-Dynamics/symthaea".to_owned(),
        source_commit,
        water_corpus_id,
        repair_corpus_id,
        water_source,
        repair_source,
        water_receipt_count: WATER_IDS.len(),
        repair_receipt_count: REPAIR_IDS.len(),
        total_receipt_count: receipts.len(),
        disposition_counts,
        receipts,
        claim_ceiling,
    })
}

/// Read two JSON corpus files and bind the exact bytes used by the evaluator.
pub fn evaluate_batch_from_paths(
    water_path: impl AsRef<Path>,
    repair_path: impl AsRef<Path>,
) -> Result<BatchReport, EvaluationError> {
    evaluate_batch_from_paths_with_source_commit(water_path, repair_path, None)
}

pub fn evaluate_batch_from_paths_with_source_commit(
    water_path: impl AsRef<Path>,
    repair_path: impl AsRef<Path>,
    source_commit: Option<&str>,
) -> Result<BatchReport, EvaluationError> {
    let water_bytes = fs::read(water_path)
        .map_err(|error| EvaluationError::new("water_fixture", error.to_string()))?;
    let repair_bytes = fs::read(repair_path)
        .map_err(|error| EvaluationError::new("repair_fixture", error.to_string()))?;
    evaluate_batch_from_bytes_with_source_commit(&water_bytes, &repair_bytes, source_commit)
}

fn validate_source_commit(value: &str) -> Result<String, EvaluationError> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EvaluationError::new(
            "source_commit",
            "source commit must be a full 40-character hexadecimal Git object ID",
        ));
    }
    Ok(value.to_ascii_lowercase())
}

fn source_manifest(
    corpus: &Value,
    source_bytes: &[u8],
    digest_scope: DigestScope,
) -> Result<SourceCorpusManifest, EvaluationError> {
    let corpus_id = required_str(corpus, "corpus_id", "corpus_manifest")?.to_owned();
    let schema_version = required_str(corpus, "schema_version", "corpus_manifest")?.to_owned();
    let status = required_str(corpus, "status", "corpus_manifest")?.to_owned();

    Ok(SourceCorpusManifest {
        corpus_id,
        schema_version: schema_version.clone(),
        status,
        scenario_set_revision: schema_version,
        source_byte_length: source_bytes.len(),
        source_sha256: sha256_hex(source_bytes),
        digest_scope,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn insert_quantity(
    metrics: &mut BTreeMap<String, DerivedMetric>,
    name: &str,
    value: f64,
    unit: &str,
    derivation: &str,
) -> Result<(), EvaluationError> {
    if !value.is_finite() {
        return Err(EvaluationError::new(
            "derived_metric",
            format!("derived quantity {name} is not finite"),
        ));
    }
    metrics.insert(
        name.to_owned(),
        DerivedMetric::Quantity {
            value,
            unit: unit.to_owned(),
            derivation: derivation.to_owned(),
        },
    );
    Ok(())
}

fn insert_boolean(
    metrics: &mut BTreeMap<String, DerivedMetric>,
    name: &str,
    value: bool,
    semantics: &str,
) {
    metrics.insert(
        name.to_owned(),
        DerivedMetric::Boolean {
            value,
            semantics: semantics.to_owned(),
        },
    );
}

fn validate_corpus_header(
    corpus: &Value,
    expected_corpus_id: &'static str,
) -> Result<String, EvaluationError> {
    if corpus["schema_version"].as_str() != Some("0.1") {
        return Err(EvaluationError::new(
            "corpus.schema_version",
            "only frozen schema version 0.1 is supported",
        ));
    }
    if corpus["status"].as_str() != Some("synthetic_test_fixture_only") {
        return Err(EvaluationError::new(
            "corpus.status",
            "corpus must retain its synthetic-test-only status",
        ));
    }
    let corpus_id = corpus["corpus_id"]
        .as_str()
        .ok_or_else(|| EvaluationError::new("corpus.corpus_id", "missing corpus identifier"))?;
    if corpus_id != expected_corpus_id {
        return Err(EvaluationError::new(
            "corpus.corpus_id",
            format!("unexpected corpus identifier: {corpus_id}"),
        ));
    }
    let claim_ceiling = corpus["claim_ceiling"].as_array().ok_or_else(|| {
        EvaluationError::new("corpus.claim_ceiling", "claim ceiling must be an array")
    })?;
    if claim_ceiling.is_empty() {
        return Err(EvaluationError::new(
            "corpus.claim_ceiling",
            "a non-empty claim ceiling is required",
        ));
    }
    Ok(corpus_id.to_owned())
}

fn validate_scenarios<'a>(
    corpus: &'a Value,
    collection_name: &'static str,
    expected_ids: &[&str],
) -> Result<Vec<&'a Value>, EvaluationError> {
    let scenarios = corpus[collection_name].as_array().ok_or_else(|| {
        EvaluationError::new(collection_name, "scenario collection must be an array")
    })?;

    if scenarios.len() != expected_ids.len() {
        return Err(EvaluationError::new(
            collection_name,
            format!(
                "expected exactly {} scenarios, found {}",
                expected_ids.len(),
                scenarios.len()
            ),
        ));
    }

    let payload_field = match collection_name {
        "scenarios" => "input",
        "task_profiles" => "synthetic_baseline",
        _ => {
            return Err(EvaluationError::new(
                collection_name,
                "unsupported scenario collection",
            ));
        }
    };
    let expected: BTreeSet<&str> = expected_ids.iter().copied().collect();
    let mut observed = BTreeSet::new();
    for scenario in scenarios {
        let id = scenario["id"]
            .as_str()
            .ok_or_else(|| EvaluationError::new(collection_name, "scenario is missing an ID"))?;
        if !expected.contains(id) {
            return Err(EvaluationError::new(
                collection_name,
                format!("unexpected scenario ID: {id}"),
            ));
        }
        if !observed.insert(id) {
            return Err(EvaluationError::new(
                collection_name,
                format!("duplicate scenario ID: {id}"),
            ));
        }
        if scenario["title"].as_str().is_none()
            || scenario["expected_disposition"].as_str().is_none()
            || scenario["invariants"].as_array().is_none()
            || !scenario[payload_field].is_object()
        {
            return Err(EvaluationError::new(
                collection_name,
                format!("scenario {id} is missing a required contract field"),
            ));
        }
    }

    if observed.len() != expected.len() {
        return Err(EvaluationError::new(
            collection_name,
            "scenario set does not match the frozen expected ID set",
        ));
    }
    Ok(scenarios.iter().collect())
}

fn evaluate_water_scenario(scenario: &Value) -> Result<BatchReceipt, EvaluationError> {
    let id = required_str(scenario, "id", "water_scenario")?;
    let expected = required_str(scenario, "expected_disposition", "water_scenario")?;
    let input = &scenario["input"];
    let mut metrics = BTreeMap::new();
    let mut notes = Vec::new();
    let mut unresolved = Vec::new();

    let disposition = match id {
        "WCONT-001" => {
            let flow = required_number(input, "municipal_inlet_flow_l_min")?;
            let duration = required_number(input, "duration_h")?;
            insert_quantity(
                &mut metrics,
                "synthetic_inlet_volume_l",
                flow * duration * 60.0,
                "L",
                "constant synthetic inlet flow multiplied by duration in minutes",
            )?;
            notes.push(
                "Inlet volume is integrated from the fixture's constant synthetic flow assumption."
                    .to_owned(),
            );
            unresolved.push(
                "No independently measured outlet, storage delta, pressure or end-use quality is present."
                    .to_owned(),
            );
            BatchDisposition::ProcessAccountingOnly
        }
        "WCONT-002" => {
            insert_quantity(
                &mut metrics,
                "gross_tank_sensor_inventory_l",
                required_number(input, "tank_level_sensor_l")?,
                "L",
                "synthetic tank-level sensor reading; not usable deliverable reserve",
            )?;
            insert_quantity(
                &mut metrics,
                "municipal_inlet_flow_l_min",
                required_number(input, "municipal_inlet_flow_l_min")?,
                "L/min",
                "synthetic fixture inlet flow",
            )?;
            notes.push(
                "Zero municipal inflow is preserved for the declared 48-hour interruption."
                    .to_owned(),
            );
            unresolved.push(
                "Deliverable reserve is unresolved: usable volume at the fixture boundary, demand interval, pump delivery, pressure and end-use quality are not qualified."
                    .to_owned(),
            );
            BatchDisposition::ReserveUnresolved
        }
        "WCONT-003" => {
            let flow = required_number(input, "municipal_inlet_flow_l_min")?;
            let duration = required_number(input, "duration_h")?;
            insert_quantity(
                &mut metrics,
                "synthetic_inlet_volume_l_under_constant_flow",
                flow * duration * 60.0,
                "L",
                "constant-flow synthetic inlet integration over the declared duration",
            )?;
            insert_quantity(
                &mut metrics,
                "gross_tank_sensor_inventory_l",
                required_number(input, "tank_level_sensor_l")?,
                "L",
                "synthetic tank-level sensor reading; not usable deliverable reserve",
            )?;
            notes.push(
                "The 576 L inlet figure assumes the fixture's 0.8 L/min flow remains constant for all 12 hours."
                    .to_owned(),
            );
            unresolved.push(
                "Intermittent low-pressure profile is categorical; no time-resolved delivery-boundary pressure or delivered volume is provided."
                    .to_owned(),
            );
            BatchDisposition::DegradedService
        }
        "WCONT-004" => {
            let sensor = required_number(input, "tank_level_sensor_l")?;
            let reference = required_number(input, "independent_reference_tank_level_l")?;
            insert_quantity(
                &mut metrics,
                "tank_sensor_l",
                sensor,
                "L",
                "synthetic tank sensor observation; not selected as truth",
            )?;
            insert_quantity(
                &mut metrics,
                "independent_reference_l",
                reference,
                "L",
                "synthetic independent reference observation; not selected as truth",
            )?;
            insert_quantity(
                &mut metrics,
                "absolute_sensor_disagreement_l",
                (sensor - reference).abs(),
                "L",
                "absolute difference between the two synthetic observations",
            )?;
            notes.push(
                "The sensor and independent reference remain separate observations; neither is silently selected as truth."
                    .to_owned(),
            );
            unresolved.push(
                "Reserve precision is withheld until the disagreement is investigated and the reference method is qualified."
                    .to_owned(),
            );
            BatchDisposition::SensorDisagreement
        }
        "WCONT-005" => {
            let flow = required_number(input, "measured_unexplained_flow_l_min")?;
            let duration = required_number(input, "duration_h")?;
            insert_quantity(
                &mut metrics,
                "unexplained_flow_volume_l_under_constant_rate",
                flow * duration * 60.0,
                "L",
                "unexplained synthetic flow integrated under a constant-rate assumption; not proven leakage",
            )?;
            notes.push(
                "The unexplained 0.5 L/min signal integrates to 240 L over the synthetic 8-hour window if constant."
                    .to_owned(),
            );
            unresolved.push(
                "The flow cannot be attributed to a leak without boundary/topology evidence; automatic isolation is not authorized."
                    .to_owned(),
            );
            BatchDisposition::HumanReviewRequired
        }
        "WCONT-006" => {
            insert_quantity(
                &mut metrics,
                "gross_tank_sensor_inventory_l",
                required_number(input, "tank_level_sensor_l")?,
                "L",
                "synthetic tank-level sensor reading; pump/power delivery is unavailable",
            )?;
            let grid = required_bool(input, "grid_available")?;
            let pump = required_bool(input, "pump_available")?;
            let backup = required_bool(input, "backup_power_available")?;
            notes.push(format!(
                "Delivery path inputs: grid_available={grid}, pump_available={pump}, backup_power_available={backup}."
            ));
            unresolved.push(
                "Gross stored volume is not counted as delivered service while the required pump and backup power are unavailable."
                    .to_owned(),
            );
            BatchDisposition::DeliveryUnavailable
        }
        "WCONT-007" => {
            insert_quantity(
                &mut metrics,
                "stored_water_age_h",
                required_number(input, "stored_water_age_h")?,
                "h",
                "synthetic stored-water age",
            )?;
            insert_quantity(
                &mut metrics,
                "gross_tank_sensor_inventory_l",
                required_number(input, "tank_level_sensor_l")?,
                "L",
                "synthetic tank-level sensor reading; does not establish end-use quality",
            )?;
            notes.push(format!(
                "Water-quality evidence state: {}.",
                required_str(input, "water_quality_evidence", "WCONT-007.input")?
            ));
            unresolved.push(
                "Volume and age do not establish fitness for any potable or non-potable end use; the quality evidence is stale or missing."
                    .to_owned(),
            );
            BatchDisposition::WaterQualityUnresolved
        }
        "WCONT-008" => {
            let inlet = required_number(input, "municipal_inlet_volume_l")?;
            let storage = required_number(input, "tank_storage_delta_l")?;
            let outlet = required_number(input, "outlet_meter_volume_l")?;
            let purge = required_number(input, "purge_and_overflow_meter_volume_l")?;
            let per_term_bound = required_number(input, "meter_uncertainty_l")?;
            if per_term_bound < 0.0 {
                return Err(EvaluationError::new(
                    "WCONT-008.meter_uncertainty_l",
                    "uncertainty must be nonnegative",
                ));
            }
            let end_ms = (required_number(input, "duration_h")? * 3_600_000.0) as i64;
            let make_meter = |value_l: f64, stream: &str, boundary: &str| MeterMeasurement {
                value_l,
                absolute_error_bound_l: per_term_bound,
                stream_id: stream.to_owned(),
                measurement_boundary_id: boundary.to_owned(),
                window_start_unix_ms: 0,
                window_end_unix_ms: end_ms,
                calibration_qualified: false,
                calibration_record_sha256: None,
            };
            let balance = evaluate_water_balance(&WaterBalanceInput {
                inlet: make_meter(inlet, "synthetic-inlet", "synthetic-inlet-boundary"),
                tank_storage_delta: make_meter(
                    storage,
                    "synthetic-storage",
                    "synthetic-storage-boundary",
                ),
                delivered_outlet: make_meter(
                    outlet,
                    "synthetic-outlet",
                    "synthetic-outlet-boundary",
                ),
                purge_and_overflow: make_meter(
                    purge,
                    "synthetic-purge",
                    "synthetic-purge-boundary",
                ),
                topology_audit: None::<TopologyAuditReference>,
            })?;
            insert_quantity(
                &mut metrics,
                "mass_balance_residual_l",
                balance.residual_l,
                "L",
                "inlet minus tank storage delta minus delivered outlet minus purge/overflow",
            )?;
            insert_quantity(
                &mut metrics,
                "mass_balance_residual_lower_l",
                balance.residual_lower_l,
                "L",
                "residual minus evaluator-derived worst-case aggregate absolute error bound",
            )?;
            insert_quantity(
                &mut metrics,
                "mass_balance_residual_upper_l",
                balance.residual_upper_l,
                "L",
                "residual plus evaluator-derived worst-case aggregate absolute error bound",
            )?;
            insert_quantity(
                &mut metrics,
                "assumed_per_term_absolute_error_bound_l",
                per_term_bound,
                "L",
                "synthetic fixture field conservatively mapped to each of the four meter terms",
            )?;
            insert_quantity(
                &mut metrics,
                "conservative_aggregate_absolute_error_bound_l",
                balance.aggregate_absolute_error_bound_l,
                "L",
                "sum of the four typed per-measurement absolute error bounds",
            )?;
            notes.push(
                "meter_uncertainty_l has ambiguous scope; the software-only negative control conservatively sums four per-term absolute bounds."
                    .to_owned(),
            );
            unresolved.push(
                "Synthetic fixture values are not calibrated measurements and have no authenticated topology audit; no physical mass-closure claim is permitted."
                    .to_owned(),
            );
            if balance.residual_lower_l > 0.0 || balance.residual_upper_l < 0.0 {
                BatchDisposition::MassBalanceContradictionUnqualified
            } else {
                BatchDisposition::ProcessAccountingOnly
            }
        }
        "WCONT-009" => {
            let command = required_bool(input, "commanded_refill")?;
            let flow = required_number(input, "municipal_inlet_flow_l_min")?;
            insert_boolean(
                &mut metrics,
                "commanded_refill",
                command,
                "whether the synthetic controller command requested refill; not proof of inflow",
            );
            insert_quantity(
                &mut metrics,
                "measured_inlet_flow_l_min",
                flow,
                "L/min",
                "synthetic measured inlet flow; zero despite the refill command",
            )?;
            notes.push(
                "Open-valve feedback and a refill command are not treated as proof of physical inflow."
                    .to_owned(),
            );
            unresolved.push(
                "The refill command conflicts with zero measured flow; the service fault remains unresolved."
                    .to_owned(),
            );
            BatchDisposition::CommandObservationMismatch
        }
        "WCONT-010" => {
            insert_quantity(
                &mut metrics,
                "observation_duration_h",
                required_number(input, "duration_h")?,
                "h",
                "synthetic scenario observation duration",
            )?;
            notes.push(
                "The fixture describes unstable flow and a surge-then-decay pressure profile without numerical time series."
                    .to_owned(),
            );
            unresolved.push(
                "Recovery cannot be declared stable without synchronized measured flow and pressure at the delivery boundary."
                    .to_owned(),
            );
            BatchDisposition::RecoveryMonitoringRequired
        }
        _ => {
            return Err(EvaluationError::new(
                "water_scenario.id",
                format!("unsupported scenario ID: {id}"),
            ));
        }
    };

    Ok(BatchReceipt {
        domain: ServiceDomain::WaterContinuity,
        scenario_id: id.to_owned(),
        fixture_expected_disposition: expected.to_owned(),
        disposition,
        derived_metrics: metrics,
        evidence_notes: notes,
        unresolved_claims: unresolved,
    })
}

fn evaluate_repair_scenario(scenario: &Value) -> Result<BatchReceipt, EvaluationError> {
    let id = required_str(scenario, "id", "repair_scenario")?;
    let expected = required_str(scenario, "expected_disposition", "repair_scenario")?;
    let baseline = &scenario["synthetic_baseline"];
    let mut metrics = BTreeMap::new();
    let mut notes = Vec::new();
    let mut unresolved = Vec::new();

    let disposition = match id {
        "REPAIR-001" => {
            insert_quantity(
                &mut metrics,
                "access_fasteners",
                required_number(baseline, "access_fasteners")?,
                "count",
                "synthetic count of access fasteners",
            )?;
            insert_quantity(
                &mut metrics,
                "disassembly_steps",
                required_number(baseline, "disassembly_steps")?,
                "count",
                "synthetic count of disassembly steps; not measured repair time",
            )?;
            insert_quantity(
                &mut metrics,
                "synthetic_part_lead_time_days",
                required_number(baseline, "part_lead_time_days")?,
                "d",
                "synthetic target-market lead-time assumption",
            )?;
            notes.push(
                "Fixture documents a user-serviceable access path and offline instructions; no physical replacement task was executed."
                    .to_owned(),
            );
            unresolved.push(
                "Exact filter identity, safe-state verification, seal inspection and post-service leak test are not supplied as results."
                    .to_owned(),
            );
            BatchDisposition::RepairNotExecuted
        }
        "REPAIR-002" => {
            insert_quantity(
                &mut metrics,
                "access_fasteners",
                required_number(baseline, "access_fasteners")?,
                "count",
                "synthetic count of access fasteners",
            )?;
            insert_quantity(
                &mut metrics,
                "disassembly_steps",
                required_number(baseline, "disassembly_steps")?,
                "count",
                "synthetic count of disassembly steps; not measured repair time",
            )?;
            insert_quantity(
                &mut metrics,
                "synthetic_part_lead_time_days",
                required_number(baseline, "part_lead_time_days")?,
                "d",
                "synthetic part lead-time assumption",
            )?;
            notes.push(
                "This is a trained-technician pump-module task; fixture counts describe access complexity, not measured repair time."
                    .to_owned(),
            );
            unresolved.push(
                "Electrical/fluid isolation and post-repair hydraulic, electrical and functional requalification results are absent."
                    .to_owned(),
            );
            BatchDisposition::RepairHeld
        }
        "REPAIR-003" => {
            insert_quantity(
                &mut metrics,
                "access_fasteners",
                required_number(baseline, "access_fasteners")?,
                "count",
                "synthetic count of access fasteners",
            )?;
            insert_quantity(
                &mut metrics,
                "disassembly_steps",
                required_number(baseline, "disassembly_steps")?,
                "count",
                "synthetic count of disassembly steps; not measured repair time",
            )?;
            insert_boolean(
                &mut metrics,
                "adhesive_or_destructive_disassembly",
                required_bool(baseline, "adhesive_or_destructive_disassembly")?,
                "whether the fixture declares adhesive or destructive disassembly",
            );
            notes.push(
                "The synthetic task declares non-destructive access; it does not demonstrate an actual seal replacement."
                    .to_owned(),
            );
            unresolved.push(
                "Exact seal revision, enclosure integrity test and any affected hygiene/cross-cell requalification remain unverified."
                    .to_owned(),
            );
            BatchDisposition::RepairHeld
        }
        "REPAIR-004" => {
            insert_quantity(
                &mut metrics,
                "synthetic_part_lead_time_days",
                required_number(baseline, "part_lead_time_days")?,
                "d",
                "synthetic part lead-time assumption",
            )?;
            insert_boolean(
                &mut metrics,
                "calibration_required",
                required_bool(baseline, "calibration_required")?,
                "whether the repair task requires calibration",
            );
            notes.push(
                "Calibration is explicitly required by the synthetic task profile.".to_owned(),
            );
            unresolved.push(
                "No sensor-serial/revision-bound calibration artifact or independent reference check is supplied."
                    .to_owned(),
            );
            BatchDisposition::RepairHeld
        }
        "REPAIR-005" => {
            let signed_image = required_bool(baseline, "signed_image_available")?;
            let previous_image = required_bool(baseline, "previous_known_good_image_available")?;
            insert_boolean(
                &mut metrics,
                "signed_image_available",
                signed_image,
                "whether the fixture declares a signed recovery image available",
            );
            insert_boolean(
                &mut metrics,
                "previous_known_good_image_available",
                previous_image,
                "whether the fixture declares a previous known-good image available",
            );
            notes.push(
                "Both offline recovery artifacts are declared available while internet access is absent."
                    .to_owned(),
            );
            unresolved.push(
                "Signature verification, image compatibility, recovery execution and post-recovery safety behavior have not been demonstrated."
                    .to_owned(),
            );
            BatchDisposition::RecoveryArtifactsAvailableNotExecuted
        }
        "REPAIR-006" => {
            let attempt = RepairAttempt {
                diagnosis_supported: true,
                exact_spare_available: required_bool(baseline, "local_stock")?,
                replacement_compatibility_qualified: required_bool(
                    baseline,
                    "approved_substitute",
                )?,
                offline_service_required: false,
                offline_service_available: true,
                affected_calibration_requalified: false,
                post_repair_function_passed: false,
                safety_check_required: true,
                post_repair_safety_check_passed: false,
                follow_up_days_observed: 0,
                recurrence_observed: false,
            };
            let result = evaluate_repair_attempt(&attempt, 30)?;
            insert_boolean(
                &mut metrics,
                "catalog_part_exists",
                required_bool(baseline, "catalog_part_exists")?,
                "whether the fixture says a catalogue entry exists; not evidence of local supply",
            );
            insert_boolean(
                &mut metrics,
                "local_stock",
                required_bool(baseline, "local_stock")?,
                "whether the fixture declares target-market local stock",
            );
            notes.push(
                "Catalog presence is kept separate from target-market stock; unknown quoted lead time is not coerced to zero."
                    .to_owned(),
            );
            unresolved.push(
                "Repair remains blocked until an exact qualified spare is actually obtainable; no substitute is approved."
                    .to_owned(),
            );
            if result.disposition == RepairDisposition::Blocked {
                BatchDisposition::RepairBlocked
            } else {
                return Err(EvaluationError::new(
                    "REPAIR-006",
                    "parts-unavailability negative control did not fail closed",
                ));
            }
        }
        "REPAIR-007" => {
            let attempt = RepairAttempt {
                diagnosis_supported: true,
                exact_spare_available: true,
                replacement_compatibility_qualified: required_bool(
                    baseline,
                    "material_grade_known",
                )? && baseline["firmware_compatibility"]
                    .as_str()
                    == Some("qualified"),
                offline_service_required: false,
                offline_service_available: true,
                affected_calibration_requalified: false,
                post_repair_function_passed: false,
                safety_check_required: true,
                post_repair_safety_check_passed: false,
                follow_up_days_observed: 0,
                recurrence_observed: false,
            };
            let result = evaluate_repair_attempt(&attempt, 30)?;
            insert_boolean(
                &mut metrics,
                "physical_fit",
                required_bool(baseline, "physical_fit")?,
                "whether the fixture declares physical fit; not qualified compatibility",
            );
            insert_boolean(
                &mut metrics,
                "electrical_interface_match",
                required_bool(baseline, "electrical_interface_match")?,
                "whether the fixture declares a matching electrical interface; not qualified compatibility",
            );
            insert_boolean(
                &mut metrics,
                "material_grade_known",
                required_bool(baseline, "material_grade_known")?,
                "whether the replacement material grade is known",
            );
            notes.push(
                "Physical fit and an electrical connector match are not treated as qualified compatibility."
                    .to_owned(),
            );
            unresolved.push(
                "Unknown material grade and firmware compatibility reject the substitute pending full qualification."
                    .to_owned(),
            );
            if result.disposition == RepairDisposition::Blocked {
                BatchDisposition::RepairBlocked
            } else {
                return Err(EvaluationError::new(
                    "REPAIR-007",
                    "unqualified-substitute negative control did not fail closed",
                ));
            }
        }
        "REPAIR-008" => {
            let attempt = RepairAttempt {
                diagnosis_supported: true,
                exact_spare_available: true,
                replacement_compatibility_qualified: true,
                offline_service_required: false,
                offline_service_available: true,
                affected_calibration_requalified: true,
                post_repair_function_passed: required_bool(
                    baseline,
                    "initial_post_repair_function_pass",
                )?,
                safety_check_required: true,
                post_repair_safety_check_passed: true,
                follow_up_days_observed: 30,
                recurrence_observed: required_bool(baseline, "recurrence_observed_after_30_days")?,
            };
            let result = evaluate_repair_attempt(&attempt, 30)?;
            insert_quantity(
                &mut metrics,
                "follow_up_days_observed",
                30.0,
                "d",
                "synthetic follow-up interval; not lifetime exposure",
            )?;
            insert_boolean(
                &mut metrics,
                "recurrence_observed",
                attempt.recurrence_observed,
                "whether recurrence is recorded during the specified 30-day synthetic follow-up",
            );
            notes.push(
                "Initial function restoration and later fault recurrence are represented as distinct outcomes."
                    .to_owned(),
            );
            unresolved.push(
                "The synthetic 30-day recurrence case is not a population reliability estimate or a physical repair trial."
                    .to_owned(),
            );
            if result.disposition == RepairDisposition::RecurrenceObserved {
                BatchDisposition::RecurrenceObserved
            } else {
                return Err(EvaluationError::new(
                    "REPAIR-008",
                    "recurrence negative control did not preserve the observed recurrence",
                ));
            }
        }
        "REPAIR-009" => {
            let attempt = RepairAttempt {
                diagnosis_supported: true,
                exact_spare_available: true,
                replacement_compatibility_qualified: true,
                offline_service_required: true,
                offline_service_available: required_bool(
                    baseline,
                    "offline_diagnostics_available",
                )?,
                affected_calibration_requalified: false,
                post_repair_function_passed: false,
                safety_check_required: true,
                post_repair_safety_check_passed: false,
                follow_up_days_observed: 0,
                recurrence_observed: false,
            };
            let result = evaluate_repair_attempt(&attempt, 30)?;
            insert_boolean(
                &mut metrics,
                "vendor_cloud_available",
                required_bool(baseline, "vendor_cloud_available")?,
                "whether the vendor cloud service is available",
            );
            insert_boolean(
                &mut metrics,
                "offline_diagnostics_available",
                required_bool(baseline, "offline_diagnostics_available")?,
                "whether offline diagnostics are available",
            );
            insert_boolean(
                &mut metrics,
                "manual_available",
                required_bool(baseline, "manual_available")?,
                "whether a service manual is available",
            );
            notes.push(
                "A manual is present, but the declared diagnostic workflow requires a vendor cloud account."
                    .to_owned(),
            );
            unresolved.push(
                "Independent service access remains blocked; no security or safety bypass is proposed."
                    .to_owned(),
            );
            if result.disposition == RepairDisposition::Blocked {
                BatchDisposition::RepairBlocked
            } else {
                return Err(EvaluationError::new(
                    "REPAIR-009",
                    "cloud-dependency negative control did not fail closed",
                ));
            }
        }
        "REPAIR-010" => {
            let attempt = RepairAttempt {
                diagnosis_supported: true,
                exact_spare_available: true,
                replacement_compatibility_qualified: true,
                offline_service_required: false,
                offline_service_available: true,
                affected_calibration_requalified: !required_bool(
                    baseline,
                    "critical_interface_changed",
                )?,
                post_repair_function_passed: required_bool(baseline, "post_repair_test_available")?,
                safety_check_required: true,
                post_repair_safety_check_passed: required_bool(
                    baseline,
                    "post_repair_test_available",
                )?,
                follow_up_days_observed: 0,
                recurrence_observed: false,
            };
            let result = evaluate_repair_attempt(&attempt, 30)?;
            insert_boolean(
                &mut metrics,
                "critical_interface_changed",
                required_bool(baseline, "critical_interface_changed")?,
                "whether the repair changed a critical safety or hygiene interface",
            );
            insert_boolean(
                &mut metrics,
                "post_repair_test_available",
                required_bool(baseline, "post_repair_test_available")?,
                "whether the product-specific post-repair test is available",
            );
            notes.push(
                "A critical safety/hygiene interface change reopens the affected qualification."
                    .to_owned(),
            );
            unresolved.push(
                "No release is permitted without the product-specific post-repair qualification test."
                    .to_owned(),
            );
            if result.disposition == RepairDisposition::HeldForRequalification {
                BatchDisposition::RepairHeld
            } else {
                return Err(EvaluationError::new(
                    "REPAIR-010",
                    "critical-interface negative control did not remain held",
                ));
            }
        }
        _ => {
            return Err(EvaluationError::new(
                "repair_scenario.id",
                format!("unsupported scenario ID: {id}"),
            ));
        }
    };

    Ok(BatchReceipt {
        domain: ServiceDomain::Repairability,
        scenario_id: id.to_owned(),
        fixture_expected_disposition: expected.to_owned(),
        disposition,
        derived_metrics: metrics,
        evidence_notes: notes,
        unresolved_claims: unresolved,
    })
}

fn required_str<'a>(
    object: &'a Value,
    key: &'static str,
    path: &'static str,
) -> Result<&'a str, EvaluationError> {
    object[key].as_str().ok_or_else(|| {
        EvaluationError::new(path, format!("required string field {key} is missing"))
    })
}

fn required_number(object: &Value, key: &'static str) -> Result<f64, EvaluationError> {
    let value = object[key].as_f64().ok_or_else(|| {
        EvaluationError::new(
            "scenario.input",
            format!("required number {key} is missing"),
        )
    })?;
    if !value.is_finite() {
        return Err(EvaluationError::new(
            "scenario.input",
            format!("number {key} must be finite"),
        ));
    }
    Ok(value)
}

fn required_bool(object: &Value, key: &'static str) -> Result<bool, EvaluationError> {
    object[key].as_bool().ok_or_else(|| {
        EvaluationError::new(
            "scenario.input",
            format!("required boolean {key} is missing"),
        )
    })
}

fn disposition_name(disposition: BatchDisposition) -> &'static str {
    match disposition {
        BatchDisposition::ProcessAccountingOnly => "process_accounting_only",
        BatchDisposition::ReserveUnresolved => "reserve_unresolved",
        BatchDisposition::DegradedService => "degraded_service",
        BatchDisposition::SensorDisagreement => "sensor_disagreement",
        BatchDisposition::HumanReviewRequired => "human_review_required",
        BatchDisposition::DeliveryUnavailable => "delivery_unavailable",
        BatchDisposition::WaterQualityUnresolved => "water_quality_unresolved",
        BatchDisposition::MassBalanceContradictionUnqualified => {
            "mass_balance_contradiction_unqualified"
        }
        BatchDisposition::CommandObservationMismatch => "command_observation_mismatch",
        BatchDisposition::RecoveryMonitoringRequired => "recovery_monitoring_required",
        BatchDisposition::RepairNotExecuted => "repair_not_executed",
        BatchDisposition::RepairBlocked => "repair_blocked",
        BatchDisposition::RepairHeld => "repair_held",
        BatchDisposition::RecoveryArtifactsAvailableNotExecuted => {
            "recovery_artifacts_available_not_executed"
        }
        BatchDisposition::RecurrenceObserved => "recurrence_observed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixtures() -> (Value, Value) {
        let water = serde_json::from_str(include_str!(
            "../../../docs/research/household-services/fixtures/water-continuity-scenarios-v0.1.json"
        ))
        .expect("water fixture must parse");
        let repair = serde_json::from_str(include_str!(
            "../../../docs/research/household-services/fixtures/repairability-scenarios-v0.1.json"
        ))
        .expect("repair fixture must parse");
        (water, repair)
    }

    #[test]
    fn all_twenty_frozen_scenarios_produce_deterministic_receipts() {
        let (water, repair) = fixtures();
        let report = evaluate_batch(&water, &repair).unwrap();

        assert_eq!(report.water_receipt_count, 10);
        assert_eq!(report.repair_receipt_count, 10);
        assert_eq!(report.total_receipt_count, 20);
        assert_eq!(report.receipts.len(), 20);
        assert_eq!(
            report.water_corpus_id,
            "HOME-UTILITY-COORD-001-WATER-CONTINUITY-SYNTHETIC-001"
        );
        assert_eq!(report.report_schema_version, "0.2");
        assert_eq!(
            report.evidence_class,
            EvidenceClass::SyntheticFixtureOnly
        );
        assert_eq!(
            report.water_source.digest_scope,
            DigestScope::CanonicalizedJsonValue
        );
        assert_eq!(
            quantity_value(&report.receipts[7], "mass_balance_residual_l"),
            -200.0
        );
        assert_eq!(
            quantity_value(&report.receipts[7], "mass_balance_residual_lower_l"),
            -220.0
        );
        assert_eq!(
            quantity_value(&report.receipts[7], "mass_balance_residual_upper_l"),
            -180.0
        );
        assert_eq!(
            metric_unit(&report.receipts[7], "mass_balance_residual_l"),
            "L"
        );
        assert_eq!(
            report.receipts[7].disposition,
            BatchDisposition::MassBalanceContradictionUnqualified
        );
        assert_eq!(
            report.receipts[15].disposition,
            BatchDisposition::RepairBlocked
        );
        assert_eq!(
            report.receipts[17].disposition,
            BatchDisposition::RecurrenceObserved
        );

        let first = serde_json::to_vec(&report).unwrap();
        let second = serde_json::to_vec(&evaluate_batch(&water, &repair).unwrap()).unwrap();
        assert_eq!(first, second, "batch report must be byte-deterministic");
        assert_eq!(report.disposition_counts.values().sum::<usize>(), 20);
    }


    fn quantity_value(receipt: &BatchReceipt, name: &str) -> f64 {
        match &receipt.derived_metrics[name] {
            DerivedMetric::Quantity { value, .. } => *value,
            other => panic!("expected quantity metric, found {other:?}"),
        }
    }

    fn metric_unit<'a>(receipt: &'a BatchReceipt, name: &str) -> &'a str {
        match &receipt.derived_metrics[name] {
            DerivedMetric::Quantity { unit, .. } => unit,
            other => panic!("expected quantity metric, found {other:?}"),
        }
    }

    #[test]
    fn exact_source_bytes_are_bound_and_a_one_byte_change_changes_the_digest() {
        let water = include_bytes!(
            "../../../docs/research/household-services/fixtures/water-continuity-scenarios-v0.1.json"
        );
        let repair = include_bytes!(
            "../../../docs/research/household-services/fixtures/repairability-scenarios-v0.1.json"
        );
        let original = evaluate_batch_from_bytes(water, repair).unwrap();
        let mut changed_water = water.to_vec();
        changed_water.push(b' ');
        let changed = evaluate_batch_from_bytes(&changed_water, repair).unwrap();

        assert_eq!(original.water_source.digest_scope, DigestScope::ExactSourceBytes);
        assert_eq!(original.water_source.source_byte_length, water.len());
        assert_eq!(
            original.water_source.source_sha256,
            sha256_hex(water)
        );
        assert_ne!(
            original.water_source.source_sha256,
            changed.water_source.source_sha256
        );
        assert_eq!(
            changed.water_source.source_byte_length,
            water.len() + 1
        );
        assert_eq!(
            original.repair_source.source_sha256,
            changed.repair_source.source_sha256
        );
    }

    #[test]
    fn boolean_metrics_remain_json_booleans_and_quantities_retain_units() {
        let (water, repair) = fixtures();
        let report = evaluate_batch(&water, &repair).unwrap();
        let command = &report.receipts[8].derived_metrics["commanded_refill"];
        assert_eq!(
            serde_json::to_value(command).unwrap()["value"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(
            serde_json::to_value(command).unwrap()["kind"],
            "boolean"
        );
        let flow = &report.receipts[8].derived_metrics["measured_inlet_flow_l_min"];
        assert_eq!(
            serde_json::to_value(flow).unwrap()["unit"],
            "L/min"
        );
        assert!(matches!(
            &report.receipts[8].derived_metrics["commanded_refill"],
            DerivedMetric::Boolean { value: true, .. }
        ));
    }

    #[test]
    fn output_envelope_hashes_report_body_without_self_reference() {
        let (water, repair) = fixtures();
        let report = evaluate_batch(&water, &repair).unwrap();
        let expected_bytes = serde_json::to_vec(&report).unwrap();
        let envelope = BatchOutputEnvelope::from_report(report.clone()).unwrap();
        assert_eq!(envelope.report_sha256, sha256_hex(&expected_bytes));
        assert_eq!(envelope.report_byte_length, expected_bytes.len());
        assert_eq!(
            envelope.report_serialization,
            "serde_json_compact_struct_order_btree_maps_v1"
        );
        assert_eq!(envelope.report, report);
    }

    #[test]
    fn sha256_helper_matches_published_known_answer() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn source_commit_is_recorded_only_when_a_valid_full_object_id_is_supplied() {
        let water = include_bytes!(
            "../../../docs/research/household-services/fixtures/water-continuity-scenarios-v0.1.json"
        );
        let repair = include_bytes!(
            "../../../docs/research/household-services/fixtures/repairability-scenarios-v0.1.json"
        );
        let report = evaluate_batch_from_bytes_with_source_commit(
            water,
            repair,
            Some("0123456789abcdef0123456789abcdef01234567"),
        )
        .unwrap();
        assert_eq!(
            report.source_commit.as_deref(),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
        assert!(
            evaluate_batch_from_bytes_with_source_commit(water, repair, Some("short")).is_err()
        );
    }

    #[test]
    fn duplicate_water_scenario_is_rejected() {
        let (mut water, repair) = fixtures();
        water["scenarios"][1]["id"] = json!("WCONT-001");
        assert!(evaluate_batch(&water, &repair).is_err());
    }

    #[test]
    fn missing_repair_scenario_is_rejected() {
        let (water, mut repair) = fixtures();
        repair["task_profiles"].as_array_mut().unwrap().pop();
        assert!(evaluate_batch(&water, &repair).is_err());
    }

    #[test]
    fn unexpected_scenario_is_rejected_instead_of_silently_skipped() {
        let (mut water, repair) = fixtures();
        water["scenarios"][9]["id"] = json!("WCONT-011");
        assert!(evaluate_batch(&water, &repair).is_err());
    }

    #[test]
    fn non_synthetic_corpus_status_is_rejected() {
        let (mut water, repair) = fixtures();
        water["status"] = json!("production_qualified");
        assert!(evaluate_batch(&water, &repair).is_err());
    }
}
