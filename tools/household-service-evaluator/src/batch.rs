use crate::{
    EvaluationError, MeterMeasurement, RepairAttempt, RepairDisposition, TopologyAuditReference,
    WaterBalanceInput, evaluate_repair_attempt, evaluate_water_balance,
};
use serde::Serialize;
use serde_json::Value;
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
pub struct BatchReceipt {
    pub domain: ServiceDomain,
    pub scenario_id: String,
    /// Retained as fixture metadata, never treated as the evaluator's result.
    pub fixture_expected_disposition: String,
    pub disposition: BatchDisposition,
    /// Deterministic metrics derived only from the synthetic fixture inputs.
    pub derived_metrics: BTreeMap<String, f64>,
    pub evidence_notes: Vec<String>,
    pub unresolved_claims: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BatchReport {
    pub report_schema_version: String,
    pub evaluator_version: String,
    pub water_corpus_id: String,
    pub repair_corpus_id: String,
    pub water_receipt_count: usize,
    pub repair_receipt_count: usize,
    pub total_receipt_count: usize,
    pub disposition_counts: BTreeMap<String, usize>,
    pub receipts: Vec<BatchReceipt>,
    pub claim_ceiling: Vec<String>,
}

/// Evaluate every expected scenario in the two frozen v0.1 corpora.
///
/// The fixture's expected disposition is copied into each receipt as reference
/// metadata only. The actual disposition is selected from validated input
/// fields and bounded deterministic calculations. Unknown, duplicate, missing
/// or additional scenarios are rejected rather than silently skipped.
pub fn evaluate_batch(
    water_corpus: &Value,
    repair_corpus: &Value,
) -> Result<BatchReport, EvaluationError> {
    let water_corpus_id = validate_corpus_header(
        water_corpus,
        "HOME-UTILITY-COORD-001-WATER-CONTINUITY-SYNTHETIC-001",
    )?;
    let repair_corpus_id = validate_corpus_header(
        repair_corpus,
        "HOME-REPAIR-QUAL-001-SYNTHETIC-REPAIR-TASKS-001",
    )?;

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
        report_schema_version: "0.1".to_owned(),
        evaluator_version: env!("CARGO_PKG_VERSION").to_owned(),
        water_corpus_id,
        repair_corpus_id,
        water_receipt_count: WATER_IDS.len(),
        repair_receipt_count: REPAIR_IDS.len(),
        total_receipt_count: receipts.len(),
        disposition_counts,
        receipts,
        claim_ceiling,
    })
}

/// Read two JSON corpus files and evaluate them without network or clock input.
pub fn evaluate_batch_from_paths(
    water_path: impl AsRef<Path>,
    repair_path: impl AsRef<Path>,
) -> Result<BatchReport, EvaluationError> {
    let water_bytes = fs::read(water_path)
        .map_err(|error| EvaluationError::new("water_fixture", error.to_string()))?;
    let repair_bytes = fs::read(repair_path)
        .map_err(|error| EvaluationError::new("repair_fixture", error.to_string()))?;

    let water_corpus: Value = serde_json::from_slice(&water_bytes)
        .map_err(|error| EvaluationError::new("water_fixture_json", error.to_string()))?;
    let repair_corpus: Value = serde_json::from_slice(&repair_bytes)
        .map_err(|error| EvaluationError::new("repair_fixture_json", error.to_string()))?;

    evaluate_batch(&water_corpus, &repair_corpus)
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
            metrics.insert(
                "synthetic_inlet_volume_l".to_owned(),
                flow * duration * 60.0,
            );
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
            metrics.insert(
                "gross_tank_sensor_inventory_l".to_owned(),
                required_number(input, "tank_level_sensor_l")?,
            );
            metrics.insert(
                "municipal_inlet_flow_l_min".to_owned(),
                required_number(input, "municipal_inlet_flow_l_min")?,
            );
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
            metrics.insert(
                "synthetic_inlet_volume_l_under_constant_flow".to_owned(),
                flow * duration * 60.0,
            );
            metrics.insert(
                "gross_tank_sensor_inventory_l".to_owned(),
                required_number(input, "tank_level_sensor_l")?,
            );
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
            metrics.insert("tank_sensor_l".to_owned(), sensor);
            metrics.insert("independent_reference_l".to_owned(), reference);
            metrics.insert(
                "absolute_sensor_disagreement_l".to_owned(),
                (sensor - reference).abs(),
            );
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
            metrics.insert(
                "unexplained_flow_volume_l_under_constant_rate".to_owned(),
                flow * duration * 60.0,
            );
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
            metrics.insert(
                "gross_tank_sensor_inventory_l".to_owned(),
                required_number(input, "tank_level_sensor_l")?,
            );
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
            metrics.insert(
                "stored_water_age_h".to_owned(),
                required_number(input, "stored_water_age_h")?,
            );
            metrics.insert(
                "gross_tank_sensor_inventory_l".to_owned(),
                required_number(input, "tank_level_sensor_l")?,
            );
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
            metrics.insert("mass_balance_residual_l".to_owned(), balance.residual_l);
            metrics.insert(
                "mass_balance_residual_lower_l".to_owned(),
                balance.residual_lower_l,
            );
            metrics.insert(
                "mass_balance_residual_upper_l".to_owned(),
                balance.residual_upper_l,
            );
            metrics.insert(
                "assumed_per_term_absolute_error_bound_l".to_owned(),
                per_term_bound,
            );
            metrics.insert(
                "conservative_aggregate_absolute_error_bound_l".to_owned(),
                balance.aggregate_absolute_error_bound_l,
            );
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
            metrics.insert("commanded_refill".to_owned(), bool_as_f64(command));
            metrics.insert("measured_inlet_flow_l_min".to_owned(), flow);
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
            metrics.insert(
                "observation_duration_h".to_owned(),
                required_number(input, "duration_h")?,
            );
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
            metrics.insert(
                "access_fasteners".to_owned(),
                required_number(baseline, "access_fasteners")?,
            );
            metrics.insert(
                "disassembly_steps".to_owned(),
                required_number(baseline, "disassembly_steps")?,
            );
            metrics.insert(
                "synthetic_part_lead_time_days".to_owned(),
                required_number(baseline, "part_lead_time_days")?,
            );
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
            metrics.insert(
                "access_fasteners".to_owned(),
                required_number(baseline, "access_fasteners")?,
            );
            metrics.insert(
                "disassembly_steps".to_owned(),
                required_number(baseline, "disassembly_steps")?,
            );
            metrics.insert(
                "synthetic_part_lead_time_days".to_owned(),
                required_number(baseline, "part_lead_time_days")?,
            );
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
            metrics.insert(
                "access_fasteners".to_owned(),
                required_number(baseline, "access_fasteners")?,
            );
            metrics.insert(
                "disassembly_steps".to_owned(),
                required_number(baseline, "disassembly_steps")?,
            );
            metrics.insert(
                "adhesive_or_destructive_disassembly".to_owned(),
                bool_as_f64(required_bool(
                    baseline,
                    "adhesive_or_destructive_disassembly",
                )?),
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
            metrics.insert(
                "synthetic_part_lead_time_days".to_owned(),
                required_number(baseline, "part_lead_time_days")?,
            );
            metrics.insert(
                "calibration_required".to_owned(),
                bool_as_f64(required_bool(baseline, "calibration_required")?),
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
            metrics.insert(
                "signed_image_available".to_owned(),
                bool_as_f64(signed_image),
            );
            metrics.insert(
                "previous_known_good_image_available".to_owned(),
                bool_as_f64(previous_image),
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
            metrics.insert(
                "catalog_part_exists".to_owned(),
                bool_as_f64(required_bool(baseline, "catalog_part_exists")?),
            );
            metrics.insert(
                "local_stock".to_owned(),
                bool_as_f64(required_bool(baseline, "local_stock")?),
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
            metrics.insert(
                "physical_fit".to_owned(),
                bool_as_f64(required_bool(baseline, "physical_fit")?),
            );
            metrics.insert(
                "electrical_interface_match".to_owned(),
                bool_as_f64(required_bool(baseline, "electrical_interface_match")?),
            );
            metrics.insert(
                "material_grade_known".to_owned(),
                bool_as_f64(required_bool(baseline, "material_grade_known")?),
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
            metrics.insert("follow_up_days_observed".to_owned(), 30.0);
            metrics.insert(
                "recurrence_observed".to_owned(),
                bool_as_f64(attempt.recurrence_observed),
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
            metrics.insert(
                "vendor_cloud_available".to_owned(),
                bool_as_f64(required_bool(baseline, "vendor_cloud_available")?),
            );
            metrics.insert(
                "offline_diagnostics_available".to_owned(),
                bool_as_f64(required_bool(baseline, "offline_diagnostics_available")?),
            );
            metrics.insert(
                "manual_available".to_owned(),
                bool_as_f64(required_bool(baseline, "manual_available")?),
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
            metrics.insert(
                "critical_interface_changed".to_owned(),
                bool_as_f64(required_bool(baseline, "critical_interface_changed")?),
            );
            metrics.insert(
                "post_repair_test_available".to_owned(),
                bool_as_f64(required_bool(baseline, "post_repair_test_available")?),
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

fn bool_as_f64(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
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
        assert_eq!(
            report.receipts[7].derived_metrics["mass_balance_residual_l"],
            -200.0
        );
        assert_eq!(
            report.receipts[7].derived_metrics["mass_balance_residual_lower_l"],
            -220.0
        );
        assert_eq!(
            report.receipts[7].derived_metrics["mass_balance_residual_upper_l"],
            -180.0
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
