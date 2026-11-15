use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::{fs, path::Path};

const SCHEMA_VERSION: &str = "0.1.0";
const BENCHMARK_ID: &str = "PLASMA-MEDICAL-002";
const CLAIM_CEILING: &str =
    "SYNTHETIC_DATA_CONTRACT_AUDIT_ONLY_NO_PHYSICAL_OR_CLINICAL_VALIDATION";

#[derive(Debug, Deserialize)]
struct Corpus {
    schema_version: String,
    benchmark_id: String,
    claim_ceiling: String,
    expected_counts: ExpectedCounts,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
struct ExpectedCounts {
    pass: usize,
    blocked: usize,
    rejected: usize,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    expected_disposition: String,
    reason_code: String,
    record: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Disposition {
    Pass,
    Blocked,
    Rejected,
}

impl Disposition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Blocked => "BLOCKED",
            Self::Rejected => "REJECTED",
        }
    }
}

#[derive(Debug)]
struct Assessment {
    disposition: Disposition,
    reason_code: &'static str,
}

fn is_sha256(value: Option<&str>) -> bool {
    value.is_some_and(|s| {
        s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    })
}

fn nonempty_string(value: Option<&Value>) -> bool {
    value.and_then(Value::as_str).is_some_and(|s| !s.trim().is_empty())
}

fn has_number(value: Option<&Value>) -> bool {
    value.is_some_and(Value::is_number)
}

fn get_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn blocked(reason_code: &'static str) -> Assessment {
    Assessment {
        disposition: Disposition::Blocked,
        reason_code,
    }
}

fn rejected(reason_code: &'static str) -> Assessment {
    Assessment {
        disposition: Disposition::Rejected,
        reason_code,
    }
}

fn check_measurement(measurement: &Value) -> Option<Assessment> {
    let Some(status) = get_str(measurement, "status") else {
        return Some(blocked("MEASUREMENT_STATUS_MISSING"));
    };
    let method_present = nonempty_string(measurement.get("method_id"));
    let raw_digest = get_str(measurement, "raw_data_sha256");

    if let Some(digest) = raw_digest {
        if !is_sha256(Some(digest)) {
            return Some(rejected("PROVENANCE_DIGEST_INVALID"));
        }
    }

    match status {
        "MEASURED" => {
            if !has_number(measurement.get("value")) {
                return Some(blocked("MEASURED_VALUE_MISSING"));
            }
            if !nonempty_string(measurement.get("quantity_id"))
                || !nonempty_string(measurement.get("unit"))
                || !method_present
            {
                return Some(blocked("MEASUREMENT_METHOD_OR_UNIT_MISSING"));
            }
            if !nonempty_string(measurement.get("observed_at")) {
                return Some(blocked("MEASUREMENT_TIMESTAMP_MISSING"));
            }
            if !is_sha256(raw_digest) {
                return Some(blocked("MEASUREMENT_RAW_DATA_MISSING"));
            }
            let calibration = measurement.get("calibration");
            let calibration_status = calibration.and_then(|c| get_str(c, "status"));
            match calibration_status {
                Some("EXPIRED") | Some("UNKNOWN") | None => {
                    return Some(blocked("CALIBRATION_NOT_VALID"));
                }
                Some("IN_DATE") => {
                    if !nonempty_string(calibration.and_then(|c| c.get("calibration_id")))
                        || !nonempty_string(calibration.and_then(|c| c.get("valid_from")))
                        || !nonempty_string(calibration.and_then(|c| c.get("valid_until")))
                    {
                        return Some(blocked("CALIBRATION_REFERENCE_MISSING"));
                    }
                    let certificate = calibration.and_then(|c| get_str(c, "certificate_sha256"));
                    if !is_sha256(certificate) {
                        return Some(blocked("CALIBRATION_CERTIFICATE_MISSING"));
                    }
                }
                Some("NOT_REQUIRED") | Some("NOT_APPLICABLE") => {}
                _ => return Some(blocked("CALIBRATION_STATUS_INVALID")),
            }
            let uncertainty = measurement.get("uncertainty");
            match uncertainty.and_then(|u| get_str(u, "status")) {
                Some("QUANTIFIED") => {
                    if !has_number(uncertainty.and_then(|u| u.get("value")))
                        || !nonempty_string(uncertainty.and_then(|u| u.get("unit")))
                    {
                        return Some(blocked("UNCERTAINTY_DECLARATION_INCOMPLETE"));
                    }
                    if !has_number(uncertainty.and_then(|u| u.get("coverage_factor")))
                        && !has_number(uncertainty.and_then(|u| u.get("confidence_level")))
                    {
                        return Some(blocked("UNCERTAINTY_COVERAGE_NOT_DECLARED"));
                    }
                }
                Some("NOT_QUANTIFIED") => {
                    return Some(blocked("UNCERTAINTY_NOT_QUANTIFIED"));
                }
                _ => return Some(blocked("UNCERTAINTY_DECLARATION_MISSING")),
            }
        }
        "BELOW_DETECTION_LIMIT" => {
            if measurement.get("value").is_some() {
                return Some(rejected("BELOW_LIMIT_CANNOT_CARRY_VALUE"));
            }
            if !has_number(measurement.get("detection_limit"))
                || !nonempty_string(measurement.get("unit"))
                || !method_present
                || !nonempty_string(measurement.get("quantity_id"))
                || !nonempty_string(measurement.get("observed_at"))
            {
                return Some(blocked("DETECTION_LIMIT_RECORD_INCOMPLETE"));
            }
            if !is_sha256(raw_digest) {
                return Some(blocked("MEASUREMENT_RAW_DATA_MISSING"));
            }
            match measurement
                .get("calibration")
                .and_then(|c| get_str(c, "status"))
            {
                Some("EXPIRED") | Some("UNKNOWN") | None => {
                    return Some(blocked("CALIBRATION_NOT_VALID"));
                }
                _ => {}
            }
            return Some(blocked("BELOW_LIMIT_NOT_QUANTIFIED_AS_MEASURED_DOSE"));
        }
        "UNKNOWN" => {
            if measurement.get("value").is_some() {
                return Some(rejected("UNKNOWN_VALUE_MUST_NOT_BE_ZERO"));
            }
            return Some(blocked("OBSERVATION_NOT_MEASURED"));
        }
        "NOT_MEASURED" | "NOT_APPLICABLE" | "WITHHELD" => {
            if measurement.get("value").is_some() {
                return Some(rejected("NON_MEASURED_STATUS_CANNOT_CARRY_VALUE"));
            }
            return Some(blocked("OBSERVATION_NOT_MEASURED"));
        }
        _ => return Some(rejected("MEASUREMENT_STATUS_INVALID")),
    }

    None
}

fn all_measurements(value: Option<&Value>) -> std::result::Result<(), Assessment> {
    let measurements = value
        .and_then(Value::as_array)
        .ok_or_else(|| blocked("MEASUREMENT_ARRAY_MISSING"))?;
    if measurements.is_empty() {
        return Err(blocked("NO_PHYSICAL_OBSERVATIONS"));
    }

    let mut measured_count = 0usize;
    for measurement in measurements {
        if let Some(problem) = check_measurement(measurement) {
            return Err(problem);
        }
        if get_str(measurement, "status") == Some("MEASURED") {
            measured_count += 1;
        }
    }
    if measured_count == 0 {
        return Err(blocked("NO_QUANTIFIED_PHYSICAL_OBSERVATIONS"));
    }
    Ok(())
}

fn audit_record(record: &Value) -> Assessment {
    if get_str(record, "schema_version") != Some(SCHEMA_VERSION)
        || get_str(record, "programme") != Some(BENCHMARK_ID)
        || get_str(record, "claim_ceiling")
            != Some("DATA_CONTRACT_AND_SYNTHETIC_AUDIT_ONLY_NO_PHYSICAL_OR_CLINICAL_VALIDATION")
    {
        return rejected("RECORD_IDENTITY_OR_CLAIM_CEILING_INVALID");
    }

    if !nonempty_string(record.get("record_id")) || !nonempty_string(record.get("indication")) {
        return rejected("RECORD_IDENTITY_OR_CLAIM_CEILING_INVALID");
    }
    if let Some(claims) = record.get("claims").and_then(Value::as_array) {
        let allowed = [
            "PHYSICAL_EXPOSURE_RECONSTRUCTABLE",
            "SPATIAL_UNIFORMITY_SUPPORTED",
            "BIOLOGICAL_RESPONSE_ASSOCIATED",
            "CLINICAL_OUTCOME_ASSOCIATED",
        ];
        if claims.iter().any(|claim| !claim.as_str().is_some_and(|s| allowed.contains(&s))) {
            return rejected("CLAIM_TYPE_INVALID");
        }
    }

    let modality = match get_str(record, "modality") {
        Some("DIRECT_CAP") => "DIRECT_CAP",
        Some("PLASMA_ACTIVATED_LIQUID") => "PLASMA_ACTIVATED_LIQUID",
        _ => return rejected("MODALITY_INVALID"),
    };

    let source = match record.get("source_configuration") {
        Some(value) => value,
        None => return blocked("SOURCE_CONFIGURATION_MISSING"),
    };
    if !nonempty_string(source.get("source_id"))
        || !nonempty_string(source.get("source_revision"))
        || !nonempty_string(source.get("discharge_architecture"))
        || !nonempty_string(source.get("gas").and_then(|g| g.get("composition")))
        || source.get("nominal_settings").and_then(Value::as_object).is_none_or(|o| o.is_empty())
    {
        return blocked("SOURCE_CONFIGURATION_INCOMPLETE");
    }

    let exposure = match record.get("exposure") {
        Some(value) => value,
        None => return blocked("EXPOSURE_MISSING"),
    };
    let exposure_type = get_str(exposure, "type");
    if exposure_type != Some(modality) {
        return rejected("MODALITY_TYPE_MISMATCH");
    }
    if !nonempty_string(exposure.get("exposure_event_id"))
        || !nonempty_string(exposure.get("event_time"))
        || !nonempty_string(exposure.get("interface"))
    {
        return blocked("EXPOSURE_EVENT_IDENTITY_INCOMPLETE");
    }

    if modality == "DIRECT_CAP" {
        let geometry = match exposure.get("geometry") {
            Some(value) => value,
            None => return blocked("SPATIAL_GEOMETRY_MISSING"),
        };
        if !nonempty_string(geometry.get("coordinate_frame"))
            || geometry.get("target_distance_mm").and_then(Value::as_f64).is_none_or(|v| v < 0.0)
            || geometry.get("treated_area_mm2").and_then(Value::as_f64).is_none_or(|v| v <= 0.0)
            || !nonempty_string(geometry.get("sampling_plan_id"))
            || geometry.get("sample_count").and_then(Value::as_u64).is_none_or(|v| v == 0)
        {
            return blocked("SPATIAL_GEOMETRY_INCOMPLETE");
        }
        if let Err(problem) = all_measurements(exposure.get("physical_observations")) {
            return problem;
        }
        if record.get("claims").and_then(Value::as_array).is_some_and(|claims| {
            claims.iter().any(|c| c.as_str() == Some("SPATIAL_UNIFORMITY_SUPPORTED"))
        }) {
            let summary = exposure.get("spatial_summary");
            if !is_sha256(summary.and_then(|s| get_str(s, "spatial_map_sha256")))
                || !is_sha256(geometry.get("spatial_map_sha256").and_then(Value::as_str))
                || geometry.get("coverage_fraction").and_then(Value::as_f64)
                    .is_none_or(|v| !(0.0..=1.0).contains(&v))
                || !summary.and_then(|s| s.get("uniformity_claimed")).and_then(Value::as_bool).unwrap_or(false)
                || !has_number(summary.and_then(|s| s.get("minimum")))
                || !has_number(summary.and_then(|s| s.get("mean")))
                || !has_number(summary.and_then(|s| s.get("maximum")))
                || !nonempty_string(summary.and_then(|s| s.get("unit")))
            {
                return blocked("SPATIAL_UNIFORMITY_EVIDENCE_INCOMPLETE");
            }
            let minimum = summary.and_then(|s| s.get("minimum")).and_then(Value::as_f64);
            let mean = summary.and_then(|s| s.get("mean")).and_then(Value::as_f64);
            let maximum = summary.and_then(|s| s.get("maximum")).and_then(Value::as_f64);
            if minimum.zip(mean).zip(maximum).is_none_or(|((min, avg), max)| min > avg || avg > max) {
                return blocked("SPATIAL_SUMMARY_ORDER_INVALID");
            }
            if !summary.and_then(|s| s.get("local_peak_reviewed")).and_then(Value::as_bool).unwrap_or(false) {
                return blocked("LOCAL_PEAK_NOT_REVIEWED");
            }
        }
        if exposure.get("spatial_summary").and_then(|s| s.get("uniformity_claimed"))
            .and_then(Value::as_bool).unwrap_or(false)
            && !exposure.get("spatial_summary").and_then(|s| s.get("local_peak_reviewed"))
                .and_then(Value::as_bool).unwrap_or(false)
        {
            return blocked("LOCAL_PEAK_NOT_REVIEWED");
        }
    } else {
        let lineage = match exposure.get("liquid_lineage") {
            Some(value) => value,
            None => return blocked("PAW_LINEAGE_MISSING"),
        };
        let required = [
            "matrix_id",
            "matrix_type",
            "generation_event_id",
            "generation_source_id",
            "container_material",
            "container_lot_id",
        ];
        if required.iter().any(|key| !nonempty_string(lineage.get(*key)))
            || !has_number(lineage.get("elapsed_minutes_at_application"))
        {
            return blocked("PAW_STABILITY_LINEAGE_INCOMPLETE");
        }
        let storage = lineage.get("storage");
        if !nonempty_string(storage.and_then(|s| s.get("condition")))
            || !has_number(storage.and_then(|s| s.get("temperature_c")))
            || !nonempty_string(storage.and_then(|s| s.get("stored_at")))
            || !nonempty_string(storage.and_then(|s| s.get("applied_at")))
        {
            return blocked("PAW_STORAGE_LINEAGE_INCOMPLETE");
        }
        if exposure.get("delivery").and_then(|d| d.get("volume_ml")).and_then(Value::as_f64)
                .is_none_or(|v| v <= 0.0)
            || exposure.get("delivery").and_then(|d| d.get("contact_duration_seconds"))
                .and_then(Value::as_f64).is_none_or(|v| v <= 0.0)
        {
            return blocked("PAW_DELIVERY_RECORD_INCOMPLETE");
        }
        if let Err(problem) = all_measurements(lineage.get("chemistry_observations")) {
            return problem;
        }
    }

    let biological_context = match record.get("biological_context") {
        Some(value) => value,
        None => return blocked("BIOLOGICAL_CONTEXT_MISSING"),
    };
    if !nonempty_string(biological_context.get("model_class"))
        || !nonempty_string(biological_context.get("model_id"))
        || !nonempty_string(biological_context.get("tissue_or_matrix"))
    {
        return blocked("BIOLOGICAL_CONTEXT_INCOMPLETE");
    }

    let exposure_id = get_str(exposure, "exposure_event_id");
    if let Some(observations) = record.get("biological_observations").and_then(Value::as_array) {
        for observation in observations {
            if get_str(observation, "linked_exposure_event_id") != exposure_id {
                return blocked("EXPOSURE_LINK_MISMATCH");
            }
            if !nonempty_string(observation.get("assay_id"))
                || observation.get("time_after_exposure_minutes").and_then(Value::as_f64).is_none()
            {
                return blocked("BIOLOGICAL_OBSERVATION_CONTEXT_INCOMPLETE");
            }
            let Some(measurement) = observation.get("measurement") else {
                return blocked("BIOLOGICAL_MEASUREMENT_MISSING");
            };
            if let Some(problem) = check_measurement(measurement) {
                return problem;
            }
        }
    } else {
        return blocked("BIOLOGICAL_OBSERVATIONS_ARRAY_MISSING");
    }

    if record.get("claims").and_then(Value::as_array).is_some_and(|claims| {
        claims.iter().any(|c| c.as_str() == Some("CLINICAL_OUTCOME_ASSOCIATED"))
    }) {
        let clinical = record.get("clinical_context");
        let required = [
            "protocol_id",
            "comparator_id",
            "standard_care_record_ref",
            "endpoint_definition",
            "analysis_plan_ref",
        ];
        if required.iter().any(|key| !nonempty_string(clinical.and_then(|c| c.get(*key))))
            || !has_number(clinical.and_then(|c| c.get("follow_up_days")))
        {
            return blocked("CLINICAL_CONTEXT_INCOMPLETE");
        }
    }

    let provenance = match record.get("provenance") {
        Some(value) => value,
        None => return blocked("PROVENANCE_MISSING"),
    };
    if !nonempty_string(provenance.get("recorded_at"))
        || !nonempty_string(provenance.get("software_revision"))
    {
        return blocked("PROVENANCE_INCOMPLETE");
    }
    let method_digest = get_str(provenance, "method_document_sha256");
    let raw_digest = get_str(provenance, "raw_data_manifest_sha256");
    if (method_digest.is_some() && !is_sha256(method_digest))
        || (raw_digest.is_some() && !is_sha256(raw_digest))
    {
        return rejected("PROVENANCE_DIGEST_INVALID");
    }
    if !is_sha256(method_digest) || !is_sha256(raw_digest) {
        return blocked("PROVENANCE_DIGEST_MISSING");
    }

    if record.get("claims").and_then(Value::as_array).is_some_and(|claims| {
        claims.iter().any(|c| c.as_str() == Some("BIOLOGICAL_RESPONSE_ASSOCIATED"))
    }) {
        let biological = record.get("biological_context");
        if !nonempty_string(biological.and_then(|b| b.get("model_class")))
            || !nonempty_string(biological.and_then(|b| b.get("model_id")))
            || !nonempty_string(biological.and_then(|b| b.get("tissue_or_matrix")))
            || record.get("biological_observations").and_then(Value::as_array).is_none_or(|a| a.is_empty())
        {
            return blocked("BIOLOGICAL_CONTEXT_INCOMPLETE");
        }
    }

    Assessment {
        disposition: Disposition::Pass,
        reason_code: if modality == "DIRECT_CAP" {
            "COMPLETE_DIRECT_CAP_RECORD"
        } else {
            "COMPLETE_PAW_LINEAGE"
        },
    }
}

fn evaluate(corpus: &Corpus) -> Result<Vec<(String, Assessment)>> {
    if corpus.schema_version != SCHEMA_VERSION
        || corpus.benchmark_id != BENCHMARK_ID
        || corpus.claim_ceiling != CLAIM_CEILING
        || corpus.cases.is_empty()
    {
        bail!("exposure fixture identity, version, claim ceiling or case set is invalid");
    }

    let mut ids = std::collections::HashSet::new();
    let mut results = Vec::with_capacity(corpus.cases.len());
    for case in &corpus.cases {
        if case.id.trim().is_empty() || !ids.insert(case.id.as_str()) {
            bail!("empty or duplicate exposure scenario id: {}", case.id);
        }
        let assessment = audit_record(&case.record);
        if assessment.disposition.as_str() != case.expected_disposition {
            bail!(
                "known-answer mismatch for {}: expected {}, got {} ({})",
                case.id,
                case.expected_disposition,
                assessment.disposition.as_str(),
                assessment.reason_code
            );
        }
        if assessment.reason_code != case.reason_code {
            bail!(
                "reason-code mismatch for {}: expected {}, got {}",
                case.id,
                case.reason_code,
                assessment.reason_code
            );
        }
        results.push((case.id.clone(), assessment));
    }

    let pass = results.iter().filter(|(_, a)| a.disposition == Disposition::Pass).count();
    let blocked = results.iter().filter(|(_, a)| a.disposition == Disposition::Blocked).count();
    let rejected = results.iter().filter(|(_, a)| a.disposition == Disposition::Rejected).count();
    if pass != corpus.expected_counts.pass
        || blocked != corpus.expected_counts.blocked
        || rejected != corpus.expected_counts.rejected
    {
        bail!("exposure fixture disposition counts do not match evaluated outcomes");
    }
    Ok(results)
}

pub fn run(root: &Path) -> Result<()> {
    let fixture_path = root.join("data/research/plasma_medical_exposure_fixture_v0_1.json");
    let fixture_text = fs::read_to_string(&fixture_path)
        .with_context(|| format!("reading {}", fixture_path.display()))?;
    let corpus: Corpus = serde_json::from_str(&fixture_text)
        .with_context(|| format!("parsing {}", fixture_path.display()))?;
    let results = evaluate(&corpus)?;
    let pass = results.iter().filter(|(_, a)| a.disposition == Disposition::Pass).count();
    let blocked = results.iter().filter(|(_, a)| a.disposition == Disposition::Blocked).count();
    let rejected = results.iter().filter(|(_, a)| a.disposition == Disposition::Rejected).count();
    println!(
        "{}: {}/{} known-answer cases matched (PASS {}, BLOCKED {}, REJECTED {})",
        BENCHMARK_ID,
        results.len(),
        results.len(),
        pass,
        blocked,
        rejected
    );
    for (id, assessment) in results {
        println!("{} {} — {}", assessment.disposition.as_str(), id, assessment.reason_code);
    }
    println!("Synthetic data-contract audit only; no physical or clinical validation.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Corpus {
        let text = include_str!("../../data/research/plasma_medical_exposure_fixture_v0_1.json");
        serde_json::from_str(text).expect("exposure fixture parses")
    }

    #[test]
    fn all_exposure_reconstructability_cases_replay_with_frozen_counts() {
        let corpus = corpus();
        let results = evaluate(&corpus).expect("known-answer replay");
        assert_eq!(results.len(), 14);
        assert_eq!(corpus.expected_counts.pass, 2);
        assert_eq!(corpus.expected_counts.blocked, 8);
        assert_eq!(corpus.expected_counts.rejected, 4);
    }

    #[test]
    fn source_settings_alone_are_not_physical_exposure() {
        let corpus = corpus();
        let case = corpus.cases.iter()
            .find(|case| case.id == "NOMINAL_SETTINGS_ONLY_NOT_MEASURED_EXPOSURE")
            .expect("settings-only case");
        assert_eq!(audit_record(&case.record).disposition, Disposition::Blocked);
    }

    #[test]
    fn unknown_and_below_limit_are_not_numeric_zero() {
        let corpus = corpus();
        for id in [
            "UNKNOWN_CANNOT_CARRY_NUMERIC_ZERO",
            "BELOW_LIMIT_IS_NOT_MEASURED_ZERO",
        ] {
            let case = corpus.cases.iter().find(|case| case.id == id).expect("negative control");
            assert_eq!(audit_record(&case.record).disposition, Disposition::Rejected);
        }
    }

    #[test]
    fn expired_calibration_cannot_support_reconstructability() {
        let corpus = corpus();
        let case = corpus.cases.iter()
            .find(|case| case.id == "EXPIRED_CALIBRATION_BLOCKS_TRACEABILITY")
            .expect("expired calibration case");
        assert_eq!(audit_record(&case.record).disposition, Disposition::Blocked);
    }

    #[test]
    fn direct_cap_and_paw_require_different_exposure_branches() {
        let corpus = corpus();
        let case = corpus.cases.iter()
            .find(|case| case.id == "MODALITY_DOES_NOT_MATCH_EXPOSURE_TYPE")
            .expect("modality mismatch case");
        assert_eq!(audit_record(&case.record).disposition, Disposition::Rejected);
    }
}
