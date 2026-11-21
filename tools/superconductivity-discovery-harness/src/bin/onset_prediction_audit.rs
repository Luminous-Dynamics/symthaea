use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Debug)]
struct Evidence {
    pair_id: String,
    kind: String,
    lower_k: Option<f64>,
    upper_k: Option<f64>,
    exact_k: Option<f64>,
    observations: Vec<EvidenceObservation>,
}
#[derive(Clone, Debug)]
struct EvidenceObservation {
    id: String,
    sample_id: String,
    execution_run_id: String,
    conditions_id: String,
    characterization_method: String,
    outcome: String,
    dwell_k: f64,
    observed_onset_k: Option<f64>,
    measurement_uncertainty_k: Option<f64>,
    detection_limit_wt_percent: f64,
    raw_data_sha256: String,
}
#[derive(Clone, Copy, Debug)]
struct Prediction {
    median_k: f64,
    log_sigma: f64,
}
#[derive(Clone, Debug)]
struct Model {
    id: String,
    artifact_sha256: String,
    uncertainty_method: String,
    predictions: HashMap<String, Prediction>,
}

fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|x| !x.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn number_field(value: &Value, key: &str) -> Result<f64, String> {
    let result = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric field {key}"))?;
    if !result.is_finite() {
        return Err(format!("{key} must be finite"));
    }
    Ok(result)
}
fn optional_number(value: &Value, key: &str) -> Result<Option<f64>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(number) => {
            let result = number
                .as_f64()
                .ok_or_else(|| format!("{key} must be numeric or null"))?;
            if !result.is_finite() {
                return Err(format!("{key} must be finite"));
            }
            Ok(Some(result))
        }
    }
}
fn string_list(value: &Value, key: &str) -> Result<Vec<String>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))?;
    let mut output = Vec::with_capacity(values.len());
    let mut seen = HashSet::new();
    for item in values {
        let text = item
            .as_str()
            .filter(|x| !x.trim().is_empty())
            .ok_or_else(|| format!("{key} contains a non-string or empty value"))?;
        if !seen.insert(text.to_owned()) {
            return Err(format!("{key} contains duplicate value {text}"));
        }
        output.push(text.to_owned());
    }
    if output.is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    Ok(output)
}
fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn parse_observation(value: &Value, pair_id: &str) -> Result<EvidenceObservation, String> {
    let id = text_field(value, "observation_id")?.to_owned();
    let sample_id = text_field(value, "sample_id")?.to_owned();
    let execution_run_id = text_field(value, "execution_run_id")?.to_owned();
    let conditions_id = text_field(value, "conditions_id")?.to_owned();
    let characterization_method = text_field(value, "characterization_method")?.to_owned();
    let outcome = text_field(value, "outcome")?.to_owned();
    let dwell_k = number_field(value, "dwell_temperature_k")?;
    let observed_onset_k = optional_number(value, "observed_onset_k")?;
    let measurement_uncertainty_k = optional_number(value, "measurement_uncertainty_k")?;
    let detection_limit_wt_percent = number_field(value, "detection_limit_wt_percent")?;
    let raw_data_sha256 = text_field(value, "raw_data_sha256")?.to_owned();
    if dwell_k <= 0.0 || detection_limit_wt_percent <= 0.0 || detection_limit_wt_percent > 100.0 {
        return Err(format!(
            "pair {pair_id} observation {id} has invalid temperature or detection limit"
        ));
    }
    if !valid_sha256(&raw_data_sha256) {
        return Err(format!(
            "pair {pair_id} observation {id} raw_data_sha256 must be 64 hexadecimal characters"
        ));
    }
    match outcome.as_str() {
        "cross_precursor_product_detected"
        | "no_cross_precursor_product"
        | "transformed_only"
        | "uncharacterizable" => {
            if observed_onset_k.is_some() || measurement_uncertainty_k.is_some() {
                return Err(format!(
                    "pair {pair_id} observation {id} must not attach an exact onset to a categorical dwell result"
                ));
            }
        }
        "exact_onset_measured" => {
            let onset = observed_onset_k.ok_or_else(|| {
                format!("pair {pair_id} exact observation {id} lacks observed_onset_k")
            })?;
            let uncertainty = measurement_uncertainty_k.ok_or_else(|| {
                format!("pair {pair_id} exact observation {id} lacks measurement_uncertainty_k")
            })?;
            if onset <= 0.0 || uncertainty <= 0.0 || (onset - dwell_k).abs() > uncertainty {
                return Err(format!(
                    "pair {pair_id} exact observation {id} has inconsistent measured onset and dwell temperature"
                ));
            }
        }
        _ => {
            return Err(format!(
                "pair {pair_id} observation {id} has unsupported outcome {outcome}"
            ));
        }
    }
    Ok(EvidenceObservation {
        id,
        sample_id,
        execution_run_id,
        conditions_id,
        characterization_method,
        outcome,
        dwell_k,
        observed_onset_k,
        measurement_uncertainty_k,
        detection_limit_wt_percent,
        raw_data_sha256,
    })
}
fn same_temperature(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9 * left.abs().max(right.abs()).max(1.0)
}
fn parse_evidence(value: &Value) -> Result<Evidence, String> {
    let pair_id = text_field(value, "pair_id")?.to_owned();
    let kind = text_field(value, "evidence_kind")?.to_owned();
    let lower_k = optional_number(value, "lower_bound_k")?;
    let upper_k = optional_number(value, "upper_bound_k")?;
    let exact_k = optional_number(value, "exact_onset_k")?;
    let precursors = string_list(value, "precursor_ids")?;
    if precursors.len() != 2 || precursors[0] == precursors[1] {
        return Err(format!(
            "pair {pair_id} must identify exactly two distinct precursor IDs"
        ));
    }
    let observation_rows = value
        .get("evidence_observations")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("pair {pair_id} evidence_observations must be an array"))?;
    if observation_rows.is_empty() {
        return Err(format!(
            "pair {pair_id} must contain at least one evidence observation"
        ));
    }
    let mut observations = Vec::with_capacity(observation_rows.len());
    let mut observation_ids = HashSet::new();
    for row in observation_rows {
        let observation = parse_observation(row, &pair_id)?;
        if !observation_ids.insert(observation.id.clone()) {
            return Err(format!(
                "pair {pair_id} contains duplicate observation ID {}",
                observation.id
            ));
        }
        observations.push(observation);
    }
    let positive = |x: Option<f64>| x.is_some_and(|v| v.is_finite() && v > 0.0);
    let has_outcome_at = |outcome: &str, temperature: f64| {
        observations.iter().any(|observation| {
            observation.outcome == outcome && same_temperature(observation.dwell_k, temperature)
        })
    };
    let has_no_cross_product_at = |temperature: f64| {
        observations.iter().any(|observation| {
            (observation.outcome == "no_cross_precursor_product"
                || observation.outcome == "transformed_only")
                && same_temperature(observation.dwell_k, temperature)
        })
    };
    let valid = match kind.as_str() {
        "left_censored" => {
            lower_k.is_none()
                && positive(upper_k)
                && exact_k.is_none()
                && has_outcome_at(
                    "cross_precursor_product_detected",
                    upper_k.unwrap_or_default(),
                )
        }
        "right_censored" => {
            positive(lower_k)
                && upper_k.is_none()
                && exact_k.is_none()
                && has_no_cross_product_at(lower_k.unwrap_or_default())
        }
        "interval_censored" => {
            positive(lower_k)
                && positive(upper_k)
                && exact_k.is_none()
                && lower_k.zip(upper_k).is_some_and(|(lower, upper)| {
                    lower < upper
                        && has_no_cross_product_at(lower)
                        && has_outcome_at("cross_precursor_product_detected", upper)
                })
        }
        "exact" => {
            lower_k.is_none()
                && upper_k.is_none()
                && positive(exact_k)
                && observations.iter().any(|observation| {
                    observation.outcome == "exact_onset_measured"
                        && observation.observed_onset_k.is_some_and(|onset| {
                            same_temperature(onset, exact_k.unwrap_or_default())
                        })
                })
        }
        _ => false,
    };
    if !valid {
        return Err(format!(
            "pair {pair_id} has inconsistent {kind} onset bounds or lacks matching characterized evidence"
        ));
    }
    Ok(Evidence {
        pair_id,
        kind,
        lower_k,
        upper_k,
        exact_k,
        observations,
    })
}
fn parse_model(value: &Value, evaluation_ids: &HashSet<String>) -> Result<Model, String> {
    let id = text_field(value, "model_id")?.to_owned();
    let artifact_sha256 = text_field(value, "model_artifact_sha256")?.to_owned();
    if !valid_sha256(&artifact_sha256) {
        return Err(format!(
            "model {id} artifact digest must be 64 hexadecimal characters"
        ));
    }
    let uncertainty_method = text_field(value, "uncertainty_method")?.to_owned();
    let training_ids = string_list(value, "training_pair_ids")?;
    if let Some(overlap) = training_ids
        .iter()
        .find(|pair_id| evaluation_ids.contains(*pair_id))
    {
        return Err(format!(
            "model {id} has evaluation leakage: {overlap} appears in training_pair_ids"
        ));
    }
    let rows = value
        .get("predictions")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("model {id} predictions must be an array"))?;
    let mut predictions = HashMap::new();
    for row in rows {
        let pair_id = text_field(row, "pair_id")?.to_owned();
        if !evaluation_ids.contains(&pair_id) {
            return Err(format!(
                "model {id} predicts unknown evaluation pair {pair_id}"
            ));
        }
        let median_k = number_field(row, "median_onset_k")?;
        let log_sigma = number_field(row, "log_sigma")?;
        if median_k <= 0.0 || log_sigma <= 0.0 || log_sigma > 2.0 {
            return Err(format!(
                "model {id} prediction for {pair_id} requires median_onset_k > 0 and log_sigma in (0,2]"
            ));
        }
        if predictions
            .insert(
                pair_id.clone(),
                Prediction {
                    median_k,
                    log_sigma,
                },
            )
            .is_some()
        {
            return Err(format!("model {id} has duplicate prediction for {pair_id}"));
        }
    }
    if predictions.len() != evaluation_ids.len() {
        return Err(format!(
            "model {id} must provide exactly one prediction for each evaluation pair (expected {}, got {})",
            evaluation_ids.len(),
            predictions.len()
        ));
    }
    Ok(Model {
        id,
        artifact_sha256,
        uncertainty_method,
        predictions,
    })
}

// Abramowitz-Stegun 7.1.26 approximation; absolute error is below 8e-8.
fn normal_cdf(x: f64) -> f64 {
    const INV_SQRT_2_PI: f64 = 0.398_942_280_401_432_7;
    let absolute = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * absolute);
    let polynomial = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let tail = INV_SQRT_2_PI * (-0.5 * absolute * absolute).exp() * polynomial;
    if x >= 0.0 { 1.0 - tail } else { tail }
}
fn normal_interval_probability(lower_z: f64, upper_z: f64) -> f64 {
    if lower_z >= upper_z {
        return 0.0;
    }
    if lower_z >= 0.0 {
        normal_cdf(-lower_z) - normal_cdf(-upper_z)
    } else {
        normal_cdf(upper_z) - normal_cdf(lower_z)
    }
}
fn observed_log_probability(evidence: &Evidence, prediction: Prediction) -> Result<f64, String> {
    let sigma = prediction.log_sigma;
    let z_for = |temperature_k: f64| (temperature_k / prediction.median_k).ln() / sigma;
    let log_probability = match evidence.kind.as_str() {
        "left_censored" => {
            let probability = normal_cdf(z_for(evidence.upper_k.expect("validated upper bound")));
            if probability > 0.0 {
                probability.ln()
            } else {
                f64::NEG_INFINITY
            }
        }
        "right_censored" => {
            let probability = normal_cdf(-z_for(evidence.lower_k.expect("validated lower bound")));
            if probability > 0.0 {
                probability.ln()
            } else {
                f64::NEG_INFINITY
            }
        }
        "interval_censored" => {
            let lower = z_for(evidence.lower_k.expect("validated lower bound"));
            let upper = z_for(evidence.upper_k.expect("validated upper bound"));
            let probability = normal_interval_probability(lower, upper);
            if probability > 0.0 {
                probability.ln()
            } else {
                f64::NEG_INFINITY
            }
        }
        "exact" => {
            let observed = evidence.exact_k.expect("validated exact onset");
            let z = z_for(observed);
            -observed.ln() - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5 * z * z
        }
        _ => return Err(format!("unsupported evidence kind {}", evidence.kind)),
    };
    if !log_probability.is_finite() {
        return Err(format!(
            "model assigns numerically zero probability to pair {} ({})",
            evidence.pair_id, evidence.kind
        ));
    }
    Ok(log_probability)
}
fn model_receipt(model: &Model, evidence: &[Evidence]) -> Result<Value, String> {
    let mut total_nll = 0.0;
    let mut class_scores: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let mut exact_absolute_error_c = 0.0;
    let mut exact_squared_error_c = 0.0;
    let mut exact_interval_width_k = 0.0;
    let mut exact_covered_90 = 0usize;
    let mut exact_count = 0usize;
    let mut per_pair = Vec::with_capacity(evidence.len());
    const Z_90: f64 = 1.644_853_626_951_472_2;
    for row in evidence {
        let prediction = *model.predictions.get(&row.pair_id).ok_or_else(|| {
            format!(
                "model {} is missing prediction for {}",
                model.id, row.pair_id
            )
        })?;
        let log_probability = observed_log_probability(row, prediction)?;
        let nll = -log_probability;
        total_nll += nll;
        let score = class_scores.entry(row.kind.clone()).or_insert((0.0, 0));
        score.0 += nll;
        score.1 += 1;
        if let Some(exact_k) = row.exact_k {
            let error_c = prediction.median_k - exact_k;
            exact_absolute_error_c += error_c.abs();
            exact_squared_error_c += error_c * error_c;
            let lower = prediction.median_k * (-Z_90 * prediction.log_sigma).exp();
            let upper = prediction.median_k * (Z_90 * prediction.log_sigma).exp();
            exact_interval_width_k += upper - lower;
            let standardized = (exact_k / prediction.median_k).ln() / prediction.log_sigma;
            if standardized.abs() <= Z_90 {
                exact_covered_90 += 1;
            }
            exact_count += 1;
        }
        per_pair.push(json!({
            "pair_id": row.pair_id,
            "evidence_kind": row.kind,
            "evidence_observations": row.observations.iter().map(|observation| json!({
                "observation_id": observation.id,
                "sample_id": observation.sample_id,
                "execution_run_id": observation.execution_run_id,
                "conditions_id": observation.conditions_id,
                "characterization_method": observation.characterization_method,
                "outcome": observation.outcome,
                "dwell_temperature_k": observation.dwell_k,
                "observed_onset_k": observation.observed_onset_k,
                "measurement_uncertainty_k": observation.measurement_uncertainty_k,
                "detection_limit_wt_percent": observation.detection_limit_wt_percent,
                "raw_data_sha256": observation.raw_data_sha256
            })).collect::<Vec<_>>(),
            "median_prediction_k": prediction.median_k,
            "log_sigma": prediction.log_sigma,
            "observed_log_probability": log_probability,
            "negative_log_likelihood": nll
        }));
    }
    let mut by_kind = BTreeMap::new();
    for (kind, (sum, count)) in class_scores {
        by_kind.insert(
            kind,
            json!({
                "count": count,
                "mean_negative_log_likelihood": sum / count as f64
            }),
        );
    }
    let exact_metrics = if exact_count > 0 {
        json!({
            "count": exact_count,
            "mae_c": exact_absolute_error_c / exact_count as f64,
            "rmse_c": (exact_squared_error_c / exact_count as f64).sqrt(),
            "90_percent_prediction_interval_coverage": exact_covered_90 as f64 / exact_count as f64,
            "mean_90_percent_prediction_interval_width_k": exact_interval_width_k / exact_count as f64,
            "scope": "exact in-situ onset measurements only; censored observations are excluded"
        })
    } else {
        json!({
            "count": 0,
            "mae_c": null,
            "rmse_c": null,
            "90_percent_prediction_interval_coverage": null,
            "mean_90_percent_prediction_interval_width_k": null,
            "scope": "no exact onset measurements in this evaluation set"
        })
    };
    Ok(json!({
        "model_id": model.id,
        "model_artifact_sha256": model.artifact_sha256,
        "uncertainty_method": model.uncertainty_method,
        "pair_count": evidence.len(),
        "mean_censored_and_exact_negative_log_likelihood": total_nll / evidence.len() as f64,
        "negative_log_likelihood_by_evidence_kind": by_kind,
        "exact_measurement_metrics": exact_metrics,
        "per_pair_scores": per_pair
    }))
}
fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator =
        validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!(
            "onset-prediction schema validation failed: {error} at {}",
            error.instance_path()
        ));
    }
    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("onset-prediction auditor accepts synthetic fixtures only".to_owned());
    }
    if text_field(&fixture, "contract")? != "SC-ONSET-PREDICTION-AUDIT-0.1" {
        return Err("unsupported onset-prediction contract".to_owned());
    }
    if text_field(&fixture, "temperature_unit")? != "kelvin" {
        return Err(
            "log-normal onset likelihood requires absolute temperature in kelvin".to_owned(),
        );
    }
    let rows = fixture
        .get("pairs")
        .and_then(Value::as_array)
        .ok_or("pairs must be an array")?;
    let mut evidence = Vec::with_capacity(rows.len());
    let mut evaluation_ids = HashSet::new();
    for row in rows {
        let parsed = parse_evidence(row)?;
        if !evaluation_ids.insert(parsed.pair_id.clone()) {
            return Err(format!("duplicate evaluation pair {}", parsed.pair_id));
        }
        evidence.push(parsed);
    }
    let models = fixture
        .get("models")
        .and_then(Value::as_array)
        .ok_or("models must be an array")?;
    let mut model_ids = HashSet::new();
    let mut receipts = Vec::with_capacity(models.len());
    for row in models {
        let model = parse_model(row, &evaluation_ids)?;
        if !model_ids.insert(model.id.clone()) {
            return Err(format!("duplicate model id {}", model.id));
        }
        receipts.push(model_receipt(&model, &evidence)?);
    }
    Ok(json!({
        "status": "synthetic_onset_prediction_audit_pass",
        "contract": "SC-ONSET-PREDICTION-AUDIT-0.1",
        "benchmark_id": text_field(&fixture, "benchmark_id")?,
        "source_reference": text_field(&fixture, "source_reference")?,
        "temperature_unit": "kelvin",
        "schema_sha256": format!("{:x}", Sha256::digest(schema_bytes)),
        "fixture_sha256": format!("{:x}", Sha256::digest(fixture_bytes)),
        "pair_count": evidence.len(),
        "evidence_kind_counts": {
            "left_censored": evidence.iter().filter(|x| x.kind == "left_censored").count(),
            "right_censored": evidence.iter().filter(|x| x.kind == "right_censored").count(),
            "interval_censored": evidence.iter().filter(|x| x.kind == "interval_censored").count(),
            "exact": evidence.iter().filter(|x| x.kind == "exact").count()
        },
        "models": receipts,
        "claim_ceiling": "Synthetic onset-evaluation plumbing only. No real reaction onset, model accuracy, phase yield, material synthesizability or laboratory outcome is established."
    }))
}
fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}
fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <fixture.json> [--out <receipt.json>]"
    );
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut fixture_path = None;
    let mut output_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--fixture" => fixture_path = Some(path),
                    "--out" => output_path = Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let receipt = audit(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|error| {
        eprintln!("SC-ONSET-PREDICTION-FAIL: {error}");
        process::exit(1);
    });
    let rendered =
        serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");
    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/superconductivity_onset_prediction_v0_1.schema.json")).unwrap(),
            fs::read(root.join("tools/superconductivity-discovery-harness/fixtures/onset_prediction_smoke_v0_1.json")).unwrap(),
        )
    }

    #[test]
    fn smoke_fixture_scores_censored_and_exact_observations_separately() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        assert_eq!(receipt["pair_count"], 6);
        assert_eq!(receipt["evidence_kind_counts"]["left_censored"], 2);
        assert_eq!(receipt["evidence_kind_counts"]["right_censored"], 1);
        assert_eq!(receipt["evidence_kind_counts"]["interval_censored"], 1);
        assert_eq!(receipt["evidence_kind_counts"]["exact"], 2);
        let candidate = receipt["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["model_id"] == "synthetic_compact_aft")
            .unwrap();
        assert_eq!(candidate["exact_measurement_metrics"]["count"], 2);
        assert_eq!(candidate["exact_measurement_metrics"]["mae_c"], 30.0);
    }

    #[test]
    fn left_right_and_interval_censoring_receive_finite_log_scores() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        for model in receipt["models"].as_array().unwrap() {
            assert!(
                model["mean_censored_and_exact_negative_log_likelihood"]
                    .as_f64()
                    .unwrap()
                    .is_finite()
            );
            let by_kind = &model["negative_log_likelihood_by_evidence_kind"];
            for kind in [
                "left_censored",
                "right_censored",
                "interval_censored",
                "exact",
            ] {
                assert!(
                    by_kind[kind]["mean_negative_log_likelihood"]
                        .as_f64()
                        .unwrap()
                        .is_finite()
                );
            }
        }
    }

    #[test]
    fn exact_error_metrics_never_count_censored_rows() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        for model in receipt["models"].as_array().unwrap() {
            assert_eq!(model["exact_measurement_metrics"]["count"], 2);
            assert!(
                model["exact_measurement_metrics"]["scope"]
                    .as_str()
                    .unwrap()
                    .contains("exact in-situ")
            );
        }
    }

    #[test]
    fn evaluation_pair_in_training_set_is_rejected() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["models"][0]["training_pair_ids"]
            .as_array_mut()
            .unwrap()
            .push(json!("synthetic_pair_01"));
        let error = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("evaluation leakage"));
    }

    #[test]
    fn duplicate_predictions_are_rejected() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        let duplicate = data["models"][0]["predictions"][0].clone();
        data["models"][0]["predictions"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        let error = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("duplicate prediction"));
    }

    #[test]
    fn invalid_interval_order_is_rejected() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["pairs"][2]["lower_bound_k"] = json!(900.0);
        data["pairs"][2]["upper_bound_k"] = json!(800.0);
        let error = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("inconsistent interval_censored"));
    }

    #[test]
    fn malformed_artifact_digest_is_rejected() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["models"][0]["model_artifact_sha256"] = json!("not-a-digest");
        let error = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("artifact digest"));
    }
    #[test]
    fn transformed_only_is_negative_evidence_only_for_cross_precursor_product_onset() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["pairs"][1]["evidence_observations"][0]["outcome"] = json!("transformed_only");
        let receipt = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap();
        let pair = receipt["models"][0]["per_pair_scores"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["pair_id"] == "synthetic_pair_02")
            .unwrap();
        assert_eq!(pair["evidence_kind"], "right_censored");
        assert_eq!(
            pair["evidence_observations"][0]["outcome"],
            "transformed_only"
        );
    }

    #[test]
    fn uncharacterizable_observation_cannot_establish_a_negative_product_bound() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["pairs"][1]["evidence_observations"][0]["outcome"] = json!("uncharacterizable");
        let error = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("lacks matching characterized evidence"));
    }

    #[test]
    fn interval_censoring_requires_both_nonreaction_and_reaction_evidence() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["pairs"][2]["evidence_observations"][0]["outcome"] = json!("uncharacterizable");
        let error = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("lacks matching characterized evidence"));
    }
}
