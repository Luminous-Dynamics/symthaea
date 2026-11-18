use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Copy, Debug)]
struct Matrix2 {
    a: f64,
    b: f64,
    d: f64,
}

#[derive(Clone, Copy, Debug)]
struct CalibrationRow {
    h: [f64; 2],
    observed: f64,
    uncertainty: f64,
}

#[derive(Clone, Copy, Debug)]
struct GaussianPosterior {
    mean: [f64; 2],
    covariance: Matrix2,
}

impl Matrix2 {
    fn determinant(self) -> f64 {
        self.a * self.d - self.b * self.b
    }

    fn inverse(self) -> Result<Self, String> {
        let determinant = self.determinant();
        if !determinant.is_finite() || determinant <= 1.0e-12 {
            return Err("precision matrix must be symmetric positive definite".to_owned());
        }
        Ok(Self {
            a: self.d / determinant,
            b: -self.b / determinant,
            d: self.a / determinant,
        })
    }

    fn add_information(self, h: [f64; 2], weight: f64) -> Self {
        Self {
            a: self.a + weight * h[0] * h[0],
            b: self.b + weight * h[0] * h[1],
            d: self.d + weight * h[1] * h[1],
        }
    }

    fn quadratic(self, h: [f64; 2]) -> f64 {
        self.a * h[0] * h[0] + 2.0 * self.b * h[0] * h[1] + self.d * h[1] * h[1]
    }
}

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty text field {key}"))
}

fn finite_number(value: &Value, context: &str) -> Result<f64, String> {
    let number = value
        .as_f64()
        .ok_or_else(|| format!("{context} must be numeric"))?;
    if !number.is_finite() {
        return Err(format!("{context} must be finite"));
    }
    Ok(number)
}

fn parse_vector2(value: &Value, context: &str) -> Result<[f64; 2], String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a two-element array"))?;
    if values.len() != 2 {
        return Err(format!("{context} must have exactly two entries"));
    }
    Ok([
        finite_number(&values[0], &format!("{context}[0]"))?,
        finite_number(&values[1], &format!("{context}[1]"))?,
    ])
}

fn parse_precision(value: &Value, context: &str) -> Result<Matrix2, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a 2x2 matrix"))?;
    if rows.len() != 2 {
        return Err(format!("{context} must have two rows"));
    }
    let row0 = rows[0]
        .as_array()
        .ok_or_else(|| format!("{context} row 0 must be an array"))?;
    let row1 = rows[1]
        .as_array()
        .ok_or_else(|| format!("{context} row 1 must be an array"))?;
    if row0.len() != 2 || row1.len() != 2 {
        return Err(format!("{context} must be exactly 2x2"));
    }
    let a = finite_number(&row0[0], context)?;
    let b0 = finite_number(&row0[1], context)?;
    let b1 = finite_number(&row1[0], context)?;
    let d = finite_number(&row1[1], context)?;
    if (b0 - b1).abs() > 1.0e-10 {
        return Err(format!("{context} must be symmetric"));
    }
    let matrix = Matrix2 {
        a,
        b: 0.5 * (b0 + b1),
        d,
    };
    let _ = matrix.inverse()?;
    Ok(matrix)
}

fn conjugate_posterior(
    prior_mean: [f64; 2],
    prior_precision: Matrix2,
    observations: &[CalibrationRow],
) -> Result<GaussianPosterior, String> {
    let mut posterior_precision = prior_precision;
    let mut information_vector = [
        prior_precision.a * prior_mean[0] + prior_precision.b * prior_mean[1],
        prior_precision.b * prior_mean[0] + prior_precision.d * prior_mean[1],
    ];
    for observation in observations {
        if !observation.uncertainty.is_finite() || observation.uncertainty <= 0.0 {
            return Err("calibration uncertainty must be positive and finite".to_owned());
        }
        if observation.h.iter().any(|value| !value.is_finite()) || !observation.observed.is_finite()
        {
            return Err("calibration row contains non-finite values".to_owned());
        }
        let weight = 1.0 / (observation.uncertainty * observation.uncertainty);
        posterior_precision = posterior_precision.add_information(observation.h, weight);
        information_vector[0] += weight * observation.h[0] * observation.observed;
        information_vector[1] += weight * observation.h[1] * observation.observed;
    }
    let covariance = posterior_precision.inverse()?;
    let mean = [
        covariance.a * information_vector[0] + covariance.b * information_vector[1],
        covariance.b * information_vector[0] + covariance.d * information_vector[1],
    ];
    if mean.iter().any(|value| !value.is_finite()) {
        return Err("posterior mean is non-finite".to_owned());
    }
    Ok(GaussianPosterior { mean, covariance })
}

fn observation_operator(observation: &Value) -> Result<[f64; 2], String> {
    parse_vector2(
        observation
            .get("forward_operator_coefficients")
            .ok_or_else(|| "observation missing forward_operator_coefficients".to_owned())?,
        "forward_operator_coefficients",
    )
}

fn fit_dataset(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
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
            "inverse-fit schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("initial inverse fitter accepts synthetic fixtures only".to_owned());
    }
    if required_text(&fixture, "contract")? != "SC-OBSERVATION-INVERSE-0.1" {
        return Err("unsupported observation-inverse contract".to_owned());
    }

    let model_target = fixture
        .get("model_target")
        .ok_or_else(|| "missing model_target".to_owned())?;
    if required_text(model_target, "model_family")? != "lattice_ginzburg_landau" {
        return Err(
            "initial conjugate fitter is restricted to lattice_ginzburg_landau targets".to_owned(),
        );
    }
    let parameter_names = model_target
        .get("target_parameters")
        .and_then(Value::as_array)
        .ok_or_else(|| "target_parameters must be an array".to_owned())?;
    if parameter_names.len() != 2 {
        return Err("initial Gaussian fitter requires exactly two target parameters".to_owned());
    }
    let parameter_names: Vec<String> = parameter_names
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|text| !text.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| "parameter names must be non-empty strings".to_owned())
        })
        .collect::<Result<_, _>>()?;
    if parameter_names[0] == parameter_names[1] {
        return Err("parameter names must be unique".to_owned());
    }
    let prior_mean = parse_vector2(
        model_target
            .get("parameter_prior_mean")
            .ok_or_else(|| "missing parameter_prior_mean".to_owned())?,
        "parameter_prior_mean",
    )?;
    let prior_precision = parse_precision(
        model_target
            .get("parameter_prior_precision")
            .ok_or_else(|| "missing parameter_prior_precision".to_owned())?,
        "parameter_prior_precision",
    )?;

    let observations = fixture
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| "observations must be an array".to_owned())?;
    let mut observation_by_id = HashMap::with_capacity(observations.len());
    for observation in observations {
        let id = required_text(observation, "observation_id")?.to_owned();
        if observation_by_id.insert(id.clone(), observation).is_some() {
            return Err(format!("duplicate observation ID {id}"));
        }
        let _ = observation_operator(observation)?;
    }

    let split = fixture
        .get("split")
        .ok_or_else(|| "missing split".to_owned())?;
    if required_text(split, "holdout_rule")? != "leave_one_modality_out" {
        return Err("initial Gaussian fitter supports leave_one_modality_out only".to_owned());
    }
    let calibration_ids = split
        .get("calibration_observation_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "calibration IDs must be an array".to_owned())?;
    let heldout_ids = split
        .get("heldout_observation_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "heldout IDs must be an array".to_owned())?;
    if calibration_ids.is_empty() || heldout_ids.is_empty() {
        return Err("calibration and heldout partitions must both be non-empty".to_owned());
    }
    let pre_registered_at = required_text(split, "pre_registered_at")?;
    if pre_registered_at.len() != 20 || !pre_registered_at.ends_with('Z') {
        return Err("split pre_registered_at must be UTC RFC3339 seconds".to_owned());
    }
    for observation in observations {
        let conditions = observation
            .get("conditions")
            .ok_or_else(|| "observation missing conditions".to_owned())?;
        let timestamp = required_text(conditions, "measurement_timestamp")?;
        if timestamp.len() != 20 || !timestamp.ends_with('Z') || timestamp < pre_registered_at {
            return Err(
                "split must be preregistered before every UTC observation timestamp".to_owned(),
            );
        }
    }

    let mut calibration_set = HashSet::new();
    let mut heldout_set = HashSet::new();
    let mut calibration_order = Vec::with_capacity(calibration_ids.len());
    let mut heldout_order = Vec::with_capacity(heldout_ids.len());
    let mut calibration_modalities = HashSet::new();
    let mut heldout_modalities = HashSet::new();
    for id_value in calibration_ids {
        let id = id_value
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| "calibration IDs must be non-empty strings".to_owned())?;
        if !calibration_set.insert(id.to_owned()) {
            return Err(format!("duplicate calibration observation {id}"));
        }
        calibration_order.push(id.to_owned());
        let observation = observation_by_id
            .get(id)
            .ok_or_else(|| format!("unknown calibration observation {id}"))?;
        calibration_modalities.insert(required_text(observation, "modality")?.to_owned());
    }
    for id_value in heldout_ids {
        let id = id_value
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| "heldout IDs must be non-empty strings".to_owned())?;
        if !heldout_set.insert(id.to_owned()) {
            return Err(format!("duplicate heldout observation {id}"));
        }
        heldout_order.push(id.to_owned());
        if calibration_set.contains(id) {
            return Err(format!(
                "observation {id} appears in calibration and holdout"
            ));
        }
        let observation = observation_by_id
            .get(id)
            .ok_or_else(|| format!("unknown heldout observation {id}"))?;
        heldout_modalities.insert(required_text(observation, "modality")?.to_owned());
    }
    if heldout_modalities.len() != 1
        || heldout_modalities
            .iter()
            .any(|modality| calibration_modalities.contains(modality))
    {
        return Err(
            "heldout modality must be exactly one modality absent from calibration".to_owned(),
        );
    }
    if calibration_set.len() + heldout_set.len() != observation_by_id.len() {
        return Err(
            "calibration and heldout sets must cover every observation exactly once".to_owned(),
        );
    }

    let mut calibration_data_rows = Vec::with_capacity(calibration_set.len());
    let mut calibration_receipt_rows = Vec::with_capacity(calibration_set.len());
    for id in &calibration_order {
        let observation = observation_by_id[id];
        let value = finite_number(
            observation
                .get("value")
                .ok_or_else(|| format!("calibration observation {id} missing value"))?,
            &format!("observation {id} value"),
        )?;
        let uncertainty = finite_number(
            observation
                .get("standard_uncertainty")
                .ok_or_else(|| format!("calibration observation {id} missing uncertainty"))?,
            &format!("observation {id} uncertainty"),
        )?;
        if uncertainty <= 0.0 {
            return Err(format!(
                "observation {id} uncertainty must be positive for Gaussian fitting"
            ));
        }
        let h = observation_operator(observation)?;
        let weight = 1.0 / (uncertainty * uncertainty);
        calibration_data_rows.push(CalibrationRow {
            h,
            observed: value,
            uncertainty,
        });
        calibration_receipt_rows.push(json!({
            "observation_id": id,
            "modality": required_text(observation, "modality")?,
            "weight_inverse_variance": weight,
            "forward_operator_coefficients": h,
            "observed_value": value,
            "standard_uncertainty": uncertainty
        }));
    }

    let posterior = conjugate_posterior(prior_mean, prior_precision, &calibration_data_rows)?;
    let posterior_mean = posterior.mean;
    let posterior_covariance = posterior.covariance;
    let posterior_precision =
        calibration_data_rows
            .iter()
            .fold(prior_precision, |precision, row| {
                precision.add_information(row.h, 1.0 / (row.uncertainty * row.uncertainty))
            });

    let mut predictive_rows = Vec::with_capacity(heldout_set.len());
    let mut total_log_predictive_density = 0.0;
    let mut standardized_residuals = Vec::with_capacity(heldout_set.len());
    for id in &heldout_order {
        let observation = observation_by_id[id];
        let observed = finite_number(
            observation
                .get("value")
                .ok_or_else(|| format!("heldout observation {id} missing value"))?,
            &format!("heldout observation {id} value"),
        )?;
        let measurement_uncertainty = finite_number(
            observation
                .get("standard_uncertainty")
                .ok_or_else(|| format!("heldout observation {id} missing uncertainty"))?,
            &format!("heldout observation {id} uncertainty"),
        )?;
        if measurement_uncertainty <= 0.0 {
            return Err(format!(
                "heldout observation {id} uncertainty must be positive"
            ));
        }
        let h = observation_operator(observation)?;
        let predicted_mean = h[0] * posterior_mean[0] + h[1] * posterior_mean[1];
        let epistemic_variance = posterior_covariance.quadratic(h);
        let predictive_variance =
            measurement_uncertainty * measurement_uncertainty + epistemic_variance;
        if !predictive_variance.is_finite() || predictive_variance <= 0.0 {
            return Err(format!(
                "heldout observation {id} predictive variance is invalid"
            ));
        }
        let residual = observed - predicted_mean;
        let standardized_residual = residual / predictive_variance.sqrt();
        let log_predictive_density = -0.5
            * ((2.0 * std::f64::consts::PI * predictive_variance).ln()
                + residual * residual / predictive_variance);
        total_log_predictive_density += log_predictive_density;
        standardized_residuals.push(standardized_residual);
        predictive_rows.push(json!({
            "observation_id": id,
            "modality": required_text(observation, "modality")?,
            "observed_value": observed,
            "predicted_mean": predicted_mean,
            "predictive_standard_deviation": predictive_variance.sqrt(),
            "epistemic_variance": epistemic_variance,
            "measurement_variance": measurement_uncertainty * measurement_uncertainty,
            "standardized_residual": standardized_residual,
            "log_predictive_density": log_predictive_density,
            "used_in_posterior": false
        }));
    }

    let prior_covariance = prior_precision.inverse()?;
    let mut calibration_modalities: Vec<String> = calibration_modalities.into_iter().collect();
    calibration_modalities.sort();
    let mut heldout_modalities: Vec<String> = heldout_modalities.into_iter().collect();
    heldout_modalities.sort();
    let dataset_id = required_text(&fixture, "dataset_id")?;
    let model_version = required_text(model_target, "model_version")?;
    let prior_id = required_text(model_target, "prior_definition_id")?;
    let operator_version = required_text(model_target, "observation_operator_version")?;
    Ok(json!({
        "status": "pass",
        "contract": "SC-OBSERVATION-INVERSE-GAUSSIAN-FIT-0.1",
        "dataset_id": dataset_id,
        "schema_sha256": format!("{:x}", Sha256::digest(schema_bytes)),
        "fixture_sha256": format!("{:x}", Sha256::digest(fixture_bytes)),
        "model": {
            "family": "linear_gaussian_observation_operator",
            "physical_model_family": "lattice_ginzburg_landau",
            "model_version": model_version,
            "prior_definition_id": prior_id,
            "observation_operator_version": operator_version,
            "parameter_names": parameter_names,
            "calibration_only_posterior": true
        },
        "prior": {
            "mean": prior_mean,
            "precision": [
                [prior_precision.a, prior_precision.b],
                [prior_precision.b, prior_precision.d]
            ],
            "covariance": [
                [prior_covariance.a, prior_covariance.b],
                [prior_covariance.b, prior_covariance.d]
            ]
        },
        "calibration": {
            "observation_count": calibration_receipt_rows.len(),
            "modalities": calibration_modalities,
            "rows": calibration_receipt_rows
        },
        "posterior": {
            "mean": posterior_mean,
            "covariance": [
                [posterior_covariance.a, posterior_covariance.b],
                [posterior_covariance.b, posterior_covariance.d]
            ],
            "marginal_95_percent_intervals": [
                [
                    posterior_mean[0] - 1.959963984540054 * posterior_covariance.a.sqrt(),
                    posterior_mean[0] + 1.959963984540054 * posterior_covariance.a.sqrt()
                ],
                [
                    posterior_mean[1] - 1.959963984540054 * posterior_covariance.d.sqrt(),
                    posterior_mean[1] + 1.959963984540054 * posterior_covariance.d.sqrt()
                ]
            ],
            "posterior_precision": [
                [posterior_precision.a, posterior_precision.b],
                [posterior_precision.b, posterior_precision.d]
            ]
        },
        "heldout_predictive_evaluation": {
            "observation_count": predictive_rows.len(),
            "modalities": heldout_modalities,
            "rows": predictive_rows,
            "total_log_predictive_density": total_log_predictive_density,
            "mean_absolute_standardized_residual": standardized_residuals
                .iter()
                .map(|value| value.abs())
                .sum::<f64>()
                / standardized_residuals.len() as f64,
            "heldout_observations_used_in_posterior": false
        },
        "assumptions": [
            "Two-parameter linear Gaussian observation operators.",
            "Independent Gaussian observation errors conditional on model parameters.",
            "Standard uncertainties are treated as known and fixed.",
            "No shared calibration nuisance parameter or model-discrepancy term is included."
        ],
        "claim_ceiling": "Synthetic two-parameter conjugate inverse baseline only; no physical GL parameter, material property, Tc, or experimental conclusion is established."
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
        "Usage: {program} --schema <schema.json> --fixture <specimen-observations.json> [--out <fit-receipt.json>]"
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
            unknown => {
                eprintln!("unknown argument: {unknown}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let receipt = fit_dataset(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|error| {
        eprintln!("SC-OBSERVATION-INVERSE-FIT-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing inverse-fit receipt should not fail");
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
    use std::fs;

    fn inputs() -> (Vec<u8>, Value) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema =
            fs::read(root.join(
                "data/schemas/superconductivity_observation_inverse_instance_v0_1.schema.json",
            ))
            .unwrap();
        let fixture: Value = serde_json::from_slice(
            &fs::read(
                root.join("data/research/superconductivity_observation_inverse_fixture_v0_1.json"),
            )
            .unwrap(),
        )
        .unwrap();
        (schema, fixture)
    }

    fn fit_mutated(fixture: Value) -> Result<Value, String> {
        let (schema, _) = inputs();
        fit_dataset(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn calibration_only_posterior_predicts_heldout_vortex_modality() {
        let (schema, fixture) = inputs();
        let receipt = fit_dataset(&schema, &serde_json::to_vec(&fixture).unwrap()).unwrap();
        assert_eq!(receipt["calibration"]["observation_count"], 2);
        assert!((receipt["posterior"]["mean"][0].as_f64().unwrap() - 0.91981258).abs() < 1.0e-5);
        assert_eq!(
            receipt["heldout_predictive_evaluation"]["heldout_observations_used_in_posterior"],
            false
        );
        assert_eq!(
            receipt["heldout_predictive_evaluation"]["observation_count"],
            1
        );
        let prediction = &receipt["heldout_predictive_evaluation"]["rows"][0];
        assert_eq!(prediction["used_in_posterior"], false);
        assert!(
            prediction["log_predictive_density"]
                .as_f64()
                .unwrap()
                .is_finite()
        );
        assert!(
            prediction["predictive_standard_deviation"]
                .as_f64()
                .unwrap()
                > 0.0
        );
    }

    #[test]
    fn changing_heldout_value_does_not_change_calibration_posterior() {
        let (_, mut fixture) = inputs();
        let baseline = fit_mutated(fixture.clone()).unwrap();
        fixture["observations"][2]["value"] = json!(0.7);
        let changed = fit_mutated(fixture).unwrap();
        assert_eq!(baseline["posterior"], changed["posterior"]);
        assert_ne!(
            baseline["heldout_predictive_evaluation"]["rows"][0]["log_predictive_density"],
            changed["heldout_predictive_evaluation"]["rows"][0]["log_predictive_density"]
        );
    }

    #[test]
    fn changing_calibration_value_changes_posterior() {
        let (_, mut fixture) = inputs();
        let baseline = fit_mutated(fixture.clone()).unwrap();
        fixture["observations"][0]["value"] = json!(1.2);
        let changed = fit_mutated(fixture).unwrap();
        assert_ne!(baseline["posterior"]["mean"], changed["posterior"]["mean"]);
    }

    #[test]
    fn zero_uncertainty_is_rejected_by_gaussian_likelihood() {
        let (_, mut fixture) = inputs();
        fixture["observations"][0]["standard_uncertainty"] = json!(0.0);
        assert!(fit_mutated(fixture).is_err());
    }

    fn normal_sample(rng: &mut rand::rngs::StdRng) -> f64 {
        use rand::Rng;
        let u1 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        let u2 = rng.gen_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    #[test]
    fn prior_predictive_sbc_rank_histograms_and_interval_coverage() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let (_, fixture) = inputs();
        let model_target = &fixture["model_target"];
        let prior_mean =
            parse_vector2(&model_target["parameter_prior_mean"], "prior_mean").unwrap();
        let prior_precision = parse_precision(
            &model_target["parameter_prior_precision"],
            "prior_precision",
        )
        .unwrap();
        let prior_covariance = prior_precision.inverse().unwrap();
        let l00 = prior_covariance.a.sqrt();
        let l10 = prior_covariance.b / l00;
        let l11 = (prior_covariance.d - l10 * l10).sqrt();

        let calibration_ids: Vec<String> = fixture["split"]["calibration_observation_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let heldout_id = fixture["split"]["heldout_observation_ids"][0]
            .as_str()
            .unwrap()
            .to_owned();
        let observations = fixture["observations"].as_array().unwrap();
        let lookup = |id: &str| {
            observations
                .iter()
                .find(|observation| observation["observation_id"] == id)
                .unwrap()
        };
        let calibration_templates: Vec<&Value> =
            calibration_ids.iter().map(|id| lookup(id)).collect();
        let heldout_template = lookup(&heldout_id);
        let heldout_h = observation_operator(heldout_template).unwrap();
        let heldout_sigma = heldout_template["standard_uncertainty"].as_f64().unwrap();

        let mut rng = StdRng::seed_from_u64(0x5eed_1281);
        let trials = 512;
        let posterior_draw_count = 128;
        let rank_bin_count = 10;
        let mut rank_histograms = [vec![0_usize; 10], vec![0_usize; 10]];
        let mut parameter_coverage = [0_usize; 2];
        let mut heldout_coverage = 0_usize;

        for _ in 0..trials {
            let z0 = normal_sample(&mut rng);
            let z1 = normal_sample(&mut rng);
            let truth = [
                prior_mean[0] + l00 * z0,
                prior_mean[1] + l10 * z0 + l11 * z1,
            ];
            let mut calibration_rows = Vec::with_capacity(calibration_templates.len());
            for template in &calibration_templates {
                let h = observation_operator(template).unwrap();
                let sigma = template["standard_uncertainty"].as_f64().unwrap();
                let observed = h[0] * truth[0] + h[1] * truth[1] + sigma * normal_sample(&mut rng);
                calibration_rows.push(CalibrationRow {
                    h,
                    observed,
                    uncertainty: sigma,
                });
            }
            let posterior =
                conjugate_posterior(prior_mean, prior_precision, &calibration_rows).unwrap();

            let posterior_l00 = posterior.covariance.a.sqrt();
            let posterior_l10 = posterior.covariance.b / posterior_l00;
            let posterior_l11 = (posterior.covariance.d - posterior_l10 * posterior_l10).sqrt();
            let mut rank0 = 0_usize;
            let mut rank1 = 0_usize;
            for _ in 0..posterior_draw_count {
                let draw_z0 = normal_sample(&mut rng);
                let draw_z1 = normal_sample(&mut rng);
                let draw = [
                    posterior.mean[0] + posterior_l00 * draw_z0,
                    posterior.mean[1] + posterior_l10 * draw_z0 + posterior_l11 * draw_z1,
                ];
                if draw[0] < truth[0] {
                    rank0 += 1;
                }
                if draw[1] < truth[1] {
                    rank1 += 1;
                }
            }
            let bin0 = ((rank0 as f64 / (posterior_draw_count + 1) as f64) * rank_bin_count as f64)
                .floor() as usize;
            let bin1 = ((rank1 as f64 / (posterior_draw_count + 1) as f64) * rank_bin_count as f64)
                .floor() as usize;
            rank_histograms[0][bin0.min(rank_bin_count - 1)] += 1;
            rank_histograms[1][bin1.min(rank_bin_count - 1)] += 1;

            let lower0 = posterior.mean[0] - 1.959963984540054 * posterior.covariance.a.sqrt();
            let upper0 = posterior.mean[0] + 1.959963984540054 * posterior.covariance.a.sqrt();
            let lower1 = posterior.mean[1] - 1.959963984540054 * posterior.covariance.d.sqrt();
            let upper1 = posterior.mean[1] + 1.959963984540054 * posterior.covariance.d.sqrt();
            if (lower0..=upper0).contains(&truth[0]) {
                parameter_coverage[0] += 1;
            }
            if (lower1..=upper1).contains(&truth[1]) {
                parameter_coverage[1] += 1;
            }

            let predicted_mean =
                heldout_h[0] * posterior.mean[0] + heldout_h[1] * posterior.mean[1];
            let predictive_variance =
                heldout_sigma * heldout_sigma + posterior.covariance.quadratic(heldout_h);
            let heldout_observed = heldout_h[0] * truth[0]
                + heldout_h[1] * truth[1]
                + heldout_sigma * normal_sample(&mut rng);
            let half_width = 1.959963984540054 * predictive_variance.sqrt();
            if (predicted_mean - half_width..=predicted_mean + half_width)
                .contains(&heldout_observed)
            {
                heldout_coverage += 1;
            }
        }

        let coverage0 = parameter_coverage[0] as f64 / trials as f64;
        let coverage1 = parameter_coverage[1] as f64 / trials as f64;
        let predictive_coverage = heldout_coverage as f64 / trials as f64;
        for coverage in [coverage0, coverage1, predictive_coverage] {
            assert!(
                (0.89..=0.99).contains(&coverage),
                "95% prior-predictive interval coverage outside tolerance: {coverage}"
            );
        }

        for (parameter_index, histogram) in rank_histograms.iter().enumerate() {
            let expected = trials as f64 / rank_bin_count as f64;
            let chi_squared = histogram
                .iter()
                .map(|count| {
                    let difference = *count as f64 - expected;
                    difference * difference / expected
                })
                .sum::<f64>();
            assert!(
                chi_squared < 25.0,
                "SBC rank histogram {parameter_index} deviates from uniformity: chi2={chi_squared}, bins={histogram:?}"
            );
            assert!(histogram.iter().all(|count| (25..=80).contains(count)));
        }
    }

    #[test]
    fn asymmetric_prior_precision_is_rejected() {
        let (_, mut fixture) = inputs();
        fixture["model_target"]["parameter_prior_precision"] = json!([[1.0, 0.2], [0.1, 1.0]]);
        assert!(fit_mutated(fixture).is_err());
    }

    #[test]
    fn inverse_fit_receipt_is_deterministic() {
        let (schema, fixture) = inputs();
        let bytes = serde_json::to_vec(&fixture).unwrap();
        let first = fit_dataset(&schema, &bytes).unwrap();
        let second = fit_dataset(&schema, &bytes).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn inverse_fitter_rejects_retrospective_split_registration() {
        let (_, mut fixture) = inputs();
        fixture["split"]["pre_registered_at"] = json!("2026-11-18T09:30:00Z");
        assert!(fit_mutated(fixture).is_err());
    }
}
