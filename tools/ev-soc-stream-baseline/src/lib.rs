use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const METHOD_VERSION: &str = "ev-soc-stream-baseline/0.1";
pub const CORPUS_VERSION: &str = "ev-soc-stream-corpus/0.1";
const PERIOD_MS: u64 = 1_000;
const EPS: f64 = 1e-12;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub vehicle_ref: String,
    pub pack_ref: String,
    pub configuration_sha256: String,
    pub sensor_calibration_sha256: String,
    pub capacity_reference_sha256: String,
    pub current_bias_calibration_sha256: String,
    pub parameter_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: String,
    pub identity: Identity,
    pub initial_soc: f64,
    pub initial_soc_bound: f64,
    pub capacity_ah: f64,
    pub capacity_uncertainty_ah: f64,
    pub current_bias_a: f64,
    pub current_bias_uncertainty_a: f64,
    pub current_measurement_uncertainty_a: f64,
    pub charge_efficiency: f64,
    pub discharge_efficiency: f64,
    pub voltage_min_v: f64,
    pub voltage_max_v: f64,
    pub temperature_min_c: f64,
    pub temperature_max_c: f64,
}

impl Config {
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != "ev-soc-stream-config/0.1" {
            return Err("UNSUPPORTED_CONFIG_VERSION".into());
        }
        let values = [
            self.initial_soc,
            self.initial_soc_bound,
            self.capacity_ah,
            self.capacity_uncertainty_ah,
            self.current_bias_a,
            self.current_bias_uncertainty_a,
            self.current_measurement_uncertainty_a,
            self.charge_efficiency,
            self.discharge_efficiency,
            self.voltage_min_v,
            self.voltage_max_v,
            self.temperature_min_c,
            self.temperature_max_c,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err("NON_FINITE_CONFIG_VALUE".into());
        }
        if !(0.0..=1.0).contains(&self.initial_soc)
            || self.initial_soc_bound < 0.0
            || self.initial_soc_bound > 1.0
        {
            return Err("INVALID_INITIAL_SOC_OR_BOUND".into());
        }
        if self.capacity_ah <= 0.0
            || self.capacity_uncertainty_ah < 0.0
            || self.capacity_uncertainty_ah >= self.capacity_ah
        {
            return Err("INVALID_CAPACITY_OR_UNCERTAINTY".into());
        }
        if self.current_bias_uncertainty_a < 0.0 || self.current_measurement_uncertainty_a < 0.0 {
            return Err("INVALID_CURRENT_UNCERTAINTY".into());
        }
        if !(0.0..=1.0).contains(&self.charge_efficiency)
            || self.charge_efficiency == 0.0
            || !(0.0..=1.0).contains(&self.discharge_efficiency)
            || self.discharge_efficiency == 0.0
        {
            return Err("INVALID_COULOMBIC_EFFICIENCY".into());
        }
        if self.voltage_min_v <= 0.0
            || self.voltage_max_v <= self.voltage_min_v
            || self.temperature_max_c <= self.temperature_min_c
        {
            return Err("INVALID_OPERATING_DOMAIN".into());
        }
        if self.identity.vehicle_ref.trim().is_empty() || self.identity.pack_ref.trim().is_empty() {
            return Err("MISSING_ASSET_IDENTITY".into());
        }
        for digest in [
            &self.identity.configuration_sha256,
            &self.identity.sensor_calibration_sha256,
            &self.identity.capacity_reference_sha256,
            &self.identity.current_bias_calibration_sha256,
            &self.identity.parameter_sha256,
        ] {
            if !is_sha256(digest) {
                return Err("INVALID_EVIDENCE_DIGEST".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub timestamp_ms: u64,
    pub current_a: f64,
    pub voltage_v: f64,
    pub temperature_c: f64,
    pub current_valid: bool,
    pub voltage_valid: bool,
    pub temperature_valid: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Estimate {
    pub method_version: String,
    pub vehicle_ref: String,
    pub pack_ref: String,
    pub configuration_sha256: String,
    pub sensor_calibration_sha256: String,
    pub capacity_reference_sha256: String,
    pub current_bias_calibration_sha256: String,
    pub parameter_sha256: String,
    pub timestamp_ms: u64,
    pub sample_count: u64,
    pub soc: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub deterministic_error_bound: f64,
    pub interval_method: String,
    pub evidence_class: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Updated(Estimate),
    Abstained { reason: String, retained: Estimate },
}

pub struct Estimator {
    config: Config,
    state: Estimate,
}

impl Estimator {
    pub fn new(config: Config, initial_timestamp_ms: u64) -> Result<Self, String> {
        config.validate()?;
        let id = &config.identity;
        let lower = (config.initial_soc - config.initial_soc_bound).max(0.0);
        let upper = (config.initial_soc + config.initial_soc_bound).min(1.0);
        let state = Estimate {
            method_version: METHOD_VERSION.into(),
            vehicle_ref: id.vehicle_ref.clone(),
            pack_ref: id.pack_ref.clone(),
            configuration_sha256: id.configuration_sha256.clone(),
            sensor_calibration_sha256: id.sensor_calibration_sha256.clone(),
            capacity_reference_sha256: id.capacity_reference_sha256.clone(),
            current_bias_calibration_sha256: id.current_bias_calibration_sha256.clone(),
            parameter_sha256: id.parameter_sha256.clone(),
            timestamp_ms: initial_timestamp_ms,
            sample_count: 0,
            soc: config.initial_soc,
            lower_bound: lower,
            upper_bound: upper,
            deterministic_error_bound: (config.initial_soc - lower).max(upper - config.initial_soc),
            interval_method: "deterministic_interval_arithmetic_v0_1".into(),
            evidence_class: "derived_estimate".into(),
        };
        Ok(Self { config, state })
    }

    pub fn estimate(&self) -> &Estimate {
        &self.state
    }

    /// One causal 1 Hz update. An abstention never changes state or timestamp.
    pub fn update(&mut self, sample: &Sample) -> Outcome {
        let abstain = |reason: &str, state: &Estimate| Outcome::Abstained {
            reason: reason.into(),
            retained: state.clone(),
        };
        let Some(expected_time) = self.state.timestamp_ms.checked_add(PERIOD_MS) else {
            return abstain("TIMESTAMP_OVERFLOW", &self.state);
        };
        if sample.timestamp_ms != expected_time {
            return abstain("SAMPLE_INTERVAL_NOT_1HZ", &self.state);
        }
        if !sample.current_valid {
            return abstain("CURRENT_SENSOR_INVALID", &self.state);
        }
        if !sample.voltage_valid {
            return abstain("VOLTAGE_SENSOR_INVALID", &self.state);
        }
        if !sample.temperature_valid {
            return abstain("TEMPERATURE_SENSOR_INVALID", &self.state);
        }
        if !sample.current_a.is_finite()
            || !sample.voltage_v.is_finite()
            || !sample.temperature_c.is_finite()
        {
            return abstain("NON_FINITE_SENSOR_SAMPLE", &self.state);
        }
        if sample.voltage_v < self.config.voltage_min_v
            || sample.voltage_v > self.config.voltage_max_v
        {
            return abstain("VOLTAGE_OUT_OF_DOMAIN", &self.state);
        }
        if sample.temperature_c < self.config.temperature_min_c
            || sample.temperature_c > self.config.temperature_max_c
        {
            return abstain("TEMPERATURE_OUT_OF_DOMAIN", &self.state);
        }

        let corrected_current = sample.current_a - self.config.current_bias_a;
        let current_radius =
            self.config.current_measurement_uncertainty_a + self.config.current_bias_uncertainty_a;
        let current_bounds = [
            corrected_current - current_radius,
            corrected_current + current_radius,
        ];
        let capacity_bounds = [
            self.config.capacity_ah - self.config.capacity_uncertainty_ah,
            self.config.capacity_ah + self.config.capacity_uncertainty_ah,
        ];
        let mut delta_min = f64::INFINITY;
        let mut delta_max = f64::NEG_INFINITY;
        for current in current_bounds {
            for capacity in capacity_bounds {
                let delta = soc_delta(
                    current,
                    capacity,
                    self.config.charge_efficiency,
                    self.config.discharge_efficiency,
                );
                delta_min = delta_min.min(delta);
                delta_max = delta_max.max(delta);
            }
        }
        let nominal_delta = soc_delta(
            corrected_current,
            self.config.capacity_ah,
            self.config.charge_efficiency,
            self.config.discharge_efficiency,
        );
        let next_soc = self.state.soc + nominal_delta;
        if !delta_min.is_finite()
            || !delta_max.is_finite()
            || !next_soc.is_finite()
            || !(0.0..=1.0).contains(&next_soc)
        {
            return abstain("SOC_BOUNDARY_OR_NUMERIC_FAILURE", &self.state);
        }
        let lower = (self.state.lower_bound + delta_min).max(0.0);
        let upper = (self.state.upper_bound + delta_max).min(1.0);
        if !lower.is_finite()
            || !upper.is_finite()
            || lower > upper + EPS
            || next_soc < lower - EPS
            || next_soc > upper + EPS
        {
            return abstain("INVALID_PROPAGATED_INTERVAL", &self.state);
        }
        let next = Estimate {
            timestamp_ms: sample.timestamp_ms,
            sample_count: self.state.sample_count + 1,
            soc: next_soc,
            lower_bound: lower,
            upper_bound: upper,
            deterministic_error_bound: (next_soc - lower).max(upper - next_soc).max(0.0),
            ..self.state.clone()
        };
        self.state = next.clone();
        Outcome::Updated(next)
    }
}

fn soc_delta(current_a: f64, capacity_ah: f64, charge_eff: f64, discharge_eff: f64) -> f64 {
    let effective_current = if current_a >= 0.0 {
        current_a * charge_eff
    } else {
        current_a / discharge_eff
    };
    effective_current / (3_600.0 * capacity_ah)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub corpus_version: String,
    pub cases: Vec<Case>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub case_id: String,
    pub config: Config,
    pub initial_timestamp_ms: u64,
    pub truth_initial_soc: f64,
    pub truth_capacity_ah: f64,
    pub samples: Vec<FixtureStep>,
    pub expected_disposition: String,
    pub expected_final_soc: Option<f64>,
    pub expected_tolerance: Option<f64>,
    pub expected_abstention_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureStep {
    pub sample: Sample,
    pub truth_current_a: f64,
    pub truth_soc: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseResult {
    pub case_id: String,
    pub disposition: String,
    pub abstention_reason: Option<String>,
    pub estimate: Estimate,
    pub truth_covered: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditReport {
    pub corpus_version: String,
    pub corpus_sha256: String,
    pub case_count: usize,
    pub estimated_count: usize,
    pub abstained_count: usize,
    pub cases: Vec<CaseResult>,
}

pub fn audit_corpus(bytes: &[u8]) -> Result<AuditReport, String> {
    let mut corpus: Corpus =
        serde_json::from_slice(bytes).map_err(|error| format!("INVALID_CORPUS:{error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.cases.is_empty() {
        return Err("UNSUPPORTED_OR_EMPTY_CORPUS".into());
    }
    corpus.cases.sort_by(|a, b| a.case_id.cmp(&b.case_id));
    let mut ids = BTreeSet::new();
    if corpus
        .cases
        .iter()
        .any(|case| case.case_id.trim().is_empty() || !ids.insert(case.case_id.clone()))
    {
        return Err("EMPTY_OR_DUPLICATE_CASE_ID".into());
    }
    let canonical = serde_json::to_vec(&corpus).map_err(|error| error.to_string())?;
    let corpus_sha256 = sha256(&canonical);
    let mut cases = Vec::new();
    let mut estimated_count = 0;
    let mut abstained_count = 0;
    for case in &corpus.cases {
        let result = audit_case(case)?;
        if result.disposition == "estimated" {
            estimated_count += 1;
        } else {
            abstained_count += 1;
        }
        cases.push(result);
    }
    Ok(AuditReport {
        corpus_version: corpus.corpus_version,
        corpus_sha256,
        case_count: cases.len(),
        estimated_count,
        abstained_count,
        cases,
    })
}

fn audit_case(case: &Case) -> Result<CaseResult, String> {
    if case.samples.is_empty()
        || !case.truth_initial_soc.is_finite()
        || !case.truth_capacity_ah.is_finite()
        || case.truth_capacity_ah <= 0.0
    {
        return Err(format!("INVALID_FIXTURE_TRUTH:{}", case.case_id));
    }
    let mut estimator = Estimator::new(case.config.clone(), case.initial_timestamp_ms)?;
    let initial = estimator.estimate();
    if case.truth_initial_soc < initial.lower_bound - EPS
        || case.truth_initial_soc > initial.upper_bound + EPS
        || (case.truth_capacity_ah - case.config.capacity_ah).abs()
            > case.config.capacity_uncertainty_ah + EPS
    {
        return Err(format!("INITIAL_TRUTH_OUTSIDE_BOUNDS:{}", case.case_id));
    }
    let mut truth_soc = case.truth_initial_soc;
    let mut reason = None;
    let mut truth_covered = true;
    for step in &case.samples {
        if step.sample.current_valid {
            let residual =
                (step.sample.current_a - case.config.current_bias_a - step.truth_current_a).abs();
            let allowance = case.config.current_measurement_uncertainty_a
                + case.config.current_bias_uncertainty_a;
            if residual > allowance + EPS {
                return Err(format!("FIXTURE_CURRENT_OUTSIDE_BOUNDS:{}", case.case_id));
            }
        }
        let before = estimator.estimate().clone();
        match estimator.update(&step.sample) {
            Outcome::Updated(estimate) => {
                truth_soc += soc_delta(
                    step.truth_current_a,
                    case.truth_capacity_ah,
                    case.config.charge_efficiency,
                    case.config.discharge_efficiency,
                );
                let expected_truth = step
                    .truth_soc
                    .ok_or_else(|| format!("MISSING_STEP_TRUTH:{}", case.case_id))?;
                if (truth_soc - expected_truth).abs() > 1e-10
                    || truth_soc < estimate.lower_bound - EPS
                    || truth_soc > estimate.upper_bound + EPS
                {
                    truth_covered = false;
                    return Err(format!("TRUTH_OUTSIDE_ESTIMATED_INTERVAL:{}", case.case_id));
                }
            }
            Outcome::Abstained {
                reason: why,
                retained,
            } => {
                if retained != before {
                    return Err(format!("ABSTENTION_MUTATED_STATE:{}", case.case_id));
                }
                reason = Some(why);
                break;
            }
        }
    }
    let disposition = if reason.is_some() {
        "abstain"
    } else {
        "estimated"
    };
    if case.expected_disposition != disposition {
        return Err(format!("DISPOSITION_MISMATCH:{}", case.case_id));
    }
    if disposition == "estimated" {
        let expected = case
            .expected_final_soc
            .ok_or_else(|| format!("MISSING_EXPECTED_FINAL_SOC:{}", case.case_id))?;
        let tolerance = case.expected_tolerance.unwrap_or(EPS);
        if (estimator.estimate().soc - expected).abs() > tolerance
            || case.expected_abstention_reason.is_some()
        {
            return Err(format!(
                "FINAL_SOC_OR_EXPECTATION_MISMATCH:{}",
                case.case_id
            ));
        }
    } else if reason.as_deref() != case.expected_abstention_reason.as_deref()
        || case.expected_final_soc.is_some()
    {
        return Err(format!("ABSTENTION_REASON_MISMATCH:{}", case.case_id));
    }
    Ok(CaseResult {
        case_id: case.case_id.clone(),
        disposition: disposition.into(),
        abstention_reason: reason,
        estimate: estimator.estimate().clone(),
        truth_covered,
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/soc_stream_scenarios_v0_1.json");

    fn report() -> AuditReport {
        audit_corpus(CORPUS).expect("frozen known-truth SOC corpus")
    }

    #[test]
    fn known_truth_corpus_has_three_estimates_and_three_abstentions() {
        let result = report();
        assert_eq!(result.case_count, 6);
        assert_eq!(result.estimated_count, 3);
        assert_eq!(result.abstained_count, 3);
        assert_eq!(result.corpus_sha256.len(), 64);
        assert!(result.cases.iter().all(|case| case.truth_covered));
    }

    #[test]
    fn current_sign_convention_is_charge_positive_and_discharge_negative() {
        let result = report();
        let charge = result
            .cases
            .iter()
            .find(|case| case.case_id == "positive-current-charge")
            .unwrap();
        let discharge = result
            .cases
            .iter()
            .find(|case| case.case_id == "negative-current-discharge")
            .unwrap();
        assert!(charge.estimate.soc > 0.5);
        assert!(discharge.estimate.soc < 0.6);
    }

    #[test]
    fn invalid_sensor_temperature_and_timestamp_gap_abstain_without_state_mutation() {
        let result = report();
        for id in [
            "invalid-current-sensor-abstains",
            "temperature-out-of-domain-abstains",
            "timestamp-gap-abstains",
        ] {
            let case = result.cases.iter().find(|item| item.case_id == id).unwrap();
            assert_eq!(case.disposition, "abstain");
            assert_eq!(case.estimate.sample_count, 0);
            assert_eq!(case.estimate.timestamp_ms, 0);
        }
    }

    #[test]
    fn independent_bias_calibration_is_bound_to_each_estimate() {
        let result = report();
        let case = result
            .cases
            .iter()
            .find(|item| item.case_id == "independent-current-bias-calibration")
            .unwrap();
        assert!((case.estimate.soc - 0.7).abs() < EPS);
        assert_eq!(case.estimate.current_bias_calibration_sha256.len(), 64);
    }

    #[test]
    fn corpus_case_order_is_canonical() {
        let original = report();
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).unwrap();
        corpus.cases.reverse();
        let bytes = serde_json::to_vec(&corpus).unwrap();
        let replay = audit_corpus(&bytes).unwrap();
        assert_eq!(original.corpus_sha256, replay.corpus_sha256);
        assert_eq!(
            serde_json::to_value(original.cases).unwrap(),
            serde_json::to_value(replay.cases).unwrap()
        );
    }

    #[test]
    fn malformed_evidence_digest_is_rejected() {
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).unwrap();
        corpus.cases[0].config.identity.capacity_reference_sha256 = "bad-digest".into();
        let bytes = serde_json::to_vec(&corpus).unwrap();
        assert!(audit_corpus(&bytes).is_err());
    }
}
