use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    assess_allostasis, assess_allostasis_with_drive, ExecutionLimits, ExecutionTrace,
    ExpectedRelation, ExperimentPreregistration, NativeInteroceptiveModel, OutcomeRef,
    ProtocolForecastSpec, RegisteredMeasure, INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
};

pub const BLINDED_METRIC_REPORT_SCHEMA_VERSION: u16 = 1;
pub const HYPOTHESIS_EVALUATION_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindedMetricValue {
    pub blind_code: String,
    pub metric_id: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindedMetricReport {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub protocol_sha256: String,
    pub execution_trace_sha256: String,
    pub values: Vec<BlindedMetricValue>,
}

impl BlindedMetricReport {
    pub fn validation_errors_against(
        &self,
        protocol: &ExperimentPreregistration,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != BLINDED_METRIC_REPORT_SCHEMA_VERSION {
            errors.push(format!(
                "blinded metric report schema version mismatch: {}",
                self.schema_version
            ));
        }
        if self.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
            errors.push(format!(
                "blinded metric model semantics version mismatch: {}",
                self.model_semantics_version
            ));
        }
        if !is_lower_hex(&self.execution_trace_sha256, 64) {
            errors.push("execution_trace_sha256 must be a lowercase SHA-256 digest".into());
        }
        match protocol.sha256() {
            Ok(expected) if expected == self.protocol_sha256 => {}
            Ok(_) => errors.push("blinded metric report protocol digest mismatch".into()),
            Err(protocol_errors) => errors.extend(
                protocol_errors
                    .into_iter()
                    .map(|error| format!("protocol: {error}")),
            ),
        }

        let known_blind_codes: BTreeSet<&str> =
            protocol.arms.iter().map(|arm| arm.blind_code.as_str()).collect();
        let known_metrics: BTreeSet<&str> = protocol
            .metrics
            .iter()
            .map(|metric| metric.metric_id.as_str())
            .collect();
        let mut seen = BTreeSet::new();
        for value in &self.values {
            if !value.value.is_finite() {
                errors.push(format!(
                    "metric {} for {} is non-finite",
                    value.metric_id, value.blind_code
                ));
            }
            if !known_blind_codes.contains(value.blind_code.as_str()) {
                errors.push(format!("unknown blind_code in metric report: {}", value.blind_code));
            }
            if !known_metrics.contains(value.metric_id.as_str()) {
                errors.push(format!("unknown metric_id in metric report: {}", value.metric_id));
            }
            if !seen.insert((value.blind_code.as_str(), value.metric_id.as_str())) {
                errors.push(format!(
                    "duplicate blinded metric pair: {}/{}",
                    value.blind_code, value.metric_id
                ));
            }
        }

        for arm in &protocol.arms {
            for metric in &protocol.metrics {
                if !seen.contains(&(arm.blind_code.as_str(), metric.metric_id.as_str())) {
                    errors.push(format!(
                        "missing blinded metric pair: {}/{}",
                        arm.blind_code, metric.metric_id
                    ));
                }
            }
        }

        errors
    }

    pub fn validate_against(
        &self,
        protocol: &ExperimentPreregistration,
    ) -> Result<(), Vec<String>> {
        let errors = self.validation_errors_against(protocol);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|error| error.to_string())
    }

    pub fn sha256(&self) -> Result<String, String> {
        hash_json(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisOutcome {
    pub hypothesis_id: String,
    pub primary: bool,
    pub left: OutcomeRef,
    pub left_value: f64,
    pub relation: ExpectedRelation,
    pub right: OutcomeRef,
    pub right_value: f64,
    pub satisfied: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisEvaluationReport {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub protocol_sha256: String,
    pub blinded_metric_sha256: String,
    pub outcomes: Vec<HypothesisOutcome>,
}

impl HypothesisEvaluationReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|error| error.to_string())
    }

    pub fn sha256(&self) -> Result<String, String> {
        hash_json(self)
    }
}

pub fn extract_blinded_metrics(
    trace: &ExecutionTrace,
    protocol: &ExperimentPreregistration,
    limits: ExecutionLimits,
) -> Result<BlindedMetricReport, Vec<String>> {
    trace.validate_against(protocol, limits)?;

    let mut values = Vec::with_capacity(protocol.arms.len() * protocol.metrics.len());
    for arm in &protocol.arms {
        let trace_arm = trace
            .arms
            .iter()
            .find(|trace_arm| trace_arm.blind_code == arm.blind_code)
            .ok_or_else(|| vec![format!("trace missing blind arm {}", arm.blind_code)])?;
        let terminal = trace_arm
            .steps
            .last()
            .ok_or_else(|| vec![format!("trace arm {} has no executed steps", arm.blind_code)])?;

        for metric in &protocol.metrics {
            let value = measure_value(metric.measure, terminal, arm.dynamics_config)?;
            if !value.is_finite() {
                return Err(vec![format!(
                    "metric {} for {} evaluated to a non-finite value",
                    metric.metric_id, arm.blind_code
                )]);
            }
            values.push(BlindedMetricValue {
                blind_code: arm.blind_code.clone(),
                metric_id: metric.metric_id.clone(),
                value,
            });
        }
    }
    values.sort_by(|left, right| {
        left.blind_code
            .cmp(&right.blind_code)
            .then_with(|| left.metric_id.cmp(&right.metric_id))
    });

    let report = BlindedMetricReport {
        schema_version: BLINDED_METRIC_REPORT_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        protocol_sha256: trace.protocol_sha256.clone(),
        execution_trace_sha256: trace
            .sha256()
            .map_err(|error| vec![format!("failed to hash execution trace: {error}")])?,
        values,
    };
    report.validate_against(protocol)?;
    Ok(report)
}

pub fn evaluate_hypotheses(
    protocol: &ExperimentPreregistration,
    blinded: &BlindedMetricReport,
) -> Result<HypothesisEvaluationReport, Vec<String>> {
    protocol.validate()?;
    blinded.validate_against(protocol)?;

    let blinded_metric_sha256 = blinded
        .sha256()
        .map_err(|error| vec![format!("failed to hash blinded metric report: {error}")])?;
    let protocol_sha256 = protocol.sha256()?;
    let mut outcomes = Vec::with_capacity(protocol.hypotheses.len());

    for hypothesis in &protocol.hypotheses {
        let left_value = lookup_outcome(protocol, blinded, &hypothesis.left)?;
        let right_value = lookup_outcome(protocol, blinded, &hypothesis.right)?;
        outcomes.push(HypothesisOutcome {
            hypothesis_id: hypothesis.hypothesis_id.clone(),
            primary: hypothesis.primary,
            left: hypothesis.left.clone(),
            left_value,
            relation: hypothesis.relation,
            right: hypothesis.right.clone(),
            right_value,
            satisfied: hypothesis.relation.is_satisfied_by(left_value, right_value),
        });
    }

    Ok(HypothesisEvaluationReport {
        schema_version: HYPOTHESIS_EVALUATION_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        protocol_sha256,
        blinded_metric_sha256,
        outcomes,
    })
}

fn lookup_outcome(
    protocol: &ExperimentPreregistration,
    blinded: &BlindedMetricReport,
    outcome: &OutcomeRef,
) -> Result<f64, Vec<String>> {
    let arm = protocol
        .arms
        .iter()
        .find(|arm| arm.arm_id == outcome.arm_id)
        .ok_or_else(|| vec![format!("unknown arm during unblinding: {}", outcome.arm_id)])?;
    blinded
        .values
        .iter()
        .find(|value| value.blind_code == arm.blind_code && value.metric_id == outcome.metric_id)
        .map(|value| value.value)
        .ok_or_else(|| {
            vec![format!(
                "missing metric {} for blinded arm {}",
                outcome.metric_id, arm.blind_code
            )]
        })
}

fn measure_value(
    measure: RegisteredMeasure,
    terminal: &crate::ExecutionStepTrace,
    dynamics_config: crate::InteroceptiveDynamicsConfig,
) -> Result<f64, Vec<String>> {
    let value = match measure {
        RegisteredMeasure::TerminalHomeostaticWeightedDeviation => {
            f64::from(terminal.homeostasis.weighted_deviation)
        }
        RegisteredMeasure::TerminalHomeostaticPeakDeviation => {
            f64::from(terminal.homeostasis.peak_deviation)
        }
        RegisteredMeasure::TerminalChannelDeviation { channel } => {
            f64::from(terminal.homeostasis.channel_deviations[channel.index()])
        }
        RegisteredMeasure::TerminalForecastDiscountedDebt { forecast } => {
            f64::from(forecast_report(forecast, terminal, dynamics_config)?.discounted_debt)
        }
        RegisteredMeasure::TerminalForecastTerminalDeviation { forecast } => {
            f64::from(forecast_report(forecast, terminal, dynamics_config)?.terminal_deviation)
        }
        RegisteredMeasure::TerminalForecastBreachExposures { forecast } => {
            f64::from(forecast_report(forecast, terminal, dynamics_config)?.breach_exposures)
        }
        RegisteredMeasure::TerminalForecastUniqueBreachedChannels { forecast } => f64::from(
            forecast_report(forecast, terminal, dynamics_config)?.unique_breached_channels,
        ),
    };
    Ok(value)
}

fn forecast_report(
    forecast: ProtocolForecastSpec,
    terminal: &crate::ExecutionStepTrace,
    dynamics_config: crate::InteroceptiveDynamicsConfig,
) -> Result<crate::AllostaticReport, Vec<String>> {
    match forecast {
        ProtocolForecastSpec::Kinematic { config } => {
            Ok(assess_allostasis(&terminal.state, config))
        }
        ProtocolForecastSpec::DynamicsAwareConstantDrive { config, drive } => {
            if (config.dt - dynamics_config.step_dt).abs() > f32::EPSILON {
                return Err(vec![
                    "dynamics-aware metric dt does not match terminal arm step_dt".into(),
                ]);
            }
            let model = NativeInteroceptiveModel::new(terminal.state.clone(), dynamics_config);
            Ok(assess_allostasis_with_drive(&model, drive, config))
        }
    }
}

fn hash_json<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let digest = Sha256::digest(&bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(encoded)
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
