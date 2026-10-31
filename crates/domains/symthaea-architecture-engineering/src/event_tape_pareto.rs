//! Evidence-bound adapter from one strict EventTape to candidate-level Pareto telemetry.
//!
//! Candidate evaluation retries and failed attempts remain charged to the candidate whose
//! content hash they evaluate. Direction-proposal costs are reported separately as campaign
//! overhead rather than silently assigned to every candidate or omitted from campaign totals.
//! Unknown meters disqualify the tape through EventTape::validate() before aggregation.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::event_tape::{
    ActionIntent, ActionOutcome, CandidateDisposition, EventTape, MeterValue, ResourceReceipt,
};
use super::pareto_objectives::{
    analyze_pareto, ParetoAnalysisPlan, ParetoAnalysisSummary, ParetoCandidate,
    ParetoError, ParetoObjectiveSpec,
};

const SCHEMA: &str = "symthaea.architecture-engineering.event-tape-pareto";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventTapeParetoError {
    pub path: String,
    pub message: String,
}

impl EventTapeParetoError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for EventTapeParetoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for EventTapeParetoError {}

/// Source metric extracted from a candidate's complete set of evaluation attempts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CandidateObjectiveMetric {
    MeanVerifiedScore,
    MinimumVerifiedScore,
    MeanReportedUncertainty,
    MaximumReportedUncertainty,
    /// Mean minus z times the sample standard-error proxy. Requires at least two verified scores.
    /// This is an optimizer-facing conservative statistic, not a calibrated confidence interval.
    LowerConfidenceBoundVerifiedScore { z_value: f64 },
    /// Verified evaluation attempts divided by all evaluation attempts, including retries.
    VerifiedAttemptRate,
    EvaluationAttemptCount,
    TotalHostCpuMillis,
    SumActionWallMillis,
    TotalAcceleratorMillis,
    MaxActionPeakMemoryBytes,
    MaxActionAcceleratorPeakMemoryBytes,
    TotalEnergyMillijoules,
    TotalInputTokens,
    TotalOutputTokens,
    TotalToolCalls,
    TotalEvaluatorCalls,
    TotalVerifierCalls,
    TotalChargedBudgetMicrounits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateObjectiveMapping {
    pub objective_id: String,
    pub metric: CandidateObjectiveMetric,
}

/// Frozen adapter configuration. Objective order must match objective_mappings order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventTapeParetoConfig {
    pub objective_specs: Vec<ParetoObjectiveSpec>,
    pub objective_mappings: Vec<CandidateObjectiveMapping>,
    pub reference_point_utility: Vec<f64>,
    /// Used only for candidates with no verified score; such candidates are marked infeasible.
    pub unverified_score_value: f64,
    /// Minimum number of Verified evaluations for a candidate to be feasibility-qualified.
    pub minimum_verified_evaluations: usize,
}

impl EventTapeParetoConfig {
    pub fn validate(&self) -> Result<(), EventTapeParetoError> {
        if !(2..=4).contains(&self.objective_specs.len()) {
            return Err(EventTapeParetoError::new(
                "config.objective_specs",
                "EventTape Pareto adapter v0.1 requires two through four objectives",
            ));
        }
        if self.objective_mappings.len() != self.objective_specs.len() {
            return Err(EventTapeParetoError::new(
                "config.objective_mappings",
                "exactly one metric mapping is required for every objective",
            ));
        }
        if self.minimum_verified_evaluations == 0 {
            return Err(EventTapeParetoError::new(
                "config.minimum_verified_evaluations",
                "minimum verified evaluation count must be positive",
            ));
        }
        if !self.unverified_score_value.is_finite() {
            return Err(EventTapeParetoError::new(
                "config.unverified_score_value",
                "unverified score fallback must be finite",
            ));
        }
        if self.reference_point_utility.len() != self.objective_specs.len()
            || self.reference_point_utility.iter().any(|value| !value.is_finite())
        {
            return Err(EventTapeParetoError::new(
                "config.reference_point_utility",
                "reference point must have one finite coordinate per objective",
            ));
        }

        let mut ids = BTreeSet::new();
        for (index, (spec, mapping)) in self.objective_specs.iter()
            .zip(&self.objective_mappings).enumerate()
        {
            if mapping.objective_id != spec.objective_id {
                return Err(EventTapeParetoError::new(
                    format!("config.objective_mappings[{index}].objective_id"),
                    "metric mapping objective ID/order differs from objective_specs",
                ));
            }
            if !ids.insert(spec.objective_id.as_str()) {
                return Err(EventTapeParetoError::new(
                    format!("config.objective_specs[{index}].objective_id"),
                    "objective IDs must be unique",
                ));
            }
            let expected_direction = expected_direction(&mapping.metric);
            if spec.direction != expected_direction {
                return Err(EventTapeParetoError::new(
                    format!("config.objective_specs[{index}].direction"),
                    "objective direction conflicts with the mapped source metric",
                ));
            }
            if let CandidateObjectiveMetric::LowerConfidenceBoundVerifiedScore { z_value } = &mapping.metric {
                if !z_value.is_finite() || *z_value < 0.0 {
                    return Err(EventTapeParetoError::new(
                        format!("config.objective_mappings[{index}].metric.z_value"),
                        "z value must be finite and non-negative",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn config_sha256(&self) -> Result<String, EventTapeParetoError> {
        self.validate()?;
        digest_serializable(self)
    }
}

fn expected_direction(metric: &CandidateObjectiveMetric) -> super::pareto_objectives::ObjectiveDirection {
    use super::pareto_objectives::ObjectiveDirection::{Maximize, Minimize};
    match metric {
        CandidateObjectiveMetric::MeanVerifiedScore
        | CandidateObjectiveMetric::MinimumVerifiedScore
        | CandidateObjectiveMetric::LowerConfidenceBoundVerifiedScore { .. }
        | CandidateObjectiveMetric::VerifiedAttemptRate => Maximize,
        CandidateObjectiveMetric::MeanReportedUncertainty
        | CandidateObjectiveMetric::MaximumReportedUncertainty => Minimize,
        CandidateObjectiveMetric::EvaluationAttemptCount
        | CandidateObjectiveMetric::TotalHostCpuMillis
        | CandidateObjectiveMetric::SumActionWallMillis
        | CandidateObjectiveMetric::TotalAcceleratorMillis
        | CandidateObjectiveMetric::MaxActionPeakMemoryBytes
        | CandidateObjectiveMetric::MaxActionAcceleratorPeakMemoryBytes
        | CandidateObjectiveMetric::TotalEnergyMillijoules
        | CandidateObjectiveMetric::TotalInputTokens
        | CandidateObjectiveMetric::TotalOutputTokens
        | CandidateObjectiveMetric::TotalToolCalls
        | CandidateObjectiveMetric::TotalEvaluatorCalls
        | CandidateObjectiveMetric::TotalVerifierCalls
        | CandidateObjectiveMetric::TotalChargedBudgetMicrounits => Minimize,
    }
}

/// Additive totals plus per-action memory high-water marks.
/// Wall time is summed action duration (worker-time), not campaign elapsed latency.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionResourceTotals {
    pub action_count: u64,
    pub host_cpu_millis: u64,
    pub sum_action_wall_millis: u64,
    pub accelerator_millis: u64,
    pub max_action_peak_memory_bytes: u64,
    pub max_action_accelerator_peak_memory_bytes: u64,
    pub energy_millijoules: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_calls: u64,
    pub evaluator_calls: u64,
    pub verifier_calls: u64,
    pub charged_budget_microunits: u64,
}

impl ActionResourceTotals {
    fn add_receipt(
        &mut self,
        receipt: &ResourceReceipt,
        path: &str,
    ) -> Result<(), EventTapeParetoError> {
        self.action_count = checked_add(self.action_count, 1, &format!("{path}.action_count"))?;
        self.host_cpu_millis = add_meter(self.host_cpu_millis, &receipt.host_cpu_millis, &format!("{path}.host_cpu_millis"))?;
        self.sum_action_wall_millis = add_meter(self.sum_action_wall_millis, &receipt.wall_millis, &format!("{path}.wall_millis"))?;
        self.accelerator_millis = add_meter(self.accelerator_millis, &receipt.accelerator_millis, &format!("{path}.accelerator_millis"))?;
        self.max_action_peak_memory_bytes = self.max_action_peak_memory_bytes.max(read_meter(&receipt.peak_memory_bytes, &format!("{path}.peak_memory_bytes"))?);
        self.max_action_accelerator_peak_memory_bytes = self.max_action_accelerator_peak_memory_bytes.max(read_meter(&receipt.accelerator_peak_memory_bytes, &format!("{path}.accelerator_peak_memory_bytes"))?);
        self.energy_millijoules = add_meter(self.energy_millijoules, &receipt.energy_millijoules, &format!("{path}.energy_millijoules"))?;
        self.input_tokens = add_meter(self.input_tokens, &receipt.input_tokens, &format!("{path}.input_tokens"))?;
        self.output_tokens = add_meter(self.output_tokens, &receipt.output_tokens, &format!("{path}.output_tokens"))?;
        self.tool_calls = add_meter(self.tool_calls, &receipt.tool_calls, &format!("{path}.tool_calls"))?;
        self.evaluator_calls = add_meter(self.evaluator_calls, &receipt.evaluator_calls, &format!("{path}.evaluator_calls"))?;
        self.verifier_calls = add_meter(self.verifier_calls, &receipt.verifier_calls, &format!("{path}.verifier_calls"))?;
        self.charged_budget_microunits = add_meter(self.charged_budget_microunits, &receipt.charged_budget_microunits, &format!("{path}.charged_budget_microunits"))?;
        Ok(())
    }


    fn checked_add_totals(
        &mut self,
        other: &Self,
        path: &str,
    ) -> Result<(), EventTapeParetoError> {
        self.action_count = checked_add(self.action_count, other.action_count, &format!("{path}.action_count"))?;
        self.host_cpu_millis = checked_add(self.host_cpu_millis, other.host_cpu_millis, &format!("{path}.host_cpu_millis"))?;
        self.sum_action_wall_millis = checked_add(self.sum_action_wall_millis, other.sum_action_wall_millis, &format!("{path}.sum_action_wall_millis"))?;
        self.accelerator_millis = checked_add(self.accelerator_millis, other.accelerator_millis, &format!("{path}.accelerator_millis"))?;
        self.max_action_peak_memory_bytes = self.max_action_peak_memory_bytes.max(other.max_action_peak_memory_bytes);
        self.max_action_accelerator_peak_memory_bytes = self.max_action_accelerator_peak_memory_bytes.max(other.max_action_accelerator_peak_memory_bytes);
        self.energy_millijoules = checked_add(self.energy_millijoules, other.energy_millijoules, &format!("{path}.energy_millijoules"))?;
        self.input_tokens = checked_add(self.input_tokens, other.input_tokens, &format!("{path}.input_tokens"))?;
        self.output_tokens = checked_add(self.output_tokens, other.output_tokens, &format!("{path}.output_tokens"))?;
        self.tool_calls = checked_add(self.tool_calls, other.tool_calls, &format!("{path}.tool_calls"))?;
        self.evaluator_calls = checked_add(self.evaluator_calls, other.evaluator_calls, &format!("{path}.evaluator_calls"))?;
        self.verifier_calls = checked_add(self.verifier_calls, other.verifier_calls, &format!("{path}.verifier_calls"))?;
        self.charged_budget_microunits = checked_add(self.charged_budget_microunits, other.charged_budget_microunits, &format!("{path}.charged_budget_microunits"))?;
        Ok(())
    }

    fn value(&self, metric: &CandidateObjectiveMetric) -> f64 {
        match metric {
            CandidateObjectiveMetric::EvaluationAttemptCount => self.action_count as f64,
            CandidateObjectiveMetric::TotalHostCpuMillis => self.host_cpu_millis as f64,
            CandidateObjectiveMetric::SumActionWallMillis => self.sum_action_wall_millis as f64,
            CandidateObjectiveMetric::TotalAcceleratorMillis => self.accelerator_millis as f64,
            CandidateObjectiveMetric::MaxActionPeakMemoryBytes => self.max_action_peak_memory_bytes as f64,
            CandidateObjectiveMetric::MaxActionAcceleratorPeakMemoryBytes => self.max_action_accelerator_peak_memory_bytes as f64,
            CandidateObjectiveMetric::TotalEnergyMillijoules => self.energy_millijoules as f64,
            CandidateObjectiveMetric::TotalInputTokens => self.input_tokens as f64,
            CandidateObjectiveMetric::TotalOutputTokens => self.output_tokens as f64,
            CandidateObjectiveMetric::TotalToolCalls => self.tool_calls as f64,
            CandidateObjectiveMetric::TotalEvaluatorCalls => self.evaluator_calls as f64,
            CandidateObjectiveMetric::TotalVerifierCalls => self.verifier_calls as f64,
            CandidateObjectiveMetric::TotalChargedBudgetMicrounits => self.charged_budget_microunits as f64,
            CandidateObjectiveMetric::MeanVerifiedScore
            | CandidateObjectiveMetric::MinimumVerifiedScore
            | CandidateObjectiveMetric::MeanReportedUncertainty
            | CandidateObjectiveMetric::MaximumReportedUncertainty
            | CandidateObjectiveMetric::LowerConfidenceBoundVerifiedScore { .. }
            | CandidateObjectiveMetric::VerifiedAttemptRate => unreachable!("score metrics are not resource totals"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEventReference {
    pub sequence: u64,
    pub event_sha256: String,
    pub event_id: String,
    pub direction_id: String,
    pub retry_of_event_id: Option<String>,
    pub disposition: CandidateDisposition,
    pub score: Option<f64>,
    pub uncertainty: Option<f64>,
    pub evidence_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEvidenceManifest {
    pub candidate_sha256: String,
    pub policy_id: String,
    pub events: Vec<CandidateEventReference>,
    pub manifest_sha256: String,
}

#[derive(Serialize)]
struct CandidateEvidenceManifestPayload<'a> {
    candidate_sha256: &'a str,
    policy_id: &'a str,
    events: &'a [CandidateEventReference],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEvaluationSummary {
    pub candidate_sha256: String,
    pub direction_ids: Vec<String>,
    pub evaluation_attempts: u64,
    pub verified_attempts: u64,
    pub verified_attempt_rate: f64,
    pub resources: ActionResourceTotals,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventTapeParetoReport {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub policy_id: String,
    pub source_tape_sha256: String,
    pub config_sha256: String,
    pub pareto_plan: ParetoAnalysisPlan,
    pub pareto_analysis: ParetoAnalysisSummary,
    /// One manifest for each distinct evaluated candidate hash.
    pub candidate_evidence_manifests: Vec<CandidateEvidenceManifest>,
    pub candidate_summaries: Vec<CandidateEvaluationSummary>,
    /// Includes candidate evaluations and direction-proposal actions.
    pub campaign_resources: ActionResourceTotals,
    /// CandidateEvaluation actions only; failed attempts and retries remain included.
    pub candidate_evaluation_resources: ActionResourceTotals,
    /// ProposeDirection actions only; never silently assigned to each candidate.
    pub direction_proposal_overhead: ActionResourceTotals,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct EventTapeParetoReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    policy_id: &'a str,
    source_tape_sha256: &'a str,
    config_sha256: &'a str,
    pareto_plan: &'a ParetoAnalysisPlan,
    pareto_analysis: &'a ParetoAnalysisSummary,
    candidate_evidence_manifests: &'a [CandidateEvidenceManifest],
    candidate_summaries: &'a [CandidateEvaluationSummary],
    campaign_resources: &'a ActionResourceTotals,
    candidate_evaluation_resources: &'a ActionResourceTotals,
    direction_proposal_overhead: &'a ActionResourceTotals,
}

#[derive(Default)]
struct CandidateAccumulator {
    resources: ActionResourceTotals,
    scores: Vec<f64>,
    uncertainties: Vec<f64>,
    verified_attempts: u64,
    directions: BTreeSet<String>,
    evidence_events: Vec<CandidateEventReference>,
}

pub fn event_tape_to_pareto(
    tape: &EventTape,
    config: &EventTapeParetoConfig,
) -> Result<EventTapeParetoReport, EventTapeParetoError> {
    config.validate()?;
    tape.validate()
        .map_err(|error| EventTapeParetoError::new("event_tape", error.to_string()))?;

    let mut campaign_resources = ActionResourceTotals::default();
    let mut candidate_evaluation_resources = ActionResourceTotals::default();
    let mut direction_proposal_overhead = ActionResourceTotals::default();
    let mut candidates = BTreeMap::<String, CandidateAccumulator>::new();

    for envelope in tape.events() {
        let payload = &envelope.payload;
        let path = format!("events[{}]", envelope.sequence);
        campaign_resources.add_receipt(&payload.resources, &path)?;

        match (&payload.intent, &payload.outcome) {
            (
                ActionIntent::EvaluateCandidate {
                    candidate_sha256,
                    direction_id,
                    ..
                },
                ActionOutcome::CandidateEvaluation {
                    disposition,
                    score,
                    evidence_sha256,
                    uncertainty,
                    ..
                },
            ) => {
                candidate_evaluation_resources.add_receipt(&payload.resources, &path)?;
                let accumulator = candidates.entry(candidate_sha256.clone()).or_default();
                accumulator.resources.add_receipt(&payload.resources, &path)?;
                accumulator.directions.insert(direction_id.clone());
                if *disposition == CandidateDisposition::Verified {
                    let verified_score = score.ok_or_else(|| {
                        EventTapeParetoError::new(
                            format!("{path}.outcome.score"),
                            "strictly validated verified evaluation is missing its score",
                        )
                    })?;
                    accumulator.scores.push(verified_score);
                    if let Some(reported_uncertainty) = uncertainty {
                        accumulator.uncertainties.push(*reported_uncertainty);
                    }
                    accumulator.verified_attempts = checked_add(
                        accumulator.verified_attempts,
                        1,
                        &format!("{path}.verified_attempts"),
                    )?;
                }
                accumulator.evidence_events.push(CandidateEventReference {
                    sequence: envelope.sequence,
                    event_sha256: envelope.event_sha256.clone(),
                    event_id: payload.event_id.clone(),
                    direction_id: direction_id.clone(),
                    retry_of_event_id: payload.retry_of_event_id.clone(),
                    disposition: *disposition,
                    score: *score,
                    uncertainty: *uncertainty,
                    evidence_sha256: evidence_sha256.clone(),
                });
            }
            (ActionIntent::ProposeDirection { .. }, ActionOutcome::DirectionVerification { .. }) => {
                direction_proposal_overhead.add_receipt(&payload.resources, &path)?;
            }
            _ => {
                return Err(EventTapeParetoError::new(
                    format!("{path}.outcome"),
                    "event intent/outcome pairing is not recognized by the strict EventTape schema",
                ));
            }
        }
    }

    let mut reconciled_resources = candidate_evaluation_resources.clone();
    reconciled_resources.checked_add_totals(
        &direction_proposal_overhead,
        "resource_reconciliation",
    )?;
    if reconciled_resources != campaign_resources {
        return Err(EventTapeParetoError::new(
            "resource_reconciliation",
            "candidate-evaluation totals plus proposal overhead do not equal campaign totals",
        ));
    }

    if candidates.is_empty() {
        return Err(EventTapeParetoError::new(
            "event_tape.candidates",
            "tape contains no candidate-evaluation events",
        ));
    }

    let mut pareto_candidates = Vec::with_capacity(candidates.len());
    let mut candidate_evidence_manifests = Vec::with_capacity(candidates.len());
    let mut candidate_summaries = Vec::with_capacity(candidates.len());

    for (candidate_sha256, accumulator) in candidates {
        let evaluation_attempts = accumulator.resources.action_count;
        let verified_attempt_rate = accumulator.verified_attempts as f64 / evaluation_attempts as f64;
        let feasible = accumulator.verified_attempts >= config.minimum_verified_evaluations;
        let constraint_violations = if feasible {
            Vec::new()
        } else {
            vec![format!(
                "verified_attempts_below_minimum:{}<{}",
                accumulator.verified_attempts, config.minimum_verified_evaluations
            )]
        };
        let evidence_payload_events = accumulator.evidence_events.clone();
        let manifest_sha256 = digest_serializable(&CandidateEvidenceManifestPayload {
            candidate_sha256: &candidate_sha256,
            policy_id: &tape.header().policy.policy_id,
            events: &evidence_payload_events,
        })?;
        let manifest = CandidateEvidenceManifest {
            candidate_sha256: candidate_sha256.clone(),
            policy_id: tape.header().policy.policy_id.clone(),
            events: accumulator.evidence_events.clone(),
            manifest_sha256,
        };

        let mean_score = if accumulator.scores.is_empty() {
            config.unverified_score_value
        } else {
            accumulator.scores.iter().sum::<f64>() / accumulator.scores.len() as f64
        };
        if !mean_score.is_finite() {
            return Err(EventTapeParetoError::new(
                format!("candidate[{candidate_sha256}].mean_score"),
                "mean verified score overflowed",
            ));
        }
        let minimum_score = accumulator.scores.iter().copied()
            .reduce(f64::min)
            .unwrap_or(config.unverified_score_value);
        if accumulator.uncertainties.len() != accumulator.verified_attempts as usize
            && config.objective_mappings.iter().any(|mapping| matches!(
                &mapping.metric,
                CandidateObjectiveMetric::MeanReportedUncertainty
                    | CandidateObjectiveMetric::MaximumReportedUncertainty
            ))
        {
            return Err(EventTapeParetoError::new(
                format!("candidate[{candidate_sha256}].uncertainty"),
                "every verified attempt must include evaluator-reported uncertainty when an uncertainty objective is selected",
            ));
        }
        let mean_uncertainty = if accumulator.uncertainties.is_empty() {
            None
        } else {
            Some(accumulator.uncertainties.iter().sum::<f64>() / accumulator.uncertainties.len() as f64)
        };
        let maximum_uncertainty = accumulator.uncertainties.iter().copied().reduce(f64::max);

        let mut objective_values = Vec::with_capacity(config.objective_mappings.len());
        for (index, mapping) in config.objective_mappings.iter().enumerate() {
            let value = match &mapping.metric {
                CandidateObjectiveMetric::MeanVerifiedScore => mean_score,
                CandidateObjectiveMetric::MinimumVerifiedScore => minimum_score,
                CandidateObjectiveMetric::MeanReportedUncertainty => mean_uncertainty.unwrap_or(
                    config.objective_specs[index].scale_upper
                ),
                CandidateObjectiveMetric::MaximumReportedUncertainty => maximum_uncertainty.unwrap_or(
                    config.objective_specs[index].scale_upper
                ),
                CandidateObjectiveMetric::LowerConfidenceBoundVerifiedScore { z_value } => {
                    if accumulator.scores.is_empty() {
                        config.unverified_score_value
                    } else if accumulator.scores.len() < 2 {
                        return Err(EventTapeParetoError::new(
                            format!("candidate[{candidate_sha256}].objective_mappings[{index}]"),
                            "lower-confidence-bound score requires at least two verified repeated evaluations",
                        ));
                    } else {
                        let n = accumulator.scores.len() as f64;
                        let mean = mean_score;
                        let sample_variance = accumulator.scores.iter()
                            .map(|score| (score - mean).powi(2))
                            .sum::<f64>() / (n - 1.0);
                        mean - *z_value * (sample_variance.sqrt() / n.sqrt())
                    }
                }
                CandidateObjectiveMetric::VerifiedAttemptRate => verified_attempt_rate,
                metric => accumulator.resources.value(metric),
            };
            if !value.is_finite() {
                return Err(EventTapeParetoError::new(
                    format!("candidate[{candidate_sha256}].objective_values[{index}]"),
                    "objective value is not finite",
                ));
            }
            objective_values.push(value);
        }

        let direction_ids = accumulator.directions.into_iter().collect::<Vec<_>>();
        candidate_summaries.push(CandidateEvaluationSummary {
            candidate_sha256: candidate_sha256.clone(),
            direction_ids,
            evaluation_attempts,
            verified_attempts: accumulator.verified_attempts,
            verified_attempt_rate,
            resources: accumulator.resources,
        });
        candidate_evidence_manifests.push(manifest);
        pareto_candidates.push(ParetoCandidate {
            candidate_id: candidate_sha256,
            objective_values,
            feasible,
            constraint_violations,
            evidence_sha256: candidate_evidence_manifests.last()
                .map(|manifest| manifest.manifest_sha256.clone())
                .ok_or_else(|| EventTapeParetoError::new("candidate.evidence", "candidate evidence manifest missing"))?,
        });
    }

    let pareto_plan = ParetoAnalysisPlan::new(
        tape.header().campaign_id.clone(),
        config.objective_specs.clone(),
        config.reference_point_utility.clone(),
        pareto_candidates,
    ).map_err(map_pareto_error)?;
    let pareto_analysis = analyze_pareto(&pareto_plan).map_err(map_pareto_error)?;

    let source_tape_sha256 = digest_serializable(tape)?;
    let config_sha256 = config.config_sha256()?;
    let report_sha256 = digest_serializable(&EventTapeParetoReportPayload {
        schema: SCHEMA,
        version: VERSION,
        campaign_id: &tape.header().campaign_id,
        policy_id: &tape.header().policy.policy_id,
        source_tape_sha256: &source_tape_sha256,
        config_sha256: &config_sha256,
        pareto_plan: &pareto_plan,
        pareto_analysis: &pareto_analysis,
        candidate_evidence_manifests: &candidate_evidence_manifests,
        candidate_summaries: &candidate_summaries,
        campaign_resources: &campaign_resources,
        candidate_evaluation_resources: &candidate_evaluation_resources,
        direction_proposal_overhead: &direction_proposal_overhead,
    })?;

    Ok(EventTapeParetoReport {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        campaign_id: tape.header().campaign_id.clone(),
        policy_id: tape.header().policy.policy_id.clone(),
        source_tape_sha256,
        config_sha256,
        pareto_plan,
        pareto_analysis,
        candidate_evidence_manifests,
        candidate_summaries,
        campaign_resources,
        candidate_evaluation_resources,
        direction_proposal_overhead,
        report_sha256,
    })
}

fn read_meter(meter: &MeterValue, path: &str) -> Result<u64, EventTapeParetoError> {
    match meter {
        MeterValue::Measured { value } => Ok(*value),
        MeterValue::Unknown { reason } => Err(EventTapeParetoError::new(
            path,
            format!("unknown resource measurement cannot enter a strict Pareto objective: {reason}"),
        )),
    }
}

fn add_meter(current: u64, meter: &MeterValue, path: &str) -> Result<u64, EventTapeParetoError> {
    checked_add(current, read_meter(meter, path)?, path)
}

fn checked_add(left: u64, right: u64, path: &str) -> Result<u64, EventTapeParetoError> {
    left.checked_add(right)
        .ok_or_else(|| EventTapeParetoError::new(path, "resource aggregation overflowed"))
}

fn map_pareto_error(error: ParetoError) -> EventTapeParetoError {
    EventTapeParetoError::new(error.path, error.message)
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, EventTapeParetoError> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        EventTapeParetoError::new("serialization", format!("cannot serialize report: {error}"))
    })?;
    let digest = Sha256::digest(encoded);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_tape::{
        ActionEvent, BudgetModel, CandidateDisposition, ComparisonMode, EventTapeHeader,
        ExecutionContract, ExecutionMode, PolicySnapshot, ProposalDisposition,
        ResourceCostWeights,
    };
    use crate::pareto_objectives::ObjectiveDirection;
    use serde_json::json;

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const H5: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn meter(value: u64) -> MeterValue {
        MeterValue::Measured { value }
    }

    fn resources(cpu: u64, wall: u64, energy: u64, peak: u64, charge: u64) -> ResourceReceipt {
        ResourceReceipt {
            host_cpu_millis: meter(cpu),
            wall_millis: meter(wall),
            accelerator_millis: meter(0),
            peak_memory_bytes: meter(peak),
            accelerator_peak_memory_bytes: meter(0),
            energy_millijoules: meter(energy),
            input_tokens: meter(100),
            output_tokens: meter(20),
            tool_calls: meter(1),
            evaluator_calls: meter(1),
            verifier_calls: meter(1),
            charged_budget_microunits: meter(charge),
        }
    }

    fn header(policy_id: &str, budget: u64) -> EventTapeHeader {
        let policy_config = json!({"seed": 42, "policy": policy_id});
        let budget_model = BudgetModel::WeightedResources {
            weights: ResourceCostWeights {
                host_cpu_millis: 1,
                wall_millis: 0,
                accelerator_millis: 0,
                host_memory_byte_millis: 0,
                accelerator_memory_byte_millis: 0,
                energy_millijoules: 0,
                input_tokens: 0,
                output_tokens: 0,
                tool_calls: 0,
                evaluator_calls: 0,
                verifier_calls: 0,
            },
        };
        EventTapeHeader {
            schema: "symthaea.ar5b4.event-tape".to_owned(),
            version: "0.1.0".to_owned(),
            campaign_id: "event-tape-pareto-test".to_owned(),
            source_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            execution_environment_sha256: H1.to_owned(),
            comparison_mode: ComparisonMode::MechanismMatched,
            proposal_source_id: "shared-proposal-source".to_owned(),
            proposal_stream_sha256: Some(H2.to_owned()),
            task_recipe_sha256: H3.to_owned(),
            comparison_profile_sha256: H4.to_owned(),
            evaluator_sha256: H5.to_owned(),
            verifier_sha256: H1.to_owned(),
            budget_model_id: "cpu-charge-v1".to_owned(),
            budget_model_sha256: digest_serializable(&budget_model).unwrap(),
            budget_model,
            budget_limit_microunits: budget,
            initial_direction_ids: vec!["A".to_owned(), "B".to_owned()],
            execution: ExecutionContract {
                mode: ExecutionMode::Sequential,
                worker_count: 1,
                seats_per_round: 1,
                first_round_claim_cap: None,
                steady_state_claim_cap: None,
            },
            policy: PolicySnapshot {
                policy_id: policy_id.to_owned(),
                algorithm_id: format!("algorithm-{policy_id}"),
                config_sha256: digest_serializable(&policy_config).unwrap(),
                config: policy_config,
                has_hidden_oracle_access: false,
            },
        }
    }

    fn candidate_event(
        tape: &mut EventTape,
        event_id: &str,
        candidate: &str,
        direction: &str,
        disposition: CandidateDisposition,
        score: Option<f64>,
        evidence: Option<&str>,
        retry: Option<&str>,
        cpu: u64,
        wall: u64,
        energy: u64,
        peak: u64,
    ) {
        let sequence = tape.events().len() as u64 + 1;
        tape.append(ActionEvent {
            event_id: event_id.to_owned(),
            policy_id: tape.header().policy.policy_id.clone(),
            worker_id: 0,
            round_id: sequence,
            batch_id: None,
            observed_through_sequence: sequence - 1,
            visible_history_sha256: String::new(),
            retry_of_event_id: retry.map(str::to_owned),
            intent: ActionIntent::EvaluateCandidate {
                candidate_sha256: candidate.to_owned(),
                parent_candidate_sha256: None,
                semantic_delta_sha256: H2.to_owned(),
                direction_id: direction.to_owned(),
            },
            outcome: ActionOutcome::CandidateEvaluation {
                task_recipe_sha256: H3.to_owned(),
                comparison_profile_sha256: H4.to_owned(),
                evaluator_sha256: H5.to_owned(),
                verifier_sha256: H1.to_owned(),
                verifier_receipt_sha256: H2.to_owned(),
                disposition,
                score,
                uncertainty: if disposition == CandidateDisposition::Verified { Some(0.0) } else { None },
                evidence_sha256: evidence.map(str::to_owned),
            },
            resources: resources(cpu, wall, energy, peak, cpu),
        }).unwrap();
    }

    fn proposal_event(tape: &mut EventTape, event_id: &str, cpu: u64) {
        let sequence = tape.events().len() as u64 + 1;
        tape.append(ActionEvent {
            event_id: event_id.to_owned(),
            policy_id: tape.header().policy.policy_id.clone(),
            worker_id: 0,
            round_id: sequence,
            batch_id: None,
            observed_through_sequence: sequence - 1,
            visible_history_sha256: String::new(),
            retry_of_event_id: None,
            intent: ActionIntent::ProposeDirection {
                descriptor_sha256: H5.to_owned(),
                parent_direction_id: Some("A".to_owned()),
                semantic_delta_sha256: H3.to_owned(),
            },
            outcome: ActionOutcome::DirectionVerification {
                task_recipe_sha256: H3.to_owned(),
                comparison_profile_sha256: H4.to_owned(),
                verifier_sha256: H1.to_owned(),
                verifier_receipt_sha256: H2.to_owned(),
                disposition: ProposalDisposition::Duplicate {
                    canonical_direction_id: "A".to_owned(),
                },
            },
            resources: resources(cpu, 1, 1, 512, cpu),
        }).unwrap();
    }

    fn config() -> EventTapeParetoConfig {
        EventTapeParetoConfig {
            objective_specs: vec![
                ParetoObjectiveSpec {
                    objective_id: "quality".to_owned(),
                    direction: ObjectiveDirection::Maximize,
                    scale_lower: 0.0,
                    scale_upper: 1.0,
                },
                ParetoObjectiveSpec {
                    objective_id: "verified_rate".to_owned(),
                    direction: ObjectiveDirection::Maximize,
                    scale_lower: 0.0,
                    scale_upper: 1.0,
                },
                ParetoObjectiveSpec {
                    objective_id: "energy".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    scale_lower: 0.0,
                    scale_upper: 100.0,
                },
            ],
            objective_mappings: vec![
                CandidateObjectiveMapping {
                    objective_id: "quality".to_owned(),
                    metric: CandidateObjectiveMetric::MeanVerifiedScore,
                },
                CandidateObjectiveMapping {
                    objective_id: "verified_rate".to_owned(),
                    metric: CandidateObjectiveMetric::VerifiedAttemptRate,
                },
                CandidateObjectiveMapping {
                    objective_id: "energy".to_owned(),
                    metric: CandidateObjectiveMetric::TotalEnergyMillijoules,
                },
            ],
            reference_point_utility: vec![0.0, 0.0, 0.0],
            unverified_score_value: 0.0,
            minimum_verified_evaluations: 1,
        }
    }

    #[test]
    fn adapter_keeps_failed_retries_charged_and_proposal_overhead_separate() {
        let mut tape = EventTape::new(header("direction-board", 100)).unwrap();
        candidate_event(&mut tape, "a1", H2, "A", CandidateDisposition::Verified, Some(0.9), Some(H3), None, 10, 2, 2, 1024);
        candidate_event(&mut tape, "a2", H2, "A", CandidateDisposition::TimedOut, None, None, Some("a1"), 20, 5, 3, 2048);
        candidate_event(&mut tape, "b1", H5, "B", CandidateDisposition::Verified, Some(0.75), Some(H4), None, 8, 3, 1, 1536);
        proposal_event(&mut tape, "p1", 7);

        let report = event_tape_to_pareto(&tape, &config()).unwrap();
        assert_eq!(report.candidate_summaries.len(), 2);
        assert_eq!(report.candidate_evaluation_resources.action_count, 3);
        assert_eq!(report.direction_proposal_overhead.action_count, 1);
        assert_eq!(report.campaign_resources.action_count, 4);
        assert_eq!(report.campaign_resources.host_cpu_millis, 45);
        let mut recomposed = report.candidate_evaluation_resources.clone();
        recomposed.checked_add_totals(&report.direction_proposal_overhead, "test").unwrap();
        assert_eq!(recomposed, report.campaign_resources);
        assert_eq!(report.candidate_evaluation_resources.host_cpu_millis, 38);
        assert_eq!(report.direction_proposal_overhead.host_cpu_millis, 7);

        let candidate_a = report.candidate_summaries.iter()
            .find(|candidate| candidate.candidate_sha256 == H2)
            .unwrap();
        assert_eq!(candidate_a.evaluation_attempts, 2);
        assert_eq!(candidate_a.verified_attempts, 1);
        assert_eq!(candidate_a.verified_attempt_rate, 0.5);
        assert_eq!(candidate_a.resources.energy_millijoules, 5);
        assert_eq!(candidate_a.resources.max_action_peak_memory_bytes, 2048);
        assert_eq!(report.pareto_analysis.nondominated_candidate_ids.len(), 2);
        assert_eq!(report.report_sha256.len(), 64);
    }

    #[test]
    fn candidate_without_verified_evaluation_is_retained_as_infeasible() {
        let mut tape = EventTape::new(header("direction-board", 100)).unwrap();
        candidate_event(&mut tape, "bad1", H2, "A", CandidateDisposition::Rejected, None, None, None, 10, 2, 2, 1024);
        candidate_event(&mut tape, "good1", H5, "B", CandidateDisposition::Verified, Some(0.8), Some(H4), None, 8, 3, 1, 1536);

        let report = event_tape_to_pareto(&tape, &config()).unwrap();
        let bad = report.pareto_plan.candidates.iter()
            .find(|candidate| candidate.candidate_id == H2)
            .unwrap();
        assert!(!bad.feasible);
        assert_eq!(bad.objective_values[0], 0.0);
        assert_eq!(report.pareto_analysis.infeasible_candidate_count, 1);
    }

    #[test]
    fn missing_required_resource_meter_blocks_adapter_before_pareto_analysis() {
        let mut tape = EventTape::new(header("direction-board", 100)).unwrap();
        candidate_event(&mut tape, "a1", H2, "A", CandidateDisposition::Verified, Some(0.9), Some(H3), None, 10, 2, 2, 1024);
        let mut payload = tape.events()[0].payload.clone();
        payload.event_id = "a2".to_owned();
        payload.round_id = 2;
        payload.observed_through_sequence = 1;
        payload.resources.energy_millijoules = MeterValue::Unknown {
            reason: "meter unavailable".to_owned(),
        };
        payload.resources.charged_budget_microunits = meter(10);
        tape.append(payload).unwrap();
        assert!(event_tape_to_pareto(&tape, &config()).is_err());
    }

    #[test]
    fn uncertainty_objective_requires_uncertainty_for_every_verified_attempt() {
        let mut tape = EventTape::new(header("direction-board", 100)).unwrap();
        candidate_event(&mut tape, "a1", H2, "A", CandidateDisposition::Verified, Some(0.9), Some(H3), None, 10, 2, 2, 1024);
        let mut payload = tape.events()[0].payload.clone();
        payload.event_id = "a2".to_owned();
        payload.round_id = 2;
        payload.observed_through_sequence = 1;
        if let ActionOutcome::CandidateEvaluation { uncertainty, .. } = &mut payload.outcome {
            *uncertainty = None;
        }
        tape.append(payload).unwrap();

        let mut uncertainty_config = config();
        uncertainty_config.objective_specs[1] = ParetoObjectiveSpec {
            objective_id: "uncertainty".to_owned(),
            direction: ObjectiveDirection::Minimize,
            scale_lower: 0.0,
            scale_upper: 1.0,
        };
        uncertainty_config.objective_mappings[1] = CandidateObjectiveMapping {
            objective_id: "uncertainty".to_owned(),
            metric: CandidateObjectiveMetric::MeanReportedUncertainty,
        };
        assert!(event_tape_to_pareto(&tape, &uncertainty_config).is_err());
    }

    #[test]
    fn objective_direction_mismatch_and_lcb_with_one_sample_are_rejected() {
        let mut invalid_config = config();
        invalid_config.objective_specs[2].direction = ObjectiveDirection::Maximize;
        assert!(invalid_config.validate().is_err());

        let mut tape = EventTape::new(header("direction-board", 100)).unwrap();
        candidate_event(&mut tape, "a1", H2, "A", CandidateDisposition::Verified, Some(0.9), Some(H3), None, 10, 2, 2, 1024);
        let mut lcb_config = config();
        lcb_config.objective_mappings[0].metric =
            CandidateObjectiveMetric::LowerConfidenceBoundVerifiedScore { z_value: 1.96 };
        assert!(event_tape_to_pareto(&tape, &lcb_config).is_err());
    }
}
