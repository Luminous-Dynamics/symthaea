//! Replay-verifiable sequential anytime Pareto histories and target-attainment receipts.
//!
//! Optimizers emit one common event format. Failed attempts and verifier work are charged.
//! Target crossings are computed from the archive prefix visible at that exact event.
//! v0.1 rejects overlapping event intervals; parallel/asynchronous execution needs a
//! separately versioned accounting contract.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::pareto_objectives::{
    analyze_pareto, analyze_reference_front_indicators, ParetoAnalysisPlan, ParetoCandidate,
    ParetoObjectiveSpec,
};

const SCHEMA: &str = "symthaea.architecture-engineering.pareto-anytime";
const VERSION: &str = "0.1.0";
const TARGET_PLAN_SCHEMA: &str = "symthaea.architecture-engineering.anytime-target-plan";
const TARGET_PLAN_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnytimeError {
    pub path: String,
    pub message: String,
}

impl AnytimeError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for AnytimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}
impl Error for AnytimeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLevel {
    Unverified,
    Recorded,
    ReplayVerified,
    ArtifactVerified,
    HeldOutVerified,
    Confirmatory,
}

impl EvidenceLevel {
    fn rank(self) -> u8 {
        match self {
            Self::Unverified => 0,
            Self::Recorded => 1,
            Self::ReplayVerified => 2,
            Self::ArtifactVerified => 3,
            Self::HeldOutVerified => 4,
            Self::Confirmatory => 5,
        }
    }

    pub fn meets(self, required: Self) -> bool {
        self.rank() >= required.rank()
    }

    fn may_qualify_target(self) -> bool {
        !matches!(self, Self::Unverified)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationOutcome {
    Succeeded,
    Failed,
    Invalid,
    TimedOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnytimeTargetMetric {
    Hypervolume,
    IgdPlus,
    AdditiveEpsilon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceMeter {
    HostCpuMillis,
    WorkerTimeMillis,
    AcceleratorMillis,
    PeakHostMemoryBytes,
    PeakAcceleratorMemoryBytes,
    EnergyMillijoules,
    InputTokens,
    OutputTokens,
    ToolCalls,
    EvaluatorCalls,
    VerifierCalls,
    NormalizedChargeMicrounits,
}

impl ResourceMeter {
    pub const ALL: [Self; 12] = [
        Self::HostCpuMillis, Self::WorkerTimeMillis, Self::AcceleratorMillis,
        Self::PeakHostMemoryBytes, Self::PeakAcceleratorMemoryBytes, Self::EnergyMillijoules,
        Self::InputTokens, Self::OutputTokens, Self::ToolCalls, Self::EvaluatorCalls,
        Self::VerifierCalls, Self::NormalizedChargeMicrounits,
    ];
}

/// Raw counters: None means unknown, never zero. Additive meters are summed; peak memory
/// uses max. Elapsed wall time comes from monotonic event completion timestamps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceCost {
    pub host_cpu_millis: Option<u64>,
    pub worker_time_millis: Option<u64>,
    pub accelerator_millis: Option<u64>,
    pub peak_host_memory_bytes: Option<u64>,
    pub peak_accelerator_memory_bytes: Option<u64>,
    pub energy_millijoules: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub tool_calls: Option<u64>,
    pub evaluator_calls: Option<u64>,
    pub verifier_calls: Option<u64>,
    pub normalized_charge_microunits: Option<u64>,
}

impl ResourceCost {
    pub fn unknown() -> Self {
        Self {
            host_cpu_millis: None, worker_time_millis: None, accelerator_millis: None,
            peak_host_memory_bytes: None, peak_accelerator_memory_bytes: None,
            energy_millijoules: None, input_tokens: None, output_tokens: None,
            tool_calls: None, evaluator_calls: None, verifier_calls: None,
            normalized_charge_microunits: None,
        }
    }

    pub fn zero() -> Self {
        Self {
            host_cpu_millis: Some(0), worker_time_millis: Some(0), accelerator_millis: Some(0),
            peak_host_memory_bytes: Some(0), peak_accelerator_memory_bytes: Some(0),
            energy_millijoules: Some(0), input_tokens: Some(0), output_tokens: Some(0),
            tool_calls: Some(0), evaluator_calls: Some(0), verifier_calls: Some(0),
            normalized_charge_microunits: Some(0),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn known(
        host_cpu_millis: u64, worker_time_millis: u64, accelerator_millis: u64,
        peak_host_memory_bytes: u64, peak_accelerator_memory_bytes: u64,
        energy_millijoules: u64, input_tokens: u64, output_tokens: u64,
        tool_calls: u64, evaluator_calls: u64, verifier_calls: u64,
        normalized_charge_microunits: u64,
    ) -> Self {
        Self {
            host_cpu_millis: Some(host_cpu_millis),
            worker_time_millis: Some(worker_time_millis),
            accelerator_millis: Some(accelerator_millis),
            peak_host_memory_bytes: Some(peak_host_memory_bytes),
            peak_accelerator_memory_bytes: Some(peak_accelerator_memory_bytes),
            energy_millijoules: Some(energy_millijoules),
            input_tokens: Some(input_tokens),
            output_tokens: Some(output_tokens),
            tool_calls: Some(tool_calls),
            evaluator_calls: Some(evaluator_calls),
            verifier_calls: Some(verifier_calls),
            normalized_charge_microunits: Some(normalized_charge_microunits),
        }
    }

    pub fn meter_is_known(&self, meter: ResourceMeter) -> bool {
        match meter {
            ResourceMeter::HostCpuMillis => self.host_cpu_millis.is_some(),
            ResourceMeter::WorkerTimeMillis => self.worker_time_millis.is_some(),
            ResourceMeter::AcceleratorMillis => self.accelerator_millis.is_some(),
            ResourceMeter::PeakHostMemoryBytes => self.peak_host_memory_bytes.is_some(),
            ResourceMeter::PeakAcceleratorMemoryBytes => self.peak_accelerator_memory_bytes.is_some(),
            ResourceMeter::EnergyMillijoules => self.energy_millijoules.is_some(),
            ResourceMeter::InputTokens => self.input_tokens.is_some(),
            ResourceMeter::OutputTokens => self.output_tokens.is_some(),
            ResourceMeter::ToolCalls => self.tool_calls.is_some(),
            ResourceMeter::EvaluatorCalls => self.evaluator_calls.is_some(),
            ResourceMeter::VerifierCalls => self.verifier_calls.is_some(),
            ResourceMeter::NormalizedChargeMicrounits => self.normalized_charge_microunits.is_some(),
        }
    }

    fn add_event(&mut self, event: &Self, path: &str) -> Result<(), AnytimeError> {
        add_sum(&mut self.host_cpu_millis, event.host_cpu_millis, &format!("{path}.host_cpu_millis"))?;
        add_sum(&mut self.worker_time_millis, event.worker_time_millis, &format!("{path}.worker_time_millis"))?;
        add_sum(&mut self.accelerator_millis, event.accelerator_millis, &format!("{path}.accelerator_millis"))?;
        add_peak(&mut self.peak_host_memory_bytes, event.peak_host_memory_bytes);
        add_peak(&mut self.peak_accelerator_memory_bytes, event.peak_accelerator_memory_bytes);
        add_sum(&mut self.energy_millijoules, event.energy_millijoules, &format!("{path}.energy_millijoules"))?;
        add_sum(&mut self.input_tokens, event.input_tokens, &format!("{path}.input_tokens"))?;
        add_sum(&mut self.output_tokens, event.output_tokens, &format!("{path}.output_tokens"))?;
        add_sum(&mut self.tool_calls, event.tool_calls, &format!("{path}.tool_calls"))?;
        add_sum(&mut self.evaluator_calls, event.evaluator_calls, &format!("{path}.evaluator_calls"))?;
        add_sum(&mut self.verifier_calls, event.verifier_calls, &format!("{path}.verifier_calls"))?;
        add_sum(&mut self.normalized_charge_microunits, event.normalized_charge_microunits, &format!("{path}.normalized_charge_microunits"))?;
        Ok(())
    }
}

fn add_sum(total: &mut Option<u64>, value: Option<u64>, path: &str) -> Result<(), AnytimeError> {
    *total = match (*total, value) {
        (Some(left), Some(right)) => Some(left.checked_add(right)
            .ok_or_else(|| AnytimeError::new(path, "cumulative resource counter overflowed u64"))?),
        _ => None,
    };
    Ok(())
}

fn add_peak(total: &mut Option<u64>, value: Option<u64>) {
    *total = match (*total, value) {
        (Some(left), Some(right)) => Some(left.max(right)),
        _ => None,
    };
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeTarget {
    pub target_id: String,
    pub metric: AnytimeTargetMetric,
    /// HV uses >=; IGD+ and additive epsilon use <=.
    pub threshold: f64,
    pub minimum_evidence_level: EvidenceLevel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeTargetPlan {
    pub schema: String,
    pub version: String,
    pub targets: Vec<AnytimeTarget>,
    /// Canonical reference front in transformed maximization-utility coordinates.
    pub reference_front_utility: Vec<Vec<f64>>,
}

impl AnytimeTargetPlan {
    pub fn new(
        targets: Vec<AnytimeTarget>,
        reference_front_utility: Vec<Vec<f64>>,
    ) -> Result<Self, AnytimeError> {
        let mut reference_front_utility = reference_front_utility;
        reference_front_utility.sort_by(|left, right| {
            for (a, b) in left.iter().zip(right) {
                let ordering = a.total_cmp(b);
                if ordering != std::cmp::Ordering::Equal {
                    return ordering;
                }
            }
            left.len().cmp(&right.len())
        });
        let plan = Self {
            schema: TARGET_PLAN_SCHEMA.to_owned(),
            version: TARGET_PLAN_VERSION.to_owned(),
            targets,
            reference_front_utility,
        };
        let dimensions = plan.reference_front_utility.first().map_or(0, Vec::len);
        plan.validate(dimensions)?;
        Ok(plan)
    }

    pub fn validate(&self, dimensions: usize) -> Result<(), AnytimeError> {
        if self.schema != TARGET_PLAN_SCHEMA || self.version != TARGET_PLAN_VERSION {
            return Err(AnytimeError::new("target_plan.schema", "unsupported target-plan schema/version"));
        }
        if self.targets.is_empty() {
            return Err(AnytimeError::new("target_plan.targets", "at least one target is required"));
        }
        let mut ids = BTreeSet::new();
        for (index, target) in self.targets.iter().enumerate() {
            if target.target_id.trim().is_empty() || !ids.insert(target.target_id.as_str()) {
                return Err(AnytimeError::new(format!("target_plan.targets[{index}].target_id"), "target IDs must be non-empty and unique"));
            }
            if !target.threshold.is_finite() || !target.minimum_evidence_level.may_qualify_target() {
                return Err(AnytimeError::new(format!("target_plan.targets[{index}]"), "threshold must be finite and Unverified is not an admissible target level"));
            }
            if (target.metric == AnytimeTargetMetric::Hypervolume && target.threshold <= 0.0)
                || (target.metric == AnytimeTargetMetric::IgdPlus && target.threshold < 0.0)
            {
                return Err(AnytimeError::new(format!("target_plan.targets[{index}].threshold"), "hypervolume targets must be positive and IGD+ targets non-negative"));
            }
        }
        if self.reference_front_utility.is_empty() {
            return Err(AnytimeError::new("target_plan.reference_front_utility", "reference front cannot be empty"));
        }
        for (index, point) in self.reference_front_utility.iter().enumerate() {
            if point.len() != dimensions || point.iter().any(|value| !value.is_finite()) {
                return Err(AnytimeError::new(format!("target_plan.reference_front_utility[{index}]"), "reference point must have objective dimension and finite values"));
            }
            for (other_index, other) in self.reference_front_utility.iter().enumerate() {
                if index != other_index && (same_vector(point, other) || dominates_utility(other, point)) {
                    return Err(AnytimeError::new(format!("target_plan.reference_front_utility[{index}]"), "reference front must contain unique mutually nondominated points"));
                }
            }
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, AnytimeError> {
        let dimensions = self.reference_front_utility.first().map_or(0, Vec::len);
        self.validate(dimensions)?;
        digest_serializable(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeHistoryHeader {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub algorithm_id: String,
    pub algorithm_build_sha256: String,
    pub evaluator_id: String,
    pub evaluator_build_sha256: String,
    pub verifier_id: String,
    pub verifier_build_sha256: String,
    pub execution_environment_sha256: String,
    pub problem_id: String,
    pub seed: u64,
    pub objective_specs: Vec<ParetoObjectiveSpec>,
    pub reference_point_utility: Vec<f64>,
    pub target_plan: AnytimeTargetPlan,
    pub cost_model_id: String,
    pub cost_model_sha256: String,
    pub max_objective_evaluation_attempts: u64,
    pub required_resource_meters: Vec<ResourceMeter>,
}

impl AnytimeHistoryHeader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: impl Into<String>,
        algorithm_id: impl Into<String>,
        algorithm_build_sha256: impl Into<String>,
        evaluator_id: impl Into<String>,
        evaluator_build_sha256: impl Into<String>,
        verifier_id: impl Into<String>,
        verifier_build_sha256: impl Into<String>,
        execution_environment_sha256: impl Into<String>,
        problem_id: impl Into<String>,
        seed: u64,
        objective_specs: Vec<ParetoObjectiveSpec>,
        reference_point_utility: Vec<f64>,
        target_plan: AnytimeTargetPlan,
        cost_model_id: impl Into<String>,
        cost_model_sha256: impl Into<String>,
        max_objective_evaluation_attempts: u64,
        required_resource_meters: Vec<ResourceMeter>,
    ) -> Result<Self, AnytimeError> {
        let header = Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            run_id: run_id.into(),
            algorithm_id: algorithm_id.into(),
            algorithm_build_sha256: algorithm_build_sha256.into(),
            evaluator_id: evaluator_id.into(),
            evaluator_build_sha256: evaluator_build_sha256.into(),
            verifier_id: verifier_id.into(),
            verifier_build_sha256: verifier_build_sha256.into(),
            execution_environment_sha256: execution_environment_sha256.into(),
            problem_id: problem_id.into(),
            seed,
            objective_specs,
            reference_point_utility,
            target_plan,
            cost_model_id: cost_model_id.into(),
            cost_model_sha256: cost_model_sha256.into(),
            max_objective_evaluation_attempts,
            required_resource_meters,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn validate(&self) -> Result<(), AnytimeError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(AnytimeError::new("header.schema", "unsupported anytime-history schema/version"));
        }
        for (path, value) in [
            ("run_id", self.run_id.as_str()),
            ("algorithm_id", self.algorithm_id.as_str()),
            ("evaluator_id", self.evaluator_id.as_str()),
            ("verifier_id", self.verifier_id.as_str()),
            ("problem_id", self.problem_id.as_str()),
            ("cost_model_id", self.cost_model_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(AnytimeError::new(format!("header.{path}"), "value must be non-empty"));
            }
        }
        validate_sha256(&self.algorithm_build_sha256, "header.algorithm_build_sha256")?;
        validate_sha256(&self.evaluator_build_sha256, "header.evaluator_build_sha256")?;
        validate_sha256(&self.verifier_build_sha256, "header.verifier_build_sha256")?;
        validate_sha256(&self.execution_environment_sha256, "header.execution_environment_sha256")?;
        validate_sha256(&self.cost_model_sha256, "header.cost_model_sha256")?;
        let dimensions = self.objective_specs.len();
        if !(2..=4).contains(&dimensions) {
            return Err(AnytimeError::new("header.objective_specs", "v0.1 exact-HV history supports two through four objectives"));
        }
        let mut objective_ids = BTreeSet::new();
        for (index, objective) in self.objective_specs.iter().enumerate() {
            if objective.objective_id.trim().is_empty() || !objective_ids.insert(objective.objective_id.as_str()) {
                return Err(AnytimeError::new(format!("header.objective_specs[{index}].objective_id"), "objective IDs must be non-empty and unique"));
            }
            if !objective.scale_lower.is_finite() || !objective.scale_upper.is_finite()
                || objective.scale_lower >= objective.scale_upper
                || !(objective.scale_upper - objective.scale_lower).is_finite()
            {
                return Err(AnytimeError::new(format!("header.objective_specs[{index}].scale"), "scale anchors must be finite and strictly increasing"));
            }
        }
        if self.reference_point_utility.len() != dimensions || self.reference_point_utility.iter().any(|v| !v.is_finite()) {
            return Err(AnytimeError::new("header.reference_point_utility", "reference point must have one finite coordinate per objective"));
        }
        self.target_plan.validate(dimensions)?;
        if self.max_objective_evaluation_attempts == 0 {
            return Err(AnytimeError::new("header.max_objective_evaluation_attempts", "budget must be positive"));
        }
        if self.required_resource_meters.is_empty() {
            return Err(AnytimeError::new("header.required_resource_meters", "strict resource qualification requires at least one meter"));
        }
        let mut meters = BTreeSet::new();
        for (index, meter) in self.required_resource_meters.iter().enumerate() {
            if !meters.insert(*meter) {
                return Err(AnytimeError::new(format!("header.required_resource_meters[{index}]"), "required meters must be unique"));
            }
        }
        Ok(())
    }

    pub fn header_sha256(&self) -> Result<String, AnytimeError> {
        self.validate()?;
        digest_serializable(self)
    }

    pub fn target_plan_sha256(&self) -> Result<String, AnytimeError> {
        self.target_plan.plan_sha256()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AnytimeAction {
    CandidateEvaluation {
        evaluation_id: String,
        candidate_id: String,
        attempt: u32,
        decision_sha256: String,
        outcome: EvaluationOutcome,
        objective_values: Option<Vec<f64>>,
        feasible: Option<bool>,
        constraint_violations: Vec<String>,
        evidence_sha256: Option<String>,
        failure_reason: Option<String>,
    },
    CandidateVerification {
        candidate_id: String,
        evidence_sha256: String,
        level: EvidenceLevel,
        verifier_receipt_sha256: String,
    },
    CandidateVerificationRejected {
        candidate_id: String,
        evidence_sha256: String,
        verifier_receipt_sha256: String,
        rejection_reason: String,
    },
    AuxiliaryAction {
        action_id: String,
        action_kind: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeEventDraft {
    /// Monotonic milliseconds relative to run start, not wall-clock timestamps.
    pub started_elapsed_millis: u64,
    pub completed_elapsed_millis: u64,
    pub resources: ResourceCost,
    pub action: AnytimeAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeEvent {
    pub ordinal: u64,
    pub started_elapsed_millis: u64,
    pub completed_elapsed_millis: u64,
    pub resources: ResourceCost,
    pub action: AnytimeAction,
    pub previous_event_sha256: String,
    pub event_sha256: String,
}

#[derive(Serialize)]
struct EventDigestPayload<'a> {
    header_sha256: &'a str,
    ordinal: u64,
    started_elapsed_millis: u64,
    completed_elapsed_millis: u64,
    resources: &'a ResourceCost,
    action: &'a AnytimeAction,
    previous_event_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeHistory {
    pub header: AnytimeHistoryHeader,
    pub header_sha256: String,
    pub events: Vec<AnytimeEvent>,
}

impl AnytimeHistory {
    pub fn new(header: AnytimeHistoryHeader) -> Result<Self, AnytimeError> {
        let header_sha256 = header.header_sha256()?;
        Ok(Self { header, header_sha256, events: Vec::new() })
    }

    pub fn append(&mut self, draft: AnytimeEventDraft) -> Result<String, AnytimeError> {
        self.validate()?;
        if draft.completed_elapsed_millis < draft.started_elapsed_millis {
            return Err(AnytimeError::new("event.time", "completion precedes event start"));
        }
        let ordinal = u64::try_from(self.events.len()).ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| AnytimeError::new("event.ordinal", "event ordinal overflowed u64"))?;
        let previous_event_sha256 = self.events.last()
            .map(|event| event.event_sha256.clone())
            .unwrap_or_else(|| self.header_sha256.clone());
        let event_sha256 = event_digest(
            &self.header_sha256, ordinal, draft.started_elapsed_millis,
            draft.completed_elapsed_millis, &draft.resources, &draft.action,
            &previous_event_sha256,
        )?;
        self.events.push(AnytimeEvent {
            ordinal,
            started_elapsed_millis: draft.started_elapsed_millis,
            completed_elapsed_millis: draft.completed_elapsed_millis,
            resources: draft.resources,
            action: draft.action,
            previous_event_sha256,
            event_sha256: event_sha256.clone(),
        });
        if let Err(error) = self.validate() {
            self.events.pop();
            return Err(error);
        }
        Ok(event_sha256)
    }

    pub fn validate(&self) -> Result<(), AnytimeError> {
        self.header.validate()?;
        if self.header_sha256 != self.header.header_sha256()? {
            return Err(AnytimeError::new("history.header_sha256", "header digest mismatch"));
        }

        let mut previous_digest = self.header_sha256.clone();
        let mut previous_completion = 0u64;
        let mut evaluation_ids = BTreeSet::new();
        let mut auxiliary_ids = BTreeSet::new();
        let mut attempts = BTreeMap::<String, u32>::new();
        let mut decisions = BTreeMap::<String, String>::new();
        let mut decision_owners = BTreeMap::<String, String>::new();
        let mut successful = BTreeMap::<String, CandidateState>::new();
        let mut objective_attempts = 0u64;

        for (index, event) in self.events.iter().enumerate() {
            let path = format!("history.events[{index}]");
            let expected_ordinal = u64::try_from(index).ok()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| AnytimeError::new(&path, "event count overflowed u64"))?;
            if event.ordinal != expected_ordinal {
                return Err(AnytimeError::new(format!("{path}.ordinal"), "ordinals must be contiguous and one-based"));
            }
            if event.started_elapsed_millis < previous_completion
                || event.completed_elapsed_millis < event.started_elapsed_millis
            {
                return Err(AnytimeError::new(format!("{path}.time"), "v0.1 sequential events must be monotonic and non-overlapping"));
            }
            if event.previous_event_sha256 != previous_digest {
                return Err(AnytimeError::new(format!("{path}.previous_event_sha256"), "event-chain predecessor mismatch"));
            }
            let expected_digest = event_digest(
                &self.header_sha256, event.ordinal, event.started_elapsed_millis,
                event.completed_elapsed_millis, &event.resources, &event.action,
                &event.previous_event_sha256,
            )?;
            if event.event_sha256 != expected_digest {
                return Err(AnytimeError::new(format!("{path}.event_sha256"), "event digest mismatch"));
            }

            match &event.action {
                AnytimeAction::CandidateEvaluation {
                    evaluation_id, candidate_id, attempt, decision_sha256, outcome,
                    objective_values, feasible, constraint_violations, evidence_sha256,
                    failure_reason,
                } => {
                    objective_attempts = objective_attempts.checked_add(1)
                        .ok_or_else(|| AnytimeError::new(&path, "objective-attempt counter overflow"))?;
                    if objective_attempts > self.header.max_objective_evaluation_attempts {
                        return Err(AnytimeError::new(&path, "objective-evaluation attempt budget exceeded"));
                    }
                    if evaluation_id.trim().is_empty() || !evaluation_ids.insert(evaluation_id.as_str()) {
                        return Err(AnytimeError::new(format!("{path}.evaluation_id"), "evaluation IDs must be non-empty and unique"));
                    }
                    if candidate_id.trim().is_empty() {
                        return Err(AnytimeError::new(format!("{path}.candidate_id"), "candidate ID must be non-empty"));
                    }
                    validate_sha256(decision_sha256, &format!("{path}.decision_sha256"))?;
                    let expected_attempt = attempts.get(candidate_id).copied().unwrap_or(0)
                        .checked_add(1)
                        .ok_or_else(|| AnytimeError::new(format!("{path}.attempt"), "attempt counter overflow"))?;
                    if *attempt != expected_attempt {
                        return Err(AnytimeError::new(format!("{path}.attempt"), "attempts must be contiguous and begin at one"));
                    }
                    attempts.insert(candidate_id.clone(), *attempt);
                    if decisions.get(candidate_id).is_some_and(|previous| previous != decision_sha256) {
                        return Err(AnytimeError::new(format!("{path}.decision_sha256"), "decision digest changed across retries"));
                    }
                    if decision_owners.get(decision_sha256).is_some_and(|owner| owner != candidate_id) {
                        return Err(AnytimeError::new(format!("{path}.decision_sha256"), "identical decision bytes must share one canonical candidate ID"));
                    }
                    decision_owners.entry(decision_sha256.clone()).or_insert_with(|| candidate_id.clone());
                    decisions.entry(candidate_id.clone()).or_insert_with(|| decision_sha256.clone());
                    if successful.contains_key(candidate_id) {
                        return Err(AnytimeError::new(format!("{path}.candidate_id"), "candidate can be admitted successfully only once in v0.1"));
                    }

                    match outcome {
                        EvaluationOutcome::Succeeded => {
                            let values = objective_values.as_ref().ok_or_else(|| AnytimeError::new(format!("{path}.objective_values"), "success requires objective values"))?;
                            if values.len() != self.header.objective_specs.len() || values.iter().any(|v| !v.is_finite()) {
                                return Err(AnytimeError::new(format!("{path}.objective_values"), "objective vector has wrong dimension or non-finite values"));
                            }
                            let is_feasible = feasible.ok_or_else(|| AnytimeError::new(format!("{path}.feasible"), "success requires an explicit feasibility result"))?;
                            let evidence = evidence_sha256.as_ref().ok_or_else(|| AnytimeError::new(format!("{path}.evidence_sha256"), "success requires evidence digest"))?;
                            validate_sha256(evidence, &format!("{path}.evidence_sha256"))?;
                            if (*is_feasible && !constraint_violations.is_empty())
                                || (!*is_feasible && (constraint_violations.is_empty() || constraint_violations.iter().any(|v| v.trim().is_empty())))
                            {
                                return Err(AnytimeError::new(format!("{path}.constraint_violations"), "feasibility and explicit constraint violations disagree"));
                            }
                            if failure_reason.is_some() {
                                return Err(AnytimeError::new(format!("{path}.failure_reason"), "successful event cannot carry failure reason"));
                            }
                            successful.insert(candidate_id.clone(), CandidateState {
                                candidate: ParetoCandidate {
                                    candidate_id: candidate_id.clone(),
                                    objective_values: values.clone(),
                                    feasible: *is_feasible,
                                    constraint_violations: constraint_violations.clone(),
                                    evidence_sha256: evidence.clone(),
                                },
                                decision_sha256: decision_sha256.clone(),
                                evidence_level: EvidenceLevel::Recorded,
                            });
                        }
                        _ => {
                            if objective_values.is_some() || feasible.is_some() || !constraint_violations.is_empty() || evidence_sha256.is_some() {
                                return Err(AnytimeError::new(&path, "failed/invalid/timed-out evaluation cannot enter the objective archive"));
                            }
                            if failure_reason.as_ref().is_none_or(|reason| reason.trim().is_empty()) {
                                return Err(AnytimeError::new(format!("{path}.failure_reason"), "failed attempt requires a reason"));
                            }
                        }
                    }
                }
                AnytimeAction::CandidateVerification { candidate_id, evidence_sha256, level, verifier_receipt_sha256 } => {
                    if candidate_id.trim().is_empty() {
                        return Err(AnytimeError::new(format!("{path}.candidate_id"), "verification candidate ID must be non-empty"));
                    }
                    validate_sha256(evidence_sha256, &format!("{path}.evidence_sha256"))?;
                    validate_sha256(verifier_receipt_sha256, &format!("{path}.verifier_receipt_sha256"))?;
                    if !matches!(level, EvidenceLevel::ReplayVerified | EvidenceLevel::ArtifactVerified | EvidenceLevel::HeldOutVerified | EvidenceLevel::Confirmatory) {
                        return Err(AnytimeError::new(format!("{path}.level"), "verification must advance beyond Recorded"));
                    }
                    let state = successful.get_mut(candidate_id).ok_or_else(|| AnytimeError::new(format!("{path}.candidate_id"), "verification must follow successful evaluation"))?;
                    if &state.candidate.evidence_sha256 != evidence_sha256 {
                        return Err(AnytimeError::new(format!("{path}.evidence_sha256"), "verifier receipt refers to different evidence digest"));
                    }
                    if level.rank() <= state.evidence_level.rank() {
                        return Err(AnytimeError::new(format!("{path}.level"), "verification level must strictly advance"));
                    }
                    state.evidence_level = *level;
                }
                AnytimeAction::CandidateVerificationRejected { candidate_id, evidence_sha256, verifier_receipt_sha256, rejection_reason } => {
                    validate_sha256(evidence_sha256, &format!("{path}.evidence_sha256"))?;
                    validate_sha256(verifier_receipt_sha256, &format!("{path}.verifier_receipt_sha256"))?;
                    if rejection_reason.trim().is_empty() {
                        return Err(AnytimeError::new(format!("{path}.rejection_reason"), "verifier rejection requires a reason"));
                    }
                    let state = successful.get(candidate_id).ok_or_else(|| AnytimeError::new(format!("{path}.candidate_id"), "rejection must follow successful candidate evaluation"))?;
                    if &state.candidate.evidence_sha256 != evidence_sha256 {
                        return Err(AnytimeError::new(format!("{path}.evidence_sha256"), "rejection receipt refers to different evidence digest"));
                    }
                }
                AnytimeAction::AuxiliaryAction { action_id, action_kind } => {
                    if action_id.trim().is_empty() || !auxiliary_ids.insert(action_id.as_str()) {
                        return Err(AnytimeError::new(format!("{path}.action_id"), "auxiliary action IDs must be non-empty and unique"));
                    }
                    if action_kind.trim().is_empty() {
                        return Err(AnytimeError::new(format!("{path}.action_kind"), "action kind must be non-empty"));
                    }
                }
            }

            previous_completion = event.completed_elapsed_millis;
            previous_digest = event.event_sha256.clone();
        }
        Ok(())
    }

    pub fn history_tip_sha256(&self) -> String {
        self.events.last().map(|event| event.event_sha256.clone())
            .unwrap_or_else(|| self.header_sha256.clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CandidateState {
    candidate: ParetoCandidate,
    decision_sha256: String,
    evidence_level: EvidenceLevel,
}

fn event_digest(
    header_sha256: &str, ordinal: u64, started_elapsed_millis: u64,
    completed_elapsed_millis: u64, resources: &ResourceCost, action: &AnytimeAction,
    previous_event_sha256: &str,
) -> Result<String, AnytimeError> {
    digest_serializable(&EventDigestPayload {
        header_sha256, ordinal, started_elapsed_millis, completed_elapsed_millis,
        resources, action, previous_event_sha256,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetAttainmentStatus {
    Reached,
    NotReachedByBudget,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeTargetResult {
    pub target_id: String,
    pub metric: AnytimeTargetMetric,
    pub threshold: f64,
    pub minimum_evidence_level: EvidenceLevel,
    pub status: TargetAttainmentStatus,
    pub first_event_ordinal: Option<u64>,
    pub first_objective_evaluation_attempt_count: Option<u64>,
    pub first_elapsed_wall_millis: Option<u64>,
    pub first_metric_value: Option<f64>,
    pub first_archive_sha256: Option<String>,
    pub first_resources: Option<ResourceCost>,
    pub first_resource_qualified: Option<bool>,
    pub final_metric_value: Option<f64>,
    pub final_archive_sha256: String,
    pub final_resources: ResourceCost,
    pub final_elapsed_wall_millis: u64,
    pub final_resource_qualified: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeRunReceipt {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub algorithm_id: String,
    pub header_sha256: String,
    pub target_plan_sha256: String,
    pub history_tip_sha256: String,
    pub event_count: u64,
    /// Includes failed, invalid, and timed-out objective-evaluation attempts.
    pub objective_evaluation_attempt_count: u64,
    pub successful_candidate_count: usize,
    pub total_resources: ResourceCost,
    pub elapsed_wall_millis: u64,
    pub required_resource_meters_satisfied: bool,
    pub target_results: Vec<AnytimeTargetResult>,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct ReceiptPayload<'a> {
    schema: &'a str,
    version: &'a str,
    run_id: &'a str,
    algorithm_id: &'a str,
    header_sha256: &'a str,
    target_plan_sha256: &'a str,
    history_tip_sha256: &'a str,
    event_count: u64,
    objective_evaluation_attempt_count: u64,
    successful_candidate_count: usize,
    total_resources: &'a ResourceCost,
    elapsed_wall_millis: u64,
    required_resource_meters_satisfied: bool,
    target_results: &'a [AnytimeTargetResult],
}

#[derive(Debug, Clone)]
struct MetricSnapshot {
    metric_value: Option<f64>,
    archive_sha256: String,
}

struct TargetProgress {
    target: AnytimeTarget,
    first: Option<TargetCrossing>,
    final_snapshot: MetricSnapshot,
    final_resources: ResourceCost,
    final_elapsed_wall_millis: u64,
    final_resource_qualified: bool,
}

struct TargetCrossing {
    event_ordinal: u64,
    objective_evaluation_attempt_count: u64,
    elapsed_wall_millis: u64,
    metric_value: f64,
    archive_sha256: String,
    resources: ResourceCost,
    resource_qualified: bool,
}

pub fn analyze_anytime_history(history: &AnytimeHistory) -> Result<AnytimeRunReceipt, AnytimeError> {
    history.validate()?;
    let header = &history.header;
    let empty_archive_sha = digest_serializable(&Vec::<String>::new())?;
    let mut candidates = BTreeMap::<String, CandidateState>::new();
    let mut totals = ResourceCost::zero();
    let mut objective_attempts = 0u64;
    let mut progress = header.target_plan.targets.iter().map(|target| TargetProgress {
        target: target.clone(),
        first: None,
        final_snapshot: MetricSnapshot {
            metric_value: None,
            archive_sha256: empty_archive_sha.clone(),
        },
        final_resources: ResourceCost::zero(),
        final_elapsed_wall_millis: 0,
        final_resource_qualified: false,
    }).collect::<Vec<_>>();

    for event in &history.events {
        totals.add_event(&event.resources, "receipt.total_resources")?;
        match &event.action {
            AnytimeAction::CandidateEvaluation { outcome, .. } => {
                objective_attempts = objective_attempts.checked_add(1)
                    .ok_or_else(|| AnytimeError::new("receipt.objective_evaluation_attempt_count", "counter overflow"))?;
                let _ = outcome;
            }
            _ => {}
        }
        match &event.action {
            AnytimeAction::CandidateEvaluation {
                candidate_id, outcome: EvaluationOutcome::Succeeded,
                objective_values: Some(values), feasible: Some(feasible),
                constraint_violations, evidence_sha256: Some(evidence), decision_sha256, ..
            } => {
                candidates.insert(candidate_id.clone(), CandidateState {
                    candidate: ParetoCandidate {
                        candidate_id: candidate_id.clone(),
                        objective_values: values.clone(),
                        feasible: *feasible,
                        constraint_violations: constraint_violations.clone(),
                        evidence_sha256: evidence.clone(),
                    },
                    decision_sha256: decision_sha256.clone(),
                    evidence_level: EvidenceLevel::Recorded,
                });
            }
            AnytimeAction::CandidateVerification { candidate_id, level, .. } => {
                let state = candidates.get_mut(candidate_id).ok_or_else(|| AnytimeError::new(
                    "history.candidate_verification",
                    "validated history lost a verified candidate",
                ))?;
                state.evidence_level = *level;
            }
            _ => {}
        }

        let resource_qualified = header.required_resource_meters.iter()
            .all(|meter| totals.meter_is_known(*meter));
        for item in &mut progress {
            let snapshot = metric_snapshot(
                header, &candidates, item.target.minimum_evidence_level, item.target.metric,
            )?;
            item.final_snapshot = snapshot.clone();
            item.final_resources = totals.clone();
            item.final_elapsed_wall_millis = event.completed_elapsed_millis;
            item.final_resource_qualified = resource_qualified;
            if item.first.is_none()
                && target_is_reached(item.target.metric, item.target.threshold, snapshot.metric_value)
            {
                item.first = Some(TargetCrossing {
                    event_ordinal: event.ordinal,
                    objective_evaluation_attempt_count: objective_attempts,
                    elapsed_wall_millis: event.completed_elapsed_millis,
                    metric_value: snapshot.metric_value.ok_or_else(|| AnytimeError::new(
                        "receipt.target.metric_value",
                        "a reached target must have a metric value",
                    ))?,
                    archive_sha256: snapshot.archive_sha256,
                    resources: totals.clone(),
                    resource_qualified,
                });
            }
        }
    }

    let last_elapsed = history.events.last().map_or(0, |event| event.completed_elapsed_millis);
    let final_resource_qualified = !history.events.is_empty()
        && header.required_resource_meters.iter().all(|meter| totals.meter_is_known(*meter));
    let successful_candidate_count = candidates.len();
    let mut target_results = Vec::with_capacity(progress.len());
    for item in progress {
        if let Some(first) = item.first {
            target_results.push(AnytimeTargetResult {
                target_id: item.target.target_id,
                metric: item.target.metric,
                threshold: item.target.threshold,
                minimum_evidence_level: item.target.minimum_evidence_level,
                status: TargetAttainmentStatus::Reached,
                first_event_ordinal: Some(first.event_ordinal),
                first_objective_evaluation_attempt_count: Some(first.objective_evaluation_attempt_count),
                first_elapsed_wall_millis: Some(first.elapsed_wall_millis),
                first_metric_value: Some(first.metric_value),
                first_archive_sha256: Some(first.archive_sha256),
                first_resources: Some(first.resources),
                first_resource_qualified: Some(first.resource_qualified),
                final_metric_value: item.final_snapshot.metric_value,
                final_archive_sha256: item.final_snapshot.archive_sha256,
                final_resources: item.final_resources,
                final_elapsed_wall_millis: item.final_elapsed_wall_millis,
                final_resource_qualified: item.final_resource_qualified,
            });
        } else {
            target_results.push(AnytimeTargetResult {
                target_id: item.target.target_id,
                metric: item.target.metric,
                threshold: item.target.threshold,
                minimum_evidence_level: item.target.minimum_evidence_level,
                status: TargetAttainmentStatus::NotReachedByBudget,
                first_event_ordinal: None,
                first_objective_evaluation_attempt_count: None,
                first_elapsed_wall_millis: None,
                first_metric_value: None,
                first_archive_sha256: None,
                first_resources: None,
                first_resource_qualified: None,
                final_metric_value: item.final_snapshot.metric_value,
                final_archive_sha256: item.final_snapshot.archive_sha256,
                final_resources: item.final_resources,
                final_elapsed_wall_millis: item.final_elapsed_wall_millis,
                final_resource_qualified: item.final_resource_qualified,
            });
        }
    }

    let event_count = u64::try_from(history.events.len())
        .map_err(|_| AnytimeError::new("receipt.event_count", "event count exceeds u64"))?;
    let target_plan_sha256 = header.target_plan_sha256()?;
    let history_tip_sha256 = history.history_tip_sha256();
    let payload = ReceiptPayload {
        schema: SCHEMA,
        version: VERSION,
        run_id: &header.run_id,
        algorithm_id: &header.algorithm_id,
        header_sha256: &history.header_sha256,
        target_plan_sha256: &target_plan_sha256,
        history_tip_sha256: &history_tip_sha256,
        event_count,
        objective_evaluation_attempt_count: objective_attempts,
        successful_candidate_count,
        total_resources: &totals,
        elapsed_wall_millis: last_elapsed,
        required_resource_meters_satisfied: final_resource_qualified,
        target_results: &target_results,
    };
    let receipt_sha256 = digest_serializable(&payload)?;

    Ok(AnytimeRunReceipt {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        run_id: header.run_id.clone(),
        algorithm_id: header.algorithm_id.clone(),
        header_sha256: history.header_sha256.clone(),
        target_plan_sha256,
        history_tip_sha256,
        event_count,
        objective_evaluation_attempt_count: objective_attempts,
        successful_candidate_count,
        total_resources: totals,
        elapsed_wall_millis: last_elapsed,
        required_resource_meters_satisfied: final_resource_qualified,
        target_results,
        receipt_sha256,
    })
}

fn metric_snapshot(
    header: &AnytimeHistoryHeader,
    candidates: &BTreeMap<String, CandidateState>,
    minimum_evidence_level: EvidenceLevel,
    metric: AnytimeTargetMetric,
) -> Result<MetricSnapshot, AnytimeError> {
    let eligible = candidates.values()
        .filter(|state| state.evidence_level.meets(minimum_evidence_level))
        .map(|state| state.candidate.clone())
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return Ok(MetricSnapshot {
            metric_value: match metric {
                AnytimeTargetMetric::Hypervolume => Some(0.0),
                AnytimeTargetMetric::IgdPlus | AnytimeTargetMetric::AdditiveEpsilon => None,
            },
            archive_sha256: digest_serializable(&Vec::<String>::new())?,
        });
    }
    let plan = ParetoAnalysisPlan::new(
        header.run_id.clone(),
        header.objective_specs.clone(),
        header.reference_point_utility.clone(),
        eligible,
    ).map_err(|error| AnytimeError::new("metric_snapshot.plan", error.to_string()))?;
    let analysis = analyze_pareto(&plan)
        .map_err(|error| AnytimeError::new("metric_snapshot.analysis", error.to_string()))?;
    let front = analysis.candidate_results.iter()
        .filter(|candidate| candidate.on_pareto_front)
        .map(|candidate| candidate.utility_vector.clone())
        .collect::<Vec<_>>();
    let decision_digests = candidates.values()
        .filter(|state| state.evidence_level.meets(minimum_evidence_level))
        .map(|state| (&state.candidate.candidate_id, &state.decision_sha256))
        .collect::<Vec<_>>();
    let archive_sha256 = digest_serializable(&(
        &analysis.plan_sha256, &analysis.nondominated_candidate_ids, &front, decision_digests,
    ))?;
    let metric_value = match metric {
        AnytimeTargetMetric::Hypervolume => Some(analysis.hypervolume),
        AnytimeTargetMetric::IgdPlus | AnytimeTargetMetric::AdditiveEpsilon => {
            if front.is_empty() {
                None
            } else {
                let indicators = analyze_reference_front_indicators(
                    &plan,
                    &header.target_plan.reference_front_utility,
                ).map_err(|error| AnytimeError::new("metric_snapshot.indicators", error.to_string()))?;
                Some(match metric {
                    AnytimeTargetMetric::IgdPlus => indicators.igd_plus,
                    AnytimeTargetMetric::AdditiveEpsilon => indicators.additive_epsilon,
                    AnytimeTargetMetric::Hypervolume => unreachable!(),
                })
            }
        }
    };
    Ok(MetricSnapshot { metric_value, archive_sha256 })
}

fn target_is_reached(metric: AnytimeTargetMetric, threshold: f64, value: Option<f64>) -> bool {
    match (metric, value) {
        (AnytimeTargetMetric::Hypervolume, Some(value)) => value >= threshold,
        (AnytimeTargetMetric::IgdPlus | AnytimeTargetMetric::AdditiveEpsilon, Some(value)) => value <= threshold,
        _ => false,
    }
}

pub fn verify_anytime_run_receipt(
    history: &AnytimeHistory,
    receipt: &AnytimeRunReceipt,
) -> Result<(), AnytimeError> {
    let reconstructed = analyze_anytime_history(history)?;
    if &reconstructed != receipt {
        return Err(AnytimeError::new("receipt", "receipt does not reconstruct from exact event history"));
    }
    Ok(())
}

fn dominates_utility(left: &[f64], right: &[f64]) -> bool {
    left.iter().zip(right).all(|(a, b)| a >= b)
        && left.iter().zip(right).any(|(a, b)| a > b)
}

fn same_vector(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a == b)
}

fn validate_sha256(value: &str, path: &str) -> Result<(), AnytimeError> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(AnytimeError::new(path, "expected lowercase 64-character SHA-256 digest"));
    }
    Ok(())
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, AnytimeError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| AnytimeError::new("serialization", format!("cannot serialize digest payload: {error}")))?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto_objectives::ObjectiveDirection;

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const H5: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn header() -> AnytimeHistoryHeader {
        let plan = AnytimeTargetPlan::new(vec![
            AnytimeTarget {
                target_id: "recorded-hv".to_owned(),
                metric: AnytimeTargetMetric::Hypervolume,
                threshold: 0.2,
                minimum_evidence_level: EvidenceLevel::Recorded,
            },
            AnytimeTarget {
                target_id: "heldout-hv".to_owned(),
                metric: AnytimeTargetMetric::Hypervolume,
                threshold: 0.2,
                minimum_evidence_level: EvidenceLevel::HeldOutVerified,
            },
            AnytimeTarget {
                target_id: "impossible-hv".to_owned(),
                metric: AnytimeTargetMetric::Hypervolume,
                threshold: 10.0,
                minimum_evidence_level: EvidenceLevel::Recorded,
            },
        ], vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        AnytimeHistoryHeader::new(
            "test-run", "test-optimizer", H1,
            "fixed-evaluator", H2, "independent-verifier", H3, H4,
            "zdt1", 7,
            vec![
                ParetoObjectiveSpec {
                    objective_id: "f1".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    scale_lower: 0.0,
                    scale_upper: 1.0,
                },
                ParetoObjectiveSpec {
                    objective_id: "f2".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    scale_lower: 0.0,
                    scale_upper: 1.0,
                },
            ],
            vec![-0.1, -0.1], plan, "fixed-test-cost-model", H2, 10,
            ResourceMeter::ALL.to_vec(),
        ).unwrap()
    }

    fn known_cost(cpu: u64, memory: u64) -> ResourceCost {
        ResourceCost::known(cpu, cpu, 0, memory, 0, 1, 0, 0, 0, 1, 0, cpu)
    }

    fn append_candidate(history: &mut AnytimeHistory, ordinal: u64, candidate_id: &str, evaluation_id: &str, objectives: Vec<f64>) {
        let start = (ordinal - 1) * 10;
        let decision_sha256 = match candidate_id {
            "candidate-a" | "a" => H3,
            "candidate-b" | "b" => H5,
            "dominated" => H1,
            _ => H2,
        };
        history.append(AnytimeEventDraft {
            started_elapsed_millis: start,
            completed_elapsed_millis: start + 1,
            resources: known_cost(5, 1024),
            action: AnytimeAction::CandidateEvaluation {
                evaluation_id: evaluation_id.to_owned(),
                candidate_id: candidate_id.to_owned(),
                attempt: 1,
                decision_sha256: decision_sha256.to_owned(),
                outcome: EvaluationOutcome::Succeeded,
                objective_values: Some(objectives),
                feasible: Some(true),
                constraint_violations: Vec::new(),
                evidence_sha256: Some(H4.to_owned()),
                failure_reason: None,
            },
        }).unwrap();
    }

    #[test]
    fn target_crossing_is_exact_prefix_and_verification_gated() {
        let mut history = AnytimeHistory::new(header()).unwrap();
        append_candidate(&mut history, 1, "candidate-a", "eval-a", vec![0.0, 1.0]);
        append_candidate(&mut history, 2, "candidate-b", "eval-b", vec![1.0, 0.0]);

        let unverified = analyze_anytime_history(&history).unwrap();
        let recorded = unverified.target_results.iter().find(|t| t.target_id == "recorded-hv").unwrap();
        let heldout = unverified.target_results.iter().find(|t| t.target_id == "heldout-hv").unwrap();
        assert_eq!(recorded.status, TargetAttainmentStatus::Reached);
        assert_eq!(recorded.first_event_ordinal, Some(2));
        assert_eq!(heldout.status, TargetAttainmentStatus::NotReachedByBudget);

        history.append(AnytimeEventDraft {
            started_elapsed_millis: 21,
            completed_elapsed_millis: 22,
            resources: known_cost(2, 2048),
            action: AnytimeAction::CandidateVerification {
                candidate_id: "candidate-a".to_owned(),
                evidence_sha256: H4.to_owned(),
                level: EvidenceLevel::HeldOutVerified,
                verifier_receipt_sha256: H5.to_owned(),
            },
        }).unwrap();
        history.append(AnytimeEventDraft {
            started_elapsed_millis: 22,
            completed_elapsed_millis: 23,
            resources: known_cost(2, 4096),
            action: AnytimeAction::CandidateVerification {
                candidate_id: "candidate-b".to_owned(),
                evidence_sha256: H4.to_owned(),
                level: EvidenceLevel::HeldOutVerified,
                verifier_receipt_sha256: H5.to_owned(),
            },
        }).unwrap();

        let receipt = analyze_anytime_history(&history).unwrap();
        let heldout = receipt.target_results.iter().find(|t| t.target_id == "heldout-hv").unwrap();
        assert_eq!(heldout.status, TargetAttainmentStatus::Reached);
        assert_eq!(heldout.first_event_ordinal, Some(4));
        assert_eq!(heldout.first_objective_evaluation_attempt_count, Some(2));
        assert_eq!(heldout.first_elapsed_wall_millis, Some(23));
        assert_eq!(heldout.first_resources.as_ref().unwrap().host_cpu_millis, Some(14));
        assert_eq!(heldout.first_resources.as_ref().unwrap().peak_host_memory_bytes, Some(4096));
        assert_eq!(receipt.total_resources.host_cpu_millis, Some(14));
        assert_eq!(receipt.elapsed_wall_millis, 23);
        assert!(receipt.required_resource_meters_satisfied);
        verify_anytime_run_receipt(&history, &receipt).unwrap();

        let mut tampered = receipt.clone();
        tampered.target_results[0].first_event_ordinal = Some(1);
        assert!(verify_anytime_run_receipt(&history, &tampered).is_err());
    }

    #[test]
    fn failed_attempts_are_charged_and_unknown_required_meters_disqualify() {
        let mut history = AnytimeHistory::new(header()).unwrap();
        history.append(AnytimeEventDraft {
            started_elapsed_millis: 0,
            completed_elapsed_millis: 1,
            resources: known_cost(7, 1024),
            action: AnytimeAction::CandidateEvaluation {
                evaluation_id: "failed-attempt".to_owned(),
                candidate_id: "candidate-a".to_owned(),
                attempt: 1,
                decision_sha256: H3.to_owned(),
                outcome: EvaluationOutcome::TimedOut,
                objective_values: None,
                feasible: None,
                constraint_violations: Vec::new(),
                evidence_sha256: None,
                failure_reason: Some("evaluator timeout".to_owned()),
            },
        }).unwrap();
        let mut unknown = known_cost(5, 2048);
        unknown.energy_millijoules = None;
        history.append(AnytimeEventDraft {
            started_elapsed_millis: 1,
            completed_elapsed_millis: 2,
            resources: unknown,
            action: AnytimeAction::CandidateEvaluation {
                evaluation_id: "successful-attempt".to_owned(),
                candidate_id: "candidate-a".to_owned(),
                attempt: 2,
                decision_sha256: H3.to_owned(),
                outcome: EvaluationOutcome::Succeeded,
                objective_values: Some(vec![0.0, 1.0]),
                feasible: Some(true),
                constraint_violations: Vec::new(),
                evidence_sha256: Some(H4.to_owned()),
                failure_reason: None,
            },
        }).unwrap();
        let receipt = analyze_anytime_history(&history).unwrap();
        assert_eq!(receipt.objective_evaluation_attempt_count, 2);
        assert_eq!(receipt.total_resources.host_cpu_millis, Some(12));
        assert_eq!(receipt.total_resources.energy_millijoules, None);
        assert!(!receipt.required_resource_meters_satisfied);
    }

    #[test]
    fn duplicate_decision_bytes_cannot_be_admitted_under_two_candidate_ids() {
        let mut history = AnytimeHistory::new(header()).unwrap();
        append_candidate(&mut history, 1, "candidate-a", "eval-a", vec![0.0, 1.0]);
        let duplicate = history.append(AnytimeEventDraft {
            started_elapsed_millis: 1,
            completed_elapsed_millis: 2,
            resources: known_cost(1, 1024),
            action: AnytimeAction::CandidateEvaluation {
                evaluation_id: "eval-b".to_owned(),
                candidate_id: "candidate-b".to_owned(),
                attempt: 1,
                decision_sha256: H3.to_owned(),
                outcome: EvaluationOutcome::Succeeded,
                objective_values: Some(vec![1.0, 0.0]),
                feasible: Some(true),
                constraint_violations: Vec::new(),
                evidence_sha256: Some(H5.to_owned()),
                failure_reason: None,
            },
        });
        assert!(duplicate.is_err());
        assert_eq!(history.events.len(), 1);
    }

    #[test]
    fn dominated_addition_does_not_increase_hypervolume() {
        let mut history = AnytimeHistory::new(header()).unwrap();
        append_candidate(&mut history, 1, "a", "a1", vec![0.0, 1.0]);
        append_candidate(&mut history, 2, "b", "b1", vec![1.0, 0.0]);
        let before = analyze_anytime_history(&history).unwrap().target_results[0].final_metric_value.unwrap();
        append_candidate(&mut history, 3, "dominated", "d1", vec![0.25, 1.0]);
        let after = analyze_anytime_history(&history).unwrap().target_results[0].final_metric_value.unwrap();
        assert!((before - after).abs() < 1e-12);
    }

    #[test]
    fn rejected_verification_is_charged_without_promoting_candidate() {
        let mut history = AnytimeHistory::new(header()).unwrap();
        append_candidate(&mut history, 1, "candidate-a", "eval-a", vec![0.0, 1.0]);
        let mut verifier_cost = known_cost(3, 2048);
        verifier_cost.verifier_calls = Some(1);
        history.append(AnytimeEventDraft {
            started_elapsed_millis: 1,
            completed_elapsed_millis: 2,
            resources: verifier_cost,
            action: AnytimeAction::CandidateVerificationRejected {
                candidate_id: "candidate-a".to_owned(),
                evidence_sha256: H4.to_owned(),
                verifier_receipt_sha256: H5.to_owned(),
                rejection_reason: "held-out regression detected".to_owned(),
            },
        }).unwrap();
        let receipt = analyze_anytime_history(&history).unwrap();
        let heldout = receipt.target_results.iter().find(|t| t.target_id == "heldout-hv").unwrap();
        assert_eq!(heldout.status, TargetAttainmentStatus::NotReachedByBudget);
        assert_eq!(receipt.total_resources.verifier_calls, Some(1));
        assert_eq!(receipt.event_count, 2);
        verify_anytime_run_receipt(&history, &receipt).unwrap();
    }

    #[test]
    fn overlapping_events_and_mismatched_verifier_evidence_are_rejected() {
        let mut history = AnytimeHistory::new(header()).unwrap();
        append_candidate(&mut history, 1, "candidate-a", "eval-a", vec![0.0, 1.0]);
        let overlap = history.append(AnytimeEventDraft {
            started_elapsed_millis: 0,
            completed_elapsed_millis: 20,
            resources: known_cost(1, 1),
            action: AnytimeAction::AuxiliaryAction {
                action_id: "parallel".to_owned(),
                action_kind: "unsupported-overlap".to_owned(),
            },
        });
        assert!(overlap.is_err());
        let mismatch = history.append(AnytimeEventDraft {
            started_elapsed_millis: 1,
            completed_elapsed_millis: 2,
            resources: known_cost(1, 1),
            action: AnytimeAction::CandidateVerification {
                candidate_id: "candidate-a".to_owned(),
                evidence_sha256: H5.to_owned(),
                level: EvidenceLevel::HeldOutVerified,
                verifier_receipt_sha256: H1.to_owned(),
            },
        });
        assert!(mismatch.is_err());
    }
}
