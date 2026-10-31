//! Hard-budget, multi-resource Pareto allocation for research-direction evaluations.
//!
//! Single-worker, one-step allocator. It preserves distinct hard constraints and exposes
//! non-dominated resource/capability trade-offs. It never scalarizes resource dimensions.
//! This does not yet reserve resources across concurrent workers; a parallel coordinator must
//! add atomic reservation/commit semantics before multi-worker execution.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA: &str = "symthaea.architecture-engineering.resource-pareto";
const VERSION: &str = "0.1.0";
const PLAN_DOMAIN: &[u8] = b"symthaea.resource-pareto.plan.v0.1\0";
const BUDGET_DOMAIN: &[u8] = b"symthaea.resource-pareto.budget.v0.1\0";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.resource-pareto.receipt.v0.1\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceParetoError {
    pub path: String,
    pub message: String,
}
impl ResourceParetoError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}
impl std::fmt::Display for ResourceParetoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}
impl std::error::Error for ResourceParetoError {}

/// Per-action resource amount or conservative upper estimate. None means unknown, not zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceAmounts {
    pub wall_time_ms: Option<u64>,
    pub cpu_time_ms: Option<u64>,
    pub gpu_time_ms: Option<u64>,
    pub peak_memory_bytes: Option<u64>,
    pub peak_gpu_memory_bytes: Option<u64>,
    pub energy_millijoules: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub tool_calls: Option<u64>,
    pub evaluator_calls: Option<u64>,
    pub candidate_executions: Option<u64>,
}

/// Independent hard limits. None means unconstrained; it never means that usage was zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceLimits {
    pub max_wall_time_ms: Option<u64>,
    pub max_cpu_time_ms: Option<u64>,
    pub max_gpu_time_ms: Option<u64>,
    pub max_peak_memory_bytes: Option<u64>,
    pub max_peak_gpu_memory_bytes: Option<u64>,
    pub max_energy_millijoules: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub max_tool_calls: Option<u64>,
    pub max_evaluator_calls: Option<u64>,
    pub max_candidate_executions: Option<u64>,
}
impl ResourceLimits {
    pub fn validate(&self) -> Result<(), ResourceParetoError> {
        for (name, value) in [
            ("max_wall_time_ms", self.max_wall_time_ms),
            ("max_cpu_time_ms", self.max_cpu_time_ms),
            ("max_gpu_time_ms", self.max_gpu_time_ms),
            ("max_peak_memory_bytes", self.max_peak_memory_bytes),
            ("max_peak_gpu_memory_bytes", self.max_peak_gpu_memory_bytes),
            ("max_energy_millijoules", self.max_energy_millijoules),
            ("max_input_tokens", self.max_input_tokens),
            ("max_output_tokens", self.max_output_tokens),
            ("max_tool_calls", self.max_tool_calls),
            ("max_evaluator_calls", self.max_evaluator_calls),
            ("max_candidate_executions", self.max_candidate_executions),
        ] {
            if value == Some(0) {
                return Err(ResourceParetoError::new(format!("limits.{name}"), "hard limit must be positive when present"));
            }
        }
        Ok(())
    }
}

/// Cumulative consumption and high-water memory peaks. Unknown measurements remain unknown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceLedger {
    pub wall_time_ms: Option<u64>,
    pub cpu_time_ms: Option<u64>,
    pub gpu_time_ms: Option<u64>,
    pub peak_memory_bytes: Option<u64>,
    pub peak_gpu_memory_bytes: Option<u64>,
    pub energy_millijoules: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub tool_calls: Option<u64>,
    pub evaluator_calls: Option<u64>,
    pub candidate_executions: Option<u64>,
}
impl Default for ResourceLedger {
    fn default() -> Self {
        Self {
            wall_time_ms: Some(0), cpu_time_ms: Some(0), gpu_time_ms: Some(0),
            peak_memory_bytes: Some(0), peak_gpu_memory_bytes: Some(0), energy_millijoules: Some(0),
            input_tokens: Some(0), output_tokens: Some(0), tool_calls: Some(0),
            evaluator_calls: Some(0), candidate_executions: Some(0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceParetoConfig {
    pub exploration_beta: f64,
    pub comparison_epsilon: f64,
}
impl Default for ResourceParetoConfig {
    fn default() -> Self {
        Self { exploration_beta: 1.0, comparison_epsilon: 1e-12 }
    }
}
impl ResourceParetoConfig {
    pub fn validate(&self) -> Result<(), ResourceParetoError> {
        if !self.exploration_beta.is_finite() || self.exploration_beta < 0.0 {
            return Err(ResourceParetoError::new("config.exploration_beta", "must be finite and non-negative"));
        }
        if !self.comparison_epsilon.is_finite() || self.comparison_epsilon <= 0.0 {
            return Err(ResourceParetoError::new("config.comparison_epsilon", "must be finite and positive"));
        }
        Ok(())
    }
}

/// One verifier-qualified direction's next action, with a calibrated gain uncertainty and
/// conservative upper bounds for each resource dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceActionCandidate {
    pub action_id: String,
    pub direction_id: String,
    pub predicted_gain_mean: f64,
    pub predicted_gain_standard_error: f64,
    pub prior_verified_attempts: u64,
    pub cost_upper_bound: ResourceAmounts,
}
impl ResourceActionCandidate {
    pub fn validate(&self) -> Result<(), ResourceParetoError> {
        if self.action_id.trim().is_empty() || self.direction_id.trim().is_empty() {
            return Err(ResourceParetoError::new("candidate", "action_id and direction_id are required"));
        }
        if !self.predicted_gain_mean.is_finite() || !(0.0..=1.0).contains(&self.predicted_gain_mean) {
            return Err(ResourceParetoError::new("candidate.predicted_gain_mean", "must be finite and within [0,1]"));
        }
        if !self.predicted_gain_standard_error.is_finite() || self.predicted_gain_standard_error < 0.0 {
            return Err(ResourceParetoError::new("candidate.predicted_gain_standard_error", "must be finite and non-negative"));
        }
        Ok(())
    }
    pub fn acquisition(&self, config: &ResourceParetoConfig) -> f64 {
        self.predicted_gain_mean + config.exploration_beta
            * self.predicted_gain_standard_error
            / ((self.prior_verified_attempts as f64) + 1.0).sqrt()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeasibilityStatus {
    Feasible,
    Infeasible { resource: String, reason: String },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateFeasibility {
    pub action_id: String,
    pub status: FeasibilityStatus,
    pub acquisition: f64,
    pub cost_upper_bound: ResourceAmounts,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoAction {
    pub action_id: String,
    pub direction_id: String,
    pub acquisition: f64,
    pub cost_upper_bound: ResourceAmounts,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceParetoPlan {
    pub schema: String,
    pub version: String,
    pub budget_state_sha256: String,
    pub selected_action_id: Option<String>,
    pub pareto_frontier: Vec<ParetoAction>,
    pub candidate_feasibility: Vec<CandidateFeasibility>,
    pub plan_sha256: String,
}
impl ResourceParetoPlan {
    pub fn validate_integrity(&self) -> Result<(), ResourceParetoError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(ResourceParetoError::new("plan.schema", "unsupported plan schema or version"));
        }
        validate_sha256("budget_state_sha256", &self.budget_state_sha256)?;
        validate_sha256("plan_sha256", &self.plan_sha256)?;
        let mut seen = BTreeSet::new();
        for action in &self.pareto_frontier {
            if !seen.insert(action.action_id.as_str()) || !action.acquisition.is_finite() {
                return Err(ResourceParetoError::new("plan.pareto_frontier", "duplicate action or non-finite acquisition"));
            }
        }
        if self.selected_action_id.as_ref().is_some_and(|id| {
            !self.pareto_frontier.iter().any(|action| &action.action_id == id)
        }) {
            return Err(ResourceParetoError::new("plan.selected_action_id", "selected action must be on the frontier"));
        }
        if plan_digest(self)? != self.plan_sha256 {
            return Err(ResourceParetoError::new("plan.plan_sha256", "digest does not match plan content"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceBudgetState {
    pub schema: String,
    pub version: String,
    pub limits: ResourceLimits,
    pub consumed: ResourceLedger,
    pub receipts: BTreeMap<String, ResourceActionReceipt>,
}
impl ResourceBudgetState {
    pub fn new(limits: ResourceLimits) -> Result<Self, ResourceParetoError> {
        limits.validate()?;
        Ok(Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            limits,
            consumed: ResourceLedger::default(),
            receipts: BTreeMap::new(),
        })
    }
    pub fn digest(&self) -> Result<String, ResourceParetoError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|e| ResourceParetoError::new("budget_state", e.to_string()))?;
        Ok(domain_digest(BUDGET_DOMAIN, &bytes))
    }
    pub fn validate(&self) -> Result<(), ResourceParetoError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(ResourceParetoError::new("budget_state.schema", "unsupported schema or version"));
        }
        self.limits.validate()?;
        let mut reconstructed = ResourceLedger::default();
        let mut seen_actions = BTreeSet::new();
        for (receipt_id, receipt) in &self.receipts {
            if receipt_id != &receipt.receipt_id {
                return Err(ResourceParetoError::new("receipts", "map key and receipt ID disagree"));
            }
            receipt.validate_integrity()?;
            if !seen_actions.insert(receipt.action_id.as_str()) {
                return Err(ResourceParetoError::new("receipts", "multiple receipts bind the same action ID"));
            }
            reconstructed = add_measurements(&reconstructed, &receipt.measured)?;
        }
        if reconstructed != self.consumed {
            return Err(ResourceParetoError::new("consumed", "resource ledger does not reconcile with complete receipts"));
        }
        for (name, limit, consumed) in self.limit_ledger_pairs() {
            if let Some(limit) = limit {
                let used = consumed.ok_or_else(|| ResourceParetoError::new(
                    format!("consumed.{name}"), "hard-budgeted resource consumption is unknown"))?;
                if used > limit {
                    return Err(ResourceParetoError::new(format!("consumed.{name}"), "hard budget already exceeded"));
                }
            }
        }
        Ok(())
    }
    pub fn check_upper_bound(&self, cost: &ResourceAmounts) -> Result<(), ResourceParetoError> {
        self.validate()?;
        for (name, limit, consumed, estimate, is_peak) in self.cost_limit_pairs(cost) {
            if let Some(limit) = limit {
                let estimate = estimate.ok_or_else(|| ResourceParetoError::new(
                    format!("cost_upper_bound.{name}"), "unknown estimate for hard-budgeted resource"))?;
                let used = consumed.ok_or_else(|| ResourceParetoError::new(
                    format!("consumed.{name}"), "hard-budgeted resource consumption is unknown"))?;
                let projected = if is_peak {
                    used.max(estimate)
                } else {
                    used.checked_add(estimate).ok_or_else(|| ResourceParetoError::new(
                        format!("cost_upper_bound.{name}"), "resource sum overflow"))?
                };
                if projected > limit {
                    return Err(ResourceParetoError::new(
                        format!("cost_upper_bound.{name}"),
                        format!("projected use {projected} exceeds hard limit {limit}"),
                    ));
                }
            }
        }
        Ok(())
    }
    pub fn record_receipt(&mut self, receipt: ResourceActionReceipt) -> Result<(), ResourceParetoError> {
        self.validate()?;
        receipt.validate_integrity()?;
        if self.receipts.contains_key(&receipt.receipt_id) {
            return Err(ResourceParetoError::new("receipt.receipt_id", "duplicate receipt ID"));
        }
        if self.receipts.values().any(|existing| existing.action_id == receipt.action_id) {
            return Err(ResourceParetoError::new("receipt.action_id", "action already has a resource receipt"));
        }
        self.check_upper_bound(&receipt.measured)?;
        let next = add_measurements(&self.consumed, &receipt.measured)?;
        self.receipts.insert(receipt.receipt_id.clone(), receipt);
        self.consumed = next;
        self.validate()
    }
    fn limit_ledger_pairs(&self) -> Vec<(&'static str, Option<u64>, Option<u64>)> {
        vec![
            ("wall_time_ms", self.limits.max_wall_time_ms, self.consumed.wall_time_ms),
            ("cpu_time_ms", self.limits.max_cpu_time_ms, self.consumed.cpu_time_ms),
            ("gpu_time_ms", self.limits.max_gpu_time_ms, self.consumed.gpu_time_ms),
            ("peak_memory_bytes", self.limits.max_peak_memory_bytes, self.consumed.peak_memory_bytes),
            ("peak_gpu_memory_bytes", self.limits.max_peak_gpu_memory_bytes, self.consumed.peak_gpu_memory_bytes),
            ("energy_millijoules", self.limits.max_energy_millijoules, self.consumed.energy_millijoules),
            ("input_tokens", self.limits.max_input_tokens, self.consumed.input_tokens),
            ("output_tokens", self.limits.max_output_tokens, self.consumed.output_tokens),
            ("tool_calls", self.limits.max_tool_calls, self.consumed.tool_calls),
            ("evaluator_calls", self.limits.max_evaluator_calls, self.consumed.evaluator_calls),
            ("candidate_executions", self.limits.max_candidate_executions, self.consumed.candidate_executions),
        ]
    }
    fn cost_limit_pairs<'a>(&'a self, cost: &'a ResourceAmounts) -> Vec<(&'static str, Option<u64>, Option<u64>, Option<u64>, bool)> {
        vec![
            ("wall_time_ms", self.limits.max_wall_time_ms, self.consumed.wall_time_ms, cost.wall_time_ms, false),
            ("cpu_time_ms", self.limits.max_cpu_time_ms, self.consumed.cpu_time_ms, cost.cpu_time_ms, false),
            ("gpu_time_ms", self.limits.max_gpu_time_ms, self.consumed.gpu_time_ms, cost.gpu_time_ms, false),
            ("peak_memory_bytes", self.limits.max_peak_memory_bytes, self.consumed.peak_memory_bytes, cost.peak_memory_bytes, true),
            ("peak_gpu_memory_bytes", self.limits.max_peak_gpu_memory_bytes, self.consumed.peak_gpu_memory_bytes, cost.peak_gpu_memory_bytes, true),
            ("energy_millijoules", self.limits.max_energy_millijoules, self.consumed.energy_millijoules, cost.energy_millijoules, false),
            ("input_tokens", self.limits.max_input_tokens, self.consumed.input_tokens, cost.input_tokens, false),
            ("output_tokens", self.limits.max_output_tokens, self.consumed.output_tokens, cost.output_tokens, false),
            ("tool_calls", self.limits.max_tool_calls, self.consumed.tool_calls, cost.tool_calls, false),
            ("evaluator_calls", self.limits.max_evaluator_calls, self.consumed.evaluator_calls, cost.evaluator_calls, false),
            ("candidate_executions", self.limits.max_candidate_executions, self.consumed.candidate_executions, cost.candidate_executions, false),
        ]
    }
}

/// Local receipt digest only; authenticate it against evaluator evidence separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceActionReceipt {
    pub receipt_id: String,
    pub action_id: String,
    pub measured: ResourceAmounts,
    pub receipt_sha256: String,
}
impl ResourceActionReceipt {
    pub fn new(receipt_id: &str, action_id: &str, measured: ResourceAmounts) -> Result<Self, ResourceParetoError> {
        if receipt_id.trim().is_empty() || action_id.trim().is_empty() {
            return Err(ResourceParetoError::new("receipt", "receipt_id and action_id are required"));
        }
        let mut result = Self { receipt_id: receipt_id.to_owned(), action_id: action_id.to_owned(), measured, receipt_sha256: String::new() };
        result.receipt_sha256 = receipt_digest(&result)?;
        Ok(result)
    }
    pub fn validate_integrity(&self) -> Result<(), ResourceParetoError> {
        if self.receipt_id.trim().is_empty() || self.action_id.trim().is_empty() {
            return Err(ResourceParetoError::new("receipt", "receipt_id and action_id are required"));
        }
        validate_sha256("receipt_sha256", &self.receipt_sha256)?;
        if receipt_digest(self)? != self.receipt_sha256 {
            return Err(ResourceParetoError::new("receipt.receipt_sha256", "digest mismatch"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ResourceParetoAllocator {
    pub config: ResourceParetoConfig,
    pub budget: ResourceBudgetState,
}
impl ResourceParetoAllocator {
    pub fn new(config: ResourceParetoConfig, budget: ResourceBudgetState) -> Result<Self, ResourceParetoError> {
        config.validate()?;
        budget.validate()?;
        Ok(Self { config, budget })
    }
    pub fn recommend(&self, candidates: &[ResourceActionCandidate]) -> Result<ResourceParetoPlan, ResourceParetoError> {
        self.budget.validate()?;
        self.config.validate()?;
        if candidates.is_empty() {
            return Err(ResourceParetoError::new("candidates", "at least one candidate is required"));
        }
        let mut seen = BTreeSet::new();
        let mut feasibility = Vec::with_capacity(candidates.len());
        let mut feasible = Vec::new();
        for candidate in candidates {
            candidate.validate()?;
            if !seen.insert(candidate.action_id.as_str()) {
                return Err(ResourceParetoError::new("candidates", "duplicate action ID"));
            }
            let acquisition = candidate.acquisition(&self.config);
            if !acquisition.is_finite() {
                return Err(ResourceParetoError::new("candidate.acquisition", "must be finite"));
            }
            match self.budget.check_upper_bound(&candidate.cost_upper_bound) {
                Ok(()) => {
                    let action = ParetoAction {
                        action_id: candidate.action_id.clone(),
                        direction_id: candidate.direction_id.clone(),
                        acquisition,
                        cost_upper_bound: candidate.cost_upper_bound.clone(),
                    };
                    feasibility.push(CandidateFeasibility {
                        action_id: candidate.action_id.clone(),
                        status: FeasibilityStatus::Feasible,
                        acquisition,
                        cost_upper_bound: candidate.cost_upper_bound.clone(),
                    });
                    feasible.push(action);
                }
                Err(error) => feasibility.push(CandidateFeasibility {
                    action_id: candidate.action_id.clone(),
                    status: FeasibilityStatus::Infeasible { resource: error.path, reason: error.message },
                    acquisition,
                    cost_upper_bound: candidate.cost_upper_bound.clone(),
                }),
            }
        }
        let mut frontier = Vec::new();
        for (index, action) in feasible.iter().enumerate() {
            let is_dominated = feasible.iter().enumerate().any(|(other_index, other)| {
                index != other_index && dominates(other, action, self.config.comparison_epsilon)
            });
            if !is_dominated {
                frontier.push(action.clone());
            }
        }
        frontier.sort_by(|a, b| a.action_id.cmp(&b.action_id));
        let selected_action_id = frontier.iter().max_by(|a, b| {
            a.acquisition.total_cmp(&b.acquisition).then_with(|| b.action_id.cmp(&a.action_id))
        }).map(|action| action.action_id.clone());
        let mut plan = ResourceParetoPlan {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            budget_state_sha256: self.budget.digest()?,
            selected_action_id,
            pareto_frontier: frontier,
            candidate_feasibility: feasibility,
            plan_sha256: String::new(),
        };
        plan.plan_sha256 = plan_digest(&plan)?;
        plan.validate_integrity()?;
        Ok(plan)
    }
}

fn dominates(left: &ParetoAction, right: &ParetoAction, epsilon: f64) -> bool {
    if left.acquisition + epsilon < right.acquisition {
        return false;
    }
    let mut strict = left.acquisition > right.acquisition + epsilon;
    for (a, b) in resource_pairs(&left.cost_upper_bound, &right.cost_upper_bound) {
        match (a, b) {
            (Some(a), Some(b)) if a <= b => strict |= a < b,
            (Some(_), Some(_)) => return false,
            (None, None) => {}
            _ => return false, // Unknown never becomes a free resource.
        }
    }
    strict
}
fn resource_pairs(a: &ResourceAmounts, b: &ResourceAmounts) -> Vec<(Option<u64>, Option<u64>)> {
    vec![
        (a.wall_time_ms, b.wall_time_ms), (a.cpu_time_ms, b.cpu_time_ms),
        (a.gpu_time_ms, b.gpu_time_ms), (a.peak_memory_bytes, b.peak_memory_bytes),
        (a.peak_gpu_memory_bytes, b.peak_gpu_memory_bytes), (a.energy_millijoules, b.energy_millijoules),
        (a.input_tokens, b.input_tokens), (a.output_tokens, b.output_tokens),
        (a.tool_calls, b.tool_calls), (a.evaluator_calls, b.evaluator_calls),
        (a.candidate_executions, b.candidate_executions),
    ]
}

fn add_measurements(current: &ResourceLedger, measured: &ResourceAmounts) -> Result<ResourceLedger, ResourceParetoError> {
    fn add(field: &str, current: Option<u64>, measured: Option<u64>) -> Result<Option<u64>, ResourceParetoError> {
        match (current, measured) {
            (Some(a), Some(b)) => Ok(Some(a.checked_add(b).ok_or_else(|| ResourceParetoError::new(field, "counter overflow"))?)),
            _ => Ok(None),
        }
    }
    fn peak(current: Option<u64>, measured: Option<u64>) -> Option<u64> {
        match (current, measured) { (Some(a), Some(b)) => Some(a.max(b)), _ => None }
    }
    Ok(ResourceLedger {
        wall_time_ms: add("wall_time_ms", current.wall_time_ms, measured.wall_time_ms)?,
        cpu_time_ms: add("cpu_time_ms", current.cpu_time_ms, measured.cpu_time_ms)?,
        gpu_time_ms: add("gpu_time_ms", current.gpu_time_ms, measured.gpu_time_ms)?,
        peak_memory_bytes: peak(current.peak_memory_bytes, measured.peak_memory_bytes),
        peak_gpu_memory_bytes: peak(current.peak_gpu_memory_bytes, measured.peak_gpu_memory_bytes),
        energy_millijoules: add("energy_millijoules", current.energy_millijoules, measured.energy_millijoules)?,
        input_tokens: add("input_tokens", current.input_tokens, measured.input_tokens)?,
        output_tokens: add("output_tokens", current.output_tokens, measured.output_tokens)?,
        tool_calls: add("tool_calls", current.tool_calls, measured.tool_calls)?,
        evaluator_calls: add("evaluator_calls", current.evaluator_calls, measured.evaluator_calls)?,
        candidate_executions: add("candidate_executions", current.candidate_executions, measured.candidate_executions)?,
    })
}

fn receipt_digest(receipt: &ResourceActionReceipt) -> Result<String, ResourceParetoError> {
    #[derive(Serialize)]
    struct Payload<'a> { receipt_id: &'a str, action_id: &'a str, measured: &'a ResourceAmounts }
    let bytes = serde_json::to_vec(&Payload { receipt_id: &receipt.receipt_id, action_id: &receipt.action_id, measured: &receipt.measured })
        .map_err(|e| ResourceParetoError::new("receipt", e.to_string()))?;
    Ok(domain_digest(RECEIPT_DOMAIN, &bytes))
}
fn plan_digest(plan: &ResourceParetoPlan) -> Result<String, ResourceParetoError> {
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'a str, version: &'a str, budget_state_sha256: &'a str,
        selected_action_id: &'a Option<String>, pareto_frontier: &'a [ParetoAction],
        candidate_feasibility: &'a [CandidateFeasibility],
    }
    let bytes = serde_json::to_vec(&Payload {
        schema: &plan.schema, version: &plan.version, budget_state_sha256: &plan.budget_state_sha256,
        selected_action_id: &plan.selected_action_id, pareto_frontier: &plan.pareto_frontier,
        candidate_feasibility: &plan.candidate_feasibility,
    }).map_err(|e| ResourceParetoError::new("plan", e.to_string()))?;
    Ok(domain_digest(PLAN_DOMAIN, &bytes))
}
fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
fn validate_sha256(field: &str, value: &str) -> Result<(), ResourceParetoError> {
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        Ok(())
    } else {
        Err(ResourceParetoError::new(field, "expected lowercase 64-character SHA-256 hex"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn limits() -> ResourceLimits {
        ResourceLimits {
            max_wall_time_ms: Some(1000), max_cpu_time_ms: Some(800), max_gpu_time_ms: Some(700),
            max_peak_memory_bytes: Some(2000), max_peak_gpu_memory_bytes: Some(1000),
            max_energy_millijoules: Some(10_000), max_input_tokens: Some(20_000),
            max_output_tokens: Some(10_000), max_tool_calls: Some(20),
            max_evaluator_calls: Some(30), max_candidate_executions: Some(20),
        }
    }
    fn amounts(wall: u64, cpu: u64, gpu: u64, energy: u64, memory: u64) -> ResourceAmounts {
        ResourceAmounts {
            wall_time_ms: Some(wall), cpu_time_ms: Some(cpu), gpu_time_ms: Some(gpu),
            peak_memory_bytes: Some(memory), peak_gpu_memory_bytes: Some(500),
            energy_millijoules: Some(energy), input_tokens: Some(100), output_tokens: Some(50),
            tool_calls: Some(1), evaluator_calls: Some(1), candidate_executions: Some(1),
        }
    }
    fn candidate(id: &str, gain: f64, cost: ResourceAmounts) -> ResourceActionCandidate {
        ResourceActionCandidate {
            action_id: id.into(), direction_id: format!("direction-{id}"),
            predicted_gain_mean: gain, predicted_gain_standard_error: 0.0,
            prior_verified_attempts: 3, cost_upper_bound: cost,
        }
    }

    #[test]
    fn hard_budget_rejects_unknown_estimates_and_over_limit_actions() {
        let state = ResourceBudgetState::new(limits()).unwrap();
        let allocator = ResourceParetoAllocator::new(ResourceParetoConfig::default(), state).unwrap();
        let mut unknown = amounts(100, 100, 100, 100, 100);
        unknown.energy_millijoules = None;
        let plan = allocator.recommend(&[
            candidate("unknown-energy", 0.9, unknown),
            candidate("over-wall", 0.8, amounts(1001, 100, 100, 100, 100)),
        ]).unwrap();
        assert!(plan.pareto_frontier.is_empty());
        assert!(plan.candidate_feasibility.iter().all(|item| item.status != FeasibilityStatus::Feasible));
    }

    #[test]
    fn incomparable_cpu_gpu_and_energy_actions_remain_on_frontier() {
        let state = ResourceBudgetState::new(limits()).unwrap();
        let allocator = ResourceParetoAllocator::new(ResourceParetoConfig::default(), state).unwrap();
        let plan = allocator.recommend(&[
            candidate("cpu", 0.7, amounts(200, 100, 600, 900, 500)),
            candidate("gpu", 0.8, amounts(200, 600, 100, 900, 500)),
            candidate("energy", 0.7, amounts(200, 500, 500, 100, 500)),
        ]).unwrap();
        assert_eq!(plan.pareto_frontier.len(), 3);
        assert_eq!(plan.selected_action_id.as_deref(), Some("gpu"));
        plan.validate_integrity().unwrap();
    }

    #[test]
    fn dominated_action_is_removed_without_weighted_cost() {
        let state = ResourceBudgetState::new(limits()).unwrap();
        let allocator = ResourceParetoAllocator::new(ResourceParetoConfig::default(), state).unwrap();
        let plan = allocator.recommend(&[
            candidate("better", 0.8, amounts(100, 100, 100, 100, 100)),
            candidate("worse", 0.7, amounts(200, 200, 200, 200, 200)),
        ]).unwrap();
        assert_eq!(plan.pareto_frontier.len(), 1);
        assert_eq!(plan.pareto_frontier[0].action_id, "better");
    }

    #[test]
    fn unknown_resource_under_a_hard_limit_fails_closed() {
        let mut l = limits();
        l.max_energy_millijoules = Some(1000);
        let allocator = ResourceParetoAllocator::new(ResourceParetoConfig::default(), ResourceBudgetState::new(l).unwrap()).unwrap();
        let mut cost = amounts(100, 100, 100, 100, 100);
        cost.energy_millijoules = None;
        let plan = allocator.recommend(&[candidate("unknown", 0.9, cost)]).unwrap();
        assert!(matches!(&plan.candidate_feasibility[0].status, FeasibilityStatus::Infeasible { .. }));
    }

    #[test]
    fn receipt_accounting_enforces_cumulative_and_peak_budgets() {
        let mut state = ResourceBudgetState::new(limits()).unwrap();
        state.record_receipt(ResourceActionReceipt::new("r1", "a1", amounts(500, 200, 100, 1000, 900)).unwrap()).unwrap();
        let second = ResourceActionReceipt::new("r2", "a2", amounts(501, 100, 100, 1000, 800)).unwrap();
        assert!(state.record_receipt(second).is_err());
        assert_eq!(state.consumed.wall_time_ms, Some(500));
        assert_eq!(state.receipts.len(), 1);
    }

    #[test]
    fn receipt_rejects_unknown_budgeted_measurement_and_tampering() {
        let mut state = ResourceBudgetState::new(limits()).unwrap();
        let mut measured = amounts(10, 10, 10, 10, 10);
        measured.energy_millijoules = None;
        let receipt = ResourceActionReceipt::new("r1", "a1", measured).unwrap();
        assert!(state.record_receipt(receipt).is_err());
        let mut valid = ResourceActionReceipt::new("r2", "a2", amounts(10, 10, 10, 10, 10)).unwrap();
        valid.measured.wall_time_ms = Some(11);
        assert!(state.record_receipt(valid).is_err());
    }

    #[test]
    fn unknown_unconstrained_resource_is_not_rewritten_as_zero() {
        let mut l = limits();
        l.max_energy_millijoules = None;
        let mut state = ResourceBudgetState::new(l).unwrap();
        let mut measured = amounts(10, 10, 10, 10, 10);
        measured.energy_millijoules = None;
        state.record_receipt(ResourceActionReceipt::new("r1", "a1", measured).unwrap()).unwrap();
        assert_eq!(state.consumed.energy_millijoules, None);
    }

    #[test]
    fn resource_ledger_must_reconcile_with_complete_receipts() {
        let mut state = ResourceBudgetState::new(limits()).unwrap();
        state.record_receipt(ResourceActionReceipt::new("r1", "a1", amounts(10, 20, 30, 40, 50)).unwrap()).unwrap();
        state.consumed.wall_time_ms = Some(11);
        assert!(state.validate().is_err());
    }

    #[test]
    fn plan_digest_detects_frontier_tampering() {
        let state = ResourceBudgetState::new(limits()).unwrap();
        let allocator = ResourceParetoAllocator::new(ResourceParetoConfig::default(), state).unwrap();
        let mut plan = allocator.recommend(&[candidate("a", 0.7, amounts(10, 10, 10, 10, 10))]).unwrap();
        plan.pareto_frontier[0].acquisition = 0.99;
        assert!(plan.validate_integrity().is_err());
    }

    #[test]
    fn plan_is_deterministic_and_action_ids_are_unique() {
        let state = ResourceBudgetState::new(limits()).unwrap();
        let allocator = ResourceParetoAllocator::new(ResourceParetoConfig::default(), state).unwrap();
        let candidates = vec![
            candidate("b", 0.7, amounts(10, 20, 20, 50, 50)),
            candidate("a", 0.7, amounts(20, 10, 20, 50, 50)),
        ];
        assert_eq!(allocator.recommend(&candidates).unwrap(), allocator.recommend(&candidates).unwrap());
        assert!(allocator.recommend(&[candidates[0].clone(), candidates[0].clone()]).is_err());
    }
}
