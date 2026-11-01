//! Finite power-optimal optional continuation under a separately checked null envelope.
//!
//! P0 controls validity: each [0,1] test-value process is a nonnegative
//! supermartingale under the declared null transition kernel, and every
//! continuation root is no larger than its stopped interim budget.
//!
//! P1 controls utility: dynamic programming selects the policy that maximizes
//! expected terminal rejection probability under the declared alternative.
//! The selected policy is then re-evaluated independently under P0; its exact
//! null rejection probability must remain <= alpha.
//!
//! This is a finite simple-P0/simple-P1 reference. P1 misspecification changes
//! power optimality, while P0 misspecification changes the validity guarantee.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

const PLAN_SCHEMA: &str = "symthaea.architecture-engineering.power-optimal-continuation-plan";
const PLAN_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.power-optimal-continuation-report";
const REPORT_VERSION: &str = "0.1.0";
const PROBABILITY_TOLERANCE: f64 = 1e-10;
const SUPERMARTINGALE_TOLERANCE: f64 = 1e-10;
const MAX_HISTORY_NODES: usize = 128;
const MAX_CONTINUATION_NODES_PER_HISTORY: usize = 128;
const MAX_TOTAL_CONTINUATION_NODES: usize = 8_192;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerContinuationError {
    pub path: String,
    pub message: String,
}

impl PowerContinuationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for PowerContinuationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for PowerContinuationError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerTreeNode {
    pub node_id: String,
    pub parent_node_id: Option<String>,
    /// Conditional transition probability under the declared null model P0.
    pub conditional_null_probability: f64,
    /// Conditional transition probability under the declared alternative P1.
    pub conditional_alternative_probability: f64,
    /// The [0,1] rejection probability represented by this test at this history.
    pub test_value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerContinuationProcess {
    pub history_node_id: String,
    pub nodes: Vec<PowerTreeNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerOptimalContinuationPlan {
    pub schema: String,
    pub version: String,
    pub plan_id: String,
    pub null_model_id: String,
    pub alternative_model_id: String,
    pub significance_level: f64,
    pub history_nodes: Vec<PowerTreeNode>,
    pub continuation_processes: Vec<PowerContinuationProcess>,
}

impl PowerOptimalContinuationPlan {
    pub fn new(
        plan_id: impl Into<String>,
        null_model_id: impl Into<String>,
        alternative_model_id: impl Into<String>,
        significance_level: f64,
        history_nodes: Vec<PowerTreeNode>,
        continuation_processes: Vec<PowerContinuationProcess>,
    ) -> Result<Self, PowerContinuationError> {
        Self {
            schema: PLAN_SCHEMA.into(),
            version: PLAN_VERSION.into(),
            plan_id: plan_id.into(),
            null_model_id: null_model_id.into(),
            alternative_model_id: alternative_model_id.into(),
            significance_level,
            history_nodes,
            continuation_processes,
        }.canonicalize()
    }

    pub fn canonicalize(mut self) -> Result<Self, PowerContinuationError> {
        self.history_nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        self.continuation_processes.sort_by(|a, b| {
            a.history_node_id.cmp(&b.history_node_id)
        });
        for process in &mut self.continuation_processes {
            process.nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), PowerContinuationError> {
        if self.schema != PLAN_SCHEMA || self.version != PLAN_VERSION {
            return Err(PowerContinuationError::new(
                "plan.schema",
                "unsupported power-optimal continuation schema/version",
            ));
        }
        if self.plan_id.trim().is_empty()
            || self.null_model_id.trim().is_empty()
            || self.alternative_model_id.trim().is_empty()
        {
            return Err(PowerContinuationError::new(
                "plan.identity",
                "plan, null-model and alternative-model IDs must be non-empty",
            ));
        }
        if self.null_model_id == self.alternative_model_id {
            return Err(PowerContinuationError::new(
                "plan.alternative_model_id",
                "null and alternative models must have distinct IDs",
            ));
        }
        if !self.significance_level.is_finite()
            || !(0.0..1.0).contains(&self.significance_level)
        {
            return Err(PowerContinuationError::new(
                "plan.significance_level",
                "alpha must be finite and strictly between zero and one",
            ));
        }
        if self.history_nodes.is_empty() || self.history_nodes.len() > MAX_HISTORY_NODES {
            return Err(PowerContinuationError::new(
                "plan.history_nodes",
                "history node count is empty or exceeds supported bounds",
            ));
        }
        if self.continuation_processes.len() != self.history_nodes.len() {
            return Err(PowerContinuationError::new(
                "plan.continuation_processes",
                "one continuation process is required for every possible stopping history",
            ));
        }

        let first_stage = analyze_power_tree(
            &self.history_nodes,
            "plan.history_nodes",
            MAX_HISTORY_NODES,
        )?;
        let root = self.history_nodes.iter()
            .find(|node| node.node_id == first_stage.root_id)
            .ok_or_else(|| PowerContinuationError::new(
                "plan.history_nodes",
                "validated first-stage root disappeared",
            ))?;
        if root.test_value > self.significance_level + SUPERMARTINGALE_TOLERANCE {
            return Err(PowerContinuationError::new(
                "plan.history_nodes.root.test_value",
                format!(
                    "null test root {} exceeds alpha {}",
                    root.test_value, self.significance_level
                ),
            ));
        }

        let history_ids = self.history_nodes.iter()
            .map(|node| node.node_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut process_ids = BTreeSet::new();
        let mut total_nodes = 0usize;
        for (index, process) in self.continuation_processes.iter().enumerate() {
            let path = format!("plan.continuation_processes[{index}]");
            if process.history_node_id.trim().is_empty()
                || !process_ids.insert(process.history_node_id.as_str())
                || !history_ids.contains(process.history_node_id.as_str())
            {
                return Err(PowerContinuationError::new(
                    format!("{path}.history_node_id"),
                    "continuation must uniquely reference a known stopping history",
                ));
            }
            total_nodes = total_nodes.checked_add(process.nodes.len())
                .ok_or_else(|| PowerContinuationError::new(
                    "plan.continuation_processes",
                    "continuation node count overflow",
                ))?;
            let summary = analyze_power_tree(
                &process.nodes,
                &format!("{path}.nodes"),
                MAX_CONTINUATION_NODES_PER_HISTORY,
            )?;
            let continuation_root = process.nodes.iter()
                .find(|node| node.node_id == summary.root_id)
                .ok_or_else(|| PowerContinuationError::new(
                    &path,
                    "validated continuation root disappeared",
                ))?;
            let interim_budget = self.history_nodes.iter()
                .find(|node| node.node_id == process.history_node_id)
                .map(|node| node.test_value)
                .ok_or_else(|| PowerContinuationError::new(
                    &path,
                    "validated stopping history disappeared",
                ))?;
            if continuation_root.test_value > interim_budget + SUPERMARTINGALE_TOLERANCE {
                return Err(PowerContinuationError::new(
                    format!("{path}.nodes.root.test_value"),
                    format!(
                        "continuation root {} exceeds the stopped interim budget {}",
                        continuation_root.test_value, interim_budget
                    ),
                ));
            }
        }
        if process_ids.len() != history_ids.len() {
            return Err(PowerContinuationError::new(
                "plan.continuation_processes",
                "one or more possible stopping histories have no continuation process",
            ));
        }
        if total_nodes > MAX_TOTAL_CONTINUATION_NODES {
            return Err(PowerContinuationError::new(
                "plan.continuation_processes",
                "total continuation node count exceeds supported bounds",
            ));
        }

        if first_stage.selected_policy_null_size
            > self.significance_level + SUPERMARTINGALE_TOLERANCE
        {
            return Err(PowerContinuationError::new(
                "plan.history_nodes",
                "first-stage alternative-optimal stopping policy violates null alpha",
            ));
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, PowerContinuationError> {
        let canonical = self.clone().canonicalize()?;
        digest(&canonical)
    }
}

trait PairedPowerNode {
    fn node_id(&self) -> &str;
    fn parent_node_id(&self) -> Option<&str>;
    fn null_probability(&self) -> f64;
    fn alternative_probability(&self) -> f64;
    fn test_value(&self) -> f64;
}

impl PairedPowerNode for PowerTreeNode {
    fn node_id(&self) -> &str { &self.node_id }
    fn parent_node_id(&self) -> Option<&str> { self.parent_node_id.as_deref() }
    fn null_probability(&self) -> f64 { self.conditional_null_probability }
    fn alternative_probability(&self) -> f64 { self.conditional_alternative_probability }
    fn test_value(&self) -> f64 { self.test_value }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerPolicyAction {
    StopAndApplyCurrentTest,
    ContinueObserving,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptivePowerAction {
    LaunchConditionalContinuation,
    ContinueFirstStage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerPolicyDecision {
    pub node_id: String,
    pub action: PowerPolicyAction,
    pub alternative_power_from_node: f64,
    pub null_size_under_selected_policy: f64,
}

#[derive(Debug, Clone)]
struct PowerTreeSummary {
    root_id: String,
    depths: BTreeMap<String, usize>,
    alternative_values: BTreeMap<String, f64>,
    null_policy_values: BTreeMap<String, f64>,
    actions: BTreeMap<String, PowerPolicyAction>,
    alternative_power: f64,
    selected_policy_null_size: f64,
}

fn analyze_power_tree<N: PairedPowerNode>(
    nodes: &[N],
    path: &str,
    max_nodes: usize,
) -> Result<PowerTreeSummary, PowerContinuationError> {
    if nodes.is_empty() || nodes.len() > max_nodes {
        return Err(PowerContinuationError::new(
            path,
            "tree node count is empty or exceeds supported bounds",
        ));
    }

    let mut by_id = BTreeMap::new();
    let mut roots = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let node_path = format!("{path}[{index}]");
        if node.node_id().trim().is_empty()
            || by_id.insert(node.node_id(), node).is_some()
        {
            return Err(PowerContinuationError::new(
                format!("{node_path}.node_id"),
                "node IDs must be non-empty and unique",
            ));
        }
        validate_probability(
            node.null_probability(),
            &format!("{node_path}.conditional_null_probability"),
        )?;
        validate_probability(
            node.alternative_probability(),
            &format!("{node_path}.conditional_alternative_probability"),
        )?;
        validate_probability(node.test_value(), &format!("{node_path}.test_value"))?;
        if node.parent_node_id().is_none() {
            roots.push(node);
        }
    }
    if roots.len() != 1 {
        return Err(PowerContinuationError::new(
            format!("{path}.root"),
            format!("tree must contain exactly one root; found {}", roots.len()),
        ));
    }
    let root = roots[0];
    if (root.null_probability() - 1.0).abs() > PROBABILITY_TOLERANCE
        || (root.alternative_probability() - 1.0).abs() > PROBABILITY_TOLERANCE
    {
        return Err(PowerContinuationError::new(
            format!("{path}.root"),
            "root null and alternative probabilities must both equal one",
        ));
    }

    let mut children: BTreeMap<String, Vec<&N>> = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        if let Some(parent_id) = node.parent_node_id() {
            if parent_id.trim().is_empty() || !by_id.contains_key(parent_id) {
                return Err(PowerContinuationError::new(
                    format!("{path}[{index}].parent_node_id"),
                    format!("empty or unknown parent ID {parent_id:?}"),
                ));
            }
            children.entry(parent_id.to_owned()).or_default().push(node);
        }
    }

    for (parent_id, child_nodes) in &children {
        let parent = by_id.get(parent_id.as_str()).copied().ok_or_else(|| {
            PowerContinuationError::new(path, "validated parent node disappeared")
        })?;
        let mut null_mass = 0.0;
        let mut null_mass_compensation = 0.0;
        let mut alternative_mass = 0.0;
        let mut alternative_mass_compensation = 0.0;
        let mut expected_null_test = 0.0;
        let mut test_compensation = 0.0;
        for child in child_nodes {
            kahan_add(
                &mut null_mass,
                &mut null_mass_compensation,
                child.null_probability(),
            );
            kahan_add(
                &mut alternative_mass,
                &mut alternative_mass_compensation,
                child.alternative_probability(),
            );
            kahan_add(
                &mut expected_null_test,
                &mut test_compensation,
                child.null_probability() * child.test_value(),
            );
        }
        if (null_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(PowerContinuationError::new(
                format!("{path}.{parent_id}.children.null_probability"),
                format!("null child probabilities must sum to one; found {null_mass}"),
            ));
        }
        if (alternative_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(PowerContinuationError::new(
                format!("{path}.{parent_id}.children.alternative_probability"),
                format!("alternative child probabilities must sum to one; found {alternative_mass}"),
            ));
        }
        if expected_null_test > parent.test_value() + SUPERMARTINGALE_TOLERANCE {
            return Err(PowerContinuationError::new(
                format!("{path}.{parent_id}.children"),
                format!(
                    "null conditional expected test value {expected_null_test} exceeds parent value {}",
                    parent.test_value()
                ),
            ));
        }
    }

    let mut depths = BTreeMap::new();
    for node in nodes {
        let mut current = node;
        let mut depth = 0usize;
        let mut seen = BTreeSet::new();
        seen.insert(current.node_id().to_owned());
        while let Some(parent_id) = current.parent_node_id() {
            if !seen.insert(parent_id.to_owned()) {
                return Err(PowerContinuationError::new(path, "tree contains a parent cycle"));
            }
            current = by_id.get(parent_id).copied().ok_or_else(|| {
                PowerContinuationError::new(path, "tree parent disappeared")
            })?;
            depth = depth.saturating_add(1);
        }
        if current.node_id() != root.node_id() {
            return Err(PowerContinuationError::new(
                path,
                "node does not descend from the unique root",
            ));
        }
        depths.insert(node.node_id().to_owned(), depth);
    }

    let mut bottom_up = nodes.iter().collect::<Vec<_>>();
    bottom_up.sort_by_key(|node| Reverse(depths.get(node.node_id()).copied().unwrap_or(0)));
    let mut alternative_values = BTreeMap::new();
    let mut actions = BTreeMap::new();
    for node in &bottom_up {
        let mut continue_power = 0.0;
        let mut compensation = 0.0;
        if let Some(child_nodes) = children.get(node.node_id()) {
            for child in child_nodes {
                let child_power = alternative_values.get(child.node_id()).copied().ok_or_else(|| {
                    PowerContinuationError::new(path, "bottom-up alternative-power value missing")
                })?;
                kahan_add(
                    &mut continue_power,
                    &mut compensation,
                    child.alternative_probability() * child_power,
                );
            }
        }
        let (power, action) = if continue_power > node.test_value() {
            (continue_power, PowerPolicyAction::ContinueObserving)
        } else {
            (node.test_value(), PowerPolicyAction::StopAndApplyCurrentTest)
        };
        alternative_values.insert(node.node_id().to_owned(), power);
        actions.insert(node.node_id().to_owned(), action);
    }

    // Re-evaluate the exact P1-optimal policy under P0. This is deliberately
    // separate from the P1 optimizer and is the operational size certificate.
    let mut null_policy_values = BTreeMap::new();
    for node in &bottom_up {
        let value = match actions.get(node.node_id()).copied() {
            Some(PowerPolicyAction::StopAndApplyCurrentTest) => node.test_value(),
            Some(PowerPolicyAction::ContinueObserving) => {
                let mut sum = 0.0;
                let mut compensation = 0.0;
                if let Some(child_nodes) = children.get(node.node_id()) {
                    for child in child_nodes {
                        let child_size = null_policy_values.get(child.node_id()).copied()
                            .ok_or_else(|| PowerContinuationError::new(
                                path,
                                "selected-policy null size value missing",
                            ))?;
                        kahan_add(
                            &mut sum,
                            &mut compensation,
                            child.null_probability() * child_size,
                        );
                    }
                }
                sum
            }
            None => return Err(PowerContinuationError::new(
                path,
                "selected policy action missing from reconstruction",
            )),
        };
        null_policy_values.insert(node.node_id().to_owned(), value);
    }

    let alternative_power = alternative_values.get(root.node_id()).copied()
        .ok_or_else(|| PowerContinuationError::new(path, "root alternative power missing"))?;
    let selected_policy_null_size = null_policy_values.get(root.node_id()).copied()
        .ok_or_else(|| PowerContinuationError::new(path, "root selected-policy null size missing"))?;
    if selected_policy_null_size > root.test_value() + SUPERMARTINGALE_TOLERANCE {
        return Err(PowerContinuationError::new(
            path,
            format!(
                "P1-optimal policy has P0 size {selected_policy_null_size} above root budget {}",
                root.test_value()
            ),
        ));
    }

    Ok(PowerTreeSummary {
        root_id: root.node_id().to_owned(),
        depths,
        alternative_values,
        null_policy_values,
        actions,
        alternative_power,
        selected_policy_null_size,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerContinuationProcessReport {
    pub history_node_id: String,
    pub continuation_root_budget: f64,
    pub maximum_alternative_power: f64,
    pub selected_policy_null_size: f64,
    pub unused_null_budget: f64,
    pub policy: Vec<PowerPolicyDecision>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerHistoryReport {
    pub node_id: String,
    pub alternative_power_from_history: f64,
    pub selected_policy_null_size_from_history: f64,
    pub optimal_action: AdaptivePowerAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerOptimalContinuationReport {
    pub schema: String,
    pub version: String,
    pub plan_id: String,
    pub null_model_id: String,
    pub alternative_model_id: String,
    pub plan_sha256: String,
    pub significance_level: f64,
    pub first_stage_only_power: f64,
    pub first_stage_only_policy_null_size: f64,
    pub maximum_adaptive_power: f64,
    pub selected_policy_null_size: f64,
    pub remaining_alpha_budget: f64,
    pub histories: Vec<PowerHistoryReport>,
    pub continuation_processes: Vec<PowerContinuationProcessReport>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    plan_id: &'a str,
    null_model_id: &'a str,
    alternative_model_id: &'a str,
    plan_sha256: &'a str,
    significance_level: f64,
    first_stage_only_power: f64,
    first_stage_only_policy_null_size: f64,
    maximum_adaptive_power: f64,
    selected_policy_null_size: f64,
    remaining_alpha_budget: f64,
    histories: &'a [PowerHistoryReport],
    continuation_processes: &'a [PowerContinuationProcessReport],
}

fn policy_decisions(
    summary: &PowerTreeSummary,
) -> Vec<PowerPolicyDecision> {
    summary.actions.iter().map(|(node_id, action)| {
        PowerPolicyDecision {
            node_id: node_id.clone(),
            action: *action,
            alternative_power_from_node: summary.alternative_values
                .get(node_id).copied().unwrap_or(f64::NAN),
            null_size_under_selected_policy: summary.null_policy_values
                .get(node_id).copied().unwrap_or(f64::NAN),
        }
    }).collect()
}

pub fn analyze_power_optimal_continuation(
    plan: &PowerOptimalContinuationPlan,
) -> Result<PowerOptimalContinuationReport, PowerContinuationError> {
    let plan = plan.clone().canonicalize()?;
    let first_stage = analyze_power_tree(
        &plan.history_nodes,
        "plan.history_nodes",
        MAX_HISTORY_NODES,
    )?;

    let mut continuations = BTreeMap::new();
    let mut continuation_reports = Vec::with_capacity(plan.continuation_processes.len());
    for process in &plan.continuation_processes {
        let summary = analyze_power_tree(
            &process.nodes,
            &format!("plan.continuation_processes[{}].nodes", process.history_node_id),
            MAX_CONTINUATION_NODES_PER_HISTORY,
        )?;
        let root = process.nodes.iter()
            .find(|node| node.node_id == summary.root_id)
            .ok_or_else(|| PowerContinuationError::new(
                "report.continuation_processes",
                "continuation root disappeared after validation",
            ))?;
        continuation_reports.push(PowerContinuationProcessReport {
            history_node_id: process.history_node_id.clone(),
            continuation_root_budget: root.test_value,
            maximum_alternative_power: summary.alternative_power,
            selected_policy_null_size: summary.selected_policy_null_size,
            unused_null_budget: root.test_value - summary.selected_policy_null_size,
            policy: policy_decisions(&summary),
        });
        continuations.insert(process.history_node_id.clone(), summary);
    }

    let mut children: BTreeMap<String, Vec<&PowerTreeNode>> = BTreeMap::new();
    for node in &plan.history_nodes {
        if let Some(parent_id) = node.parent_node_id.as_deref() {
            children.entry(parent_id.to_owned()).or_default().push(node);
        }
    }
    let mut combined_power = BTreeMap::new();
    let mut combined_actions: BTreeMap<String, AdaptivePowerAction> = BTreeMap::new();
    let mut bottom_up = plan.history_nodes.iter().collect::<Vec<_>>();
    bottom_up.sort_by_key(|node| Reverse(
        first_stage.depths.get(&node.node_id).copied().unwrap_or(0)
    ));
    for node in &bottom_up {
        let mut continue_power = 0.0;
        let mut compensation = 0.0;
        if let Some(child_nodes) = children.get(&node.node_id) {
            for child in child_nodes {
                let child_power = combined_power.get(&child.node_id).copied()
                    .ok_or_else(|| PowerContinuationError::new(
                        "report.combined_policy",
                        "bottom-up adaptive power value missing",
                    ))?;
                kahan_add(
                    &mut continue_power,
                    &mut compensation,
                    child.conditional_alternative_probability * child_power,
                );
            }
        }
        let launch_power = continuations.get(&node.node_id)
            .map(|summary| summary.alternative_power)
            .ok_or_else(|| PowerContinuationError::new(
                "report.combined_policy",
                format!("continuation process missing for history {}", node.node_id),
            ))?;
        let (power, action) = if continue_power > launch_power {
            (continue_power, AdaptivePowerAction::ContinueFirstStage)
        } else {
            (launch_power, AdaptivePowerAction::LaunchConditionalContinuation)
        };
        combined_power.insert(node.node_id.clone(), power);
        combined_actions.insert(node.node_id.clone(), action);
    }

    // Measure the chosen P1-optimal outer policy under P0 with a second DP.
    let mut combined_null_size = BTreeMap::new();
    for node in &bottom_up {
        let size = match combined_actions.get(&node.node_id).copied() {
            Some(AdaptivePowerAction::LaunchConditionalContinuation) => {
                continuations.get(&node.node_id)
                    .map(|summary| summary.selected_policy_null_size)
                    .ok_or_else(|| PowerContinuationError::new(
                        "report.combined_policy",
                        "selected continuation null-size result missing",
                    ))?
            }
            Some(AdaptivePowerAction::ContinueFirstStage) => {
                let mut sum = 0.0;
                let mut compensation = 0.0;
                if let Some(child_nodes) = children.get(&node.node_id) {
                    for child in child_nodes {
                        let child_size = combined_null_size.get(&child.node_id).copied()
                            .ok_or_else(|| PowerContinuationError::new(
                                "report.combined_policy",
                                "selected-policy child null-size result missing",
                            ))?;
                        kahan_add(
                            &mut sum,
                            &mut compensation,
                            child.conditional_null_probability * child_size,
                        );
                    }
                }
                sum
            }
            None => return Err(PowerContinuationError::new(
                "report.combined_policy",
                "outer policy action missing from reconstruction",
            )),
        };
        if size > node.test_value + SUPERMARTINGALE_TOLERANCE {
            return Err(PowerContinuationError::new(
                format!("report.combined_policy.{}", node.node_id),
                format!(
                    "P1-optimal outer policy has conditional P0 size {size} above interim budget {}",
                    node.test_value
                ),
            ));
        }
        combined_null_size.insert(node.node_id.clone(), size);
    }

    let root_id = &first_stage.root_id;
    let maximum_adaptive_power = combined_power.get(root_id).copied()
        .ok_or_else(|| PowerContinuationError::new(
            "report.combined_policy",
            "root alternative power missing",
        ))?;
    let selected_policy_null_size = combined_null_size.get(root_id).copied()
        .ok_or_else(|| PowerContinuationError::new(
            "report.combined_policy",
            "root selected-policy null size missing",
        ))?;
    if selected_policy_null_size > plan.significance_level + SUPERMARTINGALE_TOLERANCE {
        return Err(PowerContinuationError::new(
            "report.selected_policy_null_size",
            format!(
                "P1-optimal policy has P0 rejection size {selected_policy_null_size} above alpha {}",
                plan.significance_level
            ),
        ));
    }

    let histories = plan.history_nodes.iter().map(|node| {
        Ok(PowerHistoryReport {
            node_id: node.node_id.clone(),
            alternative_power_from_history: combined_power.get(&node.node_id).copied()
                .ok_or_else(|| PowerContinuationError::new(
                    "report.histories",
                    "history alternative power missing",
                ))?,
            selected_policy_null_size_from_history: combined_null_size.get(&node.node_id).copied()
                .ok_or_else(|| PowerContinuationError::new(
                    "report.histories",
                    "history selected-policy null size missing",
                ))?,
            optimal_action: combined_actions.get(&node.node_id).copied()
                .ok_or_else(|| PowerContinuationError::new(
                    "report.histories",
                    "history action missing",
                ))?,
        })
    }).collect::<Result<Vec<_>, PowerContinuationError>>()?;

    let plan_sha256 = plan.plan_sha256()?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        plan_id: &plan.plan_id,
        null_model_id: &plan.null_model_id,
        alternative_model_id: &plan.alternative_model_id,
        plan_sha256: &plan_sha256,
        significance_level: plan.significance_level,
        first_stage_only_power: first_stage.alternative_power,
        first_stage_only_policy_null_size: first_stage.selected_policy_null_size,
        maximum_adaptive_power,
        selected_policy_null_size,
        remaining_alpha_budget: plan.significance_level - selected_policy_null_size,
        histories: &histories,
        continuation_processes: &continuation_reports,
    };
    let report_sha256 = digest(&payload)?;
    Ok(PowerOptimalContinuationReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        plan_id: plan.plan_id,
        null_model_id: plan.null_model_id,
        alternative_model_id: plan.alternative_model_id,
        plan_sha256,
        significance_level: plan.significance_level,
        first_stage_only_power: first_stage.alternative_power,
        first_stage_only_policy_null_size: first_stage.selected_policy_null_size,
        maximum_adaptive_power,
        selected_policy_null_size,
        remaining_alpha_budget: plan.significance_level - selected_policy_null_size,
        histories,
        continuation_processes: continuation_reports,
        report_sha256,
    })
}

pub fn verify_power_optimal_continuation_report(
    plan: &PowerOptimalContinuationPlan,
    report: &PowerOptimalContinuationReport,
) -> Result<(), PowerContinuationError> {
    if analyze_power_optimal_continuation(plan)? != *report {
        return Err(PowerContinuationError::new(
            "report",
            "stored report differs from power-optimal continuation reconstruction",
        ));
    }
    Ok(())
}

fn validate_probability(value: f64, path: &str) -> Result<(), PowerContinuationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(PowerContinuationError::new(
            path,
            "probability must be finite and lie in [0, 1]",
        ));
    }
    Ok(())
}

fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
}

fn digest<T: Serialize>(value: &T) -> Result<String, PowerContinuationError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        PowerContinuationError::new("serialization", error.to_string())
    })?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(
        id: &str,
        parent: Option<&str>,
        p0: f64,
        p1: f64,
        test: f64,
    ) -> PowerTreeNode {
        PowerTreeNode {
            node_id: id.into(),
            parent_node_id: parent.map(str::to_owned),
            conditional_null_probability: p0,
            conditional_alternative_probability: p1,
            test_value: test,
        }
    }

    fn continuation(
        history_id: &str,
        root: f64,
        p0_hit: f64,
        p1_hit: f64,
        hit_test: f64,
        miss_test: f64,
    ) -> PowerContinuationProcess {
        PowerContinuationProcess {
            history_node_id: history_id.into(),
            nodes: vec![
                node("c-root", None, 1.0, 1.0, root),
                node("hit", Some("c-root"), p0_hit, p1_hit, hit_test),
                node("miss", Some("c-root"), 1.0 - p0_hit, 1.0 - p1_hit, miss_test),
            ],
        }
    }

    fn fixture() -> PowerOptimalContinuationPlan {
        PowerOptimalContinuationPlan::new(
            "power-optimal-adaptive-continuation",
            "null-p0",
            "alternative-p1",
            0.05,
            vec![
                node("root", None, 1.0, 1.0, 0.05),
                node("early", Some("root"), 0.4, 0.7, 0.08),
                node("late", Some("root"), 0.6, 0.3, 0.03),
                node("late-good", Some("late"), 0.5, 0.8, 0.03),
                node("late-bad", Some("late"), 0.5, 0.2, 0.02),
            ],
            vec![
                continuation("root", 0.01, 0.5, 0.8, 0.02, 0.0),
                continuation("early", 0.06, 0.5, 0.8, 0.1, 0.0),
                continuation("late", 0.025, 0.5, 0.7, 0.02, 0.01),
                continuation("late-good", 0.02, 0.1, 0.8, 0.1, 0.0),
                continuation("late-bad", 0.01, 0.5, 0.2, 0.01, 0.0),
            ],
        ).unwrap()
    }

    #[test]
    fn power_optimal_policy_is_separately_size_controlled_under_null() {
        let plan = fixture();
        let report = analyze_power_optimal_continuation(&plan).unwrap();
        assert!((report.first_stage_only_power - 0.065).abs() < 1e-12);
        assert!((report.first_stage_only_policy_null_size - 0.05).abs() < 1e-12);
        assert!((report.maximum_adaptive_power - 0.0758).abs() < 1e-12);
        assert!((report.selected_policy_null_size - 0.026).abs() < 1e-12);
        assert!(report.selected_policy_null_size <= report.significance_level);
        let root = report.histories.iter().find(|row| row.node_id == "root").unwrap();
        assert_eq!(root.optimal_action, AdaptivePowerAction::ContinueFirstStage);
        let late = report.histories.iter().find(|row| row.node_id == "late").unwrap();
        assert_eq!(late.optimal_action, AdaptivePowerAction::ContinueFirstStage);
        let early = report.histories.iter().find(|row| row.node_id == "early").unwrap();
        assert_eq!(early.optimal_action, AdaptivePowerAction::LaunchConditionalContinuation);
        verify_power_optimal_continuation_report(&plan, &report).unwrap();
    }

    #[test]
    fn dynamic_program_matches_all_outer_fixture_policies() {
        let enumerated = [
            0.016, // launch continuation at the root
            0.7 * 0.08 + 0.3 * 0.025, // launch at early and late histories
            0.7 * 0.08 + 0.3 * (0.8 * 0.08 + 0.2 * 0.01), // continue through late
        ];
        let maximum = enumerated.into_iter().fold(f64::NEG_INFINITY, f64::max);
        let report = analyze_power_optimal_continuation(&fixture()).unwrap();
        assert!((maximum - 0.0758).abs() < 1e-12);
        assert!((report.maximum_adaptive_power - maximum).abs() < 1e-12);
    }

    #[test]
    fn null_supermartingale_overrun_fails_closed() {
        let mut plan = fixture();
        let process = plan.continuation_processes.iter_mut()
            .find(|process| process.history_node_id == "early").unwrap();
        process.nodes.iter_mut().find(|node| node.node_id == "hit").unwrap()
            .test_value = 0.2;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn alternative_probability_mass_mismatch_fails_closed() {
        let mut plan = fixture();
        plan.history_nodes.iter_mut().find(|node| node.node_id == "early").unwrap()
            .conditional_alternative_probability = 0.8;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn selected_alternative_policy_is_audited_under_null_transitions() {
        let mut plan = fixture();
        // The optimizer's action is P1-driven, but P0 remains the only size measure.
        // A valid input still produces an independently reconstructed P0 size.
        let report = analyze_power_optimal_continuation(&plan).unwrap();
        assert!((report.selected_policy_null_size - 0.026).abs() < 1e-12);
        assert!((report.remaining_alpha_budget - 0.024).abs() < 1e-12);
        plan.alternative_model_id = "different-alternative".into();
        let changed = analyze_power_optimal_continuation(&plan).unwrap();
        assert_ne!(report.plan_sha256, changed.plan_sha256);
    }

    #[test]
    fn missing_process_and_malformed_tree_fail_closed() {
        let mut missing = fixture();
        missing.continuation_processes.pop();
        assert!(missing.validate().is_err());

        let mut malformed = fixture();
        malformed.continuation_processes.iter_mut()
            .find(|process| process.history_node_id == "late").unwrap()
            .nodes[1].conditional_null_probability = f64::NAN;
        assert!(malformed.validate().is_err());
    }

    #[test]
    fn canonical_order_and_report_tampering_are_detected() {
        let plan = fixture();
        let report = analyze_power_optimal_continuation(&plan).unwrap();
        let mut reordered = plan.clone();
        reordered.history_nodes.reverse();
        reordered.continuation_processes.reverse();
        for process in &mut reordered.continuation_processes {
            process.nodes.reverse();
        }
        assert_eq!(report, analyze_power_optimal_continuation(&reordered).unwrap());

        let mut altered = report.clone();
        altered.selected_policy_null_size += 0.01;
        assert!(verify_power_optimal_continuation_report(&plan, &altered).is_err());
    }
}
