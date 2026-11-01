//! Finite-filtration optional-stopping and conditional-continuation oracle.
//!
//! The first-stage [0,1] test process and every continuation process are checked
//! as nonnegative supermartingales on finite probability trees. At each observed
//! history, a policy may either launch its conditional continuation process or
//! continue collecting first-stage observations.
//!
//! A bottom-up dynamic program computes the maximum null rejection expectation
//! over every such adaptive policy and reconstructs a maximizing policy. This is
//! an exact finite-tree oracle for the optional-stopping / optional-continuation
//! contract; it is not a continuous-time or model-misspecification guarantee.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{cmp::Reverse, collections::{BTreeMap, BTreeSet}, error::Error, fmt};

const PLAN_SCHEMA: &str = "symthaea.architecture-engineering.finite-stopping-continuation-plan";
const PLAN_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.finite-stopping-continuation-report";
const REPORT_VERSION: &str = "0.1.0";
const PROBABILITY_TOLERANCE: f64 = 1e-10;
const SUPERMARTINGALE_TOLERANCE: f64 = 1e-10;
const MAX_HISTORY_NODES: usize = 128;
const MAX_CONTINUATION_NODES_PER_HISTORY: usize = 128;
const MAX_TOTAL_CONTINUATION_NODES: usize = 8_192;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoppingContinuationError {
    pub path: String,
    pub message: String,
}

impl StoppingContinuationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for StoppingContinuationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for StoppingContinuationError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoppingHistoryNode {
    pub node_id: String,
    pub parent_node_id: Option<String>,
    /// Conditional null probability of reaching this node from its parent.
    /// The unique root must have probability one.
    pub conditional_null_probability: f64,
    /// Interim [0,1] test value at this observed history.
    pub test_value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalContinuationNode {
    pub node_id: String,
    pub parent_node_id: Option<String>,
    pub conditional_null_probability: f64,
    /// Conditional [0,1] test process value at this continuation history.
    pub test_value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalContinuationProcess {
    /// The first-stage history at which this continuation process may be launched.
    pub history_node_id: String,
    /// A finite conditional probability tree rooted at the continuation's initial test.
    pub nodes: Vec<ConditionalContinuationNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteStoppingContinuationPlan {
    pub schema: String,
    pub version: String,
    pub plan_id: String,
    pub significance_level: f64,
    /// One finite first-stage filtration; any history can be a stopping time.
    pub history_nodes: Vec<StoppingHistoryNode>,
    /// One conditionally anytime-valid continuation process for every history.
    pub continuation_processes: Vec<ConditionalContinuationProcess>,
}

impl FiniteStoppingContinuationPlan {
    pub fn new(
        plan_id: impl Into<String>,
        significance_level: f64,
        history_nodes: Vec<StoppingHistoryNode>,
        continuation_processes: Vec<ConditionalContinuationProcess>,
    ) -> Result<Self, StoppingContinuationError> {
        Self {
            schema: PLAN_SCHEMA.into(),
            version: PLAN_VERSION.into(),
            plan_id: plan_id.into(),
            significance_level,
            history_nodes,
            continuation_processes,
        }.canonicalize()
    }

    /// Canonicalize every ID-indexed tree before validation and hashing.
    pub fn canonicalize(mut self) -> Result<Self, StoppingContinuationError> {
        self.history_nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        self.continuation_processes.sort_by(|a, b| a.history_node_id.cmp(&b.history_node_id));
        for process in &mut self.continuation_processes {
            process.nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), StoppingContinuationError> {
        if self.schema != PLAN_SCHEMA || self.version != PLAN_VERSION {
            return Err(StoppingContinuationError::new(
                "plan.schema",
                "unsupported finite stopping-continuation schema/version",
            ));
        }
        if self.plan_id.trim().is_empty() {
            return Err(StoppingContinuationError::new(
                "plan.plan_id",
                "plan ID must be non-empty",
            ));
        }
        if !self.significance_level.is_finite()
            || !(0.0..1.0).contains(&self.significance_level)
        {
            return Err(StoppingContinuationError::new(
                "plan.significance_level",
                "alpha must be finite and strictly between zero and one",
            ));
        }
        if self.history_nodes.is_empty() || self.history_nodes.len() > MAX_HISTORY_NODES {
            return Err(StoppingContinuationError::new(
                "plan.history_nodes",
                "history node count is empty or exceeds the supported ceiling",
            ));
        }
        if self.continuation_processes.len() != self.history_nodes.len() {
            return Err(StoppingContinuationError::new(
                "plan.continuation_processes",
                "exactly one conditional continuation process is required for every possible stopping history",
            ));
        }

        let history_summary = validate_probability_tree(
            &self.history_nodes,
            "plan.history_nodes",
            MAX_HISTORY_NODES,
        )?;
        let history_root = self.history_nodes.iter()
            .find(|node| node.node_id == history_summary.root_id)
            .ok_or_else(|| StoppingContinuationError::new(
                "plan.history_nodes",
                "validated root history disappeared",
            ))?;
        if history_root.test_value > self.significance_level + SUPERMARTINGALE_TOLERANCE {
            return Err(StoppingContinuationError::new(
                "plan.history_nodes.root.test_value",
                format!(
                    "initial test value {} exceeds alpha {}",
                    history_root.test_value, self.significance_level
                ),
            ));
        }

        let history_ids = self.history_nodes.iter()
            .map(|node| node.node_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut process_history_ids = BTreeSet::new();
        let mut total_continuation_nodes = 0usize;
        for (process_index, process) in self.continuation_processes.iter().enumerate() {
            let process_path = format!("plan.continuation_processes[{process_index}]");
            if process.history_node_id.trim().is_empty()
                || !process_history_ids.insert(process.history_node_id.as_str())
                || !history_ids.contains(process.history_node_id.as_str())
            {
                return Err(StoppingContinuationError::new(
                    format!("{process_path}.history_node_id"),
                    "continuation process must uniquely reference a known history node",
                ));
            }
            total_continuation_nodes = total_continuation_nodes
                .checked_add(process.nodes.len())
                .ok_or_else(|| StoppingContinuationError::new(
                    "plan.continuation_processes",
                    "total continuation node count overflow",
                ))?;
            let summary = validate_probability_tree(
                &process.nodes,
                &format!("{process_path}.nodes"),
                MAX_CONTINUATION_NODES_PER_HISTORY,
            )?;
            let continuation_root = process.nodes.iter()
                .find(|node| node.node_id == summary.root_id)
                .ok_or_else(|| StoppingContinuationError::new(
                    &process_path,
                    "validated continuation root disappeared",
                ))?;
            let interim_budget = self.history_nodes.iter()
                .find(|node| node.node_id == process.history_node_id)
                .map(|node| node.test_value)
                .ok_or_else(|| StoppingContinuationError::new(
                    &process_path,
                    "validated stopping history disappeared",
                ))?;
            if continuation_root.test_value > interim_budget + SUPERMARTINGALE_TOLERANCE {
                return Err(StoppingContinuationError::new(
                    format!("{process_path}.nodes.root.test_value"),
                    format!(
                        "continuation root budget {} exceeds stopped interim value {}",
                        continuation_root.test_value, interim_budget
                    ),
                ));
            }
            if summary.maximum_optional_stopping_value
                > continuation_root.test_value + SUPERMARTINGALE_TOLERANCE
            {
                return Err(StoppingContinuationError::new(
                    format!("{process_path}.nodes"),
                    "conditional continuation process is not anytime-valid at its root budget",
                ));
            }
        }
        if process_history_ids.len() != history_ids.len() {
            return Err(StoppingContinuationError::new(
                "plan.continuation_processes",
                "one or more possible stopping histories have no conditional continuation process",
            ));
        }
        if total_continuation_nodes > MAX_TOTAL_CONTINUATION_NODES {
            return Err(StoppingContinuationError::new(
                "plan.continuation_processes",
                "total continuation node count exceeds the supported ceiling",
            ));
        }
        if history_summary.maximum_optional_stopping_value
            > self.significance_level + SUPERMARTINGALE_TOLERANCE
        {
            return Err(StoppingContinuationError::new(
                "plan.history_nodes",
                "first-stage process exceeds alpha under an optional stopping rule",
            ));
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, StoppingContinuationError> {
        let canonical = self.clone().canonicalize()?;
        digest(&canonical)
    }
}

trait TestTreeNode {
    fn node_id(&self) -> &str;
    fn parent_node_id(&self) -> Option<&str>;
    fn conditional_probability(&self) -> f64;
    fn test_value(&self) -> f64;
}

impl TestTreeNode for StoppingHistoryNode {
    fn node_id(&self) -> &str { &self.node_id }
    fn parent_node_id(&self) -> Option<&str> { self.parent_node_id.as_deref() }
    fn conditional_probability(&self) -> f64 { self.conditional_null_probability }
    fn test_value(&self) -> f64 { self.test_value }
}

impl TestTreeNode for ConditionalContinuationNode {
    fn node_id(&self) -> &str { &self.node_id }
    fn parent_node_id(&self) -> Option<&str> { self.parent_node_id.as_deref() }
    fn conditional_probability(&self) -> f64 { self.conditional_null_probability }
    fn test_value(&self) -> f64 { self.test_value }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionalStoppingAction {
    /// Stop at this history and use the test value at the current node.
    StopAtCurrentHistory,
    /// Continue observing the next conditional outcome.
    ContinueObserving,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalStoppingDecision {
    pub node_id: String,
    pub action: OptionalStoppingAction,
    pub optimal_expected_test_value: f64,
}

#[derive(Debug, Clone)]
struct ProbabilityTreeSummary {
    root_id: String,
    path_probabilities: BTreeMap<String, f64>,
    depths: BTreeMap<String, usize>,
    leaves: Vec<String>,
    optimal_values: BTreeMap<String, f64>,
    optimal_actions: BTreeMap<String, OptionalStoppingAction>,
    maximum_optional_stopping_value: f64,
}

/// Validate a finite probability tree and compute the exact best optional-stopping
/// expectation. At each node, dynamic programming chooses between stopping now
/// and taking the conditional expectation of the best child policy.
fn validate_probability_tree<N: TestTreeNode>(
    nodes: &[N],
    path: &str,
    max_nodes: usize,
) -> Result<ProbabilityTreeSummary, StoppingContinuationError> {
    if nodes.is_empty() || nodes.len() > max_nodes {
        return Err(StoppingContinuationError::new(
            path,
            "tree node count is empty or exceeds the supported ceiling",
        ));
    }

    let mut by_id = BTreeMap::new();
    let mut roots = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let node_path = format!("{path}[{index}]");
        if node.node_id().trim().is_empty() || by_id.insert(node.node_id(), node).is_some() {
            return Err(StoppingContinuationError::new(
                format!("{node_path}.node_id"),
                "tree node IDs must be non-empty and unique",
            ));
        }
        validate_probability(
            node.conditional_probability(),
            &format!("{node_path}.conditional_null_probability"),
        )?;
        validate_probability(node.test_value(), &format!("{node_path}.test_value"))?;
        match node.parent_node_id() {
            None => roots.push(node),
            Some(parent) if parent.trim().is_empty() => {
                return Err(StoppingContinuationError::new(
                    format!("{node_path}.parent_node_id"),
                    "parent ID cannot be empty",
                ));
            }
            Some(_) => {}
        }
    }
    if roots.len() != 1 {
        return Err(StoppingContinuationError::new(
            format!("{path}.root"),
            format!("tree must contain exactly one root; found {}", roots.len()),
        ));
    }
    let root = roots[0];
    if (root.conditional_probability() - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(StoppingContinuationError::new(
            format!("{path}.root.conditional_null_probability"),
            "root probability must equal one",
        ));
    }

    let mut children: BTreeMap<String, Vec<&N>> = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        if let Some(parent_id) = node.parent_node_id() {
            if !by_id.contains_key(parent_id) {
                return Err(StoppingContinuationError::new(
                    format!("{path}[{index}].parent_node_id"),
                    format!("unknown parent node {parent_id:?}"),
                ));
            }
            children.entry(parent_id.to_owned()).or_default().push(node);
        }
    }

    for (parent_id, child_nodes) in &children {
        let parent = by_id.get(parent_id.as_str()).ok_or_else(|| {
            StoppingContinuationError::new(path, "validated parent node disappeared")
        })?;
        let mut mass = 0.0;
        let mut mass_compensation = 0.0;
        let mut expected_test = 0.0;
        let mut test_compensation = 0.0;
        for child in child_nodes {
            kahan_add(&mut mass, &mut mass_compensation, child.conditional_probability());
            kahan_add(
                &mut expected_test,
                &mut test_compensation,
                child.conditional_probability() * child.test_value(),
            );
        }
        if (mass - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(StoppingContinuationError::new(
                format!("{path}.{parent_id}.children"),
                format!("conditional child probabilities must sum to one; found {mass}"),
            ));
        }
        if expected_test > parent.test_value() + SUPERMARTINGALE_TOLERANCE {
            return Err(StoppingContinuationError::new(
                format!("{path}.{parent_id}.children"),
                format!(
                    "conditional expected child test value {expected_test} exceeds parent test value {}",
                    parent.test_value()
                ),
            ));
        }
    }

    // Propagate path probabilities from the unique root. A detached cycle or
    // unreachable node never receives a probability and therefore fails closed.
    let mut path_probabilities = BTreeMap::new();
    path_probabilities.insert(root.node_id().to_owned(), 1.0);
    for _ in 0..nodes.len() {
        let before = path_probabilities.len();
        for node in nodes {
            if let Some(parent_id) = node.parent_node_id() {
                if let Some(parent_probability) = path_probabilities.get(parent_id).copied() {
                    path_probabilities
                        .entry(node.node_id().to_owned())
                        .or_insert(parent_probability * node.conditional_probability());
                }
            }
        }
        if path_probabilities.len() == nodes.len() || path_probabilities.len() == before {
            break;
        }
    }
    if path_probabilities.len() != nodes.len() {
        return Err(StoppingContinuationError::new(
            path,
            "tree contains a disconnected component or parent cycle",
        ));
    }

    let mut depths = BTreeMap::new();
    for node in nodes {
        let mut current = node;
        let mut depth = 0usize;
        let mut seen = BTreeSet::new();
        seen.insert(current.node_id().to_owned());
        while let Some(parent_id) = current.parent_node_id() {
            if !seen.insert(parent_id.to_owned()) {
                return Err(StoppingContinuationError::new(
                    path,
                    "tree contains a parent cycle",
                ));
            }
            current = by_id.get(parent_id).copied().ok_or_else(|| {
                StoppingContinuationError::new(path, "tree parent disappeared")
            })?;
            depth = depth.saturating_add(1);
        }
        if current.node_id() != root.node_id() {
            return Err(StoppingContinuationError::new(
                path,
                "node does not descend from the unique root",
            ));
        }
        depths.insert(node.node_id().to_owned(), depth);
    }

    let mut bottom_up = nodes.iter().collect::<Vec<_>>();
    bottom_up.sort_by_key(|node| Reverse(depths.get(node.node_id()).copied().unwrap_or(0)));
    let mut optimal_values = BTreeMap::new();
    let mut optimal_actions = BTreeMap::new();
    for node in bottom_up {
        let mut continue_value = 0.0;
        let mut continue_compensation = 0.0;
        if let Some(child_nodes) = children.get(node.node_id()) {
            for child in child_nodes {
                let child_value = optimal_values.get(child.node_id()).copied().ok_or_else(|| {
                    StoppingContinuationError::new(
                        path,
                        "bottom-up optional-stopping recursion is missing a child value",
                    )
                })?;
                kahan_add(
                    &mut continue_value,
                    &mut continue_compensation,
                    child.conditional_probability() * child_value,
                );
            }
        }
        let stop_value = node.test_value();
        let (optimal_value, action) = if continue_value > stop_value {
            (continue_value, OptionalStoppingAction::ContinueObserving)
        } else {
            (stop_value, OptionalStoppingAction::StopAtCurrentHistory)
        };
        optimal_values.insert(node.node_id().to_owned(), optimal_value);
        optimal_actions.insert(node.node_id().to_owned(), action);
    }

    let maximum_optional_stopping_value = optimal_values
        .get(root.node_id())
        .copied()
        .ok_or_else(|| StoppingContinuationError::new(path, "root optional-stopping value missing"))?;
    if maximum_optional_stopping_value
        > root.test_value() + SUPERMARTINGALE_TOLERANCE
    {
        return Err(StoppingContinuationError::new(
            path,
            "dynamic program found an optional-stopping expectation above the root budget",
        ));
    }

    let mut leaves = nodes.iter()
        .filter(|node| !children.contains_key(node.node_id()))
        .map(|node| node.node_id().to_owned())
        .collect::<Vec<_>>();
    leaves.sort();
    Ok(ProbabilityTreeSummary {
        root_id: root.node_id().to_owned(),
        path_probabilities,
        depths,
        leaves,
        optimal_values,
        optimal_actions,
        maximum_optional_stopping_value,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationProcessReport {
    pub history_node_id: String,
    pub continuation_root_value: f64,
    pub maximum_conditional_optional_stopping_size: f64,
    pub terminal_leaf_size: f64,
    pub conditional_budget_slack: f64,
    pub optimal_stopping_policy: Vec<OptionalStoppingDecision>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoppingHistoryReport {
    pub node_id: String,
    pub history_probability: f64,
    pub interim_test_value: f64,
    pub first_process_maximum_stopping_value: f64,
    pub continuation_root_value: f64,
    pub continuation_maximum_optional_stopping_size: f64,
    pub continuation_terminal_leaf_size: f64,
    pub combined_policy_value: f64,
    pub optimal_action: OptionalStoppingAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteStoppingContinuationReport {
    pub schema: String,
    pub version: String,
    pub plan_id: String,
    pub plan_sha256: String,
    pub significance_level: f64,
    pub maximum_first_process_optional_stopping_size: f64,
    pub maximum_first_process_optional_stopping_policy: Vec<OptionalStoppingDecision>,
    pub maximum_adaptive_continuation_size: f64,
    pub histories: Vec<StoppingHistoryReport>,
    pub continuation_processes: Vec<ContinuationProcessReport>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    plan_id: &'a str,
    plan_sha256: &'a str,
    significance_level: f64,
    maximum_first_process_optional_stopping_size: f64,
    maximum_first_process_optional_stopping_policy: &'a [OptionalStoppingDecision],
    maximum_adaptive_continuation_size: f64,
    histories: &'a [StoppingHistoryReport],
    continuation_processes: &'a [ContinuationProcessReport],
}

pub fn analyze_finite_stopping_continuation(
    plan: &FiniteStoppingContinuationPlan,
) -> Result<FiniteStoppingContinuationReport, StoppingContinuationError> {
    let plan = plan.clone().canonicalize()?;
    let history_summary = validate_probability_tree(
        &plan.history_nodes,
        "plan.history_nodes",
        MAX_HISTORY_NODES,
    )?;

    let mut continuation_summaries = BTreeMap::new();
    let mut continuation_reports = Vec::with_capacity(plan.continuation_processes.len());
    for process in &plan.continuation_processes {
        let summary = validate_probability_tree(
            &process.nodes,
            &format!("plan.continuation_processes[{}].nodes", process.history_node_id),
            MAX_CONTINUATION_NODES_PER_HISTORY,
        )?;
        let root = process.nodes.iter()
            .find(|node| node.node_id == summary.root_id)
            .ok_or_else(|| StoppingContinuationError::new(
                "report.continuation_processes",
                "continuation root disappeared after validation",
            ))?;
        let terminal_leaf_size = compensated_sum(summary.leaves.iter().map(|leaf_id| {
            let probability = summary.path_probabilities.get(leaf_id).copied().unwrap_or(f64::NAN);
            let value = process.nodes.iter()
                .find(|node| node.node_id == *leaf_id)
                .map(|node| node.test_value)
                .unwrap_or(f64::NAN);
            probability * value
        }));
        if !terminal_leaf_size.is_finite()
            || terminal_leaf_size > summary.maximum_optional_stopping_value + SUPERMARTINGALE_TOLERANCE
        {
            return Err(StoppingContinuationError::new(
                "report.continuation_processes",
                "terminal continuation size exceeds its optional-stopping bound",
            ));
        }
        continuation_reports.push(ContinuationProcessReport {
            history_node_id: process.history_node_id.clone(),
            continuation_root_value: root.test_value,
            maximum_conditional_optional_stopping_size: summary.maximum_optional_stopping_value,
            terminal_leaf_size,
            conditional_budget_slack: root.test_value
                - summary.maximum_optional_stopping_value,
            optimal_stopping_policy: stopping_decisions(&summary),
        });
        continuation_summaries.insert(process.history_node_id.clone(), summary);
    }

    let mut children: BTreeMap<String, Vec<&StoppingHistoryNode>> = BTreeMap::new();
    for node in &plan.history_nodes {
        if let Some(parent_id) = node.parent_node_id.as_deref() {
            children.entry(parent_id.to_owned()).or_default().push(node);
        }
    }

    // Maximize the total rejection probability over every adaptive policy:
    // launch the conditional continuation now, or keep observing and choose later.
    let mut combined_values = BTreeMap::new();
    let mut combined_actions = BTreeMap::new();
    let mut bottom_up = plan.history_nodes.iter().collect::<Vec<_>>();
    bottom_up.sort_by_key(|node| Reverse(
        history_summary.depths.get(&node.node_id).copied().unwrap_or(0)
    ));
    for node in bottom_up {
        let mut continue_value = 0.0;
        let mut continue_compensation = 0.0;
        if let Some(child_nodes) = children.get(&node.node_id) {
            for child in child_nodes {
                let child_value = combined_values.get(&child.node_id).copied().ok_or_else(|| {
                    StoppingContinuationError::new(
                        "report.combined_policy",
                        "bottom-up adaptive policy recursion is missing a child value",
                    )
                })?;
                kahan_add(
                    &mut continue_value,
                    &mut continue_compensation,
                    child.conditional_null_probability * child_value,
                );
            }
        }
        let stop_value = continuation_summaries
            .get(&node.node_id)
            .map(|summary| summary.maximum_optional_stopping_value)
            .ok_or_else(|| StoppingContinuationError::new(
                "report.combined_policy",
                "conditional continuation summary missing for a history",
            ))?;
        let (value, action) = if continue_value > stop_value {
            (continue_value, OptionalStoppingAction::Continue)
        } else {
            (stop_value, OptionalStoppingAction::StopHere)
        };
        if value > node.test_value + SUPERMARTINGALE_TOLERANCE {
            return Err(StoppingContinuationError::new(
                format!("report.combined_policy.{}", node.node_id),
                format!(
                    "maximized continuation value {value} exceeds interim test budget {}",
                    node.test_value
                ),
            ));
        }
        combined_values.insert(node.node_id.clone(), value);
        combined_actions.insert(node.node_id.clone(), action);
    }

    let maximum_adaptive_continuation_size = combined_values
        .get(&history_summary.root_id)
        .copied()
        .ok_or_else(|| StoppingContinuationError::new(
            "report.combined_policy",
            "root combined-policy value missing",
        ))?;
    if maximum_adaptive_continuation_size
        > plan.significance_level + SUPERMARTINGALE_TOLERANCE
    {
        return Err(StoppingContinuationError::new(
            "report.maximum_adaptive_continuation_size",
            "maximized optional continuation policy exceeds alpha",
        ));
    }

    let mut histories = Vec::with_capacity(plan.history_nodes.len());
    for node in &plan.history_nodes {
        let continuation = continuation_summaries.get(&node.node_id)
            .ok_or_else(|| StoppingContinuationError::new(
                "report.histories",
                "continuation summary missing for a history",
            ))?;
        let process = plan.continuation_processes.iter()
            .find(|process| process.history_node_id == node.node_id)
            .ok_or_else(|| StoppingContinuationError::new(
                "report.histories",
                "continuation process missing for a history",
            ))?;
        let continuation_root = process.nodes.iter()
            .find(|candidate| candidate.node_id == continuation.root_id)
            .ok_or_else(|| StoppingContinuationError::new(
                "report.histories",
                "continuation root missing for a history",
            ))?;
        let terminal_leaf_size = compensated_sum(continuation.leaves.iter().map(|leaf_id| {
            let path_probability = continuation.path_probabilities.get(leaf_id).copied().unwrap_or(f64::NAN);
            let value = process.nodes.iter().find(|candidate| candidate.node_id == *leaf_id)
                .map(|candidate| candidate.test_value).unwrap_or(f64::NAN);
            path_probability * value
        }));
        histories.push(StoppingHistoryReport {
            node_id: node.node_id.clone(),
            history_probability: history_summary.path_probabilities
                .get(&node.node_id).copied().unwrap_or(f64::NAN),
            interim_test_value: node.test_value,
            first_process_maximum_stopping_value: history_summary
                .optimal_values.get(&node.node_id).copied().unwrap_or(f64::NAN),
            continuation_root_value: continuation_root.test_value,
            continuation_maximum_optional_stopping_size: continuation.maximum_optional_stopping_value,
            continuation_terminal_leaf_size: terminal_leaf_size,
            combined_policy_value: combined_values.get(&node.node_id).copied().unwrap_or(f64::NAN),
            optimal_action: combined_actions.get(&node.node_id).copied().unwrap_or(
                OptionalStoppingAction::StopHere
            ),
        });
    }

    if histories.iter().any(|row| {
        !row.history_probability.is_finite()
            || !row.first_process_maximum_stopping_value.is_finite()
            || !row.continuation_terminal_leaf_size.is_finite()
            || !row.combined_policy_value.is_finite()
    }) {
        return Err(StoppingContinuationError::new(
            "report.histories",
            "reconstructed history report contains a non-finite value",
        ));
    }

    let plan_sha256 = plan.plan_sha256()?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        plan_id: &plan.plan_id,
        plan_sha256: &plan_sha256,
        significance_level: plan.significance_level,
        maximum_first_process_optional_stopping_size: history_summary.maximum_optional_stopping_value,
        maximum_first_process_optional_stopping_policy: &stopping_decisions(&history_summary),
        maximum_adaptive_continuation_size,
        histories: &histories,
        continuation_processes: &continuation_reports,
    };
    let report_sha256 = digest(&payload)?;
    Ok(FiniteStoppingContinuationReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        plan_id: plan.plan_id,
        plan_sha256,
        significance_level: plan.significance_level,
        maximum_first_process_optional_stopping_size: history_summary.maximum_optional_stopping_value,
        maximum_first_process_optional_stopping_policy: stopping_decisions(&history_summary),
        maximum_adaptive_continuation_size,
        histories,
        continuation_processes: continuation_reports,
        report_sha256,
    })
}

pub fn verify_finite_stopping_continuation_report(
    plan: &FiniteStoppingContinuationPlan,
    report: &FiniteStoppingContinuationReport,
) -> Result<(), StoppingContinuationError> {
    if analyze_finite_stopping_continuation(plan)? != *report {
        return Err(StoppingContinuationError::new(
            "report",
            "stored report differs from finite stopping-time continuation reconstruction",
        ));
    }
    Ok(())
}

fn stopping_decisions(summary: &ProbabilityTreeSummary) -> Vec<OptionalStoppingDecision> {
    summary.optimal_actions.iter().map(|(node_id, action)| {
        OptionalStoppingDecision {
            node_id: node_id.clone(),
            action: *action,
            optimal_expected_test_value: summary.optimal_values
                .get(node_id)
                .copied()
                .unwrap_or(f64::NAN),
        }
    }).collect()
}

fn validate_probability(value: f64, path: &str) -> Result<(), StoppingContinuationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(StoppingContinuationError::new(
            path,
            "value must be finite and lie in [0, 1]",
        ));
    }
    Ok(())
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for value in values {
        kahan_add(&mut sum, &mut compensation, value);
    }
    sum
}

fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
}

fn digest<T: Serialize>(value: &T) -> Result<String, StoppingContinuationError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        StoppingContinuationError::new("serialization", error.to_string())
    })?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn continuation_tree(
        root_value: f64,
        yes_probability: f64,
        yes_value: f64,
        no_probability: f64,
        no_value: f64,
    ) -> ConditionalContinuationProcess {
        ConditionalContinuationProcess {
            history_node_id: String::new(),
            nodes: vec![
                ConditionalContinuationNode {
                    node_id: "c-root".into(),
                    parent_node_id: None,
                    conditional_null_probability: 1.0,
                    test_value: root_value,
                },
                ConditionalContinuationNode {
                    node_id: "no".into(),
                    parent_node_id: Some("c-root".into()),
                    conditional_null_probability: no_probability,
                    test_value: no_value,
                },
                ConditionalContinuationNode {
                    node_id: "yes".into(),
                    parent_node_id: Some("c-root".into()),
                    conditional_null_probability: yes_probability,
                    test_value: yes_value,
                },
            ],
        }
    }

    fn fixture() -> FiniteStoppingContinuationPlan {
        let history_nodes = vec![
            StoppingHistoryNode {
                node_id: "root".into(),
                parent_node_id: None,
                conditional_null_probability: 1.0,
                test_value: 0.05,
            },
            StoppingHistoryNode {
                node_id: "early".into(),
                parent_node_id: Some("root".into()),
                conditional_null_probability: 0.4,
                test_value: 0.08,
            },
            StoppingHistoryNode {
                node_id: "late".into(),
                parent_node_id: Some("root".into()),
                conditional_null_probability: 0.6,
                test_value: 0.03,
            },
            StoppingHistoryNode {
                node_id: "late-good".into(),
                parent_node_id: Some("late".into()),
                conditional_null_probability: 0.5,
                test_value: 0.03,
            },
            StoppingHistoryNode {
                node_id: "late-bad".into(),
                parent_node_id: Some("late".into()),
                conditional_null_probability: 0.5,
                test_value: 0.02,
            },
        ];
        let mut continuations = Vec::new();
        let specs = [
            ("root", 0.01, 0.5, 0.02, 0.0),
            ("early", 0.06, 0.5, 0.1, 0.0),
            ("late", 0.025, 0.5, 0.02, 0.01),
            ("late-good", 0.02, 0.1, 0.1, 0.0),
            ("late-bad", 0.01, 0.5, 0.01, 0.0),
        ];
        for (history_id, root_value, yes_probability, yes_value, no_value) in specs {
            let no_probability = 1.0 - yes_probability;
            let mut process = continuation_tree(
                root_value,
                yes_probability,
                yes_value,
                no_probability,
                no_value,
            );
            process.history_node_id = history_id.into();
            continuations.push(process);
        }
        FiniteStoppingContinuationPlan::new(
            "finite-adaptive-stopping-continuation",
            0.05,
            history_nodes,
            continuations,
        ).unwrap()
    }

    #[test]
    fn dynamic_program_certifies_every_optional_stopping_policy() {
        let plan = fixture();
        let report = analyze_finite_stopping_continuation(&plan).unwrap();
        assert!((report.maximum_first_process_optional_stopping_size - 0.05).abs() < 1e-12);
        assert!((report.maximum_adaptive_continuation_size - 0.039).abs() < 1e-12);
        assert!(report.maximum_adaptive_continuation_size <= report.significance_level);
        let root = report.histories.iter().find(|row| row.node_id == "root").unwrap();
        assert_eq!(root.optimal_action, OptionalStoppingAction::ContinueObserving);
        let late = report.histories.iter().find(|row| row.node_id == "late").unwrap();
        assert_eq!(late.optimal_action, OptionalStoppingAction::StopAtCurrentHistory);
        assert!((late.combined_policy_value - 0.025).abs() < 1e-12);
        verify_finite_stopping_continuation_report(&plan, &report).unwrap();
    }

    #[test]
    fn dynamic_program_matches_exhaustive_policy_enumeration() {
        // The only nontrivial policy choices are at root and late:
        // stop at root = 0.01; continue then stop at late = 0.039;
        // continue through late = 0.033.
        let enumerated = [
            0.01,
            0.4 * 0.06 + 0.6 * 0.025,
            0.4 * 0.06 + 0.6 * (0.5 * 0.02 + 0.5 * 0.01),
        ];
        let report = analyze_finite_stopping_continuation(&fixture()).unwrap();
        let exhaustive_max = enumerated.into_iter().fold(f64::NEG_INFINITY, f64::max);
        assert!((report.maximum_adaptive_continuation_size - exhaustive_max).abs() < 1e-12);
    }

    #[test]
    fn first_stage_supermartingale_overrun_fails_closed() {
        let mut plan = fixture();
        plan.history_nodes.iter_mut()
            .find(|node| node.node_id == "late-good").unwrap()
            .test_value = 0.06;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn conditional_budget_and_continuation_supermartingale_overruns_fail_closed() {
        let mut root_overrun = fixture();
        let process = root_overrun.continuation_processes.iter_mut()
            .find(|process| process.history_node_id == "early").unwrap();
        process.nodes.iter_mut().find(|node| node.node_id == "c-root").unwrap()
            .test_value = 0.09;
        assert!(root_overrun.validate().is_err());

        let mut conditional_overrun = fixture();
        let process = conditional_overrun.continuation_processes.iter_mut()
            .find(|process| process.history_node_id == "early").unwrap();
        process.nodes.iter_mut().find(|node| node.node_id == "yes").unwrap()
            .test_value = 0.2;
        assert!(conditional_overrun.validate().is_err());
    }

    #[test]
    fn missing_processes_cycles_and_invalid_mass_fail_closed() {
        let mut missing = fixture();
        missing.continuation_processes.pop();
        assert!(missing.validate().is_err());

        let mut cycle = fixture();
        cycle.history_nodes.iter_mut()
            .find(|node| node.node_id == "root").unwrap()
            .parent_node_id = Some("late".into());
        assert!(cycle.validate().is_err());

        let mut invalid_mass = fixture();
        invalid_mass.history_nodes.iter_mut()
            .find(|node| node.node_id == "early").unwrap()
            .conditional_null_probability = 0.3;
        assert!(invalid_mass.validate().is_err());
    }

    #[test]
    fn canonical_order_and_report_tampering_are_detected() {
        let plan = fixture();
        let report = analyze_finite_stopping_continuation(&plan).unwrap();
        let mut reordered = plan.clone();
        reordered.history_nodes.reverse();
        reordered.continuation_processes.reverse();
        for process in &mut reordered.continuation_processes {
            process.nodes.reverse();
        }
        assert_eq!(report, analyze_finite_stopping_continuation(&reordered).unwrap());

        let mut altered = report.clone();
        altered.maximum_adaptive_continuation_size += 0.01;
        assert!(verify_finite_stopping_continuation_report(&plan, &altered).is_err());
    }
}
