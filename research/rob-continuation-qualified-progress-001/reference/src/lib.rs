//! Descriptive evaluator for safe accepted subgoal progress that preserves
//! mandatory mission continuation. Future-option and viability evidence are
//! consumed from upstream receipts; this crate does not compute physical viability.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgoalOutcome {
    SafeAccepted,
    AcceptedButUnsafe,
    Rejected,
    Aborted,
    Held,
    Unknown,
    NotAttempted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuationDisposition {
    Qualified,
    Insufficient,
    Unknown,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgoalDefinition {
    pub subgoal_id: String,
    pub mandatory: bool,
}

/// Directed prerequisite edge: predecessor must precede successor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SubgoalEdge {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionManifest {
    pub mission_id: String,
    pub graph_digest: String,
    pub mission_order: Vec<String>,
    pub subgoals: Vec<SubgoalDefinition>,
    pub edges: Vec<SubgoalEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionArmManifest {
    pub arm_id: String,
    pub policy_digest: String,
    pub harness_digest: String,
    pub task_manifest_digest: String,
    pub robot_configuration_digest: String,
    pub environment_manifest_digest: String,
    pub resource_budget_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuationReceipt {
    pub receipt_digest: String,
    pub terminal_state_digest: String,
    pub required_mandatory_successor_ids: Vec<String>,
    pub feasible_mandatory_successor_ids: Vec<String>,
    pub continuation_witness_digest: String,
    pub disposition: ContinuationDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgoalEvaluation {
    pub subgoal_id: String,
    pub outcome: SubgoalOutcome,
    pub acceptance_receipt_digest: Option<String>,
    /// Stable identity of the accepted physical output; one output cannot be credited twice.
    pub accepted_output_digest: Option<String>,
    pub safety_receipt_digest: Option<String>,
    /// Number of failed/repeated attempts before this terminal outcome.
    pub rework_attempt_count: u16,
    /// Upstream receipt documenting the rework attempts, when non-zero.
    pub rework_receipt_digest: Option<String>,
    pub physical_transition_receipt_digest: Option<String>,
    pub predecessor_state_digest: Option<String>,
    pub result_state_digest: Option<String>,
    pub option_ledger_receipt_digest: Option<String>,
    pub option_predecessor_state_digest: Option<String>,
    pub option_result_state_digest: Option<String>,
    pub continuation: Option<ContinuationReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionEvaluationInput {
    pub mission: MissionManifest,
    pub arm: MissionArmManifest,
    pub evaluations: Vec<SubgoalEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgoalProgressReport {
    pub subgoal_id: String,
    pub outcome: SubgoalOutcome,
    pub independently_accepted: bool,
    pub safe_accepted: bool,
    /// Number of separately recorded rework attempts for this subgoal.
    pub rework_attempt_count: u16,
    pub physical_option_join_valid: bool,
    pub continuation_qualified: bool,
    pub continuation_unknown: bool,
    pub mandatory_successor_count: u16,
    pub feasible_mandatory_successor_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionProgressReport {
    pub mission_id: String,
    pub arm_id: String,
    pub scheduled_subgoals: u16,
    pub native_completed_subgoals: u16,
    pub independently_accepted_subgoals: u16,
    pub safe_accepted_subgoals: u16,
    pub accepted_but_unsafe_subgoals: u16,
    pub continuation_qualified_subgoals: u16,
    pub unknown_subgoals: u16,
    /// Sum of separately receipted rework attempts across scheduled subgoals.
    pub total_rework_attempts: u64,
    pub longest_safe_accepted_prefix: u16,
    pub longest_continuation_qualified_prefix: u16,
    pub mandatory_safe_completion: bool,
    pub mandatory_continuation_completion: bool,
    pub subgoals: Vec<SubgoalProgressReport>,
    pub actuation_authorized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressError {
    EmptyIdentity(&'static str),
    DuplicateSubgoal(String),
    InvalidMissionOrder,
    InvalidEdge,
    DependencyCycle,
    DuplicateEvaluation(String),
    MissingEvaluation(String),
    ReceiptReused(String),
    DuplicateAcceptedOutput(String),
    InvalidReworkEvidence(String),
    IncompleteAcceptedOutcome(String),
    PhysicalOptionStateMismatch(String),
    ContinuationStateMismatch(String),
    InvalidContinuationCoverage(String),
    RejectedContinuation(String),
    ArithmeticOverflow,
}

fn require_nonempty(name: &'static str, value: &str) -> Result<(), ProgressError> {
    if value.trim().is_empty() {
        return Err(ProgressError::EmptyIdentity(name));
    }
    Ok(())
}

fn validate_mission(mission: &MissionManifest) -> Result<BTreeMap<String, bool>, ProgressError> {
    require_nonempty("mission_id", &mission.mission_id)?;
    require_nonempty("graph_digest", &mission.graph_digest)?;
    if mission.subgoals.is_empty() || mission.mission_order.len() != mission.subgoals.len() {
        return Err(ProgressError::InvalidMissionOrder);
    }
    let mut nodes = BTreeMap::new();
    for node in &mission.subgoals {
        require_nonempty("subgoal_id", &node.subgoal_id)?;
        if nodes.insert(node.subgoal_id.clone(), node.mandatory).is_some() {
            return Err(ProgressError::DuplicateSubgoal(node.subgoal_id.clone()));
        }
    }
    let ordered = mission.mission_order.iter().cloned().collect::<BTreeSet<_>>();
    if ordered.len() != mission.mission_order.len()
        || ordered.len() != nodes.len()
        || ordered.iter().any(|id| !nodes.contains_key(id))
    {
        return Err(ProgressError::InvalidMissionOrder);
    }

    let order_index = mission
        .mission_order
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut seen_edges = BTreeSet::new();
    let mut indegree = nodes.keys().map(|id| (id.clone(), 0_usize)).collect::<BTreeMap<_, _>>();
    let mut successors: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for edge in &mission.edges {
        if edge.from == edge.to
            || !nodes.contains_key(&edge.from)
            || !nodes.contains_key(&edge.to)
            || !seen_edges.insert(edge.clone())
        {
            return Err(ProgressError::InvalidEdge);
        }
        *indegree.get_mut(&edge.to).ok_or(ProgressError::InvalidEdge)? += 1;
        successors.entry(edge.from.clone()).or_default().push(edge.to.clone());
    }

    let mut ready = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect::<BTreeSet<_>>();
    let mut visited = 0_usize;
    while let Some(node) = ready.pop_first() {
        visited += 1;
        for successor in successors.get(&node).into_iter().flatten() {
            let degree = indegree.get_mut(successor).ok_or(ProgressError::InvalidEdge)?;
            *degree = degree.checked_sub(1).ok_or(ProgressError::InvalidEdge)?;
            if *degree == 0 {
                ready.insert(successor.clone());
            }
        }
    }
    if visited != nodes.len() {
        return Err(ProgressError::DependencyCycle);
    }
    for edge in &mission.edges {
        if order_index[edge.from.as_str()] >= order_index[edge.to.as_str()] {
            return Err(ProgressError::InvalidMissionOrder);
        }
    }
    Ok(nodes)
}

fn mandatory_descendants(
    mission: &MissionManifest,
    start: &str,
    mandatory: &BTreeMap<String, bool>,
    completed_before: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut successors: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in &mission.edges {
        successors.entry(&edge.from).or_default().push(&edge.to);
    }
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::from([start]);
    while let Some(node) = queue.pop_front() {
        for successor in successors.get(node).into_iter().flatten() {
            if visited.insert((*successor).to_owned()) {
                queue.push_back(successor);
            }
        }
    }
    visited
        .into_iter()
        .filter(|id| mandatory.get(id) == Some(&true) && !completed_before.contains(id))
        .collect()
}

fn validate_receipt_uniqueness(evaluations: &[SubgoalEvaluation]) -> Result<(), ProgressError> {
    let mut seen = BTreeSet::new();
    let mut accepted_outputs = BTreeSet::new();
    for evaluation in evaluations {
        let continuation = evaluation.continuation.as_ref();
        if matches!(
            evaluation.outcome,
            SubgoalOutcome::SafeAccepted | SubgoalOutcome::AcceptedButUnsafe
        ) {
            let output = evaluation
                .accepted_output_digest
                .as_deref()
                .ok_or_else(|| ProgressError::IncompleteAcceptedOutcome(
                    evaluation.subgoal_id.clone(),
                ))?;
            if !accepted_outputs.insert(output.to_owned()) {
                return Err(ProgressError::DuplicateAcceptedOutput(output.to_owned()));
            }
        }
        for receipt in [
            evaluation.acceptance_receipt_digest.as_deref(),
            evaluation.rework_receipt_digest.as_deref(),
            evaluation.safety_receipt_digest.as_deref(),
            evaluation.physical_transition_receipt_digest.as_deref(),
            evaluation.option_ledger_receipt_digest.as_deref(),
            continuation.map(|r| r.receipt_digest.as_str()),
            continuation.map(|r| r.continuation_witness_digest.as_str()),
        ]
        .into_iter()
        .flatten()
        {
            require_nonempty("evidence_receipt_digest", receipt)?;
            if !seen.insert(receipt.to_owned()) {
                return Err(ProgressError::ReceiptReused(receipt.to_owned()));
            }
        }
    }
    Ok(())
}

fn validate_transition(evaluation: &SubgoalEvaluation) -> Result<bool, ProgressError> {
    let accepted = matches!(
        evaluation.outcome,
        SubgoalOutcome::SafeAccepted | SubgoalOutcome::AcceptedButUnsafe
    );
    match (evaluation.rework_attempt_count, evaluation.rework_receipt_digest.as_deref()) {
        (0, None) => {}
        (0, Some(_)) => return Err(ProgressError::InvalidReworkEvidence(evaluation.subgoal_id.clone())),
        (_, Some(receipt)) if !receipt.trim().is_empty() => {}
        _ => return Err(ProgressError::InvalidReworkEvidence(evaluation.subgoal_id.clone())),
    }
    if !accepted {
        if evaluation.continuation.is_some() {
            return Err(ProgressError::IncompleteAcceptedOutcome(evaluation.subgoal_id.clone()));
        }
        return Ok(false);
    }
    let required = [
        evaluation.acceptance_receipt_digest.as_deref(),
        evaluation.accepted_output_digest.as_deref(),
        evaluation.safety_receipt_digest.as_deref(),
        evaluation.physical_transition_receipt_digest.as_deref(),
        evaluation.predecessor_state_digest.as_deref(),
        evaluation.result_state_digest.as_deref(),
        evaluation.option_ledger_receipt_digest.as_deref(),
        evaluation.option_predecessor_state_digest.as_deref(),
        evaluation.option_result_state_digest.as_deref(),
    ];
    if required
        .into_iter()
        .any(|value| value.is_none_or(|value| value.trim().is_empty()))
    {
        return Err(ProgressError::IncompleteAcceptedOutcome(evaluation.subgoal_id.clone()));
    }
    if evaluation.predecessor_state_digest != evaluation.option_predecessor_state_digest
        || evaluation.result_state_digest != evaluation.option_result_state_digest
    {
        return Err(ProgressError::PhysicalOptionStateMismatch(evaluation.subgoal_id.clone()));
    }
    Ok(true)
}

fn validate_continuation(
    evaluation: &SubgoalEvaluation,
    required_mandatory: &BTreeSet<String>,
) -> Result<(bool, bool, u16), ProgressError> {
    let Some(receipt) = &evaluation.continuation else {
        return Ok((false, true, 0));
    };
    require_nonempty("continuation_receipt_digest", &receipt.receipt_digest)?;
    require_nonempty("terminal_state_digest", &receipt.terminal_state_digest)?;
    require_nonempty("continuation_witness_digest", &receipt.continuation_witness_digest)?;
    if evaluation.result_state_digest.as_deref() != Some(receipt.terminal_state_digest.as_str()) {
        return Err(ProgressError::ContinuationStateMismatch(evaluation.subgoal_id.clone()));
    }
    let required = receipt.required_mandatory_successor_ids.iter().cloned().collect::<BTreeSet<_>>();
    let feasible = receipt.feasible_mandatory_successor_ids.iter().cloned().collect::<BTreeSet<_>>();
    if required.len() != receipt.required_mandatory_successor_ids.len()
        || feasible.len() != receipt.feasible_mandatory_successor_ids.len()
        || required != *required_mandatory
        || !feasible.is_subset(&required)
    {
        return Err(ProgressError::InvalidContinuationCoverage(evaluation.subgoal_id.clone()));
    }
    let feasible_count = u16::try_from(feasible.len()).map_err(|_| ProgressError::ArithmeticOverflow)?;
    match receipt.disposition {
        ContinuationDisposition::Qualified if feasible == required => Ok((true, false, feasible_count)),
        ContinuationDisposition::Qualified => {
            Err(ProgressError::InvalidContinuationCoverage(evaluation.subgoal_id.clone()))
        }
        ContinuationDisposition::Insufficient if feasible != required => Ok((false, false, feasible_count)),
        ContinuationDisposition::Insufficient => {
            Err(ProgressError::InvalidContinuationCoverage(evaluation.subgoal_id.clone()))
        }
        ContinuationDisposition::Unknown => Ok((false, true, feasible_count)),
        ContinuationDisposition::Rejected => {
            Err(ProgressError::RejectedContinuation(evaluation.subgoal_id.clone()))
        }
    }
}

pub fn evaluate_mission_progress(
    input: &MissionEvaluationInput,
) -> Result<MissionProgressReport, ProgressError> {
    for (name, value) in [
        ("arm_id", input.arm.arm_id.as_str()),
        ("policy_digest", input.arm.policy_digest.as_str()),
        ("harness_digest", input.arm.harness_digest.as_str()),
        ("task_manifest_digest", input.arm.task_manifest_digest.as_str()),
        ("robot_configuration_digest", input.arm.robot_configuration_digest.as_str()),
        ("environment_manifest_digest", input.arm.environment_manifest_digest.as_str()),
        ("resource_budget_digest", input.arm.resource_budget_digest.as_str()),
    ] {
        require_nonempty(name, value)?;
    }
    let mandatory = validate_mission(&input.mission)?;
    let mut evaluations = BTreeMap::new();
    for evaluation in &input.evaluations {
        if !mandatory.contains_key(&evaluation.subgoal_id) {
            return Err(ProgressError::DuplicateEvaluation(evaluation.subgoal_id.clone()));
        }
        if evaluations.insert(evaluation.subgoal_id.clone(), evaluation).is_some() {
            return Err(ProgressError::DuplicateEvaluation(evaluation.subgoal_id.clone()));
        }
    }
    for id in &input.mission.mission_order {
        if !evaluations.contains_key(id) {
            return Err(ProgressError::MissingEvaluation(id.clone()));
        }
    }
    if evaluations.len() != mandatory.len() {
        return Err(ProgressError::InvalidMissionOrder);
    }
    validate_receipt_uniqueness(&input.evaluations)?;

    let mut safe_accepted = BTreeSet::new();
    let mut native_completed = 0_u16;
    let mut safe_count = 0_u16;
    let mut unsafe_count = 0_u16;
    let mut unknown_count = 0_u16;
    for evaluation in &input.evaluations {
        validate_transition(evaluation)?;
        match evaluation.outcome {
            SubgoalOutcome::SafeAccepted => {
                native_completed += 1;
                safe_count += 1;
                safe_accepted.insert(evaluation.subgoal_id.clone());
            }
            SubgoalOutcome::AcceptedButUnsafe => {
                native_completed += 1;
                unsafe_count += 1;
            }
            SubgoalOutcome::Unknown => unknown_count += 1,
            SubgoalOutcome::Rejected | SubgoalOutcome::Aborted | SubgoalOutcome::Held | SubgoalOutcome::NotAttempted => {}
        }
    }

    let mut reports = Vec::with_capacity(input.mission.mission_order.len());
    let mut continuation_count = 0_u16;
    let mut continuation_unknown_count = 0_u16;
    let mut total_rework_attempts = 0_u64;
    let mut safe_prefix = 0_u16;
    let mut continuation_prefix = 0_u16;
    let mut safe_prefix_open = true;
    let mut continuation_prefix_open = true;
    let mut completed_before = BTreeSet::new();

    for id in &input.mission.mission_order {
        let evaluation = evaluations[id];
        let safe = evaluation.outcome == SubgoalOutcome::SafeAccepted;
        let required =
            mandatory_descendants(&input.mission, id, &mandatory, &completed_before);
        total_rework_attempts = total_rework_attempts
            .checked_add(u64::from(evaluation.rework_attempt_count))
            .ok_or(ProgressError::ArithmeticOverflow)?;
        let (qualified, continuation_unknown, feasible_count) = if safe {
            validate_continuation(evaluation, &required)?
        } else {
            if evaluation.continuation.is_some() {
                return Err(ProgressError::IncompleteAcceptedOutcome(id.clone()));
            }
            (false, false, 0)
        };
        if qualified {
            continuation_count += 1;
        }
        if continuation_unknown || (safe && evaluation.continuation.is_none()) {
            continuation_unknown_count += 1;
        }
        if safe_prefix_open && safe {
            safe_prefix += 1;
        } else {
            safe_prefix_open = false;
        }
        if continuation_prefix_open && qualified {
            continuation_prefix += 1;
        } else {
            continuation_prefix_open = false;
        }
        reports.push(SubgoalProgressReport {
            subgoal_id: id.clone(),
            outcome: evaluation.outcome,
            independently_accepted: matches!(evaluation.outcome, SubgoalOutcome::SafeAccepted | SubgoalOutcome::AcceptedButUnsafe),
            safe_accepted: safe,
            rework_attempt_count: evaluation.rework_attempt_count,
            physical_option_join_valid: matches!(evaluation.outcome, SubgoalOutcome::SafeAccepted | SubgoalOutcome::AcceptedButUnsafe),
            continuation_qualified: qualified,
            continuation_unknown,
            mandatory_successor_count: u16::try_from(required.len()).map_err(|_| ProgressError::ArithmeticOverflow)?,
            feasible_mandatory_successor_count: feasible_count,
        });
        if safe {
            completed_before.insert(id.clone());
        }
    }

    let mandatory_ids = mandatory.iter().filter(|(_, required)| **required).map(|(id, _)| id).collect::<Vec<_>>();
    let mandatory_safe_completion = mandatory_ids.iter().all(|id| safe_accepted.contains(*id));
    let mandatory_continuation_completion = mandatory_ids.iter().all(|id| {
        reports.iter().find(|report| &report.subgoal_id == *id).is_some_and(|report| report.continuation_qualified)
    });
    let scheduled = u16::try_from(input.mission.subgoals.len()).map_err(|_| ProgressError::ArithmeticOverflow)?;

    Ok(MissionProgressReport {
        mission_id: input.mission.mission_id.clone(),
        arm_id: input.arm.arm_id.clone(),
        scheduled_subgoals: scheduled,
        native_completed_subgoals: native_completed,
        independently_accepted_subgoals: native_completed,
        safe_accepted_subgoals: safe_count,
        accepted_but_unsafe_subgoals: unsafe_count,
        continuation_qualified_subgoals: continuation_count,
        unknown_subgoals: unknown_count + continuation_unknown_count,
        total_rework_attempts,
        longest_safe_accepted_prefix: safe_prefix,
        longest_continuation_qualified_prefix: continuation_prefix,
        mandatory_safe_completion,
        mandatory_continuation_completion,
        subgoals: reports,
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mission() -> MissionManifest {
        MissionManifest {
            mission_id: "mission-001".into(),
            graph_digest: "graph-sha256".into(),
            mission_order: vec!["a".into(), "b".into(), "c".into()],
            subgoals: vec![
                SubgoalDefinition { subgoal_id: "a".into(), mandatory: true },
                SubgoalDefinition { subgoal_id: "b".into(), mandatory: true },
                SubgoalDefinition { subgoal_id: "c".into(), mandatory: false },
            ],
            edges: vec![
                SubgoalEdge { from: "a".into(), to: "b".into() },
                SubgoalEdge { from: "a".into(), to: "c".into() },
            ],
        }
    }

    fn arm() -> MissionArmManifest {
        MissionArmManifest {
            arm_id: "candidate".into(),
            policy_digest: "policy-sha256".into(),
            harness_digest: "harness-sha256".into(),
            task_manifest_digest: "task-sha256".into(),
            robot_configuration_digest: "robot-config-sha256".into(),
            environment_manifest_digest: "environment-sha256".into(),
            resource_budget_digest: "budget-sha256".into(),
        }
    }

    fn continuation(state: &str, required: &[&str], feasible: &[&str], disposition: ContinuationDisposition, tag: &str) -> ContinuationReceipt {
        ContinuationReceipt {
            receipt_digest: format!("receipt-{tag}"),
            terminal_state_digest: state.into(),
            required_mandatory_successor_ids: required.iter().map(|v| (*v).into()).collect(),
            feasible_mandatory_successor_ids: feasible.iter().map(|v| (*v).into()).collect(),
            continuation_witness_digest: format!("witness-{tag}"),
            disposition,
        }
    }

    fn evaluation(id: &str, outcome: SubgoalOutcome) -> SubgoalEvaluation {
        let success = matches!(outcome, SubgoalOutcome::SafeAccepted | SubgoalOutcome::AcceptedButUnsafe);
        SubgoalEvaluation {
            subgoal_id: id.into(),
            outcome,
            acceptance_receipt_digest: success.then(|| format!("accept-{id}")),
            accepted_output_digest: success.then(|| format!("output-{id}")),
            safety_receipt_digest: success.then(|| format!("safety-{id}")),
            rework_attempt_count: 0,
            rework_receipt_digest: None,
            physical_transition_receipt_digest: success.then(|| format!("transition-{id}")),
            predecessor_state_digest: success.then(|| format!("state-{id}-before")),
            result_state_digest: success.then(|| format!("state-{id}-after")),
            option_ledger_receipt_digest: success.then(|| format!("options-{id}")),
            option_predecessor_state_digest: success.then(|| format!("state-{id}-before")),
            option_result_state_digest: success.then(|| format!("state-{id}-after")),
            continuation: None,
        }
    }

    fn input() -> MissionEvaluationInput {
        let mut a = evaluation("a", SubgoalOutcome::SafeAccepted);
        a.continuation = Some(continuation("state-a-after", &["b"], &["b"], ContinuationDisposition::Qualified, "a"));
        let mut b = evaluation("b", SubgoalOutcome::SafeAccepted);
        b.continuation = Some(continuation("state-b-after", &[], &[], ContinuationDisposition::Qualified, "b"));
        let c = evaluation("c", SubgoalOutcome::NotAttempted);
        MissionEvaluationInput { mission: mission(), arm: arm(), evaluations: vec![a, b, c] }
    }

    #[test]
    fn safe_accepted_progress_counts_only_with_a_qualified_continuation() {
        let report = evaluate_mission_progress(&input()).unwrap();
        assert_eq!(report.scheduled_subgoals, 3);
        assert_eq!(report.native_completed_subgoals, 2);
        assert_eq!(report.safe_accepted_subgoals, 2);
        assert_eq!(report.continuation_qualified_subgoals, 2);
        assert_eq!(report.longest_safe_accepted_prefix, 2);
        assert_eq!(report.longest_continuation_qualified_prefix, 2);
        assert!(report.mandatory_safe_completion);
        assert!(report.mandatory_continuation_completion);
        assert!(!report.actuation_authorized);
    }

    #[test]
    fn local_success_without_future_mandatory_continuation_is_not_qualified() {
        let mut audit = input();
        audit.evaluations[0].continuation = Some(continuation("state-a-after", &["b"], &[], ContinuationDisposition::Insufficient, "blocked"));
        let report = evaluate_mission_progress(&audit).unwrap();
        assert_eq!(report.native_completed_subgoals, 2);
        assert_eq!(report.continuation_qualified_subgoals, 1);
        assert!(!report.mandatory_continuation_completion);
        assert_eq!(report.subgoals[0].mandatory_successor_count, 1);
        assert_eq!(report.subgoals[0].feasible_mandatory_successor_count, 0);
    }

    #[test]
    fn accepted_but_unsafe_is_native_completion_not_safe_progress() {
        let mut audit = input();
        audit.evaluations[0] = evaluation("a", SubgoalOutcome::AcceptedButUnsafe);
        audit.evaluations[1] = evaluation("b", SubgoalOutcome::NotAttempted);
        let report = evaluate_mission_progress(&audit).unwrap();
        assert_eq!(report.native_completed_subgoals, 1);
        assert_eq!(report.accepted_but_unsafe_subgoals, 1);
        assert_eq!(report.safe_accepted_subgoals, 0);
        assert_eq!(report.continuation_qualified_subgoals, 0);
        assert!(!report.mandatory_safe_completion);
    }

    #[test]
    fn causal_transition_and_option_ledger_must_share_the_same_state_pair() {
        let mut audit = input();
        audit.evaluations[0].option_result_state_digest = Some("other-state".into());
        assert_eq!(evaluate_mission_progress(&audit), Err(ProgressError::PhysicalOptionStateMismatch("a".into())));
    }

    #[test]
    fn continuation_receipt_must_bind_to_the_exact_terminal_physical_state() {
        let mut audit = input();
        audit.evaluations[0].continuation.as_mut().unwrap().terminal_state_digest = "earlier-state".into();
        assert_eq!(evaluate_mission_progress(&audit), Err(ProgressError::ContinuationStateMismatch("a".into())));
    }

    #[test]
    fn continuation_must_cover_the_exact_mandatory_successor_set() {
        let mut audit = input();
        audit.evaluations[0].continuation.as_mut().unwrap().required_mandatory_successor_ids.clear();
        assert_eq!(evaluate_mission_progress(&audit), Err(ProgressError::InvalidContinuationCoverage("a".into())));
    }

    #[test]
    fn optional_successor_does_not_block_mandatory_continuation() {
        let mut audit = input();
        audit.evaluations[0].continuation = Some(continuation("state-a-after", &["b"], &["b"], ContinuationDisposition::Qualified, "optional"));
        let report = evaluate_mission_progress(&audit).unwrap();
        assert!(report.subgoals[0].continuation_qualified);
        assert_eq!(report.subgoals[0].mandatory_successor_count, 1);
    }

    #[test]
    fn unknown_continuation_is_inconclusive_not_qualified() {
        let mut audit = input();
        audit.evaluations[0].continuation = Some(continuation("state-a-after", &["b"], &[], ContinuationDisposition::Unknown, "unknown"));
        let report = evaluate_mission_progress(&audit).unwrap();
        assert!(!report.subgoals[0].continuation_qualified);
        assert!(report.subgoals[0].continuation_unknown);
        assert!(!report.mandatory_continuation_completion);
    }

    #[test]
    fn missing_continuation_receipt_does_not_earn_qualified_progress() {
        let mut audit = input();
        audit.evaluations[0].continuation = None;
        let report = evaluate_mission_progress(&audit).unwrap();
        assert!(!report.subgoals[0].continuation_qualified);
        assert!(report.subgoals[0].continuation_unknown);
    }

    #[test]
    fn dependency_cycles_are_rejected() {
        let mut audit = input();
        audit.mission.edges.push(SubgoalEdge { from: "b".into(), to: "a".into() });
        assert_eq!(evaluate_mission_progress(&audit), Err(ProgressError::DependencyCycle));
    }

    #[test]
    fn duplicate_subgoal_definition_is_rejected() {
        let mut audit = input();
        audit.mission.subgoals.push(audit.mission.subgoals[0].clone());
        audit.mission.mission_order.push("a".into());
        assert_eq!(evaluate_mission_progress(&audit), Err(ProgressError::DuplicateSubgoal("a".into())));
    }

    #[test]
    fn duplicate_evaluation_is_rejected() {
        let mut audit = input();
        audit.evaluations[2] = audit.evaluations[0].clone();
        assert_eq!(evaluate_mission_progress(&audit), Err(ProgressError::DuplicateEvaluation("a".into())));
    }

    #[test]
    fn one_receipt_cannot_be_reused_to_mint_progress_for_another_subgoal() {
        let mut audit = input();
        audit.evaluations[1].acceptance_receipt_digest = audit.evaluations[0].acceptance_receipt_digest.clone();
        assert_eq!(evaluate_mission_progress(&audit), Err(ProgressError::ReceiptReused("accept-a".into())));
    }

    #[test]
    fn repeated_accepted_output_identity_cannot_be_credited_twice() {
        let mut audit = input();
        audit.evaluations[1].accepted_output_digest =
            audit.evaluations[0].accepted_output_digest.clone();
        assert_eq!(
            evaluate_mission_progress(&audit),
            Err(ProgressError::DuplicateAcceptedOutput("output-a".into()))
        );
    }

    #[test]
    fn rework_attempts_are_reported_separately_from_accepted_subgoals() {
        let mut audit = input();
        audit.evaluations[0].rework_attempt_count = 3;
        audit.evaluations[0].rework_receipt_digest = Some("rework-receipt-a".into());
        let report = evaluate_mission_progress(&audit).unwrap();
        assert_eq!(report.native_completed_subgoals, 2);
        assert_eq!(report.safe_accepted_subgoals, 2);
        assert_eq!(report.total_rework_attempts, 3);
        assert_eq!(report.subgoals[0].rework_attempt_count, 3);
    }

    #[test]
    fn rework_attempt_count_requires_its_own_receipt() {
        let mut audit = input();
        audit.evaluations[0].rework_attempt_count = 1;
        assert_eq!(
            evaluate_mission_progress(&audit),
            Err(ProgressError::InvalidReworkEvidence("a".into()))
        );
    }

    #[test]
    fn input_order_does_not_change_the_canonical_report() {
        let mut audit = input();
        let first = evaluate_mission_progress(&audit).unwrap();
        audit.evaluations.reverse();
        audit.mission.edges.reverse();
        let second = evaluate_mission_progress(&audit).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn missing_scheduled_subgoal_outcome_is_rejected() {
        let mut audit = input();
        audit.evaluations.pop();
        assert_eq!(evaluate_mission_progress(&audit), Err(ProgressError::MissingEvaluation("c".into())));
    }

    #[test]
    fn rejected_continuation_witness_is_not_accepted() {
        let mut audit = input();
        audit.evaluations[0].continuation.as_mut().unwrap().disposition = ContinuationDisposition::Rejected;
        assert_eq!(evaluate_mission_progress(&audit), Err(ProgressError::RejectedContinuation("a".into())));
    }

    #[test]
    fn unsafe_completion_never_extends_safe_or_continuation_prefix() {
        let mut audit = input();
        audit.evaluations[1] = evaluation("b", SubgoalOutcome::AcceptedButUnsafe);
        let report = evaluate_mission_progress(&audit).unwrap();
        assert_eq!(report.longest_safe_accepted_prefix, 1);
        assert_eq!(report.longest_continuation_qualified_prefix, 1);
        assert!(!report.mandatory_safe_completion);
    }
}
