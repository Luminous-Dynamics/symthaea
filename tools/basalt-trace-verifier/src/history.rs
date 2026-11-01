use super::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct HistorySearchConfig {
    /// Hard cap on expanded (remaining operations, model state) search nodes.
    pub max_search_states: usize,
    /// V1 is intentionally bounded to a 64-bit operation mask with a stricter 32-op cap.
    pub max_operations: usize,
}

impl Default for HistorySearchConfig {
    fn default() -> Self {
        Self {
            max_search_states: 100_000,
            max_operations: 32,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HistorySearchReport {
    pub report_version: u32,
    pub status: String,
    pub trace_id: String,
    pub operations_checked: usize,
    pub states_explored: usize,
    pub memo_hits: usize,
    pub max_enabled_frontier: usize,
    pub search_budget: usize,
    pub search_budget_exhausted: bool,
    pub linearization: Vec<String>,
    pub counterexamples: Vec<String>,
    pub claim_ceiling: String,
}

struct SearchContext<'a> {
    trace: &'a Trace,
    predecessors: Vec<u64>,
    max_search_states: usize,
    states_explored: usize,
    memo_hits: usize,
    max_enabled_frontier: usize,
    memoized_failures: HashSet<(u64, String)>,
    search_budget_exhausted: bool,
    counterexamples: BTreeSet<String>,
}

fn state_fingerprint(state: &ModelState) -> String {
    let bytes = serde_json::to_vec(state)
        .expect("ModelState only contains serializable deterministic ordered fields");
    sha256_label(&bytes)
}

fn expected_optional_matches<T: PartialEq>(expected: &Option<T>, actual: &Option<T>) -> bool {
    expected.as_ref().is_none_or(|expected| Some(expected) == actual.as_ref())
}

/// Missing optional output fields mean that the backend did not expose that field.
/// A present field must exactly match the pure transition result.
fn observed_result_matches(expected: &ExpectedResult, actual: &ComputedResult) -> bool {
    expected.outcome == actual.outcome
        && expected_optional_matches(&expected.receipt_id, &actual.receipt_id)
        && expected_optional_matches(
            &expected.receipt_acceptance_revision,
            &actual.receipt_acceptance_revision,
        )
        && expected_optional_matches(
            &expected.current_authority_revision,
            &actual.current_authority_revision,
        )
        && expected_optional_matches(
            &expected.observed_store_revision,
            &actual.observed_store_revision,
        )
}

fn final_state_matches(state: &ModelState, trace: &Trace) -> bool {
    let mut expected_sources = BTreeMap::<String, SourceCursor>::new();
    for cursor in &trace.final_observation.source_cursors {
        if expected_sources
            .insert(cursor.source_id.clone(), cursor.clone())
            .is_some()
        {
            return false;
        }
    }

    let mut expected_logs = BTreeMap::<String, String>::new();
    for cursor in &trace.final_observation.log_cursors {
        if expected_logs
            .insert(cursor.log_origin_id.clone(), cursor.checkpoint.clone())
            .is_some()
        {
            return false;
        }
    }

    let mut expected_receipts = BTreeMap::<String, ModelReceipt>::new();
    for receipt in &trace.final_observation.accepted_receipts {
        if expected_receipts
            .insert(
                receipt.receipt_id.clone(),
                ModelReceipt {
                    receipt_digest: receipt.receipt_digest.clone(),
                    acceptance_revision: parse_revision(&receipt.receipt_acceptance_revision)
                        .unwrap_or(u64::MAX),
                },
            )
            .is_some()
        {
            return false;
        }
    }

    state.revision
        == parse_revision(&trace.final_observation.authority_revision).unwrap_or(u64::MAX)
        && state.policy_digest == trace.final_observation.policy_digest
        && state.sources == expected_sources
        && state.logs == expected_logs
        && state.receipts == expected_receipts
}

fn add_edge(predecessors: &mut [u64], before: usize, after: usize) {
    predecessors[after] |= 1_u64 << before;
}

fn build_partial_order(trace: &Trace) -> Result<(Vec<u64>, Vec<usize>), Vec<String>> {
    let count = trace.operations.len();
    let mut errors = Vec::new();
    let mut indexes = HashMap::<String, usize>::new();

    for (index, operation) in trace.operations.iter().enumerate() {
        if indexes.insert(operation.op_id.clone(), index).is_some() {
            push_error(
                &mut errors,
                format!("duplicate operation ID {}", operation.op_id),
            );
        }
        if operation.completed_at_ns < operation.invoked_at_ns {
            push_error(
                &mut errors,
                format!("{} completes before invocation", operation.op_id),
            );
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let mut predecessors = vec![0_u64; count];

    for (index, operation) in trace.operations.iter().enumerate() {
        for dependency in &operation.happens_after {
            match indexes.get(dependency) {
                Some(previous) => add_edge(&mut predecessors, *previous, index),
                None => push_error(
                    &mut errors,
                    format!(
                        "{} references missing happens_after operation {}",
                        operation.op_id, dependency
                    ),
                ),
            }
        }
    }

    // Client program order is independent of runner identity.
    let mut clients = BTreeMap::<String, Vec<(u64, usize)>>::new();
    for (index, operation) in trace.operations.iter().enumerate() {
        clients
            .entry(operation.client_id.clone())
            .or_default()
            .push((operation.client_sequence, index));
    }
    for (client, sequence) in &mut clients {
        sequence.sort_by_key(|(client_sequence, _)| *client_sequence);
        for (position, (actual_sequence, _)) in sequence.iter().enumerate() {
            if *actual_sequence != position as u64 {
                push_error(
                    &mut errors,
                    format!("client {client} has a sequence gap or duplicate"),
                );
            }
        }
        for pair in sequence.windows(2) {
            add_edge(&mut predecessors, pair[0].1, pair[1].1);
        }
    }

    // Monotonic clocks are only comparable within one runner.
    for (left_index, left) in trace.operations.iter().enumerate() {
        for (right_index, right) in trace.operations.iter().enumerate() {
            if left_index != right_index
                && left.runner_id == right.runner_id
                && left.completed_at_ns <= right.invoked_at_ns
            {
                add_edge(&mut predecessors, left_index, right_index);
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Produce one topological ordering only for structural validation.
    // The semantic search below does not privilege this arbitrary tie-break.
    let all = if count == 64 {
        u64::MAX
    } else {
        (1_u64 << count) - 1
    };
    let mut remaining = all;
    let mut done = 0_u64;
    let mut topological = Vec::with_capacity(count);
    while remaining != 0 {
        let next = (0..count).find(|index| {
            let bit = 1_u64 << index;
            remaining & bit != 0 && predecessors[*index] & !done == 0
        });
        let Some(index) = next else {
            return Err(vec![
                "causal, per-client and same-runner real-time constraints contain a cycle"
                    .to_string(),
            ]);
        };
        let bit = 1_u64 << index;
        remaining &= !bit;
        done |= bit;
        topological.push(index);
    }

    Ok((predecessors, topological))
}

fn visit(
    context: &mut SearchContext<'_>,
    remaining: u64,
    state: ModelState,
    order: &mut Vec<usize>,
) -> bool {
    if remaining == 0 {
        if final_state_matches(&state, context.trace) {
            return true;
        }
        if context.counterexamples.len() < 16 {
            context
                .counterexamples
                .insert("a complete ordering does not produce the observed final state".into());
        }
        return false;
    }

    if context.states_explored >= context.max_search_states {
        context.search_budget_exhausted = true;
        return false;
    }

    let key = (remaining, state_fingerprint(&state));
    if context.memoized_failures.contains(&key) {
        context.memo_hits += 1;
        return false;
    }

    context.states_explored += 1;
    let enabled: Vec<usize> = (0..context.trace.operations.len())
        .filter(|index| {
            let bit = 1_u64 << index;
            remaining & bit != 0 && context.predecessors[*index] & remaining == 0
        })
        .collect();
    context.max_enabled_frontier = context.max_enabled_frontier.max(enabled.len());

    if enabled.is_empty() {
        if context.counterexamples.len() < 16 {
            context
                .counterexamples
                .insert("no operation is enabled in a nonempty remaining history".into());
        }
        context.memoized_failures.insert(key);
        return false;
    }

    for index in enabled {
        let operation = &context.trace.operations[index];
        let mut next_state = state.clone();
        let mut transition_errors = Vec::new();
        let computed = apply_operation(
            &mut next_state,
            context.trace,
            operation,
            &mut transition_errors,
        );
        if !transition_errors.is_empty()
            || !observed_result_matches(&operation.expected_result, &computed)
        {
            if context.counterexamples.len() < 16 {
                context.counterexamples.insert(format!(
                    "{} observed {:?}, but transition produced {:?}",
                    operation.op_id, operation.expected_result, computed
                ));
            }
            continue;
        }

        order.push(index);
        if visit(
            context,
            remaining & !(1_u64 << index),
            next_state,
            order,
        ) {
            return true;
        }
        order.pop();

        if context.search_budget_exhausted {
            return false;
        }
    }

    context.memoized_failures.insert(key);
    false
}

/// Checks whether any sequential execution explains a fully completed observed
/// history while respecting client order, explicit dependencies and same-runner
/// real-time precedence. The declared reference_linearization_order is ignored.
pub fn verify_observed_history(
    trace: &Trace,
    config: &HistorySearchConfig,
) -> HistorySearchReport {
    let claim_ceiling = "Bounded search over trace-shaped, fully completed operations. Final modeled revision/cursors/policy/receipt fields are compared; the opaque final state digest is not recomputed from a complete canonical manifest. This does not parse raw backend event streams, resolve pending operations, verify witness signatures, test physical durability, or qualify a backend.";

    let invalid = |errors: Vec<String>| HistorySearchReport {
        report_version: 1,
        status: "invalid_evidence".to_string(),
        trace_id: trace.trace_id.clone(),
        operations_checked: 0,
        states_explored: 0,
        memo_hits: 0,
        max_enabled_frontier: 0,
        search_budget: config.max_search_states,
        search_budget_exhausted: false,
        linearization: Vec::new(),
        counterexamples: errors,
        claim_ceiling: claim_ceiling.to_string(),
    };

    let count = trace.operations.len();
    if count == 0 {
        return invalid(vec!["history has no operations".to_string()]);
    }
    if count > 32 || count > config.max_operations {
        return HistorySearchReport {
            report_version: 1,
            status: "inconclusive".to_string(),
            trace_id: trace.trace_id.clone(),
            operations_checked: count,
            states_explored: 0,
            memo_hits: 0,
            max_enabled_frontier: 0,
            search_budget: config.max_search_states,
            search_budget_exhausted: true,
            linearization: Vec::new(),
            counterexamples: vec![format!(
                "v1 exact-search limit is {} operations; history contains {count}",
                config.max_operations.min(32)
            )],
            claim_ceiling: claim_ceiling.to_string(),
        };
    }
    if trace.operations.iter().any(|operation| {
        operation.completed_at_ns < operation.invoked_at_ns
    }) {
        return invalid(vec![
            "completed history contains an impossible operation interval".to_string(),
        ]);
    }

    let (predecessors, topological) = match build_partial_order(trace) {
        Ok(value) => value,
        Err(errors) => return invalid(errors),
    };

    let mut structural_trace = trace.clone();
    structural_trace.reference_linearization_order = topological
        .iter()
        .map(|index| trace.operations[*index].op_id.clone())
        .collect();
    let mut structural_errors = Vec::new();
    validate_structure(&structural_trace, &mut structural_errors);
    validate_retry_groups(&structural_trace, &mut structural_errors);
    validate_witnesses(&structural_trace, &mut structural_errors);
    if !structural_errors.is_empty() {
        return invalid(structural_errors);
    }

    let mut context = SearchContext {
        trace,
        predecessors,
        max_search_states: config.max_search_states,
        states_explored: 0,
        memo_hits: 0,
        max_enabled_frontier: 0,
        memoized_failures: HashSet::new(),
        search_budget_exhausted: false,
        counterexamples: BTreeSet::new(),
    };
    let all = (1_u64 << count) - 1;
    let mut order = Vec::with_capacity(count);
    let found = visit(
        &mut context,
        all,
        initial_model_state(trace),
        &mut order,
    );

    let status = if found {
        "pass"
    } else if context.search_budget_exhausted {
        "inconclusive"
    } else {
        "fail"
    };

    HistorySearchReport {
        report_version: 1,
        status: status.to_string(),
        trace_id: trace.trace_id.clone(),
        operations_checked: count,
        states_explored: context.states_explored,
        memo_hits: context.memo_hits,
        max_enabled_frontier: context.max_enabled_frontier,
        search_budget: context.max_search_states,
        search_budget_exhausted: context.search_budget_exhausted,
        linearization: if found {
            order
                .iter()
                .map(|index| trace.operations[*index].op_id.clone())
                .collect()
        } else {
            Vec::new()
        },
        counterexamples: if found {
            Vec::new()
        } else if context.counterexamples.is_empty() {
            vec!["no legal sequential interpretation found".to_string()]
        } else {
            context.counterexamples.into_iter().collect()
        },
        claim_ceiling: claim_ceiling.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Trace {
        let raw = include_bytes!("../../../benchmarks/basalt/storage-conformance-trace.v1.json");
        serde_json::from_slice(raw).expect("reference fixture must deserialize")
    }

    fn exhaustive_count(
        trace: &Trace,
        predecessors: &[u64],
        remaining: u64,
        state: ModelState,
    ) -> usize {
        if remaining == 0 {
            return usize::from(final_state_matches(&state, trace));
        }
        let mut total = 0usize;
        for index in 0..trace.operations.len() {
            let bit = 1_u64 << index;
            if remaining & bit == 0 || predecessors[index] & remaining != 0 {
                continue;
            }
            let operation = &trace.operations[index];
            let mut next = state.clone();
            let mut errors = Vec::new();
            let result = apply_operation(&mut next, trace, operation, &mut errors);
            if errors.is_empty() && observed_result_matches(&operation.expected_result, &result) {
                total += exhaustive_count(trace, predecessors, remaining & !bit, next);
            }
        }
        total
    }

    fn three_overlapping_restore_reads() -> Trace {
        let mut trace = fixture();
        let witness = trace.witness_observations.last().unwrap().clone();
        let template = trace
            .operations
            .iter()
            .find(|operation| operation.kind == "restore_verify")
            .unwrap()
            .clone();

        let mut operations = Vec::new();
        for (index, (client, start, end)) in [
            ("client-A", 10, 100),
            ("client-B", 20, 90),
            ("client-C", 30, 80),
        ]
        .into_iter()
        .enumerate()
        {
            let mut operation = template.clone();
            operation.op_id = format!("restore-{index}");
            operation.client_id = client.to_string();
            operation.client_sequence = 0;
            operation.runner_id = "runner-overlap".to_string();
            operation.invoked_at_ns = start;
            operation.completed_at_ns = end;
            operation.happens_after.clear();
            operation.request.restored_incarnation_id = Some(witness.incarnation_id.clone());
            operation.request.restored_authority_revision = Some(witness.authority_revision.clone());
            operation.request.restored_state_digest = Some(witness.state_digest.clone());
            operation.request.witness_sequence_seen = Some(witness.sequence);
            operation.expected_result = ExpectedResult {
                outcome: "restore_verified".to_string(),
                receipt_id: None,
                receipt_acceptance_revision: None,
                current_authority_revision: Some(witness.authority_revision.clone()),
                observed_store_revision: Some(witness.authority_revision.clone()),
                reason: None,
            };
            operations.push(operation);
        }

        trace.initial_state = InitialState {
            authority_revision: trace.final_observation.authority_revision.clone(),
            state_digest: trace.final_observation.state_digest.clone(),
            policy_digest: trace.final_observation.policy_digest.clone(),
            source_cursors: trace.final_observation.source_cursors.clone(),
            log_cursors: trace.final_observation.log_cursors.clone(),
        };
        trace.final_observation.accepted_receipts.clear();
        trace.operations = operations;
        trace.reference_linearization_order.clear();
        trace.faults.clear();
        trace.negative_controls.clear();
        trace.invariant_expectations.clear();
        trace
    }

    #[test]
    fn finds_a_legal_order_without_using_the_reference_tie_break() {
        let mut trace = fixture();
        trace.reference_linearization_order = vec!["this-order-is-not-authoritative".to_string()];
        trace.operations.reverse();
        let report = verify_observed_history(&trace, &HistorySearchConfig::default());
        assert_eq!(report.status, "pass", "{:#?}", report.counterexamples);
        assert_eq!(report.linearization.len(), trace.operations.len());
        assert!(!report.linearization.contains(&"this-order-is-not-authoritative".to_string()));
    }

    #[test]
    fn well_formed_but_impossible_result_is_exhaustively_rejected() {
        let mut trace = fixture();
        trace
            .operations
            .iter_mut()
            .find(|operation| operation.op_id == "op-restore-old-snapshot")
            .unwrap()
            .expected_result
            .outcome = "compare_conflict".to_string();
        let report = verify_observed_history(&trace, &HistorySearchConfig::default());
        assert_eq!(report.status, "fail");
        assert!(report.search_budget_exhausted == false);
    }

    #[test]
    fn memoized_search_matches_unoptimized_exhaustive_oracle_on_overlapping_reads() {
        let mut trace = three_overlapping_restore_reads();
        let (predecessors, _) = build_partial_order(&trace).unwrap();
        let all = (1_u64 << trace.operations.len()) - 1;
        let oracle_count = exhaustive_count(&trace, &predecessors, all, initial_model_state(&trace));
        assert_eq!(oracle_count, 6, "all six read-only orderings should be legal");

        trace.operations.reverse();
        let report = verify_observed_history(&trace, &HistorySearchConfig::default());
        assert_eq!(report.status, "pass", "{:#?}", report.counterexamples);
        assert_eq!(report.linearization.len(), 3);
    }

    #[test]
    fn memoized_search_agrees_with_exhaustive_oracle_on_reference_trace() {
        let mut trace = fixture();
        trace.operations.reverse();
        let (predecessors, _) = build_partial_order(&trace).unwrap();
        let all = (1_u64 << trace.operations.len()) - 1;
        let oracle_count = exhaustive_count(&trace, &predecessors, all, initial_model_state(&trace));
        assert_eq!(oracle_count, 1, "the shared-predecessor race has one legal winner order");

        let report = verify_observed_history(&trace, &HistorySearchConfig::default());
        assert_eq!(report.status, "pass", "{:#?}", report.counterexamples);
    }

    #[test]
    fn exhausted_search_never_reports_pass_or_fail() {
        let trace = fixture();
        let report = verify_observed_history(
            &trace,
            &HistorySearchConfig {
                max_search_states: 1,
                max_operations: 32,
            },
        );
        assert_eq!(report.status, "inconclusive");
        assert!(report.search_budget_exhausted);
    }

    #[test]
    fn dependency_cycle_is_invalid_evidence() {
        let mut trace = fixture();
        trace.operations[0].happens_after = vec![trace.operations[1].op_id.clone()];
        trace.operations[1].happens_after = vec![trace.operations[0].op_id.clone()];
        let report = verify_observed_history(&trace, &HistorySearchConfig::default());
        assert_eq!(report.status, "invalid_evidence");
        assert!(report.counterexamples.iter().any(|x| x.contains("cycle")));
    }

    #[test]
    fn cross_runner_monotonic_times_do_not_create_edges() {
        let mut trace = fixture();
        trace.reference_linearization_order.clear();
        for operation in &mut trace.operations {
            operation.runner_id = format!("runner-{}", operation.client_id);
            operation.invoked_at_ns = 0;
            operation.completed_at_ns = 1;
        }
        let report = verify_observed_history(&trace, &HistorySearchConfig::default());
        assert!(["pass", "fail", "inconclusive"].contains(&report.status.as_str()));
        assert_ne!(report.status, "invalid_evidence");
    }

    #[test]
    fn two_overlapping_read_only_restore_checks_admit_both_orders() {
        let mut trace = fixture();
        let witness = trace.witness_observations.last().unwrap().clone();
        let mut first = trace
            .operations
            .iter()
            .find(|operation| operation.kind == "restore_verify")
            .unwrap()
            .clone();
        first.op_id = "restore-A".to_string();
        first.client_id = "client-A".to_string();
        first.client_sequence = 0;
        first.runner_id = "runner-1".to_string();
        first.invoked_at_ns = 10;
        first.completed_at_ns = 100;
        first.happens_after.clear();
        first.request.restored_incarnation_id = Some(witness.incarnation_id.clone());
        first.request.restored_authority_revision = Some(witness.authority_revision.clone());
        first.request.restored_state_digest = Some(witness.state_digest.clone());
        first.request.witness_sequence_seen = Some(witness.sequence);
        first.expected_result = ExpectedResult {
            outcome: "restore_verified".to_string(),
            receipt_id: None,
            receipt_acceptance_revision: None,
            current_authority_revision: Some(witness.authority_revision.clone()),
            observed_store_revision: Some(witness.authority_revision.clone()),
            reason: None,
        };

        let mut second = first.clone();
        second.op_id = "restore-B".to_string();
        second.client_id = "client-B".to_string();
        second.invoked_at_ns = 20;
        second.completed_at_ns = 90;

        trace.operations = vec![first.clone(), second.clone()];
        trace.initial_state = InitialState {
            authority_revision: trace.final_observation.authority_revision.clone(),
            state_digest: trace.final_observation.state_digest.clone(),
            policy_digest: trace.final_observation.policy_digest.clone(),
            source_cursors: trace.final_observation.source_cursors.clone(),
            log_cursors: trace.final_observation.log_cursors.clone(),
        };
        trace.final_observation.accepted_receipts.clear();
        trace.reference_linearization_order.clear();
        trace.faults.clear();
        trace.negative_controls.clear();
        trace.invariant_expectations.clear();

        let report = verify_observed_history(&trace, &HistorySearchConfig::default());
        assert_eq!(report.status, "pass", "{:#?}", report.counterexamples);
        assert_eq!(report.linearization.len(), 2);

        trace.operations.reverse();
        let reverse = verify_observed_history(&trace, &HistorySearchConfig::default());
        assert_eq!(reverse.status, "pass", "{:#?}", reverse.counterexamples);
        assert_eq!(reverse.linearization, vec!["restore-B", "restore-A"]);
    }
}
