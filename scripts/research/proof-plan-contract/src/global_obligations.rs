//! Global proof-obligation accounting for ProofPlanV1.
//!
//! This pass checks that generated goal obligations are produced, transformed
//! or discharged exactly once within an explicitly declared proof alternative.
//! It does not validate mathematics or prove that a discharge is kernel-valid.

use crate::{
    AlternativeStatus, EdgeRelation, NodeKind, PlanAlternative, PlanStatus, ProofPlan,
};
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlternativeObligationReport {
    pub alternative_id: String,
    pub produced_subgoals: Vec<String>,
    pub transformed_goals: Vec<String>,
    pub discharged_goals: Vec<String>,
    pub terminal_goals: Vec<String>,
    pub unresolved_goals: Vec<String>,
    pub cycle_nodes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalObligationReport {
    pub alternatives: Vec<AlternativeObligationReport>,
    pub primary_alternative_count: usize,
    pub errors: Vec<String>,
}

fn report_for_alternative(
    plan: &ProofPlan,
    alternative: &PlanAlternative,
    errors: &mut Vec<String>,
) -> AlternativeObligationReport {
    let node_ids: HashSet<&str> = alternative.node_ids.iter().map(String::as_str).collect();
    let nodes: HashMap<&str, _> = plan.nodes.iter().map(|node| (node.id.as_str(), node)).collect();

    let unique_node_ids: HashSet<&str> = alternative.node_ids.iter().map(String::as_str).collect();
    if unique_node_ids.len() != alternative.node_ids.len() {
        errors.push(format!("alternative {} repeats node IDs", alternative.id));
    }

    let mut creators = HashMap::<String, Vec<String>>::new();
    let mut consumers = HashMap::<String, Vec<(String, EdgeRelation)>>::new();
    let mut transitions = HashMap::<String, Vec<String>>::new();
    let mut produced = BTreeSet::<String>::new();
    let mut transformed = BTreeSet::<String>::new();
    let mut discharged = BTreeSet::<String>::new();
    let mut unresolved = BTreeSet::<String>::new();

    for edge in &plan.edges {
        if !node_ids.contains(edge.source_id.as_str())
            || !node_ids.contains(edge.target_id.as_str())
        {
            continue;
        }
        if edge.relation == EdgeRelation::CreatesSubgoal {
            creators
                .entry(edge.target_id.clone())
                .or_default()
                .push(edge.source_id.clone());
        }
    }

    // Every consumption edge must be represented in the action's declared inputs.
    for edge in &plan.edges {
        if !node_ids.contains(edge.source_id.as_str())
            || !node_ids.contains(edge.target_id.as_str())
            || !matches!(edge.relation, EdgeRelation::Discharges | EdgeRelation::Transforms)
        {
            continue;
        }
        let source = nodes.get(edge.source_id.as_str());
        let declared = source
            .and_then(|node| node.action_spec.as_ref())
            .is_some_and(|action| action.input_goal_ids.contains(&edge.target_id));
        if !declared {
            errors.push(format!(
                "alternative {} action edge {} -> {} consumes a goal absent from action_spec.input_goal_ids",
                alternative.id, edge.source_id, edge.target_id
            ));
        }
    }

    for node_id in &alternative.node_ids {
        let Some(node) = nodes.get(node_id.as_str()) else {
            continue;
        };
        if node.kind != NodeKind::PlanAction {
            continue;
        }
        let Some(action) = &node.action_spec else {
            continue;
        };

        let specified_outputs: BTreeSet<&str> =
            action.output_subgoal_ids.iter().map(String::as_str).collect();
        if specified_outputs.len() != action.output_subgoal_ids.len() {
            errors.push(format!(
                "alternative {} action {} repeats an output subgoal ID",
                alternative.id, node.id
            ));
        }

        let edge_outputs: BTreeSet<&str> = plan
            .edges
            .iter()
            .filter(|edge| {
                edge.source_id == node.id
                    && edge.relation == EdgeRelation::CreatesSubgoal
                    && node_ids.contains(edge.target_id.as_str())
            })
            .map(|edge| edge.target_id.as_str())
            .collect();

        if specified_outputs != edge_outputs {
            errors.push(format!(
                "alternative {} action {} output_subgoal_ids do not exactly match creates_subgoal edges",
                alternative.id, node.id
            ));
        }

        for output_id in &action.output_subgoal_ids {
            if !node_ids.contains(output_id.as_str()) {
                errors.push(format!(
                    "alternative {} action {} creates subgoal {} outside its node set",
                    alternative.id, node.id, output_id
                ));
            }
            if nodes.get(output_id.as_str()).is_none_or(|output| output.kind != NodeKind::Subgoal) {
                errors.push(format!(
                    "alternative {} action {} output {} is not a subgoal node",
                    alternative.id, node.id, output_id
                ));
            }
            produced.insert(output_id.clone());
        }

        for input_id in &action.input_goal_ids {
            if !node_ids.contains(input_id.as_str()) {
                errors.push(format!(
                    "alternative {} action {} consumes goal {} outside its node set",
                    alternative.id, node.id, input_id
                ));
                continue;
            }

            let relations: Vec<EdgeRelation> = plan
                .edges
                .iter()
                .filter(|edge| {
                    edge.source_id == node.id
                        && edge.target_id == *input_id
                        && matches!(
                            edge.relation,
                            EdgeRelation::Discharges | EdgeRelation::Transforms
                        )
                        && node_ids.contains(edge.target_id.as_str())
                })
                .map(|edge| edge.relation)
                .collect();

            if relations.len() != 1 {
                errors.push(format!(
                    "alternative {} action {} must account for input goal {} exactly once; found {} discharge/transform edges",
                    alternative.id,
                    node.id,
                    input_id,
                    relations.len()
                ));
                continue;
            }

            let relation = relations[0];
            consumers
                .entry(input_id.clone())
                .or_default()
                .push((node.id.clone(), relation));

            match relation {
                EdgeRelation::Discharges => {
                    discharged.insert(input_id.clone());
                }
                EdgeRelation::Transforms => {
                    transformed.insert(input_id.clone());
                    if action.output_subgoal_ids.is_empty() {
                        errors.push(format!(
                            "alternative {} transforms goal {} without creating a successor subgoal",
                            alternative.id, input_id
                        ));
                    } else {
                        transitions
                            .entry(input_id.clone())
                            .or_default()
                            .extend(action.output_subgoal_ids.iter().cloned());
                    }
                }
                _ => {}
            }
        }
    }

    // Every subgoal must have one and only one producer in this alternative.
    for node_id in &alternative.node_ids {
        let Some(node) = nodes.get(node_id.as_str()) else {
            continue;
        };
        if node.kind != NodeKind::Subgoal {
            continue;
        }
        let producer_count = creators.get(node_id).map_or(0, Vec::len);
        if producer_count == 0 {
            errors.push(format!(
                "alternative {} subgoal {} has no producing action",
                alternative.id, node_id
            ));
        } else if producer_count > 1 {
            errors.push(format!(
                "alternative {} subgoal {} has {} producing actions",
                alternative.id, node_id, producer_count
            ));
        }
    }

    // Every goal state in the route must be accounted for exactly once.
    for node_id in &alternative.node_ids {
        let Some(node) = nodes.get(node_id.as_str()) else {
            continue;
        };
        if !matches!(node.kind, NodeKind::GoalState | NodeKind::Subgoal) {
            continue;
        }

        if node.kind == NodeKind::GoalState && node.id != alternative.root_goal_id {
            errors.push(format!(
                "alternative {} contains non-root goal_state {}",
                alternative.id, node.id
            ));
        }

        let consumer_count = consumers.get(node_id).map_or(0, Vec::len);
        if consumer_count == 0 {
            unresolved.insert(node.id.clone());
        } else if consumer_count > 1 {
            errors.push(format!(
                "alternative {} goal {} is consumed {} times; expected exactly once",
                alternative.id, node.id, consumer_count
            ));
        }
    }

    let cycle_nodes = find_transition_cycles(&transitions);
    if !cycle_nodes.is_empty() {
        errors.push(format!(
            "alternative {} contains a goal-transition cycle: {}",
            alternative.id,
            cycle_nodes.join(" -> ")
        ));
    }

    AlternativeObligationReport {
        alternative_id: alternative.id.clone(),
        produced_subgoals: produced.into_iter().collect(),
        transformed_goals: transformed.into_iter().collect(),
        discharged_goals: discharged.iter().cloned().collect(),
        terminal_goals: discharged.into_iter().collect(),
        unresolved_goals: unresolved.into_iter().collect(),
        cycle_nodes,
    }
}

fn find_transition_cycles(transitions: &HashMap<String, Vec<String>>) -> Vec<String> {
    fn visit(
        node: &str,
        graph: &HashMap<String, Vec<String>>,
        marks: &mut HashMap<String, u8>,
        stack: &mut Vec<String>,
        cycles: &mut BTreeSet<String>,
    ) {
        match marks.get(node).copied().unwrap_or(0) {
            2 => return,
            1 => {
                if let Some(start) = stack.iter().position(|entry| entry == node) {
                    for entry in &stack[start..] {
                        cycles.insert(entry.clone());
                    }
                    cycles.insert(node.to_owned());
                }
                return;
            }
            _ => {}
        }

        marks.insert(node.to_owned(), 1);
        stack.push(node.to_owned());
        if let Some(next_nodes) = graph.get(node) {
            for next in next_nodes {
                visit(next, graph, marks, stack, cycles);
            }
        }
        stack.pop();
        marks.insert(node.to_owned(), 2);
    }

    let mut marks = HashMap::<String, u8>::new();
    let mut stack = Vec::new();
    let mut cycles = BTreeSet::new();
    for node in transitions.keys() {
        visit(node, transitions, &mut marks, &mut stack, &mut cycles);
    }
    cycles.into_iter().collect()
}

pub fn analyze_global_obligations(plan: &ProofPlan) -> GlobalObligationReport {
    let mut errors = Vec::new();
    let primary_alternatives: Vec<&PlanAlternative> = plan
        .alternatives
        .iter()
        .filter(|alternative| matches!(&alternative.status, AlternativeStatus::Primary))
        .collect();
    let primary_alternative_count = primary_alternatives.len();

    if plan.status == PlanStatus::Accepted && primary_alternative_count != 1 {
        errors.push(format!(
            "accepted plan requires exactly one primary alternative; found {primary_alternative_count}"
        ));
    }

    let alternatives: Vec<AlternativeObligationReport> = plan
        .alternatives
        .iter()
        .map(|alternative| report_for_alternative(plan, alternative, &mut errors))
        .collect();

    if plan.status == PlanStatus::Accepted {
        if let Some(primary) = alternatives.iter().find(|report| {
            primary_alternatives
                .iter()
                .any(|alternative| alternative.id == report.alternative_id)
        }) {
            if !primary.unresolved_goals.is_empty() {
                errors.push(format!(
                    "accepted plan has unresolved primary-route obligations: {}",
                    primary.unresolved_goals.join(", ")
                ));
            }
            if primary.terminal_goals.is_empty() {
                errors.push(
                    "accepted primary route has no terminal discharged goal".to_owned(),
                );
            }
            if !primary.cycle_nodes.is_empty() {
                errors.push(
                    "accepted primary route contains a goal-transition cycle".to_owned(),
                );
            }
        } else {
            errors.push("accepted plan has no analyzable primary alternative".to_owned());
        }
    }

    GlobalObligationReport {
        alternatives,
        primary_alternative_count,
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_plan, PlanStatus};
    use serde_json::Value;

    const FIXTURE: &str =
        include_str!("../../../../docs/research/PROOF_PLAN_AND_COMM_V1.json");

    fn fixture() -> crate::ProofPlan {
        parse_plan(FIXTURE).expect("valid provisional ProofPlan fixture")
    }

    fn find_action_mut<'a>(
        plan: &'a mut Value,
        action_id: &str,
    ) -> &'a mut Value {
        plan["nodes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|node| node["id"] == action_id)
            .unwrap()
    }

    fn plan_value() -> Value {
        serde_json::from_str(FIXTURE).unwrap()
    }

    fn analyzed(value: &Value) -> GlobalObligationReport {
        let input = serde_json::to_string(value).unwrap();
        let plan = parse_plan(&input).unwrap();
        analyze_global_obligations(&plan)
    }

    #[test]
    fn and_comm_primary_route_accounts_for_both_directions() {
        let report = analyze_global_obligations(&fixture());
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.primary_alternative_count, 1);
        let primary = report
            .alternatives
            .iter()
            .find(|alternative| alternative.alternative_id == "ALT_PRIMARY")
            .unwrap();
        assert_eq!(primary.produced_subgoals.len(), 2);
        assert_eq!(primary.discharged_goals.len(), 2);
        assert!(primary.transformed_goals.is_empty());
        assert!(primary.unresolved_goals.is_empty());
        assert_eq!(primary.terminal_goals.len(), 2);
    }

    #[test]
    fn missing_discharge_is_reported_as_unresolved_for_provisional_plan() {
        let mut value = plan_value();
        value["edges"].as_array_mut().unwrap().retain(|edge| {
            !(edge["source_id"] == "A_REVERSE"
                && edge["target_id"] == "G_REVERSE"
                && edge["relation"] == "discharges")
        });
        let report = analyzed(&value);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        let primary = &report.alternatives[0];
        assert!(primary.unresolved_goals.iter().any(|goal| goal == "G_REVERSE"));
    }

    #[test]
    fn double_discharge_is_rejected() {
        let mut value = plan_value();
        find_action_mut(&mut value, "A_FORWARD")["action_spec"]["input_goal_ids"] =
            serde_json::json!(["G_FORWARD", "G_REVERSE"]);
        value["edges"].as_array_mut().unwrap().push(serde_json::json!({
            "source_id":"A_FORWARD","target_id":"G_REVERSE",
            "relation":"discharges","evidence_refs":["E_SOURCE"]
        }));
        let report = analyzed(&value);
        assert!(report.errors.iter().any(|error| error.contains("consumed 2 times")));
    }

    #[test]
    fn orphan_create_edge_not_listed_by_action_is_rejected() {
        let mut value = plan_value();
        value["edges"].as_array_mut().unwrap().push(serde_json::json!({
            "source_id":"A_FORWARD","target_id":"G_REVERSE",
            "relation":"creates_subgoal","evidence_refs":["E_SOURCE"]
        }));
        let report = analyzed(&value);
        assert!(report.errors.iter().any(|error| error.contains("output_subgoal_ids do not exactly match")));
        assert!(report.errors.iter().any(|error| error.contains("producing actions")));
    }

    #[test]
    fn subgoal_without_producer_is_rejected() {
        let mut value = plan_value();
        value["nodes"].as_array_mut().unwrap().push(serde_json::json!({
            "id":"G_ORPHAN","kind":"subgoal","label":"orphan",
            "statement":"True","epistemic_class":"mathematical_interpretation",
            "evidence_refs":["E_SOURCE"]
        }));
        value["alternatives"][0]["node_ids"]
            .as_array_mut().unwrap().push(Value::String("G_ORPHAN".to_owned()));
        let report = analyzed(&value);
        assert!(report.errors.iter().any(|error| error.contains("has no producing action")));
    }

    #[test]
    fn transformed_goal_without_successor_is_rejected() {
        let mut value = plan_value();
        let edges = value["edges"].as_array_mut().unwrap();
        for edge in edges {
            if edge["source_id"] == "A_FORWARD"
                && edge["target_id"] == "G_FORWARD"
                && edge["relation"] == "discharges"
            {
                edge["relation"] = Value::String("transforms".to_owned());
            }
        }
        let report = analyzed(&value);
        assert!(report.errors.iter().any(|error| error.contains("without creating a successor")));
    }

    #[test]
    fn accepted_plan_cannot_retain_unresolved_obligations() {
        let mut value = plan_value();
        value["status"] = Value::String("accepted".to_owned());
        value["edges"].as_array_mut().unwrap().retain(|edge| {
            !(edge["source_id"] == "A_REVERSE"
                && edge["target_id"] == "G_REVERSE"
                && edge["relation"] == "discharges")
        });
        let report = analyzed(&value);
        assert!(report.errors.iter().any(|error| error.contains("unresolved primary-route obligations")));
    }

    #[test]
    fn multiple_primary_alternatives_block_acceptance() {
        let mut plan = fixture();
        let mut second = plan.alternatives[0].clone();
        second.id = "ALT_PRIMARY_2".to_owned();
        plan.alternatives.push(second);
        plan.status = PlanStatus::Accepted;
        let report = analyze_global_obligations(&plan);
        assert!(report.errors.iter().any(|error| error.contains("exactly one primary alternative")));
    }

    #[test]
    fn goal_transition_cycle_is_detected() {
        let mut value = plan_value();
        value["nodes"].as_array_mut().unwrap().extend([
            serde_json::json!({
                "id":"G_LOOP_A","kind":"subgoal","label":"loop A",
                "statement":"True","epistemic_class":"mathematical_interpretation",
                "evidence_refs":["E_SOURCE"]
            }),
            serde_json::json!({
                "id":"G_LOOP_B","kind":"subgoal","label":"loop B",
                "statement":"True","epistemic_class":"mathematical_interpretation",
                "evidence_refs":["E_SOURCE"]
            }),
            serde_json::json!({
                "id":"A_CYCLE_A","kind":"plan_action","label":"cycle A",
                "epistemic_class":"mathematical_interpretation","evidence_refs":["E_SOURCE"],
                "action_spec":{
                    "operation":"normalize","input_goal_ids":["G_LOOP_B"],
                    "output_subgoal_ids":["G_LOOP_A"],"consumed_side_condition_ids":[],
                    "justification_node_ids":["J_IFF"],"rationale":"synthetic cycle control",
                    "evidence_refs":["E_SOURCE"]
                }
            }),
            serde_json::json!({
                "id":"A_CYCLE_B","kind":"plan_action","label":"cycle B",
                "epistemic_class":"mathematical_interpretation","evidence_refs":["E_SOURCE"],
                "action_spec":{
                    "operation":"normalize","input_goal_ids":["G_LOOP_A"],
                    "output_subgoal_ids":["G_LOOP_B"],"consumed_side_condition_ids":[],
                    "justification_node_ids":["J_IFF"],"rationale":"synthetic cycle control",
                    "evidence_refs":["E_SOURCE"]
                }
            })
        ]);
        value["alternatives"][0]["node_ids"].as_array_mut().unwrap().extend([
            Value::String("G_LOOP_A".to_owned()),
            Value::String("G_LOOP_B".to_owned()),
            Value::String("A_CYCLE_A".to_owned()),
            Value::String("A_CYCLE_B".to_owned())
        ]);
        value["edges"].as_array_mut().unwrap().extend([
            serde_json::json!({
                "source_id":"A_CYCLE_A","target_id":"G_LOOP_B",
                "relation":"transforms","evidence_refs":["E_SOURCE"]
            }),
            serde_json::json!({
                "source_id":"A_CYCLE_A","target_id":"G_LOOP_A",
                "relation":"creates_subgoal","evidence_refs":["E_SOURCE"]
            }),
            serde_json::json!({
                "source_id":"A_CYCLE_B","target_id":"G_LOOP_A",
                "relation":"transforms","evidence_refs":["E_SOURCE"]
            }),
            serde_json::json!({
                "source_id":"A_CYCLE_B","target_id":"G_LOOP_B",
                "relation":"creates_subgoal","evidence_refs":["E_SOURCE"]
            })
        ]);
        let report = analyzed(&value);
        assert!(report.errors.iter().any(|error| error.contains("goal-transition cycle")));
    }
}}
