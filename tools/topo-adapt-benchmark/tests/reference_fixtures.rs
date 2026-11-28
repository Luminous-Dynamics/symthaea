use topo_adapt_benchmark::{evaluate_episode, Episode, Fixture, FlowResult, FIXTURE_JSON};

fn fixture() -> Fixture {
    let fixture = Fixture::parse(FIXTURE_JSON).expect("fixture parses");
    fixture.validate().expect("fixture validates");
    fixture
}

fn episode<'a>(fixture: &'a Fixture, id: &str) -> &'a Episode {
    fixture.cases.iter().find(|c| c.id == id).expect("case exists")
}

#[test]
fn stationary_demand_prefers_no_edit() {
    let fixture = fixture();
    let case = episode(&fixture, "stationary_demand_control");

    let no_edit = evaluate_episode(case, &fixture.graph_semantics, 0, &[]).unwrap();
    let unnecessary_edit = evaluate_episode(
        case,
        &fixture.graph_semantics,
        0,
        &["useful_only_if_sink_6_activates"],
    )
    .unwrap();

    assert_eq!(no_edit, FlowResult { max_flow: 6, min_cut: 6, requested_flow: 6, accepted_edge_additions: 0, edit_cost_units: 0, edge_maintenance_cost_units: 5 });
    assert_eq!(unnecessary_edit.max_flow, no_edit.max_flow);
    assert_eq!(unnecessary_edit.min_cut, no_edit.min_cut);
    assert_eq!(unnecessary_edit.accepted_edge_additions, 1);
    assert_eq!(unnecessary_edit.edit_cost_units, 1);
    assert_eq!(unnecessary_edit.edge_maintenance_cost_units, 6);
    assert!(unnecessary_edit.edit_cost_units > no_edit.edit_cost_units);
}

#[test]
fn useful_new_sink_edge_restores_full_capacity_feasible_service() {
    let fixture = fixture();
    let case = episode(&fixture, "new_sink_demand_shift");

    let no_edit = evaluate_episode(case, &fixture.graph_semantics, 1, &[]).unwrap();
    let useful = evaluate_episode(
        case,
        &fixture.graph_semantics,
        1,
        &["connect_new_sink_path"],
    )
    .unwrap();
    let decoy = evaluate_episode(case, &fixture.graph_semantics, 1, &["decoy_dead_end_7"]).unwrap();

    assert_eq!(no_edit, FlowResult { max_flow: 6, min_cut: 6, requested_flow: 8, accepted_edge_additions: 0, edit_cost_units: 0, edge_maintenance_cost_units: 5 });
    assert_eq!(useful, FlowResult { max_flow: 8, min_cut: 8, requested_flow: 8, accepted_edge_additions: 1, edit_cost_units: 1, edge_maintenance_cost_units: 6 });
    assert_eq!(decoy.max_flow, no_edit.max_flow);
    assert_eq!(decoy.min_cut, no_edit.min_cut);
    assert_eq!(decoy.edit_cost_units, 1);
    assert_eq!(decoy.edge_maintenance_cost_units, 6);
}

#[test]
fn alternate_edge_restores_service_after_bridge_failure() {
    let fixture = fixture();
    let case = episode(&fixture, "bridge_failure_reroute");

    let no_edit = evaluate_episode(case, &fixture.graph_semantics, 1, &[]).unwrap();
    let useful = evaluate_episode(case, &fixture.graph_semantics, 1, &["restore_sink_route"]).unwrap();
    let decoy = evaluate_episode(case, &fixture.graph_semantics, 1, &["decoy_dead_end_4"]).unwrap();

    assert_eq!(no_edit, FlowResult { max_flow: 0, min_cut: 0, requested_flow: 4, accepted_edge_additions: 0, edit_cost_units: 0, edge_maintenance_cost_units: 3 });
    assert_eq!(useful, FlowResult { max_flow: 4, min_cut: 4, requested_flow: 4, accepted_edge_additions: 1, edit_cost_units: 1, edge_maintenance_cost_units: 4 });
    assert_eq!(decoy.max_flow, no_edit.max_flow);
    assert_eq!(decoy.min_cut, no_edit.min_cut);
    assert_eq!(decoy.edit_cost_units, 1);
    assert_eq!(decoy.edge_maintenance_cost_units, 4);
}

#[test]
fn node_relabeling_preserves_flow_and_cut() {
    let fixture = fixture();
    let mut case = episode(&fixture, "new_sink_demand_shift").clone();
    for node in &mut case.nodes {
        node.id += 100;
    }
    for edge in case.active_edges.iter_mut().chain(case.candidate_edges.iter_mut()) {
        edge.from += 100;
        edge.to += 100;
    }
    for step in &mut case.demand_schedule {
        for sink in &mut step.sinks {
            sink.node += 100;
        }
    }
    for intervention in &mut case.interventions {
        intervention.from = intervention.from.map(|x| x + 100);
        intervention.to = intervention.to.map(|x| x + 100);
    }

    let result = evaluate_episode(&case, &fixture.graph_semantics, 1, &["connect_new_sink_path"]).unwrap();
    assert_eq!(result, FlowResult { max_flow: 8, min_cut: 8, requested_flow: 8, accepted_edge_additions: 1, edit_cost_units: 1, edge_maintenance_cost_units: 6 });
}

#[test]
fn edge_order_does_not_change_outcome() {
    let fixture = fixture();
    let mut case = episode(&fixture, "bridge_failure_reroute").clone();
    let baseline = evaluate_episode(
        &case,
        &fixture.graph_semantics,
        1,
        &["restore_sink_route"],
    )
    .unwrap();

    case.active_edges.reverse();
    case.candidate_edges.reverse();
    let reordered = evaluate_episode(
        &case,
        &fixture.graph_semantics,
        1,
        &["restore_sink_route"],
    )
    .unwrap();
    assert_eq!(baseline, reordered);
}

#[test]
fn uniform_capacity_scaling_scales_flow_and_cut() {
    let fixture = fixture();
    let mut case = episode(&fixture, "new_sink_demand_shift").clone();
    for node in &mut case.nodes {
        node.supply = node.supply.map(|x| x * 2);
        node.demand = node.demand.map(|x| x * 2);
    }
    for edge in case.active_edges.iter_mut().chain(case.candidate_edges.iter_mut()) {
        edge.capacity *= 2;
    }
    for step in &mut case.demand_schedule {
        for sink in &mut step.sinks {
            sink.demand *= 2;
        }
    }
    for intervention in &mut case.interventions {
        intervention.capacity_after = intervention.capacity_after.map(|x| x * 2);
    }

    let result = evaluate_episode(
        &case,
        &fixture.graph_semantics,
        1,
        &["connect_new_sink_path"],
    )
    .unwrap();
    assert_eq!(result.max_flow, 16);
    assert_eq!(result.min_cut, 16);
    assert_eq!(result.requested_flow, 16);
    assert_eq!(result.accepted_edge_additions, 1);
    assert_eq!(result.edit_cost_units, 1);
    assert_eq!(result.edge_maintenance_cost_units, 6);
}

#[test]
fn edit_budget_and_unknown_candidate_fail_closed() {
    let fixture = fixture();
    let case = episode(&fixture, "new_sink_demand_shift");

    assert!(evaluate_episode(
        case,
        &fixture.graph_semantics,
        1,
        &["connect_new_sink_path", "decoy_dead_end_7"],
    )
    .is_err());

    assert!(evaluate_episode(case, &fixture.graph_semantics, 1, &["unknown-edge"]).is_err());
}

#[test]
fn malformed_graphs_and_costs_fail_closed() {
    let original = fixture();

    let mut duplicate_node = episode(&original, "stationary_demand_control").clone();
    duplicate_node.nodes[1].id = duplicate_node.nodes[0].id;
    assert!(Fixture {
        schema_version: original.schema_version.clone(),
        graph_semantics: original.graph_semantics.clone(),
        cases: vec![duplicate_node],
    }
    .validate()
    .is_err());

    let mut malformed_cost = episode(&original, "new_sink_demand_shift").clone();
    malformed_cost.candidate_edges[0].edit_cost = Some(2);
    assert!(Fixture {
        schema_version: original.schema_version.clone(),
        graph_semantics: original.graph_semantics.clone(),
        cases: vec![malformed_cost],
    }
    .validate()
    .is_err());

    let mut invalid_endpoint = episode(&original, "bridge_failure_reroute").clone();
    invalid_endpoint.candidate_edges[0].to = u16::MAX;
    assert!(Fixture {
        schema_version: original.schema_version.clone(),
        graph_semantics: original.graph_semantics.clone(),
        cases: vec![invalid_endpoint],
    }
    .validate()
    .is_err());
}
