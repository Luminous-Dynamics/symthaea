// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_neuroarch_types::{
    BudgetClass, CircuitDescriptor, CircuitId, CircuitImplementation, EdgeDescriptor, EdgeDirection,
    EdgeId, EdgeTransform, InputMergePolicy, RecurrenceKind, SemanticChannel, TimescaleClass,
    TopologyDescriptor, TopologyError, TOPOLOGY_SCHEMA_VERSION,
};

fn circuit(id: u32, role: &str) -> CircuitDescriptor {
    CircuitDescriptor {
        id: CircuitId(id),
        role: role.to_string(),
        timescale_class: TimescaleClass::Medium,
        state_dimension: 128,
        unit_count: 2,
        implementation: CircuitImplementation::Named("fixture".to_string()),
        input_merge_policy: InputMergePolicy::Single,
        modulation_profile: None,
    }
}

fn edge(id: u64, source: u32, target: u32, channel: &str) -> EdgeDescriptor {
    EdgeDescriptor {
        id: EdgeId(id),
        source: CircuitId(source),
        target: CircuitId(target),
        channel: SemanticChannel::Named(channel.to_string()),
        direction: EdgeDirection::Directed,
        recurrence: RecurrenceKind::FeedForward,
        budget_class: BudgetClass::Local,
        transform: EdgeTransform::Direct,
    }
}

fn topology() -> TopologyDescriptor {
    TopologyDescriptor {
        schema_version: TOPOLOGY_SCHEMA_VERSION,
        circuits: vec![circuit(1, "a"), circuit(2, "b"), circuit(3, "c")],
        edges: vec![edge(10, 1, 2, "x"), edge(11, 2, 3, "y")],
        allow_parallel_channels: false,
    }
}

fn assert_commitment_changes(mutated: TopologyDescriptor) {
    let base = topology();
    assert_ne!(base.commitment().unwrap(), mutated.commitment().unwrap());
}

fn decode_hex(input: &str) -> Vec<u8> {
    assert_eq!(input.len() % 2, 0);
    input
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let pair = std::str::from_utf8(pair).unwrap();
            u8::from_str_radix(pair, 16).unwrap()
        })
        .collect()
}

#[test]
fn canonical_v1_bytes_match_golden() {
    let expected = decode_hex(concat!(
        "73796d74686165612d6e6575726f617263682d746f706f6c6f67792d763100010000030000000100",
        "00000100000061028000000000000000020000000000000002070000006669787475726501000200",
        "00000100000062028000000000000000020000000000000002070000006669787475726501000300",
        "00000100000063028000000000000000020000000000000002070000006669787475726501000200",
        "00000a000000000000000100000002000000030100000078000001000b0000000000000002000000",
        "0300000003010000007900000100",
    ));
    assert_eq!(topology().canonical_bytes().unwrap(), expected);
}

#[test]
fn rejects_duplicate_and_dangling_identity() {
    let mut duplicate_circuit = topology();
    duplicate_circuit.circuits[1].id = CircuitId(1);
    assert_eq!(
        duplicate_circuit.validate(),
        Err(TopologyError::DuplicateCircuitId(CircuitId(1)))
    );

    let mut duplicate_edge = topology();
    duplicate_edge.edges[1].id = EdgeId(10);
    assert_eq!(
        duplicate_edge.validate(),
        Err(TopologyError::DuplicateEdgeId(EdgeId(10)))
    );

    let mut dangling_source = topology();
    dangling_source.edges[0].source = CircuitId(999);
    assert!(matches!(
        dangling_source.validate(),
        Err(TopologyError::DanglingSource { .. })
    ));

    let mut dangling_target = topology();
    dangling_target.edges[0].target = CircuitId(999);
    assert!(matches!(
        dangling_target.validate(),
        Err(TopologyError::DanglingTarget { .. })
    ));
}

#[test]
fn recurrent_self_routes_are_explicit() {
    let mut forbidden = topology();
    forbidden.edges[0].target = forbidden.edges[0].source;
    assert_eq!(
        forbidden.validate(),
        Err(TopologyError::ForbiddenSelfEdge(EdgeId(10)))
    );

    let mut recurrent = topology();
    recurrent.edges[0].target = recurrent.edges[0].source;
    recurrent.edges[0].recurrence = RecurrenceKind::Recurrent;
    assert!(recurrent.validate().is_ok());
}

#[test]
fn parallel_channel_policy_is_fail_closed() {
    let mut denied = topology();
    denied.edges.push(edge(12, 1, 2, "z"));
    assert!(matches!(
        denied.validate(),
        Err(TopologyError::DuplicateStructuralEdge { .. })
    ));

    let mut allowed = topology();
    allowed.allow_parallel_channels = true;
    allowed.edges.push(edge(12, 1, 2, "z"));
    assert!(allowed.validate().is_ok());

    let mut duplicate_channel = topology();
    duplicate_channel.allow_parallel_channels = true;
    duplicate_channel.edges.push(edge(12, 1, 2, "x"));
    assert!(matches!(
        duplicate_channel.validate(),
        Err(TopologyError::DuplicateChannelEdge { .. })
    ));
}

#[test]
fn rejects_invalid_schema_shape_and_timescale() {
    let mut bad_schema = topology();
    bad_schema.schema_version += 1;
    assert!(matches!(
        bad_schema.validate(),
        Err(TopologyError::UnsupportedSchemaVersion(_))
    ));

    let no_compute = TopologyDescriptor {
        schema_version: TOPOLOGY_SCHEMA_VERSION,
        circuits: vec![CircuitDescriptor {
            id: CircuitId(0),
            role: "input".to_string(),
            timescale_class: TimescaleClass::Stateless,
            state_dimension: 128,
            unit_count: 1,
            implementation: CircuitImplementation::ExternalInput,
            input_merge_policy: InputMergePolicy::None,
            modulation_profile: None,
        }],
        edges: vec![],
        allow_parallel_channels: false,
    };
    assert_eq!(
        no_compute.validate(),
        Err(TopologyError::NoComputationalCircuits)
    );

    let mut zero_state = topology();
    zero_state.circuits[0].state_dimension = 0;
    assert_eq!(
        zero_state.validate(),
        Err(TopologyError::InvalidStateDimension(CircuitId(1)))
    );

    let mut zero_units = topology();
    zero_units.circuits[0].unit_count = 0;
    assert_eq!(
        zero_units.validate(),
        Err(TopologyError::InvalidUnitCount(CircuitId(1)))
    );

    let mut zero_tau = topology();
    zero_tau.circuits[0].timescale_class = TimescaleClass::CustomNanos(0);
    assert_eq!(zero_tau.validate(), Err(TopologyError::InvalidTimescale));
}

#[test]
fn every_v1_circuit_identity_field_is_commitment_bound() {
    let mut mutated = topology();
    mutated.circuits[0].role = "different".to_string();
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.circuits[0].timescale_class = TimescaleClass::Fast;
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.circuits[0].state_dimension += 1;
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.circuits[0].unit_count += 1;
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.circuits[0].implementation = CircuitImplementation::Named("other".to_string());
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.circuits[0].input_merge_policy = InputMergePolicy::BundleAll;
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.circuits[0].modulation_profile = Some("mod:a".to_string());
    assert_commitment_changes(mutated);
}

#[test]
fn every_v1_edge_identity_field_is_commitment_bound() {
    let mut mutated = topology();
    mutated.edges[0].source = CircuitId(3);
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.edges[0].target = CircuitId(3);
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.edges[0].channel = SemanticChannel::Named("other".to_string());
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.edges[0].direction = EdgeDirection::Bidirectional;
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.edges[0].recurrence = RecurrenceKind::Recurrent;
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.edges[0].budget_class = BudgetClass::Global;
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.edges[0].transform = EdgeTransform::Bind;
    assert_commitment_changes(mutated);

    let mut mutated = topology();
    mutated.allow_parallel_channels = true;
    assert_commitment_changes(mutated);
}
