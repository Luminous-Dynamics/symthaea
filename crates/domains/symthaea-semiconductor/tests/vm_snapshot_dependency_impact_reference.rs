#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use symthaea_semiconductor::benchmark::{
    LabelConsumptionEligibilityV1, VM_DEPENDENCY_EDGE_SCHEMA_V1, VM_DEPENDENCY_GRAPH_SCHEMA_V1,
    VM_IMPACT_CLOSURE_INPUT_SCHEMA_V1, VmDependencyDimensionV1, VmImpactClosureInputV1,
    VmImpactClosureStatusV1, VmModelDerivationKindV1, VmModelSnapshotNodeV1,
    VmSnapshotCurrentDispositionV1, VmSnapshotDependencyEdgeV1, VmSnapshotDependencyGraphV1,
    VmSnapshotImpactStateV1, VmSnapshotLabelCurrentStatusV1, VmSnapshotPurposeV1,
    VmSnapshotRequalificationReportV1, compute_vm_snapshot_impact_closure_v1,
    seal_vm_snapshot_dependency_edge_v1, seal_vm_snapshot_dependency_graph_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn sha256_json<T: Serialize + ?Sized>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn node(
    id: &str,
    kind: VmModelDerivationKindV1,
    purpose: VmSnapshotPurposeV1,
    artifact: char,
    manifest: char,
) -> VmModelSnapshotNodeV1 {
    VmModelSnapshotNodeV1 {
        snapshot_id: id.into(),
        artifact_sha256: digest(artifact),
        manifest_sha256: digest(manifest),
        purpose,
        derivation_kind: kind,
    }
}

fn edge(
    id: &str,
    parent: &VmModelSnapshotNodeV1,
    child: &VmModelSnapshotNodeV1,
    dimensions: Vec<VmDependencyDimensionV1>,
) -> VmSnapshotDependencyEdgeV1 {
    let mut value = VmSnapshotDependencyEdgeV1 {
        schema: VM_DEPENDENCY_EDGE_SCHEMA_V1.into(),
        edge_id: id.into(),
        parent_snapshot_id: parent.snapshot_id.clone(),
        child_snapshot_id: child.snapshot_id.clone(),
        parent_artifact_sha256: parent.artifact_sha256.clone(),
        inherited_dimensions: dimensions,
        derivation_evidence_sha256: digest('e'),
        edge_sha256: String::new(),
    };
    value.edge_sha256 = seal_vm_snapshot_dependency_edge_v1(&value).unwrap();
    value
}

fn graph(
    nodes: Vec<VmModelSnapshotNodeV1>,
    edges: Vec<VmSnapshotDependencyEdgeV1>,
) -> VmSnapshotDependencyGraphV1 {
    let mut value = VmSnapshotDependencyGraphV1 {
        schema: VM_DEPENDENCY_GRAPH_SCHEMA_V1.into(),
        graph_id: "vm-independent-oracle-dag-v1".into(),
        inventory_complete: true,
        ordering_complete: true,
        nodes,
        edges,
        graph_sha256: String::new(),
    };
    value.graph_sha256 = seal_vm_snapshot_dependency_graph_v1(&value).unwrap();
    value
}

fn requalification(
    snapshot: &VmModelSnapshotNodeV1,
    disposition: VmSnapshotCurrentDispositionV1,
) -> VmSnapshotRequalificationReportV1 {
    let mut report = VmSnapshotRequalificationReportV1 {
        schema: "luminous.semi-bench.vm-snapshot-requalification.v1".into(),
        snapshot_id: snapshot.snapshot_id.clone(),
        historical_manifest_sha256: snapshot.manifest_sha256.clone(),
        requalified_at_k1_unix_seconds: 1_880_000_000,
        current_disposition: disposition,
        consumed_label_count: 1,
        current_eligible_count: usize::from(
            disposition == VmSnapshotCurrentDispositionV1::CurrentEligible,
        ),
        current_unknown_count: usize::from(
            disposition == VmSnapshotCurrentDispositionV1::Indeterminate,
        ),
        current_blocked_count: usize::from(matches!(
            disposition,
            VmSnapshotCurrentDispositionV1::RebuildRequired
                | VmSnapshotCurrentDispositionV1::HistoricalOnly
        )),
        label_statuses: vec![VmSnapshotLabelCurrentStatusV1 {
            label_id: format!("label-{}", snapshot.snapshot_id),
            k0_projection_sha256: digest('k'),
            k1_eligibility: match disposition {
                VmSnapshotCurrentDispositionV1::CurrentEligible => {
                    symthaea_semiconductor::benchmark::LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
                }
                VmSnapshotCurrentDispositionV1::Indeterminate => {
                    symthaea_semiconductor::benchmark::LabelConsumptionEligibilityV1::Unknown
                }
                VmSnapshotCurrentDispositionV1::RebuildRequired
                | VmSnapshotCurrentDispositionV1::HistoricalOnly => {
                    symthaea_semiconductor::benchmark::LabelConsumptionEligibilityV1::Blocked
                }
            },
            k1_report_sha256: digest('r'),
            reasons: Vec::new(),
        }],
        historical_manifest_rewritten: false,
        training_snapshot_mutated: false,
        model_retrained: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    report.report_sha256 = sha256_json(&canonical);
    report
}

fn frozen_fixture() -> VmImpactClosureInputV1 {
    let root_a = node(
        "root-a",
        VmModelDerivationKindV1::IndependentRetrain,
        VmSnapshotPurposeV1::Training,
        'a',
        '1',
    );
    let root_b = node(
        "root-b",
        VmModelDerivationKindV1::IndependentRetrain,
        VmSnapshotPurposeV1::Training,
        'b',
        '2',
    );
    let child = node(
        "child",
        VmModelDerivationKindV1::FineTuneFromParent,
        VmSnapshotPurposeV1::Training,
        'c',
        '3',
    );
    let ensemble = node(
        "ensemble",
        VmModelDerivationKindV1::EnsembleOfParents,
        VmSnapshotPurposeV1::Training,
        'd',
        '4',
    );
    let evaluation = node(
        "evaluation",
        VmModelDerivationKindV1::EvaluationOnly,
        VmSnapshotPurposeV1::Evaluation,
        'e',
        '5',
    );
    let independent = node(
        "independent",
        VmModelDerivationKindV1::IndependentRetrain,
        VmSnapshotPurposeV1::Training,
        'f',
        '6',
    );

    let edges = vec![
        edge(
            "root-a-child",
            &root_a,
            &child,
            vec![VmDependencyDimensionV1::ModelWeights],
        ),
        edge(
            "child-ensemble",
            &child,
            &ensemble,
            vec![VmDependencyDimensionV1::ModelWeights],
        ),
        edge(
            "root-b-ensemble",
            &root_b,
            &ensemble,
            vec![VmDependencyDimensionV1::ModelWeights],
        ),
        edge(
            "root-a-evaluation",
            &root_a,
            &evaluation,
            vec![VmDependencyDimensionV1::EvaluationTarget],
        ),
        edge(
            "root-a-independent-reference",
            &root_a,
            &independent,
            vec![VmDependencyDimensionV1::FeatureSchema],
        ),
    ];

    let snapshots = vec![root_a, root_b, child, ensemble, evaluation, independent];
    let mut reports = Vec::new();
    for snapshot in &snapshots {
        let disposition = match snapshot.snapshot_id.as_str() {
            "root-a" => VmSnapshotCurrentDispositionV1::RebuildRequired,
            "root-b" => VmSnapshotCurrentDispositionV1::Indeterminate,
            _ => VmSnapshotCurrentDispositionV1::CurrentEligible,
        };
        reports.push(requalification(snapshot, disposition));
    }

    VmImpactClosureInputV1 {
        schema: VM_IMPACT_CLOSURE_INPUT_SCHEMA_V1.into(),
        input_id: "frozen-independent-closure-fixture-v1".into(),
        graph: graph(snapshots, edges),
        snapshot_requalification_reports: reports,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum OracleRisk {
    None,
    Historical,
    Unknown,
    Blocked,
}

fn direct_risk(disposition: VmSnapshotCurrentDispositionV1) -> OracleRisk {
    match disposition {
        VmSnapshotCurrentDispositionV1::CurrentEligible => OracleRisk::None,
        VmSnapshotCurrentDispositionV1::RebuildRequired => OracleRisk::Blocked,
        VmSnapshotCurrentDispositionV1::Indeterminate => OracleRisk::Unknown,
        VmSnapshotCurrentDispositionV1::HistoricalOnly => OracleRisk::Historical,
    }
}

fn propagate(parent: OracleRisk, dimensions: &[VmDependencyDimensionV1]) -> OracleRisk {
    let mut risk = OracleRisk::None;
    if dimensions.iter().any(|dimension| {
        matches!(
            dimension,
            VmDependencyDimensionV1::ModelWeights | VmDependencyDimensionV1::CalibrationState
        )
    }) {
        risk = risk.max(parent);
    }
    if dimensions.contains(&VmDependencyDimensionV1::EvaluationTarget) {
        risk = risk.max(match parent {
            OracleRisk::Blocked | OracleRisk::Historical => OracleRisk::Historical,
            other => other,
        });
    }
    risk
}

/// Independent fixed-point relaxation over the graph; it does not call the
/// production topological sort or risk-propagation functions.
fn independent_risk_oracle(
    input: &VmImpactClosureInputV1,
) -> BTreeMap<String, (OracleRisk, OracleRisk)> {
    let mut direct = BTreeMap::new();
    let mut inherited: BTreeMap<String, OracleRisk> = BTreeMap::new();

    for report in &input.snapshot_requalification_reports {
        direct.insert(
            report.snapshot_id.clone(),
            direct_risk(report.current_disposition),
        );
        inherited.insert(report.snapshot_id.clone(), OracleRisk::None);
    }

    let mut changed = true;
    let mut iterations = 0;
    while changed {
        changed = false;
        iterations += 1;
        assert!(
            iterations <= input.graph.nodes.len() + 1,
            "fixture graph must be acyclic"
        );

        for edge in &input.graph.edges {
            let parent = direct[&edge.parent_snapshot_id].max(inherited[&edge.parent_snapshot_id]);
            let candidate = propagate(parent, &edge.inherited_dimensions);
            let current = inherited[&edge.child_snapshot_id];
            if candidate > current {
                inherited.insert(edge.child_snapshot_id.clone(), candidate);
                changed = true;
            }
        }
    }

    direct
        .into_iter()
        .map(|(id, risk)| {
            let inherited_risk = inherited[&id];
            (id, (risk, inherited_risk))
        })
        .collect()
}

fn independent_label_sources(
    input: &VmImpactClosureInputV1,
) -> (
    BTreeMap<String, BTreeSet<String>>,
    BTreeMap<String, BTreeSet<String>>,
) {
    let mut direct: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut inherited: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut direct_risks = BTreeMap::new();
    let mut inherited_risks = BTreeMap::new();

    for report in &input.snapshot_requalification_reports {
        direct.insert(
            report.snapshot_id.clone(),
            report
                .label_statuses
                .iter()
                .filter(|label| {
                    label.k1_eligibility
                        != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
                })
                .map(|label| label.label_id.clone())
                .collect(),
        );
        inherited.insert(report.snapshot_id.clone(), BTreeSet::new());
        direct_risks.insert(
            report.snapshot_id.clone(),
            direct_risk(report.current_disposition),
        );
        inherited_risks.insert(report.snapshot_id.clone(), OracleRisk::None);
    }

    let mut changed = true;
    let mut iterations = 0;
    while changed {
        changed = false;
        iterations += 1;
        assert!(
            iterations <= input.graph.nodes.len() + 1,
            "fixture graph must be acyclic"
        );
        for edge in &input.graph.edges {
            let parent_risk = direct_risks[&edge.parent_snapshot_id]
                .max(inherited_risks[&edge.parent_snapshot_id]);
            let propagated_risk = propagate(parent_risk, &edge.inherited_dimensions);
            if propagated_risk == OracleRisk::None {
                continue;
            }

            let parent_sources = direct[&edge.parent_snapshot_id]
                .union(&inherited[&edge.parent_snapshot_id])
                .cloned()
                .collect::<BTreeSet<_>>();
            let child_sources = inherited.entry(edge.child_snapshot_id.clone()).or_default();
            let before = child_sources.len();
            child_sources.extend(parent_sources);
            if child_sources.len() != before {
                changed = true;
            }
            let current = inherited_risks[&edge.child_snapshot_id];
            if propagated_risk > current {
                inherited_risks.insert(edge.child_snapshot_id.clone(), propagated_risk);
                changed = true;
            }
        }
    }

    (direct, inherited)
}

fn expected_state(direct: OracleRisk, inherited: OracleRisk) -> VmSnapshotImpactStateV1 {
    match direct.max(inherited) {
        OracleRisk::None => VmSnapshotImpactStateV1::Unaffected,
        OracleRisk::Blocked if direct == OracleRisk::Blocked => {
            VmSnapshotImpactStateV1::DirectRebuildRequired
        }
        OracleRisk::Blocked => VmSnapshotImpactStateV1::InheritedRebuildRequired,
        OracleRisk::Unknown if direct == OracleRisk::Unknown => {
            VmSnapshotImpactStateV1::DirectIndeterminate
        }
        OracleRisk::Unknown => VmSnapshotImpactStateV1::InheritedIndeterminate,
        OracleRisk::Historical => VmSnapshotImpactStateV1::HistoricalOnly,
    }
}

/// Independent DFS parent-before-child ordering check. We check edge constraints
/// rather than matching the implementation's exact tie-break order.
fn visit_parents(
    id: &str,
    parents: &BTreeMap<String, Vec<String>>,
    seen: &mut BTreeSet<String>,
    output: &mut Vec<String>,
) {
    if !seen.insert(id.to_string()) {
        return;
    }
    for parent in parents.get(id).into_iter().flatten() {
        visit_parents(parent, parents, seen, output);
    }
    output.push(id.to_string());
}

fn independent_topological_order(input: &VmImpactClosureInputV1) -> Vec<String> {
    let mut parents: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for edge in &input.graph.edges {
        parents
            .entry(edge.child_snapshot_id.clone())
            .or_default()
            .push(edge.parent_snapshot_id.clone());
    }
    for parent_ids in parents.values_mut() {
        parent_ids.sort();
    }
    let mut ids: Vec<String> = input
        .graph
        .nodes
        .iter()
        .map(|node| node.snapshot_id.clone())
        .collect();
    ids.sort();

    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for id in &ids {
        visit_parents(id, &parents, &mut seen, &mut output);
    }
    output
}

#[test]
fn frozen_dag_matches_independent_fixed_point_impact_oracle() {
    let input = frozen_fixture();
    let expected = independent_risk_oracle(&input);
    let (expected_direct_labels, expected_inherited_labels) = independent_label_sources(&input);
    let report = compute_vm_snapshot_impact_closure_v1(&input).unwrap();

    assert_eq!(report.status, VmImpactClosureStatusV1::Complete);
    let actual: BTreeMap<String, VmSnapshotImpactStateV1> = report
        .entries
        .iter()
        .map(|entry| (entry.snapshot_id.clone(), entry.state))
        .collect();

    for (snapshot_id, (direct, inherited)) in &expected {
        assert_eq!(
            actual[snapshot_id],
            expected_state(*direct, *inherited),
            "snapshot {snapshot_id}"
        );
        let entry = report
            .entries
            .iter()
            .find(|entry| entry.snapshot_id == *snapshot_id)
            .unwrap();
        assert_eq!(
            entry.direct_affected_label_ids,
            expected_direct_labels[snapshot_id]
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            "direct label lineage for {snapshot_id}"
        );
        assert_eq!(
            entry.inherited_affected_label_ids,
            expected_inherited_labels[snapshot_id]
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            "inherited label lineage for {snapshot_id}"
        );
    }

    let expected_direct: BTreeSet<String> = expected
        .iter()
        .filter(|(_, (direct, _))| *direct != OracleRisk::None)
        .map(|(id, _)| id.clone())
        .collect();
    let expected_transitive: BTreeSet<String> = expected
        .iter()
        .filter(|(_, (direct, inherited))| {
            *direct == OracleRisk::None && *inherited != OracleRisk::None
        })
        .map(|(id, _)| id.clone())
        .collect();

    assert_eq!(
        report
            .direct_impact_snapshot_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>(),
        expected_direct
    );
    assert_eq!(
        report
            .transitive_impact_snapshot_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>(),
        expected_transitive
    );

    let expected_rebuild: BTreeSet<String> = expected
        .iter()
        .filter(|(id, (direct, inherited))| {
            let purpose = input
                .graph
                .nodes
                .iter()
                .find(|node| node.snapshot_id == **id)
                .unwrap()
                .purpose;
            purpose == VmSnapshotPurposeV1::Training
                && matches!((*direct).max(*inherited), OracleRisk::Blocked)
        })
        .map(|(id, _)| id.clone())
        .collect();
    assert_eq!(
        report
            .rebuild_order_snapshot_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>(),
        expected_rebuild
    );

    let order = independent_topological_order(&input);
    let positions: BTreeMap<String, usize> = order
        .iter()
        .enumerate()
        .map(|(index, id)| (id.clone(), index))
        .collect();
    for edge in &input.graph.edges {
        assert!(
            positions[&edge.parent_snapshot_id] < positions[&edge.child_snapshot_id],
            "parent must precede child: {} -> {}",
            edge.parent_snapshot_id,
            edge.child_snapshot_id
        );
    }
}

#[test]
fn incomplete_inventory_is_unknown_even_when_visible_nodes_look_clean() {
    let mut input = frozen_fixture();
    input.graph.inventory_complete = false;
    input.graph.graph_sha256 = seal_vm_snapshot_dependency_graph_v1(&input.graph).unwrap();
    let report = compute_vm_snapshot_impact_closure_v1(&input).unwrap();
    assert_eq!(report.status, VmImpactClosureStatusV1::Unknown);
    assert!(!report.unresolved_snapshot_ids.is_empty());
    assert!(report.entries.is_empty());
}

#[test]
fn permuting_nodes_edges_and_reports_preserves_closure_receipt() {
    let mut input = frozen_fixture();
    let first = compute_vm_snapshot_impact_closure_v1(&input).unwrap();

    input.graph.nodes.reverse();
    input.graph.edges.reverse();
    input.graph.graph_sha256 = seal_vm_snapshot_dependency_graph_v1(&input.graph).unwrap();
    input.snapshot_requalification_reports.reverse();

    let second = compute_vm_snapshot_impact_closure_v1(&input).unwrap();
    assert_eq!(first.report_sha256, second.report_sha256);
}
