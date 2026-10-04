use std::collections::BTreeMap;

use symthaea_communication::{
    compare_graphs, drop_last_edge, duplicate_last_edge, graph_hash, rename_identifiers,
    reorder_collections, ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph,
    InterlinguaBenchmarkCase, InterlinguaPerturbation,
};

fn fixture() -> GroundedConceptGraph {
    GroundedConceptGraph {
        nodes: vec![
            ConceptNode {
                id: "agent-1".into(),
                kind: ConceptKind::Agent,
                label: Some("sender".into()),
                grounded_by: vec!["obs-1".into()],
                confidence: 0.96,
            },
            ConceptNode {
                id: "event-1".into(),
                kind: ConceptKind::Event,
                label: Some("approach".into()),
                grounded_by: vec!["obs-2".into()],
                confidence: 0.91,
            },
            ConceptNode {
                id: "object-1".into(),
                kind: ConceptKind::Object,
                label: Some("target".into()),
                grounded_by: vec!["obs-3".into()],
                confidence: 0.88,
            },
        ],
        edges: vec![
            ConceptEdge {
                source: "agent-1".into(),
                relation: "initiates".into(),
                target: "event-1".into(),
                evidence_ids: vec!["obs-2".into()],
                confidence: 0.91,
            },
            ConceptEdge {
                source: "event-1".into(),
                relation: "targets".into(),
                target: "object-1".into(),
                evidence_ids: vec!["obs-3".into()],
                confidence: 0.88,
            },
        ],
    }
}

fn main() -> Result<(), String> {
    let expected = fixture();

    let cases = [
        (
            "exact",
            1001,
            InterlinguaPerturbation::Exact,
            expected.clone(),
        ),
        (
            "reordered",
            1002,
            InterlinguaPerturbation::ReorderedCollections,
            reorder_collections(&expected),
        ),
        (
            "renamed-identifiers",
            1003,
            InterlinguaPerturbation::RenamedIdentifiers,
            rename_identifiers(&expected, "r"),
        ),
        (
            "missing-edge",
            1004,
            InterlinguaPerturbation::MissingEdge,
            drop_last_edge(&expected),
        ),
        (
            "duplicate-edge",
            1005,
            InterlinguaPerturbation::DuplicateEdge,
            duplicate_last_edge(&expected),
        ),
    ];

    let mut report = Vec::with_capacity(cases.len());
    for (case_id, seed, perturbation, observed) in cases {
        let metrics = compare_graphs(&expected, &observed)?;
        report.push(InterlinguaBenchmarkCase {
            case_id: case_id.into(),
            seed,
            perturbation,
            metrics,
        });
    }

    let expected_hash = graph_hash(&expected)?;

    let mut summary = BTreeMap::new();
    summary.insert("benchmark_schema_version", 1_u32);
    summary.insert("cases", report.len() as u32);
    summary.insert(
        "structurally_equivalent_cases",
        report
            .iter()
            .filter(|case_| case_.metrics.structural_equivalence)
            .count() as u32,
    );

    let exact_ok = report.iter().find(|case_| case_.case_id == "exact").map(|case_| case_.metrics.structural_equivalence).unwrap_or(false);
    let reordered_ok = report.iter().find(|case_| case_.case_id == "reordered").map(|case_| case_.metrics.structural_equivalence).unwrap_or(false);
    let renamed_ok = report.iter().find(|case_| case_.case_id == "renamed-identifiers").map(|case_| case_.metrics.structural_equivalence).unwrap_or(false);
    let missing_rejected = report.iter().find(|case_| case_.case_id == "missing-edge").map(|case_| !case_.metrics.structural_equivalence && case_.metrics.edge_recall < 1.0).unwrap_or(false);
    let duplicate_rejected = report.iter().find(|case_| case_.case_id == "duplicate-edge").map(|case_| !case_.metrics.structural_equivalence && case_.metrics.edge_precision < 1.0).unwrap_or(false);
    if !(exact_ok && reordered_ok && renamed_ok && missing_rejected && duplicate_rejected) {
        return Err("interlingua N0 acceptance assertions failed".into());
    }

    let output = serde_json::json!({
        "benchmark": "neurosemantic-interlingua-n0",
        "claim_boundary": "synthetic_structural_preservation_only",
        "expected_graph_hash": expected_hash,
        "summary": summary,
        "cases": report,
    });

    println!("{}", serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?);
    Ok(())
}
