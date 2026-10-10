use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

use symthaea_communication::hdc_codec::HdcBinaryFrame;
use symthaea_communication::hdc_ontology::{
    HdcConceptIdentityBinding, HdcOntologyCodebook, HdcOntologyDecodePolicy,
    HdcOntologyEmpiricalCalibration, HdcOntologyManifest, HdcRelationIdentityBinding,
    HDC_ONTOLOGY_ADAPTER_ID, HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
};
use symthaea_communication::{ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph};
use symthaea_core::hdc::binary_hv::BinaryHV;

const SCHEME: &str = "scheme:empirical-calibration-n0-v1";
const RELATIONS: usize = 3;

fn node(index: usize, prefix: &str) -> ConceptNode {
    ConceptNode {
        id: format!("{prefix}-node-{index}"),
        kind: match index % 3 {
            0 => ConceptKind::Agent,
            1 => ConceptKind::Event,
            _ => ConceptKind::Object,
        },
        label: Some(format!("{prefix}-label-{index}")),
        grounded_by: vec![format!("{prefix}-grounding-{index}")],
        confidence: 1.0,
    }
}

fn make_graph(
    prefix: &str,
    node_ids: &[usize],
    edges: &[(usize, usize, usize)],
) -> (GroundedConceptGraph, HdcOntologyManifest) {
    let nodes = node_ids.iter().map(|index| node(*index, prefix)).collect::<Vec<_>>();
    let graph_edges = edges
        .iter()
        .map(|(source, relation, target)| ConceptEdge {
            source: format!("{prefix}-node-{source}"),
            relation: format!("{prefix}-relation-{relation}"),
            target: format!("{prefix}-node-{target}"),
            evidence_ids: Vec::new(),
            confidence: 1.0,
        })
        .collect::<Vec<_>>();
    let graph = GroundedConceptGraph { nodes, edges: graph_edges };

    let manifest = HdcOntologyManifest {
        schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
        scheme_id: SCHEME.into(),
        mapping_provenance_hash: symthaea_communication::content_hash(
            b"empirical-calibration-upstream-map-v1",
        ),
        concepts: graph
            .nodes
            .iter()
            .map(|node| HdcConceptIdentityBinding {
                node_id: node.id.clone(),
                concept_id: format!(
                    "concept:{:02}",
                    node.id.split('-').last().unwrap().parse::<usize>().unwrap()
                ),
                kind: node.kind.clone(),
                grounding_ids: node.grounded_by.clone(),
            })
            .collect(),
        relations: (0..RELATIONS)
            .map(|relation| HdcRelationIdentityBinding {
                local_relation: format!("{prefix}-relation-{relation}"),
                relation_id: format!("relation:{relation}"),
            })
            .collect(),
    };

    (graph, manifest)
}

fn expected_concepts(
    graph: &GroundedConceptGraph,
) -> BTreeMap<String, String> {
    graph
        .nodes
        .iter()
        .map(|node| {
            (
                node.id.clone(),
                format!(
                    "concept:{:02}",
                    node.id.split('-').last().unwrap().parse::<usize>().unwrap()
                ),
            )
        })
        .collect()
}

fn expected_relations(edges: &[(usize, usize, usize)]) -> Vec<String> {
    edges
        .iter()
        .map(|(_, relation, _)| format!("relation:{relation}"))
        .collect()
}

fn execution_revision() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|revision| revision.trim().to_string())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| "local".into())
}

fn main() -> Result<(), String> {
    let training = make_graph(
        "training",
        &(0..8).collect::<Vec<_>>(),
        &[
            (0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 0, 4),
            (4, 1, 5), (5, 2, 6), (6, 0, 7),
        ],
    );
    let codebook = HdcOntologyCodebook::from_training_graphs(
        0xCA1B_2026,
        &[training.0.clone()],
        &training.1,
    )?;

    let calibration_specs = [
        ("calib-a", &[0, 1, 2][..], &[(0, 0, 1), (1, 1, 2)][..]),
        ("calib-b", &[2, 3, 4, 5][..], &[(2, 2, 3), (3, 0, 4), (4, 1, 5)][..]),
        ("calib-c", &[0, 2, 4, 6][..], &[(0, 1, 2), (2, 2, 4), (4, 0, 6)][..]),
    ];

    let baseline = HdcOntologyDecodePolicy::conservative_default();
    let mut calibration_metrics = Vec::new();
    for (prefix, nodes, edges) in calibration_specs {
        let (graph, manifest) = make_graph(prefix, nodes, edges);
        let representation = codebook.encode_graph(&graph, &manifest)?;
        let metrics = codebook.measure_roundtrip(
            &graph,
            &representation,
            &manifest,
            &manifest,
            &expected_concepts(&graph),
            &expected_relations(edges),
            baseline,
        )?;
        calibration_metrics.push(metrics);
    }

    let calibration = HdcOntologyEmpiricalCalibration::from_clean_metrics(
        baseline,
        &calibration_metrics,
        0.10,
    )?;
    let calibrated_policy = calibration.policy();

    let evaluation_specs = [
        ("eval-a", &[1, 3, 5, 7][..], &[(1, 2, 5), (5, 1, 7), (7, 0, 3)][..]),
        (
            "eval-b",
            &[0, 2, 3, 6, 7][..],
            &[(0, 0, 3), (3, 2, 6), (6, 1, 7), (7, 2, 2)][..],
        ),
    ];

    let calibration_edges = calibration_specs
        .iter()
        .flat_map(|(_, _, edges)| edges.iter().map(|(s, r, t)| (s, r, t)))
        .map(|(source, relation, target)| {
            (
                format!("concept:{source:02}"),
                format!("relation:{relation}"),
                format!("concept:{target:02}"),
            )
        })
        .collect::<BTreeSet<_>>();

    let mut evaluation_metrics = Vec::new();
    for (prefix, nodes, edges) in evaluation_specs {
        let (graph, manifest) = make_graph(prefix, nodes, edges);
        let stable_edges = edges
            .iter()
            .map(|(source, relation, target)| {
                (
                    format!("concept:{source:02}"),
                    format!("relation:{relation}"),
                    format!("concept:{target:02}"),
                )
            })
            .collect::<BTreeSet<_>>();
        if !stable_edges.is_disjoint(&calibration_edges) {
            return Err(format!("evaluation case {prefix} overlaps calibration edge triples"));
        }

        let representation = codebook.encode_graph(&graph, &manifest)?;
        let metrics = codebook.measure_roundtrip(
            &graph,
            &representation,
            &manifest,
            &manifest,
            &expected_concepts(&graph),
            &expected_relations(edges),
            calibrated_policy,
        )?;
        evaluation_metrics.push(metrics);
    }

    let mut null_abstentions = 0_u32;
    const NULL_SAMPLES: u32 = 4096;
    let null_samples = NULL_SAMPLES;
    for index in 0..null_samples {
        let (graph, manifest) = make_graph("null", &[0, 1, 2], &[(0, 0, 1), (1, 1, 2)]);
        let clean = codebook.encode_graph(&graph, &manifest)?;

        // Draw directly from the binary transport space. This probes the
        // decoder boundary without spending the null sweep on an unrelated
        // continuous->binary quantization path.
        let node_binary = BinaryHV::random(0xDADA_0000 + index as u64);
        let edge_binary = BinaryHV::random(0xEDED_0000 + index as u64);

        let null_representation = symthaea_communication::hdc_ontology::HdcOntologyRepresentation {
            node_frame: HdcBinaryFrame::from_binary(&node_binary),
            edge_frame: HdcBinaryFrame::from_binary(&edge_binary),
            // Quantization telemetry is not consulted by the decoder; retain
            // the validated clean telemetry solely to keep the representation
            // schema complete for this N0 null-frame experiment.
            ..clean
        };

        match codebook.decode_graph_with_policy(
            &null_representation,
            &manifest,
            &manifest,
            calibrated_policy,
        ) {
            Ok(decoded) => {
                return Err(format!(
                    "random null sample {index} was accepted as identities: {:?}",
                    decoded.concept_ids_by_node
                ));
            }
            Err(_) => null_abstentions += 1,
        }
    }

    let min_clean_score = evaluation_metrics
        .iter()
        .flat_map(|metrics| [metrics.node_min_selected_score, metrics.edge_min_selected_score])
        .fold(f64::INFINITY, f64::min);
    let min_clean_margin = evaluation_metrics
        .iter()
        .flat_map(|metrics| [metrics.node_selection_margin, metrics.edge_selection_margin])
        .fold(f64::INFINITY, f64::min);

    if evaluation_metrics.iter().any(|metrics| {
        !metrics.structural_equivalence
            || !metrics.concept_identity_exact
            || !metrics.relation_identity_exact
            || metrics.node_precision != 1.0
            || metrics.node_recall != 1.0
            || metrics.edge_precision != 1.0
            || metrics.edge_recall != 1.0
            || metrics.node_selection_margin < calibrated_policy.min_margin
            || metrics.edge_selection_margin < calibrated_policy.min_margin
    }) {
        return Err("calibrated evaluation split failed exact retrieval gates".into());
    }

    if null_abstentions != null_samples {
        return Err("empirical calibration failed the null abstention gate".into());
    }

    let output = json!({
        "benchmark": "neurosemantic-hdc-grounded-identity-calibration-n0",
        "benchmark_schema_version": HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
        "execution_revision": execution_revision(),
        "adapter_id": HDC_ONTOLOGY_ADAPTER_ID,
        "claim_boundary": "empirical_clean_calibration_disjoint_evaluation_and_null_abstention_only",
        "calibration": {
            "cases": calibration_metrics.len(),
            "codebook_hash": calibration.codebook_hash,

            "lower_quantile": calibration.quantile,
            "baseline_min_score": baseline.min_score,
            "baseline_min_margin": baseline.min_margin,
            "calibrated_min_score": calibrated_policy.min_score,
            "calibrated_min_margin": calibrated_policy.min_margin,
        },
        "evaluation": {
            "cases": evaluation_metrics.len(),
            "minimum_clean_selected_score": min_clean_score,
            "minimum_clean_selection_margin": min_clean_margin,
            "all_exact": true,
        },
        "null_test": {
            "samples": null_samples,
            "generator": "BinaryHV::random",
            "node_seed_base": 0xDADA_0000_u64,
            "edge_seed_base": 0xEDED_0000_u64,
            "abstentions": null_abstentions,
            "accepted": 0,
        },
        "guarantee_boundary": "empirical_only; no distribution_free_or_conformal_coverage_claim",
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&output).map_err(|error| error.to_string())?
    );
    Ok(())
}
