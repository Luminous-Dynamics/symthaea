// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public-interface tests for the report-only architecture Pareto corpus.
//! Fixture values are synthetic examples, not exosuit measurements.

use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use symthaea_core::architecture_pareto::{
    evaluate_architecture_pareto, ArchitectureComparisonRequestV1,
    ArchitectureEvidenceLevelV1, ArchitectureMetricV1, ArchitectureValidationFindingV1,
    ARCHITECTURE_PARETO_SCHEMA_VERSION,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema_version: u16,
    benchmark_id: String,
    evidence_level: ArchitectureEvidenceLevelV1,
    claim_ceiling: String,
    comparisons: Vec<Comparison>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Comparison {
    id: String,
    request: ArchitectureComparisonRequestV1,
    expected_pareto_candidates: Vec<String>,
    expected_dominated_by: BTreeMap<String, Vec<String>>,
}

fn corpus() -> Corpus {
    let corpus: Corpus = serde_json::from_str(include_str!(
        "../../../../docs/research/open-mobility/fixtures/architecture-pareto-scenarios-v1.json"
    ))
    .expect("versioned architecture Pareto corpus parses");
    assert_eq!(corpus.schema_version, ARCHITECTURE_PARETO_SCHEMA_VERSION);
    assert_eq!(corpus.benchmark_id, "OPEN-MOBILITY-ARCHITECTURE-PARETO-001");
    assert_eq!(corpus.evidence_level, ArchitectureEvidenceLevelV1::SyntheticReplay);
    assert!(corpus.claim_ceiling.contains("Synthetic comparison only"));
    assert!(!corpus.comparisons.is_empty());

    let mut comparison_ids = HashSet::new();
    for comparison in &corpus.comparisons {
        assert!(!comparison.id.trim().is_empty());
        assert!(comparison_ids.insert(comparison.id.as_str()), "duplicate comparison ID");
        assert_eq!(comparison.id, comparison.request.benchmark_case_id);
        let mut candidate_ids = HashSet::new();
        for candidate in &comparison.request.candidates {
            assert!(candidate_ids.insert(candidate.candidate_id.as_str()), "duplicate candidate ID");
        }
    }
    corpus
}

#[test]
fn architecture_pareto_fixture_matches_expected_frontier_and_dominators() {
    let cases = corpus();
    for scenario in &cases.comparisons {
        let report = evaluate_architecture_pareto(&scenario.request)
            .unwrap_or_else(|error| panic!("scenario {} invalid: {error:?}", scenario.id));
        assert_eq!(report.benchmark_case_id, scenario.id);
        assert_eq!(report.evidence_level, ArchitectureEvidenceLevelV1::SyntheticReplay);

        let actual_frontier: Vec<String> = report.assessments.iter()
            .filter(|assessment| assessment.is_pareto_optimal)
            .map(|assessment| assessment.candidate_id.clone())
            .collect();
        assert_eq!(
            actual_frontier, scenario.expected_pareto_candidates,
            "unexpected frontier for {}",
            scenario.id
        );
        let actual_dominated_ids: HashSet<&str> = report.assessments.iter()
            .filter(|assessment| !assessment.is_pareto_optimal)
            .map(|assessment| assessment.candidate_id.as_str())
            .collect();
        let expected_dominated_ids: HashSet<&str> = scenario.expected_dominated_by.keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            actual_dominated_ids, expected_dominated_ids,
            "scenario {} must enumerate every and only dominated candidate",
            scenario.id
        );

        for (candidate_id, expected_dominators) in &scenario.expected_dominated_by {
            let assessment = report.assessments.iter()
                .find(|assessment| assessment.candidate_id == *candidate_id)
                .expect("expected dominated candidate is present");
            let actual_dominators: Vec<String> = assessment.dominated_by.iter()
                .map(|domination| domination.dominating_candidate_id.clone())
                .collect();
            assert_eq!(
                &actual_dominators, expected_dominators,
                "unexpected dominators for {candidate_id}"
            );
            assert!(
                assessment.dominated_by.iter().all(|domination| !domination.improved_dimensions.is_empty()),
                "dominance must state at least one improved dimension"
            );
        }
    }
}

#[test]
fn architecture_pareto_receipts_are_deterministic() {
    let cases = corpus();
    for scenario in &cases.comparisons {
        let expected = evaluate_architecture_pareto(&scenario.request)
            .expect("fixture comparison is valid");
        let expected_bytes = serde_json::to_vec(&expected).expect("report serializes");
        for replay in 0..100 {
            let actual = evaluate_architecture_pareto(&scenario.request)
                .expect("fixture comparison is valid");
            assert_eq!(actual, expected, "scenario {} replay {}", scenario.id, replay);
            assert_eq!(
                serde_json::to_vec(&actual).expect("report serializes"),
                expected_bytes,
                "serialized receipt changed for {}",
                scenario.id
            );
        }
    }
}

#[test]
fn invalid_mass_and_missing_values_are_not_silently_ranked() {
    let cases = corpus();
    let mut request = cases.comparisons[0].request.clone();
    request.candidates[0].metrics.mass_distribution.distal_limb_kg += 0.4;
    let result = evaluate_architecture_pareto(&request);
    assert!(matches!(
        result,
        Err(ref findings) if findings.iter().any(|finding| matches!(
            finding,
            ArchitectureValidationFindingV1::MassDistributionMismatch { .. }
        ))
    ));

    let serialized = serde_json::to_value(&cases.comparisons[0].request.candidates[0])
        .expect("candidate serializes");
    let mut missing_metric = serialized;
    missing_metric["metrics"].as_object_mut().expect("metrics object").remove("energy_wh_per_task");
    assert!(
        serde_json::from_value::<symthaea_core::architecture_pareto::ArchitectureCandidateV1>(missing_metric).is_err(),
        "missing required metric must not default to zero"
    );

    // The objective schema includes the full wearer/system denominator, not a single
    // summary score that would let remote mass or distal mass disappear.
    let _required_dimensions = [
        ArchitectureMetricV1::WearerBorneMassKg,
        ArchitectureMetricV1::TotalSystemMassKg,
        ArchitectureMetricV1::OffboardMassKg,
        ArchitectureMetricV1::DistalLimbMassKg,
        ArchitectureMetricV1::EnergyWhPerTask,
        ArchitectureMetricV1::ComfortScore,
    ];
}
