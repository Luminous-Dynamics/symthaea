use symthaea_research_protocol::{
    AnalysisPlanRef, BaselineSpec, FrozenProtocol, HypothesisDirection, HypothesisRole,
    HypothesisSpec, MetricRole, MetricSpec, MultiplicityPolicy, ResearchProtocol,
    ResearchRunRegistration, StoppingRule,
};
use symthaea_research_replication::{
    IndependenceDimension, IndependenceEvidence, LineageRelation, ReplicationAssessment,
    ReplicationComparisonEvidence, ReplicationDesign, ReplicationError, ReplicationOutcome,
};
use symthaea_research_result::{
    MetricOutcome, MetricResult, ResearchResultManifest, ResultArtifactKind, ResultArtifactRef,
};

fn frozen() -> FrozenProtocol {
    ResearchProtocol::new(
        "replication-fixture-v1",
        "1",
        "Does the preregistered effect reproduce on held-out evidence?",
        vec![HypothesisSpec::new(
            "h-primary",
            "the measured effect differs from the baseline",
            HypothesisRole::Primary,
            HypothesisDirection::TwoSided,
        )
        .unwrap()],
        vec![MetricSpec::new(
            "primary-effect",
            "primary effect",
            "unit",
            MetricRole::Primary,
            "held-out mean",
        )
        .unwrap()],
        vec![BaselineSpec::new(
            "baseline",
            "frozen baseline",
            "fixture/baseline-v1",
        )
        .unwrap()],
        vec![],
        StoppingRule::FixedSampleCount(10),
        MultiplicityPolicy::NotApplicable,
        AnalysisPlanRef::new("analysis", "1", "sha256:analysis-plan").unwrap(),
        "frozen dataset lineage",
        "frozen seed lineage",
    )
    .unwrap()
    .freeze(1_000)
    .unwrap()
}

fn result(
    frozen: &FrozenProtocol,
    id: &str,
    source: &str,
    dataset: &str,
    environment: &str,
    seeds: &str,
) -> ResearchResultManifest {
    let run = ResearchRunRegistration::new(
        frozen,
        format!("run-{id}"),
        1_100,
        source,
        dataset,
        environment,
        seeds,
    )
    .unwrap();

    ResearchResultManifest::new(
        frozen,
        run,
        format!("result-{id}"),
        2_000,
        vec![],
        vec![],
        false,
        vec![ResultArtifactRef::new(
            "analysis",
            ResultArtifactKind::Analysis,
            format!("sha256:analysis-{id}"),
            "preregistered analysis output",
        )
        .unwrap()],
        vec![MetricResult::new(
            "primary-effect",
            MetricOutcome::Numeric {
                value: 1.0,
                unit: "unit".into(),
            },
        )
        .unwrap()],
        vec![],
    )
    .unwrap()
}

fn comparison() -> ReplicationComparisonEvidence {
    ReplicationComparisonEvidence::new(
        "frozen replication-comparison plan v1",
        "sha256:replication-comparison",
        "compare the preregistered primary metric without post-hoc redefinition",
    )
    .unwrap()
}

#[test]
fn direct_replication_requires_new_data() {
    let frozen = frozen();
    let original = result(&frozen, "a", "source", "dataset", "env", "seeds-a");
    let followup = result(&frozen, "b", "source", "dataset", "env", "seeds-b");

    let error = ReplicationAssessment::new(
        "replication-1",
        ReplicationDesign::DirectReplication,
        &original,
        &followup,
        vec![],
        ReplicationOutcome::Concordant,
        comparison(),
    )
    .unwrap_err();

    assert_eq!(error, ReplicationError::DirectReplicationReusedDataset);
}

#[test]
fn direct_replication_can_reuse_implementation_but_not_data() {
    let frozen = frozen();
    let original = result(
        &frozen,
        "a",
        "same-source",
        "dataset-a",
        "same-env",
        "seeds-a",
    );
    let followup = result(
        &frozen,
        "b",
        "same-source",
        "dataset-b",
        "same-env",
        "seeds-b",
    );

    let assessment = ReplicationAssessment::new(
        "replication-2",
        ReplicationDesign::DirectReplication,
        &original,
        &followup,
        vec![],
        ReplicationOutcome::Concordant,
        comparison(),
    )
    .unwrap();

    assert_eq!(
        assessment.factual_lineage.source_commit,
        LineageRelation::Same
    );
    assert_eq!(
        assessment.factual_lineage.dataset_manifest,
        LineageRelation::Different
    );
    assessment.verify_digest().unwrap();
}

#[test]
fn exact_reproduction_requires_exact_environment_lineage() {
    let frozen = frozen();
    let original = result(&frozen, "a", "source", "dataset", "env-a", "seeds");
    let followup = result(&frozen, "b", "source", "dataset", "env-b", "seeds");

    let error = ReplicationAssessment::new(
        "reproduction-1",
        ReplicationDesign::ExactReproduction,
        &original,
        &followup,
        vec![],
        ReplicationOutcome::Concordant,
        comparison(),
    )
    .unwrap_err();

    assert_eq!(error, ReplicationError::ExactReproductionEnvironmentChanged);
}

#[test]
fn exact_reproduction_accepts_same_lineage_with_distinct_result_records() {
    let frozen = frozen();
    let original = result(&frozen, "a", "source", "dataset", "env", "seeds");
    let followup = result(&frozen, "b", "source", "dataset", "env", "seeds");

    let assessment = ReplicationAssessment::new(
        "reproduction-2",
        ReplicationDesign::ExactReproduction,
        &original,
        &followup,
        vec![],
        ReplicationOutcome::Concordant,
        comparison(),
    )
    .unwrap();

    assert_eq!(assessment.factual_lineage.protocol, LineageRelation::Same);
    assert_eq!(
        assessment.factual_lineage.reproducibility_capsule,
        LineageRelation::Same
    );
}

#[test]
fn reanalysis_requires_same_dataset_but_may_change_implementation() {
    let frozen = frozen();
    let original = result(&frozen, "a", "source-a", "dataset", "env", "seeds-a");
    let followup = result(&frozen, "b", "source-b", "dataset", "env", "seeds-b");

    let assessment = ReplicationAssessment::new(
        "reanalysis-1",
        ReplicationDesign::Reanalysis,
        &original,
        &followup,
        vec![],
        ReplicationOutcome::Mixed,
        comparison(),
    )
    .unwrap();

    assert_eq!(
        assessment.factual_lineage.dataset_manifest,
        LineageRelation::Same
    );
    assert_eq!(
        assessment.factual_lineage.source_commit,
        LineageRelation::Different
    );
}

#[test]
fn reanalysis_rejects_a_new_dataset_lineage() {
    let frozen = frozen();
    let original = result(&frozen, "a", "source-a", "dataset-a", "env", "seeds");
    let followup = result(&frozen, "b", "source-b", "dataset-b", "env", "seeds");

    let error = ReplicationAssessment::new(
        "reanalysis-2",
        ReplicationDesign::Reanalysis,
        &original,
        &followup,
        vec![],
        ReplicationOutcome::Mixed,
        comparison(),
    )
    .unwrap_err();

    assert_eq!(error, ReplicationError::ReanalysisDatasetChanged);
}

#[test]
fn institutional_independence_is_evidence_backed_not_inferred() {
    let evidence = IndependenceEvidence::new(
        IndependenceDimension::Institution,
        "follow-up acquisition and analysis were performed by a separate laboratory",
        "sha256:independence-attestation",
    )
    .unwrap();

    assert_eq!(evidence.dimension, IndependenceDimension::Institution);
    assert_eq!(
        evidence.evidence_digest,
        "sha256:independence-attestation"
    );
}

#[test]
fn conceptual_replication_keeps_different_lineages_visible() {
    let frozen_a = frozen();
    let frozen_b = ResearchProtocol::new(
        "conceptual-fixture-v2",
        "2",
        "Does a related mechanism generalize under a changed protocol?",
        vec![HypothesisSpec::new(
            "h-related",
            "related mechanism persists",
            HypothesisRole::Primary,
            HypothesisDirection::TwoSided,
        )
        .unwrap()],
        vec![MetricSpec::new(
            "primary-effect",
            "primary effect",
            "unit",
            MetricRole::Primary,
            "held-out mean",
        )
        .unwrap()],
        vec![BaselineSpec::new(
            "baseline",
            "changed baseline",
            "fixture/baseline-v2",
        )
        .unwrap()],
        vec![],
        StoppingRule::FixedSampleCount(20),
        MultiplicityPolicy::NotApplicable,
        AnalysisPlanRef::new("analysis", "2", "sha256:analysis-plan-v2").unwrap(),
        "new population / dataset",
        "new seed lineage",
    )
    .unwrap()
    .freeze(1_000)
    .unwrap();

    let original = result(&frozen_a, "a", "source-a", "dataset-a", "env-a", "seeds-a");
    let followup = result(&frozen_b, "b", "source-b", "dataset-b", "env-b", "seeds-b");

    let assessment = ReplicationAssessment::new(
        "conceptual-1",
        ReplicationDesign::ConceptualReplication,
        &original,
        &followup,
        vec![],
        ReplicationOutcome::Inconclusive,
        comparison(),
    )
    .unwrap();

    assert_eq!(assessment.factual_lineage.protocol, LineageRelation::Different);
    assert_eq!(
        assessment.factual_lineage.dataset_manifest,
        LineageRelation::Different
    );
}
