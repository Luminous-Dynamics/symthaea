use symthaea_futures_core::{
    AbstentionReason, ForecastOutput, Horizon, TrajectoryGenerator,
};
use symthaea_living_watershed_witness::{
    assess_replication, frozen_witness_protocol, run_witness, ClimatologyForecaster,
    PersistenceForecaster, SealedWatershedFixture, SyntheticWatershedSpec, WitnessRunLineage,
};
use symthaea_research_replication::{LineageRelation, ReplicationDesign, ReplicationOutcome};
use symthaea_research_result::MetricOutcome;

fn lineage(run: &str, manifest: &str, registered: i64, completed: i64) -> WitnessRunLineage {
    WitnessRunLineage::new(
        run,
        manifest,
        registered,
        completed,
        "source:test",
        "repro:test",
        "seeds:no-rng-v0",
    )
    .unwrap()
}

#[test]
fn fixture_composes_conserved_hydrology_and_bounded_ecology() {
    let fixture = SealedWatershedFixture::generate(
        SyntheticWatershedSpec::drydown("watershed-a", 6).unwrap(),
    )
    .unwrap();
    assert_eq!(fixture.forecast_history().len(), 6);
    assert!(fixture.forecast_history().observations().iter().all(|observation| {
        (0.0..=1.0).contains(&observation.soil_moisture_fraction)
            && (0.0..=1.0).contains(&observation.ecological_moisture_multiplier)
    }));
    assert!(!fixture.dataset_manifest_digest().is_empty());
    assert!(!fixture.verification_digest().unwrap().is_empty());
}

#[test]
fn sufficiently_long_history_scores_both_neutral_baselines() {
    let frozen = frozen_witness_protocol(1).unwrap();
    let fixture = SealedWatershedFixture::generate(
        SyntheticWatershedSpec::drydown("watershed-a", 6).unwrap(),
    )
    .unwrap();
    let execution = run_witness(&frozen, &fixture, lineage("run-a", "manifest-a", 2, 3)).unwrap();
    assert!(execution.persistence.brier_score.unwrap().is_finite());
    assert!(execution.climatology.brier_score.unwrap().is_finite());
    execution.result_manifest.verify_digest().unwrap();
}

#[test]
fn short_history_keeps_typed_abstention_and_primary_not_computed() {
    let frozen = frozen_witness_protocol(1).unwrap();
    let fixture = SealedWatershedFixture::generate(
        SyntheticWatershedSpec::drydown("watershed-short", 1).unwrap(),
    )
    .unwrap();
    let execution =
        run_witness(&frozen, &fixture, lineage("run-short", "manifest-short", 2, 3)).unwrap();
    assert_eq!(
        execution.climatology.abstention_reason(),
        Some(AbstentionReason::InsufficientObservationHistory)
    );
    let metric = execution
        .result_manifest
        .metrics
        .iter()
        .find(|metric| metric.metric_id == "climatology-brier")
        .unwrap();
    assert!(matches!(
        &metric.outcome,
        MetricOutcome::NotComputed { .. }
    ));
    execution.result_manifest.verify_digest().unwrap();
}

#[test]
fn unsupported_horizon_abstains_instead_of_guessing() {
    let fixture = SealedWatershedFixture::generate(
        SyntheticWatershedSpec::drydown("watershed-a", 6).unwrap(),
    )
    .unwrap();
    let persistence =
        PersistenceForecaster::default().generate(fixture.forecast_history(), Horizon(2));
    assert!(matches!(
        persistence,
        ForecastOutput::Abstain(AbstentionReason::HorizonBeyondValidatedRange)
    ));
    let climatology =
        ClimatologyForecaster::default().generate(fixture.forecast_history(), Horizon(2));
    assert!(matches!(
        climatology,
        ForecastOutput::Abstain(AbstentionReason::HorizonBeyondValidatedRange)
    ));
}

#[test]
fn exact_reproduction_is_distinct_from_direct_replication() {
    let frozen = frozen_witness_protocol(1).unwrap();
    let fixture = SealedWatershedFixture::generate(
        SyntheticWatershedSpec::drydown("watershed-a", 6).unwrap(),
    )
    .unwrap();
    let first = run_witness(&frozen, &fixture, lineage("run-a", "manifest-a", 2, 3)).unwrap();
    let second = run_witness(&frozen, &fixture, lineage("run-b", "manifest-b", 4, 5)).unwrap();
    let assessment = assess_replication(
        "exact-a",
        ReplicationDesign::ExactReproduction,
        &first,
        &second,
        ReplicationOutcome::Concordant,
    )
    .unwrap();
    assert_eq!(
        assessment.factual_lineage.dataset_manifest,
        LineageRelation::Same
    );
    assessment.verify_digest().unwrap();
    assert!(
        assess_replication(
            "not-direct",
            ReplicationDesign::DirectReplication,
            &first,
            &second,
            ReplicationOutcome::Concordant,
        )
        .is_err()
    );
}

#[test]
fn new_fixture_can_form_direct_replication_lineage_without_auto_concordance() {
    let frozen = frozen_witness_protocol(1).unwrap();
    let original_fixture = SealedWatershedFixture::generate(
        SyntheticWatershedSpec::drydown("watershed-a", 6).unwrap(),
    )
    .unwrap();
    let replication_spec = SyntheticWatershedSpec::new(
        "watershed-b",
        100.0,
        5.0,
        70.0,
        1.0,
        6,
        0.20,
        0.70,
        0.10,
        0.55,
    )
    .unwrap();
    let replication_fixture = SealedWatershedFixture::generate(replication_spec).unwrap();
    let original = run_witness(
        &frozen,
        &original_fixture,
        lineage("run-a", "manifest-a", 2, 3),
    )
    .unwrap();
    let followup = run_witness(
        &frozen,
        &replication_fixture,
        lineage("run-b", "manifest-b", 4, 5),
    )
    .unwrap();
    let assessment = assess_replication(
        "direct-a",
        ReplicationDesign::DirectReplication,
        &original,
        &followup,
        ReplicationOutcome::Inconclusive,
    )
    .unwrap();
    assert_eq!(assessment.factual_lineage.protocol, LineageRelation::Same);
    assert_eq!(
        assessment.factual_lineage.dataset_manifest,
        LineageRelation::Different
    );
    assert_eq!(assessment.outcome, ReplicationOutcome::Inconclusive);
}
