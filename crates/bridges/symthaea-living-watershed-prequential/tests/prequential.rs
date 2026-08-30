use std::cell::RefCell;

use symthaea_futures_core::{
    AssumptionId, ForecastDistribution, ForecastOutput, Horizon, OutcomeRegion, OutcomeSpaceId,
    TrajectoryGenerator,
};
use symthaea_living_watershed_prequential::{
    BindingViolation, Candidate, PrequentialEpisodeSpec, PrequentialError, WETLAND_STRESS_OUTCOME_SPACE,
    evaluate_candidates, frozen_prequential_protocol, prepare_episode, run_prequential_baselines,
};
use symthaea_living_watershed_witness::{
    ClimatologyForecaster, PersistenceForecaster, SyntheticWatershedSpec, WatershedHistory,
    WitnessRunLineage,
};
use symthaea_research_replication::{
    ReplicationAssessment, ReplicationComparisonEvidence, ReplicationDesign, ReplicationOutcome,
};
use symthaea_research_result::MetricOutcome;

fn lineage(run: &str, manifest: &str, registered: i64, completed: i64) -> WitnessRunLineage {
    WitnessRunLineage::new(
        run,
        manifest,
        registered,
        completed,
        "source:prequential-test",
        "repro:prequential-test",
        "seeds:no-rng-v1",
    )
    .unwrap()
}

fn plan(first_origin: usize, evaluation_steps: usize) -> symthaea_living_watershed_prequential::PrequentialEpisodePlan {
    prepare_episode(
        PrequentialEpisodeSpec::new(
            SyntheticWatershedSpec::drydown("episode-a", first_origin).unwrap(),
            evaluation_steps,
        )
        .unwrap(),
    )
    .unwrap()
}

fn binary_distribution(
    issued_at_tick: u64,
    horizon: Horizon,
    outcome_space: &str,
    p_true: f64,
    unsupported_mass: f64,
) -> ForecastOutput {
    ForecastOutput::Distribution(
        ForecastDistribution::try_from_raw(
            issued_at_tick,
            horizon,
            OutcomeSpaceId(outcome_space.into()),
            vec![
                (
                    p_true,
                    OutcomeRegion::Boolean(true),
                    vec![AssumptionId("test".into())],
                ),
                (
                    1.0 - p_true - unsupported_mass,
                    OutcomeRegion::Boolean(false),
                    vec![AssumptionId("test".into())],
                ),
            ],
            unsupported_mass,
        )
        .unwrap(),
    )
}

struct TraceForecaster {
    calls: RefCell<Vec<(usize, u64)>>,
}

impl TraceForecaster {
    fn new() -> Self {
        Self {
            calls: RefCell::new(Vec::new()),
        }
    }
}

impl TrajectoryGenerator for TraceForecaster {
    type Observation = WatershedHistory;

    fn generate(&self, history: &Self::Observation, horizon: Horizon) -> ForecastOutput {
        let last = history.last().unwrap();
        self.calls.borrow_mut().push((history.len(), last.tick));
        binary_distribution(
            last.tick,
            horizon,
            WETLAND_STRESS_OUTCOME_SPACE,
            0.5,
            0.0,
        )
    }
}

#[derive(Clone, Copy)]
enum Malformation {
    IssueTick,
    Horizon,
    OutcomeSpace,
    NonBinary,
    UnsupportedMass,
}

struct MalformedForecaster(Malformation);

impl TrajectoryGenerator for MalformedForecaster {
    type Observation = WatershedHistory;

    fn generate(&self, history: &Self::Observation, _horizon: Horizon) -> ForecastOutput {
        let tick = history.last().unwrap().tick;
        match self.0 {
            Malformation::IssueTick => binary_distribution(
                tick + 1,
                Horizon(1),
                WETLAND_STRESS_OUTCOME_SPACE,
                0.5,
                0.0,
            ),
            Malformation::Horizon => binary_distribution(
                tick,
                Horizon(2),
                WETLAND_STRESS_OUTCOME_SPACE,
                0.5,
                0.0,
            ),
            Malformation::OutcomeSpace => {
                binary_distribution(tick, Horizon(1), "wrong-target", 0.5, 0.0)
            }
            Malformation::NonBinary => ForecastOutput::Distribution(
                ForecastDistribution::try_from_raw(
                    tick,
                    Horizon(1),
                    OutcomeSpaceId(WETLAND_STRESS_OUTCOME_SPACE.into()),
                    vec![(
                        1.0,
                        OutcomeRegion::Boolean(true),
                        vec![AssumptionId("test".into())],
                    )],
                    0.0,
                )
                .unwrap(),
            ),
            Malformation::UnsupportedMass => binary_distribution(
                tick,
                Horizon(1),
                WETLAND_STRESS_OUTCOME_SPACE,
                0.4,
                0.2,
            ),
        }
    }
}

#[test]
fn plan_is_content_addressed_and_structurally_fixed() {
    let mut prepared = plan(1, 4);
    prepared.verify_digest().unwrap();
    assert_eq!(
        prepared
            .commitments
            .iter()
            .map(|commitment| commitment.origin)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );

    prepared.commitments[0].dataset_manifest_digest.push('x');
    assert!(matches!(
        prepared.verify_digest(),
        Err(PrequentialError::PlanDigestMismatch)
    ));
}

#[test]
fn recomputed_plan_digest_does_not_hide_fixture_commitment_tampering() {
    let mut prepared = plan(2, 2);
    prepared.commitments[0].verification_commitment_digest = "attacker-selected".into();
    prepared.plan_digest = prepared.compute_digest().unwrap();
    prepared.verify_digest().unwrap();

    let persistence = PersistenceForecaster::default();
    let candidates = [Candidate::new("persistence", &persistence).unwrap()];
    assert!(matches!(
        evaluate_candidates(&prepared, &candidates),
        Err(PrequentialError::FixtureCommitmentMismatch { origin: 2 })
    ));
}

#[test]
fn rolling_origins_expose_only_the_available_prefix() {
    let prepared = plan(1, 4);
    let first = TraceForecaster::new();
    let second = TraceForecaster::new();
    let candidates = [
        Candidate::new("trace-a", &first).unwrap(),
        Candidate::new("trace-b", &second).unwrap(),
    ];
    let evaluation = evaluate_candidates(&prepared, &candidates).unwrap();

    let expected = vec![(1, 0), (2, 1), (3, 2), (4, 3)];
    assert_eq!(*first.calls.borrow(), expected);
    assert_eq!(*second.calls.borrow(), expected);
    assert_eq!(
        evaluation
            .steps
            .iter()
            .map(|step| step.actual.tick)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    assert!(evaluation.steps.iter().all(|step| {
        step.forecasts
            .iter()
            .all(|forecast| !forecast.issued.output_digest.is_empty())
    }));
}

#[test]
fn stale_issue_tick_fails_closed() {
    let prepared = plan(3, 1);
    let bad = MalformedForecaster(Malformation::IssueTick);
    let candidates = [Candidate::new("bad", &bad).unwrap()];
    let error = evaluate_candidates(&prepared, &candidates).unwrap_err();
    assert!(matches!(
        error,
        PrequentialError::Binding(BindingViolation::WrongIssueTick { .. })
    ));
}

#[test]
fn all_other_semantic_binding_mismatches_fail_closed() {
    for malformation in [
        Malformation::Horizon,
        Malformation::OutcomeSpace,
        Malformation::NonBinary,
        Malformation::UnsupportedMass,
    ] {
        let prepared = plan(3, 1);
        let bad = MalformedForecaster(malformation);
        let candidates = [Candidate::new("bad", &bad).unwrap()];
        assert!(matches!(
            evaluate_candidates(&prepared, &candidates),
            Err(PrequentialError::Binding(_))
        ));
    }
}

#[test]
fn coverage_is_retained_beside_selective_mean_score() {
    let prepared = plan(1, 4);
    let persistence = PersistenceForecaster::default();
    let climatology = ClimatologyForecaster::default();
    let candidates = [
        Candidate::new("persistence-v0", &persistence).unwrap(),
        Candidate::new("empirical-climatology-v0", &climatology).unwrap(),
    ];
    let evaluation = evaluate_candidates(&prepared, &candidates).unwrap();
    let persistence = &evaluation.aggregates[0];
    let climatology = &evaluation.aggregates[1];

    assert_eq!(persistence.coverage, 1.0);
    assert_eq!(persistence.scored_steps, 4);
    assert_eq!(climatology.coverage, 0.5);
    assert_eq!(climatology.scored_steps, 2);
    assert_eq!(climatology.abstained_steps, 2);
    assert!(climatology.mean_brier_scored_cases.unwrap().is_finite());
}

#[test]
fn mutated_public_v0_template_is_revalidated_at_v1_boundary() {
    let mut template = SyntheticWatershedSpec::drydown("episode-a", 2).unwrap();
    template.stress_multiplier_threshold = f64::NAN;
    assert!(PrequentialEpisodeSpec::new(template, 2).is_err());
}

#[test]
fn frozen_protocol_must_match_exact_episode_design() {
    let prepared = plan(1, 4);
    let wrong = frozen_prequential_protocol(1, 1, 3).unwrap();
    assert!(matches!(
        run_prequential_baselines(&wrong, &prepared, lineage("run-a", "manifest-a", 2, 3)),
        Err(PrequentialError::ProtocolDesignMismatch)
    ));
}

#[test]
fn result_manifest_retains_mean_scores_coverage_and_registered_plan() {
    let prepared = plan(1, 4);
    let frozen = frozen_prequential_protocol(1, 1, 4).unwrap();
    let execution = run_prequential_baselines(
        &frozen,
        &prepared,
        lineage("run-a", "manifest-a", 2, 3),
    )
    .unwrap();

    execution.result_manifest.verify_digest().unwrap();
    assert_eq!(
        execution.result_manifest.run.dataset_manifest_digest,
        prepared.plan_digest
    );
    let coverage = execution
        .result_manifest
        .metrics
        .iter()
        .find(|metric| metric.metric_id == "climatology-coverage")
        .unwrap();
    assert!(matches!(
        coverage.outcome,
        MetricOutcome::Numeric { value, ref unit }
            if value == 0.5 && unit == "fraction"
    ));
}

#[test]
fn distinct_episode_plan_can_be_direct_replication_under_same_protocol() {
    let original_plan = plan(1, 4);
    let mut followup_template = SyntheticWatershedSpec::drydown("episode-b", 1).unwrap();
    followup_template.initial_storage_mm = 70.0;
    let followup_plan = prepare_episode(
        PrequentialEpisodeSpec::new(followup_template, 4).unwrap(),
    )
    .unwrap();
    assert_ne!(original_plan.plan_digest, followup_plan.plan_digest);

    let frozen = frozen_prequential_protocol(1, 1, 4).unwrap();
    let original = run_prequential_baselines(
        &frozen,
        &original_plan,
        lineage("run-a", "manifest-a", 2, 3),
    )
    .unwrap();
    let followup = run_prequential_baselines(
        &frozen,
        &followup_plan,
        lineage("run-b", "manifest-b", 4, 5),
    )
    .unwrap();

    let comparison = ReplicationComparisonEvidence::new(
        "prequential-v1-frozen-comparison",
        "digest:comparison-plan",
        "Mechanism replication; outcome remains explicit rather than inferred from score similarity.",
    )
    .unwrap();
    let assessment = ReplicationAssessment::new(
        "direct-preq",
        ReplicationDesign::DirectReplication,
        &original.result_manifest,
        &followup.result_manifest,
        vec![],
        ReplicationOutcome::Inconclusive,
        comparison,
    )
    .unwrap();
    assessment.verify_digest().unwrap();
}
