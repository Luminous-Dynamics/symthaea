//! Rolling-origin commit-reveal validation for Living Watershed / Wetland Watch.
//!
//! v0 proves one held-out forecast can traverse the research stack. This crate strengthens that
//! mechanism into a prequential episode: every candidate sees only the history available at the
//! forecast origin, every candidate output is validated and content-digested before the next
//! state is revealed, then the already-issued outputs are scored against that revealed state.
//!
//! This remains a synthetic mechanism witness. A digest is an integrity commitment, not a secrecy
//! primitive; real blinded campaigns should separate fixture custody, model execution, and
//! verification, and should not expose small-state commitments to an untrusted model process.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use symthaea_futures_calibration::{BrierScore, ScoringRule};
use symthaea_futures_core::{
    ForecastDistribution, ForecastOutput, Horizon, OutcomeRegion, OutcomeSpaceId,
    TrajectoryGenerator,
};
use symthaea_living_watershed_witness::{
    ClimatologyForecaster, PersistenceForecaster, SealedWatershedFixture, SyntheticWatershedSpec,
    WatershedHistory, WetlandObservation, WitnessRunLineage,
};
use symthaea_research_protocol::{
    AnalysisPlanRef, BaselineSpec, FrozenProtocol, HypothesisDirection, HypothesisRole,
    HypothesisSpec, MetricRole, MetricSpec, MultiplicityPolicy, ResearchProtocol,
    ResearchRunRegistration, StoppingRule,
};
use symthaea_research_result::{
    ClaimDisposition, ClaimInterpretation, MetricOutcome, MetricResult, ResearchResultManifest,
    ResultArtifactKind, ResultArtifactRef, ResultClaim,
};
use thiserror::Error;

pub const WETLAND_STRESS_OUTCOME_SPACE: &str =
    "living-watershed/wetland-stress-next-step/v0";
pub const PREQUENTIAL_BRIER_UNIT: &str = "brier_multiclass";

const PLAN_SCHEMA: &str = "symthaea-living-watershed-prequential-plan/v1";
const OUTPUT_SCHEMA: &str = "symthaea-living-watershed-prequential-output/v1";
const LEDGER_SCHEMA: &str = "symthaea-living-watershed-prequential-ledger/v1";
const VERIFICATION_SCHEMA: &str = "symthaea-living-watershed-verification/v0";
const ANALYSIS_PLAN: &str = r#"Living Watershed prequential v1 analysis plan

Purpose: strengthen the synthetic mechanism witness from one held-out transition to a fixed
rolling-origin episode without making a real-world wetland skill claim.

For each preregistered origin:
1. regenerate the exact precommitted sealed v0 fixture for that origin;
2. expose only its predictor-visible WatershedHistory to every candidate;
3. obtain every candidate ForecastOutput;
4. fail closed if any distribution has a stale issue tick, wrong horizon, wrong outcome space,
   non-binary support, or non-zero unsupported mass;
5. content-digest every accepted output before verification;
6. reveal the next deterministic observation only after all candidate outputs are committed;
7. verify the reveal against the v0 verification commitment;
8. score already-issued distributions with the canonical Futures Laboratory multiclass Brier rule;
9. retain typed abstentions without numeric sentinels;
10. aggregate mean Brier over scored cases and report coverage separately.

Mean score is conditional on issuing a forecast and MUST NOT be interpreted without coverage.
This plan does not rank models with unequal coverage and does not test Sentinel accuracy, HDC
benefit, ecological validity, or intervention utility.
"#;

#[derive(Debug, Error)]
pub enum PrequentialError {
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("invalid prequential specification: {0}")]
    InvalidSpec(String),
    #[error("living-watershed fixture failed: {0}")]
    Fixture(String),
    #[error(transparent)]
    Binding(#[from] BindingViolation),
    #[error("episode plan digest mismatch")]
    PlanDigestMismatch,
    #[error("prepared fixture commitment mismatch at origin {origin}")]
    FixtureCommitmentMismatch { origin: usize },
    #[error("verification commitment mismatch at origin {origin}")]
    VerificationCommitmentMismatch { origin: usize },
    #[error("frozen protocol does not match the prepared prequential design")]
    ProtocolDesignMismatch,
    #[error("forecast scoring failed: {0}")]
    Scoring(String),
    #[error("serialization failed: {0}")]
    Serialization(String),
    #[error("research protocol failed: {0}")]
    Protocol(String),
    #[error("research result failed: {0}")]
    ResearchResult(String),
    #[error("candidate set must not be empty")]
    EmptyCandidateSet,
    #[error("duplicate candidate id: {0}")]
    DuplicateCandidateId(String),
}

pub type Result<T> = std::result::Result<T, PrequentialError>;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum BindingViolation {
    #[error("forecast issue tick mismatch: expected {expected}, got {got}")]
    WrongIssueTick { expected: u64, got: u64 },
    #[error("forecast horizon mismatch: expected {expected:?}, got {got:?}")]
    WrongHorizon { expected: Horizon, got: Horizon },
    #[error("forecast outcome space mismatch: expected {expected}, got {got}")]
    WrongOutcomeSpace { expected: String, got: String },
    #[error("wetland-stress forecast requires exactly one true and one false Boolean branch")]
    NonBinarySupport,
    #[error("wetland-stress forecast requires zero unsupported mass, got {value}")]
    UnsupportedMass { value: f64 },
}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        Err(PrequentialError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn digest_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| PrequentialError::Serialization(error.to_string()))?;
    Ok(digest_bytes(&bytes))
}

fn analysis_plan_digest() -> String {
    digest_bytes(ANALYSIS_PLAN.as_bytes())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrequentialEpisodeSpec {
    /// The v0 fixture template. `history_steps` is the first forecast origin.
    pub template: SyntheticWatershedSpec,
    /// Number of sequential one-step forecast/verification cases.
    pub evaluation_steps: usize,
}

impl PrequentialEpisodeSpec {
    pub fn new(template: SyntheticWatershedSpec, evaluation_steps: usize) -> Result<Self> {
        if evaluation_steps == 0 {
            return Err(PrequentialError::InvalidSpec(
                "evaluation_steps must be > 0".into(),
            ));
        }
        // Reconstruct through the v0 validated constructor because its fields are public and a
        // caller may have mutated a previously valid value.
        let template = SyntheticWatershedSpec::new(
            template.fixture_id,
            template.capacity_mm,
            template.potential_evapotranspiration_mm_per_day,
            template.initial_storage_mm,
            template.precipitation_mm_per_day,
            template.history_steps,
            template.wilting_fraction,
            template.optimum_fraction,
            template.minimum_moisture_multiplier,
            template.stress_multiplier_threshold,
        )
        .map_err(|error| PrequentialError::Fixture(error.to_string()))?;
        template
            .history_steps
            .checked_add(evaluation_steps)
            .ok_or_else(|| PrequentialError::InvalidSpec("origin range overflow".into()))?;
        Ok(Self {
            template,
            evaluation_steps,
        })
    }

    pub fn first_origin(&self) -> usize {
        self.template.history_steps
    }

    pub fn end_origin_exclusive(&self) -> usize {
        self.first_origin() + self.evaluation_steps
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepCommitment {
    pub origin: usize,
    pub dataset_manifest_digest: String,
    pub verification_commitment_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrequentialEpisodePlan {
    pub spec: PrequentialEpisodeSpec,
    pub commitments: Vec<StepCommitment>,
    pub plan_digest: String,
}

#[derive(Serialize)]
struct PlanDigestView<'a> {
    schema: &'static str,
    spec: &'a PrequentialEpisodeSpec,
    commitments: &'a [StepCommitment],
}

impl PrequentialEpisodePlan {
    fn digest_view(&self) -> PlanDigestView<'_> {
        PlanDigestView {
            schema: PLAN_SCHEMA,
            spec: &self.spec,
            commitments: &self.commitments,
        }
    }

    pub fn compute_digest(&self) -> Result<String> {
        digest_serializable(&self.digest_view())
    }

    fn verify_structure(&self) -> Result<()> {
        if self.commitments.len() != self.spec.evaluation_steps {
            return Err(PrequentialError::InvalidSpec(format!(
                "plan has {} commitments but evaluation_steps is {}",
                self.commitments.len(), self.spec.evaluation_steps
            )));
        }
        for (offset, commitment) in self.commitments.iter().enumerate() {
            let expected = self.spec.first_origin() + offset;
            if commitment.origin != expected {
                return Err(PrequentialError::InvalidSpec(format!(
                    "commitment origin mismatch at offset {offset}: expected {expected}, got {}",
                    commitment.origin
                )));
            }
            non_empty(
                &commitment.dataset_manifest_digest,
                "dataset manifest digest",
            )?;
            non_empty(
                &commitment.verification_commitment_digest,
                "verification commitment digest",
            )?;
        }
        Ok(())
    }

    pub fn verify_digest(&self) -> Result<()> {
        self.verify_structure()?;
        if self.compute_digest()? != self.plan_digest {
            return Err(PrequentialError::PlanDigestMismatch);
        }
        Ok(())
    }
}

/// Prepare all integrity commitments before a research run is registered.
///
/// The returned digests commit to each sealed fixture and held-out next state, but are not a
/// secrecy guarantee. They are never passed through the candidate `TrajectoryGenerator` input.
pub fn prepare_episode(spec: PrequentialEpisodeSpec) -> Result<PrequentialEpisodePlan> {
    let spec = PrequentialEpisodeSpec::new(spec.template, spec.evaluation_steps)?;
    let mut commitments = Vec::with_capacity(spec.evaluation_steps);
    for origin in spec.first_origin()..spec.end_origin_exclusive() {
        let fixture = fixture_at_origin(&spec.template, origin)?;
        commitments.push(StepCommitment {
            origin,
            dataset_manifest_digest: fixture.dataset_manifest_digest().to_string(),
            verification_commitment_digest: fixture
                .verification_digest()
                .map_err(|error| PrequentialError::Fixture(error.to_string()))?,
        });
    }
    let mut plan = PrequentialEpisodePlan {
        spec,
        commitments,
        plan_digest: String::new(),
    };
    plan.plan_digest = plan.compute_digest()?;
    plan.verify_digest()?;
    Ok(plan)
}

fn fixture_at_origin(
    template: &SyntheticWatershedSpec,
    origin: usize,
) -> Result<SealedWatershedFixture> {
    if origin == 0 {
        return Err(PrequentialError::InvalidSpec(
            "forecast origin must be > 0".into(),
        ));
    }
    let mut spec = template.clone();
    spec.history_steps = origin;
    SealedWatershedFixture::generate(spec)
        .map_err(|error| PrequentialError::Fixture(error.to_string()))
}

pub struct Candidate<'a> {
    pub id: &'a str,
    pub generator: &'a dyn TrajectoryGenerator<Observation = WatershedHistory>,
}

impl<'a> Candidate<'a> {
    pub fn new(
        id: &'a str,
        generator: &'a dyn TrajectoryGenerator<Observation = WatershedHistory>,
    ) -> Result<Self> {
        non_empty(id, "candidate id")?;
        Ok(Self { id, generator })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssuedForecast {
    pub forecaster_id: String,
    pub origin: usize,
    pub history_dataset_digest: String,
    pub output: ForecastOutput,
    /// Commitment produced before the verification outcome is revealed.
    pub output_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedForecast {
    pub issued: IssuedForecast,
    pub brier_score: Option<f64>,
}

impl ResolvedForecast {
    pub fn abstained(&self) -> bool {
        matches!(&self.issued.output, ForecastOutput::Abstain(_))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrequentialStepReport {
    /// The tick being forecast and then revealed.
    pub origin: usize,
    pub history_dataset_digest: String,
    pub verification_commitment_digest: String,
    pub actual: WetlandObservation,
    pub forecasts: Vec<ResolvedForecast>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecasterAggregate {
    pub forecaster_id: String,
    /// Arithmetic mean over scored cases only. Read with `coverage`.
    pub mean_brier_scored_cases: Option<f64>,
    pub scored_steps: usize,
    pub abstained_steps: usize,
    pub total_steps: usize,
    pub coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrequentialEvaluation {
    pub episode_plan_digest: String,
    pub steps: Vec<PrequentialStepReport>,
    pub aggregates: Vec<ForecasterAggregate>,
    pub forecast_ledger_digest: String,
}

fn validate_distribution_binding(
    distribution: &ForecastDistribution,
    history: &WatershedHistory,
    requested_horizon: Horizon,
) -> std::result::Result<(), BindingViolation> {
    let expected_tick = history.last().map_or(0, |observation| observation.tick);
    if distribution.issued_at_tick() != expected_tick {
        return Err(BindingViolation::WrongIssueTick {
            expected: expected_tick,
            got: distribution.issued_at_tick(),
        });
    }
    if distribution.horizon() != requested_horizon {
        return Err(BindingViolation::WrongHorizon {
            expected: requested_horizon,
            got: distribution.horizon(),
        });
    }
    if distribution.outcome_space().0.as_str() != WETLAND_STRESS_OUTCOME_SPACE {
        return Err(BindingViolation::WrongOutcomeSpace {
            expected: WETLAND_STRESS_OUTCOME_SPACE.into(),
            got: distribution.outcome_space().0.clone(),
        });
    }
    let unsupported = distribution.unsupported_mass().get();
    if unsupported != 0.0 {
        return Err(BindingViolation::UnsupportedMass { value: unsupported });
    }

    let mut seen_true = false;
    let mut seen_false = false;
    if distribution.branches().len() != 2 {
        return Err(BindingViolation::NonBinarySupport);
    }
    for branch in distribution.branches() {
        match &branch.outcome {
            OutcomeRegion::Boolean(true) if !seen_true => seen_true = true,
            OutcomeRegion::Boolean(false) if !seen_false => seen_false = true,
            _ => return Err(BindingViolation::NonBinarySupport),
        }
    }
    if !(seen_true && seen_false) {
        return Err(BindingViolation::NonBinarySupport);
    }
    Ok(())
}

fn issue_candidate(
    candidate: &Candidate<'_>,
    fixture: &SealedWatershedFixture,
    origin: usize,
) -> Result<IssuedForecast> {
    let requested_horizon = Horizon(1);
    let output = candidate
        .generator
        .generate(fixture.forecast_history(), requested_horizon);
    if let ForecastOutput::Distribution(distribution) = &output {
        validate_distribution_binding(
            distribution,
            fixture.forecast_history(),
            requested_horizon,
        )?;
    }
    let output_digest = digest_serializable(&(
        OUTPUT_SCHEMA,
        candidate.id,
        origin,
        fixture.dataset_manifest_digest(),
        &output,
    ))?;
    Ok(IssuedForecast {
        forecaster_id: candidate.id.into(),
        origin,
        history_dataset_digest: fixture.dataset_manifest_digest().into(),
        output,
        output_digest,
    })
}

fn reveal_committed_outcome(
    template: &SyntheticWatershedSpec,
    commitment: &StepCommitment,
) -> Result<WetlandObservation> {
    let next_origin = commitment
        .origin
        .checked_add(1)
        .ok_or_else(|| PrequentialError::InvalidSpec("reveal origin overflow".into()))?;
    let reveal_fixture = fixture_at_origin(template, next_origin)?;
    let actual = reveal_fixture
        .forecast_history()
        .last()
        .cloned()
        .ok_or_else(|| PrequentialError::InvalidSpec("reveal fixture had empty history".into()))?;
    if actual.tick != commitment.origin as u64 {
        return Err(PrequentialError::InvalidSpec(format!(
            "reveal tick mismatch: expected {}, got {}",
            commitment.origin, actual.tick
        )));
    }
    let reconstructed = digest_serializable(&(
        VERIFICATION_SCHEMA,
        commitment.dataset_manifest_digest.as_str(),
        &actual,
    ))?;
    if reconstructed != commitment.verification_commitment_digest {
        return Err(PrequentialError::VerificationCommitmentMismatch {
            origin: commitment.origin,
        });
    }
    Ok(actual)
}

fn resolve_forecast(issued: IssuedForecast, actual: &WetlandObservation) -> Result<ResolvedForecast> {
    let brier_score = match &issued.output {
        ForecastOutput::Distribution(distribution) => Some(
            BrierScore
                .score(distribution, &OutcomeRegion::Boolean(actual.wetland_stress))
                .map_err(|error| PrequentialError::Scoring(error.to_string()))?
                .get(),
        ),
        ForecastOutput::Abstain(_) => None,
    };
    Ok(ResolvedForecast {
        issued,
        brier_score,
    })
}

/// Evaluate all candidates over the fixed rolling-origin plan.
///
/// At each origin every candidate output is obtained, semantically validated, and content-digested
/// before `reveal_committed_outcome` is called. A binding failure from any candidate aborts the
/// origin before verification is revealed or any candidate is scored.
pub fn evaluate_candidates(
    plan: &PrequentialEpisodePlan,
    candidates: &[Candidate<'_>],
) -> Result<PrequentialEvaluation> {
    plan.verify_digest()?;
    if candidates.is_empty() {
        return Err(PrequentialError::EmptyCandidateSet);
    }
    let mut ids = HashSet::new();
    for candidate in candidates {
        if !ids.insert(candidate.id) {
            return Err(PrequentialError::DuplicateCandidateId(candidate.id.into()));
        }
    }

    let mut steps = Vec::with_capacity(plan.commitments.len());
    for commitment in &plan.commitments {
        let fixture = fixture_at_origin(&plan.spec.template, commitment.origin)?;
        let verification_digest = fixture
            .verification_digest()
            .map_err(|error| PrequentialError::Fixture(error.to_string()))?;
        if fixture.dataset_manifest_digest() != commitment.dataset_manifest_digest
            || verification_digest != commitment.verification_commitment_digest
        {
            return Err(PrequentialError::FixtureCommitmentMismatch {
                origin: commitment.origin,
            });
        }

        // Load-bearing ordering: issue EVERY candidate before revealing the next state.
        let mut issued = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            issued.push(issue_candidate(candidate, &fixture, commitment.origin)?);
        }

        let actual = reveal_committed_outcome(&plan.spec.template, commitment)?;
        let forecasts = issued
            .into_iter()
            .map(|forecast| resolve_forecast(forecast, &actual))
            .collect::<Result<Vec<_>>>()?;
        steps.push(PrequentialStepReport {
            origin: commitment.origin,
            history_dataset_digest: commitment.dataset_manifest_digest.clone(),
            verification_commitment_digest: commitment.verification_commitment_digest.clone(),
            actual,
            forecasts,
        });
    }

    let mut aggregates = Vec::with_capacity(candidates.len());
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let mut score_sum = 0.0;
        let mut scored_steps = 0usize;
        let mut abstained_steps = 0usize;
        for step in &steps {
            let report = &step.forecasts[candidate_index];
            match report.brier_score {
                Some(score) => {
                    score_sum += score;
                    scored_steps += 1;
                }
                None => abstained_steps += 1,
            }
        }
        let total_steps = steps.len();
        let mean_brier_scored_cases = if scored_steps == 0 {
            None
        } else {
            Some(score_sum / scored_steps as f64)
        };
        let coverage = scored_steps as f64 / total_steps as f64;
        aggregates.push(ForecasterAggregate {
            forecaster_id: candidate.id.into(),
            mean_brier_scored_cases,
            scored_steps,
            abstained_steps,
            total_steps,
            coverage,
        });
    }

    let forecast_ledger_digest = digest_serializable(&(
        LEDGER_SCHEMA,
        plan.plan_digest.as_str(),
        &steps,
        &aggregates,
    ))?;
    Ok(PrequentialEvaluation {
        episode_plan_digest: plan.plan_digest.clone(),
        steps,
        aggregates,
        forecast_ledger_digest,
    })
}

#[derive(Debug, Clone)]
pub struct PrequentialExecution {
    pub episode_plan_digest: String,
    pub evaluation: PrequentialEvaluation,
    pub result_manifest: ResearchResultManifest,
}

pub fn frozen_prequential_protocol(
    frozen_at_unix_ms: i64,
    first_origin: usize,
    evaluation_steps: usize,
) -> Result<FrozenProtocol> {
    if first_origin == 0 || evaluation_steps == 0 {
        return Err(PrequentialError::InvalidSpec(
            "first_origin and evaluation_steps must both be > 0".into(),
        ));
    }
    first_origin
        .checked_add(evaluation_steps)
        .ok_or_else(|| PrequentialError::InvalidSpec("origin range overflow".into()))?;
    let evaluation_steps_u64 = u64::try_from(evaluation_steps)
        .map_err(|_| PrequentialError::InvalidSpec("evaluation_steps exceeds u64".into()))?;
    let analysis = AnalysisPlanRef::new(
        "living-watershed-prequential-v1-analysis",
        "1",
        analysis_plan_digest(),
    )
    .map_err(|error| PrequentialError::Protocol(error.to_string()))?;
    let hypothesis = HypothesisSpec::new(
        "lw-preq-h1-commit-reveal",
        "Across the fixed rolling-origin episode, each declared baseline will either emit an exactly bound binary wetland-stress distribution or a typed abstention before the next state is revealed; invalid forecast bindings are not scoreable.",
        HypothesisRole::Primary,
        HypothesisDirection::Qualitative,
    )
    .map_err(|error| PrequentialError::Protocol(error.to_string()))?;

    let metrics = vec![
        MetricSpec::new(
            "persistence-mean-brier",
            "Persistence mean multiclass Brier score over scored cases",
            PREQUENTIAL_BRIER_UNIT,
            MetricRole::Primary,
            "arithmetic mean over issued distributions only; coverage is a mandatory companion metric",
        )
        .map_err(|error| PrequentialError::Protocol(error.to_string()))?,
        MetricSpec::new(
            "climatology-mean-brier",
            "Empirical climatology mean multiclass Brier score over scored cases",
            PREQUENTIAL_BRIER_UNIT,
            MetricRole::Primary,
            "arithmetic mean over issued distributions only; coverage is a mandatory companion metric",
        )
        .map_err(|error| PrequentialError::Protocol(error.to_string()))?,
        MetricSpec::new(
            "persistence-coverage",
            "Persistence forecast coverage",
            "fraction",
            MetricRole::Safety,
            "scored cases divided by preregistered evaluation cases",
        )
        .map_err(|error| PrequentialError::Protocol(error.to_string()))?,
        MetricSpec::new(
            "climatology-coverage",
            "Empirical climatology forecast coverage",
            "fraction",
            MetricRole::Safety,
            "scored cases divided by preregistered evaluation cases",
        )
        .map_err(|error| PrequentialError::Protocol(error.to_string()))?,
        MetricSpec::new(
            "evaluation-cases",
            "Resolved rolling-origin verification cases",
            "count",
            MetricRole::Secondary,
            "fixed count",
        )
        .map_err(|error| PrequentialError::Protocol(error.to_string()))?,
    ];

    let protocol = ResearchProtocol::new(
        "living-watershed-prequential-v1",
        "1",
        format!(
            "Can the Living Watershed mechanism preserve forecast-before-reveal ordering and exact forecast binding across {evaluation_steps} sequential one-step cases beginning at origin {first_origin}?"
        ),
        vec![hypothesis],
        metrics,
        vec![
            BaselineSpec::new(
                "persistence-v0",
                "One-step persistence with fixed 0.8 confidence.",
                "symthaea-living-watershed-witness::PersistenceForecaster/v0",
            )
            .map_err(|error| PrequentialError::Protocol(error.to_string()))?,
            BaselineSpec::new(
                "empirical-climatology-v0",
                "Empirical predictor-visible stress frequency with minimum history 3.",
                "symthaea-living-watershed-witness::ClimatologyForecaster/v0",
            )
            .map_err(|error| PrequentialError::Protocol(error.to_string()))?,
        ],
        vec![],
        StoppingRule::FixedSampleCount(evaluation_steps_u64),
        MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
        analysis,
        format!(
            "One content-addressed precommitted synthetic rolling-origin episode; first origin {first_origin}; exactly {evaluation_steps} sequential one-step verification cases. All candidate outputs at an origin must commit before the next state is revealed."
        ),
        "No RNG is used by the synthetic fixture or declared baselines; the registered run still binds an explicit seed-manifest digest for lineage compatibility with future stochastic candidates.",
    )
    .map_err(|error| PrequentialError::Protocol(error.to_string()))?;
    protocol
        .freeze(frozen_at_unix_ms)
        .map_err(|error| PrequentialError::Protocol(error.to_string()))
}

fn verify_protocol_matches_plan(
    frozen: &FrozenProtocol,
    plan: &PrequentialEpisodePlan,
) -> Result<()> {
    frozen
        .verify_digest()
        .map_err(|error| PrequentialError::Protocol(error.to_string()))?;
    let expected = frozen_prequential_protocol(
        frozen.frozen_at_unix_ms(),
        plan.spec.first_origin(),
        plan.spec.evaluation_steps,
    )?;
    if frozen.digest() != expected.digest() {
        return Err(PrequentialError::ProtocolDesignMismatch);
    }
    Ok(())
}

fn aggregate<'a>(evaluation: &'a PrequentialEvaluation, id: &str) -> Result<&'a ForecasterAggregate> {
    evaluation
        .aggregates
        .iter()
        .find(|aggregate| aggregate.forecaster_id == id)
        .ok_or_else(|| PrequentialError::InvalidSpec(format!("missing aggregate for {id}")))
}

fn mean_score_metric(metric_id: &str, aggregate: &ForecasterAggregate) -> Result<MetricResult> {
    let outcome = match aggregate.mean_brier_scored_cases {
        Some(value) => MetricOutcome::Numeric {
            value,
            unit: PREQUENTIAL_BRIER_UNIT.into(),
        },
        None => MetricOutcome::NotComputed {
            reason: format!(
                "{} abstained on all {} preregistered cases",
                aggregate.forecaster_id, aggregate.total_steps
            ),
        },
    };
    MetricResult::new(metric_id, outcome)
        .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
        .with_artifact("forecast-ledger")
        .map_err(|error| PrequentialError::ResearchResult(error.to_string()))
}

fn coverage_metric(metric_id: &str, aggregate: &ForecasterAggregate) -> Result<MetricResult> {
    MetricResult::new(
        metric_id,
        MetricOutcome::Numeric {
            value: aggregate.coverage,
            unit: "fraction".into(),
        },
    )
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
    .with_artifact("forecast-ledger")
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))
}

pub fn run_prequential_baselines(
    frozen: &FrozenProtocol,
    plan: &PrequentialEpisodePlan,
    lineage: WitnessRunLineage,
) -> Result<PrequentialExecution> {
    plan.verify_digest()?;
    verify_protocol_matches_plan(frozen, plan)?;
    let run = ResearchRunRegistration::new(
        frozen,
        lineage.run_id,
        lineage.registered_at_unix_ms,
        lineage.source_commit,
        plan.plan_digest.clone(),
        lineage.reproducibility_capsule_digest,
        lineage.seed_manifest_digest,
    )
    .map_err(|error| PrequentialError::Protocol(error.to_string()))?;

    let persistence = PersistenceForecaster::default();
    let climatology = ClimatologyForecaster::default();
    let candidates = [
        Candidate::new("persistence-v0", &persistence)?,
        Candidate::new("empirical-climatology-v0", &climatology)?,
    ];
    let evaluation = evaluate_candidates(plan, &candidates)?;
    let persistence_aggregate = aggregate(&evaluation, "persistence-v0")?;
    let climatology_aggregate = aggregate(&evaluation, "empirical-climatology-v0")?;

    let verification_ledger_digest = digest_serializable(
        &evaluation
            .steps
            .iter()
            .map(|step| {
                (
                    step.origin,
                    step.verification_commitment_digest.as_str(),
                    &step.actual,
                )
            })
            .collect::<Vec<_>>(),
    )?;

    let artifacts = vec![
        ResultArtifactRef::new(
            "analysis-plan",
            ResultArtifactKind::Analysis,
            analysis_plan_digest(),
            "Frozen Living Watershed prequential v1 analysis plan",
        )
        .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?,
        ResultArtifactRef::new(
            "episode-plan",
            ResultArtifactKind::RawOutput,
            plan.plan_digest.clone(),
            "Pre-run rolling-origin fixture and verification commitments",
        )
        .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?,
        ResultArtifactRef::new(
            "forecast-ledger",
            ResultArtifactKind::ForecastLedger,
            evaluation.forecast_ledger_digest.clone(),
            "All pre-reveal candidate outputs, resolutions, scores, abstentions, and coverage",
        )
        .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?,
        ResultArtifactRef::new(
            "verification-ledger",
            ResultArtifactKind::Verification,
            verification_ledger_digest,
            "Revealed outcomes bound back to their pre-run verification commitments",
        )
        .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?,
    ];

    let metrics = vec![
        mean_score_metric("persistence-mean-brier", persistence_aggregate)?,
        mean_score_metric("climatology-mean-brier", climatology_aggregate)?,
        coverage_metric("persistence-coverage", persistence_aggregate)?,
        coverage_metric("climatology-coverage", climatology_aggregate)?,
        MetricResult::new(
            "evaluation-cases",
            MetricOutcome::Numeric {
                value: evaluation.steps.len() as f64,
                unit: "count".into(),
            },
        )
        .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
        .with_artifact("verification-ledger")
        .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?,
    ];

    let claim = ResultClaim::new(
        "lw-preq-claim-commit-reveal",
        "Every preregistered rolling-origin case completed with all candidate outputs committed before reveal; every scored distribution satisfied exact issue-tick, horizon, outcome-space, binary-support, and unsupported-mass bindings; abstentions and coverage were retained explicitly.",
        ClaimDisposition::ConsistentWithHypothesis,
        ClaimInterpretation::Confirmatory,
    )
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
    .for_hypothesis("lw-preq-h1-commit-reveal")
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
    .with_metric("persistence-mean-brier")
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
    .with_metric("climatology-mean-brier")
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
    .with_metric("persistence-coverage")
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
    .with_metric("climatology-coverage")
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
    .with_artifact("forecast-ledger")
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?
    .with_artifact("verification-ledger")
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?;

    let result_manifest = ResearchResultManifest::new(
        frozen,
        run,
        lineage.manifest_id,
        lineage.completed_at_unix_ms,
        vec![],
        vec![],
        false,
        artifacts,
        metrics,
        vec![claim],
    )
    .map_err(|error| PrequentialError::ResearchResult(error.to_string()))?;

    Ok(PrequentialExecution {
        episode_plan_digest: plan.plan_digest.clone(),
        evaluation,
        result_manifest,
    })
}

/// Helper for tests and adapters that need to construct the exact target space explicitly.
pub fn wetland_outcome_space() -> OutcomeSpaceId {
    OutcomeSpaceId(WETLAND_STRESS_OUTCOME_SPACE.into())
}
