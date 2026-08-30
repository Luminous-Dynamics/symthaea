//! Deterministic Living Watershed / Wetland Watch research witness.
//!
//! This crate is deliberately a **mechanism witness**, not a real-world wetland predictor.
//! It composes existing reduced-order Earth-system hydrology, ecology moisture response,
//! Futures Laboratory forecast/scoring contracts, and the research protocol/result/replication
//! evidence stack.
//!
//! The load-bearing boundary is an observation firewall: trajectory generators receive only
//! [`WatershedHistory`]. The held-out next state remains private inside [`SealedWatershedFixture`]
//! and is supplied to the canonical Futures scorer only after a forecast has been emitted.

use serde::{Deserialize, Serialize};
use symthaea_ecology::SoilMoistureResponse;
use symthaea_earth_system::HydrologyBucket;
use symthaea_futures_calibration::{BrierScore, ScoringRule};
use symthaea_futures_core::{
    AbstentionReason, AssumptionId, ForecastDistribution, ForecastOutput, Horizon, OutcomeRegion,
    OutcomeSpaceId, TrajectoryGenerator,
};
use symthaea_research_protocol::{
    AnalysisPlanRef, BaselineSpec, FrozenProtocol, HypothesisDirection, HypothesisRole,
    HypothesisSpec, MetricRole, MetricSpec, MultiplicityPolicy, ResearchProtocol,
    ResearchRunRegistration, StoppingRule,
};
use symthaea_research_replication::{
    ReplicationAssessment, ReplicationComparisonEvidence, ReplicationDesign, ReplicationOutcome,
};
use symthaea_research_result::{
    ClaimDisposition, ClaimInterpretation, MetricOutcome, MetricResult, ResearchResultManifest,
    ResultArtifactKind, ResultArtifactRef, ResultClaim,
};
use thiserror::Error;

const FIXTURE_SCHEMA: &str = "symthaea-living-watershed-fixture/v0";
const FORECAST_LEDGER_SCHEMA: &str = "symthaea-living-watershed-forecast-ledger/v0";
const OUTCOME_SPACE: &str = "living-watershed/wetland-stress-next-step/v0";
const BRIER_UNIT: &str = "brier_multiclass";
const ANALYSIS_PLAN: &str = r#"Living Watershed v0 analysis plan

Purpose: validate the end-to-end evidence mechanics, not real-world ecological skill.

1. Generate a deterministic reduced-order hydrology trajectory.
2. Convert bounded soil-moisture state into a bounded ecology moisture-response multiplier.
3. Seal the final trajectory state as the one-step held-out verification outcome.
4. Give forecasters only the preceding history.
5. Evaluate two declared baselines: persistence and empirical climatology.
6. If a baseline emits a distribution, score it with the canonical Futures Laboratory multiclass Brier scorer.
7. If a baseline abstains, retain that abstention and record its primary score as NotComputed with a reason.
8. Retain all primary metrics and the digests of analysis, fixture, forecast ledger, and verification evidence.

This plan does not test Sentinel accuracy, wetland ecology validity, HDC benefit, or policy utility.
"#;
const REPLICATION_COMPARISON_PLAN: &str = r#"Living Watershed v0 replication comparison

Compare result-manifest lineage under the caller-declared replication design. Preserve exact
same/different protocol/source/data/environment/seed relations. The caller must classify the
scientific outcome explicitly; this helper does not infer concordance from score similarity.
"#;

#[derive(Debug, Error)]
pub enum WitnessError {
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("invalid watershed specification: {0}")]
    InvalidSpec(String),
    #[error("earth-system model failed: {0}")]
    EarthSystem(String),
    #[error("ecology model failed: {0}")]
    Ecology(String),
    #[error("forecast scoring failed: {0}")]
    Scoring(String),
    #[error("serialization failed: {0}")]
    Serialization(String),
    #[error("research protocol failed: {0}")]
    Protocol(String),
    #[error("research result failed: {0}")]
    ResearchResult(String),
    #[error("replication assessment failed: {0}")]
    Replication(String),
}

pub type Result<T> = std::result::Result<T, WitnessError>;

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        Err(WitnessError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn digest_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(|error| WitnessError::Serialization(error.to_string()))?;
    Ok(digest_bytes(&bytes))
}

fn analysis_plan_digest() -> String {
    digest_bytes(ANALYSIS_PLAN.as_bytes())
}

fn replication_plan_digest() -> String {
    digest_bytes(REPLICATION_COMPARISON_PLAN.as_bytes())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticWatershedSpec {
    pub fixture_id: String,
    pub capacity_mm: f64,
    pub potential_evapotranspiration_mm_per_day: f64,
    pub initial_storage_mm: f64,
    pub precipitation_mm_per_day: f64,
    /// Number of observations visible to the predictor. One additional state is generated and sealed.
    pub history_steps: usize,
    pub wilting_fraction: f64,
    pub optimum_fraction: f64,
    pub minimum_moisture_multiplier: f64,
    /// `wetland_stress = ecological_moisture_multiplier < threshold`.
    pub stress_multiplier_threshold: f64,
}

impl SyntheticWatershedSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fixture_id: impl Into<String>,
        capacity_mm: f64,
        potential_evapotranspiration_mm_per_day: f64,
        initial_storage_mm: f64,
        precipitation_mm_per_day: f64,
        history_steps: usize,
        wilting_fraction: f64,
        optimum_fraction: f64,
        minimum_moisture_multiplier: f64,
        stress_multiplier_threshold: f64,
    ) -> Result<Self> {
        let fixture_id = fixture_id.into();
        non_empty(&fixture_id, "fixture id")?;
        if history_steps == 0 {
            return Err(WitnessError::InvalidSpec("history_steps must be > 0".into()));
        }
        if !stress_multiplier_threshold.is_finite()
            || !(0.0..=1.0).contains(&stress_multiplier_threshold)
        {
            return Err(WitnessError::InvalidSpec(
                "stress_multiplier_threshold must be finite and in [0, 1]".into(),
            ));
        }
        HydrologyBucket::try_new(capacity_mm, potential_evapotranspiration_mm_per_day)
            .map_err(|error| WitnessError::EarthSystem(error.to_string()))?
            .validate_storage(initial_storage_mm)
            .map_err(|error| WitnessError::EarthSystem(error.to_string()))?;
        if !precipitation_mm_per_day.is_finite() || precipitation_mm_per_day < 0.0 {
            return Err(WitnessError::InvalidSpec(
                "precipitation_mm_per_day must be finite and non-negative".into(),
            ));
        }
        SoilMoistureResponse::try_new(
            wilting_fraction,
            optimum_fraction,
            minimum_moisture_multiplier,
        )
        .map_err(|error| WitnessError::Ecology(error.to_string()))?;
        Ok(Self {
            fixture_id,
            capacity_mm,
            potential_evapotranspiration_mm_per_day,
            initial_storage_mm,
            precipitation_mm_per_day,
            history_steps,
            wilting_fraction,
            optimum_fraction,
            minimum_moisture_multiplier,
            stress_multiplier_threshold,
        })
    }

    /// Deterministic dry-down fixture used by tests and examples.
    pub fn drydown(fixture_id: impl Into<String>, history_steps: usize) -> Result<Self> {
        Self::new(
            fixture_id,
            100.0,
            5.0,
            90.0,
            0.5,
            history_steps,
            0.20,
            0.70,
            0.10,
            0.55,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WetlandObservation {
    pub tick: u64,
    pub soil_moisture_fraction: f64,
    pub ecological_moisture_multiplier: f64,
    pub wetland_stress: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WatershedHistory {
    observations: Vec<WetlandObservation>,
}

impl WatershedHistory {
    pub fn observations(&self) -> &[WetlandObservation] {
        &self.observations
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    pub fn last(&self) -> Option<&WetlandObservation> {
        self.observations.last()
    }
}

/// Dataset fixture with a deliberately private held-out outcome.
///
/// `TrajectoryGenerator<Observation = WatershedHistory>` implementations cannot receive this
/// type through the witness runner, so the held-out state is not part of their input surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealedWatershedFixture {
    spec: SyntheticWatershedSpec,
    history: WatershedHistory,
    held_out: WetlandObservation,
    dataset_manifest_digest: String,
}

#[derive(Serialize)]
struct FixtureDigestView<'a> {
    schema: &'static str,
    spec: &'a SyntheticWatershedSpec,
    history: &'a WatershedHistory,
    held_out: &'a WetlandObservation,
}

impl SealedWatershedFixture {
    pub fn generate(spec: SyntheticWatershedSpec) -> Result<Self> {
        let bucket = HydrologyBucket::try_new(
            spec.capacity_mm,
            spec.potential_evapotranspiration_mm_per_day,
        )
        .map_err(|error| WitnessError::EarthSystem(error.to_string()))?;
        let response = SoilMoistureResponse::try_new(
            spec.wilting_fraction,
            spec.optimum_fraction,
            spec.minimum_moisture_multiplier,
        )
        .map_err(|error| WitnessError::Ecology(error.to_string()))?;
        let samples = bucket
            .exact_trajectory(
                spec.initial_storage_mm,
                spec.precipitation_mm_per_day,
                1.0,
                spec.history_steps,
            )
            .map_err(|error| WitnessError::EarthSystem(error.to_string()))?;

        let mut observations = Vec::with_capacity(samples.len());
        for (tick, sample) in samples.into_iter().enumerate() {
            if sample.budget_residual_mm.abs() > 1.0e-8 {
                return Err(WitnessError::EarthSystem(format!(
                    "water budget residual exceeded witness tolerance at tick {tick}: {} mm",
                    sample.budget_residual_mm
                )));
            }
            let (ecological_moisture_multiplier, _) = response
                .multiplier_and_derivative(sample.soil_moisture_fraction)
                .map_err(|error| WitnessError::Ecology(error.to_string()))?;
            let wetland_stress = ecological_moisture_multiplier < spec.stress_multiplier_threshold;
            observations.push(WetlandObservation {
                tick: tick as u64,
                soil_moisture_fraction: sample.soil_moisture_fraction,
                ecological_moisture_multiplier,
                wetland_stress,
            });
        }
        let held_out = observations
            .pop()
            .ok_or_else(|| WitnessError::InvalidSpec("fixture produced no held-out state".into()))?;
        let history = WatershedHistory { observations };
        if history.is_empty() {
            return Err(WitnessError::InvalidSpec(
                "fixture must retain at least one predictor-visible observation".into(),
            ));
        }

        let digest_view = FixtureDigestView {
            schema: FIXTURE_SCHEMA,
            spec: &spec,
            history: &history,
            held_out: &held_out,
        };
        let dataset_manifest_digest = digest_serializable(&digest_view)?;
        Ok(Self {
            spec,
            history,
            held_out,
            dataset_manifest_digest,
        })
    }

    pub fn fixture_id(&self) -> &str {
        &self.spec.fixture_id
    }

    /// The only predictor-facing view exposed by the witness runner.
    pub fn forecast_history(&self) -> &WatershedHistory {
        &self.history
    }

    pub fn dataset_manifest_digest(&self) -> &str {
        &self.dataset_manifest_digest
    }

    pub fn verification_digest(&self) -> Result<String> {
        digest_serializable(&(
            "symthaea-living-watershed-verification/v0",
            self.dataset_manifest_digest.as_str(),
            &self.held_out,
        ))
    }

    fn actual_outcome(&self) -> OutcomeRegion {
        OutcomeRegion::Boolean(self.held_out.wetland_stress)
    }

    fn score_brier(&self, distribution: &ForecastDistribution) -> Result<f64> {
        BrierScore
            .score(distribution, &self.actual_outcome())
            .map(|score| score.get())
            .map_err(|error| WitnessError::Scoring(error.to_string()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistenceForecaster {
    confidence: f64,
}

impl PersistenceForecaster {
    pub fn new(confidence: f64) -> Result<Self> {
        if !confidence.is_finite() || !(0.5..=1.0).contains(&confidence) {
            return Err(WitnessError::InvalidSpec(
                "persistence confidence must be finite and in [0.5, 1]".into(),
            ));
        }
        Ok(Self { confidence })
    }
}

impl Default for PersistenceForecaster {
    fn default() -> Self {
        Self { confidence: 0.8 }
    }
}

impl TrajectoryGenerator for PersistenceForecaster {
    type Observation = WatershedHistory;

    fn generate(&self, history: &Self::Observation, horizon: Horizon) -> ForecastOutput {
        if horizon != Horizon(1) {
            return ForecastOutput::Abstain(AbstentionReason::HorizonBeyondValidatedRange);
        }
        let Some(last) = history.last() else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };
        let p_true = if last.wetland_stress {
            self.confidence
        } else {
            1.0 - self.confidence
        };
        boolean_forecast(last.tick, p_true, "persistence-v0")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClimatologyForecaster {
    min_history: usize,
}

impl ClimatologyForecaster {
    pub fn new(min_history: usize) -> Result<Self> {
        if min_history == 0 {
            return Err(WitnessError::InvalidSpec(
                "climatology min_history must be > 0".into(),
            ));
        }
        Ok(Self { min_history })
    }
}

impl Default for ClimatologyForecaster {
    fn default() -> Self {
        Self { min_history: 3 }
    }
}

impl TrajectoryGenerator for ClimatologyForecaster {
    type Observation = WatershedHistory;

    fn generate(&self, history: &Self::Observation, horizon: Horizon) -> ForecastOutput {
        if horizon != Horizon(1) {
            return ForecastOutput::Abstain(AbstentionReason::HorizonBeyondValidatedRange);
        }
        if history.len() < self.min_history {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        }
        let stressed = history
            .observations()
            .iter()
            .filter(|observation| observation.wetland_stress)
            .count();
        let p_true = stressed as f64 / history.len() as f64;
        let issued_at_tick = history.last().map_or(0, |observation| observation.tick);
        boolean_forecast(issued_at_tick, p_true, "empirical-climatology-v0")
    }
}

fn boolean_forecast(issued_at_tick: u64, p_true: f64, assumption: &str) -> ForecastOutput {
    let branches = vec![
        (
            p_true,
            OutcomeRegion::Boolean(true),
            vec![AssumptionId(assumption.into())],
        ),
        (
            1.0 - p_true,
            OutcomeRegion::Boolean(false),
            vec![AssumptionId(assumption.into())],
        ),
    ];
    match ForecastDistribution::try_from_raw(
        issued_at_tick,
        Horizon(1),
        OutcomeSpaceId(OUTCOME_SPACE.into()),
        branches,
        0.0,
    ) {
        Ok(distribution) => ForecastOutput::Distribution(distribution),
        Err(_) => ForecastOutput::Abstain(AbstentionReason::UnresolvedOutcomeSpace),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecasterReport {
    pub forecaster_id: String,
    pub output: ForecastOutput,
    pub brier_score: Option<f64>,
    pub output_digest: String,
}

impl ForecasterReport {
    fn from_output(
        forecaster_id: &str,
        output: ForecastOutput,
        fixture: &SealedWatershedFixture,
    ) -> Result<Self> {
        let brier_score = match &output {
            ForecastOutput::Distribution(distribution) => Some(fixture.score_brier(distribution)?),
            ForecastOutput::Abstain(_) => None,
        };
        let output_digest = digest_serializable(&(
            "symthaea-living-watershed-forecast-output/v0",
            forecaster_id,
            fixture.dataset_manifest_digest(),
            &output,
        ))?;
        Ok(Self {
            forecaster_id: forecaster_id.into(),
            output,
            brier_score,
            output_digest,
        })
    }

    pub fn abstention_reason(&self) -> Option<AbstentionReason> {
        match self.output {
            ForecastOutput::Distribution(_) => None,
            ForecastOutput::Abstain(reason) => Some(reason),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WitnessRunLineage {
    pub run_id: String,
    pub manifest_id: String,
    pub registered_at_unix_ms: i64,
    pub completed_at_unix_ms: i64,
    pub source_commit: String,
    pub reproducibility_capsule_digest: String,
    pub seed_manifest_digest: String,
}

impl WitnessRunLineage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: impl Into<String>,
        manifest_id: impl Into<String>,
        registered_at_unix_ms: i64,
        completed_at_unix_ms: i64,
        source_commit: impl Into<String>,
        reproducibility_capsule_digest: impl Into<String>,
        seed_manifest_digest: impl Into<String>,
    ) -> Result<Self> {
        let run_id = run_id.into();
        let manifest_id = manifest_id.into();
        let source_commit = source_commit.into();
        let reproducibility_capsule_digest = reproducibility_capsule_digest.into();
        let seed_manifest_digest = seed_manifest_digest.into();
        non_empty(&run_id, "run id")?;
        non_empty(&manifest_id, "manifest id")?;
        non_empty(&source_commit, "source commit")?;
        non_empty(
            &reproducibility_capsule_digest,
            "reproducibility capsule digest",
        )?;
        non_empty(&seed_manifest_digest, "seed manifest digest")?;
        if completed_at_unix_ms < registered_at_unix_ms {
            return Err(WitnessError::InvalidSpec(
                "completed_at_unix_ms cannot precede registration".into(),
            ));
        }
        Ok(Self {
            run_id,
            manifest_id,
            registered_at_unix_ms,
            completed_at_unix_ms,
            source_commit,
            reproducibility_capsule_digest,
            seed_manifest_digest,
        })
    }
}

#[derive(Debug, Clone)]
pub struct LivingWatershedExecution {
    pub fixture_id: String,
    pub dataset_manifest_digest: String,
    pub persistence: ForecasterReport,
    pub climatology: ForecasterReport,
    pub result_manifest: ResearchResultManifest,
}

pub fn frozen_witness_protocol(frozen_at_unix_ms: i64) -> Result<FrozenProtocol> {
    let analysis_plan = AnalysisPlanRef::new(
        "living-watershed-v0-analysis",
        "0",
        analysis_plan_digest(),
    )
    .map_err(|error| WitnessError::Protocol(error.to_string()))?;
    let hypothesis = HypothesisSpec::new(
        "lw-h1-scoreable-baselines",
        "On a sufficiently long deterministic watershed history, the declared one-step persistence and empirical-climatology baselines emit canonical scoreable wetland-stress forecasts.",
        HypothesisRole::Primary,
        HypothesisDirection::Qualitative,
    )
    .map_err(|error| WitnessError::Protocol(error.to_string()))?;
    let persistence_metric = MetricSpec::new(
        "persistence-brier",
        "Persistence baseline multiclass Brier score",
        BRIER_UNIT,
        MetricRole::Primary,
        "single held-out one-step outcome",
    )
    .map_err(|error| WitnessError::Protocol(error.to_string()))?
    .with_success_criterion("A finite canonical Futures Laboratory score is produced when the baseline emits a distribution.")
    .map_err(|error| WitnessError::Protocol(error.to_string()))?;
    let climatology_metric = MetricSpec::new(
        "climatology-brier",
        "Empirical climatology baseline multiclass Brier score",
        BRIER_UNIT,
        MetricRole::Primary,
        "single held-out one-step outcome",
    )
    .map_err(|error| WitnessError::Protocol(error.to_string()))?
    .with_success_criterion("A finite canonical Futures Laboratory score is produced when minimum history is satisfied; otherwise NotComputed must retain the typed abstention.")
    .map_err(|error| WitnessError::Protocol(error.to_string()))?;
    let climatology_output_metric = MetricSpec::new(
        "climatology-output-kind",
        "Empirical climatology output class",
        "category",
        MetricRole::Safety,
        "single run disposition",
    )
    .map_err(|error| WitnessError::Protocol(error.to_string()))?;
    let protocol = ResearchProtocol::new(
        "living-watershed-v0",
        "0",
        "Can the Living Watershed v0 mechanism carry a sealed deterministic watershed history through neutral forecast/abstention, canonical scoring, and immutable result lineage without exposing the held-out outcome to the predictor?",
        vec![hypothesis],
        vec![persistence_metric, climatology_metric, climatology_output_metric],
        vec![
            BaselineSpec::new(
                "persistence-v0",
                "One-step persistence with fixed 0.8 confidence in the last observed stress state.",
                "symthaea-living-watershed-witness::PersistenceForecaster/v0",
            )
            .map_err(|error| WitnessError::Protocol(error.to_string()))?,
            BaselineSpec::new(
                "empirical-climatology-v0",
                "Empirical stress frequency over predictor-visible history, minimum history 3.",
                "symthaea-living-watershed-witness::ClimatologyForecaster/v0",
            )
            .map_err(|error| WitnessError::Protocol(error.to_string()))?,
        ],
        vec![],
        StoppingRule::FixedEpisodeCount(1),
        MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
        analysis_plan,
        "One content-addressed deterministic synthetic watershed fixture per run; held-out state is sealed from the TrajectoryGenerator input surface.",
        "No RNG is used by the v0 fixture or declared baselines; each run still binds an explicit seed-manifest digest so later stochastic lanes cannot silently inherit a no-seed convention.",
    )
    .map_err(|error| WitnessError::Protocol(error.to_string()))?;
    protocol
        .freeze(frozen_at_unix_ms)
        .map_err(|error| WitnessError::Protocol(error.to_string()))
}

pub fn run_witness(
    frozen: &FrozenProtocol,
    fixture: &SealedWatershedFixture,
    lineage: WitnessRunLineage,
) -> Result<LivingWatershedExecution> {
    let run = ResearchRunRegistration::new(
        frozen,
        lineage.run_id,
        lineage.registered_at_unix_ms,
        lineage.source_commit,
        fixture.dataset_manifest_digest().to_string(),
        lineage.reproducibility_capsule_digest,
        lineage.seed_manifest_digest,
    )
    .map_err(|error| WitnessError::Protocol(error.to_string()))?;

    // Observation firewall: only this history reference reaches either TrajectoryGenerator.
    let history = fixture.forecast_history();
    let persistence_output = PersistenceForecaster::default().generate(history, Horizon(1));
    let climatology_output = ClimatologyForecaster::default().generate(history, Horizon(1));

    // The held-out outcome enters only after both outputs exist, inside from_output -> score_brier.
    let persistence = ForecasterReport::from_output("persistence-v0", persistence_output, fixture)?;
    let climatology =
        ForecasterReport::from_output("empirical-climatology-v0", climatology_output, fixture)?;

    let forecast_ledger_digest = digest_serializable(&(
        FORECAST_LEDGER_SCHEMA,
        fixture.dataset_manifest_digest(),
        &persistence,
        &climatology,
    ))?;
    let verification_digest = fixture.verification_digest()?;
    let artifacts = vec![
        ResultArtifactRef::new(
            "analysis-plan",
            ResultArtifactKind::Analysis,
            analysis_plan_digest(),
            "Frozen Living Watershed v0 analysis plan",
        )
        .map_err(|error| WitnessError::ResearchResult(error.to_string()))?,
        ResultArtifactRef::new(
            "fixture-manifest",
            ResultArtifactKind::RawOutput,
            fixture.dataset_manifest_digest().to_string(),
            "Content-addressed sealed synthetic watershed fixture",
        )
        .map_err(|error| WitnessError::ResearchResult(error.to_string()))?,
        ResultArtifactRef::new(
            "forecast-ledger",
            ResultArtifactKind::ForecastLedger,
            forecast_ledger_digest,
            "Canonical baseline forecast/abstention outputs and score lineage",
        )
        .map_err(|error| WitnessError::ResearchResult(error.to_string()))?,
        ResultArtifactRef::new(
            "held-out-verification",
            ResultArtifactKind::Verification,
            verification_digest,
            "Held-out next-state verification digest disclosed to scoring only after forecast emission",
        )
        .map_err(|error| WitnessError::ResearchResult(error.to_string()))?,
    ];

    let persistence_metric = score_metric("persistence-brier", &persistence)?;
    let climatology_metric = score_metric("climatology-brier", &climatology)?;
    let climatology_kind = MetricResult::new(
        "climatology-output-kind",
        MetricOutcome::Categorical(output_kind_label(&climatology.output)),
    )
    .map_err(|error| WitnessError::ResearchResult(error.to_string()))?
    .with_artifact("forecast-ledger")
    .map_err(|error| WitnessError::ResearchResult(error.to_string()))?;

    let both_scoreable = persistence.brier_score.is_some() && climatology.brier_score.is_some();
    let claim = ResultClaim::new(
        "lw-claim-scoreable-baselines",
        if both_scoreable {
            "Both preregistered baseline lanes emitted canonical one-step distributions and received finite Futures Laboratory multiclass Brier scores."
        } else {
            "At least one preregistered baseline abstained; the missing primary score is retained explicitly rather than converted to a numeric sentinel."
        },
        if both_scoreable {
            ClaimDisposition::ConsistentWithHypothesis
        } else {
            ClaimDisposition::Inconclusive
        },
        ClaimInterpretation::Confirmatory,
    )
    .map_err(|error| WitnessError::ResearchResult(error.to_string()))?
    .for_hypothesis("lw-h1-scoreable-baselines")
    .map_err(|error| WitnessError::ResearchResult(error.to_string()))?
    .with_metric("persistence-brier")
    .map_err(|error| WitnessError::ResearchResult(error.to_string()))?
    .with_metric("climatology-brier")
    .map_err(|error| WitnessError::ResearchResult(error.to_string()))?
    .with_artifact("forecast-ledger")
    .map_err(|error| WitnessError::ResearchResult(error.to_string()))?;

    let result_manifest = ResearchResultManifest::new(
        frozen,
        run,
        lineage.manifest_id,
        lineage.completed_at_unix_ms,
        vec![],
        vec![],
        false,
        artifacts,
        vec![persistence_metric, climatology_metric, climatology_kind],
        vec![claim],
    )
    .map_err(|error| WitnessError::ResearchResult(error.to_string()))?;

    Ok(LivingWatershedExecution {
        fixture_id: fixture.fixture_id().into(),
        dataset_manifest_digest: fixture.dataset_manifest_digest().into(),
        persistence,
        climatology,
        result_manifest,
    })
}

fn score_metric(metric_id: &str, report: &ForecasterReport) -> Result<MetricResult> {
    let outcome = match report.brier_score {
        Some(value) => MetricOutcome::Numeric {
            value,
            unit: BRIER_UNIT.into(),
        },
        None => MetricOutcome::NotComputed {
            reason: match report.abstention_reason() {
                Some(reason) => format!("forecaster abstained: {}", abstention_reason_label(reason)),
                None => "forecaster emitted no scoreable distribution".into(),
            },
        },
    };
    MetricResult::new(metric_id, outcome)
        .map_err(|error| WitnessError::ResearchResult(error.to_string()))?
        .with_artifact("forecast-ledger")
        .map_err(|error| WitnessError::ResearchResult(error.to_string()))
}

fn output_kind_label(output: &ForecastOutput) -> String {
    match output {
        ForecastOutput::Distribution(_) => "distribution".into(),
        ForecastOutput::Abstain(reason) => format!("abstain:{}", abstention_reason_label(*reason)),
    }
}

fn abstention_reason_label(reason: AbstentionReason) -> &'static str {
    match reason {
        AbstentionReason::InsufficientObservationHistory => "insufficient-observation-history",
        AbstentionReason::OutOfDistributionScenario => "out-of-distribution-scenario",
        AbstentionReason::ModelDisagreementTooHigh => "model-disagreement-too-high",
        AbstentionReason::HorizonBeyondValidatedRange => "horizon-beyond-validated-range",
        AbstentionReason::UnresolvedOutcomeSpace => "unresolved-outcome-space",
        AbstentionReason::ObservationPolicyTooLossy => "observation-policy-too-lossy",
    }
}

/// Build an evidence-bearing replication assessment without inferring the scientific outcome.
///
/// `outcome` is explicit on purpose: similar numeric scores are not sufficient to declare two
/// scientific findings concordant without a frozen comparison rule appropriate to the claim.
pub fn assess_replication(
    assessment_id: impl Into<String>,
    design: ReplicationDesign,
    original: &LivingWatershedExecution,
    followup: &LivingWatershedExecution,
    outcome: ReplicationOutcome,
) -> Result<ReplicationAssessment> {
    let comparison = ReplicationComparisonEvidence::new(
        "living-watershed-v0-frozen-comparison-plan",
        replication_plan_digest(),
        "Mechanism-witness comparison only; caller supplies the outcome classification and no score similarity is promoted automatically to scientific concordance.",
    )
    .map_err(|error| WitnessError::Replication(error.to_string()))?;
    ReplicationAssessment::new(
        assessment_id,
        design,
        &original.result_manifest,
        &followup.result_manifest,
        vec![],
        outcome,
        comparison,
    )
    .map_err(|error| WitnessError::Replication(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_research_replication::LineageRelation;

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
        let execution = run_witness(&frozen, &fixture, lineage("run-short", "manifest-short", 2, 3)).unwrap();
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
        assert!(matches!(metric.outcome, MetricOutcome::NotComputed { .. }));
        execution.result_manifest.verify_digest().unwrap();
    }

    #[test]
    fn unsupported_horizon_abstains_instead_of_guessing() {
        let fixture = SealedWatershedFixture::generate(
            SyntheticWatershedSpec::drydown("watershed-a", 6).unwrap(),
        )
        .unwrap();
        let output = PersistenceForecaster::default().generate(fixture.forecast_history(), Horizon(2));
        assert!(matches!(
            output,
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
        assert_eq!(assessment.factual_lineage.dataset_manifest, LineageRelation::Same);
        assessment.verify_digest().unwrap();
        assert!(assess_replication(
            "not-direct",
            ReplicationDesign::DirectReplication,
            &first,
            &second,
            ReplicationOutcome::Concordant,
        )
        .is_err());
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
        assert_eq!(assessment.factual_lineage.dataset_manifest, LineageRelation::Different);
        assert_eq!(assessment.outcome, ReplicationOutcome::Inconclusive);
    }
}
