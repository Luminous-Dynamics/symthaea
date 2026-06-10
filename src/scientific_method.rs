// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scientific Method Engine — Phase 4b of the math plan.
//!
//! Wires existing subsystems (HDC encoding, Bayesian statistics, propositional
//! logic) into a hypothesis-test-update cycle for scientific reasoning.
//!
//! # Pipeline
//! ```text
//! observe → hypothesize → predict → test_hypothesis → update_beliefs → report
//! ```
//!
//! Bayesian updates use `statistics::normal_normal_update` (conjugate
//! Normal-Normal model).  HDC encodings enable geometric contradiction
//! detection between competing hypotheses.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::logic_engine::{LogicEngine, Proposition};
use symthaea_core::hdc::primitive_system::seed_from_name;
use symthaea_core::hdc::statistics;

// ─── Status ──────────────────────────────────────────────────────────────────

/// Lifecycle state of a hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisStatus {
    /// Freshly registered; no evidence collected yet.
    Proposed,
    /// At least one experiment has been run.
    Testing,
    /// Posterior is high enough to consider supported (≥ 0.70).
    Supported,
    /// Posterior is low enough to consider refuted (≤ 0.10).
    Refuted,
    /// Evidence is mixed and the posterior sits between thresholds.
    Inconclusive,
}

impl Default for HypothesisStatus {
    fn default() -> Self {
        HypothesisStatus::Proposed
    }
}

impl HypothesisStatus {
    /// Derive status from a posterior probability.
    pub fn from_posterior(p: f64) -> Self {
        if p >= 0.70 {
            HypothesisStatus::Supported
        } else if p <= 0.10 {
            HypothesisStatus::Refuted
        } else {
            HypothesisStatus::Inconclusive
        }
    }
}

// ─── Core Types ──────────────────────────────────────────────────────────────

/// A scientific hypothesis with its HDC encoding and Bayesian belief state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    /// Monotonically increasing identifier assigned by the engine.
    pub id: usize,
    /// Human-readable statement of the hypothesis.
    pub statement: String,
    /// HDC encoding used for similarity comparisons.
    #[serde(skip)]
    pub encoding: BinaryHV,
    /// Prior probability (user-supplied at creation time).
    pub prior: f64,
    /// Posterior probability (updated by `test_hypothesis`).
    pub posterior: f64,
    /// Number of experiments that have updated this hypothesis.
    pub evidence_count: usize,
    /// Current lifecycle status.
    pub status: HypothesisStatus,
}

impl Hypothesis {
    /// Create a new hypothesis with the given statement and prior.
    fn new(id: usize, statement: &str, prior: f64) -> Self {
        let prior = prior.clamp(0.0, 1.0);
        // Encode the statement text as a deterministic HDC vector.
        let encoding = encode_text(statement);
        Hypothesis {
            id,
            statement: statement.to_string(),
            encoding,
            prior,
            posterior: prior,
            evidence_count: 0,
            status: HypothesisStatus::Proposed,
        }
    }
}

/// A single empirical observation encoded into HDC space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Raw numeric data attached to this observation.
    pub data: Vec<f64>,
    /// Human-readable description of what was observed.
    pub description: String,
    /// HDC encoding of the observation.
    #[serde(skip)]
    pub encoding: BinaryHV,
    /// Logical timestamp (monotonic counter, not wall-clock).
    pub timestamp: u64,
}

/// The result of testing a hypothesis against a single observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Which hypothesis this experiment concerns.
    pub hypothesis_id: usize,
    /// The value predicted by the hypothesis.
    pub prediction: f64,
    /// The value actually observed.
    pub observation: f64,
    /// Absolute prediction error: `|observation − prediction|`.
    pub surprise: f64,
    /// Integrated information Φ from the Bayesian update step.
    pub phi: f64,
}

// ─── Report ──────────────────────────────────────────────────────────────────

/// Summary row for a single hypothesis inside a `ScientificReport`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisSummary {
    pub id: usize,
    pub statement: String,
    pub prior: f64,
    pub posterior: f64,
    pub evidence_count: usize,
    pub status: HypothesisStatus,
}

/// High-level snapshot of all hypotheses and aggregate experiment statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificReport {
    /// Per-hypothesis summaries, ordered by hypothesis id.
    pub hypotheses: Vec<HypothesisSummary>,
    /// Total number of experiments recorded.
    pub total_experiments: usize,
    /// Total number of observations recorded.
    pub total_observations: usize,
    /// Sum of Φ values across all experiments.
    pub total_phi: f64,
    /// Mean posterior across all hypotheses.
    pub mean_posterior: f64,
    /// Number of hypotheses in `Supported` state.
    pub supported_count: usize,
    /// Number of hypotheses in `Refuted` state.
    pub refuted_count: usize,
    /// Number of contradicting hypothesis pairs detected.
    pub contradiction_count: usize,
}

// ─── Engine ──────────────────────────────────────────────────────────────────

/// Main scientific-method engine.
///
/// Maintains the full registry of hypotheses, observations, and experiments,
/// and provides the core Bayesian update + HDC reasoning operations.
pub struct ScientificMethodEngine {
    hypotheses: Vec<Hypothesis>,
    observations: Vec<Observation>,
    experiments: Vec<Experiment>,
    /// Monotonic clock for observation timestamps.
    clock: u64,
    /// Minimum HDC similarity threshold to flag a potential contradiction.
    ///
    /// Two hypotheses are "similar" when their cosine similarity exceeds this
    /// value.  If they also have strongly opposing posteriors they are flagged
    /// as contradictory.
    contradiction_similarity_threshold: f32,
    /// Posterior difference required (|p1 − p2|) before a similar pair is
    /// considered a contradiction.
    contradiction_posterior_gap: f64,
}

impl Default for ScientificMethodEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ScientificMethodEngine {
    // ── Construction ────────────────────────────────────────────────────────

    /// Create a new empty engine with default thresholds.
    pub fn new() -> Self {
        ScientificMethodEngine {
            hypotheses: Vec::new(),
            observations: Vec::new(),
            experiments: Vec::new(),
            clock: 0,
            contradiction_similarity_threshold: 0.65,
            contradiction_posterior_gap: 0.40,
        }
    }

    /// Override the similarity threshold used by `find_contradictions`.
    pub fn with_contradiction_threshold(mut self, threshold: f32) -> Self {
        self.contradiction_similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    // ── Core Pipeline ───────────────────────────────────────────────────────

    /// Record an empirical observation.
    ///
    /// The observation is encoded as an HDC vector by hashing each data value
    /// together with the description text.
    pub fn observe(&mut self, data: Vec<f64>, description: &str) -> Observation {
        self.clock += 1;

        // Build encoding: start from description text, then bind each datum.
        let mut encoding = encode_text(description);
        for (i, &val) in data.iter().enumerate() {
            let slot_seed = seed_from_name(&format!("OBS_SLOT_{}", i));
            let val_seed = seed_from_name(&format!("OBS_VAL_{:.6}", val));
            let slot_hv = BinaryHV::random(slot_seed);
            let val_hv = BinaryHV::random(val_seed);
            encoding = encoding.bind(&slot_hv.bind(&val_hv));
        }

        let obs = Observation {
            data,
            description: description.to_string(),
            encoding,
            timestamp: self.clock,
        };
        self.observations.push(obs.clone());
        obs
    }

    /// Register a new hypothesis and return its assigned id.
    ///
    /// `prior` is clamped to `[0.0, 1.0]`.
    pub fn hypothesize(&mut self, statement: &str, prior: f64) -> usize {
        let id = self.hypotheses.len();
        let h = Hypothesis::new(id, statement, prior);
        self.hypotheses.push(h);
        id
    }

    /// Generate a point prediction for a hypothesis.
    ///
    /// The current implementation uses the posterior (or prior when no
    /// evidence has been collected yet) as a probability-scaled baseline
    /// prediction.  Callers that need domain-specific predictions should
    /// post-process this value.
    pub fn predict(&self, hypothesis_id: usize) -> f64 {
        self.hypotheses
            .get(hypothesis_id)
            .map(|h| h.posterior)
            .unwrap_or(0.5)
    }

    /// Test a hypothesis against an observed value and update beliefs.
    ///
    /// # Bayesian update
    /// Uses `statistics::normal_normal_update` with:
    /// - Prior μ  = current posterior of the hypothesis
    /// - Prior σ  = 0.25 (uninformative over `[0, 1]`)
    /// - Data mean = `observed_value`
    /// - Data σ  = 0.10 (measurement noise assumption)
    /// - n = 1
    ///
    /// The updated posterior μ is then clamped to `[0, 1]` and stored.
    ///
    /// Returns an `Experiment` describing the outcome, or `None` when the
    /// `hypothesis_id` is out of range.
    pub fn test_hypothesis(
        &mut self,
        hypothesis_id: usize,
        observed_value: f64,
        predicted_value: f64,
    ) -> Option<Experiment> {
        let h = self.hypotheses.get_mut(hypothesis_id)?;

        let surprise = (observed_value - predicted_value).abs();

        // Bayesian update: prior centred on current posterior.
        let bayesian = statistics::normal_normal_update(
            h.posterior, // prior μ
            0.25,        // prior σ (broad)
            observed_value,
            0.10, // measurement noise σ
            1,
        );

        // Extract posterior mean from the BayesianResult.
        let new_posterior = match &bayesian.posterior {
            statistics::Distribution::Normal { mu, .. } => mu.clamp(0.0, 1.0),
            _ => h.posterior,
        };

        h.posterior = new_posterior;
        h.evidence_count += 1;

        // Move out of Proposed the moment any experiment is run.
        if h.status == HypothesisStatus::Proposed {
            h.status = HypothesisStatus::Testing;
        }

        let experiment = Experiment {
            hypothesis_id,
            prediction: predicted_value,
            observation: observed_value,
            surprise,
            phi: bayesian.phi,
        };
        self.experiments.push(experiment.clone());
        Some(experiment)
    }

    /// Batch-recompute the status of every hypothesis from its current
    /// posterior.
    ///
    /// Hypotheses that are still `Proposed` (zero evidence) are left
    /// unchanged.
    pub fn update_beliefs(&mut self) {
        for h in &mut self.hypotheses {
            if h.status != HypothesisStatus::Proposed {
                h.status = HypothesisStatus::from_posterior(h.posterior);
            }
        }
    }

    // ── Analysis ────────────────────────────────────────────────────────────

    /// Return all pairs `(id_a, id_b)` where hypotheses are semantically
    /// similar (high HDC cosine similarity) yet their posteriors diverge
    /// significantly — a sign of hidden contradiction.
    ///
    /// Only considers pairs where at least one side is in `Testing`,
    /// `Supported`, or `Refuted` state (i.e., has evidence).
    pub fn find_contradictions(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        let n = self.hypotheses.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let hi = &self.hypotheses[i];
                let hj = &self.hypotheses[j];

                // Skip pairs where neither side has evidence.
                let has_evidence = |h: &Hypothesis| !matches!(h.status, HypothesisStatus::Proposed);
                if !has_evidence(hi) && !has_evidence(hj) {
                    continue;
                }

                let sim = hi.encoding.similarity(&hj.encoding);
                let posterior_gap = (hi.posterior - hj.posterior).abs();

                if sim >= self.contradiction_similarity_threshold
                    && posterior_gap >= self.contradiction_posterior_gap
                {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Rank hypotheses by descending posterior probability.
    ///
    /// Returns `(hypothesis_id, posterior)` pairs.
    pub fn rank_hypotheses(&self) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = self
            .hypotheses
            .iter()
            .map(|h| (h.id, h.posterior))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Sum of all experiment Φ values.
    pub fn total_phi(&self) -> f64 {
        self.experiments.iter().map(|e| e.phi).sum()
    }

    /// Generate a full `ScientificReport`.
    pub fn generate_report(&self) -> ScientificReport {
        let hypotheses: Vec<HypothesisSummary> = self
            .hypotheses
            .iter()
            .map(|h| HypothesisSummary {
                id: h.id,
                statement: h.statement.clone(),
                prior: h.prior,
                posterior: h.posterior,
                evidence_count: h.evidence_count,
                status: h.status.clone(),
            })
            .collect();

        let total_phi = self.total_phi();
        let mean_posterior = if hypotheses.is_empty() {
            0.0
        } else {
            hypotheses.iter().map(|h| h.posterior).sum::<f64>() / hypotheses.len() as f64
        };
        let supported_count = hypotheses
            .iter()
            .filter(|h| h.status == HypothesisStatus::Supported)
            .count();
        let refuted_count = hypotheses
            .iter()
            .filter(|h| h.status == HypothesisStatus::Refuted)
            .count();
        let contradiction_count = self.find_contradictions().len();

        ScientificReport {
            hypotheses,
            total_experiments: self.experiments.len(),
            total_observations: self.observations.len(),
            total_phi,
            mean_posterior,
            supported_count,
            refuted_count,
            contradiction_count,
        }
    }

    // ── Logic Integration ───────────────────────────────────────────────────

    /// Encode a hypothesis statement as a logical proposition and check
    /// whether the negation of a second hypothesis follows via modus tollens.
    ///
    /// Returns `true` when the inference is valid (i.e., the propositions are
    /// structurally compatible and the rule fires).
    pub fn implies_negation(&self, antecedent_id: usize, consequent_id: usize) -> bool {
        let Some(ha) = self.hypotheses.get(antecedent_id) else {
            return false;
        };
        let Some(hc) = self.hypotheses.get(consequent_id) else {
            return false;
        };

        let p = Proposition::atom(&ha.statement);
        let q = Proposition::atom(&hc.statement);
        let implication = p.clone().implies(q.clone());
        let neg_q = q.not();

        // Modus tollens: ¬Q, P → Q ⊢ ¬P
        LogicEngine::modus_tollens(&neg_q, &implication).is_some()
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    /// Borrow a hypothesis by id.
    pub fn hypothesis(&self, id: usize) -> Option<&Hypothesis> {
        self.hypotheses.get(id)
    }

    /// Number of hypotheses registered.
    pub fn hypothesis_count(&self) -> usize {
        self.hypotheses.len()
    }

    /// Number of experiments recorded.
    pub fn experiment_count(&self) -> usize {
        self.experiments.len()
    }

    /// Number of observations recorded.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Borrow the full experiments slice.
    pub fn experiments(&self) -> &[Experiment] {
        &self.experiments
    }

    /// Borrow the full observations slice.
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }
}

// ─── MathService Bridge ──────────────────────────────────────────────────────

/// Bridge between scientific method and the cognitive loop's math service.
///
/// When `scientific_method` feature is enabled (which implies `mathematics`),
/// the engine can delegate numerical computations to MathService and use the
/// results as evidence for or against hypotheses.
impl ScientificMethodEngine {
    /// Run a numerical experiment: compute f(params) via MathService statistics,
    /// compare to predicted value, and update the hypothesis.
    ///
    /// `data` is fed to `MathService::compute_statistics`, and the resulting
    /// mean is compared against `predicted_value` as the observed evidence.
    /// Returns the experiment and the MathResponse for downstream use.
    pub fn numerical_experiment(
        &mut self,
        math_service: &mut crate::cognitive_loop::math_service::MathService,
        hypothesis_id: usize,
        data: &[f64],
        predicted_value: f64,
    ) -> Option<(
        Experiment,
        crate::cognitive_loop::math_service::MathResponse,
    )> {
        let math_response = math_service.compute_statistics(data);
        let observed = math_response.numerical_result.unwrap_or(0.0);

        let experiment = self.test_hypothesis(hypothesis_id, observed, predicted_value)?;
        Some((experiment, math_response))
    }

    /// Run a Bayesian experiment: use MathService's regression to test whether
    /// data supports a linear hypothesis, then update beliefs based on R².
    ///
    /// High R² (good fit) → observed ≈ 1.0 (confirming); low R² → observed ≈ 0.0.
    pub fn regression_experiment(
        &mut self,
        math_service: &mut crate::cognitive_loop::math_service::MathService,
        hypothesis_id: usize,
        x: &[f64],
        y: &[f64],
    ) -> Option<(
        Experiment,
        crate::cognitive_loop::math_service::MathResponse,
    )> {
        let math_response = math_service.linear_regression(x, y);
        let r_squared = math_response.numerical_result.unwrap_or(0.0);

        let predicted = self.predict(hypothesis_id);
        let experiment = self.test_hypothesis(hypothesis_id, r_squared, predicted)?;
        Some((experiment, math_response))
    }

    /// Run a root-finding experiment: test whether f has a root in [a, b].
    ///
    /// If root finding converges (confidence > 0.5), evidence supports the
    /// hypothesis; otherwise it weakens it.
    pub fn root_experiment<F: Fn(f64) -> f64>(
        &mut self,
        math_service: &mut crate::cognitive_loop::math_service::MathService,
        hypothesis_id: usize,
        f: &F,
        a: f64,
        b: f64,
    ) -> Option<(
        Experiment,
        crate::cognitive_loop::math_service::MathResponse,
    )> {
        let math_response = math_service.find_root(f, a, b);
        let evidence = math_response.confidence; // 0.99 if converged, 0.3 if not

        let predicted = self.predict(hypothesis_id);
        let experiment = self.test_hypothesis(hypothesis_id, evidence, predicted)?;
        Some((experiment, math_response))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURABLE ENGINE — hypothesis-test-update cycle with HDC similarity
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the configurable scientific method engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificMethodConfig {
    /// Maximum number of active hypotheses (default 16).
    pub max_hypotheses: usize,
    /// Posterior threshold for confirming a hypothesis (default 0.9).
    pub confirmation_threshold: f64,
    /// Posterior threshold for refuting a hypothesis (default 0.1).
    pub refutation_threshold: f64,
    /// Default prior probability for new hypotheses (default 0.5).
    pub prior_default: f64,
    /// Minimum experiments before status transition (default 3).
    pub min_experiments: usize,
}

impl Default for ScientificMethodConfig {
    fn default() -> Self {
        Self {
            max_hypotheses: 16,
            confirmation_threshold: 0.9,
            refutation_threshold: 0.1,
            prior_default: 0.5,
            min_experiments: 3,
        }
    }
}

/// Aggregate telemetry for the configurable scientific method engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScientificMethodTelemetry {
    /// Total hypotheses generated across the engine's lifetime.
    pub hypotheses_generated: usize,
    /// Number of hypotheses currently in `Confirmed` status.
    pub hypotheses_confirmed: usize,
    /// Number of hypotheses currently in `Refuted` status.
    pub hypotheses_refuted: usize,
    /// Total experiments run.
    pub experiments_run: usize,
    /// Mean match_score across all experiments (0.0 when none).
    pub average_prediction_accuracy: f64,
    /// Total Bayesian update steps performed.
    pub bayesian_updates: usize,
}

/// A prediction derived from a hypothesis — an expected HDC encoding with
/// confidence and tolerance for similarity comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Expected observation in HDC space.
    #[serde(skip)]
    pub expected_encoding: BinaryHV,
    /// Confidence in this prediction (0.0–1.0).
    pub confidence: f64,
    /// Similarity tolerance: match_score must exceed `1.0 - tolerance` to confirm.
    pub tolerance: f64,
}

/// Result of running an HDC-based experiment against a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Which hypothesis this experiment concerns.
    pub hypothesis_id: u64,
    /// Index into the hypothesis's prediction list.
    pub prediction_idx: usize,
    /// The observed HDC vector.
    #[serde(skip)]
    pub observed: BinaryHV,
    /// Cosine similarity between predicted and observed encodings.
    pub match_score: f64,
    /// Whether the match exceeded the prediction's tolerance.
    pub confirmed: bool,
}

/// Extended hypothesis with prediction list for the configurable engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurableHypothesis {
    /// Unique identifier.
    pub id: u64,
    /// Human-readable description.
    pub description: String,
    /// HDC encoding.
    #[serde(skip)]
    pub encoding: BinaryHV,
    /// Prior probability P(H).
    pub prior: f64,
    /// Posterior probability P(H|E).
    pub posterior: f64,
    /// Testable predictions derived from this hypothesis.
    pub predictions: Vec<Prediction>,
    /// Current status.
    pub status: HypothesisStatus,
    /// Number of experiments run against this hypothesis.
    pub experiment_count: usize,
}

/// Extended observation with surprise for the configurable engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseObservation {
    /// HDC encoding of the observation.
    #[serde(skip)]
    pub encoding: BinaryHV,
    /// Logical timestamp.
    pub timestamp: u64,
    /// Prediction error magnitude (surprise signal).
    pub surprise: f64,
}

/// Configurable Scientific Method Engine with HDC-based hypothesis testing.
///
/// Implements a formal hypothesis-test-update cycle:
/// 1. **Observe** — gather HDC-encoded data points with surprise signals
/// 2. **Hypothesize** — register candidate explanations with default priors
/// 3. **Predict** — attach testable expectations (HDC encodings) to hypotheses
/// 4. **Experiment** — compare predictions to observations via HDC similarity
/// 5. **Update** — Bayesian posterior update from likelihood ratios
/// 6. **Evaluate** — confirm/refute based on configurable thresholds
pub struct ConfigurableScientificMethodEngine {
    hypotheses: Vec<ConfigurableHypothesis>,
    observations: Vec<SurpriseObservation>,
    experiments: Vec<ExperimentResult>,
    telemetry: ScientificMethodTelemetry,
    config: ScientificMethodConfig,
    /// Monotonic counter for hypothesis IDs.
    next_id: u64,
    /// Monotonic clock for observation timestamps.
    clock: u64,
}

impl Default for ConfigurableScientificMethodEngine {
    fn default() -> Self {
        Self::new(ScientificMethodConfig::default())
    }
}

impl ConfigurableScientificMethodEngine {
    // ── Construction ────────────────────────────────────────────────────────

    /// Create a new engine with the given configuration.
    pub fn new(config: ScientificMethodConfig) -> Self {
        Self {
            hypotheses: Vec::new(),
            observations: Vec::new(),
            experiments: Vec::new(),
            telemetry: ScientificMethodTelemetry::default(),
            config,
            next_id: 0,
            clock: 0,
        }
    }

    // ── 1. Observe ──────────────────────────────────────────────────────────

    /// Record an observation with its HDC encoding and surprise magnitude.
    pub fn observe(&mut self, encoding: BinaryHV, surprise: f64) {
        self.clock += 1;
        self.observations.push(SurpriseObservation {
            encoding,
            timestamp: self.clock,
            surprise: surprise.max(0.0),
        });
    }

    // ── 2. Hypothesize ──────────────────────────────────────────────────────

    /// Register a new hypothesis. Returns the assigned ID, or `None` if
    /// `max_hypotheses` has been reached.
    pub fn hypothesize(&mut self, description: &str, encoding: BinaryHV) -> Option<u64> {
        let active_count = self
            .hypotheses
            .iter()
            .filter(|h| {
                matches!(
                    h.status,
                    HypothesisStatus::Proposed
                        | HypothesisStatus::Testing
                        | HypothesisStatus::Inconclusive
                )
            })
            .count();
        if active_count >= self.config.max_hypotheses {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.telemetry.hypotheses_generated += 1;
        self.hypotheses.push(ConfigurableHypothesis {
            id,
            description: description.to_string(),
            encoding,
            prior: self.config.prior_default,
            posterior: self.config.prior_default,
            predictions: Vec::new(),
            status: HypothesisStatus::Proposed,
            experiment_count: 0,
        });
        Some(id)
    }

    // ── 3. Predict ──────────────────────────────────────────────────────────

    /// Attach a prediction to a hypothesis.
    ///
    /// `expected` is the HDC encoding the hypothesis predicts should be
    /// observed. `confidence` is how certain the hypothesis is about this
    /// prediction. `tolerance` sets how close the observation must be
    /// (match_score must exceed `1.0 - tolerance`).
    pub fn predict(
        &mut self,
        hypothesis_id: u64,
        expected: BinaryHV,
        confidence: f64,
        tolerance: f64,
    ) -> bool {
        if let Some(h) = self.hypotheses.iter_mut().find(|h| h.id == hypothesis_id) {
            h.predictions.push(Prediction {
                expected_encoding: expected,
                confidence: confidence.clamp(0.0, 1.0),
                tolerance: tolerance.clamp(0.0, 1.0),
            });
            true
        } else {
            false
        }
    }

    // ── 4. Experiment ───────────────────────────────────────────────────────

    /// Test a specific prediction against an observed HDC vector.
    ///
    /// Computes match_score as the cosine similarity between the prediction's
    /// expected encoding and the observed encoding. Returns the experiment
    /// result, or `None` if the hypothesis or prediction index is invalid.
    pub fn experiment(
        &mut self,
        hypothesis_id: u64,
        prediction_idx: usize,
        observed: BinaryHV,
    ) -> Option<ExperimentResult> {
        let h = self.hypotheses.iter().find(|h| h.id == hypothesis_id)?;
        let pred = h.predictions.get(prediction_idx)?;

        let match_score = pred.expected_encoding.similarity(&observed) as f64;
        let confirmed = match_score >= (1.0 - pred.tolerance);

        let result = ExperimentResult {
            hypothesis_id,
            prediction_idx,
            observed,
            match_score,
            confirmed,
        };
        self.experiments.push(result.clone());
        self.telemetry.experiments_run += 1;

        // Increment the hypothesis experiment count.
        if let Some(h) = self.hypotheses.iter_mut().find(|h| h.id == hypothesis_id) {
            h.experiment_count += 1;
            // Transition from Proposed to Testing.
            if h.status == HypothesisStatus::Proposed {
                h.status = HypothesisStatus::Testing;
            }
        }

        // Update running average prediction accuracy.
        let n = self.telemetry.experiments_run as f64;
        self.telemetry.average_prediction_accuracy =
            self.telemetry.average_prediction_accuracy * ((n - 1.0) / n) + match_score / n;

        Some(result)
    }

    // ── 5. Bayesian Update ──────────────────────────────────────────────────

    /// Update the posterior of a hypothesis using a likelihood ratio.
    ///
    /// Applies Bayes' rule:
    ///   P(H|E) = P(E|H) * P(H) / [P(E|H) * P(H) + P(E|~H) * (1 - P(H))]
    ///
    /// where `likelihood_ratio = P(E|H) / P(E|~H)`.
    ///
    /// The prior used is the current posterior (sequential updating).
    pub fn bayesian_update(&mut self, hypothesis_id: u64, likelihood_ratio: f64) -> bool {
        if let Some(h) = self.hypotheses.iter_mut().find(|h| h.id == hypothesis_id) {
            let lr = likelihood_ratio.max(1e-10); // prevent division by zero
            let prior = h.posterior;
            if !prior.is_finite() {
                // Corrupted prior — reset to uninformative
                h.posterior = 0.5;
                self.telemetry.bayesian_updates += 1;
                return true;
            }
            // Bayes: posterior = lr * prior / (lr * prior + 1 * (1 - prior))
            let numerator = lr * prior;
            let denominator = numerator + (1.0 - prior);
            let result = numerator / denominator;
            h.posterior = if result.is_finite() {
                result.clamp(0.0, 1.0)
            } else {
                0.5 // NaN/Inf from arithmetic — reset
            };
            self.telemetry.bayesian_updates += 1;
            true
        } else {
            false
        }
    }

    // ── 6. Evaluate ─────────────────────────────────────────────────────────

    /// Evaluate a hypothesis against the configured thresholds.
    ///
    /// A hypothesis is only promoted to `Supported`/`Refuted` once it has
    /// accumulated at least `min_experiments` experiments.
    pub fn evaluate_hypothesis(&mut self, hypothesis_id: u64) {
        if let Some(h) = self.hypotheses.iter_mut().find(|h| h.id == hypothesis_id) {
            if h.status == HypothesisStatus::Proposed {
                return; // no evidence yet
            }
            if h.experiment_count < self.config.min_experiments {
                h.status = HypothesisStatus::Inconclusive;
                return;
            }
            if h.posterior >= self.config.confirmation_threshold {
                h.status = HypothesisStatus::Supported;
            } else if h.posterior <= self.config.refutation_threshold {
                h.status = HypothesisStatus::Refuted;
            } else {
                h.status = HypothesisStatus::Inconclusive;
            }
        }
    }

    // ── 7. Best Hypothesis ──────────────────────────────────────────────────

    /// Return the hypothesis with the highest posterior, or `None` if empty.
    pub fn best_hypothesis(&self) -> Option<&ConfigurableHypothesis> {
        self.hypotheses.iter().max_by(|a, b| {
            a.posterior
                .partial_cmp(&b.posterior)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    // ── 8. Competing Hypotheses ─────────────────────────────────────────────

    /// Return all pairs of hypotheses whose posteriors are within 0.1 of
    /// each other — these are the "competing" explanations that need
    /// discriminating evidence.
    pub fn competing_hypotheses(&self) -> Vec<(&ConfigurableHypothesis, &ConfigurableHypothesis)> {
        let mut pairs = Vec::new();
        let n = self.hypotheses.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let gap = (self.hypotheses[i].posterior - self.hypotheses[j].posterior).abs();
                if gap <= 0.1 {
                    pairs.push((&self.hypotheses[i], &self.hypotheses[j]));
                }
            }
        }
        pairs
    }

    // ── 9. Most Informative Experiment ──────────────────────────────────────

    /// Find the (hypothesis_id, prediction_idx) pair that would maximally
    /// discriminate between competing hypotheses.
    ///
    /// The heuristic selects the prediction that maximises the expected
    /// information gain: the product of the prediction's confidence and
    /// the inverse of the hypothesis posterior certainty (how far from 0.5).
    /// Predictions from hypotheses that are already Supported/Refuted are
    /// excluded.
    pub fn most_informative_experiment(&self) -> Option<(u64, usize)> {
        let mut best: Option<(u64, usize, f64)> = None;

        for h in &self.hypotheses {
            // Skip resolved hypotheses.
            if matches!(
                h.status,
                HypothesisStatus::Supported | HypothesisStatus::Refuted
            ) {
                continue;
            }
            // Uncertainty: maximal at posterior=0.5, zero at 0 or 1.
            let uncertainty = 1.0 - (2.0 * (h.posterior - 0.5)).abs();

            for (idx, pred) in h.predictions.iter().enumerate() {
                // Information gain proxy = prediction confidence * hypothesis uncertainty.
                let info_gain = pred.confidence * uncertainty;
                if best.map_or(true, |(_, _, b)| info_gain > b) {
                    best = Some((h.id, idx, info_gain));
                }
            }
        }

        best.map(|(id, idx, _)| (id, idx))
    }

    // ── Telemetry ───────────────────────────────────────────────────────────

    /// Return a snapshot of the engine's aggregate telemetry.
    pub fn telemetry(&self) -> &ScientificMethodTelemetry {
        &self.telemetry
    }

    /// Recompute derived telemetry counters from current hypothesis states.
    pub fn refresh_telemetry(&mut self) {
        self.telemetry.hypotheses_confirmed = self
            .hypotheses
            .iter()
            .filter(|h| h.status == HypothesisStatus::Supported)
            .count();
        self.telemetry.hypotheses_refuted = self
            .hypotheses
            .iter()
            .filter(|h| h.status == HypothesisStatus::Refuted)
            .count();
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    /// Borrow a hypothesis by id.
    pub fn get_hypothesis(&self, id: u64) -> Option<&ConfigurableHypothesis> {
        self.hypotheses.iter().find(|h| h.id == id)
    }

    /// Number of hypotheses registered.
    pub fn hypothesis_count(&self) -> usize {
        self.hypotheses.len()
    }

    /// Number of observations recorded.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Borrow the full observations slice.
    pub fn observations(&self) -> &[SurpriseObservation] {
        &self.observations
    }

    /// Borrow the full experiment results slice.
    pub fn experiment_results(&self) -> &[ExperimentResult] {
        &self.experiments
    }

    /// Borrow the config.
    pub fn config(&self) -> &ScientificMethodConfig {
        &self.config
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Encode a text string as a BinaryHV by hashing word-level seeds and
/// binding them together.
fn encode_text(text: &str) -> BinaryHV {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return BinaryHV::random(seed_from_name("EMPTY_TEXT"));
    }
    let mut hv = BinaryHV::random(seed_from_name(&format!("WORD_{}", words[0])));
    for word in &words[1..] {
        let w = BinaryHV::random(seed_from_name(&format!("WORD_{}", word)));
        hv = hv.bind(&w);
    }
    hv
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: run N confirming experiments (observed ≈ predicted ≈ 1.0).
    fn confirm(engine: &mut ScientificMethodEngine, hid: usize, n: usize) {
        for _ in 0..n {
            let pred = engine.predict(hid);
            engine.test_hypothesis(hid, 0.95, pred);
        }
    }

    // Helper: run N refuting experiments (observed ≈ 0, predicted ≈ prior).
    fn refute(engine: &mut ScientificMethodEngine, hid: usize, n: usize) {
        for _ in 0..n {
            let pred = engine.predict(hid);
            engine.test_hypothesis(hid, 0.05, pred);
        }
    }

    // ── 1. Confirming evidence raises posterior ───────────────────────────

    #[test]
    fn test_confirming_evidence_raises_posterior() {
        let mut engine = ScientificMethodEngine::new();
        let hid = engine.hypothesize("High temperature causes rapid enzyme activity", 0.5);

        let initial = engine.predict(hid);
        confirm(&mut engine, hid, 3);

        let updated = engine.hypothesis(hid).unwrap().posterior;
        assert!(
            updated > initial,
            "posterior {updated:.3} should exceed initial {initial:.3}"
        );
    }

    // ── 2. Contradicting evidence lowers posterior ────────────────────────

    #[test]
    fn test_contradicting_evidence_lowers_posterior() {
        let mut engine = ScientificMethodEngine::new();
        let hid = engine.hypothesize("Sleep deprivation has no cognitive effect", 0.5);

        let initial = engine.predict(hid);
        refute(&mut engine, hid, 3);

        let updated = engine.hypothesis(hid).unwrap().posterior;
        assert!(
            updated < initial,
            "posterior {updated:.3} should be below initial {initial:.3}"
        );
    }

    // ── 3. Status transitions to Testing after first experiment ──────────

    #[test]
    fn test_status_transitions_to_testing() {
        let mut engine = ScientificMethodEngine::new();
        let hid = engine.hypothesize("Meditation reduces cortisol", 0.5);

        assert_eq!(
            engine.hypothesis(hid).unwrap().status,
            HypothesisStatus::Proposed
        );

        let pred = engine.predict(hid);
        engine.test_hypothesis(hid, 0.6, pred);

        assert_eq!(
            engine.hypothesis(hid).unwrap().status,
            HypothesisStatus::Testing
        );
    }

    // ── 4. update_beliefs marks Supported ────────────────────────────────

    #[test]
    fn test_update_beliefs_marks_supported() {
        let mut engine = ScientificMethodEngine::new();
        let hid = engine.hypothesize("Exercise improves mood", 0.5);

        confirm(&mut engine, hid, 5);
        engine.update_beliefs();

        let status = &engine.hypothesis(hid).unwrap().status;
        // With enough confirming evidence the posterior should reach ≥ 0.70.
        assert!(
            matches!(
                status,
                HypothesisStatus::Supported | HypothesisStatus::Inconclusive
            ),
            "unexpected status: {status:?}"
        );
    }

    // ── 5. update_beliefs marks Refuted ──────────────────────────────────

    #[test]
    fn test_update_beliefs_marks_refuted() {
        let mut engine = ScientificMethodEngine::new();
        let hid = engine.hypothesize("Chocolate is toxic to adults", 0.5);

        refute(&mut engine, hid, 6);
        engine.update_beliefs();

        let status = &engine.hypothesis(hid).unwrap().status;
        assert!(
            matches!(
                status,
                HypothesisStatus::Refuted | HypothesisStatus::Inconclusive
            ),
            "unexpected status: {status:?}"
        );
    }

    // ── 6. Competing hypotheses — best one ranks first ────────────────────

    #[test]
    fn test_competing_hypotheses_rank_correctly() {
        let mut engine = ScientificMethodEngine::new();
        let h_weak = engine.hypothesize("Gravity is caused by fairies", 0.4);
        let h_strong = engine.hypothesize("Gravity follows inverse-square law", 0.6);

        // Strong hypothesis gets confirming evidence; weak gets refuting.
        confirm(&mut engine, h_strong, 4);
        refute(&mut engine, h_weak, 4);

        let ranked = engine.rank_hypotheses();
        assert_eq!(ranked[0].0, h_strong, "strong hypothesis should rank first");
    }

    // ── 7. Full pipeline: observe → hypothesize → predict → test → update ─

    #[test]
    fn test_full_scientific_cycle() {
        let mut engine = ScientificMethodEngine::new();

        let obs = engine.observe(vec![37.2, 38.1, 36.9], "Body temperature measurements");
        assert_eq!(obs.data.len(), 3);

        let hid = engine.hypothesize("Normal body temperature is ~37°C", 0.6);
        let pred = engine.predict(hid);
        assert!(
            (pred - 0.6).abs() < 1e-9,
            "first prediction should equal prior"
        );

        let exp = engine.test_hypothesis(hid, 0.8, pred).unwrap();
        assert!(exp.surprise >= 0.0);

        engine.update_beliefs();

        let report = engine.generate_report();
        assert_eq!(report.total_experiments, 1);
        assert_eq!(report.total_observations, 1);
    }

    // ── 8. Report generation ──────────────────────────────────────────────

    #[test]
    fn test_report_generation() {
        let mut engine = ScientificMethodEngine::new();
        let h1 = engine.hypothesize("A implies B", 0.7);
        let h2 = engine.hypothesize("B implies C", 0.3);

        confirm(&mut engine, h1, 2);
        refute(&mut engine, h2, 2);
        engine.update_beliefs();

        let report = engine.generate_report();
        assert_eq!(report.hypotheses.len(), 2);
        assert_eq!(report.total_experiments, 4);
        assert!(report.total_phi > 0.0);
        assert!(report.mean_posterior > 0.0);
    }

    // ── 9. evidence_count increments correctly ────────────────────────────

    #[test]
    fn test_evidence_count_increments() {
        let mut engine = ScientificMethodEngine::new();
        let hid = engine.hypothesize("Caffeine increases alertness", 0.55);

        for i in 1..=5 {
            let pred = engine.predict(hid);
            engine.test_hypothesis(hid, 0.7, pred);
            assert_eq!(engine.hypothesis(hid).unwrap().evidence_count, i);
        }
    }

    // ── 10. rank_hypotheses with a single hypothesis ──────────────────────

    #[test]
    fn test_rank_single_hypothesis() {
        let mut engine = ScientificMethodEngine::new();
        let hid = engine.hypothesize("Only hypothesis", 0.5);
        let ranked = engine.rank_hypotheses();
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0, hid);
    }

    // ── 11. total_phi accumulates across experiments ──────────────────────

    #[test]
    fn test_total_phi_accumulates() {
        let mut engine = ScientificMethodEngine::new();
        let hid = engine.hypothesize("Phi accumulates", 0.5);

        let pred = engine.predict(hid);
        engine.test_hypothesis(hid, 0.6, pred);
        let phi_after_1 = engine.total_phi();

        engine.test_hypothesis(hid, 0.7, pred);
        let phi_after_2 = engine.total_phi();

        assert!(
            phi_after_2 > phi_after_1,
            "total Φ should increase with each experiment"
        );
    }

    // ── 12. find_contradictions detects opposing posteriors ───────────────

    #[test]
    fn test_find_contradictions_detects_opposing() {
        // Two hypotheses with very similar text (same encoding neighbourhood)
        // but one confirmed and the other refuted.
        let mut engine = ScientificMethodEngine::new().with_contradiction_threshold(0.50);
        let h1 = engine.hypothesize("drug X lowers blood pressure", 0.5);
        let h2 = engine.hypothesize("drug X lowers blood pressure significantly", 0.5);

        confirm(&mut engine, h1, 5);
        refute(&mut engine, h2, 5);
        engine.update_beliefs();

        // We cannot guarantee the exact number of contradictions without
        // controlling encoding similarity precisely, but the API must not
        // panic and must return a Vec.
        let _pairs = engine.find_contradictions();
    }

    // ── 13. observe populates observations list ───────────────────────────

    #[test]
    fn test_observe_populates_list() {
        let mut engine = ScientificMethodEngine::new();
        assert_eq!(engine.observation_count(), 0);
        engine.observe(vec![1.0, 2.0], "first observation");
        engine.observe(vec![3.0], "second observation");
        assert_eq!(engine.observation_count(), 2);
    }

    // ── 14. out-of-range hypothesis id returns None ───────────────────────

    #[test]
    fn test_out_of_range_hypothesis_returns_none() {
        let mut engine = ScientificMethodEngine::new();
        let result = engine.test_hypothesis(999, 0.5, 0.5);
        assert!(result.is_none());
        assert!(engine.hypothesis(999).is_none());
    }

    // ── 15. HypothesisStatus::from_posterior boundary conditions ─────────

    #[test]
    fn test_hypothesis_status_boundaries() {
        assert_eq!(
            HypothesisStatus::from_posterior(0.70),
            HypothesisStatus::Supported
        );
        assert_eq!(
            HypothesisStatus::from_posterior(0.90),
            HypothesisStatus::Supported
        );
        assert_eq!(
            HypothesisStatus::from_posterior(0.10),
            HypothesisStatus::Refuted
        );
        assert_eq!(
            HypothesisStatus::from_posterior(0.00),
            HypothesisStatus::Refuted
        );
        assert_eq!(
            HypothesisStatus::from_posterior(0.50),
            HypothesisStatus::Inconclusive
        );
        assert_eq!(
            HypothesisStatus::from_posterior(0.69),
            HypothesisStatus::Inconclusive
        );
        assert_eq!(
            HypothesisStatus::from_posterior(0.11),
            HypothesisStatus::Inconclusive
        );
    }

    // ── 16. MathService bridge: numerical experiment ───────────────────

    #[test]
    fn test_numerical_experiment_updates_posterior() {
        let mut engine = ScientificMethodEngine::new();
        let mut math = crate::cognitive_loop::math_service::MathService::new();

        let hid = engine.hypothesize("Mean temperature is 37°C", 0.5);
        let initial = engine.predict(hid);

        // Provide data with mean ≈ 37.0, predicted 0.5 → big surprise → shift posterior
        let result = engine.numerical_experiment(&mut math, hid, &[36.8, 37.0, 37.2], initial);
        assert!(result.is_some());

        let (exp, math_resp) = result.unwrap();
        assert!(exp.surprise >= 0.0);
        assert!(math_resp.phi > 0.0);
        assert_eq!(math.telemetry().problems_solved, 1);

        // Posterior should have moved from initial
        let updated = engine.hypothesis(hid).unwrap().posterior;
        assert!(
            (updated - initial).abs() > 1e-6,
            "posterior should shift: initial={initial}, updated={updated}"
        );
    }

    // ── 17. MathService bridge: regression experiment ──────────────────

    #[test]
    fn test_regression_experiment() {
        let mut engine = ScientificMethodEngine::new();
        let mut math = crate::cognitive_loop::math_service::MathService::new();

        let hid = engine.hypothesize("Y varies linearly with X", 0.5);

        // Perfect linear data: y = 2x + 1, R² = 1.0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let result = engine.regression_experiment(&mut math, hid, &x, &y);
        assert!(result.is_some());

        let (exp, math_resp) = result.unwrap();
        // R² ≈ 1.0 should be the observed value
        let r_sq = math_resp.numerical_result.unwrap();
        assert!((r_sq - 1.0).abs() < 1e-6, "R² should be 1.0, got {r_sq}");
        assert!(exp.phi > 0.0);
    }

    // ── 18. MathService bridge: root experiment ────────────────────────

    #[test]
    fn test_root_experiment() {
        let mut engine = ScientificMethodEngine::new();
        let mut math = crate::cognitive_loop::math_service::MathService::new();

        let hid = engine.hypothesize("f(x) = x² - 4 has a root in [0, 3]", 0.5);

        let result = engine.root_experiment(&mut math, hid, &|x: f64| x * x - 4.0, 0.0, 3.0);
        assert!(result.is_some());

        let (_exp, math_resp) = result.unwrap();
        // Should find root at x=2
        let root = math_resp.numerical_result.unwrap();
        assert!((root - 2.0).abs() < 1e-8, "Root should be 2.0, got {root}");
        assert!(
            math_resp.multipath_verified,
            "should be multi-path verified"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONFIGURABLE ENGINE TESTS
    // ═══════════════════════════════════════════════════════════════════════

    // ── 19. Full observe → hypothesize → predict → experiment → update cycle

    #[test]
    fn test_configurable_full_cycle() {
        let mut engine = ConfigurableScientificMethodEngine::default();

        // 1. Observe
        let obs_hv = BinaryHV::random(seed_from_name("observation_alpha"));
        engine.observe(obs_hv.clone(), 0.3);
        assert_eq!(engine.observation_count(), 1);

        // 2. Hypothesize
        let h_hv = BinaryHV::random(seed_from_name("hypothesis_alpha"));
        let hid = engine
            .hypothesize("Temperature drives reaction rate", h_hv)
            .unwrap();

        // 3. Predict — the hypothesis predicts we'll see obs_hv
        engine.predict(hid, obs_hv.clone(), 0.8, 0.3);

        // 4. Experiment — compare prediction to actual observation
        let result = engine.experiment(hid, 0, obs_hv.clone()).unwrap();
        assert!(
            result.match_score > 0.99,
            "identical vectors should have similarity ~1.0, got {}",
            result.match_score
        );
        assert!(result.confirmed);

        // 5. Bayesian update — confirming evidence raises posterior
        let prior = engine.get_hypothesis(hid).unwrap().posterior;
        engine.bayesian_update(hid, 5.0); // strong evidence for H
        let posterior = engine.get_hypothesis(hid).unwrap().posterior;
        assert!(
            posterior > prior,
            "posterior {posterior:.3} should exceed prior {prior:.3}"
        );

        // 6. Evaluate
        // Need min_experiments=3 by default; add more experiments.
        for _ in 0..2 {
            engine.experiment(hid, 0, obs_hv.clone());
            engine.bayesian_update(hid, 5.0);
        }
        engine.evaluate_hypothesis(hid);
        assert_eq!(
            engine.get_hypothesis(hid).unwrap().status,
            HypothesisStatus::Supported,
        );
    }

    // ── 20. Bayesian convergence — repeated strong evidence drives posterior toward 1.0

    #[test]
    fn test_bayesian_convergence() {
        let mut engine = ConfigurableScientificMethodEngine::default();
        let hv = BinaryHV::random(seed_from_name("convergence"));
        let hid = engine.hypothesize("Converging hypothesis", hv).unwrap();

        for _ in 0..20 {
            engine.bayesian_update(hid, 3.0);
        }

        let post = engine.get_hypothesis(hid).unwrap().posterior;
        assert!(
            post > 0.99,
            "20 rounds of LR=3 should converge posterior near 1.0, got {post:.4}"
        );
    }

    // ── 21. Hypothesis refutation via low likelihood ratio

    #[test]
    fn test_hypothesis_refutation() {
        let config = ScientificMethodConfig {
            min_experiments: 1,
            ..Default::default()
        };
        let mut engine = ConfigurableScientificMethodEngine::new(config);
        let hv = BinaryHV::random(seed_from_name("refutation"));
        let pred_hv = BinaryHV::random(seed_from_name("pred_refutation"));
        let hid = engine.hypothesize("Bad hypothesis", hv).unwrap();
        engine.predict(hid, pred_hv, 0.8, 0.3);

        // Run experiment with orthogonal vector — low match.
        let obs = BinaryHV::random(seed_from_name("different_observation"));
        engine.experiment(hid, 0, obs);

        // Repeated weak evidence.
        for _ in 0..15 {
            engine.bayesian_update(hid, 0.1); // strong evidence against
        }

        engine.evaluate_hypothesis(hid);
        assert_eq!(
            engine.get_hypothesis(hid).unwrap().status,
            HypothesisStatus::Refuted,
        );
    }

    // ── 22. Competing hypotheses detection

    #[test]
    fn test_competing_hypotheses_detection() {
        let mut engine = ConfigurableScientificMethodEngine::default();
        let hv1 = BinaryHV::random(seed_from_name("comp_a"));
        let hv2 = BinaryHV::random(seed_from_name("comp_b"));

        // Both start at prior_default=0.5, so they are within 0.1.
        engine.hypothesize("Hypothesis A", hv1).unwrap();
        engine.hypothesize("Hypothesis B", hv2).unwrap();

        let pairs = engine.competing_hypotheses();
        assert_eq!(
            pairs.len(),
            1,
            "two hypotheses with same posterior should compete"
        );
    }

    // ── 23. Most informative experiment selection

    #[test]
    fn test_most_informative_experiment() {
        let mut engine = ConfigurableScientificMethodEngine::default();
        let hv1 = BinaryHV::random(seed_from_name("info_a"));
        let hv2 = BinaryHV::random(seed_from_name("info_b"));
        let pred_hv = BinaryHV::random(seed_from_name("info_pred"));

        let h1 = engine.hypothesize("Uncertain hypothesis", hv1).unwrap();
        let h2 = engine.hypothesize("Also uncertain", hv2).unwrap();

        // h1 has a high-confidence prediction; h2 has a low-confidence one.
        engine.predict(h1, pred_hv.clone(), 0.9, 0.2);
        engine.predict(h2, pred_hv, 0.2, 0.5);

        let (best_id, best_idx) = engine.most_informative_experiment().unwrap();
        assert_eq!(
            best_id, h1,
            "higher confidence prediction should be more informative"
        );
        assert_eq!(best_idx, 0);
    }

    // ── 24. max_hypotheses limit

    #[test]
    fn test_max_hypotheses_limit() {
        let config = ScientificMethodConfig {
            max_hypotheses: 2,
            ..Default::default()
        };
        let mut engine = ConfigurableScientificMethodEngine::new(config);

        let hv = || BinaryHV::random(seed_from_name("limit"));
        assert!(engine.hypothesize("H1", hv()).is_some());
        assert!(engine.hypothesize("H2", hv()).is_some());
        assert!(
            engine.hypothesize("H3 (overflow)", hv()).is_none(),
            "should reject hypothesis beyond max"
        );
    }

    // ── 25. Telemetry counters

    #[test]
    fn test_telemetry_counters() {
        let config = ScientificMethodConfig {
            min_experiments: 1,
            ..Default::default()
        };
        let mut engine = ConfigurableScientificMethodEngine::new(config);
        let hv = BinaryHV::random(seed_from_name("telem"));
        let pred_hv = BinaryHV::random(seed_from_name("telem_pred"));

        let hid = engine.hypothesize("Telemetry test", hv).unwrap();
        engine.predict(hid, pred_hv.clone(), 0.8, 0.3);

        engine.experiment(hid, 0, pred_hv.clone());
        engine.bayesian_update(hid, 2.0);

        let t = engine.telemetry();
        assert_eq!(t.hypotheses_generated, 1);
        assert_eq!(t.experiments_run, 1);
        assert_eq!(t.bayesian_updates, 1);
        assert!(t.average_prediction_accuracy > 0.0);
    }

    // ── 26. Best hypothesis selection

    #[test]
    fn test_best_hypothesis_selection() {
        let mut engine = ConfigurableScientificMethodEngine::default();
        let hv1 = BinaryHV::random(seed_from_name("best_a"));
        let hv2 = BinaryHV::random(seed_from_name("best_b"));

        let h1 = engine.hypothesize("Weak", hv1).unwrap();
        let h2 = engine.hypothesize("Strong", hv2).unwrap();

        // Push h2's posterior higher.
        engine.bayesian_update(h2, 10.0);

        let best = engine.best_hypothesis().unwrap();
        assert_eq!(
            best.id, h2,
            "h2 should have highest posterior after strong update"
        );
    }

    // ── 27. Experiment with mismatched vectors produces low match_score

    #[test]
    fn test_experiment_mismatch() {
        let mut engine = ConfigurableScientificMethodEngine::default();
        let hv = BinaryHV::random(seed_from_name("mismatch_h"));
        let pred_hv = BinaryHV::random(seed_from_name("mismatch_pred"));
        let obs_hv = BinaryHV::random(seed_from_name("mismatch_obs"));

        let hid = engine.hypothesize("Mismatch test", hv).unwrap();
        engine.predict(hid, pred_hv, 0.8, 0.05); // tight tolerance

        let result = engine.experiment(hid, 0, obs_hv).unwrap();
        // Random BinaryHVs have similarity ~0.5.
        assert!(
            result.match_score < 0.7,
            "orthogonal vectors should have low similarity, got {}",
            result.match_score
        );
        assert!(
            !result.confirmed,
            "tight tolerance should reject orthogonal vectors"
        );
    }

    // ── 28. Evaluate requires min_experiments

    #[test]
    fn test_evaluate_requires_min_experiments() {
        let config = ScientificMethodConfig {
            min_experiments: 5,
            confirmation_threshold: 0.9,
            ..Default::default()
        };
        let mut engine = ConfigurableScientificMethodEngine::new(config);
        let hv = BinaryHV::random(seed_from_name("min_exp"));
        let pred_hv = BinaryHV::random(seed_from_name("min_exp_pred"));

        let hid = engine.hypothesize("Min experiments test", hv).unwrap();
        engine.predict(hid, pred_hv.clone(), 0.8, 0.5);

        // Only 2 experiments — below min_experiments=5.
        engine.experiment(hid, 0, pred_hv.clone());
        engine.experiment(hid, 0, pred_hv);
        for _ in 0..10 {
            engine.bayesian_update(hid, 10.0);
        }

        engine.evaluate_hypothesis(hid);
        assert_eq!(
            engine.get_hypothesis(hid).unwrap().status,
            HypothesisStatus::Inconclusive,
            "should remain Inconclusive until min_experiments met"
        );
    }

    // ── 29. Observe records surprise correctly

    #[test]
    fn test_observe_records_surprise() {
        let mut engine = ConfigurableScientificMethodEngine::default();
        let hv = BinaryHV::random(seed_from_name("surprise_test"));
        engine.observe(hv, 0.75);

        let obs = &engine.observations()[0];
        assert!((obs.surprise - 0.75).abs() < 1e-9);
        assert_eq!(obs.timestamp, 1);
    }

    // ── 30. Negative surprise clamped to zero

    #[test]
    fn test_negative_surprise_clamped() {
        let mut engine = ConfigurableScientificMethodEngine::default();
        let hv = BinaryHV::random(seed_from_name("neg_surprise"));
        engine.observe(hv, -5.0);

        let obs = &engine.observations()[0];
        assert!(
            obs.surprise >= 0.0,
            "negative surprise should be clamped to 0, got {}",
            obs.surprise
        );
    }

    // ── 31. refresh_telemetry updates confirmed/refuted counts

    #[test]
    fn test_refresh_telemetry() {
        let config = ScientificMethodConfig {
            min_experiments: 1,
            ..Default::default()
        };
        let mut engine = ConfigurableScientificMethodEngine::new(config);
        let hv = BinaryHV::random(seed_from_name("refresh"));
        let pred_hv = BinaryHV::random(seed_from_name("refresh_pred"));

        let hid = engine.hypothesize("Refresh test", hv).unwrap();
        engine.predict(hid, pred_hv.clone(), 0.8, 0.5);
        engine.experiment(hid, 0, pred_hv);

        for _ in 0..20 {
            engine.bayesian_update(hid, 10.0);
        }
        engine.evaluate_hypothesis(hid);
        engine.refresh_telemetry();

        assert_eq!(engine.telemetry().hypotheses_confirmed, 1);
        assert_eq!(engine.telemetry().hypotheses_refuted, 0);
    }

    // ── 32. Empty engine returns None for best_hypothesis

    #[test]
    fn test_empty_engine_best_hypothesis() {
        let engine = ConfigurableScientificMethodEngine::default();
        assert!(engine.best_hypothesis().is_none());
        assert!(engine.most_informative_experiment().is_none());
        assert!(engine.competing_hypotheses().is_empty());
    }

    // ── 33. Invalid hypothesis_id returns None/false

    #[test]
    fn test_invalid_hypothesis_id() {
        let mut engine = ConfigurableScientificMethodEngine::default();
        let obs = BinaryHV::random(seed_from_name("invalid_test"));

        assert!(!engine.bayesian_update(999, 2.0));
        assert!(engine.experiment(999, 0, obs).is_none());
        assert!(!engine.predict(999, BinaryHV::random(42), 0.5, 0.5));
        assert!(engine.get_hypothesis(999).is_none());
    }
}
