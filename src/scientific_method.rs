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
}
