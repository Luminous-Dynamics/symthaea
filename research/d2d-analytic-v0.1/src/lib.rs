//! Dream-to-Discovery (D2D) analytic protocol fixture.
//!
//! This crate demonstrates a deliberately small experiment lifecycle:
//! competing hypotheses -> budgeted experiment selection -> prediction seal
//! -> evaluator-only execution -> comparison with sealed predictions.
//!
//! The fixture uses a one-dimensional damped-velocity system with known truth.
//! It is not a Symtropy adapter, learned world model, or cryptographic evidence
//! system. The FNV-style fingerprints below are deterministic fixture IDs only.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/// Errors returned by the bounded D2D analytic fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum D2dError {
    InvalidModel(&'static str),
    InvalidExperiment(&'static str),
    NoFeasibleExperiment,
    NoDiscriminatingExperiment,
    SealAfterExecution,
    ExperimentMismatch,
    NoPredictions,
}

impl fmt::Display for D2dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModel(reason) => write!(f, "invalid model: {reason}"),
            Self::InvalidExperiment(reason) => write!(f, "invalid experiment: {reason}"),
            Self::NoFeasibleExperiment => write!(f, "no experiment fits the declared rollout budget"),
            Self::NoDiscriminatingExperiment => {
                write!(f, "no affordable experiment separates the supplied hypotheses")
            }
            Self::SealAfterExecution => {
                write!(f, "prediction seal must precede the execution sequence")
            }
            Self::ExperimentMismatch => {
                write!(f, "prediction seal and execution refer to different experiments")
            }
            Self::NoPredictions => write!(f, "prediction seal contains no hypotheses"),
        }
    }
}

impl Error for D2dError {}

/// One candidate dynamics explanation available to the agent.
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    id: String,
    friction: f64,
}

impl Hypothesis {
    pub fn new(id: impl Into<String>, friction: f64) -> Result<Self, D2dError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(D2dError::InvalidModel("hypothesis ID cannot be empty"));
        }
        if !friction.is_finite() || friction < 0.0 {
            return Err(D2dError::InvalidModel(
                "friction must be finite and non-negative",
            ));
        }
        Ok(Self { id, friction })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn friction(&self) -> f64 {
        self.friction
    }
}

/// Agent-visible experiment design for a one-dimensional damped-velocity world.
///
/// Dynamics are integrated as:
///
///     v[t+1] = v[t] + (control - friction * v[t]) * dt
///
/// Only the agent's candidate friction hypotheses are used for prediction.
/// The evaluator's true friction is held in a separate EvaluatorWorld.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentDesign {
    id: String,
    initial_velocity: f64,
    control: f64,
    dt: f64,
    steps: u32,
    sensor_noise_std: f64,
}

impl ExperimentDesign {
    pub fn new(
        id: impl Into<String>,
        initial_velocity: f64,
        control: f64,
        dt: f64,
        steps: u32,
        sensor_noise_std: f64,
    ) -> Result<Self, D2dError> {
        let design = Self {
            id: id.into(),
            initial_velocity,
            control,
            dt,
            steps,
            sensor_noise_std,
        };
        design.validate()?;
        Ok(design)
    }

    fn validate(&self) -> Result<(), D2dError> {
        if self.id.trim().is_empty() {
            return Err(D2dError::InvalidExperiment("experiment ID cannot be empty"));
        }
        if !self.initial_velocity.is_finite() || !self.control.is_finite() {
            return Err(D2dError::InvalidExperiment(
                "initial velocity and control must be finite",
            ));
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(D2dError::InvalidExperiment(
                "time step must be finite and positive",
            ));
        }
        if self.steps == 0 {
            return Err(D2dError::InvalidExperiment(
                "experiment must contain at least one step",
            ));
        }
        if !self.sensor_noise_std.is_finite() || self.sensor_noise_std <= 0.0 {
            return Err(D2dError::InvalidExperiment(
                "sensor noise standard deviation must be finite and positive",
            ));
        }
        Ok(())
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }

    pub fn sensor_noise_std(&self) -> f64 {
        self.sensor_noise_std
    }

    fn fixture_fingerprint(&self) -> u64 {
        let mut hash = FNV_OFFSET;
        hash_bytes(&mut hash, self.id.as_bytes());
        hash_u64(&mut hash, self.initial_velocity.to_bits());
        hash_u64(&mut hash, self.control.to_bits());
        hash_u64(&mut hash, self.dt.to_bits());
        hash_u64(&mut hash, u64::from(self.steps));
        hash_u64(&mut hash, self.sensor_noise_std.to_bits());
        hash
    }
}

/// The agent's candidate model. It has no reference to evaluator truth.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentModel {
    revision: String,
    hypotheses: Vec<Hypothesis>,
}

impl AgentModel {
    pub fn new(
        revision: impl Into<String>,
        hypotheses: Vec<Hypothesis>,
    ) -> Result<Self, D2dError> {
        let revision = revision.into();
        if revision.trim().is_empty() {
            return Err(D2dError::InvalidModel("model revision cannot be empty"));
        }
        if hypotheses.len() < 2 {
            return Err(D2dError::InvalidModel(
                "at least two competing hypotheses are required",
            ));
        }

        let mut ids = HashSet::new();
        for hypothesis in &hypotheses {
            if !ids.insert(hypothesis.id.as_str()) {
                return Err(D2dError::InvalidModel("hypothesis IDs must be unique"));
            }
        }

        Ok(Self {
            revision,
            hypotheses,
        })
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn hypotheses(&self) -> &[Hypothesis] {
        &self.hypotheses
    }

    /// Predict terminal velocity using the model-owned version of a hypothesis.
    pub fn predict_terminal_velocity(
        &self,
        hypothesis: &Hypothesis,
        experiment: &ExperimentDesign,
    ) -> Result<f64, D2dError> {
        experiment.validate()?;
        let owned = self
            .hypotheses
            .iter()
            .find(|item| item.id == hypothesis.id)
            .ok_or(D2dError::InvalidModel(
                "prediction requested for a hypothesis outside this model",
            ))?;
        Ok(integrate_terminal_velocity(
            experiment.initial_velocity,
            experiment.control,
            owned.friction,
            experiment.dt,
            experiment.steps,
        ))
    }

    /// Select the affordable experiment with the greatest discrimination
    /// separation per estimated rollout cost.
    ///
    /// This transparent score is a heuristic, not exact expected information gain:
    ///
    ///     score = (max predicted outcome - min predicted outcome)
    ///             / noise / sqrt(steps)
    ///
    /// It provides a baseline against which more principled selectors can be
    /// tested in later benchmark arms.
    pub fn select_experiment(
        &self,
        candidates: &[ExperimentDesign],
        rollout_budget: u32,
    ) -> Result<ExperimentSelection, D2dError> {
        let mut best: Option<(usize, f64)> = None;
        let mut any_affordable = false;

        for (index, experiment) in candidates.iter().enumerate() {
            experiment.validate()?;
            if experiment.steps > rollout_budget {
                continue;
            }
            any_affordable = true;

            let mut minimum = f64::INFINITY;
            let mut maximum = f64::NEG_INFINITY;
            for hypothesis in &self.hypotheses {
                let prediction = self.predict_terminal_velocity(hypothesis, experiment)?;
                minimum = minimum.min(prediction);
                maximum = maximum.max(prediction);
            }

            let separation = maximum - minimum;
            let score = separation
                / experiment.sensor_noise_std
                / f64::from(experiment.steps).sqrt();

            if !score.is_finite() {
                return Err(D2dError::InvalidExperiment(
                    "discrimination score is not finite",
                ));
            }
            if score <= 0.0 {
                continue;
            }

            let replace = match best {
                None => true,
                Some((best_index, best_score)) => {
                    score > best_score
                        || (score == best_score
                            && experiment.id < candidates[best_index].id)
                }
            };
            if replace {
                best = Some((index, score));
            }
        }

        if !any_affordable {
            return Err(D2dError::NoFeasibleExperiment);
        }
        let (index, score) = best.ok_or(D2dError::NoDiscriminatingExperiment)?;
        Ok(ExperimentSelection {
            experiment: candidates[index].clone(),
            discrimination_score_per_cost: score,
        })
    }
}

/// Selected experiment and its transparent heuristic score.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSelection {
    pub experiment: ExperimentDesign,
    pub discrimination_score_per_cost: f64,
}

/// One model prediction committed before the experiment is executed.
#[derive(Debug, Clone, PartialEq)]
pub struct SealedPrediction {
    hypothesis_id: String,
    terminal_velocity: f64,
}

impl SealedPrediction {
    pub fn hypothesis_id(&self) -> &str {
        &self.hypothesis_id
    }

    pub fn terminal_velocity(&self) -> f64 {
        self.terminal_velocity
    }
}

/// An immutable-by-interface prediction record for this process.
///
/// Its fingerprint is a deterministic, non-cryptographic fixture identifier.
/// It is not a signature, tamper-proof commitment, or security boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionSeal {
    experiment_id: String,
    model_revision: String,
    sequence: u64,
    experiment_fingerprint: u64,
    fingerprint: u64,
    predictions: Vec<SealedPrediction>,
}

impl PredictionSeal {
    pub fn create(
        model: &AgentModel,
        experiment: &ExperimentDesign,
        sequence: u64,
    ) -> Result<Self, D2dError> {
        experiment.validate()?;
        let predictions: Vec<SealedPrediction> = model
            .hypotheses
            .iter()
            .map(|hypothesis| {
                Ok(SealedPrediction {
                    hypothesis_id: hypothesis.id.clone(),
                    terminal_velocity: model
                        .predict_terminal_velocity(hypothesis, experiment)?,
                })
            })
            .collect::<Result<_, D2dError>>()?;

        if predictions.is_empty() {
            return Err(D2dError::NoPredictions);
        }

        let experiment_fingerprint = experiment.fixture_fingerprint();
        let fingerprint = prediction_fingerprint(
            &model.revision,
            &experiment.id,
            sequence,
            experiment_fingerprint,
            &predictions,
        );

        Ok(Self {
            experiment_id: experiment.id.clone(),
            model_revision: model.revision.clone(),
            sequence,
            experiment_fingerprint,
            fingerprint,
            predictions,
        })
    }

    pub fn experiment_id(&self) -> &str {
        &self.experiment_id
    }

    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn predictions(&self) -> &[SealedPrediction] {
        &self.predictions
    }
}

/// Evaluator-only world configuration.
///
/// Keep this type and its construction in the evaluator process in any real
/// deployment. This same-process fixture demonstrates the API boundary only;
/// it does not enforce isolation or prevent a learner from reading this value.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatorWorld {
    true_friction: f64,
}

impl EvaluatorWorld {
    pub fn from_known_truth(true_friction: f64) -> Result<Self, D2dError> {
        if !true_friction.is_finite() || true_friction < 0.0 {
            return Err(D2dError::InvalidModel(
                "evaluator friction must be finite and non-negative",
            ));
        }
        Ok(Self { true_friction })
    }

    /// Execute the exact experiment in the evaluator's known-truth world.
    pub fn execute(
        &self,
        experiment: &ExperimentDesign,
        sequence: u64,
    ) -> Result<ExecutionReceipt, D2dError> {
        experiment.validate()?;
        let terminal_velocity = integrate_terminal_velocity(
            experiment.initial_velocity,
            experiment.control,
            self.true_friction,
            experiment.dt,
            experiment.steps,
        );

        Ok(ExecutionReceipt {
            experiment_id: experiment.id.clone(),
            branch_id: experiment.id.clone(),
            sequence,
            experiment_fingerprint: experiment.fixture_fingerprint(),
            terminal_velocity,
            disposition: BranchDisposition::Completed,
        })
    }
}

/// Evaluator-produced outcome. The true parameter is deliberately not exposed.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReceipt {
    experiment_id: String,
    branch_id: String,
    sequence: u64,
    experiment_fingerprint: u64,
    terminal_velocity: f64,
    disposition: BranchDisposition,
}

impl ExecutionReceipt {
    pub fn experiment_id(&self) -> &str {
        &self.experiment_id
    }

    pub fn branch_id(&self) -> &str {
        &self.branch_id
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn terminal_velocity(&self) -> f64 {
        self.terminal_velocity
    }

    pub fn disposition(&self) -> BranchDisposition {
        self.disposition
    }
}

/// Prediction error for one candidate hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct HypothesisError {
    pub hypothesis_id: String,
    pub absolute_error: f64,
}

/// Independent comparison of a sealed prediction against an executed outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationResult {
    pub experiment_id: String,
    pub inferred_hypothesis: String,
    pub observed_terminal_velocity: f64,
    pub best_absolute_error: f64,
    pub errors_by_hypothesis: Vec<HypothesisError>,
    pub prediction_fingerprint: u64,
}

pub fn evaluate_sealed_prediction(
    seal: &PredictionSeal,
    execution: &ExecutionReceipt,
) -> Result<EvaluationResult, D2dError> {
    if seal.sequence >= execution.sequence {
        return Err(D2dError::SealAfterExecution);
    }
    if seal.experiment_id != execution.experiment_id
        || seal.experiment_fingerprint != execution.experiment_fingerprint
        || execution.disposition != BranchDisposition::Completed
    {
        return Err(D2dError::ExperimentMismatch);
    }
    if seal.predictions.is_empty() {
        return Err(D2dError::NoPredictions);
    }

    let errors_by_hypothesis: Vec<HypothesisError> = seal
        .predictions
        .iter()
        .map(|prediction| HypothesisError {
            hypothesis_id: prediction.hypothesis_id.clone(),
            absolute_error: (prediction.terminal_velocity - execution.terminal_velocity).abs(),
        })
        .collect();

    let best = errors_by_hypothesis
        .iter()
        .min_by(|left, right| {
            left.absolute_error
                .total_cmp(&right.absolute_error)
                .then_with(|| left.hypothesis_id.cmp(&right.hypothesis_id))
        })
        .ok_or(D2dError::NoPredictions)?;

    Ok(EvaluationResult {
        experiment_id: seal.experiment_id.clone(),
        inferred_hypothesis: best.hypothesis_id.clone(),
        observed_terminal_velocity: execution.terminal_velocity,
        best_absolute_error: best.absolute_error,
        errors_by_hypothesis,
        prediction_fingerprint: seal.fingerprint,
    })
}

/// Every expected branch remains represented, including branches that did not
/// complete. A missing branch is unknown/missing; it is never inferred to be
/// failed, infeasible, or successful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchDisposition {
    Completed,
    Failed,
    Incomplete,
    Unsupported,
    ResourceBound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchRecord {
    branch_id: String,
    disposition: BranchDisposition,
}

impl BranchRecord {
    pub fn new(branch_id: impl Into<String>, disposition: BranchDisposition) -> Self {
        Self {
            branch_id: branch_id.into(),
            disposition,
        }
    }

    pub fn branch_id(&self) -> &str {
        &self.branch_id
    }

    pub fn disposition(&self) -> BranchDisposition {
        self.disposition
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchAudit {
    pub expected_count: usize,
    pub reported_count: usize,
    pub completed_count: usize,
    pub non_completed_count: usize,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub duplicate_expected: Vec<String>,
    pub duplicate_reports: Vec<String>,
}

impl BranchAudit {
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
            && self.unexpected.is_empty()
            && self.duplicate_expected.is_empty()
            && self.duplicate_reports.is_empty()
    }
}

/// Audit the expected-branch denominator without converting unknowns into
/// failure or infeasibility labels.
pub fn audit_expected_branches(
    expected: &[String],
    observed: &[BranchRecord],
) -> BranchAudit {
    let mut expected_set = HashSet::new();
    let mut duplicate_expected = Vec::new();
    for branch_id in expected {
        if !expected_set.insert(branch_id.as_str()) {
            duplicate_expected.push(branch_id.clone());
        }
    }

    let mut seen_reports = HashSet::new();
    let mut duplicate_reports = Vec::new();
    let mut unexpected = Vec::new();
    let mut completed_count = 0;
    let mut non_completed_count = 0;

    for record in observed {
        if !expected_set.contains(record.branch_id.as_str()) {
            unexpected.push(record.branch_id.clone());
            continue;
        }
        if !seen_reports.insert(record.branch_id.as_str()) {
            duplicate_reports.push(record.branch_id.clone());
            continue;
        }
        match record.disposition {
            BranchDisposition::Completed => completed_count += 1,
            BranchDisposition::Failed
            | BranchDisposition::Incomplete
            | BranchDisposition::Unsupported
            | BranchDisposition::ResourceBound => non_completed_count += 1,
        }
    }

    let missing = expected
        .iter()
        .filter(|branch_id| !seen_reports.contains(branch_id.as_str()))
        .cloned()
        .collect();

    BranchAudit {
        expected_count: expected.len(),
        reported_count: observed.len(),
        completed_count,
        non_completed_count,
        missing,
        unexpected,
        duplicate_expected,
        duplicate_reports,
    }
}

fn integrate_terminal_velocity(
    initial_velocity: f64,
    control: f64,
    friction: f64,
    dt: f64,
    steps: u32,
) -> f64 {
    let mut velocity = initial_velocity;
    for _ in 0..steps {
        velocity += (control - friction * velocity) * dt;
    }
    velocity
}

// Stable fixture fingerprint only. Do not use as a security commitment.
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    hash_u64(hash, bytes.len() as u64);
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn hash_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn prediction_fingerprint(
    model_revision: &str,
    experiment_id: &str,
    sequence: u64,
    experiment_fingerprint: u64,
    predictions: &[SealedPrediction],
) -> u64 {
    let mut hash = FNV_OFFSET;
    hash_bytes(&mut hash, model_revision.as_bytes());
    hash_bytes(&mut hash, experiment_id.as_bytes());
    hash_u64(&mut hash, sequence);
    hash_u64(&mut hash, experiment_fingerprint);
    hash_u64(&mut hash, predictions.len() as u64);
    for prediction in predictions {
        hash_bytes(&mut hash, prediction.hypothesis_id.as_bytes());
        hash_u64(&mut hash, prediction.terminal_velocity.to_bits());
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(friction_values: &[f64]) -> AgentModel {
        let hypotheses = friction_values
            .iter()
            .enumerate()
            .map(|(index, friction)| {
                Hypothesis::new(format!("h{index}"), *friction).unwrap()
            })
            .collect();
        AgentModel::new("fixture-model-v1", hypotheses).unwrap()
    }

    fn candidates() -> Vec<ExperimentDesign> {
        [1, 2, 5, 10]
            .into_iter()
            .map(|steps| {
                ExperimentDesign::new(
                    format!("coast-{steps}"),
                    1.0,
                    0.0,
                    0.1,
                    steps,
                    0.1,
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn selects_most_discriminating_experiment_within_budget() {
        let selected = model(&[0.1, 0.8])
            .select_experiment(&candidates(), 5)
            .unwrap();
        assert_eq!(selected.experiment.steps(), 5);
        assert!(selected.discrimination_score_per_cost > 0.0);
    }

    #[test]
    fn selection_is_order_independent_for_unique_scores() {
        let model = model(&[0.1, 0.8]);
        let candidates = candidates();
        let forward = model.select_experiment(&candidates, 5).unwrap();
        let mut reversed = candidates;
        reversed.reverse();
        let backward = model.select_experiment(&reversed, 5).unwrap();
        assert_eq!(forward.experiment.id(), backward.experiment.id());
        assert_eq!(forward.experiment.steps(), backward.experiment.steps());
    }

    #[test]
    fn sealed_predictions_identify_the_known_truth() {
        let model = model(&[0.1, 0.8]);
        let selected = model.select_experiment(&candidates(), 5).unwrap();
        let seal = PredictionSeal::create(&model, &selected.experiment, 10).unwrap();
        let world = EvaluatorWorld::from_known_truth(0.8).unwrap();
        let execution = world.execute(&selected.experiment, 11).unwrap();
        let evaluation = evaluate_sealed_prediction(&seal, &execution).unwrap();

        assert_eq!(evaluation.inferred_hypothesis, "h1");
        assert!(evaluation.best_absolute_error < 1e-12);
        assert_eq!(evaluation.prediction_fingerprint, seal.fingerprint());
    }

    #[test]
    fn rejects_a_prediction_seal_created_after_execution() {
        let model = model(&[0.1, 0.8]);
        let design = candidates().remove(2);
        let seal = PredictionSeal::create(&model, &design, 12).unwrap();
        let execution = EvaluatorWorld::from_known_truth(0.8)
            .unwrap()
            .execute(&design, 11)
            .unwrap();

        assert_eq!(
            evaluate_sealed_prediction(&seal, &execution),
            Err(D2dError::SealAfterExecution)
        );
    }

    #[test]
    fn rejects_an_experiment_substitution() {
        let model = model(&[0.1, 0.8]);
        let designs = candidates();
        let seal = PredictionSeal::create(&model, &designs[2], 1).unwrap();
        let substituted = ExperimentDesign::new(
            "coast-5",
            2.0,
            0.0,
            0.1,
            5,
            0.1,
        )
        .unwrap();
        let execution = EvaluatorWorld::from_known_truth(0.8)
            .unwrap()
            .execute(&substituted, 2)
            .unwrap();

        assert_eq!(
            evaluate_sealed_prediction(&seal, &execution),
            Err(D2dError::ExperimentMismatch)
        );
    }

    #[test]
    fn abstains_when_no_affordable_experiment_separates_hypotheses() {
        let result = model(&[0.4, 0.4]).select_experiment(&candidates(), 10);
        assert_eq!(result, Err(D2dError::NoDiscriminatingExperiment));
    }

    #[test]
    fn rejects_when_no_candidate_fits_the_rollout_budget() {
        let result = model(&[0.1, 0.8]).select_experiment(&candidates(), 0);
        assert_eq!(result, Err(D2dError::NoFeasibleExperiment));
    }

    #[test]
    fn branch_audit_preserves_missing_and_non_completed_dispositions() {
        let expected = vec![
            "branch-a".to_string(),
            "branch-b".to_string(),
            "branch-c".to_string(),
        ];
        let observed = vec![
            BranchRecord::new("branch-a", BranchDisposition::Completed),
            BranchRecord::new("branch-b", BranchDisposition::Unsupported),
        ];
        let audit = audit_expected_branches(&expected, &observed);

        assert_eq!(audit.expected_count, 3);
        assert_eq!(audit.completed_count, 1);
        assert_eq!(audit.non_completed_count, 1);
        assert_eq!(audit.missing, vec!["branch-c".to_string()]);
        assert!(!audit.is_complete());
    }

    #[test]
    fn branch_audit_detects_duplicate_and_unexpected_reports() {
        let expected = vec!["branch-a".to_string()];
        let observed = vec![
            BranchRecord::new("branch-a", BranchDisposition::Completed),
            BranchRecord::new("branch-a", BranchDisposition::Failed),
            BranchRecord::new("branch-x", BranchDisposition::Completed),
        ];
        let audit = audit_expected_branches(&expected, &observed);

        assert_eq!(audit.duplicate_reports, vec!["branch-a".to_string()]);
        assert_eq!(audit.unexpected, vec!["branch-x".to_string()]);
        assert!(!audit.is_complete());
    }

    #[test]
    fn branch_audit_flags_duplicate_expected_ids() {
        let expected = vec!["branch-a".to_string(), "branch-a".to_string()];
        let observed = vec![BranchRecord::new(
            "branch-a",
            BranchDisposition::Completed,
        )];
        let audit = audit_expected_branches(&expected, &observed);

        assert_eq!(audit.duplicate_expected, vec!["branch-a".to_string()]);
        assert!(!audit.is_complete());
    }
}
