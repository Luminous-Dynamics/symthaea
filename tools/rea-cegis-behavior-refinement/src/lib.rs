//! Finite-domain counterexample-guided behavioral hypothesis refinement.
//!
//! V0 is a pure, deterministic kernel over synthetic fixtures. It does not
//! execute targets, admit REA Evidence, authenticate runners, or establish
//! behavior outside the finite input domain supplied by the caller.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementError(pub String);

impl fmt::Display for RefinementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RefinementError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Termination {
    Exit(i32),
    Signal(u32),
    Timeout,
    LaunchError(String),
}

/// Exact outcome projection used by this finite-domain kernel.
///
/// `filesystem_delta_sha256: None` means the fixture does not model a
/// filesystem delta. It is not evidence that the target had no such effects.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Outcome {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub termination: Termination,
    pub filesystem_delta_sha256: Option<String>,
}

impl Outcome {
    pub fn exited(
        stdout: impl Into<Vec<u8>>,
        stderr: impl Into<Vec<u8>>,
        code: i32,
    ) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: stderr.into(),
            termination: Termination::Exit(code),
            filesystem_delta_sha256: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prediction {
    pub input: Vec<u8>,
    pub outcome: Outcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehavioralHypothesis {
    pub id: String,
    /// Exact target artifact SHA-256, lowercase hexadecimal.
    pub target_sha256: String,
    /// Digest of the declared environment/runtime scope, lowercase hex.
    pub environment_sha256: String,
    /// Predictions are defined only for this finite, explicit input domain.
    pub predictions: Vec<Prediction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureState {
    Complete,
    NonComplete { reason: String },
}

/// V0 accepts synthetic oracle observations only. Real observations must pass
/// through the separate REA Evidence admission and capture-binding path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationOrigin {
    SyntheticFixture,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    pub target_sha256: String,
    pub environment_sha256: String,
    pub input: Vec<u8>,
    pub capture: CaptureState,
    pub outcome: Option<Outcome>,
    pub origin: ObservationOrigin,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementContext {
    pub target_sha256: String,
    pub environment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HypothesisDisposition {
    ConsistentWithObservedSyntheticCases,
    Contradicted {
        observation_indices: Vec<usize>,
    },
    Unresolved {
        reasons: Vec<String>,
        matched_observations: usize,
    },
    NotEvaluated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypothesisResult {
    pub hypothesis_id: String,
    pub disposition: HypothesisDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementReport {
    pub context: RefinementContext,
    pub hypotheses: Vec<HypothesisResult>,
    /// Inputs whose repeated complete synthetic observations disagreed.
    pub nondeterministic_inputs: Vec<Vec<u8>>,
    pub synthetic_observation_count: usize,
    pub claim_ceiling: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeScore {
    pub input: Vec<u8>,
    pub distinct_predicted_outcomes: usize,
    pub separated_hypothesis_pairs: usize,
}

fn validate_sha256(value: &str, field: &str) -> Result<(), RefinementError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RefinementError(format!(
            "{field} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_hypothesis(
    hypothesis: &BehavioralHypothesis,
    context: &RefinementContext,
) -> Result<BTreeMap<Vec<u8>, Outcome>, RefinementError> {
    if hypothesis.id.trim().is_empty() {
        return Err(RefinementError("hypothesis ID must not be empty".into()));
    }
    validate_sha256(&hypothesis.target_sha256, "hypothesis target_sha256")?;
    validate_sha256(
        &hypothesis.environment_sha256,
        "hypothesis environment_sha256",
    )?;
    if hypothesis.target_sha256 != context.target_sha256 {
        return Err(RefinementError(format!(
            "hypothesis {} target digest does not match refinement context",
            hypothesis.id
        )));
    }
    if hypothesis.environment_sha256 != context.environment_sha256 {
        return Err(RefinementError(format!(
            "hypothesis {} environment digest does not match refinement context",
            hypothesis.id
        )));
    }
    if hypothesis.predictions.is_empty() {
        return Err(RefinementError(format!(
            "hypothesis {} has an empty finite input domain",
            hypothesis.id
        )));
    }

    let mut indexed = BTreeMap::new();
    for prediction in &hypothesis.predictions {
        if indexed
            .insert(prediction.input.clone(), prediction.outcome.clone())
            .is_some()
        {
            return Err(RefinementError(format!(
                "hypothesis {} contains duplicate input predictions",
                hypothesis.id
            )));
        }
        if let Some(digest) = &prediction.outcome.filesystem_delta_sha256 {
            validate_sha256(digest, "filesystem_delta_sha256")?;
        }
    }
    Ok(indexed)
}

/// Refine hypotheses using complete synthetic observations.
///
/// Non-complete captures never eliminate a hypothesis. If repeated complete
/// observations for one input disagree, all observations for that input are
/// withheld from deterministic-hypothesis elimination. Target/environment
/// mismatches reject the entire call.
pub fn refine_hypotheses(
    context: &RefinementContext,
    hypotheses: &[BehavioralHypothesis],
    observations: &[Observation],
) -> Result<RefinementReport, RefinementError> {
    validate_sha256(&context.target_sha256, "context target_sha256")?;
    validate_sha256(
        &context.environment_sha256,
        "context environment_sha256",
    )?;

    let mut ids = BTreeSet::new();
    let mut indexed_hypotheses = Vec::with_capacity(hypotheses.len());
    for hypothesis in hypotheses {
        if !ids.insert(hypothesis.id.clone()) {
            return Err(RefinementError(format!(
                "duplicate hypothesis ID {}",
                hypothesis.id
            )));
        }
        indexed_hypotheses.push((
            hypothesis.id.clone(),
            validate_hypothesis(hypothesis, context)?,
        ));
    }

    let mut complete_by_input: BTreeMap<Vec<u8>, BTreeSet<Outcome>> = BTreeMap::new();
    for observation in observations {
        if observation.target_sha256 != context.target_sha256 {
            return Err(RefinementError(
                "observation target digest does not match refinement context".into(),
            ));
        }
        if observation.environment_sha256 != context.environment_sha256 {
            return Err(RefinementError(
                "observation environment digest does not match refinement context".into(),
            ));
        }
        match (&observation.capture, &observation.outcome) {
            (CaptureState::Complete, Some(outcome)) => {
                if let Some(digest) = &outcome.filesystem_delta_sha256 {
                    validate_sha256(digest, "observation filesystem_delta_sha256")?;
                }
                complete_by_input
                    .entry(observation.input.clone())
                    .or_default()
                    .insert(outcome.clone());
            }
            (CaptureState::Complete, None) => {
                return Err(RefinementError(
                    "complete capture must include a complete outcome".into(),
                ));
            }
            (CaptureState::NonComplete { .. }, _) => {}
        }
    }

    let nondeterministic: BTreeSet<Vec<u8>> = complete_by_input
        .iter()
        .filter_map(|(input, outcomes)| (outcomes.len() > 1).then_some(input.clone()))
        .collect();

    let mut results = Vec::with_capacity(indexed_hypotheses.len());
    for (hypothesis_id, predictions) in indexed_hypotheses {
        let mut contradicted = Vec::new();
        let mut reasons = Vec::new();
        let mut matched = 0usize;

        for (index, observation) in observations.iter().enumerate() {
            let Some(expected) = predictions.get(&observation.input) else {
                continue;
            };
            if nondeterministic.contains(&observation.input) {
                reasons.push(format!(
                    "observation {index}: repeated complete outcomes disagree for this input"
                ));
                continue;
            }
            match (&observation.capture, &observation.outcome) {
                (CaptureState::Complete, Some(observed)) => {
                    if expected == observed {
                        matched += 1;
                    } else {
                        contradicted.push(index);
                    }
                }
                (CaptureState::NonComplete { reason }, _) => {
                    reasons.push(format!(
                        "observation {index}: capture not complete: {reason}"
                    ));
                }
                (CaptureState::Complete, None) => unreachable!("validated above"),
            }
        }

        let disposition = if !contradicted.is_empty() {
            HypothesisDisposition::Contradicted {
                observation_indices: contradicted,
            }
        } else if !reasons.is_empty() {
            HypothesisDisposition::Unresolved {
                reasons,
                matched_observations: matched,
            }
        } else if matched > 0 {
            HypothesisDisposition::ConsistentWithObservedSyntheticCases
        } else {
            HypothesisDisposition::NotEvaluated
        };
        results.push(HypothesisResult {
            hypothesis_id,
            disposition,
        });
    }

    Ok(RefinementReport {
        context: context.clone(),
        hypotheses: results,
        nondeterministic_inputs: nondeterministic.into_iter().collect(),
        synthetic_observation_count: observations.len(),
        claim_ceiling: "synthetic finite-domain consistency only; no target execution, Evidence admission, runner authenticity, or generalization",
    })
}

/// Rank inputs that distinguish a shared hypothesis set.
///
/// Only inputs predicted by every hypothesis are eligible. Ranking maximizes
/// distinct predicted outcomes, then separated unordered pairs, with
/// byte-lexicographic tie breaking. This is not EIG, EVSI, obligation closure,
/// or an authorization decision.
pub fn rank_distinguishing_inputs(
    context: &RefinementContext,
    hypotheses: &[BehavioralHypothesis],
    candidate_inputs: &[Vec<u8>],
) -> Result<Vec<ProbeScore>, RefinementError> {
    if hypotheses.is_empty() {
        return Err(RefinementError(
            "at least one hypothesis is required".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut indexed = Vec::with_capacity(hypotheses.len());
    for hypothesis in hypotheses {
        if !ids.insert(hypothesis.id.clone()) {
            return Err(RefinementError(format!(
                "duplicate hypothesis ID {}",
                hypothesis.id
            )));
        }
        indexed.push(validate_hypothesis(hypothesis, context)?);
    }

    let mut unique_candidates = BTreeSet::new();
    let mut scores = Vec::new();
    for input in candidate_inputs {
        if !unique_candidates.insert(input.clone()) {
            continue;
        }
        let mut outcomes = BTreeSet::new();
        let mut predicted = Vec::with_capacity(indexed.len());
        let mut common = true;
        for hypothesis in &indexed {
            match hypothesis.get(input) {
                Some(outcome) => {
                    outcomes.insert(outcome.clone());
                    predicted.push(outcome);
                }
                None => {
                    common = false;
                    break;
                }
            }
        }
        if !common || indexed.len() < 2 {
            continue;
        }

        let mut separated_pairs = 0usize;
        for left in 0..predicted.len() {
            for right in (left + 1)..predicted.len() {
                if predicted[left] != predicted[right] {
                    separated_pairs += 1;
                }
            }
        }
        scores.push(ProbeScore {
            input: input.clone(),
            distinct_predicted_outcomes: outcomes.len(),
            separated_hypothesis_pairs: separated_pairs,
        });
    }

    scores.sort_by(|a, b| {
        b.distinct_predicted_outcomes
            .cmp(&a.distinct_predicted_outcomes)
            .then_with(|| {
                b.separated_hypothesis_pairs
                    .cmp(&a.separated_hypothesis_pairs)
            })
            .then_with(|| a.input.cmp(&b.input))
    });
    Ok(scores)
}

/// Group hypotheses indistinguishable over explicitly named finite inputs.
///
/// Every evaluation input must be predicted by every hypothesis; otherwise
/// the comparison is under-specified and rejects.
pub fn observational_equivalence_classes(
    context: &RefinementContext,
    hypotheses: &[BehavioralHypothesis],
    evaluation_inputs: &[Vec<u8>],
) -> Result<Vec<Vec<String>>, RefinementError> {
    if hypotheses.is_empty() || evaluation_inputs.is_empty() {
        return Err(RefinementError(
            "hypotheses and evaluation_inputs must both be non-empty".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut groups: BTreeMap<Vec<Outcome>, Vec<String>> = BTreeMap::new();
    for hypothesis in hypotheses {
        if !ids.insert(hypothesis.id.clone()) {
            return Err(RefinementError(format!(
                "duplicate hypothesis ID {}",
                hypothesis.id
            )));
        }
        let predictions = validate_hypothesis(hypothesis, context)?;
        let mut signature = Vec::with_capacity(evaluation_inputs.len());
        for input in evaluation_inputs {
            signature.push(predictions.get(input).cloned().ok_or_else(|| {
                RefinementError(format!(
                    "evaluation input is outside hypothesis {} domain",
                    hypothesis.id
                ))
            })?);
        }
        groups
            .entry(signature)
            .or_default()
            .push(hypothesis.id.clone());
    }

    let mut classes: Vec<Vec<String>> = groups.into_values().collect();
    for class in &mut classes {
        class.sort();
    }
    classes.sort_by(|a, b| a[0].cmp(&b[0]));
    Ok(classes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinimizedCounterexample {
    pub bytes: Vec<u8>,
    pub original_len: usize,
    pub minimized_len: usize,
    pub predicate_calls: usize,
    /// No single-byte deletion preserves the discrepancy. This is not a
    /// guarantee of globally minimum size.
    pub one_deletion_minimal: bool,
}

/// Greedily remove bytes while a caller-supplied discrepancy predicate remains
/// true. This function never executes a process; callers own the oracle and
/// must rerun both sides under identical declared conditions.
pub fn minimize_counterexample<F>(
    original: &[u8],
    mut preserves_discrepancy: F,
) -> Result<MinimizedCounterexample, RefinementError>
where
    F: FnMut(&[u8]) -> bool,
{
    let mut calls = 1usize;
    if !preserves_discrepancy(original) {
        return Err(RefinementError(
            "original input does not reproduce the declared discrepancy".into(),
        ));
    }

    let original_len = original.len();
    let mut current = original.to_vec();
    let mut index = 0usize;
    while index < current.len() {
        let mut candidate = current.clone();
        candidate.remove(index);
        calls += 1;
        if preserves_discrepancy(&candidate) {
            current = candidate;
            index = 0;
        } else {
            index += 1;
        }
    }

    calls += 1;
    if !preserves_discrepancy(&current) {
        return Err(RefinementError(
            "minimized counterexample failed final discrepancy reproduction".into(),
        ));
    }

    Ok(MinimizedCounterexample {
        minimized_len: current.len(),
        bytes: current,
        original_len,
        predicate_calls: calls,
        one_deletion_minimal: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const ENV: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn context() -> RefinementContext {
        RefinementContext {
            target_sha256: TARGET.into(),
            environment_sha256: ENV.into(),
        }
    }

    fn outcome(stdout: &[u8]) -> Outcome {
        Outcome::exited(stdout.to_vec(), Vec::<u8>::new(), 0)
    }

    fn hypothesis(
        id: &str,
        predictions: Vec<(Vec<u8>, Outcome)>,
    ) -> BehavioralHypothesis {
        BehavioralHypothesis {
            id: id.into(),
            target_sha256: TARGET.into(),
            environment_sha256: ENV.into(),
            predictions: predictions
                .into_iter()
                .map(|(input, outcome)| Prediction { input, outcome })
                .collect(),
        }
    }

    fn observation(input: &[u8], result: Outcome) -> Observation {
        Observation {
            target_sha256: TARGET.into(),
            environment_sha256: ENV.into(),
            input: input.to_vec(),
            capture: CaptureState::Complete,
            outcome: Some(result),
            origin: ObservationOrigin::SyntheticFixture,
        }
    }

    #[test]
    fn complete_counterexample_eliminates_only_contradicted_hypothesis() {
        let hypotheses = vec![
            hypothesis("ascii", vec![(b"utf8".to_vec(), outcome(b"reject"))]),
            hypothesis("utf8", vec![(b"utf8".to_vec(), outcome(b"accept"))]),
        ];
        let report = refine_hypotheses(
            &context(),
            &hypotheses,
            &[observation(b"utf8", outcome(b"accept"))],
        )
        .unwrap();

        assert_eq!(
            report.hypotheses[0].disposition,
            HypothesisDisposition::Contradicted {
                observation_indices: vec![0]
            }
        );
        assert_eq!(
            report.hypotheses[1].disposition,
            HypothesisDisposition::ConsistentWithObservedSyntheticCases
        );
        assert!(report.claim_ceiling.contains("synthetic finite-domain"));
    }

    #[test]
    fn partial_capture_never_eliminates_a_hypothesis() {
        let hypotheses = vec![hypothesis(
            "h1",
            vec![(b"x".to_vec(), outcome(b"expected"))],
        )];
        let observation = Observation {
            target_sha256: TARGET.into(),
            environment_sha256: ENV.into(),
            input: b"x".to_vec(),
            capture: CaptureState::NonComplete {
                reason: "stderr unavailable".into(),
            },
            outcome: Some(outcome(b"different")),
            origin: ObservationOrigin::SyntheticFixture,
        };
        let report = refine_hypotheses(&context(), &hypotheses, &[observation]).unwrap();

        assert!(matches!(
            report.hypotheses[0].disposition,
            HypothesisDisposition::Unresolved { .. }
        ));
    }

    #[test]
    fn scope_mismatch_rejects_instead_of_reusing_observation() {
        let hypotheses = vec![hypothesis(
            "h1",
            vec![(b"x".to_vec(), outcome(b"ok"))],
        )];
        let mut observation = observation(b"x", outcome(b"ok"));
        observation.target_sha256 = "c".repeat(64);

        assert!(refine_hypotheses(&context(), &hypotheses, &[observation]).is_err());
    }

    #[test]
    fn repeated_disagreement_is_nondeterministic_not_a_counterexample() {
        let hypotheses = vec![hypothesis(
            "deterministic",
            vec![(b"x".to_vec(), outcome(b"first"))],
        )];
        let observations = vec![
            observation(b"x", outcome(b"first")),
            observation(b"x", outcome(b"second")),
        ];
        let report = refine_hypotheses(&context(), &hypotheses, &observations).unwrap();

        assert_eq!(report.nondeterministic_inputs, vec![b"x".to_vec()]);
        assert!(matches!(
            report.hypotheses[0].disposition,
            HypothesisDisposition::Unresolved { .. }
        ));
    }

    #[test]
    fn distinguishing_probe_ranks_maximum_outcome_partition_first() {
        let hypotheses = vec![
            hypothesis(
                "h1",
                vec![
                    (b"a".to_vec(), outcome(b"0")),
                    (b"b".to_vec(), outcome(b"0")),
                ],
            ),
            hypothesis(
                "h2",
                vec![
                    (b"a".to_vec(), outcome(b"1")),
                    (b"b".to_vec(), outcome(b"0")),
                ],
            ),
            hypothesis(
                "h3",
                vec![
                    (b"a".to_vec(), outcome(b"2")),
                    (b"b".to_vec(), outcome(b"1")),
                ],
            ),
        ];
        let scores = rank_distinguishing_inputs(
            &context(),
            &hypotheses,
            &[b"b".to_vec(), b"a".to_vec()],
        )
        .unwrap();

        assert_eq!(scores[0].input, b"a".to_vec());
        assert_eq!(scores[0].distinct_predicted_outcomes, 3);
        assert_eq!(scores[0].separated_hypothesis_pairs, 3);
    }

    #[test]
    fn equivalence_is_only_over_the_named_finite_inputs() {
        let hypotheses = vec![
            hypothesis(
                "h1",
                vec![
                    (b"x".to_vec(), outcome(b"same")),
                    (b"y".to_vec(), outcome(b"same")),
                ],
            ),
            hypothesis(
                "h2",
                vec![
                    (b"x".to_vec(), outcome(b"same")),
                    (b"y".to_vec(), outcome(b"same")),
                ],
            ),
            hypothesis(
                "h3",
                vec![
                    (b"x".to_vec(), outcome(b"different")),
                    (b"y".to_vec(), outcome(b"same")),
                ],
            ),
        ];
        let classes = observational_equivalence_classes(
            &context(),
            &hypotheses,
            &[b"x".to_vec(), b"y".to_vec()],
        )
        .unwrap();

        assert_eq!(
            classes,
            vec![
                vec!["h1".to_string(), "h2".to_string()],
                vec!["h3".to_string()]
            ]
        );
    }

    #[test]
    fn one_deletion_minimizer_preserves_original_discrepancy() {
        let minimized = minimize_counterexample(b"prefix:BAD:suffix", |candidate| {
            candidate.windows(3).any(|window| window == b"BAD")
        })
        .unwrap();

        assert_eq!(minimized.bytes, b"BAD");
        assert_eq!(minimized.original_len, 17);
        assert!(minimized.one_deletion_minimal);
        assert!(minimized.predicate_calls > 1);
    }

    #[test]
    fn minimizer_rejects_input_that_does_not_reproduce_discrepancy() {
        assert!(minimize_counterexample(b"ordinary", |_| false).is_err());
    }

    #[test]
    fn duplicate_hypothesis_ids_reject() {
        let hypotheses = vec![
            hypothesis("same", vec![(b"x".to_vec(), outcome(b"0"))]),
            hypothesis("same", vec![(b"x".to_vec(), outcome(b"1"))]),
        ];

        assert!(refine_hypotheses(&context(), &hypotheses, &[]).is_err());
    }

    #[test]
    fn duplicate_input_predictions_reject() {
        let hypotheses = vec![hypothesis(
            "h1",
            vec![
                (b"x".to_vec(), outcome(b"0")),
                (b"x".to_vec(), outcome(b"1")),
            ],
        )];

        assert!(refine_hypotheses(&context(), &hypotheses, &[]).is_err());
    }

    #[test]
    fn complete_capture_without_outcome_rejects() {
        let hypotheses = vec![hypothesis(
            "h1",
            vec![(b"x".to_vec(), outcome(b"0"))],
        )];
        let observation = Observation {
            target_sha256: TARGET.into(),
            environment_sha256: ENV.into(),
            input: b"x".to_vec(),
            capture: CaptureState::Complete,
            outcome: None,
            origin: ObservationOrigin::SyntheticFixture,
        };

        assert!(refine_hypotheses(&context(), &hypotheses, &[observation]).is_err());
    }
}
