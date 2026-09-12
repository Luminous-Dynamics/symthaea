// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Matched-transition campaign evidence for EUREKA-002B.
//!
//! Candidate and every shortcut baseline are scored against one canonical
//! public transition identity. Missing baseline coverage remains visible.

use super::action_execution::{ActionExecutionStatus, QualifiedActionReceipt};
use super::baselines::{
    FittedShortcutBaselines, HeldOutTransitionCorpus, PublicTransitionRecord, ShortcutBaselineKind,
};
use super::consequence::{
    ConsequenceScore, ConsequenceScoringError, copy_current_state_baseline, score_consequence,
};
use super::custody::{
    COPY_CURRENT_STATE_BASELINE_ID, CustodyError, FrozenPrediction, MatchedConsequenceTrial,
    PredictionCustodian,
};
use super::hidden_world::{PublicAction, StepReceipt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum BaselineIdentity {
    CopyCurrentState,
    Shortcut(ShortcutBaselineKind),
}

impl BaselineIdentity {
    pub(super) const fn stable_id(self) -> &'static str {
        match self {
            Self::CopyCurrentState => COPY_CURRENT_STATE_BASELINE_ID,
            Self::Shortcut(kind) => kind.stable_id(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct BaselineScoreRecord {
    pub identity: BaselineIdentity,
    pub score: ConsequenceScore,
    pub fit_corpus_digest: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrialDisposition {
    Scored,
    Abstained,
    OutOfDomain,
}

impl TrialDisposition {
    fn from_score(score: ConsequenceScore) -> Self {
        match score {
            ConsequenceScore::Scored(_) => Self::Scored,
            ConsequenceScore::AbstainedInsufficientEvidence => Self::Abstained,
            ConsequenceScore::OutOfQualifiedDomain => Self::OutOfDomain,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PairedTransitionEvidence {
    pub world_digest: u64,
    pub transition_digest: u64,
    pub action: PublicAction,
    pub candidate_commitment_digest: u64,
    pub candidate_disposition: TrialDisposition,
    pub candidate_score: ConsequenceScore,
    pub baselines: Vec<BaselineScoreRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CampaignPairingError {
    ActionNotApplied,
    MissingRealizedAction,
    MissingPostState,
    MissingTransitionIdentity,
    RequestRealizationMismatch,
    WrongWorld,
    WrongTransition,
    WrongAction,
    CopyBaselineMismatch,
    Scoring(ConsequenceScoringError),
    Custody(CustodyError),
}

/// Convert canonical realized-action evidence into the exact `StepReceipt`
/// consumed by prospective custody. This is the only EUREKA-002B adapter from
/// action realization into candidate reveal/scoring.
pub(super) fn score_frozen_after_qualified_action(
    custodian: &mut PredictionCustodian,
    frozen: FrozenPrediction,
    receipt: &QualifiedActionReceipt,
) -> Result<MatchedConsequenceTrial, CampaignPairingError> {
    if receipt.status != ActionExecutionStatus::Applied {
        return Err(CampaignPairingError::ActionNotApplied);
    }
    let realized = receipt
        .realized
        .ok_or(CampaignPairingError::MissingRealizedAction)?;
    if realized != receipt.requested {
        return Err(CampaignPairingError::RequestRealizationMismatch);
    }
    let post_state = receipt
        .post_state
        .clone()
        .ok_or(CampaignPairingError::MissingPostState)?;
    receipt
        .transition_digest
        .ok_or(CampaignPairingError::MissingTransitionIdentity)?;
    let step = StepReceipt {
        action: realized,
        observation: post_state,
    };
    let trial = custodian
        .score_after_reveal(frozen, &step)
        .map_err(CampaignPairingError::Custody)?;
    if Some(trial.transition_digest) != receipt.transition_digest {
        return Err(CampaignPairingError::WrongTransition);
    }
    Ok(trial)
}

pub(super) fn score_matched_baselines(
    fitted: &FittedShortcutBaselines,
    transition: &PublicTransitionRecord,
) -> Result<Vec<BaselineScoreRecord>, CampaignPairingError> {
    let copy = copy_current_state_baseline(&transition.pre_state, transition.action);
    let copy_score = score_consequence(&transition.pre_state, &copy, &transition.post_state)
        .map_err(CampaignPairingError::Scoring)?;
    let mut scores = vec![BaselineScoreRecord {
        identity: BaselineIdentity::CopyCurrentState,
        score: copy_score,
        fit_corpus_digest: None,
    }];

    for kind in ShortcutBaselineKind::ALL {
        let prediction = fitted.predict(kind, &transition.pre_state, transition.action);
        let score = score_consequence(
            &transition.pre_state,
            &prediction,
            &transition.post_state,
        )
        .map_err(CampaignPairingError::Scoring)?;
        scores.push(BaselineScoreRecord {
            identity: BaselineIdentity::Shortcut(kind),
            score,
            fit_corpus_digest: Some(fitted.fit_corpus_digest()),
        });
    }
    Ok(scores)
}

pub(super) fn pair_candidate_with_baselines(
    fitted: &FittedShortcutBaselines,
    transition: &PublicTransitionRecord,
    candidate: MatchedConsequenceTrial,
) -> Result<PairedTransitionEvidence, CampaignPairingError> {
    if candidate.world_digest != transition.world_digest {
        return Err(CampaignPairingError::WrongWorld);
    }
    if candidate.transition_digest != transition.transition_digest {
        return Err(CampaignPairingError::WrongTransition);
    }
    if candidate.action != transition.action {
        return Err(CampaignPairingError::WrongAction);
    }

    let baselines = score_matched_baselines(fitted, transition)?;
    let copy = baselines
        .iter()
        .find(|entry| entry.identity == BaselineIdentity::CopyCurrentState)
        .expect("copy baseline always present");
    if copy.score != candidate.copy_baseline_score {
        return Err(CampaignPairingError::CopyBaselineMismatch);
    }

    Ok(PairedTransitionEvidence {
        world_digest: transition.world_digest,
        transition_digest: transition.transition_digest,
        action: transition.action,
        candidate_commitment_digest: candidate.commitment_digest,
        candidate_disposition: TrialDisposition::from_score(candidate.candidate_score),
        candidate_score: candidate.candidate_score,
        baselines,
    })
}

/// Score an already-frozen held-out corpus with the same fitted baselines.
/// This does not manufacture candidate evidence; it only produces baseline
/// coverage/results for the exact corpus.
pub(super) fn score_held_out_baseline_corpus(
    fitted: &FittedShortcutBaselines,
    corpus: &HeldOutTransitionCorpus,
) -> Result<Vec<(u64, Vec<BaselineScoreRecord>)>, CampaignPairingError> {
    corpus
        .records()
        .iter()
        .map(|transition| {
            score_matched_baselines(fitted, transition)
                .map(|scores| (transition.transition_digest, scores))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::eureka::baselines::{BaselineFitCorpus, CorpusError};
    use crate::benchmarks::eureka::consequence::{ConsequencePrediction, PredictionOutcome};
    use crate::benchmarks::eureka::hidden_world::{
        CorpusPartition, EvaluatorWorld, FixtureFamily, PublicValue, WorldBuildProfile,
    };

    fn world(seed: u64, partition: CorpusPartition) -> EvaluatorWorld {
        EvaluatorWorld::build(WorldBuildProfile {
            family: FixtureFamily::CausalBits,
            seed,
            mechanism_variant: 0,
            partition,
        })
    }

    fn capture(
        seed: u64,
        partition: CorpusPartition,
        action: PublicAction,
    ) -> PublicTransitionRecord {
        let mut evaluator = world(seed, partition);
        let receipt = evaluator.execute_qualified_action(action);
        PublicTransitionRecord::from_qualified_receipt(
            partition,
            FixtureFamily::CausalBits,
            receipt,
        )
        .unwrap()
    }

    fn fit() -> FittedShortcutBaselines {
        let corpus = BaselineFitCorpus::freeze(vec![
            capture(1, CorpusPartition::Development, PublicAction::NoOp),
            capture(2, CorpusPartition::Calibration, PublicAction::NoOp),
            capture(
                3,
                CorpusPartition::Development,
                PublicAction::Pulse { slot: 0 },
            ),
        ])
        .unwrap();
        FittedShortcutBaselines::fit(&corpus)
    }

    #[test]
    fn qualified_action_is_the_reveal_path_for_prospective_candidate() {
        let mut evaluator = world(11, CorpusPartition::HeldOutEvaluation);
        let pre = evaluator.runtime().observe();
        let legal = evaluator.runtime().legal_actions();
        let action = PublicAction::NoOp;
        let candidate = ConsequencePrediction {
            action,
            outcome: PredictionOutcome::Predicted {
                fields: pre.fields.clone(),
            },
        };
        let mut custodian = PredictionCustodian::new(evaluator.world_digest());
        let frozen = custodian.freeze(&pre, &legal, &candidate).unwrap();
        let receipt = evaluator.execute_qualified_action(action);
        let trial = score_frozen_after_qualified_action(&mut custodian, frozen, &receipt).unwrap();
        assert_eq!(Some(trial.transition_digest), receipt.transition_digest);
    }

    #[test]
    fn candidate_and_all_baselines_bind_one_transition_identity() {
        let mut evaluator = world(11, CorpusPartition::HeldOutEvaluation);
        let pre = evaluator.runtime().observe();
        let legal = evaluator.runtime().legal_actions();
        let action = PublicAction::NoOp;
        let candidate_prediction = ConsequencePrediction {
            action,
            outcome: PredictionOutcome::Predicted {
                fields: pre.fields.clone(),
            },
        };
        let mut custodian = PredictionCustodian::new(evaluator.world_digest());
        let frozen = custodian
            .freeze(&pre, &legal, &candidate_prediction)
            .unwrap();
        let receipt = evaluator.execute_qualified_action(action);
        let trial = score_frozen_after_qualified_action(&mut custodian, frozen, &receipt).unwrap();
        let transition = PublicTransitionRecord::from_qualified_receipt(
            CorpusPartition::HeldOutEvaluation,
            FixtureFamily::CausalBits,
            receipt,
        )
        .unwrap();

        let paired = pair_candidate_with_baselines(&fit(), &transition, trial).unwrap();
        assert_eq!(paired.transition_digest, transition.transition_digest);
        assert_eq!(paired.baselines.len(), 1 + ShortcutBaselineKind::ALL.len());
        assert!(paired.baselines.iter().all(|entry| !entry.identity.stable_id().is_empty()));
    }

    #[test]
    fn baseline_abstention_remains_coverage_loss_not_candidate_victory() {
        let training = capture(1, CorpusPartition::Development, PublicAction::NoOp);
        let fitted = FittedShortcutBaselines::fit(&BaselineFitCorpus::freeze(vec![training]).unwrap());
        let held_out = capture(14, CorpusPartition::HeldOutEvaluation, PublicAction::Pulse { slot: 0 });
        let scores = score_matched_baselines(&fitted, &held_out).unwrap();
        let exact = scores
            .iter()
            .find(|entry| entry.identity == BaselineIdentity::Shortcut(ShortcutBaselineKind::ExactLookup))
            .unwrap();
        assert_eq!(exact.score, ConsequenceScore::AbstainedInsufficientEvidence);
    }

    #[test]
    fn candidate_lineage_mismatch_fails_closed() {
        let transition = capture(11, CorpusPartition::HeldOutEvaluation, PublicAction::NoOp);
        let fake = MatchedConsequenceTrial {
            world_digest: transition.world_digest ^ 1,
            commitment_digest: 1,
            transition_digest: transition.transition_digest,
            commit_sequence: 1,
            reveal_sequence: 2,
            action: transition.action,
            candidate_score: ConsequenceScore::OutOfQualifiedDomain,
            copy_baseline_score: ConsequenceScore::OutOfQualifiedDomain,
            baseline_id: COPY_CURRENT_STATE_BASELINE_ID,
        };
        assert_eq!(
            pair_candidate_with_baselines(&fit(), &transition, fake),
            Err(CampaignPairingError::WrongWorld)
        );
    }

    #[test]
    fn frozen_held_out_corpus_is_shared_by_every_baseline() {
        let a = capture(11, CorpusPartition::HeldOutEvaluation, PublicAction::NoOp);
        let b = capture(12, CorpusPartition::ExternalReplication, PublicAction::NoOp);
        let corpus = HeldOutTransitionCorpus::freeze(vec![a.clone(), b.clone()]).unwrap();
        let results = score_held_out_baseline_corpus(&fit(), &corpus).unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<u64> = results.iter().map(|(digest, _)| *digest).collect();
        assert!(ids.contains(&a.transition_digest));
        assert!(ids.contains(&b.transition_digest));
        for (_, scores) in results {
            assert_eq!(scores.len(), 1 + ShortcutBaselineKind::ALL.len());
        }
    }

    #[test]
    fn development_data_cannot_be_smuggled_into_held_out_corpus() {
        let development = capture(1, CorpusPartition::Development, PublicAction::NoOp);
        assert_eq!(
            HeldOutTransitionCorpus::freeze(vec![development]),
            Err(CorpusError::DisallowedHeldOutPartition)
        );
    }

    #[test]
    fn candidate_copy_score_must_match_independent_rescore() {
        let transition = capture(11, CorpusPartition::HeldOutEvaluation, PublicAction::NoOp);
        let fake = MatchedConsequenceTrial {
            world_digest: transition.world_digest,
            commitment_digest: 1,
            transition_digest: transition.transition_digest,
            commit_sequence: 1,
            reveal_sequence: 2,
            action: transition.action,
            candidate_score: ConsequenceScore::OutOfQualifiedDomain,
            copy_baseline_score: ConsequenceScore::OutOfQualifiedDomain,
            baseline_id: COPY_CURRENT_STATE_BASELINE_ID,
        };
        assert_eq!(
            pair_candidate_with_baselines(&fit(), &transition, fake),
            Err(CampaignPairingError::CopyBaselineMismatch)
        );
    }

    #[test]
    fn baseline_fit_digest_is_retained_on_fitted_controls() {
        let training = capture(1, CorpusPartition::Development, PublicAction::NoOp);
        let corpus = BaselineFitCorpus::freeze(vec![training]).unwrap();
        let fitted = FittedShortcutBaselines::fit(&corpus);
        let held_out = capture(11, CorpusPartition::HeldOutEvaluation, PublicAction::NoOp);
        let scores = score_matched_baselines(&fitted, &held_out).unwrap();
        assert!(scores.iter().filter(|s| s.fit_corpus_digest.is_some()).all(|s| {
            s.fit_corpus_digest == Some(corpus.digest())
        }));
    }

    #[test]
    fn no_hidden_value_is_needed_to_score_matched_baselines() {
        let training = capture(1, CorpusPartition::Development, PublicAction::NoOp);
        let fitted = FittedShortcutBaselines::fit(&BaselineFitCorpus::freeze(vec![training]).unwrap());
        let transition = PublicTransitionRecord {
            world_digest: 77,
            transition_digest: 88,
            partition: CorpusPartition::HeldOutEvaluation,
            family: FixtureFamily::CausalBits,
            pre_state: super::super::hidden_world::PublicObservation {
                step: 0,
                fields: vec![PublicValue::Bit(false)],
            },
            action: PublicAction::NoOp,
            post_state: super::super::hidden_world::PublicObservation {
                step: 1,
                fields: vec![PublicValue::Bit(true)],
            },
        };
        assert_eq!(score_matched_baselines(&fitted, &transition).unwrap().len(), 5);
    }
}
