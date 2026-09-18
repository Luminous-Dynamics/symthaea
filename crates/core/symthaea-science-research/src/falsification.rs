// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Adversarial falsification campaigns bound to frozen protocols and executions.
//!
//! Discovering a falsifier may demote or block a hypothesis. Surviving attempted
//! falsification never creates a positive authority transition. There is
//! deliberately no `Promote` authority effect.

use crate::{FramedDigest, FrozenStudyExecution, FrozenStudyProtocol, ResearchId, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FALSIFICATION_CAMPAIGN_SCHEMA: &str = "symthaea.falsification-campaign.v1";
const CAMPAIGN_DOMAIN: &str = "symthaea.falsification-campaign.identity.v1";
const REPORT_DOMAIN: &str = "symthaea.falsification-report.identity.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ChallengeKind {
    RegisteredCriterion,
    NegativeControl,
    Placebo,
    ShuffledControl,
    Ablation,
    NullModel,
    AlternativeMechanism,
    ParameterPerturbation,
    BoundaryCondition,
    DistributionShift,
    ReverseCausalDirection,
    SolverCrossCheck,
    SensitivityAnalysis,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ChallengeOrigin {
    ProtocolRegistered,
    PostHocAdversarial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ChallengeOutcome {
    Falsified,
    Survived,
    Inconclusive,
    NotExecuted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FalsificationChallenge {
    pub challenge_id: ResearchId,
    pub hypothesis_id: ResearchId,
    pub kind: ChallengeKind,
    pub origin: ChallengeOrigin,
    pub registered_falsifier_id: Option<ResearchId>,
    pub criterion_sha256: Sha256Digest,
    pub method_sha256: Sha256Digest,
    pub environment_sha256: Sha256Digest,
    pub result_artifact_sha256: Option<Sha256Digest>,
    pub outcome: ChallengeOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FalsificationCampaign {
    pub schema_version: String,
    pub campaign_id: ResearchId,
    pub subject_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
    pub execution_sha256: Sha256Digest,
    pub challenges: Vec<FalsificationChallenge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FalsificationIssue {
    WrongSchemaVersion { found: String },
    SubjectBindingMismatch,
    ProtocolBindingMismatch,
    ExecutionBindingMismatch,
    DuplicateChallenge { challenge_id: ResearchId },
    UnknownHypothesis { hypothesis_id: ResearchId },
    RegisteredChallengeMissingFalsifierId { challenge_id: ResearchId },
    PostHocChallengeCarriesRegisteredFalsifier { challenge_id: ResearchId },
    UnknownRegisteredFalsifier { falsifier_id: ResearchId },
    DuplicateRegisteredFalsifierResult { falsifier_id: ResearchId },
    RegisteredFalsifierHypothesisMismatch { falsifier_id: ResearchId },
    RegisteredCriterionMismatch { falsifier_id: ResearchId },
    ExecutedChallengeMissingResultArtifact { challenge_id: ResearchId },
}

impl FalsificationCampaign {
    pub fn validate_against(
        &self,
        protocol: &FrozenStudyProtocol,
        execution: &FrozenStudyExecution,
    ) -> Vec<FalsificationIssue> {
        let mut issues = Vec::new();
        if self.schema_version != FALSIFICATION_CAMPAIGN_SCHEMA {
            issues.push(FalsificationIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.subject_sha256 != protocol.protocol().subject_sha256
            || self.subject_sha256 != execution.execution().subject_sha256
        {
            issues.push(FalsificationIssue::SubjectBindingMismatch);
        }
        if &self.protocol_sha256 != protocol.protocol_sha256() {
            issues.push(FalsificationIssue::ProtocolBindingMismatch);
        }
        if &self.execution_sha256 != execution.execution_sha256() {
            issues.push(FalsificationIssue::ExecutionBindingMismatch);
        }

        let hypotheses = protocol
            .protocol()
            .hypotheses
            .iter()
            .map(|item| item.hypothesis_id.clone())
            .collect::<BTreeSet<_>>();
        let registered = protocol
            .protocol()
            .falsifiers
            .iter()
            .map(|item| (item.falsifier_id.clone(), item))
            .collect::<BTreeMap<_, _>>();

        let mut challenge_ids = BTreeSet::new();
        let mut consumed_registered = BTreeSet::new();
        for challenge in &self.challenges {
            if !challenge_ids.insert(challenge.challenge_id.clone()) {
                issues.push(FalsificationIssue::DuplicateChallenge {
                    challenge_id: challenge.challenge_id.clone(),
                });
            }
            if !hypotheses.contains(&challenge.hypothesis_id) {
                issues.push(FalsificationIssue::UnknownHypothesis {
                    hypothesis_id: challenge.hypothesis_id.clone(),
                });
            }
            if matches!(
                challenge.outcome,
                ChallengeOutcome::Falsified | ChallengeOutcome::Survived | ChallengeOutcome::Inconclusive
            ) && challenge.result_artifact_sha256.is_none()
            {
                issues.push(FalsificationIssue::ExecutedChallengeMissingResultArtifact {
                    challenge_id: challenge.challenge_id.clone(),
                });
            }

            match challenge.origin {
                ChallengeOrigin::ProtocolRegistered => {
                    let Some(falsifier_id) = challenge.registered_falsifier_id.as_ref() else {
                        issues.push(FalsificationIssue::RegisteredChallengeMissingFalsifierId {
                            challenge_id: challenge.challenge_id.clone(),
                        });
                        continue;
                    };
                    if !consumed_registered.insert(falsifier_id.clone()) {
                        issues.push(FalsificationIssue::DuplicateRegisteredFalsifierResult {
                            falsifier_id: falsifier_id.clone(),
                        });
                    }
                    let Some(spec) = registered.get(falsifier_id) else {
                        issues.push(FalsificationIssue::UnknownRegisteredFalsifier {
                            falsifier_id: falsifier_id.clone(),
                        });
                        continue;
                    };
                    if spec.hypothesis_id != challenge.hypothesis_id {
                        issues.push(FalsificationIssue::RegisteredFalsifierHypothesisMismatch {
                            falsifier_id: falsifier_id.clone(),
                        });
                    }
                    if spec.criterion_sha256 != challenge.criterion_sha256 {
                        issues.push(FalsificationIssue::RegisteredCriterionMismatch {
                            falsifier_id: falsifier_id.clone(),
                        });
                    }
                }
                ChallengeOrigin::PostHocAdversarial => {
                    if challenge.registered_falsifier_id.is_some() {
                        issues.push(
                            FalsificationIssue::PostHocChallengeCarriesRegisteredFalsifier {
                                challenge_id: challenge.challenge_id.clone(),
                            },
                        );
                    }
                }
            }
        }
        issues
    }

    pub fn freeze_against(
        self,
        protocol: &FrozenStudyProtocol,
        execution: &FrozenStudyExecution,
    ) -> Result<FrozenFalsificationCampaign, Vec<FalsificationIssue>> {
        let issues = self.validate_against(protocol, execution);
        if !issues.is_empty() {
            return Err(issues);
        }
        let campaign_sha256 = self.compute_digest();
        Ok(FrozenFalsificationCampaign {
            campaign: self,
            campaign_sha256,
        })
    }

    fn compute_digest(&self) -> Sha256Digest {
        let mut digest = FramedDigest::new(CAMPAIGN_DOMAIN);
        digest.text(FALSIFICATION_CAMPAIGN_SCHEMA);
        digest.text(self.campaign_id.as_str());
        digest.text(self.subject_sha256.as_str());
        digest.text(self.protocol_sha256.as_str());
        digest.text(self.execution_sha256.as_str());
        let mut challenges = self.challenges.clone();
        challenges.sort_by(|left, right| left.challenge_id.cmp(&right.challenge_id));
        for item in challenges {
            digest.text(item.challenge_id.as_str());
            digest.text(item.hypothesis_id.as_str());
            digest.text(challenge_kind_tag(item.kind));
            digest.text(challenge_origin_tag(item.origin));
            match item.registered_falsifier_id {
                Some(id) => {
                    digest.text("registered");
                    digest.text(id.as_str());
                }
                None => digest.text("unregistered"),
            }
            digest.text(item.criterion_sha256.as_str());
            digest.text(item.method_sha256.as_str());
            digest.text(item.environment_sha256.as_str());
            match item.result_artifact_sha256 {
                Some(artifact) => {
                    digest.text("result");
                    digest.text(artifact.as_str());
                }
                None => digest.text("no-result"),
            }
            digest.text(challenge_outcome_tag(item.outcome));
        }
        digest.digest()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenFalsificationCampaign {
    campaign: FalsificationCampaign,
    campaign_sha256: Sha256Digest,
}

impl FrozenFalsificationCampaign {
    pub fn campaign(&self) -> &FalsificationCampaign {
        &self.campaign
    }
    pub fn campaign_sha256(&self) -> &Sha256Digest {
        &self.campaign_sha256
    }
    pub fn report(
        &self,
        protocol: &FrozenStudyProtocol,
        execution: &FrozenStudyExecution,
    ) -> Result<FalsificationReport, Vec<FalsificationIssue>> {
        let issues = self.campaign.validate_against(protocol, execution);
        if !issues.is_empty() {
            return Err(issues);
        }
        Ok(FalsificationReport::derive(protocol, execution, self))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HypothesisChallengeStatus {
    Refuted,
    RequiredChallengesIncomplete,
    SurvivedRequiredChallenges,
    NoRequiredRegisteredChallenges,
}

/// Deliberately has no positive/promoting variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FalsificationAuthorityEffect {
    DemoteOrBlock,
    Incomplete,
    NoPromotion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FalsificationStatus {
    Refuted,
    Incomplete,
    SurvivedChallenges,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisFalsificationFinding {
    pub hypothesis_id: ResearchId,
    pub status: HypothesisChallengeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FalsificationReport {
    protocol_sha256: Sha256Digest,
    execution_sha256: Sha256Digest,
    campaign_sha256: Sha256Digest,
    status: FalsificationStatus,
    authority_effect: FalsificationAuthorityEffect,
    hypotheses: Vec<HypothesisFalsificationFinding>,
    report_sha256: Sha256Digest,
}

impl FalsificationReport {
    fn derive(
        protocol: &FrozenStudyProtocol,
        execution: &FrozenStudyExecution,
        campaign: &FrozenFalsificationCampaign,
    ) -> Self {
        let mut findings = Vec::new();
        let mut any_refuted = false;
        let mut any_incomplete = false;
        let by_hypothesis = campaign
            .campaign
            .challenges
            .iter()
            .fold(BTreeMap::<ResearchId, Vec<&FalsificationChallenge>>::new(), |mut map, item| {
                map.entry(item.hypothesis_id.clone()).or_default().push(item);
                map
            });

        for hypothesis in &protocol.protocol().hypotheses {
            let challenges = by_hypothesis
                .get(&hypothesis.hypothesis_id)
                .cloned()
                .unwrap_or_default();
            let refuted = challenges
                .iter()
                .any(|item| item.outcome == ChallengeOutcome::Falsified);
            let required_ids = protocol
                .protocol()
                .falsifiers
                .iter()
                .filter(|item| item.required && item.hypothesis_id == hypothesis.hypothesis_id)
                .map(|item| item.falsifier_id.clone())
                .collect::<BTreeSet<_>>();
            let registered_results = challenges
                .iter()
                .filter_map(|item| {
                    (item.origin == ChallengeOrigin::ProtocolRegistered)
                        .then(|| item.registered_falsifier_id.as_ref().map(|id| (id.clone(), item.outcome)))
                        .flatten()
                })
                .collect::<BTreeMap<_, _>>();
            let complete = required_ids.iter().all(|id| {
                matches!(
                    registered_results.get(id).copied(),
                    Some(ChallengeOutcome::Falsified | ChallengeOutcome::Survived)
                )
            });

            let status = if refuted {
                any_refuted = true;
                HypothesisChallengeStatus::Refuted
            } else if required_ids.is_empty() {
                HypothesisChallengeStatus::NoRequiredRegisteredChallenges
            } else if !complete {
                any_incomplete = true;
                HypothesisChallengeStatus::RequiredChallengesIncomplete
            } else {
                HypothesisChallengeStatus::SurvivedRequiredChallenges
            };
            findings.push(HypothesisFalsificationFinding {
                hypothesis_id: hypothesis.hypothesis_id.clone(),
                status,
            });
        }

        findings.sort_by(|left, right| left.hypothesis_id.cmp(&right.hypothesis_id));
        let (status, authority_effect) = if any_refuted {
            (
                FalsificationStatus::Refuted,
                FalsificationAuthorityEffect::DemoteOrBlock,
            )
        } else if any_incomplete {
            (
                FalsificationStatus::Incomplete,
                FalsificationAuthorityEffect::Incomplete,
            )
        } else {
            (
                FalsificationStatus::SurvivedChallenges,
                FalsificationAuthorityEffect::NoPromotion,
            )
        };
        let report_sha256 = report_digest(
            protocol.protocol_sha256(),
            execution.execution_sha256(),
            campaign.campaign_sha256(),
            status,
            authority_effect,
            &findings,
        );
        Self {
            protocol_sha256: protocol.protocol_sha256().clone(),
            execution_sha256: execution.execution_sha256().clone(),
            campaign_sha256: campaign.campaign_sha256().clone(),
            status,
            authority_effect,
            hypotheses: findings,
            report_sha256,
        }
    }

    pub fn status(&self) -> FalsificationStatus {
        self.status
    }
    pub fn authority_effect(&self) -> FalsificationAuthorityEffect {
        self.authority_effect
    }
    pub fn hypotheses(&self) -> &[HypothesisFalsificationFinding] {
        &self.hypotheses
    }
    pub fn report_sha256(&self) -> &Sha256Digest {
        &self.report_sha256
    }
}

fn report_digest(
    protocol: &Sha256Digest,
    execution: &Sha256Digest,
    campaign: &Sha256Digest,
    status: FalsificationStatus,
    effect: FalsificationAuthorityEffect,
    findings: &[HypothesisFalsificationFinding],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(REPORT_DOMAIN);
    digest.text(protocol.as_str());
    digest.text(execution.as_str());
    digest.text(campaign.as_str());
    digest.text(falsification_status_tag(status));
    digest.text(authority_effect_tag(effect));
    for item in findings {
        digest.text(item.hypothesis_id.as_str());
        digest.text(hypothesis_status_tag(item.status));
    }
    digest.digest()
}

const fn challenge_kind_tag(value: ChallengeKind) -> &'static str {
    match value {
        ChallengeKind::RegisteredCriterion => "registered-criterion",
        ChallengeKind::NegativeControl => "negative-control",
        ChallengeKind::Placebo => "placebo",
        ChallengeKind::ShuffledControl => "shuffled-control",
        ChallengeKind::Ablation => "ablation",
        ChallengeKind::NullModel => "null-model",
        ChallengeKind::AlternativeMechanism => "alternative-mechanism",
        ChallengeKind::ParameterPerturbation => "parameter-perturbation",
        ChallengeKind::BoundaryCondition => "boundary-condition",
        ChallengeKind::DistributionShift => "distribution-shift",
        ChallengeKind::ReverseCausalDirection => "reverse-causal-direction",
        ChallengeKind::SolverCrossCheck => "solver-cross-check",
        ChallengeKind::SensitivityAnalysis => "sensitivity-analysis",
        ChallengeKind::Other => "other",
    }
}
const fn challenge_origin_tag(value: ChallengeOrigin) -> &'static str {
    match value {
        ChallengeOrigin::ProtocolRegistered => "protocol-registered",
        ChallengeOrigin::PostHocAdversarial => "post-hoc-adversarial",
    }
}
const fn challenge_outcome_tag(value: ChallengeOutcome) -> &'static str {
    match value {
        ChallengeOutcome::Falsified => "falsified",
        ChallengeOutcome::Survived => "survived",
        ChallengeOutcome::Inconclusive => "inconclusive",
        ChallengeOutcome::NotExecuted => "not-executed",
    }
}
const fn hypothesis_status_tag(value: HypothesisChallengeStatus) -> &'static str {
    match value {
        HypothesisChallengeStatus::Refuted => "refuted",
        HypothesisChallengeStatus::RequiredChallengesIncomplete => "required-challenges-incomplete",
        HypothesisChallengeStatus::SurvivedRequiredChallenges => "survived-required-challenges",
        HypothesisChallengeStatus::NoRequiredRegisteredChallenges => "no-required-registered-challenges",
    }
}
const fn falsification_status_tag(value: FalsificationStatus) -> &'static str {
    match value {
        FalsificationStatus::Refuted => "refuted",
        FalsificationStatus::Incomplete => "incomplete",
        FalsificationStatus::SurvivedChallenges => "survived-challenges",
    }
}
const fn authority_effect_tag(value: FalsificationAuthorityEffect) -> &'static str {
    match value {
        FalsificationAuthorityEffect::DemoteOrBlock => "demote-or-block",
        FalsificationAuthorityEffect::Incomplete => "incomplete",
        FalsificationAuthorityEffect::NoPromotion => "no-promotion",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnalysisPlan, ControlKind, ControlSpec, ExecutedControl, ExecutedOutcome, FalsifierSpec,
        HypothesisRole, HypothesisSpec, MultiplicityPolicy, OutcomeRole, OutcomeSpec,
        PredictionDirection, PredictionSpec, SamplingPlan, StudyExecution, StudyIntent,
        StudyProtocol, STUDY_EXECUTION_SCHEMA, STUDY_PROTOCOL_SCHEMA,
    };

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }
    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn protocol() -> FrozenStudyProtocol {
        StudyProtocol {
            schema_version: STUDY_PROTOCOL_SCHEMA.into(),
            protocol_id: id("SCI-FALSIFY-PROTOCOL"),
            subject_sha256: sha("subject"),
            intent: StudyIntent::Confirmatory,
            registered_at_unix_ms: Some(100),
            registration_evidence_sha256: Some(sha("registration")),
            planned_protected_observation_start_unix_ms: Some(200),
            hypotheses: vec![HypothesisSpec {
                hypothesis_id: id("H-1"),
                role: HypothesisRole::Primary,
                statement_sha256: sha("h1"),
                null_statement_sha256: Some(sha("null")),
                predictions: vec![PredictionSpec {
                    outcome_id: id("Y-1"),
                    direction: PredictionDirection::Increase,
                    quantitative_envelope_sha256: None,
                }],
            }],
            outcomes: vec![OutcomeSpec {
                outcome_id: id("Y-1"),
                role: OutcomeRole::Primary,
                measurement_sha256: sha("measure"),
            }],
            controls: vec![ControlSpec {
                control_id: id("C-1"),
                kind: ControlKind::Negative,
                protocol_sha256: sha("control"),
            }],
            falsifiers: vec![FalsifierSpec {
                falsifier_id: id("F-1"),
                hypothesis_id: id("H-1"),
                criterion_sha256: sha("criterion"),
                required: true,
            }],
            analysis: AnalysisPlan {
                analysis_plan_sha256: sha("analysis"),
                estimator_or_test_sha256: sha("estimator"),
                code_sha256: Some(sha("code")),
                environment_sha256: Some(sha("env")),
                multiplicity_policy: MultiplicityPolicy::NotApplicable,
                multiplicity_policy_sha256: None,
            },
            sampling: SamplingPlan {
                target_units: Some(100),
                recruitment_or_generation_sha256: sha("sampling"),
                exclusion_rule_sha256: sha("exclusions"),
                stopping_rule_sha256: sha("stopping"),
            },
            deviation_policy_sha256: sha("deviation-policy"),
            supersedes_protocol_sha256: None,
        }
        .freeze()
        .unwrap()
    }

    fn execution(protocol: &FrozenStudyProtocol) -> FrozenStudyExecution {
        StudyExecution {
            schema_version: STUDY_EXECUTION_SCHEMA.into(),
            execution_id: id("SCI-FALSIFY-EXEC"),
            protocol_sha256: protocol.protocol_sha256().clone(),
            subject_sha256: protocol.protocol().subject_sha256.clone(),
            first_protected_observation_access_unix_ms: 200,
            collection_started_unix_ms: 200,
            collection_ended_unix_ms: 300,
            analysis_started_unix_ms: 301,
            analysis_ended_unix_ms: 400,
            data_snapshot_sha256: sha("data"),
            units_started: 100,
            units_completed: 95,
            units_excluded_after_observation: 0,
            recruitment_or_generation_sha256: sha("sampling"),
            exclusion_rule_sha256: sha("exclusions"),
            stopping_rule_sha256: sha("stopping"),
            analysis_plan_sha256: sha("analysis"),
            estimator_or_test_sha256: sha("estimator"),
            code_sha256: sha("code"),
            environment_sha256: sha("env"),
            multiplicity_policy: MultiplicityPolicy::NotApplicable,
            multiplicity_policy_sha256: None,
            outcomes: vec![ExecutedOutcome {
                outcome_id: id("Y-1"),
                artifact_sha256: sha("y1"),
            }],
            controls: vec![ExecutedControl {
                control_id: id("C-1"),
                artifact_sha256: sha("c1"),
            }],
            declared_deviations: Vec::new(),
        }
        .freeze_against(protocol)
        .unwrap()
    }

    fn campaign(
        protocol: &FrozenStudyProtocol,
        execution: &FrozenStudyExecution,
        outcome: ChallengeOutcome,
    ) -> FalsificationCampaign {
        FalsificationCampaign {
            schema_version: FALSIFICATION_CAMPAIGN_SCHEMA.into(),
            campaign_id: id("FC-1"),
            subject_sha256: protocol.protocol().subject_sha256.clone(),
            protocol_sha256: protocol.protocol_sha256().clone(),
            execution_sha256: execution.execution_sha256().clone(),
            challenges: vec![FalsificationChallenge {
                challenge_id: id("CH-1"),
                hypothesis_id: id("H-1"),
                kind: ChallengeKind::RegisteredCriterion,
                origin: ChallengeOrigin::ProtocolRegistered,
                registered_falsifier_id: Some(id("F-1")),
                criterion_sha256: sha("criterion"),
                method_sha256: sha("method"),
                environment_sha256: sha("challenge-env"),
                result_artifact_sha256: match outcome {
                    ChallengeOutcome::NotExecuted => None,
                    _ => Some(sha("challenge-result")),
                },
                outcome,
            }],
        }
    }

    #[test]
    fn surviving_required_challenges_never_promotes() {
        let plan = protocol();
        let execution = execution(&plan);
        let report = campaign(&plan, &execution, ChallengeOutcome::Survived)
            .freeze_against(&plan, &execution)
            .unwrap()
            .report(&plan, &execution)
            .unwrap();
        assert_eq!(report.status(), FalsificationStatus::SurvivedChallenges);
        assert_eq!(report.authority_effect(), FalsificationAuthorityEffect::NoPromotion);
    }

    #[test]
    fn triggered_registered_falsifier_demotes_or_blocks() {
        let plan = protocol();
        let execution = execution(&plan);
        let report = campaign(&plan, &execution, ChallengeOutcome::Falsified)
            .freeze_against(&plan, &execution)
            .unwrap()
            .report(&plan, &execution)
            .unwrap();
        assert_eq!(report.status(), FalsificationStatus::Refuted);
        assert_eq!(report.authority_effect(), FalsificationAuthorityEffect::DemoteOrBlock);
    }

    #[test]
    fn missing_required_challenge_is_incomplete() {
        let plan = protocol();
        let execution = execution(&plan);
        let mut draft = campaign(&plan, &execution, ChallengeOutcome::Survived);
        draft.challenges.clear();
        let report = draft
            .freeze_against(&plan, &execution)
            .unwrap()
            .report(&plan, &execution)
            .unwrap();
        assert_eq!(report.status(), FalsificationStatus::Incomplete);
    }

    #[test]
    fn registered_criterion_substitution_fails_closed() {
        let plan = protocol();
        let execution = execution(&plan);
        let mut draft = campaign(&plan, &execution, ChallengeOutcome::Survived);
        draft.challenges[0].criterion_sha256 = sha("different-criterion");
        assert!(draft
            .freeze_against(&plan, &execution)
            .unwrap_err()
            .iter()
            .any(|issue| matches!(issue, FalsificationIssue::RegisteredCriterionMismatch { .. })));
    }

    #[test]
    fn registered_falsifier_cannot_be_consumed_twice() {
        let plan = protocol();
        let execution = execution(&plan);
        let mut draft = campaign(&plan, &execution, ChallengeOutcome::Survived);
        let mut duplicate = draft.challenges[0].clone();
        duplicate.challenge_id = id("CH-DUP");
        duplicate.outcome = ChallengeOutcome::Inconclusive;
        draft.challenges.push(duplicate);
        assert!(draft
            .freeze_against(&plan, &execution)
            .unwrap_err()
            .iter()
            .any(|issue| matches!(issue, FalsificationIssue::DuplicateRegisteredFalsifierResult { .. })));
    }

    #[test]
    fn post_hoc_counterexample_can_refute_without_becoming_registered() {
        let plan = protocol();
        let execution = execution(&plan);
        let mut draft = campaign(&plan, &execution, ChallengeOutcome::Survived);
        draft.challenges.push(FalsificationChallenge {
            challenge_id: id("CH-POST"),
            hypothesis_id: id("H-1"),
            kind: ChallengeKind::BoundaryCondition,
            origin: ChallengeOrigin::PostHocAdversarial,
            registered_falsifier_id: None,
            criterion_sha256: sha("posthoc-criterion"),
            method_sha256: sha("posthoc-method"),
            environment_sha256: sha("posthoc-env"),
            result_artifact_sha256: Some(sha("counterexample")),
            outcome: ChallengeOutcome::Falsified,
        });
        let report = draft
            .freeze_against(&plan, &execution)
            .unwrap()
            .report(&plan, &execution)
            .unwrap();
        assert_eq!(report.status(), FalsificationStatus::Refuted);
    }

    #[test]
    fn campaign_identity_is_order_independent() {
        let plan = protocol();
        let execution = execution(&plan);
        let mut left = campaign(&plan, &execution, ChallengeOutcome::Survived);
        left.challenges.push(FalsificationChallenge {
            challenge_id: id("CH-2"),
            hypothesis_id: id("H-1"),
            kind: ChallengeKind::Ablation,
            origin: ChallengeOrigin::PostHocAdversarial,
            registered_falsifier_id: None,
            criterion_sha256: sha("ablation-criterion"),
            method_sha256: sha("ablation-method"),
            environment_sha256: sha("ablation-env"),
            result_artifact_sha256: Some(sha("ablation-result")),
            outcome: ChallengeOutcome::Survived,
        });
        let mut right = left.clone();
        right.challenges.reverse();
        assert_eq!(
            left.freeze_against(&plan, &execution).unwrap().campaign_sha256(),
            right.freeze_against(&plan, &execution).unwrap().campaign_sha256()
        );
    }
}
