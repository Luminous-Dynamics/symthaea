// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Operationally constrained, replayable sequential-design policy primitives.
//!
//! This module deliberately accepts only experimenter-controlled actions.
//! It has no field for participant confidence criteria, latent evidence
//! thresholds, or other observer states. Candidate assessments are supplied
//! by a separately qualified model layer; this module does not calculate
//! Fisher information, Bayesian EIG, model-discrepancy scores, or posteriors.
//!
//! The first policy implementation provides:
//! - hard operational and model-adequacy eligibility gates;
//! - minimum representativeness / model-discrimination constraints;
//! - forced randomized audit-stratum coverage;
//! - randomized exploration plus robust-utility exploitation;
//! - marginal action-selection propensities and deterministic receipts.
//!
//! SplitMix64 is for reproducible research policy simulation, not
//! cryptographic randomness and not deployment-grade random assignment.

use std::collections::BTreeMap;
use std::fmt;

/// The stimulus class is an experimenter-controlled assignment, not an
/// inferred observer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StimulusClass {
    Signal,
    Noise,
}

/// One allowed trial-level experimenter action.
///
/// Notice what is intentionally absent: no confidence criterion, evidence
/// threshold, internal response bias, or participant speed-accuracy setting
/// can be requested through this type.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationalAction {
    pub action_id: u32,
    pub audit_stratum: u16,
    pub condition_id: u16,
    pub stimulus_class: StimulusClass,
    /// The experimenter-controlled stimulus level in the caller's declared
    /// units (for example, log contrast or calibrated intensity).
    pub stimulus_level: f64,
    pub response_scale_id: u16,
    /// Optional, experimenter-controlled acquisition deadline.
    pub rt_deadline_ms: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionSpaceError {
    Empty,
    DuplicateActionId,
    NonFiniteStimulusLevel,
    ZeroResponseDeadline,
    MissingActionSetDigest,
}

impl fmt::Display for ActionSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Empty => "operational action space must not be empty",
            Self::DuplicateActionId => "operational action IDs must be unique",
            Self::NonFiniteStimulusLevel => "stimulus levels must be finite",
            Self::ZeroResponseDeadline => "response deadline must be positive when supplied",
            Self::MissingActionSetDigest => "action-set digest must not be empty",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ActionSpaceError {}

/// Frozen, canonically ordered set of allowed experimenter actions.
///
/// The digest is supplied by the caller from its canonical serialization.
/// This module treats it as an opaque commitment and does not claim to hash
/// or authenticate it.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationalActionSet {
    actions: Vec<OperationalAction>,
    digest: String,
}

impl OperationalActionSet {
    pub fn new(
        mut actions: Vec<OperationalAction>,
        digest: impl Into<String>,
    ) -> Result<Self, ActionSpaceError> {
        if actions.is_empty() {
            return Err(ActionSpaceError::Empty);
        }
        let digest = digest.into();
        if digest.trim().is_empty() {
            return Err(ActionSpaceError::MissingActionSetDigest);
        }
        for action in &actions {
            if !action.stimulus_level.is_finite() {
                return Err(ActionSpaceError::NonFiniteStimulusLevel);
            }
            if action.rt_deadline_ms == Some(0) {
                return Err(ActionSpaceError::ZeroResponseDeadline);
            }
        }
        actions.sort_by_key(|action| action.action_id);
        if actions
            .windows(2)
            .any(|pair| pair[0].action_id == pair[1].action_id)
        {
            return Err(ActionSpaceError::DuplicateActionId);
        }
        Ok(Self { actions, digest })
    }

    pub fn actions(&self) -> &[OperationalAction] {
        &self.actions
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    fn action(&self, action_id: u32) -> Option<&OperationalAction> {
        self.actions
            .binary_search_by_key(&action_id, |action| action.action_id)
            .ok()
            .map(|index| &self.actions[index])
    }
}

/// Model-layer assessment for one operational action.
///
/// Values are optional so an unqualified action can remain explicitly
/// represented without inventing scores. An eligible action must provide all
/// five values. worst_case_target_utility is a caller-computed robust utility
/// over a frozen ambiguity/scenario set; this module does not compute it.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateAssessment {
    pub action_id: u32,
    pub operational_mapping_qualified: bool,
    pub model_adequacy_qualified: bool,
    pub target_information: Option<f64>,
    pub model_discrimination_information: Option<f64>,
    /// A declared [0, 1] representativeness score.
    pub representativeness: Option<f64>,
    /// The selector consumes a maximized, finite worst-case utility.
    pub worst_case_target_utility: Option<f64>,
    pub participant_burden_seconds: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryError {
    MissingHistoryDigest,
    UnknownActionId,
    CountOverflow,
}

impl fmt::Display for HistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingHistoryDigest => "history digest must not be empty",
            Self::UnknownActionId => "history contains an action outside the frozen action set",
            Self::CountOverflow => "history action counts overflowed u64",
        };
        f.write_str(message)
    }
}

impl std::error::Error for HistoryError {}

/// Sufficient action-allocation summary from the complete recorded history.
///
/// digest must commit to the full ordered history, not merely this count
/// summary. This type does not compute or verify that digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyHistory {
    pub action_counts: BTreeMap<u32, u64>,
    pub digest: String,
}

impl PolicyHistory {
    pub fn new(
        action_counts: BTreeMap<u32, u64>,
        digest: impl Into<String>,
        action_set: &OperationalActionSet,
    ) -> Result<Self, HistoryError> {
        let digest = digest.into();
        if digest.trim().is_empty() {
            return Err(HistoryError::MissingHistoryDigest);
        }
        for action_id in action_counts.keys() {
            if action_set.action(*action_id).is_none() {
                return Err(HistoryError::UnknownActionId);
            }
        }
        let mut total = 0u64;
        for count in action_counts.values() {
            total = total.checked_add(*count).ok_or(HistoryError::CountOverflow)?;
        }
        let _ = total;
        Ok(Self { action_counts, digest })
    }

    fn count_for_action(&self, action_id: u32) -> u64 {
        self.action_counts.get(&action_id).copied().unwrap_or(0)
    }

    fn count_for_stratum(
        &self,
        action_set: &OperationalActionSet,
        stratum: u16,
    ) -> Result<u64, HistoryError> {
        let mut total = 0u64;
        for action in action_set.actions() {
            if action.audit_stratum == stratum {
                total = total
                    .checked_add(self.count_for_action(action.action_id))
                    .ok_or(HistoryError::CountOverflow)?;
            }
        }
        Ok(total)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SequentialPolicyConfig {
    pub policy_version: String,
    /// Frozen identifier for the candidate generative/observation model family.
    pub model_set_version: String,
    /// Caller-computed digest of the complete canonical assessment vector.
    pub assessment_set_digest: String,
    pub seed: u64,
    pub decision_index: u64,
    /// Marginal exploration probability outside forced audit decisions.
    pub exploration_probability: f64,
    pub minimum_representativeness: f64,
    pub minimum_model_discrimination_information: Option<f64>,
    /// Minimum completed trials required in each listed audit stratum.
    pub audit_floor_by_stratum: BTreeMap<u16, u64>,
    /// Optional remaining participant-time budget for the next action.
    pub remaining_participant_time_seconds: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    ForcedAudit,
    Exploration,
    Exploitation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExclusionReason {
    OperationalMappingUnqualified,
    ModelAdequacyUnqualified,
    IncompleteAssessment,
    BelowRepresentativenessFloor,
    BelowModelDiscriminationFloor,
    ExceedsRemainingParticipantTime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExcludedAction {
    pub action_id: u32,
    pub reason: ExclusionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionError {
    MissingPolicyVersion,
    MissingModelSetVersion,
    MissingAssessmentSetDigest,
    InvalidExplorationProbability,
    InvalidRepresentativenessFloor,
    InvalidModelDiscriminationFloor,
    InvalidTimeBudget,
    AssessmentCountMismatch,
    DuplicateAssessment,
    UnknownAssessmentAction,
    InvalidAssessmentValue,
    AuditStratumMissingFromActionSet,
    AuditStratumHasNoQualifiedAction,
    NoQualifiedAction,
    History(HistoryError),
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingPolicyVersion => "policy version must not be empty",
            Self::MissingModelSetVersion => "model-set version must not be empty",
            Self::MissingAssessmentSetDigest => "assessment-set digest must not be empty",
            Self::InvalidExplorationProbability => "exploration probability must be in [0, 1]",
            Self::InvalidRepresentativenessFloor => "representativeness floor must be in [0, 1]",
            Self::InvalidModelDiscriminationFloor => {
                "model-discrimination floor must be finite and nonnegative"
            }
            Self::InvalidTimeBudget => "remaining participant-time budget must be finite and nonnegative",
            Self::AssessmentCountMismatch => "exactly one assessment is required for each action",
            Self::DuplicateAssessment => "duplicate action assessment",
            Self::UnknownAssessmentAction => "assessment references an unknown action",
            Self::InvalidAssessmentValue => "assessment values must satisfy their declared domains",
            Self::AuditStratumMissingFromActionSet => {
                "an audit-floor stratum is absent from the frozen action set"
            }
            Self::AuditStratumHasNoQualifiedAction => {
                "an audit floor is due but its stratum has no qualified action"
            }
            Self::NoQualifiedAction => "no operationally and statistically qualified action remains",
            Self::History(error) => return write!(f, "invalid policy history: {error}"),
        };
        f.write_str(message)
    }
}

impl std::error::Error for SelectionError {}

impl From<HistoryError> for SelectionError {
    fn from(value: HistoryError) -> Self {
        Self::History(value)
    }
}

/// Marginal propensity for one eligible action under the selected policy.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionPropensity {
    pub action_id: u32,
    pub probability: f64,
}

/// Replay record for a single sequential action decision.
///
/// selection_probability is the marginal probability of the selected action
/// under the configured randomized policy, not merely the realized branch
/// probability.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyReceipt {
    pub policy_version: String,
    pub model_set_version: String,
    pub assessment_set_digest: String,
    pub action_set_digest: String,
    pub history_digest: String,
    pub policy_config: SequentialPolicyConfig,
    pub candidate_assessments: Vec<CandidateAssessment>,
    pub action_propensities: Vec<ActionPropensity>,
    pub random_generator: &'static str,
    pub seed: u64,
    pub decision_index: u64,
    pub branch_random_u64: Option<u64>,
    pub candidate_random_u64: Option<u64>,
    pub selection_mode: SelectionMode,
    pub eligible_action_ids: Vec<u32>,
    pub excluded_actions: Vec<ExcludedAction>,
    pub selected_action: OperationalAction,
    pub selection_probability: f64,
    pub selected_target_information: f64,
    pub selected_model_discrimination_information: f64,
    pub selected_representativeness: f64,
    pub selected_worst_case_target_utility: f64,
    pub selected_participant_burden_seconds: f64,
    pub forced_audit_stratum: Option<u16>,
}

/// Selected action plus its immutable replay receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionResult {
    pub action: OperationalAction,
    pub receipt: PolicyReceipt,
}

#[derive(Debug, Clone, Copy)]
struct EligibleCandidate<'a> {
    action: &'a OperationalAction,
    assessment: &'a CandidateAssessment,
}

impl<'a> EligibleCandidate<'a> {
    fn action_id(self) -> u32 {
        self.action.action_id
    }

    fn utility(self) -> f64 {
        self.assessment
            .worst_case_target_utility
            .expect("qualified assessment has a utility")
    }
}

/// Deterministically select the next operational action.
///
/// Forced audit coverage takes precedence over exploration/exploitation.
/// Otherwise, the policy explores uniformly with the configured probability
/// and exploits the action with the highest supplied worst-case utility.
/// Ties in exploitation are resolved by the lowest action ID.
pub fn select_next_action(
    action_set: &OperationalActionSet,
    assessments: &[CandidateAssessment],
    history: &PolicyHistory,
    config: &SequentialPolicyConfig,
) -> Result<SelectionResult, SelectionError> {
    validate_config(config)?;
    validate_history(action_set, history)?;

    if assessments.len() != action_set.actions().len() {
        return Err(SelectionError::AssessmentCountMismatch);
    }

    let mut assessment_by_id = BTreeMap::new();
    for assessment in assessments {
        if action_set.action(assessment.action_id).is_none() {
            return Err(SelectionError::UnknownAssessmentAction);
        }
        if assessment_by_id
            .insert(assessment.action_id, assessment)
            .is_some()
        {
            return Err(SelectionError::DuplicateAssessment);
        }
        validate_assessment_values(assessment)?;
    }

    let mut canonical_assessments = assessments.to_vec();
    canonical_assessments.sort_by_key(|assessment| assessment.action_id);

    let mut eligible = Vec::new();
    let mut excluded = Vec::new();
    for action in action_set.actions() {
        let assessment = assessment_by_id
            .get(&action.action_id)
            .copied()
            .ok_or(SelectionError::AssessmentCountMismatch)?;
        let reason = exclusion_reason(assessment, config);
        if let Some(reason) = reason {
            excluded.push(ExcludedAction {
                action_id: action.action_id,
                reason,
            });
        } else {
            eligible.push(EligibleCandidate { action, assessment });
        }
    }
    if eligible.is_empty() {
        return Err(SelectionError::NoQualifiedAction);
    }

    let mut rng = SplitMix64::new(derive_seed(config.seed, config.decision_index));

    if let Some(stratum) = due_audit_stratum(action_set, history, config)? {
        let audit_candidates: Vec<_> = eligible
            .iter()
            .copied()
            .filter(|candidate| candidate.action.audit_stratum == stratum)
            .collect();
        if audit_candidates.is_empty() {
            return Err(SelectionError::AuditStratumHasNoQualifiedAction);
        }
        let (index, draw) = rng.uniform_index(audit_candidates.len());
        let selected = audit_candidates[index];
        let audit_probability = 1.0 / audit_candidates.len() as f64;
        let propensities = eligible
            .iter()
            .map(|candidate| ActionPropensity {
                action_id: candidate.action_id(),
                probability: if candidate.action.audit_stratum == stratum {
                    audit_probability
                } else {
                    0.0
                },
            })
            .collect();
        return Ok(build_result(
            selected,
            action_set,
            history,
            config,
            &eligible,
            &canonical_assessments,
            excluded,
            SelectionMode::ForcedAudit,
            propensities,
            None,
            Some(draw),
            Some(stratum),
        ));
    }

    let exploitation_winner = best_candidate(&eligible);
    let (branch_draw, branch_value) = rng.next_unit_interval();
    let explore = branch_value < config.exploration_probability;

    let (selected, mode, candidate_draw) = if explore {
        let (index, draw) = rng.uniform_index(eligible.len());
        (eligible[index], SelectionMode::Exploration, Some(draw))
    } else {
        (exploitation_winner, SelectionMode::Exploitation, None)
    };

    let random_mass = config.exploration_probability / eligible.len() as f64;
    let propensities = eligible
        .iter()
        .map(|candidate| ActionPropensity {
            action_id: candidate.action_id(),
            probability: random_mass
                + if candidate.action_id() == exploitation_winner.action_id() {
                    1.0 - config.exploration_probability
                } else {
                    0.0
                },
        })
        .collect();

    Ok(build_result(
        selected,
        action_set,
        history,
        config,
        &eligible,
        &canonical_assessments,
        excluded,
        mode,
        propensities,
        Some(branch_draw),
        candidate_draw,
        None,
    ))
}

fn validate_config(config: &SequentialPolicyConfig) -> Result<(), SelectionError> {
    if config.policy_version.trim().is_empty() {
        return Err(SelectionError::MissingPolicyVersion);
    }
    if config.model_set_version.trim().is_empty() {
        return Err(SelectionError::MissingModelSetVersion);
    }
    if config.assessment_set_digest.trim().is_empty() {
        return Err(SelectionError::MissingAssessmentSetDigest);
    }
    if !config.exploration_probability.is_finite()
        || !(0.0..=1.0).contains(&config.exploration_probability)
    {
        return Err(SelectionError::InvalidExplorationProbability);
    }
    if !config.minimum_representativeness.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_representativeness)
    {
        return Err(SelectionError::InvalidRepresentativenessFloor);
    }
    if let Some(floor) = config.minimum_model_discrimination_information {
        if !floor.is_finite() || floor < 0.0 {
            return Err(SelectionError::InvalidModelDiscriminationFloor);
        }
    }
    if let Some(budget) = config.remaining_participant_time_seconds {
        if !budget.is_finite() || budget < 0.0 {
            return Err(SelectionError::InvalidTimeBudget);
        }
    }
    Ok(())
}

fn validate_history(
    action_set: &OperationalActionSet,
    history: &PolicyHistory,
) -> Result<(), SelectionError> {
    if history.digest.trim().is_empty() {
        return Err(SelectionError::History(HistoryError::MissingHistoryDigest));
    }
    let mut total = 0u64;
    for (action_id, count) in &history.action_counts {
        if action_set.action(*action_id).is_none() {
            return Err(SelectionError::History(HistoryError::UnknownActionId));
        }
        total = total
            .checked_add(*count)
            .ok_or(SelectionError::History(HistoryError::CountOverflow))?;
    }
    let _ = total;
    Ok(())
}

fn validate_assessment_values(
    assessment: &CandidateAssessment,
) -> Result<(), SelectionError> {
    for value in [
        assessment.target_information,
        assessment.model_discrimination_information,
        assessment.representativeness,
        assessment.worst_case_target_utility,
        assessment.participant_burden_seconds,
    ]
    .into_iter()
    .flatten()
    {
        if !value.is_finite() {
            return Err(SelectionError::InvalidAssessmentValue);
        }
    }
    if assessment.target_information.is_some_and(|value| value < 0.0)
        || assessment
            .model_discrimination_information
            .is_some_and(|value| value < 0.0)
        || assessment
            .representativeness
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        || assessment
            .participant_burden_seconds
            .is_some_and(|value| value < 0.0)
    {
        return Err(SelectionError::InvalidAssessmentValue);
    }
    Ok(())
}

fn exclusion_reason(
    assessment: &CandidateAssessment,
    config: &SequentialPolicyConfig,
) -> Option<ExclusionReason> {
    if !assessment.operational_mapping_qualified {
        return Some(ExclusionReason::OperationalMappingUnqualified);
    }
    if !assessment.model_adequacy_qualified {
        return Some(ExclusionReason::ModelAdequacyUnqualified);
    }
    let (
        Some(target_information),
        Some(model_discrimination),
        Some(representativeness),
        Some(worst_case_utility),
        Some(burden),
    ) = (
        assessment.target_information,
        assessment.model_discrimination_information,
        assessment.representativeness,
        assessment.worst_case_target_utility,
        assessment.participant_burden_seconds,
    )
    else {
        return Some(ExclusionReason::IncompleteAssessment);
    };
    let _ = (target_information, worst_case_utility);
    if representativeness < config.minimum_representativeness {
        return Some(ExclusionReason::BelowRepresentativenessFloor);
    }
    if config
        .minimum_model_discrimination_information
        .is_some_and(|floor| model_discrimination < floor)
    {
        return Some(ExclusionReason::BelowModelDiscriminationFloor);
    }
    if config
        .remaining_participant_time_seconds
        .is_some_and(|budget| burden > budget)
    {
        return Some(ExclusionReason::ExceedsRemainingParticipantTime);
    }
    None
}

fn due_audit_stratum(
    action_set: &OperationalActionSet,
    history: &PolicyHistory,
    config: &SequentialPolicyConfig,
) -> Result<Option<u16>, SelectionError> {
    let mut due: Option<(u16, f64)> = None;
    for (&stratum, &floor) in &config.audit_floor_by_stratum {
        if floor == 0 {
            continue;
        }
        if !action_set
            .actions()
            .iter()
            .any(|action| action.audit_stratum == stratum)
        {
            return Err(SelectionError::AuditStratumMissingFromActionSet);
        }
        let completed = history.count_for_stratum(action_set, stratum)?;
        if completed >= floor {
            continue;
        }
        let coverage = completed as f64 / floor as f64;
        match due {
            None => due = Some((stratum, coverage)),
            Some((best_stratum, best_coverage)) => {
                if coverage < best_coverage
                    || (coverage == best_coverage && stratum < best_stratum)
                {
                    due = Some((stratum, coverage));
                }
            }
        }
    }
    Ok(due.map(|(stratum, _)| stratum))
}

fn best_candidate<'a>(eligible: &'a [EligibleCandidate<'a>]) -> EligibleCandidate<'a> {
    let mut best = eligible[0];
    for candidate in eligible.iter().copied().skip(1) {
        if candidate.utility() > best.utility()
            || (candidate.utility() == best.utility()
                && candidate.action_id() < best.action_id())
        {
            best = candidate;
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn build_result(
    selected: EligibleCandidate<'_>,
    action_set: &OperationalActionSet,
    history: &PolicyHistory,
    config: &SequentialPolicyConfig,
    eligible: &[EligibleCandidate<'_>],
    candidate_assessments: &[CandidateAssessment],
    excluded: Vec<ExcludedAction>,
    mode: SelectionMode,
    action_propensities: Vec<ActionPropensity>,
    branch_random_u64: Option<u64>,
    candidate_random_u64: Option<u64>,
    audit_stratum: Option<u16>,
) -> SelectionResult {
    let assessment = selected.assessment;
    let selection_probability = action_propensities
        .iter()
        .find(|propensity| propensity.action_id == selected.action_id())
        .map(|propensity| propensity.probability)
        .expect("selected action has a recorded propensity");
    SelectionResult {
        action: selected.action.clone(),
        receipt: PolicyReceipt {
            policy_version: config.policy_version.clone(),
            model_set_version: config.model_set_version.clone(),
            assessment_set_digest: config.assessment_set_digest.clone(),
            action_set_digest: action_set.digest().to_owned(),
            history_digest: history.digest.clone(),
            policy_config: config.clone(),
            candidate_assessments: candidate_assessments.to_vec(),
            action_propensities,
            random_generator: "splitmix64-v1",
            seed: config.seed,
            decision_index: config.decision_index,
            branch_random_u64,
            candidate_random_u64,
            selection_mode: mode,
            eligible_action_ids: eligible
                .iter()
                .map(|candidate| candidate.action_id())
                .collect(),
            excluded_actions: excluded,
            selected_action: selected.action.clone(),
            selection_probability,
            selected_target_information: assessment.target_information.unwrap(),
            selected_model_discrimination_information: assessment
                .model_discrimination_information
                .unwrap(),
            selected_representativeness: assessment.representativeness.unwrap(),
            selected_worst_case_target_utility: assessment
                .worst_case_target_utility
                .unwrap(),
            selected_participant_burden_seconds: assessment
                .participant_burden_seconds
                .unwrap(),
            forced_audit_stratum: audit_stratum,
        },
    }
}

fn derive_seed(seed: u64, decision_index: u64) -> u64 {
    seed ^ decision_index.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_unit_interval(&mut self) -> (u64, f64) {
        let raw = self.next_u64();
        let unit = ((raw >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0);
        (raw, unit)
    }

    fn uniform_index(&mut self, upper_exclusive: usize) -> (usize, u64) {
        debug_assert!(upper_exclusive > 0);
        let upper = upper_exclusive as u64;
        let zone = u64::MAX - (u64::MAX % upper);
        loop {
            let raw = self.next_u64();
            if raw < zone {
                return ((raw % upper) as usize, raw);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(action_id: u32, stratum: u16, level: f64) -> OperationalAction {
        OperationalAction {
            action_id,
            audit_stratum: stratum,
            condition_id: 0,
            stimulus_class: if action_id % 2 == 0 {
                StimulusClass::Signal
            } else {
                StimulusClass::Noise
            },
            stimulus_level: level,
            response_scale_id: 1,
            rt_deadline_ms: Some(2_000),
        }
    }

    fn assessment(
        action_id: u32,
        utility: f64,
        representativeness: f64,
    ) -> CandidateAssessment {
        CandidateAssessment {
            action_id,
            operational_mapping_qualified: true,
            model_adequacy_qualified: true,
            target_information: Some(0.4 + action_id as f64 / 100.0),
            model_discrimination_information: Some(0.3),
            representativeness: Some(representativeness),
            worst_case_target_utility: Some(utility),
            participant_burden_seconds: Some(1.0),
        }
    }

    fn make_config() -> SequentialPolicyConfig {
        SequentialPolicyConfig {
            policy_version: "spectra-policy-test-v1".to_owned(),
            model_set_version: "spectra-uvsd-models-v1".to_owned(),
            assessment_set_digest: "sha256:test-assessments".to_owned(),
            seed: 7,
            decision_index: 11,
            exploration_probability: 0.2,
            minimum_representativeness: 0.5,
            minimum_model_discrimination_information: Some(0.1),
            audit_floor_by_stratum: BTreeMap::new(),
            remaining_participant_time_seconds: Some(10.0),
        }
    }

    fn make_history(
        action_set: &OperationalActionSet,
        counts: &[(u32, u64)],
    ) -> PolicyHistory {
        PolicyHistory::new(
            counts.iter().copied().collect(),
            "sha256:test-history",
            action_set,
        )
        .unwrap()
    }

    fn make_assessments() -> Vec<CandidateAssessment> {
        vec![
            assessment(1, 0.8, 0.9),
            assessment(2, 1.2, 0.8),
            assessment(3, 0.4, 0.7),
            assessment(4, 1.0, 0.95),
        ]
    }

    #[test]
    fn action_space_rejects_invalid_numeric_and_duplicate_values() {
        assert_eq!(
            OperationalActionSet::new(Vec::new(), "sha256:actions"),
            Err(ActionSpaceError::Empty)
        );
        let invalid = OperationalActionSet::new(
            vec![action(1, 1, f64::NAN)],
            "sha256:actions",
        );
        assert_eq!(invalid, Err(ActionSpaceError::NonFiniteStimulusLevel));
        let duplicate = OperationalActionSet::new(
            vec![action(1, 1, 0.0), action(1, 2, 1.0)],
            "sha256:actions",
        );
        assert_eq!(duplicate, Err(ActionSpaceError::DuplicateActionId));
    }

    #[test]
    fn exploitation_selects_highest_robust_utility_after_hard_gates() {
        let action_set = OperationalActionSet::new(
            vec![action(1, 1, -1.0), action(2, 1, 0.0), action(3, 2, 1.0)],
            "sha256:action-set",
        )
        .unwrap();
        let history = make_history(&action_set, &[]);
        let mut assessments = vec![
            assessment(1, 0.8, 0.9),
            assessment(2, 1.2, 0.8),
            assessment(3, 8.0, 0.9),
        ];
        assessments[2].operational_mapping_qualified = false;
        let mut config = make_config();
        config.exploration_probability = 0.0;
        let result = select_next_action(&action_set, &assessments, &history, &config).unwrap();
        assert_eq!(result.action.action_id, 2);
        assert_eq!(result.receipt.selection_mode, SelectionMode::Exploitation);
        assert!((result.receipt.selection_probability - 1.0).abs() < 1.0e-12);
        assert_eq!(
            result.receipt.excluded_actions,
            vec![ExcludedAction {
                action_id: 3,
                reason: ExclusionReason::OperationalMappingUnqualified,
            }]
        );
    }

    #[test]
    fn audit_floor_precedes_exploitation_and_records_probability() {
        let action_set = OperationalActionSet::new(
            vec![
                action(1, 10, 0.0),
                action(2, 10, 1.0),
                action(3, 20, 2.0),
                action(4, 20, 3.0),
            ],
            "sha256:action-set",
        )
        .unwrap();
        let history = make_history(&action_set, &[(3, 5), (4, 5)]);
        let assessments = make_assessments();
        let mut config = make_config();
        config.audit_floor_by_stratum = [(10, 2), (20, 2)].into_iter().collect();
        config.exploration_probability = 0.0;
        let result = select_next_action(&action_set, &assessments, &history, &config).unwrap();
        assert_eq!(result.receipt.selection_mode, SelectionMode::ForcedAudit);
        assert_eq!(result.receipt.forced_audit_stratum, Some(10));
        assert_eq!(result.action.audit_stratum, 10);
        assert!((result.receipt.selection_probability - 0.5).abs() < 1.0e-12);
        assert!(result.receipt.candidate_random_u64.is_some());
        let propensity_sum = result
            .receipt
            .action_propensities
            .iter()
            .map(|entry| entry.probability)
            .sum::<f64>();
        assert!((propensity_sum - 1.0).abs() < 1.0e-12);
        assert!(result
            .receipt
            .action_propensities
            .iter()
            .filter(|entry| entry.action_id == 3 || entry.action_id == 4)
            .all(|entry| entry.probability == 0.0));
    }

    #[test]
    fn exploration_and_exploitation_propensities_are_marginal_policy_probabilities() {
        let action_set = OperationalActionSet::new(
            vec![action(1, 1, 0.0), action(2, 1, 1.0), action(3, 1, 2.0)],
            "sha256:action-set",
        )
        .unwrap();
        let history = make_history(&action_set, &[]);
        let assessments = vec![
            assessment(1, 0.8, 0.9),
            assessment(2, 1.2, 0.8),
            assessment(3, 0.4, 0.7),
        ];
        let mut config = make_config();
        config.exploration_probability = 0.3;
        let mut saw_exploration = false;
        let mut saw_exploitation = false;
        for seed in 0..100 {
            config.seed = seed;
            let result =
                select_next_action(&action_set, &assessments, &history, &config).unwrap();
            match result.receipt.selection_mode {
                SelectionMode::Exploration => {
                    saw_exploration = true;
                    let expected = if result.action.action_id == 2 { 0.8 } else { 0.1 };
                    assert!((result.receipt.selection_probability - expected).abs() < 1.0e-12);
                }
                SelectionMode::Exploitation => {
                    saw_exploitation = true;
                    assert_eq!(result.action.action_id, 2);
                    assert!((result.receipt.selection_probability - 0.8).abs() < 1.0e-12);
                }
                SelectionMode::ForcedAudit => panic!("no audit floor was configured"),
            }
        }
        assert!(saw_exploration);
        assert!(saw_exploitation);
    }

    #[test]
    fn fixed_seed_history_and_policy_replay_identical_receipt() {
        let action_set = OperationalActionSet::new(
            vec![action(1, 1, 0.0), action(2, 1, 1.0), action(3, 2, 2.0)],
            "sha256:action-set",
        )
        .unwrap();
        let history = make_history(&action_set, &[(1, 2), (2, 4)]);
        let assessments = vec![
            assessment(1, 0.8, 0.9),
            assessment(2, 1.2, 0.8),
            assessment(3, 0.4, 0.7),
        ];
        let config = make_config();
        let first = select_next_action(&action_set, &assessments, &history, &config).unwrap();
        let second = select_next_action(&action_set, &assessments, &history, &config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.receipt.history_digest, "sha256:test-history");
        assert_eq!(first.receipt.action_set_digest, "sha256:action-set");
        assert_eq!(first.receipt.model_set_version, "spectra-uvsd-models-v1");
        assert_eq!(first.receipt.assessment_set_digest, "sha256:test-assessments");
        assert_eq!(first.receipt.candidate_assessments.len(), 3);
        let propensity_sum = first
            .receipt
            .action_propensities
            .iter()
            .map(|entry| entry.probability)
            .sum::<f64>();
        assert!((propensity_sum - 1.0).abs() < 1.0e-12);
        assert_eq!(first.receipt.random_generator, "splitmix64-v1");
    }

    #[test]
    fn unqualified_actions_never_enter_candidate_pool() {
        let action_set = OperationalActionSet::new(
            vec![action(1, 1, 0.0), action(2, 1, 1.0)],
            "sha256:action-set",
        )
        .unwrap();
        let history = make_history(&action_set, &[]);
        let mut assessments = vec![assessment(1, 1.0, 0.9), assessment(2, 9.0, 0.1)];
        assessments[1].operational_mapping_qualified = false;
        let mut config = make_config();
        config.exploration_probability = 1.0;
        let result = select_next_action(&action_set, &assessments, &history, &config).unwrap();
        assert_eq!(result.action.action_id, 1);
        assert_eq!(result.receipt.eligible_action_ids, vec![1]);
    }

    #[test]
    fn missing_qualified_scores_fail_closed() {
        let action_set =
            OperationalActionSet::new(vec![action(1, 1, 0.0)], "sha256:action-set").unwrap();
        let history = make_history(&action_set, &[]);
        let mut candidate = assessment(1, 1.0, 0.9);
        candidate.worst_case_target_utility = None;
        let result =
            select_next_action(&action_set, &[candidate], &history, &make_config());
        assert_eq!(result, Err(SelectionError::NoQualifiedAction));
    }

    #[test]
    fn due_audit_stratum_without_a_qualified_action_fails_closed() {
        let action_set = OperationalActionSet::new(
            vec![action(1, 10, 0.0), action(2, 20, 1.0)],
            "sha256:action-set",
        )
        .unwrap();
        let history = make_history(&action_set, &[]);
        let mut candidate_one = assessment(1, 10.0, 0.9);
        candidate_one.operational_mapping_qualified = false;
        let assessments = vec![candidate_one, assessment(2, 1.0, 0.9)];
        let mut config = make_config();
        config.audit_floor_by_stratum = [(10, 1)].into_iter().collect();
        let result = select_next_action(&action_set, &assessments, &history, &config);
        assert_eq!(
            result,
            Err(SelectionError::AuditStratumHasNoQualifiedAction)
        );
    }

    #[test]
    fn unknown_history_action_is_rejected() {
        let action_set =
            OperationalActionSet::new(vec![action(1, 1, 0.0)], "sha256:action-set").unwrap();
        let history = PolicyHistory {
            action_counts: [(99, 1)].into_iter().collect(),
            digest: "sha256:history".to_owned(),
        };
        let result =
            select_next_action(&action_set, &[assessment(1, 1.0, 0.9)], &history, &make_config());
        assert_eq!(result, Err(SelectionError::History(HistoryError::UnknownActionId)));
    }
}
