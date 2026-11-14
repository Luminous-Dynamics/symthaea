//! Finite operational-continuity witnesses across organization changes.
//!
//! This module compares two bounded labelled transition systems. It supports
//! one-way simulation (baseline behavior is preserved by the candidate) and
//! two-way bisimulation-style matching (both systems match under the frozen
//! action/observation alphabet). State identifiers may differ; behavior,
//! capabilities and required invariants determine the relation.
//!
//! A passing result is relative to these finite models and their declared
//! observation alphabet. It is not a universal identity, life, or
//! consciousness claim. Stochastic approximate relations are a separate
//! future extension and must consume a qualified statistical assay report.

use std::collections::{BTreeMap, BTreeSet};

/// Stable identifier for one state in a finite transition model.
pub type StateId = String;

/// State-local externally visible labels and declared contracts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StateProfile {
    /// Observable state predicates. Equality is exact in v0.1.
    pub observations: BTreeSet<String>,
    /// Capabilities available in this state.
    pub capabilities: BTreeSet<String>,
    /// Invariants observed to hold in this state.
    pub held_invariants: BTreeSet<String>,
}

/// One labelled transition. Action and observation labels are frozen by the
/// experiment manifest; state IDs are local to their respective models.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Transition {
    pub from: StateId,
    pub action: String,
    pub observation: String,
    pub to: StateId,
}

impl Transition {
    pub fn new(
        from: impl Into<StateId>,
        action: impl Into<String>,
        observation: impl Into<String>,
        to: impl Into<StateId>,
    ) -> Self {
        Self {
            from: from.into(),
            action: action.into(),
            observation: observation.into(),
            to: to.into(),
        }
    }
}

/// Finite labelled transition system derived from a frozen organization and
/// its declared observation model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransitionSystem {
    pub states: BTreeMap<StateId, StateProfile>,
    pub initial_states: BTreeSet<StateId>,
    pub transitions: BTreeSet<Transition>,
}

impl TransitionSystem {
    pub fn insert_state(
        &mut self,
        id: impl Into<StateId>,
        profile: StateProfile,
        initial: bool,
    ) -> bool {
        let id = id.into();
        if self.states.contains_key(&id) {
            return false;
        }
        if initial {
            self.initial_states.insert(id.clone());
        }
        self.states.insert(id, profile);
        true
    }

    pub fn insert_transition(&mut self, transition: Transition) -> bool {
        self.transitions.insert(transition)
    }
}

/// Simulation preserves every registered baseline transition in the
/// candidate; the candidate may have additional behavior. Bisimulation also
/// requires every candidate transition to be matched by the baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationMode {
    Simulation,
    Bisimulation,
}

/// Search limits are part of the frozen experiment protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuityPolicy {
    pub mode: RelationMode,
    pub required_invariants: BTreeSet<String>,
    pub max_state_pairs: usize,
    pub max_transition_match_checks: usize,
}

impl Default for ContinuityPolicy {
    fn default() -> Self {
        Self {
            mode: RelationMode::Simulation,
            required_invariants: BTreeSet::new(),
            max_state_pairs: 100_000,
            max_transition_match_checks: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuityQualification {
    Preserved,
    NotPreserved,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuityFindingCode {
    EmptyBaselineStateSet,
    EmptyCandidateStateSet,
    EmptyBaselineInitialSet,
    EmptyCandidateInitialSet,
    UnknownBaselineInitialState,
    UnknownCandidateInitialState,
    UnknownBaselineTransitionSource,
    UnknownBaselineTransitionTarget,
    UnknownCandidateTransitionSource,
    UnknownCandidateTransitionTarget,
    EmptyTransitionAction,
    EmptyTransitionObservation,
    EmptyObservationLabel,
    EmptyCapabilityLabel,
    EmptyInvariantLabel,
    StatePairBudgetExceeded,
    TransitionMatchBudgetExceeded,
    UnmatchedBaselineInitialState,
    UnmatchedCandidateInitialState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuityFinding {
    pub code: ContinuityFindingCode,
    pub baseline_state: Option<StateId>,
    pub candidate_state: Option<StateId>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuityReport {
    pub qualification: ContinuityQualification,
    pub mode: RelationMode,
    /// Greatest relation found under the frozen finite model and policy.
    pub related_state_pairs: Vec<(StateId, StateId)>,
    /// Initial-state pairs included in the final relation.
    pub related_initial_pairs: Vec<(StateId, StateId)>,
    pub refinement_iterations: usize,
    pub transition_match_checks: usize,
    pub findings: Vec<ContinuityFinding>,
}

impl ContinuityReport {
    pub fn preserves_operational_continuity(&self) -> bool {
        self.qualification == ContinuityQualification::Preserved
            && self.findings.is_empty()
    }
}

fn push_finding(
    findings: &mut Vec<ContinuityFinding>,
    code: ContinuityFindingCode,
    baseline_state: Option<StateId>,
    candidate_state: Option<StateId>,
    detail: impl Into<String>,
) {
    findings.push(ContinuityFinding {
        code,
        baseline_state,
        candidate_state,
        detail: detail.into(),
    });
}

fn validate_model(
    model: &TransitionSystem,
    baseline: bool,
    findings: &mut Vec<ContinuityFinding>,
) {
    if model.states.is_empty() {
        push_finding(
            findings,
            if baseline {
                ContinuityFindingCode::EmptyBaselineStateSet
            } else {
                ContinuityFindingCode::EmptyCandidateStateSet
            },
            None,
            None,
            "transition system has no states",
        );
    }

    if model.initial_states.is_empty() {
        push_finding(
            findings,
            if baseline {
                ContinuityFindingCode::EmptyBaselineInitialSet
            } else {
                ContinuityFindingCode::EmptyCandidateInitialSet
            },
            None,
            None,
            "transition system has no declared initial state",
        );
    }

    for initial in &model.initial_states {
        if !model.states.contains_key(initial) {
            push_finding(
                findings,
                if baseline {
                    ContinuityFindingCode::UnknownBaselineInitialState
                } else {
                    ContinuityFindingCode::UnknownCandidateInitialState
                },
                if baseline { Some(initial.clone()) } else { None },
                if baseline { None } else { Some(initial.clone()) },
                "initial state is absent from the state map",
            );
        }
    }

    for (state_id, profile) in &model.states {
        for label in &profile.observations {
            if label.trim().is_empty() {
                push_finding(
                    findings,
                    ContinuityFindingCode::EmptyObservationLabel,
                    if baseline { Some(state_id.clone()) } else { None },
                    if baseline { None } else { Some(state_id.clone()) },
                    "state contains an empty observation label",
                );
            }
        }
        for capability in &profile.capabilities {
            if capability.trim().is_empty() {
                push_finding(
                    findings,
                    ContinuityFindingCode::EmptyCapabilityLabel,
                    if baseline { Some(state_id.clone()) } else { None },
                    if baseline { None } else { Some(state_id.clone()) },
                    "state contains an empty capability label",
                );
            }
        }
        for invariant in &profile.held_invariants {
            if invariant.trim().is_empty() {
                push_finding(
                    findings,
                    ContinuityFindingCode::EmptyInvariantLabel,
                    if baseline { Some(state_id.clone()) } else { None },
                    if baseline { None } else { Some(state_id.clone()) },
                    "state contains an empty invariant label",
                );
            }
        }
    }

    for transition in &model.transitions {
        if transition.action.trim().is_empty() {
            push_finding(
                findings,
                ContinuityFindingCode::EmptyTransitionAction,
                if baseline { Some(transition.from.clone()) } else { None },
                if baseline { None } else { Some(transition.from.clone()) },
                "transition action label is empty",
            );
        }
        if transition.observation.trim().is_empty() {
            push_finding(
                findings,
                ContinuityFindingCode::EmptyTransitionObservation,
                if baseline { Some(transition.from.clone()) } else { None },
                if baseline { None } else { Some(transition.from.clone()) },
                "transition observation label is empty",
            );
        }
        if !model.states.contains_key(&transition.from) {
            push_finding(
                findings,
                if baseline {
                    ContinuityFindingCode::UnknownBaselineTransitionSource
                } else {
                    ContinuityFindingCode::UnknownCandidateTransitionSource
                },
                if baseline { Some(transition.from.clone()) } else { None },
                if baseline { None } else { Some(transition.from.clone()) },
                "transition source is not declared in the state map",
            );
        }
        if !model.states.contains_key(&transition.to) {
            push_finding(
                findings,
                if baseline {
                    ContinuityFindingCode::UnknownBaselineTransitionTarget
                } else {
                    ContinuityFindingCode::UnknownCandidateTransitionTarget
                },
                if baseline { Some(transition.to.clone()) } else { None },
                if baseline { None } else { Some(transition.to.clone()) },
                "transition target is not declared in the state map",
            );
        }
    }
}

fn profiles_compatible(
    baseline: &StateProfile,
    candidate: &StateProfile,
    policy: &ContinuityPolicy,
) -> bool {
    if baseline.observations != candidate.observations {
        return false;
    }

    if !policy
        .required_invariants
        .is_subset(&baseline.held_invariants)
        || !policy
            .required_invariants
            .is_subset(&candidate.held_invariants)
    {
        return false;
    }

    match policy.mode {
        RelationMode::Simulation => baseline
            .capabilities
            .is_subset(&candidate.capabilities),
        RelationMode::Bisimulation => baseline.capabilities == candidate.capabilities,
    }
}

fn outgoing<'a>(
    model: &'a TransitionSystem,
    state: &str,
) -> impl Iterator<Item = &'a Transition> {
    model.transitions.iter().filter(move |transition| transition.from == state)
}

fn transition_labels_match(left: &Transition, right: &Transition) -> bool {
    left.action == right.action && left.observation == right.observation
}

/// Establish a bounded finite simulation or bisimulation-style relation.
///
/// - Simulation: every baseline initial state and transition must be matched
///   by the candidate. Additional candidate transitions/capabilities are
///   permitted, provided required invariants and state observations are kept.
/// - Bisimulation: matching is required in both directions, and state-local
///   capability sets must be equal.
///
/// Budget exhaustion is Inconclusive, not a proof of non-equivalence.
pub fn evaluate_operational_continuity(
    baseline: &TransitionSystem,
    candidate: &TransitionSystem,
    policy: &ContinuityPolicy,
) -> ContinuityReport {
    let mut findings = Vec::new();
    validate_model(baseline, true, &mut findings);
    validate_model(candidate, false, &mut findings);

    if !findings.is_empty() {
        return ContinuityReport {
            qualification: ContinuityQualification::NotPreserved,
            mode: policy.mode,
            related_state_pairs: Vec::new(),
            related_initial_pairs: Vec::new(),
            refinement_iterations: 0,
            transition_match_checks: 0,
            findings,
        };
    }

    let pair_count = baseline.states.len().checked_mul(candidate.states.len());
    if pair_count.is_none_or(|count| count > policy.max_state_pairs) {
        push_finding(
            &mut findings,
            ContinuityFindingCode::StatePairBudgetExceeded,
            None,
            None,
            "Cartesian state-pair universe exceeds the frozen search budget",
        );
        return ContinuityReport {
            qualification: ContinuityQualification::Inconclusive,
            mode: policy.mode,
            related_state_pairs: Vec::new(),
            related_initial_pairs: Vec::new(),
            refinement_iterations: 0,
            transition_match_checks: 0,
            findings,
        };
    }

    let mut relation: BTreeSet<(StateId, StateId)> = BTreeSet::new();
    for (baseline_id, baseline_profile) in &baseline.states {
        for (candidate_id, candidate_profile) in &candidate.states {
            if profiles_compatible(baseline_profile, candidate_profile, policy) {
                relation.insert((baseline_id.clone(), candidate_id.clone()));
            }
        }
    }

    let mut iterations = 0usize;
    let mut checks = 0usize;

    loop {
        let mut remove = BTreeSet::new();
        for (baseline_id, candidate_id) in &relation {
            let pair = (baseline_id.clone(), candidate_id.clone());

            for baseline_transition in outgoing(baseline, baseline_id) {
                let mut matched = false;
                for candidate_transition in outgoing(candidate, candidate_id) {
                    checks = checks.saturating_add(1);
                    if checks > policy.max_transition_match_checks {
                        push_finding(
                            &mut findings,
                            ContinuityFindingCode::TransitionMatchBudgetExceeded,
                            Some(baseline_id.clone()),
                            Some(candidate_id.clone()),
                            "transition-pair checks exceed the frozen matching budget",
                        );
                        return ContinuityReport {
                            qualification: ContinuityQualification::Inconclusive,
                            mode: policy.mode,
                            related_state_pairs: relation.iter().cloned().collect(),
                            related_initial_pairs: Vec::new(),
                            refinement_iterations: iterations,
                            transition_match_checks: checks,
                            findings,
                        };
                    }

                    if transition_labels_match(baseline_transition, candidate_transition)
                        && relation.contains(&(
                            baseline_transition.to.clone(),
                            candidate_transition.to.clone(),
                        ))
                    {
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    remove.insert(pair.clone());
                    break;
                }
            }

            if remove.contains(&pair) {
                continue;
            }

            if policy.mode == RelationMode::Bisimulation {
                for candidate_transition in outgoing(candidate, candidate_id) {
                    let mut matched = false;
                    for baseline_transition in outgoing(baseline, baseline_id) {
                        checks = checks.saturating_add(1);
                        if checks > policy.max_transition_match_checks {
                            push_finding(
                                &mut findings,
                                ContinuityFindingCode::TransitionMatchBudgetExceeded,
                                Some(baseline_id.clone()),
                                Some(candidate_id.clone()),
                                "transition-pair checks exceed the frozen matching budget",
                            );
                            return ContinuityReport {
                                qualification: ContinuityQualification::Inconclusive,
                                mode: policy.mode,
                                related_state_pairs: relation.iter().cloned().collect(),
                                related_initial_pairs: Vec::new(),
                                refinement_iterations: iterations,
                                transition_match_checks: checks,
                                findings,
                            };
                        }

                        if transition_labels_match(candidate_transition, baseline_transition)
                            && relation.contains(&(
                                baseline_transition.to.clone(),
                                candidate_transition.to.clone(),
                            ))
                        {
                            matched = true;
                            break;
                        }
                    }
                    if !matched {
                        remove.insert(pair.clone());
                        break;
                    }
                }
            }
        }

        if remove.is_empty() {
            break;
        }
        for pair in remove {
            relation.remove(&pair);
        }
        iterations = iterations.saturating_add(1);
    }

    let mut initial_pairs = Vec::new();
    for baseline_initial in &baseline.initial_states {
        let matching = candidate.initial_states.iter().find(|candidate_initial| {
            relation.contains(&(
                baseline_initial.clone(),
                candidate_initial.as_str().to_owned(),
            ))
        });
        if let Some(candidate_initial) = matching {
            initial_pairs.push((
                baseline_initial.clone(),
                candidate_initial.as_str().to_owned(),
            ));
        } else {
            push_finding(
                &mut findings,
                ContinuityFindingCode::UnmatchedBaselineInitialState,
                Some(baseline_initial.clone()),
                None,
                "baseline initial state has no related candidate initial state",
            );
        }
    }

    if policy.mode == RelationMode::Bisimulation {
        for candidate_initial in &candidate.initial_states {
            if !baseline
                .initial_states
                .iter()
                .any(|baseline_initial| relation.contains(&(baseline_initial.clone(), candidate_initial.as_str().to_owned())))
            {
                push_finding(
                    &mut findings,
                    ContinuityFindingCode::UnmatchedCandidateInitialState,
                    None,
                    Some(candidate_initial.clone()),
                    "candidate initial state has no related baseline initial state",
                );
            }
        }
    }

    let qualification = if findings.is_empty() {
        ContinuityQualification::Preserved
    } else {
        ContinuityQualification::NotPreserved
    };

    ContinuityReport {
        qualification,
        mode: policy.mode,
        related_state_pairs: relation.iter().cloned().collect(),
        related_initial_pairs: initial_pairs,
        refinement_iterations: iterations,
        transition_match_checks: checks,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(
        observations: &[&str],
        capabilities: &[&str],
        invariants: &[&str],
    ) -> StateProfile {
        StateProfile {
            observations: observations.iter().map(|value| (*value).to_owned()).collect(),
            capabilities: capabilities.iter().map(|value| (*value).to_owned()).collect(),
            held_invariants: invariants.iter().map(|value| (*value).to_owned()).collect(),
        }
    }

    fn two_state_system(prefix: &str) -> TransitionSystem {
        let mut system = TransitionSystem::default();
        let start = format!("{prefix}-start");
        let done = format!("{prefix}-done");
        assert!(system.insert_state(start.clone(), profile(&["idle"], &["predict"], &["policy"]), true));
        assert!(system.insert_state(done.clone(), profile(&["ready"], &["predict"], &["policy"]), false));
        assert!(system.insert_transition(Transition::new(start, "refresh", "model-updated", done)));
        system
    }

    #[test]
    fn unchanged_system_passes_simulation_and_bisimulation() {
        let system = two_state_system("same");
        let simulation = evaluate_operational_continuity(&system, &system, &ContinuityPolicy::default());
        assert!(simulation.preserves_operational_continuity(), "{:?}", simulation.findings);

        let policy = ContinuityPolicy {
            mode: RelationMode::Bisimulation,
            ..ContinuityPolicy::default()
        };
        let bisimulation = evaluate_operational_continuity(&system, &system, &policy);
        assert!(bisimulation.preserves_operational_continuity(), "{:?}", bisimulation.findings);
    }

    #[test]
    fn renamed_state_ids_preserve_behavioral_continuity() {
        let baseline = two_state_system("old");
        let candidate = two_state_system("new");
        let report = evaluate_operational_continuity(&baseline, &candidate, &ContinuityPolicy::default());
        assert!(report.preserves_operational_continuity(), "{:?}", report.findings);
        assert!(report.related_state_pairs.iter().any(|(left, right)| left != right));
    }

    #[test]
    fn candidate_may_add_a_capability_under_one_way_simulation() {
        let baseline = two_state_system("old");
        let mut candidate = two_state_system("new");
        candidate.states.get_mut("new-start").unwrap().capabilities.insert("summarize".into());
        candidate.states.get_mut("new-done").unwrap().capabilities.insert("summarize".into());

        let simulation = evaluate_operational_continuity(&baseline, &candidate, &ContinuityPolicy::default());
        assert!(simulation.preserves_operational_continuity(), "{:?}", simulation.findings);

        let bisimulation_policy = ContinuityPolicy {
            mode: RelationMode::Bisimulation,
            ..ContinuityPolicy::default()
        };
        let bisimulation = evaluate_operational_continuity(&baseline, &candidate, &bisimulation_policy);
        assert!(!bisimulation.preserves_operational_continuity());
    }

    #[test]
    fn capability_loss_breaks_simulation() {
        let baseline = two_state_system("old");
        let mut candidate = two_state_system("new");
        candidate.states.get_mut("new-done").unwrap().capabilities.clear();
        let report = evaluate_operational_continuity(&baseline, &candidate, &ContinuityPolicy::default());
        assert!(!report.preserves_operational_continuity());
        assert!(report.findings.iter().any(|item| item.code == ContinuityFindingCode::UnmatchedBaselineInitialState));
    }

    #[test]
    fn missing_required_invariant_breaks_continuity() {
        let baseline = two_state_system("old");
        let mut candidate = two_state_system("new");
        candidate.states.get_mut("new-start").unwrap().held_invariants.clear();
        let policy = ContinuityPolicy {
            required_invariants: ["policy".to_owned()].into(),
            ..ContinuityPolicy::default()
        };
        let report = evaluate_operational_continuity(&baseline, &candidate, &policy);
        assert!(!report.preserves_operational_continuity());
    }

    #[test]
    fn baseline_transition_must_be_matched() {
        let baseline = two_state_system("old");
        let mut candidate = two_state_system("new");
        candidate.transitions.clear();
        let report = evaluate_operational_continuity(&baseline, &candidate, &ContinuityPolicy::default());
        assert!(!report.preserves_operational_continuity());
    }

    #[test]
    fn extra_candidate_transition_is_permitted_by_simulation_but_not_bisimulation() {
        let baseline = two_state_system("old");
        let mut candidate = two_state_system("new");
        assert!(candidate.insert_state("new-extra", profile(&["extra"], &["predict"], &["policy"]), false));
        assert!(candidate.insert_transition(Transition::new("new-start", "diagnose", "diagnostic-event", "new-extra")));

        let simulation = evaluate_operational_continuity(&baseline, &candidate, &ContinuityPolicy::default());
        assert!(simulation.preserves_operational_continuity(), "{:?}", simulation.findings);

        let policy = ContinuityPolicy {
            mode: RelationMode::Bisimulation,
            ..ContinuityPolicy::default()
        };
        let bisimulation = evaluate_operational_continuity(&baseline, &candidate, &policy);
        assert!(!bisimulation.preserves_operational_continuity());
    }

    #[test]
    fn state_observation_mismatch_breaks_relation() {
        let baseline = two_state_system("old");
        let mut candidate = two_state_system("new");
        candidate.states.get_mut("new-done").unwrap().observations = ["different".to_owned()].into();
        let report = evaluate_operational_continuity(&baseline, &candidate, &ContinuityPolicy::default());
        assert!(!report.preserves_operational_continuity());
    }

    #[test]
    fn unknown_transition_target_is_malformed_not_equivalent() {
        let baseline = two_state_system("old");
        let mut candidate = two_state_system("new");
        candidate.transitions.clear();
        assert!(candidate.insert_transition(Transition::new("new-start", "refresh", "model-updated", "ghost")));
        let report = evaluate_operational_continuity(&baseline, &candidate, &ContinuityPolicy::default());
        assert_eq!(report.qualification, ContinuityQualification::NotPreserved);
        assert!(report.findings.iter().any(|item| {
            item.code == ContinuityFindingCode::UnknownCandidateTransitionTarget
        }));
    }

    #[test]
    fn empty_state_universe_is_not_a_pass() {
        let baseline = TransitionSystem::default();
        let candidate = two_state_system("new");
        let report = evaluate_operational_continuity(&baseline, &candidate, &ContinuityPolicy::default());
        assert_eq!(report.qualification, ContinuityQualification::NotPreserved);
        assert!(report.findings.iter().any(|item| {
            item.code == ContinuityFindingCode::EmptyBaselineStateSet
        }));
    }

    #[test]
    fn state_pair_budget_exhaustion_is_inconclusive() {
        let baseline = two_state_system("old");
        let candidate = two_state_system("new");
        let policy = ContinuityPolicy {
            max_state_pairs: 1,
            ..ContinuityPolicy::default()
        };
        let report = evaluate_operational_continuity(&baseline, &candidate, &policy);
        assert_eq!(report.qualification, ContinuityQualification::Inconclusive);
        assert!(report.findings.iter().any(|item| {
            item.code == ContinuityFindingCode::StatePairBudgetExceeded
        }));
    }

    #[test]
    fn transition_match_budget_exhaustion_is_inconclusive() {
        let baseline = two_state_system("old");
        let candidate = two_state_system("new");
        let policy = ContinuityPolicy {
            max_transition_match_checks: 0,
            ..ContinuityPolicy::default()
        };
        let report = evaluate_operational_continuity(&baseline, &candidate, &policy);
        assert_eq!(report.qualification, ContinuityQualification::Inconclusive);
    }

    #[test]
    fn bisimulation_requires_candidate_initial_states_to_be_matched() {
        let baseline = two_state_system("old");
        let mut candidate = two_state_system("new");
        assert!(candidate.insert_state("new-other-initial", profile(&["other"], &["predict"], &["policy"]), true));
        let policy = ContinuityPolicy {
            mode: RelationMode::Bisimulation,
            ..ContinuityPolicy::default()
        };
        let report = evaluate_operational_continuity(&baseline, &candidate, &policy);
        assert!(!report.preserves_operational_continuity());
        assert!(report.findings.iter().any(|item| {
            item.code == ContinuityFindingCode::UnmatchedCandidateInitialState
        }));
    }
}
