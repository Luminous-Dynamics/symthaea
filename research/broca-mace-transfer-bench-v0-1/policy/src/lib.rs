// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Action {
    A,
    B,
}

impl Action {
    pub fn name(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Regime {
    Baseline,
    Shifted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Objective {
    Quality,
    Cost,
    Speed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Scope {
    pub regime: Regime,
    pub objective: Objective,
}

// Candidate-visible experience. No future labels or evaluator-only utility fields.
#[derive(Clone, Copy, Debug)]
pub struct Experience {
    pub event_id: u32,
    pub source_family: u16,
    pub scope: Scope,
    pub action: Action,
    pub observed_utility: f64,
    pub intervention: bool,
    pub prediction_pre_registered: bool,
    pub outcome_reconciled: bool,
    pub source_trusted: bool,
    pub source_retracted: bool,
    pub confounders_controlled: bool,
    pub delayed_horizon_complete: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub scope: Scope,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Policy {
    NoMemory,
    FrequencyOnly,
    ScopeAware,
    CausalQualified,
    InvariantTransport,
    ActiveProbe,
}

impl Policy {
    pub const ALL: [Policy; 6] = [
        Policy::NoMemory,
        Policy::FrequencyOnly,
        Policy::ScopeAware,
        Policy::CausalQualified,
        Policy::InvariantTransport,
        Policy::ActiveProbe,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::NoMemory => "no_memory",
            Self::FrequencyOnly => "frequency_only",
            Self::ScopeAware => "scope_aware",
            Self::CausalQualified => "causal_qualified",
            Self::InvariantTransport => "invariant_transport",
            Self::ActiveProbe => "active_probe",
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Decision {
    pub action: Option<Action>,
    pub causal_claim: Option<Action>,
    pub evidence_ids: Vec<u32>,
    /// Request a bounded environment probe before committing to a downstream action.
    pub requires_probe: bool,
}

impl Decision {
    pub fn abstain() -> Self {
        Self::default()
    }

    pub fn request_probe() -> Self {
        Self {
            action: None,
            causal_claim: None,
            evidence_ids: Vec::new(),
            requires_probe: true,
        }
    }

    fn choose(action: Action, evidence_ids: Vec<u32>) -> Self {
        Self {
            action: Some(action),
            causal_claim: None,
            evidence_ids,
            requires_probe: false,
        }
    }
}

// This API exposes no evaluator/oracle type. Policies see only Experience and Query.
pub fn decide(policy: Policy, memory: &[Experience], query: Query) -> Decision {
    match policy {
        Policy::NoMemory => Decision::choose(Action::B, Vec::new()),
        Policy::FrequencyOnly => frequency_policy(memory),
        Policy::ScopeAware => scope_aware_policy(memory, query),
        Policy::CausalQualified => causal_qualified_policy(memory, query),
        Policy::InvariantTransport => invariant_transport_policy(memory, query),
        Policy::ActiveProbe => active_probe_policy(memory, query),
    }
}

fn frequency_policy(memory: &[Experience]) -> Decision {
    let mut totals: BTreeMap<Action, (f64, usize, Vec<u32>)> = BTreeMap::new();

    for item in memory {
        let entry = totals.entry(item.action).or_insert((0.0, 0, Vec::new()));
        entry.0 += item.observed_utility;
        entry.1 += 1;
        entry.2.push(item.event_id);
    }

    let mean = |action: Action| {
        totals
            .get(&action)
            .map(|(sum, count, _)| *sum / *count as f64)
            .unwrap_or(f64::NEG_INFINITY)
    };

    let action = if mean(Action::A) > mean(Action::B) {
        Action::A
    } else {
        Action::B
    };
    let evidence_ids = totals
        .get(&action)
        .map(|(_, _, ids)| ids.clone())
        .unwrap_or_default();

    Decision::choose(action, evidence_ids)
}

fn scope_aware_policy(memory: &[Experience], query: Query) -> Decision {
    let scoped: Vec<Experience> = memory
        .iter()
        .copied()
        .filter(|item| item.scope == query.scope)
        .collect();
    frequency_policy(&scoped)
}

pub fn is_qualified(item: &Experience, query: Query) -> bool {
    item.scope == query.scope
        && item.observed_utility.is_finite()
        && (0.0..=1.0).contains(&item.observed_utility)
        && item.intervention
        && item.prediction_pre_registered
        && item.outcome_reconciled
        && item.source_trusted
        && !item.source_retracted
        && item.confounders_controlled
        && item.delayed_horizon_complete
}

#[derive(Clone, Debug, Default)]
struct QualifiedGroup {
    source_families: BTreeSet<u16>,
    source_events: BTreeSet<u32>,
    sum: f64,
    count: usize,
    evidence_ids: Vec<u32>,
}

impl QualifiedGroup {
    fn add(&mut self, item: &Experience) {
        if !self.source_events.insert(item.event_id) {
            return;
        }
        self.source_families.insert(item.source_family);
        self.sum += item.observed_utility;
        self.count += 1;
        self.evidence_ids.push(item.event_id);
    }

    fn independent_support(&self) -> usize {
        self.source_families.len()
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn is_qualified(&self) -> bool {
        self.independent_support() >= 2
            && self.source_events.len() >= 2
            && self.count >= 2
    }
}

fn causal_qualified_policy(memory: &[Experience], query: Query) -> Decision {
    let mut event_frequency: BTreeMap<u32, usize> = BTreeMap::new();
    for item in memory {
        *event_frequency.entry(item.event_id).or_default() += 1;
    }

    let mut groups: BTreeMap<Action, QualifiedGroup> = BTreeMap::new();
    for item in memory.iter().filter(|item| {
        event_frequency.get(&item.event_id) == Some(&1) && is_qualified(item, query)
    }) {
        groups.entry(item.action).or_default().add(item);
    }

    let a = groups.get(&Action::A).cloned().unwrap_or_default();
    let b = groups.get(&Action::B).cloned().unwrap_or_default();
    let a_ok = a.is_qualified();
    let b_ok = b.is_qualified();

    match (a_ok, b_ok) {
        (false, false) => Decision::abstain(),
        (true, false) => Decision::choose(Action::A, a.evidence_ids),
        (false, true) => Decision::choose(Action::B, b.evidence_ids),
        (true, true) => {
            let action = if a.mean() > b.mean() { Action::A } else { Action::B };
            let difference = (a.mean() - b.mean()).abs();
            let mut evidence_ids = a.evidence_ids;
            evidence_ids.extend(b.evidence_ids);

            Decision {
                action: Some(action),
                causal_claim: (difference >= 0.05).then_some(action),
                evidence_ids,
                requires_probe: false,
            }
        }
    }
}

/// Candidate evidence eligibility for a cross-regime transport test.
/// The optimization objective remains exact; regime compatibility is checked
/// separately by the invariant-transport policy.
pub fn is_transfer_qualified(item: &Experience, query: Query) -> bool {
    item.scope.objective == query.scope.objective
        && item.observed_utility.is_finite()
        && (0.0..=1.0).contains(&item.observed_utility)
        && item.intervention
        && item.prediction_pre_registered
        && item.outcome_reconciled
        && item.source_trusted
        && !item.source_retracted
        && item.confounders_controlled
        && item.delayed_horizon_complete
}

const INVARIANCE_TOLERANCE: f64 = 0.12;

fn invariant_transport_policy(memory: &[Experience], query: Query) -> Decision {
    let mut event_frequency: BTreeMap<u32, usize> = BTreeMap::new();
    for item in memory {
        *event_frequency.entry(item.event_id).or_default() += 1;
    }

    let mut groups: BTreeMap<(Regime, Action), QualifiedGroup> = BTreeMap::new();
    for item in memory.iter().filter(|item| {
        event_frequency.get(&item.event_id) == Some(&1)
            && is_transfer_qualified(item, query)
    }) {
        groups
            .entry((item.scope.regime, item.action))
            .or_default()
            .add(item);
    }

    let pool_action = |action: Action| -> Option<QualifiedGroup> {
        let target = groups.get(&(query.scope.regime, action))?;
        if target.count == 0 || target.source_families.is_empty() {
            return None;
        }

        let mut pooled = target.clone();
        let mut validated_source_regimes = 0usize;

        for ((regime, candidate_action), source) in &groups {
            if *candidate_action != action
                || *regime == query.scope.regime
                || !source.is_qualified()
            {
                continue;
            }

            if !source.source_families.is_disjoint(&target.source_families) {
                return None;
            }

            if (source.mean() - target.mean()).abs() > INVARIANCE_TOLERANCE {
                return None;
            }

            pooled.sum += source.sum;
            pooled.count += source.count;
            pooled.evidence_ids.extend(source.evidence_ids.iter().copied());
            pooled
                .source_families
                .extend(source.source_families.iter().copied());
            pooled
                .source_events
                .extend(source.source_events.iter().copied());
            validated_source_regimes += 1;
        }

        (validated_source_regimes > 0).then_some(pooled)
    };

    let Some(a) = pool_action(Action::A) else {
        return Decision::abstain();
    };
    let Some(b) = pool_action(Action::B) else {
        return Decision::abstain();
    };

    let action = if a.mean() > b.mean() { Action::A } else { Action::B };
    let difference = (a.mean() - b.mean()).abs();
    let mut evidence_ids = a.evidence_ids;
    evidence_ids.extend(b.evidence_ids);

    Decision {
        action: Some(action),
        causal_claim: (difference >= 0.05).then_some(action),
        evidence_ids,
        requires_probe: false,
    }
}

fn active_probe_policy(memory: &[Experience], query: Query) -> Decision {
    let exact = causal_qualified_policy(memory, query);
    if exact.action.is_some() {
        return exact;
    }

    let transported = invariant_transport_policy(memory, query);
    if transported.action.is_some() {
        return transported;
    }

    Decision::request_probe()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(
        id: u32,
        family: u16,
        action: Action,
        utility: f64,
        scope: Scope,
    ) -> Experience {
        Experience {
            event_id: id,
            source_family: family,
            scope,
            action,
            observed_utility: utility,
            intervention: true,
            prediction_pre_registered: true,
            outcome_reconciled: true,
            source_trusted: true,
            source_retracted: false,
            confounders_controlled: true,
            delayed_horizon_complete: true,
        }
    }

    #[test]
    fn valid_independent_interventions_can_change_a_later_action() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let memory = vec![
            sample(1, 10, Action::A, 0.9, scope),
            sample(2, 11, Action::A, 0.9, scope),
        ];
        let decision = decide(
            Policy::CausalQualified,
            &memory,
            Query { scope },
        );
        assert_eq!(decision.action, Some(Action::A));
    }

    #[test]
    fn correlated_repetitions_do_not_satisfy_independent_support() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let memory = vec![
            sample(1, 10, Action::A, 0.9, scope),
            sample(2, 10, Action::A, 0.9, scope),
            sample(3, 10, Action::A, 0.9, scope),
        ];
        let decision = decide(
            Policy::CausalQualified,
            &memory,
            Query { scope },
        );
        assert_eq!(decision.action, None);
    }

    #[test]
    fn same_event_with_two_claimed_source_families_is_not_independent_support() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let memory = vec![
            sample(7, 10, Action::A, 0.99, scope),
            sample(7, 11, Action::A, 0.99, scope),
            sample(8, 20, Action::B, 0.65, scope),
            sample(9, 21, Action::B, 0.68, scope),
        ];
        let frequency = decide(Policy::FrequencyOnly, &memory, Query { scope });
        let qualified = decide(Policy::CausalQualified, &memory, Query { scope });

        assert_eq!(frequency.action, Some(Action::A));
        assert_eq!(qualified.action, Some(Action::B));
        assert_eq!(qualified.evidence_ids, vec![8, 9]);
    }

    fn transport_case(
        event_id: u32,
        family: u16,
        regime: Regime,
        action: Action,
        utility: f64,
    ) -> Experience {
        sample(
            event_id,
            family,
            action,
            utility,
            Scope {
                regime,
                objective: Objective::Quality,
            },
        )
    }

    #[test]
    fn invariant_effects_can_transport_into_a_few_shot_target_regime() {
        let memory = vec![
            transport_case(1, 10, Regime::Baseline, Action::A, 0.92),
            transport_case(2, 11, Regime::Baseline, Action::A, 0.90),
            transport_case(3, 20, Regime::Baseline, Action::B, 0.32),
            transport_case(4, 21, Regime::Baseline, Action::B, 0.34),
            transport_case(5, 30, Regime::Shifted, Action::A, 0.91),
            transport_case(6, 31, Regime::Shifted, Action::B, 0.33),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        let exact = decide(Policy::CausalQualified, &memory, query);
        let transported = decide(Policy::InvariantTransport, &memory, query);

        assert_eq!(exact.action, None);
        assert_eq!(transported.action, Some(Action::A));
        assert_eq!(transported.evidence_ids.len(), 6);
    }

    #[test]
    fn invariant_transport_abstains_when_the_effect_changes_across_regimes() {
        let memory = vec![
            transport_case(1, 10, Regime::Baseline, Action::A, 0.95),
            transport_case(2, 11, Regime::Baseline, Action::A, 0.92),
            transport_case(3, 20, Regime::Baseline, Action::B, 0.32),
            transport_case(4, 21, Regime::Baseline, Action::B, 0.35),
            transport_case(5, 30, Regime::Shifted, Action::A, 0.25),
            transport_case(6, 31, Regime::Shifted, Action::B, 0.86),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        let transported = decide(Policy::InvariantTransport, &memory, query);
        assert_eq!(transported.action, None);
        assert!(transported.evidence_ids.is_empty());
    }

    #[test]
    fn invariant_transport_refuses_cross_regime_source_family_overlap() {
        let memory = vec![
            transport_case(1, 10, Regime::Baseline, Action::A, 0.92),
            transport_case(2, 11, Regime::Baseline, Action::A, 0.90),
            transport_case(3, 20, Regime::Baseline, Action::B, 0.32),
            transport_case(4, 21, Regime::Baseline, Action::B, 0.34),
            transport_case(5, 10, Regime::Shifted, Action::A, 0.91),
            transport_case(6, 20, Regime::Shifted, Action::B, 0.33),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        assert_eq!(
            decide(Policy::InvariantTransport, &memory, query).action,
            None
        );
    }

    #[test]
    fn active_probe_is_requested_only_after_exact_and_transport_paths_fail() {
        let memory = vec![
            sample(
                1,
                10,
                Action::A,
                0.9,
                Scope {
                    regime: Regime::Baseline,
                    objective: Objective::Quality,
                },
            ),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        let decision = decide(Policy::ActiveProbe, &memory, query);
        assert_eq!(decision.action, None);
        assert!(decision.requires_probe);
        assert!(decision.evidence_ids.is_empty());
    }

    #[test]
    fn active_probe_uses_exact_scope_memory_without_probe() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let memory = vec![
            sample(1, 10, Action::A, 0.9, scope),
            sample(2, 11, Action::A, 0.9, scope),
            sample(3, 20, Action::B, 0.3, scope),
            sample(4, 21, Action::B, 0.3, scope),
        ];
        let decision = decide(Policy::ActiveProbe, &memory, Query { scope });
        assert_eq!(decision.action, Some(Action::A));
        assert!(!decision.requires_probe);
    }

    #[test]
    fn scope_mismatch_is_not_eligible() {
        let source_scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Speed,
        };
        let query_scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Cost,
        };
        let item = sample(1, 10, Action::A, 0.9, source_scope);
        assert!(!is_qualified(&item, Query { scope: query_scope }));
    }
}
