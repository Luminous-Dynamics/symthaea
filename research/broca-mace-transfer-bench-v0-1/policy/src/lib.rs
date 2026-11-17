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
}

impl Policy {
    pub const ALL: [Policy; 4] = [
        Policy::NoMemory,
        Policy::FrequencyOnly,
        Policy::ScopeAware,
        Policy::CausalQualified,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::NoMemory => "no_memory",
            Self::FrequencyOnly => "frequency_only",
            Self::ScopeAware => "scope_aware",
            Self::CausalQualified => "causal_qualified",
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Decision {
    pub action: Option<Action>,
    pub causal_claim: Option<Action>,
    pub evidence_ids: Vec<u32>,
}

impl Decision {
    pub fn abstain() -> Self {
        Self::default()
    }

    fn choose(action: Action, evidence_ids: Vec<u32>) -> Self {
        Self {
            action: Some(action),
            causal_claim: None,
            evidence_ids,
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
    sum: f64,
    count: usize,
    evidence_ids: Vec<u32>,
}

impl QualifiedGroup {
    fn add(&mut self, item: &Experience) {
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
        self.independent_support() >= 2 && self.count >= 2
    }
}

fn causal_qualified_policy(memory: &[Experience], query: Query) -> Decision {
    let mut groups: BTreeMap<Action, QualifiedGroup> = BTreeMap::new();

    for item in memory.iter().filter(|item| is_qualified(item, query)) {
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
            }
        }
    }
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
