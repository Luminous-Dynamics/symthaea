// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    RetainRaw,
    CompressWithSourceLinks,
    PromoteToSemantic,
    PromoteToProcedure,
    PromoteToRule,
    Quarantine,
    Supersede,
    ForgetFromActiveIndex,
    EraseForPrivacy,
    RetainProvisional,
}

const ACTIONS: [Action; 10] = [
    Action::RetainRaw,
    Action::CompressWithSourceLinks,
    Action::PromoteToSemantic,
    Action::PromoteToProcedure,
    Action::PromoteToRule,
    Action::Quarantine,
    Action::Supersede,
    Action::ForgetFromActiveIndex,
    Action::EraseForPrivacy,
    Action::RetainProvisional,
];

impl Action {
    fn name(self) -> &'static str {
        match self {
            Self::RetainRaw => "retain_raw",
            Self::CompressWithSourceLinks => "compress_with_source_links",
            Self::PromoteToSemantic => "promote_to_semantic",
            Self::PromoteToProcedure => "promote_to_procedure",
            Self::PromoteToRule => "promote_to_rule",
            Self::Quarantine => "quarantine",
            Self::Supersede => "supersede",
            Self::ForgetFromActiveIndex => "forget_from_active_index",
            Self::EraseForPrivacy => "erase_for_privacy",
            Self::RetainProvisional => "retain_provisional",
        }
    }

    fn is_promotion(self) -> bool {
        matches!(
            self,
            Self::PromoteToSemantic | Self::PromoteToProcedure | Self::PromoteToRule
        )
    }

    fn storage_units(self) -> u32 {
        match self {
            Self::RetainRaw => 100,
            Self::CompressWithSourceLinks => 30,
            Self::PromoteToSemantic => 40,
            Self::PromoteToProcedure => 50,
            Self::PromoteToRule => 60,
            Self::Quarantine => 15,
            Self::Supersede => 20,
            Self::ForgetFromActiveIndex => 5,
            Self::EraseForPrivacy => 0,
            Self::RetainProvisional => 65,
        }
    }
}

// This is the only input a candidate policy receives. It deliberately excludes
// evaluator-only outcomes, answer keys, future transfer results, and oracle actions.
#[derive(Clone, Copy, Debug)]
struct Observation {
    episode_count: u32,
    independent_source_families: u32,
    independently_observed_successes: u32,
    contradiction_count: u32,
    unverified_source_count: u32,
    scope_matches: bool,
    prediction_pre_registered: bool,
    rare_safety_detail: bool,
    source_retracted: bool,
    privacy_erasure_requested: bool,
    compressible: bool,
    non_load_bearing: bool,
}

// Evaluator-only truth. No candidate policy accepts this type.
#[derive(Clone, Copy, Debug)]
struct OracleTruth {
    independent_source_families: u32,
    independently_observed_successes: u32,
    transferable: bool,
    rule_generalizable: bool,
    poisoned: bool,
    distribution_shift: bool,
    rare_safety_detail: bool,
    source_retracted: bool,
    privacy_erasure_required: bool,
    compressible: bool,
    non_load_bearing: bool,
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    observation: Observation,
    truth: OracleTruth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Policy {
    RawOnly,
    EagerCompression,
    FrequencyOnly,
    SourceDiversityOnly,
    Governed,
}

impl Policy {
    const ALL: [Policy; 5] = [
        Policy::RawOnly,
        Policy::EagerCompression,
        Policy::FrequencyOnly,
        Policy::SourceDiversityOnly,
        Policy::Governed,
    ];

    fn name(self) -> &'static str {
        match self {
            Policy::RawOnly => "raw_only",
            Policy::EagerCompression => "eager_compression",
            Policy::FrequencyOnly => "frequency_only",
            Policy::SourceDiversityOnly => "source_diversity_only",
            Policy::Governed => "governed",
        }
    }
}

// Policies receive observations only. This function cannot inspect OracleTruth.
fn decide(policy: Policy, observation: &Observation) -> Action {
    match policy {
        Policy::RawOnly => Action::RetainRaw,
        Policy::EagerCompression => Action::CompressWithSourceLinks,
        Policy::FrequencyOnly => {
            if observation.episode_count >= 3 {
                Action::PromoteToProcedure
            } else if observation.compressible {
                Action::CompressWithSourceLinks
            } else {
                Action::RetainProvisional
            }
        }
        Policy::SourceDiversityOnly => {
            if observation.independent_source_families >= 3
                && observation.independently_observed_successes >= 3
            {
                Action::PromoteToProcedure
            } else if observation.compressible {
                Action::CompressWithSourceLinks
            } else {
                Action::RetainProvisional
            }
        }
        Policy::Governed => {
            if observation.privacy_erasure_requested {
                Action::EraseForPrivacy
            } else if observation.source_retracted {
                Action::Supersede
            } else if observation.contradiction_count > 0
                || observation.unverified_source_count > 0
            {
                Action::Quarantine
            } else if observation.rare_safety_detail {
                Action::RetainRaw
            } else if observation.non_load_bearing {
                Action::ForgetFromActiveIndex
            } else if !observation.scope_matches {
                Action::RetainProvisional
            } else if observation.independent_source_families >= 5
                && observation.independently_observed_successes >= 5
                && observation.prediction_pre_registered
            {
                Action::PromoteToRule
            } else if observation.independent_source_families >= 3
                && observation.independently_observed_successes >= 3
                && observation.prediction_pre_registered
            {
                Action::PromoteToProcedure
            } else if observation.compressible {
                Action::CompressWithSourceLinks
            } else {
                Action::RetainProvisional
            }
        }
    }
}

// The evaluator owns the utility function. It is not visible to candidate policies.
fn utility(action: Action, truth: &OracleTruth) -> i32 {
    if truth.privacy_erasure_required {
        return if action == Action::EraseForPrivacy { 12 } else { -100 };
    }

    if truth.source_retracted {
        return if action == Action::Supersede { 12 } else { -60 };
    }

    if truth.poisoned {
        return match action {
            Action::Quarantine => 12,
            Action::PromoteToSemantic | Action::PromoteToProcedure | Action::PromoteToRule => -100,
            Action::EraseForPrivacy => -20,
            _ => 0,
        };
    }

    if truth.rare_safety_detail {
        return match action {
            Action::RetainRaw => 12,
            Action::CompressWithSourceLinks => -20,
            Action::PromoteToSemantic | Action::PromoteToProcedure | Action::PromoteToRule => -100,
            Action::ForgetFromActiveIndex => -80,
            _ => 0,
        };
    }

    if truth.distribution_shift {
        return match action {
            Action::RetainProvisional => 12,
            Action::RetainRaw => 9,
            Action::Quarantine => 3,
            Action::PromoteToSemantic | Action::PromoteToProcedure | Action::PromoteToRule => -100,
            _ => 0,
        };
    }

    if truth.non_load_bearing {
        return match action {
            Action::ForgetFromActiveIndex => 12,
            Action::RetainProvisional => 8,
            Action::RetainRaw => 4,
            _ => 0,
        };
    }

    if truth.rule_generalizable
        && truth.independent_source_families >= 5
        && truth.independently_observed_successes >= 5
    {
        return match action {
            Action::PromoteToRule => 12,
            Action::PromoteToProcedure => 10,
            Action::PromoteToSemantic => 8,
            Action::CompressWithSourceLinks => 7,
            Action::RetainRaw => 6,
            Action::RetainProvisional => 4,
            _ => 0,
        };
    }

    if truth.transferable
        && truth.independent_source_families >= 3
        && truth.independently_observed_successes >= 3
    {
        return match action {
            Action::PromoteToProcedure => 12,
            Action::PromoteToSemantic => 10,
            Action::CompressWithSourceLinks => 8,
            Action::RetainRaw => 7,
            Action::RetainProvisional => 5,
            Action::PromoteToRule => 2,
            _ => 0,
        };
    }

    if truth.compressible {
        return match action {
            Action::CompressWithSourceLinks => 12,
            Action::PromoteToSemantic => 9,
            Action::RetainRaw => 7,
            Action::RetainProvisional => 5,
            _ => 0,
        };
    }

    match action {
        Action::RetainProvisional => 12,
        Action::RetainRaw => 10,
        Action::PromoteToSemantic => 5,
        Action::CompressWithSourceLinks => 4,
        _ => 0,
    }
}

fn oracle_action(truth: &OracleTruth) -> Action {
    ACTIONS
        .into_iter()
        .max_by(|left, right| {
            utility(*left, truth)
                .cmp(&utility(*right, truth))
                .then_with(|| right.storage_units().cmp(&left.storage_units()))
        })
        .unwrap_or(Action::RetainProvisional)
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "independent_transfer_support",
            observation: Observation {
                episode_count: 3,
                independent_source_families: 3,
                independently_observed_successes: 3,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: true,
                prediction_pre_registered: true,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_requested: false,
                compressible: true,
                non_load_bearing: false,
            },
            truth: OracleTruth {
                independent_source_families: 3,
                independently_observed_successes: 3,
                transferable: true,
                rule_generalizable: false,
                poisoned: false,
                distribution_shift: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_required: false,
                compressible: true,
                non_load_bearing: false,
            },
        },
        Scenario {
            name: "correlated_duplicate_echo",
            observation: Observation {
                episode_count: 8,
                independent_source_families: 1,
                independently_observed_successes: 8,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: true,
                prediction_pre_registered: true,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_requested: false,
                compressible: true,
                non_load_bearing: false,
            },
            truth: OracleTruth {
                independent_source_families: 1,
                independently_observed_successes: 8,
                transferable: false,
                rule_generalizable: false,
                poisoned: false,
                distribution_shift: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_required: false,
                compressible: true,
                non_load_bearing: false,
            },
        },
        Scenario {
            name: "poisoned_contradictory_episode",
            observation: Observation {
                episode_count: 3,
                independent_source_families: 3,
                independently_observed_successes: 3,
                contradiction_count: 1,
                unverified_source_count: 1,
                scope_matches: true,
                prediction_pre_registered: true,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_requested: false,
                compressible: true,
                non_load_bearing: false,
            },
            truth: OracleTruth {
                independent_source_families: 3,
                independently_observed_successes: 3,
                transferable: false,
                rule_generalizable: false,
                poisoned: true,
                distribution_shift: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_required: false,
                compressible: true,
                non_load_bearing: false,
            },
        },
        Scenario {
            name: "domain_shift",
            observation: Observation {
                episode_count: 5,
                independent_source_families: 5,
                independently_observed_successes: 5,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: false,
                prediction_pre_registered: true,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_requested: false,
                compressible: true,
                non_load_bearing: false,
            },
            truth: OracleTruth {
                independent_source_families: 5,
                independently_observed_successes: 5,
                transferable: false,
                rule_generalizable: false,
                poisoned: false,
                distribution_shift: true,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_required: false,
                compressible: true,
                non_load_bearing: false,
            },
        },
        Scenario {
            name: "rare_safety_critical_gotcha",
            observation: Observation {
                episode_count: 5,
                independent_source_families: 4,
                independently_observed_successes: 4,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: true,
                prediction_pre_registered: true,
                rare_safety_detail: true,
                source_retracted: false,
                privacy_erasure_requested: false,
                compressible: true,
                non_load_bearing: false,
            },
            truth: OracleTruth {
                independent_source_families: 4,
                independently_observed_successes: 4,
                transferable: true,
                rule_generalizable: false,
                poisoned: false,
                distribution_shift: false,
                rare_safety_detail: true,
                source_retracted: false,
                privacy_erasure_required: false,
                compressible: true,
                non_load_bearing: false,
            },
        },
        Scenario {
            name: "source_retracted",
            observation: Observation {
                episode_count: 4,
                independent_source_families: 3,
                independently_observed_successes: 4,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: true,
                prediction_pre_registered: true,
                rare_safety_detail: false,
                source_retracted: true,
                privacy_erasure_requested: false,
                compressible: true,
                non_load_bearing: false,
            },
            truth: OracleTruth {
                independent_source_families: 3,
                independently_observed_successes: 4,
                transferable: false,
                rule_generalizable: false,
                poisoned: false,
                distribution_shift: false,
                rare_safety_detail: false,
                source_retracted: true,
                privacy_erasure_required: false,
                compressible: true,
                non_load_bearing: false,
            },
        },
        Scenario {
            name: "privacy_erasure",
            observation: Observation {
                episode_count: 2,
                independent_source_families: 1,
                independently_observed_successes: 0,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: true,
                prediction_pre_registered: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_requested: true,
                compressible: false,
                non_load_bearing: true,
            },
            truth: OracleTruth {
                independent_source_families: 1,
                independently_observed_successes: 0,
                transferable: false,
                rule_generalizable: false,
                poisoned: false,
                distribution_shift: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_required: true,
                compressible: false,
                non_load_bearing: true,
            },
        },
        Scenario {
            name: "under_supported_candidate",
            observation: Observation {
                episode_count: 2,
                independent_source_families: 2,
                independently_observed_successes: 2,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: true,
                prediction_pre_registered: true,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_requested: false,
                compressible: false,
                non_load_bearing: false,
            },
            truth: OracleTruth {
                independent_source_families: 2,
                independently_observed_successes: 2,
                transferable: false,
                rule_generalizable: false,
                poisoned: false,
                distribution_shift: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_required: false,
                compressible: false,
                non_load_bearing: false,
            },
        },
        Scenario {
            name: "safe_compression",
            observation: Observation {
                episode_count: 4,
                independent_source_families: 2,
                independently_observed_successes: 2,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: true,
                prediction_pre_registered: true,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_requested: false,
                compressible: true,
                non_load_bearing: false,
            },
            truth: OracleTruth {
                independent_source_families: 2,
                independently_observed_successes: 2,
                transferable: false,
                rule_generalizable: false,
                poisoned: false,
                distribution_shift: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_required: false,
                compressible: true,
                non_load_bearing: false,
            },
        },
        Scenario {
            name: "independent_rule_support",
            observation: Observation {
                episode_count: 6,
                independent_source_families: 5,
                independently_observed_successes: 5,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: true,
                prediction_pre_registered: true,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_requested: false,
                compressible: true,
                non_load_bearing: false,
            },
            truth: OracleTruth {
                independent_source_families: 5,
                independently_observed_successes: 5,
                transferable: true,
                rule_generalizable: true,
                poisoned: false,
                distribution_shift: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_required: false,
                compressible: true,
                non_load_bearing: false,
            },
        },
        Scenario {
            name: "non_load_bearing_active_item",
            observation: Observation {
                episode_count: 1,
                independent_source_families: 1,
                independently_observed_successes: 0,
                contradiction_count: 0,
                unverified_source_count: 0,
                scope_matches: true,
                prediction_pre_registered: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_requested: false,
                compressible: false,
                non_load_bearing: true,
            },
            truth: OracleTruth {
                independent_source_families: 1,
                independently_observed_successes: 0,
                transferable: false,
                rule_generalizable: false,
                poisoned: false,
                distribution_shift: false,
                rare_safety_detail: false,
                source_retracted: false,
                privacy_erasure_required: false,
                compressible: false,
                non_load_bearing: true,
            },
        },
    ]
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Metrics {
    exact_oracle_matches: u32,
    total_regret: i32,
    total_utility: i32,
    total_storage_units: u32,
    harmful_promotions: u32,
    poisoned_case_escapes: u32,
    rare_detail_losses: u32,
    stale_reuse: u32,
    privacy_erasure_misses: u32,
    decisions: u32,
}

fn evaluate(policy: Policy, cases: &[Scenario]) -> Metrics {
    let mut metrics = Metrics::default();

    for case in cases {
        let selected = decide(policy, &case.observation);
        let oracle = oracle_action(&case.truth);
        let selected_utility = utility(selected, &case.truth);
        let oracle_utility = utility(oracle, &case.truth);

        metrics.decisions += 1;
        metrics.exact_oracle_matches += if selected == oracle { 1 } else { 0 };
        metrics.total_regret += oracle_utility - selected_utility;
        metrics.total_utility += selected_utility;
        metrics.total_storage_units += selected.storage_units();

        if selected.is_promotion()
            && (!case.truth.transferable
                || case.truth.poisoned
                || case.truth.distribution_shift
                || case.truth.source_retracted
                || case.truth.independent_source_families < 3
                || case.truth.independently_observed_successes < 3)
        {
            metrics.harmful_promotions += 1;
        }

        if case.truth.poisoned && selected != Action::Quarantine {
            metrics.poisoned_case_escapes += 1;
        }
        if case.truth.rare_safety_detail && selected != Action::RetainRaw {
            metrics.rare_detail_losses += 1;
        }
        if case.truth.source_retracted
            && !matches!(selected, Action::Supersede | Action::EraseForPrivacy)
        {
            metrics.stale_reuse += 1;
        }
        if case.truth.privacy_erasure_required && selected != Action::EraseForPrivacy {
            metrics.privacy_erasure_misses += 1;
        }
    }

    metrics
}

fn print_report(cases: &[Scenario]) {
    println!("Synthetic consolidation governance benchmark v0.1");
    println!(
        "cases={}, policies={}, seed=not_applicable (fully deterministic)",
        cases.len(),
        Policy::ALL.len()
    );
    println!();
    println!(
        concat!(
            "policy,oracle_exact_pct,mean_utility,mean_regret,storage_units_total,",
            "harmful_promotions,poison_escapes,rare_detail_losses,stale_reuse,",
            "privacy_erasure_misses"
        )
    );

    for policy in Policy::ALL {
        let metrics = evaluate(policy, cases);
        let count = metrics.decisions.max(1) as f64;
        let exact_pct = 100.0 * metrics.exact_oracle_matches as f64 / count;
        let mean_utility = metrics.total_utility as f64 / count;
        let mean_regret = metrics.total_regret as f64 / count;

        println!(
            "{},{:.1},{:.2},{:.2},{},{},{},{},{},{}",
            policy.name(),
            exact_pct,
            mean_utility,
            mean_regret,
            metrics.total_storage_units,
            metrics.harmful_promotions,
            metrics.poisoned_case_escapes,
            metrics.rare_detail_losses,
            metrics.stale_reuse,
            metrics.privacy_erasure_misses
        );
    }

    println!();
    println!("case,governed_action,oracle_action,utility,regret");
    for case in cases {
        let selected = decide(Policy::Governed, &case.observation);
        let oracle = oracle_action(&case.truth);
        let selected_utility = utility(selected, &case.truth);
        let regret = utility(oracle, &case.truth) - selected_utility;
        println!(
            "{},{},{},{},{}",
            case.name,
            selected.name(),
            oracle.name(),
            selected_utility,
            regret
        );
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct MemoryProjection {
    // Synthetic payload-presence markers only. No personal content is stored.
    source_payloads: BTreeSet<String>,
    active_items: BTreeSet<String>,
    provisional_items: BTreeSet<String>,
    quarantined_items: BTreeSet<String>,
    derived_children: BTreeMap<String, BTreeSet<String>>,
    erased_tombstones: BTreeSet<String>,
}

impl MemoryProjection {
    fn link_derivation(&mut self, parent: &str, child: &str) {
        self.derived_children
            .entry(parent.to_string())
            .or_default()
            .insert(child.to_string());
    }

    fn apply(&mut self, action: Action, source_id: &str, derived_id: &str) {
        self.source_payloads.insert(source_id.to_string());

        match action {
            Action::RetainRaw => {
                self.active_items.insert(source_id.to_string());
            }
            Action::CompressWithSourceLinks
            | Action::PromoteToSemantic
            | Action::PromoteToProcedure
            | Action::PromoteToRule => {
                self.active_items.remove(source_id);
                self.provisional_items.remove(source_id);
                self.active_items.insert(derived_id.to_string());
                self.link_derivation(source_id, derived_id);
                self.source_payloads.insert(derived_id.to_string());
            }
            Action::Quarantine => {
                self.active_items.remove(source_id);
                self.provisional_items.remove(source_id);
                self.quarantined_items.insert(source_id.to_string());
                self.quarantine_descendants(source_id);
            }
            Action::Supersede => {
                self.active_items.remove(source_id);
                self.provisional_items.remove(source_id);
                self.active_items.insert(derived_id.to_string());
                self.link_derivation(source_id, derived_id);
            }
            Action::ForgetFromActiveIndex => {
                self.active_items.remove(source_id);
                self.provisional_items.remove(source_id);
            }
            Action::EraseForPrivacy => self.erase_source_and_descendants(source_id),
            Action::RetainProvisional => {
                self.active_items.remove(source_id);
                self.provisional_items.insert(source_id.to_string());
            }
        }
    }

    fn descendants(&self, root: &str) -> BTreeSet<String> {
        let mut discovered = BTreeSet::new();
        let mut pending = vec![root.to_string()];

        while let Some(parent) = pending.pop() {
            if let Some(children) = self.derived_children.get(&parent) {
                for child in children {
                    if discovered.insert(child.clone()) {
                        pending.push(child.clone());
                    }
                }
            }
        }

        discovered
    }

    fn quarantine_descendants(&mut self, root: &str) {
        let descendants = self.descendants(root);
        for descendant in descendants {
            self.active_items.remove(&descendant);
            self.provisional_items.remove(&descendant);
            self.quarantined_items.insert(descendant);
        }
    }

    fn erase_source_and_descendants(&mut self, root: &str) {
        let mut all = self.descendants(root);
        all.insert(root.to_string());

        for item in &all {
            self.source_payloads.remove(item);
            self.active_items.remove(item);
            self.provisional_items.remove(item);
            self.quarantined_items.remove(item);
            self.erased_tombstones.insert(item.clone());
            self.derived_children.remove(item);
        }

        for children in self.derived_children.values_mut() {
            children.retain(|child| !all.contains(child));
        }
    }

    fn rollback(&mut self, snapshot: &Self, action: Action) -> Result<(), &'static str> {
        if action == Action::EraseForPrivacy {
            return Err("privacy erasure is irreversible and cannot be rolled back");
        }
        *self = snapshot.clone();
        Ok(())
    }
}

fn main() {
    let cases = scenarios();
    print_report(&cases);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn governed_policy_uses_observed_scope_and_source_metadata() {
        let cases = scenarios();
        for case in &cases {
            let selected = decide(Policy::Governed, &case.observation);

            if case.truth.poisoned {
                assert_eq!(selected, Action::Quarantine, "{}", case.name);
            }
            if case.truth.source_retracted {
                assert_eq!(selected, Action::Supersede, "{}", case.name);
            }
            if case.truth.privacy_erasure_required {
                assert_eq!(selected, Action::EraseForPrivacy, "{}", case.name);
            }
            if case.truth.rare_safety_detail {
                assert_eq!(selected, Action::RetainRaw, "{}", case.name);
            }
        }
    }

    #[test]
    fn repeated_correlated_episodes_do_not_count_as_independent_support() {
        let cases = scenarios();
        let case = cases
            .iter()
            .find(|case| case.name == "correlated_duplicate_echo")
            .expect("fixture exists");

        assert_eq!(
            decide(Policy::FrequencyOnly, &case.observation),
            Action::PromoteToProcedure
        );
        assert_eq!(
            decide(Policy::Governed, &case.observation),
            Action::CompressWithSourceLinks
        );
    }

    #[test]
    fn governed_policy_matches_the_symbolic_oracle_on_v0_1_fixtures() {
        let cases = scenarios();
        for case in &cases {
            assert_eq!(
                decide(Policy::Governed, &case.observation),
                oracle_action(&case.truth),
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn governed_baseline_has_no_known_governance_failures_on_v0_1() {
        let metrics = evaluate(Policy::Governed, &scenarios());
        assert_eq!(metrics.harmful_promotions, 0);
        assert_eq!(metrics.poisoned_case_escapes, 0);
        assert_eq!(metrics.rare_detail_losses, 0);
        assert_eq!(metrics.stale_reuse, 0);
        assert_eq!(metrics.privacy_erasure_misses, 0);
        assert_eq!(metrics.total_regret, 0);
    }

    #[test]
    fn simple_baselines_expose_the_expected_failure_modes() {
        let cases = scenarios();
        let frequency = evaluate(Policy::FrequencyOnly, &cases);
        let eager = evaluate(Policy::EagerCompression, &cases);

        assert!(frequency.harmful_promotions > 0);
        assert!(frequency.poisoned_case_escapes > 0);
        assert!(frequency.stale_reuse > 0);
        assert!(frequency.privacy_erasure_misses > 0);
        assert!(eager.rare_detail_losses > 0);
        assert!(eager.privacy_erasure_misses > 0);
    }

    #[test]
    fn oracle_is_an_upper_bound_and_has_zero_regret() {
        for case in scenarios() {
            let oracle = oracle_action(&case.truth);
            let best = ACTIONS
                .into_iter()
                .map(|action| utility(action, &case.truth))
                .max()
                .expect("action set is non-empty");
            assert_eq!(utility(oracle, &case.truth), best);
        }
    }

    #[test]
    fn privacy_erasure_removes_all_derived_payloads_and_cannot_be_rolled_back() {
        let mut state = MemoryProjection::default();
        state.source_payloads.extend([
            "source-a".to_string(),
            "summary-a".to_string(),
            "procedure-a".to_string(),
        ]);
        state.active_items.extend([
            "source-a".to_string(),
            "summary-a".to_string(),
            "procedure-a".to_string(),
        ]);
        state.link_derivation("source-a", "summary-a");
        state.link_derivation("summary-a", "procedure-a");

        state.apply(Action::EraseForPrivacy, "source-a", "unused");

        for erased in ["source-a", "summary-a", "procedure-a"] {
            assert!(!state.source_payloads.contains(erased));
            assert!(!state.active_items.contains(erased));
            assert!(state.erased_tombstones.contains(erased));
        }

        let snapshot = MemoryProjection::default();
        assert!(state
            .rollback(&snapshot, Action::EraseForPrivacy)
            .is_err());
    }

    #[test]
    fn reversible_projection_transitions_restore_exact_state() {
        let mut state = MemoryProjection::default();
        state.source_payloads.insert("source-a".to_string());
        state.active_items.insert("source-a".to_string());
        let before = state.clone();

        state.apply(
            Action::CompressWithSourceLinks,
            "source-a",
            "summary-a",
        );
        assert_ne!(state, before);
        state
            .rollback(&before, Action::CompressWithSourceLinks)
            .expect("compression is reversible");
        assert_eq!(state, before);
    }

    #[test]
    fn forgetting_from_active_index_does_not_erase_source_payload() {
        let mut state = MemoryProjection::default();
        state.source_payloads.insert("source-a".to_string());
        state.active_items.insert("source-a".to_string());

        state.apply(
            Action::ForgetFromActiveIndex,
            "source-a",
            "unused",
        );

        assert!(!state.active_items.contains("source-a"));
        assert!(state.source_payloads.contains("source-a"));
        assert!(!state.erased_tombstones.contains("source-a"));
    }

    #[test]
    fn repeated_runs_are_deterministic() {
        let cases = scenarios();
        for policy in Policy::ALL {
            assert_eq!(evaluate(policy, &cases), evaluate(policy, &cases));
        }
    }
}
