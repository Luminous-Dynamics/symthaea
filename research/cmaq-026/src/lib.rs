#![forbid(unsafe_code)]

//! Finite, probability-free decision-value audit for active probes.
//! Candidate requests contain action/probe choices only; evaluator worlds remain separate.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpec {
    pub action_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeSpec {
    pub probe_id: String,
    /// Probe cost is expressed in the same declared loss units as action regret.
    pub cost_loss_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionContract {
    pub contract_id: String,
    pub contract_sha256: String,
    pub source_manifest_sha256: String,
    pub candidate_checkpoint_sha256: String,
    pub actions: Vec<ActionSpec>,
    pub probes: Vec<ProbeSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionOutcome {
    pub loss_units: u64,
    pub invariant_safe: bool,
}

/// Evaluator-only world data. Never pass this structure to candidate decision logic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionWorld {
    pub world_id: String,
    pub environment_family_id: String,
    pub independent_cluster_id: String,
    pub action_outcomes: BTreeMap<String, ActionOutcome>,
    /// Observation value the candidate would receive if this probe were executed.
    pub probe_observations: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateDecision {
    Act {
        action_id: String,
    },
    Probe {
        probe_id: String,
        /// Policy keyed only by candidate-visible probe observations, never by world ID.
        action_by_observation: BTreeMap<String, String>,
    },
    Unknown {
        /// Explicit fallback action used while the distinction remains unresolved.
        fallback_action_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionError {
    EmptyWorldSet,
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    DuplicateAction(String),
    DuplicateProbe(String),
    DuplicateWorld(String),
    ClusterAliasedAcrossFamilies,
    ActionCoverageMismatch,
    ProbeCoverageMismatch,
    NoCommonInvariantSafeAction,
    UnknownAction(String),
    UnknownProbe(String),
    ProbePolicyObservationMismatch,
    NoInvariantSafeActionInWorld(String),
    RegretOverflow,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecisionValueScore {
    pub world_count: usize,
    pub independent_cluster_count: usize,
    pub environment_family_count: usize,
    pub action_count: usize,
    pub probe_count: usize,
    pub decision_critical_ambiguity: bool,
    /// Minimax regret of the best single invariant-safe action before probing.
    pub best_no_probe_worst_case_regret: u64,
    /// None means at least one world violates an invariant under the candidate policy.
    pub candidate_worst_case_regret: Option<u64>,
    pub probe_cost_loss_units: u64,
    /// Candidate worst-case regret plus probe cost, when the candidate policy is invariant-safe.
    pub candidate_total_worst_case_regret: Option<u64>,
    /// Positive means the candidate's selected policy beats the best robust no-probe action.
    pub decision_value_vs_no_probe: Option<i128>,
    pub probe_observation_partition_count: usize,
    /// True when every observation bucket has at least one action optimal in all its worlds.
    pub probe_resolves_action_disagreement: Option<bool>,
    pub invariant_violation_count: usize,
    pub selected_probe_id: Option<String>,
    pub unnecessary_probe: bool,
    pub information_only_probe: bool,
    pub selected_fallback: bool,
}

impl DecisionValueScore {
    pub fn net_probe_benefit(&self) -> Option<i128> {
        self.decision_value_vs_no_probe
    }
}

/// Scores one candidate decision against a finite, evaluator-only set of possible worlds.
///
/// No probabilities are assumed. The baseline is the best invariant-safe single action under
/// minimax regret. A probe is useful only when its observation-conditioned policy reduces
/// worst-case decision regret by more than its explicit cost.
pub fn score_decision_value(
    contract: &DecisionContract,
    worlds: &[DecisionWorld],
    candidate: &CandidateDecision,
) -> Result<DecisionValueScore, DecisionError> {
    if worlds.is_empty() {
        return Err(DecisionError::EmptyWorldSet);
    }
    validate_contract(contract)?;

    let action_ids: BTreeSet<&str> = contract
        .actions
        .iter()
        .map(|a| a.action_id.as_str())
        .collect();
    let probe_ids: BTreeSet<&str> = contract
        .probes
        .iter()
        .map(|p| p.probe_id.as_str())
        .collect();
    let mut world_ids = HashSet::new();
    let mut cluster_families = HashMap::new();
    let mut clusters = HashSet::new();
    let mut families = HashSet::new();
    let mut best_actions = BTreeMap::new();

    for world in worlds {
        for (value, name) in [
            (&world.world_id, "world_id"),
            (&world.environment_family_id, "environment_family_id"),
            (&world.independent_cluster_id, "independent_cluster_id"),
        ] {
            if value.trim().is_empty() {
                return Err(DecisionError::EmptyIdentity(name));
            }
        }
        if !world_ids.insert(world.world_id.clone()) {
            return Err(DecisionError::DuplicateWorld(world.world_id.clone()));
        }
        match cluster_families.entry(world.independent_cluster_id.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                if entry.get() != &world.environment_family_id {
                    return Err(DecisionError::ClusterAliasedAcrossFamilies);
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(world.environment_family_id.clone());
            }
        }
        clusters.insert(world.independent_cluster_id.clone());
        families.insert(world.environment_family_id.clone());

        let world_action_ids: BTreeSet<&str> =
            world.action_outcomes.keys().map(String::as_str).collect();
        if world_action_ids != action_ids {
            return Err(DecisionError::ActionCoverageMismatch);
        }
        let world_probe_ids: BTreeSet<&str> = world
            .probe_observations
            .keys()
            .map(String::as_str)
            .collect();
        if world_probe_ids != probe_ids
            || world
                .probe_observations
                .values()
                .any(|value| value.trim().is_empty())
        {
            return Err(DecisionError::ProbeCoverageMismatch);
        }

        let per_world_best = world
            .action_outcomes
            .iter()
            .filter(|(_, outcome)| outcome.invariant_safe)
            .map(|(id, outcome)| (id.clone(), outcome.loss_units))
            .min_by_key(|(_, loss)| *loss)
            .map(|(_, loss)| loss)
            .ok_or_else(|| DecisionError::NoInvariantSafeActionInWorld(world.world_id.clone()))?;
        let optimal: BTreeSet<String> = world
            .action_outcomes
            .iter()
            .filter(|(_, outcome)| outcome.invariant_safe && outcome.loss_units == per_world_best)
            .map(|(id, _)| id.clone())
            .collect();
        best_actions.insert(world.world_id.clone(), optimal);
    }

    let baseline = best_robust_no_probe_regret(contract, worlds)?;
    let common_optimal = common_optimal_action(worlds, &best_actions).is_some();
    let mut score = DecisionValueScore {
        world_count: worlds.len(),
        independent_cluster_count: clusters.len(),
        environment_family_count: families.len(),
        action_count: contract.actions.len(),
        probe_count: contract.probes.len(),
        decision_critical_ambiguity: !common_optimal,
        best_no_probe_worst_case_regret: baseline,
        ..DecisionValueScore::default()
    };

    let (action_for_world, probe_cost, selected_probe, partition_count, resolves, fallback) =
        match candidate {
            CandidateDecision::Act { action_id } => {
                validate_action(action_id, &action_ids)?;
                (
                    worlds
                        .iter()
                        .map(|w| (w.world_id.clone(), action_id.clone()))
                        .collect(),
                    0,
                    None,
                    0,
                    None,
                    false,
                )
            }
            CandidateDecision::Unknown { fallback_action_id } => {
                validate_action(fallback_action_id, &action_ids)?;
                (
                    worlds
                        .iter()
                        .map(|w| (w.world_id.clone(), fallback_action_id.clone()))
                        .collect(),
                    0,
                    None,
                    0,
                    None,
                    true,
                )
            }
            CandidateDecision::Probe {
                probe_id,
                action_by_observation,
            } => {
                let spec = contract
                    .probes
                    .iter()
                    .find(|p| p.probe_id == *probe_id)
                    .ok_or_else(|| DecisionError::UnknownProbe(probe_id.clone()))?;
                let mut outcomes = BTreeSet::new();
                for world in worlds {
                    outcomes.insert(world.probe_observations[probe_id].clone());
                }
                let policy_observations: BTreeSet<&str> =
                    action_by_observation.keys().map(String::as_str).collect();
                let expected_observations: BTreeSet<&str> =
                    outcomes.iter().map(String::as_str).collect();
                if policy_observations != expected_observations {
                    return Err(DecisionError::ProbePolicyObservationMismatch);
                }
                for action_id in action_by_observation.values() {
                    validate_action(action_id, &action_ids)?;
                }
                let mapping = worlds
                    .iter()
                    .map(|world| {
                        (
                            world.world_id.clone(),
                            action_by_observation[&world.probe_observations[probe_id]].clone(),
                        )
                    })
                    .collect();
                let resolved = probe_resolves_disagreement(worlds, &best_actions, probe_id);
                (
                    mapping,
                    spec.cost_loss_units,
                    Some(probe_id.clone()),
                    outcomes.len(),
                    Some(resolved),
                    false,
                )
            }
        };

    let mut worst_regret = 0u64;
    let mut violation_count = 0usize;
    for world in worlds {
        let chosen = &action_for_world[&world.world_id];
        let outcome = &world.action_outcomes[chosen];
        if !outcome.invariant_safe {
            violation_count += 1;
            continue;
        }
        let oracle_best = world
            .action_outcomes
            .values()
            .filter(|candidate| candidate.invariant_safe)
            .map(|candidate| candidate.loss_units)
            .min()
            .expect("validated each world has a safe action");
        let regret = outcome.loss_units.saturating_sub(oracle_best);
        worst_regret = worst_regret.max(regret);
    }

    let candidate_regret = (violation_count == 0).then_some(worst_regret);
    let candidate_total = candidate_regret
        .map(|regret| {
            regret
                .checked_add(probe_cost)
                .ok_or(DecisionError::RegretOverflow)
        })
        .transpose()?;

    score.candidate_worst_case_regret = candidate_regret;
    score.probe_cost_loss_units = probe_cost;
    score.candidate_total_worst_case_regret = candidate_total;
    score.decision_value_vs_no_probe =
        candidate_total.map(|total| i128::from(baseline) - i128::from(total));
    score.probe_observation_partition_count = partition_count;
    score.probe_resolves_action_disagreement = resolves;
    score.invariant_violation_count = violation_count;
    score.selected_probe_id = selected_probe;
    score.unnecessary_probe = score.selected_probe_id.is_some()
        && score
            .decision_value_vs_no_probe
            .is_some_and(|value| value <= 0);
    score.information_only_probe =
        score.selected_probe_id.is_some() && partition_count > 1 && score.unnecessary_probe;
    score.selected_fallback = fallback;
    Ok(score)
}

fn validate_contract(contract: &DecisionContract) -> Result<(), DecisionError> {
    for (value, name) in [
        (&contract.contract_id, "contract_id"),
        (&contract.contract_sha256, "contract_sha256"),
        (&contract.source_manifest_sha256, "source_manifest_sha256"),
        (
            &contract.candidate_checkpoint_sha256,
            "candidate_checkpoint_sha256",
        ),
    ] {
        if value.trim().is_empty() {
            return Err(DecisionError::EmptyIdentity(name));
        }
    }
    for (name, digest) in [
        ("contract_sha256", &contract.contract_sha256),
        ("source_manifest_sha256", &contract.source_manifest_sha256),
        (
            "candidate_checkpoint_sha256",
            &contract.candidate_checkpoint_sha256,
        ),
    ] {
        if !is_sha256_hex(digest) {
            return Err(DecisionError::InvalidSha256(name));
        }
    }
    let mut actions = HashSet::new();
    for action in &contract.actions {
        if action.action_id.trim().is_empty() {
            return Err(DecisionError::EmptyIdentity("action_id"));
        }
        if !actions.insert(action.action_id.clone()) {
            return Err(DecisionError::DuplicateAction(action.action_id.clone()));
        }
    }
    if actions.is_empty() {
        return Err(DecisionError::EmptyIdentity("actions"));
    }
    let mut probes = HashSet::new();
    for probe in &contract.probes {
        if probe.probe_id.trim().is_empty() {
            return Err(DecisionError::EmptyIdentity("probe_id"));
        }
        if !probes.insert(probe.probe_id.clone()) {
            return Err(DecisionError::DuplicateProbe(probe.probe_id.clone()));
        }
    }
    Ok(())
}

fn validate_action(action: &str, action_ids: &BTreeSet<&str>) -> Result<(), DecisionError> {
    if action_ids.contains(action) {
        Ok(())
    } else {
        Err(DecisionError::UnknownAction(action.to_owned()))
    }
}

fn best_robust_no_probe_regret(
    contract: &DecisionContract,
    worlds: &[DecisionWorld],
) -> Result<u64, DecisionError> {
    let mut best: Option<u64> = None;
    for action in &contract.actions {
        let mut worst = 0u64;
        let mut safe_everywhere = true;
        for world in worlds {
            let outcome = &world.action_outcomes[&action.action_id];
            if !outcome.invariant_safe {
                safe_everywhere = false;
                break;
            }
            let oracle_best = world
                .action_outcomes
                .values()
                .filter(|candidate| candidate.invariant_safe)
                .map(|candidate| candidate.loss_units)
                .min()
                .expect("validated each world has a safe action");
            worst = worst.max(outcome.loss_units.saturating_sub(oracle_best));
        }
        if safe_everywhere {
            best = Some(best.map_or(worst, |current| current.min(worst)));
        }
    }
    best.ok_or(DecisionError::NoCommonInvariantSafeAction)
}

fn common_optimal_action(
    worlds: &[DecisionWorld],
    best_actions: &BTreeMap<String, BTreeSet<String>>,
) -> Option<String> {
    let mut common: Option<BTreeSet<String>> = None;
    for world in worlds {
        let current = best_actions.get(&world.world_id)?.clone();
        common = Some(match common {
            None => current,
            Some(previous) => previous.intersection(&current).cloned().collect(),
        });
    }
    common.and_then(|set| set.into_iter().next())
}

fn probe_resolves_disagreement(
    worlds: &[DecisionWorld],
    best_actions: &BTreeMap<String, BTreeSet<String>>,
    probe_id: &str,
) -> bool {
    let mut partitions: BTreeMap<&str, Vec<&DecisionWorld>> = BTreeMap::new();
    for world in worlds {
        partitions
            .entry(world.probe_observations[probe_id].as_str())
            .or_default()
            .push(world);
    }
    partitions.values().all(|partition| {
        let mut common: Option<BTreeSet<String>> = None;
        for world in partition {
            let current = match best_actions.get(&world.world_id) {
                Some(value) => value.clone(),
                None => return false,
            };
            common = Some(match common {
                None => current,
                Some(previous) => previous.intersection(&current).cloned().collect(),
            });
        }
        common.is_some_and(|set| !set.is_empty())
    })
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn contract(probes: Vec<ProbeSpec>) -> DecisionContract {
        DecisionContract {
            contract_id: "decision-v1".into(),
            contract_sha256: digest('a'),
            source_manifest_sha256: digest('b'),
            candidate_checkpoint_sha256: digest('c'),
            actions: vec![
                ActionSpec {
                    action_id: "hold".into(),
                },
                ActionSpec {
                    action_id: "open".into(),
                },
                ActionSpec {
                    action_id: "close".into(),
                },
            ],
            probes,
        }
    }

    fn world(
        id: &str,
        family: &str,
        cluster: &str,
        outcomes: [(&str, u64, bool); 3],
        observations: &[(&str, &str)],
    ) -> DecisionWorld {
        DecisionWorld {
            world_id: id.into(),
            environment_family_id: family.into(),
            independent_cluster_id: cluster.into(),
            action_outcomes: outcomes
                .into_iter()
                .map(|(action, loss, safe)| {
                    (
                        action.into(),
                        ActionOutcome {
                            loss_units: loss,
                            invariant_safe: safe,
                        },
                    )
                })
                .collect(),
            probe_observations: observations
                .iter()
                .map(|(probe, value)| ((*probe).into(), (*value).into()))
                .collect(),
        }
    }

    fn probe_policy(probe: &str, mapping: &[(&str, &str)]) -> CandidateDecision {
        CandidateDecision::Probe {
            probe_id: probe.into(),
            action_by_observation: mapping
                .iter()
                .map(|(observation, action)| ((*observation).into(), (*action).into()))
                .collect(),
        }
    }

    #[test]
    fn high_uncertainty_but_same_optimal_action_has_zero_probe_value() {
        let c = contract(vec![ProbeSpec {
            probe_id: "sensor".into(),
            cost_loss_units: 2,
        }]);
        let worlds = vec![
            world(
                "w1",
                "f1",
                "c1",
                [("hold", 0, true), ("open", 8, true), ("close", 9, true)],
                &[("sensor", "a")],
            ),
            world(
                "w2",
                "f2",
                "c2",
                [("hold", 0, true), ("open", 9, true), ("close", 7, true)],
                &[("sensor", "b")],
            ),
        ];
        let score = score_decision_value(
            &c,
            &worlds,
            &probe_policy("sensor", &[("a", "hold"), ("b", "hold")]),
        )
        .unwrap();
        assert!(!score.decision_critical_ambiguity);
        assert_eq!(score.best_no_probe_worst_case_regret, 0);
        assert_eq!(score.decision_value_vs_no_probe, Some(-2));
        assert!(score.unnecessary_probe);
        assert!(score.information_only_probe);
    }

    #[test]
    fn low_world_count_but_action_critical_probe_has_positive_value() {
        let c = contract(vec![ProbeSpec {
            probe_id: "diagnostic".into(),
            cost_loss_units: 1,
        }]);
        let worlds = vec![
            world(
                "w1",
                "f1",
                "c1",
                [("hold", 8, true), ("open", 0, true), ("close", 10, true)],
                &[("diagnostic", "left")],
            ),
            world(
                "w2",
                "f2",
                "c2",
                [("hold", 8, true), ("open", 2, true), ("close", 0, true)],
                &[("diagnostic", "right")],
            ),
        ];
        let score = score_decision_value(
            &c,
            &worlds,
            &probe_policy("diagnostic", &[("left", "open"), ("right", "close")]),
        )
        .unwrap();
        assert!(score.decision_critical_ambiguity);
        assert_eq!(score.best_no_probe_worst_case_regret, 2);
        assert_eq!(score.candidate_worst_case_regret, Some(0));
        assert_eq!(score.decision_value_vs_no_probe, Some(1));
        assert_eq!(score.probe_resolves_action_disagreement, Some(true));
        assert!(!score.unnecessary_probe);
    }

    #[test]
    fn information_rich_probe_can_be_decision_irrelevant() {
        let c = contract(vec![ProbeSpec {
            probe_id: "rich".into(),
            cost_loss_units: 3,
        }]);
        let worlds = vec![
            world(
                "w1",
                "f1",
                "c1",
                [("hold", 0, true), ("open", 4, true), ("close", 5, true)],
                &[("rich", "x")],
            ),
            world(
                "w2",
                "f2",
                "c2",
                [("hold", 0, true), ("open", 5, true), ("close", 4, true)],
                &[("rich", "y")],
            ),
            world(
                "w3",
                "f3",
                "c3",
                [("hold", 0, true), ("open", 9, true), ("close", 8, true)],
                &[("rich", "z")],
            ),
        ];
        let score = score_decision_value(
            &c,
            &worlds,
            &probe_policy("rich", &[("x", "hold"), ("y", "hold"), ("z", "hold")]),
        )
        .unwrap();
        assert_eq!(score.probe_observation_partition_count, 3);
        assert!(!score.decision_critical_ambiguity);
        assert!(score.information_only_probe);
        assert_eq!(score.decision_value_vs_no_probe, Some(-3));
    }

    #[test]
    fn cheap_nonseparating_probe_is_charged_and_has_negative_value() {
        let c = contract(vec![ProbeSpec {
            probe_id: "cheap".into(),
            cost_loss_units: 1,
        }]);
        let worlds = vec![
            world(
                "w1",
                "f1",
                "c1",
                [("hold", 3, true), ("open", 0, true), ("close", 5, true)],
                &[("cheap", "same")],
            ),
            world(
                "w2",
                "f2",
                "c2",
                [("hold", 3, true), ("open", 5, true), ("close", 0, true)],
                &[("cheap", "same")],
            ),
        ];
        let score =
            score_decision_value(&c, &worlds, &probe_policy("cheap", &[("same", "hold")])).unwrap();
        assert!(score.decision_critical_ambiguity);
        assert_eq!(score.probe_observation_partition_count, 1);
        assert_eq!(score.decision_value_vs_no_probe, Some(-1));
        assert!(score.unnecessary_probe);
    }

    #[test]
    fn lower_information_probe_can_dominate_full_world_discrimination() {
        let c = contract(vec![
            ProbeSpec {
                probe_id: "cheap".into(),
                cost_loss_units: 1,
            },
            ProbeSpec {
                probe_id: "expensive".into(),
                cost_loss_units: 5,
            },
        ]);
        let worlds = vec![
            world(
                "w1",
                "f1",
                "c1",
                [("hold", 6, true), ("open", 0, true), ("close", 8, true)],
                &[("cheap", "a"), ("expensive", "1")],
            ),
            world(
                "w2",
                "f2",
                "c2",
                [("hold", 6, true), ("open", 0, true), ("close", 8, true)],
                &[("cheap", "a"), ("expensive", "2")],
            ),
            world(
                "w3",
                "f3",
                "c3",
                [("hold", 6, true), ("open", 8, true), ("close", 0, true)],
                &[("cheap", "b"), ("expensive", "3")],
            ),
        ];
        let cheap = score_decision_value(
            &c,
            &worlds,
            &probe_policy("cheap", &[("a", "open"), ("b", "close")]),
        )
        .unwrap();
        let expensive = score_decision_value(
            &c,
            &worlds,
            &probe_policy("expensive", &[("1", "open"), ("2", "open"), ("3", "close")]),
        )
        .unwrap();
        assert_eq!(cheap.probe_observation_partition_count, 2);
        assert_eq!(expensive.probe_observation_partition_count, 3);
        assert_eq!(cheap.candidate_worst_case_regret, Some(0));
        assert_eq!(expensive.candidate_worst_case_regret, Some(0));
        assert_eq!(cheap.decision_value_vs_no_probe, Some(5));
        assert_eq!(expensive.decision_value_vs_no_probe, Some(1));
        assert!(cheap.decision_value_vs_no_probe > expensive.decision_value_vs_no_probe);
    }

    #[test]
    fn invariant_violations_are_never_averaged_away() {
        let c = contract(vec![]);
        let worlds = vec![
            world(
                "w1",
                "f1",
                "c1",
                [("hold", 0, true), ("open", 1, true), ("close", 2, false)],
                &[],
            ),
            world(
                "w2",
                "f2",
                "c2",
                [("hold", 0, true), ("open", 2, true), ("close", 1, false)],
                &[],
            ),
        ];
        let score = score_decision_value(
            &c,
            &worlds,
            &CandidateDecision::Act {
                action_id: "close".into(),
            },
        )
        .unwrap();
        assert_eq!(score.invariant_violation_count, 2);
        assert_eq!(score.candidate_worst_case_regret, None);
        assert_eq!(score.decision_value_vs_no_probe, None);
    }

    #[test]
    fn no_shared_invariant_safe_action_fails_closed() {
        let c = contract(vec![]);
        let worlds = vec![
            world(
                "w1",
                "f1",
                "c1",
                [("hold", 0, true), ("open", 1, false), ("close", 2, false)],
                &[],
            ),
            world(
                "w2",
                "f2",
                "c2",
                [("hold", 0, false), ("open", 1, true), ("close", 2, false)],
                &[],
            ),
        ];
        assert_eq!(
            score_decision_value(
                &c,
                &worlds,
                &CandidateDecision::Unknown {
                    fallback_action_id: "hold".into()
                }
            ),
            Err(DecisionError::NoCommonInvariantSafeAction)
        );
    }

    #[test]
    fn candidate_policy_is_observation_keyed_not_world_keyed() {
        let c = contract(vec![ProbeSpec {
            probe_id: "sensor".into(),
            cost_loss_units: 1,
        }]);
        let worlds = vec![world(
            "w1",
            "f1",
            "c1",
            [("hold", 0, true), ("open", 1, true), ("close", 2, true)],
            &[("sensor", "x")],
        )];
        assert_eq!(
            score_decision_value(&c, &worlds, &probe_policy("sensor", &[("w1", "hold")])),
            Err(DecisionError::ProbePolicyObservationMismatch)
        );
    }
}
