// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observed branching replay worlds for SYM-RSI-001.
//!
//! Multiple real trajectories from the same frozen fixture/seed/split may be merged
//! into a branching replay world. A branch exists only when that transition was
//! actually executed. If two traces claim different outcomes for the same observed
//! prefix and action, construction fails rather than averaging away the conflict.

use super::exact_replay::{ExactReplayWorld, ReplayError};
use super::experience_tree::{ExperienceNodeId, ExperienceTree, ExperienceTreeError};
use super::replay_policy::ReplayPolicyScore;
use super::sym_rsi_experiment::EvaluationSplit;
use super::sym_rsi_fixtures::{FixtureDomainKind, FixtureState};
use super::sym_rsi_runner::{fixture_action_digest, FixturePolicy, FixtureRunTrace};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFixtureWorld {
    pub domain: FixtureDomainKind,
    pub split: EvaluationSplit,
    pub seed: u64,
    pub root_node_id: ExperienceNodeId,
    pub experience_tree: ExperienceTree,
    states: BTreeMap<ExperienceNodeId, FixtureState>,
}

impl ReplayFixtureWorld {
    pub fn state(&self, id: ExperienceNodeId) -> Option<&FixtureState> {
        self.states.get(&id)
    }

    pub fn evaluate<P: FixturePolicy>(
        &self,
        policy: &mut P,
    ) -> Result<ReplayWorldEvaluation, ReplayCorpusError> {
        let replay = ExactReplayWorld::new(&self.experience_tree);
        let mut node_id = self.root_node_id;
        let mut state = self
            .states
            .get(&node_id)
            .cloned()
            .ok_or(ReplayCorpusError::MissingRawState(node_id))?;
        let mut best_quality = state.quality;
        let mut attempted_steps = 0_u64;
        let mut supported_steps = 0_u64;
        let mut unsupported_actions = 0_u64;

        while !state.terminal {
            let legal = self.domain.legal_actions(&state, self.split);
            if legal.is_empty() {
                return Err(ReplayCorpusError::NoLegalActions(node_id));
            }
            let action = policy
                .choose_action(self.domain, self.seed, self.split, &state, &legal)
                .ok_or_else(|| ReplayCorpusError::PolicyDeclinedAction {
                    policy_id: policy.policy_id().to_owned(),
                    node_id,
                })?;
            if !legal.contains(&action) {
                return Err(ReplayCorpusError::IllegalPolicyAction {
                    policy_id: policy.policy_id().to_owned(),
                    node_id,
                    action,
                });
            }

            attempted_steps += 1;
            match replay.step(node_id, &fixture_action_digest(self.domain, action)) {
                Ok(child) => {
                    supported_steps += 1;
                    node_id = child.id;
                    state = self
                        .states
                        .get(&node_id)
                        .cloned()
                        .ok_or(ReplayCorpusError::MissingRawState(node_id))?;
                    best_quality = best_quality.max(state.quality);
                }
                Err(ReplayError::UnsupportedAction { .. }) => {
                    unsupported_actions += 1;
                    break;
                }
                Err(other) => return Err(ReplayCorpusError::Replay(other)),
            }
        }

        let coverage = if attempted_steps == 0 {
            1.0
        } else {
            supported_steps as f64 / attempted_steps as f64
        };

        Ok(ReplayWorldEvaluation {
            policy_id: policy.policy_id().to_owned(),
            domain: self.domain,
            split: self.split,
            seed: self.seed,
            best_solution_quality: best_quality,
            attempted_steps,
            supported_steps,
            unsupported_actions,
            replay_coverage: coverage,
            reached_terminal: state.terminal,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayWorldEvaluation {
    pub policy_id: String,
    pub domain: FixtureDomainKind,
    pub split: EvaluationSplit,
    pub seed: u64,
    pub best_solution_quality: f64,
    pub attempted_steps: u64,
    pub supported_steps: u64,
    pub unsupported_actions: u64,
    pub replay_coverage: f64,
    pub reached_terminal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayCorpusEvaluation {
    pub policy_id: String,
    pub world_count: usize,
    pub mean_best_solution_quality: f64,
    pub attempted_steps: u64,
    pub supported_steps: u64,
    pub unsupported_worlds: usize,
    pub terminal_worlds: usize,
    pub replay_coverage: f64,
}

impl ReplayCorpusEvaluation {
    pub fn as_policy_score(&self) -> ReplayPolicyScore {
        ReplayPolicyScore {
            policy_id: self.policy_id.clone(),
            replay_quality: self.mean_best_solution_quality,
            evaluation_cost: self.attempted_steps as f64,
            parallelism_credit: 0.0,
        }
    }
}

/// Merge only genuinely observed trajectories from one exact world.
pub fn merge_observed_traces(
    traces: &[FixtureRunTrace],
) -> Result<ReplayFixtureWorld, ReplayCorpusError> {
    let first = traces.first().ok_or(ReplayCorpusError::EmptyTraces)?;
    for trace in &traces[1..] {
        if trace.domain != first.domain
            || trace.split != first.split
            || trace.seed != first.seed
            || trace.initial_state != first.initial_state
        {
            return Err(ReplayCorpusError::WorldMismatch);
        }
    }

    let first_root = first
        .experience_tree
        .node(first.root_node_id)
        .cloned()
        .ok_or(ReplayCorpusError::MissingSourceNode(first.root_node_id))?;

    let root_id = 1;
    let mut tree = ExperienceTree::new();
    let mut root = first_root;
    root.id = root_id;
    root.parent = None;
    tree.append(root).map_err(ReplayCorpusError::ExperienceTree)?;

    let mut states = BTreeMap::new();
    states.insert(root_id, first.initial_state.clone());
    let mut next_id = root_id + 1;

    for trace in traces {
        let mut merged_parent = root_id;

        for step in &trace.observed_steps {
            let source_parent = trace
                .experience_tree
                .node(step.parent_node_id)
                .ok_or(ReplayCorpusError::MissingSourceNode(step.parent_node_id))?;
            let merged_parent_node = tree
                .node(merged_parent)
                .ok_or(ReplayCorpusError::MissingMergedNode(merged_parent))?;
            if source_parent.world_state_digest != merged_parent_node.world_state_digest
                || source_parent.evidence_digest != merged_parent_node.evidence_digest
            {
                return Err(ReplayCorpusError::EvidencePrefixMismatch {
                    node_id: merged_parent,
                });
            }

            let source = trace
                .experience_tree
                .node(step.node_id)
                .cloned()
                .ok_or(ReplayCorpusError::MissingSourceNode(step.node_id))?;

            let mut existing = None;
            for child_id in tree.children_of(merged_parent) {
                let child = tree
                    .node(*child_id)
                    .ok_or(ReplayCorpusError::MissingMergedNode(*child_id))?;
                if child.action_digest == source.action_digest {
                    if child.world_state_digest != source.world_state_digest
                        || child.observation_digest != source.observation_digest
                        || child.realized_outcome_digest != source.realized_outcome_digest
                        || child.utility != source.utility
                    {
                        return Err(ReplayCorpusError::NonDeterministicTransition {
                            parent: merged_parent,
                            action_digest: source.action_digest.clone(),
                        });
                    }
                    existing = Some(*child_id);
                    break;
                }
            }

            let merged_child = if let Some(id) = existing {
                let raw_state = states
                    .get(&id)
                    .ok_or(ReplayCorpusError::MissingRawState(id))?;
                if raw_state != &step.transition.next_state {
                    return Err(ReplayCorpusError::RawStateMismatch(id));
                }
                id
            } else {
                let id = next_id;
                next_id += 1;
                let mut node = source;
                node.id = id;
                node.parent = Some(merged_parent);
                tree.append(node)
                    .map_err(ReplayCorpusError::ExperienceTree)?;
                states.insert(id, step.transition.next_state.clone());
                id
            };

            merged_parent = merged_child;
        }
    }

    Ok(ReplayFixtureWorld {
        domain: first.domain,
        split: first.split,
        seed: first.seed,
        root_node_id: root_id,
        experience_tree: tree,
        states,
    })
}

/// Evaluate one policy on a frozen set of replay worlds, resetting policy state
/// between worlds by cloning the candidate.
pub fn score_policy_on_replay_worlds<P: FixturePolicy + Clone>(
    policy: &P,
    worlds: &[ReplayFixtureWorld],
) -> Result<ReplayCorpusEvaluation, ReplayCorpusError> {
    if worlds.is_empty() {
        return Err(ReplayCorpusError::EmptyWorlds);
    }

    let mut quality_sum = 0.0;
    let mut attempted_steps = 0_u64;
    let mut supported_steps = 0_u64;
    let mut unsupported_worlds = 0_usize;
    let mut terminal_worlds = 0_usize;

    for world in worlds {
        let mut fresh_policy = policy.clone();
        let evaluation = world.evaluate(&mut fresh_policy)?;
        quality_sum += evaluation.best_solution_quality;
        attempted_steps += evaluation.attempted_steps;
        supported_steps += evaluation.supported_steps;
        if evaluation.unsupported_actions > 0 {
            unsupported_worlds += 1;
        }
        if evaluation.reached_terminal {
            terminal_worlds += 1;
        }
    }

    let replay_coverage = if attempted_steps == 0 {
        1.0
    } else {
        supported_steps as f64 / attempted_steps as f64
    };

    Ok(ReplayCorpusEvaluation {
        policy_id: policy.policy_id().to_owned(),
        world_count: worlds.len(),
        mean_best_solution_quality: quality_sum / worlds.len() as f64,
        attempted_steps,
        supported_steps,
        unsupported_worlds,
        terminal_worlds,
        replay_coverage,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayCorpusError {
    EmptyTraces,
    EmptyWorlds,
    WorldMismatch,
    MissingSourceNode(ExperienceNodeId),
    MissingMergedNode(ExperienceNodeId),
    MissingRawState(ExperienceNodeId),
    RawStateMismatch(ExperienceNodeId),
    EvidencePrefixMismatch {
        node_id: ExperienceNodeId,
    },
    NonDeterministicTransition {
        parent: ExperienceNodeId,
        action_digest: String,
    },
    NoLegalActions(ExperienceNodeId),
    PolicyDeclinedAction {
        policy_id: String,
        node_id: ExperienceNodeId,
    },
    IllegalPolicyAction {
        policy_id: String,
        node_id: ExperienceNodeId,
        action: u8,
    },
    ExperienceTree(ExperienceTreeError),
    Replay(ReplayError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        canonical_sym_rsi_001_fixture_manifest, run_fixture_policy, FixedHashPolicy,
    };

    #[derive(Clone)]
    struct PickIndexPolicy {
        id: &'static str,
        index: usize,
    }

    impl FixturePolicy for PickIndexPolicy {
        fn policy_id(&self) -> &str {
            self.id
        }

        fn choose_action(
            &mut self,
            _domain: FixtureDomainKind,
            _seed: u64,
            _split: EvaluationSplit,
            _state: &FixtureState,
            legal_actions: &[u8],
        ) -> Option<u8> {
            legal_actions.get(self.index.min(legal_actions.len().saturating_sub(1))).copied()
        }
    }

    #[test]
    fn merge_creates_branches_only_from_observed_trajectories() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let mut first = PickIndexPolicy { id: "first", index: 0 };
        let mut last = PickIndexPolicy { id: "last", index: usize::MAX };
        let trace_a = run_fixture_policy(
            &manifest,
            FixtureDomainKind::BranchingSearch,
            EvaluationSplit::TrainingReplay,
            1,
            &mut first,
        )
        .unwrap();
        let trace_b = run_fixture_policy(
            &manifest,
            FixtureDomainKind::BranchingSearch,
            EvaluationSplit::TrainingReplay,
            1,
            &mut last,
        )
        .unwrap();

        let world = merge_observed_traces(&[trace_a, trace_b]).unwrap();
        assert_eq!(world.experience_tree.children_of(world.root_node_id).len(), 2);
    }

    #[test]
    fn replay_distinguishes_supported_and_unseen_actions() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let mut first = PickIndexPolicy { id: "first", index: 0 };
        let mut last = PickIndexPolicy { id: "last", index: usize::MAX };
        let trace_a = run_fixture_policy(
            &manifest,
            FixtureDomainKind::BranchingSearch,
            EvaluationSplit::TrainingReplay,
            1,
            &mut first,
        )
        .unwrap();
        let trace_b = run_fixture_policy(
            &manifest,
            FixtureDomainKind::BranchingSearch,
            EvaluationSplit::TrainingReplay,
            1,
            &mut last,
        )
        .unwrap();
        let world = merge_observed_traces(&[trace_a, trace_b]).unwrap();

        let mut supported = PickIndexPolicy { id: "first", index: 0 };
        let supported_eval = world.evaluate(&mut supported).unwrap();
        assert_eq!(supported_eval.replay_coverage, 1.0);
        assert!(supported_eval.reached_terminal);

        let mut unseen = PickIndexPolicy { id: "middle", index: 1 };
        let unseen_eval = world.evaluate(&mut unseen).unwrap();
        assert_eq!(unseen_eval.supported_steps, 0);
        assert_eq!(unseen_eval.unsupported_actions, 1);
        assert_eq!(unseen_eval.replay_coverage, 0.0);
    }

    #[test]
    fn corpus_score_maps_into_bounded_policy_objective() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let mut collector = FixedHashPolicy::new("collector", 7);
        let trace = run_fixture_policy(
            &manifest,
            FixtureDomainKind::RuggedOptimization,
            EvaluationSplit::TrainingReplay,
            1,
            &mut collector,
        )
        .unwrap();
        let world = merge_observed_traces(&[trace]).unwrap();
        let candidate = FixedHashPolicy::new("collector", 7);
        let score = score_policy_on_replay_worlds(&candidate, &[world]).unwrap();
        assert_eq!(score.replay_coverage, 1.0);
        assert_eq!(score.as_policy_score().policy_id, "collector");
        assert!(score.as_policy_score().replay_quality.is_finite());
    }

    #[test]
    fn traces_from_different_worlds_cannot_be_merged() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let mut p1 = FixedHashPolicy::new("p", 7);
        let mut p2 = FixedHashPolicy::new("p", 7);
        let a = run_fixture_policy(
            &manifest,
            FixtureDomainKind::BranchingSearch,
            EvaluationSplit::TrainingReplay,
            1,
            &mut p1,
        )
        .unwrap();
        let b = run_fixture_policy(
            &manifest,
            FixtureDomainKind::BranchingSearch,
            EvaluationSplit::TrainingReplay,
            2,
            &mut p2,
        )
        .unwrap();
        assert!(matches!(
            merge_observed_traces(&[a, b]),
            Err(ReplayCorpusError::WorldMismatch)
        ));
    }
}
