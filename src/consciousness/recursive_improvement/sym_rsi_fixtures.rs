// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic micro-worlds for the first SYM-RSI-001 mechanism test.
//!
//! These fixtures are intentionally small. Their purpose is causal attribution:
//! establish whether exact historical replay improves exploration, and whether
//! grounded dreaming adds value beyond replay, before introducing large domain
//! stacks whose unrelated complexity could obscure the mechanism under test.

use super::sym_rsi_experiment::{
    DomainSeedPlan, EvaluationSplit, ExperimentArm, ExperimentDomainSpec, SymRsiExperimentManifest,
    SYM_RSI_001_MANIFEST_SCHEMA,
};
use serde::{Deserialize, Serialize};

pub const SYM_RSI_001_FIXTURE_ADAPTER_VERSION: &str = "sym-rsi-fixtures-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixtureDomainKind {
    BranchingSearch,
    DelayedNavigation,
    RuggedOptimization,
}

impl FixtureDomainKind {
    pub const ALL: [Self; 3] = [
        Self::BranchingSearch,
        Self::DelayedNavigation,
        Self::RuggedOptimization,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::BranchingSearch => "sym-rsi-branching-search-v1",
            Self::DelayedNavigation => "sym-rsi-delayed-navigation-v1",
            Self::RuggedOptimization => "sym-rsi-rugged-optimization-v1",
        }
    }

    pub fn horizon(self, split: EvaluationSplit) -> u32 {
        let ood = split == EvaluationSplit::OutOfDistribution;
        match self {
            Self::BranchingSearch => {
                if ood { 8 } else { 6 }
            }
            Self::DelayedNavigation => {
                if ood { 14 } else { 10 }
            }
            Self::RuggedOptimization => {
                if ood { 20 } else { 12 }
            }
        }
    }

    pub fn reset(self, seed: u64, split: EvaluationSplit) -> FixtureState {
        match self {
            Self::BranchingSearch => FixtureState {
                step: 0,
                a: 0,
                b: 0,
                aux: 0,
                quality: 0.0,
                terminal: false,
            },
            Self::DelayedNavigation => {
                let size = navigation_grid_size(split);
                let center = size / 2;
                FixtureState {
                    step: 0,
                    a: center,
                    b: center,
                    aux: 0,
                    quality: 0.0,
                    terminal: false,
                }
            }
            Self::RuggedOptimization => {
                let max_x = rugged_max_x(split);
                let x = (splitmix64(seed) % (max_x as u64 + 1)) as i32;
                FixtureState {
                    step: 0,
                    a: x,
                    b: 0,
                    aux: 0,
                    quality: rugged_quality(seed, x, max_x),
                    terminal: false,
                }
            }
        }
    }

    pub fn legal_actions(self, state: &FixtureState, split: EvaluationSplit) -> Vec<u8> {
        if state.terminal {
            return Vec::new();
        }
        match self {
            Self::BranchingSearch => {
                let width = if split == EvaluationSplit::OutOfDistribution {
                    4
                } else {
                    3
                };
                (0..width).collect()
            }
            Self::DelayedNavigation => {
                let size = navigation_grid_size(split);
                let mut actions = Vec::with_capacity(4);
                if state.b > 0 {
                    actions.push(0); // north
                }
                if state.a + 1 < size {
                    actions.push(1); // east
                }
                if state.b + 1 < size {
                    actions.push(2); // south
                }
                if state.a > 0 {
                    actions.push(3); // west
                }
                actions
            }
            Self::RuggedOptimization => {
                let max_x = rugged_max_x(split);
                let mut actions = Vec::with_capacity(4);
                if state.a > 0 {
                    actions.push(0); // -1
                }
                if state.a < max_x {
                    actions.push(1); // +1
                }
                if state.a >= 3 {
                    actions.push(2); // -3
                }
                if state.a + 3 <= max_x {
                    actions.push(3); // +3
                }
                actions
            }
        }
    }

    pub fn step(
        self,
        seed: u64,
        split: EvaluationSplit,
        state: &FixtureState,
        action: u8,
    ) -> Result<FixtureTransition, FixtureDomainError> {
        if state.terminal {
            return Err(FixtureDomainError::TerminalState(self));
        }
        if !self.legal_actions(state, split).contains(&action) {
            return Err(FixtureDomainError::IllegalAction {
                domain: self,
                action,
            });
        }

        match self {
            Self::BranchingSearch => self.step_branching(seed, split, state, action),
            Self::DelayedNavigation => self.step_navigation(seed, split, state, action),
            Self::RuggedOptimization => self.step_rugged(seed, split, state, action),
        }
    }

    fn step_branching(
        self,
        seed: u64,
        split: EvaluationSplit,
        state: &FixtureState,
        action: u8,
    ) -> Result<FixtureTransition, FixtureDomainError> {
        let horizon = self.horizon(split);
        let width = if split == EvaluationSplit::OutOfDistribution {
            4_u64
        } else {
            3_u64
        };
        let target = (splitmix64(seed ^ (state.step as u64 + 1).wrapping_mul(0x9E37_79B9))
            % width) as u8;
        let matches = state.a + i32::from(action == target);
        let next_step = state.step + 1;
        let quality = matches as f64 / horizon as f64;
        let next = FixtureState {
            step: next_step,
            a: matches,
            b: i32::from(action),
            aux: state
                .aux
                .wrapping_mul(5)
                .wrapping_add(action as u64 + 1),
            quality,
            terminal: next_step >= horizon,
        };
        Ok(FixtureTransition {
            next_state: next,
            observed_quality: quality,
        })
    }

    fn step_navigation(
        self,
        seed: u64,
        split: EvaluationSplit,
        state: &FixtureState,
        action: u8,
    ) -> Result<FixtureTransition, FixtureDomainError> {
        let size = navigation_grid_size(split);
        let (mut x, mut y) = (state.a, state.b);
        match action {
            0 => y -= 1,
            1 => x += 1,
            2 => y += 1,
            3 => x -= 1,
            _ => unreachable!("legal action checked above"),
        }

        let (goal_x, goal_y) = navigation_goal(seed, size);
        let hazard = navigation_hazard(seed, x, y);
        let hazards = state.aux + u64::from(hazard);
        let reached_goal = x == goal_x && y == goal_y;
        let next_step = state.step + 1;
        let terminal = reached_goal || next_step >= self.horizon(split);
        // Deliberately sparse outcome: no quality signal until the goal is reached.
        let quality = if reached_goal {
            (1.0 - hazards as f64 * 0.05).max(0.5)
        } else {
            0.0
        };

        Ok(FixtureTransition {
            next_state: FixtureState {
                step: next_step,
                a: x,
                b: y,
                aux: hazards,
                quality,
                terminal,
            },
            observed_quality: quality,
        })
    }

    fn step_rugged(
        self,
        seed: u64,
        split: EvaluationSplit,
        state: &FixtureState,
        action: u8,
    ) -> Result<FixtureTransition, FixtureDomainError> {
        let max_x = rugged_max_x(split);
        let delta = match action {
            0 => -1,
            1 => 1,
            2 => -3,
            3 => 3,
            _ => unreachable!("legal action checked above"),
        };
        let x = state.a + delta;
        let quality = rugged_quality(seed, x, max_x);
        let next_step = state.step + 1;
        Ok(FixtureTransition {
            next_state: FixtureState {
                step: next_step,
                a: x,
                b: 0,
                aux: state.aux.wrapping_add(1),
                quality,
                terminal: next_step >= self.horizon(split),
            },
            observed_quality: quality,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureState {
    pub step: u32,
    /// Domain-specific primary integer state.
    pub a: i32,
    /// Domain-specific secondary integer state.
    pub b: i32,
    /// Domain-specific counter/path accumulator.
    pub aux: u64,
    pub quality: f64,
    pub terminal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureTransition {
    pub next_state: FixtureState,
    pub observed_quality: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixtureDomainError {
    TerminalState(FixtureDomainKind),
    IllegalAction {
        domain: FixtureDomainKind,
        action: u8,
    },
}

/// Freeze the initial mechanism-test domains and seed partitions.
///
/// The caller supplies provenance identities because those bind the exact
/// preregistration, subject commit/tree, and execution environment at run time.
pub fn canonical_sym_rsi_001_fixture_manifest(
    preregistration_digest: impl Into<String>,
    subject_digest: impl Into<String>,
    environment_digest: impl Into<String>,
) -> SymRsiExperimentManifest {
    let seeds = DomainSeedPlan {
        training_replay: vec![1, 2, 3, 4, 5, 6, 7, 8],
        held_out_replay: vec![101, 102, 103, 104],
        fresh_execution: vec![201, 202, 203, 204],
        out_of_distribution: vec![1001, 1002, 1003, 1004],
    };

    SymRsiExperimentManifest {
        schema: SYM_RSI_001_MANIFEST_SCHEMA.into(),
        experiment_id: "SYM-RSI-001".into(),
        preregistration_digest: preregistration_digest.into(),
        subject_digest: subject_digest.into(),
        environment_digest: environment_digest.into(),
        arms: ExperimentArm::ALL.to_vec(),
        domains: FixtureDomainKind::ALL
            .into_iter()
            .map(|domain| ExperimentDomainSpec {
                domain_id: domain.id().into(),
                adapter_version: SYM_RSI_001_FIXTURE_ADAPTER_VERSION.into(),
                max_evaluator_calls: 128,
                seeds: seeds.clone(),
            })
            .collect(),
        beta_cost: 0.05,
        beta_parallelism: 0.01,
        held_out_quality_tolerance: 0.02,
    }
}

fn navigation_grid_size(split: EvaluationSplit) -> i32 {
    if split == EvaluationSplit::OutOfDistribution {
        7
    } else {
        5
    }
}

fn navigation_goal(seed: u64, size: i32) -> (i32, i32) {
    match splitmix64(seed ^ 0xA11C_E5E1) % 4 {
        0 => (0, 0),
        1 => (size - 1, 0),
        2 => (0, size - 1),
        _ => (size - 1, size - 1),
    }
}

fn navigation_hazard(seed: u64, x: i32, y: i32) -> bool {
    let cell = ((x as u64) << 32) ^ y as u64;
    splitmix64(seed ^ cell ^ 0xD3A1_A7ED) % 11 == 0
}

fn rugged_max_x(split: EvaluationSplit) -> i32 {
    if split == EvaluationSplit::OutOfDistribution {
        63
    } else {
        31
    }
}

fn rugged_quality(seed: u64, x: i32, max_x: i32) -> f64 {
    let optimum = (splitmix64(seed ^ 0x0F71_AA55) % (max_x as u64 + 1)) as i32;
    let distance = (x - optimum).unsigned_abs() as f64;
    let closeness = 1.0 - distance / max_x.max(1) as f64;
    let rugged = (splitmix64(seed ^ (x as u64).wrapping_mul(0x9E37_79B9)) % 1000) as f64
        / 999.0;
    (0.7 * closeness + 0.3 * rugged).clamp(0.0, 1.0)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_manifest_is_valid_and_frozen_to_three_domains() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        assert_eq!(manifest.validate(), Ok(()));
        assert_eq!(manifest.domains.len(), 3);
        assert_eq!(
            manifest
                .domains
                .iter()
                .map(|d| d.domain_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "sym-rsi-branching-search-v1",
                "sym-rsi-delayed-navigation-v1",
                "sym-rsi-rugged-optimization-v1",
            ]
        );
    }

    #[test]
    fn fixtures_are_deterministic_for_same_seed_split_and_action() {
        for domain in FixtureDomainKind::ALL {
            let split = EvaluationSplit::TrainingReplay;
            let state_a = domain.reset(42, split);
            let state_b = domain.reset(42, split);
            assert_eq!(state_a, state_b);
            let action = domain.legal_actions(&state_a, split)[0];
            assert_eq!(
                domain.step(42, split, &state_a, action),
                domain.step(42, split, &state_b, action)
            );
        }
    }

    #[test]
    fn ood_slice_changes_problem_structure() {
        assert!(
            FixtureDomainKind::BranchingSearch.horizon(EvaluationSplit::OutOfDistribution)
                > FixtureDomainKind::BranchingSearch.horizon(EvaluationSplit::HeldOutReplay)
        );
        assert_eq!(
            FixtureDomainKind::BranchingSearch
                .legal_actions(
                    &FixtureDomainKind::BranchingSearch
                        .reset(1, EvaluationSplit::OutOfDistribution),
                    EvaluationSplit::OutOfDistribution,
                )
                .len(),
            4
        );
        assert_eq!(navigation_grid_size(EvaluationSplit::HeldOutReplay), 5);
        assert_eq!(navigation_grid_size(EvaluationSplit::OutOfDistribution), 7);
        assert_eq!(rugged_max_x(EvaluationSplit::HeldOutReplay), 31);
        assert_eq!(rugged_max_x(EvaluationSplit::OutOfDistribution), 63);
    }

    #[test]
    fn navigation_reward_is_sparse() {
        let domain = FixtureDomainKind::DelayedNavigation;
        let split = EvaluationSplit::HeldOutReplay;
        let state = domain.reset(7, split);
        let action = domain.legal_actions(&state, split)[0];
        let transition = domain.step(7, split, &state, action).unwrap();
        assert_eq!(transition.observed_quality, 0.0);
    }

    #[test]
    fn illegal_and_post_terminal_actions_fail_closed() {
        let domain = FixtureDomainKind::BranchingSearch;
        let split = EvaluationSplit::HeldOutReplay;
        let state = domain.reset(1, split);
        assert_eq!(
            domain.step(1, split, &state, 99),
            Err(FixtureDomainError::IllegalAction { domain, action: 99 })
        );

        let mut terminal = state;
        terminal.terminal = true;
        assert_eq!(
            domain.step(1, split, &terminal, 0),
            Err(FixtureDomainError::TerminalState(domain))
        );
    }
}
