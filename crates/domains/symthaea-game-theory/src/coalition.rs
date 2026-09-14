// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded coalition-deviation analysis for validated finite normal-form games.
//!
//! This module deliberately does **not** claim coalition-proof Nash equilibrium
//! or strong-Nash certification. It performs an explicit finite joint-deviation
//! search under a declared coalition-size bound, improvement criterion and
//! candidate-evaluation budget, and it preserves the exact witness/termination
//! reason in the result.

use crate::NPlayerGame;

/// How coalition members must benefit for a joint deviation to count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalitionImprovementCriterion {
    /// Every coalition member must receive a strictly higher payoff.
    AllStrict,
    /// No coalition member may lose and at least one must strictly improve.
    AllWeakAtLeastOneStrict,
}

/// Why a bounded coalition search terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalitionSearchTermination {
    /// A witness satisfying the declared criterion was found.
    WitnessFound,
    /// Every candidate inside the declared coalition-size bound was evaluated.
    ExhaustedSearchSpace,
    /// The explicit candidate-evaluation budget was exhausted before completion.
    BudgetExhausted,
}

/// Exact witness for one profitable coalition deviation.
#[derive(Debug, Clone, PartialEq)]
pub struct CoalitionDeviationWitness {
    /// Sorted, unique player indices participating in the coalition.
    pub coalition: Vec<usize>,
    /// Complete strategy profile before the deviation.
    pub baseline_profile: Vec<usize>,
    /// Complete strategy profile after the deviation.
    pub deviated_profile: Vec<usize>,
    /// Baseline payoff for each coalition member, in `coalition` order.
    pub baseline_payoffs: Vec<f64>,
    /// Deviated payoff for each coalition member, in `coalition` order.
    pub deviated_payoffs: Vec<f64>,
    /// Per-member payoff gain, in `coalition` order.
    pub gains: Vec<f64>,
    /// Improvement rule this witness satisfies.
    pub criterion: CoalitionImprovementCriterion,
}

/// Result of a deterministic bounded coalition-deviation search.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedCoalitionDeviationReport {
    /// Baseline strategy profile being challenged.
    pub baseline_profile: Vec<usize>,
    /// Largest coalition size included in the search.
    pub max_coalition_size: usize,
    /// Improvement rule applied to every candidate.
    pub criterion: CoalitionImprovementCriterion,
    /// Maximum number of changed candidate profiles permitted to be evaluated.
    pub candidate_budget: usize,
    /// Number of canonical coalitions reached by the deterministic search.
    pub coalitions_considered: usize,
    /// Number of changed candidate profiles actually evaluated.
    pub candidate_profiles_evaluated: usize,
    /// Exact termination reason.
    pub termination: CoalitionSearchTermination,
    /// First witness in deterministic coalition/profile order, if one was found.
    pub witness: Option<CoalitionDeviationWitness>,
}

impl NPlayerGame {
    /// Evaluate one caller-declared coalition/profile deviation.
    ///
    /// Coalition indices must be non-empty, strictly increasing and in range.
    /// Non-members must retain exactly the same strategies between profiles.
    /// Coalition members may keep their own strategy while benefiting from
    /// another member's action, but at least one coalition action must change.
    pub fn coalition_deviation_witness(
        &self,
        baseline_profile: &[usize],
        coalition: &[usize],
        deviated_profile: &[usize],
        criterion: CoalitionImprovementCriterion,
    ) -> Result<Option<CoalitionDeviationWitness>, String> {
        // Reuse the public canonical profile validation path without exposing
        // the private validator from the N-player module.
        self.profile_index(baseline_profile)?;
        self.profile_index(deviated_profile)?;
        validate_coalition(coalition, self.players())?;

        if baseline_profile == deviated_profile {
            return Err("coalition deviation must change at least one strategy".to_string());
        }

        let mut coalition_cursor = 0usize;
        for player in 0..self.players() {
            let is_member = coalition_cursor < coalition.len() && coalition[coalition_cursor] == player;
            if is_member {
                coalition_cursor += 1;
            } else if baseline_profile[player] != deviated_profile[player] {
                return Err("coalition deviation changed a non-member strategy".to_string());
            }
        }

        let mut baseline_payoffs = Vec::with_capacity(coalition.len());
        let mut deviated_payoffs = Vec::with_capacity(coalition.len());
        let mut gains = Vec::with_capacity(coalition.len());

        for &player in coalition {
            let baseline_payoff = self.payoff(player, baseline_profile)?;
            let deviated_payoff = self.payoff(player, deviated_profile)?;
            let gain = deviated_payoff - baseline_payoff;
            if !gain.is_finite() {
                return Err("coalition payoff gain overflowed finite f64 range".to_string());
            }
            baseline_payoffs.push(baseline_payoff);
            deviated_payoffs.push(deviated_payoff);
            gains.push(gain);
        }

        let satisfies = match criterion {
            CoalitionImprovementCriterion::AllStrict => gains.iter().all(|gain| *gain > 0.0),
            CoalitionImprovementCriterion::AllWeakAtLeastOneStrict => {
                gains.iter().all(|gain| *gain >= 0.0) && gains.iter().any(|gain| *gain > 0.0)
            }
        };

        if !satisfies {
            return Ok(None);
        }

        Ok(Some(CoalitionDeviationWitness {
            coalition: coalition.to_vec(),
            baseline_profile: baseline_profile.to_vec(),
            deviated_profile: deviated_profile.to_vec(),
            baseline_payoffs,
            deviated_payoffs,
            gains,
            criterion,
        }))
    }

    /// Deterministically search for a coalition deviation within explicit bounds.
    ///
    /// Coalitions are visited by increasing size and then lexicographically.
    /// Within each coalition, strategy assignments are visited by ascending
    /// mixed-radix index. The unchanged baseline assignment is skipped and does
    /// not consume the candidate budget.
    pub fn bounded_coalition_deviation_search(
        &self,
        baseline_profile: &[usize],
        max_coalition_size: usize,
        criterion: CoalitionImprovementCriterion,
        candidate_budget: usize,
    ) -> Result<BoundedCoalitionDeviationReport, String> {
        self.profile_index(baseline_profile)?;
        if max_coalition_size == 0 || max_coalition_size > self.players() {
            return Err("max coalition size must be in 1..=player_count".to_string());
        }
        if candidate_budget == 0 {
            return Err("candidate budget must be greater than zero".to_string());
        }

        let mut coalitions_considered = 0usize;
        let mut candidate_profiles_evaluated = 0usize;

        for coalition_size in 1..=max_coalition_size {
            let mut coalition: Vec<usize> = (0..coalition_size).collect();

            loop {
                coalitions_considered = coalitions_considered
                    .checked_add(1)
                    .ok_or_else(|| "coalition counter overflow".to_string())?;

                let assignment_count = coalition_assignment_count(self, &coalition)?;
                for assignment_index in 0..assignment_count {
                    let deviated_profile = coalition_assignment_profile(
                        self,
                        baseline_profile,
                        &coalition,
                        assignment_index,
                    )?;

                    if deviated_profile == baseline_profile {
                        continue;
                    }

                    if candidate_profiles_evaluated == candidate_budget {
                        return Ok(BoundedCoalitionDeviationReport {
                            baseline_profile: baseline_profile.to_vec(),
                            max_coalition_size,
                            criterion,
                            candidate_budget,
                            coalitions_considered,
                            candidate_profiles_evaluated,
                            termination: CoalitionSearchTermination::BudgetExhausted,
                            witness: None,
                        });
                    }

                    candidate_profiles_evaluated = candidate_profiles_evaluated
                        .checked_add(1)
                        .ok_or_else(|| "candidate-profile counter overflow".to_string())?;

                    if let Some(witness) = self.coalition_deviation_witness(
                        baseline_profile,
                        &coalition,
                        &deviated_profile,
                        criterion,
                    )? {
                        return Ok(BoundedCoalitionDeviationReport {
                            baseline_profile: baseline_profile.to_vec(),
                            max_coalition_size,
                            criterion,
                            candidate_budget,
                            coalitions_considered,
                            candidate_profiles_evaluated,
                            termination: CoalitionSearchTermination::WitnessFound,
                            witness: Some(witness),
                        });
                    }
                }

                if !advance_combination(&mut coalition, self.players()) {
                    break;
                }
            }
        }

        Ok(BoundedCoalitionDeviationReport {
            baseline_profile: baseline_profile.to_vec(),
            max_coalition_size,
            criterion,
            candidate_budget,
            coalitions_considered,
            candidate_profiles_evaluated,
            termination: CoalitionSearchTermination::ExhaustedSearchSpace,
            witness: None,
        })
    }
}

fn validate_coalition(coalition: &[usize], players: usize) -> Result<(), String> {
    if coalition.is_empty() {
        return Err("coalition must contain at least one player".to_string());
    }

    let mut previous = None;
    for &player in coalition {
        if player >= players {
            return Err("coalition player index out of range".to_string());
        }
        if let Some(previous_player) = previous {
            if player <= previous_player {
                return Err("coalition player indices must be strictly increasing".to_string());
            }
        }
        previous = Some(player);
    }
    Ok(())
}

fn coalition_assignment_count(game: &NPlayerGame, coalition: &[usize]) -> Result<usize, String> {
    let mut count = 1usize;
    for &player in coalition {
        count = count
            .checked_mul(game.strategy_counts()[player])
            .ok_or_else(|| "coalition assignment count overflows usize".to_string())?;
    }
    Ok(count)
}

fn coalition_assignment_profile(
    game: &NPlayerGame,
    baseline_profile: &[usize],
    coalition: &[usize],
    mut assignment_index: usize,
) -> Result<Vec<usize>, String> {
    let assignment_count = coalition_assignment_count(game, coalition)?;
    if assignment_index >= assignment_count {
        return Err("coalition assignment index out of range".to_string());
    }

    let mut profile = baseline_profile.to_vec();
    for coalition_position in (0..coalition.len()).rev() {
        let player = coalition[coalition_position];
        let strategy_count = game.strategy_counts()[player];
        profile[player] = assignment_index % strategy_count;
        assignment_index /= strategy_count;
    }
    Ok(profile)
}

fn advance_combination(combination: &mut [usize], universe_size: usize) -> bool {
    let width = combination.len();
    for position in (0..width).rev() {
        let maximum = universe_size - width + position;
        if combination[position] < maximum {
            combination[position] += 1;
            for next in (position + 1)..width {
                combination[next] = combination[next - 1] + 1;
            }
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prisoners_dilemma() -> NPlayerGame {
        NPlayerGame::new(
            vec![2, 2],
            vec![vec![3.0, 0.0, 5.0, 1.0], vec![3.0, 5.0, 0.0, 1.0]],
        )
        .unwrap()
    }

    #[test]
    fn nash_profile_can_still_have_profitable_grand_coalition_deviation() {
        let game = prisoners_dilemma();
        assert!(game.is_pure_nash(&[1, 1]).unwrap());

        let report = game
            .bounded_coalition_deviation_search(
                &[1, 1],
                2,
                CoalitionImprovementCriterion::AllStrict,
                32,
            )
            .unwrap();

        assert_eq!(report.termination, CoalitionSearchTermination::WitnessFound);
        let witness = report.witness.unwrap();
        assert_eq!(witness.coalition, vec![0, 1]);
        assert_eq!(witness.baseline_profile, vec![1, 1]);
        assert_eq!(witness.deviated_profile, vec![0, 0]);
        assert_eq!(witness.baseline_payoffs, vec![1.0, 1.0]);
        assert_eq!(witness.deviated_payoffs, vec![3.0, 3.0]);
        assert_eq!(witness.gains, vec![2.0, 2.0]);
    }

    #[test]
    fn declared_witness_rejects_non_member_strategy_changes() {
        let common = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let game = NPlayerGame::new(vec![2, 2, 2], vec![common.clone(), common.clone(), common])
            .unwrap();

        assert!(
            game.coalition_deviation_witness(
                &[0, 0, 1],
                &[0],
                &[1, 1, 1],
                CoalitionImprovementCriterion::AllStrict,
            )
            .is_err()
        );
    }

    #[test]
    fn coalition_indices_must_be_canonical() {
        let game = prisoners_dilemma();
        assert!(
            game.coalition_deviation_witness(
                &[1, 1],
                &[],
                &[0, 0],
                CoalitionImprovementCriterion::AllStrict,
            )
            .is_err()
        );
        assert!(
            game.coalition_deviation_witness(
                &[1, 1],
                &[1, 0],
                &[0, 0],
                CoalitionImprovementCriterion::AllStrict,
            )
            .is_err()
        );
        assert!(
            game.coalition_deviation_witness(
                &[1, 1],
                &[0, 0],
                &[0, 0],
                CoalitionImprovementCriterion::AllStrict,
            )
            .is_err()
        );
    }

    #[test]
    fn weak_and_strict_improvement_criteria_are_distinct() {
        // Baseline [1,1] pays (1,0); candidate [0,0] pays (1,1).
        let game = NPlayerGame::new(
            vec![2, 2],
            vec![vec![1.0, 0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();

        assert_eq!(
            game.coalition_deviation_witness(
                &[1, 1],
                &[0, 1],
                &[0, 0],
                CoalitionImprovementCriterion::AllStrict,
            )
            .unwrap(),
            None
        );

        let weak = game
            .coalition_deviation_witness(
                &[1, 1],
                &[0, 1],
                &[0, 0],
                CoalitionImprovementCriterion::AllWeakAtLeastOneStrict,
            )
            .unwrap()
            .unwrap();
        assert_eq!(weak.gains, vec![0.0, 1.0]);
    }

    #[test]
    fn tiny_budget_is_reported_as_incomplete_not_no_deviation() {
        let game = NPlayerGame::new(vec![2, 2], vec![vec![0.0; 4], vec![0.0; 4]]).unwrap();
        let report = game
            .bounded_coalition_deviation_search(
                &[0, 0],
                2,
                CoalitionImprovementCriterion::AllStrict,
                1,
            )
            .unwrap();

        assert_eq!(report.termination, CoalitionSearchTermination::BudgetExhausted);
        assert_eq!(report.candidate_profiles_evaluated, 1);
        assert!(report.witness.is_none());
    }

    #[test]
    fn sufficient_budget_can_exhaust_search_without_witness() {
        let game = NPlayerGame::new(vec![2, 2], vec![vec![0.0; 4], vec![0.0; 4]]).unwrap();
        let report = game
            .bounded_coalition_deviation_search(
                &[0, 0],
                2,
                CoalitionImprovementCriterion::AllStrict,
                32,
            )
            .unwrap();

        assert_eq!(
            report.termination,
            CoalitionSearchTermination::ExhaustedSearchSpace
        );
        assert!(report.witness.is_none());
        // Two singleton coalitions contribute one changed profile each; the
        // grand coalition contributes the three assignments other than [0,0].
        assert_eq!(report.candidate_profiles_evaluated, 5);
    }

    #[test]
    fn coalition_gain_overflow_fails_closed() {
        let game = NPlayerGame::new(vec![2], vec![vec![-f64::MAX, f64::MAX]]).unwrap();
        assert!(
            game.bounded_coalition_deviation_search(
                &[0],
                1,
                CoalitionImprovementCriterion::AllStrict,
                4,
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_search_bounds_are_rejected() {
        let game = prisoners_dilemma();
        assert!(
            game.bounded_coalition_deviation_search(
                &[1, 1],
                0,
                CoalitionImprovementCriterion::AllStrict,
                10,
            )
            .is_err()
        );
        assert!(
            game.bounded_coalition_deviation_search(
                &[1, 1],
                3,
                CoalitionImprovementCriterion::AllStrict,
                10,
            )
            .is_err()
        );
        assert!(
            game.bounded_coalition_deviation_search(
                &[1, 1],
                2,
                CoalitionImprovementCriterion::AllStrict,
                0,
            )
            .is_err()
        );
    }
}
