//! Canonical Strategic IR and deterministic finite-game analysis primitives.
//!
//! This module is the semantic boundary between situation models and strategic
//! solvers. It intentionally contains no HDC, network, execution, or authority
//! concerns. Solver results describe what was analyzed; they do not authorize
//! an action.

/// Stable identifier for an agent/player in a strategic model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerId(pub usize);

/// A finite action/strategy identifier local to a player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(pub usize);

/// A pure strategy profile. The position in the vector is the player index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StrategyProfile(pub Vec<ActionId>);

/// A finite normal-form strategic problem before validation.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategicProblem {
    /// Number of available actions for each player.
    pub action_counts: Vec<usize>,
    /// Payoffs indexed by a flattened row-major strategy profile.
    /// Each entry contains one payoff per player.
    pub payoffs: Vec<Vec<f64>>,
}

impl StrategicProblem {
    /// Construct a finite normal-form problem.
    pub fn new(action_counts: Vec<usize>, payoffs: Vec<Vec<f64>>) -> Self {
        Self {
            action_counts,
            payoffs,
        }
    }

    /// Validate structural and numerical invariants.
    pub fn validate(self) -> Result<ValidatedGame, ValidationError> {
        if self.action_counts.is_empty() {
            return Err(ValidationError::NoPlayers);
        }
        if self.action_counts.iter().any(|&n| n == 0) {
            return Err(ValidationError::ZeroActions);
        }

        let expected_profiles = self
            .action_counts
            .iter()
            .try_fold(1usize, |acc, &n| acc.checked_mul(n))
            .ok_or(ValidationError::ProfileCountOverflow)?;

        if self.payoffs.len() != expected_profiles {
            return Err(ValidationError::PayoffTableSize {
                expected: expected_profiles,
                actual: self.payoffs.len(),
            });
        }

        let players = self.action_counts.len();
        for (profile, payoff) in self.payoffs.iter().enumerate() {
            if payoff.len() != players {
                return Err(ValidationError::PayoffArity {
                    profile,
                    expected: players,
                    actual: payoff.len(),
                });
            }
            if payoff.iter().any(|x| !x.is_finite()) {
                return Err(ValidationError::NonFinitePayoff { profile });
            }
        }

        Ok(ValidatedGame {
            action_counts: self.action_counts,
            payoffs: self.payoffs,
        })
    }
}

/// A structurally valid finite normal-form game.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedGame {
    action_counts: Vec<usize>,
    payoffs: Vec<Vec<f64>>,
}

impl ValidatedGame {
    pub fn player_count(&self) -> usize {
        self.action_counts.len()
    }

    pub fn action_count(&self, player: PlayerId) -> Option<usize> {
        self.action_counts.get(player.0).copied()
    }

    pub fn profile_count(&self) -> usize {
        self.payoffs.len()
    }

    pub fn payoff(&self, profile: &StrategyProfile, player: PlayerId) -> Option<f64> {
        if profile.0.len() != self.player_count() || player.0 >= self.player_count() {
            return None;
        }
        let index = self.profile_index(profile)?;
        self.payoffs.get(index)?.get(player.0).copied()
    }

    pub fn profiles(&self) -> ProfileIter<'_> {
        ProfileIter::new(&self.action_counts)
    }

    fn profile_index(&self, profile: &StrategyProfile) -> Option<usize> {
        if profile.0.len() != self.player_count() {
            return None;
        }
        let mut index = 0usize;
        for (player, action) in profile.0.iter().enumerate() {
            if action.0 >= self.action_counts[player] {
                return None;
            }
            index = index * self.action_counts[player] + action.0;
        }
        Some(index)
    }
}

/// Lazy iterator over all pure strategy profiles.
pub struct ProfileIter<'a> {
    counts: &'a [usize],
    current: Option<Vec<usize>>,
}

impl<'a> ProfileIter<'a> {
    fn new(counts: &'a [usize]) -> Self {
        Self {
            counts,
            current: Some(vec![0; counts.len()]),
        }
    }
}

impl Iterator for ProfileIter<'_> {
    type Item = StrategyProfile;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.as_mut()?;
        let result = StrategyProfile(current.iter().copied().map(ActionId).collect());

        for i in (0..current.len()).rev() {
            current[i] += 1;
            if current[i] < self.counts[i] {
                return Some(result);
            }
            current[i] = 0;
        }
        self.current = None;
        Some(result)
    }
}

/// A requested class of strategic analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisTask {
    FindEquilibrium,
    EvaluateStrategy,
    FindCounterexample,
}

/// The mathematical method actually used by a solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisMethod {
    ExhaustiveEnumeration,
    Analytic,
    Approximate,
    MonteCarlo,
    Heuristic,
}

/// How much of the relevant search space was covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completeness {
    Exhaustive,
    ProvenComplete,
    BoundedSearch,
    Sampled,
    Heuristic,
    Unknown,
}

/// A pure Nash equilibrium certificate.
#[derive(Debug, Clone, PartialEq)]
pub struct PureNashEquilibrium {
    pub profile: StrategyProfile,
}

/// Structured strategic output. This is analysis evidence, not authority.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategicResult {
    pub equilibria: Vec<PureNashEquilibrium>,
    pub method: AnalysisMethod,
    pub completeness: Completeness,
}

/// Errors produced while validating a strategic problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    NoPlayers,
    ZeroActions,
    ProfileCountOverflow,
    PayoffTableSize { expected: usize, actual: usize },
    PayoffArity { profile: usize, expected: usize, actual: usize },
    NonFinitePayoff { profile: usize },
}

/// Errors produced by strategic analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisError {
    UnsupportedTask(AnalysisTask),
}

/// Declares mathematical and information-structure assumptions required by a solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverCapabilities {
    pub supports_imperfect_information: bool,
    pub requires_perfect_recall: bool,
    pub supports_chance: bool,
    pub supports_general_sum: bool,
}

/// Solver interface for validated strategic models.
pub trait StrategicSolver {
    fn method(&self) -> AnalysisMethod;
    fn supported_tasks(&self) -> &'static [AnalysisTask];

    fn solve(
        &self,
        game: &ValidatedGame,
        task: AnalysisTask,
    ) -> Result<StrategicResult, AnalysisError>;
}

/// Exhaustive pure-strategy Nash solver for finite normal-form games.
#[derive(Debug, Default, Clone, Copy)]
pub struct PureNashSolver;

impl PureNashSolver {
    fn is_best_response(
        game: &ValidatedGame,
        profile: &StrategyProfile,
        player: PlayerId,
    ) -> bool {
        let Some(current) = game.payoff(profile, player) else {
            return false;
        };

        (0..game.action_counts[player.0]).all(|action| {
            let mut deviation = profile.clone();
            deviation.0[player.0] = ActionId(action);
            game.payoff(&deviation, player)
                .is_some_and(|candidate| current >= candidate)
        })
    }
}

impl StrategicSolver for PureNashSolver {
    fn method(&self) -> AnalysisMethod {
        AnalysisMethod::ExhaustiveEnumeration
    }

    fn supported_tasks(&self) -> &'static [AnalysisTask] {
        &[AnalysisTask::FindEquilibrium]
    }

    fn solve(
        &self,
        game: &ValidatedGame,
        task: AnalysisTask,
    ) -> Result<StrategicResult, AnalysisError> {
        if task != AnalysisTask::FindEquilibrium {
            return Err(AnalysisError::UnsupportedTask(task));
        }

        let equilibria = game
            .profiles()
            .filter(|profile| {
                (0..game.player_count())
                    .all(|player| Self::is_best_response(game, profile, PlayerId(player)))
            })
            .map(|profile| PureNashEquilibrium { profile })
            .collect();

        Ok(StrategicResult {
            equilibria,
            method: self.method(),
            completeness: Completeness::Exhaustive,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prisoner's_dilemma() -> StrategicProblem {
        // Profiles: (C,C), (C,D), (D,C), (D,D)
        StrategicProblem::new(
            vec![2, 2],
            vec![
                vec![3.0, 3.0],
                vec![0.0, 5.0],
                vec![5.0, 0.0],
                vec![1.0, 1.0],
            ],
        )
    }

    #[test]
    fn validates_profile_table() {
        let game = prisoner's_dilemma().validate().unwrap();
        assert_eq!(game.player_count(), 2);
        assert_eq!(game.profile_count(), 4);
        assert_eq!(
            game.payoff(&StrategyProfile(vec![ActionId(1), ActionId(1)]), PlayerId(0)),
            Some(1.0)
        );
    }

    #[test]
    fn rejects_malformed_payoff_table() {
        let error = StrategicProblem::new(vec![2, 2], vec![vec![0.0, 0.0]])
            .validate()
            .unwrap_err();
        assert!(matches!(
            error,
            ValidationError::PayoffTableSize {
                expected: 4,
                actual: 1
            }
        ));
    }

    #[test]
    fn pure_nash_is_exhaustive_and_deviation_valid() {
        let game = prisoner's_dilemma().validate().unwrap();
        let result = PureNashSolver.solve(&game, AnalysisTask::FindEquilibrium).unwrap();

        assert_eq!(result.completeness, Completeness::Exhaustive);
        assert_eq!(
            result.equilibria,
            vec![PureNashEquilibrium {
                profile: StrategyProfile(vec![ActionId(1), ActionId(1)])
            }]
        );

        for eq in &result.equilibria {
            for player in 0..game.player_count() {
                let base = game.payoff(&eq.profile, PlayerId(player)).unwrap();
                for action in 0..game.action_count(PlayerId(player)).unwrap() {
                    let mut deviation = eq.profile.clone();
                    deviation.0[player] = ActionId(action);
                    assert!(
                        base >= game.payoff(&deviation, PlayerId(player)).unwrap(),
                        "returned equilibrium permits a profitable unilateral deviation"
                    );
                }
            }
        }
    }

    #[test]
    fn player_count_generalizes_beyond_two() {
        let game = StrategicProblem::new(
            vec![2, 2, 2],
            vec![vec![0.0, 0.0, 0.0]; 8],
        )
        .validate()
        .unwrap();

        let result = PureNashSolver.solve(&game, AnalysisTask::FindEquilibrium).unwrap();
        assert_eq!(result.equilibria.len(), 8);
    }

    #[test]
    fn affine_payoff_transform_preserves_pure_equilibria() {
        let base = prisoner's_dilemma().validate().unwrap();
        let transformed = StrategicProblem::new(
            vec![2, 2],
            base
                .profiles()
                .map(|profile| {
                    (0..2)
                        .map(|player| 7.0 * base.payoff(&profile, PlayerId(player)).unwrap() + 11.0)
                        .collect()
                })
                .collect(),
        )
        .validate()
        .unwrap();

        let solver = PureNashSolver;
        let a = solver.solve(&base, AnalysisTask::FindEquilibrium).unwrap();
        let b = solver.solve(&transformed, AnalysisTask::FindEquilibrium).unwrap();
        assert_eq!(a.equilibria, b.equilibria);
    }
}
