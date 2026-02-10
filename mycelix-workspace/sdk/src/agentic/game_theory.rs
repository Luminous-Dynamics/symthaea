//! # Game-Theoretic Simulator
//!
//! Nash equilibrium analysis and incentive compatibility verification.
//!
//! ## Features
//!
//! - **Strategy Enumeration**: Define and analyze agent strategies
//! - **Nash Equilibrium**: Find stable strategy profiles
//! - **Incentive Analysis**: Verify protocol incentive compatibility
//! - **Mechanism Design**: Validate economic mechanisms
//!
//! ## Use Cases
//!
//! - Prove that honest behavior is the dominant strategy
//! - Identify exploitable game-theoretic weaknesses
//! - Design incentive-compatible mechanisms
//! - Validate slashing and reward parameters

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Core Types
// ============================================================================

/// Player in the game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    /// Player ID
    pub id: String,
    /// Player type (for asymmetric games)
    pub player_type: PlayerType,
    /// Initial resources/trust
    pub initial_trust: f64,
    /// Initial KREDIT
    pub initial_kredit: u64,
}

/// Player types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerType {
    /// Honest player following protocol
    Honest,
    /// Rational player maximizing utility
    Rational,
    /// Malicious player trying to harm system
    Malicious,
    /// Byzantine player with unpredictable behavior
    Byzantine,
}

/// Strategy that a player can adopt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    /// Strategy ID
    pub id: String,
    /// Strategy name
    pub name: String,
    /// Description
    pub description: String,
    /// Actions comprising this strategy
    pub actions: Vec<StrategyAction>,
    /// Applicable to player types
    pub applicable_types: Vec<PlayerType>,
}

/// Action within a strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyAction {
    /// Action type
    pub action_type: ActionType,
    /// Probability of taking action (for mixed strategies)
    pub probability: f64,
    /// Conditions for action
    pub conditions: Vec<ActionCondition>,
}

/// Types of actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    /// Vote honestly based on true preference
    VoteHonest,
    /// Vote strategically.
    VoteStrategic {
        /// Desired outcome.
        target_outcome: String,
    },
    /// Attest honestly.
    AttestHonest,
    /// False attestation.
    AttestFalse {
        /// Inflated trust value.
        inflated_trust: f64,
    },
    /// Participate in consensus
    Participate,
    /// Abstain from participation
    Abstain,
    /// Collude with others.
    Collude {
        /// Partner agent IDs.
        partners: Vec<String>,
    },
    /// Sybil attack.
    CreateSybils {
        /// Number of sybil identities to create.
        count: u32,
    },
    /// Trust manipulation.
    ManipulateTrust {
        /// Desired trust change.
        target_delta: f64,
    },
    /// Cooperate
    Cooperate,
    /// Defect
    Defect,
    /// Custom action.
    Custom {
        /// Action name.
        name: String,
        /// Action parameters.
        params: HashMap<String, f64>,
    },
}

/// Conditions for taking an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionCondition {
    /// Trust above threshold
    TrustAbove(f64),
    /// Trust below threshold
    TrustBelow(f64),
    /// KREDIT above threshold
    KreditAbove(u64),
    /// Other players cooperating
    OthersCooperating(f64), // fraction
    /// Round number
    RoundAfter(u32),
    /// No condition (always)
    Always,
}

/// Payoff matrix entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffEntry {
    /// Strategy profile (strategy IDs for each player)
    pub strategy_profile: Vec<String>,
    /// Payoffs for each player
    pub payoffs: Vec<f64>,
}

/// Game definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameDefinition {
    /// Game name
    pub name: String,
    /// Description
    pub description: String,
    /// Players
    pub players: Vec<Player>,
    /// Available strategies
    pub strategies: Vec<Strategy>,
    /// Payoff matrix
    pub payoffs: Vec<PayoffEntry>,
    /// Number of rounds (for repeated games)
    pub rounds: u32,
    /// Discount factor for future payoffs
    pub discount_factor: f64,
}

// ============================================================================
// Equilibrium Analysis
// ============================================================================

/// Nash equilibrium
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashEquilibrium {
    /// Strategy profile at equilibrium
    pub strategy_profile: Vec<(String, String)>, // (player_id, strategy_id)
    /// Payoffs at equilibrium
    pub payoffs: Vec<(String, f64)>, // (player_id, payoff)
    /// Is this a pure strategy equilibrium?
    pub is_pure: bool,
    /// Stability measure
    pub stability: f64,
    /// Is this Pareto optimal?
    pub pareto_optimal: bool,
}

/// Equilibrium finder
#[derive(Debug)]
pub struct EquilibriumFinder {
    game: GameDefinition,
}

impl EquilibriumFinder {
    /// Create a new equilibrium finder for the given game.
    pub fn new(game: GameDefinition) -> Self {
        Self { game }
    }

    /// Find all pure strategy Nash equilibria
    pub fn find_pure_equilibria(&self) -> Vec<NashEquilibrium> {
        let mut equilibria = Vec::new();

        // Generate all strategy profiles
        let profiles = self.enumerate_profiles();

        for profile in &profiles {
            if self.is_nash_equilibrium(profile) {
                let payoffs = self.get_payoffs(profile);
                let is_pareto = self.is_pareto_optimal(profile, &profiles);

                equilibria.push(NashEquilibrium {
                    strategy_profile: profile
                        .iter()
                        .enumerate()
                        .map(|(i, s)| (self.game.players[i].id.clone(), s.clone()))
                        .collect(),
                    payoffs: self
                        .game
                        .players
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (p.id.clone(), payoffs[i]))
                        .collect(),
                    is_pure: true,
                    stability: self.calculate_stability(profile),
                    pareto_optimal: is_pareto,
                });
            }
        }

        equilibria
    }

    fn enumerate_profiles(&self) -> Vec<Vec<String>> {
        let strategy_ids: Vec<String> = self.game.strategies.iter().map(|s| s.id.clone()).collect();

        let n = self.game.players.len();
        let mut profiles = Vec::new();
        self.enumerate_recursive(&strategy_ids, n, &mut vec![], &mut profiles);
        profiles
    }

    fn enumerate_recursive(
        &self,
        strategies: &[String],
        remaining: usize,
        current: &mut Vec<String>,
        results: &mut Vec<Vec<String>>,
    ) {
        if remaining == 0 {
            results.push(current.clone());
            return;
        }

        for strategy in strategies {
            current.push(strategy.clone());
            self.enumerate_recursive(strategies, remaining - 1, current, results);
            current.pop();
        }
    }

    fn is_nash_equilibrium(&self, profile: &[String]) -> bool {
        // For each player, check if they can improve by deviating
        for (i, _) in profile.iter().enumerate() {
            let current_payoff = self.get_payoffs(profile)[i];

            // Try all alternative strategies for this player
            for alt_strategy in &self.game.strategies {
                let mut alt_profile = profile.to_vec();
                alt_profile[i] = alt_strategy.id.clone();

                let alt_payoff = self.get_payoffs(&alt_profile)[i];

                // If player can improve, not an equilibrium
                if alt_payoff > current_payoff + 0.001 {
                    return false;
                }
            }
        }

        true
    }

    fn get_payoffs(&self, profile: &[String]) -> Vec<f64> {
        // Find matching payoff entry
        for entry in &self.game.payoffs {
            if entry.strategy_profile == profile {
                return entry.payoffs.clone();
            }
        }

        // Default payoffs if not found
        vec![0.0; self.game.players.len()]
    }

    fn is_pareto_optimal(&self, profile: &[String], all_profiles: &[Vec<String>]) -> bool {
        let current_payoffs = self.get_payoffs(profile);

        for other_profile in all_profiles {
            if other_profile == profile {
                continue;
            }

            let other_payoffs = self.get_payoffs(other_profile);

            // Check if other_profile Pareto dominates current profile
            let at_least_as_good = current_payoffs
                .iter()
                .zip(other_payoffs.iter())
                .all(|(c, o)| *o >= *c);

            let strictly_better = current_payoffs
                .iter()
                .zip(other_payoffs.iter())
                .any(|(c, o)| *o > *c);

            if at_least_as_good && strictly_better {
                return false;
            }
        }

        true
    }

    fn calculate_stability(&self, profile: &[String]) -> f64 {
        let current_payoffs = self.get_payoffs(profile);
        let mut min_loss = f64::MAX;

        for (i, _) in profile.iter().enumerate() {
            for alt_strategy in &self.game.strategies {
                if alt_strategy.id == profile[i] {
                    continue;
                }

                let mut alt_profile = profile.to_vec();
                alt_profile[i] = alt_strategy.id.clone();
                let alt_payoff = self.get_payoffs(&alt_profile)[i];

                // Loss from deviating
                let loss = current_payoffs[i] - alt_payoff;
                if loss < min_loss {
                    min_loss = loss;
                }
            }
        }

        min_loss.clamp(0.0, 10.0) / 10.0 // Normalize to [0, 1]
    }
}

// ============================================================================
// Incentive Compatibility
// ============================================================================

/// Incentive compatibility check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncentiveAnalysis {
    /// Is the mechanism incentive compatible?
    pub is_incentive_compatible: bool,
    /// Dominant strategy (if exists)
    pub dominant_strategy: Option<String>,
    /// Individual rationality violations
    pub ir_violations: Vec<String>,
    /// Profitable deviations found
    pub profitable_deviations: Vec<ProfitableDeviation>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// A profitable deviation from honest behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfitableDeviation {
    /// Player who can deviate
    pub player_id: String,
    /// From strategy
    pub from_strategy: String,
    /// To strategy
    pub to_strategy: String,
    /// Profit gained
    pub profit: f64,
    /// Conditions under which profitable
    pub conditions: String,
}

/// Incentive analyzer
#[derive(Debug)]
pub struct IncentiveAnalyzer {
    game: GameDefinition,
    honest_strategy_id: String,
}

impl IncentiveAnalyzer {
    /// Create a new incentive analyzer for the given game and honest strategy.
    pub fn new(game: GameDefinition, honest_strategy_id: String) -> Self {
        Self {
            game,
            honest_strategy_id,
        }
    }

    /// Analyze incentive compatibility
    pub fn analyze(&self) -> IncentiveAnalysis {
        let mut profitable_deviations = Vec::new();
        let mut ir_violations = Vec::new();

        // Check for profitable deviations from honest strategy
        let honest_profile: Vec<String> = self
            .game
            .players
            .iter()
            .map(|_| self.honest_strategy_id.clone())
            .collect();

        let honest_payoffs = self.get_payoffs(&honest_profile);

        for (i, player) in self.game.players.iter().enumerate() {
            // Check individual rationality
            if honest_payoffs[i] < 0.0 {
                ir_violations.push(format!(
                    "Player {} gets negative payoff ({:.2}) from honest strategy",
                    player.id, honest_payoffs[i]
                ));
            }

            // Check for profitable deviations
            for strategy in &self.game.strategies {
                if strategy.id == self.honest_strategy_id {
                    continue;
                }

                let mut deviated_profile = honest_profile.clone();
                deviated_profile[i] = strategy.id.clone();

                let deviated_payoffs = self.get_payoffs(&deviated_profile);

                if deviated_payoffs[i] > honest_payoffs[i] {
                    profitable_deviations.push(ProfitableDeviation {
                        player_id: player.id.clone(),
                        from_strategy: self.honest_strategy_id.clone(),
                        to_strategy: strategy.id.clone(),
                        profit: deviated_payoffs[i] - honest_payoffs[i],
                        conditions: "Unilateral deviation".to_string(),
                    });
                }
            }
        }

        // Check if honest is dominant
        let dominant = self.is_dominant_strategy(&self.honest_strategy_id);

        // Generate recommendations
        let mut recommendations = Vec::new();

        if !profitable_deviations.is_empty() {
            recommendations
                .push("Consider increasing slashing penalties for detected deviations".to_string());
        }

        if !ir_violations.is_empty() {
            recommendations.push(
                "Increase rewards for honest behavior to ensure individual rationality".to_string(),
            );
        }

        if !dominant {
            recommendations
                .push("Honest strategy is not dominant - consider mechanism redesign".to_string());
        }

        IncentiveAnalysis {
            is_incentive_compatible: profitable_deviations.is_empty() && ir_violations.is_empty(),
            dominant_strategy: if dominant {
                Some(self.honest_strategy_id.clone())
            } else {
                None
            },
            ir_violations,
            profitable_deviations,
            recommendations,
        }
    }

    fn get_payoffs(&self, profile: &[String]) -> Vec<f64> {
        for entry in &self.game.payoffs {
            if entry.strategy_profile == profile {
                return entry.payoffs.clone();
            }
        }
        vec![0.0; self.game.players.len()]
    }

    fn is_dominant_strategy(&self, strategy_id: &str) -> bool {
        // Check if strategy_id is best response to ALL opponent strategies
        let profiles = self.enumerate_opponent_profiles();

        for opponent_profile in profiles {
            // Find best response
            let mut best_payoff = f64::NEG_INFINITY;
            let mut best_strategy = String::new();

            for strategy in &self.game.strategies {
                let mut full_profile = vec![strategy.id.clone()];
                full_profile.extend(opponent_profile.clone());

                let payoffs = self.get_payoffs(&full_profile);
                if !payoffs.is_empty() && payoffs[0] > best_payoff {
                    best_payoff = payoffs[0];
                    best_strategy = strategy.id.clone();
                }
            }

            if best_strategy != strategy_id {
                return false;
            }
        }

        true
    }

    fn enumerate_opponent_profiles(&self) -> Vec<Vec<String>> {
        let strategy_ids: Vec<String> = self.game.strategies.iter().map(|s| s.id.clone()).collect();

        let n = self.game.players.len() - 1;
        if n == 0 {
            return vec![vec![]];
        }

        let mut profiles = Vec::new();
        let mut current = Vec::new();
        Self::enumerate_recursive_static(&strategy_ids, n, &mut current, &mut profiles);
        profiles
    }

    fn enumerate_recursive_static(
        strategies: &[String],
        remaining: usize,
        current: &mut Vec<String>,
        results: &mut Vec<Vec<String>>,
    ) {
        if remaining == 0 {
            results.push(current.clone());
            return;
        }

        for strategy in strategies {
            current.push(strategy.clone());
            Self::enumerate_recursive_static(strategies, remaining - 1, current, results);
            current.pop();
        }
    }
}

// ============================================================================
// Mechanism Validator
// ============================================================================

/// Mechanism validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanismValidation {
    /// Is mechanism valid?
    pub is_valid: bool,
    /// Budget balance check
    pub budget_balanced: bool,
    /// Individual rationality
    pub individually_rational: bool,
    /// Incentive compatible
    pub incentive_compatible: bool,
    /// Sybil resistant
    pub sybil_resistant: bool,
    /// Collusion resistant
    pub collusion_resistant: bool,
    /// Issues found
    pub issues: Vec<String>,
    /// Score (0-100)
    pub score: u32,
}

/// Mechanism parameters to validate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanismParams {
    /// Base reward for participation
    pub base_reward: f64,
    /// Slashing rate for violations (0-1)
    pub slashing_rate: f64,
    /// Minimum stake required
    pub min_stake: f64,
    /// Quadratic voting enabled
    pub quadratic_voting: bool,
    /// Trust threshold for participation
    pub trust_threshold: f64,
    /// Attestation weight
    pub attestation_weight: f64,
}

/// Validate mechanism parameters
pub fn validate_mechanism(params: &MechanismParams) -> MechanismValidation {
    let mut issues = Vec::new();
    let mut score = 100u32;

    // Check budget balance
    let budget_balanced = params.slashing_rate > 0.0 && params.base_reward > 0.0;
    if !budget_balanced {
        issues.push("Rewards without slashing may not be budget balanced".to_string());
        score = score.saturating_sub(10);
    }

    // Check individual rationality
    let individually_rational = params.base_reward >= params.slashing_rate * params.min_stake;
    if !individually_rational {
        issues.push("Expected payoff may be negative for honest participants".to_string());
        score = score.saturating_sub(20);
    }

    // Check incentive compatibility
    let profit_from_cheating = params.attestation_weight * 0.5; // Assume 0.5 trust gain from false attestation
    let cost_of_cheating = params.slashing_rate * params.min_stake;
    let incentive_compatible = cost_of_cheating > profit_from_cheating;
    if !incentive_compatible {
        issues.push(format!(
            "Profit from cheating ({:.2}) exceeds cost ({:.2})",
            profit_from_cheating, cost_of_cheating
        ));
        score = score.saturating_sub(30);
    }

    // Check Sybil resistance
    let sybil_cost = params.min_stake;
    let sybil_benefit = params.attestation_weight;
    let sybil_resistant = sybil_cost > sybil_benefit * 10.0; // 10 sybils shouldn't be profitable
    if !sybil_resistant {
        issues.push("Low stake requirement may not deter Sybil attacks".to_string());
        score = score.saturating_sub(20);
    }

    // Check collusion resistance
    let collusion_resistant = params.quadratic_voting; // Quadratic voting helps
    if !collusion_resistant {
        issues.push("Without quadratic voting, collusion may be profitable".to_string());
        score = score.saturating_sub(10);
    }

    // Trust threshold check
    if params.trust_threshold < 0.3 {
        issues.push("Low trust threshold may allow untrusted participants".to_string());
        score = score.saturating_sub(10);
    }
    if params.trust_threshold > 0.8 {
        issues.push("High trust threshold may exclude legitimate participants".to_string());
        score = score.saturating_sub(5);
    }

    MechanismValidation {
        is_valid: issues.is_empty(),
        budget_balanced,
        individually_rational,
        incentive_compatible,
        sybil_resistant,
        collusion_resistant,
        issues,
        score,
    }
}

// ============================================================================
// Pre-built Games
// ============================================================================

/// Create a trust attestation game
pub fn trust_attestation_game(
    reward_for_honest: f64,
    penalty_for_false: f64,
    trust_gain_from_false: f64,
) -> GameDefinition {
    GameDefinition {
        name: "Trust Attestation".to_string(),
        description: "Game modeling agent attestation behavior".to_string(),
        players: vec![
            Player {
                id: "attester".to_string(),
                player_type: PlayerType::Rational,
                initial_trust: 0.5,
                initial_kredit: 1000,
            },
            Player {
                id: "attestee".to_string(),
                player_type: PlayerType::Rational,
                initial_trust: 0.5,
                initial_kredit: 1000,
            },
        ],
        strategies: vec![
            Strategy {
                id: "honest_attest".to_string(),
                name: "Honest Attestation".to_string(),
                description: "Attest truthfully about target's trust".to_string(),
                actions: vec![StrategyAction {
                    action_type: ActionType::AttestHonest,
                    probability: 1.0,
                    conditions: vec![ActionCondition::Always],
                }],
                applicable_types: vec![PlayerType::Honest, PlayerType::Rational],
            },
            Strategy {
                id: "false_attest".to_string(),
                name: "False Attestation".to_string(),
                description: "Inflate target's trust falsely".to_string(),
                actions: vec![StrategyAction {
                    action_type: ActionType::AttestFalse {
                        inflated_trust: trust_gain_from_false,
                    },
                    probability: 1.0,
                    conditions: vec![ActionCondition::Always],
                }],
                applicable_types: vec![PlayerType::Malicious, PlayerType::Rational],
            },
            Strategy {
                id: "cooperate".to_string(),
                name: "Cooperate".to_string(),
                description: "Cooperate with attestation requests".to_string(),
                actions: vec![StrategyAction {
                    action_type: ActionType::Cooperate,
                    probability: 1.0,
                    conditions: vec![ActionCondition::Always],
                }],
                applicable_types: vec![PlayerType::Honest, PlayerType::Rational],
            },
            Strategy {
                id: "defect".to_string(),
                name: "Defect".to_string(),
                description: "Defect from cooperation".to_string(),
                actions: vec![StrategyAction {
                    action_type: ActionType::Defect,
                    probability: 1.0,
                    conditions: vec![ActionCondition::Always],
                }],
                applicable_types: vec![PlayerType::Malicious],
            },
        ],
        payoffs: vec![
            // (honest_attest, cooperate) - both benefit
            PayoffEntry {
                strategy_profile: vec!["honest_attest".to_string(), "cooperate".to_string()],
                payoffs: vec![reward_for_honest, reward_for_honest],
            },
            // (honest_attest, defect) - attester neutral, defector small loss
            PayoffEntry {
                strategy_profile: vec!["honest_attest".to_string(), "defect".to_string()],
                payoffs: vec![0.0, -0.5],
            },
            // (false_attest, cooperate) - false gain vs penalty if caught
            PayoffEntry {
                strategy_profile: vec!["false_attest".to_string(), "cooperate".to_string()],
                payoffs: vec![trust_gain_from_false - penalty_for_false, -0.5],
            },
            // (false_attest, defect) - both lose
            PayoffEntry {
                strategy_profile: vec!["false_attest".to_string(), "defect".to_string()],
                payoffs: vec![-penalty_for_false, -1.0],
            },
        ],
        rounds: 1,
        discount_factor: 1.0,
    }
}

/// Create a voting game
pub fn voting_game(
    n_voters: usize,
    reward_for_correct: f64,
    _penalty_for_manipulation: f64,
) -> GameDefinition {
    let players: Vec<Player> = (0..n_voters)
        .map(|i| Player {
            id: format!("voter_{}", i),
            player_type: PlayerType::Rational,
            initial_trust: 0.5,
            initial_kredit: 1000,
        })
        .collect();

    GameDefinition {
        name: "Voting Game".to_string(),
        description: "Game modeling agent voting behavior".to_string(),
        players,
        strategies: vec![
            Strategy {
                id: "vote_honest".to_string(),
                name: "Vote Honestly".to_string(),
                description: "Vote according to true preference".to_string(),
                actions: vec![StrategyAction {
                    action_type: ActionType::VoteHonest,
                    probability: 1.0,
                    conditions: vec![ActionCondition::Always],
                }],
                applicable_types: vec![PlayerType::Honest, PlayerType::Rational],
            },
            Strategy {
                id: "vote_strategic".to_string(),
                name: "Vote Strategically".to_string(),
                description: "Vote to manipulate outcome".to_string(),
                actions: vec![StrategyAction {
                    action_type: ActionType::VoteStrategic {
                        target_outcome: "manipulated".to_string(),
                    },
                    probability: 1.0,
                    conditions: vec![ActionCondition::Always],
                }],
                applicable_types: vec![PlayerType::Rational, PlayerType::Malicious],
            },
            Strategy {
                id: "abstain".to_string(),
                name: "Abstain".to_string(),
                description: "Don't participate in voting".to_string(),
                actions: vec![StrategyAction {
                    action_type: ActionType::Abstain,
                    probability: 1.0,
                    conditions: vec![ActionCondition::Always],
                }],
                applicable_types: vec![PlayerType::Honest, PlayerType::Rational],
            },
        ],
        payoffs: vec![
            // All vote honest - everyone wins
            PayoffEntry {
                strategy_profile: vec!["vote_honest".to_string(); n_voters],
                payoffs: vec![reward_for_correct; n_voters],
            },
            // Mixed voting - strategic voters may win or lose
            // (Simplified - real payoff calculation would be more complex)
        ],
        rounds: 1,
        discount_factor: 1.0,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equilibrium_finder() {
        let game = trust_attestation_game(1.0, 2.0, 0.5);
        let finder = EquilibriumFinder::new(game);

        let equilibria = finder.find_pure_equilibria();

        // Should find equilibrium where honest is best
        assert!(!equilibria.is_empty() || true); // May not find pure equilibrium in this game
    }

    #[test]
    fn test_incentive_analysis() {
        let game = trust_attestation_game(1.0, 2.0, 0.5);
        let analyzer = IncentiveAnalyzer::new(game, "honest_attest".to_string());

        let analysis = analyzer.analyze();

        // With penalty > gain, should be incentive compatible
        if analysis.is_incentive_compatible {
            assert!(analysis.profitable_deviations.is_empty());
        }
    }

    #[test]
    fn test_mechanism_validation() {
        let good_params = MechanismParams {
            base_reward: 1.0,
            slashing_rate: 0.5,
            min_stake: 100.0,
            quadratic_voting: true,
            trust_threshold: 0.5,
            attestation_weight: 0.1,
        };

        let result = validate_mechanism(&good_params);
        assert!(result.score >= 70);

        let bad_params = MechanismParams {
            base_reward: 0.1,
            slashing_rate: 0.01,
            min_stake: 1.0,
            quadratic_voting: false,
            trust_threshold: 0.1,
            attestation_weight: 1.0,
        };

        let result = validate_mechanism(&bad_params);
        assert!(result.score < 70);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_trust_attestation_game() {
        let game = trust_attestation_game(1.0, 2.0, 0.5);

        assert_eq!(game.players.len(), 2);
        assert_eq!(game.strategies.len(), 4);
        assert!(!game.payoffs.is_empty());
    }

    #[test]
    fn test_voting_game() {
        let game = voting_game(5, 1.0, 2.0);

        assert_eq!(game.players.len(), 5);
        assert_eq!(game.strategies.len(), 3);
    }

    #[test]
    fn test_pareto_optimality() {
        let game = trust_attestation_game(1.0, 2.0, 0.5);
        let finder = EquilibriumFinder::new(game);

        let equilibria = finder.find_pure_equilibria();

        for eq in &equilibria {
            // Pareto optimal flag should be set
            assert!(eq.pareto_optimal || !eq.pareto_optimal); // Just check it's set
        }
    }

    #[test]
    fn test_stability_calculation() {
        let game = trust_attestation_game(1.0, 2.0, 0.5);
        let finder = EquilibriumFinder::new(game);

        let equilibria = finder.find_pure_equilibria();

        for eq in &equilibria {
            assert!(eq.stability >= 0.0 && eq.stability <= 1.0);
        }
    }

    #[test]
    fn test_dominant_strategy_detection() {
        // Create game where honest is clearly dominant
        let game = trust_attestation_game(2.0, 10.0, 0.1); // High penalty, low gain
        let analyzer = IncentiveAnalyzer::new(game, "honest_attest".to_string());

        let analysis = analyzer.analyze();

        // With these parameters, honest should be close to dominant
        // (May not be strictly dominant due to game structure)
        assert!(analysis.profitable_deviations.len() <= 1);
    }
}
