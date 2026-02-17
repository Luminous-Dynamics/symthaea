//! Vote Weight Calculation
//!
//! Calculates voting power based on:
//! - MATL trust score
//! - Stake holdings
//! - Participation history
//! - FL contribution record
//! - Identity assurance level
//!
//! ## Weight Formula
//!
//! ```text
//! weight = base_stake × MATL_multiplier × participation_bonus × assurance_factor
//!
//! where:
//!   MATL_multiplier = 0.5 + (MATL_score × 0.5)  // Range: 0.5-1.0
//!   participation_bonus = 1.0 + 0.1 × recent_participation_rate  // Range: 1.0-1.1
//!   assurance_factor = 0.7 + 0.1 × assurance_level  // Range: 0.7-1.1
//! ```

use std::collections::HashMap;

/// Vote weight configuration
#[derive(Debug, Clone)]
pub struct WeightConfig {
    /// Minimum base weight for any voter
    pub min_base_weight: f32,

    /// Maximum weight cap (prevents plutocracy)
    pub max_weight_cap: f32,

    /// Weight contribution from MATL score (0.0-1.0)
    pub matl_weight_factor: f32,

    /// Weight contribution from stake (0.0-1.0)
    pub stake_weight_factor: f32,

    /// Maximum participation bonus
    pub max_participation_bonus: f32,

    /// Enable quadratic voting (sqrt of stake)
    pub quadratic_voting: bool,

    /// FL contribution weight bonus
    pub fl_contribution_bonus: f32,

    /// Assurance level weight factors
    pub assurance_factors: [f32; 5], // E0-E4
}

impl Default for WeightConfig {
    fn default() -> Self {
        Self {
            min_base_weight: 1.0,
            max_weight_cap: 10000.0,
            matl_weight_factor: 0.4,
            stake_weight_factor: 0.4,
            max_participation_bonus: 0.1,
            quadratic_voting: false,
            fl_contribution_bonus: 0.02, // 2% bonus per 10 contributions
            assurance_factors: [0.7, 0.8, 0.9, 1.0, 1.1], // E0-E4
        }
    }
}

impl WeightConfig {
    /// Configuration for token-weighted voting
    pub fn token_weighted() -> Self {
        Self {
            stake_weight_factor: 0.7,
            matl_weight_factor: 0.2,
            quadratic_voting: false,
            ..Default::default()
        }
    }

    /// Configuration for quadratic voting
    pub fn quadratic() -> Self {
        Self {
            stake_weight_factor: 0.5,
            matl_weight_factor: 0.3,
            quadratic_voting: true,
            ..Default::default()
        }
    }

    /// Configuration for reputation-weighted voting
    pub fn reputation_weighted() -> Self {
        Self {
            stake_weight_factor: 0.2,
            matl_weight_factor: 0.6,
            fl_contribution_bonus: 0.05,
            ..Default::default()
        }
    }
}

/// Input for weight calculation
#[derive(Debug, Clone)]
pub struct WeightInput {
    /// Voter DID
    pub did: String,

    /// Base stake amount
    pub stake: f32,

    /// MATL trust score (0.0-1.0)
    pub matl_score: f32,

    /// Assurance level (0-4)
    pub assurance_level: u8,

    /// Recent participation rate (0.0-1.0)
    pub participation_rate: f32,

    /// FL contribution count
    pub fl_contributions: u32,

    /// Optional delegated weight
    pub delegated_weight: f32,
}

impl WeightInput {
    /// Create a new weight input
    pub fn new(did: impl Into<String>, stake: f32, matl_score: f32) -> Self {
        Self {
            did: did.into(),
            stake,
            matl_score,
            assurance_level: 1,
            participation_rate: 0.0,
            fl_contributions: 0,
            delegated_weight: 0.0,
        }
    }

    /// Set assurance level
    pub fn with_assurance(mut self, level: u8) -> Self {
        self.assurance_level = level.min(4);
        self
    }

    /// Set participation rate
    pub fn with_participation(mut self, rate: f32) -> Self {
        self.participation_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set FL contributions
    pub fn with_fl_contributions(mut self, count: u32) -> Self {
        self.fl_contributions = count;
        self
    }

    /// Set delegated weight
    pub fn with_delegated(mut self, weight: f32) -> Self {
        self.delegated_weight = weight;
        self
    }
}

/// Result of weight calculation
#[derive(Debug, Clone)]
pub struct WeightResult {
    /// Voter DID
    pub did: String,

    /// Final calculated weight
    pub total_weight: f32,

    /// Component breakdown
    pub components: WeightComponents,

    /// Whether weight was capped
    pub was_capped: bool,

    /// Original uncapped weight
    pub uncapped_weight: f32,
}

/// Breakdown of weight components
#[derive(Debug, Clone, Default)]
pub struct WeightComponents {
    /// Base weight from stake
    pub base_weight: f32,

    /// MATL multiplier applied
    pub matl_multiplier: f32,

    /// Participation bonus
    pub participation_bonus: f32,

    /// Assurance factor
    pub assurance_factor: f32,

    /// FL contribution bonus
    pub fl_bonus: f32,

    /// Delegated weight added
    pub delegated_weight: f32,
}

/// Vote weight calculator
#[derive(Debug, Clone)]
pub struct WeightCalculator {
    config: WeightConfig,
}

impl WeightCalculator {
    /// Create a new weight calculator
    pub fn new(config: WeightConfig) -> Self {
        Self { config }
    }

    /// Calculate vote weight for a single voter
    pub fn calculate(&self, input: &WeightInput) -> WeightResult {
        let mut components = WeightComponents::default();

        // Calculate base weight from stake
        let base_stake = if self.config.quadratic_voting {
            input.stake.sqrt()
        } else {
            input.stake
        };
        components.base_weight = base_stake.max(self.config.min_base_weight);

        // Calculate MATL multiplier (0.5-1.0 range)
        components.matl_multiplier = 0.5 + (input.matl_score.clamp(0.0, 1.0) * 0.5);

        // Calculate participation bonus
        components.participation_bonus =
            1.0 + (input.participation_rate * self.config.max_participation_bonus);

        // Get assurance factor
        let assurance_idx = (input.assurance_level as usize).min(4);
        components.assurance_factor = self.config.assurance_factors[assurance_idx];

        // Calculate FL contribution bonus
        let fl_bonus_rate = (input.fl_contributions / 10) as f32 * self.config.fl_contribution_bonus;
        components.fl_bonus = fl_bonus_rate.min(0.2); // Cap at 20% bonus

        // Add delegated weight
        components.delegated_weight = input.delegated_weight;

        // Calculate weighted components
        let stake_component = components.base_weight * self.config.stake_weight_factor;
        let matl_component =
            components.base_weight * self.config.matl_weight_factor * components.matl_multiplier;

        // Combine components
        let base_total = stake_component + matl_component;
        let with_bonuses = base_total
            * components.participation_bonus
            * components.assurance_factor
            * (1.0 + components.fl_bonus);

        let uncapped_weight = with_bonuses + components.delegated_weight;

        // Apply cap
        let total_weight = uncapped_weight.min(self.config.max_weight_cap);
        let was_capped = uncapped_weight > self.config.max_weight_cap;

        WeightResult {
            did: input.did.clone(),
            total_weight,
            components,
            was_capped,
            uncapped_weight,
        }
    }

    /// Calculate weights for multiple voters
    pub fn calculate_batch(&self, inputs: &[WeightInput]) -> Vec<WeightResult> {
        inputs.iter().map(|input| self.calculate(input)).collect()
    }

    /// Calculate total weight for a set of voters
    pub fn total_weight(&self, inputs: &[WeightInput]) -> f32 {
        self.calculate_batch(inputs)
            .iter()
            .map(|r| r.total_weight)
            .sum()
    }

    /// Get weight distribution statistics
    pub fn weight_statistics(&self, inputs: &[WeightInput]) -> WeightStatistics {
        let results = self.calculate_batch(inputs);

        if results.is_empty() {
            return WeightStatistics::default();
        }

        let weights: Vec<f32> = results.iter().map(|r| r.total_weight).collect();
        let total: f32 = weights.iter().sum();
        let count = weights.len() as f32;

        let mean = total / count;
        let variance: f32 =
            weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / count;
        let std_dev = variance.sqrt();

        let mut sorted = weights.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let capped_count = results.iter().filter(|r| r.was_capped).count();

        // Calculate Gini coefficient
        let n = weights.len() as f32;
        let sum_of_differences: f32 = weights
            .iter()
            .flat_map(|wi| weights.iter().map(move |wj| (wi - wj).abs()))
            .sum();
        let gini = sum_of_differences / (2.0 * n * total);

        WeightStatistics {
            total_weight: total,
            mean_weight: mean,
            median_weight: median,
            std_deviation: std_dev,
            min_weight: *sorted.first().unwrap_or(&0.0),
            max_weight: *sorted.last().unwrap_or(&0.0),
            capped_count,
            gini_coefficient: gini,
        }
    }
}

/// Weight distribution statistics
#[derive(Debug, Clone, Default)]
pub struct WeightStatistics {
    pub total_weight: f32,
    pub mean_weight: f32,
    pub median_weight: f32,
    pub std_deviation: f32,
    pub min_weight: f32,
    pub max_weight: f32,
    pub capped_count: usize,
    pub gini_coefficient: f32,
}

/// Calculate vote weight using the standard formula
/// This is a convenience function for simple use cases
pub fn calculate_vote_weight(matl_score: f32, stake: f32, participation_rate: f32) -> f32 {
    let matl_multiplier = 0.5 + (matl_score.clamp(0.0, 1.0) * 0.5);
    let participation_bonus = 1.0 + (participation_rate * 0.1);

    stake.max(1.0) * matl_multiplier * participation_bonus
}

/// Calculate normalized weight (0.0-1.0) relative to total
pub fn normalize_weight(weight: f32, total_weight: f32) -> f32 {
    if total_weight > 0.0 {
        weight / total_weight
    } else {
        0.0
    }
}

/// Build weight lookup map from voter data
pub fn build_weight_map(inputs: &[WeightInput], config: &WeightConfig) -> HashMap<String, f32> {
    let calculator = WeightCalculator::new(config.clone());
    calculator
        .calculate_batch(inputs)
        .into_iter()
        .map(|r| (r.did, r.total_weight))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_weight_calculation() {
        let calculator = WeightCalculator::new(WeightConfig::default());

        let input = WeightInput::new("did:mycelix:alice", 100.0, 0.8)
            .with_assurance(2)
            .with_participation(0.5);

        let result = calculator.calculate(&input);

        assert!(result.total_weight > 0.0);
        assert!(!result.was_capped);
        assert_eq!(result.did, "did:mycelix:alice");
    }

    #[test]
    fn test_matl_affects_weight() {
        let calculator = WeightCalculator::new(WeightConfig::default());

        let high_matl = WeightInput::new("did:mycelix:high", 100.0, 0.9);
        let low_matl = WeightInput::new("did:mycelix:low", 100.0, 0.3);

        let high_result = calculator.calculate(&high_matl);
        let low_result = calculator.calculate(&low_matl);

        assert!(high_result.total_weight > low_result.total_weight);
    }

    #[test]
    fn test_quadratic_voting() {
        let linear_calc = WeightCalculator::new(WeightConfig::default());
        let quadratic_calc = WeightCalculator::new(WeightConfig::quadratic());

        let input = WeightInput::new("did:mycelix:alice", 10000.0, 0.5);

        let linear_result = linear_calc.calculate(&input);
        let quadratic_result = quadratic_calc.calculate(&input);

        // Quadratic should give less weight for high stake
        assert!(quadratic_result.total_weight < linear_result.total_weight);
    }

    #[test]
    fn test_weight_capping() {
        let config = WeightConfig {
            max_weight_cap: 100.0,
            ..Default::default()
        };
        let calculator = WeightCalculator::new(config);

        let input = WeightInput::new("did:mycelix:whale", 100000.0, 1.0);

        let result = calculator.calculate(&input);

        assert!(result.was_capped);
        assert_eq!(result.total_weight, 100.0);
        assert!(result.uncapped_weight > 100.0);
    }

    #[test]
    fn test_participation_bonus() {
        let calculator = WeightCalculator::new(WeightConfig::default());

        let active_voter = WeightInput::new("did:mycelix:active", 100.0, 0.7).with_participation(1.0);
        let inactive_voter =
            WeightInput::new("did:mycelix:inactive", 100.0, 0.7).with_participation(0.0);

        let active_result = calculator.calculate(&active_voter);
        let inactive_result = calculator.calculate(&inactive_voter);

        assert!(active_result.total_weight > inactive_result.total_weight);
        assert!(active_result.components.participation_bonus > 1.0);
    }

    #[test]
    fn test_fl_contribution_bonus() {
        let calculator = WeightCalculator::new(WeightConfig::default());

        let contributor = WeightInput::new("did:mycelix:contrib", 100.0, 0.7).with_fl_contributions(50);
        let non_contributor =
            WeightInput::new("did:mycelix:non", 100.0, 0.7).with_fl_contributions(0);

        let contrib_result = calculator.calculate(&contributor);
        let non_result = calculator.calculate(&non_contributor);

        assert!(contrib_result.total_weight > non_result.total_weight);
        assert!(contrib_result.components.fl_bonus > 0.0);
    }

    #[test]
    fn test_convenience_function() {
        let weight = calculate_vote_weight(0.8, 100.0, 0.5);

        // MATL multiplier: 0.5 + 0.8*0.5 = 0.9
        // Participation bonus: 1.0 + 0.5*0.1 = 1.05
        // Expected: 100 * 0.9 * 1.05 = 94.5
        assert!((weight - 94.5).abs() < 0.01);
    }

    #[test]
    fn test_weight_statistics() {
        let calculator = WeightCalculator::new(WeightConfig::default());

        let inputs = vec![
            WeightInput::new("did:1", 100.0, 0.5),
            WeightInput::new("did:2", 200.0, 0.6),
            WeightInput::new("did:3", 150.0, 0.7),
        ];

        let stats = calculator.weight_statistics(&inputs);

        assert!(stats.total_weight > 0.0);
        assert!(stats.mean_weight > 0.0);
        assert!(stats.median_weight > 0.0);
        assert!(stats.gini_coefficient >= 0.0 && stats.gini_coefficient <= 1.0);
    }

    #[test]
    fn test_delegated_weight() {
        let calculator = WeightCalculator::new(WeightConfig::default());

        let with_delegation = WeightInput::new("did:mycelix:delegate", 100.0, 0.7).with_delegated(50.0);
        let without_delegation = WeightInput::new("did:mycelix:solo", 100.0, 0.7);

        let with_result = calculator.calculate(&with_delegation);
        let without_result = calculator.calculate(&without_delegation);

        assert!(with_result.total_weight > without_result.total_weight);
        assert!((with_result.total_weight - without_result.total_weight - 50.0).abs() < 0.01);
    }
}
