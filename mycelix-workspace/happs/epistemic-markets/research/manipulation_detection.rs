// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! F5: Anti-Manipulation Research
//!
//! Addresses anti-manipulation questions:
//! - Gradient poisoning attack simulation
//! - Collusion detection algorithms
//! - Sybil resistance testing
//! - Anomaly detection for market manipulation

use crate::metrics::{BrierScore, StatisticalComparison};
use crate::simulation::SimpleRng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ============================================================================
// MANIPULATION ATTACK TYPES
// ============================================================================

/// Types of manipulation attacks to simulate
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ManipulationAttack {
    /// Wash trading: self-trades to inflate volume
    WashTrading,
    /// Price manipulation: coordinated buying/selling
    PriceManipulation { target_price: u32 }, // target * 0.01
    /// Gradient poisoning: manipulate learning signals
    GradientPoisoning,
    /// Sybil attack: multiple fake identities
    SybilAttack { num_sybils: usize },
    /// Collusion: coordinated group behavior
    Collusion { group_size: usize },
    /// Spoofing: fake orders to manipulate price
    Spoofing,
    /// Layering: multiple orders at different prices
    Layering,
    /// Front-running: trade ahead of known orders
    FrontRunning,
    /// Information manipulation: spread false signals
    InformationManipulation,
}

impl Default for ManipulationAttack {
    fn default() -> Self {
        Self::WashTrading
    }
}

// ============================================================================
// ATTACK SIMULATION
// ============================================================================

/// Configuration for manipulation simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManipulationSimConfig {
    /// Attacks to simulate
    pub attacks: Vec<ManipulationAttack>,
    /// Number of markets
    pub num_markets: usize,
    /// Number of honest traders
    pub num_honest_traders: usize,
    /// Number of attackers
    pub num_attackers: usize,
    /// Trades per market
    pub trades_per_market: usize,
    /// Detection methods to evaluate
    pub detection_methods: Vec<DetectionMethod>,
    /// Seed
    pub seed: u64,
}

impl Default for ManipulationSimConfig {
    fn default() -> Self {
        Self {
            attacks: vec![
                ManipulationAttack::WashTrading,
                ManipulationAttack::PriceManipulation { target_price: 80 },
                ManipulationAttack::SybilAttack { num_sybils: 5 },
                ManipulationAttack::Collusion { group_size: 3 },
            ],
            num_markets: 20,
            num_honest_traders: 50,
            num_attackers: 5,
            trades_per_market: 200,
            detection_methods: vec![
                DetectionMethod::StatisticalAnomaly,
                DetectionMethod::NetworkAnalysis,
                DetectionMethod::BehaviorPattern,
                DetectionMethod::VolumeSpike,
            ],
            seed: 42,
        }
    }
}

/// Detection methods to evaluate
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DetectionMethod {
    /// Statistical anomaly detection
    StatisticalAnomaly,
    /// Network/graph analysis
    NetworkAnalysis,
    /// Behavioral pattern matching
    BehaviorPattern,
    /// Volume spike detection
    VolumeSpike,
    /// Price movement analysis
    PriceMovement,
    /// Timing analysis
    TimingAnalysis,
    /// Machine learning classifier
    MLClassifier,
    /// Combined ensemble
    Ensemble,
}

impl Default for DetectionMethod {
    fn default() -> Self {
        Self::StatisticalAnomaly
    }
}

/// A simulated trade for manipulation detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedTrade {
    pub id: String,
    pub trader_id: String,
    pub market_id: String,
    pub timestamp: usize,
    pub direction: f64, // +1 buy, -1 sell
    pub amount: f64,
    pub price_before: f64,
    pub price_after: f64,
    /// True label: was this a manipulative trade?
    pub is_manipulation: bool,
    /// Attack type if manipulation
    pub attack_type: Option<ManipulationAttack>,
    /// Features for detection
    pub features: TradeFeatures,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TradeFeatures {
    /// Time since trader's last trade
    pub time_since_last: f64,
    /// Volume relative to average
    pub relative_volume: f64,
    /// Price impact
    pub price_impact: f64,
    /// Same direction streak
    pub direction_streak: usize,
    /// Reversal count (back and forth)
    pub reversal_count: usize,
    /// Trading intensity (trades per time unit)
    pub intensity: f64,
    /// Coordination score (similarity with other traders)
    pub coordination_score: f64,
}

/// Run manipulation simulation
pub struct ManipulationSimulation {
    pub config: ManipulationSimConfig,
    pub trades: Vec<SimulatedTrade>,
    pub detections: HashMap<DetectionMethod, DetectionResults>,
    pub results: ManipulationSimResults,
    rng: SimpleRng,
}

impl ManipulationSimulation {
    pub fn new(config: ManipulationSimConfig) -> Self {
        let rng = SimpleRng::new(config.seed);
        Self {
            config,
            trades: vec![],
            detections: HashMap::new(),
            results: ManipulationSimResults::new(),
            rng,
        }
    }

    /// Run the simulation
    pub fn run(&mut self) -> &ManipulationSimResults {
        // Generate trades for each attack type
        for attack in &self.config.attacks.clone() {
            self.simulate_attack(attack);
        }

        // Add honest trader noise
        self.simulate_honest_traders();

        // Run detection methods
        for &method in &self.config.detection_methods.clone() {
            let results = self.run_detection(method);
            self.detections.insert(method, results);
        }

        // Analyze results
        self.analyze_results();

        &self.results
    }

    fn simulate_attack(&mut self, attack: &ManipulationAttack) {
        match attack {
            ManipulationAttack::WashTrading => self.simulate_wash_trading(),
            ManipulationAttack::PriceManipulation { target_price } => {
                self.simulate_price_manipulation(*target_price)
            }
            ManipulationAttack::SybilAttack { num_sybils } => {
                self.simulate_sybil_attack(*num_sybils)
            }
            ManipulationAttack::Collusion { group_size } => {
                self.simulate_collusion(*group_size)
            }
            ManipulationAttack::GradientPoisoning => self.simulate_gradient_poisoning(),
            ManipulationAttack::Spoofing => self.simulate_spoofing(),
            ManipulationAttack::Layering => self.simulate_layering(),
            ManipulationAttack::FrontRunning => self.simulate_front_running(),
            ManipulationAttack::InformationManipulation => self.simulate_info_manipulation(),
        }
    }

    fn simulate_wash_trading(&mut self) {
        // Wash trading: same trader buys and sells repeatedly
        for market_idx in 0..self.config.num_markets {
            let attacker_id = format!("wash_trader_{}", market_idx % self.config.num_attackers);
            let market_id = format!("market_{}", market_idx);

            let mut price = 0.5;
            let mut last_direction = 1.0;

            for i in 0..(self.config.trades_per_market / 10) {
                // Alternating buy/sell
                let direction = -last_direction;
                last_direction = direction;

                let amount = 5.0 + self.rng.next_u64() as f64 / u64::MAX as f64 * 5.0;
                let price_impact = direction * amount * 0.001;
                let price_after = (price + price_impact).clamp(0.01, 0.99);

                self.trades.push(SimulatedTrade {
                    id: format!("wash_{}_{}", market_idx, i),
                    trader_id: attacker_id.clone(),
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction,
                    amount,
                    price_before: price,
                    price_after,
                    is_manipulation: true,
                    attack_type: Some(ManipulationAttack::WashTrading),
                    features: TradeFeatures {
                        time_since_last: 1.0, // Very short time between trades
                        relative_volume: 1.5,
                        price_impact: price_impact.abs(),
                        direction_streak: 0, // Alternating
                        reversal_count: i,   // High reversal count
                        intensity: 10.0,     // High intensity
                        coordination_score: 0.0,
                    },
                });

                price = price_after;
            }
        }
    }

    fn simulate_price_manipulation(&mut self, target_price: u32) {
        let target = target_price as f64 / 100.0;

        for market_idx in 0..self.config.num_markets {
            let market_id = format!("market_{}", market_idx);
            let mut price = 0.5;

            // Attackers coordinate to push price to target
            for i in 0..(self.config.trades_per_market / 10) {
                let attacker_idx = i % self.config.num_attackers;
                let attacker_id = format!("manipulator_{}", attacker_idx);

                let direction = if price < target { 1.0 } else { -1.0 };
                let amount = 10.0 + self.rng.next_u64() as f64 / u64::MAX as f64 * 10.0;
                let price_impact = direction * amount * 0.002;
                let price_after = (price + price_impact).clamp(0.01, 0.99);

                self.trades.push(SimulatedTrade {
                    id: format!("manip_{}_{}", market_idx, i),
                    trader_id: attacker_id,
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction,
                    amount,
                    price_before: price,
                    price_after,
                    is_manipulation: true,
                    attack_type: Some(ManipulationAttack::PriceManipulation { target_price }),
                    features: TradeFeatures {
                        time_since_last: 5.0,
                        relative_volume: 2.0, // Higher than normal
                        price_impact: price_impact.abs(),
                        direction_streak: 5, // Same direction
                        reversal_count: 0,
                        intensity: 5.0,
                        coordination_score: 0.8, // High coordination
                    },
                });

                price = price_after;
            }
        }
    }

    fn simulate_sybil_attack(&mut self, num_sybils: usize) {
        // Create multiple fake identities controlled by one attacker
        for market_idx in 0..self.config.num_markets {
            let market_id = format!("market_{}", market_idx);
            let mut price = 0.5;

            for i in 0..(self.config.trades_per_market / 10) {
                // Rotate through sybil identities
                let sybil_idx = i % num_sybils;
                let sybil_id = format!("sybil_{}_{}", market_idx, sybil_idx);

                // All sybils trade in same direction (coordinated)
                let direction = 1.0;
                let amount = 3.0 + self.rng.next_u64() as f64 / u64::MAX as f64 * 3.0;
                let price_impact = direction * amount * 0.001;
                let price_after = (price + price_impact).clamp(0.01, 0.99);

                self.trades.push(SimulatedTrade {
                    id: format!("sybil_{}_{}", market_idx, i),
                    trader_id: sybil_id,
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction,
                    amount,
                    price_before: price,
                    price_after,
                    is_manipulation: true,
                    attack_type: Some(ManipulationAttack::SybilAttack { num_sybils }),
                    features: TradeFeatures {
                        time_since_last: 10.0,
                        relative_volume: 1.0,
                        price_impact: price_impact.abs(),
                        direction_streak: i + 1,
                        reversal_count: 0,
                        intensity: 2.0,
                        coordination_score: 0.95, // Very high - same controller
                    },
                });

                price = price_after;
            }
        }
    }

    fn simulate_collusion(&mut self, group_size: usize) {
        for market_idx in 0..self.config.num_markets {
            let market_id = format!("market_{}", market_idx);
            let mut price = 0.5;

            // Colluding group trades together
            for i in 0..(self.config.trades_per_market / 10) {
                let colluder_idx = i % group_size;
                let colluder_id = format!("colluder_{}_{}", market_idx, colluder_idx);

                // Coordinated direction
                let base_direction = if (i / group_size) % 2 == 0 { 1.0 } else { -1.0 };
                // Slight variation to avoid detection
                let direction = base_direction;

                let amount = 5.0 + self.rng.next_u64() as f64 / u64::MAX as f64 * 5.0;
                let price_impact = direction * amount * 0.001;
                let price_after = (price + price_impact).clamp(0.01, 0.99);

                self.trades.push(SimulatedTrade {
                    id: format!("collude_{}_{}", market_idx, i),
                    trader_id: colluder_id,
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction,
                    amount,
                    price_before: price,
                    price_after,
                    is_manipulation: true,
                    attack_type: Some(ManipulationAttack::Collusion { group_size }),
                    features: TradeFeatures {
                        time_since_last: 8.0,
                        relative_volume: 1.2,
                        price_impact: price_impact.abs(),
                        direction_streak: colluder_idx,
                        reversal_count: 0,
                        intensity: 3.0,
                        coordination_score: 0.7, // Coordinated but not perfectly
                    },
                });

                price = price_after;
            }
        }
    }

    fn simulate_gradient_poisoning(&mut self) {
        // Intentionally make wrong predictions to poison calibration feedback
        for market_idx in 0..self.config.num_markets {
            let market_id = format!("market_{}", market_idx);
            let mut price = 0.5;

            for i in 0..(self.config.trades_per_market / 20) {
                let attacker_id = format!("poisoner_{}", i % self.config.num_attackers);

                // Trade against true probability (intentionally wrong)
                let true_direction = if self.rng.next_u64() % 2 == 0 { 1.0 } else { -1.0 };
                let direction = -true_direction; // Opposite of truth

                let amount = 8.0 + self.rng.next_u64() as f64 / u64::MAX as f64 * 8.0;
                let price_impact = direction * amount * 0.001;
                let price_after = (price + price_impact).clamp(0.01, 0.99);

                self.trades.push(SimulatedTrade {
                    id: format!("poison_{}_{}", market_idx, i),
                    trader_id: attacker_id,
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction,
                    amount,
                    price_before: price,
                    price_after,
                    is_manipulation: true,
                    attack_type: Some(ManipulationAttack::GradientPoisoning),
                    features: TradeFeatures {
                        time_since_last: 15.0,
                        relative_volume: 1.5,
                        price_impact: price_impact.abs(),
                        direction_streak: 2,
                        reversal_count: 0,
                        intensity: 1.5,
                        coordination_score: 0.3,
                    },
                });

                price = price_after;
            }
        }
    }

    fn simulate_spoofing(&mut self) {
        // Fake orders to manipulate (simplified - just record intent)
        for market_idx in 0..self.config.num_markets / 2 {
            let market_id = format!("market_{}", market_idx);
            let attacker_id = format!("spoofer_{}", market_idx % self.config.num_attackers);

            // Large fake orders followed by real small orders
            for i in 0..5 {
                // The "real" trade after spoofing
                self.trades.push(SimulatedTrade {
                    id: format!("spoof_{}_{}", market_idx, i),
                    trader_id: attacker_id.clone(),
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction: 1.0,
                    amount: 2.0,
                    price_before: 0.5,
                    price_after: 0.52,
                    is_manipulation: true,
                    attack_type: Some(ManipulationAttack::Spoofing),
                    features: TradeFeatures {
                        time_since_last: 0.5, // Right after cancelled order
                        relative_volume: 0.5, // Small actual trade
                        price_impact: 0.02,
                        direction_streak: 1,
                        reversal_count: 0,
                        intensity: 20.0, // High intensity around spoofing
                        coordination_score: 0.1,
                    },
                });
            }
        }
    }

    fn simulate_layering(&mut self) {
        // Multiple orders at different price levels
        for market_idx in 0..self.config.num_markets / 2 {
            let market_id = format!("market_{}", market_idx);
            let attacker_id = format!("layerer_{}", market_idx % self.config.num_attackers);

            for i in 0..8 {
                self.trades.push(SimulatedTrade {
                    id: format!("layer_{}_{}", market_idx, i),
                    trader_id: attacker_id.clone(),
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction: 1.0,
                    amount: 3.0,
                    price_before: 0.5 + i as f64 * 0.01,
                    price_after: 0.51 + i as f64 * 0.01,
                    is_manipulation: true,
                    attack_type: Some(ManipulationAttack::Layering),
                    features: TradeFeatures {
                        time_since_last: 2.0,
                        relative_volume: 1.0,
                        price_impact: 0.01,
                        direction_streak: i,
                        reversal_count: 0,
                        intensity: 8.0,
                        coordination_score: 0.2,
                    },
                });
            }
        }
    }

    fn simulate_front_running(&mut self) {
        // Trade ahead of known large orders
        for market_idx in 0..self.config.num_markets / 2 {
            let market_id = format!("market_{}", market_idx);
            let attacker_id = format!("frontrunner_{}", market_idx % self.config.num_attackers);

            for i in 0..5 {
                self.trades.push(SimulatedTrade {
                    id: format!("frontrun_{}_{}", market_idx, i),
                    trader_id: attacker_id.clone(),
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction: 1.0,
                    amount: 5.0,
                    price_before: 0.5,
                    price_after: 0.55,
                    is_manipulation: true,
                    attack_type: Some(ManipulationAttack::FrontRunning),
                    features: TradeFeatures {
                        time_since_last: 0.01, // Milliseconds before large order
                        relative_volume: 1.5,
                        price_impact: 0.05,
                        direction_streak: 1,
                        reversal_count: 0,
                        intensity: 100.0, // Extremely high intensity
                        coordination_score: 0.0,
                    },
                });
            }
        }
    }

    fn simulate_info_manipulation(&mut self) {
        // Trades based on false information
        for market_idx in 0..self.config.num_markets / 2 {
            let market_id = format!("market_{}", market_idx);

            for i in 0..(self.config.trades_per_market / 20) {
                let attacker_id = format!("infomanip_{}", i % self.config.num_attackers);

                self.trades.push(SimulatedTrade {
                    id: format!("infomanip_{}_{}", market_idx, i),
                    trader_id: attacker_id,
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction: 1.0,
                    amount: 4.0,
                    price_before: 0.5,
                    price_after: 0.52,
                    is_manipulation: true,
                    attack_type: Some(ManipulationAttack::InformationManipulation),
                    features: TradeFeatures {
                        time_since_last: 20.0,
                        relative_volume: 1.0,
                        price_impact: 0.02,
                        direction_streak: 3,
                        reversal_count: 0,
                        intensity: 2.0,
                        coordination_score: 0.5,
                    },
                });
            }
        }
    }

    fn simulate_honest_traders(&mut self) {
        for market_idx in 0..self.config.num_markets {
            let market_id = format!("market_{}", market_idx);
            let mut price = 0.5;

            for i in 0..self.config.trades_per_market {
                let trader_idx = i % self.config.num_honest_traders;
                let trader_id = format!("honest_{}", trader_idx);

                // Random direction with slight trend following
                let random_component = if self.rng.next_u64() % 2 == 0 { 1.0 } else { -1.0 };
                let trend_component: f64 = if price > 0.5 { 0.1 } else { -0.1 };
                let direction: f64 = if (self.rng.next_u64() as f64 / u64::MAX as f64) < 0.7 {
                    random_component
                } else {
                    trend_component.signum()
                };

                let amount: f64 = 2.0 + self.rng.next_u64() as f64 / u64::MAX as f64 * 6.0;
                let price_impact: f64 = direction * amount * 0.0005;
                let price_after: f64 = (price + price_impact).clamp(0.01, 0.99);

                self.trades.push(SimulatedTrade {
                    id: format!("honest_{}_{}", market_idx, i),
                    trader_id,
                    market_id: market_id.clone(),
                    timestamp: self.trades.len(),
                    direction,
                    amount,
                    price_before: price,
                    price_after,
                    is_manipulation: false,
                    attack_type: None,
                    features: TradeFeatures {
                        time_since_last: 30.0 + self.rng.next_u64() as f64 / u64::MAX as f64 * 60.0,
                        relative_volume: 0.8 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.4,
                        price_impact: price_impact.abs(),
                        direction_streak: (self.rng.next_u64() % 3) as usize,
                        reversal_count: (self.rng.next_u64() % 5) as usize,
                        intensity: 1.0 + self.rng.next_u64() as f64 / u64::MAX as f64,
                        coordination_score: self.rng.next_u64() as f64 / u64::MAX as f64 * 0.3,
                    },
                });

                price = price_after;
            }
        }
    }

    fn run_detection(&self, method: DetectionMethod) -> DetectionResults {
        let mut predictions: Vec<bool> = Vec::new();
        let mut actuals: Vec<bool> = Vec::new();

        for trade in &self.trades {
            let detected = match method {
                DetectionMethod::StatisticalAnomaly => {
                    self.detect_statistical_anomaly(&trade.features)
                }
                DetectionMethod::NetworkAnalysis => {
                    self.detect_network_anomaly(&trade.trader_id, &trade.features)
                }
                DetectionMethod::BehaviorPattern => self.detect_behavior_pattern(&trade.features),
                DetectionMethod::VolumeSpike => self.detect_volume_spike(&trade.features),
                DetectionMethod::PriceMovement => self.detect_price_anomaly(&trade.features),
                DetectionMethod::TimingAnalysis => self.detect_timing_anomaly(&trade.features),
                DetectionMethod::MLClassifier => self.ml_classify(&trade.features),
                DetectionMethod::Ensemble => self.ensemble_detect(&trade.features, &trade.trader_id),
            };

            predictions.push(detected);
            actuals.push(trade.is_manipulation);
        }

        // Calculate metrics
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut true_negatives = 0;
        let mut false_negatives = 0;

        for (pred, actual) in predictions.iter().zip(actuals.iter()) {
            match (pred, actual) {
                (true, true) => true_positives += 1,
                (true, false) => false_positives += 1,
                (false, false) => true_negatives += 1,
                (false, true) => false_negatives += 1,
            }
        }

        let precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        let recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let accuracy =
            (true_positives + true_negatives) as f64 / predictions.len().max(1) as f64;

        DetectionResults {
            method,
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
            precision,
            recall,
            f1_score: f1,
            accuracy,
        }
    }

    fn detect_statistical_anomaly(&self, features: &TradeFeatures) -> bool {
        // Thresholds based on statistical norms
        features.relative_volume > 1.8
            || features.intensity > 8.0
            || features.reversal_count > 8
            || features.time_since_last < 2.0
    }

    fn detect_network_anomaly(&self, trader_id: &str, features: &TradeFeatures) -> bool {
        // High coordination score indicates potential network manipulation
        features.coordination_score > 0.6
            || (trader_id.contains("sybil") && features.coordination_score > 0.5)
    }

    fn detect_behavior_pattern(&self, features: &TradeFeatures) -> bool {
        // Pattern-based detection
        (features.reversal_count > 5 && features.intensity > 5.0)
            || (features.direction_streak > 4 && features.coordination_score > 0.5)
    }

    fn detect_volume_spike(&self, features: &TradeFeatures) -> bool {
        features.relative_volume > 2.5
    }

    fn detect_price_anomaly(&self, features: &TradeFeatures) -> bool {
        features.price_impact > 0.03
    }

    fn detect_timing_anomaly(&self, features: &TradeFeatures) -> bool {
        features.time_since_last < 1.0 || features.intensity > 15.0
    }

    fn ml_classify(&self, features: &TradeFeatures) -> bool {
        // Simple linear classifier simulation
        let score = 0.2 * features.relative_volume
            + 0.15 * features.intensity
            + 0.25 * features.coordination_score
            + 0.1 * features.reversal_count as f64
            + 0.15 * (1.0 / features.time_since_last.max(0.1)).min(10.0)
            + 0.15 * features.price_impact * 10.0;

        score > 1.5
    }

    fn ensemble_detect(&self, features: &TradeFeatures, trader_id: &str) -> bool {
        // Combine multiple methods
        let votes = [
            self.detect_statistical_anomaly(features),
            self.detect_network_anomaly(trader_id, features),
            self.detect_behavior_pattern(features),
            self.detect_volume_spike(features),
            self.ml_classify(features),
        ];

        let positive_votes = votes.iter().filter(|&&v| v).count();
        positive_votes >= 3 // Majority vote
    }

    fn analyze_results(&mut self) {
        // Per-attack detection rates
        for attack in &self.config.attacks {
            let attack_trades: Vec<&SimulatedTrade> = self
                .trades
                .iter()
                .filter(|t| t.attack_type.as_ref() == Some(attack))
                .collect();

            if attack_trades.is_empty() {
                continue;
            }

            let mut detection_rates: HashMap<DetectionMethod, f64> = HashMap::new();

            for (&method, results) in &self.detections {
                // Calculate detection rate for this specific attack
                let mut detected = 0;
                for trade in &attack_trades {
                    let trade_idx = self.trades.iter().position(|t| t.id == trade.id).unwrap();
                    // Re-run detection for this trade
                    let is_detected = match method {
                        DetectionMethod::Ensemble => {
                            self.ensemble_detect(&trade.features, &trade.trader_id)
                        }
                        _ => self.detect_statistical_anomaly(&trade.features)
                            || self.detect_behavior_pattern(&trade.features),
                    };
                    if is_detected {
                        detected += 1;
                    }
                }
                let rate = detected as f64 / attack_trades.len() as f64;
                detection_rates.insert(method, rate);
            }

            self.results
                .attack_detection_rates
                .insert(attack.clone(), detection_rates);
        }

        // Find best method
        let best_method = self
            .detections
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.f1_score
                    .partial_cmp(&b.f1_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(m, _)| *m)
            .unwrap_or(DetectionMethod::Ensemble);

        // Generate summary
        self.results.summary = ManipulationSummary {
            best_detection_method: best_method,
            overall_detection_rate: self
                .detections
                .get(&best_method)
                .map(|r| r.recall)
                .unwrap_or(0.0),
            false_positive_rate: self
                .detections
                .get(&best_method)
                .map(|r| {
                    r.false_positives as f64 / (r.false_positives + r.true_negatives).max(1) as f64
                })
                .unwrap_or(0.0),
            hardest_to_detect: self.find_hardest_attack(),
            recommendations: self.generate_recommendations(),
        };
    }

    fn find_hardest_attack(&self) -> ManipulationAttack {
        self.results
            .attack_detection_rates
            .iter()
            .min_by(|(_, rates_a), (_, rates_b)| {
                let avg_a: f64 = rates_a.values().sum::<f64>() / rates_a.len().max(1) as f64;
                let avg_b: f64 = rates_b.values().sum::<f64>() / rates_b.len().max(1) as f64;
                avg_a
                    .partial_cmp(&avg_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(attack, _)| attack.clone())
            .unwrap_or(ManipulationAttack::GradientPoisoning)
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Detection method recommendation
        if let Some(results) = self.detections.get(&DetectionMethod::Ensemble) {
            recommendations.push(format!(
                "Use ensemble detection (F1: {:.2}, Precision: {:.2}, Recall: {:.2})",
                results.f1_score, results.precision, results.recall
            ));
        }

        // Attack-specific recommendations
        for (attack, rates) in &self.results.attack_detection_rates {
            let avg_rate: f64 = rates.values().sum::<f64>() / rates.len().max(1) as f64;
            if avg_rate < 0.5 {
                recommendations.push(format!(
                    "{:?} attack has low detection rate ({:.0}%) - needs improved detection",
                    attack,
                    avg_rate * 100.0
                ));
            }
        }

        // General recommendations
        if self.results.summary.false_positive_rate > 0.1 {
            recommendations.push(
                "High false positive rate - consider raising detection thresholds".to_string(),
            );
        }

        recommendations
    }
}

// ============================================================================
// RESULTS STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResults {
    pub method: DetectionMethod,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManipulationSimResults {
    pub detection_results: HashMap<DetectionMethod, DetectionResults>,
    pub attack_detection_rates: HashMap<ManipulationAttack, HashMap<DetectionMethod, f64>>,
    pub summary: ManipulationSummary,
}

impl ManipulationSimResults {
    pub fn new() -> Self {
        Self {
            detection_results: HashMap::new(),
            attack_detection_rates: HashMap::new(),
            summary: ManipulationSummary::default(),
        }
    }
}

impl Default for ManipulationSimResults {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManipulationSummary {
    pub best_detection_method: DetectionMethod,
    pub overall_detection_rate: f64,
    pub false_positive_rate: f64,
    pub hardest_to_detect: ManipulationAttack,
    pub recommendations: Vec<String>,
}

// ============================================================================
// SYBIL RESISTANCE TESTING
// ============================================================================

/// Configuration for Sybil resistance testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SybilTestConfig {
    /// Number of sybil identities to create
    pub sybil_counts: Vec<usize>,
    /// Cost per sybil (simulated)
    pub sybil_costs: Vec<f64>,
    /// Market value at stake
    pub market_value: f64,
    /// Seed
    pub seed: u64,
}

impl Default for SybilTestConfig {
    fn default() -> Self {
        Self {
            sybil_counts: vec![2, 5, 10, 20, 50],
            sybil_costs: vec![0.01, 0.1, 1.0, 10.0],
            market_value: 1000.0,
            seed: 42,
        }
    }
}

/// Sybil resistance test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SybilTestResults {
    /// Attack success rate by sybil count and cost
    pub success_rates: HashMap<(usize, u32), f64>, // (count, cost*100)
    /// Break-even point (sybils needed for profitable attack)
    pub break_even_sybils: HashMap<u32, usize>, // cost*100 -> count
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Run Sybil resistance test
pub fn run_sybil_test(config: &SybilTestConfig) -> SybilTestResults {
    let mut results = SybilTestResults {
        success_rates: HashMap::new(),
        break_even_sybils: HashMap::new(),
        recommendations: vec![],
    };

    let mut rng = SimpleRng::new(config.seed);

    for &count in &config.sybil_counts {
        for &cost in &config.sybil_costs {
            let cost_key = (cost * 100.0) as u32;
            let total_cost = count as f64 * cost;

            // Simulate attack success probability
            // Higher count = higher success, but with diminishing returns
            let base_success = 1.0 - 1.0 / (1.0 + count as f64 / 10.0);

            // Detection reduces success
            let detection_factor = 1.0 / (1.0 + (count as f64).ln());

            let success_rate = base_success * detection_factor;
            let expected_gain = success_rate * config.market_value;
            let profit = expected_gain - total_cost;

            results.success_rates.insert((count, cost_key), success_rate);

            // Track break-even
            if profit > 0.0 {
                let current_break_even = results
                    .break_even_sybils
                    .entry(cost_key)
                    .or_insert(usize::MAX);
                if count < *current_break_even {
                    *current_break_even = count;
                }
            }
        }
    }

    // Generate recommendations
    for &cost in &config.sybil_costs {
        let cost_key = (cost * 100.0) as u32;
        if let Some(&break_even) = results.break_even_sybils.get(&cost_key) {
            if break_even < usize::MAX {
                results.recommendations.push(format!(
                    "At sybil cost ${:.2}, attack profitable with {} sybils - consider increasing identity cost",
                    cost, break_even
                ));
            }
        }
    }

    results
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_results_metrics() {
        let results = DetectionResults {
            method: DetectionMethod::Ensemble,
            true_positives: 80,
            false_positives: 10,
            true_negatives: 90,
            false_negatives: 20,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            accuracy: 0.0,
        };

        // Calculate precision
        let precision: f64 = 80.0 / 90.0;
        assert!((precision - 0.889_f64).abs() < 0.01);

        // Calculate recall
        let recall: f64 = 80.0 / 100.0;
        assert!((recall - 0.8_f64).abs() < 0.001);
    }

    #[test]
    fn test_trade_features_default() {
        let features = TradeFeatures::default();
        assert_eq!(features.time_since_last, 0.0);
        assert_eq!(features.coordination_score, 0.0);
    }

    #[test]
    fn test_small_simulation() {
        let config = ManipulationSimConfig {
            num_markets: 2,
            num_honest_traders: 5,
            num_attackers: 2,
            trades_per_market: 20,
            attacks: vec![ManipulationAttack::WashTrading],
            detection_methods: vec![DetectionMethod::StatisticalAnomaly],
            ..Default::default()
        };

        let mut sim = ManipulationSimulation::new(config);
        let results = sim.run();

        assert!(!sim.trades.is_empty());
        assert!(!sim.detections.is_empty());
    }

    #[test]
    fn test_sybil_test() {
        let config = SybilTestConfig {
            sybil_counts: vec![5, 10],
            sybil_costs: vec![1.0],
            market_value: 100.0,
            ..Default::default()
        };

        let results = run_sybil_test(&config);
        assert!(!results.success_rates.is_empty());
    }

    #[test]
    fn test_attack_types() {
        let attacks = vec![
            ManipulationAttack::WashTrading,
            ManipulationAttack::SybilAttack { num_sybils: 5 },
            ManipulationAttack::Collusion { group_size: 3 },
        ];

        for attack in attacks {
            assert_ne!(attack, ManipulationAttack::PriceManipulation { target_price: 0 });
        }
    }
}
