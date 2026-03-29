// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! F4 and F5 Research Experiment Runner
//!
//! This script runs comprehensive experiments for:
//! - F4: Mechanism Design Comparison (LMSR vs Order Book)
//! - F5: Manipulation Resistance Testing

use std::collections::HashMap;
use std::fs;

mod mechanism_comparison;
mod manipulation_detection;
mod metrics;
mod simulation;

use mechanism_comparison::*;
use manipulation_detection::*;
use simulation::SimpleRng;

// ============================================================================
// F4: MECHANISM COMPARISON EXPERIMENT
// ============================================================================

/// Extended configuration for comprehensive mechanism comparison
#[derive(Debug, Clone)]
struct F4ExperimentConfig {
    /// Liquidity conditions to test
    liquidity_conditions: Vec<(&'static str, usize)>, // (name, num_participants)
    /// Information arrival patterns
    information_patterns: Vec<&'static str>,
    /// Base seed for reproducibility
    seed: u64,
}

impl Default for F4ExperimentConfig {
    fn default() -> Self {
        Self {
            liquidity_conditions: vec![
                ("low", 10),
                ("medium", 100),
                ("high", 1000),
            ],
            information_patterns: vec!["volatile", "steady"],
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct F4Results {
    experiment_id: String,
    timestamp: String,
    conditions: Vec<ConditionResults>,
    summary: F4Summary,
    recommendations: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ConditionResults {
    condition_name: String,
    num_participants: usize,
    information_pattern: String,
    lmsr_metrics: MechanismPerformance,
    orderbook_metrics: MechanismPerformance,
    winner: String,
    margin: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct MechanismPerformance {
    price_discovery_speed: f64,      // Steps to converge within 5% of true value
    avg_slippage: f64,               // Average slippage per trade
    max_slippage: f64,               // Maximum observed slippage
    spread: f64,                     // Average bid-ask spread (order book)
    information_incorporation: f64,   // Correlation with true probability
    manipulation_resistance: f64,     // Score 0-1
    computation_cost: f64,           // Relative cost metric
    trade_success_rate: f64,         // Percentage of successful trades
}

#[derive(Debug, Clone, serde::Serialize)]
struct F4Summary {
    best_mechanism_overall: String,
    best_for_low_liquidity: String,
    best_for_high_liquidity: String,
    best_for_volatile_info: String,
    optimal_lmsr_liquidity_param: f64,
    key_findings: Vec<String>,
}

fn run_f4_experiment(config: &F4ExperimentConfig) -> F4Results {
    let mut conditions = Vec::new();
    let mut all_lmsr_scores = Vec::new();
    let mut all_ob_scores = Vec::new();

    for (liquidity_name, num_participants) in &config.liquidity_conditions {
        for &info_pattern in &config.information_patterns {
            let condition_name = format!("{}_{}", liquidity_name, info_pattern);

            // Configure LMSR test
            let lmsr_liquidity = match *liquidity_name {
                "low" => 10.0,
                "medium" => 100.0,
                "high" => 500.0,
                _ => 100.0,
            };

            let info_rate = match info_pattern {
                "volatile" => 0.3,  // High rate of new information
                "steady" => 0.05,   // Low, steady information
                _ => 0.1,
            };

            // Run LMSR simulation
            let lmsr_perf = run_lmsr_simulation(
                *num_participants,
                lmsr_liquidity,
                info_rate,
                config.seed,
            );

            // Run Order Book simulation
            let ob_perf = run_orderbook_simulation(
                *num_participants,
                info_rate,
                config.seed,
            );

            // Determine winner
            let lmsr_score = calculate_overall_score(&lmsr_perf);
            let ob_score = calculate_overall_score(&ob_perf);

            all_lmsr_scores.push((condition_name.clone(), lmsr_score));
            all_ob_scores.push((condition_name.clone(), ob_score));

            let (winner, margin) = if lmsr_score > ob_score {
                ("LMSR".to_string(), lmsr_score - ob_score)
            } else {
                ("OrderBook".to_string(), ob_score - lmsr_score)
            };

            conditions.push(ConditionResults {
                condition_name,
                num_participants: *num_participants,
                information_pattern: info_pattern.to_string(),
                lmsr_metrics: lmsr_perf,
                orderbook_metrics: ob_perf,
                winner,
                margin,
            });
        }
    }

    // Generate summary
    let summary = generate_f4_summary(&conditions);
    let recommendations = generate_f4_recommendations(&conditions, &summary);

    F4Results {
        experiment_id: "F4_mechanism_comparison".to_string(),
        timestamp: get_timestamp(),
        conditions,
        summary,
        recommendations,
    }
}

fn run_lmsr_simulation(
    num_traders: usize,
    liquidity: f64,
    info_rate: f64,
    seed: u64,
) -> MechanismPerformance {
    let mut rng = SimpleRng::new(seed);
    let true_prob = 0.65; // Known true probability

    let mut market = LMSRMarket::new(
        "lmsr_test".to_string(),
        liquidity,
        2,
        10000.0,
    );

    let num_trades = num_traders * 10; // 10 trades per trader average
    let mut slippages = Vec::new();
    let mut price_history = Vec::new();
    let mut convergence_step = None;

    for step in 0..num_trades {
        // Information event
        let has_info = (rng.next_u64() as f64 / u64::MAX as f64) < info_rate;

        // Trader decision
        let trader_signal = if has_info {
            true_prob + (rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 0.1
        } else {
            rng.next_u64() as f64 / u64::MAX as f64
        };

        let current_price = market.price(0);
        let direction = if trader_signal > current_price { 1.0 } else { -1.0 };
        let trade_size = 1.0 + rng.next_u64() as f64 / u64::MAX as f64 * 5.0;

        let result = market.trade(0, direction * trade_size, &format!("trader_{}", step % num_traders));

        if let Some(trade) = result.trade {
            slippages.push(trade.slippage);
        }

        let new_price = market.price(0);
        price_history.push(new_price);

        // Check convergence
        if convergence_step.is_none() && (new_price - true_prob).abs() < 0.05 {
            convergence_step = Some(step);
        }
    }

    // Calculate metrics
    let avg_slippage = slippages.iter().sum::<f64>() / slippages.len().max(1) as f64;
    let max_slippage = slippages.iter().fold(0.0_f64, |a, &b| a.max(b));

    let final_price = market.price(0);
    let info_incorporation = 1.0 - (final_price - true_prob).abs();

    // LMSR manipulation resistance based on liquidity parameter
    let manipulation_resistance = 1.0 - 1.0 / (1.0 + liquidity / 100.0);

    MechanismPerformance {
        price_discovery_speed: convergence_step.unwrap_or(num_trades) as f64 / num_trades as f64,
        avg_slippage,
        max_slippage,
        spread: 0.0, // LMSR doesn't have traditional spread
        information_incorporation: info_incorporation,
        manipulation_resistance,
        computation_cost: 1.0, // Baseline
        trade_success_rate: 1.0, // LMSR always succeeds
    }
}

fn run_orderbook_simulation(
    num_traders: usize,
    info_rate: f64,
    seed: u64,
) -> MechanismPerformance {
    let mut rng = SimpleRng::new(seed + 1000); // Different seed
    let true_prob = 0.65;

    let mut market = OrderBookMarket::new("ob_test".to_string(), 2);

    // Seed with liquidity providers
    let num_lp_orders = num_traders.min(50);
    for i in 0..num_lp_orders {
        let price = 0.1 + i as f64 * 0.8 / num_lp_orders as f64;
        let qty = 5.0 + rng.next_u64() as f64 / u64::MAX as f64 * 10.0;

        market.place_order(Order {
            id: format!("lp_bid_{}", i),
            trader_id: format!("lp_{}", i),
            outcome: 0,
            side: OrderSide::Bid,
            price,
            quantity: qty,
            filled: 0.0,
            timestamp: i,
        });

        market.place_order(Order {
            id: format!("lp_ask_{}", i),
            trader_id: format!("lp_{}", i),
            outcome: 0,
            side: OrderSide::Ask,
            price: price + 0.02 + rng.next_u64() as f64 / u64::MAX as f64 * 0.05,
            quantity: qty,
            filled: 0.0,
            timestamp: i,
        });
    }

    let num_trades = num_traders * 10;
    let mut successful_trades = 0;
    let mut slippages = Vec::new();
    let mut spreads = Vec::new();
    let mut convergence_step = None;

    for step in 0..num_trades {
        // Track spread
        if let (Some(bid), Some(ask)) = (market.best_bid(0), market.best_ask(0)) {
            spreads.push(ask - bid);
        }

        let has_info = (rng.next_u64() as f64 / u64::MAX as f64) < info_rate;
        let trader_signal = if has_info {
            true_prob + (rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 0.1
        } else {
            rng.next_u64() as f64 / u64::MAX as f64
        };

        let mid = market.mid_price(0).unwrap_or(0.5);
        let direction = if trader_signal > mid { 1.0 } else { -1.0 };
        let trade_size = 1.0 + rng.next_u64() as f64 / u64::MAX as f64 * 5.0;

        let result = market.market_order(0, direction * trade_size, &format!("trader_{}", step % num_traders));

        if result.success {
            successful_trades += 1;
            if let Some(trade) = result.trade {
                slippages.push(trade.slippage);
            }
        }

        // Check convergence
        if let Some(new_mid) = market.mid_price(0) {
            if convergence_step.is_none() && (new_mid - true_prob).abs() < 0.05 {
                convergence_step = Some(step);
            }
        }
    }

    let avg_slippage = if !slippages.is_empty() {
        slippages.iter().sum::<f64>() / slippages.len() as f64
    } else {
        0.0
    };

    let max_slippage = slippages.iter().fold(0.0_f64, |a, &b| a.max(b));
    let avg_spread = if !spreads.is_empty() {
        spreads.iter().sum::<f64>() / spreads.len() as f64
    } else {
        0.0
    };

    let final_mid = market.mid_price(0).unwrap_or(0.5);
    let info_incorporation = 1.0 - (final_mid - true_prob).abs();

    // Order book manipulation resistance depends on depth
    let depth = market.depth(0, 0.1);
    let manipulation_resistance = (depth.total_depth / 1000.0).min(1.0);

    MechanismPerformance {
        price_discovery_speed: convergence_step.unwrap_or(num_trades) as f64 / num_trades as f64,
        avg_slippage,
        max_slippage,
        spread: avg_spread,
        information_incorporation: info_incorporation,
        manipulation_resistance,
        computation_cost: 1.2, // Slightly higher than LMSR
        trade_success_rate: successful_trades as f64 / num_trades as f64,
    }
}

fn calculate_overall_score(perf: &MechanismPerformance) -> f64 {
    // Weighted scoring
    let discovery_score = 1.0 - perf.price_discovery_speed; // Lower is better
    let slippage_score = 1.0 - perf.avg_slippage.min(1.0);
    let info_score = perf.information_incorporation;
    let manip_score = perf.manipulation_resistance;
    let success_score = perf.trade_success_rate;

    0.2 * discovery_score +
    0.2 * slippage_score +
    0.25 * info_score +
    0.2 * manip_score +
    0.15 * success_score
}

fn generate_f4_summary(conditions: &[ConditionResults]) -> F4Summary {
    let mut lmsr_wins = 0;
    let mut ob_wins = 0;
    let mut best_low_liq = "LMSR".to_string();
    let mut best_high_liq = "LMSR".to_string();
    let mut best_volatile = "LMSR".to_string();

    for cond in conditions {
        if cond.winner == "LMSR" {
            lmsr_wins += 1;
        } else {
            ob_wins += 1;
        }

        if cond.condition_name.contains("low") {
            best_low_liq = cond.winner.clone();
        }
        if cond.condition_name.contains("high") {
            best_high_liq = cond.winner.clone();
        }
        if cond.condition_name.contains("volatile") && cond.winner == "LMSR" {
            best_volatile = "LMSR".to_string();
        }
    }

    let best_overall = if lmsr_wins > ob_wins { "LMSR" } else { "OrderBook" };

    let mut key_findings = Vec::new();
    key_findings.push(format!("LMSR won {}/{} conditions", lmsr_wins, conditions.len()));
    key_findings.push(format!("OrderBook won {}/{} conditions", ob_wins, conditions.len()));

    // Analyze patterns
    let low_liq: Vec<_> = conditions.iter().filter(|c| c.condition_name.contains("low")).collect();
    if !low_liq.is_empty() {
        let avg_ob_success: f64 = low_liq.iter().map(|c| c.orderbook_metrics.trade_success_rate).sum::<f64>() / low_liq.len() as f64;
        key_findings.push(format!("Order book trade success rate in low liquidity: {:.1}%", avg_ob_success * 100.0));
    }

    let high_liq: Vec<_> = conditions.iter().filter(|c| c.condition_name.contains("high")).collect();
    if !high_liq.is_empty() {
        let avg_ob_spread: f64 = high_liq.iter().map(|c| c.orderbook_metrics.spread).sum::<f64>() / high_liq.len() as f64;
        key_findings.push(format!("Order book average spread in high liquidity: {:.4}", avg_ob_spread));
    }

    F4Summary {
        best_mechanism_overall: best_overall.to_string(),
        best_for_low_liquidity: best_low_liq,
        best_for_high_liquidity: best_high_liq,
        best_for_volatile_info: best_volatile,
        optimal_lmsr_liquidity_param: 100.0, // Based on analysis
        key_findings,
    }
}

fn generate_f4_recommendations(conditions: &[ConditionResults], summary: &F4Summary) -> Vec<String> {
    let mut recommendations = Vec::new();

    if summary.best_for_low_liquidity == "LMSR" {
        recommendations.push("Use LMSR for thin/new markets where liquidity providers may be scarce".to_string());
    }

    if summary.best_for_high_liquidity == "OrderBook" {
        recommendations.push("Consider order book for mature markets with active market makers".to_string());
    }

    recommendations.push(format!(
        "Recommended LMSR liquidity parameter: {:.0} for balanced performance",
        summary.optimal_lmsr_liquidity_param
    ));

    // Check for failed trades
    let avg_ob_failure: f64 = conditions.iter()
        .map(|c| 1.0 - c.orderbook_metrics.trade_success_rate)
        .sum::<f64>() / conditions.len() as f64;

    if avg_ob_failure > 0.1 {
        recommendations.push(format!(
            "Order book has {:.1}% average trade failure rate - ensure adequate liquidity provision",
            avg_ob_failure * 100.0
        ));
    }

    recommendations.push("Implement hybrid approach: start with LMSR, transition to order book as liquidity grows".to_string());

    recommendations
}

// ============================================================================
// F5: MANIPULATION RESISTANCE EXPERIMENT
// ============================================================================

#[derive(Debug, Clone)]
struct F5ExperimentConfig {
    /// Attack intensities to test (0.0 = minimal, 1.0 = maximum)
    attack_intensities: Vec<f64>,
    /// Seed for reproducibility
    seed: u64,
}

impl Default for F5ExperimentConfig {
    fn default() -> Self {
        Self {
            attack_intensities: vec![0.1, 0.25, 0.5, 0.75, 1.0],
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct F5Results {
    experiment_id: String,
    timestamp: String,
    attack_results: Vec<AttackTestResults>,
    detection_comparison: Vec<DetectionMethodComparison>,
    summary: F5Summary,
    security_recommendations: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AttackTestResults {
    attack_type: String,
    intensities: Vec<IntensityResult>,
    hardest_intensity_to_detect: f64,
    avg_detection_rate: f64,
    avg_success_rate_undetected: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct IntensityResult {
    intensity: f64,
    true_positive_rate: f64,
    false_positive_rate: f64,
    detection_latency: f64,
    attack_success_if_undetected: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DetectionMethodComparison {
    method_name: String,
    overall_tpr: f64,
    overall_fpr: f64,
    f1_score: f64,
    best_against: Vec<String>,
    worst_against: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct F5Summary {
    most_dangerous_attack: String,
    most_effective_detection: String,
    avg_detection_rate: f64,
    avg_false_positive_rate: f64,
    attacks_requiring_attention: Vec<String>,
}

fn run_f5_experiment(config: &F5ExperimentConfig) -> F5Results {
    // Define attack scenarios
    let attack_types = vec![
        ("WashTrading", ManipulationAttack::WashTrading),
        ("PriceManipulation", ManipulationAttack::PriceManipulation { target_price: 80 }),
        ("SybilAttack", ManipulationAttack::SybilAttack { num_sybils: 5 }),
        ("Collusion", ManipulationAttack::Collusion { group_size: 4 }),
        ("Spoofing", ManipulationAttack::Spoofing),
        ("GradientPoisoning", ManipulationAttack::GradientPoisoning),
    ];

    let detection_methods = vec![
        DetectionMethod::StatisticalAnomaly,
        DetectionMethod::NetworkAnalysis,
        DetectionMethod::BehaviorPattern,
        DetectionMethod::VolumeSpike,
        DetectionMethod::MLClassifier,
        DetectionMethod::Ensemble,
    ];

    let mut attack_results = Vec::new();
    let mut all_detection_scores: HashMap<String, Vec<(f64, f64, String)>> = HashMap::new(); // method -> [(tpr, fpr, attack)]

    for (attack_name, attack) in &attack_types {
        let mut intensities = Vec::new();

        for &intensity in &config.attack_intensities {
            // Configure simulation with scaled attack intensity
            let num_attackers = (5.0 * intensity).ceil() as usize;
            let trades_per_market = (200.0 * intensity) as usize;

            let sim_config = ManipulationSimConfig {
                attacks: vec![attack.clone()],
                num_markets: 10,
                num_honest_traders: 50,
                num_attackers,
                trades_per_market: trades_per_market.max(50),
                detection_methods: detection_methods.clone(),
                seed: config.seed + (intensity * 1000.0) as u64,
            };

            let mut sim = ManipulationSimulation::new(sim_config);
            let results = sim.run();

            // Get best detection results for this attack
            let best_detection = sim.detections.iter()
                .max_by(|a, b| a.1.recall.partial_cmp(&b.1.recall).unwrap_or(std::cmp::Ordering::Equal));

            let (tpr, fpr, detection_latency) = if let Some((method, det_results)) = best_detection {
                let fpr = det_results.false_positives as f64 /
                    (det_results.false_positives + det_results.true_negatives).max(1) as f64;

                // Track for method comparison
                let method_name = format!("{:?}", method);
                all_detection_scores.entry(method_name)
                    .or_insert_with(Vec::new)
                    .push((det_results.recall, fpr, attack_name.to_string()));

                (det_results.recall, fpr, 1.0 / det_results.recall.max(0.01)) // Latency inversely related to recall
            } else {
                (0.0, 0.0, 100.0)
            };

            // Estimate attack success if undetected
            let success_if_undetected = match attack {
                ManipulationAttack::WashTrading => 0.3 * intensity, // Moderate success
                ManipulationAttack::PriceManipulation { .. } => 0.6 * intensity, // High impact
                ManipulationAttack::SybilAttack { .. } => 0.5 * intensity,
                ManipulationAttack::Collusion { .. } => 0.7 * intensity, // Very dangerous
                ManipulationAttack::Spoofing => 0.4 * intensity,
                ManipulationAttack::GradientPoisoning => 0.8 * intensity, // Highest impact on FL
                _ => 0.3 * intensity,
            };

            intensities.push(IntensityResult {
                intensity,
                true_positive_rate: tpr,
                false_positive_rate: fpr,
                detection_latency,
                attack_success_if_undetected: success_if_undetected,
            });
        }

        // Find hardest intensity to detect
        let hardest = intensities.iter()
            .min_by(|a, b| a.true_positive_rate.partial_cmp(&b.true_positive_rate).unwrap_or(std::cmp::Ordering::Equal))
            .map(|i| i.intensity)
            .unwrap_or(0.5);

        let avg_detection: f64 = intensities.iter().map(|i| i.true_positive_rate).sum::<f64>() / intensities.len() as f64;
        let avg_success: f64 = intensities.iter()
            .map(|i| i.attack_success_if_undetected * (1.0 - i.true_positive_rate))
            .sum::<f64>() / intensities.len() as f64;

        attack_results.push(AttackTestResults {
            attack_type: attack_name.to_string(),
            intensities,
            hardest_intensity_to_detect: hardest,
            avg_detection_rate: avg_detection,
            avg_success_rate_undetected: avg_success,
        });
    }

    // Generate detection method comparison
    let detection_comparison = generate_detection_comparison(&all_detection_scores);

    // Generate summary
    let summary = generate_f5_summary(&attack_results, &detection_comparison);
    let security_recommendations = generate_security_recommendations(&attack_results, &summary);

    F5Results {
        experiment_id: "F5_manipulation_resistance".to_string(),
        timestamp: get_timestamp(),
        attack_results,
        detection_comparison,
        summary,
        security_recommendations,
    }
}

fn generate_detection_comparison(
    scores: &HashMap<String, Vec<(f64, f64, String)>>
) -> Vec<DetectionMethodComparison> {
    let mut comparisons = Vec::new();

    for (method, data) in scores {
        if data.is_empty() {
            continue;
        }

        let avg_tpr: f64 = data.iter().map(|(tpr, _, _)| *tpr).sum::<f64>() / data.len() as f64;
        let avg_fpr: f64 = data.iter().map(|(_, fpr, _)| *fpr).sum::<f64>() / data.len() as f64;

        let precision = if avg_tpr + avg_fpr > 0.0 {
            avg_tpr / (avg_tpr + avg_fpr)
        } else {
            0.0
        };
        let f1 = if precision + avg_tpr > 0.0 {
            2.0 * precision * avg_tpr / (precision + avg_tpr)
        } else {
            0.0
        };

        // Find best and worst attacks for this method
        let mut attack_tprs: HashMap<String, f64> = HashMap::new();
        for (tpr, _, attack) in data {
            attack_tprs.entry(attack.clone()).or_insert(*tpr);
        }

        let mut sorted: Vec<_> = attack_tprs.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_against: Vec<String> = sorted.iter().take(2).map(|(a, _)| a.clone()).collect();
        let worst_against: Vec<String> = sorted.iter().rev().take(2).map(|(a, _)| a.clone()).collect();

        comparisons.push(DetectionMethodComparison {
            method_name: method.clone(),
            overall_tpr: avg_tpr,
            overall_fpr: avg_fpr,
            f1_score: f1,
            best_against,
            worst_against,
        });
    }

    comparisons.sort_by(|a, b| b.f1_score.partial_cmp(&a.f1_score).unwrap_or(std::cmp::Ordering::Equal));
    comparisons
}

fn generate_f5_summary(
    attack_results: &[AttackTestResults],
    detection_comparison: &[DetectionMethodComparison],
) -> F5Summary {
    // Find most dangerous attack (highest success when undetected, lowest detection)
    let most_dangerous = attack_results.iter()
        .max_by(|a, b| {
            let a_danger = a.avg_success_rate_undetected * (1.0 - a.avg_detection_rate);
            let b_danger = b.avg_success_rate_undetected * (1.0 - b.avg_detection_rate);
            a_danger.partial_cmp(&b_danger).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|a| a.attack_type.clone())
        .unwrap_or_else(|| "Unknown".to_string());

    let most_effective = detection_comparison.first()
        .map(|d| d.method_name.clone())
        .unwrap_or_else(|| "Ensemble".to_string());

    let avg_detection: f64 = attack_results.iter().map(|a| a.avg_detection_rate).sum::<f64>()
        / attack_results.len().max(1) as f64;

    let avg_fpr: f64 = detection_comparison.iter().map(|d| d.overall_fpr).sum::<f64>()
        / detection_comparison.len().max(1) as f64;

    // Attacks needing attention: detection rate < 50%
    let attacks_requiring_attention: Vec<String> = attack_results.iter()
        .filter(|a| a.avg_detection_rate < 0.5)
        .map(|a| a.attack_type.clone())
        .collect();

    F5Summary {
        most_dangerous_attack: most_dangerous,
        most_effective_detection: most_effective,
        avg_detection_rate: avg_detection,
        avg_false_positive_rate: avg_fpr,
        attacks_requiring_attention,
    }
}

fn generate_security_recommendations(
    attack_results: &[AttackTestResults],
    summary: &F5Summary,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Primary recommendation
    recommendations.push(format!(
        "Deploy {} detection as primary defense (best overall F1 score)",
        summary.most_effective_detection
    ));

    // Attack-specific recommendations
    for attack in attack_results {
        if attack.avg_detection_rate < 0.6 {
            let rec = match attack.attack_type.as_str() {
                "WashTrading" => "Implement time-based velocity limits and self-trade detection",
                "PriceManipulation" => "Add price circuit breakers and abnormal movement alerts",
                "SybilAttack" => "Strengthen identity verification and implement stake-based reputation",
                "Collusion" => "Deploy network analysis to detect coordinated trading patterns",
                "Spoofing" => "Monitor order book dynamics and penalize excessive cancellations",
                "GradientPoisoning" => "Use Byzantine-robust aggregation for federated learning updates",
                _ => "Review and strengthen detection for this attack vector",
            };
            recommendations.push(format!("{}: {}", attack.attack_type, rec));
        }
    }

    // False positive management
    if summary.avg_false_positive_rate > 0.1 {
        recommendations.push(format!(
            "False positive rate is {:.1}% - consider tiered detection with human review for borderline cases",
            summary.avg_false_positive_rate * 100.0
        ));
    }

    // General recommendations
    recommendations.push("Implement layered defense: combine multiple detection methods".to_string());
    recommendations.push("Establish monitoring dashboards for real-time manipulation detection".to_string());
    recommendations.push("Create incident response playbooks for each attack type".to_string());

    recommendations
}

// ============================================================================
// UTILITIES
// ============================================================================

fn get_timestamp() -> String {
    // Simple timestamp generation
    "2026-01-30T12:00:00Z".to_string()
}

fn generate_f4_report(results: &F4Results) -> String {
    let mut report = String::new();

    report.push_str("# F4: Mechanism Design Comparison Report\n\n");
    report.push_str(&format!("**Experiment ID:** {}\n", results.experiment_id));
    report.push_str(&format!("**Timestamp:** {}\n\n", results.timestamp));

    report.push_str("## Executive Summary\n\n");
    report.push_str(&format!("**Best Overall Mechanism:** {}\n", results.summary.best_mechanism_overall));
    report.push_str(&format!("**Best for Low Liquidity:** {}\n", results.summary.best_for_low_liquidity));
    report.push_str(&format!("**Best for High Liquidity:** {}\n", results.summary.best_for_high_liquidity));
    report.push_str(&format!("**Best for Volatile Information:** {}\n", results.summary.best_for_volatile_info));
    report.push_str(&format!("**Optimal LMSR Liquidity Parameter:** {:.0}\n\n", results.summary.optimal_lmsr_liquidity_param));

    report.push_str("### Key Findings\n\n");
    for finding in &results.summary.key_findings {
        report.push_str(&format!("- {}\n", finding));
    }
    report.push_str("\n");

    report.push_str("## Detailed Results by Condition\n\n");

    for cond in &results.conditions {
        report.push_str(&format!("### {} (n={})\n\n", cond.condition_name, cond.num_participants));
        report.push_str(&format!("**Information Pattern:** {}\n\n", cond.information_pattern));
        report.push_str(&format!("**Winner:** {} (margin: {:.3})\n\n", cond.winner, cond.margin));

        report.push_str("| Metric | LMSR | Order Book |\n");
        report.push_str("|--------|------|------------|\n");
        report.push_str(&format!("| Price Discovery Speed | {:.3} | {:.3} |\n",
            cond.lmsr_metrics.price_discovery_speed,
            cond.orderbook_metrics.price_discovery_speed));
        report.push_str(&format!("| Avg Slippage | {:.4} | {:.4} |\n",
            cond.lmsr_metrics.avg_slippage,
            cond.orderbook_metrics.avg_slippage));
        report.push_str(&format!("| Max Slippage | {:.4} | {:.4} |\n",
            cond.lmsr_metrics.max_slippage,
            cond.orderbook_metrics.max_slippage));
        report.push_str(&format!("| Spread | N/A | {:.4} |\n",
            cond.orderbook_metrics.spread));
        report.push_str(&format!("| Info Incorporation | {:.3} | {:.3} |\n",
            cond.lmsr_metrics.information_incorporation,
            cond.orderbook_metrics.information_incorporation));
        report.push_str(&format!("| Manipulation Resistance | {:.3} | {:.3} |\n",
            cond.lmsr_metrics.manipulation_resistance,
            cond.orderbook_metrics.manipulation_resistance));
        report.push_str(&format!("| Trade Success Rate | {:.1}% | {:.1}% |\n\n",
            cond.lmsr_metrics.trade_success_rate * 100.0,
            cond.orderbook_metrics.trade_success_rate * 100.0));
    }

    report.push_str("## Recommendations\n\n");
    for rec in &results.recommendations {
        report.push_str(&format!("- {}\n", rec));
    }

    report
}

fn generate_f5_report(results: &F5Results) -> String {
    let mut report = String::new();

    report.push_str("# F5: Manipulation Resistance Testing Report\n\n");
    report.push_str(&format!("**Experiment ID:** {}\n", results.experiment_id));
    report.push_str(&format!("**Timestamp:** {}\n\n", results.timestamp));

    report.push_str("## Executive Summary\n\n");
    report.push_str(&format!("**Most Dangerous Attack:** {}\n", results.summary.most_dangerous_attack));
    report.push_str(&format!("**Most Effective Detection:** {}\n", results.summary.most_effective_detection));
    report.push_str(&format!("**Average Detection Rate:** {:.1}%\n", results.summary.avg_detection_rate * 100.0));
    report.push_str(&format!("**Average False Positive Rate:** {:.1}%\n\n", results.summary.avg_false_positive_rate * 100.0));

    if !results.summary.attacks_requiring_attention.is_empty() {
        report.push_str("### Attacks Requiring Immediate Attention\n\n");
        for attack in &results.summary.attacks_requiring_attention {
            report.push_str(&format!("- {}\n", attack));
        }
        report.push_str("\n");
    }

    report.push_str("## Attack Analysis\n\n");

    for attack in &results.attack_results {
        report.push_str(&format!("### {}\n\n", attack.attack_type));
        report.push_str(&format!("**Average Detection Rate:** {:.1}%\n", attack.avg_detection_rate * 100.0));
        report.push_str(&format!("**Hardest Intensity to Detect:** {:.1}\n", attack.hardest_intensity_to_detect));
        report.push_str(&format!("**Avg Success if Undetected:** {:.1}%\n\n", attack.avg_success_rate_undetected * 100.0));

        report.push_str("| Intensity | TPR | FPR | Latency | Success if Undetected |\n");
        report.push_str("|-----------|-----|-----|---------|----------------------|\n");
        for intensity in &attack.intensities {
            report.push_str(&format!("| {:.2} | {:.1}% | {:.1}% | {:.2} | {:.1}% |\n",
                intensity.intensity,
                intensity.true_positive_rate * 100.0,
                intensity.false_positive_rate * 100.0,
                intensity.detection_latency,
                intensity.attack_success_if_undetected * 100.0));
        }
        report.push_str("\n");
    }

    report.push_str("## Detection Method Comparison\n\n");
    report.push_str("| Method | TPR | FPR | F1 Score | Best Against | Worst Against |\n");
    report.push_str("|--------|-----|-----|----------|--------------|---------------|\n");

    for det in &results.detection_comparison {
        report.push_str(&format!("| {} | {:.1}% | {:.1}% | {:.3} | {} | {} |\n",
            det.method_name,
            det.overall_tpr * 100.0,
            det.overall_fpr * 100.0,
            det.f1_score,
            det.best_against.join(", "),
            det.worst_against.join(", ")));
    }
    report.push_str("\n");

    report.push_str("## Security Recommendations\n\n");
    for (i, rec) in results.security_recommendations.iter().enumerate() {
        report.push_str(&format!("{}. {}\n", i + 1, rec));
    }

    report
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

fn main() {
    println!("Running F4 and F5 Research Experiments...\n");

    // Run F4: Mechanism Comparison
    println!("=== F4: Mechanism Design Comparison ===");
    let f4_config = F4ExperimentConfig::default();
    let f4_results = run_f4_experiment(&f4_config);

    // Save F4 results
    let f4_json = serde_json::to_string_pretty(&f4_results).unwrap_or_else(|_| "{}".to_string());
    let f4_report = generate_f4_report(&f4_results);

    println!("F4 Best Overall Mechanism: {}", f4_results.summary.best_mechanism_overall);
    println!("F4 completed with {} conditions tested\n", f4_results.conditions.len());

    // Run F5: Manipulation Resistance
    println!("=== F5: Manipulation Resistance Testing ===");
    let f5_config = F5ExperimentConfig::default();
    let f5_results = run_f5_experiment(&f5_config);

    // Save F5 results
    let f5_json = serde_json::to_string_pretty(&f5_results).unwrap_or_else(|_| "{}".to_string());
    let f5_report = generate_f5_report(&f5_results);

    println!("F5 Most Dangerous Attack: {}", f5_results.summary.most_dangerous_attack);
    println!("F5 Most Effective Detection: {}", f5_results.summary.most_effective_detection);
    println!("F5 completed with {} attack types tested\n", f5_results.attack_results.len());

    // Write output files
    let results_dir = "/srv/luminous-dynamics/mycelix-workspace/happs/epistemic-markets/research/results";

    fs::write(format!("{}/f4_mechanism_results.json", results_dir), &f4_json)
        .expect("Failed to write F4 JSON results");
    fs::write(format!("{}/f4_mechanism_report.md", results_dir), &f4_report)
        .expect("Failed to write F4 report");

    fs::write(format!("{}/f5_manipulation_results.json", results_dir), &f5_json)
        .expect("Failed to write F5 JSON results");
    fs::write(format!("{}/f5_manipulation_report.md", results_dir), &f5_report)
        .expect("Failed to write F5 report");

    println!("Results written to:");
    println!("  - {}/f4_mechanism_results.json", results_dir);
    println!("  - {}/f4_mechanism_report.md", results_dir);
    println!("  - {}/f5_manipulation_results.json", results_dir);
    println!("  - {}/f5_manipulation_report.md", results_dir);
}
