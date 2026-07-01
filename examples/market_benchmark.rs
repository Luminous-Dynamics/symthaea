// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Market Pattern Recognition Benchmark
//!
//! Demonstrates HDC-based market pattern recognition:
//! - Generate synthetic market data with known regimes
//! - Train pattern memory on historical data
//! - Backtest prediction accuracy
//! - Compare vs random baseline
//!
//! Run with: cargo run --example market_benchmark

use std::collections::HashMap;
use symthaea::markets::{
    MarketRegime, MarketSimulator, MarketStateEncoder, OHLCV, PatternMemory, SimulatorConfig,
    TechnicalIndicators,
};

fn main() {
    println!("=== Symthaea Market Pattern Recognition Benchmark ===\n");

    // Configuration
    let train_days = 500;
    let test_days = 200;
    let lookback = 10;

    println!("Configuration:");
    println!("  Training period: {} days", train_days);
    println!("  Test period: {} days", test_days);
    println!("  Pattern lookback: {} candles", lookback);
    println!();

    // Generate training data
    println!("--- Generating Training Data ---");
    let mut train_sim = MarketSimulator::new(
        SimulatorConfig {
            initial_price: 100.0,
            base_volatility: 2.0,
            regime_persistence: 0.92,
            ..Default::default()
        },
        42,
    );

    let train_candles = train_sim.generate(train_days);
    let train_indicators = train_sim.calculate_indicators();

    // Count regime distribution in training
    let mut regime_counts: HashMap<String, u32> = HashMap::new();
    for _ in 0..train_days {
        train_sim.next_candle();
        *regime_counts
            .entry(format!("{:?}", train_sim.regime()))
            .or_insert(0) += 1;
    }
    train_sim.reset(42);
    train_sim.generate(train_days);

    println!("Training data regime distribution:");
    for (regime, count) in &regime_counts {
        println!(
            "  {}: {} ({:.1}%)",
            regime,
            count,
            *count as f64 / train_days as f64 * 100.0
        );
    }

    // Train pattern memory
    println!("\n--- Training Pattern Memory ---");
    let mut memory = PatternMemory::new();
    memory.set_threshold(0.5);

    let encoder = MarketStateEncoder::new();
    let mut patterns_stored = 0;

    for i in lookback..train_candles.len() - 1 {
        let window_candles = &train_candles[i - lookback..i];
        let window_indicators = &train_indicators[i - lookback..i];

        // Determine regime from recent price action
        let regime = classify_window(window_candles, &train_indicators[i]);

        // Calculate outcome (next day return)
        let outcome = train_candles[i].change_pct() as f32;

        // Store pattern
        let label = format!("{:?}_{}", regime, i);
        memory.store_pattern(window_candles, window_indicators, regime, outcome, &label);
        patterns_stored += 1;
    }

    println!("Stored {} patterns", patterns_stored);
    println!("Unique patterns: {}", memory.pattern_count());
    println!("Regime prototypes: {}", memory.regime_count());

    // Generate test data
    println!("\n--- Generating Test Data ---");
    let mut test_sim = MarketSimulator::new(
        SimulatorConfig {
            initial_price: train_sim.price(),
            base_volatility: 2.0,
            regime_persistence: 0.92,
            ..Default::default()
        },
        12345, // Different seed
    );

    let test_candles = test_sim.generate(test_days);
    let test_indicators = test_sim.calculate_indicators();

    println!(
        "Test data: {} candles, final price: {:.2}",
        test_candles.len(),
        test_sim.price()
    );

    // Benchmark: HDC Pattern Matching
    println!("\n--- HDC Pattern Matching Benchmark ---");
    let hdc_results = run_benchmark(&memory, &test_candles, &test_indicators, lookback);
    print_results("HDC Pattern Matching", &hdc_results);

    // Benchmark: Random Baseline
    println!("\n--- Random Baseline ---");
    let random_results = run_random_baseline(&test_candles, lookback, 54321);
    print_results("Random Baseline", &random_results);

    // Benchmark: Trend Following (simple baseline)
    println!("\n--- Trend Following Baseline ---");
    let trend_results = run_trend_following(&test_candles, &test_indicators, lookback);
    print_results("Trend Following", &trend_results);

    // Summary comparison
    println!("\n=== Summary Comparison ===");
    println!(
        "{:<25} {:>12} {:>12} {:>12} {:>12}",
        "Strategy", "Return %", "Win Rate", "Sharpe", "Max DD %"
    );
    println!("{:-<73}", "");
    print_summary_row("HDC Pattern Matching", &hdc_results);
    print_summary_row("Random Baseline", &random_results);
    print_summary_row("Trend Following", &trend_results);

    // Regime classification accuracy
    println!("\n--- Regime Classification Accuracy ---");
    test_regime_classification(&memory, &test_candles, &test_indicators, lookback);
}

/// Classify a window of candles into a regime
fn classify_window(candles: &[OHLCV], indicators: &TechnicalIndicators) -> MarketRegime {
    let recent_change: f64 = candles.iter().map(|c| c.change_pct()).sum();
    let avg_change = recent_change / candles.len() as f64;

    let volatility = indicators.atr / indicators.sma_short.max(1.0) * 100.0;

    if avg_change > 1.0 && indicators.sma_short > indicators.sma_long {
        MarketRegime::BullTrend
    } else if avg_change < -1.0 && indicators.sma_short < indicators.sma_long {
        MarketRegime::BearTrend
    } else if volatility > 3.0 {
        MarketRegime::Volatile
    } else if volatility < 1.0 {
        MarketRegime::Quiet
    } else if avg_change > 0.5 && volatility > 2.0 {
        MarketRegime::Breakout
    } else if avg_change < -0.5 && volatility > 2.0 {
        MarketRegime::Breakdown
    } else {
        MarketRegime::Ranging
    }
}

#[derive(Default)]
struct BenchmarkResults {
    total_return: f64,
    trade_count: u32,
    win_count: u32,
    max_drawdown: f64,
    returns: Vec<f64>,
}

impl BenchmarkResults {
    fn win_rate(&self) -> f64 {
        if self.trade_count > 0 {
            self.win_count as f64 / self.trade_count as f64
        } else {
            0.0
        }
    }

    fn sharpe_ratio(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        let avg: f64 = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let variance: f64 =
            self.returns.iter().map(|r| (r - avg).powi(2)).sum::<f64>() / self.returns.len() as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            avg / std_dev * (252.0_f64).sqrt()
        } else {
            0.0
        }
    }
}

/// Run HDC pattern matching benchmark
fn run_benchmark(
    memory: &PatternMemory,
    candles: &[OHLCV],
    indicators: &[TechnicalIndicators],
    lookback: usize,
) -> BenchmarkResults {
    let mut results = BenchmarkResults::default();
    let mut equity: f64 = 1.0;
    let mut peak: f64 = 1.0;

    let mut skipped_low_conf = 0;
    let mut skipped_neutral = 0;

    for i in lookback..candles.len() - 1 {
        let window_candles = &candles[i - lookback..i];
        let window_indicators = &indicators[i - lookback..i];

        // Get prediction
        let (predicted, confidence) = memory.predict_outcome(window_candles, window_indicators);

        // Skip low confidence predictions (lowered threshold)
        if confidence < 0.05 {
            skipped_low_conf += 1;
            continue;
        }

        // Position based on prediction direction (more sensitive)
        let position = if predicted > 0.1 {
            1.0
        } else if predicted < -0.1 {
            -1.0
        } else {
            skipped_neutral += 1;
            continue; // Skip neutral
        };

        let actual_return = candles[i + 1].change_pct() / 100.0;
        let trade_return = position * actual_return * confidence as f64;

        results.returns.push(trade_return);
        results.trade_count += 1;
        if trade_return > 0.0 {
            results.win_count += 1;
        }

        equity *= 1.0 + trade_return;
        peak = peak.max(equity);
        let dd = (peak - equity) / peak;
        results.max_drawdown = results.max_drawdown.max(dd);
    }

    results.total_return = (equity - 1.0) * 100.0;

    // Debug output
    if results.trade_count == 0 {
        println!(
            "  [DEBUG] Skipped {} low confidence, {} neutral predictions",
            skipped_low_conf, skipped_neutral
        );
    }

    results
}

/// Run random baseline
fn run_random_baseline(candles: &[OHLCV], lookback: usize, seed: u64) -> BenchmarkResults {
    let mut results = BenchmarkResults::default();
    let mut equity: f64 = 1.0;
    let mut peak: f64 = 1.0;
    let mut rng_state = seed;

    for i in lookback..candles.len() - 1 {
        // Random position
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        let position = if rng_state % 3 == 0 {
            1.0
        } else if rng_state % 3 == 1 {
            -1.0
        } else {
            continue; // Skip
        };

        let actual_return = candles[i + 1].change_pct() / 100.0;
        let trade_return = position * actual_return;

        results.returns.push(trade_return);
        results.trade_count += 1;
        if trade_return > 0.0 {
            results.win_count += 1;
        }

        equity *= 1.0 + trade_return;
        peak = peak.max(equity);
        let dd = (peak - equity) / peak;
        results.max_drawdown = results.max_drawdown.max(dd);
    }

    results.total_return = (equity - 1.0) * 100.0;
    results
}

/// Run simple trend following baseline
fn run_trend_following(
    candles: &[OHLCV],
    indicators: &[TechnicalIndicators],
    lookback: usize,
) -> BenchmarkResults {
    let mut results = BenchmarkResults::default();
    let mut equity: f64 = 1.0;
    let mut peak: f64 = 1.0;

    for i in lookback..candles.len() - 1 {
        let ind = &indicators[i];

        // Simple trend following: long if short MA > long MA
        let position = if ind.sma_short > ind.sma_long * 1.01 {
            1.0
        } else if ind.sma_short < ind.sma_long * 0.99 {
            -1.0
        } else {
            continue;
        };

        let actual_return = candles[i + 1].change_pct() / 100.0;
        let trade_return = position * actual_return;

        results.returns.push(trade_return);
        results.trade_count += 1;
        if trade_return > 0.0 {
            results.win_count += 1;
        }

        equity *= 1.0 + trade_return;
        peak = peak.max(equity);
        let dd = (peak - equity) / peak;
        results.max_drawdown = results.max_drawdown.max(dd);
    }

    results.total_return = (equity - 1.0) * 100.0;
    results
}

/// Test regime classification accuracy
fn test_regime_classification(
    memory: &PatternMemory,
    candles: &[OHLCV],
    indicators: &[TechnicalIndicators],
    lookback: usize,
) {
    let mut correct = 0;
    let mut total = 0;
    let mut regime_accuracy: HashMap<String, (u32, u32)> = HashMap::new();

    for i in lookback..candles.len() {
        let window_candles = &candles[i - lookback..i];
        let window_indicators = &indicators[i - lookback..i];

        // Actual regime based on heuristics
        let actual = classify_window(window_candles, &indicators[i]);

        // Predicted regime
        let (predicted, confidence) = memory.classify_regime(window_candles, window_indicators);

        let actual_name = format!("{:?}", actual);
        let entry = regime_accuracy.entry(actual_name.clone()).or_insert((0, 0));
        entry.1 += 1;

        if actual == predicted && confidence > 0.3 {
            correct += 1;
            entry.0 += 1;
        }
        total += 1;
    }

    println!(
        "Overall accuracy: {:.1}% ({}/{})",
        correct as f64 / total as f64 * 100.0,
        correct,
        total
    );
    println!("\nPer-regime accuracy:");
    for (regime, (correct, total)) in &regime_accuracy {
        if *total > 0 {
            println!(
                "  {}: {:.1}% ({}/{})",
                regime,
                *correct as f64 / *total as f64 * 100.0,
                correct,
                total
            );
        }
    }
}

fn print_results(name: &str, results: &BenchmarkResults) {
    println!("{}:", name);
    println!("  Total Return: {:.2}%", results.total_return);
    println!("  Trade Count: {}", results.trade_count);
    println!("  Win Rate: {:.1}%", results.win_rate() * 100.0);
    println!("  Sharpe Ratio: {:.2}", results.sharpe_ratio());
    println!("  Max Drawdown: {:.2}%", results.max_drawdown * 100.0);
}

fn print_summary_row(name: &str, results: &BenchmarkResults) {
    println!(
        "{:<25} {:>11.2}% {:>11.1}% {:>12.2} {:>11.2}%",
        name,
        results.total_return,
        results.win_rate() * 100.0,
        results.sharpe_ratio(),
        results.max_drawdown * 100.0
    );
}
