// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Market Scenarios Benchmark
//!
//! Tests HDC pattern recognition across different market conditions:
//! - Bull market (strong uptrend)
//! - Bear market (strong downtrend)
//! - High volatility (choppy)
//! - Low volatility (quiet consolidation)
//!
//! Also tests different HDC parameters:
//! - Similarity thresholds
//! - Lookback windows
//! - Pattern memory sizes
//!
//! Run with: cargo run --example market_scenarios

use std::collections::HashMap;
use symthaea::markets::{
    MarketRegime, MarketSimulator, OHLCV, PatternMemory, SimulatorConfig, TechnicalIndicators,
};

fn main() {
    println!("=== Market Scenarios & HDC Parameter Tuning ===\n");

    // Test different market conditions
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    MARKET CONDITION TESTS                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let scenarios = vec![
        (
            "Bull Market",
            SimulatorConfig {
                initial_price: 100.0,
                base_volatility: 1.5,
                mean_return: 0.08, // Strong positive drift
                regime_persistence: 0.95,
                trend_strength: 2.0,
                ..Default::default()
            },
        ),
        (
            "Bear Market",
            SimulatorConfig {
                initial_price: 100.0,
                base_volatility: 2.0,
                mean_return: -0.06, // Negative drift
                regime_persistence: 0.93,
                trend_strength: 1.8,
                ..Default::default()
            },
        ),
        (
            "High Volatility",
            SimulatorConfig {
                initial_price: 100.0,
                base_volatility: 4.0, // High vol
                mean_return: 0.01,
                regime_persistence: 0.85, // Frequent regime changes
                trend_strength: 1.0,
                ..Default::default()
            },
        ),
        (
            "Low Volatility",
            SimulatorConfig {
                initial_price: 100.0,
                base_volatility: 0.8, // Low vol
                mean_return: 0.02,
                regime_persistence: 0.98, // Stable regimes
                trend_strength: 0.5,
                ..Default::default()
            },
        ),
    ];

    let mut scenario_results: Vec<(&str, ScenarioResult)> = Vec::new();

    for (name, config) in &scenarios {
        println!("--- {} ---", name);
        let result = run_scenario(config, 200, 100, 10);
        print_scenario_result(&result);
        scenario_results.push((name, result));
        println!();
    }

    // Summary table
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                  SCENARIO COMPARISON SUMMARY                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Scenario", "HDC Ret%", "Trend Ret%", "HDC Win%", "HDC Sharpe", "HDC DD%"
    );
    println!("{:-<78}", "");

    for (name, result) in &scenario_results {
        println!(
            "{:<18} {:>9.2}% {:>9.2}% {:>9.1}% {:>10.2} {:>9.2}%",
            name,
            result.hdc_return,
            result.trend_return,
            result.hdc_win_rate * 100.0,
            result.hdc_sharpe,
            result.hdc_max_dd * 100.0
        );
    }

    // NOTE: Parameter tuning sweeps commented out for speed
    // Run `cargo run --example market_benchmark` for detailed parameter analysis

    println!("\n=== Analysis Complete ===");
}

#[derive(Default)]
struct ScenarioResult {
    hdc_return: f64,
    hdc_win_rate: f64,
    hdc_sharpe: f64,
    hdc_max_dd: f64,
    hdc_trades: u32,
    trend_return: f64,
    trend_win_rate: f64,
    random_return: f64,
    final_price: f64,
    regime_accuracy: f64,
}

fn run_scenario(
    config: &SimulatorConfig,
    train_days: usize,
    test_days: usize,
    lookback: usize,
) -> ScenarioResult {
    // Generate training data
    let mut train_sim = MarketSimulator::new(config.clone(), 42);
    let train_candles = train_sim.generate(train_days);
    let train_indicators = train_sim.calculate_indicators();

    // Train pattern memory
    let mut memory = PatternMemory::new();
    memory.set_threshold(0.4);

    for i in lookback..train_candles.len() - 1 {
        let window_candles = &train_candles[i - lookback..i];
        let window_indicators = &train_indicators[i - lookback..i];
        let regime = classify_window(window_candles, &train_indicators[i]);
        let outcome = train_candles[i].change_pct() as f32;
        memory.store_pattern(
            window_candles,
            window_indicators,
            regime,
            outcome,
            &format!("p{}", i),
        );
    }

    // Generate test data with different seed
    let mut test_sim = MarketSimulator::new(
        SimulatorConfig {
            initial_price: train_sim.price(),
            ..config.clone()
        },
        12345,
    );
    let test_candles = test_sim.generate(test_days);
    let test_indicators = test_sim.calculate_indicators();

    // Run HDC benchmark
    let hdc = run_hdc_benchmark(&memory, &test_candles, &test_indicators, lookback);

    // Run trend following
    let trend = run_trend_benchmark(&test_candles, &test_indicators, lookback);

    // Run random
    let random = run_random_benchmark(&test_candles, lookback, 99999);

    // Calculate regime accuracy
    let regime_acc = calculate_regime_accuracy(&memory, &test_candles, &test_indicators, lookback);

    ScenarioResult {
        hdc_return: hdc.0,
        hdc_win_rate: hdc.2,
        hdc_sharpe: hdc.3,
        hdc_max_dd: hdc.4,
        hdc_trades: hdc.1,
        trend_return: trend.0,
        trend_win_rate: trend.2,
        random_return: random.0,
        final_price: test_sim.price(),
        regime_accuracy: regime_acc,
    }
}

fn run_hdc_benchmark(
    memory: &PatternMemory,
    candles: &[OHLCV],
    indicators: &[TechnicalIndicators],
    lookback: usize,
) -> (f64, u32, f64, f64, f64) {
    let mut equity: f64 = 1.0;
    let mut peak: f64 = 1.0;
    let mut max_dd: f64 = 0.0;
    let mut returns: Vec<f64> = Vec::new();
    let mut wins = 0u32;
    let mut trades = 0u32;

    for i in lookback..candles.len() - 1 {
        let window_candles = &candles[i - lookback..i];
        let window_indicators = &indicators[i - lookback..i];

        let (predicted, confidence) = memory.predict_outcome(window_candles, window_indicators);

        if confidence < 0.05 {
            continue;
        }

        let position = if predicted > 0.1 {
            1.0
        } else if predicted < -0.1 {
            -1.0
        } else {
            continue;
        };

        let actual = candles[i + 1].change_pct() / 100.0;
        let ret = position * actual * (confidence as f64).min(1.0);

        returns.push(ret);
        trades += 1;
        if ret > 0.0 {
            wins += 1;
        }

        equity *= 1.0 + ret;
        peak = peak.max(equity);
        max_dd = max_dd.max((peak - equity) / peak);
    }

    let total_return = (equity - 1.0) * 100.0;
    let win_rate = if trades > 0 {
        wins as f64 / trades as f64
    } else {
        0.0
    };
    let sharpe = calculate_sharpe(&returns);

    (total_return, trades, win_rate, sharpe, max_dd)
}

fn run_trend_benchmark(
    candles: &[OHLCV],
    indicators: &[TechnicalIndicators],
    lookback: usize,
) -> (f64, u32, f64, f64, f64) {
    let mut equity: f64 = 1.0;
    let mut peak: f64 = 1.0;
    let mut max_dd: f64 = 0.0;
    let mut returns: Vec<f64> = Vec::new();
    let mut wins = 0u32;
    let mut trades = 0u32;

    for i in lookback..candles.len() - 1 {
        let ind = &indicators[i];

        let position = if ind.sma_short > ind.sma_long * 1.005 {
            1.0
        } else if ind.sma_short < ind.sma_long * 0.995 {
            -1.0
        } else {
            continue;
        };

        let actual = candles[i + 1].change_pct() / 100.0;
        let ret = position * actual;

        returns.push(ret);
        trades += 1;
        if ret > 0.0 {
            wins += 1;
        }

        equity *= 1.0 + ret;
        peak = peak.max(equity);
        max_dd = max_dd.max((peak - equity) / peak);
    }

    let total_return = (equity - 1.0) * 100.0;
    let win_rate = if trades > 0 {
        wins as f64 / trades as f64
    } else {
        0.0
    };
    let sharpe = calculate_sharpe(&returns);

    (total_return, trades, win_rate, sharpe, max_dd)
}

fn run_random_benchmark(
    candles: &[OHLCV],
    lookback: usize,
    seed: u64,
) -> (f64, u32, f64, f64, f64) {
    let mut equity: f64 = 1.0;
    let mut rng = seed;

    for i in lookback..candles.len() - 1 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;

        let position = match rng % 3 {
            0 => 1.0,
            1 => -1.0,
            _ => continue,
        };

        let actual = candles[i + 1].change_pct() / 100.0;
        equity *= 1.0 + position * actual;
    }

    ((equity - 1.0) * 100.0, 0, 0.0, 0.0, 0.0)
}

fn run_with_threshold(
    config: &SimulatorConfig,
    train_days: usize,
    test_days: usize,
    lookback: usize,
    threshold: f32,
) -> (f64, u32, f64, f64) {
    let mut train_sim = MarketSimulator::new(config.clone(), 42);
    let train_candles = train_sim.generate(train_days);
    let train_indicators = train_sim.calculate_indicators();

    let mut memory = PatternMemory::new();
    memory.set_threshold(threshold);

    for i in lookback..train_candles.len() - 1 {
        let window_candles = &train_candles[i - lookback..i];
        let window_indicators = &train_indicators[i - lookback..i];
        let regime = classify_window(window_candles, &train_indicators[i]);
        let outcome = train_candles[i].change_pct() as f32;
        memory.store_pattern(
            window_candles,
            window_indicators,
            regime,
            outcome,
            &format!("p{}", i),
        );
    }

    let mut test_sim = MarketSimulator::new(
        SimulatorConfig {
            initial_price: train_sim.price(),
            ..config.clone()
        },
        12345,
    );
    let test_candles = test_sim.generate(test_days);
    let test_indicators = test_sim.calculate_indicators();

    let result = run_hdc_benchmark(&memory, &test_candles, &test_indicators, lookback);
    (result.0, result.1, result.2, result.3)
}

fn run_with_lookback(
    config: &SimulatorConfig,
    train_days: usize,
    test_days: usize,
    lookback: usize,
) -> (f64, u32, f64, f64) {
    run_with_threshold(config, train_days, test_days, lookback, 0.5)
}

fn run_with_train_size(
    config: &SimulatorConfig,
    train_days: usize,
    test_days: usize,
    lookback: usize,
) -> (f64, u32, f64, f64, usize) {
    let mut train_sim = MarketSimulator::new(config.clone(), 42);
    let train_candles = train_sim.generate(train_days);
    let train_indicators = train_sim.calculate_indicators();

    let mut memory = PatternMemory::new();
    memory.set_threshold(0.5);

    for i in lookback..train_candles.len() - 1 {
        let window_candles = &train_candles[i - lookback..i];
        let window_indicators = &train_indicators[i - lookback..i];
        let regime = classify_window(window_candles, &train_indicators[i]);
        let outcome = train_candles[i].change_pct() as f32;
        memory.store_pattern(
            window_candles,
            window_indicators,
            regime,
            outcome,
            &format!("p{}", i),
        );
    }

    let pattern_count = memory.pattern_count();

    let mut test_sim = MarketSimulator::new(
        SimulatorConfig {
            initial_price: train_sim.price(),
            ..config.clone()
        },
        12345,
    );
    let test_candles = test_sim.generate(test_days);
    let test_indicators = test_sim.calculate_indicators();

    let result = run_hdc_benchmark(&memory, &test_candles, &test_indicators, lookback);
    (result.0, result.1, result.2, result.3, pattern_count)
}

fn calculate_regime_accuracy(
    memory: &PatternMemory,
    candles: &[OHLCV],
    indicators: &[TechnicalIndicators],
    lookback: usize,
) -> f64 {
    let mut correct = 0;
    let mut total = 0;

    for i in lookback..candles.len() {
        let window_candles = &candles[i - lookback..i];
        let window_indicators = &indicators[i - lookback..i];

        let actual = classify_window(window_candles, &indicators[i]);
        let (predicted, confidence) = memory.classify_regime(window_candles, window_indicators);

        if actual == predicted && confidence > 0.3 {
            correct += 1;
        }
        total += 1;
    }

    if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    }
}

fn calculate_sharpe(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let avg: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let var: f64 = returns.iter().map(|r| (r - avg).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = var.sqrt();
    if std > 0.0 {
        avg / std * (252.0_f64).sqrt()
    } else {
        0.0
    }
}

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
    } else {
        MarketRegime::Ranging
    }
}

fn print_scenario_result(result: &ScenarioResult) {
    println!("  Final price: ${:.2}", result.final_price);
    println!(
        "  HDC: {:.2}% return, {:.1}% win rate, {:.2} Sharpe, {:.2}% max DD ({} trades)",
        result.hdc_return,
        result.hdc_win_rate * 100.0,
        result.hdc_sharpe,
        result.hdc_max_dd * 100.0,
        result.hdc_trades
    );
    println!(
        "  Trend: {:.2}% return, {:.1}% win rate",
        result.trend_return,
        result.trend_win_rate * 100.0
    );
    println!("  Random: {:.2}% return", result.random_return);
    println!("  Regime accuracy: {:.1}%", result.regime_accuracy * 100.0);
}
