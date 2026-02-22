//! Diagnostic tests for symthaea-flight learning dynamics.
//!
//! Three tests:
//! 1. Training curve: 10 episodes x 500 steps, per-episode metrics
//! 2. Wind benchmark: 3 episodes x 1000 steps, baseline vs FEP comparison
//! 3. Within-episode learning curve: 1 episode x 1000 steps with telemetry

use symthaea_core::genesis::GenesisSeed;
use symthaea_flight::benchmarks::{WindBenchmarkConfig, WindGust, run_wind_benchmark};
use symthaea_flight::fep_agent::FlightFepConfig;
use symthaea_flight::training::FlightTrainer;
use symthaea_flight::{
    FlightConfig, FlightController, QuadrotorHdcEncoder, ActiveInferenceFlightAgent,
};

#[test]
#[ignore] // ~30s: run manually with `cargo test -p symthaea-flight -- --ignored --nocapture`
fn diagnostic_training_curve() {
    println!("\n{}", "=".repeat(80));
    println!("=== DIAGNOSTIC 1: Training Curve (10 episodes x 500 steps) ===");
    println!("{}\n", "=".repeat(80));

    let config = FlightConfig {
        num_episodes: 10,
        steps_per_episode: 500,
        ..FlightConfig::default()
    };

    let mut trainer = FlightTrainer::new(config);
    let metrics = trainer.train();

    println!(
        "{:<8} {:<20} {:<20} {:<18} {:<15} {:<18}",
        "Episode", "Avg Pos Error", "Final Pos Error", "Avg Free Energy", "Hover Frac", "Explore Count"
    );
    println!("{}", "-".repeat(99));

    for m in &metrics {
        println!(
            "{:<8} {:<20.6} {:<20.6} {:<18.6} {:<15.4} {:<18}",
            m.episode,
            m.avg_position_error,
            m.final_position_error,
            m.avg_free_energy,
            m.hover_fraction,
            m.exploration_count,
        );
    }

    // Summary statistics
    let first_3_avg: f64 = metrics[..3].iter().map(|m| m.avg_position_error).sum::<f64>() / 3.0;
    let last_3_avg: f64 = metrics[7..].iter().map(|m| m.avg_position_error).sum::<f64>() / 3.0;
    let best_ep = metrics.iter().min_by(|a, b| a.avg_position_error.total_cmp(&b.avg_position_error)).unwrap();
    let worst_ep = metrics.iter().max_by(|a, b| a.avg_position_error.total_cmp(&b.avg_position_error)).unwrap();

    println!("\n--- Summary ---");
    println!("First 3 episodes avg error:  {:.6}", first_3_avg);
    println!("Last 3 episodes avg error:   {:.6}", last_3_avg);
    println!("Best episode:  {} (avg_err={:.6})", best_ep.episode, best_ep.avg_position_error);
    println!("Worst episode: {} (avg_err={:.6})", worst_ep.episode, worst_ep.avg_position_error);
    println!("Improvement ratio (first3/last3): {:.3}", first_3_avg / last_3_avg);

    let total_explorations: usize = metrics.iter().map(|m| m.exploration_count).sum();
    println!("Total explorations across all episodes: {}", total_explorations);
    println!();
}

#[test]
#[ignore] // ~30s: run manually with `cargo test -p symthaea-flight -- --ignored --nocapture`
fn diagnostic_wind_benchmark() {
    println!("\n{}", "=".repeat(80));
    println!("=== DIAGNOSTIC 2: Wind Benchmark (3 episodes x 1000 steps) ===");
    println!("{}\n", "=".repeat(80));

    let wind_config = WindBenchmarkConfig {
        flight_config: FlightConfig {
            num_episodes: 1,
            steps_per_episode: 1000,
            ..FlightConfig::default()
        },
        gusts: vec![
            WindGust {
                force: [0.05, 0.0, 0.0],  // Lateral gust
                start_time: 0.5,
                duration: 0.2,
            },
            WindGust {
                force: [0.0, 0.0, -0.03], // Downdraft
                start_time: 1.0,
                duration: 0.3,
            },
        ],
        eval_episodes: 3,
    };

    let result = run_wind_benchmark(&wind_config);

    println!(
        "{:<10} {:<22} {:<22}",
        "Episode", "Baseline Avg Error", "FEP Avg Error"
    );
    println!("{}", "-".repeat(54));

    for (i, (base, fep)) in result.baseline_metrics.iter().zip(result.fep_metrics.iter()).enumerate() {
        println!(
            "{:<10} {:<22.6} {:<22.6}",
            i, base, fep,
        );
    }

    println!("\n--- Recovery & Deviation ---");
    println!("Recovery steps per gust (FEP): {:?}", result.recovery_steps);
    println!("Max deviation (FEP):      {:.6}", result.max_deviation);
    println!("Max deviation (baseline):  {:.6}", result.baseline_max_deviation);

    let baseline_avg: f64 = result.baseline_metrics.iter().sum::<f64>() / result.baseline_metrics.len() as f64;
    let fep_avg: f64 = result.fep_metrics.iter().sum::<f64>() / result.fep_metrics.len() as f64;
    println!("\nOverall baseline avg error: {:.6}", baseline_avg);
    println!("Overall FEP avg error:     {:.6}", fep_avg);
    if fep_avg < baseline_avg {
        println!("FEP improvement: {:.2}%", (1.0 - fep_avg / baseline_avg) * 100.0);
    } else {
        println!("FEP degradation: {:.2}%", (fep_avg / baseline_avg - 1.0) * 100.0);
    }
    println!();
}

#[test]
#[ignore] // ~30s: run manually with `cargo test -p symthaea-flight -- --ignored --nocapture`
fn diagnostic_within_episode_learning_curve() {
    println!("\n{}", "=".repeat(80));
    println!("=== DIAGNOSTIC 3: Within-Episode Learning Curve (1 episode x 1000 steps) ===");
    println!("{}\n", "=".repeat(80));

    let config = FlightConfig {
        num_episodes: 1,
        steps_per_episode: 1000,
        collect_telemetry: true,
        ..FlightConfig::default()
    };

    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let trainer = FlightTrainer::new(config.clone());

    let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = FlightController::new(&genesis, &config);
    let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

    let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);

    assert_eq!(metrics.telemetry.len(), 1000, "Expected 1000 telemetry entries");

    let checkpoints = [0, 100, 200, 500, 999];
    println!(
        "{:<8} {:<16} {:<16} {:<16} {:<14} {:<14} {:<14}",
        "Step", "Pos Error", "Att Error", "Free Energy", "Tau Factor", "LR", "Thrust"
    );
    println!("{}", "-".repeat(98));

    for &step in &checkpoints {
        let t = &metrics.telemetry[step];
        println!(
            "{:<8} {:<16.6} {:<16.6} {:<16.6} {:<14.4} {:<14.6} {:<14.6}",
            t.step, t.position_error, t.attitude_error, t.free_energy,
            t.tau_factor, t.learning_rate, t.command.thrust,
        );
    }

    // Additional analysis: error trend in windows
    println!("\n--- Error Trend (100-step windows) ---");
    println!("{:<12} {:<18} {:<18} {:<18}", "Window", "Avg Pos Error", "Avg Att Error", "Avg Speed");
    println!("{}", "-".repeat(66));

    for window_start in (0..1000).step_by(100) {
        let window_end = (window_start + 100).min(1000);
        let window = &metrics.telemetry[window_start..window_end];
        let avg_pos: f64 = window.iter().map(|t| t.position_error).sum::<f64>() / window.len() as f64;
        let avg_att: f64 = window.iter().map(|t| t.attitude_error).sum::<f64>() / window.len() as f64;
        let avg_speed: f64 = window.iter().map(|t| t.speed).sum::<f64>() / window.len() as f64;
        println!(
            "{:<12} {:<18.6} {:<18.6} {:<18.6}",
            format!("{}-{}", window_start, window_end - 1),
            avg_pos, avg_att, avg_speed,
        );
    }

    // Altitude trace at checkpoints
    println!("\n--- Altitude Trace ---");
    println!("Target altitude: 0.1 m");
    for &step in &checkpoints {
        let t = &metrics.telemetry[step];
        println!("  Step {:>4}: altitude = {:.6} m (error = {:.6} m)", step, t.altitude, t.position_error);
    }

    // FEP action analysis: tau_factor distribution
    let tau_below_1 = metrics.telemetry.iter().filter(|t| t.tau_factor < 0.99).count();
    let tau_above_1 = metrics.telemetry.iter().filter(|t| t.tau_factor > 1.01).count();
    let tau_at_1 = metrics.telemetry.len() - tau_below_1 - tau_above_1;
    println!("\n--- FEP Action Distribution ---");
    println!("Tau < 1.0 (DropTau):   {} steps ({:.1}%)", tau_below_1, tau_below_1 as f64 / 10.0);
    println!("Tau = 1.0 (neutral):   {} steps ({:.1}%)", tau_at_1, tau_at_1 as f64 / 10.0);
    println!("Tau > 1.0 (RaiseTau):  {} steps ({:.1}%)", tau_above_1, tau_above_1 as f64 / 10.0);

    println!("\nEpisode summary:");
    println!("  avg_position_error: {:.6}", metrics.avg_position_error);
    println!("  final_position_error: {:.6}", metrics.final_position_error);
    println!("  avg_free_energy: {:.6}", metrics.avg_free_energy);
    println!("  hover_fraction: {:.4}", metrics.hover_fraction);
    println!("  exploration_count: {}", metrics.exploration_count);
    println!();
}
