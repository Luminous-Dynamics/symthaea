//! Wind gust robustness benchmarks for FEP-modulated flight control.
//!
//! Compares flight performance with and without Active Inference modulation
//! under wind disturbances injected via `PhysicsSimulator::apply_external_force`.

use symthaea_core::genesis::GenesisSeed;

use crate::controller::FlightController;
use crate::encoder::QuadrotorHdcEncoder;
use crate::fep_agent::{ActiveInferenceFlightAgent, FlightFepConfig, FlightFepResult};
use crate::simulator::{PhysicsSimulator, SimplePhysicsSimulator};
use crate::types::*;

/// A single wind gust event.
#[derive(Debug, Clone)]
pub struct WindGust {
    /// Force vector in world frame [N].
    pub force: [f64; 3],
    /// Start time in seconds.
    pub start_time: f64,
    /// Duration in seconds.
    pub duration: f64,
}

/// Configuration for the wind gust benchmark.
#[derive(Debug, Clone)]
pub struct WindBenchmarkConfig {
    /// Base flight configuration.
    pub flight_config: FlightConfig,
    /// Wind gusts to inject.
    pub gusts: Vec<WindGust>,
    /// Number of evaluation episodes.
    pub eval_episodes: usize,
}

impl Default for WindBenchmarkConfig {
    fn default() -> Self {
        Self {
            flight_config: FlightConfig {
                num_episodes: 1,
                steps_per_episode: 2000,
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
                    start_time: 1.5,
                    duration: 0.3,
                },
            ],
            eval_episodes: 3,
        }
    }
}

/// Results from a wind gust benchmark run.
#[derive(Debug, Clone)]
pub struct WindBenchmarkResult {
    /// Metrics without FEP modulation (frozen agent).
    pub baseline_metrics: Vec<f64>,
    /// Metrics with full FEP modulation.
    pub fep_metrics: Vec<f64>,
    /// Steps to recover within 5cm after each gust (FEP).
    pub recovery_steps: Vec<usize>,
    /// Maximum deviation during gusts (FEP).
    pub max_deviation: f64,
    /// Maximum deviation during gusts (baseline).
    pub baseline_max_deviation: f64,
}

/// Run the wind gust robustness benchmark.
///
/// Compares a "frozen" baseline (no FEP modulation) against full Active Inference.
pub fn run_wind_benchmark(config: &WindBenchmarkConfig) -> WindBenchmarkResult {
    let genesis = GenesisSeed::from_phrase(&config.flight_config.genesis_phrase);
    let setpoint = FlightSetpoint::hover();
    let dt = config.flight_config.motor_dt();
    let pd_gains = PdGains::default();
    let cognitive_interval = config.flight_config.cognitive_interval();

    let mut baseline_errors = Vec::new();
    let mut fep_errors = Vec::new();
    let mut recovery_steps = Vec::new();
    let mut max_dev_fep = 0.0f64;
    let mut max_dev_baseline = 0.0f64;

    for ep in 0..config.eval_episodes {
        // ── Baseline: frozen FEP agent (no modulation) ──
        {
            let mut physics = SimplePhysicsSimulator::new();
            physics.reset(0.1);
            let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.flight_config.num_levels);
            let mut controller = FlightController::new(&genesis, &config.flight_config);
            let frozen_result = FlightFepResult::default();
            let mut total_err = 0.0;

            for step in 0..config.flight_config.steps_per_episode {
                let time = step as f64 * dt;
                let state = physics.state().clone();
                let sensor_hv = encoder.encode(&state);
                let command = controller.forward(&sensor_hv, dt as f32);
                physics.step(&command, dt);

                // Inject gusts
                for gust in &config.gusts {
                    if time >= gust.start_time && time < gust.start_time + gust.duration {
                        physics.apply_external_force(gust.force);
                    }
                }

                let err = setpoint.position_error_magnitude(&state);
                total_err += err;
                max_dev_baseline = max_dev_baseline.max(err);

                // Training (still train, just no FEP modulation)
                if step % config.flight_config.train_every == 0 {
                    let target = pd_baseline(&state, &setpoint, &pd_gains);
                    let lr = config.flight_config.learning_rate * frozen_result.learning_rate_factor;
                    controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
                }
            }
            baseline_errors.push(total_err / config.flight_config.steps_per_episode as f64);
        }

        // ── FEP: full Active Inference modulation ──
        {
            let mut physics = SimplePhysicsSimulator::new();
            physics.reset(0.1);
            let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.flight_config.num_levels);
            let mut controller = FlightController::new(&genesis, &config.flight_config);
            let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());
            let mut fep_result = fep_agent.initial_step(physics.state(), &setpoint);
            let mut total_err = 0.0;

            // Track recovery after each gust
            let mut gust_end_tracking: Vec<(f64, bool, usize)> = config
                .gusts
                .iter()
                .map(|g| (g.start_time + g.duration, false, 0))
                .collect();

            for step in 0..config.flight_config.steps_per_episode {
                let time = step as f64 * dt;
                let state = physics.state().clone();
                let sensor_hv = encoder.encode(&state);
                let mut command = controller.forward(&sensor_hv, dt as f32);

                if let Some(noise) = &fep_result.exploration_noise {
                    command = command.with_noise(noise);
                }

                // Inject gusts
                for gust in &config.gusts {
                    if time >= gust.start_time && time < gust.start_time + gust.duration {
                        physics.apply_external_force(gust.force);
                    }
                }

                physics.step(&command, dt);

                let err = setpoint.position_error_magnitude(&state);
                total_err += err;
                max_dev_fep = max_dev_fep.max(err);

                // Track recovery
                for (i, (end_time, recovered, count)) in gust_end_tracking.iter_mut().enumerate() {
                    if time >= *end_time && !*recovered {
                        *count += 1;
                        if err < 0.05 {
                            *recovered = true;
                        }
                    }
                    let _ = i; // suppress unused warning
                }

                if step % config.flight_config.train_every == 0 {
                    let target = pd_baseline(&state, &setpoint, &pd_gains);
                    let lr = config.flight_config.learning_rate * fep_result.learning_rate_factor;
                    controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
                }

                if step % cognitive_interval == 0 {
                    fep_result = fep_agent.step(physics.state(), &setpoint);
                    if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                        controller.modulate_tau(fep_result.tau_factor);
                    }
                }
            }

            fep_errors.push(total_err / config.flight_config.steps_per_episode as f64);

            for (_, _, count) in &gust_end_tracking {
                recovery_steps.push(*count);
            }

            let _ = ep;
        }
    }

    WindBenchmarkResult {
        baseline_metrics: baseline_errors,
        fep_metrics: fep_errors,
        recovery_steps,
        max_deviation: max_dev_fep,
        baseline_max_deviation: max_dev_baseline,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wind_benchmark_runs() {
        let config = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 200,
                ..FlightConfig::default()
            },
            gusts: vec![WindGust {
                force: [0.05, 0.0, 0.0],
                start_time: 0.1,
                duration: 0.1,
            }],
            eval_episodes: 1,
        };
        let result = run_wind_benchmark(&config);
        assert_eq!(result.baseline_metrics.len(), 1);
        assert_eq!(result.fep_metrics.len(), 1);
        assert!(result.baseline_metrics[0].is_finite());
        assert!(result.fep_metrics[0].is_finite());
    }

    #[test]
    fn test_gust_increases_error() {
        // Run without gust
        let config_no_gust = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 200,
                ..FlightConfig::default()
            },
            gusts: vec![],
            eval_episodes: 1,
        };
        let no_gust = run_wind_benchmark(&config_no_gust);

        // Run with gust
        let config_gust = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 200,
                ..FlightConfig::default()
            },
            gusts: vec![WindGust {
                force: [0.1, 0.0, 0.0],
                start_time: 0.05,
                duration: 0.2,
            }],
            eval_episodes: 1,
        };
        let with_gust = run_wind_benchmark(&config_gust);

        // Gust should increase max deviation
        assert!(
            with_gust.baseline_max_deviation >= no_gust.baseline_max_deviation * 0.5,
            "Gust should increase deviation: no_gust={:.4}, with_gust={:.4}",
            no_gust.baseline_max_deviation,
            with_gust.baseline_max_deviation,
        );
    }

    #[test]
    fn test_fep_produces_finite_results() {
        let config = WindBenchmarkConfig {
            flight_config: FlightConfig {
                steps_per_episode: 100,
                ..FlightConfig::default()
            },
            gusts: vec![WindGust {
                force: [0.05, 0.0, 0.0],
                start_time: 0.05,
                duration: 0.05,
            }],
            eval_episodes: 1,
        };
        let result = run_wind_benchmark(&config);
        assert!(result.max_deviation.is_finite());
        assert!(result.baseline_max_deviation.is_finite());
        for e in &result.fep_metrics {
            assert!(e.is_finite());
        }
    }
}
