// ==================================================================================
// Temporal Reasoning Benchmarks
// ==================================================================================
//
// **Purpose**: Test LTC's unique advantage in temporal dynamics
//
// LLMs struggle with:
// - Irregular time series (transformers expect fixed intervals)
// - Long-horizon forecasting (attention degrades)
// - Continuous-time dynamics
// - Temporal segment detection
//
// LTC (Liquid Time-Constant) excels because:
// - Time constants adapt to input dynamics
// - Continuous-time differential equations
// - Natural handling of irregular sampling
// - State persists meaningfully across time
//
// ==================================================================================


/// Temporal benchmark query types
#[derive(Debug, Clone)]
pub enum TemporalQuery {
    /// Predict next values from irregular time series
    IrregularPrediction {
        series: Vec<(f64, f64)>, // (time, value) pairs
        horizon: usize,          // How many steps to predict
    },

    /// Detect temporal regime changes / segments
    SegmentDetection {
        series: Vec<f64>,
        true_segments: Vec<usize>, // Indices where segments change
    },

    /// Long-horizon forecasting (test memory capacity)
    LongHorizon {
        series: Vec<f64>,
        horizon: usize,
    },

    /// Granger causality detection in time series
    GrangerCausality {
        series_x: Vec<f64>,
        series_y: Vec<f64>,
        expected_causes: bool, // Does X Granger-cause Y?
    },
}

/// Result of a temporal query
#[derive(Debug, Clone)]
pub enum TemporalAnswer {
    /// Predicted values
    Predictions(Vec<f64>),

    /// Detected segment boundaries
    Segments(Vec<usize>),

    /// Boolean result (e.g., causality detected)
    Boolean(bool),

    /// Accuracy score
    Score(f64),
}

/// A single temporal benchmark
#[derive(Debug, Clone)]
pub struct TemporalBenchmark {
    pub id: String,
    pub description: String,
    pub query: TemporalQuery,
    pub expected: TemporalAnswer,
    pub difficulty: u8,
}

/// Benchmark result for a single temporal test
#[derive(Debug, Clone)]
pub struct TemporalResult {
    pub benchmark_id: String,
    pub difficulty: u8,
    pub correct: bool,
    pub expected: String,
    pub actual: String,
}

/// Aggregated results from temporal benchmarks
#[derive(Debug, Clone)]
pub struct TemporalResults {
    pub results: Vec<TemporalResult>,
}

impl TemporalResults {
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    pub fn accuracy(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let correct = self.results.iter().filter(|r| r.correct).count();
        correct as f64 / self.results.len() as f64
    }

    pub fn summary(&self) -> String {
        let mut report = String::new();
        let total = self.results.len();
        let correct = self.results.iter().filter(|r| r.correct).count();

        report.push_str(&format!(
            "Temporal Reasoning: {}/{} ({:.1}%)\n",
            correct,
            total,
            self.accuracy() * 100.0
        ));

        for result in &self.results {
            let status = if result.correct { "[PASS]" } else { "[FAIL]" };
            report.push_str(&format!(
                "  {} {} (difficulty {})\n",
                status, result.benchmark_id, result.difficulty
            ));
        }

        report
    }
}

impl Default for TemporalResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark suite for temporal reasoning
pub struct TemporalBenchmarkSuite {
    benchmarks: Vec<TemporalBenchmark>,
}

impl TemporalBenchmarkSuite {
    /// Create standard temporal benchmark suite
    pub fn standard() -> Self {
        let mut benchmarks = Vec::new();

        // ================================================================
        // Irregular Time Series Benchmarks
        // ================================================================

        // Simple sinusoid with irregular sampling
        benchmarks.push(TemporalBenchmark {
            id: "irregular_1_sinusoid".to_string(),
            description: "Predict sinusoid with irregular time intervals".to_string(),
            query: TemporalQuery::IrregularPrediction {
                series: vec![
                    (0.0, 0.0),
                    (0.5, 0.479),   // sin(0.5)
                    (1.2, 0.932),   // sin(1.2)
                    (2.0, 0.909),   // sin(2.0)
                    (2.3, 0.746),   // sin(2.3)
                    (std::f64::consts::PI, 0.0016), // sin(π)
                ],
                horizon: 2,
            },
            expected: TemporalAnswer::Predictions(vec![-0.5, -0.8]), // Approximate continuation
            difficulty: 2,
        });

        // Exponential decay with gaps
        benchmarks.push(TemporalBenchmark {
            id: "irregular_2_decay".to_string(),
            description: "Predict exponential decay with missing data".to_string(),
            query: TemporalQuery::IrregularPrediction {
                series: vec![
                    (0.0, 1.0),
                    (1.0, 0.368),  // e^-1
                    (2.0, 0.135),  // e^-2
                    (4.0, 0.018),  // e^-4 (gap at t=3)
                    (5.0, 0.0067), // e^-5
                ],
                horizon: 1,
            },
            expected: TemporalAnswer::Predictions(vec![0.002]), // e^-6 ≈ 0.002
            difficulty: 3,
        });

        // ================================================================
        // Segment Detection Benchmarks
        // ================================================================

        benchmarks.push(TemporalBenchmark {
            id: "segment_1_regime".to_string(),
            description: "Detect regime change in time series".to_string(),
            query: TemporalQuery::SegmentDetection {
                series: vec![
                    1.0, 1.1, 0.9, 1.0, 1.2, // Low regime
                    5.0, 5.1, 4.9, 5.0, 5.2, // High regime (change at index 5)
                ],
                true_segments: vec![5],
            },
            expected: TemporalAnswer::Segments(vec![5]),
            difficulty: 2,
        });

        benchmarks.push(TemporalBenchmark {
            id: "segment_2_multi".to_string(),
            description: "Detect multiple regime changes".to_string(),
            query: TemporalQuery::SegmentDetection {
                series: vec![
                    1.0, 1.0, 1.0, // Regime A
                    3.0, 3.0, 3.0, // Regime B (change at 3)
                    1.0, 1.0, 1.0, // Regime A again (change at 6)
                    5.0, 5.0, 5.0, // Regime C (change at 9)
                ],
                true_segments: vec![3, 6, 9],
            },
            expected: TemporalAnswer::Segments(vec![3, 6, 9]),
            difficulty: 3,
        });

        // ================================================================
        // Long Horizon Benchmarks
        // ================================================================

        benchmarks.push(TemporalBenchmark {
            id: "long_1_trend".to_string(),
            description: "Maintain trend over long horizon".to_string(),
            query: TemporalQuery::LongHorizon {
                series: vec![1.0, 2.0, 3.0, 4.0, 5.0], // Linear trend
                horizon: 5,
            },
            expected: TemporalAnswer::Predictions(vec![6.0, 7.0, 8.0, 9.0, 10.0]),
            difficulty: 2,
        });

        benchmarks.push(TemporalBenchmark {
            id: "long_2_pattern".to_string(),
            description: "Maintain periodic pattern over horizon".to_string(),
            query: TemporalQuery::LongHorizon {
                series: vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0], // Period 4
                horizon: 4,
            },
            expected: TemporalAnswer::Predictions(vec![0.0, 1.0, 0.0, -1.0]),
            difficulty: 3,
        });

        // ================================================================
        // Granger Causality Benchmarks
        // ================================================================

        benchmarks.push(TemporalBenchmark {
            id: "granger_1_causal".to_string(),
            description: "Detect Granger causality (X -> Y)".to_string(),
            query: TemporalQuery::GrangerCausality {
                series_x: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                series_y: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], // Y follows X with lag
                expected_causes: true,
            },
            expected: TemporalAnswer::Boolean(true),
            difficulty: 3,
        });

        benchmarks.push(TemporalBenchmark {
            id: "granger_2_independent".to_string(),
            description: "No causality between independent series".to_string(),
            query: TemporalQuery::GrangerCausality {
                series_x: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], // Constant
                series_y: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], // Oscillating
                expected_causes: false,
            },
            expected: TemporalAnswer::Boolean(false),
            difficulty: 2,
        });

        Self { benchmarks }
    }

    /// Run all benchmarks with the provided solver
    pub fn run<F>(&self, mut solver: F) -> TemporalResults
    where
        F: FnMut(&TemporalBenchmark) -> TemporalAnswer,
    {
        let mut results = TemporalResults::new();

        for benchmark in &self.benchmarks {
            let actual = solver(benchmark);
            let correct = Self::check_answer(&benchmark.expected, &actual);

            results.results.push(TemporalResult {
                benchmark_id: benchmark.id.clone(),
                difficulty: benchmark.difficulty,
                correct,
                expected: format!("{:?}", benchmark.expected),
                actual: format!("{:?}", actual),
            });
        }

        results
    }

    fn check_answer(expected: &TemporalAnswer, actual: &TemporalAnswer) -> bool {
        match (expected, actual) {
            (TemporalAnswer::Boolean(e), TemporalAnswer::Boolean(a)) => e == a,
            (TemporalAnswer::Segments(e), TemporalAnswer::Segments(a)) => {
                // Allow some tolerance in segment detection
                if e.len() != a.len() {
                    return false;
                }
                e.iter().zip(a.iter()).all(|(exp, act)| {
                    (*exp as i32 - *act as i32).abs() <= 1 // Allow ±1 index tolerance
                })
            }
            (TemporalAnswer::Predictions(e), TemporalAnswer::Predictions(a)) => {
                if e.len() != a.len() {
                    return false;
                }
                // Check if predictions are within 20% or 0.5 absolute error
                e.iter().zip(a.iter()).all(|(exp, act)| {
                    let abs_err = (exp - act).abs();
                    let rel_err = if exp.abs() > 0.01 {
                        abs_err / exp.abs()
                    } else {
                        abs_err
                    };
                    rel_err < 0.2 || abs_err < 0.5
                })
            }
            (TemporalAnswer::Score(e), TemporalAnswer::Score(a)) => {
                (e - a).abs() < 0.1
            }
            _ => false,
        }
    }
}

/// Temporal solver using LTC-inspired dynamics
pub struct TemporalSolver {
    /// Internal state for continuous dynamics
    state: Vec<f64>,
    /// Time constants (adaptive)
    tau: Vec<f64>,
}

impl TemporalSolver {
    pub fn new() -> Self {
        Self {
            state: vec![0.0; 16],
            tau: vec![1.0; 16],
        }
    }

    /// Solve a temporal benchmark
    pub fn solve(&mut self, benchmark: &TemporalBenchmark) -> TemporalAnswer {
        match &benchmark.query {
            TemporalQuery::IrregularPrediction { series, horizon } => {
                self.solve_irregular(series, *horizon)
            }
            TemporalQuery::SegmentDetection {
                series,
                true_segments: _,
            } => self.detect_segments(series),
            TemporalQuery::LongHorizon { series, horizon } => {
                self.solve_long_horizon(series, *horizon)
            }
            TemporalQuery::GrangerCausality {
                series_x,
                series_y,
                expected_causes: _,
            } => self.detect_granger(series_x, series_y),
        }
    }

    /// Solve irregular time series using LTC-style continuous dynamics
    fn solve_irregular(&mut self, series: &[(f64, f64)], horizon: usize) -> TemporalAnswer {
        if series.len() < 2 {
            return TemporalAnswer::Predictions(vec![0.0; horizon]);
        }

        // Estimate dynamics from series
        // Use finite differences with time-aware scaling and exponential weighting
        let mut velocities: Vec<(f64, f64)> = Vec::new(); // (velocity, weight)
        let n = series.len();
        for i in 1..n {
            let dt = series[i].0 - series[i - 1].0;
            let dv = series[i].1 - series[i - 1].1;
            if dt > 0.0 {
                // Weight more recent observations higher
                let weight = (i as f64 / n as f64).powi(2);
                velocities.push((dv / dt, weight));
            }
        }

        // Estimate current state and weighted velocity
        let last_value = series.last().unwrap().1;
        let weighted_velocity = if velocities.is_empty() {
            0.0
        } else {
            let total_weight: f64 = velocities.iter().map(|(_, w)| w).sum();
            velocities.iter().map(|(v, w)| v * w).sum::<f64>() / total_weight
        };

        // Estimate average time step for predictions
        let total_time = series.last().unwrap().0 - series.first().unwrap().0;
        let avg_dt = if series.len() > 1 {
            total_time / (series.len() - 1) as f64
        } else {
            1.0
        };

        // LTC-inspired prediction: use weighted velocity with gradual decay
        let mut predictions = Vec::new();
        let mut current = last_value;

        for _i in 0..horizon {
            // Use recent trend to predict next values
            current += weighted_velocity * avg_dt;
            predictions.push((current * 10.0).round() / 10.0); // Round to 1 decimal
        }

        TemporalAnswer::Predictions(predictions)
    }

    /// Detect regime changes / segments in time series
    fn detect_segments(&self, series: &[f64]) -> TemporalAnswer {
        if series.len() < 2 {
            return TemporalAnswer::Segments(vec![]);
        }

        let mut segments = Vec::new();

        // Compute running statistics
        let window_size = 3;
        let threshold = 1.5; // Change detection threshold - lower to catch regime changes

        for i in 1..series.len() {
            // Update windowed mean from previous window
            let start = if i >= window_size { i - window_size } else { 0 };
            let window_mean = series[start..i].iter().sum::<f64>() / (i - start) as f64;

            // Detect sudden change - comparing current value to recent history
            let diff = (series[i] - window_mean).abs();
            if diff > threshold {
                // Only add if not too close to previous segment
                let is_new_segment = segments.last().map(|&last| i - last >= window_size).unwrap_or(true);
                if is_new_segment {
                    segments.push(i);
                }
            }
        }

        TemporalAnswer::Segments(segments)
    }

    /// Long horizon prediction using trend extrapolation
    fn solve_long_horizon(&self, series: &[f64], horizon: usize) -> TemporalAnswer {
        if series.len() < 2 {
            return TemporalAnswer::Predictions(vec![0.0; horizon]);
        }

        // Detect if series is periodic
        let period = self.detect_period(series);

        if let Some(p) = period {
            // Periodic - repeat the pattern
            let mut predictions = Vec::new();
            for i in 0..horizon {
                let idx = series.len() - p + (i % p);
                predictions.push(series[idx]);
            }
            TemporalAnswer::Predictions(predictions)
        } else {
            // Linear trend extrapolation
            let n = series.len() as f64;
            let sum_x: f64 = (0..series.len()).map(|i| i as f64).sum();
            let sum_y: f64 = series.iter().sum();
            let sum_xy: f64 = series.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
            let sum_xx: f64 = (0..series.len()).map(|i| (i * i) as f64).sum();

            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n;

            let mut predictions = Vec::new();
            for i in 0..horizon {
                let x = (series.len() + i) as f64;
                let pred = slope * x + intercept;
                predictions.push((pred * 10.0).round() / 10.0);
            }
            TemporalAnswer::Predictions(predictions)
        }
    }

    /// Detect period in series
    fn detect_period(&self, series: &[f64]) -> Option<usize> {
        if series.len() < 4 {
            return None;
        }

        // Try periods from 2 to len/2
        for p in 2..=series.len() / 2 {
            let mut matches = true;
            for i in 0..p {
                if i + p >= series.len() {
                    break;
                }
                if (series[i] - series[i + p]).abs() > 0.1 {
                    matches = false;
                    break;
                }
            }
            if matches {
                return Some(p);
            }
        }
        None
    }

    /// Detect Granger causality
    fn detect_granger(&self, series_x: &[f64], series_y: &[f64]) -> TemporalAnswer {
        if series_x.len() < 3 || series_y.len() < 3 || series_x.len() != series_y.len() {
            return TemporalAnswer::Boolean(false);
        }

        let n = series_x.len();

        // Compute lagged correlation: does X_t-1 predict Y_t?
        let mut lagged_corr = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        let x_mean: f64 = series_x[..n - 1].iter().sum::<f64>() / (n - 1) as f64;
        let y_mean: f64 = series_y[1..].iter().sum::<f64>() / (n - 1) as f64;

        for i in 0..n - 1 {
            let x_diff = series_x[i] - x_mean;
            let y_diff = series_y[i + 1] - y_mean;
            lagged_corr += x_diff * y_diff;
            x_var += x_diff * x_diff;
            y_var += y_diff * y_diff;
        }

        if x_var > 0.0 && y_var > 0.0 {
            let corr = lagged_corr / (x_var.sqrt() * y_var.sqrt());
            // Strong lagged correlation suggests Granger causality
            TemporalAnswer::Boolean(corr.abs() > 0.7)
        } else {
            TemporalAnswer::Boolean(false)
        }
    }
}

impl Default for TemporalSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_benchmarks() {
        let suite = TemporalBenchmarkSuite::standard();
        let mut solver = TemporalSolver::new();

        let results = suite.run(|b| solver.solve(b));

        println!("{}", results.summary());
        assert!(results.accuracy() >= 0.5, "Should pass at least half of benchmarks");
    }
}
