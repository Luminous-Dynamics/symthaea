// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Architecture Benchmark
//!
//! Performance benchmarks for the Mega-Unified Architecture v4.0 components.
//!
//! ## Components Benchmarked
//!
//! 1. **ThalamicRouter** - Cognitive depth routing performance
//! 2. **ClosedLearningLoop** - Q-learning strategy selection
//! 3. **ActiveInferenceBridge** - PAC modulation index computation
//! 4. **EpisodicMemoryBridge** - Memory encode/recall operations
//! 5. **GoalSystemBridge** - Goal attention bias computation
//! 6. **WorldModelBridge** - Hierarchical prediction updates
//! 7. **Full Cognitive Cycle** - End-to-end unified cycle timing

use crate::cognitive_loop::{
    ActiveInferenceBridge, ClosedLearningLoop, CognitiveGoal, CognitiveLoopConfig,
    CognitiveLoopService, CycleLearningResult, EpisodicMemoryBridge, GoalSystemBridge,
    ResponseStrategy, ThalamicRouter, WorldModelBridge,
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Configuration for unified architecture benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedArchitectureBenchmarkConfig {
    /// Number of iterations per benchmark
    pub iterations: usize,
    /// Number of warmup iterations
    pub warmup: usize,
    /// Number of memories for memory benchmark
    pub memory_count: usize,
    /// Embedding dimension for memory tests
    pub embedding_dim: usize,
    /// Number of goals for goal system test
    pub goal_count: usize,
    /// Number of cycles for full cycle benchmark
    pub cycle_count: usize,
}

impl Default for UnifiedArchitectureBenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            warmup: 100,
            memory_count: 100,
            embedding_dim: 64,
            goal_count: 5,
            cycle_count: 100,
        }
    }
}

/// Results from a single benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_time_us: u64,
    pub mean_time_us: f64,
    pub median_time_us: f64,
    pub min_time_us: u64,
    pub max_time_us: u64,
    pub std_dev_us: f64,
    pub throughput_ops_per_sec: f64,
}

impl BenchmarkResult {
    pub fn from_timings(name: impl Into<String>, timings: &[Duration]) -> Self {
        let mut times_us: Vec<u64> = timings.iter().map(|d| d.as_micros() as u64).collect();
        times_us.sort();

        let iterations = times_us.len();
        let total_time_us: u64 = times_us.iter().sum();
        let mean_time_us = total_time_us as f64 / iterations as f64;
        let median_time_us = if iterations.is_multiple_of(2) {
            (times_us[iterations / 2 - 1] + times_us[iterations / 2]) as f64 / 2.0
        } else {
            times_us[iterations / 2] as f64
        };
        let min_time_us = *times_us.first().unwrap_or(&0);
        let max_time_us = *times_us.last().unwrap_or(&0);

        let variance: f64 = times_us
            .iter()
            .map(|&t| (t as f64 - mean_time_us).powi(2))
            .sum::<f64>()
            / iterations as f64;
        let std_dev_us = variance.sqrt();

        let throughput_ops_per_sec = if mean_time_us > 0.0 {
            1_000_000.0 / mean_time_us
        } else {
            0.0
        };

        Self {
            name: name.into(),
            iterations,
            total_time_us,
            mean_time_us,
            median_time_us,
            min_time_us,
            max_time_us,
            std_dev_us,
            throughput_ops_per_sec,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "{}: mean={:.2}µs median={:.2}µs min={}µs max={}µs std={:.2}µs ({:.0} ops/sec)",
            self.name,
            self.mean_time_us,
            self.median_time_us,
            self.min_time_us,
            self.max_time_us,
            self.std_dev_us,
            self.throughput_ops_per_sec,
        )
    }
}

/// Full benchmark suite results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedArchitectureBenchmarkResults {
    pub config: UnifiedArchitectureBenchmarkConfig,
    pub thalamic_router: BenchmarkResult,
    pub closed_learning_loop_select: BenchmarkResult,
    pub closed_learning_loop_update: BenchmarkResult,
    pub active_inference_observe: BenchmarkResult,
    pub active_inference_modulation: BenchmarkResult,
    pub episodic_memory_encode: BenchmarkResult,
    pub episodic_memory_recall: BenchmarkResult,
    pub goal_system_attention: BenchmarkResult,
    pub world_model_update: BenchmarkResult,
    pub full_cycle: BenchmarkResult,
    pub total_benchmark_time_secs: f64,
}

impl UnifiedArchitectureBenchmarkResults {
    pub fn summary(&self) -> String {
        let lines = vec![
            "═══════════════════════════════════════════════════════════════".to_string(),
            "UNIFIED ARCHITECTURE BENCHMARK RESULTS".to_string(),
            "═══════════════════════════════════════════════════════════════".to_string(),
            format!(
                "Iterations: {}, Warmup: {}",
                self.config.iterations, self.config.warmup
            ),
            "".to_string(),
            "Component Performance:".to_string(),
            format!("  {}", self.thalamic_router.summary()),
            format!("  {}", self.closed_learning_loop_select.summary()),
            format!("  {}", self.closed_learning_loop_update.summary()),
            format!("  {}", self.active_inference_observe.summary()),
            format!("  {}", self.active_inference_modulation.summary()),
            format!("  {}", self.episodic_memory_encode.summary()),
            format!("  {}", self.episodic_memory_recall.summary()),
            format!("  {}", self.goal_system_attention.summary()),
            format!("  {}", self.world_model_update.summary()),
            "".to_string(),
            "Full Cycle:".to_string(),
            format!("  {}", self.full_cycle.summary()),
            "".to_string(),
            format!(
                "Total benchmark time: {:.2}s",
                self.total_benchmark_time_secs
            ),
            "═══════════════════════════════════════════════════════════════".to_string(),
        ];
        lines.join("\n")
    }
}

/// Run the unified architecture benchmark suite
pub fn run_unified_architecture_benchmark(
    config: UnifiedArchitectureBenchmarkConfig,
) -> UnifiedArchitectureBenchmarkResults {
    let start = Instant::now();

    // Benchmark ThalamicRouter
    let thalamic_router = benchmark_thalamic_router(&config);

    // Benchmark ClosedLearningLoop
    let (closed_learning_loop_select, closed_learning_loop_update) =
        benchmark_closed_learning_loop(&config);

    // Benchmark ActiveInferenceBridge
    let (active_inference_observe, active_inference_modulation) =
        benchmark_active_inference(&config);

    // Benchmark EpisodicMemoryBridge
    let (episodic_memory_encode, episodic_memory_recall) = benchmark_episodic_memory(&config);

    // Benchmark GoalSystemBridge
    let goal_system_attention = benchmark_goal_system(&config);

    // Benchmark WorldModelBridge
    let world_model_update = benchmark_world_model(&config);

    // Benchmark full cognitive cycle
    let full_cycle = benchmark_full_cycle(&config);

    let total_benchmark_time_secs = start.elapsed().as_secs_f64();

    UnifiedArchitectureBenchmarkResults {
        config,
        thalamic_router,
        closed_learning_loop_select,
        closed_learning_loop_update,
        active_inference_observe,
        active_inference_modulation,
        episodic_memory_encode,
        episodic_memory_recall,
        goal_system_attention,
        world_model_update,
        full_cycle,
        total_benchmark_time_secs,
    }
}

fn benchmark_thalamic_router(config: &UnifiedArchitectureBenchmarkConfig) -> BenchmarkResult {
    let mut router = ThalamicRouter::default();
    let mut timings = Vec::with_capacity(config.iterations);

    // Warmup
    for _ in 0..config.warmup {
        router.route(0.5, 0.5, 0.5, 0.5);
    }

    // Benchmark
    for i in 0..config.iterations {
        let novelty = (i as f32 / config.iterations as f32).sin().abs();
        let urgency = (i as f32 / config.iterations as f32 * 2.0).cos().abs();
        let complexity = (i % 10) as f32 / 10.0;
        let emotion = (i % 7) as f32 / 7.0;

        let start = Instant::now();
        let _ = router.route(novelty, urgency, complexity, emotion);
        timings.push(start.elapsed());
    }

    BenchmarkResult::from_timings("ThalamicRouter.route()", &timings)
}

fn benchmark_closed_learning_loop(
    config: &UnifiedArchitectureBenchmarkConfig,
) -> (BenchmarkResult, BenchmarkResult) {
    let mut loop_ = ClosedLearningLoop::default();
    let mut select_timings = Vec::with_capacity(config.iterations);
    let mut update_timings = Vec::with_capacity(config.iterations);

    // Warmup
    for _ in 0..config.warmup {
        loop_.select_strategy(0.5, None);
        loop_.update(CycleLearningResult {
            strategy_used: ResponseStrategy::Supportive,
            reward: 0.5,
            successful: true,
            prediction_error: 0.2,
            coherence: 0.7,
        });
    }

    // Benchmark select_strategy
    for i in 0..config.iterations {
        let phi = (i as f64 / config.iterations as f64).sin().abs();

        let start = Instant::now();
        let _ = loop_.select_strategy(phi, Some(0.5));
        select_timings.push(start.elapsed());
    }

    // Benchmark update
    let strategies = [
        ResponseStrategy::Detailed,
        ResponseStrategy::Concise,
        ResponseStrategy::Clarifying,
        ResponseStrategy::Supportive,
        ResponseStrategy::Exploratory,
    ];

    for i in 0..config.iterations {
        let result = CycleLearningResult {
            strategy_used: strategies[i % 5],
            reward: (i as f32 / config.iterations as f32) - 0.5,
            successful: i % 2 == 0,
            prediction_error: (i as f32 / config.iterations as f32).sin().abs(),
            coherence: 0.7,
        };

        let start = Instant::now();
        loop_.update(result);
        update_timings.push(start.elapsed());
    }

    (
        BenchmarkResult::from_timings("ClosedLearningLoop.select_strategy()", &select_timings),
        BenchmarkResult::from_timings("ClosedLearningLoop.update()", &update_timings),
    )
}

fn benchmark_active_inference(
    config: &UnifiedArchitectureBenchmarkConfig,
) -> (BenchmarkResult, BenchmarkResult) {
    let mut bridge = ActiveInferenceBridge::default();
    let mut observe_timings = Vec::with_capacity(config.iterations);
    let mut modulation_timings = Vec::with_capacity(config.iterations);

    // Warmup
    for _ in 0..config.warmup {
        bridge.observe_resolution(0.8, true);
    }

    // Benchmark observe_resolution
    for i in 0..config.iterations {
        let confidence = (i as f64 / config.iterations as f64).sin().abs();
        let success = i % 3 != 0;

        let start = Instant::now();
        bridge.observe_resolution(confidence, success);
        observe_timings.push(start.elapsed());
    }

    // Benchmark modulation_index
    for _ in 0..config.iterations {
        let start = Instant::now();
        let _ = bridge.modulation_index();
        modulation_timings.push(start.elapsed());
    }

    (
        BenchmarkResult::from_timings(
            "ActiveInferenceBridge.observe_resolution()",
            &observe_timings,
        ),
        BenchmarkResult::from_timings(
            "ActiveInferenceBridge.modulation_index()",
            &modulation_timings,
        ),
    )
}

fn benchmark_episodic_memory(
    config: &UnifiedArchitectureBenchmarkConfig,
) -> (BenchmarkResult, BenchmarkResult) {
    let mut bridge = EpisodicMemoryBridge::default();
    let mut encode_timings = Vec::with_capacity(config.iterations);
    let mut recall_timings = Vec::with_capacity(config.iterations);

    // Pre-populate with some memories for recall benchmark
    for i in 0..config.memory_count.min(50) {
        let embedding: Vec<f32> = (0..config.embedding_dim)
            .map(|j| ((i * j) as f32).sin())
            .collect();
        bridge.encode(format!("memory {i}"), embedding, 0.5, 0.6, i);
    }

    // Benchmark encode
    for i in 0..config.iterations {
        let embedding: Vec<f32> = (0..config.embedding_dim)
            .map(|j| ((i * j) as f32).cos())
            .collect();

        let start = Instant::now();
        let _ = bridge.encode(format!("test memory {i}"), embedding, 0.5, 0.6, i);
        encode_timings.push(start.elapsed());
    }

    // Benchmark recall
    for i in 0..config.iterations {
        let query: Vec<f32> = (0..config.embedding_dim)
            .map(|j| ((i * j) as f32).sin())
            .collect();

        let start = Instant::now();
        let _ = bridge.recall(&query, 5, 0.3);
        recall_timings.push(start.elapsed());
    }

    (
        BenchmarkResult::from_timings("EpisodicMemoryBridge.encode()", &encode_timings),
        BenchmarkResult::from_timings("EpisodicMemoryBridge.recall()", &recall_timings),
    )
}

fn benchmark_goal_system(config: &UnifiedArchitectureBenchmarkConfig) -> BenchmarkResult {
    let mut bridge = GoalSystemBridge::new();
    let mut timings = Vec::with_capacity(config.iterations);

    // Add some goals
    for i in 0..config.goal_count {
        bridge.add_goal(CognitiveGoal::new(
            format!("goal_{i}"),
            format!("Test goal {i}"),
            (i as f32 + 1.0) / (config.goal_count as f32 + 1.0),
        ));
    }

    // Warmup
    for _ in 0..config.warmup {
        let _ = bridge.attention_bias();
    }

    // Benchmark
    for _ in 0..config.iterations {
        let start = Instant::now();
        let _ = bridge.attention_bias();
        timings.push(start.elapsed());
    }

    BenchmarkResult::from_timings("GoalSystemBridge.attention_bias()", &timings)
}

fn benchmark_world_model(config: &UnifiedArchitectureBenchmarkConfig) -> BenchmarkResult {
    let mut bridge = WorldModelBridge::default();
    let mut timings = Vec::with_capacity(config.iterations);

    // Warmup
    for _ in 0..config.warmup {
        let input: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        bridge.update_sensory(&input);
    }

    // Benchmark
    for i in 0..config.iterations {
        let input: Vec<f32> = (0..64).map(|j| ((i * j) as f32).sin()).collect();

        let start = Instant::now();
        bridge.update_sensory(&input);
        timings.push(start.elapsed());
    }

    BenchmarkResult::from_timings("WorldModelBridge.update_sensory()", &timings)
}

fn benchmark_full_cycle(config: &UnifiedArchitectureBenchmarkConfig) -> BenchmarkResult {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default())
        .expect("Failed to create CognitiveLoopService");
    let mut timings = Vec::with_capacity(config.cycle_count);

    let test_inputs = [
        "Hello, how are you today?",
        "I'm thinking about consciousness",
        "What is the meaning of life?",
        "Can you help me understand HDC?",
        "This is a test of the cognitive loop",
    ];

    // Warmup
    for i in 0..10 {
        service.cycle(test_inputs[i % test_inputs.len()]);
    }

    // Benchmark
    for i in 0..config.cycle_count {
        let input = test_inputs[i % test_inputs.len()];

        let start = Instant::now();
        let _ = service.cycle(input);
        timings.push(start.elapsed());
    }

    BenchmarkResult::from_timings("CognitiveLoopService.cycle()", &timings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_single_task() {
        let config = UnifiedArchitectureBenchmarkConfig {
            iterations: 10,
            warmup: 2,
            memory_count: 10,
            embedding_dim: 16,
            goal_count: 3,
            cycle_count: 10,
        };

        let results = run_unified_architecture_benchmark(config);

        // Verify all benchmarks ran
        assert!(results.thalamic_router.iterations > 0);
        assert!(results.closed_learning_loop_select.iterations > 0);
        assert!(results.active_inference_observe.iterations > 0);
        assert!(results.episodic_memory_encode.iterations > 0);
        assert!(results.goal_system_attention.iterations > 0);
        assert!(results.world_model_update.iterations > 0);
        assert!(results.full_cycle.iterations > 0);

        // Print summary
        println!("{}", results.summary());
    }

    #[test]
    fn test_benchmark_result_from_timings() {
        let timings = vec![
            Duration::from_micros(100),
            Duration::from_micros(200),
            Duration::from_micros(150),
            Duration::from_micros(120),
            Duration::from_micros(180),
        ];

        let result = BenchmarkResult::from_timings("test", &timings);

        assert_eq!(result.iterations, 5);
        assert_eq!(result.min_time_us, 100);
        assert_eq!(result.max_time_us, 200);
        assert!((result.mean_time_us - 150.0).abs() < 0.1);
    }
}
