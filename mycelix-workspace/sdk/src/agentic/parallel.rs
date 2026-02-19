//! # Parallel Simulation Engine
//!
//! High-performance parallel execution for large-scale agent simulations.
//!
//! ## Features
//!
//! - **Parallel Tick Processing**: Process agent actions concurrently using rayon
//! - **Batch Detection**: Batch processing for gaming/sybil detection
//! - **Memory-Efficient**: Pre-allocated buffers to reduce allocation overhead
//! - **Lock-Free Updates**: Atomic operations for trust score aggregation

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use super::{
    adversarial::{
        GamingDetectionConfig, GamingDetector, QuarantineManager, QuarantineReason, SybilDetector,
    },
    ActionOutcome, AgentClass, AgentConstraints, AgentId, AgentStatus, BehaviorLogEntry,
    EpistemicStats, InstrumentalActor, UncertaintyCalibration,
};
use crate::matl::KVector;

/// Configuration for parallel simulation
#[derive(Clone, Debug)]
pub struct ParallelSimConfig {
    /// Number of simulation ticks
    pub ticks: u64,
    /// Duration per tick in seconds
    pub tick_duration_secs: u64,
    /// Enable gaming detection
    pub detect_gaming: bool,
    /// Enable sybil detection
    pub detect_sybils: bool,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Minimum agents before parallelization kicks in
    pub parallel_threshold: usize,
}

impl Default for ParallelSimConfig {
    fn default() -> Self {
        Self {
            ticks: 100,
            tick_duration_secs: 60,
            detect_gaming: true,
            detect_sybils: true,
            chunk_size: 100,
            parallel_threshold: 500,
        }
    }
}

/// Agent behavior parameters for simulation
#[derive(Clone, Debug)]
pub struct SimAgentBehavior {
    /// Activity rate (actions per hour)
    pub activity_rate: f64,
    /// Base success rate (0.0-1.0)
    pub success_rate: f64,
    /// Success rate variance
    pub success_variance: f64,
    /// Timing regularity (higher = more predictable)
    pub timing_regularity: f64,
}

impl Default for SimAgentBehavior {
    fn default() -> Self {
        Self {
            activity_rate: 10.0,
            success_rate: 0.85,
            success_variance: 0.05,
            timing_regularity: 0.3,
        }
    }
}

/// Simulated agent with behavior configuration
pub struct SimAgent {
    /// The underlying agent
    pub agent: InstrumentalActor,
    /// Behavior configuration
    pub behavior: SimAgentBehavior,
    /// Sponsor ID
    pub sponsor: String,
}

/// Result of a single simulation tick
#[derive(Clone, Debug)]
pub struct ParallelTickResult {
    /// Tick number
    pub tick: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Total agents
    pub total_agents: usize,
    /// Active agents
    pub active_agents: usize,
    /// Quarantined agents
    pub quarantined_agents: usize,
    /// Average trust score
    pub avg_trust: f64,
    /// Gaming incidents detected
    pub gaming_incidents: usize,
    /// Sybil evidence count
    pub sybil_evidence: usize,
}

/// Thread-safe aggregators for parallel collection
pub struct TickAggregators {
    total_trust: AtomicU64,
    active_count: AtomicUsize,
    gaming_incidents: AtomicUsize,
}

impl TickAggregators {
    /// Create new aggregators
    pub fn new() -> Self {
        Self {
            total_trust: AtomicU64::new(0),
            active_count: AtomicUsize::new(0),
            gaming_incidents: AtomicUsize::new(0),
        }
    }

    /// Add trust score (f64 encoded as u64 bits)
    pub fn add_trust(&self, trust: f64) {
        // Use fetch_add with bit representation for atomic f64 accumulation
        loop {
            let current = self.total_trust.load(Ordering::Relaxed);
            let current_f = f64::from_bits(current);
            let new_f = current_f + trust;
            let new = new_f.to_bits();
            if self
                .total_trust
                .compare_exchange_weak(current, new, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Increment active count
    pub fn inc_active(&self) {
        self.active_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment gaming incidents
    pub fn inc_gaming(&self) {
        self.gaming_incidents.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total trust
    pub fn total_trust(&self) -> f64 {
        f64::from_bits(self.total_trust.load(Ordering::Acquire))
    }

    /// Get active count
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Acquire)
    }

    /// Get gaming incidents
    pub fn gaming_incidents(&self) -> usize {
        self.gaming_incidents.load(Ordering::Acquire)
    }
}

impl Default for TickAggregators {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-allocated random buffer for simulation
pub struct RandomBuffer {
    values: Vec<f64>,
    index: usize,
}

impl RandomBuffer {
    /// Create with capacity (uses simple LCG for speed)
    pub fn new(capacity: usize, seed: u64) -> Self {
        let mut values = Vec::with_capacity(capacity);
        let mut state = seed;

        for _ in 0..capacity {
            // Linear Congruential Generator (fast, good enough for simulation)
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            values.push((state as f64) / (u64::MAX as f64));
        }

        Self { values, index: 0 }
    }

    /// Get next random value (0.0 - 1.0)
    pub fn next(&mut self) -> f64 {
        let val = self.values[self.index % self.values.len()];
        self.index += 1;
        val
    }

    /// Reset index
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Get batch of random values
    pub fn next_batch(&mut self, count: usize) -> &[f64] {
        let start = self.index % self.values.len();
        let end = (start + count).min(self.values.len());
        self.index += count;
        &self.values[start..end]
    }
}

/// High-performance parallel simulation engine
pub struct ParallelSimEngine {
    /// Configuration
    config: ParallelSimConfig,
    /// Agents indexed by ID
    agents: HashMap<String, SimAgent>,
    /// Current tick
    current_tick: u64,
    /// Random buffer
    random_buffer: RandomBuffer,
    /// Gaming detector
    gaming_detector: GamingDetector,
    /// Sybil detector
    sybil_detector: SybilDetector,
    /// Quarantine manager
    quarantine: QuarantineManager,
    /// Tick results
    tick_results: Vec<ParallelTickResult>,
}

impl ParallelSimEngine {
    /// Create new parallel simulation engine
    pub fn new(config: ParallelSimConfig) -> Self {
        // Pre-allocate enough random values for expected simulation
        let ticks = config.ticks as usize;
        let random_capacity = ticks * 1000 * 10;

        Self {
            config,
            agents: HashMap::with_capacity(10000),
            current_tick: 0,
            random_buffer: RandomBuffer::new(random_capacity.min(10_000_000), 42),
            gaming_detector: GamingDetector::new(GamingDetectionConfig::default()),
            sybil_detector: SybilDetector::new(0.9),
            quarantine: QuarantineManager::new(),
            tick_results: Vec::with_capacity(ticks),
        }
    }

    /// Add agent to simulation
    pub fn add_agent(&mut self, id: &str, behavior: SimAgentBehavior, sponsor: &str) {
        let agent = InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: sponsor.to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: Vec::with_capacity(100),
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: EpistemicStats::default(),
            output_history: Vec::new(),
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: Vec::new(),
        };

        self.agents.insert(
            id.to_string(),
            SimAgent {
                agent,
                behavior,
                sponsor: sponsor.to_string(),
            },
        );
    }

    /// Add population of similar agents
    pub fn add_population(
        &mut self,
        prefix: &str,
        behavior: SimAgentBehavior,
        count: usize,
        sponsor: &str,
    ) {
        for i in 0..count {
            self.add_agent(&format!("{}-{}", prefix, i), behavior.clone(), sponsor);
        }
    }

    /// Run simulation tick (parallel if above threshold)
    pub fn tick(&mut self) -> ParallelTickResult {
        let timestamp = self.current_tick * self.config.tick_duration_secs;

        if self.agents.len() >= self.config.parallel_threshold {
            self.parallel_tick(timestamp)
        } else {
            self.sequential_tick(timestamp)
        }
    }

    /// Sequential tick for small simulations
    fn sequential_tick(&mut self, timestamp: u64) -> ParallelTickResult {
        let mut gaming_incidents = 0;
        let mut total_trust = 0.0;
        let mut active_count = 0;

        // Simulate actions
        for sim_agent in self.agents.values_mut() {
            if sim_agent.agent.status != AgentStatus::Active {
                continue;
            }

            if self
                .quarantine
                .is_quarantined(sim_agent.agent.agent_id.as_str())
            {
                continue;
            }

            let activity_variance = self.random_buffer.next();
            let actions = (sim_agent.behavior.activity_rate + activity_variance * 0.5) as usize;

            for _ in 0..actions.min(5) {
                let success_roll = self.random_buffer.next();
                let threshold_variance = self.random_buffer.next();

                let threshold = sim_agent.behavior.success_rate
                    + (threshold_variance - 0.5) * sim_agent.behavior.success_variance * 2.0;

                let outcome = if success_roll < threshold {
                    ActionOutcome::Success
                } else {
                    ActionOutcome::Error
                };

                sim_agent.agent.behavior_log.push(BehaviorLogEntry {
                    timestamp,
                    action_type: "action".to_string(),
                    kredit_consumed: 10,
                    counterparties: vec![],
                    outcome,
                });
            }

            total_trust += sim_agent.agent.k_vector.trust_score() as f64;
            active_count += 1;
        }

        // Gaming detection
        if self.config.detect_gaming {
            let agent_ids: Vec<_> = self.agents.keys().cloned().collect();
            for id in agent_ids {
                if let Some(sim_agent) = self.agents.get(&id) {
                    if sim_agent.agent.behavior_log.len() >= 20 {
                        let result = self.gaming_detector.analyze(&sim_agent.agent, timestamp);
                        if result.suspicion_score > 0.6 {
                            gaming_incidents += 1;
                            self.quarantine.quarantine(
                                &id,
                                QuarantineReason::GamingDetected,
                                vec![format!("Score: {:.2}", result.suspicion_score)],
                                timestamp,
                            );
                        }
                    }
                }
            }
        }

        let result = ParallelTickResult {
            tick: self.current_tick,
            timestamp,
            total_agents: self.agents.len(),
            active_agents: active_count,
            quarantined_agents: self.quarantine.entries.len(),
            avg_trust: if active_count > 0 {
                total_trust / active_count as f64
            } else {
                0.0
            },
            gaming_incidents,
            sybil_evidence: 0,
        };

        self.tick_results.push(result.clone());
        self.current_tick += 1;
        result
    }

    /// Parallel tick for large simulations
    fn parallel_tick(&mut self, timestamp: u64) -> ParallelTickResult {
        let aggregators = Arc::new(TickAggregators::new());

        // Collect agent data for parallel processing
        let agent_data: Vec<_> = self
            .agents
            .iter()
            .filter(|(_, sim)| sim.agent.status == AgentStatus::Active)
            .filter(|(id, _)| !self.quarantine.is_quarantined(id))
            .map(|(id, sim)| (id.clone(), sim.behavior.clone()))
            .collect();

        // Generate actions in parallel chunks
        let chunk_size = self.config.chunk_size;
        let new_entries: Vec<(String, Vec<BehaviorLogEntry>)> = agent_data
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|(id, behavior)| {
                        let mut entries = Vec::new();
                        use rand::Rng;
                        let mut rng = rand::thread_rng();

                        let actions = (behavior.activity_rate + rng.gen::<f64>() * 0.5) as usize;

                        for _ in 0..actions.min(5) {
                            let success_roll: f64 = rng.gen();
                            let threshold_variance: f64 = rng.gen();

                            let threshold = behavior.success_rate
                                + (threshold_variance - 0.5) * behavior.success_variance * 2.0;

                            let outcome = if success_roll < threshold {
                                ActionOutcome::Success
                            } else {
                                ActionOutcome::Error
                            };

                            entries.push(BehaviorLogEntry {
                                timestamp,
                                action_type: "action".to_string(),
                                kredit_consumed: 10,
                                counterparties: vec![],
                                outcome,
                            });
                        }

                        (id.clone(), entries)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Apply entries and collect metrics
        for (id, entries) in new_entries {
            if let Some(sim_agent) = self.agents.get_mut(&id) {
                sim_agent.agent.behavior_log.extend(entries);
                aggregators.add_trust(sim_agent.agent.k_vector.trust_score() as f64);
                aggregators.inc_active();
            }
        }

        // Parallel gaming detection
        let gaming_incidents = if self.config.detect_gaming {
            let detectable: Vec<_> = self
                .agents
                .iter()
                .filter(|(_, sim)| sim.agent.behavior_log.len() >= 20)
                .map(|(id, sim)| (id.clone(), sim.agent.clone()))
                .collect();

            let quarantine_candidates: Vec<(String, f64)> = detectable
                .par_iter()
                .filter_map(|(id, agent)| {
                    let mut detector = GamingDetector::new(GamingDetectionConfig::default());
                    let result = detector.analyze(agent, timestamp);
                    if result.suspicion_score > 0.6 {
                        Some((id.clone(), result.suspicion_score))
                    } else {
                        None
                    }
                })
                .collect();

            let count = quarantine_candidates.len();

            // Apply quarantines sequentially
            for (id, score) in quarantine_candidates {
                self.quarantine.quarantine(
                    &id,
                    QuarantineReason::GamingDetected,
                    vec![format!("Score: {:.2}", score)],
                    timestamp,
                );
            }

            count
        } else {
            0
        };

        let active_count = aggregators.active_count();
        let result = ParallelTickResult {
            tick: self.current_tick,
            timestamp,
            total_agents: self.agents.len(),
            active_agents: active_count,
            quarantined_agents: self.quarantine.entries.len(),
            avg_trust: if active_count > 0 {
                aggregators.total_trust() / active_count as f64
            } else {
                0.0
            },
            gaming_incidents,
            sybil_evidence: 0,
        };

        self.tick_results.push(result.clone());
        self.current_tick += 1;
        result
    }

    /// Run full simulation
    pub fn run(&mut self) -> Vec<ParallelTickResult> {
        for _ in 0..self.config.ticks {
            self.tick();
        }
        self.tick_results.clone()
    }

    /// Get agent count
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Get tick results
    pub fn results(&self) -> &[ParallelTickResult] {
        &self.tick_results
    }
}

// ============================================================================
// SIMD-optimized K-Vector operations
// ============================================================================

/// Batch K-Vector operations for SIMD optimization
pub struct KVectorBatch {
    /// K-Vector components stored in SoA layout
    k_r: Vec<f32>,
    k_a: Vec<f32>,
    k_i: Vec<f32>,
    k_p: Vec<f32>,
    k_m: Vec<f32>,
    k_s: Vec<f32>,
    k_h: Vec<f32>,
    k_topo: Vec<f32>,
    k_v: Vec<f32>,
    k_coherence: Vec<f32>,
}

impl KVectorBatch {
    /// Create batch with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            k_r: Vec::with_capacity(capacity),
            k_a: Vec::with_capacity(capacity),
            k_i: Vec::with_capacity(capacity),
            k_p: Vec::with_capacity(capacity),
            k_m: Vec::with_capacity(capacity),
            k_s: Vec::with_capacity(capacity),
            k_h: Vec::with_capacity(capacity),
            k_topo: Vec::with_capacity(capacity),
            k_v: Vec::with_capacity(capacity),
            k_coherence: Vec::with_capacity(capacity),
        }
    }

    /// Add K-Vector to batch
    pub fn push(&mut self, kv: &KVector) {
        self.k_r.push(kv.k_r);
        self.k_a.push(kv.k_a);
        self.k_i.push(kv.k_i);
        self.k_p.push(kv.k_p);
        self.k_m.push(kv.k_m);
        self.k_s.push(kv.k_s);
        self.k_h.push(kv.k_h);
        self.k_topo.push(kv.k_topo);
        self.k_v.push(kv.k_v);
        self.k_coherence.push(kv.k_coherence);
    }

    /// Compute mean K-Vector
    pub fn mean(&self) -> KVector {
        let n = self.k_r.len() as f32;
        if n == 0.0 {
            return KVector::default();
        }

        KVector::new(
            self.k_r.iter().sum::<f32>() / n,
            self.k_a.iter().sum::<f32>() / n,
            self.k_i.iter().sum::<f32>() / n,
            self.k_p.iter().sum::<f32>() / n,
            self.k_m.iter().sum::<f32>() / n,
            self.k_s.iter().sum::<f32>() / n,
            self.k_h.iter().sum::<f32>() / n,
            self.k_topo.iter().sum::<f32>() / n,
            self.k_v.iter().sum::<f32>() / n,
            self.k_coherence.iter().sum::<f32>() / n,
        )
    }

    /// Compute trust scores in batch (SIMD-friendly)
    pub fn trust_scores(&self) -> Vec<f32> {
        // Weights matching KVector::trust_score()
        const W_R: f32 = 0.2;
        const W_A: f32 = 0.2;
        const W_I: f32 = 0.15;
        const W_P: f32 = 0.15;
        const W_M: f32 = 0.1;
        const W_S: f32 = 0.1;
        const W_H: f32 = 0.05;
        const W_TOPO: f32 = 0.05;

        (0..self.k_r.len())
            .map(|i| {
                W_R * self.k_r[i]
                    + W_A * self.k_a[i]
                    + W_I * self.k_i[i]
                    + W_P * self.k_p[i]
                    + W_M * self.k_m[i]
                    + W_S * self.k_s[i]
                    + W_H * self.k_h[i]
                    + W_TOPO * self.k_topo[i]
            })
            .collect()
    }

    /// Compute distances from reference K-Vector (batch)
    pub fn distances_from(&self, reference: &KVector) -> Vec<f32> {
        (0..self.k_r.len())
            .map(|i| {
                let dr = self.k_r[i] - reference.k_r;
                let da = self.k_a[i] - reference.k_a;
                let di = self.k_i[i] - reference.k_i;
                let dp = self.k_p[i] - reference.k_p;
                let dm = self.k_m[i] - reference.k_m;
                let ds = self.k_s[i] - reference.k_s;
                let dh = self.k_h[i] - reference.k_h;
                let dt = self.k_topo[i] - reference.k_topo;

                (dr * dr + da * da + di * di + dp * dp + dm * dm + ds * ds + dh * dh + dt * dt)
                    .sqrt()
            })
            .collect()
    }

    /// Length of batch
    pub fn len(&self) -> usize {
        self.k_r.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.k_r.is_empty()
    }
}

// ============================================================================
// Benchmark utilities
// ============================================================================

/// Benchmark result
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    /// Name of benchmark
    pub name: String,
    /// Number of agents
    pub agent_count: usize,
    /// Number of ticks
    pub tick_count: u64,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Average tick time in microseconds
    pub avg_tick_us: f64,
    /// Peak memory usage estimate (bytes)
    pub peak_memory_bytes: usize,
}

/// Run benchmark with given agent count
pub fn benchmark_simulation(agent_count: usize, tick_count: u64) -> BenchmarkResult {
    use std::time::Instant;

    let mut engine = ParallelSimEngine::new(ParallelSimConfig {
        ticks: tick_count,
        parallel_threshold: 500,
        ..Default::default()
    });

    // Add agents
    for i in 0..agent_count {
        engine.add_agent(
            &format!("agent-{}", i),
            SimAgentBehavior::default(),
            "sponsor",
        );
    }

    let start = Instant::now();
    engine.run();
    let duration = start.elapsed();

    let memory_estimate = agent_count
        * (std::mem::size_of::<SimAgent>() + 100 * std::mem::size_of::<BehaviorLogEntry>());

    BenchmarkResult {
        name: format!("parallel_sim_{}agents", agent_count),
        agent_count,
        tick_count,
        duration_ms: duration.as_millis() as u64,
        avg_tick_us: duration.as_micros() as f64 / tick_count as f64,
        peak_memory_bytes: memory_estimate,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::UncertaintyCalibration;

    #[test]
    fn test_random_buffer() {
        let mut buffer = RandomBuffer::new(1000, 42);

        // Values should be in [0, 1)
        for _ in 0..100 {
            let v = buffer.next();
            assert!(v >= 0.0 && v < 1.0);
        }

        // Reset and get same sequence
        buffer.reset();
        let v1 = buffer.next();
        buffer.reset();
        let v2 = buffer.next();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_tick_aggregators() {
        let agg = TickAggregators::new();

        agg.add_trust(0.5);
        agg.add_trust(0.7);
        agg.inc_active();
        agg.inc_active();
        agg.inc_gaming();

        assert!((agg.total_trust() - 1.2).abs() < 0.001);
        assert_eq!(agg.active_count(), 2);
        assert_eq!(agg.gaming_incidents(), 1);
    }

    #[test]
    fn test_parallel_engine_small() {
        let mut engine = ParallelSimEngine::new(ParallelSimConfig {
            ticks: 10,
            parallel_threshold: 1000, // Force sequential
            ..Default::default()
        });

        engine.add_population("test", SimAgentBehavior::default(), 100, "sponsor");
        assert_eq!(engine.agent_count(), 100);

        let results = engine.run();
        assert_eq!(results.len(), 10);

        // All ticks should have agents
        for result in &results {
            assert_eq!(result.total_agents, 100);
        }
    }

    #[test]
    fn test_parallel_engine_large() {
        let mut engine = ParallelSimEngine::new(ParallelSimConfig {
            ticks: 5,
            parallel_threshold: 100, // Force parallel
            chunk_size: 50,
            ..Default::default()
        });

        engine.add_population("test", SimAgentBehavior::default(), 1000, "sponsor");
        assert_eq!(engine.agent_count(), 1000);

        let results = engine.run();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_kvector_batch() {
        let mut batch = KVectorBatch::with_capacity(3);

        batch.push(&KVector::new(
            0.6, 0.5, 0.7, 0.6, 0.3, 0.4, 0.5, 0.3, 0.6, 0.55,
        ));
        batch.push(&KVector::new(
            0.8, 0.7, 0.9, 0.8, 0.5, 0.6, 0.7, 0.5, 0.8, 0.75,
        ));
        batch.push(&KVector::new(
            0.4, 0.3, 0.5, 0.4, 0.1, 0.2, 0.3, 0.1, 0.4, 0.35,
        ));

        assert_eq!(batch.len(), 3);

        let mean = batch.mean();
        assert!((mean.k_r - 0.6).abs() < 0.001);

        let scores = batch.trust_scores();
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_benchmark() {
        let result = benchmark_simulation(100, 5);
        assert_eq!(result.agent_count, 100);
        assert_eq!(result.tick_count, 5);
        assert!(result.duration_ms > 0 || result.avg_tick_us > 0.0);
    }
}
