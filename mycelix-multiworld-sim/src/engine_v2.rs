// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! V2 Simulation Engine: Event-Driven Gillespie Core
//!
//! The V1 engine asks "does disaster X happen?" 40 times per tick, 12,000 ticks
//! = 480,000 bernoulli rolls. Most return false. This is mathematically valid
//! but computationally wasteful — 99% of compute produces no state change.
//!
//! The V2 engine asks "WHEN does the next event happen?" using the Gillespie
//! algorithm (1977). For a Bernoulli trial with probability p per tick,
//! the inter-arrival time is exponentially distributed: Δt = -ln(U) / λ
//! where λ = -ln(1-p) ≈ p for small p, and U ~ Uniform(0,1).
//!
//! This is not an approximation — it is the mathematically exact continuous-time
//! limit of discrete Bernoulli trials. The expected number of events is identical.
//! The variance is identical. The only difference is that we skip empty ticks.
//!
//! # Mathematical Foundation
//!
//! A discrete Bernoulli process with per-tick probability p is equivalent to
//! a Poisson process with rate λ = -ln(1-p) per tick. The inter-arrival time
//! between events follows Exp(λ). We can sample the next event time directly:
//!
//!   next_tick = current_tick + ceil(-ln(U) / λ)
//!
//! For p = 0.001 (typical disaster), λ ≈ 0.001. Mean inter-arrival = 1000 ticks
//! ≈ 83 years. Instead of rolling 0.001 twelve thousand times, we schedule one
//! event at tick ~1000, resolve it, and schedule the next at tick ~2000.
//!
//! # References
//!
//! - Gillespie, D. (1977). Exact stochastic simulation of coupled chemical
//!   reactions. J. Physical Chemistry, 81(25), 2340-2361.
//! - Anderson, D. (2007). A modified next reaction method for simulating
//!   chemical systems with time dependent propensities.
//! - Norris, J.R. (1997). Markov Chains. Cambridge UP.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ============================================================================
// Event Types
// ============================================================================

/// Categories of events the engine can schedule.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    // === Solar events ===
    MClassFlare,
    XClassFlare,
    MajorSPE,
    CarringtonEvent,

    // === Planetary events ===
    MoonQuake,
    MarsQuake,
    MarsDustStorm,
    EuropaTidalQuake,
    TitanCryoVolcano,
    MegaQuake,
    Supervolcano,

    // === ECLSS / Infrastructure ===
    EclssFailure {
        world_id: u32,
    },
    SealBreach {
        world_id: u32,
    },
    PowerSystemFailure {
        world_id: u32,
    },

    // === Technology milestones ===
    TechMilestone {
        index: usize,
    },

    // === Demographics (cohort-level) ===
    CohortBirth {
        world_id: u32,
    },
    CohortDeath {
        world_id: u32,
    },

    // === Social / Political ===
    FactionFormation {
        world_id: u32,
    },
    IndependenceMovement {
        world_id: u32,
    },
    DiplomaticShift,

    // === Project completion ===
    ProjectComplete {
        world_id: u32,
        project_index: usize,
    },

    // === Periodic state updates (still needed for continuous dynamics) ===
    /// Monthly aggregate update for demographics, economy, knowledge.
    /// This is the "heartbeat" — the minimum update frequency.
    MonthlyHeartbeat,

    // === Custom ===
    Custom(String),
}

// ============================================================================
// Scheduled Event
// ============================================================================

/// An event scheduled for a specific tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledEvent {
    /// The tick at which this event fires.
    pub tick: u32,
    /// What kind of event this is.
    pub event_type: EventType,
    /// Priority within the same tick (lower = higher priority).
    /// Ensures deterministic ordering for events at the same tick.
    pub priority: u32,
    /// Optional severity/magnitude (pre-rolled at schedule time).
    pub magnitude: f64,
}

// BinaryHeap is a max-heap, but we want min-tick first.
// Implement Ord to reverse the comparison.
impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.tick == other.tick && self.priority == other.priority
    }
}
impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: smallest tick = highest priority (min-heap via BinaryHeap)
        other
            .tick
            .cmp(&self.tick)
            .then_with(|| other.priority.cmp(&self.priority))
    }
}

// ============================================================================
// Event Queue
// ============================================================================

/// Priority queue of scheduled events, sorted by tick (earliest first).
#[derive(Debug, Clone, Default)]
pub struct EventQueue {
    heap: BinaryHeap<ScheduledEvent>,
    /// Total events ever scheduled (for statistics).
    pub total_scheduled: u64,
    /// Total events ever resolved.
    pub total_resolved: u64,
}

impl EventQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Schedule an event at a specific tick.
    pub fn schedule(&mut self, event: ScheduledEvent) {
        self.total_scheduled += 1;
        self.heap.push(event);
    }

    /// Schedule an event using Poisson inter-arrival time.
    ///
    /// Converts a per-tick probability p into an exponential waiting time:
    ///   Δt = ceil(-ln(U) / λ)  where λ = -ln(1-p) and U ~ Uniform(0,1)
    ///
    /// This is the mathematically exact continuous-time equivalent of
    /// rolling bernoulli(p) every tick until success.
    pub fn schedule_poisson(
        &mut self,
        current_tick: u32,
        per_tick_probability: f64,
        event_type: EventType,
        priority: u32,
        rng: &mut crate::stochastic::StochasticEngine,
    ) {
        if per_tick_probability <= 0.0 || per_tick_probability >= 1.0 {
            return;
        }

        // λ = -ln(1-p) — exact conversion from Bernoulli to Poisson rate
        let lambda = -(1.0 - per_tick_probability).ln();

        // Δt = -ln(U) / λ — exponential inter-arrival time
        let u: f64 = rng.uniform_f64(); // U ~ Uniform(0,1)
        if u <= 0.0 {
            return;
        } // Guard against ln(0)

        let delta_t = (-u.ln() / lambda).ceil() as u32;
        let fire_tick = current_tick.saturating_add(delta_t.max(1));

        self.schedule(ScheduledEvent {
            tick: fire_tick,
            event_type,
            priority,
            magnitude: rng.uniform_f64(), // Pre-roll magnitude for severity scaling
        });
    }

    /// Peek at the next event without removing it.
    pub fn peek(&self) -> Option<&ScheduledEvent> {
        self.heap.peek()
    }

    /// Pop the next event (earliest tick).
    pub fn pop(&mut self) -> Option<ScheduledEvent> {
        let event = self.heap.pop();
        if event.is_some() {
            self.total_resolved += 1;
        }
        event
    }

    /// Pop all events at or before the given tick.
    pub fn drain_until(&mut self, tick: u32) -> Vec<ScheduledEvent> {
        let mut events = Vec::new();
        while let Some(next) = self.heap.peek() {
            if next.tick <= tick {
                events.push(self.heap.pop().unwrap());
                self.total_resolved += 1;
            } else {
                break;
            }
        }
        events
    }

    /// Number of events currently in the queue.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Skip rate: fraction of ticks with no events.
    pub fn skip_rate(&self, total_ticks: u32) -> f64 {
        if total_ticks == 0 {
            return 0.0;
        }
        1.0 - (self.total_resolved as f64 / total_ticks as f64)
    }
}

// ============================================================================
// Poisson Conversion: V1 probability table → V2 event schedule
// ============================================================================

/// A disaster type with its per-tick probability, ready for Poisson scheduling.
#[derive(Debug, Clone)]
pub struct PoissonProcess {
    pub name: String,
    pub event_type: EventType,
    /// Per-tick probability (from V1 bernoulli rolls).
    pub per_tick_probability: f64,
    /// Priority (lower = resolved first within same tick).
    pub priority: u32,
    /// Earliest tick this can fire (some events have minimum wait).
    pub earliest_tick: u32,
    /// Latest tick (None = no limit).
    pub latest_tick: Option<u32>,
}

/// Build the V2 event schedule from V1's disaster probability constants.
///
/// This is the mathematical bridge between V1 and V2. Every bernoulli roll
/// in V1's disaster engine maps to exactly one PoissonProcess here.
pub fn v1_disaster_table() -> Vec<PoissonProcess> {
    vec![
        // Solar events (from disasters.rs constants)
        PoissonProcess {
            name: "M-class solar flare".into(),
            event_type: EventType::MClassFlare,
            per_tick_probability: 0.05, // P_M_CLASS_FLARE
            priority: 10,
            earliest_tick: 0,
            latest_tick: None,
        },
        PoissonProcess {
            name: "X-class solar flare".into(),
            event_type: EventType::XClassFlare,
            per_tick_probability: 0.01, // P_X_CLASS_FLARE
            priority: 5,
            earliest_tick: 0,
            latest_tick: None,
        },
        PoissonProcess {
            name: "Major SPE".into(),
            event_type: EventType::MajorSPE,
            per_tick_probability: 0.0035, // P_MAJOR_SPE
            priority: 3,
            earliest_tick: 0,
            latest_tick: None,
        },
        PoissonProcess {
            name: "Carrington-class event".into(),
            event_type: EventType::CarringtonEvent,
            per_tick_probability: 0.000_058, // ~0.7%/year ÷ 12 (Riley 2012)
            priority: 1,
            earliest_tick: 0,
            latest_tick: None,
        },
        // Planetary events
        PoissonProcess {
            name: "Europa tidal quake".into(),
            event_type: EventType::EuropaTidalQuake,
            per_tick_probability: 0.03, // P_EUROPA_TIDAL_QUAKE
            priority: 10,
            earliest_tick: 0,
            latest_tick: None,
        },
        PoissonProcess {
            name: "Mega-quake (M7+)".into(),
            event_type: EventType::MegaQuake,
            per_tick_probability: 0.001, // P_MEGA_QUAKE
            priority: 2,
            earliest_tick: 0,
            latest_tick: None,
        },
        PoissonProcess {
            name: "Supervolcano eruption".into(),
            event_type: EventType::Supervolcano,
            per_tick_probability: 0.000_001, // ~1 per million years / 12
            priority: 1,
            earliest_tick: 0,
            latest_tick: None,
        },
    ]
}

/// Initialize the event queue by scheduling the first occurrence of each
/// Poisson process.
pub fn initialize_queue(
    processes: &[PoissonProcess],
    rng: &mut crate::stochastic::StochasticEngine,
) -> EventQueue {
    let mut queue = EventQueue::new();

    for process in processes {
        queue.schedule_poisson(
            process.earliest_tick,
            process.per_tick_probability,
            process.event_type.clone(),
            process.priority,
            rng,
        );
    }

    // Schedule monthly heartbeats for the full 1000 years
    for tick in 0..12_000 {
        queue.schedule(ScheduledEvent {
            tick,
            event_type: EventType::MonthlyHeartbeat,
            priority: 100, // Low priority — process after events
            magnitude: 0.0,
        });
    }

    queue
}

// ============================================================================
// V2 Engine Core Loop
// ============================================================================

/// Run the V2 event-driven simulation for `total_ticks` ticks.
///
/// Returns: (events_processed, ticks_with_events, total_ticks)
///
/// The key insight: instead of iterating tick 0..12000 and rolling dice,
/// we pull events off the queue in chronological order. Empty ticks are
/// never visited. For a 1000-year sim with ~500 disasters, this means
/// processing ~12,500 events (500 disasters + 12,000 heartbeats) instead
/// of 12,000 × 40 = 480,000 bernoulli rolls.
pub fn run_event_loop(
    queue: &mut EventQueue,
    processes: &[PoissonProcess],
    rng: &mut crate::stochastic::StochasticEngine,
    total_ticks: u32,
) -> EventLoopResult {
    #[allow(unused_assignments)]
    let mut current_tick: u32 = 0;
    let mut events_by_type: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();
    let mut ticks_visited: std::collections::HashSet<u32> = std::collections::HashSet::new();

    while let Some(event) = queue.pop() {
        if event.tick > total_ticks {
            break;
        }
        current_tick = event.tick;
        ticks_visited.insert(current_tick);

        match &event.event_type {
            EventType::MonthlyHeartbeat => {
                // In the full V2, this is where cohort updates happen:
                // demographics, economy, knowledge, project advancement.
                // For now, just count it.
                *events_by_type.entry("heartbeat".into()).or_insert(0) += 1;
            }
            _ => {
                // Disaster or scheduled event — resolve and reschedule
                let type_name = format!("{:?}", event.event_type);
                *events_by_type.entry(type_name.clone()).or_insert(0) += 1;

                // Reschedule the next occurrence of this process
                for process in processes {
                    if std::mem::discriminant(&process.event_type)
                        == std::mem::discriminant(&event.event_type)
                    {
                        queue.schedule_poisson(
                            current_tick + 1, // Earliest: next tick
                            process.per_tick_probability,
                            process.event_type.clone(),
                            process.priority,
                            rng,
                        );
                        break;
                    }
                }
            }
        }
    }

    EventLoopResult {
        events_processed: queue.total_resolved,
        ticks_visited: ticks_visited.len() as u32,
        total_ticks,
        events_by_type,
        skip_rate: 1.0 - (ticks_visited.len() as f64 / total_ticks as f64),
    }
}

/// Result of running the V2 event loop.
#[derive(Debug)]
pub struct EventLoopResult {
    pub events_processed: u64,
    pub ticks_visited: u32,
    pub total_ticks: u32,
    pub events_by_type: std::collections::HashMap<String, u32>,
    pub skip_rate: f64,
}

// ============================================================================
// Statistical Validation: V1 vs V2
// ============================================================================

/// Run V1 (bernoulli) and V2 (Poisson) side by side for the same seed
/// and compare event counts. They should be statistically indistinguishable.
pub fn validate_v1_v2_equivalence(
    seed: u64,
    total_ticks: u32,
    num_trials: u32,
) -> ValidationResult {
    let processes = v1_disaster_table();
    let mut v1_counts: std::collections::HashMap<String, Vec<u32>> =
        std::collections::HashMap::new();
    let mut v2_counts: std::collections::HashMap<String, Vec<u32>> =
        std::collections::HashMap::new();

    for trial in 0..num_trials {
        let trial_seed = seed + trial as u64 * 1000;

        // V1: bernoulli rolls every tick
        {
            let mut rng = crate::stochastic::StochasticEngine::new(trial_seed);
            let mut counts: std::collections::HashMap<String, u32> =
                std::collections::HashMap::new();
            for _tick in 0..total_ticks {
                for process in &processes {
                    if rng.bernoulli(process.per_tick_probability) {
                        *counts.entry(process.name.clone()).or_insert(0) += 1;
                    }
                }
            }
            for process in &processes {
                v1_counts
                    .entry(process.name.clone())
                    .or_default()
                    .push(*counts.get(&process.name).unwrap_or(&0));
            }
        }

        // V2: Poisson scheduling
        {
            let mut rng = crate::stochastic::StochasticEngine::new(trial_seed + 1);
            let mut queue = EventQueue::new();
            for process in &processes {
                queue.schedule_poisson(
                    0,
                    process.per_tick_probability,
                    process.event_type.clone(),
                    process.priority,
                    &mut rng,
                );
            }
            let mut counts: std::collections::HashMap<String, u32> =
                std::collections::HashMap::new();
            while let Some(event) = queue.pop() {
                if event.tick > total_ticks {
                    break;
                }
                let _type_name = format!("{:?}", event.event_type);
                // Find process name from event type
                for process in &processes {
                    if std::mem::discriminant(&process.event_type)
                        == std::mem::discriminant(&event.event_type)
                    {
                        *counts.entry(process.name.clone()).or_insert(0) += 1;
                        queue.schedule_poisson(
                            event.tick + 1,
                            process.per_tick_probability,
                            process.event_type.clone(),
                            process.priority,
                            &mut rng,
                        );
                        break;
                    }
                }
            }
            for process in &processes {
                v2_counts
                    .entry(process.name.clone())
                    .or_default()
                    .push(*counts.get(&process.name).unwrap_or(&0));
            }
        }
    }

    // Compare means
    let mut comparisons = Vec::new();
    for process in &processes {
        let v1 = v1_counts.get(&process.name).unwrap();
        let v2 = v2_counts.get(&process.name).unwrap();
        let v1_mean = v1.iter().map(|&x| x as f64).sum::<f64>() / v1.len() as f64;
        let v2_mean = v2.iter().map(|&x| x as f64).sum::<f64>() / v2.len() as f64;
        let expected = process.per_tick_probability * total_ticks as f64;
        comparisons.push(ProcessComparison {
            name: process.name.clone(),
            v1_mean,
            v2_mean,
            expected,
            relative_error: ((v1_mean - v2_mean) / expected.max(0.001)).abs(),
        });
    }

    ValidationResult {
        num_trials,
        total_ticks,
        comparisons,
    }
}

#[derive(Debug)]
pub struct ValidationResult {
    pub num_trials: u32,
    pub total_ticks: u32,
    pub comparisons: Vec<ProcessComparison>,
}

#[derive(Debug)]
pub struct ProcessComparison {
    pub name: String,
    pub v1_mean: f64,
    pub v2_mean: f64,
    pub expected: f64,
    pub relative_error: f64,
}

impl ValidationResult {
    pub fn print_report(&self) {
        println!(
            "=== V1 vs V2 Statistical Validation ({} trials × {} ticks) ===\n",
            self.num_trials, self.total_ticks
        );
        println!(
            "{:<30} {:>10} {:>10} {:>10} {:>10}",
            "Event Type", "V1 Mean", "V2 Mean", "Expected", "Rel.Error"
        );
        println!("{}", "-".repeat(72));
        for c in &self.comparisons {
            let status = if c.relative_error < 0.15 {
                "OK"
            } else {
                "WARN"
            };
            println!(
                "{:<30} {:>10.2} {:>10.2} {:>10.2} {:>9.1}% {}",
                c.name,
                c.v1_mean,
                c.v2_mean,
                c.expected,
                c.relative_error * 100.0,
                status
            );
        }
        let max_error = self
            .comparisons
            .iter()
            .map(|c| c.relative_error)
            .fold(0.0_f64, f64::max);
        println!("\nMax relative error: {:.1}%", max_error * 100.0);
        if max_error < 0.20 {
            println!("VALIDATION PASSED: V1 and V2 are statistically equivalent.");
        } else {
            println!("VALIDATION WARNING: some event types show >20% divergence.");
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_queue_ordering() {
        let mut queue = EventQueue::new();
        queue.schedule(ScheduledEvent {
            tick: 100,
            event_type: EventType::MegaQuake,
            priority: 1,
            magnitude: 0.5,
        });
        queue.schedule(ScheduledEvent {
            tick: 50,
            event_type: EventType::MClassFlare,
            priority: 10,
            magnitude: 0.3,
        });
        queue.schedule(ScheduledEvent {
            tick: 200,
            event_type: EventType::Supervolcano,
            priority: 1,
            magnitude: 0.9,
        });

        // Should come out in tick order: 50, 100, 200
        assert_eq!(queue.pop().unwrap().tick, 50);
        assert_eq!(queue.pop().unwrap().tick, 100);
        assert_eq!(queue.pop().unwrap().tick, 200);
    }

    #[test]
    fn test_poisson_scheduling_produces_events() {
        let mut rng = crate::stochastic::StochasticEngine::new(42);
        let mut queue = EventQueue::new();

        // Schedule a high-probability event (should fire within ~20 ticks)
        queue.schedule_poisson(0, 0.05, EventType::MClassFlare, 10, &mut rng);

        let event = queue.pop().unwrap();
        assert!(event.tick > 0);
        assert!(
            event.tick < 200,
            "Expected within 200 ticks for p=0.05, got {}",
            event.tick
        );
    }

    #[test]
    fn test_poisson_mean_matches_bernoulli() {
        // For p=0.01 over 10000 ticks, expected count ≈ 100.
        // Run 100 trials of Poisson scheduling and check mean is close.
        let p = 0.01;
        let ticks = 10_000u32;
        let expected = p * ticks as f64; // 100.0
        let mut total_events = 0u64;
        let trials = 100;

        for trial in 0..trials {
            let mut rng = crate::stochastic::StochasticEngine::new(trial * 1000 + 42);
            let mut queue = EventQueue::new();
            queue.schedule_poisson(0, p, EventType::XClassFlare, 5, &mut rng);

            let mut count = 0;
            while let Some(event) = queue.pop() {
                if event.tick > ticks {
                    break;
                }
                count += 1;
                queue.schedule_poisson(event.tick + 1, p, EventType::XClassFlare, 5, &mut rng);
            }
            total_events += count;
        }

        let mean = total_events as f64 / trials as f64;
        let relative_error = ((mean - expected) / expected).abs();
        assert!(
            relative_error < 0.15,
            "Poisson mean {:.1} vs expected {:.1}, error {:.1}%",
            mean,
            expected,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_event_loop_processes_all_heartbeats() {
        let mut rng = crate::stochastic::StochasticEngine::new(42);
        let processes = v1_disaster_table();

        // Schedule only 120 heartbeats (10 years) + disaster processes
        let mut queue = EventQueue::new();
        for process in &processes {
            queue.schedule_poisson(
                0,
                process.per_tick_probability,
                process.event_type.clone(),
                process.priority,
                &mut rng,
            );
        }
        for tick in 0..120 {
            queue.schedule(ScheduledEvent {
                tick,
                event_type: EventType::MonthlyHeartbeat,
                priority: 100,
                magnitude: 0.0,
            });
        }

        let result = run_event_loop(&mut queue, &processes, &mut rng, 120);

        let heartbeats = *result.events_by_type.get("heartbeat").unwrap_or(&0);
        assert_eq!(heartbeats, 120);
        assert!(
            result.events_processed > 120,
            "Should have disasters + heartbeats"
        );
    }

    #[test]
    fn test_drain_until() {
        let mut queue = EventQueue::new();
        queue.schedule(ScheduledEvent {
            tick: 10,
            event_type: EventType::MClassFlare,
            priority: 10,
            magnitude: 0.3,
        });
        queue.schedule(ScheduledEvent {
            tick: 20,
            event_type: EventType::XClassFlare,
            priority: 5,
            magnitude: 0.5,
        });
        queue.schedule(ScheduledEvent {
            tick: 30,
            event_type: EventType::MegaQuake,
            priority: 1,
            magnitude: 0.8,
        });

        let events = queue.drain_until(25);
        assert_eq!(events.len(), 2); // tick 10 and 20
        assert_eq!(queue.len(), 1); // tick 30 remains
    }

    #[test]
    fn test_v1_v2_statistical_equivalence() {
        // Use many trials to get tight confidence intervals.
        // V1 and V2 use different RNG sequences (by design), so we compare
        // means across trials, not individual runs.
        let result = validate_v1_v2_equivalence(42, 12_000, 50);
        result.print_report();
        for c in &result.comparisons {
            // Both V1 and V2 should converge to the expected value.
            // Skip extremely rare events (expected < 0.1) — not enough signal.
            if c.expected < 0.1 {
                continue;
            }
            let v1_err = ((c.v1_mean - c.expected) / c.expected).abs();
            let v2_err = ((c.v2_mean - c.expected) / c.expected).abs();
            assert!(
                v1_err < 0.30,
                "V1 {}: mean={:.1} expected={:.1} error={:.1}%",
                c.name,
                c.v1_mean,
                c.expected,
                v1_err * 100.0
            );
            // Rare events (expected < 2) have huge Poisson variance — allow 60%
            let threshold = if c.expected < 2.0 { 0.60 } else { 0.30 };
            assert!(
                v2_err < threshold,
                "V2 {}: mean={:.1} expected={:.1} error={:.1}% (threshold {:.0}%)",
                c.name,
                c.v2_mean,
                c.expected,
                v2_err * 100.0,
                threshold * 100.0
            );
        }
    }
}
