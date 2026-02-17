//! Metrics collection for aggregator monitoring.

use crate::{NodeId, Round};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Aggregator metrics for monitoring and observability.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AggregatorMetrics {
    /// Total gradients received.
    pub total_gradients: u64,

    /// Total rounds completed.
    pub total_rounds: u64,

    /// Total aggregations performed.
    pub total_aggregations: u64,

    /// Submission counts per node.
    #[serde(skip)]
    pub submissions_per_node: HashMap<NodeId, u64>,

    /// Average aggregation time in microseconds.
    pub avg_aggregation_time_us: f64,

    /// Maximum aggregation time in microseconds.
    pub max_aggregation_time_us: u64,

    /// Average gradients per round.
    pub avg_gradients_per_round: f64,

    /// Last gradient dimension.
    pub last_gradient_dim: usize,

    /// Byzantine rejections (gradients flagged as malicious).
    pub byzantine_rejections: u64,

    /// Memory high water mark in bytes.
    pub memory_high_water_bytes: usize,

    /// Internal tracking.
    #[serde(skip)]
    aggregation_times: Vec<Duration>,
    #[serde(skip)]
    gradients_per_round: Vec<usize>,
}

impl AggregatorMetrics {
    /// Create new metrics instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a gradient submission.
    pub fn record_submission(&mut self, node_id: &str, dimension: usize) {
        self.total_gradients += 1;
        self.last_gradient_dim = dimension;

        *self
            .submissions_per_node
            .entry(node_id.to_string())
            .or_insert(0) += 1;
    }

    /// Record an aggregation.
    pub fn record_aggregation(&mut self, round: Round, num_gradients: usize, duration: Duration) {
        self.total_rounds = round + 1;
        self.total_aggregations += 1;

        // Track aggregation time
        self.aggregation_times.push(duration);
        let duration_us = duration.as_micros() as u64;
        self.max_aggregation_time_us = self.max_aggregation_time_us.max(duration_us);

        // Update average
        let total_time: Duration = self.aggregation_times.iter().sum();
        self.avg_aggregation_time_us =
            total_time.as_micros() as f64 / self.aggregation_times.len() as f64;

        // Track gradients per round
        self.gradients_per_round.push(num_gradients);
        self.avg_gradients_per_round =
            self.gradients_per_round.iter().sum::<usize>() as f64 / self.gradients_per_round.len() as f64;

        // Prune old data to prevent unbounded growth
        if self.aggregation_times.len() > 1000 {
            self.aggregation_times.drain(0..500);
            self.gradients_per_round.drain(0..500);
        }
    }

    /// Record a Byzantine rejection.
    pub fn record_byzantine_rejection(&mut self) {
        self.byzantine_rejections += 1;
    }

    /// Update memory high water mark.
    pub fn update_memory_high_water(&mut self, current_bytes: usize) {
        self.memory_high_water_bytes = self.memory_high_water_bytes.max(current_bytes);
    }

    /// Get Prometheus-format metrics.
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP fl_gradients_total Total gradients received\n");
        output.push_str("# TYPE fl_gradients_total counter\n");
        output.push_str(&format!("fl_gradients_total {}\n", self.total_gradients));

        output.push_str("# HELP fl_rounds_total Total rounds completed\n");
        output.push_str("# TYPE fl_rounds_total counter\n");
        output.push_str(&format!("fl_rounds_total {}\n", self.total_rounds));

        output.push_str("# HELP fl_aggregation_time_us Average aggregation time in microseconds\n");
        output.push_str("# TYPE fl_aggregation_time_us gauge\n");
        output.push_str(&format!(
            "fl_aggregation_time_us {:.2}\n",
            self.avg_aggregation_time_us
        ));

        output.push_str("# HELP fl_byzantine_rejections Total Byzantine rejections\n");
        output.push_str("# TYPE fl_byzantine_rejections counter\n");
        output.push_str(&format!(
            "fl_byzantine_rejections {}\n",
            self.byzantine_rejections
        ));

        output.push_str("# HELP fl_memory_high_water_bytes Peak memory usage\n");
        output.push_str("# TYPE fl_memory_high_water_bytes gauge\n");
        output.push_str(&format!(
            "fl_memory_high_water_bytes {}\n",
            self.memory_high_water_bytes
        ));

        output
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Per-round statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundStats {
    pub round: Round,
    pub num_gradients: usize,
    pub aggregation_time: Duration,
    pub nodes_participated: Vec<NodeId>,
    pub byzantine_detected: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Round statistics collector.
#[derive(Clone, Debug, Default)]
pub struct RoundStatsCollector {
    history: Vec<RoundStats>,
    max_history: usize,
}

impl RoundStatsCollector {
    /// Create new collector with history limit.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
        }
    }

    /// Record stats for a round.
    pub fn record(
        &mut self,
        round: Round,
        num_gradients: usize,
        aggregation_time: Duration,
        nodes: Vec<NodeId>,
        byzantine: usize,
    ) {
        let stats = RoundStats {
            round,
            num_gradients,
            aggregation_time,
            nodes_participated: nodes,
            byzantine_detected: byzantine,
            timestamp: chrono::Utc::now(),
        };

        self.history.push(stats);

        // Prune old history
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get recent round stats.
    pub fn recent(&self, n: usize) -> &[RoundStats] {
        let start = self.history.len().saturating_sub(n);
        &self.history[start..]
    }

    /// Get stats for a specific round.
    pub fn get(&self, round: Round) -> Option<&RoundStats> {
        self.history.iter().find(|s| s.round == round)
    }

    /// Calculate participation rate over recent rounds.
    pub fn participation_rate(&self, total_nodes: usize, rounds: usize) -> f64 {
        let recent = self.recent(rounds);
        if recent.is_empty() || total_nodes == 0 {
            return 0.0;
        }

        let total_participated: usize = recent.iter().map(|s| s.num_gradients).sum();
        let total_possible = total_nodes * recent.len();

        total_participated as f64 / total_possible as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let mut metrics = AggregatorMetrics::new();

        metrics.record_submission("node1", 1000);
        metrics.record_submission("node2", 1000);
        assert_eq!(metrics.total_gradients, 2);

        metrics.record_aggregation(0, 2, Duration::from_millis(10));
        assert_eq!(metrics.total_rounds, 1);
        assert!(metrics.avg_aggregation_time_us > 0.0);
    }

    #[test]
    fn test_prometheus_output() {
        let mut metrics = AggregatorMetrics::new();
        metrics.total_gradients = 100;
        metrics.total_rounds = 10;

        let output = metrics.to_prometheus();
        assert!(output.contains("fl_gradients_total 100"));
        assert!(output.contains("fl_rounds_total 10"));
    }

    #[test]
    fn test_round_stats_collector() {
        let mut collector = RoundStatsCollector::new(100);

        collector.record(0, 5, Duration::from_millis(10), vec!["n1".into()], 0);
        collector.record(1, 4, Duration::from_millis(12), vec!["n1".into()], 1);

        let recent = collector.recent(2);
        assert_eq!(recent.len(), 2);

        let rate = collector.participation_rate(5, 2);
        assert!(rate > 0.0);
    }
}
