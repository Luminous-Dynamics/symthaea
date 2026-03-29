// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Congestion control algorithms

use std::time::{Duration, Instant};

/// Congestion control state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionState {
    SlowStart,
    CongestionAvoidance,
    Recovery,
}

/// BBR-inspired congestion controller
pub struct CongestionController {
    /// Current congestion window (bytes)
    cwnd: u64,
    /// Slow start threshold
    ssthresh: u64,
    /// Current state
    state: CongestionState,
    /// Bytes in flight
    bytes_in_flight: u64,
    /// RTT samples
    min_rtt: Duration,
    latest_rtt: Duration,
    /// Bandwidth estimate (bytes/second)
    bandwidth_estimate: f64,
    /// Pacing rate (bytes/second)
    pacing_rate: f64,
    /// Last update time
    last_update: Instant,
    /// Packets acked since last loss
    packets_since_loss: u64,
}

impl CongestionController {
    /// Initial congestion window (10 packets * 1400 bytes)
    const INITIAL_CWND: u64 = 14000;
    /// Minimum congestion window
    const MIN_CWND: u64 = 2800;
    /// Maximum congestion window (1 MB)
    const MAX_CWND: u64 = 1024 * 1024;

    pub fn new() -> Self {
        Self {
            cwnd: Self::INITIAL_CWND,
            ssthresh: u64::MAX,
            state: CongestionState::SlowStart,
            bytes_in_flight: 0,
            min_rtt: Duration::from_millis(100),
            latest_rtt: Duration::from_millis(100),
            bandwidth_estimate: 1_000_000.0, // 1 MB/s initial
            pacing_rate: 1_000_000.0,
            last_update: Instant::now(),
            packets_since_loss: 0,
        }
    }

    /// Record RTT sample
    pub fn on_rtt_sample(&mut self, rtt: Duration) {
        self.latest_rtt = rtt;
        if rtt < self.min_rtt {
            self.min_rtt = rtt;
        }

        // Update bandwidth estimate
        self.update_bandwidth_estimate();
    }

    /// Record acknowledged bytes
    pub fn on_ack(&mut self, bytes: u64) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes);
        self.packets_since_loss += 1;

        match self.state {
            CongestionState::SlowStart => {
                // Exponential growth
                self.cwnd = (self.cwnd + bytes).min(Self::MAX_CWND);

                if self.cwnd >= self.ssthresh {
                    self.state = CongestionState::CongestionAvoidance;
                }
            }
            CongestionState::CongestionAvoidance => {
                // Linear growth (approximately 1 MSS per RTT)
                let increment = (bytes as f64 * 1400.0 / self.cwnd as f64) as u64;
                self.cwnd = (self.cwnd + increment).min(Self::MAX_CWND);
            }
            CongestionState::Recovery => {
                // Stay in recovery until new acks
                if self.packets_since_loss > 3 {
                    self.state = CongestionState::CongestionAvoidance;
                }
            }
        }

        self.update_pacing_rate();
    }

    /// Handle packet loss
    pub fn on_loss(&mut self) {
        self.packets_since_loss = 0;

        match self.state {
            CongestionState::SlowStart => {
                self.ssthresh = self.cwnd / 2;
                self.cwnd = self.ssthresh.max(Self::MIN_CWND);
                self.state = CongestionState::Recovery;
            }
            CongestionState::CongestionAvoidance => {
                self.ssthresh = self.cwnd / 2;
                self.cwnd = self.ssthresh.max(Self::MIN_CWND);
                self.state = CongestionState::Recovery;
            }
            CongestionState::Recovery => {
                // Already in recovery, don't reduce further
            }
        }

        self.update_pacing_rate();
    }

    /// Handle timeout
    pub fn on_timeout(&mut self) {
        self.ssthresh = self.cwnd / 2;
        self.cwnd = Self::INITIAL_CWND;
        self.state = CongestionState::SlowStart;
        self.packets_since_loss = 0;
        self.update_pacing_rate();
    }

    /// Record bytes sent
    pub fn on_send(&mut self, bytes: u64) {
        self.bytes_in_flight += bytes;
    }

    /// Check if we can send more data
    pub fn can_send(&self) -> bool {
        self.bytes_in_flight < self.cwnd
    }

    /// Get available send window
    pub fn available_window(&self) -> u64 {
        self.cwnd.saturating_sub(self.bytes_in_flight)
    }

    /// Get current congestion window
    pub fn cwnd(&self) -> u64 {
        self.cwnd
    }

    /// Get pacing rate (bytes/second)
    pub fn pacing_rate(&self) -> f64 {
        self.pacing_rate
    }

    /// Get pacing interval for given packet size
    pub fn pacing_interval(&self, packet_size: u64) -> Duration {
        let seconds = packet_size as f64 / self.pacing_rate;
        Duration::from_secs_f64(seconds)
    }

    /// Get bandwidth estimate (bytes/second)
    pub fn bandwidth_estimate(&self) -> f64 {
        self.bandwidth_estimate
    }

    /// Get current state
    pub fn state(&self) -> CongestionState {
        self.state
    }

    fn update_bandwidth_estimate(&mut self) {
        // BDP-based bandwidth estimate
        if self.min_rtt.as_secs_f64() > 0.0 {
            self.bandwidth_estimate = self.cwnd as f64 / self.min_rtt.as_secs_f64();
        }
    }

    fn update_pacing_rate(&mut self) {
        // Pace at 1.25x estimated bandwidth to probe for more
        self.pacing_rate = self.bandwidth_estimate * 1.25;
    }
}

impl Default for CongestionController {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate limiter for smooth packet transmission
pub struct RateLimiter {
    tokens: f64,
    max_tokens: f64,
    rate: f64, // tokens per second
    last_update: Instant,
}

impl RateLimiter {
    pub fn new(rate_bytes_per_second: f64, burst_size: f64) -> Self {
        Self {
            tokens: burst_size,
            max_tokens: burst_size,
            rate: rate_bytes_per_second,
            last_update: Instant::now(),
        }
    }

    /// Update rate
    pub fn set_rate(&mut self, rate: f64) {
        self.refill();
        self.rate = rate;
    }

    /// Check if we can send bytes
    pub fn can_send(&mut self, bytes: u64) -> bool {
        self.refill();
        self.tokens >= bytes as f64
    }

    /// Consume tokens for sending
    pub fn consume(&mut self, bytes: u64) -> bool {
        self.refill();
        if self.tokens >= bytes as f64 {
            self.tokens -= bytes as f64;
            true
        } else {
            false
        }
    }

    /// Get time until enough tokens for bytes
    pub fn time_until_ready(&mut self, bytes: u64) -> Duration {
        self.refill();
        if self.tokens >= bytes as f64 {
            Duration::ZERO
        } else {
            let needed = bytes as f64 - self.tokens;
            Duration::from_secs_f64(needed / self.rate)
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.max_tokens);
        self.last_update = now;
    }
}

/// Adaptive bitrate controller
pub struct BitrateController {
    current_bitrate: u32,
    min_bitrate: u32,
    max_bitrate: u32,
    target_buffer_ms: u32,
    /// History of buffer levels
    buffer_history: Vec<f32>,
    /// History of bandwidth estimates
    bandwidth_history: Vec<f32>,
    window_size: usize,
}

impl BitrateController {
    pub fn new(min_bitrate: u32, max_bitrate: u32, initial_bitrate: u32) -> Self {
        Self {
            current_bitrate: initial_bitrate,
            min_bitrate,
            max_bitrate,
            target_buffer_ms: 3000,
            buffer_history: Vec::new(),
            bandwidth_history: Vec::new(),
            window_size: 10,
        }
    }

    /// Update with current conditions
    pub fn update(&mut self, buffer_level_ms: f32, bandwidth_kbps: f32) {
        // Track history
        if self.buffer_history.len() >= self.window_size {
            self.buffer_history.remove(0);
        }
        self.buffer_history.push(buffer_level_ms);

        if self.bandwidth_history.len() >= self.window_size {
            self.bandwidth_history.remove(0);
        }
        self.bandwidth_history.push(bandwidth_kbps);

        // Calculate target bitrate
        let avg_bandwidth = if self.bandwidth_history.is_empty() {
            bandwidth_kbps
        } else {
            self.bandwidth_history.iter().sum::<f32>() / self.bandwidth_history.len() as f32
        };

        let buffer_ratio = buffer_level_ms / self.target_buffer_ms as f32;

        let target_bitrate = if buffer_ratio < 0.5 {
            // Buffer critically low, reduce bitrate aggressively
            (avg_bandwidth * 0.5) as u32
        } else if buffer_ratio < 0.8 {
            // Buffer low, reduce bitrate
            (avg_bandwidth * 0.7) as u32
        } else if buffer_ratio > 1.5 {
            // Buffer healthy, can increase
            (avg_bandwidth * 0.9) as u32
        } else {
            // Buffer stable
            (avg_bandwidth * 0.8) as u32
        };

        // Smooth bitrate changes
        let new_bitrate = if target_bitrate > self.current_bitrate {
            // Increase slowly
            self.current_bitrate + (target_bitrate - self.current_bitrate) / 4
        } else {
            // Decrease quickly
            self.current_bitrate - (self.current_bitrate - target_bitrate) / 2
        };

        self.current_bitrate = new_bitrate.clamp(self.min_bitrate, self.max_bitrate);
    }

    /// Get current recommended bitrate
    pub fn bitrate(&self) -> u32 {
        self.current_bitrate
    }

    /// Check if we should switch quality
    pub fn should_switch(&self) -> Option<u32> {
        if self.bandwidth_history.len() < 3 {
            return None;
        }

        // Check for consistent bandwidth changes
        let recent: Vec<_> = self.bandwidth_history.iter().rev().take(3).collect();
        let trend = recent[0] - recent[2];

        if trend > self.current_bitrate as f32 * 0.2 {
            // Bandwidth increasing, consider upgrade
            Some(self.current_bitrate + self.current_bitrate / 4)
        } else if trend < -(self.current_bitrate as f32 * 0.2) {
            // Bandwidth decreasing, consider downgrade
            Some(self.current_bitrate - self.current_bitrate / 3)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_congestion_controller() {
        let mut cc = CongestionController::new();

        assert_eq!(cc.state(), CongestionState::SlowStart);
        assert!(cc.can_send());

        // Simulate some acks
        cc.on_send(1400);
        cc.on_ack(1400);

        assert!(cc.cwnd() >= CongestionController::INITIAL_CWND);
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(100_000.0, 10_000.0);

        assert!(limiter.can_send(1000));
        assert!(limiter.consume(5000));
        assert!(limiter.can_send(5000));
        assert!(!limiter.can_send(10000));
    }

    #[test]
    fn test_bitrate_controller() {
        let mut controller = BitrateController::new(64_000, 320_000, 128_000);

        controller.update(3000.0, 200.0);
        assert!(controller.bitrate() >= 64_000);
        assert!(controller.bitrate() <= 320_000);
    }
}
