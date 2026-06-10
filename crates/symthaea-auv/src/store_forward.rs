// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Store-and-forward telemetry for underwater communication blackouts.
//!
//! When submerged, the AUV cannot transmit to the Mycelix DHT. Readings
//! are buffered in a circular buffer and batch-submitted on surfacing.

use crate::types::ChemicalReadings;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A buffered reading waiting for transmission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferedReading {
    /// Timestamp (seconds since mission start).
    pub timestamp: f64,
    /// GPS position at time of reading (may be last known if submerged).
    pub position: [f64; 3],
    /// Depth at time of reading.
    pub depth: f64,
    /// Chemical sensor readings.
    pub chemical: ChemicalReadings,
    /// Whether WHO standards were met at this reading.
    pub meets_who: bool,
    /// Potability score at this reading.
    pub potability_score: f64,
}

/// Store-and-forward buffer for underwater telemetry.
pub struct StoreForwardBuffer {
    /// Ring buffer of readings.
    buffer: VecDeque<BufferedReading>,
    /// Maximum capacity.
    capacity: usize,
    /// Total readings collected (including dropped due to overflow).
    total_collected: u64,
    /// Whether currently in blackout (submerged, no comms).
    pub in_blackout: bool,
    /// Depth threshold for blackout (meters). Below this = no comms.
    pub blackout_depth: f64,
}

impl StoreForwardBuffer {
    /// Create with given capacity (default 10,000 readings).
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity.min(10_000)),
            capacity,
            total_collected: 0,
            in_blackout: false,
            blackout_depth: 2.0, // Can't transmit below 2m
        }
    }

    /// Default buffer (10,000 readings capacity).
    pub fn default_capacity() -> Self {
        Self::new(10_000)
    }

    /// Record a reading. If at capacity, oldest reading is dropped.
    pub fn record(&mut self, reading: BufferedReading) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(reading);
        self.total_collected += 1;
    }

    /// Update blackout status based on current depth.
    pub fn update_blackout(&mut self, depth: f64) {
        self.in_blackout = depth > self.blackout_depth;
    }

    /// Drain all buffered readings for transmission (on surfacing).
    /// Returns readings in chronological order.
    pub fn drain_for_transmission(&mut self) -> Vec<BufferedReading> {
        self.buffer.drain(..).collect()
    }

    /// Number of readings currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Total readings collected over lifetime.
    pub fn total_collected(&self) -> u64 {
        self.total_collected
    }

    /// Whether the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Clear all buffered readings.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_reading(t: f64) -> BufferedReading {
        let chem = ChemicalReadings::clean_freshwater();
        BufferedReading {
            timestamp: t,
            position: [0.0, 0.0, 10.0],
            depth: 10.0,
            chemical: chem,
            meets_who: chem.meets_who_standards(),
            potability_score: chem.potability_score(),
        }
    }

    #[test]
    fn test_record_and_drain() {
        let mut buf = StoreForwardBuffer::new(100);
        buf.record(sample_reading(1.0));
        buf.record(sample_reading(2.0));
        assert_eq!(buf.buffered_count(), 2);
        let readings = buf.drain_for_transmission();
        assert_eq!(readings.len(), 2);
        assert_eq!(buf.buffered_count(), 0);
        assert_eq!(buf.total_collected(), 2);
    }

    #[test]
    fn test_capacity_overflow() {
        let mut buf = StoreForwardBuffer::new(3);
        for i in 0..5 {
            buf.record(sample_reading(i as f64));
        }
        assert_eq!(buf.buffered_count(), 3);
        assert_eq!(buf.total_collected(), 5);
        // Oldest should be dropped
        let readings = buf.drain_for_transmission();
        assert!(
            (readings[0].timestamp - 2.0).abs() < 1e-10,
            "Oldest should be t=2.0"
        );
    }

    #[test]
    fn test_blackout_detection() {
        let mut buf = StoreForwardBuffer::new(100);
        buf.update_blackout(1.0);
        assert!(!buf.in_blackout, "Surface should not be blackout");
        buf.update_blackout(5.0);
        assert!(buf.in_blackout, "Submerged should be blackout");
    }

    #[test]
    fn test_is_full() {
        let mut buf = StoreForwardBuffer::new(2);
        assert!(!buf.is_full());
        buf.record(sample_reading(1.0));
        buf.record(sample_reading(2.0));
        assert!(buf.is_full());
    }
}
