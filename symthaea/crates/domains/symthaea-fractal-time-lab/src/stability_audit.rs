// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/// Captures streaming stability signals from the cognitive loop.
pub struct StabilityAuditHarvester {
    buffer: Vec<f64>,
    max_len: usize,
}

impl StabilityAuditHarvester {
    pub fn new(max_len: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_len),
            max_len,
        }
    }

    /// Record a stability signal from the sensor.
    pub fn record(&mut self, ema: f64) {
        if self.buffer.len() >= self.max_len {
            self.buffer.remove(0);
        }
        self.buffer.push(ema);
    }

    pub fn buffer(&self) -> &[f64] {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harvester_captures_sequence() {
        let mut harvester = StabilityAuditHarvester::new(10);
        for i in 0..15 {
            harvester.record(i as f64);
        }
        assert_eq!(harvester.buffer().len(), 10);
        assert_eq!(harvester.buffer()[0], 5.0);
    }
}
