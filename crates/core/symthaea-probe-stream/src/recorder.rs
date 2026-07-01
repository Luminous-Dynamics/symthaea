// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Trajectory recorder
//!
//! Recorder collecting projected hypervectors for offline study or distillation.

use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use symthaea_hdc_ltc::ContinuousHV;

pub struct TrajectoryRecorder {
    hdc_dim: usize,
    capacity: usize,
    history: Vec<(f64, ContinuousHV)>,
}

impl TrajectoryRecorder {
    pub fn new(hdc_dim: usize, capacity: usize) -> Self {
        Self {
            hdc_dim,
            capacity,
            history: Vec::with_capacity(capacity),
        }
    }

    pub fn record(&mut self, t: f64, hv: &ContinuousHV) {
        if self.history.len() >= self.capacity {
            self.history.remove(0);
        }
        self.history.push((t, hv.clone()));
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(path)?;
        // Write header
        file.write_all(b"TRAJ0001")?;
        file.write_all(&(self.hdc_dim as u64).to_le_bytes())?;
        file.write_all(&(self.history.len() as u64).to_le_bytes())?;

        // Write points
        for (t, hv) in &self.history {
            file.write_all(&t.to_le_bytes())?;
            for &val in &hv.values {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }
}
