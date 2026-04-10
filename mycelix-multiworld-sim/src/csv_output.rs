// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! CSV time-series output for scientific analysis.
//!
//! Writes one row per world per tick, producing machine-readable data amenable
//! to R, Python, Julia, or any analysis tool. The output format is designed for
//! reproducibility: each row is self-contained with tick, year, world, and all
//! key metrics.
//!
//! # Usage
//!
//! ```rust,no_run
//! let mut sim = MultiWorldSimulator::new(config);
//! sim.enable_csv_output("output/run_seed42.csv".into());
//! sim.run();
//! ```
//!
//! # Output Format
//!
//! ```csv
//! tick,year,world,location,population,mean_phi,collective_phi,gini,allostatic_load,self_sufficiency,energy,food,water,infrastructure,governance,disasters_active,trust
//! ```

use crate::world::World;
use std::io::Write;

/// CSV column header line.
pub const CSV_HEADER: &str = "tick,year,world,location,population,mean_phi,collective_phi,gini,allostatic_load,self_sufficiency,energy,food,water,infrastructure,governance,disasters_active,trust";

/// Write the CSV header to a writer.
pub fn write_header(writer: &mut impl Write) -> std::io::Result<()> {
    writeln!(writer, "{}", CSV_HEADER)
}

/// Write one row per world for the current tick.
pub fn write_tick(
    writer: &mut impl Write,
    tick: u32,
    worlds: &[World],
    disasters_active: u32,
) -> std::io::Result<()> {
    let year = tick as f64 / 12.0;
    for world in worlds {
        if world.population() == 0 && tick > world.founded_tick + 12 {
            continue; // Skip extinct worlds after first year
        }
        let energy = world.resources.stock_level("energy").unwrap_or(0.0);
        let food = world.resources.stock_level("food").unwrap_or(0.0);
        let water = world.resources.stock_level("water").unwrap_or(0.0);

        writeln!(
            writer,
            "{},{:.2},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{:.1},{:.1},{:.4},{},{},{}",
            tick,
            year,
            world.name,
            world.location,
            world.population(),
            world.mean_phi(),
            world.governance.stability_score * world.mean_phi(), // collective phi proxy
            world.economy.gini_coefficient,
            world.mean_allostatic_load(),
            world.resources.self_sufficiency(),
            energy,
            food,
            water,
            world.infrastructure_level,
            world.governance.authority_level,
            disasters_active,
            world.trust_level,
        )?;
    }
    Ok(())
}

/// Buffered CSV writer that accumulates rows and flushes to disk.
pub struct CsvRecorder {
    buffer: Vec<u8>,
    path: std::path::PathBuf,
    header_written: bool,
}

impl CsvRecorder {
    /// Create a new CSV recorder that will write to the given path.
    pub fn new(path: std::path::PathBuf) -> Self {
        Self {
            buffer: Vec::with_capacity(64 * 1024), // 64KB buffer
            path,
            header_written: false,
        }
    }

    /// Record one tick's data for all worlds.
    pub fn record_tick(
        &mut self,
        tick: u32,
        worlds: &[World],
        disasters_active: u32,
    ) -> std::io::Result<()> {
        if !self.header_written {
            write_header(&mut self.buffer)?;
            self.header_written = true;
        }
        write_tick(&mut self.buffer, tick, worlds, disasters_active)?;

        // Flush every 1000 ticks (~83 years) to balance performance and safety
        if tick % 1000 == 0 {
            self.flush()?;
        }
        Ok(())
    }

    /// Flush buffer to disk.
    pub fn flush(&mut self) -> std::io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(&self.buffer)?;
        self.buffer.clear();
        Ok(())
    }
}

impl Drop for CsvRecorder {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_contains_all_columns() {
        let cols: Vec<&str> = CSV_HEADER.split(',').collect();
        assert!(cols.contains(&"tick"));
        assert!(cols.contains(&"year"));
        assert!(cols.contains(&"world"));
        assert!(cols.contains(&"population"));
        assert!(cols.contains(&"mean_phi"));
        assert!(cols.contains(&"gini"));
        assert!(cols.contains(&"self_sufficiency"));
        assert_eq!(cols.len(), 17, "Expected 17 columns, got {}", cols.len());
    }

    #[test]
    fn write_header_produces_valid_csv() {
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.ends_with('\n'));
        assert!(!output.contains('\r')); // Unix line endings
    }
}
