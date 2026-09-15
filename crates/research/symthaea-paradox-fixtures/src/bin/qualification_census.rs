// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Emit the complete PARADOX-002A confirmatory fixture/oracle census as a
//! deterministic binary stream for qualification-time commitment.
//!
//! This program has no behavior authority. It only serializes already-qualified
//! fixture and oracle canonical bytes in the frozen confirmatory traversal order.

use std::io::{self, BufWriter, Write};

use symthaea_paradox_fixtures::{
    ALL_CONDITIONS, CONFIRMATORY_SEEDS, CONFIRMATORY_TRIALS_PER_CONDITION, FixtureGenerator,
    qualify_fixture,
};

const DOMAIN: &[u8] = b"SYMT-PARADOX-002A-CONFIRMATORY-CENSUS-V1\0";

fn write_record<W: Write>(out: &mut W, bytes: &[u8]) -> io::Result<()> {
    out.write_all(&(bytes.len() as u64).to_le_bytes())?;
    out.write_all(bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    out.write_all(DOMAIN)?;
    out.write_all(&(CONFIRMATORY_SEEDS.len() as u64).to_le_bytes())?;
    out.write_all(&(ALL_CONDITIONS.len() as u64).to_le_bytes())?;
    out.write_all(&(CONFIRMATORY_TRIALS_PER_CONDITION as u64).to_le_bytes())?;

    for seed in CONFIRMATORY_SEEDS {
        for condition in ALL_CONDITIONS {
            for trial_index in 0..CONFIRMATORY_TRIALS_PER_CONDITION {
                let fixture = FixtureGenerator::generate(condition, seed, trial_index)?;
                let report = qualify_fixture(&fixture)?;
                let fixture_bytes = fixture.canonical_bytes();
                let report_bytes = report.canonical_bytes();

                write_record(&mut out, &fixture_bytes)?;
                write_record(&mut out, &report_bytes)?;
            }
        }
    }

    out.flush()?;
    Ok(())
}
