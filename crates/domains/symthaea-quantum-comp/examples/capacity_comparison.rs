//! Calibrated capacity comparison: does classical or phase-HDC bundling
//! support more superposed items before forced-choice recall degrades?
//! See `src/capacity_comparison.rs` module docs for the design and
//! `docs/RESEARCH_NOTES.md` for the recorded result.

use symthaea_quantum_comp::{CapacitySweepConfig, CapacitySweepRunner};

fn main() -> symthaea_quantum_comp::Result<()> {
    let report = CapacitySweepRunner::new(CapacitySweepConfig::default())?.run()?;
    print!("{}", report.to_text());
    Ok(())
}
