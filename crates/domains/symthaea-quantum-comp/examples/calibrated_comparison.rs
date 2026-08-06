//! Calibrated replacement for `comparative_report`'s cross-representation
//! comparison. See `src/calibrated_comparison.rs` module docs and
//! `docs/RESEARCH_NOTES.md` ("First independent run and a real finding,
//! 2026-07-24") for why the uncalibrated version's headline effect size
//! (classical_minus_phase_noisy_dz = -33.36) was a noise-model artifact, not
//! a substrate finding.

use symthaea_quantum_comp::{CalibratedSweepConfig, CalibratedSweepRunner};

fn main() -> symthaea_quantum_comp::Result<()> {
    let report = CalibratedSweepRunner::new(CalibratedSweepConfig::default())?.run()?;
    for point in &report.points {
        println!("{}", point.to_text());
        println!();
    }
    println!("--- CSV ---");
    print!("{}", report.to_csv());
    Ok(())
}
