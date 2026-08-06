//! Calibrated continuous-value comparison: does storing a scalar directly as
//! a phase angle recover it more precisely than classical HDC's necessary
//! discretization (thermometer coding)? See
//! `src/continuous_value_comparison.rs` module docs for the design and
//! `docs/RESEARCH_NOTES.md` for the recorded result.

use symthaea_quantum_comp::{ContinuousValueSweepConfig, ContinuousValueSweepRunner};

fn main() -> symthaea_quantum_comp::Result<()> {
    let report = ContinuousValueSweepRunner::new(ContinuousValueSweepConfig::default())?.run()?;
    print!("{}", report.to_text());
    Ok(())
}
