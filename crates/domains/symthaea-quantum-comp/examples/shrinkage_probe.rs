//! Does a partial (shrunk) bias correction beat full debiasing in the
//! high-noise regime where phase-HDC wins the continuous-value comparison?
//! See `src/continuous_value_comparison.rs`'s `ShrinkageProbeRunner` doc
//! comment for the design and `docs/RESEARCH_NOTES.md` for the recorded
//! result.

use symthaea_quantum_comp::{ShrinkageProbeConfig, ShrinkageProbeRunner};

fn main() -> symthaea_quantum_comp::Result<()> {
    for target_ber in [0.20_f32, 0.30, 0.40, 0.45, 0.48] {
        let config = ShrinkageProbeConfig {
            target_ber,
            trials: 200,
            ..ShrinkageProbeConfig::default()
        };
        let report = ShrinkageProbeRunner::new(config)?.run()?;
        print!("{}", report.to_csv());
        if let Some(best) = report.best_lambda() {
            println!(
                "best_lambda={} best_mae={:.6}\n",
                best.lambda, best.classical_error.mean
            );
        }
    }
    Ok(())
}
