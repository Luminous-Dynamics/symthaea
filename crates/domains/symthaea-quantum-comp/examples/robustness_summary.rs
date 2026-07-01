use symthaea_quantum_comp::{NoiseRobustnessSummary, NoiseSweepConfig, NoiseSweepRunner};

fn main() -> symthaea_quantum_comp::Result<()> {
    let sweep = NoiseSweepRunner::new(NoiseSweepConfig::default())?.run()?;
    let summary = NoiseRobustnessSummary::from_sweep(&sweep, 0.75);
    println!("{}", sweep.to_text_table());
    println!("{}", summary.to_text());
    Ok(())
}
