use symthaea_quantum_comp::{BindingProbeConfig, NoiseSweepConfig, NoiseSweepRunner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = BindingProbeConfig {
        dimension: 512,
        trials: 12,
        noise: 0.0,
        seed: 0x5159_4D54_4841_4541,
        topology_threshold: 0.55,
    };
    let sweep = NoiseSweepRunner::new(NoiseSweepConfig {
        base,
        steps: 6,
        max_noise: 0.25,
    })?
    .run()?;
    println!("{}", sweep.to_text_table());
    Ok(())
}
