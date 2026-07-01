use symthaea_quantum_comp::{ExperimentMatrixConfig, ExperimentMatrixRunner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExperimentMatrixConfig {
        dimensions: vec![128, 256, 512],
        noise_levels: vec![0.0, 0.05, 0.10, 0.20],
        trials: 8,
        replicates: 4,
        seed: 0xA16A_0006,
        seed_stride: 0x9E37_79B9_7F4A_7C15,
        topology_threshold: 0.55,
    };
    let report = ExperimentMatrixRunner::new(config)?.run()?;
    println!("{}", report.to_markdown());
    Ok(())
}
