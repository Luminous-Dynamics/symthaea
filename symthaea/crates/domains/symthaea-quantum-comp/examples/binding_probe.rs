use symthaea_quantum_comp::{BindingProbeConfig, BindingProbeRunner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = BindingProbeConfig {
        dimension: 1024,
        trials: 16,
        noise: 0.05,
        seed: 0x5159_4D54_4841_4541,
        topology_threshold: 0.55,
    };
    let runner = BindingProbeRunner::new(config)?;
    let report = runner.run()?;
    println!("{}", report.to_text());
    println!("{}", report.to_json_like());
    Ok(())
}
