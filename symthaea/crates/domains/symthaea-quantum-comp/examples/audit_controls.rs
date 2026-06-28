use symthaea_quantum_comp::{
    NegativeControlConfig, NegativeControlRunner, NoiseRobustnessSummary, NoiseSweepConfig,
    NoiseSweepRunner, audit_negative_control, audit_robustness, robustness_to_markdown,
};

fn main() -> symthaea_quantum_comp::Result<()> {
    let controls = NegativeControlRunner::new(NegativeControlConfig::default())?.run()?;
    let control_audit = audit_negative_control(&controls, 0.30);
    println!("{}", controls.to_text());
    println!("{}", control_audit.to_text());

    let sweep = NoiseSweepRunner::new(NoiseSweepConfig::default())?.run()?;
    let robustness = NoiseRobustnessSummary::from_sweep(&sweep, 0.75);
    let robustness_audit = audit_robustness(&robustness);
    println!("{}", robustness_to_markdown(&robustness));
    println!("{}", robustness_audit.to_text());
    Ok(())
}
