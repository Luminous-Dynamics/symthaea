use symthaea_quantum_comp::{ComparativeBindingConfig, ComparativeBindingRunner};

fn main() -> symthaea_quantum_comp::Result<()> {
    let report = ComparativeBindingRunner::new(ComparativeBindingConfig::default())?.run()?;
    println!("{}", report.to_text());
    Ok(())
}
