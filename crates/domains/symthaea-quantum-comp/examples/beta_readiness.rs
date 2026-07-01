use symthaea_quantum_comp::current_beta_readiness;

fn main() {
    let report = current_beta_readiness();
    println!("{}", report.to_markdown());
}
