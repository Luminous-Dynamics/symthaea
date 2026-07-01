use symthaea_quantum_comp::{NegativeControlConfig, NegativeControlRunner};

fn main() {
    let config = NegativeControlConfig {
        dimension: 512,
        trials: 12,
        noise: 0.04,
        seed: 20260622,
    };
    let report = NegativeControlRunner::new(config)
        .expect("valid config")
        .run()
        .expect("probe runs");
    println!("{}", report.to_text());
}
