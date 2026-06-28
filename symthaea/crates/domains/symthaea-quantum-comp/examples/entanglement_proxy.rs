use symthaea_quantum_comp::{EntanglementProxyConfig, EntanglementProxyRunner};

fn main() {
    let config = EntanglementProxyConfig {
        dimension: 512,
        trials: 12,
        decoherence: 0.08,
        seed: 20260622,
    };
    let report = EntanglementProxyRunner::new(config)
        .expect("valid config")
        .run()
        .expect("probe runs");
    println!("{}", report.to_text());
}
