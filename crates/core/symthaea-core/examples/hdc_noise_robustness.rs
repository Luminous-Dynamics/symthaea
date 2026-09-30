use symthaea_core::hdc::noise_robustness_harness::{
    run_noise_robustness, NoiseRobustnessSpec,
};

fn main() {
    let evidence = run_noise_robustness(NoiseRobustnessSpec::default())
        .expect("noise robustness harness should execute");
    println!(
        "{}",
        serde_json::to_string_pretty(&evidence).expect("evidence should serialize")
    );
}
