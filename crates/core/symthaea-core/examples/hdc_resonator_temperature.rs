use symthaea_core::hdc::resonator_temperature_harness::{run_temperature_sweep, TemperatureSpec};

fn main() {
    let evidence = run_temperature_sweep(TemperatureSpec::default())
        .expect("temperature sweep specification should be valid");
    println!(
        "{}",
        serde_json::to_string_pretty(&evidence).expect("temperature evidence serializes")
    );
}
