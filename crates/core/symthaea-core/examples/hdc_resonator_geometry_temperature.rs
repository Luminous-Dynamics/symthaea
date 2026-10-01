use symthaea_core::hdc::resonator_geometry_temperature_harness::{
    run_interaction_sweep, InteractionSpec,
};

fn main() {
    let evidence =
        run_interaction_sweep(InteractionSpec::default()).expect("interaction sweep should run");
    println!(
        "{}",
        serde_json::to_string_pretty(&evidence).expect("interaction evidence serializes")
    );
}
