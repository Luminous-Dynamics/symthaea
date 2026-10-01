use serde_json::to_string_pretty;
use symthaea_core::hdc::resonator_separation_harness::{
    run_separation_sweep, SeparationSpec,
};

fn main() {
    let evidence =
        run_separation_sweep(SeparationSpec::default())
            .expect("default separation specification must be valid");
    println!(
        "{}",
        to_string_pretty(&evidence).expect("separation evidence serializes")
    );
}
