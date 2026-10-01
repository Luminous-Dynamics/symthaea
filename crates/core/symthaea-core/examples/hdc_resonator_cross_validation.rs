use serde_json::to_string_pretty;
use symthaea_core::hdc::resonator_cross_validation_harness::{
    run_cross_validation, CrossValidationSpec,
};

fn main() {
    let evidence = run_cross_validation(CrossValidationSpec::default())
        .expect("default resonator cross-validation specification must be valid");
    println!(
        "{}",
        to_string_pretty(&evidence).expect("cross-validation evidence serializes")
    );
}
