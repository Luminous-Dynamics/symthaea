use serde_json::to_string_pretty;
use symthaea_core::hdc::two_factor_resonator_harness::{run_two_factor, TwoFactorSpec};

fn main() {
    let evidence =
        run_two_factor(TwoFactorSpec::default()).expect("default two-factor specification must be valid");
    println!(
        "{}",
        to_string_pretty(&evidence).expect("two-factor evidence serializes")
    );
}
