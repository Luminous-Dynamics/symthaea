use symthaea_core::hdc::dimension_sweep::{run_dimension_sweep, DimensionSweepSpec};

fn main() {
    let spec = DimensionSweepSpec::default_observatory();
    let evidence = run_dimension_sweep(&spec).expect("default dimension sweep must validate");
    println!(
        "{}",
        serde_json::to_string_pretty(&evidence).expect("dimension sweep evidence is JSON")
    );
}
