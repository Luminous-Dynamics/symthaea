use symthaea_core::hdc::dimension_task_harness::{run_dimension_task, DimensionTaskSpec};

fn main() {
    let evidence = run_dimension_task(DimensionTaskSpec::default())
        .expect("dimension task harness should execute");
    println!(
        "{}",
        serde_json::to_string_pretty(&evidence).expect("evidence should serialize")
    );
}
