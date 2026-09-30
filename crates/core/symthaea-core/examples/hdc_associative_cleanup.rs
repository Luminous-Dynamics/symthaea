use symthaea_core::hdc::associative_cleanup_harness::{
    run_associative_cleanup, AssociativeCleanupSpec,
};

fn main() {
    let evidence = run_associative_cleanup(AssociativeCleanupSpec::default())
        .expect("associative cleanup harness should execute");
    println!(
        "{}",
        serde_json::to_string_pretty(&evidence).expect("evidence should serialize")
    );
}
