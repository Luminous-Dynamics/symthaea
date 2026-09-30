use symthaea_core::hdc::sequence_order_harness::{
    run_sequence_order, SequenceOrderSpec,
};

fn main() {
    let evidence =
        run_sequence_order(SequenceOrderSpec::default()).expect("sequence-order harness should execute");
    println!(
        "{}",
        serde_json::to_string_pretty(&evidence).expect("evidence should serialize")
    );
}
