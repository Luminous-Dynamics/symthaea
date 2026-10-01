use symthaea_core::hdc::resonator_cleanup_rule_harness::run;

fn main() {
    let evidence = run();
    println!("{}", serde_json::to_string_pretty(&evidence).expect("evidence serializes"));
}
