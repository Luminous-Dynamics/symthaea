use symthaea_quantum_comp::current_validation_snapshot;

fn main() {
    let snapshot = current_validation_snapshot();
    println!("{}", snapshot.to_markdown());
}
