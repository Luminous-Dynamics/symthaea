use symthaea_quantum_comp::PairedDifferenceSummary;

fn main() {
    let classical = [0.91, 0.88, 0.84, 0.79, 0.70];
    let phase = [0.89, 0.87, 0.85, 0.76, 0.69];
    let summary = PairedDifferenceSummary::from_pairs(&classical, &phase, 1e-6)
        .expect("equal nonempty samples");
    println!("{}", summary.to_text("classical", "phase"));
}
